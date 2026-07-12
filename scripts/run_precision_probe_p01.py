#!/usr/bin/env python3
"""Run the frozen precision-probe-p01 matched NF4/bf16 screen.

The runner deliberately separates outcome persistence from outcome access.
Both NF4 e01 repetitions are first serialized, losslessly packaged, and
verified.  The extension decision then receives only two small timing/status
receipts.  Score extraction and render-ledger construction occur only after
that decision has been durably frozen.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
PACKAGER_DIR = (
    ROOT / "scripts/historical/coherent_canary_v12_e01/treatment_packaging"
)
sys.path.insert(0, str(PACKAGER_DIR))

import artifact_packager  # noqa: E402
from coherent_canary_case import (  # noqa: E402
    PHASE_A_SCHEMA,
    TREATMENT_SCHEMA,
    run_phase_a_case,
    run_treatment_case,
)
from coherent_canary_runtime import (  # noqa: E402
    execute_fresh_plan,
    execute_replay_plan,
    snapshot_hashes,
    tensor_sha256,
)
from coherent_canary_technical import (  # noqa: E402
    run_fresh_self_replacement,
    run_generated_forced_identity,
)
from coherent_canary_tokens import (  # noqa: E402
    build_fresh_destination_plan,
    build_role_native_plan,
)


PROTOCOL_ID = "precision-probe-p01"
RUN_SCHEMA = "precision_probe_p01_run_manifest_v1"
TECHNICAL_SCHEMA = "precision_probe_p01_technical_raw_v1"
OUTCOME_SCHEMA = "precision_probe_p01_outcome_raw_v1"
COMPACT_SCHEMA = "precision_probe_p01_compact_scores_generations_v1"
DECISION_SCHEMA = "precision_probe_p01_outcome_blind_extension_decision_v1"
COMPLETION_SCHEMA = "precision_probe_p01_completion_v1"

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_SLUG = "Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
REGIMES = ("nf4", "bf16")

DEFAULT_PREREG = ROOT / "PRECISION-PROBE-P01-PREREGISTRATION.md"
DEFAULT_IDENTITY = ROOT / "data/coherent_canary_v12/generated_forced_identity_fixture.json"
DEFAULT_TECHNICAL = ROOT / "data/coherent_canary_v12/technical_control_fixture.json"
DEFAULT_CASES = {
    "e01": ROOT / "data/coherent_canary_v12/revision2/session_d/e01.json",
    "e02": ROOT / "data/coherent_canary_v12/revision2/session_d/e02.json",
    "e03": ROOT / "data/coherent_canary_v12/revision2/session_e/e03.json",
}
DEFAULT_OUTPUT_DIR = ROOT / "results/precision_probe_p01"

FORMAL_FLAGS = {
    "formal_v12_decision_eligible": False,
    "v12_reentry_authorized": False,
    "component_reuse_does_not_inherit_v12_eligibility": True,
    "semantic_evidence_eligible": False,
}


class PrecisionProbeRunnerError(RuntimeError):
    """The frozen p01 contract, technical gate, or persistence step differed."""


@dataclass(frozen=True, slots=True)
class ExtensionTiming:
    """The entire information surface exposed to the extension decision."""

    completion_status: str
    outcome_wall_time_seconds: float


@dataclass(frozen=True, slots=True)
class OutcomePaths:
    raw_scratch: Path
    raw_package: Path
    phase_a_checkpoint_scratch: Path
    phase_a_checkpoint_package: Path
    compact: Path
    renders: Path


@dataclass(frozen=True, slots=True)
class OutcomeReceipt:
    regime: str
    case_id: str
    repeat_index: int
    completion_status: str
    outcome_wall_time_seconds: float
    raw_package: Mapping[str, Any] | None
    recovery_raw_path: str | None
    compact_path: str | None = None
    renders_path: str | None = None

    def extension_timing(self) -> ExtensionTiming:
        # Projection is intentional: no path, payload, score, or generation can
        # cross the decision boundary.
        return ExtensionTiming(
            completion_status=self.completion_status,
            outcome_wall_time_seconds=self.outcome_wall_time_seconds,
        )


@dataclass(frozen=True, slots=True)
class ProviderClock:
    elapsed_seconds_at_start: float
    monotonic_at_start: float

    def elapsed(self) -> float:
        return self.elapsed_seconds_at_start + (
            time.monotonic() - self.monotonic_at_start
        )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionProbeRunnerError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            json_safe(value), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return str(value)


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def display_path(path: Path, repo: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def binding(path: Path, *, repo: Path) -> dict[str, Any]:
    require(path.is_file(), f"bound file is absent: {path}")
    digest, size = sha256_file(path)
    return {
        "path": display_path(path, repo),
        "sha256": digest,
        "size_bytes": size,
    }


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except Exception as exc:
        raise PrecisionProbeRunnerError(f"cannot parse {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def git_value(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _atomic_bytes(path: Path, data: bytes, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and path.exists():
        raise PrecisionProbeRunnerError(f"refusing to overwrite artifact: {path}")
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if not replace and path.exists():
            raise PrecisionProbeRunnerError(
                f"artifact appeared before atomic publish: {path}"
            )
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems do not support directory fsync.  The file was
            # still atomically renamed and file-fsynced.
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_create_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, canonical_json_bytes(value), replace=False)


def atomic_replace_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, canonical_json_bytes(value), replace=True)


def atomic_create_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("utf-8"), replace=False)


def prepare_precision_subject(
    regime: str, repo: Path, allow_download: bool
) -> dict[str, Any]:
    """Late import keeps unit tests independent while preserving one loader API."""
    module = importlib.import_module("precision_probe_p01_loader")
    function = getattr(module, "prepare_precision_subject", None)
    require(callable(function), "precision p01 loader entry point is absent")
    result = function(regime, repo, allow_download)
    require(isinstance(result, dict), "precision p01 loader returned no mapping")
    return result


def execution_record(result: Any) -> dict[str, Any]:
    return {
        "calls": list(result.calls),
        "token_ids": list(result.executed_token_ids),
        "logical_positions": list(result.logical_positions),
        "physical_positions": list(result.physical_positions),
        "physical_end": int(result.physical_end),
        "logical_end": int(result.logical_end),
        "snapshot_hashes": snapshot_hashes(result.snapshot),
        "last_logits_sha256": tensor_sha256(result.last_logits),
    }


def deterministic_repeats(
    run_once: Callable[[], Any], label: str
) -> dict[str, Any]:
    records = [execution_record(run_once()), execution_record(run_once())]
    require(records[0] == records[1], f"{label} deterministic repeat differs")
    return {"status": "PASS", "repeat_count": 2, "records": records}


def _kv_dtype_evidence(value: Any) -> list[str]:
    observed: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"k_dtype", "v_dtype"}:
                observed.append(str(item))
            else:
                observed.extend(_kv_dtype_evidence(item))
    elif isinstance(value, list):
        for item in value:
            observed.extend(_kv_dtype_evidence(item))
    return observed


def _assert_technical_gate(document: Mapping[str, Any]) -> None:
    identity = document.get("generated_forced_identity", {})
    deterministic = document.get("deterministic_repeats", {})
    replacement = document.get("fresh_self_replacement", {})
    require(identity.get("status") == "PASS", "generated/forced identity failed")
    require(
        set(deterministic) == {"correct_history_N", "fresh_destination"}
        and all(row.get("status") == "PASS" for row in deterministic.values()),
        "deterministic replay/fresh checks failed",
    )
    regions = replacement.get("regions", [])
    require(
        replacement.get("status") == "PASS" and len(regions) == 9
        and all(row.get("status") == "PASS" for row in regions),
        "fresh self-replacement did not return nine passing cells",
    )
    dtypes = _kv_dtype_evidence(
        {
            "identity": identity,
            "deterministic": deterministic,
            "replacement": replacement,
        }
    )
    require(bool(dtypes), "technical gate exposed no K/V dtype evidence")
    require(
        set(dtypes) == {"torch.bfloat16"},
        f"technical gate observed non-bf16 K/V dtype: {sorted(set(dtypes))}",
    )


def _stage(document: dict[str, Any], name: str, function: Callable[[], Any]) -> Any:
    started = time.monotonic()
    started_at = utc_now()
    status = "ERROR"
    try:
        value = function()
        status = "COMPLETE"
        return value
    finally:
        document.setdefault("stage_timings", {})[name] = {
            "status": status,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "wall_time_seconds": time.monotonic() - started,
        }


def decide_extension(
    *,
    e01_timings: Sequence[ExtensionTiming],
    provider_elapsed_seconds: float,
    hourly_cost_usd: float,
    provider_wall_cap_seconds: float,
) -> dict[str, Any]:
    """Apply the frozen all-or-none formula without any outcome access.

    ``ExtensionTiming`` has slots and contains only completion status and wall
    time.  No artifact path or model-facing payload is accepted by this
    function.
    """
    require(
        len(e01_timings) == 2
        and all(type(row) is ExtensionTiming for row in e01_timings),
        "extension decision requires exactly two scalar timing receipts",
    )
    require(
        math.isfinite(provider_elapsed_seconds) and provider_elapsed_seconds >= 0,
        "provider elapsed time is invalid",
    )
    require(
        math.isfinite(hourly_cost_usd) and hourly_cost_usd > 0,
        "hourly provider rate must be positive",
    )
    require(
        math.isfinite(provider_wall_cap_seconds)
        and 0 < provider_wall_cap_seconds <= 7200,
        "provider wall cap must be in (0, 7200] seconds",
    )
    statuses = [row.completion_status for row in e01_timings]
    times = [float(row.outcome_wall_time_seconds) for row in e01_timings]
    complete = all(status == "COMPLETE" for status in statuses)
    valid_times = complete and all(math.isfinite(value) and value >= 0 for value in times)
    hard_deadline = min(
        7200.0,
        float(provider_wall_cap_seconds),
        4.0 * 3600.0 / float(hourly_cost_usd),
    )
    slower = max(times) if valid_times else None
    forecast = (
        provider_elapsed_seconds + 1.20 * (6.0 * slower + 900.0)
        if slower is not None else None
    )
    allowed = bool(
        complete and valid_times and forecast is not None
        and forecast <= hard_deadline
    )
    return {
        "schema": DECISION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        **FORMAL_FLAGS,
        "decision": "EXTEND_E02_E03" if allowed else "BASE_E01_ONLY",
        "extension_allowed": allowed,
        "selected_case_repeats": (
            {"e01": 2, "e02": 1, "e03": 1}
            if allowed else {"e01": 2}
        ),
        "inputs": {
            "nf4_e01_completion_statuses": statuses,
            "nf4_e01_outcome_wall_time_seconds": times,
            "slower_complete_nf4_e01_seconds": slower,
            "provider_elapsed_seconds": provider_elapsed_seconds,
            "hourly_cost_usd": hourly_cost_usd,
            "provider_wall_cap_seconds": provider_wall_cap_seconds,
            "four_dollar_rate_ceiling_seconds": 4.0 * 3600.0 / hourly_cost_usd,
            "effective_hard_deadline_seconds": hard_deadline,
            "forecast_seconds": forecast,
        },
        "formula": "elapsed + 1.20 * (6*t + 900) <= min(7200, cap, 4*3600/rate)",
        "outcome_information_surface": [
            "completion_status", "outcome_wall_time_seconds"
        ],
        "score_or_generation_accessible_to_decision": False,
        "created_at_utc": utc_now(),
    }


def _package_binding(
    package: Path, manifest: Mapping[str, Any], verification: Mapping[str, Any],
    *, repo: Path,
) -> dict[str, Any]:
    manifest_path = package / artifact_packager.MANIFEST_NAME
    digest, size = sha256_file(manifest_path)
    return {
        "path": display_path(package, repo),
        "manifest_path": display_path(manifest_path, repo),
        "manifest_sha256": digest,
        "manifest_size_bytes": size,
        "chunk_count": int(manifest["chunk_count"]),
        "original_sha256": str(verification["original_sha256"]),
        "original_size_bytes": int(verification["original_size_bytes"]),
        "compressed_sha256": str(verification["compressed_sha256"]),
        "compressed_size_bytes": int(verification["compressed_size_bytes"]),
        "verification_status": str(verification["status"]),
    }


def persist_lossless_raw(
    document: Mapping[str, Any], *, scratch_raw: Path, package: Path, repo: Path
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Create, package, and byte-verify raw JSON without parsing outcomes.

    The packager's JSON validation is intentionally disabled at this boundary:
    the bytes came directly from ``canonical_json_bytes``, while parsing any
    NF4 e01 score/generation before the extension decision is prohibited.
    Verification reconstructs and hashes the byte stream but does not parse the
    source document.
    """
    atomic_create_json(scratch_raw, document)
    try:
        manifest = artifact_packager.pack(
            scratch_raw, package, validate_json=False
        )
        verification = artifact_packager.verify_and_reconstruct(package)
        raw_sha, raw_size = sha256_file(scratch_raw)
        require(
            verification.get("status") == "VERIFIED"
            and verification.get("original_sha256") == raw_sha
            and verification.get("original_size_bytes") == raw_size,
            "lossless package verification differs from scratch raw bytes",
        )
        packaged = _package_binding(package, manifest, verification, repo=repo)
        scratch_raw.unlink()
        return packaged, None, None
    except Exception as exc:
        # The recovery raw is deliberately retained.  The caller records it in
        # the manifest and pod pull receipt instead of losing expensive work.
        return (
            None,
            display_path(scratch_raw, repo),
            f"{type(exc).__name__}: {exc}",
        )


def _load_packaged_document(
    package: Path, *, scratch_dir: Path
) -> dict[str, Any]:
    reconstructed = scratch_dir / (
        f"reconstructed-{package.stem}-{uuid.uuid4().hex}.json"
    )
    verification = artifact_packager.verify_and_reconstruct(
        package, reconstructed
    )
    require(verification.get("status") == "VERIFIED", "raw package did not verify")
    try:
        return load_object(reconstructed)
    finally:
        if reconstructed.exists():
            reconstructed.unlink()


def _probe_compact(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "probe", "suffix_ids", "correct_text", "counterfactual_text",
        "correct", "counterfactual", "margin", "margin_float32_bits",
        "margin_arithmetic", "generation",
    )
    return {key: json_safe(record[key]) for key in keys if key in record}


def compact_outcome(document: Mapping[str, Any]) -> dict[str, Any]:
    phase = document.get("phase_a")
    treatment = document.get("treatment")
    compact: dict[str, Any] = {
        "schema": COMPACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        **FORMAL_FLAGS,
        "status": document.get("status"),
        "regime": document.get("regime"),
        "case_id": document.get("case_id"),
        "repeat_index": document.get("repeat_index"),
        "outcome_wall_time_seconds": document.get("outcome_wall_time_seconds"),
        "raw_package": document.get("raw_package_binding"),
        "phase_a_scores": {},
        "treatment_fresh_scores": {},
        "treatment_arms": [],
    }
    if isinstance(phase, Mapping):
        scores = phase.get("scores", {})
        if isinstance(scores, Mapping):
            compact["phase_a_scores"] = {
                str(name): _probe_compact(row)
                for name, row in scores.items() if isinstance(row, Mapping)
            }
    if isinstance(treatment, Mapping):
        fresh = treatment.get("fresh_scores", {})
        if isinstance(fresh, Mapping):
            compact["treatment_fresh_scores"] = {
                str(name): _probe_compact(row)
                for name, row in fresh.items() if isinstance(row, Mapping)
            }
        arms = treatment.get("arms", [])
        if isinstance(arms, list):
            for arm in arms:
                if not isinstance(arm, Mapping):
                    continue
                row = {
                    key: json_safe(arm[key]) for key in (
                        "arm_kind", "schedule", "region", "cell",
                        "key_source", "value_source", "control_status",
                        "shared_fresh_baseline", "diagnostics",
                    ) if key in arm
                }
                scores = arm.get("scores", {})
                row["scores"] = {
                    str(name): _probe_compact(score)
                    for name, score in scores.items()
                    if isinstance(score, Mapping)
                } if isinstance(scores, Mapping) else {}
                compact["treatment_arms"].append(row)
    return compact


def _literal_block(value: Any) -> str:
    return "~~~~text\n" + str(value) + "\n~~~~\n"


def render_outcome_ledger(document: Mapping[str, Any]) -> str:
    lines = [
        f"# P01 render ledger: {document.get('regime')} / "
        f"{document.get('case_id')} / repeat {document.get('repeat_index')}",
        "",
        f"- Protocol: `{PROTOCOL_ID}`",
        f"- Status: `{document.get('status')}`",
        "- Formal v12 decision eligible: `false`",
        f"- Raw original SHA-256: `"
        f"{document.get('raw_package_binding', {}).get('original_sha256')}`",
        "",
        "## Literal visible histories",
        "",
    ]
    phase = document.get("phase_a", {})
    visible = phase.get("visible_messages", {}) if isinstance(phase, Mapping) else {}
    if isinstance(visible, Mapping):
        for name, messages in visible.items():
            lines.extend([f"### {name}", ""])
            if isinstance(messages, list):
                for index, message in enumerate(messages):
                    role = message.get("role") if isinstance(message, Mapping) else None
                    content = message.get("content") if isinstance(message, Mapping) else message
                    lines.extend([
                        f"#### message {index} / {role}", "",
                        _literal_block(content),
                    ])

    lines.extend(["## Literal probe generations", ""])

    def add_probe(label: str, row: Mapping[str, Any]) -> None:
        generation = row.get("generation", {})
        lines.extend([
            f"### {label}", "",
            "**Probe**", "", _literal_block(row.get("probe", "")),
            f"- Correct target: `{row.get('correct_text')}`",
            f"- Counterfactual target: `{row.get('counterfactual_text')}`",
            f"- Content token IDs: `{generation.get('content_ids')}`",
            f"- Stop reason: `{generation.get('stop_reason')}`",
            "", "**Decoded generation**", "",
            _literal_block(generation.get("decoded_content", "")),
        ])

    phase_scores = phase.get("scores", {}) if isinstance(phase, Mapping) else {}
    if isinstance(phase_scores, Mapping):
        for name, row in phase_scores.items():
            if isinstance(row, Mapping):
                add_probe(f"Phase A / {name}", row)
    treatment = document.get("treatment", {})
    if isinstance(treatment, Mapping):
        fresh = treatment.get("fresh_scores", {})
        if isinstance(fresh, Mapping):
            for name, row in fresh.items():
                if isinstance(row, Mapping):
                    add_probe(f"Treatment fresh / {name}", row)
        arms = treatment.get("arms", [])
        if isinstance(arms, list):
            for index, arm in enumerate(arms):
                if not isinstance(arm, Mapping):
                    continue
                selector = "/".join(str(arm.get(key)) for key in (
                    "schedule", "region", "cell"
                ))
                scores = arm.get("scores", {})
                if isinstance(scores, Mapping):
                    for name, row in scores.items():
                        if isinstance(row, Mapping):
                            add_probe(
                                f"Treatment arm {index} / {selector} / {name}", row
                            )
                elif arm.get("control_status"):
                    lines.extend([
                        f"### Treatment arm {index} / {selector}", "",
                        f"- Control status: `{arm.get('control_status')}`", "",
                    ])
    return "\n".join(lines).rstrip() + "\n"


def render_technical_ledger(document: Mapping[str, Any]) -> str:
    lines = [
        f"# P01 technical generation ledger: {document.get('regime')}", "",
        f"- Protocol: `{PROTOCOL_ID}`",
        f"- Status: `{document.get('status')}`",
        "- Formal v12 decision eligible: `false`",
        f"- Raw original SHA-256: `"
        f"{document.get('raw_package_binding', {}).get('original_sha256')}`", "",
        "## Generated-versus-forced identity renders", "",
    ]
    identity = document.get("generated_forced_identity", {})
    branches = identity.get("separate_branches", []) if isinstance(identity, Mapping) else []
    if not branches:
        lines.extend(["No identity generation was completed.", ""])
    for index, branch in enumerate(branches):
        if not isinstance(branch, Mapping):
            continue
        for mode in ("generated", "forced"):
            record = branch.get(mode)
            if not isinstance(record, Mapping):
                continue
            generation = record.get("generation", {})
            lines.extend([
                f"### branch {index + 1} / {mode}", "",
                f"- Content token IDs: `{generation.get('content_ids')}`",
                f"- Stop reason: `{generation.get('stop_reason')}`", "",
                _literal_block(generation.get("decoded_content", "")),
            ])
    return "\n".join(lines).rstrip() + "\n"


def _planned_paths(run_dir: Path, stamp: str) -> dict[str, Any]:
    scratch = run_dir / ".scratch"
    result: dict[str, Any] = {
        "run_manifest": run_dir / (
            f"precision-probe-p01-run-manifest_{MODEL_SLUG}_{stamp}.json"
        ),
        "extension_decision": run_dir / (
            f"precision-probe-p01-extension-decision_{MODEL_SLUG}_{stamp}.json"
        ),
        "completion": run_dir / (
            f"precision-probe-p01-completion_{MODEL_SLUG}_{stamp}.json"
        ),
        "technical": {},
        "outcomes": {},
    }
    for regime in REGIMES:
        base = f"precision-probe-p01-technical-{regime}-raw_{MODEL_SLUG}_{stamp}"
        result["technical"][regime] = {
            "raw_scratch": scratch / f"{base}.json",
            "raw_package": run_dir / f"{base}.lossless-package",
            "identity_checkpoint_scratch": scratch / (
                f"precision-probe-p01-technical-{regime}-identity-checkpoint-raw_"
                f"{MODEL_SLUG}_{stamp}.json"
            ),
            "identity_checkpoint_package": run_dir / (
                f"precision-probe-p01-technical-{regime}-identity-checkpoint-raw_"
                f"{MODEL_SLUG}_{stamp}.lossless-package"
            ),
            "renders": run_dir / f"{base.replace('-raw_', '-renders_')}.md",
        }
        for case_id, repeats in {"e01": 2, "e02": 1, "e03": 1}.items():
            for repeat in range(1, repeats + 1):
                key = (regime, case_id, repeat)
                prefix = (
                    f"precision-probe-p01-outcome-{regime}-{case_id}-repeat{repeat}"
                )
                raw_base = f"{prefix}-raw_{MODEL_SLUG}_{stamp}"
                checkpoint_base = (
                    f"precision-probe-p01-phase-a-checkpoint-{regime}-{case_id}"
                    f"-repeat{repeat}-raw_{MODEL_SLUG}_{stamp}"
                )
                result["outcomes"][key] = OutcomePaths(
                    raw_scratch=scratch / f"{raw_base}.json",
                    raw_package=run_dir / f"{raw_base}.lossless-package",
                    phase_a_checkpoint_scratch=scratch / f"{checkpoint_base}.json",
                    phase_a_checkpoint_package=run_dir / (
                        f"{checkpoint_base}.lossless-package"
                    ),
                    compact=run_dir / f"{prefix}-compact_{MODEL_SLUG}_{stamp}.json",
                    renders=run_dir / f"{prefix}-renders_{MODEL_SLUG}_{stamp}.md",
                )
    return result


def _technical_paths_record(paths: Mapping[str, Path], repo: Path) -> dict[str, str]:
    return {key: display_path(path, repo) for key, path in paths.items()}


def run_regime_technical(
    *,
    regime: str,
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    bindings: Mapping[str, Any],
    provider_clock: ProviderClock,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    technical_started = time.monotonic()
    document: dict[str, Any] = {
        "schema": TECHNICAL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        **FORMAL_FLAGS,
        "status": "STARTED",
        "regime": regime,
        "model": args.model,
        "revision": args.revision,
        "started_at_utc": utc_now(),
        "bindings": bindings,
        "planned_paths": _technical_paths_record(paths, args.repo),
        "stage_timings": {},
        "durable_checkpoints": [],
    }
    prepared: dict[str, Any] | None = None
    fatal_interrupt: BaseException | None = None
    try:
        prepared = _stage(
            document, "prepare_precision_subject",
            lambda: prepare_precision_subject(regime, args.repo, args.allow_download),
        )
        require("model" in prepared and "tokenizer" in prepared,
                "loader omitted model or tokenizer")
        runtime = prepared.get("runtime_fingerprint")
        require(isinstance(runtime, Mapping), "loader omitted runtime fingerprint")
        require(
            isinstance(prepared.get("bindings"), Mapping)
            and bool(prepared["bindings"]),
            "loader omitted subject bindings",
        )
        eos_ids = runtime.get("eos_ids")
        require(isinstance(eos_ids, list) and eos_ids,
                "runtime fingerprint omitted EOS IDs")
        document["runtime_fingerprint"] = json_safe(runtime)
        document["subject_attestation"] = json_safe({
            key: value for key, value in prepared.items()
            if key not in {"model", "tokenizer"}
        })
        identity_fixture = load_object(args.identity_fixture)
        technical_fixture = load_object(args.technical_fixture)
        model = prepared["model"]
        tokenizer = prepared["tokenizer"]
        document["generated_forced_identity"] = _stage(
            document, "generated_forced_identity",
            lambda: run_generated_forced_identity(
                model, tokenizer, identity_fixture, [int(value) for value in eos_ids]
            ),
        )
        identity_checkpoint = {
            **document,
            "status": "CHECKPOINT",
            "checkpoint_stage": "generated_forced_identity",
            "checkpoint_completed_at_utc": utc_now(),
            "provider_elapsed_seconds": provider_clock.elapsed(),
        }
        checkpoint_binding, checkpoint_recovery, checkpoint_error = (
            persist_lossless_raw(
                identity_checkpoint,
                scratch_raw=paths["identity_checkpoint_scratch"],
                package=paths["identity_checkpoint_package"],
                repo=args.repo,
            )
        )
        document["durable_checkpoints"].append({
            "stage": "generated_forced_identity",
            "raw_package": checkpoint_binding,
            "recovery_raw_path": checkpoint_recovery,
            "persistence_error": checkpoint_error,
        })
        require(
            checkpoint_binding is not None and checkpoint_error is None,
            "generated/forced identity checkpoint did not persist and verify",
        )
        require(
            document["generated_forced_identity"].get("status") == "PASS",
            "generated/forced identity failed after durable persistence",
        )
        middle = int(technical_fixture["middle_end_msg"])
        n_plan = build_role_native_plan(
            tokenizer, technical_fixture["correct"], middle_end_msg=middle
        )
        fresh_plan = build_fresh_destination_plan(
            tokenizer, technical_fixture["correct"], middle_end_msg=middle
        )
        document["deterministic_repeats"] = {
            "correct_history_N": _stage(
                document, "correct_history_N_repeats",
                lambda: deterministic_repeats(
                    lambda: execute_replay_plan(model, n_plan),
                    "correct-history N replay",
                ),
            ),
            "fresh_destination": _stage(
                document, "fresh_destination_repeats",
                lambda: deterministic_repeats(
                    lambda: execute_fresh_plan(model, fresh_plan),
                    "fresh destination",
                ),
            ),
        }
        document["fresh_self_replacement"] = _stage(
            document, "fresh_self_replacement",
            lambda: run_fresh_self_replacement(model, fresh_plan),
        )
        _assert_technical_gate(document)
        document["status"] = "PASS"
    except BaseException as exc:
        document["status"] = "ERROR"
        document["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if not isinstance(exc, Exception):
            fatal_interrupt = exc
    document["completed_at_utc"] = utc_now()
    document["provider_elapsed_seconds"] = provider_clock.elapsed()
    document["hourly_cost_usd"] = args.hourly_cost_usd
    document["provider_cost_at_completion_usd"] = (
        document["provider_elapsed_seconds"] * args.hourly_cost_usd / 3600.0
    )
    document["technical_wrapper_wall_time_seconds"] = (
        time.monotonic() - technical_started
    )
    document["technical_wrapper_estimated_cost_usd"] = (
        document["technical_wrapper_wall_time_seconds"]
        * args.hourly_cost_usd / 3600.0
    )
    try:
        package_binding, recovery, package_error = persist_lossless_raw(
            document,
            scratch_raw=paths["raw_scratch"],
            package=paths["raw_package"],
            repo=args.repo,
        )
        if package_error:
            document["persistence_error"] = package_error
        if package_binding is not None:
            document["raw_package_binding"] = package_binding
        atomic_create_text(paths["renders"], render_technical_ledger(document))
    except BaseException:
        _release_subject(prepared)
        raise
    receipt = {
        "status": document["status"] if package_binding else "PERSISTENCE_ERROR",
        "regime": regime,
        "raw_package": package_binding,
        "renders": binding(paths["renders"], repo=args.repo),
        "recovery_raw_path": recovery,
        "persistence_error": package_error,
        "runtime_fingerprint": document.get("runtime_fingerprint"),
        "subject_bindings": document.get("subject_attestation", {}).get(
            "bindings"
        ),
        "stage_timings": document.get("stage_timings", {}),
        "durable_checkpoints": document.get("durable_checkpoints", []),
    }
    if document["status"] != "PASS" or package_binding is None:
        _release_subject(prepared)
        prepared = None
    if fatal_interrupt is not None:
        raise fatal_interrupt
    return prepared, receipt


def _assert_outcome_envelope(
    phase: Any, treatment: Any, *, case_id: str
) -> None:
    require(
        isinstance(phase, Mapping)
        and phase.get("schema") == PHASE_A_SCHEMA
        and phase.get("case_id") == case_id,
        "Phase-A payload envelope differs",
    )
    require(
        isinstance(treatment, Mapping)
        and treatment.get("schema") == TREATMENT_SCHEMA
        and treatment.get("case_id") == case_id
        and treatment.get("primary_arm_count") == 31
        and treatment.get("placebo_control_count") == 3,
        "treatment payload envelope differs",
    )


def execute_outcome(
    *,
    regime: str,
    case_id: str,
    repeat_index: int,
    prepared: Mapping[str, Any],
    args: argparse.Namespace,
    paths: OutcomePaths,
    common_bindings: Mapping[str, Any],
    provider_clock: ProviderClock,
) -> OutcomeReceipt:
    case_path = args.case_paths[case_id]
    case = load_object(case_path)
    require(case.get("case_id") == case_id, "case file ID differs")
    runtime = prepared["runtime_fingerprint"]
    eos_ids = [int(value) for value in runtime["eos_ids"]]
    document: dict[str, Any] = {
        "schema": OUTCOME_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        **FORMAL_FLAGS,
        "status": "STARTED",
        "regime": regime,
        "case_id": case_id,
        "repeat_index": repeat_index,
        "model": args.model,
        "revision": args.revision,
        "started_at_utc": utc_now(),
        "bindings": {
            **common_bindings,
            "case": binding(case_path, repo=args.repo),
        },
        "runtime_fingerprint": json_safe(runtime),
        "subject_attestation": json_safe({
            key: value for key, value in prepared.items()
            if key not in {"model", "tokenizer"}
        }),
        "stage_timings": {},
        "durable_checkpoints": [],
    }
    outcome_started = time.monotonic()
    fatal_interrupt: BaseException | None = None
    try:
        phase = _stage(
            document, "phase_a",
            lambda: run_phase_a_case(
                prepared["model"], prepared["tokenizer"], case,
                eos_ids=eos_ids,
            ),
        )
        document["phase_a"] = phase
        phase_checkpoint = {
            **document,
            "status": "CHECKPOINT",
            "checkpoint_stage": "phase_a",
            "checkpoint_completed_at_utc": utc_now(),
            "provider_elapsed_seconds": provider_clock.elapsed(),
        }
        checkpoint_binding, checkpoint_recovery, checkpoint_error = (
            persist_lossless_raw(
                phase_checkpoint,
                scratch_raw=paths.phase_a_checkpoint_scratch,
                package=paths.phase_a_checkpoint_package,
                repo=args.repo,
            )
        )
        document["durable_checkpoints"].append({
            "stage": "phase_a",
            "raw_package": checkpoint_binding,
            "recovery_raw_path": checkpoint_recovery,
            "persistence_error": checkpoint_error,
        })
        require(
            checkpoint_binding is not None and checkpoint_error is None,
            "Phase-A checkpoint did not persist and verify",
        )
        treatment = _stage(
            document, "treatment",
            lambda: run_treatment_case(
                prepared["model"], prepared["tokenizer"], case,
                eos_ids=eos_ids,
            ),
        )
        _assert_outcome_envelope(phase, treatment, case_id=case_id)
        document["treatment"] = treatment
        document["status"] = "COMPLETE"
    except BaseException as exc:
        document["status"] = "ERROR"
        document["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if not isinstance(exc, Exception):
            fatal_interrupt = exc
    wrapper_wall = time.monotonic() - outcome_started
    outcome_wall = sum(
        float(document["stage_timings"].get(stage, {}).get(
            "wall_time_seconds", 0.0
        ))
        for stage in ("phase_a", "treatment")
    )
    document["outcome_execution_wall_time_seconds"] = outcome_wall
    document["outcome_wall_time_seconds"] = outcome_wall
    document["outcome_wrapper_wall_time_through_treatment_seconds"] = wrapper_wall
    document["completed_at_utc"] = utc_now()
    document["provider_elapsed_seconds"] = provider_clock.elapsed()
    document["hourly_cost_usd"] = args.hourly_cost_usd
    document["outcome_execution_estimated_cost_usd"] = (
        outcome_wall * args.hourly_cost_usd / 3600.0
    )
    document["outcome_wrapper_estimated_cost_through_treatment_usd"] = (
        wrapper_wall * args.hourly_cost_usd / 3600.0
    )
    document["provider_cost_at_completion_usd"] = (
        document["provider_elapsed_seconds"] * args.hourly_cost_usd / 3600.0
    )

    package_binding, recovery, package_error = persist_lossless_raw(
        document, scratch_raw=paths.raw_scratch, package=paths.raw_package,
        repo=args.repo,
    )
    completion_status = document["status"]
    if package_binding is None:
        completion_status = "PERSISTENCE_ERROR"
    receipt = OutcomeReceipt(
        regime=regime,
        case_id=case_id,
        repeat_index=repeat_index,
        completion_status=completion_status,
        outcome_wall_time_seconds=outcome_wall,
        raw_package=package_binding,
        recovery_raw_path=recovery,
    )
    if fatal_interrupt is not None:
        if package_binding is not None:
            # A fatal signal prevents the extension decision from running, so
            # exposing the already-produced partial render can no longer bias
            # scheduling.  Preserve it before propagating the interruption.
            document["raw_package_binding"] = package_binding
            atomic_create_text(paths.renders, render_outcome_ledger(document))
        raise fatal_interrupt
    return receipt


def materialize_derived(
    receipt: OutcomeReceipt, *, paths: OutcomePaths, args: argparse.Namespace
) -> OutcomeReceipt:
    require(receipt.raw_package is not None, "cannot derive from absent raw package")
    document = _load_packaged_document(
        paths.raw_package, scratch_dir=paths.raw_scratch.parent
    )
    document["raw_package_binding"] = dict(receipt.raw_package)
    compact = compact_outcome(document)
    atomic_create_json(paths.compact, compact)
    atomic_create_text(paths.renders, render_outcome_ledger(document))
    return OutcomeReceipt(
        regime=receipt.regime,
        case_id=receipt.case_id,
        repeat_index=receipt.repeat_index,
        completion_status=receipt.completion_status,
        outcome_wall_time_seconds=receipt.outcome_wall_time_seconds,
        raw_package=receipt.raw_package,
        recovery_raw_path=receipt.recovery_raw_path,
        compact_path=str(paths.compact),
        renders_path=str(paths.renders),
    )


def _receipt_record(receipt: OutcomeReceipt, repo: Path) -> dict[str, Any]:
    row = asdict(receipt)
    if receipt.compact_path:
        row["compact"] = binding(Path(receipt.compact_path), repo=repo)
        row["compact_path"] = row["compact"]["path"]
    if receipt.renders_path:
        row["renders"] = binding(Path(receipt.renders_path), repo=repo)
        row["renders_path"] = row["renders"]["path"]
    return json_safe(row)


def _release_subject(prepared: dict[str, Any] | None) -> None:
    if prepared is None:
        return
    prepared.pop("model", None)
    prepared.pop("tokenizer", None)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _deadline_seconds(args: argparse.Namespace) -> float:
    return min(
        7200.0,
        float(args.provider_wall_cap_seconds),
        4.0 * 3600.0 / float(args.hourly_cost_usd),
    )


def _within_deadline(args: argparse.Namespace, clock: ProviderClock) -> bool:
    return clock.elapsed() < _deadline_seconds(args)


def _diff_paths(left: Any, right: Any, prefix: str = "$", *, limit: int = 500) -> list[str]:
    differences: list[str] = []

    def walk(a: Any, b: Any, path: str) -> None:
        if len(differences) >= limit:
            return
        if type(a) is not type(b):
            differences.append(path + " (type)")
        elif isinstance(a, Mapping):
            for key in sorted(set(a) | set(b), key=str):
                child = f"{path}.{key}"
                if key not in a or key not in b:
                    differences.append(child + " (missing)")
                else:
                    walk(a[key], b[key], child)
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append(path + " (length)")
            for index, (av, bv) in enumerate(zip(a, b)):
                walk(av, bv, f"{path}[{index}]")
        elif a != b:
            differences.append(path)

    walk(left, right, prefix)
    return differences


def repeat_stability(
    receipts: Sequence[OutcomeReceipt], paths: Mapping[tuple[str, str, int], OutcomePaths]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for regime in REGIMES:
        pair = sorted(
            [row for row in receipts if row.regime == regime and row.case_id == "e01"],
            key=lambda row: row.repeat_index,
        )
        if len(pair) != 2 or any(
            row.completion_status != "COMPLETE" or row.raw_package is None for row in pair
        ):
            result[regime] = {"status": "UNAVAILABLE_INCOMPLETE_REPEATS"}
            continue
        first = _load_packaged_document(
            paths[(regime, "e01", 1)].raw_package,
            scratch_dir=paths[(regime, "e01", 1)].raw_scratch.parent,
        )
        second = _load_packaged_document(
            paths[(regime, "e01", 2)].raw_package,
            scratch_dir=paths[(regime, "e01", 2)].raw_scratch.parent,
        )
        model_facing_first = {
            "phase_a": first.get("phase_a"), "treatment": first.get("treatment")
        }
        model_facing_second = {
            "phase_a": second.get("phase_a"), "treatment": second.get("treatment")
        }
        differences = _diff_paths(model_facing_first, model_facing_second)
        result[regime] = {
            "status": "PASS" if not differences else "FAIL",
            "normalized_scope": ["phase_a", "treatment"],
            "repeat1_sha256": hashlib.sha256(
                canonical_json_bytes(model_facing_first)
            ).hexdigest(),
            "repeat2_sha256": hashlib.sha256(
                canonical_json_bytes(model_facing_second)
            ).hexdigest(),
            "differing_field_count_at_most_500": len(differences),
            "differing_fields": differences,
            "difference_list_truncated": len(differences) >= 500,
        }
    return result


def selected_outcomes_complete(
    receipts: Sequence[OutcomeReceipt], selected_case_repeats: Mapping[str, Any]
) -> bool:
    expected = {
        (regime, case_id, repeat)
        for regime in REGIMES
        for case_id, count in selected_case_repeats.items()
        for repeat in range(1, int(count) + 1)
    }
    observed = {(row.regime, row.case_id, row.repeat_index) for row in receipts}
    if observed != expected or len(receipts) != len(expected):
        return False
    return all(
        row.completion_status == "COMPLETE"
        and isinstance(row.raw_package, Mapping)
        and row.raw_package.get("verification_status") == "VERIFIED"
        and isinstance(row.compact_path, str)
        and Path(row.compact_path).is_file()
        and isinstance(row.renders_path, str)
        and Path(row.renders_path).is_file()
        and row.recovery_raw_path is None
        for row in receipts
    )


def matched_runtime_gate(
    nf4_technical: Mapping[str, Any], bf16_technical: Mapping[str, Any]
) -> dict[str, Any]:
    if nf4_technical.get("status") != "PASS" or bf16_technical.get("status") != "PASS":
        return {
            "status": "NOT_EVALUABLE_TECHNICAL_INCOMPLETE",
            "differing_fields": [],
        }
    nf4_runtime = nf4_technical.get("runtime_fingerprint")
    bf16_runtime = bf16_technical.get("runtime_fingerprint")
    nf4_bindings = nf4_technical.get("subject_bindings")
    bf16_bindings = bf16_technical.get("subject_bindings")
    if not all(isinstance(value, Mapping) for value in (
            nf4_runtime, bf16_runtime, nf4_bindings, bf16_bindings)):
        return {"status": "NOT_EVALUABLE_BINDINGS_ABSENT", "differing_fields": []}
    runtime_fields = (
        "model_id", "requested_revision", "resolved_snapshot", "architecture",
        "attention_backend", "kv_dtype", "eos_ids", "geometry",
        "repository_commit", "dependency_versions", "gpu_uuid",
    )
    binding_fields = (
        "repository", "dependencies", "host", "checkpoint", "g0",
        "tokenizer", "model", "eos", "placement",
    )
    missing = [
        f"{side}.{group}.{field}"
        for side, runtime, bindings in (
            ("nf4", nf4_runtime, nf4_bindings),
            ("bf16", bf16_runtime, bf16_bindings),
        )
        for group, source, fields in (
            ("runtime", runtime, runtime_fields),
            ("bindings", bindings, binding_fields),
        )
        for field in fields if field not in source
    ]
    if missing:
        return {
            "status": "NOT_EVALUABLE_BINDINGS_ABSENT",
            "missing_fields": missing,
            "differing_fields": [],
        }
    nf4_view = {
        "runtime": {field: nf4_runtime.get(field) for field in runtime_fields},
        "bindings": {field: nf4_bindings.get(field) for field in binding_fields},
    }
    bf16_view = {
        "runtime": {field: bf16_runtime.get(field) for field in runtime_fields},
        "bindings": {field: bf16_bindings.get(field) for field in binding_fields},
    }
    differences = _diff_paths(nf4_view, bf16_view)
    return {
        "status": "PASS" if not differences else "FAIL",
        "compared_runtime_fields": list(runtime_fields),
        "compared_subject_binding_fields": list(binding_fields),
        "differing_fields": differences,
        "nf4_matched_view_sha256": hashlib.sha256(
            canonical_json_bytes(nf4_view)
        ).hexdigest(),
        "bf16_matched_view_sha256": hashlib.sha256(
            canonical_json_bytes(bf16_view)
        ).hexdigest(),
    }


def _inventory_tree(
    run_dir: Path, *, repo: Path, exclude: set[Path]
) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        if path in exclude:
            continue
        digest, size = sha256_file(path)
        rows.append({
            "path": display_path(path, repo),
            "sha256": digest,
            "size_bytes": size,
        })
    return rows


def _manifest_checkpoint(path: Path, document: dict[str, Any], clock: ProviderClock,
                         args: argparse.Namespace) -> None:
    document["updated_at_utc"] = utc_now()
    document["provider_elapsed_seconds"] = clock.elapsed()
    document["estimated_provider_cost_usd"] = (
        document["provider_elapsed_seconds"] * args.hourly_cost_usd / 3600.0
    )
    atomic_replace_json(path, document)


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any], int]:
    require(args.model == MODEL_ID, "model differs from frozen p01 subject")
    require(args.revision == REVISION, "revision differs from frozen p01 subject")
    require(math.isfinite(args.hourly_cost_usd) and args.hourly_cost_usd > 0,
            "hourly provider rate must be positive")
    require(
        math.isfinite(args.provider_elapsed_seconds_at_start)
        and args.provider_elapsed_seconds_at_start >= 0,
        "provider elapsed seconds at start must be nonnegative",
    )
    require(
        math.isfinite(args.provider_wall_cap_seconds)
        and 0 < args.provider_wall_cap_seconds <= 7200,
        "provider wall cap must be in (0, 7200]",
    )
    require(
        args.provider_elapsed_seconds_at_start < _deadline_seconds(args),
        "provider ceiling was already exhausted before runner start",
    )
    stamp = utc_stamp()
    run_dir = args.output_dir / f"precision-probe-p01_{MODEL_SLUG}_{stamp}"
    require(not run_dir.exists(), f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / ".scratch").mkdir()
    planned = _planned_paths(run_dir, stamp)
    manifest_path: Path = planned["run_manifest"]
    completion_path: Path = planned["completion"]
    print(json.dumps({
        "event": "RUN",
        "protocol_id": PROTOCOL_ID,
        "model": args.model,
        "revision": args.revision,
        "run_manifest_path": str(manifest_path),
        "completion_marker_path": str(completion_path),
        "run_directory": str(run_dir),
    }, sort_keys=True), flush=True)

    clock = ProviderClock(
        elapsed_seconds_at_start=args.provider_elapsed_seconds_at_start,
        monotonic_at_start=time.monotonic(),
    )
    input_paths = {
        "preregistration": args.prereg,
        "identity_fixture": args.identity_fixture,
        "technical_fixture": args.technical_fixture,
        "case_e01": args.case_paths["e01"],
        "case_e02": args.case_paths["e02"],
        "case_e03": args.case_paths["e03"],
        "runner": Path(__file__).resolve(),
        "artifact_packager": Path(artifact_packager.__file__).resolve(),
        "precision_subject_loader": ROOT / "src/precision_probe_p01_loader.py",
        "coherent_canary_case": ROOT / "src/coherent_canary_case.py",
        "coherent_canary_runtime": ROOT / "src/coherent_canary_runtime.py",
        "coherent_canary_technical": ROOT / "src/coherent_canary_technical.py",
        "coherent_canary_tokens": ROOT / "src/coherent_canary_tokens.py",
        "coherent_canary_controls": ROOT / "src/coherent_canary_controls.py",
        "coherent_canary_schema": ROOT / "src/coherent_canary_schema.py",
        "coherent_state_tokens": ROOT / "src/coherent_state_tokens.py",
    }
    common_bindings = {
        name: binding(path, repo=args.repo) for name, path in input_paths.items()
    }
    manifest: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        **FORMAL_FLAGS,
        "status": "STARTED",
        "model": args.model,
        "revision": args.revision,
        "run_stamp_utc": stamp,
        "run_directory": display_path(run_dir, args.repo),
        "started_at_utc": utc_now(),
        "bindings": common_bindings,
        "provenance": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "repository_head": git_value(args.repo, "rev-parse", "HEAD"),
            "repository_branch": git_value(args.repo, "branch", "--show-current"),
        },
        "provider": {
            "hourly_cost_usd": args.hourly_cost_usd,
            "elapsed_seconds_at_runner_start": args.provider_elapsed_seconds_at_start,
            "provider_wall_cap_seconds": args.provider_wall_cap_seconds,
            "effective_hard_deadline_seconds": _deadline_seconds(args),
            "cost_cap_usd": 4.0,
        },
        "planned_output_names": {
            "extension_decision": display_path(planned["extension_decision"], args.repo),
            "completion": display_path(completion_path, args.repo),
            "technical": {
                regime: _technical_paths_record(planned["technical"][regime], args.repo)
                for regime in REGIMES
            },
            "outcomes": {
                f"{regime}/{case_id}/repeat{repeat}": {
                    key: display_path(value, args.repo)
                    for key, value in asdict(paths).items()
                }
                for (regime, case_id, repeat), paths in planned["outcomes"].items()
            },
        },
        "regimes": {regime: {"status": "PENDING", "outcomes": []}
                    for regime in REGIMES},
        "extension_decision": None,
        "selected_case_repeats": None,
        "matched_runtime_gate": None,
        "repeat_stability": None,
    }
    _manifest_checkpoint(manifest_path, manifest, clock, args)
    all_receipts: list[OutcomeReceipt] = []
    nf4_prepared: dict[str, Any] | None = None
    fatal_interrupt: BaseException | None = None
    try:
        nf4_prepared, technical_receipt = run_regime_technical(
            regime="nf4", args=args, paths=planned["technical"]["nf4"],
            bindings=common_bindings, provider_clock=clock,
        )
        manifest["regimes"]["nf4"].update({
            "status": "TECHNICAL_PASS" if nf4_prepared else "TECHNICAL_FAIL",
            "technical": technical_receipt,
        })
        _manifest_checkpoint(manifest_path, manifest, clock, args)

        nf4_e01: list[OutcomeReceipt] = []
        for repeat in (1, 2):
            if nf4_prepared is None or not _within_deadline(args, clock):
                receipt = OutcomeReceipt(
                    regime="nf4", case_id="e01", repeat_index=repeat,
                    completion_status=(
                        "SKIPPED_TECHNICAL_FAIL" if nf4_prepared is None
                        else "SKIPPED_PROVIDER_CEILING"
                    ),
                    outcome_wall_time_seconds=0.0,
                    raw_package=None, recovery_raw_path=None,
                )
            else:
                receipt = execute_outcome(
                    regime="nf4", case_id="e01", repeat_index=repeat,
                    prepared=nf4_prepared, args=args,
                    paths=planned["outcomes"][("nf4", "e01", repeat)],
                    common_bindings=common_bindings, provider_clock=clock,
                )
            nf4_e01.append(receipt)
            all_receipts.append(receipt)
            manifest["regimes"]["nf4"]["outcomes"].append(
                _receipt_record(receipt, args.repo)
            )
            _manifest_checkpoint(manifest_path, manifest, clock, args)

        # This is the sole extension decision.  Only scalar projections cross
        # the boundary; raw packages remain unopened and no derived artifacts
        # exist yet.
        decision = decide_extension(
            e01_timings=[row.extension_timing() for row in nf4_e01],
            provider_elapsed_seconds=clock.elapsed(),
            hourly_cost_usd=args.hourly_cost_usd,
            provider_wall_cap_seconds=args.provider_wall_cap_seconds,
        )
        atomic_create_json(planned["extension_decision"], decision)
        manifest["extension_decision"] = binding(
            planned["extension_decision"], repo=args.repo
        )
        manifest["selected_case_repeats"] = decision["selected_case_repeats"]
        _manifest_checkpoint(manifest_path, manifest, clock, args)

        # Only now may the two NF4 e01 packages be parsed/summarized/rendered.
        for index, receipt in enumerate(nf4_e01):
            if receipt.raw_package is None:
                continue
            derived = materialize_derived(
                receipt,
                paths=planned["outcomes"][("nf4", "e01", receipt.repeat_index)],
                args=args,
            )
            nf4_e01[index] = derived
            all_receipts[all_receipts.index(receipt)] = derived
        manifest["regimes"]["nf4"]["outcomes"] = [
            _receipt_record(row, args.repo)
            for row in all_receipts if row.regime == "nf4"
        ]
        _manifest_checkpoint(manifest_path, manifest, clock, args)

        if decision["extension_allowed"]:
            for case_id in ("e02", "e03"):
                if nf4_prepared is None or not _within_deadline(args, clock):
                    receipt = OutcomeReceipt(
                        regime="nf4", case_id=case_id, repeat_index=1,
                        completion_status="SKIPPED_PROVIDER_CEILING",
                        outcome_wall_time_seconds=0.0,
                        raw_package=None, recovery_raw_path=None,
                    )
                else:
                    receipt = execute_outcome(
                        regime="nf4", case_id=case_id, repeat_index=1,
                        prepared=nf4_prepared, args=args,
                        paths=planned["outcomes"][("nf4", case_id, 1)],
                        common_bindings=common_bindings, provider_clock=clock,
                    )
                    if receipt.raw_package is not None:
                        receipt = materialize_derived(
                            receipt,
                            paths=planned["outcomes"][("nf4", case_id, 1)],
                            args=args,
                        )
                all_receipts.append(receipt)
                manifest["regimes"]["nf4"]["outcomes"] = [
                    _receipt_record(row, args.repo)
                    for row in all_receipts if row.regime == "nf4"
                ]
                _manifest_checkpoint(manifest_path, manifest, clock, args)
        manifest["regimes"]["nf4"]["status"] = "OUTCOMES_FINISHED"
        _release_subject(nf4_prepared)
        nf4_prepared = None
        _manifest_checkpoint(manifest_path, manifest, clock, args)

        if _within_deadline(args, clock):
            bf16_prepared, technical_receipt = run_regime_technical(
                regime="bf16", args=args, paths=planned["technical"]["bf16"],
                bindings=common_bindings, provider_clock=clock,
            )
        else:
            bf16_prepared = None
            technical_receipt = {
                "status": "SKIPPED_PROVIDER_CEILING", "regime": "bf16",
                "raw_package": None, "renders": None,
                "recovery_raw_path": None, "persistence_error": None,
                "runtime_fingerprint": None, "stage_timings": {},
            }
        runtime_match = matched_runtime_gate(
            manifest["regimes"]["nf4"].get("technical", {}),
            technical_receipt,
        )
        manifest["matched_runtime_gate"] = runtime_match
        if runtime_match["status"] != "PASS":
            _release_subject(bf16_prepared)
            bf16_prepared = None
            if technical_receipt.get("status") == "PASS":
                technical_receipt["status"] = "PAIRED_RUNTIME_MISMATCH"
            technical_receipt["paired_runtime_match"] = runtime_match
        manifest["regimes"]["bf16"].update({
            "status": "TECHNICAL_PASS" if bf16_prepared else "TECHNICAL_FAIL",
            "technical": technical_receipt,
        })
        _manifest_checkpoint(manifest_path, manifest, clock, args)
        try:
            selected = decision["selected_case_repeats"]
            for case_id in ("e01", "e02", "e03"):
                repeat_count = int(selected.get(case_id, 0))
                for repeat in range(1, repeat_count + 1):
                    if bf16_prepared is None or not _within_deadline(args, clock):
                        receipt = OutcomeReceipt(
                            regime="bf16", case_id=case_id, repeat_index=repeat,
                            completion_status=(
                                "SKIPPED_TECHNICAL_FAIL" if bf16_prepared is None
                                else "SKIPPED_PROVIDER_CEILING"
                            ),
                            outcome_wall_time_seconds=0.0,
                            raw_package=None, recovery_raw_path=None,
                        )
                    else:
                        receipt = execute_outcome(
                            regime="bf16", case_id=case_id, repeat_index=repeat,
                            prepared=bf16_prepared, args=args,
                            paths=planned["outcomes"][("bf16", case_id, repeat)],
                            common_bindings=common_bindings, provider_clock=clock,
                        )
                        if receipt.raw_package is not None:
                            receipt = materialize_derived(
                                receipt,
                                paths=planned["outcomes"][("bf16", case_id, repeat)],
                                args=args,
                            )
                    all_receipts.append(receipt)
                    manifest["regimes"]["bf16"]["outcomes"] = [
                        _receipt_record(row, args.repo)
                        for row in all_receipts if row.regime == "bf16"
                    ]
                    _manifest_checkpoint(manifest_path, manifest, clock, args)
            manifest["regimes"]["bf16"]["status"] = "OUTCOMES_FINISHED"
        finally:
            _release_subject(bf16_prepared)

        manifest["repeat_stability"] = repeat_stability(
            all_receipts, planned["outcomes"]
        )
        technical_pass = all(
            manifest["regimes"][regime].get("technical", {}).get("status") == "PASS"
            for regime in REGIMES
        )
        provider_ceiling_pass = clock.elapsed() <= _deadline_seconds(args)
        manifest["provider"]["completion_within_effective_deadline"] = (
            provider_ceiling_pass
        )
        manifest["status"] = (
            "COMPLETE" if (
                technical_pass and provider_ceiling_pass
                and selected_outcomes_complete(
                    all_receipts, decision["selected_case_repeats"]
                )
            ) else "PARTIAL"
        )
    except BaseException as exc:
        manifest["status"] = "ERROR"
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if not isinstance(exc, Exception):
            fatal_interrupt = exc
    finally:
        _release_subject(nf4_prepared)
        if (manifest.get("status") == "COMPLETE"
                and clock.elapsed() > _deadline_seconds(args)):
            manifest["status"] = "PARTIAL"
            manifest["provider"]["completion_within_effective_deadline"] = False
        manifest["completed_at_utc"] = utc_now()
        _manifest_checkpoint(manifest_path, manifest, clock, args)
        if (manifest.get("status") == "COMPLETE"
                and clock.elapsed() > _deadline_seconds(args)):
            manifest["status"] = "PARTIAL"
            manifest["provider"]["completion_within_effective_deadline"] = False
            _manifest_checkpoint(manifest_path, manifest, clock, args)
        completion_elapsed = clock.elapsed()
        completion = {
            "schema": COMPLETION_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            **FORMAL_FLAGS,
            "status": manifest["status"],
            "model": args.model,
            "revision": args.revision,
            "created_at_utc": utc_now(),
            "run_manifest": binding(manifest_path, repo=args.repo),
            "matched_runtime_gate": manifest.get("matched_runtime_gate"),
            "artifact_inventory": _inventory_tree(
                run_dir, repo=args.repo, exclude={completion_path}
            ),
            "recovery_raw_files": [
                path.relative_to(run_dir).as_posix()
                for path in sorted((run_dir / ".scratch").glob("*.json"))
            ],
            "provider_elapsed_seconds": completion_elapsed,
            "estimated_provider_cost_usd": (
                completion_elapsed * args.hourly_cost_usd / 3600.0
            ),
        }
        atomic_create_json(completion_path, completion)

    if fatal_interrupt is not None:
        raise fatal_interrupt

    exit_code = 0 if manifest["status"] == "COMPLETE" else (
        2 if manifest["status"] == "PARTIAL" else 1
    )
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "status": manifest["status"],
        "run_manifest_path": str(manifest_path),
        "completion_marker_path": str(completion_path),
        "exit_code": exit_code,
    }, sort_keys=True), flush=True)
    return completion_path, manifest, exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--identity-fixture", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--technical-fixture", type=Path, default=DEFAULT_TECHNICAL)
    parser.add_argument("--case-e01", type=Path, default=DEFAULT_CASES["e01"])
    parser.add_argument("--case-e02", type=Path, default=DEFAULT_CASES["e02"])
    parser.add_argument("--case-e03", type=Path, default=DEFAULT_CASES["e03"])
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--hourly-cost-usd", required=True, type=float)
    parser.add_argument(
        "--provider-elapsed-seconds-at-start", required=True, type=float,
        help="provider wall seconds already consumed before this runner began",
    )
    parser.add_argument(
        "--provider-wall-cap-seconds", type=float, default=7200.0,
        help="operational provider wall cap; the frozen maximum is 7200",
    )
    args = parser.parse_args(argv)
    args.repo = args.repo.resolve()
    args.output_dir = args.output_dir.resolve()
    args.prereg = args.prereg.resolve()
    args.identity_fixture = args.identity_fixture.resolve()
    args.technical_fixture = args.technical_fixture.resolve()
    args.case_paths = {
        "e01": args.case_e01.resolve(),
        "e02": args.case_e02.resolve(),
        "e03": args.case_e03.resolve(),
    }
    return args


def main() -> None:
    _, _, code = run(parse_args())
    raise SystemExit(code)


if __name__ == "__main__":
    main()
