#!/usr/bin/env python3
"""Strict independent analysis of a frozen precision-probe-p02 run.

The paid runner's compact JSON and Markdown ledgers are not treated as data.
Every lossless package is reconstructed and verified first; the primary
matched-repeat-1 estimands and all derived artifacts are then recomputed from
the reconstructed raw documents.  P02 is a one-fixture descriptive screen of
a bundled weight-representation/kernel axis.  This analyzer deliberately
computes no p-value, confidence interval, ratio, equivalence result, formal
release decision, or quantization-dependence inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "scripts", ROOT / "src"):
    value = str(import_root)
    if value not in sys.path:
        sys.path.insert(0, value)

# Reuse the already-tested lossless-package and float32 parsers, plus only the
# pure P02 compact/render constructors.  Eligibility, grid validation,
# estimands, decision reconstruction, and repeat guards are implemented here.
import analyze_precision_probe_p01 as common  # noqa: E402
import run_precision_probe_p01 as p01_runner  # noqa: E402
import run_precision_probe_p02 as p02_runner  # noqa: E402


PROTOCOL_ID = "precision-probe-p02"
ANALYSIS_SCHEMA = "precision_probe_p02_independent_analysis_v1"
PACKAGE_SCHEMA = common.PACKAGE_SCHEMA
OUTCOME_SCHEMA = "precision_probe_p02_outcome_raw_v1"
TECHNICAL_SCHEMA = "precision_probe_p02_technical_raw_v1"
CHECKPOINT_SCHEMA = "precision_probe_p02_checkpoint_v1"
RUN_MANIFEST_SCHEMA = "precision_probe_p02_run_manifest_v1"
COMPLETION_SCHEMA = "precision_probe_p02_completion_v1"
DECISION_SCHEMA = "precision_probe_p02_continuation_decision_v1"
TIMING_SCHEMA = "precision_probe_p02_timing_receipt_v1"
PHASE_A_SCHEMA = "coherent_state_decision_canary_v12_phase_a_raw_v1"
TREATMENT_SCHEMA = "precision_probe_p02_targeted_treatment_raw_v1"
FOUNDATION_SCHEMA = "precision_probe_p02_treatment_foundation_v1"
OUTER_RECEIPT_SCHEMA = "precision_probe_p02_pod_receipt_v1"
RUNTIME_ATTESTATION_SCHEMA = "precision_probe_p02_runtime_attestation_v1"

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
PREREG_SHA256 = "dbcece8f189a0574b776149cf597dc1ce8a3b592981440062c749ddb404e221d"
DESIGN_ID = "coherent-state-decision-canary-v12"
CASE_ID = "e01"
REGIMES = ("nf4", "bf16")
R2 = "R2_boundary"
GENERATION_CELLS = ("FF", "FC", "FW", "CC", "WW")
GRAFT_SELECTORS = (
    ("N", R2, "FC"),
    ("N", R2, "FW"),
    ("N", R2, "CC"),
    ("N", R2, "WW"),
    ("P", R2, "FC"),
    ("P", R2, "FW"),
)
DECISION_NAMES = (
    "initial_schedule",
    "bf16_technical_admission",
    "bf16_repeat1",
    "bf16_repeat2",
)
RIDER_STATUSES = {
    "NOT_AUTHORIZED",
    "MATCHED_COMPLETE",
    "NF4_ONLY_TIMING_STOP",
    "AUTHORIZED_INCOMPLETE",
}
FORMAL_FLAGS = {
    "formal_v12_decision_eligible": False,
    "v12_reentry_authorized": False,
    "component_reuse_does_not_inherit_v12_eligibility": True,
    "semantic_evidence_eligible": False,
}
REQUIRED_SOURCE_BINDINGS = {
    "preregistration",
    "identity_fixture",
    "technical_fixture",
    "case_e01",
    "runner",
    "targeted_case",
    "p01_runner_reused_technical",
    "p01_subject_loader",
    "artifact_packager",
    "coherent_canary_case",
    "coherent_canary_runtime",
    "coherent_canary_technical",
    "coherent_canary_tokens",
    "coherent_canary_controls",
    "coherent_canary_schema",
    "coherent_state_tokens",
}
MATCHED_RUNTIME_FIELDS = (
    "model_id",
    "requested_revision",
    "resolved_snapshot",
    "architecture",
    "attention_backend",
    "kv_dtype",
    "eos_ids",
    "geometry",
    "repository_commit",
    "dependency_versions",
    "gpu_uuid",
)
MATCHED_SUBJECT_FIELDS = (
    "repository",
    "dependencies",
    "host",
    "checkpoint",
    "g0",
    "tokenizer",
    "model",
    "eos",
    "placement",
)
EXPECTED_DEPENDENCIES = {
    "torch": "2.12.1",
    "transformers": "4.57.6",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.49.2",
    "huggingface-hub": "0.36.2",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
}


AnalysisError = common.AnalysisError


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _version_tuple(value: Any, label: str) -> tuple[int, ...]:
    require(isinstance(value, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value),
            f"{label} version differs")
    return tuple(int(part) for part in value.split("."))


def _single(paths: Sequence[Path], label: str) -> Path:
    require(len(paths) == 1, f"P02 requires exactly one {label}")
    return paths[0]


def _formal_boundary(document: Mapping[str, Any], label: str) -> None:
    require(
        document.get("protocol_id") == PROTOCOL_ID
        and all(document.get(key) is value for key, value in FORMAL_FLAGS.items()),
        f"{label} protocol/formal boundary differs",
    )


def _safe_relative(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} path is absent")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts,
            f"{label} path is unsafe")
    return path


def _resolve_run_path(run_dir: Path, declared: Any, label: str) -> Path:
    relative = _safe_relative(declared, label)
    normalized = relative.as_posix()
    candidates: list[Path] = []
    direct = (ROOT / relative).resolve()
    if direct.exists() and (direct == run_dir or run_dir in direct.parents):
        candidates.append(direct)
    for path in run_dir.rglob(relative.name):
        rel = path.relative_to(run_dir).as_posix()
        if (normalized == rel or normalized.endswith("/" + rel)
                or rel.endswith("/" + normalized)):
            candidates.append(path.resolve())
    unique = list(dict.fromkeys(candidates))
    require(len(unique) == 1, f"{label} path cannot be resolved uniquely")
    return unique[0]


def _verify_file_binding(run_dir: Path, row: Any, label: str) -> Path:
    require(
        isinstance(row, Mapping)
        and set(row) == {"path", "sha256", "size_bytes"},
        f"{label} binding shape differs",
    )
    path = _resolve_run_path(run_dir, row.get("path"), label)
    require(
        path.is_file()
        and not path.is_symlink()
        and row.get("sha256") == common.file_sha256(path)
        and row.get("size_bytes") == path.stat().st_size,
        f"{label} binding bytes differ",
    )
    return path


def _verify_package_binding(
    run_dir: Path,
    binding: Any,
    package_row: Mapping[str, Any],
    label: str,
) -> Path:
    fields = {
        "path",
        "manifest_path",
        "manifest_sha256",
        "manifest_size_bytes",
        "chunk_count",
        "original_sha256",
        "original_size_bytes",
        "compressed_sha256",
        "compressed_size_bytes",
        "verification_status",
    }
    require(isinstance(binding, Mapping) and set(binding) == fields,
            f"{label} package binding shape differs")
    package_dir = _resolve_run_path(run_dir, binding.get("path"), label)
    require(package_dir.is_dir() and package_dir == package_row["package_dir"],
            f"{label} package path differs")
    manifest_path = _resolve_run_path(
        run_dir, binding.get("manifest_path"), f"{label} package manifest")
    require(manifest_path == package_dir / "manifest.json",
            f"{label} package manifest path differs")
    manifest = common.load_object(manifest_path)
    require(
        binding.get("verification_status") == "VERIFIED"
        and binding.get("manifest_sha256") == common.file_sha256(manifest_path)
        and binding.get("manifest_size_bytes") == manifest_path.stat().st_size
        and binding.get("chunk_count") == manifest.get("chunk_count")
        and binding.get("original_sha256")
        == manifest.get("original", {}).get("sha256")
        and binding.get("original_size_bytes")
        == manifest.get("original", {}).get("size_bytes")
        and binding.get("compressed_sha256")
        == manifest.get("compression", {}).get("compressed_sha256")
        and binding.get("compressed_size_bytes")
        == manifest.get("compression", {}).get("compressed_size_bytes")
        and package_row["package_manifest_sha256"]
        == binding.get("manifest_sha256")
        and package_row["raw_artifact_sha256"]
        == binding.get("original_sha256"),
        f"{label} package receipt differs from reconstructed bytes",
    )
    return package_dir


def package_documents(run_dir: Path) -> list[dict[str, Any]]:
    """Reconstruct and byte-verify every lossless package under one run."""
    manifests = sorted(run_dir.rglob("*.lossless-package/manifest.json"))
    require(manifests, "P02 run contains no lossless packages")
    rows: list[dict[str, Any]] = []
    for manifest_path in manifests:
        try:
            manifest, raw = common.reconstruct_package(manifest_path.parent)
        except common.AnalysisError as exc:
            raise AnalysisError(str(exc)) from exc
        require(manifest.get("schema") == PACKAGE_SCHEMA,
                f"package schema differs: {manifest_path.parent}")
        _formal_boundary(raw, f"packaged document {manifest_path.parent.name}")
        require(raw.get("schema") in {
            TECHNICAL_SCHEMA, OUTCOME_SCHEMA, CHECKPOINT_SCHEMA,
        }, f"unexpected P02 package document schema: {raw.get('schema')}")
        rows.append({
            "package_dir": manifest_path.parent.resolve(),
            "package_manifest_sha256": common.file_sha256(manifest_path),
            "raw_artifact_sha256": manifest["original"]["sha256"],
            "manifest": manifest,
            "raw": raw,
        })
    return rows


def _validate_source_bindings(common_bindings: Any) -> dict[str, Any]:
    require(
        isinstance(common_bindings, Mapping)
        and set(common_bindings) == REQUIRED_SOURCE_BINDINGS,
        "P02 source-binding name set differs",
    )
    verified: dict[str, Any] = {}
    for name in sorted(common_bindings):
        row = common_bindings[name]
        require(
            isinstance(row, Mapping)
            and set(row) == {"path", "sha256", "size_bytes"},
            f"source binding shape differs: {name}",
        )
        relative = _safe_relative(row.get("path"), f"source binding {name}")
        path = (ROOT / relative).resolve()
        require(
            path.is_relative_to(ROOT)
            and path.is_file()
            and row.get("sha256") == common.file_sha256(path)
            and row.get("size_bytes") == path.stat().st_size,
            f"source binding no longer matches bytes: {name}",
        )
        verified[name] = dict(row)
    require(
        verified["preregistration"]["sha256"] == PREREG_SHA256
        and common.file_sha256(ROOT / "PRECISION-PROBE-P02-PREREGISTRATION.md")
        == PREREG_SHA256,
        "P02 preregistration binding differs from frozen SHA-256",
    )
    return {
        "status": "PASS",
        "binding_names": sorted(verified),
        "canonical_sha256": common.sha256_bytes(common.canonical_bytes(verified)),
        "bindings": verified,
    }


def _validate_raw_bindings(
    raw: Mapping[str, Any], common_bindings: Mapping[str, Any], label: str,
) -> None:
    expected = dict(common_bindings)
    if raw.get("schema") == OUTCOME_SCHEMA:
        expected["case"] = common_bindings["case_e01"]
    require(raw.get("bindings") == expected,
            f"{label} source bindings differ from run manifest")


def _verify_completion_inventory(
    run_dir: Path, completion_path: Path, completion: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = completion.get("artifact_inventory")
    require(isinstance(inventory, list) and inventory,
            "P02 completion inventory is empty")
    actual = {
        path.relative_to(run_dir).as_posix(): path
        for path in run_dir.rglob("*")
        if path.is_file() and path != completion_path
    }
    mapped: dict[str, Mapping[str, Any]] = {}
    for row in inventory:
        require(
            isinstance(row, Mapping)
            and set(row) == {"path", "sha256", "size_bytes"}
            and isinstance(row.get("path"), str),
            "P02 completion inventory row differs",
        )
        declared = str(row["path"])
        matches = [rel for rel in actual
                   if declared == rel or declared.endswith("/" + rel)]
        require(len(matches) == 1 and matches[0] not in mapped,
                f"completion inventory path cannot be resolved: {declared}")
        mapped[matches[0]] = row
    require(set(mapped) == set(actual),
            "P02 completion inventory is not exhaustive")
    for relative, path in actual.items():
        row = mapped[relative]
        require(
            row.get("sha256") == common.file_sha256(path)
            and row.get("size_bytes") == path.stat().st_size,
            f"completion inventory bytes differ: {relative}",
        )
    return {
        "status": "PASS",
        "file_count": len(actual),
        "canonical_sha256": common.sha256_bytes(common.canonical_bytes([
            {"path": key, "sha256": mapped[key]["sha256"],
             "size_bytes": mapped[key]["size_bytes"]}
            for key in sorted(mapped)
        ])),
    }


def _terminal_documents(
    run_dir: Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    completion_path = _single(sorted(run_dir.glob(
        "precision-probe-p02-completion_*.json")), "runner completion")
    manifest_path = _single(sorted(run_dir.glob(
        "precision-probe-p02-run-manifest_*.json")), "run manifest")
    completion = common.load_object(completion_path)
    manifest = common.load_object(manifest_path)
    require(completion.get("schema") == COMPLETION_SCHEMA,
            "P02 completion schema differs")
    require(manifest.get("schema") == RUN_MANIFEST_SCHEMA,
            "P02 manifest schema differs")
    _formal_boundary(completion, "runner completion")
    _formal_boundary(manifest, "run manifest")
    require(
        completion.get("status") == manifest.get("status")
        and completion.get("status") in {"COMPLETE", "PARTIAL", "ERROR"},
        "P02 terminal status differs",
    )
    require(
        completion.get("model") == manifest.get("model") == MODEL_ID
        and completion.get("revision") == manifest.get("revision") == REVISION,
        "P02 terminal subject differs",
    )
    bound = _verify_file_binding(
        run_dir, completion.get("run_manifest"), "completion run manifest")
    require(bound == manifest_path.resolve(),
            "completion bound a different run manifest")
    return completion_path, completion, manifest_path, manifest


def _validate_outer_receipt(run_dir: Path) -> dict[str, Any]:
    """Verify the unique outer pod tree if a receipt for this run is present."""
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(run_dir.parent.glob("precision-probe-p02-receipt_*.json")):
        document = common.load_object(path)
        artifacts = document.get("artifacts")
        if isinstance(artifacts, list) and any(
            isinstance(row, Mapping)
            and f"/{run_dir.name}/" in f"/{row.get('path', '')}/"
            for row in artifacts
        ):
            candidates.append((path, document))
    if not candidates:
        return {"status": "NOT_AVAILABLE", "receipt": None}
    require(len(candidates) == 1,
            "P02 run is covered by more than one outer pod receipt")
    path, document = candidates[0]
    required_fields = {
        "schema", "protocol_id", "preregistration_sha256",
        "completed_at_utc", "expected_commit", "observed_commit", "model",
        "revision", "runner_exit", "runner_completion_status",
        "runner_completion_path", "packaging_exit", "lossless_package_count",
        "outcome_package_count", "recovery_raw_count", "terminal_status",
        "formal_v12_decision_eligible", "v12_reentry_authorized",
        "component_reuse_does_not_inherit_v12_eligibility", "provider",
        "artifacts",
    }
    require(set(document) == required_fields,
            "outer pod receipt field set differs")
    require(
        document.get("schema") == OUTER_RECEIPT_SCHEMA
        and document.get("protocol_id") == PROTOCOL_ID
        and document.get("preregistration_sha256") == PREREG_SHA256
        and document.get("model") == MODEL_ID
        and document.get("revision") == REVISION
        and document.get("formal_v12_decision_eligible") is False
        and document.get("v12_reentry_authorized") is False
        and document.get("component_reuse_does_not_inherit_v12_eligibility")
        is True
        and document.get("expected_commit") == document.get("observed_commit")
        and isinstance(document.get("expected_commit"), str)
        and re.fullmatch(r"[0-9a-f]{40}", document["expected_commit"]),
        "outer pod receipt identity/boundary differs",
    )
    artifacts = document["artifacts"]
    require(isinstance(artifacts, list) and artifacts,
            "outer pod artifact inventory is empty")
    resolved: dict[str, Path] = {}
    for index, row in enumerate(artifacts):
        require(
            isinstance(row, Mapping)
            and set(row) == {"path", "sha256", "size_bytes"},
            f"outer artifact row {index} differs",
        )
        relative = _safe_relative(row.get("path"), f"outer artifact {index}")
        declared = relative.as_posix()
        require(declared.startswith("results/precision_probe_p02/"),
                f"outer artifact {index} escaped the P02 tree")
        direct = (ROOT / relative).resolve()
        matches = [direct] if direct.is_file() else [
            candidate.resolve()
            for candidate in run_dir.parent.rglob(relative.name)
            if candidate.is_file()
            and (declared.endswith("/" + candidate.relative_to(
                run_dir.parent).as_posix())
                 or candidate == run_dir.parent / relative.name)
        ]
        matches = list(dict.fromkeys(matches))
        require(len(matches) == 1 and declared not in resolved,
                f"outer artifact {declared} cannot be resolved uniquely")
        local = matches[0]
        require(
            not local.is_symlink()
            and row.get("sha256") == common.file_sha256(local)
            and row.get("size_bytes") == local.stat().st_size,
            f"outer artifact bytes differ: {declared}",
        )
        resolved[declared] = local
    run_files = {item.resolve() for item in run_dir.rglob("*") if item.is_file()}
    require(run_files <= set(resolved.values()),
            "outer receipt does not exhaustively bind the runner tree")
    runtime_paths = [local for declared, local in resolved.items()
                     if "precision-probe-p02-runtime_" in Path(declared).name]
    require(len(runtime_paths) == 1,
            "outer receipt does not bind exactly one runtime attestation")
    runtime = common.load_object(runtime_paths[0])
    require(
        runtime.get("schema") == RUNTIME_ATTESTATION_SCHEMA
        and runtime.get("protocol_id") == PROTOCOL_ID
        and runtime.get("preregistration_sha256") == PREREG_SHA256
        and runtime.get("expected_commit") == document["expected_commit"]
        and runtime.get("observed_commit") == document["observed_commit"]
        and runtime.get("model") == MODEL_ID
        and runtime.get("revision") == REVISION
        and runtime.get("python") == "3.12.11"
        and runtime.get("torch_cuda") == "13.0",
        "outer runtime attestation differs",
    )
    provider = document.get("provider")
    require(isinstance(provider, Mapping), "outer provider record is absent")
    rate = provider.get("hourly_cost_usd")
    cap = provider.get("scientific_work_cap_seconds")
    require(
        isinstance(rate, (int, float)) and not isinstance(rate, bool)
        and math.isfinite(float(rate)) and float(rate) > 0
        and isinstance(cap, int) and not isinstance(cap, bool)
        and cap == min(9000, math.floor(3.90 * 3600 / float(rate)))
        and provider.get("provider_wall_cap_seconds") == cap
        and provider.get("scientific_budget_usd") == 3.90
        and provider.get("absolute_envelope_usd") == 4.0,
        "outer provider cap differs",
    )
    package_count = sum(
        path.name == "manifest.json"
        and path.parent.name.endswith(".lossless-package")
        for path in resolved.values()
    )
    outcome_count = sum(
        path.name == "manifest.json"
        and path.parent.name.startswith("precision-probe-p02-outcome-")
        for path in resolved.values()
    )
    require(
        document.get("lossless_package_count") == package_count
        and document.get("outcome_package_count") == outcome_count
        and document.get("terminal_status") in {"PASS", "ERROR"},
        "outer receipt package counts/status differ",
    )
    completion_paths = [
        local for declared, local in resolved.items()
        if Path(declared).name.startswith("precision-probe-p02-completion_")
    ]
    require(len(completion_paths) == 1,
            "outer receipt does not bind exactly one runner completion")
    outer_completion = common.load_object(completion_paths[0])
    declared_completion = document.get("runner_completion_path")
    require(
        isinstance(declared_completion, str)
        and resolved.get(declared_completion) == completion_paths[0]
        and document.get("runner_completion_status")
        == outer_completion.get("status"),
        "outer receipt runner-completion binding differs",
    )
    return {
        "status": "PASS",
        "receipt": {
            "path": path.relative_to(run_dir.parent).as_posix(),
            "sha256": common.file_sha256(path),
            "terminal_status": document["terminal_status"],
            "expected_commit": document["expected_commit"],
            "artifact_count": len(artifacts),
        },
    }


def _technical_gate(
    run_dir: Path,
    documents: Sequence[Mapping[str, Any]],
    common_bindings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    rows = [row for row in documents
            if row["raw"].get("schema") == TECHNICAL_SCHEMA
            and row["raw"].get("status") != "CHECKPOINT"]
    require(len(rows) == 2 and all(row["raw"].get("status") == "PASS"
                                   for row in rows),
            "P02 requires exactly two passing final technical packages")
    by_regime: dict[str, Mapping[str, Any]] = {}
    summary: dict[str, Any] = {}
    for row in rows:
        raw = row["raw"]
        regime = raw.get("regime")
        require(regime in REGIMES and regime not in by_regime,
                "P02 technical regime coverage differs")
        label = f"{regime} technical"
        _formal_boundary(raw, label)
        _validate_raw_bindings(raw, common_bindings, label)
        require(raw.get("model") == MODEL_ID and raw.get("revision") == REVISION,
                f"{label} subject differs")
        subject = raw.get("subject_attestation")
        runtime = raw.get("runtime_fingerprint")
        require(
            isinstance(subject, Mapping)
            and isinstance(subject.get("bindings"), Mapping)
            and subject.get("runtime_fingerprint") == runtime,
            f"{label} subject/runtime binding differs",
        )
        bindings = subject["bindings"]
        require(
            isinstance(runtime, Mapping)
            and runtime.get("protocol_id") == "precision-probe-p01"
            and runtime.get("regime") == regime
            and runtime.get("model_id") == MODEL_ID
            and runtime.get("requested_revision") == REVISION
            and runtime.get("resolved_snapshot") == REVISION
            and runtime.get("architecture") == "Qwen3MoeForCausalLM"
            and runtime.get("attention_backend") == "eager"
            and runtime.get("kv_dtype") == "torch.bfloat16"
            and runtime.get("eos_ids") == [151643, 151645]
            and runtime.get("dependency_versions") == EXPECTED_DEPENDENCIES
            and bindings.get("protocol_id") == "precision-probe-p01"
            and bindings.get("regime") == regime,
            f"{label} runtime fingerprint differs",
        )
        g0 = bindings.get("g0")
        weights = bindings.get("weights")
        kv = bindings.get("kv")
        host = bindings.get("host")
        loading = bindings.get("loading_info")
        require(
            isinstance(g0, Mapping) and g0.get("passes") is True
            and g0.get("topology", {}).get("checkpoint_key_count") == 18_867
            and g0.get("topology", {}).get("eligible_linear_count") == 18_672
            and g0.get("topology", {}).get("eligible_logical_elements")
            == 29_909_581_824,
            f"{label} G0 topology differs",
        )
        require(
            isinstance(kv, Mapping) and kv.get("observed_real_forward") is True
            and kv.get("layers") == 48
            and kv.get("dtype") == "torch.bfloat16"
            and kv.get("expected_shape", [None, None, None, None])[:2] == [1, 4]
            and kv.get("expected_shape", [None, None, None, None])[-1:] == [128],
            f"{label} real-forward KV gate differs",
        )
        require(
            isinstance(host, Mapping)
            and host.get("gpu_name") == "NVIDIA A100 80GB PCIe"
            and isinstance(host.get("gpu_uuid"), str) and host["gpu_uuid"],
            f"{label} host gate differs",
        )
        require(
            _version_tuple(host.get("driver_version"), f"{label} NVIDIA driver")
            >= (580, 65, 6)
            and host.get("torch_cuda_version") == "13.0"
            and host.get("device_count") == 1
            and isinstance(host.get("memory_total_mib"), int)
            and host["memory_total_mib"] >= 80_000,
            f"{label} host driver/CUDA/memory admission differs",
        )
        require(isinstance(loading, Mapping) and loading
                and all(value == [] for value in loading.values()),
                f"{label} loading-info gate differs")
        require(isinstance(weights, Mapping), f"{label} weight gate is absent")
        if regime == "nf4":
            sentinels = weights.get("sentinels")
            require(
                weights.get("linear4bit_modules") == 18_672
                and weights.get("expert_linear4bit_modules") == 18_432
                and weights.get("expert_logical_coverage") == {
                    "observed": 28_991_029_248,
                    "expected": 28_991_029_248,
                }
                and weights.get("total_logical_quantized_coverage") == {
                    "observed": 29_909_581_824,
                    "expected": 29_909_581_824,
                    "checkpoint_total": 30_532_122_624,
                }
                and weights.get("quant_type") == "nf4"
                and weights.get("double_quantization") is True
                and weights.get("compute_dtype") == "torch.bfloat16"
                and weights.get("ordinary_linears") == ["lm_head"]
                and isinstance(sentinels, list) and len(sentinels) == 9
                and all(float(item.get("max_abs_error", 0)) > 0
                        and float(item.get("mean_abs_error", 0)) > 0
                        for item in sentinels),
                "NF4 full-coverage/sentinel gate differs",
            )
        else:
            require(
                weights.get("regime") == "bf16"
                and weights.get("logical_parameter_elements") == 30_532_122_624
                and weights.get("floating_parameter_dtype") == "torch.bfloat16"
                and weights.get("quantization_modules") == [],
                "bfloat16 full-precision weight gate differs",
            )
        identity = raw.get("generated_forced_identity")
        deterministic = raw.get("deterministic_repeats")
        replacement = raw.get("fresh_self_replacement")
        require(
            isinstance(identity, Mapping) and identity.get("status") == "PASS"
            and isinstance(deterministic, Mapping)
            and set(deterministic) == {"correct_history_N", "fresh_destination"}
            and all(item.get("status") == "PASS"
                    for item in deterministic.values())
            and isinstance(replacement, Mapping)
            and replacement.get("status") == "PASS"
            and len(replacement.get("regions", [])) == 9
            and all(item.get("status") == "PASS"
                    for item in replacement["regions"]),
            f"{label} identity/surgery gate differs",
        )
        by_regime[str(regime)] = row
        summary[str(regime)] = {
            "package": {
                "path": row["package_dir"].relative_to(run_dir).as_posix(),
                "manifest_sha256": row["package_manifest_sha256"],
                "raw_artifact_sha256": row["raw_artifact_sha256"],
            },
            "runtime_fingerprint": dict(runtime),
            "host": dict(host),
            "weights": dict(weights),
            "kv": dict(kv),
            "status": "PASS",
        }
    require(set(by_regime) == set(REGIMES), "P02 technical regime set differs")
    return {"status": "PASS", "regimes": summary}, by_regime


def _matched_runtime_gate(
    technical: Mapping[str, Mapping[str, Any]], declared: Any,
) -> dict[str, Any]:
    require(isinstance(declared, Mapping), "matched-runtime gate is absent")
    views: dict[str, dict[str, Any]] = {}
    for regime in REGIMES:
        raw = technical[regime]["raw"]
        runtime = raw["runtime_fingerprint"]
        bindings = raw["subject_attestation"]["bindings"]
        missing = [f"runtime.{field}" for field in MATCHED_RUNTIME_FIELDS
                   if field not in runtime] + [
            f"bindings.{field}" for field in MATCHED_SUBJECT_FIELDS
            if field not in bindings
        ]
        require(not missing,
                f"{regime} matched-runtime view is incomplete: {missing}")
        views[regime] = {
            "runtime": {field: runtime[field] for field in MATCHED_RUNTIME_FIELDS},
            "bindings": {field: bindings[field]
                         for field in MATCHED_SUBJECT_FIELDS},
        }
    differences = common.recursive_differences(views["nf4"], views["bf16"])
    require(not differences,
            f"independently reconstructed matched runtimes differ: {differences}")
    hashes = {
        regime: common.sha256_bytes(common.canonical_bytes(views[regime]) + b"\n")
        for regime in REGIMES
    }
    expected = {
        "status": "PASS",
        "compared_runtime_fields": list(MATCHED_RUNTIME_FIELDS),
        "compared_subject_binding_fields": list(MATCHED_SUBJECT_FIELDS),
        "differing_fields": [],
        "nf4_matched_view_sha256": hashes["nf4"],
        "bf16_matched_view_sha256": hashes["bf16"],
    }
    require(dict(declared) == expected,
            "declared matched-runtime gate differs from technical raws")
    return {
        "status": "PASS",
        "same_host": True,
        "view_sha256": hashes["nf4"],
        "differing_json_pointers": [],
        "runtime_fields": list(MATCHED_RUNTIME_FIELDS),
        "subject_binding_fields": list(MATCHED_SUBJECT_FIELDS),
    }


def _generation_record(score: Mapping[str, Any]) -> dict[str, Any]:
    generation = score["generation"]
    record = {
        "decoded_content": generation["decoded_content"],
        "content_ids": list(generation["content_ids"]),
        "stop_reason": generation.get("stop_reason"),
        "cap_hit": generation.get("cap_hit"),
    }
    record["canonical_sha256"] = common.sha256_bytes(
        common.canonical_bytes(record))
    return record


def _generation_pattern(
    fresh: Mapping[str, Any],
    n_cells: Mapping[str, Mapping[str, Any]],
    probe: str,
) -> dict[str, Any]:
    cells = {
        "FF": _generation_record(fresh[probe]),
        **{cell: _generation_record(n_cells[cell][probe])
           for cell in ("FC", "FW", "CC", "WW")},
    }
    fresh_signature = (cells["FF"]["decoded_content"],
                       cells["FF"]["content_ids"])
    changes = {
        cell: (cells[cell]["decoded_content"], cells[cell]["content_ids"])
        != fresh_signature
        for cell in ("FC", "FW", "CC", "WW")
    }
    return {
        "probe": probe,
        "literal_generation_tuple": [cells[cell] for cell in GENERATION_CELLS],
        "literal_decoded_content_tuple": [
            cells[cell]["decoded_content"] for cell in GENERATION_CELLS
        ],
        "generation_hash_tuple": [
            cells[cell]["canonical_sha256"] for cell in GENERATION_CELLS
        ],
        "cells": cells,
        "change_from_fresh": changes,
        "four_bit_change_vector": [
            changes[cell] for cell in ("FC", "FW", "CC", "WW")
        ],
    }


def _damage(
    phase_scores: Mapping[str, Mapping[str, Any]], probe: str,
) -> dict[str, Any]:
    oracle = phase_scores[f"A_C_{probe}"]
    fresh = phase_scores[f"FF_{probe}"]
    margin = oracle["margin"] - fresh["margin"]
    return {
        "margin": margin,
        "correct_target_logprob": (
            oracle["correct"]["mean_logprob"]
            - fresh["correct"]["mean_logprob"]
        ),
        "counterfactual_target_logprob": (
            oracle["counterfactual"]["mean_logprob"]
            - fresh["counterfactual"]["mean_logprob"]
        ),
        "oracle_generation": oracle["generation"],
        "fresh_generation": fresh["generation"],
        "per_token": common.token_contrast(
            oracle, fresh, recorded_margin_difference=margin),
    }


def analyze_outcome(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one complete P02 raw outcome and compute frozen estimands."""
    require(raw.get("schema") == OUTCOME_SCHEMA
            and raw.get("protocol_id") == PROTOCOL_ID
            and raw.get("status") == "COMPLETE",
            "complete P02 outcome envelope differs")
    _formal_boundary(raw, "complete outcome")
    regime = raw.get("regime")
    repeat = raw.get("repeat_index")
    require(regime in REGIMES and raw.get("case_id") == CASE_ID
            and repeat in {1, 2}, "P02 outcome identity differs")
    require(raw.get("model") == MODEL_ID and raw.get("revision") == REVISION,
            "P02 outcome subject differs")
    phase = raw.get("phase_a")
    treatment = raw.get("treatment")
    require(
        isinstance(phase, Mapping)
        and phase.get("schema") == PHASE_A_SCHEMA
        and phase.get("design_id") == DESIGN_ID
        and phase.get("case_id") == CASE_ID
        and isinstance(phase.get("executions"), Mapping)
        and set(phase["executions"]) == {"C_N", "W_N", "F"}
        and phase.get("treatment_scores_present") is False,
        "P02 full Phase-A schema differs",
    )
    phase_raw = phase.get("scores")
    phase_keys = {
        f"{prefix}_{probe}"
        for prefix in ("A_C", "A_W", "FF")
        for probe in ("focal", "nonfocal")
    }
    require(isinstance(phase_raw, Mapping) and set(phase_raw) == phase_keys,
            "P02 Phase-A score grid differs")
    phase_scores = {
        key: common.score_summary(value, f"phase_a.{key}")
        for key, value in phase_raw.items()
    }
    require(
        isinstance(treatment, Mapping)
        and treatment.get("schema") == TREATMENT_SCHEMA
        and treatment.get("protocol_id") == PROTOCOL_ID
        and treatment.get("design_id") == DESIGN_ID
        and treatment.get("case_id") == CASE_ID
        and treatment.get("repeat_index") == repeat,
        "P02 targeted-treatment identity differs",
    )
    foundation = treatment.get("foundation")
    require(
        isinstance(foundation, Mapping)
        and foundation.get("schema") == FOUNDATION_SCHEMA
        and foundation.get("protocol_id") == PROTOCOL_ID
        and foundation.get("design_id") == DESIGN_ID
        and foundation.get("case_id") == CASE_ID
        and foundation.get("repeat_index") == repeat
        and foundation.get("source_plan_order") == ["C_N", "W_N", "C_P", "W_P"]
        and foundation.get("boundary_regions") == [R2],
        "P02 treatment foundation differs",
    )
    fresh = common.score_pair(treatment.get("fresh_scores"), "treatment.fresh")
    arms = treatment.get("arms")
    expected_count = 8 if repeat == 1 else 7
    require(
        isinstance(arms, list) and len(arms) == expected_count
        and treatment.get("arm_count") == expected_count
        and treatment.get("executed_graft_count") == 6
        and treatment.get("zero_increment_anchor_count") == 1
        and treatment.get("placebo_attempt_count") == (1 if repeat == 1 else 0)
        and treatment.get("phase_a_scores_present") is False,
        "P02 targeted-treatment counts differ",
    )
    ff = arms[0]
    foundation_ff = foundation.get("ff_anchor")
    require(
        isinstance(ff, Mapping)
        and ff.get("arm_kind") == "fresh_baseline"
        and ff.get("execution_kind") == "zero_increment_reuse"
        and (ff.get("schedule"), ff.get("region"), ff.get("cell"))
        == ("N", R2, "FF")
        and ff.get("key_source") == "F"
        and ff.get("value_source") == "F"
        and ff.get("shared_fresh_baseline") is True
        and isinstance(foundation_ff, Mapping)
        and foundation_ff == ff
        and foundation.get("fresh_scores") == treatment.get("fresh_scores")
        and ff.get("scores") == treatment.get("fresh_scores"),
        "P02 serialized FF zero-increment identity differs",
    )
    # Parse after the serialized-equality assertion so FF cannot merely carry
    # an independently generated but numerically similar score record.
    require(common.score_pair(ff.get("scores"), "treatment.FF") == fresh,
            "P02 FF parsed scores differ from direct fresh")
    grafts: dict[tuple[str, str, str], dict[str, Any]] = {}
    selectors: list[tuple[Any, Any, Any]] = []
    for index, expected in enumerate(GRAFT_SELECTORS, start=1):
        row = arms[index]
        require(isinstance(row, Mapping), f"P02 graft {index} is absent")
        selector = (row.get("schedule"), row.get("region"), row.get("cell"))
        selectors.append(selector)
        require(selector == expected and row.get("arm_kind") == "primary",
                f"P02 graft order differs at index {index}: {selector}")
        grafts[expected] = common.score_pair(
            row.get("scores"), f"treatment.arm{index}.{expected}")
    require(tuple(selectors) == GRAFT_SELECTORS and len(grafts) == 6,
            "P02 six-graft grid differs")
    placebo: dict[str, Any] | None = None
    if repeat == 1:
        row = arms[7]
        require(
            isinstance(row, Mapping)
            and row.get("arm_kind") == "placebo_control"
            and (row.get("schedule"), row.get("region"), row.get("cell"))
            == ("N", R2, "V_PLACEBO")
            and row.get("control_status") in {"AVAILABLE", "PLACEBO_UNAVAILABLE"}
            and isinstance(row.get("diagnostics"), Mapping)
            and row["diagnostics"].get("status") == row.get("control_status"),
            "P02 repeat-1 final placebo attempt differs",
        )
        if row["control_status"] == "AVAILABLE":
            scores = common.score_pair(row.get("scores"), "treatment.placebo")
            placebo = {
                "status": "AVAILABLE",
                "scores": scores,
                "movement_from_fresh": common.movement_from_fresh(scores, fresh),
                "diagnostics": dict(row["diagnostics"]),
            }
        else:
            require("scores" not in row,
                    "unavailable P02 placebo unexpectedly contains scores")
            placebo = {
                "status": "PLACEBO_UNAVAILABLE",
                "scores": None,
                "movement_from_fresh": None,
                "diagnostics": dict(row["diagnostics"]),
            }
    available = int(placebo is not None and placebo["status"] == "AVAILABLE")
    require(treatment.get("available_placebo_control_count") == available,
            "P02 available-placebo count differs")

    n_cells = {cell: grafts[("N", R2, cell)]
               for cell in ("FC", "FW", "CC", "WW")}
    p_cells = {cell: grafts[("P", R2, cell)] for cell in ("FC", "FW")}
    value_n = common.contrast(
        n_cells, correct_cell="FC", wrong_cell="FW", fresh=fresh)
    full_n = common.contrast(
        n_cells, correct_cell="CC", wrong_cell="WW", fresh=fresh)
    value_p = common.contrast(
        p_cells, correct_cell="FC", wrong_cell="FW", fresh=fresh)
    damage = {probe: _damage(phase_scores, probe)
              for probe in ("focal", "nonfocal")}
    return {
        "regime": regime,
        "case_id": CASE_ID,
        "repeat_index": repeat,
        "E1_gross_compaction_damage": {
            **damage,
            "signed_focal_minus_nonfocal_margin_damage": (
                damage["focal"]["margin"] - damage["nonfocal"]["margin"]
            ),
        },
        "E2_literal_generated_behavior": {
            probe: _generation_pattern(fresh, n_cells, probe)
            for probe in ("focal", "nonfocal")
        },
        "E3_graft_contrasts": {
            "N_R2_D_value": value_n,
            "N_R2_D_full_KV": full_n,
        },
        "secondary_controls": {
            "P_R2_D_value": value_p,
            "N_minus_P_value_shift": {
                "focal": value_n["D_focal"] - value_p["D_focal"],
                "nonfocal": value_n["D_nonfocal"] - value_p["D_nonfocal"],
            },
            "FF_fresh_serialized_identity": True,
            "placebo": placebo if repeat == 1 else {
                "status": "NOT_ATTEMPTED_BY_PROTOCOL_REPEAT2",
                "scores": None,
            },
        },
    }


def _scalar_estimands(analysis: Mapping[str, Any]) -> dict[str, float]:
    damage = analysis["E1_gross_compaction_damage"]
    e3 = analysis["E3_graft_contrasts"]
    secondary = analysis["secondary_controls"]
    return {
        "E1_focal_margin_damage": float(damage["focal"]["margin"]),
        "E1_nonfocal_margin_damage": float(damage["nonfocal"]["margin"]),
        "E1_focal_minus_nonfocal": float(
            damage["signed_focal_minus_nonfocal_margin_damage"]),
        "E3_N_R2_D_value_focal": float(e3["N_R2_D_value"]["D_focal"]),
        "E3_N_R2_D_value_nonfocal": float(e3["N_R2_D_value"]["D_nonfocal"]),
        "E3_N_R2_D_full_KV_focal": float(e3["N_R2_D_full_KV"]["D_focal"]),
        "E3_N_R2_D_full_KV_nonfocal": float(
            e3["N_R2_D_full_KV"]["D_nonfocal"]),
        "secondary_P_R2_D_value_focal": float(
            secondary["P_R2_D_value"]["D_focal"]),
        "secondary_P_R2_D_value_nonfocal": float(
            secondary["P_R2_D_value"]["D_nonfocal"]),
        "secondary_N_minus_P_shift_focal": float(
            secondary["N_minus_P_value_shift"]["focal"]),
        "secondary_N_minus_P_shift_nonfocal": float(
            secondary["N_minus_P_value_shift"]["nonfocal"]),
    }


def validate_repeat2_layout(
    statuses: Mapping[tuple[str, int], str],
    *,
    declared_rider_status: str,
    initial_repeat2_authorized: bool,
    bf16_repeat2_authorized: bool,
) -> str:
    """Enforce the frozen repeat rider without making repeat 2 primary."""
    require(declared_rider_status in RIDER_STATUSES,
            "P02 repeat-2 rider status is outside the frozen vocabulary")
    for regime in REGIMES:
        require(statuses.get((regime, 1)) == "COMPLETE",
                f"matched repeat 1 is absent/incomplete for {regime}")
    require(set(statuses) <= {(regime, repeat)
                              for regime in REGIMES for repeat in (1, 2)},
            "P02 outcome receipt grid contains an unexpected key")
    nf4_complete = statuses.get(("nf4", 2)) == "COMPLETE"
    bf16_complete = statuses.get(("bf16", 2)) == "COMPLETE"
    require(not bf16_complete or nf4_complete,
            "bfloat16 repeat 2 completed without its NF4 rider")
    require(initial_repeat2_authorized or not (nf4_complete or bf16_complete),
            "repeat 2 ran without initial rider authorization")
    require(not bf16_repeat2_authorized or
            (initial_repeat2_authorized and nf4_complete),
            "bfloat16 repeat-2 authorization lacks its frozen prerequisites")
    expected = (
        "NOT_AUTHORIZED" if not initial_repeat2_authorized
        else "MATCHED_COMPLETE" if nf4_complete and bf16_complete
        else "NF4_ONLY_TIMING_STOP" if nf4_complete
        and not bf16_repeat2_authorized
        else "AUTHORIZED_INCOMPLETE"
    )
    require(declared_rider_status == expected,
            f"declared repeat-2 rider status differs: {declared_rider_status} != {expected}")
    return expected


def _verify_derived(
    raw: Mapping[str, Any], raw_binding: Mapping[str, Any],
    compact_path: Path, render_path: Path,
) -> None:
    document = dict(raw)
    document["raw_package_binding"] = dict(raw_binding)
    require(common.load_object(compact_path) == p02_runner.compact_outcome(document),
            "P02 compact content differs from reconstructed raw")
    require(render_path.read_text(encoding="utf-8")
            == p02_runner.render_outcome_ledger(document),
            "P02 render content differs from reconstructed raw")


def _verify_technical_render(
    raw: Mapping[str, Any], raw_binding: Mapping[str, Any], path: Path,
) -> None:
    document = dict(raw)
    document["raw_package_binding"] = dict(raw_binding)
    expected = p01_runner.render_technical_ledger(document).replace(
        "P01 technical generation ledger", "P02 technical generation ledger"
    ).replace("precision-probe-p01", PROTOCOL_ID)
    require(path.read_text(encoding="utf-8") == expected,
            "P02 technical render differs from reconstructed raw")


def _verify_checkpoint_chain(
    run_dir: Path,
    receipt: Mapping[str, Any],
    final_row: Mapping[str, Any],
    package_by_dir: Mapping[Path, Mapping[str, Any]],
) -> None:
    raw = final_row["raw"]
    repeat = int(raw["repeat_index"])
    chain = receipt.get("checkpoint_chain")
    expected_stages = ["phase_a", "treatment_foundation"] + [
        f"treatment_arm_{index:02d}"
        for index in range(1, 8 if repeat == 1 else 7)
    ]
    require(isinstance(chain, list) and len(chain) == len(expected_stages)
            and raw.get("durable_checkpoints") == chain,
            "P02 checkpoint-chain coverage differs")
    phase_binding = receipt.get("phase_package")
    require(chain[0].get("raw_package") == phase_binding,
            "P02 phase-package receipt differs from checkpoint chain")
    foundation_binding = chain[1].get("raw_package")
    previous: Mapping[str, Any] | None = None
    for index, (row, stage) in enumerate(zip(chain, expected_stages), start=1):
        require(
            isinstance(row, Mapping)
            and row.get("sequence") == index
            and row.get("stage") == stage
            and row.get("recovery_raw_path") is None
            and row.get("persistence_error") is None,
            f"P02 checkpoint-chain row {index} differs",
        )
        package_path = _resolve_run_path(
            run_dir, row.get("raw_package", {}).get("path"),
            f"checkpoint {index} package")
        package_row = package_by_dir.get(package_path)
        require(package_row is not None,
                f"checkpoint {index} package was not reconstructed")
        _verify_package_binding(
            run_dir, row.get("raw_package"), package_row,
            f"checkpoint {index}")
        checkpoint = package_row["raw"]
        require(
            checkpoint.get("schema") == CHECKPOINT_SCHEMA
            and checkpoint.get("regime") == raw.get("regime")
            and checkpoint.get("case_id") == CASE_ID
            and checkpoint.get("repeat_index") == repeat
            and checkpoint.get("stage") == stage
            and checkpoint.get("previous_checkpoint") == previous,
            f"checkpoint {index} raw envelope/previous binding differs",
        )
        if index == 1:
            require(checkpoint.get("payload_name") == "phase_a"
                    and checkpoint.get("payload") == raw.get("phase_a")
                    and checkpoint.get("phase_package") is None
                    and checkpoint.get("foundation_package") is None,
                    "Phase-A checkpoint payload differs from final raw")
        elif index == 2:
            require(checkpoint.get("payload_name") == "foundation"
                    and checkpoint.get("payload")
                    == raw.get("treatment", {}).get("foundation")
                    and checkpoint.get("phase_package") == phase_binding
                    and checkpoint.get("foundation_package") is None,
                    "foundation checkpoint payload differs from final raw")
        else:
            arm_index = index - 2
            require(row.get("arm_index") == arm_index
                    and checkpoint.get("payload_name") == "arm"
                    and checkpoint.get("payload")
                    == raw.get("treatment", {}).get("arms", [])[arm_index]
                    and checkpoint.get("phase_package") == phase_binding
                    and checkpoint.get("foundation_package") == foundation_binding,
                    f"arm checkpoint {arm_index} payload differs from final raw")
        previous = row


def _decision_common(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    _formal_boundary(document, f"decision {name}")
    require(
        document.get("schema") == DECISION_SCHEMA
        and document.get("decision_name") == name
        and isinstance(document.get("authorized"), bool)
        and isinstance(document.get("inputs"), Mapping)
        and document.get("outcome_information_surface")
        == ["completion_status", "durable_elapsed_seconds"]
        and document.get("technical_information_surface")
        == ["status", "durable_elapsed_seconds"]
        and document.get("score_generation_path_or_payload_accessible") is False,
        f"P02 decision contract differs: {name}",
    )
    return document["inputs"]


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value))
            and (float(value) > 0 if positive else float(value) >= 0),
            f"{label} scalar differs")
    return float(value)


def _close(left: Any, right: float, label: str, tolerance: float = 1e-9) -> None:
    require(isinstance(left, (int, float)) and not isinstance(left, bool)
            and math.isclose(float(left), right, rel_tol=0, abs_tol=tolerance),
            f"{label} derived scalar differs")


def _validate_decisions(
    run_dir: Path,
    completion: Mapping[str, Any],
    manifest: Mapping[str, Any],
    technical_receipts: Mapping[str, Mapping[str, Any]],
    outcome_receipts: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    completion_bindings = completion.get("continuation_decisions")
    manifest_bindings = manifest.get("continuation_decisions")
    require(isinstance(completion_bindings, Mapping)
            and set(completion_bindings) == set(DECISION_NAMES)
            and completion_bindings == manifest_bindings,
            "P02 continuation-decision binding set differs")
    documents: dict[str, Mapping[str, Any]] = {}
    for name in DECISION_NAMES:
        path = _verify_file_binding(
            run_dir, completion_bindings[name], f"decision {name}")
        document = common.load_object(path)
        _decision_common(document, name)
        documents[name] = document

    cap = completion.get("scientific_cap_seconds")
    require(isinstance(cap, int) and not isinstance(cap, bool) and cap > 300,
            "P02 scientific cap differs")
    deadline = cap - 300.0
    nf4_tech = technical_receipts["nf4"]
    nf4_r1 = outcome_receipts[("nf4", 1)]
    L = _number(nf4_tech.get("durable_elapsed_seconds"), "NF4 L")
    t = _number(nf4_r1.get("durable_elapsed_seconds"), "NF4 t")
    lhat = max(L, 900.0)

    initial = documents["initial_schedule"]
    inputs = initial["inputs"]
    e = _number(inputs.get("provider_elapsed_seconds"), "initial elapsed")
    require(inputs.get("nf4_technical_status") == "PASS"
            and inputs.get("nf4_repeat1_completion_status") == "COMPLETE",
            "initial decision status inputs differ")
    _close(inputs.get("nf4_technical_durable_elapsed_seconds"), L,
           "initial L", 1e-12)
    _close(inputs.get("nf4_repeat1_durable_elapsed_seconds"), t,
           "initial t", 1e-12)
    _close(inputs.get("technical_forecast_seconds"), lhat, "initial Lhat")
    require(inputs.get("scientific_cap_seconds") == cap,
            "initial decision cap differs")
    _close(inputs.get("forecast_deadline_seconds"), deadline,
           "initial deadline")
    matched_forecast = e + 1.25 * (lhat + t)
    rider_forecast = e + 1.25 * (t + lhat + 2.0 * t)
    _close(inputs.get("matched_repeat1_forecast_seconds"), matched_forecast,
           "initial matched forecast")
    _close(inputs.get("both_repeat2_forecast_seconds"), rider_forecast,
           "initial rider forecast")
    matched = matched_forecast <= deadline
    riders = matched and rider_forecast <= deadline
    require(
        initial.get("authorized") is matched
        and initial.get("matched_repeat1_authorized") is matched
        and initial.get("both_repeat2_authorized") is riders
        and initial.get("decision") == (
            "MATCHED_R1_AND_BOTH_R2" if riders else
            "MATCHED_R1_ONLY" if matched else "STOP_NF4_ONLY")
        and initial.get("formula") == (
            "matched: e+1.25*(max(L,900)+t)<=C-300; "
            "riders: e+1.25*(t+max(L,900)+2*t)<=C-300"
        ),
        "initial decision differs from frozen timing rule",
    )
    require(matched, "matched repeat 1 was not outcome-blind authorized")

    admission = documents["bf16_technical_admission"]
    inputs = admission["inputs"]
    e = _number(inputs.get("provider_elapsed_seconds"), "bf16 admission elapsed")
    forecast = e + 1.25 * lhat
    allowed = matched and forecast <= deadline
    _close(inputs.get("nf4_technical_durable_elapsed_seconds"), L,
           "bf16 admission L", 1e-12)
    _close(inputs.get("technical_forecast_seconds"), lhat,
           "bf16 admission Lhat")
    _close(inputs.get("forecast_seconds"), forecast,
           "bf16 admission forecast")
    require(inputs.get("initial_matched_authorized") is matched
            and inputs.get("nf4_technical_status") == "PASS"
            and inputs.get("scientific_cap_seconds") == cap
            and admission.get("authorized") is allowed
            and admission.get("decision") == (
                "LOAD_BF16" if allowed else "STOP_BEFORE_BF16_LOAD")
            and admission.get("formula") == "e+1.25*max(L,900)<=C-300",
            "bfloat16 technical-admission decision differs")
    require(allowed, "bfloat16 technical work lacked outcome-blind admission")

    bf16_tech = technical_receipts["bf16"]
    repeat1 = documents["bf16_repeat1"]
    inputs = repeat1["inputs"]
    e = _number(inputs.get("provider_elapsed_seconds"), "bf16 r1 elapsed")
    forecast = e + 1.25 * t
    allowed = (matched and bf16_tech.get("status") == "PASS"
               and manifest.get("matched_runtime_gate", {}).get("status") == "PASS"
               and forecast <= deadline)
    _close(inputs.get("nf4_repeat1_durable_elapsed_seconds"), t,
           "bf16 r1 t", 1e-12)
    _close(inputs.get("forecast_seconds"), forecast, "bf16 r1 forecast")
    require(inputs.get("initial_matched_authorized") is matched
            and inputs.get("nf4_repeat1_completion_status") == "COMPLETE"
            and inputs.get("bf16_technical_status") == "PASS"
            and inputs.get("matched_runtime_status") == "PASS"
            and inputs.get("scientific_cap_seconds") == cap
            and repeat1.get("authorized") is allowed
            and repeat1.get("decision") == (
                "RUN_BF16_REPEAT1" if allowed else "TIMING_OR_GATE_STOP")
            and repeat1.get("formula") == "elapsed+1.25*t_nf4<=C-300",
            "bfloat16 repeat-1 decision differs")
    require(allowed, "bfloat16 repeat 1 lacked outcome-blind authorization")

    repeat2 = documents["bf16_repeat2"]
    inputs = repeat2["inputs"]
    bf16_r1 = outcome_receipts[("bf16", 1)]
    tb = _number(bf16_r1.get("durable_elapsed_seconds"), "bfloat16 t")
    e = _number(inputs.get("provider_elapsed_seconds"), "bf16 r2 elapsed")
    proxy = max(t, tb)
    forecast = e + 1.25 * proxy
    nf4_r2_status = outcome_receipts.get(("nf4", 2), {}).get(
        "completion_status", "ABSENT")
    r2_allowed = (riders and nf4_r2_status == "COMPLETE"
                  and forecast <= deadline)
    _close(inputs.get("nf4_repeat1_durable_elapsed_seconds"), t,
           "bf16 r2 NF4 t", 1e-12)
    _close(inputs.get("bf16_repeat1_durable_elapsed_seconds"), tb,
           "bf16 r2 bfloat16 t", 1e-12)
    _close(inputs.get("forecast_proxy_seconds"), proxy,
           "bf16 r2 proxy")
    _close(inputs.get("forecast_seconds"), forecast, "bf16 r2 forecast")
    require(inputs.get("initial_repeat2_authorized") is riders
            and inputs.get("nf4_repeat1_completion_status") == "COMPLETE"
            and inputs.get("nf4_repeat2_completion_status") == nf4_r2_status
            and inputs.get("bf16_repeat1_completion_status") == "COMPLETE"
            and inputs.get("scientific_cap_seconds") == cap
            and repeat2.get("authorized") is r2_allowed
            and repeat2.get("decision") == (
                "RUN_BF16_REPEAT2" if r2_allowed else "SKIP_BF16_REPEAT2")
            and repeat2.get("formula")
            == "elapsed+1.25*max(t_nf4,t_bf16)<=C-300",
            "bfloat16 repeat-2 decision differs")
    return {
        "status": "PASS",
        "initial_matched_repeat1_authorized": matched,
        "initial_repeat2_authorized": riders,
        "bf16_repeat2_authorized": r2_allowed,
        "documents": {name: {
            "decision": documents[name].get("decision"),
            "authorized": documents[name].get("authorized"),
        } for name in DECISION_NAMES},
    }


def _repeat_view(raw: Mapping[str, Any]) -> dict[str, Any]:
    treatment = raw["treatment"]
    return {
        "phase_a": raw["phase_a"],
        "treatment": {
            "fresh_scores": treatment["fresh_scores"],
            "arms": treatment["arms"][:7],
        },
    }


def _repeat_and_cross_guard(
    analyses: Mapping[tuple[str, int], Mapping[str, Any]],
    raws: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    stability: dict[str, Any] = {}
    available_within: dict[str, list[float]] = {}
    generated_instability = False
    for regime in REGIMES:
        if (regime, 2) not in analyses:
            stability[regime] = {"status": "NOT_AVAILABLE"}
            continue
        left = _repeat_view(raws[(regime, 1)])
        right = _repeat_view(raws[(regime, 2)])
        differences = common.recursive_differences(left, right)
        left_analysis = analyses[(regime, 1)]
        right_analysis = analyses[(regime, 2)]
        scalar_left = _scalar_estimands(left_analysis)
        scalar_right = _scalar_estimands(right_analysis)
        scalar_delta = {
            name: scalar_right[name] - scalar_left[name]
            for name in scalar_left
        }
        generation_stable = all(
            left_analysis["E2_literal_generated_behavior"][probe][
                "generation_hash_tuple"
            ] == right_analysis["E2_literal_generated_behavior"][probe][
                "generation_hash_tuple"
            ]
            for probe in ("focal", "nonfocal")
        )
        # The preregistration blocks E2/E3 interpretation on either literal
        # generation instability or any common scientific-payload hash
        # instability.  A nonempty recursive diff necessarily changes that
        # payload's canonical hash even when the decoded strings are stable.
        generated_instability = (
            generated_instability or bool(differences) or not generation_stable
        )
        stability[regime] = {
            "status": "PASS" if not differences else "FAIL",
            "repeat1_sha256": common.sha256_bytes(common.canonical_bytes(left)),
            "repeat2_sha256": common.sha256_bytes(common.canonical_bytes(right)),
            "differing_json_pointers": differences,
            "differing_field_count": len(differences),
            "generation_hashes_and_text_stable": generation_stable,
            "scalar_repeat2_minus_repeat1": scalar_delta,
            "note": "Repeat executions are a stability check, not independent cases.",
        }
        for name, value in scalar_delta.items():
            available_within.setdefault(name, []).append(abs(value))

    nf4 = _scalar_estimands(analyses[("nf4", 1)])
    bf16 = _scalar_estimands(analyses[("bf16", 1)])
    guard: dict[str, Any] = {}
    for name in nf4:
        cross = nf4[name] - bf16[name]
        within_values = available_within.get(name, [])
        maximum = max(within_values) if within_values else None
        guard[name] = {
            "nf4_repeat1": nf4[name],
            "bf16_repeat1": bf16[name],
            "nf4_minus_bf16": cross,
            "absolute_cross_runtime_delta": abs(cross),
            "maximum_available_absolute_within_regime_repeat_delta": maximum,
            "numerical_interpretation_permitted_by_repeat_guard": (
                None if maximum is None else abs(cross) > maximum
            ),
        }
    return {
        "within_regime_repeat_stability": stability,
        "cross_runtime_vs_within_repeat_guard": guard,
        "E2_E3_blocked_by_generation_or_hash_instability": generated_instability,
        "guard_note": (
            "A scalar cross-runtime delta no larger than the maximum available "
            "within-regime repeat delta receives no numerical interpretation. "
            "Absent repeat 2 leaves this guard unavailable, not passed."
        ),
    }


def _matched_primary(
    analyses: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    nf4 = analyses[("nf4", 1)]
    bf16 = analyses[("bf16", 1)]
    nf4_scalars = _scalar_estimands(nf4)
    bf16_scalars = _scalar_estimands(bf16)
    e2: dict[str, Any] = {}
    for probe in ("focal", "nonfocal"):
        n = nf4["E2_literal_generated_behavior"][probe]
        b = bf16["E2_literal_generated_behavior"][probe]
        e2[probe] = {
            "nf4_literal_generation_tuple": n["literal_generation_tuple"],
            "bf16_literal_generation_tuple": b["literal_generation_tuple"],
            "nf4_generation_hash_tuple": n["generation_hash_tuple"],
            "bf16_generation_hash_tuple": b["generation_hash_tuple"],
            "literal_tuples_equal": n["literal_generation_tuple"]
            == b["literal_generation_tuple"],
            "nf4_change_vector": n["four_bit_change_vector"],
            "bf16_change_vector": b["four_bit_change_vector"],
            "change_vectors_equal": n["four_bit_change_vector"]
            == b["four_bit_change_vector"],
        }
    return {
        "primary_repeat": 1,
        "E1_E3_and_secondary_scalars": {
            "nf4": nf4_scalars,
            "bf16": bf16_scalars,
            "nf4_minus_bf16": {
                name: nf4_scalars[name] - bf16_scalars[name]
                for name in nf4_scalars
            },
        },
        "E2_literal_generated_behavior": e2,
        "placebo_status": {
            regime: analyses[(regime, 1)]["secondary_controls"]["placebo"]
            for regime in REGIMES
        },
        "interpretation_scope": (
            "Matched repeat 1 only; one engineered e01 fixture under a bundled "
            "weight-representation/kernel runtime axis; descriptive screening only."
        ),
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    require(run_dir.is_dir(), f"P02 run directory is absent: {run_dir}")
    outer = _validate_outer_receipt(run_dir)
    completion_path, completion, manifest_path, manifest = _terminal_documents(
        run_dir)
    inventory = _verify_completion_inventory(run_dir, completion_path, completion)
    source = _validate_source_bindings(manifest.get("bindings"))
    common_bindings = source["bindings"]

    provider = manifest.get("provider")
    require(isinstance(provider, Mapping), "P02 provider record is absent")
    rate = _number(provider.get("hourly_cost_usd"), "provider rate", positive=True)
    cap = provider.get("scientific_cap_seconds")
    require(
        isinstance(cap, int) and not isinstance(cap, bool)
        and cap == min(9000, math.floor(3.90 * 3600 / rate))
        and completion.get("scientific_cap_seconds") == cap
        and provider.get("scientific_budget_usd") == 3.90
        and provider.get("forecast_reserve_seconds") == 300.0,
        "P02 provider/scientific cap differs",
    )
    completion_elapsed = _number(
        completion.get("provider_elapsed_seconds"), "completion elapsed")
    estimated_cost = _number(
        completion.get("estimated_provider_cost_usd"), "completion estimated cost")
    require(completion_elapsed <= cap and estimated_cost <= 3.90,
            "P02 completion exceeded its scientific work cap")

    documents = package_documents(run_dir)
    package_by_dir = {row["package_dir"]: row for row in documents}
    require(len(package_by_dir) == len(documents),
            "duplicate P02 lossless package directory")
    technical_summary, technical_rows = _technical_gate(
        run_dir, documents, common_bindings)
    declared_gate = manifest.get("matched_runtime_gate")
    require(completion.get("matched_runtime_gate") == declared_gate,
            "completion/manifest matched-runtime gates differ")
    matched = _matched_runtime_gate(technical_rows, declared_gate)

    regimes = manifest.get("regimes")
    require(isinstance(regimes, Mapping) and set(regimes) == set(REGIMES),
            "P02 manifest regime set differs")
    technical_receipts: dict[str, Mapping[str, Any]] = {}
    outcome_receipts: dict[tuple[str, int], Mapping[str, Any]] = {}
    final_outcomes: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in documents:
        raw = row["raw"]
        if raw.get("schema") == OUTCOME_SCHEMA:
            key = (raw.get("regime"), raw.get("repeat_index"))
            require(key[0] in REGIMES and key[1] in {1, 2},
                    "P02 packaged outcome key differs")
            require(key not in final_outcomes,
                    f"duplicate P02 final outcome package: {key}")
            final_outcomes[key] = row

    for regime in REGIMES:
        regime_row = regimes[regime]
        require(isinstance(regime_row, Mapping)
                and isinstance(regime_row.get("outcomes"), list),
                f"{regime} manifest regime record differs")
        technical_receipt = regime_row.get("technical")
        require(isinstance(technical_receipt, Mapping)
                and technical_receipt.get("status") == "PASS"
                and technical_receipt.get("regime") == regime
                and technical_receipt.get("recovery_raw_path") is None
                and technical_receipt.get("persistence_error") is None,
                f"{regime} technical receipt differs")
        technical_receipts[regime] = technical_receipt
        technical_row = technical_rows[regime]
        _verify_package_binding(
            run_dir, technical_receipt.get("raw_package"), technical_row,
            f"{regime} technical")
        require(technical_receipt.get("runtime_fingerprint")
                == technical_row["raw"].get("runtime_fingerprint")
                and technical_receipt.get("subject_bindings")
                == technical_row["raw"].get("subject_attestation", {}).get(
                    "bindings"),
                f"{regime} technical receipt subject differs from raw")
        technical_render = _verify_file_binding(
            run_dir, technical_receipt.get("renders"),
            f"{regime} technical render")
        _verify_technical_render(
            technical_row["raw"], technical_receipt["raw_package"],
            technical_render)
        timing_path = _verify_file_binding(
            run_dir, technical_receipt.get("timing_receipt"),
            f"{regime} technical timing")
        timing = common.load_object(timing_path)
        _formal_boundary(timing, f"{regime} technical timing")
        require(timing.get("schema") == TIMING_SCHEMA
                and timing.get("timing_kind") == "technical"
                and timing.get("regime") == regime
                and timing.get("status") == "PASS"
                and timing.get("durable_elapsed_seconds")
                == technical_receipt.get("durable_elapsed_seconds")
                and timing.get("raw_package")
                == technical_receipt.get("raw_package")
                and timing.get("renders") == technical_receipt.get("renders"),
                f"{regime} technical timing receipt differs")

        for receipt in regime_row["outcomes"]:
            require(isinstance(receipt, Mapping)
                    and receipt.get("regime") == regime
                    and receipt.get("case_id") == CASE_ID
                    and receipt.get("repeat_index") in {1, 2},
                    f"{regime} outcome receipt differs")
            key = (regime, int(receipt["repeat_index"]))
            require(key not in outcome_receipts,
                    f"duplicate manifest outcome receipt: {key}")
            outcome_receipts[key] = receipt

    for regime in REGIMES:
        require((regime, 1) in outcome_receipts
                and outcome_receipts[(regime, 1)].get("completion_status")
                == "COMPLETE",
                f"matched repeat 1 is absent/incomplete for {regime}")
    receipted_packages = {
        key for key, row in outcome_receipts.items()
        if row.get("raw_package") is not None
    }
    require(set(final_outcomes) == receipted_packages,
            "final outcome package set differs from manifest receipts")
    require(completion.get("matched_repeat1_eligible") is True
            and manifest.get("matched_repeat1_eligible") is True,
            "runner did not mark matched repeat 1 eligible")

    decisions = _validate_decisions(
        run_dir, completion, manifest, technical_receipts, outcome_receipts)
    statuses = {key: str(row.get("completion_status"))
                for key, row in outcome_receipts.items()}
    rider = validate_repeat2_layout(
        statuses,
        declared_rider_status=str(completion.get("repeat2_rider_status")),
        initial_repeat2_authorized=bool(
            decisions["initial_repeat2_authorized"]),
        bf16_repeat2_authorized=bool(decisions["bf16_repeat2_authorized"]),
    )
    require(manifest.get("repeat2_rider_status") == rider,
            "manifest/completion repeat-2 rider statuses differ")

    analyses: dict[tuple[str, int], Mapping[str, Any]] = {}
    raws: dict[tuple[str, int], Mapping[str, Any]] = {}
    verification: dict[str, Any] = {}
    for key, receipt in sorted(outcome_receipts.items()):
        status = receipt.get("completion_status")
        package_binding = receipt.get("raw_package")
        if package_binding is None:
            require(status != "COMPLETE" and receipt.get("phase_package") is None
                    and receipt.get("compact") is None
                    and receipt.get("renders") is None,
                    f"unpackaged P02 receipt carries completed/derived state: {key}")
            continue
        package_dir = _resolve_run_path(
            run_dir, package_binding.get("path"), f"{key} raw package")
        final_row = package_by_dir.get(package_dir)
        require(final_row is not None and final_row.get("raw", {}).get("schema")
                == OUTCOME_SCHEMA, f"{key} raw package was not reconstructed")
        _verify_package_binding(run_dir, package_binding, final_row, f"{key} outcome")
        raw = final_row["raw"]
        require(raw.get("regime") == key[0]
                and raw.get("repeat_index") == key[1]
                and raw.get("status") == status,
                f"{key} receipt differs from final raw")
        _formal_boundary(raw, f"{key} outcome")
        _validate_raw_bindings(raw, common_bindings, f"{key} outcome")
        require(raw.get("runtime_fingerprint")
                == technical_rows[key[0]]["raw"].get("runtime_fingerprint")
                and raw.get("subject_attestation")
                == technical_rows[key[0]]["raw"].get("subject_attestation"),
                f"{key} outcome subject differs from passing technical gate")
        timing_path = _verify_file_binding(
            run_dir, receipt.get("timing_receipt"), f"{key} timing receipt")
        timing = common.load_object(timing_path)
        _formal_boundary(timing, f"{key} timing receipt")
        require(timing.get("schema") == TIMING_SCHEMA
                and timing.get("timing_kind") == "outcome"
                and timing.get("regime") == key[0]
                and timing.get("repeat_index") == key[1]
                and timing.get("completion_status") == status
                and timing.get("durable_elapsed_seconds")
                == receipt.get("durable_elapsed_seconds")
                and timing.get("raw_package") == package_binding
                and timing.get("phase_package") == receipt.get("phase_package")
                and timing.get("checkpoint_chain")
                == receipt.get("checkpoint_chain"),
                f"{key} timing receipt differs")
        if status == "COMPLETE":
            require(receipt.get("recovery_raw_path") is None
                    and timing.get("recovery_raw_path") is None
                    and timing.get("persistence_error") is None,
                    f"{key} complete outcome retained recovery/persistence error")
            analysis = analyze_outcome(raw)
            _verify_checkpoint_chain(
                run_dir, receipt, final_row, package_by_dir)
            compact_path = _verify_file_binding(
                run_dir, receipt.get("compact"), f"{key} compact")
            render_path = _verify_file_binding(
                run_dir, receipt.get("renders"), f"{key} render")
            _verify_derived(raw, package_binding, compact_path, render_path)
            analyses[key] = analysis
            raws[key] = raw
            verification[f"{key[0]}:e01:r{key[1]}"] = {
                "raw_package": "VERIFIED",
                "phase_foundation_and_arm_checkpoint_chain": "VERIFIED",
                "phase_a_grid": "VERIFIED",
                "targeted_arm_order": "FF_PLUS_SIX_GRAFTS_AND_R1_PLACEBO_VERIFIED",
                "compact_consistency": "PASS",
                "render_consistency": "PASS",
            }
        else:
            require(key[1] == 2, "matched repeat 1 is not complete")
            if receipt.get("compact") is not None and receipt.get("renders") is not None:
                compact_path = _verify_file_binding(
                    run_dir, receipt["compact"], f"{key} partial compact")
                render_path = _verify_file_binding(
                    run_dir, receipt["renders"], f"{key} partial render")
                _verify_derived(raw, package_binding, compact_path, render_path)

    mandatory = {("nf4", 1), ("bf16", 1)}
    require(mandatory <= set(analyses),
            "matched repeat-1 raw packages are absent or invalid")
    matched_primary = _matched_primary(analyses)
    stability = _repeat_and_cross_guard(analyses, raws)
    outcome_rows = {
        f"{regime}:e01:r{repeat}": {
            "regime": regime,
            "case_id": CASE_ID,
            "repeat": repeat,
            "primary": repeat == 1,
            "analysis": analysis,
        }
        for (regime, repeat), analysis in sorted(analyses.items())
    }
    return {
        "schema": ANALYSIS_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir),
        **FORMAL_FLAGS,
        "fixed_fixture_exploratory_only": True,
        "primary_dataset": "matched repeat 1 only",
        "runner_terminal_status": completion["status"],
        "outer_pod_receipt": outer,
        "terminal_files": {
            "manifest": {
                "path": manifest_path.relative_to(run_dir).as_posix(),
                "sha256": common.file_sha256(manifest_path),
            },
            "completion": {
                "path": completion_path.relative_to(run_dir).as_posix(),
                "sha256": common.file_sha256(completion_path),
            },
            "completion_inventory": inventory,
        },
        "all_lossless_packages": {
            "status": "VERIFIED",
            "count": len(documents),
            "manifest_sha256s": sorted(
                row["package_manifest_sha256"] for row in documents),
        },
        "source_bindings": source,
        "technical_gates": technical_summary,
        "matched_runtime_gate": matched,
        "continuation_decisions": decisions,
        "repeat2_rider_status": rider,
        "package_and_derived_verification": verification,
        "outcomes": outcome_rows,
        "matched_repeat1_descriptive": matched_primary,
        "repeat_stability_and_interpretation_guard": stability,
        "inference": {
            "p_values_computed": False,
            "confidence_intervals_computed": False,
            "ratios_computed": False,
            "equivalence_test_computed": False,
            "formal_release_decision_computed": False,
            "population_interaction_claim_authorized": False,
            "quantization_dependence_claim_authorized": False,
            "note": (
                "P02 is a fixed-case descriptive screen of a bundled weight/"
                "kernel runtime axis. Divergence is only a screening flag for a "
                "new preregistration; it is not efficacy or causal attribution."
            ),
        },
    }


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True,
                          ensure_ascii=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output = args.run_dir.parent / (
            f"precision-probe-p02-independent-analysis_"
            f"Qwen3-30B-A3B-Instruct-2507_{stamp}.json"
        )
    else:
        args.output = args.output.resolve()
    return args


def main() -> None:
    args = parse_args()
    result = analyze_run(args.run_dir)
    write_exclusive(args.output, result)
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "analysis_status": "COMPLETE",
        "output": str(args.output),
        "output_sha256": common.file_sha256(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
