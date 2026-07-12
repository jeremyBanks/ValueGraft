#!/usr/bin/env python3
"""Run the frozen outcome-blind precision-probe-p02 matched screen.

P02 reuses the P01 subject loader and technical gate without alteration.  Its
scientific runner differs in three deliberate ways: it executes only the
preregistered targeted treatment, checkpoints the treatment foundation and
every arm synchronously, and freezes every continuation decision before any
P02 raw package is reconstructed for compact output or rendering.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# These imports are the component-reuse boundary.  In particular, P02 does
# not copy or weaken the P01 loader or technical assertions.
import run_precision_probe_p01 as p01  # noqa: E402
from coherent_canary_case import PHASE_A_SCHEMA, run_phase_a_case  # noqa: E402
from precision_probe_p02_case import (  # noqa: E402
    TREATMENT_SCHEMA,
    assert_targeted_treatment,
    run_targeted_treatment_case,
)


PROTOCOL_ID = "precision-probe-p02"
RUN_SCHEMA = "precision_probe_p02_run_manifest_v1"
TECHNICAL_SCHEMA = "precision_probe_p02_technical_raw_v1"
OUTCOME_SCHEMA = "precision_probe_p02_outcome_raw_v1"
COMPACT_SCHEMA = "precision_probe_p02_compact_scores_generations_v1"
DECISION_SCHEMA = "precision_probe_p02_continuation_decision_v1"
TIMING_SCHEMA = "precision_probe_p02_timing_receipt_v1"
CHECKPOINT_SCHEMA = "precision_probe_p02_checkpoint_v1"
COMPLETION_SCHEMA = "precision_probe_p02_completion_v1"

MODEL_ID = p01.MODEL_ID
MODEL_SLUG = p01.MODEL_SLUG
REVISION = p01.REVISION
REGIMES = p01.REGIMES
CASE_ID = "e01"

DEFAULT_PREREG = ROOT / "PRECISION-PROBE-P02-PREREGISTRATION.md"
DEFAULT_IDENTITY = p01.DEFAULT_IDENTITY
DEFAULT_TECHNICAL = p01.DEFAULT_TECHNICAL
DEFAULT_CASE = p01.DEFAULT_CASES[CASE_ID]
DEFAULT_OUTPUT_DIR = ROOT / "results/precision_probe_p02"
PREREG_SHA256 = "dbcece8f189a0574b776149cf597dc1ce8a3b592981440062c749ddb404e221d"
CASE_E01_SHA256 = "6a2ad7ae0bf094fa5727e76fb710aaeba7bc72082bd93082a5e9a42e09126090"
IDENTITY_FIXTURE_SHA256 = "6123c5462f15cdd303af4629758ab82e6458f08db3ed9d840664478aaf9a7651"
TECHNICAL_FIXTURE_SHA256 = "cfc05dfe62a4dafbb7947bc2b8df9cccf125a6dfa5b2ec70d11a8f5762467921"

FORMAL_FLAGS = dict(p01.FORMAL_FLAGS)
SCIENTIFIC_BUDGET_USD = 3.90
MAX_SCIENTIFIC_SECONDS = 9000
FORECAST_RESERVE_SECONDS = 300.0
TECHNICAL_FLOOR_SECONDS = 900.0
# The preregistration's upper planning bound for one full targeted outcome.
# It is used only before the first empirical outcome duration exists.
INITIAL_OUTCOME_FORECAST_SECONDS = 24.0 * 60.0
FORECAST_MULTIPLIER = 1.25

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


class PrecisionProbeP02RunnerError(RuntimeError):
    """The P02 contract, technical gate, or persistence step differed."""


@dataclass(frozen=True, slots=True)
class TechnicalDecisionTiming:
    """Scalar-only technical information admitted to scheduling code."""

    status: str
    durable_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class OutcomeDecisionTiming:
    """Scalar-only outcome information admitted to scheduling code."""

    completion_status: str
    durable_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class OutcomePaths:
    raw_scratch: Path
    raw_package: Path
    phase_scratch: Path
    phase_package: Path
    foundation_scratch: Path
    foundation_package: Path
    arm_scratch: tuple[Path, ...]
    arm_packages: tuple[Path, ...]
    timing_receipt: Path
    compact: Path
    renders: Path


@dataclass(frozen=True, slots=True)
class OutcomeReceipt:
    regime: str
    case_id: str
    repeat_index: int
    completion_status: str
    durable_elapsed_seconds: float
    raw_package: Mapping[str, Any] | None
    phase_package: Mapping[str, Any] | None
    checkpoint_chain: tuple[Mapping[str, Any], ...]
    timing_receipt: Mapping[str, Any] | None
    recovery_raw_path: str | None
    compact_path: str | None = None
    renders_path: str | None = None

    def decision_timing(self) -> OutcomeDecisionTiming:
        return OutcomeDecisionTiming(
            completion_status=self.completion_status,
            durable_elapsed_seconds=float(self.durable_elapsed_seconds),
        )


# Re-export the exact P01 provider clock for the ops/test surface.
ProviderClock = p01.ProviderClock


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionProbeP02RunnerError(message)


def scientific_cap_seconds(
    hourly_cost_usd: float, provider_wall_cap_seconds: float | None = None
) -> int:
    require(
        math.isfinite(hourly_cost_usd) and hourly_cost_usd > 0,
        "hourly provider rate must be positive",
    )
    computed = min(
        MAX_SCIENTIFIC_SECONDS,
        math.floor(SCIENTIFIC_BUDGET_USD * 3600.0 / hourly_cost_usd),
    )
    if provider_wall_cap_seconds is None:
        cap = computed
    else:
        require(
            math.isfinite(provider_wall_cap_seconds)
            and 0 < provider_wall_cap_seconds <= MAX_SCIENTIFIC_SECONDS,
            "provider wall cap must be in (0, 9000]",
        )
        cap = min(computed, math.floor(provider_wall_cap_seconds))
    require(cap > FORECAST_RESERVE_SECONDS, "scientific cap lacks reserve")
    return int(cap)


def _valid_elapsed(value: float) -> bool:
    return math.isfinite(value) and value >= 0


def _decision_document(
    name: str,
    *,
    decision: str,
    authorized: bool,
    inputs: Mapping[str, Any],
    formula: str,
) -> dict[str, Any]:
    require(name in DECISION_NAMES, "unknown P02 continuation decision")
    return {
        "schema": DECISION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        **FORMAL_FLAGS,
        "decision_name": name,
        "decision": decision,
        "authorized": bool(authorized),
        "inputs": p01.json_safe(inputs),
        "formula": formula,
        "outcome_information_surface": [
            "completion_status", "durable_elapsed_seconds"
        ],
        "technical_information_surface": [
            "status", "durable_elapsed_seconds"
        ],
        "score_generation_path_or_payload_accessible": False,
        "created_at_utc": p01.utc_now(),
    }


def decide_initial_schedule(
    *,
    nf4_technical: TechnicalDecisionTiming,
    nf4_repeat1: OutcomeDecisionTiming,
    provider_elapsed_seconds: float,
    scientific_cap: int,
) -> dict[str, Any]:
    """Freeze matched-r1 and both-r2 authorization from scalar receipts only."""
    require(type(nf4_technical) is TechnicalDecisionTiming,
            "initial decision requires scalar technical timing")
    require(type(nf4_repeat1) is OutcomeDecisionTiming,
            "initial decision requires scalar outcome timing")
    require(_valid_elapsed(provider_elapsed_seconds), "provider elapsed is invalid")
    require(isinstance(scientific_cap, int) and scientific_cap > 300,
            "scientific cap is invalid")
    L = float(nf4_technical.durable_elapsed_seconds)
    t = float(nf4_repeat1.durable_elapsed_seconds)
    valid = (
        nf4_technical.status == "PASS"
        and nf4_repeat1.completion_status == "COMPLETE"
        and _valid_elapsed(L)
        and _valid_elapsed(t)
    )
    lhat = max(L, TECHNICAL_FLOOR_SECONDS) if _valid_elapsed(L) else None
    matched_forecast = (
        provider_elapsed_seconds + FORECAST_MULTIPLIER * (lhat + t)
        if valid and lhat is not None else None
    )
    rider_forecast = (
        provider_elapsed_seconds
        + FORECAST_MULTIPLIER * (t + lhat + 2.0 * t)
        if valid and lhat is not None else None
    )
    deadline = scientific_cap - FORECAST_RESERVE_SECONDS
    matched = bool(valid and matched_forecast is not None
                   and matched_forecast <= deadline)
    riders = bool(matched and rider_forecast is not None
                  and rider_forecast <= deadline)
    return _decision_document(
        "initial_schedule",
        decision=("MATCHED_R1_AND_BOTH_R2" if riders else
                  "MATCHED_R1_ONLY" if matched else "STOP_NF4_ONLY"),
        authorized=matched,
        inputs={
            "nf4_technical_status": nf4_technical.status,
            "nf4_technical_durable_elapsed_seconds": L,
            "nf4_repeat1_completion_status": nf4_repeat1.completion_status,
            "nf4_repeat1_durable_elapsed_seconds": t,
            "technical_forecast_seconds": lhat,
            "provider_elapsed_seconds": provider_elapsed_seconds,
            "scientific_cap_seconds": scientific_cap,
            "forecast_deadline_seconds": deadline,
            "matched_repeat1_forecast_seconds": matched_forecast,
            "both_repeat2_forecast_seconds": rider_forecast,
        },
        formula=(
            "matched: e+1.25*(max(L,900)+t)<=C-300; "
            "riders: e+1.25*(t+max(L,900)+2*t)<=C-300"
        ),
    ) | {
        "matched_repeat1_authorized": matched,
        "both_repeat2_authorized": riders,
    }


def decide_bf16_technical_admission(
    *,
    initial_matched_authorized: bool,
    nf4_technical: TechnicalDecisionTiming,
    provider_elapsed_seconds: float,
    scientific_cap: int,
) -> dict[str, Any]:
    """Apply the per-stage forecast before loading the bfloat16 subject."""
    require(type(initial_matched_authorized) is bool,
            "bf16 technical admission requires scalar authorization")
    require(type(nf4_technical) is TechnicalDecisionTiming,
            "bf16 technical admission requires scalar technical timing")
    L = float(nf4_technical.durable_elapsed_seconds)
    lhat = max(L, TECHNICAL_FLOOR_SECONDS) if _valid_elapsed(L) else None
    forecast = (
        provider_elapsed_seconds + FORECAST_MULTIPLIER * lhat
        if lhat is not None else None
    )
    allowed = bool(
        initial_matched_authorized
        and nf4_technical.status == "PASS"
        and forecast is not None
        and forecast <= scientific_cap - FORECAST_RESERVE_SECONDS
    )
    return _decision_document(
        "bf16_technical_admission",
        decision="LOAD_BF16" if allowed else "STOP_BEFORE_BF16_LOAD",
        authorized=allowed,
        inputs={
            "initial_matched_authorized": initial_matched_authorized,
            "nf4_technical_status": nf4_technical.status,
            "nf4_technical_durable_elapsed_seconds": L,
            "technical_forecast_seconds": lhat,
            "provider_elapsed_seconds": provider_elapsed_seconds,
            "scientific_cap_seconds": scientific_cap,
            "forecast_seconds": forecast,
        },
        formula="e+1.25*max(L,900)<=C-300",
    )


def decide_bf16_repeat1(
    *,
    initial_matched_authorized: bool,
    nf4_repeat1: OutcomeDecisionTiming,
    bf16_technical: TechnicalDecisionTiming,
    matched_runtime_status: str,
    provider_elapsed_seconds: float,
    scientific_cap: int,
) -> dict[str, Any]:
    """Apply the preregistered matched-repeat-1 decision after bf16 gating."""
    require(type(nf4_repeat1) is OutcomeDecisionTiming,
            "bf16 r1 decision requires scalar outcome timing")
    require(type(bf16_technical) is TechnicalDecisionTiming,
            "bf16 r1 decision requires scalar technical timing")
    t = float(nf4_repeat1.durable_elapsed_seconds)
    forecast = (
        provider_elapsed_seconds + FORECAST_MULTIPLIER * t
        if _valid_elapsed(t) else None
    )
    allowed = bool(
        initial_matched_authorized
        and nf4_repeat1.completion_status == "COMPLETE"
        and bf16_technical.status == "PASS"
        and matched_runtime_status == "PASS"
        and forecast is not None
        and forecast <= scientific_cap - FORECAST_RESERVE_SECONDS
    )
    return _decision_document(
        "bf16_repeat1",
        decision="RUN_BF16_REPEAT1" if allowed else "TIMING_OR_GATE_STOP",
        authorized=allowed,
        inputs={
            "initial_matched_authorized": initial_matched_authorized,
            "nf4_repeat1_completion_status": nf4_repeat1.completion_status,
            "nf4_repeat1_durable_elapsed_seconds": t,
            "bf16_technical_status": bf16_technical.status,
            "matched_runtime_status": matched_runtime_status,
            "provider_elapsed_seconds": provider_elapsed_seconds,
            "scientific_cap_seconds": scientific_cap,
            "forecast_seconds": forecast,
        },
        formula="elapsed+1.25*t_nf4<=C-300",
    )


def decide_bf16_repeat2(
    *,
    initial_repeat2_authorized: bool,
    nf4_repeat1: OutcomeDecisionTiming,
    nf4_repeat2: OutcomeDecisionTiming,
    bf16_repeat1: OutcomeDecisionTiming,
    provider_elapsed_seconds: float,
    scientific_cap: int,
) -> dict[str, Any]:
    """Apply the symmetric rider rule after the matched pair is attempted."""
    require(type(nf4_repeat1) is OutcomeDecisionTiming,
            "bf16 r2 decision requires scalar NF4 r1 timing")
    require(type(nf4_repeat2) is OutcomeDecisionTiming,
            "bf16 r2 decision requires scalar NF4 r2 status")
    require(type(bf16_repeat1) is OutcomeDecisionTiming,
            "bf16 r2 decision requires scalar bf16 r1 timing")
    tn = float(nf4_repeat1.durable_elapsed_seconds)
    tb = float(bf16_repeat1.durable_elapsed_seconds)
    valid = _valid_elapsed(tn) and _valid_elapsed(tb)
    proxy = max(tn, tb) if valid else None
    forecast = (
        provider_elapsed_seconds + FORECAST_MULTIPLIER * proxy
        if proxy is not None else None
    )
    allowed = bool(
        initial_repeat2_authorized
        and nf4_repeat2.completion_status == "COMPLETE"
        and bf16_repeat1.completion_status == "COMPLETE"
        and forecast is not None
        and forecast <= scientific_cap - FORECAST_RESERVE_SECONDS
    )
    return _decision_document(
        "bf16_repeat2",
        decision="RUN_BF16_REPEAT2" if allowed else "SKIP_BF16_REPEAT2",
        authorized=allowed,
        inputs={
            "initial_repeat2_authorized": initial_repeat2_authorized,
            "nf4_repeat1_completion_status": nf4_repeat1.completion_status,
            "nf4_repeat1_durable_elapsed_seconds": tn,
            "nf4_repeat2_completion_status": nf4_repeat2.completion_status,
            "bf16_repeat1_completion_status": bf16_repeat1.completion_status,
            "bf16_repeat1_durable_elapsed_seconds": tb,
            "forecast_proxy_seconds": proxy,
            "provider_elapsed_seconds": provider_elapsed_seconds,
            "scientific_cap_seconds": scientific_cap,
            "forecast_seconds": forecast,
        },
        formula="elapsed+1.25*max(t_nf4,t_bf16)<=C-300",
    )


def skipped_decision(name: str, reason: str) -> dict[str, Any]:
    return _decision_document(
        name,
        decision="NOT_REACHED",
        authorized=False,
        inputs={"reason": reason},
        formula="not evaluated because an earlier frozen gate stopped work",
    )


def _planned_paths(run_dir: Path, stamp: str) -> dict[str, Any]:
    scratch = run_dir / ".scratch"
    result: dict[str, Any] = {
        "run_manifest": run_dir / (
            f"precision-probe-p02-run-manifest_{MODEL_SLUG}_{stamp}.json"
        ),
        "completion": run_dir / (
            f"precision-probe-p02-completion_{MODEL_SLUG}_{stamp}.json"
        ),
        "decisions": {
            name: run_dir / (
                f"precision-probe-p02-decision-{name.replace('_', '-')}_"
                f"{MODEL_SLUG}_{stamp}.json"
            ) for name in DECISION_NAMES
        },
        "technical": {},
        "outcomes": {},
    }
    for regime in REGIMES:
        base = f"precision-probe-p02-technical-{regime}-raw_{MODEL_SLUG}_{stamp}"
        result["technical"][regime] = {
            "raw_scratch": scratch / f"{base}.json",
            "raw_package": run_dir / f"{base}.lossless-package",
            "identity_checkpoint_scratch": scratch / (
                f"precision-probe-p02-technical-{regime}-identity-checkpoint-raw_"
                f"{MODEL_SLUG}_{stamp}.json"
            ),
            "identity_checkpoint_package": run_dir / (
                f"precision-probe-p02-technical-{regime}-identity-checkpoint-raw_"
                f"{MODEL_SLUG}_{stamp}.lossless-package"
            ),
            "timing_receipt": run_dir / (
                f"precision-probe-p02-technical-{regime}-timing_"
                f"{MODEL_SLUG}_{stamp}.json"
            ),
            "renders": run_dir / f"{base.replace('-raw_', '-renders_')}.md",
        }
        for repeat in (1, 2):
            prefix = f"precision-probe-p02-outcome-{regime}-e01-repeat{repeat}"
            raw_base = f"{prefix}-raw_{MODEL_SLUG}_{stamp}"
            phase_base = (
                f"precision-probe-p02-phase-a-{regime}-e01-repeat{repeat}-raw_"
                f"{MODEL_SLUG}_{stamp}"
            )
            foundation_base = (
                f"precision-probe-p02-foundation-{regime}-e01-repeat{repeat}-raw_"
                f"{MODEL_SLUG}_{stamp}"
            )
            arm_count = 7 if repeat == 1 else 6
            arm_scratch = tuple(
                scratch / (
                    f"precision-probe-p02-arm-{regime}-e01-repeat{repeat}-"
                    f"index{index:02d}-raw_{MODEL_SLUG}_{stamp}.json"
                ) for index in range(1, arm_count + 1)
            )
            arm_packages = tuple(
                run_dir / (
                    f"precision-probe-p02-arm-{regime}-e01-repeat{repeat}-"
                    f"index{index:02d}-raw_{MODEL_SLUG}_{stamp}.lossless-package"
                ) for index in range(1, arm_count + 1)
            )
            result["outcomes"][(regime, repeat)] = OutcomePaths(
                raw_scratch=scratch / f"{raw_base}.json",
                raw_package=run_dir / f"{raw_base}.lossless-package",
                phase_scratch=scratch / f"{phase_base}.json",
                phase_package=run_dir / f"{phase_base}.lossless-package",
                foundation_scratch=scratch / f"{foundation_base}.json",
                foundation_package=run_dir / f"{foundation_base}.lossless-package",
                arm_scratch=arm_scratch,
                arm_packages=arm_packages,
                timing_receipt=run_dir / (
                    f"{prefix}-timing_{MODEL_SLUG}_{stamp}.json"
                ),
                compact=run_dir / f"{prefix}-compact_{MODEL_SLUG}_{stamp}.json",
                renders=run_dir / f"{prefix}-renders_{MODEL_SLUG}_{stamp}.md",
            )
    return result


def _path_record(paths: Mapping[str, Path], repo: Path) -> dict[str, str]:
    return {key: p01.display_path(path, repo) for key, path in paths.items()}


def _package_checkpoint(
    document: Mapping[str, Any],
    *,
    scratch: Path,
    package: Path,
    repo: Path,
    stage: str,
) -> dict[str, Any]:
    packaged, recovery, error = p01.persist_lossless_raw(
        document, scratch_raw=scratch, package=package, repo=repo
    )
    row = {
        "stage": stage,
        "raw_package": packaged,
        "recovery_raw_path": recovery,
        "persistence_error": error,
    }
    require(packaged is not None and error is None,
            f"{stage} checkpoint did not persist and verify")
    return row


def _technical_timing(receipt: Mapping[str, Any]) -> TechnicalDecisionTiming:
    return TechnicalDecisionTiming(
        status=str(receipt.get("status")),
        durable_elapsed_seconds=float(
            receipt.get("durable_elapsed_seconds", math.nan)
        ),
    )


def run_regime_technical(
    *,
    regime: str,
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    bindings: Mapping[str, Any],
    provider_clock: ProviderClock,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Run P01's exact loader and technical gate, with P02 persistence."""
    technical_started = time.monotonic()  # immediately before subject prepare
    document: dict[str, Any] = {
        "schema": TECHNICAL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        **FORMAL_FLAGS,
        "status": "STARTED",
        "regime": regime,
        "model": args.model,
        "revision": args.revision,
        "started_at_utc": p01.utc_now(),
        "bindings": bindings,
        "planned_paths": _path_record(paths, args.repo),
        "stage_timings": {},
        "durable_checkpoints": [],
    }
    prepared: dict[str, Any] | None = None
    fatal_interrupt: BaseException | None = None
    try:
        prepared = p01._stage(
            document,
            "prepare_precision_subject",
            lambda: p01.prepare_precision_subject(
                regime, args.repo, args.allow_download
            ),
        )
        require("model" in prepared and "tokenizer" in prepared,
                "P01 loader omitted model or tokenizer")
        runtime = prepared.get("runtime_fingerprint")
        require(isinstance(runtime, Mapping),
                "P01 loader omitted runtime fingerprint")
        require(isinstance(prepared.get("bindings"), Mapping)
                and bool(prepared["bindings"]),
                "P01 loader omitted subject bindings")
        eos_ids = runtime.get("eos_ids")
        require(isinstance(eos_ids, list) and eos_ids,
                "runtime fingerprint omitted EOS IDs")
        document["runtime_fingerprint"] = p01.json_safe(runtime)
        document["subject_attestation"] = p01.json_safe({
            key: value for key, value in prepared.items()
            if key not in {"model", "tokenizer"}
        })
        identity_fixture = p01.load_object(args.identity_fixture)
        technical_fixture = p01.load_object(args.technical_fixture)
        model = prepared["model"]
        tokenizer = prepared["tokenizer"]
        document["generated_forced_identity"] = p01._stage(
            document,
            "generated_forced_identity",
            lambda: p01.run_generated_forced_identity(
                model, tokenizer, identity_fixture,
                [int(value) for value in eos_ids],
            ),
        )
        identity_checkpoint = {
            **document,
            "status": "CHECKPOINT",
            "checkpoint_stage": "generated_forced_identity",
            "checkpoint_completed_at_utc": p01.utc_now(),
            "provider_elapsed_seconds": provider_clock.elapsed(),
        }
        checkpoint = _package_checkpoint(
            identity_checkpoint,
            scratch=paths["identity_checkpoint_scratch"],
            package=paths["identity_checkpoint_package"],
            repo=args.repo,
            stage="generated_forced_identity",
        )
        document["durable_checkpoints"].append(checkpoint)
        require(
            document["generated_forced_identity"].get("status") == "PASS",
            "generated/forced identity failed after durable persistence",
        )
        middle = int(technical_fixture["middle_end_msg"])
        n_plan = p01.build_role_native_plan(
            tokenizer, technical_fixture["correct"], middle_end_msg=middle
        )
        fresh_plan = p01.build_fresh_destination_plan(
            tokenizer, technical_fixture["correct"], middle_end_msg=middle
        )
        document["deterministic_repeats"] = {
            "correct_history_N": p01._stage(
                document,
                "correct_history_N_repeats",
                lambda: p01.deterministic_repeats(
                    lambda: p01.execute_replay_plan(model, n_plan),
                    "correct-history N replay",
                ),
            ),
            "fresh_destination": p01._stage(
                document,
                "fresh_destination_repeats",
                lambda: p01.deterministic_repeats(
                    lambda: p01.execute_fresh_plan(model, fresh_plan),
                    "fresh destination",
                ),
            ),
        }
        document["fresh_self_replacement"] = p01._stage(
            document,
            "fresh_self_replacement",
            lambda: p01.run_fresh_self_replacement(model, fresh_plan),
        )
        p01._assert_technical_gate(document)
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

    document["completed_at_utc"] = p01.utc_now()
    document["provider_elapsed_seconds"] = provider_clock.elapsed()
    document["hourly_cost_usd"] = args.hourly_cost_usd
    document["technical_wrapper_seconds_before_final_persistence"] = (
        time.monotonic() - technical_started
    )
    package_binding, recovery, package_error = p01.persist_lossless_raw(
        document,
        scratch_raw=paths["raw_scratch"],
        package=paths["raw_package"],
        repo=args.repo,
    )
    render_document = dict(document)
    if package_binding is not None:
        render_document["raw_package_binding"] = package_binding
    technical_render = p01.render_technical_ledger(render_document)
    technical_render = technical_render.replace(
        "P01 technical generation ledger", "P02 technical generation ledger"
    ).replace("precision-probe-p01", PROTOCOL_ID)
    p01.atomic_create_text(paths["renders"], technical_render)
    # L ends only after the persisted ledger has been read back and bound.
    render_binding = p01.binding(paths["renders"], repo=args.repo)
    durable_elapsed = time.monotonic() - technical_started
    status = document["status"] if package_binding is not None else "PERSISTENCE_ERROR"
    timing_document = {
        "schema": TIMING_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        **FORMAL_FLAGS,
        "timing_kind": "technical",
        "regime": regime,
        "status": status,
        "durable_elapsed_seconds": durable_elapsed,
        "measurement_start": "immediately_before_prepare_precision_subject",
        "measurement_end": "verified_final_raw_and_persisted_render_ledger",
        "raw_package": package_binding,
        "renders": render_binding,
        "provider_elapsed_seconds": provider_clock.elapsed(),
        "created_at_utc": p01.utc_now(),
    }
    p01.atomic_create_json(paths["timing_receipt"], timing_document)
    timing_binding = p01.binding(paths["timing_receipt"], repo=args.repo)
    receipt = {
        "status": status,
        "regime": regime,
        "durable_elapsed_seconds": durable_elapsed,
        "raw_package": package_binding,
        "renders": render_binding,
        "timing_receipt": timing_binding,
        "runtime_fingerprint": document.get("runtime_fingerprint"),
        "subject_bindings": document.get("subject_attestation", {}).get(
            "bindings"
        ),
        "recovery_raw_path": recovery,
        "persistence_error": package_error,
        "stage_timings": document.get("stage_timings", {}),
        "durable_checkpoints": document.get("durable_checkpoints", []),
    }
    if status != "PASS":
        p01._release_subject(prepared)
        prepared = None
    if fatal_interrupt is not None:
        p01._release_subject(prepared)
        raise fatal_interrupt
    return prepared, receipt


def _assert_phase(phase: Any, *, case_id: str) -> None:
    executions = phase.get("executions", {}) if isinstance(phase, Mapping) else {}
    scores = phase.get("scores", {}) if isinstance(phase, Mapping) else {}
    require(
        isinstance(phase, Mapping)
        and phase.get("schema") == PHASE_A_SCHEMA
        and phase.get("case_id") == case_id,
        "Phase-A payload envelope differs",
    )
    require(
        isinstance(executions, Mapping)
        and set(executions) == {"C_N", "W_N", "F"}
        and isinstance(scores, Mapping)
        and set(scores) == {
            "A_C_focal", "A_W_focal", "FF_focal",
            "A_C_nonfocal", "A_W_nonfocal", "FF_nonfocal",
        }
        and phase.get("treatment_scores_present") is False,
        "P02 did not receive the reused full Phase-A payload",
    )


def _checkpoint_document(
    *,
    stage: str,
    regime: str,
    repeat_index: int,
    payload_name: str,
    payload: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    phase_package: Mapping[str, Any] | None,
    foundation_package: Mapping[str, Any] | None,
    provider_clock: ProviderClock,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        **FORMAL_FLAGS,
        "stage": stage,
        "regime": regime,
        "case_id": CASE_ID,
        "repeat_index": repeat_index,
        "payload_name": payload_name,
        "payload": payload,
        "previous_checkpoint": previous,
        "phase_package": phase_package,
        "foundation_package": foundation_package,
        "provider_elapsed_seconds": provider_clock.elapsed(),
        "created_at_utc": p01.utc_now(),
    }


def execute_outcome(
    *,
    regime: str,
    repeat_index: int,
    prepared: Mapping[str, Any],
    args: argparse.Namespace,
    paths: OutcomePaths,
    common_bindings: Mapping[str, Any],
    provider_clock: ProviderClock,
) -> OutcomeReceipt:
    """Execute and synchronously persist one complete P02 selected outcome."""
    require(repeat_index in (1, 2), "P02 repeat index must be 1 or 2")
    case = p01.load_object(args.case_e01)
    require(case.get("case_id") == CASE_ID, "case file ID differs")
    runtime = prepared["runtime_fingerprint"]
    eos_ids = [int(value) for value in runtime["eos_ids"]]
    document: dict[str, Any] = {
        "schema": OUTCOME_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        **FORMAL_FLAGS,
        "status": "STARTED",
        "regime": regime,
        "case_id": CASE_ID,
        "repeat_index": repeat_index,
        "model": args.model,
        "revision": args.revision,
        "started_at_utc": p01.utc_now(),
        "bindings": {
            **common_bindings,
            "case": p01.binding(args.case_e01, repo=args.repo),
        },
        "runtime_fingerprint": p01.json_safe(runtime),
        "subject_attestation": p01.json_safe({
            key: value for key, value in prepared.items()
            if key not in {"model", "tokenizer"}
        }),
        "stage_timings": {},
        "durable_checkpoints": [],
    }
    started = time.monotonic()  # immediately before Phase A
    phase_binding: Mapping[str, Any] | None = None
    foundation_binding: Mapping[str, Any] | None = None
    checkpoint_chain: list[Mapping[str, Any]] = []
    fatal_interrupt: BaseException | None = None

    def persist_foundation(foundation: Mapping[str, Any]) -> None:
        nonlocal foundation_binding
        checkpoint = _checkpoint_document(
            stage="treatment_foundation",
            regime=regime,
            repeat_index=repeat_index,
            payload_name="foundation",
            payload=foundation,
            previous=(checkpoint_chain[-1] if checkpoint_chain else None),
            phase_package=phase_binding,
            foundation_package=None,
            provider_clock=provider_clock,
        )
        row = _package_checkpoint(
            checkpoint,
            scratch=paths.foundation_scratch,
            package=paths.foundation_package,
            repo=args.repo,
            stage="treatment_foundation",
        )
        foundation_binding = row["raw_package"]
        chain_row = {
            "sequence": len(checkpoint_chain) + 1,
            **row,
        }
        checkpoint_chain.append(chain_row)
        document["durable_checkpoints"] = list(checkpoint_chain)

    def persist_arm(index: int, arm: Mapping[str, Any]) -> None:
        require(1 <= index <= len(paths.arm_packages),
                "arm checkpoint index exceeds frozen allocation")
        checkpoint = _checkpoint_document(
            stage=f"treatment_arm_{index:02d}",
            regime=regime,
            repeat_index=repeat_index,
            payload_name="arm",
            payload=arm,
            previous=(checkpoint_chain[-1] if checkpoint_chain else None),
            phase_package=phase_binding,
            foundation_package=foundation_binding,
            provider_clock=provider_clock,
        )
        row = _package_checkpoint(
            checkpoint,
            scratch=paths.arm_scratch[index - 1],
            package=paths.arm_packages[index - 1],
            repo=args.repo,
            stage=f"treatment_arm_{index:02d}",
        )
        chain_row = {
            "sequence": len(checkpoint_chain) + 1,
            "arm_index": index,
            "selector": {
                key: arm.get(key) for key in ("schedule", "region", "cell")
            },
            **row,
        }
        checkpoint_chain.append(chain_row)
        document["durable_checkpoints"] = list(checkpoint_chain)

    try:
        phase = p01._stage(
            document,
            "phase_a",
            lambda: run_phase_a_case(
                prepared["model"], prepared["tokenizer"], case,
                eos_ids=eos_ids,
            ),
        )
        _assert_phase(phase, case_id=CASE_ID)
        document["phase_a"] = phase
        phase_checkpoint = _checkpoint_document(
            stage="phase_a",
            regime=regime,
            repeat_index=repeat_index,
            payload_name="phase_a",
            payload=phase,
            previous=None,
            phase_package=None,
            foundation_package=None,
            provider_clock=provider_clock,
        )
        phase_row = _package_checkpoint(
            phase_checkpoint,
            scratch=paths.phase_scratch,
            package=paths.phase_package,
            repo=args.repo,
            stage="phase_a",
        )
        phase_binding = phase_row["raw_package"]
        checkpoint_chain.append({"sequence": 1, **phase_row})
        document["durable_checkpoints"] = list(checkpoint_chain)

        treatment = p01._stage(
            document,
            "targeted_treatment",
            lambda: run_targeted_treatment_case(
                prepared["model"], prepared["tokenizer"], case,
                eos_ids=eos_ids,
                repeat_index=repeat_index,
                on_foundation=persist_foundation,
                on_arm=persist_arm,
            ),
        )
        assert_targeted_treatment(
            treatment, case_id=CASE_ID, repeat_index=repeat_index
        )
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

    document["completed_at_utc"] = p01.utc_now()
    document["provider_elapsed_seconds"] = provider_clock.elapsed()
    document["hourly_cost_usd"] = args.hourly_cost_usd
    document["wrapper_seconds_before_final_persistence"] = (
        time.monotonic() - started
    )
    package_binding, recovery, package_error = p01.persist_lossless_raw(
        document,
        scratch_raw=paths.raw_scratch,
        package=paths.raw_package,
        repo=args.repo,
    )
    durable_elapsed = time.monotonic() - started
    completion_status = (
        document["status"] if package_binding is not None else "PERSISTENCE_ERROR"
    )
    timing_document = {
        "schema": TIMING_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        **FORMAL_FLAGS,
        "timing_kind": "outcome",
        "regime": regime,
        "case_id": CASE_ID,
        "repeat_index": repeat_index,
        "completion_status": completion_status,
        "durable_elapsed_seconds": durable_elapsed,
        "measurement_start": "immediately_before_phase_a",
        "measurement_end": "verified_final_targeted_treatment_raw_package",
        "derived_artifacts_included": False,
        "raw_package": package_binding,
        "phase_package": phase_binding,
        "checkpoint_chain": checkpoint_chain,
        "recovery_raw_path": recovery,
        "persistence_error": package_error,
        "provider_elapsed_seconds": provider_clock.elapsed(),
        "created_at_utc": p01.utc_now(),
    }
    p01.atomic_create_json(paths.timing_receipt, timing_document)
    timing_binding = p01.binding(paths.timing_receipt, repo=args.repo)
    receipt = OutcomeReceipt(
        regime=regime,
        case_id=CASE_ID,
        repeat_index=repeat_index,
        completion_status=completion_status,
        durable_elapsed_seconds=durable_elapsed,
        raw_package=package_binding,
        phase_package=phase_binding,
        checkpoint_chain=tuple(checkpoint_chain),
        timing_receipt=timing_binding,
        recovery_raw_path=recovery,
    )
    if fatal_interrupt is not None:
        # Do not render here: all outcome-derived access remains forbidden
        # until the outer runner has durably frozen all four decisions.  The
        # verified raw package (or recovery raw) already preserves the work.
        raise fatal_interrupt
    return receipt


def _probe_compact(record: Mapping[str, Any]) -> dict[str, Any]:
    return p01._probe_compact(record)


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
                    key: p01.json_safe(arm[key]) for key in (
                        "arm_kind", "execution_kind", "schedule", "region",
                        "cell", "key_source", "value_source",
                        "control_status", "shared_fresh_baseline", "diagnostics",
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


def render_outcome_ledger(document: Mapping[str, Any]) -> str:
    rendered = p01.render_outcome_ledger(document)
    return rendered.replace("P01 render ledger", "P02 render ledger").replace(
        "precision-probe-p01", PROTOCOL_ID
    )


def materialize_derived(
    receipt: OutcomeReceipt, *, paths: OutcomePaths, args: argparse.Namespace
) -> OutcomeReceipt:
    """Reconstruct and parse only after all four decisions are immutable."""
    require(receipt.raw_package is not None,
            "cannot derive from absent outcome raw package")
    document = p01._load_packaged_document(
        paths.raw_package, scratch_dir=paths.raw_scratch.parent
    )
    document["raw_package_binding"] = dict(receipt.raw_package)
    p01.atomic_create_json(paths.compact, compact_outcome(document))
    p01.atomic_create_text(paths.renders, render_outcome_ledger(document))
    return replace(
        receipt,
        compact_path=str(paths.compact),
        renders_path=str(paths.renders),
    )


def _receipt_record(receipt: OutcomeReceipt, repo: Path) -> dict[str, Any]:
    """Return the literal manifest outcome-row contract consumed by ops."""
    return {
        "regime": receipt.regime,
        "case_id": receipt.case_id,
        "repeat_index": receipt.repeat_index,
        "completion_status": receipt.completion_status,
        "durable_elapsed_seconds": receipt.durable_elapsed_seconds,
        "raw_package": p01.json_safe(receipt.raw_package),
        "phase_package": p01.json_safe(receipt.phase_package),
        "checkpoint_chain": p01.json_safe(receipt.checkpoint_chain),
        "timing_receipt": p01.json_safe(receipt.timing_receipt),
        "recovery_raw_path": receipt.recovery_raw_path,
        "compact": (
            p01.binding(Path(receipt.compact_path), repo=repo)
            if receipt.compact_path else None
        ),
        "renders": (
            p01.binding(Path(receipt.renders_path), repo=repo)
            if receipt.renders_path else None
        ),
    }


def skipped_outcome(
    regime: str, repeat_index: int, status: str
) -> OutcomeReceipt:
    return OutcomeReceipt(
        regime=regime,
        case_id=CASE_ID,
        repeat_index=repeat_index,
        completion_status=status,
        durable_elapsed_seconds=0.0,
        raw_package=None,
        phase_package=None,
        checkpoint_chain=(),
        timing_receipt=None,
        recovery_raw_path=None,
    )


def skipped_technical(regime: str, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "regime": regime,
        "durable_elapsed_seconds": 0.0,
        "raw_package": None,
        "renders": None,
        "timing_receipt": None,
        "runtime_fingerprint": None,
        "subject_bindings": None,
        "recovery_raw_path": None,
        "persistence_error": None,
        "stage_timings": {},
        "durable_checkpoints": [],
    }


def _scientific_repeat_view(document: Mapping[str, Any]) -> dict[str, Any]:
    treatment = document.get("treatment", {})
    arms = treatment.get("arms", []) if isinstance(treatment, Mapping) else []
    return {
        "phase_a": document.get("phase_a"),
        "treatment": {
            "fresh_scores": treatment.get("fresh_scores")
            if isinstance(treatment, Mapping) else None,
            # FF plus six exact grafts; the repeat-1-only placebo is excluded.
            "arms": arms[:7] if isinstance(arms, list) else None,
        },
    }


def repeat_stability(
    receipts: Sequence[OutcomeReceipt],
    paths: Mapping[tuple[str, int], OutcomePaths],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for regime in REGIMES:
        pair = sorted(
            [row for row in receipts if row.regime == regime],
            key=lambda row: row.repeat_index,
        )
        if len(pair) != 2 or any(
            row.completion_status != "COMPLETE" or row.raw_package is None
            for row in pair
        ):
            result[regime] = {"status": "UNAVAILABLE_INCOMPLETE_REPEATS"}
            continue
        first = p01._load_packaged_document(
            paths[(regime, 1)].raw_package,
            scratch_dir=paths[(regime, 1)].raw_scratch.parent,
        )
        second = p01._load_packaged_document(
            paths[(regime, 2)].raw_package,
            scratch_dir=paths[(regime, 2)].raw_scratch.parent,
        )
        left = _scientific_repeat_view(first)
        right = _scientific_repeat_view(second)
        differences = p01._diff_paths(left, right)
        result[regime] = {
            "status": "PASS" if not differences else "FAIL",
            "normalized_scope": [
                "phase_a", "treatment.fresh_scores",
                "treatment.arms[FF+six_grafts]",
            ],
            "repeat1_sha256": hashlib.sha256(
                p01.canonical_json_bytes(left)
            ).hexdigest(),
            "repeat2_sha256": hashlib.sha256(
                p01.canonical_json_bytes(right)
            ).hexdigest(),
            "differing_field_count_at_most_500": len(differences),
            "differing_fields": differences,
            "difference_list_truncated": len(differences) >= 500,
        }
    return result


def _receipt_for(
    receipts: Sequence[OutcomeReceipt], regime: str, repeat_index: int
) -> OutcomeReceipt | None:
    rows = [row for row in receipts
            if row.regime == regime and row.repeat_index == repeat_index]
    require(len(rows) <= 1, "duplicate P02 outcome receipt")
    return rows[0] if rows else None


def matched_repeat1_eligible(
    receipts: Sequence[OutcomeReceipt],
    technical: Mapping[str, Mapping[str, Any]],
    runtime_gate: Mapping[str, Any],
) -> bool:
    if runtime_gate.get("status") != "PASS":
        return False
    if any(technical.get(regime, {}).get("status") != "PASS"
           for regime in REGIMES):
        return False
    for regime in REGIMES:
        row = _receipt_for(receipts, regime, 1)
        if row is None or not (
            row.completion_status == "COMPLETE"
            and isinstance(row.raw_package, Mapping)
            and row.raw_package.get("verification_status") == "VERIFIED"
            and isinstance(row.phase_package, Mapping)
            and row.phase_package.get("verification_status") == "VERIFIED"
            and row.recovery_raw_path is None
            and isinstance(row.compact_path, str)
            and Path(row.compact_path).is_file()
            and isinstance(row.renders_path, str)
            and Path(row.renders_path).is_file()
        ):
            return False
    return True


def repeat2_rider_status(
    initial_decision: Mapping[str, Any],
    bf16_repeat2_decision: Mapping[str, Any],
    receipts: Sequence[OutcomeReceipt],
) -> str:
    if not initial_decision.get("both_repeat2_authorized"):
        return "NOT_AUTHORIZED"
    nf4 = _receipt_for(receipts, "nf4", 2)
    bf16 = _receipt_for(receipts, "bf16", 2)
    if all(row is not None and row.completion_status == "COMPLETE"
           for row in (nf4, bf16)):
        return "MATCHED_COMPLETE"
    if (nf4 is not None and nf4.completion_status == "COMPLETE"
            and not bf16_repeat2_decision.get("authorized")):
        return "NF4_ONLY_TIMING_STOP"
    return "AUTHORIZED_INCOMPLETE"


def _inventory_tree(
    run_dir: Path, *, repo: Path, exclude: set[Path]
) -> list[dict[str, Any]]:
    return p01._inventory_tree(run_dir, repo=repo, exclude=exclude)


def _manifest_checkpoint(
    path: Path,
    document: dict[str, Any],
    clock: ProviderClock,
    args: argparse.Namespace,
) -> None:
    document["updated_at_utc"] = p01.utc_now()
    document["provider_elapsed_seconds"] = clock.elapsed()
    document["estimated_provider_cost_usd"] = (
        document["provider_elapsed_seconds"] * args.hourly_cost_usd / 3600.0
    )
    p01.atomic_replace_json(path, document)


def _decision_binding(
    *,
    name: str,
    document: Mapping[str, Any],
    planned: Mapping[str, Any],
    args: argparse.Namespace,
    manifest: dict[str, Any],
    clock: ProviderClock,
) -> dict[str, Any]:
    path = planned["decisions"][name]
    p01.atomic_create_json(path, document)
    bound = p01.binding(path, repo=args.repo)
    manifest["continuation_decisions"][name] = bound
    _manifest_checkpoint(planned["run_manifest"], manifest, clock, args)
    return bound


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any], int]:
    require(args.model == MODEL_ID, "model differs from frozen P02 subject")
    require(args.revision == REVISION,
            "revision differs from frozen P02 subject")
    prereg_sha, _ = p01.sha256_file(args.prereg)
    require(prereg_sha == PREREG_SHA256,
            "P02 preregistration bytes differ from frozen SHA-256")
    for label, path, expected_sha in (
        ("e01 case", args.case_e01, CASE_E01_SHA256),
        ("identity fixture", args.identity_fixture, IDENTITY_FIXTURE_SHA256),
        ("technical fixture", args.technical_fixture, TECHNICAL_FIXTURE_SHA256),
    ):
        observed_sha, _ = p01.sha256_file(path)
        require(observed_sha == expected_sha,
                f"P02 {label} bytes differ from frozen SHA-256")
    cap = scientific_cap_seconds(
        args.hourly_cost_usd, args.provider_wall_cap_seconds
    )
    require(
        _valid_elapsed(args.provider_elapsed_seconds_at_start),
        "provider elapsed seconds at start must be nonnegative",
    )
    require(args.provider_elapsed_seconds_at_start < cap,
            "scientific cap was exhausted before runner start")

    stamp = p01.utc_stamp()
    run_dir = args.output_dir / f"precision-probe-p02_{MODEL_SLUG}_{stamp}"
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
        "scientific_cap_seconds": cap,
    }, sort_keys=True), flush=True)

    clock = ProviderClock(
        elapsed_seconds_at_start=args.provider_elapsed_seconds_at_start,
        monotonic_at_start=time.monotonic(),
    )
    input_paths = {
        "preregistration": args.prereg,
        "identity_fixture": args.identity_fixture,
        "technical_fixture": args.technical_fixture,
        "case_e01": args.case_e01,
        "runner": Path(__file__).resolve(),
        "targeted_case": ROOT / "src/precision_probe_p02_case.py",
        "p01_runner_reused_technical": Path(p01.__file__).resolve(),
        "p01_subject_loader": ROOT / "src/precision_probe_p01_loader.py",
        "artifact_packager": Path(p01.artifact_packager.__file__).resolve(),
        "coherent_canary_case": ROOT / "src/coherent_canary_case.py",
        "coherent_canary_runtime": ROOT / "src/coherent_canary_runtime.py",
        "coherent_canary_technical": ROOT / "src/coherent_canary_technical.py",
        "coherent_canary_tokens": ROOT / "src/coherent_canary_tokens.py",
        "coherent_canary_controls": ROOT / "src/coherent_canary_controls.py",
        "coherent_canary_schema": ROOT / "src/coherent_canary_schema.py",
        "coherent_state_tokens": ROOT / "src/coherent_state_tokens.py",
    }
    common_bindings = {
        name: p01.binding(path, repo=args.repo)
        for name, path in input_paths.items()
    }
    manifest: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        **FORMAL_FLAGS,
        "status": "STARTED",
        "model": args.model,
        "revision": args.revision,
        "run_stamp_utc": stamp,
        "run_directory": p01.display_path(run_dir, args.repo),
        "started_at_utc": p01.utc_now(),
        "bindings": common_bindings,
        "provenance": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "repository_head": p01.git_value(args.repo, "rev-parse", "HEAD"),
            "repository_branch": p01.git_value(
                args.repo, "branch", "--show-current"
            ),
        },
        "provider": {
            "hourly_cost_usd": args.hourly_cost_usd,
            "elapsed_seconds_at_runner_start": (
                args.provider_elapsed_seconds_at_start
            ),
            "provider_wall_cap_seconds": args.provider_wall_cap_seconds,
            "scientific_cap_seconds": cap,
            "scientific_budget_usd": SCIENTIFIC_BUDGET_USD,
            "forecast_reserve_seconds": FORECAST_RESERVE_SECONDS,
        },
        "planned_output_names": p01.json_safe({
            "completion": completion_path,
            "decisions": planned["decisions"],
            "technical": planned["technical"],
            "outcomes": {
                f"{regime}/e01/repeat{repeat}": asdict(paths)
                for (regime, repeat), paths in planned["outcomes"].items()
            },
        }),
        "continuation_decisions": {},
        "regimes": {
            regime: {"status": "PENDING", "technical": None, "outcomes": []}
            for regime in REGIMES
        },
        "matched_runtime_gate": {
            "status": "NOT_EVALUATED", "differing_fields": []
        },
        "repeat_stability": {},
    }
    _manifest_checkpoint(manifest_path, manifest, clock, args)

    receipts: list[OutcomeReceipt] = []
    nf4_prepared: dict[str, Any] | None = None
    bf16_prepared: dict[str, Any] | None = None
    initial_decision: dict[str, Any] | None = None
    bf16_r2_decision: dict[str, Any] | None = None
    fatal_interrupt: BaseException | None = None

    def refresh_outcomes(regime: str) -> None:
        manifest["regimes"][regime]["outcomes"] = [
            _receipt_record(row, args.repo)
            for row in receipts if row.regime == regime
        ]
        _manifest_checkpoint(manifest_path, manifest, clock, args)

    try:
        nf4_technical_forecast = (
            clock.elapsed()
            + FORECAST_MULTIPLIER * TECHNICAL_FLOOR_SECONDS
        )
        manifest["initial_nf4_technical_forecast_seconds"] = (
            nf4_technical_forecast
        )
        if nf4_technical_forecast <= cap - FORECAST_RESERVE_SECONDS:
            nf4_prepared, nf4_technical = run_regime_technical(
                regime="nf4",
                args=args,
                paths=planned["technical"]["nf4"],
                bindings=common_bindings,
                provider_clock=clock,
            )
        else:
            nf4_technical = skipped_technical(
                "nf4", "SKIPPED_INITIAL_STAGE_FORECAST"
            )
        manifest["regimes"]["nf4"]["technical"] = nf4_technical
        _manifest_checkpoint(manifest_path, manifest, clock, args)

        if nf4_prepared is None:
            nf4_r1 = skipped_outcome("nf4", 1, "SKIPPED_TECHNICAL_FAIL")
        elif (clock.elapsed()
              + FORECAST_MULTIPLIER * INITIAL_OUTCOME_FORECAST_SECONDS
              > cap - FORECAST_RESERVE_SECONDS):
            nf4_r1 = skipped_outcome(
                "nf4", 1, "SKIPPED_INITIAL_OUTCOME_FORECAST"
            )
        else:
            nf4_r1 = execute_outcome(
                regime="nf4",
                repeat_index=1,
                prepared=nf4_prepared,
                args=args,
                paths=planned["outcomes"][("nf4", 1)],
                common_bindings=common_bindings,
                provider_clock=clock,
            )
        receipts.append(nf4_r1)
        refresh_outcomes("nf4")

        initial_decision = decide_initial_schedule(
            nf4_technical=_technical_timing(nf4_technical),
            nf4_repeat1=nf4_r1.decision_timing(),
            provider_elapsed_seconds=clock.elapsed(),
            scientific_cap=cap,
        )
        _decision_binding(
            name="initial_schedule",
            document=initial_decision,
            planned=planned,
            args=args,
            manifest=manifest,
            clock=clock,
        )

        if initial_decision["both_repeat2_authorized"] and nf4_prepared:
            nf4_r2 = execute_outcome(
                regime="nf4",
                repeat_index=2,
                prepared=nf4_prepared,
                args=args,
                paths=planned["outcomes"][("nf4", 2)],
                common_bindings=common_bindings,
                provider_clock=clock,
            )
        else:
            nf4_r2 = skipped_outcome(
                "nf4", 2,
                "SKIPPED_NOT_AUTHORIZED" if nf4_prepared
                else "SKIPPED_TECHNICAL_FAIL",
            )
        receipts.append(nf4_r2)
        refresh_outcomes("nf4")

        bf16_admission = decide_bf16_technical_admission(
            initial_matched_authorized=bool(
                initial_decision["matched_repeat1_authorized"]
            ),
            nf4_technical=_technical_timing(nf4_technical),
            provider_elapsed_seconds=clock.elapsed(),
            scientific_cap=cap,
        )
        _decision_binding(
            name="bf16_technical_admission",
            document=bf16_admission,
            planned=planned,
            args=args,
            manifest=manifest,
            clock=clock,
        )
        manifest["regimes"]["nf4"]["status"] = "OUTCOMES_FINISHED"
        p01._release_subject(nf4_prepared)
        nf4_prepared = None
        _manifest_checkpoint(manifest_path, manifest, clock, args)

        if bf16_admission["authorized"]:
            bf16_prepared, bf16_technical = run_regime_technical(
                regime="bf16",
                args=args,
                paths=planned["technical"]["bf16"],
                bindings=common_bindings,
                provider_clock=clock,
            )
            manifest["regimes"]["bf16"]["status"] = "TECHNICAL_FINISHED"
        else:
            bf16_technical = skipped_technical(
                "bf16", "SKIPPED_TIMING_ADMISSION"
            )
            manifest["regimes"]["bf16"]["status"] = "TIMING_ABORTED"
        manifest["regimes"]["bf16"]["technical"] = bf16_technical

        runtime_gate = p01.matched_runtime_gate(
            nf4_technical, bf16_technical
        )
        manifest["matched_runtime_gate"] = runtime_gate
        if runtime_gate.get("status") != "PASS":
            p01._release_subject(bf16_prepared)
            bf16_prepared = None
        _manifest_checkpoint(manifest_path, manifest, clock, args)

        bf16_r1_decision = decide_bf16_repeat1(
            initial_matched_authorized=bool(
                initial_decision["matched_repeat1_authorized"]
            ),
            nf4_repeat1=nf4_r1.decision_timing(),
            bf16_technical=_technical_timing(bf16_technical),
            matched_runtime_status=str(runtime_gate.get("status")),
            provider_elapsed_seconds=clock.elapsed(),
            scientific_cap=cap,
        )
        _decision_binding(
            name="bf16_repeat1",
            document=bf16_r1_decision,
            planned=planned,
            args=args,
            manifest=manifest,
            clock=clock,
        )

        if bf16_r1_decision["authorized"] and bf16_prepared:
            bf16_r1 = execute_outcome(
                regime="bf16",
                repeat_index=1,
                prepared=bf16_prepared,
                args=args,
                paths=planned["outcomes"][("bf16", 1)],
                common_bindings=common_bindings,
                provider_clock=clock,
            )
        else:
            bf16_r1 = skipped_outcome(
                "bf16", 1,
                "SKIPPED_TIMING_OR_GATE",
            )
        receipts.append(bf16_r1)
        refresh_outcomes("bf16")

        bf16_r2_decision = decide_bf16_repeat2(
            initial_repeat2_authorized=bool(
                initial_decision["both_repeat2_authorized"]
            ),
            nf4_repeat1=nf4_r1.decision_timing(),
            nf4_repeat2=nf4_r2.decision_timing(),
            bf16_repeat1=bf16_r1.decision_timing(),
            provider_elapsed_seconds=clock.elapsed(),
            scientific_cap=cap,
        )
        _decision_binding(
            name="bf16_repeat2",
            document=bf16_r2_decision,
            planned=planned,
            args=args,
            manifest=manifest,
            clock=clock,
        )

        if bf16_r2_decision["authorized"] and bf16_prepared:
            bf16_r2 = execute_outcome(
                regime="bf16",
                repeat_index=2,
                prepared=bf16_prepared,
                args=args,
                paths=planned["outcomes"][("bf16", 2)],
                common_bindings=common_bindings,
                provider_clock=clock,
            )
        else:
            bf16_r2 = skipped_outcome(
                "bf16", 2, "SKIPPED_NOT_AUTHORIZED"
            )
        receipts.append(bf16_r2)
        manifest["regimes"]["bf16"]["status"] = (
            "OUTCOMES_FINISHED"
            if bf16_admission["authorized"] else "TIMING_ABORTED"
        )
        refresh_outcomes("bf16")
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
        p01._release_subject(nf4_prepared)
        p01._release_subject(bf16_prepared)

        # Even an error freezes the remaining decisions as NOT_REACHED.  This
        # uses no raw package and does not derive or inspect an outcome value.
        for name in DECISION_NAMES:
            if name not in manifest["continuation_decisions"]:
                try:
                    _decision_binding(
                        name=name,
                        document=skipped_decision(
                            name, "earlier execution error or timing stop"
                        ),
                        planned=planned,
                        args=args,
                        manifest=manifest,
                        clock=clock,
                    )
                except BaseException as decision_exc:
                    manifest["status"] = "ERROR"
                    manifest.setdefault("finalization_errors", []).append({
                        "type": type(decision_exc).__name__,
                        "message": str(decision_exc),
                    })

        # The raw/derived boundary opens only after all four immutable decision
        # files exist.  Fatal interrupts retain raw artifacts and propagate.
        all_decisions_frozen = (
            set(manifest["continuation_decisions"]) == set(DECISION_NAMES)
        )
        if all_decisions_frozen and fatal_interrupt is None:
            updated: list[OutcomeReceipt] = []
            for receipt in receipts:
                if receipt.raw_package is not None:
                    try:
                        receipt = materialize_derived(
                            receipt,
                            paths=planned["outcomes"][
                                (receipt.regime, receipt.repeat_index)
                            ],
                            args=args,
                        )
                    except BaseException as exc:
                        manifest["status"] = "ERROR"
                        manifest.setdefault("finalization_errors", []).append({
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "regime": receipt.regime,
                            "repeat_index": receipt.repeat_index,
                        })
                updated.append(receipt)
            receipts = updated
            for regime in REGIMES:
                manifest["regimes"][regime]["outcomes"] = [
                    _receipt_record(row, args.repo)
                    for row in receipts if row.regime == regime
                ]
            try:
                manifest["repeat_stability"] = repeat_stability(
                    receipts, planned["outcomes"]
                )
            except BaseException as exc:
                manifest["repeat_stability"] = {
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }

        if initial_decision is None:
            initial_decision = {"both_repeat2_authorized": False}
        if bf16_r2_decision is None:
            bf16_r2_decision = {"authorized": False}
        rider_status = repeat2_rider_status(
            initial_decision, bf16_r2_decision, receipts
        )
        require(rider_status in RIDER_STATUSES,
                "repeat-2 rider status is outside frozen vocabulary")
        runtime_gate = manifest.get("matched_runtime_gate", {})
        technical_records = {
            regime: manifest["regimes"][regime].get("technical") or {}
            for regime in REGIMES
        }
        eligible = matched_repeat1_eligible(
            receipts, technical_records, runtime_gate
        )
        recovery_raw_files = [
            path.relative_to(run_dir).as_posix()
            for path in sorted((run_dir / ".scratch").rglob("*.json"))
        ]
        completion_elapsed = clock.elapsed()
        estimated_cost = completion_elapsed * args.hourly_cost_usd / 3600.0
        if manifest.get("status") != "ERROR":
            manifest["status"] = (
                "COMPLETE" if (
                    eligible
                    and not recovery_raw_files
                    and completion_elapsed <= cap
                    and estimated_cost <= SCIENTIFIC_BUDGET_USD
                ) else "PARTIAL"
            )
        manifest["matched_repeat1_eligible"] = eligible
        manifest["repeat2_rider_status"] = rider_status
        manifest["completed_at_utc"] = p01.utc_now()
        _manifest_checkpoint(manifest_path, manifest, clock, args)

        completion_elapsed = clock.elapsed()
        estimated_cost = completion_elapsed * args.hourly_cost_usd / 3600.0
        if (manifest["status"] == "COMPLETE"
                and (completion_elapsed > cap
                     or estimated_cost > SCIENTIFIC_BUDGET_USD)):
            manifest["status"] = "PARTIAL"
            _manifest_checkpoint(manifest_path, manifest, clock, args)
        completion = {
            "schema": COMPLETION_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            **FORMAL_FLAGS,
            "status": manifest["status"],
            "model": args.model,
            "revision": args.revision,
            "created_at_utc": p01.utc_now(),
            "scientific_cap_seconds": cap,
            "matched_repeat1_eligible": eligible,
            "repeat2_rider_status": rider_status,
            "matched_runtime_gate": runtime_gate,
            "run_manifest": p01.binding(manifest_path, repo=args.repo),
            "continuation_decisions": dict(
                manifest["continuation_decisions"]
            ),
            "artifact_inventory": _inventory_tree(
                run_dir, repo=args.repo, exclude={completion_path}
            ),
            "recovery_raw_files": recovery_raw_files,
            "provider_elapsed_seconds": completion_elapsed,
            "estimated_provider_cost_usd": estimated_cost,
        }
        p01.atomic_create_json(completion_path, completion)

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
    parser.add_argument("--case-e01", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--hourly-cost-usd", required=True, type=float)
    parser.add_argument(
        "--provider-elapsed-seconds-at-start", required=True, type=float,
        help="provider wall seconds already consumed before this runner began",
    )
    parser.add_argument(
        "--provider-wall-cap-seconds", type=float, default=9000.0,
        help="operational cap; it may only tighten the frozen C formula",
    )
    args = parser.parse_args(argv)
    args.repo = args.repo.resolve()
    args.output_dir = args.output_dir.resolve()
    args.prereg = args.prereg.resolve()
    args.identity_fixture = args.identity_fixture.resolve()
    args.technical_fixture = args.technical_fixture.resolve()
    args.case_e01 = args.case_e01.resolve()
    return args


def main() -> None:
    _, _, code = run(parse_args())
    raise SystemExit(code)


if __name__ == "__main__":
    main()
