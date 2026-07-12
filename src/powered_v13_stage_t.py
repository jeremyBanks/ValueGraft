"""Closed Stage-T technical runner for powered successor v13.

The production entry point reaches exactly two semantic-N=0 inputs in the
frozen order ``technical_e01`` then ``technical_long``.  It owns no provider
allocation, watchdog clock, production pool, entropy, selector, or analysis
surface.  A parent process supervises one worker so an errored worker is dead
before the existing store quarantine primitive is invoked.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

import torch

from coherent_canary_runtime import (
    _run_events,
    append_block_to_snapshot,
    continue_fresh_plan,
    execute_fresh_plan,
    execute_replay_plan,
    extract_rows,
    force_content_q1,
    greedy_generate_q1,
    model_device,
    probe_suffix_ids,
    require_generated_forced_identity,
    score_target_from_prefix_q1,
    snapshot_hashes,
    tensor_sha256,
)
from powered_v13_bundle import (
    build_descriptor,
    build_placebo_receipt,
    file_sha256 as bundle_file_sha256,
    load_verified_bundle,
    materialize_verified_bundle,
    reconstruct_arm_boundary,
    save_bundle,
    snapshot_sha256,
)
from powered_v13_ladder import run_stage_t_ladder
from powered_v13_schema import (
    DESIGN_ID, MODEL_REVISION, N_SCHEDULE, PRIMARY_ARMS, R2,
)
from powered_v13_store import (
    ACTIVE_LOCK_NAME,
    TERMINAL_EVIDENCE_SCHEMA,
    active_evidence_chain_sha256,
    append_case_record,
    begin_case,
    build_bounds,
    build_identity,
    file_sha256 as store_file_sha256,
    finalize_case,
    linux_process_start_token,
    quarantine_abandoned_case,
)
from powered_v13_subject import MODEL_ID, open_exact_subject
from powered_v13_technical import (
    TECHNICAL_CASE_IDS,
    build_technical_case,
    canonical_json_bytes as technical_json_bytes,
)
from powered_v13_tokens import probe_target_ids


SCHEMA = "coherent-state-powered-successor-v13-stage-t-runner-v1"
RUN_SUMMARY_SCHEMA = "coherent-state-powered-successor-v13-stage-t-summary-v1"
FOUNDATION_RECEIPT_SCHEMA = (
    "coherent-state-powered-successor-v13-stage-t-foundation-receipt-v1")
ARM_ARTIFACT_SCHEMA = "coherent-state-powered-successor-v13-stage-t-arm-v1"
ARM_CHECKPOINT_SCHEMA = (
    "coherent-state-powered-successor-v13-stage-t-arm-checkpoint-v1")
GATE_ARTIFACT_SCHEMA = "coherent-state-powered-successor-v13-stage-t-gate-v1"
RUN_ERROR_SCHEMA = "coherent-state-powered-successor-v13-stage-t-error-v1"
SUPERVISOR_SCHEMA = "coherent-state-powered-successor-v13-stage-t-supervisor-v1"

TECHNICAL_CASE_ORDER = ("technical_e01", "technical_long")
RENDER_ID = "r1"
CASE_DEADLINES_SECONDS = {
    "technical_e01": 1200,
    "technical_long": 1800,
}
MAX_GENERATED_PROBE_TOKENS = 64
MODEL_SLUG = re.sub(r"[^A-Za-z0-9._-]+", "-", MODEL_ID.split("/")[-1])
DEFAULT_OUTPUT_PARENT = Path("results/coherent_state_powered_v13")
VALIDATOR_RELATIVE_PATH = Path(
    "scripts/validate_powered_v13_technical_terminal.py")

# Literal names are required for the dynamic sys.modules stop gate.  They are
# not imported, discovered, or used to materialize any scientific input.
FORBIDDEN_PRODUCTION_MODULES = (
    "powered_v13_pool",
    "powered_v13_recipe",
    "powered_v13_permutation",
    "powered_v13_stimuli",
    "powered_v13_stats",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


class V13StageTRunnerError(RuntimeError):
    """The technical runner crossed or failed one frozen boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V13StageTRunnerError(message)


def _sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None,
             f"{label} is not lowercase SHA-256")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(technical_json_bytes(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _precise_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _compact_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Write canonical JSON+LF exclusively, fsync, reread, and hash it."""

    raw = technical_json_bytes(dict(value)) + b"\n"
    path = Path(path)
    _require(not path.exists() and not path.is_symlink(),
             f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)
    observed = path.read_bytes()
    _require(observed == raw, f"artifact readback differs: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_bytes(observed),
        "size_bytes": len(observed),
    }


def _safe_relative(path: Path, root: Path) -> str:
    path = Path(path)
    root = Path(root)
    _require(path.is_file() and not path.is_symlink(),
             f"bound artifact is absent or symlinked: {path}")
    try:
        relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise V13StageTRunnerError(
            f"bound artifact escapes run directory: {path}") from exc
    pure = PurePosixPath(relative.as_posix())
    _require(not pure.is_absolute() and ".." not in pure.parts,
             "bound artifact path is unsafe")
    return pure.as_posix()


def _artifact_binding(path: Path, *, root: Path, kind: str) -> dict[str, Any]:
    path = Path(path)
    relative = _safe_relative(path, root)
    size = path.stat().st_size
    _require(size > 0, f"bound artifact is empty: {relative}")
    return {
        "kind": kind,
        "path": relative,
        "sha256": store_file_sha256(path),
        "size_bytes": size,
    }


def _assert_import_boundary() -> None:
    loaded = sorted(
        name for name in sys.modules
        if any(name == stem or name.startswith(stem + ".")
               for stem in FORBIDDEN_PRODUCTION_MODULES)
    )
    _require(not loaded,
             f"forbidden Stage-T production modules are loaded: {loaded}")


def _create_unique_output_directory(
    parent: Path, *, compact_utc: Callable[[], str] = _compact_utc_now,
) -> Path:
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    _require(parent.is_dir() and not parent.is_symlink(),
             "Stage-T output parent is not a real directory")
    timestamp = compact_utc()
    _require(isinstance(timestamp, str)
             and re.fullmatch(r"[0-9]{8}T[0-9]{12}Z", timestamp) is not None,
             "compact UTC timestamp differs")
    output = parent / f"powered-v13-stage-t_{MODEL_SLUG}_{timestamp}"
    try:
        output.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise V13StageTRunnerError(
            f"refusing to reuse Stage-T output directory: {output}") from exc
    _fsync_directory(parent)
    return output


def _cuda_synchronize() -> None:
    _require(torch.cuda.is_available(), "Stage-T timing requires CUDA")
    torch.cuda.synchronize()


def _reset_peak_vram() -> None:
    _require(torch.cuda.is_available(), "Stage-T VRAM measurement requires CUDA")
    torch.cuda.reset_peak_memory_stats()


def _vram_metrics() -> dict[str, Any]:
    _require(torch.cuda.is_available(), "Stage-T VRAM measurement requires CUDA")
    device = torch.cuda.current_device()
    return {
        "device": str(torch.device("cuda", device)),
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "max_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)),
    }


def _evict_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class StageTConfig:
    repo: Path
    receipt_directory: Path
    run_directory: Path
    primary_batch_id: str


@dataclass
class FoundationState:
    descriptor: dict[str, Any]
    materialized: Any
    vp_receipt: dict[str, Any]
    vp_receipt_sha256: str
    vp_diagnostics: dict[str, Any]
    vp_status: str
    bundle_binding: dict[str, Any]
    descriptor_binding: dict[str, Any]
    receipt_binding: dict[str, Any]
    selected_row_count: int
    live_token_count: int
    r2_end: int


@dataclass(frozen=True)
class StageTSeams:
    open_subject: Callable[..., Any]
    prepare_case: Callable[..., dict[str, Any]]
    run_ladder: Callable[..., dict[str, Any]]
    run_long_identity: Callable[..., dict[str, Any]]
    capture_foundation: Callable[..., FoundationState]
    run_arm: Callable[..., dict[str, Any]]
    process_start_token: Callable[[int], str]
    build_store_identity: Callable[..., dict[str, Any]]
    build_store_bounds: Callable[..., dict[str, Any]]
    begin_store_case: Callable[..., dict[str, Any]]
    append_store_record: Callable[..., dict[str, Any]]
    active_chain_sha256: Callable[..., str]
    invoke_terminal_validator: Callable[..., dict[str, Any]]
    finalize_store_case: Callable[..., dict[str, Any]]
    utc_now: Callable[[], str]
    precise_utc_now: Callable[[], str]
    monotonic: Callable[[], float]
    cuda_synchronize: Callable[[], None]
    reset_peak_vram: Callable[[], None]
    vram_metrics: Callable[[], dict[str, Any]]
    evict_cuda: Callable[[], None]


class StageRecorder:
    """GPU-synchronized timings and VRAM snapshots; never a provider clock."""

    def __init__(self, seams: StageTSeams) -> None:
        self.seams = seams
        self.records: list[dict[str, Any]] = []

    def measure(self, label: str, operation: Callable[[], Any]) -> Any:
        self.seams.cuda_synchronize()
        self.seams.reset_peak_vram()
        before = self.seams.vram_metrics()
        started_utc = self.seams.precise_utc_now()
        started = self.seams.monotonic()
        status = "ERROR"
        error_type = None
        try:
            value = operation()
            status = "PASS"
            return value
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            self.seams.cuda_synchronize()
            elapsed = self.seams.monotonic() - started
            _require(math.isfinite(elapsed) and elapsed >= 0,
                     f"stage {label} timing is invalid")
            self.records.append({
                "stage": label,
                "status": status,
                "error_type": error_type,
                "started_utc": started_utc,
                "completed_utc": self.seams.precise_utc_now(),
                "wall_time_seconds": elapsed,
                "vram_before": before,
                "vram_after": self.seams.vram_metrics(),
            })


def _timing_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(records), "timing summary has no measured stages")
    elapsed = [float(row["wall_time_seconds"]) for row in records]
    _require(all(math.isfinite(value) and value >= 0 for value in elapsed),
             "timing summary contains an invalid wall time")
    allocated = [
        int(row["vram_after"]["max_memory_allocated_bytes"])
        for row in records
    ]
    reserved = [
        int(row["vram_after"]["max_memory_reserved_bytes"])
        for row in records
    ]
    _require(all(value >= 0 for value in allocated + reserved),
             "timing summary contains invalid peak VRAM")
    allocated_deltas = [
        int(row["vram_after"]["max_memory_allocated_bytes"])
        - int(row["vram_before"]["memory_allocated_bytes"])
        for row in records
    ]
    _require(all(value >= 0 for value in allocated_deltas),
             "timing summary contains invalid peak-VRAM delta")
    return {
        "measured_stage_count": len(records),
        "max_memory_allocated_bytes": max(allocated),
        "max_memory_allocated_delta_bytes": max(allocated_deltas),
        "max_memory_reserved_bytes": max(reserved),
        "peak_stats_reset_before_each_stage": True,
    }


def _execution_record(result: Any) -> dict[str, Any]:
    return {
        "calls": list(result.calls),
        "q1_token_logprobs": list(result.q1_token_logprobs),
        "executed_token_ids": list(result.executed_token_ids),
        "logical_positions": list(result.logical_positions),
        "physical_positions": list(result.physical_positions),
        "physical_end": int(result.physical_end),
        "logical_end": int(result.logical_end),
        "snapshot_hashes": snapshot_hashes(result.snapshot),
        "last_logits_sha256": tensor_sha256(result.last_logits),
    }


def _generation_record(value: Any) -> dict[str, Any]:
    return {
        "content_ids": list(value.content_ids),
        "token_logprobs": list(value.token_logprobs),
        "token_logprob_float32_bits": list(
            value.token_logprob_float32_bits),
        "logical_positions": list(value.logical_positions),
        "physical_positions": list(value.physical_positions),
        "stop_reason": value.stop_reason,
        "stop_candidate_id": int(value.stop_candidate_id),
        "stop_candidate_logprob": float(value.stop_candidate_logprob),
        "stop_candidate_logprob_float32_bits":
            value.stop_candidate_logprob_float32_bits,
        "eos_ids": list(value.eos_ids),
        "cap_hit": bool(value.cap_hit),
    }


def _float32_from_bits(value: str) -> float:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{8}", value),
             "float32 bit witness differs")
    return struct.unpack("<f", bytes.fromhex(value))[0]


def _margin_record(correct: Mapping[str, Any], counter: Mapping[str, Any]) \
        -> dict[str, Any]:
    left = _float32_from_bits(correct["mean_logprob_float32_bits"])
    right = _float32_from_bits(counter["mean_logprob_float32_bits"])
    bits = struct.pack("<f", left - right).hex()
    value = _float32_from_bits(bits)
    _require(math.isfinite(value), "technical probe margin is nonfinite")
    return {"value": value, "float32_bits": bits}


def _snapshot_to_cpu(snapshot: Sequence[tuple[torch.Tensor, torch.Tensor]]):
    before = snapshot_sha256(snapshot)
    result = tuple(
        (keys.detach().to(device="cpu").clone().contiguous(),
         values.detach().to(device="cpu").clone().contiguous())
        for keys, values in snapshot)
    _require(snapshot_sha256(result) == before,
             "GPU-to-CPU foundation transfer changed bits")
    return result


def _prepare_technical_case(tokenizer: Any, repo: Path, case_id: str) \
        -> dict[str, Any]:
    _require(case_id in TECHNICAL_CASE_ORDER, "runner technical case differs")
    case = build_technical_case(tokenizer, repo, case_id)
    _require(case.get("case_id") == case_id and case.get("semantic_n") == 0
             and case.get("production_pool_imported") is False
             and case.get("production_entropy_requested") is False
             and case.get("production_probe_reachable") is False
             and case.get("score_values_present") is False,
             f"{case_id} technical input boundary differs")
    probe = case["technical_probe"]
    fresh = case["plan_objects"]["F"]
    context = case["context_messages"]["F"]
    suffix = probe_suffix_ids(
        tokenizer, context, fresh.token_ids, str(probe["probe"]))
    correct = probe_target_ids(
        tokenizer, context, str(probe["probe"]),
        str(probe["correct_target"]))
    counter = probe_target_ids(
        tokenizer, context, str(probe["probe"]),
        str(probe["counterfactual_target"]))
    _require(bool(suffix) and bool(correct) and bool(counter)
             and correct != counter,
             f"{case_id} technical probe token contract differs")
    result = dict(case)
    result["probe_runtime"] = {
        "suffix_ids": list(suffix),
        "suffix_ids_sha256": _sha256_json(list(suffix)),
        "correct_target_ids": list(correct),
        "correct_target_ids_sha256": _sha256_json(list(correct)),
        "counterfactual_target_ids": list(counter),
        "counterfactual_target_ids_sha256": _sha256_json(list(counter)),
    }
    execution_plans: dict[str, Any] = {}
    for history in ("F", "C", "W"):
        plan = case["plan_objects"][history]
        plan.validate()
        token_ids = list(plan.token_ids)
        geometry = plan.geometry()
        _require(_sha256_json(token_ids) ==
                 case["plans"][history]["token_ids_sha256"]
                 and _sha256_json(geometry) ==
                 case["plans"][history]["geometry_sha256"],
                 f"{case_id} {history} execution plan binding differs")
        logical_positions = (
            list(plan.logical_positions)
            if hasattr(plan, "logical_positions")
            else list(range(len(token_ids))))
        physical_positions = (
            list(plan.physical_positions)
            if hasattr(plan, "physical_positions")
            else list(range(len(token_ids))))
        execution_plans[history] = {
            "token_ids": token_ids,
            "logical_positions": logical_positions,
            "physical_positions": physical_positions,
            "geometry": geometry,
            "token_origin": (
                "fixed_technical_compact_visible_history" if history == "F"
                else f"fixed_technical_{history}_source_history"),
        }
    context_messages = {
        history: case["context_messages"][history]
        for history in ("F", "C", "W")
    }
    _require(all(
        _sha256_json(context_messages[history]) ==
        case["context_messages_sha256"][history]
        for history in ("F", "C", "W")),
        f"{case_id} context-message binding differs")
    execution_input = {
        "context_messages": context_messages,
        "plans": execution_plans,
        "every_token_origin_persisted": True,
    }
    result["execution_input"] = execution_input
    result["execution_input_sha256"] = _sha256_json(execution_input)
    return result


def _public_case_evidence(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in case.items()
        if key not in {"plan_objects", "context_messages"}
    }


def _terminal_signature(result: Any) -> dict[str, Any]:
    return {
        "snapshot_hashes": snapshot_hashes(result.snapshot),
        "last_logits_sha256": tensor_sha256(result.last_logits),
        "physical_end": int(result.physical_end),
        "logical_end": int(result.logical_end),
    }


def _replay_resume_q1_witness(
    boundary: Any, replay_plan: Any, start_at: int,
) -> dict[str, Any] | None:
    event = next(
        (candidate for candidate in replay_plan.events
         if candidate.token_start == start_at),
        None)
    _require(event is not None, "replay resume event is absent")
    if event.kind != "q1":
        return None
    token_id = int(replay_plan.token_ids[start_at])
    logprob = torch.log_softmax(
        boundary.last_logits.float(), dim=-1)[0, token_id]
    value = float(logprob.detach().cpu())
    _require(math.isfinite(value), "replay resume q1 log probability is nonfinite")
    return {
        "physical_position": start_at,
        "logical_position": start_at,
        "token_id": token_id,
        "logprob": value,
        "logprob_float32_bits": struct.pack("<f", value).hex(),
    }


def _run_long_identity(model: Any, replay_plan: Any) -> dict[str, Any]:
    """Only the frozen long L0 repeat and R2-stop/rebuild L3 identity."""

    replay_plan.validate()
    token_count = len(replay_plan.token_ids)
    _require(4000 <= token_count <= 5000,
             "technical-long identity did not receive 4.5k replay geometry")
    _r2_start, r2_end = replay_plan.regions.interval(R2)
    first = execute_replay_plan(model, replay_plan)
    first_record = _execution_record(first)
    first_terminal = _terminal_signature(first)
    repeat = execute_replay_plan(model, replay_plan)
    repeat_record = _execution_record(repeat)
    _require(first_record == repeat_record,
             "technical-long deterministic repeat differs")
    del first, repeat
    _evict_cuda()

    boundary = execute_replay_plan(model, replay_plan, stop_at=r2_end)
    # There is intentionally no second replay implementation here.  This is
    # the exact event engine underlying execute_replay_plan, resumed from the
    # persisted R2 cache so the long identity keeps identical call widths.
    resume_q1 = _replay_resume_q1_witness(boundary, replay_plan, r2_end)
    continued = _run_events(
        model, replay_plan.token_ids, replay_plan.events,
        stop_at=token_count, destination=False,
        initial_snapshot=boundary.snapshot)
    if resume_q1 is not None:
        continued.q1_token_logprobs.insert(0, resume_q1)
    boundary_record = _execution_record(boundary)
    continued_record = _execution_record(continued)
    _require(first_record["calls"] == boundary.calls + continued.calls
             and first_record["executed_token_ids"] ==
             boundary.executed_token_ids + continued.executed_token_ids
             and first_record["logical_positions"] ==
             boundary.logical_positions + continued.logical_positions
             and first_record["physical_positions"] ==
             boundary.physical_positions + continued.physical_positions
             and first_record["q1_token_logprobs"] ==
             boundary_record["q1_token_logprobs"] +
             continued_record["q1_token_logprobs"]
             and first_terminal == _terminal_signature(continued),
             "technical-long R2 boundary continuation differs")
    del boundary, continued
    _evict_cuda()
    return {
        "schema": "coherent-state-powered-successor-v13-stage-t-long-identity-v1",
        "status": "PASS",
        "schedule": N_SCHEDULE,
        "token_count": token_count,
        "r2_physical_end": r2_end,
        "l0_first": first_record,
        "l0_repeat": repeat_record,
        "l3_boundary": boundary_record,
        "l3_continuation": continued_record,
    }


def _release_fields(fingerprint: Mapping[str, Any]) -> tuple[str, str, str]:
    release = fingerprint.get("release_binding")
    cuda = fingerprint.get("cuda")
    _require(isinstance(release, Mapping) and isinstance(cuda, Mapping),
             "runtime fingerprint release/CUDA binding is absent")
    return (
        _sha(release.get("release_evidence_sha256"), "release evidence"),
        _sha(fingerprint.get("fingerprint_sha256"), "runtime fingerprint"),
        str(cuda.get("gpu_uuid")),
    )


def _capture_foundation(
    config: StageTConfig,
    handle: Any,
    case: Mapping[str, Any],
    input_binding: Mapping[str, Any],
    gate_binding: Mapping[str, Any],
    recorder: StageRecorder,
) -> FoundationState:
    case_id = str(case["case_id"])
    timing_index = len(recorder.records)
    model = handle.model
    plans = case["plan_objects"]
    evidence = case["plans"]
    c_start, c_end = plans["C"].regions.interval(R2)
    w_start, w_end = plans["W"].regions.interval(R2)
    f_start, f_end = plans["F"].physical_regions.interval(R2)
    _require((c_start, c_end) == (w_start, w_end)
             and c_end - c_start == f_end - f_start,
             f"{case_id} source/fresh R2 widths differ")

    c_result = recorder.measure(
        f"{case_id}.source_C_R2_replay",
        lambda: execute_replay_plan(model, plans["C"], stop_at=c_end))
    w_result = recorder.measure(
        f"{case_id}.source_W_R2_replay",
        lambda: execute_replay_plan(model, plans["W"], stop_at=w_end))
    f_result = recorder.measure(
        f"{case_id}.fresh_F_R2_replay",
        lambda: execute_fresh_plan(model, plans["F"], stop_at=f_end))

    capture_records = {
        "C": _execution_record(c_result),
        "W": _execution_record(w_result),
        "F": _execution_record(f_result),
    }
    selected_sources, fresh_cpu = recorder.measure(
        f"{case_id}.r2_extract_and_cpu_transfer",
        lambda: (
            {
                "C": extract_rows(
                    c_result.snapshot, c_start, c_end,
                    max_rows=256, to_cpu=True),
                "W": extract_rows(
                    w_result.snapshot, w_start, w_end,
                    max_rows=256, to_cpu=True),
            },
            _snapshot_to_cpu(f_result.snapshot),
        ))
    source_hashes = {
        history: snapshot_sha256(snapshot)
        for history, snapshot in selected_sources.items()
    }
    del c_result, w_result, f_result
    _evict_cuda()

    release_sha, runtime_sha, gpu_uuid = _release_fields(
        handle.runtime_fingerprint)
    selected_map = {
        history: {
            "token_ids": evidence[history]["r2_token_ids"],
            "logical_positions": evidence[history]["r2_logical_positions"],
        }
        for history in ("F", "C", "W")
    }
    foundation_bindings = {
        "fixture_sha256": _sha(
            case["technical_input_sha256"], "technical input SHA"),
        "render_attempt_sha256": _sha256_json({
            "case_id": case_id,
            "carrier": case["carrier"],
            "render_id": RENDER_ID,
            "origin": "fixed_stage_t_technical_carrier",
        }),
        "review_sha256": _sha256_json({
            "case_id": case_id,
            "technical_probe_sha256": case["technical_probe_sha256"],
            "semantic_n": 0,
            "inferential_use": "FORBIDDEN",
        }),
        "phase_a_sha256": _sha256_json({
            "case_id": case_id,
            "phase_a": "NOT_APPLICABLE_STAGE_T",
            "semantic_n": 0,
        }),
    }
    keys, _values = fresh_cpu[0]
    descriptor = build_descriptor(
        release_sha256=release_sha,
        runtime_fingerprint_sha256=runtime_sha,
        primary_batch_id=config.primary_batch_id,
        gpu_uuid=gpu_uuid,
        case_id=case_id,
        stable_candidate_id=case["stable_technical_id"],
        render_id=RENDER_ID,
        schedule=N_SCHEDULE,
        plan_sha256s={
            history: evidence[history]["geometry_sha256"]
            for history in ("F", "C", "W")
        },
        foundation_bindings=foundation_bindings,
        selected_map_sha256=_sha256_json(selected_map),
        selected_token_ids={
            history: evidence[history]["r2_token_ids"]
            for history in ("F", "C", "W")
        },
        logical_position_ids={
            history: evidence[history]["r2_logical_positions"]
            for history in ("F", "C", "W")
        },
        fresh_boundary_token_count=f_end,
        r1_start=plans["F"].physical_regions.content_start,
        r1_end=plans["F"].physical_regions.content_end,
        r2_end=f_end,
        source_intervals={"C": (c_start, c_end), "W": (w_start, w_end)},
        source_rows_sha256=source_hashes,
        kv_heads=int(keys.shape[1]),
        head_dim=int(keys.shape[3]),
    )

    artifact_dir = config.run_directory / "artifacts" / case_id
    descriptor_path = artifact_dir / "foundation_descriptor.json"
    _write_json_exclusive(descriptor_path, descriptor)
    # The independent validator requires canonical kind order, and the store
    # canonicalizes bindings by path.  Keep these three filenames ordered as
    # bundle, descriptor, receipt.
    bundle_path = artifact_dir / "foundation_bundle.safetensors"
    save_result = recorder.measure(
        f"{case_id}.bundle_save_cpu",
        lambda: save_bundle(
            bundle_path,
            fresh_boundary=fresh_cpu,
            selected_sources=selected_sources,
            descriptor=descriptor))
    _require(save_result["sha256"] == bundle_file_sha256(bundle_path),
             f"{case_id} saved bundle read hash differs")

    loaded = recorder.measure(
        f"{case_id}.bundle_load_verify_cpu",
        lambda: load_verified_bundle(
            bundle_path,
            expected_file_sha256=save_result["sha256"],
            expected_descriptor=descriptor))
    cpu_readback = {
        "verified": loaded.verified,
        "materialized": loaded.materialized,
        "file_sha256": loaded.sha256,
        "manifest_sha256": loaded.manifest_sha256,
        "tensor_index_sha256": loaded.tensor_index_sha256,
        "anchor_sha256": loaded.anchor_sha256,
    }
    _require(cpu_readback["verified"] is True
             and cpu_readback["materialized"] is False
             and cpu_readback["file_sha256"] == save_result["sha256"],
             f"{case_id} CPU bundle readback differs")
    del loaded

    vp = recorder.measure(
        f"{case_id}.deterministic_vp",
        lambda: build_placebo_receipt(
            bundle_path,
            expected_file_sha256=save_result["sha256"],
            expected_descriptor=descriptor))
    vp_receipt = dict(vp["receipt"])
    vp_status = str(vp_receipt["status"])
    _require(vp_status in {"AVAILABLE", "PLACEBO_UNAVAILABLE"}
             and vp["receipt_sha256"] == _sha256_json(vp_receipt),
             f"{case_id} deterministic VP receipt differs")
    vp_receipt_path = artifact_dir / "vp_receipt.json"
    vp_diagnostics_path = artifact_dir / "vp_diagnostics.json"
    vp_receipt_file = _write_json_exclusive(vp_receipt_path, vp_receipt)
    vp_diagnostics_file = _write_json_exclusive(
        vp_diagnostics_path, dict(vp["diagnostics"]))
    _require(vp_receipt_file["sha256"] == vp["receipt_sha256"]
             and vp_diagnostics_file["sha256"] == vp["diagnostics_sha256"],
             f"{case_id} VP artifact hashes differ")

    materialized = recorder.measure(
        f"{case_id}.bundle_materialize_cuda",
        lambda: materialize_verified_bundle(
            bundle_path,
            expected_file_sha256=save_result["sha256"],
            expected_descriptor=descriptor,
            device=model_device(model)))
    _require(materialized.verified is True
             and materialized.materialized is True
             and materialized.descriptor == descriptor,
             f"{case_id} CUDA materialized bundle differs")

    foundation_receipt = {
        "schema": FOUNDATION_RECEIPT_SCHEMA,
        "design_id": DESIGN_ID,
        "case_id": case_id,
        "render_id": RENDER_ID,
        "schedule": N_SCHEDULE,
        "semantic_n": 0,
        "inferential_use": "FORBIDDEN",
        "technical_input": dict(input_binding),
        "build_gate": dict(gate_binding),
        "capture_records": capture_records,
        "selected_source_snapshot_sha256": source_hashes,
        "fresh_boundary_snapshot_sha256": snapshot_sha256(fresh_cpu),
        "descriptor_sha256": _sha256_json(descriptor),
        "bundle": dict(save_result),
        "cpu_readback": cpu_readback,
        "vp": {
            "status": vp_status,
            "receipt_path": _safe_relative(vp_receipt_path, config.run_directory),
            "receipt_sha256": vp["receipt_sha256"],
            "diagnostics_path": _safe_relative(
                vp_diagnostics_path, config.run_directory),
            "diagnostics_sha256": vp["diagnostics_sha256"],
        },
        "cuda_materialization": {
            "device": materialized.device,
            "seal_sha256": materialized.seal_sha256,
            "transfer_hashes": [list(row)
                                for row in materialized.transfer_hashes],
        },
        "selected_row_count": f_end - f_start,
        "fresh_boundary_token_count": f_end,
        "timing_vram": recorder.records[timing_index:],
        "timing_vram_summary": _timing_summary(
            recorder.records[timing_index:]),
    }
    receipt_path = artifact_dir / "foundation_receipt.json"
    _write_json_exclusive(receipt_path, foundation_receipt)

    return FoundationState(
        descriptor=descriptor,
        materialized=materialized,
        vp_receipt=vp_receipt,
        vp_receipt_sha256=vp["receipt_sha256"],
        vp_diagnostics=dict(vp["diagnostics"]),
        vp_status=vp_status,
        bundle_binding=_artifact_binding(
            bundle_path, root=config.run_directory, kind="foundation_bundle"),
        descriptor_binding=_artifact_binding(
            descriptor_path, root=config.run_directory,
            kind="foundation_descriptor"),
        receipt_binding=_artifact_binding(
            receipt_path, root=config.run_directory,
            kind="foundation_receipt"),
        selected_row_count=f_end - f_start,
        live_token_count=(
            max(int(evidence[history]["token_count"])
                for history in ("F", "C", "W"))
            + len(case["probe_runtime"]["suffix_ids"])
            + MAX_GENERATED_PROBE_TOKENS),
        r2_end=f_end,
    )


def _run_arm(
    handle: Any,
    case: Mapping[str, Any],
    foundation: FoundationState,
    arm: str,
    recorder: StageRecorder,
) -> dict[str, Any]:
    _require(arm in PRIMARY_ARMS, "runner arm differs from frozen order")
    case_id = str(case["case_id"])
    base = {
        "schema": ARM_ARTIFACT_SCHEMA,
        "design_id": DESIGN_ID,
        "case_id": case_id,
        "render_id": RENDER_ID,
        "schedule": N_SCHEDULE,
        "arm_id": arm,
        "semantic_n": 0,
        "inferential_use": "FORBIDDEN",
        "technical_probe_sha256": case["technical_probe_sha256"],
        "bundle_file_sha256": foundation.bundle_binding["sha256"],
        "descriptor_sha256": foundation.descriptor_binding["sha256"],
        "vp_status": foundation.vp_status,
        "vp_receipt_sha256": foundation.vp_receipt_sha256,
        "stage_a_progression_gate_applies":
            case_id == "technical_e01" and arm == "VP",
    }
    if arm == "VP" and foundation.vp_status == "PLACEBO_UNAVAILABLE":
        return {
            **base,
            "execution_status": "PLACEBO_UNAVAILABLE",
            "stage_a_progression_allowed": (
                False if case_id == "technical_e01" else None),
            "boundary_reconstruction": None,
            "continuation": None,
            "probe": None,
            "vp_receipt": foundation.vp_receipt,
            "vp_diagnostics": foundation.vp_diagnostics,
        }

    placebo = foundation.vp_receipt if arm == "VP" else None
    placebo_sha = foundation.vp_receipt_sha256 if arm == "VP" else None
    model = handle.model
    boundary, replacement = recorder.measure(
        f"{case_id}.arm_{arm}.reconstruct_boundary",
        lambda: reconstruct_arm_boundary(
            foundation.materialized,
            arm=arm,
            placebo_receipt=placebo,
            expected_placebo_receipt_sha256=placebo_sha))
    fresh_plan = case["plan_objects"]["F"]
    continued = recorder.measure(
        f"{case_id}.arm_{arm}.continuation",
        lambda: continue_fresh_plan(
            model, fresh_plan, boundary, start_at=foundation.r2_end))

    probe_runtime = case["probe_runtime"]
    suffix = probe_runtime["suffix_ids"]
    generated_prefix = recorder.measure(
        f"{case_id}.arm_{arm}.probe_prefix_generated",
        lambda: append_block_to_snapshot(
            model, continued.snapshot, suffix,
            logical_start=continued.logical_end,
            label="technical_probe_user_and_assistant_header"))
    correct_score = recorder.measure(
        f"{case_id}.arm_{arm}.score_correct",
        lambda: score_target_from_prefix_q1(
            model, generated_prefix.snapshot, generated_prefix.last_logits,
            target_ids=probe_runtime["correct_target_ids"],
            logical_target_start=generated_prefix.logical_end))
    counter_score = recorder.measure(
        f"{case_id}.arm_{arm}.score_counterfactual",
        lambda: score_target_from_prefix_q1(
            model, generated_prefix.snapshot, generated_prefix.last_logits,
            target_ids=probe_runtime["counterfactual_target_ids"],
            logical_target_start=generated_prefix.logical_end))
    eos_ids = handle.runtime_fingerprint.get("eos_ids")
    _require(isinstance(eos_ids, list) and bool(eos_ids),
             "runtime EOS set is absent")
    generated = recorder.measure(
        f"{case_id}.arm_{arm}.greedy_probe",
        lambda: greedy_generate_q1(
            model, generated_prefix.snapshot, generated_prefix.last_logits,
            logical_start=generated_prefix.logical_end,
            eos_ids=eos_ids,
            max_content_tokens=MAX_GENERATED_PROBE_TOKENS))
    forced_prefix = recorder.measure(
        f"{case_id}.arm_{arm}.probe_prefix_forced",
        lambda: append_block_to_snapshot(
            model, continued.snapshot, suffix,
            logical_start=continued.logical_end,
            label="technical_probe_user_and_assistant_header"))
    forced = recorder.measure(
        f"{case_id}.arm_{arm}.forced_probe",
        lambda: force_content_q1(
            model, forced_prefix.snapshot, forced_prefix.last_logits,
            content_ids=generated.content_ids,
            logical_start=forced_prefix.logical_end,
            eos_ids=eos_ids))
    identity = require_generated_forced_identity(
        generated_prefix, generated, forced_prefix, forced,
        content_start=generated_prefix.physical_end)
    decoded = handle.tokenizer.decode(
        generated.content_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False)
    artifact = {
        **base,
        "execution_status": "COMPLETE",
        "stage_a_progression_allowed": (
            True if case_id == "technical_e01" and arm == "VP" else None),
        "boundary_reconstruction": replacement,
        "continuation": _execution_record(continued),
        "probe": {
            "contract": case["technical_probe"],
            "suffix_ids": list(suffix),
            "suffix_ids_sha256": probe_runtime["suffix_ids_sha256"],
            "correct_target_ids": probe_runtime["correct_target_ids"],
            "correct_target_ids_sha256":
                probe_runtime["correct_target_ids_sha256"],
            "counterfactual_target_ids":
                probe_runtime["counterfactual_target_ids"],
            "counterfactual_target_ids_sha256":
                probe_runtime["counterfactual_target_ids_sha256"],
            "generated_prefix": _execution_record(generated_prefix),
            "forced_prefix": _execution_record(forced_prefix),
            "correct_score": correct_score,
            "counterfactual_score": counter_score,
            "margin": _margin_record(correct_score, counter_score),
            "greedy_generation": {
                **_generation_record(generated),
                "decoded_content": decoded,
                "decoded_content_utf8_sha256": _sha256_bytes(
                    decoded.encode("utf-8")),
            },
            "forced_generation": _generation_record(forced),
            "generated_forced_identity": identity,
            "semantic_n": 0,
            "inferential_use": "FORBIDDEN",
        },
        "vp_receipt": foundation.vp_receipt if arm == "VP" else None,
        "vp_diagnostics": foundation.vp_diagnostics if arm == "VP" else None,
    }
    # Canonical encoding is also the final nonfinite/unsupported-value gate.
    technical_json_bytes(artifact)
    del boundary, continued, generated_prefix, forced_prefix, generated, forced
    _evict_cuda()
    return artifact


def _invoke_terminal_validator(
    *, repo: Path, store_root: Path, identity_path: Path,
    terminal_evidence_path: Path, output_path: Path, runner_pid: int,
) -> dict[str, Any]:
    script = Path(repo) / VALIDATOR_RELATIVE_PATH
    _require(script.is_file() and not script.is_symlink(),
             "independent terminal validator is absent or symlinked")
    command = [
        sys.executable,
        str(script),
        "--store-root", str(store_root),
        "--identity", str(identity_path),
        "--terminal-evidence", str(terminal_evidence_path),
        "--output", str(output_path),
        "--runner-pid", str(runner_pid),
    ]
    result = subprocess.run(
        command, cwd=repo, text=True, capture_output=True, check=False)
    _require(result.returncode == 0,
             "independent terminal validator failed: "
             f"stdout={result.stdout[-2000:]!r} stderr={result.stderr[-2000:]!r}")
    _require(output_path.is_file() and not output_path.is_symlink(),
             "independent terminal validator wrote no receipt")
    try:
        receipt = json.loads(output_path.read_bytes())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise V13StageTRunnerError(
            f"independent terminal receipt is invalid JSON: {exc}") from exc
    _require(isinstance(receipt, Mapping)
             and receipt.get("validation_status") == "PASS",
             "independent terminal receipt did not pass")
    return dict(receipt)


PRODUCTION_SEAMS = StageTSeams(
    open_subject=open_exact_subject,
    prepare_case=_prepare_technical_case,
    run_ladder=run_stage_t_ladder,
    run_long_identity=_run_long_identity,
    capture_foundation=_capture_foundation,
    run_arm=_run_arm,
    process_start_token=linux_process_start_token,
    build_store_identity=build_identity,
    build_store_bounds=build_bounds,
    begin_store_case=begin_case,
    append_store_record=append_case_record,
    active_chain_sha256=active_evidence_chain_sha256,
    invoke_terminal_validator=_invoke_terminal_validator,
    finalize_store_case=finalize_case,
    utc_now=_utc_now,
    precise_utc_now=_precise_utc_now,
    monotonic=time.monotonic,
    cuda_synchronize=_cuda_synchronize,
    reset_peak_vram=_reset_peak_vram,
    vram_metrics=_vram_metrics,
    evict_cuda=_evict_cuda,
)


def _declared_case_bounds(case: Mapping[str, Any]) -> tuple[int, int]:
    plan_tokens = max(
        int(case["plans"][history]["token_count"])
        for history in ("F", "C", "W"))
    selected_rows = int(case["plans"]["F"]["r2_width"])
    live_tokens = plan_tokens + len(case["probe_runtime"]["suffix_ids"]) + \
        MAX_GENERATED_PROBE_TOKENS
    _require(live_tokens <= 7000 and 0 < selected_rows <= 256,
             f"{case['case_id']} declared live-state bounds differ")
    return live_tokens, selected_rows


def _check_deadline(
    *, case_id: str, started: float, seams: StageTSeams,
) -> None:
    elapsed = seams.monotonic() - started
    _require(math.isfinite(elapsed) and elapsed >= 0,
             f"{case_id} case clock differs")
    _require(elapsed <= CASE_DEADLINES_SECONDS[case_id],
             f"{case_id} exceeded its measured per-case deadline")


def _append_foundation_record(
    config: StageTConfig, seams: StageTSeams, *, identity: Mapping[str, Any],
    pid: int, process_token: str, foundation: FoundationState,
    deadline: int,
) -> dict[str, Any]:
    return seams.append_store_record(
        config.run_directory,
        identity=identity,
        pid=pid,
        process_start_token=process_token,
        record_kind="FOUNDATION_LOAD",
        bounds=seams.build_store_bounds(
            completed_arm_ids=(),
            live_token_count=foundation.live_token_count,
            selected_row_count=foundation.selected_row_count,
            case_deadline_seconds=deadline),
        created_utc=seams.utc_now(),
        payload={
            "foundation_bundle_sha256": foundation.bundle_binding["sha256"],
            "foundation_descriptor_sha256":
                foundation.descriptor_binding["sha256"],
            "foundation_receipt_sha256": foundation.receipt_binding["sha256"],
        },
        artifact_bindings=[
            foundation.bundle_binding,
            foundation.descriptor_binding,
            foundation.receipt_binding,
        ],
    )


def _terminalize_case(
    config: StageTConfig,
    seams: StageTSeams,
    *,
    identity: Mapping[str, Any],
    identity_path: Path,
    case_id: str,
    pid: int,
    process_token: str,
    live_tokens: int,
    selected_rows: int,
    deadline: int,
    case_started: float,
) -> dict[str, Any]:
    chain_sha = seams.active_chain_sha256(
        config.run_directory,
        identity=identity,
        pid=pid,
        process_start_token=process_token)
    evidence = {
        "schema": TERMINAL_EVIDENCE_SCHEMA,
        "design_id": DESIGN_ID,
        "identity_sha256": _sha256_json(identity),
        "terminal_kind": "TERMINAL_TECHNICAL",
        "evidence_chain_sha256": chain_sha,
        "outcome": "TECHNICAL_COMPLETE",
        "accepted_attempt_index": None,
        "rejection_codes": [],
        "completed_arm_ids": list(PRIMARY_ARMS),
    }
    artifact_dir = config.run_directory / "artifacts" / case_id
    evidence_path = artifact_dir / "terminal_evidence.json"
    receipt_path = artifact_dir / "independent_terminal_validation.json"
    evidence_file = _write_json_exclusive(evidence_path, evidence)
    receipt = seams.invoke_terminal_validator(
        repo=config.repo,
        store_root=config.run_directory,
        identity_path=identity_path,
        terminal_evidence_path=evidence_path,
        output_path=receipt_path,
        runner_pid=pid)
    receipt_binding = _artifact_binding(
        receipt_path,
        root=config.run_directory,
        kind="independent_terminal_validation_receipt")
    evidence_binding = _artifact_binding(
        evidence_path,
        root=config.run_directory,
        kind="technical_terminal_evidence")
    _require(evidence_file["sha256"] == evidence_binding["sha256"],
             "terminal evidence readback hash differs")
    # Check the scientific-case deadline before making the chain terminal.
    # The subsequent synchronous append/finalize is durability cleanup, not
    # another model or validator operation.
    _check_deadline(
        case_id=case_id, started=case_started, seams=seams)
    terminal_record = seams.append_store_record(
        config.run_directory,
        identity=identity,
        pid=pid,
        process_start_token=process_token,
        record_kind="TERMINAL_TECHNICAL",
        bounds=seams.build_store_bounds(
            completed_arm_ids=PRIMARY_ARMS,
            live_token_count=live_tokens,
            selected_row_count=selected_rows,
            case_deadline_seconds=deadline),
        created_utc=seams.utc_now(),
        payload={
            "status": "TECHNICAL_COMPLETE",
            "evidence_chain_sha256": chain_sha,
            "terminal_evidence_sha256": evidence_binding["sha256"],
            "validation_receipt_sha256": receipt_binding["sha256"],
        },
        artifact_bindings=[evidence_binding, receipt_binding])
    index = seams.finalize_store_case(
        config.run_directory,
        identity=identity,
        pid=pid,
        process_start_token=process_token)
    return {
        "evidence_chain_sha256": chain_sha,
        "terminal_record": terminal_record,
        "validator_receipt": receipt,
        "validator_receipt_sha256": receipt_binding["sha256"],
        "index": index,
    }


def run_stage_t(
    config: StageTConfig,
    *,
    seams: StageTSeams = PRODUCTION_SEAMS,
) -> dict[str, Any]:
    """Run the exact two-case Stage-T worker; errors are parent-quarantined."""

    _require(tuple(TECHNICAL_CASE_IDS) == TECHNICAL_CASE_ORDER,
             "technical module case order differs")
    _require(config.run_directory.is_dir()
             and not config.run_directory.is_symlink(),
             "Stage-T run directory is absent or symlinked")
    _require(isinstance(config.primary_batch_id, str)
             and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*",
                              config.primary_batch_id) is not None
             and len(config.primary_batch_id.encode("utf-8")) <= 128,
             "primary batch ID is not a safe bounded slug")
    _assert_import_boundary()
    run_started = seams.precise_utc_now()
    recorder = StageRecorder(seams)
    pid = os.getpid()
    process_token = seams.process_start_token(pid)
    started_file = _write_json_exclusive(
        config.run_directory / "RUN_STARTED.json",
        {
            "schema": SCHEMA,
            "design_id": DESIGN_ID,
            "status": "STARTED",
            "model_id": MODEL_ID,
            "output_path": str(config.run_directory),
            "case_order": list(TECHNICAL_CASE_ORDER),
            "semantic_n": 0,
            "inferential_use": "FORBIDDEN",
            "started_utc": run_started,
            "runner_pid": pid,
        })
    handle = None
    model = tokenizer = None
    case: dict[str, Any] | None = None
    foundation: FoundationState | None = None
    cases: list[dict[str, Any]] = []
    try:
        # Subject/release/CUDA attestation and the e01 build ladder are strict
        # prerequisites, not measured canary work.  Before the ladder passes,
        # only read post-load VRAM as attestation evidence; do not synchronize
        # or reset CUDA peak statistics for canary timing.
        subject_started_utc = seams.precise_utc_now()
        subject_started = seams.monotonic()
        handle = seams.open_subject(config.repo, config.receipt_directory)
        subject_elapsed = seams.monotonic() - subject_started
        _require(math.isfinite(subject_elapsed) and subject_elapsed >= 0,
                 "exact-subject open timing is invalid")
        subject_completed_utc = seams.precise_utc_now()
        model = handle.model
        tokenizer = handle.tokenizer
        fingerprint = handle.runtime_fingerprint
        release_sha, runtime_sha, gpu_uuid = _release_fields(fingerprint)
        attestation_file = _write_json_exclusive(
            config.run_directory / "RUNTIME_ATTESTATION.json", fingerprint)
        subject_measurement_file = _write_json_exclusive(
            config.run_directory / "SUBJECT_OPEN_MEASUREMENT.json", {
                "schema": (
                    "coherent-state-powered-successor-v13-stage-t-"
                    "subject-open-measurement-v1"),
                "design_id": DESIGN_ID,
                "model_id": MODEL_ID,
                "runtime_fingerprint_sha256": runtime_sha,
                "started_utc": subject_started_utc,
                "completed_utc": subject_completed_utc,
                "wall_time_seconds": subject_elapsed,
                "vram_after_attested_load": seams.vram_metrics(),
                "measurement_class": "PREREQUISITE_ATTESTATION",
            })
        _assert_import_boundary()

        for case_id in TECHNICAL_CASE_ORDER:
            _assert_import_boundary()
            case_started_utc = seams.precise_utc_now()
            case_started = seams.monotonic()
            input_started_utc = seams.precise_utc_now()
            input_started = seams.monotonic()
            case = seams.prepare_case(tokenizer, config.repo, case_id)
            _require(case["case_id"] == case_id and case["semantic_n"] == 0,
                     "prepared technical case differs")
            live_tokens, selected_rows = _declared_case_bounds(case)
            deadline = CASE_DEADLINES_SECONDS[case_id]
            artifact_dir = config.run_directory / "artifacts" / case_id
            input_file = _write_json_exclusive(
                artifact_dir / "technical_input.json",
                _public_case_evidence(case))
            input_elapsed = seams.monotonic() - input_started
            _require(math.isfinite(input_elapsed) and input_elapsed >= 0,
                     f"{case_id} input build timing is invalid")
            input_timing = {
                "started_utc": input_started_utc,
                "completed_utc": seams.precise_utc_now(),
                "wall_time_seconds": input_elapsed,
                "artifact_sha256": input_file["sha256"],
                "readback_verified": True,
            }
            identity = seams.build_store_identity(
                release_sha256=release_sha,
                runtime_fingerprint_sha256=runtime_sha,
                primary_batch_id=config.primary_batch_id,
                gpu_uuid=gpu_uuid,
                case_id=case_id,
                render_id=RENDER_ID,
                schedule=N_SCHEDULE,
                mode="technical")
            identity_path = artifact_dir / "identity.json"
            _write_json_exclusive(identity_path, identity)
            seams.begin_store_case(
                config.run_directory,
                identity=identity,
                bounds=seams.build_store_bounds(
                    completed_arm_ids=(),
                    live_token_count=live_tokens,
                    selected_row_count=selected_rows,
                    case_deadline_seconds=deadline),
                pid=pid,
                process_start_token=process_token,
                created_utc=seams.utc_now())

            if case_id == "technical_e01":
                gate_started_utc = seams.precise_utc_now()
                gate_started = seams.monotonic()
                gate_result = seams.run_ladder(
                    model, case["plan_objects"]["F"])
                gate_elapsed = seams.monotonic() - gate_started
                _require(math.isfinite(gate_elapsed) and gate_elapsed >= 0,
                         "technical_e01 ladder timing is invalid")
                gate_kind = "POWERED_V13_LADDER"
                gate_timing: list[dict[str, Any]] = []
                gate_wall_timing = {
                    "started_utc": gate_started_utc,
                    "completed_utc": seams.precise_utc_now(),
                    "wall_time_seconds": gate_elapsed,
                    "measurement_class": "PREREQUISITE_BUILD_LADDER",
                    "cuda_peak_stats_reset": False,
                }
            else:
                gate_index = len(recorder.records)
                gate_result = recorder.measure(
                    "technical_long.l0_l3_identity",
                    lambda: seams.run_long_identity(
                        model, case["plan_objects"]["C"]))
                gate_kind = "TECHNICAL_LONG_L0_L3"
                gate_timing = recorder.records[gate_index:]
                gate_wall_timing = {
                    key: gate_timing[-1][key]
                    for key in (
                        "started_utc", "completed_utc", "wall_time_seconds")
                }
            _require(gate_result.get("status") == "PASS",
                     f"{case_id} build gate did not pass")
            gate_artifact = {
                "schema": GATE_ARTIFACT_SCHEMA,
                "design_id": DESIGN_ID,
                "case_id": case_id,
                "gate_kind": gate_kind,
                "status": "PASS",
                "semantic_n": 0,
                "inferential_use": "FORBIDDEN",
                "technical_input_sha256": input_file["sha256"],
                "input_build_verify_timing": input_timing,
                "evidence": gate_result,
                "gate_wall_timing": gate_wall_timing,
                "timing_vram": gate_timing,
            }
            gate_path = artifact_dir / "build_gate.json"
            _write_json_exclusive(gate_path, gate_artifact)
            gate_binding = _artifact_binding(
                gate_path, root=config.run_directory, kind="build_gate")
            input_binding = _artifact_binding(
                artifact_dir / "technical_input.json",
                root=config.run_directory, kind="technical_input")
            _check_deadline(
                case_id=case_id, started=case_started, seams=seams)

            foundation = recorder.measure(
                f"{case_id}.foundation_total",
                lambda: seams.capture_foundation(
                    config, handle, case, input_binding, gate_binding, recorder))
            _require(foundation.live_token_count == live_tokens
                     and foundation.selected_row_count == selected_rows,
                     f"{case_id} observed/declaration state bounds differ")
            _append_foundation_record(
                config, seams,
                identity=identity,
                pid=pid,
                process_token=process_token,
                foundation=foundation,
                deadline=deadline)
            _check_deadline(
                case_id=case_id, started=case_started, seams=seams)

            arm_rows: list[dict[str, Any]] = []
            completed: list[str] = []
            for arm in PRIMARY_ARMS:
                timing_index = len(recorder.records)
                artifact = recorder.measure(
                    f"{case_id}.arm_{arm}.total",
                    lambda arm=arm: seams.run_arm(
                        handle, case, foundation, arm, recorder))
                _require(artifact.get("arm_id") == arm
                         and artifact.get("semantic_n") == 0
                         and artifact.get("inferential_use") == "FORBIDDEN",
                         f"{case_id} {arm} artifact boundary differs")
                arm_timings = recorder.records[timing_index:]
                _require(bool(arm_timings)
                         and arm_timings[-1]["stage"] ==
                         f"{case_id}.arm_{arm}.total",
                         f"{case_id} {arm} total timing is absent")
                artifact["timing_vram"] = arm_timings
                artifact["arm_wall_time_seconds"] = arm_timings[-1][
                    "wall_time_seconds"]
                artifact["arm_peak_vram"] = _timing_summary(arm_timings)
                artifact_path = artifact_dir / f"arm_{arm}_artifact.json"
                artifact_file = _write_json_exclusive(artifact_path, artifact)
                completed.append(arm)
                checkpoint = {
                    "schema": ARM_CHECKPOINT_SCHEMA,
                    "design_id": DESIGN_ID,
                    "case_id": case_id,
                    "arm_id": arm,
                    "status": "CHECKPOINTED",
                    "completed_arm_ids": list(completed),
                    "artifact_sha256": artifact_file["sha256"],
                    "semantic_n": 0,
                    "inferential_use": "FORBIDDEN",
                    "created_utc": seams.precise_utc_now(),
                }
                checkpoint_path = artifact_dir / f"arm_{arm}_checkpoint.json"
                checkpoint_file = _write_json_exclusive(
                    checkpoint_path, checkpoint)
                artifact_binding = _artifact_binding(
                    artifact_path,
                    root=config.run_directory,
                    kind=f"arm_{arm}_artifact")
                checkpoint_binding = _artifact_binding(
                    checkpoint_path,
                    root=config.run_directory,
                    kind=f"arm_{arm}_checkpoint")
                _require(artifact_binding["sha256"] == artifact_file["sha256"]
                         and checkpoint_binding["sha256"] ==
                         checkpoint_file["sha256"],
                         f"{case_id} {arm} checkpoint readback differs")
                seams.append_store_record(
                    config.run_directory,
                    identity=identity,
                    pid=pid,
                    process_start_token=process_token,
                    record_kind=f"ARM_{arm}",
                    bounds=seams.build_store_bounds(
                        completed_arm_ids=completed,
                        live_token_count=live_tokens,
                        selected_row_count=selected_rows,
                        case_deadline_seconds=deadline),
                    created_utc=seams.utc_now(),
                    payload={
                        "arm_id": arm,
                        "checkpoint_sha256": checkpoint_binding["sha256"],
                        "artifact_sha256": artifact_binding["sha256"],
                    },
                    artifact_bindings=[checkpoint_binding, artifact_binding])
                arm_rows.append({
                    "arm_id": arm,
                    "execution_status": artifact["execution_status"],
                    "artifact_sha256": artifact_binding["sha256"],
                    "checkpoint_sha256": checkpoint_binding["sha256"],
                    "probe_executed": artifact.get("probe") is not None,
                    "inferential_use": "FORBIDDEN",
                })
                _check_deadline(
                    case_id=case_id, started=case_started, seams=seams)

            terminal = _terminalize_case(
                config, seams,
                identity=identity,
                identity_path=identity_path,
                case_id=case_id,
                pid=pid,
                process_token=process_token,
                live_tokens=live_tokens,
                selected_rows=selected_rows,
                deadline=deadline,
                case_started=case_started)
            case_completed = seams.monotonic()
            case_elapsed = case_completed - case_started
            _require(math.isfinite(case_elapsed) and case_elapsed >= 0,
                     f"{case_id} total timing is invalid")
            cases.append({
                "case_id": case_id,
                "semantic_n": 0,
                "inferential_use": "FORBIDDEN",
                "identity": identity,
                "technical_input_sha256": input_file["sha256"],
                "gate_sha256": gate_binding["sha256"],
                "vp_status": foundation.vp_status,
                "stage_a_progression_gate_applies":
                    case_id == "technical_e01",
                "stage_a_progression_allowed":
                    (foundation.vp_status == "AVAILABLE"
                     if case_id == "technical_e01" else None),
                "arms": arm_rows,
                "terminal": terminal,
                "case_started_utc": case_started_utc,
                "case_completed_utc": seams.precise_utc_now(),
                "case_wall_time_seconds": case_elapsed,
            })
            foundation.materialized = None
            case = None
            foundation = None
            seams.evict_cuda()
            _assert_import_boundary()

        e01 = next(
            row for row in cases if row["case_id"] == "technical_e01")
        progression = e01["vp_status"] == "AVAILABLE"
        summary = {
            "schema": RUN_SUMMARY_SCHEMA,
            "design_id": DESIGN_ID,
            "status": "PASS",
            "model_id": MODEL_ID,
            "model_slug": MODEL_SLUG,
            "run_directory": str(config.run_directory),
            "primary_batch_id": config.primary_batch_id,
            "semantic_n": 0,
            "inferential_use": "FORBIDDEN",
            "case_order": [row["case_id"] for row in cases],
            "cases": cases,
            "stage_a_progression_allowed": progression,
            "stage_a_progression_reason": (
                "TECHNICAL_E01_VP_AVAILABLE" if progression
                else "TECHNICAL_E01_VP_UNAVAILABLE"),
            "technical_long_vp_is_diagnostic_only": True,
            "runtime_attestation_sha256": attestation_file["sha256"],
            "subject_open_measurement_sha256":
                subject_measurement_file["sha256"],
            "run_started_sha256": started_file["sha256"],
            "timing_vram": recorder.records,
            "completed_utc": seams.precise_utc_now(),
            "production_modules_absent": True,
        }
        _write_json_exclusive(
            config.run_directory / "RUN_COMPLETE.json", summary)
        return summary
    except BaseException as exc:
        partial = {
            "schema": (
                "coherent-state-powered-successor-v13-stage-t-"
                "partial-evidence-v1"),
            "design_id": DESIGN_ID,
            "status": "PARTIAL_ERROR",
            "semantic_n": 0,
            "inferential_use": "FORBIDDEN",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "completed_case_ids": [row["case_id"] for row in cases],
            "timing_vram": recorder.records,
            "observed_utc": seams.precise_utc_now(),
        }
        try:
            _write_json_exclusive(
                config.run_directory / "PARTIAL_RUN_EVIDENCE.json", partial)
        except BaseException:
            pass
        raise
    finally:
        if foundation is not None:
            foundation.materialized = None
        foundation = None
        case = None
        model = None
        tokenizer = None
        try:
            if handle is not None:
                handle.close()
        finally:
            handle = None
            seams.evict_cuda()


def _worker_entry(
    config: StageTConfig,
    *,
    seams: StageTSeams = PRODUCTION_SEAMS,
) -> int:
    try:
        run_stage_t(config, seams=seams)
        return 0
    except BaseException as exc:
        error = {
            "schema": RUN_ERROR_SCHEMA,
            "design_id": DESIGN_ID,
            "status": "ERROR",
            "model_id": MODEL_ID,
            "run_directory": str(config.run_directory),
            "semantic_n": 0,
            "inferential_use": "FORBIDDEN",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "observed_utc": seams.precise_utc_now(),
        }
        try:
            _write_json_exclusive(config.run_directory / "RUN_ERROR.json", error)
        except BaseException:
            pass
        print(f"STAGE-T WORKER ERROR: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 2


def _worker_command(
    *, script: Path, repo: Path, receipt_directory: Path,
    output_parent: Path, run_directory: Path, primary_batch_id: str,
) -> list[str]:
    return [
        sys.executable,
        str(script),
        "--repo", str(repo),
        "--receipt-directory", str(receipt_directory),
        "--output-parent", str(output_parent),
        "--primary-batch-id", primary_batch_id,
        "--_worker-run-directory", str(run_directory),
    ]


def _supervise_created_run(
    *,
    repo: Path,
    receipt_directory: Path,
    output_parent: Path,
    run_directory: Path,
    script: Path,
    primary_batch_id: str,
    run_process: Callable[..., Any] = subprocess.run,
    quarantine: Callable[..., dict[str, Any]] = quarantine_abandoned_case,
    utc_now: Callable[[], str] = _utc_now,
) -> int:
    """Run one worker; quarantine only after a failed worker is dead."""

    command = _worker_command(
        script=script,
        repo=repo,
        receipt_directory=receipt_directory,
        output_parent=output_parent,
        run_directory=run_directory,
        primary_batch_id=primary_batch_id)
    result = run_process(command, cwd=repo, check=False)
    returncode = int(result.returncode)
    quarantine_record = None
    if returncode != 0 and (run_directory / ACTIVE_LOCK_NAME).exists():
        quarantine_record = quarantine(
            run_directory,
            observed_utc=utc_now(),
            reason=f"stage-t-worker-exit-{returncode}")
    supervisor = {
        "schema": SUPERVISOR_SCHEMA,
        "design_id": DESIGN_ID,
        "status": "WORKER_EXITED_ZERO" if returncode == 0 else "WORKER_FAILED",
        "worker_returncode": returncode,
        "worker_command": command,
        "quarantine_record": quarantine_record,
        "provider_clock_owned_by_runner": False,
        "watchdog_duplicated": False,
        "observed_utc": utc_now(),
    }
    _write_json_exclusive(run_directory / "SUPERVISOR.json", supervisor)
    return returncode


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    result.add_argument("--receipt-directory", required=True, type=Path)
    result.add_argument("--primary-batch-id", required=True,
                        help="externally assigned admitted-host batch slug")
    result.add_argument("--output-parent", type=Path,
                        default=DEFAULT_OUTPUT_PARENT)
    result.add_argument("--_worker-run-directory", type=Path,
                        help=argparse.SUPPRESS)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo = args.repo.resolve()
    receipt_directory = args.receipt_directory.resolve()
    output_parent = (
        args.output_parent if args.output_parent.is_absolute()
        else repo / args.output_parent).resolve()
    _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*",
                          args.primary_batch_id) is not None
             and len(args.primary_batch_id.encode("utf-8")) <= 128,
             "primary batch ID is not a safe bounded slug")
    if args._worker_run_directory is not None:
        run_directory = args._worker_run_directory.resolve()
        _require(run_directory.is_dir(), "worker run directory is absent")
        print(
            "WORKER powered-v13-stage-t "
            f"model={MODEL_ID}@{MODEL_REVISION} output={run_directory}",
            flush=True)
        return _worker_entry(StageTConfig(
            repo=repo,
            receipt_directory=receipt_directory,
            run_directory=run_directory,
            primary_batch_id=args.primary_batch_id))

    run_directory = _create_unique_output_directory(output_parent)
    print(
        "RUN powered-v13-stage-t "
        f"model={MODEL_ID}@{MODEL_REVISION} output={run_directory}",
        flush=True)
    script = repo / "scripts/run_powered_v13_stage_t.py"
    return _supervise_created_run(
        repo=repo,
        receipt_directory=receipt_directory,
        output_parent=output_parent,
        run_directory=run_directory,
        script=script,
        primary_batch_id=args.primary_batch_id)


__all__ = [
    "CASE_DEADLINES_SECONDS",
    "FORBIDDEN_PRODUCTION_MODULES",
    "FoundationState",
    "MODEL_SLUG",
    "PRODUCTION_SEAMS",
    "SCHEMA",
    "StageTConfig",
    "StageTSeams",
    "TECHNICAL_CASE_ORDER",
    "V13StageTRunnerError",
    "main",
    "run_stage_t",
]
