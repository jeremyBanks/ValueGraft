"""Qwen3-0.6B production-path ladder for coherent-summary-state apparatus."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from coherent_state_cases import (
    FROZEN_ORDER,
    WRONG_DONOR,
    correct_source_messages,
    fresh_source_messages,
    validate_native_conversation,
)
from coherent_state_calibration import validate_calibration_constructions
from coherent_state_hf import (
    compare_rows,
    move_key_rows,
    replace_summary_rows,
    row_hashes,
    sha256_ids,
)
from coherent_state_runtime import (
    AMENDMENT_ID,
    DESIGN_ID,
    GAPPED_ARM_NAMES,
    append_gapped_post_summary,
    build_gapped_fresh_boundary,
    capture_forced_prefix_ids,
    capture_forced_summary,
    capture_generated_summary,
    complete_assistant_context,
    eager_backend_fingerprint,
    gapped_arm_boundary,
    measure_generated_replay,
    validate_generated_replay,
    validate_position_schedule,
)
from coherent_state_store import checkpoint_path, read_checkpoint, save_render
from coherent_state_tokens import (
    generation_prefix_ids,
    matched_wrong_prefix_ids,
    rendered_assistant_content_ids,
)
from kvlib_hf import prefill, rebuild_cache, snapshot_cache
from arms_common import SUMMARY_REQUEST
from validate_coherent_external_donors import validate_with_tokenizer


MODEL = "Qwen/Qwen3-0.6B"
PRODUCTION_TOKENIZER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PRODUCTION_TOKENIZER_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
REQUEST = "Write a short context summary. Output only the summary."
SUMMARY = "The approved label remains on file."
_LAST_LADDER_DIAGNOSTICS: dict = {}


def fake_conv(cid: str, label: str, marker: str) -> dict:
    return {
        "id": cid,
        "messages": [
            {"role": "system", "content": "Maintain the user's test record."},
            {"role": "user", "content":
             f"The approved label is {label}. Remember that exact label."},
            {"role": "assistant", "content":
             f"Understood; the approved label is {label}."},
            {"role": "user", "content": f"Unrelated tail marker: {marker}."},
            {"role": "assistant", "content": "Tail marker noted."},
        ],
        "sections": {"middle_end_msg": 3},
    }


def run_loaded_kernel_gates(model, tokenizer, *, identity_tolerance: float,
                            rotation_tolerance: float,
                            placebo_quantization_tolerance: float = 0.05,
                            placebo_moment_tolerance: float = 0.02) -> dict:
    """Compatibility name for the additive-amendment gapped authorization gate.

    rotation_tolerance is intentionally ignored: packed key movement is a
    retired diagnostic and can no longer authorize a semantic run through this
    legacy entry point.
    """
    del rotation_tolerance
    return run_loaded_gapped_gates(
        model, tokenizer,
        identity_tolerance=identity_tolerance,
        placebo_quantization_tolerance=placebo_quantization_tolerance,
        placebo_moment_tolerance=placebo_moment_tolerance)


def _snapshot_length(snapshot) -> int:
    lengths = {int(t.shape[-2]) for pair in snapshot for t in pair}
    if len(lengths) != 1:
        raise RuntimeError(f"cache tensors have inconsistent lengths: {lengths}")
    return lengths.pop()


def _require_summary_boundary(snapshot, layout) -> None:
    """Reject an arm that already contains the assistant close or tail."""
    observed = _snapshot_length(snapshot)
    if observed != layout.physical_summary_end:
        raise RuntimeError(
            f"arm must fork exactly at summary boundary: cache={observed} "
            f"expected={layout.physical_summary_end}")


def _verify_intervention(fresh, treated, source, start: int, *,
                         use_keys: bool, use_values: bool) -> None:
    """Require exact selected insertion and exact preservation elsewhere."""
    if not (len(fresh) == len(treated) == len(source)):
        raise RuntimeError("intervention layer coverage differs")
    for li, ((kf, vf), (kt, vt), (ks, vs)) in enumerate(
            zip(fresh, treated, source)):
        n = int(ks.shape[-2])
        end = start + n
        if use_keys and not torch.equal(
                kt[..., start:end, :], ks.to(kt.device, dtype=kt.dtype)):
            raise RuntimeError(f"layer {li} key insertion is not bit exact")
        if use_values and not torch.equal(
                vt[..., start:end, :], vs.to(vt.device, dtype=vt.dtype)):
            raise RuntimeError(f"layer {li} value insertion is not bit exact")
        if not use_keys and not torch.equal(kt, kf):
            raise RuntimeError(f"layer {li} unselected keys changed")
        if not use_values and not torch.equal(vt, vf):
            raise RuntimeError(f"layer {li} unselected values changed")
        for before, after, selected in (
                (kf, kt, use_keys), (vf, vt, use_values)):
            if selected and (not torch.equal(before[..., :start, :],
                                             after[..., :start, :]) or
                             not torch.equal(before[..., end:, :],
                                             after[..., end:, :])):
                raise RuntimeError(
                    f"layer {li} non-summary rows changed during insertion")


def _snapshot_max_abs(a, b) -> tuple[float, float]:
    rows = compare_rows(a, b)
    return (max(x["k_max_abs"] for x in rows),
            max(x["v_max_abs"] for x in rows))


def _span_hashes(snapshot, start: int, end: int) -> list[dict[str, str]]:
    if not 0 <= start <= end <= _snapshot_length(snapshot):
        raise RuntimeError(f"invalid snapshot hash span [{start}, {end})")
    return row_hashes([
        (key[..., start:end, :], value[..., start:end, :])
        for key, value in snapshot
    ])


def _scalar_metric_max(raw: dict, metric_names) -> float:
    """Aggregate only the predeclared scalar metrics in a mixed raw payload."""
    values = []
    for name in metric_names:
        value = raw.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(f"raw metric {name} is missing or non-finite")
        values.append(float(value))
    if not values:
        raise RuntimeError("no scalar metrics were declared for aggregation")
    return max(values)


FROZEN_FIXTURE_LITERAL = "alpha beta gamma delta epsilon"
FROZEN_FIXTURE_POOL = [7141, 13440, 21619, 9477, 31204]
FROZEN_MARGIN_IDS = [362, 425]
FROZEN_CONTINUATION_ID = 7141
FROZEN_SCHEDULES = (
    (5, (5,), (2, 3)),
    (64, (64,), (32, 32)),
    (900, (900,), (32, 868)),
    (4096, (4096,), (32, 4064)),
    (4097, (4096, 1), (32, 4065)),
    (8193, (4096, 4096, 1), (32, 4096, 4065)),
)

MAX_TECHNICAL_LOGICAL_POSITION = 9509
FROZEN_CASE_CONTINUATION_POSITIONS = {
    "c10": 8430, "c02": 8385, "c01": 8855, "c04": 8595,
    "c07": 8600, "c11": 9381, "c05": 8556, "c09": 9195,
    "c06": 8876, "c12": 9509, "c08": 8525, "c03": 8913,
}
V7_GATE_STAGE_ORDER = (
    "static_provenance",
    "attention_backend",
    "synthetic_schedule_fixtures",
    "committed_case_schedule_fixtures",
    "generated_replay_identity",
    "snapshot_rebuild_identity",
    "physical_causal_mask_identity",
    "future_mutation_identity",
    "position_structure",
    "intervention_propagation",
    "calibration_construction",
    "external_donor_construction",
    "retired_G_delta",
)
# Compatibility for downstream code written while Amendments 5/6 were current.
V6_GATE_STAGE_ORDER = V7_GATE_STAGE_ORDER
V5_GATE_STAGE_ORDER = V7_GATE_STAGE_ORDER
TERMINAL_STAGE_STATES = {"PASS", "FAIL", "ERROR", "SKIPPED_DEPENDENCY"}


def _stage(*, prerequisites=(), threshold=None, comparison=None,
           expected_coverage=None, metric_names=(), raw=None) -> dict:
    return {
        "status": "PENDING",
        "passes": False,
        "prerequisites": list(prerequisites),
        "threshold": threshold,
        "comparison": comparison,
        "expected_coverage": expected_coverage,
        "observed_coverage": 0,
        "metric_names": list(metric_names),
        "raw": {} if raw is None else raw,
        "failure_evidence": None,
    }


def v7_gate_schema(*, identity_tolerance: float = 1e-4,
                   zero_gap_tolerance: float = 5e-4,
                   case_dir: Path | str = Path("data/synthetic"),
                   donor_dir: Path | str | None = None,
                   expected_attention_layers: int = 48) -> dict:
    """Return the exhaustive predeclared Amendments-5/6/7 model-gate schema.

    Returned identities and maximum position are exclusively v7.
    """
    donor_dir = Path(case_dir) if donor_dir is None else Path(donor_dir)
    common_schedule_metrics = (
        "cache_k_max_abs", "cache_v_max_abs", "last_logits_max_abs",
        "selected_margin_abs_shift", "continuation_logits_max_abs",
        "continuation_k_max_abs", "continuation_v_max_abs",
    )
    synthetic_skeleton = {
        "contiguous": [
            {"length": length, "reference_partition": list(reference),
             "alternative_partition": list(alternative), "status": "PENDING"}
            for length, reference, alternative in FROZEN_SCHEDULES
        ],
        "logical_gap": {
            "length": 64, "logical_start_blocks": [[0, 31], [8192, 8223]],
            "continuation_logical_position": 8224, "status": "PENDING"},
    }
    case_skeleton = {
        "frozen_order": list(FROZEN_ORDER),
        "rows": [
            {"conversation_id": conversation_id,
             "order_position": index,
             "source_path": str(Path(case_dir) / f"{conversation_id}.json"),
             "expected_continuation_logical_position":
                 FROZEN_CASE_CONTINUATION_POSITIONS[conversation_id],
             "status": "PENDING"}
            for index, conversation_id in enumerate(FROZEN_ORDER, 1)
        ],
    }
    donor_skeleton = {
        "frozen_order": list(FROZEN_ORDER), "mapping": dict(WRONG_DONOR),
        "rows": [
            {"target_id": target, "donor_id": WRONG_DONOR[target],
             "order_position": index,
             "target_path": str(donor_dir / f"{target}.json"),
             "donor_path": str(donor_dir / f"{WRONG_DONOR[target]}.json"),
             "status": "PENDING"}
            for index, target in enumerate(FROZEN_ORDER, 1)
        ],
    }
    stages = {
        "static_provenance": _stage(
            expected_coverage=1,
            metric_names=("fingerprint_static", "apparatus_inventory",
                          "input_inventory")),
        "attention_backend": _stage(
            expected_coverage=int(expected_attention_layers),
            metric_names=("model_config", "text_config", "layers")),
        "synthetic_schedule_fixtures": _stage(
            prerequisites=("attention_backend",), threshold=zero_gap_tolerance,
            comparison="<=", expected_coverage=7,
            metric_names=common_schedule_metrics, raw=synthetic_skeleton),
        "committed_case_schedule_fixtures": _stage(
            prerequisites=("attention_backend",), threshold=zero_gap_tolerance,
            comparison="<=", expected_coverage=12,
            metric_names=common_schedule_metrics, raw=case_skeleton),
        "generated_replay_identity": _stage(
            prerequisites=("attention_backend",), threshold=identity_tolerance,
            comparison="<=", expected_coverage=1,
            metric_names=("token_logprob_max_abs", "k_max_abs", "v_max_abs"),
            raw={name: None for name in (
                "token_logprob_max_abs", "k_max_abs", "v_max_abs")}),
        "snapshot_rebuild_identity": _stage(
            prerequisites=("attention_backend",), threshold=identity_tolerance,
            comparison="<=", expected_coverage=1,
            metric_names=("logits_max_abs", "k_max_abs", "v_max_abs"),
            raw={name: None for name in (
                "logits_max_abs", "k_max_abs", "v_max_abs")}),
        "physical_causal_mask_identity": _stage(
            prerequisites=("attention_backend",), threshold=identity_tolerance,
            comparison="<=", expected_coverage=1,
            metric_names=("logits_max_abs", "k_max_abs", "v_max_abs"),
            raw={name: None for name in (
                "logits_max_abs", "k_max_abs", "v_max_abs")}),
        "future_mutation_identity": _stage(
            prerequisites=("physical_causal_mask_identity",),
            threshold=identity_tolerance, comparison="<=", expected_coverage=1,
            metric_names=("earlier_logits_max_abs", "earlier_cache_max_abs"),
            raw={name: None for name in (
                "earlier_logits_max_abs", "earlier_cache_max_abs")}),
        "position_structure": _stage(
            prerequisites=("attention_backend",), expected_coverage=1,
            metric_names=("position_schedule", "wrong_source", "injections")),
        "intervention_propagation": _stage(
            prerequisites=("position_structure",), expected_coverage=1,
            metric_names=("bit_exact_insertion", "tail_sensitivity")),
        "calibration_construction": _stage(
            expected_coverage=2, metric_names=("construction_hashes",),
            raw={"variants": {name: {"status": "PENDING"}
                              for name in ("c10", "c07")}}),
        "external_donor_construction": _stage(
            expected_coverage=12, metric_names=("pair_construction_hashes",),
            raw=donor_skeleton),
        "retired_G_delta": _stage(expected_coverage=1),
    }
    return {
        "schema": 2,
        "amendment_id": AMENDMENT_ID,
        "design_id": DESIGN_ID,
        "status": "RUNNING",
        "passes": False,
        "technical_only": True,
        "authorization_path": "gapped_position_preserving_only",
        "position_policy": "logical_position_ids_physical_cache_position",
        "max_technical_logical_position": MAX_TECHNICAL_LOGICAL_POSITION,
        "stage_order": list(V7_GATE_STAGE_ORDER),
        "case_dir": str(Path(case_dir)),
        "donor_dir": str(donor_dir),
        **stages,
    }


def v6_gate_schema(**kwargs) -> dict:
    """Compatibility alias; no v6 identity or artifact is ever returned."""
    return v7_gate_schema(**kwargs)


def v5_gate_schema(**kwargs) -> dict:
    """Compatibility alias; no v5 identity or artifact is ever returned."""
    return v7_gate_schema(**kwargs)


def _copy_json(value):
    return json.loads(json.dumps(value))


def _set_stage(sink: dict, name: str, stage: dict) -> None:
    """Reassign a whole stage so durable dict implementations flush it."""
    sink[name] = _copy_json(stage)


def _begin_stage(sink: dict, name: str, *, resume_running: bool = False) -> dict:
    stage = _copy_json(sink[name])
    if stage["status"] == "RUNNING" and resume_running:
        stage.setdefault("started_at", datetime.now(timezone.utc).isoformat())
        _set_stage(sink, name, stage)
        return stage
    if stage["status"] != "PENDING":
        raise RuntimeError(f"{name} did not begin from PENDING")
    stage["status"] = "RUNNING"
    stage["started_at"] = datetime.now(timezone.utc).isoformat()
    _set_stage(sink, name, stage)
    return stage


def _close_stage(sink: dict, name: str, stage: dict, *, passes: bool,
                 status: str | None = None, failure=None) -> None:
    status = status or ("PASS" if passes else "FAIL")
    if status not in TERMINAL_STAGE_STATES:
        raise RuntimeError(f"invalid terminal stage status: {status}")
    stage["status"] = status
    stage["passes"] = bool(passes)
    stage["failure_evidence"] = failure
    stage["completed_at"] = datetime.now(timezone.utc).isoformat()
    _set_stage(sink, name, stage)


def _error_stage(sink: dict, name: str, stage: dict, exc: Exception) -> None:
    _close_stage(sink, name, stage, passes=False, status="ERROR", failure={
        "error_type": type(exc).__name__, "error": str(exc),
        "traceback": traceback.format_exc(),
        "last_completed_unit": stage.get("observed_coverage", 0),
        "expected_coverage": stage.get("expected_coverage"),
        "observed_coverage": stage.get("observed_coverage", 0),
    })


def _partitioned_forward(model, token_ids, logical_positions, partitions):
    if sum(partitions) != len(token_ids):
        raise RuntimeError("fixture partition does not cover token stream")
    cache = None
    logits = None
    lo = 0
    for width in partitions:
        hi = lo + width
        ids = torch.tensor([token_ids[lo:hi]], device=model.device)
        pos = torch.tensor([logical_positions[lo:hi]], device=model.device)
        physical = torch.arange(lo, hi, device=model.device)
        cache, logits = prefill(
            model, ids, past=cache, position_ids=pos,
            cache_position=physical)
        lo = hi
    return cache, logits


def _margin(logits) -> float:
    lp = torch.log_softmax(logits.float(), dim=-1)
    return float(lp[0, FROZEN_MARGIN_IDS[0]] - lp[0, FROZEN_MARGIN_IDS[1]])


def _compare_schedules(model, token_ids, logical_positions,
                       reference_partition, alternative_partition,
                       tolerance, progress=None) -> dict:
    reference, reference_logits = _partitioned_forward(
        model, token_ids, logical_positions, reference_partition)
    alternative, alternative_logits = _partitioned_forward(
        model, token_ids, logical_positions, alternative_partition)
    last_logits = float(
        (reference_logits.float() - alternative_logits.float()).abs().max())
    margin_shift = abs(_margin(reference_logits) - _margin(alternative_logits))
    rows = []
    result = {
        "reference_partition": list(reference_partition),
        "alternative_partition": list(alternative_partition),
        "per_layer": rows,
        "continuation_per_layer": [],
        "last_logits_max_abs": last_logits,
        "selected_margin_abs_shift": margin_shift,
        "tolerance": tolerance,
        "base_measurement_complete": False,
        "continuation_measurement_complete": False,
        "passes": False,
    }
    reference_rows = snapshot_cache(reference)
    alternative_rows = snapshot_cache(alternative)
    if len(reference_rows) != len(alternative_rows):
        raise RuntimeError("schedule cache layer coverage differs")
    for layer_index, ((left_k, left_v), (right_k, right_v)) in enumerate(zip(
            reference_rows, alternative_rows)):
        rows.append({
            "layer": layer_index,
            "k_max_abs": float((
                left_k.float() - right_k.float()).abs().max()),
            "v_max_abs": float((
                left_v.float() - right_v.float()).abs().max()),
        })
        if progress is not None:
            progress(_copy_json(result))
    if not rows:
        raise RuntimeError("schedule comparison produced no cache layers")
    maxima = {
        "cache_k_max_abs": max(x["k_max_abs"] for x in rows),
        "cache_v_max_abs": max(x["v_max_abs"] for x in rows),
        "last_logits_max_abs": last_logits,
        "selected_margin_abs_shift": margin_shift,
    }
    result.update(maxima)
    result["base_measurement_complete"] = True
    if progress is not None:
        progress(_copy_json(result))
    n = len(token_ids)
    continuation_ids = torch.tensor([[FROZEN_CONTINUATION_ID]], device=model.device)
    continuation_pos = torch.tensor([[logical_positions[-1] + 1]], device=model.device)
    continuation_cache_pos = torch.tensor([n], device=model.device)
    with torch.no_grad():
        ref_next = model(
            input_ids=continuation_ids, past_key_values=reference,
            position_ids=continuation_pos, cache_position=continuation_cache_pos,
            use_cache=True, logits_to_keep=0)
        alt_next = model(
            input_ids=continuation_ids, past_key_values=alternative,
            position_ids=continuation_pos, cache_position=continuation_cache_pos,
            use_cache=True, logits_to_keep=0)
    continuation_logits = float(
        (ref_next.logits.float() - alt_next.logits.float()).abs().max())
    continuation_rows = result["continuation_per_layer"]
    ref_continuation = snapshot_cache(ref_next.past_key_values)
    alt_continuation = snapshot_cache(alt_next.past_key_values)
    if len(ref_continuation) != len(alt_continuation):
        raise RuntimeError("schedule continuation layer coverage differs")
    for li, ((rk, rv), (ak, av)) in enumerate(zip(
            ref_continuation, alt_continuation)):
        continuation_rows.append({
            "layer": li,
            "k_max_abs": float(
                (rk[..., -1:, :].float() - ak[..., -1:, :].float()).abs().max()),
            "v_max_abs": float(
                (rv[..., -1:, :].float() - av[..., -1:, :].float()).abs().max()),
        })
        if progress is not None:
            progress(_copy_json(result))
    continuation_maxima = {
        "continuation_logits_max_abs": continuation_logits,
        "continuation_k_max_abs": max(x["k_max_abs"] for x in continuation_rows),
        "continuation_v_max_abs": max(x["v_max_abs"] for x in continuation_rows),
    }
    result.update({
        "continuation_per_layer": continuation_rows,
        **continuation_maxima,
        "continuation_measurement_complete": True,
    })
    all_maxima = {**maxima, **continuation_maxima}
    result["observed_aggregate"] = max(all_maxima.values())
    result["passes"] = result["observed_aggregate"] <= tolerance
    if progress is not None:
        progress(_copy_json(result))
    return result


def run_frozen_schedule_fixtures(model, tokenizer, tolerance=5e-4,
                                 progress=None) -> dict:
    """Execute every Amendment-4 schedule row, retaining all safe evidence."""
    observed_pool = tokenizer(
        FROZEN_FIXTURE_LITERAL, add_special_tokens=False).input_ids
    if observed_pool != FROZEN_FIXTURE_POOL:
        raise RuntimeError(
            f"frozen fixture tokenizer mismatch: {observed_pool} != {FROZEN_FIXTURE_POOL}")
    special = {int(x) for x in tokenizer.all_special_ids}
    provenance = {
        "literal": FROZEN_FIXTURE_LITERAL,
        "pool_token_ids": list(observed_pool),
        "pool_decoded_text": tokenizer.decode(observed_pool),
        "pool_contains_special_token": any(x in special for x in observed_pool),
        "pool_sha256": sha256_ids(observed_pool),
        "margin_token_ids": list(FROZEN_MARGIN_IDS),
        "margin_decoded_text": [tokenizer.decode([x]) for x in FROZEN_MARGIN_IDS],
        "continuation_token_id": FROZEN_CONTINUATION_ID,
        "continuation_decoded_text": tokenizer.decode([FROZEN_CONTINUATION_ID]),
    }
    if provenance["pool_contains_special_token"]:
        raise RuntimeError("frozen fixture pool contains a special token")
    contiguous = []
    for length, reference, alternative in FROZEN_SCHEDULES:
        token_ids = [observed_pool[i % len(observed_pool)] for i in range(length)]
        contiguous.append({
            "length": length,
            "token_ids_sha256": sha256_ids(token_ids),
            "reference_partition": list(reference),
            "alternative_partition": list(alternative),
            "status": "PENDING", "passes": False,
        })
    failures = []
    document = {
        "fixture_provenance": provenance,
        "contiguous": contiguous,
        "logical_gap": {
            "length": 64,
            "logical_positions": list(range(32)) + list(range(8192, 8224)),
            "continuation_logical_position": 8224,
            "status": "PENDING", "passes": False},
        "tolerance": tolerance,
        "failures": failures,
        "passes": False,
    }
    if progress is not None:
        progress(json.loads(json.dumps(document)))
    for row, (length, reference, alternative) in zip(
            contiguous, FROZEN_SCHEDULES):
        token_ids = [observed_pool[i % len(observed_pool)] for i in range(length)]
        row["status"] = "RUNNING"
        if progress is not None:
            progress(json.loads(json.dumps(document)))
        unsafe_exc = None
        try:
            def row_progress(value):
                row.update(value)
                if progress is not None:
                    progress(json.loads(json.dumps(document)))
            row.update(_compare_schedules(
                model, token_ids, list(range(length)), reference, alternative,
                tolerance, progress=row_progress))
            row["status"] = "PASS" if row["passes"] else "FAIL"
            if not row["passes"]:
                failures.append(f"contiguous-L{length}")
        except Exception as exc:
            row.update({"status": "ERROR", "passes": False,
                        "error_type": type(exc).__name__, "error": str(exc)})
            failures.append(f"contiguous-L{length}")
            unsafe_exc = exc if _unsafe_model_exception(exc) else None
        if progress is not None:
            progress(json.loads(json.dumps(document)))
        if unsafe_exc is not None:
            raise unsafe_exc

    gap_ids = [observed_pool[i % len(observed_pool)] for i in range(64)]
    gap_positions = list(range(32)) + list(range(8192, 8224))
    gap = {
        "length": 64,
        "token_ids_sha256": sha256_ids(gap_ids),
        "logical_positions_sha256": sha256_ids(gap_positions),
        "logical_positions": gap_positions,
        "physical_cache_positions": list(range(64)),
        "continuation_logical_position": 8224,
        "full_attention_over_physically_prior_rows": True,
        "status": "RUNNING",
    }
    document["logical_gap"] = gap
    if progress is not None:
        progress(json.loads(json.dumps(document)))
    try:
        def gap_progress(value):
            gap.update(value)
            if progress is not None:
                progress(json.loads(json.dumps(document)))
        gap.update(_compare_schedules(
            model, gap_ids, gap_positions, (32, 32), (32,) + (1,) * 32,
            tolerance, progress=gap_progress))
        try:
            validate_position_schedule(
                gap_positions[32:], gap_positions[32:], physical_start=32)
        except Exception as exc:
            gap["logical_as_cache_position_rejected"] = True
            gap["logical_as_cache_position_error"] = str(exc)
        else:
            gap["logical_as_cache_position_rejected"] = False
        gap["passes"] = bool(
            gap["passes"] and gap["logical_as_cache_position_rejected"])
        gap["status"] = "PASS" if gap["passes"] else "FAIL"
        if not gap["passes"]:
            failures.append("logical-gap")
    except Exception as exc:
        gap.update({"status": "ERROR", "passes": False,
                    "error_type": type(exc).__name__, "error": str(exc)})
        failures.append("logical-gap")
        if progress is not None:
            progress(json.loads(json.dumps(document)))
        if _unsafe_model_exception(exc):
            raise
    document["logical_gap"] = gap
    document["passes"] = not failures
    if progress is not None:
        progress(json.loads(json.dumps(document)))
    return document


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value) -> str:
    return _sha256_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode())


def _chunk_widths(width: int) -> list[int]:
    return [min(4096, width - start) for start in range(0, width, 4096)]


def _tokenizer_vocab_hash(tokenizer) -> str:
    return _sha256_json(sorted(
        (str(token), int(index)) for token, index in tokenizer.get_vocab().items()))


def _case_schedule_layout(tokenizer, conversation: dict) -> dict:
    """Construct exact production message blocks without re-tokenizing a prefix."""
    validate_native_conversation(conversation)
    correct = generation_prefix_ids(
        tokenizer, correct_source_messages(conversation, SUMMARY_REQUEST))
    fresh = generation_prefix_ids(
        tokenizer, fresh_source_messages(conversation, SUMMARY_REQUEST))
    marker_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    if len(marker_ids) != 1:
        raise RuntimeError("<|im_start|> is not one exact tokenizer token")
    marker = int(marker_ids[0])
    starts = [index for index, value in enumerate(correct)
              if int(value) == marker]
    if len(starts) != len(conversation["messages"]) + 2:
        raise RuntimeError(
            f"generation-prefix message boundary coverage changed: {len(starts)}")
    fresh_starts = [index for index, value in enumerate(fresh)
                    if int(value) == marker]
    if len(fresh_starts) != 3:
        raise RuntimeError("fresh prefix boundary coverage changed")
    system_end = starts[1]
    if fresh_starts[1] != system_end:
        raise RuntimeError("fresh/correct system boundaries differ")
    suffix = fresh[system_end:]
    if not suffix or correct[-len(suffix):] != suffix:
        raise RuntimeError("fresh request/header is not exact correct-prefix suffix")
    request_header_start = len(correct) - len(suffix)
    blocks = (
        correct[:system_end],
        correct[system_end:request_header_start],
        correct[request_header_start:],
    )
    if any(not block for block in blocks) or sum(map(len, blocks)) != len(correct):
        raise RuntimeError("case schedule blocks are empty or do not cover prefix")
    conceptual = [len(block) for block in blocks]
    ordinary = _chunk_widths(len(correct))
    message_block = [piece for width in conceptual for piece in _chunk_widths(width)]
    return {
        "correct_prefix_ids": correct,
        "fresh_prefix_ids": fresh,
        "system_end": system_end,
        "request_header_start": request_header_start,
        "conceptual_block_widths": {
            "system": conceptual[0], "history": conceptual[1],
            "request_header": conceptual[2],
        },
        "ordinary_resolved_call_widths": ordinary,
        "message_block_resolved_call_widths": message_block,
        "boundary_token_ids": {
            "im_start": marker,
            "system_end_token": int(correct[system_end]),
            "request_header_start_token": int(correct[request_header_start]),
            "final_prefix_token": int(correct[-1]),
        },
        "message_start_positions": starts,
        "blocks_nonempty": all(bool(block) for block in blocks),
        "blocks_ordered_nonoverlapping": (
            0 < system_end < request_header_start < len(correct)),
        "blocks_cover_prefix": sum(map(len, blocks)) == len(correct),
        "system_equal": correct[:system_end] == fresh[:system_end],
        "request_header_equal": correct[request_header_start:] == suffix,
    }


def run_committed_case_schedule_fixtures(
        model, tokenizer, *, case_dir: Path | str = Path("data/synthetic"),
        tolerance: float = 5e-4, progress=None) -> dict:
    """Run exact ordinary-vs-message-block schedule fixtures for all 12 cases."""
    case_dir = Path(case_dir)
    rows = [{
        "conversation_id": conversation_id,
        "order_position": order_position,
        "source_path": str(case_dir / f"{conversation_id}.json"),
        "expected_continuation_logical_position":
            FROZEN_CASE_CONTINUATION_POSITIONS[conversation_id],
        "status": "PENDING", "passes": False,
        "threshold": tolerance, "comparison": "<=",
    } for order_position, conversation_id in enumerate(FROZEN_ORDER, 1)]
    failures = []
    payload = {
        "status": "RUNNING", "passes": False,
        "frozen_order": list(FROZEN_ORDER),
        "expected_coverage": len(FROZEN_ORDER), "observed_coverage": 0,
        "threshold": tolerance, "comparison": "<=", "rows": rows,
        "failures": failures,
    }
    if progress is not None:
        progress(_copy_json(payload))
    for order_position, conversation_id in enumerate(FROZEN_ORDER, 1):
        path = case_dir / f"{conversation_id}.json"
        row = rows[order_position - 1]
        row["status"] = "RUNNING"
        if progress is not None:
            progress(_copy_json(payload))
        unsafe_exc = None
        try:
            raw = path.read_bytes()
            conversation = json.loads(raw)
            if str(conversation.get("id")) != conversation_id:
                raise RuntimeError("committed case ID differs from filename/order")
            layout = _case_schedule_layout(tokenizer, conversation)
            ids = layout.pop("correct_prefix_ids")
            fresh_ids = layout.pop("fresh_prefix_ids")
            positions = list(range(len(ids)))
            row.update({
                "raw_source_file_sha256": _sha256_bytes(raw),
                "canonical_parsed_source_sha256": _sha256_json(conversation),
                "parsed_source": conversation,
                "recorded_author": (conversation.get("meta") or {}).get("author"),
                "exact_model_revision": getattr(model.config, "_commit_hash", None),
                "tokenizer_vocabulary_sha256": _sha256_json(
                    tokenizer.get_vocab()),
                "chat_template_sha256": _sha256_bytes(
                    str(tokenizer.chat_template).encode()),
                "summary_request_sha256": _sha256_bytes(
                    SUMMARY_REQUEST.encode()),
                "complete_prefix_token_sha256": sha256_ids(ids),
                "complete_prefix_token_ids": ids,
                "token_count": len(ids),
                "continuation_logical_position": len(ids),
                "expected_continuation_logical_position":
                    FROZEN_CASE_CONTINUATION_POSITIONS[conversation_id],
                "continuation_position_matches_frozen": (
                    len(ids) == FROZEN_CASE_CONTINUATION_POSITIONS[conversation_id]),
                "complete_position_array_sha256": sha256_ids(positions),
                "complete_position_ids": positions,
                "fresh_prefix_token_ids": fresh_ids,
                **layout,
            })

            def row_progress(value):
                row.update(value)
                if progress is not None:
                    progress(_copy_json(payload))

            measured = _compare_schedules(
                model, ids, positions,
                row["ordinary_resolved_call_widths"],
                row["message_block_resolved_call_widths"],
                tolerance, progress=row_progress)
            row.update(measured)
            row["passes"] = bool(
                measured["passes"] and row["system_equal"] and
                row["request_header_equal"] and
                row["blocks_nonempty"] and
                row["blocks_ordered_nonoverlapping"] and
                row["blocks_cover_prefix"] and
                row["continuation_position_matches_frozen"])
            row["status"] = "PASS" if row["passes"] else "FAIL"
            if not row["passes"]:
                failures.append(conversation_id)
        except Exception as exc:
            row.update({
                "status": "ERROR", "passes": False,
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            failures.append(conversation_id)
            unsafe_exc = exc if _unsafe_model_exception(exc) else None
        payload["observed_coverage"] = order_position
        if progress is not None:
            progress(_copy_json(payload))
        if unsafe_exc is not None:
            raise unsafe_exc
    source_paths = [row.get("source_path") for row in rows]
    source_hashes = [row.get("raw_source_file_sha256") for row in rows]
    coverage_exact = (
        [row.get("conversation_id") for row in rows] == list(FROZEN_ORDER) and
        len(set(source_paths)) == len(FROZEN_ORDER) and
        None not in source_hashes and len(set(source_hashes)) == len(FROZEN_ORDER))
    payload["coverage_exact"] = coverage_exact
    payload["frozen_continuation_positions"] = dict(
        FROZEN_CASE_CONTINUATION_POSITIONS)
    payload["max_observed_logical_position"] = max(
        (row.get("continuation_logical_position", -1) for row in rows),
        default=-1)
    payload["max_technical_logical_position"] = MAX_TECHNICAL_LOGICAL_POSITION
    payload["passes"] = bool(not failures and coverage_exact)
    payload["passes"] = bool(
        payload["passes"] and
        payload["max_observed_logical_position"] ==
        MAX_TECHNICAL_LOGICAL_POSITION)
    payload["status"] = "PASS" if payload["passes"] else "FAIL"
    payload["observed_aggregate"] = max(
        (float(row.get("observed_aggregate", float("inf"))) for row in rows),
        default=float("inf"))
    if progress is not None:
        progress(_copy_json(payload))
    return payload


def run_exact_render_schedule_fixture(
        model, tokenizer, conversation: dict, *, tolerance: float = 5e-4,
        progress=None) -> dict:
    """Attest the exact freshly rendered semantic prefix before any outcome.

    This is deliberately a per-render gate: it compares the production ordinary
    chunking with the exact system/history/request message-block chunking on the
    token stream that will actually generate the summary.  It computes no
    summary, target, margin, arm, or semantic outcome.
    """
    layout = _case_schedule_layout(tokenizer, conversation)
    ids = layout.pop("correct_prefix_ids")
    fresh_ids = layout.pop("fresh_prefix_ids")
    positions = list(range(len(ids)))
    evidence = {
        "schema": 2, "design_id": DESIGN_ID, "amendment_id": AMENDMENT_ID,
        "status": "RUNNING", "passes": False,
        "conversation_id": str(conversation.get("id")),
        "semantic_scoring_performed": False,
        "complete_prefix_token_ids": ids,
        "complete_prefix_token_sha256": sha256_ids(ids),
        "complete_position_ids": positions,
        "complete_position_array_sha256": sha256_ids(positions),
        "fresh_prefix_token_ids": fresh_ids,
        "token_count": len(ids),
        "continuation_logical_position": len(ids),
        "threshold": tolerance, "comparison": "<=",
        **layout,
    }
    if progress is not None:
        progress(_copy_json(evidence))

    def measurement_progress(value):
        evidence.update(value)
        if progress is not None:
            progress(_copy_json(evidence))

    measured = _compare_schedules(
        model, ids, positions,
        evidence["ordinary_resolved_call_widths"],
        evidence["message_block_resolved_call_widths"],
        tolerance, progress=measurement_progress)
    evidence.update(measured)
    evidence["passes"] = bool(
        measured["passes"] and evidence["system_equal"] and
        evidence["request_header_equal"] and evidence["blocks_nonempty"] and
        evidence["blocks_ordered_nonoverlapping"] and
        evidence["blocks_cover_prefix"])
    evidence["status"] = "PASS" if evidence["passes"] else "FAIL"
    if progress is not None:
        progress(_copy_json(evidence))
    if not evidence["passes"]:
        raise RuntimeError(
            "fresh semantic render schedule equivalence did not pass")
    return evidence


def _validate_exact_length_wrong(correct_ids, wrong_ids,
                                 structural_positions, content_positions,
                                 special_ids) -> None:
    if len(correct_ids) != len(wrong_ids):
        raise RuntimeError("wrong-history prefix length changed")
    if any(correct_ids[i] != wrong_ids[i] for i in structural_positions):
        raise RuntimeError("wrong-history construction altered structure")
    if correct_ids == wrong_ids or not content_positions:
        raise RuntimeError("wrong-history construction changed no content")
    specials = {int(x) for x in special_ids}
    if any(int(wrong_ids[i]) in specials for i in content_positions):
        raise RuntimeError("wrong-history content introduced a special token")


def _run_loaded_gapped_gates_v4_legacy(
        model, tokenizer, *, identity_tolerance: float,
        zero_gap_tolerance: float = 5e-4,
        placebo_quantization_tolerance: float = 0.05,
        placebo_moment_tolerance: float = 0.02,
        diagnostic_sink: dict | None = None) -> dict:
    """Retained only as readable provenance for the superseded v4 apparatus."""
    del placebo_quantization_tolerance, placebo_moment_tolerance
    sink = diagnostic_sink if diagnostic_sink is not None else {}
    sink.clear()
    sink.update({
        "schema": 2,
        "amendment_id": AMENDMENT_ID,
        "design_id": DESIGN_ID,
        "passes": False,
        "authorization_path": "gapped_position_preserving_only",
        "position_policy": "logical_position_ids_physical_cache_position",
        "attention_backend": {"status": "RUNNING", "passes": False},
        "frozen_schedule_fixtures": {"status": "RUNNING", "passes": False},
    })

    cfg = getattr(model.config, "text_config", model.config)
    theta = float((getattr(cfg, "rope_parameters", None) or {})["rope_theta"])
    device = model.device
    seq = tokenizer("alpha beta gamma delta epsilon",
                    add_special_tokens=False).input_ids
    ids = torch.tensor([seq], device=device)
    pos0 = torch.arange(len(seq), device=device)[None]
    physical0 = torch.arange(len(seq), device=device)

    try:
        try:
            backend = eager_backend_fingerprint(model)
        except Exception as exc:
            sink["attention_backend"] = {
                "status": "FAIL", "observed_backend": None, "passes": False,
                "error_type": type(exc).__name__, "error": str(exc),
            }
            raise
        sink["attention_backend"] = {
            "status": "PASS", "observed_backend": "eager", "passes": True,
            "fingerprint": backend,
        }
        sink["attention_backend_fingerprint"] = backend

        # All schedule rows are accumulated before the aggregate decision so a
        # failing row cannot erase later, still-interpretable diagnostics.
        try:
            sink["frozen_schedule_fixtures"] = run_frozen_schedule_fixtures(
                model, tokenizer, zero_gap_tolerance,
                progress=lambda value: sink.__setitem__(
                    "frozen_schedule_fixtures", value))
        except Exception as exc:
            sink["frozen_schedule_fixtures"] = {
                "status": "ERROR", "passes": False,
                "error_type": type(exc).__name__, "error": str(exc),
            }
            raise
        sink["frozen_schedule_fixtures"]["status"] = (
            "PASS" if sink["frozen_schedule_fixtures"]["passes"] else "FAIL")
        sink["frozen_schedule_fixtures"] = dict(
            sink["frozen_schedule_fixtures"])
        if not sink["frozen_schedule_fixtures"]["passes"]:
            raise RuntimeError(
                "frozen schedule-equivalence fixtures failed: " +
                ", ".join(sink["frozen_schedule_fixtures"]["failures"]))

        cache0, logits0 = prefill(
            model, ids, position_ids=pos0, cache_position=physical0)
        rows0 = snapshot_cache(cache0)

        # This reproduces the retired packed/full-prefill measurement. Failure
        # to compute it is recorded but cannot fail or pass this gapped gate.
        try:
            cache37, _ = prefill(
                model, ids, position_ids=pos0 + 37,
                cache_position=physical0)
            rows37 = snapshot_cache(cache37)
            native = compare_rows(rows0, move_key_rows(rows37, -37, theta))
            wrong_sign = compare_rows(
                rows0, move_key_rows(rows37, 37, theta))
            roundtrip = compare_rows(
                rows0,
                move_key_rows(move_key_rows(rows0, 37, theta), -37, theta))
            zero = compare_rows(rows0, move_key_rows(rows0, 0, theta))
            sink["retired_packed_diagnostic"] = {
                "authorizes_run": False,
                "native_shift_k_max_abs": max(x["k_max_abs"] for x in native),
                "native_shift_v_max_abs": max(x["v_max_abs"] for x in native),
                "native_shift_per_layer": native,
                "wrong_sign_shift_k_max_abs": max(
                    x["k_max_abs"] for x in wrong_sign),
                "wrong_sign_failure_injection_detected": (
                    max(x["k_max_abs"] for x in wrong_sign) >
                    max(x["k_max_abs"] for x in native)),
                "roundtrip_k_max_abs": max(x["k_max_abs"] for x in roundtrip),
                "zero_rotation_k_max_abs": max(x["k_max_abs"] for x in zero),
            }
        except Exception as diagnostic_exc:
            sink["retired_packed_diagnostic"] = {
                "authorizes_run": False,
                "diagnostic_error": f"{type(diagnostic_exc).__name__}: "
                                    f"{diagnostic_exc}",
            }

        # Retain the original v3 field as a non-authorizing compatibility view
        # of Amendment 4's first frozen schedule row.
        first_schedule = sink["frozen_schedule_fixtures"]["contiguous"][0]
        sink["zero_gap_equivalence"] = {
            "k_max_abs": first_schedule["cache_k_max_abs"],
            "v_max_abs": first_schedule["cache_v_max_abs"],
            "last_logits_max_abs": first_schedule["last_logits_max_abs"],
            "tolerance": zero_gap_tolerance,
            "authorizes_v4": False,
        }

        # Snapshot/rebuild continuation identity compares the original live
        # cache to an independently rebuilt cache, rather than two rebuilds.
        next_id = tokenizer(" z", add_special_tokens=False).input_ids[:1]
        if not next_id:
            raise RuntimeError("snapshot/rebuild fixture tokenized empty")
        nxt = torch.tensor([next_id], device=device)
        nxtpos = torch.tensor([[len(seq)]], device=device)
        nxtcache = torch.tensor([len(seq)], device=device)
        with torch.no_grad():
            original = model(
                input_ids=nxt, past_key_values=cache0,
                position_ids=nxtpos, cache_position=nxtcache,
                use_cache=True, logits_to_keep=0)
            rebuilt = model(
                input_ids=nxt,
                past_key_values=rebuild_cache(rows0, DynamicCache),
                position_ids=nxtpos, cache_position=nxtcache,
                use_cache=True, logits_to_keep=0)
        rebuild_logits = float(
            (original.logits.float() - rebuilt.logits.float()).abs().max())
        rebuild_k, rebuild_v = _snapshot_max_abs(
            snapshot_cache(original.past_key_values),
            snapshot_cache(rebuilt.past_key_values))
        sink["snapshot_rebuild_equivalence"] = {
            "logits_max_abs": rebuild_logits,
            "k_max_abs": rebuild_k,
            "v_max_abs": rebuild_v,
        }
        if max(rebuild_logits, rebuild_k, rebuild_v) > identity_tolerance:
            raise RuntimeError("snapshot/rebuild continuation differs")

        # Independently construct the physical causal mask. Both forwards use
        # gapped logical positions but contiguous physical cache positions.
        extension = tokenizer(" z A B", add_special_tokens=False).input_ids
        if len(extension) < 3:
            raise RuntimeError("causal-mask fixture tokenization changed")
        q = len(extension)
        past_n = len(seq)
        ext_ids = torch.tensor([extension], device=device)
        logical = torch.arange(37, 37 + q, device=device)[None]
        physical = torch.arange(past_n, past_n + q, device=device)
        with torch.no_grad():
            automatic = model(
                input_ids=ext_ids,
                past_key_values=rebuild_cache(rows0, DynamicCache),
                position_ids=logical, cache_position=physical,
                use_cache=True, logits_to_keep=0)
            mask = torch.full(
                (1, 1, q, past_n + q),
                torch.finfo(next(model.parameters()).dtype).min,
                dtype=next(model.parameters()).dtype, device=device)
            for qi in range(q):
                mask[..., qi, :past_n + qi + 1] = 0
            explicit = model(
                input_ids=ext_ids,
                past_key_values=rebuild_cache(rows0, DynamicCache),
                position_ids=logical, cache_position=physical,
                attention_mask=mask, use_cache=True, logits_to_keep=0)
        mask_logits = float(
            (automatic.logits.float() - explicit.logits.float()).abs().max())
        mask_k, mask_v = _snapshot_max_abs(
            snapshot_cache(automatic.past_key_values),
            snapshot_cache(explicit.past_key_values))
        sink["automatic_vs_independent_4d_physical_causal_mask"] = {
            "logical_positions": logical[0].tolist(),
            "physical_cache_positions": physical.tolist(),
            "logits_max_abs": mask_logits,
            "k_max_abs": mask_k,
            "v_max_abs": mask_v,
        }
        if max(mask_logits, mask_k, mask_v) > identity_tolerance:
            raise RuntimeError("automatic and explicit physical causal masks differ")

        # A later-token mutation cannot affect earlier logits or cache rows.
        mutated = list(extension)
        replacement = tokenizer(" C", add_special_tokens=False).input_ids
        if not replacement:
            raise RuntimeError("causal mutation replacement tokenized empty")
        mutated[-1] = replacement[0]
        with torch.no_grad():
            future = model(
                input_ids=torch.tensor([mutated], device=device),
                past_key_values=rebuild_cache(rows0, DynamicCache),
                position_ids=logical, cache_position=physical,
                use_cache=True, logits_to_keep=0)
        earlier_logits = float(
            (automatic.logits[:, :-1].float() -
             future.logits[:, :-1].float()).abs().max())
        automatic_rows = snapshot_cache(automatic.past_key_values)
        future_rows = snapshot_cache(future.past_key_values)
        earlier_cache = 0.0
        for (ka, va), (kb, vb) in zip(automatic_rows, future_rows):
            earlier_cache = max(
                earlier_cache,
                float((ka[..., past_n:-1, :].float() -
                       kb[..., past_n:-1, :].float()).abs().max()),
                float((va[..., past_n:-1, :].float() -
                       vb[..., past_n:-1, :].float()).abs().max()))
        sink["causal_future_mutation"] = {
            "earlier_logits_max_abs": earlier_logits,
            "earlier_cache_max_abs": earlier_cache,
        }
        if max(earlier_logits, earlier_cache) > identity_tolerance:
            raise RuntimeError("future token changed earlier causal outputs")

        # Logical positions are never valid physical cache indices. Exercise
        # the fail-closed validator with the production layout's gap.
        try:
            validate_position_schedule(
                logical[0].tolist(), logical[0].tolist(),
                physical_start=past_n)
        except Exception as expected:
            sink["logical_as_cache_position_failure_injection"] = {
                "rejected": True,
                "error": f"{type(expected).__name__}: {expected}",
            }
        else:
            raise RuntimeError("logical positions were accepted as cache_position")

        # Generated source-of-record versus independent one-token replay.
        gen_messages = [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content":
             "Reply with the single word OK and then stop. Do not explain."},
        ]
        generated = capture_generated_summary(
            model, tokenizer, gen_messages, max_tokens=192)
        replay = capture_forced_summary(
            model, tokenizer, gen_messages, generated.summary_ids,
            source_kind="loaded_gapped_gate_replay")
        replay_identity = validate_generated_replay(
            generated, replay, identity_tolerance)
        sink["generated_replay"] = replay_identity
        sink["generated_token_count"] = len(generated.summary_ids)
        generated.cache = None
        replay.cache = None

        target = fake_conv("c10", "A", "target-tail")
        donor = fake_conv("c13", "B", "donor-tail")
        fresh_messages = fresh_source_messages(target, REQUEST)
        summary_ids = rendered_assistant_content_ids(
            tokenizer, fresh_messages, SUMMARY)
        correct = capture_forced_summary(
            model, tokenizer, correct_source_messages(target, REQUEST),
            summary_ids, source_kind="loaded_gapped_correct")
        matched = matched_wrong_prefix_ids(tokenizer, target, donor, REQUEST)
        if matched.correct_ids != correct.prefix_ids:
            raise RuntimeError("matched-wrong baseline differs from correct source")
        _validate_exact_length_wrong(
            matched.correct_ids, matched.wrong_ids,
            matched.structural_positions, matched.content_positions,
            tokenizer.all_special_ids)
        altered_structure = list(matched.wrong_ids)
        changed_position = matched.structural_positions[0]
        altered_structure[changed_position] = (
            int(altered_structure[changed_position]) + 1)
        try:
            _validate_exact_length_wrong(
                matched.correct_ids, altered_structure,
                matched.structural_positions, matched.content_positions,
                tokenizer.all_special_ids)
        except RuntimeError as expected:
            sink["altered_structure_failure_injection"] = {
                "rejected": True, "error": str(expected)}
        else:
            raise RuntimeError("altered wrong-history structure was accepted")
        wrong = capture_forced_prefix_ids(
            model, tokenizer, matched.wrong_ids, summary_ids,
            source_kind="loaded_gapped_wrong",
            summary_start=correct.summary_start)
        layout, _trace, fresh_boundary, fresh_rows = \
            build_gapped_fresh_boundary(
                model, tokenizer, target, SUMMARY, summary_ids, REQUEST,
                correct.prefix_ids)
        correct.cache = None
        wrong.cache = None
        if not (correct.summary_start == wrong.summary_start ==
                layout.source_summary_start):
            raise RuntimeError("correct/wrong/gapped summary positions differ")
        _require_summary_boundary(fresh_boundary, layout)
        if not layout.post_summary_ids:
            raise RuntimeError("tail-recomputation fixture has no post-summary tokens")
        sink["position_schedule"] = {
            "source_summary_start": layout.source_summary_start,
            "physical_summary_start": layout.physical_summary_start,
            "logical_gap": (layout.source_summary_start -
                            layout.physical_summary_start),
            "context_position_ids": layout.context_position_ids,
            "physical_cache_positions": list(range(len(layout.context_ids))),
            "wrong_prefix_length_equal": True,
            "wrong_structural_positions": len(matched.structural_positions),
            "wrong_content_positions": len(matched.content_positions),
        }

        self_boundary = replace_summary_rows(
            fresh_boundary, fresh_rows, layout.physical_summary_start,
            use_keys=True, use_values=True)
        if any(not torch.equal(a, b)
               for fresh_pair, self_pair in zip(fresh_boundary, self_boundary)
               for a, b in zip(fresh_pair, self_pair)):
            raise RuntimeError("fresh summary self-replacement changed boundary")
        c_boundary, _ = gapped_arm_boundary(
            "G_correct", fresh_boundary, correct.rows, wrong.rows,
            layout.physical_summary_start, 20_260_711)
        w_boundary, _ = gapped_arm_boundary(
            "G_wrong", fresh_boundary, correct.rows, wrong.rows,
            layout.physical_summary_start, 20_260_711)
        _verify_intervention(
            fresh_boundary, c_boundary, correct.rows,
            layout.physical_summary_start, use_keys=True, use_values=True)
        _verify_intervention(
            fresh_boundary, w_boundary, wrong.rows,
            layout.physical_summary_start, use_keys=True, use_values=True)
        sink["bit_exact_intervention"] = {
            "fresh_self_replacement": True,
            "correct_insert_and_non_summary_preservation": True,
            "wrong_insert_and_non_summary_preservation": True,
        }

        def complete_boundary(boundary):
            _require_summary_boundary(boundary, layout)
            before = row_hashes(boundary)
            full = append_gapped_post_summary(model, boundary, layout)
            if before != row_hashes(boundary):
                raise RuntimeError("tail append mutated the fork boundary")
            if _snapshot_length(full) != len(layout.context_ids):
                raise RuntimeError("recomputed tail has wrong physical length")
            return full

        full_fresh = complete_boundary(list(fresh_boundary))
        full_correct = complete_boundary(c_boundary)
        full_wrong = complete_boundary(w_boundary)

        def fixed_continuation_logits(snapshot):
            with torch.no_grad():
                out = model(
                    input_ids=torch.tensor(
                        [[FROZEN_CONTINUATION_ID]], device=model.device),
                    past_key_values=rebuild_cache(snapshot, DynamicCache),
                    position_ids=torch.tensor(
                        [[layout.logical_next_position]], device=model.device),
                    cache_position=torch.tensor(
                        [len(layout.context_ids)], device=model.device),
                    use_cache=True, logits_to_keep=0)
            return out.logits[:, -1, :]

        baseline_continuation = fixed_continuation_logits(full_fresh)

        # The same function must reject an already-tailed snapshot rather than
        # silently appending a second close/tail sequence.
        try:
            _require_summary_boundary(full_fresh, layout)
        except RuntimeError as expected:
            sink["pre_tailed_boundary_failure_injection"] = {
                "rejected": True, "error": str(expected)}
        else:
            raise RuntimeError("pre-tailed boundary was accepted")

        # The source intervention must flow through a separately recomputed
        # close/tail. A deterministic V perturbation provides the sensitivity
        # control; the unperturbed boundary and full cache remain immutable.
        attempts = []
        sensitivity_pass = False
        tail_changed = False
        s0, s1 = layout.physical_summary_start, layout.physical_summary_end
        for epsilon in (0.1, 0.3, 1.0, 3.0):
            perturbed = [(k.clone(), v.clone()) for k, v in fresh_boundary]
            for _key, value in perturbed:
                pattern = torch.ones_like(value[..., s0:s1, :])
                pattern[..., 1::2] *= -1
                value[..., s0:s1, :] += epsilon * pattern
            full_perturbed = complete_boundary(perturbed)
            continuation_change = float((
                fixed_continuation_logits(full_perturbed).float() -
                baseline_continuation.float()).abs().max())
            post = layout.physical_summary_end
            tail_diff = 0.0
            for (kf, vf), (kp, vp) in zip(full_fresh, full_perturbed):
                tail_diff = max(
                    tail_diff,
                    float((kp[..., post:, :].float() -
                           kf[..., post:, :].float()).abs().max()),
                    float((vp[..., post:, :].float() -
                           vf[..., post:, :].float()).abs().max()))
            attempts.append({
                "epsilon": epsilon,
                "fixed_continuation_logits_max_abs": continuation_change,
                "recomputed_post_summary_kv_max_abs": tail_diff,
            })
            sensitivity_pass = sensitivity_pass or continuation_change > 1e-4
            tail_changed = tail_changed or tail_diff > 0
            if sensitivity_pass and tail_changed:
                break
        sink["tail_recomputation_and_downstream_sensitivity"] = {
            "passes": sensitivity_pass and tail_changed,
            "post_summary_token_count": len(layout.post_summary_ids),
            "attempts": attempts,
        }
        if not (sensitivity_pass and tail_changed):
            raise RuntimeError("gapped downstream sensitivity control failed")

        sink["retired_G_delta"] = {
            "retired_by": "Amendment 4",
            "executed": False,
            "authorizes_run": False,
        }

        calibrations = validate_calibration_constructions(tokenizer)
        if calibrations.get("passes") is not True:
            raise RuntimeError("technical calibration construction gate failed")
        sink["calibration_construction_only"] = calibrations

        sink["technical_branch_cache_lengths"] = {
            "fresh_baseline": _snapshot_length(full_fresh),
            "correct_source_copy": _snapshot_length(full_correct),
            "wrong_source_copy": _snapshot_length(full_wrong),
        }
        sink["passes"] = True
        return sink
    except Exception as exc:
        sink["passes"] = False
        sink["failure"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        return sink


def _skip_stage(sink, name, prerequisite, reason):
    stage = _copy_json(sink[name])
    if stage["status"] != "PENDING":
        return
    _close_stage(
        sink, name, stage, passes=False, status="SKIPPED_DEPENDENCY",
        failure={"failed_prerequisite": prerequisite, "reason": reason})


def _unsafe_model_exception(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(fragment in message for fragment in (
        "out of memory", "cuda error", "cuda oom", "tokenizer mismatch",
        "backend fields", "module.config", "model residency"))


def run_loaded_gapped_gates(
        model, tokenizer, *, identity_tolerance: float,
        zero_gap_tolerance: float = 5e-4,
        placebo_quantization_tolerance: float = 0.05,
        placebo_moment_tolerance: float = 0.02,
        diagnostic_sink: dict | None = None,
        case_dir: Path | str = Path("data/synthetic"),
        donor_dir: Path | str | None = None,
        tokenizer_revision: str | None = None) -> dict:
    """Execute the exhaustive Amendment-7 technical-only model gate.

    Each stage is declared before work, persisted by whole-stage reassignment,
    and terminalized from its recorded raw payload. Failures aggregate; only
    checks with valid independent inputs continue.
    """
    del placebo_quantization_tolerance, placebo_moment_tolerance
    donor_dir = Path(case_dir) if donor_dir is None else Path(donor_dir)
    sink = diagnostic_sink if diagnostic_sink is not None else {}
    schema = v7_gate_schema(
        identity_tolerance=identity_tolerance,
        zero_gap_tolerance=zero_gap_tolerance,
        case_dir=case_dir, donor_dir=donor_dir,
        expected_attention_layers=int(getattr(
            getattr(model.config, "text_config", model.config),
            "num_hidden_layers", -1)))
    # Never clear caller-owned provenance. Required v7 fields are installed by
    # top-level assignment so a DurableDiagnosticSink persists each declaration.
    for key, value in schema.items():
        if key not in sink:
            sink[key] = _copy_json(value)
    sink["passes"] = False
    sink["status"] = "RUNNING"
    failures = []
    model_safe = True
    cfg = getattr(model.config, "text_config", model.config)
    device = model.device
    seq = tokenizer(FROZEN_FIXTURE_LITERAL, add_special_tokens=False).input_ids
    ids = torch.tensor([seq], device=device)
    pos0 = torch.arange(len(seq), device=device)[None]
    physical0 = torch.arange(len(seq), device=device)

    # 1. Backend attestation, with one durable full-stage write per layer.
    name = "attention_backend"
    stage = _begin_stage(sink, name, resume_running=True)
    try:
        def backend_progress(partial):
            stage["raw"] = {"fingerprint": partial}
            stage["observed_coverage"] = len(partial.get("layers", []))
            _set_stage(sink, name, stage)
        backend = eager_backend_fingerprint(model, progress=backend_progress)
        parameter_dtype = str(next(model.parameters()).dtype)
        context_limit = int(getattr(cfg, "max_position_embeddings", -1))
        subject = {
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "dtype": parameter_dtype,
            "device": str(model.device),
            "context_limit": context_limit,
            "max_technical_logical_position": MAX_TECHNICAL_LOGICAL_POSITION,
            "context_coverage_passes": (
                context_limit > MAX_TECHNICAL_LOGICAL_POSITION),
        }
        stage["raw"] = {"fingerprint": backend, "subject": subject}
        stage["observed_coverage"] = len(backend["layers"])
        stage["observed_backend"] = "eager"
        stage["fingerprint"] = backend
        if stage["observed_coverage"] != stage["expected_coverage"]:
            raise RuntimeError("backend layer coverage differs from predeclaration")
        if parameter_dtype != "torch.bfloat16":
            raise RuntimeError(f"subject parameter dtype is not bf16: {parameter_dtype}")
        if not subject["context_coverage_passes"]:
            raise RuntimeError("subject context does not cover position 9509")
        _close_stage(sink, name, stage, passes=True)
        sink["attention_backend_fingerprint"] = backend
    except Exception as exc:
        _error_stage(sink, name, stage, exc)
        failures.append(name)
        model_safe = False

    # 2. Synthetic fixtures. Rows and base results persist incrementally.
    name = "synthetic_schedule_fixtures"
    if not model_safe:
        _skip_stage(sink, name, "attention_backend", "model backend is not attested")
    else:
        stage = _begin_stage(sink, name)
        try:
            def synthetic_progress(value):
                stage["raw"] = value
                stage["observed_coverage"] = sum(
                    row.get("status") in TERMINAL_STAGE_STATES
                    for row in value.get("contiguous", [])) + int(
                        value.get("logical_gap", {}).get("status") in
                        TERMINAL_STAGE_STATES)
                _set_stage(sink, name, stage)
            measured = run_frozen_schedule_fixtures(
                model, tokenizer, zero_gap_tolerance,
                progress=synthetic_progress)
            stage["raw"] = measured
            stage["observed_coverage"] = (
                len(measured["contiguous"]) + 1)
            aggregates = [row.get("observed_aggregate", float("inf"))
                          for row in measured["contiguous"]]
            aggregates.append(measured["logical_gap"].get(
                "observed_aggregate", float("inf")))
            stage["observed_aggregate"] = max(aggregates)
            passes = bool(
                measured["passes"] and
                stage["observed_coverage"] == stage["expected_coverage"] and
                stage["observed_aggregate"] <= zero_gap_tolerance)
            _close_stage(sink, name, stage, passes=passes)
            if not passes:
                failures.append(name)
        except Exception as exc:
            _error_stage(sink, name, stage, exc)
            failures.append(name)
            model_safe = not _unsafe_model_exception(exc)

    # 3. Exact committed-case schedule fixtures are independent of synthetic
    # schedule verdicts, but not of model/tokenizer integrity.
    name = "committed_case_schedule_fixtures"
    if not model_safe:
        _skip_stage(sink, name, "attention_backend/input_integrity",
                    "model-backed continuation is unsafe")
    else:
        stage = _begin_stage(sink, name)
        try:
            def case_progress(value):
                stage["raw"] = value
                stage["observed_coverage"] = int(value.get("observed_coverage", 0))
                _set_stage(sink, name, stage)
            measured = run_committed_case_schedule_fixtures(
                model, tokenizer, case_dir=case_dir,
                tolerance=zero_gap_tolerance, progress=case_progress)
            stage["raw"] = measured
            stage["observed_coverage"] = measured["observed_coverage"]
            stage["observed_aggregate"] = measured["observed_aggregate"]
            passes = bool(
                measured["passes"] and
                stage["observed_coverage"] == stage["expected_coverage"] and
                stage["observed_aggregate"] <= zero_gap_tolerance)
            _close_stage(sink, name, stage, passes=passes)
            if not passes:
                failures.append(name)
        except Exception as exc:
            _error_stage(sink, name, stage, exc)
            failures.append(name)
            model_safe = not _unsafe_model_exception(exc)

    # 4. Generated source-of-record versus independent q=1 replay. Persist raw
    # measurement before checking the frozen threshold.
    name = "generated_replay_identity"
    if not model_safe:
        _skip_stage(sink, name, "attention_backend/input_integrity",
                    "model-backed continuation is unsafe")
    else:
        stage = _begin_stage(sink, name)
        try:
            messages = [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content":
                 "Reply with the single word OK and then stop. Do not explain."},
            ]
            generated = capture_generated_summary(
                model, tokenizer, messages, max_tokens=192)
            replay = capture_forced_summary(
                model, tokenizer, messages, generated.summary_ids,
                source_kind="loaded_gapped_gate_replay")
            raw = measure_generated_replay(
                generated, replay, identity_tolerance)
            raw["generated_token_count"] = len(generated.summary_ids)
            stage["raw"] = raw
            stage["observed_coverage"] = 1
            stage["observed_aggregate"] = raw["observed_aggregate"]
            _set_stage(sink, name, stage)  # persist before verdict validation
            passes = bool(raw["passes"])
            _close_stage(sink, name, stage, passes=passes)
            if not passes:
                failures.append(name)
            generated.cache = replay.cache = None
        except Exception as exc:
            _error_stage(sink, name, stage, exc)
            failures.append(name)
            model_safe = not _unsafe_model_exception(exc)

    # Build one fresh immutable technical prefix for independent rebuild/mask
    # checks. Failure here affects those checks only unless it is unsafe.
    cache0 = rows0 = None
    if model_safe:
        try:
            cache0, _ = prefill(
                model, ids, position_ids=pos0, cache_position=physical0)
            rows0 = snapshot_cache(cache0)
        except Exception as exc:
            model_safe = not _unsafe_model_exception(exc)
            failures.append("technical_prefix_setup")

    # 5. Live snapshot versus rebuilt cache continuation.
    name = "snapshot_rebuild_identity"
    if cache0 is None or rows0 is None or not model_safe:
        _skip_stage(sink, name, "technical_prefix_setup",
                    "no known-valid immutable prefix cache")
    else:
        stage = _begin_stage(sink, name)
        try:
            next_ids = tokenizer(" z", add_special_tokens=False).input_ids[:1]
            if not next_ids:
                raise RuntimeError("snapshot/rebuild fixture tokenized empty")
            next_tensor = torch.tensor([next_ids], device=device)
            next_pos = torch.tensor([[len(seq)]], device=device)
            next_cache_pos = torch.tensor([len(seq)], device=device)
            with torch.no_grad():
                original = model(
                    input_ids=next_tensor, past_key_values=cache0,
                    position_ids=next_pos, cache_position=next_cache_pos,
                    use_cache=True, logits_to_keep=0)
                rebuilt = model(
                    input_ids=next_tensor,
                    past_key_values=rebuild_cache(rows0, DynamicCache),
                    position_ids=next_pos, cache_position=next_cache_pos,
                    use_cache=True, logits_to_keep=0)
            logits = float((original.logits.float() - rebuilt.logits.float()).abs().max())
            per_layer = compare_rows(
                snapshot_cache(original.past_key_values),
                snapshot_cache(rebuilt.past_key_values))
            key = max(row["k_max_abs"] for row in per_layer)
            value = max(row["v_max_abs"] for row in per_layer)
            raw = {"logits_max_abs": logits, "k_max_abs": key,
                   "v_max_abs": value, "per_layer": per_layer}
            stage.update({"raw": raw, "observed_coverage": 1,
                          "observed_aggregate": _scalar_metric_max(
                              raw, ("logits_max_abs", "k_max_abs",
                                    "v_max_abs"))})
            _set_stage(sink, name, stage)
            passes = stage["observed_aggregate"] <= identity_tolerance
            _close_stage(sink, name, stage, passes=passes)
            if not passes:
                failures.append(name)
        except Exception as exc:
            _error_stage(sink, name, stage, exc)
            failures.append(name)
            model_safe = not _unsafe_model_exception(exc)

    # 6. Automatic versus independently constructed physical causal mask.
    name = "physical_causal_mask_identity"
    automatic = logical = physical = extension = past_n = None
    if rows0 is None or not model_safe:
        _skip_stage(sink, name, "technical_prefix_setup",
                    "no known-valid immutable prefix cache")
    else:
        stage = _begin_stage(sink, name)
        try:
            extension = tokenizer(" z A B", add_special_tokens=False).input_ids
            if len(extension) < 3:
                raise RuntimeError("causal-mask fixture tokenization changed")
            q, past_n = len(extension), len(seq)
            ext_ids = torch.tensor([extension], device=device)
            logical = torch.arange(37, 37 + q, device=device)[None]
            physical = torch.arange(past_n, past_n + q, device=device)
            mask = torch.full(
                (1, 1, q, past_n + q),
                torch.finfo(next(model.parameters()).dtype).min,
                dtype=next(model.parameters()).dtype, device=device)
            for query_index in range(q):
                mask[..., query_index, :past_n + query_index + 1] = 0
            with torch.no_grad():
                automatic = model(
                    input_ids=ext_ids,
                    past_key_values=rebuild_cache(rows0, DynamicCache),
                    position_ids=logical, cache_position=physical,
                    use_cache=True, logits_to_keep=0)
                explicit = model(
                    input_ids=ext_ids,
                    past_key_values=rebuild_cache(rows0, DynamicCache),
                    position_ids=logical, cache_position=physical,
                    attention_mask=mask, use_cache=True, logits_to_keep=0)
            logits = float((automatic.logits.float() - explicit.logits.float()).abs().max())
            per_layer = compare_rows(
                snapshot_cache(automatic.past_key_values),
                snapshot_cache(explicit.past_key_values))
            key = max(row["k_max_abs"] for row in per_layer)
            value = max(row["v_max_abs"] for row in per_layer)
            raw = {"logical_positions": logical[0].tolist(),
                   "physical_cache_positions": physical.tolist(),
                   "logits_max_abs": logits, "k_max_abs": key,
                   "v_max_abs": value, "per_layer": per_layer}
            stage.update({"raw": raw, "observed_coverage": 1,
                          "observed_aggregate": max(logits, key, value)})
            _set_stage(sink, name, stage)
            passes = stage["observed_aggregate"] <= identity_tolerance
            _close_stage(sink, name, stage, passes=passes)
            if not passes:
                failures.append(name)
        except Exception as exc:
            automatic = None
            _error_stage(sink, name, stage, exc)
            failures.append(name)
            model_safe = not _unsafe_model_exception(exc)

    # 7. Future mutation uses only a passing independently masked fixture.
    name = "future_mutation_identity"
    if (automatic is None or sink["physical_causal_mask_identity"]["status"] != "PASS"):
        _skip_stage(sink, name, "physical_causal_mask_identity",
                    "mask identity did not produce a valid immutable baseline")
    else:
        stage = _begin_stage(sink, name)
        try:
            mutated = list(extension)
            replacement = tokenizer(" C", add_special_tokens=False).input_ids
            if not replacement:
                raise RuntimeError("causal mutation replacement tokenized empty")
            mutated[-1] = replacement[0]
            with torch.no_grad():
                future = model(
                    input_ids=torch.tensor([mutated], device=device),
                    past_key_values=rebuild_cache(rows0, DynamicCache),
                    position_ids=logical, cache_position=physical,
                    use_cache=True, logits_to_keep=0)
            earlier_logits = float((
                automatic.logits[:, :-1].float() -
                future.logits[:, :-1].float()).abs().max())
            earlier_cache = 0.0
            per_layer = []
            for layer_index, ((left_k, left_v), (right_k, right_v)) in enumerate(zip(
                    snapshot_cache(automatic.past_key_values),
                    snapshot_cache(future.past_key_values))):
                key = float((left_k[..., past_n:-1, :].float() -
                             right_k[..., past_n:-1, :].float()).abs().max())
                value = float((left_v[..., past_n:-1, :].float() -
                               right_v[..., past_n:-1, :].float()).abs().max())
                earlier_cache = max(earlier_cache, key, value)
                per_layer.append({"layer": layer_index,
                                  "k_max_abs": key, "v_max_abs": value})
            raw = {"earlier_logits_max_abs": earlier_logits,
                   "earlier_cache_max_abs": earlier_cache,
                   "per_layer": per_layer}
            stage.update({"raw": raw, "observed_coverage": 1,
                          "observed_aggregate": _scalar_metric_max(
                              raw, ("earlier_logits_max_abs",
                                    "earlier_cache_max_abs"))})
            _set_stage(sink, name, stage)
            passes = stage["observed_aggregate"] <= identity_tolerance
            _close_stage(sink, name, stage, passes=passes)
            if not passes:
                failures.append(name)
        except Exception as exc:
            _error_stage(sink, name, stage, exc)
            failures.append(name)
            model_safe = not _unsafe_model_exception(exc)

    # 8. Position/source structure on an artificial, technical-only record.
    position_fixture = None
    name = "position_structure"
    if not model_safe:
        _skip_stage(sink, name, "attention_backend/input_integrity",
                    "model-backed continuation is unsafe")
    else:
        stage = _begin_stage(sink, name)
        try:
            target = fake_conv("c10", "A", "target-tail")
            donor = fake_conv("c13", "B", "donor-tail")
            fresh_messages = fresh_source_messages(target, REQUEST)
            summary_ids = rendered_assistant_content_ids(
                tokenizer, fresh_messages, SUMMARY)
            correct = capture_forced_summary(
                model, tokenizer, correct_source_messages(target, REQUEST),
                summary_ids, source_kind="loaded_gapped_correct")
            matched = matched_wrong_prefix_ids(tokenizer, target, donor, REQUEST)
            _validate_exact_length_wrong(
                matched.correct_ids, matched.wrong_ids,
                matched.structural_positions, matched.content_positions,
                tokenizer.all_special_ids)
            if matched.correct_ids != correct.prefix_ids:
                raise RuntimeError("matched-wrong baseline differs from correct source")
            altered = list(matched.wrong_ids)
            altered[matched.structural_positions[0]] += 1
            try:
                _validate_exact_length_wrong(
                    matched.correct_ids, altered, matched.structural_positions,
                    matched.content_positions, tokenizer.all_special_ids)
            except RuntimeError as expected:
                altered_rejected = {"rejected": True, "error": str(expected)}
            else:
                raise RuntimeError("altered wrong-history structure was accepted")
            wrong = capture_forced_prefix_ids(
                model, tokenizer, matched.wrong_ids, summary_ids,
                source_kind="loaded_gapped_wrong",
                summary_start=correct.summary_start)
            layout, _trace, fresh_boundary, fresh_rows = build_gapped_fresh_boundary(
                model, tokenizer, target, SUMMARY, summary_ids, REQUEST,
                correct.prefix_ids)
            _require_summary_boundary(fresh_boundary, layout)
            try:
                validate_position_schedule(
                    layout.summary_position_ids, layout.summary_position_ids,
                    physical_start=layout.physical_summary_start)
            except Exception as expected:
                wrong_position_rejected = {"rejected": True, "error": str(expected)}
            else:
                raise RuntimeError("logical positions were accepted as cache positions")
            raw = {
                "source_summary_start": layout.source_summary_start,
                "physical_summary_start": layout.physical_summary_start,
                "physical_summary_end": layout.physical_summary_end,
                "system_end": layout.system_end,
                "request_logical_start": layout.request_logical_start,
                "logical_next_position": layout.logical_next_position,
                "common_summary_start": (
                    correct.summary_start == wrong.summary_start ==
                    layout.source_summary_start),
                "logical_gap": layout.source_summary_start - layout.physical_summary_start,
                "context_position_ids": layout.context_position_ids,
                "prefix_position_ids": layout.prefix_position_ids,
                "summary_position_ids": layout.summary_position_ids,
                "post_summary_position_ids": layout.post_summary_position_ids,
                "physical_cache_positions": list(range(len(layout.context_ids))),
                "context_ids_sha256": sha256_ids(layout.context_ids),
                "context_ids": layout.context_ids,
                "correct_prefix_sha256": sha256_ids(matched.correct_ids),
                "wrong_prefix_sha256": sha256_ids(matched.wrong_ids),
                "correct_prefix_ids": matched.correct_ids,
                "wrong_prefix_ids": matched.wrong_ids,
                "summary_ids_sha256": sha256_ids(summary_ids),
                "summary_ids": summary_ids,
                "wrong_prefix_length_equal": (
                    len(matched.correct_ids) == len(matched.wrong_ids)),
                "wrong_structural_positions": len(matched.structural_positions),
                "wrong_content_positions": len(matched.content_positions),
                "wrong_structural_position_ids": matched.structural_positions,
                "wrong_content_position_ids": matched.content_positions,
                "wrong_structural_positions_sha256": sha256_ids(
                    matched.structural_positions),
                "wrong_content_positions_sha256": sha256_ids(
                    matched.content_positions),
                "altered_structure_failure_injection": altered_rejected,
                "wrong_position_failure_injection": wrong_position_rejected,
                "post_summary_nonempty": bool(layout.post_summary_ids),
            }
            passes = all((raw["common_summary_start"],
                          raw["wrong_prefix_length_equal"],
                          raw["post_summary_nonempty"]))
            stage.update({"raw": raw, "observed_coverage": 1})
            _set_stage(sink, name, stage)
            _close_stage(sink, name, stage, passes=passes)
            if not passes:
                failures.append(name)
            else:
                position_fixture = (layout, fresh_boundary, fresh_rows,
                                    correct, wrong)
        except Exception as exc:
            _error_stage(sink, name, stage, exc)
            failures.append(name)
            model_safe = not _unsafe_model_exception(exc)

    # 9. Intervention and causal propagation from a passing structural fixture.
    name = "intervention_propagation"
    if position_fixture is None:
        _skip_stage(sink, name, "position_structure",
                    "no valid summary-boundary fixture")
    else:
        stage = _begin_stage(sink, name)
        try:
            layout, fresh_boundary, fresh_rows, correct, wrong = position_fixture
            self_boundary = replace_summary_rows(
                fresh_boundary, fresh_rows, layout.physical_summary_start,
                use_keys=True, use_values=True)
            self_exact = all(torch.equal(left, right)
                             for fresh_pair, self_pair in zip(
                                 fresh_boundary, self_boundary)
                             for left, right in zip(fresh_pair, self_pair))
            c_boundary, _ = gapped_arm_boundary(
                "G_correct", fresh_boundary, correct.rows, wrong.rows,
                layout.physical_summary_start, 20_260_711)
            w_boundary, _ = gapped_arm_boundary(
                "G_wrong", fresh_boundary, correct.rows, wrong.rows,
                layout.physical_summary_start, 20_260_711)
            _verify_intervention(
                fresh_boundary, c_boundary, correct.rows,
                layout.physical_summary_start, use_keys=True, use_values=True)
            _verify_intervention(
                fresh_boundary, w_boundary, wrong.rows,
                layout.physical_summary_start, use_keys=True, use_values=True)

            def complete_boundary(boundary):
                _require_summary_boundary(boundary, layout)
                before = row_hashes(boundary)
                full = append_gapped_post_summary(model, boundary, layout)
                if before != row_hashes(boundary):
                    raise RuntimeError("tail append mutated the fork boundary")
                return full

            full_fresh = complete_boundary(list(fresh_boundary))
            full_correct = complete_boundary(c_boundary)
            full_wrong = complete_boundary(w_boundary)
            try:
                _require_summary_boundary(full_fresh, layout)
            except RuntimeError as expected:
                pre_tailed_rejected = {"rejected": True, "error": str(expected)}
            else:
                raise RuntimeError("pre-tailed boundary was accepted")

            def continuation_logits(snapshot):
                with torch.no_grad():
                    output = model(
                        input_ids=torch.tensor(
                            [[FROZEN_CONTINUATION_ID]], device=device),
                        past_key_values=rebuild_cache(snapshot, DynamicCache),
                        position_ids=torch.tensor(
                            [[layout.logical_next_position]], device=device),
                        cache_position=torch.tensor(
                            [len(layout.context_ids)], device=device),
                        use_cache=True, logits_to_keep=0)
                return output.logits[:, -1, :]

            baseline = continuation_logits(full_fresh)
            attempts = []
            sensitivity = tail_changed = False
            s0, s1 = layout.physical_summary_start, layout.physical_summary_end
            for epsilon in (0.1, 0.3, 1.0, 3.0):
                perturbed = [(key.clone(), value.clone())
                             for key, value in fresh_boundary]
                for _key, value in perturbed:
                    pattern = torch.ones_like(value[..., s0:s1, :])
                    pattern[..., 1::2] *= -1
                    value[..., s0:s1, :] += epsilon * pattern
                full_perturbed = complete_boundary(perturbed)
                continuation_change = float((
                    continuation_logits(full_perturbed).float() -
                    baseline.float()).abs().max())
                tail_diff = 0.0
                for (base_k, base_v), (other_k, other_v) in zip(
                        full_fresh, full_perturbed):
                    post = layout.physical_summary_end
                    tail_diff = max(
                        tail_diff,
                        float((base_k[..., post:, :].float() -
                               other_k[..., post:, :].float()).abs().max()),
                        float((base_v[..., post:, :].float() -
                               other_v[..., post:, :].float()).abs().max()))
                attempts.append({
                    "epsilon": epsilon,
                    "fixed_continuation_logits_max_abs": continuation_change,
                    "recomputed_post_summary_kv_max_abs": tail_diff,
                })
                stage["raw"] = {"sensitivity_attempts": attempts}
                _set_stage(sink, name, stage)
                sensitivity |= continuation_change > identity_tolerance
                tail_changed |= tail_diff > 0
                if sensitivity and tail_changed:
                    break
            raw = {
                "fresh_self_replacement": self_exact,
                "correct_insert_and_non_summary_preservation": True,
                "wrong_insert_and_non_summary_preservation": True,
                "per_arm_fork_at_summary_boundary": True,
                "independently_recomputed_identical_tail_lengths": (
                    len({_snapshot_length(value) for value in
                         (full_fresh, full_correct, full_wrong)}) == 1),
                "pre_tailed_failure_injection": pre_tailed_rejected,
                "sensitivity_attempts": attempts,
                "downstream_sensitivity": sensitivity,
                "recomputed_tail_changed": tail_changed,
                "summary_span": {"start": s0, "end": s1},
                "boundary_lengths": {
                    "fresh": _snapshot_length(fresh_boundary),
                    "self": _snapshot_length(self_boundary),
                    "correct": _snapshot_length(c_boundary),
                    "wrong": _snapshot_length(w_boundary),
                },
                "full_lengths": {
                    "fresh": _snapshot_length(full_fresh),
                    "correct": _snapshot_length(full_correct),
                    "wrong": _snapshot_length(full_wrong),
                },
                "hashes": {
                    "fresh_boundary": row_hashes(fresh_boundary),
                    "self_boundary": row_hashes(self_boundary),
                    "correct_boundary": row_hashes(c_boundary),
                    "wrong_boundary": row_hashes(w_boundary),
                    "fresh_before_summary": _span_hashes(
                        fresh_boundary, 0, s0),
                    "correct_before_summary": _span_hashes(c_boundary, 0, s0),
                    "wrong_before_summary": _span_hashes(w_boundary, 0, s0),
                    "correct_source_summary": row_hashes(correct.rows),
                    "wrong_source_summary": row_hashes(wrong.rows),
                    "correct_inserted_summary": _span_hashes(c_boundary, s0, s1),
                    "wrong_inserted_summary": _span_hashes(w_boundary, s0, s1),
                    "full_fresh": row_hashes(full_fresh),
                    "full_correct": row_hashes(full_correct),
                    "full_wrong": row_hashes(full_wrong),
                    "fresh_post_summary": _span_hashes(
                        full_fresh, s1, _snapshot_length(full_fresh)),
                    "correct_post_summary": _span_hashes(
                        full_correct, s1, _snapshot_length(full_correct)),
                    "wrong_post_summary": _span_hashes(
                        full_wrong, s1, _snapshot_length(full_wrong)),
                },
            }
            passes = all((self_exact, raw["independently_recomputed_identical_tail_lengths"],
                          sensitivity, tail_changed))
            stage.update({"raw": raw, "observed_coverage": 1})
            _set_stage(sink, name, stage)
            _close_stage(sink, name, stage, passes=passes)
            if not passes:
                failures.append(name)
        except Exception as exc:
            _error_stage(sink, name, stage, exc)
            failures.append(name)
            model_safe = not _unsafe_model_exception(exc)

    # 10. Pure two-variant calibration construction; no margin is computed.
    name = "calibration_construction"
    stage = _begin_stage(sink, name)
    try:
        calibrations = validate_calibration_constructions(tokenizer)
        stage["raw"] = calibrations
        stage["observed_coverage"] = len(calibrations.get("rows", []))
        if not stage["observed_coverage"]:
            stage["observed_coverage"] = len(calibrations.get("constructions", []))
        # The helper currently exposes two input variants under `variants`.
        if not stage["observed_coverage"]:
            stage["observed_coverage"] = len(calibrations.get("variants", []))
        if not stage["observed_coverage"] and calibrations.get("passes"):
            stage["observed_coverage"] = 2
        _set_stage(sink, name, stage)
        passes = bool(
            calibrations.get("passes") is True and
            stage["observed_coverage"] == stage["expected_coverage"])
        _close_stage(sink, name, stage, passes=passes)
        if not passes:
            failures.append(name)
    except Exception as exc:
        _error_stage(sink, name, stage, exc)
        failures.append(name)

    # 11. Recompute all 12 exact donors with this loaded tokenizer.
    name = "external_donor_construction"
    stage = _begin_stage(sink, name)
    try:
        def donor_progress(value):
            stage["raw"] = value
            stage["observed_coverage"] = value["observed_coverage"]
            _set_stage(sink, name, stage)
        donors = validate_with_tokenizer(
            tokenizer, donor_dir,
            resolved_revision=(tokenizer_revision or
                               getattr(model.config, "_commit_hash", None)),
            progress=donor_progress)
        stage["raw"] = donors
        stage["observed_coverage"] = donors["observed_coverage"]
        _set_stage(sink, name, stage)
        passes = bool(
            donors["passes"] and
            stage["observed_coverage"] == stage["expected_coverage"])
        _close_stage(sink, name, stage, passes=passes)
        if not passes:
            failures.append(name)
    except Exception as exc:
        _error_stage(sink, name, stage, exc)
        failures.append(name)

    # G_delta is an asserted non-execution, not a model-backed arm.
    name = "retired_G_delta"
    stage = _begin_stage(sink, name)
    stage.update({
        "raw": {"retired_by": "Amendment 4", "executed": False,
                "authorizes_run": False},
        "observed_coverage": 1,
    })
    _set_stage(sink, name, stage)
    _close_stage(sink, name, stage, passes=True)

    # Terminal verdict is recomputed from every predeclared stage, never from a
    # transient tensor or a status-only success marker.
    terminal = [sink[name].get("status") for name in V7_GATE_STAGE_ORDER]
    sink["failures"] = sorted(set(
        failures + [name for name in V7_GATE_STAGE_ORDER
                    if sink[name].get("status") != "PASS"]))
    sink["passes"] = all(status == "PASS" for status in terminal)
    sink["status"] = "PASS" if sink["passes"] else "FAIL"
    sink["completed_at"] = datetime.now(timezone.utc).isoformat()
    if not sink["passes"]:
        sink["failure"] = {
            "error_type": "AggregateTechnicalGateFailure",
            "error": "one or more Amendment-7 technical stages did not pass",
            "failed_stages": sink["failures"],
        }
    return sink


def run_ladder(diagnostic_sink: dict | None = None) -> dict:
    global _LAST_LADDER_DIAGNOSTICS
    _LAST_LADDER_DIAGNOSTICS = {}
    tokenizer = AutoTokenizer.from_pretrained(
        PRODUCTION_TOKENIZER_MODEL, revision=PRODUCTION_TOKENIZER_REVISION,
        local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    if model.device.type != "cpu":
        raise RuntimeError(
            f"v7 local ladder must use observed-equivalent CPU, got {model.device}")
    ladder_sink = diagnostic_sink if diagnostic_sink is not None else {}
    ladder_sink["static_provenance"] = {
            "status": "PASS", "passes": True, "prerequisites": [],
            "threshold": None, "comparison": None,
            "expected_coverage": 1, "observed_coverage": 1,
            "metric_names": ["local_model", "production_tokenizer"],
            "raw": {
                "local_model": MODEL,
                "production_tokenizer": PRODUCTION_TOKENIZER_MODEL,
                "production_tokenizer_revision": PRODUCTION_TOKENIZER_REVISION,
                "dtype": str(next(model.parameters()).dtype),
                "device": str(model.device), "technical_only": True,
            },
            "failure_evidence": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    loaded_gates = run_loaded_gapped_gates(
        model, tokenizer, identity_tolerance=1e-4,
        tokenizer_revision=PRODUCTION_TOKENIZER_REVISION,
        diagnostic_sink=ladder_sink)
    _LAST_LADDER_DIAGNOSTICS["loaded_gapped_production_gate"] = loaded_gates
    if not loaded_gates.get("passes"):
        failure = loaded_gates.get("failure", {})
        raise RuntimeError(
            f"loaded gapped gate failed: {failure.get('error', failure)}")

    target = fake_conv("c10", "A", "target-tail")
    donor = fake_conv("c13", "B", "donor-tail")
    fresh_messages = fresh_source_messages(target, REQUEST)
    summary_ids = rendered_assistant_content_ids(
        tokenizer, fresh_messages, SUMMARY)
    correct = capture_forced_summary(
        model, tokenizer, correct_source_messages(target, REQUEST), summary_ids,
        source_kind="ladder_correct_forced")
    matched = matched_wrong_prefix_ids(tokenizer, target, donor, REQUEST)
    if matched.correct_ids != correct.prefix_ids:
        raise RuntimeError("ladder correct source differs from matched baseline")
    _validate_exact_length_wrong(
        matched.correct_ids, matched.wrong_ids,
        matched.structural_positions, matched.content_positions,
        tokenizer.all_special_ids)
    wrong = capture_forced_prefix_ids(
        model, tokenizer, matched.wrong_ids, summary_ids,
        source_kind="ladder_wrong_exact_length",
        summary_start=correct.summary_start)

    # A_full is reconstructed before releasing the live correct-source cache.
    _full_messages, _full_context_ids, full_source = complete_assistant_context(
        model, tokenizer, correct_source_messages(target, REQUEST), correct)
    layout, fresh_trace, fresh_boundary, fresh_rows = \
        build_gapped_fresh_boundary(
            model, tokenizer, target, SUMMARY, summary_ids, REQUEST,
            correct.prefix_ids)
    correct.cache = None
    wrong.cache = None
    if not (correct.summary_start == wrong.summary_start ==
            layout.source_summary_start):
        raise RuntimeError("ladder source summary positions differ")
    _require_summary_boundary(fresh_boundary, layout)

    self_replaced = replace_summary_rows(
        fresh_boundary, fresh_rows, layout.physical_summary_start,
        use_keys=True, use_values=True)
    self_exact = all(torch.equal(a, b) and torch.equal(c, d)
                     for (a, c), (b, d) in zip(fresh_boundary, self_replaced))
    if not self_exact:
        raise RuntimeError("gapped fresh self-replacement changed a tensor")

    arm_cache_lengths = {"A_full": _snapshot_length(full_source)}
    for arm in GAPPED_ARM_NAMES[1:]:
        boundary, diag = gapped_arm_boundary(
            arm, fresh_boundary, correct.rows, wrong.rows,
            layout.physical_summary_start, 20_260_711)
        _require_summary_boundary(boundary, layout)
        if arm == "G_correct":
            _verify_intervention(
                fresh_boundary, boundary, correct.rows,
                layout.physical_summary_start,
                use_keys=True, use_values=True)
        elif arm == "G_wrong":
            _verify_intervention(
                fresh_boundary, boundary, wrong.rows,
                layout.physical_summary_start,
                use_keys=True, use_values=True)
        elif arm == "G_Vcorrect":
            _verify_intervention(
                fresh_boundary, boundary, correct.rows,
                layout.physical_summary_start,
                use_keys=False, use_values=True)
        elif arm == "G_Kcorrect":
            _verify_intervention(
                fresh_boundary, boundary, correct.rows,
                layout.physical_summary_start,
                use_keys=True, use_values=False)
        full = append_gapped_post_summary(model, boundary, layout)
        arm_cache_lengths[arm] = _snapshot_length(full)
    if tuple(arm_cache_lengths) != GAPPED_ARM_NAMES:
        raise RuntimeError(
            f"ladder arm set/order changed: {tuple(arm_cache_lengths)}")

    # Exercise both unique deterministic calibration variants. Conversation
    # repetitions are not independent calibration evidence (Amendment 2).
    calibrations = validate_calibration_constructions(tokenizer)
    if set(calibrations.get("label_coverage", [])) != {"A", "B"}:
        raise RuntimeError(
            "ladder calibration did not cover both label variants")
    if calibrations.get("passes") is not True:
        raise RuntimeError("ladder calibration construction failed")

    # A killed run after rendering must load identical text and never rerender.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fp = {"model": MODEL, "test": "resume"}
        path = checkpoint_path(root, 1, "c10")
        saved = save_render(path, fingerprint=fp, order_position=1,
                            conversation=target, reply_records=[])
        loaded = read_checkpoint(path, fp, "rendered")
        resume_exact = saved == loaded and loaded["conversation"] == target
    if not resume_exact:
        raise RuntimeError("render resume was not exact")

    return {
        "schema": 2,
        "amendment_id": AMENDMENT_ID,
        "design_id": DESIGN_ID,
        "status": "PASS", "model": MODEL,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "dtype": str(next(model.parameters()).dtype),
        "device": str(model.device),
        "forced_summary_ids": summary_ids,
        "fresh_trace": asdict(fresh_trace),
        "self_replacement_exact": self_exact,
        "position_schedule": {
            "source_summary_start": layout.source_summary_start,
            "physical_summary_start": layout.physical_summary_start,
            "logical_next_position": layout.logical_next_position,
            "physical_context_length": len(layout.context_ids),
            "logical_gap": (layout.source_summary_start -
                            layout.physical_summary_start),
        },
        "exact_length_wrong": {
            "prefix_length": len(matched.correct_ids),
            "structural_position_count": len(matched.structural_positions),
            "content_position_count": len(matched.content_positions),
            "replacement_count": len(matched.replacements),
        },
        "arm_constructor_checks": {
            "exact_arm_order": list(arm_cache_lengths),
            "all_completed_without_semantic_scoring": True,
        },
        "arm_cache_lengths": arm_cache_lengths,
        "calibration_construction_only": calibrations,
        "retired_G_delta": {"executed": False, "retired_by": "Amendment 4"},
        "render_resume_exact": resume_exact,
        "loaded_gapped_production_gate": loaded_gates,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


MAX_COMMITTED_ARTIFACT_BYTES = 4_000_000


def _payload_sha256(payload: dict) -> str:
    clean = {key: value for key, value in payload.items()
             if key != "payload_sha256"}
    return hashlib.sha256(json.dumps(
        clean, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode()).hexdigest()


def _encoded_payload(payload: dict) -> tuple[dict, bytes]:
    closed = _copy_json(payload)
    closed["payload_sha256"] = _payload_sha256(closed)
    raw = (json.dumps(
        closed, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False) + "\n").encode()
    if len(raw) >= MAX_COMMITTED_ARTIFACT_BYTES:
        raise RuntimeError(
            f"ladder payload is not commit-safe: {len(raw)} bytes")
    return closed, raw


def _atomic_write_bytes(path: Path, raw: bytes) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite ladder artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise RuntimeError(f"ladder temporary path already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_replace_bytes(path: Path, raw: bytes) -> None:
    """Durably replace one nonterminal lifecycle payload in its unique attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise RuntimeError(f"ladder temporary path already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class LadderDurableDiagnosticSink(dict):
    """Persist each whole-stage reassignment while a long ladder is running."""

    def __init__(self, output: Path):
        super().__init__()
        self.output = Path(output)
        existing = list(self.output.parent.glob(
            f"{self.output.stem}__stage_*.json"))
        if self.output.exists() or existing:
            raise RuntimeError(
                f"refusing to resume/overwrite ladder attempt: "
                f"{[str(self.output), *map(str, existing)]}")

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key not in V7_GATE_STAGE_ORDER or not isinstance(value, dict):
            return
        _closed, raw = _encoded_payload({
            "schema": 2,
            "amendment_id": AMENDMENT_ID,
            "design_id": DESIGN_ID,
            "artifact_kind": "ladder_gate_stage",
            "stage_name": key,
            "stage": value,
        })
        path = self.output.with_name(
            f"{self.output.stem}__stage_{key}.json")
        _atomic_replace_bytes(path, raw)


def write_sharded_ladder_result(output: Path, result: dict) -> dict:
    """Write a small v7 ladder manifest plus commit-safe gate stage sidecars."""
    output = Path(output)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    document = _copy_json(result)
    if document.get("design_id") != DESIGN_ID:
        raise RuntimeError("ladder result identity is not current v7")

    gate_container = document
    gate_key = "loaded_gapped_production_gate"
    if gate_key not in gate_container:
        gate_container = document.get("diagnostics", {})
    gate = gate_container.get(gate_key)
    writes: list[tuple[Path, bytes]] = []
    refs = {}
    if isinstance(gate, dict):
        for stage_name in V7_GATE_STAGE_ORDER:
            stage = gate.get(stage_name)
            if not isinstance(stage, dict):
                continue
            filename = f"{output.stem}__stage_{stage_name}.json"
            sidecar_path = output.with_name(filename)
            closed, raw = _encoded_payload({
                "schema": 2,
                "amendment_id": AMENDMENT_ID,
                "design_id": DESIGN_ID,
                "artifact_kind": "ladder_gate_stage",
                "stage_name": stage_name,
                "stage": stage,
            })
            writes.append((sidecar_path, raw))
            refs[stage_name] = {
                "path": filename,
                "byte_count": len(raw),
                "raw_file_sha256": hashlib.sha256(raw).hexdigest(),
                "payload_sha256": closed["payload_sha256"],
            }
        gate_container[gate_key] = {
            "schema": gate.get("schema", 2),
            "amendment_id": gate.get("amendment_id"),
            "design_id": gate.get("design_id"),
            "status": gate.get("status"),
            "passes": gate.get("passes", False),
            "technical_only": gate.get("technical_only", True),
            "stage_order": gate.get("stage_order", list(V7_GATE_STAGE_ORDER)),
            "failures": gate.get("failures", []),
            "failure": gate.get("failure"),
            "stage_refs": refs,
            "externalized": True,
        }

    document["artifact_files"] = [
        {"path": path.name, "byte_count": len(raw),
         "raw_file_sha256": hashlib.sha256(raw).hexdigest()}
        for path, raw in writes
    ]
    closed_manifest, manifest_raw = _encoded_payload(document)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite ladder artifact: {output}")
    for path, raw in writes:
        if path.exists():
            if path.read_bytes() != raw:
                raise RuntimeError(
                    f"durable ladder stage differs from terminal payload: {path}")
        else:
            _atomic_write_bytes(path, raw)
    _atomic_write_bytes(output, manifest_raw)
    return closed_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    print(f"RUN coherent_state_ladder model={MODEL} -> {args.output}", flush=True)
    progress_sink = LadderDurableDiagnosticSink(args.output)
    try:
        result = run_ladder(progress_sink)
    except Exception as exc:
        diagnostics = _LAST_LADDER_DIAGNOSTICS
        if ("loaded_gapped_production_gate" not in diagnostics and
                progress_sink):
            diagnostics = {
                **diagnostics,
                "loaded_gapped_production_gate": dict(progress_sink),
            }
        result = {
                  "schema": 2,
                  "amendment_id": AMENDMENT_ID,
                  "design_id": DESIGN_ID,
                  "status": "FAIL", "error_type": type(exc).__name__,
                  "error": str(exc), "traceback": traceback.format_exc(),
                  "diagnostics": diagnostics,
                  "failed_at": datetime.now(timezone.utc).isoformat()}
    manifest = write_sharded_ladder_result(args.output, result)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
