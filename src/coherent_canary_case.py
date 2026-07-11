"""Model-facing case execution for coherent-state canary v12.

The pre-treatment function deliberately exposes only full-history oracles,
fresh-compaction behavior, and forced-carrier support.  Treatment row sources
and graft scores live in a separate function/script so ordinary workflow cannot
accidentally inspect them while deciding eligibility.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct
from typing import Any, Mapping, Sequence

from coherent_canary_runtime import (
    continue_fresh_plan,
    execute_fresh_plan,
    execute_replay_plan,
    extract_rows,
    replace_rows,
    snapshot_hashes,
    tensor_sha256,
)
from coherent_canary_schema import DESIGN_ID, R1, R2, R3
from coherent_canary_technical import (
    compact_messages,
    probe_record,
    source_messages,
)
from coherent_canary_tokens import (
    build_fresh_destination_plan,
    build_role_native_plan,
    build_turn_aligned_plan,
)


PHASE_A_SCHEMA = "coherent_state_decision_canary_v12_phase_a_raw_v1"
TREATMENT_SCHEMA = "coherent_state_decision_canary_v12_treatment_raw_v1"
REGIONS = (R1, R2, R3)
ARM_SOURCES = {
    "FF": ("F", "F"), "FC": ("F", "C"), "FW": ("F", "W"),
    "CF": ("C", "F"), "CC": ("C", "C"), "CW": ("C", "W"),
    "WF": ("W", "F"), "WC": ("W", "C"), "WW": ("W", "W"),
}
P_CELLS = ("CC", "WW", "FC", "FW")


class CanaryCaseError(RuntimeError):
    """A case or its model-facing execution differs from the frozen design."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryCaseError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _decode_float32(bits: str) -> float:
    _require(isinstance(bits, str) and len(bits) == 8 and
             all(character in "0123456789abcdef" for character in bits),
             "model float is not lowercase little-endian float32 bits")
    value = struct.unpack("<f", bytes.fromhex(bits))[0]
    _require(math.isfinite(value), "model float is nonfinite")
    return float(value)


def case_histories(case: Mapping[str, Any]) -> tuple[list[dict], list[dict], int]:
    _require(case.get("schema") == "coherent_state_decision_canary_v12_case_draft_v1" and
             case.get("design_id") == DESIGN_ID and
             isinstance(case.get("case_id"), str),
             "case schema/design/id differs")
    variants = case.get("variants")
    _require(isinstance(variants, Mapping) and
             set(variants) == {"correct", "wrong_focal"},
             "case variant set differs")
    correct = variants["correct"].get("messages")
    wrong = variants["wrong_focal"].get("messages")
    middle = case.get("middle_end_msg")
    _require(isinstance(correct, list) and isinstance(wrong, list) and
             isinstance(middle, int) and 2 < middle < len(correct) == len(wrong),
             "case history/middle geometry differs")
    _require([row.get("role") for row in correct] ==
             [row.get("role") for row in wrong],
             "case variant role sequences differ")
    _require(correct[middle:] == wrong[middle:],
             "case retained tails are not byte-identical")
    return list(correct), list(wrong), middle


def build_case_plans(tokenizer, case: Mapping[str, Any]) -> dict[str, Any]:
    correct, wrong, middle = case_histories(case)
    plans = {
        "C_N": build_role_native_plan(
            tokenizer, correct, middle_end_msg=middle),
        "W_N": build_role_native_plan(
            tokenizer, wrong, middle_end_msg=middle),
        "C_P": build_turn_aligned_plan(
            tokenizer, correct, middle_end_msg=middle),
        "W_P": build_turn_aligned_plan(
            tokenizer, wrong, middle_end_msg=middle),
        "F": build_fresh_destination_plan(
            tokenizer, correct, middle_end_msg=middle),
    }
    wrong_fresh = build_fresh_destination_plan(
        tokenizer, wrong, middle_end_msg=middle)
    _require(plans["F"] == wrong_fresh,
             "correct/wrong fresh destinations differ")
    for left, right in (("C_N", "W_N"), ("C_P", "W_P")):
        _require(plans[left].regions == plans[right].regions and
                 len(plans[left].token_ids) == len(plans[right].token_ids),
                 f"{left}/{right} source carrier geometry differs")
    return plans


def plan_record(plan_id: str, plan) -> dict[str, Any]:
    record = {
        "plan_id": plan_id,
        "kind": "fresh" if plan_id == "F" else "source",
        "token_ids": list(plan.token_ids),
        "geometry": plan.geometry(),
    }
    record["record_sha256"] = _sha(record)
    return record


def execution_record(result) -> dict[str, Any]:
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


def carrier_support_record(result, plan) -> dict[str, Any]:
    start, end = plan.regions.interval(R1)
    rows = [row for row in result.q1_token_logprobs
            if start <= int(row["logical_position"]) < end]
    _require([int(row["logical_position"]) for row in rows] ==
             list(range(start, end)),
             "forced carrier q1 support does not cover exact R1")
    _require([int(row["token_id"]) for row in rows] ==
             list(plan.token_ids[start:end]),
             "forced carrier support token IDs differ from the plan")
    logprobs = [_decode_float32(str(row["logprob_float32_bits"]))
                for row in rows]
    return {
        "logical_start": start,
        "logical_end": end,
        "token_ids": [int(row["token_id"]) for row in rows],
        "token_logprob_float32_bits": [
            str(row["logprob_float32_bits"]) for row in rows],
        "mean_nll": -statistics.fmean(logprobs),
        "all_finite": True,
    }


def run_phase_a_case(model, tokenizer, case: Mapping[str, Any], *,
                     eos_ids: Sequence[int]) -> dict[str, Any]:
    """Run only the frozen pre-treatment evidence for one engineered case."""
    correct, wrong, middle = case_histories(case)
    plans = build_case_plans(tokenizer, case)
    c_result = execute_replay_plan(model, plans["C_N"])
    w_result = execute_replay_plan(model, plans["W_N"])
    fresh_result = execute_fresh_plan(model, plans["F"])

    correct_visible = source_messages(correct, middle)
    wrong_visible = source_messages(wrong, middle)
    compact_visible = compact_messages(correct, middle)
    focal = case.get("focal")
    nonfocal = case.get("nonfocal_control")
    _require(isinstance(focal, Mapping) and isinstance(nonfocal, Mapping),
             "case focal/nonfocal records differ")
    score_arguments = {
        "focal": (str(focal["probe"]), str(focal["correct_target"]),
                  str(focal["counterfactual_target"])),
        "nonfocal": (str(nonfocal["probe"]), str(nonfocal["target"]),
                     str(nonfocal["countertarget"])),
    }
    scores: dict[str, dict[str, Any]] = {}
    for probe_name, (probe, target, countertarget) in score_arguments.items():
        scores[f"A_C_{probe_name}"] = probe_record(
            model, tokenizer, c_result.snapshot, correct_visible,
            plans["C_N"].token_ids, len(plans["C_N"].token_ids),
            probe, target, countertarget, eos_ids)
        scores[f"A_W_{probe_name}"] = probe_record(
            model, tokenizer, w_result.snapshot, wrong_visible,
            plans["W_N"].token_ids, len(plans["W_N"].token_ids),
            probe, target, countertarget, eos_ids)
        scores[f"FF_{probe_name}"] = probe_record(
            model, tokenizer, fresh_result.snapshot, compact_visible,
            plans["F"].token_ids, plans["F"].logical_positions[-1] + 1,
            probe, target, countertarget, eos_ids)

    return {
        "schema": PHASE_A_SCHEMA,
        "design_id": DESIGN_ID,
        "case_id": str(case["case_id"]),
        "plans": {name: plan_record(name, plan)
                  for name, plan in plans.items()},
        "executions": {
            "C_N": execution_record(c_result),
            "W_N": execution_record(w_result),
            "F": execution_record(fresh_result),
        },
        "forced_carrier_support": {
            "C_N": carrier_support_record(c_result, plans["C_N"]),
            "W_N": carrier_support_record(w_result, plans["W_N"]),
        },
        "scores": scores,
        "visible_messages": {
            "A_C": correct_visible,
            "A_W": wrong_visible,
            "FF": compact_visible,
        },
        "treatment_scores_present": False,
    }


def _selected_rows(executions: Mapping[str, Any], plans: Mapping[str, Any],
                   boundary, *, schedule: str, region: str,
                   source: str):
    fresh_start, fresh_end = plans["F"].physical_regions.interval(region)
    if source == "F":
        return extract_rows(boundary.snapshot, fresh_start, fresh_end)
    plan_id = f"{source}_{schedule}"
    logical_start, logical_end = plans[plan_id].regions.interval(region)
    return extract_rows(
        executions[plan_id].snapshot, logical_start, logical_end)


def _mixed_rows(key_rows, value_rows):
    _require(len(key_rows) == len(value_rows) and bool(key_rows),
             "mixed treatment row layer coverage differs")
    result = []
    for layer, ((keys, _), (_, values)) in enumerate(zip(key_rows, value_rows)):
        _require(keys.shape == values.shape,
                 f"mixed treatment K/V geometry differs at layer {layer}")
        result.append((keys, values))
    return result


def _arm_record(model, tokenizer, case: Mapping[str, Any], plans,
                executions, boundaries, *, schedule: str, region: str,
                cell: str, eos_ids: Sequence[int]) -> dict[str, Any]:
    _require(cell in ARM_SOURCES and schedule in ("N", "P") and
             region in REGIONS, "treatment selector differs")
    key_source, value_source = ARM_SOURCES[cell]
    boundary = boundaries[region]
    key_rows = _selected_rows(
        executions, plans, boundary, schedule=schedule, region=region,
        source=key_source)
    value_rows = _selected_rows(
        executions, plans, boundary, schedule=schedule, region=region,
        source=value_source)
    rows = _mixed_rows(key_rows, value_rows)
    start, end = plans["F"].physical_regions.interval(region)
    replaced, insertion = replace_rows(
        boundary.snapshot, rows, start, use_keys=True, use_values=True)
    completed = continue_fresh_plan(
        model, plans["F"], replaced, start_at=end)
    correct, _, middle = case_histories(case)
    visible = compact_messages(correct, middle)
    focal = case["focal"]
    nonfocal = case["nonfocal_control"]
    logical_end = plans["F"].logical_positions[-1] + 1
    return {
        "schedule": schedule, "region": region, "cell": cell,
        "key_source": key_source, "value_source": value_source,
        "insertion": insertion,
        "source_row_hashes": snapshot_hashes(rows),
        "boundary_before_hashes": snapshot_hashes(boundary.snapshot),
        "boundary_after_hashes": snapshot_hashes(replaced),
        "continuation": execution_record(completed),
        "scores": {
            "focal": probe_record(
                model, tokenizer, completed.snapshot, visible,
                plans["F"].token_ids, logical_end, str(focal["probe"]),
                str(focal["correct_target"]),
                str(focal["counterfactual_target"]), eos_ids),
            "nonfocal": probe_record(
                model, tokenizer, completed.snapshot, visible,
                plans["F"].token_ids, logical_end,
                str(nonfocal["probe"]), str(nonfocal["target"]),
                str(nonfocal["countertarget"]), eos_ids),
        },
    }


def run_treatment_case(model, tokenizer, case: Mapping[str, Any], *,
                       eos_ids: Sequence[int]) -> dict[str, Any]:
    """Execute the frozen graft arms only after a Phase-A release."""
    correct, _, middle = case_histories(case)
    plans = build_case_plans(tokenizer, case)
    executions = {
        plan_id: execute_replay_plan(model, plans[plan_id])
        for plan_id in ("C_N", "W_N", "C_P", "W_P")
    }
    direct_fresh = execute_fresh_plan(model, plans["F"])
    boundaries = {
        region: execute_fresh_plan(
            model, plans["F"],
            stop_at=plans["F"].physical_regions.interval(region)[1])
        for region in REGIONS
    }
    visible = compact_messages(correct, middle)
    focal = case["focal"]
    nonfocal = case["nonfocal_control"]
    logical_end = plans["F"].logical_positions[-1] + 1
    fresh_scores = {
        "focal": probe_record(
            model, tokenizer, direct_fresh.snapshot, visible,
            plans["F"].token_ids, logical_end, str(focal["probe"]),
            str(focal["correct_target"]),
            str(focal["counterfactual_target"]), eos_ids),
        "nonfocal": probe_record(
            model, tokenizer, direct_fresh.snapshot, visible,
            plans["F"].token_ids, logical_end, str(nonfocal["probe"]),
            str(nonfocal["target"]), str(nonfocal["countertarget"]),
            eos_ids),
    }
    arms = []
    for region in REGIONS:
        for cell in ARM_SOURCES:
            if cell == "FF":
                arms.append({
                    "schedule": "N", "region": region, "cell": "FF",
                    "key_source": "F", "value_source": "F",
                    "shared_fresh_baseline": True,
                    "scores": fresh_scores,
                })
            else:
                arms.append(_arm_record(
                    model, tokenizer, case, plans, executions, boundaries,
                    schedule="N", region=region, cell=cell, eos_ids=eos_ids))
    for cell in P_CELLS:
        arms.append(_arm_record(
            model, tokenizer, case, plans, executions, boundaries,
            schedule="P", region=R2, cell=cell, eos_ids=eos_ids))
    return {
        "schema": TREATMENT_SCHEMA,
        "design_id": DESIGN_ID,
        "case_id": str(case["case_id"]),
        "plans": {name: plan_record(name, plan)
                  for name, plan in plans.items()},
        "source_executions": {
            name: execution_record(result) for name, result in executions.items()},
        "fresh_execution": execution_record(direct_fresh),
        "fresh_scores": fresh_scores,
        "arms": arms,
        "arm_count": len(arms),
        "phase_a_scores_present": False,
    }
