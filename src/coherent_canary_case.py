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
    execute_fresh_plan,
    execute_replay_plan,
    snapshot_hashes,
    tensor_sha256,
)
from coherent_canary_schema import DESIGN_ID, R1
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
