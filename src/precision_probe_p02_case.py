"""Targeted model-facing treatment for frozen precision-probe-p02.

This module intentionally reuses the v12/P01 source execution, graft, scoring,
and placebo primitives unchanged.  It changes only orchestration: one R2
foundation, one zero-increment FF anchor, six grafts in the preregistered order,
and (for repeat 1 only) one final R2 placebo attempt.  Persistence callbacks are
synchronous gates; a callback must return before the next model operation can
begin.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from coherent_canary_case import (
    _arm_record,
    _placebo_arm_record,
    build_case_plans,
    case_histories,
    execution_record,
    plan_record,
)
from coherent_canary_runtime import execute_fresh_plan, execute_replay_plan
from coherent_canary_schema import DESIGN_ID, R2
from coherent_canary_technical import compact_messages, probe_record


PROTOCOL_ID = "precision-probe-p02"
FOUNDATION_SCHEMA = "precision_probe_p02_treatment_foundation_v1"
TREATMENT_SCHEMA = "precision_probe_p02_targeted_treatment_raw_v1"

TARGETED_GRAFTS = (
    ("N", R2, "FC"),
    ("N", R2, "FW"),
    ("N", R2, "CC"),
    ("N", R2, "WW"),
    ("P", R2, "FC"),
    ("P", R2, "FW"),
)

FoundationCallback = Callable[[Mapping[str, Any]], None]
ArmCallback = Callable[[int, Mapping[str, Any]], None]


class PrecisionProbeP02CaseError(RuntimeError):
    """The targeted P02 case execution differed from its frozen arm set."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrecisionProbeP02CaseError(message)


def _fresh_scores(model, tokenizer, case: Mapping[str, Any], plans,
                  direct_fresh, *, eos_ids: Sequence[int]) -> dict[str, Any]:
    correct, _, middle = case_histories(case)
    visible = compact_messages(correct, middle)
    focal = case.get("focal")
    nonfocal = case.get("nonfocal_control")
    _require(isinstance(focal, Mapping) and isinstance(nonfocal, Mapping),
             "P02 focal/nonfocal records differ")
    logical_end = plans["F"].logical_positions[-1] + 1
    return {
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


def _ff_anchor(fresh_scores: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "arm_kind": "fresh_baseline",
        "execution_kind": "zero_increment_reuse",
        "schedule": "N",
        "region": R2,
        "cell": "FF",
        "key_source": "F",
        "value_source": "F",
        "shared_fresh_baseline": True,
        "scores": fresh_scores,
    }


def assert_targeted_treatment(payload: Mapping[str, Any], *, case_id: str,
                              repeat_index: int) -> None:
    _require(repeat_index in (1, 2),
             "P02 targeted treatment repeat index differs")
    _require(
        payload.get("schema") == TREATMENT_SCHEMA
        and payload.get("protocol_id") == PROTOCOL_ID
        and payload.get("design_id") == DESIGN_ID
        and payload.get("case_id") == case_id
        and payload.get("repeat_index") == repeat_index,
        "P02 targeted treatment identity differs",
    )
    foundation = payload.get("foundation")
    _require(
        isinstance(foundation, Mapping)
        and foundation.get("schema") == FOUNDATION_SCHEMA
        and foundation.get("case_id") == case_id
        and foundation.get("repeat_index") == repeat_index,
        "P02 treatment foundation differs",
    )
    arms = payload.get("arms")
    _require(isinstance(arms, list), "P02 targeted arm list is absent")
    expected_count = 8 if repeat_index == 1 else 7
    _require(len(arms) == expected_count,
             "P02 targeted arm count differs")
    ff = arms[0]
    _require(
        isinstance(ff, Mapping)
        and ff.get("arm_kind") == "fresh_baseline"
        and ff.get("execution_kind") == "zero_increment_reuse"
        and (ff.get("schedule"), ff.get("region"), ff.get("cell")) ==
        ("N", R2, "FF")
        and ff.get("shared_fresh_baseline") is True
        # Object identity proves zero-increment reuse without comparing or
        # traversing any raw score value before lossless persistence.
        and ff.get("scores") is payload.get("fresh_scores"),
        "P02 FF anchor differs",
    )
    selectors = [
        (row.get("schedule"), row.get("region"), row.get("cell"))
        for row in arms[1:7] if isinstance(row, Mapping)
    ]
    _require(selectors == list(TARGETED_GRAFTS)
             and all(arms[index].get("arm_kind") == "primary"
                     for index in range(1, 7)),
             "P02 six-graft order differs")
    if repeat_index == 1:
        placebo = arms[7]
        _require(
            isinstance(placebo, Mapping)
            and placebo.get("arm_kind") == "placebo_control"
            and (placebo.get("schedule"), placebo.get("region"),
                 placebo.get("cell")) == ("N", R2, "V_PLACEBO")
            and placebo.get("control_status") in {
                "AVAILABLE", "PLACEBO_UNAVAILABLE"},
            "P02 repeat-1 placebo differs",
        )
    _require(
        payload.get("executed_graft_count") == 6
        and payload.get("zero_increment_anchor_count") == 1
        and payload.get("placebo_attempt_count") ==
        (1 if repeat_index == 1 else 0)
        and payload.get("phase_a_scores_present") is False,
        "P02 declared treatment counts differ",
    )


def run_targeted_treatment_case(
    model,
    tokenizer,
    case: Mapping[str, Any],
    *,
    eos_ids: Sequence[int],
    repeat_index: int,
    on_foundation: FoundationCallback | None = None,
    on_arm: ArmCallback | None = None,
) -> dict[str, Any]:
    """Run the exact targeted P02 treatment and synchronously checkpoint it."""
    _require(repeat_index in (1, 2), "P02 repeat index must be 1 or 2")
    correct, _, middle = case_histories(case)
    plans = build_case_plans(tokenizer, case)
    executions = {
        plan_id: execute_replay_plan(model, plans[plan_id])
        for plan_id in ("C_N", "W_N", "C_P", "W_P")
    }
    direct_fresh = execute_fresh_plan(model, plans["F"])
    r2_end = plans["F"].physical_regions.interval(R2)[1]
    r2_boundary = execute_fresh_plan(model, plans["F"], stop_at=r2_end)
    boundaries = {R2: r2_boundary}
    fresh_scores = _fresh_scores(
        model, tokenizer, case, plans, direct_fresh, eos_ids=eos_ids)
    ff = _ff_anchor(fresh_scores)
    foundation: dict[str, Any] = {
        "schema": FOUNDATION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "design_id": DESIGN_ID,
        "case_id": str(case["case_id"]),
        "repeat_index": repeat_index,
        "plans": {name: plan_record(name, plan)
                  for name, plan in plans.items()},
        "source_executions": {
            name: execution_record(result)
            for name, result in executions.items()
        },
        "fresh_execution": execution_record(direct_fresh),
        "r2_boundary_execution": execution_record(r2_boundary),
        "fresh_scores": fresh_scores,
        "ff_anchor": ff,
        "visible_messages": compact_messages(correct, middle),
        "source_plan_order": ["C_N", "W_N", "C_P", "W_P"],
        "boundary_regions": [R2],
    }
    if on_foundation is not None:
        on_foundation(foundation)

    arms: list[dict[str, Any]] = [ff]
    for arm_index, (schedule, region, cell) in enumerate(
            TARGETED_GRAFTS, start=1):
        arm = _arm_record(
            model, tokenizer, case, plans, executions, boundaries,
            schedule=schedule, region=region, cell=cell, eos_ids=eos_ids)
        arms.append(arm)
        if on_arm is not None:
            on_arm(arm_index, arm)

    if repeat_index == 1:
        placebo = _placebo_arm_record(
            model, tokenizer, case, plans, executions, boundaries,
            region=R2, eos_ids=eos_ids)
        arms.append(placebo)
        if on_arm is not None:
            on_arm(7, placebo)

    payload = {
        "schema": TREATMENT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "design_id": DESIGN_ID,
        "case_id": str(case["case_id"]),
        "repeat_index": repeat_index,
        "foundation": foundation,
        "fresh_scores": fresh_scores,
        "arms": arms,
        "arm_count": len(arms),
        "executed_graft_count": 6,
        "zero_increment_anchor_count": 1,
        "placebo_attempt_count": 1 if repeat_index == 1 else 0,
        "available_placebo_control_count": sum(
            row.get("arm_kind") == "placebo_control"
            and row.get("control_status") == "AVAILABLE"
            for row in arms
        ),
        "phase_a_scores_present": False,
    }
    assert_targeted_treatment(
        payload, case_id=str(case["case_id"]), repeat_index=repeat_index)
    return payload
