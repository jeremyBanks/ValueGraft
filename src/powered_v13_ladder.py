"""Minimal same-schedule L0/L1/L3 ladder for powered-v13 Stage T.

The ladder never compares a q1 replay with a different call decomposition.
Every identity uses the exact fresh production plan and its frozen event widths.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from coherent_canary_runtime import (
    continue_fresh_plan,
    execute_fresh_plan,
    extract_rows,
    replace_rows,
    snapshot_hashes,
    tensor_sha256,
)
from powered_v13_schema import N_SCHEDULE, R2


SCHEMA = "coherent-state-powered-successor-v13-stage-t-ladder-v1"


class V13LadderError(RuntimeError):
    """One same-schedule production-path identity differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V13LadderError(message)


def _terminal_signature(result: Any) -> dict[str, Any]:
    return {
        "snapshot_hashes": snapshot_hashes(result.snapshot),
        "last_logits_sha256": tensor_sha256(result.last_logits),
        "physical_end": result.physical_end,
        "logical_end": result.logical_end,
    }


def _same_terminal(left: Any, right: Any, label: str) -> dict[str, Any]:
    left_signature = _terminal_signature(left)
    right_signature = _terminal_signature(right)
    _require(left_signature == right_signature,
             f"{label} terminal rows/logits differ")
    return left_signature


def _null_split_reconcat(snapshot, split_at: int):
    rebuilt = []
    for layer, (keys, values) in enumerate(snapshot):
        _require(keys.shape == values.shape and keys.ndim == 4,
                 f"L1 layer {layer} K/V geometry differs")
        length = int(keys.shape[-2])
        _require(0 <= split_at <= length,
                 f"L1 split lies outside layer {layer}")
        new_keys = torch.cat(
            (keys[..., :split_at, :], keys[..., split_at:, :]), dim=-2,
        ).clone().contiguous()
        new_values = torch.cat(
            (values[..., :split_at, :], values[..., split_at:, :]), dim=-2,
        ).clone().contiguous()
        _require(torch.equal(new_keys, keys) and torch.equal(new_values, values),
                 f"L1 layer {layer} null split/reconcat changed rows")
        rebuilt.append((new_keys, new_values))
    return rebuilt


def run_stage_t_ladder(model, fresh_plan) -> dict[str, Any]:
    """Run the closed Stage-T ladder on one exact fresh destination plan."""

    fresh_plan.validate()
    r2_start, r2_end = fresh_plan.physical_regions.interval(R2)
    _require(0 < r2_start < r2_end < len(fresh_plan.token_ids),
             "ladder R2 is not an interior continuation boundary")

    # L0: identical model, plan, calls, positions, and cache start.
    full_first = execute_fresh_plan(model, fresh_plan)
    full_repeat = execute_fresh_plan(model, fresh_plan)
    _require(full_first.calls == full_repeat.calls
             and full_first.executed_token_ids == full_repeat.executed_token_ids
             and full_first.logical_positions == full_repeat.logical_positions
             and full_first.physical_positions == full_repeat.physical_positions,
             "L0 repeated execution trace differs")
    l0_signature = _same_terminal(full_first, full_repeat, "L0 repeat")

    boundary = execute_fresh_plan(model, fresh_plan, stop_at=r2_end)
    continued = continue_fresh_plan(
        model, fresh_plan, boundary.snapshot, start_at=r2_end)

    # L3: stopping/rebuilding at R2 and using the same remaining calls is exact.
    _require(full_first.calls == boundary.calls + continued.calls,
             "L3 uninterrupted and boundary call traces differ")
    _require(full_first.executed_token_ids ==
             boundary.executed_token_ids + continued.executed_token_ids,
             "L3 uninterrupted and boundary token IDs differ")
    _require(full_first.logical_positions ==
             boundary.logical_positions + continued.logical_positions
             and full_first.physical_positions ==
             boundary.physical_positions + continued.physical_positions,
             "L3 uninterrupted and boundary positions differ")
    l3_signature = _same_terminal(
        full_first, continued, "L3 boundary continuation")

    # L1: a null split/reconcat changes no cache row and preserves the same
    # continuation event widths.  It does not compare monolithic with q1 calls.
    split_at = r2_start
    null_boundary = _null_split_reconcat(boundary.snapshot, split_at)
    _require(snapshot_hashes(null_boundary) == snapshot_hashes(boundary.snapshot),
             "L1 null boundary hashes differ")
    null_continued = continue_fresh_plan(
        model, fresh_plan, null_boundary, start_at=r2_end)
    _require(null_continued.calls == continued.calls,
             "L1 continuation calls differ")
    l1_signature = _same_terminal(
        continued, null_continued, "L1 null surgery")

    # L3 self-source identities exercise the exact replacement primitive for
    # all channel modes used by FF/FC/FW/CC/WW without changing the state.
    selected = extract_rows(
        boundary.snapshot, r2_start, r2_end,
        max_rows=256, to_cpu=False)
    self_replacement: dict[str, Mapping[str, Any]] = {}
    for label, use_keys, use_values in (
        ("K_ONLY", True, False),
        ("V_ONLY", False, True),
        ("K_AND_V", True, True),
    ):
        replaced, evidence = replace_rows(
            boundary.snapshot, selected, r2_start,
            use_keys=use_keys, use_values=use_values)
        _require(snapshot_hashes(replaced) == snapshot_hashes(boundary.snapshot),
                 f"L3 {label} self-replacement boundary differs")
        observed = continue_fresh_plan(
            model, fresh_plan, replaced, start_at=r2_end)
        _require(observed.calls == continued.calls,
                 f"L3 {label} self-replacement calls differ")
        signature = _same_terminal(
            continued, observed, f"L3 {label} self-replacement")
        self_replacement[label] = {
            "replacement": evidence,
            "terminal": signature,
        }

    return {
        "schema": SCHEMA,
        "status": "PASS",
        "schedule": N_SCHEDULE,
        "r2_physical_start": r2_start,
        "r2_physical_end": r2_end,
        "event_count": len(fresh_plan.events),
        "l0_repeat": l0_signature,
        "l1_null_split_reconcat": l1_signature,
        "l3_boundary_continuation": l3_signature,
        "l3_fresh_self_replacement": self_replacement,
        "different_call_decomposition_compared": False,
    }


__all__ = ["SCHEMA", "V13LadderError", "run_stage_t_ladder"]
