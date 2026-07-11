"""Differentiable, detached-public-path control for canary v12."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from typing import Sequence

import torch

from coherent_canary_controls import bf16_gradient_ulp_edit_row
from coherent_canary_runtime import (
    CanaryRuntimeError,
    Snapshot,
    _float32_bits,
    _one_prefix_margin,
    collect_fresh_region_margin_gradients,
    continue_fresh_plan,
    execute_fresh_plan,
    replace_rows,
    tensor_sha256,
)
from coherent_canary_schema import FreshDestinationPlan, R2


ULP_COUNTS = (1, 2, 4, 8, 16, 32, 64)
MIN_MARGIN_MOVEMENT = 1e-4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryRuntimeError(message)


def _edited_rows(fresh_rows: Snapshot, gradients: Snapshot, *,
                 ulp_count: int, direction: int):
    edited: Snapshot = []
    diagnostics = []
    for layer, ((fresh_k, fresh_v), (gradient_k, gradient_v)) in enumerate(
            zip(fresh_rows, gradients)):
        _require(fresh_k.dtype == fresh_v.dtype == torch.bfloat16,
                 f"path-control layer {layer} is not bf16")
        k, v = fresh_k.clone(), fresh_v.clone()
        layer_rows = []
        for channel, base, gradient, output in (
            ("K", fresh_k, gradient_k, k),
            ("V", fresh_v, gradient_v, v),
        ):
            _require(base.shape == gradient.shape,
                     f"path-control {channel} gradient geometry differs")
            for token_row in range(base.shape[-2]):
                row = base[..., token_row, :]
                grad = gradient[..., token_row, :]
                changed, row_diagnostics = bf16_gradient_ulp_edit_row(
                    row, grad, ulp_count=ulp_count, direction=direction)
                output[..., token_row, :] = changed
                layer_rows.append({
                    "layer": layer,
                    "channel": channel,
                    "token_row": token_row,
                    "gradient_sha256": tensor_sha256(grad),
                    **row_diagnostics,
                })
        edited.append((k, v))
        diagnostics.extend(layer_rows)
    _require(any(row["selected"] for row in diagnostics),
             "path control selected no nonzero-gradient row")
    return edited, diagnostics


def _score_detached_rows(model, plan: FreshDestinationPlan, *, region: str,
                         rows: Snapshot, suffix_ids: Sequence[int],
                         correct_id: int, counterfactual_id: int):
    physical_start, physical_end = plan.physical_regions.interval(region)
    boundary = execute_fresh_plan(model, plan, stop_at=physical_end)
    inserted, insertion = replace_rows(
        boundary.snapshot, rows, physical_start,
        use_keys=True, use_values=True)
    completed = continue_fresh_plan(
        model, plan, inserted, start_at=physical_end)
    margin, correct, counterfactual = _one_prefix_margin(
        model, completed.snapshot, suffix_ids=suffix_ids,
        logical_context_end=plan.logical_positions[-1] + 1,
        correct_id=correct_id, counterfactual_id=counterfactual_id,
        enable_grad=False)
    values = [float(x.detach().cpu()) for x in (margin, correct, counterfactual)]
    _require(all(math.isfinite(x) for x in values),
             "path-control detached score is nonfinite")
    return {
        "margin": values[0],
        "margin_float32_bits": _float32_bits(margin),
        "correct_logprob": values[1],
        "correct_logprob_float32_bits": _float32_bits(correct),
        "counterfactual_logprob": values[2],
        "counterfactual_logprob_float32_bits": _float32_bits(counterfactual),
        "insertion": insertion,
        "execution_trace": {
            "boundary_calls": boundary.calls,
            "boundary_token_ids": boundary.executed_token_ids,
            "boundary_logical_positions": boundary.logical_positions,
            "boundary_physical_positions": boundary.physical_positions,
            "continuation_calls": completed.calls,
            "continuation_token_ids": completed.executed_token_ids,
            "continuation_logical_positions": completed.logical_positions,
            "continuation_physical_positions": completed.physical_positions,
            "probe_suffix_ids": [int(x) for x in suffix_ids],
            "probe_logical_positions": list(range(
                plan.logical_positions[-1] + 1,
                plan.logical_positions[-1] + 1 + len(suffix_ids))),
            "probe_physical_positions": list(range(
                len(plan.token_ids), len(plan.token_ids) + len(suffix_ids))),
        },
    }


def run_bidirectional_path_control(
        model, plan: FreshDestinationPlan, *, region: str,
        suffix_ids: Sequence[int], correct_id: int, counterfactual_id: int,
) -> dict:
    _require(region == R2, "technical path control must use frozen R2")
    gradient = collect_fresh_region_margin_gradients(
        model, plan, region=region, suffix_ids=suffix_ids,
        correct_id=correct_id, counterfactual_id=counterfactual_id)
    attempts = []
    for ulp_count in ULP_COUNTS:
        plus_rows, plus_diagnostics = _edited_rows(
            gradient.fresh_rows, gradient.gradients,
            ulp_count=ulp_count, direction=1)
        minus_rows, minus_diagnostics = _edited_rows(
            gradient.fresh_rows, gradient.gradients,
            ulp_count=ulp_count, direction=-1)
        _require(len(plus_diagnostics) == len(minus_diagnostics),
                 "path-control direction coverage differs")
        for plus, minus in zip(plus_diagnostics, minus_diagnostics):
            _require((plus["layer"], plus["channel"], plus["token_row"],
                      plus.get("flat_index")) ==
                     (minus["layer"], minus["channel"], minus["token_row"],
                      minus.get("flat_index")),
                     "path-control directions selected different coordinates")
        plus_score = _score_detached_rows(
            model, plan, region=region, rows=plus_rows,
            suffix_ids=suffix_ids, correct_id=correct_id,
            counterfactual_id=counterfactual_id)
        minus_score = _score_detached_rows(
            model, plan, region=region, rows=minus_rows,
            suffix_ids=suffix_ids, correct_id=correct_id,
            counterfactual_id=counterfactual_id)
        plus_movement = plus_score["margin"] - gradient.baseline_margin
        minus_movement = gradient.baseline_margin - minus_score["margin"]
        passed = (plus_movement >= MIN_MARGIN_MOVEMENT and
                  minus_movement >= MIN_MARGIN_MOVEMENT)
        attempt = {
            "ulp_count": ulp_count,
            "baseline_margin": gradient.baseline_margin,
            "plus": plus_score,
            "minus": minus_score,
            "plus_margin_movement": plus_movement,
            "minus_margin_movement": minus_movement,
            "passes": passed,
            "plus_row_diagnostics": plus_diagnostics,
            "minus_row_diagnostics": minus_diagnostics,
            "plus_row_hashes": [{
                "layer": index, "k_sha256": tensor_sha256(keys),
                "v_sha256": tensor_sha256(values),
            } for index, (keys, values) in enumerate(plus_rows)],
            "minus_row_hashes": [{
                "layer": index, "k_sha256": tensor_sha256(keys),
                "v_sha256": tensor_sha256(values),
            } for index, (keys, values) in enumerate(minus_rows)],
        }
        attempts.append(attempt)
        if passed:
            return {
                "schema": "coherent_canary_v12_bidirectional_path_control_v1",
                "status": "PASS",
                "region": region,
                "correct_target_id": int(correct_id),
                "counterfactual_target_id": int(counterfactual_id),
                "probe_suffix_ids": [int(x) for x in suffix_ids],
                "plan_geometry_sha256": hashlib.sha256(json.dumps(
                    plan.geometry(), sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest(),
                "plan_token_ids_sha256": hashlib.sha256(b"".join(
                    int(value).to_bytes(8, "little", signed=True)
                    for value in plan.token_ids
                )).hexdigest(),
                "chosen_ulp_count": ulp_count,
                "minimum_margin_movement": MIN_MARGIN_MOVEMENT,
                "gradient_baseline": {
                    key: value for key, value in asdict(gradient).items()
                    if key not in ("fresh_rows", "gradients")
                },
                "fresh_row_hashes": [{
                    "layer": index, "k_sha256": tensor_sha256(keys),
                    "v_sha256": tensor_sha256(values),
                    "k_gradient_sha256": tensor_sha256(
                        gradient.gradients[index][0]),
                    "v_gradient_sha256": tensor_sha256(
                        gradient.gradients[index][1]),
                } for index, (keys, values) in enumerate(gradient.fresh_rows)],
                "attempts": attempts,
            }
    # A scientifically adverse result must remain inspectable rather than being
    # lost as an exception after the final attempt.
    return {
        "schema": "coherent_canary_v12_bidirectional_path_control_v1",
        "status": "FAIL",
        "region": region,
        "correct_target_id": int(correct_id),
        "counterfactual_target_id": int(counterfactual_id),
        "probe_suffix_ids": [int(x) for x in suffix_ids],
        "plan_geometry_sha256": hashlib.sha256(json.dumps(
            plan.geometry(), sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "plan_token_ids_sha256": hashlib.sha256(b"".join(
            int(value).to_bytes(8, "little", signed=True)
            for value in plan.token_ids
        )).hexdigest(),
        "chosen_ulp_count": None,
        "minimum_margin_movement": MIN_MARGIN_MOVEMENT,
        "gradient_baseline": {
            key: value for key, value in asdict(gradient).items()
            if key not in ("fresh_rows", "gradients")
        },
        "fresh_row_hashes": [{
            "layer": index, "k_sha256": tensor_sha256(keys),
            "v_sha256": tensor_sha256(values),
            "k_gradient_sha256": tensor_sha256(gradient.gradients[index][0]),
            "v_gradient_sha256": tensor_sha256(gradient.gradients[index][1]),
        } for index, (keys, values) in enumerate(gradient.fresh_rows)],
        "attempts": attempts,
    }
