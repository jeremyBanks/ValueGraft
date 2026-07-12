"""Deterministic whole-V-row placebo for powered successor v13.

This module is deliberately state-only: it accepts already captured bf16 value
rows and has no access to probes, targets, logits, continuations, or model
execution.  Candidate construction and acceptance implement Section 7 of
``COHERENT-STATE-POWERED-SUCCESSOR-V13-PREREGISTRATION.md``.

The two hash-domain class strings are exported as ``CONTENT_CLASS`` and
``STRUCTURAL_CLASS``.  They are literal canonical-JSON inputs, not display
labels, so callers must not substitute aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import torch


DESIGN_ID = "coherent-state-powered-successor-v13"
CONTENT_CLASS = "content"
STRUCTURAL_CLASS = "structural"
CLASS_NAMES = (CONTENT_CLASS, STRUCTURAL_CLASS)
RENDER_IDS = ("r1", "r2")
LAYER_COUNT = 48
MOVED_COUNT_SCHEDULE = (2, 4, 8, 16, 32, 64)
MIN_ACTIVE_LAYERS = 24
AGGREGATE_RATIO_INTERVAL = (0.75, 1.33)
MEDIAN_RATIO_INTERVAL = (0.50, 2.00)
MAX_ABS_AGGREGATE_COSINE = 0.20

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class V13PlaceboError(ValueError):
    """Input geometry or identity differs from the frozen v13 contract."""


@dataclass(frozen=True)
class RowAssignment:
    """Assign one correct-history donor token row to one fresh destination."""

    destination: int
    donor: int


@dataclass(frozen=True)
class ValueRowPlaceboResult:
    """A selected placebo, or a fully diagnosed unavailable result."""

    status: str
    values: Mapping[str, torch.Tensor] | None
    row_map: Mapping[str, tuple[RowAssignment, ...]] | None
    diagnostics: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V13PlaceboError(message)


def canonical_json_bytes(value: Any) -> bytes:
    """Encode the exact canonical UTF-8 JSON representation used by v13."""

    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise V13PlaceboError(
            f"value is not canonical-JSON encodable: {exc}") from exc
    return rendered.encode("utf-8")


def canonical_row_permutation(
    width: int,
    *,
    design_id: str,
    stable_candidate_id: str,
    render_id: str,
    class_name: str,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Return Section 7's SHA-sorted row order and its ordered digests."""

    _validate_identity(design_id, stable_candidate_id, render_id)
    _require(class_name in CLASS_NAMES,
             f"class_name must be one of {CLASS_NAMES!r}")
    _require(not isinstance(width, bool) and isinstance(width, int) and width >= 1,
             "width must be a positive integer")
    keyed: list[tuple[bytes, int]] = []
    for row_index in range(width):
        digest = sha256(canonical_json_bytes([
            design_id,
            stable_candidate_id,
            render_id,
            class_name,
            row_index,
        ])).digest()
        keyed.append((digest, row_index))
    keyed.sort(key=lambda item: (item[0], item[1]))
    return (
        tuple(row_index for _, row_index in keyed),
        tuple(digest.hex() for digest, _ in keyed),
    )


def retained_moved_counts(width: int) -> tuple[int, ...]:
    """Filter ``[2,4,8,16,32,64,m]`` with first-occurrence retention."""

    _require(not isinstance(width, bool) and isinstance(width, int) and width >= 1,
             "width must be a positive integer")
    retained: list[int] = []
    for count in (*MOVED_COUNT_SCHEDULE, width):
        if 2 <= count <= width and count not in retained:
            retained.append(count)
    return tuple(retained)


def candidate_row_map(
    permutations: Mapping[str, Sequence[int]],
    *,
    content_count: int,
    structural_count: int,
    direction: int,
) -> dict[str, tuple[RowAssignment, ...]]:
    """Construct a candidate map in protocol donor/permutation order."""

    _require(set(permutations) == set(CLASS_NAMES),
             "permutations must contain exactly content and structural")
    _require(direction in (+1, -1), "direction must be +1 or -1")
    counts = {
        CONTENT_CLASS: content_count,
        STRUCTURAL_CLASS: structural_count,
    }
    result: dict[str, tuple[RowAssignment, ...]] = {}
    for class_name in CLASS_NAMES:
        permutation = tuple(permutations[class_name])
        _require(
            len(permutation) == len(set(permutation)) and
            set(permutation) == set(range(len(permutation))),
            f"{class_name} is not a complete row permutation",
        )
        count = counts[class_name]
        _require(not isinstance(count, bool) and isinstance(count, int) and
                 2 <= count <= len(permutation),
                 f"invalid {class_name} moved count")
        selected = permutation[:count]
        assignments = tuple(
            RowAssignment(
                destination=selected[(offset + direction) % count],
                donor=donor,
            )
            for offset, donor in enumerate(selected)
        )
        result[class_name] = assignments
    return result


def apply_value_row_map(
    fresh: Mapping[str, torch.Tensor],
    correct: Mapping[str, torch.Tensor],
    row_map: Mapping[str, Sequence[RowAssignment]],
) -> dict[str, torch.Tensor]:
    """Apply a selected map by exact bf16 row copies onto cloned fresh values.

    Donors are always read from ``correct`` rather than from the accumulating
    result, so a cycle cannot accidentally become an in-place rotation.
    """

    _validate_apply_inputs(fresh, correct, row_map)
    result = {
        class_name: fresh[class_name].detach().clone()
        for class_name in CLASS_NAMES
    }
    with torch.no_grad():
        for class_name in CLASS_NAMES:
            for assignment in row_map[class_name]:
                result[class_name][:, assignment.destination, :, :].copy_(
                    correct[class_name][:, assignment.donor, :, :])
    return result


def build_value_row_placebo(
    fresh: Mapping[str, torch.Tensor],
    correct: Mapping[str, torch.Tensor],
    wrong: Mapping[str, torch.Tensor],
    *,
    design_id: str,
    stable_candidate_id: str,
    render_id: str,
) -> ValueRowPlaceboResult:
    """Select and apply the first v13 placebo candidate that passes.

    Invalid identities/dtypes/geometries raise ``V13PlaceboError`` because they
    are contract violations, not scientific unavailability.  Numerically valid
    geometry that cannot meet the frozen applied-control thresholds returns
    ``PLACEBO_UNAVAILABLE`` with diagnostics for every rejected candidate.
    """

    _validate_identity(design_id, stable_candidate_id, render_id)
    geometry = _validate_source_inputs(fresh, correct, wrong)
    cpu = _cpu_source_copies(fresh, correct, wrong)
    widths = {
        class_name: int(cpu["fresh"][class_name].shape[1])
        for class_name in CLASS_NAMES
    }
    permutations: dict[str, tuple[int, ...]] = {}
    permutation_digests: dict[str, tuple[str, ...]] = {}
    counts: dict[str, tuple[int, ...]] = {}
    for class_name in CLASS_NAMES:
        order, digests = canonical_row_permutation(
            widths[class_name],
            design_id=design_id,
            stable_candidate_id=stable_candidate_id,
            render_id=render_id,
            class_name=class_name,
        )
        permutations[class_name] = order
        permutation_digests[class_name] = digests
        counts[class_name] = retained_moved_counts(widths[class_name])

    top_diagnostics: dict[str, Any] = {
        "schema": "powered_v13_value_row_placebo_v1",
        "design_id": design_id,
        "stable_candidate_id": stable_candidate_id,
        "render_id": render_id,
        "class_order": list(CLASS_NAMES),
        "geometry": geometry,
        "source_sha256": {
            source_name: {
                class_name: _tensor_sha256(cpu[source_name][class_name])
                for class_name in CLASS_NAMES
            }
            for source_name in ("fresh", "correct", "wrong")
        },
        "permutations": {
            class_name: {
                "row_order": list(permutations[class_name]),
                "ordered_digest_hex": list(permutation_digests[class_name]),
                "retained_counts": list(counts[class_name]),
            }
            for class_name in CLASS_NAMES
        },
        "search_order": (
            "content_count_outer_structural_count_inner_direction_+1_then_-1"),
        "thresholds": {
            "minimum_active_layers": MIN_ACTIVE_LAYERS,
            "aggregate_ratio_inclusive": list(AGGREGATE_RATIO_INTERVAL),
            "median_ratio_inclusive": list(MEDIAN_RATIO_INTERVAL),
            "maximum_abs_aggregate_cosine_inclusive":
                MAX_ABS_AGGREGATE_COSINE,
        },
        "candidate_capacity": (
            len(counts[CONTENT_CLASS]) *
            len(counts[STRUCTURAL_CLASS]) * 2
        ),
        "candidate_diagnostics": [],
    }

    prepared, preparation_diagnostics = _prepare_numeric_state(cpu)
    top_diagnostics["numeric_preparation"] = preparation_diagnostics

    if not counts[CONTENT_CLASS] or not counts[STRUCTURAL_CLASS]:
        top_diagnostics.update({
            "status": "PLACEBO_UNAVAILABLE",
            "unavailable_reason": "NO_RETAINED_MOVED_COUNT_FOR_BOTH_CLASSES",
            "evaluated_candidate_count": 0,
            "accepted_candidate_ordinal": None,
            "accepted_row_map": None,
            "result_sha256": None,
        })
        return ValueRowPlaceboResult(
            status="PLACEBO_UNAVAILABLE",
            values=None,
            row_map=None,
            diagnostics=top_diagnostics,
        )

    ordinal = 0
    for content_count in counts[CONTENT_CLASS]:
        for structural_count in counts[STRUCTURAL_CLASS]:
            for direction in (+1, -1):
                ordinal += 1
                row_map = candidate_row_map(
                    permutations,
                    content_count=content_count,
                    structural_count=structural_count,
                    direction=direction,
                )
                candidate = _evaluate_candidate(
                    cpu,
                    prepared,
                    row_map,
                    preparation_status=str(preparation_diagnostics["status"]),
                    ordinal=ordinal,
                    content_count=content_count,
                    structural_count=structural_count,
                    direction=direction,
                )
                top_diagnostics["candidate_diagnostics"].append(candidate)
                if candidate["accepted"]:
                    values = apply_value_row_map(fresh, correct, row_map)
                    serial_map = _serial_row_map(row_map)
                    top_diagnostics.update({
                        "status": "AVAILABLE",
                        "unavailable_reason": None,
                        "evaluated_candidate_count": ordinal,
                        "accepted_candidate_ordinal": ordinal,
                        "accepted_row_map": serial_map,
                        "result_sha256": {
                            class_name: _tensor_sha256(values[class_name])
                            for class_name in CLASS_NAMES
                        },
                    })
                    return ValueRowPlaceboResult(
                        status="AVAILABLE",
                        values=values,
                        row_map=row_map,
                        diagnostics=top_diagnostics,
                    )

    top_diagnostics.update({
        "status": "PLACEBO_UNAVAILABLE",
        "unavailable_reason": "NO_CANDIDATE_PASSED_FROZEN_THRESHOLDS",
        "evaluated_candidate_count": ordinal,
        "accepted_candidate_ordinal": None,
        "accepted_row_map": None,
        "result_sha256": None,
    })
    return ValueRowPlaceboResult(
        status="PLACEBO_UNAVAILABLE",
        values=None,
        row_map=None,
        diagnostics=top_diagnostics,
    )


def _validate_identity(
    design_id: str, stable_candidate_id: str, render_id: str,
) -> None:
    _require(design_id == DESIGN_ID, "design_id differs from frozen v13")
    _require(isinstance(stable_candidate_id, str) and
             bool(_SHA256_HEX.fullmatch(stable_candidate_id)),
             "stable_candidate_id must be 64 lowercase hex digits")
    _require(render_id in RENDER_IDS, "render_id must be r1 or r2")


def _validate_mapping_keys(value: Mapping[str, Any], label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    _require(set(value) == set(CLASS_NAMES),
             f"{label} must contain exactly content and structural")


def _validate_source_inputs(
    fresh: Mapping[str, torch.Tensor],
    correct: Mapping[str, torch.Tensor],
    wrong: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    for label, source in (("fresh", fresh), ("correct", correct),
                          ("wrong", wrong)):
        _validate_mapping_keys(source, label)
        for class_name in CLASS_NAMES:
            _require(isinstance(source[class_name], torch.Tensor),
                     f"{label}.{class_name} must be a tensor")

    reference_device = fresh[CONTENT_CLASS].device
    reference_heads: int | None = None
    reference_dimension: int | None = None
    geometry: dict[str, Any] = {
        "layer_count": LAYER_COUNT,
        "dtype": "torch.bfloat16",
        "device": str(reference_device),
        "classes": {},
    }
    for class_name in CLASS_NAMES:
        shape = tuple(fresh[class_name].shape)
        _require(len(shape) == 4,
                 f"fresh.{class_name} must have shape [48,m,kv_heads,head_dim]")
        _require(shape[0] == LAYER_COUNT and shape[1] >= 1 and
                 shape[2] >= 1 and shape[3] >= 1,
                 f"fresh.{class_name} has invalid v13 geometry")
        for label, source in (("fresh", fresh), ("correct", correct),
                              ("wrong", wrong)):
            tensor = source[class_name]
            _require(tuple(tensor.shape) == shape,
                     f"{label}.{class_name} geometry differs")
            _require(tensor.dtype == torch.bfloat16,
                     f"{label}.{class_name} must be exact torch.bfloat16")
            _require(tensor.device == reference_device,
                     "all source tensors must share one device")
        if reference_heads is None:
            reference_heads = shape[2]
            reference_dimension = shape[3]
        _require(shape[2] == reference_heads and shape[3] == reference_dimension,
                 "content/structural head geometry differs")
        geometry["classes"][class_name] = {
            "shape": list(shape),
            "width": shape[1],
            "kv_heads": shape[2],
            "head_dim": shape[3],
        }
    return geometry


def _validate_apply_inputs(
    fresh: Mapping[str, torch.Tensor],
    correct: Mapping[str, torch.Tensor],
    row_map: Mapping[str, Sequence[RowAssignment]],
) -> None:
    _validate_mapping_keys(fresh, "fresh")
    _validate_mapping_keys(correct, "correct")
    _validate_mapping_keys(row_map, "row_map")
    for class_name in CLASS_NAMES:
        left = fresh[class_name]
        right = correct[class_name]
        _require(isinstance(left, torch.Tensor) and
                 isinstance(right, torch.Tensor),
                 f"{class_name} sources must be tensors")
        _require(left.dtype == right.dtype == torch.bfloat16,
                 f"{class_name} sources must be torch.bfloat16")
        _require(left.shape == right.shape and left.ndim == 4 and
                 left.shape[0] == LAYER_COUNT,
                 f"{class_name} source geometry differs")
        _require(left.device == right.device,
                 f"{class_name} source devices differ")
        assignments = tuple(row_map[class_name])
        _require(bool(assignments), f"{class_name} row map is empty")
        destinations: set[int] = set()
        for assignment in assignments:
            _require(isinstance(assignment, RowAssignment),
                     f"{class_name} row map entry has wrong type")
            _require(0 <= assignment.destination < left.shape[1] and
                     0 <= assignment.donor < left.shape[1],
                     f"{class_name} row map entry is out of range")
            _require(assignment.destination not in destinations,
                     f"{class_name} row map repeats a destination")
            destinations.add(assignment.destination)


def _cpu_source_copies(
    fresh: Mapping[str, torch.Tensor],
    correct: Mapping[str, torch.Tensor],
    wrong: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        source_name: {
            class_name: tensor.detach().to(device="cpu").contiguous()
            for class_name, tensor in source.items()
        }
        for source_name, source in (
            ("fresh", fresh), ("correct", correct), ("wrong", wrong))
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = (value.detach().to(device="cpu").contiguous()
           .view(torch.uint8).numpy().tobytes())
    return sha256(raw).hexdigest()


def _nonfinite_locations(
    tensor: torch.Tensor, *, source_name: str, class_name: str,
) -> list[dict[str, Any]]:
    locations = torch.nonzero(~torch.isfinite(tensor), as_tuple=False).tolist()
    return [
        {
            "source": source_name,
            "class": class_name,
            "layer": int(index[0]),
            "row": int(index[1]),
            "kv_head": int(index[2]),
            "head_dimension": int(index[3]),
        }
        for index in locations
    ]


def _flat_layer_values(
    class_tensors: Mapping[str, torch.Tensor], layer_index: int,
) -> Iterable[float]:
    for class_name in CLASS_NAMES:
        yield from class_tensors[class_name][layer_index].reshape(-1).tolist()


def _ordered_fsum_squares(values: Iterable[float]) -> float:
    return math.fsum(value * value for value in values)


def _prepare_numeric_state(
    cpu: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    nonfinite_inputs: list[dict[str, Any]] = []
    for source_name in ("fresh", "correct", "wrong"):
        for class_name in CLASS_NAMES:
            nonfinite_inputs.extend(_nonfinite_locations(
                cpu[source_name][class_name],
                source_name=source_name,
                class_name=class_name,
            ))
    diagnostics: dict[str, Any] = {
        "scalar_cast_rule": "bf16_to_ieee_float64_before_subtraction",
        "sum_rule": "python_math.fsum_frozen_coordinate_order",
        "nonfinite_input_locations": nonfinite_inputs,
        "nonfinite_difference_locations": [],
        "base_layer_diagnostics": [],
    }
    if nonfinite_inputs:
        diagnostics["status"] = "NONFINITE_INPUT"
        return None, diagnostics

    float64: dict[str, dict[str, torch.Tensor]] = {
        source_name: {
            class_name: cpu[source_name][class_name].to(dtype=torch.float64)
            for class_name in CLASS_NAMES
        }
        for source_name in ("fresh", "correct", "wrong")
    }
    b = {
        class_name: (
            float64["correct"][class_name] - float64["fresh"][class_name])
        for class_name in CLASS_NAMES
    }
    d = {
        class_name: (
            float64["correct"][class_name] - float64["wrong"][class_name])
        for class_name in CLASS_NAMES
    }
    nonfinite_differences: list[dict[str, Any]] = []
    for difference_name, values in (("b_correct_minus_fresh", b),
                                    ("d_correct_minus_wrong", d)):
        for class_name in CLASS_NAMES:
            for item in _nonfinite_locations(
                    values[class_name],
                    source_name=difference_name,
                    class_name=class_name):
                nonfinite_differences.append(item)
    diagnostics["nonfinite_difference_locations"] = nonfinite_differences
    if nonfinite_differences:
        diagnostics["status"] = "NONFINITE_DIFFERENCE"
        return None, diagnostics

    b_squares: list[float] = []
    d_squares: list[float] = []
    active_layers: list[int] = []
    accumulator_nonfinite = False
    for layer_index in range(LAYER_COUNT):
        b_square = _ordered_fsum_squares(_flat_layer_values(b, layer_index))
        d_square = _ordered_fsum_squares(_flat_layer_values(d, layer_index))
        b_norm = math.sqrt(b_square) if b_square >= 0.0 else math.nan
        d_norm = math.sqrt(d_square) if d_square >= 0.0 else math.nan
        finite = all(math.isfinite(value) for value in (
            b_square, d_square, b_norm, d_norm))
        active = finite and b_norm > 0.0
        if active:
            active_layers.append(layer_index)
        if not finite:
            accumulator_nonfinite = True
        b_squares.append(b_square)
        d_squares.append(d_square)
        diagnostics["base_layer_diagnostics"].append({
            "layer_index": layer_index,
            "b_sum_squares": b_square,
            "b_norm": b_norm,
            "d_sum_squares": d_square,
            "d_norm": d_norm,
            "active": active,
            "inactive_reason": None if active else (
                "ZERO_REAL_DELTA" if finite and b_norm == 0.0
                else "NONFINITE_BASE_ACCUMULATOR"),
        })

    diagnostics["status"] = (
        "NONFINITE_ACCUMULATOR" if accumulator_nonfinite else "READY")
    diagnostics["active_layer_indices"] = active_layers
    diagnostics["active_layer_count"] = len(active_layers)
    if accumulator_nonfinite:
        return None, diagnostics
    return {
        "float64": float64,
        "b": b,
        "d": d,
        "b_squares": tuple(b_squares),
        "d_squares": tuple(d_squares),
        "active_layers": tuple(active_layers),
    }, diagnostics


def _evaluate_candidate(
    cpu: Mapping[str, Mapping[str, torch.Tensor]],
    prepared: Mapping[str, Any] | None,
    row_map: Mapping[str, tuple[RowAssignment, ...]],
    *,
    preparation_status: str,
    ordinal: int,
    content_count: int,
    structural_count: int,
    direction: int,
) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {
        "candidate_ordinal": ordinal,
        "content_count": content_count,
        "structural_count": structural_count,
        "direction": direction,
        "row_map": _serial_row_map(row_map),
        "accepted": False,
        "rejection_reasons": [],
        "active_layer_count": 0,
        "active_layer_indices": [],
        "moved_destination_checks": [],
        "layer_diagnostics": [],
        "aggregate_a_sum_squares": None,
        "aggregate_b_sum_squares": None,
        "aggregate_d_sum_squares": None,
        "aggregate_a_norm": None,
        "aggregate_b_norm": None,
        "aggregate_d_norm": None,
        "aggregate_ratio": None,
        "median_ratio": None,
        "aggregate_dot_a_d": None,
        "aggregate_cosine_a_d": None,
        "checks": {
            "active_layer_count": False,
            "every_moved_destination_changed": False,
            "aggregate_ratio": False,
            "median_ratio": False,
            "absolute_aggregate_cosine": False,
        },
    }
    if prepared is None:
        diagnostic["rejection_reasons"].append(
            f"NUMERIC_PREPARATION_NOT_READY:{preparation_status}")
        return diagnostic

    active_layers: tuple[int, ...] = prepared["active_layers"]
    diagnostic["active_layer_count"] = len(active_layers)
    diagnostic["active_layer_indices"] = list(active_layers)

    moved_checks: list[dict[str, Any]] = []
    every_moved_changed = True
    for class_name in CLASS_NAMES:
        for assignment in row_map[class_name]:
            differing_layers: list[int] = []
            for layer_index in active_layers:
                donor = cpu["correct"][class_name][
                    layer_index, assignment.donor]
                destination = cpu["fresh"][class_name][
                    layer_index, assignment.destination]
                if not _bytewise_equal(donor, destination):
                    differing_layers.append(layer_index)
            changed = bool(differing_layers)
            every_moved_changed = every_moved_changed and changed
            moved_checks.append({
                "class": class_name,
                "destination": assignment.destination,
                "donor": assignment.donor,
                "differs_from_fresh_in_active_layer": changed,
                "differing_active_layer_indices": differing_layers,
            })
    diagnostic["moved_destination_checks"] = moved_checks

    a_squares: list[float] = []
    a_d_dots: list[float] = []
    ratios: list[tuple[float, int]] = []
    nonfinite_accumulator = False
    for layer_index in active_layers:
        a_rows: list[tuple[list[float], list[float]]] = []
        for class_name in CLASS_NAMES:
            by_destination = sorted(
                row_map[class_name], key=lambda item: item.destination)
            for assignment in by_destination:
                a_row = (
                    prepared["float64"]["correct"][class_name][
                        layer_index, assignment.donor] -
                    prepared["float64"]["fresh"][class_name][
                        layer_index, assignment.destination]
                ).reshape(-1).tolist()
                d_row = prepared["d"][class_name][
                    layer_index, assignment.destination].reshape(-1).tolist()
                a_rows.append((a_row, d_row))

        # Zero coordinates at unselected destinations are omitted.  Inserting
        # exact +0.0 terms cannot change math.fsum; selected rows remain in the
        # frozen class/token/head/dimension order.
        a_square = math.fsum(
            value * value
            for a_row, _ in a_rows
            for value in a_row
        )
        a_d_dot = math.fsum(
            left * right
            for a_row, d_row in a_rows
            for left, right in zip(a_row, d_row, strict=True)
        )
        b_square = prepared["b_squares"][layer_index]
        a_norm = math.sqrt(a_square) if a_square >= 0.0 else math.nan
        b_norm = math.sqrt(b_square) if b_square >= 0.0 else math.nan
        ratio = a_norm / b_norm if b_norm > 0.0 else math.nan
        finite = all(math.isfinite(value) for value in (
            a_square, a_d_dot, b_square, a_norm, b_norm, ratio))
        if not finite:
            nonfinite_accumulator = True
        a_squares.append(a_square)
        a_d_dots.append(a_d_dot)
        ratios.append((ratio, layer_index))
        diagnostic["layer_diagnostics"].append({
            "layer_index": layer_index,
            "a_sum_squares": a_square,
            "a_norm": a_norm,
            "b_sum_squares": b_square,
            "b_norm": b_norm,
            "displacement_ratio": ratio,
            "dot_a_d": a_d_dot,
            "finite": finite,
        })

    aggregate_a_square = math.fsum(a_squares)
    aggregate_b_square = math.fsum(
        prepared["b_squares"][layer_index]
        for layer_index in active_layers)
    aggregate_d_square = math.fsum(
        prepared["d_squares"][layer_index]
        for layer_index in active_layers)
    aggregate_dot = math.fsum(a_d_dots)
    aggregate_a_norm = (
        math.sqrt(aggregate_a_square)
        if aggregate_a_square >= 0.0 else math.nan)
    aggregate_b_norm = (
        math.sqrt(aggregate_b_square)
        if aggregate_b_square >= 0.0 else math.nan)
    aggregate_d_norm = (
        math.sqrt(aggregate_d_square)
        if aggregate_d_square >= 0.0 else math.nan)
    aggregate_ratio = (
        aggregate_a_norm / aggregate_b_norm
        if aggregate_b_norm > 0.0 else math.nan)
    cosine_denominator = aggregate_a_norm * aggregate_d_norm
    aggregate_cosine = (
        aggregate_dot / cosine_denominator
        if cosine_denominator > 0.0 else math.nan)

    ratios.sort(key=lambda item: (item[0], item[1]))
    median_ratio: float
    if not ratios:
        median_ratio = math.nan
    elif len(ratios) % 2:
        median_ratio = ratios[len(ratios) // 2][0]
    else:
        right = len(ratios) // 2
        median_ratio = math.fsum(
            (ratios[right - 1][0], ratios[right][0])) / 2.0

    aggregate_values = (
        aggregate_a_square, aggregate_b_square, aggregate_d_square,
        aggregate_dot, aggregate_a_norm, aggregate_b_norm,
        aggregate_d_norm, aggregate_ratio, aggregate_cosine, median_ratio,
    )
    if not all(math.isfinite(value) for value in aggregate_values):
        nonfinite_accumulator = True

    diagnostic.update({
        "aggregate_a_sum_squares": aggregate_a_square,
        "aggregate_b_sum_squares": aggregate_b_square,
        "aggregate_d_sum_squares": aggregate_d_square,
        "aggregate_a_norm": aggregate_a_norm,
        "aggregate_b_norm": aggregate_b_norm,
        "aggregate_d_norm": aggregate_d_norm,
        "aggregate_ratio": aggregate_ratio,
        "median_ratio": median_ratio,
        "sorted_layer_ratios": [
            {"ratio": ratio, "layer_index": layer_index}
            for ratio, layer_index in ratios
        ],
        "aggregate_dot_a_d": aggregate_dot,
        "aggregate_cosine_a_d": aggregate_cosine,
    })

    active_ok = len(active_layers) >= MIN_ACTIVE_LAYERS
    aggregate_ratio_ok = (
        math.isfinite(aggregate_ratio) and
        AGGREGATE_RATIO_INTERVAL[0] <= aggregate_ratio <=
        AGGREGATE_RATIO_INTERVAL[1])
    median_ratio_ok = (
        math.isfinite(median_ratio) and
        MEDIAN_RATIO_INTERVAL[0] <= median_ratio <=
        MEDIAN_RATIO_INTERVAL[1])
    cosine_ok = (
        math.isfinite(aggregate_cosine) and
        abs(aggregate_cosine) <= MAX_ABS_AGGREGATE_COSINE)
    diagnostic["checks"] = {
        "active_layer_count": active_ok,
        "every_moved_destination_changed": every_moved_changed,
        "aggregate_ratio": aggregate_ratio_ok,
        "median_ratio": median_ratio_ok,
        "absolute_aggregate_cosine": cosine_ok,
    }

    reasons: list[str] = diagnostic["rejection_reasons"]
    if not active_ok:
        reasons.append("ACTIVE_LAYER_COUNT_BELOW_24")
    if not every_moved_changed:
        reasons.append("MOVED_DESTINATION_UNCHANGED_IN_ALL_ACTIVE_LAYERS")
    if aggregate_a_norm == 0.0:
        reasons.append("ZERO_A_AGGREGATE_DENOMINATOR")
    if aggregate_b_norm == 0.0:
        reasons.append("ZERO_B_AGGREGATE_DENOMINATOR")
    if aggregate_d_norm == 0.0:
        reasons.append("ZERO_D_AGGREGATE_DENOMINATOR")
    if nonfinite_accumulator:
        reasons.append("NONFINITE_DIFFERENCE_OR_ACCUMULATOR")
    if not aggregate_ratio_ok:
        reasons.append("AGGREGATE_RATIO_OUTSIDE_INCLUSIVE_INTERVAL")
    if not median_ratio_ok:
        reasons.append("MEDIAN_RATIO_OUTSIDE_INCLUSIVE_INTERVAL")
    if not cosine_ok:
        reasons.append("ABS_AGGREGATE_COSINE_ABOVE_INCLUSIVE_MAXIMUM")

    diagnostic["accepted"] = not reasons
    return diagnostic


def _bytewise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_bytes = left.detach().contiguous().view(torch.uint8)
    right_bytes = right.detach().contiguous().view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def _serial_row_map(
    row_map: Mapping[str, Sequence[RowAssignment]],
) -> dict[str, list[dict[str, int]]]:
    return {
        class_name: [
            {"destination": assignment.destination, "donor": assignment.donor}
            for assignment in row_map[class_name]
        ]
        for class_name in CLASS_NAMES
    }


__all__ = [
    "AGGREGATE_RATIO_INTERVAL",
    "CLASS_NAMES",
    "CONTENT_CLASS",
    "DESIGN_ID",
    "LAYER_COUNT",
    "MAX_ABS_AGGREGATE_COSINE",
    "MEDIAN_RATIO_INTERVAL",
    "MIN_ACTIVE_LAYERS",
    "MOVED_COUNT_SCHEDULE",
    "RENDER_IDS",
    "STRUCTURAL_CLASS",
    "RowAssignment",
    "V13PlaceboError",
    "ValueRowPlaceboResult",
    "apply_value_row_map",
    "build_value_row_placebo",
    "candidate_row_map",
    "canonical_json_bytes",
    "canonical_row_permutation",
    "retained_moved_counts",
]
