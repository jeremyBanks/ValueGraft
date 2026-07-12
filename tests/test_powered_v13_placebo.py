from __future__ import annotations

from hashlib import sha256
import json
import math

import pytest
import torch

import powered_v13_placebo as placebo


CASE_ID = "0" * 64


def _sources(
    *,
    content_width: int = 2,
    structural_width: int = 2,
    active_layers: int = 48,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor],
           dict[str, torch.Tensor]]:
    """Make a hand-solvable first-candidate pass with scalar V rows."""

    fresh = {
        placebo.CONTENT_CLASS: torch.zeros(
            (48, content_width, 1, 1), dtype=torch.bfloat16),
        placebo.STRUCTURAL_CLASS: torch.zeros(
            (48, structural_width, 1, 1), dtype=torch.bfloat16),
    }
    correct = {name: value.clone() for name, value in fresh.items()}
    wrong = {name: value.clone() for name, value in fresh.items()}
    for layer in range(active_layers):
        correct[placebo.CONTENT_CLASS][layer, 0, 0, 0] = 1.0
        correct[placebo.CONTENT_CLASS][layer, 1, 0, 0] = 2.0
        correct[placebo.STRUCTURAL_CLASS][layer, 0, 0, 0] = 3.0
        correct[placebo.STRUCTURAL_CLASS][layer, 1, 0, 0] = 4.0
        # d=C-W=[1,-2,0,0], exactly orthogonal to the swapped
        # first-candidate displacement a=[2,1,4,3].
        wrong[placebo.CONTENT_CLASS][layer, 0, 0, 0] = 0.0
        wrong[placebo.CONTENT_CLASS][layer, 1, 0, 0] = 4.0
        wrong[placebo.STRUCTURAL_CLASS][layer, 0, 0, 0] = 3.0
        wrong[placebo.STRUCTURAL_CLASS][layer, 1, 0, 0] = 4.0
    return fresh, correct, wrong


def _build(fresh, correct, wrong, *, render_id="r1"):
    return placebo.build_value_row_placebo(
        fresh,
        correct,
        wrong,
        design_id=placebo.DESIGN_ID,
        stable_candidate_id=CASE_ID,
        render_id=render_id,
    )


def _raw_bytes(value: torch.Tensor) -> bytes:
    return value.contiguous().view(torch.uint8).numpy().tobytes()


def test_golden_canonical_json_permutation_is_independently_recomputed():
    order, digests = placebo.canonical_row_permutation(
        5,
        design_id=placebo.DESIGN_ID,
        stable_candidate_id=CASE_ID,
        render_id="r1",
        class_name="content",
    )
    assert order == (4, 0, 1, 3, 2)
    assert digests == (
        "0747ee16f205b885fcd234764af7b15ac2ce2e329e64fc88b3f48175f63897cf",
        "b389a230f8ae40e6321af0377c2ea13edb504c92ac14480cdcc9b11600ab2d37",
        "d50416b4cb41324e4756bca899d38e062f4f1e3b75dd90a47c53f5c80090d3de",
        "fb029ca39083582149c2a311fc98005d5b72c99fac5d6f8e094aa850ce4c5858",
        "fee8b58e0570f6e6e48bc3b3a430e0163bb1bae4c61edfc32bbd27aceba9b5f4",
    )

    independent = []
    for row in range(5):
        encoded = json.dumps(
            [placebo.DESIGN_ID, CASE_ID, "r1", "content", row],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        independent.append((sha256(encoded).digest(), row))
    independent.sort(key=lambda item: (item[0], item[1]))
    assert order == tuple(row for _, row in independent)
    assert digests == tuple(digest.hex() for digest, _ in independent)


def test_golden_map_uses_donor_to_shifted_destination_and_classes_never_mix():
    permutations = {
        "content": (4, 0, 1, 3, 2),
        "structural": (0, 2, 1, 4, 3),
    }
    row_map = placebo.candidate_row_map(
        permutations,
        content_count=4,
        structural_count=4,
        direction=+1,
    )
    assert row_map["content"] == (
        placebo.RowAssignment(destination=0, donor=4),
        placebo.RowAssignment(destination=1, donor=0),
        placebo.RowAssignment(destination=3, donor=1),
        placebo.RowAssignment(destination=4, donor=3),
    )
    assert row_map["structural"] == (
        placebo.RowAssignment(destination=2, donor=0),
        placebo.RowAssignment(destination=1, donor=2),
        placebo.RowAssignment(destination=4, donor=1),
        placebo.RowAssignment(destination=0, donor=4),
    )
    reverse = placebo.candidate_row_map(
        permutations,
        content_count=4,
        structural_count=4,
        direction=-1,
    )
    assert reverse["content"] == (
        placebo.RowAssignment(destination=3, donor=4),
        placebo.RowAssignment(destination=4, donor=0),
        placebo.RowAssignment(destination=0, donor=1),
        placebo.RowAssignment(destination=1, donor=3),
    )


@pytest.mark.parametrize(
    ("width", "expected"),
    ((1, ()), (2, (2,)), (3, (2, 3)), (8, (2, 4, 8)),
     (65, (2, 4, 8, 16, 32, 64, 65))),
)
def test_retained_count_schedule_filters_and_deduplicates(width, expected):
    assert placebo.retained_moved_counts(width) == expected


def test_synthetic_available_case_matches_hand_calculated_fsum_diagnostics():
    fresh, correct, wrong = _sources()
    result = _build(fresh, correct, wrong)
    assert result.status == "AVAILABLE"
    assert result.values is not None
    assert result.row_map is not None
    diagnostics = result.diagnostics
    assert diagnostics["accepted_candidate_ordinal"] == 1
    assert diagnostics["evaluated_candidate_count"] == 1
    candidate = diagnostics["candidate_diagnostics"][0]
    assert candidate["accepted"] is True
    assert candidate["active_layer_count"] == 48
    assert candidate["aggregate_a_sum_squares"] == 48.0 * 30.0
    assert candidate["aggregate_b_sum_squares"] == 48.0 * 30.0
    assert candidate["aggregate_d_sum_squares"] == 48.0 * 5.0
    assert candidate["aggregate_dot_a_d"] == 0.0
    assert candidate["aggregate_ratio"] == 1.0
    assert candidate["median_ratio"] == 1.0
    assert candidate["aggregate_cosine_a_d"] == 0.0
    assert all(candidate["checks"].values())


def test_accepted_map_is_a_lossless_whole_bf16_row_copy():
    fresh, correct, wrong = _sources()
    result = _build(fresh, correct, wrong)
    assert result.row_map is not None and result.values is not None
    replay = placebo.apply_value_row_map(fresh, correct, result.row_map)
    for class_name in placebo.CLASS_NAMES:
        assert _raw_bytes(replay[class_name]) == _raw_bytes(result.values[class_name])
        mapped_destinations = {
            assignment.destination for assignment in result.row_map[class_name]}
        for assignment in result.row_map[class_name]:
            assert _raw_bytes(replay[class_name][:, assignment.destination]) == (
                _raw_bytes(correct[class_name][:, assignment.donor]))
        for row in range(fresh[class_name].shape[1]):
            if row not in mapped_destinations:
                assert _raw_bytes(replay[class_name][:, row]) == (
                    _raw_bytes(fresh[class_name][:, row]))


def test_apply_map_leaves_unselected_rows_byte_identical():
    fresh = {
        name: torch.arange(48 * 5, dtype=torch.float32).reshape(48, 5, 1, 1)
        .to(torch.bfloat16)
        for name in placebo.CLASS_NAMES
    }
    correct = {name: value + torch.tensor(1000, dtype=torch.bfloat16)
               for name, value in fresh.items()}
    row_map = {
        "content": (
            placebo.RowAssignment(destination=0, donor=1),
            placebo.RowAssignment(destination=1, donor=0),
        ),
        "structural": (
            placebo.RowAssignment(destination=3, donor=4),
            placebo.RowAssignment(destination=4, donor=3),
        ),
    }
    applied = placebo.apply_value_row_map(fresh, correct, row_map)
    assert _raw_bytes(applied["content"][:, 2:]) == _raw_bytes(fresh["content"][:, 2:])
    assert _raw_bytes(applied["structural"][:, :3]) == _raw_bytes(fresh["structural"][:, :3])


def test_fewer_than_24_active_layers_is_unavailable_with_every_reject_saved():
    fresh, correct, wrong = _sources(active_layers=23)
    result = _build(fresh, correct, wrong)
    assert result.status == "PLACEBO_UNAVAILABLE"
    assert result.values is None and result.row_map is None
    assert result.diagnostics["candidate_capacity"] == 2
    assert result.diagnostics["evaluated_candidate_count"] == 2
    assert len(result.diagnostics["candidate_diagnostics"]) == 2
    for candidate in result.diagnostics["candidate_diagnostics"]:
        assert candidate["active_layer_count"] == 23
        assert "ACTIVE_LAYER_COUNT_BELOW_24" in candidate["rejection_reasons"]


def test_nonfinite_input_marks_all_candidates_unavailable_with_exact_location():
    fresh, correct, wrong = _sources()
    correct["content"][7, 1, 0, 0] = float("inf")
    result = _build(fresh, correct, wrong)
    assert result.status == "PLACEBO_UNAVAILABLE"
    preparation = result.diagnostics["numeric_preparation"]
    assert preparation["status"] == "NONFINITE_INPUT"
    assert preparation["nonfinite_input_locations"] == [{
        "source": "correct",
        "class": "content",
        "layer": 7,
        "row": 1,
        "kv_head": 0,
        "head_dimension": 0,
    }]
    assert result.diagnostics["evaluated_candidate_count"] == 2
    assert all(
        candidate["rejection_reasons"] == [
            "NUMERIC_PREPARATION_NOT_READY:NONFINITE_INPUT"]
        for candidate in result.diagnostics["candidate_diagnostics"])


def test_zero_b_and_undefined_cosine_are_explicit_rejection_paths():
    fresh, correct, wrong = _sources(active_layers=0)
    wrong["content"].fill_(1.0)
    result = _build(fresh, correct, wrong)
    assert result.status == "PLACEBO_UNAVAILABLE"
    candidate = result.diagnostics["candidate_diagnostics"][0]
    assert candidate["aggregate_a_norm"] == 0.0
    assert candidate["aggregate_b_norm"] == 0.0
    assert candidate["aggregate_d_norm"] == 0.0  # only active layers aggregate
    assert math.isnan(candidate["aggregate_cosine_a_d"])
    assert "ZERO_A_AGGREGATE_DENOMINATOR" in candidate["rejection_reasons"]
    assert "ZERO_B_AGGREGATE_DENOMINATOR" in candidate["rejection_reasons"]
    assert "ZERO_D_AGGREGATE_DENOMINATOR" in candidate["rejection_reasons"]


def test_zero_d_rejects_even_when_a_b_and_all_layers_are_valid():
    fresh, correct, _ = _sources()
    wrong = {name: value.clone() for name, value in correct.items()}
    result = _build(fresh, correct, wrong)
    assert result.status == "PLACEBO_UNAVAILABLE"
    candidate = result.diagnostics["candidate_diagnostics"][0]
    assert candidate["active_layer_count"] == 48
    assert candidate["aggregate_a_norm"] > 0.0
    assert candidate["aggregate_b_norm"] > 0.0
    assert candidate["aggregate_d_norm"] == 0.0
    assert math.isnan(candidate["aggregate_cosine_a_d"])
    assert "ZERO_D_AGGREGATE_DENOMINATOR" in candidate["rejection_reasons"]


def test_cosine_gate_rejects_semantically_aligned_permutation():
    fresh, correct, _ = _sources()
    wrong = {name: value.clone() for name, value in fresh.items()}
    result = _build(fresh, correct, wrong)
    assert result.status == "PLACEBO_UNAVAILABLE"
    candidate = result.diagnostics["candidate_diagnostics"][0]
    assert candidate["aggregate_cosine_a_d"] > 0.20
    assert "ABS_AGGREGATE_COSINE_ABOVE_INCLUSIVE_MAXIMUM" in (
        candidate["rejection_reasons"])


def test_byte_change_gate_rejects_a_moved_row_equal_to_fresh_everywhere():
    fresh, correct, wrong = _sources()
    # The zero donor is moved onto another zero fresh destination by both
    # directions of the two-cycle.  Other rows keep every layer active.
    correct["content"][:, 0, 0, 0] = 0.0
    wrong = {name: value.clone() for name, value in fresh.items()}
    result = _build(fresh, correct, wrong)
    assert result.status == "PLACEBO_UNAVAILABLE"
    candidate = result.diagnostics["candidate_diagnostics"][0]
    unchanged = [
        check for check in candidate["moved_destination_checks"]
        if not check["differs_from_fresh_in_active_layer"]
    ]
    assert unchanged
    assert "MOVED_DESTINATION_UNCHANGED_IN_ALL_ACTIVE_LAYERS" in (
        candidate["rejection_reasons"])


def test_ratio_gate_rejects_large_position_baseline_displacement():
    fresh, correct, wrong = _sources()
    for class_name in placebo.CLASS_NAMES:
        fresh[class_name][:, 1, 0, 0] = 10.0
        correct[class_name][:, 0, 0, 0] = 1.0
        correct[class_name][:, 1, 0, 0] = 11.0
        wrong[class_name] = correct[class_name].clone()
        wrong[class_name][:, 0, 0, 0] -= 1.0
    result = _build(fresh, correct, wrong)
    assert result.status == "PLACEBO_UNAVAILABLE"
    candidate = result.diagnostics["candidate_diagnostics"][0]
    assert candidate["aggregate_ratio"] > 1.33
    assert "AGGREGATE_RATIO_OUTSIDE_INCLUSIVE_INTERVAL" in (
        candidate["rejection_reasons"])


def test_width_one_reports_unavailable_without_synthetic_candidate():
    shape = (48, 1, 1, 1)
    fresh = {name: torch.zeros(shape, dtype=torch.bfloat16)
             for name in placebo.CLASS_NAMES}
    correct = {name: value.clone() for name, value in fresh.items()}
    wrong = {name: value.clone() for name, value in fresh.items()}
    result = _build(fresh, correct, wrong)
    assert result.status == "PLACEBO_UNAVAILABLE"
    assert result.diagnostics["unavailable_reason"] == (
        "NO_RETAINED_MOVED_COUNT_FOR_BOTH_CLASSES")
    assert result.diagnostics["candidate_capacity"] == 0
    assert result.diagnostics["candidate_diagnostics"] == []


def test_full_98_candidate_search_has_literal_nested_order():
    shape = (48, 65, 1, 1)
    fresh = {name: torch.zeros(shape, dtype=torch.bfloat16)
             for name in placebo.CLASS_NAMES}
    correct = {name: value.clone() for name, value in fresh.items()}
    wrong = {name: value.clone() for name, value in fresh.items()}
    result = _build(fresh, correct, wrong)
    candidates = result.diagnostics["candidate_diagnostics"]
    assert len(candidates) == 98
    observed = [
        (item["content_count"], item["structural_count"], item["direction"])
        for item in candidates
    ]
    counts = (2, 4, 8, 16, 32, 64, 65)
    assert observed == [
        (content, structural, direction)
        for content in counts
        for structural in counts
        for direction in (+1, -1)
    ]


def test_deterministic_repeat_is_byte_and_diagnostic_identical():
    fresh, correct, wrong = _sources()
    first = _build(fresh, correct, wrong, render_id="r2")
    second = _build(fresh, correct, wrong, render_id="r2")
    assert first.status == second.status == "AVAILABLE"
    assert first.row_map == second.row_map
    assert first.diagnostics == second.diagnostics
    assert first.values is not None and second.values is not None
    for class_name in placebo.CLASS_NAMES:
        assert _raw_bytes(first.values[class_name]) == (
            _raw_bytes(second.values[class_name]))


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda sources: sources[0]["content"].float(), "torch.bfloat16"),
        (lambda sources: sources[0]["content"][:47], "invalid v13 geometry"),
    ),
)
def test_invalid_dtype_or_geometry_is_contract_error_not_unavailable(
        mutation, match):
    fresh, correct, wrong = _sources()
    fresh["content"] = mutation((fresh, correct, wrong))
    with pytest.raises(placebo.V13PlaceboError, match=match):
        _build(fresh, correct, wrong)


def test_identity_fields_are_frozen_hash_inputs():
    fresh, correct, wrong = _sources()
    with pytest.raises(placebo.V13PlaceboError, match="design_id"):
        placebo.build_value_row_placebo(
            fresh, correct, wrong,
            design_id="v13-ish", stable_candidate_id=CASE_ID, render_id="r1")
    with pytest.raises(placebo.V13PlaceboError, match="lowercase hex"):
        placebo.build_value_row_placebo(
            fresh, correct, wrong,
            design_id=placebo.DESIGN_ID,
            stable_candidate_id="A" * 64,
            render_id="r1")
    with pytest.raises(placebo.V13PlaceboError, match="r1 or r2"):
        placebo.build_value_row_placebo(
            fresh, correct, wrong,
            design_id=placebo.DESIGN_ID,
            stable_candidate_id=CASE_ID,
            render_id="render-1")
