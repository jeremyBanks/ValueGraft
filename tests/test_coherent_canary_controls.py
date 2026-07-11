from __future__ import annotations

import pytest
import torch

from coherent_canary_controls import (
    CanaryControlError,
    norm_matched_value_placebo_row,
)


def test_float64_control_is_deterministic_orthogonal_and_norm_matched() -> None:
    fresh = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=torch.float64)
    correct = fresh + torch.tensor(
        [[0.5, 0.25, -0.75], [0.2, -0.3, 0.4]], dtype=torch.float64)
    wrong = fresh - torch.tensor(
        [[0.1, -0.2, 0.05], [0.3, 0.15, -0.1]], dtype=torch.float64)

    first, first_diag = norm_matched_value_placebo_row(
        fresh, correct, wrong, case_id="e01", layer_index=7, row_index=11)
    second, second_diag = norm_matched_value_placebo_row(
        fresh, correct, wrong, case_id="e01", layer_index=7, row_index=11)
    delta = (correct - wrong).reshape(-1)
    applied = (first - fresh).reshape(-1)

    assert torch.equal(first, second)
    assert first_diag == second_diag
    assert torch.dot(delta, applied).item() == pytest.approx(0.0, abs=1e-14)
    assert torch.linalg.vector_norm(applied).item() == pytest.approx(
        torch.linalg.vector_norm(delta).item(), rel=1e-14)
    assert first_diag["pre_cast_cosine_with_delta"] == pytest.approx(0.0, abs=1e-14)
    assert first_diag["applied_relative_norm_error"] == pytest.approx(0.0, abs=1e-14)


def test_seed_coordinates_change_the_direction() -> None:
    fresh = torch.zeros((4, 8), dtype=torch.float32)
    correct = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    wrong = torch.zeros_like(correct)
    left, left_diag = norm_matched_value_placebo_row(
        fresh, correct, wrong, case_id="e02", layer_index=1, row_index=2)
    right, right_diag = norm_matched_value_placebo_row(
        fresh, correct, wrong, case_id="e02", layer_index=1, row_index=3)
    assert not torch.equal(left, right)
    assert left_diag["seed_material"] != right_diag["seed_material"]


def test_bfloat16_reports_rounding_separately() -> None:
    fresh = torch.linspace(-2, 2, 32, dtype=torch.bfloat16).reshape(4, 8)
    correct = fresh + torch.linspace(0.01, 0.32, 32, dtype=torch.bfloat16).reshape(4, 8)
    wrong = fresh - torch.linspace(0.02, 0.12, 32, dtype=torch.bfloat16).reshape(4, 8)
    result, diagnostics = norm_matched_value_placebo_row(
        fresh, correct, wrong, case_id="e03", layer_index=2, row_index=9)
    assert result.dtype == torch.bfloat16
    assert diagnostics["target_delta_l2"] > 0
    assert diagnostics["pre_cast_u_l2"] == pytest.approx(
        diagnostics["target_delta_l2"], rel=1e-12)
    assert abs(diagnostics["pre_cast_cosine_with_delta"]) < 1e-12
    assert diagnostics["applied_delta_l2"] > 0
    assert diagnostics["applied_relative_norm_error"] >= 0
    assert diagnostics["placebo_sha256"] != diagnostics["fresh_sha256"]


def test_zero_delta_returns_exact_fresh_row() -> None:
    fresh = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    result, diagnostics = norm_matched_value_placebo_row(
        fresh, fresh, fresh, case_id="e04", layer_index=0, row_index=0)
    assert torch.equal(result, fresh)
    assert result.data_ptr() != fresh.data_ptr()
    assert diagnostics["zero_semantic_delta"] is True
    assert diagnostics["placebo_sha256"] == diagnostics["fresh_sha256"]


@pytest.mark.parametrize("case_id", ["", "E01", "bad id", "../e01"])
def test_invalid_case_id_fails_closed(case_id: str) -> None:
    row = torch.zeros((2, 2), dtype=torch.float32)
    with pytest.raises(CanaryControlError, match="case_id"):
        norm_matched_value_placebo_row(
            row, row, row, case_id=case_id, layer_index=0, row_index=0)


def test_shape_dtype_and_finiteness_fail_closed() -> None:
    row = torch.zeros((2, 2), dtype=torch.float32)
    with pytest.raises(CanaryControlError, match="shapes"):
        norm_matched_value_placebo_row(
            row, row[:, :1], row, case_id="e01", layer_index=0, row_index=0)
    with pytest.raises(CanaryControlError, match="dtypes"):
        norm_matched_value_placebo_row(
            row, row.double(), row, case_id="e01", layer_index=0, row_index=0)
    bad = row.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(CanaryControlError, match="nonfinite"):
        norm_matched_value_placebo_row(
            row, bad, row, case_id="e01", layer_index=0, row_index=0)
