from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_c10_schedule_origin",
    ROOT / "scripts" / "diagnose_c10_schedule_origin.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _snapshot(*, layers: int = 2, rows: int = 5) -> MODULE.Snapshot:
    return [(
        torch.zeros((1, 2, rows, 2), dtype=torch.bfloat16),
        torch.zeros((1, 2, rows, 2), dtype=torch.bfloat16),
    ) for _ in range(layers)]


def _comparison(*, exact: bool, first_layer: int | None = None) -> dict:
    return {
        "bit_exact": exact,
        "first_divergence_layer_row_kind": (
            None if exact else [first_layer, 0, "K"]),
    }


def test_compare_region_records_per_layer_per_row_and_coordinates():
    left = _snapshot(rows=5)
    right = _snapshot(rows=5)
    right[1][0][..., 2, 0] = 0.5
    observed = MODULE.compare_region(
        left, right, left_start=0, right_start=0, width=5,
        include_per_row=True, label="test")
    assert observed["bit_exact"] is False
    assert observed["k_max_abs"] == 0.5
    assert observed["v_max_abs"] == 0.0
    assert observed["first_divergence_layer_row_kind"] == [1, 2, "K"]
    assert observed["first_divergence_row_layer_kind"] == [2, 1, "K"]
    assert observed["per_row_aggregate_across_layers"][2] == {
        "left_row": 2, "right_row": 2,
        "k_bit_exact": False, "v_bit_exact": True,
        "k_max_abs": 0.5, "v_max_abs": 0.0,
    }
    assert observed["per_layer"][1]["per_row"][2] == {
        "left_row": 2, "right_row": 2,
        "k_bit_exact": False, "v_bit_exact": True,
        "k_max_abs": 0.5, "v_max_abs": 0.0,
    }


def test_compare_region_rejects_out_of_range_rows():
    with pytest.raises(MODULE.DiagnosticError, match="requested rows"):
        MODULE.compare_region(
            _snapshot(rows=4), _snapshot(rows=4), left_start=0,
            right_start=0, width=5, include_per_row=True, label="test")


def test_shape_dependence_class_requires_exact_causal_control():
    observed = MODULE.classify_three_way(
        _comparison(exact=False, first_layer=1),
        _comparison(exact=True),
        _comparison(exact=False, first_layer=0))
    assert observed["outcome_class"] == "QUERY_SHAPE_ROUNDING"
    assert observed["fallback_required"] is False


def test_future_token_influence_has_priority_over_a_b_difference():
    observed = MODULE.classify_three_way(
        _comparison(exact=False, first_layer=1),
        _comparison(exact=False, first_layer=2),
        _comparison(exact=False, first_layer=0))
    assert observed["outcome_class"] == "FUTURE_TOKEN_INFLUENCE"


def test_layer_zero_a_b_difference_is_construction_bug_class():
    observed = MODULE.classify_three_way(
        _comparison(exact=False, first_layer=0),
        _comparison(exact=True),
        _comparison(exact=False, first_layer=0))
    assert observed["outcome_class"] == "CONSTRUCTION_DIVERGENCE"


def test_layer_zero_construction_class_precedes_future_influence():
    observed = MODULE.classify_three_way(
        _comparison(exact=False, first_layer=0),
        _comparison(exact=False, first_layer=1),
        _comparison(exact=False, first_layer=0))
    assert observed["outcome_class"] == "CONSTRUCTION_DIVERGENCE"


def test_all_first23_exact_predeclares_fallback():
    observed = MODULE.classify_three_way(
        _comparison(exact=True), _comparison(exact=True),
        _comparison(exact=False, first_layer=0))
    assert observed["outcome_class"] == "NO_FIRST_BOUNDARY_DIVERGENCE"
    assert observed["fallback_required"] is True


def test_positive_control_must_change_rows_after_22():
    with pytest.raises(MODULE.DiagnosticError, match="positive control"):
        MODULE.classify_three_way(
            _comparison(exact=True), _comparison(exact=True),
            _comparison(exact=True))


def test_repeat_comparison_fails_closed_on_one_bit_difference():
    left = _snapshot()
    right = _snapshot()
    logits_left = torch.zeros((1, 3), dtype=torch.bfloat16)
    logits_right = logits_left.clone()
    result = MODULE.compare_repeat_snapshots(
        left, right, logits_left, logits_right, "repeat")
    assert result["bit_exact"] is True
    right[0][1][..., 0, 0] = 1.0
    with pytest.raises(MODULE.DiagnosticError, match="not bit-identical"):
        MODULE.compare_repeat_snapshots(
            left, right, logits_left, logits_right, "repeat")


def test_raw_bit_identity_distinguishes_positive_and_negative_zero():
    positive = torch.tensor([0.0], dtype=torch.bfloat16)
    negative = torch.tensor([-0.0], dtype=torch.bfloat16)
    assert torch.equal(positive, negative) is True
    assert MODULE.tensors_bit_equal(positive, negative) is False

    left = _snapshot(rows=1)
    right = _snapshot(rows=1)
    right[0][0][..., 0, 0] = -0.0
    observed = MODULE.compare_region(
        left, right, left_start=0, right_start=0, width=1,
        include_per_row=True, label="signed-zero")
    assert observed["bit_exact"] is False
    assert observed["k_max_abs"] == 0.0
    assert observed["first_divergence_layer_row_kind"] == [0, 0, "K"]


def test_repeat_gate_records_but_does_not_gate_on_logits():
    snapshot = _snapshot()
    left_logits = torch.zeros((1, 3), dtype=torch.bfloat16)
    right_logits = left_logits.clone()
    right_logits[0, 0] = 1.0
    observed = MODULE.compare_repeat_snapshots(
        snapshot, _snapshot(), left_logits, right_logits, "repeat")
    assert observed["cache_bit_exact"] is True
    assert observed["bit_exact"] is True
    assert observed["logits_bit_exact"] is False


def test_sealed_payload_detects_tampering():
    sealed = MODULE.seal_payload({
        "schema": 1, "diagnostic_id": MODULE.DIAGNOSTIC_ID,
        "status": "COMPLETE"})
    MODULE.verify_sealed_payload(sealed)
    tampered = json.loads(json.dumps(sealed))
    tampered["status"] = "FAIL"
    with pytest.raises(MODULE.DiagnosticError, match="payload hash"):
        MODULE.verify_sealed_payload(tampered)


def test_repeat_failure_has_dedicated_exception_class():
    left = _snapshot()
    right = _snapshot()
    right[0][0][..., 0, 0] = 1.0
    logits = torch.zeros((1, 3), dtype=torch.bfloat16)
    with pytest.raises(MODULE.NondeterminismError):
        MODULE.compare_repeat_snapshots(
            left, right, logits, logits.clone(), "repeat")


def test_repeated_branch_failure_preserves_both_snapshot_witnesses(monkeypatch):
    first = _snapshot()
    second = _snapshot()
    second[0][0][..., 0, 0] = 1.0
    logits = torch.zeros((1, 3), dtype=torch.bfloat16)
    calls = iter(((first, logits), (second, logits.clone())))
    monkeypatch.setattr(MODULE, "_run_branch", lambda _model, _ids: next(calls))

    with pytest.raises(MODULE.NondeterminismError) as caught:
        MODULE._run_repeated_branch(None, "A", [1, 2, 3, 4, 5])

    failed = caught.value.evidence["failed_branch"]
    assert failed["repeat_1"]["snapshot"]["layer_count"] == 2
    assert failed["repeat_2"]["snapshot"]["layer_count"] == 2
    assert failed["repeat_1"]["snapshot"]["aggregate_sha256"] != \
        failed["repeat_2"]["snapshot"]["aggregate_sha256"]
    assert failed["repeat_identity"]["cache_bit_exact"] is False


def test_constants_match_frozen_diagnostic_contract():
    assert MODULE.DIAGNOSTIC_ID == \
        "coherent-state-c10-schedule-origin-v1"
    assert MODULE.CASE_PATH == "data/synthetic/c10.json"
    assert MODULE.REPLACEMENT_CASE_PATH == "data/synthetic/c02.json"
    assert MODULE.PREFIX_LENGTH == 4096
    assert MODULE.FIRST_BLOCK_LENGTH == 23


def test_diagnostic_files_remain_outside_frozen_apparatus_inventory():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from coherent_state_integrity import apparatus_inventory

    inventory = apparatus_inventory(ROOT)
    paths = {row["path"] for row in inventory["files"]}
    assert "scripts/diagnose_c10_schedule_origin.py" not in paths
    assert inventory["file_count"] == 35
    assert inventory["aggregate_sha256"] == \
        "818a60623c4858f0796865a124d255897325136f147ad611c1810026c9352715"
