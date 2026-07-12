from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/compare_precision_probe_p01_p02.py"
SPEC = importlib.util.spec_from_file_location(
    "compare_precision_probe_p01_p02", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

P01 = next((ROOT / "results/precision_probe_p01_analysis").glob(
    "precision-probe-p01-postrun-partial-descriptive_*.json"
))
P02 = next((ROOT / "results/precision_probe_p02_analysis").glob(
    "precision-probe-p02-independent-analysis_*.json"
))


def _documents() -> tuple[dict, dict, dict, dict]:
    p01, p01_binding = MODULE.load_artifact(P01, "test P01")
    p02, p02_binding = MODULE.load_artifact(P02, "test P02")
    return p01, p02, p01_binding, p02_binding


def _compare(p01: dict, p02: dict, p01_binding: dict, p02_binding: dict) -> dict:
    return MODULE.compare_documents(p01, p02, p01_binding, p02_binding)


def test_committed_analyzer_artifacts_compare_exactly() -> None:
    result = _compare(*_documents())

    assert result["comparison_status"] == "PASS"
    assert result["p01"]["boundary"]["status"] == "POST_RUN_PARTIAL_DESCRIPTIVE"
    assert result["p01"]["boundary"]["bf16_repeat2"] == {
        "status": "ERROR",
        "error_type": "KeyboardInterrupt",
        "phase_a_present": False,
        "treatment_present": False,
    }
    assert result["p02"]["boundary"]["hard_eligibility"] == "PASS"
    assert result["p02"]["boundary"]["repeat2_rider_status"] == "MATCHED_COMPLETE"

    for regime in MODULE.REGIMES:
        assert all(
            value == 0.0
            for value in result["p02_minus_p01"]["by_regime"][regime].values()
        )
    assert all(
        value == 0.0
        for value in result["p02_minus_p01"][
            "within_protocol_delta_difference"
        ].values()
    )
    equality = result["exact_equalities"]
    assert equality["all_compared_estimands_equal"] is True
    assert equality["all_five_cell_generation_payloads_equal"] is True
    assert equality["all_placebo_payloads_equal"] is True

    assert result["p01"]["input_artifact"]["sha256"] == (
        "c370d1cc2dbe2a9ee5db64a19ce8837f9879f3b8fc008210756d2985d41aac7e"
    )
    assert result["p02"]["input_artifact"]["sha256"] == (
        "7b4b77e178f0b50d532459512dd028a044877627d4490197510eba135d5ab551"
    )


def test_emits_exact_generation_payloads_placebos_and_host_facts() -> None:
    result = _compare(*_documents())
    p01_nf4 = result["p01"]["repeat1"]["nf4"]
    p02_nf4 = result["p02"]["repeat1"]["nf4"]

    assert p01_nf4["generations"]["focal"]["cell_order"] == [
        "FF", "FC", "FW", "CC", "WW"
    ]
    assert p01_nf4["generations"]["focal"]["decoded_content"] == [
        "Atlas 4.8 was not selected under the recorded mandatory selection rule.",
        "Ring 3", "Ring 3", "Ring 3", "Ring 3",
    ]
    assert p01_nf4["generations"]["focal"] == p02_nf4["generations"]["focal"]
    assert p01_nf4["placebo"]["status"] == "PLACEBO_UNAVAILABLE"
    assert p01_nf4["placebo"]["failed_layer_index"] == 1
    assert p01_nf4["placebo"]["failed_row_index"] == 76
    assert p01_nf4["placebo"] == p02_nf4["placebo"]

    p01_host = result["p01"]["runtime_facts"]["regimes"]["nf4"]["host"]
    p02_host = result["p02"]["runtime_facts"]["regimes"]["nf4"]["host"]
    assert p01_host["gpu_uuid"] == "GPU-470c3e18-9f87-0a04-f3bd-e974d59e1903"
    assert p02_host["gpu_uuid"] == "GPU-8a42830e-71ab-fb23-351d-125ccbd5bdb2"
    assert p01_host["driver_version"] == "580.159.04"
    assert p02_host["driver_version"] == "580.159.03"
    assert result["exact_equalities"]["runtime"][
        "same_gpu_uuid_across_protocols"
    ] is False


def test_every_frozen_a8_expectation_is_addressed_without_new_inference() -> None:
    result = _compare(*_documents())
    frozen = result["frozen_A8_expectations"]
    expected_ids = {
        "A8-new-admitted-host",
        "A8-gross-focal-margin-damage",
        "A8-placebo-remains-unavailable",
        "A8-graft-cells-fail-exact-target",
        "A8-exact-five-cell-literal-tuples",
        "A8-nf4-fresh-anomaly",
        "A8-bf16-literal-and-repeat-stability",
        "A8-D-full-KV-sign-split",
        "A8-E3-scalar-reading",
        "A8-repeat2-governs-stability",
        "A8-formal-and-claim-boundaries-remain-closed",
        "A8-technical-failure-conditional",
        "A8-no-value-driven-grid-extension",
    }
    assert frozen["all_expectations_explicitly_addressed"] is True
    assert frozen["expectation_count"] == len(expected_ids)
    assert {row["id"] for row in frozen["evaluations"]} == expected_ids

    boundary = result["inference_boundary"]
    assert boundary["thresholds_added"] is False
    assert boundary["ratios_computed"] is False
    assert boundary["p_values_computed"] is False
    assert boundary["confidence_intervals_computed"] is False
    assert boundary["quantization_dependence_claim_authorized"] is False
    assert boundary["semantic_transfer_claim_authorized"] is False


def test_synthetic_scalar_subtraction_has_no_tolerance_or_ratio() -> None:
    left = {name: float(index + 1) for index, name in enumerate(MODULE.SCALAR_NAMES)}
    right = {name: float(index) for index, name in enumerate(MODULE.SCALAR_NAMES)}
    assert MODULE._subtract(left, right) == {
        name: 1.0 for name in MODULE.SCALAR_NAMES
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p01, p02: p01.__setitem__("schema", "wrong"),
         "P01 schema/protocol differs"),
        (lambda p01, p02: p01.__setitem__("semantic_evidence_eligible", True),
         "P01 formal boundary differs"),
        (lambda p01, p02: p01["required_bf16_repeat2_disposition"].__setitem__(
            "treatment_present", True), "P01 bfloat16 repeat-2 partial boundary differs"),
        (lambda p01, p02: p02["outer_pod_receipt"].__setitem__("status", "FAIL"),
         "outer receipt did not establish hard eligibility"),
        (lambda p01, p02: p02.__setitem__("repeat2_rider_status", "NOT_AUTHORIZED"),
         "MATCHED_COMPLETE authorization differs"),
        (lambda p01, p02: p02["matched_runtime_gate"].__setitem__(
            "same_host", False), "P02 matched-runtime declaration differs"),
        (lambda p01, p02: p02["repeat_stability_and_interpretation_guard"][
            "within_regime_repeat_stability"
        ]["nf4"].__setitem__("differing_field_count", 1),
         "P02 nf4 repeat stability differs"),
        (lambda p01, p02: p02.__setitem__("runner_terminal_status", "ERROR"),
         "P02 terminal/primary boundary differs"),
    ],
)
def test_hard_boundaries_fail_closed(mutation, message: str) -> None:
    p01, p02, p01_binding, p02_binding = _documents()
    p01 = copy.deepcopy(p01)
    p02 = copy.deepcopy(p02)
    mutation(p01, p02)
    with pytest.raises(MODULE.ComparisonError, match=message):
        _compare(p01, p02, p01_binding, p02_binding)


def test_extra_schema_field_and_malformed_artifact_sha_fail_closed() -> None:
    p01, p02, p01_binding, p02_binding = _documents()
    p01["unexpected"] = "field"
    with pytest.raises(MODULE.ComparisonError, match="P01 root keys differ"):
        _compare(p01, p02, p01_binding, p02_binding)

    p01, p02, p01_binding, p02_binding = _documents()
    p02_binding["sha256"] = "0" * 63
    with pytest.raises(MODULE.ComparisonError, match="lowercase SHA-256"):
        _compare(p01, p02, p01_binding, p02_binding)


def test_generation_payload_tampering_fails_its_canonical_sha() -> None:
    p01, p02, p01_binding, p02_binding = _documents()
    p02 = copy.deepcopy(p02)
    record = p02["outcomes"]["nf4:e01:r1"]["analysis"][
        "E2_literal_generated_behavior"
    ]["focal"]["cells"]["FF"]
    record["decoded_content"] = "tampered"
    with pytest.raises(MODULE.ComparisonError, match="canonical generation SHA differs"):
        _compare(p01, p02, p01_binding, p02_binding)


def test_cli_writes_exclusively_and_records_input_hashes(tmp_path: Path) -> None:
    output = tmp_path / "comparison.json"
    assert MODULE.main([
        "--p01", str(P01), "--p02", str(P02), "--output", str(output)
    ]) == 0
    result = json.loads(output.read_text())
    assert result["comparison_status"] == "PASS"
    assert result["p01"]["input_artifact"]["path"] == str(P01.resolve())
    assert result["p02"]["input_artifact"]["path"] == str(P02.resolve())

    with pytest.raises(FileExistsError):
        MODULE.main([
            "--p01", str(P01), "--p02", str(P02), "--output", str(output)
        ])
