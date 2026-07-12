from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_precision_probe_p01.py"
SPEC = importlib.util.spec_from_file_location("analyze_precision_probe_p01", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PHASE_RAW = ROOT / (
    "results/coherent_canary_v12_phase_a/"
    "coherent-canary-v12-phase-a-e01_exact-subject_20260712T030639073087Z.json"
)
TREATMENT_PACKAGE = ROOT / (
    "results/coherent_canary_v12_treatment_package/"
    "coherent-canary-v12-treatment-e01_exact-subject_20260712T040237879370Z"
)


def historical_combined_outcome() -> dict:
    phase_run = MODULE.load_object(PHASE_RAW)
    _, treatment_run = MODULE.reconstruct_package(TREATMENT_PACKAGE)
    return {
        "protocol_id": MODULE.PROTOCOL_ID,
        "formal_v12_decision_eligible": False,
        "v12_reentry_authorized": False,
        "component_reuse_does_not_inherit_v12_eligibility": True,
        "regime": "bf16",
        "case_id": "e01",
        "repeat_index": 1,
        "phase_a": phase_run["phase_a"],
        "treatment": treatment_run["treatment"],
    }


def test_reconstructs_historical_lossless_package() -> None:
    manifest, raw = MODULE.reconstruct_package(TREATMENT_PACKAGE)
    assert manifest["original"]["sha256"] == (
        "f6622d978c80d8f4f874c208ef9f6d9546ab3f1bf7676655c9a0b4662718d082"
    )
    assert raw["case_id"] == "e01"
    assert raw["treatment"]["primary_arm_count"] == 31


def test_recovers_known_e01_metrics_without_runner_summaries() -> None:
    result = MODULE.analyze_outcome(historical_combined_outcome())
    primary = result["N_regional_estimands"]["R2_boundary"]["value_only"]
    full_kv = result["N_regional_estimands"]["R2_boundary"]["full_KV"]

    assert result["primary_D_value_N_R2"] == pytest.approx(0.1745452881)
    assert result["primary_nonfocal_D_value_N_R2"] == pytest.approx(-0.0000696182)
    assert primary["SEL_absolute_nonfocal_penalty_historical"] == pytest.approx(
        0.1744756699
    )
    assert primary["Hplus"] == pytest.approx(0.7102603912)
    assert primary["U"] == pytest.approx(0.1816501617)
    assert primary["Uplus"] == pytest.approx(0.9459857941)
    assert full_kv["D_focal"] == pytest.approx(0.0972366333)
    assert result["phase_a_oracle_to_fresh_damage"]["focal"][
        "margin"
    ] == pytest.approx(22.2984294891)
    assert result["available_placebo_control_count"] == 0
    assert len(result["all_primary_generations"]) == 62


def test_signed_and_historical_selectivity_are_not_conflated() -> None:
    result = MODULE.analyze_outcome(historical_combined_outcome())
    row = result["N_regional_estimands"]["R2_boundary"]["value_only"]
    assert row["SEL_signed_p01"] != row[
        "SEL_absolute_nonfocal_penalty_historical"
    ]
    assert row["SEL_signed_p01"] == pytest.approx(
        row["D_focal"] - row["D_nonfocal"]
    )


def test_identification_fails_closed_on_v12_eligibility() -> None:
    raw = historical_combined_outcome()
    raw["formal_v12_decision_eligible"] = True
    with pytest.raises(MODULE.AnalysisError, match="ineligibility"):
        MODULE.identify_outcome(raw)


def test_recursive_diff_reports_json_pointer_paths() -> None:
    left = {"phase_a": {"scores": [{"margin": 1.0}]}, "same": [1, 2]}
    right = {"phase_a": {"scores": [{"margin": 2.0}]}, "same": [1, 2]}
    assert MODULE.recursive_differences(left, right) == [
        "/phase_a/scores/0/margin"
    ]
