from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_precision_probe_p01_partial.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_precision_probe_p01_partial", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def committed_run_dir() -> Path:
    rows = sorted(path for path in (ROOT / "results/precision_probe_p01").glob(
        "precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_*"
    ) if path.is_dir())
    assert len(rows) == 1
    return rows[0]


@pytest.fixture(scope="module")
def committed_analysis() -> dict:
    return MODULE.analyze_run(committed_run_dir())


def test_committed_partial_run_is_fully_verified(
    committed_analysis: dict,
) -> None:
    result = committed_analysis
    assert result["schema"] == MODULE.ANALYSIS_SCHEMA
    assert result["analysis_status"] == MODULE.ANALYSIS_STATUS
    assert result["post_run_analysis"] is True
    assert result["terminal_partial_run"] is True
    assert result["runner_terminal_status"] in MODULE.TERMINAL_NONPASS
    assert result["formal_v12_decision_eligible"] is False
    assert result["v12_reentry_authorized"] is False
    assert result["complete_outcome_keys"] == [
        ["bf16", "e01", 1],
        ["nf4", "e01", 1],
        ["nf4", "e01", 2],
    ]
    assert result["technical_gates"]["status"] == "PASS"
    assert result["technical_gates"]["same_host"] is True
    assert result["matched_runtime_gate"]["status"] == "PASS"
    assert result["independently_reconstructed_matched_runtime"][
        "status"] == "PASS"
    assert result["source_bindings"]["status"] == "PASS"
    assert result["extension_decision"]["decision"] == "BASE_E01_ONLY"

    verification = result["package_and_derived_verification"]
    assert set(verification) == {
        "bf16:e01:r1", "nf4:e01:r1", "nf4:e01:r2"
    }
    assert all(row == {
        "raw_package": "VERIFIED",
        "arm_grid": "31_PRIMARY_PLUS_3_PLACEBO_ATTEMPTS_VERIFIED",
        "compact_consistency": "PASS",
        "render_consistency": "PASS",
    } for row in verification.values())


def test_requires_bf16_repeat2_absent_or_incomplete(
    committed_analysis: dict,
) -> None:
    disposition = committed_analysis[
        "required_bf16_repeat2_disposition"]
    assert disposition["key"] == ["bf16", "e01", 2]
    assert disposition["status"] != "COMPLETE"
    assert disposition["compact_present"] is False
    assert disposition["phase_a_present"] is False
    assert disposition["treatment_present"] is False
    assert disposition["render_consistency"] in {"PASS", "ABSENT"}

    with pytest.raises(MODULE.AnalysisError,
                       match="complete-outcome set differs"):
        MODULE.validate_partial_layout(
            MODULE.EXPECTED_COMPLETE_KEYS |
            {MODULE.EXPECTED_INCOMPLETE_KEY},
            [],
        )
    with pytest.raises(MODULE.AnalysisError,
                       match="incomplete-outcome set differs"):
        MODULE.validate_partial_layout(
            set(MODULE.EXPECTED_COMPLETE_KEYS),
            [("nf4", "e02", 1)],
        )


def test_matched_r1_is_continuous_and_generation_descriptive(
    committed_analysis: dict,
) -> None:
    matched = committed_analysis["matched_repeat1_descriptive"]
    assert set(matched["nf4"]) == set(matched["bf16"]) == set(
        matched["nf4_minus_bf16"])
    assert all(math.isfinite(value)
               for group in ("nf4", "bf16", "nf4_minus_bf16")
               for value in matched[group].values())
    for probe, row in matched["generation_concordance"].items():
        assert probe in {"focal", "nonfocal"}
        assert set(row["literal_cells_equal"]) == set(
            MODULE.GENERATION_CELLS
        )
        assert set(row["nf4_change_from_fresh"]) == {
            "FC", "FW", "CC", "WW"
        }
        assert set(row["bf16_change_from_fresh"]) == {
            "FC", "FW", "CC", "WW"
        }
        assert isinstance(row["change_vectors_equal"], bool)


def test_no_inferential_upgrade_is_computed(committed_analysis: dict) -> None:
    inference = committed_analysis["inference"]
    assert inference["p_values_computed"] is False
    assert inference["confidence_intervals_computed"] is False
    assert inference["equivalence_test_computed"] is False
    assert inference["formal_release_decision_computed"] is False
    assert inference["population_interaction_claim_authorized"] is False
    assert inference[
        "small_cross_runtime_contrast_interpretation_permitted"] is False
    assert "post-run" in inference["note"].lower()
    assert "descriptive" in inference["note"].lower()


def test_compact_and_render_are_recomputed_from_raw(tmp_path: Path) -> None:
    run_dir = committed_run_dir()
    manifest = MODULE.strict.load_object(next(run_dir.glob(
        "precision-probe-p01-run-manifest_*.json")))
    receipt = manifest["regimes"]["nf4"]["outcomes"][0]
    package_dir = MODULE._resolve_run_path(
        run_dir, receipt["raw_package"]["path"], "test package")
    _, raw = MODULE.strict.reconstruct_package(package_dir)
    original_compact = MODULE._resolve_run_path(
        run_dir, receipt["compact"]["path"], "test compact")
    original_render = MODULE._resolve_run_path(
        run_dir, receipt["renders"]["path"], "test render")
    compact = tmp_path / "compact.json"
    render = tmp_path / "render.md"
    shutil.copyfile(original_compact, compact)
    shutil.copyfile(original_render, render)

    MODULE.verify_outcome_derived_content(
        raw, receipt["raw_package"], compact, render)

    tampered = json.loads(compact.read_text())
    tampered["status"] = "TAMPERED"
    compact.write_text(json.dumps(tampered) + "\n")
    with pytest.raises(MODULE.AnalysisError,
                       match="compact content differs"):
        MODULE.verify_outcome_derived_content(
            raw, receipt["raw_package"], compact, render)


def test_source_bindings_fail_closed_on_byte_mismatch() -> None:
    run_dir = committed_run_dir()
    manifest = MODULE.strict.load_object(next(run_dir.glob(
        "precision-probe-p01-run-manifest_*.json")))
    bindings = {name: dict(row)
                for name, row in manifest["bindings"].items()}
    bindings["case_e01"]["sha256"] = "0" * 64
    with pytest.raises(MODULE.AnalysisError,
                       match="no longer matches bytes: case_e01"):
        MODULE._validate_source_bindings(bindings)
