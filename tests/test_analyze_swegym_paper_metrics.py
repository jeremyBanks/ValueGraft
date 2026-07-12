from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_swegym_paper_metrics.py"
SPEC = importlib.util.spec_from_file_location("analyze_swegym_paper_metrics", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_percentile_bootstrap_is_seeded_and_counts_signs() -> None:
    first = MODULE.percentile_bootstrap([-1.0, 0.0, 2.0], n_reps=250, seed=7)
    second = MODULE.percentile_bootstrap([-1.0, 0.0, 2.0], n_reps=250, seed=7)
    assert first == second
    assert first["n"] == 3
    assert first["mean"] == pytest.approx(1.0 / 3.0)
    assert (first["n_negative"], first["n_zero"], first["n_positive"]) == (1, 1, 1)


def test_committed_reanalysis_counts_and_point_estimates() -> None:
    report = MODULE.build_report(
        REPO_ROOT,
        generated_at_utc="2026-07-12T00:00:00Z",
        n_reps=100,
        seed=0,
    )

    checks = report["invariant_checks"]
    assert checks["map_fitting_n"] == 41
    assert checks["fresh_evaluation_n"] == 57
    assert checks["partial_original_confirmation_n"] == 45
    assert checks["fit_fresh_overlap_n"] == 0
    assert checks["fit_confirmation_overlap_n"] == 0
    assert checks["unique_fixed_scalar_n"] == 173

    selected = report["selected_map_out_of_fitting"]
    assert selected["fresh57"]["continuous"]["selected_map_minus_baseline"][
        "mean"
    ] == pytest.approx(0.011742353249699225)
    assert selected["original45_partial_confirmation"]["continuous"][
        "selected_map_minus_baseline"
    ]["mean"] == pytest.approx(0.01583593553405697)
    assert selected["pooled102"]["continuous"]["selected_map_minus_baseline"][
        "mean"
    ] == pytest.approx(0.013548345433974703)

    structural = selected["pooled102"]["structural_match_vs_baseline"]
    assert structural["n"] == 102
    assert structural["reference_matches"] == 53
    assert structural["selected_map_matches"] == 53
    assert structural["selected_map_fixes"] == 3
    assert structural["selected_map_breaks"] == 3

    scalar = report["fixed_scalar_alpha_0_75"]
    assert scalar["legacy_single_call_original75"]["mean"] == pytest.approx(
        0.015643370373565186
    )
    assert scalar["chunked_original75"]["mean"] == pytest.approx(
        0.013331335522045138
    )
    assert scalar["chunked_disjoint98"]["mean"] == pytest.approx(
        -0.0016888082090843917
    )
    assert scalar["chunked_unique_pooled173"]["mean"] == pytest.approx(
        0.004822814795740549
    )
    assert scalar["chunked_pool_heterogeneity"]["mean_difference"] == pytest.approx(
        -0.01502014373112953
    )
    assert scalar["paired_schedule_apparatus_difference"]["continuous"][
        "mean"
    ] == pytest.approx(-0.002312034851520049)


def test_output_inventory_names_every_consumed_score_file() -> None:
    report = MODULE.build_report(
        REPO_ROOT,
        generated_at_utc="2026-07-12T00:00:00Z",
        n_reps=10,
        seed=0,
    )
    directories = {
        item["role"]: item
        for item in report["inputs"]
        if "directory" in item
    }
    assert directories["original_run_1"]["score_file_count"] == 75
    assert directories["original_run_1"]["manifest_present"] is False
    assert directories["original_run_2"]["score_file_count"] == 75
    assert directories["disjoint_profile"]["score_file_count"] == 98
    assert directories["disjoint_champion_eval"]["score_file_count"] == 98
    assert directories["original_partial_confirmation"]["score_file_count"] == 45
    for item in directories.values():
        assert len(item["file_set_sha256"]) == 64
        assert all(len(file["sha256"]) == 64 for file in item["files"])


def test_report_declares_numeric_dataset_order() -> None:
    report = MODULE.build_report(
        REPO_ROOT,
        generated_at_utc="2026-07-12T00:00:00Z",
        n_reps=10,
        seed=0,
    )
    assert "sorted numerically by saved dataset idx" in report[
        "statistical_contract"
    ]["unit"]


def test_preserved_shared_system_prompt_reconstructs_exactly() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "swegym_paper_reanalysis"
        / "swegym-shared-system-prompt_provenance_20260712T054055Z.json"
    )
    artifact = json.loads(path.read_text())
    content = "\n".join(artifact["lines"])
    assert len(content) == artifact["content_length_characters"] == 4758
    expected_sha256 = "1120aa8819abb372428afb82f6a5f49d1d243e4bf58cb27fd481809acd339e84"
    assert artifact["content_sha256_utf8"] == expected_sha256
    assert hashlib.sha256(content.encode()).hexdigest() == expected_sha256
    assert artifact["source"]["rows_scanned"] == 491
    assert artifact["source"]["all_rows_share_exact_system_prompt"] is True
