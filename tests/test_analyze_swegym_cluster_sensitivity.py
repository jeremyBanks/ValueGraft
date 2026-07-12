from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_swegym_cluster_sensitivity.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_swegym_cluster_sensitivity", SCRIPT
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _synthetic_messages(snapshot: str = "owner__repo__1.2") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "generic tools"},
        {
            "role": "user",
            "content": (
                f"<uploaded_files>\n/workspace/{snapshot}\n</uploaded_files>\n"
                "Consider this task:\n<pr_description>\nFix the exact bug.\n"
                "</pr_description>\nUse the repository."
            ),
        },
        {
            "role": "assistant",
            "content": f"<execute_bash>cd /workspace/{snapshot} && pwd</execute_bash>",
        },
    ]


def test_identity_parser_uses_explicit_snapshot_and_content_hash_only() -> None:
    identity = MODULE.parse_trajectory_identity(7, _synthetic_messages())
    assert identity.owner == "owner"
    assert identity.repository == "repo"
    assert identity.version == "1.2"
    assert identity.snapshot_key == "owner__repo__1.2"
    assert identity.repository_key == "owner/repo"
    expected = hashlib.sha256(
        "owner__repo__1.2\0Fix the exact bug.".encode()
    ).hexdigest()
    assert identity.task_sha256 == expected
    assert identity.later_assistant_repeats_exact_workspace_path is True

    serialized = json.dumps(identity.serialized(), sort_keys=True)
    assert "Fix the exact bug" not in serialized
    assert "<pr_description>" not in serialized
    assert "/workspace/" not in serialized


@pytest.mark.parametrize(
    "snapshot",
    ["owner__repo", "owner__repo__version__extra", "owner____version"],
)
def test_identity_parser_rejects_ambiguous_snapshot_keys(snapshot: str) -> None:
    with pytest.raises(ValueError, match="owner__repo__version"):
        MODULE.parse_trajectory_identity(0, _synthetic_messages(snapshot))


def test_cluster_bootstrap_is_seeded_and_declares_changed_estimand() -> None:
    rows = [
        MODULE.AnalysisRow("x", 0, 0.0, "task-a", "snap-a", "repo-a"),
        MODULE.AnalysisRow("x", 1, 0.0, "task-a", "snap-a", "repo-a"),
        MODULE.AnalysisRow("x", 2, 3.0, "task-b", "snap-b", "repo-b"),
    ]
    first = MODULE.cluster_percentile_bootstrap(
        rows, key="task_sha256", n_reps=250, seed=9, estimand="row_weighted"
    )
    second = MODULE.cluster_percentile_bootstrap(
        rows, key="task_sha256", n_reps=250, seed=9, estimand="row_weighted"
    )
    equal = MODULE.cluster_percentile_bootstrap(
        rows, key="task_sha256", n_reps=250, seed=9, estimand="cluster_equal"
    )
    assert first == second
    assert first["n_rows"] == 3
    assert first["n_clusters"] == 2
    assert first["mean"] == pytest.approx(1.0)
    assert equal["mean"] == pytest.approx(1.5)
    assert first["cluster_size"]["histogram"] == {"1": 1, "2": 1}


@pytest.fixture(scope="module")
def committed_report() -> dict:
    return MODULE.build_report(
        REPO_ROOT,
        generated_at_utc="2026-07-12T00:00:00Z",
        n_reps=10_000,
        seed=0,
    )


def test_real_parser_has_full_validated_coverage(committed_report: dict) -> None:
    validation = committed_report["identity_recovery_validation"]
    assert validation["parser_coverage"] == {
        "rows_total": 491,
        "rows_parsed": 491,
        "fraction": 1.0,
    }
    assert validation["shared_system_prompt"] == {
        "distinct_hashes": 1,
        "sha256": MODULE.EXPECTED_SYSTEM_SHA256,
        "rows": 491,
    }
    assert validation[
        "exact_workspace_path_reappears_in_later_assistant_messages"
    ] == {"rows": 491, "of": 491}
    assert validation["task_partition_matches_exact_initial_user_partition"] is True
    assert validation["pr_description_hash_maps_to_one_snapshot_for_every_task"] is True
    assert validation["all_full_trajectory_transcripts_distinct"] is True
    assert validation["repeated_task_clusters_contain_distinct_rollouts"] is True
    counts = validation["global_cluster_counts"]
    assert (counts["tasks"], counts["versioned_snapshots"], counts["repositories"]) == (
        293,
        87,
        11,
    )
    assert counts["repeated_task_clusters"] == 107
    assert counts["rows_in_repeated_task_clusters"] == 305

    crosscheck = validation["committed_result_gold_action_repository_crosscheck"]
    assert crosscheck["rows_checked"] == 102
    assert crosscheck["rows_with_exact_expected_gold_action_snapshot_key"] == 101
    assert crosscheck["rows_with_unique_exact_expected_gold_action_snapshot_key"] == 101
    assert crosscheck["mismatched_dataset_indices"] == []
    assert crosscheck["missing_dataset_indices"] == [74]


def test_task_overlap_is_recovered_across_trajectory_split(committed_report: dict) -> None:
    overlap = committed_report["selection_and_overlap"]
    assert (overlap["map_fit"]["n_rows"], overlap["map_fit"]["n_tasks"]) == (
        41,
        38,
    )
    assert (overlap["fresh57"]["n_rows"], overlap["fresh57"]["n_tasks"]) == (
        57,
        54,
    )
    assert overlap["fresh57"]["task_clusters_also_in_map_fit"] == 5
    assert overlap["fresh57"]["rows_whose_task_also_in_map_fit"] == 5
    assert overlap["original45_partial_confirmation"]["n_tasks"] == 28
    assert overlap["original45_partial_confirmation"][
        "task_clusters_also_in_map_fit"
    ] == 2
    assert overlap["original45_partial_confirmation"][
        "rows_whose_task_also_in_map_fit"
    ] == 3
    assert overlap["pooled102"]["n_tasks"] == 76
    assert overlap["pooled102"]["task_clusters_also_in_map_fit"] == 7
    assert overlap["pooled102"]["rows_whose_task_also_in_map_fit"] == 8
    assert overlap["fresh_confirmation_task_overlap"]["n_task_clusters"] == 6
    assert overlap["confirmation_is_exact_first45_original_eligible_prefix"] is True


def test_cluster_aware_intervals_and_strict_subset(committed_report: dict) -> None:
    results = committed_report["selected_map_minus_B"]
    pooled = results["pooled102"]
    iid = pooled["iid_trajectory"]
    task = pooled["task_cluster_row_weighted"]
    snapshot = pooled["snapshot_cluster_row_weighted_sensitivity"]
    repository = pooled["repository_cluster_row_weighted_sensitivity"]
    assert iid["mean"] == pytest.approx(0.013548345433974703)
    assert iid["ci_95_percentile"] == pytest.approx(
        {"lower": 0.008334605849432334, "upper": 0.019087182824960088}
    )
    assert task["n_clusters"] == 76
    assert task["ci_95_percentile"] == pytest.approx(
        {"lower": 0.007625219024597577, "upper": 0.019542428199856788}
    )
    assert snapshot["n_clusters"] == 39
    assert snapshot["ci_95_percentile"] == pytest.approx(
        {"lower": 0.008001662451495898, "upper": 0.019690332460503474}
    )
    assert repository["n_clusters"] == 9
    assert repository["ci_95_percentile"] == pytest.approx(
        {"lower": 0.010867766326350559, "upper": 0.018537324736329644}
    )

    strict = results["strict_task_disjoint_from_fit"]["pooled"]
    assert strict["iid_trajectory"]["n_rows"] == 94
    assert strict["task_cluster_row_weighted"]["n_clusters"] == 69
    assert strict["task_cluster_row_weighted"]["mean"] == pytest.approx(
        0.014236501673651463
    )
    assert strict["task_cluster_row_weighted"]["ci_95_percentile"] == pytest.approx(
        {"lower": 0.007920707565206974, "upper": 0.02059803718713701}
    )


def test_report_binds_inputs_and_serializes_no_literal_tasks(
    committed_report: dict,
) -> None:
    parquet = committed_report["inputs"]["parquet"]
    assert parquet["sha256"] == MODULE.EXPECTED_PARQUET_SHA256
    assert parquet["size_bytes"] == 10_379_409
    assert parquet["expected_sha256_verified"] is True
    assert len(committed_report["derived_cluster_mapping"]) == 491

    serialized = json.dumps(committed_report, sort_keys=True)
    assert "/workspace/getmoto__moto__" not in serialized
    assert "/workspace/python__mypy__" not in serialized
    assert "I've uploaded a python code repository" not in serialized
    assert "Fix the exact bug." not in serialized
