import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import audit_graft_dose_recovery as audit  # noqa: E402


@pytest.fixture(scope="session")
def tokenizer():
    return audit._load_tokenizer(local_files_only=True)


@pytest.fixture(scope="session")
def tracked_paths():
    return audit._git_tracked_paths(REPO)


def test_distribution_uses_linear_interpolated_quartiles():
    observed = audit.distribution([1, 2, 3, 4])
    assert observed == {
        "n": 4,
        "min": 1.0,
        "q25": 1.75,
        "median": 2.5,
        "mean": 2.5,
        "q75": 3.25,
        "max": 4.0,
        "sum": 10.0,
    }


def test_legacy_exact_reconstruction_from_tracked_checkpoints(
    tokenizer, tracked_paths
):
    result = audit.audit_legacy(REPO, tokenizer, tracked_paths)

    assert result["status"] == "EXACT_RECONSTRUCTION_PASS"
    assert result["repository_inputs_committed"] is True
    assert result["inputs"]["all_git_tracked"] is True
    assert result["cross_checks"][
        "all_reconstructed_totals_equal_persisted_n_pairs"
    ] is True
    assert result["cross_checks"][
        "all_regions_one_full_contiguous_difflib_block"
    ] is True
    assert result["cross_checks"][
        "all_eligible_non_special_source_rows_aligned"
    ] is True

    total = result["distributions"]["aligned_positions_total"]
    assert total["n"] == 18
    assert total["min"] == 1091
    assert total["median"] == 1252.5
    assert total["mean"] == pytest.approx(1549.611111111111)
    assert total["max"] == 2787
    assert total["sum"] == 27893

    assert result["pooled"]["aligned_positions_summary"] == 14842
    assert result["pooled"]["aligned_positions_tail"] == 13051
    assert result["pooled"]["summary_share_pct"] == pytest.approx(
        53.21048291686086
    )
    assert result["pooled"]["tail_share_pct"] == pytest.approx(
        46.78951708313914
    )
    assert result["boilerplate_assessment"][
        "near_empty_intervention_disproved"
    ] is True

    slots = result["variant_tensor_slot_coverage"]
    assert slots["per_head"]["active_layer_kv_head_slots"] == 109
    assert slots["per_layer"]["active_layer_kv_head_slots"] == 108
    assert slots["intersection"]["active_layer_kv_head_slots"] == 74
    assert slots["union"]["active_layer_kv_head_slots"] == 143


def test_swe_missing_parquet_is_explicitly_non_self_contained(
    tmp_path, tracked_paths
):
    missing = tmp_path / "swegym.parquet"
    result = audit.audit_swe(REPO, None, tracked_paths, missing)

    assert result["status"] == "UNAVAILABLE_IN_CLEAN_CLONE"
    assert result["repository_inputs_committed"] is False
    assert result["summary_alignment_recoverable"] is False
    assert "generated per-trajectory summary text" in result[
        "missing_committed_fields"
    ]


@pytest.mark.skipif(
    not (REPO / "swegym.parquet").is_file(),
    reason="ignored local parquet is intentionally absent from a clean clone",
)
def test_swe_hash_matched_local_tail_reconstruction(tokenizer, tracked_paths):
    result = audit.audit_swe(
        REPO, tokenizer, tracked_paths, REPO / "swegym.parquet"
    )

    assert result["status"] == "PARTIAL_LOCAL_RECONSTRUCTION_PASS"
    assert result["repository_inputs_committed"] is False
    assert result["clean_clone_reproducibility"] is False
    assert result["parquet"]["sha256_observed"] == audit.SWE_PARQUET_SHA256
    assert result["parquet"]["git_tracked"] is False
    assert result["cross_checks"][
        "all_saved_token_counts_and_cut_metadata_match"
    ] is True
    assert result["cross_checks"][
        "all_tails_one_full_contiguous_difflib_block"
    ] is True
    assert result["cross_checks"][
        "all_tail_eligible_non_special_source_rows_aligned"
    ] is True

    evaluation = result["samples"]["fresh_eval_57"]
    assert evaluation["n_trajectories"] == 57
    assert evaluation["tail_aligned_positions"]["min"] == 137
    assert evaluation["tail_aligned_positions"]["median"] == 1466
    assert evaluation["tail_aligned_positions"]["mean"] == pytest.approx(
        1640.6491228070176
    )
    assert evaluation["tail_aligned_positions"]["max"] == 8543
    assert evaluation["pooled"]["tail_aligned_positions"] == 93517
    assert evaluation["pooled"]["generated_summary_source_width"] == 6634
    assert evaluation["pooled"][
        "tail_share_of_aligned_positions_lower_bound_pct"
    ] == pytest.approx(93.3760022366227)

    confirmation = result["samples"]["confirmation_45"]
    assert confirmation["pooled"][
        "tail_share_of_aligned_positions_lower_bound_pct"
    ] == pytest.approx(93.91627428758369)
    assert result["summary_alignment_recoverable"] is False
    assert result["boilerplate_assessment"][
        "near_empty_tail_intervention_disproved"
    ] is True
    assert result["mediation_status"]["summary_only_ablation_run"] is False
    assert result["mediation_status"]["tail_only_ablation_run"] is False
