import json
from argparse import Namespace
from pathlib import Path

import pytest

from local_n48_phase_a import (
    _begins,
    _contains_sequence,
    _nonfocal_valid,
    _score_finite_complete,
    _selected_paths,
    _validate_frozen_inputs,
    CARRIER_BANK_DEFAULT,
    MODEL_DEFAULT,
)


REPO = Path(__file__).resolve().parents[1]
ROSTER = REPO / "data/coherent_state_local_n48_v3/phase-a-roster-v1.json"


def test_v3_rank_one_selection_excludes_development_and_review_failures():
    roster = json.loads(ROSTER.read_text())
    paths = _selected_paths(roster, None, 1)
    assert len(paths) == 6
    assert all("rank-01_" in path.name for path in paths)
    assert not any("threshold_eligibility" in str(path) for path in paths)
    assert not any("categorical_set_membership" in str(path) for path in paths)


def test_v3_rank_two_selection_uses_next_frozen_candidates():
    roster = json.loads(ROSTER.read_text())
    paths = _selected_paths(roster, None, 2)
    assert len(paths) == 14
    assert any("threshold_eligibility/rank-02_" in str(path) for path in paths)
    assert any("categorical_set_membership/rank-02_" in str(path)
               for path in paths)


def test_frozen_input_contract_accepts_only_exact_bound_inputs():
    roster = json.loads(ROSTER.read_text())
    carrier_bank = json.loads(CARRIER_BANK_DEFAULT.read_text())
    args = Namespace(
        model=MODEL_DEFAULT,
        roster=ROSTER,
        carrier_bank=CARRIER_BANK_DEFAULT,
    )
    _validate_frozen_inputs(args, roster, carrier_bank)
    args.model = "some-other-model"
    with pytest.raises(ValueError, match="model argument"):
        _validate_frozen_inputs(args, roster, carrier_bank)


def test_fixture_override_is_limited_to_permitted_hash_bound_roster_record():
    roster = json.loads(ROSTER.read_text())
    permitted = Path(next(
        row["fixture_path"] for row in roster["records"]
        if row["stratum_id"] == "threshold_eligibility"
        and row["permutation_rank"] == 2))
    assert _selected_paths(roster, permitted, 10) == [permitted]

    excluded = Path(next(
        row["fixture_path"] for row in roster["records"]
        if row["stratum_id"] == "threshold_eligibility"
        and row["permutation_rank"] == 1))
    with pytest.raises(ValueError, match="not one permitted"):
        _selected_paths(roster, excluded, 10)
    with pytest.raises(ValueError, match="not one permitted"):
        _selected_paths(roster, Path("data/example.json"), 10)


def test_greedy_prefix_matching_uses_exact_content_ids():
    score = {
        "target": {"content_token_ids": [10, 11]},
        "countertarget": {"content_token_ids": [20]},
        "greedy_content_token_ids": [10, 11, 99],
    }
    assert _begins(score, "target")
    assert not _begins(score, "countertarget")


def test_exact_token_subsequence_and_nonfocal_rule():
    assert _contains_sequence([1, 2, 3, 4], [2, 3])
    assert not _contains_sequence([1, 2, 3, 4], [2, 4])
    score = {
        "target": {"content_token_ids": [2, 3]},
        "countertarget": {"content_token_ids": [8]},
        "greedy_content_token_ids": [1, 2, 3, 4],
        "margin": 1.0,
    }
    assert _nonfocal_valid(score) == (True, True, True)


def test_finite_complete_score_gate():
    score = {
        "probe": "Probe?",
        "target": {
            "text": "Violet",
            "content_token_ids": [1, 2],
            "token_logprobs": [-1.0, -2.0],
            "mean_logprob": -1.5,
        },
        "countertarget": {
            "text": "Comet",
            "content_token_ids": [3],
            "token_logprobs": [-4.0],
            "mean_logprob": -4.0,
        },
        "margin": 2.5,
        "greedy_token_ids": [1, 2, 4],
        "greedy_content_token_ids": [1, 2],
        "greedy_cap": 16,
        "greedy_stop_reason": "eos",
    }
    assert _score_finite_complete(score)
    score["margin"] = float("nan")
    assert not _score_finite_complete(score)
