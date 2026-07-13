import json
from pathlib import Path

from local_n48_phase_a import _begins, _selected_paths


REPO = Path(__file__).resolve().parents[1]
ROSTER = REPO / "data/coherent_state_local_n48_v2/phase-a-roster-v1.json"


def test_frozen_roster_selection_skips_only_review_failure_at_rank_one():
    roster = json.loads(ROSTER.read_text())
    paths = _selected_paths(roster, None, 1)
    assert len(paths) == 7
    assert all("rank-01_" in path.name for path in paths)
    assert not any("categorical_set_membership" in str(path) for path in paths)


def test_fixture_override_is_literal():
    roster = json.loads(ROSTER.read_text())
    fixture = Path("data/example.json")
    assert _selected_paths(roster, fixture, 10) == [fixture]


def test_greedy_prefix_matching_uses_exact_content_ids():
    score = {
        "target": {"content_token_ids": [10, 11]},
        "countertarget": {"content_token_ids": [20]},
        "greedy_content_token_ids": [10, 11, 99],
    }
    assert _begins(score, "target")
    assert not _begins(score, "countertarget")
