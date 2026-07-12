from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from powered_v13_cycle import (
    CycleObservation,
    V13CycleError,
    build_checkpoint,
    write_checkpoint_exclusive,
)


def observation(**changes: object) -> CycleObservation:
    base = CycleObservation(
        checkpoint_utc="2026-07-12T17:00:00Z",
        goal_elapsed_seconds=7200,
        provider_observed_utc="2026-07-12T16:59:00Z",
        provider_balance_usd="43.50",
        provider_spend_limit_usd="80.00",
        active_pod_ids=(),
        phase="static_freeze",
        status="in_progress",
        phase_spent_usd="11.00",
        phase_cap_usd="12.00",
        projected_remaining_phase_a_usd="1.00",
        projected_core_usd="30.00",
        projected_audit_usd="4.50",
        reserve_usd="8.00",
        independent_n=1,
        target_n=48,
        independent_unit_ids=("e01",),
        git_commit="a" * 40,
        git_clean=True,
        completed_gates=("static-design",),
        pending_gates=(),
    )
    return replace(base, **changes)


def test_exact_gap_and_budget_inequalities_pass() -> None:
    result = build_checkpoint(observation())
    assert result["derived"]["semantic_gap_n"] == 47
    assert result["derived"]["semantic_progress"] == "1/48"
    assert result["inequalities"]["phase_cap"]["lhs_usd"] == "12.00"
    assert result["inequalities"]["phase_cap"]["pass"]
    assert result["inequalities"]["future_balance"]["lhs_usd"] == "43.50"
    assert result["inequalities"]["future_balance"]["pass"]
    assert result["assessment"]["status"] == "PASS"


def test_stale_provider_observation_holds_progression() -> None:
    result = build_checkpoint(observation(
        provider_observed_utc="2026-07-12T16:54:59Z"))
    assert result["derived"]["provider_observation_age_seconds"] == 301
    assert not result["derived"]["provider_observation_fresh"]
    assert result["assessment"]["status"] == "HOLD"
    assert "STALE_PROVIDER_OBSERVATION" in result["assessment"]["hold_reasons"]


def test_current_balance_excludes_already_spent_phase_dollars() -> None:
    result = build_checkpoint(observation())
    future = result["inequalities"]["future_balance"]
    assert future["excluded_already_spent_term"] == "phase_spent_usd"
    assert "phase_spent_usd" not in future["included_terms"]
    # Adding the already-spent $11 would fail against $43.50.  The frozen
    # current-balance comparison correctly passes at exactly $43.50.
    assert future["lhs_usd"] == "43.50"
    assert future["pass"]


def test_phase_cap_failure_is_a_hold() -> None:
    result = build_checkpoint(observation(phase_spent_usd="11.01"))
    assert result["inequalities"]["phase_cap"]["lhs_usd"] == "12.01"
    assert not result["inequalities"]["phase_cap"]["pass"]
    assert "PHASE_CAP_FAILED" in result["assessment"]["hold_reasons"]


def test_future_balance_failure_is_a_hold() -> None:
    result = build_checkpoint(observation(provider_balance_usd="43.49"))
    assert not result["inequalities"]["future_balance"]["pass"]
    assert "FUTURE_BALANCE_FAILED" in result["assessment"]["hold_reasons"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"independent_n": 2}, "differs from the unique semantic-unit IDs"),
        ({"independent_n": 49, "independent_unit_ids": tuple(
            f"case-{index}" for index in range(49))}, "cannot exceed"),
        ({"independent_unit_ids": ("e01", "e01"), "independent_n": 2},
         "contains duplicates"),
        ({"independent_n": True}, "must be an integer"),
    ],
)
def test_independent_n_is_bound_to_unique_semantic_units(
        changes: dict[str, object], message: str) -> None:
    with pytest.raises(V13CycleError, match=message):
        build_checkpoint(observation(**changes))


def test_clean_and_dirty_git_states_are_bound_literally() -> None:
    clean = build_checkpoint(observation())
    dirty = build_checkpoint(observation(git_clean=False))
    assert clean["observation"]["git_clean"] is True
    assert clean["assessment"]["status"] == "PASS"
    assert dirty["observation"]["git_clean"] is False
    assert dirty["assessment"]["status"] == "HOLD"
    assert "DIRTY_GIT_TREE" in dirty["assessment"]["hold_reasons"]


def test_pending_gates_are_progress_and_fail_closed() -> None:
    result = build_checkpoint(observation(
        completed_gates=("static-design",),
        pending_gates=("fixture-review", "release-tests")))
    assert result["derived"]["completed_gate_count"] == 1
    assert result["derived"]["pending_gate_count"] == 2
    assert not result["derived"]["named_gates_complete"]
    assert "PENDING_GATES" in result["assessment"]["hold_reasons"]


def test_exclusive_output_preserves_first_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "cycle.json"
    first = build_checkpoint(observation())
    write_checkpoint_exclusive(path, first)
    with pytest.raises(FileExistsError):
        write_checkpoint_exclusive(path, build_checkpoint(observation(
            checkpoint_utc="2026-07-12T17:01:00Z")))
    assert json.loads(path.read_text()) == first


def test_money_must_not_arrive_as_binary_float() -> None:
    with pytest.raises(V13CycleError, match="decimal string"):
        build_checkpoint(observation(provider_balance_usd=43.5))

