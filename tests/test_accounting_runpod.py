from __future__ import annotations

import hashlib
from dataclasses import replace
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext

import pytest

from accounting.core import AccountingRole, MeasurementClass
from accounting.runpod import (
    BalanceAdjustment,
    BalanceObservation,
    BalanceWindow,
    EvidenceBinding,
    LocalPodEpisode,
    ProviderPodTotal,
    ProviderSnapshot,
    RatePoint,
    RunPodAccountingError,
    RunPodAmbiguityError,
    assert_exact_runpod_reconciliation,
    build_runpod_ledger,
    local_episode_from_state_snapshot,
    precision_probe_episode_from_records,
    provider_snapshot_from_sanitized,
    reconstruct_local_rate_cost,
)


def _binding(name: str, character: str = "a") -> EvidenceBinding:
    return EvidenceBinding(
        name, hashlib.sha256(f"{name}:{character}".encode()).hexdigest()
    )


def _episode(
    pod_id: str = "pod-project",
    *,
    episode_id: str = "episode-project",
    start: str = "2026-07-12T00:00:00Z",
    terminal: str | None = "2026-07-12T01:00:00Z",
    observed: str | None = None,
    rates: tuple[tuple[str, str], ...] = (("2026-07-12T00:00:00Z", "1"),),
) -> LocalPodEpisode:
    source = _binding(f"local:{episode_id}")
    return LocalPodEpisode(
        episode_id=episode_id,
        protocol_id=episode_id,
        pod_id=pod_id,
        started_at_utc=start,
        terminal_at_utc=terminal,
        observed_through_utc=observed,
        rate_points=tuple(
            RatePoint(at, Decimal(rate), source) for at, rate in rates
        ),
        bindings=(source,),
        terminal_binding=source if terminal is not None else None,
        observed_through_binding=source if observed is not None else None,
        owns_full_pod_lifetime=True,
    )


def _snapshot(
    capture: str,
    pods: tuple[tuple[str, str, int], ...],
    *,
    suffix: str,
    complete: bool = True,
    active: tuple[str, ...] = (),
) -> ProviderSnapshot:
    totals = tuple(
        ProviderPodTotal(pod_id, Decimal(amount), billed_ms)
        for pod_id, amount, billed_ms in pods
    )
    return ProviderSnapshot(
        captured_at_utc=capture,
        pagination_complete=complete,
        active_inventory_complete=True,
        active_pod_ids=active,
        pod_totals=totals,
        account_total_usd=(
            sum((item.amount_usd for item in totals), Decimal("0"))
            if complete
            else None
        ),
        account_billed_time_ms=(
            sum(item.billed_time_ms for item in totals) if complete else None
        ),
        binding=_binding(f"provider:{suffix}", suffix[0]),
    )


def _balance(
    at: str, amount: str, name: str, *, active: tuple[str, ...] = ()
) -> BalanceObservation:
    return BalanceObservation(
        observed_at_utc=at,
        balance_usd=Decimal(amount),
        active_inventory_complete=True,
        active_pod_ids=active,
        binding=_binding(f"balance:{name}", name[0]),
    )


def test_duplicate_snapshot_cannot_fake_settlement_or_double_count() -> None:
    episode = _episode()
    one = _snapshot(
        "2026-07-12T02:00:00Z",
        ((episode.pod_id, "1", 3_600_000),),
        suffix="a1",
    )
    ledger = build_runpod_ledger(
        snapshots=(one, one), project_episodes=(episode,)
    )
    assert len(ledger.snapshot_ids) == 1
    assert ledger.project_additive_known_usd == Decimal("0")
    assert ledger.episodes[0].provider_settled is False
    assert (
        ledger.episodes[0].provider_crosscheck_row.measurement_class
        is MeasurementClass.LOWER_BOUND
    )
    assert (
        ledger.episodes[0].provider_crosscheck_row.accounting_role
        is AccountingRole.CROSSCHECK_NONADDITIVE
    )
    assert ledger.project_rows[0].measurement_class is MeasurementClass.UNKNOWN

    two = _snapshot(
        "2026-07-12T03:00:00Z",
        ((episode.pod_id, "1", 3_600_000),),
        suffix="b2",
    )
    settled = build_runpod_ledger(
        snapshots=(one, one, two), project_episodes=(episode,)
    )
    assert len(settled.snapshot_ids) == 2
    assert settled.project_additive_known_usd == Decimal("1")
    assert settled.episodes[0].provider_settled is True
    assert (
        settled.project_rows[0].measurement_class
        is MeasurementClass.EXACT_SOURCE_RECORD
    )


def test_missing_terminal_is_lower_bound_and_missing_all_boundaries_is_unknown() -> None:
    open_episode = _episode(
        terminal=None,
        observed="2026-07-12T00:30:00Z",
    )
    ledger = build_runpod_ledger(
        snapshots=(), project_episodes=(open_episode,)
    )
    assert ledger.project_additive_known_usd == Decimal("0")
    assert (
        ledger.episodes[0].rate_crosscheck_row.measurement_class
        is MeasurementClass.LOWER_BOUND
    )
    assert ledger.episodes[0].rate_crosscheck_row.quantity == Decimal("0.5")
    assert ledger.project_rows[0].measurement_class is MeasurementClass.UNKNOWN

    no_boundary = replace(
        open_episode,
        observed_through_utc=None,
        observed_through_binding=None,
    )
    unknown = build_runpod_ledger(
        snapshots=(), project_episodes=(no_boundary,)
    )
    assert unknown.project_rows[0].measurement_class is MeasurementClass.UNKNOWN
    assert unknown.project_rows[0].quantity is None
    assert unknown.project_additive_known_usd == Decimal("0")


def test_changing_rate_is_integrated_piecewise_with_decimal_only() -> None:
    episode = _episode(
        rates=(
            ("2026-07-12T00:00:00Z", "1"),
            ("2026-07-12T00:30:00Z", "2"),
        )
    )
    estimate = reconstruct_local_rate_cost(episode)
    assert estimate.amount_usd == Decimal("1.5")
    assert estimate.measurement_class is MeasurementClass.RECONSTRUCTED

    with pytest.raises(RunPodAccountingError, match="must be Decimal"):
        replace(episode.rate_points[0], hourly_rate_usd=1.0)  # type: ignore[arg-type]


def test_overlapping_distinct_pod_rates_stay_separate_and_same_pod_is_ambiguous() -> None:
    first = _episode(pod_id="pod-a", episode_id="episode-a")
    second = _episode(
        pod_id="pod-b",
        episode_id="episode-b",
        start="2026-07-12T00:30:00Z",
        terminal="2026-07-12T01:30:00Z",
        rates=(("2026-07-12T00:30:00Z", "2"),),
    )
    provider_a = _snapshot(
        "2026-07-12T02:00:00Z",
        (("pod-a", "1", 3_600_000), ("pod-b", "2", 3_600_000)),
        suffix="c-overlap",
    )
    provider_b = _snapshot(
        "2026-07-12T03:00:00Z",
        (("pod-a", "1", 3_600_000), ("pod-b", "2", 3_600_000)),
        suffix="d-overlap",
    )
    balance = BalanceWindow(
        "overlapping-pods-one-account-window",
        _balance("2026-07-11T23:59:00Z", "10", "e-overlap-start"),
        _balance("2026-07-12T03:01:00Z", "7", "f-overlap-end"),
        ("pod-a", "pod-b"),
        True,
        _binding("exclusive:overlapping-pods", "a"),
    )
    ledger = build_runpod_ledger(
        snapshots=(provider_a, provider_b),
        project_episodes=(first, second),
        balance_windows=(balance,),
    )
    assert ledger.project_additive_known_usd == Decimal("3")
    assert len(ledger.project_rows) == 2
    assert sum(
        (item.rate_crosscheck_row.quantity for item in ledger.episodes),
        Decimal("0"),
    ) == Decimal("3")
    assert all(
        item.rate_crosscheck_row.accounting_role
        is AccountingRole.CROSSCHECK_NONADDITIVE
        for item in ledger.episodes
    )
    assert ledger.exact_reconciliation is True

    duplicate_pod = replace(second, pod_id=first.pod_id)
    with pytest.raises(RunPodAmbiguityError, match="multiple project episodes"):
        build_runpod_ledger(
            snapshots=(), project_episodes=(first, duplicate_pod)
        )


def test_balance_topup_is_adjusted_but_always_nonadditive() -> None:
    episode = _episode()
    adjustment = BalanceAdjustment(
        adjustment_id="topup-1",
        occurred_at_utc="2026-07-12T00:30:00Z",
        balance_effect_usd=Decimal("5"),
        kind="top_up_or_credit",
        binding=_binding("cash:topup", "c"),
    )
    window = BalanceWindow(
        window_id="with-topup",
        start=_balance("2026-07-11T23:59:00Z", "10", "d-start"),
        end=_balance("2026-07-12T01:01:00Z", "14", "e-end"),
        covered_pod_ids=(episode.pod_id,),
        exclusive_provider_activity=True,
        exclusivity_binding=_binding("balance:topup-exclusive", "f"),
        adjustments=(adjustment,),
    )
    ledger = build_runpod_ledger(
        snapshots=(), project_episodes=(episode,), balance_windows=(window,)
    )
    result = ledger.balance_reconciliations[0]
    assert result.raw_balance_decrease_usd == Decimal("-4")
    assert result.adjusted_consumption_usd == Decimal("1")
    assert result.row.accounting_role is AccountingRole.CROSSCHECK_NONADDITIVE
    assert ledger.project_additive_known_usd == Decimal("0")


def test_account_lifetime_activity_is_not_project_spend() -> None:
    episode = _episode()
    snapshot = _snapshot(
        "2026-07-12T02:00:00Z",
        (
            (episode.pod_id, "1", 3_600_000),
            ("pod-unrelated", "7", 25_200_000),
        ),
        suffix="f3",
    )
    ledger = build_runpod_ledger(
        snapshots=(snapshot,), project_episodes=(episode,)
    )
    assert ledger.project_additive_known_usd == Decimal("0")
    assert ledger.episodes[0].provider_crosscheck_row.quantity == Decimal("1")
    assert ledger.account_lifetime_observed_usd == Decimal("8")
    assert ledger.unmatched_provider_pod_ids == ("pod-unrelated",)
    assert all(
        row.accounting_role is AccountingRole.CONTEXT_NONADDITIVE
        for row in ledger.account_lifetime_rows
    )


def test_exact_provider_rate_balance_reconciliation_gate() -> None:
    episode = _episode()
    first = _snapshot(
        "2026-07-12T02:00:00Z",
        ((episode.pod_id, "1", 3_600_000),),
        suffix="g4",
    )
    second = _snapshot(
        "2026-07-12T03:00:00Z",
        ((episode.pod_id, "1", 3_600_000),),
        suffix="h5",
    )
    window = BalanceWindow(
        window_id="exact",
        start=_balance("2026-07-11T23:59:00Z", "10", "i-start"),
        end=_balance("2026-07-12T03:01:00Z", "9", "j-end"),
        covered_pod_ids=(episode.pod_id,),
        exclusive_provider_activity=True,
        exclusivity_binding=_binding("balance:exact-exclusive", "k"),
    )
    ledger = build_runpod_ledger(
        snapshots=(first, second),
        project_episodes=(episode,),
        balance_windows=(window,),
    )
    assert ledger.exact_reconciliation is True
    assert ledger.episodes[0].provider_minus_rate_usd == Decimal("0")
    assert ledger.balance_reconciliations[0].adjusted_minus_primary_usd == Decimal("0")
    assert_exact_runpod_reconciliation(ledger)

    changed = replace(second, pod_totals=(ProviderPodTotal(
        episode.pod_id, Decimal("1.01"), 3_600_000
    ),), account_total_usd=Decimal("1.01"))
    disagreement = build_runpod_ledger(
        snapshots=(first, changed),
        project_episodes=(episode,),
        balance_windows=(window,),
    )
    with pytest.raises(RunPodAccountingError, match="exact reconciliation failed"):
        assert_exact_runpod_reconciliation(disagreement)


def test_precision_probe_p01_and_p02_safe_evidence_extractors() -> None:
    p01 = {
        "schema": "precision_probe_p01_provider_budget_v1",
        "protocol_id": "precision-probe-p01",
        "provider_pod_id": "p01-pod",
        "provider_clock_started_epoch": 1_000,
        "provider_elapsed_seconds_at_job_record": 7_024,
        "hourly_cost_usd": Decimal("1.39"),
        "ignored_raw_field": {"must_not": "survive"},
    }
    p01_episode = precision_probe_episode_from_records(
        p01, budget_binding=_binding("p01:provider", "k")
    )
    p01_estimate = reconstruct_local_rate_cost(p01_episode)
    assert p01_estimate.measurement_class is MeasurementClass.LOWER_BOUND
    with localcontext(Context(prec=100, rounding=ROUND_HALF_EVEN)):
        p01_expected = (
            Decimal(7_024) * Decimal("1.39") / Decimal(3_600)
        ).quantize(Decimal("0.000000000000001"), rounding=ROUND_HALF_EVEN)
    assert p01_estimate.amount_usd == p01_expected
    assert "ignored_raw_field" not in str(p01_episode.to_dict())

    p02 = {
        "schema": "precision_probe_p02_provider_budget_v1",
        "protocol_id": "precision-probe-p02",
        "provider_pod_id": "p02-pod",
        "provider_clock_started_epoch": 10_000,
        "provider_elapsed_seconds_at_job_record": 4_242,
        "hourly_cost_usd": Decimal("1.39"),
    }
    settlement = {
        "schema": "precision_probe_p02_provider_settlement_v1",
        "protocol_id": "precision-probe-p02",
        "provider_pod_id": "p02-pod",
        "provider_clock_started_epoch": 10_000,
        "delete_returned_epoch": 14_268,
        "hourly_cost_usd": Decimal("1.39"),
    }
    p02_episode = precision_probe_episode_from_records(
        p02,
        budget_binding=_binding("p02:provider", "l"),
        settlement=settlement,
        settlement_binding=_binding("p02:settlement", "m"),
    )
    p02_estimate = reconstruct_local_rate_cost(p02_episode)
    assert p02_estimate.measurement_class is MeasurementClass.RECONSTRUCTED
    with localcontext(Context(prec=100, rounding=ROUND_HALF_EVEN)):
        p02_expected = (
            Decimal(4_268) * Decimal("1.39") / Decimal(3_600)
        ).quantize(Decimal("0.000000000000001"), rounding=ROUND_HALF_EVEN)
    assert p02_estimate.amount_usd == p02_expected

    with localcontext() as changed_process_context:
        changed_process_context.prec = 7
        assert reconstruct_local_rate_cost(p02_episode).amount_usd == p02_expected


def test_safe_snapshot_parser_rejects_payload_fields_and_binary_float() -> None:
    safe = {
        "schema": "runpod_sanitized_provider_snapshot_v1",
        "captured_at_utc": "2026-07-12T00:00:00Z",
        "pagination_complete": True,
        "active_inventory_complete": True,
        "active_pod_ids": [],
        "pod_totals": [
            {
                "pod_id": "pod-a",
                "amount_usd": "1.25",
                "billed_time_ms": 3_600_000,
                "gpu_type_id": "A100",
            }
        ],
        "account_total_usd": "1.25",
        "account_billed_time_ms": 3_600_000,
    }
    parsed = provider_snapshot_from_sanitized(
        safe, binding=_binding("provider:safe", "n")
    )
    assert parsed.pod_totals[0].amount_usd == Decimal("1.25")
    assert "provider:safe" in parsed.evidence_ref

    with pytest.raises(RunPodAccountingError, match="unexpected fields"):
        provider_snapshot_from_sanitized(
            {**safe, "raw_response": {"credential": "forbidden"}},
            binding=_binding("provider:unsafe", "o"),
        )
    floated = {**safe, "account_total_usd": 1.25}
    with pytest.raises(RunPodAccountingError, match="bool or float"):
        provider_snapshot_from_sanitized(
            floated, binding=_binding("provider:float", "p")
        )


def test_local_state_sanitizer_retains_only_allowlisted_evidence() -> None:
    state = {
        "id": "pod-local",
        "createdAt": "2026-07-12 10:29:07.365 +0000 UTC",
        "costPerHr": Decimal("1.39"),
        "consumerUserId": "must-disappear",
        "env": {"SECRET": "must-disappear"},
        "publicIp": "must-disappear",
    }
    episode = local_episode_from_state_snapshot(
        state, binding=_binding("local:sanitized", "1")
    )
    serialized = str(episode.to_dict())
    assert episode.started_at_utc == "2026-07-12T10:29:07.365000Z"
    assert "must-disappear" not in serialized


def test_fail_closed_on_provider_regression_and_dropped_lifetime_row() -> None:
    first = _snapshot(
        "2026-07-12T02:00:00Z", (("pod-a", "2", 2_000),), suffix="q6"
    )
    decreased = _snapshot(
        "2026-07-12T03:00:00Z", (("pod-a", "1", 2_000),), suffix="r7"
    )
    with pytest.raises(RunPodAmbiguityError, match="amount decreased"):
        build_runpod_ledger(snapshots=(first, decreased), project_episodes=())

    dropped = _snapshot("2026-07-12T03:00:00Z", (), suffix="s8")
    with pytest.raises(RunPodAmbiguityError, match="dropped prior"):
        build_runpod_ledger(snapshots=(first, dropped), project_episodes=())

    partial = _snapshot(
        "2026-07-12T01:00:00Z",
        (("pod-partial", "0.5", 1_000),),
        suffix="t9",
        complete=False,
    )
    later_complete = _snapshot("2026-07-12T04:00:00Z", (), suffix="a0")
    with pytest.raises(RunPodAmbiguityError, match="pod-partial"):
        build_runpod_ledger(
            snapshots=(partial, later_complete), project_episodes=()
        )


def test_balance_windows_cannot_overlap_or_reuse_adjustment_evidence() -> None:
    shared_start = _balance("2026-07-12T00:00:00Z", "10", "u-start")
    first_end = _balance("2026-07-12T01:00:00Z", "9", "v-end")
    second_end = _balance("2026-07-12T02:00:00Z", "8", "w-end")
    first = BalanceWindow(
        window_id="overlap-a",
        start=shared_start,
        end=first_end,
        covered_pod_ids=(),
        exclusive_provider_activity=True,
        exclusivity_binding=_binding("exclusive:overlap-a", "b"),
    )
    second = BalanceWindow(
        window_id="overlap-b",
        start=shared_start,
        end=second_end,
        covered_pod_ids=(),
        exclusive_provider_activity=True,
        exclusivity_binding=_binding("exclusive:overlap-b", "c"),
    )
    with pytest.raises(RunPodAmbiguityError, match="balance windows overlap"):
        build_runpod_ledger(
            snapshots=(), project_episodes=(), balance_windows=(first, second)
        )

    middle = _balance("2026-07-12T01:00:00Z", "9", "x-middle")
    shared_adjustment_binding = _binding("transactions.json:7", "d")
    adjustment_a = BalanceAdjustment(
        "adjust-a",
        "2026-07-12T00:30:00Z",
        Decimal("1"),
        "credit",
        shared_adjustment_binding,
    )
    adjustment_b = BalanceAdjustment(
        "adjust-b",
        "2026-07-12T01:30:00Z",
        Decimal("1"),
        "credit",
        shared_adjustment_binding,
    )
    disjoint_a = BalanceWindow(
        "disjoint-a",
        shared_start,
        middle,
        (),
        True,
        _binding("exclusive:disjoint-a", "e"),
        (adjustment_a,),
    )
    disjoint_b = BalanceWindow(
        "disjoint-b",
        middle,
        second_end,
        (),
        True,
        _binding("exclusive:disjoint-b", "f"),
        (adjustment_b,),
    )
    with pytest.raises(RunPodAmbiguityError, match="binding reused"):
        build_runpod_ledger(
            snapshots=(),
            project_episodes=(),
            balance_windows=(disjoint_a, disjoint_b),
        )


def test_exclusive_balance_window_requires_bound_consistent_inventory() -> None:
    start = _balance("2026-07-12T00:00:00Z", "10", "g-start")
    end = _balance("2026-07-12T01:00:00Z", "9", "h-end")
    with pytest.raises(RunPodAccountingError, match="exclusivity attestation"):
        BalanceWindow("unbound", start, end, (), True)

    active_start = _balance(
        "2026-07-12T00:00:00Z", "10", "i-active", active=("other-pod",)
    )
    with pytest.raises(RunPodAmbiguityError, match="contradicts"):
        BalanceWindow(
            "contradicted",
            active_start,
            end,
            (),
            True,
            _binding("exclusive:contradicted", "j"),
        )

    incomplete_start = replace(start, active_inventory_complete=False)
    with pytest.raises(RunPodAmbiguityError, match="contradicts"):
        BalanceWindow(
            "incomplete",
            incomplete_start,
            end,
            (),
            True,
            _binding("exclusive:incomplete", "k"),
        )


def test_local_boundaries_must_be_bound_to_frozen_evidence() -> None:
    state = {
        "id": "pod-boundary",
        "createdAt": "2026-07-12 00:00:00 +0000 UTC",
        "costPerHr": Decimal("1"),
    }
    source = _binding("local:boundary-state", "l")
    with pytest.raises(RunPodAccountingError, match="terminal_binding"):
        local_episode_from_state_snapshot(
            state,
            binding=source,
            terminal_at_utc="2026-07-12T01:00:00Z",
        )
    with pytest.raises(RunPodAccountingError, match="observed-through"):
        local_episode_from_state_snapshot(
            state,
            binding=source,
            observed_through_utc="2026-07-12T00:30:00Z",
        )


def test_provider_ids_are_strict_strings_and_predating_rows_fail() -> None:
    safe = {
        "schema": "runpod_sanitized_provider_snapshot_v1",
        "captured_at_utc": "2026-07-12T00:00:00Z",
        "pagination_complete": True,
        "active_inventory_complete": True,
        "active_pod_ids": [],
        "pod_totals": [
            {
                "pod_id": 7,
                "amount_usd": "1",
                "billed_time_ms": 1,
                "gpu_type_id": None,
            }
        ],
        "account_total_usd": "1",
        "account_billed_time_ms": 1,
    }
    with pytest.raises(RunPodAccountingError, match="pod_id"):
        provider_snapshot_from_sanitized(
            safe, binding=_binding("provider:numeric-id", "m")
        )

    numeric_active = {**safe, "pod_totals": [], "account_total_usd": "0"}
    numeric_active["account_billed_time_ms"] = 0
    numeric_active["active_pod_ids"] = [7]
    with pytest.raises(RunPodAccountingError, match="active_pod_ids"):
        provider_snapshot_from_sanitized(
            numeric_active, binding=_binding("provider:numeric-active", "n")
        )

    episode = _episode(
        pod_id="pod-old",
        episode_id="episode-new",
        start="2026-07-12T01:00:00Z",
        terminal="2026-07-12T02:00:00Z",
        rates=(("2026-07-12T01:00:00Z", "1"),),
    )
    predating = _snapshot(
        "2026-07-12T00:30:00Z",
        (("pod-old", "0.1", 1_000),),
        suffix="b1",
    )
    with pytest.raises(RunPodAmbiguityError, match="predates"):
        build_runpod_ledger(
            snapshots=(predating,), project_episodes=(episode,)
        )
