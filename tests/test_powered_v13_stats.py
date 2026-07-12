import math
import random

import pytest

from powered_v13_stats import (
    CELLS,
    STRATA,
    V13StatsError,
    alpha_ledger,
    analyze_look,
    clopper_pearson_upper,
    collapse_render_rows,
    hoeffding_ucb,
    select_balanced_look,
    sequential_decision,
)


def render_rows(per_stratum=6, full_offset=0.0, value_offset=0.0):
    rows = []
    for s_index, stratum in enumerate(STRATA):
        for rank in range(1, per_stratum + 1):
            case_id = f"{stratum}-c{rank}"
            base = s_index * 0.01 + rank * 0.001
            for origin, render_delta in (("C", -0.01), ("W", 0.01)):
                rows.append({
                    "case_id": case_id,
                    "stratum": stratum,
                    "eligible_rank": rank,
                    "render_origin": origin,
                    "full_kv": full_offset + base + render_delta,
                    "value_only": value_offset - base + render_delta,
                })
    return rows


def test_alpha_ledger_is_exact_and_has_six_events():
    ledger = alpha_ledger()
    assert ledger["spent"] == pytest.approx(0.05, abs=1e-15)
    assert len(ledger["events"]) == 6
    assert {row["cell"] for row in ledger["events"]} == set(CELLS)


def test_two_renders_collapse_and_do_not_increase_n():
    fixtures = collapse_render_rows(render_rows(per_stratum=3))
    assert len(fixtures) == 24
    selected = select_balanced_look(fixtures, 24)
    assert len(selected) == 24
    assert all(len(fixture.render_values) == 2 for fixture in selected)


def test_duplicate_or_missing_render_fails_closed():
    rows = render_rows(per_stratum=3)
    rows[1]["render_origin"] = "C"
    with pytest.raises(V13StatsError, match="origins"):
        collapse_render_rows(rows)
    rows = render_rows(per_stratum=3)[:-1]
    with pytest.raises(V13StatsError, match="expected two"):
        collapse_render_rows(rows)


def test_missing_rank_and_duplicate_case_fail_closed():
    fixtures = collapse_render_rows(render_rows(per_stratum=3))
    missing = [fixture for fixture in fixtures
               if not (fixture.stratum == "s1" and fixture.eligible_rank == 2)]
    with pytest.raises(V13StatsError, match="lacks frozen ranks"):
        select_balanced_look(missing, 24)
    duplicated = fixtures + [fixtures[0]]
    with pytest.raises(V13StatsError, match="duplicate fixture ID"):
        select_balanced_look(duplicated, 24)


def test_overshoot_is_masked_and_row_order_is_irrelevant():
    fixtures = collapse_render_rows(render_rows(per_stratum=6))
    first = analyze_look(fixtures, 24)
    shuffled = fixtures[:]
    random.Random(7).shuffle(shuffled)
    second = analyze_look(shuffled, 24)
    assert first.case_ids == second.case_ids
    assert first.primary_ucb == pytest.approx(second.primary_ucb, abs=1e-15)
    assert all("c4" not in case_id and "c5" not in case_id and "c6" not in case_id
               for case_id in first.case_ids)


def test_zero_variance_ucb_equals_mean():
    rows = render_rows(per_stratum=6, full_offset=0.1, value_offset=0.2)
    # Remove both within-stratum rank and render variation.
    for row in rows:
        row["full_kv"] = 0.1
        row["value_only"] = 0.2
    result = analyze_look(collapse_render_rows(rows), 24)
    assert result.cells["full_kv"].ucb == pytest.approx(0.1)
    assert math.isinf(result.cells["full_kv"].degrees_of_freedom)
    assert result.cells["value_only"].ucb == pytest.approx(0.2)
    assert result.stop_for_bound


def test_nonfinite_value_fails_closed():
    rows = render_rows(per_stratum=3)
    rows[0]["full_kv"] = float("nan")
    with pytest.raises(V13StatsError, match="nonfinite"):
        collapse_render_rows(rows)


def test_sequential_stops_at_first_resolved_look():
    rows = render_rows(per_stratum=6)
    for row in rows:
        row["full_kv"] = 0.0
        row["value_only"] = 0.0
    selected, looks = sequential_decision(collapse_render_rows(rows))
    assert selected.n_total == 24
    assert len(looks) == 1


def test_one_sided_zero_success_bounds_match_formula():
    for n, alpha in ((12, 0.005), (18, 0.010), (24, 0.035)):
        expected = 1.0 - alpha ** (1.0 / n)
        assert clopper_pearson_upper(0, n, alpha) == pytest.approx(expected)


def test_hoeffding_is_bounded_and_rejects_out_of_range():
    value = hoeffding_ucb([0.0] * 24, lower=-1.0, upper=1.0, alpha=0.05)
    assert 0.0 < value <= 1.0
    with pytest.raises(V13StatsError, match="outside"):
        hoeffding_ucb([2.0], lower=-1.0, upper=1.0, alpha=0.05)

