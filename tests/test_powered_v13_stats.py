import math
import random

import pytest

from powered_v13_stats import (
    CLIP_UPPER,
    DELTA_CLIP,
    FINAL_N,
    STRATA,
    V13StatsError,
    alpha_ledger,
    analyze_primary,
    collapse_render_rows,
    hoeffding_ucb,
    nominal_stratified_t_ucb,
    responder_prevalence_ucb,
    select_final_sample,
)


def render_rows(per_stratum=6, full_offset=0.0, value_offset=0.0):
    rows = []
    for s_index, stratum in enumerate(STRATA):
        for rank in range(1, per_stratum + 1):
            case_id = f"{stratum}-c{rank}"
            base = s_index * 0.01 + rank * 0.001
            for render_id, render_delta in (("r1", -0.01), ("r2", 0.01)):
                rows.append({
                    "case_id": case_id,
                    "stratum": stratum,
                    "eligible_rank": rank,
                    "render_id": render_id,
                    "render_origin": "C",
                    "full_kv": full_offset + base + render_delta,
                    "value_only": value_offset - base + render_delta,
                })
    return rows


def constant_rows(full=0.0, value=0.0, per_stratum=6):
    rows = render_rows(per_stratum=per_stratum)
    for row in rows:
        row["full_kv"] = full
        row["value_only"] = value
    return rows


def test_alpha_ledger_is_exact_and_has_three_events():
    ledger = alpha_ledger()
    assert ledger["spent"] == pytest.approx(0.05, abs=1e-15)
    assert [row["alpha"] for row in ledger["events"]] == [0.02, 0.02, 0.01]


def test_two_c_origin_renders_collapse_and_do_not_increase_n():
    fixtures = collapse_render_rows(render_rows())
    assert len(fixtures) == FINAL_N
    assert len(select_final_sample(fixtures)) == FINAL_N
    assert all(len(fixture.render_values) == 2 for fixture in fixtures)


def test_wrong_origin_duplicate_or_missing_render_fails_closed():
    rows = render_rows()
    rows[1]["render_origin"] = "W"
    with pytest.raises(V13StatsError, match="C-origin"):
        collapse_render_rows(rows)
    rows = render_rows()
    rows[1]["render_id"] = "r1"
    with pytest.raises(V13StatsError, match="render IDs"):
        collapse_render_rows(rows)
    with pytest.raises(V13StatsError, match="expected two"):
        collapse_render_rows(render_rows()[:-1])


def test_missing_rank_and_duplicate_case_fail_closed():
    fixtures = collapse_render_rows(render_rows())
    missing = [fixture for fixture in fixtures
               if not (fixture.stratum == "s1" and fixture.eligible_rank == 2)]
    with pytest.raises(V13StatsError, match="lacks frozen ranks"):
        select_final_sample(missing)
    with pytest.raises(V13StatsError, match="duplicate fixture ID"):
        select_final_sample(fixtures + [fixtures[0]])


def test_overshoot_is_masked_and_row_order_is_irrelevant():
    fixtures = collapse_render_rows(render_rows(per_stratum=8))
    first = analyze_primary(fixtures)
    shuffled = fixtures[:]
    random.Random(7).shuffle(shuffled)
    second = analyze_primary(shuffled)
    assert first.case_ids == second.case_ids
    assert first.clipped_primary_ucb == pytest.approx(
        second.clipped_primary_ucb, abs=1e-15)
    assert all("c7" not in case_id and "c8" not in case_id
               for case_id in first.case_ids)


def test_all_zero_values_retain_positive_radius_and_tail_bound():
    result = analyze_primary(collapse_render_rows(constant_rows()))
    expected_radius = math.sqrt(math.log(1.0 / 0.02) / (2.0 * FINAL_N))
    assert result.cells["full_kv"].ucb == pytest.approx(expected_radius)
    assert result.cells["value_only"].ucb == pytest.approx(expected_radius)
    assert result.responder_count == 0
    assert result.responder_ucb == pytest.approx(1.0 - 0.01 ** (1 / FINAL_N))
    assert result.joint_resolved


def test_raw_outliers_are_clipped_and_flagged():
    rows = constant_rows()
    for row in rows[:2]:
        row["full_kv"] = 5.0
    result = analyze_primary(collapse_render_rows(rows))
    assert result.cells["full_kv"].clipped_count_high == 1
    assert result.responder_count == 1
    assert result.responder_ucb > 0.10
    assert not result.joint_resolved


def test_nonfinite_value_fails_closed():
    rows = render_rows()
    rows[0]["full_kv"] = float("nan")
    with pytest.raises(V13StatsError, match="nonfinite"):
        collapse_render_rows(rows)


def test_responder_zero_bound_and_nonzero_hoeffding_branch():
    assert responder_prevalence_ucb(0, 48, 0.01) == pytest.approx(
        1.0 - 0.01 ** (1.0 / 48))
    assert responder_prevalence_ucb(1, 48, 0.01) > 0.20


def test_nominal_zero_variance_returns_no_interval():
    fixtures = collapse_render_rows(constant_rows(full=0.1, value=0.2))
    result = nominal_stratified_t_ucb(fixtures, cell="full_kv")
    assert result.estimate == pytest.approx(0.1)
    assert result.ucb is None


def test_hoeffding_is_bounded_and_rejects_out_of_range():
    value = hoeffding_ucb([0.0] * 48, lower=-0.5, upper=0.5, alpha=0.02)
    assert 0.0 < value <= CLIP_UPPER
    assert value < DELTA_CLIP
    with pytest.raises(V13StatsError, match="outside"):
        hoeffding_ucb([2.0], lower=-0.5, upper=0.5, alpha=0.02)
