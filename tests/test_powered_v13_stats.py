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
    behavioral_recovery_counts,
    collapse_render_rows,
    descriptive_cell_summary,
    fieller_ratio_upper,
    hoeffding_ucb,
    nominal_ordinary_t_ucb,
    nominal_stratified_t_ucb,
    pooled_within_fixture_render_variance,
    responder_prevalence_ucb,
    select_final_sample,
    stratified_cluster_bootstrap,
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
    ordinary = nominal_ordinary_t_ucb(fixtures, cell="full_kv")
    assert ordinary.estimate == pytest.approx(0.1)
    assert ordinary.ucb is None


def test_hoeffding_is_bounded_and_rejects_out_of_range():
    value = hoeffding_ucb([0.0] * 48, lower=-0.5, upper=0.5, alpha=0.02)
    assert 0.0 < value <= CLIP_UPPER
    assert value < DELTA_CLIP
    with pytest.raises(V13StatsError, match="outside"):
        hoeffding_ucb([2.0], lower=-0.5, upper=0.5, alpha=0.02)


def test_stratified_cluster_bootstrap_is_seeded_and_keeps_fixture_n():
    fixtures = collapse_render_rows(render_rows())
    first = stratified_cluster_bootstrap(fixtures, replicates=2_000)
    second = stratified_cluster_bootstrap(fixtures, replicates=2_000)
    assert first == second
    assert first.replicates == 2_000
    assert len(first.index_stream_sha256) == 64
    assert first.cells["full_kv"].replicates == 2_000
    assert first.cells["value_only"].replicates == 2_000
    assert first.cells["full_kv"].estimate == pytest.approx(
        analyze_primary(fixtures).cells["full_kv"].raw_mean)


def test_default_bootstrap_stream_has_frozen_golden_hashes():
    result = stratified_cluster_bootstrap(
        collapse_render_rows(render_rows()))
    assert result.index_stream_sha256 == (
        "8f4482ef33ca2c3c00dc523a674446fbffc045d1ffa330fd80975ab31b80b494")
    assert result.cells["full_kv"].replicate_values_sha256 == (
        "db66c581a2526d06dda2a4ebe83102d015969fb650e142cb10a99862e91bf5f8")
    assert result.cells["value_only"].replicate_values_sha256 == (
        "c37947872fb35e47ea78de53b63cc8ca438c1a69cbc79e2527d05f1ffce3996a")
    assert result.cells["full_kv"].ucb == pytest.approx(
        0.03889583333333334)
    assert result.cells["value_only"].ucb == pytest.approx(
        -0.03810416666666667)


def test_bootstrap_constant_values_and_bad_configuration():
    fixtures = collapse_render_rows(constant_rows(full=0.125, value=-0.25))
    result = stratified_cluster_bootstrap(fixtures, replicates=64)
    assert result.cells["full_kv"].ucb == 0.125
    assert result.cells["value_only"].ucb == -0.25
    with pytest.raises(V13StatsError, match="replicate"):
        stratified_cluster_bootstrap(fixtures, replicates=0)
    with pytest.raises(V13StatsError, match="seed"):
        stratified_cluster_bootstrap(fixtures, seed=True)


def test_pooled_two_render_variance_uses_within_fixture_sample_variance():
    fixtures = collapse_render_rows(render_rows())
    full = pooled_within_fixture_render_variance(fixtures, cell="full_kv")
    value = pooled_within_fixture_render_variance(fixtures, cell="value_only")
    assert full.fixture_count == FINAL_N
    assert full.pooled_variance == pytest.approx(0.0002)
    assert value.pooled_variance == pytest.approx(0.0002)
    assert full.pooled_standard_deviation == pytest.approx(math.sqrt(0.0002))


def test_descriptive_summary_freezes_strata_signs_and_fixture_order():
    fixtures = collapse_render_rows(render_rows())
    summary = descriptive_cell_summary(fixtures, cell="full_kv")
    assert summary.mean == pytest.approx(0.0385)
    assert summary.positive_count == FINAL_N
    assert summary.zero_count == 0
    assert summary.negative_count == 0
    assert len(summary.stratum_means) == len(STRATA)
    assert len(summary.leave_one_stratum_out_means) == len(STRATA)
    assert [row[0] for row in summary.fixture_values[:2]] == [
        "s1-c1", "s1-c2"]


def test_behavioral_endpoint_requires_both_renders_and_fixed_denominator():
    fixtures = collapse_render_rows(render_rows())
    rows = []
    for fixture in fixtures:
        for render_id in ("r1", "r2"):
            rows.append({
                "case_id": fixture.case_id,
                "render_id": render_id,
                "render_origin": "C",
                "a_c_begins_target": True,
                "ff_begins_target": False,
                "cc_begins_target": fixture.case_id != "s1-c1",
                "fc_begins_target": (
                    fixture.case_id == "s1-c1" and render_id == "r1"),
                "a_c_valid_generation": True,
                "ff_valid_generation": True,
                "cc_valid_generation": True,
                "fc_valid_generation": True,
            })
    result = behavioral_recovery_counts(fixtures, rows)
    assert result.denominator == FINAL_N
    assert result.full_kv_both_render_count == FINAL_N - 1
    assert result.value_only_both_render_count == 0
    assert result.either_cell_both_render_count == FINAL_N - 1

    rows[0]["ff_begins_target"] = True
    with pytest.raises(V13StatsError, match="FF eligibility"):
        behavioral_recovery_counts(fixtures, rows)
    rows[0]["ff_begins_target"] = False
    rows[0]["cc_valid_generation"] = False
    with pytest.raises(V13StatsError, match="generation is invalid"):
        behavioral_recovery_counts(fixtures, rows)


def test_fieller_bounded_and_unbounded_denominator_support():
    numerators = [0.1 + index / 10_000 for index in range(FINAL_N)]
    bounded = fieller_ratio_upper(numerators, [5.0] * FINAL_N)
    assert bounded.bounded
    assert math.isfinite(bounded.upper)
    assert bounded.upper >= bounded.ratio_estimate

    alternating = [-1.0 if index % 2 else 1.0 for index in range(FINAL_N)]
    unbounded = fieller_ratio_upper(numerators, alternating)
    assert not unbounded.bounded
    assert unbounded.upper == math.inf
    with pytest.raises(V13StatsError, match="exactly 48"):
        fieller_ratio_upper(numerators[:-1], [5.0] * (FINAL_N - 1))
