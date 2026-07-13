import math
import random

import pytest

from local_n48_stats import (
    CARRIER_IDS,
    CLIP_LOWER,
    CLIP_UPPER,
    FINAL_N,
    STRATA,
    LocalN48StatsError,
    analyze_treatment_records as analyze_records,
    clopper_pearson_upper,
    recompute_hoeffding_stdlib,
)


def analyze_treatment_records(rows):
    expected = sorted({row["candidate_id"] for row in rows
                       if isinstance(row, dict)
                       and isinstance(row.get("candidate_id"), str)})
    return analyze_records(rows, expected_candidate_ids=expected)


def treatment_rows(*, optional=False):
    rows = []
    for stratum_index, stratum in enumerate(STRATA):
        for member in range(1, 7):
            candidate_id = f"{stratum}-c{member}"
            center = (stratum_index * 6 + member) / 1_000
            for carrier_id, carrier_shift in (
                    ("fixed_a", -0.01), ("fixed_b", 0.01)):
                x = center + carrier_shift
                row = {
                    "candidate_id": candidate_id,
                    "stratum": stratum,
                    "carrier_id": carrier_id,
                    "l_c_b": -4.0,
                    "l_w_b": -2.0,
                    "l_c_e_c": -4.0 + x,
                    "l_w_e_c": -3.0,
                    "l_c_e_w": -3.98,
                    "l_w_e_w": -2.5,
                    "l_c_a_c": 1.0,
                    "l_w_a_c": -1.0,
                }
                if optional:
                    row.update({
                        "l_c_vp": -4.01,
                        "l_w_vp": -2.01,
                    })
                rows.append(row)
    return rows


def test_frozen_8x6x2_analysis_collapses_carriers_inside_conversation():
    rows = treatment_rows()
    result = analyze_treatment_records(rows)
    centers = [(stratum_index * 6 + member) / 1_000
               for stratum_index in range(8) for member in range(1, 7)]
    expected_mean = math.fsum(centers) / FINAL_N
    expected_radius = math.sqrt(math.log(25.0) / (2.0 * FINAL_N))

    assert result.n_input_rows == 96
    assert result.n_conversations == FINAL_N
    assert result.independent_unit == "conversation/candidate_id"
    assert result.carrier_ids == CARRIER_IDS
    assert len(result.conversation_effects) == FINAL_N
    assert result.primary_raw.mean == pytest.approx(expected_mean)
    assert result.primary_hoeffding.mean_z == pytest.approx(expected_mean)
    assert result.primary_hoeffding.radius == pytest.approx(
        expected_radius, abs=1e-16)
    assert result.primary_hoeffding.lcb == pytest.approx(
        expected_mean - expected_radius)
    assert result.primary_hoeffding.ucb == pytest.approx(
        expected_mean + expected_radius)
    assert result.primary_raw.per_carrier_means["fixed_a"] == pytest.approx(
        expected_mean - 0.01)
    assert result.primary_raw.per_carrier_means["fixed_b"] == pytest.approx(
        expected_mean + 0.01)
    assert result.raw_over_half.successes == 0
    assert result.raw_over_half.clopper_pearson_ucb == pytest.approx(
        1.0 - 0.01 ** (1.0 / FINAL_N))


def test_raw_sd_and_two_sided_t_interval_are_conversation_level():
    result = analyze_treatment_records(treatment_rows())
    values = [effect.raw_x for effect in result.conversation_effects]
    mean = math.fsum(values) / FINAL_N
    variance = math.fsum((value - mean) ** 2 for value in values) / (
        FINAL_N - 1)
    sd = math.sqrt(variance)
    half_width = result.primary_raw.t_critical * sd / math.sqrt(FINAL_N)
    assert result.primary_raw.n_conversations == FINAL_N
    assert result.primary_raw.sample_sd == pytest.approx(sd)
    assert result.primary_raw.standard_error == pytest.approx(
        sd / math.sqrt(FINAL_N))
    assert result.primary_raw.t_ci_95.confidence == 0.95
    assert result.primary_raw.t_ci_95.lower == pytest.approx(mean - half_width)
    assert result.primary_raw.t_ci_95.upper == pytest.approx(mean + half_width)


def test_clipping_and_raw_tail_count_use_collapsed_conversations():
    rows = treatment_rows()
    candidate = rows[0]["candidate_id"]
    for row in rows:
        if row["candidate_id"] == candidate:
            row["l_c_e_c"] = row["l_c_b"] + 0.6
    result = analyze_treatment_records(rows)
    effect = next(row for row in result.conversation_effects
                  if row.candidate_id == candidate)
    assert effect.raw_x == pytest.approx(0.6)
    assert effect.clipped_z == CLIP_UPPER
    assert result.raw_over_half.successes == 1
    assert 0.0 < result.raw_over_half.clopper_pearson_ucb < 1.0


def test_clipping_occurs_after_fixed_carriers_are_averaged():
    rows = treatment_rows()
    candidate = rows[0]["candidate_id"]
    candidate_rows = [row for row in rows if row["candidate_id"] == candidate]
    candidate_rows[0]["l_c_e_c"] = candidate_rows[0]["l_c_b"] + 2.0
    candidate_rows[1]["l_c_e_c"] = candidate_rows[1]["l_c_b"]
    result = analyze_treatment_records(rows)
    effect = next(row for row in result.conversation_effects
                  if row.candidate_id == candidate)
    assert effect.raw_x == 1.0
    assert effect.clipped_z == 0.5


def test_strict_tail_threshold_and_all_successes_special_case():
    rows = treatment_rows()
    for row in rows:
        row["l_c_e_c"] = row["l_c_b"] + 0.5
    assert analyze_treatment_records(rows).raw_over_half.successes == 0
    for row in rows:
        row["l_c_e_c"] = row["l_c_b"] + 0.500001
    result = analyze_treatment_records(rows)
    assert result.raw_over_half.successes == FINAL_N
    assert result.raw_over_half.clopper_pearson_ucb == 1.0
    assert clopper_pearson_upper(FINAL_N, FINAL_N) == 1.0


def test_mandatory_specificity_margin_damage_and_optional_placebo_companions():
    result = analyze_treatment_records(treatment_rows(optional=True))
    expected = {
        "absolute_l_c_b",
        "absolute_l_w_b",
        "absolute_l_c_e_c",
        "absolute_l_w_e_c",
        "absolute_l_c_e_w",
        "absolute_l_w_e_w",
        "absolute_l_c_a_c",
        "absolute_l_w_a_c",
        "specificity_correct_minus_wrong",
        "wrong_history_movement_from_fresh",
        "correct_target_damage",
        "margin_damage",
        "fresh_margin",
        "correct_graft_margin",
        "wrong_graft_margin",
        "graft_margin_specificity",
        "margin_recovery_from_fresh",
    }
    assert set(result.companion_metrics) == expected
    x_mean = result.primary_raw.mean
    assert result.companion_metrics[
        "specificity_correct_minus_wrong"].mean == pytest.approx(x_mean - 0.02)
    assert result.companion_metrics[
        "wrong_history_movement_from_fresh"].mean == pytest.approx(0.02)
    assert result.companion_metrics[
        "correct_target_damage"].mean == pytest.approx(5.0)
    assert result.companion_metrics["margin_damage"].mean == pytest.approx(4.0)
    assert result.companion_metrics["absolute_l_c_b"].mean == -4.0
    assert result.companion_metrics["absolute_l_w_b"].mean == -2.0
    assert result.companion_metrics["absolute_l_w_e_c"].mean == -3.0
    assert result.companion_metrics["absolute_l_c_e_w"].mean == -3.98
    assert result.companion_metrics["absolute_l_w_e_w"].mean == -2.5
    assert result.companion_metrics["absolute_l_c_a_c"].mean == 1.0
    assert result.companion_metrics["absolute_l_w_a_c"].mean == -1.0
    assert result.companion_metrics["fresh_margin"].mean == -2.0
    assert result.companion_metrics[
        "correct_graft_margin"].mean == pytest.approx(x_mean - 1.0)
    assert result.companion_metrics[
        "wrong_graft_margin"].mean == pytest.approx(-1.48)
    assert result.companion_metrics[
        "graft_margin_specificity"].mean == pytest.approx(x_mean + 0.48)
    assert result.companion_metrics[
        "margin_recovery_from_fresh"].mean == pytest.approx(x_mean + 1.0)
    assert all(summary.n_conversations == FINAL_N
               for summary in result.companion_metrics.values())
    absolute = result.companion_metrics["absolute_l_c_e_c"]
    expected_values = [-4.0 + effect.raw_x
                       for effect in result.conversation_effects]
    assert absolute.mean == pytest.approx(
        math.fsum(expected_values) / FINAL_N)
    assert absolute.sample_sd == pytest.approx(
        math.sqrt(math.fsum(
            (value - absolute.mean) ** 2 for value in expected_values
        ) / (FINAL_N - 1)))
    assert absolute.t_ci_95.lower < absolute.mean < absolute.t_ci_95.upper
    assert set(result.applied_placebo_metrics) == {
        "placebo_movement_from_fresh",
        "correct_minus_placebo",
        "placebo_margin",
        "correct_minus_placebo_margin",
    }
    placebo = result.applied_placebo_metrics["placebo_movement_from_fresh"]
    assert placebo.n_condition_rows_available == 96
    assert placebo.n_condition_rows_missing == 0
    assert placebo.n_complete_conversations == 48
    assert placebo.complete_conversation_mean == pytest.approx(-0.01)
    assert result.applied_placebo_metrics[
        "correct_minus_placebo"].complete_conversation_mean == pytest.approx(
            x_mean + 0.01)


def test_placebo_pair_fails_closed_within_row_but_allows_explicit_missingness():
    rows = treatment_rows(optional=True)
    del rows[0]["l_w_vp"]
    with pytest.raises(LocalN48StatsError, match="incomplete C/W placebo"):
        analyze_treatment_records(rows)

    rows = treatment_rows(optional=True)
    for field in ("l_c_vp", "l_w_vp"):
        del rows[0][field]
        del rows[1][field]
        del rows[2][field]
    result = analyze_treatment_records(rows)
    summary = result.applied_placebo_metrics["placebo_margin"]
    assert summary.n_condition_rows_available == 93
    assert summary.n_condition_rows_missing == 3
    assert summary.n_complete_conversations == 46
    assert summary.n_partial_conversations == 1
    assert summary.n_unavailable_conversations == 1
    assert summary.per_carrier_counts == {"fixed_a": 46, "fixed_b": 47}


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.pop(), "exactly 96 rows"),
    (lambda rows: rows.append(dict(rows[0])), "exactly 96 rows"),
    (lambda rows: rows[1].update(carrier_id="fixed_a"),
     "duplicate candidate/carrier"),
    (lambda rows: rows[0].update(carrier_id="other"), "unknown carrier_id"),
    (lambda rows: rows[0].update(stratum="other"), "unknown stratum"),
])
def test_missing_duplicate_and_unknown_rows_fail_closed(mutation, match):
    rows = treatment_rows()
    mutation(rows)
    with pytest.raises(LocalN48StatsError, match=match):
        analyze_treatment_records(rows)


def test_unbalanced_strata_and_inconsistent_candidate_stratum_fail_closed():
    rows = treatment_rows()
    moved = rows[0]["candidate_id"]
    for row in rows:
        if row["candidate_id"] == moved:
            row["stratum"] = STRATA[1]
    with pytest.raises(LocalN48StatsError, match="has 5 conversations"):
        analyze_treatment_records(rows)

    rows = treatment_rows()
    rows[0]["stratum"] = STRATA[1]
    with pytest.raises(LocalN48StatsError, match="inconsistent strata"):
        analyze_treatment_records(rows)


@pytest.mark.parametrize("field", [
    "l_c_e_c", "l_w_e_c", "l_c_e_w", "l_w_e_w",
    "l_c_b", "l_w_b", "l_c_a_c", "l_w_a_c",
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_required_values_fail_closed(field, value):
    rows = treatment_rows()
    rows[0][field] = value
    with pytest.raises(LocalN48StatsError, match="nonfinite|not numeric"):
        analyze_treatment_records(rows)


def test_missing_required_and_nonfinite_optional_values_fail_closed():
    rows = treatment_rows(optional=True)
    rows[0]["l_c_vp"] = float("nan")
    with pytest.raises(LocalN48StatsError, match="l_c_vp is nonfinite"):
        analyze_treatment_records(rows)


@pytest.mark.parametrize("field", [
    "l_c_e_c", "l_w_e_c", "l_c_e_w", "l_w_e_w",
    "l_c_b", "l_w_b", "l_c_a_c", "l_w_a_c",
])
def test_every_required_arm_score_is_required(field):
    rows = treatment_rows()
    del rows[0][field]
    with pytest.raises(LocalN48StatsError, match=f"missing {field}"):
        analyze_treatment_records(rows)


def test_exact_treatment_release_candidate_ids_are_required():
    rows = treatment_rows()
    expected = sorted({row["candidate_id"] for row in rows})
    analyze_records(rows, expected_candidate_ids=expected)
    wrong = list(expected)
    wrong[0] = "not-the-frozen-candidate"
    with pytest.raises(LocalN48StatsError, match="differ from treatment release"):
        analyze_records(rows, expected_candidate_ids=wrong)
    with pytest.raises(LocalN48StatsError, match="not 48 unique"):
        analyze_records(rows, expected_candidate_ids=expected[:-1])


def test_row_order_is_irrelevant_and_effect_order_is_frozen():
    rows = treatment_rows()
    first = analyze_treatment_records(rows)
    random.Random(41).shuffle(rows)
    second = analyze_treatment_records(rows)
    assert first == second
    assert [effect.stratum for effect in first.conversation_effects[:6]] == [
        STRATA[0]] * 6


def test_stdlib_hoeffding_recomputation_matches_and_fails_closed():
    result = analyze_treatment_records(treatment_rows())
    values = [effect.clipped_z for effect in result.conversation_effects]
    independent = recompute_hoeffding_stdlib(values)
    assert independent == result.primary_hoeffding
    with pytest.raises(LocalN48StatsError, match="exactly 48"):
        recompute_hoeffding_stdlib(values[:-1])
    values[0] = CLIP_UPPER + 0.001
    with pytest.raises(LocalN48StatsError, match="outside support"):
        recompute_hoeffding_stdlib(values)


def test_support_clamps_hoeffding_endpoints_but_preserves_unclamped_values():
    rows = treatment_rows()
    for row in rows:
        row["l_c_e_c"] = row["l_c_b"] + 10.0
    result = analyze_treatment_records(rows)
    assert result.primary_hoeffding.mean_z == CLIP_UPPER
    assert result.primary_hoeffding.ucb == CLIP_UPPER
    assert result.primary_hoeffding.unclamped_ucb > CLIP_UPPER
    assert result.primary_hoeffding.lcb > CLIP_LOWER
