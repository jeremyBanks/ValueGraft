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
    analyze_treatment_records,
    clopper_pearson_upper,
    recompute_hoeffding_stdlib,
)


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
                    "l_c_e_c": -4.0 + x,
                }
                if optional:
                    row.update({
                        "l_c_e_w": -3.98,
                        "l_c_vp": -4.01,
                        "l_c_a_c": 1.0,
                        "l_w_a_c": -1.0,
                        "l_w_b": -2.0,
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


def test_optional_specificity_placebo_and_damage_companions():
    result = analyze_treatment_records(treatment_rows(optional=True))
    expected = {
        "specificity_correct_minus_wrong",
        "wrong_history_movement_from_fresh",
        "placebo_movement_from_fresh",
        "correct_minus_placebo",
        "correct_target_damage",
        "margin_damage",
    }
    assert set(result.companion_metrics) == expected
    x_mean = result.primary_raw.mean
    assert result.companion_metrics[
        "specificity_correct_minus_wrong"].mean == pytest.approx(x_mean - 0.02)
    assert result.companion_metrics[
        "wrong_history_movement_from_fresh"].mean == pytest.approx(0.02)
    assert result.companion_metrics[
        "placebo_movement_from_fresh"].mean == pytest.approx(-0.01)
    assert result.companion_metrics[
        "correct_minus_placebo"].mean == pytest.approx(x_mean + 0.01)
    assert result.companion_metrics[
        "correct_target_damage"].mean == pytest.approx(5.0)
    assert result.companion_metrics["margin_damage"].mean == pytest.approx(4.0)
    assert all(summary.n_conversations == FINAL_N
               for summary in result.companion_metrics.values())


def test_partial_optional_companion_fails_closed():
    rows = treatment_rows()
    rows[0]["l_c_e_w"] = -4.0
    with pytest.raises(LocalN48StatsError, match="present in 1/96"):
        analyze_treatment_records(rows)

    rows = treatment_rows()
    for row in rows:
        row["l_w_a_c"] = -1.0
        row["l_w_b"] = -2.0
    with pytest.raises(LocalN48StatsError, match="requires l_c_a_c"):
        analyze_treatment_records(rows)


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


@pytest.mark.parametrize("field,value", [
    ("l_c_e_c", float("nan")),
    ("l_c_b", float("inf")),
    ("l_c_e_c", True),
])
def test_nonfinite_or_boolean_required_values_fail_closed(field, value):
    rows = treatment_rows()
    rows[0][field] = value
    with pytest.raises(LocalN48StatsError, match="nonfinite|not numeric"):
        analyze_treatment_records(rows)


def test_missing_required_and_nonfinite_optional_values_fail_closed():
    rows = treatment_rows()
    del rows[0]["l_c_b"]
    with pytest.raises(LocalN48StatsError, match="missing l_c_b"):
        analyze_treatment_records(rows)

    rows = treatment_rows(optional=True)
    rows[0]["l_c_vp"] = float("nan")
    with pytest.raises(LocalN48StatsError, match="l_c_vp is nonfinite"):
        analyze_treatment_records(rows)


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
