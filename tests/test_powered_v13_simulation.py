import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.simulate_powered_v13_stats import (
    CELLS,
    DEFAULT_POPULATION_PER_STRATUM,
    DEPENDENCE_TARGETS,
    FAMILIES,
    POWER_MEAN_SWEEP,
    POWER_SD_SWEEP,
    POWER_TAIL_SWEEP,
    PRIMARY_ESTIMAND_LABEL,
    UNFILTERED_DIAGNOSTIC_LABEL,
    artifact_provenance,
    build_coverage_populations,
    build_mean_sweep_population,
    build_sd_sweep_population,
    build_tail_sweep_population,
    build_validation_result,
    conditional_damage_label_diagnostic,
    dependence_gate_pass,
    exact_binomial_upper_acceptance_count,
    golden_checks,
    run_coverage_matrix,
    run_power_validation,
    release_gate_status,
    sample_indices_without_replacement,
    summarize_population,
    write_json_exclusive,
)


def test_exact_two_sided_99pct_binomial_upper_count_is_frozen():
    expected = int(stats.binom.ppf(0.995, 200_000, 0.05))
    assert expected == 10_252
    assert exact_binomial_upper_acceptance_count(200_000) == expected


def test_coverage_matrix_has_literal_populations_and_dependence():
    populations = build_coverage_populations(20260712)
    assert set(populations) == {
        (family, target)
        for family in FAMILIES
        for target in DEPENDENCE_TARGETS
    }
    for (family, target), population in populations.items():
        assert family in FAMILIES
        assert population.shape == (8, 4096, 2)
        summary = summarize_population(population, target)
        binding = summary["population_array_binding"]
        assert binding["encoding"] == "little-endian float64 C-order bytes"
        canonical = np.ascontiguousarray(population, dtype="<f8")
        assert binding["sha256"] == hashlib.sha256(
            canonical.tobytes(order="C")
        ).hexdigest()
        dependence = summary["dependence"]
        realized = dependence["realized_clipped_pearson_overall"]
        assert dependence["target_clipped_pearson"] == target
        assert realized is not None
        assert abs(realized - target) <= 0.03

        clipped = np.clip(population, -0.5, 0.5)
        responder = np.max(population, axis=2) > 0.5
        truth = summary["finite_population_truth"]
        for cell_index, cell in enumerate(CELLS):
            assert truth["clipped_means"][cell] == pytest.approx(
                clipped[:, :, cell_index].mean(), abs=1e-15
            )
        assert truth["responder_prevalence_any_cell"] == pytest.approx(
            responder.mean(), abs=1e-15
        )


def test_dependence_tolerance_fails_closed():
    assert dependence_gate_pass(0.529999, 0.5)
    assert not dependence_gate_pass(0.530001, 0.5)
    assert not dependence_gate_pass(None, 0.5)
    assert not dependence_gate_pass(float("nan"), 0.5)


def test_vectorized_sampler_is_without_replacement_and_uniform():
    rng = np.random.default_rng(17)
    indices = sample_indices_without_replacement(
        rng, trials=20_000, population_size=20, sample_size=6
    )
    assert indices.shape == (20_000, 6)
    assert np.all(np.sort(indices, axis=1)[:, 1:] !=
                  np.sort(indices, axis=1)[:, :-1])
    inclusion = np.bincount(indices.reshape(-1), minlength=20) / 20_000
    assert np.max(np.abs(inclusion - 0.30)) < 0.015


def test_literal_power_population_truths_after_integer_rounding():
    for requested_mean in POWER_MEAN_SWEEP:
        population = build_mean_sweep_population(requested_mean, 31)
        truth = summarize_population(population)["finite_population_truth"]
        realized = (
            round((requested_mean + 0.5) * 4096) / 4096 - 0.5
        )
        assert truth["responder_prevalence_any_cell"] == 0.0
        for cell in CELLS:
            assert truth["clipped_means"][cell] == pytest.approx(
                realized, abs=1e-15
            )

    for requested_sd in POWER_SD_SWEEP:
        population = build_sd_sweep_population(requested_sd, 31)
        truth = summarize_population(population)["finite_population_truth"]
        assert truth["responder_prevalence_any_cell"] == 0.0
        for cell in CELLS:
            assert truth["clipped_means"][cell] == 0.0
            assert math.sqrt(truth["clipped_variances"][cell]) == pytest.approx(
                requested_sd, abs=1e-15
            )

    for requested_tail in POWER_TAIL_SWEEP:
        population = build_tail_sweep_population(requested_tail, 31)
        truth = summarize_population(population)["finite_population_truth"]
        realized_tail = round(requested_tail * 4096) / 4096
        assert truth["responder_prevalence_any_cell"] == pytest.approx(
            realized_tail, abs=1e-15
        )
        if requested_tail:
            assert np.all(
                population[np.max(population, axis=2) > 0.5] == 5.0
            )
        for cell in CELLS:
            assert abs(truth["clipped_means"][cell]) < 6e-5


def test_small_coverage_and_power_runs_have_exact_literal_surfaces():
    coverage = run_coverage_matrix(
        trials=250,
        seed=73,
        population_per_stratum=DEFAULT_POPULATION_PER_STRATUM,
        batch_size=64,
    )
    assert coverage["scenario_count"] == 18
    assert coverage["literal_matrix_complete"]
    assert set(coverage["literal_scenario_keys"]) == set(
        coverage["scenarios"]
    )
    assert all(
        row["dependence_gate"]["pass"]
        for row in coverage["scenarios"].values()
    )
    assert all(
        sum(row["simulation"]["sampling_diagnostics"]
            ["responder_count_histogram"].values()) == 250
        for row in coverage["scenarios"].values()
    )

    power = run_power_validation(
        trials=500,
        seed=73,
        population_per_stratum=DEFAULT_POPULATION_PER_STRATUM,
        batch_size=128,
    )
    assert power["literal_grid_complete"]
    assert power["literal_grid_population_count"] == 17
    assert power["literal_axes"] == {
        "A_clipped_mean": list(POWER_MEAN_SWEEP),
        "B_clipped_sd": list(POWER_SD_SWEEP),
        "C_any_cell_responder_prevalence": list(POWER_TAIL_SWEEP),
    }
    assert power["headline_zero_responder_gates"][
        "worst_variance_mean_005"
    ]["source_sweep"] == "A:clipped_mean:mean_0.050"
    assert power["headline_zero_responder_gates"][
        "mean_zero_sd_025"
    ]["source_sweep"] == "B:clipped_sd:sd_0.250"


def test_non_authorizing_test_mode_replays_deterministically():
    kwargs = {
        "trials": 25,
        "power_trials": 25,
        "seed": 991,
        "population_per_stratum": 4096,
        "batch_size": 4096,
        "test_mode": True,
    }
    first = build_validation_result(**kwargs)
    second = build_validation_result(**kwargs)
    assert first == second
    assert first["mode"] == "non_authorizing_test"
    assert not first["gates"]["coverage_trial_requirement_met"]
    assert not first["gates"]["power_trial_requirement_met"]
    assert not first["gates"]["non_test_mode_met"]
    assert not first["gates"]["release_eligible"]


def test_conditional_damage_diagnostic_never_labels_unfiltered_r_primary():
    diagnostic = conditional_damage_label_diagnostic(101)
    assert PRIMARY_ESTIMAND_LABEL == "R | eligible"
    assert diagnostic["primary_estimand"]["label"] == PRIMARY_ESTIMAND_LABEL
    assert diagnostic["unfiltered_diagnostic"]["label"] == (
        UNFILTERED_DIAGNOSTIC_LABEL
    )
    assert diagnostic["unfiltered_diagnostic"]["scope"] == (
        "not a v13 primary claim"
    )
    assert diagnostic["all_checks_pass"]
    assert golden_checks(101)["all_checks_pass"]


def test_reduced_validation_cannot_be_release_eligible():
    with pytest.raises(ValueError, match="coverage and power trials"):
        build_validation_result(
            trials=199_999,
            power_trials=200_000,
            seed=1,
            population_per_stratum=4096,
            batch_size=32,
            test_mode=False,
        )


def _passing_gate_inputs():
    scenario = {"dependence_gate": {"pass": True}}
    return {
        "trials": 200_000,
        "power_trials": 200_000,
        "population_per_stratum": 4096,
        "batch_size": 4096,
        "test_mode": False,
        "coverage": {
            "scenario_count": 18,
            "literal_matrix_complete": True,
            "all_scenarios_pass": True,
            "scenarios": {str(index): scenario for index in range(18)},
        },
        "power": {
            "literal_grid_population_count": 17,
            "literal_grid_complete": True,
            "headline_gates_pass": True,
        },
        "goldens": {"all_checks_pass": True},
        "provenance": {"git_clean": True},
    }


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    [
        ("trials", 199_999, "coverage_trial_requirement_met"),
        ("power_trials", 199_999, "power_trial_requirement_met"),
        ("population_per_stratum", 400, "population_4096_per_stratum_met"),
        ("batch_size", 2048, "release_batch_size_met"),
        ("test_mode", True, "non_test_mode_met"),
    ],
)
def test_each_literal_release_precondition_fails_closed(field, value, gate):
    inputs = _passing_gate_inputs()
    inputs[field] = value
    result = release_gate_status(**inputs)
    assert not result[gate]
    assert not result["release_eligible"]


def test_dirty_git_and_incomplete_surfaces_force_release_false():
    inputs = _passing_gate_inputs()
    inputs["provenance"] = {"git_clean": False}
    assert not release_gate_status(**inputs)["release_eligible"]

    inputs = _passing_gate_inputs()
    inputs["coverage"]["scenarios"].pop("17")
    assert not release_gate_status(**inputs)["release_eligible"]

    inputs = _passing_gate_inputs()
    inputs["power"]["literal_grid_population_count"] = 16
    assert not release_gate_status(**inputs)["release_eligible"]


def test_frozen_population_size_rejects_non_4096():
    with pytest.raises(ValueError, match="exactly 4096"):
        build_coverage_populations(1, population_per_stratum=400)


def test_artifact_provenance_hashes_the_literal_script_and_core():
    provenance = artifact_provenance()
    assert provenance["python_version"]
    assert provenance["numpy_version"]
    assert provenance["scipy_version"]
    assert len(provenance["script_sha256"]) == hashlib.sha256().digest_size * 2
    assert len(provenance["production_stats_sha256"]) == (
        hashlib.sha256().digest_size * 2
    )


def test_output_is_exclusive_create(tmp_path):
    output = tmp_path / "simulation.json"
    write_json_exclusive(output, {"first": True})
    with pytest.raises(FileExistsError):
        write_json_exclusive(output, {"second": True})
