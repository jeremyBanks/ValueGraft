#!/usr/bin/env python3
"""Finite-population coverage and power validation for the v13 bounded rule.

The release validation uses 200,000 independent repetitions for every member
of the frozen 6-family by 3-dependence coverage matrix.  Every repetition is a
stratified simple random sample without replacement: six units are selected
from each of eight fixed finite stratum populations.  Population truths are
computed from the literal arrays before sampling.

``--test-mode`` permits a deliberately non-authorizing reduced run for local
tests and code review.  A reduced artifact always records
``release_eligible=false``, even if its diagnostic gates happen to pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np
import scipy
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from powered_v13_stats import (  # noqa: E402
    CELLS,
    CLIP_LOWER,
    CLIP_UPPER,
    DELTA_CLIP,
    DELTA_TAIL,
    FINAL_N,
    MEAN_FAMILY_ALPHA,
    PER_STRATUM,
    STRATA,
    TAIL_ALPHA,
    alpha_ledger,
)


SCHEMA = "coherent_state_powered_v13_finite_population_simulation_v3"
DESIGN_ID = "coherent-state-powered-successor-v13"
FAMILIES = (
    "two_point",
    "uniform",
    "beta",
    "skewed",
    "rare_responder",
    "contaminated",
)
DEPENDENCE_TARGETS = (0.0, 0.5, 0.9)
MIN_RELEASE_TRIALS = 200_000
DEFAULT_POPULATION_PER_STRATUM = 4096
RELEASE_BATCH_SIZE = 4096
MONTE_CARLO_FAMILY_ALPHA = 0.05
MONTE_CARLO_TWO_SIDED_CONFIDENCE = 0.99
PRIMARY_ESTIMAND_LABEL = "R | eligible"
UNFILTERED_DIAGNOSTIC_LABEL = "R (unfiltered; diagnostic only)"

MEAN_CELL_ALPHA = MEAN_FAMILY_ALPHA / len(CELLS)
MEAN_RADIUS = (CLIP_UPPER - CLIP_LOWER) * math.sqrt(
    math.log(1.0 / MEAN_CELL_ALPHA) / (2.0 * FINAL_N)
)
ZERO_RESPONDER_UCB = 1.0 - TAIL_ALPHA ** (1.0 / FINAL_N)
DEPENDENCE_ABSOLUTE_TOLERANCE = 0.03


def _stable_seed(base_seed: int, *labels: object) -> int:
    payload = json.dumps(
        [int(base_seed), *labels], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _require_population_size(population_per_stratum: int) -> None:
    if population_per_stratum != DEFAULT_POPULATION_PER_STRATUM:
        raise ValueError(
            "the frozen simulation population is exactly 4096 units per stratum"
        )


def _draw_marginal(
    family: str,
    rng: np.random.Generator,
    size: int,
    stratum_index: int,
) -> np.ndarray:
    """Draw one fixed marginal population with deliberate stratum variation."""
    centered_index = stratum_index - (len(STRATA) - 1) / 2.0
    if family == "two_point":
        probabilities = (0.38, 0.42, 0.46, 0.50, 0.54, 0.58, 0.62, 0.66)
        return np.where(
            rng.random(size) < probabilities[stratum_index],
            CLIP_UPPER,
            CLIP_LOWER,
        )
    if family == "uniform":
        values = rng.uniform(CLIP_LOWER, CLIP_UPPER, size=size)
        return np.clip(values + centered_index * 0.018, CLIP_LOWER, CLIP_UPPER)
    if family == "beta":
        alpha = 1.6 + 0.12 * stratum_index
        beta = 4.8 - 0.14 * stratum_index
        return rng.beta(alpha, beta, size=size) - 0.5
    if family == "skewed":
        values = 0.5 - rng.beta(2.0 + 0.08 * stratum_index, 8.0, size=size)
        return np.clip(values + centered_index * 0.008, CLIP_LOWER, CLIP_UPPER)
    if family == "rare_responder":
        probabilities = (0.005, 0.010, 0.015, 0.020,
                         0.025, 0.030, 0.040, 0.050)
        baseline = -0.035 + 0.01 * stratum_index
        return np.where(rng.random(size) < probabilities[stratum_index],
                        4.0, baseline)
    if family == "contaminated":
        positive_probabilities = (0.005, 0.010, 0.015, 0.020,
                                  0.025, 0.030, 0.040, 0.050)
        negative_probabilities = (0.040, 0.030, 0.025, 0.020,
                                  0.015, 0.010, 0.008, 0.005)
        values = np.clip(
            rng.normal(centered_index * 0.012, 0.18, size=size),
            CLIP_LOWER,
            CLIP_UPPER,
        )
        uniforms = rng.random(size)
        positive = uniforms < positive_probabilities[stratum_index]
        negative = (
            (uniforms >= positive_probabilities[stratum_index])
            & (uniforms < positive_probabilities[stratum_index]
               + negative_probabilities[stratum_index])
        )
        values[positive] = 3.0
        values[negative] = -3.0
        return values
    raise ValueError(f"unknown family {family!r}")


def build_coverage_populations(
    seed: int,
    population_per_stratum: int = DEFAULT_POPULATION_PER_STRATUM,
) -> dict[tuple[str, float], np.ndarray]:
    """Build the fixed 6 x 3 matrix, shaped [stratum,population,cell].

    The second cell is a permutation of the first cell's complete flattened
    equal-stratum population, preserving its literal marginal.  An exact target
    fraction of positions is held fixed and the remainder is permuted.  The
    resulting clipped Pearson correlation is measured and must lie within 0.03
    of the target; a population that misses that construction gate fails closed.
    """
    _require_population_size(population_per_stratum)
    result: dict[tuple[str, float], np.ndarray] = {}
    for family in FAMILIES:
        first_cell = np.empty(
            (len(STRATA), population_per_stratum), dtype=np.float64
        )
        for stratum_index, stratum in enumerate(STRATA):
            rng = np.random.default_rng(
                _stable_seed(seed, "coverage_population", family, stratum)
            )
            first_cell[stratum_index, :] = _draw_marginal(
                family, rng, population_per_stratum, stratum_index
            )

        for dependence_target in DEPENDENCE_TARGETS:
            flattened = first_cell.reshape(-1)
            rng = np.random.default_rng(
                _stable_seed(seed, "coverage_dependence", family,
                             dependence_target)
            )
            order = rng.permutation(flattened.size)
            shared_count = int(round(dependence_target * flattened.size))
            shared = order[:shared_count]
            remainder = order[shared_count:]
            second = np.empty_like(flattened)
            second[shared] = flattened[shared]
            if remainder.size:
                donors = remainder[rng.permutation(remainder.size)]
                second[remainder] = flattened[donors]
            population = np.stack(
                [first_cell, second.reshape(first_cell.shape)], axis=2
            )
            realized = summarize_population(
                population, dependence_target
            )["dependence"]["realized_clipped_pearson_overall"]
            if not dependence_gate_pass(realized, dependence_target):
                raise AssertionError(
                    f"{family} dependence {realized} misses target "
                    f"{dependence_target} by more than "
                    f"{DEPENDENCE_ABSOLUTE_TOLERANCE}"
                )
            result[(family, dependence_target)] = population
    return result


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size == 0:
        raise ValueError("correlation arrays differ or are empty")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    if denominator == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def dependence_gate_pass(realized: float | None, target: float) -> bool:
    return (
        realized is not None
        and math.isfinite(realized)
        and abs(realized - target) <= DEPENDENCE_ABSOLUTE_TOLERANCE
    )


def summarize_population(
    population: np.ndarray,
    dependence_target: float | None = None,
) -> dict[str, object]:
    """Return literal finite-population truths and realized diagnostics."""
    expected = (len(STRATA), population.shape[1], len(CELLS))
    if population.shape != expected:
        raise ValueError(
            f"population has shape {population.shape}, expected {expected}"
        )
    if not np.isfinite(population).all():
        raise ValueError("population contains nonfinite values")
    clipped = np.clip(population, CLIP_LOWER, CLIP_UPPER)
    responder = np.max(population, axis=2) > CLIP_UPPER
    per_stratum_means = clipped.mean(axis=1)
    per_stratum_variances = clipped.var(axis=1)
    per_stratum_tail = responder.mean(axis=1)

    within_left = clipped[:, :, 0] - per_stratum_means[:, None, 0]
    within_right = clipped[:, :, 1] - per_stratum_means[:, None, 1]
    per_stratum_correlations = [
        _safe_correlation(clipped[index, :, 0], clipped[index, :, 1])
        for index in range(len(STRATA))
    ]
    canonical_population = np.ascontiguousarray(population, dtype="<f8")
    return {
        "population_per_stratum": int(population.shape[1]),
        "sample_per_stratum": PER_STRATUM,
        "population_array_binding": {
            "shape": list(canonical_population.shape),
            "encoding": "little-endian float64 C-order bytes",
            "sha256": hashlib.sha256(
                canonical_population.tobytes(order="C")
            ).hexdigest(),
        },
        "finite_population_truth": {
            "clipped_means": {
                cell: float(per_stratum_means[:, cell_index].mean())
                for cell_index, cell in enumerate(CELLS)
            },
            "clipped_variances": {
                cell: float(clipped[:, :, cell_index].var())
                for cell_index, cell in enumerate(CELLS)
            },
            "responder_prevalence_any_cell": float(per_stratum_tail.mean()),
            "per_stratum": {
                stratum: {
                    "clipped_means": {
                        cell: float(per_stratum_means[stratum_index, cell_index])
                        for cell_index, cell in enumerate(CELLS)
                    },
                    "clipped_variances": {
                        cell: float(
                            per_stratum_variances[stratum_index, cell_index]
                        )
                        for cell_index, cell in enumerate(CELLS)
                    },
                    "responder_prevalence_any_cell": float(
                        per_stratum_tail[stratum_index]
                    ),
                }
                for stratum_index, stratum in enumerate(STRATA)
            },
        },
        "dependence": {
            "target_clipped_pearson": dependence_target,
            "construction": (
                "hold the target fraction of flattened equal-stratum rows "
                "fixed and permute the remaining preserved marginal"
                if dependence_target is not None else None
            ),
            "realized_clipped_pearson_overall": _safe_correlation(
                clipped[:, :, 0], clipped[:, :, 1]
            ),
            "realized_clipped_pearson_within_strata": _safe_correlation(
                within_left, within_right
            ),
            "realized_clipped_pearson_per_stratum": {
                stratum: per_stratum_correlations[stratum_index]
                for stratum_index, stratum in enumerate(STRATA)
            },
        },
        "stratum_heterogeneity": {
            "clipped_mean_population_sd": {
                cell: float(per_stratum_means[:, cell_index].std())
                for cell_index, cell in enumerate(CELLS)
            },
            "clipped_mean_range": {
                cell: float(np.ptp(per_stratum_means[:, cell_index]))
                for cell_index, cell in enumerate(CELLS)
            },
            "responder_prevalence_population_sd": float(
                per_stratum_tail.std()
            ),
            "responder_prevalence_range": float(np.ptp(per_stratum_tail)),
        },
    }


def sample_indices_without_replacement(
    rng: np.random.Generator,
    trials: int,
    population_size: int,
    sample_size: int = PER_STRATUM,
) -> np.ndarray:
    """Vectorized ordered SRS indices, independently for every trial.

    Position j is redrawn until it differs from positions 0..j-1.  Conditional
    on the preceding positions it is uniform over the remaining population, so
    the unordered row is a simple random sample without replacement.
    """
    if trials <= 0 or not 0 < sample_size <= population_size:
        raise ValueError("invalid trials, population size, or sample size")
    indices = np.empty((trials, sample_size), dtype=np.int64)
    for column in range(sample_size):
        candidate = rng.integers(0, population_size, size=trials)
        if column:
            collision = np.any(
                candidate[:, None] == indices[:, :column], axis=1
            )
            while bool(np.any(collision)):
                candidate[collision] = rng.integers(
                    0, population_size, size=int(collision.sum())
                )
                collision = np.any(
                    candidate[:, None] == indices[:, :column], axis=1
                )
        indices[:, column] = candidate
    return indices


def _bounded_batch(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent vectorized implementation for raw=[trial,N,cell]."""
    if raw.ndim != 3 or raw.shape[1:] != (FINAL_N, len(CELLS)):
        raise ValueError("raw batch has wrong shape")
    clipped = np.clip(raw, CLIP_LOWER, CLIP_UPPER)
    ucbs = np.minimum(CLIP_UPPER, clipped.mean(axis=1) + MEAN_RADIUS)
    responder_counts = (raw.max(axis=2) > CLIP_UPPER).sum(axis=1)
    positive_branch = np.minimum(
        1.0,
        responder_counts / FINAL_N
        + math.sqrt(math.log(1.0 / TAIL_ALPHA) / (2.0 * FINAL_N)),
    )
    tail_ucbs = np.where(
        responder_counts == 0, ZERO_RESPONDER_UCB, positive_branch
    )
    return ucbs, responder_counts, tail_ucbs


def simulate_population(
    population: np.ndarray,
    trials: int,
    seed: int,
    *,
    batch_size: int = 4096,
) -> dict[str, object]:
    """Repeatedly sample one fixed stratified finite population."""
    if trials <= 0 or batch_size <= 0:
        raise ValueError("trials and batch_size must be positive")
    summary = summarize_population(population)
    truth = summary["finite_population_truth"]
    true_means = np.asarray(
        [truth["clipped_means"][cell] for cell in CELLS], dtype=np.float64
    )
    true_tail = float(truth["responder_prevalence_any_cell"])
    rng = np.random.default_rng(seed)

    per_cell_noncoverage = np.zeros(len(CELLS), dtype=np.int64)
    mean_family_noncoverage = 0
    tail_noncoverage = 0
    joint_noncoverage = 0
    joint_resolution = 0
    clipped_means_resolution = 0
    zero_responders = 0
    responder_total = 0
    ucb_sum = np.zeros(len(CELLS), dtype=np.float64)
    tail_ucb_sum = 0.0
    sample_mean_sum = np.zeros(len(CELLS), dtype=np.float64)
    responder_histogram = np.zeros(FINAL_N + 1, dtype=np.int64)

    completed = 0
    population_size = population.shape[1]
    while completed < trials:
        batch = min(batch_size, trials - completed)
        raw = np.empty((batch, FINAL_N, len(CELLS)), dtype=np.float64)
        for stratum_index in range(len(STRATA)):
            indices = sample_indices_without_replacement(
                rng, batch, population_size, PER_STRATUM
            )
            start = stratum_index * PER_STRATUM
            raw[:, start:start + PER_STRATUM, :] = population[
                stratum_index, indices, :
            ]

        clipped_sample_means = np.clip(
            raw, CLIP_LOWER, CLIP_UPPER
        ).mean(axis=1)
        ucbs, responder_counts, tail_ucbs = _bounded_batch(raw)
        cell_failures = ucbs < true_means[None, :]
        mean_failures = np.any(cell_failures, axis=1)
        tail_failures = tail_ucbs < true_tail
        joint_failures = mean_failures | tail_failures
        means_resolved = np.max(ucbs, axis=1) <= DELTA_CLIP
        resolved = means_resolved & (tail_ucbs <= DELTA_TAIL)

        per_cell_noncoverage += cell_failures.sum(axis=0)
        mean_family_noncoverage += int(mean_failures.sum())
        tail_noncoverage += int(tail_failures.sum())
        joint_noncoverage += int(joint_failures.sum())
        joint_resolution += int(resolved.sum())
        clipped_means_resolution += int(means_resolved.sum())
        zero_responders += int((responder_counts == 0).sum())
        responder_total += int(responder_counts.sum())
        ucb_sum += ucbs.sum(axis=0)
        tail_ucb_sum += float(tail_ucbs.sum())
        sample_mean_sum += clipped_sample_means.sum(axis=0)
        responder_histogram += np.bincount(
            responder_counts, minlength=FINAL_N + 1
        )
        completed += batch

    return {
        "trials": trials,
        "noncoverage": {
            "per_cell_count": {
                cell: int(per_cell_noncoverage[cell_index])
                for cell_index, cell in enumerate(CELLS)
            },
            "per_cell_rate": {
                cell: float(per_cell_noncoverage[cell_index] / trials)
                for cell_index, cell in enumerate(CELLS)
            },
            "mean_family_count": mean_family_noncoverage,
            "mean_family_rate": mean_family_noncoverage / trials,
            "tail_count": tail_noncoverage,
            "tail_rate": tail_noncoverage / trials,
            "joint_count": joint_noncoverage,
            "joint_rate": joint_noncoverage / trials,
        },
        "resolution": {
            "joint_count": joint_resolution,
            "joint_rate": joint_resolution / trials,
            "clipped_means_count": clipped_means_resolution,
            "clipped_means_rate": clipped_means_resolution / trials,
        },
        "sampling_diagnostics": {
            "mean_clipped_sample_mean": {
                cell: float(sample_mean_sum[cell_index] / trials)
                for cell_index, cell in enumerate(CELLS)
            },
            "mean_ucb": {
                cell: float(ucb_sum[cell_index] / trials)
                for cell_index, cell in enumerate(CELLS)
            },
            "zero_responder_count": zero_responders,
            "zero_responder_rate": zero_responders / trials,
            "mean_responder_count": responder_total / trials,
            "mean_tail_ucb": tail_ucb_sum / trials,
            "responder_count_histogram": {
                str(index): int(count)
                for index, count in enumerate(responder_histogram)
                if count
            },
        },
    }


def exact_binomial_upper_acceptance_count(
    trials: int,
    probability: float = MONTE_CARLO_FAMILY_ALPHA,
    confidence: float = MONTE_CARLO_TWO_SIDED_CONFIDENCE,
) -> int:
    """Upper count of the equal-tail exact Binomial acceptance interval."""
    if trials <= 0 or not 0.0 < probability < 1.0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid exact-binomial arguments")
    upper_probability = 1.0 - (1.0 - confidence) / 2.0
    return int(stats.binom.ppf(upper_probability, trials, probability))


def run_coverage_matrix(
    *,
    trials: int,
    seed: int,
    population_per_stratum: int,
    batch_size: int,
) -> dict[str, object]:
    populations = build_coverage_populations(seed, population_per_stratum)
    acceptance_count = exact_binomial_upper_acceptance_count(trials)
    rows: dict[str, object] = {}
    for family in FAMILIES:
        for dependence_target in DEPENDENCE_TARGETS:
            key = f"{family}__dependence_{dependence_target:.1f}"
            population = populations[(family, dependence_target)]
            population_summary = summarize_population(
                population, dependence_target=dependence_target
            )
            realized_dependence = population_summary["dependence"][
                "realized_clipped_pearson_overall"
            ]
            dependence_pass = dependence_gate_pass(
                realized_dependence, dependence_target
            )
            simulation = simulate_population(
                population,
                trials,
                _stable_seed(seed, "coverage_sampling", family,
                             dependence_target),
                batch_size=batch_size,
            )
            joint_count = simulation["noncoverage"]["joint_count"]
            rows[key] = {
                "family": family,
                "dependence_target": dependence_target,
                "population": population_summary,
                "simulation": simulation,
                "exact_binomial_acceptance": {
                    "null_probability": MONTE_CARLO_FAMILY_ALPHA,
                    "two_sided_confidence": MONTE_CARLO_TWO_SIDED_CONFIDENCE,
                    "upper_count_inclusive": acceptance_count,
                    "pass": joint_count <= acceptance_count,
                },
                "dependence_gate": {
                    "metric": "flattened equal-stratum finite-population clipped Pearson",
                    "absolute_tolerance": DEPENDENCE_ABSOLUTE_TOLERANCE,
                    "realized": realized_dependence,
                    "pass": dependence_pass,
                },
            }
    literal_keys = {
        f"{family}__dependence_{dependence_target:.1f}"
        for family in FAMILIES
        for dependence_target in DEPENDENCE_TARGETS
    }
    literal_matrix_complete = set(rows) == literal_keys
    if not literal_matrix_complete:
        raise AssertionError("coverage matrix is incomplete")
    return {
        "trials_per_scenario": trials,
        "scenario_count": len(rows),
        "literal_scenario_keys": sorted(literal_keys),
        "literal_matrix_complete": literal_matrix_complete,
        "exact_binomial_upper_acceptance_count": acceptance_count,
        "scenarios": rows,
        "all_scenarios_pass": all(
            row["exact_binomial_acceptance"]["pass"]
            and row["dependence_gate"]["pass"]
            for row in rows.values()
        ),
    }


def _shuffle_strata(
    template: np.ndarray,
    seed: int,
    label: str,
) -> np.ndarray:
    population = np.empty(
        (len(STRATA), template.shape[0], len(CELLS)), dtype=np.float64
    )
    for stratum_index, stratum in enumerate(STRATA):
        rng = np.random.default_rng(_stable_seed(seed, label, stratum))
        population[stratum_index] = template[
            rng.permutation(template.shape[0])
        ]
    return population


POWER_MEAN_SWEEP = (-0.10, 0.0, 0.05, 0.10, 0.15)
POWER_SD_SWEEP = (0.0, 0.125, 0.25, 0.375, 0.5)
POWER_TAIL_SWEEP = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)


def _independent_two_point_template(
    *,
    low: float,
    high: float,
    high_probability: float,
    population_size: int,
) -> np.ndarray:
    """Two fixed marginals with the nearest integer independent joint table."""
    high_count = int(round(high_probability * population_size))
    both_high = int(round(high_count * high_count / population_size))
    high_low = high_count - both_high
    low_high = high_count - both_high
    both_low = population_size - both_high - high_low - low_high
    counts = (both_high, high_low, low_high, both_low)
    if min(counts) < 0 or sum(counts) != population_size:
        raise ValueError("invalid rounded independent joint table")
    values = ((high, high), (high, low), (low, high), (low, low))
    return np.concatenate(
        [
            np.repeat(np.asarray([value], dtype=np.float64), count, axis=0)
            for value, count in zip(values, counts)
        ],
        axis=0,
    )


def build_mean_sweep_population(
    clipped_mean: float,
    seed: int,
    population_per_stratum: int = DEFAULT_POPULATION_PER_STRATUM,
) -> np.ndarray:
    """Sweep A: independent {-0.5,+0.5} cells with p=m+0.5."""
    _require_population_size(population_per_stratum)
    if clipped_mean not in POWER_MEAN_SWEEP:
        raise ValueError("clipped mean is outside the frozen sweep")
    template = _independent_two_point_template(
        low=-0.5,
        high=0.5,
        high_probability=clipped_mean + 0.5,
        population_size=population_per_stratum,
    )
    return _shuffle_strata(template, seed, f"power_mean:{clipped_mean}")


def build_sd_sweep_population(
    clipped_sd: float,
    seed: int,
    population_per_stratum: int = DEFAULT_POPULATION_PER_STRATUM,
) -> np.ndarray:
    """Sweep B: independent {-s,+s} cells with exact mean zero."""
    _require_population_size(population_per_stratum)
    if clipped_sd not in POWER_SD_SWEEP:
        raise ValueError("clipped SD is outside the frozen sweep")
    template = _independent_two_point_template(
        low=-clipped_sd,
        high=clipped_sd,
        high_probability=0.5,
        population_size=population_per_stratum,
    )
    return _shuffle_strata(template, seed, f"power_sd:{clipped_sd}")


def build_tail_sweep_population(
    responder_prevalence: float,
    seed: int,
    population_per_stratum: int = DEFAULT_POPULATION_PER_STRATUM,
) -> np.ndarray:
    """Sweep C: shared raw-5 responders and an approximately centering baseline."""
    _require_population_size(population_per_stratum)
    if responder_prevalence not in POWER_TAIL_SWEEP:
        raise ValueError("responder prevalence is outside the frozen sweep")
    responder_count = int(round(responder_prevalence * population_per_stratum))
    baseline = (
        0.0
        if responder_prevalence == 0.0
        else -0.5 * responder_prevalence / (1.0 - responder_prevalence)
    )
    template = np.repeat(
        np.asarray([[baseline, baseline]], dtype=np.float64),
        population_per_stratum,
        axis=0,
    )
    template[:responder_count, :] = 5.0
    return _shuffle_strata(
        template, seed, f"power_tail:{responder_prevalence}"
    )


def _power_row(
    population: np.ndarray,
    *,
    trials: int,
    seed: int,
    batch_size: int,
) -> dict[str, object]:
    population_summary = summarize_population(population)
    simulation = simulate_population(
        population, trials, seed, batch_size=batch_size
    )
    truth = population_summary["finite_population_truth"]
    return {
        "population": population_summary,
        "realized_clipped_means": truth["clipped_means"],
        "realized_clipped_variances": truth["clipped_variances"],
        "realized_clipped_sds": {
            cell: math.sqrt(truth["clipped_variances"][cell]) for cell in CELLS
        },
        "realized_responder_prevalence_any_cell": truth[
            "responder_prevalence_any_cell"
        ],
        "simulation": simulation,
        "joint_resolution_probability": simulation["resolution"]["joint_rate"],
        "zero_responder_probability": simulation["sampling_diagnostics"][
            "zero_responder_rate"
        ],
    }


def run_power_validation(
    *,
    trials: int,
    seed: int,
    population_per_stratum: int,
    batch_size: int,
) -> dict[str, object]:
    mean_rows: dict[str, object] = {}
    for clipped_mean in POWER_MEAN_SWEEP:
        key = f"mean_{clipped_mean:.3f}"
        row = _power_row(
            build_mean_sweep_population(
                clipped_mean, seed, population_per_stratum
            ),
            trials=trials,
            seed=_stable_seed(seed, "power_mean_sampling", clipped_mean),
            batch_size=batch_size,
        )
        row["requested_clipped_mean"] = clipped_mean
        row["construction"] = (
            "independent cells on {-0.5,+0.5} with P(+0.5)=m+0.5"
        )
        mean_rows[key] = row

    sd_rows: dict[str, object] = {}
    for clipped_sd in POWER_SD_SWEEP:
        key = f"sd_{clipped_sd:.3f}"
        row = _power_row(
            build_sd_sweep_population(clipped_sd, seed, population_per_stratum),
            trials=trials,
            seed=_stable_seed(seed, "power_sd_sampling", clipped_sd),
            batch_size=batch_size,
        )
        row["requested_clipped_sd"] = clipped_sd
        row["construction"] = "independent cells on {-s,+s} with mean zero"
        sd_rows[key] = row

    tail_rows: dict[str, object] = {}
    for responder_prevalence in POWER_TAIL_SWEEP:
        key = f"tail_{responder_prevalence:.4f}"
        row = _power_row(
            build_tail_sweep_population(
                responder_prevalence, seed, population_per_stratum
            ),
            trials=trials,
            seed=_stable_seed(seed, "power_tail_sampling",
                              responder_prevalence),
            batch_size=batch_size,
        )
        row["requested_responder_prevalence"] = responder_prevalence
        row["construction"] = (
            "both cells raw=5 on rounded responder count; otherwise "
            "-0.5*p/(1-p) using requested p"
        )
        tail_rows[key] = row

    mean_headline = mean_rows["mean_0.050"]
    sd_headline = sd_rows["sd_0.250"]
    headline = {
        "worst_variance_mean_005": {
            "source_sweep": "A:clipped_mean:mean_0.050",
            "required_joint_resolution_probability": 0.80,
            "observed_joint_resolution_probability": mean_headline[
                "joint_resolution_probability"
            ],
            "zero_responder_truth": (
                mean_headline["realized_responder_prevalence_any_cell"] == 0.0
            ),
        },
        "mean_zero_sd_025": {
            "source_sweep": "B:clipped_sd:sd_0.250",
            "required_joint_resolution_probability": 0.95,
            "observed_joint_resolution_probability": sd_headline[
                "joint_resolution_probability"
            ],
            "zero_responder_truth": (
                sd_headline["realized_responder_prevalence_any_cell"] == 0.0
            ),
        },
    }
    for row in headline.values():
        row["pass"] = (
            row["zero_responder_truth"]
            and row["observed_joint_resolution_probability"]
            >= row["required_joint_resolution_probability"]
        )

    expected_mean_keys = {f"mean_{value:.3f}" for value in POWER_MEAN_SWEEP}
    expected_sd_keys = {f"sd_{value:.3f}" for value in POWER_SD_SWEEP}
    expected_tail_keys = {f"tail_{value:.4f}" for value in POWER_TAIL_SWEEP}
    literal_grid_complete = (
        set(mean_rows) == expected_mean_keys
        and set(sd_rows) == expected_sd_keys
        and set(tail_rows) == expected_tail_keys
    )
    return {
        "trials_per_population": trials,
        "literal_axes": {
            "A_clipped_mean": list(POWER_MEAN_SWEEP),
            "B_clipped_sd": list(POWER_SD_SWEEP),
            "C_any_cell_responder_prevalence": list(POWER_TAIL_SWEEP),
        },
        "sweeps": {
            "A_clipped_mean": mean_rows,
            "B_clipped_sd": sd_rows,
            "C_any_cell_responder_prevalence": tail_rows,
        },
        "literal_grid_population_count": (
            len(mean_rows) + len(sd_rows) + len(tail_rows)
        ),
        "literal_grid_complete": literal_grid_complete,
        "headline_zero_responder_gates": headline,
        "headline_gates_pass": all(row["pass"] for row in headline.values()),
    }


def conditional_damage_label_diagnostic(seed: int) -> dict[str, object]:
    """Demonstrate why the damage-screened estimand must remain conditional."""
    rng = np.random.default_rng(_stable_seed(seed, "conditional_damage"))
    population_size = 4096
    damage = rng.uniform(4.5, 5.5, size=population_size)
    raw_recovery = (
        0.02
        + 0.40 * (damage - 5.0)
        + rng.normal(0.0, 0.025, size=population_size)
    )
    eligible = damage >= 5.0
    correlation = _safe_correlation(damage, raw_recovery)
    primary_label = PRIMARY_ESTIMAND_LABEL
    unfiltered_label = UNFILTERED_DIAGNOSTIC_LABEL
    checks = {
        "primary_label_is_literal_conditional_recipe": (
            primary_label == "R | eligible"
        ),
        "unfiltered_label_is_diagnostic_only": "diagnostic only" in unfiltered_label,
        "primary_and_unfiltered_labels_differ": primary_label != unfiltered_label,
        "damage_recovery_correlation_is_material": (
            correlation is not None and abs(correlation) >= 0.5
        ),
        "eligibility_changes_recovery_mean": not math.isclose(
            float(raw_recovery[eligible].mean()),
            float(raw_recovery.mean()),
            rel_tol=0.0,
            abs_tol=1e-3,
        ),
    }
    return {
        "damage_gate_nats": 5.0,
        "damage_support_nats": [float(damage.min()), float(damage.max())],
        "population_size": population_size,
        "eligible_count": int(eligible.sum()),
        "damage_raw_recovery_correlation": correlation,
        "primary_estimand": {
            "label": primary_label,
            "raw_recovery_mean": float(raw_recovery[eligible].mean()),
            "scope": "finite engineered recipe conditional on all Phase-A eligibility gates",
        },
        "unfiltered_diagnostic": {
            "label": unfiltered_label,
            "raw_recovery_mean": float(raw_recovery.mean()),
            "scope": "not a v13 primary claim",
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def golden_checks(seed: int) -> dict[str, object]:
    """Small deterministic edge cases for the independent formula."""
    zeros = np.zeros((1, FINAL_N, len(CELLS)), dtype=np.float64)
    zero_ucbs, zero_counts, zero_tail = _bounded_batch(zeros)
    one_responder = zeros.copy()
    one_responder[0, 0, 0] = 5.0
    one_ucbs, one_counts, one_tail = _bounded_batch(one_responder)
    clipped_extremes = np.clip(
        np.asarray([-5.0, 5.0]), CLIP_LOWER, CLIP_UPPER
    )
    conditional = conditional_damage_label_diagnostic(seed)
    checks = {
        "all_zero_retains_positive_mean_radius": bool(
            np.all(zero_ucbs == MEAN_RADIUS) and MEAN_RADIUS > 0.0
        ),
        "all_zero_uses_positive_tail_bound": bool(
            zero_counts[0] == 0
            and zero_tail[0] == ZERO_RESPONDER_UCB
            and ZERO_RESPONDER_UCB > 0.0
        ),
        "all_zero_jointly_resolves": bool(
            np.max(zero_ucbs) <= DELTA_CLIP and zero_tail[0] <= DELTA_TAIL
        ),
        "one_responder_uses_nonzero_branch_and_does_not_resolve": bool(
            one_counts[0] == 1
            and one_tail[0] > DELTA_TAIL
            and not (
                np.max(one_ucbs) <= DELTA_CLIP and one_tail[0] <= DELTA_TAIL
            )
        ),
        "raw_values_clip_at_both_limits": bool(
            np.array_equal(
                clipped_extremes, np.asarray([CLIP_LOWER, CLIP_UPPER])
            )
        ),
        "conditional_damage_labels_pass": conditional["all_checks_pass"],
    }
    return {
        "mean_radius": MEAN_RADIUS,
        "zero_responder_ucb": ZERO_RESPONDER_UCB,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "conditional_damage_label_diagnostic": conditional,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    core_path = root / "src" / "powered_v13_stats.py"
    lock_path = root / "uv.lock"
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_clean = status == ""
    except (OSError, subprocess.CalledProcessError):
        commit = None
        git_clean = False
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "script_path": str(script_path.relative_to(root)),
        "script_sha256": _sha256_file(script_path),
        "production_stats_path": str(core_path.relative_to(root)),
        "production_stats_sha256": _sha256_file(core_path),
        "uv_lock_sha256": _sha256_file(lock_path) if lock_path.exists() else None,
        "git_commit": commit if git_clean else None,
        "git_clean": git_clean,
    }


def release_gate_status(
    *,
    trials: int,
    power_trials: int,
    population_per_stratum: int,
    batch_size: int,
    test_mode: bool,
    coverage: dict[str, object],
    power: dict[str, object],
    goldens: dict[str, object],
    provenance: dict[str, object],
) -> dict[str, object]:
    coverage_trial_requirement_met = trials >= MIN_RELEASE_TRIALS
    power_trial_requirement_met = power_trials >= MIN_RELEASE_TRIALS
    population_requirement_met = (
        population_per_stratum == DEFAULT_POPULATION_PER_STRATUM
    )
    batch_size_requirement_met = batch_size == RELEASE_BATCH_SIZE
    exact_matrix_met = (
        coverage.get("scenario_count") == 18
        and coverage.get("literal_matrix_complete") is True
    )
    scenarios = coverage.get("scenarios", {})
    all_dependence_met = (
        isinstance(scenarios, dict)
        and len(scenarios) == 18
        and all(
            row["dependence_gate"]["pass"] for row in scenarios.values()
        )
    )
    literal_power_grid_met = (
        power.get("literal_grid_population_count") == 17
        and power.get("literal_grid_complete") is True
    )
    diagnostic_gates_pass = (
        exact_matrix_met
        and all_dependence_met
        and coverage.get("all_scenarios_pass") is True
        and literal_power_grid_met
        and power.get("headline_gates_pass") is True
        and goldens.get("all_checks_pass") is True
    )
    clean_git = provenance.get("git_clean") is True
    non_test_mode = not test_mode
    release_eligible = (
        non_test_mode
        and coverage_trial_requirement_met
        and power_trial_requirement_met
        and population_requirement_met
        and batch_size_requirement_met
        and clean_git
        and diagnostic_gates_pass
    )
    return {
        "full_trial_requirement": MIN_RELEASE_TRIALS,
        "coverage_trial_requirement_met": coverage_trial_requirement_met,
        "power_trial_requirement_met": power_trial_requirement_met,
        "population_4096_per_stratum_met": population_requirement_met,
        "release_batch_size_met": batch_size_requirement_met,
        "exact_18_scenario_matrix_met": exact_matrix_met,
        "all_dependence_tolerances_met": all_dependence_met,
        "literal_power_grid_met": literal_power_grid_met,
        "coverage_pass": coverage.get("all_scenarios_pass") is True,
        "headline_power_pass": power.get("headline_gates_pass") is True,
        "golden_checks_pass": goldens.get("all_checks_pass") is True,
        "clean_git_provenance_met": clean_git,
        "non_test_mode_met": non_test_mode,
        "diagnostic_gates_pass": diagnostic_gates_pass,
        "release_eligible": release_eligible,
    }


def build_validation_result(
    *,
    trials: int,
    power_trials: int,
    seed: int,
    population_per_stratum: int,
    batch_size: int,
    test_mode: bool,
) -> dict[str, object]:
    if not test_mode and (
        trials < MIN_RELEASE_TRIALS or power_trials < MIN_RELEASE_TRIALS
    ):
        raise ValueError(
            "release validation requires at least "
            f"{MIN_RELEASE_TRIALS} coverage and power trials"
        )
    provenance = artifact_provenance()
    coverage = run_coverage_matrix(
        trials=trials,
        seed=seed,
        population_per_stratum=population_per_stratum,
        batch_size=batch_size,
    )
    power = run_power_validation(
        trials=power_trials,
        seed=seed,
        population_per_stratum=population_per_stratum,
        batch_size=batch_size,
    )
    goldens = golden_checks(seed)
    gates = release_gate_status(
        trials=trials,
        power_trials=power_trials,
        population_per_stratum=population_per_stratum,
        batch_size=batch_size,
        test_mode=test_mode,
        coverage=coverage,
        power=power,
        goldens=goldens,
        provenance=provenance,
    )
    return {
        "schema": SCHEMA,
        "design_id": DESIGN_ID,
        "seed": seed,
        "mode": "non_authorizing_test" if test_mode else "release_validation",
        "test_mode": test_mode,
        "estimand": {
            "primary_label": PRIMARY_ESTIMAND_LABEL,
            "scope": "R conditioned on the frozen Phase-A eligibility rules",
            "unfiltered_recipe_is_not_primary": True,
        },
        "sampling_design": {
            "strata": list(STRATA),
            "sample_per_stratum": PER_STRATUM,
            "total_sample_size": FINAL_N,
            "population_per_stratum": population_per_stratum,
            "simulation_batch_size": batch_size,
            "release_batch_size": RELEASE_BATCH_SIZE,
            "sampling": "independent stratified simple random samples without replacement",
        },
        "formula": {
            "clip": [CLIP_LOWER, CLIP_UPPER],
            "clipped_range": CLIP_UPPER - CLIP_LOWER,
            "mean_alpha_per_cell": MEAN_CELL_ALPHA,
            "mean_radius": MEAN_RADIUS,
            "tail_alpha": TAIL_ALPHA,
            "zero_responder_ucb": ZERO_RESPONDER_UCB,
            "delta_clip": DELTA_CLIP,
            "delta_tail": DELTA_TAIL,
        },
        "alpha_ledger": alpha_ledger(),
        "provenance": provenance,
        "coverage_validation": coverage,
        "power_validation": power,
        "golden_checks": goldens,
        "gates": gates,
    }


def write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=MIN_RELEASE_TRIALS)
    parser.add_argument(
        "--power-trials",
        type=int,
        help="defaults to --trials",
    )
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--population-per-stratum",
        type=int,
        default=DEFAULT_POPULATION_PER_STRATUM,
    )
    parser.add_argument("--batch-size", type=int, default=RELEASE_BATCH_SIZE)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    power_trials = args.trials if args.power_trials is None else args.power_trials
    result = build_validation_result(
        trials=args.trials,
        power_trials=power_trials,
        seed=args.seed,
        population_per_stratum=args.population_per_stratum,
        batch_size=args.batch_size,
        test_mode=args.test_mode,
    )
    write_json_exclusive(args.output, result)
    coverage_rows = result["coverage_validation"]["scenarios"]
    print(json.dumps({
        "output": str(args.output),
        "mode": result["mode"],
        "coverage_scenarios": len(coverage_rows),
        "max_joint_noncoverage_count": max(
            row["simulation"]["noncoverage"]["joint_count"]
            for row in coverage_rows.values()
        ),
        "exact_binomial_upper_acceptance_count": result[
            "coverage_validation"
        ]["exact_binomial_upper_acceptance_count"],
        "headline_power_pass": result["gates"]["headline_power_pass"],
        "release_eligible": result["gates"]["release_eligible"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
