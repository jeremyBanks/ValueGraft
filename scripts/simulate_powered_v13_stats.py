#!/usr/bin/env python3
"""Pre-outcome coverage and stopping simulations for the v13 UCB rule."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from powered_v13_stats import CELLS, DELTA_NAT, LOOKS, alpha_ledger


def vectorized_look(samples: np.ndarray, per_stratum: int, alpha: float):
    """Independent vectorized implementation of the preregistered formula.

    samples shape = [trials, 8 strata, 6 ranks, 2 cells].
    """
    values = samples[:, :, :per_stratum, :]
    stratum_means = values.mean(axis=2)
    stratum_vars = values.var(axis=2, ddof=1)
    components = (1.0 / 8.0) ** 2 * stratum_vars / per_stratum
    variance = components.sum(axis=1)
    standard_error = np.sqrt(variance)
    denominator = (components * components / (per_stratum - 1)).sum(axis=1)
    degrees = np.divide(
        variance * variance, denominator,
        out=np.full_like(variance, np.inf), where=denominator > 0.0)
    estimate = stratum_means.mean(axis=1)
    critical = stats.t.ppf(1.0 - alpha / len(CELLS), degrees)
    critical = np.where(variance == 0.0, 0.0, critical)
    return estimate, estimate + critical * standard_error


def run_selector(samples: np.ndarray, truth: np.ndarray):
    trials = samples.shape[0]
    selected = np.full(trials, len(LOOKS) - 1, dtype=np.int64)
    unresolved = np.ones(trials, dtype=bool)
    look_rows = []
    for look_index, (n_total, per_stratum, alpha) in enumerate(LOOKS):
        estimate, ucbs = vectorized_look(samples, per_stratum, alpha)
        primary = ucbs.max(axis=1)
        stop = unresolved & (primary <= DELTA_NAT)
        selected[stop] = look_index
        unresolved[stop] = False
        look_rows.append((n_total, estimate, ucbs, primary))
    selected_ucbs = np.empty((trials, len(CELLS)), dtype=np.float64)
    selected_primary = np.empty(trials, dtype=np.float64)
    for index, (_, _, ucbs, primary) in enumerate(look_rows):
        mask = selected == index
        selected_ucbs[mask] = ucbs[mask]
        selected_primary[mask] = primary[mask]
    noncoverage = np.any(selected_ucbs < truth[None, :], axis=1)
    false_bound = ((truth.max() > DELTA_NAT) &
                   (selected_primary <= DELTA_NAT))
    return {
        "selected_counts": {
            str(LOOKS[index][0]): int(np.sum(selected == index))
            for index in range(len(LOOKS))
        },
        "selected_rates": {
            str(LOOKS[index][0]): float(np.mean(selected == index))
            for index in range(len(LOOKS))
        },
        "simultaneous_noncoverage_rate": float(np.mean(noncoverage)),
        "false_bound_rate": float(np.mean(false_bound)),
        "mean_selected_primary_ucb": float(np.mean(selected_primary)),
    }


def correlated_normal(rng, trials, mean, sd, rho):
    covariance = np.asarray([[1.0, rho], [rho, 1.0]])
    chol = np.linalg.cholesky(covariance)
    z = rng.standard_normal((trials, 8, 6, 2)) @ chol.T
    return mean[None, None, None, :] + sd * z


def stress_samples(rng, trials, kind, mean, sd):
    shape = (trials, 8, 6, 2)
    if kind == "t3":
        z = rng.standard_t(3, size=shape) / math.sqrt(3.0)
    elif kind == "skew_lognormal":
        raw = rng.lognormal(mean=0.0, sigma=1.0, size=shape)
        z = (raw - math.exp(0.5)) / math.sqrt((math.e - 1.0) * math.e)
    elif kind == "rare_responder":
        hit = rng.random(shape) < 0.05
        raw = rng.normal(0.0, 0.25, size=shape)
        z = raw + hit * 4.0
        z = (z - z.mean()) / z.std()
    elif kind == "single_outlier":
        z = rng.standard_normal(shape)
        trial_indices = np.arange(trials)
        strata = rng.integers(0, 8, size=trials)
        ranks = rng.integers(0, 6, size=trials)
        cells = rng.integers(0, 2, size=trials)
        z[trial_indices, strata, ranks, cells] += 8.0
        z = (z - z.mean()) / z.std()
    else:
        raise ValueError(kind)
    return mean[None, None, None, :] + sd * z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-trials", type=int, default=200_000)
    parser.add_argument("--stress-trials", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    truth = np.asarray([0.0, 0.0])
    sd = 0.5
    normal = {}
    for rho in (0.0, 0.5, 0.9):
        samples = correlated_normal(
            rng, args.normal_trials, truth, sd, rho)
        normal[str(rho)] = run_selector(samples, truth)
    stress = {}
    for kind in ("t3", "skew_lognormal", "rare_responder", "single_outlier"):
        samples = stress_samples(
            rng, args.stress_trials, kind, truth, sd)
        stress[kind] = run_selector(samples, truth)
    positive_truth = np.asarray([0.30, 0.20])
    positive = run_selector(
        correlated_normal(rng, args.normal_trials, positive_truth, sd, 0.5),
        positive_truth)
    result = {
        "schema": "coherent_state_powered_v13_statistical_simulation_v1",
        "design_id": "coherent-state-powered-successor-v13",
        "seed": args.seed,
        "normal_trials": args.normal_trials,
        "stress_trials": args.stress_trials,
        "alpha_ledger": alpha_ledger(),
        "normal_null": normal,
        "stress_null": stress,
        "normal_positive": positive,
        "interpretation": {
            "normal_gate": "simultaneous_noncoverage_rate <= 0.05 + Monte Carlo tolerance",
            "stress": "diagnostic; t undercoverage is reported and never relabeled distribution-free",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "normal_noncoverage": {
            key: row["simultaneous_noncoverage_rate"]
            for key, row in normal.items()
        },
        "positive_false_bound": positive["false_bound_rate"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
