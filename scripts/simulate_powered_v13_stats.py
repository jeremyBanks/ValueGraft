#!/usr/bin/env python3
"""Pre-outcome coverage/power simulation for the corrected v13 bounded rule."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from powered_v13_stats import (  # noqa: E402
    CLIP_LOWER, CLIP_UPPER, DELTA_CLIP, DELTA_TAIL, FINAL_N,
    MEAN_FAMILY_ALPHA, TAIL_ALPHA, alpha_ledger,
)


MEAN_GAMMA = MEAN_FAMILY_ALPHA / 2.0
RADIUS = (CLIP_UPPER - CLIP_LOWER) * math.sqrt(
    math.log(1.0 / MEAN_GAMMA) / (2.0 * FINAL_N))


def bounded_analysis(raw: np.ndarray):
    """Independent vectorized implementation; raw=[trial,N,2]."""
    clipped = np.clip(raw, CLIP_LOWER, CLIP_UPPER)
    ucbs = np.minimum(CLIP_UPPER, clipped.mean(axis=1) + RADIUS)
    responders = (raw.max(axis=2) > CLIP_UPPER).sum(axis=1)
    zero_ucb = 1.0 - TAIL_ALPHA ** (1.0 / FINAL_N)
    nonzero_ucb = np.minimum(
        1.0,
        responders / FINAL_N + math.sqrt(
            math.log(1.0 / TAIL_ALPHA) / (2.0 * FINAL_N)))
    tail_ucb = np.where(responders == 0, zero_ucb, nonzero_ucb)
    return ucbs, responders, tail_ucb


def evaluate(raw, true_clipped_means, true_tail_mean):
    ucbs, responders, tail_ucb = bounded_analysis(raw)
    mean_noncoverage = np.any(
        ucbs < np.asarray(true_clipped_means)[None, :], axis=1)
    tail_noncoverage = tail_ucb < true_tail_mean
    joint_noncoverage = mean_noncoverage | tail_noncoverage
    resolved = ((ucbs.max(axis=1) <= DELTA_CLIP) &
                (tail_ucb <= DELTA_TAIL))
    return {
        "mean_noncoverage_rate": float(mean_noncoverage.mean()),
        "tail_noncoverage_rate": float(tail_noncoverage.mean()),
        "joint_noncoverage_rate": float(joint_noncoverage.mean()),
        "joint_resolution_rate": float(resolved.mean()),
        "zero_responder_rate": float((responders == 0).mean()),
        "mean_primary_ucb": float(ucbs.max(axis=1).mean()),
        "mean_tail_ucb": float(tail_ucb.mean()),
    }


def correlated_uniform(rng, shape, rho):
    z1 = rng.standard_normal(shape)
    z2 = rho * z1 + math.sqrt(1.0 - rho * rho) * rng.standard_normal(shape)
    return stats.norm.cdf(np.stack([z1, z2], axis=-1))


def scenarios(rng, trials):
    result = {}
    # Symmetric bounded uniform with exact clipped mean zero, no responders.
    for rho in (0.0, 0.5, 0.9):
        u = correlated_uniform(rng, (trials, FINAL_N), rho)
        raw = u - 0.5
        result[f"uniform_rho_{rho}"] = evaluate(raw, [0.0, 0.0], 0.0)

    # Skewed beta has known bounded mean 2/7 - 1/2.
    raw = rng.beta(2.0, 5.0, size=(trials, FINAL_N, 2)) - 0.5
    beta_mean = 2.0 / 7.0 - 0.5
    result["skew_beta"] = evaluate(raw, [beta_mean, beta_mean], 0.0)

    # Heavy-tailed symmetric raw values; clipping preserves mean zero.
    raw = rng.standard_t(3, size=(trials, FINAL_N, 2))
    result["t3_clipped"] = evaluate(raw, [0.0, 0.0],
                                     float(1.0 - stats.t.cdf(0.5, 3) ** 2))

    # Old-draft counterexample: each cell is 5 with p=.06 else zero.
    hit = rng.random((trials, FINAL_N, 2)) < 0.06
    raw = hit.astype(np.float64) * 5.0
    result["rare_responder_p006"] = evaluate(
        raw, [0.03, 0.03], 1.0 - 0.94 ** 2)

    # One-percent positive contamination per cell.
    hit = rng.random((trials, FINAL_N, 2)) < 0.01
    raw = hit.astype(np.float64) * 5.0
    result["positive_contamination_p001"] = evaluate(
        raw, [0.005, 0.005], 1.0 - 0.99 ** 2)
    return result


def power_scenarios(rng, trials):
    # Worst-variance bounded endpoints on {-0.5,+0.5}, mean .05 => p=.55.
    worst = np.where(
        rng.random((trials, FINAL_N, 2)) < 0.55, 0.5, -0.5)
    worst_result = evaluate(worst, [0.05, 0.05], 0.0)

    # Mean zero, SD .25, no responders.
    centered = np.clip(
        rng.normal(0.0, 0.25, size=(trials, FINAL_N, 2)), -0.5, 0.5)
    centered_result = evaluate(centered, [0.0, 0.0], 0.0)
    return {"worst_variance_mean_005": worst_result,
            "mean_zero_sd_025": centered_result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    coverage = scenarios(rng, args.trials)
    power = power_scenarios(rng, args.trials)
    mc_tolerance = 0.05 + stats.norm.ppf(0.995) * math.sqrt(
        0.05 * 0.95 / args.trials)
    coverage_pass = all(
        row["joint_noncoverage_rate"] <= mc_tolerance
        for row in coverage.values())
    power_pass = (
        power["worst_variance_mean_005"]["joint_resolution_rate"] >= 0.80
        and power["mean_zero_sd_025"]["joint_resolution_rate"] >= 0.95)
    result = {
        "schema": "coherent_state_powered_v13_bounded_simulation_v2",
        "design_id": "coherent-state-powered-successor-v13",
        "seed": args.seed,
        "trials_per_scenario": args.trials,
        "formula": {
            "clip": [CLIP_LOWER, CLIP_UPPER],
            "mean_gamma_per_cell": MEAN_GAMMA,
            "mean_radius": RADIUS,
            "tail_alpha": TAIL_ALPHA,
            "delta_clip": DELTA_CLIP,
            "delta_tail": DELTA_TAIL,
        },
        "alpha_ledger": alpha_ledger(),
        "monte_carlo_joint_noncoverage_tolerance_99pct": mc_tolerance,
        "coverage": coverage,
        "power": power,
        "gates": {
            "coverage_pass": coverage_pass,
            "power_pass": power_pass,
            "all_pass": coverage_pass and power_pass,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "coverage_pass": coverage_pass,
        "power_pass": power_pass,
        "worst_power": power["worst_variance_mean_005"]["joint_resolution_rate"],
        "zero_sd_power": power["mean_zero_sd_025"]["joint_resolution_rate"],
        "max_noncoverage": max(row["joint_noncoverage_rate"]
                               for row in coverage.values()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
