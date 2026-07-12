#!/usr/bin/env python3
"""Recompute the paper-facing SWE-Gym metrics from committed score files.

This is a CPU-only, zero-model-call analysis.  It deliberately keeps two
questions separate:

1. the selected layer map on trajectories that did not select the map; and
2. the fixed scalar graft across the original and later index pools.

The output is a uniquely named JSON audit artifact containing every input path
and SHA-256, the statistical contract, invariant checks, results, and the
limitations that the saved score files cannot resolve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_SLUG = "Qwen3-30B-A3B-Instruct-2507"
DEFAULT_REPS = 10_000
DEFAULT_SEED = 0

DEFAULT_PATHS = {
    "original_run_1": "results/swegym_30b_bf16",
    "original_run_2": "results/swegym_30b_bf16_brief",
    "disjoint_profile": "results/swegym_tune_20260710T145330Z_brief",
    "disjoint_champion_eval": "results/swegym_champeval_20260710T145330Z_brief",
    "original_partial_confirmation": "results/swegym_confirm_20260711T011101Z_brief",
    "selected_map": "data/champion_configs/swegym_tuned_20260710T145330Z.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def percentile_bootstrap(
    values: Sequence[float], *, n_reps: int = DEFAULT_REPS, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    """Percentile bootstrap of a mean, matching analyze_swegym_tune.py.

    Quantile indices are ``int(.025*n_reps)`` and ``int(.975*n_reps)`` after
    sorting.  Values must already be in a stable order so a fixed RNG seed is
    byte-reproducible despite finite Monte Carlo sampling.
    """

    if not values:
        raise ValueError("bootstrap requires at least one value")
    if n_reps <= 0:
        raise ValueError("n_reps must be positive")
    n = len(values)
    rng = random.Random(seed)
    sampled = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_reps)
    ]
    sampled.sort()
    return {
        "n": n,
        "mean": mean(values),
        "ci_95_percentile": {
            "lower": sampled[int(0.025 * n_reps)],
            "upper": sampled[int(0.975 * n_reps)],
        },
        "n_positive": sum(value > 0 for value in values),
        "n_zero": sum(value == 0 for value in values),
        "n_negative": sum(value < 0 for value in values),
    }


def independent_pool_difference_bootstrap(
    original: Sequence[float],
    disjoint: Sequence[float],
    *,
    n_reps: int = DEFAULT_REPS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Bootstrap mean(disjoint) - mean(original), resampling pools separately."""

    if not original or not disjoint:
        raise ValueError("both pools must be non-empty")
    rng = random.Random(seed)
    sampled: list[float] = []
    for _ in range(n_reps):
        original_mean = sum(
            original[rng.randrange(len(original))] for _ in range(len(original))
        ) / len(original)
        disjoint_mean = sum(
            disjoint[rng.randrange(len(disjoint))] for _ in range(len(disjoint))
        ) / len(disjoint)
        sampled.append(disjoint_mean - original_mean)
    sampled.sort()
    return {
        "contrast": "mean(disjoint98) - mean(original75_repeat_averaged)",
        "mean_difference": mean(disjoint) - mean(original),
        "ci_95_percentile": {
            "lower": sampled[int(0.025 * n_reps)],
            "upper": sampled[int(0.975 * n_reps)],
        },
        "resampling": "independent within each pool on every replicate",
    }


@dataclass(frozen=True)
class Row:
    idx: int
    split: str | None
    path: Path
    arms: dict[str, dict[str, Any]]
    gold_has_action: bool
    model: str | None
    dtype: str | None


def load_rows(directory: Path) -> dict[int, Row]:
    rows: dict[int, Row] = {}
    paths = sorted(directory.glob("t*.json"))
    if not paths:
        raise ValueError(f"no t*.json score files in {directory}")
    for path in paths:
        doc = json.loads(path.read_text())
        idx = int(doc["idx"])
        if idx in rows:
            raise ValueError(f"duplicate idx {idx} in {directory}")
        arms = doc.get("arms")
        if not isinstance(arms, dict):
            raise ValueError(f"missing arms in {path}")
        rows[idx] = Row(
            idx=idx,
            split=doc.get("split"),
            path=path,
            arms=arms,
            gold_has_action=doc.get("gold_action") is not None,
            model=doc.get("model"),
            dtype=doc.get("dtype"),
        )
    return rows


def arm_tf(row: Row, arm: str) -> float:
    try:
        value = row.arms[arm]["tf_mean"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{row.path} lacks {arm}.tf_mean") from exc
    if not isinstance(value, (int, float)):
        raise ValueError(f"{row.path} has non-numeric {arm}.tf_mean")
    return float(value)


def deltas(rows: Iterable[Row], arm: str, reference: str = "B") -> list[float]:
    return [arm_tf(row, arm) - arm_tf(row, reference) for row in rows]


def action_match(row: Row, arm: str) -> bool:
    try:
        value = row.arms[arm]["action_match"]["match"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{row.path} lacks {arm}.action_match.match") from exc
    if not isinstance(value, bool):
        raise ValueError(f"{row.path} has non-boolean {arm}.action_match.match")
    return value


def structural_match_report(
    rows: Iterable[Row],
    *,
    arm: str = "E-champion",
    reference: str = "B",
    n_reps: int = DEFAULT_REPS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    eligible = sorted((row for row in rows if row.gold_has_action), key=lambda row: row.idx)
    if not eligible:
        raise ValueError("no rows with a saved gold action")
    reference_values = [action_match(row, reference) for row in eligible]
    arm_values = [action_match(row, arm) for row in eligible]
    paired = [int(a) - int(b) for a, b in zip(arm_values, reference_values)]
    return {
        "definition": (
            "saved composite demonstration-action structural match: tool plus "
            "path/command fields implemented by the historical scorer"
        ),
        "n": len(eligible),
        "reference_matches": sum(reference_values),
        "reference_rate": sum(reference_values) / len(eligible),
        "selected_map_matches": sum(arm_values),
        "selected_map_rate": sum(arm_values) / len(eligible),
        "selected_map_fixes": sum(a and not b for a, b in zip(arm_values, reference_values)),
        "selected_map_breaks": sum(b and not a for a, b in zip(arm_values, reference_values)),
        "paired_selected_minus_reference": percentile_bootstrap(
            paired, n_reps=n_reps, seed=seed
        ),
    }


def describe_input_directory(repo_root: Path, role: str, directory: Path) -> dict[str, Any]:
    score_paths = sorted(directory.glob("t*.json"))
    manifest = directory / "manifest.json"
    all_paths = sorted(score_paths + ([manifest] if manifest.exists() else []))
    files = [
        {"path": str(path.relative_to(repo_root)), "sha256": sha256_file(path)}
        for path in all_paths
    ]
    listing = "".join(f"{item['path']}\t{item['sha256']}\n" for item in files)
    return {
        "role": role,
        "directory": str(directory.relative_to(repo_root)),
        "score_file_count": len(score_paths),
        "manifest_present": manifest.exists(),
        "file_set_sha256": hashlib.sha256(listing.encode()).hexdigest(),
        "file_set_sha256_definition": "sha256 of sorted '<repo-path>\\t<file-sha256>\\n' rows",
        "files": files,
    }


def continuous_pair(
    rows: Sequence[Row], *, n_reps: int, seed: int
) -> dict[str, Any]:
    return {
        "selected_map_minus_baseline": percentile_bootstrap(
            deltas(rows, "E-champion", "B"), n_reps=n_reps, seed=seed
        ),
        "selected_map_minus_fixed_scalar": percentile_bootstrap(
            deltas(rows, "E-champion", "E-tuned"), n_reps=n_reps, seed=seed
        ),
        "fixed_scalar_minus_baseline": percentile_bootstrap(
            deltas(rows, "E-tuned", "B"), n_reps=n_reps, seed=seed
        ),
    }


def _assert_models(inputs: dict[str, dict[int, Row]]) -> None:
    observed = {
        row.model
        for rows in inputs.values()
        for row in rows.values()
        if row.model is not None
    }
    if observed != {MODEL_ID}:
        raise ValueError(f"wrong or heterogeneous subject model(s): {sorted(observed)}")


def _assert_equal_scalar_measurements(
    left: dict[int, Row], right: dict[int, Row]
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("profile and champion-eval IDs differ")
    differences = [
        (arm_tf(right[idx], "E-tuned") - arm_tf(right[idx], "B"))
        - (arm_tf(left[idx], "E-tuned") - arm_tf(left[idx], "B"))
        for idx in sorted(left)
    ]
    if any(value != 0.0 for value in differences):
        raise ValueError("scalar measurements changed between disjoint profile and champion-eval")
    return {
        "n": len(differences),
        "all_per_trajectory_deltas_exactly_equal_as_parsed_floats": True,
        "max_absolute_difference": max(map(abs, differences), default=0.0),
        "counting_rule": "count once, from disjoint_profile; do not double-count the identical replay",
    }


def _assert_selected_map_references(
    row_sets: Sequence[dict[int, Row]], selected_map_sha256: str
) -> dict[str, Any]:
    checked = 0
    for rows in row_sets:
        for row in rows.values():
            try:
                recorded = row.arms["E-champion"]["intervention"]["champion"][
                    "config_sha256"
                ]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"{row.path} lacks an E-champion selected-map hash"
                ) from exc
            if recorded != selected_map_sha256:
                raise ValueError(
                    f"{row.path} references selected-map hash {recorded}, expected "
                    f"{selected_map_sha256}"
                )
            checked += 1
    return {
        "rows_checked": checked,
        "all_E_champion_rows_reference_frozen_map_sha256": True,
        "selected_map_sha256": selected_map_sha256,
    }


def git_head(repo_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_report(
    repo_root: Path,
    *,
    generated_at_utc: str,
    n_reps: int = DEFAULT_REPS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    paths = {key: repo_root / value for key, value in DEFAULT_PATHS.items()}
    row_sets = {
        key: load_rows(paths[key])
        for key in (
            "original_run_1",
            "original_run_2",
            "disjoint_profile",
            "disjoint_champion_eval",
            "original_partial_confirmation",
        )
    }
    _assert_models(row_sets)

    original_1 = row_sets["original_run_1"]
    original_2 = row_sets["original_run_2"]
    profile = row_sets["disjoint_profile"]
    champion_eval = row_sets["disjoint_champion_eval"]
    confirmation = row_sets["original_partial_confirmation"]

    if set(original_1) != set(original_2) or len(original_1) != 75:
        raise ValueError("the two original runs must contain the same 75 trajectory IDs")
    if len(profile) != 98 or len(champion_eval) != 98:
        raise ValueError("the later index pool must contain 98 trajectories")
    if set(original_1) & set(profile):
        raise ValueError("original and later index pools overlap")
    if len(set(original_1) | set(profile)) != 173:
        raise ValueError("the unique fixed-scalar pool must contain 173 trajectories")
    if not set(confirmation).issubset(set(original_1)) or len(confirmation) != 45:
        raise ValueError("partial confirmation must contain 45 original-pool trajectories")

    fit_ids = {idx for idx, row in profile.items() if row.split == "tune"}
    fresh_eval_ids = {idx for idx, row in champion_eval.items() if row.split == "eval"}
    if len(fit_ids) != 41 or len(fresh_eval_ids) != 57:
        raise ValueError("expected 41 map-fitting and 57 fresh evaluation trajectories")
    if fit_ids & fresh_eval_ids:
        raise ValueError("fresh evaluation overlaps map-fitting IDs")
    if set(confirmation) & set(profile):
        raise ValueError("original-pool confirmation overlaps the map-fitting pool")

    selected_map_path = paths["selected_map"]
    selected_map_sha = sha256_file(selected_map_path)
    selected_map = json.loads(selected_map_path.read_text())
    if selected_map_sha != "6faa2d7227d46c86f0c614bef808d0424d4f124203a9dc7f7558e2b07e994d53":
        raise ValueError("selected-map hash differs from the recorded frozen map")

    fresh_rows = [champion_eval[idx] for idx in sorted(fresh_eval_ids)]
    confirmation_rows = [confirmation[idx] for idx in sorted(confirmation)]
    pooled_selected_rows = fresh_rows + confirmation_rows

    # The same original trajectories were rendered twice.  Average their
    # per-trajectory treatment deltas first, so the inferential N remains 75.
    original_run_1_rows = [original_1[idx] for idx in sorted(original_1)]
    original_run_2_rows = [original_2[idx] for idx in sorted(original_2)]
    disjoint_rows = [profile[idx] for idx in sorted(profile)]
    original_run_1_values = deltas(original_run_1_rows, "E-tuned", "B")
    original_run_2_values = deltas(original_run_2_rows, "E-tuned", "B")
    original_average_values = [
        (
            arm_tf(original_1[idx], "E-tuned")
            - arm_tf(original_1[idx], "B")
            + arm_tf(original_2[idx], "E-tuned")
            - arm_tf(original_2[idx], "B")
        )
        / 2.0
        for idx in sorted(original_1)
    ]
    disjoint_values = deltas(disjoint_rows, "E-tuned", "B")
    pooled_scalar_values = original_average_values + disjoint_values

    directory_inputs = [
        describe_input_directory(repo_root, key, paths[key])
        for key in (
            "original_run_1",
            "original_run_2",
            "disjoint_profile",
            "disjoint_champion_eval",
            "original_partial_confirmation",
        )
    ]
    map_input = {
        "role": "selected_map",
        "path": str(selected_map_path.relative_to(repo_root)),
        "sha256": selected_map_sha,
    }
    program_path = Path(__file__).resolve()

    return {
        "schema": "swegym-paper-metrics-reanalysis/v1",
        "generated_at_utc": generated_at_utc,
        "cost": {
            "new_gpu_spend_usd": 0.0,
            "external_model_calls": 0,
            "description": "CPU-only arithmetic over already committed JSON score files",
        },
        "analysis_program": {
            "path": str(program_path.relative_to(repo_root)),
            "sha256": sha256_file(program_path),
            "repository_head_at_analysis": git_head(repo_root),
        },
        "subject": {
            "model_id": MODEL_ID,
            "resolved_revision_where_recorded": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
            "load_dtype_where_recorded": "torch.bfloat16",
        },
        "statistical_contract": {
            "continuous_metric": (
                "per trajectory: arm.tf_mean - reference.tf_mean, in nats per target token; "
                "tf_mean is the saved teacher-forced mean log-probability of the demonstrated next action"
            ),
            "unit": (
                "one saved trajectory score row; every component is sorted numerically by saved dataset "
                "idx before bootstrap, and pooled components are concatenated in the stated order"
            ),
            "bootstrap": {
                "method": "ordinary percentile bootstrap of the arithmetic mean, sampling trajectories with replacement",
                "replicates": n_reps,
                "seed": seed,
                "rng": "Python random.Random",
                "lower_index": int(0.025 * n_reps),
                "upper_index": int(0.975 * n_reps),
                "multiplicity_adjustment": None,
            },
            "repeated_original_rule": (
                "average the two E-tuned-minus-B measurements within each of the same 75 IDs, "
                "then bootstrap 75 trajectory averages"
            ),
            "pooled_scalar_rule": (
                "concatenate 75 repeat-averaged original-pool deltas with 98 disjoint-pool deltas; N=173"
            ),
            "selected_map_pooling_rule": (
                "concatenate fresh-pool eval-split 57 with completed original-pool confirmation 45; "
                "both sets are disjoint from the 41 IDs used to select the map; N=102"
            ),
        },
        "inputs": directory_inputs + [map_input],
        "invariant_checks": {
            "subject_model_uniform": True,
            "original_run_ids_equal": True,
            "original_unique_n": 75,
            "later_unique_n": 98,
            "original_later_overlap_n": 0,
            "unique_fixed_scalar_n": 173,
            "map_fitting_n": len(fit_ids),
            "fresh_evaluation_n": len(fresh_eval_ids),
            "fit_fresh_overlap_n": len(fit_ids & fresh_eval_ids),
            "partial_original_confirmation_n": len(confirmation),
            "fit_confirmation_overlap_n": len(fit_ids & set(confirmation)),
            "selected_map_sha256_verified": True,
            "selected_map_alpha_map": selected_map.get("alpha_map"),
            "selected_map_row_references": _assert_selected_map_references(
                [champion_eval, confirmation], selected_map_sha
            ),
            "later_scalar_duplicate_measurement": _assert_equal_scalar_measurements(
                profile, champion_eval
            ),
        },
        "selected_map_out_of_fitting": {
            "fresh57": {
                "source": DEFAULT_PATHS["disjoint_champion_eval"],
                "selection_exclusion": "split=eval; disjoint from 41 split=tune map-fitting IDs",
                "continuous": continuous_pair(fresh_rows, n_reps=n_reps, seed=seed),
                "structural_match_vs_baseline": structural_match_report(
                    fresh_rows, n_reps=n_reps, seed=seed
                ),
            },
            "original45_partial_confirmation": {
                "source": DEFAULT_PATHS["original_partial_confirmation"],
                "selection_exclusion": "original index pool; wholly disjoint from later-pool map fitting",
                "continuous": continuous_pair(
                    confirmation_rows, n_reps=n_reps, seed=seed
                ),
                "structural_match_vs_baseline": structural_match_report(
                    confirmation_rows, n_reps=n_reps, seed=seed
                ),
            },
            "pooled102": {
                "source_components": ["fresh57", "original45_partial_confirmation"],
                "continuous": continuous_pair(
                    pooled_selected_rows, n_reps=n_reps, seed=seed
                ),
                "structural_match_vs_baseline": structural_match_report(
                    pooled_selected_rows, n_reps=n_reps, seed=seed
                ),
            },
        },
        "fixed_scalar_alpha_0_75": {
            "original_run_1_75": percentile_bootstrap(
                original_run_1_values, n_reps=n_reps, seed=seed
            ),
            "original_run_2_75": percentile_bootstrap(
                original_run_2_values, n_reps=n_reps, seed=seed
            ),
            "original75_repeat_averaged": percentile_bootstrap(
                original_average_values, n_reps=n_reps, seed=seed
            ),
            "disjoint98": percentile_bootstrap(
                disjoint_values, n_reps=n_reps, seed=seed
            ),
            "unique_pooled173": percentile_bootstrap(
                pooled_scalar_values, n_reps=n_reps, seed=seed
            ),
            "pool_heterogeneity": independent_pool_difference_bootstrap(
                original_average_values,
                disjoint_values,
                n_reps=n_reps,
                seed=seed,
            ),
        },
        "missing_data_and_interpretation_caveats": [
            (
                "The original_run_1 directory has no manifest and its rows omit dtype. Its bf16/brief "
                "classification is historical/path-level provenance, not born-annotated proof in these files."
            ),
            (
                "The two original runs repeat the same 75 trajectory IDs; they are averaged within ID and "
                "never treated as 150 independent observations."
            ),
            (
                "The disjoint profile and champion-eval directories contain exactly identical saved scalar "
                "E-tuned-minus-B deltas for all 98 IDs; this analysis counts that scalar measurement once."
            ),
            (
                "The original-pool selected-map job stopped after 45 of the requested 75 rows. Its manifest "
                "does not contain a finalized instance-id list; the actual 45 t*.json files define the result."
            ),
            (
                "No selected-map-matched placebo was run. Scalar/layer-sweep placebos do not identify whether "
                "the selected-map lead is recovered hidden-history information or map-specific disturbance."
            ),
            (
                "Structural match is agreement with the demonstrated next action under the historical parser, "
                "not unique correctness, action execution, patch application, tests, or task success."
            ),
            (
                "The bootstrap treats trajectory rows as independent. Saved score JSONs do not carry a stable "
                "underlying task/repository cluster ID, so task-clustered uncertainty is not recomputed here."
            ),
            (
                "Per-trajectory summary text, exact target text/token IDs, K/V tensors, and upstream row-level "
                "trajectory-generator identity were not saved, so this is score recomputation rather than an "
                "independent rerun of model inference."
            ),
            (
                "The confirmation subset is a stopped prefix, not a randomized completion of the planned 75; "
                "pooling 57+45 is descriptive wholly out-of-fitting evidence, not a preregistered meta-analysis."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--timestamp",
        default=None,
        help="UTC YYYYMMDDTHHMMSSZ stamp (default: current UTC)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    stamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generated_at = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    ).isoformat().replace("+00:00", "Z")
    output = args.output
    if output is None:
        output = (
            repo_root
            / "results"
            / "swegym_paper_reanalysis"
            / f"swegym_paper_metrics_{MODEL_SLUG}_{stamp}.json"
        )
    elif not output.is_absolute():
        output = repo_root / output

    print(f"RUN swegym-paper-metrics model={MODEL_ID} -> {output}", flush=True)
    report = build_report(
        repo_root,
        generated_at_utc=generated_at,
        n_reps=args.bootstrap_reps,
        seed=args.bootstrap_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"WROTE {output} sha256={sha256_file(output)}", flush=True)


if __name__ == "__main__":
    main()
