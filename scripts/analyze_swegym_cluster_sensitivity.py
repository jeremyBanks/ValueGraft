#!/usr/bin/env python3
"""Recover observable SWE-Gym task/repository clusters and recompute uncertainty.

This is a CPU-only, zero-model-call analysis over the local, ignored
``swegym.parquet`` and already committed score rows.  It binds the join to the
exact parquet SHA-256 used by the paper, derives content-addressed task IDs from
the initial user task, and never serializes literal task or system-prompt text.

The primary sensitivity is a one-stage pairs cluster bootstrap over exact-task
clusters for the selected-map-minus-B mean.  Versioned repository snapshots and
normalized owner/repository keys are reported as coarser sensitivity analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_SLUG = "Qwen3-30B-A3B-Instruct-2507"
EXPECTED_PARQUET_SHA256 = (
    "ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1"
)
EXPECTED_SYSTEM_SHA256 = (
    "1120aa8819abb372428afb82f6a5f49d1d243e4bf58cb27fd481809acd339e84"
)
EXPECTED_SELECTED_MAP_SHA256 = (
    "6faa2d7227d46c86f0c614bef808d0424d4f124203a9dc7f7558e2b07e994d53"
)
DEFAULT_REPS = 10_000
DEFAULT_SEED = 0

DEFAULT_PATHS = {
    "parquet": "swegym.parquet",
    "map_fitting": "results/swegym_tune_20260710T145330Z_brief",
    "selected_map_evaluation": "results/swegym_champeval_20260710T145330Z_brief",
    "partial_confirmation": "results/swegym_confirm_20260711T011101Z_brief",
    "original_pool": "results/swegym_30b_bf16_brief",
    "selected_map": "data/champion_configs/swegym_tuned_20260710T145330Z.json",
}

UPLOADED_PREFIX_RE = re.compile(
    r"\A<uploaded_files>\r?\n/workspace/(?P<snapshot>[^/\r\n]+)"
    r"\r?\n</uploaded_files>"
)
WORKSPACE_SNAPSHOT_RE = re.compile(
    r"/workspace/([A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+)"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arithmetic_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


@dataclass(frozen=True)
class TrajectoryIdentity:
    dataset_idx: int
    owner: str
    repository: str
    version: str
    snapshot_key: str
    repository_key: str
    task_sha256: str
    pr_description_sha256: str
    initial_user_sha256: str
    trajectory_messages_sha256: str
    system_sha256: str
    later_assistant_repeats_exact_workspace_path: bool

    def serialized(self) -> dict[str, Any]:
        """Return only hashes and normalized identity keys, never literal tasks."""

        return {
            "dataset_idx": self.dataset_idx,
            "repository": {
                "owner": self.owner,
                "name": self.repository,
                "version": self.version,
                "repository_key": self.repository_key,
                "snapshot_key": self.snapshot_key,
            },
            "task_sha256": self.task_sha256,
            "pr_description_sha256": self.pr_description_sha256,
            "initial_user_sha256": self.initial_user_sha256,
            "trajectory_messages_sha256": self.trajectory_messages_sha256,
            "system_sha256": self.system_sha256,
            "later_assistant_repeats_exact_workspace_path": (
                self.later_assistant_repeats_exact_workspace_path
            ),
        }


def _first_pr_description(initial_user: str) -> str:
    start_tag = "<pr_description>"
    end_tag = "</pr_description>"
    start = initial_user.find(start_tag)
    if start < 0:
        raise ValueError("initial user task lacks <pr_description>")
    start += len(start_tag)
    end = initial_user.find(end_tag, start)
    if end < 0:
        raise ValueError("initial user task lacks </pr_description>")
    # Match the historical task scaffold's formatting while preserving all
    # internal bytes.  Leading/trailing whitespace is scaffold punctuation, not
    # task identity; stripping it also makes the formula explicit and stable.
    return initial_user[start:end].strip()


def parse_trajectory_identity(
    dataset_idx: int, messages: Sequence[dict[str, Any]]
) -> TrajectoryIdentity:
    """Parse the explicit uploaded snapshot and a content-addressed task key.

    The observable task key is SHA-256 of
    ``snapshot_key + U+0000 + exact stripped first PR-description block``.
    It is an artifact-stable surrogate for dependence clustering, not a claim to
    recover an absent upstream SWE task/PR identifier.
    """

    if len(messages) < 2:
        raise ValueError(f"row {dataset_idx}: fewer than two messages")
    if messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ValueError(f"row {dataset_idx}: expected initial system,user roles")
    systems = [m for m in messages if m.get("role") == "system"]
    if len(systems) != 1:
        raise ValueError(f"row {dataset_idx}: expected exactly one system message")

    system = systems[0].get("content")
    initial_user = messages[1].get("content")
    if not isinstance(system, str) or not isinstance(initial_user, str):
        raise ValueError(f"row {dataset_idx}: non-string initial content")

    match = UPLOADED_PREFIX_RE.match(initial_user)
    if not match:
        raise ValueError(f"row {dataset_idx}: malformed uploaded-files prefix")
    snapshot_key = match.group("snapshot")
    components = snapshot_key.split("__")
    if len(components) != 3 or any(not component for component in components):
        raise ValueError(
            f"row {dataset_idx}: snapshot key must be owner__repo__version"
        )
    owner, repository, version = components
    repository_key = f"{owner}/{repository}"

    pr_description = _first_pr_description(initial_user)
    task_sha256 = sha256_text(f"{snapshot_key}\0{pr_description}")
    exact_workspace_path = f"/workspace/{snapshot_key}"
    later_assistant = "\n".join(
        str(message.get("content", ""))
        for message in messages[2:]
        if message.get("role") == "assistant"
    )
    canonical_messages = json.dumps(
        list(messages), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )

    return TrajectoryIdentity(
        dataset_idx=dataset_idx,
        owner=owner,
        repository=repository,
        version=version,
        snapshot_key=snapshot_key,
        repository_key=repository_key,
        task_sha256=task_sha256,
        pr_description_sha256=sha256_text(pr_description),
        initial_user_sha256=sha256_text(initial_user),
        trajectory_messages_sha256=sha256_text(canonical_messages),
        system_sha256=sha256_text(system),
        later_assistant_repeats_exact_workspace_path=(
            exact_workspace_path in later_assistant
        ),
    )


def load_identities(parquet_path: Path) -> list[TrajectoryIdentity]:
    frame = pd.read_parquet(parquet_path)
    if list(frame.columns) != ["messages"]:
        raise ValueError(
            f"expected parquet to contain only messages, got {list(frame.columns)}"
        )
    identities: list[TrajectoryIdentity] = []
    for idx, raw_messages in enumerate(frame["messages"].tolist()):
        identities.append(parse_trajectory_identity(idx, list(raw_messages)))
    return identities


@dataclass(frozen=True)
class ScoreRow:
    idx: int
    split: str | None
    path: Path
    model: str | None
    arms: dict[str, dict[str, Any]]
    gold_action: dict[str, Any] | None


def load_score_rows(directory: Path) -> dict[int, ScoreRow]:
    rows: dict[int, ScoreRow] = {}
    paths = sorted(directory.glob("t*.json"))
    if not paths:
        raise ValueError(f"no t*.json rows in {directory}")
    for path in paths:
        doc = json.loads(path.read_text())
        idx = int(doc["idx"])
        if path.stem != f"t{idx:04d}":
            raise ValueError(f"filename/index mismatch: {path} vs {idx}")
        if idx in rows:
            raise ValueError(f"duplicate score row {idx} in {directory}")
        arms = doc.get("arms")
        if not isinstance(arms, dict):
            raise ValueError(f"missing arms in {path}")
        rows[idx] = ScoreRow(
            idx=idx,
            split=doc.get("split"),
            path=path,
            model=doc.get("model"),
            arms=arms,
            gold_action=doc.get("gold_action"),
        )
    return rows


def arm_tf(row: ScoreRow, arm: str) -> float:
    try:
        value = row.arms[arm]["tf_mean"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{row.path} lacks {arm}.tf_mean") from exc
    if not isinstance(value, (int, float)):
        raise ValueError(f"{row.path} has non-numeric {arm}.tf_mean")
    return float(value)


def assert_selected_map_reference(row: ScoreRow, selected_map_sha256: str) -> None:
    try:
        observed = row.arms["E-champion"]["intervention"]["champion"][
            "config_sha256"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{row.path} lacks selected-map provenance") from exc
    if observed != selected_map_sha256:
        raise ValueError(
            f"{row.path} selected-map hash {observed}, expected {selected_map_sha256}"
        )


@dataclass(frozen=True)
class AnalysisRow:
    component: str
    dataset_idx: int
    value: float
    task_sha256: str
    snapshot_key: str
    repository_key: str


def sign_counts(values: Sequence[float]) -> dict[str, int]:
    return {
        "n_positive": sum(value > 0 for value in values),
        "n_zero": sum(value == 0 for value in values),
        "n_negative": sum(value < 0 for value in values),
    }


def _percentile_interval(samples: list[float], n_reps: int) -> dict[str, float]:
    samples.sort()
    return {
        "lower": samples[int(0.025 * n_reps)],
        "upper": samples[int(0.975 * n_reps)],
    }


def iid_percentile_bootstrap(
    rows: Sequence[AnalysisRow], *, n_reps: int, seed: int
) -> dict[str, Any]:
    if not rows:
        raise ValueError("iid bootstrap needs rows")
    values = [row.value for row in rows]
    n = len(values)
    rng = random.Random(seed)
    samples = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_reps)
    ]
    return {
        "method": "ordinary trajectory-level percentile bootstrap",
        "n_rows": n,
        "mean": arithmetic_mean(values),
        "ci_95_percentile": _percentile_interval(samples, n_reps),
        **sign_counts(values),
    }


def _cluster_groups(
    rows: Sequence[AnalysisRow], key: str
) -> dict[str, list[AnalysisRow]]:
    if key not in {"task_sha256", "snapshot_key", "repository_key"}:
        raise ValueError(f"unsupported cluster key: {key}")
    groups: dict[str, list[AnalysisRow]] = defaultdict(list)
    for row in rows:
        groups[str(getattr(row, key))].append(row)
    return dict(groups)


def cluster_percentile_bootstrap(
    rows: Sequence[AnalysisRow],
    *,
    key: str,
    n_reps: int,
    seed: int,
    estimand: str = "row_weighted",
) -> dict[str, Any]:
    """One-stage pairs bootstrap of whole clusters.

    ``row_weighted`` retains the paper's trajectory-weighted mean: each sampled
    cluster contributes all member rows, and the replicate divides by the total
    number of contributed rows.  ``cluster_equal`` instead averages one mean per
    sampled cluster and is reported only as an alternative-estimand sensitivity.
    """

    if not rows:
        raise ValueError("cluster bootstrap needs rows")
    if estimand not in {"row_weighted", "cluster_equal"}:
        raise ValueError(f"unknown estimand: {estimand}")
    groups = _cluster_groups(rows, key)
    cluster_keys = sorted(groups)
    cluster_stats = [
        (
            sum(row.value for row in groups[cluster_key]),
            len(groups[cluster_key]),
        )
        for cluster_key in cluster_keys
    ]
    n_clusters = len(cluster_stats)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_reps):
        chosen = [cluster_stats[rng.randrange(n_clusters)] for _ in range(n_clusters)]
        if estimand == "row_weighted":
            samples.append(
                sum(cluster_sum for cluster_sum, _ in chosen)
                / sum(cluster_n for _, cluster_n in chosen)
            )
        else:
            samples.append(
                sum(cluster_sum / cluster_n for cluster_sum, cluster_n in chosen)
                / n_clusters
            )

    values = [row.value for row in rows]
    if estimand == "row_weighted":
        point = arithmetic_mean(values)
    else:
        point = arithmetic_mean(
            [cluster_sum / cluster_n for cluster_sum, cluster_n in cluster_stats]
        )
    sizes = [cluster_n for _, cluster_n in cluster_stats]
    return {
        "method": "one-stage whole-cluster pairs percentile bootstrap",
        "cluster_key": key,
        "estimand": estimand,
        "n_rows": len(rows),
        "n_clusters": n_clusters,
        "mean": point,
        "ci_95_percentile": _percentile_interval(samples, n_reps),
        "cluster_size": {
            "minimum": min(sizes),
            "maximum": max(sizes),
            "histogram": {str(k): v for k, v in sorted(Counter(sizes).items())},
        },
        **sign_counts(values),
    }


def analysis_bundle(
    rows: Sequence[AnalysisRow], *, n_reps: int, seed: int
) -> dict[str, Any]:
    return {
        "iid_trajectory": iid_percentile_bootstrap(
            rows, n_reps=n_reps, seed=seed
        ),
        "task_cluster_row_weighted": cluster_percentile_bootstrap(
            rows,
            key="task_sha256",
            n_reps=n_reps,
            seed=seed,
            estimand="row_weighted",
        ),
        "task_cluster_equal_weight_sensitivity": cluster_percentile_bootstrap(
            rows,
            key="task_sha256",
            n_reps=n_reps,
            seed=seed,
            estimand="cluster_equal",
        ),
        "snapshot_cluster_row_weighted_sensitivity": cluster_percentile_bootstrap(
            rows,
            key="snapshot_key",
            n_reps=n_reps,
            seed=seed,
            estimand="row_weighted",
        ),
        "repository_cluster_row_weighted_sensitivity": cluster_percentile_bootstrap(
            rows,
            key="repository_key",
            n_reps=n_reps,
            seed=seed,
            estimand="row_weighted",
        ),
        "repository_cluster_equal_weight_sensitivity": cluster_percentile_bootstrap(
            rows,
            key="repository_key",
            n_reps=n_reps,
            seed=seed,
            estimand="cluster_equal",
        ),
    }


def describe_directory(repo_root: Path, role: str, directory: Path) -> dict[str, Any]:
    score_paths = sorted(directory.glob("t*.json"))
    manifest = directory / "manifest.json"
    paths = score_paths + ([manifest] if manifest.exists() else [])
    files = [
        {"path": str(path.relative_to(repo_root)), "sha256": sha256_file(path)}
        for path in sorted(paths)
    ]
    listing = "".join(
        f"{item['path']}\t{item['sha256']}\n" for item in files
    )
    return {
        "role": role,
        "directory": str(directory.relative_to(repo_root)),
        "score_file_count": len(score_paths),
        "manifest_present": manifest.exists(),
        "file_set_sha256": sha256_text(listing),
        "file_set_sha256_definition": (
            "sha256 of sorted '<repo-path>\\t<file-sha256>\\n' rows"
        ),
        "files": files,
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


def _partition_equivalent(
    identities: Sequence[TrajectoryIdentity], left: str, right: str
) -> bool:
    left_to_right: dict[str, set[str]] = defaultdict(set)
    right_to_left: dict[str, set[str]] = defaultdict(set)
    for identity in identities:
        left_value = str(getattr(identity, left))
        right_value = str(getattr(identity, right))
        left_to_right[left_value].add(right_value)
        right_to_left[right_value].add(left_value)
    return all(len(values) == 1 for values in left_to_right.values()) and all(
        len(values) == 1 for values in right_to_left.values()
    )


def _set_overlap_report(
    ids: Sequence[int],
    identities: Sequence[TrajectoryIdentity],
    fit_tasks: set[str],
) -> dict[str, Any]:
    tasks = {identities[idx].task_sha256 for idx in ids}
    snapshots = {identities[idx].snapshot_key for idx in ids}
    repositories = {identities[idx].repository_key for idx in ids}
    return {
        "n_rows": len(ids),
        "n_tasks": len(tasks),
        "n_snapshots": len(snapshots),
        "n_repositories": len(repositories),
        "task_clusters_also_in_map_fit": len(tasks & fit_tasks),
        "rows_whose_task_also_in_map_fit": sum(
            identities[idx].task_sha256 in fit_tasks for idx in ids
        ),
        "overlapping_fit_task_sha256": sorted(tasks & fit_tasks),
    }


def _cluster_summaries(
    rows: Sequence[AnalysisRow], key: str
) -> list[dict[str, Any]]:
    groups = _cluster_groups(rows, key)
    return [
        {
            "cluster_key": cluster_key,
            "n_rows": len(groups[cluster_key]),
            "mean_selected_map_minus_B": arithmetic_mean(
                [row.value for row in groups[cluster_key]]
            ),
        }
        for cluster_key in sorted(groups)
    ]


def build_report(
    repo_root: Path,
    *,
    parquet_path: Path | None = None,
    generated_at_utc: str,
    n_reps: int = DEFAULT_REPS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    paths = {key: repo_root / value for key, value in DEFAULT_PATHS.items()}
    if parquet_path is not None:
        paths["parquet"] = parquet_path.resolve()
    if n_reps <= 0:
        raise ValueError("bootstrap reps must be positive")

    parquet_sha256 = sha256_file(paths["parquet"])
    if parquet_sha256 != EXPECTED_PARQUET_SHA256:
        raise ValueError(
            f"parquet SHA-256 {parquet_sha256}, expected {EXPECTED_PARQUET_SHA256}"
        )
    identities = load_identities(paths["parquet"])
    if len(identities) != 491:
        raise ValueError(f"expected 491 trajectories, got {len(identities)}")
    system_hashes = Counter(identity.system_sha256 for identity in identities)
    if system_hashes != Counter({EXPECTED_SYSTEM_SHA256: 491}):
        raise ValueError(f"unexpected system-prompt hashes: {system_hashes}")

    rows = {
        role: load_score_rows(paths[role])
        for role in (
            "map_fitting",
            "selected_map_evaluation",
            "partial_confirmation",
            "original_pool",
        )
    }
    profile = rows["map_fitting"]
    evaluation = rows["selected_map_evaluation"]
    confirmation = rows["partial_confirmation"]
    original_pool = rows["original_pool"]
    if set(profile) != set(evaluation) or len(profile) != 98:
        raise ValueError("map-fitting and evaluation runs must share 98 IDs")
    if any(profile[idx].split != evaluation[idx].split for idx in profile):
        raise ValueError("map-fitting and evaluation split labels differ")
    if len(original_pool) != 75:
        raise ValueError("expected 75 original-pool rows")

    fit_ids = sorted(idx for idx, row in profile.items() if row.split == "tune")
    fresh_ids = sorted(idx for idx, row in evaluation.items() if row.split == "eval")
    confirmation_ids = sorted(confirmation)
    if len(fit_ids) != 41 or len(fresh_ids) != 57 or len(confirmation_ids) != 45:
        raise ValueError("expected fit/fresh/confirmation counts 41/57/45")
    if set(fit_ids) & set(fresh_ids):
        raise ValueError("fit and fresh trajectory IDs overlap")
    if confirmation_ids != sorted(original_pool)[:45]:
        raise ValueError("confirmation rows are not the first 45 original eligible IDs")
    for idx in fit_ids + fresh_ids + confirmation_ids:
        if not 0 <= idx < len(identities):
            raise ValueError(f"dataset index out of parquet range: {idx}")

    selected_map_sha256 = sha256_file(paths["selected_map"])
    if selected_map_sha256 != EXPECTED_SELECTED_MAP_SHA256:
        raise ValueError("selected-map config differs from frozen hash")
    for row in list(evaluation.values()) + list(confirmation.values()):
        assert_selected_map_reference(row, selected_map_sha256)

    observed_models = {
        row.model
        for row_set in rows.values()
        for row in row_set.values()
        if row.model is not None
    }
    if observed_models != {MODEL_ID}:
        raise ValueError(f"wrong or heterogeneous model IDs: {observed_models}")

    def make_analysis_rows(
        component: str, selected_ids: Iterable[int], source: dict[int, ScoreRow]
    ) -> list[AnalysisRow]:
        output: list[AnalysisRow] = []
        for idx in sorted(selected_ids):
            identity = identities[idx]
            output.append(
                AnalysisRow(
                    component=component,
                    dataset_idx=idx,
                    value=arm_tf(source[idx], "E-champion")
                    - arm_tf(source[idx], "B"),
                    task_sha256=identity.task_sha256,
                    snapshot_key=identity.snapshot_key,
                    repository_key=identity.repository_key,
                )
            )
        return output

    fresh_rows = make_analysis_rows("fresh57", fresh_ids, evaluation)
    confirmation_rows = make_analysis_rows(
        "original45_partial_confirmation", confirmation_ids, confirmation
    )
    pooled_rows = fresh_rows + confirmation_rows
    fit_tasks = {identities[idx].task_sha256 for idx in fit_ids}
    strict_rows = [row for row in pooled_rows if row.task_sha256 not in fit_tasks]
    strict_fresh = [row for row in fresh_rows if row.task_sha256 not in fit_tasks]
    strict_confirmation = [
        row for row in confirmation_rows if row.task_sha256 not in fit_tasks
    ]

    fresh_tasks = {row.task_sha256 for row in fresh_rows}
    confirmation_tasks = {row.task_sha256 for row in confirmation_rows}

    result_repo_validation = {
        "definition": (
            "extract /workspace/owner__repo__version from saved gold_action only, "
            "then compare with the parquet-derived snapshot key"
        ),
        "rows_checked": len(pooled_rows),
        "rows_with_any_gold_action_snapshot_key": 0,
        "rows_with_exact_expected_gold_action_snapshot_key": 0,
        "rows_with_unique_exact_expected_gold_action_snapshot_key": 0,
        "mismatched_dataset_indices": [],
        "missing_dataset_indices": [],
    }
    sources = {"fresh57": evaluation, "original45_partial_confirmation": confirmation}
    for row in pooled_rows:
        score_row = sources[row.component][row.dataset_idx]
        gold = json.dumps(score_row.gold_action, sort_keys=True)
        candidates = set(WORKSPACE_SNAPSHOT_RE.findall(gold))
        expected = identities[row.dataset_idx].snapshot_key
        result_repo_validation["rows_with_any_gold_action_snapshot_key"] += bool(
            candidates
        )
        result_repo_validation[
            "rows_with_exact_expected_gold_action_snapshot_key"
        ] += expected in candidates
        result_repo_validation[
            "rows_with_unique_exact_expected_gold_action_snapshot_key"
        ] += candidates == {expected}
        if not candidates:
            result_repo_validation["missing_dataset_indices"].append(row.dataset_idx)
        elif expected not in candidates:
            result_repo_validation["mismatched_dataset_indices"].append(
                row.dataset_idx
            )

    task_sizes = Counter(identity.task_sha256 for identity in identities)
    snapshot_sizes = Counter(identity.snapshot_key for identity in identities)
    repository_sizes = Counter(identity.repository_key for identity in identities)
    transcript_hashes = {
        identity.trajectory_messages_sha256 for identity in identities
    }
    repeated_tasks = {task for task, count in task_sizes.items() if count > 1}

    program_path = Path(__file__).resolve()
    parquet_relative = (
        str(paths["parquet"].relative_to(repo_root))
        if paths["parquet"].is_relative_to(repo_root)
        else str(paths["parquet"])
    )
    directory_inputs = [
        describe_directory(repo_root, role, paths[role])
        for role in (
            "map_fitting",
            "selected_map_evaluation",
            "partial_confirmation",
            "original_pool",
        )
    ]

    return {
        "schema": "swegym-cluster-sensitivity/v1",
        "generated_at_utc": generated_at_utc,
        "cost": {
            "new_gpu_spend_usd": 0.0,
            "external_model_calls": 0,
            "description": "CPU-only parsing and arithmetic over local parquet plus committed JSON rows",
        },
        "analysis_program": {
            "path": str(program_path.relative_to(repo_root)),
            "sha256": sha256_file(program_path),
            "repository_head_at_analysis": git_head(repo_root),
        },
        "subject": {"model_id": MODEL_ID},
        "statistical_contract": {
            "metric": (
                "per trajectory E-champion.tf_mean - B.tf_mean, nats per target token; "
                "tf_mean is saved teacher-forced mean log-probability of the demonstrated next action"
            ),
            "row_order": (
                "fresh evaluation rows sorted by numeric dataset idx, followed by confirmation rows "
                "sorted by numeric dataset idx"
            ),
            "observable_task_key": (
                "sha256_utf8(snapshot_key + U+0000 + stripped content of the first "
                "<pr_description>...</pr_description> block in the initial user message)"
            ),
            "snapshot_key": "literal owner__repository__version basename in the initial <uploaded_files> path",
            "repository_key": "owner/repository parsed from the asserted three-part snapshot key",
            "iid_comparison": (
                "sample N trajectory rows with replacement; arithmetic mean per replicate"
            ),
            "row_weighted_cluster_bootstrap": (
                "sort cluster keys lexicographically; draw G whole clusters with replacement from G; "
                "retain every member trajectory with multiplicity; divide summed row deltas by the "
                "total number of contributed rows"
            ),
            "cluster_equal_sensitivity": (
                "same whole-cluster draws, but average one within-cluster mean per draw; this changes "
                "the estimand and is not substituted for the trajectory-weighted headline"
            ),
            "strict_no_fit_task_rule": (
                "exclude every selected-map evaluation/confirmation row whose observable task_sha256 "
                "appears among the 41 map-fitting rows"
            ),
            "bootstrap": {
                "replicates": n_reps,
                "seed": seed,
                "rng": "Python random.Random",
                "interval": "ordinary percentile",
                "lower_index": int(0.025 * n_reps),
                "upper_index": int(0.975 * n_reps),
                "multiplicity_adjustment": None,
            },
        },
        "inputs": {
            "parquet": {
                "path": parquet_relative,
                "tracked_in_git": False,
                "size_bytes": paths["parquet"].stat().st_size,
                "sha256": parquet_sha256,
                "expected_sha256_verified": True,
                "rows": len(identities),
                "columns": ["messages"],
            },
            "selected_map": {
                "path": str(paths["selected_map"].relative_to(repo_root)),
                "sha256": selected_map_sha256,
                "expected_sha256_verified": True,
            },
            "score_directories": directory_inputs,
        },
        "identity_recovery_validation": {
            "parser_coverage": {
                "rows_total": len(identities),
                "rows_parsed": len(identities),
                "fraction": 1.0,
            },
            "all_rows_begin_system_then_user": True,
            "all_rows_have_exactly_one_system_message": True,
            "shared_system_prompt": {
                "distinct_hashes": len(system_hashes),
                "sha256": EXPECTED_SYSTEM_SHA256,
                "rows": system_hashes[EXPECTED_SYSTEM_SHA256],
            },
            "exact_workspace_path_reappears_in_later_assistant_messages": {
                "rows": sum(
                    identity.later_assistant_repeats_exact_workspace_path
                    for identity in identities
                ),
                "of": len(identities),
            },
            "task_partition_matches_exact_initial_user_partition": _partition_equivalent(
                identities, "task_sha256", "initial_user_sha256"
            ),
            "pr_description_hash_maps_to_one_snapshot_for_every_task": (
                _partition_equivalent(
                    identities, "pr_description_sha256", "task_sha256"
                )
            ),
            "all_full_trajectory_transcripts_distinct": (
                len(transcript_hashes) == len(identities)
            ),
            "distinct_full_trajectory_transcripts": len(transcript_hashes),
            "repeated_task_clusters_contain_distinct_rollouts": all(
                len(
                    {
                        identity.trajectory_messages_sha256
                        for identity in identities
                        if identity.task_sha256 == task
                    }
                )
                == task_sizes[task]
                for task in repeated_tasks
            ),
            "global_cluster_counts": {
                "tasks": len(task_sizes),
                "versioned_snapshots": len(snapshot_sizes),
                "repositories": len(repository_sizes),
                "task_cluster_size_histogram": {
                    str(k): v
                    for k, v in sorted(Counter(task_sizes.values()).items())
                },
                "repeated_task_clusters": len(repeated_tasks),
                "rows_in_repeated_task_clusters": sum(
                    task_sizes[task] for task in repeated_tasks
                ),
            },
            "committed_result_gold_action_repository_crosscheck": result_repo_validation,
        },
        "selection_and_overlap": {
            "map_fit": _set_overlap_report(fit_ids, identities, fit_tasks),
            "fresh57": _set_overlap_report(fresh_ids, identities, fit_tasks),
            "original45_partial_confirmation": _set_overlap_report(
                confirmation_ids, identities, fit_tasks
            ),
            "pooled102": _set_overlap_report(
                fresh_ids + confirmation_ids, identities, fit_tasks
            ),
            "fresh_confirmation_task_overlap": {
                "n_task_clusters": len(fresh_tasks & confirmation_tasks),
                "task_sha256": sorted(fresh_tasks & confirmation_tasks),
                "fresh_rows_whose_task_occurs_in_confirmation": sum(
                    row.task_sha256 in confirmation_tasks for row in fresh_rows
                ),
                "confirmation_rows_whose_task_occurs_in_fresh": sum(
                    row.task_sha256 in fresh_tasks for row in confirmation_rows
                ),
            },
            "confirmation_is_exact_first45_original_eligible_prefix": True,
        },
        "selected_map_minus_B": {
            "fresh57": analysis_bundle(fresh_rows, n_reps=n_reps, seed=seed),
            "original45_partial_confirmation": analysis_bundle(
                confirmation_rows, n_reps=n_reps, seed=seed
            ),
            "pooled102": analysis_bundle(pooled_rows, n_reps=n_reps, seed=seed),
            "strict_task_disjoint_from_fit": {
                "fresh": analysis_bundle(
                    strict_fresh, n_reps=n_reps, seed=seed
                ),
                "confirmation": analysis_bundle(
                    strict_confirmation, n_reps=n_reps, seed=seed
                ),
                "pooled": analysis_bundle(
                    strict_rows, n_reps=n_reps, seed=seed
                ),
            },
        },
        "cluster_summaries_for_pooled102": {
            "task": _cluster_summaries(pooled_rows, "task_sha256"),
            "snapshot": _cluster_summaries(pooled_rows, "snapshot_key"),
            "repository": _cluster_summaries(pooled_rows, "repository_key"),
        },
        "derived_cluster_mapping": [
            identity.serialized() for identity in identities
        ],
        "limitations": [
            (
                "task_sha256 is a content-addressed observable-task surrogate, not a recovered upstream "
                "SWE issue/PR ID. Exact matching can under-merge semantically identical tasks whose literal "
                "descriptions differ; it does not falsely merge different literal tasks."
            ),
            (
                "Repository and version come from the explicit uploaded workspace basename. The parser "
                "asserts the owner__repository__version convention observed in all 491 rows; it cannot prove "
                "the immutable upstream code revision behind a version label."
            ),
            (
                "The parquet is ignored rather than committed. This artifact binds the mapping to its exact "
                "SHA-256 and serializes only hashes/normalized keys so cluster inference remains auditable "
                "without publishing the large literal task corpus."
            ),
            (
                "The historical tune/eval split hashes trajectory index rather than task identity, so exact "
                "task clusters cross the fitting boundary. The strict no-fit-task analysis removes every "
                "observed overlap but remains a post-hoc sensitivity."
            ),
            (
                "The original confirmation is a nonrandom, budget-stopped 45/75 prefix. Pooling it with the "
                "fresh 57 is descriptive and not a preregistered meta-analysis."
            ),
            (
                "Only nine normalized repositories occur in pooled102, with unequal cluster sizes; the "
                "repository bootstrap is a coarse sensitivity, not a well-powered repository-population CI."
            ),
            (
                "Intervals condition on the observed pool and selected map, use no multiplicity adjustment, "
                "and do not repair the absent selected-map-matched placebo, action execution, task success, "
                "upstream row-level generator identity, or unsaved per-trajectory summary text."
            ),
        ],
        "paper_blocker_assessment": {
            "statistical_lead_overturned": False,
            "current_manuscript_requires_correction_before_release": True,
            "reason": (
                "Task-clustered and strict task-disjoint intervals remain above zero, but the current "
                "manuscript says stable cluster IDs were unavailable and does not disclose exact-task "
                "fit/evaluation overlap."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--parquet", type=Path, default=None)
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
            / "swegym_cluster_sensitivity"
            / f"swegym_cluster_sensitivity_{MODEL_SLUG}_{stamp}.json"
        )
    elif not output.is_absolute():
        output = repo_root / output

    parquet = args.parquet
    if parquet is not None and not parquet.is_absolute():
        parquet = repo_root / parquet
    print(
        f"RUN swegym-cluster-sensitivity model={MODEL_ID} -> {output}",
        flush=True,
    )
    report = build_report(
        repo_root,
        parquet_path=parquet,
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
