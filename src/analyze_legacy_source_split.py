#!/usr/bin/env python3
"""Recompute the legacy synthetic result by conversation-body source block.

The headline legacy estimate pools c07--c24.  Those conversations are not one
homogeneous corpus: c07--c12 were rendered by a local Qwen3-4B model, whereas
c13--c24 were authored by Claude-family assistants.  This script preserves the
original plant-weighted estimand and conversation-clustered percentile
bootstrap, but reports the two blocks separately as an exploratory diagnostic.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_INPUTS = {
    "per-head": Path(
        "results/champion_validate/"
        "headscan_30b_bf16_heads_shuffle_pos_20260709T194156Z.json"
    ),
    "per-layer": Path(
        "results/champion_validate/"
        "headscan_30b_bf16_layers_shuffle_pos_20260709T194156Z.json"
    ),
    "intersection": Path(
        "results/champion_validate/"
        "headscan_30b_bf16_intersection_shuffle_pos_20260709T194156Z.json"
    ),
    "union": Path(
        "results/champion_validate/"
        "headscan_30b_bf16_union_shuffle_pos_20260709T194156Z.json"
    ),
}

BLOCKS = {
    "qwen3_4b_rendered_c07_c12": set(range(7, 13)),
    "assistant_authored_aliases_c13_c24": set(range(13, 25)),
    "pooled_c07_c24": set(range(7, 25)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def conv_number(conv_id: str) -> int:
    if len(conv_id) != 3 or not conv_id.startswith("c"):
        raise ValueError(f"expected cNN conversation id, got {conv_id!r}")
    return int(conv_id[1:])


def cluster_bootstrap(
    rows: list[dict[str, Any]], *, n_boot: int, seed: int
) -> dict[str, Any]:
    by_conv: dict[str, list[float]] = {}
    for row in rows:
        value = row.get("raw_EB")
        if value is None:
            value = float(row["lp_E"]) - float(row["lp_B"])
        by_conv.setdefault(str(row["conversation_id"]), []).append(float(value))

    convs = sorted(by_conv)
    clusters = [by_conv[conv] for conv in convs]
    flat = [value for cluster in clusters for value in cluster]
    if not flat:
        raise ValueError("empty analysis block")

    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        sampled_flat = [value for cluster in sampled for value in cluster]
        boot_means.append(sum(sampled_flat) / len(sampled_flat))
    boot_means.sort()

    return {
        "mean_raw_EB_nats_per_token": sum(flat) / len(flat),
        "ci_95_percentile": [
            boot_means[int(0.025 * n_boot)],
            boot_means[int(0.975 * n_boot)],
        ],
        "n_plants": len(flat),
        "n_conversations": len(convs),
        "conversation_ids": convs,
    }


def analyze(
    inputs: dict[str, Path], *, n_boot: int = 10_000, seed: int = 42
) -> dict[str, Any]:
    configurations: dict[str, Any] = {}
    input_records: list[dict[str, Any]] = []
    expected_ids = {f"c{number:02d}" for number in range(7, 25)}

    for label, path in inputs.items():
        document = json.loads(path.read_text())
        traces = document.get("traces")
        if not isinstance(traces, list) or not traces:
            raise ValueError(f"{path}: missing non-empty traces array")
        observed_ids = {str(row["conversation_id"]) for row in traces}
        if observed_ids != expected_ids:
            raise ValueError(
                f"{path}: conversation set mismatch; "
                f"missing={sorted(expected_ids - observed_ids)}, "
                f"extra={sorted(observed_ids - expected_ids)}"
            )

        block_results: dict[str, Any] = {}
        for block_label, block_numbers in BLOCKS.items():
            block_rows = [
                row
                for row in traces
                if conv_number(str(row["conversation_id"])) in block_numbers
            ]
            block_results[block_label] = cluster_bootstrap(
                block_rows, n_boot=n_boot, seed=seed
            )
        configurations[label] = block_results
        input_records.append(
            {
                "configuration": label,
                "path": str(path),
                "sha256": sha256(path),
            }
        )

    return {
        "schema_version": 1,
        "analysis": "legacy synthetic conversation-body source stratification",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_git_head": git_head(),
        "estimand": (
            "plant-weighted mean raw_EB = lp_E - lp_B in nats/token, with "
            "conversation-clustered percentile bootstrap"
        ),
        "bootstrap": {"seed": seed, "replicates": n_boot},
        "status": "exploratory_source_stratification",
        "causal_caveat": (
            "The blocks also differ in length, authoring process, plant "
            "construction, and similarity to the tuning corpus. Source family "
            "is not causally isolated. Intervals are nominal and conditional "
            "on these realized rendered conversations."
        ),
        "inputs": input_records,
        "configurations": configurations,
    }


def parse_inputs(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_INPUTS)
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--input must be LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        if not label or label in parsed:
            raise ValueError(f"invalid or duplicate input label {label!r}")
        parsed[label] = Path(raw_path)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        help="LABEL=PATH; repeat for each configuration (defaults to four paper rows)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="bootstrap seed (42 reproduces the source files' headline intervals)",
    )
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.n_boot <= 0:
        parser.error("--n-boot must be positive")
    result = analyze(
        parse_inputs(args.input), n_boot=args.n_boot, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
