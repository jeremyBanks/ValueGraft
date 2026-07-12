#!/usr/bin/env python3
"""Audit whether the evaluated legacy head maps derive from tracked profiles."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any


SLOT_RE = re.compile(r"^L(\d+)H(\d+)$")
CONFIG_NAMES = ("heads", "layers", "intersection", "union")
HEADLINE_STAMP = "20260709T194156Z"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slots_from_head_map(head_map: dict[str, list[int]]) -> set[tuple[int, int]]:
    return {(int(layer), int(head)) for layer, heads in head_map.items() for head in heads}


def mean_positive_slots(rows: list[dict[tuple[int, int], float]]) -> set[tuple[int, int]]:
    if not rows:
        return set()
    slots = set(rows[0])
    assert all(set(row) == slots for row in rows)
    return {
        slot
        for slot in slots
        if sum(row[slot] for row in rows) / len(rows) > 0.0
    }


def serialize_slots(slots: set[tuple[int, int]]) -> list[str]:
    return [f"L{layer}H{head}" for layer, head in sorted(slots)]


def load_profile(profile_dir: Path) -> tuple[list[Path], list[dict[tuple[int, int], float]]]:
    files = sorted(profile_dir.glob("*.json"))
    assert files, f"no JSON profile rows in {profile_dir}"
    rows: list[dict[tuple[int, int], float]] = []
    for path in files:
        data = json.loads(path.read_text())
        baseline = float(data["B"])
        row: dict[tuple[int, int], float] = {}
        for key, value in data.items():
            match = SLOT_RE.fullmatch(key)
            if match:
                row[(int(match.group(1)), int(match.group(2)))] = float(value) - baseline
        assert len(row) == 192, (path, len(row))
        rows.append(row)
    return files, rows


def load_embedded_configs(repo: Path) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths: dict[str, Path] = {}
    configs: dict[str, dict[str, Any]] = {}
    for name in CONFIG_NAMES:
        path = (
            repo
            / "results"
            / "champion_validate"
            / f"headscan_30b_bf16_{name}_shuffle_pos_{HEADLINE_STAMP}.json"
        )
        paths[name] = path
        configs[name] = json.loads(path.read_text())["champion_cfg"]
    return paths, configs


def build_report(repo: Path, generated_at_utc: str) -> dict[str, Any]:
    profile_dir = repo / "results" / "tune_head_30b_bf16"
    profile_files, rows = load_profile(profile_dir)
    result_paths, configs = load_embedded_configs(repo)

    tracked_positive = mean_positive_slots(rows)
    evaluated_heads = slots_from_head_map(configs["heads"]["head_map"])
    overlap = tracked_positive & evaluated_heads

    matching_masks: list[int] = []
    for size in range(1, len(rows) + 1):
        for indices in itertools.combinations(range(len(rows)), size):
            if mean_positive_slots([rows[index] for index in indices]) == evaluated_heads:
                matching_masks.append(sum(1 << index for index in indices))

    layer_config_path = repo / "data" / "champion_configs" / "layers_30b_bf16.json"
    tracked_layer_config = json.loads(layer_config_path.read_text())
    layer_set = {int(layer) for layer, alpha in tracked_layer_config["alpha_map"].items() if alpha}
    evaluated_intersection = slots_from_head_map(configs["intersection"]["head_map"])
    evaluated_union = slots_from_head_map(configs["union"]["head_map"])
    reconstructed_intersection = {slot for slot in evaluated_heads if slot[0] in layer_set}
    reconstructed_union = evaluated_heads | {(layer, head) for layer in layer_set for head in range(4)}

    return {
        "schema": "legacy-head-map-lineage-audit/v1",
        "generated_at_utc": generated_at_utc,
        "scope": {
            "claim": "derivation lineage of evaluated legacy per-head/intersection/union maps",
            "selection_rule": "mean per-row marginal over selected profile rows > 0",
            "subset_search": "all nonempty subsets of the tracked ten-row profile",
        },
        "tracked_profile": {
            "directory": str(profile_dir.relative_to(repo)),
            "row_count": len(profile_files),
            "row_ids": [path.stem for path in profile_files],
            "files": [
                {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
                for path in profile_files
            ],
            "positive_slot_count": len(tracked_positive),
            "positive_slots": serialize_slots(tracked_positive),
            "first_tracked_commit": "2db7c6d88037ae3c983b7862be95a1550c61b08d",
        },
        "evaluated_configs": {
            name: {
                "result_path": str(result_paths[name].relative_to(repo)),
                "result_sha256": sha256_file(result_paths[name]),
                "label": configs[name]["label"],
                "head_slot_count": len(slots_from_head_map(configs[name].get("head_map", {}))),
                "alpha_layer_count": len(configs[name].get("alpha_map", {})),
            }
            for name in CONFIG_NAMES
        },
        "head_map_comparison": {
            "evaluated_slot_count": len(evaluated_heads),
            "tracked_positive_slot_count": len(tracked_positive),
            "overlap_count": len(overlap),
            "evaluated_but_not_tracked_positive_count": len(evaluated_heads - tracked_positive),
            "tracked_positive_but_not_evaluated_count": len(tracked_positive - evaluated_heads),
            "evaluated_but_not_tracked_positive": serialize_slots(evaluated_heads - tracked_positive),
            "tracked_positive_but_not_evaluated": serialize_slots(tracked_positive - evaluated_heads),
            "nonempty_profile_subset_count_tested": (1 << len(rows)) - 1,
            "matching_subset_count": len(matching_masks),
            "matching_subset_masks": matching_masks,
            "derivation_reproduced": tracked_positive == evaluated_heads,
        },
        "combination_checks": {
            "tracked_layer_config_path": str(layer_config_path.relative_to(repo)),
            "tracked_layer_config_sha256": sha256_file(layer_config_path),
            "tracked_layer_map_equals_evaluated_layer_map": (
                tracked_layer_config["alpha_map"] == configs["layers"]["alpha_map"]
            ),
            "selected_layer_count": len(layer_set),
            "intersection_reconstructs_from_evaluated_heads_and_layers": (
                reconstructed_intersection == evaluated_intersection
            ),
            "union_reconstructs_from_evaluated_heads_and_layers": (
                reconstructed_union == evaluated_union
            ),
            "reconstructed_intersection_slot_count": len(reconstructed_intersection),
            "reconstructed_union_slot_count": len(reconstructed_union),
        },
        "inference": {
            "status": "BROKEN_DERIVATION_LINEAGE",
            "observed": (
                "The evaluated 109-slot head map is not the positive-mean map from the "
                "tracked ten-row profile and is not reproduced by any nonempty subset."
            ),
            "plausible_loss_mechanism_not_proven": (
                "The head-scan job re-derived a pod-local profile/config, while the launcher "
                "did not upload results/ and the validation commit retained embedded configs "
                "but not that pod-local derivation."
            ),
            "schedule_lineage": (
                "The tracked layer/head profiles predate commit 317a0dd1, which introduced "
                "4096-token chunked prefill. The later head-scan job likely re-derived the "
                "missing head profile under chunked prefill before chunked headline validation."
            ),
            "result_disposition": (
                "Evaluated outputs remain descriptive for their embedded fixed interventions; "
                "head/intersection/union selection provenance is not reproducible."
            ),
        },
        "git_lineage_facts": {
            "tracked_layer_profile_commit": "84bcee776da63983c14d3d6b6af00c37c332c349",
            "tracked_head_profile_commit": "2db7c6d88037ae3c983b7862be95a1550c61b08d",
            "chunked_prefill_commit": "317a0dd1a13a25b614e867a39270ee176e73926f",
            "tracked_profiles_predate_chunked_prefill": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()
    report = build_report(args.repo.resolve(), args.timestamp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"WROTE {args.output} status={report['inference']['status']} "
        f"matching_subsets={report['head_map_comparison']['matching_subset_count']}"
    )


if __name__ == "__main__":
    main()
