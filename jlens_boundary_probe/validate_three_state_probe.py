#!/usr/bin/env python3
"""Validate and summarize a three-state ValueGraft/J-lens boundary probe."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


REQUIRED_STATES = ("full_context", "fresh_compacted", "grafted_compacted")
REQUIRED_SEQUENCES = (
    "full_context",
    "fresh_compacted",
    "alpha0_grafted_compacted",
    "grafted_compacted",
)


def top_ids(entries: list[dict[str, Any]]) -> tuple[int, ...]:
    return tuple(int(item["token_id"]) for item in entries)


def top_scores(entries: list[dict[str, Any]]) -> tuple[float, ...]:
    return tuple(float(item["score"]) for item in entries)


def scores_match(left: tuple[float, ...], right: tuple[float, ...], *, tol: float = 1e-9) -> bool:
    return len(left) == len(right) and all(abs(a - b) <= tol for a, b in zip(left, right))


def jaccard_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    a = set(left)
    b = set(right)
    if not a and not b:
        return 0.0
    return 1.0 - (len(a & b) / len(a | b))


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def rows_for(data: dict[str, Any], state: str) -> list[dict[str, Any]]:
    seq = data["post_boundary_probe"]["forced_target_sequences"][state]
    return seq["rows"]


def validate(data: dict[str, Any], *, allow_missing_provenance: bool) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []

    graft = data.get("graft", {})
    probe = data.get("post_boundary_probe", {})
    require(graft.get("available") is True, "graft.available must be true", failures)
    require("states" in probe, "post_boundary_probe.states is missing", failures)
    require("forced_target_sequences" in probe, "post_boundary_probe.forced_target_sequences is missing", failures)

    state_keys = set(probe.get("states", {}).keys())
    seq_keys = set(probe.get("forced_target_sequences", {}).keys())
    require(set(REQUIRED_STATES).issubset(state_keys), f"missing post-boundary states: {set(REQUIRED_STATES) - state_keys}", failures)
    require(set(REQUIRED_SEQUENCES).issubset(seq_keys), f"missing forced target sequences: {set(REQUIRED_SEQUENCES) - seq_keys}", failures)

    provenance_fields = (
        "policy",
        "pairs",
        "alpha",
        "changed_value_layers",
        "changed_value_layer_count",
        "changed_value_slot_count",
        "cache_old_summary",
        "cache_fresh_probe",
        "cache_grafted_probe",
    )
    missing_provenance = [field for field in provenance_fields if field not in graft]
    if missing_provenance:
        message = f"missing graft provenance fields: {missing_provenance}"
        if allow_missing_provenance:
            warnings.append(message)
        else:
            failures.append(message)

    if "pairs" in graft:
        require(int(graft["pairs"]) > 0, "graft.pairs must be positive", failures)
    if "changed_value_layer_count" in graft:
        require(int(graft["changed_value_layer_count"]) > 0, "changed_value_layer_count must be positive", failures)

    if failures:
        return failures, warnings

    sequences = {state: rows_for(data, state) for state in REQUIRED_SEQUENCES}
    lengths = {state: len(rows) for state, rows in sequences.items()}
    require(len(set(lengths.values())) == 1, f"forced sequence lengths differ: {lengths}", failures)
    reference_len = lengths["full_context"]
    require(reference_len > 0, "forced sequence must be nonempty", failures)
    target = data.get("probe_target", {})
    if "token_count" in target:
        require(
            reference_len == int(target["token_count"]),
            f"forced sequence length {reference_len} differs from probe_target.token_count {target['token_count']}",
            failures,
        )

    sequence_meta = data["post_boundary_probe"]["forced_target_sequences"]
    forced_texts = {state: sequence_meta[state].get("forced_text") for state in REQUIRED_SEQUENCES}
    require(len(set(forced_texts.values())) == 1, f"forced_text differs across sequences: {forced_texts}", failures)
    if target.get("text") is not None:
        require(
            forced_texts["full_context"] == target["text"],
            "forced_text differs from probe_target.text",
            failures,
        )

    reference_tokens = [row["token_id"] for row in sequences["full_context"]]
    for state, rows in sequences.items():
        token_ids = [row["token_id"] for row in rows]
        require(token_ids == reference_tokens, f"{state} forced token IDs do not match full_context", failures)

    fresh_rows = sequences["fresh_compacted"]
    alpha0_rows = sequences["alpha0_grafted_compacted"]
    for i, (fresh, alpha0) in enumerate(zip(fresh_rows, alpha0_rows)):
        require(fresh["argmax_token_id"] == alpha0["argmax_token_id"], f"alpha0 argmax differs from fresh at row {i}", failures)
        require(
            top_ids(fresh["next_token_top"]) == top_ids(alpha0["next_token_top"]),
            f"alpha0 next-token top-k differs from fresh at row {i}",
            failures,
        )
        require(
            scores_match(top_scores(fresh["next_token_top"]), top_scores(alpha0["next_token_top"])),
            f"alpha0 next-token top-k scores differ from fresh at row {i}",
            failures,
        )
        fresh_layers = fresh.get("layers", {})
        alpha0_layers = alpha0.get("layers", {})
        require(set(fresh_layers) == set(alpha0_layers), f"alpha0 layer set differs from fresh at row {i}", failures)
        require("48" in fresh_layers, f"layer 48 is missing at row {i}", failures)
        for layer in fresh_layers:
            require(
                top_ids(fresh_layers[layer]) == top_ids(alpha0_layers[layer]),
                f"alpha0 layer {layer} top-k differs from fresh at row {i}",
                failures,
            )
            require(
                scores_match(top_scores(fresh_layers[layer]), top_scores(alpha0_layers[layer])),
                f"alpha0 layer {layer} top-k scores differ from fresh at row {i}",
                failures,
            )

    return failures, warnings


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    sequences = {state: rows_for(data, state) for state in REQUIRED_SEQUENCES}
    full = sequences["full_context"]
    fresh = sequences["fresh_compacted"]
    grafted = sequences["grafted_compacted"]
    alpha0 = sequences["alpha0_grafted_compacted"]

    layers = sorted(full[0].get("layers", {}).keys(), key=lambda x: int(x))
    layer_metrics: dict[str, dict[str, float]] = {}
    for layer in layers:
        ff: list[float] = []
        fg: list[float] = []
        gf: list[float] = []
        fa0: list[float] = []
        for a, b, c, z in zip(full, fresh, grafted, alpha0):
            full_ids = top_ids(a["layers"][layer])
            fresh_ids = top_ids(b["layers"][layer])
            graft_ids = top_ids(c["layers"][layer])
            alpha0_ids = top_ids(z["layers"][layer])
            ff.append(jaccard_distance(full_ids, fresh_ids))
            fg.append(jaccard_distance(full_ids, graft_ids))
            gf.append(jaccard_distance(fresh_ids, graft_ids))
            fa0.append(jaccard_distance(fresh_ids, alpha0_ids))
        layer_metrics[layer] = {
            "mean_full_fresh": mean(ff),
            "mean_full_grafted": mean(fg),
            "mean_fresh_grafted": mean(gf),
            "mean_fresh_alpha0": mean(fa0),
            "mean_closure_full_fresh_minus_full_grafted": mean([a - b for a, b in zip(ff, fg)]),
        }

    argmax_rescues: list[dict[str, Any]] = []
    argmax_regressions: list[dict[str, Any]] = []
    graft_changes: list[dict[str, Any]] = []
    for i, (a, b, c) in enumerate(zip(full, fresh, grafted)):
        entry = {
            "index": i,
            "forced_token": a["token"],
            "full_argmax": a["argmax_token"],
            "fresh_argmax": b["argmax_token"],
            "grafted_argmax": c["argmax_token"],
        }
        if b["argmax_token_id"] != c["argmax_token_id"]:
            graft_changes.append(entry)
        if b["argmax_token_id"] != a["argmax_token_id"] and c["argmax_token_id"] == a["argmax_token_id"]:
            argmax_rescues.append(entry)
        if b["argmax_token_id"] == a["argmax_token_id"] and c["argmax_token_id"] != a["argmax_token_id"]:
            argmax_regressions.append(entry)

    return {
        "model": data.get("model"),
        "lens": data.get("lens"),
        "probe_target": data.get("probe_target", {}),
        "graft": data.get("graft", {}),
        "forced_token_count": len(full),
        "layers": layers,
        "layer_metrics": layer_metrics,
        "argmax_rescue_count": len(argmax_rescues),
        "argmax_regression_count": len(argmax_regressions),
        "graft_changed_argmax_count": len(graft_changes),
        "argmax_rescues": argmax_rescues,
        "argmax_regressions": argmax_regressions,
        "graft_changes": graft_changes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--allow-missing-provenance", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit summary JSON")
    args = parser.parse_args()

    with args.artifact.open() as f:
        data = json.load(f)

    failures, warnings = validate(data, allow_missing_provenance=args.allow_missing_provenance)
    if failures:
        print("INVALID three-state probe artifact:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
        return 1

    summary = summarize(data)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print("VALID three-state probe artifact")
    if warnings:
        for warning in warnings:
            print(f"warning: {warning}")
    print(f"model: {summary['model']}")
    print(f"forced target tokens: {summary['forced_token_count']}")
    print(f"graft policy: {summary['graft'].get('policy', '<missing>')}")
    print(f"graft pairs: {summary['graft'].get('pairs')} alpha: {summary['graft'].get('alpha')}")
    print(f"changed value layers: {summary['graft'].get('changed_value_layer_count')}")
    print(f"argmax rescues: {summary['argmax_rescue_count']}")
    print(f"argmax regressions: {summary['argmax_regression_count']}")
    print(f"argmax changed by graft: {summary['graft_changed_argmax_count']}")
    print("layer metrics:")
    for layer, metrics in summary["layer_metrics"].items():
        print(
            f"  layer {layer}: "
            f"full-fresh={metrics['mean_full_fresh']:.4f} "
            f"full-grafted={metrics['mean_full_grafted']:.4f} "
            f"fresh-grafted={metrics['mean_fresh_grafted']:.4f} "
            f"fresh-alpha0={metrics['mean_fresh_alpha0']:.4f} "
            f"closure={metrics['mean_closure_full_fresh_minus_full_grafted']:.4f}"
        )
    if summary["argmax_rescues"]:
        print("argmax rescues:")
        for item in summary["argmax_rescues"]:
            print(
                f"  {item['index']:02d} {item['forced_token']!r}: "
                f"fresh {item['fresh_argmax']!r} -> grafted/full {item['grafted_argmax']!r}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
