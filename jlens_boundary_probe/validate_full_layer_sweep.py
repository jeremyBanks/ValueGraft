#!/usr/bin/env python3
"""Validate full-layer J-lens sweep artifacts.

This script encodes the output contract for ``full_layer_sweep.py`` and for the
derived compact summary JSON. It should be run before interpreting the sweep
results. Its purpose is to make schema assumptions explicit instead of relying
on ad hoc jq exploration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ValidationError(Exception):
    pass


def fail(path: str, message: str) -> None:
    raise ValidationError(f"{path}: {message}")


def require(condition: bool, path: str, message: str) -> None:
    if not condition:
        fail(path, message)


def require_type(value: Any, typ: type | tuple[type, ...], path: str) -> None:
    if not isinstance(value, typ):
        expected = (
            " or ".join(t.__name__ for t in typ)
            if isinstance(typ, tuple)
            else typ.__name__
        )
        fail(path, f"expected {expected}, got {type(value).__name__}")


def require_keys(obj: dict[str, Any], keys: set[str], path: str) -> None:
    actual = set(obj)
    missing = sorted(keys - actual)
    extra = sorted(actual - keys)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if extra:
            parts.append(f"extra {extra}")
        fail(path, "; ".join(parts))


def require_number(value: Any, path: str) -> None:
    require_type(value, (int, float), path)
    require(not isinstance(value, bool), path, "expected number, got bool")


def require_probability(value: Any, path: str) -> None:
    require_number(value, path)
    require(0.0 <= float(value) <= 1.0, path, "expected value in [0, 1]")


def validate_target(target: Any, path: str) -> None:
    require_type(target, dict, path)
    require_keys(target, {"label", "phrase", "token_index_in_phrase"}, path)
    require_type(target["label"], str, f"{path}.label")
    require_type(target["phrase"], str, f"{path}.phrase")
    require_type(target["token_index_in_phrase"], int, f"{path}.token_index_in_phrase")
    require(target["token_index_in_phrase"] >= 0, path, "negative token index")


def validate_targets(targets: Any, path: str) -> None:
    require_type(targets, list, path)
    for i, target in enumerate(targets):
        validate_target(target, f"{path}[{i}]")


def validate_topk_token(row: Any, path: str) -> None:
    require_type(row, dict, path)
    require_keys(row, {"token_id", "token", "score"}, path)
    require_type(row["token_id"], int, f"{path}.token_id")
    require_type(row["token"], str, f"{path}.token")
    require_number(row["score"], f"{path}.score")


def validate_topk(rows: Any, path: str, top_k: int) -> None:
    require_type(rows, list, path)
    require(0 < len(rows) <= top_k, path, f"expected 1..{top_k} rows")
    for i, row in enumerate(rows):
        validate_topk_token(row, f"{path}[{i}]")


def validate_compact_top(rows: Any, path: str, max_len: int = 8) -> None:
    require_type(rows, list, path)
    require(len(rows) <= max_len, path, f"expected at most {max_len} entries")
    for i, row in enumerate(rows):
        require_type(row, str, f"{path}[{i}]")


def validate_layer_summary(row: Any, path: str, layers: set[int]) -> None:
    require_type(row, dict, path)
    require_keys(
        row,
        {
            "layer",
            "mean_jaccard_distance",
            "mean_rank_weighted_distance",
            "top1_change_rate",
            "mean_old_lens_vs_next_jaccard",
            "mean_fresh_lens_vs_next_jaccard",
        },
        path,
    )
    require_type(row["layer"], int, f"{path}.layer")
    require(row["layer"] in layers, f"{path}.layer", "layer not in sampled_layers")
    for key in (
        "mean_jaccard_distance",
        "mean_rank_weighted_distance",
        "top1_change_rate",
        "mean_old_lens_vs_next_jaccard",
        "mean_fresh_lens_vs_next_jaccard",
    ):
        require_probability(row[key], f"{path}.{key}")


def validate_metric_row(row: Any, path: str, summary_tokens: int, layers: set[int]) -> None:
    require_type(row, dict, path)
    require_keys(
        row,
        {"p", "l", "jd", "j", "wd", "wo", "i", "top1_changed", "old_next_j", "fresh_next_j"},
        path,
    )
    require_type(row["p"], int, f"{path}.p")
    require(0 <= row["p"] < summary_tokens, f"{path}.p", "position out of range")
    require_type(row["l"], int, f"{path}.l")
    require(row["l"] in layers, f"{path}.l", "layer not in sampled_layers")
    require_type(row["i"], int, f"{path}.i")
    require(row["i"] >= 0, f"{path}.i", "negative intersection count")
    require_type(row["top1_changed"], bool, f"{path}.top1_changed")
    for key in ("jd", "j", "wd", "wo", "old_next_j", "fresh_next_j"):
        require_probability(row[key], f"{path}.{key}")


def validate_token_summary(row: Any, path: str, summary_tokens: int, layers: set[int]) -> None:
    require_type(row, dict, path)
    require_keys(
        row,
        {
            "p",
            "token_id",
            "token",
            "kind",
            "context",
            "targets",
            "mean_wd",
            "max_wd",
            "best_layer",
            "top1_change_rate",
            "old_best_top",
            "fresh_best_top",
            "old_next_top",
            "fresh_next_top",
        },
        path,
    )
    require_type(row["p"], int, f"{path}.p")
    require(0 <= row["p"] < summary_tokens, f"{path}.p", "position out of range")
    require_type(row["token_id"], int, f"{path}.token_id")
    require_type(row["token"], str, f"{path}.token")
    require_type(row["kind"], str, f"{path}.kind")
    require_type(row["context"], str, f"{path}.context")
    validate_targets(row["targets"], f"{path}.targets")
    require_probability(row["mean_wd"], f"{path}.mean_wd")
    require_probability(row["max_wd"], f"{path}.max_wd")
    require_type(row["best_layer"], int, f"{path}.best_layer")
    require(row["best_layer"] in layers, f"{path}.best_layer", "not in sampled_layers")
    require_probability(row["top1_change_rate"], f"{path}.top1_change_rate")
    for key in ("old_best_top", "fresh_best_top", "old_next_top", "fresh_next_top"):
        validate_compact_top(row[key], f"{path}.{key}")


def validate_expanded_row(
    row: Any,
    path: str,
    summary_tokens: int,
    layers: set[int],
    top_k: int,
) -> None:
    require_type(row, dict, path)
    require_keys(
        row,
        {
            "p",
            "l",
            "jd",
            "j",
            "wd",
            "wo",
            "i",
            "top1_changed",
            "old_next_j",
            "fresh_next_j",
            "token_id",
            "token",
            "kind",
            "context",
            "targets",
            "write_time_top",
            "fresh_top",
            "write_time_next_top",
            "fresh_next_top",
        },
        path,
    )
    validate_metric_row(
        {k: row[k] for k in ("p", "l", "jd", "j", "wd", "wo", "i", "top1_changed", "old_next_j", "fresh_next_j")},
        path,
        summary_tokens,
        layers,
    )
    require_type(row["token_id"], int, f"{path}.token_id")
    require_type(row["token"], str, f"{path}.token")
    require_type(row["kind"], str, f"{path}.kind")
    require_type(row["context"], str, f"{path}.context")
    validate_targets(row["targets"], f"{path}.targets")
    for key in ("write_time_top", "fresh_top", "write_time_next_top", "fresh_next_top"):
        validate_topk(row[key], f"{path}.{key}", top_k)


def validate_compact_expanded_row(row: Any, path: str, summary_tokens: int, layers: set[int]) -> None:
    require_type(row, dict, path)
    require_keys(
        row,
        {
            "demo",
            "p",
            "layer",
            "token",
            "kind",
            "context",
            "targets",
            "rank_weighted_distance",
            "jaccard_distance",
            "top1_changed",
            "old_lens_vs_next_jaccard",
            "fresh_lens_vs_next_jaccard",
            "write_time_top",
            "fresh_top",
            "write_time_next_top",
            "fresh_next_top",
        },
        path,
    )
    require_type(row["demo"], str, f"{path}.demo")
    require_type(row["p"], int, f"{path}.p")
    require(0 <= row["p"] < summary_tokens, f"{path}.p", "position out of range")
    require_type(row["layer"], int, f"{path}.layer")
    require(row["layer"] in layers, f"{path}.layer", "not in sampled_layers")
    require_type(row["token"], str, f"{path}.token")
    require_type(row["kind"], str, f"{path}.kind")
    require_type(row["context"], str, f"{path}.context")
    validate_targets(row["targets"], f"{path}.targets")
    for key in (
        "rank_weighted_distance",
        "jaccard_distance",
        "old_lens_vs_next_jaccard",
        "fresh_lens_vs_next_jaccard",
    ):
        require_probability(row[key], f"{path}.{key}")
    require_type(row["top1_changed"], bool, f"{path}.top1_changed")
    for key in ("write_time_top", "fresh_top", "write_time_next_top", "fresh_next_top"):
        validate_compact_top(row[key], f"{path}.{key}")


def validate_lens(raw: dict[str, Any], path: str) -> tuple[list[int], set[int]]:
    require_type(raw.get("lens"), dict, f"{path}.lens")
    require_keys(raw["lens"], {"repo", "filename", "revision", "source_layers", "sampled_layers"}, f"{path}.lens")
    for key in ("repo", "filename", "revision"):
        require_type(raw["lens"][key], str, f"{path}.lens.{key}")
    require_type(raw["lens"]["sampled_layers"], list, f"{path}.lens.sampled_layers")
    layers = raw["lens"]["sampled_layers"]
    require(layers, f"{path}.lens.sampled_layers", "must not be empty")
    require(all(isinstance(layer, int) for layer in layers), f"{path}.lens.sampled_layers", "all layers must be ints")
    require(len(set(layers)) == len(layers), f"{path}.lens.sampled_layers", "duplicate layers")
    return layers, set(layers)


def validate_raw_artifact(raw: dict[str, Any]) -> dict[str, Any]:
    path = "raw"
    require_keys(raw, {"model", "lens", "top_k", "top_n", "note", "demos"}, path)
    require_type(raw["model"], str, f"{path}.model")
    require_type(raw["note"], str, f"{path}.note")
    require_type(raw["top_k"], int, f"{path}.top_k")
    require(raw["top_k"] > 0, f"{path}.top_k", "must be positive")
    require_type(raw["top_n"], int, f"{path}.top_n")
    require(raw["top_n"] > 0, f"{path}.top_n", "must be positive")
    layers, layer_set = validate_lens(raw, path)
    require_type(raw["demos"], dict, f"{path}.demos")
    require(raw["demos"], f"{path}.demos", "must not be empty")

    total_summary_tokens = 0
    total_grid_rows = 0
    for demo_name, demo in raw["demos"].items():
        demo_path = f"{path}.demos.{demo_name}"
        require_type(demo_name, str, demo_path)
        require_type(demo, dict, demo_path)
        require_keys(
            demo,
            {
                "shape",
                "summary_text",
                "token_counts",
                "layer_summary",
                "token_summary",
                "grid_scores",
                "top_token_layers_raw",
                "top_token_layers_semantic",
            },
            demo_path,
        )
        require_type(demo["shape"], str, f"{demo_path}.shape")
        require_type(demo["summary_text"], str, f"{demo_path}.summary_text")
        require_type(demo["token_counts"], dict, f"{demo_path}.token_counts")
        require_keys(
            demo["token_counts"],
            {"old_matched", "fresh_matched", "summary_tokens", "anchor_positions", "grid_rows"},
            f"{demo_path}.token_counts",
        )
        for key in ("old_matched", "fresh_matched", "summary_tokens", "anchor_positions", "grid_rows"):
            require_type(demo["token_counts"][key], int, f"{demo_path}.token_counts.{key}")
            require(demo["token_counts"][key] >= 0, f"{demo_path}.token_counts.{key}", "must be non-negative")

        summary_tokens = demo["token_counts"]["summary_tokens"]
        expected_grid_rows = summary_tokens * len(layers)
        require_type(demo["layer_summary"], list, f"{demo_path}.layer_summary")
        require(len(demo["layer_summary"]) == len(layers), f"{demo_path}.layer_summary", "wrong layer count")
        for i, row in enumerate(demo["layer_summary"]):
            validate_layer_summary(row, f"{demo_path}.layer_summary[{i}]", layer_set)

        require_type(demo["token_summary"], list, f"{demo_path}.token_summary")
        require(len(demo["token_summary"]) == summary_tokens, f"{demo_path}.token_summary", "wrong token count")
        seen_positions = set()
        for i, row in enumerate(demo["token_summary"]):
            validate_token_summary(row, f"{demo_path}.token_summary[{i}]", summary_tokens, layer_set)
            seen_positions.add(row["p"])
        require(seen_positions == set(range(summary_tokens)), f"{demo_path}.token_summary", "positions are not exactly 0..summary_tokens-1")

        require_type(demo["grid_scores"], list, f"{demo_path}.grid_scores")
        require(len(demo["grid_scores"]) == expected_grid_rows, f"{demo_path}.grid_scores", "wrong grid row count")
        require(demo["token_counts"]["grid_rows"] == expected_grid_rows, f"{demo_path}.token_counts.grid_rows", "wrong recorded grid row count")
        seen_grid = set()
        for i, row in enumerate(demo["grid_scores"]):
            validate_metric_row(row, f"{demo_path}.grid_scores[{i}]", summary_tokens, layer_set)
            pair = (row["p"], row["l"])
            require(pair not in seen_grid, f"{demo_path}.grid_scores[{i}]", "duplicate token/layer row")
            seen_grid.add(pair)
        require(len(seen_grid) == expected_grid_rows, f"{demo_path}.grid_scores", "missing token/layer rows")

        for key in ("top_token_layers_raw", "top_token_layers_semantic"):
            require_type(demo[key], list, f"{demo_path}.{key}")
            require(len(demo[key]) <= raw["top_n"], f"{demo_path}.{key}", "more rows than top_n")
            for i, row in enumerate(demo[key]):
                validate_expanded_row(row, f"{demo_path}.{key}[{i}]", summary_tokens, layer_set, raw["top_k"])

        total_summary_tokens += summary_tokens
        total_grid_rows += expected_grid_rows

    return {
        "model": raw["model"],
        "top_k": raw["top_k"],
        "layers": layers,
        "demo_names": set(raw["demos"]),
        "total_summary_tokens": total_summary_tokens,
        "total_grid_rows": total_grid_rows,
    }


def validate_summary_artifact(summary: dict[str, Any], raw_info: dict[str, Any] | None = None) -> None:
    path = "summary"
    require_keys(
        summary,
        {
            "model",
            "lens",
            "top_k",
            "source_raw_file",
            "total_summary_tokens",
            "total_grid_rows",
            "demos",
            "global_layer_summary",
            "global_kind_summary",
            "global_top_tokens_by_mean_change",
            "global_top_token_layers_raw_compact",
            "global_top_token_layers_semantic_compact",
        },
        path,
    )
    require_type(summary["model"], str, f"{path}.model")
    require_type(summary["source_raw_file"], str, f"{path}.source_raw_file")
    require_type(summary["top_k"], int, f"{path}.top_k")
    require(summary["top_k"] > 0, f"{path}.top_k", "must be positive")
    layers, layer_set = validate_lens(summary, path)
    require_type(summary["total_summary_tokens"], int, f"{path}.total_summary_tokens")
    require_type(summary["total_grid_rows"], int, f"{path}.total_grid_rows")
    require_type(summary["demos"], dict, f"{path}.demos")

    if raw_info is not None:
        require(summary["model"] == raw_info["model"], f"{path}.model", "does not match raw")
        require(summary["top_k"] == raw_info["top_k"], f"{path}.top_k", "does not match raw")
        require(layers == raw_info["layers"], f"{path}.lens.sampled_layers", "does not match raw")
        require(set(summary["demos"]) == raw_info["demo_names"], f"{path}.demos", "demo names do not match raw")
        require(summary["total_summary_tokens"] == raw_info["total_summary_tokens"], f"{path}.total_summary_tokens", "does not match raw")
        require(summary["total_grid_rows"] == raw_info["total_grid_rows"], f"{path}.total_grid_rows", "does not match raw")

    computed_tokens = 0
    computed_rows = 0
    for demo_name, demo in summary["demos"].items():
        demo_path = f"{path}.demos.{demo_name}"
        require_type(demo, dict, demo_path)
        require_keys(
            demo,
            {
                "shape",
                "token_counts",
                "layer_summary",
                "top_tokens_by_mean_change",
                "top_token_layers_raw_compact",
                "top_token_layers_semantic_compact",
            },
            demo_path,
        )
        require_type(demo["shape"], str, f"{demo_path}.shape")
        require_type(demo["token_counts"], dict, f"{demo_path}.token_counts")
        require_keys(
            demo["token_counts"],
            {"old_matched", "fresh_matched", "summary_tokens", "anchor_positions", "grid_rows"},
            f"{demo_path}.token_counts",
        )
        summary_tokens = demo["token_counts"]["summary_tokens"]
        grid_rows = demo["token_counts"]["grid_rows"]
        require_type(summary_tokens, int, f"{demo_path}.token_counts.summary_tokens")
        require_type(grid_rows, int, f"{demo_path}.token_counts.grid_rows")
        require(grid_rows == summary_tokens * len(layers), f"{demo_path}.token_counts.grid_rows", "wrong grid row count")

        require_type(demo["layer_summary"], list, f"{demo_path}.layer_summary")
        require(len(demo["layer_summary"]) == len(layers), f"{demo_path}.layer_summary", "wrong layer count")
        for i, row in enumerate(demo["layer_summary"]):
            validate_layer_summary(row, f"{demo_path}.layer_summary[{i}]", layer_set)

        for key in ("top_tokens_by_mean_change",):
            require_type(demo[key], list, f"{demo_path}.{key}")
            for i, row in enumerate(demo[key]):
                validate_token_summary(row, f"{demo_path}.{key}[{i}]", summary_tokens, layer_set)

        for key in ("top_token_layers_raw_compact", "top_token_layers_semantic_compact"):
            require_type(demo[key], list, f"{demo_path}.{key}")
            for i, row in enumerate(demo[key]):
                validate_compact_expanded_row(row, f"{demo_path}.{key}[{i}]", summary_tokens, layer_set)
                require(row["demo"] == demo_name, f"{demo_path}.{key}[{i}].demo", "wrong demo label")

        computed_tokens += summary_tokens
        computed_rows += grid_rows

    require(summary["total_summary_tokens"] == computed_tokens, f"{path}.total_summary_tokens", "does not equal demo sum")
    require(summary["total_grid_rows"] == computed_rows, f"{path}.total_grid_rows", "does not equal demo sum")

    require_type(summary["global_layer_summary"], list, f"{path}.global_layer_summary")
    require(len(summary["global_layer_summary"]) == len(layers), f"{path}.global_layer_summary", "wrong layer count")
    for i, row in enumerate(summary["global_layer_summary"]):
        validate_layer_summary(row, f"{path}.global_layer_summary[{i}]", layer_set)

    require_type(summary["global_kind_summary"], list, f"{path}.global_kind_summary")
    for i, row in enumerate(summary["global_kind_summary"]):
        require_type(row, dict, f"{path}.global_kind_summary[{i}]")
        require_keys(
            row,
            {"kind", "rows", "mean_rank_weighted_distance", "mean_jaccard_distance", "top1_change_rate"},
            f"{path}.global_kind_summary[{i}]",
        )
        require_type(row["kind"], str, f"{path}.global_kind_summary[{i}].kind")
        require_type(row["rows"], int, f"{path}.global_kind_summary[{i}].rows")
        require(row["rows"] >= 0, f"{path}.global_kind_summary[{i}].rows", "negative row count")
        for key in ("mean_rank_weighted_distance", "mean_jaccard_distance", "top1_change_rate"):
            require_probability(row[key], f"{path}.global_kind_summary[{i}].{key}")

    for key in ("global_top_tokens_by_mean_change",):
        require_type(summary[key], list, f"{path}.{key}")
        for i, row in enumerate(summary[key]):
            require_type(row, dict, f"{path}.{key}[{i}]")
            require("demo" in row, f"{path}.{key}[{i}]", "missing demo")
            demo_name = row["demo"]
            require_type(demo_name, str, f"{path}.{key}[{i}].demo")
            require(demo_name in summary["demos"], f"{path}.{key}[{i}].demo", "unknown demo")
            row_without_demo = {k: v for k, v in row.items() if k != "demo"}
            summary_tokens = summary["demos"][demo_name]["token_counts"]["summary_tokens"]
            validate_token_summary(row_without_demo, f"{path}.{key}[{i}]", summary_tokens, layer_set)

    for key in ("global_top_token_layers_raw_compact", "global_top_token_layers_semantic_compact"):
        require_type(summary[key], list, f"{path}.{key}")
        for i, row in enumerate(summary[key]):
            require_type(row, dict, f"{path}.{key}[{i}]")
            demo_name = row.get("demo")
            require_type(demo_name, str, f"{path}.{key}[{i}].demo")
            require(demo_name in summary["demos"], f"{path}.{key}[{i}].demo", "unknown demo")
            summary_tokens = summary["demos"][demo_name]["token_counts"]["summary_tokens"]
            validate_compact_expanded_row(row, f"{path}.{key}[{i}]", summary_tokens, layer_set)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path, help="Raw qwen36_full_layer_sweep.json artifact")
    parser.add_argument("--summary", type=Path, help="Optional compact summary artifact")
    args = parser.parse_args()

    try:
        raw = load_json(args.raw)
        require_type(raw, dict, "raw")
        raw_info = validate_raw_artifact(raw)

        if args.summary is not None:
            summary = load_json(args.summary)
            require_type(summary, dict, "summary")
            validate_summary_artifact(summary, raw_info)

    except ValidationError as exc:
        print(f"INVALID: {exc}")
        return 1

    print(
        "VALID:",
        f"demos={len(raw_info['demo_names'])}",
        f"layers={len(raw_info['layers'])}",
        f"summary_tokens={raw_info['total_summary_tokens']}",
        f"grid_rows={raw_info['total_grid_rows']}",
    )
    if args.summary is not None:
        print("VALID: compact summary matches raw totals and schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

