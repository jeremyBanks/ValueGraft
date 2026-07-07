#!/usr/bin/env python3
"""Analyze batch three-state ValueGraft intervention probe artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("artifact")
    p.add_argument("--output", default="outputs/qwen36_intervention_batch_summary.json")
    p.add_argument("--top-k", type=int, default=8)
    return p.parse_args()


def top_ids(row: dict[str, Any], layer: int, k: int) -> set[int]:
    return {int(item["token_id"]) for item in row["layers"][str(layer)][:k]}


def top_tokens(row: dict[str, Any], layer: int, k: int) -> list[str]:
    return [str(item["token"]) for item in row["layers"][str(layer)][:k]]


def jaccard_distance(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return 0.0
    return 1.0 - (len(a & b) / len(union))


def sequence_rows(case: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return case["post_boundary_probe"]["forced_target_sequences"][name]["rows"]


def available_sequences(case: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    seqs = {
        name: seq["rows"]
        for name, seq in case["post_boundary_probe"]["forced_target_sequences"].items()
    }
    for key, seq in case["post_boundary_probe"].get("alpha_sweep_sequences", {}).items():
        seqs[f"alpha_{key}"] = seq["rows"]
    return seqs


def alpha_sort_key(name: str) -> float:
    try:
        return float(name.removeprefix("alpha_"))
    except ValueError:
        return float("inf")


def focus_positions(case: dict[str, Any]) -> list[int]:
    positions: set[int] = set()
    for vals in case["probe_target"].get("focus_positions", {}).values():
        positions.update(int(v) for v in vals)
    count = int(case["probe_target"]["token_count"])
    return sorted(p for p in positions if 0 <= p < count)


def argmax_metrics(full: list[dict[str, Any]], fresh: list[dict[str, Any]], other: list[dict[str, Any]], positions: list[int]) -> dict[str, int]:
    rescues = regressions = changed = 0
    for i in positions:
        f = int(full[i]["argmax_token_id"])
        b = int(fresh[i]["argmax_token_id"])
        o = int(other[i]["argmax_token_id"])
        if b != f and o == f:
            rescues += 1
        if b == f and o != f:
            regressions += 1
        if o != b:
            changed += 1
    return {"rescues": rescues, "regressions": regressions, "changed_argmax": changed}


def layer_metrics(
    full: list[dict[str, Any]],
    fresh: list[dict[str, Any]],
    other: list[dict[str, Any]],
    layers: list[int],
    positions: list[int],
    k: int,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for layer in layers:
        full_fresh = []
        full_other = []
        fresh_other = []
        for i in positions:
            full_ids = top_ids(full[i], layer, k)
            fresh_ids = top_ids(fresh[i], layer, k)
            other_ids = top_ids(other[i], layer, k)
            full_fresh.append(jaccard_distance(full_ids, fresh_ids))
            full_other.append(jaccard_distance(full_ids, other_ids))
            fresh_other.append(jaccard_distance(fresh_ids, other_ids))
        ff = sum(full_fresh) / len(full_fresh) if full_fresh else 0.0
        fo = sum(full_other) / len(full_other) if full_other else 0.0
        bo = sum(fresh_other) / len(fresh_other) if fresh_other else 0.0
        out[str(layer)] = {
            "full_vs_fresh": round(ff, 6),
            "full_vs_condition": round(fo, 6),
            "fresh_vs_condition": round(bo, 6),
            "closure": round(ff - fo, 6),
        }
    return out


def token_example(
    case: dict[str, Any],
    seqs: dict[str, list[dict[str, Any]]],
    pos: int,
    layer: int,
    k: int,
) -> dict[str, Any]:
    target_tokens = case["probe_target"]["tokens"]
    out = {
        "relative_position": pos,
        "target_token": target_tokens[pos],
        "layer": layer,
        "next_argmax": {},
        "lens_top": {},
    }
    for name in ["full_context", "fresh_compacted", "grafted_compacted", "shifted_grafted_compacted"]:
        if name not in seqs:
            continue
        row = seqs[name][pos]
        out["next_argmax"][name] = row["argmax_token"]
        out["lens_top"][name] = top_tokens(row, layer, k)
    return out


def analyze_case(case: dict[str, Any], layers: list[int], k: int) -> dict[str, Any]:
    if "error" in case:
        return {"case_id": case["case_id"], "error": case["error"]}
    if not case.get("graft", {}).get("available"):
        return {"case_id": case["case_id"], "error": "graft unavailable"}

    seqs = available_sequences(case)
    full = seqs["full_context"]
    fresh = seqs["fresh_compacted"]
    n = len(full)
    all_positions = list(range(n))
    fpos = focus_positions(case)
    if not fpos:
        fpos = all_positions

    condition_names = [
        "alpha0_grafted_compacted",
        "grafted_compacted",
        "shifted_grafted_compacted",
    ]
    condition_names.extend(
        sorted(
            (
                name
                for name in seqs
                if name.startswith("alpha_") and name != "alpha_0"
            ),
            key=alpha_sort_key,
        )
    )
    conditions = {}
    for name in condition_names:
        if name not in seqs:
            continue
        rows = seqs[name]
        conditions[name] = {
            "all_tokens": {
                "argmax": argmax_metrics(full, fresh, rows, all_positions),
                "layers": layer_metrics(full, fresh, rows, layers, all_positions, k),
            },
            "focus_tokens": {
                "positions": fpos,
                "tokens": [case["probe_target"]["tokens"][i] for i in fpos],
                "argmax": argmax_metrics(full, fresh, rows, fpos),
                "layers": layer_metrics(full, fresh, rows, layers, fpos, k),
            },
        }

    candidate_rows = []
    graft = conditions.get("grafted_compacted")
    shifted = conditions.get("shifted_grafted_compacted")
    if graft:
        for pos in all_positions:
            for layer in layers:
                base = layer_metrics(full, fresh, seqs["grafted_compacted"], [layer], [pos], k)[str(layer)]
                shifted_layer = None
                if shifted and "shifted_grafted_compacted" in seqs:
                    shifted_layer = layer_metrics(
                        full,
                        fresh,
                        seqs["shifted_grafted_compacted"],
                        [layer],
                        [pos],
                        k,
                    )[str(layer)]
                candidate_rows.append(
                    {
                        "relative_position": pos,
                        "target_token": case["probe_target"]["tokens"][pos],
                        "layer": layer,
                        "graft_closure": base["closure"],
                        "shifted_closure": None if shifted_layer is None else shifted_layer["closure"],
                        "graft_minus_shifted": None
                        if shifted_layer is None
                        else round(base["closure"] - shifted_layer["closure"], 6),
                        "is_focus": pos in fpos,
                    }
                )
    candidate_rows.sort(
        key=lambda r: (
            r["is_focus"],
            r["graft_minus_shifted"] if r["graft_minus_shifted"] is not None else -999,
            r["graft_closure"],
        ),
        reverse=True,
    )
    examples = [
        token_example(case, seqs, row["relative_position"], row["layer"], k)
        | {"score": row}
        for row in candidate_rows[:12]
    ]
    return {
        "case_id": case["case_id"],
        "demo_name": case["demo_name"],
        "rationale": case["rationale"],
        "summary_source": case.get("summary_source"),
        "probe_user": case["probe_user"],
        "probe_target": case["probe_target"],
        "token_counts": case["token_counts"],
        "graft": {
            "pairs": case["graft"]["pairs"],
            "alpha": case["graft"]["alpha"],
            "changed_value_layers": case["graft"]["changed_value_layers"],
            "negative_control": case["graft"]["negative_control"],
        },
        "conditions": conditions,
        "candidate_examples": examples,
    }


def main() -> None:
    args = parse_args()
    artifact = json.loads(Path(args.artifact).read_text())
    layers = [int(x) for x in artifact["lens"]["sampled_layers"]]
    cases = {
        cid: analyze_case(case, layers, args.top_k)
        for cid, case in artifact["cases"].items()
    }
    summary = {
        "source_artifact": args.artifact,
        "model": artifact["model"],
        "layers": layers,
        "top_k": args.top_k,
        "case_count": len(cases),
        "cases_with_errors": [cid for cid, case in cases.items() if "error" in case],
        "cases": cases,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
