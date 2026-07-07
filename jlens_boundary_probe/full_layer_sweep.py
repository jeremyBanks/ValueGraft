#!/usr/bin/env python3
"""Full summary-token x layer J-lens sweep.

This extends multi_demo_scan.py. Instead of sampling only a few layers, it
records compact change metrics for every aligned summary token at every fitted
J-lens layer, then saves full top-k readouts for the highest-change rows.
"""

from __future__ import annotations

import argparse
import json
import math
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from boundary_probe import (
    DEFAULT_LENS_REPO,
    DEFAULT_LENS_REVISION,
    DEFAULT_QWEN36_LENS,
    choose_layers,
    dtype_from_name,
    find_subsequence,
    render_ids,
    topk_from_logits,
)
from multi_demo_scan import DEMOS, Demo, matched_summary_messages
from pokemon_probe import labeled_positions_in_text


@dataclass(frozen=True)
class StateReadout:
    input_ids: list[int]
    summary_start: int
    lens_top: dict[int, list[list[dict[str, Any]]]]
    next_top: list[list[dict[str, Any]]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-filename", default=DEFAULT_QWEN36_LENS)
    p.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
    p.add_argument("--demos", default="all")
    p.add_argument("--output", default="outputs/qwen36_full_layer_sweep.json")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-n", type=int, default=300)
    p.add_argument("--layers", default="all")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return p.parse_args()


def input_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def ranked_weights(rows: list[dict[str, Any]]) -> dict[int, float]:
    return {int(row["token_id"]): 1.0 / math.log2(rank + 2) for rank, row in enumerate(rows)}


def overlap_metrics(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    a_ids = {int(row["token_id"]) for row in a}
    b_ids = {int(row["token_id"]) for row in b}
    union = a_ids | b_ids
    intersection = a_ids & b_ids
    jaccard = len(intersection) / len(union) if union else 0.0

    aw = ranked_weights(a)
    bw = ranked_weights(b)
    w_union = set(aw) | set(bw)
    denom = sum(max(aw.get(tid, 0.0), bw.get(tid, 0.0)) for tid in w_union)
    numer = sum(min(aw.get(tid, 0.0), bw.get(tid, 0.0)) for tid in w_union)
    weighted = numer / denom if denom else 0.0
    return {
        "jaccard": jaccard,
        "jaccard_distance": 1.0 - jaccard,
        "rank_weighted_overlap": weighted,
        "rank_weighted_distance": 1.0 - weighted,
        "intersection": len(intersection),
    }


def token_kind(token: str) -> str:
    stripped = token.strip()
    if not stripped:
        return "space"
    if all(ch in string.punctuation or ch in "—–‑•|`" for ch in stripped):
        return "punctuation"
    if any(ch.isdigit() for ch in stripped) and any(ch.isalpha() for ch in stripped):
        return "alnum"
    if any(ch.isdigit() for ch in stripped):
        return "number"
    if any(ch.isalpha() for ch in stripped):
        return "word"
    return "other"


def context_window(tokenizer: Any, ids: list[int], rel_pos: int, radius: int = 8) -> str:
    lo = max(0, rel_pos - radius)
    hi = min(len(ids), rel_pos + radius + 1)
    return tokenizer.decode(ids[lo:hi])


def compact_top(rows: list[dict[str, Any]], n: int = 8) -> list[str]:
    return [str(row["token"]) for row in rows[:n]]


def label_index(tokenizer: Any, demo: Demo) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    labels = labeled_positions_in_text(
        tokenizer, demo.summary, demo.anchors, label_prefix="summary:"
    )
    for item in labels:
        out.setdefault(item.position, []).append(
            {
                "label": item.label,
                "phrase": item.phrase,
                "token_index_in_phrase": item.token_index_in_phrase,
            }
        )
    return out


def capture_state_readout(
    model: torch.nn.Module,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    input_ids: list[int],
    summary_start: int,
    summary_len: int,
    layers: list[int],
    top_k: int,
) -> StateReadout:
    from jlens.hooks import ActivationRecorder

    positions = [summary_start + i for i in range(summary_len)]
    device_ids = torch.tensor([input_ids], device=lens_model.input_device)
    record_at = sorted(set(layers))
    lens_top: dict[int, list[list[dict[str, Any]]]] = {}
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=record_at) as rec:
        lens_model.forward(device_ids)
        for layer in record_at:
            residual = rec.activations[layer][0, positions].float()
            logits = lens_model.unembed(lens.transport(residual, layer))
            vals, idx = torch.topk(logits.float().cpu(), top_k, dim=-1)
            layer_rows: list[list[dict[str, Any]]] = []
            for row_vals, row_idx in zip(vals.tolist(), idx.tolist()):
                layer_rows.append(
                    [
                        {
                            "token_id": int(token_id),
                            "token": tokenizer.decode([int(token_id)]),
                            "score": float(score),
                        }
                        for score, token_id in zip(row_vals, row_idx)
                    ]
                )
            lens_top[layer] = layer_rows
            del residual, logits, vals, idx

    with torch.no_grad():
        out = model(input_ids=torch.tensor([input_ids], device=input_device(model)), use_cache=False)
        logits = out.logits[0, positions, :]
        next_top = [topk_from_logits(tokenizer, logits[i], top_k) for i in range(summary_len)]
        del logits, out

    return StateReadout(input_ids=input_ids, summary_start=summary_start, lens_top=lens_top, next_top=next_top)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize_demo(
    demo: Demo,
    tokenizer: Any,
    summary_ids: list[int],
    layers: list[int],
    labels_by_rel: dict[int, list[dict[str, Any]]],
    old: StateReadout,
    fresh: StateReadout,
    top_n: int,
) -> dict[str, Any]:
    grid_scores: list[dict[str, Any]] = []
    by_layer: dict[int, list[dict[str, Any]]] = {layer: [] for layer in layers}
    by_token: dict[int, list[dict[str, Any]]] = {pos: [] for pos in range(len(summary_ids))}

    for rel_pos, token_id in enumerate(summary_ids):
        for layer in layers:
            old_top = old.lens_top[layer][rel_pos]
            fresh_top = fresh.lens_top[layer][rel_pos]
            m = overlap_metrics(old_top, fresh_top)
            next_overlap_old = overlap_metrics(old.next_top[rel_pos], old_top)
            next_overlap_fresh = overlap_metrics(fresh.next_top[rel_pos], fresh_top)
            row = {
                "p": rel_pos,
                "l": layer,
                "jd": round(m["jaccard_distance"], 6),
                "j": round(m["jaccard"], 6),
                "wd": round(m["rank_weighted_distance"], 6),
                "wo": round(m["rank_weighted_overlap"], 6),
                "i": m["intersection"],
                "top1_changed": old_top[0]["token_id"] != fresh_top[0]["token_id"],
                "old_next_j": round(next_overlap_old["jaccard"], 6),
                "fresh_next_j": round(next_overlap_fresh["jaccard"], 6),
            }
            grid_scores.append(row)
            by_layer[layer].append(row)
            by_token[rel_pos].append(row)

    token_summary = []
    for rel_pos, rows in by_token.items():
        best = max(rows, key=lambda r: (r["wd"], r["jd"]))
        token = tokenizer.decode([summary_ids[rel_pos]])
        token_summary.append(
            {
                "p": rel_pos,
                "token_id": int(summary_ids[rel_pos]),
                "token": token,
                "kind": token_kind(token),
                "context": context_window(tokenizer, summary_ids, rel_pos),
                "targets": labels_by_rel.get(rel_pos, []),
                "mean_wd": round(mean([r["wd"] for r in rows]), 6),
                "max_wd": best["wd"],
                "best_layer": best["l"],
                "top1_change_rate": round(mean([1.0 if r["top1_changed"] else 0.0 for r in rows]), 6),
                "old_best_top": compact_top(old.lens_top[best["l"]][rel_pos]),
                "fresh_best_top": compact_top(fresh.lens_top[best["l"]][rel_pos]),
                "old_next_top": compact_top(old.next_top[rel_pos]),
                "fresh_next_top": compact_top(fresh.next_top[rel_pos]),
            }
        )

    layer_summary = []
    for layer, rows in by_layer.items():
        layer_summary.append(
            {
                "layer": layer,
                "mean_jaccard_distance": round(mean([r["jd"] for r in rows]), 6),
                "mean_rank_weighted_distance": round(mean([r["wd"] for r in rows]), 6),
                "top1_change_rate": round(mean([1.0 if r["top1_changed"] else 0.0 for r in rows]), 6),
                "mean_old_lens_vs_next_jaccard": round(mean([r["old_next_j"] for r in rows]), 6),
                "mean_fresh_lens_vs_next_jaccard": round(mean([r["fresh_next_j"] for r in rows]), 6),
            }
        )

    ranked = sorted(grid_scores, key=lambda r: (r["wd"], r["jd"]), reverse=True)
    semantic_ranked = [
        r
        for r in ranked
        if token_kind(tokenizer.decode([summary_ids[r["p"]]])) not in {"space", "punctuation"}
    ]

    def expand(row: dict[str, Any]) -> dict[str, Any]:
        p = row["p"]
        layer = row["l"]
        token = tokenizer.decode([summary_ids[p]])
        return {
            **row,
            "token_id": int(summary_ids[p]),
            "token": token,
            "kind": token_kind(token),
            "context": context_window(tokenizer, summary_ids, p),
            "targets": labels_by_rel.get(p, []),
            "write_time_top": old.lens_top[layer][p],
            "fresh_top": fresh.lens_top[layer][p],
            "write_time_next_top": old.next_top[p],
            "fresh_next_top": fresh.next_top[p],
        }

    return {
        "shape": demo.shape,
        "summary_text": demo.summary,
        "token_counts": {
            "old_matched": len(old.input_ids),
            "fresh_matched": len(fresh.input_ids),
            "summary_tokens": len(summary_ids),
            "anchor_positions": sum(len(v) for v in labels_by_rel.values()),
            "grid_rows": len(grid_scores),
        },
        "layer_summary": layer_summary,
        "token_summary": token_summary,
        "grid_scores": grid_scores,
        "top_token_layers_raw": [expand(r) for r in ranked[:top_n]],
        "top_token_layers_semantic": [expand(r) for r in semantic_ranked[:top_n]],
    }


def scan_demo(
    demo: Demo,
    model: torch.nn.Module,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    layers: list[int],
    top_k: int,
    top_n: int,
) -> dict[str, Any]:
    summary_ids = tokenizer(demo.summary, add_special_tokens=False).input_ids
    old_ids = render_ids(tokenizer, matched_summary_messages(demo.messages, demo.summary, True), False)
    fresh_ids = render_ids(tokenizer, matched_summary_messages(demo.messages, demo.summary, False), False)
    old_start = find_subsequence(old_ids, summary_ids)
    fresh_start = find_subsequence(fresh_ids, summary_ids)
    if old_start is None or fresh_start is None:
        raise ValueError(f"could not find summary span for {demo.name}")

    labels_by_rel = label_index(tokenizer, demo)
    old = capture_state_readout(
        model, lens_model, lens, tokenizer, old_ids, old_start, len(summary_ids), layers, top_k
    )
    fresh = capture_state_readout(
        model, lens_model, lens, tokenizer, fresh_ids, fresh_start, len(summary_ids), layers, top_k
    )
    return summarize_demo(demo, tokenizer, summary_ids, layers, labels_by_rel, old, fresh, top_n)


def main() -> None:
    args = parse_args()
    if args.demos == "all":
        demos = DEMOS
    else:
        names = {x.strip() for x in args.demos.split(",") if x.strip()}
        demos = [demo for demo in DEMOS if demo.name in names]
        missing = names - {demo.name for demo in demos}
        if missing:
            raise ValueError(f"unknown demos: {sorted(missing)}")

    import jlens

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype_from_name(args.dtype),
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    lens = jlens.JacobianLens.from_pretrained(
        args.lens_repo,
        filename=args.lens_filename,
        revision=args.lens_revision,
    )
    layers = choose_layers(args.layers, lens_model.n_layers, lens.source_layers)

    result: dict[str, Any] = {
        "model": args.model,
        "lens": {
            "repo": args.lens_repo,
            "filename": args.lens_filename,
            "revision": args.lens_revision,
            "source_layers": lens.source_layers,
            "sampled_layers": layers,
        },
        "top_k": args.top_k,
        "top_n": args.top_n,
        "note": (
            "Broad qualitative sweep: every summary token crossed with every "
            "requested fitted J-lens layer. grid_scores keep compact metrics "
            "for all token-layer rows; top_token_layers_* include full top-k "
            "readouts for the largest write-time/fresh changes."
        ),
        "demos": {},
    }
    for demo in demos:
        print(f"scan {demo.name}", flush=True)
        result["demos"][demo.name] = scan_demo(
            demo, model, lens_model, lens, tokenizer, layers, args.top_k, args.top_n
        )
        print(f"done {demo.name}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(out_path)


if __name__ == "__main__":
    main()
