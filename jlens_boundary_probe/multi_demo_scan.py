#!/usr/bin/env python3
"""Scan full summaries for write-time vs fresh J-lens readout divergence.

This is qualitative demo tooling. It scans every token in each summary, compares
top-k readouts at matched write-time and fresh positions, and ranks the most
divergent tokens.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from boundary_probe import (
    DEFAULT_LENS_REPO,
    DEFAULT_LENS_REVISION,
    DEFAULT_QWEN36_LENS,
    SUMMARY_REQUEST,
    capture_topk_for_positions,
    choose_layers,
    dtype_from_name,
    find_subsequence,
    render_ids,
)
from plain_conversation_probe import (
    BLOCK_PARTY_MESSAGES,
    SUMMARY_ANCHORS as BLOCK_PARTY_ANCHORS,
    SUMMARIES as BLOCK_PARTY_SUMMARIES,
)
from pokemon_probe import (
    POKEMON_MESSAGES,
    SUMMARY_ANCHORS as POKEMON_ANCHORS,
    SUMMARY_VARIANTS as POKEMON_SUMMARIES,
    labeled_positions_in_text,
)


CODING_MESSAGES = [
    {
        "role": "system",
        "content": "You preserve project-local names, branches, and decisions exactly.",
    },
    {
        "role": "user",
        "content": (
            "We're triaging the checkout incident. Mercury means the billing "
            "service named Mercury, not the planet, not the element, and not "
            "the metrics dashboard. It owns invoice finalization."
        ),
    },
    {
        "role": "assistant",
        "content": "Mercury is the billing service for invoice finalization.",
    },
    {
        "role": "user",
        "content": (
            "Falcon is the old rollback branch. We rejected Falcon because it "
            "drops subscription coupons. The current branch is Raven, and "
            "Raven keeps coupons while disabling the bad tax-cache path."
        ),
    },
    {
        "role": "assistant",
        "content": "Use Raven, not Falcon. Falcon was rejected because it drops coupons.",
    },
    {
        "role": "user",
        "content": (
            "Patch 17 is stale. The hotfix label for the war room is R3. If "
            "someone asks what patch is live, answer R3, not Patch 17."
        ),
    },
    {
        "role": "assistant",
        "content": "Live hotfix label is R3; Patch 17 is stale.",
    },
    {
        "role": "user",
        "content": (
            "Nova is the canary host in us-east-2, not the observability "
            "project. Do not restart Nova; only tail its logs. Restarting "
            "Nova would erase the repro buffer."
        ),
    },
    {
        "role": "assistant",
        "content": "Nova is the canary host; tail logs only, do not restart it.",
    },
    {
        "role": "user",
        "content": (
            "The red button in the runbook means pause webhook delivery. It "
            "does not mean emergency rollback anymore. The rollback path is "
            "the Raven branch plus the R3 hotfix."
        ),
    },
    {
        "role": "assistant",
        "content": "Red button means pause webhooks; rollback is Raven plus R3.",
    },
]


CODING_SUMMARY = (
    "Checkout incident handoff:\n"
    "| Name | Meaning | Decision |\n"
    "| --- | --- | --- |\n"
    "| Mercury | billing service that owns invoice finalization | inspect there, not planet/element/dashboard |\n"
    "| Falcon | old rollback branch | rejected because it drops subscription coupons |\n"
    "| Raven | current rollback branch | keep coupons, disable bad tax-cache path |\n"
    "| Patch 17 | stale patch label | do not cite as live |\n"
    "| R3 | current live hotfix label | use this in war-room status |\n"
    "| Nova | canary host in us-east-2 | tail logs only, do not restart; restart erases repro buffer |\n"
    "| red button | runbook action to pause webhook delivery | not emergency rollback |\n"
    "Rollback path is Raven branch plus R3 hotfix."
)


HOME_MESSAGES = [
    {
        "role": "system",
        "content": "You preserve household planning names and constraints exactly.",
    },
    {
        "role": "user",
        "content": (
            "Trip planning notes. Cedar is the cabin name, not wood for the "
            "deck and not the cedar closet. Cedar is where we sleep Friday."
        ),
    },
    {
        "role": "assistant",
        "content": "Cedar is the cabin where you sleep Friday.",
    },
    {
        "role": "user",
        "content": (
            "Basil is our neighbor Basil, not the herb. Basil has the spare "
            "key and can feed the cat, but Basil cannot water plants because "
            "he leaves Sunday morning."
        ),
    },
    {
        "role": "assistant",
        "content": "Basil is the neighbor with the spare key; cat yes, plants no.",
    },
    {
        "role": "user",
        "content": (
            "Delta is the ferry route, not the airline. We booked Delta 6 at "
            "7:40. The 8:20 route was full. If I ask about Delta, mean the "
            "ferry route."
        ),
    },
    {
        "role": "assistant",
        "content": "Delta means ferry route Delta 6 at 7:40, not the airline.",
    },
    {
        "role": "user",
        "content": (
            "Orange is the color of the lockbox tag, not fruit and not the "
            "Orange Line. The orange key opens the kayak shed; the brass key "
            "opens Cedar."
        ),
    },
    {
        "role": "assistant",
        "content": "Orange marks the kayak-shed key; brass opens Cedar.",
    },
    {
        "role": "user",
        "content": (
            "The printed checklist says bring the big cooler, but that is "
            "stale. We switched to two soft coolers because the big cooler "
            "doesn't fit in Maya's trunk."
        ),
    },
    {
        "role": "assistant",
        "content": "Big cooler is stale/rejected; bring two soft coolers.",
    },
]


HOME_SUMMARY = (
    "Weekend trip checklist:\n"
    "- Cedar = cabin where we sleep Friday; not deck wood or cedar closet.\n"
    "- Basil = neighbor with spare key; he can feed the cat but cannot water plants because he leaves Sunday morning.\n"
    "- Delta = ferry route, not airline; booked Delta 6 at 7:40, 8:20 was full.\n"
    "- Orange = lockbox tag color; orange key opens kayak shed, brass key opens Cedar.\n"
    "- Big cooler instruction is stale; bring two soft coolers because the big cooler does not fit in Maya's trunk."
)


SUPPORT_MESSAGES = [
    {
        "role": "system",
        "content": "You preserve support-ticket names, customer facts, and current statuses exactly.",
    },
    {
        "role": "user",
        "content": (
            "Support ticket notes. Atlas is the customer account, not our "
            "internal Atlas product. Atlas has three seats and one suspended "
            "admin."
        ),
    },
    {
        "role": "assistant",
        "content": "Atlas is the customer account with three seats and one suspended admin.",
    },
    {
        "role": "user",
        "content": (
            "Sage is the customer's bookkeeper, not Sage accounting software. "
            "Sage can approve invoices but cannot reset MFA."
        ),
    },
    {
        "role": "assistant",
        "content": "Sage is a person/bookkeeper; invoices yes, MFA reset no.",
    },
    {
        "role": "user",
        "content": (
            "Ticket T-88 is stale. The active escalation is T-104. If we cite "
            "the escalation number in the handoff, use T-104."
        ),
    },
    {
        "role": "assistant",
        "content": "Active escalation is T-104; T-88 is stale.",
    },
    {
        "role": "user",
        "content": (
            "The amber path means read-only recovery. Do not run the purple "
            "path; purple deletes the sandbox tenant. Customer asked for "
            "amber only."
        ),
    },
    {
        "role": "assistant",
        "content": "Use amber read-only recovery only; do not run purple.",
    },
]


SUPPORT_SUMMARY = (
    "{\n"
    '  "ticket": "support handoff",\n'
    '  "Atlas": "customer account, not internal product; three seats, one suspended admin",\n'
    '  "Sage": "customer bookkeeper, not accounting software; can approve invoices, cannot reset MFA",\n'
    '  "stale_escalation": "T-88",\n'
    '  "active_escalation": "T-104",\n'
    '  "recovery_path": "amber = read-only recovery",\n'
    '  "forbidden_path": "purple deletes sandbox tenant; do not run it"\n'
    "}"
)


@dataclass(frozen=True)
class Demo:
    name: str
    messages: list[dict[str, str]]
    summary: str
    anchors: list[str]
    shape: str


DEMOS = [
    Demo(
        "pokemon_canonical",
        POKEMON_MESSAGES,
        POKEMON_SUMMARIES["canonical"],
        POKEMON_ANCHORS,
        "dense game conversation, bullet summary",
    ),
    Demo(
        "pokemon_lean",
        POKEMON_MESSAGES,
        POKEMON_SUMMARIES["lean"],
        POKEMON_ANCHORS,
        "dense game conversation, compact paragraph summary",
    ),
    Demo(
        "block_party_canonical",
        BLOCK_PARTY_MESSAGES,
        BLOCK_PARTY_SUMMARIES["block_party_canonical"],
        BLOCK_PARTY_ANCHORS,
        "ordinary event planning, bullet summary",
    ),
    Demo(
        "block_party_lean",
        BLOCK_PARTY_MESSAGES,
        BLOCK_PARTY_SUMMARIES["block_party_lean"],
        BLOCK_PARTY_ANCHORS,
        "ordinary event planning, compact paragraph summary",
    ),
    Demo(
        "checkout_table",
        CODING_MESSAGES,
        CODING_SUMMARY,
        ["Mercury", "Falcon", "Raven", "Patch 17", "R3", "Nova", "red button"],
        "software incident, Markdown table summary",
    ),
    Demo(
        "home_checklist",
        HOME_MESSAGES,
        HOME_SUMMARY,
        ["Cedar", "Basil", "Delta", "Orange", "big cooler", "soft coolers"],
        "household trip planning, checklist summary",
    ),
    Demo(
        "support_json",
        SUPPORT_MESSAGES,
        SUPPORT_SUMMARY,
        ["Atlas", "Sage", "T-88", "T-104", "amber", "purple"],
        "support ticket, JSON-like summary",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-filename", default=DEFAULT_QWEN36_LENS)
    p.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
    p.add_argument("--demos", default="all")
    p.add_argument("--output", default="outputs/qwen36_multi_demo_scan.json")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--top-n", type=int, default=40)
    p.add_argument("--layers", default="quarter")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return p.parse_args()


def matched_summary_messages(
    messages: list[dict[str, str]], summary_text: str, include_old_context: bool
) -> list[dict[str, str]]:
    prefix = messages if include_old_context else [messages[0]]
    return [
        *prefix,
        {"role": "user", "content": SUMMARY_REQUEST},
        {"role": "assistant", "content": summary_text},
        {"role": "user", "content": "Continue from this compacted state."},
    ]


def top_tokens(row: dict[str, Any], layer: int) -> list[str]:
    return [x["token"] for x in row["layers"][str(layer)]]


def layer_divergence(old_row: dict[str, Any], fresh_row: dict[str, Any], layer: int) -> float:
    old = set(top_tokens(old_row, layer))
    fresh = set(top_tokens(fresh_row, layer))
    union = old | fresh
    if not union:
        return 0.0
    return 1.0 - (len(old & fresh) / len(union))


def summarize_row(
    tokenizer: Any,
    summary_ids: list[int],
    rel_pos: int,
    old_row: dict[str, Any],
    fresh_row: dict[str, Any],
    layers: list[int],
    labels_by_rel: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    divergences = {str(layer): layer_divergence(old_row, fresh_row, layer) for layer in layers}
    layer_scores = list(divergences.values())
    lo = max(0, rel_pos - 8)
    hi = min(len(summary_ids), rel_pos + 9)
    return {
        "relative_position": rel_pos,
        "token_id": summary_ids[rel_pos],
        "token": tokenizer.decode([summary_ids[rel_pos]]),
        "context_window": tokenizer.decode(summary_ids[lo:hi]),
        "targets": labels_by_rel.get(rel_pos, []),
        "divergence": {
            "mean": sum(layer_scores) / len(layer_scores),
            "max": max(layer_scores),
            "by_layer": divergences,
        },
        "write_time": old_row,
        "fresh": fresh_row,
    }


def scan_demo(
    demo: Demo,
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

    rel_labels = labeled_positions_in_text(
        tokenizer, demo.summary, demo.anchors, label_prefix="summary:"
    )
    labels_by_rel: dict[int, list[dict[str, Any]]] = {}
    for item in rel_labels:
        labels_by_rel.setdefault(item.position, []).append(
            {
                "label": item.label,
                "phrase": item.phrase,
                "token_index_in_phrase": item.token_index_in_phrase,
            }
        )

    positions = list(range(len(summary_ids)))
    old_rows = capture_topk_for_positions(
        lens_model,
        lens,
        tokenizer,
        old_ids,
        layers,
        [old_start + p for p in positions],
        top_k,
    )
    fresh_rows = capture_topk_for_positions(
        lens_model,
        lens,
        tokenizer,
        fresh_ids,
        layers,
        [fresh_start + p for p in positions],
        top_k,
    )
    if len(old_rows) != len(summary_ids) or len(fresh_rows) != len(summary_ids):
        raise RuntimeError(f"scan row count mismatch for {demo.name}")

    paired = []
    ranking = []
    for rel_pos, (old_row, fresh_row) in enumerate(zip(old_rows, fresh_rows)):
        row = summarize_row(
            tokenizer,
            summary_ids,
            rel_pos,
            old_row,
            fresh_row,
            layers,
            labels_by_rel,
        )
        paired.append(row)
        ranking.append(
            {
                "relative_position": rel_pos,
                "token": row["token"],
                "context_window": row["context_window"],
                "targets": row["targets"],
                "divergence": row["divergence"],
            }
        )
    ranking.sort(key=lambda x: (x["divergence"]["mean"], x["divergence"]["max"]), reverse=True)
    top_positions = {item["relative_position"] for item in ranking[:top_n]}
    return {
        "shape": demo.shape,
        "summary_text": demo.summary,
        "token_counts": {
            "old_matched": len(old_ids),
            "fresh_matched": len(fresh_ids),
            "summary_tokens": len(summary_ids),
            "anchor_positions": len(rel_labels),
        },
        "ranked_positions": ranking,
        "top_differences": [row for row in paired if row["relative_position"] in top_positions],
    }


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

    torch_dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
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

    result = {
        "model": args.model,
        "lens": {
            "repo": args.lens_repo,
            "filename": args.lens_filename,
            "revision": args.lens_revision,
            "source_layers": lens.source_layers,
            "sampled_layers": layers,
        },
        "note": (
            "Qualitative illustration only. Each demo scans every summary token "
            "and ranks positions by top-k readout divergence between write-time "
            "and fresh matched wrappers."
        ),
        "top_k": args.top_k,
        "top_n": args.top_n,
        "demos": {},
    }
    for demo in demos:
        result["demos"][demo.name] = scan_demo(
            demo, lens_model, lens, tokenizer, layers, args.top_k, args.top_n
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(out_path)


if __name__ == "__main__":
    main()
