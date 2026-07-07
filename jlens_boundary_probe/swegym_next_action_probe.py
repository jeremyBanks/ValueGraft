#!/usr/bin/env python3
"""J-lens readouts for SWE-Gym next-action tokens.

This is qualitative probe tooling. It reloads existing SWE-Gym trajectories,
cuts them before an assistant action, generates a compact summary, and compares
top-k lens readouts for the true next action under:

1. the full context before the action, and
2. a compacted context consisting of a summary plus recent tail messages.

It does not run an agent loop and does not modify the target repositories.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from boundary_probe import (
    DEFAULT_LENS_REPO,
    DEFAULT_LENS_REVISION,
    DEFAULT_QWEN36_LENS,
    capture_topk_for_positions,
    choose_layers,
    dtype_from_name,
    find_subsequence,
    render_ids,
    token_tensor,
)


MIN_TOK = 6000
MAX_TOK = 15000

SUMMARY_REQUEST_SWE = (
    "Context is about to be condensed. This is not a request to continue the "
    "coding task, and you must not call tools or write function-call XML. "
    "Write a compact plain-English state summary for an AI coding agent that "
    "will continue this task seeing only the summary plus recent messages. "
    "Preserve the task, exact repository paths, files already inspected, "
    "commands or tools used, errors encountered, decisions made, and the next "
    "likely action. Be concrete with names, paths, numbers, and line ranges. "
    "No commentary before or after."
)
SUMMARY_PREFILL = "Summary:\n"


@dataclass(frozen=True)
class Cut:
    ctx: list[dict[str, str]]
    target: dict[str, str]
    cut_msg: int
    n_tokens_full: int
    n_ctx_tokens: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-filename", default=DEFAULT_QWEN36_LENS)
    p.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
    p.add_argument("--parquet", default="../swegym.parquet")
    p.add_argument("--indices", default="5")
    p.add_argument("--output", default="outputs/qwen36_swegym_next_action_probe.json")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--top-n", type=int, default=24)
    p.add_argument("--layers", default="quarter")
    p.add_argument("--max-new-summary-tokens", type=int, default=220)
    p.add_argument("--max-target-tokens", type=int, default=120)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return p.parse_args()


def clean_messages(raw: list[Any]) -> list[dict[str, str]]:
    out = []
    for m in raw:
        role = str(m["role"])
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        out.append({"role": role, "content": content})
    return out


def load_trajectories(path: str) -> list[list[dict[str, str]]]:
    import pandas as pd

    df = pd.read_parquet(path)
    return [clean_messages(list(row)) for row in df["messages"]]


def im_start_id(tokenizer: Any) -> int:
    ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"unexpected <|im_start|> ids: {ids}")
    return int(ids[0])


def canonical_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    """Canonical non-final rendering for Qwen-style chat templates."""
    dummy = messages + [{"role": "user", "content": "x"}]
    ids = render_ids(tokenizer, dummy, False)
    starts = [i for i, tok in enumerate(ids) if tok == im_start_id(tokenizer)]
    if len(starts) != len(messages) + 1:
        raise ValueError(f"{len(starts)} im_starts for {len(messages)} messages")
    return ids[: starts[len(messages)]]


def message_token_starts(tokenizer: Any, ids: list[int], n_msgs: int) -> list[int]:
    starts = [i for i, tok in enumerate(ids) if tok == im_start_id(tokenizer)]
    if len(starts) != n_msgs:
        raise ValueError(f"{len(starts)} im_starts for {n_msgs} messages")
    return starts


def find_cut(tokenizer: Any, messages: list[dict[str, str]]) -> Cut | None:
    ids = canonical_ids(tokenizer, messages)
    if not (MIN_TOK <= len(ids) <= MAX_TOK):
        return None
    starts = message_token_starts(tokenizer, ids, len(messages))
    candidates = [
        i
        for i in range(4, len(messages))
        if messages[i]["role"] == "assistant"
        and 0.60 * len(ids) <= starts[i] <= 0.85 * len(ids)
    ]
    if not candidates:
        return None
    cut = candidates[len(candidates) // 2]
    ctx = messages[:cut]
    if ctx[-1]["role"] != "user":
        return None
    return Cut(ctx, messages[cut], cut, len(ids), starts[cut])


def generate_summary(
    model: torch.nn.Module,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> dict[str, Any]:
    req_messages = messages + [{"role": "user", "content": SUMMARY_REQUEST_SWE}]
    req_ids = render_ids(tokenizer, req_messages, True)
    prefill_ids = tokenizer(SUMMARY_PREFILL, add_special_tokens=False).input_ids
    prompt_ids = req_ids + prefill_ids
    eos = model.config.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    with torch.no_grad():
        out = model.generate(
            input_ids=token_tensor(model, prompt_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad,
            eos_token_id=eos,
        )
    all_ids = out[0].tolist()
    gen_ids = all_ids[len(prompt_ids) :]
    if eos is not None:
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        gen_ids = [tok for tok in gen_ids if tok not in eos_ids]
    summary_text = (SUMMARY_PREFILL + tokenizer.decode(gen_ids, skip_special_tokens=True)).strip()
    return {
        "text": summary_text,
        "request_ids": req_ids,
        "gen_ids": tokenizer(summary_text, add_special_tokens=False).input_ids,
    }


def build_compacted_messages(
    messages: list[dict[str, str]], summary_text: str, tail_start_msg: int
) -> list[dict[str, str]]:
    note = (
        "[Context note] Earlier parts of this coding task were compacted. "
        "Summary of what came before:\n\n" + summary_text
    )
    return [
        messages[0],
        {"role": "assistant", "content": note},
        *messages[tail_start_msg:],
    ]


def tail_start_for_context(tokenizer: Any, ctx: list[dict[str, str]]) -> int:
    ids = canonical_ids(tokenizer, ctx)
    starts = message_token_starts(tokenizer, ids, len(ctx))
    target = 0.75 * len(ids)
    return min(range(1, len(ctx)), key=lambda i: abs(starts[i] - target))


def top_tokens(row: dict[str, Any], layer: int) -> list[str]:
    return [x["token"] for x in row["layers"][str(layer)]]


def layer_divergence(full_row: dict[str, Any], compact_row: dict[str, Any], layer: int) -> float:
    full = set(top_tokens(full_row, layer))
    compact = set(top_tokens(compact_row, layer))
    union = full | compact
    if not union:
        return 0.0
    return 1.0 - (len(full & compact) / len(union))


def action_anchor_phrases(target_text: str) -> list[str]:
    phrases: list[str] = []
    patterns = [
        r"<function=([^>]+)>",
        r"<parameter=command>(.*?)</parameter>",
        r"<parameter=path>(.*?)</parameter>",
        r"<parameter=view_range>(.*?)</parameter>",
        r"/workspace/[^\s<]+",
        r"[\w.-]+\.py",
        r"\[[0-9]+,\s*[0-9]+\]",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, target_text, flags=re.S):
            value = match.group(1) if match.groups() else match.group(0)
            value = value.strip()
            if value and value not in phrases:
                phrases.append(value)
    return phrases


def offset_mapping_for_text(
    tokenizer: Any, text: str, text_ids: list[int]
) -> list[tuple[int, int]] | None:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception:
        return None
    offsets = encoded.get("offset_mapping")
    ids = encoded.get("input_ids")
    if offsets is None or ids is None or list(ids) != text_ids:
        return None
    return [(int(start), int(end)) for start, end in offsets]


def phrase_char_occurrences(text: str, phrase: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    while True:
        found = text.find(phrase, start)
        if found < 0:
            return spans
        spans.append((found, found + len(phrase)))
        start = found + max(1, len(phrase))


def token_positions_for_char_span(
    offsets: list[tuple[int, int]], char_span: tuple[int, int]
) -> list[int]:
    lo, hi = char_span
    return [
        pos
        for pos, (start, end) in enumerate(offsets)
        if end > lo and start < hi
    ]


def labeled_positions(
    tokenizer: Any, text: str, text_ids: list[int], phrases: list[str]
) -> dict[int, list[dict[str, Any]]]:
    labels: dict[int, list[dict[str, Any]]] = {}
    offsets = offset_mapping_for_text(tokenizer, text, text_ids)
    if offsets is not None:
        for phrase in phrases:
            for char_span in phrase_char_occurrences(text, phrase):
                positions = token_positions_for_char_span(offsets, char_span)
                for i, pos in enumerate(positions):
                    labels.setdefault(pos, []).append(
                        {
                            "phrase": phrase,
                            "token_index_in_phrase": i,
                            "token": tokenizer.decode([text_ids[pos]]),
                            "char_span": list(char_span),
                        }
                    )
        return labels

    for phrase in phrases:
        phrase_ids = tokenizer(phrase, add_special_tokens=False).input_ids
        start = find_subsequence(text_ids, phrase_ids)
        if start is None:
            continue
        for i, tok in enumerate(phrase_ids):
            labels.setdefault(start + i, []).append(
                {
                    "phrase": phrase,
                    "token_index_in_phrase": i,
                    "token": tokenizer.decode([tok]),
                }
            )
    return labels


def phrase_spans(
    tokenizer: Any, text: str, text_ids: list[int], phrases: list[str]
) -> list[dict[str, Any]]:
    spans = []
    offsets = offset_mapping_for_text(tokenizer, text, text_ids)
    if offsets is not None:
        for phrase in phrases:
            for char_span in phrase_char_occurrences(text, phrase):
                positions = token_positions_for_char_span(offsets, char_span)
                if not positions:
                    continue
                spans.append(
                    {
                        "phrase": phrase,
                        "start": min(positions),
                        "end": max(positions) + 1,
                        "n_tokens": len(positions),
                        "tokens": [tokenizer.decode([text_ids[pos]]) for pos in positions],
                        "char_span": list(char_span),
                    }
                )
        return spans

    for phrase in phrases:
        phrase_ids = tokenizer(phrase, add_special_tokens=False).input_ids
        start = find_subsequence(text_ids, phrase_ids)
        if start is None:
            continue
        spans.append(
            {
                "phrase": phrase,
                "start": start,
                "end": start + len(phrase_ids),
                "n_tokens": len(phrase_ids),
                "tokens": [tokenizer.decode([tok]) for tok in phrase_ids],
            }
        )
    return spans


def summarize_token_row(
    tokenizer: Any,
    target_ids: list[int],
    rel_pos: int,
    full_row: dict[str, Any],
    compact_row: dict[str, Any],
    layers: list[int],
    labels_by_rel: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    divergences = {
        str(layer): layer_divergence(full_row, compact_row, layer) for layer in layers
    }
    scores = list(divergences.values())
    lo = max(0, rel_pos - 8)
    hi = min(len(target_ids), rel_pos + 9)
    return {
        "relative_position": rel_pos,
        "token_id": int(target_ids[rel_pos]),
        "token": tokenizer.decode([target_ids[rel_pos]]),
        "context_window": tokenizer.decode(target_ids[lo:hi]),
        "anchors": labels_by_rel.get(rel_pos, []),
        "divergence": {
            "mean": sum(scores) / len(scores),
            "max": max(scores),
            "by_layer": divergences,
        },
        "full_context": full_row,
        "compacted_context": compact_row,
    }


def summarize_spans(
    rows: list[dict[str, Any]], spans: list[dict[str, Any]], layers: list[int]
) -> list[dict[str, Any]]:
    out = []
    by_pos = {row["relative_position"]: row for row in rows}
    for span in spans:
        members = [
            by_pos[pos]
            for pos in range(span["start"], span["end"])
            if pos in by_pos
        ]
        if not members:
            continue
        representative = max(
            members,
            key=lambda row: (row["divergence"]["mean"], row["divergence"]["max"]),
        )
        layer_preview = {}
        for layer in layers:
            key = str(layer)
            layer_preview[key] = {
                "representative_token": representative["token"],
                "full_context": representative["full_context"]["layers"][key][:8],
                "compacted_context": representative["compacted_context"]["layers"][key][:8],
            }
        mean_scores = [row["divergence"]["mean"] for row in members]
        max_scores = [row["divergence"]["max"] for row in members]
        out.append(
            {
                **span,
                "divergence": {
                    "mean": sum(mean_scores) / len(mean_scores),
                    "max": max(max_scores),
                    "representative_position": representative["relative_position"],
                    "representative_token": representative["token"],
                },
                "layer_preview": layer_preview,
            }
        )
    out.sort(key=lambda row: (row["divergence"]["mean"], row["divergence"]["max"]), reverse=True)
    return out


def scan_next_action(
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    cut: Cut,
    summary: dict[str, Any],
    layers: list[int],
    top_k: int,
    top_n: int,
    max_target_tokens: int,
) -> dict[str, Any]:
    ctx_ids = canonical_ids(tokenizer, cut.ctx)
    tail_start_msg = tail_start_for_context(tokenizer, cut.ctx)
    compacted = build_compacted_messages(cut.ctx, summary["text"], tail_start_msg)
    compact_ids = canonical_ids(tokenizer, compacted)

    full_prompt = render_ids(tokenizer, cut.ctx, True)
    compact_prompt = render_ids(tokenizer, compacted, True)
    if full_prompt[: len(ctx_ids)] != ctx_ids:
        raise ValueError("full generation prompt is not prefixed by canonical ctx")
    if compact_prompt[: len(compact_ids)] != compact_ids:
        raise ValueError("compacted generation prompt is not prefixed by canonical ctx")

    target_text = str(cut.target["content"])
    target_ids_all = tokenizer(target_text, add_special_tokens=False).input_ids
    target_ids = target_ids_all[:max_target_tokens]
    if not target_ids:
        raise ValueError("empty target action")
    positions = list(range(len(target_ids)))
    full_ids = full_prompt + target_ids
    compact_ids_for_action = compact_prompt + target_ids
    full_rows = capture_topk_for_positions(
        lens_model,
        lens,
        tokenizer,
        full_ids,
        layers,
        [len(full_prompt) + pos for pos in positions],
        top_k,
    )
    compact_rows = capture_topk_for_positions(
        lens_model,
        lens,
        tokenizer,
        compact_ids_for_action,
        layers,
        [len(compact_prompt) + pos for pos in positions],
        top_k,
    )
    anchors = action_anchor_phrases(target_text)
    labels_by_rel = labeled_positions(tokenizer, target_text, target_ids, anchors)
    spans = phrase_spans(tokenizer, target_text, target_ids, anchors)
    paired = [
        summarize_token_row(
            tokenizer,
            target_ids,
            rel_pos,
            full_row,
            compact_row,
            layers,
            labels_by_rel,
        )
        for rel_pos, (full_row, compact_row) in enumerate(zip(full_rows, compact_rows))
    ]
    ranking = [
        {
            "relative_position": row["relative_position"],
            "token": row["token"],
            "context_window": row["context_window"],
            "anchors": row["anchors"],
            "divergence": row["divergence"],
        }
        for row in paired
    ]
    ranking.sort(
        key=lambda row: (row["divergence"]["mean"], row["divergence"]["max"]),
        reverse=True,
    )
    selected = {row["relative_position"] for row in ranking[:top_n]}
    for pos, labels in labels_by_rel.items():
        if labels:
            selected.add(pos)
    return {
        "cut": {
            "cut_msg": cut.cut_msg,
            "n_tokens_full": cut.n_tokens_full,
            "n_ctx_tokens": cut.n_ctx_tokens,
            "tail_start_msg": tail_start_msg,
        },
        "token_counts": {
            "ctx": len(ctx_ids),
            "summary": len(summary["gen_ids"]),
            "compacted": len(compact_ids),
            "full_action_prompt": len(full_prompt),
            "compact_action_prompt": len(compact_prompt),
            "target_all": len(target_ids_all),
            "target_scanned": len(target_ids),
        },
        "summary_text": summary["text"],
        "target_text": target_text,
        "action_anchors": anchors,
        "anchor_spans": spans,
        "span_summaries": summarize_spans(paired, spans, layers),
        "ranked_positions": ranking,
        "selected_rows": [row for row in paired if row["relative_position"] in selected],
    }


def main() -> None:
    args = parse_args()
    indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]

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
    trajectories = load_trajectories(args.parquet)

    results: dict[str, Any] = {
        "model": args.model,
        "lens": {
            "repo": args.lens_repo,
            "filename": args.lens_filename,
            "revision": args.lens_revision,
            "source_layers": lens.source_layers,
            "sampled_layers": layers,
        },
        "summary_request": SUMMARY_REQUEST_SWE,
        "indices": indices,
        "trajectories": {},
    }
    for idx in indices:
        cut = find_cut(tokenizer, trajectories[idx])
        if cut is None:
            results["trajectories"][str(idx)] = {"skipped": "no eligible cut"}
            continue
        summary = generate_summary(model, tokenizer, cut.ctx, args.max_new_summary_tokens)
        results["trajectories"][str(idx)] = scan_next_action(
            lens_model,
            lens,
            tokenizer,
            cut,
            summary,
            layers,
            args.top_k,
            args.top_n,
            args.max_target_tokens,
        )
        print(f"scanned trajectory {idx}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
