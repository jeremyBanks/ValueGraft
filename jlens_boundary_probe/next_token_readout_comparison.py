#!/usr/bin/env python3
"""Compare J-lens readouts with ordinary next-token logits.

This probe answers a narrow control question for the qualitative J-lens demos:
are the vocabulary-like J-lens readouts mostly the same signal as the model's
actual next-token distribution at the same anchor position?

For selected summary anchors, it records:

- J-lens top-k at sampled layers for the anchor token.
- Actual next-token top-k from the model logits at that same position.
- Top-k set overlap between the lens readout and next-token logits.
- Simple private/lexical concept hits for both distributions.

The examples remain qualitative. The overlap and concept-hit summaries are a
guard against over-reading the lens as if it were just next-token prediction.
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
    choose_layers,
    dtype_from_name,
    find_subsequence,
    render_ids,
    topk_from_logits,
)
from plain_conversation_probe import (
    SUMMARIES as PLAIN_SUMMARIES,
    matched_summary_messages as plain_matched_summary_messages,
)
from pokemon_probe import (
    SUMMARY_VARIANTS as POKEMON_SUMMARIES,
    LabeledPosition,
    labeled_positions_in_text,
    matched_summary_messages as pokemon_matched_summary_messages,
)


@dataclass(frozen=True)
class Case:
    case_id: str
    source: str
    summary_name: str
    phrase: str
    occurrence: int
    private_terms: tuple[str, ...]
    lexical_terms: tuple[str, ...]
    note: str


CASES = [
    Case(
        "pokemon_vacuum_1",
        "pokemon",
        "canonical",
        "Vacuum",
        1,
        ("zig", "pickup", "nick", "nickname"),
        ("vacuum", "clean", "cleaner", "suction", "dust"),
        "Second Vacuum mention: the local text says Vacuum never battles.",
    ),
    Case(
        "pokemon_dex_0",
        "pokemon",
        "canonical",
        "Dex",
        0,
        ("owe", "owed", "owes", "trade", "traded", "trades", "promise", "exchange"),
        ("dex", "nav", "entry", "entries", "completion", "progress", "tracker"),
        "Dex should mean a person/trade obligation, not Pokedex progress.",
    ),
    Case(
        "plain_maple_0",
        "plain",
        "block_party_canonical",
        "Maple",
        0,
        ("refers", "means", "denotes", "room", "library", "="),
        ("street", "ave", "avenue", "park", "town", "city", "maple"),
        "Maple is the library room, not a generic street/place name.",
    ),
    Case(
        "plain_b410_0",
        "plain",
        "block_party_canonical",
        "B-410",
        0,
        ("obsolete", "outdated", "deprecated", "expired", "stale"),
        ("municipal", "city", "civic", "permit", "town", "code"),
        "B-410 is stale; P-771 is current.",
    ),
    Case(
        "plain_crane_0",
        "plain",
        "block_party_canonical",
        "Crane",
        0,
        ("stage", "delivers", "risers", "refers", "means", "company"),
        ("crane", "operator", "rental", "lease", "license", "equipment"),
        "Crane is a stage-rental company, not generic equipment/operator.",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-filename", default=DEFAULT_QWEN36_LENS)
    p.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
    p.add_argument("--output", default="outputs/qwen36_next_token_readout_comparison.json")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--layers", default="48,62")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return p.parse_args()


def summary_for_case(case: Case) -> str:
    if case.source == "pokemon":
        return POKEMON_SUMMARIES[case.summary_name]
    if case.source == "plain":
        return PLAIN_SUMMARIES[case.summary_name]
    raise ValueError(f"unknown source: {case.source}")


def messages_for_case(case: Case, include_old_context: bool) -> list[dict[str, str]]:
    summary = summary_for_case(case)
    if case.source == "pokemon":
        return pokemon_matched_summary_messages(summary, include_old_context)
    if case.source == "plain":
        return plain_matched_summary_messages(summary, include_old_context)
    raise ValueError(f"unknown source: {case.source}")


def selected_label(tokenizer: Any, case: Case) -> LabeledPosition:
    summary = summary_for_case(case)
    labels = [
        item
        for item in labeled_positions_in_text(tokenizer, summary, [case.phrase])
        if item.token_index_in_phrase == 0
    ]
    if case.occurrence >= len(labels):
        raise ValueError(f"{case.case_id}: occurrence {case.occurrence} not found in {labels}")
    return labels[case.occurrence]


def context_window(tokenizer: Any, ids: list[int], pos: int, radius: int = 10) -> str:
    lo = max(0, pos - radius)
    hi = min(len(ids), pos + radius + 1)
    return tokenizer.decode(ids[lo:hi])


def norm_token(token: str) -> str:
    return token.strip().lower()


def term_hits(rows: list[dict[str, Any]], terms: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    lowered_terms = tuple(t.lower() for t in terms)
    for rank, row in enumerate(rows, start=1):
        tok = norm_token(str(row["token"]))
        for term in lowered_terms:
            if term and term in tok:
                out.append({"rank": rank, "term": term, "token": row["token"], "score": row["score"]})
                break
    return out


def overlap(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    aset = {int(row["token_id"]) for row in a}
    bset = {int(row["token_id"]) for row in b}
    inter = aset & bset
    union = aset | bset
    return {
        "intersection": len(inter),
        "union": len(union),
        "jaccard": (len(inter) / len(union)) if union else 0.0,
        "overlap_tokens": [row["token"] for row in a if int(row["token_id"]) in inter],
    }


def input_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def capture_case_state(
    model: torch.nn.Module,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    case: Case,
    include_old_context: bool,
    layers: list[int],
    top_k: int,
) -> dict[str, Any]:
    from jlens.hooks import ActivationRecorder

    summary = summary_for_case(case)
    summary_ids = tokenizer(summary, add_special_tokens=False).input_ids
    ids = render_ids(tokenizer, messages_for_case(case, include_old_context), False)
    summary_start = find_subsequence(ids, summary_ids)
    if summary_start is None:
        raise ValueError(f"{case.case_id}: could not locate summary in rendered ids")
    rel_label = selected_label(tokenizer, case)
    pos = summary_start + rel_label.position

    device_ids = torch.tensor([ids], device=lens_model.input_device)
    record_at = sorted(set(layers))
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=record_at) as rec:
        lens_model.forward(device_ids)
        activations = {i: rec.activations[i].detach() for i in record_at}

    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=input_device(model)), use_cache=False)
        next_logits = out.logits[0, pos, :]

    next_top = topk_from_logits(tokenizer, next_logits, top_k)
    layer_rows = {}
    for layer in layers:
        residual = activations[layer][0, pos].float().unsqueeze(0)
        logits = lens_model.unembed(lens.transport(residual, layer))[0]
        lens_top = topk_from_logits(tokenizer, logits, top_k)
        layer_rows[str(layer)] = {
            "lens_top": lens_top,
            "lens_vs_next_overlap": overlap(lens_top, next_top),
            "lens_private_hits": term_hits(lens_top, case.private_terms),
            "lens_lexical_hits": term_hits(lens_top, case.lexical_terms),
        }

    actual_next_id = ids[pos + 1] if pos + 1 < len(ids) else None
    return {
        "state": "write_time" if include_old_context else "fresh",
        "token_position": pos,
        "summary_relative_position": rel_label.position,
        "anchor_token_id": int(ids[pos]),
        "anchor_token": tokenizer.decode([int(ids[pos])]),
        "actual_next_token_id": int(actual_next_id) if actual_next_id is not None else None,
        "actual_next_token": tokenizer.decode([int(actual_next_id)]) if actual_next_id is not None else None,
        "context_window": context_window(tokenizer, ids, pos),
        "next_token_top": next_top,
        "next_private_hits": term_hits(next_top, case.private_terms),
        "next_lexical_hits": term_hits(next_top, case.lexical_terms),
        "layers": layer_rows,
    }


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for case in result["cases"]:
        for state_name, state in case["states"].items():
            for layer, layer_row in state["layers"].items():
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "state": state_name,
                        "layer": int(layer),
                        "overlap_jaccard": layer_row["lens_vs_next_overlap"]["jaccard"],
                        "overlap_n": layer_row["lens_vs_next_overlap"]["intersection"],
                        "lens_private_hits": len(layer_row["lens_private_hits"]),
                        "next_private_hits": len(state["next_private_hits"]),
                        "lens_lexical_hits": len(layer_row["lens_lexical_hits"]),
                        "next_lexical_hits": len(state["next_lexical_hits"]),
                    }
                )
    if not rows:
        return {"rows": []}
    return {
        "rows": rows,
        "mean_overlap_jaccard": sum(r["overlap_jaccard"] for r in rows) / len(rows),
        "mean_overlap_n": sum(r["overlap_n"] for r in rows) / len(rows),
        "lens_private_hit_rows": sum(1 for r in rows if r["lens_private_hits"] > 0),
        "next_private_hit_rows": sum(1 for r in rows if r["next_private_hits"] > 0),
        "lens_lexical_hit_rows": sum(1 for r in rows if r["lens_lexical_hits"] > 0),
        "next_lexical_hit_rows": sum(1 for r in rows if r["next_lexical_hits"] > 0),
    }


def main() -> None:
    args = parse_args()

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
        "note": (
            "Qualitative control: compare J-lens readouts at selected anchor "
            "tokens with actual model next-token logits at the same positions."
        ),
        "cases": [],
    }

    for case in CASES:
        row = {
            "case_id": case.case_id,
            "source": case.source,
            "summary_name": case.summary_name,
            "phrase": case.phrase,
            "occurrence": case.occurrence,
            "private_terms": list(case.private_terms),
            "lexical_terms": list(case.lexical_terms),
            "note": case.note,
            "states": {
                "write_time": capture_case_state(
                    model, lens_model, lens, tokenizer, case, True, layers, args.top_k
                ),
                "fresh": capture_case_state(
                    model, lens_model, lens, tokenizer, case, False, layers, args.top_k
                ),
            },
        }
        result["cases"].append(row)
        print(f"done {case.case_id}", flush=True)

    result["summary"] = summarize_result(result)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(out_path)


if __name__ == "__main__":
    main()
