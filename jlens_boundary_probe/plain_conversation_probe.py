#!/usr/bin/env python3
"""J-lens readouts for ordinary conversation-summarization demos.

This complements the Pokemon demo with a more mundane planning conversation.
It is qualitative illustration only.
"""

from __future__ import annotations

import argparse
import json
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
    render_text,
)
from pokemon_probe import (
    LabeledPosition,
    capture_labeled,
    labeled_positions_in_text,
    probe_labels_for_rendered_text,
)


SYSTEM = (
    "You are a careful assistant. Preserve private names, decisions, stale "
    "facts, constraints, and local jargon exactly."
)


BLOCK_PARTY_MESSAGES = [
    {"role": "system", "content": SYSTEM},
    {
        "role": "user",
        "content": (
            "I'm organizing the Riverside block party and using this chat as "
            "the working memory. First naming thing: Maple is the library's "
            "Maple Room, not the tree grove outside and definitely not the "
            "syrup sponsor. We use Maple for quiet storage and volunteer "
            "check-in."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Noted: Maple means the library's Maple Room for storage and "
            "volunteer check-in, not trees or syrup."
        ),
    },
    {
        "role": "user",
        "content": (
            "Robin is our volunteer coordinator, the human Robin from the "
            "library board. She has the van keys and the vendor phone list, "
            "but she can't lift boxes because of her shoulder. Don't assign "
            "Robin to setup lifting."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Robin is the person coordinating volunteers, holding the van "
            "keys and vendor list, but she should not lift boxes."
        ),
    },
    {
        "role": "user",
        "content": (
            "We compared three rain plans. Red meant cancel everything. Green "
            "meant tents in the parking lot, but the fire marshal rejected it "
            "because it blocks the lane. Blue means move music into Harbor "
            "Hall and keep food trucks on Alder Street. We chose Blue."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Decision recorded: Blue is the chosen rain plan: music in Harbor "
            "Hall, food trucks on Alder Street. Green is rejected because it "
            "blocks the fire lane; Red means cancellation."
        ),
    },
    {
        "role": "user",
        "content": (
            "The old permit number B-410 is stale. City events reissued it "
            "yesterday as P-771, and that is the number that should go on the "
            "insurance form. If I ask about the permit, use P-771, not B-410."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Permit number is P-771. B-410 is stale and should not be used on "
            "the insurance form."
        ),
    },
    {
        "role": "user",
        "content": (
            "Orchid is the dessert vendor, not flowers for the welcome table. "
            "Their invoice is still missing. Orchid wants power on circuit C, "
            "and they promised nut-free labels by Thursday night."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Orchid means the dessert vendor. Missing invoice, needs circuit C "
            "power, and owes nut-free labels by Thursday night."
        ),
    },
    {
        "role": "user",
        "content": (
            "About tables: the big oak table from the archive looked charming, "
            "but we ruled it out because it narrows the ADA route. Use six "
            "folding tables instead. If anyone says the big table, remind "
            "them it was rejected."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "The big oak table is rejected because it narrows the ADA route. "
            "Use six folding tables instead."
        ),
    },
    {
        "role": "user",
        "content": (
            "Schedule snag: Friday is not the party day. Friday is the final "
            "insurance upload deadline. The party is Saturday at 2 pm, and "
            "setup starts Saturday at 10. Please keep Friday separate."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Friday is only the insurance upload deadline. Party is Saturday "
            "at 2 pm, setup Saturday at 10."
        ),
    },
    {
        "role": "user",
        "content": (
            "Last detail before you summarize: Crane is the stage rental "
            "company, not equipment we are renting. They deliver risers at 9 "
            "on Saturday. Their driver calls Robin, but Robin only hands over "
            "keys; Mateo and Jules unload."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Crane is the stage rental company. They deliver risers at 9 on "
            "Saturday; driver calls Robin, but Mateo and Jules unload."
        ),
    },
]


SUMMARIES = {
    "block_party_canonical": (
        "Riverside block party state:\n"
        "- Maple means the library's Maple Room for storage and volunteer "
        "check-in, not the tree grove or syrup sponsor.\n"
        "- Robin is the human volunteer coordinator from the library board. "
        "She has the van keys and vendor phone list, but must not lift boxes "
        "because of her shoulder.\n"
        "- Rain plan decision: Blue is chosen. Blue means music moves into "
        "Harbor Hall and food trucks stay on Alder Street. Green was rejected "
        "by the fire marshal because tents block the lane. Red means cancel.\n"
        "- Permit: B-410 is stale. Use P-771 on the insurance form.\n"
        "- Orchid is the dessert vendor, not flowers. Orchid's invoice is "
        "missing; they need circuit C power and owe nut-free labels by "
        "Thursday night.\n"
        "- The big oak table was rejected because it narrows the ADA route. "
        "Use six folding tables instead.\n"
        "- Friday is the insurance upload deadline, not the party day. The "
        "party is Saturday at 2 pm and setup starts Saturday at 10.\n"
        "- Crane is the stage rental company, not equipment. Crane delivers "
        "risers at 9 on Saturday; their driver calls Robin, but Mateo and "
        "Jules unload."
    ),
    "block_party_lean": (
        "Riverside block party: Maple = library Maple Room, not tree/syrup. "
        "Robin = human volunteer coordinator with van keys and vendor list; "
        "no lifting. Blue = chosen rain plan: music in Harbor Hall, food "
        "trucks on Alder. Green rejected; Red cancels. Permit B-410 stale; "
        "use P-771. Orchid = dessert vendor, missing invoice, circuit C, "
        "nut-free labels by Thursday. Big oak table rejected; use six folding "
        "tables. Friday = insurance upload deadline only; party Saturday 2 "
        "pm, setup 10. Crane = stage rental company delivering risers; driver "
        "calls Robin, Mateo/Jules unload."
    ),
}


SUMMARY_ANCHORS = [
    "Maple",
    "Robin",
    "Blue",
    "Green",
    "Red",
    "B-410",
    "P-771",
    "Orchid",
    "big oak table",
    "Friday",
    "Saturday",
    "Crane",
    "Mateo",
    "Jules",
]


PROBES = [
    {
        "id": "maple_use",
        "text": "What is Maple for again?",
        "anchors": ["Maple"],
    },
    {
        "id": "robin_lifting",
        "text": "Can Robin help unload the risers?",
        "anchors": ["Robin", "risers"],
    },
    {
        "id": "blue_plan",
        "text": "What exactly does Blue mean if it rains?",
        "anchors": ["Blue"],
    },
    {
        "id": "permit_number",
        "text": "Which permit number goes on the insurance form?",
        "anchors": ["permit", "insurance"],
    },
    {
        "id": "orchid_invoice",
        "text": "What do we still need from Orchid?",
        "anchors": ["Orchid"],
    },
    {
        "id": "big_table",
        "text": "Should we use the big table?",
        "anchors": ["big table"],
    },
    {
        "id": "friday_meaning",
        "text": "Is Friday the party day?",
        "anchors": ["Friday", "party"],
    },
    {
        "id": "crane_delivery",
        "text": "What is Crane bringing?",
        "anchors": ["Crane"],
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-filename", default=DEFAULT_QWEN36_LENS)
    p.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
    p.add_argument("--summaries", default="block_party_canonical,block_party_lean")
    p.add_argument("--output", default="outputs/qwen36_plain_probe.json")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--layers", default="quarter")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.add_argument("--skip-probes", action="store_true")
    return p.parse_args()


def matched_summary_messages(summary_text: str, include_old_context: bool) -> list[dict[str, str]]:
    prefix = BLOCK_PARTY_MESSAGES if include_old_context else [{"role": "system", "content": SYSTEM}]
    return [
        *prefix,
        {"role": "user", "content": SUMMARY_REQUEST},
        {"role": "assistant", "content": summary_text},
        {"role": "user", "content": "Continue from this compacted state."},
    ]


def compacted_messages(summary_text: str, probe_text: str) -> list[dict[str, str]]:
    note = (
        "[Context note] Earlier parts of this conversation were compacted. "
        "Summary of what came before:\n\n" + summary_text
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "assistant", "content": note},
        {"role": "user", "content": probe_text},
    ]


def shifted_labels(labels: list[LabeledPosition], shift: int) -> list[LabeledPosition]:
    return [
        LabeledPosition(
            shift + item.position,
            item.label,
            item.phrase,
            item.token_index_in_phrase,
        )
        for item in labels
    ]


def run_summary(
    name: str,
    summary_text: str,
    model: torch.nn.Module,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    layers: list[int],
    top_k: int,
    include_probes: bool,
) -> dict[str, Any]:
    summary_ids = tokenizer(summary_text, add_special_tokens=False).input_ids
    old_ids = render_ids(tokenizer, matched_summary_messages(summary_text, True), False)
    fresh_ids = render_ids(tokenizer, matched_summary_messages(summary_text, False), False)
    old_start = find_subsequence(old_ids, summary_ids)
    fresh_start = find_subsequence(fresh_ids, summary_ids)
    if old_start is None or fresh_start is None:
        raise ValueError(f"could not find summary tokens for {name}")

    relative_labels = labeled_positions_in_text(
        tokenizer, summary_text, SUMMARY_ANCHORS, label_prefix="summary:"
    )
    out: dict[str, Any] = {
        "summary_text": summary_text,
        "token_counts": {
            "old_matched": len(old_ids),
            "fresh_matched": len(fresh_ids),
            "summary_tokens": len(summary_ids),
            "summary_anchor_positions": len(relative_labels),
        },
        "states": {
            "write_time_matched_summary_anchors": capture_labeled(
                lens_model,
                lens,
                tokenizer,
                old_ids,
                layers,
                shifted_labels(relative_labels, old_start),
                top_k,
            ),
            "fresh_matched_summary_anchors": capture_labeled(
                lens_model,
                lens,
                tokenizer,
                fresh_ids,
                layers,
                shifted_labels(relative_labels, fresh_start),
                top_k,
            ),
        },
        "probes": {},
    }

    if include_probes:
        for probe in PROBES:
            full_messages = BLOCK_PARTY_MESSAGES + [{"role": "user", "content": probe["text"]}]
            compact_messages = compacted_messages(summary_text, probe["text"])
            full_text = render_text(tokenizer, full_messages, False)
            compact_text = render_text(tokenizer, compact_messages, False)
            full_ids = render_ids(tokenizer, full_messages, False)
            compact_ids = render_ids(tokenizer, compact_messages, False)
            full_labels = probe_labels_for_rendered_text(
                tokenizer, full_text, probe["text"], probe["anchors"]
            )
            compact_labels = probe_labels_for_rendered_text(
                tokenizer, compact_text, probe["text"], probe["anchors"]
            )
            out["probes"][probe["id"]] = {
                "text": probe["text"],
                "anchors": probe["anchors"],
                "token_counts": {
                    "full_context": len(full_ids),
                    "compacted_context": len(compact_ids),
                },
                "states": {
                    "full_context_probe_tokens": capture_labeled(
                        lens_model,
                        lens,
                        tokenizer,
                        full_ids,
                        layers,
                        full_labels,
                        top_k,
                    ),
                    "compacted_context_probe_tokens": capture_labeled(
                        lens_model,
                        lens,
                        tokenizer,
                        compact_ids,
                        layers,
                        compact_labels,
                        top_k,
                    ),
                },
            }
    return out


def main() -> None:
    args = parse_args()
    requested = [v.strip() for v in args.summaries.split(",") if v.strip()]
    unknown = [v for v in requested if v not in SUMMARIES]
    if unknown:
        raise ValueError(f"unknown summaries: {unknown}; known={sorted(SUMMARIES)}")

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
        "note": "Illustration-only probe. Do not treat as experiment evidence.",
        "summaries": {},
    }
    for name in requested:
        result["summaries"][name] = run_summary(
            name,
            SUMMARIES[name],
            model,
            lens_model,
            lens,
            tokenizer,
            layers,
            args.top_k,
            include_probes=not args.skip_probes,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(out_path)


if __name__ == "__main__":
    main()
