#!/usr/bin/env python3
"""J-lens readouts for the canonical Pokemon compaction demo.

This stays isolated from the live experiment code. It samples two kinds of
states:

1. The same summary tokens when written after the full old context vs freshly
   encoded as a compacted-context summary.
2. Ambiguous probe-question tokens under full old context vs compacted summary
   context.
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
    render_text,
)


SYSTEM = (
    "You are a careful assistant. Preserve private names, constraints, "
    "decisions, and stale-vs-current facts exactly."
)

POKEMON_MESSAGES = [
    {"role": "system", "content": SYSTEM},
    {
        "role": "user",
        "content": (
            "Starting a new Pokemon Emerald run tonight. Rules for this one: "
            "no shop-bought healing items in battle, and if a team member "
            "faints it goes in the PC box permanently. I'm calling that the "
            "graveyard rule. Starter is Mudkip, nicknamed Soup."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Noted: Emerald run, no shop-bought battle healing, fainted team "
            "members go permanently to the PC box under the graveyard rule, "
            "and Soup is the Mudkip starter."
        ),
    },
    {
        "role": "user",
        "content": (
            "Caught a Zigzagoon early: nickname Vacuum, because Pickup. He's "
            "a utility slot, not a fighter. Also grabbed a Ralts; nicknaming "
            "her Ghost because she keeps Teleporting away from trainer fights."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Vacuum is the Zigzagoon utility/Pickup slot, not a fighter. "
            "Ghost is the Ralts."
        ),
    },
    {
        "role": "user",
        "content": (
            "Decision time on the early team: I was torn between Breloom and "
            "Hariyama for the fighting slot. Went with Breloom because Spore "
            "is too good with the graveyard rule in play. Hariyama's ruled "
            "out, don't suggest it."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Decision recorded: Breloom over Hariyama for the fighting slot; "
            "Hariyama should stay ruled out."
        ),
    },
    {
        "role": "user",
        "content": (
            "Soup just evolved. Also I misplayed against Wattson and Ghost is "
            "now in the graveyard. Not talking about it. The new Ralts "
            "replacement is nicknamed Ghost2, and if she dies too I'm "
            "dropping psychics entirely."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Sorry about Ghost. Current state: Ghost is in the graveyard, "
            "Ghost2 is the replacement Ralts, and psychics are dropped if "
            "Ghost2 dies."
        ),
    },
    {
        "role": "user",
        "content": (
            "Money situation is rough because of the no-shop-healing rule. "
            "I'm sitting at about 3,200 after buying repels. The plan is to "
            "farm the trainers on Route 118 with Vacuum's Pickup for free "
            "items instead."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "At this point the money figure is about 3,200 after repels, and "
            "the plan is Route 118 farming plus Vacuum's Pickup."
        ),
    },
    {
        "role": "user",
        "content": (
            "For Flannery I'm planning around Soup plus rain from Castform if "
            "I can trade for one. My friend Dex owes me a trade from the Ruby "
            "save. The deal was my spare Makuhita for his Castform, even "
            "though I'm not using Makuhita."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Dex owes a Makuhita-for-Castform trade from Ruby, and the "
            "Castform rain plan supports Soup for Flannery."
        ),
    },
    {
        "role": "user",
        "content": (
            "Long-term team sketch: Soup for waters, Breloom for "
            "fighting/status, Vacuum utility only and never battles gyms, "
            "Ghost2 for psychic, one flyer TBD. I'm leaning Swellow over "
            "Altaria because I want Guts, and Altaria is ruled out anyway "
            "after the Ghost incident soured me on dragons. Last slot open "
            "for a surprise."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Long-term: Soup, Breloom, Vacuum as non-gym utility, Ghost2, "
            "likely Swellow over ruled-out Altaria, and one open slot."
        ),
    },
    {
        "role": "user",
        "content": (
            "At Victory Road prep now. Team is level 43-47. I used the Master "
            "Ball on Rayquaza like an idiot. I keep running out of Ultra "
            "Balls trying to catch a Bagon for the open slot. The graveyard "
            "has four residents now and Ghost2 is NOT one of them: she made "
            "it."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Victory Road prep: team levels 43-47, Master Ball spent on "
            "Rayquaza, Ultra Balls are needed for Bagon, and Ghost2 survived."
        ),
    },
    {
        "role": "user",
        "content": (
            "Elite Four shopping list: max repels, revives are banned by my "
            "rules so extra Hyper Potions for between fights only, and I want "
            "to stock maybe 20 more balls for the post-game legendaries."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "For Elite Four prep: max repels, no revives, extra Hyper Potions "
            "only between fights, and about 20 more Ultra Balls/post-game "
            "balls for legendaries."
        ),
    },
]


SUMMARY_VARIANTS = {
    "canonical": (
        "Pokemon Emerald run state:\n"
        "- Rules: no shop-bought healing items in battle; the graveyard rule "
        "means fainted team members go permanently to the PC box; revives are "
        "banned.\n"
        "- Soup is the evolved Mudkip starter and water core.\n"
        "- Vacuum is the Zigzagoon with Pickup, a utility slot only; Vacuum "
        "never battles gyms or major fights such as Drake.\n"
        "- Original Ralts Ghost died at Wattson and is in the graveyard. "
        "Replacement Ralts Ghost2 is alive and made it to Victory Road.\n"
        "- Fighting slot decision: chose Breloom for Spore; Hariyama, the big "
        "hand guy, is ruled out and should not be suggested.\n"
        "- Dex trade: spare Makuhita for Dex's Castform from the Ruby save; "
        "Castform was for rain support.\n"
        "- Cash: 3,200 was an earlier stale figure after buying repels; do "
        "not treat it as current cash.\n"
        "- Victory Road/Elite Four: team level 43-47, Master Ball spent on "
        "Rayquaza, needs Bagon for the open slot, stock about 20 more Ultra "
        "Balls for post-game legendaries, max repels, and Hyper Potions only "
        "between fights."
    ),
    "lean": (
        "Pokemon Emerald notes: Soup is the evolved Mudkip. Graveyard rule "
        "still active; no shop healing in battle and no revives. Vacuum is "
        "Pickup/utility only, not for gyms or Drake. Ghost died; Ghost2 made "
        "it. Breloom stayed; Hariyama/big hand guy is out. Dex owes "
        "Makuhita-for-Castform. 3,200 cash is stale. At Victory Road and "
        "Elite Four prep: levels 43-47, Master Ball spent on Rayquaza, catch "
        "Bagon, buy about 20 Ultra Balls, max repels, Hyper Potions between "
        "fights only."
    ),
}


SUMMARY_ANCHORS = [
    "Soup",
    "graveyard",
    "Vacuum",
    "Pickup",
    "Drake",
    "Ghost",
    "Ghost2",
    "Breloom",
    "Hariyama",
    "big hand guy",
    "Dex",
    "Makuhita",
    "Castform",
    "3,200",
    "Ultra Balls",
    "Bagon",
    "Rayquaza",
    "Hyper Potions",
]


PROBES = [
    {
        "id": "ghost_elite_four",
        "text": "Can I bring Ghost to the Elite Four?",
        "anchors": ["Ghost", "Elite Four"],
    },
    {
        "id": "big_hand_guy",
        "text": "Should I reconsider the big hand guy?",
        "anchors": ["big hand guy", "hand"],
    },
    {
        "id": "dex_trade",
        "text": "What did I promise Dex?",
        "anchors": ["Dex"],
    },
    {
        "id": "utility_drake",
        "text": "Who's my utility guy and can he fight Drake?",
        "anchors": ["utility", "Drake"],
    },
    {
        "id": "balls_high_levels",
        "text": "What type of ball should I use at high levels?",
        "anchors": ["ball", "levels"],
    },
    {
        "id": "cash_staleness",
        "text": "How much cash do I have to work with?",
        "anchors": ["cash"],
    },
]


@dataclass(frozen=True)
class LabeledPosition:
    position: int
    label: str
    phrase: str
    token_index_in_phrase: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-filename", default=DEFAULT_QWEN36_LENS)
    p.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
    p.add_argument("--variants", default="canonical,lean")
    p.add_argument("--output", default="outputs/qwen36_pokemon_probe.json")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--layers", default="quarter")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.add_argument("--skip-probes", action="store_true")
    return p.parse_args()


def compacted_messages(summary_text: str, probe_text: str | None = None) -> list[dict[str, str]]:
    note = (
        "[Context note] Earlier parts of this conversation were compacted. "
        "Summary of what came before:\n\n" + summary_text
    )
    out = [
        {"role": "system", "content": SYSTEM},
        {"role": "assistant", "content": note},
    ]
    out.append({"role": "user", "content": probe_text or "Continue from this compacted state."})
    return out


def matched_summary_messages(
    summary_text: str, include_old_context: bool
) -> list[dict[str, str]]:
    """Place summary text in an identical non-final assistant-summary wrapper.

    This avoids comparing "assistant generation after the full old context" to
    "assistant context note in a fresh transcript" when we want the cleaner
    contrast: same local wrapper, old context present vs absent.
    """
    prefix = POKEMON_MESSAGES if include_old_context else [{"role": "system", "content": SYSTEM}]
    return [
        *prefix,
        {"role": "user", "content": SUMMARY_REQUEST},
        {"role": "assistant", "content": summary_text},
        {"role": "user", "content": "Continue from this compacted state."},
    ]


def summary_write_ids(tokenizer: Any, summary_text: str) -> tuple[list[int], int, list[int]]:
    req_messages = POKEMON_MESSAGES + [{"role": "user", "content": SUMMARY_REQUEST}]
    req_ids = render_ids(tokenizer, req_messages, True)
    summary_ids = tokenizer(summary_text, add_special_tokens=False).input_ids
    return req_ids + summary_ids, len(req_ids), summary_ids


def token_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(int(a), int(b)) for a, b in encoded.offset_mapping]
    return list(encoded.input_ids), offsets


def phrase_spans(text: str, phrase: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    end = len(text) if end is None else end
    spans = []
    pos = start
    while pos < end:
        hit = text.find(phrase, pos, end)
        if hit < 0:
            break
        span_end = hit + len(phrase)
        before = text[hit - 1] if hit > 0 else ""
        after = text[span_end] if span_end < len(text) else ""
        if not (before.isalnum() or before == "_" or after.isalnum() or after == "_"):
            spans.append((hit, span_end))
        pos = hit + max(1, len(phrase))
    return spans


def labeled_positions_in_text(
    tokenizer: Any,
    text: str,
    phrases: list[str],
    base_position: int = 0,
    label_prefix: str = "",
    char_start: int = 0,
    char_end: int | None = None,
) -> list[LabeledPosition]:
    _, offsets = token_offsets(tokenizer, text)
    found: list[LabeledPosition] = []
    for phrase in phrases:
        for span_start, span_end in phrase_spans(text, phrase, char_start, char_end):
            overlapping = [
                i
                for i, (tok_start, tok_end) in enumerate(offsets)
                if tok_start < span_end and tok_end > span_start
            ]
            for phrase_i, tok_i in enumerate(overlapping):
                found.append(
                    LabeledPosition(
                        position=base_position + tok_i,
                        label=f"{label_prefix}{phrase}",
                        phrase=phrase,
                        token_index_in_phrase=phrase_i,
                    )
                )
    return dedupe_labeled(found)


def dedupe_labeled(items: list[LabeledPosition]) -> list[LabeledPosition]:
    seen = set()
    out = []
    for item in items:
        key = (item.position, item.label, item.token_index_in_phrase)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def attach_labels(rows: list[dict[str, Any]], labels: list[LabeledPosition]) -> list[dict[str, Any]]:
    by_pos: dict[int, list[dict[str, Any]]] = {}
    for item in labels:
        by_pos.setdefault(item.position, []).append(
            {
                "label": item.label,
                "phrase": item.phrase,
                "token_index_in_phrase": item.token_index_in_phrase,
            }
        )
    for row in rows:
        row["targets"] = by_pos.get(int(row["position"]), [])
    return rows


def capture_labeled(
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    input_ids: list[int],
    layers: list[int],
    labels: list[LabeledPosition],
    top_k: int,
) -> list[dict[str, Any]]:
    positions = sorted({item.position for item in labels})
    rows = capture_topk_for_positions(
        lens_model, lens, tokenizer, input_ids, layers, positions, top_k
    )
    return attach_labels(rows, labels)


def probe_labels_for_rendered_text(
    tokenizer: Any,
    rendered_text: str,
    probe_text: str,
    anchors: list[str],
) -> list[LabeledPosition]:
    probe_start = rendered_text.rfind(probe_text)
    if probe_start < 0:
        raise ValueError(f"could not find probe text in rendered chat: {probe_text!r}")
    return labeled_positions_in_text(
        tokenizer,
        rendered_text,
        anchors,
        char_start=probe_start,
        char_end=probe_start + len(probe_text),
    )


def run_variant(
    variant: str,
    summary_text: str,
    model: torch.nn.Module,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    layers: list[int],
    top_k: int,
    include_probes: bool,
) -> dict[str, Any]:
    old_ids, old_summary_start, summary_ids = summary_write_ids(tokenizer, summary_text)
    b_messages = compacted_messages(summary_text)
    b_ids = render_ids(tokenizer, b_messages, False)
    b_summary_start = find_subsequence(b_ids, summary_ids)
    if b_summary_start is None:
        raise ValueError(f"could not locate {variant} summary tokens in compacted context")

    old_matched_ids = render_ids(
        tokenizer, matched_summary_messages(summary_text, include_old_context=True), False
    )
    fresh_matched_ids = render_ids(
        tokenizer, matched_summary_messages(summary_text, include_old_context=False), False
    )
    old_matched_summary_start = find_subsequence(old_matched_ids, summary_ids)
    fresh_matched_summary_start = find_subsequence(fresh_matched_ids, summary_ids)
    if old_matched_summary_start is None or fresh_matched_summary_start is None:
        raise ValueError(f"could not locate {variant} summary tokens in matched wrappers")

    relative_summary_labels = labeled_positions_in_text(
        tokenizer, summary_text, SUMMARY_ANCHORS, label_prefix="summary:"
    )
    old_summary_labels = [
        LabeledPosition(
            old_summary_start + item.position,
            item.label,
            item.phrase,
            item.token_index_in_phrase,
        )
        for item in relative_summary_labels
    ]
    fresh_summary_labels = [
        LabeledPosition(
            b_summary_start + item.position,
            item.label,
            item.phrase,
            item.token_index_in_phrase,
        )
        for item in relative_summary_labels
    ]
    old_matched_summary_labels = [
        LabeledPosition(
            old_matched_summary_start + item.position,
            item.label,
            item.phrase,
            item.token_index_in_phrase,
        )
        for item in relative_summary_labels
    ]
    fresh_matched_summary_labels = [
        LabeledPosition(
            fresh_matched_summary_start + item.position,
            item.label,
            item.phrase,
            item.token_index_in_phrase,
        )
        for item in relative_summary_labels
    ]

    out: dict[str, Any] = {
        "summary_text": summary_text,
        "token_counts": {
            "old_with_summary": len(old_ids),
            "fresh_compacted": len(b_ids),
            "old_matched": len(old_matched_ids),
            "fresh_matched": len(fresh_matched_ids),
            "summary_tokens": len(summary_ids),
            "summary_anchor_positions": len(relative_summary_labels),
        },
        "states": {
            "write_time_summary_anchors": capture_labeled(
                lens_model,
                lens,
                tokenizer,
                old_ids,
                layers,
                old_summary_labels,
                top_k,
            ),
            "fresh_compacted_summary_anchors": capture_labeled(
                lens_model,
                lens,
                tokenizer,
                b_ids,
                layers,
                fresh_summary_labels,
                top_k,
            ),
            "write_time_matched_summary_anchors": capture_labeled(
                lens_model,
                lens,
                tokenizer,
                old_matched_ids,
                layers,
                old_matched_summary_labels,
                top_k,
            ),
            "fresh_matched_summary_anchors": capture_labeled(
                lens_model,
                lens,
                tokenizer,
                fresh_matched_ids,
                layers,
                fresh_matched_summary_labels,
                top_k,
            ),
        },
        "probes": {},
    }

    if include_probes:
        for probe in PROBES:
            full_messages = POKEMON_MESSAGES + [{"role": "user", "content": probe["text"]}]
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
    requested = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in requested if v not in SUMMARY_VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; known={sorted(SUMMARY_VARIANTS)}")

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
        "variants": {},
    }
    for variant in requested:
        result["variants"][variant] = run_variant(
            variant,
            SUMMARY_VARIANTS[variant],
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
