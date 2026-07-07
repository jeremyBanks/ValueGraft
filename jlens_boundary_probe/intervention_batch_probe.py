#!/usr/bin/env python3
"""Batch three-state ValueGraft intervention probes for stronger scenarios.

This script reuses the cache surgery code in boundary_probe.py, but avoids
reloading the 27B model for each qualitative case. Unlike boundary_probe.py's
default smoke test, these cases use preauthored summaries from the broad
write-time/fresh sweep so we can target labels that previously showed large
semantic separation.
"""

from __future__ import annotations

import argparse
import copy
import json
import traceback
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
    alignment_pairs_by_exact_tokens,
    alpha_label,
    blend_values,
    build_b_messages,
    cache_debug_summary,
    capture_forced_sequence_from_cache,
    capture_forced_token_from_cache,
    capture_topk_for_positions,
    choose_layers,
    dtype_from_name,
    find_subsequence,
    graftable_value_layers,
    model_input_device,
    parse_alpha_sweep,
    rebuild_standard_cache,
    render_ids,
    shifted_alignment_pairs,
    snapshot_standard_kv,
)
from multi_demo_scan import DEMOS, Demo


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    demo_name: str
    probe_user: str
    probe_target: str
    focus_phrases: tuple[str, ...]
    rationale: str


CASES: list[ProbeCase] = [
    ProbeCase(
        "block_permit",
        "block_party_canonical",
        "Which permit number goes on the insurance form, and what old number should not be used? Answer in one sentence.",
        "Use P-771 on the insurance form, not B-410.",
        ("P-771", "B-410", "insurance form"),
        "Prior sweep: B-410/P-771 exposes stale-vs-current permit state.",
    ),
    ProbeCase(
        "block_maple",
        "block_party_canonical",
        "What is Maple for in the Riverside plan? Answer in one sentence.",
        "Maple is the library's Maple Room for storage and volunteer check-in.",
        ("Maple", "Maple Room", "storage", "volunteer check-in"),
        "Prior sweep: Maple separates local room meaning from generic place/name priors.",
    ),
    ProbeCase(
        "checkout_falcon",
        "checkout_table",
        "Should we use Falcon or Raven for rollback, and why? Answer in one sentence.",
        "Use Raven for rollback; Falcon was rejected because it drops subscription coupons.",
        ("Raven", "Falcon", "rejected", "subscription coupons"),
        "Prior sweep: Falcon surfaces rejected/obsolete status under write-time context.",
    ),
    ProbeCase(
        "checkout_patch",
        "checkout_table",
        "What hotfix label is live, and what stale patch should not be cited? Answer in one sentence.",
        "R3 is the current live hotfix label, not Patch 17.",
        ("R3", "current live hotfix", "Patch 17"),
        "Prior sweep: Patch 17/R3 separates stale patch label from current hotfix label.",
    ),
    ProbeCase(
        "home_delta",
        "home_checklist",
        "What does Delta mean in the trip plan? Answer in one sentence.",
        "Delta is ferry route Delta 6 at 7:40, not the airline.",
        ("Delta", "Delta 6", "7:40", "airline"),
        "Prior sweep: Delta separates ferry-route meaning from airline priors.",
    ),
    ProbeCase(
        "home_cooler",
        "home_checklist",
        "What cooler should we bring, and what instruction is stale? Answer in one sentence.",
        "Bring two soft coolers; the big cooler instruction is stale.",
        ("two soft coolers", "big cooler", "stale"),
        "Prior sweep: cooler surfaces rejected/replaced status under write-time context.",
    ),
    ProbeCase(
        "pokemon_ghost2",
        "pokemon_canonical",
        "Can Ghost2 be brought forward, and what happened to Ghost? Answer in one sentence.",
        "Ghost2 is alive and made it to Victory Road; Ghost is in the graveyard.",
        ("Ghost2", "alive", "Victory Road", "Ghost", "graveyard"),
        "Prior sweep: Ghost/Ghost2 separates dead-vs-survived Ralts state.",
    ),
    ProbeCase(
        "pokemon_dex",
        "pokemon_canonical",
        "What does Dex owe from the Ruby save? Answer in one sentence.",
        "Dex owes the Makuhita-for-Castform trade from Ruby.",
        ("Dex", "Makuhita", "Castform", "Ruby"),
        "Prior sweep: Dex separates private trade obligation from Pokedex/DexNav priors.",
    ),
    ProbeCase(
        "support_escalation",
        "support_json",
        "Which escalation should be cited, and which one is stale? Answer in one sentence.",
        "Use T-104 as the active escalation, not T-88.",
        ("T-104", "active escalation", "T-88"),
        "Prior sweep: support labels separate active-vs-stale ticket state.",
    ),
    ProbeCase(
        "support_amber",
        "support_json",
        "Which recovery path should be used, and which path is forbidden? Answer in one sentence.",
        "Use the amber read-only recovery path; do not run purple.",
        ("amber", "read-only recovery", "purple"),
        "Prior sweep: amber/purple separates current safe path from forbidden destructive path.",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-filename", default=DEFAULT_QWEN36_LENS)
    p.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
    p.add_argument("--cases", default="all")
    p.add_argument("--output", default="outputs/qwen36_intervention_batch_probe.json")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--layers", default="16,32,48,62")
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--alpha-sweep", default="0,0.25,0.5,0.75,1")
    p.add_argument("--tail-messages", type=int, default=2)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return p.parse_args()


def selected_cases(spec: str) -> list[ProbeCase]:
    if spec == "all":
        return CASES
    names = {x.strip() for x in spec.split(",") if x.strip()}
    out = [case for case in CASES if case.case_id in names]
    missing = names - {case.case_id for case in out}
    if missing:
        raise ValueError(f"unknown cases: {sorted(missing)}")
    return out


def demo_by_name(name: str) -> Demo:
    for demo in DEMOS:
        if demo.name == name:
            return demo
    raise KeyError(name)


def token_tensor_for(model: torch.nn.Module, ids: list[int]) -> torch.Tensor:
    return torch.tensor([ids], device=model_input_device(model))


def find_focus_positions(tokenizer: Any, text: str, text_ids: list[int], phrases: tuple[str, ...]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping")
        ids = encoded.get("input_ids")
        if offsets is not None and list(ids) == text_ids:
            for phrase in phrases:
                positions: list[int] = []
                start = 0
                while True:
                    found = text.find(phrase, start)
                    if found < 0:
                        break
                    lo, hi = found, found + len(phrase)
                    positions.extend(
                        i
                        for i, (a, b) in enumerate(offsets)
                        if int(b) > lo and int(a) < hi
                    )
                    start = found + max(1, len(phrase))
                out[phrase] = sorted(set(positions))
            return out
    except Exception:
        pass

    for phrase in phrases:
        phrase_ids = tokenizer(phrase, add_special_tokens=False).input_ids
        start = find_subsequence(text_ids, phrase_ids)
        out[phrase] = [] if start is None else list(range(start, start + len(phrase_ids)))
    return out


def snapshot_from_ids(model: torch.nn.Module, ids: list[int]) -> tuple[dict[str, Any], Any]:
    with torch.no_grad():
        return snapshot_standard_kv(model, ids)


def clone_snap(snap: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(snap)


def run_case(
    case: ProbeCase,
    demo: Demo,
    model: torch.nn.Module,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    layers: list[int],
    top_k: int,
    alpha: float,
    alpha_sweep: str,
    tail_messages: int,
) -> dict[str, Any]:
    summary_ids = tokenizer(demo.summary, add_special_tokens=False).input_ids
    summary_messages = [
        *demo.messages,
        {"role": "user", "content": SUMMARY_REQUEST},
        {"role": "assistant", "content": demo.summary},
    ]
    old_ids = render_ids(tokenizer, summary_messages, False)
    old_summary_start = find_subsequence(old_ids, summary_ids)
    if old_summary_start is None:
        raise ValueError("could not locate summary in old write-time context")
    old_summary_range = (old_summary_start, old_summary_start + len(summary_ids))

    tail_start_msg = max(1, len(demo.messages) - tail_messages)
    compacted_messages = build_b_messages(demo.messages, demo.summary, tail_start_msg)
    full_probe_messages = [*demo.messages, {"role": "user", "content": case.probe_user}]
    compacted_probe_messages = [*compacted_messages, {"role": "user", "content": case.probe_user}]
    full_probe_ids = render_ids(tokenizer, full_probe_messages, False)
    compacted_ids = render_ids(tokenizer, compacted_messages, False)
    compacted_probe_ids = render_ids(tokenizer, compacted_probe_messages, False)
    compacted_summary_start = find_subsequence(compacted_probe_ids, summary_ids)
    if compacted_summary_start is None:
        raise ValueError("could not locate summary in compacted probe context")
    compacted_summary_range = (
        compacted_summary_start,
        compacted_summary_start + len(summary_ids),
    )

    target_ids = tokenizer(case.probe_target, add_special_tokens=False).input_ids
    focus_positions = find_focus_positions(tokenizer, case.probe_target, target_ids, case.focus_phrases)

    old_snap, _ = snapshot_from_ids(model, old_ids)
    full_probe_snap, _ = snapshot_from_ids(model, full_probe_ids)
    compacted_probe_snap, _ = snapshot_from_ids(model, compacted_probe_ids)
    pairs = alignment_pairs_by_exact_tokens(
        compacted_probe_ids,
        old_ids,
        compacted_summary_range,
        old_summary_range,
    )
    if not pairs:
        raise ValueError("no exact summary-token alignment pairs")

    value_layers = graftable_value_layers(compacted_probe_snap, old_snap)
    alpha_values = parse_alpha_sweep(alpha_sweep, alpha)
    shifted_pairs = shifted_alignment_pairs(pairs)
    alpha0_probe_snap = blend_values(compacted_probe_snap, old_snap, pairs, 0.0)
    grafted_probe_snap = blend_values(compacted_probe_snap, old_snap, pairs, alpha)
    shifted_probe_snap = (
        blend_values(compacted_probe_snap, old_snap, shifted_pairs, alpha)
        if shifted_pairs
        else None
    )

    full_probe_gen_ids = render_ids(tokenizer, full_probe_messages, True)
    compacted_probe_gen_ids = render_ids(tokenizer, compacted_probe_messages, True)
    if full_probe_gen_ids[: len(full_probe_ids)] != full_probe_ids:
        raise ValueError("full generation prompt does not extend full prefix")
    if compacted_probe_gen_ids[: len(compacted_probe_ids)] != compacted_probe_ids:
        raise ValueError("compacted generation prompt does not extend compacted prefix")
    full_suffix = full_probe_gen_ids[len(full_probe_ids) :]
    compacted_suffix = compacted_probe_gen_ids[len(compacted_probe_ids) :]

    full_post = capture_forced_token_from_cache(
        model,
        lens_model,
        lens,
        tokenizer,
        full_probe_snap,
        full_suffix,
        len(full_probe_ids),
        layers,
        top_k,
        label="full_context",
    )
    forced_token_id = int(full_post["token_id"])
    fresh_post = capture_forced_token_from_cache(
        model,
        lens_model,
        lens,
        tokenizer,
        compacted_probe_snap,
        compacted_suffix,
        len(compacted_probe_ids),
        layers,
        top_k,
        forced_token_id=forced_token_id,
        label="fresh_compacted",
    )
    grafted_post = capture_forced_token_from_cache(
        model,
        lens_model,
        lens,
        tokenizer,
        grafted_probe_snap,
        compacted_suffix,
        len(compacted_probe_ids),
        layers,
        top_k,
        forced_token_id=forced_token_id,
        label="grafted_compacted",
    )

    def forced_sequence(label: str, snap: dict[str, Any]) -> dict[str, Any]:
        suffix = full_suffix if label == "full_context" else compacted_suffix
        prefix_len = len(full_probe_ids) if label == "full_context" else len(compacted_probe_ids)
        return capture_forced_sequence_from_cache(
            model,
            lens_model,
            lens,
            tokenizer,
            snap,
            suffix,
            prefix_len,
            target_ids,
            layers,
            top_k,
            label=label,
        )

    full_sequence = forced_sequence("full_context", full_probe_snap)
    fresh_sequence = forced_sequence("fresh_compacted", compacted_probe_snap)
    alpha0_sequence = forced_sequence("alpha0_grafted_compacted", alpha0_probe_snap)
    grafted_sequence = forced_sequence("grafted_compacted", grafted_probe_snap)
    shifted_sequence = (
        forced_sequence("shifted_grafted_compacted", shifted_probe_snap)
        if shifted_probe_snap is not None
        else None
    )

    alpha_sequences: dict[str, Any] = {"0": alpha0_sequence}
    for a in alpha_values:
        if abs(a) < 1e-12 or abs(a - alpha) < 1e-12:
            continue
        alpha_snap = blend_values(compacted_probe_snap, old_snap, pairs, a)
        alpha_sequences[f"{a:g}"] = forced_sequence(alpha_label(a), alpha_snap)

    result: dict[str, Any] = {
        "case_id": case.case_id,
        "demo_name": case.demo_name,
        "rationale": case.rationale,
        "summary_source": "preauthored from prior write-time/fresh sweep",
        "summary_text": demo.summary,
        "probe_user": case.probe_user,
        "probe_target": {
            "text": case.probe_target,
            "token_count": len(target_ids),
            "tokens": [tokenizer.decode([tid]) for tid in target_ids],
            "focus_phrases": list(case.focus_phrases),
            "focus_positions": focus_positions,
        },
        "token_counts": {
            "old_write_time": len(old_ids),
            "summary_tokens": len(summary_ids),
            "fresh_compacted": len(compacted_ids),
            "full_probe": len(full_probe_ids),
            "fresh_probe": len(compacted_probe_ids),
            "tail_start_msg": tail_start_msg,
            "tail_message_count": len(demo.messages) - tail_start_msg,
        },
        "graft": {
            "attempted": True,
            "available": True,
            "policy": "V-only summary-token value-cache blend; fresh keys and linear-attention recurrent state preserved",
            "pairs": len(pairs),
            "alpha": alpha,
            "alpha_sweep": alpha_values,
            "changed_value_layers": value_layers,
            "changed_value_layer_count": len(value_layers),
            "changed_value_slot_count": len(value_layers) * len(pairs),
            "negative_control": {
                "name": "shifted_grafted_compacted",
                "available": shifted_probe_snap is not None,
                "policy": (
                    "same V-only blend and alpha, but old summary-token value "
                    "positions are cyclically shifted by one before injection"
                ),
                "shifted_pairs": len(shifted_pairs),
            },
            "cache_old_summary": cache_debug_summary(old_snap),
            "cache_full_probe": cache_debug_summary(full_probe_snap),
            "cache_fresh_probe": cache_debug_summary(compacted_probe_snap),
            "cache_grafted_probe": cache_debug_summary(grafted_probe_snap),
            "alignment": {
                "old_summary_range": list(old_summary_range),
                "fresh_probe_summary_range": list(compacted_summary_range),
            },
        },
        "states": {
            "old_context_final_token": capture_topk_for_positions(
                lens_model,
                lens,
                tokenizer,
                render_ids(tokenizer, demo.messages, False),
                layers,
                [len(render_ids(tokenizer, demo.messages, False)) - 1],
                top_k,
            ),
            "grafted_post_token": [grafted_post],
        },
        "post_boundary_probe": {
            "design": (
                "Preauthored summary from prior sweep is used as the literal "
                "summary text. Full context keeps the original conversation. "
                "Fresh compacted uses summary+tail text. Grafted compacted "
                "starts from the same fresh compacted text but V-grafts "
                "aligned summary-token value-cache entries from the write-time "
                "summary path."
            ),
            "forced_token_source": "full_context_argmax",
            "forced_token_id": forced_token_id,
            "forced_token": full_post["token"],
            "states": {
                "full_context": full_post,
                "fresh_compacted": fresh_post,
                "grafted_compacted": grafted_post,
            },
            "forced_target_sequences": {
                "full_context": full_sequence,
                "fresh_compacted": fresh_sequence,
                "alpha0_grafted_compacted": alpha0_sequence,
                "grafted_compacted": grafted_sequence,
            },
            "alpha_sweep_sequences": alpha_sequences,
        },
    }
    if shifted_sequence is not None:
        result["post_boundary_probe"]["forced_target_sequences"][
            "shifted_grafted_compacted"
        ] = shifted_sequence
    return result


def main() -> None:
    args = parse_args()
    cases = selected_cases(args.cases)

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
        "alpha": args.alpha,
        "alpha_sweep": parse_alpha_sweep(args.alpha_sweep, args.alpha),
        "tail_messages": args.tail_messages,
        "cases": {},
    }

    for case in cases:
        print(f"RUN {case.case_id}", flush=True)
        try:
            result["cases"][case.case_id] = run_case(
                case,
                demo_by_name(case.demo_name),
                model,
                lens_model,
                lens,
                tokenizer,
                layers,
                args.top_k,
                args.alpha,
                args.alpha_sweep,
                args.tail_messages,
            )
            print(f"DONE {case.case_id}", flush=True)
        except Exception as exc:
            result["cases"][case.case_id] = {
                "case_id": case.case_id,
                "demo_name": case.demo_name,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"ERROR {case.case_id}: {type(exc).__name__}: {exc}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(out_path)


if __name__ == "__main__":
    main()
