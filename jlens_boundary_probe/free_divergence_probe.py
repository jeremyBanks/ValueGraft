#!/usr/bin/env python3
"""FREE-GENERATION divergence-lens probe for ValueGraft on Qwen3.6-27B.

Why this differs from every prior lens probe here
-------------------------------------------------
Every prior lens probe (``boundary_probe``, ``intervention_batch_probe``,
``strong_example_probe``) *teacher-forced the SAME gold continuation* under all
three cache states (full / fresh-compacted / aligned-graft). Forcing an
identical trajectory PINS the states together and SUPPRESSES divergence, so the
J-lens readouts came out subtle. This probe removes the pin: it lets B
(fresh-compacted) and E (aligned-graft) FREELY generate, finds the FORK where a
subtle internal difference blooms into a behavioral divergence, and puts the
J-lens THERE.

Core method (kept)
------------------
For every usable plant we build the three cache states with the *existing*
machinery (``generate_summary`` + ``snapshot_standard_kv`` +
``alignment_pairs_by_exact_tokens`` + ``blend_values`` from ``boundary_probe``,
the same compaction/graft path as ``gap_closure_27b``):
    A = full-context, B = fresh-compacted, E = aligned-graft (alpha_V=0.75).
Then for each plant probe we FREELY greedy-generate B and E separately, find the
first argmax fork, and read the J-lens across layers under B / E / A at the fork
window.

FABLE GUT-CHECK FIXES (07-07) folded in -- this is the honest version
---------------------------------------------------------------------
1. DISTRIBUTION, NOT 3 HEROES: the case set is now EVERY usable ``sense`` and
   ``referent`` plant in ``data/synthetic/c*.json`` (contaminated plants
   skipped), ~43 cases, so the headline is a DISTRIBUTION and any vivid case is
   one point on it -- not a standalone anecdote.
2. FORK MARGIN: at the fork position we record the MARGIN of the argmax (top-1
   minus top-2 logit) for both E and B. A near-tie fork is a decoder-amplified
   coin-flip, not a robust mechanistic pull; the margin lets the analysis flag
   it.
3. DECODING ROBUSTNESS: B and E are also generated at two extra temperatures
   (0.3, 0.7) with a fixed seed. We record whether E's lean toward the RIGHT
   concept is STABLE across the greedy + 2 sampled runs (e.g. "3/3 runs
   E->right"). A robust exhibit's lean survives decoding variation.
4. THREE-BUCKET CLASSIFICATION (kills heads-I-win): each case falls in exactly
   one bucket --
     (a) FORK_TOWARD_A : E forks to the right concept, decisive margin, stable.
     (b) SUBTLE_LEAN   : small but CONSISTENT lean toward right (weak margin but
                          stable direction).
     (c) DISCONFIRMING : E does NOT fork from B at all, OR forks toward the
                          WRONG concept, OR its lean does not survive decoding.
   Bucket (c) counts AGAINST the claim and is reported as such -- never
   relabeled "subtle." The three-bucket COUNTS are the headline.

Right/wrong concept vocabulary is DERIVED from the plant data (never authored to
steer the model): ``right_terms`` come from the plant's disambiguating
``keywords``; ``wrong_terms`` come from the "... not the X" alternative in the
gold plus any ``anti_keywords``. Referent plants have no clean "wrong" concept
vocabulary, so for them "fork toward wrong" is undetectable and DISCONFIRMING is
reached only via no-fork / no-right-lean -- noted honestly in the output.

Self-test locally with ``--dry-run`` (no torch / model / lens / tokenizer): it
builds every case, prints the count, a couple of sample specs, and the estimated
forward-pass count.
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Any

# ``gap_closure_27b`` is import-safe with NO ML deps (it defers torch past its
# own dry-run gate), so we reuse its corpus loader + plant-spec construction
# directly -- the SAME construction the behavioral gap-closure numbers use.
from gap_closure_27b import (
    DEFAULT_DATA_DIR,
    PlantSpec,
    conversation_paths,
    plant_specs_for,
)

# Only sense + referent plants: both are "the label/subject survives compaction
# but its specific referent is lost" cases where a graft could restore the role.
USABLE_CATEGORIES = ("sense", "referent")

STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "not", "but", "into", "being",
    "with", "on", "in", "to", "by", "that", "this", "was", "were", "our", "their",
    "his", "her", "its", "them", "then", "than", "from", "over", "each", "only",
    "both", "one", "two", "three", "set", "new", "big", "small", "used", "use",
    "when", "where", "which", "what", "work", "project", "thing", "idea",
}


# ---------------------------------------------------------------------------
# Right/wrong concept vocabulary, DERIVED from plant data (not authored to
# steer the model -- it drives only the read-out heuristic and classification).
# ---------------------------------------------------------------------------
def _content_words(text: str, min_len: int = 4) -> list[str]:
    words = re.findall(r"[a-z][a-z\-]+", text.lower())
    return [w for w in words if len(w) >= min_len and w not in STOPWORDS]


def _derive_label(plant: dict[str, Any]) -> str | None:
    """The overloaded label token, pulled from the first quoted phrase in the
    plant's establishing user turn (sense plants say e.g. "'Nimbus' is ...")."""
    mu = str(plant.get("middle_user", ""))
    # A true quoted label: opening quote at a word boundary, short content with
    # no sentence punctuation, closing quote followed by space/punct/end. This
    # avoids matching in-word apostrophes ("don't", "we'll").
    m = re.search(
        r"(?:(?<=\s)|^)['‘“\"]([^'’”\".]{1,40}?)['’”\"](?=[\s,.:;)]|$)",
        mu,
    )
    if m:
        return m.group(1).strip()
    return None


def derive_terms(plant: dict[str, Any]) -> dict[str, Any]:
    keywords = [str(k).lower().strip() for k in plant.get("keywords", []) if str(k).strip()]
    anti = [str(k).lower().strip() for k in plant.get("anti_keywords", []) if str(k).strip()]
    gold = str(plant.get("gold", ""))
    gold_l = gold.lower()

    # RIGHT: the disambiguating role keywords curated into the data.
    right = list(dict.fromkeys(keywords))
    if not right:  # fallback: content words from the head of the gold role
        head = gold_l.split(" not ")[0] if " not " in gold_l else gold_l
        right = list(dict.fromkeys(_content_words(head)))[:6]

    # WRONG: the "... not the X" alternative in the gold + anti_keywords.
    wrong: list[str] = list(anti)
    m = re.search(r"\bnot\b\s+(?:the\s+|a\s+|an\s+)?(.+)$", gold_l)
    if m:
        tail = re.split(r"[,.;:]", m.group(1).strip())[0].strip().rstrip(". ")
        if tail:
            wrong.append(tail)
            wrong += _content_words(tail)
    wrong = list(dict.fromkeys([w for w in wrong if w]))

    return {
        "right_terms": right,
        "wrong_terms": wrong,
        "wrong_concept_detectable": bool(wrong),
        "label": _derive_label(plant),
    }


def _contains_any(text: str, terms: list[str]) -> list[str]:
    low = text.lower()
    return [t for t in terms if t and t in low]


def classify_text(text: str, terms: dict[str, Any]) -> dict[str, Any]:
    right_hits = _contains_any(text, terms["right_terms"])
    wrong_hits = _contains_any(text, terms["wrong_terms"])
    r, w = len(right_hits), len(wrong_hits)
    if r > w and r > 0:
        verdict = "RIGHT"
    elif w > r and w > 0:
        verdict = "WRONG"
    elif r > 0 and w > 0:
        verdict = "MIXED"
    else:
        verdict = "NEITHER"
    return {
        "verdict": verdict,
        "right_hits": right_hits,
        "wrong_hits": wrong_hits,
        "leans_right": (r > w and r > 0),
        "leans_wrong": (w > r and w > 0),
    }


# ===========================================================================
# Everything below needs torch. Kept out of module scope so --dry-run runs on a
# CPU-only box with no ML deps.
# ===========================================================================
def _lens_layers_readout(lens_model, lens, tokenizer, activations, layers, top_k):
    """J-lens top-k per layer from recorded activations (last fed position)."""
    from boundary_probe import topk_from_logits

    out: dict[str, Any] = {}
    for layer in layers:
        residual = activations[layer][0, -1].float().unsqueeze(0)
        logits_l = lens_model.unembed(lens.transport(residual, layer))[0]
        out[str(layer)] = topk_from_logits(tokenizer, logits_l, top_k)
    return out


def free_generate(
    model,
    tokenizer,
    snap,
    suffix_ids,
    prefix_len,
    n_new,
    top_k,
    eos_ids,
    label,
    temperature=0.0,
    seed=0,
):
    """Free generation (NOT teacher-forced) from a cache snapshot.

    ``temperature==0`` -> greedy argmax; ``temperature>0`` -> temperature
    sampling with a fixed ``seed`` (for the decoding-robustness check). Returns
    generated token ids, decoded text, and a per-step trace that includes the
    argmax MARGIN (top-1 minus top-2 logit) at every step.
    """
    import torch

    from boundary_probe import (
        model_input_device,
        rebuild_standard_cache,
        token_tensor,
        topk_from_logits,
    )

    if temperature and temperature > 0:
        torch.manual_seed(seed)

    cache = rebuild_standard_cache(snap, model)
    dev = model_input_device(model)
    next_position = prefix_len

    if not suffix_ids:
        raise ValueError("free generation requires a non-empty generation suffix")
    pos = torch.arange(next_position, next_position + len(suffix_ids), device=dev)[None]
    with torch.no_grad():
        out = model(
            input_ids=token_tensor(model, suffix_ids),
            past_key_values=cache,
            position_ids=pos,
            use_cache=True,
            logits_to_keep=1,
        )
    logits = out.logits[:, -1, :]
    next_position += len(suffix_ids)

    def pick(lg):
        top = topk_from_logits(tokenizer, lg[0], max(top_k, 2))
        margin = (top[0]["score"] - top[1]["score"]) if len(top) >= 2 else None
        if temperature and temperature > 0:
            probs = torch.softmax(lg[0].float() / temperature, dim=-1)
            tok = int(torch.multinomial(probs, 1).item())
        else:
            tok = int(top[0]["token_id"])
        return tok, top, margin

    gen_ids: list[int] = []
    rows: list[dict[str, Any]] = []
    for step in range(n_new):
        tok_id, top, margin = pick(logits)
        rows.append(
            {
                "step": step,
                "token_id": tok_id,
                "token": tokenizer.decode([tok_id]),
                "margin": margin,
                "next_token_top": top[:top_k],
            }
        )
        gen_ids.append(tok_id)
        if tok_id in eos_ids:
            break
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([[tok_id]], device=dev),
                past_key_values=cache,
                position_ids=torch.tensor([[next_position]], device=dev),
                use_cache=True,
                logits_to_keep=1,
            )
        logits = out.logits[:, -1, :]
        next_position += 1

    text_ids = [t for t in gen_ids if t not in eos_ids]
    return {
        "label": label,
        "temperature": temperature,
        "gen_token_ids": gen_ids,
        "gen_token_count": len(gen_ids),
        "text": tokenizer.decode(text_ids).strip(),
        "steps": rows,
    }


def readout_at_steps(
    model,
    lens_model,
    lens,
    tokenizer,
    snap,
    suffix_ids,
    prefix_len,
    cont_ids,
    steps,
    layers,
    top_k,
    label,
):
    """Teacher-force ``cont_ids`` through a snapshot; J-lens read-out at ``steps``.

    ``step`` is a generation index: after feeding ``cont_ids[:step]`` the residual
    at the last fed position PREDICTS the token at generation position ``step``.
    Used uniformly for A / B / E at the fork window so the ONLY difference is the
    cache state, not the tokens.
    """
    import torch

    from boundary_probe import (
        model_input_device,
        rebuild_standard_cache,
        token_tensor,
        topk_from_logits,
    )
    from jlens.hooks import ActivationRecorder

    want = sorted({s for s in steps if s >= 0})
    if not want:
        return {"label": label, "steps": {}}
    max_step = max(want)

    cache = rebuild_standard_cache(snap, model)
    dev = model_input_device(model)
    next_position = prefix_len
    final_layer = lens_model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})

    if not suffix_ids:
        raise ValueError("readout requires a non-empty generation suffix")
    pos = torch.arange(next_position, next_position + len(suffix_ids), device=dev)[None]
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=record_at) as rec:
        out = model(
            input_ids=token_tensor(model, suffix_ids),
            past_key_values=cache,
            position_ids=pos,
            use_cache=True,
            logits_to_keep=1,
        )
        activations = {i: rec.activations[i].detach() for i in record_at}
    logits = out.logits[:, -1, :]
    next_position += len(suffix_ids)

    captured: dict[str, Any] = {}

    def maybe_capture(step_idx: int):
        if step_idx in want:
            argmax_id = int(torch.argmax(logits, dim=-1).item())
            trajectory_token = (
                int(cont_ids[step_idx]) if step_idx < len(cont_ids) else None
            )
            captured[str(step_idx)] = {
                "step": step_idx,
                "argmax_token_id": argmax_id,
                "argmax_token": tokenizer.decode([argmax_id]),
                "trajectory_token_id": trajectory_token,
                "trajectory_token": (
                    tokenizer.decode([trajectory_token])
                    if trajectory_token is not None
                    else None
                ),
                "next_token_top": topk_from_logits(tokenizer, logits[0], top_k),
                "layers": _lens_layers_readout(
                    lens_model, lens, tokenizer, activations, layers, top_k
                ),
            }

    maybe_capture(0)
    for i in range(max_step):
        if i >= len(cont_ids):
            break
        tok_id = int(cont_ids[i])
        with torch.no_grad(), ActivationRecorder(lens_model.layers, at=record_at) as rec:
            out = model(
                input_ids=torch.tensor([[tok_id]], device=dev),
                past_key_values=cache,
                position_ids=torch.tensor([[next_position]], device=dev),
                use_cache=True,
                logits_to_keep=1,
            )
            activations = {i2: rec.activations[i2].detach() for i2 in record_at}
        logits = out.logits[:, -1, :]
        next_position += 1
        maybe_capture(i + 1)

    return {"label": label, "steps": captured}


def _fork_lean(readout_step: dict[str, Any] | None, terms: dict[str, Any]) -> dict[str, Any]:
    """Aggregate a single state's lens read-out at one step into right/wrong lean."""
    right_terms, wrong_terms = terms["right_terms"], terms["wrong_terms"]
    right_hits: list[str] = []
    wrong_hits: list[str] = []
    if readout_step is not None:
        for _layer, rows in readout_step.get("layers", {}).items():
            for item in rows:
                tok = str(item.get("token", ""))
                right_hits += _contains_any(tok, right_terms)
                wrong_hits += _contains_any(tok, wrong_terms)
    r, w = len(right_hits), len(wrong_hits)
    if r > w:
        lean = "RIGHT"
    elif w > r:
        lean = "WRONG"
    elif r == w == 0:
        lean = "NEITHER"
    else:
        lean = "MIXED"
    return {
        "lean": lean,
        "right_lens_hits": sorted(set(right_hits)),
        "wrong_lens_hits": sorted(set(wrong_hits)),
        "right_count": r,
        "wrong_count": w,
        "lean_toward_A_magnitude": r - w,
    }


def find_fork(b_ids: list[int], e_ids: list[int]) -> int | None:
    n = min(len(b_ids), len(e_ids))
    for i in range(n):
        if b_ids[i] != e_ids[i]:
            return i
    if len(b_ids) != len(e_ids):
        return n  # one ran longer; the length difference is itself a fork
    return None


def build_conv_states(conv, model, tokenizer, alpha, max_new_summary_tokens):
    """Compact one conversation ONCE (model-generated summary) and build the
    A/B/E cache snapshots shared by all of that conversation's plants. Same path
    as ``gap_closure_27b.main``. Requires torch."""
    from boundary_probe import (
        alignment_pairs_by_exact_tokens,
        blend_values,
        build_b_messages,
        find_subsequence,
        generate_summary,
        graftable_value_layers,
        render_ids,
        snapshot_standard_kv,
    )

    msgs = conv["messages"][:-1]
    tsm = conv["sections"]["middle_end_msg"]

    summary = generate_summary(model, tokenizer, msgs, max_new_summary_tokens)
    summary_text = summary["text"]
    b_msgs = build_b_messages(msgs, summary_text, tsm)

    full_ids = render_ids(tokenizer, msgs, False)
    b_ids = render_ids(tokenizer, b_msgs, False)
    summary_text_ids = tokenizer(summary_text, add_special_tokens=False).input_ids
    b_summary_start = find_subsequence(b_ids, summary_text_ids)
    if b_summary_start is None:
        raise ValueError("could not locate summary text in fresh-compacted ids")
    b_summary_range = (b_summary_start, b_summary_start + len(summary_text_ids))

    pairs = alignment_pairs_by_exact_tokens(
        b_ids,
        summary["old_ids"],
        b_summary_range,
        (summary["s_start"], summary["s_end"]),
    )
    if not pairs:
        raise ValueError("no exact summary-token alignment pairs")

    a_snap, _ = snapshot_standard_kv(model, full_ids)
    b_snap, _ = snapshot_standard_kv(model, b_ids)
    old_snap, _ = snapshot_standard_kv(model, summary["old_ids"])
    e_snap = blend_values(b_snap, old_snap, pairs, alpha)

    return {
        "msgs": msgs,
        "b_msgs": b_msgs,
        "A_snap": a_snap,
        "B_snap": b_snap,
        "E_snap": e_snap,
        "full_ids": full_ids,
        "b_ids": b_ids,
        "summary_text": summary_text,
        "pairs": len(pairs),
        "graft_value_layers": graftable_value_layers(b_snap, old_snap),
        "token_counts": {
            "full": len(full_ids),
            "fresh_compacted": len(b_ids),
            "old_with_summary": len(summary["old_ids"]),
            "summary_tokens": len(summary_text_ids),
        },
    }


def classify_bucket(
    fork_found, e_right, e_wrong, margin_E, decisive_threshold, e_right_count, n_runs
) -> dict[str, Any]:
    """Pre-registered THREE-bucket classification (kills heads-I-win)."""
    stable_majority = e_right_count >= (n_runs // 2 + 1)
    stable_all = e_right_count == n_runs
    decisive = margin_E is not None and margin_E >= decisive_threshold

    if not fork_found:
        bucket, why = "DISCONFIRMING", "no fork: E did not diverge from B under free-gen"
    elif e_wrong:
        bucket, why = "DISCONFIRMING", "E forks toward the WRONG concept"
    elif not e_right:
        bucket, why = "DISCONFIRMING", "E does not move toward the RIGHT concept"
    elif not stable_majority:
        bucket, why = "DISCONFIRMING", "E's right-lean does not survive decoding variation"
    elif decisive and stable_all:
        bucket, why = "FORK_TOWARD_A", "clean fork toward right concept, decisive margin, stable"
    else:
        bucket, why = "SUBTLE_LEAN", "small but consistent lean toward right concept"

    return {
        "bucket": bucket,
        "reason": why,
        "decisive_margin": decisive,
        "stable_majority": stable_majority,
        "stable_all": stable_all,
    }


def run_plant(
    spec: PlantSpec,
    plant: dict[str, Any],
    st,
    model,
    lens_model,
    lens,
    tokenizer,
    layers,
    top_k,
    n_new,
    temperatures,
    seed,
    decisive_threshold,
    eos_ids,
) -> dict[str, Any]:
    from boundary_probe import render_ids

    terms = derive_terms(plant)

    def suffix(mm, probe):
        gen = render_ids(tokenizer, mm + [{"role": "user", "content": probe}], True)
        cn = render_ids(tokenizer, mm, False)
        if gen[: len(cn)] != cn:
            raise ValueError("probe generation prompt does not extend prefix")
        return gen[len(cn):]

    sa = suffix(st["msgs"], spec.probe)      # A (full) side
    sb = suffix(st["b_msgs"], spec.probe)    # B/E (compacted) side
    a_prefix, b_prefix = st["token_counts"]["full"], st["token_counts"]["fresh_compacted"]

    # --- greedy free-generation (the exhibit trajectory) ---
    b_free = free_generate(model, tokenizer, st["B_snap"], sb, b_prefix,
                           n_new, top_k, eos_ids, "fresh_compacted")
    e_free = free_generate(model, tokenizer, st["E_snap"], sb, b_prefix,
                           n_new, top_k, eos_ids, "aligned_graft")
    a_free = free_generate(model, tokenizer, st["A_snap"], sa, a_prefix,
                           n_new, top_k, eos_ids, "full_context")

    fork = find_fork(b_free["gen_token_ids"], e_free["gen_token_ids"])

    # --- fork MARGIN (decoder-amplification guard) ---
    def margin_at(free, idx):
        if idx is None or idx >= len(free["steps"]):
            return None
        return free["steps"][idx]["margin"]

    margin_E = margin_at(e_free, fork)
    margin_B = margin_at(b_free, fork)

    # --- decoding-robustness: greedy + sampled temperatures, is E's lean stable? ---
    b_greedy_v = classify_text(b_free["text"], terms)
    e_greedy_v = classify_text(e_free["text"], terms)
    runs = [{
        "temperature": 0.0,
        "e_text": e_free["text"],
        "e_leans_right": e_greedy_v["leans_right"],
        "b_leans_wrong": b_greedy_v["leans_wrong"],
    }]
    for temp in temperatures:
        b_s = free_generate(model, tokenizer, st["B_snap"], sb, b_prefix,
                            n_new, top_k, eos_ids, "fresh_compacted", temperature=temp, seed=seed)
        e_s = free_generate(model, tokenizer, st["E_snap"], sb, b_prefix,
                            n_new, top_k, eos_ids, "aligned_graft", temperature=temp, seed=seed)
        b_v = classify_text(b_s["text"], terms)
        e_v = classify_text(e_s["text"], terms)
        runs.append({
            "temperature": temp,
            "e_text": e_s["text"],
            "e_leans_right": e_v["leans_right"],
            "b_leans_wrong": b_v["leans_wrong"],
        })
    n_runs = len(runs)
    e_right_count = sum(1 for r in runs if r["e_leans_right"])
    decoding_stability = {
        "n_runs": n_runs,
        "e_right_count": e_right_count,
        "summary": f"{e_right_count}/{n_runs} runs E->right",
        "runs": runs,
    }

    # --- lens read-out at the fork window (the interpretability core) ---
    max_common = min(len(b_free["gen_token_ids"]), len(e_free["gen_token_ids"]))
    center = fork if fork is not None else max(0, max_common - 1)
    window = sorted({s for s in (center - 1, center, center + 1) if 0 <= s < max_common}) or [0]

    b_read = readout_at_steps(model, lens_model, lens, tokenizer, st["B_snap"], sb, b_prefix,
                              b_free["gen_token_ids"], window, layers, top_k, "fresh_compacted")
    e_read = readout_at_steps(model, lens_model, lens, tokenizer, st["E_snap"], sb, b_prefix,
                              e_free["gen_token_ids"], window, layers, top_k, "aligned_graft")
    a_read = readout_at_steps(model, lens_model, lens, tokenizer, st["A_snap"], sa, a_prefix,
                              b_free["gen_token_ids"], window, layers, top_k, "full_context")

    fork_key = str(center)
    fork_lean = {
        "A_full": _fork_lean(a_read["steps"].get(fork_key), terms),
        "B_fresh": _fork_lean(b_read["steps"].get(fork_key), terms),
        "E_graft": _fork_lean(e_read["steps"].get(fork_key), terms),
    }

    # --- THREE-BUCKET classification ---
    cls = classify_bucket(
        fork_found=fork is not None,
        e_right=e_greedy_v["leans_right"],
        e_wrong=e_greedy_v["leans_wrong"],
        margin_E=margin_E,
        decisive_threshold=decisive_threshold,
        e_right_count=e_right_count,
        n_runs=n_runs,
    )

    return {
        "plant_id": spec.plant_id,
        "conversation_id": spec.conv_id,
        "category": spec.category,
        "label": terms["label"],
        "gold": spec.gold,
        "probe_user": spec.probe,
        "right_terms": terms["right_terms"],
        "wrong_terms": terms["wrong_terms"],
        "wrong_concept_detectable": terms["wrong_concept_detectable"],
        "free_generation": {
            "full_context_A": {"text": a_free["text"], "verdict": classify_text(a_free["text"], terms)["verdict"]},
            "fresh_compacted_B": {"text": b_free["text"], "verdict": b_greedy_v["verdict"]},
            "aligned_graft_E": {"text": e_free["text"], "verdict": e_greedy_v["verdict"]},
            "B_steps": b_free["steps"],
            "E_steps": e_free["steps"],
            "A_steps": a_free["steps"],
        },
        "fork_index": fork,
        "fork_found": fork is not None,
        "fork_tokens": {
            "B": (b_free["steps"][fork]["token"] if fork is not None and fork < len(b_free["steps"]) else None),
            "E": (e_free["steps"][fork]["token"] if fork is not None and fork < len(e_free["steps"]) else None),
        },
        "margin_E": margin_E,
        "margin_B": margin_B,
        "decoding_stability": decoding_stability,
        "fork_window_steps": window,
        "fork_center_step": center,
        "fork_lens_readout": {
            "A_full": a_read["steps"],
            "B_fresh": b_read["steps"],
            "E_graft": e_read["steps"],
        },
        "fork_lean_at_center": fork_lean,
        "lean_toward_A_magnitude": fork_lean["E_graft"]["lean_toward_A_magnitude"],
        "bucket": cls["bucket"],
        "bucket_detail": cls,
        "token_counts": st["token_counts"],
        "graft_pairs": st["pairs"],
    }


# ---------------------------------------------------------------------------
# Dry run (no torch): build every case, print the count, sample specs, cost.
# ---------------------------------------------------------------------------
def collect_specs(data_dir: Path, include_contaminated: bool):
    """Return [(PlantSpec, plant_dict)] for every usable sense/referent plant."""
    out = []
    for path in conversation_paths(data_dir):
        conv = json.loads(path.read_text())
        plant_by_id = {p.get("id"): p for p in conv.get("plants", [])}
        for spec in plant_specs_for(conv, include_contaminated):
            if spec.category not in USABLE_CATEGORIES:
                continue
            out.append((spec, plant_by_id[spec.plant_id]))
    return out


def estimate_forward_passes(n_cases, n_new, temperatures, window=3):
    n_decodings = 1 + len(temperatures)  # greedy + sampled temps
    free_gen = n_cases * 3 * n_decodings * n_new  # N x 3 states x decodings x tokens
    lens_readout = n_cases * 3 * (window + 1)      # A/B/E teacher-forced over fork window
    prefills = 12 * 3                              # ~12 convs x (A/B/old) snapshot prefills
    summaries = 12 * 256                           # ~12 conv summaries (<=256 tok each)
    total = free_gen + lens_readout + prefills + summaries
    return {
        "n_cases": n_cases,
        "n_decodings_per_case": n_decodings,
        "free_gen_headline_formula": f"{n_cases} cases x 3 states x {n_decodings} decodings x {n_new} tok = {free_gen}",
        "free_gen_passes": free_gen,
        "lens_readout_passes": lens_readout,
        "snapshot_prefills": prefills,
        "summary_gen_passes": summaries,
        "total_forward_passes_approx": total,
    }


def dry_run(data_dir: Path, include_contaminated: bool, n_new: int, temperatures) -> int:
    print("=" * 74)
    print("FREE-DIVERGENCE PROBE  --  DRY RUN (no model / no lens / no tokenizer)")
    print(f"data dir: {data_dir}")
    print(f"categories: {USABLE_CATEGORIES}   include_contaminated={include_contaminated}")
    print("=" * 74)
    cases = collect_specs(data_dir, include_contaminated)
    if not cases:
        print("NO CASES FOUND")
        return 1

    by_cat: dict[str, int] = {c: 0 for c in USABLE_CATEGORIES}
    ok = True
    for spec, _plant in cases:
        by_cat[spec.category] += 1

    print(f"\nBUILT {len(cases)} CASES  (" + ", ".join(f"{c}={by_cat[c]}" for c in USABLE_CATEGORIES) + ")")

    # A couple of sample specs (one sense, one referent) with derived vocab.
    shown = {"sense": 0, "referent": 0}
    print("\n--- SAMPLE SPECS ---")
    for spec, plant in cases:
        if shown.get(spec.category, 99) >= 1:
            continue
        shown[spec.category] = shown.get(spec.category, 0) + 1
        terms = derive_terms(plant)
        print(f"\n### {spec.plant_id}  ({spec.category})  [{spec.conv_id}]")
        print(f"  label (derived)   : {terms['label']!r}")
        print(f"  gold (RIGHT)      : {spec.gold}")
        print(f"  probe_user        : {spec.probe}")
        print(f"  right_terms       : {terms['right_terms']}")
        print(f"  wrong_terms       : {terms['wrong_terms']}")
        print(f"  wrong detectable  : {terms['wrong_concept_detectable']}")
        if not spec.gold.strip():
            ok = False

    est = estimate_forward_passes(len(cases), n_new, temperatures)
    print("\n--- COST ESTIMATE ---")
    print(f"  decodings/case    : {est['n_decodings_per_case']} (greedy + temps {list(temperatures)})")
    print(f"  free-gen headline : {est['free_gen_headline_formula']}")
    print(f"  lens-readout      : {est['lens_readout_passes']} passes (A/B/E over fork window)")
    print(f"  snapshot prefills : {est['snapshot_prefills']}  summaries: {est['summary_gen_passes']}")
    print(f"  TOTAL (approx)    : {est['total_forward_passes_approx']} single-token forward passes")
    # ~40-60ms/token decode on a 27B on one A100 -> rough wall-clock.
    secs = est["total_forward_passes_approx"] * 0.05
    print(f"  wall-clock @50ms/tok ~ {secs/60:.1f} min  (well under 60 min on one A100)")

    print("\n" + "=" * 74)
    print(f"DRY RUN {'PASSED' if ok else 'FAILED'}: {len(cases)} free-divergence case(s).")
    print("Design: model-compact each conv once (A/B/E), free greedy-gen B & E per")
    print("plant probe, find first argmax fork, record MARGIN + decoding stability,")
    print("read J-lens at the fork, classify into 3 buckets. No teacher-forced pin.")
    print("=" * 74)
    return 0 if ok else 1


def parse_args() -> argparse.Namespace:
    try:
        from boundary_probe import (
            DEFAULT_LENS_REPO,
            DEFAULT_LENS_REVISION,
            DEFAULT_QWEN36_LENS,
        )
    except Exception:
        DEFAULT_LENS_REPO = "neuronpedia/jacobian-lens"
        DEFAULT_LENS_REVISION = "qwen-n1000"
        DEFAULT_QWEN36_LENS = (
            "qwen3.6-27b/jlens/Salesforce-wikitext/"
            "Qwen3.6-27B_jacobian_lens_n1000.pt"
        )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-filename", default=DEFAULT_QWEN36_LENS)
    p.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output", default="outputs/qwen36_free_divergence.json")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--layers", default="8,16,24,32,40,48,56,62")
    p.add_argument("--alpha", type=float, default=0.75,
                   help="alpha_V for the aligned graft state E")
    p.add_argument("--max-new-tokens", type=int, default=36,
                   help="free-generation length for B and E")
    p.add_argument("--max-new-summary-tokens", type=int, default=256)
    p.add_argument("--temperatures", default="0.3,0.7",
                   help="extra sampled decoding temperatures for the robustness check")
    p.add_argument("--seed", type=int, default=1234,
                   help="fixed seed for the sampled decoding-robustness runs")
    p.add_argument("--margin-threshold", type=float, default=2.0,
                   help="fork logit-margin (top1-top2) at/above which a fork counts as 'decisive'")
    p.add_argument("--include-contaminated", action="store_true", default=False,
                   help="include plants flagged contaminated_early/tail (default: skip)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.add_argument("--dry-run", action="store_true",
                   help="construct + print cases without loading the model")
    return p.parse_args()


def parse_temperatures(arg: str) -> list[float]:
    return [float(x) for x in arg.split(",") if x.strip()]


def _histogram(values: list[float], edges) -> dict[str, int]:
    hist = {f"<{edges[0]}": 0}
    for a, b in zip(edges, edges[1:]):
        hist[f"[{a},{b})"] = 0
    hist[f">={edges[-1]}"] = 0
    for v in values:
        if v < edges[0]:
            hist[f"<{edges[0]}"] += 1
        elif v >= edges[-1]:
            hist[f">={edges[-1]}"] += 1
        else:
            for a, b in zip(edges, edges[1:]):
                if a <= v < b:
                    hist[f"[{a},{b})"] += 1
                    break
    return hist


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    buckets = {"FORK_TOWARD_A": 0, "SUBTLE_LEAN": 0, "DISCONFIRMING": 0}
    by_cat_bucket: dict[str, dict[str, int]] = {}
    lean_mags: list[float] = []
    margins: list[float] = []
    lines: list[str] = []
    n = 0
    for pid, case in result["cases"].items():
        if "error" in case:
            lines.append(f"{pid}: ERROR {case['error']}")
            continue
        n += 1
        b = case["bucket"]
        buckets[b] = buckets.get(b, 0) + 1
        cat = case.get("category", "?")
        by_cat_bucket.setdefault(cat, {"FORK_TOWARD_A": 0, "SUBTLE_LEAN": 0, "DISCONFIRMING": 0})
        by_cat_bucket[cat][b] = by_cat_bucket[cat].get(b, 0) + 1
        lm = case.get("lean_toward_A_magnitude")
        if isinstance(lm, (int, float)):
            lean_mags.append(lm)
        me = case.get("margin_E")
        if isinstance(me, (int, float)):
            margins.append(me)
        lines.append(
            f"{pid} ({cat}): {b} | fork@{case['fork_index']} "
            f"marginE={None if case['margin_E'] is None else round(case['margin_E'],2)} "
            f"| {case['decoding_stability']['summary']} "
            f"| leanA={lm}"
        )
    return {
        "n_cases": n,
        "three_bucket_counts": buckets,
        "three_bucket_counts_by_category": by_cat_bucket,
        "headline": (
            f"{buckets['FORK_TOWARD_A']} toward-A, {buckets['SUBTLE_LEAN']} subtle, "
            f"{buckets['DISCONFIRMING']} disconfirming of N={n}"
        ),
        "lean_toward_A_magnitude_distribution": {
            "values": lean_mags,
            "n": len(lean_mags),
            "mean": (sum(lean_mags) / len(lean_mags)) if lean_mags else None,
            "histogram": _histogram(lean_mags, [-2, -1, 0, 1, 2, 3]),
        },
        "fork_margin_E_distribution": {
            "values": margins,
            "n": len(margins),
            "mean": (sum(margins) / len(margins)) if margins else None,
            "histogram": _histogram(margins, [0, 1, 2, 4, 8]),
        },
        "per_case": lines,
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    temperatures = parse_temperatures(args.temperatures)

    if args.dry_run:
        raise SystemExit(dry_run(data_dir, args.include_contaminated, args.max_new_tokens, temperatures))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from boundary_probe import choose_layers, dtype_from_name
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

    eos = model.config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple, set)) else {eos}
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    result: dict[str, Any] = {
        "model": args.model,
        "design": (
            "FREE-GENERATION divergence lens probe, DISTRIBUTION version. Every "
            "usable sense/referent plant (contaminated skipped) is a case. Each "
            "conversation is compacted once (model summary; A=full, B=fresh, "
            "E=aligned-graft alpha_V). B and E FREELY generate after each plant "
            "probe; we find the first argmax fork, record its MARGIN (top1-top2) "
            "for E and B, check whether E's right-concept lean is STABLE across "
            "greedy + sampled temperatures, read the J-lens at the fork under "
            "A/B/E, and classify each case into three pre-registered buckets "
            "(FORK_TOWARD_A / SUBTLE_LEAN / DISCONFIRMING). The headline is the "
            "three-bucket distribution; any vivid case is one point on it."
        ),
        "lens": {
            "repo": args.lens_repo,
            "filename": args.lens_filename,
            "revision": args.lens_revision,
            "source_layers": lens.source_layers,
            "sampled_layers": layers,
        },
        "top_k": args.top_k,
        "alpha_V": args.alpha,
        "max_new_tokens": args.max_new_tokens,
        "temperatures": temperatures,
        "seed": args.seed,
        "margin_threshold": args.margin_threshold,
        "categories": list(USABLE_CATEGORIES),
        "include_contaminated": args.include_contaminated,
        "cases": {},
    }

    # Group cases by conversation so each conv is compacted (A/B/E) only once.
    from collections import OrderedDict
    by_conv: "OrderedDict[str, list[tuple[PlantSpec, dict]]]" = OrderedDict()
    for spec, plant in collect_specs(data_dir, args.include_contaminated):
        by_conv.setdefault(spec.conv_id, []).append((spec, plant))

    for path in conversation_paths(data_dir):
        conv = json.loads(path.read_text())
        cid = conv["id"]
        specs = by_conv.get(cid)
        if not specs:
            continue
        print(f"CONV {cid}: {len(specs)} plant(s)", flush=True)
        try:
            st = build_conv_states(conv, model, tokenizer, args.alpha, args.max_new_summary_tokens)
        except Exception as exc:  # noqa: BLE001
            for spec, _plant in specs:
                result["cases"][spec.plant_id] = {
                    "plant_id": spec.plant_id,
                    "conversation_id": cid,
                    "error": f"conv-build {type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            print(f"ERROR CONV {cid}: {type(exc).__name__}: {exc}", flush=True)
            continue

        for spec, plant in specs:
            print(f"  RUN {spec.plant_id}", flush=True)
            try:
                result["cases"][spec.plant_id] = run_plant(
                    spec, plant, st, model, lens_model, lens, tokenizer,
                    layers, args.top_k, args.max_new_tokens, temperatures,
                    args.seed, args.margin_threshold, eos_ids,
                )
                print(f"  DONE {spec.plant_id}: {result['cases'][spec.plant_id]['bucket']}", flush=True)
            except Exception as exc:  # noqa: BLE001
                result["cases"][spec.plant_id] = {
                    "plant_id": spec.plant_id,
                    "conversation_id": cid,
                    "category": spec.category,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                print(f"  ERROR {spec.plant_id}: {type(exc).__name__}: {exc}", flush=True)

        del st
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result["summary"] = summarize(result)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(out_path)
    print("SUMMARY (three-bucket distribution is the headline):")
    print("  " + result["summary"]["headline"])
    print(f"  by category: {result['summary']['three_bucket_counts_by_category']}")
    print(f"  lean-toward-A magnitude histogram: "
          f"{result['summary']['lean_toward_A_magnitude_distribution']['histogram']}")
    for line in result["summary"]["per_case"]:
        print("  " + line)


if __name__ == "__main__":
    main()
