#!/usr/bin/env python3
"""EFFECT-BOUNDING probe for ValueGraft on Qwen3.6-27B (spin-proof, single number).

WHAT THIS IS (and is NOT)
-------------------------
This is the pre-registered, placebo-controlled BOUNDING experiment specified in
``free-divergence-design-rationale.md`` -> "TRAJECTORY IDEA -> REFRAMED (Fable
gut-check 07-07): BOUND the effect, don't hunt vividness". It is NOT a
vividness / fork / trajectory hunt. It produces ONE defensible number with a
confidence interval and a mechanical verdict, nothing to narrate.

The metric is PURELY BEHAVIORAL (no J-lens is loaded or needed). The "lens
model" framing only fixes the model + dose (Qwen3.6-27B, alpha_V=0.75) so the
number is comparable to the paper's lens-regime claims.

DESIGN (the confound-killer)
----------------------------
For every usable sense/referent plant (the SAME 43-case set the free-divergence
probe used -- ``free_divergence_probe.collect_specs``), we compact each
conversation ONCE with the existing machinery and build FOUR cache states that
share bit-identical keys and differ only in a subset of value-cache rows:

  * A  = full-context (no compaction)                                [reference]
  * B  = fresh-compacted (system + model summary + retained tail)    [baseline]
  * E  = B with the aligned summary-token VALUES V-grafted (alpha_V) [treatment]
  * P  = B with SHUFFLED values V-grafted: same alpha, same grafted
         positions, but each fresh slot receives the value of the WRONG summary
         token (a seeded derangement of the source alignment).        [PLACEBO]

We then TEACHER-FORCE the SAME shared gold continuation under A/B/E/P over a
fixed PRE-DIVERGENCE window (the first K tokens of the gold continuation). Every
arm scores IDENTICAL tokens at IDENTICAL positions -- the one thing that must be
right. The ONLY difference between arms is the cache state.

METRIC (per case)
-----------------
Per position k in [0, K): the gold-token logprob under each state. Per case we
average over the window:
    delta_EB   = mean_k( logprob_E[k] - logprob_B[k] )
    delta_PB   = mean_k( logprob_P[k] - logprob_B[k] )   (placebo vs baseline)
    delta_EP   = mean_k( logprob_E[k] - logprob_P[k] )   (real vs placebo)

AGGREGATE (across cases)
------------------------
Mean of each per-case delta with a PAIRED bootstrap 95% CI (resample cases with
replacement; the pairing across A/B/E/P is preserved because every state scores
the same tokens for the same case).

PRE-REGISTERED NULL (stated mechanically from the CIs)
------------------------------------------------------
NULL holds (graft indistinguishable from a random-value graft) if EITHER:
  (i)  the (E-B) 95% CI includes 0, OR
  (ii) the (E-placebo) 95% CI includes 0.
The graft's effect is BOUNDED as real-and-above-placebo ONLY if BOTH the (E-B)
CI and the (E-placebo) CI exclude 0 on the positive side. The verdict is emitted
directly from the CIs -- no soft "subtle" bucket to hide in.

SELF-TEST
---------
``--dry-run`` (no torch / model / tokenizer): builds every case, prints N, the
window K, a couple of sample specs, and the estimated forward-pass count / cost.
"""

from __future__ import annotations

import argparse
import json
import random
import traceback
from pathlib import Path
from typing import Any

# Both of these are import-safe with NO ML deps (torch is deferred past their own
# dry-run gates), so ``--dry-run`` runs on a CPU-only box. We reuse the EXACT case
# set + plant-spec construction the free-divergence probe used, so the bound is
# directly comparable to that experiment's 43 cases.
from free_divergence_probe import USABLE_CATEGORIES, collect_specs
from gap_closure_27b import PlantSpec, conversation_paths


# ---------------------------------------------------------------------------
# Placebo: a seeded derangement of the source (write-time) alignment. Same
# grafted fresh positions, same alpha, same real value-vector distribution
# (norm-matched, drawn from the same summary), but each fresh slot gets the
# value of the WRONG summary token. This is the "random-value graft" baseline.
# ---------------------------------------------------------------------------
def make_placebo_pairs(pairs: list[tuple[int, int]], seed: int) -> list[tuple[int, int]]:
    """Permute the OLD (source) index of each pair into a derangement.

    New (destination) positions are UNCHANGED -- the graft touches exactly the
    same cache rows with exactly the same alpha. Only the *content* moved in is
    scrambled: fresh slot i receives summary-value j != its true source.
    """
    if len(pairs) < 2:
        return list(pairs)  # cannot derange < 2 items; degenerate, handled upstream
    new_idx = [n for n, _ in pairs]
    old_idx = [o for _, o in pairs]
    rng = random.Random(seed)
    perm = list(range(len(old_idx)))
    # Retry a full shuffle until no element maps to itself (derangement). For
    # len>=2 this terminates almost surely; cap attempts and fall back to a
    # rotation which is guaranteed fixed-point-free.
    for _ in range(64):
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(len(perm))):
            break
    else:
        perm = perm[1:] + perm[:1]  # rotation: guaranteed no fixed point
    return [(new_idx[i], old_idx[perm[i]]) for i in range(len(pairs))]


# ---------------------------------------------------------------------------
# Cost / dry-run helpers (no torch).
# ---------------------------------------------------------------------------
def estimate_forward_passes(n_cases: int, k: int, n_convs: int,
                            summary_tokens: int = 256) -> dict[str, Any]:
    states = 4  # A / B / E / placebo
    # Each state's window is scored in ONE teacher-forced forward call (feed
    # suffix + gold[:K-1], keep K logits). We count it as K scored positions.
    tf_calls = n_cases * states
    tf_scored_positions = n_cases * states * k
    # Per conversation: one summary generation (<= summary_tokens) + 3 snapshot
    # prefills (A full, B fresh, old-with-summary). Placebo is a CPU tensor op
    # on the existing B/old snapshots -- NO extra prefill or forward pass.
    snapshot_prefills = n_convs * 3
    summary_gen_passes = n_convs * summary_tokens
    total = tf_scored_positions + snapshot_prefills + summary_gen_passes
    return {
        "n_cases": n_cases,
        "window_K": k,
        "states_per_case": states,
        "teacher_forced_calls": tf_calls,
        "teacher_forced_calls_formula": f"{n_cases} cases x {states} states = {tf_calls} forward calls",
        "teacher_forced_scored_positions": tf_scored_positions,
        "snapshot_prefills": snapshot_prefills,
        "summary_gen_passes_approx": summary_gen_passes,
        "total_forward_passes_approx": total,
    }


def dry_run(data_dir: Path, include_contaminated: bool, k: int, seed: int) -> int:
    print("=" * 76)
    print("EFFECT-BOUND PROBE  --  DRY RUN (no model / no lens / no tokenizer / no torch)")
    print(f"data dir   : {data_dir}")
    print(f"categories : {USABLE_CATEGORIES}   include_contaminated={include_contaminated}")
    print(f"window K   : {k}   placebo_seed={seed}")
    print("=" * 76)

    cases = collect_specs(data_dir, include_contaminated)
    if not cases:
        print("NO CASES FOUND")
        return 1

    by_cat: dict[str, int] = {c: 0 for c in USABLE_CATEGORIES}
    conv_ids = set()
    ok = True
    for spec, _plant in cases:
        by_cat[spec.category] = by_cat.get(spec.category, 0) + 1
        conv_ids.add(spec.conv_id)
        if not spec.gold.strip():
            ok = False

    n_convs = len(conv_ids)
    print(f"\nBUILT {len(cases)} CASES  (" +
          ", ".join(f"{c}={by_cat[c]}" for c in USABLE_CATEGORIES) +
          f")  across {n_convs} conversation(s)")

    # Sample specs (one per category) + show the placebo pairing on a toy alignment.
    print("\n--- SAMPLE SPECS ---")
    shown: dict[str, int] = {}
    for spec, _plant in cases:
        if shown.get(spec.category, 0) >= 1:
            continue
        shown[spec.category] = shown.get(spec.category, 0) + 1
        gold_head = spec.gold[:100] + ("..." if len(spec.gold) > 100 else "")
        print(f"\n### {spec.plant_id}  ({spec.category})  [{spec.conv_id}]")
        print(f"  probe_user     : {spec.probe}")
        print(f"  GOLD (forced)  : {gold_head}")
        print(f"  first K={k} tokens of GOLD are the scored PRE-DIVERGENCE window")

    demo_pairs = [(10, 100), (11, 101), (12, 102), (13, 103), (14, 104)]
    placebo = make_placebo_pairs(demo_pairs, seed)
    print("\n--- PLACEBO PAIRING (illustrative, seed-deterministic) ---")
    print(f"  true source  pairs : {demo_pairs}")
    print(f"  placebo pairs      : {placebo}")
    print(f"  (dest positions identical; each source index moved to a WRONG one)")

    est = estimate_forward_passes(len(cases), k, n_convs)
    print("\n--- COST ESTIMATE ---")
    print(f"  states/case        : {est['states_per_case']} (A / B / E / placebo)")
    print(f"  teacher-forced     : {est['teacher_forced_calls_formula']}")
    print(f"  scored positions   : {est['teacher_forced_scored_positions']}"
          f" (= {len(cases)} cases x 4 states x K={k})")
    print(f"  snapshot prefills  : {est['snapshot_prefills']} ({n_convs} convs x 3 [A/B/old])")
    print(f"  summary generation : ~{est['summary_gen_passes_approx']} tok"
          f" ({n_convs} convs x <=256)")
    print(f"  TOTAL (approx)     : {est['total_forward_passes_approx']} single-token-equiv forward passes")
    secs = est["total_forward_passes_approx"] * 0.05  # ~50ms/token on a 27B / one A100
    print(f"  wall-clock @50ms   : ~{secs/60:.1f} min (well under 30 min on one A100;")
    print(f"                        NO long rollouts -- only K-token teacher-forced windows)")

    print("\n" + "=" * 76)
    print(f"DRY RUN {'PASSED' if ok else 'FAILED'}: {len(cases)} effect-bound case(s), K={k}.")
    print("Design: compact each conv once (A/B/E/placebo, shared keys); teacher-force")
    print("the SAME gold[:K] window under all four; metric = mean gold-logprob delta;")
    print("paired bootstrap CI for (E-B), (placebo-B), (E-placebo); mechanical verdict.")
    print("=" * 76)
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Aggregation: paired bootstrap CI (no numpy dependency required).
# ---------------------------------------------------------------------------
def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def paired_bootstrap_ci(per_case: list[float], n_boot: int, seed: int,
                        alpha: float = 0.05) -> dict[str, Any]:
    """Bootstrap CI for the MEAN of a paired per-case delta series."""
    n = len(per_case)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None,
                "excludes_zero": False, "n_boot": n_boot}
    mean = sum(per_case) / n
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += per_case[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    ci_low = _percentile(boots, alpha / 2)
    ci_high = _percentile(boots, 1 - alpha / 2)
    return {
        "n": n,
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "excludes_zero": (ci_low > 0.0) or (ci_high < 0.0),
        "ci_level": 1 - alpha,
        "n_boot": n_boot,
    }


def parse_args() -> argparse.Namespace:
    from gap_closure_27b import DEFAULT_DATA_DIR

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output", default="../results/effect_bound/summary.json")
    p.add_argument("--alpha", type=float, default=0.75,
                   help="alpha_V for the aligned value-graft state E (and placebo P)")
    p.add_argument("--window-k", type=int, default=12,
                   help="pre-divergence window: number of leading gold tokens scored (8-16)")
    p.add_argument("--max-new-summary-tokens", type=int, default=256)
    p.add_argument("--n-boot", type=int, default=10000, help="bootstrap resamples")
    p.add_argument("--placebo-seed", type=int, default=20260707,
                   help="seed for the placebo derangement (fixed for reproducibility)")
    p.add_argument("--boot-seed", type=int, default=12345, help="seed for the bootstrap")
    p.add_argument("--include-contaminated", action="store_true", default=False)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.add_argument("--dry-run", action="store_true",
                   help="construct + print cases + cost without loading the model")
    return p.parse_args()


# ===========================================================================
# Everything below needs torch. Kept out of module scope so --dry-run runs on a
# CPU-only box with no ML deps.
# ===========================================================================
def build_conv_states(conv, model, tokenizer, alpha, placebo_seed, max_new_summary_tokens):
    """Compact one conversation ONCE and build A / B / E / placebo snapshots.

    Same compaction + graft path as ``gap_closure_27b.main``; the placebo reuses
    the identical (b_snap, old_snap) with a DERANGED alignment. Requires torch.
    """
    from boundary_probe import (
        alignment_pairs_by_exact_tokens,
        blend_values,
        build_b_messages,
        find_subsequence,
        generate_summary,
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
    placebo_pairs = make_placebo_pairs(pairs, placebo_seed)

    a_snap, _ = snapshot_standard_kv(model, full_ids)
    b_snap, _ = snapshot_standard_kv(model, b_ids)
    old_snap, _ = snapshot_standard_kv(model, summary["old_ids"])
    e_snap = blend_values(b_snap, old_snap, pairs, alpha)
    p_snap = blend_values(b_snap, old_snap, placebo_pairs, alpha)

    return {
        "msgs": msgs,
        "b_msgs": b_msgs,
        "A_snap": a_snap,
        "B_snap": b_snap,
        "E_snap": e_snap,
        "P_snap": p_snap,
        "full_ids": full_ids,
        "b_ids": b_ids,
        "summary_text": summary_text,
        "pairs": len(pairs),
        "token_counts": {
            "full": len(full_ids),
            "fresh_compacted": len(b_ids),
            "old_with_summary": len(summary["old_ids"]),
            "summary_tokens": len(summary_text_ids),
        },
    }


def tf_gold_logprobs_per_pos(model, snap, feed_ids, target_ids, npos):
    """PER-POSITION teacher-forced logprob of each target token.

    Mirrors ``gap_closure_27b.tf_gold_logprob`` but returns the FULL per-position
    list (not just the mean) so the window can be inspected. feed = prefix_suffix
    + target[:-1]; the last ``len(target)`` logit positions score the target.
    """
    import torch

    from boundary_probe import model_input_device, rebuild_standard_cache, token_tensor

    n = len(target_ids)
    cache = rebuild_standard_cache(snap, model)
    dev = model_input_device(model)
    pos = torch.arange(npos, npos + len(feed_ids), device=dev)[None]
    with torch.no_grad():
        out = model(
            input_ids=token_tensor(model, feed_ids),
            past_key_values=cache,
            position_ids=pos,
            use_cache=True,
            logits_to_keep=n,
        )
    lp = torch.log_softmax(out.logits[0].float(), dim=-1)
    tgt = torch.tensor(target_ids, device=lp.device)
    vals = lp[torch.arange(n, device=lp.device), tgt]
    return [float(x) for x in vals.tolist()]


def run_plant(spec: PlantSpec, st, model, tokenizer, window_k) -> dict[str, Any]:
    from boundary_probe import render_ids

    def suffix(mm, probe):
        gen = render_ids(tokenizer, mm + [{"role": "user", "content": probe}], True)
        cn = render_ids(tokenizer, mm, False)
        if gen[: len(cn)] != cn:
            raise ValueError("probe generation prompt does not extend prefix")
        return gen[len(cn):]

    # Full gold, truncated to the pre-divergence window K.
    gold_ids = tokenizer(spec.gold, add_special_tokens=False).input_ids
    tgt = gold_ids[:window_k]
    if len(tgt) < 2:
        raise ValueError(f"gold shorter than 2 tokens after windowing (have {len(tgt)})")

    sa = suffix(st["msgs"], spec.probe)      # A (full) side
    sb = suffix(st["b_msgs"], spec.probe)    # B / E / placebo (compacted) side
    a_prefix = st["token_counts"]["full"]
    b_prefix = st["token_counts"]["fresh_compacted"]

    feed_ids = tgt[:-1]  # appended after the suffix; last K logits score tgt

    lp_A = tf_gold_logprobs_per_pos(model, st["A_snap"], sa + feed_ids, tgt, a_prefix)
    lp_B = tf_gold_logprobs_per_pos(model, st["B_snap"], sb + feed_ids, tgt, b_prefix)
    lp_E = tf_gold_logprobs_per_pos(model, st["E_snap"], sb + feed_ids, tgt, b_prefix)
    lp_P = tf_gold_logprobs_per_pos(model, st["P_snap"], sb + feed_ids, tgt, b_prefix)

    k = len(tgt)
    d_EB = [lp_E[i] - lp_B[i] for i in range(k)]
    d_PB = [lp_P[i] - lp_B[i] for i in range(k)]
    d_EP = [lp_E[i] - lp_P[i] for i in range(k)]
    mean = lambda xs: sum(xs) / len(xs)

    return {
        "plant_id": spec.plant_id,
        "conversation_id": spec.conv_id,
        "category": spec.category,
        "probe": spec.probe,
        "gold": spec.gold,
        "window_k": k,
        "gold_window_token_count": k,
        "logprob_per_pos": {"A_full": lp_A, "B_fresh": lp_B, "E_graft": lp_E, "P_placebo": lp_P},
        "window_mean_logprob": {
            "A_full": mean(lp_A), "B_fresh": mean(lp_B),
            "E_graft": mean(lp_E), "P_placebo": mean(lp_P),
        },
        "delta_EB": mean(d_EB),   # treatment vs baseline (THE headline per-case value)
        "delta_PB": mean(d_PB),   # placebo vs baseline
        "delta_EP": mean(d_EP),   # treatment vs placebo (real beats random?)
        "token_counts": st["token_counts"],
        "graft_pairs": st["pairs"],
    }


def aggregate(cases: dict[str, Any], n_boot: int, boot_seed: int) -> dict[str, Any]:
    d_EB = [c["delta_EB"] for c in cases.values() if "delta_EB" in c]
    d_PB = [c["delta_PB"] for c in cases.values() if "delta_PB" in c]
    d_EP = [c["delta_EP"] for c in cases.values() if "delta_EP" in c]

    # Distinct sub-seeds keep the three CIs from sharing a resample pattern.
    ci_EB = paired_bootstrap_ci(d_EB, n_boot, boot_seed + 1)
    ci_PB = paired_bootstrap_ci(d_PB, n_boot, boot_seed + 2)
    ci_EP = paired_bootstrap_ci(d_EP, n_boot, boot_seed + 3)

    eb_excl = ci_EB["excludes_zero"] and (ci_EB["mean"] is not None and ci_EB["mean"] > 0)
    ep_excl = ci_EP["excludes_zero"] and (ci_EP["mean"] is not None and ci_EP["mean"] > 0)

    # PRE-REGISTERED mechanical verdict.
    null_by_EB = not ci_EB["excludes_zero"]
    null_by_EP = not ci_EP["excludes_zero"]
    if null_by_EB or null_by_EP:
        verdict = "NULL"
        if null_by_EB and null_by_EP:
            why = "(E-B) CI includes 0 AND (E-placebo) CI includes 0"
        elif null_by_EB:
            why = "(E-B) CI includes 0: graft not distinguishable from baseline B"
        else:
            why = "(E-placebo) CI includes 0: graft not distinguishable from a random-value graft"
    elif eb_excl and ep_excl:
        verdict = "EFFECT_BOUNDED_ABOVE_PLACEBO"
        why = ("both (E-B) and (E-placebo) 95% CIs exclude 0 on the positive side: "
               "the value-graft raises gold-continuation logprob over baseline AND "
               "beats a norm-matched random-value graft")
    else:
        verdict = "NULL"
        why = "effect present but not positive-directional in both CIs"

    return {
        "n_cases_scored": len(d_EB),
        "delta_E_minus_B": ci_EB,
        "delta_placebo_minus_B": ci_PB,
        "delta_E_minus_placebo": ci_EP,
        "verdict": verdict,
        "verdict_reason": why,
        "preregistered_null_rule": (
            "NULL if (E-B) CI includes 0 OR (E-placebo) CI includes 0; "
            "EFFECT_BOUNDED_ABOVE_PLACEBO only if BOTH CIs exclude 0 positively."
        ),
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    if args.dry_run:
        raise SystemExit(dry_run(data_dir, args.include_contaminated,
                                 args.window_k, args.placebo_seed))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from boundary_probe import dtype_from_name

    torch_dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    result: dict[str, Any] = {
        "model": args.model,
        "experiment": "effect_bound",
        "design": (
            "Pre-registered, placebo-controlled BOUNDING test of the aligned "
            "value-graft. Four cache states share bit-identical keys: A=full, "
            "B=fresh-compacted, E=B with aligned summary-token VALUES V-grafted "
            "(alpha_V), P=B with the SAME grafted positions but a seeded "
            "DERANGEMENT of the source alignment (norm-matched random-value "
            "graft). All four teacher-force the SAME gold[:K] pre-divergence "
            "window; metric = mean gold-token logprob delta over the window. "
            "Paired bootstrap 95% CIs for (E-B), (placebo-B), (E-placebo). "
            "Purely behavioral -- no J-lens. Case set = free_divergence_probe's "
            "sense/referent plants (comparable to the 43-case probe)."
        ),
        "alpha_V": args.alpha,
        "window_k": args.window_k,
        "n_boot": args.n_boot,
        "placebo_seed": args.placebo_seed,
        "boot_seed": args.boot_seed,
        "categories": list(USABLE_CATEGORIES),
        "include_contaminated": args.include_contaminated,
        "cases": {},
    }

    # Group cases by conversation so each conv is compacted only once.
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
            st = build_conv_states(conv, model, tokenizer, args.alpha,
                                   args.placebo_seed, args.max_new_summary_tokens)
        except Exception as exc:  # noqa: BLE001
            for spec, _plant in specs:
                result["cases"][spec.plant_id] = {
                    "plant_id": spec.plant_id, "conversation_id": cid,
                    "error": f"conv-build {type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            print(f"ERROR CONV {cid}: {type(exc).__name__}: {exc}", flush=True)
            continue

        for spec, _plant in specs:
            try:
                row = run_plant(spec, st, model, tokenizer, args.window_k)
                result["cases"][spec.plant_id] = row
                print(f"  {spec.plant_id} ({spec.category}): "
                      f"dEB={row['delta_EB']:+.4f} dPB={row['delta_PB']:+.4f} "
                      f"dEP={row['delta_EP']:+.4f}", flush=True)
            except Exception as exc:  # noqa: BLE001
                result["cases"][spec.plant_id] = {
                    "plant_id": spec.plant_id, "conversation_id": cid,
                    "category": spec.category,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                print(f"  ERROR {spec.plant_id}: {type(exc).__name__}: {exc}", flush=True)

        del st
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scored = {k: v for k, v in result["cases"].items() if "delta_EB" in v}
    result["summary"] = aggregate(scored, args.n_boot, args.boot_seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(out_path)
    s = result["summary"]
    print("SUMMARY (bounding numbers are the headline):")
    for name, key in (("E - B         ", "delta_E_minus_B"),
                      ("placebo - B   ", "delta_placebo_minus_B"),
                      ("E - placebo   ", "delta_E_minus_placebo")):
        d = s[key]
        print(f"  {name}: mean={d['mean']:+.4f}  95% CI=[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
              f"  excludes_0={d['excludes_zero']}")
    print(f"  VERDICT: {s['verdict']} -- {s['verdict_reason']}")


if __name__ == "__main__":
    main()
