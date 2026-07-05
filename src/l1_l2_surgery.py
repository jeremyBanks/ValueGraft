"""L1 — null surgery; L2 — trivial eviction.

Protocol (used experiment-wide): build each arm's cache over prompt[:-1],
compute first-continuation logits by running the last prompt token as a
1-token decode step, then greedy-generate and teacher-force-score. This keeps
all forward passes identically shaped across arms.

L1a: split each layer's K/V arrays and reconcatenate → bit-identical.
L1b: GappedKVCache with an empty evicted range → bit-identical.
L2:  evict an irrelevant filler span, original positions kept → continuation
     should stay semantically on-track; drift quantified by teacher-forced
     logprob deltas along the reference continuation.
"""

import math
import sys

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "src")
from kvlib import (
    GappedKVCache,
    first_step_logits,
    greedy_generate,
    make_prompt_ids,
    max_abs_diff,
    prefill,
    rebuild_cache,
    snapshot_cache,
    teacher_forced_logprobs,
)

import os
MODEL = os.environ.get("SC_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
MAX_TOKENS = 60

FILLER = (
    "By the way, completely unrelated aside: I watched a documentary about "
    "tide pools last night. Sea anemones apparently fight in slow motion over "
    "territory, which was oddly soothing to watch. Barnacles cement their "
    "heads to rocks permanently. Anyway, that has nothing to do with anything."
)

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "We're planning a small conference. Venue options were the Harbor Hotel and the University Hall; we rejected the hotel due to cost."},
    {"role": "assistant", "content": "Got it — University Hall it is. Shall we discuss catering next?"},
    {"role": "user", "content": FILLER},
    {"role": "assistant", "content": "That does sound soothing! Now, back to the conference planning — catering?"},
    {"role": "user", "content": "Yes. Also remember the keynote speaker is Dr. Alvarez. What should we sort out first for catering?"},
]


def gapped_from_snapshot(snap, evict_start, evict_end):
    out = []
    for k, v, off in snap:
        gk = mx.concatenate([k[..., :evict_start, :], k[..., evict_end:, :]], axis=2)
        gv = mx.concatenate([v[..., :evict_start, :], v[..., evict_end:, :]], axis=2)
        out.append(GappedKVCache(gk, gv, off))
    return out


def main():
    model, tokenizer = load(MODEL)
    eos_ids = set(tokenizer.eos_token_ids or [tokenizer.eos_token_id])

    ids = make_prompt_ids(tokenizer, MESSAGES)
    n = len(ids)
    print(f"prompt tokens: {n}")

    # Reference arm: prefill all but last token; 1-token decode for the last.
    cache, _ = prefill(model, ids[:-1])
    snap = snapshot_cache(cache)  # snapshot BEFORE the last-token step
    logits0 = first_step_logits(model, cache, ids[-1])
    ref_toks, _ = greedy_generate(model, cache, logits0, MAX_TOKENS, eos_ids)
    print("ref:", tokenizer.decode(ref_toks)[:100].replace("\n", " "))

    def run_arm(mk_cache):
        c = mk_cache()
        l0 = first_step_logits(model, c, ids[-1])
        toks, _ = greedy_generate(model, c, l0, MAX_TOKENS, eos_ids)
        c2 = mk_cache()
        l0b = first_step_logits(model, c2, ids[-1])
        lps, _ = teacher_forced_logprobs(model, c2, l0b, ref_toks)
        return toks, lps

    # Reference teacher-forced logprobs (of its own continuation).
    c_ref = rebuild_cache(snap)
    lref = first_step_logits(model, c_ref, ids[-1])
    ref_lps, _ = teacher_forced_logprobs(model, c_ref, lref, ref_toks)

    # --- L1a: split + reconcat ---
    mid = (n - 1) // 2
    t1, lps1 = run_arm(lambda: rebuild_cache(
        [(mx.concatenate([k[..., :mid, :], k[..., mid:, :]], axis=2),
          mx.concatenate([v[..., :mid, :], v[..., mid:, :]], axis=2), off)
         for k, v, off in snap]))
    d1 = max(abs(a - b) for a, b in zip(ref_lps, lps1))
    print(f"L1a split+reconcat: identical={t1 == ref_toks} max|dlp|={d1}")

    # --- L1b: gapped cache, empty eviction ---
    t2, lps2 = run_arm(lambda: gapped_from_snapshot(snap, mid, mid))
    d2 = max(abs(a - b) for a, b in zip(ref_lps, lps2))
    print(f"L1b gapped(empty):  identical={t2 == ref_toks} max|dlp|={d2}")
    l1_ok = t1 == ref_toks and d1 == 0.0 and t2 == ref_toks and d2 == 0.0

    # --- L2: evict the filler user message (message located via <|im_start|>
    # scan; prefix re-tokenization is unstable for this template because the
    # final assistant message gets an empty <think> block earlier ones lack).
    im_start = tokenizer.encode("<|im_start|>")[0]
    starts = [i for i, t in enumerate(ids) if t == im_start]
    ev_start, ev_end = starts[3], starts[4]
    print(f"evicting filler tokens [{ev_start}, {ev_end}) = {ev_end - ev_start} tokens")

    t3, lps3 = run_arm(lambda: gapped_from_snapshot(snap, ev_start, ev_end))
    mean_ref = sum(ref_lps) / len(ref_lps)
    mean_l2 = sum(lps3) / len(lps3)
    print(f"L2 evict filler: mean t-f logprob ref={mean_ref:.4f} evicted={mean_l2:.4f} "
          f"delta={mean_l2 - mean_ref:.4f}")
    print("L2 gen:", tokenizer.decode(t3)[:100].replace("\n", " "))

    # Soft criterion: drift small in logprob terms (< 0.15 nats mean delta).
    l2_ok = abs(mean_l2 - mean_ref) < 0.15
    print("L1 PASS" if l1_ok else "L1 FAIL")
    print(f"L2 {'PASS' if l2_ok else 'FAIL'} (criterion: |mean t-f logprob delta| < 0.15 nats)")
    return 0 if (l1_ok and l2_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
