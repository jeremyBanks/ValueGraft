"""L3 — null-transplant identities; L4 — tokenization stability audit (toy).

L3a: Arm E with alpha=0 must reproduce Arm B exactly (teacher-forced
     logprobs identical along a fixed continuation).
L3b: Arm E with alpha=1 and old context == new context (no compaction) must
     reproduce Arm A exactly.
L4:  count tail/S tokens in Arm B's context NOT covered by the exact-twin
     alignment (expect only a handful at region edges).

Also runs the full arm set (A, B, C, D, E-post alpha=1) on the toy
conversation as a smoke test, printing teacher-forced mean logprobs.
"""

import sys

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "src")
import arms
from arms import (
    N_SINK,
    arm_a_build,
    arm_b_build,
    arm_c_snapshot,
    arm_e_snapshot,
    build_alignment,
    build_b_messages,
    canonical_ids,
    gapped_cache_from,
    generate_summary,
    message_token_starts,
    pick_tail_start,
    render,
)
from kvlib import (
    first_step_logits,
    greedy_generate,
    prefill,
    rebuild_cache,
    snapshot_cache,
    teacher_forced_logprobs,
)

import os
MODEL = os.environ.get("SC_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "We're planning a small conference. Venue options were the Harbor Hotel, the University Hall, and the Riverside Pavilion; we picked the second option because of cost."},
    {"role": "assistant", "content": "Got it — University Hall it is. Shall we discuss catering next?"},
    {"role": "user", "content": "One more thing: the projector rental quote came to $487 for the weekend. Also, I tried the caterer called GreenFork and they failed the tasting badly, so they're out."},
    {"role": "assistant", "content": "Noted on both counts. So we need a different caterer, and the projector is sorted."},
    {"role": "user", "content": "Right. Let's talk schedule now: two days, keynote each morning, workshops after lunch."},
    {"role": "assistant", "content": "A classic structure. Day 1: keynote 9-10, workshops 1-5. Day 2 mirrors it. Sound good?"},
    {"role": "user", "content": "Sounds good. For the venue we picked, what should we double-check before signing? No need to re-explain the venue choice itself."},
]


def tf_mean(model, snap_or_cache, ids_last, cont_ids, rebuild):
    cache = rebuild()
    l0 = first_step_logits(model, cache, ids_last)
    lps, _ = teacher_forced_logprobs(model, cache, l0, cont_ids)
    return lps


def main():
    model, tokenizer = load(MODEL)
    eos_ids = set(tokenizer.eos_token_ids or [tokenizer.eos_token_id])
    special_ids = set(tokenizer.all_special_ids)

    msgs = MESSAGES
    ids = canonical_ids(tokenizer, msgs)
    full_ids = render(tokenizer, msgs, True)  # + generation prompt
    assert full_ids[: len(ids)] == ids
    print(f"conversation: {len(ids)} canonical tokens")

    # tail = last user message (message 7)
    starts = message_token_starts(tokenizer, ids, len(msgs))
    tail_start_msg = 5
    tail_start_tok = starts[tail_start_msg]

    # Reference continuation from Arm A (greedy, 50 tokens).
    ca, _ = prefill(model, full_ids[:-1])
    snap_a_full = snapshot_cache(ca)
    la = first_step_logits(model, ca, full_ids[-1])
    cont, _ = greedy_generate(model, ca, la, 50, eos_ids)
    print("A cont:", tokenizer.decode(cont)[:90].replace("\n", " "))

    def build_and_score(mk):
        return tf_mean(model, None, full_ids[-1], cont, mk)

    lps_a = build_and_score(lambda: rebuild_cache(snap_a_full))

    # Summary over msgs (its final message is a user turn already).
    summ = generate_summary(model, tokenizer, msgs)
    print("S:", summ["text"][:120].replace("\n", " "), "...")

    # ---- Arm B ----
    b_msgs = build_b_messages(msgs, summ["text"], tail_start_msg)
    b_ids_canon = canonical_ids(tokenizer, b_msgs)
    b_ids = render(tokenizer, b_msgs, True)
    assert b_ids[: len(b_ids_canon)] == b_ids_canon
    cb, _ = prefill(model, b_ids[:-1])
    snap_b = snapshot_cache(cb)
    lps_b = build_and_score(lambda: rebuild_cache(snap_b))

    # ---- L4: alignment coverage ----
    b_starts = message_token_starts(tokenizer, b_ids_canon, len(b_msgs))
    regions = [
        ((b_starts[2], len(b_ids_canon)), (tail_start_tok, summ["conv_end"])),
        ((b_starts[1], b_starts[2]), (summ["s_start"], summ["s_end"])),
    ]
    pairs = build_alignment(b_ids[:-1], summ["old_ids"], special_ids, regions)
    tail_span_new = len(b_ids) - 1 - b_starts[2]
    s_len = len(summ["gen_ids"])
    covered_new = {n for n, _ in pairs}
    tail_covered = sum(1 for p in range(b_starts[2], len(b_ids_canon)) if p in covered_new)
    print(f"L4: pairs={len(pairs)}; tail tokens {tail_span_new}, covered {tail_covered} "
          f"(specials excluded); S tokens {s_len}")

    # ---- L3a: E alpha=0 == B ----
    e0 = arm_e_snapshot(snap_b, summ["snapshot"], pairs, alpha=0.0)
    lps_e0 = build_and_score(lambda: rebuild_cache(e0))
    d3a = max(abs(x - y) for x, y in zip(lps_b, lps_e0))
    print(f"L3a (E a=0 == B): max|dlp| = {d3a}")

    # ---- L3b: E alpha=1, old==new, no compaction == A ----
    pairs_id = build_alignment(
        full_ids[:-1], full_ids[:-1], special_ids,
        [((0, len(full_ids) - 1), (0, len(full_ids) - 1))],
    )
    e1 = arm_e_snapshot(snap_a_full, snap_a_full, pairs_id, alpha=1.0)
    lps_e1 = build_and_score(lambda: rebuild_cache(e1))
    d3b = max(abs(x - y) for x, y in zip(lps_a, lps_e1))
    print(f"L3b (E a=1 identity == A): max|dlp| = {d3b}")

    # ---- smoke: real arms ----
    lps_e_post = build_and_score(lambda: rebuild_cache(
        arm_e_snapshot(snap_b, summ["snapshot"], pairs, alpha=1.0)))

    # Arm C / D: gapped caches + generation prompt appended on top.
    def mk_c(include_summary):
        def f():
            c = gapped_cache_from(arm_c_snapshot(summ, tail_start_tok, include_summary))
            # append the generation-prompt suffix (everything after canonical
            # conv end in full_ids), minus the final token (protocol).
            from kvlib import extend_cache
            extend_cache(model, c, full_ids[len(ids):-1])
            return c
        return f

    lps_c = build_and_score(mk_c(True))
    lps_d = build_and_score(mk_c(False))

    def m(x):
        return sum(x) / len(x)

    print(f"\nteacher-forced mean logprob of A's continuation:")
    print(f"  A       {m(lps_a):8.4f}")
    print(f"  B       {m(lps_b):8.4f}")
    print(f"  C       {m(lps_c):8.4f}")
    print(f"  D       {m(lps_d):8.4f}")
    print(f"  E a=1   {m(lps_e_post):8.4f}")

    ok = d3a == 0.0 and d3b == 0.0
    print("L3 PASS" if ok else "L3 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
