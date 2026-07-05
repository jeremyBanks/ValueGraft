"""LongMemEval-S evaluation of compaction arms.

Per question: build a conversation from the evidence session(s) + distractor
sessions (deterministic per-question sampling) sized ~TOKEN_TARGET, with the
evidence strictly inside the evicted region (early-middle) and distractor
sessions as the retained tail. Append the question as the final user turn
under each arm and record the greedy answer.

Arms: A (full), B (summary+tail), B-min-pack, H-pack, E-tuned (ValueGraft at
the scale-tuned setting), H-gap. Summary = terse (shadow) request, matching
the Phase-2 configuration.

Leakage audit: flags whether the gold answer string appears in the summary.

Output: results/longmemeval_<tag>/<qid>.json
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "src")
from arms import (
    SUMMARY_REQUEST_BRIEF,
    arm_c_snapshot,
    arm_e_snapshot,
    arm_h_pack_snapshot,
    bmin_pack_ids,
    build_b_messages,
    canonical_ids,
    gapped_cache_from,
    generate_summary,
    message_token_starts,
    render,
)
from kvlib import (
    extend_cache,
    first_step_logits,
    greedy_generate,
    prefill,
    rebuild_cache,
    snapshot_cache,
)
from run_arms import ArmSet, clear
from run_phase2 import packed_probe_suffix, probe_suffix_ids, answer

MODEL = os.environ.get("SC_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
TAG = os.environ.get("SC_LME_TAG", "4b")
N_QUESTIONS = int(os.environ.get("SC_LME_N", "48"))
TOKEN_TARGET = 12000
TYPES = ("single-session-user", "single-session-assistant", "knowledge-update")
# tuned ValueGraft settings per scale (from the alpha sweeps)
TUNED = {"4b": ("midband", 0.25), "30b": ("global", 0.75)}

DATA = ("/Users/jeb/.cache/huggingface/hub/datasets--xiaowu0162--"
        "longmemeval-cleaned/snapshots/98d7416c24c778c2fee6e6f3006e7a073259d48f/"
        "longmemeval_s_cleaned.json")


def session_msgs(session):
    return [{"role": t["role"], "content": t["content"]} for t in session
            if t.get("role") in ("user", "assistant") and t.get("content")]


def build_conversation(q, tokenizer, rng):
    """Return (msgs, tail_start_msg, evidence_span_ok) or None if unbuildable."""
    sess_by_id = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    ev_ids = [s for s in q["answer_session_ids"] if s in sess_by_id]
    if not ev_ids:
        return None
    distractors = [sid for sid in q["haystack_session_ids"]
                   if sid not in q["answer_session_ids"]]
    rng.shuffle(distractors)

    def tok_len(msgs):
        return len(canonical_ids(tokenizer, msgs))

    system = [{"role": "system", "content": "You are a helpful assistant."}]
    ev_msgs = []
    for sid in ev_ids:
        ev_msgs += session_msgs(sess_by_id[sid])

    # assemble: pre-distractors | evidence | mid-distractors | TAIL distractors
    # target: evidence ends before the 65% token mark; tail is last ~25%.
    parts_msgs = [list(ev_msgs)]  # start with evidence, prepend/append around
    pre, post = [], []
    for sid in distractors:
        m = session_msgs(sess_by_id[sid])
        if not m:
            continue
        (pre if len(pre) <= len(post) else post).append(m)
        total = sum(map(len, pre)) + len(ev_msgs) + sum(map(len, post))
        msgs = system + [x for s_ in pre for x in s_] + ev_msgs + \
               [x for s_ in post for x in s_]
        if tok_len(msgs) >= TOKEN_TARGET:
            break
    msgs = system + [x for s_ in pre for x in s_] + ev_msgs + \
           [x for s_ in post for x in s_]
    if not post:
        return None
    n_tok = tok_len(msgs)
    if n_tok < 6000 or n_tok > 16000:
        return None
    # tail = last few distractor sessions (whole sessions, >= ~20% of tokens)
    ids = canonical_ids(tokenizer, msgs)
    starts = message_token_starts(tokenizer, ids, len(msgs))
    ev_start_msg = 1 + sum(map(len, pre))
    ev_end_msg = ev_start_msg + len(ev_msgs)
    # choose tail_start_msg: first message boundary after 75% mark that is
    # also strictly after the evidence
    target = 0.75 * len(ids)
    cands = [i for i in range(ev_end_msg, len(msgs))
             if starts[i] >= target] or [max(ev_end_msg, len(msgs) - 4)]
    tail_start_msg = cands[0]
    if tail_start_msg <= ev_end_msg - 1 or tail_start_msg >= len(msgs):
        return None
    evict_frac = (starts[tail_start_msg] - starts[ev_end_msg - 1]) / len(ids)
    return msgs, tail_start_msg, {
        "n_tokens": n_tok, "ev_start_msg": ev_start_msg,
        "ev_end_msg": ev_end_msg, "tail_start_msg": tail_start_msg,
        "tail_frac": 1 - starts[tail_start_msg] / len(ids),
    }


def main():
    model, tokenizer = load(MODEL)
    eos_ids = set(tokenizer.eos_token_ids or [tokenizer.eos_token_id])
    rope_base = model.args.rope_theta
    gate, alpha = TUNED[TAG]
    n_layers = len(model.layers)
    layer_set = (set(range(n_layers // 3, 2 * n_layers // 3))
                 if gate == "midband" else None)

    data = json.load(open(DATA))
    rng = random.Random(42)
    pool = [q for q in data if q["question_type"] in TYPES]
    rng.shuffle(pool)
    outdir = Path(f"results/longmemeval_{TAG}")
    outdir.mkdir(parents=True, exist_ok=True)

    done = 0
    for q in pool:
        if done >= N_QUESTIONS:
            break
        outfile = outdir / f"{q['question_id']}.json"
        if outfile.exists():
            done += 1
            continue
        built = build_conversation(q, tokenizer, random.Random(q["question_id"]))
        if built is None:
            continue
        msgs, tail_start_msg, meta = built
        t0 = time.time()
        try:
            aset = ArmSet(model, tokenizer, msgs, tail_start_msg,
                          summary_request=SUMMARY_REQUEST_BRIEF)
            aset.prepare_a()
        except AssertionError as e:
            print(f"{q['question_id']}: build failed ({e}); skipping")
            continue
        s_leak = str(q["answer"]).lower() in aset.summary["text"].lower()

        e_snap = arm_e_snapshot(aset.b_snap(), aset.summary["snapshot"],
                                aset.pairs, alpha, layer_set=layer_set)
        hp_snap = arm_h_pack_snapshot(aset.summary, rope_base)
        bmp = bmin_pack_ids(aset.summary, aset.ids)
        bmp_snap = snapshot_cache(prefill(model, bmp)[0])

        probe = q["question"]
        arm_defs = [
            ("A", lambda: rebuild_cache(aset._a_snap), "render", msgs),
            ("B", lambda: rebuild_cache(aset.b_snap()), "render", aset.b_msgs),
            ("E-tuned", lambda: rebuild_cache(e_snap), "render", aset.b_msgs),
            ("B-min-pack", lambda: rebuild_cache(bmp_snap), "packed", None),
            ("H-pack", lambda: rebuild_cache(hp_snap), "packed", None),
            ("H-gap", lambda: gapped_cache_from(arm_c_snapshot(
                aset.summary, aset.summary["conv_end"], True)), "render", msgs),
        ]
        answers = {}
        for name, mk, mode, rmsgs in arm_defs:
            sfx = (probe_suffix_ids(tokenizer, rmsgs, probe) if mode == "render"
                   else packed_probe_suffix(tokenizer, probe))
            answers[name] = answer(model, tokenizer, mk(), sfx, eos_ids)
        json.dump({
            "question_id": q["question_id"], "question": probe,
            "answer": str(q["answer"]), "question_type": q["question_type"],
            "model": MODEL, "meta": meta, "s_leak": s_leak,
            "summary_text": aset.summary["text"], "arms": answers,
        }, open(outfile, "w"), indent=1, ensure_ascii=False)
        done += 1
        print(f"== {q['question_id']} ({q['question_type']}, "
              f"{meta['n_tokens']} tok, leak={s_leak}) done in "
              f"{time.time()-t0:.0f}s [{done}/{N_QUESTIONS}]")
        clear(aset, e_snap, hp_snap, bmp_snap)


if __name__ == "__main__":
    main()
