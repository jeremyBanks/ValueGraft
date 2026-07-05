"""LongMemEval runner for the HF/CUDA pod. Mirrors run_longmemeval.py.

Arms: A, B, B-min-pack, H-pack, H-gap, E-tuned (global alpha via SC_E_ALPHA,
default 0.75 per the 30B sweep).

Env: SC_HF_MODEL (default Qwen/Qwen3-30B-A3B-Instruct-2507), SC_LME_TAG,
SC_LME_N, SC_LME_TYPES (comma list; default the 3 single-evidence types;
"all" includes multi-session + temporal), SC_LME_DATA (path to
longmemeval_s_cleaned.json).

Output: results/longmemeval_<tag>/<qid>.json (same schema as local).
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, "src")
from arms_common import (
    SUMMARY_REQUEST_BRIEF,
    build_alignment,
    build_b_messages,
    bmin_pack_ids,
    canonical_ids,
    message_token_starts,
    render_hf,
)
from arms_hf import (
    answer_hf,
    arm_h_gap_snapshot_hf,
    arm_h_pack_snapshot_hf,
    generate_summary_hf,
    hf_prefill_ids,
    rope_base,
    to_ids,
)
from kvlib_hf import blend_values, prefill, snapshot_cache

MODEL = os.environ.get("SC_HF_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
TAG = os.environ.get("SC_LME_TAG", "30b_bf16")
N_QUESTIONS = int(os.environ.get("SC_LME_N", "500"))
E_ALPHA = float(os.environ.get("SC_E_ALPHA", "0.75"))
TOKEN_TARGET = 12000
_types = os.environ.get("SC_LME_TYPES", "")
TYPES = (None if _types == "all" else
         tuple(_types.split(",")) if _types else
         ("single-session-user", "single-session-assistant",
          "knowledge-update"))
DATA = os.environ.get("SC_LME_DATA", "data/longmemeval_s_cleaned.json")
# SC_SHARD="k/N": process only items where pool_index % N == k (after the
# fixed-seed shuffle, so shards are deterministic and disjoint)
_shard = os.environ.get("SC_SHARD", "0/1")
SHARD_K, SHARD_N = (int(x) for x in _shard.split("/"))


def session_msgs(session):
    return [{"role": t["role"], "content": t["content"]} for t in session
            if t.get("role") in ("user", "assistant") and t.get("content")]


def build_conversation(q, tokenizer, rng):
    sess_by_id = dict(zip(q["haystack_session_ids"], q["haystack_sessions"]))
    ev_ids = [s for s in q["answer_session_ids"] if s in sess_by_id]
    if not ev_ids:
        return None
    distractors = [sid for sid in q["haystack_session_ids"]
                   if sid not in q["answer_session_ids"]]
    rng.shuffle(distractors)

    def tok_len(msgs):
        return len(canonical_ids(tokenizer, msgs, renderer=render_hf))

    system = [{"role": "system", "content": "You are a helpful assistant."}]
    ev_msgs = []
    for sid in ev_ids:
        ev_msgs += session_msgs(sess_by_id[sid])
    pre, post = [], []
    for sid in distractors:
        m = session_msgs(sess_by_id[sid])
        if not m:
            continue
        (pre if len(pre) <= len(post) else post).append(m)
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
    ids = canonical_ids(tokenizer, msgs, renderer=render_hf)
    starts = message_token_starts(tokenizer, ids, len(msgs))
    ev_end_msg = 1 + sum(map(len, pre)) + len(ev_msgs)
    target = 0.75 * len(ids)
    cands = [i for i in range(ev_end_msg, len(msgs))
             if starts[i] >= target] or [max(ev_end_msg, len(msgs) - 4)]
    tail_start_msg = cands[0]
    if tail_start_msg <= ev_end_msg - 1 or tail_start_msg >= len(msgs):
        return None
    return msgs, tail_start_msg, {"n_tokens": n_tok,
                                  "tail_start_msg": tail_start_msg}


def probe_suffix(tokenizer, msgs, probe):
    full = render_hf(tokenizer,
                     msgs + [{"role": "user", "content": probe}], True)
    canon = canonical_ids(tokenizer, msgs, renderer=render_hf)
    assert full[: len(canon)] == canon
    return full[len(canon):]


_PACK_FRAME = {}


def packed_suffix(tokenizer, probe):
    if not _PACK_FRAME:
        msgs = [{"role": "system", "content": "x"}]
        a = render_hf(tokenizer,
                      msgs + [{"role": "user", "content": "\x00Q\x00"}], True)
        b = render_hf(tokenizer, msgs, False)
        sfx = a[len(b):]
        marker = tokenizer("\x00Q\x00", add_special_tokens=False).input_ids
        for i in range(len(sfx) - len(marker) + 1):
            if sfx[i:i + len(marker)] == marker:
                _PACK_FRAME["pre"], _PACK_FRAME["post"] = sfx[:i], sfx[i + len(marker):]
                break
        assert "pre" in _PACK_FRAME
    return (_PACK_FRAME["pre"]
            + tokenizer(probe, add_special_tokens=False).input_ids
            + _PACK_FRAME["post"])


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    base = rope_base(model)
    n_layers = model.config.num_hidden_layers

    data = json.load(open(DATA))
    rng = random.Random(42)
    pool = [q for q in data if TYPES is None or q["question_type"] in TYPES]
    rng.shuffle(pool)
    pool = pool[SHARD_K::SHARD_N]
    outdir = Path(f"results/longmemeval_{TAG}")
    outdir.mkdir(parents=True, exist_ok=True)

    done = 0
    for q in pool:
        if done >= N_QUESTIONS:
            break
        outfile = outdir / f"{q['question_id']}.json"
        if outfile.exists():
            try:  # validate resumable artifact; drop truncated files
                json.load(open(outfile))
                done += 1
                continue
            except Exception:
                outfile.unlink()
        built = build_conversation(q, tokenizer, random.Random(q["question_id"]))
        if built is None:
            continue
        msgs, tail_start_msg, meta = built
        t0 = time.time()
        try:
            summary = generate_summary_hf(model, tokenizer, msgs,
                                          request=SUMMARY_REQUEST_BRIEF)
        except AssertionError as e:
            print(f"{q['question_id']}: build failed ({e})", flush=True)
            continue
        conv_ids = canonical_ids(tokenizer, msgs, renderer=render_hf)
        b_msgs = build_b_messages(msgs, summary["text"], tail_start_msg)
        b_ids = canonical_ids(tokenizer, b_msgs, renderer=render_hf)
        s_leak = str(q["answer"]).lower() in summary["text"].lower()
        probe = q["question"]
        answers = {}

        # A
        a_snap, _ = hf_prefill_ids(model, conv_ids)
        answers["A"] = answer_hf(model, tokenizer, a_snap,
                                 probe_suffix(tokenizer, msgs, probe),
                                 len(conv_ids))
        del a_snap
        # B (+ alignment for E)
        b_snap, _ = hf_prefill_ids(model, b_ids)
        answers["B"] = answer_hf(model, tokenizer, b_snap,
                                 probe_suffix(tokenizer, b_msgs, probe),
                                 len(b_ids))
        b_starts = message_token_starts(tokenizer, b_ids, len(b_msgs))
        starts = message_token_starts(tokenizer, conv_ids, len(msgs))
        regions = [
            ((b_starts[2], len(b_ids)), (starts[tail_start_msg],
                                         summary["conv_end"])),
            ((b_starts[1], b_starts[2]), (summary["s_start"],
                                          summary["s_end"])),
        ]
        special_ids = set(tokenizer.all_special_ids)
        pairs = build_alignment(b_ids, summary["old_ids"], special_ids,
                                regions)
        e_snap = blend_values(b_snap, summary["snapshot"], pairs, E_ALPHA)
        answers["E-tuned"] = answer_hf(model, tokenizer, e_snap,
                                       probe_suffix(tokenizer, b_msgs, probe),
                                       len(b_ids))
        del b_snap, e_snap
        # packed arms
        bmp = bmin_pack_ids(summary, conv_ids)
        bmp_snap, _ = hf_prefill_ids(model, bmp)
        answers["B-min-pack"] = answer_hf(model, tokenizer, bmp_snap,
                                          packed_suffix(tokenizer, probe),
                                          len(bmp))
        del bmp_snap
        hp = arm_h_pack_snapshot_hf(summary, base)
        answers["H-pack"] = answer_hf(model, tokenizer, hp,
                                      packed_suffix(tokenizer, probe),
                                      hp[0][0].shape[2])
        del hp
        hg = arm_h_gap_snapshot_hf(summary)
        answers["H-gap"] = answer_hf(model, tokenizer, hg,
                                     packed_suffix(tokenizer, probe),
                                     summary["s_end"])
        del hg, summary

        tmp = outfile.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"question_id": q["question_id"], "question": probe,
                       "answer": str(q["answer"]),
                       "question_type": q["question_type"], "model": MODEL,
                       "meta": meta, "s_leak": s_leak, "arms": answers},
                      f, indent=1, ensure_ascii=False)
        tmp.rename(outfile)
        done += 1
        print(f"== {q['question_id']} ({q['question_type']}, "
              f"{meta['n_tokens']} tok, leak={s_leak}) done in "
              f"{time.time()-t0:.0f}s [{done}/{N_QUESTIONS}]", flush=True)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
