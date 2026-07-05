"""Stage 1b: LongMemEval STANDARD protocol under compaction.

Unlike run_lme_hf (subsampled controlled contrasts), this uses the FULL
haystack (all ~50 sessions, chronological, ~100-130K tokens) — the
benchmark's real setting. Arms:

  A          : full haystack (standard-protocol baseline; citable)
  B          : production compaction — keep the most recent sessions
               (~TAIL_TOKENS) + terse summary of everything older, re-encoded
  B-min-pack : sinks + summary tokens fresh
  H-pack     : sinks + summary's write-time entries, packed
  E-tuned    : B + value graft at alpha (aligned tail + summary twins)

Judging later uses the benchmark's own evaluation prompt text (judge model =
Sonnet; deviation noted). Env: SC_HF_MODEL, SC_LME_TAG (default 30b_full),
SC_LME_N (default 100), SC_SHARD, SC_E_ALPHA, SC_LME_DATA.
Output: results/longmemeval_full_<tag>/<qid>.json
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
    arm_h_pack_snapshot_hf,
    generate_summary_hf,
    hf_prefill_ids,
    rope_base,
)
from kvlib_hf import blend_values, rebuild_cache

MODEL = os.environ.get("SC_HF_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
TAG = os.environ.get("SC_LME_TAG", "30b_full")
N_QUESTIONS = int(os.environ.get("SC_LME_N", "100"))
E_ALPHA = float(os.environ.get("SC_E_ALPHA", "0.75"))
TAIL_TOKENS = 12000
MAX_FULL = 200000
_shard = os.environ.get("SC_SHARD", "0/1")
SHARD_K, SHARD_N = (int(x) for x in _shard.split("/"))
DATA = os.environ.get("SC_LME_DATA", "longmemeval_s_cleaned.json")


def session_msgs(session):
    return [{"role": t["role"], "content": t["content"]} for t in session
            if t.get("role") in ("user", "assistant") and t.get("content")]


def build_full(q, tokenizer):
    """Full chronological haystack; tail boundary = message whose start is
    nearest (len - TAIL_TOKENS)."""
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    ev_sids = set(q["answer_session_ids"])
    ev_msg_range = None
    for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
        m = session_msgs(sess)
        if sid in ev_sids and m:
            ev_msg_range = (len(msgs), len(msgs) + len(m))
        msgs += m
    ids = canonical_ids(tokenizer, msgs, renderer=render_hf)
    if len(ids) > MAX_FULL or ev_msg_range is None:
        return None
    starts = message_token_starts(tokenizer, ids, len(msgs))
    target = len(ids) - TAIL_TOKENS
    if target <= 0:
        return None
    tail_start_msg = min(range(1, len(msgs)),
                         key=lambda i: abs(starts[i] - target))
    # evidence must be evicted for the interesting case; keep item either way
    ev_in_tail = ev_msg_range[1] > tail_start_msg
    return msgs, tail_start_msg, {
        "n_tokens": len(ids), "tail_start_msg": tail_start_msg,
        "evidence_in_tail": ev_in_tail,
    }


def probe_suffix(tokenizer, msgs, probe):
    full = render_hf(tokenizer, msgs + [{"role": "user", "content": probe}],
                     True)
    canon = canonical_ids(tokenizer, msgs, renderer=render_hf)
    assert full[: len(canon)] == canon
    return full[len(canon):]


_PACK_FRAME = {}


def packed_suffix(tokenizer, probe):
    if not _PACK_FRAME:
        m = [{"role": "system", "content": "x"}]
        a = render_hf(tokenizer, m + [{"role": "user", "content": "\x00Q\x00"}], True)
        b = render_hf(tokenizer, m, False)
        sfx = a[len(b):]
        mark = tokenizer("\x00Q\x00", add_special_tokens=False).input_ids
        for i in range(len(sfx) - len(mark) + 1):
            if sfx[i:i + len(mark)] == mark:
                _PACK_FRAME["pre"], _PACK_FRAME["post"] = sfx[:i], sfx[i + len(mark):]
                break
    return (_PACK_FRAME["pre"]
            + tokenizer(probe, add_special_tokens=False).input_ids
            + _PACK_FRAME["post"])


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa")
    model.eval()
    base = rope_base(model)

    data = json.load(open(DATA))
    rng = random.Random(43)
    pool = list(data)
    rng.shuffle(pool)
    pool = pool[SHARD_K::SHARD_N]
    outdir = Path(f"results/longmemeval_full_{TAG}")
    outdir.mkdir(parents=True, exist_ok=True)

    done = 0
    for q in pool:
        if done >= N_QUESTIONS:
            break
        outfile = outdir / f"{q['question_id']}.json"
        if outfile.exists():
            try:
                json.load(open(outfile)); done += 1; continue
            except Exception:
                outfile.unlink()
        built = build_full(q, tokenizer)
        if built is None:
            continue
        msgs, tail_start_msg, meta = built
        t0 = time.time()
        # summary of the EVICTED region only (older sessions) — generated
        # in-context at the end of the full conversation, per our method
        try:
            summary = generate_summary_hf(model, tokenizer, msgs,
                                          request=SUMMARY_REQUEST_BRIEF,
                                          max_tokens=400)
        except (AssertionError, torch.cuda.OutOfMemoryError) as e:
            print(f"{q['question_id']}: failed ({type(e).__name__})", flush=True)
            torch.cuda.empty_cache()
            continue
        conv_ids = canonical_ids(tokenizer, msgs, renderer=render_hf)
        b_msgs = build_b_messages(msgs, summary["text"], tail_start_msg)
        b_ids = canonical_ids(tokenizer, b_msgs, renderer=render_hf)
        s_leak = str(q["answer"]).lower() in summary["text"].lower()
        probe = q["question"]
        answers = {}

        a_snap, _ = hf_prefill_ids(model, conv_ids)
        answers["A"] = answer_hf(model, tokenizer, a_snap,
                                 probe_suffix(tokenizer, msgs, probe),
                                 len(conv_ids))
        del a_snap; torch.cuda.empty_cache()
        b_snap, _ = hf_prefill_ids(model, b_ids)
        answers["B"] = answer_hf(model, tokenizer, b_snap,
                                 probe_suffix(tokenizer, b_msgs, probe),
                                 len(b_ids))
        starts = message_token_starts(tokenizer, conv_ids, len(msgs))
        b_starts = message_token_starts(tokenizer, b_ids, len(b_msgs))
        regions = [
            ((b_starts[2], len(b_ids)), (starts[tail_start_msg],
                                         summary["conv_end"])),
            ((b_starts[1], b_starts[2]), (summary["s_start"],
                                          summary["s_end"])),
        ]
        pairs = build_alignment(b_ids, summary["old_ids"],
                                set(tokenizer.all_special_ids), regions)
        e_snap = blend_values(b_snap, summary["snapshot"], pairs, E_ALPHA)
        answers["E-tuned"] = answer_hf(model, tokenizer, e_snap,
                                       probe_suffix(tokenizer, b_msgs, probe),
                                       len(b_ids))
        del b_snap, e_snap; torch.cuda.empty_cache()
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
        del hp, summary; torch.cuda.empty_cache()

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
              f"{meta['n_tokens']} tok, ev_tail={meta['evidence_in_tail']}, "
              f"leak={s_leak}) done in {time.time()-t0:.0f}s "
              f"[{done}/{N_QUESTIONS}]", flush=True)


if __name__ == "__main__":
    main()
