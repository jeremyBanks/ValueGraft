"""Stage 2: SWE-Gym/OpenHands coding-trace evaluation under compaction.

Per trajectory (filtered to fit budget): context = messages up to a cut at
~75% of tokens (cut lands before an assistant turn); target = that next
assistant action. Arms A / B / B-min-pack / H-pack / E-tuned built over the
context (evicting sinks..tail_start like the chat experiments; tail = last
~25% incl. the most recent tool results, per production practice).

Metrics per arm:
  tf: mean teacher-forced logprob of the true next action (primary)
  gen: greedy 200-token generation (for behavioral analysis offline):
       stored raw; plus auto flags — repeats-failed-command (a command line
       that previously produced an error/traceback in the EVICTED region).

Env: SC_HF_MODEL, SC_SWE_TAG, SC_SWE_N (default 75), SC_SHARD, SC_E_ALPHA.
Output: results/swegym_<tag>/<idx>.json
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, "src")
from arms_common import (
    SUMMARY_REQUEST_BRIEF,
    SUMMARY_REQUEST_PROD,
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
    to_ids,
)
from kvlib_hf import blend_values, rebuild_cache, tf_logprobs

MODEL = os.environ.get("SC_HF_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
TAG = os.environ.get("SC_SWE_TAG", "30b_bf16")
N_TRAJ = int(os.environ.get("SC_SWE_N", "75"))
E_ALPHA = float(os.environ.get("SC_E_ALPHA", "0.75"))
# Summary condition: SC_SUMMARY=brief -> terse mechanism-isolation summary (the
# condition the +0.0156 anchor result used); default -> SUMMARY_REQUEST_PROD, the
# production-faithful OpenHands-condenser-style summary (task/state/paths/decisions,
# 300-500w) — the std arm we now want as first-class enrichment (Fable's #1).
SUMM_REQ = SUMMARY_REQUEST_BRIEF if os.environ.get("SC_SUMMARY") == "brief" else SUMMARY_REQUEST_PROD
SUMM_TAG = "brief" if os.environ.get("SC_SUMMARY") == "brief" else "prod"
_shard = os.environ.get("SC_SHARD", "0/1")
SHARD_K, SHARD_N = (int(x) for x in _shard.split("/"))
PARQUET = os.environ.get("SC_SWE_DATA", "swegym.parquet")
MAX_TOK, MIN_TOK = 15000, 6000


def load_trajectories():
    import pandas as pd
    df = pd.read_parquet(PARQUET)
    return [list(r) for r in df["messages"]]


def find_cut(tokenizer, msgs):
    """Largest assistant-turn index whose prefix stays within budget and
    leaves >=15% of tokens after it; returns (ctx_msgs, target_msg, meta)."""
    ids = canonical_ids(tokenizer, msgs, renderer=render_hf)
    if not (MIN_TOK <= len(ids) <= MAX_TOK):
        return None
    starts = message_token_starts(tokenizer, ids, len(msgs))
    # candidate cuts: assistant turns in the 60-85% token range
    cands = [i for i in range(4, len(msgs))
             if msgs[i]["role"] == "assistant"
             and 0.60 * len(ids) <= starts[i] <= 0.85 * len(ids)]
    if not cands:
        return None
    cut = cands[len(cands) // 2]
    ctx = msgs[:cut]
    if ctx[-1]["role"] != "user":
        return None
    return ctx, msgs[cut], {"n_tokens_full": len(ids), "cut_msg": cut,
                            "n_ctx_tokens": starts[cut]}


ERR_PAT = re.compile(r"error|traceback|failed|exception", re.I)
CMD_PAT = re.compile(r"<execute_bash>\s*(.+?)\s*</execute_bash>", re.S)


def failed_commands(msgs, lo, hi):
    """Commands in msgs[lo:hi] whose following user turn shows an error."""
    out = set()
    for i in range(lo, min(hi, len(msgs) - 1)):
        if msgs[i]["role"] != "assistant":
            continue
        m = CMD_PAT.search(str(msgs[i]["content"]))
        if m and ERR_PAT.search(str(msgs[i + 1]["content"])[:2000]):
            out.add(m.group(1).strip().split("\n")[0][:120])
    return out


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    base = rope_base(model)

    trajs = load_trajectories()
    outdir = Path(f"results/swegym_{TAG}_{SUMM_TAG}")
    outdir.mkdir(parents=True, exist_ok=True)
    done = 0
    for idx in range(SHARD_K, len(trajs), SHARD_N):
        if done >= N_TRAJ:
            break
        outfile = outdir / f"t{idx:04d}.json"
        if outfile.exists():
            try:
                json.load(open(outfile)); done += 1; continue
            except Exception:
                outfile.unlink()
        msgs = trajs[idx]
        built = None
        try:
            built = find_cut(tokenizer, msgs)
        except AssertionError:
            pass
        if built is None:
            continue
        ctx, target, meta = built
        t0 = time.time()
        try:
            summary = generate_summary_hf(model, tokenizer, ctx,
                                          request=SUMM_REQ)
        except AssertionError as e:
            print(f"t{idx}: summary failed ({e})", flush=True); continue
        ctx_ids = canonical_ids(tokenizer, ctx, renderer=render_hf)
        starts = message_token_starts(tokenizer, ctx_ids, len(ctx))
        # tail = message boundary nearest 75% of ctx
        tgt75 = 0.75 * len(ctx_ids)
        tail_start_msg = min(range(1, len(ctx)),
                             key=lambda i: abs(starts[i] - tgt75))
        b_msgs = build_b_messages(ctx, summary["text"], tail_start_msg)
        b_ids = canonical_ids(tokenizer, b_msgs, renderer=render_hf)
        s_leak = False  # no single gold string here
        # target: teacher-force the rendered next assistant turn
        gp_full = render_hf(tokenizer, ctx, True)
        assert gp_full[: len(ctx_ids)] == ctx_ids
        gp_suffix = gp_full[len(ctx_ids):]
        tgt_ids = tokenizer(str(target["content"]),
                            add_special_tokens=False).input_ids[:600]
        failed_evicted = sorted(failed_commands(ctx, 1, tail_start_msg))

        res = {"arms": {}}

        def eval_arm(name, snap, next_pos):
            cache = rebuild_cache(snap, DynamicCache)
            feed = gp_suffix + tgt_ids[:-1]
            pos = torch.arange(next_pos, next_pos + len(feed),
                               device=model.device)[None]
            lps = tf_logprobs(model, cache, feed, tgt_ids, position_ids=pos)
            gen = answer_hf(model, tokenizer, snap, gp_suffix, next_pos,
                            max_tokens=200)
            m = CMD_PAT.search(gen)
            first_cmd = (m.group(1).strip().split("\n")[0][:120] if m else "")
            res["arms"][name] = {
                "tf_mean": sum(lps) / len(lps),
                "gen": gen[:500],
                "gen_first_cmd": first_cmd,
                "repeats_failed": first_cmd in failed_evicted if first_cmd
                                  else False,
            }

        # A
        a_snap, _ = hf_prefill_ids(model, ctx_ids)
        eval_arm("A", a_snap, len(ctx_ids)); del a_snap
        # B + E
        b_snap, _ = hf_prefill_ids(model, b_ids)
        eval_arm("B", b_snap, len(b_ids))
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
        eval_arm("E-tuned", e_snap, len(b_ids)); del b_snap, e_snap
        # packed arms (gp suffix belongs to the chat frame; packed contexts
        # still get the same generation-prompt tokens after their storage)
        bmp = bmin_pack_ids(summary, ctx_ids)
        bmp_snap, _ = hf_prefill_ids(model, bmp)
        eval_arm("B-min-pack", bmp_snap, len(bmp)); del bmp_snap
        hp = arm_h_pack_snapshot_hf(summary, base)
        eval_arm("H-pack", hp, hp[0][0].shape[2]); del hp

        tmp = outfile.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"idx": idx, "meta": meta, "model": MODEL, "dtype": "bfloat16",
                       "summary_tokens": len(summary["gen_ids"]),
                       "failed_evicted_commands": failed_evicted,
                       "n_target_tokens": len(tgt_ids), **res},
                      f, indent=1)
        tmp.rename(outfile)
        done += 1
        print(f"== t{idx:04d} done in {time.time()-t0:.0f}s "
              f"[{done}/{N_TRAJ}] A={res['arms']['A']['tf_mean']:.3f} "
              f"B={res['arms']['B']['tf_mean']:.3f}", flush=True)
        del summary
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
