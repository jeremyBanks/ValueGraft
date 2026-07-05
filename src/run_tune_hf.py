"""Stage T: on-pod tuning of ValueGraft for the actual bf16 model.

Phases (env SC_TUNE_PHASE):
  sweep   : global alpha grid on VAL conversations (CONT teacher-forcing)
  layer   : per-layer profile (alpha=1, one layer at a time) on VAL
  head    : per-KV-head-per-layer profile (alpha=1, one (layer, head) slot at
            a time) on VAL  [192 slots at 48x4]
  eval    : evaluate derived candidate rules ONCE on HOLDOUT
            (rules read from results/tune_rules.json, written by analysis)

Conversations: our synthetic (c*) + natural (n*) corpora, same VAL/HOLD split
as local sweeps (VAL: c01-c06,n01-n04; HOLD: c07-c12,n05-n08).

Output: results/tune_<phase>_<tag>/<conv>.json
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, "src")
from arms_common import (
    SUMMARY_REQUEST,
    build_alignment,
    build_b_messages,
    canonical_ids,
    message_token_starts,
    render_hf,
)
from arms_hf import generate_summary_hf, hf_prefill_ids, rope_base
from kvlib_hf import rebuild_cache, tf_logprobs

MODEL = os.environ.get("SC_HF_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
TAG = os.environ.get("SC_TUNE_TAG", "30b_bf16")
PHASE = os.environ.get("SC_TUNE_PHASE", "sweep")
ALPHAS = [float(x) for x in os.environ.get(
    "SC_ALPHAS", "0.25,0.5,0.75,1.0,1.25").split(",")]
VAL = ["c01", "c02", "c03", "c04", "c05", "c06", "n01", "n02", "n03", "n04"]
HOLD = ["c07", "c08", "c09", "c10", "c11", "c12", "n05", "n06", "n07", "n08"]


def blend(b_snap, old_snap, pairs, alpha, layer_set=None, head_map=None):
    """head_map: {layer_idx: [kv_head_idx,...]} restricting the blend."""
    new_idx = torch.tensor([n for n, _ in pairs])
    old_idx = torch.tensor([o for _, o in pairs])
    out = []
    for li, (k, v) in enumerate(b_snap):
        heads = None
        if head_map is not None:
            heads = head_map.get(li)
            if not heads:
                out.append((k, v)); continue
        elif layer_set is not None and li not in layer_set:
            out.append((k, v)); continue
        v2 = v.clone()
        vf = v2[..., new_idx, :].float()
        vo = old_snap[li][1][..., old_idx, :].float()
        mixed = ((1 - alpha) * vf + alpha * vo).to(v2.dtype)
        if heads is None:
            v2[..., new_idx, :] = mixed
        else:
            for h in heads:
                v2[:, h, new_idx, :] = mixed[:, h]
        out.append((k, v2))
    return out


def load_jobs():
    jobs = []
    for p in sorted(Path("data/synthetic").glob("c*.json")):
        c = json.load(open(p))
        jobs.append((c["id"], c["messages"][:-1],
                     c["sections"]["middle_end_msg"],
                     ("gp", c["messages"][-1]["content"])))
    for p in sorted(Path("data/natural").glob("n*.json")):
        c = json.load(open(p))
        msgs = c["messages"][: len(c["messages"]) - c["holdout_msgs"]]
        jobs.append((c["id"], msgs, None, ("suffix", c["messages"])))
    return {j[0]: j for j in jobs}


def build_ctx(model, tokenizer, cid, msgs, tsm, cont_spec):
    ids = canonical_ids(tokenizer, msgs, renderer=render_hf)
    starts = message_token_starts(tokenizer, ids, len(msgs))
    if tsm is None:
        target = 0.75 * len(ids)
        tsm = min(range(1, len(msgs)), key=lambda i: abs(starts[i] - target))
    summary = generate_summary_hf(model, tokenizer, msgs,
                                  request=SUMMARY_REQUEST)
    b_msgs = build_b_messages(msgs, summary["text"], tsm)
    b_ids = canonical_ids(tokenizer, b_msgs, renderer=render_hf)
    b_starts = message_token_starts(tokenizer, b_ids, len(b_msgs))
    regions = [
        ((b_starts[2], len(b_ids)), (starts[tsm], summary["conv_end"])),
        ((b_starts[1], b_starts[2]), (summary["s_start"], summary["s_end"])),
    ]
    pairs = build_alignment(b_ids, summary["old_ids"],
                            set(tokenizer.all_special_ids), regions)
    b_snap, _ = hf_prefill_ids(model, b_ids)
    kind, payload = cont_spec
    if kind == "gp":
        gp = render_hf(tokenizer, msgs, True)
        assert gp[: len(ids)] == ids
        tgt = tokenizer(payload, add_special_tokens=False).input_ids
        feed, targets = gp[len(ids):] + tgt[:-1], tgt
        next_pos = len(b_ids)
    else:
        fullc = canonical_ids(tokenizer, payload, renderer=render_hf)
        assert fullc[: len(ids)] == ids
        sfx = fullc[len(ids):]
        feed, targets = sfx[:-1], sfx[1:]
        next_pos = len(b_ids)
    return b_snap, summary, pairs, feed, targets, next_pos


def score(model, snap, feed, targets, next_pos):
    cache = rebuild_cache(snap, DynamicCache)
    pos = torch.arange(next_pos, next_pos + len(feed),
                       device=model.device)[None]
    lps = tf_logprobs(model, cache, feed, targets, position_ids=pos)
    return sum(lps) / len(lps)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    jobs = load_jobs()
    convs = VAL if PHASE in ("sweep", "layer", "head") else HOLD
    outdir = Path(f"results/tune_{PHASE}_{TAG}")
    outdir.mkdir(parents=True, exist_ok=True)
    rules = (json.load(open("tune_rules.json"))
             if PHASE == "eval" else None)

    for cid in convs:
        outfile = outdir / f"{cid}.json"
        if outfile.exists():
            print(f"{cid}: done"); continue
        t0 = time.time()
        b_snap, summary, pairs, feed, targets, next_pos = build_ctx(
            model, tokenizer, *jobs[cid])
        out = {"B": score(model, b_snap, feed, targets, next_pos)}
        old = summary["snapshot"]
        if PHASE == "sweep":
            for a in ALPHAS:
                out[f"a{a:g}"] = score(model, blend(b_snap, old, pairs, a),
                                       feed, targets, next_pos)
        elif PHASE == "layer":
            for l in range(n_layers):
                out[f"L{l}"] = score(
                    model, blend(b_snap, old, pairs, 1.0, layer_set={l}),
                    feed, targets, next_pos)
        elif PHASE == "head":
            for l in range(n_layers):
                for h in range(n_kv):
                    out[f"L{l}H{h}"] = score(
                        model, blend(b_snap, old, pairs, 1.0,
                                     head_map={l: [h]}),
                        feed, targets, next_pos)
        elif PHASE == "eval":
            for name, rule in rules.items():
                out[name] = score(
                    model, blend(b_snap, old, pairs, rule["alpha"],
                                 layer_set=(set(rule["layers"])
                                            if rule.get("layers") else None),
                                 head_map=({int(k): v for k, v in
                                            rule["head_map"].items()}
                                           if rule.get("head_map") else None)),
                    feed, targets, next_pos)
        tmp = outfile.with_suffix(".tmp")
        json.dump(out, open(tmp, "w"), indent=1)
        tmp.rename(outfile)
        del b_snap, old, summary
        torch.cuda.empty_cache()
        best = max((v, k) for k, v in out.items() if k != "B")
        print(f"== {cid} done in {time.time()-t0:.0f}s B={out['B']:.4f} "
              f"best={best[1]}:{best[0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
