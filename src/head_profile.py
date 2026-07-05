"""Per-(layer, head) graft profile at 4B (MLX, local). DIAGNOSTIC.

For each VAL conversation: blend old V into fresh at alpha=1 for exactly ONE
(layer, kv-head) slot at a time; teacher-force the held-out continuation.
288 slots x 10 conversations. Output: results/head_profile_4b/<conv>.json
"""

import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "src")
from alpha_sweep import score
from arms import canonical_ids, render
from kvlib import prefill, snapshot_cache
from run_arms import ArmSet, clear

MODEL = os.environ.get("SC_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
VAL = ["c01", "c02", "c03", "c04", "c05", "c06", "n01", "n02", "n03", "n04"]


def blend_slot(b_snap, old_snap, pairs, layer, head, alpha=1.0):
    new_idx = mx.array([n for n, _ in pairs])
    old_idx = mx.array([o for _, o in pairs])
    out = []
    for li, (k, v, off) in enumerate(b_snap):
        if li != layer:
            out.append((k, v, off))
            continue
        v2 = mx.array(v)
        vf = v2[:, head:head+1, new_idx, :].astype(mx.float32)
        vo = old_snap[li][1][:, head:head+1, old_idx, :].astype(mx.float32)
        v2[:, head:head+1, new_idx, :] = ((1 - alpha) * vf + alpha * vo).astype(v.dtype)
        out.append((k, v2, off))
    return out


def main():
    model, tokenizer = load(MODEL)
    n_layers = len(model.layers)
    n_kv = model.args.num_key_value_heads
    outdir = Path("results/head_profile_4b")
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for p in sorted(Path("data/synthetic").glob("c*.json")):
        c = json.load(open(p))
        jobs[c["id"]] = (c["messages"][:-1], c["sections"]["middle_end_msg"],
                         ("gp", c["messages"][-1]["content"]))
    for p in sorted(Path("data/natural").glob("n*.json")):
        c = json.load(open(p))
        msgs = c["messages"][: len(c["messages"]) - c["holdout_msgs"]]
        jobs[c["id"]] = (msgs, None, ("suffix", c["messages"]))

    for cid in VAL:
        outfile = outdir / f"{cid}.json"
        if outfile.exists():
            print(f"{cid}: done"); continue
        msgs, tsm, cont_spec = jobs[cid]
        t0 = time.time()
        aset = ArmSet(model, tokenizer, msgs, tsm)
        kind, payload = cont_spec
        if kind == "gp":
            gp = render(tokenizer, msgs, True)
            tgt = tokenizer.encode(payload, add_special_tokens=False)
            feed, targets = gp[len(aset.ids):] + tgt[:-1], tgt
        else:
            fullc = canonical_ids(tokenizer, payload)
            sfx = fullc[len(aset.ids):]
            feed, targets = sfx[:-1], sfx[1:]
        b_snap, old, pairs = aset.b_snap(), aset.summary["snapshot"], aset.pairs
        out = {"B": score(model, tokenizer, b_snap, feed, targets)}
        for l in range(n_layers):
            for h in range(n_kv):
                out[f"L{l}H{h}"] = score(
                    model, tokenizer, blend_slot(b_snap, old, pairs, l, h),
                    feed, targets)
        json.dump(out, open(outfile, "w"), indent=1)
        best = max((v, k) for k, v in out.items() if k != "B")
        print(f"== {cid} done in {time.time()-t0:.0f}s B={out['B']:.4f} "
              f"best={best[1]}:{best[0]:.4f}", flush=True)
        clear(aset, b_snap, old)


if __name__ == "__main__":
    main()
