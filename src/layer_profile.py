"""Per-layer graft profile (DIAGNOSTIC, not tuning).

For each conversation: build the Arm B context once, then for each single
layer l apply ValueGraft at alpha=1 to layer l ONLY and teacher-force the
held-out continuation. Output: per-layer delta vs B for every conversation.

No selection or tuned claims are made from this — it is a mechanism profile
("where in the depth does graftable signal live"), reported whole, per
per-head-alpha-speculation.md's cautions.

Output: results/layer_profile_<tag>/<conv>.json  {layer -> mean logprob}
"""

import json
import os
import sys
import time
from pathlib import Path

from mlx_lm import load

sys.path.insert(0, "src")
from alpha_sweep import score  # (model, tokenizer, snap, feed, targets)
from arms import arm_e_snapshot, canonical_ids, render
from kvlib import prefill, snapshot_cache
from run_arms import ArmSet, clear

MODEL = os.environ.get("SC_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
TAG = os.environ.get("SC_PROF_TAG", "4b")
ALPHA = 1.0


def main():
    model, tokenizer = load(MODEL)
    n_layers = len(model.layers)
    outdir = Path(f"results/layer_profile_{TAG}")
    outdir.mkdir(parents=True, exist_ok=True)

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

    for cid, msgs, tsm, cont_spec in jobs:
        outfile = outdir / f"{cid}.json"
        if outfile.exists():
            print(f"{cid}: done, skipping")
            continue
        t0 = time.time()
        aset = ArmSet(model, tokenizer, msgs, tsm)
        kind, payload = cont_spec
        if kind == "gp":
            gp = render(tokenizer, msgs, True)
            assert gp[: len(aset.ids)] == aset.ids
            cont_ids = tokenizer.encode(payload, add_special_tokens=False)
            feed, targets = gp[len(aset.ids):] + cont_ids[:-1], cont_ids
        else:
            full = canonical_ids(tokenizer, payload)
            assert full[: len(aset.ids)] == aset.ids
            sfx = full[len(aset.ids):]
            feed, targets = sfx[:-1], sfx[1:]

        b_snap, old_snap, pairs = (aset.b_snap(), aset.summary["snapshot"],
                                   aset.pairs)
        out = {"B": score(model, tokenizer, b_snap, feed, targets)}
        for l in range(n_layers):
            snap = arm_e_snapshot(b_snap, old_snap, pairs, ALPHA,
                                  layer_set={l})
            out[f"L{l}"] = score(model, tokenizer, snap, feed, targets)
        json.dump(out, open(outfile, "w"), indent=1)
        best_l = max((v, k) for k, v in out.items() if k != "B")
        print(f"== {cid} done in {time.time()-t0:.0f}s  B={out['B']:.4f} "
              f"best {best_l[1]}={best_l[0]:.4f}")
        clear(aset, b_snap, old_snap)


if __name__ == "__main__":
    main()
