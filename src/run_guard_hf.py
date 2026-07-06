"""Pre-registered contamination guard for the 57-slot mask (DECISIONS):
wrong-conversation values THROUGH THE SAME SLOTS must not help.
For each HOLD conv: blend posslots at a=1 with old-values taken from the
PREVIOUS holdout conv's snapshot (positions clamped to donor length).
Output: results/guard_posslots_30b_bf16/<conv>.json {B, guard}
"""

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
os.environ.setdefault("SC_TUNE_PHASE", "guard")
import run_tune_hf as T
from kvlib_hf import blend_values

HOLD = T.HOLD


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(T.MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        T.MODEL, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    head_map = {int(k): v for k, v in json.load(
        open("tune_configs.json"))["posslots"]["head_map"].items()}
    jobs = T.load_jobs()
    outdir = Path("results/guard_posslots_30b_bf16")
    outdir.mkdir(parents=True, exist_ok=True)
    prev = None  # (snapshot, its length)
    ctxs = {}
    for cid in HOLD:
        ctxs[cid] = T.build_ctx(model, tokenizer, *jobs[cid])
        b_snap, summary, pairs, feed, targets, next_pos = ctxs[cid]
        out = {"B": T.score(model, b_snap, feed, targets, next_pos)}
        if prev is not None:
            donor, dlen = prev
            wpairs = [(n, min(o, dlen - 1)) for n, o in pairs]
            g = blend_values(b_snap, donor, wpairs, 1.0, head_map=head_map)
            out["guard"] = T.score(model, g, feed, targets, next_pos)
            del g
        json.dump(out, open(outdir / f"{cid}.json", "w"), indent=1)
        prev = (summary["snapshot"], summary["snapshot"][0][0].shape[2])
        print(f"== {cid}: {out}", flush=True)
        del b_snap
        torch.cuda.empty_cache()
    print("GUARD_DONE", flush=True)


if __name__ == "__main__":
    main()
