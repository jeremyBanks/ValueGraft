"""Brief-summary (summary-shadow) condition: PROBE mode only, terse summary.

The terse summary omits specifics by instruction, so most plants become
absent-from-summary — the load-bearing leak class. Arms: B, C, H-gap, B-min,
E-post a=0.5/1.0 (A and D are summary-independent; copy from the main run at
analysis time). Results: results/raw_brief/<cid>.json.
"""

import json
import sys
import time
from pathlib import Path

from mlx_lm import load

sys.path.insert(0, "src")
from arms import SUMMARY_REQUEST_BRIEF
from run_arms import MODEL, ArmSet, run_probe_mode

ARMS = {"B", "C", "H-gap", "B-min", "E-post-a0.5", "E-post-a1.0"}


def main():
    model, tokenizer = load(MODEL)
    outdir = Path("results/raw_brief")
    outdir.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    import run_arms

    for p in sorted(Path("data/synthetic").glob("c*.json")):
        conv = json.load(open(p))
        cid = conv["id"]
        if only and cid not in only:
            continue
        outfile = outdir / f"{cid}.json"
        if outfile.exists():
            print(f"{cid}: done already, skipping")
            continue
        t0 = time.time()
        print(f"== brief {cid}")
        # monkey-source: run_probe_mode builds its own ArmSet; inline a local
        # variant here instead for the custom summary request.
        aset = ArmSet(model, tokenizer, conv["messages"],
                      conv["sections"]["middle_end_msg"],
                      summary_request=SUMMARY_REQUEST_BRIEF)
        aset.prepare_a()
        from kvlib import extend_cache, first_step_logits, greedy_generate
        from arms import render
        eos_ids = set(tokenizer.eos_token_ids or [tokenizer.eos_token_id])
        out = {"stats": aset.stats, "arms": {}}
        for name, mk, ctx_ids, render_msgs in aset.variants(
            e_post_alphas=[0.5, 1.0], e_inter_alphas=[]
        ):
            if name not in ARMS:
                continue
            answers = {}
            for plant in conv["plants"]:
                full = render(tokenizer, render_msgs +
                              [{"role": "user", "content": plant["probe"]}], True)
                assert full[: len(ctx_ids)] == ctx_ids
                cache = mk()
                if len(full) - len(ctx_ids) > 1:
                    extend_cache(model, cache, full[len(ctx_ids):-1])
                logits = first_step_logits(model, cache, full[-1])
                toks, _ = greedy_generate(model, cache, logits, 160, eos_ids)
                answers[plant["id"]] = tokenizer.decode(toks).strip()
            out["arms"][name] = answers
            print(f"    PROBE {name}: done")
        json.dump({"id": cid, "model": MODEL, "probes": out,
                   "wall_seconds": time.time() - t0},
                  open(outfile, "w"), indent=1, ensure_ascii=False)
        print(f"== {cid} brief done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
