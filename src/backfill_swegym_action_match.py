"""Backfill the discrete next-action-match metric on EXISTING SWE-Gym result dirs
(free step 0). Reads the persisted `gen` fields + the parquet (gold action via
meta.cut_msg), computes tool/path/command match per arm, and writes a SIDECAR
`action_match_backfill.json` per dir (the committed per-trajectory result files are
NOT mutated -- provenance/loss-lesson). Also prints the continuous logprob effect
and, when multiple dirs cover the SAME trajectory indices, a test-retest POOLED
(per-trajectory averaged) estimate.

Usage:
  python3 src/backfill_swegym_action_match.py \
      --parquet swegym.parquet \
      --dirs results/swegym_30b_bf16 results/swegym_30b_bf16_brief
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")
from swegym_action_match import extract_action, match_action  # noqa: E402


def boot(xs, seed=0, nb=10000):
    n = len(xs)
    if not n:
        return None
    rng = random.Random(seed)
    ms = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(nb))
    return {"n": n, "mean": sum(xs) / n, "lo": ms[int(.025 * nb)],
            "hi": ms[int(.975 * nb)], "pos": sum(1 for v in xs if v > 0)}


def fmt(b):
    if b is None:
        return "(none)"
    flag = "CI>0" if b["lo"] > 0 else ("CI<0" if b["hi"] < 0 else "spans0")
    return (f"mean={b['mean']:+.4f} CI[{b['lo']:+.4f},{b['hi']:+.4f}] "
            f"{b['pos']}/{b['n']} {flag}")


def load_trajs(parquet):
    import pandas as pd  # noqa: PLC0415
    return [list(r) for r in pd.read_parquet(parquet)["messages"]]


def process_dir(d, trajs, graft="E-tuned", ref="B"):
    rows = {}
    for f in sorted(glob.glob(str(Path(d) / "t*.json"))):
        j = json.load(open(f))
        a = j.get("arms") or {}
        if graft not in a or ref not in a:
            continue
        idx = j["idx"]
        gold = extract_action(trajs[idx][j["meta"]["cut_msg"]]["content"])
        rows[idx] = {
            "eb": a[graft]["tf_mean"] - a[ref]["tf_mean"],
            "gold_action": bool(gold),
            "ref_match": match_action(gold, extract_action(a[ref].get("gen")))["match"]
            if gold else None,
            "graft_match": match_action(gold, extract_action(a[graft].get("gen")))["match"]
            if gold else None,
        }
    return rows


def report(name, rows, graft, ref):
    eb = boot([r["eb"] for r in rows.values()])
    ga = [r for r in rows.values() if r["gold_action"]]
    ng = len(ga)
    rmatch = sum(1 for r in ga if r["ref_match"])
    gmatch = sum(1 for r in ga if r["graft_match"])
    gwin = sum(1 for r in ga if r["graft_match"] and not r["ref_match"])
    rwin = sum(1 for r in ga if r["ref_match"] and not r["graft_match"])
    disc = boot([int(r["graft_match"]) - int(r["ref_match"]) for r in ga])
    print(f"\n== {name}  (n={len(rows)}) ==")
    print(f"  continuous {graft}-{ref} (tf logprob): {fmt(eb)}")
    print(f"  gold has a discrete action: {ng}/{len(rows)}")
    print(f"  action-match rate  {ref}={rmatch}/{ng} ({100*rmatch/ng:.0f}%)  "
          f"{graft}={gmatch}/{ng} ({100*gmatch/ng:.0f}%)")
    print(f"  discriminating: {graft}-fixes={gwin}  {ref}-breaks={rwin}  "
          f"net={gwin-rwin:+d}")
    print(f"  paired discrete ({graft}_match - {ref}_match): {fmt(disc)}")
    return {"continuous_eb": eb, "n_gold_action": ng,
            "match_rate": {ref: rmatch / ng, graft: gmatch / ng},
            "graft_fixes": gwin, "ref_breaks": rwin,
            "paired_discrete": disc}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default="swegym.parquet")
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--graft", default="E-tuned")
    ap.add_argument("--ref", default="B")
    args = ap.parse_args()
    trajs = load_trajs(args.parquet)

    per_dir = {}
    for d in args.dirs:
        rows = process_dir(d, trajs, args.graft, args.ref)
        if not rows:
            print(f"\n== {d}: no usable {args.graft}/{args.ref} arms =="); continue
        summ = report(d, rows, args.graft, args.ref)
        per_dir[d] = rows
        sidecar = Path(d) / "action_match_backfill.json"
        sidecar.write_text(json.dumps({
            "note": ("discrete next-action-match backfilled from persisted gen "
                     "fields (free step 0); originals NOT mutated"),
            "graft": args.graft, "ref": args.ref, "summary": summ,
            "per_trajectory": {str(i): r for i, r in rows.items()}}, indent=1))
        print(f"  WROTE {sidecar}")

    # test-retest pool over dirs that share trajectory indices
    if len(per_dir) >= 2:
        idsets = [set(r) for r in per_dir.values()]
        common = set.intersection(*idsets)
        if common:
            allids = set.union(*idsets)
            overlap = "SAME set (test-retest)" if common == allids else "PARTIAL"
            pooled = []
            for i in sorted(common):
                vals = [per_dir[d][i]["eb"] for d in per_dir if i in per_dir[d]]
                pooled.append(sum(vals) / len(vals))
            print(f"\n== POOLED across {len(per_dir)} dirs "
                  f"({len(common)} shared idx, {overlap}) ==")
            print(f"  per-trajectory AVERAGED continuous {args.graft}-{args.ref} "
                  f"(denoised): {fmt(boot(pooled))}")
            print("  NOTE: shared-index dirs are NOT independent N -- this denoises "
                  "the SAME trajectories. Independent N needs NEW disjoint "
                  "trajectories (SC_SWE_MIN_IDX).")


if __name__ == "__main__":
    main()
