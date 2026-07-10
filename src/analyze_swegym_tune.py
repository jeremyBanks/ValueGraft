"""In-domain SWE-Gym graft optimization analysis (CPU-only; no torch).

Reads a run directory produced by run_swegym_hf.py with the in-domain optimization
arms (SC_E_ALPHAS / SC_SWE_PLACEBO / SC_PROFILE_REGIONS) and answers, with held-out
discipline: (1) is flat alpha=0.75 optimal on brief-SWE-Gym, or does another alpha
recover more of the next-action gap; (2) does a per-region-profiled, SWE-Gym-tuned
per-layer champion beat the naive scalar; (3) is the graft content-specific here.

HELD-OUT DISCIPLINE (critical at N~=75): every trajectory carries a deterministic
"split" ("tune"|"eval"). Alpha is SELECTED on TUNE and REPORTED on the DISJOINT
EVAL split; the layer champion is BUILT from TUNE-split per-region marginals and is
meant to be EVALUATED by a SECOND harness run on the EVAL split (pass that run dir
as --champion-eval). We also print the in-sample (all-trajectory) alpha curve as a
descriptive aid, clearly labelled as NOT the held-out claim.

Metric: paired per-trajectory (arm.tf_mean - B.tf_mean), teacher-forced next-action
mean logprob (a PROXY, not resolve rate). CI = percentile bootstrap over
trajectories (each trajectory is one cluster; one target each), n=10000, seed=0.

Usage:
  python3 src/analyze_swegym_tune.py --run results/swegym_tune_<stamp>_brief \
      [--write-champion data/champion_configs/swegym_tuned_<stamp>.json] \
      [--champion-eval results/swegym_champeval_<stamp>_brief]
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path


def boot(diffs, n_boot=10000, seed=0):
    n = len(diffs)
    if n == 0:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {"n": n, "mean": sum(diffs) / n,
            "lo": means[int(0.025 * n_boot)], "hi": means[int(0.975 * n_boot)],
            "pos": sum(1 for x in diffs if x > 0)}


def load_run(run_dir):
    """Return {idx: {"split":..., "arms": {name: tf_mean}}} for every scored file."""
    out = {}
    for f in sorted(glob.glob(str(Path(run_dir) / "t*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        arms = {k: v.get("tf_mean") for k, v in (d.get("arms") or {}).items()
                if isinstance(v, dict) and v.get("tf_mean") is not None}
        if "B" not in arms:
            continue
        out[d["idx"]] = {"split": d.get("split", "eval"), "arms": arms}
    return out


def paired(rows, arm, ref="B", split=None):
    """[arm.tf - ref.tf] over trajectories (optionally restricted to a split)."""
    out = []
    for r in rows.values():
        if split and r["split"] != split:
            continue
        a = r["arms"]
        if arm in a and ref in a:
            out.append(a[arm] - a[ref])
    return out


def fmt(b):
    if b is None:
        return "   (none)"
    flag = "CI>0" if b["lo"] > 0 else ("CI<0" if b["hi"] < 0 else "spans0")
    return (f"mean={b['mean']:+.4f} CI[{b['lo']:+.4f},{b['hi']:+.4f}] "
            f"{b['pos']}/{b['n']} {flag}")


def alpha_arms(rows):
    names = set()
    for r in rows.values():
        names.update(k for k in r["arms"] if k.startswith("E-a"))
    return sorted(names, key=lambda s: float(s[3:]))


def region_arms(rows):
    names = set()
    for r in rows.values():
        names.update(k for k in r["arms"] if k.startswith("R")
                     and k[1:].isdigit())
    return sorted(names, key=lambda s: int(s[1:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="stage-1 run dir (sweep+profile)")
    ap.add_argument("--write-champion", default=None,
                    help="path to write the TUNE-built per-layer champion config")
    ap.add_argument("--champion-eval", default=None,
                    help="stage-2 run dir that evaluated the champion held-out")
    ap.add_argument("--profile-alpha", type=float, default=1.0,
                    help="alpha the regions were profiled at (for champion map)")
    args = ap.parse_args()

    rows = load_run(args.run)
    n_tune = sum(1 for r in rows.values() if r["split"] == "tune")
    n_eval = sum(1 for r in rows.values() if r["split"] == "eval")
    print(f"== SWE-Gym in-domain optimization :: {args.run}")
    print(f"   trajectories: {len(rows)} (tune={n_tune}, eval={n_eval})")
    print(f"   metric: teacher-forced next-action mean logprob (PROXY). "
          f"CI=bootstrap over trajectories.\n")

    # ---- scalar reference (the +0.013 to beat) ------------------------------
    print("-- scalar baseline arm E-tuned (flat alpha) vs B --")
    for split in ("all", "tune", "eval"):
        b = boot(paired(rows, "E-tuned", split=None if split == "all" else split))
        print(f"   {split:5}: {fmt(b)}")
    print()

    # ---- ALPHA SWEEP --------------------------------------------------------
    aarms = alpha_arms(rows)
    if aarms:
        print("-- ALPHA SWEEP (E-a{alpha} - B) --")
        print(f"   {'alpha':>6} {'ALL (in-sample)':>34} {'EVAL (held-out)':>34} "
              f"{'content-spec E-P (all)':>34}")
        best = None
        for a in aarms:
            al = a[3:]
            b_all = boot(paired(rows, a))
            b_tune = boot(paired(rows, a, split="tune"))
            b_eval = boot(paired(rows, a, split="eval"))
            p = a.replace("E-a", "P-a")
            cs = boot([x for r in rows.values()
                       if a in r["arms"] and p in r["arms"]
                       for x in [r["arms"][a] - r["arms"][p]]])
            print(f"   {al:>6} {fmt(b_all):>34} {fmt(b_eval):>34} "
                  f"{fmt(cs):>34}")
            if b_tune and (best is None or b_tune["mean"] > best[1]):
                best = (a, b_tune["mean"])
        print()
        if best:
            a = best[0]
            b_eval = boot(paired(rows, a, split="eval"))
            print(f"   SELECTED alpha on TUNE = {a[3:]} (tune mean={best[1]:+.4f}); "
                  f"its HELD-OUT EVAL result: {fmt(b_eval)}")
            b_scalar_eval = boot(paired(rows, "E-tuned", split="eval"))
            print(f"   vs scalar 0.75 on EVAL: {fmt(b_scalar_eval)}")
            print("   NOTE: alpha selected on ~half the trajectories; EVAL CI is "
                  "thin. A tuned alpha 'winning' in-sample but not held-out = "
                  "overfitting, report as such.\n")

    # ---- PER-REGION PROFILE + champion build --------------------------------
    rarms = region_arms(rows)
    if rarms:
        print("-- PER-REGION MARGINALS on TUNE (R{k} - B), profile alpha "
              f"{args.profile_alpha} --")
        champ_regions = []
        # recover each region's layer set from the arm intervention block
        region_layers = {}
        for f in glob.glob(str(Path(args.run) / "t*.json")):
            d = json.load(open(f))
            for k, v in (d.get("arms") or {}).items():
                if k.startswith("R") and k[1:].isdigit():
                    iv = v.get("intervention") or {}
                    if iv.get("layers"):
                        region_layers[k] = iv["layers"]
            if len(region_layers) >= len(rarms):
                break
        for rk in rarms:
            b = boot(paired(rows, rk, split="tune"))
            lays = region_layers.get(rk, [])
            span = f"L{min(lays)}-{max(lays)}" if lays else "?"
            keep = b is not None and b["mean"] > 0
            print(f"   {rk:>4} ({span:>9}): {fmt(b)}  {'-> keep' if keep else ''}")
            if keep:
                champ_regions.extend(lays)
        champ_layers = sorted(set(champ_regions))
        print(f"\n   CHAMPION (union of positive-TUNE regions): "
              f"{len(champ_layers)} layers {champ_layers}")
        if args.write_champion and champ_layers:
            cfg = {"label": Path(args.write_champion).stem,
                   "note": ("SWE-Gym-tuned per-layer champion: union of regions "
                            "with positive TUNE-split next-action marginal. "
                            "EVALUATE held-out on the EVAL split."),
                   "alpha_map": {str(l): args.profile_alpha for l in champ_layers}}
            Path(args.write_champion).write_text(json.dumps(cfg, indent=1))
            print(f"   WROTE champion config -> {args.write_champion}")
        print()

    # ---- STAGE 2 held-out champion evaluation -------------------------------
    if args.champion_eval:
        ev = load_run(args.champion_eval)
        eval_only = {i: r for i, r in ev.items() if r["split"] == "eval"}
        print(f"-- HELD-OUT CHAMPION EVAL :: {args.champion_eval} "
              f"({len(eval_only)} eval trajectories) --")
        for arm in ("E-tuned", "E-champion"):
            print(f"   {arm:11} - B: {fmt(boot(paired(eval_only, arm)))}")
        # head-to-head champion vs scalar on the eval split
        hh = [r['arms']['E-champion'] - r['arms']['E-tuned']
              for r in eval_only.values()
              if 'E-champion' in r['arms'] and 'E-tuned' in r['arms']]
        print(f"   E-champion - E-tuned (paired): {fmt(boot(hh))}")
        print("   -> champion BEATS scalar iff this CI clears 0; flat CI = the "
              "in-domain layer tuning added nothing (report either way).")


if __name__ == "__main__":
    main()
