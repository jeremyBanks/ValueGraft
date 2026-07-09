#!/usr/bin/env python3
"""Conversation-clustered bootstrap CIs for the JUDGED (meaning-recovery) metric,
Qwen3-30B-A3B bf16 (task #30).

WHAT THIS REPRODUCES
--------------------
FINDINGS.md F1 reports a meaning-judged category table with a "graft - Compacted"
column: sense +12pp, referent +10pp, stance +2pp. Those are BARE point estimates
with no confidence interval. This script puts a proper CI on each.

DATA (verified on disk, not assumed)
------------------------------------
The MEANING re-judge (Sonnet 5, RECOVERED/PARTIAL/MISSED, PARTIAL=0.5) lives in:
  results/judge_semantic/verdicts_*.json        -> GRAFT arms  E-post-a0.25, E-post-a1.0
  results/judge_semantic_base/verdicts_*.json   -> BASE arms   A (Original), B (Compacted)
Verdict key format: "sem|<conv>|<arm>|<plant>" e.g. "sem|c01|E-post-a1.0|c01-sense-2".
Category is the middle token of the plant id (referent / sense / stance).

The published point estimate = pooled-graft mean - Compacted(B) mean, per category,
where "pooled graft" pools BOTH E-post doses (a0.25 and a1.0). This script confirms
that construction reproduces +12/+10/+2 before CI'ing it.

Structure: 64 plants (sense 23, referent 18, stance 23) across 12 conversations
(c01..c12). Each plant has exactly 2 graft dose-verdicts + 1 Compacted verdict.
Plants within a conversation are correlated, so we resample CONVERSATIONS with
replacement (cluster bootstrap), not individual plants. Plant-level resampling
would be anti-conservative.

ESTIMATOR (matches the published number exactly)
------------------------------------------------
For a category c: theta_c = mean(graft dose-verdict scores in c)
                            - mean(Compacted verdict scores in c).
Cluster bootstrap: resample the 12 conv ids with replacement; pool all graft and
all Compacted verdicts belonging to the drawn convs; recompute theta_c; 95% CI =
2.5/97.5 percentiles over B resamples.

Usage: python3 scripts/judged_bootstrap.py [--nboot 20000] [--seed 0]
No GPU, on-disk only.
"""
import argparse
import glob
import json
import os
import random

SCORE = {"RECOVERED": 1.0, "PARTIAL": 0.5, "MISSED": 0.0}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAFT_GLOB = os.path.join(REPO, "results/judge_semantic/verdicts_*.json")
BASE_GLOB = os.path.join(REPO, "results/judge_semantic_base/verdicts_*.json")
CATS = ["referent", "sense", "stance"]


def load_verdicts(pattern):
    merged = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            merged.update(json.load(fh))
    return merged


def parse_key(key):
    _, conv, arm, plant = key.split("|")
    cat = plant.split("-")[1]
    return conv, arm, plant, cat


def collect(graft, base):
    """Return per-category {conv: {'graft': [scores], 'base': [scores]}} plus counts."""
    data = {c: {} for c in CATS}
    plants = {c: set() for c in CATS}
    convs = set()
    for key, verd in graft.items():
        conv, arm, plant, cat = parse_key(key)
        if cat not in CATS:
            continue
        convs.add(conv)
        plants[cat].add((conv, plant))
        data[cat].setdefault(conv, {"graft": [], "base": []})["graft"].append(SCORE[verd])
    for key, verd in base.items():
        conv, arm, plant, cat = parse_key(key)
        if arm != "B" or cat not in CATS:  # B == Compacted baseline only
            continue
        convs.add(conv)
        data[cat].setdefault(conv, {"graft": [], "base": []})["base"].append(SCORE[verd])
    return data, plants, sorted(convs)


def theta(conv_list, per_conv):
    """Pooled graft mean - pooled base mean over the given (multiset of) convs."""
    g, b = [], []
    for conv in conv_list:
        cell = per_conv.get(conv)
        if not cell:
            continue
        g.extend(cell["graft"])
        b.extend(cell["base"])
    if not g or not b:
        return None
    return sum(g) / len(g) - sum(b) / len(b)


def bootstrap(per_conv, convs, nboot, rng):
    point = theta(convs, per_conv)
    draws = []
    n = len(convs)
    for _ in range(nboot):
        sample = [convs[rng.randrange(n)] for _ in range(n)]
        val = theta(sample, per_conv)
        if val is not None:
            draws.append(val)
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws))]
    return point, lo, hi, len(draws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graft-glob", default=GRAFT_GLOB,
                    help="verdicts glob for graft arms (default: c01-c12 committed)")
    ap.add_argument("--base-glob", default=BASE_GLOB,
                    help="verdicts glob for base arms (default: c01-c12 committed)")
    ap.add_argument("--label", default="Qwen3-30B-A3B 4bit MLX, BRIEF summary",
                    help="run label (the committed c01-c12 metric is 4bit MLX / brief, NOT bf16)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    graft = load_verdicts(args.graft_glob)
    base = load_verdicts(args.base_glob)
    data, plants, convs = collect(graft, base)

    print(f"JUDGED meaning-recovery metric | {args.label} | Sonnet 5 judge")
    print(f"score: RECOVERED=1, PARTIAL=0.5, MISSED=0 | graft=pooled(E-post-a0.25,a1.0)")
    print(f"baseline=Compacted(arm B) | conversation-clustered bootstrap, "
          f"nboot={args.nboot} seed={args.seed}")
    print(f"N conversations (clusters) = {len(convs)}: {convs}")
    print()
    header = f"{'category':10s} {'Nplant':>6s} {'Ngraft':>6s} {'Nbase':>5s} " \
             f"{'graft-Compacted (pp)':>22s} {'95% CI (pp)':>22s}  excludes 0?"
    print(header)
    print("-" * len(header))
    for cat in CATS:
        per_conv = data[cat]
        n_graft = sum(len(c["graft"]) for c in per_conv.values())
        n_base = sum(len(c["base"]) for c in per_conv.values())
        point, lo, hi, nd = bootstrap(per_conv, convs, args.nboot, rng)
        excl = "YES" if (lo > 0 or hi < 0) else "no (spans 0)"
        print(f"{cat:10s} {len(plants[cat]):6d} {n_graft:6d} {n_base:5d} "
              f"{point*100:+21.1f} {'['+format(lo*100,'+.1f')+', '+format(hi*100,'+.1f')+']':>22s}"
              f"  {excl}")


if __name__ == "__main__":
    main()
