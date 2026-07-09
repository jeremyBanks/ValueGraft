#!/usr/bin/env python3
"""reproduce.py — one-command reproduction of the paper's load-bearing numbers FROM DISK (no GPU).

Every headline number in the paper is recomputed here from committed result files, so a
reader can verify the claims without a pod. The expensive GPU steps (rendering + scoring)
are documented at the bottom but not required for this analysis path.

Usage:
  python3 scripts/reproduce.py                # run everything
  python3 scripts/reproduce.py --only honesty # honesty (PRIMARY POSITIVE)
  python3 scripts/reproduce.py --only sense   # judged-sense render-fragility (the decider)
  python3 scripts/reproduce.py --only arch    # cross-architecture sign map
Sections map to CLAIMS.md rows (F2, REC-5, CA-*).
"""
import argparse, glob, json, os, random, subprocess, sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def P(*a): return os.path.join(REPO, *a)


# ---------- F2 honesty (PRIMARY POSITIVE) : conversation-clustered bootstrap ----------
def _perconv(rows, kind, arm, pos):
    byc = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["kind"] == kind and r["arm"] == arm:
            byc[r["conv"]][1] += 1
            if r["verdict"] == pos:
                byc[r["conv"]][0] += 1
    return byc

def _cluster_ci(A, B, nboot=20000, seed=0):
    rng = random.Random(seed)
    convs = sorted(set(A) | set(B))
    def rate(sample, t):
        p = q = 0
        for c in sample:
            if c in t: p += t[c][0]; q += t[c][1]
        return p / q if q else None
    pt = rate(convs, A) - rate(convs, B)
    draws = []
    n = len(convs)
    for _ in range(nboot):
        s = [convs[rng.randrange(n)] for _ in range(n)]
        ra, rb = rate(s, A), rate(s, B)
        if ra is not None and rb is not None: draws.append(ra - rb)
    draws.sort()
    return pt, draws[int(.025 * len(draws))], draws[int(.975 * len(draws))], len(convs)

def honesty():
    print("\n=== F2 HONESTY (PRIMARY POSITIVE) — conversation-clustered bootstrap ===")
    print("source: results/phase2_30b_scored.json (30B).  fabrication↓ = Compacted(B) − H-pack.")
    rows = json.load(open(P("results/phase2_30b_scored.json")))
    for kind in ["decoy", "evicted_fact"]:
        A = _perconv(rows, kind, "B", "FABRICATED")
        B = _perconv(rows, kind, "H-pack", "FABRICATED")
        pt, lo, hi, nc = _cluster_ci(A, B)
        excl = "SIGNIFICANT" if (lo > 0 or hi < 0) else "spans 0"
        print(f"  {kind:12s} fabrication↓ = {pt*100:+5.1f}pp  CI[{lo*100:+.1f},{hi*100:+.1f}]  clusters={nc}  {excl}")
        # decomposition: layout vs write-time-KV-within-layout
        L = _cluster_ci(_perconv(rows, kind, "B", "FABRICATED"), _perconv(rows, kind, "B-min-pack", "FABRICATED"))
        K = _cluster_ci(_perconv(rows, kind, "B-min-pack", "FABRICATED"), _perconv(rows, kind, "H-pack", "FABRICATED"))
        print(f"      ├ packed-layout alone      = {L[0]*100:+5.1f}pp CI[{L[1]*100:+.1f},{L[2]*100:+.1f}] {'sig' if (L[1]>0 or L[2]<0) else 'ns'}")
        print(f"      └ write-time-KV beyond it  = {K[0]*100:+5.1f}pp CI[{K[1]*100:+.1f},{K[2]*100:+.1f}] {'sig' if (K[1]>0 or K[2]<0) else 'ns'}  <- KV-specific claim")
        # recall check: is H-pack accurate on evicted? (the DEAD 38/48 overclaim)
        if kind == "evicted_fact":
            cor = sum(1 for r in rows if r["kind"] == kind and r["arm"] == "H-pack" and r["verdict"] == "CORRECT")
            adm = sum(1 for r in rows if r["kind"] == kind and r["arm"] == "H-pack" and r["verdict"] == "ADMITTED")
            n = sum(1 for r in rows if r["kind"] == kind and r["arm"] == "H-pack")
            print(f"      RECALL check: H-pack CORRECT={cor}/{n} (buys HONESTY not RECALL; admits {adm}/{n}). '38/48 accurate' = DEAD overclaim.")
    print("  CAVEATS: arm=H-pack (keys+values cousin, not value-only); n=24/12-clusters; NOT re-render-tested;")
    print("           scope=mid-task agentic compaction (washes out on retrieval QA at 30B).")


# ---------- REC-5 judged sense render-fragility (the decider) ----------
def _bootstrap(graft_glob, base_glob, label):
    cmd = [sys.executable, P("scripts/judged_bootstrap.py"), "--nboot", "20000", "--seed", "0",
           "--graft-glob", P(graft_glob), "--base-glob", P(base_glob), "--label", label]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.strip().startswith(("referent", "sense", "stance")):
            print("    " + line)

def sense():
    print("\n=== REC-5 JUDGED SENSE — RENDER-FRAGILE (the positive control that FAILED) ===")
    print("  Same strict judge, two renders → the +8.7→+1.0 collapse is the RENDER (MoE nondeterminism, n=12):")
    print("  [original render, re-judged by the strict judge]")
    _bootstrap("results/judge_semantic/reverdict_*.json", "results/judge_semantic_base/reverdict_*.json", "orig-render/strict-judge")
    print("  [clean current-code re-render, same judge]")
    _bootstrap("results/judge_semantic_brief_base/verdicts_*.json", "results/judge_semantic_base_brief_base/verdicts_*.json", "clean-render/strict-judge")
    print("  VERDICT: sense collapses +8.7[0.0,17.7] → +1.0[-5.2,+7.3]. Meaning-recovery is NOT robust; report as non-reproduction.")


# ---------- cross-architecture sign map ----------
def arch():
    print("\n=== CROSS-ARCHITECTURE referent raw_EB (mechanism color; the EFFECT it maps is fragile) ===")
    for f in sorted(glob.glob(P("results/cross_arch_done/*.json"))):
        name = os.path.basename(f)
        try:
            d = json.load(open(f))
        except Exception:
            continue
        r = (d.get("by_category_robust", {}) or {}).get("referent", {}) or {}
        eb = r.get("raw_EB") if r.get("raw_EB") is not None else r.get("raw_EB_mean")
        ci = r.get("raw_EB_ci_cluster") or r.get("raw_EB_ci")
        if eb is not None:
            print(f"  {name:48s} referent raw_EB={eb:+.3f}  CI={ci}")
    print("  NOTE (field trap): for Mistral cite raw_EB_ci_cluster, NOT raw_EB_ci (different ci_method).")
    print("  Frame honestly: one FRAGILE positive pole (MoE anchor) vs several solid negative poles; not a settled 'reversal'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["honesty", "sense", "arch", "all"], default="all")
    a = ap.parse_args()
    print("ValueGraft — reproduction of load-bearing numbers from disk (no GPU). See CLAIMS.md for provenance.")
    if a.only in ("honesty", "all"): honesty()
    if a.only in ("sense", "all"): sense()
    if a.only in ("arch", "all"): arch()
    print("\nGPU path (not needed for the above): render+score via src/run_arms.py (judged, MLX 4-bit local)")
    print("and src/cross_arch_probe.py (raw_EB, HF bf16 pods); then scripts/build_judge_batches.py → Sonnet judge → judged_bootstrap.py.")


if __name__ == "__main__":
    main()
