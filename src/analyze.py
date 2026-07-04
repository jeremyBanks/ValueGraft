"""Aggregate results: normalized gap closure, bootstrap CIs, headline tables.

Gap closure for arm X on a conversation: (X - B) / (A - B), computed on mean
per-token continuation logprob. 1.0 = fully recovers the oracle; 0 = no better
than standard compaction; negative = worse than B.

Outputs results/analysis.md (tables) and results/gap_closure.json.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def bootstrap_ci(vals, n=10000, seed=7):
    rng = random.Random(seed)
    if len(vals) == 1:
        return (vals[0], vals[0])
    means = sorted(
        sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n)
    )
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    rows = []
    for rp in sorted(Path("results/raw").glob("*.json")):
        res = json.load(open(rp))
        arms = res["cont"]["arms"]
        a, b = arms["A"]["mean_logprob"], arms["B"]["mean_logprob"]
        for name, d in arms.items():
            rows.append({
                "conv": res["id"],
                "kind": "synthetic" if res["id"].startswith("c") else "natural",
                "arm": name,
                "mean_logprob": d["mean_logprob"],
                "closure": (d["mean_logprob"] - b) / (a - b) if a != b else 0.0,
            })

    by_arm = defaultdict(list)
    for r in rows:
        by_arm[(r["kind"], r["arm"])].append(r)

    lines = ["# Continuation logprob analysis\n"]
    out = {}
    for kind in ("synthetic", "natural"):
        arm_names = sorted({a for k, a in by_arm if k == kind})
        if not arm_names:
            continue
        lines.append(f"\n## {kind} (n={len({r['conv'] for r in rows if r['kind']==kind})})\n")
        lines.append("| arm | mean logprob | gap closure | 95% CI |")
        lines.append("|---|---|---|---|")
        order = ["A", "B", "C", "D"] + sorted(
            a for a in arm_names if a.startswith("E")
        )
        for arm in [a for a in order if a in arm_names]:
            rs = by_arm[(kind, arm)]
            lp = sum(r["mean_logprob"] for r in rs) / len(rs)
            cl = [r["closure"] for r in rs]
            mcl = sum(cl) / len(cl)
            lo, hi = bootstrap_ci(cl)
            lines.append(f"| {arm} | {lp:.4f} | {mcl:.3f} | [{lo:.3f}, {hi:.3f}] |")
            out[f"{kind}|{arm}"] = {
                "mean_logprob": lp, "closure_mean": mcl,
                "closure_ci": [lo, hi],
                "per_conv": {r["conv"]: r["closure"] for r in rs},
            }

    # per-conversation table for the appendix
    lines.append("\n## Per-conversation gap closure\n")
    convs = sorted({r["conv"] for r in rows})
    arms_all = [a for a in ["C", "D", "E-post-a1.0", "E-inter-a1.0"]
                if any(r["arm"] == a for r in rows)]
    lines.append("| conv | A | B | " + " | ".join(arms_all) + " |")
    lines.append("|" + "---|" * (len(arms_all) + 3))
    for cv in convs:
        rr = {r["arm"]: r for r in rows if r["conv"] == cv}
        cells = [f"{rr['A']['mean_logprob']:.3f}", f"{rr['B']['mean_logprob']:.3f}"]
        cells += [f"{rr[a]['closure']:.2f}" if a in rr else "-" for a in arms_all]
        lines.append(f"| {cv} | " + " | ".join(cells) + " |")
    Path("results/analysis.md").write_text("\n".join(lines) + "\n")
    json.dump(out, open("results/gap_closure.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
