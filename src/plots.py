"""Headline figures.

Fig 1: normalized gap closure (continuation logprob) per arm, synthetic vs
       natural, with bootstrap CIs — the brief's headline figure.
Fig 2: E-post alpha curve (gap closure vs alpha).
Fig 3: micro-experiment margins (oracle / fresh / V-swap alphas / KV-swap).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_gap_closure():
    gc = json.load(open("results/gap_closure.json"))
    for kind in ("synthetic", "natural"):
        arms = [(k.split("|")[1], v) for k, v in gc.items() if k.startswith(kind)]
        arms = [(a, v) for a, v in arms if a not in ("A", "B")]
        if not arms:
            continue
        arms.sort(key=lambda x: x[0])
        names = [a for a, _ in arms]
        means = [v["closure_mean"] for _, v in arms]
        los = [v["closure_mean"] - v["closure_ci"][0] for _, v in arms]
        his = [v["closure_ci"][1] - v["closure_mean"] for _, v in arms]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.axhline(0, color="#888", lw=1, label="B (standard compaction)")
        ax.axhline(1, color="#2a9d2a", lw=1, ls="--", label="A (full context)")
        ax.errorbar(range(len(names)), means, yerr=[los, his], fmt="o",
                    capsize=4, color="#1f4e9c")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("normalized gap closure (arm−B)/(A−B)")
        ax.set_title(f"Continuation logprob gap closure — {kind}")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(f"results/fig_gap_closure_{kind}.png", dpi=150)
        print(f"wrote results/fig_gap_closure_{kind}.png")


def fig_alpha_curve():
    gc = json.load(open("results/gap_closure.json"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, marker in (("synthetic", "o"), ("natural", "s")):
        for prefix, ls in (("E-post", "-"), ("E-inter", "--")):
            pts = []
            for k, v in gc.items():
                if not k.startswith(kind):
                    continue
                arm = k.split("|")[1]
                if arm.startswith(prefix + "-a"):
                    pts.append((float(arm.split("a")[-1]), v["closure_mean"]))
            if pts:
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        marker=marker, ls=ls, label=f"{prefix} ({kind})")
    ax.axhline(0, color="#888", lw=1)
    ax.set_xlabel("alpha (fraction of old V blended in)")
    ax.set_ylabel("gap closure")
    ax.set_title("Value-transplant alpha sweep")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/fig_alpha_curve.png", dpi=150)
    print("wrote results/fig_alpha_curve.png")


def fig_micro():
    rows = json.load(open("results/micro_sense.json"))
    keys = ["fresh", "vswap0.25", "vswap0.5", "vswap0.75", "vswap1.0",
            "kvswap1.0", "oracle"]
    means = []
    for k in keys:
        vals = [(r[f"{k}_std"] + r[f"{k}_flip"]) / 2 for r in rows]
        means.append(sum(vals) / len(vals))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#888"] + ["#1f4e9c"] * 4 + ["#7a3fb0", "#2a9d2a"]
    ax.bar(range(len(keys)), means, color=colors)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.set_ylabel("logprob margin: planted sense − default sense")
    ax.set_title("Micro: does a transplanted V carry word sense? (n=6 items)")
    fig.tight_layout()
    fig.savefig("results/fig_micro_sense.png", dpi=150)
    print("wrote results/fig_micro_sense.png")


if __name__ == "__main__":
    for fn in (fig_gap_closure, fig_alpha_curve, fig_micro):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: {e}")
