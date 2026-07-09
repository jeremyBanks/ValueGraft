#!/usr/bin/env python3
"""BLOCK ANALYSIS for the headline-reproduction (value-graft) experiment.

Reads one or more cross_arch_probe result JSONs (each carrying per-plant
``traces``) and reads them as a BLOCK design:

    BASELINE = c01..c12  (positive control: must reproduce the validated
                          referent ~+0.10 with a CI excluding 0)
    FRESH    = c13..c24  (held-out convs the effect was never tuned on:
                          the real test of whether it carries)

The cell boundary is parameterized (``--baseline-max`` / ``--fresh-max``).

Design requirements enforced here (from the Fable execution review + the
frozen PREREGISTRATION.md):

1. Split plants into cells by conversation id; assert every plant parses and
   lands in exactly one cell.
2. RELATIVE COMPETENCE FLOOR computed ONCE, CENTRALLY, over the POOLED lp_A of
   ALL plants across ALL convs and ALL pooled pods:
       floor = median(lp_A) - K * MADN(lp_A),  MADN = 1.4826 * median|lp_A-median|
       K = 3.0 (frozen);  undefined (no drop) if < min-N finite lp_A.
   A SINGLE threshold applied identically to both cells. Never per-cell/per-pod.
3. Per cell x category: raw_EB mean, CONVERSATION-CLUSTERED bootstrap 95% CI
   (resample whole convs, not plants), N plants and N convs.
4. Per cell x category HEADROOM (A-B) and how many plants the headroom gate
   (0.3) drops; a null at LOW headroom (nothing to recover) is flagged
   distinctly from a null at HIGH headroom (the likely false-null).
5. Plant MORPHOLOGY per cell for the referent category (semantic-referent
   phrases vs short identifiers) so a fresh null caused by morphology drift is
   visible, not hidden.
6. The headline read + the pre-registered stopping rule (EXTEND if the FRESH
   referent CI spans zero at n=12 fresh convs).

Pure-Python, no numpy/torch/mlx. Reuses the exact cluster-bootstrap procedure
from src/cross_arch_probe.py (copied here so the script has no heavy imports).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys

# --- frozen defaults (mirror PREREGISTRATION.md) -----------------------------
COMPETENCE_K = 3.0            # extreme-outlier cutoff on lp_A
COMPETENCE_MIN_N = 8         # below this, floor undefined (permissive fallback)
HEADROOM_GATE = 0.3          # A-B headroom floor
MORPH_WORD_THRESHOLD = 3     # gold words <= this -> "short identifier"
N_BOOT = 10000
SEED = 0

CATEGORIES = ["referent", "sense", "stance", "ruled_out", "evicted_fact"]
CONV_RE = re.compile(r"^c(\d+)$")


# --- bootstrap (copied verbatim in behaviour from cross_arch_probe.py) --------
def bootstrap_ci_95_cluster(clusters, n_boot=N_BOOT, seed=SEED):
    """CLUSTER percentile bootstrap 95% CI of the mean.

    ``clusters`` = list of lists, one inner list per CONVERSATION. Resampling is
    over CONVERSATIONS (whole clusters, with replacement); probes inside a
    resampled conversation stay together. The point estimate is the pooled grand
    mean (unchanged by clustering); only the interval width reflects the
    conversation-level correlation. Returns mean, lo, hi, n (plants), n_clusters.
    """
    clusters = [[v for v in c if v is not None] for c in clusters]
    clusters = [c for c in clusters if c]
    flat = [v for c in clusters for v in c]
    n_c = len(clusters)
    if not flat:
        return {"mean": None, "lo": None, "hi": None, "n": 0, "n_clusters": 0}
    mean = sum(flat) / len(flat)
    if n_c == 1:
        return {"mean": mean, "lo": mean, "hi": mean,
                "n": len(flat), "n_clusters": 1}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        tot = 0.0
        cnt = 0
        for _ in range(n_c):
            c = clusters[rng.randrange(n_c)]
            tot += sum(c)
            cnt += len(c)
        means.append(tot / cnt)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(flat), "n_clusters": n_c}


def madn(values):
    """Normal-consistent MAD: 1.4826 * median(|x - median(x)|)."""
    med = statistics.median(values)
    return 1.4826 * statistics.median([abs(v - med) for v in values])


# --- loading / cell assignment ------------------------------------------------
def conv_index(conv_id):
    """Parse 'cNN' -> int NN. Raises on anything that does not parse."""
    m = CONV_RE.match(str(conv_id))
    if not m:
        raise ValueError(f"conversation_id does not parse as cNN: {conv_id!r}")
    return int(m.group(1))


def assign_cell(conv_id, baseline_max, fresh_max):
    idx = conv_index(conv_id)
    if idx < 1:
        raise ValueError(f"conversation index < 1: {conv_id!r}")
    if idx <= baseline_max:
        return "BASELINE"
    if idx <= fresh_max:
        return "FRESH"
    raise ValueError(
        f"{conv_id!r} (index {idx}) falls outside both cells "
        f"(baseline 1..{baseline_max}, fresh {baseline_max + 1}..{fresh_max}). "
        f"Widen --fresh-max or fix the input.")


def load_plants(paths, baseline_max, fresh_max):
    """Load + pool traces from every JSON. Tag each plant with its pod (source
    basename) and its cell. Assert lp_A/lp_B/lp_E present and cell valid."""
    plants = []
    per_pod = {}
    for path in paths:
        with open(path) as fh:
            doc = json.load(fh)
        traces = doc.get("traces")
        if not traces:
            raise ValueError(f"{path}: no per-plant 'traces' array present")
        pod = os.path.basename(path)
        per_pod[pod] = {"model": doc.get("model"), "n": len(traces)}
        for t in traces:
            for f in ("conversation_id", "category", "lp_A", "lp_B", "lp_E"):
                if t.get(f) is None:
                    raise ValueError(
                        f"{path}: plant {t.get('plant_id')!r} missing '{f}'")
            cell = assign_cell(t["conversation_id"], baseline_max, fresh_max)
            raw_eb = t.get("raw_EB")
            if raw_eb is None:
                raw_eb = t["lp_E"] - t["lp_B"]
            plants.append({
                "pod": pod,
                "plant_id": t.get("plant_id"),
                "conv": t["conversation_id"],
                "category": t["category"],
                "cell": cell,
                "lp_A": float(t["lp_A"]),
                "lp_B": float(t["lp_B"]),
                "lp_E": float(t["lp_E"]),
                "raw_EB": float(raw_eb),
                "headroom": float(t["lp_A"]) - float(t["lp_B"]),
                "gold": t.get("gold") or "",
                "n_gold_tokens": t.get("n_gold_tokens"),
            })
    return plants, per_pod


# --- central pooled competence floor -----------------------------------------
def central_competence_floor(plants, k, min_n):
    """ONE floor over the POOLED lp_A of ALL plants (both cells, all pods)."""
    lpa = [p["lp_A"] for p in plants]
    finite = [v for v in lpa if v == v and v not in (float("inf"), float("-inf"))]
    if len(finite) < min_n:
        return {"floor": None, "n_lpa": len(finite), "median": None,
                "madn": None, "reason": f"< {min_n} finite lp_A (permissive)"}
    med = statistics.median(finite)
    md = madn(finite)
    return {"floor": med - k * md, "n_lpa": len(finite),
            "median": med, "madn": md, "reason": None}


# --- morphology ---------------------------------------------------------------
def morphology(gold, word_threshold):
    """Classify a referent gold phrase: short identifier vs semantic phrase."""
    n_words = len(gold.split())
    return "short_id" if n_words <= word_threshold else "sem_phrase"


# --- per-cell / per-category stats -------------------------------------------
def cell_category_stats(plants, headroom_gate, n_boot, seed):
    """plants here are already competence-passing and in ONE cell + category."""
    # cluster by conversation for the honest CI
    by_conv = {}
    for p in plants:
        by_conv.setdefault(p["conv"], []).append(p["raw_EB"])
    clusters = list(by_conv.values())
    ci = bootstrap_ci_95_cluster(clusters, n_boot=n_boot, seed=seed)

    headrooms = [p["headroom"] for p in plants]
    mean_headroom = sum(headrooms) / len(headrooms) if headrooms else None
    n_headroom_dropped = sum(1 for h in headrooms if h < headroom_gate)

    # headroom-gated estimate (only plants with room to recover)
    gated = [p for p in plants if p["headroom"] >= headroom_gate]
    gated_by_conv = {}
    for p in gated:
        gated_by_conv.setdefault(p["conv"], []).append(p["raw_EB"])
    gated_ci = bootstrap_ci_95_cluster(
        list(gated_by_conv.values()), n_boot=n_boot, seed=seed)

    ci_excludes_zero = (
        ci["lo"] is not None and (ci["lo"] > 0 or ci["hi"] < 0))
    is_null = (ci["n"] > 0 and not ci_excludes_zero)
    low_headroom = (mean_headroom is not None and mean_headroom < headroom_gate)

    return {
        "n_plants": len(plants),
        "n_convs": len(clusters),
        "raw_EB_mean": ci["mean"],
        "ci_lo": ci["lo"],
        "ci_hi": ci["hi"],
        "ci_excludes_zero": ci_excludes_zero,
        "is_null": is_null,
        "mean_headroom": mean_headroom,
        "n_headroom_dropped": n_headroom_dropped,
        "gated_n_plants": gated_ci["n"],
        "gated_n_convs": gated_ci["n_clusters"],
        "gated_raw_EB_mean": gated_ci["mean"],
        "gated_ci_lo": gated_ci["lo"],
        "gated_ci_hi": gated_ci["hi"],
        "low_headroom": low_headroom,
        # a null-with-low-headroom is EXPECTED (nothing evicted to recover);
        # a null-with-high-headroom is the most likely FALSE null.
        "null_kind": (
            None if not is_null
            else "low-headroom (nothing to recover; not a real miss)"
            if low_headroom
            else "HIGH-headroom (likely false-null -- investigate)"),
    }


def fmt(x, nd=3):
    return "  n/a " if x is None else f"{x:+.{nd}f}"


def run(paths, baseline_max, fresh_max, k, min_n, headroom_gate,
        word_threshold, n_boot, seed, out=sys.stdout):
    plants, per_pod = load_plants(paths, baseline_max, fresh_max)

    p = lambda *a: print(*a, file=out)
    p("=" * 78)
    p("BLOCK ANALYSIS -- headline-reproduction value-graft experiment")
    p("=" * 78)
    p(f"Pooled {len(paths)} result JSON(s):")
    for pod, meta in per_pod.items():
        p(f"  - {pod}  model={meta['model']}  n_plants={meta['n']}")
    p(f"Cells: BASELINE = c01..c{baseline_max:02d}   "
      f"FRESH = c{baseline_max + 1:02d}..c{fresh_max:02d}")
    p(f"Total plants pooled: {len(plants)}")

    # ---- cell assignment sanity ----
    cell_counts = {"BASELINE": 0, "FRESH": 0}
    for pl in plants:
        cell_counts[pl["cell"]] += 1
    baseline_convs = sorted({pl["conv"] for pl in plants
                             if pl["cell"] == "BASELINE"})
    fresh_convs = sorted({pl["conv"] for pl in plants if pl["cell"] == "FRESH"})
    p(f"  BASELINE: {cell_counts['BASELINE']} plants over {len(baseline_convs)} "
      f"convs {baseline_convs}")
    p(f"  FRESH:    {cell_counts['FRESH']} plants over {len(fresh_convs)} "
      f"convs {fresh_convs}")

    # ---- central pooled competence floor (ONCE, over ALL plants) ----
    floor_info = central_competence_floor(plants, k, min_n)
    p("")
    p("-" * 78)
    p("CENTRAL POOLED COMPETENCE FLOOR (one threshold, both cells + all pods)")
    p("-" * 78)
    if floor_info["floor"] is None:
        p(f"  floor UNDEFINED: {floor_info['reason']} "
          f"(n finite lp_A = {floor_info['n_lpa']}) -> NO plants dropped")
        floor = float("-inf")
    else:
        floor = floor_info["floor"]
        p(f"  median(lp_A) = {floor_info['median']:.4f}   "
          f"MADN(lp_A) = {floor_info['madn']:.4f}   K = {k}")
        p(f"  floor = median - K*MADN = {floor:.4f}   "
          f"(over {floor_info['n_lpa']} pooled lp_A)")

    for cell in ("BASELINE", "FRESH"):
        cp = [pl for pl in plants if pl["cell"] == cell]
        dropped = [pl for pl in cp if pl["lp_A"] < floor]
        p(f"  {cell}: {len(dropped)} / {len(cp)} plants below floor "
          f"(dropped for competence)")

    kept = [pl for pl in plants if pl["lp_A"] >= floor]

    # ---- per cell x category table ----
    p("")
    p("-" * 78)
    p("PER CELL x CATEGORY  (competence-passing plants)")
    p("-" * 78)
    hdr = (f"{'cell':<9} {'category':<12} {'N':>3} {'cv':>3} "
           f"{'raw_EB':>8} {'CI_lo':>8} {'CI_hi':>8} {'sig':>4} "
           f"{'headrm':>7} {'hdrop':>5}")
    results = {"BASELINE": {}, "FRESH": {}}
    for cell in ("BASELINE", "FRESH"):
        p(f"[{cell}]")
        p("  " + hdr)
        for cat in CATEGORIES:
            sub = [pl for pl in kept
                   if pl["cell"] == cell and pl["category"] == cat]
            if not sub:
                p(f"  {cell:<9} {cat:<12} {'0':>3}  (no plants)")
                results[cell][cat] = None
                continue
            st = cell_category_stats(sub, headroom_gate, n_boot, seed)
            results[cell][cat] = st
            sig = "yes" if st["ci_excludes_zero"] else "no"
            p("  " + (f"{cell:<9} {cat:<12} {st['n_plants']:>3} "
                      f"{st['n_convs']:>3} {fmt(st['raw_EB_mean'])} "
                      f"{fmt(st['ci_lo'])} {fmt(st['ci_hi'])} {sig:>4} "
                      f"{fmt(st['mean_headroom'])} {st['n_headroom_dropped']:>5}"))
            if st["is_null"]:
                p(f"      -> NULL: {st['null_kind']}")
            # headroom-gated estimate (only where there is room to recover)
            if st["gated_n_plants"] and st["gated_n_plants"] != st["n_plants"]:
                p(f"      headroom-gated (>= {headroom_gate}): "
                  f"N={st['gated_n_plants']} cv={st['gated_n_convs']} "
                  f"raw_EB={fmt(st['gated_raw_EB_mean'])} "
                  f"CI=[{fmt(st['gated_ci_lo'])},{fmt(st['gated_ci_hi'])}]")
        p("")

    # ---- morphology (referent) ----
    p("-" * 78)
    p("REFERENT PLANT MORPHOLOGY per cell "
      f"(short_id = gold <= {word_threshold} words; else sem_phrase)")
    p("-" * 78)
    for cell in ("BASELINE", "FRESH"):
        refs = [pl for pl in kept
                if pl["cell"] == cell and pl["category"] == "referent"]
        counts = {"sem_phrase": 0, "short_id": 0}
        for pl in refs:
            counts[morphology(pl["gold"], word_threshold)] += 1
        total = len(refs) or 1
        p(f"  {cell}: {counts['sem_phrase']} sem_phrase "
          f"({100 * counts['sem_phrase'] / total:.0f}%), "
          f"{counts['short_id']} short_id "
          f"({100 * counts['short_id'] / total:.0f}%)  [N={len(refs)}]")

    # ---- headline read ----
    p("")
    p("=" * 78)
    p("HEADLINE READ")
    p("=" * 78)
    b_ref = results["BASELINE"].get("referent")
    f_ref = results["FRESH"].get("referent")

    if b_ref is None:
        p("  BASELINE referent: NO plants -- cannot run positive control (SUSPECT).")
    else:
        pc_pass = b_ref["ci_excludes_zero"] and (b_ref["raw_EB_mean"] or 0) > 0
        p(f"  POSITIVE CONTROL (BASELINE c01..c{baseline_max:02d} referent):")
        p(f"    raw_EB = {fmt(b_ref['raw_EB_mean'])}  "
          f"CI = [{fmt(b_ref['ci_lo'])}, {fmt(b_ref['ci_hi'])}]  "
          f"(N={b_ref['n_plants']} plants, {b_ref['n_convs']} convs)")
        if pc_pass:
            p("    -> PASS: reproduces a positive referent effect, CI excludes 0.")
            if b_ref["raw_EB_mean"] is not None and b_ref["raw_EB_mean"] < 0.05:
                p("       (NOTE: positive but well below the ~+0.10 referent -- "
                  "weaker than the validated magnitude.)")
        else:
            p("    -> FAIL: positive control did NOT reproduce (CI spans 0 or "
              "negative). THE RUN IS SUSPECT -- do not trust the FRESH read.")

    if f_ref is None:
        p("  FRESH referent: NO plants in c"
          f"{baseline_max + 1:02d}..c{fresh_max:02d} "
          "(fixture may not include the FRESH block).")
    else:
        p(f"  HELD-OUT TEST (FRESH c{baseline_max + 1:02d}..c{fresh_max:02d} "
          "referent):")
        p(f"    raw_EB = {fmt(f_ref['raw_EB_mean'])}  "
          f"CI = [{fmt(f_ref['ci_lo'])}, {fmt(f_ref['ci_hi'])}]  "
          f"(N={f_ref['n_plants']} plants, {f_ref['n_convs']} convs)")
        if f_ref["ci_excludes_zero"] and (f_ref["raw_EB_mean"] or 0) > 0:
            p("    -> CARRIES: FRESH referent effect positive, CI excludes 0.")
        elif f_ref["ci_excludes_zero"]:
            p("    -> SIGN FLIP: FRESH CI excludes 0 but is NEGATIVE.")
        else:
            p(f"    -> NULL at FRESH ({f_ref['null_kind']}).")
            # pre-registered stopping rule
            if f_ref["n_convs"] <= (fresh_max - baseline_max):
                p("")
                p("    >>> EXTEND: run c25-c36 for n=24 fresh <<<")
                p("        (pre-registered stopping rule: FRESH referent CI "
                  f"spans zero at n={f_ref['n_convs']} fresh convs.)")

    p("=" * 78)
    return {"floor": floor_info, "results": results,
            "cell_counts": cell_counts}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_json", nargs="+",
                    help="one or more cross_arch_probe result JSONs to POOL")
    ap.add_argument("--baseline-max", type=int, default=12,
                    help="last BASELINE conv index (default 12 -> c01..c12)")
    ap.add_argument("--fresh-max", type=int, default=24,
                    help="last FRESH conv index (default 24 -> c13..c24)")
    ap.add_argument("--competence-k", type=float, default=COMPETENCE_K)
    ap.add_argument("--competence-min-n", type=int, default=COMPETENCE_MIN_N)
    ap.add_argument("--headroom-gate", type=float, default=HEADROOM_GATE)
    ap.add_argument("--morph-word-threshold", type=int,
                    default=MORPH_WORD_THRESHOLD)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--json-out", default=None,
                    help="optional path to also dump the computed stats as JSON")
    args = ap.parse_args(argv)

    if args.baseline_max < 1 or args.fresh_max <= args.baseline_max:
        ap.error("require 1 <= baseline-max < fresh-max")

    summary = run(args.result_json, args.baseline_max, args.fresh_max,
                  args.competence_k, args.competence_min_n, args.headroom_gate,
                  args.morph_word_threshold, args.n_boot, args.seed)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"\n[wrote stats JSON -> {args.json_out}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
