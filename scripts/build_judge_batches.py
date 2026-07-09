#!/usr/bin/env python3
"""Build Sonnet judge batches for the meaning-recovery metric.

Reproduces the (previously ad-hoc, uncommitted) batch construction that produced
results/judge_semantic{,_base}/batch_*.json for c01-c12, so the identical
construction can be applied to held-out convs (c13-c24, c25-c33) for a
comparable judged verdict.

Two sides, DIFFERENT wording (this asymmetry is baked into the c01-c12 metric and
must be preserved):
  GRAFT side (arms E-post-a0.25, E-post-a1.0) -> results/<graft_out>/batch_*.json
  BASE  side (arms A, B)                       -> results/<base_out>/batch_*.json
Categories judged: referent, sense, stance (NOT ruled_out / evicted_fact / strong_prior).
Answer text comes from <raw_dir>/<conv>.json : probes.arms.<arm>.<plant_id>.
Probe + gold come from data/synthetic/<conv>.json : plants[] (single `probe` field).
Item = {"key": "sem|<conv>|<arm>|<plant_id>", "prompt": <preamble>+body}. Keys deduped.

Usage:
  # correctness proof: rebuild c01-c12 from banked raw + diff vs committed batches
  python3 scripts/build_judge_batches.py --validate

  # held-out: build batches to judge
  python3 scripts/build_judge_batches.py --raw-dir results/raw_30b_holdout \
      --convs c13 c14 c15 c16 c17 c18 c19 c20 c21 c22 c23 c24 \
      --graft-out results/judge_semantic_holdout \
      --base-out  results/judge_semantic_base_holdout
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CATEGORIES = ("referent", "sense", "stance")
GRAFT_ARMS = ("E-post-a0.25", "E-post-a1.0")
BASE_ARMS = ("A", "B")

# --- verbatim preambles (extracted from the committed batches) ---------------
GRAFT_PRE = {
    "sense": "The user's question uses a term/reference that is ambiguous on its own but had a specific intended meaning established earlier in the (now-compacted) conversation.",
    "stance": "Earlier in the (now-compacted) conversation the user established a preference/constraint. A good answer should honor that disposition even without restating it.",
    "referent": "The user's question refers to a specific decision/choice/entity established earlier in the (now-compacted) conversation.",
}
BASE_PRE = {
    "sense": "Context: ambiguous term with a specific meaning established earlier (now compacted).",
    "stance": "Context: user established a preference/constraint earlier; a good answer honors it even without restating it.",
    "referent": "Context: refers to a specific decision/entity established earlier (now compacted).",
}
GRAFT_CLOSE = ("Did the assistant's answer correctly recover the intended "
               "meaning/referent/disposition (not necessarily exact words, but "
               "the right sense)? Reply one word: RECOVERED, MISSED, or PARTIAL.")
BASE_CLOSE = ("Did it correctly recover the intended meaning/referent/disposition "
              "(right sense, not exact words)? One word: RECOVERED, MISSED, or PARTIAL.")


def graft_prompt(cat, probe, gold, answer):
    return (f"{GRAFT_PRE[cat]}\n\n"
            f"User's question: {probe}\n"
            f"The correct meaning/answer (ground truth): {gold}\n\n"
            f"Assistant's answer:\n{answer}\n\n"
            f"{GRAFT_CLOSE}")


def base_prompt(cat, probe, gold, answer):
    return (f"{BASE_PRE[cat]}\n\n"
            f"User's question: {probe}\n"
            f"Correct meaning (ground truth): {gold}\n\n"
            f"Assistant's answer:\n{answer}\n\n"
            f"{BASE_CLOSE}")


def load_plants(conv):
    d = json.load(open(REPO / "data" / "synthetic" / f"{conv}.json"))
    return {p["id"]: p for p in (d.get("plants") or [])
            if p.get("category") in CATEGORIES}


def load_answers(raw_dir, conv):
    p = Path(raw_dir) / f"{conv}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return (d.get("probes") or {}).get("arms") or {}


def build(raw_dir, convs):
    """Return (graft_items, base_items) as lists of {key,prompt}, keys deduped."""
    graft, base, seen_g, seen_b = [], [], set(), set()
    missing = []
    for conv in convs:
        plants = load_plants(conv)
        arms = load_answers(raw_dir, conv)
        if arms is None:
            missing.append(conv)
            continue
        for pid, plant in plants.items():
            cat, probe, gold = plant["category"], plant["probe"], plant["gold"]
            for arm in GRAFT_ARMS:
                ans = (arms.get(arm) or {}).get(pid)
                if ans is None:
                    continue
                key = f"sem|{conv}|{arm}|{pid}"
                if key in seen_g:
                    continue
                seen_g.add(key)
                graft.append({"key": key, "prompt": graft_prompt(cat, probe, gold, ans)})
            for arm in BASE_ARMS:
                ans = (arms.get(arm) or {}).get(pid)
                if ans is None:
                    continue
                key = f"sem|{conv}|{arm}|{pid}"
                if key in seen_b:
                    continue
                seen_b.add(key)
                base.append({"key": key, "prompt": base_prompt(cat, probe, gold, ans)})
    return graft, base, missing


def chunk_write(items, outdir, per=64):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(items), per):
        json.dump(items[i:i + per], open(outdir / f"batch_{i//per:02d}.json", "w"),
                  ensure_ascii=False, indent=1)


def load_committed(globpat):
    items = {}
    for f in sorted(glob.glob(str(REPO / globpat))):
        for it in json.load(open(f)):
            items[it["key"]] = it["prompt"]
    return items


def validate():
    """Rebuild c01-c12 from results/raw_30b and diff vs the committed batches."""
    convs = [f"c{n:02d}" for n in range(1, 13)]
    graft, base, missing = build(REPO / "results/raw_30b", convs)
    if missing:
        print(f"  WARN raw missing for: {missing}")
    ok = True
    for side, built, globpat in [
        ("GRAFT", graft, "results/judge_semantic/batch_*.json"),
        ("BASE", base, "results/judge_semantic_base/batch_*.json"),
    ]:
        committed = load_committed(globpat)
        built_d = {it["key"]: it["prompt"] for it in built}
        bk, ck = set(built_d), set(committed)
        only_built, only_comm = bk - ck, ck - bk
        mismatch = [k for k in (bk & ck) if built_d[k] != committed[k]]
        print(f"== {side}: built={len(bk)} committed={len(ck)} "
              f"exact_match={len(bk & ck) - len(mismatch)} "
              f"text_mismatch={len(mismatch)} only_built={len(only_built)} only_committed={len(only_comm)}")
        if only_built:
            print(f"   only_built (sample): {sorted(only_built)[:5]}")
        if only_comm:
            print(f"   only_committed (sample): {sorted(only_comm)[:5]}")
        for k in mismatch[:2]:
            b, c = built_d[k], committed[k]
            i = next((j for j in range(min(len(b), len(c))) if b[j] != c[j]), min(len(b), len(c)))
            print(f"   MISMATCH {k} at char {i}:\n     built: {b[max(0,i-40):i+40]!r}\n     comm : {c[max(0,i-40):i+40]!r}")
        if only_built or only_comm or mismatch:
            ok = False
    print("VALIDATE:", "PASS — builder reproduces committed batches" if ok
          else "FAIL — see diffs above")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--raw-dir")
    ap.add_argument("--convs", nargs="*")
    ap.add_argument("--graft-out")
    ap.add_argument("--base-out")
    a = ap.parse_args()
    if a.validate:
        sys.exit(0 if validate() else 1)
    if not (a.raw_dir and a.convs and a.graft_out and a.base_out):
        ap.error("need --raw-dir --convs --graft-out --base-out (or --validate)")
    graft, base, missing = build(a.raw_dir, a.convs)
    if missing:
        print(f"WARN: no raw answers for {missing}")
    chunk_write(graft, a.graft_out)
    chunk_write(base, a.base_out)
    print(f"GRAFT: {len(graft)} items -> {a.graft_out}")
    print(f"BASE:  {len(base)} items -> {a.base_out}")
    ng = len({i['key'].split('|')[1] for i in graft})
    print(f"convs covered: {ng}  categories: {sorted(CATEGORIES)}  "
          f"graft_arms: {list(GRAFT_ARMS)}  base_arms: {list(BASE_ARMS)}")


if __name__ == "__main__":
    main()
