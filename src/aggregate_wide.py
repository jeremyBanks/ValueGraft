#!/usr/bin/env python3
"""Aggregate the wide cross-arch sweep into the scientific picture:
per-model referent SIGN, the dissociation, QK-norm (H1) prediction test, and the
within-vendor dense/MoE de-confound pair check. Reads results/cross_arch_wide/*.json
(harvested from the pods) + data/model_geometry.json. Prints a human summary used to
compose the significance notifications. Pure stdlib."""
import glob
import json
import os

WIDE = "results/cross_arch_wide"
# the two pre-registered de-confound pairs (MoE, dense)
PAIRS = [
    ("Qwen__Qwen3-30B-A3B-Instruct-2507", "Qwen__Qwen3-32B", "Qwen3"),
    ("google__gemma-4-26B-A4B-it", "google__gemma-4-31B-it", "Gemma-4"),
]


def _sign(ci):
    if not ci or None in ci:
        return "no-CI"
    lo, hi = ci
    if lo > 0:
        return "POS"
    if hi < 0:
        return "NEG"
    return "null(spans0)"


def load():
    rows = {}
    for f in glob.glob(f"{WIDE}/*.json"):
        if os.path.basename(f).startswith("_"):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        slug = os.path.basename(f)[:-5]
        bcr = d.get("by_category_robust") or {}
        ref = bcr.get("referent") or {}
        hp = d.get("model_hparams") or {}
        rows[slug] = {
            "status": d.get("status"),
            "ref_ci": ref.get("raw_EB_ci"),
            "ref_sign": _sign(ref.get("raw_EB_ci")),
            "sense": (bcr.get("sense") or {}).get("raw_EB_ci"),
            "stance": (bcr.get("stance") or {}).get("raw_EB_ci"),
            "qk_norm": hp.get("qk_norm"),
            "qk_src": hp.get("qk_norm_source"),
            "gqa": hp.get("gqa_ratio"),
            "head_dim": hp.get("head_dim"),
            "reason": d.get("reason"),
        }
    return rows


def main():
    rows = load()
    print(f"=== WIDE SWEEP: {len(rows)} model results ===")
    ok = {k: v for k, v in rows.items() if v["status"] == "OK"}
    for slug, v in sorted(rows.items()):
        if v["status"] != "OK":
            print(f"  {slug:44s} status={v['status']} {str(v['reason'])[:50]}")
            continue
        print(f"  {slug:44s} referent {v['ref_sign']:12s} "
              f"qk_norm={v['qk_norm']}({v['qk_src']}) gqa={v['gqa']}")

    # H1: QK-norm presence -> positive referent sign
    print("\n=== H1 (pre-registered): QK-norm present -> POSITIVE referent sign ===")
    hits = miss = 0
    for slug, v in ok.items():
        if v["qk_norm"] is None or v["ref_sign"] in ("no-CI",):
            continue
        predicted_pos = bool(v["qk_norm"])
        is_pos = v["ref_sign"] == "POS"
        is_neg = v["ref_sign"] == "NEG"
        if is_pos and predicted_pos:
            hits += 1
        elif is_neg and not predicted_pos:
            hits += 1
        elif (is_pos and not predicted_pos) or (is_neg and predicted_pos):
            miss += 1
    print(f"  consistent={hits} contradicting={miss} "
          f"(null/spans-0 models are inconclusive-per-model, not counted)")

    # de-confound pairs: sign should NOT flip within a pair
    print("\n=== De-confound pairs (sign should HOLD within pair = geometry not FFN) ===")
    for moe, dense, name in PAIRS:
        a, b = ok.get(moe), ok.get(dense)
        if not a or not b:
            print(f"  {name}: incomplete ({'MoE ok' if a else 'MoE pending'}, "
                  f"{'dense ok' if b else 'dense pending'})")
            continue
        flip = (a["ref_sign"] == "POS" and b["ref_sign"] == "NEG") or \
               (a["ref_sign"] == "NEG" and b["ref_sign"] == "POS")
        print(f"  {name}: MoE={a['ref_sign']} dense={b['ref_sign']} -> "
              f"{'SIGN FLIP (falsifies geometry-not-FFN!)' if flip else 'holds'}")


if __name__ == "__main__":
    main()
