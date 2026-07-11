"""Frozen analysis for COHERENT-STATE-PREREGISTRATION.md.

Input is one committed JSON checkpoint per conversation. The independent unit is
the conversation; this module never expands plants into pseudo-replicates.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import random
import statistics


ARMS = ("A_full", "F_fresh", "C_coherent", "W_wrong",
        "V_only", "K_only", "D_delta")
CONTRASTS = {
    "CF": ("C_coherent", "F_fresh"),
    "CW": ("C_coherent", "W_wrong"),
    "VF": ("V_only", "F_fresh"),
    "KF": ("K_only", "F_fresh"),
    "CV": ("C_coherent", "V_only"),
    "CK": ("C_coherent", "K_only"),
    "VD": ("V_only", "D_delta"),
    "AF": ("A_full", "F_fresh"),
}

# Two-sided 95% Student-t critical values by degrees of freedom. The frozen
# experiment ends at n=12, but values through 30 keep fixtures/use explicit.
T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


class AnalysisError(RuntimeError):
    pass


def mean(values):
    return sum(values) / len(values) if values else None


def t_interval(values):
    vals = [float(v) for v in values]
    if not vals:
        return None
    m = mean(vals)
    if len(vals) == 1:
        return {"n": 1, "mean": m, "lo": None, "hi": None,
                "method": "Student-t unavailable at n=1"}
    df = len(vals) - 1
    crit = T975.get(df, 1.96)
    half = crit * statistics.stdev(vals) / math.sqrt(len(vals))
    return {"n": len(vals), "mean": m, "lo": m - half, "hi": m + half,
            "method": "two-sided 95% Student-t"}


def bootstrap_interval(values, *, n_boot=10_000, seed=20_260_711):
    vals = [float(v) for v in values]
    if not vals:
        return None
    rng = random.Random(seed)
    n = len(vals)
    boots = sorted(mean([vals[rng.randrange(n)] for _ in range(n)])
                   for _ in range(n_boot))
    lo = boots[int(0.025 * n_boot)]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    return {"n": n, "mean": mean(vals), "lo": lo, "hi": hi,
            "n_boot": n_boot, "seed": seed,
            "method": "conversation bootstrap percentile 95%"}


def load_checkpoints(run_dir: Path):
    docs = []
    for path in sorted(glob.glob(str(run_dir / "conv_*.json"))):
        with open(path) as f:
            doc = json.load(f)
        if doc.get("status") != "scored":
            continue
        doc["_path"] = path
        docs.append(doc)
    docs.sort(key=lambda d: int(d["order_position"]))
    if len({d["conversation_id"] for d in docs}) != len(docs):
        raise AnalysisError("duplicate conversation IDs")
    return docs


def validate_docs(docs):
    for expected, doc in enumerate(docs, 1):
        if int(doc.get("order_position", -1)) != expected:
            raise AnalysisError(
                f"non-contiguous frozen order at {doc.get('_path')}: "
                f"got {doc.get('order_position')} expected {expected}")
        outcomes = doc.get("conversation_outcomes") or {}
        missing = [arm for arm in ARMS if arm not in outcomes]
        if missing:
            raise AnalysisError(f"{doc.get('_path')} missing arms {missing}")
        if not all(math.isfinite(float(outcomes[a])) for a in ARMS):
            raise AnalysisError(f"{doc.get('_path')} has non-finite outcome")


def contrast_rows(docs):
    out = {name: [] for name in CONTRASTS}
    for doc in docs:
        ys = doc["conversation_outcomes"]
        for name, (lhs, rhs) in CONTRASTS.items():
            out[name].append(float(ys[lhs]) - float(ys[rhs]))
    return out


def calibration_fires(docs):
    rows = []
    for doc in docs:
        cal = doc.get("calibration_outcomes") or {}
        if not all(a in cal for a in ("C_coherent", "F_fresh", "W_wrong")):
            return {"fires": False, "reason": "missing calibration rows",
                    "both_directional": 0, "n": len(docs)}
        cf = float(cal["C_coherent"]) - float(cal["F_fresh"])
        cw = float(cal["C_coherent"]) - float(cal["W_wrong"])
        rows.append((cf, cw))
    both = sum(cf > 0 and cw > 0 for cf, cw in rows)
    mcf, mcw = mean([x[0] for x in rows]), mean([x[1] for x in rows])
    return {"fires": bool(mcf > 0 and mcw > 0 and both >= 4),
            "mean_CF": mcf, "mean_CW": mcw,
            "both_directional": both, "n": len(rows)}


def regime_gate(docs):
    ys_a = [float(d["conversation_outcomes"]["A_full"]) for d in docs]
    af = [float(d["conversation_outcomes"]["A_full"]) -
          float(d["conversation_outcomes"]["F_fresh"]) for d in docs]
    competence_count = sum(x > 0 for x in ys_a)
    headroom_count = sum(x > 0 for x in af)
    return {
        "competence_count_positive": competence_count,
        "headroom_count_positive": headroom_count,
        "mean_A": mean(ys_a),
        "mean_AF": mean(af),
        "passes": bool(competence_count >= 4 and mean(ys_a) > 0 and
                       headroom_count >= 4 and mean(af) >= 0.30),
    }


def technical_gate(docs):
    failed = []
    for doc in docs:
        gates = doc.get("gates") or {}
        if not gates.get("technical_pass", False):
            failed.append({"conversation_id": doc.get("conversation_id"),
                           "failures": gates.get("failures") or ["unspecified"]})
    return {"passes": not failed, "failed": failed}


def interpret(stats):
    cf, cw = stats["contrasts"]["CF"]["t"], stats["contrasts"]["CW"]["t"]
    vf, vd = stats["contrasts"]["VF"]["t"], stats["contrasts"]["VD"]["t"]
    cv, kf = stats["contrasts"]["CV"]["t"], stats["contrasts"]["KF"]["t"]
    clears = lambda x: x and x["lo"] is not None and x["lo"] > 0
    if not stats["technical_gate"]["passes"]:
        return "VOID_TECHNICAL"
    if not stats["regime_gate"]["passes"]:
        return "REGIME_INADEQUATE"
    channel = clears(cf) and clears(cw)
    if channel and clears(vf) and clears(vd):
        return "HISTORY_CHANNEL_AND_VALUE_ONLY_GAIN"
    if channel and not clears(vf) and clears(cv):
        return "HISTORY_CHANNEL_KV_SPLIT_LOSES_IT"
    if channel and clears(kf) and not clears(vf):
        return "HISTORY_CHANNEL_KEY_DOMINANT_LEAD"
    if clears(vf) and not clears(vd):
        return "VALUE_GAIN_NOT_BEYOND_MATCHED_PLACEBO"
    if channel:
        return "HISTORY_SPECIFIC_CHANNEL_MECHANISM_MIXED"
    if clears(cf) and not clears(cw):
        return "COHERENCE_WITHOUT_HISTORY_SPECIFICITY"
    return "NO_CONFIRMATORY_CHANNEL"


def analyze(docs):
    validate_docs(docs)
    rows = contrast_rows(docs)
    technical = technical_gate(docs)
    # The regime gate is frozen at the first six conversations. At N=12 it must
    # not be recomputed over the extension, which could reverse the preregistered
    # stage-one decision after treatment outcomes were observed.
    regime_docs = docs[:6]
    regime = regime_gate(regime_docs) if regime_docs else {"passes": False}
    regime["frozen_at_n"] = len(regime_docs)
    calibration = calibration_fires(docs)
    contrasts = {
        name: {"rows": vals, "t": t_interval(vals),
               "bootstrap": bootstrap_interval(vals)}
        for name, vals in rows.items()
    }
    result = {
        "schema": 1,
        "n_conversations": len(docs),
        "conversation_ids": [d["conversation_id"] for d in docs],
        "technical_gate": technical,
        "regime_gate": regime,
        "calibration": calibration,
        "contrasts": contrasts,
    }
    if len(docs) == 6:
        cf, cw = contrasts["CF"]["t"]["mean"], contrasts["CW"]["t"]["mean"]
        if not technical["passes"]:
            decision = "STOP_TECHNICAL"
        elif not regime["passes"]:
            decision = "STOP_REGIME"
        elif cf <= 0 and cw <= 0 and not calibration["fires"]:
            decision = "STOP_FUTILITY"
        else:
            decision = "EXTEND_TO_12"
        result["serial_decision"] = decision
    elif len(docs) == 12:
        result["serial_decision"] = "FINAL_N12"
    else:
        result["serial_decision"] = "INTERIM_NOT_ACTIONABLE"
    result["interpretation"] = interpret(result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--output", type=Path,
                    help="optional explicit unique output path")
    args = ap.parse_args()
    docs = load_checkpoints(args.run_dir)
    result = analyze(docs)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.write_text(text)
        print(f"WROTE {args.output}")
    print(text, end="")


if __name__ == "__main__":
    main()
