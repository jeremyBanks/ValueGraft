"""ValueGraft alpha sweep (CONT-only) with gate variants.

For each conversation (synthetic + natural), builds the Arm B context once,
then scores the held-out continuation under E-post grafts across an alpha
grid, plus two cheap adaptive gates at fixed alpha:

  span-gate : per-pair alpha scaled by min(1, run_length/16) — long exact
              spans get full alpha, short/common matches get less.
  mid-band  : alpha applied only in the middle third of layers.
  late-band : alpha applied only in the last third of layers.

Grid via SC_ALPHAS (default fine-low for 4B). Tuning/holdout split is applied
at analysis time (this script just records everything).

Output: results/alpha_sweep_<tag>/<conv>.json
"""

import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "src")
from arms import arm_e_snapshot, build_alignment, build_b_messages, \
    canonical_ids, generate_summary, message_token_starts, render
from kvlib import batched_teacher_forced, prefill, rebuild_cache, \
    snapshot_cache
from run_arms import ArmSet, clear

MODEL = os.environ.get("SC_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
TAG = os.environ.get("SC_SWEEP_TAG", "4b")
ALPHAS = [float(x) for x in os.environ.get(
    "SC_ALPHAS", "0,0.03,0.06,0.10,0.15,0.20,0.25,0.35").split(",")]
GATE_ALPHA = float(os.environ.get("SC_GATE_ALPHA", "0.25"))


def run_lengths(pairs):
    """Length of the contiguous run each pair belongs to."""
    runs = []
    start = 0
    for i in range(1, len(pairs) + 1):
        if i == len(pairs) or pairs[i][0] != pairs[i - 1][0] + 1 \
                or pairs[i][1] != pairs[i - 1][1] + 1:
            runs.append((start, i))
            start = i
    lengths = [0] * len(pairs)
    for s, e in runs:
        for j in range(s, e):
            lengths[j] = e - s
    return lengths


def arm_e_span_gated(b_snap, old_snap, pairs, alpha):
    """E-post with per-pair alpha scaled by exact-run length."""
    lens = run_lengths(pairs)
    new_idx = mx.array([n for n, _ in pairs])
    old_idx = mx.array([o for _, o in pairs])
    w = mx.array([alpha * min(1.0, l / 16.0) for l in lens],
                 dtype=mx.float32)[None, None, :, None]
    out = []
    for li, (k, v, off) in enumerate(b_snap):
        v_old = old_snap[li][1][..., old_idx, :].astype(mx.float32)
        v_fresh = v[..., new_idx, :].astype(mx.float32)
        blended = ((1 - w) * v_fresh + w * v_old).astype(v.dtype)
        v2 = mx.array(v)
        v2[..., new_idx, :] = blended
        out.append((k, v2, off))
    return out


def score(model, tokenizer, snap, feed, targets):
    cache = rebuild_cache(snap)
    lps = batched_teacher_forced(model, cache, feed, targets)
    return sum(lps) / len(lps)


def run_one(model, tokenizer, msgs, tail_start_msg, cont_spec, n_layers):
    aset = ArmSet(model, tokenizer, msgs, tail_start_msg)
    kind, payload = cont_spec
    if kind == "gp":  # synthetic: gen-prompt + final assistant text
        gp = render(tokenizer, msgs, True)
        assert gp[: len(aset.ids)] == aset.ids
        cont_ids = tokenizer.encode(payload, add_special_tokens=False)
        feed = gp[len(aset.ids):] + cont_ids[:-1]
        targets = cont_ids
    else:  # natural: canonical multi-message holdout suffix
        full_canon = canonical_ids(tokenizer, payload)
        assert full_canon[: len(aset.ids)] == aset.ids
        suffix = full_canon[len(aset.ids):]
        feed, targets = suffix[:-1], suffix[1:]
    b_snap, old_snap, pairs = aset.b_snap(), aset.summary["snapshot"], aset.pairs

    out = {}
    for a in ALPHAS:
        snap = b_snap if a == 0 else arm_e_snapshot(b_snap, old_snap, pairs, a)
        out[f"a{a:g}"] = score(model, tokenizer, snap, feed, targets)
    third = n_layers // 3
    out[f"midband_a{GATE_ALPHA}"] = score(
        model, tokenizer,
        arm_e_snapshot(b_snap, old_snap, pairs, GATE_ALPHA,
                       layer_set=set(range(third, 2 * third))),
        feed, targets)
    out[f"lateband_a{GATE_ALPHA}"] = score(
        model, tokenizer,
        arm_e_snapshot(b_snap, old_snap, pairs, GATE_ALPHA,
                       layer_set=set(range(2 * third, n_layers))),
        feed, targets)
    out[f"spangate_a{GATE_ALPHA}"] = score(
        model, tokenizer,
        arm_e_span_gated(b_snap, old_snap, pairs, GATE_ALPHA),
        feed, targets)
    # references
    a_cache, _ = prefill(model, aset.ids)
    out["A"] = score(model, tokenizer, snapshot_cache(a_cache), feed, targets)
    clear(aset, a_cache)
    return out


def main():
    model, tokenizer = load(MODEL)
    n_layers = len(model.layers)
    outdir = Path(f"results/alpha_sweep_{TAG}")
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for p in sorted(Path("data/synthetic").glob("c*.json")):
        c = json.load(open(p))
        jobs.append((c["id"], c["messages"][:-1],
                     c["sections"]["middle_end_msg"],
                     ("gp", c["messages"][-1]["content"])))
    for p in sorted(Path("data/natural").glob("n*.json")):
        c = json.load(open(p))
        msgs = c["messages"][: len(c["messages"]) - c["holdout_msgs"]]
        # natural continuation: first held-out assistant message's content
        jobs.append((c["id"], msgs, None, ("suffix", c["messages"])))
    only = set(sys.argv[1:])
    for cid, msgs, tsm, cont in jobs:
        if only and cid not in only:
            continue
        outfile = outdir / f"{cid}.json"
        if outfile.exists():
            print(f"{cid}: done, skipping")
            continue
        t0 = time.time()
        res = run_one(model, tokenizer, msgs, tsm, cont, n_layers)
        json.dump(res, open(outfile, "w"), indent=1)
        best = max((v, k) for k, v in res.items() if k != "A")
        print(f"== {cid} done in {time.time()-t0:.0f}s  A={res['A']:.4f} "
              f"B(a0)={res['a0']:.4f} best={best[1]}:{best[0]:.4f}")


if __name__ == "__main__":
    main()
