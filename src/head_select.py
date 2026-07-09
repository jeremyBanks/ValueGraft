"""PER-HEAD CHAMPION SELECTION (+ combination with the per-layer champion).

DERIVE side of the per-head champion scan. The single-slot per-(layer, kv-head)
VAL profile itself is produced by `run_tune_hf.py` PHASE=head (blend ONE (layer,
head) slot at alpha=1, teacher-force the held-out continuation, score minus B),
which writes results/tune_head_<tag>/<conv>.json with keys {"B", "L<l>H<h>":...}
for every VAL conversation. This script only RANKS + SELECTS from that profile
and emits champion configs. Selection uses VAL ONLY (never the validation set).

Emits three head_map champion configs (schema in cross_arch_probe.load_champion_
graft_cfg — {"alpha": a, "head_map": {layer: [kv_head,...]}}):

  heads_<tag>.json          the per-head champion: every (layer, kv-head) slot
                            whose MEAN VAL marginal dEB > 0 (default rule), at
                            alpha=SC_HEAD_ALPHA (default 1.0).
  heads_x_layers_<tag>_intersection.json
                            per-head slots RESTRICTED to layers the per-layer
                            champion also selected (both families agree the layer
                            carries content) -- the conservative combination.
  heads_x_layers_<tag>_union.json
                            per-head slots UNION every head of every per-layer-
                            champion layer -- the inclusive combination.

Both combinations are head_map form at a single uniform alpha (SC_HEAD_ALPHA);
the head_map schema cannot also carry the per-layer 0.75/1.0 alpha-map, so the
combined configs graft the union/intersection SLOTS at that one alpha. The
per-layer champion is validated separately (its own alpha-map) for the side-by-
side. This is an honest simplification, documented in the notes.

OVERFITTING GUARD: exactly ONE pre-registered selection rule is applied on VAL,
producing exactly the configs above; each is then validated ONCE on the disjoint
held-out set. The selection budget is therefore the small fixed set of configs
here (heads / intersection / union), NOT the 192 single-slot comparisons -- those
only RANK; the arbiter is the single held-out placebo-controlled CI per config.

Env:
  SC_TUNE_TAG    profile tag (default 30b_bf16) -> reads results/tune_head_<tag>/
  SC_HEAD_ALPHA  graft alpha for the emitted head_map configs (default 1.0)
  SC_HEAD_SELECT selection rule: "positive" (default; mean marginal > 0),
                 "topn" (top SC_HEAD_TOPN slots by mean marginal),
                 "topfrac" (top SC_HEAD_FRAC fraction of slots)
  SC_HEAD_TOPN   int, used by rule=topn (default 40)
  SC_HEAD_FRAC   float in (0,1], used by rule=topfrac (default 0.25)
  SC_LAYERS_CONFIG  per-layer champion json for the combination
                    (default data/champion_configs/layers_<tag>.json)
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

SLOT_RE = re.compile(r"^L(\d+)H(\d+)$")


def load_profile(tag):
    """Mean VAL marginal dEB per (layer, head) slot + support count.

    Returns (marg, support, n_convs, n_kv) where marg[(l,h)] is the mean over VAL
    conversations of score(graft slot (l,h) alone @a=1) - B, and support[(l,h)]
    is the number of VAL convs in which that slot's marginal was > 0.
    """
    files = sorted(glob.glob(f"results/tune_head_{tag}/*.json"))
    assert files, (f"no per-head profile in results/tune_head_{tag}/ -- run "
                   f"run_tune_hf.py SC_TUNE_PHASE=head SC_TUNE_TAG={tag} first")
    vals = defaultdict(list)
    for f in files:
        d = json.load(open(f))
        B = d["B"]
        for k, v in d.items():
            m = SLOT_RE.match(k)
            if m:
                vals[(int(m.group(1)), int(m.group(2)))].append(v - B)
    marg = {s: sum(xs) / len(xs) for s, xs in vals.items()}
    support = {s: sum(1 for x in xs if x > 0) for s, xs in vals.items()}
    n_kv = max(h for _, h in marg) + 1
    return marg, support, len(files), n_kv


def select_slots(marg):
    """Apply the single pre-registered selection rule -> set of (layer, head)."""
    rule = os.environ.get("SC_HEAD_SELECT", "positive")
    if rule == "positive":
        return {s for s, m in marg.items() if m > 0.0}
    ranked = sorted(marg, key=lambda s: marg[s], reverse=True)
    ranked = [s for s in ranked if marg[s] > 0.0]  # never graft a hurting slot
    if rule == "topn":
        n = int(os.environ.get("SC_HEAD_TOPN", "40"))
        return set(ranked[:n])
    if rule == "topfrac":
        frac = float(os.environ.get("SC_HEAD_FRAC", "0.25"))
        k = max(1, round(frac * len(marg)))
        return set(ranked[:k])
    raise ValueError(f"unknown SC_HEAD_SELECT={rule!r}")


def head_map_of(slots):
    """{(l,h),...} -> {layer: sorted [heads]} (JSON-ready with str keys later)."""
    hm = defaultdict(list)
    for l, h in sorted(slots):
        hm[l].append(h)
    return {l: sorted(hs) for l, hs in hm.items()}


def load_layer_set(path):
    if not Path(path).exists():
        return None
    cfg = json.loads(Path(path).read_text())
    amap = cfg.get("alpha_map") or {}
    return {int(k) for k, v in amap.items() if float(v) != 0.0}


def write_cfg(path, label, alpha, hm, note):
    out = {"label": label, "note": note, "alpha": alpha,
           "head_map": {str(l): hs for l, hs in sorted(hm.items())}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    n_slots = sum(len(hs) for hs in hm.values())
    print(f"WROTE {path}: {len(hm)} layers, {n_slots} (layer,head) slots @a={alpha}")
    return n_slots


def main():
    tag = os.environ.get("SC_TUNE_TAG", "30b_bf16")
    alpha = float(os.environ.get("SC_HEAD_ALPHA", "1.0"))
    rule = os.environ.get("SC_HEAD_SELECT", "positive")
    layers_cfg = os.environ.get(
        "SC_LAYERS_CONFIG", f"data/champion_configs/layers_{tag}.json")

    marg, support, n_convs, n_kv = load_profile(tag)
    n_slots_total = len(marg)
    print(f"per-head profile: {n_slots_total} slots ({n_slots_total // n_kv} "
          f"layers x {n_kv} kv-heads) over {n_convs} VAL convs; rule={rule}")

    slots = select_slots(marg)
    assert slots, "selection produced no slots -- no positive-marginal heads?"
    head_hm = head_map_of(slots)

    top = sorted(slots, key=lambda s: marg[s], reverse=True)[:12]
    print("top slots by VAL marginal dEB:")
    for l, h in top:
        print(f"    L{l}H{h}: mean={marg[(l, h)]:+.4f} "
              f"support={support[(l, h)]}/{n_convs}")

    # --- per-head champion
    write_cfg(f"data/champion_configs/heads_{tag}.json",
              f"heads_{tag}", alpha, head_hm,
              (f"PER-HEAD value-graft champion: (layer,kv-head) slots with mean "
               f"VAL marginal dEB>0 (rule={rule}), derived from the {tag} per-head "
               f"profile (results/tune_head_{tag}); {sum(len(v) for v in head_hm.values())} "
               f"slots @a={alpha}. Selected on VAL (c01-c06,n01-n04) ONLY."))

    # --- combinations with the per-layer champion
    layer_set = load_layer_set(layers_cfg)
    if layer_set is None:
        print(f"NOTE: per-layer config {layers_cfg} absent -- combinations skipped "
              f"(the head-scan job builds/points to it before this stage).")
        return

    inter_hm = {l: hs for l, hs in head_hm.items() if l in layer_set}
    if inter_hm:
        write_cfg(f"data/champion_configs/heads_x_layers_{tag}_intersection.json",
                  f"heads_x_layers_{tag}_intersection", alpha, inter_hm,
                  (f"INTERSECTION combination: per-head slots restricted to the "
                   f"{len(layer_set)} layers the per-layer champion ({layers_cfg}) "
                   f"also selected; head_map @a={alpha}. Conservative combined "
                   f"champion. Selected on VAL only."))
    else:
        print("WARN: intersection empty (no per-head slot in a per-layer layer)")

    union_hm = {l: list(hs) for l, hs in head_hm.items()}
    for l in layer_set:
        union_hm[l] = list(range(n_kv))  # per-layer champion = all heads of layer
    union_hm = {l: sorted(set(hs)) for l, hs in union_hm.items()}
    write_cfg(f"data/champion_configs/heads_x_layers_{tag}_union.json",
              f"heads_x_layers_{tag}_union", alpha, union_hm,
              (f"UNION combination: per-head slots UNION every kv-head of every "
               f"per-layer champion layer ({layers_cfg}); head_map @a={alpha}. "
               f"Inclusive combined champion. NB: uniform alpha (head_map schema "
               f"cannot also carry the per-layer 0.75/1.0 map). Selected on VAL only."))


if __name__ == "__main__":
    main()
