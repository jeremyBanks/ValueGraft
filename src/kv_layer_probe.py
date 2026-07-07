"""Per-layer KEY-graft probe (B4++): does ANY single layer benefit from K-graft?

The coarse uniform sweep (src/kv_sweep.py, FINDINGS "Key-grafting does NOT help")
found that grafting KEYS with a UNIFORM alpha_K across all layers does not beat
value-only grafting on referent (the retrieval-hard category). But a single
uniform alpha_K averages over all layers and could MASK a layer-specific effect:
maybe keys help referent only when grafted at one particular layer, and hurt
elsewhere, so the uniform average washes out.

This probe isolates each layer's key contribution. For the 30B path, on the same
sense/referent/stance corpus and against the SAME full (A) / compacted (B)
baselines as kv_sweep, it measures gap-closure under:

  * baseline  v_only        : alpha_K=0,    alpha_V=0.75         (the working ref)
  * per layer v_plus_k_at_L : alpha_V=0.75 EVERYWHERE + alpha_K=0.75 at ONLY L
                              (single-layer keys ON TOP of the working V-graft)
  * per layer k_only_at_L   : alpha_K=0.75 at ONLY L, alpha_V=0  (pure 1-layer K)

gap_closure = (E - B) / (A - B) on mean per-token gold-continuation logprob.

KEY QUESTION: for each layer L, does v_plus_k_at_L LIFT referent gap-closure
ABOVE the v_only baseline (30B referent v_only ~ +0.057)? Any positive
per-layer "lift" is a layer whose keys add referent recovery that the uniform
sweep hid.

Machinery is reused verbatim from kv_sweep: same summary generation, exact-twin
alignment, A/B prefills, teacher-forced gold-continuation logprob (kv_sweep.tf),
same graft() call. The ONLY thing that varies is layer_set on the K-graft.

Efficiency. The value-graft snapshot (V everywhere, alpha_V=0.75) is built ONCE
per conversation. For each sampled layer L we then apply blend_keys at
layer_set={L} on TOP of that shared V-snapshot (blend_keys clones only layer L's
keys and passes every other layer + all grafted values through by reference), so
peak memory stays at A + B + v_snap + one single-layer E snapshot.

Sampling (defaults, override via env):
  * SC_LAYER_STRIDE  (default 4)   -> layers range(0, n_layers, stride).
    Qwen3-30B-A3B has 48 layers -> {0,4,...,44} = 12 layers. Set to 1 for all 48.
  * SC_LAYER_LIST                  -> explicit comma list, overrides stride.
  * SC_PROBE_CATS    (default "sense,referent,stance") -> categories to score.
    Set "referent" to run only the target category (much cheaper).
  * SC_N_LAYERS      (default 48)  -> assumed layer count for the --dry-run
    estimate ONLY (the real run reads it from the snapshot).

Env: SC_HF_MODEL (default Qwen3-30B-A3B-Instruct-2507).
Usage:
  python src/kv_layer_probe.py --dry-run   # no model: print plan + cost estimate
  python src/kv_layer_probe.py             # all data/synthetic/c*.json
  python src/kv_layer_probe.py c01         # one conv (self-test)
Out: results/kv_layer_probe/<conv>.json
"""
import json, os, sys
from pathlib import Path

MODEL = os.environ.get("SC_HF_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
CATS_ALL = ("sense", "referent", "stance")
PROBE_CATS = tuple(c.strip() for c in
                   os.environ.get("SC_PROBE_CATS", ",".join(CATS_ALL)).split(",")
                   if c.strip())
ALPHA_V = 0.75    # working value-graft dose (project standard)
ALPHA_K = 0.75    # single-layer key dose (matches the coarse sweep's K strength)


def sampled_layers(n_layers):
    """Which layers get the single-layer K-graft."""
    explicit = os.environ.get("SC_LAYER_LIST", "").strip()
    if explicit:
        return [int(x) for x in explicit.split(",") if x.strip() != ""]
    stride = int(os.environ.get("SC_LAYER_STRIDE", "4"))
    return list(range(0, n_layers, max(1, stride)))


def _plant_ok(pl):
    """Same usability filter as kv_sweep.process_conv (minus the tokenizer-only
    tgt-length check, which the dry run cannot evaluate)."""
    if pl["category"] not in PROBE_CATS:
        return False
    if pl.get("contaminated_early") or pl.get("contaminated_tail"):
        return False
    return bool(str(pl.get("gold", "")).strip())


# --------------------------------------------------------------------------
# dry run: iteration + cost model, NO model, stdlib only
# --------------------------------------------------------------------------

def dry_run(only=None):
    n_layers = int(os.environ.get("SC_N_LAYERS", "48"))
    layers = sampled_layers(n_layers)
    convs = []
    for p in sorted(Path("data/synthetic").glob("c*.json")):
        c = json.load(open(p))
        if only and c["id"] != only:
            continue
        np_ = sum(1 for pl in c["plants"] if _plant_ok(pl))
        if np_:
            convs.append((c["id"], np_))
    total_plants = sum(n for _, n in convs)
    nL = len(layers)

    # Scored forward passes per conv with np plants:
    #   2 prefills (A, B)
    # + 2*np  baselines (la, lb)
    # + 1*np  v_only score
    # + nL*np v_plus_k_at_L   (one score per layer per plant)
    # + nL*np k_only_at_L
    per_conv_fp = lambda np_: 2 + (2 + 1 + 2 * nL) * np_
    total_fp = sum(per_conv_fp(n) for _, n in convs)
    n_summary_gen = len(convs)   # one summary generation per conv

    print("=== kv_layer_probe DRY RUN (no model) ===")
    print(f"model             : {MODEL}")
    print(f"categories scored : {list(PROBE_CATS)}")
    print(f"assumed n_layers  : {n_layers}  (SC_N_LAYERS; real run reads snapshot)")
    print(f"sampled layers    : {layers}")
    print(f"  -> {nL} layers  x2 conditions (v_plus_k_at_L, k_only_at_L)")
    print(f"alpha_V={ALPHA_V}  alpha_K={ALPHA_K}")
    print(f"conversations     : {len(convs)}")
    print(f"usable plants      (pre-tokenizer filter): {total_plants}")
    for cid, n in convs:
        print(f"    {cid}: {n} plants -> {per_conv_fp(n)} scored forward passes")
    print("--- cost ---")
    print(f"per-conv scored FP = 2 + (2 + 1 + 2*{nL})*np = 2 + {3 + 2 * nL}*np")
    print(f"TOTAL scored forward passes : {total_fp}")
    print(f"TOTAL summary generations   : {n_summary_gen}")
    print(f"(coarse kv_sweep for reference was ~560 scored FP + 12 gens; "
          f"this is ~{total_fp / 560:.1f}x)")
    print("note: real usable-plant count is slightly lower than the number "
          "above\n      (the tgt>=2-token filter needs the tokenizer); "
          "kv_sweep found 67 total\n      (21 referent / 22 sense / 24 stance) "
          "across all 12 convs.")


# --------------------------------------------------------------------------
# real run: per-conversation layer probe (reuses kv_sweep machinery)
# --------------------------------------------------------------------------

def process_conv(model, tok, rope_fn, c):
    """Per-plant, per-layer gap-closure for one conversation, or None if no
    usable probes. A/B baselines and the value-graft snapshot mirror
    kv_sweep.process_conv exactly; we then vary layer_set on the K-graft."""
    import torch
    from transformers import DynamicCache  # noqa: F401 (kept for parity)
    from arms_common import (SUMMARY_REQUEST, build_alignment, build_b_messages,
                             canonical_ids, message_token_starts, render_hf)
    from arms_hf import generate_summary_hf, hf_prefill_ids
    from kv_graft import graft
    from kv_sweep import tf

    msgs = c["messages"][:-1]; tsm = c["sections"]["middle_end_msg"]
    ids = canonical_ids(tok, msgs, renderer=render_hf)
    starts = message_token_starts(tok, ids, len(msgs))
    summ = generate_summary_hf(model, tok, msgs, request=SUMMARY_REQUEST)
    b_msgs = build_b_messages(msgs, summ["text"], tsm)
    b_ids = canonical_ids(tok, b_msgs, renderer=render_hf)
    b_starts = message_token_starts(tok, b_ids, len(b_msgs))
    regions = [((b_starts[2], len(b_ids)), (starts[tsm], summ["conv_end"])),
               ((b_starts[1], b_starts[2]), (summ["s_start"], summ["s_end"]))]
    pairs = build_alignment(b_ids, summ["old_ids"], set(tok.all_special_ids), regions)

    a_snap, _ = hf_prefill_ids(model, ids)
    b_snap, _ = hf_prefill_ids(model, b_ids)
    n_layers = len(b_snap)
    layers = sampled_layers(n_layers)

    # Collect plants + A/B baselines once (shared across all conditions).
    plants = []
    for pl in c["plants"]:
        if pl["category"] not in PROBE_CATS:
            continue
        if pl.get("contaminated_early") or pl.get("contaminated_tail"):
            continue
        gold = str(pl.get("gold", "")).strip()
        if not gold:
            continue
        tgt = tok(gold, add_special_tokens=False).input_ids[:80]
        if len(tgt) < 2:
            continue

        def suffix(mm):
            full = render_hf(tok, mm + [{"role": "user", "content": pl["probe"]}], True)
            cn = canonical_ids(tok, mm, renderer=render_hf)
            return full[len(cn):]

        sb = suffix(b_msgs)
        la = tf(model, a_snap, suffix(msgs) + tgt[:-1], tgt, len(ids))
        lb = tf(model, b_snap, sb + tgt[:-1], tgt, len(b_ids))
        plants.append({"pl": pl, "sb": sb, "tgt": tgt, "la": la, "lb": lb})

    if not plants:
        del a_snap, b_snap
        return None, layers, n_layers

    res = {}
    for pr in plants:
        pl = pr["pl"]
        res[pl["id"]] = {"category": pl["category"], "lp_A": pr["la"],
                         "lp_B": pr["lb"], "layers": {}}

    def gc(le, la, lb):
        return (le - lb) / (la - lb) if abs(la - lb) > 1e-6 else None

    def score(snap, name_le):
        for pr in plants:
            pl = pr["pl"]
            le = tf(model, snap, pr["sb"] + pr["tgt"][:-1], pr["tgt"], len(b_ids))
            yield pl["id"], le, gc(le, pr["la"], pr["lb"])

    # 1) v_only baseline snapshot (values everywhere, no keys). Built ONCE and
    #    reused as the fresh base for every v_plus_k_at_L below.
    v_snap = graft((b_snap, summ["snapshot"]), pairs,
                   alpha_K=0.0, alpha_V=ALPHA_V,
                   positions_fresh=None, positions_write=None, rope_fn=rope_fn)
    for pid, le, g in score(v_snap, "v_only"):
        res[pid]["v_only"] = {"alpha_K": 0.0, "alpha_V": ALPHA_V,
                              "lp_E": le, "gap_closure": g}

    # 2) per layer: v_plus_k_at_L (K on TOP of v_snap) and k_only_at_L (K on B).
    for L in layers:
        vk = graft((v_snap, summ["snapshot"]), pairs,
                   alpha_K=ALPHA_K, alpha_V=0.0, layer_set={L},
                   positions_fresh=None, positions_write=None, rope_fn=rope_fn)
        vk_scores = {pid: (le, g) for pid, le, g in score(vk, "v_plus_k")}
        del vk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ko = graft((b_snap, summ["snapshot"]), pairs,
                   alpha_K=ALPHA_K, alpha_V=0.0, layer_set={L},
                   positions_fresh=None, positions_write=None, rope_fn=rope_fn)
        ko_scores = {pid: (le, g) for pid, le, g in score(ko, "k_only")}
        del ko
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for pid in res:
            vle, vg = vk_scores[pid]; kle, kg = ko_scores[pid]
            res[pid]["layers"][str(L)] = {
                "v_plus_k_at_L": {"lp_E": vle, "gap_closure": vg},
                "k_only_at_L":   {"lp_E": kle, "gap_closure": kg}}

    del v_snap, a_snap, b_snap
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res, layers, n_layers


def summarize(res, layers):
    """category-level v_only mean + per-layer v_plus_k / k_only means and the
    LIFT (v_plus_k mean - v_only mean) that answers the key question."""
    summary = {}
    for cat in PROBE_CATS:
        rows = [r for r in res.values() if r["category"] == cat]
        v_vals = [r["v_only"]["gap_closure"] for r in rows
                  if r.get("v_only", {}).get("gap_closure") is not None]
        v_mean = (sum(v_vals) / len(v_vals)) if v_vals else None
        per_layer = {}
        for L in layers:
            key = str(L)
            vk = [r["layers"][key]["v_plus_k_at_L"]["gap_closure"] for r in rows
                  if r["layers"].get(key, {}).get("v_plus_k_at_L", {})
                  .get("gap_closure") is not None]
            ko = [r["layers"][key]["k_only_at_L"]["gap_closure"] for r in rows
                  if r["layers"].get(key, {}).get("k_only_at_L", {})
                  .get("gap_closure") is not None]
            vk_mean = (sum(vk) / len(vk)) if vk else None
            per_layer[key] = {
                "v_plus_k_mean": vk_mean, "v_plus_k_n": len(vk),
                "k_only_mean": (sum(ko) / len(ko)) if ko else None,
                "k_only_n": len(ko),
                "lift_vs_v_only": (vk_mean - v_mean)
                if (vk_mean is not None and v_mean is not None) else None}
        summary[cat] = {"v_only_mean": v_mean, "v_only_n": len(v_vals),
                        "per_layer": per_layer}
    return summary


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = args[0] if args else None
    if "--dry-run" in sys.argv:
        dry_run(only)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from arms_hf import rope_base
    from kv_graft import make_rope_fn

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto"); model.eval()
    rope_fn = make_rope_fn(rope_base(model), layout="half")
    out = Path("results/kv_layer_probe"); out.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path("data/synthetic").glob("c*.json")):
        c = json.load(open(p)); cid = c["id"]
        if only and cid != only:
            continue
        of = out / f"{cid}.json"
        if of.exists() and not only:
            print(cid, "done"); continue
        res, layers, n_layers = process_conv(model, tok, rope_fn, c)
        if res is None:
            print(f"== {cid}: no usable probes", flush=True); continue
        doc = {"model": MODEL, "alpha_V": ALPHA_V, "alpha_K": ALPHA_K,
               "n_layers": n_layers, "layers": layers,
               "categories": list(PROBE_CATS),
               "conditions": ["v_only", "v_plus_k_at_L", "k_only_at_L"],
               "plants": res, "summary": summarize(res, layers)}
        json.dump(doc, open(of, "w"), indent=1)
        print(f"== {cid}: {len(res)} probes over {len(layers)} layers", flush=True)
    print("KV_LAYER_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
