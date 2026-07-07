"""Behavioral K/V-policy sweep (B4+): the paper's central open question.

Value-only grafting (V-Graft, alpha_V=0.75) floored on the "referent" category
(recovering the specific evicted DECISION). This sweeps the GRAFT POLICY over
KEY-grafting (K-Graft, RoPE re-rotated) to test whether adding keys recovers
referent gap-closure where value-only could not.

Identical machinery to gap_closure_cat.py -- same full (A) / compacted (B)
baselines, same exact-twin alignment, same teacher-forced GOLD-continuation
logprob metric, same 30B path (kvlib_hf snapshots). The ONLY thing that varies
is the graft call: kv_graft.graft with each policy's (alpha_K, alpha_V).

gap_closure = (E - B) / (A - B) on mean per-token gold-continuation logprob,
stratified by sense / referent / stance.

Position mapping (the K-Graft crux). Both the fresh (B / b_snap) and the
write-time (summary) snapshots are plain CONTIGUOUS prefills, so a storage
index equals its RoPE position in each. The alignment pairs are
(new_idx_in_b_ids, old_idx_in_old_ids); blend_keys re-rotates each write-time
key by delta = new_pos - old_pos. Hence positions_fresh / positions_write are
both None (identity storage->position). If B ever used gapped/packed retention
this would need explicit position arrays -- it does not here.

The V-only policy (alpha_K=0, alpha_V=0.75) reproduces gap_closure_cat exactly
(graft with alpha_K==0 skips blend_keys and calls kvlib_hf.blend_values with
the same args) -- kept as the control / sanity anchor.

Env: SC_HF_MODEL (default Qwen3-30B-A3B-Instruct-2507).
Usage:
  python src/kv_sweep.py            # sweep all data/synthetic/c*.json
  python src/kv_sweep.py c01        # just one conv (self-test)
Out: results/kv_sweep/<conv>.json
"""
import json, os, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
sys.path.insert(0, "src")
from arms_common import (SUMMARY_REQUEST, build_alignment, build_b_messages,
    canonical_ids, message_token_starts, render_hf)
from arms_hf import generate_summary_hf, hf_prefill_ids, rope_base
from kvlib_hf import rebuild_cache, tf_logprobs, blend_values
from kv_graft import graft, make_rope_fn

MODEL = os.environ.get("SC_HF_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
CATS = ("sense", "referent", "stance")

# Graft policies: name -> (alpha_K, alpha_V). alpha_V=0.75 is the paper's tuned
# value-graft strength; K strengths bracket it.
POLICIES = {
    "v_only":      (0.0,  0.75),   # paper baseline / control (== gap_closure_cat)
    "k_only":      (0.75, 0.0),    # keys only
    "coupled":     (0.75, 0.75),   # keys+values, matched
    "ind_k50":     (0.50, 0.75),   # independent samples
    "ind_v50":     (0.75, 0.50),
    "ind_k100":    (1.0,  0.75),
}


def tf(model, snap, feed, targets, npos):
    cache = rebuild_cache(snap, DynamicCache)
    pos = torch.arange(npos, npos + len(feed), device=model.device)[None]
    return sum(tf_logprobs(model, cache, feed, targets, position_ids=pos)) / max(1, len(targets))


def process_conv(model, tok, rope_fn, c):
    """Return per-plant results dict for one conversation, or None if no usable
    probes. Mirrors gap_closure_cat.main's per-conv body; sweeps POLICIES."""
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

    # Collect the plants and their A/B baselines once (shared across policies).
    plants = []
    for pl in c["plants"]:
        if pl["category"] not in CATS:
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

        sa = suffix(msgs); sb = suffix(b_msgs)
        la = tf(model, a_snap, sa + tgt[:-1], tgt, len(ids))
        lb = tf(model, b_snap, sb + tgt[:-1], tgt, len(b_ids))
        plants.append({"pl": pl, "sb": sb, "tgt": tgt, "la": la, "lb": lb})

    if not plants:
        del a_snap, b_snap
        return None

    # init result skeleton
    res = {}
    for pr in plants:
        pl = pr["pl"]
        res[pl["id"]] = {"category": pl["category"], "lp_A": pr["la"],
                         "lp_B": pr["lb"], "policies": {}}

    # Sweep policies: build one grafted snapshot per policy, score all plants,
    # then free it (keeps peak memory to A + B + one E snapshot).
    for name, (aK, aV) in POLICIES.items():
        e_snap = graft((b_snap, summ["snapshot"]), pairs,
                       alpha_K=aK, alpha_V=aV,
                       positions_fresh=None, positions_write=None,
                       rope_fn=rope_fn)
        for pr in plants:
            pl = pr["pl"]; la = pr["la"]; lb = pr["lb"]
            le = tf(model, e_snap, pr["sb"] + pr["tgt"][:-1], pr["tgt"], len(b_ids))
            gc = (le - lb) / (la - lb) if abs(la - lb) > 1e-6 else None
            res[pl["id"]]["policies"][name] = {
                "alpha_K": aK, "alpha_V": aV, "lp_E": le, "gap_closure": gc}
        del e_snap
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del a_snap, b_snap
    return res


def summarize(res):
    """category x policy -> {mean_gap_closure, n} over probes with finite gc."""
    summary = {}
    for cat in CATS:
        summary[cat] = {}
        for name in POLICIES:
            vals = [r["policies"][name]["gap_closure"] for r in res.values()
                    if r["category"] == cat
                    and r["policies"].get(name, {}).get("gap_closure") is not None]
            summary[cat][name] = {
                "mean_gap_closure": (sum(vals) / len(vals)) if vals else None,
                "n": len(vals)}
    return summary


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto"); model.eval()
    rope_fn = make_rope_fn(rope_base(model), layout="half")
    out = Path("results/kv_sweep"); out.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path("data/synthetic").glob("c*.json")):
        c = json.load(open(p)); cid = c["id"]
        if only and cid != only:
            continue
        of = out / f"{cid}.json"
        if of.exists() and not only:
            print(cid, "done"); continue
        res = process_conv(model, tok, rope_fn, c)
        if res is None:
            print(f"== {cid}: no usable probes", flush=True); continue
        doc = {"model": MODEL, "policies": {k: {"alpha_K": v[0], "alpha_V": v[1]}
                                            for k, v in POLICIES.items()},
               "plants": res, "summary": summarize(res)}
        json.dump(doc, open(of, "w"), indent=1)
        print(f"== {cid}: {len(res)} probes over {len(POLICIES)} policies", flush=True)
    print("KV_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
