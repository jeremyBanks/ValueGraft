"""Concept readout (B9): mechanistic test of F1 — does value-grafting actually
INSERT the evicted concept into the model's internal state, or is recovery only
behavioral inference?

For each sense/referent/stance probe we build three arms (identical to
gap_closure_cat.py): A = full context, B = compacted, E = compacted + write-time
value graft. We then feed the probe question and, AT THE PROBE POSITION (the last
token, whose next-token prediction is the model's answer), read the residual
stream at EVERY layer with a lens (logit-lens by default; J-lens optional),
project to the vocabulary, and measure the logprob / rank of the GOLD CONCEPT
TOKEN (the disambiguating keyword the compaction destroyed).

Metric per probe, at the best layer (layer where the concept is most readable
under full context A): concept-token logprob and rank under A / B / E.

Expected if F1 is MECHANISTIC (not just behavioral): on sense/referent the gold
concept is more present under E (grafted) than B (compacted), approaching A;
on stance (where compaction did no damage) the effect is null.

Lens choice (see DECISIONS 07-07 07:09/07:15 and docs/concept-readout-design.md):
  logit-lens = robust, well-understood baseline (this is the primary path);
  J-lens     = principled refinement, hypothesis-GENERATION tool with an
               uncharacterized false-positive rate — offered behind --lens jlens
               for cross-checking, never as the sole measurement.

Env : SC_HF_MODEL (default Qwen/Qwen3-30B-A3B-Instruct-2507)
      SC_CR_ALPHA (graft alpha, default 0.75)
CLI : python src/concept_readout.py [--convs c01 c02 ...] [--limit N]
                                    [--lens logit|jlens] [--out DIR]
Out : results/concept_readout/<conv>.json  + a printed per-category summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, "src")
from arms_common import (  # noqa: E402
    SUMMARY_REQUEST,
    build_alignment,
    build_b_messages,
    canonical_ids,
    message_token_starts,
    render_hf,
)
from arms_hf import generate_summary_hf, hf_prefill_ids  # noqa: E402
from kvlib_hf import blend_values, rebuild_cache  # noqa: E402

MODEL = os.environ.get("SC_HF_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
ALPHA = float(os.environ.get("SC_CR_ALPHA", "0.75"))
CATS = ("sense", "referent", "stance")


# ---------------------------------------------------------------------------
# Lens
# ---------------------------------------------------------------------------
def _final_norm(model):
    """The RMSNorm applied before the unembedding (Qwen3: model.model.norm)."""
    m = getattr(model, "model", model)
    for attr in ("norm", "final_layernorm"):
        n = getattr(m, attr, None)
        if n is not None:
            return n
    raise AttributeError("cannot locate final norm on model")


def logit_lens_layers(model, hidden_states, pos, gold_ids):
    """Logit-lens readout of gold_ids at position `pos`, for every layer.

    hidden_states: tuple(len = n_layers+1) of [B, T, H] (output_hidden_states).
    Applies the model's FINAL norm then the unembedding to each layer's residual
    (nostalgebraist logit-lens, final-norm variant — the robust baseline).

    Returns per-layer (mean-gold-logprob, mean-gold-rank) as two lists indexed by
    transformer block (layer 1..n_layers; the embedding layer 0 is skipped).
    """
    norm = _final_norm(model)
    lm_head = model.get_output_embeddings()
    gold = torch.tensor(gold_ids, device=hidden_states[0].device)
    lps, ranks = [], []
    for li in range(1, len(hidden_states)):
        h = hidden_states[li][0, pos, :]            # [H]
        with torch.no_grad():
            logits = lm_head(norm(h.unsqueeze(0))).float().squeeze(0)  # [V]
        logprobs = torch.log_softmax(logits, dim=-1)
        gl = logprobs[gold]                          # [n_gold]
        # rank of each gold token = # tokens strictly more probable
        rk = (logits.unsqueeze(0) > logits[gold].unsqueeze(1)).sum(dim=1)
        lps.append(float(gl.mean()))
        ranks.append(float(rk.float().mean()))
    return lps, ranks


def jlens_readout_layers(model, tokenizer, feed_ids, pos, gold_ids):
    """Optional J-lens path (hypothesis-generation tool; see module docstring).

    Requires the `jlens` package and a fitted lens for the model. Runs a clean
    forward pass (no cache) over feed_ids and reads gold_ids at `pos` through the
    averaged-Jacobian transport into the final-layer basis before unembed.
    """
    try:
        from jlens import load_lens_for  # type: ignore
        from jlens.hooks import ActivationRecorder  # type: ignore
    except Exception as e:  # pragma: no cover - env dependent
        raise RuntimeError(
            "J-lens requested but the `jlens` package is unavailable. Use the "
            "robust default `--lens logit` for a first pass (see "
            "docs/concept-readout-design.md)."
        ) from e
    lens_model, lens = load_lens_for(model, tokenizer)
    gold = torch.tensor(gold_ids)
    ids = torch.tensor([feed_ids], device=lens_model.input_device)
    layers = list(range(len(lens_model.layers)))
    lps, ranks = [], []
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=layers) as rec:
        lens_model.forward(ids)
        for li in layers:
            residual = rec.activations[li][0, pos].float()
            logits = lens_model.unembed(lens.transport(residual, li)).float()
            logprobs = torch.log_softmax(logits, dim=-1)
            gl = logprobs[gold.to(logits.device)]
            rk = (logits.unsqueeze(0) > logits[gold.to(logits.device)].unsqueeze(1)).sum(dim=1)
            lps.append(float(gl.mean()))
            ranks.append(float(rk.float().mean()))
    return lps, ranks


# ---------------------------------------------------------------------------
# Gold concept tokens
# ---------------------------------------------------------------------------
def gold_concept_tokens(tokenizer, plant):
    """First sub-token of each disambiguating keyword (mid-sentence variant via a
    leading space), plus the first token of the gold phrase as a fallback. These
    are the vocabulary tokens whose presence the graft should restore."""
    toks = []
    for kw in plant.get("keywords", []):
        kw = str(kw).strip()
        if not kw:
            continue
        ids = tokenizer(" " + kw, add_special_tokens=False).input_ids
        if ids:
            toks.append(ids[0])
    gold = str(plant.get("gold", "")).strip()
    if gold:
        ids = tokenizer(gold, add_special_tokens=False).input_ids
        if ids:
            toks.append(ids[0])
    # dedupe, preserve order
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Arm readout
# ---------------------------------------------------------------------------
def arm_lens(model, tokenizer, snap, feed_ids, next_position, gold_ids, lens):
    """Prefill-context snapshot + feed the probe suffix; read gold_ids at the
    LAST fed position (the answer-deciding position) through the lens, per layer.
    Returns (logprobs_by_layer, ranks_by_layer)."""
    if lens == "jlens":
        # J-lens runs its own clean forward (no cache); feed the FULL sequence.
        # `feed_ids` here must be the full context+suffix (handled by caller).
        return jlens_readout_layers(model, tokenizer, feed_ids, len(feed_ids) - 1, gold_ids)
    cache = rebuild_cache(snap, DynamicCache)
    dev = model.device
    pos = torch.arange(next_position, next_position + len(feed_ids), device=dev)[None]
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([feed_ids], device=dev),
            past_key_values=cache,
            position_ids=pos,
            use_cache=True,
            output_hidden_states=True,
        )
    return logit_lens_layers(model, out.hidden_states, -1, gold_ids)


def best_layer_summary(lp_A, lp_B, lp_E, rk_A, rk_B, rk_E):
    """Pick the layer where the concept is most readable under full context A,
    then report all three arms there. Also report the argmax-over-layers-of-E-gain
    layer as a secondary view."""
    L = len(lp_A)
    bl = max(range(L), key=lambda i: lp_A[i])  # best layer chosen on A (unbiased wrt E)
    return {
        "n_layers": L,
        "best_layer": bl,
        "best_layer_frac": round(bl / max(1, L - 1), 3),
        "lp_A": lp_A[bl], "lp_B": lp_B[bl], "lp_E": lp_E[bl],
        "rank_A": rk_A[bl], "rank_B": rk_B[bl], "rank_E": rk_E[bl],
        "E_minus_B": lp_E[bl] - lp_B[bl],
        "A_minus_B": lp_A[bl] - lp_B[bl],
        "gap_closure": ((lp_E[bl] - lp_B[bl]) / (lp_A[bl] - lp_B[bl])
                        if abs(lp_A[bl] - lp_B[bl]) > 1e-6 else None),
    }


def suffix_ids(tokenizer, msgs, probe):
    """Probe question + generation prompt, as ids appended after `msgs`."""
    full = render_hf(tokenizer, msgs + [{"role": "user", "content": probe}], True)
    cn = canonical_ids(tokenizer, msgs, renderer=render_hf)
    return full[len(cn):]


def process_conv(model, tokenizer, c, limit, lens):
    msgs = c["messages"][:-1]
    tsm = c["sections"]["middle_end_msg"]
    ids = canonical_ids(tokenizer, msgs, renderer=render_hf)
    message_token_starts(tokenizer, ids, len(msgs))  # sanity assert

    summ = generate_summary_hf(model, tokenizer, msgs, request=SUMMARY_REQUEST)
    b_msgs = build_b_messages(msgs, summ["text"], tsm)
    b_ids = canonical_ids(tokenizer, b_msgs, renderer=render_hf)
    starts = message_token_starts(tokenizer, ids, len(msgs))
    b_starts = message_token_starts(tokenizer, b_ids, len(b_msgs))
    regions = [((b_starts[2], len(b_ids)), (starts[tsm], summ["conv_end"])),
               ((b_starts[1], b_starts[2]), (summ["s_start"], summ["s_end"]))]
    pairs = build_alignment(b_ids, summ["old_ids"], set(tokenizer.all_special_ids), regions)

    a_snap, _ = hf_prefill_ids(model, ids)
    b_snap, _ = hf_prefill_ids(model, b_ids)
    e_snap = blend_values(b_snap, summ["snapshot"], pairs, ALPHA)

    res = {}
    n_done = 0
    for pl in c["plants"]:
        if pl["category"] not in CATS:
            continue
        if pl.get("contaminated_early") or pl.get("contaminated_tail"):
            continue
        gold_ids = gold_concept_tokens(tokenizer, pl)
        if not gold_ids:
            continue
        sa = suffix_ids(tokenizer, msgs, pl["probe"])
        sb = suffix_ids(tokenizer, b_msgs, pl["probe"])
        if lens == "jlens":
            # J-lens: clean forward over full sequence; A uses full ctx, B/E can't
            # be distinguished by a lens that ignores the cache, so we read A and B
            # only (E requires cache-level intervention -> logit path). Flag it.
            lp_A, rk_A = arm_lens(model, tokenizer, None, ids + sa, None, gold_ids, lens)
            lp_B, rk_B = arm_lens(model, tokenizer, None, b_ids + sb, None, gold_ids, lens)
            lp_E, rk_E = lp_B, rk_B  # placeholder; see design doc caveat
        else:
            lp_A, rk_A = arm_lens(model, tokenizer, a_snap, sa, len(ids), gold_ids, lens)
            lp_B, rk_B = arm_lens(model, tokenizer, b_snap, sb, len(b_ids), gold_ids, lens)
            lp_E, rk_E = arm_lens(model, tokenizer, e_snap, sb, len(b_ids), gold_ids, lens)
        summary = best_layer_summary(lp_A, lp_B, lp_E, rk_A, rk_B, rk_E)
        summary["category"] = pl["category"]
        summary["gold_token_ids"] = gold_ids
        summary["gold_tokens"] = [tokenizer.decode([t]) for t in gold_ids]
        summary["per_layer"] = {"lp_A": lp_A, "lp_B": lp_B, "lp_E": lp_E}
        res[pl["id"]] = summary
        n_done += 1
        if limit and n_done >= limit:
            break

    del a_snap, b_snap, e_snap
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def print_summary(all_res):
    by_cat = {}
    for conv, res in all_res.items():
        for pid, r in res.items():
            by_cat.setdefault(r["category"], []).append(r)
    print("\n=== concept-readout summary (best layer chosen on A) ===")
    print(f"{'category':10} {'n':>3} {'lp_A':>7} {'lp_B':>7} {'lp_E':>7} "
          f"{'E-B':>7} {'A-B':>7} {'%probes E>B':>12}")
    for cat in CATS:
        rows = by_cat.get(cat, [])
        if not rows:
            continue
        n = len(rows)
        m = lambda k: sum(r[k] for r in rows) / n
        pct = 100.0 * sum(1 for r in rows if r["lp_E"] > r["lp_B"]) / n
        print(f"{cat:10} {n:>3} {m('lp_A'):>7.3f} {m('lp_B'):>7.3f} "
              f"{m('lp_E'):>7.3f} {m('E_minus_B'):>7.3f} {m('A_minus_B'):>7.3f} "
              f"{pct:>11.0f}%")
    print("Expected if F1 is mechanistic: sense/referent E>B approaching A; "
          "stance null.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", nargs="*", default=None,
                    help="conversation ids (default: all data/synthetic/c*.json)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max probes per conversation (0 = no limit)")
    ap.add_argument("--lens", choices=["logit", "jlens"], default="logit")
    ap.add_argument("--out", default="results/concept_readout")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path("data/synthetic").glob("c*.json"))
    if args.convs:
        want = set(args.convs)
        paths = [p for p in paths if p.stem in want]

    all_res = {}
    for p in paths:
        c = json.load(open(p))
        cid = c["id"]
        res = process_conv(model, tok, c, args.limit, args.lens)
        json.dump(res, open(out / f"{cid}.json", "w"), indent=1)
        all_res[cid] = res
        print(f"== {cid}: {len(res)} probes", flush=True)

    print_summary(all_res)
    print("CONCEPT_READOUT_DONE", flush=True)


if __name__ == "__main__":
    main()
