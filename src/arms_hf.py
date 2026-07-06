"""HF transformers arm builders (pod stack). Mirrors arms.py semantics using
kvlib_hf primitives + arms_common token logic."""

import torch
from transformers import DynamicCache

import sys
sys.path.insert(0, "src")
from arms_common import (
    N_SINK,
    SUMMARY_REQUEST,
    SUMMARY_REQUEST_BRIEF,
    build_alignment,
    build_b_messages,
    bmin_pack_ids,
    canonical_ids,
    message_token_starts,
    render_hf,
)
from kvlib_hf import (
    blend_values,
    evict_span,
    greedy_generate,
    prefill,
    rebuild_cache,
    rotate_keys,
    snapshot_cache,
)


def rope_base(model):
    cfg = getattr(model.config, "text_config", model.config)
    rt = getattr(cfg, "rope_theta", None)
    if rt is not None:
        return rt
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict):
        return rp.get("rope_theta") or rp["full_attention"]["rope_theta"]
    raise ValueError("cannot determine rope base; hybrid models need "
                     "per-layer-type handling (see DECISIONS 07-05)")


def to_ids(model, ids):
    return torch.tensor([list(ids)], device=model.device)


def hf_prefill_ids(model, ids):
    cache, logits = prefill(model, to_ids(model, ids))
    return snapshot_cache(cache), logits


def generate_summary_hf(model, tokenizer, msgs, request=None, max_tokens=900, snapshot=True):
    conv_ids = canonical_ids(tokenizer, msgs, renderer=render_hf)
    req_ids = render_hf(
        tokenizer, msgs + [{"role": "user",
                            "content": request or SUMMARY_REQUEST}], True)
    assert req_ids[: len(conv_ids)] == conv_ids
    cache, logits = prefill(model, to_ids(model, req_ids))
    eos = model.config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    toks = greedy_generate(model, cache, logits, max_tokens, eos_ids,
                           next_position=len(req_ids))
    gen_ids = toks[:-1]
    return {
        "text": tokenizer.decode(gen_ids).strip(),
        "gen_ids": gen_ids,
        "conv_end": len(conv_ids),
        "s_start": len(req_ids),
        "s_end": len(req_ids) + len(gen_ids),
        "old_ids": req_ids + gen_ids,
        "snapshot": snapshot_cache(cache),
    }


def arm_h_pack_snapshot_hf(summary, base, pack_offset=None):
    snap = summary["snapshot"]
    s0, s1 = summary["s_start"], summary["s_end"]
    target = N_SINK if pack_offset is None else pack_offset
    delta = target - s0
    out = []
    for k, v in snap:
        ks = rotate_keys(k[..., s0:s1, :], delta, base)
        out.append((torch.cat([k[..., :N_SINK, :], ks], dim=2),
                    torch.cat([v[..., :N_SINK, :], v[..., s0:s1, :]], dim=2)))
    return out


def arm_h_gap_snapshot_hf(summary):
    """Sinks + S entries at ORIGINAL positions (gapped). Storage is
    contiguous; positions of retained entries stay original because keys are
    post-RoPE; new tokens get explicit position_ids continuing from s_end."""
    snap = summary["snapshot"]
    s0, s1 = summary["s_start"], summary["s_end"]
    return [
        (torch.cat([k[..., :N_SINK, :], k[..., s0:s1, :]], dim=2),
         torch.cat([v[..., :N_SINK, :], v[..., s0:s1, :]], dim=2))
        for k, v in snap
    ]


def answer_hf(model, tokenizer, snap, suffix_ids, next_position,
              max_tokens=160):
    """Extend a snapshot with suffix (explicit positions) and greedy-answer."""
    cache = rebuild_cache(snap, DynamicCache)
    dev = model.device
    pos = torch.arange(next_position, next_position + len(suffix_ids),
                       device=dev)[None]
    with torch.no_grad():
        out = model(input_ids=to_ids(model, suffix_ids),
                    past_key_values=cache, position_ids=pos, use_cache=True,
                    logits_to_keep=1)
    eos = model.config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    toks = greedy_generate(model, cache, out.logits[:, -1, :], max_tokens,
                           eos_ids, next_position + len(suffix_ids))
    return tokenizer.decode([t for t in toks if t not in eos_ids]).strip()
