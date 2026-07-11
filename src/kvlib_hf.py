"""HuggingFace transformers port of the cache-surgery core (for cloud GPUs).

Mirrors kvlib.py semantics on transformers' DynamicCache:
  - prefill / snapshot / rebuild round-trip
  - gapped retention (position counter decoupled from storage) via explicit
    position_ids + a full attention mask over stored entries
  - value blending at aligned positions (E-post)
  - key re-rotation for packed retention (H-pack)

Verified facts to re-establish per model (l_hf_ladder.py): keys stored
post-RoPE; round-trip identity; null-surgery identity; rotation identity.

transformers>=5 DynamicCache stores per-layer tensors in
`cache.layers[i].keys/values` of shape [B, n_kv_heads, T, head_dim]
(fallback: legacy `key_cache`/`value_cache` lists).
"""

from __future__ import annotations

import torch


def _layers(cache):
    if hasattr(cache, "layers"):
        return cache.layers
    return None


def snapshot_cache(cache):
    """-> list of (K, V) tensors, cloned."""
    out = []
    ls = _layers(cache)
    if ls is not None:
        for l in ls:
            out.append((l.keys.clone(), l.values.clone()))
    else:
        for k, v in zip(cache.key_cache, cache.value_cache):
            out.append((k.clone(), v.clone()))
    return out


def rebuild_cache(snap, cache_cls, *, clone=True):
    cache = cache_cls()
    for i, (k, v) in enumerate(snap):
        cache.update(k.clone() if clone else k,
                     v.clone() if clone else v, i)
    return cache


PREFILL_CHUNK = 4096


def prefill(model, input_ids, past=None, position_ids=None, attention_mask=None,
            cache_position=None):
    """Forward pass building/extending a cache; returns (cache, last_logits).
    Long inputs are fed in PREFILL_CHUNK pieces to bound activation memory.

    Query partitioning is an execution variable, not a general identity: the
    calling assay must separately gate the exact backend, dtype, model, lengths,
    and partitions it relies on.
    """
    n = input_ids.shape[1]
    if n <= PREFILL_CHUNK or attention_mask is not None:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                past_key_values=past,
                position_ids=position_ids,
                cache_position=cache_position,
                attention_mask=attention_mask,
                use_cache=True,
                logits_to_keep=1,
            )
        return out.past_key_values, out.logits[:, -1, :]
    cache = past
    logits = None
    for lo in range(0, n, PREFILL_CHUNK):
        hi = min(lo + PREFILL_CHUNK, n)
        pos = position_ids[:, lo:hi] if position_ids is not None else None
        cpos = cache_position[lo:hi] if cache_position is not None else None
        with torch.no_grad():
            out = model(
                input_ids=input_ids[:, lo:hi],
                past_key_values=cache,
                position_ids=pos,
                cache_position=cpos,
                use_cache=True,
                logits_to_keep=1,
            )
        cache, logits = out.past_key_values, out.logits[:, -1, :]
    return cache, logits


def greedy_generate(model, cache, first_logits, max_tokens, eos_ids,
                    next_position, temperature=0.0, top_p=1.0, seed=None):
    """Decode with explicit position ids. temperature==0 -> greedy;
    otherwise seeded nucleus sampling (Qwen3 card recommends ~0.7/0.8)."""
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    toks = []
    logits = first_logits
    pos = next_position
    for _ in range(max_tokens):
        if temperature and temperature > 0:
            probs = torch.softmax(logits.float() / temperature, dim=-1)[0]
            sp, si = torch.sort(probs, descending=True)
            keep = torch.cumsum(sp, 0) - sp < top_p
            keep[0] = True
            sp, si = sp[keep], si[keep]
            t = int(si[torch.multinomial(sp.cpu() / sp.sum().cpu(), 1,
                                         generator=gen)].item())
        else:
            t = int(torch.argmax(logits, dim=-1).item())
        toks.append(t)
        if t in eos_ids:
            break
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([[t]], device=model.device),
                past_key_values=cache,
                position_ids=torch.tensor([[pos]], device=model.device),
                use_cache=True,
            )
        logits = out.logits[:, -1, :]
        pos += 1
    return toks


def evict_span(snap, start, end):
    """Remove [start, end) from every layer's K/V (gapped retention)."""
    return [
        (torch.cat([k[..., :start, :], k[..., end:, :]], dim=2),
         torch.cat([v[..., :start, :], v[..., end:, :]], dim=2))
        for k, v in snap
    ]


def blend_values(b_snap, old_snap, pairs, alpha, layer_set=None,
                 head_map=None):
    """E-post: V[new_idx] <- (1-a) V_fresh + a V_old[old_idx].
    alpha: float, or dict {layer_idx: alpha} (missing layers untouched).
    head_map: {layer_idx: [kv_head,...]} restricting which heads blend."""
    new_idx = torch.tensor([n for n, _ in pairs])
    old_idx = torch.tensor([o for _, o in pairs])
    per_layer = isinstance(alpha, dict)
    out = []
    for li, (k, v) in enumerate(b_snap):
        a = alpha.get(li, 0.0) if per_layer else alpha
        heads = head_map.get(li) if head_map else None
        if (layer_set is not None and li not in layer_set) or                 (per_layer and abs(a) < 1e-6) or                 (head_map is not None and not heads):
            out.append((k, v))
            continue
        v2 = v.clone()
        vf = v2[..., new_idx, :].float()
        vo = old_snap[li][1][..., old_idx, :].float()
        mixed = ((1 - a) * vf + a * vo).to(v2.dtype)
        if heads is None:
            v2[..., new_idx, :] = mixed
        else:
            for h in heads:
                v2[:, h, new_idx, :] = mixed[:, h]
        out.append((k, v2))
    return out


def rope_thetas(head_dim, base, device):
    i = torch.arange(0, head_dim // 2, dtype=torch.float32, device=device)
    return base ** (-2.0 * i / head_dim)


def rotate_keys(keys, delta, base):
    """Rotate post-RoPE keys by delta positions (half-split layout, matches
    Llama/Qwen rotate_half convention)."""
    B, H, L, D = keys.shape
    half = D // 2
    th = rope_thetas(D, base, keys.device)
    ang = float(delta) * th
    cos, sin = torch.cos(ang), torch.sin(ang)
    kf = keys.float()
    x1, x2 = kf[..., :half], kf[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos],
                     dim=-1).to(keys.dtype)


def tf_logprobs(model, cache, feed_ids, target_ids, position_ids=None,
                cache_position=None):
    """Teacher-forced logprobs of target_ids, where the last len(target_ids)
    logit positions of the feed score them (mirrors kvlib.batched_teacher_forced).
    feed = [prefix..., t_0..t_{n-2}]; returns list of n logprobs."""
    import torch
    n = len(target_ids)
    dev = model.device
    ids = torch.tensor([feed_ids], device=dev)
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=cache,
                    position_ids=position_ids, cache_position=cache_position,
                    use_cache=True,
                    logits_to_keep=n)
    lp = torch.log_softmax(out.logits[0].float(), dim=-1)
    tgt = torch.tensor(target_ids, device=dev)
    return lp[torch.arange(n), tgt].tolist()
