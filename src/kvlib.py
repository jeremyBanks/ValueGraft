"""Shared utilities for KV-cache surgery experiments.

Facts verified against mlx_lm 0.31.3 source (models/qwen3.py, models/cache.py):
- Keys are cached POST-RoPE (rotation applied before cache.update_and_fetch).
- Values are cached unrotated.
- Query positions come from cache.offset at generation time.
- KVCache stores keys/values as [B, n_kv_heads, seq_padded, head_dim] arrays,
  padded up to multiples of KVCache.step (256); .state trims to offset.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import KVCache


def make_prompt_ids(tokenizer, messages, add_generation_prompt=True):
    """Token IDs for a chat conversation via the model's chat template."""
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=True
    )


def prefill(model, token_ids, batch_size=1024):
    """Prefill token_ids through model, returning (cache, last_logits).

    Processes in chunks to bound memory. Returns the per-layer cache list and
    the logits for the final position (needed to start greedy generation).
    """
    cache = [KVCache() for _ in range(len(model.layers))]
    ids = mx.array(token_ids)[None]
    logits = None
    for i in range(0, ids.shape[1], batch_size):
        chunk = ids[:, i : i + batch_size]
        logits = model(chunk, cache=cache)
        mx.eval([c.state for c in cache])
    return cache, logits[:, -1, :]


def greedy_generate(model, cache, first_logits, max_tokens, eos_ids):
    """Greedy decode from an existing cache. Returns (token_ids, per_step_logits).

    first_logits: logits at the last prefilled position [1, vocab].
    """
    tokens = []
    step_logits = []
    logits = first_logits
    for _ in range(max_tokens):
        tok = mx.argmax(logits, axis=-1)
        tok_id = tok.item()
        tokens.append(tok_id)
        step_logits.append(logits)
        if tok_id in eos_ids:
            break
        logits = model(tok[None], cache=cache)[:, -1, :]
    return tokens, step_logits


def snapshot_cache(cache):
    """Serialize a cache list to plain (keys, values, offset) tuples (trimmed)."""
    out = []
    for c in cache:
        k, v = c.state  # trimmed to offset
        out.append((mx.array(k), mx.array(v), c.offset))
    return out


def rebuild_cache(snapshot):
    """Manually reconstruct KVCache objects from snapshot_cache output."""
    cache = []
    for k, v, offset in snapshot:
        c = KVCache()
        c.keys = mx.array(k)
        c.values = mx.array(v)
        c.offset = offset
        cache.append(c)
    return cache


def max_abs_diff(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
