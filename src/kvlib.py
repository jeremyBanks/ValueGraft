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


def first_step_logits(model, cache, last_token_id):
    """Run the final prompt token as a 1-token decode step.

    Protocol: every arm builds its cache over prompt[:-1], then computes the
    first continuation logits this way, so all cross-arm comparisons use
    identically-shaped forward passes (batched prefill and 1-token decode are
    not numerically interchangeable).
    """
    return model(mx.array([[last_token_id]]), cache=cache)[:, -1, :]


def teacher_forced_logprobs(model, cache, first_logits, cont_ids):
    """Per-token logprobs of a fixed continuation under a cache (decode steps).

    Returns (logprobs, per_step_logits). Mutates cache.
    """
    logprobs = []
    step_logits = []
    logits = first_logits
    for t in cont_ids:
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        logprobs.append(lp[0, t].item())
        step_logits.append(logits)
        logits = model(mx.array([[t]]), cache=cache)[:, -1, :]
    return logprobs, step_logits


def sampled_generate(model, cache, first_logits, max_tokens, eos_ids, temp=0.7):
    """Temperature-sampled decode from an existing cache. Returns token ids
    (without the terminating eos). Determinism comes from mx.random.seed set
    by the caller."""
    tokens = []
    logits = first_logits
    for _ in range(max_tokens):
        tok = mx.random.categorical(logits.astype(mx.float32) / temp)
        tok_id = tok.item()
        if tok_id in eos_ids:
            break
        tokens.append(tok_id)
        logits = model(tok[None], cache=cache)[:, -1, :]
    return tokens


def truncate_cache(cache, length):
    """Truncate KVCache list in place to `length` tokens."""
    for c in cache:
        c.keys = c.keys[..., :length, :]
        c.values = c.values[..., :length, :]
        c.offset = length


def extend_cache(model, cache, token_ids, batch_size=1024):
    """Prefill additional tokens onto an existing cache; returns last logits."""
    ids = mx.array(token_ids)[None]
    logits = None
    for i in range(0, ids.shape[1], batch_size):
        logits = model(ids[:, i : i + batch_size], cache=cache)
        mx.eval([c.state for c in cache])
    return logits[:, -1, :]


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


class GappedKVCache:
    """KV cache whose stored entries may be fewer than the position counter.

    Used for Arm C (summary-anchored eviction): retained entries keep their
    original (post-RoPE) positions, and `offset` continues from the original
    conversation end so new queries/keys are rotated in-distribution.

    Storage length and position counter are decoupled: `offset` drives RoPE;
    `keys.shape[2]` drives the attention mask. New entries are appended
    contiguously after the retained ones (the positional gap lives in the
    rotation already baked into the retained keys, not in the array layout).

    Implements `make_mask` so mlx_lm's create_attention_mask() builds a mask
    sized to actual storage: for N new tokens, all previously stored entries
    are attendable and the new block is causal.
    """

    def __init__(self, keys, values, position_offset):
        self.keys = keys
        self.values = values
        self.offset = position_offset

    def update_and_fetch(self, keys, values):
        self.keys = mx.concatenate([self.keys, keys], axis=2)
        self.values = mx.concatenate([self.values, values], axis=2)
        self.offset += keys.shape[2]
        return self.keys, self.values

    def make_mask(self, N, return_array=False, window_size=None):
        assert window_size is None
        if N == 1:
            return None
        S = self.keys.shape[2]  # called before update_and_fetch for this step
        linds = mx.arange(N)[:, None]
        rinds = mx.arange(S + N)[None]
        return linds + S >= rinds

    @property
    def state(self):
        return self.keys, self.values

    def size(self):
        return self.offset


def max_abs_diff(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
