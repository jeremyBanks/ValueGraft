"""L0 — Cache identity test.

Prefill a conversation; serialize the cache; rebuild it manually; verify
continued generation is token-identical and logits match to float tolerance.
Also empirically confirms cached keys are post-RoPE by recomputing layer-0
keys by hand.
"""

import sys

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "src")
from kvlib import (
    greedy_generate,
    make_prompt_ids,
    max_abs_diff,
    prefill,
    rebuild_cache,
    snapshot_cache,
)

import os
MODEL = os.environ.get("SC_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
MAX_TOKENS = 60

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "We're planning a small conference. Venue options were the Harbor Hotel and the University Hall; we rejected the hotel due to cost."},
    {"role": "assistant", "content": "Got it — University Hall it is. Shall we discuss catering next?"},
    {"role": "user", "content": "Yes. Also remember the keynote speaker is Dr. Alvarez. What should we sort out first for catering?"},
]


def main():
    model, tokenizer = load(MODEL)
    eos_ids = set(tokenizer.eos_token_ids or [tokenizer.eos_token_id])

    ids = make_prompt_ids(tokenizer, MESSAGES)
    print(f"prompt tokens: {len(ids)}")

    # Reference: prefill and generate.
    cache_a, logits_a = prefill(model, ids)
    snap = snapshot_cache(cache_a)
    toks_a, steps_a = greedy_generate(model, cache_a, logits_a, MAX_TOKENS, eos_ids)

    # Rebuild the full cache from the serialized snapshot and continue from the
    # same prefill logits. Any divergence is then attributable to the cache
    # round-trip itself. (Note: recomputing the last position via a 1-token
    # forward pass is NOT numerically identical to batched prefill — different
    # kernels — so we deliberately reuse logits_a for both paths.)
    cache_b = rebuild_cache(snap)
    toks_b, steps_b = greedy_generate(model, cache_b, logits_a, MAX_TOKENS, eos_ids)

    identical = toks_a == toks_b
    print(f"token-identical continuation: {identical}")
    print("gen A:", tokenizer.decode(toks_a))
    if not identical:
        print("gen B:", tokenizer.decode(toks_b))
        for i, (x, y) in enumerate(zip(toks_a, toks_b)):
            if x != y:
                print(f"first divergence at step {i}: {x} vs {y}")
                break
    ld = max(max_abs_diff(a, b) for a, b in zip(steps_a, steps_b))
    print("per-step logits max|diff| over continuation:", ld)

    # RoPE placement check: recompute layer-0 keys by hand, pre- and post-RoPE,
    # and compare with what the cache stored.
    layer0 = model.layers[0].self_attn
    x = model.model.embed_tokens(mx.array(ids)[None])
    # replicate the pre-attention layernorm
    h = model.layers[0].input_layernorm(x)
    B, L, _ = h.shape
    k = layer0.k_norm(layer0.k_proj(h).reshape(B, L, layer0.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    k_rot = layer0.rope(k, offset=0)
    stored_k = snap[0][0][..., :L, :]
    print("layer0 stored-vs-unrotated max|diff|:", max_abs_diff(stored_k, k))
    print("layer0 stored-vs-rotated   max|diff|:", max_abs_diff(stored_k, k_rot))

    ok = identical and ld < 1e-2
    print("L0 PASS" if ok else "L0 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
