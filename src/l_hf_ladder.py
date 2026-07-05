"""Identity ladder for the transformers port (CPU, Qwen3-0.6B by default).

HF-L0: prefill -> snapshot -> rebuild -> continue is token-identical, and
       cached keys are post-RoPE (layer-0 check vs manual rotation).
HF-L1: evict empty span == identity.
HF-L2: gapped eviction of a filler span with original positions -> coherent
       continuation (soft check: same first tokens as full cache or sane).
HF-LH: rotate_keys(k, 0) exact; +37/-37 roundtrip small; repositioned keys
       match natively-positioned keys within kernel noise floor.
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, "src")
from kvlib_hf import (
    blend_values,
    evict_span,
    greedy_generate,
    prefill,
    rebuild_cache,
    rotate_keys,
    snapshot_cache,
)

MODEL = os.environ.get("SC_HF_MODEL", "Qwen/Qwen3-0.6B")

MSGS = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "We're planning a small conference. Venue options were the Harbor Hotel and the University Hall; we rejected the hotel due to cost."},
    {"role": "assistant", "content": "Got it — University Hall it is. Shall we discuss catering next?"},
    {"role": "user", "content": "By the way, completely unrelated: I watched a documentary about tide pools last night, sea anemones fight in slow motion. Anyway."},
    {"role": "assistant", "content": "Fun! Back to planning — catering next?"},
    {"role": "user", "content": "Yes. The keynote speaker is Dr. Alvarez. What should we sort out first for catering?"},
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    dev = model.device
    eos_ids = {model.config.eos_token_id} if isinstance(
        model.config.eos_token_id, int) else set(model.config.eos_token_id)

    text = tok.apply_chat_template(MSGS, add_generation_prompt=True,
                                   tokenize=False)
    ids = torch.tensor([tok(text, add_special_tokens=False).input_ids],
                       device=dev)
    n = ids.shape[1]
    print(f"prompt tokens: {n}")

    # HF-L0
    cache, logits = prefill(model, ids)
    snap = snapshot_cache(cache)
    ref = greedy_generate(model, cache, logits, 30, eos_ids, n)
    c2 = rebuild_cache(snap, DynamicCache)
    t2 = greedy_generate(model, c2, logits, 30, eos_ids, n)
    print(f"HF-L0 roundtrip token-identical: {ref == t2}")
    print("gen:", tok.decode(ref)[:80].replace("\n", " "))

    # post-RoPE key check: layer-0 keys for same tokens at two offsets differ
    pad = tok("Filler words here. " * 8, return_tensors="pt").input_ids.to(dev)
    c3, _ = prefill(model, torch.cat([pad, ids], dim=1))
    k_shift = snapshot_cache(c3)[0][0][..., pad.shape[1]:, :]
    k_base = snap[0][0]
    d_raw = (k_shift - k_base).abs().max().item()
    base = (getattr(model.config, "rope_theta", None)
            or model.config.rope_parameters["rope_theta"])
    k_moved = rotate_keys(k_shift, -pad.shape[1], base)
    d_rot = (k_moved.float() - k_base.float()).abs().max().item()
    print(f"post-RoPE check: raw diff {d_raw:.4f} -> after re-rotation {d_rot:.6f}")

    # HF-L1: empty eviction identity
    c4 = rebuild_cache(evict_span(snap, 10, 10), DynamicCache)
    t4 = greedy_generate(model, c4, logits, 30, eos_ids, n)
    print(f"HF-L1 empty eviction identical: {t4 == ref}")

    # HF-L2: evict the filler user message tokens (locate via im_start scan)
    ims = tok.encode("<|im_start|>")[0]
    flat = ids[0].tolist()
    starts = [i for i, t in enumerate(flat) if t == ims]
    ev0, ev1 = starts[3], starts[4]
    c5 = rebuild_cache(evict_span(snap, ev0, ev1), DynamicCache)
    # gapped: positions continue from n; attention over shorter storage is
    # fine because DynamicCache length only drives causal masking for new toks
    t5 = greedy_generate(model, c5, logits, 30, eos_ids, n)
    same_prefix = next((i for i, (a, b) in enumerate(zip(ref, t5)) if a != b),
                       min(len(ref), len(t5)))
    print(f"HF-L2 filler eviction shared prefix: {same_prefix} tokens")

    # HF-LH rotation identities
    k = snap[0][0]
    d0 = (rotate_keys(k, 0, base) - k).abs().max().item()
    d1 = (rotate_keys(rotate_keys(k, 37, base), -37, base).float()
          - k.float()).abs().max().item()
    print(f"HF-LH rotate0 {d0}; +37/-37 roundtrip {d1:.6f}")

    ok = (ref == t2) and (t4 == ref) and d0 == 0.0 and d1 < 1e-3 \
        and d_rot < max(1e-3, d_raw * 0.01) and same_prefix >= 5
    print("HF LADDER PASS" if ok else "HF LADDER FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
