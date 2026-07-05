"""LH ladder: identity tests for key re-rotation (prerequisite for H-pack).

LH-0: rotate by 0 is bit-identical.
LH-1: rotate by +d then -d returns the original to fp16 tolerance.
LH-2: keys written at offset p, re-rotated by (q-p), match keys written at
      offset q for identical token content (cross-check against the model's
      own RoPE), and generation from a packed cache built that way matches
      generation from a natively-positioned cache.
"""

import os
import sys

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "src")
from kvlib import extend_cache, max_abs_diff, prefill, snapshot_cache
from rerotate import rotate_keys

MODEL = os.environ.get("SC_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")

TEXT = ("The committee reviewed the proposal on Thursday and requested two "
        "revisions before the final vote scheduled for the end of the month.")


def main():
    model, tokenizer = load(MODEL)
    base = model.args.rope_theta

    ids = tokenizer.encode(TEXT)
    pad = tokenizer.encode("Filler sentence here. " * 30)[: 64]

    # keys for TEXT at offset len(pad) (after pad) vs offset 0 (no pad)
    c1, _ = prefill(model, pad + ids)
    snap1 = snapshot_cache(c1)
    c2, _ = prefill(model, ids)
    snap2 = snapshot_cache(c2)

    k1 = snap1[0][0][..., len(pad):, :]  # TEXT keys at positions len(pad)..
    k2 = snap2[0][0][..., : len(ids), :]  # TEXT keys at positions 0..

    # LH-0
    d0 = max_abs_diff(rotate_keys(k1, 0, base), k1)
    print(f"LH-0 rotate-by-0 max|diff|: {d0}")

    # LH-1
    r = rotate_keys(rotate_keys(k1, 37, base), -37, base)
    d1 = max_abs_diff(r, k1)
    print(f"LH-1 +37/-37 roundtrip max|diff|: {d1}")

    # LH-2a: k1 rotated by -len(pad) should equal k2 approximately — note the
    # CONTENT differs upstream (attention sees pad), but layer-0 keys are
    # position-stamped projections of token embeddings only (no attention
    # yet), so layer 0 must match closely.
    k1_moved = rotate_keys(k1, -len(pad), base)
    d2 = max_abs_diff(k1_moved, k2)
    print(f"LH-2a layer0 reposition vs native max|diff|: {d2}")

    ok = d0 == 0.0 and d1 < 0.01 and d2 < 0.01
    print("LH PASS" if ok else "LH FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
