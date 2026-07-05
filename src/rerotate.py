"""Key re-rotation for packed retention (H-pack).

Qwen3 caches keys post-RoPE. Moving a cached key from position p_old to
p_new requires rotating it by delta = p_new - p_old, exploiting RoPE's
additive composition: R(p_new) = R(delta) @ R(p_old).

mlx_lm's Qwen3 uses non-traditional (half-split) RoPE: for head_dim d, the
rotation pairs dimension i with dimension i + d/2, angle p * theta_i where
theta_i = base ** (-2i/d), i in [0, d/2).

rotate_keys applies R(delta) directly to cached key arrays
[B, H, L, D] — same delta for every position in the slice (packing moves a
contiguous block rigidly), or per-position deltas.

Identity requirements (LH ladder, run on-model before use):
  LH-0: rotate_keys(k, 0) == k bit-identically.
  LH-1: rotate_keys(rotate_keys(k, +d), -d) == k to fp16 tolerance.
  LH-2: rotate_keys(K_written_at_p, q - p) matches K_written_at_q for the
        same token content (computed by prefilling the same text at two
        offsets) to fp16 tolerance.
"""

import mlx.core as mx


def rope_thetas(head_dim, base):
    i = mx.arange(0, head_dim // 2, dtype=mx.float32)
    return base ** (-2.0 * i / head_dim)


def rotate_keys(keys, delta, base):
    """Rotate cached (post-RoPE) keys by `delta` positions.

    keys: [B, H, L, D] (fp16/fp32). delta: scalar int, or 1-D array of length
    L (per-position). base: rope theta base (model config rope_theta).
    """
    B, H, L, D = keys.shape
    half = D // 2
    th = rope_thetas(D, base)  # [half]
    if isinstance(delta, (int, float)):
        ang = float(delta) * th  # [half]
        cos = mx.cos(ang)[None, None, None, :]
        sin = mx.sin(ang)[None, None, None, :]
    else:
        ang = delta.astype(mx.float32)[:, None] * th[None, :]  # [L, half]
        cos = mx.cos(ang)[None, None, :, :]
        sin = mx.sin(ang)[None, None, :, :]
    kf = keys.astype(mx.float32)
    x1 = kf[..., :half]
    x2 = kf[..., half:]
    out = mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return out.astype(keys.dtype)
