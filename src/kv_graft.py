"""KEY-grafting with RoPE re-rotation (K-Graft), plus a unified V/K graft.

Companion to kvlib_hf.blend_values (V-Graft). Operates on the same snapshot
format used on the 30B path: a list of per-layer ``(K, V)`` tensors, each of
shape ``[B, n_kv_heads, T, head_dim]`` with keys stored POST-RoPE.

RoPE re-rotation math (the crux)
--------------------------------
A raw key ``k_raw`` at position ``p`` is stored already rotated:
``k_wt = R(p_old)·k_raw``. To graft it into a new (compacted) position
``p_new`` we need ``R(p_new)·k_raw``. Because rotary rotations compose by angle,

    R(p_new)·k_raw = R(p_new)·R(p_old)^{-1}·k_wt = R(p_new - p_old)·k_wt.

So re-rotation is just applying the RoPE rotation for the position DELTA
``(p_new - p_old)`` to the stored write-time key. This module implements that
rotation for both the half-split (Llama/Qwen ``rotate_half``) and the
interleaved (GPT-NeoX pair) layouts; Qwen3 uses half-split.
"""

from __future__ import annotations

from collections import defaultdict

import torch


# --------------------------------------------------------------------------
# RoPE re-rotation primitives
# --------------------------------------------------------------------------

def _rope_thetas(head_dim, base, device):
    """Inverse frequencies, matching kvlib_hf.rope_thetas exactly."""
    i = torch.arange(0, head_dim // 2, dtype=torch.float32, device=device)
    return base ** (-2.0 * i / head_dim)


def rerotate_half(keys, delta, base):
    """Re-rotate post-RoPE keys by ``delta`` positions, half-split layout.

    Mirrors kvlib_hf.rotate_keys / HF apply_rotary_pos_emb with rotate_half:
    the head_dim is split into [x1 | x2] and rotated as a single block of
    ``head_dim/2`` complex pairs (x1_j, x2_j).
    """
    D = keys.shape[-1]
    half = D // 2
    th = _rope_thetas(D, base, keys.device)
    ang = float(delta) * th
    cos, sin = torch.cos(ang), torch.sin(ang)
    kf = keys.float()
    x1, x2 = kf[..., :half], kf[..., half:]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(keys.dtype)


def rerotate_interleaved(keys, delta, base):
    """Re-rotate post-RoPE keys by ``delta`` positions, interleaved layout.

    Pairs are adjacent dims (x0,x1),(x2,x3),... (GPT-NeoX / original RoPE).
    Provided for completeness; Qwen3/Llama in HF use the half-split form.
    """
    D = keys.shape[-1]
    th = _rope_thetas(D, base, keys.device)
    ang = float(delta) * th
    cos, sin = torch.cos(ang), torch.sin(ang)
    kf = keys.float()
    x_even = kf[..., 0::2]
    x_odd = kf[..., 1::2]
    r_even = x_even * cos - x_odd * sin
    r_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(kf)
    out[..., 0::2] = r_even
    out[..., 1::2] = r_odd
    return out.to(keys.dtype)


def make_rope_fn(base, layout="half"):
    """Build a ``rope_fn(keys, delta) -> rotated_keys`` for blend_keys/graft."""
    if layout in ("half", "half_split", "rotate_half"):
        return lambda keys, delta: rerotate_half(keys, delta, base)
    if layout in ("interleaved", "neox", "pairs"):
        return lambda keys, delta: rerotate_interleaved(keys, delta, base)
    raise ValueError(f"unknown RoPE layout: {layout!r}")


# --------------------------------------------------------------------------
# position helpers
# --------------------------------------------------------------------------

def _pos(positions, idx):
    """Map a storage index -> RoPE position.

    ``positions`` may be None (position == storage index) or a list/tensor
    indexed by storage slot.
    """
    if positions is None:
        return idx
    if torch.is_tensor(positions):
        return int(positions[idx].item())
    return int(positions[idx])


# --------------------------------------------------------------------------
# K-Graft: blend re-rotated write-time keys into fresh keys
# --------------------------------------------------------------------------

def blend_keys(fresh_snapshot, write_time_snapshot, pairs, alpha_K,
               positions_fresh=None, positions_write=None, rope_fn=None,
               layer_set=None, head_map=None):
    """K[fresh_idx] <- (1-a)*K_fresh + a*rerotate(K_write[write_idx], delta).

    For each aligned pair ``(fresh_idx, write_idx)`` the write-time key is
    re-rotated by ``delta = p_fresh - p_write`` (positions resolved via
    ``positions_fresh`` / ``positions_write``; default: position == storage
    index) and blended into the fresh key at ``fresh_idx``.

    Mirrors kvlib_hf.blend_values:
      * ``alpha_K`` may be a float or ``{layer_idx: alpha}`` (missing layers
        untouched).
      * ``layer_set`` restricts which layers are touched.
      * ``head_map`` = ``{layer_idx: [kv_head, ...]}`` restricts which heads
        blend.

    Values are passed through unchanged (clone kept alongside blended keys).
    """
    if rope_fn is None:
        raise ValueError("blend_keys requires rope_fn (see make_rope_fn)")

    # Resolve deltas and group pairs by delta so each rotation is a single
    # batched call over the head/dim axes.
    by_delta = defaultdict(lambda: ([], []))  # delta -> (fresh_idx[], write_idx[])
    for f_idx, w_idx in pairs:
        delta = _pos(positions_fresh, f_idx) - _pos(positions_write, w_idx)
        fi, wi = by_delta[delta]
        fi.append(f_idx)
        wi.append(w_idx)

    per_layer = isinstance(alpha_K, dict)
    out = []
    for li, (k, v) in enumerate(fresh_snapshot):
        a = alpha_K.get(li, 0.0) if per_layer else alpha_K
        heads = head_map.get(li) if head_map else None
        if (layer_set is not None and li not in layer_set) or \
           (per_layer and abs(a) < 1e-6) or \
           (head_map is not None and not heads):
            out.append((k, v))
            continue
        k2 = k.clone()
        kw_all = write_time_snapshot[li][0]
        for delta, (fi, wi) in by_delta.items():
            f_idx = torch.tensor(fi, device=k2.device)
            w_idx = torch.tensor(wi, device=kw_all.device)
            kw = kw_all.index_select(-2, w_idx).to(k2.device)
            kr = rope_fn(kw, delta).float()
            kf = k2.index_select(-2, f_idx).float()
            mixed = ((1 - a) * kf + a * kr).to(k2.dtype)
            if heads is None:
                k2.index_copy_(-2, f_idx, mixed)
            else:
                for h in heads:
                    k2[:, h, f_idx, :] = mixed[:, h]
        out.append((k2, v))
    return out


# --------------------------------------------------------------------------
# Unified graft (V-only / K-only / coupled / independent)
# --------------------------------------------------------------------------

def graft(snapshots, pairs, alpha_K=0.0, alpha_V=0.0,
          positions_fresh=None, positions_write=None, rope_fn=None,
          layer_set=None, head_map=None, blend_values=None):
    """Unified K/V graft over a ``(fresh_snapshot, write_time_snapshot)`` pair.

    Modes (by choice of alphas):
      * V-only     : alpha_K == 0, alpha_V != 0
      * K-only     : alpha_V == 0, alpha_K != 0
      * coupled    : alpha_K == alpha_V
      * independent: alpha_K != alpha_V, both nonzero

    The V part reuses kvlib_hf.blend_values (pass it as ``blend_values`` or it
    is imported lazily). alpha_K == 0 leaves keys BIT-IDENTICAL to fresh
    (blend_keys is skipped entirely).
    """
    fresh, write = snapshots
    snap = fresh

    v_active = (isinstance(alpha_V, dict) and any(abs(x) > 1e-6 for x in alpha_V.values())) \
        or (not isinstance(alpha_V, dict) and abs(alpha_V) > 1e-6)
    k_active = (isinstance(alpha_K, dict) and any(abs(x) > 1e-6 for x in alpha_K.values())) \
        or (not isinstance(alpha_K, dict) and abs(alpha_K) > 1e-6)

    if v_active:
        if blend_values is None:
            from kvlib_hf import blend_values as blend_values
        snap = blend_values(snap, write, pairs, alpha_V,
                            layer_set=layer_set, head_map=head_map)

    if k_active:
        snap = blend_keys(snap, write, pairs, alpha_K,
                          positions_fresh=positions_fresh,
                          positions_write=positions_write, rope_fn=rope_fn,
                          layer_set=layer_set, head_map=head_map)

    return snap
