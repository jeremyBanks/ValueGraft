"""Core primitives for the preregistered coherent-summary-state experiment.

This module is intentionally isolated from the legacy cross-architecture harness.
It provides the one-token incremental capture path, bounded summary-row handling,
exact identity diagnostics, summary-only K/V arm construction, and a treatment-
delta-matched derangement. The driver and production ladder live in separate files.

State contract
--------------
* A full cache is live only while one conversation/source is processed.
* Persisted tensor state is bounded to <= ``max_summary_tokens`` rows per layer.
* Every source row set is keyed by exact prefix/summary token hashes + model/code
  provenance in the caller's manifest.
* The driver evicts full caches immediately after extracting/hashing summary rows.
* Exact replay of prefix + summary IDs is the equivalence proof; the production
  ladder exercises it before any paid semantic result is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Iterable, Sequence

import torch

from kvlib_hf import rotate_keys


Snapshot = list[tuple[torch.Tensor, torch.Tensor]]


class CoherentStateError(RuntimeError):
    """Fail-closed apparatus error."""


def sha256_ids(ids: Sequence[int]) -> str:
    h = hashlib.sha256()
    for token_id in ids:
        h.update(int(token_id).to_bytes(8, "little", signed=True))
    return h.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def cache_layers(cache):
    if hasattr(cache, "layers"):
        return cache.layers
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    raise CoherentStateError(f"unsupported cache type: {type(cache).__name__}")


def _layer_kv(layer) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(layer, tuple):
        return layer
    if hasattr(layer, "keys") and hasattr(layer, "values"):
        return layer.keys, layer.values
    raise CoherentStateError(f"unsupported cache layer: {type(layer).__name__}")


def extract_summary_rows(cache, start: int, end: int, *,
                         max_summary_tokens: int = 900,
                         to_cpu: bool = True) -> Snapshot:
    """Clone only ``[start:end)`` K/V rows from every cache layer.

    This is the retained-state size assertion required by the repository's
    stateful-change checklist. Full-cache cloning is deliberately absent.
    """
    if not (0 <= start < end):
        raise CoherentStateError(f"invalid summary span [{start}, {end})")
    n = end - start
    if n > max_summary_tokens:
        raise CoherentStateError(
            f"summary rows {n} exceed asserted bound {max_summary_tokens}")
    out: Snapshot = []
    for li, layer in enumerate(cache_layers(cache)):
        k, v = _layer_kv(layer)
        if k.shape[-2] < end or v.shape[-2] < end:
            raise CoherentStateError(
                f"layer {li} cache shorter than summary end: "
                f"K={k.shape[-2]} V={v.shape[-2]} end={end}")
        ks = k[..., start:end, :].detach().clone()
        vs = v[..., start:end, :].detach().clone()
        if to_cpu:
            ks, vs = ks.cpu(), vs.cpu()
        out.append((ks.contiguous(), vs.contiguous()))
    if not out:
        raise CoherentStateError("cache contained no layers")
    return out


def row_hashes(rows: Snapshot) -> list[dict[str, str]]:
    return [
        {"layer": str(li), "k_sha256": sha256_tensor(k),
         "v_sha256": sha256_tensor(v)}
        for li, (k, v) in enumerate(rows)
    ]


def compare_rows(a: Snapshot, b: Snapshot) -> list[dict[str, float | int]]:
    if len(a) != len(b):
        raise CoherentStateError(f"layer-count mismatch: {len(a)} != {len(b)}")
    out = []
    for li, ((ka, va), (kb, vb)) in enumerate(zip(a, b)):
        if ka.shape != kb.shape or va.shape != vb.shape:
            raise CoherentStateError(
                f"layer {li} row-shape mismatch: {ka.shape}/{va.shape} "
                f"!= {kb.shape}/{vb.shape}")
        out.append({
            "layer": li,
            "k_max_abs": float((ka.float() - kb.float()).abs().max().item()),
            "v_max_abs": float((va.float() - vb.float()).abs().max().item()),
        })
    return out


def move_key_rows(rows: Snapshot, delta: int, rope_theta: float) -> Snapshot:
    """Re-rotate every post-RoPE key row by one contiguous-span position delta."""
    return [(rotate_keys(k, delta, rope_theta), v.clone()) for k, v in rows]


def replace_summary_rows(fresh_snapshot: Snapshot, source_rows: Snapshot,
                         destination_start: int, *, use_keys: bool,
                         use_values: bool) -> Snapshot:
    """Return a full destination snapshot with exactly one summary span replaced."""
    if not use_keys and not use_values:
        return [(k.clone(), v.clone()) for k, v in fresh_snapshot]
    if len(fresh_snapshot) != len(source_rows):
        raise CoherentStateError("source/destination layer-count mismatch")
    out: Snapshot = []
    for li, ((kf, vf), (ks, vs)) in enumerate(zip(fresh_snapshot, source_rows)):
        n = ks.shape[-2]
        if vs.shape[-2] != n or ks.shape[:-2] + ks.shape[-1:] != \
                kf.shape[:-2] + kf.shape[-1:] or \
                vs.shape[:-2] + vs.shape[-1:] != vf.shape[:-2] + vf.shape[-1:]:
            raise CoherentStateError(f"layer {li} source-row geometry mismatch")
        end = destination_start + n
        if destination_start < 0 or end > kf.shape[-2] or end > vf.shape[-2]:
            raise CoherentStateError(
                f"layer {li} destination [{destination_start},{end}) out of range")
        k2, v2 = kf.clone(), vf.clone()
        if use_keys:
            k2[..., destination_start:end, :] = ks.to(k2.device, dtype=k2.dtype)
        if use_values:
            v2[..., destination_start:end, :] = vs.to(v2.device, dtype=v2.dtype)
        out.append((k2, v2))
    return out


def fixed_point_free_permutation(n: int, seed: int) -> list[int]:
    if n < 2:
        raise CoherentStateError("a derangement needs at least two rows")
    rng = random.Random(seed)
    base = list(range(n))
    # Sattolo produces one cycle, hence no fixed points, for n >= 2.
    perm = base[:]
    for i in range(n - 1, 0, -1):
        j = rng.randrange(i)
        perm[i], perm[j] = perm[j], perm[i]
    if any(i == p for i, p in enumerate(perm)):
        raise CoherentStateError("internal derangement failure")
    return perm


@dataclass(frozen=True)
class DerangementDiagnostics:
    seed: int
    layer: int
    head: int
    permutation: tuple[int, ...]
    fixed_points: int
    max_multiset_diff: float
    mean_diff: float
    covariance_diff: float
    applied_delta_max_abs_error: float
    applied_multiset_diff: float
    applied_mean_diff: float
    applied_covariance_diff: float


def _moment_diffs(original: torch.Tensor,
                  permuted: torch.Tensor) -> tuple[float, float, float]:
    """Diagnostics over [N,D] rows; permutation should preserve all exactly."""
    so = original.float()[torch.argsort(original.float().sum(-1))]
    sp = permuted.float()[torch.argsort(permuted.float().sum(-1))]
    multiset = float((so - sp).abs().max().item())
    mean = float((original.float().mean(0) - permuted.float().mean(0)).abs().max().item())
    oc = original.float() - original.float().mean(0)
    pc = permuted.float() - permuted.float().mean(0)
    cov_o = oc.T @ oc / max(1, original.shape[0] - 1)
    cov_p = pc.T @ pc / max(1, permuted.shape[0] - 1)
    cov = float((cov_o - cov_p).abs().max().item())
    return multiset, mean, cov


def delta_deranged_snapshot(fresh_snapshot: Snapshot, correct_rows: Snapshot,
                            destination_start: int, seed: int) \
        -> tuple[Snapshot, list[DerangementDiagnostics]]:
    """Apply a per-layer/head derangement of ``V_correct - V_fresh`` to fresh V.

    Keys and all non-summary rows remain bit-identical to fresh. Each head's exact
    delta-row multiset is preserved, so row norms and covariance are matched by
    construction rather than approximately rescaled.
    """
    if len(fresh_snapshot) != len(correct_rows):
        raise CoherentStateError("delta source/destination layer-count mismatch")
    out: Snapshot = []
    diagnostics: list[DerangementDiagnostics] = []
    for li, ((kf, vf), (_, vc)) in enumerate(zip(fresh_snapshot, correct_rows)):
        n = vc.shape[-2]
        end = destination_start + n
        if n < 2 or end > vf.shape[-2]:
            raise CoherentStateError(f"layer {li} invalid delta span")
        v2 = vf.clone()
        fresh_rows = vf[..., destination_start:end, :].float()
        corr_rows = vc.to(vf.device).float()
        if fresh_rows.shape != corr_rows.shape:
            raise CoherentStateError(f"layer {li} delta geometry mismatch")
        for head in range(fresh_rows.shape[1]):
            head_seed = seed + li * 100_003 + head * 1_009
            perm = fixed_point_free_permutation(n, head_seed)
            idx = torch.tensor(perm, device=vf.device)
            delta = corr_rows[:, head] - fresh_rows[:, head]
            dperm = delta.index_select(-2, idx)
            v2[:, head, destination_start:end, :] = \
                (fresh_rows[:, head] + dperm).to(v2.dtype)
            multiset, mean, cov = _moment_diffs(delta[0], dperm[0])
            applied = (v2[:, head, destination_start:end, :].float()
                       - fresh_rows[:, head])
            applied_error = float((applied - dperm).abs().max().item())
            _unused_sort_diff, applied_mean, applied_cov = _moment_diffs(
                delta[0], applied[0])
            diagnostics.append(DerangementDiagnostics(
                seed=head_seed, layer=li, head=head,
                permutation=tuple(perm), fixed_points=0,
                max_multiset_diff=multiset, mean_diff=mean,
                covariance_diff=cov,
                applied_delta_max_abs_error=applied_error,
                # The intended-to-applied row correspondence is known (dperm),
                # so its max pointwise error is a valid upper bound on optimal
                # multiset matching. Sorting by near-equal row sums is unstable.
                applied_multiset_diff=applied_error,
                applied_mean_diff=applied_mean,
                applied_covariance_diff=applied_cov))
        out.append((kf.clone(), v2))
    return out, diagnostics


def require_exact_span(container_ids: Sequence[int], summary_ids: Sequence[int],
                       expected_start: int) -> tuple[int, int]:
    n = len(summary_ids)
    if n == 0:
        raise CoherentStateError("empty summary token sequence")
    end = expected_start + n
    if expected_start < 0 or end > len(container_ids):
        raise CoherentStateError("summary span outside destination IDs")
    if list(container_ids[expected_start:end]) != list(summary_ids):
        raise CoherentStateError(
            "exact summary IDs absent at the declared destination span")
    # Ambiguous duplicate occurrence is dangerous: a wrong wrapper span could pass.
    matches = [i for i in range(len(container_ids) - n + 1)
               if list(container_ids[i:i + n]) == list(summary_ids)]
    if matches != [expected_start]:
        raise CoherentStateError(
            f"summary occurrence is not unique/exact: matches={matches}, "
            f"expected={expected_start}")
    return expected_start, end


@dataclass
class IncrementalTrace:
    token_ids: list[int]
    token_logprobs: list[float]
    start_position: int
    end_position: int
    ended_on_eos: bool


def append_ids_stepwise(model, cache, first_logits: torch.Tensor,
                        token_ids: Sequence[int], start_position: int,
                        *, start_cache_position: int | None = None,
                        attention_masks: Iterable[torch.Tensor | None] | None = None):
    """Teacher-force exact IDs one token at a time through the common kernel path."""
    ids = [int(x) for x in token_ids]
    if not ids:
        raise CoherentStateError("cannot append an empty token sequence")
    masks = list(attention_masks) if attention_masks is not None else [None] * len(ids)
    if len(masks) != len(ids):
        raise CoherentStateError("one attention-mask entry is required per token")
    logits = first_logits
    logprobs: list[float] = []
    for offset, (token_id, mask) in enumerate(zip(ids, masks)):
        lp = torch.log_softmax(logits.float(), dim=-1)[0, token_id]
        logprobs.append(float(lp.item()))
        pos = start_position + offset
        kwargs = {"input_ids": torch.tensor([[token_id]], device=model.device),
                  "past_key_values": cache,
                  "position_ids": torch.tensor([[pos]], device=model.device),
                  "use_cache": True}
        if start_cache_position is not None:
            kwargs["cache_position"] = torch.tensor(
                [start_cache_position + offset], device=model.device)
        if mask is not None:
            kwargs["attention_mask"] = mask
        with torch.no_grad():
            out = model(**kwargs)
        cache, logits = out.past_key_values, out.logits[:, -1, :]
    return cache, logits, IncrementalTrace(
        token_ids=ids, token_logprobs=logprobs,
        start_position=start_position, end_position=start_position + len(ids),
        ended_on_eos=False)


def generate_greedy_incremental(model, cache, first_logits: torch.Tensor,
                                start_position: int, max_tokens: int,
                                eos_ids: set[int], *,
                                start_cache_position: int | None = None):
    """Generate and append non-EOS tokens; reaching ``max_tokens`` fails closed."""
    if max_tokens <= 0:
        raise CoherentStateError("max_tokens must be positive")
    ids: list[int] = []
    logprobs: list[float] = []
    logits = first_logits
    for offset in range(max_tokens):
        token_id = int(torch.argmax(logits, dim=-1).item())
        if token_id in eos_ids:
            return cache, logits, IncrementalTrace(
                token_ids=ids, token_logprobs=logprobs,
                start_position=start_position,
                end_position=start_position + len(ids), ended_on_eos=True)
        lp = torch.log_softmax(logits.float(), dim=-1)[0, token_id]
        ids.append(token_id)
        logprobs.append(float(lp.item()))
        pos = start_position + offset
        with torch.no_grad():
            kwargs = {
                "input_ids": torch.tensor([[token_id]], device=model.device),
                "past_key_values": cache,
                "position_ids": torch.tensor([[pos]], device=model.device),
                "use_cache": True,
            }
            if start_cache_position is not None:
                kwargs["cache_position"] = torch.tensor(
                    [start_cache_position + offset], device=model.device)
            out = model(**kwargs)
        cache, logits = out.past_key_values, out.logits[:, -1, :]
    raise CoherentStateError(
        f"generation hit {max_tokens}-token cap without EOS; render is invalid")
