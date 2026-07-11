"""Additive model-facing primitives for coherent-state canary v12.

This module consumes only the frozen v12 replay plans.  It deliberately does
not import v10 arm, layout, store, release, or harvester code.  Full caches are
live only inside a call; callers persist bounded selected rows and hashes.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import math
import struct
from typing import Sequence

import torch
from transformers import DynamicCache

from coherent_canary_schema import DestinationEvent, FreshDestinationPlan, ReplayEvent, ReplayPlan
from coherent_state_tokens import generation_prefix_ids


Snapshot = list[tuple[torch.Tensor, torch.Tensor]]
MAX_LIVE_CACHE_TOKENS = 7000


class CanaryRuntimeError(RuntimeError):
    """Fail-closed v12 execution error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryRuntimeError(message)


def model_device(model) -> torch.device:
    value = getattr(model, "device", None)
    if value is not None:
        return torch.device(value)
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise CanaryRuntimeError("model has no discoverable device") from exc


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _cache_layers(cache):
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise CanaryRuntimeError(f"unsupported cache type: {type(cache).__name__}")
    return layers


def snapshot_cache(cache, *, detach: bool = True, to_cpu: bool = False) -> Snapshot:
    rows: Snapshot = []
    for index, layer in enumerate(_cache_layers(cache)):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        _require(isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor),
                 f"cache layer {index} lacks tensor K/V")
        if detach:
            keys, values = keys.detach(), values.detach()
        keys, values = keys.clone(), values.clone()
        if to_cpu:
            keys, values = keys.cpu(), values.cpu()
        rows.append((keys.contiguous(), values.contiguous()))
    _require(bool(rows), "cache has no layers")
    snapshot_physical_length(rows)
    return rows


def rebuild_cache(snapshot: Snapshot, *, clone: bool = True) -> DynamicCache:
    _require(bool(snapshot), "cannot rebuild an empty cache")
    data = []
    for keys, values in snapshot:
        data.append((keys.clone() if clone else keys,
                     values.clone() if clone else values))
    return DynamicCache(ddp_cache_data=data)


def snapshot_physical_length(snapshot: Snapshot) -> int:
    _require(bool(snapshot), "snapshot has no layers")
    lengths = []
    for index, (keys, values) in enumerate(snapshot):
        _require(keys.ndim == 4 and values.ndim == 4,
                 f"snapshot layer {index} is not [B,H,T,D]")
        _require(keys.shape[:3] == values.shape[:3],
                 f"snapshot layer {index} K/V prefix geometry differs")
        lengths.append(int(keys.shape[-2]))
    _require(len(set(lengths)) == 1, "snapshot layer lengths differ")
    return lengths[0]


def snapshot_hashes(snapshot: Snapshot) -> list[dict]:
    return [{
        "layer": index,
        "k_shape": list(keys.shape),
        "v_shape": list(values.shape),
        "k_dtype": str(keys.dtype),
        "v_dtype": str(values.dtype),
        "k_sha256": tensor_sha256(keys),
        "v_sha256": tensor_sha256(values),
    } for index, (keys, values) in enumerate(snapshot)]


def extract_rows(snapshot: Snapshot, start: int, end: int, *,
                 max_rows: int = 256, to_cpu: bool = True) -> Snapshot:
    _require(0 <= start < end, "invalid selected-row interval")
    _require(end - start <= max_rows,
             f"selected rows {end - start} exceed asserted bound {max_rows}")
    _require(end <= snapshot_physical_length(snapshot),
             "selected-row interval exceeds snapshot")
    out: Snapshot = []
    for keys, values in snapshot:
        k = keys[..., start:end, :].detach().clone()
        v = values[..., start:end, :].detach().clone()
        if to_cpu:
            k, v = k.cpu(), v.cpu()
        out.append((k.contiguous(), v.contiguous()))
    return out


def replace_rows(base: Snapshot, source: Snapshot, destination_start: int, *,
                 use_keys: bool, use_values: bool) -> tuple[Snapshot, dict]:
    _require(use_keys or use_values, "row replacement selected no channel")
    _require(len(base) == len(source) and bool(base), "replacement layer coverage differs")
    base_length = snapshot_physical_length(base)
    row_count = int(source[0][0].shape[-2])
    _require(row_count > 0 and destination_start >= 0 and
             destination_start + row_count <= base_length,
             "replacement interval exceeds destination")
    output: Snapshot = []
    per_layer = []
    for index, ((base_k, base_v), (source_k, source_v)) in enumerate(zip(base, source)):
        _require(source_k.shape[-2] == source_v.shape[-2] == row_count,
                 f"source layer {index} row count differs")
        _require(base_k.shape[:2] + base_k.shape[3:] ==
                 source_k.shape[:2] + source_k.shape[3:],
                 f"source layer {index} key geometry differs")
        _require(base_v.shape[:2] + base_v.shape[3:] ==
                 source_v.shape[:2] + source_v.shape[3:],
                 f"source layer {index} value geometry differs")
        end = destination_start + row_count
        k, v = base_k.clone(), base_v.clone()
        if use_keys:
            k[..., destination_start:end, :] = source_k.to(k.device, k.dtype)
        if use_values:
            v[..., destination_start:end, :] = source_v.to(v.device, v.dtype)
        _require((not use_keys or torch.equal(
            k[..., destination_start:end, :], source_k.to(k.device, k.dtype))),
            f"layer {index} selected key insertion differs")
        _require((not use_values or torch.equal(
            v[..., destination_start:end, :], source_v.to(v.device, v.dtype))),
            f"layer {index} selected value insertion differs")
        _require(torch.equal(k[..., :destination_start, :],
                             base_k[..., :destination_start, :]) and
                 torch.equal(k[..., end:, :], base_k[..., end:, :]),
                 f"layer {index} non-selected key rows changed")
        _require(torch.equal(v[..., :destination_start, :],
                             base_v[..., :destination_start, :]) and
                 torch.equal(v[..., end:, :], base_v[..., end:, :]),
                 f"layer {index} non-selected value rows changed")
        _require(use_keys or torch.equal(k, base_k),
                 f"layer {index} unselected keys changed")
        _require(use_values or torch.equal(v, base_v),
                 f"layer {index} unselected values changed")
        output.append((k, v))
        per_layer.append({
            "layer": index,
            "before_k_sha256": tensor_sha256(base_k),
            "before_v_sha256": tensor_sha256(base_v),
            "after_k_sha256": tensor_sha256(k),
            "after_v_sha256": tensor_sha256(v),
            "source_k_sha256": tensor_sha256(source_k),
            "source_v_sha256": tensor_sha256(source_v),
        })
    return output, {
        "destination_start": destination_start,
        "destination_end": destination_start + row_count,
        "row_count": row_count,
        "use_keys": use_keys,
        "use_values": use_values,
        "per_layer": per_layer,
    }


@dataclass
class ExecutionResult:
    snapshot: Snapshot
    last_logits: torch.Tensor
    calls: list[dict]
    q1_token_logprobs: list[dict]
    physical_end: int
    logical_end: int


@dataclass
class GenerationResult:
    snapshot: Snapshot
    content_ids: list[int]
    token_logprobs: list[float]
    token_logprob_float32_bits: list[str]
    logical_positions: list[int]
    physical_positions: list[int]
    stop_reason: str
    stop_candidate_id: int
    stop_candidate_logprob: float
    stop_candidate_logprob_float32_bits: str
    eos_ids: list[int]
    cap_hit: bool


def _forward(model, cache, token_ids: Sequence[int], logical_positions: Sequence[int],
             physical_positions: Sequence[int], *, enable_grad: bool):
    _require(len(token_ids) == len(logical_positions) == len(physical_positions) > 0,
             "forward arrays differ in length")
    device = model_device(model)
    context = nullcontext() if enable_grad else torch.no_grad()
    with context:
        result = model(
            input_ids=torch.tensor([list(token_ids)], device=device),
            past_key_values=cache,
            position_ids=torch.tensor([list(logical_positions)], device=device),
            cache_position=torch.tensor(list(physical_positions), device=device),
            use_cache=True,
            logits_to_keep=1,
        )
    return result.past_key_values, result.logits[:, -1, :]


def _run_events(model, token_ids: Sequence[int], events, *,
                stop_at: int, destination: bool, initial_snapshot: Snapshot | None = None,
                enable_grad: bool = False) -> ExecutionResult:
    _require(0 < stop_at <= MAX_LIVE_CACHE_TOKENS,
             f"live cache stop {stop_at} exceeds bound {MAX_LIVE_CACHE_TOKENS}")
    cache = rebuild_cache(initial_snapshot, clone=not enable_grad) if initial_snapshot else None
    physical_cursor = snapshot_physical_length(initial_snapshot) if initial_snapshot else 0
    last_logits = None
    calls: list[dict] = []
    q1_lps: list[dict] = []
    for event in events:
        event_start = event.physical_start if destination else event.token_start
        event_end = event.physical_end if destination else event.token_end
        logical_start = event.logical_start if destination else event.token_start
        logical_end = event.logical_end if destination else event.token_end
        if event_end <= physical_cursor:
            continue
        _require(event_start == physical_cursor,
                 f"event does not continue cache: {event_start} != {physical_cursor}")
        _require(event_end <= stop_at, "requested stop cuts inside a model call")
        ids = list(token_ids[event_start:event_end])
        if event.kind == "q1" and last_logits is not None:
            token_id = ids[0]
            lp = torch.log_softmax(last_logits.float(), dim=-1)[0, token_id]
            q1_lps.append({"physical_position": event_start,
                           "logical_position": logical_start,
                           "token_id": token_id, "logprob": float(lp.detach().cpu())})
        cache, last_logits = _forward(
            model, cache, ids, range(logical_start, logical_end),
            range(event_start, event_end), enable_grad=enable_grad)
        physical_cursor = event_end
        calls.append({
            **asdict(event),
            "physical_start": event_start,
            "physical_end": event_end,
            "logical_start": logical_start,
            "logical_end": logical_end,
            "token_ids_sha256": hashlib.sha256(
                b"".join(int(x).to_bytes(8, "little", signed=True) for x in ids)
            ).hexdigest(),
        })
        if physical_cursor == stop_at:
            break
    _require(physical_cursor == stop_at, f"execution ended at {physical_cursor} != {stop_at}")
    _require(cache is not None and last_logits is not None, "execution produced no cache/logits")
    snapshot = snapshot_cache(cache, detach=not enable_grad)
    _require(snapshot_physical_length(snapshot) == stop_at,
             "executed cache physical length differs")
    return ExecutionResult(snapshot, last_logits, calls, q1_lps,
                           physical_cursor, logical_end)


def execute_replay_plan(model, plan: ReplayPlan, *, stop_at: int | None = None) -> ExecutionResult:
    plan.validate()
    stop = len(plan.token_ids) if stop_at is None else int(stop_at)
    return _run_events(model, plan.token_ids, plan.events, stop_at=stop,
                       destination=False)


def execute_fresh_plan(model, plan: FreshDestinationPlan, *,
                       stop_at: int | None = None) -> ExecutionResult:
    plan.validate()
    stop = len(plan.token_ids) if stop_at is None else int(stop_at)
    return _run_events(model, plan.token_ids, plan.events, stop_at=stop,
                       destination=True)


def continue_fresh_plan(model, plan: FreshDestinationPlan, boundary: Snapshot, *,
                        start_at: int) -> ExecutionResult:
    plan.validate()
    _require(snapshot_physical_length(boundary) == start_at,
             "boundary snapshot length differs from continuation start")
    return _run_events(model, plan.token_ids, plan.events,
                       stop_at=len(plan.token_ids), destination=True,
                       initial_snapshot=boundary)


def probe_suffix_ids(tokenizer, context_messages: list[dict],
                     context_ids: Sequence[int], probe: str) -> list[int]:
    messages = list(context_messages) + [{"role": "user", "content": probe}]
    prefix = [int(x) for x in generation_prefix_ids(tokenizer, messages)]
    context = [int(x) for x in context_ids]
    _require(prefix[:len(context)] == context,
             "probe generation prefix does not extend exact visible context")
    suffix = prefix[len(context):]
    _require(bool(suffix), "probe generation suffix is empty")
    return suffix


def score_target_q1(model, snapshot: Snapshot, *, suffix_ids: Sequence[int],
                    target_ids: Sequence[int], logical_context_end: int) -> dict:
    targets = [int(x) for x in target_ids]
    _require(bool(targets), "target token sequence is empty")
    cache = rebuild_cache(snapshot)
    physical = snapshot_physical_length(snapshot)
    suffix = [int(x) for x in suffix_ids]
    cache, logits = _forward(
        model, cache, suffix,
        range(logical_context_end, logical_context_end + len(suffix)),
        range(physical, physical + len(suffix)), enable_grad=False)
    logical = logical_context_end + len(suffix)
    physical += len(suffix)
    logprobs = []
    for index, token_id in enumerate(targets):
        lp = torch.log_softmax(logits.float(), dim=-1)[0, token_id]
        value = float(lp.detach().cpu())
        _require(math.isfinite(value), "target log probability is nonfinite")
        logprobs.append(value)
        if index + 1 < len(targets):
            cache, logits = _forward(
                model, cache, [token_id], [logical], [physical], enable_grad=False)
            logical += 1
            physical += 1
    return {
        "target_token_ids": targets,
        "token_logprobs": logprobs,
        "mean_logprob": sum(logprobs) / len(logprobs),
        "probe_suffix_ids": suffix,
        "teacher_forcing_feed_ids": suffix + targets[:-1],
        "logical_feed_positions": list(range(
            logical_context_end, logical_context_end + len(suffix) + len(targets) - 1)),
        "physical_feed_positions": list(range(
            snapshot_physical_length(snapshot),
            snapshot_physical_length(snapshot) + len(suffix) + len(targets) - 1)),
    }


def _float32_bits(value: torch.Tensor) -> str:
    scalar = value.detach().to(device="cpu", dtype=torch.float32).reshape(())
    return struct.pack("<f", float(scalar)).hex()


def _candidate(logits: torch.Tensor) -> tuple[int, torch.Tensor]:
    values = logits.float()
    token_id = int(torch.argmax(values, dim=-1).item())
    logprob = torch.log_softmax(values, dim=-1)[0, token_id]
    _require(torch.isfinite(logprob).item(), "generation candidate logprob is nonfinite")
    return token_id, logprob


def greedy_generate_q1(model, prefix_snapshot: Snapshot, prefix_logits: torch.Tensor, *,
                       logical_start: int, eos_ids: Sequence[int],
                       max_content_tokens: int = 64) -> GenerationResult:
    eos = sorted({int(x) for x in eos_ids})
    _require(bool(eos) and all(x >= 0 for x in eos), "EOS set is empty/invalid")
    _require(max_content_tokens == 64, "v12 generation cap must equal 64")
    cache = rebuild_cache(prefix_snapshot)
    physical = snapshot_physical_length(prefix_snapshot)
    logical = int(logical_start)
    logits = prefix_logits.detach().clone()
    ids: list[int] = []
    lps: list[float] = []
    bits: list[str] = []
    logical_positions: list[int] = []
    physical_positions: list[int] = []
    for count in range(max_content_tokens + 1):
        token_id, lp = _candidate(logits)
        if token_id in eos:
            return GenerationResult(
                snapshot=snapshot_cache(cache), content_ids=ids,
                token_logprobs=lps, token_logprob_float32_bits=bits,
                logical_positions=logical_positions,
                physical_positions=physical_positions,
                stop_reason="model_eos", stop_candidate_id=token_id,
                stop_candidate_logprob=float(lp.detach().cpu()),
                stop_candidate_logprob_float32_bits=_float32_bits(lp),
                eos_ids=eos, cap_hit=False)
        if count == max_content_tokens:
            return GenerationResult(
                snapshot=snapshot_cache(cache), content_ids=ids,
                token_logprobs=lps, token_logprob_float32_bits=bits,
                logical_positions=logical_positions,
                physical_positions=physical_positions,
                stop_reason="max_content_tokens", stop_candidate_id=token_id,
                stop_candidate_logprob=float(lp.detach().cpu()),
                stop_candidate_logprob_float32_bits=_float32_bits(lp),
                eos_ids=eos, cap_hit=True)
        ids.append(token_id)
        lps.append(float(lp.detach().cpu()))
        bits.append(_float32_bits(lp))
        logical_positions.append(logical)
        physical_positions.append(physical)
        cache, logits = _forward(
            model, cache, [token_id], [logical], [physical], enable_grad=False)
        logical += 1
        physical += 1
    raise CanaryRuntimeError("unreachable generation loop exit")


def force_content_q1(model, prefix_snapshot: Snapshot, prefix_logits: torch.Tensor, *,
                     content_ids: Sequence[int], logical_start: int,
                     eos_ids: Sequence[int]) -> GenerationResult:
    ids = [int(x) for x in content_ids]
    _require(bool(ids), "forced identity content is empty")
    eos = sorted({int(x) for x in eos_ids})
    _require(bool(eos), "EOS set is empty")
    _require(not set(ids).intersection(eos), "forced content contains EOS")
    cache = rebuild_cache(prefix_snapshot)
    physical = snapshot_physical_length(prefix_snapshot)
    logical = int(logical_start)
    logits = prefix_logits.detach().clone()
    lps: list[float] = []
    bits: list[str] = []
    logical_positions: list[int] = []
    physical_positions: list[int] = []
    for token_id in ids:
        lp = torch.log_softmax(logits.float(), dim=-1)[0, token_id]
        _require(torch.isfinite(lp).item(), "forced content logprob is nonfinite")
        lps.append(float(lp.detach().cpu()))
        bits.append(_float32_bits(lp))
        logical_positions.append(logical)
        physical_positions.append(physical)
        cache, logits = _forward(
            model, cache, [token_id], [logical], [physical], enable_grad=False)
        logical += 1
        physical += 1
    stop_id, stop_lp = _candidate(logits)
    return GenerationResult(
        snapshot=snapshot_cache(cache), content_ids=ids,
        token_logprobs=lps, token_logprob_float32_bits=bits,
        logical_positions=logical_positions, physical_positions=physical_positions,
        stop_reason="model_eos" if stop_id in eos else "forced_stop_not_eos",
        stop_candidate_id=stop_id,
        stop_candidate_logprob=float(stop_lp.detach().cpu()),
        stop_candidate_logprob_float32_bits=_float32_bits(stop_lp),
        eos_ids=eos, cap_hit=False)


def require_generated_forced_identity(generated: GenerationResult,
                                      forced: GenerationResult, *,
                                      content_start: int) -> dict:
    _require(generated.stop_reason == "model_eos" and not generated.cap_hit,
             "generated identity branch did not stop normally")
    _require(bool(generated.content_ids), "generated identity content is empty")
    _require(generated.content_ids == forced.content_ids,
             "generated/forced content IDs differ")
    _require(generated.logical_positions == forced.logical_positions and
             generated.physical_positions == forced.physical_positions,
             "generated/forced content positions differ")
    _require(generated.token_logprob_float32_bits ==
             forced.token_logprob_float32_bits,
             "generated/forced token logprob bits differ")
    _require(generated.stop_candidate_id == forced.stop_candidate_id and
             generated.stop_candidate_logprob_float32_bits ==
             forced.stop_candidate_logprob_float32_bits and
             forced.stop_candidate_id in forced.eos_ids,
             "generated/forced EOS witness differs")
    end = content_start + len(generated.content_ids)
    generated_rows = extract_rows(
        generated.snapshot, content_start, end,
        max_rows=64, to_cpu=True)
    forced_rows = extract_rows(
        forced.snapshot, content_start, end,
        max_rows=64, to_cpu=True)
    _require(len(generated_rows) == len(forced_rows),
             "generated/forced layer counts differ")
    per_layer = []
    for index, ((gk, gv), (fk, fv)) in enumerate(zip(generated_rows, forced_rows)):
        _require(torch.equal(gk, fk) and torch.equal(gv, fv),
                 f"generated/forced K/V rows differ at layer {index}")
        per_layer.append({
            "layer": index,
            "k_shape": list(gk.shape), "v_shape": list(gv.shape),
            "k_dtype": str(gk.dtype), "v_dtype": str(gv.dtype),
            "k_sha256": tensor_sha256(gk), "v_sha256": tensor_sha256(gv),
        })
    return {
        "status": "GENERATED_FORCED_IDENTITY_PASS",
        "content_ids": list(generated.content_ids),
        "content_start": content_start,
        "content_end": end,
        "token_logprob_float32_bits": list(
            generated.token_logprob_float32_bits),
        "stop_candidate_id": generated.stop_candidate_id,
        "stop_candidate_logprob_float32_bits":
            generated.stop_candidate_logprob_float32_bits,
        "per_layer_content_rows": per_layer,
    }
