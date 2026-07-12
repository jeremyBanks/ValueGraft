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
    executed_token_ids: list[int]
    logical_positions: list[int]
    physical_positions: list[int]
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


@dataclass
class GradientResult:
    fresh_rows: Snapshot
    gradients: Snapshot
    baseline_margin: float
    baseline_margin_float32_bits: str
    correct_logprob: float
    correct_logprob_float32_bits: str
    counterfactual_logprob: float
    counterfactual_logprob_float32_bits: str
    physical_region: tuple[int, int]
    logical_region: tuple[int, int]
    probe_suffix_ids: list[int]
    gradient_boundary_calls: list[dict]
    gradient_continuation_calls: list[dict]
    public_full_calls: list[dict]
    gradient_boundary_token_ids: list[int]
    gradient_boundary_logical_positions: list[int]
    gradient_boundary_physical_positions: list[int]
    gradient_continuation_token_ids: list[int]
    gradient_continuation_logical_positions: list[int]
    gradient_continuation_physical_positions: list[int]
    public_full_token_ids: list[int]
    public_full_logical_positions: list[int]
    public_full_physical_positions: list[int]
    probe_logical_positions: list[int]
    probe_physical_positions: list[int]


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
    executed_ids: list[int] = []
    executed_logical: list[int] = []
    executed_physical: list[int] = []
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
                           "token_id": token_id,
                           "logprob": float(lp.detach().cpu()),
                           "logprob_float32_bits": _float32_bits(lp)})
        cache, last_logits = _forward(
            model, cache, ids, range(logical_start, logical_end),
            range(event_start, event_end), enable_grad=enable_grad)
        physical_cursor = event_end
        executed_ids.extend(ids)
        executed_logical.extend(range(logical_start, logical_end))
        executed_physical.extend(range(event_start, event_end))
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
                           executed_ids, executed_logical, executed_physical,
                           physical_cursor, logical_end)


def execute_replay_plan(model, plan: ReplayPlan, *, stop_at: int | None = None) -> ExecutionResult:
    plan.validate()
    stop = len(plan.token_ids) if stop_at is None else int(stop_at)
    return _run_events(model, plan.token_ids, plan.events, stop_at=stop,
                       destination=False)


def execute_prefix_block(model, token_ids: Sequence[int], *,
                         logical_positions: Sequence[int] | None = None,
                         label: str = "generation_prefix") -> ExecutionResult:
    ids = [int(x) for x in token_ids]
    _require(0 < len(ids) <= 4096, "prefix block width lies outside 1..4096")
    logical = (list(range(len(ids))) if logical_positions is None else
               [int(x) for x in logical_positions])
    _require(len(logical) == len(ids) and all(
        right == left + 1 for left, right in zip(logical, logical[1:])),
        "prefix logical positions are not one contiguous span")
    cache, logits = _forward(
        model, None, ids, logical, range(len(ids)), enable_grad=False)
    snapshot = snapshot_cache(cache)
    event = {
        "kind": "prefill", "label": label, "role": "structural",
        "message_index": 0, "physical_start": 0,
        "physical_end": len(ids), "logical_start": logical[0],
        "logical_end": logical[-1] + 1,
        "token_ids_sha256": hashlib.sha256(b"".join(
            int(x).to_bytes(8, "little", signed=True) for x in ids
        )).hexdigest(),
    }
    return ExecutionResult(
        snapshot=snapshot, last_logits=logits, calls=[event],
        q1_token_logprobs=[], executed_token_ids=ids,
        logical_positions=logical, physical_positions=list(range(len(ids))),
        physical_end=len(ids), logical_end=logical[-1] + 1)


def append_block_to_snapshot(model, snapshot: Snapshot, token_ids: Sequence[int], *,
                             logical_start: int,
                             label: str = "structural_suffix") -> ExecutionResult:
    ids = [int(x) for x in token_ids]
    _require(0 < len(ids) <= 4096, "appended block width lies outside 1..4096")
    physical_start = snapshot_physical_length(snapshot)
    _require(physical_start + len(ids) <= MAX_LIVE_CACHE_TOKENS,
             "appended block would exceed live-cache bound")
    logical = list(range(int(logical_start), int(logical_start) + len(ids)))
    physical = list(range(physical_start, physical_start + len(ids)))
    cache = rebuild_cache(snapshot)
    cache, logits = _forward(
        model, cache, ids, logical, physical, enable_grad=False)
    event = {
        "kind": "prefill", "label": label, "role": "structural",
        "message_index": 0, "physical_start": physical_start,
        "physical_end": physical_start + len(ids),
        "logical_start": logical[0], "logical_end": logical[-1] + 1,
        "token_ids_sha256": hashlib.sha256(b"".join(
            int(x).to_bytes(8, "little", signed=True) for x in ids
        )).hexdigest(),
    }
    return ExecutionResult(
        snapshot=snapshot_cache(cache), last_logits=logits, calls=[event],
        q1_token_logprobs=[], executed_token_ids=ids,
        logical_positions=logical, physical_positions=physical,
        physical_end=physical_start + len(ids), logical_end=logical[-1] + 1)


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
    """Append one immutable probe prefix, then score a target from its fork."""
    suffix = [int(x) for x in suffix_ids]
    _require(bool(suffix), "target-scoring probe suffix is empty")
    prefix = append_block_to_snapshot(
        model, snapshot, suffix, logical_start=logical_context_end,
        label="probe_user_and_assistant_header")
    result = score_target_from_prefix_q1(
        model, prefix.snapshot, prefix.last_logits, target_ids=target_ids,
        logical_target_start=prefix.logical_end)
    targets = result["target_token_ids"]
    result.update({
        "probe_suffix_ids": suffix,
        "teacher_forcing_feed_ids": suffix + targets[:-1],
        "logical_feed_positions": list(range(
            logical_context_end,
            logical_context_end + len(suffix) + len(targets) - 1)),
        "physical_feed_positions": list(range(
            snapshot_physical_length(snapshot),
            snapshot_physical_length(snapshot) + len(suffix) + len(targets) - 1)),
        "probe_prefix_row_hashes": snapshot_hashes(prefix.snapshot),
        "probe_prefix_last_logits_sha256": tensor_sha256(prefix.last_logits),
    })
    return result


def score_target_from_prefix_q1(
        model, prefix_snapshot: Snapshot, prefix_logits: torch.Tensor, *,
        target_ids: Sequence[int], logical_target_start: int) -> dict:
    """Score a target from a prebuilt immutable assistant-generation prefix.

    This is the production primitive used when multiple targets and generation
    must fork from one exact probe-prefix cache/logit state.
    """
    targets = [int(x) for x in target_ids]
    _require(bool(targets), "target token sequence is empty")
    cache = rebuild_cache(prefix_snapshot)
    physical = snapshot_physical_length(prefix_snapshot)
    _require(physical + len(targets) - 1 <= MAX_LIVE_CACHE_TOKENS,
             "target scoring would exceed live-cache bound")
    logits = prefix_logits.detach().clone()
    logical = int(logical_target_start)
    logprobs = []
    logprob_bits = []
    logprob_tensors = []
    for index, token_id in enumerate(targets):
        lp = torch.log_softmax(logits.float(), dim=-1)[0, token_id]
        value = float(lp.detach().cpu())
        _require(math.isfinite(value), "target log probability is nonfinite")
        logprobs.append(value)
        logprob_bits.append(_float32_bits(lp))
        logprob_tensors.append(lp.detach().to(device="cpu", dtype=torch.float32))
        if index + 1 < len(targets):
            cache, logits = _forward(
                model, cache, [token_id], [logical], [physical], enable_grad=False)
            logical += 1
            physical += 1
    mean = torch.stack(logprob_tensors).mean(dtype=torch.float32)
    return {
        "target_token_ids": targets,
        "token_logprobs": logprobs,
        "token_logprob_float32_bits": logprob_bits,
        "mean_logprob": float(mean),
        "mean_logprob_float32_bits": _float32_bits(mean),
        "probe_suffix_ids": [],
        "teacher_forcing_feed_ids": targets[:-1],
        "logical_feed_positions": list(range(
            logical_target_start, logical_target_start + len(targets) - 1)),
        "physical_feed_positions": list(range(
            snapshot_physical_length(prefix_snapshot),
            snapshot_physical_length(prefix_snapshot) + len(targets) - 1)),
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
    _require(physical + max_content_tokens <= MAX_LIVE_CACHE_TOKENS,
             "generation would exceed live-cache bound")
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
    _require(physical + len(ids) <= MAX_LIVE_CACHE_TOKENS,
             "forced content would exceed live-cache bound")
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


def require_generated_forced_identity(
        generated_prefix: ExecutionResult, generated: GenerationResult,
        forced_prefix: ExecutionResult, forced: GenerationResult, *,
        content_start: int) -> dict:
    _require(generated_prefix.executed_token_ids == forced_prefix.executed_token_ids,
             "generated/forced prefix token IDs differ")
    _require(generated_prefix.logical_positions == forced_prefix.logical_positions and
             generated_prefix.physical_positions == forced_prefix.physical_positions,
             "generated/forced prefix positions differ")
    _require(generated_prefix.calls == forced_prefix.calls,
             "generated/forced prefix events differ")
    _require(generated_prefix.logical_end == forced_prefix.logical_end,
             "generated/forced prefix logical ends differ")
    _require(generated_prefix.physical_end == forced_prefix.physical_end == content_start,
             "generated/forced content_start is not the exact prefix end")
    _require(snapshot_hashes(generated_prefix.snapshot) ==
             snapshot_hashes(forced_prefix.snapshot),
             "generated/forced prefix K/V rows differ")
    _require(tensor_sha256(generated_prefix.last_logits) ==
             tensor_sha256(forced_prefix.last_logits),
             "generated/forced prefix logits differ")
    _require(bool(generated.content_ids), "generated identity content is empty")
    _require(generated.content_ids == forced.content_ids,
             "generated/forced content IDs differ")
    _require(generated.logical_positions == forced.logical_positions and
             generated.physical_positions == forced.physical_positions,
             "generated/forced content positions differ")
    _require(generated.eos_ids == forced.eos_ids,
             "generated/forced EOS sets differ")
    _require(generated.physical_positions and
             generated.physical_positions[0] == content_start,
             "generated content rows do not start at prefix end")
    expected_logical = list(range(
        generated_prefix.logical_end,
        generated_prefix.logical_end + len(generated.content_ids)))
    _require(generated.logical_positions == forced.logical_positions ==
             expected_logical,
             "generated/forced content logical positions do not start at prefix end")
    _require(generated.token_logprob_float32_bits ==
             forced.token_logprob_float32_bits,
             "generated/forced token logprob bits differ")
    _require(generated.stop_candidate_id == forced.stop_candidate_id and
             generated.stop_candidate_logprob_float32_bits ==
             forced.stop_candidate_logprob_float32_bits,
             "generated/forced stop-candidate witness differs")
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
    _require(generated.stop_reason == forced.stop_reason == "model_eos" and
             not generated.cap_hit and not forced.cap_hit and
             generated.stop_candidate_id in generated.eos_ids,
             "generated identity branch did not stop normally")
    return {
        "status": "GENERATED_FORCED_IDENTITY_PASS",
        "prefix_token_ids": list(generated_prefix.executed_token_ids),
        "prefix_logical_positions": list(generated_prefix.logical_positions),
        "prefix_physical_positions": list(generated_prefix.physical_positions),
        "prefix_calls": list(generated_prefix.calls),
        "prefix_row_hashes": snapshot_hashes(generated_prefix.snapshot),
        "prefix_last_logits_sha256": tensor_sha256(
            generated_prefix.last_logits),
        "eos_ids": list(generated.eos_ids),
        "content_ids": list(generated.content_ids),
        "content_logical_positions": list(generated.logical_positions),
        "content_physical_positions": list(generated.physical_positions),
        "content_start": content_start,
        "content_end": end,
        "token_logprob_float32_bits": list(
            generated.token_logprob_float32_bits),
        "stop_candidate_id": generated.stop_candidate_id,
        "stop_candidate_logprob_float32_bits":
            generated.stop_candidate_logprob_float32_bits,
        "per_layer_content_rows": per_layer,
    }


def _selected_leaf_boundary(snapshot: Snapshot, start: int, end: int):
    _require(0 <= start < end <= snapshot_physical_length(snapshot),
             "gradient selected interval is invalid")
    full: Snapshot = []
    leaves: list[torch.Tensor] = []
    fresh_rows: Snapshot = []
    for keys, values in snapshot:
        key_leaf = keys[..., start:end, :].detach().clone().requires_grad_(True)
        value_leaf = values[..., start:end, :].detach().clone().requires_grad_(True)
        full_keys = torch.cat(
            (keys[..., :start, :].detach(), key_leaf,
             keys[..., end:, :].detach()), dim=-2)
        full_values = torch.cat(
            (values[..., :start, :].detach(), value_leaf,
             values[..., end:, :].detach()), dim=-2)
        full.append((full_keys, full_values))
        leaves.extend((key_leaf, value_leaf))
        fresh_rows.append((key_leaf.detach().cpu().clone(),
                           value_leaf.detach().cpu().clone()))
    return full, leaves, fresh_rows


def _one_prefix_margin(model, snapshot: Snapshot, *, suffix_ids: Sequence[int],
                       logical_context_end: int, correct_id: int,
                       counterfactual_id: int, enable_grad: bool):
    cache = rebuild_cache(snapshot, clone=not enable_grad)
    physical = snapshot_physical_length(snapshot)
    suffix = [int(x) for x in suffix_ids]
    cache, logits = _forward(
        model, cache, suffix,
        range(logical_context_end, logical_context_end + len(suffix)),
        range(physical, physical + len(suffix)), enable_grad=enable_grad)
    logprobs = torch.log_softmax(logits.float(), dim=-1)[0]
    correct = logprobs[int(correct_id)]
    counterfactual = logprobs[int(counterfactual_id)]
    return correct - counterfactual, correct, counterfactual


def collect_fresh_region_margin_gradients(
        model, plan: FreshDestinationPlan, *, region: str,
        suffix_ids: Sequence[int], correct_id: int, counterfactual_id: int,
) -> GradientResult:
    """Differentiate one immutable one-token margin through a fresh region.

    Model weights must already be frozen. Only the selected cached K/V rows are
    leaves. The function independently reruns the public no-grad path and
    requires bit-exact baseline margin/log-probability equality.
    """
    _require(all(not parameter.requires_grad for parameter in model.parameters()),
             "model weights are not frozen for path control")
    physical_start, physical_end = plan.physical_regions.interval(region)
    logical_start, logical_end = plan.source_regions.interval(region)
    boundary = execute_fresh_plan(model, plan, stop_at=physical_end)
    leaf_boundary, leaves, fresh_rows = _selected_leaf_boundary(
        boundary.snapshot, physical_start, physical_end)
    differentiated = _run_events(
        model, plan.token_ids, plan.events, stop_at=len(plan.token_ids),
        destination=True, initial_snapshot=leaf_boundary, enable_grad=True)
    logical_context_end = plan.logical_positions[-1] + 1
    margin, correct, counterfactual = _one_prefix_margin(
        model, differentiated.snapshot, suffix_ids=suffix_ids,
        logical_context_end=logical_context_end, correct_id=correct_id,
        counterfactual_id=counterfactual_id, enable_grad=True)
    raw_gradients = torch.autograd.grad(
        margin, leaves, allow_unused=False, retain_graph=False, create_graph=False)
    gradients: Snapshot = []
    for layer in range(len(fresh_rows)):
        key_gradient = raw_gradients[2 * layer].detach().cpu().clone()
        value_gradient = raw_gradients[2 * layer + 1].detach().cpu().clone()
        _require(torch.isfinite(key_gradient).all().item() and
                 torch.isfinite(value_gradient).all().item(),
                 f"nonfinite path-control gradient at layer {layer}")
        gradients.append((key_gradient, value_gradient))
    grad_margin_bits = _float32_bits(margin)
    grad_correct_bits = _float32_bits(correct)
    grad_counterfactual_bits = _float32_bits(counterfactual)
    differentiated_calls = list(differentiated.calls)
    differentiated_ids = list(differentiated.executed_token_ids)
    differentiated_logical = list(differentiated.logical_positions)
    differentiated_physical = list(differentiated.physical_positions)
    del differentiated, leaf_boundary, leaves, raw_gradients

    public = execute_fresh_plan(model, plan)
    public_margin, public_correct, public_counterfactual = _one_prefix_margin(
        model, public.snapshot, suffix_ids=suffix_ids,
        logical_context_end=logical_context_end, correct_id=correct_id,
        counterfactual_id=counterfactual_id, enable_grad=False)
    _require(_float32_bits(public_margin) == grad_margin_bits and
             _float32_bits(public_correct) == grad_correct_bits and
             _float32_bits(public_counterfactual) == grad_counterfactual_bits,
             "gradient baseline differs from public no-grad path")
    return GradientResult(
        fresh_rows=fresh_rows, gradients=gradients,
        baseline_margin=float(public_margin.detach().cpu()),
        baseline_margin_float32_bits=_float32_bits(public_margin),
        correct_logprob=float(public_correct.detach().cpu()),
        correct_logprob_float32_bits=_float32_bits(public_correct),
        counterfactual_logprob=float(public_counterfactual.detach().cpu()),
        counterfactual_logprob_float32_bits=_float32_bits(public_counterfactual),
        physical_region=(physical_start, physical_end),
        logical_region=(logical_start, logical_end),
        probe_suffix_ids=[int(x) for x in suffix_ids],
        gradient_boundary_calls=list(boundary.calls),
        gradient_continuation_calls=differentiated_calls,
        public_full_calls=list(public.calls),
        gradient_boundary_token_ids=list(boundary.executed_token_ids),
        gradient_boundary_logical_positions=list(boundary.logical_positions),
        gradient_boundary_physical_positions=list(boundary.physical_positions),
        gradient_continuation_token_ids=differentiated_ids,
        gradient_continuation_logical_positions=differentiated_logical,
        gradient_continuation_physical_positions=differentiated_physical,
        public_full_token_ids=list(public.executed_token_ids),
        public_full_logical_positions=list(public.logical_positions),
        public_full_physical_positions=list(public.physical_positions),
        probe_logical_positions=list(range(
            logical_context_end, logical_context_end + len(suffix_ids))),
        probe_physical_positions=list(range(
            len(plan.token_ids), len(plan.token_ids) + len(suffix_ids))),
    )
