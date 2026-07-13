"""Direct-local MLX replay and scoring primitives for the local N48 study.

The token planners, rather than this module, define the model-call schedule.
For a source replay, every :class:`ReplayEvent` is issued literally: ``prefill``
events are one bounded multi-token call and ``q1`` events are one single-token
call.  A q1 token's log probability is taken from the preceding call's logits
*before* that token is fed.  The fresh replay uses the same rule over
``DestinationEvent`` records.  It stores the system and visible suffix in a
packed physical cache while advancing RoPE/cache ``offset`` across the evicted
logical gap.

Snapshots use the existing ``kvlib`` representation ``(keys, values, offset)``.
The K/V arrays contain packed physical rows; ``offset`` is the next logical
position and can therefore exceed the physical row count for a fresh replay.
Alignment pairs follow the existing MLX convention
``(destination_physical_position, source_logical_position)``.

This module deliberately contains no treatment, placebo, eligibility, or
Phase-A policy.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence, TypeAlias

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from arms_common import canonical_ids_any, render_hf
from kvlib import GappedKVCache, snapshot_cache
from powered_v13_schema import (
    DestinationEvent,
    FreshDestinationPlan,
    ReplayEvent,
    ReplayPlan,
    require_matching_geometry,
)
from powered_v13_tokens import (
    build_fresh_destination_plan,
    build_probe_generation_prefix,
    build_role_native_plan,
    complete_source_messages,
    probe_target_ids,
)


Snapshot: TypeAlias = list[tuple[mx.array, mx.array, int]]
AlignmentPair: TypeAlias = tuple[int, int]


class LocalN48CoreError(RuntimeError):
    """A local replay, cache, alignment, or probe contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalN48CoreError(message)


def _plain_ids(values: object, label: str, *, allow_empty: bool = False) \
        -> list[int]:
    _require(
        isinstance(values, Sequence) and not isinstance(values, (str, bytes)),
        f"{label} is not a token-ID sequence",
    )
    result = list(values)
    _require(allow_empty or bool(result), f"{label} is empty")
    _require(
        all(type(value) is int and value >= 0 for value in result),
        f"{label} contains a non-plain or negative token ID",
    )
    return result


def _planning_tokenizer(tokenizer):
    """Return the callable HF tokenizer inside an MLX TokenizerWrapper.

    MLX deliberately exposes encode/decode/template methods through a thin
    non-callable wrapper, while the frozen v13 token planners also use the HF
    batch-call API. Unwrapping does not change tokenizer bytes or behavior.
    """
    # HF tokenizers are already callable and may themselves expose a private
    # non-callable Rust backend named ``_tokenizer``. Only unwrap when the
    # outer object is not callable (the MLX wrapper case).
    candidate = tokenizer if callable(tokenizer) else getattr(
        tokenizer, "_tokenizer", None)
    _require(callable(candidate),
             "tokenizer is neither callable nor an MLX wrapper with _tokenizer")
    return candidate


def _fixture_history(
    fixture: Mapping[str, Any], variant: str,
) -> tuple[list[dict[str, str]], int]:
    _require(isinstance(fixture, Mapping), "fixture is not a mapping")
    _require(variant in ("C", "W"), "variant must be exactly 'C' or 'W'")
    variants = fixture.get("variants")
    _require(isinstance(variants, Mapping), "fixture variants are absent")
    row = variants.get(variant)
    _require(isinstance(row, Mapping), f"fixture variant {variant} is absent")
    raw_messages = row.get("messages")
    _require(
        isinstance(raw_messages, list) and len(raw_messages) >= 5,
        f"fixture variant {variant} messages are invalid",
    )
    messages: list[dict[str, str]] = []
    for index, raw in enumerate(raw_messages):
        _require(
            isinstance(raw, Mapping) and set(raw) == {"role", "content"},
            f"fixture {variant} message {index} fields differ",
        )
        role, content = raw.get("role"), raw.get("content")
        _require(
            role in ("system", "user", "assistant")
            and isinstance(content, str) and bool(content.strip()),
            f"fixture {variant} message {index} is invalid",
        )
        messages.append({"role": str(role), "content": content})
    middle = fixture.get("middle_end_msg")
    _require(
        type(middle) is int and 2 < middle < len(messages),
        "fixture middle_end_msg is invalid",
    )
    return messages, middle


def _model_layer_count(model) -> int:
    layers = getattr(model, "layers", None)
    try:
        count = len(layers)
    except (TypeError, AttributeError) as exc:
        raise LocalN48CoreError("model layers are unavailable") from exc
    _require(count > 0, "model has no layers")
    return count


def _new_standard_cache(model) -> list[KVCache]:
    return [KVCache() for _ in range(_model_layer_count(model))]


def _cache_geometry(cache) -> tuple[int, int]:
    _require(bool(cache), "cache has no layers")
    physical_lengths: list[int] = []
    logical_offsets: list[int] = []
    for layer, item in enumerate(cache):
        state = getattr(item, "state", None)
        _require(
            isinstance(state, tuple) and len(state) == 2,
            f"cache layer {layer} has no K/V state",
        )
        keys, values = state
        _require(
            isinstance(keys, mx.array) and isinstance(values, mx.array)
            and keys.ndim == values.ndim == 4,
            f"cache layer {layer} is not [B,H,T,D] K/V",
        )
        _require(
            keys.shape[:3] == values.shape[:3],
            f"cache layer {layer} K/V prefix geometry differs",
        )
        offset = getattr(item, "offset", None)
        _require(
            type(offset) is int and offset >= int(keys.shape[2]),
            f"cache layer {layer} logical offset is invalid",
        )
        physical_lengths.append(int(keys.shape[2]))
        logical_offsets.append(offset)
    _require(
        len(set(physical_lengths)) == 1,
        "cache physical lengths differ across layers",
    )
    _require(
        len(set(logical_offsets)) == 1,
        "cache logical offsets differ across layers",
    )
    return physical_lengths[0], logical_offsets[0]


def _forward(model, cache, token_ids: Sequence[int]) -> mx.array:
    ids = _plain_ids(token_ids, "model call token IDs")
    logits = model(mx.array([ids]), cache=cache)
    _require(
        isinstance(logits, mx.array) and logits.ndim == 3
        and logits.shape[0] == 1 and logits.shape[1] == len(ids)
        and logits.shape[2] > 0,
        "model call logits geometry differs",
    )
    last_logits = logits[:, -1, :]
    mx.eval(last_logits)
    mx.eval([item.state for item in cache])
    return last_logits


def _token_logprob(last_logits: mx.array, token_id: int) -> float:
    _require(
        type(token_id) is int and 0 <= token_id < int(last_logits.shape[-1]),
        "scored token ID is outside the logits vocabulary",
    )
    values = last_logits.astype(mx.float32)
    logprobs = values - mx.logsumexp(values, axis=-1, keepdims=True)
    result = float(logprobs[0, token_id].item())
    _require(math.isfinite(result), "token log probability is nonfinite")
    return result


def _call_record(
    event: ReplayEvent | DestinationEvent,
    token_ids: list[int],
    *,
    physical_start: int,
    physical_end: int,
    logical_start: int,
    logical_end: int,
) -> dict[str, Any]:
    return {
        **asdict(event),
        "physical_start": physical_start,
        "physical_end": physical_end,
        "logical_start": logical_start,
        "logical_end": logical_end,
        "token_ids": list(token_ids),
    }


def _carrier_logprob_record(
    last_logits: mx.array,
    token_id: int,
    *,
    physical_position: int,
    logical_position: int,
) -> dict[str, Any]:
    return {
        "physical_position": physical_position,
        "logical_position": logical_position,
        "token_id": token_id,
        "logprob": _token_logprob(last_logits, token_id),
    }


def _validate_snapshot(snapshot: Snapshot, label: str = "snapshot") \
        -> tuple[int, int]:
    _require(isinstance(snapshot, list) and bool(snapshot), f"{label} is empty")
    physical_lengths: list[int] = []
    logical_offsets: list[int] = []
    for layer, row in enumerate(snapshot):
        _require(
            isinstance(row, tuple) and len(row) == 3,
            f"{label} layer {layer} is not a (K,V,offset) tuple",
        )
        keys, values, offset = row
        _require(
            isinstance(keys, mx.array) and isinstance(values, mx.array)
            and keys.ndim == values.ndim == 4,
            f"{label} layer {layer} is not [B,H,T,D] K/V",
        )
        _require(
            keys.shape[:3] == values.shape[:3]
            and int(keys.shape[2]) > 0,
            f"{label} layer {layer} K/V geometry differs",
        )
        _require(
            type(offset) is int and offset >= int(keys.shape[2]),
            f"{label} layer {layer} logical offset is invalid",
        )
        physical_lengths.append(int(keys.shape[2]))
        logical_offsets.append(offset)
    _require(
        len(set(physical_lengths)) == 1,
        f"{label} physical lengths differ across layers",
    )
    _require(
        len(set(logical_offsets)) == 1,
        f"{label} logical offsets differ across layers",
    )
    return physical_lengths[0], logical_offsets[0]


def _rebuild_snapshot(snapshot: Snapshot):
    physical, logical = _validate_snapshot(snapshot)
    cache = []
    for keys, values, offset in snapshot:
        copied_keys, copied_values = mx.array(keys), mx.array(values)
        if logical == physical:
            item = KVCache()
            item.keys = copied_keys
            item.values = copied_values
            item.offset = offset
        else:
            item = GappedKVCache(copied_keys, copied_values, offset)
        cache.append(item)
    return cache


@dataclass
class ReplayResult:
    """One exact source or fresh replay boundary."""

    mode: str
    variant: str
    plan: ReplayPlan | FreshDestinationPlan
    token_ids: list[int]
    context_messages: list[dict[str, str]]
    snapshot: Snapshot
    terminal_next_logits: mx.array
    calls: list[dict[str, Any]]
    carrier_token_logprobs: list[dict[str, Any]]
    physical_end: int
    logical_end: int

    @property
    def plan_geometry(self) -> dict[str, Any]:
        return self.plan.geometry()

    @property
    def complete_messages(self) -> list[dict[str, str]]:
        """Exact messages whose canonical render equals ``token_ids``."""
        return self.context_messages


@dataclass
class ValueAlignment:
    """Exact visible-token value-row twins for correct and wrong donors."""

    correct_pairs: list[AlignmentPair]
    wrong_pairs: list[AlignmentPair]
    content_correct_pairs: list[AlignmentPair]
    content_wrong_pairs: list[AlignmentPair]
    structural_correct_pairs: list[AlignmentPair]
    structural_wrong_pairs: list[AlignmentPair]
    rows: list[dict[str, Any]]
    destination_token_ids: list[int]
    destination_logical_positions: list[int]
    system_width: int
    content_physical_interval: tuple[int, int]
    source_token_counts: dict[str, int]


def replay_source(
    model,
    tokenizer,
    fixture: Mapping[str, Any],
    variant: str,
    carrier_text: str,
) -> ReplayResult:
    """Execute a C or W role-native plan with its literal event schedule."""

    planner = _planning_tokenizer(tokenizer)
    history, middle = _fixture_history(fixture, variant)
    plan = build_role_native_plan(
        planner,
        history,
        middle_end_msg=middle,
        carrier_content=carrier_text,
    ).validate()
    cache = _new_standard_cache(model)
    calls: list[dict[str, Any]] = []
    carrier_lps: list[dict[str, Any]] = []
    last_logits: mx.array | None = None
    carrier_start, carrier_end = plan.regions.content_start, plan.regions.content_end

    for event in plan.events:
        ids = _plain_ids(
            plan.token_ids[event.token_start:event.token_end],
            f"source {event.label} token IDs",
        )
        if event.kind == "q1" and carrier_start <= event.token_start < carrier_end:
            _require(last_logits is not None, "carrier q1 lacks preceding logits")
            carrier_lps.append(_carrier_logprob_record(
                last_logits,
                ids[0],
                physical_position=event.token_start,
                logical_position=event.token_start,
            ))
        last_logits = _forward(model, cache, ids)
        physical, logical = _cache_geometry(cache)
        _require(
            physical == event.token_end and logical == event.token_end,
            f"source event {event.label} cache end differs",
        )
        calls.append(_call_record(
            event,
            ids,
            physical_start=event.token_start,
            physical_end=event.token_end,
            logical_start=event.token_start,
            logical_end=event.token_end,
        ))

    expected_carrier_ids = plan.token_ids[carrier_start:carrier_end]
    _require(
        [row["token_id"] for row in carrier_lps] == expected_carrier_ids,
        "source carrier q1 logprob coverage differs",
    )
    snapshot = snapshot_cache(cache)
    physical, logical = _validate_snapshot(snapshot, "source snapshot")
    _require(
        physical == logical == len(plan.token_ids),
        "source terminal snapshot geometry differs",
    )
    _require(last_logits is not None, "source terminal next logits are absent")
    messages = complete_source_messages(
        history,
        middle_end_msg=middle,
        carrier_content=carrier_text,
    )
    canonical = _plain_ids(
        canonical_ids_any(planner, messages, render_hf),
        "source complete-message render",
    )
    _require(canonical == plan.token_ids, "source complete-message IDs differ")
    return ReplayResult(
        mode="source",
        variant=variant,
        plan=plan,
        token_ids=list(plan.token_ids),
        context_messages=messages,
        snapshot=snapshot,
        terminal_next_logits=mx.array(last_logits),
        calls=calls,
        carrier_token_logprobs=carrier_lps,
        physical_end=physical,
        logical_end=logical,
    )


def replay_fresh(
    model,
    tokenizer,
    fixture: Mapping[str, Any],
    variant: str,
    carrier_text: str,
) -> ReplayResult:
    """Execute the packed fresh plan while retaining source logical positions."""

    planner = _planning_tokenizer(tokenizer)
    history, middle = _fixture_history(fixture, variant)
    plan = build_fresh_destination_plan(
        planner,
        history,
        middle_end_msg=middle,
        carrier_content=carrier_text,
    ).validate()
    cache = _new_standard_cache(model)
    calls: list[dict[str, Any]] = []
    carrier_lps: list[dict[str, Any]] = []
    last_logits: mx.array | None = None
    carrier_start = plan.physical_regions.content_start
    carrier_end = plan.physical_regions.content_end

    for ordinal, event in enumerate(plan.events):
        _require(
            event.physical_start == (calls[-1]["physical_end"] if calls else 0),
            f"fresh event {event.label} does not continue physical rows",
        )
        if ordinal == 1:
            physical, logical = _cache_geometry(cache)
            _require(
                physical == plan.system_width and logical == plan.system_width,
                "fresh retained-system cache geometry differs",
            )
            _require(
                event.logical_start == plan.suffix_source_start,
                "fresh first suffix event does not begin at the logical gap end",
            )
            cache = [
                GappedKVCache(mx.array(item.state[0]), mx.array(item.state[1]),
                              event.logical_start)
                for item in cache
            ]

        ids = _plain_ids(
            plan.token_ids[event.physical_start:event.physical_end],
            f"fresh {event.label} token IDs",
        )
        if event.kind == "q1" and carrier_start <= event.physical_start < carrier_end:
            _require(last_logits is not None, "fresh carrier q1 lacks preceding logits")
            carrier_lps.append(_carrier_logprob_record(
                last_logits,
                ids[0],
                physical_position=event.physical_start,
                logical_position=event.logical_start,
            ))
        last_logits = _forward(model, cache, ids)
        physical, logical = _cache_geometry(cache)
        _require(
            physical == event.physical_end and logical == event.logical_end,
            f"fresh event {event.label} cache end differs",
        )
        calls.append(_call_record(
            event,
            ids,
            physical_start=event.physical_start,
            physical_end=event.physical_end,
            logical_start=event.logical_start,
            logical_end=event.logical_end,
        ))

    expected_carrier_ids = plan.token_ids[carrier_start:carrier_end]
    _require(
        [row["token_id"] for row in carrier_lps] == expected_carrier_ids,
        "fresh carrier q1 logprob coverage differs",
    )
    snapshot = snapshot_cache(cache)
    physical, logical = _validate_snapshot(snapshot, "fresh snapshot")
    _require(
        physical == len(plan.token_ids)
        and logical == plan.logical_positions[-1] + 1,
        "fresh terminal snapshot geometry differs",
    )
    _require(last_logits is not None, "fresh terminal next logits are absent")
    full_messages = complete_source_messages(
        history,
        middle_end_msg=middle,
        carrier_content=carrier_text,
    )
    compact_messages = deepcopy(full_messages[:1]) + deepcopy(full_messages[middle:])
    canonical = _plain_ids(
        canonical_ids_any(planner, compact_messages, render_hf),
        "fresh compact-message render",
    )
    _require(canonical == plan.token_ids, "fresh compact-message IDs differ")
    return ReplayResult(
        mode="fresh",
        variant=variant,
        plan=plan,
        token_ids=list(plan.token_ids),
        context_messages=compact_messages,
        snapshot=snapshot,
        terminal_next_logits=mx.array(last_logits),
        calls=calls,
        carrier_token_logprobs=carrier_lps,
        physical_end=physical,
        logical_end=logical,
    )


def snapshots_bit_exact(left: Snapshot, right: Snapshot) -> bool:
    """Return true only for identical offsets, dtypes, shapes, and raw bits."""

    _validate_snapshot(left, "left snapshot")
    _validate_snapshot(right, "right snapshot")
    if len(left) != len(right):
        return False
    for (left_k, left_v, left_offset), (right_k, right_v, right_offset) in zip(
            left, right):
        if left_offset != right_offset:
            return False
        for left_value, right_value in ((left_k, right_k), (left_v, right_v)):
            if (left_value.shape != right_value.shape
                    or left_value.dtype != right_value.dtype):
                return False
            left_bits = mx.contiguous(left_value).view(mx.uint8)
            right_bits = mx.contiguous(right_value).view(mx.uint8)
            if not bool(mx.array_equal(left_bits, right_bits).item()):
                return False
    return True


def arrays_bit_exact(left: mx.array, right: mx.array) -> bool:
    """Return true only for identical MLX array dtypes, shapes, and raw bits."""

    _require(isinstance(left, mx.array), "left value is not an MLX array")
    _require(isinstance(right, mx.array), "right value is not an MLX array")
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_bits = mx.contiguous(left).view(mx.uint8)
    right_bits = mx.contiguous(right).view(mx.uint8)
    return bool(mx.array_equal(left_bits, right_bits).item())


def _special_ids(tokenizer) -> set[int]:
    raw = getattr(tokenizer, "all_special_ids", None)
    _require(isinstance(raw, Sequence), "tokenizer special IDs are unavailable")
    values = set(_plain_ids(raw, "tokenizer special IDs", allow_empty=True))
    return values


def build_value_alignment_pairs(
    tokenizer,
    fixture: Mapping[str, Any],
    carrier_text: str,
) -> ValueAlignment:
    """Build exact non-system, non-special visible-row twins for C and W."""

    planner = _planning_tokenizer(tokenizer)
    correct_history, correct_middle = _fixture_history(fixture, "C")
    wrong_history, wrong_middle = _fixture_history(fixture, "W")
    _require(correct_middle == wrong_middle, "C/W compaction boundaries differ")
    correct = build_role_native_plan(
        planner,
        correct_history,
        middle_end_msg=correct_middle,
        carrier_content=carrier_text,
    ).validate()
    wrong = build_role_native_plan(
        planner,
        wrong_history,
        middle_end_msg=wrong_middle,
        carrier_content=carrier_text,
    ).validate()
    require_matching_geometry(correct, wrong)
    fresh_correct = build_fresh_destination_plan(
        planner,
        correct_history,
        middle_end_msg=correct_middle,
        carrier_content=carrier_text,
    ).validate()
    fresh_wrong = build_fresh_destination_plan(
        planner,
        wrong_history,
        middle_end_msg=wrong_middle,
        carrier_content=carrier_text,
    ).validate()
    _require(
        fresh_correct == fresh_wrong,
        "C/W packed fresh plans differ",
    )

    special_ids = _special_ids(planner)
    content_start, content_end = fresh_correct.physical_regions.content_start, \
        fresh_correct.physical_regions.content_end
    correct_pairs: list[AlignmentPair] = []
    wrong_pairs: list[AlignmentPair] = []
    content_correct: list[AlignmentPair] = []
    content_wrong: list[AlignmentPair] = []
    structural_correct: list[AlignmentPair] = []
    structural_wrong: list[AlignmentPair] = []
    rows: list[dict[str, Any]] = []

    for destination, source_position in enumerate(fresh_correct.logical_positions):
        token_id = fresh_correct.token_ids[destination]
        if destination < fresh_correct.system_width or token_id in special_ids:
            continue
        _require(
            source_position >= fresh_correct.suffix_source_start,
            "aligned non-system row lies inside the evicted gap",
        )
        _require(
            source_position < len(correct.token_ids)
            and source_position < len(wrong.token_ids),
            "aligned source position exceeds a donor stream",
        )
        _require(
            correct.token_ids[source_position]
            == wrong.token_ids[source_position]
            == token_id,
            "visible destination/source token IDs differ",
        )
        _require(token_id not in special_ids, "aligned donor token is special")
        correct_pair = (destination, source_position)
        wrong_pair = (destination, source_position)
        correct_pairs.append(correct_pair)
        wrong_pairs.append(wrong_pair)
        partition = "content" if content_start <= destination < content_end \
            else "structural"
        if partition == "content":
            content_correct.append(correct_pair)
            content_wrong.append(wrong_pair)
        else:
            structural_correct.append(correct_pair)
            structural_wrong.append(wrong_pair)
        rows.append({
            "destination_physical_position": destination,
            "source_logical_position": source_position,
            "token_id": token_id,
            "partition": partition,
        })

    _require(bool(correct_pairs), "alignment has no eligible visible rows")
    _require(bool(content_correct), "alignment has no carrier-content rows")
    _require(bool(structural_correct), "alignment has no structural rows")
    _require(
        correct_pairs == wrong_pairs
        and content_correct == content_wrong
        and structural_correct == structural_wrong,
        "correct/wrong alignment geometry differs",
    )
    _require(
        set(correct_pairs) == set(content_correct).union(structural_correct)
        and not set(content_correct).intersection(structural_correct),
        "content/structural alignment partition differs",
    )
    return ValueAlignment(
        correct_pairs=correct_pairs,
        wrong_pairs=wrong_pairs,
        content_correct_pairs=content_correct,
        content_wrong_pairs=content_wrong,
        structural_correct_pairs=structural_correct,
        structural_wrong_pairs=structural_wrong,
        rows=rows,
        destination_token_ids=list(fresh_correct.token_ids),
        destination_logical_positions=list(fresh_correct.logical_positions),
        system_width=fresh_correct.system_width,
        content_physical_interval=(content_start, content_end),
        source_token_counts={
            "C": len(correct.token_ids),
            "W": len(wrong.token_ids),
            "fresh": len(fresh_correct.token_ids),
        },
    )


def _score_target_from_prefix(
    model,
    prefix_snapshot: Snapshot,
    prefix_logits: mx.array,
    target_ids: Sequence[int],
    target_text: str,
) -> dict[str, Any]:
    ids = _plain_ids(target_ids, f"target {target_text!r} content IDs")
    cache = _rebuild_snapshot(prefix_snapshot)
    logits = mx.array(prefix_logits)
    logprobs: list[float] = []
    for index, token_id in enumerate(ids):
        logprobs.append(_token_logprob(logits, token_id))
        if index + 1 < len(ids):
            logits = _forward(model, cache, [token_id])
    return {
        "text": target_text,
        "content_token_ids": ids,
        "token_logprobs": logprobs,
        "mean_logprob": sum(logprobs) / len(logprobs),
    }


def _eos_ids(tokenizer) -> set[int]:
    raw = getattr(tokenizer, "eos_token_ids", None)
    if raw is None:
        single = getattr(tokenizer, "eos_token_id", None)
        raw = [] if single is None else [single]
    elif type(raw) is int:
        raw = [raw]
    values = set(_plain_ids(raw, "tokenizer EOS IDs"))
    _require(bool(values), "tokenizer EOS set is empty")
    return values


def _greedy_from_prefix(
    model,
    prefix_snapshot: Snapshot,
    prefix_logits: mx.array,
    eos_ids: set[int],
    cap: int,
) -> dict[str, Any]:
    cache = _rebuild_snapshot(prefix_snapshot)
    logits = mx.array(prefix_logits)
    selected: list[int] = []
    content: list[int] = []
    stop_reason = "cap"
    for _ in range(cap):
        token_id = int(mx.argmax(logits, axis=-1).item())
        selected.append(token_id)
        if token_id in eos_ids:
            stop_reason = "eos"
            break
        content.append(token_id)
        logits = _forward(model, cache, [token_id])
    return {
        "greedy_token_ids": selected,
        "greedy_content_token_ids": content,
        "greedy_cap": cap,
        "greedy_stop_reason": stop_reason,
        "eos_token_ids": sorted(eos_ids),
    }


def score_probe(
    model,
    tokenizer,
    snapshot: Snapshot,
    context_messages: list[dict[str, str]],
    probe: str,
    target: str,
    countertarget: str,
    *,
    greedy_cap: int = 8,
) -> dict[str, Any]:
    """Teacher-force exact focal alternatives and greedily decode one fork."""

    planner = _planning_tokenizer(tokenizer)
    _require(
        isinstance(context_messages, list) and bool(context_messages)
        and context_messages[-1].get("role") == "assistant",
        "probe context must end with an assistant message",
    )
    _require(isinstance(probe, str) and bool(probe.strip()), "probe is empty")
    _require(
        isinstance(target, str) and bool(target.strip())
        and isinstance(countertarget, str) and bool(countertarget.strip())
        and target != countertarget,
        "target/countertarget strings are invalid",
    )
    _require(
        type(greedy_cap) is int and 1 <= greedy_cap <= 64,
        "greedy_cap lies outside 1..64",
    )
    context_ids = _plain_ids(
        canonical_ids_any(planner, context_messages, render_hf),
        "probe context IDs",
    )
    physical_context_end, logical_context_end = _validate_snapshot(
        snapshot, "probe base snapshot")
    _require(
        physical_context_end == len(context_ids),
        "probe snapshot physical rows differ from context IDs",
    )
    generation_prefix = build_probe_generation_prefix(
        planner, context_messages, probe)
    _require(
        generation_prefix[:len(context_ids)] == context_ids
        and len(generation_prefix) > len(context_ids),
        "probe generation prefix is not an exact context extension",
    )
    suffix_ids = generation_prefix[len(context_ids):]
    prefix_cache = _rebuild_snapshot(snapshot)
    prefix_logits = _forward(model, prefix_cache, suffix_ids)
    prefix_snapshot = snapshot_cache(prefix_cache)
    prefix_physical_end, prefix_logical_end = _validate_snapshot(
        prefix_snapshot, "probe prefix snapshot")
    _require(
        prefix_physical_end == physical_context_end + len(suffix_ids)
        and prefix_logical_end == logical_context_end + len(suffix_ids),
        "probe prefix snapshot geometry differs",
    )

    target_ids = probe_target_ids(
        planner, context_messages, probe, target)
    countertarget_ids = probe_target_ids(
        planner, context_messages, probe, countertarget)
    _require(target_ids != countertarget_ids, "target token sequences are equal")
    correct = _score_target_from_prefix(
        model, prefix_snapshot, prefix_logits, target_ids, target)
    counter = _score_target_from_prefix(
        model, prefix_snapshot, prefix_logits, countertarget_ids, countertarget)
    greedy = _greedy_from_prefix(
        model,
        prefix_snapshot,
        prefix_logits,
        _eos_ids(planner),
        greedy_cap,
    )
    return {
        "probe": probe,
        "context_token_ids": context_ids,
        "context_physical_end": physical_context_end,
        "context_logical_end": logical_context_end,
        "probe_suffix_token_ids": suffix_ids,
        "probe_prefix_physical_end": prefix_physical_end,
        "probe_prefix_logical_end": prefix_logical_end,
        "target": correct,
        "countertarget": counter,
        "margin": correct["mean_logprob"] - counter["mean_logprob"],
        **greedy,
    }


__all__ = [
    "AlignmentPair",
    "LocalN48CoreError",
    "ReplayResult",
    "Snapshot",
    "ValueAlignment",
    "build_value_alignment_pairs",
    "arrays_bit_exact",
    "replay_fresh",
    "replay_source",
    "score_probe",
    "snapshots_bit_exact",
]
