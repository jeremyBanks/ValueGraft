"""Tokenizer-only dynamic-carrier planning for powered successor v13."""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

from arms_common import canonical_ids_any, render_hf
from coherent_state_tokens import (
    generation_prefix_ids,
    rendered_assistant_content_ids,
)
from powered_v13_schema import (
    ANCHOR_ASSISTANT,
    ANCHOR_USER,
    CARRIER_REQUEST,
    R1,
    CarrierRegions,
    DestinationEvent,
    FreshDestinationPlan,
    ReplayEvent,
    ReplayPlan,
    V13SchemaError,
)


def _plain_token_ids(
    values: object,
    label: str,
    *,
    allow_empty: bool = False,
    maximum_token_id: int | None = None,
) -> list[int]:
    if (not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))):
        raise V13SchemaError(f"{label} is not a token-ID sequence")
    result = list(values)
    if not allow_empty and not result:
        raise V13SchemaError(f"{label} is empty")
    if any(type(value) is not int or value < 0 for value in result):
        raise V13SchemaError(
            f"{label} contains a non-plain or negative token ID")
    if maximum_token_id is not None:
        if type(maximum_token_id) is not int or maximum_token_id < 0:
            raise V13SchemaError(f"{label} token-ID maximum is invalid")
        if any(value > maximum_token_id for value in result):
            raise V13SchemaError(
                f"{label} contains an out-of-vocabulary token ID")
    return result


def _maximum_token_id(tokenizer) -> int:
    try:
        width = len(tokenizer)
    except (TypeError, AttributeError) as exc:
        raise V13SchemaError("tokenizer vocabulary width is unavailable") from exc
    if type(width) is not int or width <= 0:
        raise V13SchemaError("tokenizer vocabulary width is invalid")
    return width - 1


def _single_id(tokenizer, text: str) -> int:
    ids = _plain_token_ids(
        tokenizer.encode(text, add_special_tokens=False),
        f"boundary {text!r}",
        maximum_token_id=_maximum_token_id(tokenizer),
    )
    if len(ids) != 1:
        raise V13SchemaError(f"boundary {text!r} is not one token: {ids}")
    return ids[0]


def message_starts(tokenizer, ids: Sequence[int]) -> list[int]:
    frozen_ids = _plain_token_ids(
        ids, "canonical message stream",
        maximum_token_id=_maximum_token_id(tokenizer),
    )
    marker = _single_id(tokenizer, "<|im_start|>")
    return [index for index, value in enumerate(frozen_ids) if value == marker]


def _bounded_prefill_events(label: str, role: str, message_index: int,
                            start: int, end: int) -> list[ReplayEvent]:
    if end <= start:
        raise V13SchemaError(f"empty prefill event: {label}")
    events = []
    cursor = start
    piece = 0
    while cursor < end:
        stop = min(end, cursor + 4096)
        suffix = f"_{piece}" if piece else ""
        events.append(ReplayEvent(
            "prefill", f"{label}{suffix}", role, message_index,
            cursor, stop).validate())
        cursor = stop
        piece += 1
    return events


def _assistant_content_bounds(tokenizer, messages: list[dict], ids: list[int],
                              starts: list[int], index: int) -> tuple[int, int]:
    preceding = messages[:index]
    content = str(messages[index].get("content", ""))
    if not content:
        raise V13SchemaError(f"assistant message {index} has empty content")
    header_end = len(_plain_token_ids(
        generation_prefix_ids(tokenizer, preceding),
        f"assistant message {index} generation prefix",
        maximum_token_id=_maximum_token_id(tokenizer),
    ))
    content_ids = _plain_token_ids(
        rendered_assistant_content_ids(tokenizer, preceding, content),
        f"assistant message {index} content",
        maximum_token_id=_maximum_token_id(tokenizer),
    )
    content_end = header_end + len(content_ids)
    message_end = starts[index + 1] if index + 1 < len(starts) else len(ids)
    if not (starts[index] < header_end < content_end <= message_end):
        raise V13SchemaError(
            f"assistant message {index} boundaries are invalid: "
            f"{starts[index]}, {header_end}, {content_end}, {message_end}")
    if ids[header_end:content_end] != content_ids:
        raise V13SchemaError(
            f"assistant message {index} content IDs differ from canonical stream")
    return header_end, content_end


def complete_source_messages(
    history_messages: list[dict],
    *,
    middle_end_msg: int,
    carrier_content: str,
    carrier_request: str = CARRIER_REQUEST,
    anchor_user: str = ANCHOR_USER,
    anchor_assistant: str = ANCHOR_ASSISTANT,
) -> list[dict]:
    """Insert one literal carrier/anchor block before the retained tail."""
    if not history_messages or history_messages[0].get("role") != "system":
        raise V13SchemaError("history must start with a system message")
    if history_messages[-1].get("role") != "assistant":
        raise V13SchemaError("history must end with an assistant message")
    if type(middle_end_msg) is not int or not (
            2 < middle_end_msg < len(history_messages)):
        raise V13SchemaError("middle_end_msg is outside the history")
    if not isinstance(carrier_content, str) or not carrier_content.strip():
        raise V13SchemaError("dynamic carrier content is empty")
    prefix = deepcopy(history_messages[:middle_end_msg])
    retained_tail = deepcopy(history_messages[middle_end_msg:])
    if prefix[-1].get("role") != "assistant":
        raise V13SchemaError("evicted prefix must end with an assistant message")
    if retained_tail[0].get("role") != "user":
        raise V13SchemaError("retained tail must begin with a user message")
    messages = prefix + [
        {"role": "user", "content": carrier_request},
        {"role": "assistant", "content": carrier_content},
        {"role": "user", "content": anchor_user},
        {"role": "assistant", "content": anchor_assistant},
    ] + retained_tail
    for index, message in enumerate(messages):
        expected = "system" if index == 0 else (
            "user" if index % 2 else "assistant")
        if message.get("role") != expected:
            raise V13SchemaError(
                f"message {index} role {message.get('role')} != {expected}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise V13SchemaError(f"message {index} content is empty")
        if "<|im_" in content:
            raise V13SchemaError(
                f"message {index} contains a chat-template literal")
    return messages


def build_role_native_plan(
    tokenizer,
    history_messages: list[dict],
    *,
    middle_end_msg: int,
    carrier_content: str,
    carrier_request: str = CARRIER_REQUEST,
    anchor_user: str = ANCHOR_USER,
    anchor_assistant: str = ANCHOR_ASSISTANT,
) -> ReplayPlan:
    messages = complete_source_messages(
        history_messages,
        middle_end_msg=middle_end_msg,
        carrier_content=carrier_content,
        carrier_request=carrier_request,
        anchor_user=anchor_user,
        anchor_assistant=anchor_assistant,
    )
    ids = _plain_token_ids(
        canonical_ids_any(tokenizer, messages, render_hf),
        "role-native canonical stream",
        maximum_token_id=_maximum_token_id(tokenizer),
    )
    starts = message_starts(tokenizer, ids)
    if len(starts) != len(messages) or starts[0] != 0:
        raise V13SchemaError("canonical message-start coverage differs")
    assistant_bounds: dict[int, tuple[int, int]] = {}
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            assistant_bounds[index] = _assistant_content_bounds(
                tokenizer, messages, ids, starts, index)
    assistant_indices = sorted(assistant_bounds)
    if not assistant_indices:
        raise V13SchemaError("canonical stream has no assistant message")

    events: list[ReplayEvent] = []
    first_index = assistant_indices[0]
    first_content_start = assistant_bounds[first_index][0]
    events.extend(_bounded_prefill_events(
        "initial_generation_prefix", "structural", first_index,
        0, first_content_start))
    for ordinal, index in enumerate(assistant_indices):
        content_start, content_end = assistant_bounds[index]
        events.extend(ReplayEvent(
            "q1", "assistant_content", "assistant", index,
            position, position + 1).validate()
            for position in range(content_start, content_end))
        if ordinal + 1 < len(assistant_indices):
            next_index = assistant_indices[ordinal + 1]
            next_content_start = assistant_bounds[next_index][0]
            events.extend(_bounded_prefill_events(
                "turn_continuation", "structural", next_index,
                content_end, next_content_start))
        else:
            events.extend(_bounded_prefill_events(
                "final_assistant_close", "structural", index,
                content_end, len(ids)))

    carrier_index = middle_end_msg + 1
    anchor_assistant_index = middle_end_msg + 3
    content_start, content_end = assistant_bounds[carrier_index]
    anchor_start, anchor_end = assistant_bounds[anchor_assistant_index]
    if anchor_end >= len(ids):
        raise V13SchemaError("anchor assistant has no canonical close suffix")
    return ReplayPlan(
        token_ids=ids,
        message_start_positions=starts,
        events=events,
        regions=CarrierRegions(
            content_start=content_start,
            content_end=content_end,
            anchor_prefix_end=anchor_start,
            anchor_content_end=anchor_end,
        ),
    ).validate()


def build_turn_aligned_plan(
    tokenizer,
    history_messages: list[dict],
    *,
    middle_end_msg: int,
    carrier_content: str,
    carrier_request: str = CARRIER_REQUEST,
    anchor_user: str = ANCHOR_USER,
    anchor_assistant: str = ANCHOR_ASSISTANT,
) -> ReplayPlan:
    source = build_role_native_plan(
        tokenizer,
        history_messages,
        middle_end_msg=middle_end_msg,
        carrier_content=carrier_content,
        carrier_request=carrier_request,
        anchor_user=anchor_user,
        anchor_assistant=anchor_assistant,
    )
    events: list[ReplayEvent] = []
    for index in range(middle_end_msg):
        start = source.message_start_positions[index]
        end = source.message_start_positions[index + 1]
        role = str(history_messages[index].get("role", ""))
        events.extend(_bounded_prefill_events(
            "historical_message", role, index, start, end))
    carrier_request_start = source.message_start_positions[middle_end_msg]
    events.extend(_bounded_prefill_events(
        "carrier_request_and_header", "structural", middle_end_msg,
        carrier_request_start, source.regions.content_start))
    events.extend(event for event in source.events
                  if event.token_start >= source.regions.content_start)
    return ReplayPlan(
        token_ids=list(source.token_ids),
        message_start_positions=list(source.message_start_positions),
        events=events,
        regions=source.regions,
    ).validate()


def build_fresh_destination_plan(
    tokenizer,
    history_messages: list[dict],
    *,
    middle_end_msg: int,
    carrier_content: str,
    carrier_request: str = CARRIER_REQUEST,
    anchor_user: str = ANCHOR_USER,
    anchor_assistant: str = ANCHOR_ASSISTANT,
) -> FreshDestinationPlan:
    """Build packed physical rows at the source stream's logical positions."""
    source = build_role_native_plan(
        tokenizer,
        history_messages,
        middle_end_msg=middle_end_msg,
        carrier_content=carrier_content,
        carrier_request=carrier_request,
        anchor_user=anchor_user,
        anchor_assistant=anchor_assistant,
    )
    starts = source.message_start_positions
    system_width = starts[1]
    suffix_source_start = starts[middle_end_msg]
    source_indices = (
        list(range(system_width)) +
        list(range(suffix_source_start, len(source.token_ids)))
    )
    token_ids = [source.token_ids[index] for index in source_indices]
    full_messages = complete_source_messages(
        history_messages,
        middle_end_msg=middle_end_msg,
        carrier_content=carrier_content,
        carrier_request=carrier_request,
        anchor_user=anchor_user,
        anchor_assistant=anchor_assistant,
    )
    compact_messages = deepcopy(full_messages[:1]) + deepcopy(
        full_messages[middle_end_msg:])
    compact_ids = _plain_token_ids(
        canonical_ids_any(tokenizer, compact_messages, render_hf),
        "fresh compact canonical stream",
        maximum_token_id=_maximum_token_id(tokenizer),
    )
    if compact_ids != token_ids:
        raise V13SchemaError("fresh compact render differs from source mapping")

    def physical(source_position: int) -> int:
        if source_position < suffix_source_start:
            if source_position >= system_width:
                raise V13SchemaError("source position lies in the evicted gap")
            return source_position
        return system_width + source_position - suffix_source_start

    events: list[DestinationEvent] = [DestinationEvent(
        "prefill", "retained_system", "system", 0,
        0, system_width, 0, system_width).validate()]
    for event in _bounded_prefill_events(
        "carrier_request_and_header", "structural", middle_end_msg,
        suffix_source_start, source.regions.content_start,
    ):
        events.append(DestinationEvent(
            event.kind, event.label, event.role, event.message_index,
            physical(event.token_start), physical(event.token_end),
            event.token_start, event.token_end).validate())
    for event in source.events:
        if event.token_start < source.regions.content_start:
            continue
        events.append(DestinationEvent(
            event.kind, event.label, event.role, event.message_index,
            physical(event.token_start), physical(event.token_end),
            event.token_start, event.token_end).validate())

    def physical_regions(regions: CarrierRegions) -> CarrierRegions:
        return CarrierRegions(
            content_start=physical(regions.content_start),
            content_end=physical(regions.content_end),
            anchor_prefix_end=physical(regions.anchor_prefix_end),
            anchor_content_end=physical(regions.anchor_content_end),
        ).validate()

    return FreshDestinationPlan(
        token_ids=token_ids,
        logical_positions=list(source_indices),
        physical_positions=list(range(len(token_ids))),
        source_token_indices=list(source_indices),
        events=events,
        source_regions=source.regions,
        physical_regions=physical_regions(source.regions),
        system_width=system_width,
        suffix_source_start=suffix_source_start,
    ).validate()


def source_noncarrier_token_count(plan: ReplayPlan) -> int:
    """Literal Section-4 skeleton metric for one complete source plan."""
    plan.validate()
    start, end = plan.regions.interval(R1)
    return len(plan.token_ids) - (end - start)


def build_probe_generation_prefix(
    tokenizer,
    completed_source_messages: list[dict],
    probe: str,
) -> list[int]:
    """Canonical source + focal/nonfocal user probe + assistant header."""
    if (not completed_source_messages or
            completed_source_messages[-1].get("role") != "assistant"):
        raise V13SchemaError("completed source must end with assistant")
    if not isinstance(probe, str) or not probe.strip():
        raise V13SchemaError("probe is empty")
    messages = deepcopy(completed_source_messages) + [
        {"role": "user", "content": probe},
    ]
    return _plain_token_ids(
        generation_prefix_ids(tokenizer, messages),
        "probe generation prefix",
        maximum_token_id=_maximum_token_id(tokenizer),
    )


def probe_target_ids(
    tokenizer,
    completed_source_messages: list[dict],
    probe: str,
    target: str,
) -> list[int]:
    """Exact assistant-content IDs beginning at the probe answer position."""
    if not isinstance(target, str) or not target.strip():
        raise V13SchemaError("probe target is empty")
    preceding = deepcopy(completed_source_messages) + [
        {"role": "user", "content": probe},
    ]
    return _plain_token_ids(
        rendered_assistant_content_ids(tokenizer, preceding, target),
        "probe target content",
        maximum_token_id=_maximum_token_id(tokenizer),
    )
