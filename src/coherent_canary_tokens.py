"""Tokenizer-only role-native event and carrier-region construction for v12."""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

from arms_common import canonical_ids_any, render_hf
from coherent_canary_schema import (
    ANCHOR_ASSISTANT,
    ANCHOR_USER,
    ENGINEERED_CARRIER_CONTENT,
    ENGINEERED_CARRIER_REQUEST,
    CanarySchemaError,
    CarrierRegions,
    ReplayEvent,
    ReplayPlan,
)
from coherent_state_tokens import (
    generation_prefix_ids,
    rendered_assistant_content_ids,
)


def _single_id(tokenizer, text: str) -> int:
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if len(ids) != 1:
        raise CanarySchemaError(f"boundary {text!r} is not one token: {ids}")
    return int(ids[0])


def message_starts(tokenizer, ids: Sequence[int]) -> list[int]:
    marker = _single_id(tokenizer, "<|im_start|>")
    return [index for index, value in enumerate(ids) if int(value) == marker]


def _bounded_prefill_events(label: str, role: str, message_index: int,
                            start: int, end: int) -> list[ReplayEvent]:
    if end <= start:
        raise CanarySchemaError(f"empty prefill event: {label}")
    out = []
    cursor = start
    piece = 0
    while cursor < end:
        stop = min(end, cursor + 4096)
        suffix = f"_{piece}" if piece else ""
        out.append(ReplayEvent(
            "prefill", f"{label}{suffix}", role, message_index,
            cursor, stop).validate())
        cursor = stop
        piece += 1
    return out


def _assistant_events(tokenizer, messages: list[dict], ids: list[int],
                      starts: list[int], index: int) -> list[ReplayEvent]:
    preceding = messages[:index]
    content = str(messages[index].get("content", ""))
    if not content:
        raise CanarySchemaError(f"assistant message {index} has empty content")
    header_end = len(generation_prefix_ids(tokenizer, preceding))
    content_ids = rendered_assistant_content_ids(
        tokenizer, preceding, content)
    content_end = header_end + len(content_ids)
    message_end = starts[index + 1] if index + 1 < len(starts) else len(ids)
    if not (starts[index] < header_end < content_end < message_end + 1):
        raise CanarySchemaError(
            f"assistant message {index} boundaries are invalid: "
            f"{starts[index]}, {header_end}, {content_end}, {message_end}")
    if ids[header_end:content_end] != content_ids:
        raise CanarySchemaError(
            f"assistant message {index} content IDs differ from canonical stream")
    events = _bounded_prefill_events(
        "assistant_open", "assistant", index, starts[index], header_end)
    events.extend(ReplayEvent(
        "q1", "assistant_content", "assistant", index, position,
        position + 1).validate() for position in range(header_end, content_end))
    events.extend(_bounded_prefill_events(
        "assistant_close", "assistant", index, content_end, message_end))
    return events


def build_role_native_plan(tokenizer, history_messages: list[dict], *,
                           carrier_request: str = ENGINEERED_CARRIER_REQUEST,
                           carrier_content: str = ENGINEERED_CARRIER_CONTENT,
                           anchor_user: str = ANCHOR_USER,
                           anchor_assistant: str = ANCHOR_ASSISTANT) -> ReplayPlan:
    if not history_messages or history_messages[0].get("role") != "system":
        raise CanarySchemaError("history must start with a system message")
    if history_messages[-1].get("role") != "assistant":
        raise CanarySchemaError("history must end with an assistant message")
    messages = deepcopy(history_messages) + [
        {"role": "user", "content": carrier_request},
        {"role": "assistant", "content": carrier_content},
        {"role": "user", "content": anchor_user},
        {"role": "assistant", "content": anchor_assistant},
    ]
    for index, message in enumerate(messages):
        expected = "system" if index == 0 else ("user" if index % 2 else "assistant")
        if message.get("role") != expected:
            raise CanarySchemaError(
                f"message {index} role {message.get('role')} != {expected}")
    ids = [int(value) for value in canonical_ids_any(
        tokenizer, messages, render_hf)]
    starts = message_starts(tokenizer, ids)
    if len(starts) != len(messages) or starts[0] != 0:
        raise CanarySchemaError(
            f"canonical message start coverage {len(starts)} != {len(messages)}")
    events: list[ReplayEvent] = []
    assistant_content_bounds: dict[int, tuple[int, int]] = {}
    for index, message in enumerate(messages):
        message_end = starts[index + 1] if index + 1 < len(starts) else len(ids)
        if message["role"] == "assistant":
            built = _assistant_events(tokenizer, messages, ids, starts, index)
            q1 = [event for event in built if event.kind == "q1"]
            assistant_content_bounds[index] = (
                q1[0].token_start, q1[-1].token_end)
            events.extend(built)
        else:
            events.extend(_bounded_prefill_events(
                f"{message['role']}_message", message["role"], index,
                starts[index], message_end))

    carrier_index = len(history_messages) + 1
    anchor_assistant_index = len(messages) - 1
    content_start, content_end = assistant_content_bounds[carrier_index]
    close_end = starts[carrier_index + 1]
    anchor_end = len(ids)
    if assistant_content_bounds[anchor_assistant_index][1] >= anchor_end:
        raise CanarySchemaError("anchor assistant has no canonical close suffix")
    return ReplayPlan(
        token_ids=ids,
        message_start_positions=starts,
        events=events,
        regions=CarrierRegions(
            content_start=content_start,
            content_end=content_end,
            close_end=close_end,
            anchor_end=anchor_end,
        ),
    ).validate()
