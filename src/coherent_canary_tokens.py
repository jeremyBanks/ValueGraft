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


def _assistant_content_bounds(tokenizer, messages: list[dict], ids: list[int],
                              starts: list[int], index: int) -> tuple[int, int]:
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
    return header_end, content_end


def build_role_native_plan(tokenizer, history_messages: list[dict], *,
                           middle_end_msg: int | None = None,
                           carrier_request: str = ENGINEERED_CARRIER_REQUEST,
                           carrier_content: str = ENGINEERED_CARRIER_CONTENT,
                           anchor_user: str = ANCHOR_USER,
                           anchor_assistant: str = ANCHOR_ASSISTANT) -> ReplayPlan:
    if not history_messages or history_messages[0].get("role") != "system":
        raise CanarySchemaError("history must start with a system message")
    if history_messages[-1].get("role") != "assistant":
        raise CanarySchemaError("history must end with an assistant message")
    if middle_end_msg is None:
        middle_end_msg = len(history_messages)
    if not isinstance(middle_end_msg, int) or not (
            2 < middle_end_msg <= len(history_messages)):
        raise CanarySchemaError("middle_end_msg is outside the history")
    prefix = deepcopy(history_messages[:middle_end_msg])
    retained_tail = deepcopy(history_messages[middle_end_msg:])
    if prefix[-1].get("role") != "assistant":
        raise CanarySchemaError("evicted prefix must end with an assistant message")
    if retained_tail and retained_tail[0].get("role") != "user":
        raise CanarySchemaError("retained tail must begin with a user message")
    messages = prefix + [
        {"role": "user", "content": carrier_request},
        {"role": "assistant", "content": carrier_content},
        {"role": "user", "content": anchor_user},
        {"role": "assistant", "content": anchor_assistant},
    ] + retained_tail
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
    assistant_content_bounds: dict[int, tuple[int, int]] = {}
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            assistant_content_bounds[index] = _assistant_content_bounds(
                tokenizer, messages, ids, starts, index)

    assistant_indices = sorted(assistant_content_bounds)
    if not assistant_indices:
        raise CanarySchemaError("canonical stream has no assistant message")

    # Match a persistent chat generation schedule rather than tokenizing each
    # role fragment as its own call.  The first production prefill contains the
    # system/user material and the first assistant generation header.  After an
    # assistant response is forced q=1, the next production addition contains
    # that response's canonical close, the next user message, and the next
    # assistant generation header in one structural prefill.  Splitting those
    # pieces would change query shape—the exact confound this schedule exists to
    # control.
    events: list[ReplayEvent] = []
    first_index = assistant_indices[0]
    first_content_start = assistant_content_bounds[first_index][0]
    events.extend(_bounded_prefill_events(
        "initial_generation_prefix", "structural", first_index,
        0, first_content_start))
    for ordinal, index in enumerate(assistant_indices):
        content_start, content_end = assistant_content_bounds[index]
        events.extend(ReplayEvent(
            "q1", "assistant_content", "assistant", index, position,
            position + 1).validate() for position in range(content_start, content_end))
        if ordinal + 1 < len(assistant_indices):
            next_index = assistant_indices[ordinal + 1]
            next_content_start = assistant_content_bounds[next_index][0]
            events.extend(_bounded_prefill_events(
                "turn_continuation", "structural", next_index,
                content_end, next_content_start))
        else:
            events.extend(_bounded_prefill_events(
                "final_assistant_close", "structural", index,
                content_end, len(ids)))

    carrier_index = middle_end_msg + 1
    anchor_assistant_index = middle_end_msg + 3
    content_start, content_end = assistant_content_bounds[carrier_index]
    anchor_content_start, anchor_content_end = assistant_content_bounds[
        anchor_assistant_index]
    if anchor_content_end >= len(ids):
        raise CanarySchemaError("anchor assistant has no canonical close suffix")
    return ReplayPlan(
        token_ids=ids,
        message_start_positions=starts,
        events=events,
        regions=CarrierRegions(
            content_start=content_start,
            content_end=content_end,
            anchor_prefix_end=anchor_content_start,
            anchor_content_end=anchor_content_end,
        ),
    ).validate()
