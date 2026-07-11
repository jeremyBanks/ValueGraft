"""Scheduled-prefix primitives for the additive paired-v11 assay.

Unlike the frozen v10 apparatus, this module treats source call partitioning as
an experimental condition.  P is derived from exact canonical message starts;
O is ordinary consecutive 4096-token prefill.  Both cover identical token IDs,
logical positions, and contiguous physical cache positions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from coherent_state_hf import (
    CoherentStateError,
    append_ids_stepwise,
    extract_summary_rows,
    generate_greedy_incremental,
    row_hashes,
    sha256_ids,
)
from coherent_state_runtime import SourceCapture, eos_ids
from coherent_state_tokens import generation_prefix_ids
from kvlib_hf import prefill


P_SCHEDULE = "turn_aligned_replay"
O_SCHEDULE = "ordinary_4096"
SCHEDULE_NAMES = (P_SCHEDULE, O_SCHEDULE)


@dataclass(frozen=True)
class PrefixSchedules:
    prefix_ids: list[int]
    message_start_positions: list[int]
    history_message_count: int
    p_conceptual_widths: list[int]
    p_call_widths: list[int]
    o_call_widths: list[int]

    def widths(self, schedule: str) -> list[int]:
        if schedule == P_SCHEDULE:
            return list(self.p_call_widths)
        if schedule == O_SCHEDULE:
            return list(self.o_call_widths)
        raise CoherentStateError(f"unknown source schedule: {schedule}")


def _content_ids(tokenizer, text: str) -> list[int]:
    return [int(value) for value in tokenizer.encode(
        text, add_special_tokens=False)]


def _message_starts(tokenizer, ids: Sequence[int]) -> list[int]:
    marker = _content_ids(tokenizer, "<|im_start|>")
    if len(marker) != 1:
        raise CoherentStateError("<|im_start|> is not one exact token")
    return [index for index, value in enumerate(ids)
            if int(value) == marker[0]]


def _chunk_widths(total: int, limit: int = 4096) -> list[int]:
    if (not isinstance(total, int) or isinstance(total, bool) or total < 1 or
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise CoherentStateError("schedule width/limit must be positive integers")
    return [min(limit, total - start) for start in range(0, total, limit)]


def derive_prefix_schedules(tokenizer, source_messages: list[dict]) \
        -> PrefixSchedules:
    """Derive exact P/O calls for a history plus final summary request.

    ``source_messages`` must end with the summary-request user message.  P uses
    one conceptual call for the system and each historical message, then one
    combined call for the final request plus assistant generation header.  Any
    conceptual call over 4096 is split only to satisfy the hard call bound.
    """
    if (len(source_messages) < 2 or
            source_messages[-1].get("role") != "user"):
        raise CoherentStateError(
            "scheduled source must contain history and end in a user request")
    prefix = generation_prefix_ids(tokenizer, source_messages)
    starts = _message_starts(tokenizer, prefix)
    expected = len(source_messages) + 1  # messages plus assistant header
    if len(starts) != expected or starts[0] != 0:
        raise CoherentStateError(
            f"generation prefix has {len(starts)} message starts, expected {expected}")
    history_count = len(source_messages) - 1
    conceptual = [starts[index + 1] - starts[index]
                  for index in range(history_count)]
    conceptual.append(len(prefix) - starts[history_count])
    if any(width < 1 for width in conceptual) or sum(conceptual) != len(prefix):
        raise CoherentStateError("P conceptual calls do not exactly cover prefix")
    p_widths = [piece for width in conceptual for piece in _chunk_widths(width)]
    o_widths = _chunk_widths(len(prefix))
    if (sum(p_widths) != len(prefix) or sum(o_widths) != len(prefix) or
            any(width > 4096 for width in p_widths + o_widths)):
        raise CoherentStateError("resolved P/O calls do not exactly cover prefix")
    return PrefixSchedules(
        prefix_ids=list(prefix),
        message_start_positions=starts,
        history_message_count=history_count,
        p_conceptual_widths=conceptual,
        p_call_widths=p_widths,
        o_call_widths=o_widths,
    )


def scheduled_prefill(model, prefix_ids: Sequence[int], widths: Sequence[int]):
    """Prefill identical contiguous IDs/positions under an explicit call schedule."""
    ids = [int(value) for value in prefix_ids]
    calls = [int(value) for value in widths]
    if (not ids or not calls or any(value < 1 or value > 4096 for value in calls)
            or sum(calls) != len(ids)):
        raise CoherentStateError("scheduled prefill calls do not cover prefix")
    cache = None
    logits = None
    start = 0
    for width in calls:
        end = start + width
        cache, logits = prefill(
            model,
            torch.tensor([ids[start:end]], device=model.device),
            past=cache,
            position_ids=torch.arange(
                start, end, device=model.device)[None],
            cache_position=torch.arange(start, end, device=model.device),
        )
        start = end
    if cache is None or logits is None or start != len(ids):
        raise CoherentStateError("scheduled prefill did not complete")
    return cache, logits


def _schedule_trace(schedules: PrefixSchedules, schedule: str) -> dict:
    return {
        "schedule": schedule,
        "prefix_token_count": len(schedules.prefix_ids),
        "prefix_token_sha256": sha256_ids(schedules.prefix_ids),
        "message_start_positions": list(schedules.message_start_positions),
        "message_start_positions_sha256": sha256_ids(
            schedules.message_start_positions),
        "history_message_count": schedules.history_message_count,
        "p_conceptual_widths": list(schedules.p_conceptual_widths),
        "p_call_widths": list(schedules.p_call_widths),
        "o_call_widths": list(schedules.o_call_widths),
        "executed_call_widths": schedules.widths(schedule),
        "logical_position_start": 0,
        "logical_position_end": len(schedules.prefix_ids),
        "physical_cache_position_start": 0,
        "physical_cache_position_end": len(schedules.prefix_ids),
    }


def capture_generated_summary_scheduled(
        model, tokenizer, source_messages: list[dict], *, schedule: str,
        max_tokens: int = 900) -> tuple[SourceCapture, PrefixSchedules]:
    """Freely generate summary IDs after an explicitly scheduled source prefix."""
    schedules = derive_prefix_schedules(tokenizer, source_messages)
    cache, first = scheduled_prefill(
        model, schedules.prefix_ids, schedules.widths(schedule))
    cache, logits, incremental = generate_greedy_incremental(
        model, cache, first, len(schedules.prefix_ids), max_tokens,
        eos_ids(model), start_cache_position=len(schedules.prefix_ids))
    if not incremental.token_ids or not incremental.ended_on_eos:
        raise CoherentStateError("scheduled summary did not end normally")
    special = {int(value) for value in tokenizer.all_special_ids}
    embedded = [value for value in incremental.token_ids if value in special]
    if embedded:
        raise CoherentStateError(
            f"scheduled summary contains embedded special tokens: {embedded[:8]}")
    rows = extract_summary_rows(
        cache, incremental.start_position, incremental.end_position)
    trace = asdict(incremental)
    trace["prefix_schedule"] = _schedule_trace(schedules, schedule)
    capture = SourceCapture(
        prefix_ids=list(schedules.prefix_ids),
        summary_ids=list(incremental.token_ids),
        summary_text=tokenizer.decode(incremental.token_ids),
        summary_start=incremental.start_position,
        summary_end=incremental.end_position,
        rows=rows,
        trace=trace,
        row_hashes=row_hashes(rows),
        prefix_sha256=sha256_ids(schedules.prefix_ids),
        source_kind=f"generated_incremental__{schedule}",
        cache=cache,
        logits=logits,
    )
    return capture, schedules


def capture_forced_summary_scheduled(
        model, tokenizer, source_messages: list[dict],
        summary_ids: Sequence[int], *, schedule: str,
        source_kind: str) -> tuple[SourceCapture, PrefixSchedules]:
    """Force exact saved summary IDs after an explicitly scheduled source prefix."""
    summary = [int(value) for value in summary_ids]
    if not summary:
        raise CoherentStateError("cannot force an empty scheduled summary")
    schedules = derive_prefix_schedules(tokenizer, source_messages)
    cache, first = scheduled_prefill(
        model, schedules.prefix_ids, schedules.widths(schedule))
    cache, logits, incremental = append_ids_stepwise(
        model, cache, first, summary, len(schedules.prefix_ids),
        start_cache_position=len(schedules.prefix_ids))
    rows = extract_summary_rows(
        cache, incremental.start_position, incremental.end_position)
    trace = asdict(incremental)
    trace["prefix_schedule"] = _schedule_trace(schedules, schedule)
    capture = SourceCapture(
        prefix_ids=list(schedules.prefix_ids),
        summary_ids=summary,
        summary_text=tokenizer.decode(summary),
        summary_start=incremental.start_position,
        summary_end=incremental.end_position,
        rows=rows,
        trace=trace,
        row_hashes=row_hashes(rows),
        prefix_sha256=sha256_ids(schedules.prefix_ids),
        source_kind=f"{source_kind}__{schedule}",
        cache=cache,
        logits=logits,
    )
    return capture, schedules
