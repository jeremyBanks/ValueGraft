"""Exact-token layouts for coherent-summary-state sources and outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from arms_common import canonical_ids_any, render_hf
from coherent_state_cases import (
    compacted_messages,
    correct_source_messages,
    validate_native_conversation,
    wrong_source_messages,
)
from coherent_state_hf import CoherentStateError, require_exact_span


@dataclass(frozen=True)
class SummaryDestinationLayout:
    messages: list[dict]
    prefix_ids: list[int]
    context_ids: list[int]
    summary_start: int
    summary_end: int
    post_summary_ids: list[int]


@dataclass(frozen=True)
class ProbeLayout:
    messages: list[dict]
    prefix_ids: list[int]
    suffix_ids: list[int]
    target_ids: list[int]


@dataclass(frozen=True)
class WrongContentReplacement:
    target_message_index: int
    donor_message_index: int
    role: str
    start: int
    end: int
    target_ids: list[int]
    donor_pool_ids: list[int]
    replacement_ids: list[int]
    cycles: int


@dataclass(frozen=True)
class MatchedWrongPrefix:
    correct_ids: list[int]
    wrong_ids: list[int]
    structural_positions: list[int]
    content_positions: list[int]
    replacements: list[WrongContentReplacement]


@dataclass(frozen=True)
class GappedDestinationLayout:
    messages: list[dict]
    prefix_ids: list[int]
    context_ids: list[int]
    physical_summary_start: int
    physical_summary_end: int
    post_summary_ids: list[int]
    system_end: int
    request_logical_start: int
    source_summary_start: int
    prefix_position_ids: list[int]
    summary_position_ids: list[int]
    post_summary_position_ids: list[int]
    context_position_ids: list[int]
    logical_next_position: int


def generation_prefix_ids(tokenizer, messages: list[dict]) -> list[int]:
    if not messages or messages[-1].get("role") != "user":
        raise CoherentStateError("generation prefix must end in a user message")
    canonical = canonical_ids_any(tokenizer, messages, render_hf)
    generated = render_hf(tokenizer, messages, True)
    if generated[:len(canonical)] != canonical or len(generated) <= len(canonical):
        raise CoherentStateError("generation prompt is not a strict canonical extension")
    return list(generated)


def _single_boundary_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise CoherentStateError(f"{text} is not a single tokenizer token")
    return int(ids[0])


def _message_starts(tokenizer, ids: Sequence[int]) -> list[int]:
    marker = _single_boundary_id(tokenizer, "<|im_start|>")
    return [i for i, token_id in enumerate(ids) if int(token_id) == marker]


def _unique_subsequence(container: Sequence[int], needle: Sequence[int],
                        lo: int, hi: int) -> tuple[int, int]:
    n = len(needle)
    matches = [i for i in range(lo, hi - n + 1)
               if list(container[i:i + n]) == list(needle)]
    if len(matches) != 1:
        raise CoherentStateError(
            f"message content is not a unique exact token span: {matches}")
    return matches[0], matches[0] + n


def matched_wrong_prefix_ids(tokenizer, target: dict, donor: dict,
                             request: str) -> MatchedWrongPrefix:
    """Replace only target evicted-content slots with frozen donor token IDs.

    All chat-structural, retained-tail, request, and assistant-header positions
    remain identical to the correct source. Donor content is truncated/cycled to
    each target slot's exact length, so the summary starts at the same position.
    """
    validate_native_conversation(target)
    validate_native_conversation(donor)
    # Reuse the frozen-donor and role-compatibility validation.
    wrong_source_messages(target, donor, request)
    messages = correct_source_messages(target, request)
    correct = generation_prefix_ids(tokenizer, messages)
    starts = _message_starts(tokenizer, correct)
    if len(starts) != len(messages) + 1:
        raise CoherentStateError(
            f"generation prefix has {len(starts)} message starts, "
            f"expected {len(messages) + 1}")

    target_end = target["sections"]["middle_end_msg"]
    donor_end = donor["sections"]["middle_end_msg"]
    donor_by_role: dict[str, list[tuple[int, list[int]]]] = {}
    special = set(int(x) for x in tokenizer.all_special_ids)
    for donor_index in range(1, donor_end):
        msg = donor["messages"][donor_index]
        pool = list(tokenizer.encode(msg["content"], add_special_tokens=False))
        if not pool or any(int(x) in special for x in pool):
            raise CoherentStateError(
                f"donor message {donor_index} has empty/special content IDs")
        donor_by_role.setdefault(msg["role"], []).append((donor_index, pool))

    wrong = list(correct)
    content_positions: set[int] = set()
    replacements = []
    role_ordinals: dict[str, int] = {}
    for target_index in range(1, target_end):
        msg = messages[target_index]
        content = list(tokenizer.encode(
            msg["content"], add_special_tokens=False))
        if not content:
            raise CoherentStateError(
                f"target message {target_index} has empty content IDs")
        start, end = _unique_subsequence(
            correct, content, starts[target_index], starts[target_index + 1])
        pools = donor_by_role.get(msg["role"], [])
        if not pools:
            raise CoherentStateError(f"donor lacks evicted role {msg['role']}")
        ordinal = role_ordinals.get(msg["role"], 0)
        role_ordinals[msg["role"]] = ordinal + 1
        donor_index, pool = pools[ordinal % len(pools)]
        replacement = [int(pool[j % len(pool)]) for j in range(end - start)]
        wrong[start:end] = replacement
        content_positions.update(range(start, end))
        replacements.append(WrongContentReplacement(
            target_message_index=target_index,
            donor_message_index=donor_index,
            role=msg["role"], start=start, end=end,
            target_ids=[int(x) for x in content],
            donor_pool_ids=[int(x) for x in pool],
            replacement_ids=replacement,
            cycles=(len(replacement) + len(pool) - 1) // len(pool),
        ))

    structural = [i for i in range(len(correct)) if i not in content_positions]
    if any(correct[i] != wrong[i] for i in structural):
        raise CoherentStateError("wrong-history construction changed structure")
    if correct == wrong:
        raise CoherentStateError("wrong-history construction changed no token")
    if len(correct) != len(wrong):
        raise CoherentStateError("wrong-history prefix length changed")
    return MatchedWrongPrefix(
        correct_ids=[int(x) for x in correct],
        wrong_ids=[int(x) for x in wrong],
        structural_positions=structural,
        content_positions=sorted(content_positions),
        replacements=replacements,
    )


def summary_destination_layout(tokenizer, conv: dict, summary_text: str,
                               summary_ids: Sequence[int], request: str) \
        -> SummaryDestinationLayout:
    msgs = compacted_messages(conv, summary_text, request)
    source_msgs = msgs[:2]
    prefix = generation_prefix_ids(tokenizer, source_msgs)
    context = canonical_ids_any(tokenizer, msgs, render_hf)
    if context[:len(prefix)] != prefix:
        raise CoherentStateError(
            "compacted assistant rendering changed the fresh generation prefix")
    s0, s1 = require_exact_span(context, list(summary_ids), len(prefix))
    return SummaryDestinationLayout(
        messages=msgs, prefix_ids=prefix, context_ids=context,
        summary_start=s0, summary_end=s1,
        post_summary_ids=context[s1:])


def gapped_destination_layout(tokenizer, conv: dict, summary_text: str,
                              summary_ids: Sequence[int], request: str,
                              correct_prefix_ids: Sequence[int]) \
        -> GappedDestinationLayout:
    """Give compact physical storage the correct source's logical positions."""
    packed = summary_destination_layout(
        tokenizer, conv, summary_text, summary_ids, request)
    starts = _message_starts(tokenizer, packed.prefix_ids)
    if len(starts) != 3:
        raise CoherentStateError(
            f"fresh generation prefix has {len(starts)} message starts, expected 3")
    system_end = starts[1]
    suffix = packed.prefix_ids[system_end:]
    correct = list(correct_prefix_ids)
    source_start = len(correct)
    request_start = source_start - len(suffix)
    if request_start < system_end:
        raise CoherentStateError("gapped logical position islands overlap")
    if correct[request_start:] != suffix:
        raise CoherentStateError(
            "fresh request/header is not the exact correct-prefix suffix")
    prefix_positions = list(range(system_end)) + list(
        range(request_start, source_start))
    summary_positions = list(range(
        source_start, source_start + len(summary_ids)))
    post_positions = list(range(
        source_start + len(summary_ids),
        source_start + len(summary_ids) + len(packed.post_summary_ids)))
    context_positions = prefix_positions + summary_positions + post_positions
    if len(context_positions) != len(packed.context_ids):
        raise CoherentStateError("gapped context position coverage mismatch")
    return GappedDestinationLayout(
        messages=packed.messages,
        prefix_ids=packed.prefix_ids,
        context_ids=packed.context_ids,
        physical_summary_start=packed.summary_start,
        physical_summary_end=packed.summary_end,
        post_summary_ids=packed.post_summary_ids,
        system_end=system_end,
        request_logical_start=request_start,
        source_summary_start=source_start,
        prefix_position_ids=prefix_positions,
        summary_position_ids=summary_positions,
        post_summary_position_ids=post_positions,
        context_position_ids=context_positions,
        logical_next_position=(post_positions[-1] + 1 if post_positions
                               else summary_positions[-1] + 1),
    )


def _im_end_id(tokenizer) -> int:
    return _single_boundary_id(tokenizer, "<|im_end|>")


def rendered_assistant_content_ids(tokenizer, preceding_messages: list[dict],
                                   text: str) -> list[int]:
    """Extract the exact assistant-content IDs following a generation prompt.

    This avoids standalone tokenization, whose first/last BPE token can differ
    at a chat-template boundary.  Any template-injected prefix (for example a
    hidden/empty reasoning wrapper) is rejected because it would make the
    nominal target differ from the scored target.
    """
    if not text:
        raise CoherentStateError("assistant target text is empty")
    prefix = generation_prefix_ids(tokenizer, preceding_messages)
    full = canonical_ids_any(
        tokenizer,
        preceding_messages + [{"role": "assistant", "content": text}],
        render_hf)
    if full[:len(prefix)] != prefix:
        raise CoherentStateError("assistant rendering changed generation prefix")
    im_end = _im_end_id(tokenizer)
    try:
        end = full.index(im_end, len(prefix))
    except ValueError as exc:
        raise CoherentStateError("assistant message lacks im_end boundary") from exc
    content = full[len(prefix):end]
    # The exact chat-template tokenization must decode to the authored target,
    # modulo tokenizer-normalized edge whitespace.  Extra injected blocks fail.
    if tokenizer.decode(content).strip() != text.strip():
        raise CoherentStateError(
            "rendered assistant content contains template-injected or altered text")
    if not content:
        raise CoherentStateError("assistant target token sequence is empty")
    return list(content)


def probe_layout(tokenizer, context_messages: list[dict],
                 context_ids: Sequence[int], probe: str, target: str) -> ProbeLayout:
    if not probe:
        raise CoherentStateError("probe text is empty")
    msgs = list(context_messages) + [{"role": "user", "content": probe}]
    prefix = generation_prefix_ids(tokenizer, msgs)
    cids = list(context_ids)
    if prefix[:len(cids)] != cids or len(prefix) <= len(cids):
        raise CoherentStateError("probe rendering is not a strict context extension")
    target_ids = rendered_assistant_content_ids(tokenizer, msgs, target)
    return ProbeLayout(
        messages=msgs, prefix_ids=prefix,
        suffix_ids=prefix[len(cids):], target_ids=target_ids)


def teacher_forcing_feed(layout: ProbeLayout) -> list[int]:
    """Tokens fed after a cached context to score every target token."""
    if not layout.suffix_ids or not layout.target_ids:
        raise CoherentStateError("teacher-forcing layout is incomplete")
    return layout.suffix_ids + layout.target_ids[:-1]
