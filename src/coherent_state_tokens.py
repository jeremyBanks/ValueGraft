"""Exact-token layouts for coherent-summary-state sources and outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from arms_common import canonical_ids_any, render_hf
from coherent_state_cases import compacted_messages
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


def generation_prefix_ids(tokenizer, messages: list[dict]) -> list[int]:
    if not messages or messages[-1].get("role") != "user":
        raise CoherentStateError("generation prefix must end in a user message")
    canonical = canonical_ids_any(tokenizer, messages, render_hf)
    generated = render_hf(tokenizer, messages, True)
    if generated[:len(canonical)] != canonical or len(generated) <= len(canonical):
        raise CoherentStateError("generation prompt is not a strict canonical extension")
    return list(generated)


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


def _im_end_id(tokenizer) -> int:
    ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(ids) != 1:
        raise CoherentStateError("<|im_end|> is not a single tokenizer token")
    return int(ids[0])


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
