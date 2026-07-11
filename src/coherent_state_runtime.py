"""Model-backed capture, arm construction, and downstream scoring.

The production driver orchestrates persistence and resume.  This module owns the
smallest GPU-facing operations so the exact same functions can be exercised by
the 0.6B ladder and the paid bf16 run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
from transformers import DynamicCache

from arms_common import canonical_ids_any, render_hf
from coherent_state_hf import (
    CoherentStateError,
    Snapshot,
    append_ids_stepwise,
    compare_rows,
    delta_deranged_snapshot,
    extract_summary_rows,
    generate_greedy_incremental,
    move_key_rows,
    replace_summary_rows,
    row_hashes,
    sha256_ids,
)
from coherent_state_tokens import (
    generation_prefix_ids,
    probe_layout,
    summary_destination_layout,
    teacher_forcing_feed,
)
from kvlib_hf import prefill, rebuild_cache, snapshot_cache, tf_logprobs


ARM_NAMES = (
    "A_full", "F_fresh", "C_coherent", "W_wrong",
    "V_only", "K_only", "D_delta",
)


@dataclass
class SourceCapture:
    prefix_ids: list[int]
    summary_ids: list[int]
    summary_text: str
    summary_start: int
    summary_end: int
    rows: Snapshot
    trace: dict
    row_hashes: list[dict]
    prefix_sha256: str
    source_kind: str
    cache: object | None = None
    logits: torch.Tensor | None = None


def _ids(model, ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor([list(ids)], device=model.device)


def _positions(model, start: int, n: int) -> torch.Tensor:
    return torch.arange(start, start + n, device=model.device)[None]


def eos_ids(model) -> set[int]:
    value = model.config.eos_token_id
    if isinstance(value, int):
        return {value}
    return {int(v) for v in value}


def capture_generated_summary(model, tokenizer, source_messages: list[dict],
                              *, max_tokens: int = 900) -> SourceCapture:
    prefix = generation_prefix_ids(tokenizer, source_messages)
    cache, first = prefill(model, _ids(model, prefix))
    cache, logits, trace = generate_greedy_incremental(
        model, cache, first, len(prefix), max_tokens, eos_ids(model))
    if not trace.token_ids or not trace.ended_on_eos:
        raise CoherentStateError("generated summary did not end normally")
    special = set(getattr(tokenizer, "all_special_ids", []))
    embedded = [t for t in trace.token_ids if t in special]
    if embedded:
        raise CoherentStateError(
            f"summary contains embedded special tokens: {embedded[:8]}")
    rows = extract_summary_rows(cache, trace.start_position, trace.end_position)
    text = tokenizer.decode(trace.token_ids)
    return SourceCapture(
        prefix_ids=prefix, summary_ids=trace.token_ids, summary_text=text,
        summary_start=trace.start_position, summary_end=trace.end_position,
        rows=rows, trace=asdict(trace), row_hashes=row_hashes(rows),
        prefix_sha256=sha256_ids(prefix), source_kind="generated_incremental",
        cache=cache, logits=logits)


def capture_forced_summary(model, tokenizer, source_messages: list[dict],
                           summary_ids: Sequence[int], *,
                           source_kind: str) -> SourceCapture:
    prefix = generation_prefix_ids(tokenizer, source_messages)
    cache, first = prefill(model, _ids(model, prefix))
    cache, logits, trace = append_ids_stepwise(
        model, cache, first, summary_ids, len(prefix))
    rows = extract_summary_rows(cache, trace.start_position, trace.end_position)
    return SourceCapture(
        prefix_ids=prefix, summary_ids=list(summary_ids),
        summary_text=tokenizer.decode(summary_ids),
        summary_start=trace.start_position, summary_end=trace.end_position,
        rows=rows, trace=asdict(trace), row_hashes=row_hashes(rows),
        prefix_sha256=sha256_ids(prefix), source_kind=source_kind,
        cache=cache, logits=logits)


def validate_generated_replay(generated: SourceCapture, replay: SourceCapture,
                              tolerance: float = 1e-4) -> dict:
    if generated.summary_ids != replay.summary_ids:
        raise CoherentStateError("generated/replay summary IDs differ")
    if generated.prefix_ids != replay.prefix_ids:
        raise CoherentStateError("generated/replay prefixes differ")
    if len(generated.trace["token_logprobs"]) != len(replay.trace["token_logprobs"]):
        raise CoherentStateError("generated/replay logprob lengths differ")
    lp_diff = max(abs(float(a) - float(b)) for a, b in zip(
        generated.trace["token_logprobs"], replay.trace["token_logprobs"]))
    row_diff = compare_rows(generated.rows, replay.rows)
    k_diff = max(x["k_max_abs"] for x in row_diff)
    v_diff = max(x["v_max_abs"] for x in row_diff)
    if max(lp_diff, k_diff, v_diff) > tolerance:
        raise CoherentStateError(
            f"generated/replay identity failed: lp={lp_diff} K={k_diff} V={v_diff}")
    return {"tolerance": tolerance, "token_logprob_max_abs": lp_diff,
            "k_max_abs": k_diff, "v_max_abs": v_diff,
            "per_layer": row_diff}


def complete_assistant_context(model, tokenizer, source_messages: list[dict],
                               capture: SourceCapture) -> tuple[list[dict], list[int], Snapshot]:
    """Append only the canonical assistant-message suffix to a live capture."""
    if capture.cache is None:
        raise CoherentStateError("live source cache was released too early")
    messages = list(source_messages) + [
        {"role": "assistant", "content": capture.summary_text}]
    context = canonical_ids_any(tokenizer, messages, render_hf)
    s0, s1 = capture.summary_start, capture.summary_end
    if context[:s0] != capture.prefix_ids or context[s0:s1] != capture.summary_ids:
        raise CoherentStateError("canonical source does not contain exact captured summary")
    suffix = context[s1:]
    cache = capture.cache
    if suffix:
        cache, _ = prefill(
            model, _ids(model, suffix), past=cache,
            position_ids=_positions(model, s1, len(suffix)))
    return messages, context, snapshot_cache(cache)


def build_fresh_destination(model, tokenizer, conv: dict,
                            summary_text: str, summary_ids: Sequence[int],
                            request: str) -> tuple[object, object, Snapshot, Snapshot]:
    """Return token layout, trace, full fresh snapshot, and bounded fresh rows."""
    layout = summary_destination_layout(
        tokenizer, conv, summary_text, summary_ids, request)
    cache, first = prefill(model, _ids(model, layout.prefix_ids))
    cache, _, trace = append_ids_stepwise(
        model, cache, first, summary_ids, layout.summary_start)
    fresh_rows = extract_summary_rows(cache, layout.summary_start,
                                      layout.summary_end)
    if layout.post_summary_ids:
        cache, _ = prefill(
            model, _ids(model, layout.post_summary_ids), past=cache,
            position_ids=_positions(model, layout.summary_end,
                                    len(layout.post_summary_ids)))
    return layout, trace, snapshot_cache(cache), fresh_rows


def arm_snapshot(arm: str, fresh_snapshot: Snapshot, correct_rows: Snapshot,
                 wrong_rows: Snapshot, destination_start: int,
                 correct_delta: int, wrong_delta: int, rope_theta: float,
                 placebo_seed: int) -> tuple[Snapshot, list[dict]]:
    """Construct one arm at a time; caller must discard it after scoring."""
    if arm not in ARM_NAMES[1:]:
        raise CoherentStateError(f"unsupported compacted arm: {arm}")
    diagnostics: list[dict] = []
    if arm == "F_fresh":
        # Scoring rebuilds/clones the cache before mutation. A shallow list avoids
        # a needless second full destination copy for the identity arm.
        return list(fresh_snapshot), diagnostics
    if arm == "C_coherent":
        moved_correct = move_key_rows(correct_rows, correct_delta, rope_theta)
        return replace_summary_rows(fresh_snapshot, moved_correct,
                                    destination_start, use_keys=True,
                                    use_values=True), diagnostics
    if arm == "W_wrong":
        moved_wrong = move_key_rows(wrong_rows, wrong_delta, rope_theta)
        return replace_summary_rows(fresh_snapshot, moved_wrong,
                                    destination_start, use_keys=True,
                                    use_values=True), diagnostics
    if arm == "V_only":
        return replace_summary_rows(fresh_snapshot, correct_rows,
                                    destination_start, use_keys=False,
                                    use_values=True), diagnostics
    if arm == "K_only":
        moved_correct = move_key_rows(correct_rows, correct_delta, rope_theta)
        return replace_summary_rows(fresh_snapshot, moved_correct,
                                    destination_start, use_keys=True,
                                    use_values=False), diagnostics
    snap, raw_diag = delta_deranged_snapshot(
        fresh_snapshot, correct_rows, destination_start, placebo_seed)
    diagnostics = [asdict(x) for x in raw_diag]
    return snap, diagnostics


def score_target(model, tokenizer, snapshot: Snapshot,
                 context_messages: list[dict], context_ids: Sequence[int],
                 probe: str, target: str, *, consume_snapshot: bool = False) -> dict:
    layout = probe_layout(tokenizer, context_messages, context_ids, probe, target)
    feed = teacher_forcing_feed(layout)
    cache = rebuild_cache(snapshot, DynamicCache, clone=not consume_snapshot)
    if consume_snapshot:
        # Transfer ownership of the full branch to DynamicCache. The caller must
        # build a new branch for another target; this bounds production scoring
        # to one persistent fresh base plus one transient working cache.
        snapshot.clear()
    pos = _positions(model, len(context_ids), len(feed))
    token_lps = tf_logprobs(model, cache, feed, layout.target_ids,
                           position_ids=pos)
    if len(token_lps) != len(layout.target_ids) or not all(
            math.isfinite(float(x)) for x in token_lps):
        raise CoherentStateError("non-finite or incomplete target logprobs")
    return {
        "text": target,
        "token_ids": layout.target_ids,
        "token_logprobs": [float(x) for x in token_lps],
        "mean_logprob": sum(token_lps) / len(token_lps),
        "probe_suffix_ids": layout.suffix_ids,
    }


def score_arm(model, tokenizer, snapshot: Snapshot,
              context_messages: list[dict], context_ids: Sequence[int],
              plants: Sequence[dict], targets: dict[str, dict]) -> dict:
    rows = []
    for plant in plants:
        target = targets[plant["id"]]
        correct = score_target(model, tokenizer, snapshot, context_messages,
                               context_ids, plant["probe"], target["correct"])
        counterfactual = score_target(
            model, tokenizer, snapshot, context_messages, context_ids,
            plant["probe"], target["counterfactual"])
        rows.append({
            "plant_id": plant["id"], "category": plant["category"],
            "probe": plant["probe"], "correct": correct,
            "counterfactual": counterfactual,
            "margin": correct["mean_logprob"] - counterfactual["mean_logprob"],
        })
    if not rows:
        raise CoherentStateError("arm has no scored plants")
    return {"plants": rows,
            "conversation_margin": sum(x["margin"] for x in rows) / len(rows)}
