"""Model-backed capture, arm construction, and downstream scoring.

The production driver orchestrates persistence and resume.  This module owns the
smallest GPU-facing operations so the exact same functions can be exercised by
the 0.6B ladder and the paid bf16 run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
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
    GappedDestinationLayout,
    gapped_destination_layout,
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
GAPPED_ARM_NAMES = (
    "A_full", "G_fresh", "G_correct", "G_wrong",
    "G_Vcorrect", "G_Kcorrect",
)
AMENDMENT_ID = "COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4-5-6-7"
DESIGN_ID = "coherent-state-gapped-v7"


def eager_backend_fingerprint(model, *, progress=None) -> dict:
    """Return and validate the resolved eager backend for every decoder layer.

    A requested load argument is insufficient provenance.  Decoder attention
    modules are enumerated in layer order and both model-level resolution fields
    and any module-local fields are retained in the fingerprint.
    """
    cfg = getattr(model.config, "text_config", model.config)

    def config_record(label, config):
        fields = {
            "_attn_implementation": getattr(
                config, "_attn_implementation", None),
            "_attn_implementation_internal": getattr(
                config, "_attn_implementation_internal", None),
        }
        fields = {key: None if value is None else str(value)
                  for key, value in fields.items()}
        if set(fields.values()) != {"eager"}:
            raise CoherentStateError(
                f"{label} attention backend fields are missing, ambiguous, "
                f"or non-eager: {fields}")
        return {
            "scope": label,
            "config_class": type(config).__name__,
            **fields,
            "resolved_implementation": "eager",
        }

    # Stable schema labels, not Python expression spellings.  The independent
    # validators require these exact names and verify the actual config classes
    # and resolved fields separately.
    model_record = config_record("model_config", model.config)
    text_record = config_record("text_config", cfg)
    records = []
    partial = {
        "requested_implementation": "eager",
        "model_config": model_record,
        "text_config": text_record,
        "text_config_is_model_config": cfg is model.config,
        "expected_layer_count": int(getattr(cfg, "num_hidden_layers", -1)),
        "layers": records,
    }
    if progress is not None:
        progress(json.loads(json.dumps(partial)))
    for name, module in model.named_modules():
        cls = type(module).__name__
        if not (hasattr(module, "q_proj") and hasattr(module, "k_proj") and
                "Attention" in cls):
            continue
        module_config = getattr(module, "config", None)
        if module_config is None:
            raise CoherentStateError(
                f"attention module {name} has no module.config")
        local = config_record(f"module:{name}.config", module_config)
        layer_index = getattr(module, "layer_idx", None)
        if not isinstance(layer_index, int):
            raise CoherentStateError(
                f"attention module {name} has invalid layer_idx={layer_index!r}")
        record = {
            "layer_index": int(getattr(module, "layer_idx", len(records))),
            "module_name": name,
            "module_class": cls,
            "module_config_class": type(module_config).__name__,
            "module_config__attn_implementation":
                local["_attn_implementation"],
            "module_config__attn_implementation_internal":
                local["_attn_implementation_internal"],
            "resolved_implementation": local["resolved_implementation"],
        }
        records.append(record)
        if progress is not None:
            progress(json.loads(json.dumps(partial)))
    expected = int(getattr(cfg, "num_hidden_layers", -1))
    if expected < 1 or len(records) != expected:
        raise CoherentStateError(
            f"attention layer enumeration mismatch: {len(records)} != {expected}")
    if [x["layer_index"] for x in records] != list(range(expected)):
        raise CoherentStateError("attention layer indices are not complete and ordered")
    resolved = {x["resolved_implementation"] for x in records}
    if resolved != {"eager"}:
        raise CoherentStateError(
            f"subject attention backend is not uniformly eager: {sorted(map(str, resolved))}")
    payload = partial
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


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


def _cache_positions(model, start: int, n: int) -> torch.Tensor:
    return torch.arange(start, start + n, device=model.device)


def validate_position_schedule(logical_positions: Sequence[int],
                               cache_positions: Sequence[int], *,
                               physical_start: int) -> dict:
    """Fail closed if logical RoPE positions leak into physical cache indices."""
    logical = [int(x) for x in logical_positions]
    physical = [int(x) for x in cache_positions]
    if not logical or len(logical) != len(physical):
        raise CoherentStateError("position schedule coverage mismatch")
    expected = list(range(physical_start, physical_start + len(physical)))
    if physical != expected:
        raise CoherentStateError(
            f"cache_position is not contiguous physical storage: {physical} != {expected}")
    if any(b <= a for a, b in zip(logical, logical[1:])):
        raise CoherentStateError("logical position_ids are not strictly increasing")
    return {
        "physical_start": physical_start,
        "physical_end": physical_start + len(physical),
        "logical_start": logical[0],
        "logical_end": logical[-1] + 1,
        "gap_from_physical": logical[0] - physical_start,
    }


def snapshot_physical_length(snapshot: Snapshot) -> int:
    """Return a cache's physical row count, rejecting malformed snapshots."""
    if not snapshot:
        raise CoherentStateError("cache snapshot has no layers")
    lengths = []
    for layer, (keys, values) in enumerate(snapshot):
        if keys.ndim < 3 or values.ndim < 3:
            raise CoherentStateError(f"cache layer {layer} has invalid rank")
        if keys.shape[-2] != values.shape[-2]:
            raise CoherentStateError(
                f"cache layer {layer} K/V physical lengths differ")
        lengths.append(int(keys.shape[-2]))
    if len(set(lengths)) != 1:
        raise CoherentStateError(
            f"cache layers have inconsistent physical lengths: {lengths}")
    return lengths[0]


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


def capture_forced_prefix_ids(model, tokenizer, prefix_ids: Sequence[int],
                              summary_ids: Sequence[int], *,
                              source_kind: str,
                              prefix_position_ids: Sequence[int] | None = None,
                              summary_start: int | None = None) -> SourceCapture:
    """Force a summary after an already frozen exact prefix token stream."""
    prefix = [int(x) for x in prefix_ids]
    if not prefix:
        raise CoherentStateError("forced exact prefix is empty")
    logical_prefix = (list(range(len(prefix))) if prefix_position_ids is None
                      else [int(x) for x in prefix_position_ids])
    if len(logical_prefix) != len(prefix):
        raise CoherentStateError("forced prefix logical-position length mismatch")
    start = len(prefix) if summary_start is None else int(summary_start)
    cache, first = prefill(
        model, _ids(model, prefix),
        position_ids=torch.tensor([logical_prefix], device=model.device),
        cache_position=_cache_positions(model, 0, len(prefix)))
    cache, logits, trace = append_ids_stepwise(
        model, cache, first, summary_ids, start,
        start_cache_position=len(prefix))
    physical_start = len(prefix)
    rows = extract_summary_rows(
        cache, physical_start, physical_start + len(summary_ids))
    return SourceCapture(
        prefix_ids=prefix, summary_ids=list(summary_ids),
        summary_text=tokenizer.decode(summary_ids),
        summary_start=start, summary_end=start + len(summary_ids),
        rows=rows, trace=asdict(trace), row_hashes=row_hashes(rows),
        prefix_sha256=sha256_ids(prefix), source_kind=source_kind,
        cache=cache, logits=logits)


def validate_generated_replay(generated: SourceCapture, replay: SourceCapture,
                              tolerance: float = 1e-4) -> dict:
    result = measure_generated_replay(generated, replay, tolerance)
    if not result["passes"]:
        raise CoherentStateError(
            "generated/replay identity failed: "
            f"lp={result['token_logprob_max_abs']} K={result['k_max_abs']} "
            f"V={result['v_max_abs']}")
    return result


def measure_generated_replay(generated: SourceCapture, replay: SourceCapture,
                             tolerance: float = 1e-4) -> dict:
    """Return complete replay measurements before applying the verdict.

    The authorization lifecycle persists this payload and only then validates
    ``passes``.  ``validate_generated_replay`` remains the fail-closed API for
    ordinary callers.
    """
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
    return {"tolerance": tolerance, "token_logprob_max_abs": lp_diff,
            "k_max_abs": k_diff, "v_max_abs": v_diff,
            "per_layer": row_diff,
            "observed_aggregate": max(lp_diff, k_diff, v_diff),
            "comparison": "<=",
            "passes": max(lp_diff, k_diff, v_diff) <= tolerance}


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


def build_gapped_fresh_boundary(
        model, tokenizer, conv: dict, summary_text: str,
        summary_ids: Sequence[int], request: str,
        correct_prefix_ids: Sequence[int]) \
        -> tuple[GappedDestinationLayout, object, Snapshot, Snapshot]:
    """Build compact physical storage through the gapped summary boundary."""
    layout = gapped_destination_layout(
        tokenizer, conv, summary_text, summary_ids, request,
        correct_prefix_ids)
    validate_position_schedule(
        layout.prefix_position_ids, range(len(layout.prefix_ids)),
        physical_start=0)
    validate_position_schedule(
        layout.summary_position_ids,
        range(layout.physical_summary_start, layout.physical_summary_end),
        physical_start=layout.physical_summary_start)
    system_ids = layout.prefix_ids[:layout.system_end]
    request_ids = layout.prefix_ids[layout.system_end:]
    cache, first = prefill(
        model, _ids(model, system_ids),
        position_ids=torch.tensor(
            [layout.prefix_position_ids[:layout.system_end]],
            device=model.device),
        cache_position=_cache_positions(model, 0, len(system_ids)))
    if request_ids:
        cache, first = prefill(
            model, _ids(model, request_ids), past=cache,
            position_ids=torch.tensor(
                [layout.prefix_position_ids[layout.system_end:]],
                device=model.device),
            cache_position=_cache_positions(
                model, len(system_ids), len(request_ids)))
    cache, _, trace = append_ids_stepwise(
        model, cache, first, summary_ids, layout.source_summary_start,
        start_cache_position=len(layout.prefix_ids))
    fresh_rows = extract_summary_rows(
        cache, layout.physical_summary_start, layout.physical_summary_end)
    boundary = snapshot_cache(cache)
    if snapshot_physical_length(boundary) != layout.physical_summary_end:
        raise CoherentStateError(
            "gapped boundary physical cache contains phantom or missing rows")
    if (trace.start_position != layout.source_summary_start or
            trace.end_position != layout.source_summary_start + len(summary_ids)):
        raise CoherentStateError("gapped fresh trace logical summary span differs")
    return layout, trace, boundary, fresh_rows


def append_gapped_post_summary(model, boundary_snapshot: Snapshot,
                               layout: GappedDestinationLayout) -> Snapshot:
    """Fork one arm at the boundary and causally append close + retained tail."""
    boundary_length = snapshot_physical_length(boundary_snapshot)
    if boundary_length != layout.physical_summary_end:
        raise CoherentStateError(
            "gapped arm must end exactly at the summary boundary; "
            f"got {boundary_length}, expected {layout.physical_summary_end}")
    cache = rebuild_cache(boundary_snapshot, DynamicCache)
    if layout.post_summary_ids:
        validate_position_schedule(
            layout.post_summary_position_ids,
            range(layout.physical_summary_end,
                  layout.physical_summary_end + len(layout.post_summary_ids)),
            physical_start=layout.physical_summary_end)
        cache, _ = prefill(
            model, _ids(model, layout.post_summary_ids), past=cache,
            position_ids=torch.tensor(
                [layout.post_summary_position_ids], device=model.device),
            cache_position=_cache_positions(
                model, layout.physical_summary_end,
                len(layout.post_summary_ids)))
    completed = snapshot_cache(cache)
    if snapshot_physical_length(completed) != len(layout.context_ids):
        raise CoherentStateError(
            "gapped post-summary append produced wrong physical cache length")
    return completed


def gapped_arm_boundary(arm: str, fresh_boundary: Snapshot,
                        correct_rows: Snapshot, wrong_rows: Snapshot,
                        destination_start: int,
                        placebo_seed: int) -> tuple[Snapshot, list[dict]]:
    """Apply only the amended same-position summary-row intervention."""
    if arm == "G_delta":
        raise CoherentStateError("G_delta is retired by Amendment 4")
    if arm not in GAPPED_ARM_NAMES[1:]:
        raise CoherentStateError(f"unsupported gapped arm: {arm}")
    if not correct_rows or not wrong_rows:
        raise CoherentStateError("gapped source rows are empty")
    summary_rows = int(correct_rows[0][0].shape[-2])
    if any(int(k.shape[-2]) != summary_rows or
           int(v.shape[-2]) != summary_rows
           for source in (correct_rows, wrong_rows) for k, v in source):
        raise CoherentStateError("gapped source summary row counts differ")
    expected_boundary = destination_start + summary_rows
    if snapshot_physical_length(fresh_boundary) != expected_boundary:
        raise CoherentStateError(
            "gapped fresh input is not an immediate summary boundary")
    if arm == "G_fresh":
        return list(fresh_boundary), []
    if arm == "G_correct":
        return replace_summary_rows(
            fresh_boundary, correct_rows, destination_start,
            use_keys=True, use_values=True), []
    if arm == "G_wrong":
        return replace_summary_rows(
            fresh_boundary, wrong_rows, destination_start,
            use_keys=True, use_values=True), []
    if arm == "G_Vcorrect":
        return replace_summary_rows(
            fresh_boundary, correct_rows, destination_start,
            use_keys=False, use_values=True), []
    if arm == "G_Kcorrect":
        return replace_summary_rows(
            fresh_boundary, correct_rows, destination_start,
            use_keys=True, use_values=False), []
    raise CoherentStateError(f"unimplemented gapped arm: {arm}")


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
                 probe: str, target: str, *, consume_snapshot: bool = False,
                 logical_context_end: int | None = None) -> dict:
    layout = probe_layout(tokenizer, context_messages, context_ids, probe, target)
    feed = teacher_forcing_feed(layout)
    physical_context_end = snapshot_physical_length(snapshot)
    if physical_context_end != len(context_ids):
        raise CoherentStateError(
            "scoring snapshot physical length differs from visible context IDs")
    cache = rebuild_cache(snapshot, DynamicCache, clone=not consume_snapshot)
    if consume_snapshot:
        # Transfer ownership of the full branch to DynamicCache. The caller must
        # build a new branch for another target; this bounds production scoring
        # to one persistent fresh base plus one transient working cache.
        snapshot.clear()
    logical_start = (len(context_ids) if logical_context_end is None
                     else int(logical_context_end))
    pos = _positions(model, logical_start, len(feed))
    cache_pos = _cache_positions(model, physical_context_end, len(feed))
    validate_position_schedule(
        range(logical_start, logical_start + len(feed)),
        range(physical_context_end, physical_context_end + len(feed)),
        physical_start=physical_context_end)
    token_lps = tf_logprobs(model, cache, feed, layout.target_ids,
                           position_ids=pos, cache_position=cache_pos)
    if len(token_lps) != len(layout.target_ids) or not all(
            math.isfinite(float(x)) for x in token_lps):
        raise CoherentStateError("non-finite or incomplete target logprobs")
    return {
        "text": target,
        "token_ids": layout.target_ids,
        "token_logprobs": [float(x) for x in token_lps],
        "mean_logprob": sum(token_lps) / len(token_lps),
        "probe_suffix_ids": layout.suffix_ids,
        "logical_position_ids": list(range(logical_start,
                                           logical_start + len(feed))),
        "physical_cache_positions": list(range(
            physical_context_end, physical_context_end + len(feed))),
    }


def score_arm(model, tokenizer, snapshot: Snapshot,
              context_messages: list[dict], context_ids: Sequence[int],
              plants: Sequence[dict], targets: dict[str, dict], *,
              logical_context_end: int | None = None) -> dict:
    rows = []
    for plant in plants:
        target = targets[plant["id"]]
        correct = score_target(model, tokenizer, snapshot, context_messages,
                               context_ids, plant["probe"], target["correct"],
                               logical_context_end=logical_context_end)
        counterfactual = score_target(
            model, tokenizer, snapshot, context_messages, context_ids,
            plant["probe"], target["counterfactual"],
            logical_context_end=logical_context_end)
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
