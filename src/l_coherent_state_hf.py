"""Qwen3-0.6B production-path ladder for coherent-summary-state apparatus."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from coherent_state_cases import (
    correct_source_messages,
    fresh_source_messages,
)
from coherent_state_calibration import validate_calibration_constructions
from coherent_state_hf import (
    compare_rows,
    move_key_rows,
    replace_summary_rows,
    row_hashes,
    sha256_ids,
)
from coherent_state_runtime import (
    AMENDMENT_ID,
    DESIGN_ID,
    GAPPED_ARM_NAMES,
    append_gapped_post_summary,
    build_gapped_fresh_boundary,
    capture_forced_prefix_ids,
    capture_forced_summary,
    capture_generated_summary,
    complete_assistant_context,
    eager_backend_fingerprint,
    gapped_arm_boundary,
    validate_generated_replay,
    validate_position_schedule,
)
from coherent_state_store import checkpoint_path, read_checkpoint, save_render
from coherent_state_tokens import (
    matched_wrong_prefix_ids,
    rendered_assistant_content_ids,
)
from kvlib_hf import prefill, rebuild_cache, snapshot_cache


MODEL = "Qwen/Qwen3-0.6B"
REQUEST = "Write a short context summary. Output only the summary."
SUMMARY = "The approved label remains on file."
_LAST_LADDER_DIAGNOSTICS: dict = {}


def fake_conv(cid: str, label: str, marker: str) -> dict:
    return {
        "id": cid,
        "messages": [
            {"role": "system", "content": "Maintain the user's test record."},
            {"role": "user", "content":
             f"The approved label is {label}. Remember that exact label."},
            {"role": "assistant", "content":
             f"Understood; the approved label is {label}."},
            {"role": "user", "content": f"Unrelated tail marker: {marker}."},
            {"role": "assistant", "content": "Tail marker noted."},
        ],
        "sections": {"middle_end_msg": 3},
    }


def run_loaded_kernel_gates(model, tokenizer, *, identity_tolerance: float,
                            rotation_tolerance: float,
                            placebo_quantization_tolerance: float = 0.05,
                            placebo_moment_tolerance: float = 0.02) -> dict:
    """Compatibility name for the additive-amendment gapped authorization gate.

    rotation_tolerance is intentionally ignored: packed key movement is a
    retired diagnostic and can no longer authorize a semantic run through this
    legacy entry point.
    """
    del rotation_tolerance
    return run_loaded_gapped_gates(
        model, tokenizer,
        identity_tolerance=identity_tolerance,
        placebo_quantization_tolerance=placebo_quantization_tolerance,
        placebo_moment_tolerance=placebo_moment_tolerance)


def _snapshot_length(snapshot) -> int:
    lengths = {int(t.shape[-2]) for pair in snapshot for t in pair}
    if len(lengths) != 1:
        raise RuntimeError(f"cache tensors have inconsistent lengths: {lengths}")
    return lengths.pop()


def _require_summary_boundary(snapshot, layout) -> None:
    """Reject an arm that already contains the assistant close or tail."""
    observed = _snapshot_length(snapshot)
    if observed != layout.physical_summary_end:
        raise RuntimeError(
            f"arm must fork exactly at summary boundary: cache={observed} "
            f"expected={layout.physical_summary_end}")


def _verify_intervention(fresh, treated, source, start: int, *,
                         use_keys: bool, use_values: bool) -> None:
    """Require exact selected insertion and exact preservation elsewhere."""
    if not (len(fresh) == len(treated) == len(source)):
        raise RuntimeError("intervention layer coverage differs")
    for li, ((kf, vf), (kt, vt), (ks, vs)) in enumerate(
            zip(fresh, treated, source)):
        n = int(ks.shape[-2])
        end = start + n
        if use_keys and not torch.equal(
                kt[..., start:end, :], ks.to(kt.device, dtype=kt.dtype)):
            raise RuntimeError(f"layer {li} key insertion is not bit exact")
        if use_values and not torch.equal(
                vt[..., start:end, :], vs.to(vt.device, dtype=vt.dtype)):
            raise RuntimeError(f"layer {li} value insertion is not bit exact")
        if not use_keys and not torch.equal(kt, kf):
            raise RuntimeError(f"layer {li} unselected keys changed")
        if not use_values and not torch.equal(vt, vf):
            raise RuntimeError(f"layer {li} unselected values changed")
        for before, after, selected in (
                (kf, kt, use_keys), (vf, vt, use_values)):
            if selected and (not torch.equal(before[..., :start, :],
                                             after[..., :start, :]) or
                             not torch.equal(before[..., end:, :],
                                             after[..., end:, :])):
                raise RuntimeError(
                    f"layer {li} non-summary rows changed during insertion")


def _snapshot_max_abs(a, b) -> tuple[float, float]:
    rows = compare_rows(a, b)
    return (max(x["k_max_abs"] for x in rows),
            max(x["v_max_abs"] for x in rows))


FROZEN_FIXTURE_LITERAL = "alpha beta gamma delta epsilon"
FROZEN_FIXTURE_POOL = [7141, 13440, 21619, 9477, 31204]
FROZEN_MARGIN_IDS = [362, 425]
FROZEN_CONTINUATION_ID = 7141
FROZEN_SCHEDULES = (
    (5, (5,), (2, 3)),
    (64, (64,), (32, 32)),
    (900, (900,), (32, 868)),
    (4096, (4096,), (32, 4064)),
    (4097, (4096, 1), (32, 4065)),
    (8193, (4096, 4096, 1), (32, 4096, 4065)),
)


def _partitioned_forward(model, token_ids, logical_positions, partitions):
    if sum(partitions) != len(token_ids):
        raise RuntimeError("fixture partition does not cover token stream")
    cache = None
    logits = None
    lo = 0
    for width in partitions:
        hi = lo + width
        ids = torch.tensor([token_ids[lo:hi]], device=model.device)
        pos = torch.tensor([logical_positions[lo:hi]], device=model.device)
        physical = torch.arange(lo, hi, device=model.device)
        cache, logits = prefill(
            model, ids, past=cache, position_ids=pos,
            cache_position=physical)
        lo = hi
    return cache, logits


def _margin(logits) -> float:
    lp = torch.log_softmax(logits.float(), dim=-1)
    return float(lp[0, FROZEN_MARGIN_IDS[0]] - lp[0, FROZEN_MARGIN_IDS[1]])


def _compare_schedules(model, token_ids, logical_positions,
                       reference_partition, alternative_partition,
                       tolerance) -> dict:
    reference, reference_logits = _partitioned_forward(
        model, token_ids, logical_positions, reference_partition)
    alternative, alternative_logits = _partitioned_forward(
        model, token_ids, logical_positions, alternative_partition)
    rows = compare_rows(snapshot_cache(reference), snapshot_cache(alternative))
    last_logits = float(
        (reference_logits.float() - alternative_logits.float()).abs().max())
    margin_shift = abs(_margin(reference_logits) - _margin(alternative_logits))
    n = len(token_ids)
    continuation_ids = torch.tensor([[FROZEN_CONTINUATION_ID]], device=model.device)
    continuation_pos = torch.tensor([[logical_positions[-1] + 1]], device=model.device)
    continuation_cache_pos = torch.tensor([n], device=model.device)
    with torch.no_grad():
        ref_next = model(
            input_ids=continuation_ids, past_key_values=reference,
            position_ids=continuation_pos, cache_position=continuation_cache_pos,
            use_cache=True, logits_to_keep=0)
        alt_next = model(
            input_ids=continuation_ids, past_key_values=alternative,
            position_ids=continuation_pos, cache_position=continuation_cache_pos,
            use_cache=True, logits_to_keep=0)
    continuation_logits = float(
        (ref_next.logits.float() - alt_next.logits.float()).abs().max())
    continuation_rows = []
    for li, ((rk, rv), (ak, av)) in enumerate(zip(
            snapshot_cache(ref_next.past_key_values),
            snapshot_cache(alt_next.past_key_values))):
        continuation_rows.append({
            "layer": li,
            "k_max_abs": float(
                (rk[..., -1:, :].float() - ak[..., -1:, :].float()).abs().max()),
            "v_max_abs": float(
                (rv[..., -1:, :].float() - av[..., -1:, :].float()).abs().max()),
        })
    maxima = {
        "cache_k_max_abs": max(x["k_max_abs"] for x in rows),
        "cache_v_max_abs": max(x["v_max_abs"] for x in rows),
        "last_logits_max_abs": last_logits,
        "selected_margin_abs_shift": margin_shift,
        "continuation_logits_max_abs": continuation_logits,
        "continuation_k_max_abs": max(x["k_max_abs"] for x in continuation_rows),
        "continuation_v_max_abs": max(x["v_max_abs"] for x in continuation_rows),
    }
    return {
        "reference_partition": list(reference_partition),
        "alternative_partition": list(alternative_partition),
        "per_layer": rows,
        "continuation_per_layer": continuation_rows,
        **maxima,
        "tolerance": tolerance,
        "passes": max(maxima.values()) <= tolerance,
    }


def run_frozen_schedule_fixtures(model, tokenizer, tolerance=5e-4,
                                 progress=None) -> dict:
    """Execute every Amendment-4 schedule row, retaining all safe evidence."""
    observed_pool = tokenizer(
        FROZEN_FIXTURE_LITERAL, add_special_tokens=False).input_ids
    if observed_pool != FROZEN_FIXTURE_POOL:
        raise RuntimeError(
            f"frozen fixture tokenizer mismatch: {observed_pool} != {FROZEN_FIXTURE_POOL}")
    special = {int(x) for x in tokenizer.all_special_ids}
    provenance = {
        "literal": FROZEN_FIXTURE_LITERAL,
        "pool_token_ids": list(observed_pool),
        "pool_decoded_text": tokenizer.decode(observed_pool),
        "pool_contains_special_token": any(x in special for x in observed_pool),
        "pool_sha256": sha256_ids(observed_pool),
        "margin_token_ids": list(FROZEN_MARGIN_IDS),
        "margin_decoded_text": [tokenizer.decode([x]) for x in FROZEN_MARGIN_IDS],
        "continuation_token_id": FROZEN_CONTINUATION_ID,
        "continuation_decoded_text": tokenizer.decode([FROZEN_CONTINUATION_ID]),
    }
    if provenance["pool_contains_special_token"]:
        raise RuntimeError("frozen fixture pool contains a special token")
    contiguous = []
    failures = []
    document = {
        "fixture_provenance": provenance,
        "contiguous": contiguous,
        "logical_gap": {"status": "PENDING", "passes": False},
        "tolerance": tolerance,
        "failures": failures,
        "passes": False,
    }
    if progress is not None:
        progress(json.loads(json.dumps(document)))
    for length, reference, alternative in FROZEN_SCHEDULES:
        token_ids = [observed_pool[i % len(observed_pool)] for i in range(length)]
        row = {
            "length": length,
            "token_ids_sha256": sha256_ids(token_ids),
            "status": "RUNNING",
        }
        try:
            row.update(_compare_schedules(
                model, token_ids, list(range(length)), reference, alternative,
                tolerance))
            row["status"] = "PASS" if row["passes"] else "FAIL"
            if not row["passes"]:
                failures.append(f"contiguous-L{length}")
        except Exception as exc:
            row.update({"status": "ERROR", "passes": False,
                        "error_type": type(exc).__name__, "error": str(exc)})
            failures.append(f"contiguous-L{length}")
        contiguous.append(row)
        if progress is not None:
            progress(json.loads(json.dumps(document)))

    gap_ids = [observed_pool[i % len(observed_pool)] for i in range(64)]
    gap_positions = list(range(32)) + list(range(8192, 8224))
    gap = {
        "length": 64,
        "token_ids_sha256": sha256_ids(gap_ids),
        "logical_positions_sha256": sha256_ids(gap_positions),
        "logical_positions": gap_positions,
        "physical_cache_positions": list(range(64)),
        "full_attention_over_physically_prior_rows": True,
        "status": "RUNNING",
    }
    try:
        gap.update(_compare_schedules(
            model, gap_ids, gap_positions, (32, 32), (32,) + (1,) * 32,
            tolerance))
        try:
            validate_position_schedule(
                gap_positions[32:], gap_positions[32:], physical_start=32)
        except Exception as exc:
            gap["logical_as_cache_position_rejected"] = True
            gap["logical_as_cache_position_error"] = str(exc)
        else:
            gap["logical_as_cache_position_rejected"] = False
        gap["passes"] = bool(
            gap["passes"] and gap["logical_as_cache_position_rejected"])
        gap["status"] = "PASS" if gap["passes"] else "FAIL"
        if not gap["passes"]:
            failures.append("logical-gap")
    except Exception as exc:
        gap.update({"status": "ERROR", "passes": False,
                    "error_type": type(exc).__name__, "error": str(exc)})
        failures.append("logical-gap")
    document["logical_gap"] = gap
    document["passes"] = not failures
    if progress is not None:
        progress(json.loads(json.dumps(document)))
    return document


def _validate_exact_length_wrong(correct_ids, wrong_ids,
                                 structural_positions, content_positions,
                                 special_ids) -> None:
    if len(correct_ids) != len(wrong_ids):
        raise RuntimeError("wrong-history prefix length changed")
    if any(correct_ids[i] != wrong_ids[i] for i in structural_positions):
        raise RuntimeError("wrong-history construction altered structure")
    if correct_ids == wrong_ids or not content_positions:
        raise RuntimeError("wrong-history construction changed no content")
    specials = {int(x) for x in special_ids}
    if any(int(wrong_ids[i]) in specials for i in content_positions):
        raise RuntimeError("wrong-history content introduced a special token")


def run_loaded_gapped_gates(
        model, tokenizer, *, identity_tolerance: float,
        zero_gap_tolerance: float = 5e-4,
        placebo_quantization_tolerance: float = 0.05,
        placebo_moment_tolerance: float = 0.02,
        diagnostic_sink: dict | None = None) -> dict:
    """Amendments-1-2-3-4 gate; all safe measurements accumulate before fail."""
    del placebo_quantization_tolerance, placebo_moment_tolerance
    sink = diagnostic_sink if diagnostic_sink is not None else {}
    sink.clear()
    sink.update({
        "schema": 2,
        "amendment_id": AMENDMENT_ID,
        "design_id": DESIGN_ID,
        "passes": False,
        "authorization_path": "gapped_position_preserving_only",
        "position_policy": "logical_position_ids_physical_cache_position",
        "attention_backend": {"status": "RUNNING", "passes": False},
        "frozen_schedule_fixtures": {"status": "RUNNING", "passes": False},
    })

    cfg = getattr(model.config, "text_config", model.config)
    theta = float((getattr(cfg, "rope_parameters", None) or {})["rope_theta"])
    device = model.device
    seq = tokenizer("alpha beta gamma delta epsilon",
                    add_special_tokens=False).input_ids
    ids = torch.tensor([seq], device=device)
    pos0 = torch.arange(len(seq), device=device)[None]
    physical0 = torch.arange(len(seq), device=device)

    try:
        try:
            backend = eager_backend_fingerprint(model)
        except Exception as exc:
            sink["attention_backend"] = {
                "status": "FAIL", "observed_backend": None, "passes": False,
                "error_type": type(exc).__name__, "error": str(exc),
            }
            raise
        sink["attention_backend"] = {
            "status": "PASS", "observed_backend": "eager", "passes": True,
            "fingerprint": backend,
        }
        sink["attention_backend_fingerprint"] = backend

        # All schedule rows are accumulated before the aggregate decision so a
        # failing row cannot erase later, still-interpretable diagnostics.
        try:
            sink["frozen_schedule_fixtures"] = run_frozen_schedule_fixtures(
                model, tokenizer, zero_gap_tolerance,
                progress=lambda value: sink.__setitem__(
                    "frozen_schedule_fixtures", value))
        except Exception as exc:
            sink["frozen_schedule_fixtures"] = {
                "status": "ERROR", "passes": False,
                "error_type": type(exc).__name__, "error": str(exc),
            }
            raise
        sink["frozen_schedule_fixtures"]["status"] = (
            "PASS" if sink["frozen_schedule_fixtures"]["passes"] else "FAIL")
        sink["frozen_schedule_fixtures"] = dict(
            sink["frozen_schedule_fixtures"])
        if not sink["frozen_schedule_fixtures"]["passes"]:
            raise RuntimeError(
                "frozen schedule-equivalence fixtures failed: " +
                ", ".join(sink["frozen_schedule_fixtures"]["failures"]))

        cache0, logits0 = prefill(
            model, ids, position_ids=pos0, cache_position=physical0)
        rows0 = snapshot_cache(cache0)

        # This reproduces the retired packed/full-prefill measurement. Failure
        # to compute it is recorded but cannot fail or pass this gapped gate.
        try:
            cache37, _ = prefill(
                model, ids, position_ids=pos0 + 37,
                cache_position=physical0)
            rows37 = snapshot_cache(cache37)
            native = compare_rows(rows0, move_key_rows(rows37, -37, theta))
            wrong_sign = compare_rows(
                rows0, move_key_rows(rows37, 37, theta))
            roundtrip = compare_rows(
                rows0,
                move_key_rows(move_key_rows(rows0, 37, theta), -37, theta))
            zero = compare_rows(rows0, move_key_rows(rows0, 0, theta))
            sink["retired_packed_diagnostic"] = {
                "authorizes_run": False,
                "native_shift_k_max_abs": max(x["k_max_abs"] for x in native),
                "native_shift_v_max_abs": max(x["v_max_abs"] for x in native),
                "native_shift_per_layer": native,
                "wrong_sign_shift_k_max_abs": max(
                    x["k_max_abs"] for x in wrong_sign),
                "wrong_sign_failure_injection_detected": (
                    max(x["k_max_abs"] for x in wrong_sign) >
                    max(x["k_max_abs"] for x in native)),
                "roundtrip_k_max_abs": max(x["k_max_abs"] for x in roundtrip),
                "zero_rotation_k_max_abs": max(x["k_max_abs"] for x in zero),
            }
        except Exception as diagnostic_exc:
            sink["retired_packed_diagnostic"] = {
                "authorizes_run": False,
                "diagnostic_error": f"{type(diagnostic_exc).__name__}: "
                                    f"{diagnostic_exc}",
            }

        # Retain the original v3 field as a non-authorizing compatibility view
        # of Amendment 4's first frozen schedule row.
        first_schedule = sink["frozen_schedule_fixtures"]["contiguous"][0]
        sink["zero_gap_equivalence"] = {
            "k_max_abs": first_schedule["cache_k_max_abs"],
            "v_max_abs": first_schedule["cache_v_max_abs"],
            "last_logits_max_abs": first_schedule["last_logits_max_abs"],
            "tolerance": zero_gap_tolerance,
            "authorizes_v4": False,
        }

        # Snapshot/rebuild continuation identity compares the original live
        # cache to an independently rebuilt cache, rather than two rebuilds.
        next_id = tokenizer(" z", add_special_tokens=False).input_ids[:1]
        if not next_id:
            raise RuntimeError("snapshot/rebuild fixture tokenized empty")
        nxt = torch.tensor([next_id], device=device)
        nxtpos = torch.tensor([[len(seq)]], device=device)
        nxtcache = torch.tensor([len(seq)], device=device)
        with torch.no_grad():
            original = model(
                input_ids=nxt, past_key_values=cache0,
                position_ids=nxtpos, cache_position=nxtcache,
                use_cache=True, logits_to_keep=0)
            rebuilt = model(
                input_ids=nxt,
                past_key_values=rebuild_cache(rows0, DynamicCache),
                position_ids=nxtpos, cache_position=nxtcache,
                use_cache=True, logits_to_keep=0)
        rebuild_logits = float(
            (original.logits.float() - rebuilt.logits.float()).abs().max())
        rebuild_k, rebuild_v = _snapshot_max_abs(
            snapshot_cache(original.past_key_values),
            snapshot_cache(rebuilt.past_key_values))
        sink["snapshot_rebuild_equivalence"] = {
            "logits_max_abs": rebuild_logits,
            "k_max_abs": rebuild_k,
            "v_max_abs": rebuild_v,
        }
        if max(rebuild_logits, rebuild_k, rebuild_v) > identity_tolerance:
            raise RuntimeError("snapshot/rebuild continuation differs")

        # Independently construct the physical causal mask. Both forwards use
        # gapped logical positions but contiguous physical cache positions.
        extension = tokenizer(" z A B", add_special_tokens=False).input_ids
        if len(extension) < 3:
            raise RuntimeError("causal-mask fixture tokenization changed")
        q = len(extension)
        past_n = len(seq)
        ext_ids = torch.tensor([extension], device=device)
        logical = torch.arange(37, 37 + q, device=device)[None]
        physical = torch.arange(past_n, past_n + q, device=device)
        with torch.no_grad():
            automatic = model(
                input_ids=ext_ids,
                past_key_values=rebuild_cache(rows0, DynamicCache),
                position_ids=logical, cache_position=physical,
                use_cache=True, logits_to_keep=0)
            mask = torch.full(
                (1, 1, q, past_n + q),
                torch.finfo(next(model.parameters()).dtype).min,
                dtype=next(model.parameters()).dtype, device=device)
            for qi in range(q):
                mask[..., qi, :past_n + qi + 1] = 0
            explicit = model(
                input_ids=ext_ids,
                past_key_values=rebuild_cache(rows0, DynamicCache),
                position_ids=logical, cache_position=physical,
                attention_mask=mask, use_cache=True, logits_to_keep=0)
        mask_logits = float(
            (automatic.logits.float() - explicit.logits.float()).abs().max())
        mask_k, mask_v = _snapshot_max_abs(
            snapshot_cache(automatic.past_key_values),
            snapshot_cache(explicit.past_key_values))
        sink["automatic_vs_independent_4d_physical_causal_mask"] = {
            "logical_positions": logical[0].tolist(),
            "physical_cache_positions": physical.tolist(),
            "logits_max_abs": mask_logits,
            "k_max_abs": mask_k,
            "v_max_abs": mask_v,
        }
        if max(mask_logits, mask_k, mask_v) > identity_tolerance:
            raise RuntimeError("automatic and explicit physical causal masks differ")

        # A later-token mutation cannot affect earlier logits or cache rows.
        mutated = list(extension)
        replacement = tokenizer(" C", add_special_tokens=False).input_ids
        if not replacement:
            raise RuntimeError("causal mutation replacement tokenized empty")
        mutated[-1] = replacement[0]
        with torch.no_grad():
            future = model(
                input_ids=torch.tensor([mutated], device=device),
                past_key_values=rebuild_cache(rows0, DynamicCache),
                position_ids=logical, cache_position=physical,
                use_cache=True, logits_to_keep=0)
        earlier_logits = float(
            (automatic.logits[:, :-1].float() -
             future.logits[:, :-1].float()).abs().max())
        automatic_rows = snapshot_cache(automatic.past_key_values)
        future_rows = snapshot_cache(future.past_key_values)
        earlier_cache = 0.0
        for (ka, va), (kb, vb) in zip(automatic_rows, future_rows):
            earlier_cache = max(
                earlier_cache,
                float((ka[..., past_n:-1, :].float() -
                       kb[..., past_n:-1, :].float()).abs().max()),
                float((va[..., past_n:-1, :].float() -
                       vb[..., past_n:-1, :].float()).abs().max()))
        sink["causal_future_mutation"] = {
            "earlier_logits_max_abs": earlier_logits,
            "earlier_cache_max_abs": earlier_cache,
        }
        if max(earlier_logits, earlier_cache) > identity_tolerance:
            raise RuntimeError("future token changed earlier causal outputs")

        # Logical positions are never valid physical cache indices. Exercise
        # the fail-closed validator with the production layout's gap.
        try:
            validate_position_schedule(
                logical[0].tolist(), logical[0].tolist(),
                physical_start=past_n)
        except Exception as expected:
            sink["logical_as_cache_position_failure_injection"] = {
                "rejected": True,
                "error": f"{type(expected).__name__}: {expected}",
            }
        else:
            raise RuntimeError("logical positions were accepted as cache_position")

        # Generated source-of-record versus independent one-token replay.
        gen_messages = [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content":
             "Reply with the single word OK and then stop. Do not explain."},
        ]
        generated = capture_generated_summary(
            model, tokenizer, gen_messages, max_tokens=192)
        replay = capture_forced_summary(
            model, tokenizer, gen_messages, generated.summary_ids,
            source_kind="loaded_gapped_gate_replay")
        replay_identity = validate_generated_replay(
            generated, replay, identity_tolerance)
        sink["generated_replay"] = replay_identity
        sink["generated_token_count"] = len(generated.summary_ids)
        generated.cache = None
        replay.cache = None

        target = fake_conv("c10", "A", "target-tail")
        donor = fake_conv("c13", "B", "donor-tail")
        fresh_messages = fresh_source_messages(target, REQUEST)
        summary_ids = rendered_assistant_content_ids(
            tokenizer, fresh_messages, SUMMARY)
        correct = capture_forced_summary(
            model, tokenizer, correct_source_messages(target, REQUEST),
            summary_ids, source_kind="loaded_gapped_correct")
        matched = matched_wrong_prefix_ids(tokenizer, target, donor, REQUEST)
        if matched.correct_ids != correct.prefix_ids:
            raise RuntimeError("matched-wrong baseline differs from correct source")
        _validate_exact_length_wrong(
            matched.correct_ids, matched.wrong_ids,
            matched.structural_positions, matched.content_positions,
            tokenizer.all_special_ids)
        altered_structure = list(matched.wrong_ids)
        changed_position = matched.structural_positions[0]
        altered_structure[changed_position] = (
            int(altered_structure[changed_position]) + 1)
        try:
            _validate_exact_length_wrong(
                matched.correct_ids, altered_structure,
                matched.structural_positions, matched.content_positions,
                tokenizer.all_special_ids)
        except RuntimeError as expected:
            sink["altered_structure_failure_injection"] = {
                "rejected": True, "error": str(expected)}
        else:
            raise RuntimeError("altered wrong-history structure was accepted")
        wrong = capture_forced_prefix_ids(
            model, tokenizer, matched.wrong_ids, summary_ids,
            source_kind="loaded_gapped_wrong",
            summary_start=correct.summary_start)
        layout, _trace, fresh_boundary, fresh_rows = \
            build_gapped_fresh_boundary(
                model, tokenizer, target, SUMMARY, summary_ids, REQUEST,
                correct.prefix_ids)
        correct.cache = None
        wrong.cache = None
        if not (correct.summary_start == wrong.summary_start ==
                layout.source_summary_start):
            raise RuntimeError("correct/wrong/gapped summary positions differ")
        _require_summary_boundary(fresh_boundary, layout)
        if not layout.post_summary_ids:
            raise RuntimeError("tail-recomputation fixture has no post-summary tokens")
        sink["position_schedule"] = {
            "source_summary_start": layout.source_summary_start,
            "physical_summary_start": layout.physical_summary_start,
            "logical_gap": (layout.source_summary_start -
                            layout.physical_summary_start),
            "context_position_ids": layout.context_position_ids,
            "physical_cache_positions": list(range(len(layout.context_ids))),
            "wrong_prefix_length_equal": True,
            "wrong_structural_positions": len(matched.structural_positions),
            "wrong_content_positions": len(matched.content_positions),
        }

        self_boundary = replace_summary_rows(
            fresh_boundary, fresh_rows, layout.physical_summary_start,
            use_keys=True, use_values=True)
        if any(not torch.equal(a, b)
               for fresh_pair, self_pair in zip(fresh_boundary, self_boundary)
               for a, b in zip(fresh_pair, self_pair)):
            raise RuntimeError("fresh summary self-replacement changed boundary")
        c_boundary, _ = gapped_arm_boundary(
            "G_correct", fresh_boundary, correct.rows, wrong.rows,
            layout.physical_summary_start, 20_260_711)
        w_boundary, _ = gapped_arm_boundary(
            "G_wrong", fresh_boundary, correct.rows, wrong.rows,
            layout.physical_summary_start, 20_260_711)
        _verify_intervention(
            fresh_boundary, c_boundary, correct.rows,
            layout.physical_summary_start, use_keys=True, use_values=True)
        _verify_intervention(
            fresh_boundary, w_boundary, wrong.rows,
            layout.physical_summary_start, use_keys=True, use_values=True)
        sink["bit_exact_intervention"] = {
            "fresh_self_replacement": True,
            "correct_insert_and_non_summary_preservation": True,
            "wrong_insert_and_non_summary_preservation": True,
        }

        def complete_boundary(boundary):
            _require_summary_boundary(boundary, layout)
            before = row_hashes(boundary)
            full = append_gapped_post_summary(model, boundary, layout)
            if before != row_hashes(boundary):
                raise RuntimeError("tail append mutated the fork boundary")
            if _snapshot_length(full) != len(layout.context_ids):
                raise RuntimeError("recomputed tail has wrong physical length")
            return full

        full_fresh = complete_boundary(list(fresh_boundary))
        full_correct = complete_boundary(c_boundary)
        full_wrong = complete_boundary(w_boundary)

        def fixed_continuation_logits(snapshot):
            with torch.no_grad():
                out = model(
                    input_ids=torch.tensor(
                        [[FROZEN_CONTINUATION_ID]], device=model.device),
                    past_key_values=rebuild_cache(snapshot, DynamicCache),
                    position_ids=torch.tensor(
                        [[layout.logical_next_position]], device=model.device),
                    cache_position=torch.tensor(
                        [len(layout.context_ids)], device=model.device),
                    use_cache=True, logits_to_keep=0)
            return out.logits[:, -1, :]

        baseline_continuation = fixed_continuation_logits(full_fresh)

        # The same function must reject an already-tailed snapshot rather than
        # silently appending a second close/tail sequence.
        try:
            _require_summary_boundary(full_fresh, layout)
        except RuntimeError as expected:
            sink["pre_tailed_boundary_failure_injection"] = {
                "rejected": True, "error": str(expected)}
        else:
            raise RuntimeError("pre-tailed boundary was accepted")

        # The source intervention must flow through a separately recomputed
        # close/tail. A deterministic V perturbation provides the sensitivity
        # control; the unperturbed boundary and full cache remain immutable.
        attempts = []
        sensitivity_pass = False
        tail_changed = False
        s0, s1 = layout.physical_summary_start, layout.physical_summary_end
        for epsilon in (0.1, 0.3, 1.0, 3.0):
            perturbed = [(k.clone(), v.clone()) for k, v in fresh_boundary]
            for _key, value in perturbed:
                pattern = torch.ones_like(value[..., s0:s1, :])
                pattern[..., 1::2] *= -1
                value[..., s0:s1, :] += epsilon * pattern
            full_perturbed = complete_boundary(perturbed)
            continuation_change = float((
                fixed_continuation_logits(full_perturbed).float() -
                baseline_continuation.float()).abs().max())
            post = layout.physical_summary_end
            tail_diff = 0.0
            for (kf, vf), (kp, vp) in zip(full_fresh, full_perturbed):
                tail_diff = max(
                    tail_diff,
                    float((kp[..., post:, :].float() -
                           kf[..., post:, :].float()).abs().max()),
                    float((vp[..., post:, :].float() -
                           vf[..., post:, :].float()).abs().max()))
            attempts.append({
                "epsilon": epsilon,
                "fixed_continuation_logits_max_abs": continuation_change,
                "recomputed_post_summary_kv_max_abs": tail_diff,
            })
            sensitivity_pass = sensitivity_pass or continuation_change > 1e-4
            tail_changed = tail_changed or tail_diff > 0
            if sensitivity_pass and tail_changed:
                break
        sink["tail_recomputation_and_downstream_sensitivity"] = {
            "passes": sensitivity_pass and tail_changed,
            "post_summary_token_count": len(layout.post_summary_ids),
            "attempts": attempts,
        }
        if not (sensitivity_pass and tail_changed):
            raise RuntimeError("gapped downstream sensitivity control failed")

        sink["retired_G_delta"] = {
            "retired_by": "Amendment 4",
            "executed": False,
            "authorizes_run": False,
        }

        calibrations = validate_calibration_constructions(tokenizer)
        if calibrations.get("passes") is not True:
            raise RuntimeError("technical calibration construction gate failed")
        sink["calibration_construction_only"] = calibrations

        sink["technical_branch_cache_lengths"] = {
            "fresh_baseline": _snapshot_length(full_fresh),
            "correct_source_copy": _snapshot_length(full_correct),
            "wrong_source_copy": _snapshot_length(full_wrong),
        }
        sink["passes"] = True
        return sink
    except Exception as exc:
        sink["passes"] = False
        sink["failure"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        return sink


def run_ladder() -> dict:
    global _LAST_LADDER_DIAGNOSTICS
    _LAST_LADDER_DIAGNOSTICS = {}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    if model.device.type != "cpu":
        raise RuntimeError(
            f"v4 local ladder must use observed-equivalent CPU, got {model.device}")
    loaded_gates = run_loaded_gapped_gates(
        model, tokenizer, identity_tolerance=1e-4)
    _LAST_LADDER_DIAGNOSTICS["loaded_gapped_production_gate"] = loaded_gates
    if not loaded_gates.get("passes"):
        failure = loaded_gates.get("failure", {})
        raise RuntimeError(
            f"loaded gapped gate failed: {failure.get('error', failure)}")

    target = fake_conv("c10", "A", "target-tail")
    donor = fake_conv("c13", "B", "donor-tail")
    fresh_messages = fresh_source_messages(target, REQUEST)
    summary_ids = rendered_assistant_content_ids(
        tokenizer, fresh_messages, SUMMARY)
    correct = capture_forced_summary(
        model, tokenizer, correct_source_messages(target, REQUEST), summary_ids,
        source_kind="ladder_correct_forced")
    matched = matched_wrong_prefix_ids(tokenizer, target, donor, REQUEST)
    if matched.correct_ids != correct.prefix_ids:
        raise RuntimeError("ladder correct source differs from matched baseline")
    _validate_exact_length_wrong(
        matched.correct_ids, matched.wrong_ids,
        matched.structural_positions, matched.content_positions,
        tokenizer.all_special_ids)
    wrong = capture_forced_prefix_ids(
        model, tokenizer, matched.wrong_ids, summary_ids,
        source_kind="ladder_wrong_exact_length",
        summary_start=correct.summary_start)

    # A_full is reconstructed before releasing the live correct-source cache.
    _full_messages, _full_context_ids, full_source = complete_assistant_context(
        model, tokenizer, correct_source_messages(target, REQUEST), correct)
    layout, fresh_trace, fresh_boundary, fresh_rows = \
        build_gapped_fresh_boundary(
            model, tokenizer, target, SUMMARY, summary_ids, REQUEST,
            correct.prefix_ids)
    correct.cache = None
    wrong.cache = None
    if not (correct.summary_start == wrong.summary_start ==
            layout.source_summary_start):
        raise RuntimeError("ladder source summary positions differ")
    _require_summary_boundary(fresh_boundary, layout)

    self_replaced = replace_summary_rows(
        fresh_boundary, fresh_rows, layout.physical_summary_start,
        use_keys=True, use_values=True)
    self_exact = all(torch.equal(a, b) and torch.equal(c, d)
                     for (a, c), (b, d) in zip(fresh_boundary, self_replaced))
    if not self_exact:
        raise RuntimeError("gapped fresh self-replacement changed a tensor")

    arm_cache_lengths = {"A_full": _snapshot_length(full_source)}
    for arm in GAPPED_ARM_NAMES[1:]:
        boundary, diag = gapped_arm_boundary(
            arm, fresh_boundary, correct.rows, wrong.rows,
            layout.physical_summary_start, 20_260_711)
        _require_summary_boundary(boundary, layout)
        if arm == "G_correct":
            _verify_intervention(
                fresh_boundary, boundary, correct.rows,
                layout.physical_summary_start,
                use_keys=True, use_values=True)
        elif arm == "G_wrong":
            _verify_intervention(
                fresh_boundary, boundary, wrong.rows,
                layout.physical_summary_start,
                use_keys=True, use_values=True)
        elif arm == "G_Vcorrect":
            _verify_intervention(
                fresh_boundary, boundary, correct.rows,
                layout.physical_summary_start,
                use_keys=False, use_values=True)
        elif arm == "G_Kcorrect":
            _verify_intervention(
                fresh_boundary, boundary, correct.rows,
                layout.physical_summary_start,
                use_keys=True, use_values=False)
        full = append_gapped_post_summary(model, boundary, layout)
        arm_cache_lengths[arm] = _snapshot_length(full)
    if tuple(arm_cache_lengths) != GAPPED_ARM_NAMES:
        raise RuntimeError(
            f"ladder arm set/order changed: {tuple(arm_cache_lengths)}")

    # Exercise both unique deterministic calibration variants. Conversation
    # repetitions are not independent calibration evidence (Amendment 2).
    calibrations = validate_calibration_constructions(tokenizer)
    if set(calibrations.get("label_coverage", [])) != {"A", "B"}:
        raise RuntimeError(
            "ladder calibration did not cover both label variants")
    if calibrations.get("passes") is not True:
        raise RuntimeError("ladder calibration construction failed")

    # A killed run after rendering must load identical text and never rerender.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fp = {"model": MODEL, "test": "resume"}
        path = checkpoint_path(root, 1, "c10")
        saved = save_render(path, fingerprint=fp, order_position=1,
                            conversation=target, reply_records=[])
        loaded = read_checkpoint(path, fp, "rendered")
        resume_exact = saved == loaded and loaded["conversation"] == target
    if not resume_exact:
        raise RuntimeError("render resume was not exact")

    return {
        "schema": 2,
        "amendment_id": AMENDMENT_ID,
        "design_id": DESIGN_ID,
        "status": "PASS", "model": MODEL,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "dtype": str(next(model.parameters()).dtype),
        "device": str(model.device),
        "forced_summary_ids": summary_ids,
        "fresh_trace": asdict(fresh_trace),
        "self_replacement_exact": self_exact,
        "position_schedule": {
            "source_summary_start": layout.source_summary_start,
            "physical_summary_start": layout.physical_summary_start,
            "logical_next_position": layout.logical_next_position,
            "physical_context_length": len(layout.context_ids),
            "logical_gap": (layout.source_summary_start -
                            layout.physical_summary_start),
        },
        "exact_length_wrong": {
            "prefix_length": len(matched.correct_ids),
            "structural_position_count": len(matched.structural_positions),
            "content_position_count": len(matched.content_positions),
            "replacement_count": len(matched.replacements),
        },
        "arm_constructor_checks": {
            "exact_arm_order": list(arm_cache_lengths),
            "all_completed_without_semantic_scoring": True,
        },
        "arm_cache_lengths": arm_cache_lengths,
        "calibration_construction_only": calibrations,
        "retired_G_delta": {"executed": False, "retired_by": "Amendment 4"},
        "render_resume_exact": resume_exact,
        "loaded_gapped_production_gate": loaded_gates,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    print(f"RUN coherent_state_ladder model={MODEL} -> {args.output}", flush=True)
    try:
        result = run_ladder()
    except Exception as exc:
        result = {
                  "schema": 2,
                  "amendment_id": AMENDMENT_ID,
                  "design_id": DESIGN_ID,
                  "status": "FAIL", "error_type": type(exc).__name__,
                  "error": str(exc), "traceback": traceback.format_exc(),
                  "diagnostics": _LAST_LADDER_DIAGNOSTICS,
                  "failed_at": datetime.now(timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
