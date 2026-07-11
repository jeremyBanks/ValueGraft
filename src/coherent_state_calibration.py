"""Prespecified ambiguous-mini-summary calibration assay."""

from __future__ import annotations

import hashlib

from coherent_state_hf import CoherentStateError, row_hashes, sha256_ids
from coherent_state_runtime import (
    AMENDMENT_ID,
    DESIGN_ID,
    append_gapped_post_summary,
    build_gapped_fresh_boundary,
    capture_forced_prefix_ids,
    capture_forced_summary,
    gapped_arm_boundary,
    score_arm,
)
from coherent_state_tokens import (
    generation_prefix_ids,
    gapped_destination_layout,
    probe_layout,
    rendered_assistant_content_ids,
)


CALIBRATION_SYSTEM = (
    "You are maintaining a two-label test record. Follow the record exactly."
)
CALIBRATION_REQUEST = (
    "Compress the record into exactly this deliberately ambiguous sentence, "
    "without naming either label: The recorded choice remains the approved one."
)
CALIBRATION_SUMMARY = "The recorded choice remains the approved one."
CALIBRATION_PROBE = (
    "Which label was approved? Answer exactly `Label A.` or `Label B.`"
)


def calibration_labels(conversation_id: str) -> tuple[str, str]:
    bit = hashlib.sha256(f"20260711:{conversation_id}:calibration".encode()).digest()[0] & 1
    return ("A", "B") if bit == 0 else ("B", "A")


def calibration_source_messages(approved: str) -> list[dict]:
    if approved not in ("A", "B"):
        raise CoherentStateError(f"invalid calibration label: {approved}")
    other = "B" if approved == "A" else "A"
    return [
        {"role": "system", "content": CALIBRATION_SYSTEM},
        {"role": "user", "content":
         f"For this calibration record, Label {approved} is approved. "
         f"Label {other} is explicitly rejected."},
        {"role": "assistant", "content":
         "Understood. I will retain which label is approved."},
        {"role": "user", "content":
         "Keep the calibration record active while we continue."},
        {"role": "assistant", "content":
         "The calibration record remains active."},
        {"role": "user", "content": CALIBRATION_REQUEST},
    ]


def calibration_fresh_messages() -> list[dict]:
    return [
        {"role": "system", "content": CALIBRATION_SYSTEM},
        {"role": "user", "content": CALIBRATION_REQUEST},
    ]


def _calibration_construction(tokenizer, conversation_id: str) -> dict:
    """Validate one frozen calibration variant without a model forward."""
    correct_label, wrong_label = calibration_labels(conversation_id)
    summary_ids = rendered_assistant_content_ids(
        tokenizer, calibration_fresh_messages(), CALIBRATION_SUMMARY)
    if len(summary_ids) < 2:
        raise CoherentStateError("calibration summary is too short")
    correct_messages = calibration_source_messages(correct_label)
    wrong_messages = calibration_source_messages(wrong_label)
    correct_ids = generation_prefix_ids(tokenizer, correct_messages)
    wrong_ids = generation_prefix_ids(tokenizer, wrong_messages)
    if len(correct_ids) != len(wrong_ids):
        raise CoherentStateError("calibration sources are not position matched")
    changed_positions = [i for i, (a, b) in enumerate(
        zip(correct_ids, wrong_ids)) if a != b]
    if not changed_positions:
        raise CoherentStateError("calibration sources differ at no token position")
    marker_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    if len(marker_ids) != 1:
        raise CoherentStateError("calibration chat-boundary marker is not one token")
    starts = [i for i, token_id in enumerate(correct_ids)
              if token_id == marker_ids[0]]
    if len(starts) != len(correct_messages) + 1:
        raise CoherentStateError("calibration message boundary count differs")
    allowed_content_positions = list(range(starts[1], starts[2]))
    allowed = set(allowed_content_positions)
    if any(i not in allowed for i in changed_positions):
        raise CoherentStateError(
            "calibration wrong source changed a non-record message slot")
    special = set(int(x) for x in tokenizer.all_special_ids)
    if any(correct_ids[i] in special or wrong_ids[i] in special
           for i in changed_positions):
        raise CoherentStateError(
            "calibration wrong source changed a structural/special token")
    structural_positions = [i for i in range(len(correct_ids))
                            if i not in changed_positions]
    if any(correct_ids[i] != wrong_ids[i] for i in structural_positions):
        raise CoherentStateError("calibration structural slots differ")
    target_correct = f"Label {correct_label}."
    target_wrong = f"Label {wrong_label}."
    correct_target_ids = tokenizer.encode(
        target_correct, add_special_tokens=False)
    wrong_target_ids = tokenizer.encode(target_wrong, add_special_tokens=False)
    if len(correct_target_ids) != len(wrong_target_ids):
        raise CoherentStateError(
            "calibration targets are not token-length matched")
    conv = {
        "id": f"calibration-{conversation_id}",
        "messages": correct_messages[:-1],
        "sections": {"middle_end_msg": 3},
    }
    layout = gapped_destination_layout(
        tokenizer, conv, CALIBRATION_SUMMARY, summary_ids,
        CALIBRATION_REQUEST, correct_ids)
    rendered_correct_ids = probe_layout(
        tokenizer, layout.messages, layout.context_ids,
        CALIBRATION_PROBE, target_correct).target_ids
    rendered_wrong_ids = probe_layout(
        tokenizer, layout.messages, layout.context_ids,
        CALIBRATION_PROBE, target_wrong).target_ids
    if len(rendered_correct_ids) != len(rendered_wrong_ids):
        raise CoherentStateError(
            "calibration targets differ in rendered scoring-token length")
    return {
        "conversation_id": conversation_id,
        "correct_label": correct_label,
        "wrong_label": wrong_label,
        "summary_text": CALIBRATION_SUMMARY,
        "summary_ids": summary_ids,
        "summary_sha256": sha256_ids(summary_ids),
        "correct_prefix_ids": correct_ids,
        "wrong_prefix_ids": wrong_ids,
        "correct_prefix_sha256": sha256_ids(correct_ids),
        "wrong_prefix_sha256": sha256_ids(wrong_ids),
        "prefix_length": len(correct_ids),
        "changed_positions": changed_positions,
        "allowed_first_record_content_positions": allowed_content_positions,
        "structural_positions": structural_positions,
        "exact_length": True,
        "changed_only_first_record_content": True,
        "structural_slots_equal": True,
        "special_ids_excluded": True,
        "targets": {
            "correct_text": target_correct,
            "wrong_text": target_wrong,
            "correct_ids": correct_target_ids,
            "wrong_ids": wrong_target_ids,
            "equal_token_length": True,
            "token_length": len(correct_target_ids),
            "rendered_correct_ids": rendered_correct_ids,
            "rendered_wrong_ids": rendered_wrong_ids,
            "rendered_equal_token_length": True,
            "rendered_token_length": len(rendered_correct_ids),
        },
    }


def validate_calibration_constructions(tokenizer) -> dict:
    """Validate both unique frozen variants without margins or model execution."""
    variants = {
        cid: _calibration_construction(tokenizer, cid)
        for cid in ("c10", "c07")
    }
    coverage = sorted({row["correct_label"] for row in variants.values()})
    if coverage != ["A", "B"]:
        raise CoherentStateError(
            f"calibration constructions do not cover both labels: {coverage}")
    return {
        "schema": 2,
        "amendment_id": AMENDMENT_ID,
        "design_id": DESIGN_ID,
        "passes": True,
        "model_forwards": 0,
        "semantic_outcomes": 0,
        "label_coverage": coverage,
        "variants": variants,
    }


def run_calibration(model, tokenizer, conversation_id: str) -> dict:
    correct_label, wrong_label = calibration_labels(conversation_id)
    summary_ids = rendered_assistant_content_ids(
        tokenizer, calibration_fresh_messages(), CALIBRATION_SUMMARY)
    if len(summary_ids) < 2:
        raise CoherentStateError("calibration summary is too short to derange")

    correct_messages = calibration_source_messages(correct_label)
    wrong_messages = calibration_source_messages(wrong_label)
    correct = capture_forced_summary(
        model, tokenizer, correct_messages, summary_ids,
        source_kind="calibration_correct_forced")
    wrong_prefix_ids = generation_prefix_ids(tokenizer, wrong_messages)
    if len(correct.prefix_ids) != len(wrong_prefix_ids):
        raise CoherentStateError("calibration sources are not position matched")
    changed_positions = [i for i, (a, b) in enumerate(
        zip(correct.prefix_ids, wrong_prefix_ids)) if a != b]
    if not changed_positions:
        raise CoherentStateError("calibration sources differ at no token position")
    marker_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    if len(marker_ids) != 1:
        raise CoherentStateError("calibration chat-boundary marker is not one token")
    starts = [i for i, token_id in enumerate(correct.prefix_ids)
              if token_id == marker_ids[0]]
    if len(starts) != len(correct_messages) + 1:
        raise CoherentStateError("calibration message boundary count differs")
    allowed_content_block = range(starts[1], starts[2])
    if any(i not in allowed_content_block for i in changed_positions):
        raise CoherentStateError(
            "calibration wrong source changed a non-record message slot")
    special = set(int(x) for x in tokenizer.all_special_ids)
    if any(correct.prefix_ids[i] in special or wrong_prefix_ids[i] in special
           for i in changed_positions):
        raise CoherentStateError(
            "calibration wrong source changed a structural/special token")
    structural_positions = [i for i in range(len(correct.prefix_ids))
                            if i not in changed_positions]
    if any(correct.prefix_ids[i] != wrong_prefix_ids[i]
           for i in structural_positions):
        raise CoherentStateError("calibration structural slots differ")
    wrong = capture_forced_prefix_ids(
        model, tokenizer, wrong_prefix_ids, summary_ids,
        source_kind="calibration_wrong_exact_length_counterfactual",
        summary_start=correct.summary_start)
    if correct.summary_start != wrong.summary_start:
        raise CoherentStateError("calibration summary starts differ")
    conv = {
        "id": f"calibration-{conversation_id}",
        "messages": calibration_source_messages(correct_label)[:-1],
        "sections": {"middle_end_msg": 3},
    }
    layout, fresh_trace, fresh_boundary, fresh_rows = build_gapped_fresh_boundary(
        model, tokenizer, conv, CALIBRATION_SUMMARY, summary_ids,
        CALIBRATION_REQUEST, correct.prefix_ids)
    correct.cache = None
    wrong.cache = None

    plant = {"id": "calibration", "category": "calibration",
             "probe": CALIBRATION_PROBE}
    target = {"calibration": {
        "correct": f"Label {correct_label}.",
        "counterfactual": f"Label {wrong_label}.",
        "basis": "frozen SHA-256 label assignment",
    }}
    # Token-length matching is checked in the rendered scoring path as well as
    # here under standalone tokenization, before any arm contrast is returned.
    a = tokenizer.encode(target["calibration"]["correct"],
                         add_special_tokens=False)
    b = tokenizer.encode(target["calibration"]["counterfactual"],
                         add_special_tokens=False)
    if len(a) != len(b):
        raise CoherentStateError("calibration targets are not token-length matched")

    outcomes = {}
    arm_details = {}
    for arm in ("G_fresh", "G_correct", "G_wrong"):
        boundary, _ = gapped_arm_boundary(
            arm, fresh_boundary, correct.rows, wrong.rows,
            layout.physical_summary_start, 20_260_711)
        snap = append_gapped_post_summary(model, boundary, layout)
        score = score_arm(
            model, tokenizer, snap, layout.messages, layout.context_ids,
            [plant], target,
            logical_context_end=layout.logical_next_position)
        outcomes[arm] = score["conversation_margin"]
        arm_details[arm] = score
        del boundary, snap

    rendered_lengths = {
        arm: {
            side: len(arm_details[arm]["plants"][0][side]["token_ids"])
            for side in ("correct", "counterfactual")
        }
        for arm in arm_details
    }
    if any(row["correct"] != row["counterfactual"]
           for row in rendered_lengths.values()):
        raise CoherentStateError(
            "calibration targets differ in rendered scoring-token length")

    return {
        "schema": 2, "amendment_id": AMENDMENT_ID, "design_id": DESIGN_ID,
        "correct_label": correct_label, "wrong_label": wrong_label,
        "summary_text": CALIBRATION_SUMMARY, "summary_ids": summary_ids,
        "source_prefix_hashes": {
            "correct": correct.prefix_sha256,
            "fresh": sha256_ids(layout.prefix_ids),
            "wrong": wrong.prefix_sha256,
        },
        "source_prefix_token_ids": {
            "correct": correct.prefix_ids,
            "wrong": wrong.prefix_ids,
            "fresh": layout.prefix_ids,
        },
        "wrong_changed_positions": changed_positions,
        "wrong_exact_length_construction": {
            "target_label": correct_label,
            "donor_label": wrong_label,
            "prefix_length": len(correct.prefix_ids),
            "changed_positions": changed_positions,
            "structural_positions": structural_positions,
            "target_ids": [correct.prefix_ids[i] for i in changed_positions],
            "replacement_ids": [wrong.prefix_ids[i] for i in changed_positions],
            "structural_slots_equal": True,
            "special_ids_excluded": True,
        },
        "wrong_decoded_prefix": tokenizer.decode(wrong.prefix_ids),
        "source_summary_row_hashes": {
            "correct": correct.row_hashes,
            "wrong": wrong.row_hashes,
            "fresh": row_hashes(fresh_rows),
        },
        "position_policy": "gapped_same_source_summary_position",
        "summary_start": correct.summary_start,
        "physical_summary_start": layout.physical_summary_start,
        "physical_summary_end": layout.physical_summary_end,
        "logical_next_position": layout.logical_next_position,
        "physical_cache_positions": list(range(len(layout.context_ids))),
        "context_position_ids": layout.context_position_ids,
        "fresh_trace": {
            "start_position": fresh_trace.start_position,
            "end_position": fresh_trace.end_position,
        },
        "rendered_target_token_lengths": rendered_lengths,
        "outcomes": outcomes, "arm_details": arm_details,
    }
