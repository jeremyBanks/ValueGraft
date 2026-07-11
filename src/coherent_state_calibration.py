"""Prespecified ambiguous-mini-summary calibration assay."""

from __future__ import annotations

import hashlib

from coherent_state_hf import CoherentStateError, row_hashes, sha256_ids
from coherent_state_runtime import (
    AMENDMENT_ID,
    DESIGN_ID,
    append_gapped_post_summary,
    build_gapped_fresh_boundary,
    capture_forced_summary,
    gapped_arm_boundary,
    score_arm,
)
from coherent_state_tokens import rendered_assistant_content_ids


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


def run_calibration(model, tokenizer, conversation_id: str) -> dict:
    correct_label, wrong_label = calibration_labels(conversation_id)
    summary_ids = rendered_assistant_content_ids(
        tokenizer, calibration_fresh_messages(), CALIBRATION_SUMMARY)
    if len(summary_ids) < 2:
        raise CoherentStateError("calibration summary is too short to derange")

    correct = capture_forced_summary(
        model, tokenizer, calibration_source_messages(correct_label), summary_ids,
        source_kind="calibration_correct_forced")
    wrong = capture_forced_summary(
        model, tokenizer, calibration_source_messages(wrong_label), summary_ids,
        source_kind="calibration_wrong_forced")
    if len(correct.prefix_ids) != len(wrong.prefix_ids):
        raise CoherentStateError("calibration sources are not position matched")
    if correct.summary_start != wrong.summary_start:
        raise CoherentStateError("calibration summary starts differ")
    changed_positions = [i for i, (a, b) in enumerate(
        zip(correct.prefix_ids, wrong.prefix_ids)) if a != b]
    if not changed_positions:
        raise CoherentStateError("calibration sources differ at no token position")
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
