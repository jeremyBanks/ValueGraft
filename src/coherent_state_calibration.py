"""Prespecified ambiguous-mini-summary calibration assay."""

from __future__ import annotations

import hashlib

from coherent_state_hf import CoherentStateError
from coherent_state_runtime import (
    arm_snapshot,
    capture_forced_summary,
    complete_assistant_context,
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
        {"role": "user", "content": CALIBRATION_REQUEST},
    ]


def calibration_fresh_messages() -> list[dict]:
    return [
        {"role": "system", "content": CALIBRATION_SYSTEM},
        {"role": "user", "content": CALIBRATION_REQUEST},
    ]


def run_calibration(model, tokenizer, conversation_id: str,
                    rope_theta: float) -> dict:
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
    fresh_messages = calibration_fresh_messages()
    fresh = capture_forced_summary(
        model, tokenizer, fresh_messages, summary_ids,
        source_kind="calibration_fresh_forced")
    context_messages, context_ids, fresh_snapshot = complete_assistant_context(
        model, tokenizer, fresh_messages, fresh)
    correct.cache = None
    wrong.cache = None
    fresh.cache = None

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
    deltas = {
        "C_coherent": len(fresh.prefix_ids) - len(correct.prefix_ids),
        "W_wrong": len(fresh.prefix_ids) - len(wrong.prefix_ids),
    }
    for arm in ("F_fresh", "C_coherent", "W_wrong"):
        snap, _ = arm_snapshot(
            arm, fresh_snapshot, correct.rows, wrong.rows,
            fresh.summary_start, deltas["C_coherent"], deltas["W_wrong"],
            rope_theta, 20_260_711)
        score = score_arm(model, tokenizer, snap, context_messages, context_ids,
                          [plant], target)
        outcomes[arm] = score["conversation_margin"]
        arm_details[arm] = score
        del snap

    return {
        "correct_label": correct_label, "wrong_label": wrong_label,
        "summary_text": CALIBRATION_SUMMARY, "summary_ids": summary_ids,
        "source_prefix_hashes": {
            "correct": correct.prefix_sha256, "fresh": fresh.prefix_sha256,
            "wrong": wrong.prefix_sha256,
        },
        "outcomes": outcomes, "arm_details": arm_details,
    }
