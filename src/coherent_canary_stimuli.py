"""Tokenizer-only validation for authored v12 decision-canary fixtures."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from arms_common import canonical_ids_any, render_hf
from coherent_canary_schema import (
    CASE_SCHEMA,
    DESIGN_ID,
    ENGINEERED_CASE_IDS,
    MODEL_ID,
    MODEL_REVISION,
    CanarySchemaError,
    require_matching_geometry,
)
from coherent_canary_tokens import build_role_native_plan, message_starts


def sha256_ints(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanarySchemaError(message)


def _messages(case: dict, variant: str) -> list[dict]:
    raw = case.get("variants", {}).get(variant, {}).get("messages")
    _require(isinstance(raw, list) and len(raw) >= 5,
             f"{variant} messages are incomplete")
    return raw


def _validate_roles(messages: list[dict], case_id: str) -> None:
    for index, message in enumerate(messages):
        expected = "system" if index == 0 else ("user" if index % 2 else "assistant")
        _require(message.get("role") == expected,
                 f"{case_id} message {index} role != {expected}")
        content = message.get("content")
        _require(isinstance(content, str) and content.strip() != "",
                 f"{case_id} message {index} content is empty")
        _require("<|im_" not in content,
                 f"{case_id} message {index} contains chat-template literal")


def _target(case: dict, key: str) -> dict:
    value = case.get(key)
    _require(isinstance(value, dict), f"missing {key} record")
    return value


def _variant_evidence(tokenizer, messages: list[dict]) -> tuple[dict, object]:
    ids = [int(value) for value in canonical_ids_any(tokenizer, messages, render_hf)]
    starts = message_starts(tokenizer, ids)
    _require(len(starts) == len(messages), "canonical message-start coverage differs")
    widths = [
        (starts[index + 1] if index + 1 < len(starts) else len(ids)) - start
        for index, start in enumerate(starts)
    ]
    content_widths = [
        len(tokenizer.encode(message["content"], add_special_tokens=False))
        for message in messages
    ]
    _require(max(content_widths) <= 4096, "an authored message exceeds 4096 tokens")
    plan = build_role_native_plan(tokenizer, messages)
    evidence = {
        "canonical_token_count": len(ids),
        "canonical_token_ids_sha256": sha256_ints(ids),
        "literal_messages_sha256": sha256_json(messages),
        "message_start_positions": starts,
        "message_start_positions_sha256": sha256_ints(starts),
        "canonical_message_widths": widths,
        "content_token_widths": content_widths,
        "role_native_event_kinds": [event.kind for event in plan.events],
        "role_native_event_widths": [event.width for event in plan.events],
        "role_native_geometry_sha256": sha256_json(plan.geometry()),
        "carrier_regions": asdict(plan.regions),
        "decoded_round_trip": True,
    }
    return evidence, plan


def validate_case(tokenizer, case: dict) -> dict:
    _require(case.get("schema") == CASE_SCHEMA, "case schema differs")
    _require(case.get("design_id") == DESIGN_ID, "design ID differs")
    case_id = str(case.get("case_id", ""))
    _require(case_id in ENGINEERED_CASE_IDS, f"unknown engineered case ID: {case_id}")
    _require(case.get("status") == "DRAFT_UNREVIEWED",
             f"{case_id} status is not DRAFT_UNREVIEWED")
    _require(case.get("execution_ready") is False,
             f"{case_id} incorrectly authorizes execution")
    _require(case.get("stratum") == "engineered", f"{case_id} stratum differs")
    provenance = case.get("authoring_provenance")
    _require(isinstance(provenance, dict) and provenance,
             f"{case_id} authoring provenance is empty")

    correct = _messages(case, "correct")
    wrong = _messages(case, "wrong_focal")
    _require(len(correct) == len(wrong), f"{case_id} message counts differ")
    _validate_roles(correct, case_id)
    _validate_roles(wrong, case_id)
    middle = case.get("middle_end_msg")
    _require(isinstance(middle, int) and 2 < middle < len(correct),
             f"{case_id} middle_end_msg is invalid")
    _require(correct[middle]["role"] == "user",
             f"{case_id} retained tail does not start with user")

    allow_raw = case.get("changed_message_allowlist")
    _require(isinstance(allow_raw, list) and allow_raw,
             f"{case_id} changed-message allowlist is empty")
    allow = {int(value) for value in allow_raw}
    _require(all(0 < value < middle for value in allow),
             f"{case_id} changed message lies outside evicted block")
    changed = set()
    for index, (left, right) in enumerate(zip(correct, wrong)):
        _require(left["role"] == right["role"],
                 f"{case_id} role differs at message {index}")
        if left["content"] != right["content"]:
            changed.add(index)
        if index not in allow:
            _require(left == right,
                     f"{case_id} non-allowlisted message {index} differs")
    _require(changed == allow,
             f"{case_id} changed indices {sorted(changed)} != allowlist {sorted(allow)}")
    _require(correct[0] == wrong[0], f"{case_id} system message differs")
    _require(correct[middle:] == wrong[middle:], f"{case_id} retained tail differs")

    correct_content_widths = [
        len(tokenizer.encode(message["content"], add_special_tokens=False))
        for message in correct
    ]
    wrong_content_widths = [
        len(tokenizer.encode(message["content"], add_special_tokens=False))
        for message in wrong
    ]
    _require(correct_content_widths == wrong_content_widths,
             f"{case_id} per-message content token widths differ")

    focal = _target(case, "focal")
    control = _target(case, "nonfocal_control")
    for key in ("plant_id", "probe", "correct_target", "counterfactual_target",
                "why_derived", "why_counterfactual_reverses"):
        _require(str(focal.get(key, "")).strip() != "",
                 f"{case_id} focal field {key} is empty")
    _require(focal.get("category") == "derived_decision",
             f"{case_id} focal category differs")
    _require(focal["correct_target"].strip() != focal["counterfactual_target"].strip(),
             f"{case_id} focal targets are identical")
    for key in ("plant_id", "probe", "target", "countertarget",
                "why_independent_of_focal"):
        _require(str(control.get(key, "")).strip() != "",
                 f"{case_id} control field {key} is empty")
    _require(control.get("category") == "unchanged_control",
             f"{case_id} control category differs")
    control_indices = {int(value) for value in control.get(
        "establishing_message_indices", [])}
    _require(control_indices and not control_indices.intersection(allow),
             f"{case_id} nonfocal control overlaps changed messages")

    correct_evidence, correct_plan = _variant_evidence(tokenizer, correct)
    wrong_evidence, wrong_plan = _variant_evidence(tokenizer, wrong)
    require_matching_geometry(correct_plan, wrong_plan)
    correct_ids = correct_plan.token_ids
    wrong_ids = wrong_plan.token_ids
    changed_positions = [
        index for index, (left, right) in enumerate(zip(correct_ids, wrong_ids))
        if left != right
    ]
    _require(changed_positions, f"{case_id} canonical streams are identical")

    model_binding = case.get("tokenizer_binding", {})
    _require(model_binding.get("model") == MODEL_ID, f"{case_id} tokenizer model differs")
    _require(model_binding.get("revision") == MODEL_REVISION,
             f"{case_id} tokenizer revision differs")
    return {
        "case_id": case_id,
        "status": "MECHANICAL_DRAFT_PASS",
        "execution_ready": False,
        "review_ready": True,
        "correct": correct_evidence,
        "wrong_focal": wrong_evidence,
        "pair": {
            "changed_message_allowlist": sorted(allow),
            "canonical_changed_position_count": len(changed_positions),
            "canonical_changed_positions_sha256": sha256_ints(changed_positions),
            "retained_tail_byte_identical": True,
            "role_native_geometry_identical": True,
        },
        "targets": {
            "focal_correct_token_count": len(tokenizer.encode(
                focal["correct_target"], add_special_tokens=False)),
            "focal_counterfactual_token_count": len(tokenizer.encode(
                focal["counterfactual_target"], add_special_tokens=False)),
            "control_target_token_count": len(tokenizer.encode(
                control["target"], add_special_tokens=False)),
            "control_countertarget_token_count": len(tokenizer.encode(
                control["countertarget"], add_special_tokens=False)),
        },
    }


def load_and_validate_case(tokenizer, path: str | Path) -> dict:
    raw = json.loads(Path(path).read_text())
    result = validate_case(tokenizer, raw)
    result["path"] = str(path)
    result["input_file_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result
