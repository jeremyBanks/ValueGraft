#!/usr/bin/env python3
"""Tokenizer-only validation for frozen v12 technical control fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from coherent_canary_schema import (
    DESIGN_ID,
    MODEL_ID,
    MODEL_REVISION,
    require_matching_geometry,
)
from coherent_canary_stimuli import sha256_ints, sha256_json
from coherent_canary_tokens import (
    build_fresh_destination_plan,
    build_role_native_plan,
    build_turn_aligned_plan,
)
from coherent_state_tokens import generation_prefix_ids


class FixtureError(RuntimeError):
    pass


TECHNICAL_CORRECT = [
    {"role": "system", "content": "Apply the stated policy exactly. At the final question answer with exactly approve or deny."},
    {"role": "user", "content": "Policy: approve only when the status is green. Current status: green."},
    {"role": "assistant", "content": "The policy and current record have been processed. Use the recorded state for the decision."},
    {"role": "user", "content": "At the final question, answer with exactly one policy label."},
    {"role": "assistant", "content": "Understood."},
]
TECHNICAL_WRONG = [dict(row) for row in TECHNICAL_CORRECT]
TECHNICAL_WRONG[1] = {
    "role": "user",
    "content": "Policy: approve only when the status is green. Current status: amber.",
}
IDENTITY_MESSAGES = [
    {"role": "system", "content": "Answer plainly."},
    {"role": "user", "content": "Write one short neutral sentence acknowledging that a record exists."},
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def validate_technical(tokenizer, path: Path) -> dict:
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    require(raw.get("schema") ==
            "coherent_state_decision_canary_v12_technical_control_fixture_v1",
            "technical schema differs")
    require(raw.get("design_id") == DESIGN_ID, "technical design differs")
    require(raw.get("status") == "FROZEN_TOKENIZER_ONLY", "technical status differs")
    require(raw.get("model") == MODEL_ID, "technical model differs")
    require(raw.get("revision") == MODEL_REVISION, "technical revision differs")
    correct = raw.get("correct")
    wrong = raw.get("wrong")
    require(correct == TECHNICAL_CORRECT, "technical correct literal differs")
    require(wrong == TECHNICAL_WRONG, "technical wrong literal differs")
    require(isinstance(correct, list) and isinstance(wrong, list), "histories absent")
    require(len(correct) == len(wrong) == 5, "technical message count differs")
    for messages in (correct, wrong):
        for index, message in enumerate(messages):
            expected = "system" if index == 0 else ("user" if index % 2 else "assistant")
            require(message.get("role") == expected, "technical roles differ")
            require(isinstance(message.get("content"), str) and message["content"],
                    "technical content absent")
    changed = [index for index, (left, right) in enumerate(zip(correct, wrong))
               if left != right]
    require(changed == raw.get("changed_message_allowlist") == [1],
            "technical change set differs")
    correct_widths = [len(tokenizer.encode(row["content"], add_special_tokens=False))
                      for row in correct]
    wrong_widths = [len(tokenizer.encode(row["content"], add_special_tokens=False))
                    for row in wrong]
    require(correct_widths == wrong_widths, "technical content widths differ")
    middle = raw.get("middle_end_msg")
    require(middle == 3, "technical boundary differs")
    require(raw.get("probe") == "Decision?", "technical probe differs")
    require(raw.get("correct_target") == "approve", "technical correct target differs")
    require(raw.get("counterfactual_target") == "deny",
            "technical counterfactual target differs")
    require(raw.get("margin") == "log_p_approve_minus_log_p_deny",
            "technical margin differs")
    require(raw.get("uses_common_carrier") ==
            "data/coherent_canary_v12/fixed_text_token_evidence_v2.json",
            "technical carrier binding differs")
    n_correct = build_role_native_plan(tokenizer, correct, middle_end_msg=middle)
    n_wrong = build_role_native_plan(tokenizer, wrong, middle_end_msg=middle)
    p_correct = build_turn_aligned_plan(tokenizer, correct, middle_end_msg=middle)
    p_wrong = build_turn_aligned_plan(tokenizer, wrong, middle_end_msg=middle)
    require_matching_geometry(n_correct, n_wrong)
    require_matching_geometry(p_correct, p_wrong)
    fresh_correct = build_fresh_destination_plan(
        tokenizer, correct, middle_end_msg=middle)
    fresh_wrong = build_fresh_destination_plan(tokenizer, wrong, middle_end_msg=middle)
    require(fresh_correct == fresh_wrong, "technical fresh plans differ")
    correct_target_ids = tokenizer.encode(
        raw["correct_target"], add_special_tokens=False)
    wrong_target_ids = tokenizer.encode(
        raw["counterfactual_target"], add_special_tokens=False)
    require(len(correct_target_ids) == len(wrong_target_ids) == 1,
            "technical targets are not one token")
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "changed_messages": changed,
        "content_token_widths": correct_widths,
        "correct_target_ids": correct_target_ids,
        "counterfactual_target_ids": wrong_target_ids,
        "n_correct_geometry_sha256": sha256_json(n_correct.geometry()),
        "n_wrong_geometry_sha256": sha256_json(n_wrong.geometry()),
        "p_correct_geometry_sha256": sha256_json(p_correct.geometry()),
        "p_wrong_geometry_sha256": sha256_json(p_wrong.geometry()),
        "fresh_geometry_sha256": sha256_json(fresh_correct.geometry()),
        "fresh_token_ids_sha256": sha256_ints(fresh_correct.token_ids),
        "fresh_logical_positions_sha256": sha256_ints(
            fresh_correct.logical_positions),
        "carrier_regions": {
            "source": n_correct.geometry()["regions"],
            "physical": fresh_correct.geometry()["physical_regions"],
        },
    }


def validate_identity(tokenizer, path: Path) -> dict:
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    require(raw.get("schema") ==
            "coherent_state_decision_canary_v12_generated_forced_identity_fixture_v1",
            "identity schema differs")
    require(raw.get("design_id") == DESIGN_ID, "identity design differs")
    require(raw.get("status") == "FROZEN_LITERAL_PRE_FORWARD",
            "identity status differs")
    require(raw.get("model") == MODEL_ID, "identity model differs")
    require(raw.get("revision") == MODEL_REVISION, "identity revision differs")
    messages = raw.get("messages")
    require(messages == IDENTITY_MESSAGES, "identity message literals differ")
    require(isinstance(messages, list) and len(messages) == 2,
            "identity messages differ")
    require([row.get("role") for row in messages] == ["system", "user"],
            "identity roles differ")
    require(raw.get("temperature") == 0, "identity temperature differs")
    require(raw.get("max_content_tokens") == 64, "identity cap differs")
    require(raw.get("normal_eos_required") is True, "identity EOS rule differs")
    prefix_ids = generation_prefix_ids(tokenizer, messages)
    return {
        "path": str(path),
        "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "generation_prefix_token_count": len(prefix_ids),
        "generation_prefix_token_ids_sha256": sha256_ints(prefix_ids),
        "message_content_token_widths": [len(tokenizer.encode(
            row["content"], add_special_tokens=False)) for row in messages],
        "temperature": 0,
        "max_content_tokens": 64,
        "normal_eos_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--technical", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    result = {
        "schema": "coherent_state_decision_canary_v12_control_fixture_validation_v1",
        "status": "TOKENIZER_FIXTURES_PASS",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "tokenizer_class": tokenizer.__class__.__name__,
        "technical": validate_technical(tokenizer, args.technical),
        "generated_forced_identity": validate_identity(tokenizer, args.identity),
        "subject_model_forward_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"TOKENIZER_FIXTURES_PASS -> {args.output}")


if __name__ == "__main__":
    main()
