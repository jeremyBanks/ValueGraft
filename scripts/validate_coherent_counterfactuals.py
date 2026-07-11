#!/usr/bin/env python3
"""Fail-closed mechanical validation for banked c10/c02 counterfactuals.

This validator loads only the exact production tokenizer.  It never loads a
model or performs a forward pass.  A MECHANICAL_PASS is deliberately weaker
than review or execution authorization: the banked v1 inputs are UNREVIEWED,
and every emitted attestation records that semantic authorization is false.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any


MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
CANDIDATE_SCHEMA = "coherent_state_counterfactual_candidate_v1"
MANIFEST_SCHEMA = "coherent_state_counterfactual_candidate_manifest_v1"
OUTPUT_SCHEMA = "coherent_state_counterfactual_mechanical_validation_v1"
DEFAULT_INPUT_DIR = "data/coherent_state_counterfactuals/banked_c10_c02"
DEFAULT_OUTPUT_DIR = "results/coherent_state_counterfactual_validation"
VALIDATOR_PATH = "scripts/validate_coherent_counterfactuals.py"
BLIND_REVIEW_NAME = "blind_naturalness_review_gpt56.json"
FACTUAL_REVIEW_NAME = "target_aware_factual_review_gpt56.json"
EXPECTED_COVERAGE = {
    ("c02", "referent"), ("c02", "sense"),
    ("c10", "referent"), ("c10", "sense"),
}
AUTHOR = "Sol"
AUTHOR_MODEL = "gpt-5.6-sol-xhigh"
SOURCE_NOTE_PATH = "notes/2026071183-counterfactual-control-feasibility-audit.md"
MANIFEST_SCOPE = "banked c10/c02 feasibility witnesses only"
MANIFEST_WARNING = (
    "Every candidate is unreviewed, not frozen, not execution-ready, and must "
    "not be used as a semantic source. These artifacts preserve only banked "
    "feasibility witnesses; actual fresh exact-30B render counterfactuals require "
    "separate authoring and review."
)
CANDIDATE_WARNING = (
    "Banked feasibility witness only. Unreviewed, not frozen, not "
    "execution-ready, and not a substitute for an actual fresh exact-30B render "
    "counterfactual."
)
SUMMARY_REQUEST = (
    "Please write a thorough context note summarizing our conversation so far, "
    "for someone who will continue this conversation without seeing it. Cover: "
    "decisions made and what was chosen over what; open threads and next steps; "
    "definitions, names, and terms we introduced and what they mean; constraints "
    "and preferences either of us stated; approaches or options we tried and "
    "ruled out, and why. Be redundant and specific; use retrieval-friendly "
    "wording. Write it as flowing prose or bullet points, roughly 300-500 words. "
    "Do not add commentary before or after the note itself."
)

MANIFEST_KEYS = {
    "schema", "status", "execution_ready", "scope", "authoring_provenance",
    "warning", "expected_candidate_count", "candidate_count",
    "all_candidates_unreviewed", "all_candidates_execution_ready_false",
    "candidates", "review_gate",
}
CANDIDATE_KEYS = {
    "schema", "status", "execution_ready", "candidate_id",
    "authoring_provenance", "base_binding", "plant_target_binding",
    "changed_message_allowlist", "unchanged_message_indices",
    "counterfactual_conversation",
    "counterfactual_conversation_canonical_sha256", "tokenizer_binding",
    "observed_banked_mechanical_validation", "review",
}
MANIFEST_ROW_KEYS = {
    "candidate_id", "path", "raw_file_sha256", "status", "execution_ready",
    "base_path", "base_raw_file_sha256", "plant_id",
    "target_row_canonical_sha256", "changed_message_indices",
    "correct_prefix_token_count", "counterfactual_prefix_token_count",
    "message_start_positions_equal", "turn_aligned_replay_call_widths",
    "ordinary_4096_call_widths", "review_overall_status",
}
BASE_BINDING_KEYS = {
    "path", "raw_file_sha256", "conversation_id",
    "complete_base_conversation_canonical_sha256", "middle_end_msg",
}
TARGET_BINDING_KEYS = {
    "plant_id", "category", "scenario_path", "scenario_raw_file_sha256",
    "scenario_plant", "scenario_plant_canonical_sha256",
    "banked_conversation_plant",
    "banked_conversation_plant_canonical_sha256", "target_path",
    "target_raw_file_sha256", "target_row", "target_row_canonical_sha256",
}
CHANGE_KEYS = {
    "message_index", "role", "base_content_sha256",
    "counterfactual_content_sha256", "base_content_token_count",
    "counterfactual_content_token_count",
}
TOKENIZER_BINDING_KEYS = {
    "model_id", "revision", "tokenizer_class_observed",
    "summary_request_sha256", "chat_template_sha256",
}
MECHANICAL_KEYS = {
    "status", "correct_prefix_token_count",
    "counterfactual_prefix_token_count", "prefix_token_counts_equal",
    "correct_prefix_token_ids_sha256",
    "counterfactual_prefix_token_ids_sha256",
    "differing_prefix_token_position_count",
    "correct_message_start_positions",
    "counterfactual_message_start_positions", "message_start_positions_equal",
    "turn_aligned_replay_call_widths", "ordinary_4096_call_widths",
    "summary_logical_start", "changed_messages_strictly_evicted",
    "all_changed_message_content_token_counts_equal",
}
REVIEW_KEYS = {
    "overall_status", "factual_counterfactual_audit",
    "blind_naturalness_coherence_review", "adjudication",
}
SOURCE_PROVENANCE_KEYS = {
    "author", "model_id", "source_note_path", "source_note_sha256",
}
CANDIDATE_PROVENANCE_KEYS = SOURCE_PROVENANCE_KEYS | {"warning"}
SAFE_RESULT_RE = re.compile(
    r"^coherent_counterfactual_validation_"
    r"Qwen3-30B-A3B-Instruct-2507_\d{8}T\d{12}Z\.json$")


class CounterfactualValidationError(RuntimeError):
    """The mechanical counterfactual contract was not satisfied."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CounterfactualValidationError(
            f"value is not canonical JSON: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _raw_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _text_sha256(text: str) -> str:
    if not isinstance(text, str):
        raise CounterfactualValidationError("content is not text")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_ints(values: list[int]) -> str:
    if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values):
        raise CounterfactualValidationError("token evidence is not integer IDs")
    return _canonical_sha256(values)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CounterfactualValidationError(message)


def _require_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} is not an object")
    _require(set(value) == expected, f"{label} fields differ")
    return value


def _safe_relative(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label} is not a path")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and ".." not in path.parts,
             f"{label} is not repo-relative")
    _require(str(path) == value, f"{label} is not normalized")
    return value


def _git(repo: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", *args], cwd=repo, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise CounterfactualValidationError(
            f"git {' '.join(args)} failed: "
            f"{proc.stderr.decode(errors='replace').strip()[:300]}")
    return proc.stdout


def _committed_bytes(repo: Path, relative: str) -> tuple[bytes, str]:
    relative = _safe_relative(relative, "committed path")
    disk_path = repo / relative
    try:
        disk = disk_path.read_bytes()
    except FileNotFoundError as exc:
        raise CounterfactualValidationError(
            f"required file is absent: {relative}") from exc
    committed = _git(repo, "show", f"HEAD:{relative}")
    _require(disk == committed,
             f"working bytes differ from committed bytes: {relative}")
    return disk, _raw_sha256(disk)


def _committed_json(repo: Path, relative: str) \
        -> tuple[dict[str, Any], bytes, str]:
    raw, digest = _committed_bytes(repo, relative)
    try:
        doc = json.loads(raw)
    except Exception as exc:
        raise CounterfactualValidationError(
            f"committed JSON is invalid: {relative}") from exc
    _require(isinstance(doc, dict), f"committed JSON is not an object: {relative}")
    return doc, raw, digest


def _token_ids(value: Any, label: str) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value.get("input_ids")
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    _require(isinstance(value, list), f"{label} did not produce a token array")
    _require(all(isinstance(item, int) and not isinstance(item, bool)
                 for item in value), f"{label} did not produce integer IDs")
    return [int(item) for item in value]


def _load_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    except Exception as exc:
        raise CounterfactualValidationError(
            f"exact production tokenizer is unavailable locally: {exc}") from exc


def _content_ids(tokenizer: Any, text: str) -> list[int]:
    return _token_ids(tokenizer.encode(
        text, add_special_tokens=False), "message content tokenization")


def _require_exact_roundtrip(tokenizer: Any, text: str, label: str) -> list[int]:
    ids = _content_ids(tokenizer, text)
    try:
        decoded = tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        decoded = tokenizer.decode(ids)
    _require(decoded == text, f"{label} decoded round-trip differs")
    return ids


def _generation_prefix_ids(tokenizer: Any, messages: list[dict[str, Any]]) \
        -> list[int]:
    _require(messages and messages[-1].get("role") == "user",
             "generation prefix does not end in a user request")
    canonical = _canonical_ids(tokenizer, messages)
    generated = _token_ids(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True),
        "generation prefix")
    _require(generated[:len(canonical)] == canonical and
             len(generated) > len(canonical),
             "generation prompt is not a strict canonical extension")
    return generated


def _canonical_ids(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    """Independent Qwen dummy-user canonical non-final rendering."""
    with_dummy = _token_ids(tokenizer.apply_chat_template(
        list(messages) + [{"role": "user", "content": "x"}],
        tokenize=True, add_generation_prompt=False), "canonical rendering")
    marker = _content_ids(tokenizer, "<|im_start|>")
    _require(len(marker) == 1, "im_start is not one exact token")
    starts = [index for index, value in enumerate(with_dummy)
              if value == marker[0]]
    _require(len(starts) == len(messages) + 1,
             "canonical dummy-user message boundaries differ")
    return with_dummy[:starts[len(messages)]]


def _message_starts(tokenizer: Any, ids: list[int], expected: int) -> list[int]:
    marker = _content_ids(tokenizer, "<|im_start|>")
    _require(len(marker) == 1, "im_start is not one exact token")
    starts = [index for index, value in enumerate(ids) if value == marker[0]]
    _require(len(starts) == expected,
             f"canonical message-start count differs: {len(starts)} != {expected}")
    _require(starts and starts[0] == 0, "canonical prefix does not start at zero")
    return starts


def _chunk_widths(width: int) -> list[int]:
    _require(isinstance(width, int) and not isinstance(width, bool) and width > 0,
             "schedule block width is not positive")
    return [min(4096, width - start) for start in range(0, width, 4096)]


def _turn_aligned_widths(starts: list[int], total: int,
                         message_count: int) -> tuple[list[int], list[int]]:
    # Each historical canonical message is a call.  The final summary request
    # and generated assistant header are intentionally one combined call.
    raw = [starts[index + 1] - starts[index]
           for index in range(message_count)]
    raw.append(total - starts[message_count])
    resolved = [piece for width in raw for piece in _chunk_widths(width)]
    return raw, resolved


def _unique_span(container: list[int], needle: list[int], lo: int, hi: int,
                 label: str) -> tuple[int, int]:
    matches = [index for index in range(lo, hi - len(needle) + 1)
               if container[index:index + len(needle)] == needle]
    _require(len(matches) == 1, f"{label} content span is not unique")
    return matches[0], matches[0] + len(needle)


def _unique_row(rows: Any, key: str, value: str, label: str) -> dict[str, Any]:
    _require(isinstance(rows, list), f"{label} rows are not an array")
    matches = [row for row in rows
               if isinstance(row, dict) and row.get(key) == value]
    _require(len(matches) == 1, f"{label} binding is not unique for {value}")
    return matches[0]


def _forbidden_control_strings(tokenizer: Any) -> set[str]:
    values = {str(value) for value in getattr(tokenizer, "all_special_tokens", [])
              if isinstance(value, str) and value}
    values.update({
        "<|im_start|>", "<|im_end|>", "<think>", "</think>",
        "{%", "%}", "{{", "}}",
    })
    return values


def _validate_content_safety(tokenizer: Any, text: str, label: str) -> list[int]:
    ids = _require_exact_roundtrip(tokenizer, text, label)
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    _require(not special_ids.intersection(ids),
             f"{label} contains a special token ID")
    forbidden = _forbidden_control_strings(tokenizer)
    _require(not any(control in text for control in forbidden),
             f"{label} contains a template control string")
    return ids


def _validate_review(review: Any, label: str) -> list[str]:
    review = _require_keys(review, REVIEW_KEYS, f"{label}.review")
    _require(review["overall_status"] == "PENDING",
             f"{label} review status is not PENDING")
    factual = _require_keys(review["factual_counterfactual_audit"], {
        "status", "reviewer", "verdict", "record_path",
    }, f"{label}.factual_review")
    blind = _require_keys(review["blind_naturalness_coherence_review"], {
        "status", "reviewer", "randomization_record", "verdict", "record_path",
    }, f"{label}.blind_review")
    adjudication = _require_keys(review["adjudication"], {
        "status", "reviewer", "verdict", "record_path",
    }, f"{label}.adjudication")
    _require(factual == {"status": "PENDING", "reviewer": None,
                         "verdict": None, "record_path": None},
             f"{label} factual review is not explicitly missing")
    _require(blind == {"status": "PENDING", "reviewer": None,
                       "randomization_record": None, "verdict": None,
                       "record_path": None},
             f"{label} blind review is not explicitly missing")
    _require(adjudication == {"status": "NOT_STARTED", "reviewer": None,
                              "verdict": None, "record_path": None},
             f"{label} adjudication state differs")
    return ["target-aware factual counterfactual audit",
            "blind randomized naturalness/coherence review"]


def _validate_candidate_doc(
    candidate: dict[str, Any], *, candidate_path: str, candidate_raw_sha256: str,
    base: dict[str, Any], base_path: str, base_raw_sha256: str,
    scenarios: list[dict[str, Any]], scenario_path: str,
    scenario_raw_sha256: str, targets: dict[str, Any], targets_path: str,
    targets_raw_sha256: str, tokenizer: Any,
    expected_provenance: dict[str, Any],
) -> dict[str, Any]:
    label = candidate.get("candidate_id", candidate_path)
    _require_keys(candidate, CANDIDATE_KEYS, label)
    _require(candidate["schema"] == CANDIDATE_SCHEMA,
             f"{label} candidate schema differs")
    _require(candidate["status"] == "UNREVIEWED",
             f"{label} must remain UNREVIEWED")
    _require(candidate["execution_ready"] is False,
             f"{label} must not be execution-ready")
    _require(isinstance(candidate["candidate_id"], str) and
             candidate["candidate_id"].startswith("banked-"),
             f"{label} candidate_id is malformed")

    provenance = _require_keys(
        candidate["authoring_provenance"], CANDIDATE_PROVENANCE_KEYS,
        f"{label}.authoring_provenance")
    _safe_relative(provenance["source_note_path"],
                   f"{label}.source_note_path")
    _require(provenance["warning"] == CANDIDATE_WARNING,
             f"{label} authoring warning differs")
    provenance_without_warning = dict(provenance)
    provenance_without_warning.pop("warning")
    _require(provenance_without_warning == expected_provenance,
             f"{label} authoring provenance differs")

    binding = _require_keys(
        candidate["base_binding"], BASE_BINDING_KEYS, f"{label}.base_binding")
    _require(binding["path"] == base_path, f"{label} base path differs")
    _require(binding["raw_file_sha256"] == base_raw_sha256,
             f"{label} base raw hash differs")
    _require(binding["complete_base_conversation_canonical_sha256"] ==
             _canonical_sha256(base), f"{label} base canonical hash differs")
    _require(binding["conversation_id"] == base.get("id"),
             f"{label} base conversation id differs")
    sections = base.get("sections")
    _require(isinstance(sections, dict) and
             binding["middle_end_msg"] == sections.get("middle_end_msg"),
             f"{label} middle_end_msg differs")
    middle_end = binding["middle_end_msg"]

    target_binding = _require_keys(
        candidate["plant_target_binding"], TARGET_BINDING_KEYS,
        f"{label}.plant_target_binding")
    plant_id = target_binding["plant_id"]
    category = target_binding["category"]
    _require(candidate["candidate_id"] == f"banked-{plant_id}",
             f"{label} candidate/plant binding differs")
    _require(target_binding["scenario_path"] == scenario_path,
             f"{label} scenario path differs")
    _require(target_binding["target_path"] == targets_path,
             f"{label} target path differs")
    _require(target_binding["scenario_raw_file_sha256"] == scenario_raw_sha256,
             f"{label} scenario raw hash differs")
    _require(target_binding["target_raw_file_sha256"] == targets_raw_sha256,
             f"{label} target raw hash differs")
    scenario_case = _unique_row(
        scenarios, "id", binding["conversation_id"], "scenario case")
    scenario_plant = _unique_row(
        scenario_case.get("plants"), "id", plant_id, "scenario plant")
    base_plant = _unique_row(base.get("plants"), "id", plant_id, "base plant")
    target_row = _unique_row(targets.get("targets"), "plant_id", plant_id,
                             "target row")
    _require(target_binding["scenario_plant"] == scenario_plant,
             f"{label} embedded scenario plant differs")
    _require(target_binding["banked_conversation_plant"] == base_plant,
             f"{label} embedded base plant differs")
    _require(target_binding["target_row"] == target_row,
             f"{label} embedded target row differs")
    _require(target_binding["scenario_plant_canonical_sha256"] ==
             _canonical_sha256(scenario_plant),
             f"{label} scenario plant hash differs")
    _require(target_binding["banked_conversation_plant_canonical_sha256"] ==
             _canonical_sha256(base_plant),
             f"{label} base plant hash differs")
    _require(target_binding["target_row_canonical_sha256"] ==
             _canonical_sha256(target_row), f"{label} target row hash differs")
    _require(category == scenario_plant.get("category") ==
             base_plant.get("category"), f"{label} plant category differs")
    _require(target_row.get("correct") and target_row.get("counterfactual") and
             target_row["correct"] != target_row["counterfactual"],
             f"{label} target alternatives are malformed")

    cf = candidate["counterfactual_conversation"]
    _require(isinstance(cf, dict), f"{label} counterfactual is not an object")
    _require(set(cf) == set(base), f"{label} conversation fields differ")
    for key in base:
        if key != "messages":
            _require(_canonical_bytes(cf[key]) == _canonical_bytes(base[key]),
                     f"{label} unchanged conversation field differs: {key}")
    base_messages = base.get("messages")
    cf_messages = cf.get("messages")
    _require(isinstance(base_messages, list) and isinstance(cf_messages, list) and
             len(base_messages) == len(cf_messages) and base_messages,
             f"{label} message count differs")
    _require(candidate["counterfactual_conversation_canonical_sha256"] ==
             _canonical_sha256(cf),
             f"{label} counterfactual conversation hash differs")

    allowlist = candidate["changed_message_allowlist"]
    _require(isinstance(allowlist, list) and allowlist,
             f"{label} changed-message allowlist is empty")
    changed_indices: list[int] = []
    for row in allowlist:
        _require_keys(row, CHANGE_KEYS, f"{label}.changed_message")
        index = row["message_index"]
        _require(isinstance(index, int) and not isinstance(index, bool) and
                 0 < index < middle_end and index < len(base_messages),
                 f"{label} changed message is not strictly evicted")
        _require(index not in changed_indices,
                 f"{label} changed-message index is duplicated")
        changed_indices.append(index)
        before, after = base_messages[index], cf_messages[index]
        _require(isinstance(before, dict) and isinstance(after, dict) and
                 set(before) == set(after) and set(before) == {"role", "content"},
                 f"{label} changed message shape differs")
        _require(before["role"] == after["role"] == row["role"],
                 f"{label} changed message role differs")
        _require(before["content"] != after["content"],
                 f"{label} allowlisted message did not change")
        _require(row["base_content_sha256"] == _text_sha256(before["content"]),
                 f"{label} base content hash differs")
        _require(row["counterfactual_content_sha256"] ==
                 _text_sha256(after["content"]),
                 f"{label} counterfactual content hash differs")

    changed_indices.sort()
    actual_changed = [index for index, (before, after) in enumerate(
        zip(base_messages, cf_messages))
        if _canonical_bytes(before) != _canonical_bytes(after)]
    _require(actual_changed == changed_indices,
             f"{label} actual changes differ from allowlist")
    unchanged = candidate["unchanged_message_indices"]
    expected_unchanged = [index for index in range(len(base_messages))
                          if index not in set(changed_indices)]
    _require(unchanged == expected_unchanged,
             f"{label} unchanged-message coverage differs")
    for index in expected_unchanged:
        _require(_canonical_bytes(base_messages[index]) ==
                 _canonical_bytes(cf_messages[index]),
                 f"{label} unchanged message {index} differs")
    _require(_canonical_bytes(base_messages[0]) == _canonical_bytes(cf_messages[0]),
             f"{label} system message differs")
    for index in range(middle_end, len(base_messages)):
        _require(_canonical_bytes(base_messages[index]) ==
                 _canonical_bytes(cf_messages[index]),
                 f"{label} retained-tail message {index} differs")

    tokenizer_binding = _require_keys(
        candidate["tokenizer_binding"], TOKENIZER_BINDING_KEYS,
        f"{label}.tokenizer_binding")
    expected_tokenizer = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "tokenizer_class_observed": type(tokenizer).__name__,
        "summary_request_sha256": _text_sha256(SUMMARY_REQUEST),
        "chat_template_sha256": _text_sha256(str(tokenizer.chat_template)),
    }
    _require(tokenizer_binding == expected_tokenizer,
             f"{label} tokenizer binding differs")

    special_ids = {int(value) for value in tokenizer.all_special_ids}
    content_evidence: dict[int, tuple[list[int], list[int]]] = {}
    for index, message in enumerate(cf_messages):
        _validate_content_safety(
            tokenizer, message["content"], f"{label}.message[{index}]")
    for row in allowlist:
        index = row["message_index"]
        before_ids = _require_exact_roundtrip(
            tokenizer, base_messages[index]["content"],
            f"{label}.base_message[{index}]")
        after_ids = _content_ids(tokenizer, cf_messages[index]["content"])
        _require(row["base_content_token_count"] == len(before_ids) and
                 row["counterfactual_content_token_count"] == len(after_ids),
                 f"{label} changed content token count differs")
        _require(len(before_ids) == len(after_ids),
                 f"{label} changed content widths differ")
        content_evidence[index] = (before_ids, after_ids)

    source_correct = list(base_messages) + [
        {"role": "user", "content": SUMMARY_REQUEST}]
    source_cf = list(cf_messages) + [
        {"role": "user", "content": SUMMARY_REQUEST}]
    correct_ids = _generation_prefix_ids(tokenizer, source_correct)
    cf_ids = _generation_prefix_ids(tokenizer, source_cf)
    _require(len(correct_ids) == len(cf_ids),
             f"{label} full canonical prefix lengths differ")
    expected_starts = len(base_messages) + 2
    correct_starts = _message_starts(tokenizer, correct_ids, expected_starts)
    cf_starts = _message_starts(tokenizer, cf_ids, expected_starts)
    _require(correct_starts == cf_starts,
             f"{label} canonical message starts differ")
    raw_widths, p_widths = _turn_aligned_widths(
        correct_starts, len(correct_ids), len(base_messages))
    cf_raw_widths, cf_p_widths = _turn_aligned_widths(
        cf_starts, len(cf_ids), len(cf_messages))
    _require(raw_widths == cf_raw_widths and p_widths == cf_p_widths,
             f"{label} per-message canonical widths differ")
    ordinary = _chunk_widths(len(correct_ids))

    differing = [index for index, (left, right) in enumerate(
                 zip(correct_ids, cf_ids)) if left != right]
    _require(differing, f"{label} full prefixes are identical")
    covered_positions: set[int] = set()
    per_message_differences: dict[str, int] = {}
    for index in changed_indices:
        before_ids, after_ids = content_evidence[index]
        before_span = _unique_span(
            correct_ids, before_ids, correct_starts[index],
            correct_starts[index + 1], f"{label}.base_message[{index}]")
        after_span = _unique_span(
            cf_ids, after_ids, cf_starts[index], cf_starts[index + 1],
            f"{label}.counterfactual_message[{index}]")
        _require(before_span == after_span,
                 f"{label} changed content span moved")
        positions = set(range(*before_span))
        local = positions.intersection(differing)
        _require(local, f"{label} changed message {index} changed no token")
        covered_positions.update(positions)
        per_message_differences[str(index)] = len(local)
    _require(set(differing).issubset(covered_positions),
             f"{label} changed token escaped approved content spans")
    _require(not any(correct_ids[index] in special_ids or cf_ids[index] in special_ids
                     for index in differing),
             f"{label} changed a chat-structural token")

    observed = _require_keys(
        candidate["observed_banked_mechanical_validation"], MECHANICAL_KEYS,
        f"{label}.observed_mechanical_validation")
    expected_observed = {
        "status": "OBSERVED_FOR_BANKED_FEASIBILITY_ONLY",
        "correct_prefix_token_count": len(correct_ids),
        "counterfactual_prefix_token_count": len(cf_ids),
        "prefix_token_counts_equal": True,
        "correct_prefix_token_ids_sha256": _sha256_ints(correct_ids),
        "counterfactual_prefix_token_ids_sha256": _sha256_ints(cf_ids),
        "differing_prefix_token_position_count": len(differing),
        "correct_message_start_positions": correct_starts,
        "counterfactual_message_start_positions": cf_starts,
        "message_start_positions_equal": True,
        "turn_aligned_replay_call_widths": p_widths,
        "ordinary_4096_call_widths": ordinary,
        "summary_logical_start": len(correct_ids),
        "changed_messages_strictly_evicted": True,
        "all_changed_message_content_token_counts_equal": True,
    }
    _require(observed == expected_observed,
             f"{label} stored mechanical evidence differs")
    missing_reviews = _validate_review(candidate["review"], label)

    return {
        "candidate_id": candidate["candidate_id"],
        "path": candidate_path,
        "raw_file_sha256": candidate_raw_sha256,
        "conversation_id": binding["conversation_id"],
        "plant_id": plant_id,
        "category": category,
        "changed_message_indices": changed_indices,
        "unchanged_message_count": len(expected_unchanged),
        "middle_end_msg": middle_end,
        "correct_prefix_token_count": len(correct_ids),
        "counterfactual_prefix_token_count": len(cf_ids),
        "correct_prefix_token_ids_sha256": _sha256_ints(correct_ids),
        "counterfactual_prefix_token_ids_sha256": _sha256_ints(cf_ids),
        "message_start_positions_sha256": _sha256_ints(correct_starts),
        "per_message_canonical_widths": raw_widths,
        "turn_aligned_replay_call_widths": p_widths,
        "ordinary_4096_call_widths": ordinary,
        "summary_logical_start": len(correct_ids),
        "differing_prefix_token_position_count": len(differing),
        "differing_prefix_token_positions_sha256": _sha256_ints(differing),
        "per_changed_message_difference_counts": per_message_differences,
        "exact_decoded_roundtrip": True,
        "system_byte_identical": True,
        "retained_tail_byte_identical": True,
        "summary_request_and_header_identical": True,
        "mechanical_status": "MECHANICAL_PASS",
        "review_status": "PENDING",
        "missing_review_attestations": missing_reviews,
        "semantic_authorized": False,
        "execution_authorized": False,
    }


def _expected_manifest_row(candidate: dict[str, Any], evidence: dict[str, Any],
                           raw_sha256: str) -> dict[str, Any]:
    binding = candidate["base_binding"]
    target = candidate["plant_target_binding"]
    observed = candidate["observed_banked_mechanical_validation"]
    return {
        "candidate_id": candidate["candidate_id"],
        "path": evidence["path"],
        "raw_file_sha256": raw_sha256,
        "status": "UNREVIEWED",
        "execution_ready": False,
        "base_path": binding["path"],
        "base_raw_file_sha256": binding["raw_file_sha256"],
        "plant_id": target["plant_id"],
        "target_row_canonical_sha256": target["target_row_canonical_sha256"],
        "changed_message_indices": evidence["changed_message_indices"],
        "correct_prefix_token_count": evidence["correct_prefix_token_count"],
        "counterfactual_prefix_token_count":
            evidence["counterfactual_prefix_token_count"],
        "message_start_positions_equal": True,
        "turn_aligned_replay_call_widths":
            observed["turn_aligned_replay_call_widths"],
        "ordinary_4096_call_widths": observed["ordinary_4096_call_widths"],
        "review_overall_status": "PENDING",
    }


def _validate_manifest_header(manifest: Any, note_sha256: str) -> dict[str, Any]:
    manifest = _require_keys(manifest, MANIFEST_KEYS, "candidate manifest")
    _require(manifest["schema"] == MANIFEST_SCHEMA,
             "candidate manifest schema differs")
    _require(manifest["status"] == "UNREVIEWED" and
             manifest["execution_ready"] is False,
             "candidate manifest is not explicitly non-authorizing")
    _require(manifest["expected_candidate_count"] == len(EXPECTED_COVERAGE) and
             manifest["candidate_count"] == len(EXPECTED_COVERAGE),
             "candidate manifest count differs")
    _require(manifest["all_candidates_unreviewed"] is True and
             manifest["all_candidates_execution_ready_false"] is True,
             "candidate manifest aggregate status differs")
    _require(manifest["scope"] == MANIFEST_SCOPE,
             "candidate manifest scope differs")
    _require(manifest["warning"] == MANIFEST_WARNING,
             "candidate manifest warning differs")
    provenance = _require_keys(
        manifest["authoring_provenance"], SOURCE_PROVENANCE_KEYS,
        "manifest authoring provenance")
    _require(provenance["author"] == AUTHOR and
             provenance["model_id"] == AUTHOR_MODEL and
             provenance["source_note_path"] == SOURCE_NOTE_PATH,
             "manifest authoring provenance differs")
    _require(provenance["source_note_sha256"] == note_sha256,
             "manifest source-note hash differs")
    review_gate = _require_keys(
        manifest["review_gate"], {"status", "requirements"},
        "manifest review gate")
    _require(review_gate == {
        "status": "PENDING",
        "requirements": [
            "target-aware factual counterfactual audit",
            "blind randomized naturalness/coherence review",
            "adjudication of any disagreement",
            "fresh exact-30B render-specific reconstruction before execution",
        ],
    }, "manifest review gate differs")
    return provenance


def _validate_manifest_row(row: Any, expected: dict[str, Any]) -> None:
    _require_keys(row, MANIFEST_ROW_KEYS, "manifest candidate row")
    _require(row == expected,
             f"manifest row differs for {expected.get('path', 'candidate')}")


def _validate_manifest_row_inventory(
    rows: Any, input_dir: str, actual_paths: list[str],
) -> tuple[list[str], list[str], list[str]]:
    _require(isinstance(rows, list) and len(rows) == len(EXPECTED_COVERAGE),
             "manifest candidate rows differ")
    paths: list[str] = []
    ids: list[str] = []
    plants: list[str] = []
    for row in rows:
        _require_keys(row, MANIFEST_ROW_KEYS, "manifest candidate row")
        path = _safe_relative(row["path"], "manifest candidate path")
        _require(PurePosixPath(path).parent == PurePosixPath(input_dir) and
                 path.endswith("_unreviewed.json"),
                 "candidate path escapes the banked input directory")
        paths.append(path)
        ids.append(row["candidate_id"])
        plants.append(row["plant_id"])
    _require(len(paths) == len(set(paths)) and len(ids) == len(set(ids)) and
             len(plants) == len(set(plants)),
             "manifest candidate coverage is not unique")
    _require(sorted(paths) == sorted(actual_paths),
             "manifest/file candidate coverage differs")
    return paths, ids, plants


def _validate_external_reviews(
    repo: Path, input_dir: str,
    candidate_bindings: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Bind the independent decoded reviews without rewriting candidate snapshots."""
    review_dir = f"{input_dir}/reviews"
    blind_path = f"{review_dir}/{BLIND_REVIEW_NAME}"
    factual_path = f"{review_dir}/{FACTUAL_REVIEW_NAME}"
    blind, blind_raw, blind_sha = _committed_json(repo, blind_path)
    factual, factual_raw, factual_sha = _committed_json(repo, factual_path)

    _require(blind.get("overall_verdict") == "FAIL",
             "blind review overall verdict is not FAIL")
    blind_candidates = blind.get("candidates")
    _require(isinstance(blind_candidates, dict) and
             len(blind_candidates) == len(candidate_bindings),
             "blind review candidate coverage differs")
    blind_by_path: dict[str, dict[str, Any]] = {}
    for row in blind_candidates.values():
        _require(isinstance(row, dict), "blind review row is not an object")
        path = _safe_relative(row.get("candidate_file"),
                              "blind review candidate path")
        _require(path not in blind_by_path, "blind review candidate path repeats")
        blind_by_path[path] = row
    _require(set(blind_by_path) == set(candidate_bindings),
             "blind review paths differ from candidate inventory")

    _require(factual.get("schema") ==
             "coherent_state_counterfactual_target_aware_factual_review_v1",
             "target-aware review schema differs")
    _require(factual.get("overall_verdict") == "REVISE_ALL_FOUR",
             "target-aware review overall verdict differs")
    factual_candidates = factual.get("candidates")
    _require(isinstance(factual_candidates, dict) and
             set(factual_candidates) ==
             {binding["candidate_id"] for binding in candidate_bindings.values()},
             "target-aware review candidate coverage differs")

    per_candidate = []
    for path in sorted(candidate_bindings):
        binding = candidate_bindings[path]
        blind_row = blind_by_path[path]
        _require(blind_row.get("sha256") == binding["raw_file_sha256"] and
                 blind_row.get("overall_candidate_verdict") == "FAIL",
                 f"blind review binding/verdict differs for {path}")
        factual_row = factual_candidates[binding["candidate_id"]]
        factual_binding = (factual_row.get("bindings")
                           if isinstance(factual_row, dict) else None)
        _require(isinstance(factual_binding, dict) and
                 factual_binding.get("candidate_path") == path and
                 factual_binding.get("candidate_raw_file_sha256") ==
                 binding["raw_file_sha256"] and
                 factual_row.get("verdict") == "REVISE",
                 f"target-aware review binding/verdict differs for {path}")
        per_candidate.append({
            "candidate_id": binding["candidate_id"],
            "path": path,
            "blind_verdict": "FAIL",
            "target_aware_verdict": "REVISE",
            "execution_authorized": False,
        })

    evidence = {
        "status": "FAIL",
        "blind_review": {
            "path": blind_path,
            "raw_file_sha256": blind_sha,
            "overall_verdict": "FAIL",
        },
        "target_aware_factual_review": {
            "path": factual_path,
            "raw_file_sha256": factual_sha,
            "overall_verdict": "REVISE_ALL_FOUR",
        },
        "candidate_verdicts": per_candidate,
        "semantic_authorized": False,
        "execution_authorized": False,
    }
    inventory = [
        {"path": blind_path, "bytes": len(blind_raw),
         "raw_file_sha256": blind_sha},
        {"path": factual_path, "bytes": len(factual_raw),
         "raw_file_sha256": factual_sha},
    ]
    return evidence, inventory


def validate_directory(repo: Path, input_dir: str = DEFAULT_INPUT_DIR,
                       tokenizer: Any | None = None) -> dict[str, Any]:
    repo = repo.resolve()
    input_dir = _safe_relative(input_dir, "input directory")
    head = _git(repo, "rev-parse", "HEAD^{commit}").decode().strip()
    _require(re.fullmatch(r"[0-9a-f]{40}", head) is not None,
             "repository HEAD is malformed")
    validator_raw, validator_sha = _committed_bytes(repo, VALIDATOR_PATH)
    manifest_path = f"{input_dir}/manifest.json"
    manifest, manifest_raw, manifest_sha = _committed_json(repo, manifest_path)
    raw_provenance = manifest.get("authoring_provenance")
    _require(isinstance(raw_provenance, dict),
             "manifest authoring provenance is not an object")
    note_path = _safe_relative(raw_provenance.get("source_note_path"),
                               "source note path")
    note_raw, note_sha = _committed_bytes(repo, note_path)
    manifest_provenance = _validate_manifest_header(manifest, note_sha)

    rows = manifest["candidates"]
    actual_paths = sorted(
        str(path.relative_to(repo).as_posix())
        for path in (repo / input_dir).glob("*_unreviewed.json"))
    _validate_manifest_row_inventory(rows, input_dir, actual_paths)

    tokenizer = tokenizer or _load_tokenizer()
    candidate_results: list[dict[str, Any]] = []
    input_rows = [
        {"path": VALIDATOR_PATH, "bytes": len(validator_raw),
         "raw_file_sha256": validator_sha},
        {"path": note_path, "bytes": len(note_raw),
         "raw_file_sha256": note_sha},
        {"path": manifest_path, "bytes": len(manifest_raw),
         "raw_file_sha256": manifest_sha},
    ]
    candidate_bindings: dict[str, dict[str, str]] = {}
    coverage: set[tuple[str, str]] = set()
    for row in rows:
        candidate_path = row["path"]
        candidate, candidate_raw, candidate_sha = _committed_json(
            repo, candidate_path)
        binding = _require_keys(
            candidate.get("base_binding"), BASE_BINDING_KEYS,
            f"{candidate_path}.base_binding")
        target_binding = _require_keys(
            candidate.get("plant_target_binding"), TARGET_BINDING_KEYS,
            f"{candidate_path}.plant_target_binding")
        base_path = _safe_relative(binding["path"], "base path")
        scenario_path = _safe_relative(
            target_binding["scenario_path"], "scenario path")
        target_path = _safe_relative(target_binding["target_path"], "target path")
        base, base_raw, base_sha = _committed_json(repo, base_path)
        scenario_raw, scenario_sha = _committed_bytes(repo, scenario_path)
        target_doc, target_raw, target_sha = _committed_json(repo, target_path)
        try:
            scenarios = json.loads(scenario_raw)
        except Exception as exc:
            raise CounterfactualValidationError("scenario JSON is invalid") from exc
        _require(isinstance(scenarios, list), "scenario JSON is not an array")
        result = _validate_candidate_doc(
            candidate, candidate_path=candidate_path,
            candidate_raw_sha256=candidate_sha, base=base,
            base_path=base_path, base_raw_sha256=base_sha,
            scenarios=scenarios, scenario_path=scenario_path,
            scenario_raw_sha256=scenario_sha, targets=target_doc,
            targets_path=target_path, targets_raw_sha256=target_sha,
            tokenizer=tokenizer, expected_provenance=manifest_provenance)
        _validate_manifest_row(
            row, _expected_manifest_row(candidate, result, candidate_sha))
        coverage.add((result["conversation_id"], result["category"]))
        candidate_results.append(result)
        candidate_bindings[candidate_path] = {
            "candidate_id": result["candidate_id"],
            "raw_file_sha256": candidate_sha,
        }
        input_rows.extend([
            {"path": candidate_path, "bytes": len(candidate_raw),
             "raw_file_sha256": candidate_sha},
            {"path": base_path, "bytes": len(base_raw),
             "raw_file_sha256": base_sha},
            {"path": scenario_path, "bytes": len(scenario_raw),
             "raw_file_sha256": scenario_sha},
            {"path": target_path, "bytes": len(target_raw),
             "raw_file_sha256": target_sha},
        ])
    _require(coverage == EXPECTED_COVERAGE,
             "candidate case/category coverage differs")
    external_reviews, review_inputs = _validate_external_reviews(
        repo, input_dir, candidate_bindings)
    input_rows.extend(review_inputs)

    unique_inputs = {row["path"]: row for row in input_rows}
    _require(all(len({item["raw_file_sha256"] for item in input_rows
                      if item["path"] == path}) == 1
                 for path in unique_inputs), "input path hashes disagree")
    inventory = [unique_inputs[path] for path in sorted(unique_inputs)]
    candidate_results.sort(key=lambda item: item["candidate_id"])
    return _seal({
        "schema": OUTPUT_SCHEMA,
        "verdict": "MECHANICAL_PASS",
        "mechanical_pass": True,
        "semantic_authorized": False,
        "execution_authorized": False,
        "model_forward_performed": False,
        "embedded_candidate_review_status": "PENDING",
        "external_review_status": "FAIL",
        "missing_review_attestations": [],
        "external_review_evidence": external_reviews,
        "warning": (
            "MECHANICAL_PASS is not semantic authorization. Independent decoded "
            "review failed every candidate; these banked feasibility witnesses "
            "are not execution-ready."
        ),
        "repository_head": head,
        "input_manifest_path": manifest_path,
        "input_manifest_raw_file_sha256": manifest_sha,
        "validator_script_path": VALIDATOR_PATH,
        "validator_script_raw_file_sha256": validator_sha,
        "committed_input_inventory": inventory,
        "committed_input_inventory_sha256": _canonical_sha256(inventory),
        "tokenizer_binding": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "tokenizer_class_observed": type(tokenizer).__name__,
            "chat_template_sha256": _text_sha256(str(tokenizer.chat_template)),
            "summary_request_sha256": _text_sha256(SUMMARY_REQUEST),
        },
        "coverage": [
            {"conversation_id": cid, "category": category}
            for cid, category in sorted(coverage)
        ],
        "candidate_count": len(candidate_results),
        "candidates": candidate_results,
    })


def _payload_sha256(doc: dict[str, Any]) -> str:
    return _canonical_sha256({key: value for key, value in doc.items()
                              if key != "payload_sha256"})


def _seal(doc: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in doc.items()
              if key != "payload_sha256"}
    result["payload_sha256"] = _payload_sha256(result)
    return result


def _require_sealed(doc: dict[str, Any]) -> None:
    _require(doc.get("payload_sha256") == _payload_sha256(doc),
             "output payload seal differs")
    _require(doc.get("schema") == OUTPUT_SCHEMA and
             doc.get("verdict") == "MECHANICAL_PASS" and
             doc.get("mechanical_pass") is True and
             doc.get("semantic_authorized") is False and
             doc.get("execution_authorized") is False and
             doc.get("model_forward_performed") is False,
             "sealed output policy invariants differ")


def _unique_output_path(repo: Path, output_dir: str) -> Path:
    output_dir = _safe_relative(output_dir, "output directory")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = (
        "coherent_counterfactual_validation_"
        f"Qwen3-30B-A3B-Instruct-2507_{stamp}.json")
    _require(SAFE_RESULT_RE.fullmatch(name) is not None,
             "generated output name is malformed")
    return repo / output_dir / name


def _write_sealed_unique(path: Path, doc: dict[str, Any]) -> None:
    _require_sealed(doc)
    raw = json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise CounterfactualValidationError(
            f"refusing to overwrite validation output: {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # Preserve any partial file as failure evidence; never silently reuse it.
        raise
    loaded = json.loads(path.read_text(encoding="utf-8"))
    _require(loaded == doc, "written output decoded round-trip differs")
    _require_sealed(loaded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check-only", action="store_true",
                        help="validate and print the sealed result without writing")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    output = None if args.check_only else _unique_output_path(repo, args.output_dir)
    if output is None:
        print(f"CHECK coherent-counterfactuals model={MODEL_ID}")
    else:
        print(f"RUN coherent-counterfactuals model={MODEL_ID} -> {output}")
    try:
        result = validate_directory(repo, args.input_dir)
        if output is not None:
            relative = output.relative_to(repo).as_posix()
            result = _seal({**result, "output_path": relative})
            _write_sealed_unique(output, result)
            print(f"MECHANICAL_PASS (semantic_authorized=false) {output}")
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except CounterfactualValidationError as exc:
        print(f"MECHANICAL_FAIL: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
