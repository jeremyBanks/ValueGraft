#!/usr/bin/env python3
"""Fail-closed validation before a coherent-state pod may terminate.

This validator intentionally has no imports from ``src``.  A deployment must be
validated against the frozen on-disk schema, not whatever assumptions happen to
be importable from the checkout that performs the harvest.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = 2
AMENDMENT_ID = "COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4-5-6-7"
DESIGN_ID = "coherent-state-gapped-v7"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
PARAMETER_DTYPE = "torch.bfloat16"
ATTENTION_BACKEND = "eager"
MAX_TECHNICAL_LOGICAL_POSITION = 9509
SYNTHETIC_LENGTHS = (5, 64, 900, 4096, 4097, 8193)
SYNTHETIC_PARTITIONS = (
    ((5,), (2, 3)),
    ((64,), (32, 32)),
    ((900,), (32, 868)),
    ((4096,), (32, 4064)),
    ((4096, 1), (32, 4065)),
    ((4096, 4096, 1), (32, 4096, 4065)),
)
FROZEN_FIXTURE_LITERAL = "alpha beta gamma delta epsilon"
FROZEN_FIXTURE_POOL = [7141, 13440, 21619, 9477, 31204]
FROZEN_MARGIN_IDS = [362, 425]
FROZEN_CONTINUATION_ID = 7141
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
CASE_CONTINUATION_POSITIONS = {
    "c10": 8430, "c02": 8385, "c01": 8855, "c04": 8595,
    "c07": 8600, "c11": 9381, "c05": 8556, "c09": 9195,
    "c06": 8876, "c12": 9509, "c08": 8525, "c03": 8913,
}
ARMS = (
    "A_full",
    "G_fresh",
    "G_correct",
    "G_wrong",
    "G_Vcorrect",
    "G_Kcorrect",
)
FROZEN_ORDER = (
    "c10", "c02", "c01", "c04", "c07", "c11",
    "c05", "c09", "c06", "c12", "c08", "c03",
)
WRONG_DONORS = {
    "c10": "c13", "c02": "c14", "c01": "c15",
    "c04": "c16", "c07": "c17", "c11": "c18",
    "c05": "c25", "c09": "c26", "c06": "c27",
    "c12": "c28", "c08": "c29", "c03": "c30",
}
EXTERNAL_AUTHORS = {"sonnet", "opus", "codex-gpt5.5", "sonnet-render"}
AMENDMENT_PATHS = tuple(
    f"COHERENT-STATE-PREREGISTRATION-AMENDMENT-{index}.md"
    for index in range(1, 8))
APPARATUS_REQUIRED = (
    *AMENDMENT_PATHS,
    "src/analyze_coherent_state.py", "src/arms_common.py",
    "src/coherent_state_calibration.py", "src/coherent_state_cases.py",
    "src/coherent_state_hf.py", "src/coherent_state_integrity.py",
    "src/coherent_state_runtime.py", "src/coherent_state_store.py",
    "src/coherent_state_tokens.py", "src/cross_arch_probe.py",
    "src/l_coherent_state_hf.py", "src/kvlib_hf.py", "src/pod.py",
    "src/run_coherent_state_hf.py", "src/validate_coherent_external_donors.py",
    "scripts/classify_pod.sh", "scripts/coherent_lifecycle_lib.sh",
    "scripts/coherent_monitor_selftest.sh", "scripts/job_coherent_state_bf16.sh",
    "scripts/job_coherent_state_semantic_bf16.sh", "scripts/launch_pod.sh",
    "scripts/preflight.py", "scripts/preflight.sh",
    "scripts/validate_coherent_harvest.py", "scripts/watch_coherent_state_pod.sh",
)
APPARATUS_GLOBS = (
    "src/coherent_state_*.py", "scripts/*coherent*.sh",
    "scripts/*coherent*.py",
)
OLD_ARMS = {
    "F_fresh", "C_coherent", "W_wrong", "V_value", "K_key", "D_delta",
    "G_delta",
}
FAILURE_RE = re.compile(
    r"(?:FATAL|Traceback \(most recent call last\)|CUDA out of memory|"
    r"OutOfMemoryError|RuntimeError|CoherentStateError|WATCH_FAILURE)",
    re.IGNORECASE,
)


def _load(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"required artifact absent: {path.name}") from exc
    except Exception as exc:
        raise ValueError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return doc


def _canonical_payload_sha256(doc: dict[str, Any]) -> str:
    payload = {key: value for key, value in doc.items()
               if key != "payload_sha256"}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def _require_payload_sha256(doc: dict[str, Any], label: str) -> None:
    if doc.get("payload_sha256") != _canonical_payload_sha256(doc):
        raise ValueError(f"{label} payload_sha256 mismatch")


def _raw_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def _git_bytes(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"cannot reconstruct launch commit with git {' '.join(args)}: "
            f"{exc.stderr.decode(errors='replace').strip()}") from exc


def _launch_blob(repo: Path, commit: str, relative: str) -> bytes:
    return _git_bytes(repo, "show", f"{commit}:{relative}")


def _expected_apparatus_inventory(repo: Path, commit: str) -> dict[str, Any]:
    tree = _git_bytes(repo, "ls-tree", "-r", "--name-only", commit).decode().splitlines()
    selected = set(APPARATUS_REQUIRED)
    selected.update(path for path in tree
                    if any(fnmatch.fnmatch(path, pattern)
                           for pattern in APPARATUS_GLOBS))
    missing = sorted(set(APPARATUS_REQUIRED) - set(tree))
    if missing:
        raise ValueError(f"launch commit lacks required apparatus files {missing}")
    rows = []
    for relative in sorted(selected):
        raw = _launch_blob(repo, commit, relative)
        rows.append({
            "path": relative, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return {
        "files": rows, "file_count": len(rows),
        "aggregate_sha256": _canonical_json_sha256(rows),
    }


def _expected_input_inventory(repo: Path, commit: str) -> dict[str, Any]:
    paths = {"data/scenarios.json", "data/coherent_state_targets.json"}
    paths.update(f"data/synthetic/{cid}.json" for cid in
                 set(FROZEN_ORDER).union(WRONG_DONORS.values()))
    rows = []
    for relative in sorted(paths):
        raw = _launch_blob(repo, commit, relative)
        rows.append({
            "path": relative, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    donor_provenance = {}
    for donor_id in WRONG_DONORS.values():
        relative = f"data/synthetic/{donor_id}.json"
        raw = _launch_blob(repo, commit, relative)
        parsed = json.loads(raw)
        author = (parsed.get("meta") or {}).get("author")
        if author not in EXTERNAL_AUTHORS:
            raise ValueError(f"launch donor {donor_id} author differs")
        donor_provenance[donor_id] = {
            "donor_id": donor_id, "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "recorded_author": author, "subject_native": False,
        }
    return {
        "files": rows,
        "aggregate_sha256": _canonical_json_sha256(rows),
        "external_donor_provenance": donor_provenance,
    }


def _expected_static_subject_metadata() -> dict[str, Any]:
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION,
            attn_implementation=ATTENTION_BACKEND, local_files_only=True)
    except Exception as exc:
        raise ValueError(f"exact production config unavailable to harvest: {exc}") from exc
    tokenizer = _validation_tokenizer()
    resolved = getattr(config, "_commit_hash", None)
    return {
        "resolved_model_revision": resolved,
        "attention_backend_requested": ATTENTION_BACKEND,
        "config_sha256": _canonical_json_sha256(config.to_dict()),
        "tokenizer_revision_requested": MODEL_REVISION,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_sha256": _canonical_json_sha256(tokenizer.get_vocab()),
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode()).hexdigest(),
        "special_tokens_map_sha256": _canonical_json_sha256(
            tokenizer.special_tokens_map),
    }


def _validate_static_fingerprint(
        static: Any, fingerprint: Any, repo: Path) -> None:
    if not isinstance(static, dict) or not isinstance(fingerprint, dict):
        raise ValueError("technical static/final fingerprint is malformed")
    commit = static.get("code_commit")
    if (not isinstance(commit, str) or len(commit) != 40 or
            any(char not in "0123456789abcdef" for char in commit.lower())):
        raise ValueError("technical launch commit is malformed")
    resolved = _git_bytes(repo, "rev-parse", f"{commit}^{{commit}}").decode().strip()
    if resolved != commit:
        raise ValueError("technical launch commit did not resolve exactly")
    expected_apparatus = _expected_apparatus_inventory(repo, commit)
    expected_inputs = _expected_input_inventory(repo, commit)
    expected_subject = _expected_static_subject_metadata()
    amendment_hashes = {
        relative: hashlib.sha256(
            _launch_blob(repo, commit, relative)).hexdigest()
        for relative in AMENDMENT_PATHS
    }
    exact_fields = {
        "schema": SCHEMA, "design_id": DESIGN_ID,
        "amendment_id": AMENDMENT_ID,
        "amendment_sha256": amendment_hashes,
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "dtype": PARAMETER_DTYPE, "code_commit": commit,
        "apparatus_inventory": expected_apparatus,
        "input_inventory": expected_inputs,
        "scenario_sha256": hashlib.sha256(_launch_blob(
            repo, commit, "data/scenarios.json")).hexdigest(),
        "targets_sha256": hashlib.sha256(_launch_blob(
            repo, commit, "data/coherent_state_targets.json")).hexdigest(),
        "summary_request_sha256": hashlib.sha256(
            SUMMARY_REQUEST.encode()).hexdigest(),
        "frozen_order": list(FROZEN_ORDER), "wrong_donors": WRONG_DONORS,
        "structural_seed": 20_260_711, "max_reply_tokens": 320,
        "max_summary_tokens": 900, "identity_tolerance": 1e-4,
        "zero_gap_tolerance": 5e-4, "attention_backend": ATTENTION_BACKEND,
        "max_technical_logical_position": MAX_TECHNICAL_LOGICAL_POSITION,
        "arms": list(ARMS), "subject_metadata": expected_subject,
    }
    runtime = static.get("runtime_environment")
    if not isinstance(runtime, dict):
        raise ValueError("technical runtime environment is malformed")
    packages = runtime.get("execution_packages")
    expected_packages = {
        "accelerate": "1.14.0", "huggingface_hub": "1.22.0",
        "safetensors": "0.8.0", "sentencepiece": "0.2.1",
        "torch": "2.4.1", "transformers": "5.0.0",
    }
    if (not str(runtime.get("python", "")).startswith("3.11") or
            "linux" not in str(runtime.get("platform", "")).lower() or
            runtime.get("torch") != "2.4.1+cu124" or
            runtime.get("transformers") != "5.0.0" or
            runtime.get("cuda") != "12.4" or
            runtime.get("gpu") != "NVIDIA A100 80GB PCIe" or
            packages != expected_packages):
        raise ValueError("technical runtime environment differs from frozen image")
    exact_fields["runtime_environment"] = runtime
    if set(static) != set(exact_fields) or any(
            static.get(key) != value for key, value in exact_fields.items()):
        raise ValueError("technical static fingerprint reconstruction differs")

    expected_final = dict(static)
    expected_final["subject_metadata"] = {
        **expected_subject,
        "attention_backend_resolved": ATTENTION_BACKEND,
        "attention_backend_fingerprint": fingerprint.get(
            "attention_backend_fingerprint"),
        "context_limit": fingerprint.get("context_limit"),
    }
    expected_final["attention_backend_fingerprint"] = fingerprint.get(
        "attention_backend_fingerprint")
    expected_final["context_limit"] = fingerprint.get("context_limit")
    if fingerprint != expected_final:
        raise ValueError("technical final/static fingerprint binding differs")


def _chunk_widths(width: int) -> list[int]:
    if not isinstance(width, int) or width < 1:
        raise ValueError("chunk width must be a positive integer")
    return [min(4096, width - start) for start in range(0, width, 4096)]


_TOKENIZER = None


def _validation_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
            _TOKENIZER = AutoTokenizer.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        except Exception as exc:
            raise ValueError(
                f"exact production tokenizer unavailable to harvest: {exc}") from exc
    return _TOKENIZER


def _generation_prefix_ids(tokenizer, messages: list[dict[str, Any]]) -> list[int]:
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("committed-case messages do not end in a user request")
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids.get("input_ids")
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def _token_ids(value: Any, label: str) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value.get("input_ids")
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise ValueError(f"{label} did not produce integer token IDs")
    return [int(item) for item in value]


def _canonical_message_ids(tokenizer, messages: list[dict[str, Any]]) -> list[int]:
    """Independent Qwen canonical non-final rendering (dummy-user boundary)."""
    with_dummy = _token_ids(tokenizer.apply_chat_template(
        list(messages) + [{"role": "user", "content": "x"}],
        tokenize=True, add_generation_prompt=False), "canonical rendering")
    marker_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    if len(marker_ids) != 1:
        raise ValueError("production tokenizer im_start marker is not singular")
    starts = [index for index, token in enumerate(with_dummy)
              if token == int(marker_ids[0])]
    if len(starts) != len(messages) + 1:
        raise ValueError("canonical rendering message boundaries differ")
    return with_dummy[:starts[len(messages)]]


def _rendered_assistant_ids(tokenizer, messages: list[dict[str, Any]],
                            text: str) -> list[int]:
    prefix = _generation_prefix_ids(tokenizer, messages)
    full = _canonical_message_ids(
        tokenizer, list(messages) + [{"role": "assistant", "content": text}])
    if full[:len(prefix)] != prefix:
        raise ValueError("assistant rendering changed its generation prefix")
    end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(end_ids) != 1:
        raise ValueError("production tokenizer im_end marker is not singular")
    try:
        end = full.index(int(end_ids[0]), len(prefix))
    except ValueError as exc:
        raise ValueError("assistant rendering lacks im_end") from exc
    content = full[len(prefix):end]
    if not content or tokenizer.decode(content).strip() != text.strip():
        raise ValueError("assistant content rendering differs")
    return content


def _calibration_labels(cid: str) -> tuple[str, str]:
    bit = hashlib.sha256(
        f"20260711:{cid}:calibration".encode()).digest()[0] & 1
    return ("A", "B") if bit == 0 else ("B", "A")


def _calibration_messages(approved: str) -> list[dict[str, str]]:
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


def _reconstruct_calibration_variant(tokenizer, cid: str) -> dict[str, Any]:
    correct_label, wrong_label = _calibration_labels(cid)
    correct_messages = _calibration_messages(correct_label)
    wrong_messages = _calibration_messages(wrong_label)
    correct = _generation_prefix_ids(tokenizer, correct_messages)
    wrong = _generation_prefix_ids(tokenizer, wrong_messages)
    if len(correct) != len(wrong):
        raise ValueError(f"calibration {cid} prefix lengths differ")
    changed = [index for index, pair in enumerate(zip(correct, wrong))
               if pair[0] != pair[1]]
    marker_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    starts = [index for index, token in enumerate(correct)
              if token == int(marker_ids[0])]
    if len(starts) != len(correct_messages) + 1:
        raise ValueError(f"calibration {cid} message boundaries differ")
    allowed = list(range(starts[1], starts[2]))
    structural = [index for index in range(len(correct))
                  if index not in changed]
    fresh = [
        {"role": "system", "content": CALIBRATION_SYSTEM},
        {"role": "user", "content": CALIBRATION_REQUEST},
    ]
    summary_ids = _rendered_assistant_ids(
        tokenizer, fresh, CALIBRATION_SUMMARY)
    compacted = [
        correct_messages[0],
        {"role": "user", "content": CALIBRATION_REQUEST},
        {"role": "assistant", "content": CALIBRATION_SUMMARY},
        *correct_messages[3:-1],
    ]
    context_ids = _canonical_message_ids(tokenizer, compacted)
    probe_messages = [
        *compacted, {"role": "user", "content": CALIBRATION_PROBE}]
    probe_prefix = _generation_prefix_ids(tokenizer, probe_messages)
    if probe_prefix[:len(context_ids)] != context_ids:
        raise ValueError(f"calibration {cid} probe prefix changed context")
    correct_text = f"Label {correct_label}."
    wrong_text = f"Label {wrong_label}."
    return {
        "conversation_id": cid,
        "correct_label": correct_label,
        "wrong_label": wrong_label,
        "summary_text": CALIBRATION_SUMMARY,
        "summary_ids": summary_ids,
        "summary_sha256": _sha256_ints(summary_ids, f"calibration[{cid}].summary"),
        "correct_prefix_ids": correct,
        "wrong_prefix_ids": wrong,
        "correct_prefix_sha256": _sha256_ints(
            correct, f"calibration[{cid}].correct"),
        "wrong_prefix_sha256": _sha256_ints(
            wrong, f"calibration[{cid}].wrong"),
        "prefix_length": len(correct),
        "changed_positions": changed,
        "allowed_first_record_content_positions": allowed,
        "structural_positions": structural,
        "exact_length": True,
        "changed_only_first_record_content": set(changed).issubset(allowed),
        "structural_slots_equal": all(correct[i] == wrong[i] for i in structural),
        "special_ids_excluded": not any(
            correct[i] in tokenizer.all_special_ids or
            wrong[i] in tokenizer.all_special_ids for i in changed),
        "targets": {
            "correct_text": correct_text,
            "wrong_text": wrong_text,
            "correct_ids": [int(value) for value in tokenizer.encode(
                correct_text, add_special_tokens=False)],
            "wrong_ids": [int(value) for value in tokenizer.encode(
                wrong_text, add_special_tokens=False)],
            "equal_token_length": True,
            "token_length": len(tokenizer.encode(
                correct_text, add_special_tokens=False)),
            "rendered_correct_ids": _rendered_assistant_ids(
                tokenizer, probe_messages, correct_text),
            "rendered_wrong_ids": _rendered_assistant_ids(
                tokenizer, probe_messages, wrong_text),
            "rendered_equal_token_length": True,
            "rendered_token_length": len(_rendered_assistant_ids(
                tokenizer, probe_messages, correct_text)),
        },
    }


def _validate_native_messages(conversation: dict[str, Any], label: str) -> int:
    messages = conversation.get("messages")
    middle_end = (conversation.get("sections") or {}).get("middle_end_msg")
    if (not isinstance(messages, list) or len(messages) < 3 or
            not isinstance(middle_end, int) or
            not 1 < middle_end < len(messages) or
            (messages[0] or {}).get("role") != "system" or
            (messages[middle_end] or {}).get("role") != "user"):
        raise ValueError(f"{label} native conversation structure differs")
    for index, message in enumerate(messages[1:], 1):
        expected = "user" if index % 2 else "assistant"
        if not isinstance(message, dict) or message.get("role") != expected:
            raise ValueError(f"{label} message role order differs")
    return middle_end


def _unique_token_span(container: list[int], needle: list[int], lo: int,
                       hi: int, label: str) -> tuple[int, int]:
    matches = [index for index in range(lo, hi - len(needle) + 1)
               if container[index:index + len(needle)] == needle]
    if len(matches) != 1:
        raise ValueError(f"{label} content span is not unique: {matches}")
    return matches[0], matches[0] + len(needle)


def _reconstruct_donor_replacements(
        tokenizer, target: dict[str, Any], donor: dict[str, Any],
        cid: str) -> dict[str, Any]:
    """Independently rebuild the exact frozen wrong-prefix construction."""
    target_end = _validate_native_messages(target, f"donor target {cid}")
    donor_end = _validate_native_messages(donor, f"donor source {cid}")
    messages = list(target["messages"]) + [
        {"role": "user", "content": SUMMARY_REQUEST}]
    correct = _generation_prefix_ids(tokenizer, messages)
    marker_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    if len(marker_ids) != 1:
        raise ValueError("production tokenizer im_start marker is not singular")
    starts = [index for index, token in enumerate(correct)
              if token == int(marker_ids[0])]
    if len(starts) != len(messages) + 1:
        raise ValueError(f"donor target {cid} message-boundary coverage differs")
    special = {int(value) for value in tokenizer.all_special_ids}
    donor_by_role: dict[str, list[tuple[int, list[int]]]] = {}
    for donor_index in range(1, donor_end):
        message = donor["messages"][donor_index]
        pool = [int(value) for value in tokenizer.encode(
            message["content"], add_special_tokens=False)]
        if not pool or any(value in special for value in pool):
            raise ValueError(f"donor source {cid} has empty/special content")
        donor_by_role.setdefault(message["role"], []).append(
            (donor_index, pool))

    wrong = list(correct)
    content: set[int] = set()
    role_ordinals: dict[str, int] = {}
    replacements = []
    for target_index in range(1, target_end):
        message = messages[target_index]
        target_ids = [int(value) for value in tokenizer.encode(
            message["content"], add_special_tokens=False)]
        if not target_ids:
            raise ValueError(f"donor target {cid} has empty content")
        start, end = _unique_token_span(
            correct, target_ids, starts[target_index], starts[target_index + 1],
            f"donor target {cid} message {target_index}")
        pools = donor_by_role.get(message["role"], [])
        if not pools:
            raise ValueError(f"donor source {cid} lacks role {message['role']}")
        ordinal = role_ordinals.get(message["role"], 0)
        role_ordinals[message["role"]] = ordinal + 1
        donor_index, pool = pools[ordinal % len(pools)]
        replacement_ids = [pool[index % len(pool)]
                           for index in range(end - start)]
        wrong[start:end] = replacement_ids
        content.update(range(start, end))
        replacements.append({
            "target_message_index": target_index,
            "donor_message_index": donor_index,
            "role": message["role"],
            "start": start,
            "end": end,
            "target_ids": target_ids,
            "donor_pool_ids": pool,
            "replacement_ids": replacement_ids,
            "cycles": math.ceil(len(replacement_ids) / len(pool)),
            "target_ids_sha256": _sha256_ints(
                target_ids, f"donor[{cid}].target_ids"),
            "source_pool_sha256": _sha256_ints(
                pool, f"donor[{cid}].donor_pool_ids"),
            "replacement_sha256": _sha256_ints(
                replacement_ids, f"donor[{cid}].replacement_ids"),
            "length": end - start,
            "contains_special_token": any(
                value in special for value in replacement_ids),
        })
    structural = [index for index in range(len(correct))
                  if index not in content]
    changed = [index for index, pair in enumerate(zip(correct, wrong))
               if pair[0] != pair[1]]
    return {
        "correct_ids": correct,
        "wrong_ids": wrong,
        "structural": structural,
        "content": sorted(content),
        "changed": changed,
        "replacements": replacements,
    }


def _backend_payload_sha256(doc: dict[str, Any]) -> str:
    payload = {key: value for key, value in doc.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def _sha256_ints(values: Any, label: str) -> str:
    if not isinstance(values, list) or any(
            not isinstance(value, int) for value in values):
        raise ValueError(f"{label} is not an integer array")
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _validate_backend_attestation(doc: Any, label: str) -> None:
    if not isinstance(doc, dict):
        raise ValueError(f"{label} backend attestation is not an object")
    required = {
        "requested_implementation", "model_config", "text_config",
        "text_config_is_model_config", "expected_layer_count", "layers", "sha256",
    }
    if set(doc) != required:
        raise ValueError(f"{label} backend attestation fields differ")
    if doc["requested_implementation"] != ATTENTION_BACKEND or \
            doc["expected_layer_count"] != 48:
        raise ValueError(f"{label} backend request/layer count differs")
    for scope in ("model_config", "text_config"):
        row = doc.get(scope)
        if not isinstance(row, dict) or row.get("scope") != scope:
            raise ValueError(f"{label} {scope} record malformed")
        if not isinstance(row.get("config_class"), str) or not row["config_class"]:
            raise ValueError(f"{label} {scope} class absent")
        for field in ("_attn_implementation",
                      "_attn_implementation_internal",
                      "resolved_implementation"):
            if row.get(field) != ATTENTION_BACKEND:
                raise ValueError(f"{label} {scope}.{field} is not eager")
    layers = doc.get("layers")
    if not isinstance(layers, list) or len(layers) != 48:
        raise ValueError(f"{label} backend layer coverage differs")
    for index, row in enumerate(layers):
        if not isinstance(row, dict) or row.get("layer_index") != index:
            raise ValueError(f"{label} backend layer order differs")
        for field in ("module_name", "module_class", "module_config_class"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"{label} layer {index} lacks {field}")
        for field in ("module_config__attn_implementation",
                      "module_config__attn_implementation_internal",
                      "resolved_implementation"):
            if row.get(field) != ATTENTION_BACKEND:
                raise ValueError(f"{label} layer {index}.{field} is not eager")
    if doc.get("sha256") != _backend_payload_sha256(doc):
        raise ValueError(f"{label} backend SHA-256 mismatch")


def _require_identity(doc: dict[str, Any], label: str) -> None:
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{label} schema is not {SCHEMA}")
    if doc.get("amendment_id") != AMENDMENT_ID:
        raise ValueError(f"{label} amendment_id mismatch")
    if doc.get("design_id") != DESIGN_ID:
        raise ValueError(f"{label} design_id mismatch")


def _require_backend_fingerprint(doc: dict[str, Any], label: str) -> None:
    if doc.get("attention_backend") != ATTENTION_BACKEND:
        raise ValueError(f"{label} does not freeze eager attention")
    _validate_backend_attestation(
        doc.get("attention_backend_fingerprint"), label)


def _validate_gate(root: Path, expected_status: str) -> dict[str, Any]:
    gate = _load(root / "production_kernel_gate.json")
    _require_identity(gate, "production gate")
    _require_payload_sha256(gate, "production gate")
    if gate.get("status") != expected_status:
        raise ValueError(
            f"production gate status {gate.get('status')!r} != {expected_status!r}")
    if not gate.get("completed_at"):
        raise ValueError("production gate lacks completed_at")
    gates = gate.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("production gate lacks gates object")
    if gate.get("model") != MODEL_ID or gate.get("revision") != MODEL_REVISION:
        raise ValueError("production gate model or revision mismatch")
    if gate.get("dtype") != PARAMETER_DTYPE:
        raise ValueError("production gate dtype mismatch")
    expected_passes = expected_status == "PASS"
    backend = gates.get("attention_backend")
    if expected_passes:
        if not isinstance(backend, dict):
            raise ValueError("production gate lacks attention-backend evidence")
        if (backend.get("observed_backend") != ATTENTION_BACKEND or
                backend.get("passes") is not True):
            raise ValueError("production gate did not attest eager attention")
        _validate_backend_attestation(
            backend.get("fingerprint"), "production gate")
    elif backend is not None:
        if not isinstance(backend, dict):
            raise ValueError("failed gate attention-backend evidence is malformed")
        if (backend.get("passes") is True and
                backend.get("observed_backend") != ATTENTION_BACKEND):
            raise ValueError("failed gate falsely attests a non-eager backend")
    if gates.get("passes") is not expected_passes:
        raise ValueError(
            f"production gates.passes must be {str(expected_passes).lower()}")
    if expected_status == "FAIL":
        evidence = (
            gate.get("error")
            or gate.get("traceback")
            or gates.get("error")
            or gates.get("traceback")
            or gates.get("failures")
        )
        if not evidence:
            raise ValueError("FAIL production gate lacks failure evidence")
    return gate


def _validate_actual_render_schedule(doc: dict[str, Any], label: str) -> None:
    evidence = doc.get("pre_score_schedule_equivalence")
    conversation = doc.get("conversation")
    if not isinstance(evidence, dict) or not isinstance(conversation, dict):
        raise ValueError(f"{label} lacks actual-render schedule evidence")
    _require_identity(evidence, f"{label} actual-render schedule")
    if (evidence.get("status") != "PASS" or evidence.get("passes") is not True or
            evidence.get("semantic_scoring_performed") is not False or
            evidence.get("conversation_id") != doc.get("conversation_id")):
        raise ValueError(f"{label} actual-render schedule verdict differs")
    tokenizer = _validation_tokenizer()
    messages = conversation.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{label} rendered conversation is malformed")
    correct = _generation_prefix_ids(
        tokenizer, list(messages) + [{"role": "user", "content": SUMMARY_REQUEST}])
    fresh = _generation_prefix_ids(
        tokenizer, [messages[0], {"role": "user", "content": SUMMARY_REQUEST}])
    positions = list(range(len(correct)))
    if (evidence.get("complete_prefix_token_ids") != correct or
            evidence.get("complete_prefix_token_sha256") !=
            _sha256_ints(correct, f"{label}.render.tokens") or
            evidence.get("complete_position_ids") != positions or
            evidence.get("complete_position_array_sha256") !=
            _sha256_ints(positions, f"{label}.render.positions") or
            evidence.get("fresh_prefix_token_ids") != fresh or
            evidence.get("token_count") != len(correct) or
            evidence.get("continuation_logical_position") != len(correct)):
        raise ValueError(f"{label} actual-render token evidence differs")
    marker_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    if len(marker_ids) != 1:
        raise ValueError("production tokenizer im_start marker is not singular")
    marker = int(marker_ids[0])
    starts = [i for i, token in enumerate(correct) if token == marker]
    fresh_starts = [i for i, token in enumerate(fresh) if token == marker]
    if len(starts) != len(messages) + 2 or len(fresh_starts) != 3:
        raise ValueError(f"{label} actual-render boundaries differ")
    system_end = starts[1]
    suffix = fresh[system_end:]
    if not suffix or correct[-len(suffix):] != suffix:
        raise ValueError(f"{label} actual-render request suffix differs")
    request_start = len(correct) - len(suffix)
    widths = {"system": system_end, "history": request_start - system_end,
              "request_header": len(correct) - request_start}
    ordinary = _chunk_widths(len(correct))
    blocks = [piece for key in ("system", "history", "request_header")
              for piece in _chunk_widths(widths[key])]
    if (evidence.get("conceptual_block_widths") != widths or
            evidence.get("ordinary_resolved_call_widths") != ordinary or
            evidence.get("message_block_resolved_call_widths") != blocks or
            evidence.get("system_end") != system_end or
            evidence.get("request_header_start") != request_start or
            evidence.get("system_equal") is not True or
            evidence.get("request_header_equal") is not True or
            evidence.get("blocks_nonempty") is not True or
            evidence.get("blocks_ordered_nonoverlapping") is not True or
            evidence.get("blocks_cover_prefix") is not True):
        raise ValueError(f"{label} actual-render partition evidence differs")
    _validate_schedule_measurement(
        evidence, layers=48, tolerance=5e-4,
        label=f"{label}.actual_render_schedule")


def _validate_gapped_destination_schedule(
        doc: dict[str, Any], label: str) -> None:
    """Reconstruct the exact compacted prefix and all schedule aggregates."""
    evidence = doc.get("pre_score_destination_schedule_equivalence")
    conversation = doc.get("conversation")
    summary = doc.get("summary")
    correct_actual = (doc.get("sources") or {}).get("correct_actual")
    if (not isinstance(evidence, dict) or not isinstance(conversation, dict) or
            not isinstance(summary, dict) or not isinstance(correct_actual, dict)):
        raise ValueError(f"{label} lacks gapped-destination schedule evidence")
    _require_identity(evidence, f"{label} gapped-destination schedule")
    if (evidence.get("conversation_id") != doc.get("conversation_id") or
            evidence.get("status") != "PASS" or
            evidence.get("passes") is not True or
            evidence.get("semantic_scoring_performed") is not False or
            evidence.get("reference_complete") is not True or
            evidence.get("alternative_complete") is not True or
            evidence.get("measurement_complete") is not True or
            evidence.get("failure_evidence") not in (None, [], {})):
        raise ValueError(f"{label} gapped-destination conversation/verdict differs")

    tokenizer = _validation_tokenizer()
    messages = conversation.get("messages")
    summary_text = summary.get("text")
    summary_ids = summary.get("token_ids")
    if (not isinstance(messages, list) or not messages or
            not isinstance(summary_text, str) or not summary_text or
            not isinstance(summary_ids, list) or not summary_ids or
            any(not isinstance(value, int) for value in summary_ids)):
        raise ValueError(f"{label} gapped-destination source evidence differs")
    correct = _generation_prefix_ids(
        tokenizer, list(messages) + [{"role": "user", "content": SUMMARY_REQUEST}])
    fresh_messages = [
        messages[0], {"role": "user", "content": SUMMARY_REQUEST}]
    fresh = _generation_prefix_ids(tokenizer, fresh_messages)
    rendered_summary = _rendered_assistant_ids(
        tokenizer, fresh_messages, summary_text)
    if (summary_ids != rendered_summary or
            summary.get("token_sha256") !=
            _sha256_ints(summary_ids, f"{label}.summary") or
            summary.get("request") != SUMMARY_REQUEST or
            summary.get("request_sha256") !=
            hashlib.sha256(SUMMARY_REQUEST.encode()).hexdigest() or
            correct_actual.get("prefix_token_ids") != correct or
            correct_actual.get("prefix_sha256") !=
            _sha256_ints(correct, f"{label}.correct_prefix") or
            correct_actual.get("prefix_token_count") != len(correct) or
            correct_actual.get("prefix_position_ids") != list(range(len(correct))) or
            correct_actual.get("summary_token_ids") != summary_ids or
            correct_actual.get("summary_token_sha256") !=
            _sha256_ints(summary_ids, f"{label}.correct_summary") or
            correct_actual.get("summary_start") != len(correct) or
            correct_actual.get("summary_end") != len(correct) + len(summary_ids) or
            correct_actual.get("summary_position_ids") != list(range(
                len(correct), len(correct) + len(summary_ids)))):
        raise ValueError(f"{label} gapped-destination saved source differs")

    marker_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    if len(marker_ids) != 1:
        raise ValueError("production tokenizer im_start marker is not singular")
    starts = [index for index, token in enumerate(fresh)
              if token == int(marker_ids[0])]
    if len(starts) != 3:
        raise ValueError(f"{label} compacted prefix boundaries differ")
    system_end = starts[1]
    suffix = fresh[system_end:]
    source_summary_start = len(correct)
    request_logical_start = source_summary_start - len(suffix)
    if (not suffix or request_logical_start < system_end or
            correct[:system_end] != fresh[:system_end] or
            correct[request_logical_start:] != suffix):
        raise ValueError(f"{label} compacted logical islands differ")
    prefix_positions = list(range(system_end)) + list(range(
        request_logical_start, source_summary_start))
    summary_positions = list(range(
        source_summary_start, source_summary_start + len(summary_ids)))
    physical_summary_start = len(fresh)
    physical_summary_end = physical_summary_start + len(summary_ids)
    production_partition = [system_end, len(fresh) - system_end]
    alternative_partition = _chunk_widths(len(fresh))
    expected_arrays = {
        "prefix_token_ids": fresh,
        "prefix_position_ids": prefix_positions,
        "summary_token_ids": summary_ids,
        "summary_position_ids": summary_positions,
        "physical_prefix_cache_position_ids": list(range(len(fresh))),
        "physical_summary_cache_position_ids": list(range(
            physical_summary_start, physical_summary_end)),
        "production_prefix_partition": production_partition,
        "alternative_prefix_partition": alternative_partition,
        "summary_step_widths": [1] * len(summary_ids),
    }
    if any(evidence.get(key) != value for key, value in expected_arrays.items()):
        raise ValueError(f"{label} gapped-destination arrays/partitions differ")
    if (evidence.get("prefix_token_sha256") !=
            _sha256_ints(fresh, f"{label}.destination_prefix") or
            evidence.get("prefix_position_sha256") !=
            _sha256_ints(prefix_positions, f"{label}.destination_positions") or
            evidence.get("summary_token_sha256") !=
            _sha256_ints(summary_ids, f"{label}.destination_summary") or
            evidence.get("summary_position_sha256") !=
            _sha256_ints(summary_positions, f"{label}.summary_positions") or
            evidence.get("system_end") != system_end or
            evidence.get("request_logical_start") != request_logical_start or
            evidence.get("source_summary_start") != source_summary_start or
            evidence.get("physical_summary_start") != physical_summary_start or
            evidence.get("physical_summary_end") != physical_summary_end):
        raise ValueError(f"{label} gapped-destination hashes/bounds differ")

    traces = []
    for trace_name in ("reference_trace", "alternative_trace"):
        trace = evidence.get(trace_name)
        if (not isinstance(trace, dict) or trace.get("token_ids") != summary_ids or
                trace.get("start_position") != source_summary_start or
                trace.get("end_position") !=
                source_summary_start + len(summary_ids) or
                trace.get("ended_on_eos") is not False):
            raise ValueError(
                f"{label} gapped-destination {trace_name} differs")
        logprobs = trace.get("token_logprobs")
        if not isinstance(logprobs, list) or len(logprobs) != len(summary_ids):
            raise ValueError(
                f"{label} gapped-destination {trace_name} coverage differs")
        traces.append([
            _finite(value, f"{label}.{trace_name}.token_logprobs[{index}]")
            for index, value in enumerate(logprobs)
        ])
    token_differences = [abs(left - right)
                         for left, right in zip(traces[0], traces[1])]
    if evidence.get("token_logprob_abs_differences") != token_differences:
        raise ValueError(f"{label} gapped-destination tokenwise trace differs")
    token_max = max(token_differences)
    if _finite(evidence.get("token_logprob_max_abs"),
               f"{label}.destination.token_max") != token_max:
        raise ValueError(f"{label} gapped-destination token aggregate differs")

    per_layer = evidence.get("per_layer")
    if not isinstance(per_layer, list) or len(per_layer) != 48 or \
            [row.get("layer") for row in per_layer] != list(range(48)):
        raise ValueError(f"{label} gapped-destination layer coverage differs")
    layer_k: list[float] = []
    layer_v: list[float] = []
    for layer, row in enumerate(per_layer):
        k_rows = row.get("k_per_summary_token_max_abs")
        v_rows = row.get("v_per_summary_token_max_abs")
        if (not isinstance(k_rows, list) or len(k_rows) != len(summary_ids) or
                not isinstance(v_rows, list) or len(v_rows) != len(summary_ids)):
            raise ValueError(
                f"{label} gapped-destination summary-row coverage differs: {layer}")
        finite_k = [_finite(value, f"{label}.layer[{layer}].K[{index}]")
                    for index, value in enumerate(k_rows)]
        finite_v = [_finite(value, f"{label}.layer[{layer}].V[{index}]")
                    for index, value in enumerate(v_rows)]
        if any(value < 0 for value in finite_k + finite_v):
            raise ValueError(
                f"{label} gapped-destination negative absolute difference: {layer}")
        k_max = max(finite_k)
        v_max = max(finite_v)
        if (_finite(row.get("k_max_abs"), f"{label}.layer[{layer}].Kmax") !=
                k_max or
                _finite(row.get("v_max_abs"), f"{label}.layer[{layer}].Vmax") !=
                v_max):
            raise ValueError(
                f"{label} gapped-destination per-layer aggregate differs: {layer}")
        layer_k.append(k_max)
        layer_v.append(v_max)
    cache_k = max(layer_k)
    cache_v = max(layer_v)
    aggregate = max(token_max, cache_k, cache_v)
    if (_finite(evidence.get("cache_k_max_abs"),
                f"{label}.destination.cache_K") != cache_k or
            _finite(evidence.get("cache_v_max_abs"),
                    f"{label}.destination.cache_V") != cache_v or
            _finite(evidence.get("observed_aggregate"),
                    f"{label}.destination.aggregate") != aggregate or
            _finite(evidence.get("threshold"),
                    f"{label}.destination.threshold") != 5e-4 or
            evidence.get("comparison") != "<=" or aggregate > 5e-4):
        raise ValueError(f"{label} gapped-destination terminal aggregate differs")


def _validate_target_score(doc: Any, label: str) -> float:
    """Recompute one persisted target mean from its token log-probabilities."""
    if not isinstance(doc, dict):
        raise ValueError(f"{label} target score is not an object")
    token_ids = doc.get("token_ids")
    token_logprobs = doc.get("token_logprobs")
    if (not isinstance(token_ids, list) or not token_ids or
            any(not isinstance(token, int) for token in token_ids) or
            not isinstance(token_logprobs, list) or not token_logprobs or
            len(token_ids) != len(token_logprobs)):
        raise ValueError(f"{label} target token/logprob coverage differs")
    values = [
        _finite(value, f"{label}.token_logprobs[{index}]")
        for index, value in enumerate(token_logprobs)
    ]
    recomputed = sum(values) / len(values)
    if _finite(doc.get("mean_logprob"), f"{label}.mean_logprob") != recomputed:
        raise ValueError(f"{label} target mean_logprob differs")
    return recomputed


def _validate_arm_score(doc: Any, label: str) -> tuple[float, list[tuple[Any, ...]]]:
    """Recompute plant margins and the conversation-level arm aggregate."""
    if not isinstance(doc, dict):
        raise ValueError(f"{label} arm score is not an object")
    plants = doc.get("plants")
    if not isinstance(plants, list) or not plants:
        raise ValueError(f"{label} arm has no plant rows")
    margins: list[float] = []
    identities: list[tuple[Any, ...]] = []
    seen_ids: set[Any] = set()
    for index, row in enumerate(plants):
        row_label = f"{label}.plants[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{row_label} is not an object")
        plant_id = row.get("plant_id")
        if not isinstance(plant_id, str) or not plant_id or plant_id in seen_ids:
            raise ValueError(f"{row_label} plant identity differs")
        seen_ids.add(plant_id)
        identities.append((plant_id, row.get("category"), row.get("probe")))
        correct = _validate_target_score(row.get("correct"), f"{row_label}.correct")
        counterfactual = _validate_target_score(
            row.get("counterfactual"), f"{row_label}.counterfactual")
        recomputed_margin = correct - counterfactual
        if _finite(row.get("margin"), f"{row_label}.margin") != recomputed_margin:
            raise ValueError(f"{row_label} plant margin differs")
        margins.append(recomputed_margin)
    recomputed_conversation = sum(margins) / len(margins)
    if _finite(doc.get("conversation_margin"),
               f"{label}.conversation_margin") != recomputed_conversation:
        raise ValueError(f"{label} conversation_margin differs")
    return recomputed_conversation, identities


def _validate_semantic_score_aggregates(doc: dict[str, Any], label: str) -> None:
    """Independently derive every decision-bearing semantic aggregate."""
    arm_scores = doc.get("arm_scores")
    outcomes = doc.get("conversation_outcomes")
    if not isinstance(arm_scores, dict) or not isinstance(outcomes, dict):
        raise ValueError(f"{label} semantic score maps are malformed")
    expected_plants = None
    for arm in ARMS:
        aggregate, plant_identities = _validate_arm_score(
            arm_scores.get(arm), f"{label}.arm_scores.{arm}")
        if expected_plants is None:
            expected_plants = plant_identities
        elif plant_identities != expected_plants:
            raise ValueError(f"{label} arm plant coverage/order differs")
        if _finite(outcomes.get(arm),
                   f"{label}.conversation_outcomes.{arm}") != aggregate:
            raise ValueError(f"{label} conversation outcome differs: {arm}")

    calibration = doc.get("calibration")
    persisted_calibration = doc.get("calibration_outcomes")
    if not isinstance(calibration, dict) or not isinstance(
            persisted_calibration, dict):
        raise ValueError(f"{label} calibration evidence is malformed")
    details = calibration.get("arm_details")
    embedded_outcomes = calibration.get("outcomes")
    calibration_arms = {"G_fresh", "G_correct", "G_wrong"}
    if (not isinstance(details, dict) or set(details) != calibration_arms or
            not isinstance(embedded_outcomes, dict) or
            set(embedded_outcomes) != calibration_arms or
            set(persisted_calibration) != calibration_arms):
        raise ValueError(f"{label} calibration arm coverage differs")
    expected_calibration_plants = None
    for arm in sorted(calibration_arms):
        aggregate, plant_identities = _validate_arm_score(
            details[arm], f"{label}.calibration.arm_details.{arm}")
        if expected_calibration_plants is None:
            expected_calibration_plants = plant_identities
        elif plant_identities != expected_calibration_plants:
            raise ValueError(f"{label} calibration plant coverage/order differs")
        if (_finite(embedded_outcomes.get(arm),
                    f"{label}.calibration.outcomes.{arm}") != aggregate or
                _finite(persisted_calibration.get(arm),
                        f"{label}.calibration_outcomes.{arm}") != aggregate):
            raise ValueError(f"{label} calibration outcome differs: {arm}")


def _validate_checkpoint(doc: dict[str, Any], path: Path, *, scored: bool,
                         expected_fingerprint: dict[str, Any] | None = None) -> None:
    _require_identity(doc, path.name)
    if scored:
        if doc.get("schema") != SCHEMA or doc.get("stage") != "scored" or \
                doc.get("status") != "scored":
            raise ValueError(f"{path.name} is not a schema-2 scored checkpoint")
        required = {
            "conversation", "summary", "sources", "destination", "arm_scores",
            "conversation_outcomes", "gates", "runtime", "fingerprint",
        }
        missing = sorted(required - doc.keys())
        if missing:
            raise ValueError(f"{path.name} missing fields {missing}")
        _validate_actual_render_schedule(doc, path.name)
        _validate_gapped_destination_schedule(doc, path.name)
        for field in ("arm_scores", "conversation_outcomes"):
            arms = doc.get(field)
            if not isinstance(arms, dict) or set(arms) != set(ARMS):
                raise ValueError(
                    f"{path.name} {field} must contain exactly {list(ARMS)}")
            stale = OLD_ARMS.intersection(arms)
            if stale:
                raise ValueError(f"{path.name} contains retired arms {sorted(stale)}")
        _validate_semantic_score_aggregates(doc, path.name)
        gates = doc.get("gates")
        if not isinstance(gates, dict) or gates.get("technical_pass") is not True:
            raise ValueError(f"{path.name} lacks technical_pass=true")
        fingerprint = doc.get("fingerprint")
        if not isinstance(fingerprint, dict):
            raise ValueError(f"{path.name} lacks fingerprint object")
        _require_identity(fingerprint, f"{path.name} fingerprint")
        if fingerprint.get("frozen_order") != list(FROZEN_ORDER):
            raise ValueError(f"{path.name} fingerprint frozen order mismatch")
        if fingerprint.get("wrong_donors") != WRONG_DONORS:
            raise ValueError(f"{path.name} fingerprint donor map mismatch")
        _require_backend_fingerprint(fingerprint, f"{path.name} fingerprint")
        if (expected_fingerprint is not None and
                fingerprint != expected_fingerprint):
            raise ValueError(
                f"{path.name} fingerprint differs from manifest fingerprint")
    else:
        if doc.get("stage") != "void" or doc.get("status") != "void":
            raise ValueError(f"{path.name} is not a void checkpoint")
        evidence = doc.get("failure") or doc.get("error") or doc.get("failures")
        if not evidence:
            raise ValueError(f"{path.name} void checkpoint lacks failure evidence")


def _checkpoint_paths(root: Path) -> list[Path]:
    return sorted(root.glob("conv_*.json"))


def _validate_terminal_integrity(root: Path, expected_status: str) -> dict[str, Any]:
    index_path = root / "terminal_artifact_index.json"
    receipt_path = root / "terminal_receipt.json"
    index = _load(index_path)
    receipt = _load(receipt_path)
    _require_identity(index, "terminal index")
    _require_identity(receipt, "terminal receipt")
    if index.get("status") != expected_status or \
            receipt.get("status") != expected_status:
        raise ValueError("terminal envelope status mismatch")
    if receipt.get("index_path") != index_path.name:
        raise ValueError("terminal receipt index path mismatch")
    if receipt.get("index_bytes") != index_path.stat().st_size or \
            receipt.get("index_raw_sha256") != _raw_file_sha256(index_path):
        raise ValueError("terminal receipt does not match index bytes")
    canonical_gate = root / "production_kernel_gate.json"
    unique: list[Path] = []
    if canonical_gate.exists():
        unique = sorted(root.glob("production_kernel_gate_*.json"))
        if len(unique) != 1:
            raise ValueError(f"terminal run has {len(unique)} unique gate attempts")
        gate = _load(canonical_gate)
        stage_refs = ((gate.get("gates") or {}).get("stage_refs") or {})
        if not isinstance(stage_refs, dict):
            raise ValueError("technical gate stage_refs is not an object")
        expected_ref_names = {
            "committed_case_schedule_fixtures",
            "external_donor_construction",
        }
        if set(stage_refs) != expected_ref_names:
            raise ValueError("technical gate heavy-stage references differ")
        sidecar_paths: list[str] = []
        for stage_name, ref in sorted(stage_refs.items()):
            if stage_name not in expected_ref_names or not isinstance(ref, dict):
                raise ValueError(f"invalid technical stage reference: {stage_name}")
            relative = ref.get("path")
            expected_name = f"technical_stage_{stage_name}.json"
            if relative != expected_name:
                raise ValueError(f"technical stage path differs: {stage_name}")
            sidecar_paths.append(relative)
        required = [
            "manifest.json", "production_kernel_gate.json", unique[0].name,
            *sidecar_paths,
        ]
        if expected_status == "FAIL":
            required.append("failure.json")
        required = sorted(required)
    else:
        required = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*.json")
            if path.name not in {index_path.name, receipt_path.name})
    if index.get("required_apparatus_payload_paths") != required:
        raise ValueError("terminal index required-path set differs")
    rows = index.get("artifacts")
    if not isinstance(rows, list) or [row.get("path") for row in rows] != required:
        raise ValueError("terminal index artifact rows differ")
    raw_hashes = {}
    for row in rows:
        path = root / row["path"]
        doc = _load(path)
        _require_identity(doc, f"indexed {row['path']}")
        _require_payload_sha256(doc, f"indexed {row['path']}")
        if row.get("bytes") != path.stat().st_size:
            raise ValueError(f"indexed byte count differs: {row['path']}")
        raw_sha = _raw_file_sha256(path)
        if row.get("raw_sha256") != raw_sha or \
                row.get("payload_sha256") != doc.get("payload_sha256"):
            raise ValueError(f"indexed hashes differ: {row['path']}")
        raw_hashes[row["path"]] = raw_sha
    if unique and canonical_gate.read_bytes() != unique[0].read_bytes():
        raise ValueError("canonical and unique gate bytes differ")
    allowed_json = set(required) | {index_path.name, receipt_path.name}
    extra = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*.json")
        if path.relative_to(root).as_posix() not in allowed_json)
    if extra:
        raise ValueError(f"terminal run has unindexed JSON payloads {extra}")
    return {
        "index_raw_sha256": _raw_file_sha256(index_path),
        "receipt_raw_sha256": _raw_file_sha256(receipt_path),
        "payload_raw_sha256": raw_hashes,
    }


def _resolve_heavy_stages(root: Path, gate: dict[str, Any]) -> dict[str, Any]:
    """Verify and materialize the two indexed heavy lifecycle sidecars."""
    gates = json.loads(json.dumps(gate.get("gates") or {}))
    refs = gates.pop("stage_refs", None)
    expected = {
        "committed_case_schedule_fixtures",
        "external_donor_construction",
    }
    if not isinstance(refs, dict) or set(refs) != expected:
        raise ValueError("technical gate heavy-stage reference set differs")
    for stage_name in sorted(expected):
        if stage_name in gates:
            raise ValueError(f"heavy stage was both inline and referenced: {stage_name}")
        ref = refs[stage_name]
        relative = f"technical_stage_{stage_name}.json"
        if not isinstance(ref, dict) or ref.get("path") != relative:
            raise ValueError(f"heavy-stage reference path differs: {stage_name}")
        path = root / relative
        sidecar = _load(path)
        _require_identity(sidecar, relative)
        _require_payload_sha256(sidecar, relative)
        if (sidecar.get("status") != gate.get("status") or
                sidecar.get("kind") != "technical_gate_stage_sidecar" or
                sidecar.get("stage_name") != stage_name or
                not isinstance(sidecar.get("lifecycle"), dict)):
            raise ValueError(f"heavy-stage sidecar identity differs: {stage_name}")
        if (ref.get("byte_count") != path.stat().st_size or
                ref.get("raw_file_sha256") != _raw_file_sha256(path) or
                ref.get("payload_sha256") != sidecar.get("payload_sha256")):
            raise ValueError(f"heavy-stage sidecar binding differs: {stage_name}")
        gates[stage_name] = sidecar["lifecycle"]
    return gates


def _find_forbidden_semantic_fields(value: Any, path: str = "") -> list[str]:
    forbidden = {
        "arm_scores", "conversation_outcomes", "calibration_outcomes",
        "technical_margins_not_semantic_outcomes", "semantic_outcomes",
        "target_scores",
    }
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in forbidden:
                found.append(child_path)
            found.extend(_find_forbidden_semantic_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_semantic_fields(
                child, f"{path}[{index}]"))
    return found


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except Exception as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _validate_hash_rows(value: Any, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) != 48:
        raise ValueError(f"{label} hash-row coverage differs")
    if [row.get("layer") for row in value] != [str(i) for i in range(48)]:
        raise ValueError(f"{label} hash-row order differs")
    for row in value:
        for key in ("k_sha256", "v_sha256"):
            digest = row.get(key)
            if (not isinstance(digest, str) or len(digest) != 64 or
                    any(char not in "0123456789abcdef" for char in digest.lower())):
                raise ValueError(f"{label} {key} differs")
    return value


def _required_pass_stage(gates: dict[str, Any], key: str) -> dict[str, Any]:
    stage = gates.get(key)
    if not isinstance(stage, dict):
        raise ValueError(f"required gate stage absent: {key}")
    if stage.get("status") != "PASS" or stage.get("passes") is not True:
        raise ValueError(f"required gate stage did not PASS: {key}")
    if stage.get("failure_evidence") not in (None, [], {}):
        raise ValueError(f"passing stage has failure evidence: {key}")
    return stage


def _validate_schedule_measurement(row: dict[str, Any], *, layers: int,
                                   tolerance: float, label: str) -> float:
    if row.get("status") != "PASS" or row.get("passes") is not True:
        raise ValueError(f"{label} schedule row did not PASS")
    if row.get("base_measurement_complete") is not True or \
            row.get("continuation_measurement_complete") is not True:
        raise ValueError(f"{label} schedule measurement is incomplete")
    if _finite(row.get("tolerance"), f"{label}.tolerance") != tolerance:
        raise ValueError(f"{label} tolerance differs")
    base = row.get("per_layer")
    continuation = row.get("continuation_per_layer")
    if not isinstance(base, list) or len(base) != layers or \
            not isinstance(continuation, list) or len(continuation) != layers:
        raise ValueError(f"{label} layer coverage differs")
    if [x.get("layer") for x in base] != list(range(layers)) or \
            [x.get("layer") for x in continuation] != list(range(layers)):
        raise ValueError(f"{label} layer order differs")
    cache_k = max(_finite(x.get("k_max_abs"), f"{label}.K") for x in base)
    cache_v = max(_finite(x.get("v_max_abs"), f"{label}.V") for x in base)
    next_k = max(_finite(x.get("k_max_abs"), f"{label}.nextK")
                 for x in continuation)
    next_v = max(_finite(x.get("v_max_abs"), f"{label}.nextV")
                 for x in continuation)
    metrics = {
        "cache_k_max_abs": cache_k,
        "cache_v_max_abs": cache_v,
        "last_logits_max_abs": _finite(
            row.get("last_logits_max_abs"), f"{label}.logits"),
        "selected_margin_abs_shift": _finite(
            row.get("selected_margin_abs_shift"), f"{label}.margin"),
        "continuation_logits_max_abs": _finite(
            row.get("continuation_logits_max_abs"), f"{label}.next_logits"),
        "continuation_k_max_abs": next_k,
        "continuation_v_max_abs": next_v,
    }
    for key, observed in metrics.items():
        if _finite(row.get(key), f"{label}.{key}") != observed:
            raise ValueError(f"{label} stored {key} aggregate differs")
    aggregate = max(metrics.values())
    if _finite(row.get("observed_aggregate"),
               f"{label}.observed_aggregate") != aggregate:
        raise ValueError(f"{label} observed aggregate differs")
    if aggregate > tolerance:
        raise ValueError(f"{label} exceeds frozen tolerance")
    return aggregate


def _validate_committed_case_source(
        row: dict[str, Any], cid: str, fingerprint: dict[str, Any],
        repo_root: Path) -> None:
    """Recompute exact file, tokenizer, token, boundary, and partition evidence."""
    source_path = row.get("source_path")
    if source_path != f"data/synthetic/{cid}.json":
        raise ValueError(f"committed-case source path differs: {cid}")
    path = repo_root / source_path
    raw_bytes = path.read_bytes()
    parsed = json.loads(raw_bytes)
    if parsed != row.get("parsed_source") or str(parsed.get("id")) != cid:
        raise ValueError(f"committed-case parsed source differs: {cid}")
    if (row.get("raw_source_file_sha256") !=
            hashlib.sha256(raw_bytes).hexdigest() or
            row.get("canonical_parsed_source_sha256") !=
            _canonical_json_sha256(parsed)):
        raise ValueError(f"committed-case source hashes differ: {cid}")
    inventory_rows = ((fingerprint.get("input_inventory") or {}).get("files") or [])
    inventory = {item.get("path"): item for item in inventory_rows
                 if isinstance(item, dict)}
    inventory_row = inventory.get(source_path)
    if (not isinstance(inventory_row, dict) or
            inventory_row.get("bytes") != len(raw_bytes) or
            inventory_row.get("sha256") != hashlib.sha256(raw_bytes).hexdigest()):
        raise ValueError(f"committed-case input inventory differs: {cid}")
    metadata = fingerprint.get("subject_metadata") or {}
    tokenizer = _validation_tokenizer()
    vocab_sha = _canonical_json_sha256(tokenizer.get_vocab())
    template_sha = hashlib.sha256(
        str(tokenizer.chat_template).encode()).hexdigest()
    request_sha = hashlib.sha256(SUMMARY_REQUEST.encode()).hexdigest()
    if (row.get("exact_model_revision") != MODEL_REVISION or
            row.get("tokenizer_vocabulary_sha256") != vocab_sha or
            metadata.get("tokenizer_vocab_sha256") != vocab_sha or
            row.get("chat_template_sha256") != template_sha or
            metadata.get("chat_template_sha256") != template_sha or
            row.get("summary_request_sha256") != request_sha or
            fingerprint.get("summary_request_sha256") != request_sha):
        raise ValueError(f"committed-case tokenizer/request binding differs: {cid}")
    recorded_author = (parsed.get("meta") or {}).get("author")
    if row.get("recorded_author") != recorded_author:
        raise ValueError(f"committed-case recorded author differs: {cid}")

    messages = parsed.get("messages")
    if not isinstance(messages, list) or not messages or \
            messages[0].get("role") != "system":
        raise ValueError(f"committed-case messages malformed: {cid}")
    correct_messages = list(messages) + [
        {"role": "user", "content": SUMMARY_REQUEST}]
    fresh_messages = [messages[0], {"role": "user", "content": SUMMARY_REQUEST}]
    correct_ids = _generation_prefix_ids(tokenizer, correct_messages)
    fresh_ids = _generation_prefix_ids(tokenizer, fresh_messages)
    marker_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    if len(marker_ids) != 1:
        raise ValueError("production tokenizer im_start marker is not singular")
    marker = int(marker_ids[0])
    starts = [index for index, token in enumerate(correct_ids) if token == marker]
    fresh_starts = [index for index, token in enumerate(fresh_ids) if token == marker]
    if len(starts) != len(messages) + 2 or len(fresh_starts) != 3:
        raise ValueError(f"committed-case message boundary coverage differs: {cid}")
    system_end = starts[1]
    if fresh_starts[1] != system_end:
        raise ValueError(f"committed-case fresh system boundary differs: {cid}")
    suffix = fresh_ids[system_end:]
    if not suffix or correct_ids[-len(suffix):] != suffix:
        raise ValueError(f"committed-case request/header suffix differs: {cid}")
    request_start = len(correct_ids) - len(suffix)
    widths = {
        "system": system_end,
        "history": request_start - system_end,
        "request_header": len(correct_ids) - request_start,
    }
    if (row.get("complete_prefix_token_ids") != correct_ids or
            row.get("fresh_prefix_token_ids") != fresh_ids or
            row.get("complete_prefix_token_sha256") !=
            _sha256_ints(correct_ids, f"{cid}.retokenized") or
            row.get("token_count") != len(correct_ids) or
            row.get("continuation_logical_position") != len(correct_ids) or
            row.get("expected_continuation_logical_position") !=
            CASE_CONTINUATION_POSITIONS[cid] or
            row.get("continuation_position_matches_frozen") is not True or
            len(correct_ids) != CASE_CONTINUATION_POSITIONS[cid]):
        raise ValueError(f"committed-case retokenized prefix differs: {cid}")
    expected_positions = list(range(len(correct_ids)))
    if (row.get("complete_position_ids") != expected_positions or
            row.get("complete_position_array_sha256") !=
            _sha256_ints(expected_positions, f"{cid}.positions")):
        raise ValueError(f"committed-case position array differs: {cid}")
    boundary = row.get("boundary_token_ids") or {}
    if (row.get("message_start_positions") != starts or
            boundary.get("im_start") != marker or
            boundary.get("system_end_token") != correct_ids[system_end] or
            boundary.get("request_header_start_token") !=
            correct_ids[request_start] or
            boundary.get("final_prefix_token") != correct_ids[-1] or
            row.get("system_end") != system_end or
            row.get("request_header_start") != request_start or
            row.get("conceptual_block_widths") != widths or
            row.get("blocks_nonempty") is not True or
            row.get("blocks_ordered_nonoverlapping") is not True or
            row.get("blocks_cover_prefix") is not True or
            row.get("system_equal") is not True or
            row.get("request_header_equal") is not True):
        raise ValueError(f"committed-case boundary evidence differs: {cid}")
    ordinary = _chunk_widths(len(correct_ids))
    message_block = [piece for key in ("system", "history", "request_header")
                     for piece in _chunk_widths(widths[key])]
    if (row.get("ordinary_resolved_call_widths") != ordinary or
            row.get("message_block_resolved_call_widths") != message_block):
        raise ValueError(f"committed-case resolved partitions differ: {cid}")


def _validate_v7_pass_gates(
        gates: dict[str, Any], *, fingerprint: dict[str, Any] | None,
        static_fingerprint: dict[str, Any] | None,
        repo_root: Path | None, verify_sources: bool) -> None:
    required_order = [
        "static_provenance", "attention_backend", "synthetic_schedule_fixtures",
        "committed_case_schedule_fixtures", "generated_replay_identity",
        "snapshot_rebuild_identity", "physical_causal_mask_identity",
        "future_mutation_identity", "position_structure",
        "intervention_propagation", "calibration_construction",
        "external_donor_construction", "retired_G_delta",
    ]
    if gates.get("stage_order") != required_order:
        raise ValueError("technical gate stage order differs")
    if gates.get("max_technical_logical_position") != \
            MAX_TECHNICAL_LOGICAL_POSITION:
        raise ValueError("technical gate maximum logical position differs")
    if verify_sources:
        if repo_root is None:
            raise ValueError("strict static validation lacks repository root")
        _validate_static_fingerprint(static_fingerprint, fingerprint, repo_root)

    static = _required_pass_stage(gates, "static_provenance")
    static_raw = static.get("raw") or {}
    if (not isinstance(static_fingerprint, dict) or
            static.get("observed_coverage") != 1 or
            static_raw.get("fingerprint_static_sha256") !=
            _canonical_json_sha256(static_fingerprint) or
            static_raw.get("apparatus_inventory_sha256") !=
            _canonical_json_sha256(static_fingerprint.get("apparatus_inventory")) or
            static_raw.get("input_inventory_sha256") !=
            _canonical_json_sha256(static_fingerprint.get("input_inventory")) or
            static_raw.get("code_commit") != static_fingerprint.get("code_commit") or
            static_raw.get("model") != MODEL_ID or
            static_raw.get("revision") != MODEL_REVISION or
            static_raw.get("dtype") != PARAMETER_DTYPE or
            static_raw.get("attention_backend") != ATTENTION_BACKEND or
            static_raw.get("technical_only") is not True):
        raise ValueError("static provenance stage differs")

    backend = _required_pass_stage(gates, "attention_backend")
    if backend.get("observed_coverage") != 48:
        raise ValueError("backend stage coverage differs")
    backend_raw = backend.get("raw") or {}
    _validate_backend_attestation(
        backend_raw.get("fingerprint"), "technical gate")
    subject = backend_raw.get("subject") or {}
    if (subject.get("resolved_revision") != MODEL_REVISION or
            subject.get("dtype") != PARAMETER_DTYPE or
            "cuda" not in str(subject.get("device", "")).lower() or
            int(subject.get("context_limit", -1)) <=
            MAX_TECHNICAL_LOGICAL_POSITION or
            subject.get("max_technical_logical_position") !=
            MAX_TECHNICAL_LOGICAL_POSITION or
            subject.get("context_coverage_passes") is not True):
        raise ValueError("technical subject attestation differs")

    synthetic = _required_pass_stage(gates, "synthetic_schedule_fixtures")
    if _finite(synthetic.get("threshold"), "synthetic threshold") != 5e-4:
        raise ValueError("synthetic stage threshold differs")
    raw = synthetic.get("raw") or {}
    provenance = raw.get("fixture_provenance") or {}
    if (provenance.get("literal") != FROZEN_FIXTURE_LITERAL or
            provenance.get("pool_token_ids") != FROZEN_FIXTURE_POOL or
            provenance.get("pool_sha256") !=
            _sha256_ints(FROZEN_FIXTURE_POOL, "synthetic.pool") or
            provenance.get("margin_token_ids") != FROZEN_MARGIN_IDS or
            provenance.get("continuation_token_id") != FROZEN_CONTINUATION_ID or
            provenance.get("pool_contains_special_token") is not False):
        raise ValueError("synthetic fixture provenance differs")
    rows = raw.get("contiguous")
    if not isinstance(rows, list) or len(rows) != len(SYNTHETIC_LENGTHS):
        raise ValueError("synthetic fixture coverage differs")
    aggregates = []
    for index, (row, length, partitions) in enumerate(zip(
            rows, SYNTHETIC_LENGTHS, SYNTHETIC_PARTITIONS)):
        if (row.get("length") != length or
                row.get("reference_partition") != list(partitions[0]) or
                row.get("alternative_partition") != list(partitions[1])):
            raise ValueError(f"synthetic fixture {index} declaration differs")
        token_ids = [FROZEN_FIXTURE_POOL[i % len(FROZEN_FIXTURE_POOL)]
                     for i in range(length)]
        if row.get("token_ids_sha256") != _sha256_ints(
                token_ids, f"synthetic[{length}].tokens"):
            raise ValueError(f"synthetic fixture {length} token hash differs")
        aggregates.append(_validate_schedule_measurement(
            row, layers=48, tolerance=5e-4,
            label=f"synthetic[{length}]"))
    gap = raw.get("logical_gap")
    if not isinstance(gap, dict):
        raise ValueError("logical-gap fixture absent")
    if (gap.get("logical_positions") != list(range(32)) + list(range(8192, 8224))
            or gap.get("physical_cache_positions") != list(range(64)) or
            gap.get("length") != 64 or
            gap.get("reference_partition") != [32, 32] or
            gap.get("alternative_partition") != [32] + [1] * 32 or
            gap.get("continuation_logical_position") != 8224 or
            gap.get("logical_as_cache_position_rejected") is not True):
        raise ValueError("logical-gap position arrays differ")
    gap_tokens = [FROZEN_FIXTURE_POOL[i % len(FROZEN_FIXTURE_POOL)]
                  for i in range(64)]
    gap_positions = list(range(32)) + list(range(8192, 8224))
    if (gap.get("token_ids_sha256") !=
            _sha256_ints(gap_tokens, "logical_gap.tokens") or
            gap.get("logical_positions_sha256") !=
            _sha256_ints(gap_positions, "logical_gap.positions")):
        raise ValueError("logical-gap hashes differ")
    if gap.get("full_attention_over_physically_prior_rows") is not True:
        raise ValueError("logical-gap full-attention assertion absent")
    aggregates.append(_validate_schedule_measurement(
        gap, layers=48, tolerance=5e-4, label="logical_gap"))
    if synthetic.get("observed_coverage") != 7 or \
            _finite(synthetic.get("observed_aggregate"),
                    "synthetic aggregate") != max(aggregates):
        raise ValueError("synthetic stage aggregate/coverage differs")
    if raw.get("passes") is not True or raw.get("failures") not in ([], None):
        raise ValueError("synthetic raw verdict differs")

    committed = _required_pass_stage(
        gates, "committed_case_schedule_fixtures")
    if _finite(committed.get("threshold"), "case threshold") != 5e-4:
        raise ValueError("case stage threshold differs")
    case_raw = committed.get("raw") or {}
    case_rows = case_raw.get("rows")
    if not isinstance(case_rows, list) or len(case_rows) != 12:
        raise ValueError("committed-case fixture coverage differs")
    case_aggregates = []
    for index, (row, cid) in enumerate(zip(case_rows, FROZEN_ORDER), 1):
        expected_tokens = CASE_CONTINUATION_POSITIONS[cid]
        if (row.get("conversation_id") != cid or
                row.get("order_position") != index or
                row.get("source_path") != f"data/synthetic/{cid}.json" or
                row.get("token_count") != expected_tokens):
            raise ValueError(f"committed-case identity differs: {cid}")
        if verify_sources:
            if not isinstance(fingerprint, dict) or repo_root is None:
                raise ValueError("strict committed-case validation lacks fingerprint/root")
            _validate_committed_case_source(row, cid, fingerprint, repo_root)
        for key in ("raw_source_file_sha256", "canonical_parsed_source_sha256",
                    "tokenizer_vocabulary_sha256", "chat_template_sha256",
                    "summary_request_sha256"):
            if not isinstance(row.get(key), str) or len(row[key]) != 64:
                raise ValueError(f"committed-case {cid} lacks {key}")
        token_ids = row.get("complete_prefix_token_ids")
        positions = row.get("complete_position_ids")
        if len(token_ids or []) != expected_tokens or \
                positions != list(range(expected_tokens)):
            raise ValueError(f"committed-case {cid} raw token/position coverage differs")
        if (_sha256_ints(token_ids, f"{cid}.tokens") !=
                row.get("complete_prefix_token_sha256") or
                _sha256_ints(positions, f"{cid}.positions") !=
                row.get("complete_position_array_sha256")):
            raise ValueError(f"committed-case {cid} token/position hash differs")
        widths = row.get("conceptual_block_widths") or {}
        if any(int(widths.get(key, 0)) <= 0
               for key in ("system", "history", "request_header")) or \
                sum(int(widths[key]) for key in widths) != expected_tokens:
            raise ValueError(f"committed-case {cid} conceptual widths differ")
        if (row.get("system_end") != widths["system"] or
                row.get("request_header_start") !=
                widths["system"] + widths["history"] or
                row.get("system_equal") is not True or
                row.get("request_header_equal") is not True):
            raise ValueError(f"committed-case {cid} boundary equality differs")
        for key in ("ordinary_resolved_call_widths",
                    "message_block_resolved_call_widths"):
            partition = row.get(key)
            if (not isinstance(partition, list) or not partition or
                    any(not isinstance(width, int) or width < 1 or width > 4096
                        for width in partition) or
                    sum(partition) != expected_tokens):
                raise ValueError(f"committed-case {cid} {key} differs")
        case_aggregates.append(_validate_schedule_measurement(
            row, layers=48, tolerance=5e-4, label=f"case[{cid}]"))
    if (committed.get("observed_coverage") != 12 or
            case_raw.get("frozen_order") != list(FROZEN_ORDER) or
            case_raw.get("coverage_exact") is not True or
            _finite(committed.get("observed_aggregate"), "case aggregate") !=
            max(case_aggregates)):
        raise ValueError("committed-case aggregate/order differs")

    generated = _required_pass_stage(gates, "generated_replay_identity")
    gen_raw = generated.get("raw") or {}
    gen_per = gen_raw.get("per_layer")
    if not isinstance(gen_per, list) or len(gen_per) != 48:
        raise ValueError("generated/replay layer coverage differs")
    if [row.get("layer") for row in gen_per] != list(range(48)):
        raise ValueError("generated/replay layer order differs")
    gen_k = max(_finite(row.get("k_max_abs"), "generated K") for row in gen_per)
    gen_v = max(_finite(row.get("v_max_abs"), "generated V") for row in gen_per)
    gen_metrics = [
        _finite(gen_raw.get("token_logprob_max_abs"), "generated logprob"),
        gen_k, gen_v,
    ]
    if (gen_raw.get("k_max_abs") != gen_k or gen_raw.get("v_max_abs") != gen_v or
            _finite(generated.get("observed_aggregate"), "generated aggregate") !=
            max(gen_metrics) or max(gen_metrics) > 1e-4):
        raise ValueError("generated/replay aggregate differs")

    for stage_name, fields in (
        ("snapshot_rebuild_identity", ("logits_max_abs", "k_max_abs", "v_max_abs")),
        ("physical_causal_mask_identity", ("logits_max_abs", "k_max_abs", "v_max_abs")),
        ("future_mutation_identity", ("earlier_logits_max_abs", "earlier_cache_max_abs")),
    ):
        stage = _required_pass_stage(gates, stage_name)
        values = [_finite((stage.get("raw") or {}).get(field),
                          f"{stage_name}.{field}") for field in fields]
        stage_raw = stage.get("raw") or {}
        per_layer = stage_raw.get("per_layer")
        if not isinstance(per_layer, list) or len(per_layer) != 48 or \
                [row.get("layer") for row in per_layer] != list(range(48)):
            raise ValueError(f"{stage_name} per-layer coverage differs")
        per_k = max(_finite(row.get("k_max_abs"), f"{stage_name}.K")
                    for row in per_layer)
        per_v = max(_finite(row.get("v_max_abs"), f"{stage_name}.V")
                    for row in per_layer)
        if stage_name in {
                "snapshot_rebuild_identity", "physical_causal_mask_identity"}:
            if (values[1] != per_k or values[2] != per_v):
                raise ValueError(f"{stage_name} per-layer aggregate differs")
        elif values[1] != max(per_k, per_v):
            raise ValueError("future mutation per-layer aggregate differs")
        if (stage.get("observed_coverage") != 1 or
                _finite(stage.get("observed_aggregate"),
                        f"{stage_name}.aggregate") != max(values) or
                max(values) > 1e-4):
            raise ValueError(f"{stage_name} aggregate differs")

    position = _required_pass_stage(gates, "position_structure")
    pos = position.get("raw") or {}
    for key in ("common_summary_start", "wrong_prefix_length_equal",
                "post_summary_nonempty"):
        if pos.get(key) is not True:
            raise ValueError(f"position structure lacks {key}")
    for key in ("altered_structure_failure_injection",
                "wrong_position_failure_injection"):
        if (pos.get(key) or {}).get("rejected") is not True:
            raise ValueError(f"position structure injection failed: {key}")
    physical = pos.get("physical_cache_positions")
    context_positions = pos.get("context_position_ids")
    prefix_positions = pos.get("prefix_position_ids")
    summary_positions = pos.get("summary_position_ids")
    post_positions = pos.get("post_summary_position_ids")
    source_start = pos.get("source_summary_start")
    physical_start = pos.get("physical_summary_start")
    physical_end = pos.get("physical_summary_end")
    system_end = pos.get("system_end")
    request_start = pos.get("request_logical_start")
    if (not all(isinstance(value, int) for value in (
            source_start, physical_start, physical_end, system_end, request_start)) or
            not 0 < system_end <= request_start < source_start or
            prefix_positions != list(range(system_end)) +
            list(range(request_start, source_start)) or
            physical_start != len(prefix_positions) or
            not isinstance(summary_positions, list) or not summary_positions or
            summary_positions != list(range(source_start,
                                             source_start + len(summary_positions))) or
            physical_end != physical_start + len(summary_positions) or
            not isinstance(post_positions, list) or
            post_positions != list(range(
                source_start + len(summary_positions),
                source_start + len(summary_positions) + len(post_positions))) or
            context_positions != prefix_positions + summary_positions + post_positions or
            pos.get("logical_next_position") != context_positions[-1] + 1 or
            pos.get("logical_gap") != source_start - physical_start):
        raise ValueError("position structure schedule evidence differs")
    if (not isinstance(physical, list) or not physical or
            physical != list(range(len(context_positions)))):
        raise ValueError("physical cache positions are not contiguous")
    correct_prefix = pos.get("correct_prefix_ids")
    wrong_prefix = pos.get("wrong_prefix_ids")
    structural = pos.get("wrong_structural_position_ids")
    content = pos.get("wrong_content_position_ids")
    if (not isinstance(correct_prefix, list) or not isinstance(wrong_prefix, list) or
            len(correct_prefix) != len(wrong_prefix) or
            pos.get("correct_prefix_sha256") !=
            _sha256_ints(correct_prefix, "position.correct") or
            pos.get("wrong_prefix_sha256") !=
            _sha256_ints(wrong_prefix, "position.wrong") or
            not isinstance(structural, list) or not isinstance(content, list) or
            pos.get("wrong_structural_positions") != len(structural) or
            pos.get("wrong_content_positions") != len(content) or
            pos.get("wrong_structural_positions_sha256") !=
            _sha256_ints(structural, "position.structural") or
            pos.get("wrong_content_positions_sha256") !=
            _sha256_ints(content, "position.content") or
            set(structural) & set(content) or
            sorted(structural + content) != list(range(len(correct_prefix))) or
            any(correct_prefix[i] != wrong_prefix[i] for i in structural) or
            not any(correct_prefix[i] != wrong_prefix[i] for i in content)):
        raise ValueError("position wrong-source evidence differs")
    for values_key, hash_key in (
            ("context_ids", "context_ids_sha256"),
            ("summary_ids", "summary_ids_sha256")):
        if pos.get(hash_key) != _sha256_ints(
                pos.get(values_key), f"position.{values_key}"):
            raise ValueError(f"position structure lacks exact {hash_key}")

    intervention = _required_pass_stage(gates, "intervention_propagation")
    inter = intervention.get("raw") or {}
    for key in ("fresh_self_replacement",
                "correct_insert_and_non_summary_preservation",
                "wrong_insert_and_non_summary_preservation",
                "per_arm_fork_at_summary_boundary",
                "independently_recomputed_identical_tail_lengths",
                "downstream_sensitivity", "recomputed_tail_changed"):
        if inter.get(key) is not True:
            raise ValueError(f"intervention stage lacks {key}")
    if (inter.get("pre_tailed_failure_injection") or {}).get("rejected") is not True:
        raise ValueError("pre-tailed failure injection did not reject")
    if not isinstance(inter.get("sensitivity_attempts"), list) or \
            not inter["sensitivity_attempts"]:
        raise ValueError("intervention sensitivity attempts absent")
    sensitivity = False
    tail_changed = False
    for index, attempt in enumerate(inter["sensitivity_attempts"]):
        if not isinstance(attempt, dict):
            raise ValueError("intervention sensitivity attempt is malformed")
        epsilon = _finite(attempt.get("epsilon"), f"intervention.epsilon[{index}]")
        logits_change = _finite(
            attempt.get("fixed_continuation_logits_max_abs"),
            f"intervention.logits_change[{index}]")
        tail_change = _finite(
            attempt.get("recomputed_post_summary_kv_max_abs"),
            f"intervention.tail_change[{index}]")
        if epsilon not in (0.1, 0.3, 1.0, 3.0) or \
                logits_change < 0 or tail_change < 0:
            raise ValueError("intervention sensitivity attempt values differ")
        sensitivity |= logits_change > 1e-4
        tail_changed |= tail_change > 0
    if (inter.get("downstream_sensitivity") is not sensitivity or
            inter.get("recomputed_tail_changed") is not tail_changed or
            not sensitivity or not tail_changed):
        raise ValueError("intervention sensitivity recomputation differs")
    span = inter.get("summary_span") or {}
    start, end = span.get("start"), span.get("end")
    boundary_lengths = inter.get("boundary_lengths") or {}
    full_lengths = inter.get("full_lengths") or {}
    if (not isinstance(start, int) or not isinstance(end, int) or
            not 0 <= start < end or
            set(boundary_lengths) != {"fresh", "self", "correct", "wrong"} or
            len(set(boundary_lengths.values())) != 1 or
            next(iter(boundary_lengths.values())) != end or
            set(full_lengths) != {"fresh", "correct", "wrong"} or
            len(set(full_lengths.values())) != 1 or
            next(iter(full_lengths.values())) <= end):
        raise ValueError("intervention span/length evidence differs")
    hashes = inter.get("hashes") or {}
    required_hashes = {
        "fresh_boundary", "self_boundary", "correct_boundary", "wrong_boundary",
        "fresh_before_summary", "correct_before_summary", "wrong_before_summary",
        "correct_source_summary", "wrong_source_summary",
        "correct_inserted_summary", "wrong_inserted_summary",
        "full_fresh", "full_correct", "full_wrong",
        "fresh_post_summary", "correct_post_summary", "wrong_post_summary",
    }
    if set(hashes) != required_hashes:
        raise ValueError("intervention hash evidence field set differs")
    for key in sorted(required_hashes):
        _validate_hash_rows(hashes[key], f"intervention.{key}")
    if (hashes["self_boundary"] != hashes["fresh_boundary"] or
            hashes["correct_before_summary"] != hashes["fresh_before_summary"] or
            hashes["wrong_before_summary"] != hashes["fresh_before_summary"] or
            hashes["correct_inserted_summary"] !=
            hashes["correct_source_summary"] or
            hashes["wrong_inserted_summary"] != hashes["wrong_source_summary"]):
        raise ValueError("intervention bit-exact hash equalities differ")
    if (hashes["correct_post_summary"] == hashes["fresh_post_summary"] and
            hashes["wrong_post_summary"] == hashes["fresh_post_summary"]):
        raise ValueError("intervention recomputed tails show no causal change")

    calibration = _required_pass_stage(gates, "calibration_construction")
    cal = calibration.get("raw") or {}
    if (cal.get("passes") is not True or cal.get("model_forwards") != 0 or
            cal.get("semantic_scoring_performed") is not False or
            calibration.get("observed_coverage") != 2 or
            cal.get("label_coverage") != ["A", "B"]):
        raise ValueError("calibration construction coverage differs")
    variants = cal.get("variants")
    if not isinstance(variants, dict) or set(variants) != {"c10", "c07"}:
        raise ValueError("calibration construction variants differ")
    tokenizer = _validation_tokenizer()
    labels = []
    for cid in ("c10", "c07"):
        row = variants[cid]
        expected_variant = _reconstruct_calibration_variant(tokenizer, cid)
        if row != expected_variant:
            raise ValueError(
                f"calibration source-derived reconstruction differs: {cid}")
        correct = row.get("correct_prefix_ids")
        wrong = row.get("wrong_prefix_ids")
        if (not isinstance(correct, list) or not isinstance(wrong, list) or
                len(correct) != len(wrong) or len(correct) != row.get("prefix_length") or
                row.get("correct_prefix_sha256") !=
                _sha256_ints(correct, f"calibration[{cid}].correct") or
                row.get("wrong_prefix_sha256") !=
                _sha256_ints(wrong, f"calibration[{cid}].wrong")):
            raise ValueError(f"calibration prefix evidence differs: {cid}")
        changed = [index for index, pair in enumerate(zip(correct, wrong))
                   if pair[0] != pair[1]]
        allowed = row.get("allowed_first_record_content_positions")
        structural = row.get("structural_positions")
        if (not changed or row.get("changed_positions") != changed or
                not isinstance(allowed, list) or not set(changed).issubset(allowed) or
                structural != [i for i in range(len(correct)) if i not in changed] or
                row.get("exact_length") is not True or
                row.get("changed_only_first_record_content") is not True or
                row.get("structural_slots_equal") is not True or
                row.get("special_ids_excluded") is not True or
                any(correct[i] in tokenizer.all_special_ids or
                    wrong[i] in tokenizer.all_special_ids for i in changed)):
            raise ValueError(f"calibration structural evidence differs: {cid}")
        summary_ids = row.get("summary_ids")
        if (_sha256_ints(summary_ids, f"calibration[{cid}].summary") !=
                row.get("summary_sha256")):
            raise ValueError(f"calibration summary hash differs: {cid}")
        correct_label, wrong_label = row.get("correct_label"), row.get("wrong_label")
        if {correct_label, wrong_label} != {"A", "B"}:
            raise ValueError(f"calibration labels differ: {cid}")
        labels.append(correct_label)
        targets = row.get("targets") or {}
        expected_correct = f"Label {correct_label}."
        expected_wrong = f"Label {wrong_label}."
        correct_target = tokenizer.encode(expected_correct, add_special_tokens=False)
        wrong_target = tokenizer.encode(expected_wrong, add_special_tokens=False)
        rendered_correct = targets.get("rendered_correct_ids")
        rendered_wrong = targets.get("rendered_wrong_ids")
        if (targets.get("correct_text") != expected_correct or
                targets.get("wrong_text") != expected_wrong or
                targets.get("correct_ids") != correct_target or
                targets.get("wrong_ids") != wrong_target or
                targets.get("equal_token_length") is not True or
                targets.get("token_length") != len(correct_target) or
                len(correct_target) != len(wrong_target) or
                not isinstance(rendered_correct, list) or
                not isinstance(rendered_wrong, list) or
                targets.get("rendered_equal_token_length") is not True or
                targets.get("rendered_token_length") != len(rendered_correct) or
                len(rendered_correct) != len(rendered_wrong)):
            raise ValueError(f"calibration target evidence differs: {cid}")
    if sorted(set(labels)) != ["A", "B"]:
        raise ValueError("calibration label coverage recomputation differs")

    donors = _required_pass_stage(gates, "external_donor_construction")
    donor = donors.get("raw") or {}
    donor_rows = donor.get("rows")
    if (not isinstance(donor_rows, list) or len(donor_rows) != 12 or
            donor.get("status") != "PASS" or donor.get("passes") is not True or
            donor.get("requested_revision") != MODEL_REVISION or
            donor.get("resolved_tokenizer_revision") != MODEL_REVISION or
            donor.get("expected_coverage") != 12 or
            donor.get("observed_coverage") != 12 or
            donor.get("mapping") != WRONG_DONORS or
            donor.get("frozen_order") != list(FROZEN_ORDER) or
            donor.get("n_unique_donor_ids") != 12 or
            donor.get("n_unique_donor_hashes") != 12 or
            donors.get("observed_coverage") != 12):
        raise ValueError("external donor coverage differs")
    donor_unhashed = {key: value for key, value in donor.items()
                      if key != "canonical_payload_sha256"}
    if donor.get("canonical_payload_sha256") != \
            _canonical_json_sha256(donor_unhashed):
        raise ValueError("external donor canonical payload hash differs")
    observed_donor_ids: list[str] = []
    observed_donor_hashes: list[str] = []
    for index, (row, cid) in enumerate(zip(donor_rows, FROZEN_ORDER), 1):
        if (row.get("order_position") != index or row.get("target_id") != cid or
                row.get("donor_id") != WRONG_DONORS[cid] or
                row.get("subject_native") is not False or
                row.get("correct_prefix_tokens") != row.get("wrong_prefix_tokens") or
                row.get("correct_wrong_length_equal") is not True or
                row.get("structural_tokens_equal") is not True or
                row.get("system_request_header_retained_tail_unchanged") is not True or
                row.get("changes_confined_to_declared_content_positions") is not True or
                row.get("replacement_spans_non_overlapping") is not True or
                row.get("replacement_coverage_exact") is not True or
                int(row.get("changed_position_count", 0)) < 1):
            raise ValueError(f"external donor row differs: {cid}")
        donor_id = WRONG_DONORS[cid]
        observed_donor_ids.append(donor_id)
        if (row.get("target_path") != f"data/synthetic/{cid}.json" or
                row.get("donor_path") != f"data/synthetic/{donor_id}.json"):
            raise ValueError(f"external donor source paths differ: {cid}")
        target_doc = donor_doc = None
        if verify_sources:
            if not isinstance(fingerprint, dict) or repo_root is None:
                raise ValueError("strict donor validation lacks fingerprint/root")
            inventory_rows = ((fingerprint.get("input_inventory") or {}).get(
                "files") or [])
            inventory = {item.get("path"): item for item in inventory_rows
                         if isinstance(item, dict)}
            for path_key, hash_key, canonical_key, expected_id in (
                    ("target_path", "target_sha256", "target_canonical_sha256", cid),
                    ("donor_path", "donor_sha256", "donor_canonical_sha256", donor_id)):
                relative = row[path_key]
                path = repo_root / relative
                raw_bytes = path.read_bytes()
                parsed = json.loads(raw_bytes)
                raw_sha = hashlib.sha256(raw_bytes).hexdigest()
                if (str(parsed.get("id")) != expected_id or
                        row.get(hash_key) != raw_sha or
                        row.get(canonical_key) != _canonical_json_sha256(parsed) or
                        (inventory.get(relative) or {}).get("bytes") != len(raw_bytes) or
                        (inventory.get(relative) or {}).get("sha256") != raw_sha):
                    raise ValueError(f"external donor source binding differs: {cid}")
            target_doc = json.loads((repo_root / row["target_path"]).read_bytes())
            donor_doc = json.loads((repo_root / row["donor_path"]).read_bytes())
            donor_author = (donor_doc.get("meta") or {}).get("author")
            if (row.get("donor_recorded_author") != donor_author or
                    donor_author not in EXTERNAL_AUTHORS):
                raise ValueError(f"external donor author differs: {cid}")
            observed_donor_hashes.append(hashlib.sha256(
                (repo_root / row["donor_path"]).read_bytes()).hexdigest())
        correct_ids = row.get("correct_prefix_ids")
        structural = row.get("structural_positions")
        content = row.get("content_positions")
        changed = row.get("changed_positions")
        for values, count_key, hash_key in (
                (correct_ids, "correct_prefix_tokens", "correct_prefix_sha256"),
                (structural, "structural_position_count",
                 "structural_positions_sha256"),
                (content, "content_position_count", "content_positions_sha256"),
                (changed, "changed_position_count", "changed_positions_sha256")):
            if (not isinstance(values, list) or row.get(count_key) != len(values) or
                    _sha256_ints(values, f"donor[{cid}].{count_key}") !=
                    row.get(hash_key)):
                raise ValueError(f"external donor row array differs: {cid}")
        if (set(structural) & set(content) or
                sorted(structural + content) != list(range(len(correct_ids))) or
                not set(changed).issubset(content) or
                any(position < 0 or position >= len(correct_ids)
                    for position in structural + content + changed)):
            raise ValueError(f"external donor position coverage differs: {cid}")
        replacements = row.get("replacements")
        if (not isinstance(replacements, list) or
                row.get("replacement_count") != len(replacements) or
                not replacements):
            raise ValueError(f"external donor replacements differ: {cid}")
        covered: list[int] = []
        tokenizer = _validation_tokenizer()
        special_ids = {int(value) for value in tokenizer.all_special_ids}
        for replacement in replacements:
            if not isinstance(replacement, dict):
                raise ValueError(f"external donor replacement malformed: {cid}")
            start, end = replacement.get("start"), replacement.get("end")
            if (not isinstance(start, int) or not isinstance(end, int) or
                    not 0 <= start < end <= len(correct_ids) or
                    replacement.get("length") != end - start or
                    replacement.get("contains_special_token") is not False or
                    any(int(value) in special_ids for value in
                        (replacement.get("replacement_ids") or []))):
                raise ValueError(f"external donor replacement bounds differ: {cid}")
            for values_key, hash_key in (
                    ("target_ids", "target_ids_sha256"),
                    ("donor_pool_ids", "source_pool_sha256"),
                    ("replacement_ids", "replacement_sha256")):
                values = replacement.get(values_key)
                if (_sha256_ints(values, f"donor[{cid}].{values_key}") !=
                        replacement.get(hash_key)):
                    raise ValueError(f"external donor replacement hash differs: {cid}")
            if (len(replacement["target_ids"]) != end - start or
                    len(replacement["replacement_ids"]) != end - start or
                    replacement["target_ids"] != correct_ids[start:end] or
                    not replacement["donor_pool_ids"] or
                    replacement["replacement_ids"] != [
                        replacement["donor_pool_ids"][i % len(
                            replacement["donor_pool_ids"])]
                        for i in range(end - start)] or
                    replacement.get("cycles") != math.ceil(
                        (end - start) / len(replacement["donor_pool_ids"])) or
                    not isinstance(replacement.get("target_message_index"), int) or
                    not isinstance(replacement.get("donor_message_index"), int) or
                    replacement.get("role") not in {"user", "assistant"}):
                raise ValueError(f"external donor replacement length differs: {cid}")
            covered.extend(range(start, end))
        if (len(covered) != len(set(covered)) or
                sorted(covered) != sorted(content)):
            raise ValueError(f"external donor replacement coverage differs: {cid}")
        reconstructed = list(correct_ids)
        for replacement in replacements:
            reconstructed[replacement["start"]:replacement["end"]] = \
                replacement["replacement_ids"]
        reconstructed_changed = [i for i, pair in enumerate(zip(
            correct_ids, reconstructed)) if pair[0] != pair[1]]
        if (reconstructed_changed != changed or
                _sha256_ints(reconstructed, f"donor[{cid}].wrong") !=
                row.get("wrong_prefix_sha256")):
            raise ValueError(f"external donor reconstructed prefix differs: {cid}")
        if verify_sources:
            expected = _reconstruct_donor_replacements(
                tokenizer, target_doc, donor_doc, cid)
            if (correct_ids != expected["correct_ids"] or
                    structural != expected["structural"] or
                    content != expected["content"] or
                    changed != expected["changed"] or
                    replacements != expected["replacements"] or
                    reconstructed != expected["wrong_ids"]):
                raise ValueError(
                    f"external donor source-derived reconstruction differs: {cid}")

    if verify_sources and (
            len(set(observed_donor_ids)) != 12 or
            set(observed_donor_ids) & set(FROZEN_ORDER) or
            len(set(observed_donor_hashes)) != 12 or
            donor.get("n_unique_donor_ids") != len(set(observed_donor_ids)) or
            donor.get("n_unique_donor_hashes") != len(set(observed_donor_hashes))):
        raise ValueError("external donor independently recomputed uniqueness differs")

    retired = _required_pass_stage(gates, "retired_G_delta")
    retired_raw = retired.get("raw") or {}
    if (retired_raw.get("executed") is not False or
            retired_raw.get("authorizes_run") is not False):
        raise ValueError("G_delta retirement assertion differs")

    if gates.get("passes") is not True or gates.get("status") != "PASS" or \
            gates.get("failures") not in ([], None):
        raise ValueError("aggregate gate verdict differs")


def _validate_complete(root: Path, log_text: str) -> dict[str, Any]:
    manifest = _load(root / "manifest.json")
    _require_identity(manifest, "manifest")
    _require_payload_sha256(manifest, "manifest")
    if (manifest.get("status") != "COMPLETE" or
            manifest.get("phase") != "COMPLETE" or
            manifest.get("technical_only") is not False):
        raise ValueError("complete harvest lacks COMPLETE manifest")
    if manifest.get("resume_probe_verified") is not True:
        raise ValueError("manifest lacks resume_probe_verified=true")
    manifest_fingerprint = manifest.get("fingerprint")
    if not isinstance(manifest_fingerprint, dict):
        raise ValueError("manifest lacks run fingerprint")
    _require_identity(manifest_fingerprint, "manifest fingerprint")
    _require_backend_fingerprint(manifest_fingerprint, "manifest fingerprint")
    if (manifest.get("attention_backend") != ATTENTION_BACKEND or
            manifest.get("attention_backend_fingerprint") !=
            manifest_fingerprint.get("attention_backend_fingerprint")):
        raise ValueError("semantic manifest/backend fingerprint binding differs")
    checks = manifest.get("authorization_checks")
    required_checks = {
        "terminal_payloads_exact", "committed_directory_bytes_exact",
        "result_commit_is_ancestor", "result_commit_on_trunk",
        "harvest_attestation_committed_exact",
        "independent_harvest_revalidation_passed",
        "apparatus_inventory_exact", "static_data_fingerprint_exact",
        "backend_attestation_exact",
    }
    if (not isinstance(checks, dict) or set(checks) != required_checks or
            any(checks[key] is not True for key in required_checks)):
        raise ValueError("semantic authorization checks are not exact PASS")
    authorization = manifest.get("semantic_authorization")
    fingerprint_authorization = manifest_fingerprint.get(
        "semantic_authorization")
    if not isinstance(authorization, dict) or \
            not isinstance(fingerprint_authorization, dict):
        raise ValueError("semantic authorization binding is absent")
    harvest = authorization.get("harvest") or {}
    expected_binding = {
        "technical_result_commit": authorization.get("result_commit"),
        "technical_run_dir": authorization.get("run_dir"),
        "gate_payload_sha256": authorization.get("gate_payload_sha256"),
        "raw_sha256": authorization.get("raw_sha256"),
        "harvest_path": harvest.get("path"),
        "harvest_raw_sha256": harvest.get("raw_sha256"),
        "harvest_payload_sha256": harvest.get("payload_sha256"),
        "apparatus_aggregate_sha256": (
            authorization.get("apparatus_inventory") or {}).get(
                "aggregate_sha256"),
        "backend_exact": True,
        "static_fingerprint_exact": True,
    }
    if fingerprint_authorization != expected_binding:
        raise ValueError("semantic fingerprint/prior authorization binding differs")
    commit = expected_binding.get("technical_result_commit")
    if (not isinstance(commit, str) or len(commit) not in (40, 64) or
            any(char not in "0123456789abcdef" for char in commit.lower())):
        raise ValueError("semantic authorization lacks exact technical_result_commit")
    for key in (
            "gate_payload_sha256",
            "harvest_raw_sha256", "harvest_payload_sha256",
            "apparatus_aggregate_sha256"):
        value = expected_binding.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"semantic authorization lacks exact {key}")
    _validate_terminal_integrity(root, "PASS")
    probe = _load(root / "resume_probe.json")
    _require_identity(probe, "resume probe")
    _require_payload_sha256(probe, "resume probe")
    if probe.get("resume_probe_verified") is not True:
        raise ValueError("resume probe was not verified on restart")
    if (root / "production_kernel_gate.json").exists():
        raise ValueError("semantic run improperly contains a local technical gate")

    paths = _checkpoint_paths(root)
    if len(paths) not in (6, 12):
        raise ValueError(f"complete harvest has invalid checkpoint N={len(paths)}")
    if manifest.get("n_conversations") != len(paths):
        raise ValueError("semantic manifest/checkpoint count differs")
    seen_positions: list[int] = []
    seen_ids: list[str] = []
    for path in paths:
        doc = _load(path)
        _require_payload_sha256(doc, path.name)
        _validate_checkpoint(
            doc, path, scored=True,
            expected_fingerprint=manifest_fingerprint)
        position = doc.get("order_position")
        if not isinstance(position, int):
            raise ValueError(f"{path.name} lacks integer order_position")
        seen_positions.append(position)
        seen_ids.append(doc.get("conversation_id"))
    if sorted(seen_positions) != list(range(1, len(paths) + 1)):
        raise ValueError(f"checkpoint positions are not contiguous: {seen_positions}")
    if seen_ids != list(FROZEN_ORDER[:len(paths)]):
        raise ValueError(f"checkpoint frozen ID order differs: {seen_ids}")
    if "COHERENT_STATE_JOB_DONE" not in log_text:
        raise ValueError("complete harvest job log lacks completion marker")
    return {
        "status": "PASS",
        "mode": "complete",
        "n_scored": len(paths),
        "resume_probe_verified": True,
        "production_gate": "PASS",
    }


def _validate_technical(root: Path, log_text: str) -> dict[str, Any]:
    manifest = _load(root / "manifest.json")
    _require_identity(manifest, "technical manifest")
    _require_payload_sha256(manifest, "technical manifest")
    if (manifest.get("status") != "TECHNICAL_PASS" or
            manifest.get("phase") != "TECHNICAL_COMPLETE"):
        raise ValueError("technical harvest lacks terminal TECHNICAL_PASS manifest")
    if (manifest.get("technical_only") is not True or
            manifest.get("model") != MODEL_ID or
            manifest.get("revision") != MODEL_REVISION or
            manifest.get("dtype") != PARAMETER_DTYPE or
            manifest.get("attention_backend") != ATTENTION_BACKEND):
        raise ValueError("technical manifest subject or mode differs")
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("technical manifest lacks run fingerprint")
    _require_identity(fingerprint, "technical manifest fingerprint")
    _require_backend_fingerprint(fingerprint, "technical manifest fingerprint")
    integrity = _validate_terminal_integrity(root, "PASS")
    gate = _validate_gate(root, "PASS")
    if gate.get("technical_only") is not True:
        raise ValueError("technical gate is not marked technical_only")
    if (manifest.get("production_kernel_gate_payload_sha256") !=
            gate.get("payload_sha256")):
        raise ValueError("technical manifest/gate payload binding differs")
    if manifest.get("production_kernel_gate_path") != \
            "production_kernel_gate.json":
        raise ValueError("technical manifest canonical gate path differs")
    unique = manifest.get("production_kernel_gate_attempt_path")
    if (not isinstance(unique, str) or Path(unique).name != unique or
            not unique.startswith("production_kernel_gate_") or
            unique == "production_kernel_gate.json"):
        raise ValueError("technical manifest unique gate path differs")
    for field in (
            "fingerprint", "fingerprint_static", "apparatus_inventory", "model", "revision",
            "dtype", "attention_backend", "attention_backend_fingerprint",
            "geometry", "context_limit"):
        if manifest.get(field) != gate.get(field):
            raise ValueError(f"technical manifest/gate {field} binding differs")
    if ((gate.get("fingerprint_static") or {}).get("apparatus_inventory") !=
            gate.get("apparatus_inventory")):
        raise ValueError("technical top-level/static apparatus binding differs")
    expanded_gates = _resolve_heavy_stages(root, gate)
    _validate_v7_pass_gates(
        expanded_gates, fingerprint=gate.get("fingerprint"),
        static_fingerprint=gate.get("fingerprint_static"),
        repo_root=Path(__file__).resolve().parent.parent,
        verify_sources=True)
    forbidden = _find_forbidden_semantic_fields(gate)
    if forbidden:
        raise ValueError(
            f"technical-only gate contains semantic outcome fields {forbidden}")
    if _checkpoint_paths(root):
        raise ValueError("technical-only harvest contains conversation checkpoints")
    if "COHERENT_STATE_TECHNICAL_DONE" not in log_text:
        raise ValueError("technical harvest job log lacks completion marker")
    if "CHECKPOINT_SCORED" in log_text or "PHASE RENDER" in log_text:
        raise ValueError("technical-only log contains semantic execution markers")
    return {
        "status": "PASS",
        "mode": "technical",
        "n_scored": 0,
        "production_gate": "PASS",
        "attention_backend": ATTENTION_BACKEND,
        "raw_sha256": {
            "gate": _raw_file_sha256(root / "production_kernel_gate.json"),
            "manifest": _raw_file_sha256(root / "manifest.json"),
            "index": integrity["index_raw_sha256"],
            "receipt": integrity["receipt_raw_sha256"],
        },
    }


def _validate_failure(root: Path, log_text: str) -> dict[str, Any]:
    failure = _load(root / "failure.json") if (root / "failure.json").exists() else {}
    evidence = FAILURE_RE.search(log_text) or failure.get("error") or failure.get("traceback")
    if not evidence:
        raise ValueError("failure harvest lacks meaningful failure evidence")

    failure_fingerprint = None
    failure_manifest = None
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = _load(manifest_path)
        _require_identity(manifest, "failure manifest")
        failure_manifest = manifest
        candidate = manifest.get("fingerprint")
        if candidate is not None and not isinstance(candidate, dict):
            raise ValueError("failure manifest fingerprint is malformed")
        failure_fingerprint = candidate

    model_ready = "MODEL_READY" in log_text
    gate_path = root / "production_kernel_gate.json"
    gate_status = None
    terminal_integrity = None
    has_index = (root / "terminal_artifact_index.json").exists()
    has_receipt = (root / "terminal_receipt.json").exists()
    if has_index != has_receipt:
        raise ValueError("failure harvest has a partial terminal envelope")
    if gate_path.exists():
        gate = _load(gate_path)
        gate_status = gate.get("status")
        if gate_status == "FAIL":
            terminal_integrity = _validate_terminal_integrity(root, "FAIL")
            _validate_gate(root, "FAIL")
        elif gate_status == "PASS":
            terminal_integrity = _validate_terminal_integrity(root, "PASS")
            _validate_gate(root, "PASS")
        else:
            raise ValueError(f"failure harvest has incomplete gate status {gate_status!r}")
    elif model_ready:
        raise ValueError("post-MODEL_READY failure lacks complete production gate")
    elif has_index:
        terminal_integrity = _validate_terminal_integrity(root, "FAIL")

    paths = _checkpoint_paths(root)
    scored = 0
    voids = 0
    for path in paths:
        doc = _load(path)
        if doc.get("status") == "scored":
            if failure_fingerprint is None:
                raise ValueError(
                    "failure harvest has scored checkpoint but no manifest fingerprint")
            _validate_checkpoint(
                doc, path, scored=True,
                expected_fingerprint=failure_fingerprint)
            scored += 1
        elif doc.get("status") == "void":
            _validate_checkpoint(doc, path, scored=False)
            voids += 1
        else:
            raise ValueError(
                f"failure harvest contains non-terminal checkpoint {path.name}")

    technical_only = bool((failure_manifest or {}).get("technical_only"))
    if gate_status == "PASS" and voids < 1 and not technical_only:
        raise ValueError("post-gate case failure lacks a void checkpoint")
    if model_ready and terminal_integrity is None:
        raise ValueError("post-MODEL_READY failure lacks verified terminal integrity")
    if model_ready and gate_status not in ("PASS", "FAIL"):
        raise ValueError("post-MODEL_READY failure lacks terminal gate evidence")
    return {
        "status": "PASS",
        "mode": "failure",
        "model_ready": model_ready,
        "production_gate": gate_status,
        "terminal_integrity_verified": terminal_integrity is not None,
        "technical_pass_rejected_at_harvest": bool(
            gate_status == "PASS" and technical_only and voids == 0),
        "n_scored": scored,
        "n_void": voids,
    }


def validate(root: Path, mode: str) -> dict[str, Any]:
    log = root / "job.log"
    if not log.exists() or not log.stat().st_size:
        raise ValueError("job log missing or empty")
    log_text = log.read_text(errors="replace")
    # Parse every top-level JSON artifact before making any mode-specific claim.
    for path in root.glob("*.json"):
        _load(path)
    if mode == "complete":
        out = _validate_complete(root, log_text)
    elif mode == "technical":
        out = _validate_technical(root, log_text)
    else:
        out = _validate_failure(root, log_text)
    out["json_files"] = len(list(root.glob("*.json")))
    out["schema"] = SCHEMA
    out["amendment_id"] = AMENDMENT_ID
    out["design_id"] = DESIGN_ID
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("mode", choices=("technical", "complete", "failure"))
    output = ap.add_mutually_exclusive_group()
    output.add_argument("--read-only", action="store_true")
    output.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = validate(args.root, args.mode)
    repo = Path(__file__).resolve().parent.parent
    if args.read_only:
        sibling = Path(f"{args.root}.harvest_validation.json")
        if sibling.exists():
            attestation = _load(sibling)
            _require_identity(attestation, "harvest attestation")
            _require_payload_sha256(attestation, "harvest attestation")
            for key, value in result.items():
                if attestation.get(key) != value:
                    raise ValueError(
                        f"harvest attestation field differs: {key}")
            try:
                run_relative = args.root.resolve().relative_to(repo).as_posix()
            except ValueError as exc:
                raise ValueError("harvest run directory is outside repository") from exc
            validator = attestation.get("validator") or {}
            if (attestation.get("status") != "PASS" or
                    attestation.get("run_dir") != run_relative or
                    validator.get("path") != "scripts/validate_coherent_harvest.py" or
                    validator.get("sha256") !=
                    _raw_file_sha256(Path(__file__).resolve()) or
                    validator.get("version") != "coherent-harvest-v7"):
                raise ValueError("harvest attestation provenance differs")
    if args.output is not None:
        if args.output.exists():
            raise ValueError(f"refusing to overwrite harvest attestation: {args.output}")
        try:
            run_relative = args.root.resolve().relative_to(repo).as_posix()
        except ValueError as exc:
            raise ValueError("harvest run directory is outside repository") from exc
        attestation = {
            **result,
            "run_dir": run_relative,
            "validator": {
                "path": "scripts/validate_coherent_harvest.py",
                "sha256": _raw_file_sha256(Path(__file__).resolve()),
                "version": "coherent-harvest-v7",
            },
        }
        attestation["payload_sha256"] = _canonical_payload_sha256(attestation)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        payload = (json.dumps(
            attestation, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False) + "\n").encode()
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(args.output)
        if _load(args.output) != attestation:
            raise ValueError("harvest attestation read-back mismatch")
        result = attestation
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
