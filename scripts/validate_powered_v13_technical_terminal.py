#!/usr/bin/env python3
"""Independent-process validator for one v13 technical preterminal chain.

This script intentionally does not import ``powered_v13_store``.  It
recomputes the exact technical chain, external artifact hashes, identity, and
terminal evidence with a separate implementation, then exclusive-writes the
receipt consumed by the store's terminal append.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Callable, Mapping, Sequence


DESIGN_ID = "coherent-state-powered-successor-v13"
RECORD_SCHEMA = "coherent-state-powered-successor-v13-case-record-v1"
LOCK_SCHEMA = "coherent-state-powered-successor-v13-active-lock-v3"
TERMINAL_EVIDENCE_SCHEMA = (
    "coherent-state-powered-successor-v13-terminal-evidence-v1")
TERMINAL_VALIDATION_SCHEMA = (
    "coherent-state-powered-successor-v13-independent-terminal-validation-v1")
VALIDATOR_ID = "powered-v13-independent-terminal-validator-v1"
N_SCHEDULE = "role_native_q1_replay"
HISTORIES = ("F", "C", "W")
ARMS = ("FF", "CC", "WW", "FC", "FW", "VP")
PRETERMINAL_KINDS = (
    "STARTED", "FOUNDATION_LOAD", *(f"ARM_{arm}" for arm in ARMS),
)
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_UTC = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_START_TOKEN = re.compile(
    r"linux-procfs-v1:[0-9a-f-]{36}:[1-9][0-9]*")


class IndependentTerminalValidationError(RuntimeError):
    """The preterminal chain does not independently recompute."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentTerminalValidationError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise IndependentTerminalValidationError(
            f"value is not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON duplicates key {key!r}")
        result[key] = value
    return result


def _strict_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    _require(not path.is_symlink() and path.is_file(),
             f"{label} is absent or symlinked")
    raw = path.read_bytes()
    _require(0 < len(raw) <= MAX_JSON_BYTES, f"{label} byte bound differs")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IndependentTerminalValidationError(
            f"{label} is not strict UTF-8 JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root is not an object")
    _require(raw == canonical_json_bytes(value) + b"\n",
             f"{label} is not canonical newline-terminated JSON")
    return value, raw


def _sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None,
             f"{label} is not lowercase SHA-256")
    return value


def _plain_int(value: object, label: str, minimum: int,
               maximum: int) -> int:
    _require(type(value) is int and minimum <= value <= maximum,
             f"{label} lies outside {minimum}..{maximum}")
    return value


def _identity(value: object) -> dict[str, Any]:
    fields = {
        "design_id", "release_sha256", "runtime_fingerprint_sha256",
        "primary_batch_id", "gpu_uuid", "case_id", "render_id", "schedule",
        "mode",
    }
    _require(isinstance(value, Mapping) and set(value) == fields,
             "technical identity field set differs")
    _require(value.get("design_id") == DESIGN_ID
             and value.get("mode") == "technical"
             and value.get("schedule") == N_SCHEDULE
             and value.get("case_id") in ("technical_e01", "technical_long")
             and value.get("render_id") in ("r1", "r2"),
             "technical identity design/mode/case/render/schedule differs")
    _sha(value.get("release_sha256"), "release hash")
    _sha(value.get("runtime_fingerprint_sha256"), "runtime hash")
    for field in ("primary_batch_id", "case_id"):
        _require(isinstance(value.get(field), str)
                 and _SLUG.fullmatch(value[field]) is not None,
                 f"identity {field} is invalid")
    _require(isinstance(value.get("gpu_uuid"), str)
             and bool(value["gpu_uuid"].strip()), "GPU UUID is empty")
    return dict(value)


def identity_sha256(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(_identity(value)))


def _bounds(value: object, completed_arms: Sequence[str]) -> dict[str, Any]:
    fields = {
        "active_case_count", "active_render_count", "history_ids",
        "completed_arm_ids", "live_token_count", "selected_row_count",
        "layer_count", "case_deadline_seconds",
    }
    _require(isinstance(value, Mapping) and set(value) == fields,
             "record bounds field set differs")
    _require(value.get("active_case_count") == 1
             and value.get("active_render_count") == 1
             and value.get("history_ids") == list(HISTORIES)
             and value.get("completed_arm_ids") == list(completed_arms)
             and value.get("layer_count") == 48,
             "record one-live/history/arm/layer bound differs")
    _plain_int(value.get("live_token_count"), "live token count", 0, 7000)
    _plain_int(value.get("selected_row_count"), "selected row count", 0, 256)
    _plain_int(value.get("case_deadline_seconds"), "case deadline", 1, 7200)
    return dict(value)


def _safe_artifact(root: Path, relative: object, label: str) -> Path:
    _require(isinstance(relative, str) and bool(relative),
             f"{label} path is empty")
    pure = PurePosixPath(relative)
    _require(not pure.is_absolute() and ".." not in pure.parts
             and "." not in pure.parts and "" not in pure.parts,
             f"{label} path escapes root")
    root_resolved = root.resolve(strict=True)
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), f"{label} path contains a symlink")
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise IndependentTerminalValidationError(
            f"{label} path is absent or escapes root") from exc
    _require(resolved.is_file(), f"{label} is not a regular file")
    return resolved


def _bindings(root: Path, value: object,
              expected_kinds: Sequence[str]) -> list[dict[str, Any]]:
    _require(isinstance(value, list), "artifact bindings are not a list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        _require(isinstance(row, Mapping) and set(row) == {
            "kind", "path", "sha256", "size_bytes",
        }, f"artifact binding {index} fields differ")
        _require(isinstance(row.get("kind"), str) and bool(row["kind"]),
                 f"artifact binding {index} kind differs")
        path = _safe_artifact(root, row.get("path"), f"artifact {index}")
        digest = _sha(row.get("sha256"), f"artifact {index} hash")
        size = _plain_int(row.get("size_bytes"), f"artifact {index} size",
                          1, MAX_ARTIFACT_BYTES)
        _require(path.stat().st_size == size and file_sha256(path) == digest,
                 f"artifact {index} external bytes differ")
        rows.append(dict(row))
    _require(len(rows) == len(expected_kinds)
             and {row["kind"] for row in rows} == set(expected_kinds),
             "artifact kind set differs")
    _require([row["path"] for row in rows] == sorted(
        row["path"] for row in rows),
        "artifact bindings are not in canonical path order")
    _require(len({row["path"] for row in rows}) == len(rows),
             "artifact binding path repeats")
    return rows


def _payload(record_kind: str, value: object,
             bindings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{record_kind} payload is not an object")
    by_kind = {row["kind"]: row for row in bindings}
    if record_kind == "STARTED":
        _require(value == {"status": "STARTED"} and not bindings,
                 "STARTED evidence differs")
        return dict(value)
    if record_kind == "FOUNDATION_LOAD":
        fields = {
            "foundation_bundle_sha256", "foundation_descriptor_sha256",
            "foundation_receipt_sha256",
        }
        mapping = {
            "foundation_bundle_sha256": "foundation_bundle",
            "foundation_descriptor_sha256": "foundation_descriptor",
            "foundation_receipt_sha256": "foundation_receipt",
        }
    else:
        arm = record_kind.removeprefix("ARM_")
        _require(arm in ARMS, "technical record arm differs")
        fields = {"arm_id", "checkpoint_sha256", "artifact_sha256"}
        _require(value.get("arm_id") == arm, "arm payload ID differs")
        mapping = {
            "checkpoint_sha256": f"arm_{arm}_checkpoint",
            "artifact_sha256": f"arm_{arm}_artifact",
        }
    _require(set(value) == fields and set(by_kind) == set(mapping.values()),
             f"{record_kind} payload/artifact fields differ")
    for field, kind in mapping.items():
        _require(_sha(value.get(field), field) == by_kind[kind]["sha256"],
                 f"{record_kind} {kind} hash differs")
    return dict(value)


def _expected_artifact_kinds(record_kind: str) -> tuple[str, ...]:
    if record_kind == "STARTED":
        return ()
    if record_kind == "FOUNDATION_LOAD":
        return tuple(sorted((
            "foundation_bundle", "foundation_descriptor", "foundation_receipt")))
    arm = record_kind.removeprefix("ARM_")
    return tuple(sorted((f"arm_{arm}_artifact", f"arm_{arm}_checkpoint")))


def validate_preterminal_chain(
    *, root: Path, identity_path: Path, terminal_evidence_path: Path,
    runner_pid: int,
) -> dict[str, Any]:
    root = Path(root)
    _require(not root.is_symlink() and root.is_dir(),
             "store root is absent or symlinked")
    _require(os.getpid() != runner_pid,
             "validator is not independent of the runner process")
    identity_document, _identity_raw = _strict_object(
        identity_path, "technical identity")
    identity = _identity(identity_document)
    digest = identity_sha256(identity)

    lock, _lock_raw = _strict_object(root / "active-case.json", "active lock")
    _require(set(lock) == {
        "schema", "design_id", "identity", "identity_sha256", "case_dir",
        "pid", "process_start_token", "attempt_id", "created_utc",
    } and lock.get("schema") == LOCK_SCHEMA
        and lock.get("design_id") == DESIGN_ID
        and lock.get("identity") == identity
        and lock.get("identity_sha256") == digest
        and lock.get("case_dir") == f"active/{digest}"
        and lock.get("pid") == runner_pid
        and isinstance(lock.get("process_start_token"), str)
        and _START_TOKEN.fullmatch(lock["process_start_token"]) is not None
        and isinstance(lock.get("attempt_id"), str)
        and re.fullmatch(r"[0-9a-f]{32}", lock["attempt_id"]) is not None
        and isinstance(lock.get("created_utc"), str)
        and _UTC.fullmatch(lock["created_utc"]) is not None,
        "active lock identity/process binding differs")

    case_relative = PurePosixPath(lock["case_dir"])
    _require(not case_relative.is_absolute()
             and all(part not in {"", ".", ".."}
                     for part in case_relative.parts),
             "active case path is unsafe")
    case_directory = root
    for part in case_relative.parts:
        case_directory = case_directory / part
        _require(not case_directory.is_symlink(),
                 "active case path contains a symlink")
    try:
        case_directory.resolve(strict=True).relative_to(
            root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise IndependentTerminalValidationError(
            "active case path is absent or escapes root") from exc
    record_directory = case_directory / "records"
    _require(not record_directory.is_symlink() and record_directory.is_dir(),
             "active record directory differs")
    paths = sorted(record_directory.iterdir())
    _require(len(paths) == len(PRETERMINAL_KINDS),
             "technical preterminal record count differs")
    chain_rows: list[dict[str, str]] = []
    seen_artifact_paths: set[str] = set()
    prior: Path | None = None
    for sequence, (path, expected_kind) in enumerate(
            zip(paths, PRETERMINAL_KINDS, strict=True)):
        record, _raw = _strict_object(path, f"record {sequence}")
        _require(set(record) == {
            "schema", "design_id", "record_kind", "sequence", "created_utc",
            "identity", "identity_sha256", "bounds", "payload",
            "artifact_bindings", "previous_record",
        } and record.get("schema") == RECORD_SCHEMA
            and record.get("design_id") == DESIGN_ID
            and record.get("record_kind") == expected_kind
            and record.get("sequence") == sequence
            and path.name == f"{sequence:03d}_{expected_kind}.json"
            and record.get("identity") == identity
            and record.get("identity_sha256") == digest
            and isinstance(record.get("created_utc"), str)
            and _UTC.fullmatch(record["created_utc"]) is not None,
            f"record {sequence} identity/schema/order differs")
        completed = () if sequence < 2 else ARMS[:sequence - 1]
        _bounds(record.get("bounds"), completed)
        expected_parent = None if prior is None else {
            "name": prior.name, "sha256": file_sha256(prior),
        }
        _require(record.get("previous_record") == expected_parent,
                 f"record {sequence} parent binding differs")
        bindings = _bindings(
            root, record.get("artifact_bindings"),
            _expected_artifact_kinds(expected_kind))
        paths_now = {row["path"] for row in bindings}
        _require(not (seen_artifact_paths & paths_now),
                 "artifact path is reused across records")
        seen_artifact_paths.update(paths_now)
        _payload(expected_kind, record.get("payload"), bindings)
        chain_rows.append({"name": path.name, "sha256": file_sha256(path)})
        prior = path

    chain_sha256 = sha256_bytes(canonical_json_bytes(chain_rows))
    supplied_evidence = Path(terminal_evidence_path)
    _require(not supplied_evidence.is_symlink(),
             "terminal evidence is symlinked")
    lexical_evidence = (supplied_evidence if supplied_evidence.is_absolute()
                        else Path.cwd() / supplied_evidence)
    try:
        evidence_relative = lexical_evidence.absolute().relative_to(
            root.absolute()).as_posix()
    except ValueError as exc:
        raise IndependentTerminalValidationError(
            "terminal evidence escapes store root") from exc
    evidence_path = _safe_artifact(
        root, evidence_relative, "terminal evidence")
    evidence, evidence_raw = _strict_object(
        evidence_path, "technical terminal evidence")
    expected_evidence = {
        "schema": TERMINAL_EVIDENCE_SCHEMA,
        "design_id": DESIGN_ID,
        "identity_sha256": digest,
        "terminal_kind": "TERMINAL_TECHNICAL",
        "evidence_chain_sha256": chain_sha256,
        "outcome": "TECHNICAL_COMPLETE",
        "accepted_attempt_index": None,
        "rejection_codes": [],
        "completed_arm_ids": list(ARMS),
    }
    _require(evidence == expected_evidence,
             "technical terminal evidence does not recompute")
    return {
        "identity": identity,
        "identity_sha256": digest,
        "evidence_chain_sha256": chain_sha256,
        "terminal_evidence_sha256": sha256_bytes(evidence_raw),
        "record_count": len(paths),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_and_write(
    *, root: Path, identity_path: Path, terminal_evidence_path: Path,
    output_path: Path, runner_pid: int,
    now: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    result = validate_preterminal_chain(
        root=root, identity_path=identity_path,
        terminal_evidence_path=terminal_evidence_path,
        runner_pid=runner_pid)
    validated_utc = now()
    _require(isinstance(validated_utc, str)
             and _UTC.fullmatch(validated_utc) is not None,
             "validator UTC differs")
    receipt = {
        "schema": TERMINAL_VALIDATION_SCHEMA,
        "design_id": DESIGN_ID,
        "identity_sha256": result["identity_sha256"],
        "terminal_kind": "TERMINAL_TECHNICAL",
        "evidence_chain_sha256": result["evidence_chain_sha256"],
        "terminal_evidence_sha256": result["terminal_evidence_sha256"],
        "validator_id": VALIDATOR_ID,
        "validation_status": "PASS",
        "validated_utc": validated_utc,
    }
    output_path = Path(output_path)
    _require(not output_path.exists() and not output_path.is_symlink(),
             "validator receipt output already exists")
    try:
        output_path.resolve().relative_to(Path(root).resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise IndependentTerminalValidationError(
            "validator receipt output escapes store root") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_json_bytes(receipt) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--store-root", required=True, type=Path)
    result.add_argument("--identity", required=True, type=Path)
    result.add_argument("--terminal-evidence", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--runner-pid", required=True, type=int)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        receipt = validate_and_write(
            root=args.store_root, identity_path=args.identity,
            terminal_evidence_path=args.terminal_evidence,
            output_path=args.output, runner_pid=args.runner_pid)
    except (IndependentTerminalValidationError, OSError) as exc:
        print(f"INDEPENDENT TECHNICAL VALIDATION ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": receipt["validation_status"],
        "identity_sha256": receipt["identity_sha256"],
        "evidence_chain_sha256": receipt["evidence_chain_sha256"],
        "receipt_sha256": file_sha256(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
