"""Append-only, one-live-case persistence for powered successor v13.

The store retains no model object or tensor in global memory.  It serializes an
immutable per-case record chain, one exact active-process lock, terminal-only
index bindings, and loss-preserving quarantine.  Partial chains are never
promoted or resumed as scientific cases.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from functools import wraps
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

from powered_v13_schema import DESIGN_ID, N_SCHEDULE, P_SCHEDULE, PRIMARY_ARMS


LOCK_SCHEMA = "coherent-state-powered-successor-v13-active-lock-v2"
RECORD_SCHEMA = "coherent-state-powered-successor-v13-case-record-v1"
INDEX_SCHEMA = "coherent-state-powered-successor-v13-session-index-v1"
QUARANTINE_SCHEMA = "coherent-state-powered-successor-v13-quarantine-v2"
TERMINAL_EVIDENCE_SCHEMA = (
    "coherent-state-powered-successor-v13-terminal-evidence-v1")
TERMINAL_VALIDATION_SCHEMA = (
    "coherent-state-powered-successor-v13-independent-terminal-validation-v1")
TERMINAL_VALIDATOR_ID = "powered-v13-independent-terminal-validator-v1"
MODES = ("phase_a", "treatment", "technical")
HISTORIES = ("F", "C", "W")
SCHEDULES = (N_SCHEDULE, P_SCHEDULE)
MAX_LIVE_TOKENS = 7000
MAX_SELECTED_ROWS = 256
REQUIRED_LAYERS = 48
MAX_CASE_DEADLINE_SECONDS = 7200
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_BOUND_ARTIFACT_BYTES = 256 * 1024 * 1024
ACTIVE_LOCK_NAME = "active-case.json"
ARCHIVED_LOCK_NAME = "ACTIVE_LOCK.json"
QUARANTINE_RECORD_NAME = "QUARANTINE.json"
METADATA_LOCK_NAME = ".powered-v13-store.lock"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_BOOT_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_LINUX_START_TOKEN = re.compile(
    r"linux-procfs-v1:"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r":([1-9][0-9]*)")


PHASE_A_ACCEPTED_SEQUENCE = (
    "STARTED", "RENDER_ATTEMPT", "PLANS_PHASE_A", "FOUNDATION_BUNDLE",
    "TERMINAL_PHASE_A_ACCEPTED",
)
PHASE_A_REJECTED_TERMINAL = "TERMINAL_PHASE_A_REJECTED"
TREATMENT_SEQUENCE = (
    "STARTED", "FOUNDATION_LOAD", "ARM_FF", "ARM_CC", "ARM_WW",
    "ARM_FC", "ARM_FW", "ARM_VP", "TERMINAL_TREATMENT",
)


class V13StoreError(RuntimeError):
    """A persistence, liveness, identity, or chain condition failed closed."""


class V13AlreadyTerminal(V13StoreError):
    """The exact identity already has a validated terminal binding."""

    def __init__(self, binding: Mapping[str, Any]):
        self.binding = deepcopy(dict(binding))
        super().__init__("exact case identity is already terminal and reusable")


class _DeadOwnerProof:
    __slots__ = ("lock_sha256", "pid", "process_start_token")

    def __init__(self, *, lock_sha256: str, pid: int, process_start_token: str):
        self.lock_sha256 = _sha(lock_sha256, "dead-owner lock hash")
        self.pid = _plain_int(pid, "dead-owner pid", 1, 2 ** 31 - 1)
        _require(_LINUX_START_TOKEN.fullmatch(process_start_token) is not None,
                 "dead-owner process token differs")
        self.process_start_token = process_start_token


@contextmanager
def _metadata_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / METADATA_LOCK_NAME
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _serialized_mutation(function):
    @wraps(function)
    def wrapped(root: Path, *args, **kwargs):
        if kwargs.pop("_metadata_lock_held", False):
            return function(Path(root), *args, **kwargs)
        with _metadata_lock(Path(root)):
            return function(Path(root), *args, **kwargs)
    return wrapped


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V13StoreError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise V13StoreError(f"value is not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None,
             f"{label} is not lowercase SHA-256")
    return value


def _slug(value: object, label: str) -> str:
    _require(isinstance(value, str) and _SLUG.fullmatch(value) is not None,
             f"{label} is not a safe nonempty slug")
    return value


def _utc(value: object, label: str) -> str:
    _require(isinstance(value, str) and _UTC.fullmatch(value) is not None,
             f"{label} is not whole-second RFC-3339 UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise V13StoreError(f"{label} is invalid") from exc
    _require(parsed.utcoffset() == timezone.utc.utcoffset(parsed),
             f"{label} is not UTC")
    return value


def _safe_relative(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} is empty")
    pure = PurePosixPath(value)
    _require(not pure.is_absolute() and pure.parts and ".." not in pure.parts
             and pure.parts[0] not in (".git", "."), f"{label} is unsafe")
    normalized = pure.as_posix()
    _require(normalized == value, f"{label} is not normalized")
    return normalized


def _plain_int(value: object, label: str, minimum: int, maximum: int) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool)
             and minimum <= value <= maximum,
             f"{label} lies outside {minimum}..{maximum}")
    return value


def _linux_process_start_token(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
) -> str:
    """Read the exact Linux boot and procfs start identity for ``pid``."""

    _plain_int(pid, "process pid", 1, 2 ** 31 - 1)
    if sys.platform != "linux" and proc_root == Path("/proc"):
        raise V13StoreError("production process identity is supported only on Linux")
    try:
        boot_id = boot_id_path.read_text(encoding="ascii").strip().lower()
    except OSError as exc:
        raise V13StoreError(f"cannot read Linux boot identity: {exc}") from exc
    _require(_BOOT_ID.fullmatch(boot_id) is not None,
             "Linux boot identity is malformed")
    try:
        stat_raw = (proc_root / str(pid) / "stat").read_bytes().strip()
    except FileNotFoundError as exc:
        raise ProcessLookupError(pid) from exc
    except OSError as exc:
        raise V13StoreError(f"cannot read Linux process identity: {exc}") from exc
    prefix = f"{pid} (".encode("ascii")
    close = stat_raw.rfind(b")")
    _require(close > 0 and stat_raw.startswith(prefix)
             and close + 2 < len(stat_raw), "Linux proc stat is malformed")
    suffix = stat_raw[close + 2:].split()
    _require(len(suffix) >= 20, "Linux proc stat lacks starttime")
    start_ticks = suffix[19]
    _require(start_ticks.isdigit() and int(start_ticks) > 0,
             "Linux proc starttime is malformed")
    return f"linux-procfs-v1:{boot_id}:{int(start_ticks)}"


def linux_process_start_token(pid: int) -> str:
    """Return the production Linux PID-reuse-safe start token."""

    return _linux_process_start_token(pid)


def process_identity_is_alive(pid: int, process_start_token: str) -> bool:
    """Check a live Linux PID against the exact recorded boot/start identity."""

    _plain_int(pid, "process pid", 1, 2 ** 31 - 1)
    _require(isinstance(process_start_token, str)
             and _LINUX_START_TOKEN.fullmatch(process_start_token) is not None,
             "process-start token is not a Linux procfs token")
    try:
        observed = linux_process_start_token(pid)
    except ProcessLookupError:
        return False
    return observed == process_start_token


def _verify_current_process_identity(
    pid: int,
    process_start_token: str,
    *,
    _test_token_reader: Callable[[int], str] | None = None,
) -> None:
    reader = _test_token_reader or linux_process_start_token
    if _test_token_reader is None:
        _require(pid == os.getpid(), "store caller PID is not the current process")
    try:
        observed = reader(pid)
    except ProcessLookupError as exc:
        raise V13StoreError("store caller process is absent") from exc
    _require(observed == process_start_token,
             "store caller process-start identity differs")


def validate_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "design_id", "release_sha256", "runtime_fingerprint_sha256",
        "primary_batch_id", "gpu_uuid", "case_id", "render_id", "schedule",
        "mode",
    }
    _require(isinstance(value, Mapping) and set(value) == fields,
             "case identity field set differs")
    _require(value.get("design_id") == DESIGN_ID, "case design ID differs")
    _sha(value.get("release_sha256"), "release_sha256")
    _sha(value.get("runtime_fingerprint_sha256"), "runtime_fingerprint_sha256")
    for key in ("primary_batch_id", "case_id"):
        _slug(value.get(key), key)
    _require(isinstance(value.get("gpu_uuid"), str)
             and bool(value["gpu_uuid"].strip()), "gpu_uuid is empty")
    _require(value.get("render_id") in ("r1", "r2"),
             "render_id must be r1 or r2")
    _require(value.get("schedule") in SCHEDULES, "case schedule differs")
    _require(value.get("mode") in MODES, "case mode differs")
    return deepcopy(dict(value))


def identity_sha256(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(validate_identity(value)))


def build_identity(
    *, release_sha256: str, runtime_fingerprint_sha256: str,
    primary_batch_id: str, gpu_uuid: str, case_id: str, render_id: str,
    schedule: str, mode: str,
) -> dict[str, Any]:
    return validate_identity({
        "design_id": DESIGN_ID,
        "release_sha256": release_sha256,
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "primary_batch_id": primary_batch_id,
        "gpu_uuid": gpu_uuid,
        "case_id": case_id,
        "render_id": render_id,
        "schedule": schedule,
        "mode": mode,
    })


def validate_bounds(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "active_case_count", "active_render_count", "history_ids",
        "completed_arm_ids", "live_token_count", "selected_row_count",
        "layer_count", "case_deadline_seconds",
    }
    _require(isinstance(value, Mapping) and set(value) == fields,
             "case state-bound field set differs")
    _require(value.get("active_case_count") == 1,
             "store requires exactly one active case")
    _require(value.get("active_render_count") == 1,
             "store requires exactly one active render")
    _require(value.get("history_ids") == list(HISTORIES),
             "store history bound differs from F/C/W")
    arms = value.get("completed_arm_ids")
    _require(isinstance(arms, list) and arms == list(PRIMARY_ARMS[:len(arms)]),
             "completed arms are not an exact frozen-order prefix")
    _plain_int(value.get("live_token_count"), "live_token_count", 0,
               MAX_LIVE_TOKENS)
    _plain_int(value.get("selected_row_count"), "selected_row_count", 0,
               MAX_SELECTED_ROWS)
    _require(value.get("layer_count") == REQUIRED_LAYERS,
             "layer count must equal 48")
    _plain_int(value.get("case_deadline_seconds"), "case_deadline_seconds", 1,
               MAX_CASE_DEADLINE_SECONDS)
    return deepcopy(dict(value))


def build_bounds(*, completed_arm_ids: Sequence[str] = (),
                 live_token_count: int = 0, selected_row_count: int = 0,
                 case_deadline_seconds: int) -> dict[str, Any]:
    return validate_bounds({
        "active_case_count": 1,
        "active_render_count": 1,
        "history_ids": list(HISTORIES),
        "completed_arm_ids": list(completed_arm_ids),
        "live_token_count": live_token_count,
        "selected_row_count": selected_row_count,
        "layer_count": REQUIRED_LAYERS,
        "case_deadline_seconds": case_deadline_seconds,
    })


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V13StoreError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    _require(len(raw) <= MAX_JSON_BYTES, f"{label} exceeds JSON byte bound")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise V13StoreError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root is not an object")
    _require(raw == canonical_json_bytes(value) + b"\n",
             f"{label} bytes are not canonical JSON plus LF")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(dict(value)) + b"\n"
    _require(len(raw) <= MAX_JSON_BYTES, "JSON artifact exceeds byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise V13StoreError(f"immutable JSON already exists: {path}") from exc
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise V13StoreError(f"cannot read {label}: {exc}") from exc
    return _strict_json(raw, label), raw


def _artifact_bindings(
    value: Sequence[Mapping[str, Any]], *, artifact_root: Path | None = None,
) -> list[dict[str, Any]]:
    _require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
             "artifact bindings are not a sequence")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(value):
        _require(isinstance(row, Mapping) and set(row) == {
            "kind", "path", "sha256", "size_bytes",
        }, f"artifact binding {index} fields differ")
        _require(isinstance(row.get("kind"), str) and bool(row["kind"]),
                 f"artifact binding {index} kind is empty")
        path = _safe_relative(row.get("path"), f"artifact binding {index} path")
        _require(path not in seen, "artifact binding path is repeated")
        seen.add(path)
        digest = _sha(row.get("sha256"), f"artifact binding {index} hash")
        size = _plain_int(row.get("size_bytes"),
                          f"artifact binding {index} size", 1,
                          MAX_BOUND_ARTIFACT_BYTES)
        result.append({
            "kind": row["kind"], "path": path,
            "sha256": digest, "size_bytes": size,
        })
        if artifact_root is not None:
            actual = artifact_root / path
            try:
                resolved_root = artifact_root.resolve(strict=True)
                resolved_actual = actual.resolve(strict=True)
                resolved_actual.relative_to(resolved_root)
            except (OSError, ValueError) as exc:
                raise V13StoreError(
                    f"bound artifact escapes or is absent: {path}") from exc
            relative_parts = PurePosixPath(path).parts
            cursor = artifact_root
            for part in relative_parts:
                cursor = cursor / part
                _require(not cursor.is_symlink(),
                         f"bound artifact path contains a symlink: {path}")
            _require(resolved_actual.is_file(),
                     f"bound artifact is absent or not regular: {path}")
            _require(resolved_actual.stat().st_size == size
                     and file_sha256(resolved_actual) == digest,
                     f"bound artifact bytes differ: {path}")
    return sorted(result, key=lambda row: row["path"])


def _bindings_by_kind(
    value: Sequence[Mapping[str, Any]], *, artifact_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    rows = _artifact_bindings(value, artifact_root=artifact_root)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        _require(row["kind"] not in result,
                 f"artifact binding kind is repeated: {row['kind']}")
        result[row["kind"]] = row
    return result


def _payload_fields(
    value: object, fields: set[str], label: str,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == fields,
             f"{label} payload field set differs")
    return deepcopy(dict(value))


def _require_binding_hashes(
    bindings: Sequence[Mapping[str, Any]],
    expected: Mapping[str, str],
    payload: Mapping[str, Any],
) -> None:
    """Require one exact external artifact kind for every named payload hash."""

    by_kind = _bindings_by_kind(bindings)
    _require(set(by_kind) == set(expected.values()),
             "record artifact kind set differs from stage schema")
    for payload_field, kind in expected.items():
        digest = _sha(payload.get(payload_field), payload_field)
        _require(by_kind[kind]["sha256"] == digest,
                 f"{kind} artifact hash differs from payload")


def _validate_record_evidence(
    record_kind: str,
    payload_value: object,
    binding_value: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the exact nonsemantic evidence schema for one durable stage."""

    if record_kind == "STARTED":
        payload = _payload_fields(payload_value, {"status"}, "STARTED")
        _require(payload["status"] == "STARTED", "STARTED status differs")
        _require(not binding_value, "STARTED cannot bind stage artifacts")
        return payload
    if record_kind == "RENDER_ATTEMPT":
        payload = _payload_fields(payload_value, {
            "attempt_index", "outcome", "attempt_sha256", "review_sha256",
            "rejection_code",
        }, "RENDER_ATTEMPT")
        _plain_int(payload["attempt_index"], "attempt_index", 0, 2)
        _require(payload["outcome"] in ("ACCEPTED", "REJECTED"),
                 "render attempt outcome differs")
        if payload["outcome"] == "ACCEPTED":
            _require(payload["rejection_code"] is None,
                     "accepted render has a rejection code")
        else:
            _slug(payload["rejection_code"], "rejection_code")
        _require_binding_hashes(binding_value, {
            "attempt_sha256": "render_attempt",
            "review_sha256": "render_review",
        }, payload)
        return payload
    if record_kind == "PLANS_PHASE_A":
        payload = _payload_fields(payload_value, {
            "plans_sha256", "phase_a_scores_sha256", "oracle_scores_sha256",
        }, "PLANS_PHASE_A")
        _require_binding_hashes(binding_value, {
            "plans_sha256": "plans",
            "phase_a_scores_sha256": "phase_a_scores",
            "oracle_scores_sha256": "oracle_scores",
        }, payload)
        return payload
    if record_kind in ("FOUNDATION_BUNDLE", "FOUNDATION_LOAD"):
        payload = _payload_fields(payload_value, {
            "foundation_bundle_sha256", "foundation_descriptor_sha256",
            "foundation_receipt_sha256",
        }, record_kind)
        _require_binding_hashes(binding_value, {
            "foundation_bundle_sha256": "foundation_bundle",
            "foundation_descriptor_sha256": "foundation_descriptor",
            "foundation_receipt_sha256": "foundation_receipt",
        }, payload)
        return payload
    if record_kind.startswith("ARM_"):
        arm = record_kind.removeprefix("ARM_")
        _require(arm in PRIMARY_ARMS, "record arm differs from frozen arms")
        payload = _payload_fields(payload_value, {
            "arm_id", "checkpoint_sha256", "artifact_sha256",
        }, record_kind)
        _require(payload["arm_id"] == arm, "arm payload ID differs")
        _require_binding_hashes(binding_value, {
            "checkpoint_sha256": f"arm_{arm}_checkpoint",
            "artifact_sha256": f"arm_{arm}_artifact",
        }, payload)
        return payload
    if record_kind == "TERMINAL_PHASE_A_ACCEPTED":
        payload = _payload_fields(payload_value, {
            "status", "evidence_chain_sha256", "terminal_evidence_sha256",
            "validation_receipt_sha256",
        }, record_kind)
        _require(payload["status"] == "PHASE_A_ACCEPTED",
                 "Phase-A accepted terminal status differs")
        _sha(payload["evidence_chain_sha256"], "evidence_chain_sha256")
        _require_binding_hashes(binding_value, {
            "terminal_evidence_sha256": "phase_a_terminal_evidence",
            "validation_receipt_sha256":
                "independent_terminal_validation_receipt",
        }, payload)
        return payload
    if record_kind == PHASE_A_REJECTED_TERMINAL:
        payload = _payload_fields(payload_value, {
            "status", "attempt_count", "evidence_chain_sha256",
            "rejection_evidence_sha256", "validation_receipt_sha256",
        }, record_kind)
        _require(payload["status"] == "ALL_RENDER_ATTEMPTS_REJECTED"
                 and payload["attempt_count"] == 3,
                 "Phase-A rejected terminal evidence differs")
        _sha(payload["evidence_chain_sha256"], "evidence_chain_sha256")
        _require_binding_hashes(binding_value, {
            "rejection_evidence_sha256": "phase_a_rejection_evidence",
            "validation_receipt_sha256":
                "independent_terminal_validation_receipt",
        }, payload)
        return payload
    if record_kind == "TERMINAL_TREATMENT":
        payload = _payload_fields(payload_value, {
            "status", "evidence_chain_sha256", "terminal_evidence_sha256",
            "validation_receipt_sha256",
        }, record_kind)
        _require(payload["status"] == "TREATMENT_COMPLETE",
                 "treatment terminal status differs")
        _sha(payload["evidence_chain_sha256"], "evidence_chain_sha256")
        _require_binding_hashes(binding_value, {
            "terminal_evidence_sha256": "treatment_terminal_evidence",
            "validation_receipt_sha256":
                "independent_terminal_validation_receipt",
        }, payload)
        return payload
    raise V13StoreError(f"record kind lacks an evidence schema: {record_kind}")


def _evidence_chain_sha256(
    chain: Sequence[tuple[Path, Mapping[str, Any]]],
) -> str:
    return sha256_bytes(canonical_json_bytes([
        {"name": path.name, "sha256": file_sha256(path)}
        for path, _record in chain
    ]))


def _validate_terminal_external_evidence(
    record: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    prior_records: Sequence[Mapping[str, Any]],
    artifact_root: Path,
) -> None:
    """Parse and recompute the external terminal evidence and validator receipt."""

    kind = record["record_kind"]
    _require(kind in {
        "TERMINAL_PHASE_A_ACCEPTED", PHASE_A_REJECTED_TERMINAL,
        "TERMINAL_TREATMENT",
    }, "external terminal validation called for a nonterminal record")
    bindings = _bindings_by_kind(
        record["artifact_bindings"], artifact_root=artifact_root)
    if kind == "TERMINAL_PHASE_A_ACCEPTED":
        evidence_kind = "phase_a_terminal_evidence"
        evidence_field = "terminal_evidence_sha256"
        accepted = [prior["payload"]["attempt_index"]
                    for prior in prior_records
                    if prior["record_kind"] == "RENDER_ATTEMPT"
                    and prior["payload"]["outcome"] == "ACCEPTED"]
        _require(len(accepted) == 1,
                 "accepted terminal lacks one accepted attempt")
        accepted_attempt_index: int | None = accepted[0]
        rejection_codes = [prior["payload"]["rejection_code"]
                           for prior in prior_records
                           if prior["record_kind"] == "RENDER_ATTEMPT"
                           and prior["payload"]["outcome"] == "REJECTED"]
        completed_arms: list[str] = []
    elif kind == PHASE_A_REJECTED_TERMINAL:
        evidence_kind = "phase_a_rejection_evidence"
        evidence_field = "rejection_evidence_sha256"
        accepted_attempt_index = None
        rejection_codes = [prior["payload"]["rejection_code"]
                           for prior in prior_records
                           if prior["record_kind"] == "RENDER_ATTEMPT"]
        _require(len(rejection_codes) == 3 and all(rejection_codes),
                 "rejected terminal lacks three rejection codes")
        completed_arms = []
    else:
        evidence_kind = "treatment_terminal_evidence"
        evidence_field = "terminal_evidence_sha256"
        accepted_attempt_index = None
        rejection_codes = []
        completed_arms = list(PRIMARY_ARMS)
    evidence_path = artifact_root / bindings[evidence_kind]["path"]
    evidence, _evidence_raw = _read_json(
        evidence_path, f"{kind} external evidence")
    expected_evidence = {
        "schema": TERMINAL_EVIDENCE_SCHEMA,
        "design_id": DESIGN_ID,
        "identity_sha256": identity_sha256(identity),
        "terminal_kind": kind,
        "evidence_chain_sha256": record["payload"]["evidence_chain_sha256"],
        "outcome": record["payload"]["status"],
        "accepted_attempt_index": accepted_attempt_index,
        "rejection_codes": rejection_codes,
        "completed_arm_ids": completed_arms,
    }
    _require(evidence == expected_evidence,
             "external terminal evidence does not recompute")
    _require(bindings[evidence_kind]["sha256"] ==
             record["payload"][evidence_field],
             "external terminal evidence hash differs")
    receipt_path = artifact_root / bindings[
        "independent_terminal_validation_receipt"]["path"]
    receipt, _receipt_raw = _read_json(
        receipt_path, "independent terminal validation receipt")
    receipt_fields = {
        "schema", "design_id", "identity_sha256", "terminal_kind",
        "evidence_chain_sha256", "terminal_evidence_sha256", "validator_id",
        "validation_status", "validated_utc",
    }
    _require(set(receipt) == receipt_fields,
             "independent terminal receipt field set differs")
    _require(receipt == {
        "schema": TERMINAL_VALIDATION_SCHEMA,
        "design_id": DESIGN_ID,
        "identity_sha256": identity_sha256(identity),
        "terminal_kind": kind,
        "evidence_chain_sha256": record["payload"]["evidence_chain_sha256"],
        "terminal_evidence_sha256": bindings[evidence_kind]["sha256"],
        "validator_id": TERMINAL_VALIDATOR_ID,
        "validation_status": "PASS",
        "validated_utc": receipt["validated_utc"],
    }, "independent terminal receipt does not recompute")
    _utc(receipt["validated_utc"], "terminal receipt validated_utc")


def _allowed_next(
    mode: str,
    prior_records: Sequence[Mapping[str, Any]],
    new_kind: str,
    new_payload: Mapping[str, Any],
) -> None:
    _require(mode in MODES, "record mode differs")
    prior_kinds = [record["record_kind"] for record in prior_records]
    if mode == "phase_a":
        _require(prior_kinds and prior_kinds[0] == "STARTED",
                 "Phase-A chain lacks STARTED")
        attempts = [record for record in prior_records
                    if record["record_kind"] == "RENDER_ATTEMPT"]
        _require(len(attempts) <= 3, "Phase-A chain exceeds three attempts")
        accepted = [record for record in attempts
                    if record["payload"]["outcome"] == "ACCEPTED"]
        _require(len(accepted) <= 1, "Phase-A chain has multiple accepted renders")
        last = prior_kinds[-1]
        allowed = (
            new_kind == "RENDER_ATTEMPT" and last in ("STARTED", "RENDER_ATTEMPT")
            and len(attempts) < 3 and not accepted
            and new_payload["attempt_index"] == len(attempts)
        ) or (new_kind == "PLANS_PHASE_A" and last == "RENDER_ATTEMPT"
              and len(accepted) == 1
              and attempts[-1]["payload"]["outcome"] == "ACCEPTED") \
            or (new_kind == "FOUNDATION_BUNDLE" and last == "PLANS_PHASE_A") \
            or (new_kind == "TERMINAL_PHASE_A_ACCEPTED"
                and last == "FOUNDATION_BUNDLE") \
            or (new_kind == PHASE_A_REJECTED_TERMINAL
                and len(attempts) == 3 and not accepted
                and last == "RENDER_ATTEMPT")
        _require(allowed, f"invalid Phase-A transition {last} -> {new_kind}")
        return
    expected = list(TREATMENT_SEQUENCE)
    _require(prior_kinds == expected[:len(prior_kinds)],
             "treatment/technical chain differs from frozen sequence")
    _require(len(prior_kinds) < len(expected)
             and new_kind == expected[len(prior_kinds)],
             f"invalid treatment transition to {new_kind}")


def _required_completed_arms(mode: str, record_kind: str) -> list[str]:
    if mode == "phase_a" or record_kind in ("STARTED", "FOUNDATION_LOAD"):
        return []
    if record_kind.startswith("ARM_"):
        arm = record_kind.removeprefix("ARM_")
        _require(arm in PRIMARY_ARMS, "record arm differs from frozen arms")
        return list(PRIMARY_ARMS[:PRIMARY_ARMS.index(arm) + 1])
    if record_kind == "TERMINAL_TREATMENT":
        return list(PRIMARY_ARMS)
    return []


def validate_record(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema", "design_id", "record_kind", "sequence", "created_utc",
        "identity", "identity_sha256", "bounds", "payload",
        "artifact_bindings", "previous_record",
    }
    _require(isinstance(value, Mapping) and set(value) == fields,
             "case record field set differs")
    _require(value.get("schema") == RECORD_SCHEMA
             and value.get("design_id") == DESIGN_ID,
             "case record schema/design differs")
    identity = validate_identity(value.get("identity", {}))
    _require(value.get("identity_sha256") == identity_sha256(identity),
             "case record identity hash differs")
    sequence = _plain_int(value.get("sequence"), "record sequence", 0, 64)
    _utc(value.get("created_utc"), "record created_utc")
    _require(isinstance(value.get("record_kind"), str)
             and bool(value["record_kind"]), "record kind is empty")
    validate_bounds(value.get("bounds", {}))
    payload = _validate_record_evidence(
        value.get("record_kind", ""), value.get("payload"),
        value.get("artifact_bindings", []))
    _require(payload == value.get("payload"),
             "record payload does not equal validated evidence")
    _artifact_bindings(value.get("artifact_bindings", []))
    previous = value.get("previous_record")
    if sequence == 0:
        _require(value.get("record_kind") == "STARTED" and previous is None,
                 "initial record is not parentless STARTED")
    else:
        _require(isinstance(previous, Mapping) and set(previous) == {
            "name", "sha256",
        }, "noninitial record lacks exact parent binding")
        _require(isinstance(previous.get("name"), str)
                 and re.fullmatch(r"[0-9]{3}_[A-Z0-9_]+\.json",
                                  previous["name"]) is not None,
                 "parent record name differs")
        _sha(previous.get("sha256"), "parent record hash")
    return deepcopy(dict(value))


def _record_name(sequence: int, kind: str) -> str:
    _require(re.fullmatch(r"[A-Z0-9_]+", kind) is not None,
             "record kind is not filename-safe")
    return f"{sequence:03d}_{kind}.json"


def _validate_chain(
    case_dir: Path, identity: Mapping[str, Any], *, artifact_root: Path | None = None,
) -> list[
        tuple[Path, dict[str, Any]]]:
    expected_identity = validate_identity(identity)
    try:
        paths = sorted((case_dir / "records").iterdir())
    except OSError as exc:
        raise V13StoreError(f"cannot enumerate case record chain: {exc}") from exc
    _require(paths and all(path.is_file() and not path.is_symlink()
                           for path in paths),
             "case record chain is empty or contains non-files")
    chain: list[tuple[Path, dict[str, Any]]] = []
    prior_path: Path | None = None
    seen_artifact_paths: set[str] = set()
    for sequence, path in enumerate(paths):
        document, _raw = _read_json(path, f"case record {path.name}")
        record = validate_record(document)
        _require(record["sequence"] == sequence
                 and path.name == _record_name(sequence, record["record_kind"]),
                 "case record filename/sequence differs")
        _require(record["identity"] == expected_identity,
                 "case record identity differs within chain")
        _artifact_bindings(
            record["artifact_bindings"], artifact_root=artifact_root)
        record_artifact_paths = {
            binding["path"] for binding in record["artifact_bindings"]}
        _require(not (seen_artifact_paths & record_artifact_paths),
                 "artifact path is reused across case stages")
        seen_artifact_paths.update(record_artifact_paths)
        _require(record["bounds"]["completed_arm_ids"] ==
                 _required_completed_arms(
                     expected_identity["mode"], record["record_kind"]),
                 "record completed-arm bound differs from stage")
        if prior_path is not None:
            _require(record["previous_record"] == {
                "name": prior_path.name, "sha256": file_sha256(prior_path),
            }, "case record parent binding differs")
        if record["record_kind"] in {
            "TERMINAL_PHASE_A_ACCEPTED", PHASE_A_REJECTED_TERMINAL,
            "TERMINAL_TREATMENT",
        }:
            _require(record["payload"]["evidence_chain_sha256"] ==
                     _evidence_chain_sha256(chain),
                     "terminal evidence-chain hash differs")
            _require(artifact_root is not None,
                     "terminal validation requires the artifact root")
            _validate_terminal_external_evidence(
                record, identity=expected_identity,
                prior_records=[prior for _prior_path, prior in chain],
                artifact_root=artifact_root)
        prior_path = path
        chain.append((path, record))
    kinds = [record["record_kind"] for _path, record in chain]
    if expected_identity["mode"] == "phase_a":
        _require(kinds[0] == "STARTED"
                 and sum(kind == "RENDER_ATTEMPT" for kind in kinds) <= 3,
                 "Phase-A chain shape differs")
        for index in range(1, len(kinds)):
            _allowed_next(
                "phase_a", [record for _path, record in chain[:index]],
                kinds[index], chain[index][1]["payload"])
    else:
        _require(kinds == list(TREATMENT_SEQUENCE[:len(kinds)]),
                 "treatment/technical chain shape differs")
    return chain


def validate_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema", "design_id", "identity", "identity_sha256", "case_dir",
        "pid", "process_start_token", "created_utc",
    }
    _require(isinstance(value, Mapping) and set(value) == fields,
             "active lock field set differs")
    _require(value.get("schema") == LOCK_SCHEMA
             and value.get("design_id") == DESIGN_ID,
             "active lock schema/design differs")
    identity = validate_identity(value.get("identity", {}))
    digest = identity_sha256(identity)
    _require(value.get("identity_sha256") == digest,
             "active lock identity hash differs")
    _require(value.get("case_dir") == f"active/{digest}",
             "active lock case directory differs")
    _plain_int(value.get("pid"), "active lock pid", 1, 2 ** 31 - 1)
    _require(isinstance(value.get("process_start_token"), str)
             and _LINUX_START_TOKEN.fullmatch(
                 value["process_start_token"]) is not None,
             "active lock process-start token is not exact Linux procfs identity")
    _utc(value.get("created_utc"), "active lock created_utc")
    return deepcopy(dict(value))


def _lock_path(root: Path) -> Path:
    return root / ACTIVE_LOCK_NAME


def _read_lock(root: Path) -> dict[str, Any]:
    document, _raw = _read_json(_lock_path(root), "active lock")
    return validate_lock(document)


@_serialized_mutation
def _begin_case(
    root: Path,
    *,
    identity: Mapping[str, Any],
    bounds: Mapping[str, Any],
    pid: int,
    process_start_token: str,
    created_utc: str,
    payload: Mapping[str, Any] | None = None,
    _test_process_start_token_reader: Callable[[int], str] | None = None,
) -> dict[str, Any]:
    """Acquire the sole active-case slot and persist STARTED before work."""

    root = Path(root)
    _verify_current_process_identity(
        pid, process_start_token,
        _test_token_reader=_test_process_start_token_reader)
    frozen_identity = validate_identity(identity)
    frozen_bounds = validate_bounds(bounds)
    _require(frozen_bounds["completed_arm_ids"] == [],
             "STARTED cannot claim completed arms")
    digest = identity_sha256(frozen_identity)
    for name in ("active", "terminal", "quarantine", "index"):
        (root / name).mkdir(parents=True, exist_ok=True)
    index_path = root / "index" / f"{digest}.json"
    terminal_dir = root / "terminal" / digest
    if index_path.exists():
        binding = terminal_reuse_binding(
            root, expected_identity=frozen_identity)
        raise V13AlreadyTerminal(binding)
    _require(not terminal_dir.exists(),
             "terminal promotion exists without a validated index; recovery required")
    _require(not _lock_path(root).exists(), "another active-case lock exists")
    _require(not any((root / "active").iterdir()),
             "active directory contains an unowned case")
    lock = validate_lock({
        "schema": LOCK_SCHEMA,
        "design_id": DESIGN_ID,
        "identity": frozen_identity,
        "identity_sha256": digest,
        "case_dir": f"active/{digest}",
        "pid": pid,
        "process_start_token": process_start_token,
        "created_utc": created_utc,
    })
    write_json_exclusive(_lock_path(root), lock)
    case_dir = root / lock["case_dir"]
    try:
        case_dir.mkdir(parents=False, exist_ok=False)
        _fsync_directory(case_dir.parent)
        (case_dir / "records").mkdir()
        _fsync_directory(case_dir)
        started = validate_record({
            "schema": RECORD_SCHEMA,
            "design_id": DESIGN_ID,
            "record_kind": "STARTED",
            "sequence": 0,
            "created_utc": _utc(created_utc, "created_utc"),
            "identity": frozen_identity,
            "identity_sha256": digest,
            "bounds": frozen_bounds,
            "payload": ({"status": "STARTED"}
                        if payload is None else dict(payload)),
            "artifact_bindings": [],
            "previous_record": None,
        })
        write_json_exclusive(
            case_dir / "records" / _record_name(0, "STARTED"), started)
    except BaseException:
        # The complete lock deliberately remains as durable evidence. Recovery
        # requires a real process probe before it may be quarantined.
        raise
    return lock


def begin_case(
    root: Path,
    *,
    identity: Mapping[str, Any],
    bounds: Mapping[str, Any],
    pid: int,
    process_start_token: str,
    created_utc: str,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Production entry point using only the built-in Linux identity reader."""

    return _begin_case(
        root, identity=identity, bounds=bounds, pid=pid,
        process_start_token=process_start_token, created_utc=created_utc,
        payload=payload)


@_serialized_mutation
def _active_evidence_chain_sha256(
    root: Path,
    *,
    identity: Mapping[str, Any],
    pid: int,
    process_start_token: str,
    _test_process_start_token_reader: Callable[[int], str] | None = None,
) -> str:
    """Hash-bind the complete validated active chain before a terminal record."""

    root = Path(root)
    _verify_current_process_identity(
        pid, process_start_token,
        _test_token_reader=_test_process_start_token_reader)
    frozen = validate_identity(identity)
    lock = _read_lock(root)
    _require(lock["identity"] == frozen and lock["pid"] == pid
             and lock["process_start_token"] == process_start_token,
             "evidence caller differs from active lock")
    chain = _validate_chain(
        root / lock["case_dir"], frozen, artifact_root=root)
    _require(chain[-1][1]["record_kind"] not in _terminal_kinds(frozen["mode"]),
             "terminal record is already present")
    return _evidence_chain_sha256(chain)


def active_evidence_chain_sha256(
    root: Path,
    *,
    identity: Mapping[str, Any],
    pid: int,
    process_start_token: str,
) -> str:
    """Production chain binding using the built-in Linux identity reader."""

    return _active_evidence_chain_sha256(
        root, identity=identity, pid=pid,
        process_start_token=process_start_token)


@_serialized_mutation
def _append_case_record(
    root: Path,
    *,
    identity: Mapping[str, Any],
    pid: int,
    process_start_token: str,
    record_kind: str,
    bounds: Mapping[str, Any],
    created_utc: str,
    payload: Mapping[str, Any] | None = None,
    artifact_bindings: Sequence[Mapping[str, Any]] = (),
    _test_process_start_token_reader: Callable[[int], str] | None = None,
) -> dict[str, Any]:
    """Synchronously append one valid next record before more model work."""

    root = Path(root)
    _verify_current_process_identity(
        pid, process_start_token,
        _test_token_reader=_test_process_start_token_reader)
    frozen_identity = validate_identity(identity)
    lock = _read_lock(root)
    _require(lock["identity"] == frozen_identity
             and lock["pid"] == pid
             and lock["process_start_token"] == process_start_token,
             "append caller differs from active lock")
    case_dir = root / lock["case_dir"]
    chain = _validate_chain(case_dir, frozen_identity, artifact_root=root)
    sequence = len(chain)
    previous_path = chain[-1][0]
    frozen_bounds = validate_bounds(bounds)
    _require(frozen_bounds["completed_arm_ids"] ==
             _required_completed_arms(frozen_identity["mode"], record_kind),
             "new record completed-arm bound differs from stage")
    frozen_artifacts = _artifact_bindings(
        artifact_bindings, artifact_root=root)
    existing_artifact_paths = {
        binding["path"]
        for _path, prior_record in chain
        for binding in prior_record["artifact_bindings"]
    }
    _require(not (existing_artifact_paths & {
        binding["path"] for binding in frozen_artifacts}),
        "new record reuses a prior artifact path")
    record = validate_record({
        "schema": RECORD_SCHEMA,
        "design_id": DESIGN_ID,
        "record_kind": record_kind,
        "sequence": sequence,
        "created_utc": _utc(created_utc, "created_utc"),
        "identity": frozen_identity,
        "identity_sha256": identity_sha256(frozen_identity),
        "bounds": frozen_bounds,
        "payload": dict(payload or {}),
        "artifact_bindings": frozen_artifacts,
        "previous_record": {
            "name": previous_path.name,
            "sha256": file_sha256(previous_path),
        },
    })
    _allowed_next(
        frozen_identity["mode"], [row for _path, row in chain],
        record_kind, record["payload"])
    if record_kind in {
        "TERMINAL_PHASE_A_ACCEPTED", PHASE_A_REJECTED_TERMINAL,
        "TERMINAL_TREATMENT",
    }:
        _require(record["payload"]["evidence_chain_sha256"] ==
                 _evidence_chain_sha256(chain),
                 "new terminal evidence-chain hash differs")
        _validate_terminal_external_evidence(
            record, identity=frozen_identity,
            prior_records=[prior for _path, prior in chain],
            artifact_root=root)
    write_json_exclusive(
        case_dir / "records" / _record_name(sequence, record_kind), record)
    return record


def append_case_record(
    root: Path,
    *,
    identity: Mapping[str, Any],
    pid: int,
    process_start_token: str,
    record_kind: str,
    bounds: Mapping[str, Any],
    created_utc: str,
    payload: Mapping[str, Any] | None = None,
    artifact_bindings: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Production append using only the built-in Linux identity reader."""

    return _append_case_record(
        root, identity=identity, pid=pid,
        process_start_token=process_start_token, record_kind=record_kind,
        bounds=bounds, created_utc=created_utc, payload=payload,
        artifact_bindings=artifact_bindings)


def _terminal_kinds(mode: str) -> tuple[str, ...]:
    if mode == "phase_a":
        return ("TERMINAL_PHASE_A_ACCEPTED", PHASE_A_REJECTED_TERMINAL)
    return ("TERMINAL_TREATMENT",)


def _build_index(identity: Mapping[str, Any], terminal_dir: Path,
                 terminal_record: Path) -> dict[str, Any]:
    frozen = validate_identity(identity)
    return {
        "schema": INDEX_SCHEMA,
        "design_id": DESIGN_ID,
        "identity": frozen,
        "identity_sha256": identity_sha256(frozen),
        "terminal_directory": f"terminal/{identity_sha256(frozen)}",
        "terminal_record": {
            "name": terminal_record.name,
            "sha256": file_sha256(terminal_record),
        },
        "record_chain_sha256": sha256_bytes(canonical_json_bytes([
            {"name": path.name, "sha256": file_sha256(path)}
            for path in sorted((terminal_dir / "records").iterdir())
        ])),
        "status": "TERMINAL_VALIDATED",
    }


def validate_index(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema", "design_id", "identity", "identity_sha256",
        "terminal_directory", "terminal_record", "record_chain_sha256",
        "status",
    }
    _require(isinstance(value, Mapping) and set(value) == fields,
             "session index field set differs")
    _require(value.get("schema") == INDEX_SCHEMA
             and value.get("design_id") == DESIGN_ID
             and value.get("status") == "TERMINAL_VALIDATED",
             "session index identity/status differs")
    identity = validate_identity(value.get("identity", {}))
    digest = identity_sha256(identity)
    _require(value.get("identity_sha256") == digest
             and value.get("terminal_directory") == f"terminal/{digest}",
             "session index identity/path differs")
    terminal = value.get("terminal_record")
    _require(isinstance(terminal, Mapping) and set(terminal) == {
        "name", "sha256",
    }, "session index terminal binding differs")
    _require(isinstance(terminal.get("name"), str)
             and any(terminal["name"].endswith(f"_{kind}.json")
                     for kind in _terminal_kinds(identity["mode"])),
             "session index terminal record name differs")
    _sha(terminal.get("sha256"), "session index terminal hash")
    _sha(value.get("record_chain_sha256"), "session index chain hash")
    return deepcopy(dict(value))


@_serialized_mutation
def _finalize_case(
    root: Path,
    *,
    identity: Mapping[str, Any],
    pid: int,
    process_start_token: str,
    _test_process_start_token_reader: Callable[[int], str] | None = None,
    _dead_owner_proof: _DeadOwnerProof | None = None,
) -> dict[str, Any]:
    """Validate terminal chain, promote directory, index it, then clear lock."""

    root = Path(root)
    if _dead_owner_proof is None:
        _verify_current_process_identity(
            pid, process_start_token,
            _test_token_reader=_test_process_start_token_reader)
    frozen = validate_identity(identity)
    lock = _read_lock(root)
    _require(lock["identity"] == frozen and lock["pid"] == pid
             and lock["process_start_token"] == process_start_token,
             "finalize caller differs from active lock")
    if _dead_owner_proof is not None:
        _require(
            _dead_owner_proof.pid == lock["pid"]
            and _dead_owner_proof.process_start_token ==
            lock["process_start_token"]
            and _dead_owner_proof.lock_sha256 == file_sha256(_lock_path(root)),
            "dead-owner recovery proof differs from active lock")
    digest = identity_sha256(frozen)
    active_dir = root / lock["case_dir"]
    terminal_dir = root / "terminal" / digest
    if active_dir.exists():
        chain = _validate_chain(active_dir, frozen, artifact_root=root)
        _require(chain[-1][1]["record_kind"] in _terminal_kinds(frozen["mode"]),
                 "case chain is not terminal")
        _require(not terminal_dir.exists(), "terminal case directory already exists")
        active_dir.rename(terminal_dir)
        _fsync_directory(active_dir.parent)
        _fsync_directory(terminal_dir.parent)
    chain = _validate_chain(terminal_dir, frozen, artifact_root=root)
    terminal_path = chain[-1][0]
    _require(chain[-1][1]["record_kind"] in _terminal_kinds(frozen["mode"]),
             "promoted case is not terminal")
    index = validate_index(_build_index(frozen, terminal_dir, terminal_path))
    index_path = root / "index" / f"{digest}.json"
    if index_path.exists():
        existing, _raw = _read_json(index_path, "session index")
        _require(validate_index(existing) == index,
                 "existing session index differs from terminal chain")
    else:
        write_json_exclusive(index_path, index)
    try:
        _lock_path(root).unlink()
    except OSError as exc:
        raise V13StoreError(f"could not clear finalized active lock: {exc}") from exc
    _fsync_directory(root)
    return index


def finalize_case(
    root: Path,
    *,
    identity: Mapping[str, Any],
    pid: int,
    process_start_token: str,
) -> dict[str, Any]:
    """Production finalization using only the built-in Linux identity reader."""

    return _finalize_case(
        root, identity=identity, pid=pid,
        process_start_token=process_start_token)


def terminal_reuse_binding(
    root: Path, *, expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a skip binding only for an exact, complete terminal chain."""

    root = Path(root)
    frozen = validate_identity(expected_identity)
    digest = identity_sha256(frozen)
    index_document, index_raw = _read_json(
        root / "index" / f"{digest}.json", "session index")
    index = validate_index(index_document)
    _require(index["identity"] == frozen, "terminal index identity differs")
    terminal_dir = root / index["terminal_directory"]
    chain = _validate_chain(terminal_dir, frozen, artifact_root=root)
    terminal_path = chain[-1][0]
    _require(chain[-1][1]["record_kind"] in _terminal_kinds(frozen["mode"])
             and index["terminal_record"] == {
                 "name": terminal_path.name,
                 "sha256": file_sha256(terminal_path),
             }, "terminal record binding differs")
    recomputed = validate_index(_build_index(frozen, terminal_dir, terminal_path))
    _require(index == recomputed, "terminal chain/index hash differs")
    return {
        "index_path": f"index/{digest}.json",
        "index_sha256": sha256_bytes(index_raw),
        "terminal_directory": index["terminal_directory"],
        "terminal_record_name": index["terminal_record"]["name"],
        "terminal_record_sha256": index["terminal_record"]["sha256"],
        "identity_sha256": digest,
        "reusable": True,
    }


def _partial_manifest(case_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not case_dir.exists():
        return rows
    marker_path = case_dir / QUARANTINE_RECORD_NAME
    for path in sorted(case_dir.rglob("*")):
        if path == marker_path:
            continue
        _require(not path.is_symlink(), "partial case contains a symlink")
        if path.is_dir():
            continue
        _require(path.is_file(), "partial case contains a non-file")
        relative = path.relative_to(case_dir).as_posix()
        size = path.stat().st_size
        _plain_int(size, f"partial file {relative} size", 0,
                   MAX_BOUND_ARTIFACT_BYTES)
        rows.append({
            "path": _safe_relative(relative, "partial manifest path"),
            "sha256": file_sha256(path),
            "size_bytes": size,
        })
    _require(len(rows) <= 128, "partial manifest exceeds file-count bound")
    return rows


def _validate_quarantine_record(
    value: Mapping[str, Any], *, quarantine_dir: Path,
) -> dict[str, Any]:
    fields = {
        "schema", "design_id", "identity", "identity_sha256", "lock_sha256",
        "transaction_sha256", "quarantine_directory", "observed_utc", "reason",
        "process_probe",
        "partial_manifest", "partial_manifest_sha256", "partial_reusable",
    }
    _require(isinstance(value, Mapping) and set(value) == fields,
             "quarantine record field set differs")
    _require(value.get("schema") == QUARANTINE_SCHEMA
             and value.get("design_id") == DESIGN_ID,
             "quarantine schema/design differs")
    identity = validate_identity(value.get("identity", {}))
    digest = identity_sha256(identity)
    lock_sha = _sha(value.get("lock_sha256"), "quarantine lock hash")
    _require(value.get("transaction_sha256") == lock_sha,
             "quarantine transaction differs from lock hash")
    expected_name = f"{digest}_{lock_sha}"
    _require(value.get("identity_sha256") == digest
             and quarantine_dir.name == expected_name
             and value.get("quarantine_directory") ==
             f"quarantine/{expected_name}",
             "quarantine identity/directory binding differs")
    _utc(value.get("observed_utc"), "quarantine observed_utc")
    _require(isinstance(value.get("reason"), str)
             and bool(value["reason"].strip()), "quarantine reason is empty")
    probe = value.get("process_probe")
    _require(isinstance(probe, Mapping) and set(probe) == {
        "kind", "pid", "process_start_token", "alive",
    } and probe.get("kind") == "linux-procfs-v1"
        and probe.get("alive") is False,
        "quarantine process probe evidence differs")
    _plain_int(probe.get("pid"), "quarantine pid", 1, 2 ** 31 - 1)
    _require(isinstance(probe.get("process_start_token"), str)
             and _LINUX_START_TOKEN.fullmatch(
                 probe["process_start_token"]) is not None,
             "quarantine process token differs")
    archived_lock = quarantine_dir / ARCHIVED_LOCK_NAME
    _require(archived_lock.is_file() and not archived_lock.is_symlink()
             and file_sha256(archived_lock) == lock_sha,
             "quarantine archived lock differs")
    manifest = value.get("partial_manifest")
    _require(isinstance(manifest, list), "quarantine partial manifest differs")
    observed_manifest = _partial_manifest(quarantine_dir)
    _require(manifest == observed_manifest,
             "quarantine partial bytes differ from manifest")
    _require(value.get("partial_manifest_sha256") == sha256_bytes(
        canonical_json_bytes(manifest)), "quarantine partial manifest hash differs")
    _require(value.get("partial_reusable") is False,
             "quarantine partial is marked reusable")
    return deepcopy(dict(value))


def _read_valid_quarantine(quarantine_dir: Path) -> dict[str, Any]:
    document, _raw = _read_json(
        quarantine_dir / QUARANTINE_RECORD_NAME, "quarantine record")
    return _validate_quarantine_record(document, quarantine_dir=quarantine_dir)


def _completed_quarantine_retry(
    root: Path, *, observed_utc: str, reason: str,
    transaction_sha256: str | None,
) -> dict[str, Any]:
    if transaction_sha256 is not None:
        _sha(transaction_sha256, "quarantine transaction_sha256")
    all_records: list[dict[str, Any]] = []
    exact_candidates: list[dict[str, Any]] = []
    quarantine_root = root / "quarantine"
    if quarantine_root.exists():
        for path in sorted(quarantine_root.iterdir()):
            if not path.is_dir() or path.is_symlink():
                continue
            marker = path / QUARANTINE_RECORD_NAME
            if marker.exists():
                record = _read_valid_quarantine(path)
                all_records.append(record)
                if transaction_sha256 is not None:
                    if record["transaction_sha256"] == transaction_sha256:
                        exact_candidates.append(record)
                elif (record["observed_utc"] == observed_utc
                      and record["reason"] == reason):
                    exact_candidates.append(record)
    if not exact_candidates and transaction_sha256 is None and len(all_records) == 1:
        exact_candidates = all_records
    _require(len(exact_candidates) == 1,
             "no unique completed quarantine matches this stable transaction")
    return exact_candidates[0]


def _invoke_fault(
    hook: Callable[[str], None] | None, boundary: str,
) -> None:
    if hook is not None:
        hook(boundary)


@_serialized_mutation
def active_quarantine_transaction_sha256(root: Path) -> str:
    """Return the stable transaction key for the current immutable lock."""

    root = Path(root)
    lock = _read_lock(root)
    digest = file_sha256(_lock_path(root))
    _require(digest == sha256_bytes(canonical_json_bytes(lock) + b"\n"),
             "active lock bytes differ")
    return digest


@_serialized_mutation
def _quarantine_abandoned_case(
    root: Path,
    *,
    observed_utc: str,
    reason: str,
    transaction_sha256: str | None = None,
    _test_process_probe: Callable[[int, str], bool] | None = None,
    _test_fault_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Internal implementation with private fault/probe injection for tests."""

    root = Path(root)
    _utc(observed_utc, "quarantine observed_utc")
    _require(isinstance(reason, str) and bool(reason.strip()),
             "quarantine reason is empty")
    if not _lock_path(root).exists():
        return _completed_quarantine_retry(
            root, observed_utc=observed_utc, reason=reason,
            transaction_sha256=transaction_sha256)
    lock = _read_lock(root)
    probe = _test_process_probe or process_identity_is_alive
    alive = probe(lock["pid"], lock["process_start_token"])
    _require(isinstance(alive, bool), "process probe did not return boolean")
    _require(not alive, "active process is still alive; quarantine forbidden")
    frozen = lock["identity"]
    digest = lock["identity_sha256"]
    lock_path = _lock_path(root)
    lock_sha = file_sha256(lock_path)
    if transaction_sha256 is not None:
        _require(transaction_sha256 == lock_sha,
                 "requested quarantine transaction differs from active lock")
    _require(lock_sha == sha256_bytes(
        canonical_json_bytes(lock) + b"\n"), "active lock bytes differ")
    active_dir = root / lock["case_dir"]
    terminal_dir = root / "terminal" / digest
    if terminal_dir.exists():
        proof = _DeadOwnerProof(
            lock_sha256=lock_sha, pid=lock["pid"],
            process_start_token=lock["process_start_token"])
        return _finalize_case(
            root, identity=frozen, pid=lock["pid"],
            process_start_token=lock["process_start_token"],
            _dead_owner_proof=proof, _metadata_lock_held=True)
    quarantine_dir = root / "quarantine" / f"{digest}_{lock_sha}"
    _require(not (active_dir.exists() and quarantine_dir.exists()),
             "active and quarantine directories both exist")
    if quarantine_dir.exists():
        record = _read_valid_quarantine(quarantine_dir)
    else:
        if not active_dir.exists():
            active_dir.mkdir(parents=False, exist_ok=False)
            _fsync_directory(active_dir.parent)
        archived_lock = active_dir / ARCHIVED_LOCK_NAME
        if archived_lock.exists():
            _require(archived_lock.is_file() and not archived_lock.is_symlink()
                     and file_sha256(archived_lock) == lock_sha,
                     "existing archived lock differs")
        else:
            try:
                os.link(lock_path, archived_lock)
            except OSError as exc:
                raise V13StoreError(f"cannot archive active lock: {exc}") from exc
            _invoke_fault(_test_fault_hook, "after_lock_archive_link")
            _fsync_directory(active_dir)
        _invoke_fault(_test_fault_hook, "after_lock_archive")
        marker = active_dir / QUARANTINE_RECORD_NAME
        if marker.exists():
            marker_document, _raw = _read_json(marker, "quarantine marker")
            record = dict(marker_document)
        else:
            manifest = _partial_manifest(active_dir)
            record = {
                "schema": QUARANTINE_SCHEMA,
                "design_id": DESIGN_ID,
                "identity": frozen,
                "identity_sha256": digest,
                "lock_sha256": lock_sha,
                "transaction_sha256": lock_sha,
                "quarantine_directory": f"quarantine/{quarantine_dir.name}",
                "observed_utc": observed_utc,
                "reason": reason,
                "process_probe": {
                    "kind": "linux-procfs-v1",
                    "pid": lock["pid"],
                    "process_start_token": lock["process_start_token"],
                    "alive": False,
                },
                "partial_manifest": manifest,
                "partial_manifest_sha256": sha256_bytes(
                    canonical_json_bytes(manifest)),
                "partial_reusable": False,
            }
            write_json_exclusive(marker, record)
        _invoke_fault(_test_fault_hook, "after_quarantine_marker")
        active_dir.rename(quarantine_dir)
        _invoke_fault(
            _test_fault_hook, "after_quarantine_rename_before_fsync")
        _fsync_directory(root / "active")
        _fsync_directory(root / "quarantine")
        _invoke_fault(_test_fault_hook, "after_quarantine_rename")
        record = _read_valid_quarantine(quarantine_dir)
    _require(file_sha256(lock_path) == lock_sha and _read_lock(root) == lock,
             "active lock changed during quarantine")
    try:
        lock_path.unlink()
    except OSError as exc:
        raise V13StoreError(f"could not clear quarantined active lock: {exc}") from exc
    _invoke_fault(_test_fault_hook, "after_quarantine_lock_unlink")
    _fsync_directory(root)
    _invoke_fault(_test_fault_hook, "after_quarantine_lock_clear")
    return deepcopy(record)


def quarantine_abandoned_case(
    root: Path,
    *,
    observed_utc: str,
    reason: str,
    transaction_sha256: str | None = None,
) -> dict[str, Any]:
    """Quarantine only after the production Linux process identity is dead."""

    return _quarantine_abandoned_case(
        root, observed_utc=observed_utc, reason=reason,
        transaction_sha256=transaction_sha256)


__all__ = [
    "ACTIVE_LOCK_NAME",
    "ARCHIVED_LOCK_NAME",
    "INDEX_SCHEMA",
    "LOCK_SCHEMA",
    "MAX_BOUND_ARTIFACT_BYTES",
    "MAX_CASE_DEADLINE_SECONDS",
    "MAX_LIVE_TOKENS",
    "MAX_SELECTED_ROWS",
    "QUARANTINE_SCHEMA",
    "QUARANTINE_RECORD_NAME",
    "RECORD_SCHEMA",
    "TERMINAL_EVIDENCE_SCHEMA",
    "TERMINAL_VALIDATION_SCHEMA",
    "TERMINAL_VALIDATOR_ID",
    "V13AlreadyTerminal",
    "V13StoreError",
    "active_evidence_chain_sha256",
    "active_quarantine_transaction_sha256",
    "append_case_record",
    "begin_case",
    "build_bounds",
    "build_identity",
    "canonical_json_bytes",
    "file_sha256",
    "finalize_case",
    "identity_sha256",
    "linux_process_start_token",
    "process_identity_is_alive",
    "quarantine_abandoned_case",
    "sha256_bytes",
    "terminal_reuse_binding",
    "validate_bounds",
    "validate_identity",
    "validate_index",
    "validate_lock",
    "validate_record",
    "write_json_exclusive",
]
