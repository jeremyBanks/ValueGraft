"""Append-only, one-live-case persistence for powered successor v13.

The store retains no model object or tensor in global memory.  It serializes an
immutable per-case record chain, one exact active-process lock, terminal-only
index bindings, and loss-preserving quarantine.  Partial chains are never
promoted or resumed as scientific cases.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

from powered_v13_schema import DESIGN_ID, N_SCHEDULE, P_SCHEDULE, PRIMARY_ARMS


LOCK_SCHEMA = "coherent-state-powered-successor-v13-active-lock-v1"
RECORD_SCHEMA = "coherent-state-powered-successor-v13-case-record-v1"
INDEX_SCHEMA = "coherent-state-powered-successor-v13-session-index-v1"
QUARANTINE_SCHEMA = "coherent-state-powered-successor-v13-quarantine-v1"
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
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")


PHASE_A_SEQUENCE = (
    "STARTED", "RENDER_ATTEMPT", "PLANS_PHASE_A", "FOUNDATION_BUNDLE",
    "TERMINAL_PHASE_A",
)
TREATMENT_SEQUENCE = (
    "STARTED", "FOUNDATION_LOAD", "ARM_FF", "ARM_CC", "ARM_WW",
    "ARM_FC", "ARM_FW", "ARM_VP", "TERMINAL_TREATMENT",
)


class V13StoreError(RuntimeError):
    """A persistence, liveness, identity, or chain condition failed closed."""


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


def _allowed_next(mode: str, prior_kinds: list[str], new_kind: str) -> None:
    _require(mode in MODES, "record mode differs")
    if mode == "phase_a":
        _require(prior_kinds and prior_kinds[0] == "STARTED",
                 "Phase-A chain lacks STARTED")
        attempts = sum(kind == "RENDER_ATTEMPT" for kind in prior_kinds)
        _require(attempts <= 3, "Phase-A chain exceeds three attempts")
        last = prior_kinds[-1]
        allowed = (
            new_kind == "RENDER_ATTEMPT" and last in ("STARTED", "RENDER_ATTEMPT")
            and attempts < 3
        ) or (new_kind == "PLANS_PHASE_A" and last == "RENDER_ATTEMPT") \
            or (new_kind == "FOUNDATION_BUNDLE" and last == "PLANS_PHASE_A") \
            or (new_kind == "TERMINAL_PHASE_A" and last == "FOUNDATION_BUNDLE")
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
    _require(isinstance(value.get("payload"), Mapping),
             "record payload is not an object")
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
        _require(record["bounds"]["completed_arm_ids"] ==
                 _required_completed_arms(
                     expected_identity["mode"], record["record_kind"]),
                 "record completed-arm bound differs from stage")
        if prior_path is not None:
            _require(record["previous_record"] == {
                "name": prior_path.name, "sha256": file_sha256(prior_path),
            }, "case record parent binding differs")
        prior_path = path
        chain.append((path, record))
    kinds = [record["record_kind"] for _path, record in chain]
    if expected_identity["mode"] == "phase_a":
        _require(kinds[0] == "STARTED"
                 and sum(kind == "RENDER_ATTEMPT" for kind in kinds) <= 3,
                 "Phase-A chain shape differs")
        for index in range(1, len(kinds)):
            _allowed_next("phase_a", kinds[:index], kinds[index])
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
             and bool(value["process_start_token"].strip()),
             "active lock process-start token is empty")
    _utc(value.get("created_utc"), "active lock created_utc")
    return deepcopy(dict(value))


def _lock_path(root: Path) -> Path:
    return root / ACTIVE_LOCK_NAME


def _read_lock(root: Path) -> dict[str, Any]:
    document, _raw = _read_json(_lock_path(root), "active lock")
    return validate_lock(document)


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
    """Acquire the sole active-case slot and persist STARTED before work."""

    root = Path(root)
    frozen_identity = validate_identity(identity)
    frozen_bounds = validate_bounds(bounds)
    _require(frozen_bounds["completed_arm_ids"] == [],
             "STARTED cannot claim completed arms")
    digest = identity_sha256(frozen_identity)
    for name in ("active", "terminal", "quarantine", "index"):
        (root / name).mkdir(parents=True, exist_ok=True)
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
        (case_dir / "records").mkdir()
        started = validate_record({
            "schema": RECORD_SCHEMA,
            "design_id": DESIGN_ID,
            "record_kind": "STARTED",
            "sequence": 0,
            "created_utc": _utc(created_utc, "created_utc"),
            "identity": frozen_identity,
            "identity_sha256": digest,
            "bounds": frozen_bounds,
            "payload": dict(payload or {}),
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
    """Synchronously append one valid next record before more model work."""

    root = Path(root)
    frozen_identity = validate_identity(identity)
    lock = _read_lock(root)
    _require(lock["identity"] == frozen_identity
             and lock["pid"] == pid
             and lock["process_start_token"] == process_start_token,
             "append caller differs from active lock")
    case_dir = root / lock["case_dir"]
    chain = _validate_chain(case_dir, frozen_identity, artifact_root=root)
    kinds = [row["record_kind"] for _path, row in chain]
    _allowed_next(frozen_identity["mode"], kinds, record_kind)
    sequence = len(chain)
    previous_path = chain[-1][0]
    frozen_bounds = validate_bounds(bounds)
    _require(frozen_bounds["completed_arm_ids"] ==
             _required_completed_arms(frozen_identity["mode"], record_kind),
             "new record completed-arm bound differs from stage")
    frozen_artifacts = _artifact_bindings(
        artifact_bindings, artifact_root=root)
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
    write_json_exclusive(
        case_dir / "records" / _record_name(sequence, record_kind), record)
    return record


def _terminal_kind(mode: str) -> str:
    return "TERMINAL_PHASE_A" if mode == "phase_a" else "TERMINAL_TREATMENT"


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
             and terminal["name"].endswith(f"_{_terminal_kind(identity['mode'])}.json"),
             "session index terminal record name differs")
    _sha(terminal.get("sha256"), "session index terminal hash")
    _sha(value.get("record_chain_sha256"), "session index chain hash")
    return deepcopy(dict(value))


def finalize_case(
    root: Path,
    *,
    identity: Mapping[str, Any],
    pid: int,
    process_start_token: str,
) -> dict[str, Any]:
    """Validate terminal chain, promote directory, index it, then clear lock."""

    root = Path(root)
    frozen = validate_identity(identity)
    lock = _read_lock(root)
    _require(lock["identity"] == frozen and lock["pid"] == pid
             and lock["process_start_token"] == process_start_token,
             "finalize caller differs from active lock")
    digest = identity_sha256(frozen)
    active_dir = root / lock["case_dir"]
    terminal_dir = root / "terminal" / digest
    if active_dir.exists():
        chain = _validate_chain(active_dir, frozen, artifact_root=root)
        _require(chain[-1][1]["record_kind"] == _terminal_kind(frozen["mode"]),
                 "case chain is not terminal")
        _require(not terminal_dir.exists(), "terminal case directory already exists")
        active_dir.rename(terminal_dir)
        _fsync_directory(terminal_dir.parent)
    chain = _validate_chain(terminal_dir, frozen, artifact_root=root)
    terminal_path = chain[-1][0]
    _require(chain[-1][1]["record_kind"] == _terminal_kind(frozen["mode"]),
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
    _require(chain[-1][1]["record_kind"] == _terminal_kind(frozen["mode"])
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
        "terminal_record_sha256": index["terminal_record"]["sha256"],
        "identity_sha256": digest,
        "reusable": True,
    }


def quarantine_abandoned_case(
    root: Path,
    *,
    process_probe: Callable[[int, str], bool],
    observed_utc: str,
    reason: str,
) -> dict[str, Any]:
    """Preserve a partial only after a real PID/start-token probe says dead."""

    root = Path(root)
    _require(callable(process_probe), "process_probe is not callable")
    lock = _read_lock(root)
    _require(isinstance(reason, str) and bool(reason.strip()),
             "quarantine reason is empty")
    alive = process_probe(lock["pid"], lock["process_start_token"])
    _require(isinstance(alive, bool), "process probe did not return boolean")
    _require(not alive, "active process is still alive; quarantine forbidden")
    frozen = lock["identity"]
    digest = lock["identity_sha256"]
    active_dir = root / lock["case_dir"]
    terminal_dir = root / "terminal" / digest
    if terminal_dir.exists():
        # A crash after directory promotion is completed, not relabeled partial.
        return finalize_case(
            root, identity=frozen, pid=lock["pid"],
            process_start_token=lock["process_start_token"])
    stamp = _utc(observed_utc, "quarantine observed_utc").replace(":", "").replace("-", "")
    quarantine_dir = root / "quarantine" / f"{digest}_{stamp}"
    _require(not quarantine_dir.exists(), "quarantine destination already exists")
    quarantine_dir.mkdir(parents=True, exist_ok=False)
    if active_dir.exists():
        active_dir.rename(quarantine_dir / "partial_case")
    record = {
        "schema": QUARANTINE_SCHEMA,
        "design_id": DESIGN_ID,
        "identity": frozen,
        "identity_sha256": digest,
        "lock_sha256": file_sha256(_lock_path(root)),
        "observed_utc": observed_utc,
        "reason": reason,
        "process_probe": {
            "pid": lock["pid"],
            "process_start_token": lock["process_start_token"],
            "alive": False,
        },
        "partial_reusable": False,
    }
    write_json_exclusive(quarantine_dir / "QUARANTINE.json", record)
    _lock_path(root).unlink()
    _fsync_directory(root)
    return deepcopy(record)


__all__ = [
    "ACTIVE_LOCK_NAME",
    "INDEX_SCHEMA",
    "LOCK_SCHEMA",
    "MAX_BOUND_ARTIFACT_BYTES",
    "MAX_CASE_DEADLINE_SECONDS",
    "MAX_LIVE_TOKENS",
    "MAX_SELECTED_ROWS",
    "QUARANTINE_SCHEMA",
    "RECORD_SCHEMA",
    "V13StoreError",
    "append_case_record",
    "begin_case",
    "build_bounds",
    "build_identity",
    "canonical_json_bytes",
    "file_sha256",
    "finalize_case",
    "identity_sha256",
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
