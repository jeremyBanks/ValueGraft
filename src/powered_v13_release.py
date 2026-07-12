"""Pure-Git Stage-A release foundation for the powered-v13 successor.

This module deliberately has no model, provider, fixture, or treatment imports.
It implements only the acyclic Stage-A boundary described by Sections 15 and
18 of the v13 preregistration:

* a canonical manifest inventories explicit blobs from an immutable parent;
* the authorization child has exactly one parent and changes exactly two paths;
* the preregistration change is one manifest-configured, exact line replacement;
* the other change is one newly added, canonical manifest;
* a post-commit receipt is written with create-if-absent semantics; and
* launch verification requires exact detached HEAD, a clean checkout, one
  receipt, exact bindings, and a fresh receipt timestamp.

The old and new status lines are opaque UTF-8 bytes supplied to the manifest
builder.  The module intentionally does not contain the real authorization
line and therefore cannot guess or perform the live v13 status transition.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, Sequence


DESIGN_ID = "coherent-state-powered-successor-v13"
STAGE_A = "STATIC_PHASE_A"
STAGE_A_MANIFEST_SCHEMA = "powered-v13-stage-a-manifest-v1"
STAGE_A_RECEIPT_SCHEMA = "powered-v13-stage-a-launch-receipt-v1"
DEFAULT_RECEIPT_MAX_AGE_SECONDS = 300
DEFAULT_RECEIPT_FUTURE_SKEW_SECONDS = 5

_GIT_OID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_HEX_RE = re.compile(r"(?:[0-9a-f]{2})+\Z")
_SAFE_PATH_RE = re.compile(r"[A-Za-z0-9._/-]+\Z")
_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z\Z")


class ReleaseVerificationError(RuntimeError):
    """A Stage-A release or launch condition failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON representation used by release records."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReleaseVerificationError(f"value is not canonical-JSON safe: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(
    repo: Path,
    *args: str,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise ReleaseVerificationError(f"git invocation failed: {exc}") from exc
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ReleaseVerificationError(
            f"git {' '.join(args)} failed with {result.returncode}: {detail}"
        )
    return result


def _git_text(repo: Path, *args: str) -> str:
    return _git(repo, *args).stdout.decode("utf-8", "strict")


def _require_exact_commit(repo: Path, value: Any, label: str) -> str:
    if not isinstance(value, str) or not _GIT_OID_RE.fullmatch(value):
        raise ReleaseVerificationError(f"{label} is not an exact lowercase Git OID")
    resolved = _git_text(repo, "rev-parse", "--verify", f"{value}^{{commit}}").strip()
    if resolved != value:
        raise ReleaseVerificationError(f"{label} does not name its exact commit")
    return value


def _safe_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or not _SAFE_PATH_RE.fullmatch(value):
        raise ReleaseVerificationError(f"{label} is not a safe repository-relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.as_posix() != value or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise ReleaseVerificationError(f"{label} is not a normalized relative path")
    if pure.parts[0] == ".git":
        raise ReleaseVerificationError(f"{label} may not address .git")
    return value


def _tree_entry(repo: Path, commit: str, path: str) -> tuple[str, str, bytes]:
    raw = _git(repo, "ls-tree", "-z", "--full-tree", commit, "--", path).stdout
    records = [record for record in raw.split(b"\0") if record]
    matching: list[tuple[str, str, str]] = []
    for record in records:
        try:
            header, raw_name = record.split(b"\t", 1)
            mode, kind, oid = header.decode("ascii").split(" ")
            name = raw_name.decode("utf-8", "strict")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ReleaseVerificationError(f"malformed ls-tree record for {path}") from exc
        if name == path:
            matching.append((mode, kind, oid))
    if len(matching) != 1:
        raise ReleaseVerificationError(
            f"{path} is absent or ambiguous in parent tree {commit}"
        )
    mode, kind, oid = matching[0]
    if kind != "blob" or mode not in {"100644", "100755"}:
        raise ReleaseVerificationError(
            f"{path} is not a regular tracked blob in {commit}: {mode} {kind}"
        )
    blob = _git(repo, "cat-file", "blob", oid).stdout
    return mode, oid, blob


def _path_absent(repo: Path, commit: str, path: str) -> bool:
    raw = _git(repo, "ls-tree", "-z", "--full-tree", commit, "--", path).stdout
    return not any(record for record in raw.split(b"\0") if record)


def _inventory(repo: Path, commit: str, paths: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        mode, _oid, blob = _tree_entry(repo, commit, path)
        rows.append(
            {
                "path": path,
                "mode": mode,
                "bytes": len(blob),
                "sha256": sha256_bytes(blob),
            }
        )
    return rows


def _status_line(value: bytes, label: str) -> bytes:
    if not isinstance(value, bytes) or not value or len(value) > 4096:
        raise ReleaseVerificationError(f"{label} must be nonempty bounded bytes")
    if value.count(b"\n") != 1 or not value.endswith(b"\n") or b"\r" in value:
        raise ReleaseVerificationError(f"{label} must be exactly one LF-terminated line")
    if b"\x00" in value:
        raise ReleaseVerificationError(f"{label} contains NUL")
    try:
        value.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise ReleaseVerificationError(f"{label} is not UTF-8") from exc
    return value


def _decode_lower_hex(value: Any, label: str) -> bytes:
    if not isinstance(value, str) or not _HEX_RE.fullmatch(value):
        raise ReleaseVerificationError(f"{label} is not nonempty lowercase byte hex")
    return bytes.fromhex(value)


def build_stage_a_manifest(
    repo: Path,
    *,
    static_root_commit: str,
    experiment_paths: Sequence[str],
    preregistration_path: str,
    manifest_path: str,
    old_status_line: bytes,
    new_status_line: bytes,
) -> dict[str, Any]:
    """Build, but do not write, a deterministic Stage-A parent-tree manifest.

    ``experiment_paths`` is deliberately explicit: no filesystem glob or live
    worktree read can silently expand the frozen inventory.  Every byte is read
    from ``static_root_commit`` through Git's object database.
    """
    repo = Path(repo).resolve()
    root = _require_exact_commit(repo, static_root_commit, "static_root_commit")
    prereg = _safe_path(preregistration_path, "preregistration_path")
    manifest_rel = _safe_path(manifest_path, "manifest_path")
    if not manifest_rel.endswith(".json"):
        raise ReleaseVerificationError("manifest_path must end in .json")
    if not isinstance(experiment_paths, Sequence) or isinstance(
        experiment_paths, (str, bytes)
    ):
        raise ReleaseVerificationError("experiment_paths must be a path sequence")
    normalized = [_safe_path(path, "experiment path") for path in experiment_paths]
    if not normalized or len(normalized) != len(set(normalized)):
        raise ReleaseVerificationError("experiment paths are empty or duplicated")
    normalized = sorted(normalized)
    if prereg not in normalized:
        raise ReleaseVerificationError("experiment inventory omits the preregistration")
    if manifest_rel in normalized:
        raise ReleaseVerificationError("new manifest may not inventory itself")
    if not _path_absent(repo, root, manifest_rel):
        raise ReleaseVerificationError("Stage-A manifest path already exists in static root")

    old = _status_line(old_status_line, "old_status_line")
    new = _status_line(new_status_line, "new_status_line")
    if old == new:
        raise ReleaseVerificationError("status transition does not change bytes")
    _mode, _oid, prereg_bytes = _tree_entry(repo, root, prereg)
    if prereg_bytes.count(old) != 1:
        raise ReleaseVerificationError("old status line is not unique in the parent")
    if new in prereg_bytes:
        raise ReleaseVerificationError("new status line already appears in the parent")

    rows = _inventory(repo, root, normalized)
    root_tree = _git_text(repo, "rev-parse", f"{root}^{{tree}}").strip()
    return {
        "schema": STAGE_A_MANIFEST_SCHEMA,
        "design_id": DESIGN_ID,
        "stage": STAGE_A,
        "static_root_commit": root,
        "static_root_tree": root_tree,
        "manifest_path": manifest_rel,
        "preregistration_path": prereg,
        "status_transition": {
            "path": prereg,
            "old_bytes_hex": old.hex(),
            "new_bytes_hex": new.hex(),
        },
        "inventory_count": len(rows),
        "inventory": rows,
        "inventory_sha256": sha256_bytes(canonical_json_bytes(rows)),
    }


def encode_stage_a_manifest(manifest: Mapping[str, Any]) -> bytes:
    """Encode a manifest as the only accepted on-disk byte representation."""
    return canonical_json_bytes(dict(manifest)) + b"\n"


def _require_object(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ReleaseVerificationError(f"{label} fields differ from the frozen schema")
    return value


def _require_plain_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ReleaseVerificationError(f"{label} is not a nonnegative integer")
    return value


def verify_stage_a_manifest_document(
    repo: Path,
    manifest: Mapping[str, Any],
    *,
    expected_manifest_path: str | None = None,
) -> dict[str, Any]:
    """Recompute every manifest binding from the named immutable parent."""
    repo = Path(repo).resolve()
    fields = {
        "schema",
        "design_id",
        "stage",
        "static_root_commit",
        "static_root_tree",
        "manifest_path",
        "preregistration_path",
        "status_transition",
        "inventory_count",
        "inventory",
        "inventory_sha256",
    }
    doc = _require_object(manifest, fields, "Stage-A manifest")
    if (
        doc["schema"] != STAGE_A_MANIFEST_SCHEMA
        or doc["design_id"] != DESIGN_ID
        or doc["stage"] != STAGE_A
    ):
        raise ReleaseVerificationError("Stage-A manifest identity differs")
    root = _require_exact_commit(repo, doc["static_root_commit"], "static_root_commit")
    expected_tree = _git_text(repo, "rev-parse", f"{root}^{{tree}}").strip()
    if doc["static_root_tree"] != expected_tree:
        raise ReleaseVerificationError("static root tree hash differs")
    manifest_rel = _safe_path(doc["manifest_path"], "manifest_path")
    prereg = _safe_path(doc["preregistration_path"], "preregistration_path")
    if expected_manifest_path is not None and manifest_rel != _safe_path(
        expected_manifest_path, "expected_manifest_path"
    ):
        raise ReleaseVerificationError("manifest path binding differs")
    if not manifest_rel.endswith(".json") or not _path_absent(repo, root, manifest_rel):
        raise ReleaseVerificationError("manifest path is not a new .json path")

    transition = _require_object(
        doc["status_transition"],
        {"path", "old_bytes_hex", "new_bytes_hex"},
        "status transition",
    )
    if transition["path"] != prereg:
        raise ReleaseVerificationError("status transition names another path")
    old = _status_line(
        _decode_lower_hex(transition["old_bytes_hex"], "old status hex"),
        "old status line",
    )
    new = _status_line(
        _decode_lower_hex(transition["new_bytes_hex"], "new status hex"),
        "new status line",
    )
    if old == new:
        raise ReleaseVerificationError("status transition does not change bytes")

    inventory = doc["inventory"]
    if not isinstance(inventory, list) or not inventory:
        raise ReleaseVerificationError("manifest inventory is empty or not a list")
    if _require_plain_int(doc["inventory_count"], "inventory_count") != len(inventory):
        raise ReleaseVerificationError("inventory_count differs")
    paths: list[str] = []
    for index, raw_row in enumerate(inventory):
        row = _require_object(
            raw_row, {"path", "mode", "bytes", "sha256"}, f"inventory[{index}]"
        )
        path = _safe_path(row["path"], f"inventory[{index}].path")
        if row["mode"] not in {"100644", "100755"}:
            raise ReleaseVerificationError(f"inventory[{index}].mode differs")
        _require_plain_int(row["bytes"], f"inventory[{index}].bytes")
        if not isinstance(row["sha256"], str) or not _SHA256_RE.fullmatch(row["sha256"]):
            raise ReleaseVerificationError(f"inventory[{index}].sha256 differs")
        paths.append(path)
    if paths != sorted(set(paths)):
        raise ReleaseVerificationError("inventory paths are not sorted and unique")
    if prereg not in paths or manifest_rel in paths:
        raise ReleaseVerificationError("inventory preregistration/manifest membership differs")
    expected_inventory = _inventory(repo, root, paths)
    if inventory != expected_inventory:
        raise ReleaseVerificationError("manifest inventory differs from static parent bytes")
    expected_inventory_sha = sha256_bytes(canonical_json_bytes(expected_inventory))
    if doc["inventory_sha256"] != expected_inventory_sha:
        raise ReleaseVerificationError("manifest inventory aggregate hash differs")

    _mode, _oid, prereg_bytes = _tree_entry(repo, root, prereg)
    if prereg_bytes.count(old) != 1:
        raise ReleaseVerificationError("old status line is not unique in the static parent")
    if new in prereg_bytes:
        raise ReleaseVerificationError("new status line already appears in static parent")
    return {
        "static_root_commit": root,
        "static_root_tree": expected_tree,
        "manifest_path": manifest_rel,
        "preregistration_path": prereg,
        "old_status_line": old,
        "new_status_line": new,
        "inventory": expected_inventory,
        "inventory_sha256": expected_inventory_sha,
    }


def _read_json_blob(repo: Path, commit: str, path: str, label: str) -> tuple[dict[str, Any], bytes]:
    _mode, _oid, raw = _tree_entry(repo, commit, path)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError(f"{label} is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseVerificationError(f"{label} root is not an object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise ReleaseVerificationError(f"{label} bytes are not canonical JSON plus LF")
    return value, raw


def _commit_parents(repo: Path, commit: str) -> list[str]:
    line = _git_text(repo, "rev-list", "--parents", "-n", "1", commit).strip()
    fields = line.split()
    if not fields or fields[0] != commit:
        raise ReleaseVerificationError("could not obtain exact authorization parents")
    return fields[1:]


def _diff_name_status(repo: Path, parent: str, child: str) -> list[tuple[str, str]]:
    text = _git_text(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "--no-renames",
        "-r",
        parent,
        child,
    )
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] not in {"A", "D", "M", "T"}:
            raise ReleaseVerificationError("authorization diff has an unsupported record")
        rows.append((fields[0], _safe_path(fields[1], "changed path")))
    return rows


def _reject_duplicate_changed_manifests(
    repo: Path, child: str, changes: Sequence[tuple[str, str]], manifest_path: str
) -> None:
    matches: list[str] = []
    for status, path in changes:
        if status == "D" or not path.endswith(".json"):
            continue
        try:
            _mode, _oid, raw = _tree_entry(repo, child, path)
            value = json.loads(raw)
        except (ReleaseVerificationError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("schema") == STAGE_A_MANIFEST_SCHEMA:
            matches.append(path)
    if matches != [manifest_path]:
        if len(matches) > 1:
            raise ReleaseVerificationError(f"duplicate Stage-A manifests in child diff: {matches}")
        raise ReleaseVerificationError("authorization child lacks its unique Stage-A manifest")


def verify_stage_a_authorization_commit(
    repo: Path,
    *,
    authorization_commit: str,
    manifest_path: str,
) -> dict[str, Any]:
    """Verify the one-parent, two-path Stage-A authorization commit."""
    repo = Path(repo).resolve()
    child = _require_exact_commit(repo, authorization_commit, "authorization_commit")
    manifest_rel = _safe_path(manifest_path, "manifest_path")
    parents = _commit_parents(repo, child)
    if len(parents) != 1:
        raise ReleaseVerificationError(
            f"authorization commit must have exactly one parent, observed {len(parents)}"
        )
    parent = parents[0]
    manifest, manifest_raw = _read_json_blob(
        repo, child, manifest_rel, "Stage-A manifest"
    )
    binding = verify_stage_a_manifest_document(
        repo, manifest, expected_manifest_path=manifest_rel
    )
    if binding["static_root_commit"] != parent:
        raise ReleaseVerificationError(
            "authorization parent differs from manifest static_root_commit"
        )

    changes = _diff_name_status(repo, parent, child)
    _reject_duplicate_changed_manifests(repo, child, changes, manifest_rel)
    prereg = binding["preregistration_path"]
    expected_changes = sorted([("A", manifest_rel), ("M", prereg)])
    if sorted(changes) != expected_changes:
        raise ReleaseVerificationError(
            f"authorization diff paths/statuses differ: {changes!r}"
        )

    parent_mode, _parent_oid, parent_bytes = _tree_entry(repo, parent, prereg)
    child_mode, _child_oid, child_bytes = _tree_entry(repo, child, prereg)
    if child_mode != parent_mode:
        raise ReleaseVerificationError("preregistration mode changed in authorization child")
    expected_child_bytes = parent_bytes.replace(
        binding["old_status_line"], binding["new_status_line"], 1
    )
    if child_bytes != expected_child_bytes:
        raise ReleaseVerificationError(
            "preregistration child is not the exact configured one-line transition"
        )
    manifest_mode, _manifest_oid, reread_manifest_raw = _tree_entry(
        repo, child, manifest_rel
    )
    if manifest_mode != "100644":
        raise ReleaseVerificationError("Stage-A manifest is not mode 100644")
    if reread_manifest_raw != manifest_raw:
        raise ReleaseVerificationError("Stage-A manifest changed during verification")
    return {
        "authorization_commit": child,
        "static_root_commit": parent,
        "static_root_tree": binding["static_root_tree"],
        "manifest_path": manifest_rel,
        "manifest_sha256": sha256_bytes(manifest_raw),
        "inventory_count": len(binding["inventory"]),
        "inventory_sha256": binding["inventory_sha256"],
        "preregistration_path": prereg,
        "changed_paths": [path for _status, path in changes],
    }


def _format_utc(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ReleaseVerificationError("receipt timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not _UTC_RE.fullmatch(value):
        raise ReleaseVerificationError(f"{label} is not canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise ReleaseVerificationError(f"{label} is not a valid UTC timestamp") from exc
    return parsed.replace(tzinfo=timezone.utc)


def receipt_payload_sha256(receipt: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    return sha256_bytes(canonical_json_bytes(payload))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ReleaseVerificationError(f"launch receipt already exists: {path}") from exc
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def create_stage_a_launch_receipt(
    repo: Path,
    output_path: Path,
    *,
    authorization_commit: str,
    manifest_path: str,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Create one exact post-commit receipt without replacing prior bytes."""
    evidence = verify_stage_a_authorization_commit(
        repo,
        authorization_commit=authorization_commit,
        manifest_path=manifest_path,
    )
    created = _format_utc(created_at or datetime.now(timezone.utc))
    receipt: dict[str, Any] = {
        "schema": STAGE_A_RECEIPT_SCHEMA,
        "design_id": DESIGN_ID,
        "stage": STAGE_A,
        "authorization_commit": evidence["authorization_commit"],
        "static_root_commit": evidence["static_root_commit"],
        "manifest_path": evidence["manifest_path"],
        "manifest_sha256": evidence["manifest_sha256"],
        "created_utc": created,
    }
    receipt["payload_sha256"] = receipt_payload_sha256(receipt)
    encoded = canonical_json_bytes(receipt) + b"\n"
    _write_bytes_exclusive(Path(output_path), encoded)
    if Path(output_path).read_bytes() != encoded:
        raise ReleaseVerificationError("launch receipt changed during exclusive write")
    return receipt


def _read_receipt(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except FileNotFoundError as exc:
        raise ReleaseVerificationError(f"launch receipt is absent: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseVerificationError(f"launch receipt is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseVerificationError("launch receipt root is not an object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise ReleaseVerificationError("launch receipt bytes are not canonical JSON plus LF")
    return value, raw


def verify_stage_a_launch_receipt(
    receipt_path: Path,
    *,
    authorization_evidence: Mapping[str, Any],
    now: datetime | None = None,
    max_age_seconds: int = DEFAULT_RECEIPT_MAX_AGE_SECONDS,
    future_skew_seconds: int = DEFAULT_RECEIPT_FUTURE_SKEW_SECONDS,
) -> dict[str, Any]:
    """Verify exact release bindings and the bounded receipt age."""
    if type(max_age_seconds) is not int or max_age_seconds <= 0:
        raise ReleaseVerificationError("max_age_seconds must be a positive integer")
    if type(future_skew_seconds) is not int or future_skew_seconds < 0:
        raise ReleaseVerificationError("future_skew_seconds must be a nonnegative integer")
    fields = {
        "schema",
        "design_id",
        "stage",
        "authorization_commit",
        "static_root_commit",
        "manifest_path",
        "manifest_sha256",
        "created_utc",
        "payload_sha256",
    }
    receipt, raw = _read_receipt(Path(receipt_path))
    _require_object(receipt, fields, "Stage-A launch receipt")
    if (
        receipt["schema"] != STAGE_A_RECEIPT_SCHEMA
        or receipt["design_id"] != DESIGN_ID
        or receipt["stage"] != STAGE_A
    ):
        raise ReleaseVerificationError("launch receipt identity differs")
    if not isinstance(receipt["payload_sha256"], str) or not _SHA256_RE.fullmatch(
        receipt["payload_sha256"]
    ):
        raise ReleaseVerificationError("launch receipt payload_sha256 is malformed")
    if receipt["payload_sha256"] != receipt_payload_sha256(receipt):
        raise ReleaseVerificationError("launch receipt payload hash mismatch")
    expected = {
        "authorization_commit": authorization_evidence.get("authorization_commit"),
        "static_root_commit": authorization_evidence.get("static_root_commit"),
        "manifest_path": authorization_evidence.get("manifest_path"),
        "manifest_sha256": authorization_evidence.get("manifest_sha256"),
    }
    for key, expected_value in expected.items():
        if receipt[key] != expected_value:
            raise ReleaseVerificationError(f"launch receipt {key} binding differs")
    created = _parse_utc(receipt["created_utc"], "created_utc")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ReleaseVerificationError("receipt verification time is timezone-naive")
    age = (current.astimezone(timezone.utc) - created).total_seconds()
    if age < -future_skew_seconds:
        raise ReleaseVerificationError("launch receipt timestamp is too far in the future")
    if age > max_age_seconds:
        raise ReleaseVerificationError("launch receipt is stale")
    return {
        "receipt_path": str(Path(receipt_path).resolve()),
        "receipt_sha256": sha256_bytes(raw),
        "payload_sha256": receipt["payload_sha256"],
        "created_utc": receipt["created_utc"],
        "age_seconds": age,
    }


def _require_detached_head(repo: Path, expected_commit: str) -> None:
    head = _git_text(repo, "rev-parse", "HEAD").strip()
    if head != expected_commit:
        raise ReleaseVerificationError(
            f"checkout HEAD {head} differs from authorization commit {expected_commit}"
        )
    symbolic = _git(repo, "symbolic-ref", "-q", "HEAD", check=False)
    if symbolic.returncode == 0:
        branch = symbolic.stdout.decode("utf-8", "replace").strip()
        raise ReleaseVerificationError(f"authorization checkout is on branch {branch}")
    if symbolic.returncode != 1:
        detail = symbolic.stderr.decode("utf-8", "replace").strip()
        raise ReleaseVerificationError(f"could not verify detached HEAD: {detail}")


def _require_clean_tree(repo: Path) -> None:
    status = _git(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ).stdout
    if status:
        rendered = status.replace(b"\0", b"\n").decode("utf-8", "replace").strip()
        raise ReleaseVerificationError(f"authorization checkout is dirty: {rendered}")


def verify_stage_a_checkout(
    repo: Path,
    *,
    authorization_commit: str,
    manifest_path: str,
    receipt_paths: Sequence[Path],
    now: datetime | None = None,
    max_receipt_age_seconds: int = DEFAULT_RECEIPT_MAX_AGE_SECONDS,
    future_skew_seconds: int = DEFAULT_RECEIPT_FUTURE_SKEW_SECONDS,
) -> dict[str, Any]:
    """Verify the complete local analogue of the remote Stage-A launch gate."""
    repo = Path(repo).resolve()
    exact_commit = _require_exact_commit(repo, authorization_commit, "authorization_commit")
    _require_detached_head(repo, exact_commit)
    _require_clean_tree(repo)
    if not isinstance(receipt_paths, Sequence) or isinstance(receipt_paths, (str, bytes)):
        raise ReleaseVerificationError("receipt_paths must be a path sequence")
    if len(receipt_paths) != 1:
        raise ReleaseVerificationError(
            f"exactly one Stage-A launch receipt is required, observed {len(receipt_paths)}"
        )
    authorization = verify_stage_a_authorization_commit(
        repo,
        authorization_commit=exact_commit,
        manifest_path=manifest_path,
    )
    receipt = verify_stage_a_launch_receipt(
        Path(receipt_paths[0]),
        authorization_evidence=authorization,
        now=now,
        max_age_seconds=max_receipt_age_seconds,
        future_skew_seconds=future_skew_seconds,
    )
    return {
        "status": "PASS",
        "design_id": DESIGN_ID,
        "stage": STAGE_A,
        "detached_head": exact_commit,
        "clean_tree": True,
        "authorization": authorization,
        "receipt": receipt,
    }
