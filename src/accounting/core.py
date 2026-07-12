"""Shared, deterministic primitives for the final project accounting.

This module deliberately does not know about any provider.  It provides two
small foundations used by every provider-specific ledger:

* JSON/SHA-256 serialization that never round-trips money through ``float``;
* immutable byte-prefix manifests for append-only local transcript files.

The manifest is a snapshot of byte prefixes, not a claim that the source files
will never grow.  Re-verification permits appends, but rejects a changed source
set, replacement, truncation, or mutation of any captured prefix.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Callable, Iterable, Iterator, Mapping


class AccountingError(RuntimeError):
    """Base error for final-accounting invariants."""


class AccountingSchemaError(AccountingError):
    """Raised when a classified ledger row is not internally coherent."""


class FreezeError(AccountingError):
    """Raised when a source set cannot be frozen or later re-verified."""


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe value while preserving exact decimal spellings.

    Audit quantities must arrive as ``Decimal`` or integers.  Rejecting floats
    prevents an apparently precise artifact from silently serializing a binary
    approximation of a provider charge.
    """

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise TypeError("non-finite Decimal is not valid accounting JSON")
        return format(value, "f")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        raise TypeError("float is forbidden in deterministic accounting JSON; use Decimal")
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("accounting JSON object keys must be strings")
            result[key] = _json_safe(child)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    raise TypeError(f"unsupported accounting JSON type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize ``value`` with one deterministic, UTF-8-safe representation."""

    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    """Hash the canonical JSON representation of ``value``."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class LedgerName(str, Enum):
    METERED_CONSUMPTION = "metered_consumption"
    CASH_TRANSACTIONS = "cash_transactions"
    SUBSCRIPTION_WORKLOAD = "subscription_workload"
    COUNTERFACTUAL_PRICING = "counterfactual_pricing"
    EXPERIMENTAL_COMPUTE = "experimental_compute"
    UNKNOWNS = "unknowns"


class MeasurementClass(str, Enum):
    EXACT_SOURCE_RECORD = "exact_source_record"
    RECONSTRUCTED = "reconstructed"
    ESTIMATED = "estimated"
    LOWER_BOUND = "lower_bound"
    UNKNOWN = "unknown"


class AccountingRole(str, Enum):
    PRIMARY_ADDITIVE = "primary_additive"
    CROSSCHECK_NONADDITIVE = "crosscheck_nonadditive"
    CONTEXT_NONADDITIVE = "context_nonadditive"


def _require_nonempty(label: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise AccountingSchemaError(f"{label} must be a nonempty string")


def _require_utc_timestamp(label: str, value: str | None) -> None:
    if value is None:
        return
    _require_nonempty(label, value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AccountingSchemaError(f"{label} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise AccountingSchemaError(f"{label} must identify UTC")


@dataclass(frozen=True)
class LedgerRow:
    """One strictly classified quantity in one non-overlapping ledger."""

    row_id: str
    ledger: LedgerName
    provider: str
    category: str
    quantity: Decimal | None
    unit: str
    measurement_class: MeasurementClass
    accounting_role: AccountingRole
    evidence_refs: tuple[str, ...]
    method_version: str
    currency: str | None = None
    overlap_key: str | None = None
    window_start_utc: str | None = None
    source_cutoff_utc: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        for label, value in (
            ("row_id", self.row_id),
            ("provider", self.provider),
            ("category", self.category),
            ("unit", self.unit),
            ("method_version", self.method_version),
        ):
            _require_nonempty(label, value)
        if not isinstance(self.ledger, LedgerName):
            raise AccountingSchemaError("ledger must be a LedgerName")
        if not isinstance(self.measurement_class, MeasurementClass):
            raise AccountingSchemaError("measurement_class must be a MeasurementClass")
        if not isinstance(self.accounting_role, AccountingRole):
            raise AccountingSchemaError("accounting_role must be an AccountingRole")
        if self.quantity is not None and not isinstance(self.quantity, Decimal):
            raise AccountingSchemaError("quantity must be Decimal or None; floats are forbidden")
        if self.quantity is not None and not self.quantity.is_finite():
            raise AccountingSchemaError("quantity must be finite")
        if self.measurement_class is MeasurementClass.UNKNOWN:
            if self.quantity is not None:
                raise AccountingSchemaError("unknown rows must have quantity=None")
            if self.accounting_role is AccountingRole.PRIMARY_ADDITIVE:
                raise AccountingSchemaError("unknown rows cannot be primary additive")
        elif self.quantity is None:
            raise AccountingSchemaError("non-unknown rows require a Decimal quantity")
        if self.currency is not None:
            _require_nonempty("currency", self.currency)
        if self.overlap_key is not None:
            _require_nonempty("overlap_key", self.overlap_key)
        _require_utc_timestamp("window_start_utc", self.window_start_utc)
        _require_utc_timestamp("source_cutoff_utc", self.source_cutoff_utc)
        if not isinstance(self.note, str):
            raise AccountingSchemaError("note must be a string")
        if not isinstance(self.evidence_refs, tuple) or not self.evidence_refs:
            raise AccountingSchemaError("evidence_refs must be a nonempty tuple")
        if any(not isinstance(ref, str) or not ref.strip() for ref in self.evidence_refs):
            raise AccountingSchemaError("evidence_refs contain an empty or non-string value")
        normalized_refs = tuple(sorted(set(self.evidence_refs)))
        object.__setattr__(self, "evidence_refs", normalized_refs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "ledger": self.ledger.value,
            "provider": self.provider,
            "category": self.category,
            "quantity": None if self.quantity is None else format(self.quantity, "f"),
            "unit": self.unit,
            "currency": self.currency,
            "measurement_class": self.measurement_class.value,
            "accounting_role": self.accounting_role.value,
            "overlap_key": self.overlap_key,
            "window_start_utc": self.window_start_utc,
            "source_cutoff_utc": self.source_cutoff_utc,
            "evidence_refs": list(self.evidence_refs),
            "method_version": self.method_version,
            "note": self.note,
        }


@dataclass(frozen=True)
class FrozenFileRecord:
    logical_path: str
    prefix_bytes: int
    mtime_ns_at_capture: int
    device_at_capture: int
    inode_at_capture: int
    prefix_sha256: str
    parsed_complete_lines: int
    malformed_complete_lines: int
    ignored_partial_trailing_lines: int

    def __post_init__(self) -> None:
        logical = PurePath(self.logical_path)
        if logical.is_absolute() or not self.logical_path or ".." in logical.parts:
            raise FreezeError(f"invalid frozen logical path: {self.logical_path!r}")
        for field in (
            "prefix_bytes",
            "mtime_ns_at_capture",
            "device_at_capture",
            "inode_at_capture",
            "parsed_complete_lines",
            "malformed_complete_lines",
            "ignored_partial_trailing_lines",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FreezeError(f"{field} must be a nonnegative integer")
        if len(self.prefix_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.prefix_sha256
        ):
            raise FreezeError("prefix_sha256 is not a lowercase SHA-256 digest")
        if self.ignored_partial_trailing_lines not in {0, 1}:
            raise FreezeError("a file can contain at most one partial trailing line")

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_path": self.logical_path,
            "prefix_bytes": self.prefix_bytes,
            "mtime_ns_at_capture": self.mtime_ns_at_capture,
            "device_at_capture": self.device_at_capture,
            "inode_at_capture": self.inode_at_capture,
            "prefix_sha256": self.prefix_sha256,
            "parsed_complete_lines": self.parsed_complete_lines,
            "malformed_complete_lines": self.malformed_complete_lines,
            "ignored_partial_trailing_lines": self.ignored_partial_trailing_lines,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenFileRecord":
        return cls(
            logical_path=str(value["logical_path"]),
            prefix_bytes=int(value["prefix_bytes"]),
            mtime_ns_at_capture=int(value["mtime_ns_at_capture"]),
            device_at_capture=int(value["device_at_capture"]),
            inode_at_capture=int(value["inode_at_capture"]),
            prefix_sha256=str(value["prefix_sha256"]),
            parsed_complete_lines=int(value["parsed_complete_lines"]),
            malformed_complete_lines=int(value["malformed_complete_lines"]),
            ignored_partial_trailing_lines=int(value["ignored_partial_trailing_lines"]),
        )


@dataclass(frozen=True)
class FreezeManifest:
    schema: str
    logical_root: str
    capture_started_at_utc: str
    capture_completed_at_utc: str
    source_set_sha256: str
    prefix_bytes_total: int
    files: tuple[FrozenFileRecord, ...]
    manifest_sha256: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "logical_root": self.logical_root,
            "capture_started_at_utc": self.capture_started_at_utc,
            "capture_completed_at_utc": self.capture_completed_at_utc,
            "source_set_sha256": self.source_set_sha256,
            "prefix_bytes_total": self.prefix_bytes_total,
            "file_count": len(self.files),
            "files": [record.to_dict() for record in self.files],
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.unsigned_dict()
        result["manifest_sha256"] = self.manifest_sha256
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FreezeManifest":
        files = tuple(FrozenFileRecord.from_dict(row) for row in value["files"])
        if int(value.get("file_count", len(files))) != len(files):
            raise FreezeError("manifest file_count does not match files")
        manifest = cls(
            schema=str(value["schema"]),
            logical_root=str(value["logical_root"]),
            capture_started_at_utc=str(value["capture_started_at_utc"]),
            capture_completed_at_utc=str(value["capture_completed_at_utc"]),
            source_set_sha256=str(value["source_set_sha256"]),
            prefix_bytes_total=int(value["prefix_bytes_total"]),
            files=files,
            manifest_sha256=str(value["manifest_sha256"]),
        )
        _verify_manifest_self_hash(manifest)
        return manifest


@dataclass(frozen=True)
class FrozenJsonRecord:
    logical_path: str
    line_number: int
    value: Mapping[str, Any]


DiscoverFiles = Callable[[Path], Iterable[Path]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _logical_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise FreezeError(f"source path escapes root: {path}") from exc
    logical = relative.as_posix()
    if not logical or logical.startswith("/") or ".." in relative.parts:
        raise FreezeError(f"invalid logical source path: {logical!r}")
    return logical


def _enumerate_sources(root: Path, discover: DiscoverFiles) -> tuple[tuple[str, Path], ...]:
    rows: dict[str, Path] = {}
    for candidate in discover(root):
        path = Path(candidate).expanduser().resolve()
        if not path.is_file():
            raise FreezeError(f"discovered source is not a regular file: {path}")
        logical = _logical_path(root, path)
        if logical in rows and rows[logical] != path:
            raise FreezeError(f"duplicate logical path discovered: {logical}")
        rows[logical] = path
    return tuple(sorted(rows.items()))


def _inspect_prefix(path: Path, prefix_bytes: int) -> tuple[str, int, int, int]:
    digest = hashlib.sha256()
    parsed = 0
    malformed = 0
    ignored_partial = 0
    remaining = prefix_bytes
    with path.open("rb") as handle:
        while remaining:
            raw = handle.readline(remaining)
            if not raw:
                raise FreezeError(f"short read while hashing captured prefix: {path}")
            digest.update(raw)
            remaining -= len(raw)
            is_final = remaining == 0
            if is_final and not raw.endswith(b"\n"):
                ignored_partial += 1
                continue
            try:
                value = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                malformed += 1
            else:
                if isinstance(value, Mapping):
                    parsed += 1
                else:
                    malformed += 1
    return digest.hexdigest(), parsed, malformed, ignored_partial


def _prefix_sha256(path: Path, prefix_bytes: int) -> str:
    digest = hashlib.sha256()
    remaining = prefix_bytes
    with path.open("rb") as handle:
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise FreezeError(f"short read while verifying captured prefix: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _source_set_sha256(paths: Iterable[str]) -> str:
    return canonical_sha256(list(paths))


def _verify_manifest_self_hash(manifest: FreezeManifest) -> None:
    expected = canonical_sha256(manifest.unsigned_dict())
    if manifest.manifest_sha256 != expected:
        raise FreezeError(
            f"freeze manifest self-hash mismatch: {manifest.manifest_sha256} != {expected}"
        )


def freeze_file_set(
    root: Path,
    *,
    logical_root: str,
    discover: DiscoverFiles,
    fail_on_malformed_complete_line: bool = True,
) -> FreezeManifest:
    """Capture and immediately re-verify a stable set of JSONL byte prefixes.

    ``discover`` is invoked twice.  The second enumeration is not advisory: a
    different logical source set aborts the freeze.
    """

    _require_nonempty("logical_root", logical_root)
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FreezeError(f"freeze root is not a directory: {root}")
    capture_started = _utc_now_iso()
    first = _enumerate_sources(root, discover)
    records: list[FrozenFileRecord] = []
    for logical, path in first:
        stat = path.stat()
        digest, parsed, malformed, partial = _inspect_prefix(path, stat.st_size)
        if fail_on_malformed_complete_line and malformed:
            raise FreezeError(f"{logical} contains {malformed} malformed complete JSONL lines")
        records.append(
            FrozenFileRecord(
                logical_path=logical,
                prefix_bytes=stat.st_size,
                mtime_ns_at_capture=stat.st_mtime_ns,
                device_at_capture=stat.st_dev,
                inode_at_capture=stat.st_ino,
                prefix_sha256=digest,
                parsed_complete_lines=parsed,
                malformed_complete_lines=malformed,
                ignored_partial_trailing_lines=partial,
            )
        )
    second = _enumerate_sources(root, discover)
    first_names = tuple(logical for logical, _ in first)
    second_names = tuple(logical for logical, _ in second)
    if first_names != second_names:
        raise FreezeError(
            f"source set changed during freeze: before={first_names!r}, after={second_names!r}"
        )
    by_name = dict(second)
    for record in records:
        path = by_name[record.logical_path]
        stat = path.stat()
        if stat.st_dev != record.device_at_capture or stat.st_ino != record.inode_at_capture:
            raise FreezeError(f"source was replaced during freeze: {record.logical_path}")
        if stat.st_size < record.prefix_bytes:
            raise FreezeError(f"source shrank during freeze: {record.logical_path}")
        if _prefix_sha256(path, record.prefix_bytes) != record.prefix_sha256:
            raise FreezeError(f"captured prefix changed during freeze: {record.logical_path}")
    capture_completed = _utc_now_iso()
    provisional = FreezeManifest(
        schema="accounting_prefix_freeze_v1",
        logical_root=logical_root,
        capture_started_at_utc=capture_started,
        capture_completed_at_utc=capture_completed,
        source_set_sha256=_source_set_sha256(first_names),
        prefix_bytes_total=sum(record.prefix_bytes for record in records),
        files=tuple(records),
        manifest_sha256="",
    )
    return replace(provisional, manifest_sha256=canonical_sha256(provisional.unsigned_dict()))


def verify_freeze_manifest(
    manifest: FreezeManifest,
    root: Path,
    *,
    discover: DiscoverFiles | None = None,
) -> None:
    """Re-verify a manifest against its original logical source root."""

    _verify_manifest_self_hash(manifest)
    root = Path(root).expanduser().resolve()
    expected_names = tuple(record.logical_path for record in manifest.files)
    if tuple(sorted(expected_names)) != expected_names or len(set(expected_names)) != len(expected_names):
        raise FreezeError("manifest logical paths are not sorted and unique")
    if manifest.prefix_bytes_total != sum(record.prefix_bytes for record in manifest.files):
        raise FreezeError("manifest prefix_bytes_total is inconsistent")
    if manifest.source_set_sha256 != _source_set_sha256(expected_names):
        raise FreezeError("manifest source_set_sha256 is inconsistent")
    if discover is not None:
        live_names = tuple(logical for logical, _ in _enumerate_sources(root, discover))
        if live_names != expected_names:
            raise FreezeError(
                f"live source set differs from manifest: expected={expected_names!r}, live={live_names!r}"
            )
    for record in manifest.files:
        path = (root / record.logical_path).resolve()
        _logical_path(root, path)
        if not path.is_file():
            raise FreezeError(f"frozen source is absent: {record.logical_path}")
        stat = path.stat()
        if stat.st_dev != record.device_at_capture or stat.st_ino != record.inode_at_capture:
            raise FreezeError(f"frozen source was replaced: {record.logical_path}")
        if stat.st_size < record.prefix_bytes:
            raise FreezeError(f"frozen source shrank: {record.logical_path}")
        digest, parsed, malformed, partial = _inspect_prefix(path, record.prefix_bytes)
        observed = (digest, parsed, malformed, partial)
        expected = (
            record.prefix_sha256,
            record.parsed_complete_lines,
            record.malformed_complete_lines,
            record.ignored_partial_trailing_lines,
        )
        if observed != expected:
            raise FreezeError(
                f"frozen source prefix or parsing metadata changed: {record.logical_path}"
            )


def iter_frozen_jsonl(
    manifest: FreezeManifest,
    root: Path,
    *,
    verify: bool = True,
    discover: DiscoverFiles | None = None,
) -> Iterator[FrozenJsonRecord]:
    """Yield complete JSON objects from exactly the manifested byte prefixes."""

    root = Path(root).expanduser().resolve()
    if verify:
        verify_freeze_manifest(manifest, root, discover=discover)
    for record in manifest.files:
        path = (root / record.logical_path).resolve()
        remaining = record.prefix_bytes
        line_number = 0
        with path.open("rb") as handle:
            while remaining:
                raw = handle.readline(remaining)
                if not raw:
                    raise FreezeError(f"short read from frozen source: {record.logical_path}")
                remaining -= len(raw)
                line_number += 1
                if remaining == 0 and not raw.endswith(b"\n"):
                    continue
                value = json.loads(raw)
                if not isinstance(value, Mapping):
                    raise FreezeError(
                        f"JSONL record is not an object: {record.logical_path}:{line_number}"
                    )
                yield FrozenJsonRecord(record.logical_path, line_number, value)
