from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from accounting.core import (
    AccountingRole,
    AccountingSchemaError,
    FreezeError,
    FreezeManifest,
    LedgerName,
    LedgerRow,
    MeasurementClass,
    canonical_json,
    canonical_sha256,
    freeze_file_set,
    iter_frozen_jsonl,
    verify_freeze_manifest,
)


def _jsonl_files(root: Path):
    return root.glob("*.jsonl")


def test_canonical_json_is_sorted_and_decimal_exact() -> None:
    value = {"z": Decimal("1.2300"), "a": [2, Decimal("0.000000001")]}
    assert canonical_json(value) == '{"a":[2,"0.000000001"],"z":"1.2300"}'
    assert canonical_sha256(value) == canonical_sha256(
        {"a": [2, Decimal("0.000000001")], "z": Decimal("1.2300")}
    )
    with pytest.raises(TypeError, match="float is forbidden"):
        canonical_json({"cost": 0.1})


def test_ledger_row_is_strict_and_decimal_safe() -> None:
    row = LedgerRow(
        row_id="runpod:pod:x",
        ledger=LedgerName.METERED_CONSUMPTION,
        provider="runpod",
        category="pod_credits_consumed",
        quantity=Decimal("1.2300"),
        unit="USD credits",
        currency="USD",
        measurement_class=MeasurementClass.EXACT_SOURCE_RECORD,
        accounting_role=AccountingRole.PRIMARY_ADDITIVE,
        overlap_key="runpod:pod:x",
        evidence_refs=("z.json", "a.json", "a.json"),
        method_version="accounting-v1",
    )
    assert row.evidence_refs == ("a.json", "z.json")
    assert row.to_dict()["quantity"] == "1.2300"
    with pytest.raises(AccountingSchemaError, match="quantity must be Decimal"):
        replace(row, quantity=1.23)  # type: ignore[arg-type]
    with pytest.raises(AccountingSchemaError, match="unknown rows must have quantity=None"):
        replace(row, measurement_class=MeasurementClass.UNKNOWN)
    with pytest.raises(AccountingSchemaError, match="must identify UTC"):
        replace(row, window_start_utc="2026-07-12T00:00:00-04:00")
    unknown = replace(
        row,
        ledger=LedgerName.UNKNOWNS,
        quantity=None,
        measurement_class=MeasurementClass.UNKNOWN,
        accounting_role=AccountingRole.CONTEXT_NONADDITIVE,
    )
    assert unknown.to_dict()["quantity"] is None


def test_freeze_roundtrip_records_partial_tail_and_allows_append(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text('{"n":1}\n')
    second.write_bytes(b'{"n":2}\n{"partial":')

    manifest = freeze_file_set(
        tmp_path, logical_root="claude", discover=_jsonl_files
    )
    by_name = {row.logical_path: row for row in manifest.files}
    assert by_name["a.jsonl"].parsed_complete_lines == 1
    assert by_name["b.jsonl"].parsed_complete_lines == 1
    assert by_name["b.jsonl"].ignored_partial_trailing_lines == 1
    assert manifest.prefix_bytes_total == first.stat().st_size + second.stat().st_size

    serialized = json.loads(canonical_json(manifest))
    restored = FreezeManifest.from_dict(serialized)
    assert restored == manifest

    first.write_text(first.read_text() + '{"n":3}\n')
    verify_freeze_manifest(restored, tmp_path, discover=_jsonl_files)
    rows = list(iter_frozen_jsonl(restored, tmp_path, discover=_jsonl_files))
    assert [(row.logical_path, row.value["n"]) for row in rows] == [
        ("a.jsonl", 1),
        ("b.jsonl", 2),
    ]


def test_freeze_rejects_source_set_change_during_capture(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text('{"n":1}\n')
    calls = 0

    def changing_discovery(root: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            (root / "b.jsonl").write_text('{"n":2}\n')
        return root.glob("*.jsonl")

    with pytest.raises(FreezeError, match="source set changed"):
        freeze_file_set(tmp_path, logical_root="claude", discover=changing_discovery)


def test_freeze_reverification_rejects_shrink_and_replacement(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"n":123456}\n')
    manifest = freeze_file_set(tmp_path, logical_root="claude", discover=_jsonl_files)

    path.write_text("{}\n")
    with pytest.raises(FreezeError, match="shrank"):
        verify_freeze_manifest(manifest, tmp_path, discover=_jsonl_files)

    path.write_text('{"n":123456}\n')
    manifest = freeze_file_set(tmp_path, logical_root="claude", discover=_jsonl_files)
    replacement = tmp_path / "replacement"
    replacement.write_text('{"n":123456}\n')
    replacement.replace(path)
    with pytest.raises(FreezeError, match="replaced"):
        verify_freeze_manifest(manifest, tmp_path, discover=_jsonl_files)


def test_freeze_rejects_complete_malformed_line_and_manifest_tamper(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text("not-json\n")
    with pytest.raises(FreezeError, match="malformed complete JSONL"):
        freeze_file_set(tmp_path, logical_root="claude", discover=_jsonl_files)

    path.write_text('{"ok":true}\n')
    manifest = freeze_file_set(tmp_path, logical_root="claude", discover=_jsonl_files)
    tampered = replace(manifest, prefix_bytes_total=manifest.prefix_bytes_total + 1)
    with pytest.raises(FreezeError, match="self-hash mismatch"):
        verify_freeze_manifest(tampered, tmp_path)
