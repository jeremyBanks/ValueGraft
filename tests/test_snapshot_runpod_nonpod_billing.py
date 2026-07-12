from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.accounting.snapshot_runpod_nonpod_billing import (
    RunPodNonPodSnapshotError,
    collect_snapshot,
    write_snapshot,
)


def test_collects_exact_allowlisted_rows_without_credential() -> None:
    def fetch(url: str, key: str):
        assert key == "secret-key"
        if "/endpoints?" in url:
            body = json.dumps(
                [
                    {
                        "amount": 0.01,
                        "diskSpaceBilledGb": 2,
                        "endpointId": "endpoint-opaque",
                        "gpuTypeId": None,
                        "podId": None,
                        "time": "2026-07-01T00:00:00Z",
                        "timeBilledMs": 3,
                    }
                ]
            ).encode()
        else:
            body = b"[]"
        return 200, "application/json", body, {}

    snapshot = collect_snapshot(
        "secret-key",
        start_time="2026-07-01T00:00:00Z",
        end_time="2026-07-12T00:00:00Z",
        fetch=fetch,
    )
    assert snapshot["combined_amount_usd"] == "0.01"
    assert snapshot["resources"]["serverless_endpoints"]["row_count"] == 1
    assert snapshot["resources"]["network_volumes"]["amount_usd"] == "0"
    assert "secret-key" not in json.dumps(snapshot)


def test_unknown_provider_field_fails_closed() -> None:
    def fetch(_: str, __: str):
        return 200, "application/json", b'[{"amount":0,"newSecretField":"x"}]', {}

    with pytest.raises(RunPodNonPodSnapshotError, match="unexpected fields"):
        collect_snapshot(
            "key",
            start_time="2026-07-01T00:00:00Z",
            end_time="2026-07-12T00:00:00Z",
            fetch=fetch,
        )


def test_write_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    write_snapshot(path, {"schema": "fixture"})
    with pytest.raises(RunPodNonPodSnapshotError, match="refusing to overwrite"):
        write_snapshot(path, {"schema": "replacement"})
