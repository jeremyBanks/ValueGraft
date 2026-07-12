from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.accounting.snapshot_openrouter_key_usage import (
    OpenRouterSnapshotError,
    collect_snapshot,
    write_snapshot,
)


def _fetch_key(api_key: str) -> tuple[int, str, bytes]:
    assert api_key == "secret-test-key"
    return (
        200,
        "application/json",
        json.dumps(
            {
                "data": {
                    "label": "private label",
                    "creator_user_id": "private user",
                    "usage": 0.0232311,
                    "usage_daily": 0,
                    "usage_weekly": 0.0232311,
                    "usage_monthly": 0.0232311,
                    "byok_usage": 0,
                    "byok_usage_daily": 0,
                    "byok_usage_weekly": 0,
                    "byok_usage_monthly": 0,
                    "limit": 50,
                    "limit_remaining": 49.9767689,
                    "limit_reset": None,
                    "include_byok_in_limit": False,
                    "is_free_tier": False,
                }
            }
        ).encode(),
    )


def _fetch_credits(api_key: str) -> tuple[int, str, bytes]:
    assert api_key == "secret-test-key"
    return (
        200,
        "application/json",
        json.dumps(
            {"data": {"total_credits": 114, "total_usage": 95.541220588}}
        ).encode(),
    )


def test_snapshot_is_exact_allowlisted_and_credential_free() -> None:
    snapshot = collect_snapshot(
        "secret-test-key", fetch_key=_fetch_key, fetch_credits=_fetch_credits
    )
    rendered = json.dumps(snapshot)
    assert snapshot["usage_counters"]["usage"] == "0.0232311"
    assert snapshot["usage_counters"]["limit_remaining"] == "49.9767689"
    assert snapshot["classification"]["cash_paid_usd"] is None
    assert snapshot["account_context_nonadditive"]["total_credits"] == "114"
    assert snapshot["account_context_nonadditive"]["total_usage"] == "95.541220588"
    assert "secret-test-key" not in rendered
    assert "private label" not in rendered
    assert "private user" not in rendered


def test_snapshot_rejects_non_numeric_usage() -> None:
    def malformed(_: str) -> tuple[int, str, bytes]:
        payload = {"data": {"usage": "1.0"}}
        body = json.dumps(payload).encode()
        return 200, "application/json", body

    with pytest.raises(OpenRouterSnapshotError):
        collect_snapshot("secret", fetch_key=malformed, fetch_credits=malformed)


def test_write_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    write_snapshot(path, {"schema": "fixture"})
    with pytest.raises(OpenRouterSnapshotError, match="refusing to overwrite"):
        write_snapshot(path, {"schema": "replacement"})
