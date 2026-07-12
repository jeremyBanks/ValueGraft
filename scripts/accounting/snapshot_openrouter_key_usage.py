#!/usr/bin/env python3
"""Capture a credential-free snapshot of one OpenRouter key's usage counters."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping
import urllib.request


KEY_ENDPOINT = "https://openrouter.ai/api/v1/key"
CREDITS_ENDPOINT = "https://openrouter.ai/api/v1/credits"
DECIMAL_FIELDS = (
    "usage",
    "usage_daily",
    "usage_weekly",
    "usage_monthly",
    "byok_usage",
    "byok_usage_daily",
    "byok_usage_weekly",
    "byok_usage_monthly",
    "limit",
    "limit_remaining",
)
REQUIRED_DECIMAL_FIELDS = DECIMAL_FIELDS[:8]
BOOL_FIELDS = ("include_byok_in_limit", "is_free_tier")
OPTIONAL_TEXT_FIELDS = ("limit_reset",)


class OpenRouterSnapshotError(RuntimeError):
    """Raised when a sanitized key-usage snapshot cannot be produced."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _decimal(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise OpenRouterSnapshotError(f"{field} is not an exact numeric value")
    parsed = Decimal(value)
    if not parsed.is_finite() or parsed < 0:
        raise OpenRouterSnapshotError(f"{field} must be a finite nonnegative value")
    return format(parsed, "f")


def _fetch_endpoint(endpoint: str, api_key: str) -> tuple[int, str, bytes]:
    request = urllib.request.Request(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return int(response.status), str(response.headers.get("Content-Type", "")), response.read()


def _default_fetch_key(api_key: str) -> tuple[int, str, bytes]:
    return _fetch_endpoint(KEY_ENDPOINT, api_key)


def _default_fetch_credits(api_key: str) -> tuple[int, str, bytes]:
    return _fetch_endpoint(CREDITS_ENDPOINT, api_key)


def _parse_response(
    response: tuple[int, str, bytes], *, label: str
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    status, content_type, body = response
    if status != 200:
        raise OpenRouterSnapshotError(f"OpenRouter {label} endpoint returned HTTP {status}")
    try:
        payload = json.loads(body, parse_float=Decimal, parse_int=Decimal)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OpenRouterSnapshotError(f"OpenRouter {label} response is not valid JSON") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), Mapping):
        raise OpenRouterSnapshotError(f"OpenRouter {label} response lacks data object")
    commitment = {
        "http_status": status,
        "content_type": content_type,
        "raw_response_sha256": hashlib.sha256(body).hexdigest(),
    }
    return payload["data"], commitment


def collect_snapshot(
    api_key: str,
    *,
    fetch_key: Callable[[str], tuple[int, str, bytes]] = _default_fetch_key,
    fetch_credits: Callable[[str], tuple[int, str, bytes]] = _default_fetch_credits,
) -> dict[str, Any]:
    """Return only allowlisted counters; never return the key or raw response."""

    if not api_key or not isinstance(api_key, str):
        raise OpenRouterSnapshotError("API key is empty")
    data, key_commitment = _parse_response(fetch_key(api_key), label="key")
    credits, credits_commitment = _parse_response(
        fetch_credits(api_key), label="credits"
    )
    counters = {field: _decimal(data.get(field), field) for field in DECIMAL_FIELDS}
    missing = [field for field in REQUIRED_DECIMAL_FIELDS if counters[field] is None]
    if missing:
        raise OpenRouterSnapshotError(
            f"OpenRouter response lacks required usage counters: {missing!r}"
        )
    for field in BOOL_FIELDS:
        if not isinstance(data.get(field), bool):
            raise OpenRouterSnapshotError(f"{field} is not boolean")
    for field in OPTIONAL_TEXT_FIELDS:
        if data.get(field) is not None and not isinstance(data.get(field), str):
            raise OpenRouterSnapshotError(f"{field} is not text or null")
    account_credits = {
        "total_credits": _decimal(credits.get("total_credits"), "total_credits"),
        "total_usage": _decimal(credits.get("total_usage"), "total_usage"),
    }
    if any(value is None for value in account_credits.values()):
        raise OpenRouterSnapshotError("OpenRouter credits response lacks required totals")
    return {
        "schema": "openrouter_key_usage_snapshot_v2",
        "captured_at_utc": _utc_now(),
        "request": {
            "key_endpoint": KEY_ENDPOINT,
            "credits_endpoint": CREDITS_ENDPOINT,
            "authorization_serialized": False,
            "official_contracts": [
                "https://openrouter.ai/docs/api/api-reference/api-keys/get-current-key",
                "https://openrouter.ai/docs/api/api-reference/credits/get-credits",
            ],
        },
        "response_commitment": {
            "key": key_commitment,
            "credits": credits_commitment,
        },
        "usage_counters": counters,
        "key_context": {
            field: data.get(field) for field in (*BOOL_FIELDS, *OPTIONAL_TEXT_FIELDS)
        },
        "account_context_nonadditive": {
            **account_credits,
            "scope": "authenticated OpenRouter account lifetime; not project attribution",
            "cash_paid_claimed": False,
        },
        "classification": {
            "measurement": "exact_source_record_for_current_key",
            "unit": "OpenRouter credits used",
            "cash_paid_usd": None,
            "project_attribution": (
                "not implied by the endpoint; the response is key-lifetime and period-to-date, "
                "without generation timestamps"
            ),
        },
    }


def write_snapshot(path: Path, snapshot: Mapping[str, Any]) -> None:
    if path.exists():
        raise OpenRouterSnapshotError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-file", type=Path, default=Path(".openrouter_key"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key = args.key_file.read_text(encoding="utf-8").strip()
    snapshot = collect_snapshot(key)
    write_snapshot(args.output, snapshot)
    print(
        "OPENROUTER KEY USAGE SNAPSHOT "
        f"usage={snapshot['usage_counters']['usage']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
