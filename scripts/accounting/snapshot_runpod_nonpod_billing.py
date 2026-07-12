#!/usr/bin/env python3
"""Capture sanitized RunPod Serverless and network-volume billing rows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping
import urllib.parse
import urllib.request


BASE = "https://rest.runpod.io/v1/billing"
CONTRACTS = {
    "serverless_endpoints": "https://docs.runpod.io/api-reference/billing/GET/billing/endpoints",
    "network_volumes": "https://docs.runpod.io/api-reference/billing/GET/billing/networkvolumes",
}
ALLOWLISTS = {
    "serverless_endpoints": {
        "amount",
        "diskSpaceBilledGb",
        "endpointId",
        "gpuTypeId",
        "podId",
        "time",
        "timeBilledMs",
    },
    "network_volumes": {
        "amount",
        "diskSpaceBilledGb",
        "highPerformanceStorageAmount",
        "highPerformanceStorageDiskSpaceBilledGb",
        "time",
    },
}


class RunPodNonPodSnapshotError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fetch(url: str, api_key: str) -> tuple[int, str, bytes, Mapping[str, str]]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return (
            int(response.status),
            str(response.headers.get("Content-Type", "")),
            response.read(),
            dict(response.headers.items()),
        )


def _exact_decimal(value: Any, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise RunPodNonPodSnapshotError(f"{field} is not exact numeric data")
    number = Decimal(value)
    if not number.is_finite() or number < 0:
        raise RunPodNonPodSnapshotError(f"{field} is not finite nonnegative data")
    return format(number, "f")


def _collect_one(
    kind: str,
    url: str,
    api_key: str,
    fetch: Callable[[str, str], tuple[int, str, bytes, Mapping[str, str]]],
) -> dict[str, Any]:
    status, content_type, body, headers = fetch(url, api_key)
    if status != 200:
        raise RunPodNonPodSnapshotError(f"{kind} returned HTTP {status}")
    try:
        raw = json.loads(body, parse_float=Decimal, parse_int=Decimal)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RunPodNonPodSnapshotError(f"{kind} returned malformed JSON") from exc
    if not isinstance(raw, list) or any(not isinstance(row, Mapping) for row in raw):
        raise RunPodNonPodSnapshotError(f"{kind} response is not an array of objects")
    rows: list[dict[str, Any]] = []
    total = Decimal(0)
    for index, source in enumerate(raw):
        unexpected = set(source) - ALLOWLISTS[kind]
        if unexpected:
            raise RunPodNonPodSnapshotError(
                f"{kind}[{index}] has unexpected fields: {sorted(unexpected)!r}"
            )
        row: dict[str, Any] = {}
        for field in sorted(source):
            value = source[field]
            if field in {
                "amount",
                "highPerformanceStorageAmount",
                "diskSpaceBilledGb",
                "highPerformanceStorageDiskSpaceBilledGb",
                "timeBilledMs",
            }:
                row[field] = _exact_decimal(value, f"{kind}[{index}].{field}")
            elif value is not None and not isinstance(value, str):
                raise RunPodNonPodSnapshotError(
                    f"{kind}[{index}].{field} is not text or null"
                )
            else:
                row[field] = value
        total += Decimal(row.get("amount", "0"))
        total += Decimal(row.get("highPerformanceStorageAmount", "0"))
        rows.append(row)
    pagination_headers = {
        key: value
        for key, value in headers.items()
        if key.lower() in {"link", "x-next-cursor", "x-total-count"}
    }
    return {
        "row_count": len(rows),
        "amount_usd": format(total, "f"),
        "rows": rows,
        "response_sha256": hashlib.sha256(body).hexdigest(),
        "http_status": status,
        "content_type": content_type,
        "pagination_headers": pagination_headers,
    }


def collect_snapshot(
    api_key: str,
    *,
    start_time: str,
    end_time: str,
    fetch: Callable[
        [str, str], tuple[int, str, bytes, Mapping[str, str]]
    ] = _fetch,
) -> dict[str, Any]:
    if not api_key:
        raise RunPodNonPodSnapshotError("API key is empty")
    common = {"bucketSize": "year", "startTime": start_time, "endTime": end_time}
    endpoint_params = {**common, "grouping": "endpointId"}
    endpoint_url = f"{BASE}/endpoints?{urllib.parse.urlencode(endpoint_params)}"
    volume_url = f"{BASE}/networkvolumes?{urllib.parse.urlencode(common)}"
    captured = _now()
    resources = {
        "serverless_endpoints": _collect_one(
            "serverless_endpoints", endpoint_url, api_key, fetch
        ),
        "network_volumes": _collect_one(
            "network_volumes", volume_url, api_key, fetch
        ),
    }
    return {
        "schema": "runpod_nonpod_billing_snapshot_v1",
        "captured_at_utc": captured,
        "window": {"start_time_utc": start_time, "end_time_utc": end_time},
        "request": {
            "authorization_serialized": False,
            "bucket_size": "year",
            "official_contracts": CONTRACTS,
        },
        "resources": resources,
        "combined_amount_usd": format(
            sum((Decimal(row["amount_usd"]) for row in resources.values()), Decimal(0)),
            "f",
        ),
        "scope": (
            "Exact provider response for non-Pod RunPod billing categories in the "
            "requested window; not evidence of cash deposits or project attribution."
        ),
    }


def write_snapshot(path: Path, snapshot: Mapping[str, Any]) -> None:
    if path.exists():
        raise RunPodNonPodSnapshotError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-file", type=Path, default=Path(".runpod_key"))
    parser.add_argument("--start-time", default="2026-07-01T00:00:00Z")
    parser.add_argument("--end-time", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = collect_snapshot(
        args.key_file.read_text(encoding="utf-8").strip(),
        start_time=args.start_time,
        end_time=args.end_time,
    )
    write_snapshot(args.output, snapshot)
    print(
        "RUNPOD NONPOD BILLING SNAPSHOT "
        f"amount={snapshot['combined_amount_usd']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
