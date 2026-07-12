#!/usr/bin/env python3
"""Write a credential-free snapshot of RunPod Pod billing and active inventory.

The collector reads the API key only to authorize two read-only REST requests.
It retains an allowlisted billing projection, active Pod IDs, response hashes,
and request parameters; raw Pod objects and the key are never serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence


REST_ROOT = "https://rest.runpod.io/v1"
BILLING_FIELDS = {
    "amount",
    "diskSpaceBilledGB",
    "podId",
    "time",
    "timeBilledMs",
}


class CollectionError(RuntimeError):
    """Raised when provider evidence does not match the safe frozen contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_decimal(label: str, value: Any) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise CollectionError(f"{label} must be a nonnegative exact Decimal")
    return value


def _require_nonnegative_int(label: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CollectionError(f"{label} must be a nonnegative integer")
    return value


def aggregate_billing_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate non-overlapping provider buckets into one exact row per Pod."""

    totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"amount": Decimal("0"), "time_billed_ms": 0}
    )
    seen_buckets: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        unexpected = sorted(set(row) - BILLING_FIELDS)
        if unexpected:
            raise CollectionError(
                f"billing row {index} contains unexpected fields: {unexpected!r}"
            )
        pod_id = row.get("podId")
        bucket = row.get("time")
        if not isinstance(pod_id, str) or not pod_id:
            raise CollectionError(f"billing row {index} lacks a string podId")
        if not isinstance(bucket, str) or not bucket:
            raise CollectionError(f"billing row {index} lacks a bucket timestamp")
        identity = (pod_id, bucket)
        if identity in seen_buckets:
            raise CollectionError(f"duplicate billing bucket: {identity!r}")
        seen_buckets.add(identity)
        amount = _require_decimal(f"billing row {index}.amount", row.get("amount"))
        billed_ms = _require_nonnegative_int(
            f"billing row {index}.timeBilledMs", row.get("timeBilledMs")
        )
        totals[pod_id]["amount"] += amount
        totals[pod_id]["time_billed_ms"] += billed_ms
    return [
        {
            "pod_id": pod_id,
            "amount_usd": format(values["amount"], "f"),
            "billed_time_ms": values["time_billed_ms"],
            "gpu_type_id": None,
        }
        for pod_id, values in sorted(totals.items())
    ]


def _request_json(
    url: str, *, authorization: str
) -> tuple[Any, bytes, Mapping[str, str], int]:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {authorization}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
        status = response.status
        headers = {key.lower(): value for key, value in response.headers.items()}
    try:
        value = json.loads(raw, parse_float=Decimal, parse_int=int)
    except Exception as exc:
        raise CollectionError(f"provider returned invalid JSON from {url}") from exc
    return value, raw, headers, status


def _active_inventory(
    *, authorization: str
) -> tuple[tuple[str, ...], str, int, Mapping[str, str]]:
    value, raw, headers, status = _request_json(
        f"{REST_ROOT}/pods", authorization=authorization
    )
    if status != 200 or not isinstance(value, list):
        raise CollectionError("active Pod inventory was not a complete JSON list")
    ids: list[str] = []
    for index, pod in enumerate(value):
        if not isinstance(pod, Mapping):
            raise CollectionError(f"active Pod row {index} is not an object")
        pod_id = pod.get("id")
        if not isinstance(pod_id, str) or not pod_id:
            raise CollectionError(f"active Pod row {index} lacks an id")
        ids.append(pod_id)
    if len(ids) != len(set(ids)):
        raise CollectionError("active Pod inventory contains duplicate IDs")
    return tuple(sorted(ids)), _sha256(raw), status, headers


def collect_snapshot(
    *, authorization: str, start_time_utc: str, end_time_utc: str
) -> dict[str, Any]:
    before, before_sha, before_status, before_headers = _active_inventory(
        authorization=authorization
    )
    parameters = {
        "bucketSize": "day",
        "grouping": "podId",
        "startTime": start_time_utc,
        "endTime": end_time_utc,
    }
    billing_url = f"{REST_ROOT}/billing/pods?{urllib.parse.urlencode(parameters)}"
    value, raw, billing_headers, billing_status = _request_json(
        billing_url, authorization=authorization
    )
    if billing_status != 200 or not isinstance(value, list):
        raise CollectionError("Pod billing response was not a complete JSON list")
    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise CollectionError(f"billing row {index} is not an object")
        rows.append(row)
    pod_totals = aggregate_billing_rows(rows)
    after, after_sha, after_status, after_headers = _active_inventory(
        authorization=authorization
    )
    if before != after:
        raise CollectionError("active Pod inventory changed during billing capture")
    account_total = sum(
        (Decimal(row["amount_usd"]) for row in pod_totals), Decimal("0")
    )
    account_time = sum(row["billed_time_ms"] for row in pod_totals)
    captured_at = _utc_now()

    pagination_headers = {
        key: value
        for key, value in billing_headers.items()
        if any(token in key for token in ("page", "cursor", "link", "total", "count"))
    }
    if pagination_headers:
        raise CollectionError(
            f"unexpected pagination metadata requires review: {sorted(pagination_headers)!r}"
        )
    snapshot = {
        "schema": "runpod_sanitized_provider_snapshot_v1",
        "captured_at_utc": captured_at,
        "pagination_complete": True,
        "active_inventory_complete": True,
        "active_pod_ids": list(after),
        "pod_totals": pod_totals,
        "account_total_usd": format(account_total, "f"),
        "account_billed_time_ms": account_time,
    }
    return {
        "schema": "runpod_billing_collection_receipt_v1",
        "captured_at_utc": captured_at,
        "request": {
            "billing_endpoint": f"{REST_ROOT}/billing/pods",
            "active_inventory_endpoint": f"{REST_ROOT}/pods",
            "parameters": parameters,
            "authorization_serialized": False,
        },
        "response_commitments": {
            "billing_http_status": billing_status,
            "billing_response_sha256": _sha256(raw),
            "billing_bucket_row_count": len(rows),
            "billing_content_type": billing_headers.get("content-type"),
            "billing_pagination_headers": pagination_headers,
            "active_before_http_status": before_status,
            "active_before_response_sha256": before_sha,
            "active_before_content_type": before_headers.get("content-type"),
            "active_after_http_status": after_status,
            "active_after_response_sha256": after_sha,
            "active_after_content_type": after_headers.get("content-type"),
            "active_inventory_stable": True,
        },
        "completeness_basis": (
            "The official GET /billing/pods contract returns a JSON array and exposes "
            "no pagination parameter; this response had no pagination headers. The "
            "official GET /pods contract returned the same complete active inventory "
            "immediately before and after billing capture."
        ),
        "snapshot": snapshot,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-file", type=Path, default=Path(".runpod_key"))
    parser.add_argument("--start-time", default="2026-07-01T00:00:00Z")
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise CollectionError(f"refusing to overwrite existing output: {args.output}")
    key = args.key_file.read_text(encoding="utf-8").strip()
    if not key or "\n" in key or "\r" in key:
        raise CollectionError("key file must contain exactly one nonempty token")
    end_time = args.end_time or _utc_now()
    receipt = collect_snapshot(
        authorization=key,
        start_time_utc=args.start_time,
        end_time_utc=end_time,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    snapshot = receipt["snapshot"]
    print(
        "RUNPOD BILLING SNAPSHOT "
        f"pods={len(snapshot['pod_totals'])} active={len(snapshot['active_pod_ids'])} "
        f"credits={snapshot['account_total_usd']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
