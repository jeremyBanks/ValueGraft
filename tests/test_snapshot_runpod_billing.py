from __future__ import annotations

from decimal import Decimal

import pytest

from scripts.accounting.snapshot_runpod_billing import (
    CollectionError,
    aggregate_billing_rows,
)


def _row(pod: str, day: str, amount: str, billed_ms: int) -> dict[str, object]:
    return {
        "amount": Decimal(amount),
        "diskSpaceBilledGB": 20,
        "podId": pod,
        "time": day,
        "timeBilledMs": billed_ms,
    }


def test_aggregate_billing_rows_is_exact_and_per_pod() -> None:
    result = aggregate_billing_rows(
        [
            _row("pod-b", "2026-07-06T00:00:00Z", "0.2", 200),
            _row("pod-a", "2026-07-05T00:00:00Z", "0.1", 100),
            _row("pod-a", "2026-07-06T00:00:00Z", "0.3", 300),
        ]
    )
    assert result == [
        {
            "pod_id": "pod-a",
            "amount_usd": "0.4",
            "billed_time_ms": 400,
            "gpu_type_id": None,
        },
        {
            "pod_id": "pod-b",
            "amount_usd": "0.2",
            "billed_time_ms": 200,
            "gpu_type_id": None,
        },
    ]


def test_aggregate_rejects_duplicate_bucket_or_unexpected_field() -> None:
    row = _row("pod-a", "2026-07-05T00:00:00Z", "0.1", 100)
    with pytest.raises(CollectionError, match="duplicate billing bucket"):
        aggregate_billing_rows([row, row])
    with pytest.raises(CollectionError, match="unexpected fields"):
        aggregate_billing_rows([{**row, "consumerUserId": "must-not-survive"}])


def test_aggregate_rejects_binary_float_and_bad_ids() -> None:
    floated = _row("pod-a", "2026-07-05T00:00:00Z", "0.1", 100)
    floated["amount"] = 0.1
    with pytest.raises(CollectionError, match="exact Decimal"):
        aggregate_billing_rows([floated])
    missing_id = _row("", "2026-07-05T00:00:00Z", "0.1", 100)
    with pytest.raises(CollectionError, match="string podId"):
        aggregate_billing_rows([missing_id])
