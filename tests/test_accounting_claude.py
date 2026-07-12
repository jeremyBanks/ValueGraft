from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from accounting.claude import (
    ClaudeAmbiguityError,
    ClaudeRateQuote,
    ClaudeRequest,
    ClaudeUsageError,
    MappingClaudeRateTable,
    ReceiptUsage,
    build_claude_requests,
    price_claude_request,
    reconcile_receipt_overlap,
    select_message_versions,
)
from accounting.core import FrozenJsonRecord


def _record(path: str, line: int, value: dict) -> FrozenJsonRecord:
    return FrozenJsonRecord(path, line, value)


def _assistant(message_id: str, usage: dict, *, model: str = "claude-fable-5") -> dict:
    return {
        "type": "assistant",
        "timestamp": "2026-07-12T00:00:00Z",
        "message": {"id": message_id, "model": model, "usage": usage},
    }


def test_global_stream_selection_uses_complete_copy_and_ignores_parent_tool_result() -> None:
    partial = _assistant(
        "msg-1",
        {
            "input_tokens": 2,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 10,
            "output_tokens": 1,
            "cache_creation": {},
            "service_tier": "standard",
        },
    )
    complete = _assistant(
        "msg-1",
        {
            "input_tokens": 2,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 10,
            "output_tokens": 3,
            "cache_creation": {},
            "service_tier": "standard",
            "iterations": [
                {
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 10,
                    "output_tokens": 3,
                    "cache_creation": {},
                }
            ],
        },
    )
    parent_tool_result = {
        "type": "user",
        "toolUseResult": {"usage": complete["message"]["usage"]},
    }
    records = [
        _record("session-a.jsonl", 1, partial),
        _record("session-b.jsonl", 2, complete),
        _record("session-a.jsonl", 3, parent_tool_result),
    ]
    selected = select_message_versions(records)
    assert len(selected) == 1
    assert selected[0].occurrence_count == 2
    assert selected[0].session_ids == ("session-a", "session-b")
    requests = build_claude_requests(records)
    assert len(requests) == 1
    assert requests[0].output_tokens == 3
    assert requests[0].total_tokens == 15


def test_equal_best_streamed_versions_with_different_usage_fail() -> None:
    left = _assistant(
        "msg-tie",
        {
            "input_tokens": 10,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 1,
            "cache_creation": {},
        },
    )
    right = _assistant(
        "msg-tie",
        {
            "input_tokens": 5,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 5,
            "output_tokens": 1,
            "cache_creation": {},
        },
    )
    with pytest.raises(ClaudeAmbiguityError, match="equal-best"):
        select_message_versions(
            [_record("a.jsonl", 1, left), _record("b.jsonl", 1, right)]
        )


def test_iterations_become_requests_with_override_ttls_tier_speed_and_context() -> None:
    usage = {
        "input_tokens": 9,
        "cache_creation_input_tokens": 12,
        "cache_read_input_tokens": 200_100,
        "output_tokens": 7,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 5,
            "ephemeral_1h_input_tokens": 7,
        },
        "service_tier": "standard",
        "speed": "fast",
        "iterations": [
            {
                "model": "claude-opus-4-8",
                "input_tokens": 2,
                "cache_creation_input_tokens": 12,
                "cache_read_input_tokens": 200_001,
                "output_tokens": 3,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 5,
                    "ephemeral_1h_input_tokens": 7,
                },
            },
            {
                "input_tokens": 7,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 99,
                "output_tokens": 4,
                "cache_creation": {},
            },
        ],
    }
    requests = build_claude_requests(
        [_record("receipt-session.jsonl", 1, _assistant("msg-iterations", usage))]
    )
    assert len(requests) == 2
    first, second = requests
    assert first.model == "claude-opus-4-8"
    assert first.cache_write_5m_tokens == 5
    assert first.cache_write_1h_tokens == 7
    assert first.cache_write_unclassified_tokens == 0
    assert first.context_class == "long_context"
    assert first.service_tier == "standard"
    assert first.speed == "fast"
    assert second.model == "claude-fable-5"
    assert second.context_class == "standard_context"


def test_empty_iterations_fall_back_to_top_level_and_preserve_unclassified_cache() -> None:
    usage = {
        "input_tokens": 2,
        "cache_creation_input_tokens": 7,
        "cache_read_input_tokens": 11,
        "output_tokens": 6,
        "cache_creation": {"ephemeral_1h_input_tokens": 3},
        "iterations": [],
    }
    request = build_claude_requests(
        [_record("session.jsonl", 1, _assistant("msg-fallback", usage))]
    )[0]
    assert request.cache_write_1h_tokens == 3
    assert request.cache_write_unclassified_tokens == 4
    assert request.total_tokens == 26

    invalid = dict(usage)
    invalid["input_tokens"] = 2.5
    with pytest.raises(ClaudeUsageError, match="not an integer"):
        build_claude_requests(
            [_record("session.jsonl", 2, _assistant("msg-invalid-count", invalid))]
        )


def _million_token_request() -> ClaudeRequest:
    return ClaudeRequest(
        request_id="msg-price:0",
        message_id="msg-price",
        request_index=0,
        model="claude-fable-5",
        service_tier="standard",
        speed="standard",
        context_class="long_context",
        uncached_input_tokens=1_000_000,
        cache_write_5m_tokens=1_000_000,
        cache_write_1h_tokens=1_000_000,
        cache_write_unclassified_tokens=0,
        cache_read_input_tokens=1_000_000,
        output_tokens=1_000_000,
        context_input_tokens=4_000_000,
        message_timestamp=None,
        selected_source_ref="session.jsonl:1",
        session_ids=("session",),
    )


def test_request_level_rate_lookup_is_exact_and_unknown_is_unpriced() -> None:
    quote = ClaudeRateQuote(
        model="claude-fable-5",
        service_tier="standard",
        context_class="long_context",
        uncached_input_usd_per_million=Decimal("1"),
        cache_write_5m_usd_per_million=Decimal("2"),
        cache_write_1h_usd_per_million=Decimal("3"),
        cache_read_usd_per_million=Decimal("4"),
        output_usd_per_million=Decimal("5"),
        source_ref="rates:test",
    )
    table = MappingClaudeRateTable([quote])
    priced = price_claude_request(_million_token_request(), table)
    assert priced.priced is True
    assert priced.amount_usd == Decimal("15")
    assert priced.rate_source_ref == "rates:test"

    missing = price_claude_request(
        replace(_million_token_request(), service_tier="priority"), table
    )
    assert missing.priced is False
    assert missing.amount_usd is None
    assert missing.status == "unpriced_rate_not_found"

    unknown_ttl = price_claude_request(
        replace(
            _million_token_request(),
            cache_write_5m_tokens=999_999,
            cache_write_unclassified_tokens=1,
        ),
        table,
    )
    assert unknown_ttl.status == "unpriced_cache_write_ttl"


def test_receipt_overlap_hook_never_auto_adds_and_flags_absent_model() -> None:
    request = replace(
        _million_token_request(),
        model="claude-fable-5",
        session_ids=("receipt-session",),
    )
    exact = ReceiptUsage(
        receipt_id="receipt-1",
        session_id="receipt-session",
        model="claude-fable-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=2_000_000,
        cache_read_input_tokens=1_000_000,
        source_ref="receipt.json",
    )
    helper = ReceiptUsage(
        receipt_id="receipt-1",
        session_id="receipt-session",
        model="claude-haiku-helper",
        input_tokens=50,
        output_tokens=5,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        source_ref="receipt.json",
    )
    rows = reconcile_receipt_overlap([request], [exact, helper])
    assert rows[0].status == "exact_overlap"
    assert rows[0].candidate_supplement is False
    assert rows[1].status == "candidate_unlogged_model_usage"
    assert rows[1].candidate_supplement is True
    assert rows[1].nonnegative_difference == rows[1].receipt_counts
