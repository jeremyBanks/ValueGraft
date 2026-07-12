"""Claude transcript reconstruction and request-level pricing interfaces.

The Claude project archive repeats assistant messages in parent and subagent
transcripts and may contain partial streamed versions of the same API message.
The only additive base here is the globally unique Claude ``message.id``.
Parent ``toolUseResult.usage`` objects are intentionally never traversed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Protocol

from .core import AccountingError, FrozenJsonRecord, canonical_sha256


class ClaudeUsageError(AccountingError):
    """Raised when Claude usage fields violate the accounting schema."""


class ClaudeAmbiguityError(ClaudeUsageError):
    """Raised when equally complete streamed copies disagree on usage."""


_USAGE_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


def _nonnegative_int(value: Any, label: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ClaudeUsageError(f"{label} is not an integer: {value!r}")
    result = value
    if result < 0:
        raise ClaudeUsageError(f"{label} is negative: {result}")
    return result


def _effective_usage_entries(usage: Mapping[str, Any]) -> tuple[tuple[Mapping[str, Any], ...], bool]:
    iterations = usage.get("iterations")
    if isinstance(iterations, list):
        rows = tuple(row for row in iterations if isinstance(row, Mapping))
        if rows:
            return rows, True
    return (usage,), False


def _entry_counts(entry: Mapping[str, Any], *, prefix: str) -> tuple[int, int, int, int]:
    return tuple(
        _nonnegative_int(entry.get(field), f"{prefix}.{field}") for field in _USAGE_FIELDS
    )  # type: ignore[return-value]


def _completeness(usage: Mapping[str, Any]) -> tuple[int, int, int, int]:
    entries, used_iterations = _effective_usage_entries(usage)
    counts = [_entry_counts(entry, prefix=f"usage[{index}]") for index, entry in enumerate(entries)]
    output = sum(row[3] for row in counts)
    total = sum(sum(row) for row in counts)
    return output, total, len(entries), int(used_iterations)


def _session_id(logical_path: str) -> str:
    first = PurePosixPath(logical_path).parts[0]
    return first[:-6] if first.endswith(".jsonl") else first


@dataclass(frozen=True)
class _Candidate:
    message_id: str
    message: Mapping[str, Any]
    timestamp: str | None
    source_ref: str
    logical_path: str
    completeness: tuple[int, int, int, int]
    usage_fingerprint: str


@dataclass(frozen=True)
class SelectedClaudeMessage:
    message_id: str
    message: Mapping[str, Any]
    timestamp: str | None
    selected_source_ref: str
    occurrence_count: int
    occurrence_source_refs: tuple[str, ...]
    session_ids: tuple[str, ...]
    completeness: tuple[int, int, int, int]
    usage_fingerprint: str


def select_message_versions(
    records: Iterable[FrozenJsonRecord],
) -> tuple[SelectedClaudeMessage, ...]:
    """Globally deduplicate Claude assistant usage by ``message.id``.

    The maximal lexicographic completeness vector is selected.  Multiple
    occurrences with the same maximal usage fingerprint are harmless copies;
    distinct fingerprints at that same maximum are an accounting ambiguity.
    """

    grouped: dict[str, list[_Candidate]] = defaultdict(list)
    for source in records:
        record = source.value
        if record.get("type") != "assistant":
            continue
        message = record.get("message")
        if not isinstance(message, Mapping):
            continue
        usage = message.get("usage")
        if not isinstance(usage, Mapping):
            continue
        message_id = message.get("id")
        if not isinstance(message_id, str) or not message_id:
            raise ClaudeUsageError(
                f"assistant usage record lacks message.id: {source.logical_path}:{source.line_number}"
            )
        source_ref = f"{source.logical_path}:{source.line_number}"
        completeness = _completeness(usage)
        fingerprint = canonical_sha256({"model": message.get("model"), "usage": usage})
        timestamp = record.get("timestamp")
        grouped[message_id].append(
            _Candidate(
                message_id=message_id,
                message=message,
                timestamp=timestamp if isinstance(timestamp, str) else None,
                source_ref=source_ref,
                logical_path=source.logical_path,
                completeness=completeness,
                usage_fingerprint=fingerprint,
            )
        )

    selected: list[SelectedClaudeMessage] = []
    for message_id, candidates in grouped.items():
        best_score = max(candidate.completeness for candidate in candidates)
        best = [candidate for candidate in candidates if candidate.completeness == best_score]
        best_fingerprints = {candidate.usage_fingerprint for candidate in best}
        if len(best_fingerprints) != 1:
            refs = sorted(candidate.source_ref for candidate in best)
            raise ClaudeAmbiguityError(
                f"equal-best streamed usage differs for {message_id}: {refs!r}"
            )
        chosen = min(best, key=lambda candidate: candidate.source_ref)
        selected.append(
            SelectedClaudeMessage(
                message_id=message_id,
                message=chosen.message,
                timestamp=chosen.timestamp,
                selected_source_ref=chosen.source_ref,
                occurrence_count=len(candidates),
                occurrence_source_refs=tuple(sorted({candidate.source_ref for candidate in candidates})),
                session_ids=tuple(sorted({_session_id(candidate.logical_path) for candidate in candidates})),
                completeness=chosen.completeness,
                usage_fingerprint=chosen.usage_fingerprint,
            )
        )
    return tuple(sorted(selected, key=lambda item: item.message_id))


@dataclass(frozen=True)
class ClaudeRequest:
    request_id: str
    message_id: str
    request_index: int
    model: str
    service_tier: str
    speed: str
    context_class: str
    uncached_input_tokens: int
    cache_write_5m_tokens: int
    cache_write_1h_tokens: int
    cache_write_unclassified_tokens: int
    cache_read_input_tokens: int
    output_tokens: int
    context_input_tokens: int
    message_timestamp: str | None
    selected_source_ref: str
    session_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "request_id",
            "message_id",
            "model",
            "service_tier",
            "speed",
            "selected_source_ref",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ClaudeUsageError(f"request {field} is empty")
        if self.request_index < 0:
            raise ClaudeUsageError("request_index is negative")
        if self.context_class not in {"standard_context", "long_context"}:
            raise ClaudeUsageError(f"unknown context_class: {self.context_class!r}")
        for field in (
            "uncached_input_tokens",
            "cache_write_5m_tokens",
            "cache_write_1h_tokens",
            "cache_write_unclassified_tokens",
            "cache_read_input_tokens",
            "output_tokens",
            "context_input_tokens",
        ):
            _nonnegative_int(getattr(self, field), f"request.{field}")
        expected_context = (
            self.uncached_input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )
        if self.context_input_tokens != expected_context:
            raise ClaudeUsageError(
                f"context_input_tokens mismatch: {self.context_input_tokens} != {expected_context}"
            )
        if not isinstance(self.session_ids, tuple) or any(
            not isinstance(session, str) or not session for session in self.session_ids
        ):
            raise ClaudeUsageError("session_ids must be a tuple of nonempty strings")

    @property
    def cache_creation_input_tokens(self) -> int:
        return (
            self.cache_write_5m_tokens
            + self.cache_write_1h_tokens
            + self.cache_write_unclassified_tokens
        )

    @property
    def total_tokens(self) -> int:
        return self.context_input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "message_id": self.message_id,
            "request_index": self.request_index,
            "model": self.model,
            "service_tier": self.service_tier,
            "speed": self.speed,
            "context_class": self.context_class,
            "uncached_input_tokens": self.uncached_input_tokens,
            "cache_write_5m_tokens": self.cache_write_5m_tokens,
            "cache_write_1h_tokens": self.cache_write_1h_tokens,
            "cache_write_unclassified_tokens": self.cache_write_unclassified_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "output_tokens": self.output_tokens,
            "context_input_tokens": self.context_input_tokens,
            "total_tokens": self.total_tokens,
            "message_timestamp": self.message_timestamp,
            "selected_source_ref": self.selected_source_ref,
            "session_ids": list(self.session_ids),
        }


def _cache_write_counts(entry: Mapping[str, Any], *, prefix: str) -> tuple[int, int, int]:
    total = _nonnegative_int(
        entry.get("cache_creation_input_tokens"),
        f"{prefix}.cache_creation_input_tokens",
    )
    detail = entry.get("cache_creation")
    if detail is None:
        detail = {}
    if not isinstance(detail, Mapping):
        raise ClaudeUsageError(f"{prefix}.cache_creation is not an object")
    five_minute = _nonnegative_int(
        detail.get("ephemeral_5m_input_tokens"), f"{prefix}.cache_creation.ephemeral_5m_input_tokens"
    )
    one_hour = _nonnegative_int(
        detail.get("ephemeral_1h_input_tokens"), f"{prefix}.cache_creation.ephemeral_1h_input_tokens"
    )
    unclassified = total - five_minute - one_hour
    if unclassified < 0:
        raise ClaudeUsageError(f"{prefix} cache-write TTL subtypes exceed total")
    return five_minute, one_hour, unclassified


def build_claude_requests(
    records: Iterable[FrozenJsonRecord],
    *,
    long_context_threshold_tokens: int = 200_000,
) -> tuple[ClaudeRequest, ...]:
    """Return one normalized row per Claude API request represented in logs."""

    if long_context_threshold_tokens < 0:
        raise ValueError("long_context_threshold_tokens must be nonnegative")
    requests: list[ClaudeRequest] = []
    for selected in select_message_versions(records):
        message = selected.message
        usage = message.get("usage")
        if not isinstance(usage, Mapping):
            raise ClaudeUsageError(f"selected message {selected.message_id} lost its usage object")
        entries, _ = _effective_usage_entries(usage)
        service_tier = usage.get("service_tier")
        speed = usage.get("speed")
        tier_label = service_tier if isinstance(service_tier, str) and service_tier else "unknown"
        speed_label = speed if isinstance(speed, str) and speed else "unknown"
        for index, entry in enumerate(entries):
            prefix = f"{selected.message_id}.request[{index}]"
            uncached, cache_creation, cache_read, output = _entry_counts(entry, prefix=prefix)
            five_minute, one_hour, unclassified = _cache_write_counts(entry, prefix=prefix)
            if cache_creation != five_minute + one_hour + unclassified:
                raise AssertionError("cache-write normalization lost tokens")
            model_value = entry.get("model") or message.get("model") or "<unknown>"
            if not isinstance(model_value, str) or not model_value:
                raise ClaudeUsageError(f"{prefix}.model is not a nonempty string")
            model = model_value
            context_input = uncached + cache_creation + cache_read
            context_class = (
                "long_context"
                if context_input > long_context_threshold_tokens
                else "standard_context"
            )
            requests.append(
                ClaudeRequest(
                    request_id=f"{selected.message_id}:{index}",
                    message_id=selected.message_id,
                    request_index=index,
                    model=model,
                    service_tier=tier_label,
                    speed=speed_label,
                    context_class=context_class,
                    uncached_input_tokens=uncached,
                    cache_write_5m_tokens=five_minute,
                    cache_write_1h_tokens=one_hour,
                    cache_write_unclassified_tokens=unclassified,
                    cache_read_input_tokens=cache_read,
                    output_tokens=output,
                    context_input_tokens=context_input,
                    message_timestamp=selected.timestamp,
                    selected_source_ref=selected.selected_source_ref,
                    session_ids=selected.session_ids,
                )
            )
    return tuple(sorted(requests, key=lambda request: request.request_id))


@dataclass(frozen=True)
class ClaudeRateQuote:
    model: str
    service_tier: str
    context_class: str
    uncached_input_usd_per_million: Decimal
    cache_write_5m_usd_per_million: Decimal
    cache_write_1h_usd_per_million: Decimal
    cache_read_usd_per_million: Decimal
    output_usd_per_million: Decimal
    source_ref: str

    def __post_init__(self) -> None:
        for label in ("model", "service_tier", "context_class", "source_ref"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value:
                raise ClaudeUsageError(f"rate quote {label} is empty")
        for field in (
            "uncached_input_usd_per_million",
            "cache_write_5m_usd_per_million",
            "cache_write_1h_usd_per_million",
            "cache_read_usd_per_million",
            "output_usd_per_million",
        ):
            value = getattr(self, field)
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ClaudeUsageError(f"rate quote {field} must be a nonnegative Decimal")


class ClaudeRateTable(Protocol):
    def lookup(
        self, *, model: str, service_tier: str, context_class: str
    ) -> ClaudeRateQuote | None: ...


class MappingClaudeRateTable:
    """A strict exact-key rate table; it performs no implicit fallbacks."""

    def __init__(self, quotes: Iterable[ClaudeRateQuote]):
        self._quotes: dict[tuple[str, str, str], ClaudeRateQuote] = {}
        for quote in quotes:
            key = (quote.model, quote.service_tier, quote.context_class)
            if key in self._quotes:
                raise ClaudeUsageError(f"duplicate Claude rate quote: {key!r}")
            self._quotes[key] = quote

    def lookup(
        self, *, model: str, service_tier: str, context_class: str
    ) -> ClaudeRateQuote | None:
        return self._quotes.get((model, service_tier, context_class))


@dataclass(frozen=True)
class ClaudePriceResult:
    request_id: str
    priced: bool
    status: str
    amount_usd: Decimal | None
    component_usd: Mapping[str, Decimal]
    rate_source_ref: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "priced": self.priced,
            "status": self.status,
            "amount_usd": None if self.amount_usd is None else format(self.amount_usd, "f"),
            "component_usd": {
                key: format(value, "f") for key, value in sorted(self.component_usd.items())
            },
            "rate_source_ref": self.rate_source_ref,
        }


def price_claude_request(
    request: ClaudeRequest, rate_table: ClaudeRateTable
) -> ClaudePriceResult:
    """Apply an exact request-level quote, or return an explicit unpriced row."""

    if request.cache_write_unclassified_tokens:
        return ClaudePriceResult(
            request_id=request.request_id,
            priced=False,
            status="unpriced_cache_write_ttl",
            amount_usd=None,
            component_usd={},
            rate_source_ref=None,
        )
    quote = rate_table.lookup(
        model=request.model,
        service_tier=request.service_tier,
        context_class=request.context_class,
    )
    if quote is None:
        return ClaudePriceResult(
            request_id=request.request_id,
            priced=False,
            status="unpriced_rate_not_found",
            amount_usd=None,
            component_usd={},
            rate_source_ref=None,
        )
    million = Decimal(1_000_000)
    components = {
        "uncached_input": Decimal(request.uncached_input_tokens)
        * quote.uncached_input_usd_per_million
        / million,
        "cache_write_5m": Decimal(request.cache_write_5m_tokens)
        * quote.cache_write_5m_usd_per_million
        / million,
        "cache_write_1h": Decimal(request.cache_write_1h_tokens)
        * quote.cache_write_1h_usd_per_million
        / million,
        "cache_read": Decimal(request.cache_read_input_tokens)
        * quote.cache_read_usd_per_million
        / million,
        "output": Decimal(request.output_tokens) * quote.output_usd_per_million / million,
    }
    return ClaudePriceResult(
        request_id=request.request_id,
        priced=True,
        status="priced",
        amount_usd=sum(components.values(), Decimal(0)),
        component_usd=components,
        rate_source_ref=quote.source_ref,
    )


@dataclass(frozen=True)
class ReceiptUsage:
    receipt_id: str
    session_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    source_ref: str

    def __post_init__(self) -> None:
        for field in ("receipt_id", "session_id", "model", "source_ref"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ClaudeUsageError(f"receipt {field} is empty")
        for field in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            _nonnegative_int(getattr(self, field), f"receipt.{field}")


@dataclass(frozen=True)
class ReceiptReconciliation:
    receipt_id: str
    session_id: str
    model: str
    status: str
    candidate_supplement: bool
    receipt_counts: Mapping[str, int]
    transcript_counts: Mapping[str, int]
    nonnegative_difference: Mapping[str, int] | None
    source_ref: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "session_id": self.session_id,
            "model": self.model,
            "status": self.status,
            "candidate_supplement": self.candidate_supplement,
            "receipt_counts": dict(self.receipt_counts),
            "transcript_counts": dict(self.transcript_counts),
            "nonnegative_difference": (
                None if self.nonnegative_difference is None else dict(self.nonnegative_difference)
            ),
            "source_ref": self.source_ref,
        }


def reconcile_receipt_overlap(
    requests: Iterable[ClaudeRequest], receipts: Iterable[ReceiptUsage]
) -> tuple[ReceiptReconciliation, ...]:
    """Compare CLI receipt model totals with overlapping transcript requests.

    This function never adds a receipt to a ledger.  A model absent from the
    matching transcript session is merely marked as a candidate supplemental
    counter; the caller must make and record that accounting decision.
    """

    request_rows = tuple(requests)
    fields = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    results: list[ReceiptReconciliation] = []
    for receipt in receipts:
        transcript = {field: 0 for field in fields}
        for request in request_rows:
            if receipt.session_id not in request.session_ids or receipt.model != request.model:
                continue
            transcript["input_tokens"] += request.uncached_input_tokens
            transcript["output_tokens"] += request.output_tokens
            transcript["cache_creation_input_tokens"] += request.cache_creation_input_tokens
            transcript["cache_read_input_tokens"] += request.cache_read_input_tokens
        observed = {field: int(getattr(receipt, field)) for field in fields}
        if observed == transcript:
            status = "exact_overlap"
            difference: Mapping[str, int] | None = {field: 0 for field in fields}
            candidate = False
        elif all(transcript[field] == 0 for field in fields) and any(
            observed[field] for field in fields
        ):
            status = "candidate_unlogged_model_usage"
            difference = dict(observed)
            candidate = True
        elif all(observed[field] >= transcript[field] for field in fields):
            status = "receipt_exceeds_transcript_overlap_unresolved"
            difference = {field: observed[field] - transcript[field] for field in fields}
            candidate = False
        else:
            status = "conflicting_overlap"
            difference = None
            candidate = False
        results.append(
            ReceiptReconciliation(
                receipt_id=receipt.receipt_id,
                session_id=receipt.session_id,
                model=receipt.model,
                status=status,
                candidate_supplement=candidate,
                receipt_counts=observed,
                transcript_counts=transcript,
                nonnegative_difference=difference,
                source_ref=receipt.source_ref,
            )
        )
    return tuple(sorted(results, key=lambda row: (row.receipt_id, row.model)))
