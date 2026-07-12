"""Deterministic, fail-closed accounting for RunPod pod consumption.

The module consumes *already sanitized* evidence.  It never fetches provider
state, accepts a credential, or retains a raw provider response.  Three kinds
of evidence remain deliberately distinct:

* complete provider snapshots contain cumulative, per-pod credits consumed;
* local launch/watchdog records reconstruct wall-clock cost at creation rate;
* account-balance windows are reconciliation evidence, never additive spend.

Provider-account lifetime activity is also kept separate from project pod
episodes.  A pod is project-attributed only when a local episode explicitly
claims the complete lifetime of that stable provider pod ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .core import (
    AccountingError,
    AccountingRole,
    LedgerName,
    LedgerRow,
    MeasurementClass,
    canonical_sha256,
)


class RunPodAccountingError(AccountingError):
    """Raised when RunPod accounting evidence violates the frozen contract."""


class RunPodAmbiguityError(RunPodAccountingError):
    """Raised when multiple plausible interpretations cannot be reconciled."""


METHOD_VERSION = "runpod-accounting-v1"
_ZERO = Decimal("0")
_MICROSECONDS_PER_HOUR = Decimal("3600000000")
_RATE_RECONSTRUCTION_QUANTUM_USD = Decimal("0.000000000000001")
# Source decimals are capped at 60 significant digits below.  A private
# 100-digit, half-even context therefore preserves source sums/products and
# deterministically rounds non-terminating divisions.  Provider and balance
# spellings remain exact; only creation-rate reconstructions are quantized to
# 1e-15 USD, matching the retained local estimate precision.
_DECIMAL_CONTEXT = Context(prec=100, rounding=ROUND_HALF_EVEN)


def _require_text(label: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise RunPodAccountingError(f"{label} must be one nonempty line")


def _require_sha256(label: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RunPodAccountingError(f"{label} is not a lowercase SHA-256 digest")


def _require_decimal(
    label: str, value: Decimal, *, nonnegative: bool = True
) -> Decimal:
    if not isinstance(value, Decimal):
        raise RunPodAccountingError(f"{label} must be Decimal; floats are forbidden")
    if not value.is_finite():
        raise RunPodAccountingError(f"{label} must be finite")
    if nonnegative and value < 0:
        raise RunPodAccountingError(f"{label} must be nonnegative")
    if len(value.as_tuple().digits) > 60:
        raise RunPodAccountingError(f"{label} exceeds the 60-digit source limit")
    return value


def _decimal_sum(values: Iterable[Decimal]) -> Decimal:
    with localcontext(_DECIMAL_CONTEXT):
        return sum(values, _ZERO)


def _decimal_subtract(left: Decimal, right: Decimal) -> Decimal:
    with localcontext(_DECIMAL_CONTEXT):
        return left - right


def _decimal_rate_cost(microseconds: int, hourly_rate: Decimal) -> Decimal:
    with localcontext(_DECIMAL_CONTEXT):
        return Decimal(microseconds) * hourly_rate / _MICROSECONDS_PER_HOUR


def _quantize_rate_total(value: Decimal) -> Decimal:
    with localcontext(_DECIMAL_CONTEXT):
        return value.quantize(
            _RATE_RECONSTRUCTION_QUANTUM_USD, rounding=ROUND_HALF_EVEN
        )


def _decimal_from_safe_json(label: str, value: Any) -> Decimal:
    """Decode a sanitized JSON decimal without admitting binary floats."""

    if isinstance(value, bool) or isinstance(value, float):
        raise RunPodAccountingError(f"{label} must not be a bool or float")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, str):
        try:
            result = Decimal(value)
        except Exception as exc:  # Decimal raises several arithmetic subclasses.
            raise RunPodAccountingError(f"{label} is not an exact decimal") from exc
    else:
        raise RunPodAccountingError(f"{label} is not an exact decimal")
    return _require_decimal(label, result)


def _require_nonnegative_int(label: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunPodAccountingError(f"{label} must be a nonnegative integer")
    return value


def _parse_utc(label: str, value: str) -> datetime:
    _require_text(label, value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RunPodAccountingError(f"{label} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RunPodAccountingError(f"{label} lacks a UTC offset")
    if parsed.utcoffset().total_seconds() != 0:
        raise RunPodAccountingError(f"{label} must identify UTC")
    parsed = parsed.astimezone(timezone.utc)
    return parsed


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_from_epoch(value: int) -> str:
    """Return an exact UTC timestamp for an integral Unix epoch second."""

    _require_nonnegative_int("epoch", value)
    return _iso_z(datetime.fromtimestamp(value, tz=timezone.utc))


def normalize_provider_timestamp(value: str) -> str:
    """Normalize ISO-8601 or RunPod's ``+0000 UTC`` timestamp spelling."""

    _require_text("provider timestamp", value)
    try:
        return _iso_z(_parse_utc("provider timestamp", value))
    except RunPodAccountingError:
        pass
    for pattern in (
        "%Y-%m-%d %H:%M:%S.%f +0000 UTC",
        "%Y-%m-%d %H:%M:%S +0000 UTC",
    ):
        try:
            parsed = datetime.strptime(value, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        return _iso_z(parsed)
    raise RunPodAccountingError(f"unrecognized provider timestamp: {value!r}")


def _duration_microseconds(start: datetime, end: datetime) -> int:
    delta = end - start
    micros = (
        delta.days * 86_400_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )
    if micros < 0:
        raise RunPodAccountingError("accounting interval ends before it begins")
    return micros


@dataclass(frozen=True)
class EvidenceBinding:
    """One logical source reference bound to its frozen SHA-256."""

    source_ref: str
    source_sha256: str

    def __post_init__(self) -> None:
        _require_text("source_ref", self.source_ref)
        logical = PurePosixPath(self.source_ref)
        if logical.is_absolute() or ".." in logical.parts:
            raise RunPodAccountingError("source_ref must be a logical relative reference")
        _require_sha256("source_sha256", self.source_sha256)

    @property
    def evidence_ref(self) -> str:
        return f"{self.source_ref}@sha256:{self.source_sha256}"

    def to_dict(self) -> dict[str, str]:
        return {
            "source_ref": self.source_ref,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class ProviderPodTotal:
    """One sanitized cumulative provider total for one stable pod ID."""

    pod_id: str
    amount_usd: Decimal
    billed_time_ms: int
    gpu_type_id: str | None = None

    def __post_init__(self) -> None:
        _require_text("pod_id", self.pod_id)
        _require_decimal("amount_usd", self.amount_usd)
        _require_nonnegative_int("billed_time_ms", self.billed_time_ms)
        if self.gpu_type_id is not None:
            _require_text("gpu_type_id", self.gpu_type_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pod_id": self.pod_id,
            "amount_usd": format(self.amount_usd, "f"),
            "billed_time_ms": self.billed_time_ms,
            "gpu_type_id": self.gpu_type_id,
        }


@dataclass(frozen=True)
class ProviderSnapshot:
    """Frozen safe-field snapshot of exhaustive provider billing + activity."""

    captured_at_utc: str
    pagination_complete: bool
    active_inventory_complete: bool
    active_pod_ids: tuple[str, ...]
    pod_totals: tuple[ProviderPodTotal, ...]
    account_total_usd: Decimal | None
    account_billed_time_ms: int | None
    binding: EvidenceBinding

    def __post_init__(self) -> None:
        _parse_utc("captured_at_utc", self.captured_at_utc)
        if not isinstance(self.binding, EvidenceBinding):
            raise RunPodAccountingError("snapshot binding must be EvidenceBinding")
        if not isinstance(self.pagination_complete, bool):
            raise RunPodAccountingError("pagination_complete must be bool")
        if not isinstance(self.active_inventory_complete, bool):
            raise RunPodAccountingError("active_inventory_complete must be bool")
        active = tuple(sorted(set(self.active_pod_ids)))
        if any(not isinstance(pod_id, str) or not pod_id for pod_id in active):
            raise RunPodAccountingError("active_pod_ids contain an invalid pod ID")
        object.__setattr__(self, "active_pod_ids", active)
        totals = tuple(sorted(self.pod_totals, key=lambda item: item.pod_id))
        if len({item.pod_id for item in totals}) != len(totals):
            raise RunPodAmbiguityError("snapshot contains duplicate per-pod totals")
        object.__setattr__(self, "pod_totals", totals)
        if self.pagination_complete:
            if self.account_total_usd is None or self.account_billed_time_ms is None:
                raise RunPodAccountingError(
                    "complete snapshot requires account total and billed time"
                )
            _require_decimal("account_total_usd", self.account_total_usd)
            _require_nonnegative_int(
                "account_billed_time_ms", self.account_billed_time_ms
            )
            amount_sum = _decimal_sum(item.amount_usd for item in totals)
            time_sum = sum(item.billed_time_ms for item in totals)
            if amount_sum != self.account_total_usd:
                raise RunPodAccountingError(
                    "complete snapshot per-pod amounts do not equal account total"
                )
            if time_sum != self.account_billed_time_ms:
                raise RunPodAccountingError(
                    "complete snapshot per-pod times do not equal account billed time"
                )
        elif self.account_total_usd is not None or self.account_billed_time_ms is not None:
            raise RunPodAccountingError(
                "incomplete snapshot cannot claim an exhaustive account total"
            )

    @property
    def captured_at(self) -> datetime:
        return _parse_utc("captured_at_utc", self.captured_at_utc)

    @property
    def snapshot_id(self) -> str:
        return f"runpod-snapshot:{canonical_sha256(self.safe_payload())}"

    @property
    def evidence_ref(self) -> str:
        return f"{self.snapshot_id}|{self.binding.evidence_ref}"

    def safe_payload(self) -> dict[str, Any]:
        return {
            "schema": "runpod_sanitized_provider_snapshot_v1",
            "captured_at_utc": self.captured_at_utc,
            "pagination_complete": self.pagination_complete,
            "active_inventory_complete": self.active_inventory_complete,
            "active_pod_ids": list(self.active_pod_ids),
            "pod_totals": [item.to_dict() for item in self.pod_totals],
            "account_total_usd": (
                None
                if self.account_total_usd is None
                else format(self.account_total_usd, "f")
            ),
            "account_billed_time_ms": self.account_billed_time_ms,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.safe_payload(),
            "snapshot_id": self.snapshot_id,
            "binding": self.binding.to_dict(),
        }


def provider_snapshot_from_sanitized(
    value: Mapping[str, Any], *, binding: EvidenceBinding
) -> ProviderSnapshot:
    """Parse a strict safe-field artifact; unexpected fields fail closed."""

    allowed = {
        "schema",
        "captured_at_utc",
        "pagination_complete",
        "active_inventory_complete",
        "active_pod_ids",
        "pod_totals",
        "account_total_usd",
        "account_billed_time_ms",
    }
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise RunPodAccountingError(
            f"sanitized provider snapshot has unexpected fields: {unexpected!r}"
        )
    if value.get("schema") != "runpod_sanitized_provider_snapshot_v1":
        raise RunPodAccountingError("wrong sanitized provider snapshot schema")
    raw_pods = value.get("pod_totals")
    if not isinstance(raw_pods, list):
        raise RunPodAccountingError("pod_totals must be a list")
    pods: list[ProviderPodTotal] = []
    pod_fields = {"pod_id", "amount_usd", "billed_time_ms", "gpu_type_id"}
    for index, raw in enumerate(raw_pods):
        if not isinstance(raw, Mapping):
            raise RunPodAccountingError(f"pod_totals[{index}] is not an object")
        extra = sorted(set(raw) - pod_fields)
        if extra:
            raise RunPodAccountingError(
                f"pod_totals[{index}] has unexpected fields: {extra!r}"
            )
        pods.append(
            ProviderPodTotal(
                pod_id=raw.get("pod_id"),
                amount_usd=_decimal_from_safe_json(
                    f"pod_totals[{index}].amount_usd", raw.get("amount_usd")
                ),
                billed_time_ms=raw.get("billed_time_ms"),
                gpu_type_id=(
                    raw["gpu_type_id"]
                    if raw.get("gpu_type_id") is not None
                    else None
                ),
            )
        )
    raw_active = value.get("active_pod_ids")
    if not isinstance(raw_active, list):
        raise RunPodAccountingError("active_pod_ids must be a list")
    complete = value.get("pagination_complete")
    account_amount = value.get("account_total_usd")
    return ProviderSnapshot(
        captured_at_utc=value.get("captured_at_utc"),
        pagination_complete=complete,
        active_inventory_complete=value.get("active_inventory_complete"),
        active_pod_ids=tuple(raw_active),
        pod_totals=tuple(pods),
        account_total_usd=(
            _decimal_from_safe_json("account_total_usd", account_amount)
            if account_amount is not None
            else None
        ),
        account_billed_time_ms=value.get("account_billed_time_ms"),
        binding=binding,
    )


@dataclass(frozen=True)
class RatePoint:
    effective_at_utc: str
    hourly_rate_usd: Decimal
    binding: EvidenceBinding

    def __post_init__(self) -> None:
        _parse_utc("rate.effective_at_utc", self.effective_at_utc)
        _require_decimal("rate.hourly_rate_usd", self.hourly_rate_usd)
        if not isinstance(self.binding, EvidenceBinding):
            raise RunPodAccountingError("rate binding must be EvidenceBinding")

    @property
    def effective_at(self) -> datetime:
        return _parse_utc("rate.effective_at_utc", self.effective_at_utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_at_utc": self.effective_at_utc,
            "hourly_rate_usd": format(self.hourly_rate_usd, "f"),
            "binding": self.binding.to_dict(),
        }


@dataclass(frozen=True)
class LocalPodEpisode:
    """Sanitized project attribution from local launch/watchdog evidence."""

    episode_id: str
    protocol_id: str
    pod_id: str
    started_at_utc: str
    terminal_at_utc: str | None
    observed_through_utc: str | None
    rate_points: tuple[RatePoint, ...]
    bindings: tuple[EvidenceBinding, ...]
    terminal_binding: EvidenceBinding | None = None
    observed_through_binding: EvidenceBinding | None = None
    owns_full_pod_lifetime: bool = False

    def __post_init__(self) -> None:
        for label in ("episode_id", "protocol_id", "pod_id"):
            _require_text(label, getattr(self, label))
        started = _parse_utc("started_at_utc", self.started_at_utc)
        terminal = (
            _parse_utc("terminal_at_utc", self.terminal_at_utc)
            if self.terminal_at_utc is not None
            else None
        )
        observed = (
            _parse_utc("observed_through_utc", self.observed_through_utc)
            if self.observed_through_utc is not None
            else None
        )
        if terminal is not None and terminal < started:
            raise RunPodAccountingError("episode terminal precedes start")
        if observed is not None and observed < started:
            raise RunPodAccountingError("episode observation precedes start")
        if terminal is not None and observed is not None and observed > terminal:
            raise RunPodAccountingError("episode observation extends past terminal")
        if (terminal is None) != (self.terminal_binding is None):
            raise RunPodAccountingError(
                "terminal boundary and terminal_binding must be supplied together"
            )
        if (observed is None) != (self.observed_through_binding is None):
            raise RunPodAccountingError(
                "observed-through boundary and binding must be supplied together"
            )
        for label, binding in (
            ("terminal_binding", self.terminal_binding),
            ("observed_through_binding", self.observed_through_binding),
        ):
            if binding is not None and not isinstance(binding, EvidenceBinding):
                raise RunPodAccountingError(f"{label} must be EvidenceBinding")
        if not isinstance(self.owns_full_pod_lifetime, bool):
            raise RunPodAccountingError("owns_full_pod_lifetime must be bool")
        if not self.bindings:
            raise RunPodAccountingError("episode requires at least one evidence binding")
        if any(not isinstance(binding, EvidenceBinding) for binding in self.bindings):
            raise RunPodAccountingError("episode bindings must be EvidenceBinding values")
        bindings = tuple(
            sorted(set(self.bindings), key=lambda item: item.evidence_ref)
        )
        object.__setattr__(self, "bindings", bindings)
        points = _normalize_rate_points(self.rate_points)
        for point in points:
            if point.effective_at < started:
                raise RunPodAccountingError("rate point precedes episode start")
            endpoint = terminal or observed
            if endpoint is not None and point.effective_at > endpoint:
                raise RunPodAccountingError("rate point follows episode evidence window")
        object.__setattr__(self, "rate_points", points)

    @property
    def started_at(self) -> datetime:
        return _parse_utc("started_at_utc", self.started_at_utc)

    @property
    def terminal_at(self) -> datetime | None:
        if self.terminal_at_utc is None:
            return None
        return _parse_utc("terminal_at_utc", self.terminal_at_utc)

    @property
    def observed_through(self) -> datetime | None:
        if self.observed_through_utc is None:
            return None
        return _parse_utc("observed_through_utc", self.observed_through_utc)

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        refs = {binding.evidence_ref for binding in self.bindings}
        refs.update(point.binding.evidence_ref for point in self.rate_points)
        refs.update(
            binding.evidence_ref
            for binding in (self.terminal_binding, self.observed_through_binding)
            if binding is not None
        )
        return tuple(sorted(refs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "protocol_id": self.protocol_id,
            "pod_id": self.pod_id,
            "started_at_utc": self.started_at_utc,
            "terminal_at_utc": self.terminal_at_utc,
            "observed_through_utc": self.observed_through_utc,
            "rate_points": [point.to_dict() for point in self.rate_points],
            "bindings": [binding.to_dict() for binding in self.bindings],
            "terminal_binding": (
                None if self.terminal_binding is None else self.terminal_binding.to_dict()
            ),
            "observed_through_binding": (
                None
                if self.observed_through_binding is None
                else self.observed_through_binding.to_dict()
            ),
            "owns_full_pod_lifetime": self.owns_full_pod_lifetime,
        }


def _normalize_rate_points(points: Sequence[RatePoint]) -> tuple[RatePoint, ...]:
    by_time: dict[datetime, RatePoint] = {}
    for point in points:
        prior = by_time.get(point.effective_at)
        if prior is not None and prior.hourly_rate_usd != point.hourly_rate_usd:
            raise RunPodAmbiguityError(
                f"conflicting hourly rates at {point.effective_at_utc}"
            )
        if prior is None or point.binding.evidence_ref < prior.binding.evidence_ref:
            by_time[point.effective_at] = point
    return tuple(by_time[key] for key in sorted(by_time))


@dataclass(frozen=True)
class LocalRateEstimate:
    amount_usd: Decimal | None
    measurement_class: MeasurementClass
    duration_microseconds: int
    evidence_refs: tuple[str, ...]
    note: str


def reconstruct_local_rate_cost(episode: LocalPodEpisode) -> LocalRateEstimate:
    """Integrate observed creation-rate segments using Decimal arithmetic."""

    endpoint = episode.terminal_at or episode.observed_through
    if endpoint is None:
        return LocalRateEstimate(
            amount_usd=None,
            measurement_class=MeasurementClass.UNKNOWN,
            duration_microseconds=0,
            evidence_refs=episode.evidence_refs,
            note="No terminal or observed-through boundary is available.",
        )
    if not episode.rate_points or episode.rate_points[0].effective_at != episode.started_at:
        return LocalRateEstimate(
            amount_usd=None,
            measurement_class=MeasurementClass.UNKNOWN,
            duration_microseconds=_duration_microseconds(episode.started_at, endpoint),
            evidence_refs=episode.evidence_refs,
            note="Hourly-rate coverage does not begin at the episode boundary.",
        )
    amount = _ZERO
    duration = 0
    for index, point in enumerate(episode.rate_points):
        segment_start = point.effective_at
        segment_end = (
            episode.rate_points[index + 1].effective_at
            if index + 1 < len(episode.rate_points)
            else endpoint
        )
        micros = _duration_microseconds(segment_start, segment_end)
        duration += micros
        amount = _decimal_sum(
            (amount, _decimal_rate_cost(micros, point.hourly_rate_usd))
        )
    expected_duration = _duration_microseconds(episode.started_at, endpoint)
    if duration != expected_duration:
        raise RunPodAccountingError("rate integration did not cover episode exactly")
    amount = _quantize_rate_total(amount)
    confidence = (
        MeasurementClass.RECONSTRUCTED
        if episode.terminal_at is not None
        else MeasurementClass.LOWER_BOUND
    )
    note = (
        f"Integrated {len(episode.rate_points)} exact local rate segment(s) "
        f"over {duration} microseconds; this is not a provider invoice."
    )
    if episode.terminal_at is None:
        note += " The pod lacked a terminal boundary, so only an observed lower bound counts."
    return LocalRateEstimate(
        amount_usd=amount,
        measurement_class=confidence,
        duration_microseconds=duration,
        evidence_refs=episode.evidence_refs,
        note=note,
    )


def local_episode_from_state_snapshot(
    value: Mapping[str, Any],
    *,
    binding: EvidenceBinding,
    episode_id: str | None = None,
    protocol_id: str = "local-runpod-state",
    terminal_at_utc: str | None = None,
    observed_through_utc: str | None = None,
    terminal_binding: EvidenceBinding | None = None,
    observation_binding: EvidenceBinding | None = None,
) -> LocalPodEpisode:
    """Allowlist safe fields from a local pod state object and discard the rest."""

    pod_id = value.get("id")
    created = value.get("createdAt")
    rate = value.get("costPerHr")
    if not isinstance(pod_id, str) or not pod_id:
        raise RunPodAccountingError("local pod state lacks id")
    if not isinstance(created, str):
        raise RunPodAccountingError("local pod state lacks createdAt")
    started = normalize_provider_timestamp(created)
    exact_rate = _decimal_from_safe_json("local pod costPerHr", rate)
    bindings = [binding]
    if terminal_binding is not None:
        bindings.append(terminal_binding)
    if observation_binding is not None:
        bindings.append(observation_binding)
    return LocalPodEpisode(
        episode_id=episode_id or f"runpod-pod-{pod_id}",
        protocol_id=protocol_id,
        pod_id=pod_id,
        started_at_utc=started,
        terminal_at_utc=terminal_at_utc,
        observed_through_utc=observed_through_utc,
        rate_points=(RatePoint(started, exact_rate, binding),),
        bindings=tuple(bindings),
        terminal_binding=terminal_binding,
        observed_through_binding=observation_binding,
        owns_full_pod_lifetime=True,
    )


def precision_probe_episode_from_records(
    budget: Mapping[str, Any],
    *,
    budget_binding: EvidenceBinding,
    settlement: Mapping[str, Any] | None = None,
    settlement_binding: EvidenceBinding | None = None,
) -> LocalPodEpisode:
    """Extract the exact safe timing fields from P01 or P02 evidence records."""

    schema = budget.get("schema")
    expected_protocols = {
        "precision_probe_p01_provider_budget_v1": "precision-probe-p01",
        "precision_probe_p02_provider_budget_v1": "precision-probe-p02",
    }
    if schema not in expected_protocols:
        raise RunPodAccountingError("unsupported precision-probe budget schema")
    protocol_id = budget.get("protocol_id")
    pod_id = budget.get("provider_pod_id")
    start_epoch = budget.get("provider_clock_started_epoch")
    elapsed = budget.get("provider_elapsed_seconds_at_job_record")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise RunPodAccountingError("precision-probe budget lacks protocol_id")
    if protocol_id != expected_protocols[schema]:
        raise RunPodAccountingError("precision-probe schema/protocol mismatch")
    if not isinstance(pod_id, str) or not pod_id:
        raise RunPodAccountingError("precision-probe budget lacks provider_pod_id")
    _require_nonnegative_int("provider_clock_started_epoch", start_epoch)
    rate = _decimal_from_safe_json("hourly_cost_usd", budget.get("hourly_cost_usd"))
    started = utc_from_epoch(start_epoch)
    terminal: str | None = None
    observed: str | None = None
    bindings = [budget_binding]
    if settlement is not None:
        if schema != "precision_probe_p02_provider_budget_v1":
            raise RunPodAccountingError("only P02 has a supported settlement record")
        if settlement_binding is None:
            raise RunPodAccountingError("settlement record lacks source binding")
        if settlement.get("schema") != "precision_probe_p02_provider_settlement_v1":
            raise RunPodAccountingError("unsupported precision-probe settlement schema")
        for field, expected in (
            ("protocol_id", protocol_id),
            ("provider_pod_id", pod_id),
            ("provider_clock_started_epoch", start_epoch),
        ):
            if settlement.get(field) != expected:
                raise RunPodAmbiguityError(f"budget/settlement {field} mismatch")
        settlement_rate = _decimal_from_safe_json(
            "settlement.hourly_cost_usd", settlement.get("hourly_cost_usd")
        )
        if settlement_rate != rate:
            raise RunPodAmbiguityError("budget/settlement hourly rate mismatch")
        delete_returned = settlement.get("delete_returned_epoch")
        _require_nonnegative_int("delete_returned_epoch", delete_returned)
        if delete_returned < start_epoch:
            raise RunPodAccountingError("DELETE return precedes provider start")
        terminal = utc_from_epoch(delete_returned)
        bindings.append(settlement_binding)
    else:
        _require_nonnegative_int("provider_elapsed_seconds_at_job_record", elapsed)
        observed = utc_from_epoch(start_epoch + elapsed)
    return LocalPodEpisode(
        episode_id=protocol_id,
        protocol_id=protocol_id,
        pod_id=pod_id,
        started_at_utc=started,
        terminal_at_utc=terminal,
        observed_through_utc=observed,
        rate_points=(RatePoint(started, rate, budget_binding),),
        bindings=tuple(bindings),
        terminal_binding=settlement_binding if terminal is not None else None,
        observed_through_binding=budget_binding if observed is not None else None,
        owns_full_pod_lifetime=True,
    )


@dataclass(frozen=True)
class BalanceObservation:
    observed_at_utc: str
    balance_usd: Decimal
    active_inventory_complete: bool
    active_pod_ids: tuple[str, ...]
    binding: EvidenceBinding

    def __post_init__(self) -> None:
        _parse_utc("balance.observed_at_utc", self.observed_at_utc)
        _require_decimal("balance.balance_usd", self.balance_usd)
        if not isinstance(self.binding, EvidenceBinding):
            raise RunPodAccountingError("balance binding must be EvidenceBinding")
        if not isinstance(self.active_inventory_complete, bool):
            raise RunPodAccountingError("active_inventory_complete must be bool")
        active = tuple(sorted(set(self.active_pod_ids)))
        if any(not isinstance(pod_id, str) or not pod_id for pod_id in active):
            raise RunPodAccountingError("balance observation has invalid active pod ID")
        object.__setattr__(self, "active_pod_ids", active)

    @property
    def observed_at(self) -> datetime:
        return _parse_utc("balance.observed_at_utc", self.observed_at_utc)

    @property
    def observation_id(self) -> str:
        safe = {
            "observed_at_utc": self.observed_at_utc,
            "balance_usd": format(self.balance_usd, "f"),
            "active_inventory_complete": self.active_inventory_complete,
            "active_pod_ids": list(self.active_pod_ids),
        }
        return f"runpod-balance:{canonical_sha256(safe)}"


@dataclass(frozen=True)
class BalanceAdjustment:
    adjustment_id: str
    occurred_at_utc: str
    balance_effect_usd: Decimal
    kind: str
    binding: EvidenceBinding

    def __post_init__(self) -> None:
        _require_text("adjustment_id", self.adjustment_id)
        _parse_utc("adjustment.occurred_at_utc", self.occurred_at_utc)
        _require_decimal(
            "adjustment.balance_effect_usd",
            self.balance_effect_usd,
            nonnegative=False,
        )
        _require_text("adjustment.kind", self.kind)
        if not isinstance(self.binding, EvidenceBinding):
            raise RunPodAccountingError("adjustment binding must be EvidenceBinding")

    @property
    def occurred_at(self) -> datetime:
        return _parse_utc("adjustment.occurred_at_utc", self.occurred_at_utc)


@dataclass(frozen=True)
class BalanceWindow:
    window_id: str
    start: BalanceObservation
    end: BalanceObservation
    covered_pod_ids: tuple[str, ...]
    exclusive_provider_activity: bool
    exclusivity_binding: EvidenceBinding | None = None
    adjustments: tuple[BalanceAdjustment, ...] = ()

    def __post_init__(self) -> None:
        _require_text("window_id", self.window_id)
        if self.end.observed_at <= self.start.observed_at:
            raise RunPodAccountingError("balance window is not forward in time")
        covered = tuple(sorted(set(self.covered_pod_ids)))
        if any(not isinstance(pod_id, str) or not pod_id for pod_id in covered):
            raise RunPodAccountingError("balance window has invalid covered pod ID")
        object.__setattr__(self, "covered_pod_ids", covered)
        if not isinstance(self.exclusive_provider_activity, bool):
            raise RunPodAccountingError("exclusive_provider_activity must be bool")
        if self.exclusive_provider_activity and self.exclusivity_binding is None:
            raise RunPodAccountingError(
                "exclusive balance window requires a source-bound exclusivity attestation"
            )
        if self.exclusivity_binding is not None and not isinstance(
            self.exclusivity_binding, EvidenceBinding
        ):
            raise RunPodAccountingError(
                "exclusivity_binding must be EvidenceBinding or None"
            )
        if self.exclusive_provider_activity and (
            not self.start.active_inventory_complete
            or not self.end.active_inventory_complete
            or self.start.active_pod_ids
            or self.end.active_pod_ids
        ):
            raise RunPodAmbiguityError(
                "exclusive balance window contradicts endpoint active-pod inventory"
            )
        adjustments = tuple(sorted(self.adjustments, key=lambda item: item.adjustment_id))
        if len({item.adjustment_id for item in adjustments}) != len(adjustments):
            raise RunPodAmbiguityError("balance window repeats an adjustment ID")
        for adjustment in adjustments:
            if not (self.start.observed_at < adjustment.occurred_at <= self.end.observed_at):
                raise RunPodAccountingError("balance adjustment falls outside its window")
        object.__setattr__(self, "adjustments", adjustments)

    @property
    def raw_balance_decrease_usd(self) -> Decimal:
        return _decimal_subtract(self.start.balance_usd, self.end.balance_usd)

    @property
    def adjusted_consumption_usd(self) -> Decimal:
        effects = _decimal_sum(item.balance_effect_usd for item in self.adjustments)
        return _decimal_subtract(
            _decimal_sum((self.start.balance_usd, effects)), self.end.balance_usd
        )

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        refs = {
            self.start.binding.evidence_ref,
            self.end.binding.evidence_ref,
        }
        refs.update(item.binding.evidence_ref for item in self.adjustments)
        if self.exclusivity_binding is not None:
            refs.add(self.exclusivity_binding.evidence_ref)
        return tuple(sorted(refs))

    @property
    def stable_id(self) -> str:
        safe = {
            "start_observation_id": self.start.observation_id,
            "end_observation_id": self.end.observation_id,
            "covered_pod_ids": list(self.covered_pod_ids),
            "exclusive_provider_activity": self.exclusive_provider_activity,
            "adjustments": [
                {
                    "adjustment_id": item.adjustment_id,
                    "occurred_at_utc": item.occurred_at_utc,
                    "balance_effect_usd": format(item.balance_effect_usd, "f"),
                    "kind": item.kind,
                }
                for item in self.adjustments
            ],
        }
        return f"runpod-balance-window:{canonical_sha256(safe)}"


@dataclass(frozen=True)
class EpisodeReconciliation:
    episode_id: str
    pod_id: str
    primary_row: LedgerRow
    provider_crosscheck_row: LedgerRow | None
    rate_crosscheck_row: LedgerRow | None
    provider_amount_usd: Decimal | None
    provider_settled: bool
    rate_amount_usd: Decimal | None
    provider_minus_rate_usd: Decimal | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "pod_id": self.pod_id,
            "primary_row": self.primary_row.to_dict(),
            "provider_crosscheck_row": (
                None
                if self.provider_crosscheck_row is None
                else self.provider_crosscheck_row.to_dict()
            ),
            "rate_crosscheck_row": (
                None
                if self.rate_crosscheck_row is None
                else self.rate_crosscheck_row.to_dict()
            ),
            "provider_amount_usd": (
                None
                if self.provider_amount_usd is None
                else format(self.provider_amount_usd, "f")
            ),
            "provider_settled": self.provider_settled,
            "rate_amount_usd": (
                None if self.rate_amount_usd is None else format(self.rate_amount_usd, "f")
            ),
            "provider_minus_rate_usd": (
                None
                if self.provider_minus_rate_usd is None
                else format(self.provider_minus_rate_usd, "f")
            ),
        }


@dataclass(frozen=True)
class BalanceReconciliation:
    window_id: str
    row: LedgerRow
    raw_balance_decrease_usd: Decimal
    adjusted_consumption_usd: Decimal
    covered_primary_usd: Decimal | None
    adjusted_minus_primary_usd: Decimal | None
    exactly_comparable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "row": self.row.to_dict(),
            "raw_balance_decrease_usd": format(self.raw_balance_decrease_usd, "f"),
            "adjusted_consumption_usd": format(self.adjusted_consumption_usd, "f"),
            "covered_primary_usd": (
                None
                if self.covered_primary_usd is None
                else format(self.covered_primary_usd, "f")
            ),
            "adjusted_minus_primary_usd": (
                None
                if self.adjusted_minus_primary_usd is None
                else format(self.adjusted_minus_primary_usd, "f")
            ),
            "exactly_comparable": self.exactly_comparable,
        }


@dataclass(frozen=True)
class RunPodLedger:
    episodes: tuple[EpisodeReconciliation, ...]
    balance_reconciliations: tuple[BalanceReconciliation, ...]
    account_lifetime_rows: tuple[LedgerRow, ...]
    snapshot_ids: tuple[str, ...]
    unmatched_provider_pod_ids: tuple[str, ...]
    unmatched_project_pod_ids: tuple[str, ...]
    exact_reconciliation_reasons: tuple[str, ...]

    @property
    def project_rows(self) -> tuple[LedgerRow, ...]:
        return tuple(item.primary_row for item in self.episodes)

    @property
    def crosscheck_rows(self) -> tuple[LedgerRow, ...]:
        provider_rows = tuple(
            item.provider_crosscheck_row
            for item in self.episodes
            if item.provider_crosscheck_row is not None
        )
        rate_rows = tuple(
            item.rate_crosscheck_row
            for item in self.episodes
            if item.rate_crosscheck_row is not None
        )
        balance_rows = tuple(item.row for item in self.balance_reconciliations)
        return provider_rows + rate_rows + balance_rows  # type: ignore[return-value]

    @property
    def project_additive_known_usd(self) -> Decimal:
        total = _ZERO
        overlap_keys: set[str] = set()
        for row in self.project_rows:
            if row.accounting_role is not AccountingRole.PRIMARY_ADDITIVE:
                continue
            if row.overlap_key is None:
                raise RunPodAccountingError("primary RunPod row lacks overlap_key")
            if row.overlap_key in overlap_keys:
                raise RunPodAmbiguityError(
                    f"duplicate project overlap key: {row.overlap_key}"
                )
            overlap_keys.add(row.overlap_key)
            if row.quantity is None:
                raise RunPodAccountingError("primary additive RunPod row lacks quantity")
            total = _decimal_sum((total, row.quantity))
        return total

    @property
    def exact_reconciliation(self) -> bool:
        return not self.exact_reconciliation_reasons

    @property
    def account_lifetime_observed_usd(self) -> Decimal:
        return _decimal_sum(
            (row.quantity for row in self.account_lifetime_rows if row.quantity is not None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "runpod_provider_ledger_v1",
            "method_version": METHOD_VERSION,
            "project_additive_known_usd": format(
                self.project_additive_known_usd, "f"
            ),
            "account_lifetime_observed_usd": format(
                self.account_lifetime_observed_usd, "f"
            ),
            "exact_reconciliation": self.exact_reconciliation,
            "exact_reconciliation_reasons": list(self.exact_reconciliation_reasons),
            "snapshot_ids": list(self.snapshot_ids),
            "unmatched_provider_pod_ids": list(self.unmatched_provider_pod_ids),
            "unmatched_project_pod_ids": list(self.unmatched_project_pod_ids),
            "episodes": [item.to_dict() for item in self.episodes],
            "balance_reconciliations": [
                item.to_dict() for item in self.balance_reconciliations
            ],
            "account_lifetime_rows": [
                row.to_dict() for row in self.account_lifetime_rows
            ],
        }


def _normalize_snapshots(
    snapshots: Iterable[ProviderSnapshot],
) -> tuple[ProviderSnapshot, ...]:
    by_id: dict[str, ProviderSnapshot] = {}
    by_time: dict[datetime, str] = {}
    for snapshot in snapshots:
        prior_id = by_time.get(snapshot.captured_at)
        if prior_id is not None and prior_id != snapshot.snapshot_id:
            raise RunPodAmbiguityError(
                f"different provider snapshots claim {snapshot.captured_at_utc}"
            )
        by_time[snapshot.captured_at] = snapshot.snapshot_id
        prior = by_id.get(snapshot.snapshot_id)
        if prior is None or snapshot.binding.evidence_ref < prior.binding.evidence_ref:
            by_id[snapshot.snapshot_id] = snapshot
    return tuple(sorted(by_id.values(), key=lambda item: item.captured_at))


def _normalize_episodes(
    episodes: Iterable[LocalPodEpisode],
) -> tuple[LocalPodEpisode, ...]:
    by_episode: dict[str, LocalPodEpisode] = {}
    by_pod: dict[str, LocalPodEpisode] = {}
    for episode in episodes:
        prior_episode = by_episode.get(episode.episode_id)
        if prior_episode is not None and prior_episode != episode:
            raise RunPodAmbiguityError(
                f"episode_id {episode.episode_id!r} has conflicting evidence"
            )
        prior_pod = by_pod.get(episode.pod_id)
        if prior_pod is not None and prior_pod != episode:
            raise RunPodAmbiguityError(
                f"pod {episode.pod_id!r} maps to multiple project episodes"
            )
        by_episode[episode.episode_id] = episode
        by_pod[episode.pod_id] = episode
    return tuple(sorted(by_episode.values(), key=lambda item: item.episode_id))


@dataclass(frozen=True)
class _ProviderObservation:
    snapshot: ProviderSnapshot
    total: ProviderPodTotal


def _provider_histories(
    snapshots: Sequence[ProviderSnapshot],
) -> dict[str, tuple[_ProviderObservation, ...]]:
    histories: dict[str, list[_ProviderObservation]] = {}
    seen_in_any_snapshot: set[str] = set()
    for snapshot in snapshots:
        current_ids = {item.pod_id for item in snapshot.pod_totals}
        if snapshot.pagination_complete:
            missing = seen_in_any_snapshot - current_ids
            if missing:
                raise RunPodAmbiguityError(
                    "complete provider snapshot dropped prior lifetime pod IDs: "
                    f"{sorted(missing)!r}"
                )
        seen_in_any_snapshot.update(current_ids)
        for total in snapshot.pod_totals:
            prior = histories.get(total.pod_id, [])
            if prior:
                previous = prior[-1].total
                if total.amount_usd < previous.amount_usd:
                    raise RunPodAmbiguityError(
                        f"provider amount decreased for pod {total.pod_id}"
                    )
                if total.billed_time_ms < previous.billed_time_ms:
                    raise RunPodAmbiguityError(
                        f"provider billed time decreased for pod {total.pod_id}"
                    )
            histories.setdefault(total.pod_id, []).append(
                _ProviderObservation(snapshot=snapshot, total=total)
            )
    return {pod_id: tuple(rows) for pod_id, rows in histories.items()}


def _is_settled(
    episode: LocalPodEpisode, observations: Sequence[_ProviderObservation]
) -> bool:
    if len(observations) < 2:
        return False
    left, right = observations[-2:]
    boundary = episode.terminal_at or episode.started_at
    return (
        left.snapshot.snapshot_id != right.snapshot.snapshot_id
        and left.snapshot.pagination_complete
        and right.snapshot.pagination_complete
        and left.snapshot.active_inventory_complete
        and right.snapshot.active_inventory_complete
        and episode.pod_id not in left.snapshot.active_pod_ids
        and episode.pod_id not in right.snapshot.active_pod_ids
        and left.snapshot.captured_at >= boundary
        and right.snapshot.captured_at >= boundary
        and left.total.amount_usd == right.total.amount_usd
        and left.total.billed_time_ms == right.total.billed_time_ms
    )


def _provider_primary_row(
    episode: LocalPodEpisode,
    observations: Sequence[_ProviderObservation],
) -> LedgerRow:
    latest = observations[-1]
    refs = set(episode.evidence_refs)
    refs.update(item.snapshot.evidence_ref for item in observations[-2:])
    return LedgerRow(
        row_id=f"runpod:project:pod:{episode.pod_id}:provider-credits",
        ledger=LedgerName.METERED_CONSUMPTION,
        provider="runpod",
        category="project_pod_credits_consumed",
        quantity=latest.total.amount_usd,
        unit="USD credits",
        currency="USD",
        measurement_class=MeasurementClass.EXACT_SOURCE_RECORD,
        accounting_role=AccountingRole.PRIMARY_ADDITIVE,
        overlap_key=f"runpod:pod:{episode.pod_id}",
        window_start_utc=episode.started_at_utc,
        source_cutoff_utc=latest.snapshot.captured_at_utc,
        evidence_refs=tuple(sorted(refs)),
        method_version=METHOD_VERSION,
        note=(
            f"Provider pod ID {episode.pod_id}; complete pod lifetime explicitly "
            "project-attributed. Two distinct complete inactive snapshots agree "
            "after termination. Credits consumed are not cash deposited."
        ),
    )


def _provider_lower_bound_crosscheck_row(
    episode: LocalPodEpisode,
    observations: Sequence[_ProviderObservation],
) -> LedgerRow:
    latest = observations[-1]
    refs = set(episode.evidence_refs)
    refs.add(latest.snapshot.evidence_ref)
    return LedgerRow(
        row_id=f"runpod:project:pod:{episode.pod_id}:provider-lower-bound",
        ledger=LedgerName.METERED_CONSUMPTION,
        provider="runpod",
        category="unsettled_provider_pod_credits_lower_bound",
        quantity=latest.total.amount_usd,
        unit="USD credits",
        currency="USD",
        measurement_class=MeasurementClass.LOWER_BOUND,
        accounting_role=AccountingRole.CROSSCHECK_NONADDITIVE,
        overlap_key=f"runpod:pod:{episode.pod_id}",
        window_start_utc=episode.started_at_utc,
        source_cutoff_utc=latest.snapshot.captured_at_utc,
        evidence_refs=tuple(sorted(refs)),
        method_version=METHOD_VERSION,
        note=(
            "Latest cumulative provider observation is not yet a distinct "
            "two-snapshot settlement. It remains a nonadditive lower bound."
        ),
    )


def _unknown_provider_primary_row(
    episode: LocalPodEpisode,
    estimate: LocalRateEstimate,
    *,
    additional_evidence_refs: Iterable[str] = (),
) -> LedgerRow:
    evidence_refs = tuple(
        sorted(set(estimate.evidence_refs).union(additional_evidence_refs))
    )
    return LedgerRow(
        row_id=f"runpod:project:pod:{episode.pod_id}:unknown-provider-cost",
        ledger=LedgerName.UNKNOWNS,
        provider="runpod",
        category="project_pod_provider_cost_unknown",
        quantity=None,
        unit="USD credits",
        currency="USD",
        measurement_class=MeasurementClass.UNKNOWN,
        accounting_role=AccountingRole.CONTEXT_NONADDITIVE,
        overlap_key=f"runpod:pod:{episode.pod_id}",
        window_start_utc=episode.started_at_utc,
        source_cutoff_utc=episode.terminal_at_utc or episode.observed_through_utc,
        evidence_refs=evidence_refs,
        method_version=METHOD_VERSION,
        note=(
            "No attributable per-pod provider billing row is available. Local "
            "creation-rate reconstruction, when possible, remains a separate "
            "nonadditive cross-check. "
            + estimate.note
        ),
    )


def _rate_crosscheck_row(
    episode: LocalPodEpisode, estimate: LocalRateEstimate
) -> LedgerRow | None:
    if estimate.measurement_class is MeasurementClass.UNKNOWN:
        return None
    if estimate.amount_usd is None:
        raise RunPodAccountingError("known rate cross-check lacks an amount")
    return LedgerRow(
        row_id=f"runpod:project:pod:{episode.pod_id}:rate-crosscheck",
        ledger=LedgerName.METERED_CONSUMPTION,
        provider="runpod",
        category="creation_rate_cost_crosscheck",
        quantity=estimate.amount_usd,
        unit="USD credits",
        currency="USD",
        measurement_class=estimate.measurement_class,
        accounting_role=AccountingRole.CROSSCHECK_NONADDITIVE,
        overlap_key=f"runpod:pod:{episode.pod_id}",
        window_start_utc=episode.started_at_utc,
        source_cutoff_utc=episode.terminal_at_utc or episode.observed_through_utc,
        evidence_refs=estimate.evidence_refs,
        method_version=METHOD_VERSION,
        note=estimate.note,
    )


def _account_lifetime_rows(
    snapshots: Sequence[ProviderSnapshot],
) -> tuple[LedgerRow, ...]:
    if not snapshots:
        return ()
    complete = [snapshot for snapshot in snapshots if snapshot.pagination_complete]
    chosen = complete[-1] if complete else snapshots[-1]
    confidence = (
        MeasurementClass.EXACT_SOURCE_RECORD
        if chosen.pagination_complete
        else MeasurementClass.LOWER_BOUND
    )
    return tuple(
        LedgerRow(
            row_id=f"runpod:account-lifetime:pod:{item.pod_id}",
            ledger=LedgerName.METERED_CONSUMPTION,
            provider="runpod",
            category="provider_account_lifetime_pod_credits_observed",
            quantity=item.amount_usd,
            unit="USD credits",
            currency="USD",
            measurement_class=confidence,
            accounting_role=AccountingRole.CONTEXT_NONADDITIVE,
            overlap_key=f"runpod:pod:{item.pod_id}",
            source_cutoff_utc=chosen.captured_at_utc,
            evidence_refs=(chosen.evidence_ref,),
            method_version=METHOD_VERSION,
            note=(
                "Provider-account lifetime context; not project-attributed and never "
                "added to project pod rows."
            ),
        )
        for item in chosen.pod_totals
    )


def _balance_reconciliations(
    windows: Iterable[BalanceWindow],
    episodes: Sequence[LocalPodEpisode],
    reconciliations: Sequence[EpisodeReconciliation],
) -> tuple[BalanceReconciliation, ...]:
    by_pod = {item.pod_id: item for item in reconciliations}
    episode_by_pod = {item.pod_id: item for item in episodes}
    by_id: dict[str, BalanceWindow] = {}
    for window in windows:
        prior = by_id.get(window.window_id)
        if prior is not None and prior != window:
            raise RunPodAmbiguityError(
                f"balance window {window.window_id!r} has conflicting evidence"
            )
        by_id[window.window_id] = window
    results: list[BalanceReconciliation] = []
    for window in sorted(by_id.values(), key=lambda item: item.window_id):
        missing = set(window.covered_pod_ids) - set(by_pod)
        if missing:
            raise RunPodAccountingError(
                f"balance window covers unknown project pods: {sorted(missing)!r}"
            )
        comparable = window.exclusive_provider_activity
        comparable = (
            comparable
            and window.start.active_inventory_complete
            and window.end.active_inventory_complete
            and not window.start.active_pod_ids
            and not window.end.active_pod_ids
        )
        total = _ZERO
        for pod_id in window.covered_pod_ids:
            reconciliation = by_pod[pod_id]
            episode = episode_by_pod[pod_id]
            row = reconciliation.primary_row
            endpoint = episode.terminal_at
            if (
                row.measurement_class is not MeasurementClass.EXACT_SOURCE_RECORD
                or row.quantity is None
                or endpoint is None
                or episode.started_at < window.start.observed_at
                or endpoint > window.end.observed_at
            ):
                comparable = False
                continue
            total = _decimal_sum((total, row.quantity))
        covered_total = total if comparable else None
        difference = (
            _decimal_subtract(window.adjusted_consumption_usd, covered_total)
            if covered_total is not None
            else None
        )
        adjustment_note = (
            f" Known balance adjustments: {len(window.adjustments)}; raw balance "
            f"decrease={format(window.raw_balance_decrease_usd, 'f')}."
            if window.adjustments
            else " No separately evidenced balance adjustments were supplied."
        )
        row = LedgerRow(
            row_id=window.stable_id,
            ledger=LedgerName.METERED_CONSUMPTION,
            provider="runpod",
            category="account_balance_consumption_crosscheck",
            quantity=window.adjusted_consumption_usd,
            unit="USD credits",
            currency="USD",
            measurement_class=MeasurementClass.RECONSTRUCTED,
            accounting_role=AccountingRole.CROSSCHECK_NONADDITIVE,
            overlap_key=f"runpod:balance-window:{window.window_id}",
            window_start_utc=window.start.observed_at_utc,
            source_cutoff_utc=window.end.observed_at_utc,
            evidence_refs=window.evidence_refs,
            method_version=METHOD_VERSION,
            note=(
                "Balance movement is a nonadditive reconciliation equation, not a "
                "provider charge or cash transaction."
                + adjustment_note
            ),
        )
        results.append(
            BalanceReconciliation(
                window_id=window.window_id,
                row=row,
                raw_balance_decrease_usd=window.raw_balance_decrease_usd,
                adjusted_consumption_usd=window.adjusted_consumption_usd,
                covered_primary_usd=covered_total,
                adjusted_minus_primary_usd=difference,
                exactly_comparable=comparable,
            )
        )
    return tuple(results)


def _normalize_balance_windows(
    windows: Iterable[BalanceWindow],
) -> tuple[BalanceWindow, ...]:
    by_id: dict[str, BalanceWindow] = {}
    by_stable_id: dict[str, BalanceWindow] = {}
    for window in windows:
        prior = by_id.get(window.window_id)
        if prior is not None and prior != window:
            raise RunPodAmbiguityError(
                f"balance window {window.window_id!r} has conflicting evidence"
            )
        by_id[window.window_id] = window
        stable_prior = by_stable_id.get(window.stable_id)
        if stable_prior is None or window.window_id < stable_prior.window_id:
            by_stable_id[window.stable_id] = window
    normalized = tuple(sorted(by_stable_id.values(), key=lambda item: item.window_id))
    adjustment_ids: dict[str, str] = {}
    adjustment_bindings: dict[str, str] = {}
    for index, left in enumerate(normalized):
        for right in normalized[index + 1 :]:
            if (
                left.start.observed_at < right.end.observed_at
                and right.start.observed_at < left.end.observed_at
            ):
                raise RunPodAmbiguityError(
                    f"balance windows overlap: {left.window_id}/{right.window_id}"
                )
        for adjustment in left.adjustments:
            prior_window = adjustment_ids.get(adjustment.adjustment_id)
            if prior_window is not None:
                raise RunPodAmbiguityError(
                    f"balance adjustment ID reused in {prior_window}/{left.window_id}: "
                    f"{adjustment.adjustment_id}"
                )
            prior_adjustment = adjustment_bindings.get(
                adjustment.binding.evidence_ref
            )
            if prior_adjustment is not None:
                raise RunPodAmbiguityError(
                    "balance adjustment evidence binding reused: "
                    f"{prior_adjustment}/{adjustment.adjustment_id}"
                )
            adjustment_ids[adjustment.adjustment_id] = left.window_id
            adjustment_bindings[
                adjustment.binding.evidence_ref
            ] = adjustment.adjustment_id
    return normalized


def _exact_reconciliation_reasons(
    episodes: Sequence[EpisodeReconciliation],
    balances: Sequence[BalanceReconciliation],
    windows: Sequence[BalanceWindow],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not episodes:
        reasons.append("no project episodes")
    for episode in episodes:
        if episode.primary_row.measurement_class is not MeasurementClass.EXACT_SOURCE_RECORD:
            reasons.append(f"{episode.episode_id}: primary row is not exact provider settlement")
        if episode.rate_crosscheck_row is None:
            reasons.append(f"{episode.episode_id}: no creation-rate cross-check")
        elif episode.provider_minus_rate_usd != _ZERO:
            reasons.append(f"{episode.episode_id}: provider/rate amounts differ")
    if not balances:
        reasons.append("no balance reconciliation window")
    adjustment_owners: dict[str, str] = {}
    for index, left in enumerate(windows):
        for right in windows[index + 1 :]:
            if (
                left.start.observed_at < right.end.observed_at
                and right.start.observed_at < left.end.observed_at
            ):
                reasons.append(
                    f"balance windows overlap: {left.window_id}/{right.window_id}"
                )
        for adjustment in left.adjustments:
            owner = adjustment_owners.get(adjustment.adjustment_id)
            if owner is not None:
                reasons.append(
                    f"balance adjustment reused: {adjustment.adjustment_id} "
                    f"in {owner}/{left.window_id}"
                )
            adjustment_owners[adjustment.adjustment_id] = left.window_id
    covered: list[str] = []
    balance_by_id = {item.window_id: item for item in balances}
    for window in windows:
        result = balance_by_id[window.window_id]
        covered.extend(window.covered_pod_ids)
        if not result.exactly_comparable:
            reasons.append(f"{window.window_id}: balance window is not exclusive/exact")
        elif result.adjusted_minus_primary_usd != _ZERO:
            reasons.append(f"{window.window_id}: balance/provider amounts differ")
    episode_pods = sorted(item.pod_id for item in episodes)
    if sorted(covered) != episode_pods:
        reasons.append("balance windows do not cover each project pod exactly once")
    return tuple(sorted(set(reasons)))


def build_runpod_ledger(
    *,
    snapshots: Iterable[ProviderSnapshot],
    project_episodes: Iterable[LocalPodEpisode],
    balance_windows: Iterable[BalanceWindow] = (),
) -> RunPodLedger:
    """Build one additive project row per pod and only nonadditive cross-checks."""

    normalized_snapshots = _normalize_snapshots(snapshots)
    normalized_episodes = _normalize_episodes(project_episodes)
    normalized_windows = _normalize_balance_windows(balance_windows)
    histories = _provider_histories(normalized_snapshots)
    reconciliations: list[EpisodeReconciliation] = []
    for episode in normalized_episodes:
        estimate = reconstruct_local_rate_cost(episode)
        observations = histories.get(episode.pod_id, ())
        if (
            episode.owns_full_pod_lifetime
            and observations
            and observations[0].snapshot.captured_at < episode.started_at
        ):
            raise RunPodAmbiguityError(
                f"provider pod {episode.pod_id} predates its claimed project lifetime"
            )
        provider_usable = bool(observations) and episode.owns_full_pod_lifetime
        if provider_usable:
            settled = _is_settled(episode, observations)
            if settled:
                primary = _provider_primary_row(episode, observations)
                provider_crosscheck = None
            else:
                provider_crosscheck = _provider_lower_bound_crosscheck_row(
                    episode, observations
                )
                primary = _unknown_provider_primary_row(
                    episode,
                    estimate,
                    additional_evidence_refs=provider_crosscheck.evidence_refs,
                )
            rate_row = _rate_crosscheck_row(episode, estimate)
            provider_amount = observations[-1].total.amount_usd
        else:
            settled = False
            primary = _unknown_provider_primary_row(episode, estimate)
            provider_crosscheck = None
            rate_row = _rate_crosscheck_row(episode, estimate)
            provider_amount = (
                observations[-1].total.amount_usd if observations else None
            )
        difference = (
            _decimal_subtract(provider_amount, estimate.amount_usd)
            if provider_amount is not None and estimate.amount_usd is not None
            else None
        )
        reconciliations.append(
            EpisodeReconciliation(
                episode_id=episode.episode_id,
                pod_id=episode.pod_id,
                primary_row=primary,
                provider_crosscheck_row=provider_crosscheck,
                rate_crosscheck_row=rate_row,
                provider_amount_usd=provider_amount,
                provider_settled=settled,
                rate_amount_usd=estimate.amount_usd,
                provider_minus_rate_usd=difference,
            )
        )
    episode_rows = tuple(reconciliations)
    balance_rows = _balance_reconciliations(
        normalized_windows, normalized_episodes, episode_rows
    )
    project_ids = {episode.pod_id for episode in normalized_episodes}
    provider_ids = set(histories)
    reasons = _exact_reconciliation_reasons(
        episode_rows, balance_rows, normalized_windows
    )
    ledger = RunPodLedger(
        episodes=episode_rows,
        balance_reconciliations=balance_rows,
        account_lifetime_rows=_account_lifetime_rows(normalized_snapshots),
        snapshot_ids=tuple(snapshot.snapshot_id for snapshot in normalized_snapshots),
        unmatched_provider_pod_ids=tuple(sorted(provider_ids - project_ids)),
        unmatched_project_pod_ids=tuple(sorted(project_ids - provider_ids)),
        exact_reconciliation_reasons=reasons,
    )
    # Force structural duplicate checks even when the caller only serializes.
    ledger.project_additive_known_usd
    return ledger


def assert_exact_runpod_reconciliation(ledger: RunPodLedger) -> None:
    """Fail unless provider, local-rate, and exclusive balance evidence agree."""

    if ledger.exact_reconciliation_reasons:
        raise RunPodAccountingError(
            "RunPod exact reconciliation failed: "
            + "; ".join(ledger.exact_reconciliation_reasons)
        )
