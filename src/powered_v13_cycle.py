"""Fail-closed cycle accounting for powered successor v13.

The module is intentionally disconnected from provider and git APIs.  A
caller must supply every observation literally.  The only work done here is
validation, exact decimal arithmetic, and derivation of the frozen Phase-A and
future-runway inequalities.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable


SCHEMA = "coherent_state_powered_v13_cycle_checkpoint_v1"
DESIGN_ID = "coherent-state-powered-successor-v13"
MAX_PROVIDER_OBSERVATION_AGE_SECONDS = 300
_FULL_GIT_SHA = re.compile(r"[0-9a-f]{40}")


class V13CycleError(ValueError):
    """An input cannot support an honest v13 cycle checkpoint."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V13CycleError(message)


def parse_utc(value: object, label: str) -> datetime:
    """Parse a literal RFC-3339 UTC timestamp.

    Requiring ``Z`` avoids accepting a local time whose displayed offset could
    later be mistaken for the provider clock.
    """
    _require(isinstance(value, str) and value.endswith("Z"),
             f"{label} must be an RFC-3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise V13CycleError(f"{label} is not a valid timestamp") from exc
    _require(parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed),
             f"{label} is not UTC")
    return parsed


def parse_money(value: object, label: str) -> Decimal:
    """Parse a nonnegative finite dollar amount without binary-float input."""
    _require(isinstance(value, (str, Decimal)) and not isinstance(value, bool),
             f"{label} must be supplied as a decimal string")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise V13CycleError(f"{label} is not a decimal amount") from exc
    _require(parsed.is_finite() and parsed >= 0,
             f"{label} must be finite and nonnegative")
    return parsed


def _canonical_money(value: Decimal) -> str:
    return format(value, "f")


def _literal_nonnegative_int(value: object, label: str) -> int:
    _require(not isinstance(value, bool) and isinstance(value, int),
             f"{label} must be an integer")
    _require(value >= 0, f"{label} must be nonnegative")
    return value


def _literal_strings(values: object, label: str) -> tuple[str, ...]:
    _require(isinstance(values, (tuple, list)), f"{label} must be a list")
    result = tuple(values)
    _require(all(isinstance(value, str) and value.strip() == value and value
                 for value in result),
             f"{label} must contain nonempty, trimmed strings")
    _require(len(result) == len(set(result)), f"{label} contains duplicates")
    return result


@dataclass(frozen=True)
class CycleObservation:
    checkpoint_utc: str
    goal_elapsed_seconds: int
    provider_observed_utc: str
    provider_balance_usd: str | Decimal
    provider_spend_limit_usd: str | Decimal
    active_pod_ids: tuple[str, ...]
    phase: str
    status: str
    phase_spent_usd: str | Decimal
    phase_cap_usd: str | Decimal
    projected_remaining_phase_a_usd: str | Decimal
    projected_core_usd: str | Decimal
    projected_audit_usd: str | Decimal
    reserve_usd: str | Decimal
    independent_n: int
    target_n: int
    independent_unit_ids: tuple[str, ...]
    git_commit: str
    git_clean: bool
    completed_gates: tuple[str, ...]
    pending_gates: tuple[str, ...]

    def validate(self) -> "ValidatedCycleObservation":
        checkpoint_time = parse_utc(self.checkpoint_utc, "checkpoint_utc")
        provider_time = parse_utc(
            self.provider_observed_utc, "provider_observed_utc")
        _require(provider_time <= checkpoint_time,
                 "provider observation cannot be later than checkpoint")

        elapsed = _literal_nonnegative_int(
            self.goal_elapsed_seconds, "goal_elapsed_seconds")
        independent_n = _literal_nonnegative_int(
            self.independent_n, "independent_n")
        target_n = _literal_nonnegative_int(self.target_n, "target_n")
        _require(target_n > 0, "target_n must be positive")
        _require(independent_n <= target_n,
                 "independent_n cannot exceed the frozen target_n")

        unit_ids = _literal_strings(
            self.independent_unit_ids, "independent_unit_ids")
        _require(len(unit_ids) == independent_n,
                 "independent_n differs from the unique semantic-unit IDs")
        pods = _literal_strings(self.active_pod_ids, "active_pod_ids")
        completed = _literal_strings(self.completed_gates, "completed_gates")
        pending = _literal_strings(self.pending_gates, "pending_gates")
        _require(not set(completed).intersection(pending),
                 "completed and pending gates overlap")
        _require(bool(completed or pending),
                 "at least one completed or pending gate is required")

        _require(isinstance(self.phase, str) and self.phase.strip() == self.phase
                 and bool(self.phase), "phase must be a nonempty trimmed string")
        _require(isinstance(self.status, str) and self.status.strip() == self.status
                 and bool(self.status), "status must be a nonempty trimmed string")
        _require(isinstance(self.git_commit, str)
                 and _FULL_GIT_SHA.fullmatch(self.git_commit) is not None,
                 "git_commit must be a full lowercase 40-hex commit")
        _require(isinstance(self.git_clean, bool),
                 "git_clean must be a literal boolean")

        return ValidatedCycleObservation(
            checkpoint_time=checkpoint_time,
            checkpoint_utc=self.checkpoint_utc,
            goal_elapsed_seconds=elapsed,
            provider_time=provider_time,
            provider_observed_utc=self.provider_observed_utc,
            provider_balance_usd=parse_money(
                self.provider_balance_usd, "provider_balance_usd"),
            provider_spend_limit_usd=parse_money(
                self.provider_spend_limit_usd, "provider_spend_limit_usd"),
            active_pod_ids=pods,
            phase=self.phase,
            status=self.status,
            phase_spent_usd=parse_money(
                self.phase_spent_usd, "phase_spent_usd"),
            phase_cap_usd=parse_money(self.phase_cap_usd, "phase_cap_usd"),
            projected_remaining_phase_a_usd=parse_money(
                self.projected_remaining_phase_a_usd,
                "projected_remaining_phase_a_usd"),
            projected_core_usd=parse_money(
                self.projected_core_usd, "projected_core_usd"),
            projected_audit_usd=parse_money(
                self.projected_audit_usd, "projected_audit_usd"),
            reserve_usd=parse_money(self.reserve_usd, "reserve_usd"),
            independent_n=independent_n,
            target_n=target_n,
            independent_unit_ids=unit_ids,
            git_commit=self.git_commit,
            git_clean=self.git_clean,
            completed_gates=completed,
            pending_gates=pending,
        )


@dataclass(frozen=True)
class ValidatedCycleObservation:
    checkpoint_time: datetime
    checkpoint_utc: str
    goal_elapsed_seconds: int
    provider_time: datetime
    provider_observed_utc: str
    provider_balance_usd: Decimal
    provider_spend_limit_usd: Decimal
    active_pod_ids: tuple[str, ...]
    phase: str
    status: str
    phase_spent_usd: Decimal
    phase_cap_usd: Decimal
    projected_remaining_phase_a_usd: Decimal
    projected_core_usd: Decimal
    projected_audit_usd: Decimal
    reserve_usd: Decimal
    independent_n: int
    target_n: int
    independent_unit_ids: tuple[str, ...]
    git_commit: str
    git_clean: bool
    completed_gates: tuple[str, ...]
    pending_gates: tuple[str, ...]


def build_checkpoint(observation: CycleObservation) -> dict[str, Any]:
    """Validate literal observations and build one deterministic checkpoint."""
    value = observation.validate()
    provider_age = (value.checkpoint_time - value.provider_time).total_seconds()
    _require(math.isfinite(provider_age) and provider_age >= 0
             and provider_age.is_integer(),
             "provider observation age must resolve to whole nonnegative seconds")
    provider_age_seconds = int(provider_age)
    provider_fresh = provider_age_seconds <= MAX_PROVIDER_OBSERVATION_AGE_SECONDS

    phase_lhs = value.phase_spent_usd + value.projected_remaining_phase_a_usd
    phase_pass = phase_lhs <= value.phase_cap_usd

    # Deliberately exclude phase_spent_usd: provider_balance_usd is the current
    # unspent balance, so adding already spent dollars would double-count them.
    future_lhs = (
        value.projected_remaining_phase_a_usd
        + value.projected_core_usd
        + value.projected_audit_usd
        + value.reserve_usd
    )
    future_pass = future_lhs <= value.provider_balance_usd
    named_gates_complete = len(value.pending_gates) == 0

    hold_reasons: list[str] = []
    if not provider_fresh:
        hold_reasons.append("STALE_PROVIDER_OBSERVATION")
    if not phase_pass:
        hold_reasons.append("PHASE_CAP_FAILED")
    if not future_pass:
        hold_reasons.append("FUTURE_BALANCE_FAILED")
    if not value.git_clean:
        hold_reasons.append("DIRTY_GIT_TREE")
    if not named_gates_complete:
        hold_reasons.append("PENDING_GATES")

    semantic_gap = value.target_n - value.independent_n
    total_gates = len(value.completed_gates) + len(value.pending_gates)
    return {
        "schema": SCHEMA,
        "design_id": DESIGN_ID,
        "scope": "cycle observation only; not treatment authorization",
        "observation": {
            "checkpoint_utc": value.checkpoint_utc,
            "goal_elapsed_seconds": value.goal_elapsed_seconds,
            "provider_observed_utc": value.provider_observed_utc,
            "provider_balance_usd": _canonical_money(
                value.provider_balance_usd),
            "provider_spend_limit_usd": _canonical_money(
                value.provider_spend_limit_usd),
            "active_pod_ids": list(value.active_pod_ids),
            "phase": value.phase,
            "status": value.status,
            "phase_spent_usd": _canonical_money(value.phase_spent_usd),
            "phase_cap_usd": _canonical_money(value.phase_cap_usd),
            "projected_remaining_phase_a_usd": _canonical_money(
                value.projected_remaining_phase_a_usd),
            "projected_core_usd": _canonical_money(
                value.projected_core_usd),
            "projected_audit_usd": _canonical_money(
                value.projected_audit_usd),
            "reserve_usd": _canonical_money(value.reserve_usd),
            "independent_n": value.independent_n,
            "target_n": value.target_n,
            "independent_unit_ids": list(value.independent_unit_ids),
            "git_commit": value.git_commit,
            "git_clean": value.git_clean,
            "completed_gates": list(value.completed_gates),
            "pending_gates": list(value.pending_gates),
        },
        "derived": {
            "goal_elapsed_hours": _canonical_money(
                Decimal(value.goal_elapsed_seconds) / Decimal(3600)),
            "provider_observation_age_seconds": provider_age_seconds,
            "provider_observation_max_age_seconds": (
                MAX_PROVIDER_OBSERVATION_AGE_SECONDS),
            "provider_observation_fresh": provider_fresh,
            "semantic_gap_n": semantic_gap,
            "semantic_collection_complete": semantic_gap == 0,
            "semantic_progress": f"{value.independent_n}/{value.target_n}",
            "completed_gate_count": len(value.completed_gates),
            "pending_gate_count": len(value.pending_gates),
            "total_gate_count": total_gates,
            "named_gates_complete": named_gates_complete,
        },
        "inequalities": {
            "phase_cap": {
                "formula": (
                    "phase_spent_usd + projected_remaining_phase_a_usd "
                    "<= phase_cap_usd"),
                "lhs_usd": _canonical_money(phase_lhs),
                "rhs_usd": _canonical_money(value.phase_cap_usd),
                "slack_usd": _canonical_money(value.phase_cap_usd - phase_lhs),
                "pass": phase_pass,
            },
            "future_balance": {
                "formula": (
                    "projected_remaining_phase_a_usd + projected_core_usd + "
                    "projected_audit_usd + reserve_usd <= "
                    "provider_balance_usd"),
                "included_terms": [
                    "projected_remaining_phase_a_usd",
                    "projected_core_usd",
                    "projected_audit_usd",
                    "reserve_usd",
                ],
                "excluded_already_spent_term": "phase_spent_usd",
                "lhs_usd": _canonical_money(future_lhs),
                "rhs_usd": _canonical_money(value.provider_balance_usd),
                "slack_usd": _canonical_money(
                    value.provider_balance_usd - future_lhs),
                "pass": future_pass,
            },
        },
        "assessment": {
            "status": "PASS" if not hold_reasons else "HOLD",
            "hold_reasons": hold_reasons,
            "paid_progression_gate_pass": not hold_reasons,
        },
    }


def write_checkpoint_exclusive(path: Path, checkpoint: dict[str, Any]) -> None:
    """Write one complete JSON document without replacing any existing path."""
    payload = (json.dumps(checkpoint, indent=2, sort_keys=True,
                          ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A failed write must not leave a file that looks like a valid
        # checkpoint and cannot be safely retried under exclusive-create.
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def parse_json_string_list(value: str, label: str) -> tuple[str, ...]:
    """Parse an explicit JSON list used by the CLI's list-valued arguments."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise V13CycleError(f"{label} is not valid JSON") from exc
    return _literal_strings(parsed, label)

