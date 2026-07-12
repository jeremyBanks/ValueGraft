"""Fail-closed provider watchdog for the powered-v13 Stage-T canary.

The watchdog owns the provider clock after one Pod ID has been returned.  Its
record fixes the smaller of the literal 3,300-second cap and the conservative
rate-derived $1.50 cap.  A terminal cleanup requires both a per-Pod 404 and a
complete active-inventory observation that omits the Pod.  Harvest failure is
recorded but never delays deletion beyond the reserved cleanup lead.

This module deliberately contains no experiment, fixture, model, pool,
permutation, or analysis import.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import urllib.error

from powered_v13_schema import DESIGN_ID


SCHEMA = "coherent-state-powered-successor-v13-stage-t-watchdog-v1"
MAX_TOTAL_SPEND_USD = Decimal("1.50")
MAX_PROVIDER_SECONDS = 3300
DELETE_LEAD_SECONDS = 120
POLL_SECONDS = 10
JOB_PROBE_TIMEOUT_SECONDS = 15
HARVEST_TIMEOUT_SECONDS = 60
JOB_RUNNING_EXIT = 0
JOB_COMPLETE_EXIT = 20
JOB_DEAD_EXIT = 21
_SHA256 = re.compile(r"[0-9a-f]{64}")
_POD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,127}")
_TERMINAL_STATUSES = {
    "TERMINATED_HARVESTED",
    "TERMINATED_HARVEST_FAILED",
}


class V13WatchdogError(RuntimeError):
    """The Stage-T provider clock or cleanup evidence differs."""


class ProviderBackend(Protocol):
    def get_pod(self, pod_id: str) -> Mapping[str, Any]: ...

    def active_pod_ids(self) -> Sequence[str]: ...

    def delete_pod(self, pod_id: str) -> None: ...


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V13WatchdogError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise V13WatchdogError(f"value is not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None,
             f"{label} is not lowercase SHA-256")
    return value


def _plain_int(value: object, label: str, minimum: int,
               maximum: int) -> int:
    _require(type(value) is int and minimum <= value <= maximum,
             f"{label} lies outside {minimum}..{maximum}")
    return value


def _money(value: object, label: str, *, positive: bool = False) -> Decimal:
    _require(isinstance(value, str), f"{label} is not an exact decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise V13WatchdogError(f"{label} is not exact decimal") from exc
    _require(result.is_finite() and (result > 0 if positive else result >= 0),
             f"{label} is invalid")
    canonical = ("0" if result == 0 else format(result.normalize(), "f"))
    _require(canonical == value,
             f"{label} is not canonical fixed-point decimal")
    return result


def _command(value: object, label: str) -> list[str]:
    _require(isinstance(value, Sequence) and not isinstance(value, (str, bytes))
             and bool(value), f"{label} is empty")
    result = list(value)
    _require(all(isinstance(item, str) and bool(item) and "\x00" not in item
                 for item in result), f"{label} contains an invalid argument")
    return result


def _event(value: object) -> dict[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == {
        "epoch", "kind", "evidence",
    }, "watchdog event fields differ")
    epoch = _plain_int(value.get("epoch"), "event epoch", 1, (1 << 63) - 1)
    kind = value.get("kind")
    evidence = value.get("evidence")
    _require(isinstance(kind, str) and re.fullmatch(r"[A-Z0-9_]+", kind)
             is not None, "watchdog event kind differs")
    _require(isinstance(evidence, Mapping), "watchdog event evidence differs")
    canonical_json_bytes(evidence)
    return {"epoch": epoch, "kind": kind, "evidence": deepcopy(dict(evidence))}


def _bounded_duration_seconds(*, created_rate: Decimal,
                              prior_spend: Decimal) -> int:
    remaining = MAX_TOTAL_SPEND_USD - prior_spend
    _require(remaining > 0, "no Stage-T provider budget remains")
    seconds = int((remaining * Decimal(3600) / created_rate).to_integral_value(
        rounding=ROUND_FLOOR))
    result = min(MAX_PROVIDER_SECONDS, seconds)
    _require(result > DELETE_LEAD_SECONDS,
             "remaining Stage-T provider window cannot fund safe cleanup")
    _require(prior_spend + created_rate * Decimal(result) / Decimal(3600)
             <= MAX_TOTAL_SPEND_USD,
             "rate-derived provider deadline exceeds $1.50")
    return result


def build_record(
    *,
    pod_id: str,
    created_cost_per_hr_usd: str,
    prior_stage_t_spend_usd: str,
    provider_clock_started_epoch: int,
    pod_state_sha256: str,
    create_response_sha256: str,
    job_sha256: str,
    release_receipt_sha256: str,
    job_probe_command: Sequence[str],
    harvest_command: Sequence[str],
) -> dict[str, Any]:
    """Build one immutable-cap record immediately after allocation."""

    _require(isinstance(pod_id, str) and _POD_ID.fullmatch(pod_id) is not None,
             "pod ID is invalid")
    rate = _money(created_cost_per_hr_usd, "created rate", positive=True)
    prior = _money(prior_stage_t_spend_usd, "prior Stage-T spend")
    start = _plain_int(provider_clock_started_epoch, "provider clock start", 1,
                       (1 << 63) - 1)
    duration = _bounded_duration_seconds(
        created_rate=rate, prior_spend=prior)
    hard_deadline = start + duration
    record = {
        "schema": SCHEMA,
        "design_id": DESIGN_ID,
        "pod_id": pod_id,
        "created_cost_per_hr_usd": format(rate, "f"),
        "prior_stage_t_spend_usd": format(prior, "f"),
        "max_total_spend_usd": format(MAX_TOTAL_SPEND_USD, "f"),
        "max_provider_seconds": MAX_PROVIDER_SECONDS,
        "provider_clock_started_epoch": start,
        "bounded_provider_seconds": duration,
        "delete_trigger_epoch": hard_deadline - DELETE_LEAD_SECONDS,
        "hard_deadline_epoch": hard_deadline,
        "delete_lead_seconds": DELETE_LEAD_SECONDS,
        "pod_state_sha256": _sha(pod_state_sha256, "pod state hash"),
        "create_response_sha256": _sha(
            create_response_sha256, "create response hash"),
        "job_sha256": _sha(job_sha256, "job hash"),
        "release_receipt_sha256": _sha(
            release_receipt_sha256, "release receipt hash"),
        "job_probe_command": _command(job_probe_command, "job probe command"),
        "harvest_command": _command(harvest_command, "harvest command"),
        "status": "ALLOCATED",
        "termination_reason": None,
        "harvest": None,
        "per_pod_404_observed": False,
        "active_inventory_absent_observed": False,
        "events": [{
            "epoch": start,
            "kind": "ALLOCATION_REGISTERED",
            "evidence": {
                "created_cost_per_hr_usd": format(rate, "f"),
                "bounded_provider_seconds": duration,
            },
        }],
    }
    return validate_record(record)


def validate_record(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema", "design_id", "pod_id", "created_cost_per_hr_usd",
        "prior_stage_t_spend_usd", "max_total_spend_usd",
        "max_provider_seconds", "provider_clock_started_epoch",
        "bounded_provider_seconds", "delete_trigger_epoch",
        "hard_deadline_epoch", "delete_lead_seconds", "pod_state_sha256",
        "create_response_sha256", "job_sha256", "release_receipt_sha256",
        "job_probe_command", "harvest_command", "status",
        "termination_reason", "harvest", "per_pod_404_observed",
        "active_inventory_absent_observed", "events",
    }
    _require(isinstance(value, Mapping) and set(value) == fields,
             "watchdog record field set differs")
    _require(value.get("schema") == SCHEMA
             and value.get("design_id") == DESIGN_ID,
             "watchdog schema/design differs")
    pod_id = value.get("pod_id")
    _require(isinstance(pod_id, str) and _POD_ID.fullmatch(pod_id) is not None,
             "watchdog pod ID differs")
    rate = _money(value.get("created_cost_per_hr_usd"), "created rate",
                  positive=True)
    prior = _money(value.get("prior_stage_t_spend_usd"),
                   "prior Stage-T spend")
    _require(value.get("max_total_spend_usd") == "1.50"
             and value.get("max_provider_seconds") == MAX_PROVIDER_SECONDS
             and value.get("delete_lead_seconds") == DELETE_LEAD_SECONDS,
             "literal Stage-T caps differ")
    start = _plain_int(value.get("provider_clock_started_epoch"),
                       "provider clock start", 1, (1 << 63) - 1)
    duration = _plain_int(value.get("bounded_provider_seconds"),
                          "bounded provider seconds", 1,
                          MAX_PROVIDER_SECONDS)
    _require(duration == _bounded_duration_seconds(
        created_rate=rate, prior_spend=prior),
        "bounded provider duration does not recompute")
    _require(value.get("hard_deadline_epoch") == start + duration
             and value.get("delete_trigger_epoch") ==
             start + duration - DELETE_LEAD_SECONDS,
             "watchdog provider deadlines differ")
    for field in ("pod_state_sha256", "create_response_sha256", "job_sha256",
                  "release_receipt_sha256"):
        _sha(value.get(field), field)
    _command(value.get("job_probe_command"), "job probe command")
    _command(value.get("harvest_command"), "harvest command")
    status = value.get("status")
    _require(status in {
        "ALLOCATED", "WATCHING", "TERMINATING", *_TERMINAL_STATUSES,
    }, "watchdog status differs")
    reason = value.get("termination_reason")
    _require(reason is None or (isinstance(reason, str)
                                and re.fullmatch(r"[a-z0-9_]+", reason)),
             "watchdog termination reason differs")
    harvest = value.get("harvest")
    if harvest is not None:
        _require(isinstance(harvest, Mapping) and set(harvest) == {
            "attempted_epoch", "returncode", "stdout_sha256",
            "stderr_sha256", "timed_out", "succeeded",
        }, "watchdog harvest evidence fields differ")
        _plain_int(harvest.get("attempted_epoch"), "harvest epoch", 1,
                   (1 << 63) - 1)
        _require(type(harvest.get("returncode")) is int,
                 "harvest return code differs")
        _sha(harvest.get("stdout_sha256"), "harvest stdout hash")
        _sha(harvest.get("stderr_sha256"), "harvest stderr hash")
        _require(type(harvest.get("timed_out")) is bool
                 and type(harvest.get("succeeded")) is bool
                 and harvest["succeeded"] == (
                     not harvest["timed_out"] and harvest["returncode"] == 0),
                 "harvest success evidence differs")
    for field in ("per_pod_404_observed",
                  "active_inventory_absent_observed"):
        _require(type(value.get(field)) is bool, f"{field} differs")
    events = value.get("events")
    _require(isinstance(events, list) and bool(events),
             "watchdog event journal is empty")
    frozen_events = [_event(row) for row in events]
    _require([row["epoch"] for row in frozen_events] == sorted(
        row["epoch"] for row in frozen_events),
        "watchdog event epochs go backwards")
    if status in _TERMINAL_STATUSES:
        _require(value["per_pod_404_observed"] is True
                 and value["active_inventory_absent_observed"] is True
                 and harvest is not None and reason is not None,
                 "terminal watchdog record lacks harvest/deletion evidence")
        _require((status == "TERMINATED_HARVESTED") == harvest["succeeded"],
                 "terminal watchdog status/harvest evidence differ")
    return deepcopy(dict(value))


def write_record_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    frozen = validate_record(value)
    raw = canonical_json_bytes(frozen) + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def read_record(path: Path) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except Exception as exc:
        raise V13WatchdogError(f"cannot read watchdog record: {exc}") from exc
    return validate_record(value)


def replace_record(path: Path, value: Mapping[str, Any]) -> None:
    frozen = validate_record(value)
    raw = canonical_json_bytes(frozen) + b"\n"
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _append_event(record: Mapping[str, Any], *, epoch: int, kind: str,
                  evidence: Mapping[str, Any] | None = None,
                  **changes: Any) -> dict[str, Any]:
    result = validate_record(record)
    result["events"].append({
        "epoch": epoch,
        "kind": kind,
        "evidence": dict(evidence or {}),
    })
    result.update(changes)
    return validate_record(result)


def probe_job(command: Sequence[str], *,
              run: Callable[..., subprocess.CompletedProcess[bytes]] =
              subprocess.run) -> str:
    """Return RUNNING/COMPLETE/DEAD/AMBIGUOUS from the frozen probe exits."""

    argv = _command(command, "job probe command")
    try:
        completed = run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=JOB_PROBE_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return "AMBIGUOUS"
    return {
        JOB_RUNNING_EXIT: "RUNNING",
        JOB_COMPLETE_EXIT: "COMPLETE",
        JOB_DEAD_EXIT: "DEAD",
    }.get(completed.returncode, "AMBIGUOUS")


def guard_decision(
    *,
    record: Mapping[str, Any],
    now_epoch: int,
    job_state: str,
    provider_pod: Mapping[str, Any] | None,
    provider_error: BaseException | None,
) -> str:
    """Pure pre-cleanup decision; uncertainty never authorizes another Pod."""

    frozen = validate_record(record)
    _plain_int(now_epoch, "decision epoch", 1, (1 << 63) - 1)
    _require(job_state in {"RUNNING", "COMPLETE", "DEAD", "AMBIGUOUS"},
             "job state differs")
    if now_epoch >= frozen["delete_trigger_epoch"]:
        return "STOP_DEADLINE"
    if provider_error is not None:
        if (isinstance(provider_error, urllib.error.HTTPError)
                and provider_error.code == 404):
            return "STOP_PROVIDER_GONE"
        return "HOLD_PROVIDER_AMBIGUOUS"
    _require(isinstance(provider_pod, Mapping),
             "provider observation is absent without an error")
    if provider_pod.get("id") != frozen["pod_id"]:
        return "STOP_PROVIDER_IDENTITY"
    try:
        observed_rate = _money(
            str(provider_pod.get("costPerHr")), "observed provider rate",
            positive=True)
    except V13WatchdogError:
        return "HOLD_PROVIDER_AMBIGUOUS"
    if observed_rate > Decimal(frozen["created_cost_per_hr_usd"]):
        return "STOP_RATE_INCREASE"
    if job_state == "COMPLETE":
        return "STOP_JOB_COMPLETE"
    if job_state == "DEAD":
        return "STOP_PROCESS_DEATH"
    if job_state == "AMBIGUOUS":
        return "HOLD_JOB_AMBIGUOUS"
    return "CONTINUE"


def run_harvest(
    command: Sequence[str], *, epoch: int,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, Any]:
    argv = _command(command, "harvest command")
    timed_out = False
    try:
        completed = run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=HARVEST_TIMEOUT_SECONDS, check=False)
        returncode = completed.returncode
        stdout = completed.stdout or b""
        stderr = completed.stderr or b""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
    except OSError as exc:
        returncode = 127
        stdout = b""
        stderr = f"{type(exc).__name__}: {exc}".encode("utf-8", "replace")
    return {
        "attempted_epoch": epoch,
        "returncode": returncode,
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "timed_out": timed_out,
        "succeeded": not timed_out and returncode == 0,
    }


def _is_http_404(value: BaseException) -> bool:
    return isinstance(value, urllib.error.HTTPError) and value.code == 404


def cleanup_once(
    *,
    backend: ProviderBackend,
    record: Mapping[str, Any],
    now_epoch: int,
) -> tuple[dict[str, Any], bool]:
    """Issue deletion once and require independent GET/inventory agreement."""

    frozen = validate_record(record)
    pod_id = frozen["pod_id"]
    delete_evidence: dict[str, Any] = {}
    try:
        backend.delete_pod(pod_id)
        delete_evidence["delete"] = "accepted"
    except BaseException as exc:
        delete_evidence["delete"] = (
            "404" if _is_http_404(exc) else type(exc).__name__)

    per_pod_404 = False
    try:
        observed = backend.get_pod(pod_id)
        delete_evidence["get"] = (
            "same_pod" if observed.get("id") == pod_id else "different_pod")
    except BaseException as exc:
        per_pod_404 = _is_http_404(exc)
        delete_evidence["get"] = (
            "404" if per_pod_404 else type(exc).__name__)

    inventory_absent = False
    try:
        ids = list(backend.active_pod_ids())
        _require(all(isinstance(value, str) and value for value in ids)
                 and len(ids) == len(set(ids)),
                 "active Pod inventory is invalid")
        inventory_absent = pod_id not in ids
        delete_evidence["active_inventory_count"] = len(ids)
        delete_evidence["active_inventory_absent"] = inventory_absent
    except BaseException as exc:
        delete_evidence["inventory_error"] = type(exc).__name__

    result = _append_event(
        frozen, epoch=now_epoch, kind="DELETE_VERIFICATION",
        evidence=delete_evidence,
        per_pod_404_observed=(
            frozen["per_pod_404_observed"] or per_pod_404),
        active_inventory_absent_observed=(
            frozen["active_inventory_absent_observed"] or inventory_absent),
    )
    confirmed = bool(per_pod_404 and inventory_absent)
    if confirmed:
        status = ("TERMINATED_HARVESTED"
                  if result["harvest"]["succeeded"]
                  else "TERMINATED_HARVEST_FAILED")
        result = _append_event(
            result, epoch=now_epoch, kind="POD_CONFIRMED_ABSENT",
            evidence={"per_pod_404": True, "active_inventory_absent": True},
            status=status,
            per_pod_404_observed=True,
            active_inventory_absent_observed=True,
        )
    return result, confirmed


def watch(
    *,
    record_path: Path,
    backend: ProviderBackend,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    probe: Callable[[Sequence[str]], str] = probe_job,
    harvest: Callable[[Sequence[str], int], Mapping[str, Any]] | None = None,
    max_cycles: int | None = None,
) -> dict[str, Any]:
    """Run until dual deletion confirmation; ``max_cycles`` is test-only."""

    record = read_record(record_path)
    _require(record["status"] not in _TERMINAL_STATUSES,
             "watchdog record is already terminal")
    now = int(clock())
    record = _append_event(
        record, epoch=now, kind="WATCHDOG_STARTED",
        status="WATCHING")
    replace_record(record_path, record)
    cycles = 0
    while True:
        cycles += 1
        now = int(clock())
        provider_pod: Mapping[str, Any] | None = None
        provider_error: BaseException | None = None
        try:
            provider_pod = backend.get_pod(record["pod_id"])
        except BaseException as exc:
            provider_error = exc
        job_state = probe(record["job_probe_command"])
        action = guard_decision(
            record=record, now_epoch=now, job_state=job_state,
            provider_pod=provider_pod, provider_error=provider_error)
        record = _append_event(
            record, epoch=now, kind="WATCHDOG_OBSERVATION",
            evidence={"action": action, "job_state": job_state})
        replace_record(record_path, record)
        if action.startswith("STOP_"):
            reason = action.removeprefix("STOP_").lower()
            if record["harvest"] is None:
                observed_harvest = (
                    dict(harvest(record["harvest_command"], now))
                    if harvest is not None
                    else run_harvest(record["harvest_command"], epoch=now))
                record = _append_event(
                    record, epoch=now, kind="HARVEST_FINISHED",
                    evidence={"succeeded": observed_harvest.get("succeeded")},
                    status="TERMINATING", termination_reason=reason,
                    harvest=observed_harvest)
                replace_record(record_path, record)
            record, confirmed = cleanup_once(
                backend=backend, record=record, now_epoch=int(clock()))
            replace_record(record_path, record)
            if confirmed:
                return record
        if max_cycles is not None and cycles >= max_cycles:
            return record
        remaining = record["delete_trigger_epoch"] - int(clock())
        sleep(max(1, min(POLL_SECONDS, remaining if remaining > 0 else 1)))


__all__ = [
    "DELETE_LEAD_SECONDS",
    "HARVEST_TIMEOUT_SECONDS",
    "JOB_COMPLETE_EXIT",
    "JOB_DEAD_EXIT",
    "JOB_RUNNING_EXIT",
    "MAX_PROVIDER_SECONDS",
    "MAX_TOTAL_SPEND_USD",
    "ProviderBackend",
    "SCHEMA",
    "V13WatchdogError",
    "build_record",
    "canonical_json_bytes",
    "cleanup_once",
    "guard_decision",
    "probe_job",
    "read_record",
    "run_harvest",
    "validate_record",
    "watch",
    "write_record_exclusive",
]
