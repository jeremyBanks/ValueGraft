#!/usr/bin/env python3
"""Own and enforce the provider clock for one precreated treatment pod."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping
import urllib.error


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "coherent_canary_v12_treatment_provider_guard_v1"
MAX_RATE_USD = 1.39
MAX_RENTAL_SECONDS = 2300
# Start deletion early enough for one ordinary 30-second provider API timeout
# before the literal 2,300-second hard deadline.
DELETE_LEAD_SECONDS = 120
POLL_SECONDS = 15


class GuardError(RuntimeError):
    """Provider allocation, clock, rate, or deletion differed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GuardError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except Exception as exc:
        raise GuardError(f"cannot parse {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def atomic_write(path: Path, value: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if exclusive:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cost(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label} is not numeric")
    result = float(value)
    require(math.isfinite(result) and result > 0, f"{label} is invalid")
    return result


def build_record(*, name: str, state_path: Path, response_path: Path,
                 job_path: Path, record_path: Path, start_epoch: int,
                 max_rate: float = MAX_RATE_USD,
                 max_seconds: int = MAX_RENTAL_SECONDS) -> dict[str, Any]:
    require(name and all(character.isalnum() or character in "_-" for character in name),
            "pod name is unsafe")
    require(not record_path.exists(), "provider guard record already exists")
    require(job_path.is_file(), "treatment job is absent")
    state = load_object(state_path)
    response = load_object(response_path)
    pod_id = state.get("id")
    require(isinstance(pod_id, str) and pod_id, "state lacks a pod ID")
    require(response.get("id") == pod_id, "create response/state pod IDs differ")
    state_rate = cost(state.get("costPerHr"), "state costPerHr")
    response_rate = cost(response.get("costPerHr"), "response costPerHr")
    require(math.isclose(state_rate, response_rate, rel_tol=0, abs_tol=1e-12),
            "create response/state rates differ")
    require(math.isclose(max_rate, MAX_RATE_USD, rel_tol=0, abs_tol=1e-12),
            "maximum provider rate must remain $1.39/hour")
    require(max_seconds == MAX_RENTAL_SECONDS,
            "hard provider-clock duration must remain 2,300 seconds")
    require(isinstance(start_epoch, int) and start_epoch > 0,
            "provider-clock start epoch differs")
    hard_deadline = start_epoch + max_seconds
    record = {
        "schema": SCHEMA,
        "name": name,
        "pod_id": pod_id,
        "state_path": str(state_path.resolve()),
        "create_response_path": str(response_path.resolve()),
        "job_path": str(job_path.resolve()),
        "job_sha256": sha256(job_path),
        "created_cost_per_hr": state_rate,
        "max_cost_per_hr": max_rate,
        "provider_clock_started_epoch": start_epoch,
        "provider_clock_started_utc": datetime.fromtimestamp(
            start_epoch, timezone.utc).isoformat(),
        "max_rental_seconds": max_seconds,
        "delete_trigger_epoch": hard_deadline - DELETE_LEAD_SECONDS,
        "hard_deadline_epoch": hard_deadline,
        "hard_deadline_utc": datetime.fromtimestamp(
            hard_deadline, timezone.utc).isoformat(),
        "status": "ALLOCATED",
        "watchdog_pid": None,
        "last_provider_observation": None,
        "terminal_reason": None,
        "events": [{
            "at_utc": utc_now(), "event": "ALLOCATION_REGISTERED",
            "cost_per_hr": state_rate,
        }],
    }
    atomic_write(record_path, record, exclusive=True)
    require(state_rate <= max_rate,
            f"returned costPerHr ${state_rate} exceeds ${max_rate}")
    return record


def load_record(path: Path) -> dict[str, Any]:
    record = load_object(path)
    require(record.get("schema") == SCHEMA and
            isinstance(record.get("pod_id"), str) and record["pod_id"] and
            record.get("max_rental_seconds") == MAX_RENTAL_SECONDS and
            record.get("max_cost_per_hr") == MAX_RATE_USD and
            record.get("hard_deadline_epoch") ==
            record.get("provider_clock_started_epoch") + MAX_RENTAL_SECONDS and
            record.get("delete_trigger_epoch") ==
            record.get("hard_deadline_epoch") - DELETE_LEAD_SECONDS,
            "provider guard record differs")
    return record


def update_record(path: Path, **changes: Any) -> dict[str, Any]:
    record = load_record(path)
    record.update(changes)
    atomic_write(path, record)
    return record


def append_event(path: Path, event: str, **evidence: Any) -> dict[str, Any]:
    record = load_record(path)
    events = record.get("events")
    require(isinstance(events, list), "provider guard events differ")
    events.append({"at_utc": utc_now(), "event": event, **evidence})
    record["events"] = events
    atomic_write(path, record)
    return record


def decision(*, now_epoch: int, record: Mapping[str, Any],
             provider_pod: Mapping[str, Any] | None,
             provider_error: Exception | None = None) -> str:
    """Pure pre-deadline policy: ambiguity preserves; deadline/rate deletes."""
    if now_epoch >= int(record["delete_trigger_epoch"]):
        return "DELETE_DEADLINE"
    if provider_error is not None:
        if isinstance(provider_error, urllib.error.HTTPError) and \
                provider_error.code == 404:
            return "GONE"
        return "PRESERVE_AMBIGUOUS"
    require(isinstance(provider_pod, Mapping), "provider observation is absent")
    observed_rate = provider_pod.get("costPerHr")
    try:
        numeric_rate = cost(observed_rate, "observed costPerHr")
    except GuardError:
        return "PRESERVE_AMBIGUOUS"
    if numeric_rate > float(record["max_cost_per_hr"]):
        return "DELETE_RATE"
    return "PRESERVE_ACTIVE"


def pod_module(state_path: Path):
    os.chdir(ROOT)
    os.environ["SC_POD_STATE"] = str(state_path)
    os.environ["SC_POD_API_TIMEOUT_S"] = "10"
    sys.path.insert(0, str(ROOT / "src"))
    if "pod" in sys.modules:
        module = importlib.reload(sys.modules["pod"])
    else:
        module = importlib.import_module("pod")
    return module


def provider_get(module, pod_id: str) -> dict[str, Any]:
    value = module.api("GET", f"/pods/{pod_id}")
    require(isinstance(value, dict) and value.get("id") == pod_id,
            "provider returned a different pod")
    return value


def mark_gone(record_path: Path, reason: str) -> None:
    record = append_event(record_path, "POD_CONFIRMED_GONE", reason=reason)
    record.update({"status": "TERMINATED", "terminal_reason": reason,
                   "terminated_at_utc": utc_now()})
    atomic_write(record_path, record)


def terminate_until_gone(*, module, pod_id: str, reason: str,
                         record_path: Path | None = None) -> None:
    if record_path is not None:
        append_event(record_path, "TERMINATION_STARTED", reason=reason)
        update_record(record_path, status="TERMINATING", terminal_reason=reason)
    while True:
        try:
            module.api("DELETE", f"/pods/{pod_id}")
            print(f"DELETE issued pod={pod_id} reason={reason}", flush=True)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                if record_path is not None:
                    mark_gone(record_path, reason)
                print(json.dumps({"pod_id": pod_id, "status": "GONE",
                                  "reason": reason}, sort_keys=True), flush=True)
                return
            print(f"DELETE API ERROR {exc.code}; retrying", flush=True)
        except Exception as exc:
            print(f"DELETE ERROR {type(exc).__name__}: {exc}; retrying", flush=True)
        time.sleep(2)
        try:
            provider_get(module, pod_id)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                if record_path is not None:
                    mark_gone(record_path, reason)
                print(json.dumps({"pod_id": pod_id, "status": "GONE",
                                  "reason": reason}, sort_keys=True), flush=True)
                return
            print(f"DELETE VERIFY API ERROR {exc.code}; retrying", flush=True)
        except Exception as exc:
            print(f"DELETE VERIFY ERROR {type(exc).__name__}: {exc}; retrying",
                  flush=True)
        time.sleep(8)


def command_init(args: argparse.Namespace) -> None:
    record = build_record(
        name=args.name, state_path=args.state, response_path=args.response,
        job_path=args.job, record_path=args.record,
        start_epoch=args.start_epoch)
    print(json.dumps({
        "pod_id": record["pod_id"],
        "cost_per_hr": record["created_cost_per_hr"],
        "delete_trigger_epoch": record["delete_trigger_epoch"],
        "hard_deadline_epoch": record["hard_deadline_epoch"],
        "record": str(args.record),
    }, sort_keys=True))


def command_watch(args: argparse.Namespace) -> None:
    record = update_record(
        args.record, status="WATCHING", watchdog_pid=os.getpid(),
        watchdog_started_at_utc=utc_now())
    append_event(args.record, "WATCHDOG_STARTED", pid=os.getpid())
    module = pod_module(Path(record["state_path"]))
    while True:
        record = load_record(args.record)
        now = int(time.time())
        if now >= int(record["delete_trigger_epoch"]):
            terminate_until_gone(
                module=module, pod_id=record["pod_id"],
                reason="hard_provider_clock_deadline", record_path=args.record)
            return
        provider_pod = None
        provider_error = None
        try:
            provider_pod = provider_get(module, record["pod_id"])
        except Exception as exc:
            provider_error = exc
        now = int(time.time())
        action = decision(
            now_epoch=now, record=record, provider_pod=provider_pod,
            provider_error=provider_error)
        if action == "GONE":
            mark_gone(args.record, "provider_404_before_deadline")
            return
        if action in {"DELETE_DEADLINE", "DELETE_RATE"}:
            reason = ("hard_provider_clock_deadline" if action == "DELETE_DEADLINE"
                      else "provider_rate_exceeded")
            terminate_until_gone(
                module=module, pod_id=record["pod_id"], reason=reason,
                record_path=args.record)
            return
        observation = {
            "at_utc": utc_now(), "action": action,
            "seconds_until_delete_trigger": max(
                0, int(record["delete_trigger_epoch"]) - now),
        }
        if provider_pod is not None:
            observation.update({
                "desiredStatus": provider_pod.get("desiredStatus"),
                "costPerHr": provider_pod.get("costPerHr"),
                "publicIp_present": bool(provider_pod.get("publicIp")),
            })
        else:
            observation["error"] = (
                f"{type(provider_error).__name__}: {provider_error}")
        update_record(args.record, last_provider_observation=observation)
        print(json.dumps(observation, sort_keys=True), flush=True)
        remaining = int(record["delete_trigger_epoch"]) - int(time.time())
        time.sleep(max(1, min(POLL_SECONDS, remaining)))


def command_status(args: argparse.Namespace) -> None:
    record = load_record(args.record)
    module = pod_module(Path(record["state_path"]))
    provider: dict[str, Any]
    try:
        value = provider_get(module, record["pod_id"])
        provider = {key: value.get(key) for key in (
            "id", "desiredStatus", "lastStatusChange", "publicIp", "costPerHr")}
    except urllib.error.HTTPError as exc:
        provider = {"http_error": exc.code}
    result = {
        "record": record,
        "provider": provider,
        "now_epoch": int(time.time()),
        "seconds_until_delete_trigger": (
            int(record["delete_trigger_epoch"]) - int(time.time())),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def command_terminate(args: argparse.Namespace) -> None:
    record = load_record(args.record)
    module = pod_module(Path(record["state_path"]))
    terminate_until_gone(
        module=module, pod_id=record["pod_id"], reason=args.reason,
        record_path=args.record)


def command_terminate_state(args: argparse.Namespace) -> None:
    state = load_object(args.state)
    pod_id = state.get("id")
    require(isinstance(pod_id, str) and pod_id, "state lacks a pod ID")
    module = pod_module(args.state)
    terminate_until_gone(module=module, pod_id=pod_id, reason=args.reason)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init")
    init.add_argument("--name", required=True)
    init.add_argument("--state", required=True, type=Path)
    init.add_argument("--response", required=True, type=Path)
    init.add_argument("--job", required=True, type=Path)
    init.add_argument("--record", required=True, type=Path)
    init.add_argument("--start-epoch", required=True, type=int)
    init.set_defaults(function=command_init)

    watch = commands.add_parser("watch")
    watch.add_argument("--record", required=True, type=Path)
    watch.set_defaults(function=command_watch)

    status = commands.add_parser("status")
    status.add_argument("--record", required=True, type=Path)
    status.set_defaults(function=command_status)

    terminate = commands.add_parser("terminate")
    terminate.add_argument("--record", required=True, type=Path)
    terminate.add_argument("--reason", required=True)
    terminate.set_defaults(function=command_terminate)

    terminate_state = commands.add_parser("terminate-state")
    terminate_state.add_argument("--state", required=True, type=Path)
    terminate_state.add_argument("--reason", required=True)
    terminate_state.set_defaults(function=command_terminate_state)
    return parser.parse_args(argv)


def main() -> None:
    try:
        args = parse_args()
        args.function(args)
    except GuardError as exc:
        print(f"PROVIDER GUARD ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
