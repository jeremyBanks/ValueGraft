#!/usr/bin/env python3
"""Initialize, run, or independently resume the powered-v13 Stage-T guard."""

from __future__ import annotations

import argparse
from decimal import Decimal
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
import urllib.error


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from powered_v13_watchdog import (  # noqa: E402
    PROVIDER_API_TIMEOUT_SECONDS,
    V13WatchdogError,
    build_record,
    file_sha256,
    read_record,
    watch,
    write_record_exclusive,
)


class RunPodBackend:
    """Small allowlisted adapter around the existing credential-owning client."""

    def __init__(self, state_path: Path):
        os.environ["SC_POD_STATE"] = str(Path(state_path).resolve())
        os.environ["SC_POD_API_TIMEOUT_S"] = str(
            PROVIDER_API_TIMEOUT_SECONDS)
        module = importlib.import_module("pod")
        self._module = importlib.reload(module)
        key_override = os.environ.get("SC_RUNPOD_KEY_PATH")
        if key_override:
            key_path = Path(key_override)
            if (not key_path.is_absolute() or not key_path.is_file()
                    or key_path.is_symlink()):
                raise V13WatchdogError(
                    "absolute RunPod API key path is absent or symlinked")
            self._module.KEY_PATH = key_path

    def get_pod(self, pod_id: str) -> Mapping[str, Any]:
        value = self._module.api("GET", f"/pods/{pod_id}")
        if not isinstance(value, Mapping) or value.get("id") != pod_id:
            raise V13WatchdogError("provider returned a different Pod")
        return value

    def active_pod_ids(self) -> list[str]:
        value = self._module.api("GET", "/pods")
        if not isinstance(value, list):
            raise V13WatchdogError(
                "provider active inventory is not a complete list")
        ids: list[str] = []
        for index, row in enumerate(value):
            if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
                raise V13WatchdogError(
                    f"provider active inventory row {index} lacks an ID")
            ids.append(row["id"])
        if len(ids) != len(set(ids)):
            raise V13WatchdogError("provider active inventory repeats a Pod ID")
        return sorted(ids)

    def delete_pod(self, pod_id: str) -> None:
        self._module.api("DELETE", f"/pods/{pod_id}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V13WatchdogError(f"JSON duplicates key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise V13WatchdogError(f"bound JSON is absent or symlinked: {path}")
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_pairs)
    except Exception as exc:
        raise V13WatchdogError(f"cannot parse bound JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V13WatchdogError(f"bound JSON root is not an object: {path}")
    return value


def _bound_file(path: Path, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise V13WatchdogError(f"{label} is absent, empty, or symlinked")
    return path


def _canonical_provider_money(value: object, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise V13WatchdogError(f"{label} is not numeric")
    try:
        observed = Decimal(str(value))
    except Exception as exc:
        raise V13WatchdogError(f"{label} is not decimal") from exc
    if not observed.is_finite() or observed <= 0:
        raise V13WatchdogError(f"{label} is invalid")
    return format(observed.normalize(), "f")


def _command_json(value: str, label: str) -> list[str]:
    try:
        observed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise V13WatchdogError(f"{label} is not JSON") from exc
    if (not isinstance(observed, list) or not observed
            or any(not isinstance(item, str) or not item for item in observed)):
        raise V13WatchdogError(f"{label} is not a nonempty string array")
    return observed


def command_init(args: argparse.Namespace) -> None:
    state_path = _bound_file(args.state, "Pod state")
    response_path = _bound_file(args.create_response, "create response")
    job_path = _bound_file(args.job, "Stage-T job")
    receipt_path = _bound_file(args.release_receipt, "Stage-T release receipt")
    state = _load_json(state_path)
    response = _load_json(response_path)
    pod_id = state.get("id")
    if not isinstance(pod_id, str) or response.get("id") != pod_id:
        raise V13WatchdogError("state/create-response Pod IDs differ")
    state_rate = _canonical_provider_money(state.get("costPerHr"), "state rate")
    response_rate = _canonical_provider_money(
        response.get("costPerHr"), "create-response rate")
    if state_rate != response_rate:
        raise V13WatchdogError("state/create-response rates differ")
    record = build_record(
        pod_id=pod_id,
        created_cost_per_hr_usd=state_rate,
        prior_stage_t_spend_usd=args.prior_stage_t_spend_usd,
        prior_stage_t_provider_seconds=args.prior_stage_t_provider_seconds,
        provider_clock_started_epoch=args.provider_clock_started_epoch,
        pod_state_sha256=file_sha256(state_path),
        create_response_sha256=file_sha256(response_path),
        job_sha256=file_sha256(job_path),
        release_receipt_sha256=file_sha256(receipt_path),
        job_probe_command=_command_json(
            args.job_probe_command_json, "job probe command"),
        harvest_command=_command_json(
            args.harvest_command_json, "harvest command"),
    )
    write_record_exclusive(args.record, record)
    print(json.dumps({
        "record": str(args.record),
        "pod_id": pod_id,
        "delete_trigger_epoch": record["delete_trigger_epoch"],
        "hard_deadline_epoch": record["hard_deadline_epoch"],
        "prior_stage_t_provider_seconds":
            record["prior_stage_t_provider_seconds"],
        "bounded_provider_seconds": record["bounded_provider_seconds"],
    }, sort_keys=True))


def command_watch(args: argparse.Namespace) -> None:
    record = read_record(args.record)
    backend = RunPodBackend(args.state)
    result = watch(record_path=args.record, backend=backend)
    print(json.dumps({
        "status": result["status"],
        "pod_id": result["pod_id"],
        "termination_reason": result["termination_reason"],
        "harvest_succeeded": result["harvest"]["succeeded"],
        "per_pod_404_observed": result["per_pod_404_observed"],
        "active_inventory_absent_observed": (
            result["active_inventory_absent_observed"]),
        "record_sha256": file_sha256(args.record),
        "release_receipt_sha256": record["release_receipt_sha256"],
    }, sort_keys=True))


def command_emergency_cleanup(args: argparse.Namespace) -> None:
    """Independent supervisor entry after the ordinary watcher dies."""

    backend = RunPodBackend(args.state)
    result = watch(
        record_path=args.record,
        backend=backend,
        probe=lambda _command: "DEAD",
    )
    print(json.dumps({
        "status": result["status"],
        "pod_id": result["pod_id"],
        "termination_reason": result["termination_reason"],
        "record_sha256": file_sha256(args.record),
    }, sort_keys=True))


def command_status(args: argparse.Namespace) -> None:
    record = read_record(args.record)
    backend = RunPodBackend(args.state)
    try:
        pod = backend.get_pod(record["pod_id"])
        per_pod = {
            "http_status": 200,
            "id": pod.get("id"),
            "desiredStatus": pod.get("desiredStatus"),
            "costPerHr": pod.get("costPerHr"),
        }
    except urllib.error.HTTPError as exc:
        per_pod = {"http_status": exc.code}
    active = backend.active_pod_ids()
    print(json.dumps({
        "record": record,
        "record_sha256": file_sha256(args.record),
        "provider": {"per_pod": per_pod, "active_pod_ids": active},
    }, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init")
    init.add_argument("--state", required=True, type=Path)
    init.add_argument("--create-response", required=True, type=Path)
    init.add_argument("--job", required=True, type=Path)
    init.add_argument("--release-receipt", required=True, type=Path)
    init.add_argument("--record", required=True, type=Path)
    init.add_argument("--prior-stage-t-spend-usd", required=True)
    init.add_argument(
        "--prior-stage-t-provider-seconds", required=True, type=int)
    init.add_argument("--provider-clock-started-epoch", required=True, type=int)
    init.add_argument("--job-probe-command-json", required=True)
    init.add_argument("--harvest-command-json", required=True)
    init.set_defaults(function=command_init)

    for name, function in (
        ("watch", command_watch),
        ("emergency-cleanup", command_emergency_cleanup),
        ("status", command_status),
    ):
        command = commands.add_parser(name)
        command.add_argument("--state", required=True, type=Path)
        command.add_argument("--record", required=True, type=Path)
        command.set_defaults(function=function)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        args.function(args)
    except (V13WatchdogError, OSError) as exc:
        print(f"STAGE-T WATCHDOG ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
