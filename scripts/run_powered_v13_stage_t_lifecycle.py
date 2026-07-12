#!/usr/bin/env python3
"""Narrow local helpers for the powered-v13 Stage-T provider lifecycle.

The allocation path is the injected ``StageTLifecycle`` API.  This CLI exposes
only deterministic release binding, fixed job-probe exits, and post-pull
harvest hashing; none of its subcommands can allocate a Pod.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from powered_v13_lifecycle import (  # noqa: E402
    V13LifecycleError,
    build_harvest_manifest,
    canonical_json_bytes,
    job_probe_exit,
    verify_release_binding,
)
from powered_v13_release import verify_stage_t_checkout  # noqa: E402


def _strict_object(path: Path, label: str) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise V13LifecycleError(f"{label} is absent or symlinked")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V13LifecycleError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise V13LifecycleError(f"{label} root is not an object")
    return value


def _write_json_exclusive(path: Path, value: dict) -> None:
    raw = canonical_json_bytes(value) + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def command_verify(args: argparse.Namespace) -> None:
    release = verify_release_binding(
        repo=args.repo,
        expected_authorization_commit=args.authorization_commit,
        manifest_path=args.manifest,
        receipt_path=args.receipt,
        expected_receipt_sha256=args.receipt_sha256,
        import_report_path=args.import_report,
        expected_import_report_sha256=args.import_report_sha256,
        verify_checkout=verify_stage_t_checkout,
    )
    print(json.dumps({
        "status": "PASS",
        "authorization_commit": release.authorization_commit,
        "receipt_sha256": release.receipt_sha256,
        "import_report_sha256": release.import_report_sha256,
        "job_path": release.job_path,
        "sync_paths": list(release.sync_paths),
    }, sort_keys=True))


def command_probe(args: argparse.Namespace) -> None:
    state = _strict_object(args.lifecycle_state, "lifecycle state")
    remote_state = None
    if args.remote_state is not None:
        remote = _strict_object(args.remote_state, "remote job state")
        remote_state = remote.get("state")
    raise SystemExit(job_probe_exit(
        lifecycle_status=state.get("status"), remote_state=remote_state))


def command_harvest(args: argparse.Namespace) -> None:
    manifest = build_harvest_manifest(args.root)
    _write_json_exclusive(args.output, manifest)
    print(json.dumps({
        "status": "PASS", "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "manifest_sha256": manifest["manifest_sha256"],
        "output": str(args.output),
    }, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-release")
    verify.add_argument("--repo", required=True, type=Path)
    verify.add_argument("--authorization-commit", required=True)
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--receipt", required=True, type=Path)
    verify.add_argument("--receipt-sha256", required=True)
    verify.add_argument("--import-report", required=True, type=Path)
    verify.add_argument("--import-report-sha256", required=True)
    verify.set_defaults(function=command_verify)
    probe = commands.add_parser("job-probe")
    probe.add_argument("--lifecycle-state", required=True, type=Path)
    probe.add_argument("--remote-state", type=Path)
    probe.set_defaults(function=command_probe)
    harvest = commands.add_parser("harvest-manifest")
    harvest.add_argument("--root", required=True, type=Path)
    harvest.add_argument("--output", required=True, type=Path)
    harvest.set_defaults(function=command_harvest)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        args.function(args)
    except V13LifecycleError as exc:
        print(f"STAGE-T LIFECYCLE ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
