#!/usr/bin/env python3
"""Verify, launch, probe, harvest, and clean up powered-v13 Stage T.

Only the explicit ``run`` subcommand can allocate.  Every other command is a
deterministic release, probe, or harvest helper used by the OS-owned watchdog.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from powered_v13_lifecycle import (  # noqa: E402
    ATTEMPT_DIRECTORY_TOKEN,
    HARVEST_ROOTS,
    JOB_TERMINAL_SCHEMA,
    LIFECYCLE_STATE_TOKEN,
    LaunchdWatcherBackend,
    MAX_ALLOCATION_ATTEMPTS,
    OpenSshTransport,
    POD_STATE_TOKEN,
    REMOTE_ARTIFACTS,
    REMOTE_EXTERNAL,
    REMOTE_REPO,
    RecoveringWatcherSupervisor,
    RunPodProvider,
    StageTLifecycle,
    STAGE_T_RECEIPT_BASENAME,
    V13LifecycleError,
    build_harvest_manifest,
    canonical_json_bytes,
    job_probe_exit,
    _verify_remote_setup_release_binding,
    verify_release_binding,
)


def _max_allocation_attempts(value: str) -> int:
    try:
        parsed = int(value, 10)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "max allocation attempts must be an integer") from exc
    if not 1 <= parsed <= MAX_ALLOCATION_ATTEMPTS:
        raise argparse.ArgumentTypeError(
            f"max allocation attempts must be between 1 and "
            f"{MAX_ALLOCATION_ATTEMPTS}")
    return parsed


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
    )
    print(json.dumps({
        "status": "PASS",
        "authorization_commit": release.authorization_commit,
        "receipt_sha256": release.receipt_sha256,
        "import_report_sha256": release.import_report_sha256,
        "job_path": release.job_path,
        "sync_paths": list(release.sync_paths),
    }, sort_keys=True))


def _command_verify_remote_setup(args: argparse.Namespace) -> None:
    """Fixed admitted-host setup gate; never the subject receipt verifier."""

    report_relative = Path(args.import_report)
    release = _verify_remote_setup_release_binding(
        repo=Path(REMOTE_REPO),
        expected_authorization_commit=args.authorization_commit,
        manifest_path=args.manifest,
        receipt_path=(Path(REMOTE_EXTERNAL) / STAGE_T_RECEIPT_BASENAME),
        expected_receipt_sha256=args.receipt_sha256,
        import_report_path=Path(REMOTE_REPO) / report_relative,
        expected_import_report_sha256=args.import_report_sha256,
    )
    print(json.dumps({
        "status": "PASS",
        "verification_role": "REMOTE_SETUP_ONLY",
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


def _endpoint(provider: RunPodProvider, pod_state: Path) -> tuple[str, int]:
    state = _strict_object(pod_state, "Pod state")
    pod_id = state.get("id")
    if not isinstance(pod_id, str) or not pod_id:
        raise V13LifecycleError("Pod state lacks an ID")
    pod = provider.get_pod(pod_id)
    host = pod.get("publicIp")
    mappings = pod.get("portMappings")
    port = mappings.get("22") if isinstance(mappings, dict) else None
    if not isinstance(host, str) or not host or not str(port).isdigit():
        raise V13LifecycleError("Pod has no exact SSH endpoint")
    observed_port = int(port)
    if not 1 <= observed_port <= 65535:
        raise V13LifecycleError("Pod SSH port is invalid")
    return host, observed_port


def _ssh_transport(key: Path, endpoint: tuple[str, int]) -> list[str]:
    return [
        "ssh", "-i", str(key), "-p", str(endpoint[1]),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=4",
    ]


def command_watchdog_probe(args: argparse.Namespace) -> None:
    lifecycle = _strict_object(args.lifecycle_state, "lifecycle state")
    status = lifecycle.get("status")
    if status != "JOB_STARTED":
        raise SystemExit(job_probe_exit(
            lifecycle_status=status, remote_state=None))
    provider = RunPodProvider(state_path=args.pod_state)
    endpoint = _endpoint(provider, args.pod_state)
    state_path = REMOTE_ARTIFACTS + "/partials/job-state.json"
    pid_path = REMOTE_ARTIFACTS.rsplit("/", 1)[0] + "/job.pid"
    remote_probe = (
        "set -euo pipefail; "
        f"state=$(cat {shlex.quote(state_path)}); "
        "case \"$state\" in "
        "*'\"state\":\"RUNNING\"'*) "
        f"pid=$(cat {shlex.quote(pid_path)} 2>/dev/null || true); "
        "if case \"$pid\" in ''|*[!0-9]*) false;; *) true;; esac "
        "&& kill -0 \"$pid\" 2>/dev/null; then printf '%s\\n' \"$state\"; "
        "else printf '{\"state\":\"DEAD\"}\\n'; fi ;; "
        "*) printf '%s\\n' \"$state\" ;; esac"
    )
    command = [
        *_ssh_transport(args.ssh_key, endpoint), "-n", f"root@{endpoint[0]}",
        remote_probe,
    ]
    try:
        observed = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=45)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise V13LifecycleError("bounded remote job probe failed") from exc
    if observed.returncode != 0:
        raise V13LifecycleError("remote job probe is ambiguous")
    try:
        remote = json.loads(observed.stdout)
    except json.JSONDecodeError as exc:
        raise V13LifecycleError("remote job state is malformed") from exc
    if not isinstance(remote, dict):
        raise V13LifecycleError("remote job state root differs")
    raise SystemExit(job_probe_exit(
        lifecycle_status=status, remote_state=remote.get("state")))


def command_watchdog_harvest(args: argparse.Namespace) -> None:
    if args.destination.exists() or args.destination.is_symlink():
        raise V13LifecycleError("bounded harvest destination already exists")
    provider = RunPodProvider(state_path=args.pod_state)
    endpoint = _endpoint(provider, args.pod_state)
    temporary = args.destination.with_name(
        f".{args.destination.name}.{os.getpid()}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise V13LifecycleError("bounded harvest temporary path already exists")
    temporary.mkdir(parents=True)
    shell = shlex.join(_ssh_transport(args.ssh_key, endpoint))
    try:
        for category in HARVEST_ROOTS:
            local = temporary / category
            local.mkdir()
            source = (
                f"root@{endpoint[0]}:{REMOTE_ARTIFACTS}/{category}/")
            result = subprocess.run([
                "rsync", "-az", "--checksum", "--partial", "-e", shell,
                source, f"{local}/",
            ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False, timeout=45)
            if result.returncode != 0:
                raise V13LifecycleError(
                    f"bounded harvest failed for {category}")
        terminal_path = temporary / "receipts/job-terminal.json"
        terminal = _strict_object(terminal_path, "Stage-T job terminal receipt")
        raw = terminal_path.read_bytes()
        if raw != canonical_json_bytes(terminal) + b"\n":
            raise V13LifecycleError(
                "Stage-T job terminal receipt is not canonical JSON plus LF")
        if (set(terminal) != {
                "schema", "status", "returncode", "primary_batch_id"}
                or terminal.get("schema") != JOB_TERMINAL_SCHEMA
                or terminal.get("primary_batch_id") != args.primary_batch_id
                or terminal.get("status") not in {"COMPLETE", "DEAD"}
                or type(terminal.get("returncode")) is not int
                or not ((terminal["status"] == "COMPLETE"
                         and terminal["returncode"] == 0)
                        or (terminal["status"] == "DEAD"
                            and terminal["returncode"] != 0))):
            raise V13LifecycleError(
                "Stage-T job terminal receipt fields/batch/returncode differ")
        manifest = build_harvest_manifest(temporary)
        os.replace(temporary, args.destination)
        _write_json_exclusive(args.output, manifest)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise V13LifecycleError("bounded remote harvest failed") from exc
    print(json.dumps({
        "status": "PASS", "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "manifest_sha256": manifest["manifest_sha256"],
        "destination": str(args.destination), "output": str(args.output),
    }, sort_keys=True))


def command_run(args: argparse.Namespace) -> None:
    release = verify_release_binding(
        repo=args.repo,
        expected_authorization_commit=args.authorization_commit,
        manifest_path=args.manifest,
        receipt_path=args.receipt,
        expected_receipt_sha256=args.receipt_sha256,
        import_report_path=args.import_report,
        expected_import_report_sha256=args.import_report_sha256,
    )
    session = args.session_root.resolve()
    transient_state = session.with_name(f"{session.name}.provider-state.json")
    if transient_state.exists() or transient_state.is_symlink():
        raise V13LifecycleError("unique provider-state path already exists")
    print(json.dumps({
        "status": "STARTING", "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "authorization_commit": release.authorization_commit,
        "session_root": str(session), "provider_state": str(transient_state),
        "prior_stage_t_provider_seconds":
            args.prior_stage_t_provider_seconds,
        "max_allocation_attempts": args.max_allocation_attempts,
    }, sort_keys=True), flush=True)
    provider = RunPodProvider(state_path=transient_state)
    transport = OpenSshTransport(
        provider=provider, ssh_key=args.ssh_key, hf_token=args.hf_token,
        authorization_commit=release.authorization_commit,
        primary_batch_id=args.primary_batch_id,
        manifest_path=release.manifest_path,
        receipt_sha256=release.receipt_sha256,
        import_report_path=release.import_report_path.relative_to(
            args.repo.resolve()).as_posix(),
        import_report_sha256=release.import_report_sha256,
        repository_url=args.repository_url)
    provider_key_path = Path(provider.module.KEY_PATH)
    if not provider_key_path.is_absolute():
        provider_key_path = args.repo.resolve() / provider_key_path
    provider_key_path = provider_key_path.resolve(strict=True)
    if not provider_key_path.is_file() or provider_key_path.is_symlink():
        raise V13LifecycleError("RunPod API key path is absent or symlinked")
    backend = LaunchdWatcherBackend(
        repo=args.repo, runpod_key_path=provider_key_path)
    supervisor = RecoveringWatcherSupervisor(backend)
    python = str(Path(sys.executable).resolve())
    helper = str(Path(__file__).resolve())
    job_probe = [
        python, helper, "watchdog-probe", "--lifecycle-state",
        LIFECYCLE_STATE_TOKEN, "--pod-state", POD_STATE_TOKEN,
        "--ssh-key", str(args.ssh_key.resolve()),
    ]
    harvest = [
        python, helper, "watchdog-harvest", "--pod-state", POD_STATE_TOKEN,
        "--ssh-key", str(args.ssh_key.resolve()), "--destination",
        f"{ATTEMPT_DIRECTORY_TOKEN}/harvest", "--output",
        f"{ATTEMPT_DIRECTORY_TOKEN}/harvest-manifest.json",
        "--primary-batch-id", args.primary_batch_id,
    ]
    lifecycle = StageTLifecycle(
        repo=args.repo, session_root=session, release=release,
        provider=provider, transport=transport, supervisor=supervisor,
        clock=time.time, prior_stage_t_spend_usd=args.prior_stage_t_spend_usd,
        prior_stage_t_provider_seconds=args.prior_stage_t_provider_seconds,
        job_probe_command=job_probe, harvest_command=harvest,
        max_allocation_attempts=args.max_allocation_attempts)
    result = lifecycle.run()
    print(json.dumps({
        "status": result["status"], "session_root": str(session),
        "observed_stage_t_spend_usd": result["observed_stage_t_spend_usd"],
        "observed_stage_t_provider_seconds":
            result["observed_stage_t_provider_seconds"],
        "admitted_pod_id": result["admitted_pod_id"],
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
    setup_verify = commands.add_parser(
        "_verify-remote-setup", help=argparse.SUPPRESS)
    setup_verify.add_argument("--authorization-commit", required=True)
    setup_verify.add_argument("--manifest", required=True)
    setup_verify.add_argument("--receipt-sha256", required=True)
    setup_verify.add_argument("--import-report", required=True)
    setup_verify.add_argument("--import-report-sha256", required=True)
    setup_verify.set_defaults(function=_command_verify_remote_setup)
    probe = commands.add_parser("job-probe")
    probe.add_argument("--lifecycle-state", required=True, type=Path)
    probe.add_argument("--remote-state", type=Path)
    probe.set_defaults(function=command_probe)
    harvest = commands.add_parser("harvest-manifest")
    harvest.add_argument("--root", required=True, type=Path)
    harvest.add_argument("--output", required=True, type=Path)
    harvest.set_defaults(function=command_harvest)
    probe_remote = commands.add_parser("watchdog-probe")
    probe_remote.add_argument("--lifecycle-state", required=True, type=Path)
    probe_remote.add_argument("--pod-state", required=True, type=Path)
    probe_remote.add_argument("--ssh-key", required=True, type=Path)
    probe_remote.set_defaults(function=command_watchdog_probe)
    pull = commands.add_parser("watchdog-harvest")
    pull.add_argument("--pod-state", required=True, type=Path)
    pull.add_argument("--ssh-key", required=True, type=Path)
    pull.add_argument("--destination", required=True, type=Path)
    pull.add_argument("--output", required=True, type=Path)
    pull.add_argument("--primary-batch-id", required=True)
    pull.set_defaults(function=command_watchdog_harvest)
    run = commands.add_parser("run")
    run.add_argument("--repo", required=True, type=Path)
    run.add_argument("--authorization-commit", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--receipt", required=True, type=Path)
    run.add_argument("--receipt-sha256", required=True)
    run.add_argument("--import-report", required=True, type=Path)
    run.add_argument("--import-report-sha256", required=True)
    run.add_argument("--session-root", required=True, type=Path)
    run.add_argument("--prior-stage-t-spend-usd", required=True)
    run.add_argument(
        "--prior-stage-t-provider-seconds", required=True, type=int)
    run.add_argument(
        "--max-allocation-attempts", type=_max_allocation_attempts,
        default=MAX_ALLOCATION_ATTEMPTS)
    run.add_argument("--ssh-key", required=True, type=Path)
    run.add_argument("--hf-token", required=True, type=Path)
    run.add_argument("--primary-batch-id", required=True)
    run.add_argument(
        "--repository-url",
        default="https://github.com/jeremyBanks/ValueGraft.git")
    run.set_defaults(function=command_run)
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
