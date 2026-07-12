"""Only paid entry point for powered-v13 Stage-T infrastructure retry 1."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Callable, Sequence

from powered_v13_infra_retry_release import (
    CARRY_IN_CONSERVATIVE_SPEND_USD,
    CARRY_IN_PROVIDER_SECONDS,
    ORIGINAL_AUTHORIZATION_COMMIT,
    ORIGINAL_MANIFEST_PATH,
    verify_infra_retry_checkout,
)
from powered_v13_release import verify_stage_t_checkout


class InfraRetryEntrypointError(RuntimeError):
    """The sole retry entry point failed before provider construction."""


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--outer-repo", required=True, type=Path)
    result.add_argument("--outer-authorization-commit", required=True)
    result.add_argument("--outer-manifest", required=True)
    result.add_argument("--outer-receipt-directory", required=True, type=Path)
    result.add_argument("--inner-repo", required=True, type=Path)
    result.add_argument("--inner-receipt-directory", required=True, type=Path)
    result.add_argument("--import-report", required=True, type=Path)
    result.add_argument("--import-report-sha256", required=True)
    result.add_argument("--session-root", required=True, type=Path)
    result.add_argument("--ssh-key", required=True, type=Path)
    result.add_argument("--hf-token", required=True, type=Path)
    result.add_argument("--primary-batch-id", required=True)
    result.add_argument(
        "--repository-url", default="https://github.com/jeremyBanks/ValueGraft.git"
    )
    return result


def _one_inner_receipt(directory: Path) -> Path:
    try:
        entries = list(Path(directory).iterdir())
    except OSError as exc:
        raise InfraRetryEntrypointError(f"cannot inspect inner receipt directory: {exc}") from exc
    expected = "powered-v13-technical-canary-launch-receipt.json"
    if len(entries) != 1 or entries[0].name != expected:
        raise InfraRetryEntrypointError(
            f"expected one fixed inner Stage-T receipt; observed {len(entries)}"
        )
    path = entries[0]
    if path.is_symlink() or not path.is_file():
        raise InfraRetryEntrypointError("inner Stage-T receipt is not a regular file")
    return path


def build_delegate_command(args: argparse.Namespace) -> list[str]:
    """Verify both authorities before constructing the fixed delegate command."""
    verify_infra_retry_checkout(
        args.outer_repo,
        authorization_commit=args.outer_authorization_commit,
        manifest_path=args.outer_manifest,
        receipt_directory=args.outer_receipt_directory,
    )
    verify_stage_t_checkout(
        args.inner_repo,
        authorization_commit=ORIGINAL_AUTHORIZATION_COMMIT,
        manifest_path=ORIGINAL_MANIFEST_PATH,
        receipt_directory=args.inner_receipt_directory,
    )
    inner_receipt = _one_inner_receipt(args.inner_receipt_directory)
    receipt_sha256 = hashlib.sha256(inner_receipt.read_bytes()).hexdigest()
    helper = args.outer_repo.resolve() / "scripts/run_powered_v13_stage_t_lifecycle.py"
    if helper.is_symlink() or not helper.is_file():
        raise InfraRetryEntrypointError("fixed outer lifecycle helper is absent or symlinked")
    return [
        str(Path(sys.executable).resolve()), str(helper), "run",
        "--repo", str(args.inner_repo.resolve()),
        "--authorization-commit", ORIGINAL_AUTHORIZATION_COMMIT,
        "--manifest", ORIGINAL_MANIFEST_PATH,
        "--receipt", str(inner_receipt.resolve()),
        "--receipt-sha256", receipt_sha256,
        "--import-report", str(args.import_report),
        "--import-report-sha256", args.import_report_sha256,
        "--session-root", str(args.session_root.resolve()),
        "--prior-stage-t-spend-usd", CARRY_IN_CONSERVATIVE_SPEND_USD,
        "--prior-stage-t-provider-seconds", str(CARRY_IN_PROVIDER_SECONDS),
        "--max-allocation-attempts", "1",
        "--ssh-key", str(args.ssh_key.resolve()),
        "--hf-token", str(args.hf_token.resolve()),
        "--primary-batch-id", args.primary_batch_id,
        "--repository-url", args.repository_url,
    ]


def run(
    args: argparse.Namespace,
    *,
    delegate: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    command = build_delegate_command(args)
    try:
        result = delegate(command, check=False)
    except OSError as exc:
        raise InfraRetryEntrypointError(f"fixed lifecycle delegate failed: {exc}") from exc
    return int(result.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        # Release and inner-release exceptions are deliberately collapsed into
        # one pre-provider terminal class at the executable boundary.
        print(f"STAGE-T INFRA RETRY ERROR: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "InfraRetryEntrypointError", "build_delegate_command", "main", "parser", "run",
]
