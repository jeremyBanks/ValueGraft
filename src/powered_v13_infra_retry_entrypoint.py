"""Only paid entry point for powered-v13 Stage-T infrastructure retry 1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import pwd
import re
import subprocess
import sys
from typing import Callable, Sequence

from powered_v13_infra_retry_release import (
    CARRY_IN_CONSERVATIVE_SPEND_USD,
    CARRY_IN_PROVIDER_SECONDS,
    CONSUMPTION_ROOT_RELATIVE,
    ORIGINAL_AUTHORIZATION_COMMIT,
    ORIGINAL_MANIFEST_PATH,
    RECEIPT_BASENAME,
    canonical_json_bytes,
    verify_infra_retry_checkout,
)
from powered_v13_lifecycle import verify_release_binding
from powered_v13_release import verify_stage_t_checkout


class InfraRetryEntrypointError(RuntimeError):
    """The sole retry entry point failed before provider construction."""


EXECUTING_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/jeremyBanks/ValueGraft.git"
CONSUMPTION_ROOT = (
    Path(pwd.getpwuid(os.getuid()).pw_dir) / CONSUMPTION_ROOT_RELATIVE
)
_BATCH_ID_RE = re.compile(
    r"stage-t-infra-retry-1-([0-9a-f]{12})-[0-9]{8}T[0-9]{6}Z\Z"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
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


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _consume_authorization(
    args: argparse.Namespace,
    *,
    outer: dict,
    inner: dict,
    outer_receipt: Path,
    inner_receipt: Path,
    session: Path,
    import_report: Path,
) -> Path:
    root = CONSUMPTION_ROOT
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InfraRetryEntrypointError(f"cannot create fixed consumption root: {exc}") from exc
    if root.is_symlink() or not root.is_dir():
        raise InfraRetryEntrypointError("fixed consumption root is not a real directory")
    authorization = outer["authorization"]
    outer_receipt_evidence = outer["receipt"]
    inner_authorization = inner["authorization"]
    inner_receipt_evidence = inner["receipt"]
    document = {
        "batch_id": args.primary_batch_id,
        "carry_in_conservative_spend_usd": CARRY_IN_CONSERVATIVE_SPEND_USD,
        "carry_in_provider_seconds": CARRY_IN_PROVIDER_SECONDS,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "import_report_path": import_report.relative_to(args.inner_repo.resolve()).as_posix(),
        "import_report_sha256": args.import_report_sha256,
        "inner_authorization_commit": ORIGINAL_AUTHORIZATION_COMMIT,
        "inner_manifest_path": ORIGINAL_MANIFEST_PATH,
        "inner_manifest_sha256": inner_authorization["manifest_sha256"],
        "inner_receipt_sha256": inner_receipt_evidence["receipt_sha256"],
        "max_allocation_attempts": 1,
        "outer_authorization_commit": args.outer_authorization_commit,
        "outer_manifest_path": args.outer_manifest,
        "outer_manifest_sha256": authorization["manifest_sha256"],
        "outer_receipt_sha256": _sha256_path(outer_receipt),
        "outer_static_root_commit": authorization["static_root_commit"],
        "schema": "powered-v13-stage-t-infra-retry-consumption-v1",
        "session_root": str(session),
    }
    # Evidence values are accessed above so a malformed verifier result fails
    # before the irrevocable marker. Bind the literal on-disk receipt bytes too.
    if outer_receipt_evidence.get("authorization_commit") != args.outer_authorization_commit:
        raise InfraRetryEntrypointError("outer receipt evidence differs before consumption")
    if _sha256_path(inner_receipt) != inner_receipt_evidence["receipt_sha256"]:
        raise InfraRetryEntrypointError("inner receipt bytes differ before consumption")
    raw = canonical_json_bytes(document) + b"\n"
    marker = root / f"{args.outer_authorization_commit}.json"
    try:
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise InfraRetryEntrypointError("outer retry authorization is already consumed") from exc
    except OSError as exc:
        raise InfraRetryEntrypointError(f"cannot create consumption record: {exc}") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # Never remove a partially written marker: existence must continue to
        # fail closed after an ambiguous consumption write.
        raise
    return marker


def build_delegate_command(args: argparse.Namespace) -> list[str]:
    """Verify both authorities before constructing the fixed delegate command."""
    match = _BATCH_ID_RE.fullmatch(args.primary_batch_id)
    if match is None or match.group(1) != args.outer_authorization_commit[:12]:
        raise InfraRetryEntrypointError(
            "primary batch ID does not bind outer authorization short hash"
        )
    session = args.session_root.resolve()
    for label, repository in (
        ("executing outer", EXECUTING_ROOT),
        ("inner", args.inner_repo.resolve()),
    ):
        try:
            session.relative_to(repository)
        except ValueError:
            pass
        else:
            raise InfraRetryEntrypointError(
                f"session root may not be inside {label} repository"
            )
    if session.name != args.primary_batch_id:
        raise InfraRetryEntrypointError("session root basename must equal primary batch ID")
    outer = verify_infra_retry_checkout(
        EXECUTING_ROOT,
        authorization_commit=args.outer_authorization_commit,
        manifest_path=args.outer_manifest,
        receipt_directory=args.outer_receipt_directory,
    )
    inner = verify_stage_t_checkout(
        args.inner_repo,
        authorization_commit=ORIGINAL_AUTHORIZATION_COMMIT,
        manifest_path=ORIGINAL_MANIFEST_PATH,
        receipt_directory=args.inner_receipt_directory,
    )
    inner_receipt = _one_inner_receipt(args.inner_receipt_directory)
    receipt_sha256 = hashlib.sha256(inner_receipt.read_bytes()).hexdigest()
    import_report = args.import_report
    if not import_report.is_absolute():
        import_report = args.inner_repo / import_report
    import_report = import_report.resolve(strict=True)
    verify_release_binding(
        repo=args.inner_repo,
        expected_authorization_commit=ORIGINAL_AUTHORIZATION_COMMIT,
        manifest_path=ORIGINAL_MANIFEST_PATH,
        receipt_path=inner_receipt,
        expected_receipt_sha256=receipt_sha256,
        import_report_path=import_report,
        expected_import_report_sha256=args.import_report_sha256,
    )
    helper = EXECUTING_ROOT / "scripts/run_powered_v13_stage_t_lifecycle.py"
    if helper.is_symlink() or not helper.is_file():
        raise InfraRetryEntrypointError("fixed outer lifecycle helper is absent or symlinked")
    outer_receipt = Path(args.outer_receipt_directory) / RECEIPT_BASENAME
    _consume_authorization(
        args, outer=outer, inner=inner, outer_receipt=outer_receipt,
        inner_receipt=inner_receipt, session=session, import_report=import_report,
    )
    return [
        str(Path(sys.executable).resolve()), str(helper), "run",
        "--repo", str(args.inner_repo.resolve()),
        "--authorization-commit", ORIGINAL_AUTHORIZATION_COMMIT,
        "--manifest", ORIGINAL_MANIFEST_PATH,
        "--receipt", str(inner_receipt.resolve()),
        "--receipt-sha256", receipt_sha256,
        "--import-report", str(import_report),
        "--import-report-sha256", args.import_report_sha256,
        "--session-root", str(session),
        "--prior-stage-t-spend-usd", CARRY_IN_CONSERVATIVE_SPEND_USD,
        "--prior-stage-t-provider-seconds", str(CARRY_IN_PROVIDER_SECONDS),
        "--max-allocation-attempts", "1",
        "--ssh-key", str(args.ssh_key.resolve()),
        "--hf-token", str(args.hf_token.resolve()),
        "--primary-batch-id", args.primary_batch_id,
        "--repository-url", REPOSITORY_URL,
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
