#!/usr/bin/env python3
"""Exclusive-write the powered-v13 Stage-T static import-closure report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from powered_v13_import_audit import (  # noqa: E402
    DESIGN_ID,
    STAGE,
    V13ImportAuditError,
    audit_stage_t_imports,
    canonical_json_bytes,
)


PLACEHOLDER_SCHEMA = "powered-v13-technical-canary-import-audit-placeholder-v1"


def _roots(value: str) -> list[str]:
    try:
        observed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("execution roots are not JSON") from exc
    if (not isinstance(observed, list) or not observed
            or any(not isinstance(item, str) for item in observed)):
        raise argparse.ArgumentTypeError(
            "execution roots must be a nonempty JSON string array")
    return observed


def _output_path(repo: Path, value: Path) -> tuple[Path, str]:
    repo = Path(repo).resolve(strict=True)
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repo / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(repo).as_posix()
    except ValueError as exc:
        raise V13ImportAuditError(
            "import-audit output is outside the exact repository") from exc
    if not relative or relative == ".":
        raise V13ImportAuditError("import-audit output path is empty")
    return candidate, relative


def _placeholder_bytes(relative: str) -> bytes:
    return canonical_json_bytes({
        "schema": PLACEHOLDER_SCHEMA,
        "design_id": DESIGN_ID,
        "stage": STAGE,
        "report_path": relative,
    }) + b"\n"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_no_symlink_component(repo: Path, relative: str) -> None:
    cursor = Path(repo).resolve(strict=True)
    for part in Path(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise V13ImportAuditError(
                "import-audit placeholder path contains a symlink")


def _replace_placeholder(
    repo: Path, output: Path, relative: str, raw: bytes,
) -> None:
    _assert_no_symlink_component(repo, relative)
    try:
        before = output.lstat()
        observed = output.read_bytes()
    except OSError as exc:
        raise V13ImportAuditError(
            "canonical import-audit placeholder is absent") from exc
    if (not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o644):
        raise V13ImportAuditError(
            "import-audit placeholder is not a mode-100644 regular file")
    if observed != _placeholder_bytes(relative):
        raise V13ImportAuditError(
            "existing import-audit output is not the exact canonical placeholder")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())

        _assert_no_symlink_component(repo, relative)
        current = output.lstat()
        if ((current.st_dev, current.st_ino, current.st_size) !=
                (before.st_dev, before.st_ino, before.st_size)
                or output.read_bytes() != observed):
            raise V13ImportAuditError(
                "import-audit placeholder changed before replacement")
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_exclusive(output: Path, raw: bytes) -> None:
    if output.exists() or output.is_symlink():
        raise V13ImportAuditError("import-audit output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(output.parent)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--execution-roots-json", required=True, type=_roots)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--replace-canonical-placeholder", action="store_true",
        help=("replace only the exact canonical self-inventory placeholder "
              "bound to --output"),
    )
    args = parser.parse_args(argv)
    try:
        repo = args.repo.resolve(strict=True)
        output, relative = _output_path(repo, args.output)
        report = audit_stage_t_imports(
            repo,
            contract_path=args.contract,
            execution_roots=args.execution_roots_json,
        )
        raw = canonical_json_bytes(report) + b"\n"
        if args.replace_canonical_placeholder:
            _replace_placeholder(repo, output, relative, raw)
            recomputed = audit_stage_t_imports(
                repo,
                contract_path=args.contract,
                execution_roots=args.execution_roots_json,
            )
            if recomputed != report or output.read_bytes() != raw:
                raise V13ImportAuditError(
                    "post-replacement import-audit recomputation differs")
        else:
            _write_exclusive(output, raw)
    except (V13ImportAuditError, OSError) as exc:
        print(f"STAGE-T IMPORT AUDIT ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": report["status"],
        "report_sha256": report["report_sha256"],
        "inventory_path_count": report["inventory_path_count"],
        "output": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
