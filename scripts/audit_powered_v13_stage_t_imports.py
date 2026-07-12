#!/usr/bin/env python3
"""Exclusive-write the powered-v13 Stage-T static import-closure report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from powered_v13_import_audit import (  # noqa: E402
    V13ImportAuditError,
    audit_stage_t_imports,
    canonical_json_bytes,
)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--execution-roots-json", required=True, type=_roots)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_stage_t_imports(
            args.repo,
            contract_path=args.contract,
            execution_roots=args.execution_roots_json,
        )
        if args.output.exists() or args.output.is_symlink():
            raise V13ImportAuditError("import-audit output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(report) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except (V13ImportAuditError, OSError) as exc:
        print(f"STAGE-T IMPORT AUDIT ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": report["status"],
        "report_sha256": report["report_sha256"],
        "inventory_path_count": report["inventory_path_count"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
