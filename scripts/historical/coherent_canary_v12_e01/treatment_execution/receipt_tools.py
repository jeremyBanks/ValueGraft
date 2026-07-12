#!/usr/bin/env python3
"""Validate and install receipt-driven exact-v12 treatment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from typing import Any, Mapping


DESIGN_ID = "coherent-state-decision-canary-v12"
RECEIPT_SCHEMA = "coherent_state_decision_canary_v12_treatment_pod_receipt_v1"
EXPECTED_COMMIT = "cbdfa481fe08de62bf8178d09ca040716d610f21"
EXPECTED_RUNTIME_SHA = "b6158b989b0b467c46589f3ecbe69acba88645d4cd193dd23d925e2e4cd69484"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
RAW_RE = re.compile(
    r"results/coherent_canary_v12_treatment/"
    r"coherent-canary-v12-treatment-e01_exact-subject_"
    r"[0-9]{8}T[0-9]{12}Z\.json"
)
HARVEST_RE = re.compile(
    r"results/coherent_canary_v12_harvest/"
    r"coherent-canary-v12-harvest-e01-exact-subject-"
    r"[0-9]{8}T[0-9]{6}Z\.json"
)
JOB_RE = re.compile(
    r"results/coherent_canary_v12_treatment/"
    r"coherent-canary-v12-treatment-e01-exact-subject-"
    r"[0-9]{8}T[0-9]{6}Z_job\.md"
)
RECEIPT_RE = re.compile(
    r"results/coherent_canary_v12_treatment/"
    r"coherent-canary-v12-treatment-e01-exact-subject-"
    r"[0-9]{8}T[0-9]{6}Z_receipt\.json"
)


class ReceiptError(RuntimeError):
    """A receipt or pulled artifact differs from the orchestration contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except Exception as exc:
        raise ReceiptError(f"cannot parse receipt {path}: {exc}") from exc
    require(isinstance(value, dict), "receipt root is not an object")
    return value


def safe_relative(value: Any, pattern: re.Pattern[str], label: str) -> str:
    require(isinstance(value, str) and pattern.fullmatch(value) is not None,
            f"{label} path differs")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts,
            f"{label} path is unsafe")
    return pure.as_posix()


def validate_row(row: Any, pattern: re.Pattern[str], label: str) -> dict[str, Any]:
    require(isinstance(row, Mapping) and set(row) == {
        "path", "sha256", "size_bytes"
    }, f"{label} artifact row differs")
    path = safe_relative(row.get("path"), pattern, label)
    digest = row.get("sha256")
    size = row.get("size_bytes")
    require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
            f"{label} SHA-256 differs")
    require(isinstance(size, int) and not isinstance(size, bool) and size > 0,
            f"{label} size differs")
    return {"path": path, "sha256": digest, "size_bytes": size}


def validate_receipt(path: Path, *, receipt_relative: str | None = None
                     ) -> dict[str, Any]:
    document = load_object(path)
    required = {
        "schema", "design_id", "case_id", "completed_at_utc",
        "expected_commit", "observed_commit", "runner_exit", "runner_status",
        "harvester_exit", "harvest_status", "terminal_status",
        "semantic_evidence_eligible", "phase_a_release_status",
        "phase_a_release_eligible", "primary_arm_count",
        "placebo_control_count", "available_placebo_control_count",
        "provider_hourly_cost_usd", "runner_wall_cap_seconds", "gpu",
        "runtime_fingerprint_sha256", "diagnostic_only",
        "formal_v12_decision_eligible", "aggregate_expansion_authorized",
        "artifacts",
    }
    require(set(document) == required, "receipt field set differs")
    require(document.get("schema") == RECEIPT_SCHEMA and
            document.get("design_id") == DESIGN_ID and
            document.get("case_id") == "e01",
            "receipt schema/design/case differs")
    require(document.get("expected_commit") == EXPECTED_COMMIT and
            document.get("observed_commit") == EXPECTED_COMMIT,
            "receipt commit binding differs")
    require(document.get("provider_hourly_cost_usd") == 1.39,
            "receipt provider rate differs")
    require(document.get("runner_wall_cap_seconds") == 1500,
            "receipt runner wall cap differs")
    require(document.get("diagnostic_only") is True and
            document.get("formal_v12_decision_eligible") is False and
            document.get("aggregate_expansion_authorized") is False,
            "receipt diagnostic/formal-release boundary differs")
    gpu = document.get("gpu")
    gpu_valid = False
    if isinstance(gpu, str) and len(gpu.splitlines()) == 1:
        gpu_parts = [part.strip() for part in gpu.split(",", 3)]
        gpu_valid = (
            len(gpu_parts) == 4 and
            gpu_parts[0] == "NVIDIA A100 80GB PCIe" and
            gpu_parts[1].startswith("GPU-") and
            gpu_parts[2] == "580.159.04" and
            gpu_parts[3].isdigit() and int(gpu_parts[3]) >= 80000
        )
    runtime_sha = document.get("runtime_fingerprint_sha256")
    require(runtime_sha is None or
            (isinstance(runtime_sha, str) and
             SHA256_RE.fullmatch(runtime_sha) is not None),
            "receipt runtime fingerprint is neither absent nor a SHA-256")
    artifacts = document.get("artifacts")
    require(isinstance(artifacts, Mapping) and set(artifacts) == {
        "raw", "harvest", "job_log"
    }, "receipt artifact set differs")
    normalized = {
        "raw": validate_row(artifacts["raw"], RAW_RE, "raw"),
        "job_log": validate_row(artifacts["job_log"], JOB_RE, "job_log"),
        "harvest": None,
    }
    if artifacts["harvest"] is not None:
        normalized["harvest"] = validate_row(
            artifacts["harvest"], HARVEST_RE, "harvest")

    terminal = document.get("terminal_status")
    require(terminal in {"PASS", "ERROR"}, "receipt terminal status differs")
    available = document.get("available_placebo_control_count")
    pass_contract = (
        document.get("runner_exit") == 0 and
        document.get("runner_status") == "PASS" and
        document.get("harvester_exit") == 0 and
        document.get("harvest_status") == "PASS" and
        document.get("semantic_evidence_eligible") is True and
        document.get("phase_a_release_status") == "PRETREATMENT_PASS" and
        document.get("phase_a_release_eligible") is True and
        document.get("primary_arm_count") == 31 and
        document.get("placebo_control_count") == 3 and
        isinstance(available, int) and not isinstance(available, bool) and
        0 <= available <= 3 and normalized["harvest"] is not None and
        gpu_valid and runtime_sha == EXPECTED_RUNTIME_SHA
    )
    require((terminal == "PASS") is pass_contract,
            "receipt terminal status differs from the reconstructed contract")

    if receipt_relative is not None:
        safe_relative(receipt_relative, RECEIPT_RE, "receipt")
    document["artifacts"] = normalized
    return document


def hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def verify_file(path: Path, row: Mapping[str, Any], label: str) -> None:
    require(path.is_file() and not path.is_symlink(),
            f"staged {label} is absent or not a regular file")
    size, digest = hash_file(path)
    require(size == row["size_bytes"] and digest == row["sha256"],
            f"staged {label} differs from receipt")


def install_exclusive(source: Path, destination: Path) -> None:
    if destination.exists():
        require(destination.is_file() and not destination.is_symlink(),
                f"existing destination is not a regular file: {destination}")
        require(hash_file(source) == hash_file(destination),
                f"refusing to overwrite different existing file: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            descriptor = -1
            shutil.copyfileobj(input_file, output, length=1 << 20)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise


def command_validate(args: argparse.Namespace) -> None:
    document = validate_receipt(
        args.receipt, receipt_relative=args.receipt_relative)
    print(json.dumps({
        "terminal_status": document["terminal_status"],
        "artifact_keys": [
            key for key, row in document["artifacts"].items()
            if row is not None
        ],
    }, sort_keys=True))


def command_list(args: argparse.Namespace) -> None:
    document = validate_receipt(args.receipt)
    for key in ("raw", "harvest", "job_log"):
        row = document["artifacts"][key]
        if row is not None:
            print(f"{key}\t{row['path']}")


def command_install(args: argparse.Namespace) -> None:
    document = validate_receipt(
        args.receipt, receipt_relative=args.receipt_relative)
    artifacts = document["artifacts"]
    installed: dict[str, str] = {}
    for key in ("raw", "harvest", "job_log"):
        row = artifacts[key]
        if row is None:
            continue
        staged = args.staging / key
        verify_file(staged, row, key)
        if key == "raw":
            destination = args.intact_raw / Path(row["path"]).name
        else:
            destination = args.repo / row["path"]
        install_exclusive(staged, destination)
        installed[key] = str(destination)

    receipt_destination = args.repo / safe_relative(
        args.receipt_relative, RECEIPT_RE, "receipt")
    install_exclusive(args.receipt, receipt_destination)
    installed["receipt"] = str(receipt_destination)
    print(json.dumps({
        "terminal_status": document["terminal_status"],
        "installed": installed,
    }, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("receipt", type=Path)
    validate.add_argument("--receipt-relative")
    validate.set_defaults(function=command_validate)

    listing = subparsers.add_parser("list")
    listing.add_argument("receipt", type=Path)
    listing.set_defaults(function=command_list)

    install = subparsers.add_parser("install")
    install.add_argument("receipt", type=Path)
    install.add_argument("--receipt-relative", required=True)
    install.add_argument("--staging", required=True, type=Path)
    install.add_argument("--repo", required=True, type=Path)
    install.add_argument("--intact-raw", required=True, type=Path)
    install.set_defaults(function=command_install)
    return parser.parse_args(argv)


def main() -> None:
    try:
        args = parse_args()
        args.function(args)
    except ReceiptError as exc:
        print(f"RECEIPT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
