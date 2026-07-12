#!/usr/bin/env python3
"""Fail-closed, outcome-agnostic post-run audit for v12 e01 treatment.

The program has no result-discovery logic.  A caller must explicitly provide
the treatment artifact and the pod-produced harvest.  It writes only beneath
an explicitly supplied, newly-created work directory.

Checks:

1. The raw treatment ``fresh_scores.{focal,nonfocal}`` must be canonically
   JSON-identical to the bound Phase-A ``FF_{focal,nonfocal}`` records.
2. The treatment bytes are reconstructed byte-for-byte in the work directory.
3. The committed tracked harvester is run in a subprocess against that
   reconstruction.
4. Pod and local harvests must become canonically identical after normalizing
   only their harvest timestamp and provenance paths.  Every other difference
   invalidates the audit.

No model or tokenizer is loaded by this checker or by the v12 harvester.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARVESTER = WORKSPACE_ROOT / "scripts/harvest_coherent_canary_v12.py"
EXPECTED_HARVESTER_SHA256 = (
    "9dbe29ad84cd001e13c5144605f506d52a4f6b06d00353436a49fb67bbbfdfad"
)
REPORT_SCHEMA = "coherent_canary_v12_e01_treatment_postrun_audit_v1"
DESIGN_ID = "coherent-state-decision-canary-v12"
RUN_SCHEMA = "coherent_state_decision_canary_v12_treatment_run_v1"
PHASE_A_RUN_SCHEMA = "coherent_state_decision_canary_v12_phase_a_run_v1"
HARVEST_SCHEMA = "coherent_state_decision_canary_v12_treatment_harvest_v1"
CASE_ID = "e01"
SUBJECT = "exact-subject"
ALLOWED_LITERAL_DIFFERENCES = {
    "/completed_at_utc": "timestamp",
    "/treatment_artifact/path": "path",
}
ALLOWED_BINDING_PATH = re.compile(r"^/verified_bindings/[^/]+/path$")


class AuditError(RuntimeError):
    """The audit could not establish a valid comparison."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except Exception as exc:
        raise AuditError(f"cannot parse {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{label} JSON root is not an object")
    return value


def write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    data = (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n").encode("utf-8")
    write_exclusive(path, data)


def resolve_path(value: str, repo: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def require_e01_exact(raw: Mapping[str, Any]) -> None:
    treatment = raw.get("treatment")
    if not (
        raw.get("schema") == RUN_SCHEMA
        and raw.get("design_id") == DESIGN_ID
        and raw.get("case_id") == CASE_ID
        and raw.get("subject") == SUBJECT
        and raw.get("semantic_evidence_eligible") is True
        and raw.get("apparatus_integration_only") is False
        and isinstance(treatment, Mapping)
        and treatment.get("case_id") == CASE_ID
        and treatment.get("design_id") == DESIGN_ID
    ):
        raise AuditError("input is not the single exact-subject e01 treatment run")


def bound_phase_a_path(
    raw: Mapping[str, Any], repo: Path, override: Path | None
) -> tuple[Path, str]:
    bindings = raw.get("bindings")
    row = bindings.get("phase_a_raw_artifact") if isinstance(bindings, Mapping) else None
    if not (
        isinstance(row, Mapping)
        and isinstance(row.get("path"), str)
        and is_sha256(row.get("sha256"))
    ):
        raise AuditError("treatment Phase-A raw binding is absent or malformed")
    path = override if override is not None else resolve_path(row["path"], repo)
    if not path.is_file():
        raise AuditError("bound Phase-A raw artifact is absent")
    observed = file_sha256(path)
    if observed != row["sha256"]:
        raise AuditError("Phase-A override/bound artifact hash differs")
    return path, observed


def phase_a_ff_records(phase_raw: Mapping[str, Any]) -> dict[str, Any]:
    phase = phase_raw.get("phase_a")
    scores = phase.get("scores") if isinstance(phase, Mapping) else None
    if not (
        phase_raw.get("schema") == PHASE_A_RUN_SCHEMA
        and phase_raw.get("design_id") == DESIGN_ID
        and phase_raw.get("case_id") == CASE_ID
        and phase_raw.get("subject") == SUBJECT
        and isinstance(scores, Mapping)
        and "FF_focal" in scores
        and "FF_nonfocal" in scores
    ):
        raise AuditError("Phase-A artifact lacks exact e01 FF focal/nonfocal records")
    return {"focal": scores["FF_focal"], "nonfocal": scores["FF_nonfocal"]}


def compare_fresh_to_phase_a(
    raw: Mapping[str, Any], phase_raw: Mapping[str, Any]
) -> dict[str, Any]:
    treatment = raw["treatment"]
    fresh = treatment.get("fresh_scores")
    if not (
        isinstance(fresh, Mapping)
        and set(fresh) == {"focal", "nonfocal"}
    ):
        raise AuditError("treatment fresh score pair is absent or malformed")
    phase_scores = phase_a_ff_records(phase_raw)
    probes: dict[str, Any] = {}
    for probe in ("focal", "nonfocal"):
        treatment_bytes = canonical_bytes(fresh[probe])
        phase_bytes = canonical_bytes(phase_scores[probe])
        probes[probe] = {
            "treatment_canonical_sha256": bytes_sha256(treatment_bytes),
            "phase_a_canonical_sha256": bytes_sha256(phase_bytes),
            "canonical_json_equal": treatment_bytes == phase_bytes,
        }
    return {
        "status": (
            "PASS" if all(row["canonical_json_equal"] for row in probes.values())
            else "INVALID"
        ),
        "comparison": "canonical JSON bytes",
        "probes": probes,
    }


def verify_tracked_harvester(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    repo = resolved.parents[1]
    try:
        relative = resolved.relative_to(repo).as_posix()
    except ValueError as exc:  # pragma: no cover - defensive
        raise AuditError("harvester path has no repository root") from exc
    expected_relative = "scripts/harvest_coherent_canary_v12.py"
    if relative != expected_relative:
        raise AuditError("refusing noncanonical harvester path")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    tracked = subprocess.run(
        ["git", "-C", str(repo), "show", f"HEAD:{relative}"],
        capture_output=True, check=False,
    )
    if head.returncode != 0 or tracked.returncode != 0 or not resolved.is_file():
        raise AuditError("tracked harvester cannot be resolved at HEAD")
    working_bytes = resolved.read_bytes()
    working_sha = bytes_sha256(working_bytes)
    head_sha = bytes_sha256(tracked.stdout)
    if working_sha != head_sha or working_sha != EXPECTED_HARVESTER_SHA256:
        raise AuditError("working/HEAD harvester differs from the pre-run frozen bytes")
    return {
        "path": str(resolved),
        "relative_path": relative,
        "git_head": head.stdout.strip(),
        "working_sha256": working_sha,
        "head_blob_sha256": head_sha,
        "pre_run_expected_sha256": EXPECTED_HARVESTER_SHA256,
        "working_matches_head": True,
        "working_matches_pre_run_freeze": True,
    }


def pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def diff_values(left: Any, right: Any, pointer: str = "") -> list[str]:
    if type(left) is not type(right):
        return [pointer or "/"]
    if isinstance(left, Mapping):
        differences: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{pointer}/{pointer_token(str(key))}"
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(diff_values(left[key], right[key], child))
        return differences
    if isinstance(left, list):
        differences = []
        if len(left) != len(right):
            differences.append(f"{pointer}/length")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(diff_values(
                left_item, right_item, f"{pointer}/{index}"))
        return differences
    return [] if left == right else [pointer or "/"]


def allowed_difference(pointer: str) -> str | None:
    if pointer in ALLOWED_LITERAL_DIFFERENCES:
        return ALLOWED_LITERAL_DIFFERENCES[pointer]
    if ALLOWED_BINDING_PATH.fullmatch(pointer):
        return "path"
    return None


def normalize_harvest(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(value))
    if "completed_at_utc" in normalized:
        normalized["completed_at_utc"] = "<HARVEST_TIMESTAMP>"
    treatment = normalized.get("treatment_artifact")
    if isinstance(treatment, dict) and "path" in treatment:
        treatment["path"] = "<TREATMENT_ARTIFACT_PATH>"
    bindings = normalized.get("verified_bindings")
    if isinstance(bindings, dict):
        for name, row in bindings.items():
            if isinstance(row, dict) and "path" in row:
                row["path"] = f"<BOUND_PATH:{name}>"
    return normalized


def validate_harvest_provenance_fields(
    value: Mapping[str, Any], label: str
) -> None:
    if not (
        value.get("schema") == HARVEST_SCHEMA
        and value.get("case_id") == CASE_ID
        and value.get("subject") == SUBJECT
    ):
        raise AuditError(f"{label} identity differs")
    stamp = value.get("completed_at_utc")
    if not isinstance(stamp, str):
        raise AuditError(f"{label} completion timestamp is not a string")
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AuditError(f"{label} completion timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise AuditError(f"{label} completion timestamp lacks a timezone")
    treatment = value.get("treatment_artifact")
    if not (
        isinstance(treatment, Mapping)
        and isinstance(treatment.get("path"), str)
        and bool(treatment["path"])
        and is_sha256(treatment.get("sha256"))
    ):
        raise AuditError(f"{label} treatment-artifact provenance differs")
    bindings = value.get("verified_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise AuditError(f"{label} verified bindings are absent")
    for name, row in bindings.items():
        if not (
            isinstance(name, str)
            and isinstance(row, Mapping)
            and isinstance(row.get("path"), str)
            and bool(row["path"])
            and is_sha256(row.get("sha256"))
        ):
            raise AuditError(f"{label} verified binding provenance differs: {name}")


def scalar_hash_at_pointer(value: Mapping[str, Any], pointer: str) -> str | None:
    current: Any = value
    try:
        for encoded in pointer.lstrip("/").split("/"):
            token = encoded.replace("~1", "/").replace("~0", "~")
            current = current[int(token)] if isinstance(current, list) else current[token]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return object_sha256(current)


def compare_harvests(
    pod: Mapping[str, Any], local: Mapping[str, Any]
) -> dict[str, Any]:
    validate_harvest_provenance_fields(pod, "pod harvest")
    validate_harvest_provenance_fields(local, "local harvest")
    raw_differences = diff_values(pod, local)
    classified = []
    unexpected = []
    for pointer in raw_differences:
        category = allowed_difference(pointer)
        if category is None:
            unexpected.append(pointer)
        else:
            classified.append({
                "pointer": pointer,
                "category": category,
                "pod_value_sha256": scalar_hash_at_pointer(pod, pointer),
                "local_value_sha256": scalar_hash_at_pointer(local, pointer),
            })
    normalized_pod = normalize_harvest(pod)
    normalized_local = normalize_harvest(local)
    normalized_equal = canonical_bytes(normalized_pod) == canonical_bytes(normalized_local)
    return {
        "status": "PASS" if normalized_equal and not unexpected else "INVALID",
        "raw_difference_pointers": raw_differences,
        "legitimate_path_or_timestamp_differences": classified,
        "unexpected_difference_pointers": unexpected,
        "pod_normalized_sha256": object_sha256(normalized_pod),
        "local_normalized_sha256": object_sha256(normalized_local),
        "normalized_canonical_json_equal": normalized_equal,
        "normalization_allowlist": [
            "/completed_at_utc",
            "/treatment_artifact/path",
            "/verified_bindings/<binding>/path",
        ],
    }


def run_harvester(
    *, harvester: Path, treatment: Path, repo: Path, output: Path,
    timeout_seconds: float,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
    command = [
        sys.executable, str(harvester),
        "--treatment", str(treatment),
        "--repo", str(repo),
        "--output", str(output),
    ]
    result = subprocess.run(
        command,
        cwd=str(harvester.resolve().parents[1]),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    report = load_object(output, "local harvest") if result.returncode == 0 else None
    return result, report


def audit(
    *, treatment_path: Path, pod_harvest_path: Path, repo: Path,
    work_dir: Path, phase_a_override: Path | None = None,
    harvester: Path = DEFAULT_HARVESTER, timeout_seconds: float = 120.0,
) -> tuple[dict[str, Any], int]:
    if work_dir.exists():
        raise AuditError("work directory already exists")
    work_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "ERROR",
        "started_at_utc": started,
        "case_id": CASE_ID,
        "subject": SUBJECT,
    }
    try:
        treatment_bytes = treatment_path.read_bytes()
        raw = load_object(treatment_path, "treatment artifact")
        require_e01_exact(raw)
        phase_path, phase_sha = bound_phase_a_path(raw, repo, phase_a_override)
        phase_raw = load_object(phase_path, "Phase-A raw artifact")
        pod_harvest = load_object(pod_harvest_path, "pod harvest")
        tracked = verify_tracked_harvester(harvester)
        report["inputs"] = {
            "treatment": {
                "path": str(treatment_path.resolve()),
                "sha256": bytes_sha256(treatment_bytes),
            },
            "phase_a_raw": {
                "path": str(phase_path.resolve()),
                "sha256": phase_sha,
            },
            "pod_harvest": {
                "path": str(pod_harvest_path.resolve()),
                "sha256": file_sha256(pod_harvest_path),
                "canonical_sha256": object_sha256(pod_harvest),
            },
            "tracked_harvester": tracked,
        }
        fresh_check = compare_fresh_to_phase_a(raw, phase_raw)
        report["fresh_phase_a_identity"] = fresh_check
        if fresh_check["status"] != "PASS":
            report["status"] = "INVALID"
            report["harvester_replay"] = {
                "status": "SKIPPED",
                "reason": "fresh/Phase-A canonical identity mismatch",
            }
            return report, 2

        reconstructed = work_dir / "reconstructed_treatment_raw.json"
        write_exclusive(reconstructed, treatment_bytes)
        reconstructed_object = load_object(reconstructed, "reconstructed treatment")
        reconstruction = {
            "path": str(reconstructed),
            "source_sha256": bytes_sha256(treatment_bytes),
            "reconstructed_sha256": file_sha256(reconstructed),
            "source_canonical_sha256": object_sha256(raw),
            "reconstructed_canonical_sha256": object_sha256(reconstructed_object),
            "byte_exact": reconstructed.read_bytes() == treatment_bytes,
            "canonical_json_equal": canonical_bytes(reconstructed_object) == canonical_bytes(raw),
        }
        report["raw_reconstruction"] = reconstruction
        if not reconstruction["byte_exact"] or not reconstruction["canonical_json_equal"]:
            raise AuditError("raw treatment reconstruction is not exact")

        local_harvest_path = work_dir / "local_harvest.json"
        process, local_harvest = run_harvester(
            harvester=harvester,
            treatment=reconstructed,
            repo=repo,
            output=local_harvest_path,
            timeout_seconds=timeout_seconds,
        )
        stdout_path = work_dir / "harvester.stdout.txt"
        stderr_path = work_dir / "harvester.stderr.txt"
        write_exclusive(stdout_path, process.stdout.encode("utf-8"))
        write_exclusive(stderr_path, process.stderr.encode("utf-8"))
        replay: dict[str, Any] = {
            "returncode": process.returncode,
            "stdout_sha256": file_sha256(stdout_path),
            "stderr_sha256": file_sha256(stderr_path),
            "local_harvest_path": str(local_harvest_path),
            "local_harvest_sha256": (
                file_sha256(local_harvest_path) if local_harvest_path.is_file() else None
            ),
        }
        report["harvester_replay"] = replay
        if process.returncode != 0 or local_harvest is None:
            report["status"] = "ERROR"
            replay["status"] = "ERROR"
            return report, 1
        replay["local_harvest_canonical_sha256"] = object_sha256(local_harvest)
        try:
            comparison = compare_harvests(pod_harvest, local_harvest)
        except AuditError as exc:
            comparison = {
                "status": "INVALID",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        replay["comparison"] = comparison
        replay["status"] = comparison["status"]
        report["status"] = comparison["status"]
        return report, 0 if comparison["status"] == "PASS" else 2
    except AuditError as exc:
        report["status"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        return report, 1
    except Exception as exc:  # fail closed and preserve a machine record
        report["status"] = "ERROR"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        return report, 1
    finally:
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", required=True, type=Path)
    parser.add_argument("--pod-harvest", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument("--phase-a", type=Path)
    parser.add_argument("--harvester", type=Path, default=DEFAULT_HARVESTER)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report, code = audit(
            treatment_path=args.treatment,
            pod_harvest_path=args.pod_harvest,
            repo=args.repo.resolve(),
            work_dir=args.work_dir,
            phase_a_override=args.phase_a,
            harvester=args.harvester,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        print(f"AUDIT_SETUP_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    report_path = args.work_dir / "audit_report.json"
    write_json_exclusive(report_path, report)
    report_sha = file_sha256(report_path)
    sidecar = {
        "schema": f"{REPORT_SCHEMA}_sha256_v1",
        "report_path": str(report_path),
        "report_sha256": report_sha,
    }
    sidecar_path = args.work_dir / "audit_report.sha256.json"
    write_json_exclusive(sidecar_path, sidecar)
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "report_sha256": report_sha,
        "sidecar": str(sidecar_path),
    }, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
