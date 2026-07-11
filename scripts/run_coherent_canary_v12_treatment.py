#!/usr/bin/env python3
"""Run one physically separate v12 treatment case after Phase-A release.

This module deliberately does not import the treatment implementation at module
load time.  It first verifies a separately persisted subject-matched Phase-A
validation report and its bound raw artifact.  Only after that release check and
an identical pinned-runtime check does it import and call ``run_treatment_case``.
Local runs are apparatus integration only; only a released exact-subject run can
be labeled eligible semantic evidence.  A technically valid but
estimand-inadequate local Phase A may still exercise the treatment apparatus;
the exact subject may not.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coherent_canary_loader import (  # noqa: E402
    EXACT_SPEC, LOCAL_SPEC, prepare_pinned_subject,
)
from coherent_canary_store import atomic_create_json, unique_output_path  # noqa: E402


DESIGN_ID = "coherent-state-decision-canary-v12"
RUN_SCHEMA = "coherent_state_decision_canary_v12_treatment_run_v1"
TREATMENT_SCHEMA = "coherent_state_decision_canary_v12_treatment_raw_v1"
PHASE_A_RUN_SCHEMA = "coherent_state_decision_canary_v12_phase_a_run_v1"
PHASE_A_PAYLOAD_SCHEMA = "coherent_state_decision_canary_v12_phase_a_raw_v1"
PHASE_A_REPORT_SCHEMA = (
    "coherent_state_decision_canary_v12_phase_a_validation_v1")
TECHNICAL_REPORT_SCHEMA = (
    "coherent_state_decision_canary_v12_technical_validation_v1")
SUBJECTS = {"local-apparatus": LOCAL_SPEC, "exact-subject": EXACT_SPEC}
REGIONS = ("R1_content", "R2_boundary", "R3_anchor")
PRIMARY_CELLS = ("FF", "FC", "FW", "CF", "CC", "CW", "WF", "WC", "WW")
P_CELLS = ("CC", "WW", "FC", "FW")
EXPECTED_PRIMARY_SELECTORS = {
    *(('N', region, cell) for region in REGIONS for cell in PRIMARY_CELLS),
    *(('P', 'R2_boundary', cell) for cell in P_CELLS),
}

DEFAULT_PREREG = ROOT / "COHERENT-STATE-DECISION-CANARY-V12-PREREGISTRATION.md"
DEFAULT_MANIFEST = ROOT / (
    "results/coherent_canary_validation/"
    "coherent_canary_revision4_full_manifest_Qwen3-30B-A3B-Instruct-2507_"
    "20260711T205951Z.json")
DEFAULT_BLIND_REVIEW = ROOT / (
    "results/coherent_canary_reviews/reviews/"
    "revision4_blind_singleton_independent_codex_20260711T210013Z.json")
DEFAULT_PAIRED_REVIEW = ROOT / (
    "results/coherent_canary_reviews/reviews/"
    "revision4_paired_diversity_independent_codex_20260711T210013Z.json")
DEFAULT_OUTPUT_DIR = ROOT / "results/coherent_canary_v12_treatment"


class TreatmentRunnerError(RuntimeError):
    """The release, pinned runtime, or treatment payload differed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TreatmentRunnerError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except Exception as exc:
        raise TreatmentRunnerError(f"cannot parse {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def display_path(path: Path, repo: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def binding(path: Path, *, repo: Path) -> dict[str, str]:
    require(path.is_file(), f"bound file absent: {path}")
    return {"path": display_path(path, repo), "sha256": file_sha256(path)}


def resolve_bound_path(value: str, repo: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def git_value(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True,
        check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def compact_model_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Keep snapshot commitments without repeating thousands of tensor rows."""
    result = {str(key): json_safe(value) for key, value in inventory.items()
              if key != "weight_tensors"}
    rows = inventory.get("weight_tensors")
    if isinstance(rows, list):
        result["weight_tensor_count"] = len(rows)
    return result


def _review_passes(blind: Mapping[str, Any], paired: Mapping[str, Any],
                   case_id: str) -> None:
    require(
        blind.get("schema") ==
        "coherent_state_decision_canary_v12_blind_review_v1" and
        blind.get("aggregate", {}).get("overall_verdict") == "PASS" and
        blind.get("shared_carrier_anchor_review", {}).get("verdict") == "PASS",
        "authoritative blind/carrier review is not PASS")
    case_rows = [row for row in paired.get("paired_case_reviews", [])
                 if isinstance(row, Mapping) and row.get("case_id") == case_id]
    require(
        paired.get("schema") ==
        "coherent_state_decision_canary_v12_paired_diversity_review_v1" and
        paired.get("overall_verdict") == "PASS" and
        paired.get("cross_case_diversity_review", {}).get("verdict") == "PASS" and
        len(case_rows) == 1 and case_rows[0].get("verdict") == "PASS",
        "authoritative paired/diversity review is not PASS for the case")


def validate_release_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Validate all pre-treatment inputs without importing treatment code."""
    case = load_object(args.case)
    case_id = case.get("case_id")
    require(case.get("design_id") == DESIGN_ID and isinstance(case_id, str),
            "case design/id differs")

    manifest = load_object(args.manifest)
    manifest_rows = [row for row in manifest.get("cases", [])
                     if isinstance(row, Mapping) and row.get("case_id") == case_id]
    require(len(manifest_rows) == 1 and
            manifest_rows[0].get("input_file_sha256") == file_sha256(args.case),
            "authoritative manifest does not bind the exact case bytes")
    blind = load_object(args.blind_review)
    paired = load_object(args.paired_review)
    _review_passes(blind, paired, case_id)

    technical = load_object(args.technical_report)
    technical_runtime = technical.get("checks", {}).get(
        "runtime_fingerprint", {})
    technical_evidence = technical_runtime.get("evidence", {})
    technical_fingerprint = technical_evidence.get("fingerprint_sha256")
    require(
        technical.get("schema") == TECHNICAL_REPORT_SCHEMA and
        technical.get("design_id") == DESIGN_ID and
        technical.get("status") == "PASS" and
        technical.get("subject") == args.subject and
        technical_runtime.get("passed") is True and
        is_sha256(technical_fingerprint),
        "subject technical validation is not a PASS")
    expected_semantic_eligibility = args.subject == "exact-subject"
    require(technical.get("semantic_release_eligible") is
            expected_semantic_eligibility,
            "technical semantic eligibility differs from subject mode")

    phase_report = load_object(args.phase_a_report)
    phase_runtime = phase_report.get("checks", {}).get(
        "runtime_fingerprint", {})
    phase_evidence = phase_runtime.get("evidence", {})
    phase_fingerprint = phase_evidence.get("fingerprint_sha256")
    phase_status = phase_report.get("status")
    allowed_statuses = ({"PRETREATMENT_PASS"} if args.subject == "exact-subject"
                        else {"PRETREATMENT_PASS", "ESTIMAND_INADEQUATE"})
    inadequacy = phase_report.get("inadequacy_reasons")
    require(
        phase_report.get("schema") == PHASE_A_REPORT_SCHEMA and
        phase_report.get("design_id") == DESIGN_ID and
        phase_report.get("case_id") == case_id and
        phase_status in allowed_statuses and
        phase_report.get("subject") == args.subject and
        phase_runtime.get("passed") is True and
        phase_fingerprint == technical_fingerprint and
        not phase_report.get("invalidity_reasons") and
        isinstance(inadequacy, list) and
        ((phase_status == "PRETREATMENT_PASS" and not inadequacy) or
         (phase_status == "ESTIMAND_INADEQUATE" and bool(inadequacy))),
        "Phase-A report is not valid for the requested subject mode")
    require(phase_report.get("semantic_release_eligible") is
            expected_semantic_eligibility,
            "Phase-A semantic eligibility differs from subject mode")

    raw_binding = phase_report.get("raw_artifact")
    require(isinstance(raw_binding, Mapping) and
            isinstance(raw_binding.get("path"), str) and
            is_sha256(raw_binding.get("sha256")),
            "Phase-A report raw-artifact binding differs")
    phase_raw_path = resolve_bound_path(raw_binding["path"], args.repo)
    require(phase_raw_path.is_file() and
            file_sha256(phase_raw_path) == raw_binding["sha256"],
            "Phase-A raw artifact is absent or its bytes differ")
    phase_raw = load_object(phase_raw_path)
    runtime_fingerprint = phase_raw.get("runtime_fingerprint")
    phase_payload = phase_raw.get("phase_a")
    require(
        phase_raw.get("schema") == PHASE_A_RUN_SCHEMA and
        phase_raw.get("design_id") == DESIGN_ID and
        phase_raw.get("case_id") == case_id and
        phase_raw.get("subject") == args.subject and
        phase_raw.get("treatment_scores_present") is False and
        isinstance(phase_payload, Mapping) and
        phase_payload.get("schema") == PHASE_A_PAYLOAD_SCHEMA and
        phase_payload.get("design_id") == DESIGN_ID and
        phase_payload.get("case_id") == case_id and
        phase_payload.get("treatment_scores_present") is False and
        isinstance(runtime_fingerprint, Mapping) and
        runtime_fingerprint.get("fingerprint_sha256") == phase_fingerprint and
        runtime_fingerprint.get("subject_spec", {}).get("key") == args.subject,
        "bound Phase-A raw artifact differs from the released case/runtime")

    paths = {
        "case": args.case,
        "preregistration": args.prereg,
        "revision4_manifest": args.manifest,
        "revision4_blind_review": args.blind_review,
        "revision4_paired_review": args.paired_review,
        "technical_validation_report": args.technical_report,
        "phase_a_validation_report": args.phase_a_report,
        "phase_a_raw_artifact": phase_raw_path,
    }
    bindings = {name: binding(path, repo=args.repo)
                for name, path in paths.items()}
    phase_bindings = phase_raw.get("bindings")
    require(isinstance(phase_bindings, Mapping),
            "Phase-A raw artifact source bindings are absent")
    for name in (
            "case", "preregistration", "revision4_manifest",
            "revision4_blind_review", "revision4_paired_review",
            "technical_validation_report"):
        require(isinstance(phase_bindings.get(name), Mapping) and
                phase_bindings[name].get("sha256") == bindings[name]["sha256"],
                f"Phase-A raw artifact {name} binding differs")
    return {
        "case": case,
        "case_id": case_id,
        "bindings": bindings,
        "runtime_fingerprint": dict(runtime_fingerprint),
        "runtime_fingerprint_sha256": phase_fingerprint,
        "phase_a_release": {
            "status": phase_report["status"],
            "subject": phase_report["subject"],
            "estimand_adequate": phase_status == "PRETREATMENT_PASS",
            "inadequacy_reasons": list(inadequacy),
            "semantic_release_eligible": phase_report[
                "semantic_release_eligible"],
            "report_sha256": bindings["phase_a_validation_report"]["sha256"],
            "raw_artifact_sha256": bindings["phase_a_raw_artifact"]["sha256"],
        },
    }


def import_treatment_entrypoint() -> Callable[..., dict[str, Any]]:
    """Import treatment code only after the caller has validated release."""
    module = importlib.import_module("coherent_canary_case")
    function = getattr(module, "run_treatment_case", None)
    require(callable(function), "run_treatment_case entry point is absent")
    return function


def _selector(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return row.get("schedule"), row.get("region"), row.get("cell")


def assert_treatment_payload(payload: Mapping[str, Any], case_id: str) -> None:
    """Check the frozen 31-cell grid and three mandatory placebo controls."""
    require(payload.get("schema") == TREATMENT_SCHEMA and
            payload.get("design_id") == DESIGN_ID and
            payload.get("case_id") == case_id,
            "treatment payload schema/design/case differs")
    require(payload.get("phase_a_scores_present") is False and
            "phase_a" not in payload and "phase_a_scores" not in payload,
            "treatment payload includes Phase-A scores")
    arms = payload.get("arms")
    require(isinstance(arms, list), "treatment arm list is absent")
    primary_rows: list[Mapping[str, Any]] = []
    placebo_rows: list[Mapping[str, Any]] = []
    for row in arms:
        require(isinstance(row, Mapping), "treatment arm row is not an object")
        if _selector(row) in EXPECTED_PRIMARY_SELECTORS:
            primary_rows.append(row)
        elif (row.get("arm_kind") == "placebo_control" and
              row.get("cell") == "V_PLACEBO"):
            placebo_rows.append(row)
        else:
            raise TreatmentRunnerError(f"unexpected treatment selector: {_selector(row)}")

    primary_selectors = [_selector(row) for row in primary_rows]
    require(len(primary_rows) == 31 and
            len(set(primary_selectors)) == 31 and
            set(primary_selectors) == EXPECTED_PRIMARY_SELECTORS,
            "treatment payload does not contain exactly the frozen 31 primary cells")
    require(payload.get("primary_arm_count") == 31,
            "declared primary arm count differs")
    require(payload.get("arm_count") == len(arms) == 34,
            "declared arm count differs from persisted arms")
    require(all(isinstance(row.get("scores"), Mapping) and
                set(row["scores"]) == {"focal", "nonfocal"}
                for row in primary_rows),
            "a primary treatment cell lacks focal/nonfocal scores")

    require(len(placebo_rows) == 3 and
            {row.get("region") for row in placebo_rows} == set(REGIONS) and
            all(row.get("schedule") == "N" for row in placebo_rows) and
            payload.get("placebo_control_count") == 3,
            "treatment payload does not contain one N-schedule placebo per region")
    statuses = [row.get("control_status") for row in placebo_rows]
    require(all(status in {"AVAILABLE", "PLACEBO_UNAVAILABLE"}
                for status in statuses),
            "placebo control status differs")
    available = sum(status == "AVAILABLE" for status in statuses)
    require(payload.get("available_placebo_control_count") == available,
            "declared available placebo count differs")
    require(all(
        (isinstance(row.get("scores"), Mapping) and
         set(row["scores"]) == {"focal", "nonfocal"})
        if row["control_status"] == "AVAILABLE" else "scores" not in row
        for row in placebo_rows),
        "placebo score presence differs from availability")


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any], int]:
    require(args.subject in SUBJECTS, f"unknown subject: {args.subject}")
    require(args.hourly_cost_usd is None or args.hourly_cost_usd >= 0,
            "hourly cost must be nonnegative")
    case_slug = args.case.stem
    require(re.fullmatch(r"[A-Za-z0-9._-]+", case_slug) is not None,
            "case filename is not a safe output slug")
    output = unique_output_path(
        args.output_dir, f"coherent-canary-v12-treatment-{case_slug}",
        args.subject)
    print(f"RUN coherent-canary-v12-treatment subject={args.subject} "
          f"case={args.case} -> {output}", flush=True)

    started = time.monotonic()
    document: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "design_id": DESIGN_ID,
        "case_id": None,
        "status": "STARTED",
        "subject": args.subject,
        "output_path": str(output),
        "started_at_utc": utc_now(),
        "phase_a_scores_present": False,
        "semantic_evidence_eligible": False,
        "apparatus_integration_only": args.subject == "local-apparatus",
        "provenance": {
            "runner_python": platform.python_version(),
            "runner_platform": platform.platform(),
            "repository_head": git_value(args.repo, "rev-parse", "HEAD"),
            "repository_branch": git_value(args.repo, "branch", "--show-current"),
        },
    }
    exit_code = 1
    try:
        release = validate_release_inputs(args)
        document["case_id"] = release["case_id"]
        document["bindings"] = release["bindings"]
        document["phase_a_release"] = release["phase_a_release"]

        load_started = time.monotonic()
        load_started_at = utc_now()
        prepared = prepare_pinned_subject(
            spec=SUBJECTS[args.subject], repo=args.repo,
            local_files_only=not args.allow_download)
        document["load_timing"] = {
            "started_at_utc": load_started_at,
            "completed_at_utc": utc_now(),
            "wall_time_seconds": time.monotonic() - load_started,
        }
        prepared_fingerprint = json_safe(prepared["runtime_fingerprint"])
        require(prepared_fingerprint == json_safe(release["runtime_fingerprint"]),
                "prepared runtime fingerprint is not identical to released Phase A")
        document["runtime_fingerprint"] = prepared_fingerprint
        document["provenance"].update(json_safe({
            "repository_authorization": prepared["repository"],
            "dependencies": prepared["dependencies"],
            "device": prepared["device"],
            "model_snapshot": prepared["model_snapshot"],
            "protocol_tokenizer_snapshot": prepared["protocol_tokenizer_snapshot"],
            "model_inventory": compact_model_inventory(
                prepared["model_inventory"]),
            "protocol_tokenizer_inventory": prepared[
                "protocol_tokenizer_inventory"],
            "model_snapshot_contract": prepared.get("model_snapshot_contract"),
        }))

        # The treatment module becomes reachable only after release and runtime
        # equivalence have both been observed in this process.
        run_treatment_case = import_treatment_entrypoint()
        treatment_started = time.monotonic()
        treatment_started_at = utc_now()
        payload = run_treatment_case(
            prepared["model"], prepared["tokenizer"], release["case"],
            eos_ids=[int(value) for value in prepared_fingerprint["eos_ids"]])
        require(isinstance(payload, Mapping), "treatment payload is not an object")
        assert_treatment_payload(payload, release["case_id"])
        document["treatment"] = json_safe(payload)
        document["treatment_timing"] = {
            "started_at_utc": treatment_started_at,
            "completed_at_utc": utc_now(),
            "wall_time_seconds": time.monotonic() - treatment_started,
        }
        document["status"] = "PASS"
        document["semantic_evidence_eligible"] = (
            args.subject == "exact-subject")
        exit_code = 0
    except Exception as exc:
        document["status"] = "ERROR"
        document["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1
    finally:
        wall = time.monotonic() - started
        document["completed_at_utc"] = utc_now()
        document["wall_time_seconds"] = wall
        document["hourly_cost_usd"] = args.hourly_cost_usd
        document["estimated_cost_usd"] = (
            wall * args.hourly_cost_usd / 3600
            if args.hourly_cost_usd is not None else None)
        document["phase_a_scores_present"] = False
        atomic_create_json(output, json_safe(document))

    print(json.dumps({
        "status": document["status"],
        "case_id": document.get("case_id"),
        "primary_arm_count": document.get("treatment", {}).get(
            "primary_arm_count"),
        "placebo_control_count": document.get("treatment", {}).get(
            "placebo_control_count"),
        "error": document.get("error", {}).get("message"),
    }, sort_keys=True), flush=True)
    return output, document, exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, choices=tuple(SUBJECTS))
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--technical-report", required=True, type=Path)
    parser.add_argument("--phase-a-report", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--blind-review", type=Path, default=DEFAULT_BLIND_REVIEW)
    parser.add_argument("--paired-review", type=Path, default=DEFAULT_PAIRED_REVIEW)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--hourly-cost-usd", type=float)
    return parser.parse_args(argv)


def main() -> None:
    _, _, code = run(parse_args())
    raise SystemExit(code)


if __name__ == "__main__":
    main()
