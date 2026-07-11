#!/usr/bin/env python3
"""Run one v12 pre-treatment case and save its complete raw Phase-A record."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coherent_canary_case import PHASE_A_SCHEMA, run_phase_a_case  # noqa: E402
from coherent_canary_loader import (  # noqa: E402
    EXACT_SPEC, LOCAL_SPEC, prepare_pinned_subject,
)
from coherent_canary_store import atomic_create_json, unique_output_path  # noqa: E402


DESIGN_ID = "coherent-state-decision-canary-v12"
SCHEMA = "coherent_state_decision_canary_v12_phase_a_run_v1"
TECHNICAL_REPORT_SCHEMA = "coherent_state_decision_canary_v12_technical_validation_v1"
SUBJECTS = {"local-apparatus": LOCAL_SPEC, "exact-subject": EXACT_SPEC}
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
DEFAULT_OUTPUT_DIR = ROOT / "results/coherent_canary_v12_phase_a"
EXPECTED_SCORE_KEYS = {
    "A_C_focal", "A_W_focal", "FF_focal",
    "A_C_nonfocal", "A_W_nonfocal", "FF_nonfocal",
}


class PhaseARunnerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PhaseARunnerError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path, *, repo: Path) -> dict[str, str]:
    require(path.is_file(), f"bound file absent: {path}")
    resolved = path.resolve()
    try:
        display = resolved.relative_to(repo.resolve()).as_posix()
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": file_sha256(path)}


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except Exception as exc:
        raise PhaseARunnerError(f"cannot parse {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def git_value(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True,
        check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def reviews_pass_for_case(*, case_id: str, blind: dict[str, Any],
                          paired: dict[str, Any]) -> bool:
    """Check the published review decisions at their actual schema fields."""
    blind_aggregate = blind.get("aggregate", {})
    paired_rows = [row for row in paired.get("paired_case_reviews", [])
                   if row.get("case_id") == case_id]
    return (
        blind.get("schema") ==
        "coherent_state_decision_canary_v12_blind_review_v1"
        and blind_aggregate.get("overall_verdict") == "PASS"
        and blind_aggregate.get("history_fail_count") == 0
        and blind_aggregate.get("history_revise_count") == 0
        and blind.get("shared_carrier_anchor_review", {}).get("verdict") ==
        "PASS"
        and paired.get("schema") ==
        "coherent_state_decision_canary_v12_paired_diversity_review_v1"
        and paired.get("overall_verdict") == "PASS"
        and len(paired_rows) == 1
        and paired_rows[0].get("verdict") == "PASS"
        and paired.get("cross_case_diversity_review", {}).get("verdict") ==
        "PASS"
    )


def compact_model_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    """Retain the inventory commitment without duplicating every tensor row."""
    result = {key: value for key, value in inventory.items()
              if key != "weight_tensors"}
    rows = inventory.get("weight_tensors")
    if isinstance(rows, list):
        result["weight_tensor_count"] = len(rows)
    return result


def verify_bound_inputs(*, subject: str, case: dict[str, Any], case_path: Path,
                        manifest: dict[str, Any], blind: dict[str, Any],
                        paired: dict[str, Any], technical: dict[str, Any]) -> None:
    require(case.get("design_id") == DESIGN_ID and isinstance(case.get("case_id"), str),
            "case design/id differs")
    case_rows = [row for row in manifest.get("cases", [])
                 if row.get("case_id") == case["case_id"]]
    require(len(case_rows) == 1 and
            case_rows[0].get("input_file_sha256") == file_sha256(case_path),
            "authoritative revision-4 manifest does not bind exact case bytes")
    require(reviews_pass_for_case(
        case_id=str(case["case_id"]), blind=blind, paired=paired),
        "authoritative reviews are not PASS for this case")
    require(technical.get("schema") == TECHNICAL_REPORT_SCHEMA and
            technical.get("design_id") == DESIGN_ID and
            technical.get("status") == "PASS" and
            technical.get("subject") == subject,
            "technical validation report subject/status differs")
    runtime_check = technical.get("checks", {}).get("runtime_fingerprint", {})
    runtime_evidence = runtime_check.get("evidence", {})
    require(runtime_check.get("passed") is True and
            isinstance(runtime_evidence.get("fingerprint_sha256"), str),
            "technical validation lacks stable runtime fingerprint PASS")
    if subject == "exact-subject":
        require(technical.get("semantic_release_eligible") is True,
                "exact subject technical report is not semantic-release eligible")


def assert_phase_a_boundary(payload: dict[str, Any], case_id: str) -> None:
    require(payload.get("schema") == PHASE_A_SCHEMA and
            payload.get("design_id") == DESIGN_ID and
            payload.get("case_id") == case_id,
            "Phase-A payload schema/design/case differs")
    require(payload.get("treatment_scores_present") is False,
            "Phase-A payload reports treatment scores")
    require(set(payload.get("scores", {})) == EXPECTED_SCORE_KEYS,
            "Phase-A score set is not oracle/fresh-only")
    require(set(payload.get("executions", {})) == {"C_N", "W_N", "F"},
            "Phase-A execution set includes treatment execution")
    require("treatment" not in payload and "treatment_source_rows" not in payload,
            "Phase-A payload contains treatment material")


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any], int]:
    require(args.subject in SUBJECTS, f"unknown subject: {args.subject}")
    require(args.hourly_cost_usd is None or args.hourly_cost_usd >= 0,
            "hourly cost must be nonnegative")
    case_slug = args.case.stem
    require(re.fullmatch(r"[A-Za-z0-9._-]+", case_slug) is not None,
            "case filename is not a safe output slug")
    output = unique_output_path(
        args.output_dir, f"coherent-canary-v12-phase-a-{case_slug}", args.subject)
    print(f"RUN coherent-canary-v12-phase-a subject={args.subject} "
          f"case={args.case} -> {output}", flush=True)
    started = time.monotonic()
    document: dict[str, Any] = {
        "schema": SCHEMA, "design_id": DESIGN_ID, "status": "STARTED",
        "subject": args.subject, "case_path": str(args.case),
        "output_path": str(output), "started_at_utc": utc_now(),
        "treatment_scores_present": False,
        "provenance": {
            "runner_python": platform.python_version(),
            "runner_platform": platform.platform(),
            "repository_head": git_value(args.repo, "rev-parse", "HEAD"),
            "repository_branch": git_value(args.repo, "branch", "--show-current"),
        },
    }
    exit_code = 1
    try:
        input_paths = {
            "case": args.case, "preregistration": args.prereg,
            "revision4_manifest": args.manifest,
            "revision4_blind_review": args.blind_review,
            "revision4_paired_review": args.paired_review,
            "technical_validation_report": args.technical_report,
        }
        document["bindings"] = {
            name: binding(path, repo=args.repo)
            for name, path in input_paths.items()}
        case = load_object(args.case)
        manifest = load_object(args.manifest)
        blind = load_object(args.blind_review)
        paired = load_object(args.paired_review)
        technical = load_object(args.technical_report)
        verify_bound_inputs(
            subject=args.subject, case=case, case_path=args.case,
            manifest=manifest, blind=blind, paired=paired, technical=technical)
        document["case_id"] = str(case["case_id"])
        document["case_input"] = case
        document["technical_validation"] = {
            "status": technical["status"], "subject": technical["subject"],
            "semantic_release_eligible": technical.get("semantic_release_eligible"),
            "runtime_fingerprint_sha256": technical["checks"][
                "runtime_fingerprint"]["evidence"]["fingerprint_sha256"],
        }

        load_started_at = utc_now()
        load_started = time.monotonic()
        prepared = prepare_pinned_subject(
            spec=SUBJECTS[args.subject], repo=args.repo,
            local_files_only=not args.allow_download)
        document["load_timing"] = {
            "started_at_utc": load_started_at,
            "completed_at_utc": utc_now(),
            "wall_time_seconds": time.monotonic() - load_started,
        }
        runtime = prepared["runtime_fingerprint"]
        require(runtime.get("fingerprint_sha256") == document[
            "technical_validation"]["runtime_fingerprint_sha256"],
            "prepared runtime fingerprint differs from technical validation")
        document["runtime_fingerprint"] = json_safe(runtime)
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
        }))
        phase_started_at = utc_now()
        phase_started = time.monotonic()
        phase_a = run_phase_a_case(
            prepared["model"], prepared["tokenizer"], case,
            eos_ids=[int(value) for value in runtime["eos_ids"]])
        assert_phase_a_boundary(phase_a, str(case["case_id"]))
        document["phase_a"] = phase_a
        document["phase_a_timing"] = {
            "started_at_utc": phase_started_at,
            "completed_at_utc": utc_now(),
            "wall_time_seconds": time.monotonic() - phase_started,
        }
        document["status"] = "PASS"
        exit_code = 0
    except Exception as exc:
        document["status"] = "ERROR"
        document["error"] = {
            "type": type(exc).__name__, "message": str(exc),
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
        # Assert the wrapper itself never re-labels pre-treatment evidence.
        document["treatment_scores_present"] = False
        atomic_create_json(output, json_safe(document))
    print(json.dumps({
        "status": document["status"],
        "case_id": document.get("case_input", {}).get("case_id"),
        "treatment_scores_present": document["treatment_scores_present"],
        "error": document.get("error", {}).get("message"),
    }, sort_keys=True), flush=True)
    return output, document, exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, choices=tuple(SUBJECTS))
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--technical-report", required=True, type=Path)
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
