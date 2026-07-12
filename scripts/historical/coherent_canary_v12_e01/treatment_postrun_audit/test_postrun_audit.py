from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("postrun_audit.py")
TRACKED_FIXTURES = ROOT / "tests/test_harvest_coherent_canary_v12.py"
REAL_PHASE_A = ROOT / (
    "results/coherent_canary_v12_phase_a/"
    "coherent-canary-v12-phase-a-e01_exact-subject_20260712T030639073087Z.json"
)
REAL_PHASE_A_SHA256 = (
    "2cd7f6f190b9f4e9598844e45e5488779b72cd54a1fd51dd5b8a5b24154d672d"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = load_module("sol_treatment_postrun_audit", SCRIPT)
FIXTURE = load_module("sol_tracked_v12_harvest_fixture", TRACKED_FIXTURES)


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def build_valid_fixture(tmp_path: Path) -> dict[str, Path]:
    args, runtime = FIXTURE.release_fixture(tmp_path)
    phase_report = json.loads(args.phase_a_report.read_text())
    phase_raw_path = tmp_path / phase_report["raw_artifact"]["path"]
    phase_raw = json.loads(phase_raw_path.read_text())
    treatment_payload = FIXTURE.treatment_payload()

    # The older tracked synthetic harvester fixture predates the cross-phase
    # nonfocal identity check. Add the production Phase-A field explicitly.
    phase_raw["phase_a"]["scores"]["FF_focal"] = deepcopy(
        treatment_payload["fresh_scores"]["focal"])
    phase_raw["phase_a"]["scores"]["FF_nonfocal"] = deepcopy(
        treatment_payload["fresh_scores"]["nonfocal"])
    write_json(phase_raw_path, phase_raw)
    phase_report["raw_artifact"]["sha256"] = FIXTURE.HARVEST.file_sha256(
        phase_raw_path)
    write_json(args.phase_a_report, phase_report)

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
    bindings = {
        name: FIXTURE.RUNNER.binding(path, repo=tmp_path)
        for name, path in paths.items()
    }
    treatment = write_json(tmp_path / "results/treatment/e01-raw.json", {
        "schema": FIXTURE.HARVEST.RUN_SCHEMA,
        "design_id": FIXTURE.HARVEST.DESIGN_ID,
        "case_id": "e01",
        "subject": "exact-subject",
        "status": "PASS",
        "phase_a_scores_present": False,
        "semantic_evidence_eligible": True,
        "apparatus_integration_only": False,
        "runtime_fingerprint": runtime,
        "bindings": bindings,
        "treatment": treatment_payload,
    })

    pod_report = FIXTURE.HARVEST.harvest(treatment, repo_root=tmp_path)
    pod_report["completed_at_utc"] = "2026-07-12T00:00:00+00:00"
    pod_report["treatment_artifact"]["path"] = "/pod/repo/e01-raw.json"
    for name, row in pod_report["verified_bindings"].items():
        row["path"] = f"/pod/repo/bound/{name}"
    pod_harvest = write_json(tmp_path / "results/pod-harvest.json", pod_report)
    return {
        "repo": tmp_path,
        "treatment": treatment,
        "phase_a": phase_raw_path,
        "pod_harvest": pod_harvest,
    }


def test_cli_emits_hashed_pass_report_for_only_path_timestamp_differences(
        tmp_path: Path):
    fixture = build_valid_fixture(tmp_path)
    work = tmp_path / "audit-pass"
    code = AUDIT.main([
        "--repo", str(fixture["repo"]),
        "--treatment", str(fixture["treatment"]),
        "--pod-harvest", str(fixture["pod_harvest"]),
        "--work-dir", str(work),
    ])
    assert code == 0
    report_path = work / "audit_report.json"
    sidecar_path = work / "audit_report.sha256.json"
    report = json.loads(report_path.read_text())
    sidecar = json.loads(sidecar_path.read_text())
    assert report["status"] == "PASS"
    assert report["fresh_phase_a_identity"]["status"] == "PASS"
    assert report["raw_reconstruction"]["byte_exact"] is True
    comparison = report["harvester_replay"]["comparison"]
    assert comparison["status"] == "PASS"
    assert comparison["unexpected_difference_pointers"] == []
    assert comparison["normalized_canonical_json_equal"] is True
    assert all(
        row["category"] in {"path", "timestamp"}
        for row in comparison["legitimate_path_or_timestamp_differences"]
    )
    assert sidecar["report_sha256"] == AUDIT.file_sha256(report_path)
    assert len(report["inputs"]["tracked_harvester"]["working_sha256"]) == 64


def test_nonfocal_fresh_phase_a_mismatch_invalidates_before_harvest(
        tmp_path: Path):
    fixture = build_valid_fixture(tmp_path)
    raw = json.loads(fixture["treatment"].read_text())
    raw["treatment"]["fresh_scores"]["nonfocal"]["correct"][
        "mean_logprob"] = -123456.0
    write_json(fixture["treatment"], raw)
    report, code = AUDIT.audit(
        treatment_path=fixture["treatment"],
        pod_harvest_path=fixture["pod_harvest"],
        repo=fixture["repo"],
        work_dir=tmp_path / "audit-fresh-mismatch",
    )
    assert code == 2
    assert report["status"] == "INVALID"
    assert report["fresh_phase_a_identity"]["probes"]["focal"][
        "canonical_json_equal"] is True
    assert report["fresh_phase_a_identity"]["probes"]["nonfocal"][
        "canonical_json_equal"] is False
    assert report["harvester_replay"]["status"] == "SKIPPED"
    assert not (tmp_path / "audit-fresh-mismatch/local_harvest.json").exists()


def test_any_nonallowlisted_harvest_difference_invalidates(tmp_path: Path):
    fixture = build_valid_fixture(tmp_path)
    pod = json.loads(fixture["pod_harvest"].read_text())
    pod["R2_primary_estimands"]["N"]["full_KV"]["D_focal"] += 1.0
    write_json(fixture["pod_harvest"], pod)
    report, code = AUDIT.audit(
        treatment_path=fixture["treatment"],
        pod_harvest_path=fixture["pod_harvest"],
        repo=fixture["repo"],
        work_dir=tmp_path / "audit-harvest-mismatch",
    )
    assert code == 2
    assert report["status"] == "INVALID"
    comparison = report["harvester_replay"]["comparison"]
    assert comparison["normalized_canonical_json_equal"] is False
    assert (
        "/R2_primary_estimands/N/full_KV/D_focal"
        in comparison["unexpected_difference_pointers"]
    )


def test_phase_a_override_must_match_treatment_binding(tmp_path: Path):
    fixture = build_valid_fixture(tmp_path)
    wrong = write_json(tmp_path / "wrong-phase-a.json", {"wrong": True})
    report, code = AUDIT.audit(
        treatment_path=fixture["treatment"],
        pod_harvest_path=fixture["pod_harvest"],
        repo=fixture["repo"],
        phase_a_override=wrong,
        work_dir=tmp_path / "audit-phase-hash-error",
    )
    assert code == 1
    assert report["status"] == "ERROR"
    assert "hash differs" in report["error"]["message"]


def test_allowlisted_timestamp_must_still_be_a_real_zoned_timestamp(
        tmp_path: Path):
    fixture = build_valid_fixture(tmp_path)
    pod = json.loads(fixture["pod_harvest"].read_text())
    pod["completed_at_utc"] = "not-a-timestamp"
    write_json(fixture["pod_harvest"], pod)
    report, code = AUDIT.audit(
        treatment_path=fixture["treatment"],
        pod_harvest_path=fixture["pod_harvest"],
        repo=fixture["repo"],
        work_dir=tmp_path / "audit-bad-timestamp",
    )
    assert code == 2
    assert report["status"] == "INVALID"
    comparison = report["harvester_replay"]["comparison"]
    assert comparison["status"] == "INVALID"
    assert "timestamp is invalid" in comparison["error"]["message"]


def test_committed_phase_a_fixture_has_both_fresh_score_sources():
    assert AUDIT.file_sha256(REAL_PHASE_A) == REAL_PHASE_A_SHA256
    phase_raw = AUDIT.load_object(REAL_PHASE_A, "committed Phase-A fixture")
    records = AUDIT.phase_a_ff_records(phase_raw)
    first = {name: AUDIT.object_sha256(value) for name, value in records.items()}
    second = {name: AUDIT.object_sha256(value) for name, value in records.items()}
    assert first == second
    assert set(first) == {"focal", "nonfocal"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in first.values())
