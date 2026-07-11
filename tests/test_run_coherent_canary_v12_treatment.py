from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_coherent_canary_v12_treatment.py"
SPEC = importlib.util.spec_from_file_location(
    "run_coherent_canary_v12_treatment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def fingerprint(subject="exact-subject") -> dict:
    models = {
        "exact-subject": (
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"),
        "local-apparatus": (
            "Qwen/Qwen3-0.6B",
            "c1899de289a04d12100db370d81485cdf75e47ca"),
    }
    model_id, revision = models[subject]
    value = {
        "requested_model": model_id,
        "requested_revision": revision,
        "dtype": "torch.bfloat16",
        "attention_backend": "eager",
        "subject_spec": {"key": subject},
        "eos_ids": [9, 10],
    }
    value["fingerprint_sha256"] = hashlib.sha256(
        MODULE.canonical_bytes(value)).hexdigest()
    return value


def fixture(tmp_path: Path, *, subject="exact-subject",
            report_status="PRETREATMENT_PASS",
            release: bool | None = None) -> tuple[object, dict]:
    if release is None:
        release = subject == "exact-subject"
    case = {
        "schema": "coherent_state_decision_canary_v12_case_draft_v1",
        "design_id": MODULE.DESIGN_ID,
        "case_id": "e01",
        "middle_end_msg": 3,
        "variants": {"correct": {"messages": []},
                     "wrong_focal": {"messages": []}},
    }
    case_path = write_json(tmp_path / "data/cases/e01.json", case)
    manifest = write_json(tmp_path / "results/manifest.json", {
        "cases": [{"case_id": "e01",
                   "input_file_sha256": MODULE.file_sha256(case_path)}],
    })
    blind = write_json(tmp_path / "results/blind.json", {
        "schema": "coherent_state_decision_canary_v12_blind_review_v1",
        "aggregate": {"overall_verdict": "PASS"},
        "shared_carrier_anchor_review": {"verdict": "PASS"},
    })
    paired = write_json(tmp_path / "results/paired.json", {
        "schema": "coherent_state_decision_canary_v12_paired_diversity_review_v1",
        "overall_verdict": "PASS",
        "paired_case_reviews": [{"case_id": "e01", "verdict": "PASS"}],
        "cross_case_diversity_review": {"verdict": "PASS"},
    })
    prereg = tmp_path / "prereg.md"
    prereg.write_text("frozen preregistration\n")
    runtime = fingerprint(subject)
    technical = write_json(tmp_path / "results/technical.json", {
        "schema": MODULE.TECHNICAL_REPORT_SCHEMA,
        "design_id": MODULE.DESIGN_ID,
        "status": "PASS",
        "subject": subject,
        "semantic_release_eligible": subject == "exact-subject",
        "checks": {"runtime_fingerprint": {
            "passed": True,
            "evidence": {"fingerprint_sha256": runtime["fingerprint_sha256"]},
        }},
    })
    phase_bindings = {
        "case": MODULE.binding(case_path, repo=tmp_path),
        "preregistration": MODULE.binding(prereg, repo=tmp_path),
        "revision4_manifest": MODULE.binding(manifest, repo=tmp_path),
        "revision4_blind_review": MODULE.binding(blind, repo=tmp_path),
        "revision4_paired_review": MODULE.binding(paired, repo=tmp_path),
        "technical_validation_report": MODULE.binding(technical, repo=tmp_path),
    }
    phase_raw = write_json(tmp_path / "results/phase-a-raw.json", {
        "schema": MODULE.PHASE_A_RUN_SCHEMA,
        "design_id": MODULE.DESIGN_ID,
        "case_id": "e01",
        "subject": subject,
        "bindings": phase_bindings,
        "runtime_fingerprint": runtime,
        "treatment_scores_present": False,
        "phase_a": {
            "schema": MODULE.PHASE_A_PAYLOAD_SCHEMA,
            "design_id": MODULE.DESIGN_ID,
            "case_id": "e01",
            "treatment_scores_present": False,
            "scores": {"oracle-only": {}},
        },
    })
    phase_report = write_json(tmp_path / "results/phase-a-report.json", {
        "schema": MODULE.PHASE_A_REPORT_SCHEMA,
        "design_id": MODULE.DESIGN_ID,
        "case_id": "e01",
        "subject": subject,
        "status": report_status,
        "semantic_release_eligible": release,
        "raw_artifact": {
            "path": phase_raw.relative_to(tmp_path).as_posix(),
            "sha256": MODULE.file_sha256(phase_raw),
        },
        "checks": {"runtime_fingerprint": {
            "passed": True,
            "evidence": {"fingerprint_sha256": runtime["fingerprint_sha256"]},
        }},
        "invalidity_reasons": [],
        "inadequacy_reasons": (
            [] if report_status == "PRETREATMENT_PASS" else ["oracle"]),
    })
    args = MODULE.parse_args([
        "--subject", subject,
        "--case", str(case_path),
        "--technical-report", str(technical),
        "--phase-a-report", str(phase_report),
        "--repo", str(tmp_path),
        "--output-dir", str(tmp_path / "results/treatment"),
        "--prereg", str(prereg),
        "--manifest", str(manifest),
        "--blind-review", str(blind),
        "--paired-review", str(paired),
        "--hourly-cost-usd", "2.0",
    ])
    return args, runtime


def primary_arms() -> list[dict]:
    rows = []
    for schedule, region, cell in sorted(MODULE.EXPECTED_PRIMARY_SELECTORS):
        rows.append({
            "schedule": schedule,
            "region": region,
            "cell": cell,
            "scores": {"focal": {}, "nonfocal": {}},
        })
    return rows


def treatment_payload() -> dict:
    arms = primary_arms()
    placebos = []
    for index, region in enumerate(MODULE.REGIONS):
        available = index != 1
        row = {
            "arm_kind": "placebo_control",
            "schedule": "N",
            "region": region,
            "cell": "V_PLACEBO",
            "control_status": (
                "AVAILABLE" if available else "PLACEBO_UNAVAILABLE"),
        }
        if available:
            row["scores"] = {"focal": {}, "nonfocal": {}}
        placebos.append(row)
    payload = {
        "schema": MODULE.TREATMENT_SCHEMA,
        "design_id": MODULE.DESIGN_ID,
        "case_id": "e01",
        "plans": {},
        "source_executions": {},
        "fresh_execution": {},
        "fresh_scores": {},
        "arms": arms + placebos,
        "arm_count": 34,
        "primary_arm_count": 31,
        "placebo_control_count": 3,
        "available_placebo_control_count": 2,
        "phase_a_scores_present": False,
    }
    return payload


def install_mocks(monkeypatch, runtime: dict, *, payload=None,
                  events: list[str] | None = None) -> None:
    def prepare(**kwargs):
        if events is not None:
            events.append("prepare")
        assert kwargs["spec"].key == runtime["subject_spec"]["key"]
        return {
            "model": "model",
            "tokenizer": "tokenizer",
            "runtime_fingerprint": deepcopy(runtime),
            "repository": {"status": "FROZEN"},
            "dependencies": {"torch": "pinned"},
            "device": {"device_type": "cuda"},
            "model_snapshot": Path("/cache/model"),
            "protocol_tokenizer_snapshot": Path("/cache/tokenizer"),
            "model_inventory": {
                "sha256": "b" * 64,
                "weight_tensors": [{"name": f"w{i}"} for i in range(5)],
            },
            "protocol_tokenizer_inventory": {"sha256": "c" * 64},
            "model_snapshot_contract": {"sha256": "d" * 64},
        }

    def treatment(model, tokenizer, case, *, eos_ids):
        if events is not None:
            events.append("execute_treatment")
        assert eos_ids == runtime["eos_ids"]
        return deepcopy(payload or treatment_payload())

    def importer():
        if events is not None:
            events.append("import_treatment")
        return treatment

    monkeypatch.setattr(MODULE, "prepare_pinned_subject", prepare)
    monkeypatch.setattr(MODULE, "import_treatment_entrypoint", importer)


def test_release_precedes_model_load_and_treatment_import_and_full_raw_is_saved(
        tmp_path, monkeypatch):
    args, runtime = fixture(tmp_path)
    events: list[str] = []
    original_validate = MODULE.validate_release_inputs

    def observed_validate(value):
        result = original_validate(value)
        events.append("release_validated")
        return result

    monkeypatch.setattr(MODULE, "validate_release_inputs", observed_validate)
    install_mocks(monkeypatch, runtime, events=events)
    output, document, code = MODULE.run(args)
    persisted = json.loads(output.read_text())
    assert code == 0
    assert persisted["status"] == document["status"] == "PASS"
    assert persisted["schema"] == MODULE.RUN_SCHEMA
    assert events == [
        "release_validated", "prepare", "import_treatment", "execute_treatment"]
    assert persisted["subject"] == "exact-subject"
    assert persisted["semantic_evidence_eligible"] is True
    assert persisted["apparatus_integration_only"] is False
    assert persisted["phase_a_scores_present"] is False
    treatment = persisted["treatment"]
    assert treatment["schema"] == MODULE.TREATMENT_SCHEMA
    assert treatment["arm_count"] == 34
    assert treatment["primary_arm_count"] == 31
    assert treatment["placebo_control_count"] == 3
    assert treatment["available_placebo_control_count"] == 2
    assert len(treatment["arms"]) == 34
    assert persisted["runtime_fingerprint"] == runtime
    assert set(persisted["bindings"]) == {
        "case", "preregistration", "revision4_manifest",
        "revision4_blind_review", "revision4_paired_review",
        "technical_validation_report", "phase_a_validation_report",
        "phase_a_raw_artifact",
    }
    assert persisted["provenance"]["model_inventory"][
        "weight_tensor_count"] == 5
    assert "weight_tensors" not in persisted["provenance"]["model_inventory"]
    assert persisted["estimated_cost_usd"] is not None


def test_unreleased_phase_a_is_saved_as_error_without_loading_or_importing(
        tmp_path, monkeypatch):
    args, runtime = fixture(
        tmp_path, report_status="ESTIMAND_INADEQUATE", release=False)

    def forbidden(**kwargs):
        raise AssertionError("model load must not happen")

    monkeypatch.setattr(MODULE, "prepare_pinned_subject", forbidden)
    monkeypatch.setattr(
        MODULE, "import_treatment_entrypoint",
        lambda: (_ for _ in ()).throw(AssertionError("treatment import must not happen")))
    output, document, code = MODULE.run(args)
    persisted = json.loads(output.read_text())
    assert code == 1
    assert persisted["status"] == document["status"] == "ERROR"
    assert "PRETREATMENT_PASS" in persisted["error"]["message"]
    assert persisted["phase_a_scores_present"] is False
    assert "treatment" not in persisted


def test_local_pretreatment_pass_runs_as_apparatus_integration_only(
        tmp_path, monkeypatch):
    args, runtime = fixture(tmp_path, subject="local-apparatus")
    install_mocks(monkeypatch, runtime)
    output, document, code = MODULE.run(args)
    persisted = json.loads(output.read_text())
    assert code == 0
    assert document["status"] == persisted["status"] == "PASS"
    assert persisted["subject"] == "local-apparatus"
    assert persisted["phase_a_release"]["semantic_release_eligible"] is False
    assert persisted["semantic_evidence_eligible"] is False
    assert persisted["apparatus_integration_only"] is True
    assert persisted["treatment"]["arm_count"] == 34


def test_phase_a_raw_hash_mismatch_fails_before_model_load(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    report = json.loads(args.phase_a_report.read_text())
    raw = tmp_path / report["raw_artifact"]["path"]
    raw.write_text(raw.read_text() + " ")
    monkeypatch.setattr(
        MODULE, "prepare_pinned_subject",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not load")))
    output, document, code = MODULE.run(args)
    assert code == 1
    assert "bytes differ" in document["error"]["message"]
    assert json.loads(output.read_text())["status"] == "ERROR"


def test_prepared_runtime_must_be_identical_before_treatment_import(
        tmp_path, monkeypatch):
    args, runtime = fixture(tmp_path)
    different = deepcopy(runtime)
    different["eos_ids"] = [99]
    imported = []
    install_mocks(monkeypatch, different, events=imported)
    output, document, code = MODULE.run(args)
    assert code == 1
    assert "not identical" in document["error"]["message"]
    assert imported == ["prepare"]
    assert json.loads(output.read_text())["status"] == "ERROR"


def test_primary_grid_or_placebo_geometry_difference_is_saved_as_error(
        tmp_path, monkeypatch):
    args, runtime = fixture(tmp_path)
    payload = treatment_payload()
    payload["arms"][-1]["region"] = "R2_boundary"
    install_mocks(monkeypatch, runtime, payload=payload)
    output, document, code = MODULE.run(args)
    assert code == 1
    assert "one N-schedule placebo per region" in document["error"]["message"]
    persisted = json.loads(output.read_text())
    assert persisted["status"] == "ERROR"
    assert "treatment" not in persisted


def test_missing_placebo_is_rejected(tmp_path, monkeypatch):
    args, runtime = fixture(tmp_path)
    payload = treatment_payload()
    payload["arms"].pop()
    payload["arm_count"] = 33
    payload["placebo_control_count"] = 2
    install_mocks(monkeypatch, runtime, payload=payload)
    output, document, code = MODULE.run(args)
    assert code == 1
    assert "declared arm count" in document["error"]["message"]
    assert json.loads(output.read_text())["status"] == "ERROR"


def test_module_has_no_static_treatment_import_and_cli_requires_release_inputs():
    tree = ast.parse(SCRIPT.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "coherent_canary_case" not in imported
    with pytest.raises(SystemExit):
        MODULE.parse_args([])
    with pytest.raises(SystemExit):
        MODULE.parse_args([
            "--subject", "exact-subject", "--case", "e01.json",
            "--technical-report", "technical.json"])
