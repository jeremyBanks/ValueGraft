from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HARVEST = load_script(
    "harvest_coherent_canary_v12_test",
    ROOT / "scripts/harvest_coherent_canary_v12.py")
RUNNER = load_script(
    "run_coherent_canary_v12_treatment_integration_test",
    ROOT / "scripts/run_coherent_canary_v12_treatment.py")


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def bits(value: float) -> str:
    return struct.pack("<f", float(value)).hex()


def runtime_fingerprint(subject: str = "exact-subject") -> dict:
    expected = HARVEST.SUBJECTS[subject]
    value = {
        "requested_model": expected["model_id"],
        "requested_revision": expected["revision"],
        "dtype": "torch.bfloat16",
        "attention_backend": "eager",
        "subject_spec": {"key": subject},
        "eos_ids": [9, 10],
    }
    value["fingerprint_sha256"] = hashlib.sha256(
        HARVEST.canonical_bytes(value)).hexdigest()
    return value


def target(token_id: int, mean: float) -> dict:
    return {
        "target_token_ids": [token_id],
        "token_logprobs": [8123.0],
        "token_logprob_float32_bits": [bits(mean)],
        "mean_logprob": 9123.0,
        "mean_logprob_float32_bits": bits(mean),
    }


def score(probe: str, correct_mean: float, counter_mean: float) -> dict:
    correct_id, counter_id = ((101, 102) if probe == "focal" else (201, 202))
    margin = HARVEST.float32(correct_mean - counter_mean)
    return {
        "correct": target(correct_id, correct_mean),
        "counterfactual": target(counter_id, counter_mean),
        "margin": -99999.0,
        "margin_float32_bits": bits(margin),
    }


def score_pair(focal: float, nonfocal: float) -> dict:
    return {
        "focal": score("focal", focal, 0.0),
        "nonfocal": score("nonfocal", nonfocal, 0.0),
    }


def treatment_payload() -> dict:
    fresh = score_pair(0.2, 0.15)
    focal = {
        ("N", "R2_boundary", "CC"): 1.5,
        ("N", "R2_boundary", "WW"): 0.5,
        ("N", "R2_boundary", "FC"): 1.2,
        ("N", "R2_boundary", "FW"): 0.4,
        ("P", "R2_boundary", "CC"): 1.4,
        ("P", "R2_boundary", "WW"): 0.45,
        ("P", "R2_boundary", "FC"): 1.1,
        ("P", "R2_boundary", "FW"): 0.35,
    }
    nonfocal = {
        ("N", "R2_boundary", "CC"): 0.2,
        ("N", "R2_boundary", "WW"): 0.1,
        ("N", "R2_boundary", "FC"): 0.18,
        ("N", "R2_boundary", "FW"): 0.08,
        ("P", "R2_boundary", "CC"): 0.19,
        ("P", "R2_boundary", "WW"): 0.1,
        ("P", "R2_boundary", "FC"): 0.17,
        ("P", "R2_boundary", "FW"): 0.08,
    }
    arms = []
    for selector in sorted(HARVEST.EXPECTED_PRIMARY_SELECTORS):
        schedule, region, cell = selector
        pair = (deepcopy(fresh) if cell == "FF" else score_pair(
            focal.get(selector, 0.3), nonfocal.get(selector, 0.11)))
        arms.append({
            "arm_kind": "primary", "schedule": schedule,
            "region": region, "cell": cell, "scores": pair,
        })
    for region in HARVEST.REGIONS:
        status = "PLACEBO_UNAVAILABLE" if region == "R3_anchor" else "AVAILABLE"
        row = {
            "arm_kind": "placebo_control", "schedule": "N",
            "region": region, "cell": HARVEST.PLACEBO_CELL,
            "control_status": status,
            "diagnostics": {
                "status": status, "row_count": 4,
                "attempt_count": 3, "zero_delta_count": 1,
                "max_applied_relative_norm_error": 0.01,
                "max_applied_abs_cosine": 0.005,
                "canonical_diagnostics_sha256": "d" * 64,
            },
        }
        if status == "AVAILABLE":
            row["scores"] = score_pair(0.25, 0.12)
        arms.append(row)
    return {
        "schema": HARVEST.TREATMENT_SCHEMA,
        "design_id": HARVEST.DESIGN_ID,
        "case_id": "e01",
        "plans": {}, "source_executions": {}, "fresh_execution": {},
        "fresh_scores": fresh,
        "arms": arms,
        "arm_count": 34,
        "primary_arm_count": 31,
        "placebo_control_count": 3,
        "available_placebo_control_count": 2,
        "phase_a_scores_present": False,
        "aggregates": {"runner_claimed_D_focal": -123456.0},
    }


def release_fixture(tmp_path: Path, *, subject: str = "exact-subject",
                    phase_status: str = "PRETREATMENT_PASS"):
    case = write_json(tmp_path / "data/e01.json", {
        "schema": "coherent_state_decision_canary_v12_case_draft_v1",
        "design_id": HARVEST.DESIGN_ID,
        "case_id": "e01",
    })
    manifest = write_json(tmp_path / "results/manifest.json", {
        "cases": [{"case_id": "e01",
                   "input_file_sha256": HARVEST.file_sha256(case)}],
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
    runtime = runtime_fingerprint(subject)
    semantic_eligible = HARVEST.SUBJECTS[subject]["semantic_evidence_eligible"]
    technical = write_json(tmp_path / "results/technical.json", {
        "schema": HARVEST.TECHNICAL_REPORT_SCHEMA,
        "design_id": HARVEST.DESIGN_ID,
        "status": "PASS", "subject": subject,
        "semantic_release_eligible": semantic_eligible,
        "checks": {"runtime_fingerprint": {
            "passed": True,
            "evidence": {"fingerprint_sha256": runtime["fingerprint_sha256"]},
        }},
    })
    phase_bindings = {
        "case": RUNNER.binding(case, repo=tmp_path),
        "preregistration": RUNNER.binding(prereg, repo=tmp_path),
        "revision4_manifest": RUNNER.binding(manifest, repo=tmp_path),
        "revision4_blind_review": RUNNER.binding(blind, repo=tmp_path),
        "revision4_paired_review": RUNNER.binding(paired, repo=tmp_path),
        "technical_validation_report": RUNNER.binding(technical, repo=tmp_path),
    }
    phase_raw = write_json(tmp_path / "results/phase-a-raw.json", {
        "schema": RUNNER.PHASE_A_RUN_SCHEMA,
        "design_id": HARVEST.DESIGN_ID,
        "case_id": "e01", "subject": subject,
        "bindings": phase_bindings,
        "runtime_fingerprint": runtime,
        "treatment_scores_present": False,
        "phase_a": {
            "schema": RUNNER.PHASE_A_PAYLOAD_SCHEMA,
            "design_id": HARVEST.DESIGN_ID,
            "case_id": "e01", "treatment_scores_present": False,
            "scores": {"oracle-only": {}},
        },
    })
    phase_report = write_json(tmp_path / "results/phase-a-report.json", {
        "schema": HARVEST.PHASE_A_REPORT_SCHEMA,
        "design_id": HARVEST.DESIGN_ID,
        "case_id": "e01", "subject": subject,
        "status": phase_status,
        "semantic_release_eligible": semantic_eligible,
        "raw_artifact": {
            "path": phase_raw.relative_to(tmp_path).as_posix(),
            "sha256": HARVEST.file_sha256(phase_raw),
        },
        "checks": {"runtime_fingerprint": {
            "passed": True,
            "evidence": {"fingerprint_sha256": runtime["fingerprint_sha256"]},
        }},
        "invalidity_reasons": [],
        "inadequacy_reasons": (
            [] if phase_status == "PRETREATMENT_PASS" else ["oracle"]),
    })
    args = RUNNER.parse_args([
        "--subject", subject,
        "--case", str(case),
        "--technical-report", str(technical),
        "--phase-a-report", str(phase_report),
        "--repo", str(tmp_path),
        "--output-dir", str(tmp_path / "results/treatment"),
        "--prereg", str(prereg),
        "--manifest", str(manifest),
        "--blind-review", str(blind),
        "--paired-review", str(paired),
    ])
    return args, runtime


def run_synthetic_treatment(tmp_path: Path, monkeypatch, *,
                            subject: str = "exact-subject",
                            phase_status: str = "PRETREATMENT_PASS") -> Path:
    args, runtime = release_fixture(
        tmp_path, subject=subject, phase_status=phase_status)

    def prepare(**kwargs):
        assert kwargs["spec"].key == subject
        return {
            "model": "model", "tokenizer": "tokenizer",
            "runtime_fingerprint": deepcopy(runtime),
            "repository": {"status": "FROZEN"},
            "dependencies": {"torch": "pinned"},
            "device": {"device_type": "cuda"},
            "model_snapshot": Path("/cache/model"),
            "protocol_tokenizer_snapshot": Path("/cache/tokenizer"),
            "model_inventory": {"weight_tensors": [{"name": "w"}]},
            "protocol_tokenizer_inventory": {"sha256": "c" * 64},
            "model_snapshot_contract": {"sha256": "d" * 64},
        }

    payload = treatment_payload()
    monkeypatch.setattr(RUNNER, "prepare_pinned_subject", prepare)
    monkeypatch.setattr(
        RUNNER, "import_treatment_entrypoint",
        lambda: (lambda model, tokenizer, case, *, eos_ids: deepcopy(payload)))
    output, document, code = RUNNER.run(args)
    assert code == 0
    assert document["status"] == "PASS"
    return output


def test_actual_treatment_runner_output_flows_directly_into_harvester(
        tmp_path, monkeypatch):
    treatment = run_synthetic_treatment(tmp_path, monkeypatch)
    report = HARVEST.harvest(treatment, repo_root=tmp_path)

    assert report["schema"] == HARVEST.REPORT_SCHEMA
    assert report["case_id"] == "e01"
    assert report["runtime"]["requested_model"] == (
        HARVEST.SUBJECTS["exact-subject"]["model_id"])
    assert report["semantic_evidence_eligible"] is True
    assert report["apparatus_integration_only"] is False
    full_n = report["R2_primary_estimands"]["N"]["full_KV"]
    value_n = report["R2_primary_estimands"]["N"]["value_only"]
    assert full_n == pytest.approx({
        "D_focal": 1.0, "D_nonfocal": 0.1, "SEL": 0.9,
        "Hplus": 1.0, "U": 1.3, "Uplus": 1.3,
    })
    assert value_n["D_focal"] == pytest.approx(0.8)
    assert value_n["SEL"] == pytest.approx(0.7)
    full_schedule = report["schedule_yardstick_components"]["full_KV"]
    assert full_schedule["D_focal_N"] == pytest.approx(1.0)
    assert full_schedule["D_focal_P"] == pytest.approx(0.95)
    assert full_schedule["absolute_N_minus_P"] == pytest.approx(0.05)
    assert report["regional_N_estimands"]["R2_boundary"]["full_KV"] == \
        pytest.approx(full_n)
    assert set(report["descriptive_cell_scores"]["N"]) == set(HARVEST.REGIONS)
    assert report["placebo_controls"]["R1_content"]["status"] == "AVAILABLE"
    assert report["placebo_controls"]["R1_content"][
        "movement_from_fresh"]["focal"]["margin_delta"] == pytest.approx(0.05)
    assert report["placebo_controls"]["R3_anchor"]["scores"] is None
    assert report["available_placebo_control_count"] == 2
    assert report["runner_status_labels_ignored"] is True
    assert report["runner_decimal_scores_and_aggregates_ignored"] is True
    assert report["inference"]["p_values_computed"] is False


def test_bound_file_hash_tampering_is_rejected(tmp_path, monkeypatch):
    treatment_path = run_synthetic_treatment(tmp_path, monkeypatch)
    treatment = json.loads(treatment_path.read_text())
    treatment["bindings"]["case"]["sha256"] = "0" * 64
    write_json(treatment_path, treatment)
    with pytest.raises(HARVEST.HarvestError, match="case bound file hash differs"):
        HARVEST.harvest(treatment_path, repo_root=tmp_path)


def test_runtime_continuity_difference_is_rejected(tmp_path, monkeypatch):
    treatment_path = run_synthetic_treatment(tmp_path, monkeypatch)
    treatment = json.loads(treatment_path.read_text())
    treatment["runtime_fingerprint"]["requested_revision"] = "wrong"
    write_json(treatment_path, treatment)
    with pytest.raises(HARVEST.HarvestError, match="not identical"):
        HARVEST.harvest(treatment_path, repo_root=tmp_path)


def test_margin_is_recomputed_from_bits_not_runner_decimals(tmp_path, monkeypatch):
    treatment_path = run_synthetic_treatment(tmp_path, monkeypatch)
    report = HARVEST.harvest(treatment_path, repo_root=tmp_path)
    score_row = report["descriptive_cell_scores"]["N"][
        "R2_boundary"]["CC"]["focal"]
    assert score_row["correct"]["mean_logprob"] == pytest.approx(1.5)
    assert score_row["margin"] == pytest.approx(1.5)

    treatment = json.loads(treatment_path.read_text())
    cc = next(row for row in treatment["treatment"]["arms"] if
              (row["schedule"], row["region"], row["cell"]) ==
              ("N", "R2_boundary", "CC"))
    cc["scores"]["focal"]["margin_float32_bits"] = bits(-77.0)
    write_json(treatment_path, treatment)
    with pytest.raises(HARVEST.HarvestError, match="margin bits differ"):
        HARVEST.harvest(treatment_path, repo_root=tmp_path)


def test_primary_or_placebo_selector_loss_is_rejected(tmp_path, monkeypatch):
    treatment_path = run_synthetic_treatment(tmp_path, monkeypatch)
    treatment = json.loads(treatment_path.read_text())
    treatment["treatment"]["arms"][-1]["region"] = "R2_boundary"
    write_json(treatment_path, treatment)
    with pytest.raises(HARVEST.HarvestError, match="duplicate placebo"):
        HARVEST.harvest(treatment_path, repo_root=tmp_path)


def test_local_runner_envelope_is_harvested_as_integration_only(
        tmp_path, monkeypatch):
    treatment_path = run_synthetic_treatment(
        tmp_path, monkeypatch, subject="local-apparatus",
        phase_status="ESTIMAND_INADEQUATE")
    report = HARVEST.harvest(treatment_path, repo_root=tmp_path)
    assert report["subject"] == "local-apparatus"
    assert report["runtime"]["requested_model"] == (
        HARVEST.SUBJECTS["local-apparatus"]["model_id"])
    assert report["semantic_evidence_eligible"] is False
    assert report["apparatus_integration_only"] is True
    assert report["phase_a_status"] == "ESTIMAND_INADEQUATE"
    assert report["phase_a_inadequacy_reasons"] == ["oracle"]
    assert "not semantic evidence" in report["inference"]["note"]
