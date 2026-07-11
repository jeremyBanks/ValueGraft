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
SCRIPT = ROOT / "scripts/validate_coherent_canary_v12_phase_a.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_coherent_canary_v12_phase_a", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def bits(value: float) -> str:
    return struct.pack("<f", value).hex()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def bound(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(),
            "sha256": MODULE.file_sha256(path)}


def create_sources(repo: Path, case_id="e01", *,
                   subject="exact-subject") -> tuple[dict, list[int]]:
    carrier_ids = [101, 102, 103]
    fixed = repo / "data/coherent_canary_v12/fixed_text_token_evidence_v2.json"
    write_json(fixed, {"texts": {"engineered_carrier_content": {
        "token_ids": carrier_ids}}})
    case = repo / f"data/cases/{case_id}.json"
    write_json(case, {"design_id": MODULE.DESIGN_ID, "case_id": case_id})
    case_sha = MODULE.file_sha256(case)
    manifest = repo / "results/manifest.json"
    write_json(manifest, {"cases": [
        {"case_id": case_id, "input_file_sha256": case_sha}]})
    blind = repo / "results/blind.json"
    write_json(blind, {
        "schema": "coherent_state_decision_canary_v12_blind_review_v1",
        "aggregate": {"overall_verdict": "PASS"},
        "shared_carrier_anchor_review": {"verdict": "PASS"},
    })
    paired = repo / "results/paired.json"
    write_json(paired, {
        "schema": "coherent_state_decision_canary_v12_paired_diversity_review_v1",
        "overall_verdict": "PASS",
        "paired_case_reviews": [{"case_id": case_id, "verdict": "PASS"}],
        "cross_case_diversity_review": {"verdict": "PASS"},
    })
    prereg = repo / "COHERENT-STATE-DECISION-CANARY-V12-PREREGISTRATION.md"
    prereg.write_text("frozen preregistration\n")
    runtime = fingerprint(subject)
    technical = repo / "results/technical.json"
    write_json(technical, {
        "schema":
        "coherent_state_decision_canary_v12_technical_validation_v1",
        "design_id": MODULE.DESIGN_ID,
        "status": "PASS", "subject": subject,
        "semantic_release_eligible": subject == "exact-subject",
        "checks": {"runtime_fingerprint": {
            "passed": True,
            "evidence": {"fingerprint_sha256":
                         runtime["fingerprint_sha256"]},
        }, "source_bindings": {
            "passed": True,
            "evidence": {"fixed_text_evidence": bound(fixed, repo)},
        }},
    })
    return {
        "case": bound(case, repo),
        "preregistration": bound(prereg, repo),
        "revision4_manifest": bound(manifest, repo),
        "revision4_blind_review": bound(blind, repo),
        "revision4_paired_review": bound(paired, repo),
        "technical_validation_report": bound(technical, repo),
    }, carrier_ids


def fingerprint(subject="exact-subject") -> dict:
    expected = MODULE.SUBJECTS[subject]
    value = {
        "requested_model": expected["model_id"],
        "requested_revision": expected["revision"],
        "dtype": "torch.bfloat16", "attention_backend": "eager",
        "geometry": expected["geometry"],
        "subject_spec": {"key": subject},
        "protocol_tokenizer_attestation": {
            "all_special_ids": [9, 10],
        },
    }
    value["fingerprint_sha256"] = hashlib.sha256(
        MODULE.canonical_bytes(value)).hexdigest()
    return value


def target(token_id: int, mean: float) -> dict:
    return {
        "target_token_ids": [token_id],
        "token_logprob_float32_bits": [bits(mean)],
        "mean_logprob_float32_bits": bits(mean),
        "mean_logprob": 999.0,
    }


def score(margin: float, generated_target: int, *, correct_id=1, counter_id=2,
          normal=True) -> dict:
    return {
        "correct": target(correct_id, margin),
        "counterfactual": target(counter_id, 0.0),
        "margin": -999.0,
        "generation": {
            "content_ids": [generated_target],
            "eos_ids": [9], "stop_candidate_id": 9,
            "stop_reason": "model_eos" if normal else "max_content_tokens",
            "cap_hit": not normal,
        },
    }


def raw_fixture(repo: Path, *, subject="exact-subject") -> tuple[Path, dict]:
    bindings, carrier_ids = create_sources(repo, subject=subject)
    scores = {
        "A_C_focal": score(1.0, 1),
        "A_W_focal": score(-1.0, 2),
        "FF_focal": score(0.2, 1),
        "A_C_nonfocal": score(0.8, 3, correct_id=3, counter_id=4),
        "A_W_nonfocal": score(0.7, 3, correct_id=3, counter_id=4),
        "FF_nonfocal": score(0.1, 3, correct_id=3, counter_id=4),
    }
    support = {branch: {
        "token_ids": carrier_ids,
        "token_logprob_float32_bits": [bits(-0.1)] * len(carrier_ids),
        "mean_nll": -999.0, "all_finite": False,
    } for branch in ("C_N", "W_N")}
    phase = {
        "schema": MODULE.PHASE_SCHEMA, "design_id": MODULE.DESIGN_ID,
        "case_id": "e01", "plans": {},
        "executions": {"C_N": {}, "W_N": {}, "F": {}},
        "forced_carrier_support": support, "scores": scores,
        "visible_messages": {}, "treatment_scores_present": False,
    }
    raw = {
        "schema": MODULE.RUN_SCHEMA, "design_id": MODULE.DESIGN_ID,
        "subject": subject, "case_id": "e01", "bindings": bindings,
        "runtime_fingerprint": fingerprint(subject), "phase_a": phase,
        "treatment_scores_present": False,
    }
    path = repo / "raw.json"
    write_json(path, raw)
    return path, raw


def test_exact_phase_a_pass_recomputes_bits_and_damage(tmp_path):
    raw_path, _ = raw_fixture(tmp_path)
    report = MODULE.validate_phase_a_raw(raw_path, repo_root=tmp_path)
    assert report["status"] == "PRETREATMENT_PASS"
    assert report["semantic_release_eligible"] is True
    assert report["fresh_damage_diagnostic"]["margin"] == pytest.approx(0.8)
    assert report["runner_status_and_decimal_margins_ignored"] is True


def test_local_pass_never_releases_semantic_treatment(tmp_path):
    raw_path, _ = raw_fixture(tmp_path, subject="local-apparatus")
    report = MODULE.validate_phase_a_raw(raw_path, repo_root=tmp_path)
    assert report["status"] == "PRETREATMENT_PASS"
    assert report["semantic_release_eligible"] is False


def test_oracle_margin_or_generation_failure_is_estimand_inadequate(tmp_path):
    raw_path, raw = raw_fixture(tmp_path)
    raw["phase_a"]["scores"]["A_W_focal"] = score(0.1, 2)
    raw["phase_a"]["scores"]["A_C_nonfocal"]["generation"]["cap_hit"] = True
    write_json(raw_path, raw)
    report = MODULE.validate_phase_a_raw(raw_path, repo_root=tmp_path)
    assert report["status"] == "ESTIMAND_INADEQUATE"
    assert "A_W_focal_margin_negative" in report["inadequacy_reasons"]
    assert "A_C_nonfocal_generation" in report["inadequacy_reasons"]


def test_fresh_damage_is_diagnostic_not_gate(tmp_path):
    raw_path, raw = raw_fixture(tmp_path)
    raw["phase_a"]["scores"]["FF_focal"] = score(1.5, 1)
    write_json(raw_path, raw)
    report = MODULE.validate_phase_a_raw(raw_path, repo_root=tmp_path)
    assert report["status"] == "PRETREATMENT_PASS"
    assert report["fresh_damage_diagnostic"]["positive_margin_damage"] is False


def test_treatment_field_or_nonfinite_support_is_invalid_technical(tmp_path):
    raw_path, raw = raw_fixture(tmp_path)
    raw["treatment_scores"] = {"CC": 1.0}
    raw["phase_a"]["forced_carrier_support"]["C_N"][
        "token_logprob_float32_bits"][0] = "0000807f"
    write_json(raw_path, raw)
    report = MODULE.validate_phase_a_raw(raw_path, repo_root=tmp_path)
    assert report["status"] == "INVALID_TECHNICAL"
    assert "schema_and_blinding" in report["invalidity_reasons"]
    assert "forced_carrier_support" in report["invalidity_reasons"]


def test_requires_exactly_six_scores_and_valid_bindings(tmp_path):
    raw_path, raw = raw_fixture(tmp_path)
    del raw["phase_a"]["scores"]["FF_nonfocal"]
    raw["bindings"]["case"]["sha256"] = "0" * 64
    write_json(raw_path, raw)
    report = MODULE.validate_phase_a_raw(raw_path, repo_root=tmp_path)
    assert report["status"] == "INVALID_TECHNICAL"
    assert "bindings" in report["invalidity_reasons"]
    # Structural binding failure prevents score interpretation in this invocation.
    assert report["fresh_damage_diagnostic"] is None


def test_unique_output_is_create_only(tmp_path):
    raw_path, _ = raw_fixture(tmp_path)
    report = MODULE.validate_phase_a_raw(raw_path, repo_root=tmp_path)
    output = tmp_path / "report.json"
    MODULE.write_unique_json(output, report)
    with pytest.raises(FileExistsError):
        MODULE.write_unique_json(output, report)
