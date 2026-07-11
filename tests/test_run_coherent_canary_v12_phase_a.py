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
SCRIPT = ROOT / "scripts" / "run_coherent_canary_v12_phase_a.py"
SPEC = importlib.util.spec_from_file_location("run_coherent_canary_v12_phase_a", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

VALIDATOR_SCRIPT = ROOT / "scripts" / "validate_coherent_canary_v12_phase_a.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_coherent_canary_v12_phase_a_integration", VALIDATOR_SCRIPT)
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
assert VALIDATOR_SPEC.loader is not None
sys.modules[VALIDATOR_SPEC.name] = VALIDATOR
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)


def write_json(path: Path, value) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def fixture(tmp_path: Path, *, subject="local-apparatus",
            semantic_release_eligible=False, fingerprint="a" * 64):
    case = {
        "schema": "coherent_state_decision_canary_v12_case_draft_v1",
        "design_id": MODULE.DESIGN_ID, "case_id": "e01",
        "middle_end_msg": 3,
        "variants": {"correct": {"messages": []},
                     "wrong_focal": {"messages": []}},
    }
    case_path = write_json(tmp_path / "e01.json", case)
    manifest = write_json(tmp_path / "manifest.json", {
        "cases": [{"case_id": "e01",
                   "input_file_sha256": MODULE.file_sha256(case_path)}]})
    blind = write_json(tmp_path / "blind.json", {
        "schema": "coherent_state_decision_canary_v12_blind_review_v1",
        "aggregate": {
            "overall_verdict": "PASS", "history_fail_count": 0,
            "history_revise_count": 0,
        },
        "shared_carrier_anchor_review": {"verdict": "PASS"},
    })
    paired = write_json(tmp_path / "paired.json", {
        "schema":
        "coherent_state_decision_canary_v12_paired_diversity_review_v1",
        "overall_verdict": "PASS",
        "paired_case_reviews": [{"case_id": "e01", "verdict": "PASS"}],
        "cross_case_diversity_review": {"verdict": "PASS"}})
    technical = write_json(tmp_path / "technical-report.json", {
        "schema": MODULE.TECHNICAL_REPORT_SCHEMA,
        "design_id": MODULE.DESIGN_ID,
        "status": "PASS", "subject": subject,
        "semantic_release_eligible": semantic_release_eligible,
        "checks": {"runtime_fingerprint": {
            "passed": True,
            "evidence": {"fingerprint_sha256": fingerprint},
        }},
    })
    prereg = tmp_path / "prereg.md"
    prereg.write_text("frozen preregistration\n")
    args = MODULE.parse_args([
        "--subject", subject,
        "--case", str(case_path),
        "--technical-report", str(technical),
        "--repo", str(tmp_path),
        "--output-dir", str(tmp_path / "results"),
        "--prereg", str(prereg),
        "--manifest", str(manifest),
        "--blind-review", str(blind),
        "--paired-review", str(paired),
        "--hourly-cost-usd", "2.0",
    ])
    return args, case


def clean_phase(case_id="e01"):
    return {
        "schema": MODULE.PHASE_A_SCHEMA,
        "design_id": MODULE.DESIGN_ID,
        "case_id": case_id,
        "plans": {},
        "executions": {"C_N": {}, "W_N": {}, "F": {}},
        "forced_carrier_support": {},
        "scores": {key: {} for key in MODULE.EXPECTED_SCORE_KEYS},
        "visible_messages": {},
        "treatment_scores_present": False,
    }


def install_mocks(monkeypatch, *, fingerprint="a" * 64, runtime_record=None,
                  phase=None, events=None):
    def prepare(**kwargs):
        if events is not None:
            events.append("prepare")
        return {
            "model": "model", "tokenizer": "tokenizer",
            "runtime_fingerprint": runtime_record or {
                "fingerprint_sha256": fingerprint, "eos_ids": [9]},
            "repository": {"status": "FROZEN"},
            "dependencies": {"torch": "pinned"},
            "device": {"device_type": "cpu"},
            "model_snapshot": Path("/cache/model"),
            "protocol_tokenizer_snapshot": Path("/cache/tokenizer"),
            "model_inventory": {"sha256": "b" * 64},
            "protocol_tokenizer_inventory": {"sha256": "c" * 64},
        }
    monkeypatch.setattr(MODULE, "prepare_pinned_subject", prepare)
    monkeypatch.setattr(MODULE, "run_phase_a_case",
                        lambda *args, **kwargs: deepcopy(phase or clean_phase()))


def bits(value: float) -> str:
    return struct.pack("<f", value).hex()


def valid_runtime(subject="local-apparatus"):
    spec = MODULE.SUBJECTS[subject]
    record = {
        "requested_model": spec.model_id,
        "requested_revision": spec.revision,
        "dtype": "torch.bfloat16", "attention_backend": "eager",
        "geometry": spec.geometry,
        "subject_spec": {"key": subject},
        "protocol_tokenizer_attestation": {"all_special_ids": [9, 10]},
        "eos_ids": [9],
    }
    record["fingerprint_sha256"] = hashlib.sha256(
        VALIDATOR.canonical_bytes(record)).hexdigest()
    return record


def target(token_id, mean):
    return {
        "target_token_ids": [token_id],
        "token_logprob_float32_bits": [bits(mean)],
        "mean_logprob_float32_bits": bits(mean),
    }


def valid_score(margin, generated, *, correct=1, counter=2):
    return {
        "correct": target(correct, margin),
        "counterfactual": target(counter, 0.0),
        "generation": {
            "content_ids": [generated], "eos_ids": [9],
            "stop_candidate_id": 9, "stop_reason": "model_eos",
            "cap_hit": False,
        },
    }


def valid_phase_a(carrier_ids):
    phase = clean_phase()
    phase["forced_carrier_support"] = {
        branch: {
            "token_ids": carrier_ids,
            "token_logprob_float32_bits": [bits(-0.1)] * len(carrier_ids),
        } for branch in ("C_N", "W_N")
    }
    phase["scores"] = {
        "A_C_focal": valid_score(1.0, 1),
        "A_W_focal": valid_score(-1.0, 2),
        "FF_focal": valid_score(0.2, 1),
        "A_C_nonfocal": valid_score(0.8, 3, correct=3, counter=4),
        "A_W_nonfocal": valid_score(0.7, 3, correct=3, counter=4),
        "FF_nonfocal": valid_score(0.1, 3, correct=3, counter=4),
    }
    return phase


def test_local_phase_a_binds_inputs_matches_runtime_and_excludes_treatment(
        tmp_path, monkeypatch):
    args, case = fixture(tmp_path)
    events = []
    install_mocks(monkeypatch, events=events)
    real_print = print

    def observed_print(*values, **kwargs):
        events.append("print")
        real_print(*values, **kwargs)

    monkeypatch.setattr("builtins.print", observed_print)
    output, document, code = MODULE.run(args)
    persisted = json.loads(output.read_text())
    assert events.index("print") < events.index("prepare")
    assert code == 0
    assert document["status"] == persisted["status"] == "PASS"
    assert persisted["case_input"] == case
    assert persisted["phase_a"]["treatment_scores_present"] is False
    assert persisted["treatment_scores_present"] is False
    assert set(persisted["phase_a"]["scores"]) == MODULE.EXPECTED_SCORE_KEYS
    assert set(persisted["phase_a"]["executions"]) == {"C_N", "W_N", "F"}
    assert set(persisted["bindings"]) == {
        "case", "preregistration", "revision4_manifest",
        "revision4_blind_review", "revision4_paired_review",
        "technical_validation_report"}
    assert all(len(row["sha256"]) == 64 for row in persisted["bindings"].values())
    assert persisted["runtime_fingerprint"]["fingerprint_sha256"] == "a" * 64
    assert persisted["estimated_cost_usd"] is not None


def test_exact_subject_requires_semantic_release_eligible_report(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, subject="exact-subject",
                      semantic_release_eligible=False)
    install_mocks(monkeypatch)
    output, document, code = MODULE.run(args)
    assert code == 1
    assert document["status"] == "ERROR"
    assert "semantic-release eligible" in document["error"]["message"]
    assert json.loads(output.read_text())["status"] == "ERROR"


def test_prepared_runtime_must_match_technical_fingerprint(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, fingerprint="a" * 64)
    install_mocks(monkeypatch, fingerprint="d" * 64)
    output, document, code = MODULE.run(args)
    assert code == 1
    assert "runtime fingerprint differs" in document["error"]["message"]
    assert json.loads(output.read_text())["treatment_scores_present"] is False


def test_treatment_payload_is_rejected_and_error_is_saved(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    phase = clean_phase()
    phase["treatment_scores_present"] = True
    phase["treatment"] = {"CC": 1.0}
    install_mocks(monkeypatch, phase=phase)
    output, document, code = MODULE.run(args)
    assert code == 1
    assert "treatment scores" in document["error"]["message"]
    persisted = json.loads(output.read_text())
    assert persisted["status"] == "ERROR"
    assert persisted["treatment_scores_present"] is False
    assert "phase_a" not in persisted


def test_explicit_subject_case_and_technical_report_are_required():
    with pytest.raises(SystemExit):
        MODULE.parse_args([])
    with pytest.raises(SystemExit):
        MODULE.parse_args(["--subject", "local-apparatus"])
    with pytest.raises(SystemExit):
        MODULE.parse_args([
            "--subject", "other", "--case", "x", "--technical-report", "y"])


def test_runner_output_is_accepted_by_independent_validator(tmp_path, monkeypatch):
    runtime = valid_runtime()
    args, _ = fixture(
        tmp_path, fingerprint=runtime["fingerprint_sha256"])
    fixed = tmp_path / "data/coherent_canary_v12/fixed_text_token_evidence_v2.json"
    fixed.parent.mkdir(parents=True)
    carrier_ids = [101, 102, 103]
    write_json(fixed, {"texts": {"engineered_carrier_content": {
        "token_ids": carrier_ids}}})
    install_mocks(
        monkeypatch, runtime_record=runtime,
        phase=valid_phase_a(carrier_ids))

    output, _, code = MODULE.run(args)
    assert code == 0
    report = VALIDATOR.validate_phase_a_raw(output, repo_root=tmp_path)
    assert report["status"] == "PRETREATMENT_PASS"
    assert report["subject"] == "local-apparatus"
    assert report["semantic_release_eligible"] is False
