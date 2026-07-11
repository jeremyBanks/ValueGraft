from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_coherent_harvest", ROOT / "scripts/validate_coherent_harvest.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc))


def identity() -> dict:
    return {
        "schema": 2,
        "amendment_id": MODULE.AMENDMENT_ID,
        "design_id": MODULE.DESIGN_ID,
    }


def backend_fingerprint() -> dict:
    return {
        "requested_implementation": "eager",
        "layers": [
            {"layer_index": i, "resolved_implementation": "eager"}
            for i in range(48)
        ],
        "sha256": "backend-fixture",
    }


def run_fingerprint() -> dict:
    return {
        **identity(),
        "frozen_order": list(MODULE.FROZEN_ORDER),
        "wrong_donors": MODULE.WRONG_DONORS,
        "attention_backend": "eager",
        "attention_backend_fingerprint": backend_fingerprint(),
    }


def gate(status: str) -> dict:
    passes = status == "PASS"
    return {
        **identity(),
        "model": MODULE.MODEL_ID,
        "revision": MODULE.MODEL_REVISION,
        "dtype": MODULE.PARAMETER_DTYPE,
        "status": status,
        "completed_at": "2026-07-11T00:00:00Z",
        "gates": {
            "passes": passes,
            "attention_backend": {
                "observed_backend": "eager",
                "passes": True,
                "fingerprint": backend_fingerprint(),
            },
            **({} if passes else {"error": "injected global gate failure"}),
        },
    }


def checkpoint(position: int, status: str = "scored") -> dict:
    base = {
        **identity(),
        "order_position": position,
        "fingerprint": run_fingerprint(),
        "conversation_id": MODULE.FROZEN_ORDER[position - 1],
    }
    if status == "void":
        return {**base, "stage": "void", "status": "void",
                "failure": {"error": "injected case failure"}}
    arms = {arm: {"conversation_margin": 0.0} for arm in MODULE.ARMS}
    return {
        **base,
        "stage": "scored",
        "status": "scored",
        "conversation": {},
        "summary": {},
        "sources": {},
        "destination": {},
        "arm_scores": arms,
        "conversation_outcomes": {arm: 0.0 for arm in MODULE.ARMS},
        "gates": {"technical_pass": True},
        "runtime": {},
    }


def complete_tree(root: Path) -> None:
    (root / "job.log").write_text("MODEL_READY\nCOHERENT_STATE_JOB_DONE\n")
    write(root / "manifest.json", {
        **identity(), "status": "COMPLETE", "resume_probe_verified": True,
        "fingerprint": run_fingerprint()})
    write(root / "resume_probe.json", {
        **identity(), "status": "VERIFIED", "resume_probe_verified": True})
    write(root / "production_kernel_gate.json", gate("PASS"))
    for position in range(1, 7):
        write(root / f"conv_{position:02d}_c{position:02d}.json",
              checkpoint(position))


def test_complete_requires_schema2_gapped_arms_and_verified_resume(tmp_path: Path):
    complete_tree(tmp_path)
    out = MODULE.validate(tmp_path, "complete")
    assert out["n_scored"] == 6
    assert out["resume_probe_verified"] is True

    probe = json.loads((tmp_path / "resume_probe.json").read_text())
    probe["resume_probe_verified"] = False
    write(tmp_path / "resume_probe.json", probe)
    with pytest.raises(ValueError, match="not verified"):
        MODULE.validate(tmp_path, "complete")


def test_technical_pass_requires_eager_gate_and_no_semantic_artifacts(
        tmp_path: Path):
    (tmp_path / "job.log").write_text(
        "MODEL_READY attention_backend=eager\nCOHERENT_STATE_TECHNICAL_DONE\n")
    write(tmp_path / "manifest.json", {
        **identity(), "status": "TECHNICAL_PASS",
        "phase": "TECHNICAL_COMPLETE", "fingerprint": run_fingerprint()})
    write(tmp_path / "production_kernel_gate.json", gate("PASS"))
    out = MODULE.validate(tmp_path, "technical")
    assert out["n_scored"] == 0
    assert out["attention_backend"] == "eager"

    bad = gate("PASS")
    bad["gates"]["attention_backend"]["observed_backend"] = "sdpa"
    write(tmp_path / "production_kernel_gate.json", bad)
    with pytest.raises(ValueError, match="eager attention"):
        MODULE.validate(tmp_path, "technical")


def test_technical_pass_rejects_checkpoint_or_semantic_log_marker(tmp_path: Path):
    (tmp_path / "job.log").write_text("COHERENT_STATE_TECHNICAL_DONE\n")
    write(tmp_path / "manifest.json", {
        **identity(), "status": "TECHNICAL_PASS",
        "phase": "TECHNICAL_COMPLETE", "fingerprint": run_fingerprint()})
    write(tmp_path / "production_kernel_gate.json", gate("PASS"))
    write(tmp_path / "conv_01_c10.json", checkpoint(1))
    with pytest.raises(ValueError, match="conversation checkpoints"):
        MODULE.validate(tmp_path, "technical")


def test_technical_pass_rejects_outcome_fields_inside_gate(tmp_path: Path):
    (tmp_path / "job.log").write_text("COHERENT_STATE_TECHNICAL_DONE\n")
    write(tmp_path / "manifest.json", {
        **identity(), "status": "TECHNICAL_PASS",
        "phase": "TECHNICAL_COMPLETE", "fingerprint": run_fingerprint()})
    bad = gate("PASS")
    bad["gates"]["technical_margins_not_semantic_outcomes"] = {
        "G_correct": 1.0}
    write(tmp_path / "production_kernel_gate.json", bad)
    with pytest.raises(ValueError, match="semantic outcome fields"):
        MODULE.validate(tmp_path, "technical")


def test_complete_rejects_retired_or_missing_arm(tmp_path: Path):
    complete_tree(tmp_path)
    path = tmp_path / "conv_01_c01.json"
    doc = json.loads(path.read_text())
    doc["conversation_outcomes"]["C_coherent"] = \
        doc["conversation_outcomes"].pop("G_correct")
    write(path, doc)
    with pytest.raises(ValueError, match="exactly"):
        MODULE.validate(tmp_path, "complete")


def test_complete_validates_every_checkpoint(tmp_path: Path):
    complete_tree(tmp_path)
    path = tmp_path / "conv_06_c06.json"
    doc = json.loads(path.read_text())
    doc["gates"]["technical_pass"] = False
    write(path, doc)
    with pytest.raises(ValueError, match="technical_pass"):
        MODULE.validate(tmp_path, "complete")


def test_complete_requires_checkpoint_fingerprint_equal_manifest(tmp_path: Path):
    complete_tree(tmp_path)
    path = tmp_path / "conv_02_c02.json"
    doc = json.loads(path.read_text())
    doc["fingerprint"]["external_donor_provenance"] = {"tampered": True}
    write(path, doc)
    with pytest.raises(ValueError, match="manifest fingerprint"):
        MODULE.validate(tmp_path, "complete")


def test_global_gate_failure_requires_complete_fail_gate(tmp_path: Path):
    (tmp_path / "job.log").write_text("MODEL_READY\nFATAL injected\n")
    write(tmp_path / "failure.json", {"error": "injected"})
    write(tmp_path / "production_kernel_gate.json", gate("FAIL"))
    out = MODULE.validate(tmp_path, "failure")
    assert out["production_gate"] == "FAIL"

    bad = gate("FAIL")
    bad["gates"].pop("error")
    write(tmp_path / "production_kernel_gate.json", bad)
    with pytest.raises(ValueError, match="lacks failure evidence"):
        MODULE.validate(tmp_path, "failure")


def test_post_gate_case_failure_requires_void_checkpoint(tmp_path: Path):
    (tmp_path / "job.log").write_text("MODEL_READY\nFATAL case failed\n")
    write(tmp_path / "failure.json", {"error": "case failed"})
    write(tmp_path / "production_kernel_gate.json", gate("PASS"))
    with pytest.raises(ValueError, match="void checkpoint"):
        MODULE.validate(tmp_path, "failure")

    write(tmp_path / "conv_01_c01.json", checkpoint(1, "void"))
    out = MODULE.validate(tmp_path, "failure")
    assert out["n_void"] == 1


def test_failure_scored_checkpoint_must_match_manifest_fingerprint(tmp_path: Path):
    (tmp_path / "job.log").write_text("MODEL_READY\nFATAL later case failed\n")
    write(tmp_path / "failure.json", {"error": "later case failed"})
    write(tmp_path / "production_kernel_gate.json", gate("PASS"))
    scored = checkpoint(1)
    fingerprint = scored["fingerprint"]
    write(tmp_path / "manifest.json", {
        **identity(), "status": "ERROR", "fingerprint": fingerprint})
    write(tmp_path / "conv_01_c10.json", scored)
    write(tmp_path / "conv_02_c02.json", checkpoint(2, "void"))
    assert MODULE.validate(tmp_path, "failure")["n_scored"] == 1

    scored["fingerprint"] = dict(scored["fingerprint"])
    scored["fingerprint"]["scenario_sha256"] = "tampered"
    write(tmp_path / "conv_01_c10.json", scored)
    with pytest.raises(ValueError, match="manifest fingerprint"):
        MODULE.validate(tmp_path, "failure")


def test_setup_failure_needs_meaningful_evidence_but_no_gate(tmp_path: Path):
    (tmp_path / "job.log").write_text("FATAL: CUDA unavailable\n")
    out = MODULE.validate(tmp_path, "failure")
    assert out["model_ready"] is False

    (tmp_path / "job.log").write_text("setup stopped\n")
    with pytest.raises(ValueError, match="meaningful"):
        MODULE.validate(tmp_path, "failure")
