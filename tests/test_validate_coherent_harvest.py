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
    if all(key in doc for key in ("schema", "amendment_id", "design_id")):
        doc = {key: value for key, value in doc.items()
               if key != "payload_sha256"}
        doc["payload_sha256"] = MODULE._canonical_payload_sha256(doc)
    path.write_text(json.dumps(doc))


def identity() -> dict:
    return {
        "schema": 2,
        "amendment_id": MODULE.AMENDMENT_ID,
        "design_id": MODULE.DESIGN_ID,
    }


def backend_fingerprint() -> dict:
    def config(scope: str) -> dict:
        return {
            "scope": scope,
            "config_class": "Qwen3MoeConfig",
            "_attn_implementation": "eager",
            "_attn_implementation_internal": "eager",
            "resolved_implementation": "eager",
        }
    doc = {
        "requested_implementation": "eager",
        "model_config": config("model_config"),
        "text_config": config("text_config"),
        "text_config_is_model_config": True,
        "expected_layer_count": 48,
        "layers": [
            {"layer_index": i,
             "module_name": f"model.layers.{i}.self_attn",
             "module_class": "Qwen3MoeAttention",
             "module_config_class": "Qwen3MoeConfig",
             "module_config__attn_implementation": "eager",
             "module_config__attn_implementation_internal": "eager",
             "resolved_implementation": "eager"}
            for i in range(48)
        ],
    }
    doc["sha256"] = MODULE._backend_payload_sha256(doc)
    return doc


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


def test_global_gate_failure_requires_complete_fail_gate(tmp_path: Path):
    (tmp_path / "job.log").write_text("MODEL_READY\nFATAL injected\n")
    write(tmp_path / "failure.json", {"error": "injected"})
    write(tmp_path / "production_kernel_gate.json", gate("FAIL"))
    out = MODULE.validate(tmp_path, "failure")
    assert out["production_gate"] == "FAIL"

    pre_backend = gate("FAIL")
    pre_backend["gates"].pop("attention_backend")
    write(tmp_path / "production_kernel_gate.json", pre_backend)
    assert MODULE.validate(tmp_path, "failure")["production_gate"] == "FAIL"

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
