from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import coherent_state_integrity as integrity
import run_coherent_state_hf as driver


def _payload(status: str = "PASS") -> dict:
    return {
        "schema": integrity.SCHEMA,
        "design_id": integrity.DESIGN_ID,
        "amendment_id": integrity.AMENDMENT_ID,
        "status": status,
    }


def test_payload_hash_is_canonical_and_detects_mutation(tmp_path: Path):
    path = tmp_path / "manifest.json"
    sealed = integrity.write_sealed_payload(path, {
        **_payload(), "nested": {"b": 2, "a": 1},
    })
    assert sealed["payload_sha256"] == integrity.payload_sha256(sealed)
    integrity.verify_payload(json.loads(path.read_text()), path.name)
    changed = json.loads(path.read_text())
    changed["nested"]["a"] = 9
    with pytest.raises(integrity.IntegrityError, match="payload_sha256"):
        integrity.verify_payload(changed, path.name)


def test_failed_payload_canonicalizes_nonfinite_evidence(tmp_path: Path):
    path = tmp_path / "failure.json"
    sealed = integrity.write_sealed_payload(path, {
        **_payload("FAIL"), "raw": [float("nan"), float("inf"), -float("inf")],
    })
    assert sealed["raw"] == ["NaN", "Infinity", "-Infinity"]
    integrity.verify_payload(json.loads(path.read_text()), path.name)


def test_terminal_envelope_has_exact_set_and_rejects_tamper(tmp_path: Path):
    for name in ("manifest.json", "production_kernel_gate.json",
                 "production_kernel_gate_unique.json"):
        integrity.write_sealed_payload(tmp_path / name, _payload())
    index, receipt = integrity.write_terminal_envelope(
        tmp_path,
        ["manifest.json", "production_kernel_gate.json",
         "production_kernel_gate_unique.json"],
        terminal_status="PASS")
    assert index["required_apparatus_payload_paths"] == [
        "manifest.json", "production_kernel_gate.json",
        "production_kernel_gate_unique.json",
    ]
    assert receipt["index_raw_sha256"] == integrity.raw_sha256(
        tmp_path / integrity.INDEX_NAME)
    integrity.verify_terminal_envelope(tmp_path)
    (tmp_path / "manifest.json").write_text("{}\n")
    with pytest.raises(integrity.IntegrityError):
        integrity.verify_terminal_envelope(tmp_path)


def test_terminal_envelope_rejects_unindexed_payload(tmp_path: Path):
    integrity.write_sealed_payload(tmp_path / "manifest.json", _payload())
    integrity.write_sealed_payload(tmp_path / "surprise.json", _payload())
    with pytest.raises(integrity.IntegrityError, match="path set"):
        integrity.write_terminal_envelope(
            tmp_path, ["manifest.json"], terminal_status="PASS")


def test_semantic_terminal_seals_and_indexes_every_json(tmp_path: Path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": 2, "design_id": driver.DESIGN_ID,
        "amendment_id": driver.AMENDMENT_ID, "status": "COMPLETE",
    }))
    (tmp_path / "analysis.json").write_text(json.dumps({"value": 1}))
    driver._seal_semantic_terminal(tmp_path, "PASS")
    envelope = integrity.verify_terminal_envelope(tmp_path)
    assert envelope["index"]["required_apparatus_payload_paths"] == [
        "analysis.json", "manifest.json"]
    for name in ("analysis.json", "manifest.json"):
        doc = json.loads((tmp_path / name).read_text())
        integrity.verify_payload(doc, name)
        assert doc["design_id"] == driver.DESIGN_ID


def test_cli_defaults_to_technical_and_semantics_requires_commit(
        monkeypatch, tmp_path: Path):
    monkeypatch.setattr(driver.sys, "argv", [
        "run_coherent_state_hf.py", "--run-dir", str(tmp_path)])
    assert driver.parse_args().technical_only is True

    monkeypatch.setattr(driver.sys, "argv", [
        "run_coherent_state_hf.py", "--run-dir", str(tmp_path),
        "--semantic-authorization", str(tmp_path / "prior")])
    with pytest.raises(SystemExit):
        driver.parse_args()

    monkeypatch.setattr(driver.sys, "argv", [
        "run_coherent_state_hf.py", "--run-dir", str(tmp_path),
        "--semantic-authorization", str(tmp_path / "prior"),
        "--technical-result-commit", "a" * 40])
    args = driver.parse_args()
    assert args.technical_only is False
    assert args.semantic_authorization == tmp_path / "prior"


def test_semantic_run_cannot_overlap_technical_authorization(tmp_path: Path):
    prior = tmp_path / "prior"
    args = SimpleNamespace(run_dir=prior, semantic_authorization=prior)
    with pytest.raises(integrity.IntegrityError, match="overlap"):
        driver._semantic_main(args)
    assert not prior.exists()


def test_failure_before_model_load_is_fully_terminalized(
        monkeypatch, tmp_path: Path):
    args = SimpleNamespace(
        run_dir=tmp_path / "attempt",
        scenarios=Path("data/scenarios.json"),
        targets=Path("data/coherent_state_targets.json"),
        donor_dir=Path("data/synthetic"),
        technical_only=True,
    )
    monkeypatch.setattr(
        driver, "_static_design_self_check",
        lambda: (_ for _ in ()).throw(RuntimeError("injected before load")))
    assert driver._technical_main(args) == 1
    run_dir = args.run_dir
    envelope = integrity.verify_terminal_envelope(run_dir)
    assert envelope["receipt"]["status"] == "FAIL"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    gate = json.loads((run_dir / "production_kernel_gate.json").read_text())
    assert manifest["status"] == "ERROR"
    assert gate["status"] == "FAIL"
    inline_stages = [
        name for name in gate["gates"]["stage_order"]
        if name not in gate["gates"]["stage_refs"]]
    assert all(gate["gates"][name]["status"] not in {"PENDING", "RUNNING"}
               for name in inline_stages)
    for ref in gate["gates"]["stage_refs"].values():
        sidecar = json.loads((run_dir / ref["path"]).read_text())
        assert sidecar["lifecycle"]["status"] == "SKIPPED_DEPENDENCY"
    assert json.loads((run_dir / "failure.json").read_text())["error"] == \
        "injected before load"
    unique = next(run_dir.glob("production_kernel_gate_*.json"))
    assert unique.read_bytes() == (run_dir / "production_kernel_gate.json").read_bytes()


def test_technical_pass_exits_after_receipt_without_runner(
        monkeypatch, tmp_path: Path):
    args = SimpleNamespace(
        run_dir=tmp_path / "attempt",
        scenarios=tmp_path / "scenarios.json",
        targets=tmp_path / "targets.json",
        donor_dir=tmp_path / "cases",
        technical_only=True,
    )
    backend = {"sha256": "a" * 64, "layers": [{}] * 48}
    static = {
        "schema": 2, "design_id": driver.DESIGN_ID,
        "amendment_id": driver.AMENDMENT_ID, "code_commit": "b" * 40,
        "apparatus_inventory": {"aggregate_sha256": "c" * 64},
    }
    monkeypatch.setattr(driver, "AMENDMENT_PATHS", ())
    monkeypatch.setattr(driver, "_static_design_self_check", lambda: None)
    monkeypatch.setattr(driver, "git_value", lambda *args: str(tmp_path))
    monkeypatch.setattr(
        driver, "runtime_provenance", lambda _run: {"code_commit": "b" * 40})
    monkeypatch.setattr(
        driver, "apparatus_inventory",
        lambda _repo: {"aggregate_sha256": "c" * 64})
    monkeypatch.setattr(
        driver, "prepare_subject_metadata",
        lambda: (object(), object(), {"config_sha256": "d" * 64}))
    monkeypatch.setattr(driver, "load_scenarios", lambda _path: {})
    monkeypatch.setattr(driver, "frozen_scenarios", lambda _rows: [])
    monkeypatch.setattr(driver, "load_and_validate_targets", lambda *_: {})
    monkeypatch.setattr(driver, "load_external_donors", lambda _path: ({}, {}))
    monkeypatch.setattr(
        driver, "_build_static_fingerprint", lambda *_args: static)
    monkeypatch.setattr(
        driver, "load_subject",
        lambda *_args, **_kwargs: (object(), object(), {"layers": 48}, backend))
    monkeypatch.setattr(driver, "model_context_limit", lambda _model: 40_960)
    called = {}

    def gates(_model, _tokenizer, **kwargs):
        called.update(kwargs)
        sink = kwargs["diagnostic_sink"]
        for stage_name in sink["stage_order"]:
            stage = dict(sink[stage_name])
            stage["status"] = "PASS"
            stage["passes"] = True
            sink[stage_name] = stage
        sink["passes"] = True
        sink["attention_backend"] = {
            "status": "PASS", "passes": True, "fingerprint": backend,
        }
        return sink

    monkeypatch.setattr(driver, "run_loaded_gapped_gates", gates)
    monkeypatch.setattr(
        driver, "Runner",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("technical process reached semantic Runner")))
    assert driver._technical_main(args) == 0
    assert called["case_dir"] == args.donor_dir
    assert called["donor_dir"] == args.donor_dir
    envelope = integrity.verify_terminal_envelope(args.run_dir)
    assert envelope["receipt"]["status"] == "PASS"
    assert not list(args.run_dir.glob("conv_*.json"))


def test_apparatus_aggregate_binds_paths_and_bytes(tmp_path: Path, monkeypatch):
    required = ("src/run_coherent_state_hf.py",)
    monkeypatch.setattr(integrity, "APPARATUS_REQUIRED", required)
    monkeypatch.setattr(integrity, "APPARATUS_GLOBS", ())
    path = tmp_path / required[0]
    path.parent.mkdir(parents=True)
    path.write_text("one\n")
    first = integrity.apparatus_inventory(tmp_path)
    path.write_text("two\n")
    second = integrity.apparatus_inventory(tmp_path)
    assert first["aggregate_sha256"] != second["aggregate_sha256"]
    assert first["files"][0]["path"] == required[0]


def test_committed_directory_requires_exact_trunk_bytes(tmp_path: Path):
    subprocess.run(["git", "init", "-q", "-b", "trunk"], cwd=tmp_path,
                   check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"],
                   cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path,
                   check=True)
    run = tmp_path / "results" / "attempt"
    run.mkdir(parents=True)
    (run / "artifact.txt").write_text("immutable\n")
    subprocess.run(["git", "add", "results/attempt/artifact.txt"],
                   cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "result"], cwd=tmp_path,
                   check=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    verified = integrity.verify_committed_directory(
        tmp_path, run, commit, commit)
    assert verified["result_commit"] == commit
    (run / "artifact.txt").write_text("tampered\n")
    with pytest.raises(integrity.IntegrityError, match="bytes differ"):
        integrity.verify_committed_directory(tmp_path, run, commit, commit)
