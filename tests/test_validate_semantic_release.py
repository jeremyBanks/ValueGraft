from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_semantic_release",
    ROOT / "scripts" / "validate_semantic_release.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True,
        text=True).stdout.strip()


def _write(path: Path, doc: dict) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = MODULE._canonical(MODULE._seal(doc)) + b"\n"
    path.write_bytes(raw)
    return raw


def _stage(name: str) -> dict:
    expected = 1
    raw = {"passes": True, "failures": []}
    threshold = None
    comparison = None
    if name == "static_provenance":
        raw = {
            "device": "cpu", "dtype": "torch.bfloat16",
            "local_model": MODULE.LADDER_MODEL,
            "production_tokenizer": MODULE.PRODUCTION_MODEL,
            "production_tokenizer_revision": MODULE.PRODUCTION_REVISION,
            "technical_only": True,
        }
    elif name == "attention_backend":
        expected = 28
        raw = {
            "fingerprint": {
                "requested_implementation": "eager",
                "expected_layer_count": 28,
                "layers": [
                    {"layer_index": index,
                     "resolved_implementation": "eager"}
                    for index in range(28)],
            },
            "subject": {},
        }
    elif name == "synthetic_schedule_fixtures":
        expected = 7
        threshold, comparison = 0.0005, "<="
        raw = {"passes": True, "failures": []}
    elif name == "committed_case_schedule_fixtures":
        expected = 12
        threshold, comparison = 0.0005, "<="
        raw = {
            "status": "PASS", "passes": True, "failures": [],
            "expected_coverage": 12, "observed_coverage": 12,
            "frozen_order": list(MODULE.FROZEN_ORDER),
            "rows": [
                {"conversation_id": cid, "status": "PASS", "passes": True}
                for cid in MODULE.FROZEN_ORDER],
        }
    return {
        "status": "PASS", "passes": True,
        "prerequisites": [], "threshold": threshold,
        "comparison": comparison, "expected_coverage": expected,
        "observed_coverage": expected, "metric_names": [], "raw": raw,
        "failure_evidence": None,
    }


def _fixture(tmp_path: Path, mutate=None):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "trunk")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    apparatus_path = repo / "src" / "dummy.py"
    apparatus_path.parent.mkdir()
    apparatus_path.write_text("VALUE = 1\n")
    _git(repo, "add", "src/dummy.py")
    _git(repo, "commit", "-m", "ladder launch")
    launch = _git(repo, "rev-parse", "HEAD")

    ladder_path = MODULE.ELIGIBLE_LADDER_PATH
    parent = repo / Path(ladder_path).parent
    refs = {}
    artifacts = []
    sidecars = {}
    for name in MODULE.STAGE_ORDER:
        filename = f"ladder__stage_{name}.json"
        doc = {
            "schema": 2, "amendment_id": MODULE.AMENDMENT_ID,
            "design_id": MODULE.DESIGN_ID,
            "artifact_kind": "ladder_gate_stage", "stage_name": name,
            "stage": _stage(name),
        }
        sidecars[name] = doc
        raw = _write(parent / filename, doc)
        loaded = json.loads(raw)
        row = {
            "path": filename, "byte_count": len(raw),
            "raw_file_sha256": hashlib.sha256(raw).hexdigest(),
            "payload_sha256": loaded["payload_sha256"],
        }
        refs[name] = dict(row)
        artifacts.append({key: row[key] for key in
                          ("path", "byte_count", "raw_file_sha256")})

    manifest = {
        "schema": 2, "amendment_id": MODULE.AMENDMENT_ID,
        "design_id": MODULE.DESIGN_ID, "status": "PASS",
        "model": MODULE.LADDER_MODEL,
        "resolved_revision": MODULE.LADDER_REVISION,
        "dtype": "torch.bfloat16", "device": "cpu",
        "loaded_gapped_production_gate": {
            "schema": 2, "amendment_id": MODULE.AMENDMENT_ID,
            "design_id": MODULE.DESIGN_ID, "status": "PASS",
            "passes": True, "technical_only": True,
            "stage_order": list(MODULE.STAGE_ORDER), "failures": [],
            "failure": None, "stage_refs": refs, "externalized": True,
        },
        "artifact_files": artifacts,
    }
    if mutate:
        mutate(manifest, sidecars, parent)
    _write(repo / ladder_path, manifest)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "ladder result")
    result = _git(repo, "rev-parse", "HEAD")
    raw = apparatus_path.read_bytes()
    apparatus = {
        "files": [{
            "path": "src/dummy.py", "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}],
        "file_count": 1,
        "aggregate_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return repo, launch, result, apparatus


def _validate(repo, launch, result, apparatus):
    prior = MODULE.LADDER_LAUNCH_COMMIT
    MODULE.LADDER_LAUNCH_COMMIT = launch
    try:
        return MODULE.validate_ladder_commit(
            repo, result_commit=result,
            ladder_path=MODULE.ELIGIBLE_LADDER_PATH,
            semantic_launch_commit=result,
            current_apparatus=apparatus)
    finally:
        MODULE.LADDER_LAUNCH_COMMIT = prior


def test_complete_ladder_and_exact_apparatus_pass(tmp_path):
    repo, launch, result, apparatus = _fixture(tmp_path)
    observed = _validate(repo, launch, result, apparatus)
    assert observed["status"] == "PASS"
    assert observed["committed_cases"] == 12
    assert observed["bridge"]["all_inventoried_bytes_exact"] is True


def test_partial_committed_cases_fail(tmp_path):
    def mutate(manifest, sidecars, parent):
        doc = sidecars["committed_case_schedule_fixtures"]
        doc["stage"]["observed_coverage"] = 11
        raw = _write(parent / "ladder__stage_committed_case_schedule_fixtures.json", doc)
        loaded = json.loads(raw)
        ref = manifest["loaded_gapped_production_gate"]["stage_refs"][
            "committed_case_schedule_fixtures"]
        ref.update(byte_count=len(raw), raw_file_sha256=hashlib.sha256(raw).hexdigest(),
                   payload_sha256=loaded["payload_sha256"])
        for row in manifest["artifact_files"]:
            if row["path"] == ref["path"]:
                row.update(byte_count=len(raw),
                           raw_file_sha256=hashlib.sha256(raw).hexdigest())
    repo, launch, result, apparatus = _fixture(tmp_path, mutate)
    with pytest.raises(MODULE.ReleaseError, match="coverage is incomplete"):
        _validate(repo, launch, result, apparatus)


def test_failed_stage_fails(tmp_path):
    def mutate(manifest, sidecars, parent):
        doc = sidecars["generated_replay_identity"]
        doc["stage"].update(status="FAIL", passes=False)
        raw = _write(parent / "ladder__stage_generated_replay_identity.json", doc)
        loaded = json.loads(raw)
        ref = manifest["loaded_gapped_production_gate"]["stage_refs"][
            "generated_replay_identity"]
        ref.update(byte_count=len(raw), raw_file_sha256=hashlib.sha256(raw).hexdigest(),
                   payload_sha256=loaded["payload_sha256"])
        for row in manifest["artifact_files"]:
            if row["path"] == ref["path"]:
                row.update(byte_count=len(raw),
                           raw_file_sha256=hashlib.sha256(raw).hexdigest())
    repo, launch, result, apparatus = _fixture(tmp_path, mutate)
    with pytest.raises(MODULE.ReleaseError, match="not terminal PASS"):
        _validate(repo, launch, result, apparatus)


def test_tampered_sidecar_reference_fails(tmp_path):
    def mutate(manifest, _sidecars, _parent):
        manifest["loaded_gapped_production_gate"]["stage_refs"][
            "snapshot_rebuild_identity"]["raw_file_sha256"] = "0" * 64
    repo, launch, result, apparatus = _fixture(tmp_path, mutate)
    with pytest.raises(MODULE.ReleaseError, match="reference differs"):
        _validate(repo, launch, result, apparatus)


def test_wrong_scientific_identity_fails(tmp_path):
    def mutate(manifest, _sidecars, _parent):
        manifest["design_id"] = "coherent-state-gapped-v999"
    repo, launch, result, apparatus = _fixture(tmp_path, mutate)
    with pytest.raises(MODULE.ReleaseError, match="identity differs"):
        _validate(repo, launch, result, apparatus)


def test_apparatus_drift_from_ladder_launch_fails(tmp_path):
    repo, launch, result, apparatus = _fixture(tmp_path)
    apparatus["files"][0]["sha256"] = "f" * 64
    with pytest.raises(MODULE.ReleaseError, match="apparatus differs"):
        _validate(repo, launch, result, apparatus)


def test_only_exact_frozen_ladder_path_is_eligible(tmp_path):
    repo, launch, result, apparatus = _fixture(tmp_path)
    prior = MODULE.LADDER_LAUNCH_COMMIT
    MODULE.LADDER_LAUNCH_COMMIT = launch
    try:
        with pytest.raises(MODULE.ReleaseError, match="not the Amendment-11 eligible"):
            MODULE.validate_ladder_commit(
                repo, result_commit=result, ladder_path="results/other.json",
                semantic_launch_commit=result, current_apparatus=apparatus)
    finally:
        MODULE.LADDER_LAUNCH_COMMIT = prior


def test_release_layer_does_not_change_v10_apparatus_inventory():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from coherent_state_integrity import apparatus_inventory
    observed = apparatus_inventory(ROOT)
    assert observed["file_count"] == 35
    assert observed["aggregate_sha256"] == \
        "818a60623c4858f0796865a124d255897325136f147ad611c1810026c9352715"
    apparatus_paths = {row["path"] for row in observed["files"]}
    assert not (apparatus_paths & set(MODULE.OVERLAY_PATHS))
    for path in MODULE.OVERLAY_PATHS:
        assert not fnmatch.fnmatch(path, "src/coherent_state_*.py")
        assert not fnmatch.fnmatch(path, "scripts/*coherent*.sh")
        assert not fnmatch.fnmatch(path, "scripts/*coherent*.py")


def test_wrapper_validates_before_invoking_launcher():
    text = (ROOT / "scripts" / "launch_semantic_release.sh").read_text()
    verify = text.index("validate_semantic_release.py verify")
    launch = text.index("scripts/launch_pod.sh")
    assert verify < launch
    assert "set -euo pipefail" in text


def test_committed_attestation_recomputes_from_descendant_launch(
        tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "trunk")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    for relative in MODULE.OVERLAY_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "release evidence")
    evidence = _git(repo, "rev-parse", "HEAD")

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import coherent_state_integrity
    apparatus = {
        "files": [{"path": "src/dummy.py", "bytes": 1, "sha256": "a" * 64}],
        "file_count": 1, "aggregate_sha256": "b" * 64}
    monkeypatch.setattr(
        coherent_state_integrity, "apparatus_inventory", lambda _repo: apparatus)
    monkeypatch.setattr(
        MODULE, "validate_ladder_commit",
        lambda *_args, **_kwargs: {
            "status": "PASS", "result_commit": "c" * 40,
            "path": MODULE.ELIGIBLE_LADDER_PATH,
            "manifest_raw_sha256": "d" * 64,
            "manifest_payload_sha256": "e" * 64,
            "stage_payload_sha256": {}, "committed_cases": 12,
            "synthetic_fixtures": 7, "attention_layers": 28,
            "bridge": {"all_inventoried_bytes_exact": True}})
    technical = SimpleNamespace(
        result_commit="f" * 40, run_dir="results/technical",
        gate_payload_sha256="1" * 64,
        raw_sha256={"manifest": "2" * 64},
        harvest={"attestation": {"payload_sha256": "3" * 64}})
    monkeypatch.setattr(
        coherent_state_integrity, "verify_prior_technical_authorization",
        lambda *_args, **_kwargs: technical)

    attestation = MODULE.build_release_attestation(
        repo, evidence_commit=evidence,
        ladder_result_commit="c" * 40,
        ladder_path=MODULE.ELIGIBLE_LADDER_PATH,
        technical_result_commit="f" * 40,
        technical_run_dir="results/technical")
    relative = "results/v10_release/release.json"
    path = repo / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(MODULE._canonical(attestation) + b"\n")
    _git(repo, "add", relative)
    _git(repo, "commit", "-m", "commit release attestation")
    launch = _git(repo, "rev-parse", "HEAD")
    _git(repo, "remote", "add", "origin", ".")
    _git(repo, "update-ref", "refs/remotes/origin/trunk", launch)

    observed = MODULE.verify_committed_attestation(repo, relative, launch)
    assert observed["status"] == "PASS"
    assert observed["launch_commit"] == launch
