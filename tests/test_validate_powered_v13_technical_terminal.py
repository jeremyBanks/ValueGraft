from __future__ import annotations

from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

import powered_v13_store as store
from powered_v13_schema import N_SCHEDULE, PRIMARY_ARMS


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_powered_v13_technical_terminal.py"
SPEC = importlib.util.spec_from_file_location("v13_independent_validator", SCRIPT)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)

T0 = "2026-07-12T21:00:00Z"
PID = 4242
BOOT_ID = "11111111-2222-3333-4444-555555555555"
START = f"linux-procfs-v1:{BOOT_ID}:123456"


def _token_reader(_pid: int) -> str:
    return START


def _identity(case_id="technical_e01"):
    return store.build_identity(
        release_sha256="1" * 64,
        runtime_fingerprint_sha256="2" * 64,
        primary_batch_id="stage-t-001",
        gpu_uuid="GPU-deadbeef",
        case_id=case_id,
        render_id="r1",
        schedule=N_SCHEDULE,
        mode="technical",
    )


def _bounds(arms=()):
    return store.build_bounds(
        completed_arm_ids=arms,
        live_token_count=1200,
        selected_row_count=74,
        case_deadline_seconds=1200,
    )


def _write(root: Path, relative: str, content: bytes) -> dict:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": relative,
        "sha256": store.file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _binding(root: Path, kind: str, tag: str) -> dict:
    row = _write(root, f"artifacts/{tag}-{kind}.bin", f"{tag}:{kind}".encode())
    return {"kind": kind, **row}


def _append(root: Path, identity, kind: str, *, arms=(), payload, bindings):
    return store._append_case_record(
        root,
        identity=identity,
        pid=PID,
        process_start_token=START,
        record_kind=kind,
        bounds=_bounds(arms),
        created_utc=T0,
        payload=payload,
        artifact_bindings=bindings,
        _test_process_start_token_reader=_token_reader,
    )


def _preterminal(root: Path, *, case_id="technical_e01"):
    identity = _identity(case_id)
    store._begin_case(
        root,
        identity=identity,
        bounds=_bounds(),
        pid=PID,
        process_start_token=START,
        created_utc=T0,
        payload={"status": "STARTED"},
        _test_process_start_token_reader=_token_reader,
    )
    bundle = _binding(root, "foundation_bundle", "foundation")
    descriptor = _binding(root, "foundation_descriptor", "foundation")
    receipt = _binding(root, "foundation_receipt", "foundation")
    _append(
        root, identity, "FOUNDATION_LOAD",
        payload={
            "foundation_bundle_sha256": bundle["sha256"],
            "foundation_descriptor_sha256": descriptor["sha256"],
            "foundation_receipt_sha256": receipt["sha256"],
        },
        bindings=[bundle, descriptor, receipt],
    )
    for index, arm in enumerate(PRIMARY_ARMS, start=1):
        checkpoint = _binding(root, f"arm_{arm}_checkpoint", arm)
        artifact = _binding(root, f"arm_{arm}_artifact", arm)
        _append(
            root, identity, f"ARM_{arm}", arms=PRIMARY_ARMS[:index],
            payload={
                "arm_id": arm,
                "checkpoint_sha256": checkpoint["sha256"],
                "artifact_sha256": artifact["sha256"],
            },
            bindings=[checkpoint, artifact],
        )
    identity_path = root / "artifacts/identity.json"
    identity_path.write_bytes(store.canonical_json_bytes(identity) + b"\n")
    chain = store._active_evidence_chain_sha256(
        root,
        identity=identity,
        pid=PID,
        process_start_token=START,
        _test_process_start_token_reader=_token_reader,
    )
    evidence = {
        "schema": store.TERMINAL_EVIDENCE_SCHEMA,
        "design_id": "coherent-state-powered-successor-v13",
        "identity_sha256": store.identity_sha256(identity),
        "terminal_kind": "TERMINAL_TECHNICAL",
        "evidence_chain_sha256": chain,
        "outcome": "TECHNICAL_COMPLETE",
        "accepted_attempt_index": None,
        "rejection_codes": [],
        "completed_arm_ids": list(PRIMARY_ARMS),
    }
    evidence_path = root / "artifacts/terminal-evidence.json"
    evidence_path.write_bytes(store.canonical_json_bytes(evidence) + b"\n")
    return identity, identity_path, evidence_path


def _existing_binding(root: Path, kind: str, path: Path) -> dict:
    return {
        "kind": kind,
        "path": path.relative_to(root).as_posix(),
        "sha256": store.file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def test_independent_validator_receipt_is_store_compatible_and_terminalizes(
        tmp_path):
    identity, identity_path, evidence_path = _preterminal(tmp_path)
    receipt_path = tmp_path / "artifacts/validator-receipt.json"
    receipt = validator.validate_and_write(
        root=tmp_path,
        identity_path=identity_path,
        terminal_evidence_path=evidence_path,
        output_path=receipt_path,
        runner_pid=PID,
        now=lambda: T0,
    )
    assert receipt["validation_status"] == "PASS"
    assert receipt["validator_id"] == store.TERMINAL_VALIDATOR_ID
    assert receipt["evidence_chain_sha256"] == \
        store._active_evidence_chain_sha256(
            tmp_path, identity=identity, pid=PID,
            process_start_token=START,
            _test_process_start_token_reader=_token_reader)

    evidence_binding = _existing_binding(
        tmp_path, "technical_terminal_evidence", evidence_path)
    receipt_binding = _existing_binding(
        tmp_path, "independent_terminal_validation_receipt", receipt_path)
    _append(
        tmp_path, identity, "TERMINAL_TECHNICAL", arms=PRIMARY_ARMS,
        payload={
            "status": "TECHNICAL_COMPLETE",
            "evidence_chain_sha256": receipt["evidence_chain_sha256"],
            "terminal_evidence_sha256": evidence_binding["sha256"],
            "validation_receipt_sha256": receipt_binding["sha256"],
        },
        bindings=[evidence_binding, receipt_binding],
    )
    index = store._finalize_case(
        tmp_path,
        identity=identity,
        pid=PID,
        process_start_token=START,
        _test_process_start_token_reader=_token_reader,
    )
    assert index["status"] == "TERMINAL_VALIDATED"


def test_cli_is_a_distinct_process_and_writes_same_receipt_contract(tmp_path):
    _identity_value, identity_path, evidence_path = _preterminal(tmp_path)
    receipt_path = tmp_path / "artifacts/validator-cli.json"
    completed = subprocess.run([
        sys.executable, str(SCRIPT),
        "--store-root", str(tmp_path),
        "--identity", str(identity_path),
        "--terminal-evidence", str(evidence_path),
        "--output", str(receipt_path),
        "--runner-pid", str(PID),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    receipt = store._strict_json(receipt_path.read_bytes(), "CLI receipt")
    assert receipt["validation_status"] == "PASS"
    assert receipt["validator_id"] == store.TERMINAL_VALIDATOR_ID


@pytest.mark.parametrize("mutation,match", [
    ("artifact", "external bytes differ"),
    ("evidence", "does not recompute"),
    ("missing_arm", "record count differs"),
])
def test_validator_rejects_corrupt_partial_or_forged_preterminal_chain(
        tmp_path, mutation, match):
    identity, identity_path, evidence_path = _preterminal(tmp_path)
    if mutation == "artifact":
        artifact = next((tmp_path / "artifacts").glob("FF-arm_FF_artifact.bin"))
        artifact.write_bytes(b"changed")
    elif mutation == "evidence":
        evidence = store._strict_json(evidence_path.read_bytes(), "evidence")
        evidence["evidence_chain_sha256"] = "0" * 64
        evidence_path.write_bytes(store.canonical_json_bytes(evidence) + b"\n")
    else:
        digest = store.identity_sha256(identity)
        records = tmp_path / "active" / digest / "records"
        (records / "007_ARM_VP.json").unlink()
    with pytest.raises(validator.IndependentTerminalValidationError, match=match):
        validator.validate_and_write(
            root=tmp_path,
            identity_path=identity_path,
            terminal_evidence_path=evidence_path,
            output_path=tmp_path / "artifacts/rejected.json",
            runner_pid=PID,
            now=lambda: T0,
        )


def test_validator_rejects_same_process_nontechnical_case_symlink_and_overwrite(
        tmp_path):
    _identity_value, identity_path, evidence_path = _preterminal(tmp_path)
    with pytest.raises(validator.IndependentTerminalValidationError,
                       match="not independent"):
        validator.validate_preterminal_chain(
            root=tmp_path, identity_path=identity_path,
            terminal_evidence_path=evidence_path, runner_pid=os.getpid())

    altered = deepcopy(store._strict_json(identity_path.read_bytes(), "identity"))
    altered["case_id"] = "semantic_case"
    identity_path.write_bytes(store.canonical_json_bytes(altered) + b"\n")
    with pytest.raises(validator.IndependentTerminalValidationError,
                       match="case/render/schedule"):
        validator.validate_preterminal_chain(
            root=tmp_path, identity_path=identity_path,
            terminal_evidence_path=evidence_path, runner_pid=PID)

    identity_path.write_bytes(
        store.canonical_json_bytes(_identity("technical_e01")) + b"\n")
    real_evidence = evidence_path.with_name("real-evidence.json")
    evidence_path.rename(real_evidence)
    evidence_path.symlink_to(real_evidence)
    with pytest.raises(validator.IndependentTerminalValidationError,
                       match="symlink"):
        validator.validate_preterminal_chain(
            root=tmp_path, identity_path=identity_path,
            terminal_evidence_path=evidence_path, runner_pid=PID)


def test_validator_source_has_no_store_import():
    source = SCRIPT.read_text()
    assert "import powered_v13_store" not in source
    assert "from powered_v13_store" not in source
