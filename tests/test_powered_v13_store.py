from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import errno
import inspect
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

import powered_v13_store as store
from powered_v13_schema import N_SCHEDULE, PRIMARY_ARMS


T0 = "2026-07-12T18:40:00Z"
PID = 4242
BOOT_ID = "11111111-2222-3333-4444-555555555555"
START = f"linux-procfs-v1:{BOOT_ID}:123456"


def _token_reader(pid: int) -> str:
    return START


def _identity(mode: str = "phase_a", **updates):
    values = {
        "release_sha256": "1" * 64,
        "runtime_fingerprint_sha256": "2" * 64,
        "primary_batch_id": "primary-001",
        "gpu_uuid": "GPU-deadbeef",
        "case_id": "a" * 64,
        "render_id": "r1",
        "schedule": N_SCHEDULE,
        "mode": mode,
    }
    values.update(updates)
    return store.build_identity(**values)


def _bounds(arms=(), *, tokens=1000, rows=64):
    return store.build_bounds(
        completed_arm_ids=arms,
        live_token_count=tokens,
        selected_row_count=rows,
        case_deadline_seconds=1200,
    )


def _binding(root: Path, kind: str, tag: str, content: bytes | None = None):
    path = root / "artifacts" / f"{tag}_{kind}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content or f"{tag}:{kind}".encode())
    return {
        "kind": kind,
        "path": path.relative_to(root).as_posix(),
        "sha256": store.file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _attempt_evidence(root: Path, index: int, outcome: str, tag: str):
    attempt = _binding(root, "render_attempt", tag)
    review = _binding(root, "render_review", tag)
    payload = {
        "attempt_index": index,
        "outcome": outcome,
        "attempt_sha256": attempt["sha256"],
        "review_sha256": review["sha256"],
        "rejection_code": None if outcome == "ACCEPTED" else "LEAKAGE_REVIEW",
    }
    return payload, [attempt, review]


def _plans_evidence(root: Path, tag: str = "plans"):
    plans = _binding(root, "plans", tag)
    scores = _binding(root, "phase_a_scores", tag)
    oracles = _binding(root, "oracle_scores", tag)
    return {
        "plans_sha256": plans["sha256"],
        "phase_a_scores_sha256": scores["sha256"],
        "oracle_scores_sha256": oracles["sha256"],
    }, [plans, scores, oracles]


def _foundation_evidence(root: Path, tag: str = "foundation"):
    bundle = _binding(root, "foundation_bundle", tag)
    descriptor = _binding(root, "foundation_descriptor", tag)
    receipt = _binding(root, "foundation_receipt", tag)
    return {
        "foundation_bundle_sha256": bundle["sha256"],
        "foundation_descriptor_sha256": descriptor["sha256"],
        "foundation_receipt_sha256": receipt["sha256"],
    }, [bundle, descriptor, receipt]


def _arm_evidence(root: Path, arm: str, tag: str | None = None):
    tag = tag or f"arm-{arm}"
    checkpoint = _binding(root, f"arm_{arm}_checkpoint", tag)
    artifact = _binding(root, f"arm_{arm}_artifact", tag)
    return {
        "arm_id": arm,
        "checkpoint_sha256": checkpoint["sha256"],
        "artifact_sha256": artifact["sha256"],
    }, [checkpoint, artifact]


def _terminal_evidence(
    root: Path,
    identity,
    *,
    terminal_kind: str,
    evidence_chain_sha256: str,
    outcome: str,
    accepted_attempt_index: int | None,
    rejection_codes: list[str],
    completed_arm_ids: list[str],
    tag: str,
):
    evidence_kind = {
        "TERMINAL_PHASE_A_ACCEPTED": "phase_a_terminal_evidence",
        "TERMINAL_PHASE_A_REJECTED": "phase_a_rejection_evidence",
        "TERMINAL_TREATMENT": "treatment_terminal_evidence",
        "TERMINAL_TECHNICAL": "technical_terminal_evidence",
    }[terminal_kind]
    evidence_document = {
        "schema": store.TERMINAL_EVIDENCE_SCHEMA,
        "design_id": "coherent-state-powered-successor-v13",
        "identity_sha256": store.identity_sha256(identity),
        "terminal_kind": terminal_kind,
        "evidence_chain_sha256": evidence_chain_sha256,
        "outcome": outcome,
        "accepted_attempt_index": accepted_attempt_index,
        "rejection_codes": rejection_codes,
        "completed_arm_ids": completed_arm_ids,
    }
    evidence = _binding(
        root, evidence_kind, tag,
        store.canonical_json_bytes(evidence_document) + b"\n")
    receipt_document = {
        "schema": store.TERMINAL_VALIDATION_SCHEMA,
        "design_id": "coherent-state-powered-successor-v13",
        "identity_sha256": store.identity_sha256(identity),
        "terminal_kind": terminal_kind,
        "evidence_chain_sha256": evidence_chain_sha256,
        "terminal_evidence_sha256": evidence["sha256"],
        "validator_id": store.TERMINAL_VALIDATOR_ID,
        "validation_status": "PASS",
        "validated_utc": T0,
    }
    receipt = _binding(
        root, "independent_terminal_validation_receipt", tag,
        store.canonical_json_bytes(receipt_document) + b"\n")
    return evidence, receipt


def _begin(root: Path, identity=None):
    identity = identity or _identity()
    return store._begin_case(
        root, identity=identity, bounds=_bounds(), pid=PID,
        process_start_token=START, created_utc=T0,
        payload={"status": "STARTED"},
        _test_process_start_token_reader=_token_reader)


def _append(
    root: Path, identity, kind: str, *, arms=(), payload, artifacts=(),
):
    return store._append_case_record(
        root, identity=identity, pid=PID, process_start_token=START,
        record_kind=kind, bounds=_bounds(arms=arms), created_utc=T0,
        payload=payload, artifact_bindings=artifacts,
        _test_process_start_token_reader=_token_reader)


def _chain_hash(root: Path, identity) -> str:
    return store._active_evidence_chain_sha256(
        root, identity=identity, pid=PID, process_start_token=START,
        _test_process_start_token_reader=_token_reader)


def _append_phase_terminal(root: Path, identity, *, accepted: bool):
    chain_hash = _chain_hash(root, identity)
    if accepted:
        kind = "TERMINAL_PHASE_A_ACCEPTED"
        evidence, receipt = _terminal_evidence(
            root, identity, terminal_kind=kind,
            evidence_chain_sha256=chain_hash, outcome="PHASE_A_ACCEPTED",
            accepted_attempt_index=1, rejection_codes=[],
            completed_arm_ids=[], tag="phase-terminal")
        payload = {
            "status": "PHASE_A_ACCEPTED",
            "evidence_chain_sha256": chain_hash,
            "terminal_evidence_sha256": evidence["sha256"],
            "validation_receipt_sha256": receipt["sha256"],
        }
    else:
        kind = "TERMINAL_PHASE_A_REJECTED"
        evidence, receipt = _terminal_evidence(
            root, identity, terminal_kind=kind,
            evidence_chain_sha256=chain_hash,
            outcome="ALL_RENDER_ATTEMPTS_REJECTED",
            accepted_attempt_index=None,
            rejection_codes=["LEAKAGE_REVIEW"] * 3,
            completed_arm_ids=[], tag="phase-rejected-terminal")
        payload = {
            "status": "ALL_RENDER_ATTEMPTS_REJECTED",
            "attempt_count": 3,
            "evidence_chain_sha256": chain_hash,
            "rejection_evidence_sha256": evidence["sha256"],
            "validation_receipt_sha256": receipt["sha256"],
        }
    return _append(
        root, identity, kind, payload=payload, artifacts=[evidence, receipt])


def _append_treatment_terminal(root: Path, identity):
    chain_hash = _chain_hash(root, identity)
    evidence, receipt = _terminal_evidence(
        root, identity, terminal_kind="TERMINAL_TREATMENT",
        evidence_chain_sha256=chain_hash, outcome="TREATMENT_COMPLETE",
        accepted_attempt_index=None, rejection_codes=[],
        completed_arm_ids=list(PRIMARY_ARMS), tag="treatment-terminal")
    return _append(
        root, identity, "TERMINAL_TREATMENT", arms=PRIMARY_ARMS,
        payload={
            "status": "TREATMENT_COMPLETE",
            "evidence_chain_sha256": chain_hash,
            "terminal_evidence_sha256": evidence["sha256"],
            "validation_receipt_sha256": receipt["sha256"],
        }, artifacts=[evidence, receipt])


def _append_technical_terminal(root: Path, identity):
    chain_hash = _chain_hash(root, identity)
    evidence, receipt = _terminal_evidence(
        root, identity, terminal_kind="TERMINAL_TECHNICAL",
        evidence_chain_sha256=chain_hash, outcome="TECHNICAL_COMPLETE",
        accepted_attempt_index=None, rejection_codes=[],
        completed_arm_ids=list(PRIMARY_ARMS), tag="technical-terminal")
    return _append(
        root, identity, "TERMINAL_TECHNICAL", arms=PRIMARY_ARMS,
        payload={
            "status": "TECHNICAL_COMPLETE",
            "evidence_chain_sha256": chain_hash,
            "terminal_evidence_sha256": evidence["sha256"],
            "validation_receipt_sha256": receipt["sha256"],
        }, artifacts=[evidence, receipt])


def _finalize(root: Path, identity):
    return store._finalize_case(
        root, identity=identity, pid=PID, process_start_token=START,
        _test_process_start_token_reader=_token_reader)


def _sigkill_quarantine_at_boundary(root_text: str, boundary: str) -> None:
    def fault(observed: str) -> None:
        if observed == boundary:
            os.kill(os.getpid(), signal.SIGKILL)

    store._quarantine_abandoned_case(
        Path(root_text), _test_process_probe=lambda pid, token: False,
        _test_fault_hook=fault, observed_utc="2026-07-12T18:48:00Z",
        reason="SIGKILL fault injection")


def _linux_begin_then_exit(root_text: str) -> None:
    pid = os.getpid()
    token = store.linux_process_start_token(pid)
    store.begin_case(
        Path(root_text), identity=_identity(), bounds=_bounds(), pid=pid,
        process_start_token=token, created_utc=T0,
        payload={"status": "STARTED"})


def _complete_phase_a(root: Path, identity=None):
    identity = identity or _identity("phase_a")
    _begin(root, identity)
    payload, artifacts = _attempt_evidence(root, 1, "ACCEPTED", "attempt-1")
    _append(root, identity, "RENDER_ATTEMPT", payload=payload, artifacts=artifacts)
    payload, artifacts = _plans_evidence(root)
    _append(root, identity, "PLANS_PHASE_A", payload=payload, artifacts=artifacts)
    payload, artifacts = _foundation_evidence(root)
    _append(root, identity, "FOUNDATION_BUNDLE", payload=payload, artifacts=artifacts)
    _append_phase_terminal(root, identity, accepted=True)
    return identity, _finalize(root, identity)


def _complete_rejected_phase_a(root: Path, identity=None):
    identity = identity or _identity("phase_a")
    _begin(root, identity)
    for index in range(1, 4):
        payload, artifacts = _attempt_evidence(
            root, index, "REJECTED", f"attempt-{index}")
        _append(root, identity, "RENDER_ATTEMPT",
                payload=payload, artifacts=artifacts)
    _append_phase_terminal(root, identity, accepted=False)
    return identity, _finalize(root, identity)


def _complete_treatment(root: Path):
    identity = _identity("treatment")
    _begin(root, identity)
    payload, artifacts = _foundation_evidence(root, "treatment-foundation")
    _append(root, identity, "FOUNDATION_LOAD", payload=payload, artifacts=artifacts)
    for index, arm in enumerate(PRIMARY_ARMS, start=1):
        payload, artifacts = _arm_evidence(root, arm)
        _append(root, identity, f"ARM_{arm}", arms=PRIMARY_ARMS[:index],
                payload=payload, artifacts=artifacts)
    _append_treatment_terminal(root, identity)
    return identity, _finalize(root, identity)


def _complete_technical(root: Path):
    identity = _identity("technical")
    _begin(root, identity)
    payload, artifacts = _foundation_evidence(root, "technical-foundation")
    _append(root, identity, "FOUNDATION_LOAD", payload=payload,
            artifacts=artifacts)
    for index, arm in enumerate(PRIMARY_ARMS, start=1):
        payload, artifacts = _arm_evidence(root, arm, f"technical-{arm}")
        _append(root, identity, f"ARM_{arm}", arms=PRIMARY_ARMS[:index],
                payload=payload, artifacts=artifacts)
    _append_technical_terminal(root, identity)
    return identity, _finalize(root, identity)


def test_identity_key_is_exact_release_runtime_batch_case_render_schedule_mode():
    identity = _identity()
    assert store.validate_identity(identity) == identity
    changed = deepcopy(identity)
    changed["extra"] = True
    with pytest.raises(store.V13StoreError, match="field set"):
        store.validate_identity(changed)


def test_bounds_enforce_one_case_render_fcw_six_arm_prefix_and_hard_limits():
    assert store.validate_bounds(_bounds(PRIMARY_ARMS[:3]))[
        "completed_arm_ids"] == list(PRIMARY_ARMS[:3])
    with pytest.raises(store.V13StoreError, match="frozen-order prefix"):
        _bounds(("CC",))
    with pytest.raises(store.V13StoreError, match="0..7000"):
        _bounds(tokens=7001)
    with pytest.raises(store.V13StoreError, match="0..256"):
        _bounds(rows=257)


def test_begin_persists_exact_lock_and_started_before_case_work(tmp_path: Path):
    identity = _identity()
    lock = _begin(tmp_path, identity)
    assert lock["process_start_token"] == START
    assert len(lock["attempt_id"]) == 32
    int(lock["attempt_id"], 16)
    records = list((tmp_path / lock["case_dir"] / "records").iterdir())
    assert [path.name for path in records] == ["000_STARTED.json"]
    assert records[0].read_bytes().endswith(b"\n")


def test_begin_rejects_unverified_process_identity(tmp_path: Path):
    with pytest.raises(store.V13StoreError, match="process-start identity"):
        store._begin_case(
            tmp_path, identity=_identity(), bounds=_bounds(), pid=PID,
            process_start_token=START, created_utc=T0,
            _test_process_start_token_reader=lambda pid: START[:-1] + "7")


def test_second_active_case_and_wrong_process_cannot_append(tmp_path: Path):
    identity = _identity()
    _begin(tmp_path, identity)
    with pytest.raises(store.V13StoreError, match="another active-case"):
        _begin(tmp_path, _identity(case_id="b" * 64))
    with pytest.raises(store.V13StoreError, match="differs from active lock"):
        store._append_case_record(
            tmp_path, identity=identity, pid=9999,
            process_start_token=START, record_kind="RENDER_ATTEMPT",
            bounds=_bounds(), created_utc=T0,
            payload=_attempt_evidence(tmp_path, 1, "ACCEPTED", "wrong")[0],
            artifact_bindings=_attempt_evidence(
                tmp_path, 1, "ACCEPTED", "wrong2")[1],
            _test_process_start_token_reader=_token_reader)


def test_phase_a_explicit_three_rejections_terminalize_without_foundation(
        tmp_path: Path):
    identity, index = _complete_rejected_phase_a(tmp_path)
    assert index["terminal_record"]["name"].endswith(
        "_TERMINAL_PHASE_A_REJECTED.json")
    assert store.terminal_reuse_binding(
        tmp_path, expected_identity=identity)["reusable"] is True


def test_phase_a_acceptance_stops_attempts_and_is_required_for_plans(tmp_path: Path):
    identity = _identity("phase_a")
    _begin(tmp_path, identity)
    payload, artifacts = _attempt_evidence(tmp_path, 1, "REJECTED", "r1")
    _append(tmp_path, identity, "RENDER_ATTEMPT", payload=payload, artifacts=artifacts)
    plans_payload, plans_artifacts = _plans_evidence(tmp_path, "too-early")
    with pytest.raises(store.V13StoreError, match="invalid Phase-A"):
        _append(tmp_path, identity, "PLANS_PHASE_A",
                payload=plans_payload, artifacts=plans_artifacts)
    payload, artifacts = _attempt_evidence(tmp_path, 2, "ACCEPTED", "a2")
    _append(tmp_path, identity, "RENDER_ATTEMPT", payload=payload, artifacts=artifacts)
    payload, artifacts = _attempt_evidence(tmp_path, 3, "REJECTED", "late")
    with pytest.raises(store.V13StoreError, match="invalid Phase-A"):
        _append(tmp_path, identity, "RENDER_ATTEMPT", payload=payload, artifacts=artifacts)
    _append(tmp_path, identity, "PLANS_PHASE_A",
            payload=plans_payload, artifacts=plans_artifacts)


def test_phase_a_attempt_indices_are_exact_and_fourth_attempt_is_forbidden(
        tmp_path: Path):
    identity = _identity("phase_a")
    _begin(tmp_path, identity)
    payload, artifacts = _attempt_evidence(tmp_path, 2, "REJECTED", "wrong-index")
    with pytest.raises(store.V13StoreError, match="invalid Phase-A"):
        _append(tmp_path, identity, "RENDER_ATTEMPT", payload=payload, artifacts=artifacts)
    for index in range(1, 4):
        payload, artifacts = _attempt_evidence(
            tmp_path, index, "REJECTED", f"r-{index}")
        _append(tmp_path, identity, "RENDER_ATTEMPT",
                payload=payload, artifacts=artifacts)
    payload, artifacts = _attempt_evidence(tmp_path, 3, "REJECTED", "fourth")
    with pytest.raises(store.V13StoreError, match="invalid Phase-A"):
        _append(tmp_path, identity, "RENDER_ATTEMPT", payload=payload, artifacts=artifacts)


def test_empty_or_arbitrary_stage_evidence_never_terminalizes(tmp_path: Path):
    identity = _identity("phase_a")
    _begin(tmp_path, identity)
    with pytest.raises(store.V13StoreError, match="payload field set"):
        _append(tmp_path, identity, "RENDER_ATTEMPT", payload={}, artifacts=[])


def test_treatment_requires_exact_six_arm_order_and_bound_prefix(tmp_path: Path):
    identity = _identity("treatment")
    _begin(tmp_path, identity)
    payload, artifacts = _foundation_evidence(tmp_path, "t-found")
    _append(tmp_path, identity, "FOUNDATION_LOAD", payload=payload, artifacts=artifacts)
    payload, artifacts = _arm_evidence(tmp_path, "CC", "wrong-order")
    with pytest.raises(store.V13StoreError, match="transition"):
        _append(tmp_path, identity, "ARM_CC", arms=PRIMARY_ARMS[:2],
                payload=payload, artifacts=artifacts)
    payload, artifacts = _arm_evidence(tmp_path, "FF", "wrong-prefix")
    with pytest.raises(store.V13StoreError, match="completed-arm"):
        _append(tmp_path, identity, "ARM_FF", arms=(),
                payload=payload, artifacts=artifacts)
    for index, arm in enumerate(PRIMARY_ARMS, start=1):
        payload, artifacts = _arm_evidence(tmp_path, arm, f"valid-{arm}")
        _append(tmp_path, identity, f"ARM_{arm}", arms=PRIMARY_ARMS[:index],
                payload=payload, artifacts=artifacts)


def test_bound_artifact_must_exist_match_payload_and_literal_bytes(tmp_path: Path):
    identity = _identity("phase_a")
    _begin(tmp_path, identity)
    payload, bindings = _attempt_evidence(tmp_path, 1, "ACCEPTED", "bytes")
    changed = deepcopy(bindings)
    changed[0]["sha256"] = "f" * 64
    with pytest.raises(store.V13StoreError, match="hash differs|bytes differ"):
        _append(tmp_path, identity, "RENDER_ATTEMPT",
                payload=payload, artifacts=changed)
    missing = deepcopy(bindings)
    missing[0]["path"] = "artifacts/missing.bin"
    with pytest.raises(store.V13StoreError, match="absent"):
        _append(tmp_path, identity, "RENDER_ATTEMPT",
                payload=payload, artifacts=missing)


def test_bound_artifact_rejects_symlink_and_escape(tmp_path: Path):
    identity = _identity("phase_a")
    _begin(tmp_path, identity)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"outside")
    link = tmp_path / "artifacts" / "linked.bin"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    attempt = {
        "kind": "render_attempt", "path": "artifacts/linked.bin",
        "sha256": store.file_sha256(outside),
        "size_bytes": outside.stat().st_size,
    }
    review = _binding(tmp_path, "render_review", "link-review")
    payload = {
        "attempt_index": 1, "outcome": "ACCEPTED",
        "attempt_sha256": attempt["sha256"],
        "review_sha256": review["sha256"], "rejection_code": None,
    }
    with pytest.raises(store.V13StoreError, match="escapes|symlink"):
        _append(tmp_path, identity, "RENDER_ATTEMPT",
                payload=payload, artifacts=[attempt, review])


def test_complete_phase_a_has_binding_only_index_and_exact_terminal_reuse(
        tmp_path: Path):
    identity, index = _complete_phase_a(tmp_path)
    assert index["status"] == "TERMINAL_VALIDATED"
    assert not (tmp_path / store.ACTIVE_LOCK_NAME).exists()
    assert "payload" not in index and "artifacts" not in index
    reuse = store.terminal_reuse_binding(tmp_path, expected_identity=identity)
    assert reuse["reusable"] is True


def test_complete_treatment_is_terminal_only_after_all_six_arms(tmp_path: Path):
    identity, index = _complete_treatment(tmp_path)
    assert index["terminal_record"]["name"].endswith(
        "_TERMINAL_TREATMENT.json")
    assert store.terminal_reuse_binding(
        tmp_path, expected_identity=identity)["reusable"] is True


def test_technical_mode_has_distinct_non_treatment_terminal(tmp_path: Path):
    identity, index = _complete_technical(tmp_path)
    assert index["terminal_record"]["name"].endswith(
        "_TERMINAL_TECHNICAL.json")
    assert "TREATMENT" not in index["terminal_record"]["name"]
    reuse = store.terminal_reuse_binding(
        tmp_path, expected_identity=identity)
    assert reuse["terminal_record_name"].endswith(
        "_TERMINAL_TECHNICAL.json")


def test_terminal_evidence_chain_hash_is_recomputed(tmp_path: Path):
    identity = _identity("phase_a")
    _begin(tmp_path, identity)
    payload, artifacts = _attempt_evidence(tmp_path, 1, "ACCEPTED", "a")
    _append(tmp_path, identity, "RENDER_ATTEMPT", payload=payload, artifacts=artifacts)
    payload, artifacts = _plans_evidence(tmp_path)
    _append(tmp_path, identity, "PLANS_PHASE_A", payload=payload, artifacts=artifacts)
    payload, artifacts = _foundation_evidence(tmp_path)
    _append(tmp_path, identity, "FOUNDATION_BUNDLE", payload=payload, artifacts=artifacts)
    evidence, receipt = _terminal_evidence(
        tmp_path, identity, terminal_kind="TERMINAL_PHASE_A_ACCEPTED",
        evidence_chain_sha256="0" * 64, outcome="PHASE_A_ACCEPTED",
        accepted_attempt_index=1, rejection_codes=[], completed_arm_ids=[],
        tag="bad-terminal")
    with pytest.raises(store.V13StoreError, match="evidence-chain"):
        _append(tmp_path, identity, "TERMINAL_PHASE_A_ACCEPTED", payload={
            "status": "PHASE_A_ACCEPTED", "evidence_chain_sha256": "0" * 64,
            "terminal_evidence_sha256": evidence["sha256"],
            "validation_receipt_sha256": receipt["sha256"],
        }, artifacts=[evidence, receipt])


def test_opaque_terminal_evidence_cannot_produce_terminal_validated(tmp_path: Path):
    identity = _identity("phase_a")
    _begin(tmp_path, identity)
    payload, artifacts = _attempt_evidence(tmp_path, 1, "ACCEPTED", "a")
    _append(tmp_path, identity, "RENDER_ATTEMPT", payload=payload, artifacts=artifacts)
    payload, artifacts = _plans_evidence(tmp_path)
    _append(tmp_path, identity, "PLANS_PHASE_A", payload=payload, artifacts=artifacts)
    payload, artifacts = _foundation_evidence(tmp_path)
    _append(tmp_path, identity, "FOUNDATION_BUNDLE", payload=payload, artifacts=artifacts)
    chain_hash = _chain_hash(tmp_path, identity)
    evidence = _binding(
        tmp_path, "phase_a_terminal_evidence", "opaque", b"opaque bytes")
    receipt_document = {
        "schema": store.TERMINAL_VALIDATION_SCHEMA,
        "design_id": "coherent-state-powered-successor-v13",
        "identity_sha256": store.identity_sha256(identity),
        "terminal_kind": "TERMINAL_PHASE_A_ACCEPTED",
        "evidence_chain_sha256": chain_hash,
        "terminal_evidence_sha256": evidence["sha256"],
        "validator_id": store.TERMINAL_VALIDATOR_ID,
        "validation_status": "PASS", "validated_utc": T0,
    }
    receipt = _binding(
        tmp_path, "independent_terminal_validation_receipt", "opaque",
        store.canonical_json_bytes(receipt_document) + b"\n")
    with pytest.raises(store.V13StoreError, match="strict UTF-8 JSON"):
        _append(tmp_path, identity, "TERMINAL_PHASE_A_ACCEPTED", payload={
            "status": "PHASE_A_ACCEPTED",
            "evidence_chain_sha256": chain_hash,
            "terminal_evidence_sha256": evidence["sha256"],
            "validation_receipt_sha256": receipt["sha256"],
        }, artifacts=[evidence, receipt])


def test_partial_case_never_has_reuse_binding(tmp_path: Path):
    identity = _identity("phase_a")
    _begin(tmp_path, identity)
    payload, artifacts = _attempt_evidence(tmp_path, 1, "REJECTED", "partial")
    _append(tmp_path, identity, "RENDER_ATTEMPT", payload=payload, artifacts=artifacts)
    with pytest.raises(store.V13StoreError, match="cannot read session index"):
        store.terminal_reuse_binding(tmp_path, expected_identity=identity)


def test_terminal_reuse_requires_exact_runtime_batch_and_gpu_identity(
        tmp_path: Path):
    identity, _index = _complete_phase_a(tmp_path)
    for changed in (
        _identity(runtime_fingerprint_sha256="e" * 64),
        _identity(primary_batch_id="primary-002"),
        _identity(gpu_uuid="GPU-other"),
    ):
        with pytest.raises(store.V13StoreError, match="cannot read session index"):
            store.terminal_reuse_binding(tmp_path, expected_identity=changed)
    assert identity != changed


def test_begin_exact_terminal_raises_explicit_already_terminal(tmp_path: Path):
    identity, _index = _complete_phase_a(tmp_path)
    with pytest.raises(store.V13AlreadyTerminal) as caught:
        _begin(tmp_path, identity)
    assert caught.value.binding["reusable"] is True
    assert not (tmp_path / store.ACTIVE_LOCK_NAME).exists()
    assert not list((tmp_path / "active").iterdir())


def test_record_or_bound_artifact_mutation_invalidates_terminal_reuse(
        tmp_path: Path):
    identity, index = _complete_phase_a(tmp_path)
    terminal = tmp_path / index["terminal_directory"]
    record = sorted((terminal / "records").iterdir())[1]
    record.write_bytes(record.read_bytes().replace(b"ACCEPTED", b"REJECTED"))
    with pytest.raises(store.V13StoreError):
        store.terminal_reuse_binding(tmp_path, expected_identity=identity)


def test_alive_process_cannot_be_quarantined_and_probe_gets_start_token(
        tmp_path: Path):
    _begin(tmp_path, _identity())
    calls = []

    def alive(pid, token):
        calls.append((pid, token))
        return True

    with pytest.raises(store.V13StoreError, match="still alive"):
        store._quarantine_abandoned_case(
            tmp_path, _test_process_probe=alive,
            observed_utc="2026-07-12T18:41:00Z", reason="synthetic kill")
    assert calls == [(PID, START)]
    assert (tmp_path / store.ACTIVE_LOCK_NAME).exists()


def test_dead_partial_is_preserved_bound_and_never_reusable(tmp_path: Path):
    identity = _identity()
    _begin(tmp_path, identity)
    payload, artifacts = _attempt_evidence(tmp_path, 1, "REJECTED", "partial")
    _append(tmp_path, identity, "RENDER_ATTEMPT", payload=payload, artifacts=artifacts)
    record = store._quarantine_abandoned_case(
        tmp_path, _test_process_probe=lambda pid, token: False,
        observed_utc="2026-07-12T18:42:00Z", reason="hard kill")
    assert record["partial_reusable"] is False
    assert not (tmp_path / store.ACTIVE_LOCK_NAME).exists()
    quarantine = list((tmp_path / "quarantine").iterdir())
    assert len(quarantine) == 1
    assert (quarantine[0] / "records" / "001_RENDER_ATTEMPT.json").exists()
    assert (quarantine[0] / store.ARCHIVED_LOCK_NAME).exists()
    archived = quarantine[0] / store.ARCHIVED_ARTIFACT_DIRECTORY
    assert archived.is_dir()
    assert any(path.is_file() for path in archived.rglob("*"))
    archived_attempt = archived / artifacts[0]["path"]
    assert store.file_sha256(archived_attempt) == artifacts[0]["sha256"]
    (tmp_path / artifacts[0]["path"]).write_bytes(b"mutated external source")
    assert store.file_sha256(archived_attempt) == artifacts[0]["sha256"]
    assert store._read_valid_quarantine(quarantine[0]) == record
    assert (quarantine[0] / store.QUARANTINE_RECORD_NAME).exists()
    with pytest.raises(store.V13StoreError):
        store.terminal_reuse_binding(tmp_path, expected_identity=identity)


def test_quarantine_retry_uses_stable_transaction_not_new_timestamp(tmp_path: Path):
    _begin(tmp_path, _identity())
    transaction = store.active_quarantine_transaction_sha256(tmp_path)
    first = store._quarantine_abandoned_case(
        tmp_path, transaction_sha256=transaction,
        _test_process_probe=lambda pid, token: False,
        observed_utc="2026-07-12T18:50:00Z", reason="first observation")
    retried = store._quarantine_abandoned_case(
        tmp_path, transaction_sha256=transaction,
        _test_process_probe=lambda pid, token: False,
        observed_utc="2026-07-12T18:51:00Z", reason="later retry")
    assert retried == first
    unambiguous = store._quarantine_abandoned_case(
        tmp_path, _test_process_probe=lambda pid, token: False,
        observed_utc="2026-07-12T18:52:00Z", reason="another retry")
    assert unambiguous == first


def test_same_second_same_identity_retries_get_distinct_quarantine_transactions(
        tmp_path: Path):
    identity = _identity()
    first_lock = _begin(tmp_path, identity)
    payload, artifacts = _attempt_evidence(
        tmp_path, 1, "REJECTED", "first-partial")
    _append(tmp_path, identity, "RENDER_ATTEMPT",
            payload=payload, artifacts=artifacts)
    first = store._quarantine_abandoned_case(
        tmp_path, _test_process_probe=lambda pid, token: False,
        observed_utc=T0, reason="same-second first")

    second_lock = _begin(tmp_path, identity)
    assert second_lock["attempt_id"] != first_lock["attempt_id"]
    payload, artifacts = _attempt_evidence(
        tmp_path, 1, "REJECTED", "second-partial")
    _append(tmp_path, identity, "RENDER_ATTEMPT",
            payload=payload, artifacts=artifacts)
    second = store._quarantine_abandoned_case(
        tmp_path, _test_process_probe=lambda pid, token: False,
        observed_utc=T0, reason="same-second second")
    assert first["transaction_sha256"] != second["transaction_sha256"]
    assert len(list((tmp_path / "quarantine").iterdir())) == 2


@pytest.mark.parametrize("fault_boundary", [
    "after_lock_archive_link", "after_lock_archive",
    "after_partial_artifact_archive",
    "after_quarantine_marker", "after_quarantine_rename_before_fsync",
    "after_quarantine_rename", "after_quarantine_lock_unlink",
    "after_quarantine_lock_clear",
])
def test_quarantine_is_idempotent_after_every_durable_boundary(
        tmp_path: Path, fault_boundary: str):
    root = tmp_path / fault_boundary
    identity = _identity(case_id=store.sha256_bytes(fault_boundary.encode()))
    _begin(root, identity)
    payload, artifacts = _attempt_evidence(root, 1, "REJECTED", "partial")
    _append(root, identity, "RENDER_ATTEMPT", payload=payload, artifacts=artifacts)
    fired = False

    def fault(boundary):
        nonlocal fired
        if boundary == fault_boundary and not fired:
            fired = True
            raise RuntimeError(boundary)

    with pytest.raises(RuntimeError, match=fault_boundary):
        store._quarantine_abandoned_case(
            root, _test_process_probe=lambda pid, token: False,
            _test_fault_hook=fault, observed_utc="2026-07-12T18:45:00Z",
            reason="fault injection")
    result = store._quarantine_abandoned_case(
        root, _test_process_probe=lambda pid, token: False,
        observed_utc="2026-07-12T18:45:00Z", reason="fault injection")
    again = store._quarantine_abandoned_case(
        root, _test_process_probe=lambda pid, token: False,
        observed_utc="2026-07-12T18:45:00Z", reason="fault injection")
    assert result == again and result["partial_reusable"] is False
    assert not (root / store.ACTIVE_LOCK_NAME).exists()
    assert not list((root / "active").iterdir())
    assert len(list((root / "quarantine").iterdir())) == 1
    next_identity = _identity(case_id=store.sha256_bytes(
        f"next:{fault_boundary}".encode()))
    assert _begin(root, next_identity)["identity"] == next_identity


@pytest.mark.parametrize("fault_boundary", [
    "after_lock_archive_link", "after_lock_archive",
    "after_partial_artifact_archive",
    "after_quarantine_marker", "after_quarantine_rename_before_fsync",
    "after_quarantine_rename", "after_quarantine_lock_unlink",
    "after_quarantine_lock_clear",
])
def test_sigkill_at_each_quarantine_boundary_recovers_exactly_once(
        tmp_path: Path, fault_boundary: str):
    root = tmp_path / f"sigkill-{fault_boundary}"
    identity = _identity(case_id=store.sha256_bytes(
        f"sigkill:{fault_boundary}".encode()))
    _begin(root, identity)
    payload, artifacts = _attempt_evidence(root, 1, "REJECTED", "partial")
    _append(root, identity, "RENDER_ATTEMPT", payload=payload, artifacts=artifacts)
    context = multiprocessing.get_context("fork")
    process = context.Process(
        target=_sigkill_quarantine_at_boundary,
        args=(str(root), fault_boundary))
    process.start()
    process.join(timeout=10)
    assert process.exitcode == -signal.SIGKILL
    recovered = store._quarantine_abandoned_case(
        root, _test_process_probe=lambda pid, token: False,
        observed_utc="2026-07-12T18:48:00Z",
        reason="SIGKILL fault injection")
    again = store._quarantine_abandoned_case(
        root, _test_process_probe=lambda pid, token: False,
        observed_utc="2026-07-12T18:48:00Z",
        reason="SIGKILL fault injection")
    assert recovered == again
    assert len(list((root / "quarantine").iterdir())) == 1
    assert not (root / store.ACTIVE_LOCK_NAME).exists()


def test_concurrent_quarantine_recovery_has_one_identical_receipt(tmp_path: Path):
    identity = _identity()
    _begin(tmp_path, identity)
    payload, artifacts = _attempt_evidence(tmp_path, 1, "REJECTED", "partial")
    _append(tmp_path, identity, "RENDER_ATTEMPT", payload=payload, artifacts=artifacts)

    def recover():
        return store._quarantine_abandoned_case(
            tmp_path, _test_process_probe=lambda pid, token: False,
            observed_utc="2026-07-12T18:47:00Z", reason="concurrent recovery")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _index: recover(), range(2)))
    assert first == second
    assert len(list((tmp_path / "quarantine").iterdir())) == 1
    assert not (tmp_path / store.ACTIVE_LOCK_NAME).exists()


def test_lock_only_begin_crash_is_quarantined_idempotently(tmp_path: Path):
    identity = _identity()
    lock = _begin(tmp_path, identity)
    active = tmp_path / lock["case_dir"]
    for path in sorted(active.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    active.rmdir()
    result = store._quarantine_abandoned_case(
        tmp_path, _test_process_probe=lambda pid, token: False,
        observed_utc="2026-07-12T18:46:00Z", reason="begin crash")
    assert result["partial_reusable"] is False
    assert any(row["path"] == store.ARCHIVED_LOCK_NAME
               for row in result["partial_manifest"])


def test_crash_after_terminal_directory_promotion_finishes_index_not_quarantine(
        tmp_path: Path):
    identity = _identity("phase_a")
    lock = _begin(tmp_path, identity)
    payload, artifacts = _attempt_evidence(tmp_path, 1, "ACCEPTED", "a")
    _append(tmp_path, identity, "RENDER_ATTEMPT", payload=payload, artifacts=artifacts)
    payload, artifacts = _plans_evidence(tmp_path)
    _append(tmp_path, identity, "PLANS_PHASE_A", payload=payload, artifacts=artifacts)
    payload, artifacts = _foundation_evidence(tmp_path)
    _append(tmp_path, identity, "FOUNDATION_BUNDLE", payload=payload, artifacts=artifacts)
    _append_phase_terminal(tmp_path, identity, accepted=True)
    digest = store.identity_sha256(identity)
    (tmp_path / lock["case_dir"]).rename(tmp_path / "terminal" / digest)
    result = store._quarantine_abandoned_case(
        tmp_path, _test_process_probe=lambda pid, token: False,
        observed_utc="2026-07-12T18:43:00Z", reason="kill after rename")
    assert result["status"] == "TERMINAL_VALIDATED"
    assert not list((tmp_path / "quarantine").iterdir())


def test_probe_must_return_literal_boolean(tmp_path: Path):
    _begin(tmp_path, _identity())
    with pytest.raises(store.V13StoreError, match="did not return boolean"):
        store._quarantine_abandoned_case(
            tmp_path, _test_process_probe=lambda pid, token: 0,
            observed_utc="2026-07-12T18:44:00Z", reason="bad probe")


def test_production_quarantine_has_no_public_probe_override():
    assert "process_probe" not in inspect.signature(
        store.quarantine_abandoned_case).parameters


def test_production_store_apis_expose_no_test_or_dead_owner_bypass():
    for function in (
        store.begin_case, store.append_case_record,
        store.active_evidence_chain_sha256, store.finalize_case,
    ):
        assert not any(name.startswith("_test") or name.startswith("_dead_owner")
                       for name in inspect.signature(function).parameters)


def test_linux_proc_parser_uses_boot_id_and_final_comm_parenthesis(tmp_path: Path):
    proc = tmp_path / "proc"
    (proc / "123").mkdir(parents=True)
    boot_path = tmp_path / "boot_id"
    boot_path.write_text(BOOT_ID + "\n", encoding="ascii")
    suffix = [b"S"] + [b"0"] * 18 + [b"987654"] + [b"0"]
    (proc / "123" / "stat").write_bytes(
        b"123 (odd ) name \xff) " + b" ".join(suffix) + b"\n")
    assert store._linux_process_start_token(
        123, proc_root=proc, boot_id_path=boot_path
    ) == f"linux-procfs-v1:{BOOT_ID}:987654"


def test_linux_proc_esrch_is_exact_dead_process_proof(
        tmp_path: Path, monkeypatch):
    boot_path = tmp_path / "boot_id"
    boot_path.write_text(BOOT_ID + "\n", encoding="ascii")
    proc = tmp_path / "proc"
    original = Path.read_bytes

    def esrch(path: Path):
        if path.name == "stat":
            raise ProcessLookupError(errno.ESRCH, "synthetic ESRCH")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", esrch)
    with pytest.raises(ProcessLookupError):
        store._linux_process_start_token(
            123, proc_root=proc, boot_id_path=boot_path)


@pytest.mark.skipif(sys.platform != "linux", reason="production procfs gate is Linux-only")
def test_real_linux_child_live_dead_pid_reuse_and_public_quarantine(tmp_path: Path):
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        token = store.linux_process_start_token(child.pid)
        assert store.process_identity_is_alive(child.pid, token)
        prefix, ticks = token.rsplit(":", 1)
        assert not store.process_identity_is_alive(
            child.pid, f"{prefix}:{int(ticks) + 1}")
    finally:
        child.terminate()
        child.wait(timeout=10)
    assert not store.process_identity_is_alive(child.pid, token)
    abandoned_root = tmp_path / "real-abandoned-child"
    context = multiprocessing.get_context("fork")
    owner = context.Process(
        target=_linux_begin_then_exit, args=(str(abandoned_root),))
    owner.start()
    owner.join(timeout=10)
    assert owner.exitcode == 0
    quarantined = store.quarantine_abandoned_case(
        abandoned_root, observed_utc="2026-07-12T18:49:00Z",
        reason="real child exited")
    assert quarantined["process_probe"]["alive"] is False
    assert not (abandoned_root / store.ACTIVE_LOCK_NAME).exists()


def test_exclusive_json_never_overwrites(tmp_path: Path):
    path = tmp_path / "immutable.json"
    store.write_json_exclusive(path, {"value": 1})
    original = path.read_bytes()
    with pytest.raises(store.V13StoreError, match="already exists"):
        store.write_json_exclusive(path, {"value": 2})
    assert path.read_bytes() == original
