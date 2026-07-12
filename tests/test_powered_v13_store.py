from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import powered_v13_store as store
from powered_v13_schema import N_SCHEDULE, PRIMARY_ARMS


T0 = "2026-07-12T18:40:00Z"
PID = 4242
START = "pid-4242-start-123456"


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


def _artifact(root: Path, name: str = "artifacts/foundation.bin",
              content: bytes = b"lossless-state"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "kind": "tensor_bundle",
        "path": name,
        "sha256": store.file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _begin(root: Path, identity=None):
    identity = identity or _identity()
    return store.begin_case(
        root, identity=identity, bounds=_bounds(arms=()), pid=PID,
        process_start_token=START, created_utc=T0,
        payload={"release_checked": True})


def _append(root: Path, identity, kind: str, *, arms=(), artifacts=()):
    return store.append_case_record(
        root, identity=identity, pid=PID, process_start_token=START,
        record_kind=kind, bounds=_bounds(arms=arms), created_utc=T0,
        payload={"kind": kind}, artifact_bindings=artifacts)


def _complete_phase_a(root: Path, identity=None):
    identity = identity or _identity("phase_a")
    _begin(root, identity)
    _append(root, identity, "RENDER_ATTEMPT")
    _append(root, identity, "PLANS_PHASE_A")
    artifact = _artifact(root)
    _append(root, identity, "FOUNDATION_BUNDLE", artifacts=[artifact])
    _append(root, identity, "TERMINAL_PHASE_A")
    return identity, store.finalize_case(
        root, identity=identity, pid=PID, process_start_token=START)


def _complete_treatment(root: Path):
    identity = _identity("treatment")
    _begin(root, identity)
    artifact = _artifact(root)
    _append(root, identity, "FOUNDATION_LOAD", artifacts=[artifact])
    for index, arm in enumerate(PRIMARY_ARMS, start=1):
        _append(root, identity, f"ARM_{arm}", arms=PRIMARY_ARMS[:index])
    _append(root, identity, "TERMINAL_TREATMENT", arms=PRIMARY_ARMS)
    return identity, store.finalize_case(
        root, identity=identity, pid=PID, process_start_token=START)


def test_identity_key_is_exact_release_runtime_batch_case_render_schedule_mode():
    identity = _identity()
    assert store.validate_identity(identity) == identity
    assert identity["design_id"] == "coherent-state-powered-successor-v13"
    assert len(store.identity_sha256(identity)) == 64
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


def test_begin_persists_lock_and_started_before_case_work(tmp_path: Path):
    identity = _identity()
    lock = _begin(tmp_path, identity)
    lock_path = tmp_path / store.ACTIVE_LOCK_NAME
    assert lock_path.exists()
    assert lock["pid"] == PID and lock["process_start_token"] == START
    case_dir = tmp_path / lock["case_dir"] / "records"
    records = list(case_dir.iterdir())
    assert [path.name for path in records] == ["000_STARTED.json"]
    assert records[0].read_bytes().endswith(b"\n")


def test_second_active_case_and_wrong_process_cannot_append(tmp_path: Path):
    identity = _identity()
    _begin(tmp_path, identity)
    with pytest.raises(store.V13StoreError, match="another active-case"):
        _begin(tmp_path, _identity(case_id="b" * 64))
    with pytest.raises(store.V13StoreError, match="differs from active lock"):
        store.append_case_record(
            tmp_path, identity=identity, pid=9999,
            process_start_token=START, record_kind="RENDER_ATTEMPT",
            bounds=_bounds(), created_utc=T0)


def test_phase_a_sequence_allows_up_to_three_attempts_then_exact_foundation(
        tmp_path: Path):
    identity = _identity("phase_a")
    _begin(tmp_path, identity)
    for _ in range(3):
        _append(tmp_path, identity, "RENDER_ATTEMPT")
    with pytest.raises(store.V13StoreError, match="invalid Phase-A"):
        _append(tmp_path, identity, "RENDER_ATTEMPT")
    _append(tmp_path, identity, "PLANS_PHASE_A")
    binding = _artifact(tmp_path)
    _append(tmp_path, identity, "FOUNDATION_BUNDLE", artifacts=[binding])
    _append(tmp_path, identity, "TERMINAL_PHASE_A")


def test_phase_a_cannot_skip_attempt_or_bundle(tmp_path: Path):
    identity = _identity("phase_a")
    _begin(tmp_path, identity)
    with pytest.raises(store.V13StoreError, match="invalid Phase-A"):
        _append(tmp_path, identity, "PLANS_PHASE_A")
    _append(tmp_path, identity, "RENDER_ATTEMPT")
    with pytest.raises(store.V13StoreError, match="invalid Phase-A"):
        _append(tmp_path, identity, "FOUNDATION_BUNDLE")


def test_treatment_requires_exact_six_arm_order_and_bound_prefix(tmp_path: Path):
    identity = _identity("treatment")
    _begin(tmp_path, identity)
    binding = _artifact(tmp_path)
    _append(tmp_path, identity, "FOUNDATION_LOAD", artifacts=[binding])
    with pytest.raises(store.V13StoreError, match="transition"):
        _append(tmp_path, identity, "ARM_CC", arms=PRIMARY_ARMS[:2])
    with pytest.raises(store.V13StoreError, match="completed-arm"):
        _append(tmp_path, identity, "ARM_FF", arms=())
    for index, arm in enumerate(PRIMARY_ARMS, start=1):
        _append(tmp_path, identity, f"ARM_{arm}", arms=PRIMARY_ARMS[:index])
    _append(tmp_path, identity, "TERMINAL_TREATMENT", arms=PRIMARY_ARMS)


def test_bound_artifact_must_exist_and_match_literal_bytes(tmp_path: Path):
    identity = _identity("phase_a")
    _begin(tmp_path, identity)
    _append(tmp_path, identity, "RENDER_ATTEMPT")
    _append(tmp_path, identity, "PLANS_PHASE_A")
    binding = _artifact(tmp_path)
    changed = dict(binding, sha256="f" * 64)
    with pytest.raises(store.V13StoreError, match="bytes differ"):
        _append(tmp_path, identity, "FOUNDATION_BUNDLE", artifacts=[changed])
    missing = dict(binding, path="artifacts/missing.bin")
    with pytest.raises(store.V13StoreError, match="absent"):
        _append(tmp_path, identity, "FOUNDATION_BUNDLE", artifacts=[missing])


def test_bound_artifact_rejects_symlink_and_escape(tmp_path: Path):
    identity = _identity("phase_a")
    _begin(tmp_path, identity)
    _append(tmp_path, identity, "RENDER_ATTEMPT")
    _append(tmp_path, identity, "PLANS_PHASE_A")
    outside = tmp_path.parent / "outside-store-artifact.bin"
    outside.write_bytes(b"outside")
    link = tmp_path / "artifacts" / "linked.bin"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    binding = {
        "kind": "tensor_bundle",
        "path": "artifacts/linked.bin",
        "sha256": store.file_sha256(outside),
        "size_bytes": outside.stat().st_size,
    }
    with pytest.raises(store.V13StoreError, match="escapes|symlink"):
        _append(tmp_path, identity, "FOUNDATION_BUNDLE", artifacts=[binding])


def test_complete_phase_a_has_hash_only_index_and_exact_terminal_reuse(
        tmp_path: Path):
    identity, index = _complete_phase_a(tmp_path)
    assert index["status"] == "TERMINAL_VALIDATED"
    assert not (tmp_path / store.ACTIVE_LOCK_NAME).exists()
    assert set(index) == {
        "schema", "design_id", "identity", "identity_sha256",
        "terminal_directory", "terminal_record", "record_chain_sha256",
        "status",
    }
    assert "payload" not in index and "artifacts" not in index
    reuse = store.terminal_reuse_binding(
        tmp_path, expected_identity=identity)
    assert reuse["reusable"] is True
    assert reuse["identity_sha256"] == store.identity_sha256(identity)


def test_complete_treatment_is_terminal_only_after_all_six_arms(tmp_path: Path):
    identity, index = _complete_treatment(tmp_path)
    assert index["terminal_record"]["name"].endswith(
        "_TERMINAL_TREATMENT.json")
    assert store.terminal_reuse_binding(
        tmp_path, expected_identity=identity)["reusable"] is True


def test_partial_case_never_has_reuse_binding(tmp_path: Path):
    identity = _identity("phase_a")
    _begin(tmp_path, identity)
    _append(tmp_path, identity, "RENDER_ATTEMPT")
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
        assert changed != identity
        with pytest.raises(store.V13StoreError, match="cannot read session index"):
            store.terminal_reuse_binding(
                tmp_path, expected_identity=changed)


def test_record_or_bound_artifact_mutation_invalidates_terminal_reuse(
        tmp_path: Path):
    identity, index = _complete_phase_a(tmp_path)
    terminal = tmp_path / index["terminal_directory"]
    record = sorted((terminal / "records").iterdir())[1]
    record.write_bytes(record.read_bytes().replace(
        b"RENDER_ATTEMPT", b"RENDER_ATXEMPT"))
    with pytest.raises(store.V13StoreError):
        store.terminal_reuse_binding(tmp_path, expected_identity=identity)


def test_alive_process_cannot_be_quarantined_and_probe_gets_start_token(
        tmp_path: Path):
    identity = _identity()
    _begin(tmp_path, identity)
    calls = []

    def alive(pid, token):
        calls.append((pid, token))
        return True

    with pytest.raises(store.V13StoreError, match="still alive"):
        store.quarantine_abandoned_case(
            tmp_path, process_probe=alive,
            observed_utc="2026-07-12T18:41:00Z", reason="synthetic kill")
    assert calls == [(PID, START)]
    assert (tmp_path / store.ACTIVE_LOCK_NAME).exists()


def test_dead_partial_is_preserved_quarantined_and_never_reusable(tmp_path: Path):
    identity = _identity()
    lock = _begin(tmp_path, identity)
    _append(tmp_path, identity, "RENDER_ATTEMPT")
    record = store.quarantine_abandoned_case(
        tmp_path, process_probe=lambda pid, token: False,
        observed_utc="2026-07-12T18:42:00Z", reason="hard kill")
    assert record["partial_reusable"] is False
    assert not (tmp_path / store.ACTIVE_LOCK_NAME).exists()
    quarantine = list((tmp_path / "quarantine").iterdir())
    assert len(quarantine) == 1
    assert (quarantine[0] / "partial_case" / "records" /
            "001_RENDER_ATTEMPT.json").exists()
    assert not (tmp_path / lock["case_dir"]).exists()
    with pytest.raises(store.V13StoreError):
        store.terminal_reuse_binding(tmp_path, expected_identity=identity)


def test_crash_after_terminal_directory_promotion_finishes_index_not_quarantine(
        tmp_path: Path):
    identity = _identity("phase_a")
    lock = _begin(tmp_path, identity)
    _append(tmp_path, identity, "RENDER_ATTEMPT")
    _append(tmp_path, identity, "PLANS_PHASE_A")
    binding = _artifact(tmp_path)
    _append(tmp_path, identity, "FOUNDATION_BUNDLE", artifacts=[binding])
    _append(tmp_path, identity, "TERMINAL_PHASE_A")
    digest = store.identity_sha256(identity)
    (tmp_path / lock["case_dir"]).rename(tmp_path / "terminal" / digest)
    result = store.quarantine_abandoned_case(
        tmp_path, process_probe=lambda pid, token: False,
        observed_utc="2026-07-12T18:43:00Z", reason="kill after rename")
    assert result["status"] == "TERMINAL_VALIDATED"
    assert not list((tmp_path / "quarantine").iterdir())
    assert store.terminal_reuse_binding(
        tmp_path, expected_identity=identity)["reusable"] is True


def test_probe_must_return_literal_boolean(tmp_path: Path):
    _begin(tmp_path, _identity())
    with pytest.raises(store.V13StoreError, match="did not return boolean"):
        store.quarantine_abandoned_case(
            tmp_path, process_probe=lambda pid, token: 0,
            observed_utc="2026-07-12T18:44:00Z", reason="bad probe")


def test_exclusive_json_never_overwrites(tmp_path: Path):
    path = tmp_path / "immutable.json"
    store.write_json_exclusive(path, {"value": 1})
    original = path.read_bytes()
    with pytest.raises(store.V13StoreError, match="already exists"):
        store.write_json_exclusive(path, {"value": 2})
    assert path.read_bytes() == original
