from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch
from transformers import DynamicCache

import powered_v13_stage_t as stage_t
import powered_v13_store as store
from powered_v13_schema import (
    CarrierRegions, PRIMARY_ARMS, ReplayEvent, ReplayPlan,
)


SHA_1 = "1" * 64
SHA_2 = "2" * 64
SHA_3 = "3" * 64
WHOLE_UTC = "2026-07-12T20:00:00Z"
PRECISE_UTC = "2026-07-12T20:00:00.000001Z"


@contextmanager
def _clean_stage_t_import_boundary():
    removed = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if any(name == stem or name.startswith(stem + ".")
               for stem in stage_t.FORBIDDEN_PRODUCTION_MODULES)
    }
    try:
        yield
    finally:
        sys.modules.update(removed)


class FakeHandle:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.model = object()
        self.tokenizer = object()
        self.runtime_fingerprint = {
            "schema": "fake-exact-subject-runtime-v1",
            "model_id": stage_t.MODEL_ID,
            "release_binding": {"release_evidence_sha256": SHA_1},
            "fingerprint_sha256": SHA_2,
            "cuda": {"gpu_uuid": "GPU-fake-stage-t"},
            "eos_ids": [1, 2],
        }
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.events.append("close")


class FakeClock:
    def __init__(self) -> None:
        self.value = 10.0

    def monotonic(self) -> float:
        self.value += 0.01
        return self.value


class LongReplayFakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    @property
    def device(self):
        return self.anchor.device

    def forward(self, input_ids, past_key_values=None, position_ids=None,
                cache_position=None, use_cache=True, logits_to_keep=1):
        del use_cache, logits_to_keep
        previous = (0 if past_key_values is None else
                    int(past_key_values.layers[0].keys.shape[-2]))
        assert cache_position.tolist() == list(range(
            previous, previous + input_ids.shape[1]))
        cache = DynamicCache() if past_key_values is None else past_key_values
        token = input_ids.to(torch.float32)
        position = position_ids.to(torch.float32)
        for layer in range(2):
            keys = torch.stack(
                (token + layer, position + layer), dim=-1).unsqueeze(1)
            values = torch.stack(
                (token - layer, position - layer), dim=-1).unsqueeze(1)
            cache.update(keys, values, layer)
        logits = torch.zeros((1, 1, 64), dtype=torch.float32)
        pivot = int((input_ids[0, -1] + position_ids[0, -1]).item())
        logits[0, 0, pivot % 64] = 3.0
        return SimpleNamespace(past_key_values=cache, logits=logits)


def _long_replay_plan() -> ReplayPlan:
    points = (0, 4000, 4100, 4400, 4401, 4450, 4500)
    events = [
        ReplayEvent(
            "q1" if end - start == 1 else "prefill",
            f"long-{index}", "technical", index,
            start, end)
        for index, (start, end) in enumerate(zip(points, points[1:]))
    ]
    return ReplayPlan(
        token_ids=[index % 63 for index in range(4500)],
        message_start_positions=list(points[:-1]),
        events=events,
        regions=CarrierRegions(
            content_start=4000,
            content_end=4100,
            anchor_prefix_end=4400,
            anchor_content_end=4401,
        ),
    ).validate()


def _fake_case(case_id: str) -> dict:
    tokens = 4500 if case_id == "technical_long" else 100
    plans = {
        history: {
            "token_count": tokens,
            "r2_width": 4,
            "geometry_sha256": SHA_3,
        }
        for history in ("F", "C", "W")
    }
    return {
        "case_id": case_id,
        "stable_technical_id": f"stable-{case_id}",
        "semantic_n": 0,
        "inferential_use": "FORBIDDEN",
        "production_pool_imported": False,
        "production_entropy_requested": False,
        "production_probe_reachable": False,
        "score_values_present": False,
        "carrier": f"fixed carrier for {case_id}",
        "technical_input_sha256": SHA_1,
        "technical_probe_sha256": SHA_2,
        "technical_probe": {
            "probe": "technical prompt",
            "correct_target": "continuity",
            "counterfactual_target": "criteria",
        },
        "probe_runtime": {
            "suffix_ids": [41, 42],
            "suffix_ids_sha256": SHA_1,
            "correct_target_ids": [43],
            "correct_target_ids_sha256": SHA_2,
            "counterfactual_target_ids": [44],
            "counterfactual_target_ids_sha256": SHA_3,
        },
        "plans": plans,
        "plan_objects": {
            "F": SimpleNamespace(case_id=case_id, history="F"),
            "C": SimpleNamespace(case_id=case_id, history="C"),
        },
        "context_messages": {"F": []},
    }


def _binding(path: Path, root: Path, kind: str) -> dict:
    return stage_t._artifact_binding(path, root=root, kind=kind)


def _write_fake_foundation_files(config, case_id: str):
    directory = config.run_directory / "artifacts" / case_id
    directory.mkdir(parents=True, exist_ok=True)
    bundle = directory / "fake_foundation_bundle.bin"
    bundle.write_bytes(f"bundle:{case_id}".encode())
    descriptor = directory / "fake_foundation_descriptor.json"
    receipt = directory / "fake_foundation_receipt.json"
    stage_t._write_json_exclusive(descriptor, {
        "case_id": case_id,
        "kind": "fake-foundation-descriptor",
    })
    stage_t._write_json_exclusive(receipt, {
        "case_id": case_id,
        "kind": "fake-foundation-receipt",
    })
    return (
        _binding(bundle, config.run_directory, "foundation_bundle"),
        _binding(descriptor, config.run_directory, "foundation_descriptor"),
        _binding(receipt, config.run_directory, "foundation_receipt"),
    )


def _fake_seams(
    events: list[str], *, unavailable_cases: set[str] | None = None,
    ladder_error: BaseException | None = None,
):
    unavailable_cases = unavailable_cases or set()
    clock = FakeClock()
    handles: list[FakeHandle] = []

    def open_subject(repo, receipt_directory):
        del repo, receipt_directory
        events.append("open")
        handle = FakeHandle(events)
        handles.append(handle)
        return handle

    def prepare_case(tokenizer, repo, case_id):
        del tokenizer, repo
        events.append(f"prepare:{case_id}")
        return _fake_case(case_id)

    def ladder(model, plan):
        del model, plan
        events.append("ladder:e01")
        if ladder_error is not None:
            raise ladder_error
        return {"status": "PASS", "gate": "powered_v13_ladder"}

    def long_identity(model, plan):
        del model
        assert plan.history == "C"
        events.append("identity:long")
        return {"status": "PASS", "gate": "long-l0-l3"}

    def capture(config, handle, case, input_binding, gate_binding, recorder):
        del handle, input_binding, gate_binding, recorder
        case_id = case["case_id"]
        events.append(f"foundation:{case_id}")
        bundle, descriptor, receipt = _write_fake_foundation_files(
            config, case_id)
        status = (
            "PLACEBO_UNAVAILABLE" if case_id in unavailable_cases
            else "AVAILABLE")
        vp_receipt = {"status": status, "case_id": case_id}
        live_tokens, selected_rows = stage_t._declared_case_bounds(case)
        return stage_t.FoundationState(
            descriptor={"case_id": case_id},
            materialized=object(),
            vp_receipt=vp_receipt,
            vp_receipt_sha256=stage_t._sha256_json(vp_receipt),
            vp_diagnostics={"deterministic_search": True},
            vp_status=status,
            bundle_binding=bundle,
            descriptor_binding=descriptor,
            receipt_binding=receipt,
            selected_row_count=selected_rows,
            live_token_count=live_tokens,
            r2_end=80,
        )

    def run_arm(handle, case, foundation, arm, recorder):
        del handle, recorder
        case_id = case["case_id"]
        events.append(f"arm:{case_id}:{arm}")
        unavailable = arm == "VP" and foundation.vp_status == \
            "PLACEBO_UNAVAILABLE"
        probe = None if unavailable else {
            "suffix_ids": list(case["probe_runtime"]["suffix_ids"]),
            "correct_target_ids": [43],
            "counterfactual_target_ids": [44],
            "calls": [{"width": 1}],
            "logical_positions": [81],
            "physical_positions": [81],
            "float32_score_bits": ["00000000", "0000803f"],
            "semantic_n": 0,
            "inferential_use": "FORBIDDEN",
        }
        return {
            "schema": stage_t.ARM_ARTIFACT_SCHEMA,
            "arm_id": arm,
            "case_id": case_id,
            "execution_status": (
                "PLACEBO_UNAVAILABLE" if unavailable else "COMPLETE"),
            "stage_a_progression_gate_applies":
                case_id == "technical_e01" and arm == "VP",
            "stage_a_progression_allowed": (
                not unavailable
                if case_id == "technical_e01" and arm == "VP" else None),
            "boundary_reconstruction": None if unavailable else {
                "selected_row_count": 4,
                "inserted_row_sha256": SHA_1,
                "outside_row_sha256": SHA_2,
            },
            "continuation": None if unavailable else {
                "executed_token_ids": [51],
                "logical_positions": [80],
                "physical_positions": [80],
                "calls": [{"width": 1}],
            },
            "probe": probe,
            "semantic_n": 0,
            "inferential_use": "FORBIDDEN",
        }

    def build_identity(**kwargs):
        return store.build_identity(**kwargs)

    def begin_case(root, *, identity, **kwargs):
        del root, kwargs
        events.append(f"begin:{identity['case_id']}")
        return {"record_kind": "STARTED"}

    def append_record(root, *, identity, record_kind, artifact_bindings,
                      **kwargs):
        del root, kwargs
        for artifact in artifact_bindings:
            assert artifact["sha256"] and artifact["size_bytes"] > 0
        events.append(f"append:{identity['case_id']}:{record_kind}")
        return {"record_kind": record_kind}

    def active_chain(root, *, identity, **kwargs):
        del root, kwargs
        events.append(f"chain:{identity['case_id']}")
        return SHA_3

    def validator(*, identity_path, terminal_evidence_path, output_path,
                  **kwargs):
        del kwargs
        identity = json.loads(identity_path.read_bytes())
        evidence = json.loads(terminal_evidence_path.read_bytes())
        events.append(f"validator:{identity['case_id']}")
        receipt = {
            "schema": store.TERMINAL_VALIDATION_SCHEMA,
            "design_id": stage_t.DESIGN_ID,
            "identity_sha256": store.identity_sha256(identity),
            "terminal_kind": "TERMINAL_TECHNICAL",
            "evidence_chain_sha256": evidence["evidence_chain_sha256"],
            "terminal_evidence_sha256": store.file_sha256(
                terminal_evidence_path),
            "validator_id": store.TERMINAL_VALIDATOR_ID,
            "validation_status": "PASS",
            "validated_utc": WHOLE_UTC,
        }
        stage_t._write_json_exclusive(output_path, receipt)
        return receipt

    def finalize(root, *, identity, **kwargs):
        del root, kwargs
        events.append(f"finalize:{identity['case_id']}")
        return {"status": "TERMINAL_VALIDATED"}

    def cuda_sync():
        events.append("cuda:synchronize")

    def reset_vram():
        events.append("cuda:reset_peak")

    def vram():
        events.append("cuda:vram")
        return {
            "device": "cuda:0",
            "memory_allocated_bytes": 100,
            "max_memory_allocated_bytes": 120,
            "memory_reserved_bytes": 140,
            "max_memory_reserved_bytes": 160,
        }

    def evict():
        events.append("evict")

    seams = stage_t.StageTSeams(
        open_subject=open_subject,
        prepare_case=prepare_case,
        run_ladder=ladder,
        run_long_identity=long_identity,
        capture_foundation=capture,
        run_arm=run_arm,
        process_start_token=lambda pid: f"fake-process-token:{pid}",
        build_store_identity=build_identity,
        build_store_bounds=store.build_bounds,
        begin_store_case=begin_case,
        append_store_record=append_record,
        active_chain_sha256=active_chain,
        invoke_terminal_validator=validator,
        finalize_store_case=finalize,
        utc_now=lambda: WHOLE_UTC,
        precise_utc_now=lambda: PRECISE_UTC,
        monotonic=clock.monotonic,
        cuda_synchronize=cuda_sync,
        reset_peak_vram=reset_vram,
        vram_metrics=vram,
        evict_cuda=evict,
    )
    return seams, handles


def _config(tmp_path: Path) -> stage_t.StageTConfig:
    run = tmp_path / "run"
    receipts = tmp_path / "receipts"
    run.mkdir()
    receipts.mkdir()
    return stage_t.StageTConfig(
        repo=Path(__file__).resolve().parents[1],
        receipt_directory=receipts,
        run_directory=run,
        primary_batch_id="technical-batch-001",
    )


def _scientific_events(events: list[str]) -> list[str]:
    prefixes = (
        "open", "prepare:", "begin:", "ladder:", "identity:",
        "foundation:", "arm:", "append:", "chain:", "validator:",
        "finalize:", "close",
    )
    return [event for event in events if event.startswith(prefixes)]


def _expected_case_events(case_id: str, gate: str) -> list[str]:
    rows = [
        f"prepare:{case_id}",
        f"begin:{case_id}",
        gate,
        f"foundation:{case_id}",
        f"append:{case_id}:FOUNDATION_LOAD",
    ]
    for arm in PRIMARY_ARMS:
        rows.extend([
            f"arm:{case_id}:{arm}",
            f"append:{case_id}:ARM_{arm}",
        ])
    rows.extend([
        f"chain:{case_id}",
        f"validator:{case_id}",
        f"append:{case_id}:TERMINAL_TECHNICAL",
        f"finalize:{case_id}",
    ])
    return rows


def test_exact_order_six_arms_checkpoints_and_noninferential_probes(tmp_path):
    events: list[str] = []
    seams, handles = _fake_seams(events)
    config = _config(tmp_path)
    with _clean_stage_t_import_boundary():
        summary = stage_t.run_stage_t(config, seams=seams)

    expected = ["open"]
    expected.extend(_expected_case_events(
        "technical_e01", "ladder:e01"))
    expected.extend(_expected_case_events(
        "technical_long", "identity:long"))
    expected.append("close")
    assert _scientific_events(events) == expected
    assert events.index("ladder:e01") < next(
        index for index, event in enumerate(events)
        if event in {"cuda:synchronize", "cuda:reset_peak"})
    assert summary["case_order"] == ["technical_e01", "technical_long"]
    assert summary["semantic_n"] == 0
    assert summary["inferential_use"] == "FORBIDDEN"
    assert summary["stage_a_progression_allowed"] is True
    assert summary["stage_a_progression_reason"] == \
        "TECHNICAL_E01_VP_AVAILABLE"
    assert handles[0].closed is True
    assert events[-1] == "evict"

    for case_id in stage_t.TECHNICAL_CASE_ORDER:
        directory = config.run_directory / "artifacts" / case_id
        for arm in PRIMARY_ARMS:
            artifact = json.loads(
                (directory / f"arm_{arm}_artifact.json").read_bytes())
            checkpoint = json.loads(
                (directory / f"arm_{arm}_checkpoint.json").read_bytes())
            assert artifact["arm_id"] == arm
            assert artifact["semantic_n"] == 0
            assert artifact["inferential_use"] == "FORBIDDEN"
            assert artifact["probe"]["inferential_use"] == "FORBIDDEN"
            assert artifact["probe"]["float32_score_bits"]
            assert artifact["timing_vram"][-1]["vram_after"][
                "max_memory_allocated_bytes"] == 120
            expected_prefix = list(
                PRIMARY_ARMS[:PRIMARY_ARMS.index(arm) + 1])
            assert checkpoint["completed_arm_ids"] == expected_prefix
        assert (directory / "independent_terminal_validation.json").is_file()
    assert (config.run_directory / "RUN_COMPLETE.json").is_file()


def test_long_identity_uses_45k_replay_and_exact_r2_resume_calls():
    evidence = stage_t._run_long_identity(
        LongReplayFakeModel(), _long_replay_plan())
    assert evidence["status"] == "PASS"
    assert evidence["token_count"] == 4500
    assert evidence["r2_physical_end"] == 4400
    assert [row["physical_end"] - row["physical_start"]
            for row in evidence["l0_first"]["calls"]] == [
                4000, 100, 300, 1, 49, 50,
            ]
    assert evidence["l0_first"]["calls"] == \
        evidence["l3_boundary"]["calls"] + \
        evidence["l3_continuation"]["calls"]
    assert evidence["l0_first"]["snapshot_hashes"] == \
        evidence["l3_continuation"]["snapshot_hashes"]
    assert evidence["l0_first"]["q1_token_logprobs"] == \
        evidence["l3_boundary"]["q1_token_logprobs"] + \
        evidence["l3_continuation"]["q1_token_logprobs"]


def test_ladder_failure_aborts_before_measured_or_long_work_and_closes(tmp_path):
    events: list[str] = []
    seams, handles = _fake_seams(
        events, ladder_error=RuntimeError("injected ladder failure"))
    config = _config(tmp_path)
    with _clean_stage_t_import_boundary():
        result = stage_t._worker_entry(config, seams=seams)

    assert result == 2
    assert "ladder:e01" in events
    assert "cuda:synchronize" not in events
    assert "cuda:reset_peak" not in events
    assert not any(event.startswith("foundation:") for event in events)
    assert not any(event.startswith("arm:") for event in events)
    assert "prepare:technical_long" not in events
    assert handles[0].closed is True
    assert events[-1] == "evict"
    error = json.loads((config.run_directory / "RUN_ERROR.json").read_bytes())
    assert error["error_type"] == "RuntimeError"
    assert "injected ladder failure" in error["error_message"]
    partial = json.loads((
        config.run_directory / "PARTIAL_RUN_EVIDENCE.json").read_bytes())
    assert partial["status"] == "PARTIAL_ERROR"
    assert partial["completed_case_ids"] == []
    assert partial["timing_vram"] == []
    assert not (config.run_directory / "RUN_COMPLETE.json").exists()


def test_subject_open_failure_preserves_original_error_and_evicts(tmp_path):
    events: list[str] = []
    seams, handles = _fake_seams(events)

    def fail_open(repo, receipt_directory):
        del repo, receipt_directory
        events.append("open:failure")
        raise RuntimeError("injected subject-open failure")

    seams = replace(seams, open_subject=fail_open)
    config = _config(tmp_path)
    with _clean_stage_t_import_boundary():
        result = stage_t._worker_entry(config, seams=seams)

    assert result == 2
    assert handles == []
    assert events == ["open:failure", "evict"]
    error = json.loads((config.run_directory / "RUN_ERROR.json").read_bytes())
    assert error["error_type"] == "RuntimeError"
    assert error["error_message"] == "injected subject-open failure"
    partial = json.loads((
        config.run_directory / "PARTIAL_RUN_EVIDENCE.json").read_bytes())
    assert partial["error_type"] == "RuntimeError"
    assert partial["completed_case_ids"] == []


def test_vp_unavailable_is_terminal_but_blocks_stage_a_progression(tmp_path):
    events: list[str] = []
    seams, _handles = _fake_seams(
        events, unavailable_cases={"technical_e01"})
    config = _config(tmp_path)
    with _clean_stage_t_import_boundary():
        summary = stage_t.run_stage_t(config, seams=seams)

    e01 = summary["cases"][0]
    assert e01["vp_status"] == "PLACEBO_UNAVAILABLE"
    assert e01["stage_a_progression_allowed"] is False
    assert summary["stage_a_progression_allowed"] is False
    assert summary["stage_a_progression_reason"] == \
        "TECHNICAL_E01_VP_UNAVAILABLE"
    assert [row["arm_id"] for row in e01["arms"]] == list(PRIMARY_ARMS)
    vp = json.loads((
        config.run_directory / "artifacts" / "technical_e01" /
        "arm_VP_artifact.json").read_bytes())
    assert vp["execution_status"] == "PLACEBO_UNAVAILABLE"
    assert vp["probe"] is None
    assert vp["timing_vram"]
    assert "append:technical_e01:ARM_VP" in events
    assert "finalize:technical_e01" in events
    assert "prepare:technical_long" in events


def test_long_vp_unavailable_is_diagnostic_and_does_not_block_stage_a(
        tmp_path):
    events: list[str] = []
    seams, _handles = _fake_seams(
        events, unavailable_cases={"technical_long"})
    config = _config(tmp_path)
    with _clean_stage_t_import_boundary():
        summary = stage_t.run_stage_t(config, seams=seams)

    e01, long = summary["cases"]
    assert e01["vp_status"] == "AVAILABLE"
    assert e01["stage_a_progression_gate_applies"] is True
    assert e01["stage_a_progression_allowed"] is True
    assert long["vp_status"] == "PLACEBO_UNAVAILABLE"
    assert long["stage_a_progression_gate_applies"] is False
    assert long["stage_a_progression_allowed"] is None
    assert summary["stage_a_progression_allowed"] is True
    assert summary["stage_a_progression_reason"] == \
        "TECHNICAL_E01_VP_AVAILABLE"
    assert summary["technical_long_vp_is_diagnostic_only"] is True
    assert "append:technical_long:ARM_VP" in events
    assert "finalize:technical_long" in events


def test_fake_runtime_reaches_real_store_and_independent_terminal_validator(
        tmp_path):
    events: list[str] = []
    seams, _handles = _fake_seams(events)
    process_token = (
        "linux-procfs-v1:11111111-2222-3333-4444-555555555555:123456")

    def token_reader(_pid):
        return process_token

    def begin(root, **kwargs):
        return store._begin_case(
            root, **kwargs,
            _test_process_start_token_reader=token_reader)

    def append(root, **kwargs):
        return store._append_case_record(
            root, **kwargs,
            _test_process_start_token_reader=token_reader)

    def chain(root, **kwargs):
        return store._active_evidence_chain_sha256(
            root, **kwargs,
            _test_process_start_token_reader=token_reader)

    def finalize(root, **kwargs):
        return store._finalize_case(
            root, **kwargs,
            _test_process_start_token_reader=token_reader)

    seams = replace(
        seams,
        process_start_token=lambda _pid: process_token,
        begin_store_case=begin,
        append_store_record=append,
        active_chain_sha256=chain,
        invoke_terminal_validator=stage_t._invoke_terminal_validator,
        finalize_store_case=finalize,
    )
    config = _config(tmp_path)
    with _clean_stage_t_import_boundary():
        summary = stage_t.run_stage_t(config, seams=seams)

    assert summary["status"] == "PASS"
    assert not (config.run_directory / stage_t.ACTIVE_LOCK_NAME).exists()
    assert len(list((config.run_directory / "index").glob("*.json"))) == 2
    for case in summary["cases"]:
        assert case["terminal"]["index"]["status"] == "TERMINAL_VALIDATED"
        assert case["terminal"]["validator_receipt"][
            "validation_status"] == "PASS"


def test_terminal_validator_is_spawned_as_a_distinct_process(tmp_path,
                                                             monkeypatch):
    script = tmp_path / stage_t.VALIDATOR_RELATIVE_PATH
    script.parent.mkdir(parents=True)
    script.write_text("# validator placeholder\n")
    identity = tmp_path / "identity.json"
    evidence = tmp_path / "terminal.json"
    output = tmp_path / "receipt.json"
    identity.write_text("{}\n")
    evidence.write_text("{}\n")
    observed = {}

    def run(command, *, cwd, text, capture_output, check):
        observed.update({
            "command": command,
            "cwd": cwd,
            "text": text,
            "capture_output": capture_output,
            "check": check,
        })
        destination = Path(command[command.index("--output") + 1])
        destination.write_text(json.dumps({
            "validation_status": "PASS",
        }) + "\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(stage_t.subprocess, "run", run)
    receipt = stage_t._invoke_terminal_validator(
        repo=tmp_path,
        store_root=tmp_path / "store",
        identity_path=identity,
        terminal_evidence_path=evidence,
        output_path=output,
        runner_pid=8765,
    )
    assert receipt["validation_status"] == "PASS"
    assert observed["command"][0] == sys.executable
    assert observed["command"][1] == str(script)
    assert observed["command"][observed["command"].index(
        "--runner-pid") + 1] == "8765"
    assert observed["cwd"] == tmp_path
    assert observed["capture_output"] is True
    assert observed["check"] is False


def test_unique_model_slug_utc_output_refuses_reuse_or_overwrite(tmp_path):
    timestamp = "20260712T200000123456Z"
    first = stage_t._create_unique_output_directory(
        tmp_path, compact_utc=lambda: timestamp)
    assert first.name == \
        f"powered-v13-stage-t_{stage_t.MODEL_SLUG}_{timestamp}"
    marker = first / "preserved.txt"
    marker.write_text("preserve me")
    with pytest.raises(stage_t.V13StageTRunnerError, match="refusing to reuse"):
        stage_t._create_unique_output_directory(
            tmp_path, compact_utc=lambda: timestamp)
    assert marker.read_text() == "preserve me"
    document = first / "exclusive.json"
    stage_t._write_json_exclusive(document, {"first": True})
    with pytest.raises(stage_t.V13StageTRunnerError, match="overwrite"):
        stage_t._write_json_exclusive(document, {"first": False})


def test_failed_dead_worker_is_quarantined_before_supervisor_receipt(tmp_path):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / stage_t.ACTIVE_LOCK_NAME).write_text("active\n")
    events: list[str] = []

    def run_process(command, **kwargs):
        del command, kwargs
        events.append("worker-returned")
        return SimpleNamespace(returncode=7)

    def quarantine(root, *, observed_utc, reason):
        assert events == ["worker-returned"]
        assert (root / stage_t.ACTIVE_LOCK_NAME).exists()
        events.append("quarantine")
        return {"observed_utc": observed_utc, "reason": reason}

    result = stage_t._supervise_created_run(
        repo=tmp_path,
        receipt_directory=tmp_path / "receipts",
        output_parent=tmp_path,
        run_directory=run_directory,
        script=tmp_path / "runner.py",
        primary_batch_id="technical-batch-001",
        run_process=run_process,
        quarantine=quarantine,
        utc_now=lambda: WHOLE_UTC,
    )
    assert result == 7
    assert events == ["worker-returned", "quarantine"]
    receipt = json.loads((run_directory / "SUPERVISOR.json").read_bytes())
    assert receipt["status"] == "WORKER_FAILED"
    assert receipt["quarantine_record"]["reason"] == "stage-t-worker-exit-7"
    assert receipt["provider_clock_owned_by_runner"] is False
    assert receipt["watchdog_duplicated"] is False


def test_loaded_production_module_is_a_dynamic_stop_gate(monkeypatch):
    forbidden = stage_t.FORBIDDEN_PRODUCTION_MODULES[0]
    monkeypatch.setitem(sys.modules, forbidden, ModuleType(forbidden))
    with pytest.raises(stage_t.V13StageTRunnerError,
                       match="forbidden Stage-T production modules"):
        stage_t._assert_import_boundary()


def test_runner_import_graph_has_no_production_input_or_analysis_module():
    source = Path(stage_t.__file__).read_text()
    for forbidden in stage_t.FORBIDDEN_PRODUCTION_MODULES:
        assert f"from {forbidden} import" not in source
        assert f"import {forbidden}" not in source
    assert "open_exact_subject" in source
    assert "run_stage_t_ladder" in source
    assert "reconstruct_arm_boundary" in source
    assert "quarantine_abandoned_case" in source
    assert "validate_powered_v13_technical_terminal.py" in source
