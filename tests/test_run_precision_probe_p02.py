from __future__ import annotations

import argparse
from dataclasses import replace
import fnmatch
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_precision_probe_p02", ROOT / "scripts/run_precision_probe_p02.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _verified(label: str) -> dict:
    return {"verification_status": "VERIFIED", "path": label}


def _runtime_gate_pass() -> dict:
    return {
        "status": "PASS",
        "compared_runtime_fields": ["model_id"],
        "compared_subject_binding_fields": ["checkpoint"],
        "differing_fields": [],
        "nf4_matched_view_sha256": "a" * 64,
        "bf16_matched_view_sha256": "a" * 64,
    }


def test_scientific_cap_is_exact_frozen_formula_and_optional_cap_only_tightens():
    assert MODULE.scientific_cap_seconds(1.39) == 9000
    assert MODULE.scientific_cap_seconds(2.0) == 7020
    assert MODULE.scientific_cap_seconds(1.39, 8000.9) == 8000
    with pytest.raises(MODULE.PrecisionProbeP02RunnerError):
        MODULE.scientific_cap_seconds(0)


def test_frozen_case_and_technical_fixture_hashes_match_committed_bytes():
    assert MODULE.p01.sha256_file(MODULE.DEFAULT_CASE)[0] == MODULE.CASE_E01_SHA256
    assert (MODULE.p01.sha256_file(MODULE.DEFAULT_IDENTITY)[0]
            == MODULE.IDENTITY_FIXTURE_SHA256)
    assert (MODULE.p01.sha256_file(MODULE.DEFAULT_TECHNICAL)[0]
            == MODULE.TECHNICAL_FIXTURE_SHA256)


def test_runner_rejects_e01_shaped_but_nonfrozen_case_before_creating_run(tmp_path):
    wrong_case = tmp_path / "e01.json"
    wrong_case.write_text('{"case_id":"e01","changed":true}\n')
    args = argparse.Namespace(
        repo=ROOT,
        output_dir=tmp_path / "results",
        model=MODULE.MODEL_ID,
        revision=MODULE.REVISION,
        prereg=MODULE.DEFAULT_PREREG,
        identity_fixture=MODULE.DEFAULT_IDENTITY,
        technical_fixture=MODULE.DEFAULT_TECHNICAL,
        case_e01=wrong_case,
        allow_download=False,
        hourly_cost_usd=1.39,
        provider_elapsed_seconds_at_start=0.0,
        provider_wall_cap_seconds=9000.0,
    )
    with pytest.raises(MODULE.PrecisionProbeP02RunnerError,
                       match="e01 case bytes"):
        MODULE.run(args)
    assert not args.output_dir.exists()


def test_continuation_interfaces_are_scalar_only_and_slotted():
    assert set(MODULE.TechnicalDecisionTiming.__slots__) == {
        "status", "durable_elapsed_seconds"
    }
    assert set(MODULE.OutcomeDecisionTiming.__slots__) == {
        "completion_status", "durable_elapsed_seconds"
    }
    assert not hasattr(MODULE.OutcomeDecisionTiming("COMPLETE", 1.0), "__dict__")
    assert list(inspect.signature(MODULE.decide_initial_schedule).parameters) == [
        "nf4_technical", "nf4_repeat1", "provider_elapsed_seconds",
        "scientific_cap",
    ]
    for function in (
        MODULE.decide_initial_schedule,
        MODULE.decide_bf16_technical_admission,
        MODULE.decide_bf16_repeat1,
        MODULE.decide_bf16_repeat2,
    ):
        source = inspect.getsource(function)
        assert "open(" not in source
        assert "load_object" not in source
        assert "_load_packaged_document" not in source
        assert "generation" not in source


def test_initial_schedule_uses_exact_matched_and_symmetric_rider_formulas():
    technical = MODULE.TechnicalDecisionTiming("PASS", 800.0)
    outcome = MODULE.OutcomeDecisionTiming("COMPLETE", 1000.0)
    both = MODULE.decide_initial_schedule(
        nf4_technical=technical,
        nf4_repeat1=outcome,
        provider_elapsed_seconds=1000.0,
        scientific_cap=9000,
    )
    assert both["matched_repeat1_authorized"] is True
    assert both["both_repeat2_authorized"] is True
    assert both["inputs"]["technical_forecast_seconds"] == 900.0
    assert both["inputs"]["matched_repeat1_forecast_seconds"] == 3375.0
    assert both["inputs"]["both_repeat2_forecast_seconds"] == 5875.0

    r1_only = MODULE.decide_initial_schedule(
        nf4_technical=technical,
        nf4_repeat1=outcome,
        provider_elapsed_seconds=5000.0,
        scientific_cap=9000,
    )
    assert r1_only["matched_repeat1_authorized"] is True
    assert r1_only["both_repeat2_authorized"] is False

    stop = MODULE.decide_initial_schedule(
        nf4_technical=technical,
        nf4_repeat1=outcome,
        provider_elapsed_seconds=7000.0,
        scientific_cap=9000,
    )
    assert stop["matched_repeat1_authorized"] is False
    assert stop["both_repeat2_authorized"] is False


def test_bf16_repeat_decisions_use_preregistered_t_and_not_r2_duration():
    r1 = MODULE.decide_bf16_repeat1(
        initial_matched_authorized=True,
        nf4_repeat1=MODULE.OutcomeDecisionTiming("COMPLETE", 100.0),
        bf16_technical=MODULE.TechnicalDecisionTiming("PASS", 100.0),
        matched_runtime_status="PASS",
        provider_elapsed_seconds=100.0,
        scientific_cap=1000,
    )
    assert r1["authorized"] is True

    # The exact rider proxy is max(t_nf4_repeat1, t_bf16_repeat1).  The very
    # large NF4 repeat-2 time below must not enter that max; repeat 2 supplies
    # completion status only at this decision.
    r2 = MODULE.decide_bf16_repeat2(
        initial_repeat2_authorized=True,
        nf4_repeat1=MODULE.OutcomeDecisionTiming("COMPLETE", 100.0),
        nf4_repeat2=MODULE.OutcomeDecisionTiming("COMPLETE", 5000.0),
        bf16_repeat1=MODULE.OutcomeDecisionTiming("COMPLETE", 120.0),
        provider_elapsed_seconds=100.0,
        scientific_cap=1000,
    )
    assert r2["inputs"]["forecast_proxy_seconds"] == 120.0
    assert r2["inputs"]["forecast_seconds"] == 250.0
    assert r2["authorized"] is True


def test_planned_paths_are_unique_short_and_match_ops_outcome_glob(tmp_path):
    planned = MODULE._planned_paths(tmp_path, "20260712T070000000000Z")
    paths = [planned["run_manifest"], planned["completion"]]
    paths.extend(planned["decisions"].values())
    for record in planned["technical"].values():
        paths.extend(record.values())
    for outcome in planned["outcomes"].values():
        values = MODULE.asdict(outcome)
        for value in values.values():
            if isinstance(value, tuple):
                paths.extend(value)
            else:
                paths.append(value)
        assert fnmatch.fnmatch(
            outcome.raw_package.name,
            "precision-probe-p02-outcome-*-raw_"
            "Qwen3-30B-A3B-Instruct-2507_*.lossless-package",
        )
        assert not fnmatch.fnmatch(
            outcome.phase_package.name,
            "precision-probe-p02-outcome-*-raw_*.lossless-package",
        )
    assert len(paths) == len(set(paths))
    assert all(len(path.name.encode("utf-8")) < 255 for path in paths)


def _outcome_fixture(tmp_path, repeat=1):
    case_path = tmp_path / "e01.json"
    case_path.write_text('{"case_id":"e01"}\n')
    paths = MODULE._planned_paths(
        tmp_path / "run", "20260712T070000000000Z"
    )["outcomes"][("nf4", repeat)]
    paths.raw_scratch.parent.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        case_e01=case_path,
        repo=tmp_path,
        model=MODULE.MODEL_ID,
        revision=MODULE.REVISION,
        hourly_cost_usd=1.0,
    )
    prepared = {
        "model": object(),
        "tokenizer": object(),
        "runtime_fingerprint": {"eos_ids": [1, 2]},
        "bindings": {"checkpoint": "bound"},
    }
    return paths, args, prepared


def _phase_fixture():
    return {
        "schema": MODULE.PHASE_A_SCHEMA,
        "case_id": "e01",
        "executions": {"C_N": {}, "W_N": {}, "F": {}},
        "scores": {
            name: {} for name in (
                "A_C_focal", "A_W_focal", "FF_focal",
                "A_C_nonfocal", "A_W_nonfocal", "FF_nonfocal",
            )
        },
        "treatment_scores_present": False,
    }


def test_outcome_persists_phase_foundation_and_each_arm_before_final_raw(
    monkeypatch, tmp_path
):
    paths, args, prepared = _outcome_fixture(tmp_path, repeat=1)
    phase = _phase_fixture()
    monkeypatch.setattr(MODULE, "run_phase_a_case", lambda *a, **k: phase)
    monkeypatch.setattr(MODULE, "assert_targeted_treatment", lambda *a, **k: None)
    callback_events = []

    def treatment(*args, repeat_index, on_foundation, on_arm, **kwargs):
        on_foundation({"schema": "foundation", "repeat_index": repeat_index})
        callback_events.append("foundation_returned")
        arms = []
        for index in range(1, 8):
            arm = {
                "arm_kind": "primary" if index <= 6 else "placebo_control",
                "schedule": "N", "region": "R2_boundary",
                "cell": f"cell-{index}", "scores": {},
            }
            on_arm(index, arm)
            callback_events.append(f"arm-{index}-returned")
            arms.append(arm)
        return {
            "schema": MODULE.TREATMENT_SCHEMA,
            "case_id": "e01", "repeat_index": repeat_index,
            "fresh_scores": {}, "arms": arms,
        }

    monkeypatch.setattr(MODULE, "run_targeted_treatment_case", treatment)
    receipt = MODULE.execute_outcome(
        regime="nf4",
        repeat_index=1,
        prepared=prepared,
        args=args,
        paths=paths,
        common_bindings={},
        provider_clock=MODULE.ProviderClock(0.0, MODULE.time.monotonic()),
    )
    assert receipt.completion_status == "COMPLETE"
    assert receipt.raw_package["verification_status"] == "VERIFIED"
    assert receipt.phase_package["verification_status"] == "VERIFIED"
    assert len(receipt.checkpoint_chain) == 9  # phase + foundation + 7 arms
    assert [row["sequence"] for row in receipt.checkpoint_chain] == list(
        range(1, 10)
    )
    assert paths.phase_package.is_dir()
    assert paths.foundation_package.is_dir()
    assert all(path.is_dir() for path in paths.arm_packages)
    assert callback_events == ["foundation_returned"] + [
        f"arm-{index}-returned" for index in range(1, 8)
    ]
    timing = json.loads(paths.timing_receipt.read_bytes())
    assert timing["derived_artifacts_included"] is False
    assert timing["completion_status"] == "COMPLETE"
    assert timing["raw_package"]["verification_status"] == "VERIFIED"


def test_arm_checkpoint_failure_stops_next_arm_and_final_raw_preserves_partial(
    monkeypatch, tmp_path
):
    paths, args, prepared = _outcome_fixture(tmp_path, repeat=1)
    phase = _phase_fixture()
    monkeypatch.setattr(MODULE, "run_phase_a_case", lambda *a, **k: phase)
    monkeypatch.setattr(MODULE, "assert_targeted_treatment", lambda *a, **k: None)
    original = MODULE.p01.persist_lossless_raw

    def persist(document, *, scratch_raw, package, repo):
        if package == paths.arm_packages[1]:
            MODULE.p01.atomic_create_json(scratch_raw, document)
            return None, MODULE.p01.display_path(scratch_raw, repo), "synthetic failure"
        return original(
            document, scratch_raw=scratch_raw, package=package, repo=repo
        )

    monkeypatch.setattr(MODULE.p01, "persist_lossless_raw", persist)
    executed = []

    def treatment(*args, repeat_index, on_foundation, on_arm, **kwargs):
        on_foundation({"schema": "foundation"})
        for index in range(1, 8):
            executed.append(index)
            on_arm(index, {"schedule": "N", "region": "R2_boundary",
                           "cell": str(index), "scores": {}})
        raise AssertionError("unreachable")

    monkeypatch.setattr(MODULE, "run_targeted_treatment_case", treatment)
    receipt = MODULE.execute_outcome(
        regime="nf4", repeat_index=1, prepared=prepared, args=args,
        paths=paths, common_bindings={},
        provider_clock=MODULE.ProviderClock(0.0, MODULE.time.monotonic()),
    )
    assert executed == [1, 2]
    assert receipt.completion_status == "ERROR"
    assert receipt.raw_package["verification_status"] == "VERIFIED"
    assert len(receipt.checkpoint_chain) == 3  # phase, foundation, arm 1
    raw = MODULE.p01._load_packaged_document(
        paths.raw_package, scratch_dir=paths.raw_scratch.parent
    )
    assert raw["status"] == "ERROR"
    assert "treatment_arm_02 checkpoint" in raw["error"]["message"]
    assert paths.arm_scratch[1].is_file()


def test_fatal_interrupt_preserves_raw_but_does_not_cross_derived_boundary(
    monkeypatch, tmp_path
):
    paths, args, prepared = _outcome_fixture(tmp_path, repeat=1)
    phase = _phase_fixture()
    monkeypatch.setattr(MODULE, "run_phase_a_case", lambda *a, **k: phase)

    def interrupt(*args, on_foundation, **kwargs):
        on_foundation({"schema": "foundation"})
        raise KeyboardInterrupt("scientific cap interrupt")

    monkeypatch.setattr(MODULE, "run_targeted_treatment_case", interrupt)
    with pytest.raises(KeyboardInterrupt, match="scientific cap interrupt"):
        MODULE.execute_outcome(
            regime="nf4", repeat_index=1, prepared=prepared, args=args,
            paths=paths, common_bindings={},
            provider_clock=MODULE.ProviderClock(0.0, MODULE.time.monotonic()),
        )
    assert paths.phase_package.is_dir()
    assert paths.foundation_package.is_dir()
    assert paths.raw_package.is_dir()
    assert paths.timing_receipt.is_file()
    assert not paths.compact.exists()
    assert not paths.renders.exists()


def _fake_receipt(regime: str, repeat: int) -> MODULE.OutcomeReceipt:
    return MODULE.OutcomeReceipt(
        regime=regime,
        case_id="e01",
        repeat_index=repeat,
        completion_status="COMPLETE",
        durable_elapsed_seconds=100.0,
        raw_package=_verified(f"{regime}-r{repeat}-raw"),
        phase_package=_verified(f"{regime}-r{repeat}-phase"),
        checkpoint_chain=(),
        timing_receipt={"path": "timing", "sha256": "a", "size_bytes": 1},
        recovery_raw_path=None,
    )


def test_full_orchestration_freezes_all_decisions_before_any_derived_parse(
    monkeypatch, tmp_path
):
    events = []

    def technical(*, regime, **kwargs):
        events.append(f"technical-{regime}")
        return ({
            "model": object(), "tokenizer": object(),
            "runtime_fingerprint": {"eos_ids": [1]},
        }, {
            "status": "PASS", "regime": regime,
            "durable_elapsed_seconds": 100.0,
            "raw_package": _verified(f"{regime}-technical"),
            "renders": {"path": "render", "sha256": "a", "size_bytes": 1},
            "timing_receipt": None, "runtime_fingerprint": {},
            "subject_bindings": {}, "recovery_raw_path": None,
            "persistence_error": None, "stage_timings": {},
            "durable_checkpoints": [],
        })

    def outcome(*, regime, repeat_index, **kwargs):
        events.append(f"outcome-{regime}-r{repeat_index}")
        return _fake_receipt(regime, repeat_index)

    def derive(receipt, *, paths, args):
        run_dirs = list(args.output_dir.glob("precision-probe-p02_*"))
        assert len(run_dirs) == 1
        decision_files = list(run_dirs[0].glob("precision-probe-p02-decision-*.json"))
        assert len(decision_files) == 4
        events.append(f"derive-{receipt.regime}-r{receipt.repeat_index}")
        MODULE.p01.atomic_create_json(paths.compact, {"ok": True})
        MODULE.p01.atomic_create_text(paths.renders, "render\n")
        return replace(
            receipt, compact_path=str(paths.compact), renders_path=str(paths.renders)
        )

    monkeypatch.setattr(MODULE, "run_regime_technical", technical)
    monkeypatch.setattr(MODULE, "execute_outcome", outcome)
    monkeypatch.setattr(MODULE, "materialize_derived", derive)
    monkeypatch.setattr(MODULE.p01, "matched_runtime_gate",
                        lambda *a, **k: _runtime_gate_pass())
    monkeypatch.setattr(MODULE, "repeat_stability",
                        lambda *a, **k: {"nf4": {"status": "PASS"},
                                        "bf16": {"status": "PASS"}})
    args = argparse.Namespace(
        repo=ROOT,
        output_dir=tmp_path,
        model=MODULE.MODEL_ID,
        revision=MODULE.REVISION,
        prereg=MODULE.DEFAULT_PREREG,
        identity_fixture=MODULE.DEFAULT_IDENTITY,
        technical_fixture=MODULE.DEFAULT_TECHNICAL,
        case_e01=MODULE.DEFAULT_CASE,
        allow_download=False,
        hourly_cost_usd=1.39,
        provider_elapsed_seconds_at_start=0.0,
        provider_wall_cap_seconds=9000.0,
    )
    completion_path, manifest, code = MODULE.run(args)
    assert code == 0
    assert manifest["status"] == "COMPLETE"
    assert events[:6] == [
        "technical-nf4", "outcome-nf4-r1", "outcome-nf4-r2",
        "technical-bf16", "outcome-bf16-r1", "outcome-bf16-r2",
    ]
    first_derived = next(index for index, value in enumerate(events)
                         if value.startswith("derive-"))
    assert first_derived == 6
    completion = json.loads(completion_path.read_bytes())
    assert set(completion) == {
        "schema", "protocol_id", "formal_v12_decision_eligible",
        "v12_reentry_authorized",
        "component_reuse_does_not_inherit_v12_eligibility",
        "semantic_evidence_eligible", "status", "model", "revision",
        "created_at_utc", "scientific_cap_seconds",
        "matched_repeat1_eligible", "repeat2_rider_status",
        "matched_runtime_gate", "run_manifest", "continuation_decisions",
        "artifact_inventory", "recovery_raw_files",
        "provider_elapsed_seconds", "estimated_provider_cost_usd",
    }
    assert set(completion["continuation_decisions"]) == set(MODULE.DECISION_NAMES)
    assert completion["matched_repeat1_eligible"] is True
    assert completion["repeat2_rider_status"] == "MATCHED_COMPLETE"


def test_primary_eligibility_ignores_asymmetric_repeat2_rider(tmp_path):
    receipts = [
        replace(_fake_receipt("nf4", 1),
                compact_path=str(tmp_path / "n.json"),
                renders_path=str(tmp_path / "n.md")),
        replace(_fake_receipt("bf16", 1),
                compact_path=str(tmp_path / "b.json"),
                renders_path=str(tmp_path / "b.md")),
        _fake_receipt("nf4", 2),
        MODULE.skipped_outcome("bf16", 2, "SKIPPED_NOT_AUTHORIZED"),
    ]
    for name in ("n.json", "n.md", "b.json", "b.md"):
        (tmp_path / name).write_text("x")
    technical = {
        "nf4": {"status": "PASS"}, "bf16": {"status": "PASS"}
    }
    assert MODULE.matched_repeat1_eligible(
        receipts, technical, _runtime_gate_pass()
    ) is True
    status = MODULE.repeat2_rider_status(
        {"both_repeat2_authorized": True}, {"authorized": False}, receipts
    )
    assert status == "NF4_ONLY_TIMING_STOP"


def test_primary_eligibility_fails_closed_on_runtime_or_technical_mismatch(tmp_path):
    receipts = []
    for regime in MODULE.REGIMES:
        compact = tmp_path / f"{regime}.json"
        render = tmp_path / f"{regime}.md"
        compact.write_text("{}")
        render.write_text("render")
        receipts.append(replace(
            _fake_receipt(regime, 1),
            compact_path=str(compact), renders_path=str(render),
        ))
    assert not MODULE.matched_repeat1_eligible(
        receipts,
        {"nf4": {"status": "PASS"}, "bf16": {"status": "ERROR"}},
        _runtime_gate_pass(),
    )
    mismatch = _runtime_gate_pass() | {"status": "FAIL"}
    assert not MODULE.matched_repeat1_eligible(
        receipts,
        {"nf4": {"status": "PASS"}, "bf16": {"status": "PASS"}},
        mismatch,
    )
