from __future__ import annotations

import argparse
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
    "run_precision_probe_p01", ROOT / "scripts/run_precision_probe_p01.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _score(decoded: str = "partner beta") -> dict:
    return {
        "probe": "Which ring?",
        "suffix_ids": [7, 8],
        "correct_text": "partner beta",
        "counterfactual_text": "staff ring",
        "correct": {
            "target_token_ids": [10, 11],
            "token_logprobs": [-0.1, -0.2],
            "token_logprob_float32_bits": ["cdccccbd", "cdcc4cbe"],
            "mean_logprob": -0.15,
        },
        "counterfactual": {
            "target_token_ids": [12, 13],
            "token_logprobs": [-2.0, -3.0],
            "token_logprob_float32_bits": ["000000c0", "000040c0"],
            "mean_logprob": -2.5,
        },
        "margin": 2.35,
        "margin_float32_bits": "66661640",
        "margin_arithmetic": "float32",
        "generation": {
            "content_ids": [10, 11],
            "decoded_content": decoded,
            "stop_reason": "model_eos",
            "cap_hit": False,
        },
    }


def _outcome_document(status: str = "COMPLETE") -> dict:
    score = _score()
    return {
        "schema": MODULE.OUTCOME_SCHEMA,
        "protocol_id": MODULE.PROTOCOL_ID,
        **MODULE.FORMAL_FLAGS,
        "status": status,
        "regime": "nf4",
        "case_id": "e01",
        "repeat_index": 1,
        "outcome_wall_time_seconds": 12.5,
        "phase_a": {
            "schema": MODULE.PHASE_A_SCHEMA,
            "case_id": "e01",
            "visible_messages": {
                "A_C": [{"role": "user", "content": "literal C"}],
                "A_W": [{"role": "user", "content": "literal W"}],
                "FF": [{"role": "assistant", "content": "literal fresh"}],
            },
            "scores": {"A_C_focal": score},
        },
        "treatment": {
            "schema": MODULE.TREATMENT_SCHEMA,
            "case_id": "e01",
            "fresh_scores": {"focal": score},
            "arms": [{
                "arm_kind": "primary", "schedule": "N",
                "region": "R2_boundary", "cell": "FC",
                "key_source": "F", "value_source": "C",
                "scores": {"focal": score, "nonfocal": _score("21 days")},
            }],
        },
    }


def test_extension_decision_has_scalar_only_information_surface():
    assert set(MODULE.ExtensionTiming.__slots__) == {
        "completion_status", "outcome_wall_time_seconds"
    }
    assert not hasattr(MODULE.ExtensionTiming("COMPLETE", 10.0), "__dict__")
    signature = inspect.signature(MODULE.decide_extension)
    assert list(signature.parameters) == [
        "e01_timings", "provider_elapsed_seconds", "hourly_cost_usd",
        "provider_wall_cap_seconds",
    ]

    decision = MODULE.decide_extension(
        e01_timings=[
            MODULE.ExtensionTiming("COMPLETE", 100.0),
            MODULE.ExtensionTiming("COMPLETE", 120.0),
        ],
        provider_elapsed_seconds=300.0,
        hourly_cost_usd=2.0,
        provider_wall_cap_seconds=7200.0,
    )
    assert decision["extension_allowed"] is True
    assert decision["selected_case_repeats"] == {
        "e01": 2, "e02": 1, "e03": 1
    }
    assert decision["inputs"]["slower_complete_nf4_e01_seconds"] == 120.0
    assert decision["score_or_generation_accessible_to_decision"] is False

    source = inspect.getsource(MODULE.decide_extension)
    assert "open(" not in source
    assert "load_object" not in source
    assert "raw_package" not in source
    assert "compact" not in source


def test_planned_names_are_unique_and_phase_checkpoints_are_not_final_outcomes(
    tmp_path
):
    planned = MODULE._planned_paths(tmp_path, "20260712T070000000000Z")
    paths = []
    for technical in planned["technical"].values():
        paths.extend(technical.values())
    for outcome in planned["outcomes"].values():
        paths.extend(Path(value) for value in MODULE.asdict(outcome).values())
        assert not fnmatch.fnmatch(
            outcome.phase_a_checkpoint_package.name,
            "precision-probe-p01-outcome-*-raw_*.lossless-package",
        )
        assert fnmatch.fnmatch(
            outcome.raw_package.name,
            "precision-probe-p01-outcome-*-raw_*.lossless-package",
        )
    paths.extend((planned["run_manifest"], planned["extension_decision"],
                  planned["completion"]))
    assert len(paths) == len(set(paths))
    assert all(len(path.name.encode("utf-8")) < 255 for path in paths)


@pytest.mark.parametrize(
    ("timings", "elapsed", "allowed"),
    [
        ([MODULE.ExtensionTiming("ERROR", 1.0),
          MODULE.ExtensionTiming("COMPLETE", 1.0)], 0.0, False),
        ([MODULE.ExtensionTiming("COMPLETE", 900.0),
          MODULE.ExtensionTiming("COMPLETE", 900.0)], 100.0, False),
    ],
)
def test_extension_fail_closed(timings, elapsed, allowed):
    decision = MODULE.decide_extension(
        e01_timings=timings,
        provider_elapsed_seconds=elapsed,
        hourly_cost_usd=2.0,
        provider_wall_cap_seconds=7200.0,
    )
    assert decision["extension_allowed"] is allowed
    assert decision["selected_case_repeats"] == {"e01": 2}


def test_lossless_raw_is_verified_before_scratch_removal(tmp_path):
    scratch = tmp_path / ".scratch" / "raw.json"
    package = tmp_path / "raw.lossless-package"
    document = {"scores": {"do_not_interpret": [1, 2, 3]}, "ok": True}
    packaged, recovery, error = MODULE.persist_lossless_raw(
        document, scratch_raw=scratch, package=package, repo=tmp_path
    )
    assert error is None
    assert recovery is None
    assert packaged["verification_status"] == "VERIFIED"
    assert not scratch.exists()
    manifest = json.loads((package / "manifest.json").read_bytes())
    assert manifest["original"]["json_validated"] is False
    reconstructed = tmp_path / "reconstructed.json"
    verification = MODULE.artifact_packager.verify_and_reconstruct(
        package, reconstructed
    )
    assert verification["status"] == "VERIFIED"
    assert json.loads(reconstructed.read_bytes()) == document


def test_packaging_failure_preserves_recovery_raw(monkeypatch, tmp_path):
    scratch = tmp_path / ".scratch" / "raw.json"
    package = tmp_path / "raw.lossless-package"

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic package failure")

    monkeypatch.setattr(MODULE.artifact_packager, "pack", fail)
    packaged, recovery, error = MODULE.persist_lossless_raw(
        {"expensive": "render"}, scratch_raw=scratch,
        package=package, repo=tmp_path,
    )
    assert packaged is None
    assert recovery == ".scratch/raw.json"
    assert "synthetic package failure" in error
    assert scratch.is_file()


def test_execute_outcome_packages_partial_phase_on_treatment_error(
    monkeypatch, tmp_path
):
    case_path = tmp_path / "e01.json"
    case_path.write_text('{"case_id":"e01"}\n')
    paths = MODULE.OutcomePaths(
        raw_scratch=tmp_path / ".scratch/raw.json",
        raw_package=tmp_path / "raw.lossless-package",
        phase_a_checkpoint_scratch=tmp_path / ".scratch/phase.json",
        phase_a_checkpoint_package=tmp_path / "phase.lossless-package",
        compact=tmp_path / "compact.json",
        renders=tmp_path / "renders.md",
    )
    phase = {"schema": MODULE.PHASE_A_SCHEMA, "case_id": "e01", "scores": {}}
    monkeypatch.setattr(MODULE, "run_phase_a_case", lambda *a, **k: phase)

    def treatment_failure(*args, **kwargs):
        raise RuntimeError("treatment stopped")

    monkeypatch.setattr(MODULE, "run_treatment_case", treatment_failure)
    args = SimpleNamespace(
        case_paths={"e01": case_path}, repo=tmp_path,
        model=MODULE.MODEL_ID, revision=MODULE.REVISION,
        hourly_cost_usd=1.0,
    )
    prepared = {
        "model": object(), "tokenizer": object(),
        "runtime_fingerprint": {"eos_ids": [1, 2]},
    }
    clock = MODULE.ProviderClock(0.0, MODULE.time.monotonic())
    receipt = MODULE.execute_outcome(
        regime="nf4", case_id="e01", repeat_index=1,
        prepared=prepared, args=args, paths=paths,
        common_bindings={}, provider_clock=clock,
    )
    assert receipt.completion_status == "ERROR"
    assert receipt.raw_package["verification_status"] == "VERIFIED"
    raw = MODULE._load_packaged_document(
        paths.raw_package, scratch_dir=paths.raw_scratch.parent
    )
    assert raw["status"] == "ERROR"
    assert raw["phase_a"] == phase
    assert "treatment stopped" in raw["error"]["message"]
    assert raw["durable_checkpoints"][0]["stage"] == "phase_a"
    assert paths.phase_a_checkpoint_package.is_dir()
    assert not paths.raw_scratch.exists()


def test_interrupt_after_phase_is_persisted_before_propagation(
    monkeypatch, tmp_path
):
    case_path = tmp_path / "e01.json"
    case_path.write_text('{"case_id":"e01"}\n')
    paths = MODULE.OutcomePaths(
        raw_scratch=tmp_path / ".scratch/raw.json",
        raw_package=tmp_path / "raw.lossless-package",
        phase_a_checkpoint_scratch=tmp_path / ".scratch/phase.json",
        phase_a_checkpoint_package=tmp_path / "phase.lossless-package",
        compact=tmp_path / "compact.json",
        renders=tmp_path / "renders.md",
    )
    phase = {"schema": MODULE.PHASE_A_SCHEMA, "case_id": "e01", "scores": {}}
    monkeypatch.setattr(MODULE, "run_phase_a_case", lambda *a, **k: phase)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("budget watchdog")

    monkeypatch.setattr(MODULE, "run_treatment_case", interrupt)
    args = SimpleNamespace(
        case_paths={"e01": case_path}, repo=tmp_path,
        model=MODULE.MODEL_ID, revision=MODULE.REVISION,
        hourly_cost_usd=1.0,
    )
    prepared = {
        "model": object(), "tokenizer": object(),
        "runtime_fingerprint": {"eos_ids": [1, 2]},
    }
    with pytest.raises(KeyboardInterrupt, match="budget watchdog"):
        MODULE.execute_outcome(
            regime="nf4", case_id="e01", repeat_index=1,
            prepared=prepared, args=args, paths=paths,
            common_bindings={}, provider_clock=MODULE.ProviderClock(
                0.0, MODULE.time.monotonic()
            ),
        )
    assert paths.phase_a_checkpoint_package.is_dir()
    assert paths.raw_package.is_dir()
    assert paths.renders.is_file()
    assert "Status: `ERROR`" in paths.renders.read_text()
    raw = MODULE._load_packaged_document(
        paths.raw_package, scratch_dir=paths.raw_scratch.parent
    )
    assert raw["status"] == "ERROR"
    assert raw["error"]["type"] == "KeyboardInterrupt"
    assert raw["phase_a"] == phase


def test_compact_and_markdown_preserve_target_tokens_and_literal_generations(
    tmp_path
):
    paths = MODULE.OutcomePaths(
        raw_scratch=tmp_path / ".scratch/raw.json",
        raw_package=tmp_path / "raw.lossless-package",
        phase_a_checkpoint_scratch=tmp_path / ".scratch/phase.json",
        phase_a_checkpoint_package=tmp_path / "phase.lossless-package",
        compact=tmp_path / "compact.json",
        renders=tmp_path / "renders.md",
    )
    document = _outcome_document()
    package, _, error = MODULE.persist_lossless_raw(
        document, scratch_raw=paths.raw_scratch,
        package=paths.raw_package, repo=tmp_path,
    )
    assert error is None
    receipt = MODULE.OutcomeReceipt(
        regime="nf4", case_id="e01", repeat_index=1,
        completion_status="COMPLETE", outcome_wall_time_seconds=12.5,
        raw_package=package, recovery_raw_path=None,
    )
    args = SimpleNamespace(repo=tmp_path)
    derived = MODULE.materialize_derived(receipt, paths=paths, args=args)
    compact = json.loads(paths.compact.read_bytes())
    record = compact["phase_a_scores"]["A_C_focal"]
    assert record["correct"]["target_token_ids"] == [10, 11]
    assert record["correct"]["token_logprobs"] == [-0.1, -0.2]
    assert record["generation"]["decoded_content"] == "partner beta"
    ledger = paths.renders.read_text()
    for literal in (
        "literal C", "literal W", "literal fresh", "partner beta", "21 days"
    ):
        assert literal in ledger
    assert derived.compact_path == str(paths.compact)
    assert derived.renders_path == str(paths.renders)


def test_technical_gate_requires_nine_cells_and_bf16_dtype():
    hashes = [{"k_dtype": "torch.bfloat16", "v_dtype": "torch.bfloat16"}]
    passing = {
        "generated_forced_identity": {
            "status": "PASS", "dtype_witness": hashes,
        },
        "deterministic_repeats": {
            "correct_history_N": {"status": "PASS", "records": hashes},
            "fresh_destination": {"status": "PASS", "records": hashes},
        },
        "fresh_self_replacement": {
            "status": "PASS",
            "regions": [
                {"status": "PASS", "fresh_selected_row_hashes": hashes}
                for _ in range(9)
            ],
        },
    }
    MODULE._assert_technical_gate(passing)
    passing["generated_forced_identity"]["dtype_witness"] = [
        {"k_dtype": "torch.float16", "v_dtype": "torch.float16"}
    ]
    with pytest.raises(MODULE.PrecisionProbeRunnerError, match="non-bf16"):
        MODULE._assert_technical_gate(passing)


def test_technical_failure_releases_loaded_subject(monkeypatch, tmp_path):
    prepared = {
        "model": object(), "tokenizer": object(),
        "runtime_fingerprint": {"eos_ids": [1], "kv_dtype": "torch.bfloat16"},
        "bindings": {"host": {"gpu_uuid": "GPU-one"}},
    }
    monkeypatch.setattr(
        MODULE, "prepare_precision_subject", lambda *args: prepared
    )
    monkeypatch.setattr(
        MODULE, "run_generated_forced_identity",
        lambda *args: {"status": "FAIL", "separate_branches": []},
    )
    released = []
    monkeypatch.setattr(MODULE, "_release_subject", lambda value: released.append(value))
    base = tmp_path / "technical"
    paths = {
        "raw_scratch": tmp_path / ".scratch/technical.json",
        "raw_package": tmp_path / "technical.lossless-package",
        "identity_checkpoint_scratch": tmp_path / ".scratch/identity.json",
        "identity_checkpoint_package": tmp_path / "identity.lossless-package",
        "renders": base.with_suffix(".md"),
    }
    args = SimpleNamespace(
        repo=tmp_path, model=MODULE.MODEL_ID, revision=MODULE.REVISION,
        allow_download=False, identity_fixture=MODULE.DEFAULT_IDENTITY,
        technical_fixture=MODULE.DEFAULT_TECHNICAL,
        hourly_cost_usd=1.0,
    )
    returned, receipt = MODULE.run_regime_technical(
        regime="nf4", args=args, paths=paths, bindings={},
        provider_clock=MODULE.ProviderClock(0.0, MODULE.time.monotonic()),
    )
    assert returned is None
    assert receipt["status"] == "ERROR"
    assert released == [prepared]
    assert paths["identity_checkpoint_package"].is_dir()
    assert paths["raw_package"].is_dir()


def test_complete_contract_requires_every_compact_and_render(tmp_path):
    compact = tmp_path / "compact.json"
    renders = tmp_path / "renders.md"
    compact.write_text("{}\n")
    renders.write_text("# render\n")
    receipts = []
    for regime in MODULE.REGIMES:
        for repeat in (1, 2):
            receipts.append(MODULE.OutcomeReceipt(
                regime=regime, case_id="e01", repeat_index=repeat,
                completion_status="COMPLETE", outcome_wall_time_seconds=1.0,
                raw_package={"verification_status": "VERIFIED"},
                recovery_raw_path=None, compact_path=str(compact),
                renders_path=str(renders),
            ))
    assert MODULE.selected_outcomes_complete(receipts, {"e01": 2}) is True
    receipts[0] = MODULE.OutcomeReceipt(
        regime="nf4", case_id="e01", repeat_index=1,
        completion_status="COMPLETE", outcome_wall_time_seconds=1.0,
        raw_package={"verification_status": "VERIFIED"},
        recovery_raw_path=None, compact_path=None, renders_path=str(renders),
    )
    assert MODULE.selected_outcomes_complete(receipts, {"e01": 2}) is False


def test_matched_runtime_gate_ignores_only_the_manipulated_regime_fields():
    runtime = {
        "model_id": MODULE.MODEL_ID,
        "requested_revision": MODULE.REVISION,
        "resolved_snapshot": MODULE.REVISION,
        "architecture": "Qwen3MoeForCausalLM",
        "attention_backend": "eager",
        "kv_dtype": "torch.bfloat16",
        "eos_ids": [151643, 151645],
        "geometry": {"layers": 48},
        "repository_commit": "a" * 40,
        "dependency_versions": {"torch": "2.12.1"},
        "gpu_uuid": "GPU-one",
        "weight_runtime": "deliberately different",
    }
    bindings = {
        key: {"same": True} for key in (
            "repository", "dependencies", "host", "checkpoint", "g0",
            "tokenizer", "model", "eos", "placement",
        )
    }
    nf4 = {
        "status": "PASS", "runtime_fingerprint": dict(runtime),
        "subject_bindings": bindings,
    }
    bf16 = {
        "status": "PASS", "runtime_fingerprint": dict(runtime),
        "subject_bindings": bindings,
    }
    assert MODULE.matched_runtime_gate(nf4, bf16)["status"] == "PASS"
    bf16["runtime_fingerprint"] = dict(runtime, gpu_uuid="GPU-two")
    result = MODULE.matched_runtime_gate(nf4, bf16)
    assert result["status"] == "FAIL"
    assert "$.runtime.gpu_uuid" in result["differing_fields"]
    absent = MODULE.matched_runtime_gate(
        {"status": "PASS", "runtime_fingerprint": {}, "subject_bindings": {}},
        {"status": "PASS", "runtime_fingerprint": {}, "subject_bindings": {}},
    )
    assert absent["status"] == "NOT_EVALUABLE_BINDINGS_ABSENT"
    assert absent["missing_fields"]


def _run_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo=ROOT,
        output_dir=tmp_path / "results/precision_probe_p01",
        model=MODULE.MODEL_ID,
        revision=MODULE.REVISION,
        prereg=MODULE.DEFAULT_PREREG,
        identity_fixture=MODULE.DEFAULT_IDENTITY,
        technical_fixture=MODULE.DEFAULT_TECHNICAL,
        case_paths=dict(MODULE.DEFAULT_CASES),
        allow_download=False,
        hourly_cost_usd=2.0,
        provider_elapsed_seconds_at_start=0.0,
        provider_wall_cap_seconds=7200.0,
    )


def test_runner_freezes_decision_before_any_nf4_derivation(
    monkeypatch, tmp_path
):
    events: list[tuple] = []

    def fake_technical(*, regime, **kwargs):
        events.append(("technical", regime))
        prepared = {
            "model": object(), "tokenizer": object(),
            "runtime_fingerprint": {"eos_ids": [1]},
        }
        runtime = {
            field: "same" for field in (
                "model_id", "requested_revision", "resolved_snapshot",
                "architecture", "attention_backend", "kv_dtype", "eos_ids",
                "geometry", "repository_commit", "dependency_versions", "gpu_uuid",
            )
        }
        bindings = {
            field: {"same": True} for field in (
                "repository", "dependencies", "host", "checkpoint", "g0",
                "tokenizer", "model", "eos", "placement",
            )
        }
        return prepared, {
            "status": "PASS", "regime": regime,
            "runtime_fingerprint": runtime, "subject_bindings": bindings,
        }

    def fake_execute(*, regime, case_id, repeat_index, paths, **kwargs):
        events.append(("execute", regime, case_id, repeat_index))
        paths.raw_package.mkdir(parents=True)
        (paths.raw_package / "manifest.json").write_text("{}\n")
        return MODULE.OutcomeReceipt(
            regime=regime, case_id=case_id, repeat_index=repeat_index,
            completion_status="COMPLETE", outcome_wall_time_seconds=10.0,
            raw_package={"verification_status": "VERIFIED", "path": str(paths.raw_package)},
            recovery_raw_path=None,
        )

    original_decision = MODULE.decide_extension

    def fake_decision(**kwargs):
        # Both immutable packages exist, but neither compact/render file has
        # been created or parsed when the decision is taken.
        events.append(("decision",))
        assert [event for event in events if event[0] == "execute"] == [
            ("execute", "nf4", "e01", 1),
            ("execute", "nf4", "e01", 2),
        ]
        assert not [event for event in events if event[0] == "derive"]
        assert all(type(row) is MODULE.ExtensionTiming
                   for row in kwargs["e01_timings"])
        return original_decision(**kwargs)

    def fake_derive(receipt, *, paths, args):
        events.append(("derive", receipt.regime, receipt.case_id, receipt.repeat_index))
        MODULE.atomic_create_json(paths.compact, {"status": "fake"})
        MODULE.atomic_create_text(paths.renders, "# fake\n")
        return MODULE.OutcomeReceipt(
            regime=receipt.regime, case_id=receipt.case_id,
            repeat_index=receipt.repeat_index,
            completion_status=receipt.completion_status,
            outcome_wall_time_seconds=receipt.outcome_wall_time_seconds,
            raw_package=receipt.raw_package, recovery_raw_path=None,
            compact_path=str(paths.compact), renders_path=str(paths.renders),
        )

    monkeypatch.setattr(MODULE, "run_regime_technical", fake_technical)
    monkeypatch.setattr(MODULE, "execute_outcome", fake_execute)
    monkeypatch.setattr(MODULE, "decide_extension", fake_decision)
    monkeypatch.setattr(MODULE, "materialize_derived", fake_derive)
    monkeypatch.setattr(MODULE, "repeat_stability", lambda *a, **k: {
        "nf4": {"status": "PASS"}, "bf16": {"status": "PASS"}
    })
    monkeypatch.setattr(MODULE, "_release_subject", lambda prepared: None)

    completion_path, manifest, code = MODULE.run(_run_args(tmp_path))
    assert code == 0
    assert manifest["status"] == "COMPLETE"
    assert events.index(("decision",)) < events.index(("derive", "nf4", "e01", 1))
    assert events.index(("derive", "nf4", "e01", 2)) < events.index(
        ("execute", "nf4", "e02", 1)
    )
    assert [event for event in events if event[:2] == ("execute", "bf16")] == [
        ("execute", "bf16", "e01", 1),
        ("execute", "bf16", "e01", 2),
        ("execute", "bf16", "e02", 1),
        ("execute", "bf16", "e03", 1),
    ]
    completion = json.loads(completion_path.read_bytes())
    assert completion["schema"] == MODULE.COMPLETION_SCHEMA
    assert completion["status"] == "COMPLETE"
    assert completion["recovery_raw_files"] == []
    assert set(completion) == {
        "schema", "protocol_id", *MODULE.FORMAL_FLAGS,
        "status", "model", "revision", "created_at_utc", "run_manifest",
        "matched_runtime_gate", "artifact_inventory", "recovery_raw_files",
        "provider_elapsed_seconds",
        "estimated_provider_cost_usd",
    }


def test_fixed_subject_and_ceiling_are_fail_closed(tmp_path):
    args = _run_args(tmp_path)
    args.model = "wrong/model"
    with pytest.raises(MODULE.PrecisionProbeRunnerError, match="model differs"):
        MODULE.run(args)
    args = _run_args(tmp_path)
    args.provider_elapsed_seconds_at_start = 7200.0
    with pytest.raises(MODULE.PrecisionProbeRunnerError, match="already exhausted"):
        MODULE.run(args)


def test_top_level_interrupt_writes_error_completion_then_propagates(
    monkeypatch, tmp_path
):
    def interrupt(**kwargs):
        raise KeyboardInterrupt("watchdog")

    monkeypatch.setattr(MODULE, "run_regime_technical", interrupt)
    with pytest.raises(KeyboardInterrupt, match="watchdog"):
        MODULE.run(_run_args(tmp_path))
    completions = list(tmp_path.rglob(
        "precision-probe-p01-completion_Qwen3-30B-A3B-Instruct-2507_*.json"
    ))
    assert len(completions) == 1
    completion = json.loads(completions[0].read_bytes())
    assert completion["status"] == "ERROR"
    assert completion["matched_runtime_gate"] is None
    manifest_path = Path(completion["run_manifest"]["path"])
    assert json.loads(manifest_path.read_bytes())["status"] == "ERROR"
