from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_precision_probe_p01.py"
SPEC = importlib.util.spec_from_file_location("analyze_precision_probe_p01", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PACKAGER_PATH = ROOT / (
    "scripts/historical/coherent_canary_v12_e01/"
    "treatment_packaging/artifact_packager.py"
)
PACKAGER_SPEC = importlib.util.spec_from_file_location(
    "p01_analysis_test_packager", PACKAGER_PATH
)
assert PACKAGER_SPEC is not None and PACKAGER_SPEC.loader is not None
PACKAGER = importlib.util.module_from_spec(PACKAGER_SPEC)
sys.modules[PACKAGER_SPEC.name] = PACKAGER
PACKAGER_SPEC.loader.exec_module(PACKAGER)

PHASE_RAW = ROOT / (
    "results/coherent_canary_v12_phase_a/"
    "coherent-canary-v12-phase-a-e01_exact-subject_20260712T030639073087Z.json"
)
TREATMENT_PACKAGE = ROOT / (
    "results/coherent_canary_v12_treatment_package/"
    "coherent-canary-v12-treatment-e01_exact-subject_20260712T040237879370Z"
)


def historical_combined_outcome() -> dict:
    phase_run = MODULE.load_object(PHASE_RAW)
    _, treatment_run = MODULE.reconstruct_package(TREATMENT_PACKAGE)
    return {
        "schema": MODULE.OUTCOME_RAW_SCHEMA,
        "protocol_id": MODULE.PROTOCOL_ID,
        "formal_v12_decision_eligible": False,
        "v12_reentry_authorized": False,
        "component_reuse_does_not_inherit_v12_eligibility": True,
        "semantic_evidence_eligible": False,
        "regime": "bf16",
        "case_id": "e01",
        "repeat_index": 1,
        "phase_a": phase_run["phase_a"],
        "treatment": treatment_run["treatment"],
    }


def test_reconstructs_historical_lossless_package() -> None:
    manifest, raw = MODULE.reconstruct_package(TREATMENT_PACKAGE)
    assert manifest["original"]["sha256"] == (
        "f6622d978c80d8f4f874c208ef9f6d9546ab3f1bf7676655c9a0b4662718d082"
    )
    assert raw["case_id"] == "e01"
    assert raw["treatment"]["primary_arm_count"] == 31


def test_recovers_known_e01_metrics_without_runner_summaries() -> None:
    result = MODULE.analyze_outcome(historical_combined_outcome())
    primary = result["N_regional_estimands"]["R2_boundary"]["value_only"]
    full_kv = result["N_regional_estimands"]["R2_boundary"]["full_KV"]

    assert result["primary_D_value_N_R2"] == pytest.approx(0.1745452881)
    assert result["primary_nonfocal_D_value_N_R2"] == pytest.approx(-0.0000696182)
    assert primary["SEL_absolute_nonfocal_penalty_historical"] == pytest.approx(
        0.1744756699
    )
    assert primary["Hplus"] == pytest.approx(0.7102603912)
    assert primary["U"] == pytest.approx(0.1816501617)
    assert primary["Uplus"] == pytest.approx(0.9459857941)
    assert full_kv["D_focal"] == pytest.approx(0.0972366333)
    assert result["phase_a_oracle_to_fresh_damage"]["focal"][
        "margin"
    ] == pytest.approx(22.2984294891)
    assert result["available_placebo_control_count"] == 0
    assert len(result["all_primary_generations"]) == 62


def test_signed_and_historical_selectivity_are_not_conflated() -> None:
    result = MODULE.analyze_outcome(historical_combined_outcome())
    row = result["N_regional_estimands"]["R2_boundary"]["value_only"]
    assert row["SEL_signed_p01"] != row[
        "SEL_absolute_nonfocal_penalty_historical"
    ]
    assert row["SEL_signed_p01"] == pytest.approx(
        row["D_focal"] - row["D_nonfocal"]
    )


def test_identification_fails_closed_on_v12_eligibility() -> None:
    raw = historical_combined_outcome()
    raw["formal_v12_decision_eligible"] = True
    with pytest.raises(MODULE.AnalysisError, match="ineligibility"):
        MODULE.identify_outcome(raw)


def test_recursive_diff_reports_json_pointer_paths() -> None:
    left = {"phase_a": {"scores": [{"margin": 1.0}]}, "same": [1, 2]}
    right = {"phase_a": {"scores": [{"margin": 2.0}]}, "same": [1, 2]}
    assert MODULE.recursive_differences(left, right) == [
        "/phase_a/scores/0/margin"
    ]


def test_phase_a_checkpoint_package_is_not_counted_as_final_outcome(
    tmp_path: Path,
) -> None:
    final = historical_combined_outcome()
    checkpoint = dict(final)
    checkpoint["status"] = "CHECKPOINT"
    final["status"] = "COMPLETE"
    generic = {
        "package_dir": tmp_path,
        "package_manifest_sha256": "a" * 64,
        "raw_artifact_sha256": "b" * 64,
    }
    rows = MODULE.package_rows(tmp_path, [
        {**generic, "raw": checkpoint},
        {**generic, "raw": final},
    ])
    assert len(rows) == 1
    assert rows[0]["key"] == ("bf16", "e01", 1)
    assert rows[0]["raw"]["status"] == "COMPLETE"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _pack(run_dir: Path, name: str, value: dict) -> Path:
    source = run_dir / f".{name}.source.json"
    package = run_dir / f"{name}.lossless-package"
    _write_json(source, value)
    PACKAGER.pack(source, package)
    source.unlink()
    return package


def _file_binding(path: Path, run_dir: Path) -> dict:
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "sha256": MODULE.file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _runtime(regime: str) -> dict:
    return {
        "schema": "precision_probe_p01_runtime_fingerprint_v1",
        "protocol_id": MODULE.PROTOCOL_ID,
        "regime": regime,
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "requested_revision":
            "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "resolved_snapshot":
            "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "architecture": "Qwen3MoeForCausalLM",
        "attention_backend": "eager",
        "weight_runtime": "nf4" if regime == "nf4" else "bf16",
        "kv_dtype": "torch.bfloat16",
        "eos_ids": [151643, 151645],
        "geometry": {"layers": 48, "kv_heads": 4, "head_dim": 128},
        "repository_commit": "a" * 40,
        "dependency_versions": {"transformers": "4.57.6"},
        "gpu_uuid": "GPU-test",
        "bindings_sha256": regime * 8,
        "fingerprint_sha256": ("1" if regime == "nf4" else "2") * 64,
    }


def _technical_document(regime: str) -> dict:
    if regime == "nf4":
        weights = {
            "regime": "nf4",
            "linear4bit_modules": 18_672,
            "expert_linear4bit_modules": 18_432,
            "expert_logical_coverage": {
                "observed": 28_991_029_248,
                "expected": 28_991_029_248,
            },
            "total_logical_quantized_coverage": {
                "observed": 29_909_581_824,
                "expected": 29_909_581_824,
                "checkpoint_total": 30_532_122_624,
            },
            "quant_type": "nf4",
            "double_quantization": True,
            "compute_dtype": "torch.bfloat16",
            "ordinary_linears": ["lm_head"],
            "sentinels": [
                {"max_abs_error": 0.5, "mean_abs_error": 0.1}
                for _ in range(9)
            ],
        }
    else:
        weights = {
            "regime": "bf16",
            "logical_parameter_elements": 30_532_122_624,
            "floating_parameter_dtype": "torch.bfloat16",
            "quantization_modules": [],
        }
    runtime = _runtime(regime)
    bindings = {
        "protocol_id": MODULE.PROTOCOL_ID,
        "regime": regime,
        "g0": {
            "passes": True,
            "topology": {
                "checkpoint_key_count": 18_867,
                "eligible_linear_count": 18_672,
                "eligible_logical_elements": 29_909_581_824,
            },
        },
        "weights": weights,
        "kv": {
            "observed_real_forward": True,
            "layers": 48,
            "dtype": "torch.bfloat16",
            "expected_shape": [1, 4, 27, 128],
        },
        "host": {
            "gpu_name": "NVIDIA A100 80GB PCIe",
            "gpu_uuid": "GPU-test",
        },
        "loading_info": {
            "missing_keys": [], "unexpected_keys": [],
            "mismatched_keys": [], "error_msgs": [],
        },
    }
    return {
        "schema": MODULE.TECHNICAL_RAW_SCHEMA,
        "protocol_id": MODULE.PROTOCOL_ID,
        "formal_v12_decision_eligible": False,
        "v12_reentry_authorized": False,
        "component_reuse_does_not_inherit_v12_eligibility": True,
        "semantic_evidence_eligible": False,
        "status": "PASS",
        "regime": regime,
        "runtime_fingerprint": runtime,
        "subject_attestation": {
            "runtime_fingerprint": runtime,
            "bindings": bindings,
        },
        "generated_forced_identity": {"status": "PASS"},
        "deterministic_repeats": {
            "correct_history_N": {"status": "PASS"},
            "fresh_destination": {"status": "PASS"},
        },
        "fresh_self_replacement": {
            "status": "PASS",
            "regions": [{"status": "PASS"} for _ in range(9)],
        },
    }


def test_analyze_run_validates_complete_gates_decision_and_checkpoint_filter(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "precision-probe-p01_test"
    run_dir.mkdir()
    for regime in MODULE.REGIMES:
        _pack(run_dir, f"technical-{regime}-raw", _technical_document(regime))

    outcome_receipts = {regime: [] for regime in MODULE.REGIMES}
    timings = {("nf4", 1): 900.0, ("nf4", 2): 900.0,
               ("bf16", 1): 800.0, ("bf16", 2): 800.0}
    for regime in MODULE.REGIMES:
        for repeat in (1, 2):
            raw = historical_combined_outcome()
            raw.update({
                "regime": regime,
                "repeat_index": repeat,
                "status": "COMPLETE",
                "runtime_fingerprint": _runtime(regime),
                "outcome_wall_time_seconds": timings[(regime, repeat)],
            })
            prefix = f"outcome-{regime}-e01-repeat{repeat}"
            package = _pack(run_dir, prefix + "-raw", raw)
            compact = run_dir / f"{prefix}-compact.json"
            renders = run_dir / f"{prefix}-renders.md"
            _write_json(compact, {"protocol_id": MODULE.PROTOCOL_ID})
            renders.write_text("# literal render\n")
            manifest = json.loads((package / "manifest.json").read_bytes())
            package_binding = {
                "path": package.name,
                "manifest_path": f"{package.name}/manifest.json",
                "manifest_sha256": MODULE.file_sha256(package / "manifest.json"),
                "manifest_size_bytes": (package / "manifest.json").stat().st_size,
                "chunk_count": manifest["chunk_count"],
                "original_sha256": manifest["original"]["sha256"],
                "original_size_bytes": manifest["original"]["size_bytes"],
                "compressed_sha256": manifest["compression"]["compressed_sha256"],
                "compressed_size_bytes": manifest["compression"][
                    "compressed_size_bytes"],
                "verification_status": "VERIFIED",
            }
            outcome_receipts[regime].append({
                "regime": regime,
                "case_id": "e01",
                "repeat_index": repeat,
                "completion_status": "COMPLETE",
                "outcome_wall_time_seconds": timings[(regime, repeat)],
                "raw_package": package_binding,
                "recovery_raw_path": None,
                "compact_path": str(compact),
                "renders_path": str(renders),
                "compact": _file_binding(compact, run_dir),
                "renders": _file_binding(renders, run_dir),
            })

    checkpoint = historical_combined_outcome()
    checkpoint.update({
        "status": "CHECKPOINT",
        "regime": "nf4",
        "repeat_index": 1,
        "runtime_fingerprint": _runtime("nf4"),
    })
    _pack(run_dir, "outcome-nf4-e01-repeat1-phase-a-checkpoint-raw",
          checkpoint)

    rate = 1.39
    elapsed = 100.0
    hard = min(7200.0, 4.0 * 3600.0 / rate)
    forecast = elapsed + 1.20 * (6.0 * 900.0 + 900.0)
    decision = {
        "schema": MODULE.DECISION_SCHEMA,
        "protocol_id": MODULE.PROTOCOL_ID,
        "formal_v12_decision_eligible": False,
        "v12_reentry_authorized": False,
        "component_reuse_does_not_inherit_v12_eligibility": True,
        "semantic_evidence_eligible": False,
        "decision": "BASE_E01_ONLY",
        "extension_allowed": False,
        "selected_case_repeats": {"e01": 2},
        "inputs": {
            "nf4_e01_completion_statuses": ["COMPLETE", "COMPLETE"],
            "nf4_e01_outcome_wall_time_seconds": [900.0, 900.0],
            "slower_complete_nf4_e01_seconds": 900.0,
            "provider_elapsed_seconds": elapsed,
            "hourly_cost_usd": rate,
            "provider_wall_cap_seconds": 7200.0,
            "four_dollar_rate_ceiling_seconds": 4.0 * 3600.0 / rate,
            "effective_hard_deadline_seconds": hard,
            "forecast_seconds": forecast,
        },
        "formula":
            "elapsed + 1.20 * (6*t + 900) <= min(7200, cap, 4*3600/rate)",
        "outcome_information_surface": [
            "completion_status", "outcome_wall_time_seconds"],
        "score_or_generation_accessible_to_decision": False,
    }
    decision_path = run_dir / "precision-probe-p01-extension-decision_test.json"
    _write_json(decision_path, decision)

    source_names = {
        "preregistration", "identity_fixture", "technical_fixture",
        "case_e01", "case_e02", "case_e03", "runner",
        "precision_subject_loader", "artifact_packager",
        "coherent_canary_case", "coherent_canary_runtime",
        "coherent_canary_technical", "coherent_canary_tokens",
        "coherent_canary_controls", "coherent_canary_schema",
        "coherent_state_tokens",
    }
    matched = {"status": "PASS", "differing_fields": []}
    manifest = {
        "schema": MODULE.RUN_MANIFEST_SCHEMA,
        "protocol_id": MODULE.PROTOCOL_ID,
        "formal_v12_decision_eligible": False,
        "v12_reentry_authorized": False,
        "component_reuse_does_not_inherit_v12_eligibility": True,
        "semantic_evidence_eligible": False,
        "status": "COMPLETE",
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "bindings": {name: {} for name in source_names},
        "provider": {"effective_hard_deadline_seconds": 7200.0},
        "regimes": {
            regime: {
                "status": "OUTCOMES_FINISHED",
                "technical": {"status": "PASS"},
                "outcomes": outcome_receipts[regime],
            } for regime in MODULE.REGIMES
        },
        "matched_runtime_gate": matched,
        "selected_case_repeats": {"e01": 2},
        "extension_decision": _file_binding(decision_path, run_dir),
    }
    manifest_path = run_dir / "precision-probe-p01-run-manifest_test.json"
    _write_json(manifest_path, manifest)

    inventory = [
        _file_binding(path, run_dir)
        for path in sorted(item for item in run_dir.rglob("*") if item.is_file())
    ]
    completion = {
        "schema": MODULE.COMPLETION_SCHEMA,
        "protocol_id": MODULE.PROTOCOL_ID,
        "formal_v12_decision_eligible": False,
        "v12_reentry_authorized": False,
        "component_reuse_does_not_inherit_v12_eligibility": True,
        "semantic_evidence_eligible": False,
        "status": "COMPLETE",
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "run_manifest": _file_binding(manifest_path, run_dir),
        "matched_runtime_gate": matched,
        "artifact_inventory": inventory,
        "recovery_raw_files": [],
        "provider_elapsed_seconds": 2000.0,
        "estimated_provider_cost_usd": 2000.0 * rate / 3600.0,
    }
    completion_path = run_dir / "precision-probe-p01-completion_test.json"
    _write_json(completion_path, completion)

    result = MODULE.analyze_run(run_dir)
    assert result["technical_gates"]["status"] == "PASS"
    assert result["runner_completion_and_manifest"][
        "selected_case_repeats"] == {"e01": 2}
    assert result["outcome_keys"] == [
        ["bf16", "e01", 1], ["bf16", "e01", 2],
        ["nf4", "e01", 1], ["nf4", "e01", 2],
    ]
    assert result["cross_runtime_primary_case_summary"]["mean"] == 0.0
