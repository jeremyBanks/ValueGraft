from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_precision_probe_p02.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_precision_probe_p02", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PACKAGER_PATH = ROOT / (
    "scripts/historical/coherent_canary_v12_e01/"
    "treatment_packaging/artifact_packager.py"
)
PACKAGER_SPEC = importlib.util.spec_from_file_location(
    "p02_analysis_test_packager", PACKAGER_PATH
)
assert PACKAGER_SPEC is not None and PACKAGER_SPEC.loader is not None
PACKAGER = importlib.util.module_from_spec(PACKAGER_SPEC)
sys.modules[PACKAGER_SPEC.name] = PACKAGER
PACKAGER_SPEC.loader.exec_module(PACKAGER)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bits(value: float) -> str:
    return struct.pack("<f", _f32(value)).hex()


def _target(value: float, token_id: int) -> dict:
    value = _f32(value)
    return {
        "target_token_ids": [token_id],
        "token_logprobs": [value],
        "token_logprob_float32_bits": [_bits(value)],
        "mean_logprob": value,
        "mean_logprob_float32_bits": _bits(value),
    }


def _score(margin: float, generation: str, probe: str) -> dict:
    correct = _f32(margin)
    counter = _f32(0.0)
    committed_margin = _f32(correct - counter)
    return {
        "probe": f"{probe} probe",
        "suffix_ids": [101, 102],
        "correct_text": "correct",
        "counterfactual_text": "counter",
        "correct": _target(correct, 11),
        "counterfactual": _target(counter, 12),
        "margin": committed_margin,
        "margin_float32_bits": _bits(committed_margin),
        "margin_arithmetic": "float32(correct_mean-counterfactual_mean)",
        "generation": {
            "decoded_content": generation,
            "content_ids": [ord(generation[0]) if generation else 0],
            "stop_reason": "eos",
            "cap_hit": False,
        },
    }


def _pair(margin: float, generation: str) -> dict:
    return {
        "focal": _score(margin, generation, "focal"),
        "nonfocal": _score(margin / 10.0, f"non-{generation}", "nonfocal"),
    }


def synthetic_outcome(
    *, regime: str = "nf4", repeat: int = 1,
    placebo_status: str = "PLACEBO_UNAVAILABLE",
) -> dict:
    phase_scores = {
        "A_C_focal": _score(10.0, "oracle", "focal"),
        "A_W_focal": _score(-3.0, "wrong", "focal"),
        "FF_focal": _score(2.0, "fresh", "focal"),
        "A_C_nonfocal": _score(1.0, "n-oracle", "nonfocal"),
        "A_W_nonfocal": _score(-0.3, "n-wrong", "nonfocal"),
        "FF_nonfocal": _score(0.2, "n-fresh", "nonfocal"),
    }
    phase = {
        "schema": MODULE.PHASE_A_SCHEMA,
        "design_id": MODULE.DESIGN_ID,
        "case_id": MODULE.CASE_ID,
        "plans": {},
        "executions": {"C_N": {}, "W_N": {}, "F": {}},
        "forced_carrier_support": {},
        "scores": phase_scores,
        "visible_messages": {
            "A_C": [{"role": "user", "content": "correct history"}],
            "A_W": [{"role": "user", "content": "wrong history"}],
            "FF": [{"role": "user", "content": "summary"}],
        },
        "treatment_scores_present": False,
    }
    fresh = _pair(2.0, "fresh")
    ff = {
        "arm_kind": "fresh_baseline",
        "execution_kind": "zero_increment_reuse",
        "schedule": "N",
        "region": MODULE.R2,
        "cell": "FF",
        "key_source": "F",
        "value_source": "F",
        "shared_fresh_baseline": True,
        "scores": fresh,
    }
    arm_margins = {
        ("N", MODULE.R2, "FC"): (5.0, "fc"),
        ("N", MODULE.R2, "FW"): (1.0, "fw"),
        ("N", MODULE.R2, "CC"): (6.0, "cc"),
        ("N", MODULE.R2, "WW"): (0.0, "ww"),
        ("P", MODULE.R2, "FC"): (4.0, "pfc"),
        ("P", MODULE.R2, "FW"): (2.0, "pfw"),
    }
    arms = [ff]
    for schedule, region, cell in MODULE.GRAFT_SELECTORS:
        margin, generation = arm_margins[(schedule, region, cell)]
        arms.append({
            "arm_kind": "primary",
            "schedule": schedule,
            "region": region,
            "cell": cell,
            "key_source": "F" if cell.startswith("F") else cell[0],
            "value_source": cell[1],
            "scores": _pair(margin, generation),
        })
    if repeat == 1:
        placebo = {
            "arm_kind": "placebo_control",
            "schedule": "N",
            "region": MODULE.R2,
            "cell": "V_PLACEBO",
            "control_status": placebo_status,
            "diagnostics": {"status": placebo_status, "attempts": 1024},
        }
        if placebo_status == "AVAILABLE":
            placebo["scores"] = _pair(2.1, "placebo")
        arms.append(placebo)
    foundation = {
        "schema": MODULE.FOUNDATION_SCHEMA,
        "protocol_id": MODULE.PROTOCOL_ID,
        "design_id": MODULE.DESIGN_ID,
        "case_id": MODULE.CASE_ID,
        "repeat_index": repeat,
        "plans": {},
        "source_executions": {},
        "fresh_execution": {},
        "r2_boundary_execution": {},
        "fresh_scores": fresh,
        "ff_anchor": ff,
        "visible_messages": [],
        "source_plan_order": ["C_N", "W_N", "C_P", "W_P"],
        "boundary_regions": [MODULE.R2],
    }
    treatment = {
        "schema": MODULE.TREATMENT_SCHEMA,
        "protocol_id": MODULE.PROTOCOL_ID,
        "design_id": MODULE.DESIGN_ID,
        "case_id": MODULE.CASE_ID,
        "repeat_index": repeat,
        "foundation": foundation,
        "fresh_scores": fresh,
        "arms": arms,
        "arm_count": len(arms),
        "executed_graft_count": 6,
        "zero_increment_anchor_count": 1,
        "placebo_attempt_count": 1 if repeat == 1 else 0,
        "available_placebo_control_count": int(
            repeat == 1 and placebo_status == "AVAILABLE"
        ),
        "phase_a_scores_present": False,
    }
    return {
        "schema": MODULE.OUTCOME_SCHEMA,
        "protocol_id": MODULE.PROTOCOL_ID,
        **MODULE.FORMAL_FLAGS,
        "status": "COMPLETE",
        "regime": regime,
        "case_id": MODULE.CASE_ID,
        "repeat_index": repeat,
        "model": MODULE.MODEL_ID,
        "revision": MODULE.REVISION,
        "bindings": {},
        "phase_a": phase,
        "treatment": treatment,
        "durable_checkpoints": [],
    }


def _pack(run_dir: Path, name: str, value: dict) -> Path:
    source = run_dir / f".{name}.json"
    package = run_dir / f"{name}.lossless-package"
    source.write_text(json.dumps(value, sort_keys=True) + "\n")
    PACKAGER.pack(source, package)
    source.unlink()
    return package


def test_analyzes_exact_targeted_grid_and_frozen_estimands() -> None:
    result = MODULE.analyze_outcome(synthetic_outcome())
    assert result["E1_gross_compaction_damage"]["focal"]["margin"] == pytest.approx(8.0)
    assert result["E1_gross_compaction_damage"]["nonfocal"]["margin"] == pytest.approx(0.8)
    e3 = result["E3_graft_contrasts"]
    assert e3["N_R2_D_value"]["D_focal"] == pytest.approx(4.0)
    assert e3["N_R2_D_full_KV"]["D_focal"] == pytest.approx(6.0)
    secondary = result["secondary_controls"]
    assert secondary["P_R2_D_value"]["D_focal"] == pytest.approx(2.0)
    assert secondary["N_minus_P_value_shift"]["focal"] == pytest.approx(2.0)
    pattern = result["E2_literal_generated_behavior"]["focal"]
    assert pattern["literal_decoded_content_tuple"] == [
        "fresh", "fc", "fw", "cc", "ww"
    ]
    assert pattern["four_bit_change_vector"] == [True, True, True, True]
    assert len(pattern["generation_hash_tuple"]) == 5
    assert secondary["FF_fresh_serialized_identity"] is True


def test_grid_and_order_fail_closed() -> None:
    raw = synthetic_outcome()
    raw["treatment"]["arms"][1], raw["treatment"]["arms"][2] = (
        raw["treatment"]["arms"][2], raw["treatment"]["arms"][1]
    )
    with pytest.raises(MODULE.AnalysisError, match="graft order"):
        MODULE.analyze_outcome(raw)

    raw = synthetic_outcome(repeat=2)
    raw["treatment"]["arms"].append(copy.deepcopy(raw["treatment"]["arms"][1]))
    raw["treatment"]["arm_count"] += 1
    with pytest.raises(MODULE.AnalysisError, match="counts"):
        MODULE.analyze_outcome(raw)


def test_ff_requires_serialized_identity_not_merely_a_declared_label() -> None:
    raw = synthetic_outcome()
    raw["treatment"]["arms"][0]["scores"] = copy.deepcopy(
        raw["treatment"]["fresh_scores"]
    )
    raw["treatment"]["arms"][0]["scores"]["focal"]["generation"][
        "decoded_content"
    ] = "not-the-direct-fresh-generation"
    with pytest.raises(MODULE.AnalysisError, match="serialized FF"):
        MODULE.analyze_outcome(raw)


def test_repeat1_placebo_unavailable_is_missing_control_not_a_null() -> None:
    result = MODULE.analyze_outcome(
        synthetic_outcome(placebo_status="PLACEBO_UNAVAILABLE")
    )
    placebo = result["secondary_controls"]["placebo"]
    assert placebo["status"] == "PLACEBO_UNAVAILABLE"
    assert placebo["scores"] is None
    assert placebo["movement_from_fresh"] is None

    available = MODULE.analyze_outcome(
        synthetic_outcome(placebo_status="AVAILABLE")
    )["secondary_controls"]["placebo"]
    assert available["status"] == "AVAILABLE"
    assert available["scores"] is not None


def test_lossless_package_reconstruction_and_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    package = _pack(tmp_path, "synthetic-outcome", synthetic_outcome())
    rows = MODULE.package_documents(tmp_path)
    assert len(rows) == 1
    assert rows[0]["raw"]["treatment"]["executed_graft_count"] == 6

    manifest = json.loads((package / "manifest.json").read_text())
    chunk = package / manifest["chunks"][0]["name"]
    wrapper = json.loads(chunk.read_text())
    payload = wrapper["payload"]
    wrapper["payload"] = ("A" if payload[0] != "A" else "B") + payload[1:]
    chunk.write_text(json.dumps(wrapper, sort_keys=True) + "\n")
    with pytest.raises(MODULE.AnalysisError, match="chunk file differs"):
        MODULE.package_documents(tmp_path)


def test_compact_and_render_are_recomputed_exactly_from_raw(
    tmp_path: Path,
) -> None:
    raw = synthetic_outcome()
    binding = {"original_sha256": "a" * 64}
    document = dict(raw)
    document["raw_package_binding"] = binding
    compact = tmp_path / "compact.json"
    render = tmp_path / "render.md"
    compact.write_text(json.dumps(
        MODULE.p02_runner.compact_outcome(document), sort_keys=True
    ) + "\n")
    render.write_text(MODULE.p02_runner.render_outcome_ledger(document))
    MODULE._verify_derived(raw, binding, compact, render)

    changed = json.loads(compact.read_text())
    changed["status"] = "TAMPERED"
    compact.write_text(json.dumps(changed) + "\n")
    with pytest.raises(MODULE.AnalysisError, match="compact content differs"):
        MODULE._verify_derived(raw, binding, compact, render)


def test_missing_matched_repeat1_is_a_hard_analysis_failure() -> None:
    with pytest.raises(MODULE.AnalysisError, match="matched repeat 1"):
        MODULE.validate_repeat2_layout(
            {("nf4", 1): "COMPLETE"},
            declared_rider_status="NOT_AUTHORIZED",
            initial_repeat2_authorized=False,
            bf16_repeat2_authorized=False,
        )


@pytest.mark.parametrize(
    ("statuses", "declared", "initial", "bf16_authorized"),
    [
        (
            {("nf4", 1): "COMPLETE", ("bf16", 1): "COMPLETE"},
            "NOT_AUTHORIZED", False, False,
        ),
        (
            {
                ("nf4", 1): "COMPLETE", ("bf16", 1): "COMPLETE",
                ("nf4", 2): "COMPLETE", ("bf16", 2): "COMPLETE",
            },
            "MATCHED_COMPLETE", True, True,
        ),
        (
            {
                ("nf4", 1): "COMPLETE", ("bf16", 1): "COMPLETE",
                ("nf4", 2): "COMPLETE", ("bf16", 2): "SKIPPED_NOT_AUTHORIZED",
            },
            "NF4_ONLY_TIMING_STOP", True, False,
        ),
        (
            {
                ("nf4", 1): "COMPLETE", ("bf16", 1): "COMPLETE",
                ("nf4", 2): "ERROR",
            },
            "AUTHORIZED_INCOMPLETE", True, False,
        ),
    ],
)
def test_repeat_rider_accepts_only_declared_preregistered_shapes(
    statuses: dict, declared: str, initial: bool, bf16_authorized: bool,
) -> None:
    assert MODULE.validate_repeat2_layout(
        statuses,
        declared_rider_status=declared,
        initial_repeat2_authorized=initial,
        bf16_repeat2_authorized=bf16_authorized,
    ) == declared


def test_repeat_rider_rejects_unauthorized_or_bf16_only_repeat2() -> None:
    base = {("nf4", 1): "COMPLETE", ("bf16", 1): "COMPLETE"}
    with pytest.raises(MODULE.AnalysisError, match="without initial"):
        MODULE.validate_repeat2_layout(
            {**base, ("nf4", 2): "COMPLETE"},
            declared_rider_status="NOT_AUTHORIZED",
            initial_repeat2_authorized=False,
            bf16_repeat2_authorized=False,
        )
    with pytest.raises(MODULE.AnalysisError, match="without its NF4"):
        MODULE.validate_repeat2_layout(
            {**base, ("bf16", 2): "COMPLETE"},
            declared_rider_status="AUTHORIZED_INCOMPLETE",
            initial_repeat2_authorized=True,
            bf16_repeat2_authorized=True,
        )


def test_repeat_guard_passes_identical_common_payloads() -> None:
    raws = {
        (regime, repeat): synthetic_outcome(regime=regime, repeat=repeat)
        for regime in MODULE.REGIMES for repeat in (1, 2)
    }
    analyses = {key: MODULE.analyze_outcome(raw) for key, raw in raws.items()}
    result = MODULE._repeat_and_cross_guard(analyses, raws)
    assert all(
        row["status"] == "PASS"
        for row in result["within_regime_repeat_stability"].values()
    )
    assert result[
        "E2_E3_blocked_by_generation_or_hash_instability"
    ] is False
    assert all(
        row["maximum_available_absolute_within_regime_repeat_delta"] == 0.0
        for row in result["cross_runtime_vs_within_repeat_guard"].values()
    )


def test_repeat_hash_instability_blocks_e2_e3_even_if_text_is_stable() -> None:
    raws = {
        (regime, repeat): synthetic_outcome(regime=regime, repeat=repeat)
        for regime in MODULE.REGIMES for repeat in (1, 2)
    }
    # Change a committed score while keeping every generated token/string fixed.
    changed = raws[("nf4", 2)]["treatment"]["arms"][1]["scores"]["focal"]
    changed["correct"] = _target(5.25, 11)
    changed["margin"] = _f32(5.25)
    changed["margin_float32_bits"] = _bits(5.25)
    analyses = {key: MODULE.analyze_outcome(raw) for key, raw in raws.items()}
    result = MODULE._repeat_and_cross_guard(analyses, raws)
    assert result["within_regime_repeat_stability"]["nf4"]["status"] == "FAIL"
    assert result["within_regime_repeat_stability"]["nf4"][
        "generation_hashes_and_text_stable"
    ] is True
    assert result[
        "E2_E3_blocked_by_generation_or_hash_instability"
    ] is True


def test_exclusive_output_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "analysis.json"
    MODULE.write_exclusive(output, {"status": "first"})
    with pytest.raises(FileExistsError):
        MODULE.write_exclusive(output, {"status": "second"})


def test_driver_version_parser_is_fail_closed() -> None:
    assert MODULE._version_tuple("580.159.04", "driver") >= (580, 65, 6)
    with pytest.raises(MODULE.AnalysisError, match="version differs"):
        MODULE._version_tuple("unknown", "driver")
