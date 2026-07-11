from __future__ import annotations

import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

import coherent_canary_case as case_module
from coherent_canary_case import (
    ARM_SOURCES, P_CELLS, REGIONS, build_case_plans, run_phase_a_case,
    run_treatment_case,
)
from coherent_canary_loader import REVISION_30B
from coherent_canary_schema import R1
from coherent_canary_controls import CanaryControlError


ROOT = Path(__file__).resolve().parents[1]
CASE_PATHS = (
    "data/coherent_canary_v12/revision2/session_d/e01.json",
    "data/coherent_canary_v12/revision2/session_d/e02.json",
    "data/coherent_canary_v12/revision2/session_e/e03.json",
    "data/coherent_canary_v12/revision4/session_h/e04.json",
    "data/coherent_canary_v12/revision2/session_f/e05.json",
    "data/coherent_canary_v12/revision2/session_f/e06.json",
)


@pytest.fixture(scope="module")
def tokenizer():
    snapshot = (Path.home() / ".cache/huggingface/hub/"
                "models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots" /
                REVISION_30B)
    return AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False)


def load_case(relative: str) -> dict:
    return json.loads((ROOT / relative).read_bytes())


def test_all_frozen_cases_build_matched_n_p_and_fresh_plans(tokenizer):
    for relative in CASE_PATHS:
        case = load_case(relative)
        plans = build_case_plans(tokenizer, case)
        assert set(plans) == {"C_N", "W_N", "C_P", "W_P", "F"}
        assert len(plans["C_N"].token_ids) == len(plans["W_N"].token_ids)
        assert len(plans["C_P"].token_ids) == len(plans["W_P"].token_ids)
        assert plans["C_N"].regions == plans["W_N"].regions
        assert plans["C_P"].regions == plans["W_P"].regions
        assert plans["F"].physical_positions == list(
            range(len(plans["F"].token_ids)))


def fake_execution(plan, *, fresh: bool):
    if fresh:
        logical = list(plan.logical_positions)
        physical = list(plan.physical_positions)
        q1 = []
        logical_end = logical[-1] + 1
    else:
        logical = list(range(len(plan.token_ids)))
        physical = list(range(len(plan.token_ids)))
        start, end = plan.regions.interval(R1)
        q1 = [{
            "physical_position": index,
            "logical_position": index,
            "token_id": int(plan.token_ids[index]),
            "logprob": -0.5,
            "logprob_float32_bits": struct.pack("<f", -0.5).hex(),
        } for index in range(start, end)]
        logical_end = len(plan.token_ids)
    return SimpleNamespace(
        calls=[{"kind": "fake"}], q1_token_logprobs=q1,
        executed_token_ids=list(plan.token_ids),
        logical_positions=logical, physical_positions=physical,
        physical_end=len(plan.token_ids), logical_end=logical_end,
        snapshot="snapshot", last_logits="logits")


def test_phase_a_contains_oracles_and_fresh_only(tokenizer, monkeypatch):
    case = load_case(CASE_PATHS[0])
    monkeypatch.setattr(
        case_module, "execute_replay_plan",
        lambda model, plan: fake_execution(plan, fresh=False))
    monkeypatch.setattr(
        case_module, "execute_fresh_plan",
        lambda model, plan: fake_execution(plan, fresh=True))
    monkeypatch.setattr(
        case_module, "snapshot_hashes", lambda snapshot: [{"layer": 0}])
    monkeypatch.setattr(
        case_module, "tensor_sha256", lambda tensor: "a" * 64)
    monkeypatch.setattr(
        case_module, "probe_record",
        lambda model, tok, snapshot, messages, context_ids, logical_end,
               probe, correct_text, wrong_text, eos_ids: {
                   "probe": probe, "correct_text": correct_text,
                   "counterfactual_text": wrong_text,
                   "generation": {"content_ids": [1]},
               })
    result = run_phase_a_case("model", tokenizer, case, eos_ids=[9])
    assert result["schema"] == case_module.PHASE_A_SCHEMA
    assert result["treatment_scores_present"] is False
    assert set(result["executions"]) == {"C_N", "W_N", "F"}
    assert set(result["scores"]) == {
        "A_C_focal", "A_W_focal", "FF_focal",
        "A_C_nonfocal", "A_W_nonfocal", "FF_nonfocal",
    }
    assert set(result["plans"]) == {"C_N", "W_N", "C_P", "W_P", "F"}
    assert result["forced_carrier_support"]["C_N"]["all_finite"] is True
    assert len(result["forced_carrier_support"]["C_N"]["token_ids"]) == 42


def test_treatment_grid_is_separate_and_complete(tokenizer, monkeypatch):
    case = load_case(CASE_PATHS[0])
    monkeypatch.setattr(
        case_module, "execute_replay_plan",
        lambda model, plan: fake_execution(plan, fresh=False))
    monkeypatch.setattr(
        case_module, "execute_fresh_plan",
        lambda model, plan, stop_at=None: fake_execution(plan, fresh=True))
    monkeypatch.setattr(
        case_module, "snapshot_hashes", lambda snapshot: [{"layer": 0}])
    monkeypatch.setattr(
        case_module, "tensor_sha256", lambda tensor: "a" * 64)
    monkeypatch.setattr(
        case_module, "probe_record",
        lambda *args, **kwargs: {"margin_float32_bits": "00000000"})
    monkeypatch.setattr(
        case_module, "_arm_record",
        lambda model, tok, source_case, plans, executions, boundaries,
               schedule, region, cell, eos_ids: {
                   "schedule": schedule, "region": region, "cell": cell,
                   "key_source": ARM_SOURCES[cell][0],
                   "value_source": ARM_SOURCES[cell][1],
                   "scores": {"focal": {}, "nonfocal": {}},
               })
    monkeypatch.setattr(
        case_module, "_placebo_arm_record",
        lambda model, tok, source_case, plans, executions, boundaries,
               region, eos_ids: {
                   "arm_kind": "placebo_control", "schedule": "N",
                   "region": region, "cell": case_module.VALUE_PLACEBO_CELL,
                   "control_status": (
                       "PLACEBO_UNAVAILABLE" if region == case_module.R2
                       else "AVAILABLE"),
               })
    result = run_treatment_case("model", tokenizer, case, eos_ids=[9])
    assert result["schema"] == case_module.TREATMENT_SCHEMA
    assert result["phase_a_scores_present"] is False
    assert result["arm_count"] == 34
    assert result["primary_arm_count"] == 31
    assert result["placebo_control_count"] == 3
    assert result["available_placebo_control_count"] == 2
    assert {(row["schedule"], row["region"], row["cell"])
            for row in result["arms"] if row["cell"] in ARM_SOURCES} == (
        {("N", region, cell) for region in REGIONS for cell in ARM_SOURCES} |
        {("P", case_module.R2, cell) for cell in P_CELLS})
    assert {(row["schedule"], row["region"], row["cell"])
            for row in result["arms"]
            if row["cell"] == case_module.VALUE_PLACEBO_CELL} == {
        ("N", region, case_module.VALUE_PLACEBO_CELL) for region in REGIONS}


def test_value_placebo_rows_are_fresh_keyed_and_compactly_audited():
    def snapshot(offset: float):
        layers = []
        for layer in range(2):
            keys = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4)
            keys = keys + layer
            values = torch.linspace(
                -1 + offset + layer, 1 + offset + layer, 16,
                dtype=torch.float32).reshape(1, 2, 2, 4)
            layers.append((keys, values))
        return layers

    fresh = snapshot(0.0)
    correct = snapshot(0.4)
    wrong = snapshot(-0.2)
    # Exercise the exact-zero branch for one layer/token row.
    correct[0][1][..., 0, :] = wrong[0][1][..., 0, :]
    rows, diagnostics = case_module._build_value_placebo_rows(
        fresh, correct, wrong, case_id="e01", destination_start=17)

    assert rows is not None
    assert diagnostics["status"] == "AVAILABLE"
    assert diagnostics["row_count"] == 4
    assert diagnostics["zero_delta_count"] == 1
    assert diagnostics["attempt_count"] == 3
    assert diagnostics["max_attempt_count_per_row"] == 1
    assert diagnostics["max_applied_relative_norm_error"] <= 0.05
    assert diagnostics["max_applied_abs_cosine"] <= 0.02
    assert len(diagnostics["canonical_diagnostics_sha256"]) == 64
    assert set(diagnostics["source_result_hashes"]) == {
        "fresh_source_rows_sha256", "correct_source_rows_sha256",
        "wrong_source_rows_sha256", "placebo_result_rows_sha256",
    }
    for (fresh_keys, fresh_values), (placebo_keys, placebo_values) in zip(
            fresh, rows):
        assert torch.equal(placebo_keys, fresh_keys)
        assert placebo_keys.data_ptr() != fresh_keys.data_ptr()
        assert not torch.equal(placebo_values, fresh_values)


def test_value_placebo_unavailability_is_a_compact_region_result(monkeypatch):
    fresh = [(torch.zeros((1, 1, 2, 4)), torch.zeros((1, 1, 2, 4)))]
    correct = [(fresh[0][0].clone(), torch.ones((1, 1, 2, 4)))]
    wrong = [(fresh[0][0].clone(), -torch.ones((1, 1, 2, 4)))]

    def unavailable(*args, **kwargs):
        raise CanaryControlError("PLACEBO_UNAVAILABLE: test construction failure")

    monkeypatch.setattr(
        case_module, "norm_matched_value_placebo_row", unavailable)
    rows, diagnostics = case_module._build_value_placebo_rows(
        fresh, correct, wrong, case_id="e01", destination_start=23)

    assert rows is None
    assert diagnostics["status"] == "PLACEBO_UNAVAILABLE"
    assert diagnostics["row_count"] == 2
    assert diagnostics["completed_row_count"] == 0
    assert diagnostics["failed_layer_index"] == 0
    assert diagnostics["failed_row_index"] == 23
    assert diagnostics["source_result_hashes"].keys() == {
        "fresh_source_rows_sha256", "correct_source_rows_sha256",
        "wrong_source_rows_sha256",
    }
