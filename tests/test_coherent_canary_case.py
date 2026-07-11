from __future__ import annotations

import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

import coherent_canary_case as case_module
from coherent_canary_case import build_case_plans, run_phase_a_case
from coherent_canary_loader import REVISION_30B
from coherent_canary_schema import R1


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
