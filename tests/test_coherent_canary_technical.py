from types import SimpleNamespace
import struct

import pytest
import torch
from transformers import DynamicCache

from coherent_canary_schema import MODEL_ID, MODEL_REVISION
from coherent_canary_technical import (
    compact_messages, run_fresh_self_replacement, source_messages,
    summarize_natural_calibration,
)
from coherent_canary_tokens import build_fresh_destination_plan, build_role_native_plan
from arms_common import canonical_ids_any, render_hf


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True)


def history():
    return [
        {"role": "system", "content": "Keep a record."},
        {"role": "user", "content": "Status is green."},
        {"role": "assistant", "content": "Recorded."},
        {"role": "user", "content": "Continue neutrally."},
        {"role": "assistant", "content": "Understood."},
    ]


class FakeCacheModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    @property
    def device(self):
        return self.anchor.device

    def forward(self, input_ids, past_key_values=None, position_ids=None,
                cache_position=None, use_cache=True, logits_to_keep=1):
        del use_cache, logits_to_keep
        past_length = (0 if past_key_values is None else
                       int(past_key_values.layers[0].keys.shape[-2]))
        assert cache_position.tolist() == list(range(
            past_length, past_length + input_ids.shape[1]))
        cache = DynamicCache() if past_key_values is None else past_key_values
        token = input_ids.to(torch.float32)
        position = position_ids.to(torch.float32)
        for layer in range(2):
            keys = torch.stack((token + layer, position + layer), dim=-1).unsqueeze(1)
            values = torch.stack((token - layer, position - layer), dim=-1).unsqueeze(1)
            cache.update(keys, values, layer)
        logits = torch.zeros((1, input_ids.shape[1], 151700), dtype=torch.float32)
        return SimpleNamespace(past_key_values=cache, logits=logits)


def natural_cell(margin: float, first_token: int, *, stop="model_eos",
                 cap=False, eos=9):
    return {
        "margin": margin,
        "margin_float32_bits": struct.pack("<f", margin).hex(),
        "generation": {
            "content_ids": [first_token], "stop_reason": stop,
            "cap_hit": cap, "stop_candidate_id": eos, "eos_ids": [eos],
        },
    }


def test_source_and_compact_message_lists_match_frozen_plans(tokenizer):
    source = source_messages(history(), 3)
    compact = compact_messages(history(), 3)
    source_plan = build_role_native_plan(tokenizer, history(), middle_end_msg=3)
    fresh_plan = build_fresh_destination_plan(tokenizer, history(), middle_end_msg=3)
    assert list(canonical_ids_any(tokenizer, source, render_hf)) == source_plan.token_ids
    assert list(canonical_ids_any(tokenizer, compact, render_hf)) == fresh_plan.token_ids
    assert [row["role"] for row in compact] == [
        "system", "user", "assistant", "user", "assistant", "user", "assistant"]


def test_fresh_self_replacement_covers_k_v_and_kv_for_all_regions(tokenizer):
    plan = build_fresh_destination_plan(tokenizer, history(), middle_end_msg=3)
    result = run_fresh_self_replacement(FakeCacheModel(), plan)
    assert result["status"] == "PASS"
    assert {(row["region"], row["mode"]) for row in result["regions"]} == {
        (region, mode)
        for region in ("R1_content", "R2_boundary", "R3_anchor")
        for mode in ("K+V", "K-only", "V-only")
    }
    assert all(row["continued_snapshot_hashes"] ==
               result["direct_snapshot_hashes"] for row in result["regions"])


def test_natural_summary_passes_only_exact_five_cell_rule_and_normal_stop():
    raw = {
        "A_g": natural_cell(1.0, 1),
        "A_a": natural_cell(-1.0, 2),
        "F": natural_cell(0.0, 1),
        "T_g": natural_cell(0.6, 1),
        "T_a": natural_cell(-0.6, 2),
    }
    result = summarize_natural_calibration(
        raw, approve_id=[1], deny_id=[2], special_ids=[9])
    assert result["status"] == "PASS"
    assert result["rho_green"] == pytest.approx(0.6)
    assert result["rho_amber"] == pytest.approx(0.6)
    raw["A_g"] = natural_cell(1.0, 1, stop="max_content_tokens", cap=True)
    assert summarize_natural_calibration(
        raw, approve_id=[1], deny_id=[2], special_ids=[9])["status"] == "ADVERSE"


def test_natural_nonpositive_denominator_is_durable_adverse_not_exception():
    raw = {
        "A_g": natural_cell(1.0, 1),
        "A_a": natural_cell(-1.0, 2),
        "F": natural_cell(2.0, 1),
        "T_g": natural_cell(2.1, 1),
        "T_a": natural_cell(-0.5, 2),
    }
    result = summarize_natural_calibration(
        raw, approve_id=[1], deny_id=[2], special_ids=[9])
    assert result["status"] == "ADVERSE"
    assert result["rho_green"] is None
    assert result["checks"]["denominators_positive"] is False
