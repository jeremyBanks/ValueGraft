from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache

import powered_v13_ladder as ladder
from powered_v13_schema import MODEL_ID, MODEL_REVISION
from powered_v13_technical import build_technical_case


class FakeCacheModel(torch.nn.Module):
    def __init__(self, *, nondeterministic=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.nondeterministic = nondeterministic
        self.calls = 0

    @property
    def device(self):
        return self.anchor.device

    def forward(self, input_ids, past_key_values=None, position_ids=None,
                cache_position=None, use_cache=True, logits_to_keep=1):
        del use_cache, logits_to_keep
        self.calls += 1
        past_length = (0 if past_key_values is None else
                       int(past_key_values.layers[0].keys.shape[-2]))
        assert cache_position.tolist() == list(range(
            past_length, past_length + input_ids.shape[1]))
        cache = DynamicCache() if past_key_values is None else past_key_values
        token = input_ids.to(torch.float32)
        position = position_ids.to(torch.float32)
        for layer in range(2):
            keys = torch.stack(
                (token + layer, position + layer), dim=-1).unsqueeze(1)
            values = torch.stack(
                (token - layer, position - layer), dim=-1).unsqueeze(1)
            cache.update(keys, values, layer)
        logits = torch.zeros((1, input_ids.shape[1], 151700), dtype=torch.float32)
        pivot = int((input_ids[0, -1] + position_ids[0, -1]).item())
        logits[0, -1, pivot % logits.shape[-1]] = 3.0
        if self.nondeterministic:
            logits[0, -1, 0] = float(self.calls)
        return SimpleNamespace(past_key_values=cache, logits=logits)


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    except OSError:
        pytest.skip("pinned production tokenizer is not cached")


def _plan(tokenizer):
    return build_technical_case(
        tokenizer, Path("."), "technical_e01")["plan_objects"]["F"]


def test_same_schedule_l0_l1_l3_and_three_self_replacements_pass(tokenizer):
    evidence = ladder.run_stage_t_ladder(FakeCacheModel(), _plan(tokenizer))
    assert evidence["status"] == "PASS"
    assert evidence["schedule"] == "role_native_q1_replay"
    assert evidence["different_call_decomposition_compared"] is False
    assert set(evidence["l3_fresh_self_replacement"]) == {
        "K_ONLY", "V_ONLY", "K_AND_V",
    }
    for row in evidence["l3_fresh_self_replacement"].values():
        assert row["replacement"]["row_count"] == 74


def test_l0_rejects_nondeterministic_logits(tokenizer):
    with pytest.raises(ladder.V13LadderError, match="L0 repeat"):
        ladder.run_stage_t_ladder(
            FakeCacheModel(nondeterministic=True), _plan(tokenizer))


def test_l1_rejects_changed_null_boundary(tokenizer, monkeypatch):
    original = ladder._null_split_reconcat

    def corrupt(snapshot, split_at):
        result = original(snapshot, split_at)
        result[0][0][..., 0, 0] += 1
        return result

    monkeypatch.setattr(ladder, "_null_split_reconcat", corrupt)
    with pytest.raises(ladder.V13LadderError, match="L1 null boundary"):
        ladder.run_stage_t_ladder(FakeCacheModel(), _plan(tokenizer))


def test_ladder_source_never_compares_monolithic_with_q1_calls():
    source = Path("src/powered_v13_ladder.py").read_text()
    assert "execute_prefix_block" not in source
    assert "different_call_decomposition_compared\": False" in source
