from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache

from coherent_canary_runtime import (
    CanaryRuntimeError,
    append_block_to_snapshot,
    collect_fresh_region_margin_gradients,
    continue_fresh_plan,
    execute_fresh_plan,
    execute_prefix_block,
    execute_replay_plan,
    extract_rows,
    force_content_q1,
    greedy_generate_q1,
    probe_suffix_ids,
    replace_rows,
    require_generated_forced_identity,
    score_target_q1,
    snapshot_physical_length,
)
from coherent_canary_path_control import run_bidirectional_path_control
from coherent_canary_schema import (
    MODEL_ID, MODEL_REVISION, R2, CarrierRegions, ReplayEvent, ReplayPlan,
)
from coherent_canary_tokens import build_fresh_destination_plan, build_role_native_plan


class FakeCacheModel(torch.nn.Module):
    def __init__(self, vocab_size=151700):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.vocab_size = vocab_size

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
        assert position_ids.shape == input_ids.shape
        assert all(right == left + 1 for left, right in zip(
            position_ids[0].tolist(), position_ids[0, 1:].tolist()))
        cache = DynamicCache() if past_key_values is None else past_key_values
        token = input_ids.to(torch.float32)
        position = position_ids.to(torch.float32)
        for layer in range(2):
            keys = torch.stack((token + layer, position + layer), dim=-1).unsqueeze(1)
            values = torch.stack((token - layer, position - layer), dim=-1).unsqueeze(1)
            cache.update(keys, values, layer)
        q = input_ids.shape[1]
        logits = torch.zeros((1, q, self.vocab_size), dtype=torch.float32)
        for offset in range(q):
            pivot = int((input_ids[0, offset] + position_ids[0, offset]).item())
            logits[0, offset, pivot % self.vocab_size] = 3.0
            logits[0, offset, (pivot + 1) % self.vocab_size] = 1.0
        return SimpleNamespace(past_key_values=cache, logits=logits)


class ScriptedFakeCacheModel(FakeCacheModel):
    def __init__(self, candidate_by_cache_length):
        super().__init__()
        self.candidate_by_cache_length = candidate_by_cache_length

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        length = int(result.past_key_values.layers[0].keys.shape[-2])
        candidate = self.candidate_by_cache_length.get(length, 3)
        result.logits.zero_()
        result.logits[:, -1, candidate] = 5.0
        return result


class GradientFakeCacheModel(FakeCacheModel):
    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        total = sum(
            layer.keys.float().sum() + 0.5 * layer.values.float().sum()
            for layer in result.past_key_values.layers)
        result.logits[..., 1] = total
        result.logits[..., 2] = -total
        return result


class GradientBf16FakeCacheModel(FakeCacheModel):
    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        for layer in result.past_key_values.layers:
            layer.keys = layer.keys.to(torch.bfloat16)
            layer.values = layer.values.to(torch.bfloat16)
        total = sum(
            layer.keys.float().sum() + 0.5 * layer.values.float().sum()
            for layer in result.past_key_values.layers)
        result.logits[..., 1] = total * 1e-3
        result.logits[..., 2] = -total * 1e-3
        return result


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    except OSError:
        pytest.skip("pinned production tokenizer is not cached")


def history():
    return [
        {"role": "system", "content": "Keep a compact record."},
        {"role": "user", "content": "The signal is green."},
        {"role": "assistant", "content": "I recorded the signal."},
        {"role": "user", "content": "Continue with a neutral checklist."},
        {"role": "assistant", "content": "Check source and owner."},
    ]


def compact_messages():
    from coherent_canary_schema import (
        ANCHOR_ASSISTANT, ANCHOR_USER, ENGINEERED_CARRIER_CONTENT,
        ENGINEERED_CARRIER_REQUEST,
    )
    rows = history()
    return [rows[0],
            {"role": "user", "content": ENGINEERED_CARRIER_REQUEST},
            {"role": "assistant", "content": ENGINEERED_CARRIER_CONTENT},
            {"role": "user", "content": ANCHOR_USER},
            {"role": "assistant", "content": ANCHOR_ASSISTANT},
            *rows[3:]]


def test_replay_and_fresh_execute_exact_event_boundaries(tokenizer):
    model = FakeCacheModel()
    source = build_role_native_plan(tokenizer, history(), middle_end_msg=3)
    fresh = build_fresh_destination_plan(tokenizer, history(), middle_end_msg=3)
    source_result = execute_replay_plan(model, source)
    assert snapshot_physical_length(source_result.snapshot) == len(source.token_ids)
    assert len(source_result.calls) == len(source.events)
    assert [row["kind"] for row in source_result.calls] == [
        event.kind for event in source.events]
    assert all(row["physical_start"] == row["logical_start"]
               for row in source_result.calls)
    r2_physical_end = fresh.physical_regions.interval(R2)[1]
    fresh_boundary = execute_fresh_plan(model, fresh, stop_at=r2_physical_end)
    assert snapshot_physical_length(fresh_boundary.snapshot) == r2_physical_end


def test_row_replacement_and_causal_continuation(tokenizer):
    model = FakeCacheModel()
    source = build_role_native_plan(tokenizer, history(), middle_end_msg=3)
    fresh = build_fresh_destination_plan(tokenizer, history(), middle_end_msg=3)
    source_result = execute_replay_plan(model, source)
    source_start, source_end = source.regions.interval(R2)
    physical_start, physical_end = fresh.physical_regions.interval(R2)
    rows = extract_rows(source_result.snapshot, source_start, source_end)
    boundary = execute_fresh_plan(model, fresh, stop_at=physical_end)
    replaced, evidence = replace_rows(
        boundary.snapshot, rows, physical_start, use_keys=True, use_values=True)
    assert evidence["destination_end"] == physical_end
    completed = continue_fresh_plan(model, fresh, replaced, start_at=physical_end)
    assert snapshot_physical_length(completed.snapshot) == len(fresh.token_ids)


def test_stop_inside_event_fails(tokenizer):
    model = FakeCacheModel()
    source = build_role_native_plan(tokenizer, history(), middle_end_msg=3)
    first = source.events[0]
    assert first.width > 1
    with pytest.raises(CanaryRuntimeError, match="cuts inside"):
        execute_replay_plan(model, source, stop_at=first.token_end - 1)


def test_probe_suffix_and_q1_target_scoring(tokenizer):
    model = FakeCacheModel()
    fresh = build_fresh_destination_plan(tokenizer, history(), middle_end_msg=3)
    result = execute_fresh_plan(model, fresh)
    suffix = probe_suffix_ids(
        tokenizer, compact_messages(), fresh.token_ids, "Decision?")
    score = score_target_q1(
        model, result.snapshot, suffix_ids=suffix, target_ids=[1, 2],
        logical_context_end=fresh.logical_positions[-1] + 1)
    assert len(score["token_logprobs"]) == 2
    assert len(score["token_logprob_float32_bits"]) == 2
    assert score["teacher_forcing_feed_ids"] == suffix + [1]
    assert len(score["logical_feed_positions"]) == len(suffix) + 1


def test_selected_row_bound_is_asserted(tokenizer):
    model = FakeCacheModel()
    source = build_role_native_plan(tokenizer, history(), middle_end_msg=3)
    result = execute_replay_plan(model, source)
    with pytest.raises(CanaryRuntimeError, match="exceed asserted bound"):
        extract_rows(result.snapshot, 0, 3, max_rows=2)


def test_generated_forced_identity_is_bit_exact_and_eos_is_not_appended():
    plan = ReplayPlan(
        token_ids=list(range(8)),
        message_start_positions=[0],
        events=[
            ReplayEvent("prefill", "prefix", "structural", 0, 0, 1),
            ReplayEvent("q1", "content", "assistant", 0, 1, 2),
            ReplayEvent("prefill", "bridge", "structural", 0, 2, 3),
            ReplayEvent("q1", "anchor", "assistant", 0, 3, 4),
            ReplayEvent("prefill", "suffix", "structural", 0, 4, 8),
        ],
        regions=CarrierRegions(1, 2, 3, 4),
    ).validate()
    model = ScriptedFakeCacheModel({8: 7, 9: 8, 10: 9})
    generated_prefix = execute_replay_plan(model, plan)
    generated = greedy_generate_q1(
        model, generated_prefix.snapshot, generated_prefix.last_logits,
        logical_start=8, eos_ids=[9])
    forced_prefix = execute_replay_plan(model, plan)
    forced = force_content_q1(
        model, forced_prefix.snapshot, forced_prefix.last_logits,
        content_ids=generated.content_ids, logical_start=8, eos_ids=[9])
    evidence = require_generated_forced_identity(
        generated_prefix, generated, forced_prefix, forced, content_start=8)
    assert generated.content_ids == [7, 8]
    assert generated.stop_candidate_id == 9
    assert generated.snapshot[0][0].shape[-2] == 10
    assert evidence["status"] == "GENERATED_FORCED_IDENTITY_PASS"


def test_prefix_block_binds_ids_and_positions():
    model = FakeCacheModel()
    result = execute_prefix_block(model, [4, 5, 6], label="identity_prefix")
    assert result.executed_token_ids == [4, 5, 6]
    assert result.logical_positions == [0, 1, 2]
    assert result.physical_positions == [0, 1, 2]
    assert result.calls[0]["label"] == "identity_prefix"
    appended = append_block_to_snapshot(
        model, result.snapshot, [7, 8], logical_start=10, label="probe_suffix")
    assert appended.physical_positions == [3, 4]
    assert appended.logical_positions == [10, 11]
    assert appended.physical_end == 5


def test_margin_gradients_cover_every_selected_key_and_value_row(tokenizer):
    model = GradientFakeCacheModel()
    fresh = build_fresh_destination_plan(tokenizer, history(), middle_end_msg=3)
    result = collect_fresh_region_margin_gradients(
        model, fresh, region=R2, suffix_ids=[5],
        correct_id=1, counterfactual_id=2)
    assert result.physical_region == fresh.physical_regions.interval(R2)
    assert result.logical_region == fresh.source_regions.interval(R2)
    assert len(result.fresh_rows) == len(result.gradients) == 2
    for (keys, values), (key_gradient, value_gradient) in zip(
            result.fresh_rows, result.gradients):
        assert keys.shape == key_gradient.shape
        assert values.shape == value_gradient.shape
        assert torch.count_nonzero(key_gradient) == key_gradient.numel()
        assert torch.count_nonzero(value_gradient) == value_gradient.numel()


def test_bidirectional_path_control_reinserts_detached_bf16_rows(tokenizer):
    model = GradientBf16FakeCacheModel()
    fresh = build_fresh_destination_plan(tokenizer, history(), middle_end_msg=3)
    result = run_bidirectional_path_control(
        model, fresh, region=R2, suffix_ids=[5],
        correct_id=1, counterfactual_id=2)
    assert result["status"] == "PASS"
    assert result["chosen_ulp_count"] in (1, 2, 4, 8, 16, 32, 64)
    chosen = result["attempts"][-1]
    assert chosen["plus_margin_movement"] >= 1e-4
    assert chosen["minus_margin_movement"] >= 1e-4
    assert chosen["plus"]["insertion"]["use_keys"] is True
    assert chosen["plus"]["insertion"]["use_values"] is True
