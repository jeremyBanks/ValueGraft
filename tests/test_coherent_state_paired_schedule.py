from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from coherent_state_hf import CoherentStateError, IncrementalTrace
from coherent_state_tokens import generation_prefix_ids
import coherent_state_paired_schedule as paired


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True)


@pytest.fixture
def source_messages():
    return [
        {"role": "system", "content": "You are a precise project assistant."},
        {"role": "user", "content": "Record the blue route as selected."},
        {"role": "assistant", "content": "The blue route is selected."},
        {"role": "user", "content": "Write the compact handoff now."},
    ]


def test_p_is_derived_from_every_history_message_then_request_header(
        tok, source_messages):
    observed = paired.derive_prefix_schedules(tok, source_messages)
    assert observed.prefix_ids == generation_prefix_ids(tok, source_messages)
    assert observed.history_message_count == len(source_messages) - 1
    assert len(observed.message_start_positions) == len(source_messages) + 1
    assert len(observed.p_conceptual_widths) == len(source_messages)
    starts = observed.message_start_positions
    expected = [starts[index + 1] - starts[index]
                for index in range(len(source_messages) - 1)]
    expected.append(len(observed.prefix_ids) - starts[len(source_messages) - 1])
    assert observed.p_conceptual_widths == expected
    assert sum(observed.p_call_widths) == len(observed.prefix_ids)
    assert sum(observed.o_call_widths) == len(observed.prefix_ids)
    assert all(width <= 4096 for width in
               observed.p_call_widths + observed.o_call_widths)


def test_matched_message_widths_produce_identical_p_and_o_geometry(tok):
    correct = [
        {"role": "system", "content": "You are precise."},
        {"role": "user", "content": "Choose the blue path."},
        {"role": "assistant", "content": "The blue path is final."},
        {"role": "user", "content": "Write the handoff."},
    ]
    wrong = [
        {"role": "system", "content": "You are precise."},
        {"role": "user", "content": "Choose the green path."},
        {"role": "assistant", "content": "The green path is final."},
        {"role": "user", "content": "Write the handoff."},
    ]
    c = paired.derive_prefix_schedules(tok, correct)
    w = paired.derive_prefix_schedules(tok, wrong)
    assert c.prefix_ids != w.prefix_ids
    assert len(c.prefix_ids) == len(w.prefix_ids)
    assert c.message_start_positions == w.message_start_positions
    assert c.p_conceptual_widths == w.p_conceptual_widths
    assert c.p_call_widths == w.p_call_widths
    assert c.o_call_widths == w.o_call_widths


def test_scheduled_prefill_executes_exact_calls_positions_and_cache_positions(
        monkeypatch):
    calls = []

    def fake_prefill(_model, input_ids, past=None, position_ids=None,
                     cache_position=None, **_kwargs):
        calls.append({
            "ids": input_ids.tolist()[0],
            "past": past,
            "positions": position_ids.tolist()[0],
            "cache_positions": cache_position.tolist(),
        })
        return f"cache-{len(calls)}", torch.tensor([[float(len(calls))]])

    monkeypatch.setattr(paired, "prefill", fake_prefill)
    model = SimpleNamespace(device=torch.device("cpu"))
    cache, logits = paired.scheduled_prefill(model, list(range(9)), [2, 3, 4])
    assert cache == "cache-3"
    assert logits.item() == 3
    assert [row["ids"] for row in calls] == [[0, 1], [2, 3, 4], [5, 6, 7, 8]]
    assert [row["past"] for row in calls] == [None, "cache-1", "cache-2"]
    assert [row["positions"] for row in calls] == [
        [0, 1], [2, 3, 4], [5, 6, 7, 8]]
    assert [row["cache_positions"] for row in calls] == [
        [0, 1], [2, 3, 4], [5, 6, 7, 8]]


@pytest.mark.parametrize("widths", [[], [2, 2], [1, 0, 4], [5000]])
def test_scheduled_prefill_rejects_incomplete_or_invalid_calls(widths):
    model = SimpleNamespace(device=torch.device("cpu"))
    with pytest.raises(CoherentStateError, match="do not cover"):
        paired.scheduled_prefill(model, list(range(5)), widths)


def test_unknown_schedule_fails_closed(tok, source_messages):
    schedules = paired.derive_prefix_schedules(tok, source_messages)
    with pytest.raises(CoherentStateError, match="unknown source schedule"):
        schedules.widths("coarse_message_block")


def test_generated_and_forced_capture_record_the_executed_schedule(
        tok, source_messages, monkeypatch):
    dummy_cache = object()
    monkeypatch.setattr(
        paired, "scheduled_prefill",
        lambda *_args, **_kwargs: (dummy_cache, torch.zeros(1, 32)))
    rows = [(torch.zeros(1, 2, 2, 4), torch.ones(1, 2, 2, 4))]
    monkeypatch.setattr(paired, "extract_summary_rows",
                        lambda *_args, **_kwargs: rows)

    generated_trace = IncrementalTrace(
        token_ids=[7, 8], token_logprobs=[-1.0, -2.0],
        start_position=len(generation_prefix_ids(tok, source_messages)),
        end_position=len(generation_prefix_ids(tok, source_messages)) + 2,
        ended_on_eos=True)
    monkeypatch.setattr(
        paired, "generate_greedy_incremental",
        lambda *_args, **_kwargs: (dummy_cache, torch.zeros(1, 32),
                                   generated_trace))
    forced_trace = IncrementalTrace(
        token_ids=[7, 8], token_logprobs=[-1.0, -2.0],
        start_position=generated_trace.start_position,
        end_position=generated_trace.end_position,
        ended_on_eos=False)
    monkeypatch.setattr(
        paired, "append_ids_stepwise",
        lambda *_args, **_kwargs: (dummy_cache, torch.zeros(1, 32), forced_trace))

    class Model:
        device = torch.device("cpu")
        config = SimpleNamespace(eos_token_id=999)

    generated, gs = paired.capture_generated_summary_scheduled(
        Model(), tok, source_messages, schedule=paired.P_SCHEDULE)
    forced, fs = paired.capture_forced_summary_scheduled(
        Model(), tok, source_messages, [7, 8], schedule=paired.O_SCHEDULE,
        source_kind="correct_forced")
    assert generated.summary_ids == forced.summary_ids == [7, 8]
    assert generated.trace["prefix_schedule"]["schedule"] == paired.P_SCHEDULE
    assert forced.trace["prefix_schedule"]["schedule"] == paired.O_SCHEDULE
    assert generated.trace["prefix_schedule"]["executed_call_widths"] == \
        gs.p_call_widths
    assert forced.trace["prefix_schedule"]["executed_call_widths"] == \
        fs.o_call_widths
    assert generated.source_kind == "generated_incremental__turn_aligned_replay"
    assert forced.source_kind == "correct_forced__ordinary_4096"
