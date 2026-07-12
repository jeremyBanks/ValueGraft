import pytest

from arms_common import canonical_ids_any, render_hf
from powered_v13_schema import (
    PRIMARY_ARMS,
    PRIMARY_REGION,
    R2,
    CarrierRegions,
    ReplayEvent,
    V13SchemaError,
    require_matching_geometry,
)
from powered_v13_tokens import (
    build_fresh_destination_plan,
    build_probe_generation_prefix,
    build_role_native_plan,
    build_turn_aligned_plan,
    complete_source_messages,
    probe_target_ids,
    source_noncarrier_token_count,
)


CARRIER = (
    "The earlier exchange established shared operating context and confirmed "
    "that decision criteria were recorded. This handoff preserves continuity "
    "for the next assistant while withholding the criteria themselves and "
    "every settled detail. Continue from the visible conversation, using the "
    "available context for any later request."
)


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            revision="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("pinned production tokenizer is not cached")


def _history(label):
    return [
        {"role": "system", "content": "Maintain the exact planning record."},
        {"role": "user", "content": "A policy and two distractors were reviewed."},
        {"role": "assistant", "content": "I will keep their evidence separate."},
        {"role": "user", "content": f"The decisive reading is {label}."},
        {"role": "assistant", "content": f"Recorded the {label} reading."},
        {"role": "user", "content": "The visible desk location is the north hall."},
        {"role": "assistant", "content": "The north hall remains directly recorded."},
    ]


def test_successor_names_do_not_inherit_v12_arm_grid():
    assert PRIMARY_REGION == R2
    assert PRIMARY_ARMS == ("FF", "CC", "WW", "FC", "FW", "VP")
    assert CarrierRegions(5, 7, 10, 11).validate().interval(R2) == (5, 10)
    with pytest.raises(V13SchemaError):
        ReplayEvent("q1", "bad", "assistant", 2, 4, 6).validate()


def test_dynamic_carrier_plans_have_identical_counterfactual_geometry(tokenizer):
    correct = build_role_native_plan(
        tokenizer, _history("green"), middle_end_msg=5,
        carrier_content=CARRIER)
    wrong = build_role_native_plan(
        tokenizer, _history("amber"), middle_end_msg=5,
        carrier_content=CARRIER)
    require_matching_geometry(correct, wrong)
    assert correct.token_ids != wrong.token_ids
    assert correct.events[0].token_start == 0
    assert correct.events[-1].token_end == len(correct.token_ids)
    assert all(event.width == 1 for event in correct.events
               if event.kind == "q1")
    assert correct.regions.content_end < correct.regions.anchor_prefix_end
    assert correct.regions.anchor_prefix_end < correct.regions.anchor_content_end


def test_turn_aligned_changes_only_precarrier_event_schedule(tokenizer):
    native = build_role_native_plan(
        tokenizer, _history("green"), middle_end_msg=5,
        carrier_content=CARRIER)
    turn = build_turn_aligned_plan(
        tokenizer, _history("green"), middle_end_msg=5,
        carrier_content=CARRIER)
    assert native.token_ids == turn.token_ids
    assert native.message_start_positions == turn.message_start_positions
    assert native.regions == turn.regions
    native_suffix = [event for event in native.events
                     if event.token_start >= native.regions.content_start]
    turn_suffix = [event for event in turn.events
                   if event.token_start >= turn.regions.content_start]
    assert native_suffix == turn_suffix
    assert native.events != turn.events


def test_fresh_destination_is_exact_dynamic_compact_render(tokenizer):
    history = _history("green")
    source = build_role_native_plan(
        tokenizer, history, middle_end_msg=5, carrier_content=CARRIER)
    fresh = build_fresh_destination_plan(
        tokenizer, history, middle_end_msg=5, carrier_content=CARRIER)
    complete = complete_source_messages(
        history, middle_end_msg=5, carrier_content=CARRIER)
    compact = complete[:1] + complete[5:]
    compact_ids = [int(value) for value in canonical_ids_any(
        tokenizer, compact, render_hf)]
    assert fresh.token_ids == compact_ids
    assert fresh.logical_positions == fresh.source_token_indices
    assert fresh.logical_positions != fresh.physical_positions
    assert fresh.source_regions == source.regions
    assert fresh.physical_regions.interval(R2)[1] <= len(fresh.token_ids)


def test_skeleton_metric_subtracts_only_dynamic_content(tokenizer):
    plan = build_role_native_plan(
        tokenizer, _history("green"), middle_end_msg=5,
        carrier_content=CARRIER)
    carrier_width = plan.regions.content_end - plan.regions.content_start
    assert source_noncarrier_token_count(plan) + carrier_width == len(
        plan.token_ids)


def test_probe_prefix_and_content_targets_share_exact_answer_position(tokenizer):
    complete = complete_source_messages(
        _history("green"), middle_end_msg=5, carrier_content=CARRIER)
    probe = "Which route label follows? Answer with only the label."
    prefix = build_probe_generation_prefix(tokenizer, complete, probe)
    source_ids = [int(value) for value in canonical_ids_any(
        tokenizer, complete, render_hf)]
    assert len(prefix) > len(source_ids)
    assert prefix[:len(source_ids)] == source_ids
    willow = probe_target_ids(tokenizer, complete, probe, "Willow")
    maple = probe_target_ids(tokenizer, complete, probe, "Maple")
    assert 1 <= len(willow) <= 4
    assert 1 <= len(maple) <= 4
    assert willow != maple


def test_dynamic_planner_rejects_missing_tail_or_empty_carrier(tokenizer):
    with pytest.raises(V13SchemaError, match="middle_end_msg"):
        build_role_native_plan(
            tokenizer, _history("green")[:5], middle_end_msg=5,
            carrier_content=CARRIER)
    with pytest.raises(V13SchemaError, match="carrier"):
        build_role_native_plan(
            tokenizer, _history("green"), middle_end_msg=5,
            carrier_content=" ")
