import pytest

from coherent_canary_schema import ENGINEERED_CARRIER_CONTENT, require_matching_geometry
from coherent_canary_tokens import build_role_native_plan, build_turn_aligned_plan


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


def _history(label: str):
    return [
        {"role": "system", "content": "Keep an exact decision record."},
        {"role": "user", "content": f"The selected label is {label}."},
        {"role": "assistant", "content": f"Recorded the label {label}."},
    ]


def _history_with_tail(label: str):
    return _history(label) + [
        {"role": "user", "content": "Continue with a neutral checklist."},
        {"role": "assistant", "content": "Use source, owner, and status columns."},
    ]


def test_role_native_plan_covers_qwen_stream_and_nested_regions(tokenizer):
    plan = build_role_native_plan(tokenizer, _history("green"))
    assert plan.events[0].token_start == 0
    assert plan.events[-1].token_end == len(plan.token_ids)
    assert all(event.width == 1 for event in plan.events if event.kind == "q1")
    assert plan.regions.content_end - plan.regions.content_start == len(
        tokenizer.encode(ENGINEERED_CARRIER_CONTENT, add_special_tokens=False))
    assert (plan.regions.content_end < plan.regions.anchor_prefix_end <
            plan.regions.anchor_content_end)
    assert plan.events[0].label == "initial_generation_prefix"
    assert plan.events[0].kind == "prefill"
    assert not any(event.label in ("assistant_open", "assistant_close")
                   for event in plan.events)
    assert plan.events[-1].label == "final_assistant_close"


def test_structural_calls_match_turn_additions(tokenizer):
    plan = build_role_native_plan(tokenizer, _history("green"))
    q1_by_message = {}
    for event in plan.events:
        if event.kind == "q1":
            q1_by_message.setdefault(event.message_index, []).append(event)
    assistant_indices = sorted(q1_by_message)
    assert len(assistant_indices) == 3  # history reply, carrier, anchor reply

    initial = plan.events[0]
    assert initial.token_end == q1_by_message[assistant_indices[0]][0].token_start
    for left_index, right_index in zip(assistant_indices, assistant_indices[1:]):
        left_end = q1_by_message[left_index][-1].token_end
        right_start = q1_by_message[right_index][0].token_start
        structural = [
            event for event in plan.events
            if event.kind == "prefill" and event.token_start == left_end
        ]
        assert len(structural) == 1
        assert structural[0].label == "turn_continuation"
        assert structural[0].token_end == right_start

    # The primary R2 boundary is the END of the carrier-to-anchor structural
    # call. It includes the carrier close plus the anchor user/header, so the
    # intervention never cuts inside a model call.
    carrier_index = assistant_indices[-2]
    carrier_end = q1_by_message[carrier_index][-1].token_end
    carrier_continuation = next(
        event for event in plan.events
        if event.kind == "prefill" and event.token_start == carrier_end)
    assert carrier_continuation.token_start < plan.regions.anchor_prefix_end
    assert plan.regions.anchor_prefix_end == carrier_continuation.token_end


def test_carrier_and_anchor_are_inserted_before_retained_tail(tokenizer):
    history = _history_with_tail("green")
    plan = build_role_native_plan(tokenizer, history, middle_end_msg=3)
    assert plan.regions.anchor_content_end < len(plan.token_ids)

    rendered = tokenizer.decode(plan.token_ids)
    carrier_offset = rendered.index("The prior discussion established")
    anchor_offset = rendered.index("Acknowledged.")
    tail_offset = rendered.index("Continue with a neutral checklist.")
    assert carrier_offset < anchor_offset < tail_offset

    # R3 ends after the q=1 acknowledgment content, before the structural call
    # containing its close plus the retained-tail user/header.
    anchor_q1 = [event for event in plan.events
                 if event.kind == "q1" and event.message_index == 6]
    assert plan.regions.anchor_content_end == anchor_q1[-1].token_end
    following = next(event for event in plan.events
                     if event.token_start == plan.regions.anchor_content_end)
    assert following.kind == "prefill"
    assert following.label == "turn_continuation"


def test_equal_width_counterfactual_has_identical_role_native_geometry(tokenizer):
    correct = build_role_native_plan(tokenizer, _history("green"))
    wrong = build_role_native_plan(tokenizer, _history("amber"))
    require_matching_geometry(correct, wrong)
    assert correct.token_ids != wrong.token_ids


def test_equal_width_counterfactual_with_tail_has_identical_geometry(tokenizer):
    correct = build_role_native_plan(
        tokenizer, _history_with_tail("green"), middle_end_msg=3)
    wrong = build_role_native_plan(
        tokenizer, _history_with_tail("amber"), middle_end_msg=3)
    require_matching_geometry(correct, wrong)


def test_turn_aligned_p_is_complete_and_matches_after_carrier_start(tokenizer):
    history = _history_with_tail("green")
    n_plan = build_role_native_plan(tokenizer, history, middle_end_msg=3)
    p_plan = build_turn_aligned_plan(tokenizer, history, middle_end_msg=3)
    assert p_plan.token_ids == n_plan.token_ids
    assert p_plan.regions == n_plan.regions
    assert p_plan.events[0].label == "historical_message"
    assert [event.label for event in p_plan.events[:3]] == [
        "historical_message", "historical_message", "historical_message"]
    assert p_plan.events[3].label == "carrier_request_and_header"
    n_suffix = [event for event in n_plan.events
                if event.token_start >= n_plan.regions.content_start]
    p_suffix = [event for event in p_plan.events
                if event.token_start >= p_plan.regions.content_start]
    assert p_suffix == n_suffix
    assert [event.width for event in p_plan.events] != [
        event.width for event in n_plan.events]


def test_turn_aligned_counterfactual_geometry_matches(tokenizer):
    correct = build_turn_aligned_plan(
        tokenizer, _history_with_tail("green"), middle_end_msg=3)
    wrong = build_turn_aligned_plan(
        tokenizer, _history_with_tail("amber"), middle_end_msg=3)
    require_matching_geometry(correct, wrong)
