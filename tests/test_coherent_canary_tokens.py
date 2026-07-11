import pytest

from coherent_canary_schema import ENGINEERED_CARRIER_CONTENT, require_matching_geometry
from coherent_canary_tokens import build_role_native_plan


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


def test_role_native_plan_covers_qwen_stream_and_nested_regions(tokenizer):
    plan = build_role_native_plan(tokenizer, _history("green"))
    assert plan.events[0].token_start == 0
    assert plan.events[-1].token_end == len(plan.token_ids)
    assert all(event.width == 1 for event in plan.events if event.kind == "q1")
    assert plan.regions.content_end - plan.regions.content_start == len(
        tokenizer.encode(ENGINEERED_CARRIER_CONTENT, add_special_tokens=False))
    assert plan.regions.content_end < plan.regions.close_end < plan.regions.anchor_end


def test_equal_width_counterfactual_has_identical_role_native_geometry(tokenizer):
    correct = build_role_native_plan(tokenizer, _history("green"))
    wrong = build_role_native_plan(tokenizer, _history("amber"))
    require_matching_geometry(correct, wrong)
    assert correct.token_ids != wrong.token_ids
