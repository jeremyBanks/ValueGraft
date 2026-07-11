import pytest

from coherent_canary_schema import MODEL_ID, MODEL_REVISION
from coherent_canary_technical import compact_messages, source_messages
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


def test_source_and_compact_message_lists_match_frozen_plans(tokenizer):
    source = source_messages(history(), 3)
    compact = compact_messages(history(), 3)
    source_plan = build_role_native_plan(tokenizer, history(), middle_end_msg=3)
    fresh_plan = build_fresh_destination_plan(tokenizer, history(), middle_end_msg=3)
    assert list(canonical_ids_any(tokenizer, source, render_hf)) == source_plan.token_ids
    assert list(canonical_ids_any(tokenizer, compact, render_hf)) == fresh_plan.token_ids
    assert [row["role"] for row in compact] == [
        "system", "user", "assistant", "user", "assistant", "user", "assistant"]
