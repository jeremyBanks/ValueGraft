import copy

import pytest

from coherent_canary_schema import CASE_SCHEMA, DESIGN_ID, MODEL_ID, MODEL_REVISION, CanarySchemaError
from coherent_canary_stimuli import validate_case


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    except OSError:
        pytest.skip("pinned production tokenizer is not cached")


def _case():
    correct = [
        {"role": "system", "content": "Keep a careful decision record."},
        {"role": "user", "content": "If the marker is green, choose north; otherwise choose south. The unchanged desk code is cedar."},
        {"role": "assistant", "content": "I will apply that rule when the marker is fixed."},
        {"role": "user", "content": "The marker is green. Record the result now."},
        {"role": "assistant", "content": "The rule and desk code are recorded."},
        {"role": "user", "content": "Now discuss only unrelated packing steps."},
        {"role": "assistant", "content": "We should label the crate and verify its seal."},
    ]
    wrong = copy.deepcopy(correct)
    wrong[3]["content"] = "The marker is amber. Record the result now."
    return {
        "schema": CASE_SCHEMA,
        "design_id": DESIGN_ID,
        "case_id": "e01",
        "status": "DRAFT_UNREVIEWED",
        "execution_ready": False,
        "stratum": "engineered",
        "length_band": "short",
        "domain": "test",
        "title": "test",
        "authoring_provenance": {"author": "test"},
        "middle_end_msg": 5,
        "variants": {"correct": {"messages": correct}, "wrong_focal": {"messages": wrong}},
        "focal": {
            "plant_id": "e01-focal", "category": "derived_decision",
            "rule_or_relation": "marker rule", "changed_input": "marker",
            "probe": "Which route?", "correct_target": "north",
            "counterfactual_target": "south", "establishing_message_indices": [1, 3],
            "downstream_reference_indices": [4], "why_derived": "requires rule application",
            "why_counterfactual_reverses": "amber selects the other branch",
        },
        "nonfocal_control": {
            "plant_id": "e01-control", "category": "unchanged_control",
            "probe": "What desk code?", "target": "cedar", "countertarget": "maple",
            "establishing_message_indices": [1],
            "why_independent_of_focal": "desk code does not affect route",
        },
        "changed_message_allowlist": [3],
        "tokenizer_binding": {"model": MODEL_ID, "revision": MODEL_REVISION},
    }


def test_valid_matched_case_builds_independent_evidence(tokenizer):
    result = validate_case(tokenizer, _case())
    assert result["status"] == "MECHANICAL_DRAFT_PASS"
    assert result["execution_ready"] is False
    assert result["pair"]["role_native_geometry_identical"] is True


def test_nonallowlisted_change_fails(tokenizer):
    case = _case()
    case["variants"]["wrong_focal"]["messages"][5]["content"] = "Changed tail."
    with pytest.raises(CanarySchemaError):
        validate_case(tokenizer, case)


def test_control_chain_may_not_overlap_focal_change(tokenizer):
    case = _case()
    case["nonfocal_control"]["establishing_message_indices"] = [4]
    case["changed_message_allowlist"] = [3, 4]
    case["variants"]["wrong_focal"]["messages"][4]["content"] = "Different record."
    with pytest.raises(CanarySchemaError):
        validate_case(tokenizer, case)
