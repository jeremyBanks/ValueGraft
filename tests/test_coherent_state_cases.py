import json

import pytest

from coherent_state_cases import (
    CaseConstructionError,
    compacted_messages,
    correct_source_messages,
    fresh_source_messages,
    select_primary_plants,
    wrong_source_messages,
)


def _conv(cid, marker):
    return {
        "id": cid,
        "messages": [
            {"role": "system", "content": f"system-{marker}"},
            {"role": "user", "content": f"early-{marker}"},
            {"role": "assistant", "content": f"answer-{marker}"},
            {"role": "user", "content": f"tail-{marker}"},
            {"role": "assistant", "content": f"tail-answer-{marker}"},
        ],
        "sections": {"middle_end_msg": 3},
    }


def test_source_layouts_preserve_only_intended_history():
    target = _conv("c10", "target")
    donor = _conv("c02", "donor")
    correct = correct_source_messages(target, "REQUEST")
    fresh = fresh_source_messages(target, "REQUEST")
    wrong = wrong_source_messages(target, donor, "REQUEST")

    assert [m["content"] for m in correct] == [
        "system-target", "early-target", "answer-target", "tail-target",
        "tail-answer-target", "REQUEST"]
    assert [m["content"] for m in fresh] == ["system-target", "REQUEST"]
    assert [m["content"] for m in wrong] == [
        "system-target", "early-donor", "answer-donor", "tail-target",
        "tail-answer-target", "REQUEST"]


def test_compacted_layout_has_no_legacy_preamble_or_evicted_block():
    target = _conv("c10", "target")
    msgs = compacted_messages(target, "EXACT SUMMARY", "REQUEST")
    assert [m["content"] for m in msgs] == [
        "system-target", "REQUEST", "EXACT SUMMARY", "tail-target",
        "tail-answer-target"]
    assert msgs[2] == {"role": "assistant", "content": "EXACT SUMMARY"}


def test_wrong_donor_mapping_fails_closed():
    with pytest.raises(CaseConstructionError, match="frozen donor"):
        wrong_source_messages(_conv("c10", "target"), _conv("c01", "donor"))


def test_primary_selection_is_first_per_category():
    scenario = {"id": "cX", "plants": [
        {"id": "r1", "category": "referent", "probe": "p", "gold": "g"},
        {"id": "r2", "category": "referent", "probe": "p2", "gold": "g2"},
        {"id": "s1", "category": "sense", "probe": "p3", "gold": "g3"},
    ]}
    assert [p["id"] for p in select_primary_plants(scenario)] == ["r1", "s1"]


def test_role_break_fails_before_construction():
    target = _conv("c10", "target")
    target["messages"][3]["role"] = "assistant"
    with pytest.raises(CaseConstructionError, match="tail must begin"):
        compacted_messages(target, "summary")
