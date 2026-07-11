from pathlib import Path

import pytest
from transformers import AutoTokenizer

from arms_common import SUMMARY_REQUEST
from coherent_state_hf import CoherentStateError
from coherent_state_tokens import (
    gapped_destination_layout,
    generation_prefix_ids,
    matched_wrong_prefix_ids,
    probe_layout,
    rendered_assistant_content_ids,
    summary_destination_layout,
    teacher_forcing_feed,
)


MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


@pytest.fixture
def conv():
    return {
        "id": "c10",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Remember the blue heron."},
            {"role": "assistant", "content": "I will remember it."},
            {"role": "user", "content": "Now discuss wetlands."},
            {"role": "assistant", "content": "Wetlands are diverse."},
        ],
        "sections": {"middle_end_msg": 3},
    }


def test_summary_destination_is_exact_and_has_no_preamble(tok, conv):
    source = [conv["messages"][0], {"role": "user", "content": SUMMARY_REQUEST}]
    summary = "The conversation concerned a blue heron."
    summary_ids = rendered_assistant_content_ids(tok, source, summary)
    layout = summary_destination_layout(tok, conv, summary, summary_ids,
                                        SUMMARY_REQUEST)
    assert layout.context_ids[layout.summary_start:layout.summary_end] == summary_ids
    assert tok.decode(layout.context_ids).count("[Context note]") == 0
    assert layout.context_ids[:layout.summary_start] == layout.prefix_ids


def test_probe_target_uses_rendered_boundary_tokens(tok, conv):
    source = [conv["messages"][0], {"role": "user", "content": SUMMARY_REQUEST}]
    summary = "The conversation concerned a blue heron."
    summary_ids = rendered_assistant_content_ids(tok, source, summary)
    destination = summary_destination_layout(
        tok, conv, summary, summary_ids, SUMMARY_REQUEST)
    p = probe_layout(tok, destination.messages, destination.context_ids,
                     "Which bird?", "A blue heron.")
    assert tok.decode(p.target_ids).strip() == "A blue heron."
    assert teacher_forcing_feed(p)[-len(p.target_ids) + 1:] == p.target_ids[:-1]


def test_wrong_summary_ids_fail_exact_span(tok, conv):
    source = [conv["messages"][0], {"role": "user", "content": SUMMARY_REQUEST}]
    ids = rendered_assistant_content_ids(tok, source, "One summary.")
    with pytest.raises(CoherentStateError, match="exact summary IDs"):
        summary_destination_layout(tok, conv, "Different summary.", ids,
                                   SUMMARY_REQUEST)


def test_generation_prefix_requires_user_final(tok):
    with pytest.raises(CoherentStateError, match="end in a user"):
        generation_prefix_ids(tok, [{"role": "assistant", "content": "x"}])


def test_wrong_prefix_changes_only_exact_length_content_slots(tok, conv):
    donor = {
        "id": "c02",
        "messages": [
            {"role": "system", "content": "Donor system."},
            {"role": "user", "content": "A red kite was selected instead."},
            {"role": "assistant", "content": "The red kite selection is recorded."},
            {"role": "user", "content": "Donor tail."},
            {"role": "assistant", "content": "Donor tail response."},
        ],
        "sections": {"middle_end_msg": 3},
    }
    matched = matched_wrong_prefix_ids(tok, conv, donor, SUMMARY_REQUEST)
    assert len(matched.correct_ids) == len(matched.wrong_ids)
    assert matched.correct_ids != matched.wrong_ids
    assert all(matched.correct_ids[i] == matched.wrong_ids[i]
               for i in matched.structural_positions)
    assert len(matched.replacements) == 2
    assert all(r.end - r.start == len(r.replacement_ids)
               for r in matched.replacements)
    assert matched.correct_ids[-20:] == matched.wrong_ids[-20:]


def test_gapped_layout_anchors_request_and_summary_to_correct_source(tok, conv):
    source = [conv["messages"][0], {"role": "user", "content": SUMMARY_REQUEST}]
    summary = "The conversation concerned a blue heron."
    summary_ids = rendered_assistant_content_ids(tok, source, summary)
    correct_prefix = generation_prefix_ids(
        tok, conv["messages"] + [{"role": "user", "content": SUMMARY_REQUEST}])
    layout = gapped_destination_layout(
        tok, conv, summary, summary_ids, SUMMARY_REQUEST, correct_prefix)
    assert layout.source_summary_start == len(correct_prefix)
    assert layout.summary_position_ids[0] == len(correct_prefix)
    assert layout.request_logical_start >= layout.system_end
    assert layout.prefix_position_ids[:layout.system_end] == list(
        range(layout.system_end))
    assert layout.context_ids[layout.physical_summary_start:
                              layout.physical_summary_end] == summary_ids
    assert len(layout.context_ids) == len(layout.context_position_ids)
    assert layout.logical_next_position == layout.context_position_ids[-1] + 1


def test_gapped_layout_rejects_changed_correct_source_system(tok, conv):
    source = [conv["messages"][0], {"role": "user", "content": SUMMARY_REQUEST}]
    summary = "The conversation concerned a blue heron."
    summary_ids = rendered_assistant_content_ids(tok, source, summary)
    correct_prefix = generation_prefix_ids(
        tok, conv["messages"] + [{"role": "user", "content": SUMMARY_REQUEST}])
    baseline = gapped_destination_layout(
        tok, conv, summary, summary_ids, SUMMARY_REQUEST, correct_prefix)
    changed = list(correct_prefix)
    special = set(tok.all_special_ids)
    position = next(i for i in range(baseline.system_end)
                    if changed[i] not in special)
    changed[position] = (changed[position] + 1) % tok.vocab_size
    with pytest.raises(CoherentStateError, match="system island"):
        gapped_destination_layout(
            tok, conv, summary, summary_ids, SUMMARY_REQUEST, changed)
