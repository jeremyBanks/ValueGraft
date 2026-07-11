from pathlib import Path

import pytest
from transformers import AutoTokenizer

from arms_common import SUMMARY_REQUEST
from coherent_state_hf import CoherentStateError
from coherent_state_tokens import (
    generation_prefix_ids,
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
