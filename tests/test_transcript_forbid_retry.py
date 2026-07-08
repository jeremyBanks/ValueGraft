from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "transcripts" / "update_conversation_notes.py"


def load_update_module():
    spec = importlib.util.spec_from_file_location("update_conversation_notes_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["update_conversation_notes_under_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_forbidden_matches_are_case_insensitive_and_deduped() -> None:
    mod = load_update_module()
    first = mod.forbidden_matches("The Angry summary mentions Tiananmen.", [r"angry", r"tiananmen"])
    second = mod.forbidden_matches("Another ANGRY version.", [r"angry", r"tiananmen"])

    merged = mod.merge_forbidden_matches(first, second)

    assert [(match.pattern, match.text.lower()) for match in merged] == [
        ("angry", "angry"),
        ("tiananmen", "tiananmen"),
    ]


def test_retry_prompt_requests_vague_language() -> None:
    mod = load_update_module()
    matches = mod.forbidden_matches("The output used angry language.", [r"angry"])

    prompt = mod.retry_prompt_for_forbidden_matches("Original prompt", matches)

    assert "Forbidden matches seen across attempts so far" in prompt
    assert "- angry" in prompt
    assert "vague, generic phrasing" in prompt
    assert "Original prompt" in prompt


def test_scrub_forbidden_lines_deletes_matching_lines() -> None:
    mod = load_update_module()
    text = "Keep this line.\nThis line sounds furious.\nAlso keep this.\n"

    scrubbed = mod.scrub_forbidden_lines(text, [r"furious"])

    assert scrubbed == "Keep this line.\nAlso keep this.\n"
