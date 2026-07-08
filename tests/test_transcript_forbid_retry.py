from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
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


def test_new_note_prefix_reuses_deleted_same_day_gap(tmp_path: Path) -> None:
    mod = load_update_module()
    early = tmp_path / "2026070401-existing.md"
    later = tmp_path / "2026070403-later.md"
    early.write_text("early", encoding="utf-8")
    later.write_text("later", encoding="utf-8")
    fake_timestamps = {
        early: datetime(2026, 7, 4, 20, 5, tzinfo=timezone.utc),
        later: datetime(2026, 7, 4, 22, 19, tzinfo=timezone.utc),
    }
    original_archive_note_files = mod.archive_note_files
    original_archive_timestamp = mod.archive_timestamp
    try:
        mod.archive_note_files = lambda _notes_dir: [early, later]
        mod.archive_timestamp = lambda path, _root: fake_timestamps[path]

        prefix = mod.compact_prefix_for_new_note(
            tmp_path,
            tmp_path,
            datetime(2026, 7, 4, 20, 9, tzinfo=timezone.utc),
        )
    finally:
        mod.archive_note_files = original_archive_note_files
        mod.archive_timestamp = original_archive_timestamp

    assert prefix == "2026070402"
