from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from notes_archive_naming import (  # noqa: E402
    archive_counter,
    compact_prefix,
    conversation_title,
    kebab_case,
    strip_known_prefix,
)


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_archive_counter_rollover() -> None:
    assert archive_counter(1) == "01"
    assert archive_counter(9) == "09"
    assert archive_counter(10) == "10"
    assert archive_counter(99) == "99"
    assert archive_counter(100) == "A0"
    assert archive_counter(101) == "A1"
    assert archive_counter(110) == "AA"
    assert archive_counter(136) == "B0"


def test_compact_prefix_uses_utc_date_and_counter() -> None:
    timestamp = datetime(2026, 7, 5, 23, 59, tzinfo=timezone.utc)
    assert compact_prefix(timestamp, 2) == "2026070502"


def test_title_prefix_stripping_and_kebab_case() -> None:
    assert strip_known_prefix("20260705020852-ValueGraft Synthesis") == "ValueGraft Synthesis"
    assert strip_known_prefix("2026070501-valuegraft-synthesis") == "valuegraft-synthesis"
    assert kebab_case("ValueGraft Synthesis!") == "valuegraft-synthesis"


def test_conversation_title_collapses_effort_and_dedupes() -> None:
    assert conversation_title(["User", "gpt-5.5-xhigh", "gpt-5.5-high"]) == "conversation-user-gpt55"
    assert conversation_title(["User", "claude-fable-5", "claude-opus-4-8"]) == (
        "conversation-user-fable5-opus48"
    )
    assert conversation_title(["User", "claude-sonnet-5"]) == "conversation-user-sonnet5"


def test_normalizer_assigns_per_day_indexes(tmp_path: Path) -> None:
    normalizer = load_script(ROOT / "scripts" / "normalize_notes_archive_names.py", "normalizer_under_test")
    notes = tmp_path / "notes"
    notes.mkdir()
    first = notes / "20260705020852-alpha-note.md"
    second = notes / "20260705191310-beta-note.md"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    renames = normalizer.plan_renames([first, second], tmp_path)
    targets = [rename.target.name for rename in renames]
    assert targets == ["2026070501-alpha-note.md", "2026070502-beta-note.md"]
