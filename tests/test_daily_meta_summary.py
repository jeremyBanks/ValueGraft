from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))


def load_script():
    path = ROOT / "scripts" / "update_daily_meta_summary.py"
    spec = importlib.util.spec_from_file_location("daily_meta_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["daily_meta_under_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cap_text_keeps_short_text() -> None:
    daily = load_script()
    text = "abc" * 10

    shown, omitted = daily.cap_text(text, threshold=100, head=60, tail=20, label="ordinary")

    assert shown == text
    assert omitted == 0


def test_cap_text_marks_omitted_middle() -> None:
    daily = load_script()
    text = "a" * 10 + "b" * 50 + "c" * 10

    shown, omitted = daily.cap_text(text, threshold=20, head=10, tail=10, label="ordinary")

    assert omitted == 50
    assert shown.startswith("a" * 10)
    assert "50 characters omitted from the middle of this ordinary note" in shown
    assert shown.endswith("c" * 10)


def test_source_paths_for_day_excludes_daily_summary(tmp_path: Path) -> None:
    daily = load_script()
    notes = tmp_path / "notes"
    notes.mkdir()
    daily_summary = notes / "20260708.md"
    ordinary = notes / "2026070801-ordinary.md"
    other_day = notes / "2026070901-other.md"
    for path in (daily_summary, ordinary, other_day):
        path.write_text(path.name, encoding="utf-8")

    assert daily.source_paths_for_day(notes, "20260708") == [ordinary]


def test_source_signature_ignores_filename_order() -> None:
    daily = load_script()

    class FakeSource:
        def __init__(self, blob_id: str):
            self.blob_id = blob_id

    left = [FakeSource("bbb"), FakeSource("aaa")]
    right = [FakeSource("aaa"), FakeSource("bbb")]

    assert daily.source_signature(left) == daily.source_signature(right)


def test_daily_entry_tracks_source_paths_for_footer_staleness(tmp_path: Path) -> None:
    daily = load_script()
    repo = tmp_path / "repo"
    repo.mkdir()
    notes = repo / "notes"
    notes.mkdir()
    first = notes / "2026070801-alpha.md"
    second = notes / "2026070802-beta.md"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    sources = [
        daily.SourceNote(first, "alpha", "blob-a", "ordinary", "alpha", 0),
        daily.SourceNote(second, "beta", "blob-b", "ordinary", "beta", 0),
    ]
    entry = daily.entry_for("20260708", notes / "20260708.md", "summary", sources, repo)

    assert entry["sources"] == [
        {"note": "notes/2026070801-alpha.md", "blob_id": "blob-a"},
        {"note": "notes/2026070802-beta.md", "blob_id": "blob-b"},
    ]


def test_daily_summary_appends_standard_sources_footer(tmp_path: Path) -> None:
    from notes_rollup import with_sources_footer

    notes = tmp_path / "notes"
    notes.mkdir()
    target = notes / "20260708.md"
    first = notes / "2026070801-alpha.md"
    second = notes / "2026070802-beta.md"
    text = with_sources_footer("# Day\n\nBody\n\n## Sources\n\n- stale", target, [second, first])

    assert text == (
        "# Day\n\n"
        "Body\n\n"
        "## Sources\n\n"
        "- [2026070801-alpha.md](2026070801-alpha.md)\n"
        "- [2026070802-beta.md](2026070802-beta.md)\n"
    )
