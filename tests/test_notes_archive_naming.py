from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from notes_archive_naming import (  # noqa: E402
    archive_counter,
    archive_day_start,
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


def test_archive_day_start_continues_across_days_and_resets_before_99() -> None:
    assert archive_day_start(1, 3) == 1
    assert archive_day_start(8, 2) == 8
    assert archive_day_start(95, 6) == 5
    assert archive_day_start(100, 1) == 10


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


def test_normalizer_excludes_daily_meta_summary(tmp_path: Path) -> None:
    normalizer = load_script(ROOT / "scripts" / "normalize_notes_archive_names.py", "normalizer_daily_meta_test")
    notes = tmp_path / "notes"
    notes.mkdir()
    daily = notes / "20260708.md"
    monthly = notes / "202607.md"
    yearly = notes / "2026.md"
    overall = notes / "README.md"
    ordinary = notes / "20260708010000-ordinary-note.md"
    daily.write_text("daily\n", encoding="utf-8")
    monthly.write_text("monthly\n", encoding="utf-8")
    yearly.write_text("yearly\n", encoding="utf-8")
    overall.write_text("summary\n", encoding="utf-8")
    ordinary.write_text("ordinary\n", encoding="utf-8")

    assert normalizer.archive_files(notes) == [ordinary]
    assert normalizer.plan_renames([daily, monthly, yearly, overall], tmp_path) == []


def test_normalizer_carries_indexes_across_days(tmp_path: Path) -> None:
    normalizer = load_script(ROOT / "scripts" / "normalize_notes_archive_names.py", "normalizer_cross_day_test")
    notes = tmp_path / "notes"
    notes.mkdir()
    first = notes / "20260704010000-first-note.md"
    second = notes / "20260704020000-second-note.md"
    third = notes / "20260705010000-third-note.md"
    for path in (first, second, third):
        path.write_text(path.stem, encoding="utf-8")

    renames = normalizer.plan_renames([first, second, third], tmp_path)
    targets = [rename.target.name for rename in renames]

    assert targets == [
        "2026070401-first-note.md",
        "2026070402-second-note.md",
        "2026070503-third-note.md",
    ]


def test_normalizer_parses_git_z_timestamps() -> None:
    normalizer = load_script(ROOT / "scripts" / "normalize_notes_archive_names.py", "normalizer_git_ts_test")
    parsed = normalizer.parse_git_timestamp("2026-07-04T20:05:15Z")
    assert parsed == datetime(2026, 7, 4, 20, 5, 15, tzinfo=timezone.utc)


def test_normalizer_uses_current_file_lifetime_add(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init"], cwd=repo, stdout=subprocess.DEVNULL)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=repo)
    notes = repo / "notes"
    notes.mkdir()
    note = notes / "2026070801-reused-name.md"
    note_rel = note.relative_to(repo).as_posix()

    def commit(message: str, iso: str) -> None:
        env = {
            **os.environ,
            "GIT_AUTHOR_DATE": iso,
            "GIT_COMMITTER_DATE": iso,
        }
        subprocess.check_call(["git", "add", "--", note_rel], cwd=repo)
        subprocess.check_call(["git", "commit", "-m", message, "--", note_rel], cwd=repo, env=env)

    note.write_text("old lifetime\n", encoding="utf-8")
    commit("old add", "2026-07-08T18:59:51Z")
    subprocess.check_call(["git", "rm", "--", note_rel], cwd=repo, stdout=subprocess.DEVNULL)
    subprocess.check_call(
        ["git", "commit", "-m", "delete old", "--", note_rel],
        cwd=repo,
        env={**os.environ, "GIT_AUTHOR_DATE": "2026-07-08T19:00:00Z", "GIT_COMMITTER_DATE": "2026-07-08T19:00:00Z"},
    )
    notes.mkdir(exist_ok=True)
    note.write_text("new lifetime\n", encoding="utf-8")
    commit("new add", "2026-07-08T14:10:08Z")

    normalizer = load_script(ROOT / "scripts" / "normalize_notes_archive_names.py", "normalizer_reused_name_test")
    timestamp = normalizer.git_creation_timestamp(note, repo)

    assert timestamp is not None
    assert timestamp.value == datetime(2026, 7, 8, 14, 10, 8, tzinfo=timezone.utc)


def test_normalizer_updates_manifest_paths_for_renamed_notes(tmp_path: Path) -> None:
    normalizer = load_script(ROOT / "scripts" / "normalize_notes_archive_names.py", "normalizer_manifest_test")
    repo = tmp_path / "repo"
    repo.mkdir()
    notes = repo / "notes"
    notes.mkdir()
    manifest = repo / "scripts" / "transcripts" / "conversation-summary-manifest.json"
    manifest.parent.mkdir(parents=True)
    old = notes / "2026070891-conversation-user-gpt55.md"
    new = notes / "2026070801-conversation-user-gpt55.md"
    old.write_text("summary\n", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "notes": [
                    {"note": old.relative_to(repo).as_posix()},
                    {"note": "notes/unchanged.md"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rename = normalizer.Rename(
        source=old,
        target=new,
        timestamp=normalizer.TimestampInfo(
            datetime(2026, 7, 8, 0, 2, 38, tzinfo=timezone.utc),
            "test",
        ),
        day_index=1,
        reasons=("test",),
    )

    updates = normalizer.manifest_path_updates([rename], repo, manifest)
    assert updates == [
        normalizer.ManifestPathUpdate(
            "notes/2026070891-conversation-user-gpt55.md",
            "notes/2026070801-conversation-user-gpt55.md",
        )
    ]


def test_normalizer_applies_rename_chain_through_existing_target(tmp_path: Path) -> None:
    normalizer = load_script(ROOT / "scripts" / "normalize_notes_archive_names.py", "normalizer_chain_test")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init"], cwd=repo, stdout=subprocess.DEVNULL)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=repo)
    notes = repo / "notes"
    notes.mkdir()
    first = notes / "2026070801-alpha.md"
    second = notes / "2026070802-beta.md"
    first.write_text("alpha\n", encoding="utf-8")
    second.write_text("beta\n", encoding="utf-8")
    subprocess.check_call(["git", "add", "--", "notes/2026070801-alpha.md", "notes/2026070802-beta.md"], cwd=repo)
    subprocess.check_call(["git", "commit", "-m", "add notes"], cwd=repo, stdout=subprocess.DEVNULL)

    renames = [
        normalizer.Rename(
            source=first,
            target=notes / "2026070802-alpha.md",
            timestamp=normalizer.TimestampInfo(datetime(2026, 7, 8, 0, 0, tzinfo=timezone.utc), "test"),
            day_index=2,
            reasons=("test",),
        ),
        normalizer.Rename(
            source=second,
            target=notes / "2026070803-beta.md",
            timestamp=normalizer.TimestampInfo(datetime(2026, 7, 8, 1, 0, tzinfo=timezone.utc), "test"),
            day_index=3,
            reasons=("test",),
        ),
    ]

    normalizer.apply_renames(renames, repo)

    assert not first.exists()
    assert not second.exists()
    assert (notes / "2026070802-alpha.md").read_text(encoding="utf-8") == "alpha\n"
    assert (notes / "2026070803-beta.md").read_text(encoding="utf-8") == "beta\n"


def test_timestamp_cache_reuses_matching_blob_entry(tmp_path: Path) -> None:
    normalizer = load_script(ROOT / "scripts" / "normalize_notes_archive_names.py", "normalizer_cache_test")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init"], cwd=repo, stdout=subprocess.DEVNULL)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=repo)
    notes = repo / "notes"
    notes.mkdir()
    note = notes / "20260704010000-cache-note.md"
    note.write_text("cache me\n", encoding="utf-8")
    rel = note.relative_to(repo).as_posix()
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2026-07-04T01:00:00Z",
        "GIT_COMMITTER_DATE": "2026-07-04T01:00:00Z",
    }
    subprocess.check_call(["git", "add", "--", rel], cwd=repo)
    subprocess.check_call(["git", "commit", "-m", "add note", "--", rel], cwd=repo, env=env)

    cache = normalizer.ArchiveTimestampCache.load(repo, repo / "cache.json")
    first = normalizer.timestamp_for(note, repo, cache)
    assert first.value == datetime(2026, 7, 4, 1, 0, tzinfo=timezone.utc)
    assert cache.save()

    reloaded = normalizer.ArchiveTimestampCache.load(repo, repo / "cache.json")
    second = normalizer.timestamp_for(note, repo, reloaded)

    assert second == first
