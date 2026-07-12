from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json


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
    assert "underlying referent" in prompt
    assert "not a word-ban or synonym substitution exercise" in prompt
    assert "narrative clues" in prompt
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


def test_new_note_prefix_carries_suffix_from_prior_days(tmp_path: Path) -> None:
    mod = load_update_module()
    first = tmp_path / "2026070401-first.md"
    second = tmp_path / "2026070402-second.md"
    third = tmp_path / "2026070504-later.md"
    for path in (first, second, third):
        path.write_text(path.stem, encoding="utf-8")
    fake_timestamps = {
        first: datetime(2026, 7, 4, 10, 0, tzinfo=timezone.utc),
        second: datetime(2026, 7, 4, 11, 0, tzinfo=timezone.utc),
        third: datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc),
    }
    original_archive_note_files = mod.archive_note_files
    original_archive_timestamp = mod.archive_timestamp
    try:
        mod.archive_note_files = lambda _notes_dir: [first, second, third]
        mod.archive_timestamp = lambda path, _root: fake_timestamps[path]

        prefix = mod.compact_prefix_for_new_note(
            tmp_path,
            tmp_path,
            datetime(2026, 7, 5, 9, 0, tzinfo=timezone.utc),
        )
    finally:
        mod.archive_note_files = original_archive_note_files
        mod.archive_timestamp = original_archive_timestamp

    assert prefix == "2026070503"


def test_provisional_note_path_uses_full_utc_timestamp(tmp_path: Path) -> None:
    mod = load_update_module()
    message = make_message(
        mod,
        "codex",
        "2026-07-06",
        3,
        "2026-07-06T05:01:24.711000Z",
        "hello",
    )
    first_ts = datetime.fromisoformat("2026-07-06T05:01:24.711000+00:00")

    path = mod.provisional_note_path_for_messages(tmp_path, [message], first_ts)

    assert path.name == "20260706050124-conversation-user.md"


def make_message(
    mod,
    platform: str,
    date: str,
    sequence: int,
    timestamp: str,
    text: str,
    message_index: int = 1,
):
    return mod.MessageRecord(
        platform=platform,
        date=date,
        sequence=sequence,
        message_index=message_index,
        timestamp=timestamp,
        role="user",
        heading_metadata="",
        text=text,
        source_line=1,
    )


def test_insert_participants_block_preserves_plain_following_paragraph() -> None:
    mod = load_update_module()
    messages = [
        make_message(mod, "codex", "2026-07-09", 1, "2026-07-09T00:00:00Z", "request"),
        mod.MessageRecord(
            platform="codex",
            date="2026-07-09",
            sequence=1,
            message_index=2,
            timestamp="2026-07-09T00:01:00Z",
            role="assistant",
            heading_metadata="  [model=gpt-5.5; effort=xhigh]",
            text="answer",
            source_line=2,
        ),
    ]
    summary = """_Opening paragraph._

**Participants:** User and old-model.

**Participants.** User and a model-written duplicate.

This plain paragraph used to be accidentally deleted.

**Handoff State.** Keep this too.
"""

    updated = mod.insert_participants_block(summary, messages)

    assert "**Participants:** User and gpt-5.5-xhigh." in updated
    assert "old-model" not in updated
    assert "model-written duplicate" not in updated
    assert "This plain paragraph used to be accidentally deleted." in updated
    assert "**Handoff State.** Keep this too." in updated


def test_model_roster_filename_and_conversation_source_footer_are_separate() -> None:
    mod = load_update_module()
    messages = [
        make_message(mod, "codex", "2026-07-10", 1, "2026-07-10T00:00:00Z", "request"),
        mod.MessageRecord(
            platform="codex",
            date="2026-07-10",
            sequence=1,
            message_index=2,
            timestamp="2026-07-10T00:01:00Z",
            role="assistant",
            heading_metadata="  [model=gpt-5.6-sol; effort=xhigh; subagent=/root/methodology_audit]",
            text="Detailed assessment.",
            source_line=2,
            source_id="019f4f2d-1b08-7b61-aa7c-8d55f26f2f5b",
        ),
    ]
    messages[0].source_id = "019f4f15-7584-7b03-9760-138202ff7c80"

    block = mod.render_participants_block(messages)

    assert block == "**Participants:** User and gpt-5.6-sol-xhigh."

    path = mod.provisional_note_path_for_messages(
        Path("notes"),
        messages,
        datetime(2026, 7, 10, tzinfo=timezone.utc),
    )
    assert path.name == "20260710000000-conversation-user-gpt56.md"

    summary = mod.insert_conversation_sources_footer("_Summary._\n", messages)
    assert summary.endswith(
        "## Conversation sources\n\n"
        "- `019f4f15-7584-7b03-9760-138202ff7c80`\n"
        "- `019f4f2d-1b08-7b61-aa7c-8d55f26f2f5b`\n"
    )
    assert "methodology_audit" not in path.name
    assert "xhigh" not in path.name


def test_codex_discovery_includes_repo_user_sessions_only(tmp_path: Path, monkeypatch) -> None:
    mod = load_update_module()
    monkeypatch.setenv("HOME", str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    sessions = tmp_path / ".codex" / "sessions" / "2026" / "07" / "10"
    sessions.mkdir(parents=True)
    explicit = sessions / "legacy.jsonl"
    explicit.write_text("{}\n", encoding="utf-8")

    def write_session(name: str, cwd: Path, thread_source: str) -> Path:
        path = sessions / name
        row = {
            "type": "session_meta",
            "payload": {"cwd": str(cwd), "thread_source": thread_source},
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return path

    included = write_session("included.jsonl", repo, "user")
    write_session("subagent.jsonl", repo, "subagent")
    write_session("other-repo.jsonl", tmp_path / "other", "user")

    discovered = mod.discover_codex_sources(explicit, repo)

    assert discovered == sorted([explicit.resolve(), included.resolve()])


def test_build_new_ranges_respects_two_hour_coalescing_gap() -> None:
    mod = load_update_module()
    segments = {
        ("codex", "2026-07-05", 1): [
            make_message(mod, "codex", "2026-07-05", 1, "2026-07-05T23:50:00Z", "first")
        ],
        ("codex", "2026-07-06", 1): [
            make_message(mod, "codex", "2026-07-06", 1, "2026-07-06T00:30:00Z", "cross day")
        ],
        ("codex", "2026-07-06", 2): [
            make_message(mod, "codex", "2026-07-06", 2, "2026-07-06T03:00:01Z", "too late")
        ],
    }

    ranges = mod.build_new_ranges(segments, {}, 1_000_000, 2.0)

    assert [[(source.date, source.sequence) for source in shard] for shard in ranges] == [
        [("2026-07-05", 1), ("2026-07-06", 1)],
        [("2026-07-06", 2)],
    ]


def test_build_new_ranges_can_disable_gap_boundary() -> None:
    mod = load_update_module()
    segments = {
        ("codex", "2026-07-05", 1): [
            make_message(mod, "codex", "2026-07-05", 1, "2026-07-05T00:00:00Z", "first")
        ],
        ("codex", "2026-07-05", 2): [
            make_message(mod, "codex", "2026-07-05", 2, "2026-07-05T09:00:00Z", "later")
        ],
    }

    ranges = mod.build_new_ranges(segments, {}, 1_000_000, None)

    assert [[(source.date, source.sequence) for source in shard] for shard in ranges] == [
        [("2026-07-05", 1), ("2026-07-05", 2)]
    ]


def test_build_new_ranges_splits_long_raw_segment_at_largest_gap_in_window() -> None:
    mod = load_update_module()
    timestamps = [
        "2026-07-05T00:00:00Z",
        "2026-07-05T03:00:00Z",
        "2026-07-05T04:10:00Z",
        "2026-07-05T04:20:00Z",
        "2026-07-05T04:50:00Z",
        "2026-07-05T05:10:00Z",
        "2026-07-05T08:00:00Z",
    ]
    segments = {
        ("codex", "2026-07-05", 1): [
            make_message(mod, "codex", "2026-07-05", 1, timestamp, f"message {index}", index)
            for index, timestamp in enumerate(timestamps, 1)
        ]
    }

    ranges = mod.build_new_ranges(
        segments,
        {},
        1_000_000,
        2.0,
        max_note_duration_hours=6.0,
        split_window_start_hours=4.0,
        split_window_end_hours=5.0,
    )

    assert [[(source.sequence, source.first_message, source.last_message) for source in shard] for shard in ranges] == [
        [(1, 1, 4)],
        [(1, 5, 7)],
    ]


def test_duration_split_falls_back_before_max_when_preferred_window_has_no_gap() -> None:
    mod = load_update_module()
    timestamps = [
        "2026-07-05T00:00:00Z",
        "2026-07-05T02:00:00Z",
        "2026-07-05T03:00:00Z",
        "2026-07-05T05:30:00Z",
        "2026-07-05T07:00:00Z",
    ]
    messages = [
        make_message(mod, "codex", "2026-07-05", 1, timestamp, f"message {index}", index)
        for index, timestamp in enumerate(timestamps, 1)
    ]

    chunks = mod.split_messages_by_duration(messages, 6.0, 4.0, 5.0)

    assert len(chunks) == 2
    assert chunks[0].messages[-1].message_index == 3
    assert chunks[0].split_after.reason == "fallback-before-max"


def test_existing_duration_repair_absorbs_latest_continuation_before_splitting() -> None:
    mod = load_update_module()
    timestamps = [
        "2026-07-05T00:00:00Z",
        "2026-07-05T02:00:00Z",
        "2026-07-05T04:20:00Z",
        "2026-07-05T07:00:00Z",
        "2026-07-05T08:00:00Z",
    ]
    segments = {
        ("codex", "2026-07-05", 1): [
            make_message(mod, "codex", "2026-07-05", 1, timestamp, f"message {index}", index)
            for index, timestamp in enumerate(timestamps, 1)
        ]
    }
    record = mod.NoteRecord(
        note="notes/existing.md",
        source_ranges=[mod.SourceRange("codex", "2026-07-05", 1, 1, 3)],
        first_timestamp=timestamps[0],
        last_timestamp=timestamps[2],
        input_hash="input",
        summary_hash="summary",
    )

    plans = mod.plan_existing_duration_repairs([record], segments, 6.0, 4.0, 5.0)

    assert len(plans) == 1
    assert plans[0].expanded_ranges[0].last_message == 5
    assert [[message.message_index for message in chunk.messages] for chunk in plans[0].chunks] == [
        [1, 2, 3],
        [4, 5],
    ]


def test_dry_run_reports_duration_repair_from_live_continuation(capsys) -> None:
    mod = load_update_module()
    timestamps = [
        "2026-07-05T00:00:00Z",
        "2026-07-05T02:00:00Z",
        "2026-07-05T04:20:00Z",
        "2026-07-05T07:00:00Z",
        "2026-07-05T08:00:00Z",
    ]
    segments = {
        ("codex", "2026-07-05", 1): [
            make_message(mod, "codex", "2026-07-05", 1, timestamp, f"message {index}", index)
            for index, timestamp in enumerate(timestamps, 1)
        ]
    }
    record = mod.NoteRecord(
        note="notes/existing.md",
        source_ranges=[mod.SourceRange("codex", "2026-07-05", 1, 1, 3)],
        first_timestamp=timestamps[0],
        last_timestamp=timestamps[2],
        input_hash="input",
        summary_hash="summary",
    )
    args = SimpleNamespace(
        max_coalesce_gap_hours=2.0,
        max_note_duration_hours=6.0,
        split_window_start_hours=4.0,
        split_window_end_hours=5.0,
        force_small_continuations=False,
        min_continuation_messages=20,
        min_continuation_chars=8_000,
        target_chars=180_000,
    )

    mod.dry_run_update_notes([record], segments, args)

    output = capsys.readouterr().out
    assert "existing notes exceeding duration policy: 1" in output
    assert "notes/existing.md" in output
    assert "current ranges: codex 2026-07-05#1 1-5" in output
    assert "would become 2 notes" in output


def test_existing_duration_repair_does_not_reabsorb_a_shared_segment_into_earlier_note() -> None:
    mod = load_update_module()
    segments = {
        ("codex", "2026-07-05", 1): [
            make_message(
                mod,
                "codex",
                "2026-07-05",
                1,
                f"2026-07-05T0{index}:00:00Z",
                f"message {index}",
                index + 1,
            )
            for index in range(9)
        ]
    }
    records = [
        mod.NoteRecord(
            note="notes/first.md",
            source_ranges=[mod.SourceRange("codex", "2026-07-05", 1, 1, 5)],
            first_timestamp="2026-07-05T00:00:00Z",
            last_timestamp="2026-07-05T04:00:00Z",
            input_hash="first",
            summary_hash="first",
        ),
        mod.NoteRecord(
            note="notes/second.md",
            source_ranges=[mod.SourceRange("codex", "2026-07-05", 1, 6, 8)],
            first_timestamp="2026-07-05T05:00:00Z",
            last_timestamp="2026-07-05T07:00:00Z",
            input_hash="second",
            summary_hash="second",
        ),
    ]

    plans = mod.plan_existing_duration_repairs(records, segments, 6.0, 4.0, 5.0)

    assert plans == []
