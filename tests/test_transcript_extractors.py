from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_claude_extractor_skips_visible_only_compaction_summaries(tmp_path: Path) -> None:
    mod = load_script(ROOT / "scripts" / "transcripts" / "extract_claude.py", "extract_claude_test")
    source = tmp_path / "claude.jsonl"
    write_jsonl(
        source,
        [
            {
                "type": "user",
                "timestamp": "2026-07-08T00:00:00Z",
                "isSidechain": False,
                "message": {"role": "user", "content": "Keep this user message."},
            },
            {
                "type": "user",
                "timestamp": "2026-07-08T00:01:00Z",
                "isSidechain": False,
                "isCompactSummary": True,
                "isVisibleInTranscriptOnly": True,
                "message": {
                    "role": "user",
                    "content": "This session is being continued from a previous conversation.",
                },
            },
            {
                "type": "user",
                "timestamp": "2026-07-08T00:02:00Z",
                "isSidechain": False,
                "isVisibleInTranscriptOnly": True,
                "message": {"role": "user", "content": "Visible-only transcript scaffolding."},
            },
            {
                "type": "assistant",
                "timestamp": "2026-07-08T00:03:00Z",
                "isSidechain": False,
                "message": {
                    "role": "assistant",
                    "model": "claude-fable-5",
                    "content": "Keep this assistant message.",
                },
            },
        ],
    )

    messages = mod.iter_messages(source)

    assert [message.text for message in messages] == [
        "Keep this user message.",
        "Keep this assistant message.",
    ]

    indexed_messages = mod.iter_messages(source, include_transcript_scaffolding=True)
    assert [message.transcript_scaffolding for message in indexed_messages] == [
        False,
        True,
        True,
        False,
    ]


def test_claude_extractor_rejects_standalone_summary_worker_session(tmp_path: Path) -> None:
    mod = load_script(
        ROOT / "scripts" / "transcripts" / "extract_claude.py",
        "extract_claude_worker_test",
    )
    source = tmp_path / "summary-worker.jsonl"
    write_jsonl(
        source,
        [
            {
                "type": "user",
                "timestamp": "2026-07-10T00:00:00Z",
                "message": {
                    "role": "user",
                    "content": (
                        "You are summarizing mainline project conversation for a future agent.\n\n"
                        "Transcript..."
                    ),
                },
            },
            {
                "type": "assistant",
                "timestamp": "2026-07-10T00:00:01Z",
                "message": {
                    "role": "assistant",
                    "content": "_Generated summary._",
                    "model": "claude-sonnet-5",
                },
            },
        ],
    )

    assert mod.iter_messages(source) == []


def test_claude_extractor_includes_only_final_subagent_answer(tmp_path: Path) -> None:
    mod = load_script(
        ROOT / "scripts" / "transcripts" / "extract_claude.py",
        "extract_claude_subagent_test",
    )
    source = tmp_path / "main-session.jsonl"
    write_jsonl(
        source,
        [
            {
                "type": "user",
                "timestamp": "2026-07-10T00:00:00Z",
                "message": {"role": "user", "content": "Main request."},
            },
            {
                "type": "assistant",
                "timestamp": "2026-07-10T00:00:01Z",
                "message": {"role": "assistant", "content": "Main answer."},
            },
        ],
    )
    subagents = tmp_path / source.stem / "subagents"
    subagents.mkdir(parents=True)
    write_jsonl(
        subagents / "agent-reviewer.jsonl",
        [
            {
                "type": "user",
                "isSidechain": True,
                "agentId": "reviewer",
                "timestamp": "2026-07-10T00:00:02Z",
                "message": {"role": "user", "content": "Review it."},
            },
            {
                "type": "assistant",
                "isSidechain": True,
                "agentId": "reviewer",
                "timestamp": "2026-07-10T00:00:03Z",
                "message": {"role": "assistant", "content": "Intermediate update."},
            },
            {
                "type": "assistant",
                "isSidechain": True,
                "agentId": "reviewer",
                "timestamp": "2026-07-10T00:00:04Z",
                "message": {"role": "assistant", "content": "Detailed final assessment."},
            },
        ],
    )

    messages = mod.iter_messages(source)

    assert [message.text for message in messages] == [
        "Main request.",
        "Main answer.",
        "Detailed final assessment.",
    ]
    assert messages[-1].subagent == "reviewer"
    assert [message.text for message in mod.iter_messages(source, include_subagent_finals=False)] == [
        "Main request.",
        "Main answer.",
    ]


def test_codex_extractor_skips_compaction_records(tmp_path: Path) -> None:
    mod = load_script(ROOT / "scripts" / "transcripts" / "extract_codex.py", "extract_codex_test")
    source = tmp_path / "rollout-2026-07-08T00-00-00-thread.jsonl"
    write_jsonl(
        source,
        [
            {
                "type": "session_meta",
                "timestamp": "2026-07-08T00:00:00Z",
                "payload": {"cwd": "/tmp/example", "model": "gpt-5.5"},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-07-08T00:01:00Z",
                "payload": {"type": "user_message", "message": "Keep this user message."},
            },
            {
                "type": "compacted",
                "timestamp": "2026-07-08T00:02:00Z",
                "payload": {
                    "replacement_history": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Old replaced message."}],
                        },
                        {
                            "type": "compaction",
                            "id": "cmp_test",
                            "encrypted_content": "opaque",
                        },
                    ]
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-07-08T00:03:00Z",
                "payload": {"type": "context_compacted"},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-07-08T00:04:00Z",
                "payload": {"type": "agent_message", "message": "Keep this assistant message."},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-07-08T00:05:00Z",
                "payload": {
                    "type": "user_message",
                    "isVisibleInTranscriptOnly": True,
                    "message": "Visible-only transcript scaffolding.",
                },
            },
        ],
    )

    _thread_id, _cwd, messages = mod.iter_messages(source)

    assert [message.text for message in messages] == [
        "Keep this user message.",
        "Keep this assistant message.",
    ]

    _thread_id, _cwd, indexed_messages = mod.iter_messages(source, include_transcript_scaffolding=True)
    assert [message.text for message in indexed_messages] == [
        "Keep this user message.",
        "Keep this assistant message.",
        "Visible-only transcript scaffolding.",
    ]
    assert [message.transcript_scaffolding for message in indexed_messages] == [
        False,
        False,
        True,
    ]


def test_codex_extractor_includes_only_final_subagent_message(tmp_path: Path) -> None:
    mod = load_script(
        ROOT / "scripts" / "transcripts" / "extract_codex.py",
        "extract_codex_subagent_test",
    )
    source = tmp_path / "rollout-2026-07-10T00-00-00-thread.jsonl"
    write_jsonl(
        source,
        [
            {
                "type": "session_meta",
                "timestamp": "2026-07-10T00:00:00Z",
                "payload": {"cwd": "/tmp/example", "model": "gpt-5.6-sol"},
            },
            {
                "type": "event_msg",
                "timestamp": "2026-07-10T00:00:01Z",
                "payload": {"type": "user_message", "message": "Main request."},
            },
            {
                "type": "response_item",
                "timestamp": "2026-07-10T00:00:02Z",
                "payload": {
                    "type": "agent_message",
                    "author": "/root/reviewer",
                    "recipient": "/root",
                    "content": [{"type": "text", "text": "Message Type: MESSAGE\nPayload: progress"}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-07-10T00:00:03Z",
                "payload": {
                    "type": "agent_message",
                    "author": "/root/reviewer",
                    "recipient": "/root",
                    "content": [
                        {
                            "type": "text",
                            "text": "Message Type: FINAL_ANSWER\nPayload: Detailed final assessment.",
                        }
                    ],
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-07-10T00:00:04Z",
                "payload": {"type": "agent_message", "message": "Main answer."},
            },
        ],
    )

    _thread, _cwd, messages = mod.iter_messages(source)

    assert [message.text for message in messages] == [
        "Main request.",
        "Message Type: FINAL_ANSWER\nPayload: Detailed final assessment.",
        "Main answer.",
    ]
    assert messages[1].subagent == "/root/reviewer"


def test_conversation_note_renderer_omits_scaffolding_but_keeps_ranges_available() -> None:
    mod = load_script(
        ROOT / "scripts" / "transcripts" / "update_conversation_notes.py",
        "update_conversation_notes_extractors_test",
    )
    messages = [
        mod.MessageRecord(
            platform="claude-code",
            date="2026-07-08",
            sequence=1,
            message_index=1,
            timestamp="2026-07-08T00:00:00Z",
            role="user",
            heading_metadata="",
            text="Visible user message.",
            source_line=10,
        ),
        mod.MessageRecord(
            platform="claude-code",
            date="2026-07-08",
            sequence=1,
            message_index=2,
            timestamp="2026-07-08T00:01:00Z",
            role="user",
            heading_metadata="",
            text="Compaction summary scaffolding.",
            source_line=11,
            transcript_scaffolding=True,
        ),
        mod.MessageRecord(
            platform="claude-code",
            date="2026-07-08",
            sequence=1,
            message_index=3,
            timestamp="2026-07-08T00:02:00Z",
            role="assistant",
            heading_metadata="  [model=claude-fable-5]",
            text="Visible assistant message.",
            source_line=12,
        ),
    ]

    rendered = mod.render_messages(messages)

    assert "Message 001 - user" in rendered
    assert "Message 003 - assistant" in rendered
    assert "Message 002" not in rendered
    assert "Compaction summary scaffolding" not in rendered
