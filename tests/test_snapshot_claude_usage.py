from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.accounting.snapshot_claude_usage import (
    ClaudeSnapshotError,
    build_claude_usage_snapshot,
    write_snapshot,
)


def _assistant(message_id: str, usage: dict, *, model: str = "claude-fable-5") -> dict:
    return {
        "type": "assistant",
        "timestamp": "2026-07-12T00:00:00Z",
        "message": {"id": message_id, "model": model, "usage": usage},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_snapshot_globally_deduplicates_and_preserves_iterations(tmp_path: Path) -> None:
    partial_usage = {
        "input_tokens": 1,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 2,
        "output_tokens": 1,
        "cache_creation": {},
        "service_tier": "standard",
        "speed": "standard",
    }
    complete_usage = {
        **partial_usage,
        "output_tokens": 7,
        "iterations": [
            {
                "model": "claude-opus-4-8",
                "input_tokens": 2,
                "cache_creation_input_tokens": 5,
                "cache_read_input_tokens": 11,
                "output_tokens": 3,
                "cache_creation": {"ephemeral_5m_input_tokens": 5},
            },
            {
                "input_tokens": 4,
                "cache_creation_input_tokens": 6,
                "cache_read_input_tokens": 12,
                "output_tokens": 4,
                "cache_creation": {"ephemeral_1h_input_tokens": 6},
            },
        ],
    }
    _write_jsonl(tmp_path / "main.jsonl", [_assistant("msg-1", partial_usage)])
    _write_jsonl(
        tmp_path / "main" / "subagents" / "agent-a.jsonl",
        [_assistant("msg-1", complete_usage)],
    )

    snapshot = build_claude_usage_snapshot(tmp_path, logical_root="fixture")

    assert snapshot["coverage"]["source_file_count"] == 2
    assert snapshot["coverage"]["complete_json_object_count"] == 2
    assert snapshot["deduplication"]["assistant_usage_occurrence_count"] == 2
    assert snapshot["deduplication"]["unique_message_count"] == 1
    assert snapshot["deduplication"]["duplicate_occurrence_count"] == 1
    assert snapshot["deduplication"]["selected_message_with_iterations_count"] == 1
    assert snapshot["requests"]["request_count"] == 2
    assert snapshot["requests"]["usage"] == {
        "uncached_input_tokens": 6,
        "cache_write_5m_tokens": 5,
        "cache_write_1h_tokens": 6,
        "cache_write_unclassified_tokens": 0,
        "cache_creation_input_tokens": 11,
        "cache_read_input_tokens": 23,
        "output_tokens": 7,
        "context_input_tokens": 40,
        "total_tokens": 47,
    }
    assert [row["model"] for row in snapshot["requests"]["by_model"]] == [
        "claude-fable-5",
        "claude-opus-4-8",
    ]
    assert '"message":' not in json.dumps(snapshot)
    assert snapshot["money"]["cash_spend_usd"] is None


def test_derived_commitments_are_stable_across_recapture(tmp_path: Path) -> None:
    usage = {
        "input_tokens": 2,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 3,
        "output_tokens": 4,
        "cache_creation": {},
    }
    _write_jsonl(tmp_path / "session.jsonl", [_assistant("msg-stable", usage)])
    first = build_claude_usage_snapshot(tmp_path, logical_root="fixture")
    second = build_claude_usage_snapshot(tmp_path, logical_root="fixture")
    assert (
        first["deduplication"]["selection_commitment_sha256"]
        == second["deduplication"]["selection_commitment_sha256"]
    )
    assert (
        first["requests"]["request_commitment_sha256"]
        == second["requests"]["request_commitment_sha256"]
    )
    assert first["requests"] == second["requests"]


def test_snapshot_rejects_partial_trailing_jsonl(tmp_path: Path) -> None:
    (tmp_path / "partial.jsonl").write_text('{"type":"assistant"', encoding="utf-8")
    with pytest.raises(ClaudeSnapshotError, match="partial trailing"):
        build_claude_usage_snapshot(tmp_path, logical_root="fixture")


def test_snapshot_rejects_empty_or_usage_free_archive(tmp_path: Path) -> None:
    with pytest.raises(ClaudeSnapshotError, match="no Claude project JSONL"):
        build_claude_usage_snapshot(tmp_path, logical_root="fixture")
    _write_jsonl(tmp_path / "user-only.jsonl", [{"type": "user", "message": {}}])
    with pytest.raises(ClaudeSnapshotError, match="no assistant usage"):
        build_claude_usage_snapshot(tmp_path, logical_root="fixture")


def test_write_snapshot_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "snapshot.json"
    write_snapshot(output, {"schema": "fixture"})
    with pytest.raises(ClaudeSnapshotError, match="refusing to overwrite"):
        write_snapshot(output, {"schema": "replacement"})
