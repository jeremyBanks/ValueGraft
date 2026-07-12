from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import pytest

import scripts.accounting.snapshot_codex_usage as snapshot_module
from accounting.codex import (
    CodexEdge,
    CodexGraph,
    CodexThread,
    parse_frozen_rollouts,
    reconstruct_dag_owned_suffix,
    snapshot_sanitized_graph,
)
from accounting.core import FrozenJsonRecord
from scripts.accounting.snapshot_codex_usage import (
    CodexSnapshotError,
    _aggregate_owned_request_usage,
    build_codex_usage_snapshot,
    default_output_path,
    write_snapshot,
)


THREAD_ID = "019f1111-aaaa-bbbb-cccc-111111111111"
RAW_ROLLOUT_NAME = "2026/07/12/private-rollout-019f1111.jsonl"
SECRET_RESPONSE_ID = "resp-private-identity"
SECRET_RESPONSE_TEXT = "PRIVATE CONVERSATION CONTENT MUST NOT LEAK"
BASE_MS = int(datetime(2026, 7, 12, tzinfo=timezone.utc).timestamp() * 1000)


def _usage(input_tokens: int) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": input_tokens,
    }


def _response(response_id: str, text: str) -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "id": response_id,
            "content": [{"type": "output_text", "text": text}],
        },
    }


def _token(total: int, last: int) -> dict:
    return {
        "type": "event_msg",
        "timestamp": "2026-07-12T00:00:01Z",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": _usage(total),
                "last_token_usage": _usage(last),
            },
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _fixture(tmp_path: Path, *, pending: bool = False) -> tuple[Path, Path, Path]:
    database = tmp_path / "state_5.sqlite"
    sessions = tmp_path / "sessions"
    project = tmp_path / "private-project-location"
    sessions.mkdir()
    project.mkdir()
    rollout = sessions / RAW_ROLLOUT_NAME
    rows = [
        {"type": "session_meta", "payload": {"id": THREAD_ID}},
        {
            "type": "turn_context",
            "payload": {"model": "gpt-5.6-sol", "effort": "ultra"},
        },
        _response(SECRET_RESPONSE_ID, SECRET_RESPONSE_TEXT),
        _token(5, 5),
    ]
    if pending:
        rows.append(_response("resp-pending-private", "PRIVATE PENDING RESPONSE"))
    _write_jsonl(rollout, rows)

    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE threads (
            id TEXT PRIMARY KEY, rollout_path TEXT, cwd TEXT, source TEXT,
            model_provider TEXT, model TEXT, reasoning_effort TEXT,
            thread_source TEXT, history_mode TEXT, created_at INTEGER,
            updated_at INTEGER, created_at_ms INTEGER, updated_at_ms INTEGER,
            tokens_used INTEGER, title TEXT, first_user_message TEXT
        );
        CREATE TABLE thread_spawn_edges (
            parent_thread_id TEXT, child_thread_id TEXT, status TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO threads VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            THREAD_ID,
            str(rollout),
            str(project),
            '{"private_source_metadata":"do-not-serialize"}',
            "openai",
            "gpt-5.6-sol",
            "ultra",
            "subagent",
            "private-history-mode",
            BASE_MS // 1000,
            (BASE_MS + 2_000) // 1000,
            BASE_MS,
            BASE_MS + 2_000,
            5,
            "PRIVATE THREAD TITLE",
            "PRIVATE FIRST USER MESSAGE",
        ),
    )
    connection.commit()
    connection.close()
    return database, sessions, project


def _build(tmp_path: Path, *, pending: bool = False) -> tuple[dict, Path, Path, Path]:
    database, sessions, project = _fixture(tmp_path, pending=pending)
    snapshot = build_codex_usage_snapshot(
        database,
        sessions_root=sessions,
        project_root=project,
        logical_root="codex-session-rollouts",
    )
    return snapshot, database, sessions, project


def test_quiescent_snapshot_includes_reconstructed_agreement_and_only_graph_files(
    tmp_path: Path,
) -> None:
    database, sessions, project = _fixture(tmp_path)
    # A malformed non-graph rollout must not enter the source set.
    (sessions / "unrelated-private.jsonl").write_text("not json\n", encoding="utf-8")

    snapshot = build_codex_usage_snapshot(
        database,
        sessions_root=sessions,
        project_root=project,
        logical_root="codex-session-rollouts",
    )

    assert snapshot["closeout_gate"] == {
        "status": "PASS",
        "classification": "reconstructed_agreement_coverage_pass",
        "reconstructed_total_authorized": True,
        "scope": (
            "Reconstructed Codex subscription workload through the frozen rollout "
            "prefixes; not a source-exact measurement or cash-spend record."
        ),
        "blocked_reasons": [],
    }
    assert snapshot["agreement"]["status"] == "reconstructed_agreement_coverage_pass"
    assert snapshot["agreement"]["totals"]["total_tokens"] == 5
    assert snapshot["source"]["opaque_prefix_manifest"]["file_count"] == 1
    assert snapshot["source"]["complete_json_object_count"] == 4
    assert snapshot["reconstructions"]["primary"]["no_graph_collapse_upper_bound"][
        "total_tokens"
    ] == 5
    assert snapshot["money"]["cash_spend_usd"] is None
    aggregate = snapshot["owned_request_aggregate"]
    assert aggregate["request_event_count"] == 1
    assert aggregate["tokens"]["total_tokens"] == 5
    assert aggregate["by_model_effort_input_context_class"] == [
        {
            "model": "gpt-5.6-sol",
            "effort": "ultra",
            "input_context_class": "standard",
            "request_event_count": 1,
            "tokens": {
                "input_tokens": 5,
                "cached_input_tokens": 0,
                "uncached_input_tokens": 5,
                "output_tokens": 0,
                "reasoning_output_tokens": 0,
                "non_reasoning_output_tokens": 0,
                "total_tokens": 5,
            },
        }
    ]


def test_snapshot_serializes_no_private_graph_or_conversation_material(
    tmp_path: Path,
) -> None:
    database, sessions, project = _fixture(tmp_path)
    snapshot = build_codex_usage_snapshot(
        database,
        sessions_root=sessions,
        project_root=project,
        logical_root="PRIVATE-CALLER-SOURCE-LABEL",
    )
    rendered = json.dumps(snapshot, sort_keys=True)

    for private_value in (
        THREAD_ID,
        RAW_ROLLOUT_NAME,
        "private-rollout-019f1111",
        str(sessions),
        str(project),
        SECRET_RESPONSE_ID,
        SECRET_RESPONSE_TEXT,
        "PRIVATE THREAD TITLE",
        "PRIVATE FIRST USER MESSAGE",
        "private_source_metadata",
        "private-history-mode",
        "PRIVATE-CALLER-SOURCE-LABEL",
    ):
        assert private_value not in rendered
    assert snapshot["source"]["opaque_prefix_manifest"]["logical_root"] == (
        "codex-session-rollouts"
    )
    public_paths = [
        row["logical_rollout_path"]
        for row in snapshot["source"]["opaque_prefix_manifest"]["files"]
    ]
    assert len(public_paths) == 1
    assert public_paths[0].startswith("rollouts/thread_")
    assert public_paths[0].endswith(".jsonl")
    assert "mtime" not in rendered
    assert "inode" not in rendered
    assert "device" not in rendered


def test_graph_mutation_during_freeze_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, sessions, project = _fixture(tmp_path)
    first = snapshot_sanitized_graph(
        database, project_root=project, sessions_root=sessions
    )
    changed_thread = replace(first.threads[0], model="gpt-private-mutated-model")
    second = CodexGraph((changed_thread,), first.edges)
    calls = iter((first, second))
    monkeypatch.setattr(
        snapshot_module, "snapshot_sanitized_graph", lambda *args, **kwargs: next(calls)
    )

    with pytest.raises(CodexSnapshotError, match="identity or topology changed"):
        build_codex_usage_snapshot(
            database,
            sessions_root=sessions,
            project_root=project,
            logical_root="codex-session-rollouts",
        )


def test_monotonic_updated_at_advance_is_reported_but_not_a_graph_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, sessions, project = _fixture(tmp_path)
    first = snapshot_sanitized_graph(
        database, project_root=project, sessions_root=sessions
    )
    advanced_thread = replace(
        first.threads[0], updated_at_ms=first.threads[0].updated_at_ms + 1
    )
    second = CodexGraph((advanced_thread,), first.edges)
    calls = iter((first, second))
    monkeypatch.setattr(
        snapshot_module, "snapshot_sanitized_graph", lambda *args, **kwargs: next(calls)
    )

    snapshot = build_codex_usage_snapshot(
        database,
        sessions_root=sessions,
        project_root=project,
        logical_root="codex-session-rollouts",
    )
    stability = snapshot["source"]["graph_stability"]
    assert stability["status"] == "PASS"
    assert stability["updated_at_advanced_thread_count"] == 1
    assert stability["graph_sha256_before"] != stability["graph_sha256_after"]


def test_partial_trailing_rollout_line_fails_closed(tmp_path: Path) -> None:
    database, sessions, project = _fixture(tmp_path)
    rollout = sessions / RAW_ROLLOUT_NAME
    with rollout.open("ab") as handle:
        handle.write(b'{"type":"response_item"')

    with pytest.raises(CodexSnapshotError, match="partial trailing JSONL"):
        build_codex_usage_snapshot(
            database,
            sessions_root=sessions,
            project_root=project,
            logical_root="codex-session-rollouts",
        )


def test_pending_live_tail_emits_blocked_metered_boundary_snapshot(
    tmp_path: Path,
) -> None:
    snapshot, _, _, _ = _build(tmp_path, pending=True)

    gate = snapshot["closeout_gate"]
    assert gate["status"] == "BLOCKED"
    assert gate["classification"] == "reconstructed_through_last_metered_boundary"
    assert gate["reconstructed_total_authorized"] is False
    assert gate["pending_response_items"] == 1
    assert gate["pending_thread_count"] == 1
    assert snapshot["agreement"] is None
    assert snapshot["reconstructions"]["primary"]["totals"]["total_tokens"] == 5
    assert snapshot["reconstructions"]["primary"]["coverage"]["status"] == (
        "FAIL_PENDING_RESPONSES"
    )
    assert snapshot["reconstructions"]["primary"][
        "no_graph_collapse_upper_bound"
    ]["total_tokens"] == 5
    rendered = json.dumps(snapshot)
    assert "PRIVATE PENDING RESPONSE" not in rendered
    assert "resp-pending-private" not in rendered


def test_owned_request_context_aggregate_excludes_copied_graph_prefix() -> None:
    root_id = "root-private-id"
    child_id = "child-private-id"
    root = CodexThread(
        thread_id=root_id,
        logical_rollout_path="root.jsonl",
        created_at_ms=BASE_MS,
        updated_at_ms=BASE_MS + 40_000,
        source="fixture",
        model_provider="openai",
        model="gpt-5.6-terra",
        reasoning_effort="medium",
        thread_source="fixture",
        history_mode="legacy",
        project_relative_cwd=".",
    )
    child = replace(
        root,
        thread_id=child_id,
        logical_rollout_path="child.jsonl",
        created_at_ms=BASE_MS + 10_000,
        model="gpt-5.6-sol",
        reasoning_effort="ultra",
    )
    graph = CodexGraph(
        tuple(sorted((root, child), key=lambda row: row.thread_id)),
        (CodexEdge(root_id, child_id, "closed"),),
    )

    def records(path: str, thread_id: str, rows: list[dict]) -> list[FrozenJsonRecord]:
        return [
            FrozenJsonRecord(path, index, row)
            for index, row in enumerate(
                [{"type": "session_meta", "payload": {"id": thread_id}}, *rows], 1
            )
        ]

    copied_response = _response("root-first", "same copied payload")
    root_rows = [
        {
            "type": "turn_context",
            "payload": {"model": "gpt-5.6-terra", "effort": "medium"},
        },
        copied_response,
        _token(7, 7),
        _response("root-second", "root standard threshold request"),
        {
            **_token(272_007, 272_000),
            "timestamp": "2026-07-12T00:00:20Z",
        },
    ]
    child_rows = [
        {
            "type": "turn_context",
            "payload": {"model": "gpt-5.6-terra", "effort": "medium"},
        },
        _response("child-copy-rewritten-id", "same copied payload"),
        _token(7, 7),
        {
            "type": "turn_context",
            "payload": {"model": "gpt-5.6-sol", "effort": "ultra"},
        },
        _response("child-owned", "child long request"),
        {
            **_token(272_008, 272_001),
            "timestamp": "2026-07-12T00:00:30Z",
        },
    ]
    corpus = parse_frozen_rollouts(
        [
            *records("root.jsonl", root_id, root_rows),
            *records("child.jsonl", child_id, child_rows),
        ],
        graph,
    )
    reconstruction = reconstruct_dag_owned_suffix(corpus)
    aggregate = _aggregate_owned_request_usage(corpus, reconstruction)

    assert reconstruction.thread_ownership[0].copied_prefix_events in {0, 1}
    assert sum(row.copied_prefix_events for row in reconstruction.thread_ownership) == 1
    assert aggregate["request_event_count"] == 3
    assert aggregate["tokens"]["input_tokens"] == 544_008
    rows = aggregate["by_model_effort_input_context_class"]
    assert [
        (
            row["model"],
            row["effort"],
            row["input_context_class"],
            row["request_event_count"],
            row["tokens"]["input_tokens"],
        )
        for row in rows
    ] == [
        ("gpt-5.6-sol", "ultra", "long", 1, 272_001),
        ("gpt-5.6-terra", "medium", "standard", 2, 272_007),
    ]


def test_write_refuses_overwrite_and_default_names_are_unique(tmp_path: Path) -> None:
    output = tmp_path / "snapshot.json"
    write_snapshot(output, {"schema": "fixture"})
    with pytest.raises(CodexSnapshotError, match="refusing to overwrite"):
        write_snapshot(output, {"schema": "replacement"})

    first = default_output_path(tmp_path)
    second = default_output_path(tmp_path)
    assert first != second
    assert first.name.startswith("codex-project-usage_codex-cli_")
    assert first.suffix == ".json"
