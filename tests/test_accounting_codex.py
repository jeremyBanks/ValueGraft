from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from accounting.codex import (
    CodexAccountingError,
    CodexAmbiguityError,
    CodexEdge,
    CodexGraph,
    CodexThread,
    TokenVector,
    assert_exact_reconstruction_agreement,
    parse_frozen_rollouts,
    reconstruct_dag_owned_suffix,
    reconstruct_epoch_deltas,
    snapshot_sanitized_graph,
)
from accounting.core import FrozenJsonRecord


def _thread(thread_id: str, created: int) -> CodexThread:
    return CodexThread(
        thread_id=thread_id,
        logical_rollout_path=f"{thread_id}.jsonl",
        created_at_ms=created,
        updated_at_ms=created + 1,
        source="test",
        model_provider="openai",
        model="gpt-test",
        reasoning_effort="xhigh",
        thread_source="fixture",
        history_mode="legacy",
        project_relative_cwd=".",
    )


def _graph(
    created: dict[str, int], edges: list[tuple[str, str]] | None = None
) -> CodexGraph:
    threads = tuple(
        sorted(
            (_thread(thread_id, time) for thread_id, time in created.items()),
            key=lambda row: row.thread_id,
        )
    )
    edge_rows = tuple(
        sorted(CodexEdge(parent, child, "closed") for parent, child in (edges or []))
    )
    return CodexGraph(threads, edge_rows)


def _source(thread_id: str, line: int, value: dict) -> FrozenJsonRecord:
    return FrozenJsonRecord(f"{thread_id}.jsonl", line, value)


def _meta(thread_id: str) -> dict:
    return {"type": "session_meta", "payload": {"id": thread_id}}


def _context(model: str, effort: str) -> dict:
    return {"type": "turn_context", "payload": {"model": model, "effort": effort}}


def _usage(input_tokens: int, output_tokens: int = 0) -> dict:
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": 0,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": 0,
        "total_tokens": input_tokens + output_tokens,
    }


def _token(
    total_input: int,
    last_input: int,
    *,
    timestamp: str,
    total_output: int = 0,
    last_output: int = 0,
) -> dict:
    return {
        "type": "event_msg",
        "timestamp": timestamp,
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": _usage(total_input, total_output),
                "last_token_usage": _usage(last_input, last_output),
            },
            "rate_limits": {"plan_type": "pro"},
        },
    }


def _heartbeat(total_input: int, *, timestamp: str) -> dict:
    row = _token(total_input, 0, timestamp=timestamp)
    row["payload"]["info"]["last_token_usage"]["total_tokens"] = 7
    return row


def _records(thread_id: str, rows: list[dict]) -> list[FrozenJsonRecord]:
    return [_source(thread_id, index, row) for index, row in enumerate([_meta(thread_id), *rows], 1)]


def _both(corpus):
    primary = reconstruct_dag_owned_suffix(corpus)
    independent = reconstruct_epoch_deltas(corpus)
    agreement = assert_exact_reconstruction_agreement(primary, independent)
    return primary, independent, agreement


def test_sanitized_graph_snapshot_uses_only_safe_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "state.sqlite"
    sessions = tmp_path / "sessions"
    project = tmp_path / "project"
    sessions.mkdir()
    project.mkdir()
    connection = sqlite3.connect(db_path)
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
    rows = [
        (
            "root",
            str(sessions / "root.jsonl"),
            str(project),
            "vscode",
            "openai",
            "gpt-test",
            "xhigh",
            "user",
            "legacy",
            1,
            2,
            1000,
            2000,
            999_999_999,
            "secret title",
            "secret prompt",
        ),
        (
            "child",
            str(sessions / "child.jsonl"),
            str(project / "subdir"),
            "subagent",
            "openai",
            "gpt-test",
            "xhigh",
            "subagent",
            "legacy",
            2,
            3,
            2000,
            3000,
            888_888_888,
            "another title",
            "another prompt",
        ),
        (
            "outside",
            str(sessions / "outside.jsonl"),
            str(tmp_path / "outside"),
            "vscode",
            "openai",
            "gpt-test",
            "low",
            "user",
            "legacy",
            1,
            1,
            1000,
            1000,
            777,
            "outside",
            "outside",
        ),
    ]
    connection.executemany(
        "INSERT INTO threads VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    connection.execute(
        "INSERT INTO thread_spawn_edges VALUES ('root','child','closed')"
    )
    connection.commit()
    connection.close()

    graph = snapshot_sanitized_graph(
        db_path, project_root=project, sessions_root=sessions
    )
    assert [thread.thread_id for thread in graph.threads] == ["child", "root"]
    assert graph.thread_by_id["child"].project_relative_cwd == "subdir"
    assert graph.edges == (CodexEdge("root", "child", "closed"),)
    serialized = graph.to_dict()
    assert "tokens_used" not in repr(serialized)
    assert "secret title" not in repr(serialized)
    assert serialized["graph_sha256"] == graph.graph_sha256


def test_parser_splits_resets_collapses_duplicates_ignores_heartbeat_and_tracks_switches() -> None:
    graph = _graph({"root": 1})
    first = _token(10, 10, timestamp="t1")
    first["payload"]["info"]["total_token_usage"]["total_tokens"] = 268_410
    records = _records(
        "root",
        [
            _context("gpt-a", "xhigh"),
            first,
            _token(10, 10, timestamp="rewritten-duplicate"),
            _heartbeat(10, timestamp="heartbeat"),
            _token(15, 5, timestamp="t2"),
            _context("gpt-b", "ultra"),
            _token(3, 3, timestamp="reset"),
        ],
    )
    corpus = parse_frozen_rollouts(records, graph)
    trace = corpus.traces[0]
    assert len(trace.events) == 3
    assert len(trace.epochs) == 2
    assert trace.duplicate_emissions == 1
    assert trace.heartbeat_records == 1
    assert trace.reported_total_mismatches == 1
    primary, _, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=18)
    assert [(row.model, row.effort, row.tokens.input_tokens) for row in primary.by_model_effort] == [
        ("gpt-a", "xhigh", 15),
        ("gpt-b", "ultra", 3),
    ]


def test_rewritten_timestamp_copied_prefix_counts_only_child_suffix() -> None:
    graph = _graph({"root": 1, "child": 2}, [("root", "child")])
    records = [
        *_records(
            "root",
            [
                _context("gpt", "xhigh"),
                _token(10, 10, timestamp="root-1"),
                _token(15, 5, timestamp="root-2"),
            ],
        ),
        *_records(
            "child",
            [
                _context("gpt", "xhigh"),
                _token(10, 10, timestamp="copy-time-changed-1"),
                _token(15, 5, timestamp="copy-time-changed-2"),
                _token(22, 7, timestamp="child-new"),
            ],
        ),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    primary, independent, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=22)
    assert primary.thread_ownership == independent.thread_ownership
    assert primary.thread_ownership[0].thread_id == "child"
    assert primary.thread_ownership[0].copied_prefix_events == 2
    assert primary.thread_ownership[0].owned_events == 1


def test_copied_prefix_can_begin_at_a_parent_history_segment() -> None:
    graph = _graph({"root": 1, "child": 2}, [("root", "child")])
    records = [
        *_records(
            "root",
            [
                _context("gpt", "xhigh"),
                _token(5, 5, timestamp="root-0"),
                _token(10, 5, timestamp="root-1"),
                _token(15, 5, timestamp="root-2"),
                _token(20, 5, timestamp="root-later"),
            ],
        ),
        *_records(
            "child",
            [
                _context("gpt", "xhigh"),
                _token(10, 5, timestamp="copied-segment-1"),
                _token(15, 5, timestamp="copied-segment-2"),
                _token(18, 3, timestamp="child-new"),
            ],
        ),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    primary, independent, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=23)
    assert primary.thread_ownership == independent.thread_ownership
    child = {row.thread_id: row for row in primary.thread_ownership}["child"]
    assert child.copied_prefix_events == 2
    assert child.owned_events == 1


def test_no_physical_prefix_and_divergent_siblings_count_identical_tuples_twice() -> None:
    graph = _graph(
        {"root": 1, "left": 2, "right": 3},
        [("root", "left"), ("root", "right")],
    )
    records = [
        *_records("root", [_context("gpt", "xhigh"), _token(10, 10, timestamp="r")]),
        *_records("left", [_context("gpt", "xhigh"), _token(15, 5, timestamp="l")]),
        *_records("right", [_context("gpt", "xhigh"), _token(15, 5, timestamp="q")]),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    primary, _, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=20)
    ownership = {row.thread_id: row for row in primary.thread_ownership}
    assert ownership["left"].copied_prefix_events == 0
    assert ownership["left"].inherited_boundary == TokenVector(input_tokens=10)
    assert ownership["right"].owned_events == 1


def test_same_single_tuple_in_independent_components_counts_twice() -> None:
    graph = _graph({"first": 1, "second": 2})
    records = [
        *_records("first", [_context("gpt", "xhigh"), _token(10, 10, timestamp="a")]),
        *_records("second", [_context("gpt", "xhigh"), _token(10, 10, timestamp="b")]),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    primary, _, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=20)
    assert primary.root_replay_aliases == ()


def test_unlinked_later_root_replay_is_collapsed_despite_timestamp_changes() -> None:
    graph = _graph({"original": 1, "replay": 2})
    original_rows = [
        _context("gpt", "xhigh"),
        _token(5, 5, timestamp="o1"),
        _token(10, 5, timestamp="o2"),
        _token(15, 5, timestamp="o3"),
    ]
    replay_rows = [
        _context("gpt", "xhigh"),
        _token(5, 5, timestamp="new1"),
        _token(10, 5, timestamp="new2"),
        _token(15, 5, timestamp="new3"),
    ]
    corpus = parse_frozen_rollouts(
        [*_records("original", original_rows), *_records("replay", replay_rows)],
        graph,
    )
    primary, independent, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=15)
    assert primary.root_replay_aliases == independent.root_replay_aliases
    assert primary.root_replay_aliases[0].replay_thread_id == "replay"
    replay_ownership = {row.thread_id: row for row in primary.thread_ownership}["replay"]
    assert replay_ownership.owned_events == 0
    assert replay_ownership.copied_prefix_events == 3


def test_root_replay_matching_two_older_roots_fails_closed() -> None:
    graph = _graph({"older-a": 1, "older-b": 2, "later": 3})
    common = [
        _token(5, 5, timestamp="1"),
        _token(10, 5, timestamp="2"),
        _token(15, 5, timestamp="3"),
    ]
    records = [
        *_records(
            "older-a",
            [_context("gpt", "xhigh"), *common, _token(20, 5, timestamp="a")],
        ),
        *_records(
            "older-b",
            [_context("gpt", "xhigh"), *common, _token(21, 6, timestamp="b")],
        ),
        *_records("later", [_context("gpt", "xhigh"), *common]),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    with pytest.raises(CodexAmbiguityError, match="multiple older roots"):
        reconstruct_dag_owned_suffix(corpus)
    with pytest.raises(CodexAmbiguityError, match="multiple older roots"):
        reconstruct_epoch_deltas(corpus)


def test_exact_agreement_gate_rejects_any_changed_total() -> None:
    graph = _graph({"root": 1})
    corpus = parse_frozen_rollouts(
        _records("root", [_context("gpt", "xhigh"), _token(10, 10, timestamp="x")]),
        graph,
    )
    primary = reconstruct_dag_owned_suffix(corpus)
    independent = reconstruct_epoch_deltas(corpus)
    changed = replace(independent, totals=TokenVector(input_tokens=11))
    with pytest.raises(CodexAccountingError, match="disagree on: totals"):
        assert_exact_reconstruction_agreement(primary, changed)
