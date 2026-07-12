from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from accounting.codex import (
    CodexAccountingError,
    CodexAmbiguityError,
    CodexEdge,
    CodexGraph,
    CodexThread,
    TokenVector,
    assert_reconstructed_agreement,
    parse_frozen_rollouts,
    reconstruct_dag_owned_suffix,
    reconstruct_epoch_deltas,
    reconstruct_source_timestamp_ownership,
    snapshot_sanitized_graph,
)
from accounting.core import FrozenJsonRecord


_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
_BASE_MS = int(_BASE.timestamp() * 1000)


def _ts(seconds: int) -> str:
    return (_BASE + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _thread(thread_id: str, created_seconds: int) -> CodexThread:
    created_ms = _BASE_MS + created_seconds * 1000
    return CodexThread(
        thread_id=thread_id,
        logical_rollout_path=f"{thread_id}.jsonl",
        created_at_ms=created_ms,
        updated_at_ms=created_ms + 1,
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
            (_thread(thread_id, seconds) for thread_id, seconds in created.items()),
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


def _response(
    response_id: str | None,
    *,
    kind: str = "message",
    text: str | None = None,
) -> dict:
    if kind == "message":
        payload = {
            "type": kind,
            "role": "assistant",
            "content": [] if text is None else [{"type": "output_text", "text": text}],
        }
    else:
        payload = {"type": kind}
    if response_id is not None:
        payload["id"] = response_id
    return {"type": "response_item", "payload": payload}


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
    at: int,
    total_output: int = 0,
    last_output: int = 0,
) -> dict:
    return {
        "type": "event_msg",
        "timestamp": _ts(at),
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": _usage(total_input, total_output),
                "last_token_usage": _usage(last_input, last_output),
            },
            "rate_limits": {"plan_type": "pro"},
        },
    }


def _call(
    total_input: int,
    last_input: int,
    *,
    at: int,
    response_id: str | None,
    response_text: str | None = None,
) -> list[dict]:
    return [
        _response(response_id, text=response_text),
        _token(total_input, last_input, at=at),
    ]


def _heartbeat(total_input: int, *, at: int) -> dict:
    row = _token(total_input, 0, at=at)
    # Codex heartbeat rows can carry a scalar context-window offset.  The
    # component vector is still zero and is the accounting source.
    row["payload"]["info"]["last_token_usage"]["total_tokens"] = 7
    return row


def _records(thread_id: str, rows: list[dict]) -> list[FrozenJsonRecord]:
    return [
        _source(thread_id, index, row)
        for index, row in enumerate([_meta(thread_id), *rows], 1)
    ]


def _both(corpus):
    primary = reconstruct_dag_owned_suffix(corpus)
    independent = reconstruct_source_timestamp_ownership(corpus)
    agreement = assert_reconstructed_agreement(primary, independent)
    return primary, independent, agreement


def test_sanitized_graph_snapshot_public_output_is_opaque_and_allowlisted(
    tmp_path: Path,
) -> None:
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
    root_id = "019f1111-aaaa-bbbb-cccc-111111111111"
    child_id = "019f2222-aaaa-bbbb-cccc-222222222222"
    secret_source = (
        '{"subagent":{"thread_id":"private-parent",'
        '"agent_nickname":"private-name","agent_role":"private-role"}}'
    )
    rows = [
        (
            root_id,
            str(sessions / "private-root-name.jsonl"),
            str(project),
            "vscode",
            "openai",
            "gpt-test",
            "xhigh",
            "user",
            "private-history-mode",
            1,
            2,
            1000,
            2000,
            999_999_999,
            "secret title",
            "secret prompt",
        ),
        (
            child_id,
            str(sessions / "private-child-name.jsonl"),
            str(project / "private-subdirectory"),
            secret_source,
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
    ]
    connection.executemany(
        "INSERT INTO threads VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    connection.execute(
        "INSERT INTO thread_spawn_edges VALUES (?,?,?)",
        (root_id, child_id, "private-edge-role"),
    )
    connection.commit()
    connection.close()

    graph = snapshot_sanitized_graph(
        db_path, project_root=project, sessions_root=sessions
    )
    serialized = graph.to_dict()
    rendered = repr(serialized)
    for private_value in (
        root_id,
        child_id,
        "private-root-name",
        "private-child-name",
        "private-subdirectory",
        "private-parent",
        "private-name",
        "private-role",
        "private-history-mode",
        "private-edge-role",
        "secret title",
        "secret prompt",
    ):
        assert private_value not in rendered
    assert serialized["schema"] == "codex_sanitized_thread_graph_v3"
    categories = {row["source_category"] for row in serialized["threads"]}
    assert categories == {"codex_app", "subagent"}
    assert all(row["thread_id"].startswith("thread_") for row in serialized["threads"])
    assert all(row["logical_rollout_path"].startswith("rollouts/thread_") for row in serialized["threads"])
    assert serialized == graph.to_dict()


def test_parser_handles_duplicate_heartbeat_explicit_reset_and_model_switch() -> None:
    graph = _graph({"root": 0})
    first = _token(10, 10, at=1)
    first["payload"]["info"]["total_token_usage"]["total_tokens"] = 268_410
    records = _records(
        "root",
        [
            _context("gpt-a", "xhigh"),
            _response("first"),
            first,
            _token(10, 10, at=2),
            _heartbeat(10, at=3),
            *_call(15, 5, at=4, response_id="second"),
            _heartbeat(0, at=5),
            _context("gpt-b", "ultra"),
            *_call(10, 10, at=6, response_id="after-reset"),
        ],
    )
    corpus = parse_frozen_rollouts(records, graph)
    trace = corpus.traces[0]
    assert len(trace.events) == 3
    assert len(trace.epochs) == 2
    assert trace.duplicate_emissions == 1
    assert trace.heartbeat_records == 2
    assert trace.explicit_zero_reset_markers == 1
    assert trace.reported_total_mismatches == 1
    primary, _, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=25)
    assert [
        (row.model, row.effort, row.tokens.input_tokens)
        for row in primary.by_model_effort
    ] == [("gpt-a", "xhigh", 15), ("gpt-b", "ultra", 10)]


@pytest.mark.parametrize("field", ["total_token_usage", "last_token_usage"])
def test_parser_rejects_unknown_usage_fields(field: str) -> None:
    graph = _graph({"root": 0})
    row = _token(5, 5, at=1)
    row["payload"]["info"][field]["future_counter"] = 1
    with pytest.raises(CodexAccountingError, match="token field set differs"):
        parse_frozen_rollouts(_records("root", [_response("r"), row]), graph)


def test_parser_rejects_advancing_zero_last_and_unknown_response_types() -> None:
    graph = _graph({"root": 0})
    advancing = _records(
        "root",
        [*_call(5, 5, at=1, response_id="r"), _heartbeat(7, at=2)],
    )
    with pytest.raises(CodexAccountingError, match="changes cumulative counters ambiguously"):
        parse_frozen_rollouts(advancing, graph)
    with pytest.raises(CodexAccountingError, match="unknown response_item type"):
        parse_frozen_rollouts(
            _records("root", [_response("x", kind="future_model_item")]), graph
        )


def test_parser_rejects_reordered_records_and_malformed_source_identity() -> None:
    graph = _graph({"root": 0})
    records = _records(
        "root",
        [_context("gpt", "xhigh"), *_call(5, 5, at=1, response_id="r")],
    )
    with pytest.raises(CodexAccountingError, match="not strictly line-ordered"):
        parse_frozen_rollouts([records[0], records[2], records[1]], graph)
    malformed = _response("r")
    malformed["payload"]["id"] = 17
    with pytest.raises(CodexAccountingError, match="id is malformed"):
        parse_frozen_rollouts(_records("root", [malformed]), graph)


def test_unchanged_counter_keeps_novel_response_for_the_next_advancing_usage() -> None:
    graph = _graph({"root": 0})
    records = _records(
        "root",
        [
            _context("gpt", "xhigh"),
            *_call(5, 5, at=1, response_id="first"),
            _response("next-call"),
            _token(5, 5, at=2),
            _token(10, 5, at=3),
        ],
    )
    corpus = parse_frozen_rollouts(records, graph)
    trace = corpus.traces[0]
    assert trace.duplicate_emissions == 1
    assert len(trace.events) == 2
    assert [row.digest for row in trace.events[0].response_evidence] != [
        row.digest for row in trace.events[1].response_evidence
    ]
    assert len(trace.events[0].response_evidence) == 1
    assert len(trace.events[1].response_evidence) == 1
    assert trace.pending_response_evidence == ()
    _, _, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=10)


def test_unchanged_counter_dedupes_old_evidence_but_novel_eof_evidence_blocks() -> None:
    graph = _graph({"root": 0})
    duplicate_corpus = parse_frozen_rollouts(
        _records(
            "root",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=1, response_id="same"),
                _response("same"),
                _token(5, 5, at=2),
            ],
        ),
        graph,
    )
    assert duplicate_corpus.traces[0].pending_response_evidence == ()
    _, _, agreement = _both(duplicate_corpus)
    assert agreement.totals == TokenVector(input_tokens=5)

    novel_corpus = parse_frozen_rollouts(
        _records(
            "root",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=1, response_id="first"),
                _response("novel-at-eof"),
                _token(5, 5, at=2),
            ],
        ),
        graph,
    )
    primary = reconstruct_dag_owned_suffix(novel_corpus)
    independent = reconstruct_source_timestamp_ownership(novel_corpus)
    assert primary.coverage.status == "FAIL_PENDING_RESPONSES"
    assert len(novel_corpus.traces[0].pending_response_evidence) == 1
    with pytest.raises(CodexAccountingError, match="coverage=FAIL_PENDING_RESPONSES"):
        assert_reconstructed_agreement(primary, independent)


def test_direct_parent_timestamp_crosscheck_matches_rewritten_physical_prefix() -> None:
    graph = _graph({"root": 0, "child": 10}, [("root", "child")])
    records = [
        *_records(
            "root",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=1, response_id="copy-1"),
                *_call(10, 5, at=2, response_id="copy-2"),
                *_call(15, 5, at=3, response_id="copy-3"),
                # This parent work happened after the child was created and is
                # not the child's inherited timestamp boundary.
                *_call(20, 5, at=20, response_id="root-later"),
            ],
        ),
        *_records(
            "child",
            [
                _context("gpt", "xhigh"),
                *_call(10, 5, at=30, response_id="copy-2"),
                *_call(15, 5, at=31, response_id="copy-3"),
                *_call(18, 3, at=32, response_id="child-owned"),
            ],
        ),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    primary, independent, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=23)
    assert primary.method != independent.method
    child = {row.thread_id: row for row in primary.thread_ownership}["child"]
    assert child.copied_prefix_events == 2
    assert child.owned_events == 1


def test_fresh_graph_child_and_logical_suffix_are_not_physical_copies() -> None:
    graph = _graph(
        {"root": 0, "fresh": 10, "suffix": 11},
        [("root", "fresh"), ("root", "suffix")],
    )
    records = [
        *_records(
            "root",
            [_context("gpt", "xhigh"), *_call(10, 10, at=1, response_id="root")],
        ),
        *_records(
            "fresh",
            [_context("gpt", "xhigh"), *_call(5, 5, at=12, response_id="fresh")],
        ),
        *_records(
            "suffix",
            [_context("gpt", "xhigh"), *_call(15, 5, at=13, response_id="suffix")],
        ),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    primary, _, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=20)
    ownership = {row.thread_id: row for row in primary.thread_ownership}
    assert ownership["fresh"].inherited_boundary == TokenVector()
    assert ownership["suffix"].inherited_boundary == TokenVector(input_tokens=10)
    assert ownership["fresh"].copied_prefix_events == 0
    assert ownership["suffix"].copied_prefix_events == 0


@pytest.mark.parametrize("event_count", [1, 2, 3])
def test_graph_counter_collision_is_reconstructed_with_explicit_upper_sensitivity(
    event_count: int,
) -> None:
    graph = _graph({"root": 0, "fresh": 10}, [("root", "fresh")])
    root_rows: list[dict] = [_context("gpt", "xhigh")]
    fresh_rows: list[dict] = [_context("gpt", "xhigh")]
    for index in range(event_count):
        root_rows.extend(
            _call(
                (index + 1) * 5,
                5,
                at=index + 1,
                response_id=f"root-{index}",
            )
        )
        fresh_rows.extend(
            _call(
                (index + 1) * 5,
                5,
                at=index + 20,
                response_id=f"fresh-{index}",
            )
        )
    corpus = parse_frozen_rollouts(
        [*_records("root", root_rows), *_records("fresh", fresh_rows)], graph
    )
    primary, _, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=event_count * 5)
    assert agreement.no_graph_collapse_upper_bound == TokenVector(
        input_tokens=event_count * 10
    )
    ownership = {row.thread_id: row for row in primary.thread_ownership}
    assert ownership["fresh"].copied_prefix_events == event_count
    classification = primary.to_dict()["structural_graph_copies"][0]
    assert classification["measurement_class"] == "reconstructed"
    assert classification["source_identity_proven"] is False
    assert "cannot_be_excluded" in classification["nonidentifiability"]


def test_graph_structural_reconstruction_marks_missing_payload_as_incomplete() -> None:
    graph = _graph({"root": 0, "child": 10}, [("root", "child")])
    root_rows: list[dict] = [_context("gpt", "xhigh")]
    child_rows: list[dict] = [_context("gpt", "xhigh")]
    for index in range(3):
        root_rows.extend(
            _call((index + 1) * 5, 5, at=index + 1, response_id=None)
        )
        child_rows.append(
            _token((index + 1) * 5, 5, at=index + 20)
        )
    corpus = parse_frozen_rollouts(
        [*_records("root", root_rows), *_records("child", child_rows)], graph
    )
    primary = reconstruct_dag_owned_suffix(corpus)
    independent = reconstruct_source_timestamp_ownership(corpus)
    assert primary.totals == TokenVector(input_tokens=15)
    assert primary.thread_ownership == independent.thread_ownership
    assert primary.structural_graph_copies == independent.structural_graph_copies
    assert primary.no_graph_collapse_upper_bound == TokenVector(input_tokens=30)
    assert primary.reconstructed_total_authorized
    serialized = assert_reconstructed_agreement(primary, independent).to_dict()
    assert serialized["status"] == "reconstructed_agreement_coverage_pass"
    assert serialized["measurement_class"] == "reconstructed"
    assert "exact" not in repr(serialized).lower()
    assert serialized["structural_graph_copies"][0]["source_identity_proven"] is False
    assert serialized["structural_graph_copies"][0]["payload_evidence"] == (
        "incomplete_structural_fallback"
    )
    assert serialized["structural_graph_copy_summary"][
        "payload_incomplete_copy_count"
    ] == 1


def test_complete_different_payloads_override_equal_graph_counter_sequence() -> None:
    graph = _graph({"root": 0, "fresh": 10}, [("root", "fresh")])
    root_rows: list[dict] = [_context("gpt", "xhigh")]
    fresh_rows: list[dict] = [_context("gpt", "xhigh")]
    for index in range(3):
        root_rows.extend(
            _call(
                (index + 1) * 5,
                5,
                at=index + 1,
                response_id=f"root-{index}",
                response_text=f"root content {index}",
            )
        )
        fresh_rows.extend(
            _call(
                (index + 1) * 5,
                5,
                at=index + 20,
                response_id=f"fresh-{index}",
                response_text=f"different fresh content {index}",
            )
        )
    corpus = parse_frozen_rollouts(
        [*_records("root", root_rows), *_records("fresh", fresh_rows)], graph
    )
    primary, independent, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=30)
    ownership = {row.thread_id: row for row in primary.thread_ownership}
    assert ownership["fresh"].owned_events == 3
    assert primary.structural_graph_copies == ()
    assert primary.rejected_graph_copy_candidates == (
        independent.rejected_graph_copy_candidates
    )
    assert primary.rejected_graph_copy_candidates[0].candidate_event_count == 3
    serialized = agreement.to_dict()
    assert serialized["structural_graph_copy_summary"][
        "payload_rejected_candidate_count"
    ] == 1


def test_copied_prefix_beginning_at_reset_epoch_is_not_lost() -> None:
    graph = _graph({"root": 0, "child": 10}, [("root", "child")])
    records = [
        *_records(
            "root",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=1, response_id="epoch-0"),
                _heartbeat(0, at=2),
                *_call(5, 5, at=3, response_id="epoch-1-a"),
                *_call(8, 3, at=4, response_id="epoch-1-b"),
            ],
        ),
        *_records(
            "child",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=20, response_id="epoch-1-a"),
                *_call(8, 3, at=21, response_id="epoch-1-b"),
                *_call(10, 2, at=22, response_id="child-owned"),
            ],
        ),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    primary, _, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=15)
    child = {row.thread_id: row for row in primary.thread_ownership}["child"]
    assert child.copied_prefix_events == 2
    assert child.owned_events == 1


@pytest.mark.parametrize("event_count", [1, 2])
def test_source_proven_short_root_replay_is_collapsed(event_count: int) -> None:
    graph = _graph({"original": 0, "replay": 10})
    original: list[dict] = [_context("gpt", "xhigh")]
    replay: list[dict] = [_context("gpt", "xhigh")]
    for index in range(event_count):
        original.extend(
            _call(
                (index + 1) * 5,
                5,
                at=index + 1,
                response_id=f"shared-{index}",
            )
        )
        replay.extend(
            _call(
                (index + 1) * 5,
                5,
                at=index + 20,
                response_id=f"shared-{index}",
            )
        )
    corpus = parse_frozen_rollouts(
        [*_records("original", original), *_records("replay", replay)], graph
    )
    primary, independent, agreement = _both(corpus)
    expected = TokenVector(input_tokens=event_count * 5)
    assert agreement.totals == expected
    assert agreement.no_root_collapse_totals == TokenVector(
        input_tokens=event_count * 10
    )
    assert primary.root_replay_aliases == independent.root_replay_aliases
    alias = primary.root_replay_aliases[0]
    assert alias.copied_event_count == event_count
    assert alias.owned_event_count == 0
    serialized = primary.to_dict()
    assert serialized["root_replay_aliases"][0]["evidence_classification"] == (
        "stable_source_identity_prefix"
    )
    assert "'original'" not in repr(serialized)
    assert "'replay'" not in repr(serialized)


def test_root_replay_copied_prefix_plus_owned_suffix_counts_the_suffix() -> None:
    graph = _graph({"original": 0, "replay": 10})
    records = [
        *_records(
            "original",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=1, response_id="shared"),
                *_call(10, 5, at=2, response_id="original-second"),
            ],
        ),
        *_records(
            "replay",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=20, response_id="shared"),
                *_call(12, 7, at=21, response_id="replay-owned"),
            ],
        ),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    primary, _, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=17)
    assert agreement.no_root_collapse_totals == TokenVector(input_tokens=22)
    alias = primary.root_replay_aliases[0]
    assert alias.copied_event_count == 1
    assert alias.owned_event_count == 1


def test_partial_root_replay_remains_a_source_for_a_later_replay_chain() -> None:
    graph = _graph({"a": 0, "b": 10, "c": 20})
    records = [
        *_records(
            "a",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=1, response_id="a-shared"),
            ],
        ),
        *_records(
            "b",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=11, response_id="a-shared"),
                *_call(12, 7, at=12, response_id="b-suffix"),
            ],
        ),
        *_records(
            "c",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=21, response_id="a-shared"),
                *_call(12, 7, at=22, response_id="b-suffix"),
            ],
        ),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    primary, _, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=12)
    assert agreement.no_root_collapse_totals == TokenVector(input_tokens=29)
    aliases = {row.replay_thread_id: row for row in primary.root_replay_aliases}
    assert aliases["b"].canonical_thread_id == "a"
    assert aliases["b"].copied_event_count == 1
    assert aliases["b"].owned_event_count == 1
    assert aliases["c"].canonical_thread_id == "b"
    assert aliases["c"].copied_event_count == 2
    assert aliases["c"].owned_event_count == 0


def test_owned_stable_identity_uniqueness_blocks_cross_topology_replay() -> None:
    graph = _graph({"a": 0, "b": 10, "c": 20}, [("a", "b")])
    records = [
        *_records(
            "a",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=1, response_id="a-item"),
            ],
        ),
        *_records(
            "b",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=11, response_id="a-item"),
                *_call(12, 7, at=12, response_id="b-owned-item"),
            ],
        ),
        *_records(
            "c",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=21, response_id="a-item"),
                *_call(12, 7, at=22, response_id="b-owned-item"),
            ],
        ),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    primary = reconstruct_dag_owned_suffix(corpus)
    independent = reconstruct_source_timestamp_ownership(corpus)
    assert primary.totals == TokenVector(input_tokens=19)
    assert primary.no_graph_collapse_upper_bound == TokenVector(input_tokens=24)
    assert len(primary.owned_stable_evidence_collisions) == 1
    collision = primary.owned_stable_evidence_collisions[0]
    assert collision.evidence_kind == "message"
    assert len(collision.locations) == 2
    assert not primary.reconstructed_total_authorized
    with pytest.raises(
        CodexAccountingError, match="owned_stable_response_identity_collision"
    ):
        assert_reconstructed_agreement(primary, independent)


def test_identical_tied_root_replay_sources_fail_conservatively() -> None:
    graph = _graph({"a": 0, "b": 10, "c": 20})
    records: list[FrozenJsonRecord] = []
    for thread_id, at in (("a", 1), ("b", 11), ("c", 21)):
        records.extend(
            _records(
                thread_id,
                [
                    _context("gpt", "xhigh"),
                    *_call(5, 5, at=at, response_id="same-source"),
                ],
            )
        )
    corpus = parse_frozen_rollouts(records, graph)
    with pytest.raises(CodexAmbiguityError, match="matches multiple roots"):
        reconstruct_dag_owned_suffix(corpus)
    with pytest.raises(CodexAmbiguityError, match="matches multiple roots"):
        reconstruct_source_timestamp_ownership(corpus)


def test_coincidental_root_counter_sequences_never_collapse() -> None:
    graph = _graph({"first": 0, "second": 10})
    records = [
        *_records(
            "first",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=1, response_id="first-a"),
                *_call(10, 5, at=2, response_id="first-b"),
                *_call(15, 5, at=3, response_id="first-c"),
            ],
        ),
        *_records(
            "second",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=20, response_id="second-a"),
                *_call(10, 5, at=21, response_id="second-b"),
                *_call(15, 5, at=22, response_id="second-c"),
            ],
        ),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    primary, _, agreement = _both(corpus)
    assert agreement.totals == TokenVector(input_tokens=30)
    assert primary.root_replay_aliases == ()
    assert primary.unresolved_root_replay_sensitivities == ()


def test_counter_only_root_match_is_reported_as_sensitivity_not_collapsed() -> None:
    graph = _graph({"first": 0, "second": 10})
    records = [
        *_records(
            "first",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=1, response_id=None),
                *_call(10, 5, at=2, response_id=None),
            ],
        ),
        *_records(
            "second",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=20, response_id=None),
                *_call(10, 5, at=21, response_id=None),
            ],
        ),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    primary = reconstruct_dag_owned_suffix(corpus)
    independent = reconstruct_epoch_deltas(corpus)
    assert primary.totals == TokenVector(input_tokens=20)
    assert primary.no_root_collapse_totals == TokenVector(input_tokens=20)
    assert primary.possible_root_collapse_lower_bound == TokenVector(input_tokens=10)
    assert primary.root_replay_aliases == ()
    assert len(primary.unresolved_root_replay_sensitivities) == 1
    assert primary.unresolved_root_replay_sensitivities[0].potential_copied_event_count == 2
    with pytest.raises(CodexAccountingError, match="unresolved_root_replay_sensitivity"):
        assert_reconstructed_agreement(primary, independent)


def test_source_evidence_ending_mid_prefix_yields_alias_plus_bounded_sensitivity() -> None:
    graph = _graph({"first": 0, "second": 10})
    records = [
        *_records(
            "first",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=1, response_id="shared"),
                *_call(10, 5, at=2, response_id=None),
            ],
        ),
        *_records(
            "second",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=20, response_id="shared"),
                *_call(10, 5, at=21, response_id=None),
            ],
        ),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    primary = reconstruct_dag_owned_suffix(corpus)
    assert primary.totals == TokenVector(input_tokens=15)
    assert primary.no_root_collapse_totals == TokenVector(input_tokens=20)
    assert primary.possible_root_collapse_lower_bound == TokenVector(input_tokens=10)
    assert primary.root_replay_aliases[0].copied_event_count == 1
    assert primary.unresolved_root_replay_sensitivities[0].potential_copied_event_count == 1


@pytest.mark.parametrize("mode", ["uncovered", "pending"])
def test_quiescent_completed_response_usage_coverage_is_a_reconstructed_gate(mode: str) -> None:
    graph = _graph({"root": 0})
    if mode == "uncovered":
        rows = [_context("gpt", "xhigh"), _token(5, 5, at=1)]
        expected = "FAIL_UNCOVERED_USAGE"
    else:
        rows = [
            _context("gpt", "xhigh"),
            *_call(5, 5, at=1, response_id="used"),
            _response("not-yet-metered"),
        ]
        expected = "FAIL_PENDING_RESPONSES"
    corpus = parse_frozen_rollouts(_records("root", rows), graph)
    primary = reconstruct_dag_owned_suffix(corpus)
    independent = reconstruct_source_timestamp_ownership(corpus)
    assert primary.coverage.status == expected
    assert not primary.reconstructed_total_authorized
    with pytest.raises(CodexAccountingError, match="coverage="):
        assert_reconstructed_agreement(primary, independent)


def test_missing_direct_attribution_is_not_imputed_from_equal_counters() -> None:
    graph = _graph({"known": 0, "unknown": 10})
    records = [
        *_records(
            "known",
            [_context("gpt", "xhigh"), *_call(5, 5, at=1, response_id="known")],
        ),
        *_records("unknown", _call(5, 5, at=20, response_id="unknown")),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    primary = reconstruct_dag_owned_suffix(corpus)
    independent = reconstruct_source_timestamp_ownership(corpus)
    assert primary.totals == TokenVector(input_tokens=10)
    assert primary.unresolved_attribution_events == 1
    assert any(row.model == "<unknown-model>" for row in primary.by_model_effort)
    with pytest.raises(CodexAccountingError, match="unresolved_model_attribution"):
        assert_reconstructed_agreement(primary, independent)


def test_wrong_spawn_boundary_stays_owned_but_uniqueness_blocks_replay() -> None:
    graph = _graph({"root": 0, "child": 10}, [("root", "child")])
    records = [
        *_records(
            "root",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=1, response_id="copy-a"),
                *_call(10, 5, at=9, response_id="copy-b"),
            ],
        ),
        *_records(
            "child",
            [
                _context("gpt", "xhigh"),
                *_call(5, 5, at=20, response_id="copy-a"),
                *_call(8, 3, at=21, response_id="owned"),
            ],
        ),
    ]
    corpus = parse_frozen_rollouts(records, graph)
    primary = reconstruct_dag_owned_suffix(corpus)
    independent = reconstruct_source_timestamp_ownership(corpus)
    assert primary.thread_ownership == independent.thread_ownership
    assert primary.structural_graph_copies == ()
    assert primary.totals == TokenVector(input_tokens=18)
    assert len(primary.owned_stable_evidence_collisions) == 1
    with pytest.raises(
        CodexAccountingError, match="owned_stable_response_identity_collision"
    ):
        assert_reconstructed_agreement(primary, independent)


def test_reconstructed_agreement_gate_rejects_any_changed_total() -> None:
    graph = _graph({"root": 0})
    corpus = parse_frozen_rollouts(
        _records(
            "root",
            [_context("gpt", "xhigh"), *_call(10, 10, at=1, response_id="r")],
        ),
        graph,
    )
    primary = reconstruct_dag_owned_suffix(corpus)
    independent = reconstruct_epoch_deltas(corpus)
    changed = replace(independent, totals=TokenVector(input_tokens=11))
    with pytest.raises(CodexAccountingError, match="disagree on: totals"):
        assert_reconstructed_agreement(primary, changed)
