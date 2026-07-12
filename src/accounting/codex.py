"""Codex subscription-workload reconstruction from frozen rollout logs.

Codex token counters are cumulative within a thread, copied into forked
threads, and occasionally reset.  Consequently neither SQLite ``tokens_used``
nor a global set of cumulative tuples is additive.  This module reconstructs
owned work twice:

* a graph/suffix method that sums owned ``last_token_usage`` rows;
* an independently implemented epoch/delta method that derives increments
  from cumulative counters and inherited boundaries.

The two paths share immutable source dataclasses only.  The final gate requires
exact agreement; neither implementation is adjusted to make real data agree.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .core import AccountingError, FrozenJsonRecord, canonical_sha256


class CodexAccountingError(AccountingError):
    """Base error for malformed or unreconstructable Codex usage."""


class CodexAmbiguityError(CodexAccountingError):
    """Raised when copied history or a replay alias is not uniquely identified."""


@dataclass(frozen=True, order=True)
class TokenVector:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0

    def __post_init__(self) -> None:
        for field in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CodexAccountingError(f"{field} must be a nonnegative integer")
        if self.cached_input_tokens > self.input_tokens:
            raise CodexAccountingError("cached input exceeds input")
        if self.reasoning_output_tokens > self.output_tokens:
            raise CodexAccountingError("reasoning output exceeds output")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def is_zero(self) -> bool:
        return not any(self.as_tuple())

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.input_tokens,
            self.cached_input_tokens,
            self.output_tokens,
            self.reasoning_output_tokens,
        )

    def componentwise_at_least(self, other: "TokenVector") -> bool:
        return all(left >= right for left, right in zip(self.as_tuple(), other.as_tuple()))

    def __add__(self, other: "TokenVector") -> "TokenVector":
        return TokenVector(
            self.input_tokens + other.input_tokens,
            self.cached_input_tokens + other.cached_input_tokens,
            self.output_tokens + other.output_tokens,
            self.reasoning_output_tokens + other.reasoning_output_tokens,
        )

    def subtract(self, other: "TokenVector") -> "TokenVector":
        if not self.componentwise_at_least(other):
            raise CodexAccountingError(f"negative token delta: {self.as_tuple()} - {other.as_tuple()}")
        return TokenVector(
            self.input_tokens - other.input_tokens,
            self.cached_input_tokens - other.cached_input_tokens,
            self.output_tokens - other.output_tokens,
            self.reasoning_output_tokens - other.reasoning_output_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "uncached_input_tokens": self.input_tokens - self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "non_reasoning_output_tokens": self.output_tokens - self.reasoning_output_tokens,
            "total_tokens": self.total_tokens,
        }


ZERO_TOKENS = TokenVector()


@dataclass(frozen=True)
class CodexThread:
    thread_id: str
    logical_rollout_path: str
    created_at_ms: int
    updated_at_ms: int
    source: str
    model_provider: str
    model: str | None
    reasoning_effort: str | None
    thread_source: str | None
    history_mode: str | None
    project_relative_cwd: str

    def __post_init__(self) -> None:
        if not self.thread_id:
            raise CodexAccountingError("thread_id is empty")
        logical = PurePosixPath(self.logical_rollout_path)
        if logical.is_absolute() or not self.logical_rollout_path or ".." in logical.parts:
            raise CodexAccountingError(
                f"invalid logical rollout path: {self.logical_rollout_path!r}"
            )
        for field in ("created_at_ms", "updated_at_ms"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CodexAccountingError(f"{field} must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "logical_rollout_path": self.logical_rollout_path,
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
            "source": self.source,
            "model_provider": self.model_provider,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "thread_source": self.thread_source,
            "history_mode": self.history_mode,
            "project_relative_cwd": self.project_relative_cwd,
        }


@dataclass(frozen=True, order=True)
class CodexEdge:
    parent_thread_id: str
    child_thread_id: str
    status: str

    def __post_init__(self) -> None:
        if not self.parent_thread_id or not self.child_thread_id:
            raise CodexAccountingError("spawn edge has an empty endpoint")
        if self.parent_thread_id == self.child_thread_id:
            raise CodexAccountingError("thread cannot spawn itself")

    def to_dict(self) -> dict[str, str]:
        return {
            "parent_thread_id": self.parent_thread_id,
            "child_thread_id": self.child_thread_id,
            "status": self.status,
        }


@dataclass(frozen=True)
class CodexGraph:
    threads: tuple[CodexThread, ...]
    edges: tuple[CodexEdge, ...]

    def __post_init__(self) -> None:
        ids = [thread.thread_id for thread in self.threads]
        paths = [thread.logical_rollout_path for thread in self.threads]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise CodexAccountingError("graph threads must be sorted with unique IDs")
        if len(paths) != len(set(paths)):
            raise CodexAccountingError("graph rollout paths are not unique")
        if list(self.edges) != sorted(self.edges):
            raise CodexAccountingError("graph edges must be sorted")
        known = set(ids)
        child_ids: set[str] = set()
        for edge in self.edges:
            if edge.parent_thread_id not in known or edge.child_thread_id not in known:
                raise CodexAccountingError("spawn edge references a thread outside the snapshot")
            if edge.child_thread_id in child_ids:
                raise CodexAccountingError(f"thread has multiple parents: {edge.child_thread_id}")
            child_ids.add(edge.child_thread_id)
        self.topological_thread_ids()

    @property
    def graph_sha256(self) -> str:
        return canonical_sha256(self.unsigned_dict())

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": "codex_sanitized_thread_graph_v1",
            "threads": [thread.to_dict() for thread in self.threads],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.unsigned_dict()
        value["graph_sha256"] = self.graph_sha256
        return value

    @property
    def thread_by_id(self) -> dict[str, CodexThread]:
        return {thread.thread_id: thread for thread in self.threads}

    @property
    def thread_id_by_path(self) -> dict[str, str]:
        return {thread.logical_rollout_path: thread.thread_id for thread in self.threads}

    @property
    def parent_by_child(self) -> dict[str, str]:
        return {edge.child_thread_id: edge.parent_thread_id for edge in self.edges}

    @property
    def roots(self) -> tuple[str, ...]:
        children = set(self.parent_by_child)
        return tuple(thread.thread_id for thread in self.threads if thread.thread_id not in children)

    def ancestors(self, thread_id: str) -> tuple[str, ...]:
        parents = self.parent_by_child
        result: list[str] = []
        current = thread_id
        while current in parents:
            current = parents[current]
            result.append(current)
        return tuple(result)

    def topological_thread_ids(self) -> tuple[str, ...]:
        children: dict[str, list[str]] = defaultdict(list)
        indegree = {thread.thread_id: 0 for thread in self.threads}
        for edge in self.edges:
            children[edge.parent_thread_id].append(edge.child_thread_id)
            indegree[edge.child_thread_id] += 1
        ready = deque(sorted(thread_id for thread_id, degree in indegree.items() if degree == 0))
        result: list[str] = []
        while ready:
            current = ready.popleft()
            result.append(current)
            for child in sorted(children[current]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(result) != len(self.threads):
            raise CodexAccountingError("spawn graph contains a cycle")
        return tuple(result)


def _within_project(cwd: Any, project_root: Path) -> str | None:
    if not isinstance(cwd, str):
        return None
    try:
        relative = Path(cwd).expanduser().resolve().relative_to(project_root)
    except (OSError, ValueError):
        return None
    return "." if not relative.parts else relative.as_posix()


def _logical_rollout_path(raw: str, sessions_root: Path) -> str:
    try:
        return Path(raw).expanduser().resolve().relative_to(sessions_root).as_posix()
    except (OSError, ValueError) as exc:
        raise CodexAccountingError(f"rollout path escapes sessions root: {raw!r}") from exc


def snapshot_sanitized_graph(
    database_path: Path,
    *,
    project_root: Path,
    sessions_root: Path,
) -> CodexGraph:
    """Read only safe graph fields from a consistent SQLite transaction."""

    database_path = Path(database_path).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    sessions_root = Path(sessions_root).expanduser().resolve()
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        raw_threads = connection.execute(
            """
            SELECT id, rollout_path, cwd, source, model_provider,
                   model, reasoning_effort, thread_source, history_mode,
                   COALESCE(created_at_ms, created_at * 1000) AS safe_created_at_ms,
                   COALESCE(updated_at_ms, updated_at * 1000) AS safe_updated_at_ms
              FROM threads
            """
        ).fetchall()
        threads: list[CodexThread] = []
        for row in raw_threads:
            relative_cwd = _within_project(row["cwd"], project_root)
            if relative_cwd is None:
                continue
            threads.append(
                CodexThread(
                    thread_id=str(row["id"]),
                    logical_rollout_path=_logical_rollout_path(
                        str(row["rollout_path"]), sessions_root
                    ),
                    created_at_ms=int(row["safe_created_at_ms"]),
                    updated_at_ms=int(row["safe_updated_at_ms"]),
                    source=str(row["source"] or ""),
                    model_provider=str(row["model_provider"] or ""),
                    model=None if row["model"] is None else str(row["model"]),
                    reasoning_effort=(
                        None
                        if row["reasoning_effort"] is None
                        else str(row["reasoning_effort"])
                    ),
                    thread_source=(
                        None if row["thread_source"] is None else str(row["thread_source"])
                    ),
                    history_mode=(
                        None if row["history_mode"] is None else str(row["history_mode"])
                    ),
                    project_relative_cwd=relative_cwd,
                )
            )
        project_ids = {thread.thread_id for thread in threads}
        raw_edges = connection.execute(
            "SELECT parent_thread_id, child_thread_id, status FROM thread_spawn_edges"
        ).fetchall()
        edges: list[CodexEdge] = []
        for row in raw_edges:
            child = str(row["child_thread_id"])
            if child not in project_ids:
                continue
            parent = str(row["parent_thread_id"])
            if parent not in project_ids:
                raise CodexAccountingError(
                    f"project thread {child} has parent outside sanitized graph: {parent}"
                )
            edges.append(CodexEdge(parent, child, str(row["status"] or "")))
        connection.rollback()
    finally:
        connection.close()
    return CodexGraph(tuple(sorted(threads, key=lambda row: row.thread_id)), tuple(sorted(edges)))


@dataclass(frozen=True)
class CodexTokenEvent:
    thread_id: str
    ordinal: int
    epoch_index: int
    total: TokenVector
    last: TokenVector
    model: str | None
    effort: str | None
    timestamp: str | None
    source_ref: str

    @property
    def semantic_signature(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self.total.as_tuple(), self.last.as_tuple()


@dataclass(frozen=True)
class CodexEpoch:
    epoch_index: int
    events: tuple[CodexTokenEvent, ...]


@dataclass(frozen=True)
class CodexThreadTrace:
    thread_id: str
    logical_rollout_path: str
    events: tuple[CodexTokenEvent, ...]
    epochs: tuple[CodexEpoch, ...]
    heartbeat_records: int
    duplicate_emissions: int
    reported_total_mismatches: int


@dataclass(frozen=True)
class CodexCorpus:
    graph: CodexGraph
    traces: tuple[CodexThreadTrace, ...]

    @property
    def trace_by_thread(self) -> dict[str, CodexThreadTrace]:
        return {trace.thread_id: trace for trace in self.traces}

    @property
    def heartbeat_records(self) -> int:
        return sum(trace.heartbeat_records for trace in self.traces)

    @property
    def duplicate_emissions(self) -> int:
        return sum(trace.duplicate_emissions for trace in self.traces)

    @property
    def reported_total_mismatches(self) -> int:
        return sum(trace.reported_total_mismatches for trace in self.traces)

    @property
    def reset_count(self) -> int:
        return sum(max(0, len(trace.epochs) - 1) for trace in self.traces)


def _strict_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CodexAccountingError(f"{label} must be a nonnegative integer")
    return value


def _parse_vector(
    value: Any,
    label: str,
    *,
    allow_total_only_heartbeat: bool = False,
    require_reported_total_match: bool = True,
) -> TokenVector:
    if not isinstance(value, Mapping):
        raise CodexAccountingError(f"{label} is not an object")
    vector = TokenVector(
        input_tokens=_strict_count(value.get("input_tokens"), f"{label}.input_tokens"),
        cached_input_tokens=_strict_count(
            value.get("cached_input_tokens"), f"{label}.cached_input_tokens"
        ),
        output_tokens=_strict_count(value.get("output_tokens"), f"{label}.output_tokens"),
        reasoning_output_tokens=_strict_count(
            value.get("reasoning_output_tokens"), f"{label}.reasoning_output_tokens"
        ),
    )
    reported_total = _strict_count(value.get("total_tokens"), f"{label}.total_tokens")
    if (
        require_reported_total_match
        and reported_total != vector.total_tokens
        and not (allow_total_only_heartbeat and vector.is_zero)
    ):
        raise CodexAccountingError(
            f"{label}.total_tokens mismatch: {reported_total} != {vector.total_tokens}"
        )
    return vector


def parse_frozen_rollouts(
    records: Iterable[FrozenJsonRecord], graph: CodexGraph
) -> CodexCorpus:
    """Parse meaningful token events from frozen rollout records."""

    thread_for_path = graph.thread_id_by_path
    events: dict[str, list[CodexTokenEvent]] = {thread.thread_id: [] for thread in graph.threads}
    contexts: dict[str, tuple[str | None, str | None]] = {
        thread.thread_id: (None, None) for thread in graph.threads
    }
    meta_seen: set[str] = set()
    heartbeats: dict[str, int] = defaultdict(int)
    duplicates: dict[str, int] = defaultdict(int)
    reported_total_mismatches: dict[str, int] = defaultdict(int)
    for source in records:
        thread_id = thread_for_path.get(source.logical_path)
        if thread_id is None:
            raise CodexAccountingError(
                f"frozen rollout is absent from sanitized graph: {source.logical_path}"
            )
        record = source.value
        payload = record.get("payload")
        if record.get("type") == "session_meta":
            if thread_id not in meta_seen:
                if not isinstance(payload, Mapping) or str(payload.get("id")) != thread_id:
                    raise CodexAccountingError(
                        f"initial session_meta ID mismatch at {source.logical_path}:{source.line_number}"
                    )
                meta_seen.add(thread_id)
            continue
        if record.get("type") == "turn_context":
            if not isinstance(payload, Mapping):
                raise CodexAccountingError(
                    f"turn_context payload is malformed at {source.logical_path}:{source.line_number}"
                )
            raw_model = payload.get("model")
            raw_effort = payload.get("effort")
            contexts[thread_id] = (
                raw_model if isinstance(raw_model, str) and raw_model else None,
                raw_effort if isinstance(raw_effort, str) and raw_effort else None,
            )
            continue
        if record.get("type") != "event_msg" or not isinstance(payload, Mapping):
            continue
        if payload.get("type") != "token_count":
            continue
        info = payload.get("info")
        if not isinstance(info, Mapping):
            raise CodexAccountingError(
                f"token_count lacks info at {source.logical_path}:{source.line_number}"
            )
        label = f"{source.logical_path}:{source.line_number}"
        last = _parse_vector(
            info.get("last_token_usage"),
            f"{label}.last",
            allow_total_only_heartbeat=True,
        )
        if last.is_zero:
            heartbeats[thread_id] += 1
            continue
        raw_total = info.get("total_token_usage")
        total = _parse_vector(
            raw_total,
            f"{label}.total",
            require_reported_total_match=False,
        )
        if not isinstance(raw_total, Mapping):
            raise CodexAccountingError(f"{label}.total is not an object")
        # Codex sometimes offsets cumulative `total_tokens` by the model
        # context-window size.  The four component counters remain internally
        # additive and are the accounting source; retain the mismatch count as
        # a diagnostic instead of coercing or summing the offset scalar.
        if raw_total.get("total_tokens") != total.total_tokens:
            reported_total_mismatches[thread_id] += 1
        prior = events[thread_id][-1] if events[thread_id] else None
        model, effort = contexts[thread_id]
        if prior is not None and total == prior.total:
            if last != prior.last:
                raise CodexAmbiguityError(
                    f"unchanged cumulative state has different nonzero last usage at {label}"
                )
            if prior.model and model and prior.model != model:
                raise CodexAmbiguityError(f"duplicate emission changes model at {label}")
            if prior.effort and effort and prior.effort != effort:
                raise CodexAmbiguityError(f"duplicate emission changes effort at {label}")
            if (prior.model is None and model is not None) or (
                prior.effort is None and effort is not None
            ):
                events[thread_id][-1] = replace(
                    prior,
                    model=prior.model or model,
                    effort=prior.effort or effort,
                )
            duplicates[thread_id] += 1
            continue
        if prior is None:
            epoch_index = 0
        elif total.componentwise_at_least(prior.total):
            if total.subtract(prior.total) != last:
                raise CodexAccountingError(f"cumulative transition does not equal last usage at {label}")
            epoch_index = prior.epoch_index
        else:
            if total != last:
                raise CodexAccountingError(f"counter reset does not restart at last usage at {label}")
            epoch_index = prior.epoch_index + 1
        timestamp = record.get("timestamp")
        events[thread_id].append(
            CodexTokenEvent(
                thread_id=thread_id,
                ordinal=len(events[thread_id]),
                epoch_index=epoch_index,
                total=total,
                last=last,
                model=model,
                effort=effort,
                timestamp=timestamp if isinstance(timestamp, str) else None,
                source_ref=label,
            )
        )
    missing_meta = set(events) - meta_seen
    if missing_meta:
        raise CodexAccountingError(f"rollouts lack matching session_meta: {sorted(missing_meta)!r}")
    traces: list[CodexThreadTrace] = []
    for thread in graph.threads:
        thread_events = tuple(events[thread.thread_id])
        epoch_rows: list[CodexEpoch] = []
        for epoch_index in sorted({event.epoch_index for event in thread_events}):
            epoch_rows.append(
                CodexEpoch(
                    epoch_index,
                    tuple(event for event in thread_events if event.epoch_index == epoch_index),
                )
            )
        traces.append(
            CodexThreadTrace(
                thread_id=thread.thread_id,
                logical_rollout_path=thread.logical_rollout_path,
                events=thread_events,
                epochs=tuple(epoch_rows),
                heartbeat_records=heartbeats[thread.thread_id],
                duplicate_emissions=duplicates[thread.thread_id],
                reported_total_mismatches=reported_total_mismatches[thread.thread_id],
            )
        )
    return CodexCorpus(graph, tuple(sorted(traces, key=lambda trace: trace.thread_id)))


@dataclass(frozen=True, order=True)
class RootReplayAlias:
    replay_thread_id: str
    canonical_thread_id: str
    copied_event_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_thread_id": self.replay_thread_id,
            "canonical_thread_id": self.canonical_thread_id,
            "copied_event_count": self.copied_event_count,
        }


@dataclass(frozen=True)
class ThreadOwnership:
    thread_id: str
    copied_prefix_events: int
    owned_events: int
    inherited_boundary: TokenVector
    replay_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "copied_prefix_events": self.copied_prefix_events,
            "owned_events": self.owned_events,
            "inherited_boundary": self.inherited_boundary.to_dict(),
            "replay_of": self.replay_of,
        }


@dataclass(frozen=True, order=True)
class AttributionTotal:
    model: str
    effort: str
    tokens: TokenVector

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "effort": self.effort, "tokens": self.tokens.to_dict()}


@dataclass(frozen=True)
class CodexReconstruction:
    method: str
    totals: TokenVector
    by_model_effort: tuple[AttributionTotal, ...]
    thread_ownership: tuple[ThreadOwnership, ...]
    root_replay_aliases: tuple[RootReplayAlias, ...]
    unresolved_attribution_events: int
    heartbeat_records: int
    duplicate_emissions: int
    reported_total_mismatches: int
    reset_count: int

    @property
    def reconstruction_sha256(self) -> str:
        return canonical_sha256(self.unsigned_dict())

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "totals": self.totals.to_dict(),
            "by_model_effort": [row.to_dict() for row in self.by_model_effort],
            "thread_ownership": [row.to_dict() for row in self.thread_ownership],
            "root_replay_aliases": [row.to_dict() for row in self.root_replay_aliases],
            "unresolved_attribution_events": self.unresolved_attribution_events,
            "heartbeat_records": self.heartbeat_records,
            "duplicate_emissions": self.duplicate_emissions,
            "reported_total_mismatches": self.reported_total_mismatches,
            "reset_count": self.reset_count,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.unsigned_dict()
        result["reconstruction_sha256"] = self.reconstruction_sha256
        return result


def _primary_detect_root_aliases(corpus: CodexCorpus) -> tuple[RootReplayAlias, ...]:
    graph = corpus.graph
    traces = corpus.trace_by_thread
    threads = graph.thread_by_id
    roots = sorted(graph.roots, key=lambda thread_id: (threads[thread_id].created_at_ms, thread_id))
    aliases: dict[str, RootReplayAlias] = {}
    for position, candidate_id in enumerate(roots):
        candidate = traces[candidate_id].events
        if len(candidate) < 3:
            continue
        matches: list[str] = []
        for older_id in roots[:position]:
            if older_id in aliases:
                continue
            older = traces[older_id].events
            if len(candidate) > len(older):
                continue
            matched = True
            for index, event in enumerate(candidate):
                if event.semantic_signature != older[index].semantic_signature:
                    matched = False
                    break
            if matched:
                matches.append(older_id)
        if len(matches) > 1:
            raise CodexAmbiguityError(
                f"root replay {candidate_id} matches multiple older roots: {matches!r}"
            )
        if len(matches) == 1:
            aliases[candidate_id] = RootReplayAlias(
                replay_thread_id=candidate_id,
                canonical_thread_id=matches[0],
                copied_event_count=len(candidate),
            )
    return tuple(sorted(aliases.values()))


def _primary_known_attributions(
    corpus: CodexCorpus,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], set[tuple[str, str]]]:
    result: dict[tuple[tuple[int, ...], tuple[int, ...]], set[tuple[str, str]]] = defaultdict(set)
    for trace in corpus.traces:
        for event in trace.events:
            if event.model is not None and event.effort is not None:
                result[event.semantic_signature].add((event.model, event.effort))
    return result


def _primary_attribution(
    event: CodexTokenEvent,
    known: Mapping[tuple[tuple[int, ...], tuple[int, ...]], set[tuple[str, str]]],
) -> tuple[str, str, bool]:
    if event.model is not None and event.effort is not None:
        return event.model, event.effort, False
    candidates = set(known.get(event.semantic_signature, set()))
    if event.model is not None:
        candidates = {row for row in candidates if row[0] == event.model}
    if event.effort is not None:
        candidates = {row for row in candidates if row[1] == event.effort}
    if len(candidates) > 1:
        raise CodexAmbiguityError(
            f"owned event has ambiguous model attribution: {event.source_ref}"
        )
    if len(candidates) == 1:
        return (*next(iter(candidates)), False)
    return event.model or "<unknown-model>", event.effort or "<unknown-effort>", True


def _primary_copied_prefix(corpus: CodexCorpus, thread_id: str) -> int:
    trace = corpus.trace_by_thread[thread_id]
    if not trace.events:
        return 0
    inferred_start = trace.events[0].total.subtract(trace.events[0].last)
    best = 0
    for ancestor_id in corpus.graph.ancestors(thread_id):
        ancestor = corpus.trace_by_thread[ancestor_id]
        for start, ancestor_event in enumerate(ancestor.events):
            if ancestor_event.semantic_signature != trace.events[0].semantic_signature:
                continue
            preceding = ZERO_TOKENS if start == 0 else ancestor.events[start - 1].total
            if preceding != inferred_start:
                continue
            copied = 0
            while (
                copied < len(trace.events)
                and start + copied < len(ancestor.events)
                and trace.events[copied].semantic_signature
                == ancestor.events[start + copied].semantic_signature
            ):
                copied += 1
            best = max(best, copied)
    return best


def _primary_inherited_boundary(
    corpus: CodexCorpus, thread_id: str, copied: int
) -> TokenVector:
    trace = corpus.trace_by_thread[thread_id]
    if copied:
        return trace.events[copied - 1].total
    if not trace.events:
        return ZERO_TOKENS
    first = trace.events[0]
    baseline = first.total.subtract(first.last)
    ancestors = corpus.graph.ancestors(thread_id)
    if not baseline.is_zero:
        matches = [
            event
            for ancestor_id in ancestors
            for event in corpus.trace_by_thread[ancestor_id].events
            if event.total == baseline
        ]
        if not matches:
            raise CodexAccountingError(
                f"thread {thread_id} starts from an unlocated inherited counter {baseline.as_tuple()}"
            )
    elif not ancestors and first.total != first.last:
        raise CodexAccountingError(f"root {thread_id} does not start from zero")
    return baseline


def _primary_validate_first_owned(
    trace: CodexThreadTrace, copied: int, boundary: TokenVector
) -> None:
    if copied >= len(trace.events):
        return
    event = trace.events[copied]
    if copied and event.epoch_index != trace.events[copied - 1].epoch_index:
        expected = event.total
    else:
        expected = event.total.subtract(boundary)
    if expected != event.last:
        raise CodexAccountingError(
            f"owned suffix boundary does not reproduce last usage: {event.source_ref}"
        )


def reconstruct_dag_owned_suffix(corpus: CodexCorpus) -> CodexReconstruction:
    """Primary reconstruction: select graph-owned suffixes, then sum last usage."""

    aliases = _primary_detect_root_aliases(corpus)
    alias_by_id = {row.replay_thread_id: row for row in aliases}
    known = _primary_known_attributions(corpus)
    totals = ZERO_TOKENS
    by_attribution: dict[tuple[str, str], TokenVector] = defaultdict(TokenVector)
    ownership: list[ThreadOwnership] = []
    unresolved = 0
    for thread_id in corpus.graph.topological_thread_ids():
        trace = corpus.trace_by_thread[thread_id]
        alias = alias_by_id.get(thread_id)
        if alias is not None:
            copied = len(trace.events)
            owned_events: tuple[CodexTokenEvent, ...] = ()
            boundary = trace.events[-1].total if trace.events else ZERO_TOKENS
        else:
            copied = _primary_copied_prefix(corpus, thread_id)
            boundary = _primary_inherited_boundary(corpus, thread_id, copied)
            _primary_validate_first_owned(trace, copied, boundary)
            owned_events = trace.events[copied:]
        ownership.append(
            ThreadOwnership(
                thread_id=thread_id,
                copied_prefix_events=copied,
                owned_events=len(owned_events),
                inherited_boundary=boundary,
                replay_of=None if alias is None else alias.canonical_thread_id,
            )
        )
        for event in owned_events:
            totals = totals + event.last
            model, effort, was_unresolved = _primary_attribution(event, known)
            unresolved += int(was_unresolved)
            by_attribution[(model, effort)] = by_attribution[(model, effort)] + event.last
    attributed = tuple(
        AttributionTotal(model, effort, tokens)
        for (model, effort), tokens in sorted(by_attribution.items())
    )
    return CodexReconstruction(
        method="dag_owned_suffix_last_usage_v1",
        totals=totals,
        by_model_effort=attributed,
        thread_ownership=tuple(sorted(ownership, key=lambda row: row.thread_id)),
        root_replay_aliases=aliases,
        unresolved_attribution_events=unresolved,
        heartbeat_records=corpus.heartbeat_records,
        duplicate_emissions=corpus.duplicate_emissions,
        reported_total_mismatches=corpus.reported_total_mismatches,
        reset_count=corpus.reset_count,
    )


# The epoch/delta implementation below intentionally does not call any of the
# primary alias, prefix, boundary, or attribution helpers above.


def _delta_detect_root_aliases(corpus: CodexCorpus) -> tuple[RootReplayAlias, ...]:
    graph = corpus.graph
    thread_rows = graph.thread_by_id
    traces = corpus.trace_by_thread
    ordered_roots = sorted(
        graph.roots, key=lambda thread_id: (thread_rows[thread_id].created_at_ms, thread_id)
    )
    classified_replays: set[str] = set()
    output: list[RootReplayAlias] = []
    for candidate_position in range(len(ordered_roots)):
        candidate_id = ordered_roots[candidate_position]
        candidate_events = traces[candidate_id].events
        if len(candidate_events) < 3:
            continue
        possible: list[str] = []
        for older_position in range(candidate_position):
            older_id = ordered_roots[older_position]
            if older_id in classified_replays:
                continue
            older_events = traces[older_id].events
            if len(older_events) < len(candidate_events):
                continue
            all_equal = True
            for event_index in range(len(candidate_events)):
                if (
                    candidate_events[event_index].total != older_events[event_index].total
                    or candidate_events[event_index].last != older_events[event_index].last
                ):
                    all_equal = False
                    break
            if all_equal:
                possible.append(older_id)
        if len(possible) > 1:
            raise CodexAmbiguityError(
                f"root replay {candidate_id} matches multiple older roots: {possible!r}"
            )
        if possible:
            classified_replays.add(candidate_id)
            output.append(
                RootReplayAlias(candidate_id, possible[0], len(candidate_events))
            )
    return tuple(sorted(output))


def _delta_duplicate_prefix(corpus: CodexCorpus, thread_id: str) -> int:
    child_events = corpus.trace_by_thread[thread_id].events
    if not child_events:
        return 0
    initial_counter = child_events[0].total.subtract(child_events[0].last)
    maximum = 0
    for ancestor_id in corpus.graph.ancestors(thread_id):
        ancestor_events = corpus.trace_by_thread[ancestor_id].events
        for candidate_start in range(len(ancestor_events)):
            first_candidate = ancestor_events[candidate_start]
            if (
                child_events[0].total != first_candidate.total
                or child_events[0].last != first_candidate.last
            ):
                continue
            prior_counter = (
                ZERO_TOKENS
                if candidate_start == 0
                else ancestor_events[candidate_start - 1].total
            )
            if prior_counter != initial_counter:
                continue
            matched = 0
            while matched < len(child_events) and candidate_start + matched < len(
                ancestor_events
            ):
                left = child_events[matched]
                right = ancestor_events[candidate_start + matched]
                if left.total != right.total or left.last != right.last:
                    break
                matched += 1
            if matched > maximum:
                maximum = matched
    return maximum


def _delta_start_boundary(
    corpus: CodexCorpus, thread_id: str, copied_count: int
) -> TokenVector:
    child = corpus.trace_by_thread[thread_id]
    if copied_count > 0:
        return child.events[copied_count - 1].total
    if not child.events:
        return ZERO_TOKENS
    initial = child.events[0].total.subtract(child.events[0].last)
    ancestor_ids = corpus.graph.ancestors(thread_id)
    if not initial.is_zero:
        found = False
        for ancestor_id in ancestor_ids:
            for ancestor_event in corpus.trace_by_thread[ancestor_id].events:
                if ancestor_event.total == initial:
                    found = True
                    break
            if found:
                break
        if not found:
            raise CodexAccountingError(
                f"thread {thread_id} starts from an unlocated inherited counter {initial.as_tuple()}"
            )
    elif not ancestor_ids and child.events[0].total != child.events[0].last:
        raise CodexAccountingError(f"root {thread_id} does not start from zero")
    return initial


def _delta_attribution_index(
    corpus: CodexCorpus,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], set[tuple[str, str]]]:
    index: dict[tuple[tuple[int, ...], tuple[int, ...]], set[tuple[str, str]]] = {}
    for trace in corpus.traces:
        for event in trace.events:
            if event.model is None or event.effort is None:
                continue
            signature = (event.total.as_tuple(), event.last.as_tuple())
            if signature not in index:
                index[signature] = set()
            index[signature].add((event.model, event.effort))
    return index


def _delta_event_attribution(
    event: CodexTokenEvent,
    index: Mapping[tuple[tuple[int, ...], tuple[int, ...]], set[tuple[str, str]]],
) -> tuple[str, str, bool]:
    if event.model is not None and event.effort is not None:
        return event.model, event.effort, False
    signature = (event.total.as_tuple(), event.last.as_tuple())
    choices = set(index.get(signature, set()))
    if event.model is not None:
        choices = {choice for choice in choices if choice[0] == event.model}
    if event.effort is not None:
        choices = {choice for choice in choices if choice[1] == event.effort}
    if len(choices) > 1:
        raise CodexAmbiguityError(
            f"owned event has ambiguous model attribution: {event.source_ref}"
        )
    if choices:
        model, effort = next(iter(choices))
        return model, effort, False
    return event.model or "<unknown-model>", event.effort or "<unknown-effort>", True


def reconstruct_epoch_deltas(corpus: CodexCorpus) -> CodexReconstruction:
    """Independent reconstruction from cumulative transitions and epoch starts."""

    aliases = _delta_detect_root_aliases(corpus)
    replay_lookup = {alias.replay_thread_id: alias for alias in aliases}
    attribution_index = _delta_attribution_index(corpus)
    grand_total = ZERO_TOKENS
    grouped: dict[tuple[str, str], TokenVector] = {}
    ownership_rows: list[ThreadOwnership] = []
    unresolved = 0
    for thread_id in corpus.graph.topological_thread_ids():
        trace = corpus.trace_by_thread[thread_id]
        replay = replay_lookup.get(thread_id)
        if replay is not None:
            duplicate_count = len(trace.events)
            boundary = trace.events[-1].total if trace.events else ZERO_TOKENS
            owned_indexes = range(len(trace.events), len(trace.events))
        else:
            duplicate_count = _delta_duplicate_prefix(corpus, thread_id)
            boundary = _delta_start_boundary(corpus, thread_id, duplicate_count)
            owned_indexes = range(duplicate_count, len(trace.events))
        ownership_rows.append(
            ThreadOwnership(
                thread_id=thread_id,
                copied_prefix_events=duplicate_count,
                owned_events=len(trace.events) - duplicate_count,
                inherited_boundary=boundary,
                replay_of=None if replay is None else replay.canonical_thread_id,
            )
        )
        for event_index in owned_indexes:
            event = trace.events[event_index]
            if event_index == duplicate_count:
                if event_index > 0 and event.epoch_index != trace.events[event_index - 1].epoch_index:
                    previous_total = ZERO_TOKENS
                else:
                    previous_total = boundary
            elif event.epoch_index != trace.events[event_index - 1].epoch_index:
                previous_total = ZERO_TOKENS
            else:
                previous_total = trace.events[event_index - 1].total
            increment = event.total.subtract(previous_total)
            if increment != event.last:
                raise CodexAccountingError(
                    f"epoch delta does not reproduce last usage: {event.source_ref}"
                )
            grand_total = grand_total + increment
            model, effort, missing = _delta_event_attribution(event, attribution_index)
            unresolved += int(missing)
            key = (model, effort)
            grouped[key] = grouped.get(key, ZERO_TOKENS) + increment
    breakdown = tuple(
        AttributionTotal(model, effort, tokens)
        for (model, effort), tokens in sorted(grouped.items())
    )
    return CodexReconstruction(
        method="epoch_cumulative_delta_v1",
        totals=grand_total,
        by_model_effort=breakdown,
        thread_ownership=tuple(sorted(ownership_rows, key=lambda row: row.thread_id)),
        root_replay_aliases=aliases,
        unresolved_attribution_events=unresolved,
        heartbeat_records=corpus.heartbeat_records,
        duplicate_emissions=corpus.duplicate_emissions,
        reported_total_mismatches=corpus.reported_total_mismatches,
        reset_count=corpus.reset_count,
    )


@dataclass(frozen=True)
class CodexAgreement:
    totals: TokenVector
    by_model_effort: tuple[AttributionTotal, ...]
    primary_sha256: str
    independent_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "exact_agreement",
            "totals": self.totals.to_dict(),
            "by_model_effort": [row.to_dict() for row in self.by_model_effort],
            "primary_sha256": self.primary_sha256,
            "independent_sha256": self.independent_sha256,
        }


def assert_exact_reconstruction_agreement(
    primary: CodexReconstruction, independent: CodexReconstruction
) -> CodexAgreement:
    """Fail unless both algorithms agree on every additive and ownership field."""

    comparisons = {
        "totals": (primary.totals, independent.totals),
        "by_model_effort": (primary.by_model_effort, independent.by_model_effort),
        "thread_ownership": (primary.thread_ownership, independent.thread_ownership),
        "root_replay_aliases": (primary.root_replay_aliases, independent.root_replay_aliases),
        "unresolved_attribution_events": (
            primary.unresolved_attribution_events,
            independent.unresolved_attribution_events,
        ),
        "heartbeat_records": (primary.heartbeat_records, independent.heartbeat_records),
        "duplicate_emissions": (primary.duplicate_emissions, independent.duplicate_emissions),
        "reported_total_mismatches": (
            primary.reported_total_mismatches,
            independent.reported_total_mismatches,
        ),
        "reset_count": (primary.reset_count, independent.reset_count),
    }
    disagreements = [name for name, (left, right) in comparisons.items() if left != right]
    if disagreements:
        raise CodexAccountingError(
            f"Codex reconstruction methods disagree on: {', '.join(disagreements)}"
        )
    return CodexAgreement(
        totals=primary.totals,
        by_model_effort=primary.by_model_effort,
        primary_sha256=primary.reconstruction_sha256,
        independent_sha256=independent.reconstruction_sha256,
    )
