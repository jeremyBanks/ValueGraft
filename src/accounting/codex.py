"""Codex subscription-workload reconstruction from frozen rollout logs.

Codex token counters are cumulative within a thread, copied into forked
threads, and occasionally reset.  SQLite ``tokens_used`` and global tuple
deduplication are therefore non-additive.  Version 3 reconstructs ownership by
two distinct structural boundaries:

* an ordered four-counter prefix on an explicit spawn ancestry;
* the direct parent's last pre-creation token state.

Graph copies are classified as reconstructed, never source-proven, and retain
a no-graph-collapse upper sensitivity.  Fully comparable normalized response
payloads can confirm or reject a structural candidate; incomplete payloads
fall back to the structural witness.  Unlinked root replay collapse separately
requires stable response-item identities.  Final reconstructed status also
requires a quiescent response-to-usage coverage and owned-identity uniqueness
gate.
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .core import AccountingError, FrozenJsonRecord, canonical_sha256


class CodexAccountingError(AccountingError):
    """Base error for malformed or unreconstructable Codex usage."""


class CodexAmbiguityError(CodexAccountingError):
    """Raised when copied history or a replay alias is not uniquely identified."""


def _opaque_identifier(kind: str, raw: str) -> str:
    digest = hashlib.sha256(f"codex-accounting-v3\0{kind}\0{raw}".encode()).hexdigest()
    return f"{kind}_{digest}"


def _public_thread_id(raw: str) -> str:
    return _opaque_identifier("thread", raw)


def _public_rollout_path(thread_id: str) -> str:
    return f"rollouts/{_public_thread_id(thread_id)}.jsonl"


def _source_category(source: str, thread_source: str | None) -> str:
    """Normalize private SQLite source metadata to a public allowlist."""
    if thread_source == "subagent" or source == "subagent" or source.lstrip().startswith("{"):
        return "subagent"
    if source == "exec":
        return "cli_exec"
    if source == "vscode":
        return "codex_app"
    return "other"


def _history_mode_category(history_mode: str | None) -> str:
    return history_mode if history_mode in {"legacy"} else "other"


def _edge_status_category(status: str) -> str:
    return status if status in {"open", "closed"} else "other"


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
            "thread_id": _public_thread_id(self.thread_id),
            "logical_rollout_path": _public_rollout_path(self.thread_id),
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
            "source_category": _source_category(self.source, self.thread_source),
            "model_provider": self.model_provider,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "history_mode_category": _history_mode_category(self.history_mode),
            "project_location": (
                "project_root" if self.project_relative_cwd == "." else "project_subdirectory"
            ),
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
            "parent_thread_id": _public_thread_id(self.parent_thread_id),
            "child_thread_id": _public_thread_id(self.child_thread_id),
            "status_category": _edge_status_category(self.status),
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
            "schema": "codex_sanitized_thread_graph_v3",
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
class ResponseEvidence:
    """Private, content-free evidence that a model response was completed."""

    kind: str
    digest: str
    stable_source_identity: bool
    normalized_payload_digest: str

    def __post_init__(self) -> None:
        digests = (self.digest, self.normalized_payload_digest)
        if not self.kind or any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        ):
            raise CodexAccountingError("response evidence is malformed")


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
    response_evidence: tuple[ResponseEvidence, ...] = ()

    @property
    def counter_signature(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self.total.as_tuple(), self.last.as_tuple()

    @property
    def semantic_signature(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Backward-compatible name; this is only a counter signature."""
        return self.counter_signature


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
    explicit_zero_reset_markers: int
    stable_source_identity_stream: tuple[str, ...]
    pending_response_evidence: tuple[ResponseEvidence, ...]


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

    @property
    def explicit_zero_reset_markers(self) -> int:
        return sum(trace.explicit_zero_reset_markers for trace in self.traces)


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
    expected_fields = {
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "total_tokens",
    }
    if set(value) != expected_fields:
        raise CodexAccountingError(f"{label} token field set differs: {sorted(value)!r}")
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


_MODEL_RESPONSE_ITEM_TYPES = {
    "reasoning",
    "function_call",
    "custom_tool_call",
    "web_search_call",
    "tool_search_call",
    "computer_call",
    "image_generation_call",
    "agent_message",
}
_KNOWN_RESPONSE_ITEM_TYPES = _MODEL_RESPONSE_ITEM_TYPES | {
    "message",
    "function_call_output",
    "custom_tool_call_output",
    "tool_search_output",
}


def _private_payload_digest(value: Mapping[str, Any]) -> str:
    """Hash private response structure without serializing it into outputs."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_response_payload_digest(payload: Mapping[str, Any]) -> str:
    """Hash semantic payload fields while stripping only volatile envelope IDs."""
    normalized = {
        key: value for key, value in payload.items() if key not in {"id", "call_id"}
    }
    return _private_payload_digest(normalized)


def _response_item_evidence(record: Mapping[str, Any], label: str) -> tuple[
    ResponseEvidence | None, str | None
]:
    """Return model coverage evidence and optional stable source identity."""
    payload = record.get("payload")
    if not isinstance(payload, Mapping):
        raise CodexAccountingError(f"response_item payload is malformed at {label}")
    kind = payload.get("type")
    if not isinstance(kind, str) or kind not in _KNOWN_RESPONSE_ITEM_TYPES:
        raise CodexAccountingError(f"unknown response_item type at {label}: {kind!r}")
    for identity_field in ("id", "call_id"):
        identity_value = payload.get(identity_field)
        if identity_value is not None and (
            not isinstance(identity_value, str) or not identity_value
        ):
            raise CodexAccountingError(
                f"response_item {identity_field} is malformed at {label}"
            )
    raw_identity = payload.get("id") or payload.get("call_id")
    stable: str | None = None
    if isinstance(raw_identity, str) and raw_identity:
        stable = hashlib.sha256(
            f"codex-source-response-v1\0{kind}\0{raw_identity}".encode("utf-8")
        ).hexdigest()
    is_model = kind in _MODEL_RESPONSE_ITEM_TYPES or (
        kind == "message" and payload.get("role") == "assistant"
    )
    if not is_model:
        return None, stable
    digest = stable or _private_payload_digest(payload)
    return ResponseEvidence(
        kind,
        digest,
        stable is not None,
        _normalized_response_payload_digest(payload),
    ), stable


def parse_frozen_rollouts(
    records: Iterable[FrozenJsonRecord], graph: CodexGraph
) -> CodexCorpus:
    """Parse token states plus content-free source and coverage evidence."""

    thread_for_path = graph.thread_id_by_path
    events: dict[str, list[CodexTokenEvent]] = {thread.thread_id: [] for thread in graph.threads}
    contexts: dict[str, tuple[str | None, str | None]] = {
        thread.thread_id: (None, None) for thread in graph.threads
    }
    meta_seen: set[str] = set()
    heartbeats: dict[str, int] = defaultdict(int)
    duplicates: dict[str, int] = defaultdict(int)
    reported_total_mismatches: dict[str, int] = defaultdict(int)
    explicit_zero_resets: dict[str, int] = defaultdict(int)
    pending_reset: dict[str, bool] = defaultdict(bool)
    last_raw_total: dict[str, TokenVector | None] = {
        thread.thread_id: None for thread in graph.threads
    }
    pending_responses: dict[str, list[ResponseEvidence]] = defaultdict(list)
    source_identity_streams: dict[str, list[str]] = defaultdict(list)
    last_line_by_path: dict[str, int] = {}
    for source in records:
        if (
            isinstance(source.line_number, bool)
            or not isinstance(source.line_number, int)
            or source.line_number <= last_line_by_path.get(source.logical_path, 0)
        ):
            raise CodexAccountingError(
                f"frozen rollout records are not strictly line-ordered: {source.logical_path}"
            )
        last_line_by_path[source.logical_path] = source.line_number
        thread_id = thread_for_path.get(source.logical_path)
        if thread_id is None:
            raise CodexAccountingError(
                f"frozen rollout is absent from sanitized graph: {source.logical_path}"
            )
        record = source.value
        if not isinstance(record, Mapping):
            raise CodexAccountingError(
                f"frozen rollout record is not an object: {source.logical_path}:{source.line_number}"
            )
        payload = record.get("payload")
        label = f"{source.logical_path}:{source.line_number}"
        if record.get("type") == "session_meta":
            # A physical fork starts with its own session_meta, then may contain
            # copied ancestor session_meta records.  Only the first one names
            # the file itself; validating every copied record against the child
            # ID rejects legitimate fork histories.
            if thread_id not in meta_seen:
                if not isinstance(payload, Mapping) or str(payload.get("id")) != thread_id:
                    raise CodexAccountingError(f"initial session_meta ID mismatch at {label}")
                meta_seen.add(thread_id)
            continue
        if record.get("type") == "response_item":
            response, stable_identity = _response_item_evidence(record, label)
            if response is not None and response not in pending_responses[thread_id]:
                pending_responses[thread_id].append(response)
            if stable_identity is not None:
                source_identity_streams[thread_id].append(stable_identity)
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
        last = _parse_vector(
            info.get("last_token_usage"),
            f"{label}.last",
            allow_total_only_heartbeat=True,
        )
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
        previous_raw = last_raw_total[thread_id]
        if last.is_zero:
            heartbeats[thread_id] += 1
            if previous_raw is None:
                if not total.is_zero:
                    raise CodexAccountingError(
                        f"initial zero-last record has a nonzero cumulative total at {label}"
                    )
            elif total == previous_raw:
                pass
            elif total.is_zero and not previous_raw.is_zero:
                explicit_zero_resets[thread_id] += 1
                pending_reset[thread_id] = True
            else:
                raise CodexAccountingError(
                    f"zero-last record changes cumulative counters ambiguously at {label}"
                )
            last_raw_total[thread_id] = total
            continue
        prior = events[thread_id][-1] if events[thread_id] else None
        model, effort = contexts[thread_id]
        if prior is not None and total == prior.total and not pending_reset[thread_id]:
            if last != prior.last:
                raise CodexAmbiguityError(
                    f"unchanged cumulative state has different nonzero last usage at {label}"
                )
            if prior.model and model and prior.model != model:
                raise CodexAmbiguityError(f"duplicate emission changes model at {label}")
            if prior.effort and effort and prior.effort != effort:
                raise CodexAmbiguityError(f"duplicate emission changes effort at {label}")
            pending = tuple(pending_responses[thread_id])
            if pending:
                prior_counts = Counter(prior.response_evidence)
                seen_counts: Counter[ResponseEvidence] = Counter()
                novel: list[ResponseEvidence] = []
                for evidence in pending:
                    seen_counts[evidence] += 1
                    if seen_counts[evidence] > prior_counts[evidence]:
                        novel.append(evidence)
                # An unchanged counter is stale telemetry, not evidence that
                # the newly completed model items belong to the prior usage
                # event.  Drop only demonstrable re-emissions; retain genuinely
                # new items for the next advancing usage record.  If no such
                # record arrives, the EOF coverage gate fails closed.
                pending_responses[thread_id][:] = novel
            if not pending_responses[thread_id]:
                events[thread_id][-1] = replace(
                    prior,
                    model=prior.model or model,
                    effort=prior.effort or effort,
                )
            duplicates[thread_id] += 1
            last_raw_total[thread_id] = total
            continue
        if pending_reset[thread_id]:
            if total != last:
                raise CodexAccountingError(
                    f"explicit reset does not restart at last usage at {label}"
                )
            epoch_index = 0 if prior is None else prior.epoch_index + 1
            pending_reset[thread_id] = False
        elif prior is None:
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
                response_evidence=tuple(pending_responses[thread_id]),
            )
        )
        pending_responses[thread_id].clear()
        last_raw_total[thread_id] = total
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
                explicit_zero_reset_markers=explicit_zero_resets[thread.thread_id],
                stable_source_identity_stream=tuple(
                    source_identity_streams[thread.thread_id]
                ),
                pending_response_evidence=tuple(pending_responses[thread.thread_id]),
            )
        )
    return CodexCorpus(graph, tuple(sorted(traces, key=lambda trace: trace.thread_id)))


@dataclass(frozen=True, order=True)
class RootReplayAlias:
    replay_thread_id: str
    canonical_thread_id: str
    copied_event_count: int
    owned_event_count: int
    source_identity_prefix_count: int
    source_identity_prefix_sha256: str
    copied_tokens: TokenVector

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_thread_id": _public_thread_id(self.replay_thread_id),
            "canonical_thread_id": _public_thread_id(self.canonical_thread_id),
            "evidence_classification": "stable_source_identity_prefix",
            "copied_event_count": self.copied_event_count,
            "owned_event_count": self.owned_event_count,
            "source_identity_prefix_count": self.source_identity_prefix_count,
            "source_identity_prefix_sha256": self.source_identity_prefix_sha256,
            "copied_tokens": self.copied_tokens.to_dict(),
        }


@dataclass(frozen=True, order=True)
class RootReplaySensitivity:
    candidate_thread_id: str
    older_thread_id: str
    potential_copied_event_count: int
    potential_duplicate_tokens: TokenVector
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_thread_id": _public_thread_id(self.candidate_thread_id),
            "older_thread_id": _public_thread_id(self.older_thread_id),
            "potential_copied_event_count": self.potential_copied_event_count,
            "potential_duplicate_tokens": self.potential_duplicate_tokens.to_dict(),
            "classification": "unknown_not_collapsed",
            "reason": self.reason,
        }


@dataclass(frozen=True, order=True)
class StructuralGraphCopy:
    child_thread_id: str
    parent_thread_id: str
    copied_event_count: int
    copied_tokens: TokenVector
    payload_evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "child_thread_id": _public_thread_id(self.child_thread_id),
            "parent_thread_id": _public_thread_id(self.parent_thread_id),
            "copied_event_count": self.copied_event_count,
            "copied_tokens": self.copied_tokens.to_dict(),
            "measurement_class": "reconstructed",
            "evidence_classification": (
                "spawn_edge_full_ordered_four_counter_sequence_ending_at_"
                "direct_parent_precreation_state"
            ),
            "source_identity_proven": False,
            "payload_evidence": self.payload_evidence,
            "nonidentifiability": (
                "an_independent_child_with_the_same_counter_sequence_cannot_be_excluded"
            ),
        }


@dataclass(frozen=True, order=True)
class RejectedGraphCopyCandidate:
    child_thread_id: str
    parent_thread_id: str
    candidate_event_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "child_thread_id": _public_thread_id(self.child_thread_id),
            "parent_thread_id": _public_thread_id(self.parent_thread_id),
            "candidate_event_count": self.candidate_event_count,
            "classification": "complete_normalized_payload_mismatch_owned",
        }


@dataclass(frozen=True, order=True)
class OwnedEvidenceCollision:
    evidence_kind: str
    evidence_digest: str
    locations: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_kind": self.evidence_kind,
            "evidence_digest": self.evidence_digest,
            "occurrence_count": len(self.locations),
            "locations": [
                {
                    "thread_id": _public_thread_id(thread_id),
                    "trace_event_ordinal": ordinal,
                }
                for thread_id, ordinal in self.locations
            ],
            "classification": "owned_stable_response_identity_collision",
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
            "thread_id": _public_thread_id(self.thread_id),
            "copied_prefix_events": self.copied_prefix_events,
            "owned_events": self.owned_events,
            "inherited_boundary": self.inherited_boundary.to_dict(),
            "replay_of": (
                None if self.replay_of is None else _public_thread_id(self.replay_of)
            ),
        }


@dataclass(frozen=True, order=True)
class AttributionTotal:
    model: str
    effort: str
    tokens: TokenVector

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "effort": self.effort, "tokens": self.tokens.to_dict()}


@dataclass(frozen=True)
class CodexCoverage:
    status: str
    owned_usage_events: int
    covered_usage_events: int
    uncovered_usage_events: int
    pending_response_items: int
    pending_thread_count: int

    @property
    def complete_coverage(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "owned_usage_events": self.owned_usage_events,
            "covered_usage_events": self.covered_usage_events,
            "uncovered_usage_events": self.uncovered_usage_events,
            "pending_response_items": self.pending_response_items,
            "pending_thread_count": self.pending_thread_count,
            "gate": "completed_model_response_items_to_owned_token_usage_v1",
        }


@dataclass(frozen=True)
class CodexReconstruction:
    method: str
    totals: TokenVector
    by_model_effort: tuple[AttributionTotal, ...]
    thread_ownership: tuple[ThreadOwnership, ...]
    root_replay_aliases: tuple[RootReplayAlias, ...]
    unresolved_root_replay_sensitivities: tuple[RootReplaySensitivity, ...]
    structural_graph_copies: tuple[StructuralGraphCopy, ...]
    rejected_graph_copy_candidates: tuple[RejectedGraphCopyCandidate, ...]
    owned_stable_evidence_collisions: tuple[OwnedEvidenceCollision, ...]
    no_root_collapse_totals: TokenVector
    possible_root_collapse_lower_bound: TokenVector
    no_graph_collapse_upper_bound: TokenVector
    coverage: CodexCoverage
    unresolved_attribution_events: int
    heartbeat_records: int
    duplicate_emissions: int
    reported_total_mismatches: int
    reset_count: int
    explicit_zero_reset_markers: int

    @property
    def reconstructed_total_authorized(self) -> bool:
        return (
            self.coverage.complete_coverage
            and not self.unresolved_root_replay_sensitivities
            and not self.owned_stable_evidence_collisions
            and self.unresolved_attribution_events == 0
        )

    @property
    def reconstruction_sha256(self) -> str:
        return canonical_sha256(self.unsigned_dict())

    def unsigned_dict(self) -> dict[str, Any]:
        graph_lengths = [row.copied_event_count for row in self.structural_graph_copies]
        graph_tokens = _sum_structural_graph_copy_tokens(self.structural_graph_copies)
        return {
            "measurement_class": "reconstructed",
            "method": self.method,
            "totals": self.totals.to_dict(),
            "by_model_effort": [row.to_dict() for row in self.by_model_effort],
            "thread_ownership": [row.to_dict() for row in self.thread_ownership],
            "root_replay_aliases": [row.to_dict() for row in self.root_replay_aliases],
            "unresolved_root_replay_sensitivities": [
                row.to_dict() for row in self.unresolved_root_replay_sensitivities
            ],
            "structural_graph_copies": [
                row.to_dict() for row in self.structural_graph_copies
            ],
            "structural_graph_copy_summary": {
                "measurement_class": "reconstructed",
                "source_identity_proven": False,
                "copy_count": len(graph_lengths),
                "copied_event_count": sum(graph_lengths),
                "minimum_prefix_events": min(graph_lengths) if graph_lengths else 0,
                "maximum_prefix_events": max(graph_lengths) if graph_lengths else 0,
                "copied_tokens": graph_tokens.to_dict(),
                "payload_confirmed_copy_count": sum(
                    row.payload_evidence == "complete_normalized_payload_equal"
                    for row in self.structural_graph_copies
                ),
                "payload_incomplete_copy_count": sum(
                    row.payload_evidence == "incomplete_structural_fallback"
                    for row in self.structural_graph_copies
                ),
                "payload_rejected_candidate_count": len(
                    self.rejected_graph_copy_candidates
                ),
            },
            "rejected_graph_copy_candidates": [
                row.to_dict() for row in self.rejected_graph_copy_candidates
            ],
            "owned_stable_evidence_collisions": [
                row.to_dict() for row in self.owned_stable_evidence_collisions
            ],
            "no_root_collapse_totals": self.no_root_collapse_totals.to_dict(),
            "possible_root_collapse_lower_bound": (
                self.possible_root_collapse_lower_bound.to_dict()
            ),
            "no_graph_collapse_upper_bound": self.no_graph_collapse_upper_bound.to_dict(),
            "coverage": self.coverage.to_dict(),
            "reconstructed_total_authorized": self.reconstructed_total_authorized,
            "unresolved_attribution_events": self.unresolved_attribution_events,
            "heartbeat_records": self.heartbeat_records,
            "duplicate_emissions": self.duplicate_emissions,
            "reported_total_mismatches": self.reported_total_mismatches,
            "reset_count": self.reset_count,
            "explicit_zero_reset_markers": self.explicit_zero_reset_markers,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.unsigned_dict()
        result["reconstruction_sha256"] = self.reconstruction_sha256
        return result


def _sum_event_last(events: Iterable[CodexTokenEvent]) -> TokenVector:
    total = ZERO_TOKENS
    for event in events:
        total = total + event.last
    return total


def _common_counter_prefix(
    left: tuple[CodexTokenEvent, ...], right: tuple[CodexTokenEvent, ...]
) -> int:
    count = 0
    while (
        count < len(left)
        and count < len(right)
        and left[count].counter_signature == right[count].counter_signature
    ):
        count += 1
    return count


def _stable_event_identities(event: CodexTokenEvent) -> tuple[str, ...]:
    """Stable model-response identities attached to exactly one usage state."""
    return tuple(
        evidence.digest
        for evidence in event.response_evidence
        if evidence.stable_source_identity
    )


def _event_payload_sequence(event: CodexTokenEvent) -> tuple[tuple[str, str], ...]:
    return tuple(
        (evidence.kind, evidence.normalized_payload_digest)
        for evidence in event.response_evidence
    )


def _compare_complete_payload_segments(
    candidate: tuple[CodexTokenEvent, ...], source: tuple[CodexTokenEvent, ...]
) -> str:
    """Return confirmed, rejected, or incomplete payload evidence."""
    if len(candidate) != len(source):
        raise CodexAccountingError("payload segments have different lengths")
    pairs = [
        (_event_payload_sequence(left), _event_payload_sequence(right))
        for left, right in zip(candidate, source)
    ]
    if not all(left and right for left, right in pairs):
        return "incomplete_structural_fallback"
    if all(left == right for left, right in pairs):
        return "complete_normalized_payload_equal"
    return "complete_normalized_payload_mismatch"


def _source_proven_counter_prefix(
    candidate: tuple[CodexTokenEvent, ...], older: tuple[CodexTokenEvent, ...]
) -> tuple[int, int, int, bool]:
    """Return proven events/IDs, counter prefix, and missing-evidence flag.

    Token counters only bound the comparison.  Each event that is actually
    classified as copied must carry the same nonempty stable response identity
    sequence in both sources.  A missing sequence is uncertainty; two present
    but different sequences are affirmative evidence of independent work.
    """

    counter_prefix = _common_counter_prefix(candidate, older)
    proven_events, proven_identities, missing_evidence = _identity_proven_prefix(
        candidate, older, counter_prefix
    )
    return proven_events, proven_identities, counter_prefix, missing_evidence


def _identity_proven_prefix(
    candidate: tuple[CodexTokenEvent, ...],
    source: tuple[CodexTokenEvent, ...],
    limit: int,
) -> tuple[int, int, bool]:
    """Prove a source prefix without using token counters as identity."""

    if limit < 0 or limit > len(candidate) or limit > len(source):
        raise CodexAccountingError("identity-prefix limit is outside its source streams")
    proven_events = 0
    proven_identities = 0
    missing_evidence = False
    for index in range(limit):
        left = _stable_event_identities(candidate[index])
        right = _stable_event_identities(source[index])
        if not left or not right:
            missing_evidence = True
            break
        if left != right:
            break
        proven_events += 1
        proven_identities += len(left)
    return proven_events, proven_identities, missing_evidence


def _primary_detect_root_aliases(
    corpus: CodexCorpus,
) -> tuple[tuple[RootReplayAlias, ...], tuple[RootReplaySensitivity, ...]]:
    """Classify roots only with stable source-response identity evidence."""
    graph = corpus.graph
    traces = corpus.trace_by_thread
    threads = graph.thread_by_id
    roots = sorted(graph.roots, key=lambda thread_id: (threads[thread_id].created_at_ms, thread_id))
    aliases: dict[str, RootReplayAlias] = {}
    sensitivities: list[RootReplaySensitivity] = []
    for position, candidate_id in enumerate(roots):
        candidate_trace = traces[candidate_id]
        evidence_matches: list[tuple[int, int, int, bool, str]] = []
        unresolved_matches: list[tuple[int, str]] = []
        for older_id in roots[:position]:
            # Partial replays remain real sources for their owned suffixes.  A
            # later root can replay that suffix too, so excluding aliases here
            # would count the intermediate suffix once in B and again in C.
            older_trace = traces[older_id]
            proven_events, proven_ids, counter_prefix, missing = (
                _source_proven_counter_prefix(
                candidate_trace.events, older_trace.events
                )
            )
            if counter_prefix == 0:
                continue
            if proven_events:
                evidence_matches.append(
                    (proven_events, proven_ids, counter_prefix, missing, older_id)
                )
            elif missing:
                unresolved_matches.append((counter_prefix, older_id))
        if evidence_matches:
            if unresolved_matches:
                raise CodexAmbiguityError(
                    f"source-proven root replay {candidate_id} also has a counter-only root match"
                )
            evidence_matches.sort(reverse=True)
            best_events, best_ids, counter_prefix, missing, older_id = evidence_matches[0]
            tied = [
                row for row in evidence_matches
                if row[:2] == (best_events, best_ids)
            ]
            if len(tied) != 1:
                raise CodexAmbiguityError(
                    f"source-proven root replay {candidate_id} matches multiple roots"
                )
            copied_tokens = _sum_event_last(candidate_trace.events[:best_events])
            stable_prefix = tuple(
                identity
                for event in candidate_trace.events[:best_events]
                for identity in _stable_event_identities(event)
            )
            aliases[candidate_id] = RootReplayAlias(
                replay_thread_id=candidate_id,
                canonical_thread_id=older_id,
                copied_event_count=best_events,
                owned_event_count=len(candidate_trace.events) - best_events,
                source_identity_prefix_count=best_ids,
                source_identity_prefix_sha256=canonical_sha256(list(stable_prefix)),
                copied_tokens=copied_tokens,
            )
            if missing and counter_prefix > best_events:
                sensitivities.append(
                    RootReplaySensitivity(
                        candidate_thread_id=candidate_id,
                        older_thread_id=older_id,
                        potential_copied_event_count=counter_prefix - best_events,
                        potential_duplicate_tokens=_sum_event_last(
                            candidate_trace.events[best_events:counter_prefix]
                        ),
                        reason="counter_prefix_continues_after_stable_source_evidence_ends",
                    )
                )
            continue
        if len(unresolved_matches) > 1:
            raise CodexAmbiguityError(
                f"counter-only root replay candidate {candidate_id} matches multiple roots"
            )
        if unresolved_matches:
            copied_events, older_id = unresolved_matches[0]
            sensitivities.append(
                RootReplaySensitivity(
                    candidate_thread_id=candidate_id,
                    older_thread_id=older_id,
                    potential_copied_event_count=copied_events,
                    potential_duplicate_tokens=_sum_event_last(
                        candidate_trace.events[:copied_events]
                    ),
                    reason="counter_prefix_without_stable_source_identity",
                )
            )
    return tuple(sorted(aliases.values())), tuple(sorted(sensitivities))


def _event_attribution(event: CodexTokenEvent) -> tuple[str, str, bool]:
    """Attribute only from the event's own turn context.

    Counter equality is not provenance: independently generated calls can have
    identical cumulative/last vectors.  Missing direct context therefore stays
    explicitly unknown and prevents reconstructed authorization.
    """
    if event.model is not None and event.effort is not None:
        return event.model, event.effort, False
    return event.model or "<unknown-model>", event.effort or "<unknown-effort>", True


def _structural_graph_copy(
    corpus: CodexCorpus, thread_id: str, copied: int, payload_evidence: str
) -> StructuralGraphCopy:
    parent_id = corpus.graph.parent_by_child.get(thread_id)
    if parent_id is None or copied <= 0:
        raise CodexAccountingError("invalid structural graph-copy boundary")
    trace = corpus.trace_by_thread[thread_id]
    return StructuralGraphCopy(
        child_thread_id=thread_id,
        parent_thread_id=parent_id,
        copied_event_count=copied,
        copied_tokens=_sum_event_last(trace.events[:copied]),
        payload_evidence=payload_evidence,
    )


def _rejected_graph_copy_candidate(
    corpus: CodexCorpus, thread_id: str, candidate_events: int
) -> RejectedGraphCopyCandidate:
    parent_id = corpus.graph.parent_by_child.get(thread_id)
    if parent_id is None or candidate_events <= 0:
        raise CodexAccountingError("invalid rejected graph-copy candidate")
    return RejectedGraphCopyCandidate(thread_id, parent_id, candidate_events)


def _primary_copied_prefix(
    corpus: CodexCorpus, thread_id: str
) -> tuple[int, StructuralGraphCopy | None, RejectedGraphCopyCandidate | None]:
    trace = corpus.trace_by_thread[thread_id]
    if not trace.events:
        return 0, None, None
    parent_id = corpus.graph.parent_by_child.get(thread_id)
    if parent_id is None:
        return 0, None, None
    parent_event = _direct_parent_precreation_event(corpus, thread_id)
    if parent_event is None:
        return 0, None, None
    parent_events = corpus.trace_by_thread[parent_id].events
    inferred_start = trace.events[0].total.subtract(trace.events[0].last)
    accepted: dict[int, str] = {}
    rejected: set[int] = set()
    for ancestor_id in corpus.graph.ancestors(thread_id):
        ancestor = corpus.trace_by_thread[ancestor_id]
        for start, ancestor_event in enumerate(ancestor.events):
            if ancestor_event.counter_signature != trace.events[0].counter_signature:
                continue
            preceding = (
                ZERO_TOKENS
                if start == 0
                or ancestor_event.epoch_index != ancestor.events[start - 1].epoch_index
                else ancestor.events[start - 1].total
            )
            if preceding != inferred_start:
                continue
            copied = 0
            while (
                copied < len(trace.events)
                and start + copied < len(ancestor.events)
                and trace.events[copied].counter_signature
                == ancestor.events[start + copied].counter_signature
            ):
                copied += 1
            if not copied:
                continue
            if trace.events[copied - 1].counter_signature != parent_event.counter_signature:
                continue
            parent_start = parent_event.ordinal - copied + 1
            if parent_start < 0:
                continue
            parent_segment = parent_events[parent_start : parent_event.ordinal + 1]
            if len(parent_segment) != copied or any(
                trace.events[index].counter_signature
                != parent_segment[index].counter_signature
                for index in range(copied)
            ):
                continue
            payload_evidence = _compare_complete_payload_segments(
                trace.events[:copied], parent_segment
            )
            if payload_evidence == "complete_normalized_payload_mismatch":
                rejected.add(copied)
            else:
                accepted[copied] = payload_evidence
    if accepted and rejected:
        raise CodexAmbiguityError(
            f"thread {thread_id} has accepted and payload-rejected structural candidates"
        )
    if len(accepted) > 1 or len(rejected) > 1:
        raise CodexAmbiguityError(
            f"thread {thread_id} has multiple structural copied-prefix lengths"
        )
    if accepted:
        copied, payload_evidence = next(iter(accepted.items()))
        return (
            copied,
            _structural_graph_copy(corpus, thread_id, copied, payload_evidence),
            None,
        )
    if rejected:
        candidate_events = next(iter(rejected))
        return (
            0,
            None,
            _rejected_graph_copy_candidate(corpus, thread_id, candidate_events),
        )
    return 0, None, None


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


def _coverage_for_ownership(
    corpus: CodexCorpus, ownership: Iterable[ThreadOwnership]
) -> CodexCoverage:
    """Require a completed model response on both sides of every usage edge."""

    ownership_by_thread = {row.thread_id: row for row in ownership}
    if set(ownership_by_thread) != set(corpus.trace_by_thread):
        raise CodexAccountingError("coverage ownership does not span the frozen corpus")
    owned = 0
    covered = 0
    for thread_id, trace in corpus.trace_by_thread.items():
        copied = ownership_by_thread[thread_id].copied_prefix_events
        if copied < 0 or copied > len(trace.events):
            raise CodexAccountingError(f"invalid copied-prefix count for {thread_id}")
        for event in trace.events[copied:]:
            owned += 1
            covered += int(bool(event.response_evidence))
    pending_by_thread = {
        trace.thread_id: len(trace.pending_response_evidence)
        for trace in corpus.traces
        if trace.pending_response_evidence
    }
    uncovered = owned - covered
    pending = sum(pending_by_thread.values())
    if not uncovered and not pending:
        status = "PASS"
    elif uncovered and pending:
        status = "FAIL_UNCOVERED_USAGE_AND_PENDING_RESPONSES"
    elif uncovered:
        status = "FAIL_UNCOVERED_USAGE"
    else:
        status = "FAIL_PENDING_RESPONSES"
    return CodexCoverage(
        status=status,
        owned_usage_events=owned,
        covered_usage_events=covered,
        uncovered_usage_events=uncovered,
        pending_response_items=pending,
        pending_thread_count=len(pending_by_thread),
    )


def _sum_alias_tokens(aliases: Iterable[RootReplayAlias]) -> TokenVector:
    total = ZERO_TOKENS
    for alias in aliases:
        total = total + alias.copied_tokens
    return total


def _sum_sensitivity_tokens(
    sensitivities: Iterable[RootReplaySensitivity],
) -> TokenVector:
    total = ZERO_TOKENS
    for sensitivity in sensitivities:
        total = total + sensitivity.potential_duplicate_tokens
    return total


def _sum_structural_graph_copy_tokens(
    copies: Iterable[StructuralGraphCopy],
) -> TokenVector:
    total = ZERO_TOKENS
    for copy in copies:
        total = total + copy.copied_tokens
    return total


def _owned_stable_evidence_collisions(
    corpus: CodexCorpus, ownership: Iterable[ThreadOwnership]
) -> tuple[OwnedEvidenceCollision, ...]:
    ownership_by_thread = {row.thread_id: row for row in ownership}
    locations: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for thread_id, trace in corpus.trace_by_thread.items():
        copied = ownership_by_thread[thread_id].copied_prefix_events
        for event in trace.events[copied:]:
            event_keys = {
                (evidence.kind, evidence.digest)
                for evidence in event.response_evidence
                if evidence.stable_source_identity
            }
            for key in sorted(event_keys):
                locations[key].append((thread_id, event.ordinal))
    return tuple(
        OwnedEvidenceCollision(kind, digest, tuple(sorted(rows)))
        for (kind, digest), rows in sorted(locations.items())
        if len(set(rows)) > 1
    )


def reconstruct_dag_owned_suffix(corpus: CodexCorpus) -> CodexReconstruction:
    """Primary reconstruction from graph ancestry and physical owned suffixes."""

    aliases, sensitivities = _primary_detect_root_aliases(corpus)
    alias_by_id = {row.replay_thread_id: row for row in aliases}
    totals = ZERO_TOKENS
    by_attribution: dict[tuple[str, str], TokenVector] = defaultdict(TokenVector)
    ownership: list[ThreadOwnership] = []
    structural_copies: list[StructuralGraphCopy] = []
    rejected_candidates: list[RejectedGraphCopyCandidate] = []
    unresolved = 0
    for thread_id in corpus.graph.topological_thread_ids():
        trace = corpus.trace_by_thread[thread_id]
        alias = alias_by_id.get(thread_id)
        if alias is not None:
            copied = alias.copied_event_count
            owned_events = trace.events[copied:]
            boundary = trace.events[copied - 1].total if copied else ZERO_TOKENS
            _primary_validate_first_owned(trace, copied, boundary)
        else:
            copied, structural_copy, rejected_candidate = _primary_copied_prefix(
                corpus, thread_id
            )
            if structural_copy is not None:
                structural_copies.append(structural_copy)
            if rejected_candidate is not None:
                rejected_candidates.append(rejected_candidate)
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
            model, effort, was_unresolved = _event_attribution(event)
            unresolved += int(was_unresolved)
            by_attribution[(model, effort)] = by_attribution[(model, effort)] + event.last
    ownership_rows = tuple(sorted(ownership, key=lambda row: row.thread_id))
    coverage = _coverage_for_ownership(corpus, ownership_rows)
    collisions = _owned_stable_evidence_collisions(corpus, ownership_rows)
    alias_tokens = _sum_alias_tokens(aliases)
    root_sensitivity_tokens = _sum_sensitivity_tokens(sensitivities)
    graph_copy_tokens = _sum_structural_graph_copy_tokens(structural_copies)
    attributed = tuple(
        AttributionTotal(model, effort, tokens)
        for (model, effort), tokens in sorted(by_attribution.items())
    )
    return CodexReconstruction(
        method="dag_structural_owned_suffix_reconstruction_v3",
        totals=totals,
        by_model_effort=attributed,
        thread_ownership=ownership_rows,
        root_replay_aliases=aliases,
        unresolved_root_replay_sensitivities=sensitivities,
        structural_graph_copies=tuple(sorted(structural_copies)),
        rejected_graph_copy_candidates=tuple(sorted(rejected_candidates)),
        owned_stable_evidence_collisions=collisions,
        no_root_collapse_totals=totals + alias_tokens,
        possible_root_collapse_lower_bound=totals.subtract(root_sensitivity_tokens),
        no_graph_collapse_upper_bound=totals + graph_copy_tokens,
        coverage=coverage,
        unresolved_attribution_events=unresolved,
        heartbeat_records=corpus.heartbeat_records,
        duplicate_emissions=corpus.duplicate_emissions,
        reported_total_mismatches=corpus.reported_total_mismatches,
        reset_count=corpus.reset_count,
        explicit_zero_reset_markers=corpus.explicit_zero_reset_markers,
    )


# Root replay classification is an explicit source-evidence decision shared by
# both reconstructions.  Graph-child ownership below is otherwise independent:
# it never searches an ancestor counter sequence.  It asks only what state the
# direct parent had reached by the child's recorded creation time.


def _parse_event_timestamp(event: CodexTokenEvent) -> datetime:
    if event.timestamp is None:
        raise CodexAccountingError(f"owned-boundary event lacks a timestamp: {event.source_ref}")
    try:
        value = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CodexAccountingError(
            f"owned-boundary event timestamp is not ISO-8601: {event.source_ref}"
        ) from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise CodexAccountingError(
            f"owned-boundary event timestamp lacks a UTC offset: {event.source_ref}"
        )
    return value.astimezone(timezone.utc)


def _direct_parent_precreation_event(
    corpus: CodexCorpus, thread_id: str
) -> CodexTokenEvent | None:
    parent_id = corpus.graph.parent_by_child.get(thread_id)
    if parent_id is None:
        return None
    seconds, milliseconds = divmod(
        corpus.graph.thread_by_id[thread_id].created_at_ms, 1000
    )
    created = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
        microsecond=milliseconds * 1000
    )
    eligible: list[tuple[datetime, int, CodexTokenEvent]] = []
    for event in corpus.trace_by_thread[parent_id].events:
        timestamp = _parse_event_timestamp(event)
        if timestamp <= created:
            eligible.append((timestamp, event.ordinal, event))
    selected = max(eligible, default=None, key=lambda row: (row[0], row[1]))
    return None if selected is None else selected[2]


def _timestamp_direct_parent_ownership(
    corpus: CodexCorpus, thread_id: str
) -> tuple[
    int,
    TokenVector,
    StructuralGraphCopy | None,
    RejectedGraphCopyCandidate | None,
]:
    trace = corpus.trace_by_thread[thread_id]
    parent_id = corpus.graph.parent_by_child.get(thread_id)
    if parent_id is None:
        if not trace.events:
            return 0, ZERO_TOKENS, None, None
        baseline = trace.events[0].total.subtract(trace.events[0].last)
        if not baseline.is_zero:
            raise CodexAccountingError(f"root {thread_id} does not start from zero")
        return 0, ZERO_TOKENS, None, None

    parent_events = corpus.trace_by_thread[parent_id].events
    parent_event = _direct_parent_precreation_event(corpus, thread_id)
    boundary = ZERO_TOKENS if parent_event is None else parent_event.total
    if not trace.events:
        return 0, ZERO_TOKENS, None, None

    baseline = trace.events[0].total.subtract(trace.events[0].last)
    if parent_event is None:
        if baseline.is_zero:
            return 0, ZERO_TOKENS, None, None
        raise CodexAccountingError(
            f"thread {thread_id} has copied counters but no pre-creation direct-parent state"
        )
    boundary_signature = parent_event.counter_signature
    hits = [
        index
        for index, event in enumerate(trace.events)
        if event.counter_signature == boundary_signature
    ]
    if len(hits) == 1:
        potential = hits[0] + 1
        parent_end = parent_event.ordinal
        parent_start = parent_end - potential + 1
        if parent_start < 0:
            raise CodexAmbiguityError(
                f"thread {thread_id} timestamp boundary cannot span its candidate prefix"
            )
        source_segment = parent_events[parent_start : parent_end + 1]
        full_sequence_match = len(source_segment) == potential and all(
            trace.events[index].counter_signature
            == source_segment[index].counter_signature
            for index in range(potential)
        )
        if not full_sequence_match:
            if baseline.is_zero:
                return 0, ZERO_TOKENS, None, None
            if baseline == boundary:
                return 0, boundary, None, None
            raise CodexAmbiguityError(
                f"thread {thread_id} reaches the timestamp boundary without a full sequence match"
            )
        payload_evidence = _compare_complete_payload_segments(
            trace.events[:potential], source_segment
        )
        if payload_evidence == "complete_normalized_payload_mismatch":
            _primary_validate_first_owned(trace, 0, baseline)
            return (
                0,
                baseline,
                None,
                _rejected_graph_copy_candidate(corpus, thread_id, potential),
            )
        _primary_validate_first_owned(trace, potential, boundary)
        return (
            potential,
            boundary,
            _structural_graph_copy(
                corpus, thread_id, potential, payload_evidence
            ),
            None,
        )
    if len(hits) > 1:
        raise CodexAmbiguityError(
            f"thread {thread_id} has {len(hits)} direct-parent timestamp-boundary matches"
        )
    if baseline == boundary:
        # A logical fork may contain only its owned suffix.
        return 0, boundary, None, None
    if baseline.is_zero:
        # Codex also creates fresh subagent rollouts: they have a graph parent
        # for orchestration provenance but do not inherit its token counter.
        return 0, ZERO_TOKENS, None, None
    raise CodexAmbiguityError(
        f"thread {thread_id} has 0 direct-parent timestamp-boundary matches"
    )


def reconstruct_source_timestamp_ownership(corpus: CodexCorpus) -> CodexReconstruction:
    """Cross-check using source-proven roots and direct-parent timestamps."""

    aliases, sensitivities = _primary_detect_root_aliases(corpus)
    replay_lookup = {alias.replay_thread_id: alias for alias in aliases}
    grand_total = ZERO_TOKENS
    grouped: dict[tuple[str, str], TokenVector] = defaultdict(TokenVector)
    ownership_rows: list[ThreadOwnership] = []
    structural_copies: list[StructuralGraphCopy] = []
    rejected_candidates: list[RejectedGraphCopyCandidate] = []
    unresolved = 0
    for thread_id in corpus.graph.topological_thread_ids():
        trace = corpus.trace_by_thread[thread_id]
        replay = replay_lookup.get(thread_id)
        if replay is not None:
            copied = replay.copied_event_count
            boundary = trace.events[copied - 1].total if copied else ZERO_TOKENS
            _primary_validate_first_owned(trace, copied, boundary)
        else:
            (
                copied,
                boundary,
                structural_copy,
                rejected_candidate,
            ) = _timestamp_direct_parent_ownership(corpus, thread_id)
            if structural_copy is not None:
                structural_copies.append(structural_copy)
            if rejected_candidate is not None:
                rejected_candidates.append(rejected_candidate)
        owned_events = trace.events[copied:]
        ownership_rows.append(
            ThreadOwnership(
                thread_id=thread_id,
                copied_prefix_events=copied,
                owned_events=len(owned_events),
                inherited_boundary=boundary,
                replay_of=None if replay is None else replay.canonical_thread_id,
            )
        )
        for event in owned_events:
            grand_total = grand_total + event.last
            model, effort, missing = _event_attribution(event)
            unresolved += int(missing)
            grouped[(model, effort)] = grouped[(model, effort)] + event.last
    ordered_ownership = tuple(sorted(ownership_rows, key=lambda row: row.thread_id))
    coverage = _coverage_for_ownership(corpus, ordered_ownership)
    collisions = _owned_stable_evidence_collisions(corpus, ordered_ownership)
    alias_tokens = _sum_alias_tokens(aliases)
    root_sensitivity_tokens = _sum_sensitivity_tokens(sensitivities)
    graph_copy_tokens = _sum_structural_graph_copy_tokens(structural_copies)
    breakdown = tuple(
        AttributionTotal(model, effort, tokens)
        for (model, effort), tokens in sorted(grouped.items())
    )
    return CodexReconstruction(
        method="direct_parent_timestamp_structural_reconstruction_v3",
        totals=grand_total,
        by_model_effort=breakdown,
        thread_ownership=ordered_ownership,
        root_replay_aliases=aliases,
        unresolved_root_replay_sensitivities=sensitivities,
        structural_graph_copies=tuple(sorted(structural_copies)),
        rejected_graph_copy_candidates=tuple(sorted(rejected_candidates)),
        owned_stable_evidence_collisions=collisions,
        no_root_collapse_totals=grand_total + alias_tokens,
        possible_root_collapse_lower_bound=grand_total.subtract(
            root_sensitivity_tokens
        ),
        no_graph_collapse_upper_bound=grand_total + graph_copy_tokens,
        coverage=coverage,
        unresolved_attribution_events=unresolved,
        heartbeat_records=corpus.heartbeat_records,
        duplicate_emissions=corpus.duplicate_emissions,
        reported_total_mismatches=corpus.reported_total_mismatches,
        reset_count=corpus.reset_count,
        explicit_zero_reset_markers=corpus.explicit_zero_reset_markers,
    )


def reconstruct_epoch_deltas(corpus: CodexCorpus) -> CodexReconstruction:
    """Compatibility name for the v3 reconstructed timestamp cross-check."""

    return reconstruct_source_timestamp_ownership(corpus)


@dataclass(frozen=True)
class CodexAgreement:
    totals: TokenVector
    no_root_collapse_totals: TokenVector
    no_graph_collapse_upper_bound: TokenVector
    by_model_effort: tuple[AttributionTotal, ...]
    root_replay_aliases: tuple[RootReplayAlias, ...]
    structural_graph_copies: tuple[StructuralGraphCopy, ...]
    rejected_graph_copy_candidates: tuple[RejectedGraphCopyCandidate, ...]
    owned_stable_evidence_collisions: tuple[OwnedEvidenceCollision, ...]
    coverage: CodexCoverage
    primary_sha256: str
    independent_sha256: str

    def to_dict(self) -> dict[str, Any]:
        graph_lengths = [row.copied_event_count for row in self.structural_graph_copies]
        return {
            "status": "reconstructed_agreement_coverage_pass",
            "measurement_class": "reconstructed",
            "totals": self.totals.to_dict(),
            "no_root_collapse_totals": self.no_root_collapse_totals.to_dict(),
            "no_graph_collapse_upper_bound": (
                self.no_graph_collapse_upper_bound.to_dict()
            ),
            "by_model_effort": [row.to_dict() for row in self.by_model_effort],
            "root_replay_aliases": [row.to_dict() for row in self.root_replay_aliases],
            "structural_graph_copies": [
                row.to_dict() for row in self.structural_graph_copies
            ],
            "structural_graph_copy_summary": {
                "measurement_class": "reconstructed",
                "source_identity_proven": False,
                "copy_count": len(graph_lengths),
                "copied_event_count": sum(graph_lengths),
                "minimum_prefix_events": min(graph_lengths) if graph_lengths else 0,
                "maximum_prefix_events": max(graph_lengths) if graph_lengths else 0,
                "payload_confirmed_copy_count": sum(
                    row.payload_evidence == "complete_normalized_payload_equal"
                    for row in self.structural_graph_copies
                ),
                "payload_incomplete_copy_count": sum(
                    row.payload_evidence == "incomplete_structural_fallback"
                    for row in self.structural_graph_copies
                ),
                "payload_rejected_candidate_count": len(
                    self.rejected_graph_copy_candidates
                ),
            },
            "rejected_graph_copy_candidates": [
                row.to_dict() for row in self.rejected_graph_copy_candidates
            ],
            "owned_stable_evidence_collisions": [
                row.to_dict() for row in self.owned_stable_evidence_collisions
            ],
            "coverage": self.coverage.to_dict(),
            "primary_sha256": self.primary_sha256,
            "independent_sha256": self.independent_sha256,
        }


def assert_reconstructed_agreement(
    primary: CodexReconstruction, independent: CodexReconstruction
) -> CodexAgreement:
    """Authorize a reconstructed total, never an exact-source measurement."""

    comparisons = {
        "totals": (primary.totals, independent.totals),
        "by_model_effort": (primary.by_model_effort, independent.by_model_effort),
        "thread_ownership": (primary.thread_ownership, independent.thread_ownership),
        "root_replay_aliases": (primary.root_replay_aliases, independent.root_replay_aliases),
        "unresolved_root_replay_sensitivities": (
            primary.unresolved_root_replay_sensitivities,
            independent.unresolved_root_replay_sensitivities,
        ),
        "structural_graph_copies": (
            primary.structural_graph_copies,
            independent.structural_graph_copies,
        ),
        "rejected_graph_copy_candidates": (
            primary.rejected_graph_copy_candidates,
            independent.rejected_graph_copy_candidates,
        ),
        "owned_stable_evidence_collisions": (
            primary.owned_stable_evidence_collisions,
            independent.owned_stable_evidence_collisions,
        ),
        "no_root_collapse_totals": (
            primary.no_root_collapse_totals,
            independent.no_root_collapse_totals,
        ),
        "possible_root_collapse_lower_bound": (
            primary.possible_root_collapse_lower_bound,
            independent.possible_root_collapse_lower_bound,
        ),
        "no_graph_collapse_upper_bound": (
            primary.no_graph_collapse_upper_bound,
            independent.no_graph_collapse_upper_bound,
        ),
        "coverage": (primary.coverage, independent.coverage),
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
        "explicit_zero_reset_markers": (
            primary.explicit_zero_reset_markers,
            independent.explicit_zero_reset_markers,
        ),
    }
    disagreements = [name for name, (left, right) in comparisons.items() if left != right]
    if disagreements:
        raise CodexAccountingError(
            f"Codex reconstruction methods disagree on: {', '.join(disagreements)}"
        )
    if (
        not primary.reconstructed_total_authorized
        or not independent.reconstructed_total_authorized
    ):
        blockers: list[str] = []
        if not primary.coverage.complete_coverage:
            blockers.append(f"coverage={primary.coverage.status}")
        if primary.unresolved_root_replay_sensitivities:
            blockers.append("unresolved_root_replay_sensitivity")
        if primary.owned_stable_evidence_collisions:
            blockers.append("owned_stable_response_identity_collision")
        if primary.unresolved_attribution_events:
            blockers.append("unresolved_model_attribution")
        raise CodexAccountingError(
            "Codex reconstruction agrees numerically but reconstructed status is blocked: "
            + ", ".join(blockers)
        )
    return CodexAgreement(
        totals=primary.totals,
        no_root_collapse_totals=primary.no_root_collapse_totals,
        no_graph_collapse_upper_bound=primary.no_graph_collapse_upper_bound,
        by_model_effort=primary.by_model_effort,
        root_replay_aliases=primary.root_replay_aliases,
        structural_graph_copies=primary.structural_graph_copies,
        rejected_graph_copy_candidates=primary.rejected_graph_copy_candidates,
        owned_stable_evidence_collisions=primary.owned_stable_evidence_collisions,
        coverage=primary.coverage,
        primary_sha256=primary.reconstruction_sha256,
        independent_sha256=independent.reconstruction_sha256,
    )
