#!/usr/bin/env python3
"""Freeze project Codex rollouts and emit a privacy-safe usage reconstruction.

The Codex state database is used only to discover the project thread graph and
its rollout files.  The collector freezes exactly those JSONL byte prefixes,
checks that the graph did not change during the freeze, and runs both ownership
reconstructions from :mod:`accounting.codex`.

The output contains the library's sanitized graph/reconstruction dictionaries
and an opaque prefix manifest.  It never serializes conversation records, raw
thread IDs, physical rollout paths, filesystem identity metadata, or source
metadata from SQLite.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from accounting.codex import (
    CodexAccountingError,
    CodexCorpus,
    CodexGraph,
    CodexReconstruction,
    TokenVector,
    assert_reconstructed_agreement,
    parse_frozen_rollouts,
    reconstruct_dag_owned_suffix,
    reconstruct_source_timestamp_ownership,
    snapshot_sanitized_graph,
)
from accounting.core import (
    FreezeError,
    FreezeManifest,
    canonical_sha256,
    freeze_file_set,
    iter_frozen_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = Path.home() / ".codex/state_5.sqlite"
DEFAULT_SESSIONS_ROOT = Path.home() / ".codex/sessions"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/end_to_end_accounting"
DEFAULT_LOGICAL_ROOT = "~/.codex/sessions"
PUBLIC_LOGICAL_ROOT = "codex-session-rollouts"
LONG_CONTEXT_INPUT_THRESHOLD = 272_000


class CodexSnapshotError(RuntimeError):
    """Raised when a Codex snapshot cannot satisfy its freeze contract."""


def _utc_filename_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def default_output_path(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Return a timestamp-unique output path for one collector invocation."""

    return Path(output_dir) / (
        "codex-project-usage_codex-cli_" f"{_utc_filename_timestamp()}.json"
    )


def _discover_graph_rollouts(graph: CodexGraph) -> Callable[[Path], Iterable[Path]]:
    logical_paths = tuple(thread.logical_rollout_path for thread in graph.threads)

    def discover(root: Path) -> Iterable[Path]:
        return tuple(root / logical_path for logical_path in logical_paths)

    return discover


def _public_freeze_manifest(
    manifest: FreezeManifest,
    graph: CodexGraph,
) -> dict[str, Any]:
    """Strip physical paths and filesystem metadata from a prefix manifest."""

    public_path_by_private_path = {
        thread.logical_rollout_path: thread.to_dict()["logical_rollout_path"]
        for thread in graph.threads
    }
    manifest_paths = tuple(row.logical_path for row in manifest.files)
    if set(manifest_paths) != set(public_path_by_private_path):
        raise CodexSnapshotError(
            "frozen rollout set does not exactly match the sanitized project graph"
        )
    rows = [
        {
            "logical_rollout_path": public_path_by_private_path[row.logical_path],
            "prefix_bytes": row.prefix_bytes,
            "prefix_sha256": row.prefix_sha256,
            "parsed_complete_lines": row.parsed_complete_lines,
            "malformed_complete_lines": row.malformed_complete_lines,
            "ignored_partial_trailing_lines": row.ignored_partial_trailing_lines,
        }
        for row in manifest.files
    ]
    rows.sort(key=lambda row: row["logical_rollout_path"])
    unsigned = {
        "schema": "codex_opaque_prefix_freeze_v1",
        # Fixed public label: never echo a caller's physical or private root.
        "logical_root": PUBLIC_LOGICAL_ROOT,
        "capture_started_at_utc": manifest.capture_started_at_utc,
        "capture_completed_at_utc": manifest.capture_completed_at_utc,
        "source_set_sha256": canonical_sha256(
            [row["logical_rollout_path"] for row in rows]
        ),
        "prefix_bytes_total": sum(row["prefix_bytes"] for row in rows),
        "file_count": len(rows),
        "files": rows,
    }
    return {**unsigned, "manifest_sha256": canonical_sha256(unsigned)}


def _reconstruction_payload_without_method(
    reconstruction: CodexReconstruction,
) -> dict[str, Any]:
    payload = reconstruction.unsigned_dict()
    payload.pop("method")
    return payload


def _require_reconstruction_agreement_before_gate(
    primary: CodexReconstruction,
    independent: CodexReconstruction,
) -> None:
    """Require both methods to match even when pending EOF evidence blocks PASS."""

    if _reconstruction_payload_without_method(
        primary
    ) != _reconstruction_payload_without_method(independent):
        raise CodexSnapshotError(
            "Codex reconstruction methods disagree before the coverage gate"
        )


def _verify_graph_stability(before: CodexGraph, after: CodexGraph) -> dict[str, Any]:
    """Require graph identity/topology stability while allowing clock advancement.

    ``updated_at_ms`` is live telemetry rather than graph identity.  A Codex
    task can advance it merely by invoking this read-only collector.  Every
    other captured field, including rollout mapping and spawn edges, must be
    byte-for-byte equal, and update clocks may only move forward.
    """

    normalized_before = CodexGraph(
        tuple(replace(thread, updated_at_ms=0) for thread in before.threads),
        before.edges,
    )
    normalized_after = CodexGraph(
        tuple(replace(thread, updated_at_ms=0) for thread in after.threads),
        after.edges,
    )
    if normalized_before != normalized_after:
        raise CodexSnapshotError(
            "project thread graph identity or topology changed during rollout freeze"
        )
    before_updates = {
        thread.thread_id: thread.updated_at_ms for thread in before.threads
    }
    after_updates = {thread.thread_id: thread.updated_at_ms for thread in after.threads}
    if any(after_updates[thread_id] < value for thread_id, value in before_updates.items()):
        raise CodexSnapshotError("project thread update clock moved backward during freeze")
    advanced = sum(
        after_updates[thread_id] > value for thread_id, value in before_updates.items()
    )
    return {
        "status": "PASS",
        "comparison": (
            "thread_identity_rollout_mapping_metadata_and_edges_equal; "
            "updated_at_ms_allowed_to_advance_monotonically"
        ),
        "graph_sha256_before": before.graph_sha256,
        "graph_sha256_after": after.graph_sha256,
        "updated_at_advanced_thread_count": advanced,
    }


def _aggregate_owned_request_usage(
    corpus: CodexCorpus,
    reconstruction: CodexReconstruction,
) -> dict[str, Any]:
    """Aggregate owned usage events without exposing any event-level record."""

    ownership = {row.thread_id: row for row in reconstruction.thread_ownership}
    if set(ownership) != set(corpus.trace_by_thread):
        raise CodexSnapshotError("owned-request aggregation does not span the corpus")
    grouped: dict[tuple[str, str, str], tuple[int, TokenVector]] = defaultdict(
        lambda: (0, TokenVector())
    )
    totals = TokenVector()
    event_count = 0
    for thread_id, trace in corpus.trace_by_thread.items():
        copied = ownership[thread_id].copied_prefix_events
        for event in trace.events[copied:]:
            model = event.model or "<unknown-model>"
            effort = event.effort or "<unknown-effort>"
            context_class = (
                "long"
                if event.last.input_tokens > LONG_CONTEXT_INPUT_THRESHOLD
                else "standard"
            )
            key = model, effort, context_class
            prior_count, prior_tokens = grouped[key]
            grouped[key] = prior_count + 1, prior_tokens + event.last
            totals = totals + event.last
            event_count += 1
    if event_count != reconstruction.coverage.owned_usage_events:
        raise CodexSnapshotError(
            "owned-request event count disagrees with reconstruction coverage"
        )
    if totals != reconstruction.totals:
        raise CodexSnapshotError(
            "owned-request token aggregate disagrees with reconstructed totals"
        )
    rows = [
        {
            "model": model,
            "effort": effort,
            "input_context_class": context_class,
            "request_event_count": count,
            "tokens": tokens.to_dict(),
        }
        for (model, effort, context_class), (count, tokens) in sorted(grouped.items())
    ]
    return {
        "schema": "codex_owned_request_usage_aggregate_v1",
        "measurement_class": "reconstructed",
        "ownership_rule": (
            "Only each thread's reconstructed owned suffix is aggregated; inherited "
            "graph prefixes and source-proven root-replay prefixes are excluded."
        ),
        "request_unit": "owned_nonzero_token_usage_event",
        "input_context_class_rule": {
            "input_token_field": "last_token_usage.input_tokens",
            "standard": f"input_tokens <= {LONG_CONTEXT_INPUT_THRESHOLD}",
            "long": f"input_tokens > {LONG_CONTEXT_INPUT_THRESHOLD}",
        },
        "request_event_count": event_count,
        "unresolved_attribution_event_count": (
            reconstruction.unresolved_attribution_events
        ),
        "tokens": totals.to_dict(),
        "by_model_effort_input_context_class": rows,
        "integrity": {
            "event_count_matches_owned_usage_coverage": True,
            "token_sum_matches_reconstructed_total": True,
        },
    }


def _build_closeout_gate(
    primary: CodexReconstruction,
    independent: CodexReconstruction,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _require_reconstruction_agreement_before_gate(primary, independent)
    coverage = primary.coverage
    if coverage.complete_coverage:
        try:
            agreement = assert_reconstructed_agreement(primary, independent)
        except CodexAccountingError as exc:
            raise CodexSnapshotError(str(exc)) from exc
        return (
            {
                "status": "PASS",
                "classification": "reconstructed_agreement_coverage_pass",
                "reconstructed_total_authorized": True,
                "scope": (
                    "Reconstructed Codex subscription workload through the frozen "
                    "rollout prefixes; not a source-exact measurement or cash-spend "
                    "record."
                ),
                "blocked_reasons": [],
            },
            agreement.to_dict(),
        )

    if (
        coverage.status != "FAIL_PENDING_RESPONSES"
        or coverage.uncovered_usage_events != 0
        or coverage.pending_response_items <= 0
    ):
        raise CodexSnapshotError(
            "refusing a blocked snapshot whose coverage failure is not solely "
            f"pending response evidence at frozen EOF: {coverage.status}"
        )
    return (
        {
            "status": "BLOCKED",
            "classification": "reconstructed_through_last_metered_boundary",
            "reconstructed_total_authorized": False,
            "scope": (
                "Provisional reconstructed workload through the last frozen metered "
                "token boundary. Completed response evidence remains pending at EOF, "
                "so this is not an authorized final total, source-exact measurement, "
                "or cash-spend record."
            ),
            "blocked_reasons": ["pending_response_evidence_at_frozen_eof"],
            "pending_response_items": coverage.pending_response_items,
            "pending_thread_count": coverage.pending_thread_count,
        },
        None,
    )


def build_codex_usage_snapshot(
    database_path: Path,
    *,
    sessions_root: Path,
    project_root: Path,
    logical_root: str = DEFAULT_LOGICAL_ROOT,
) -> dict[str, Any]:
    """Freeze the project graph's rollouts and reconstruct Codex token usage."""

    database_path = Path(database_path).expanduser().resolve()
    sessions_root = Path(sessions_root).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    try:
        graph_before = snapshot_sanitized_graph(
            database_path,
            project_root=project_root,
            sessions_root=sessions_root,
        )
        if not graph_before.threads:
            raise CodexSnapshotError("refusing closeout with no project Codex threads")
        manifest = freeze_file_set(
            sessions_root,
            logical_root=logical_root,
            discover=_discover_graph_rollouts(graph_before),
            fail_on_malformed_complete_line=True,
        )
        partial_lines = sum(
            row.ignored_partial_trailing_lines for row in manifest.files
        )
        if partial_lines:
            raise CodexSnapshotError(
                f"refusing closeout with {partial_lines} partial trailing JSONL line(s)"
            )
        graph_after = snapshot_sanitized_graph(
            database_path,
            project_root=project_root,
            sessions_root=sessions_root,
        )
        graph_stability = _verify_graph_stability(graph_before, graph_after)
        records = tuple(iter_frozen_jsonl(manifest, sessions_root, verify=True))
        corpus = parse_frozen_rollouts(records, graph_before)
        primary = reconstruct_dag_owned_suffix(corpus)
        independent = reconstruct_source_timestamp_ownership(corpus)
    except FreezeError as exc:
        raise CodexSnapshotError(str(exc)) from exc
    except CodexAccountingError as exc:
        raise CodexSnapshotError(str(exc)) from exc

    gate, agreement = _build_closeout_gate(primary, independent)
    owned_request_aggregate = _aggregate_owned_request_usage(corpus, primary)
    public_manifest = _public_freeze_manifest(manifest, graph_before)
    snapshot: dict[str, Any] = {
        "schema": "codex_project_usage_snapshot_v1",
        "measurement_class": "reconstructed",
        "closeout_gate": gate,
        "source": {
            "graph_stability": graph_stability,
            "sanitized_graph": graph_before.to_dict(),
            "opaque_prefix_manifest": public_manifest,
            "complete_json_object_count": len(records),
        },
        "reconstructions": {
            "primary": primary.to_dict(),
            "independent": independent.to_dict(),
        },
        "owned_request_aggregate": owned_request_aggregate,
        "agreement": agreement,
        "money": {
            "cash_spend_usd": None,
            "cash_spend_classification": (
                "unknown; local Codex state and rollout logs expose subscription "
                "workload telemetry, not an observed incremental cash charge"
            ),
        },
    }
    snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
    return snapshot


def write_snapshot(path: Path, snapshot: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise CodexSnapshotError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--sessions-root", type=Path, default=DEFAULT_SESSIONS_ROOT)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--logical-root", default=DEFAULT_LOGICAL_ROOT)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", type=Path)
    output_group.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = args.output or default_output_path(args.output_dir)
    print(f"RUN codex_project_usage model=codex-cli → {output}")
    snapshot = build_codex_usage_snapshot(
        args.database,
        sessions_root=args.sessions_root,
        project_root=args.project_root,
        logical_root=args.logical_root,
    )
    write_snapshot(output, snapshot)
    totals = snapshot["reconstructions"]["primary"]["totals"]
    print(
        "CODEX USAGE SNAPSHOT "
        f"status={snapshot['closeout_gate']['status']} "
        f"threads={len(snapshot['source']['sanitized_graph']['threads'])} "
        f"tokens={totals['total_tokens']} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
