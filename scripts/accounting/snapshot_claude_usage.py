#!/usr/bin/env python3
"""Freeze local Claude project logs and write a deterministic usage report.

Claude Code stores append-only JSONL transcripts.  A single API response can
appear repeatedly while it streams and can also be copied into parent and
subagent transcripts.  This collector therefore freezes exact byte prefixes,
deduplicates assistant usage globally by ``message.id``, and expands a
selected message's ``usage.iterations`` into request-level rows before
aggregation.

The output contains hashes and aggregate counters, never conversation text.
It reports token consumption only.  Subscription cash spend and a
counterfactual list-price total remain explicitly unknown unless a separate,
time-versioned billing source is supplied.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from accounting.claude import ClaudeRequest, build_claude_requests, select_message_versions
from accounting.core import (
    FreezeError,
    canonical_sha256,
    freeze_file_set,
    iter_frozen_jsonl,
)


DEFAULT_SOURCE_ROOT = Path.home() / ".claude/projects/-Users-jeb-experimentation"
DEFAULT_LOGICAL_ROOT = "~/.claude/projects/-Users-jeb-experimentation"
_TOKEN_FIELDS = (
    "uncached_input_tokens",
    "cache_write_5m_tokens",
    "cache_write_1h_tokens",
    "cache_write_unclassified_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
    "context_input_tokens",
    "total_tokens",
)


class ClaudeSnapshotError(RuntimeError):
    """Raised when a token snapshot cannot satisfy its closeout contract."""


def discover_claude_logs(root: Path) -> Iterable[Path]:
    """Return every mainline and nested subagent JSONL under one project root."""

    return sorted(root.rglob("*.jsonl"))


def _empty_usage() -> dict[str, int]:
    return {field: 0 for field in _TOKEN_FIELDS}


def _add_request(target: dict[str, int], request: ClaudeRequest) -> None:
    for field in _TOKEN_FIELDS:
        target[field] += int(getattr(request, field))


def _aggregate_rows(
    requests: Iterable[ClaudeRequest],
    *,
    dimensions: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, Any]] = defaultdict(
        lambda: {"request_count": 0, "usage": _empty_usage()}
    )
    for request in requests:
        key = tuple(str(getattr(request, dimension)) for dimension in dimensions)
        row = grouped[key]
        row["request_count"] += 1
        _add_request(row["usage"], request)
    result: list[dict[str, Any]] = []
    for key in sorted(grouped):
        aggregate = grouped[key]
        result.append(
            {
                **dict(zip(dimensions, key, strict=True)),
                "request_count": aggregate["request_count"],
                "usage": aggregate["usage"],
            }
        )
    return result


def build_claude_usage_snapshot(
    source_root: Path,
    *,
    logical_root: str = DEFAULT_LOGICAL_ROOT,
) -> dict[str, Any]:
    """Freeze ``source_root`` and derive one exact token snapshot from it."""

    source_root = Path(source_root).expanduser().resolve()
    manifest = freeze_file_set(
        source_root,
        logical_root=logical_root,
        discover=discover_claude_logs,
        fail_on_malformed_complete_line=True,
    )
    if not manifest.files:
        raise ClaudeSnapshotError("refusing closeout with no Claude project JSONL files")
    partial_lines = sum(row.ignored_partial_trailing_lines for row in manifest.files)
    if partial_lines:
        raise ClaudeSnapshotError(
            f"refusing closeout with {partial_lines} partial trailing JSONL line(s)"
        )

    # The manifest is the cutoff.  Appends after capture do not alter these
    # prefixes; a later project-wide closeout should create a new snapshot.
    records = tuple(iter_frozen_jsonl(manifest, source_root, verify=True))
    selected = select_message_versions(records)
    requests = build_claude_requests(records)
    if not selected:
        raise ClaudeSnapshotError("refusing closeout with no assistant usage messages")

    expected_request_count = sum(row.completeness[2] for row in selected)
    if len(requests) != expected_request_count:
        raise ClaudeSnapshotError(
            "request expansion did not preserve selected usage iterations: "
            f"{len(requests)} != {expected_request_count}"
        )
    request_ids = tuple(request.request_id for request in requests)
    if len(request_ids) != len(set(request_ids)):
        raise ClaudeSnapshotError("request IDs are not globally unique")

    total_usage = _empty_usage()
    for request in requests:
        _add_request(total_usage, request)
    if total_usage["context_input_tokens"] != (
        total_usage["uncached_input_tokens"]
        + total_usage["cache_creation_input_tokens"]
        + total_usage["cache_read_input_tokens"]
    ):
        raise ClaudeSnapshotError("aggregate context-input identity failed")
    if total_usage["total_tokens"] != (
        total_usage["context_input_tokens"] + total_usage["output_tokens"]
    ):
        raise ClaudeSnapshotError("aggregate total-token identity failed")

    selection_commitment = [
        {
            "message_id": row.message_id,
            "selected_source_ref": row.selected_source_ref,
            "occurrence_count": row.occurrence_count,
            "occurrence_source_refs": list(row.occurrence_source_refs),
            "session_ids": list(row.session_ids),
            "completeness": list(row.completeness),
            "usage_fingerprint": row.usage_fingerprint,
        }
        for row in selected
    ]
    request_commitment = [request.to_dict() for request in requests]
    timestamps = sorted(row.timestamp for row in selected if row.timestamp is not None)
    malformed_lines = sum(row.malformed_complete_lines for row in manifest.files)

    return {
        "schema": "claude_project_usage_snapshot_v1",
        "coverage": {
            "status": "complete_frozen_prefix",
            "logical_root": logical_root,
            "source_cutoff_utc": manifest.capture_completed_at_utc,
            "source_file_count": len(manifest.files),
            "source_prefix_bytes": manifest.prefix_bytes_total,
            "complete_json_object_count": len(records),
            "malformed_complete_line_count": malformed_lines,
            "ignored_partial_trailing_line_count": partial_lines,
            "message_timestamp_min_utc": timestamps[0] if timestamps else None,
            "message_timestamp_max_utc": timestamps[-1] if timestamps else None,
            "selected_message_missing_timestamp_count": sum(
                row.timestamp is None for row in selected
            ),
        },
        "deduplication": {
            "global_identity_key": "Claude assistant message.id",
            "selection_rule": (
                "Select the lexicographically maximal completeness vector "
                "(output tokens, all tokens, iteration count, iterations present); "
                "equal-best differing usage fingerprints fail closed."
            ),
            "assistant_usage_occurrence_count": sum(
                row.occurrence_count for row in selected
            ),
            "unique_message_count": len(selected),
            "duplicate_occurrence_count": sum(
                row.occurrence_count - 1 for row in selected
            ),
            "cross_session_duplicate_message_count": sum(
                len(row.session_ids) > 1 for row in selected
            ),
            "selected_message_with_iterations_count": sum(
                bool(row.completeness[3]) for row in selected
            ),
            "selection_commitment_sha256": canonical_sha256(selection_commitment),
        },
        "requests": {
            "expansion_rule": (
                "Each nonempty usage.iterations entry is one request; otherwise the "
                "selected top-level usage object is one request."
            ),
            "request_count": len(requests),
            "zero_token_request_count": sum(request.total_tokens == 0 for request in requests),
            "request_commitment_sha256": canonical_sha256(request_commitment),
            "usage": total_usage,
            "by_model": _aggregate_rows(requests, dimensions=("model",)),
            "by_request_class": _aggregate_rows(
                requests,
                dimensions=("model", "service_tier", "speed", "context_class"),
            ),
        },
        "money": {
            "cash_spend_usd": None,
            "cash_spend_classification": (
                "unknown; local Claude CLI logs expose token usage, while this project "
                "used a subscription/subsidized surface rather than an observed "
                "incremental cash charge"
            ),
            "counterfactual_list_price_usd": None,
            "counterfactual_pricing_status": (
                "not computed; no separately frozen, time-versioned request-level "
                "rate table was supplied"
            ),
        },
        "freeze_manifest": manifest.to_dict(),
    }


def write_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise ClaudeSnapshotError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--logical-root", default=DEFAULT_LOGICAL_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        snapshot = build_claude_usage_snapshot(
            args.source_root,
            logical_root=args.logical_root,
        )
        write_snapshot(args.output, snapshot)
    except FreezeError as exc:
        raise ClaudeSnapshotError(str(exc)) from exc
    print(
        "CLAUDE USAGE SNAPSHOT "
        f"files={snapshot['coverage']['source_file_count']} "
        f"messages={snapshot['deduplication']['unique_message_count']} "
        f"requests={snapshot['requests']['request_count']} "
        f"tokens={snapshot['requests']['usage']['total_tokens']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
