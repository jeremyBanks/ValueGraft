#!/usr/bin/env python3
"""Validate ValueGraft intervention probe artifacts and compact summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_SEQUENCES = {
    "full_context",
    "fresh_compacted",
    "alpha0_grafted_compacted",
    "grafted_compacted",
    "shifted_grafted_compacted",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("artifact")
    p.add_argument("--summary")
    return p.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"INVALID: {message}")


def row_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("token_id"),
        row.get("token"),
        row.get("argmax_token_id"),
        row.get("argmax_token"),
        json.dumps(row.get("layers"), sort_keys=True, ensure_ascii=False),
    )


def validate_case(case_id: str, case: dict[str, Any], layers: set[int]) -> tuple[int, int]:
    if "error" in case:
        fail(f"{case_id}: case error present: {case['error']}")
    target = case.get("probe_target") or {}
    target_tokens = target.get("tokens") or []
    target_count = int(target.get("token_count", -1))
    if target_count != len(target_tokens):
        fail(f"{case_id}: token_count does not match token list")
    if target_count <= 0:
        fail(f"{case_id}: empty target")

    graft = case.get("graft") or {}
    if not graft.get("available"):
        fail(f"{case_id}: graft unavailable")
    pairs = int(graft.get("pairs", -1))
    summary_tokens = int((case.get("token_counts") or {}).get("summary_tokens", -2))
    if pairs != summary_tokens:
        fail(f"{case_id}: graft pairs {pairs} != summary tokens {summary_tokens}")
    if int(graft.get("changed_value_slot_count", -1)) != pairs * len(graft.get("changed_value_layers", [])):
        fail(f"{case_id}: changed value slot count is inconsistent")

    probe = case.get("post_boundary_probe") or {}
    seqs = (probe.get("forced_target_sequences") or {})
    missing = REQUIRED_SEQUENCES - set(seqs)
    if missing:
        fail(f"{case_id}: missing sequences {sorted(missing)}")

    for name, seq in seqs.items():
        rows = seq.get("rows") or []
        if len(rows) != target_count:
            fail(f"{case_id}/{name}: row count {len(rows)} != target count {target_count}")
        for i, row in enumerate(rows):
            if row.get("relative_position") != i:
                fail(f"{case_id}/{name}: bad relative_position at row {i}")
            if row.get("token_id") is None or row.get("token") is None:
                fail(f"{case_id}/{name}: missing target token at row {i}")
            row_layers = {int(k) for k in (row.get("layers") or {}).keys()}
            if row_layers != layers:
                fail(f"{case_id}/{name}: bad layer set at row {i}: {sorted(row_layers)}")

    fresh_rows = seqs["fresh_compacted"]["rows"]
    alpha0_rows = seqs["alpha0_grafted_compacted"]["rows"]
    for i, (fresh, alpha0) in enumerate(zip(fresh_rows, alpha0_rows, strict=True)):
        if row_signature(fresh) != row_signature(alpha0):
            fail(f"{case_id}: alpha-zero differs from fresh at row {i}")

    return target_count, pairs


def main() -> None:
    args = parse_args()
    artifact = json.loads(Path(args.artifact).read_text())
    layers = {int(x) for x in artifact.get("lens", {}).get("sampled_layers", [])}
    if not layers:
        fail("no sampled layers")
    cases = artifact.get("cases") or {}
    if not cases:
        fail("no cases")

    total_tokens = 0
    total_pairs = 0
    for case_id, case in cases.items():
        tokens, pairs = validate_case(case_id, case, layers)
        total_tokens += tokens
        total_pairs += pairs

    if args.summary:
        summary = json.loads(Path(args.summary).read_text())
        if summary.get("case_count") != len(cases):
            fail("summary case_count does not match raw artifact")
        if set(summary.get("cases", {}).keys()) != set(cases.keys()):
            fail("summary case set does not match raw artifact")
        if summary.get("cases_with_errors"):
            fail(f"summary reports case errors: {summary['cases_with_errors']}")

    print(
        "VALID: "
        f"cases={len(cases)} target_tokens={total_tokens} "
        f"summary_pairs={total_pairs} layers={sorted(layers)}"
    )
    if args.summary:
        print("VALID: compact summary matches case set and reports no errors")


if __name__ == "__main__":
    main()
