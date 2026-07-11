#!/usr/bin/env python3
"""Validate authored v12 cases with the pinned production tokenizer only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coherent_canary_schema import (  # noqa: E402
    ENGINEERED_CASE_IDS,
    MODEL_ID,
    MODEL_REVISION,
)
from coherent_canary_stimuli import load_and_validate_case  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--require-complete-engineered-set", action="store_true",
        help="fail unless paths resolve once each to e01--e06 in frozen order",
    )
    args = parser.parse_args()
    paths = []
    for path in args.paths:
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.json")))
        else:
            paths.append(path)
    if not paths:
        raise SystemExit("no authored case files found")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    rows = [load_and_validate_case(tokenizer, path) for path in paths]
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate case IDs: {ids}")
    if args.require_complete_engineered_set:
        if tuple(ids) != ENGINEERED_CASE_IDS:
            raise SystemExit(
                f"complete engineered set/order differs: {ids} != "
                f"{list(ENGINEERED_CASE_IDS)}")
        expected_bands = {
            "e01": "short", "e02": "short", "e03": "short", "e04": "mid",
        }
        for row in rows:
            expected = expected_bands.get(row["case_id"])
            if expected is not None and row["length_band"] != expected:
                raise SystemExit(
                    f"{row['case_id']} band {row['length_band']} != {expected}")
    output = {
        "schema": "coherent_state_canary_v12_stimulus_validation_v1",
        "status": "MECHANICAL_DRAFT_PASS",
        "semantic_authorized": False,
        "execution_authorized": False,
        "complete_engineered_set_required": bool(
            args.require_complete_engineered_set),
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"MECHANICAL_DRAFT_PASS {len(rows)} cases -> {args.output}")


if __name__ == "__main__":
    main()
