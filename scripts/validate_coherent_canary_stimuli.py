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

from coherent_canary_schema import MODEL_ID, MODEL_REVISION  # noqa: E402
from coherent_canary_stimuli import load_and_validate_case  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
    output = {
        "schema": "coherent_state_canary_v12_stimulus_validation_v1",
        "status": "MECHANICAL_DRAFT_PASS",
        "semantic_authorized": False,
        "execution_authorized": False,
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
