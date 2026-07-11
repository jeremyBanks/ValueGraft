"""Persist the frozen external donor-slot mapping under the production tokenizer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from arms_common import SUMMARY_REQUEST
from coherent_state_cases import FROZEN_ORDER, WRONG_DONOR
from coherent_state_runtime import AMENDMENT_ID, DESIGN_ID
from coherent_state_tokens import matched_wrong_prefix_ids


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(donor_dir: Path) -> dict:
    snapshot = Path(snapshot_download(
        MODEL, revision=REVISION, local_files_only=True))
    resolved = snapshot.name
    if resolved != REVISION:
        raise RuntimeError(
            f"production tokenizer revision {resolved!r} != {REVISION!r}")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True)
    rows = []
    donor_hashes = set()
    for target_id in FROZEN_ORDER:
        donor_id = WRONG_DONOR[target_id]
        target_path = donor_dir / f"{target_id}.json"
        donor_path = donor_dir / f"{donor_id}.json"
        target = json.loads(target_path.read_text())
        donor = json.loads(donor_path.read_text())
        matched = matched_wrong_prefix_ids(
            tokenizer, target, donor, SUMMARY_REQUEST)
        donor_hash = sha256_file(donor_path)
        donor_hashes.add(donor_hash)
        decoded = tokenizer.decode(matched.wrong_ids)
        rows.append({
            "target_id": target_id,
            "donor_id": donor_id,
            "target_path": str(target_path),
            "donor_path": str(donor_path),
            "target_sha256": sha256_file(target_path),
            "donor_sha256": donor_hash,
            "donor_recorded_author": (donor.get("meta") or {}).get("author"),
            "correct_prefix_tokens": len(matched.correct_ids),
            "wrong_prefix_tokens": len(matched.wrong_ids),
            "structural_position_count": len(matched.structural_positions),
            "content_position_count": len(matched.content_positions),
            "replacement_count": len(matched.replacements),
            "replacement_character_count": decoded.count("\ufffd"),
            "replacements": [asdict(row) for row in matched.replacements],
            "passes": True,
        })
    if len(donor_hashes) != len(FROZEN_ORDER):
        raise RuntimeError("external donor file hashes are not unique")
    return {
        "schema": 2,
        "amendment_id": AMENDMENT_ID,
        "design_id": DESIGN_ID,
        "status": "PASS",
        "model": MODEL,
        "requested_revision": REVISION,
        "resolved_tokenizer_revision": resolved,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "mapping": WRONG_DONOR,
        "n_unique_donors": len(donor_hashes),
        "rows": rows,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor-dir", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    print(
        f"RUN coherent_external_donors model={MODEL}@{REVISION} -> {args.output}",
        flush=True)
    result = validate(args.donor_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS n={result['n_unique_donors']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
