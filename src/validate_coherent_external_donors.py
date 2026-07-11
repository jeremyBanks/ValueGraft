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
from coherent_state_hf import sha256_ids
from coherent_state_tokens import matched_wrong_prefix_ids


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
EXTERNAL_AUTHORS = {"sonnet", "opus", "codex-gpt5.5", "sonnet-render"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_with_tokenizer(tokenizer, donor_dir: Path, *,
                            resolved_revision: str | None = None,
                            progress=None) -> dict:
    """Recompute every frozen donor construction from the bound source files.

    This is intentionally pure with respect to model loading so the exact-model
    authorization gate can reuse its already-loaded production tokenizer.
    """
    donor_dir = Path(donor_dir)
    rows = [{
        "order_position": index,
        "target_id": target,
        "donor_id": WRONG_DONOR[target],
        "target_path": str(donor_dir / f"{target}.json"),
        "donor_path": str(donor_dir / f"{WRONG_DONOR[target]}.json"),
        "status": "PENDING", "passes": False,
    } for index, target in enumerate(FROZEN_ORDER, 1)]
    donor_hashes = set()
    donor_ids = []
    target_ids = set(FROZEN_ORDER)
    for order_position, target_id in enumerate(FROZEN_ORDER, 1):
        donor_id = WRONG_DONOR[target_id]
        donor_ids.append(donor_id)
        target_path = donor_dir / f"{target_id}.json"
        donor_path = donor_dir / f"{donor_id}.json"
        target_raw, donor_raw = target_path.read_bytes(), donor_path.read_bytes()
        target = json.loads(target_raw)
        donor = json.loads(donor_raw)
        if str(target.get("id")) != target_id or str(donor.get("id")) != donor_id:
            raise RuntimeError(f"target/donor identity mismatch for {target_id}")
        matched = matched_wrong_prefix_ids(
            tokenizer, target, donor, SUMMARY_REQUEST)
        donor_hash = hashlib.sha256(donor_raw).hexdigest()
        donor_hashes.add(donor_hash)
        donor_author = (donor.get("meta") or {}).get("author")
        specials = {int(value) for value in tokenizer.all_special_ids}
        replacement_rows = []
        covered = []
        for replacement in matched.replacements:
            row = asdict(replacement)
            covered.extend(range(row["start"], row["end"]))
            row.update({
                "target_ids_sha256": sha256_ids(row["target_ids"]),
                "source_pool_sha256": sha256_ids(row["donor_pool_ids"]),
                "replacement_sha256": sha256_ids(row["replacement_ids"]),
                "length": row["end"] - row["start"],
                "contains_special_token": any(
                    int(value) in specials for value in row["replacement_ids"]),
            })
            replacement_rows.append(row)
        structural_equal = all(
            matched.correct_ids[index] == matched.wrong_ids[index]
            for index in matched.structural_positions)
        coverage_exact = (
            len(covered) == len(set(covered)) and
            sorted(covered) == sorted(matched.content_positions))
        changed = [index for index, (left, right) in enumerate(zip(
            matched.correct_ids, matched.wrong_ids)) if left != right]
        passes = bool(
            donor_id not in target_ids and
            donor_author in EXTERNAL_AUTHORS and
            len(matched.correct_ids) == len(matched.wrong_ids) and
            structural_equal and changed and coverage_exact and
            all(not row["contains_special_token"] for row in replacement_rows))
        if not passes:
            raise RuntimeError(f"external donor construction failed for {target_id}")
        rows[order_position - 1] = {
            "status": "PASS",
            "order_position": order_position,
            "target_id": target_id,
            "donor_id": donor_id,
            "target_path": str(target_path),
            "donor_path": str(donor_path),
            "target_sha256": hashlib.sha256(target_raw).hexdigest(),
            "donor_sha256": donor_hash,
            "target_canonical_sha256": sha256_json(target),
            "donor_canonical_sha256": sha256_json(donor),
            "donor_recorded_author": donor_author,
            "subject_native": False,
            "correct_prefix_tokens": len(matched.correct_ids),
            "wrong_prefix_tokens": len(matched.wrong_ids),
            "correct_prefix_ids": [int(value) for value in matched.correct_ids],
            "correct_prefix_sha256": sha256_ids(matched.correct_ids),
            "wrong_prefix_sha256": sha256_ids(matched.wrong_ids),
            "structural_position_count": len(matched.structural_positions),
            "structural_positions": [int(value) for value in
                                     matched.structural_positions],
            "structural_positions_sha256": sha256_ids(
                matched.structural_positions),
            "content_position_count": len(matched.content_positions),
            "content_positions": [int(value) for value in
                                  matched.content_positions],
            "content_positions_sha256": sha256_ids(matched.content_positions),
            "changed_position_count": len(changed),
            "changed_positions": changed,
            "changed_positions_sha256": sha256_ids(changed),
            "replacement_count": len(matched.replacements),
            "replacement_coverage_exact": coverage_exact,
            "replacement_spans_non_overlapping": len(covered) == len(set(covered)),
            "structural_tokens_equal": structural_equal,
            "system_request_header_retained_tail_unchanged": structural_equal,
            "changes_confined_to_declared_content_positions": (
                set(changed).issubset(matched.content_positions)),
            "correct_wrong_length_equal": True,
            "replacements": replacement_rows,
            "passes": True,
        }
        if progress is not None:
            progress({
                "status": "RUNNING", "passes": False,
                "requested_revision": REVISION,
                "resolved_tokenizer_revision": resolved_revision,
                "mapping": dict(WRONG_DONOR),
                "frozen_order": list(FROZEN_ORDER),
                "expected_coverage": len(FROZEN_ORDER),
                "observed_coverage": order_position,
                "rows": json.loads(json.dumps(rows)),
            })
    if donor_ids != [WRONG_DONOR[target] for target in FROZEN_ORDER]:
        raise RuntimeError("external donor mapping order changed")
    if len(set(donor_ids)) != len(FROZEN_ORDER):
        raise RuntimeError("external donor IDs are not unique")
    if set(donor_ids) & target_ids:
        raise RuntimeError("external donor overlaps a scored target")
    if len(donor_hashes) != len(FROZEN_ORDER):
        raise RuntimeError("external donor file hashes are not unique")
    payload = {
        "status": "PASS",
        "passes": True,
        "requested_revision": REVISION,
        "resolved_tokenizer_revision": resolved_revision,
        "mapping": dict(WRONG_DONOR),
        "frozen_order": list(FROZEN_ORDER),
        "expected_coverage": len(FROZEN_ORDER),
        "observed_coverage": len(FROZEN_ORDER),
        "n_unique_donor_ids": len(set(donor_ids)),
        "n_unique_donor_hashes": len(donor_hashes),
        "rows": rows,
    }
    payload["canonical_payload_sha256"] = sha256_json(payload)
    return payload


def validate(donor_dir: Path) -> dict:
    snapshot = Path(snapshot_download(
        MODEL, revision=REVISION, local_files_only=True))
    resolved = snapshot.name
    if resolved != REVISION:
        raise RuntimeError(
            f"production tokenizer revision {resolved!r} != {REVISION!r}")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True)
    validated = validate_with_tokenizer(
        tokenizer, donor_dir, resolved_revision=resolved)
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
        **validated,
        "n_unique_donors": validated["n_unique_donor_hashes"],
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
    payload = (json.dumps(
        result, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False) + "\n").encode()
    if len(payload) >= 4_000_000:
        raise SystemExit(
            f"refusing non-commit-safe donor artifact: {len(payload)} bytes")
    args.output.write_bytes(payload)
    print(f"PASS n={result['n_unique_donors']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
