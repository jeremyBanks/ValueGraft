"""Pure tokenizer/text gate for the two frozen local-N48-v2 carriers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, "src")
from local_n48_carriers import _bounded_contains, _forbidden_forms  # noqa: E402
from powered_v13_tokens import (  # noqa: E402
    build_fresh_destination_plan,
    build_role_native_plan,
)
from provenance import sha256_file  # noqa: E402


SNAPSHOT_DEFAULT = Path(
    "/Users/jeb/.cache/huggingface/hub/"
    "models--mlx-community--Qwen3-30B-A3B-Instruct-2507-4bit/"
    "snapshots/e9675aa3ca5f900ccef55267914466d55ab325fa")


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_DEFAULT)
    parser.add_argument("--bank", type=Path, default=Path(
        "data/coherent_state_local_n48_v2/fixed-carriers-v1.json"))
    parser.add_argument("--candidates", type=Path, default=Path(
        "data/coherent_state_local_n48_v1/candidates"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.snapshot, local_files_only=True)
    bank = json.loads(args.bank.read_text())
    records = []
    for fixture_path in sorted(args.candidates.glob("*/*.json")):
        raw = fixture_path.read_bytes()
        fixture = json.loads(raw)
        c_messages = fixture["variants"]["C"]["messages"]
        w_messages = fixture["variants"]["W"]["messages"]
        middle = fixture["middle_end_msg"]
        for carrier in bank["carriers"]:
            text = carrier["text"]
            ids = tokenizer.encode(text, add_special_tokens=False)
            checks = {
                "utf8_sha256": hashlib.sha256(text.encode()).hexdigest()
                    == carrier["utf8_sha256"],
                "word_count": len(text.split()) == carrier["word_count"]
                    and 40 <= len(text.split()) <= 60,
                "token_count": 40 <= len(ids) <= 80,
                "no_special_ids": not any(
                    value in set(tokenizer.all_special_ids) for value in ids),
                "no_digits": not any(character.isdigit() for character in text),
                "no_forbidden_surface": not any(
                    _bounded_contains(text, surface)
                    for surface in _forbidden_forms(fixture)),
            }
            c_plan = build_role_native_plan(
                tokenizer, c_messages, middle_end_msg=middle,
                carrier_content=text)
            w_plan = build_role_native_plan(
                tokenizer, w_messages, middle_end_msg=middle,
                carrier_content=text)
            c_fresh = build_fresh_destination_plan(
                tokenizer, c_messages, middle_end_msg=middle,
                carrier_content=text)
            w_fresh = build_fresh_destination_plan(
                tokenizer, w_messages, middle_end_msg=middle,
                carrier_content=text)
            c_content = c_plan.token_ids[
                c_plan.regions.content_start:c_plan.regions.content_end]
            w_content = w_plan.token_ids[
                w_plan.regions.content_start:w_plan.regions.content_end]
            fresh_region = c_fresh.physical_regions
            f_content = c_fresh.token_ids[
                fresh_region.content_start:fresh_region.content_end]
            checks.update({
                "carrier_ids_equal_C_W_B": c_content == w_content == f_content == ids,
                "fresh_visible_ids_equal_C_W":
                    c_fresh.token_ids == w_fresh.token_ids,
                "carrier_region_width_equal_C_W_B":
                    len(c_content) == len(w_content) == len(f_content),
            })
            records.append({
                "candidate_id": fixture["stable_candidate_id"],
                "stratum": fixture["stratum_id"],
                "rank": fixture["materialization_authorization"][
                    "permutation_rank"],
                "fixture_path": str(fixture_path),
                "fixture_sha256": hashlib.sha256(raw).hexdigest(),
                "carrier_id": carrier["carrier_id"],
                "content_token_count": len(ids),
                "content_token_ids": ids,
                "content_token_ids_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
            })

    failed = [record for record in records if record["status"] != "PASS"]
    result = {
        "schema": "coherent_state_local_n48_v2_fixed_carrier_gate_v1",
        "design_id": bank["design_id"],
        "status": "PASS" if not failed else "FAIL",
        "subject_model_called": False,
        "bank_path": str(args.bank),
        "bank_sha256": sha256_file(args.bank),
        "snapshot_revision": args.snapshot.name,
        "tokenizer_json_sha256": sha256_file(args.snapshot / "tokenizer.json"),
        "tokenizer_config_sha256": sha256_file(
            args.snapshot / "tokenizer_config.json"),
        "chat_template_sha256": sha256_file(
            args.snapshot / "chat_template.jinja"),
        "candidate_count": len({record["candidate_id"] for record in records}),
        "carrier_count": len(bank["carriers"]),
        "candidate_carrier_checks": len(records),
        "pass_count": len(records) - len(failed),
        "fail_count": len(failed),
        "records": records,
    }
    _write_exclusive(args.out, result)
    print(json.dumps({
        "status": result["status"],
        "candidate_count": result["candidate_count"],
        "candidate_carrier_checks": len(records),
        "fail_count": len(failed),
        "output": str(args.out),
    }, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
