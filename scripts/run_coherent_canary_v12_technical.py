#!/usr/bin/env python3
"""Run the lean v12 technical gate and persist one complete raw JSON record."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coherent_canary_loader import (  # noqa: E402
    EXACT_SPEC, LOCAL_SPEC, prepare_pinned_subject,
)
from coherent_canary_runtime import (  # noqa: E402
    execute_fresh_plan, execute_replay_plan, snapshot_hashes, tensor_sha256,
)
from coherent_canary_store import (  # noqa: E402
    atomic_create_json, unique_output_path,
)
from coherent_canary_technical import (  # noqa: E402
    run_fresh_self_replacement, run_generated_forced_identity,
    run_natural_calibration, run_path_control,
)
from coherent_canary_tokens import (  # noqa: E402
    build_fresh_destination_plan, build_role_native_plan,
)


DESIGN_ID = "coherent-state-decision-canary-v12"
SCHEMA = "coherent_state_decision_canary_v12_technical_raw_v1"
SUBJECTS = {"local-apparatus": LOCAL_SPEC, "exact-subject": EXACT_SPEC}
DEFAULT_PREREG = ROOT / "COHERENT-STATE-DECISION-CANARY-V12-PREREGISTRATION.md"
DEFAULT_IDENTITY = ROOT / "data/coherent_canary_v12/generated_forced_identity_fixture.json"
DEFAULT_TECHNICAL = ROOT / "data/coherent_canary_v12/technical_control_fixture.json"
DEFAULT_CARRIER = ROOT / "data/coherent_canary_v12/fixed_text_token_evidence_v2.json"
DEFAULT_OUTPUT_DIR = ROOT / "results/coherent_canary_v12_technical"


class TechnicalRunnerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TechnicalRunnerError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except Exception as exc:
        raise TechnicalRunnerError(f"cannot read fixture {path}: {exc}") from exc
    require(isinstance(value, dict), f"fixture is not an object: {path}")
    return value


def binding(path: Path, *, repo: Path) -> dict[str, str]:
    require(path.is_file(), f"bound file absent: {path}")
    resolved = path.resolve()
    try:
        display = resolved.relative_to(repo.resolve()).as_posix()
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": file_sha256(path)}


def git_value(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True,
        check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def compact_model_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    """Keep reproducibility commitments without duplicating 18k tensor rows."""
    result = {key: value for key, value in inventory.items()
              if key != "weight_tensors"}
    rows = inventory.get("weight_tensors")
    if isinstance(rows, list):
        result["weight_tensor_count"] = len(rows)
    return result


def compact_path_control(record: dict[str, Any]) -> dict[str, Any]:
    """Retain path decisions while bounding a technical JSON artifact.

    Per-row diagnostics are already checked by the model-facing function.  The
    raw artifact keeps their canonical hash and coverage counts rather than
    repeating thousands of structurally identical rows for every ULP attempt.
    """
    result = dict(record)
    attempts = []
    for raw_attempt in record.get("attempts", []):
        attempt = dict(raw_attempt)
        for direction in ("plus", "minus"):
            key = f"{direction}_row_diagnostics"
            rows = attempt.pop(key, None)
            require(isinstance(rows, list) and rows,
                    f"path {direction} row diagnostics are absent")
            encoded = json.dumps(
                rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
            attempt[f"{key}_summary"] = {
                "row_count": len(rows),
                "selected_count": sum(
                    1 for row in rows if isinstance(row, dict) and
                    row.get("selected") is True),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        attempts.append(attempt)
    result["attempts"] = attempts
    return result


def execution_record(result) -> dict[str, Any]:
    return {
        "calls": list(result.calls),
        "token_ids": list(result.executed_token_ids),
        "logical_positions": list(result.logical_positions),
        "physical_positions": list(result.physical_positions),
        "physical_end": int(result.physical_end),
        "logical_end": int(result.logical_end),
        "snapshot_hashes": snapshot_hashes(result.snapshot),
        "last_logits_sha256": tensor_sha256(result.last_logits),
    }


def deterministic_repeats(run_once: Callable[[], Any], label: str) -> dict[str, Any]:
    records = [execution_record(run_once()), execution_record(run_once())]
    require(records[0] == records[1], f"{label} deterministic repeat differs")
    return {"status": "PASS", "repeat_count": 2, "records": records}


def stage(document: dict[str, Any], name: str, function: Callable[[], Any]) -> Any:
    started_at = utc_now()
    started = time.monotonic()
    value = function()
    document["stage_timings"][name] = {
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "wall_time_seconds": time.monotonic() - started,
    }
    return value


def run(args: argparse.Namespace) -> tuple[Path, dict[str, Any], int]:
    require(args.subject in SUBJECTS, f"unknown subject: {args.subject}")
    require(args.hourly_cost_usd is None or args.hourly_cost_usd >= 0,
            "hourly cost must be nonnegative")
    output = unique_output_path(
        args.output_dir, "coherent-canary-v12-technical", args.subject)
    # Required observability: this happens before fixture parsing or model load.
    print(f"RUN coherent-canary-v12-technical subject={args.subject} -> {output}",
          flush=True)
    started_wall = time.monotonic()
    started_at = utc_now()
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "design_id": DESIGN_ID,
        "status": "STARTED",
        "subject": args.subject,
        "started_at_utc": started_at,
        "output_path": str(output),
        "bindings": {},
        "source_bindings": {},
        "provenance": {
            "runner_python": platform.python_version(),
            "runner_platform": platform.platform(),
            "repository_head": git_value(args.repo, "rev-parse", "HEAD"),
            "repository_branch": git_value(args.repo, "branch", "--show-current"),
        },
        "stage_timings": {},
        "diagnostic_summary": {},
    }
    exit_code = 1
    try:
        paths = {
            "preregistration": args.prereg,
            "identity_fixture": args.identity_fixture,
            "technical_fixture": args.technical_fixture,
            "carrier_fixture": args.carrier_fixture,
        }
        document["bindings"] = {
            name: binding(path, repo=args.repo) for name, path in paths.items()}
        document["source_bindings"] = {
            "preregistration": document["bindings"]["preregistration"],
            "identity_fixture": document["bindings"]["identity_fixture"],
            "technical_fixture": document["bindings"]["technical_fixture"],
            "fixed_text_evidence": document["bindings"]["carrier_fixture"],
        }
        identity_fixture = load_json(args.identity_fixture)
        technical_fixture = load_json(args.technical_fixture)

        prepared = stage(document, "prepare_pinned_subject", lambda:
            prepare_pinned_subject(
                spec=SUBJECTS[args.subject], repo=args.repo,
                local_files_only=not args.allow_download))
        model = prepared["model"]
        tokenizer = prepared["tokenizer"]
        eos_ids = [int(value) for value in
                   prepared["runtime_fingerprint"]["eos_ids"]]
        document["provenance"].update(json_safe({
            "runtime_fingerprint": prepared["runtime_fingerprint"],
            "repository_authorization": prepared["repository"],
            "dependencies": prepared["dependencies"],
            "device": prepared["device"],
            "model_snapshot": prepared["model_snapshot"],
            "protocol_tokenizer_snapshot": prepared["protocol_tokenizer_snapshot"],
            "model_inventory": compact_model_inventory(
                prepared["model_inventory"]),
            "protocol_tokenizer_inventory": prepared[
                "protocol_tokenizer_inventory"],
            "model_snapshot_contract": prepared["model_snapshot_contract"],
        }))
        document["runtime_fingerprint"] = json_safe(
            prepared["runtime_fingerprint"])

        document["generated_forced_identity"] = stage(
            document, "generated_forced_identity", lambda:
            run_generated_forced_identity(
                model, tokenizer, identity_fixture, eos_ids))

        middle = int(technical_fixture["middle_end_msg"])
        correct_history = technical_fixture["correct"]
        n_plan = build_role_native_plan(
            tokenizer, correct_history, middle_end_msg=middle)
        fresh_plan = build_fresh_destination_plan(
            tokenizer, correct_history, middle_end_msg=middle)
        document["deterministic_repeats"] = {
            "correct_history_N": stage(
                document, "correct_history_N_repeats", lambda:
                deterministic_repeats(
                    lambda: execute_replay_plan(model, n_plan),
                    "correct-history N replay")),
            "fresh_destination": stage(
                document, "fresh_destination_repeats", lambda:
                deterministic_repeats(
                    lambda: execute_fresh_plan(model, fresh_plan),
                    "fresh destination")),
        }

        document["fresh_self_replacement"] = stage(
            document, "fresh_self_replacement", lambda:
            run_fresh_self_replacement(model, fresh_plan))
        replacement_rows = document["fresh_self_replacement"].get("regions", [])
        require(len(replacement_rows) == 9 and
                all(row.get("status") == "PASS" for row in replacement_rows),
                "fresh self-replacement did not produce nine passing cells")

        path_control = stage(
            document, "path_control", lambda:
            run_path_control(model, tokenizer, technical_fixture))
        document["path_control"] = compact_path_control(path_control)
        path_pass = document["path_control"].get("status") == "PASS"
        require(isinstance(document["path_control"].get("attempts"), list) and
                document["path_control"]["attempts"],
                "path control did not persist attempts")
        if path_pass:
            document["natural_calibration"] = stage(
                document, "natural_calibration", lambda:
                run_natural_calibration(
                    model, tokenizer, technical_fixture, eos_ids))
        else:
            document["natural_calibration"] = {
                "status": "SKIPPED_PATH_CONTROL_FAIL"}

        required_pass = (
            document["generated_forced_identity"].get("status") == "PASS" and
            all(row.get("status") == "PASS" for row in
                document["deterministic_repeats"].values()) and
            document["fresh_self_replacement"].get("status") == "PASS" and
            path_pass
        )
        document["status"] = "PASS" if required_pass else "FAIL"
        document["diagnostic_summary"] = {
            "identity": document["generated_forced_identity"].get("status"),
            "correct_history_N_repeat": document["deterministic_repeats"][
                "correct_history_N"].get("status"),
            "fresh_repeat": document["deterministic_repeats"][
                "fresh_destination"].get("status"),
            "fresh_self_replacement_cells": len(replacement_rows),
            "path_control": document["path_control"].get("status"),
            "path_attempt_count": len(document["path_control"]["attempts"]),
            "natural_calibration": document["natural_calibration"].get("status"),
        }
        exit_code = 0 if document["status"] == "PASS" else 2
    except Exception as exc:
        document["status"] = "ERROR"
        document["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        document["diagnostic_summary"] = {
            "error_type": type(exc).__name__, "error_message": str(exc)}
        exit_code = 1
    finally:
        wall = time.monotonic() - started_wall
        document["completed_at_utc"] = utc_now()
        document["wall_time_seconds"] = wall
        document["hourly_cost_usd"] = args.hourly_cost_usd
        document["estimated_cost_usd"] = (
            wall * args.hourly_cost_usd / 3600
            if args.hourly_cost_usd is not None else None)
        atomic_create_json(output, json_safe(document))
    print(json.dumps(document["diagnostic_summary"], sort_keys=True), flush=True)
    return output, document, exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject", required=True, choices=tuple(SUBJECTS))
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--identity-fixture", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--technical-fixture", type=Path, default=DEFAULT_TECHNICAL)
    parser.add_argument("--carrier-fixture", type=Path, default=DEFAULT_CARRIER)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--hourly-cost-usd", type=float)
    return parser.parse_args(argv)


def main() -> None:
    _, _, code = run(parse_args())
    raise SystemExit(code)


if __name__ == "__main__":
    main()
