#!/usr/bin/env python3
"""Exclusive-create one powered-v13 cycle checkpoint from literal inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from powered_v13_cycle import (  # noqa: E402
    CycleObservation,
    V13CycleError,
    build_checkpoint,
    parse_json_string_list,
    write_checkpoint_exclusive,
)


def _literal_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected literal true or false")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--checkpoint-utc", required=True)
    result.add_argument("--goal-elapsed-seconds", required=True, type=int)
    result.add_argument("--provider-observed-utc", required=True)
    result.add_argument("--provider-balance-usd", required=True)
    result.add_argument("--provider-spend-limit-usd", required=True)
    result.add_argument("--active-pod-ids-json", required=True)
    result.add_argument("--phase", required=True)
    result.add_argument("--status", required=True)
    result.add_argument("--phase-spent-usd", required=True)
    result.add_argument("--phase-cap-usd", required=True)
    result.add_argument("--projected-remaining-phase-a-usd", required=True)
    result.add_argument("--projected-core-usd", required=True)
    result.add_argument("--projected-audit-usd", required=True)
    result.add_argument("--reserve-usd", required=True)
    result.add_argument("--independent-n", required=True, type=int)
    result.add_argument("--target-n", required=True, type=int)
    result.add_argument("--independent-unit-ids-json", required=True)
    result.add_argument("--git-commit", required=True)
    result.add_argument("--git-clean", required=True, type=_literal_bool)
    result.add_argument("--completed-gates-json", required=True)
    result.add_argument("--pending-gates-json", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        observation = CycleObservation(
            checkpoint_utc=args.checkpoint_utc,
            goal_elapsed_seconds=args.goal_elapsed_seconds,
            provider_observed_utc=args.provider_observed_utc,
            provider_balance_usd=args.provider_balance_usd,
            provider_spend_limit_usd=args.provider_spend_limit_usd,
            active_pod_ids=parse_json_string_list(
                args.active_pod_ids_json, "active_pod_ids"),
            phase=args.phase,
            status=args.status,
            phase_spent_usd=args.phase_spent_usd,
            phase_cap_usd=args.phase_cap_usd,
            projected_remaining_phase_a_usd=(
                args.projected_remaining_phase_a_usd),
            projected_core_usd=args.projected_core_usd,
            projected_audit_usd=args.projected_audit_usd,
            reserve_usd=args.reserve_usd,
            independent_n=args.independent_n,
            target_n=args.target_n,
            independent_unit_ids=parse_json_string_list(
                args.independent_unit_ids_json, "independent_unit_ids"),
            git_commit=args.git_commit,
            git_clean=args.git_clean,
            completed_gates=parse_json_string_list(
                args.completed_gates_json, "completed_gates"),
            pending_gates=parse_json_string_list(
                args.pending_gates_json, "pending_gates"),
        )
        checkpoint = build_checkpoint(observation)
        write_checkpoint_exclusive(args.output, checkpoint)
    except (V13CycleError, FileExistsError, OSError) as exc:
        parser().error(str(exc))
    print(json.dumps({
        "output": str(args.output),
        "assessment": checkpoint["assessment"]["status"],
        "semantic_gap_n": checkpoint["derived"]["semantic_gap_n"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

