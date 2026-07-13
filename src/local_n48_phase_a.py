"""Treatment-blind Phase-A screening for local coherent-state N48 v2.

This runner may replay full C/W histories, build fresh B, and score oracle/fresh
validity. It never imports or constructs correct-history, wrong-history, or
placebo treatment arms. One exclusive JSON checkpoint is written per candidate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, "src")
from arms import arm_e_snapshot  # noqa: E402
from local_n48_core import (  # noqa: E402
    build_value_alignment_pairs,
    replay_fresh,
    replay_source,
    score_probe,
    snapshots_bit_exact,
)
from provenance import (  # noqa: E402
    build_manifest,
    capture_mlx_provenance,
    write_run_manifest,
)


DESIGN_ID = "coherent-state-local-mlx-n48-v2"
MODEL_DEFAULT = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
ROSTER_DEFAULT = Path("data/coherent_state_local_n48_v2/phase-a-roster-v1.json")
CARRIER_BANK_DEFAULT = Path(
    "data/coherent_state_local_n48_v2/fixed-carriers-v1.json")
DAMAGE_MIN = 5.0
GREEDY_CAP = 16


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _replay_record(result) -> dict:
    return {
        "mode": result.mode,
        "variant": result.variant,
        "token_ids": result.token_ids,
        "plan_geometry": result.plan_geometry,
        "calls": result.calls,
        "carrier_token_logprobs": result.carrier_token_logprobs,
        "carrier_mean_logprob": (
            math.fsum(row["logprob"] for row in result.carrier_token_logprobs)
            / len(result.carrier_token_logprobs)),
        "physical_end": result.physical_end,
        "logical_end": result.logical_end,
    }


def _begins(score: dict, alternative: str) -> bool:
    wanted = score[alternative]["content_token_ids"]
    observed = score["greedy_content_token_ids"]
    return observed[:len(wanted)] == wanted


def _score_record(score: dict) -> dict:
    return {
        "probe": score["probe"],
        "target": score["target"],
        "countertarget": score["countertarget"],
        "margin": score["margin"],
        "greedy_token_ids": score["greedy_token_ids"],
        "greedy_content_token_ids": score["greedy_content_token_ids"],
        "greedy_cap": score["greedy_cap"],
        "greedy_stop_reason": score["greedy_stop_reason"],
        "probe_suffix_token_ids": score["probe_suffix_token_ids"],
        "context_physical_end": score["context_physical_end"],
        "context_logical_end": score["context_logical_end"],
        "probe_prefix_physical_end": score["probe_prefix_physical_end"],
        "probe_prefix_logical_end": score["probe_prefix_logical_end"],
    }


def screen_carrier(model, tokenizer, fixture: dict, carrier: dict) -> dict:
    started = time.time()
    text = carrier["text"]
    alignment = build_value_alignment_pairs(tokenizer, fixture, text)
    source_c = replay_source(model, tokenizer, fixture, "C", text)
    source_w = replay_source(model, tokenizer, fixture, "W", text)
    fresh = replay_fresh(model, tokenizer, fixture, "C", text)
    fresh_repeat = replay_fresh(model, tokenizer, fixture, "C", text)

    repeat_identity = snapshots_bit_exact(fresh.snapshot, fresh_repeat.snapshot)
    identity_pairs = [(destination, destination)
                      for destination, _source in alignment.correct_pairs]
    sham_snapshot = arm_e_snapshot(
        fresh.snapshot, fresh.snapshot, identity_pairs, 1.0)
    sham_identity = snapshots_bit_exact(fresh.snapshot, sham_snapshot)

    focal = fixture["focal"]
    focal_args = (
        focal["probe"], focal["C_target"], focal["W_target"])
    focal_a_c = score_probe(
        model, tokenizer, source_c.snapshot, source_c.context_messages,
        *focal_args, greedy_cap=GREEDY_CAP)
    focal_a_w = score_probe(
        model, tokenizer, source_w.snapshot, source_w.context_messages,
        *focal_args, greedy_cap=GREEDY_CAP)
    focal_b = score_probe(
        model, tokenizer, fresh.snapshot, fresh.context_messages,
        *focal_args, greedy_cap=GREEDY_CAP)

    nonfocal = fixture["nonfocal_control"]
    nonfocal_args = (
        nonfocal["probe"], nonfocal["target"], nonfocal["countertarget"])
    nonfocal_a_c = score_probe(
        model, tokenizer, source_c.snapshot, source_c.context_messages,
        *nonfocal_args, greedy_cap=GREEDY_CAP)
    nonfocal_a_w = score_probe(
        model, tokenizer, source_w.snapshot, source_w.context_messages,
        *nonfocal_args, greedy_cap=GREEDY_CAP)
    nonfocal_b = score_probe(
        model, tokenizer, fresh.snapshot, fresh.context_messages,
        *nonfocal_args, greedy_cap=GREEDY_CAP)

    dplus = (focal_a_c["target"]["mean_logprob"]
             - focal_b["target"]["mean_logprob"])
    dmargin = focal_a_c["margin"] - focal_b["margin"]
    checks = {
        "fresh_repeat_bit_exact": repeat_identity,
        "fresh_sham_graft_bit_exact": sham_identity,
        "oracle_C_greedy_begins_C_target": _begins(focal_a_c, "target"),
        "oracle_C_margin_positive": focal_a_c["margin"] > 0.0,
        "oracle_W_greedy_begins_W_target": _begins(
            focal_a_w, "countertarget"),
        "oracle_W_margin_negative": focal_a_w["margin"] < 0.0,
        "fresh_does_not_begin_C_target": not _begins(focal_b, "target"),
        "correct_target_damage_at_least_5": dplus >= DAMAGE_MIN,
        "margin_damage_at_least_5": dmargin >= DAMAGE_MIN,
        "nonfocal_oracle_C_begins_target": _begins(nonfocal_a_c, "target"),
        "nonfocal_oracle_C_margin_positive": nonfocal_a_c["margin"] > 0.0,
        "nonfocal_oracle_W_begins_target": _begins(nonfocal_a_w, "target"),
        "nonfocal_oracle_W_margin_positive": nonfocal_a_w["margin"] > 0.0,
        "nonfocal_fresh_begins_target": _begins(nonfocal_b, "target"),
        "nonfocal_fresh_margin_positive": nonfocal_b["margin"] > 0.0,
    }
    result = {
        "carrier_id": carrier["carrier_id"],
        "carrier_utf8_sha256": carrier["utf8_sha256"],
        "carrier_text": text,
        "status": "ELIGIBLE" if all(checks.values()) else "INELIGIBLE",
        "checks": checks,
        "correct_target_damage": dplus,
        "margin_damage": dmargin,
        "alignment": {
            "row_count": len(alignment.correct_pairs),
            "content_row_count": len(alignment.content_correct_pairs),
            "structural_row_count": len(alignment.structural_correct_pairs),
            "system_width": alignment.system_width,
            "content_physical_interval": list(
                alignment.content_physical_interval),
            "source_token_counts": alignment.source_token_counts,
            "rows": alignment.rows,
        },
        "replays": {
            "A_C": _replay_record(source_c),
            "A_W": _replay_record(source_w),
            "B": _replay_record(fresh),
            "B_repeat": _replay_record(fresh_repeat),
        },
        "scores": {
            "focal_A_C": _score_record(focal_a_c),
            "focal_A_W": _score_record(focal_a_w),
            "focal_B": _score_record(focal_b),
            "nonfocal_A_C": _score_record(nonfocal_a_c),
            "nonfocal_A_W": _score_record(nonfocal_a_w),
            "nonfocal_B": _score_record(nonfocal_b),
        },
        "wall_seconds": time.time() - started,
    }
    del source_c, source_w, fresh, fresh_repeat, sham_snapshot
    gc.collect()
    mx.clear_cache()
    return result


def screen_fixture(model, tokenizer, fixture_path: Path,
                   carrier_bank: dict, manifest: dict) -> dict:
    started = time.time()
    raw = fixture_path.read_bytes()
    fixture = json.loads(raw)
    carriers = []
    for carrier in carrier_bank["carriers"]:
        carriers.append(screen_carrier(
            model, tokenizer, fixture, carrier))
    eligible = all(row["status"] == "ELIGIBLE" for row in carriers)
    return {
        "schema": "coherent_state_local_n48_v2_phase_a_candidate_v1",
        "design_id": DESIGN_ID,
        "phase": "A_TREATMENT_BLIND",
        "treatment_outcomes_computed": False,
        "candidate_id": fixture["stable_candidate_id"],
        "stratum": fixture["stratum_id"],
        "rank": fixture["materialization_authorization"]["permutation_rank"],
        "fixture_path": str(fixture_path),
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "status": "ELIGIBLE" if eligible else "INELIGIBLE",
        "carrier_conditions": carriers,
        "wall_seconds": time.time() - started,
        "_manifest": manifest,
    }


def _selected_paths(roster: dict, fixture: Path | None,
                    max_rank: int) -> list[Path]:
    if fixture is not None:
        return [fixture]
    records = [row for row in roster["records"]
               if row["screening_disposition"] ==
               "SCREEN_IN_FROZEN_RANK_ORDER"
               and row["permutation_rank"] <= max_rank]
    order = {name: index for index, name in enumerate(
        roster["frozen_strata_order"])}
    records.sort(key=lambda row: (
        order[row["stratum_id"]], row["screening_order_within_stratum"]))
    return [Path(row["fixture_path"]) for row in records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--roster", type=Path, default=ROSTER_DEFAULT)
    parser.add_argument("--carrier-bank", type=Path,
                        default=CARRIER_BANK_DEFAULT)
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to reuse output directory {args.out}")
    args.out.mkdir(parents=True)
    roster = json.loads(args.roster.read_text())
    carrier_bank = json.loads(args.carrier_bank.read_text())
    paths = _selected_paths(roster, args.fixture, args.max_rank)

    model, tokenizer = load(args.model)
    provenance = capture_mlx_provenance(model, args.model)
    manifest = build_manifest(
        model_provenance=provenance,
        dtype_env=None,
        harness="src/local_n48_phase_a.py",
        intervention={
            "arm": "A_C,A_W,B and B-sham identity only",
            "graft_type": None,
            "alpha": None,
            "alignment": "exact visible-token twins; no treatment constructed",
        },
        metric={
            "definition": ("Phase-A teacher-forced C/W exact-target logprob, "
                           "margin, greedy validity, and nonfocal control"),
            "is_proxy": False,
            "resolve_rate_measured": False,
        },
        condition={
            "summary_kind": "two fixed external carrier conditions",
            "summary_request_sha256": None,
            "summary_source": "fixed external; q1 forced under C/W/fresh",
        },
        corpus={
            "name": "local N48 frozen phase-A roster",
            "split": "treatment-blind eligibility",
            "n": 0,
            "instance_ids": [path.stem for path in paths],
        },
        extra={
            "design_id": DESIGN_ID,
            "roster_path": str(args.roster),
            "roster_sha256": _sha(args.roster),
            "carrier_bank_path": str(args.carrier_bank),
            "carrier_bank_sha256": _sha(args.carrier_bank),
            "treatment_outcomes_computed": False,
            "damage_min": DAMAGE_MIN,
            "greedy_cap": GREEDY_CAP,
        },
    )
    write_run_manifest(args.out, manifest)
    print(json.dumps({
        "model": args.model,
        "resolved_revision": provenance["resolved_revision"],
        "output": str(args.out),
        "candidate_count": len(paths),
        "semantic_n": 0,
    }), flush=True)
    for index, path in enumerate(paths, start=1):
        result = screen_fixture(
            model, tokenizer, path, carrier_bank, manifest)
        output = args.out / "candidates" / (
            f"{result['stratum']}_rank-{result['rank']:02d}_"
            f"{result['candidate_id']}.json")
        _write_exclusive(output, result)
        print(json.dumps({
            "index": index,
            "candidate_id": result["candidate_id"],
            "stratum": result["stratum"],
            "rank": result["rank"],
            "status": result["status"],
            "carrier_statuses": {
                row["carrier_id"]: row["status"]
                for row in result["carrier_conditions"]},
            "wall_seconds": result["wall_seconds"],
            "output": str(output),
        }), flush=True)


if __name__ == "__main__":
    main()
