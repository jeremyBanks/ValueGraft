#!/usr/bin/env python3
"""Independent, stdlib-only recomputation of the v13 primary bound.

This implementation is intentionally isolated from the production statistical
module.  It imports neither that module nor NumPy/SciPy.  The constants and
formulae below are transcribed directly from the frozen v13 protocol so this
program can serve as a genuinely separate final-bound recomputation.

Input is a JSON list of render-level rows.  Output is deterministic canonical
JSON.  The numerical conclusion label does not incorporate external technical,
placebo-completeness, budget, or portability labels.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


DESIGN_ID = "coherent-state-powered-successor-v13"
SCHEMA = "coherent_state_powered_v13_final_bound_projection_v1"
STRATA = tuple(f"s{index}" for index in range(1, 9))
CELLS = ("full_kv", "value_only")
RENDER_IDS = ("r1", "r2")
FINAL_N = 48
PER_STRATUM = 6
_FAMILY_ALPHA_EXACT = Fraction(1, 20)
_MEAN_ALPHA_PER_CELL_EXACT = Fraction(1, 50)
_TAIL_ALPHA_EXACT = Fraction(1, 100)
FAMILY_ALPHA = float(_FAMILY_ALPHA_EXACT)
MEAN_ALPHA_PER_CELL = float(_MEAN_ALPHA_PER_CELL_EXACT)
TAIL_ALPHA = float(_TAIL_ALPHA_EXACT)
CLIP_LOWER = -0.5
CLIP_UPPER = 0.5
DELTA_CLIP = 0.35
DELTA_TAIL = 0.10


class IndependentRecomputeError(ValueError):
    """The supplied rows differ from the frozen v13 analysis contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentRecomputeError(message)


def _finite_number(value: object, label: str) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)),
             f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is nonfinite")
    return result


def collapse_render_rows(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, Any]]:
    """Collapse exactly two unique C-origin render rows per fixture."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping),
                 f"render row {index} is not a mapping")
        case_id = row.get("case_id")
        _require(isinstance(case_id, str) and bool(case_id),
                 f"render row {index} case_id is empty")
        grouped.setdefault(case_id, []).append(row)
    _require(bool(grouped), "no render rows")

    fixtures: list[dict[str, Any]] = []
    for case_id, case_rows in grouped.items():
        _require(len(case_rows) == 2,
                 f"{case_id} has {len(case_rows)} render rows, expected two")
        _require(all(row.get("render_origin") == "C" for row in case_rows),
                 f"{case_id} renders are not both C-origin")
        render_ids = [row.get("render_id") for row in case_rows]
        _require(set(render_ids) == set(RENDER_IDS)
                 and len(set(render_ids)) == 2,
                 f"{case_id} render IDs differ from r1/r2")
        strata = {row.get("stratum") for row in case_rows}
        _require(len(strata) == 1 and next(iter(strata)) in STRATA,
                 f"{case_id} stratum differs")
        ranks = {row.get("eligible_rank") for row in case_rows}
        _require(len(ranks) == 1, f"{case_id} eligible rank differs")
        rank = next(iter(ranks))
        _require(not isinstance(rank, bool) and isinstance(rank, int)
                 and rank >= 1,
                 f"{case_id} eligible rank is not a positive integer")

        ordered = sorted(case_rows, key=lambda row: str(row["render_id"]))
        render_values = [
            [
                str(row["render_id"]),
                _finite_number(row.get("full_kv"),
                               f"{case_id}:{row['render_id']}:full_kv"),
                _finite_number(row.get("value_only"),
                               f"{case_id}:{row['render_id']}:value_only"),
            ]
            for row in ordered
        ]
        fixtures.append({
            "case_id": case_id,
            "stratum": next(iter(strata)),
            "eligible_rank": rank,
            "full_kv": math.fsum(row[1] for row in render_values) / 2.0,
            "value_only": math.fsum(row[2] for row in render_values) / 2.0,
            "render_values": render_values,
        })

    return sorted(
        fixtures,
        key=lambda fixture: (
            fixture["stratum"],
            fixture["eligible_rank"],
            fixture["case_id"],
        ),
    )


def select_final_sample(
    fixtures: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Select frozen eligible ranks one through six in all eight strata."""
    case_ids = [fixture["case_id"] for fixture in fixtures]
    _require(len(case_ids) == len(set(case_ids)), "duplicate fixture ID")

    selected: list[Mapping[str, Any]] = []
    for stratum in STRATA:
        members = sorted(
            (fixture for fixture in fixtures
             if fixture["stratum"] == stratum),
            key=lambda fixture: (
                fixture["eligible_rank"], fixture["case_id"]),
        )
        ranks = [fixture["eligible_rank"] for fixture in members]
        _require(len(ranks) == len(set(ranks)),
                 f"duplicate eligible rank in {stratum}")
        required = list(range(1, PER_STRATUM + 1))
        _require(ranks[:PER_STRATUM] == required,
                 f"{stratum} lacks frozen ranks {required}")
        selected.extend(members[:PER_STRATUM])
    _require(len(selected) == FINAL_N, "final sample size differs")
    return selected


def _alpha_ledger() -> dict[str, Any]:
    exact_spent = (
        len(CELLS) * _MEAN_ALPHA_PER_CELL_EXACT + _TAIL_ALPHA_EXACT)
    _require(exact_spent == _FAMILY_ALPHA_EXACT,
             "exact primary alpha ledger does not sum to 1/20")
    events = [
        {
            "n_total": FINAL_N,
            "endpoint": "clipped_mean",
            "cell": cell,
            "alpha": float(_MEAN_ALPHA_PER_CELL_EXACT),
        }
        for cell in CELLS
    ] + [{
        "n_total": FINAL_N,
        "endpoint": "large_responder_prevalence",
        "cell": "any_primary_cell",
        "alpha": float(_TAIL_ALPHA_EXACT),
    }]
    spent = math.fsum(float(event["alpha"]) for event in events)
    _require(spent == FAMILY_ALPHA,
             "primary alpha ledger does not sum to 0.05")
    return {"family_alpha": FAMILY_ALPHA, "spent": spent, "events": events}


def _numeric_conclusion_label(
    *,
    joint_resolved: bool,
    full_cell_resolved: bool,
    value_cell_resolved: bool,
) -> str:
    if joint_resolved:
        return "JOINT_BOUND_RESOLVED"
    if full_cell_resolved and value_cell_resolved:
        return "CLIPPED_MEAN_BOUND_ONLY"
    if full_cell_resolved or value_cell_resolved:
        return "CELL_BOUND_ONLY"
    return "BOUND_NOT_RESOLVED_AT_N48"


def recompute(rows: Iterable[Mapping[str, object]]) -> dict[str, Any]:
    # Arithmetic order is frozen independently of the production module:
    # STRATA order, then eligible rank/case ID, then CELLS order.  Every mean
    # uses math.fsum over that order.
    fixtures = collapse_render_rows(rows)
    selected = select_final_sample(fixtures)
    radius = (CLIP_UPPER - CLIP_LOWER) * math.sqrt(
        math.log(1.0 / MEAN_ALPHA_PER_CELL) / (2.0 * FINAL_N))

    cell_results: dict[str, dict[str, Any]] = {}
    for cell in CELLS:
        raw = [_finite_number(fixture[cell], cell) for fixture in selected]
        clipped = [min(CLIP_UPPER, max(CLIP_LOWER, value))
                   for value in raw]
        stratum_raw_means: dict[str, float] = {}
        stratum_clipped_means: dict[str, float] = {}
        for stratum in STRATA:
            indices = [index for index, fixture in enumerate(selected)
                       if fixture["stratum"] == stratum]
            _require(len(indices) == PER_STRATUM,
                     f"{stratum} count differs")
            stratum_raw_means[stratum] = (
                math.fsum(raw[index] for index in indices) / PER_STRATUM)
            stratum_clipped_means[stratum] = (
                math.fsum(clipped[index] for index in indices) / PER_STRATUM)

        clipped_mean = math.fsum(clipped) / FINAL_N
        cell_results[cell] = {
            "cell": cell,
            "raw_mean": math.fsum(raw) / FINAL_N,
            "clipped_mean": clipped_mean,
            "radius": radius,
            "ucb": min(CLIP_UPPER, clipped_mean + radius),
            "alpha": MEAN_ALPHA_PER_CELL,
            "clipped_count_low": sum(value < CLIP_LOWER for value in raw),
            "clipped_count_high": sum(value > CLIP_UPPER for value in raw),
            "stratum_raw_means": stratum_raw_means,
            "stratum_clipped_means": stratum_clipped_means,
        }

    responder_count = sum(
        max(float(fixture["full_kv"]), float(fixture["value_only"]))
        > CLIP_UPPER
        for fixture in selected
    )
    if responder_count == 0:
        responder_ucb = 1.0 - TAIL_ALPHA ** (1.0 / FINAL_N)
        tail_branch = "zero_am_gm"
    else:
        responder_ucb = min(
            1.0,
            responder_count / FINAL_N
            + math.sqrt(math.log(1.0 / TAIL_ALPHA) / (2.0 * FINAL_N)),
        )
        tail_branch = "nonzero_hoeffding"

    clipped_primary_ucb = max(
        cell_results[cell]["ucb"] for cell in CELLS)
    full_cell_resolved = (
        cell_results["full_kv"]["ucb"] <= DELTA_CLIP)
    value_cell_resolved = (
        cell_results["value_only"]["ucb"] <= DELTA_CLIP)
    clipped_means_resolved = clipped_primary_ucb <= DELTA_CLIP
    joint_resolved = (
        clipped_means_resolved and responder_ucb <= DELTA_TAIL)

    return {
        "schema": SCHEMA,
        "design_id": DESIGN_ID,
        "alpha_ledger": _alpha_ledger(),
        "n_collapsed_input": len(fixtures),
        "n_total": FINAL_N,
        "cells": cell_results,
        "clipped_primary_ucb": clipped_primary_ucb,
        "responder_count": responder_count,
        "responder_ucb": responder_ucb,
        "tail_branch": tail_branch,
        "joint_resolved": joint_resolved,
        "clipped_means_resolved": clipped_means_resolved,
        "value_cell_resolved": value_cell_resolved,
        "full_cell_resolved": full_cell_resolved,
        "numeric_conclusion_label": _numeric_conclusion_label(
            joint_resolved=joint_resolved,
            full_cell_resolved=full_cell_resolved,
            value_cell_resolved=value_cell_resolved,
        ),
        "case_ids": sorted(str(fixture["case_id"])
                           for fixture in selected),
        "selected_collapsed": [dict(fixture) for fixture in selected],
    }


def canonical_bytes(result: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _read_rows(path: str) -> list[Mapping[str, object]]:
    text = sys.stdin.read() if path == "-" else Path(path).read_text()
    document = json.loads(text)
    _require(isinstance(document, list), "input JSON is not a row list")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="render-row JSON list, or - for stdin (default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write canonical JSON here; fails if the path already exists",
    )
    args = parser.parse_args()
    output = canonical_bytes(recompute(_read_rows(args.input)))
    if args.output is None:
        sys.stdout.buffer.write(output)
    else:
        with args.output.open("xb") as stream:
            stream.write(output)


if __name__ == "__main__":
    main()
