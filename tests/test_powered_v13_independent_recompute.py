import ast
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest

from powered_v13_stats import (
    CELLS,
    DESIGN_ID,
    STRATA,
    V13StatsError,
    alpha_ledger,
    analyze_primary,
    collapse_render_rows,
    select_final_sample,
)


ROOT = Path(__file__).resolve().parents[1]
INDEPENDENT_SCRIPT = (
    ROOT / "scripts" / "recompute_powered_v13_bound_independent.py")
SCHEMA = "coherent_state_powered_v13_final_bound_projection_v1"


def _render_rows(per_stratum, value_fn, render_fn=None):
    rows = []
    for stratum_index, stratum in enumerate(STRATA, start=1):
        for rank in range(1, per_stratum + 1):
            full_kv, value_only = value_fn(stratum_index, rank)
            case_id = f"{stratum}-c{rank}"
            for render_id in ("r1", "r2"):
                render_full, render_value = (
                    (full_kv, value_only)
                    if render_fn is None
                    else render_fn(
                        stratum_index,
                        rank,
                        render_id,
                        full_kv,
                        value_only,
                    )
                )
                rows.append({
                    "case_id": case_id,
                    "stratum": stratum,
                    "eligible_rank": rank,
                    "render_id": render_id,
                    "render_origin": "C",
                    "full_kv": render_full,
                    "value_only": render_value,
                })
    return rows


def _mixed_values(stratum_index, rank):
    # Dyadic values make exact-byte comparison independent of summation-library
    # rounding while exercising both clipping directions and CELL_BOUND_ONLY.
    full_kv = (
        0.1875
        + (stratum_index - 4.5) / 16.0
        + (rank - 3.5) / 32.0
    )
    value_only = (
        -(stratum_index - 4.5) / 32.0
        + (rank - 3.5) / 16.0
    )
    if (stratum_index, rank) == (1, 1):
        full_kv = -0.75
    if (stratum_index, rank) == (8, 6):
        value_only = 0.75
    if (stratum_index, rank) == (4, 3):
        full_kv = 0.5
    return full_kv, value_only


def _mixed_renders(
    stratum_index,
    rank,
    render_id,
    full_kv,
    value_only,
):
    del stratum_index, rank
    full_delta = -1.0 / 64.0 if render_id == "r1" else 1.0 / 64.0
    value_delta = 1.0 / 128.0 if render_id == "r1" else -1.0 / 128.0
    return full_kv + full_delta, value_only + value_delta


def _case_rows():
    zero = _render_rows(6, lambda _stratum, _rank: (0.0, 0.0))
    one_responder = _render_rows(
        6,
        lambda stratum, rank: (
            5.0 if (stratum, rank) == (1, 1) else 0.0,
            0.0,
        ),
    )
    mixed = _render_rows(6, _mixed_values, _mixed_renders)
    shuffled = [dict(row) for row in mixed]
    random.Random(20260712).shuffle(shuffled)
    overshoot = _render_rows(
        8,
        lambda _stratum, rank: (
            5.0 if rank == 7 else 0.0,
            5.0 if rank == 8 else 0.0,
        ),
    )
    non_dyadic_rng = random.Random(2)
    non_dyadic = _render_rows(
        6,
        lambda _stratum, _rank: (
            non_dyadic_rng.random() - 0.5,
            non_dyadic_rng.random() - 0.5,
        ),
    )
    cancellation_pattern = (
        1e16, 1.0, -1e16, -1e16, 1.0, 1e16)
    catastrophic_cancellation = _render_rows(
        6,
        lambda _stratum, rank: (
            cancellation_pattern[rank - 1],
            0.0,
        ),
    )
    return {
        "constant_zero": zero,
        "one_5nat_responder": one_responder,
        "mixed_strata": mixed,
        "row_shuffle": shuffled,
        "overshoot": overshoot,
        "non_dyadic": non_dyadic,
        "catastrophic_cancellation": catastrophic_cancellation,
    }


def _numeric_conclusion_label(result):
    if result.joint_resolved:
        return "JOINT_BOUND_RESOLVED"
    if result.full_cell_resolved and result.value_cell_resolved:
        return "CLIPPED_MEAN_BOUND_ONLY"
    if result.full_cell_resolved or result.value_cell_resolved:
        return "CELL_BOUND_ONLY"
    return "BOUND_NOT_RESOLVED_AT_N48"


def _production_projection(rows):
    fixtures = collapse_render_rows(rows)
    selected = select_final_sample(fixtures)
    result = analyze_primary(fixtures)
    cells = {}
    for cell in CELLS:
        value = result.cells[cell]
        cells[cell] = {
            "cell": value.cell,
            "raw_mean": value.raw_mean,
            "clipped_mean": value.clipped_mean,
            "radius": value.radius,
            "ucb": value.ucb,
            "alpha": value.alpha,
            "clipped_count_low": value.clipped_count_low,
            "clipped_count_high": value.clipped_count_high,
            "stratum_raw_means": dict(value.stratum_raw_means),
            "stratum_clipped_means": dict(value.stratum_clipped_means),
        }
    selected_collapsed = [{
        "case_id": fixture.case_id,
        "stratum": fixture.stratum,
        "eligible_rank": fixture.eligible_rank,
        "full_kv": fixture.full_kv,
        "value_only": fixture.value_only,
        "render_values": [list(row) for row in fixture.render_values],
    } for fixture in selected]
    return {
        "schema": SCHEMA,
        "design_id": DESIGN_ID,
        "alpha_ledger": alpha_ledger(),
        "n_collapsed_input": len(fixtures),
        "n_total": result.n_total,
        "cells": cells,
        "clipped_primary_ucb": result.clipped_primary_ucb,
        "responder_count": result.responder_count,
        "responder_ucb": result.responder_ucb,
        "tail_branch": (
            "zero_am_gm"
            if result.responder_count == 0
            else "nonzero_hoeffding"
        ),
        "joint_resolved": result.joint_resolved,
        "clipped_means_resolved": result.clipped_means_resolved,
        "value_cell_resolved": result.value_cell_resolved,
        "full_cell_resolved": result.full_cell_resolved,
        "numeric_conclusion_label": _numeric_conclusion_label(result),
        "case_ids": list(result.case_ids),
        "selected_collapsed": selected_collapsed,
    }


def _canonical_bytes(document):
    return (json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n").encode()


def _independent_bytes(rows):
    completed = subprocess.run(
        [sys.executable, str(INDEPENDENT_SCRIPT)],
        cwd=ROOT,
        input=json.dumps(rows, separators=(",", ":")).encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    assert completed.stderr == b""
    return completed.stdout


GOLDEN = {
    "constant_zero": {
        "sha256": "d92573326debf06a6e3eba32e53efa5034a991e2a0a19b9f4caa52552fc47178",
        "n_collapsed_input": 48,
        "ucbs": (0.20186688594189123, 0.20186688594189123),
        "responder_count": 0,
        "responder_ucb": 0.09148242434831322,
        "label": "JOINT_BOUND_RESOLVED",
    },
    "one_5nat_responder": {
        "sha256": "1bf6ba2ecba896a74578edb17e6b60152f8de6f9f3e9efb42dab35b3cd8d0159",
        "n_collapsed_input": 48,
        "ucbs": (0.21228355260855788, 0.20186688594189123),
        "responder_count": 1,
        "responder_ucb": 0.2398550737398722,
        "label": "CLIPPED_MEAN_BOUND_ONLY",
    },
    "mixed_strata": {
        "sha256": "cde740a1cd8291884997c127cfbd4e1717676c51a19bf90be0e863305ba678d4",
        "n_collapsed_input": 48,
        "ucbs": (0.38871584427522454, 0.21130699010855788),
        "responder_count": 1,
        "responder_ucb": 0.2398550737398722,
        "label": "CELL_BOUND_ONLY",
    },
    "row_shuffle": {
        "sha256": "cde740a1cd8291884997c127cfbd4e1717676c51a19bf90be0e863305ba678d4",
        "n_collapsed_input": 48,
        "ucbs": (0.38871584427522454, 0.21130699010855788),
        "responder_count": 1,
        "responder_ucb": 0.2398550737398722,
        "label": "CELL_BOUND_ONLY",
    },
    "overshoot": {
        "sha256": "efc9c5672dfdf9ccd080eabb9737909584e9680db69959f1fb4f538877ce8031",
        "n_collapsed_input": 64,
        "ucbs": (0.20186688594189123, 0.20186688594189123),
        "responder_count": 0,
        "responder_ucb": 0.09148242434831322,
        "label": "JOINT_BOUND_RESOLVED",
    },
    "non_dyadic": {
        "sha256": "e64557104e592b39e9885dea5f6929a27c09adc85631c78b15143f53742b7825",
        "n_collapsed_input": 48,
        "ucbs": (0.2951307236899457, 0.21352439101127596),
        "responder_count": 0,
        "responder_ucb": 0.09148242434831322,
        "label": "JOINT_BOUND_RESOLVED",
    },
    "catastrophic_cancellation": {
        "sha256": "022a6a1df68e5593327720523978ad4d906cdd1c72c2eb7961e6302a70f13fdc",
        "n_collapsed_input": 48,
        "ucbs": (0.3685335526085579, 0.20186688594189123),
        "responder_count": 32,
        "responder_ucb": 0.8856884070732055,
        "label": "CELL_BOUND_ONLY",
    },
}


def test_independent_recompute_has_only_stdlib_imports():
    source = INDEPENDENT_SCRIPT.read_text()
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots <= {
        "__future__", "argparse", "fractions", "json", "math", "pathlib",
        "sys", "typing",
    }
    assert "powered_v13_stats" not in source


@pytest.mark.parametrize("case_name", tuple(GOLDEN))
def test_independent_recompute_matches_production_canonical_bytes(case_name):
    rows = _case_rows()[case_name]
    production_bytes = _canonical_bytes(_production_projection(rows))
    independent_bytes = _independent_bytes(rows)
    assert independent_bytes == production_bytes

    observed = json.loads(independent_bytes)
    expected = GOLDEN[case_name]
    assert hashlib.sha256(independent_bytes).hexdigest() == expected["sha256"]
    assert observed["n_collapsed_input"] == expected["n_collapsed_input"]
    assert tuple(observed["cells"][cell]["ucb"] for cell in CELLS) == (
        expected["ucbs"])
    assert observed["responder_count"] == expected["responder_count"]
    assert observed["responder_ucb"] == expected["responder_ucb"]
    assert observed["numeric_conclusion_label"] == expected["label"]


def test_shuffle_and_overshoot_selection_invariants():
    cases = _case_rows()
    mixed = _independent_bytes(cases["mixed_strata"])
    shuffled = _independent_bytes(cases["row_shuffle"])
    assert shuffled == mixed

    zero = json.loads(_independent_bytes(cases["constant_zero"]))
    overshoot = json.loads(_independent_bytes(cases["overshoot"]))
    assert overshoot["n_collapsed_input"] == 64
    zero.pop("n_collapsed_input")
    overshoot.pop("n_collapsed_input")
    assert overshoot == zero
    assert all(
        int(case_id.rsplit("c", 1)[1]) <= 6
        for case_id in overshoot["case_ids"]
    )


@pytest.mark.parametrize(
    "case_name", ("non_dyadic", "catastrophic_cancellation"))
def test_stable_fsum_goldens_expose_previous_numpy_mean_drift(case_name):
    fixtures = collapse_render_rows(_case_rows()[case_name])
    selected = select_final_sample(fixtures)
    values = [fixture.full_kv for fixture in selected]
    previous_numpy_mean = float(np.asarray(values, dtype=np.float64).mean())
    stable_mean = math.fsum(values) / len(values)
    assert previous_numpy_mean != stable_mean
    assert analyze_primary(fixtures).cells["full_kv"].raw_mean == stable_mean


@pytest.mark.parametrize(
    ("field", "value"),
    (("full_kv", True), ("eligible_rank", True)),
)
def test_boolean_corruption_fails_closed_in_both_paths(field, value):
    rows = _case_rows()["constant_zero"]
    rows[0][field] = value
    rows[1][field] = value
    with pytest.raises(V13StatsError):
        collapse_render_rows(rows)
    completed = subprocess.run(
        [sys.executable, str(INDEPENDENT_SCRIPT)],
        cwd=ROOT,
        input=json.dumps(rows, separators=(",", ":")).encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode != 0
    assert completed.stdout == b""


def test_alpha_ledger_is_exact_in_decimal_rationals():
    ledger = alpha_ledger()
    exact = sum(
        (Fraction(str(event["alpha"])) for event in ledger["events"]),
        start=Fraction(0, 1),
    )
    assert exact == Fraction(1, 20)
