"""Frozen statistical core for coherent-state powered successor v13.

This module contains no model code and performs no file discovery.  Callers
must supply literal render-level rows from the release-bound primary sample.
The module enforces the independent-unit contract: exactly two independently
seeded C-origin renders collapse to one fixture value, and the sole inferential
analysis contains six fixtures from every one of eight fixed strata.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


DESIGN_ID = "coherent-state-powered-successor-v13"
STRATA = tuple(f"s{index}" for index in range(1, 9))
CELLS = ("full_kv", "value_only")
RENDER_IDS = ("r1", "r2")
FINAL_N = 48
PER_STRATUM = 6
_FAMILY_ALPHA_EXACT = Fraction(1, 20)
_MEAN_FAMILY_ALPHA_EXACT = Fraction(1, 25)
_TAIL_ALPHA_EXACT = Fraction(1, 100)
FAMILY_ALPHA = float(_FAMILY_ALPHA_EXACT)
MEAN_FAMILY_ALPHA = float(_MEAN_FAMILY_ALPHA_EXACT)
TAIL_ALPHA = float(_TAIL_ALPHA_EXACT)
CLIP_LOWER = -0.5
CLIP_UPPER = 0.5
DELTA_CLIP = 0.35
DELTA_TAIL = 0.10


class V13StatsError(ValueError):
    """Input differs from the frozen v13 independent-unit/statistical design."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V13StatsError(message)


def _finite_number(value: object, label: str) -> float:
    _require(not isinstance(value, (bool, np.bool_)) and
             isinstance(value, (int, float, np.integer, np.floating)),
             f"{label} is not numeric")
    out = float(value)
    _require(math.isfinite(out), f"{label} is nonfinite")
    return out


@dataclass(frozen=True)
class FixtureValue:
    case_id: str
    stratum: str
    eligible_rank: int
    full_kv: float
    value_only: float
    render_values: tuple[tuple[str, float, float], ...]

    def validate(self) -> "FixtureValue":
        _require(isinstance(self.case_id, str) and self.case_id,
                 "case_id is empty")
        _require(self.stratum in STRATA, f"unknown stratum {self.stratum!r}")
        _require(not isinstance(self.eligible_rank, bool) and
                 isinstance(self.eligible_rank, int) and
                 self.eligible_rank >= 1,
                 "eligible_rank must be a positive integer")
        _finite_number(self.full_kv, "full_kv")
        _finite_number(self.value_only, "value_only")
        _require(len(self.render_values) == 2,
                 f"{self.case_id} does not have exactly two renders")
        render_ids = tuple(row[0] for row in self.render_values)
        _require(set(render_ids) == set(RENDER_IDS) and len(set(render_ids)) == 2,
                 f"{self.case_id} render IDs differ from r1/r2")
        for render_id, full_kv, value_only in self.render_values:
            _require(render_id in RENDER_IDS, "invalid render ID")
            _finite_number(full_kv, f"{self.case_id}:{render_id}:full_kv")
            _finite_number(value_only, f"{self.case_id}:{render_id}:value_only")
        _require(
            self.full_kv ==
            math.fsum(row[1] for row in self.render_values) / 2.0,
            f"{self.case_id} full_kv is not the render mean")
        _require(
            self.value_only ==
            math.fsum(row[2] for row in self.render_values) / 2.0,
            f"{self.case_id} value_only is not the render mean")
        return self


def collapse_render_rows(rows: Iterable[Mapping[str, object]]) -> list[FixtureValue]:
    """Collapse exactly two unique C-origin render rows per fixture."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "render row is not a mapping")
        case_id = row.get("case_id")
        _require(isinstance(case_id, str) and case_id,
                 "render row case_id is empty")
        grouped.setdefault(case_id, []).append(row)
    _require(bool(grouped), "no render rows")

    fixtures: list[FixtureValue] = []
    for case_id, case_rows in grouped.items():
        _require(len(case_rows) == 2,
                 f"{case_id} has {len(case_rows)} render rows, expected two")
        origins = [row.get("render_origin") for row in case_rows]
        _require(len(origins) == 2 and all(origin == "C" for origin in origins),
                 f"{case_id} renders are not both C-origin")
        render_ids = [row.get("render_id") for row in case_rows]
        _require(set(render_ids) == set(RENDER_IDS) and len(set(render_ids)) == 2,
                 f"{case_id} render IDs differ from r1/r2")
        strata = {row.get("stratum") for row in case_rows}
        ranks = {row.get("eligible_rank") for row in case_rows}
        _require(len(strata) == 1 and next(iter(strata)) in STRATA,
                 f"{case_id} stratum differs")
        _require(len(ranks) == 1 and
                 not isinstance(next(iter(ranks)), bool) and
                 isinstance(next(iter(ranks)), int),
                 f"{case_id} eligible rank differs")
        ordered = sorted(case_rows, key=lambda row: str(row["render_id"]))
        render_values = tuple(
            (str(row["render_id"]),
             _finite_number(row.get("full_kv"), f"{case_id}:full_kv"),
             _finite_number(row.get("value_only"), f"{case_id}:value_only"))
            for row in ordered
        )
        fixture = FixtureValue(
            case_id=case_id,
            stratum=str(next(iter(strata))),
            eligible_rank=int(next(iter(ranks))),
            full_kv=math.fsum(row[1] for row in render_values) / 2.0,
            value_only=math.fsum(row[2] for row in render_values) / 2.0,
            render_values=render_values,
        ).validate()
        fixtures.append(fixture)

    case_ids = [fixture.case_id for fixture in fixtures]
    _require(len(case_ids) == len(set(case_ids)), "duplicate fixture IDs")
    return sorted(fixtures, key=lambda fixture: (fixture.stratum,
                                                  fixture.eligible_rank,
                                                  fixture.case_id))


def select_final_sample(fixtures: Sequence[FixtureValue]) -> list[FixtureValue]:
    """Select exactly eligible ranks 1..6 in every stratum."""
    seen_ids: set[str] = set()
    selected: list[FixtureValue] = []
    for fixture in fixtures:
        fixture.validate()
        _require(fixture.case_id not in seen_ids,
                 f"duplicate fixture ID {fixture.case_id}")
        seen_ids.add(fixture.case_id)
    for stratum in STRATA:
        members = sorted(
            (fixture for fixture in fixtures if fixture.stratum == stratum),
            key=lambda fixture: (fixture.eligible_rank, fixture.case_id))
        ranks = [fixture.eligible_rank for fixture in members]
        _require(len(ranks) == len(set(ranks)),
                 f"duplicate eligible rank in {stratum}")
        required = list(range(1, PER_STRATUM + 1))
        _require(ranks[:PER_STRATUM] == required,
                 f"{stratum} lacks frozen ranks {required}")
        selected.extend(members[:PER_STRATUM])
    _require(len(selected) == FINAL_N, "final sample size differs")
    return selected


@dataclass(frozen=True)
class BoundedCellResult:
    cell: str
    raw_mean: float
    clipped_mean: float
    radius: float
    ucb: float
    alpha: float
    clipped_count_low: int
    clipped_count_high: int
    stratum_raw_means: Mapping[str, float]
    stratum_clipped_means: Mapping[str, float]


@dataclass(frozen=True)
class PrimaryResult:
    n_total: int
    cells: Mapping[str, BoundedCellResult]
    clipped_primary_ucb: float
    responder_count: int
    responder_ucb: float
    joint_resolved: bool
    clipped_means_resolved: bool
    value_cell_resolved: bool
    full_cell_resolved: bool
    case_ids: tuple[str, ...]


def responder_prevalence_ucb(successes: int, n: int, alpha: float) -> float:
    """Distribution-free bound for average responder prevalence.

    Zero successes uses the AM--GM inversion, valid for heterogeneous unit
    prevalences.  Nonzero counts use Hoeffding for independent bounded
    indicators (and is also conservative for sampling without replacement).
    """
    _require(isinstance(successes, int) and isinstance(n, int),
             "responder counts must be integers")
    _require(0 <= successes <= n and n > 0, "invalid responder counts")
    _require(0.0 < alpha < 1.0, "responder alpha outside (0,1)")
    if successes == 0:
        return 1.0 - alpha ** (1.0 / n)
    return min(1.0, successes / n + math.sqrt(
        math.log(1.0 / alpha) / (2.0 * n)))


def analyze_primary(fixtures: Sequence[FixtureValue]) -> PrimaryResult:
    # Arithmetic order is frozen: STRATA order, then eligible rank/case ID,
    # then CELLS order.  All reported means use math.fsum so the production and
    # stdlib-only recomputation do not inherit a backend reduction algorithm.
    selected = select_final_sample(fixtures)
    cell_results: dict[str, BoundedCellResult] = {}
    gamma = float(_MEAN_FAMILY_ALPHA_EXACT / len(CELLS))
    radius = (CLIP_UPPER - CLIP_LOWER) * math.sqrt(
        math.log(1.0 / gamma) / (2.0 * FINAL_N))
    for cell in CELLS:
        raw = np.asarray([getattr(fixture, cell) for fixture in selected],
                         dtype=np.float64)
        _require(np.isfinite(raw).all(), f"{cell} contains nonfinite values")
        clipped = np.clip(raw, CLIP_LOWER, CLIP_UPPER)
        raw_strata: dict[str, float] = {}
        clipped_strata: dict[str, float] = {}
        for stratum in STRATA:
            mask = np.asarray([fixture.stratum == stratum
                               for fixture in selected])
            _require(int(mask.sum()) == PER_STRATUM,
                     f"{stratum} count differs")
            raw_strata[stratum] = (
                math.fsum(float(value) for value in raw[mask]) /
                PER_STRATUM)
            clipped_strata[stratum] = (
                math.fsum(float(value) for value in clipped[mask]) /
                PER_STRATUM)
        clipped_mean = math.fsum(float(value) for value in clipped) / FINAL_N
        cell_results[cell] = BoundedCellResult(
            cell=cell,
            raw_mean=math.fsum(float(value) for value in raw) / FINAL_N,
            clipped_mean=clipped_mean, radius=radius,
            ucb=min(CLIP_UPPER, clipped_mean + radius), alpha=gamma,
            clipped_count_low=int(np.sum(raw < CLIP_LOWER)),
            clipped_count_high=int(np.sum(raw > CLIP_UPPER)),
            stratum_raw_means=raw_strata,
            stratum_clipped_means=clipped_strata)
    responders = sum(
        max(fixture.full_kv, fixture.value_only) > CLIP_UPPER
        for fixture in selected)
    tail_ucb = responder_prevalence_ucb(responders, FINAL_N, TAIL_ALPHA)
    primary_ucb = max(result.ucb for result in cell_results.values())
    return PrimaryResult(
        n_total=FINAL_N, cells=cell_results,
        clipped_primary_ucb=primary_ucb,
        responder_count=int(responders), responder_ucb=tail_ucb,
        joint_resolved=(primary_ucb <= DELTA_CLIP and
                        tail_ucb <= DELTA_TAIL),
        clipped_means_resolved=primary_ucb <= DELTA_CLIP,
        value_cell_resolved=cell_results["value_only"].ucb <= DELTA_CLIP,
        full_cell_resolved=cell_results["full_kv"].ucb <= DELTA_CLIP,
        case_ids=tuple(sorted(fixture.case_id for fixture in selected)))


@dataclass(frozen=True)
class NominalCellResult:
    cell: str
    estimate: float
    standard_error: float | None
    degrees_of_freedom: float | None
    critical_value: float | None
    ucb: float | None
    alpha: float


def nominal_stratified_t_ucb(fixtures: Sequence[FixtureValue], *, cell: str,
                             alpha: float = 0.05) -> NominalCellResult:
    """Model-based companion only; never used for a primary conclusion."""
    _require(cell in CELLS, f"unknown cell {cell!r}")
    _require(0.0 < alpha < 1.0, "nominal alpha outside (0,1)")
    selected = select_final_sample(fixtures)
    weight = 1.0 / len(STRATA)
    stratum_estimates: list[float] = []
    components: list[float] = []
    for stratum in STRATA:
        values = np.asarray([getattr(fixture, cell) for fixture in selected
                             if fixture.stratum == stratum], dtype=np.float64)
        stratum_mean = math.fsum(float(value) for value in values) / values.size
        stratum_estimates.append(weight * stratum_mean)
        variance = (0.0 if bool(np.all(values == values[0]))
                    else float(values.var(ddof=1)))
        components.append(weight * weight * variance / values.size)
    estimate = math.fsum(stratum_estimates)
    variance_estimate = math.fsum(components)
    if variance_estimate <= 0.0:
        return NominalCellResult(cell, estimate, None, None, None, None, alpha)
    denominator = math.fsum(
        component * component / (PER_STRATUM - 1)
        for component in components if component > 0.0)
    _require(denominator > 0.0, "nominal df denominator is zero")
    degrees = variance_estimate * variance_estimate / denominator
    critical = float(stats.t.ppf(1.0 - alpha, degrees))
    standard_error = math.sqrt(variance_estimate)
    return NominalCellResult(
        cell, estimate, standard_error, degrees, critical,
        estimate + critical * standard_error, alpha)


def clopper_pearson_interval(successes: int, n: int, alpha: float
                             ) -> tuple[float, float]:
    _require(isinstance(successes, int) and isinstance(n, int),
             "binomial counts must be integers")
    _require(0 <= successes <= n and n > 0, "invalid binomial counts")
    _require(0.0 < alpha < 1.0, "binomial alpha outside (0,1)")
    lower = 0.0 if successes == 0 else float(
        stats.beta.ppf(alpha / 2.0, successes, n - successes + 1))
    upper = 1.0 if successes == n else float(
        stats.beta.ppf(1.0 - alpha / 2.0, successes + 1,
                       n - successes))
    return lower, upper


def clopper_pearson_upper(successes: int, n: int, alpha: float) -> float:
    """One-sided exact upper confidence bound."""
    _require(isinstance(successes, int) and isinstance(n, int),
             "binomial counts must be integers")
    _require(0 <= successes <= n and n > 0, "invalid binomial counts")
    _require(0.0 < alpha < 1.0, "binomial alpha outside (0,1)")
    if successes == n:
        return 1.0
    return float(stats.beta.ppf(1.0 - alpha, successes + 1,
                                n - successes))


def hoeffding_ucb(values: Sequence[float], *, lower: float, upper: float,
                  alpha: float) -> float:
    """Distribution-free UCB for a predeclared bounded/clipped estimand."""
    _require(len(values) > 0, "no bounded values")
    _require(math.isfinite(lower) and math.isfinite(upper) and lower < upper,
             "invalid Hoeffding bounds")
    _require(0.0 < alpha < 1.0, "Hoeffding alpha outside (0,1)")
    array = np.asarray(values, dtype=np.float64)
    _require(np.isfinite(array).all(), "bounded values are nonfinite")
    _require(bool(np.all(array >= lower) and np.all(array <= upper)),
             "value lies outside preregistered bounds")
    radius = (upper - lower) * math.sqrt(
        math.log(1.0 / alpha) / (2.0 * len(array)))
    mean = math.fsum(float(value) for value in array) / len(array)
    return min(upper, mean + radius)


def alpha_ledger() -> dict[str, object]:
    cell_alpha_exact = _MEAN_FAMILY_ALPHA_EXACT / len(CELLS)
    exact_spent = len(CELLS) * cell_alpha_exact + _TAIL_ALPHA_EXACT
    _require(exact_spent == _FAMILY_ALPHA_EXACT,
             "exact primary alpha ledger does not sum to 1/20")
    events = [
        {"n_total": FINAL_N, "endpoint": "clipped_mean",
         "cell": cell, "alpha": float(cell_alpha_exact)}
        for cell in CELLS
    ] + [{
        "n_total": FINAL_N, "endpoint": "large_responder_prevalence",
        "cell": "any_primary_cell", "alpha": float(_TAIL_ALPHA_EXACT),
    }]
    spent = math.fsum(float(event["alpha"]) for event in events)
    _require(spent == FAMILY_ALPHA,
             "primary alpha ledger does not sum to 0.05")
    return {"family_alpha": FAMILY_ALPHA, "spent": spent, "events": events}
