"""Frozen statistical core for coherent-state powered successor v13.

This module contains no model code and performs no file discovery.  Callers
must supply literal render-level rows from the release-bound primary sample.
The module enforces the independent-unit contract: exactly two render origins
collapse to one fixture value, and balanced looks contain the first 3/4/6
eligible fixtures from every one of eight fixed strata.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


DESIGN_ID = "coherent-state-powered-successor-v13"
STRATA = tuple(f"s{index}" for index in range(1, 9))
CELLS = ("full_kv", "value_only")
RENDER_ORIGINS = ("C", "W")
LOOKS = ((24, 3, 0.005), (32, 4, 0.010), (48, 6, 0.035))
FAMILY_ALPHA = 0.05
DELTA_NAT = 0.25


class V13StatsError(ValueError):
    """Input differs from the frozen v13 independent-unit/statistical design."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V13StatsError(message)


def _finite_number(value: object, label: str) -> float:
    _require(isinstance(value, (int, float, np.integer, np.floating)),
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
        _require(isinstance(self.eligible_rank, int) and self.eligible_rank >= 1,
                 "eligible_rank must be a positive integer")
        _finite_number(self.full_kv, "full_kv")
        _finite_number(self.value_only, "value_only")
        _require(len(self.render_values) == 2,
                 f"{self.case_id} does not have exactly two renders")
        origins = tuple(row[0] for row in self.render_values)
        _require(set(origins) == set(RENDER_ORIGINS) and len(set(origins)) == 2,
                 f"{self.case_id} render origins differ from C/W")
        for origin, full_kv, value_only in self.render_values:
            _require(origin in RENDER_ORIGINS, "invalid render origin")
            _finite_number(full_kv, f"{self.case_id}:{origin}:full_kv")
            _finite_number(value_only, f"{self.case_id}:{origin}:value_only")
        _require(math.isclose(
            self.full_kv,
            sum(row[1] for row in self.render_values) / 2.0,
            rel_tol=0.0, abs_tol=1e-15),
            f"{self.case_id} full_kv is not the render mean")
        _require(math.isclose(
            self.value_only,
            sum(row[2] for row in self.render_values) / 2.0,
            rel_tol=0.0, abs_tol=1e-15),
            f"{self.case_id} value_only is not the render mean")
        return self


def collapse_render_rows(rows: Iterable[Mapping[str, object]]) -> list[FixtureValue]:
    """Collapse exactly one C-origin and one W-origin row per fixture."""
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
        _require(set(origins) == set(RENDER_ORIGINS) and len(set(origins)) == 2,
                 f"{case_id} render origins differ from C/W")
        strata = {row.get("stratum") for row in case_rows}
        ranks = {row.get("eligible_rank") for row in case_rows}
        _require(len(strata) == 1 and next(iter(strata)) in STRATA,
                 f"{case_id} stratum differs")
        _require(len(ranks) == 1 and isinstance(next(iter(ranks)), int),
                 f"{case_id} eligible rank differs")
        ordered = sorted(case_rows, key=lambda row: str(row["render_origin"]))
        render_values = tuple(
            (str(row["render_origin"]),
             _finite_number(row.get("full_kv"), f"{case_id}:full_kv"),
             _finite_number(row.get("value_only"), f"{case_id}:value_only"))
            for row in ordered
        )
        fixture = FixtureValue(
            case_id=case_id,
            stratum=str(next(iter(strata))),
            eligible_rank=int(next(iter(ranks))),
            full_kv=sum(row[1] for row in render_values) / 2.0,
            value_only=sum(row[2] for row in render_values) / 2.0,
            render_values=render_values,
        ).validate()
        fixtures.append(fixture)

    case_ids = [fixture.case_id for fixture in fixtures]
    _require(len(case_ids) == len(set(case_ids)), "duplicate fixture IDs")
    return sorted(fixtures, key=lambda fixture: (fixture.stratum,
                                                  fixture.eligible_rank,
                                                  fixture.case_id))


def select_balanced_look(fixtures: Sequence[FixtureValue], n_total: int
                         ) -> list[FixtureValue]:
    """Select exactly ranks 1..m in every stratum, masking any overshoot."""
    look = next((row for row in LOOKS if row[0] == n_total), None)
    _require(look is not None, f"N={n_total} is not a frozen look")
    per_stratum = int(look[1])
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
            key=lambda fixture: fixture.eligible_rank)
        ranks = [fixture.eligible_rank for fixture in members]
        _require(len(ranks) == len(set(ranks)),
                 f"duplicate eligible rank in {stratum}")
        required = list(range(1, per_stratum + 1))
        _require(ranks[:per_stratum] == required,
                 f"{stratum} lacks frozen ranks {required}")
        selected.extend(members[:per_stratum])
    _require(len(selected) == n_total, "balanced look size differs")
    return selected


@dataclass(frozen=True)
class StratifiedCellResult:
    cell: str
    estimate: float
    standard_error: float
    degrees_of_freedom: float
    critical_value: float
    ucb: float
    gamma: float
    stratum_means: Mapping[str, float]
    stratum_sds: Mapping[str, float]


def stratified_cell_ucb(fixtures: Sequence[FixtureValue], *, cell: str,
                        alpha_for_look: float) -> StratifiedCellResult:
    _require(cell in CELLS, f"unknown primary cell {cell!r}")
    _require(0.0 < alpha_for_look < 1.0, "look alpha outside (0,1)")
    gamma = alpha_for_look / len(CELLS)
    weight = 1.0 / len(STRATA)
    means: dict[str, float] = {}
    sds: dict[str, float] = {}
    variance_terms: list[float] = []
    df_terms: list[float] = []
    for stratum in STRATA:
        values = np.asarray([
            getattr(fixture, cell) for fixture in fixtures
            if fixture.stratum == stratum
        ], dtype=np.float64)
        _require(values.size >= 2, f"{stratum} has fewer than two fixtures")
        _require(np.isfinite(values).all(), f"{stratum}:{cell} is nonfinite")
        mean = float(np.mean(values))
        # NumPy's two-pass arithmetic can return an ~1e-34 artifact for a
        # literally constant non-binary float such as 0.1.  The frozen zero-SD
        # branch is about exact observed equality, so detect that explicitly.
        variance = (0.0 if bool(np.all(values == values[0]))
                    else float(np.var(values, ddof=1)))
        sd = math.sqrt(variance)
        component = weight * weight * variance / values.size
        means[stratum] = mean
        sds[stratum] = sd
        variance_terms.append(component)
        if component > 0.0:
            df_terms.append(component * component / (values.size - 1))
    estimate = sum(weight * means[stratum] for stratum in STRATA)
    variance_estimate = sum(variance_terms)
    standard_error = math.sqrt(variance_estimate)
    if variance_estimate == 0.0:
        degrees_of_freedom = math.inf
        critical = 0.0
        ucb = estimate
    else:
        denominator = sum(df_terms)
        _require(denominator > 0.0, "Satterthwaite denominator is zero")
        degrees_of_freedom = variance_estimate * variance_estimate / denominator
        _require(math.isfinite(degrees_of_freedom) and degrees_of_freedom > 0.0,
                 "invalid Satterthwaite degrees of freedom")
        critical = float(stats.t.ppf(1.0 - gamma, degrees_of_freedom))
        _require(math.isfinite(critical) and critical > 0.0,
                 "invalid Student-t critical value")
        ucb = estimate + critical * standard_error
    return StratifiedCellResult(
        cell=cell, estimate=estimate, standard_error=standard_error,
        degrees_of_freedom=degrees_of_freedom, critical_value=critical,
        ucb=ucb, gamma=gamma, stratum_means=means, stratum_sds=sds)


@dataclass(frozen=True)
class LookResult:
    n_total: int
    per_stratum: int
    alpha_for_look: float
    cells: Mapping[str, StratifiedCellResult]
    primary_ucb: float
    stop_for_bound: bool
    case_ids: tuple[str, ...]


def analyze_look(fixtures: Sequence[FixtureValue], n_total: int,
                 *, delta_nat: float = DELTA_NAT) -> LookResult:
    _require(math.isfinite(delta_nat), "delta is nonfinite")
    look = next((row for row in LOOKS if row[0] == n_total), None)
    _require(look is not None, f"N={n_total} is not a frozen look")
    selected = select_balanced_look(fixtures, n_total)
    alpha = float(look[2])
    cells = {
        cell: stratified_cell_ucb(selected, cell=cell,
                                  alpha_for_look=alpha)
        for cell in CELLS
    }
    primary_ucb = max(result.ucb for result in cells.values())
    return LookResult(
        n_total=n_total, per_stratum=int(look[1]), alpha_for_look=alpha,
        cells=cells, primary_ucb=primary_ucb,
        stop_for_bound=primary_ucb <= delta_nat,
        case_ids=tuple(sorted(fixture.case_id for fixture in selected)))


def sequential_decision(fixtures: Sequence[FixtureValue], *,
                        delta_nat: float = DELTA_NAT
                        ) -> tuple[LookResult, tuple[LookResult, ...]]:
    """Evaluate only frozen looks and return the selected terminal look."""
    results: list[LookResult] = []
    for n_total, _, _ in LOOKS:
        result = analyze_look(fixtures, n_total, delta_nat=delta_nat)
        results.append(result)
        if result.stop_for_bound:
            break
    return results[-1], tuple(results)


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
    return min(upper, float(np.mean(array)) + radius)


def alpha_ledger() -> dict[str, object]:
    events = [
        {"n_total": n_total, "cell": cell, "alpha": alpha / len(CELLS)}
        for n_total, _, alpha in LOOKS for cell in CELLS
    ]
    spent = math.fsum(float(event["alpha"]) for event in events)
    _require(math.isclose(spent, FAMILY_ALPHA, rel_tol=0.0, abs_tol=1e-15),
             "primary alpha ledger does not sum to 0.05")
    return {"family_alpha": FAMILY_ALPHA, "spent": spent, "events": events}
