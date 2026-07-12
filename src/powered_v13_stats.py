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
import hashlib
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
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 5_553_677_475_614_063_507
BOOTSTRAP_ALPHA = 0.05


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
    _require(not isinstance(successes, bool) and not isinstance(n, bool)
             and isinstance(successes, int) and isinstance(n, int),
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


def nominal_ordinary_t_ucb(fixtures: Sequence[FixtureValue], *, cell: str,
                           alpha: float = 0.05) -> NominalCellResult:
    """Ordinary-iid model-based companion; never a primary conclusion."""
    _require(cell in CELLS, f"unknown cell {cell!r}")
    _require(0.0 < alpha < 1.0, "nominal alpha outside (0,1)")
    selected = select_final_sample(fixtures)
    values = np.asarray([getattr(fixture, cell) for fixture in selected],
                        dtype=np.float64)
    estimate = math.fsum(float(value) for value in values) / FINAL_N
    if bool(np.all(values == values[0])):
        return NominalCellResult(
            cell, estimate, None, None, None, None, alpha)
    standard_error = math.sqrt(float(values.var(ddof=1)) / FINAL_N)
    degrees = float(FINAL_N - 1)
    critical = float(stats.t.ppf(1.0 - alpha, degrees))
    return NominalCellResult(
        cell, estimate, standard_error, degrees, critical,
        estimate + critical * standard_error, alpha)


@dataclass(frozen=True)
class BootstrapCellResult:
    cell: str
    estimate: float
    ucb: float
    alpha: float
    replicates: int
    replicate_values_sha256: str


@dataclass(frozen=True)
class StratifiedBootstrapResult:
    seed: int
    replicates: int
    index_stream_sha256: str
    cells: Mapping[str, BootstrapCellResult]


def stratified_cluster_bootstrap(
    fixtures: Sequence[FixtureValue],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = BOOTSTRAP_ALPHA,
) -> StratifiedBootstrapResult:
    """Frozen fixture-cluster percentile bootstrap companion.

    The same within-stratum resample indices are used for both cells.  Render
    values remain collapsed within their fixture and therefore never inflate N.
    """
    _require(not isinstance(replicates, bool) and isinstance(replicates, int)
             and replicates > 0, "bootstrap replicate count is invalid")
    _require(not isinstance(seed, bool) and isinstance(seed, int)
             and 0 <= seed < 2 ** 63, "bootstrap seed is invalid")
    _require(0.0 < alpha < 1.0, "bootstrap alpha outside (0,1)")
    selected = select_final_sample(fixtures)
    by_cell: dict[str, np.ndarray] = {}
    for cell in CELLS:
        by_cell[cell] = np.asarray([
            [getattr(fixture, cell) for fixture in selected
             if fixture.stratum == stratum]
            for stratum in STRATA
        ], dtype=np.float64)
        _require(by_cell[cell].shape == (len(STRATA), PER_STRATUM),
                 f"{cell} bootstrap stratum geometry differs")
        _require(np.isfinite(by_cell[cell]).all(),
                 f"{cell} bootstrap values are nonfinite")

    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(
        0, PER_STRATUM,
        size=(replicates, len(STRATA), PER_STRATUM),
        dtype=np.int16,
    )
    index_hash = hashlib.sha256(indices.tobytes(order="C")).hexdigest()
    stratum_axis = np.arange(len(STRATA), dtype=np.intp)[None, :, None]
    results: dict[str, BootstrapCellResult] = {}
    for cell in CELLS:
        sampled = by_cell[cell][stratum_axis, indices]
        means = sampled.sum(axis=(1, 2), dtype=np.float64) / FINAL_N
        _require(np.isfinite(means).all(),
                 f"{cell} bootstrap means are nonfinite")
        ucb = float(np.quantile(means, 1.0 - alpha, method="higher"))
        estimate = math.fsum(
            getattr(fixture, cell) for fixture in selected) / FINAL_N
        results[cell] = BootstrapCellResult(
            cell=cell,
            estimate=estimate,
            ucb=ucb,
            alpha=alpha,
            replicates=replicates,
            replicate_values_sha256=hashlib.sha256(
                means.tobytes(order="C")).hexdigest(),
        )
    return StratifiedBootstrapResult(
        seed=seed,
        replicates=replicates,
        index_stream_sha256=index_hash,
        cells=results,
    )


@dataclass(frozen=True)
class RenderVarianceResult:
    cell: str
    pooled_variance: float
    pooled_standard_deviation: float
    fixture_count: int


def pooled_within_fixture_render_variance(
    fixtures: Sequence[FixtureValue], *, cell: str,
) -> RenderVarianceResult:
    """Average of the 48 two-render within-fixture sample variances."""
    _require(cell in CELLS, f"unknown cell {cell!r}")
    selected = select_final_sample(fixtures)
    cell_index = 1 if cell == "full_kv" else 2
    variances = []
    for fixture in selected:
        left = fixture.render_values[0][cell_index]
        right = fixture.render_values[1][cell_index]
        variances.append((left - right) ** 2 / 2.0)
    pooled = math.fsum(variances) / FINAL_N
    _require(math.isfinite(pooled) and pooled >= 0.0,
             "pooled render variance is invalid")
    return RenderVarianceResult(
        cell=cell,
        pooled_variance=pooled,
        pooled_standard_deviation=math.sqrt(pooled),
        fixture_count=FINAL_N,
    )


def _frozen_median(values: Sequence[float]) -> float:
    ordered = sorted(_finite_number(value, "median value")
                     for value in values)
    _require(bool(ordered), "median values are empty")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return math.fsum((ordered[middle - 1], ordered[middle])) / 2.0


@dataclass(frozen=True)
class DescriptiveCellResult:
    cell: str
    mean: float
    median: float
    mad: float
    minimum: float
    maximum: float
    positive_count: int
    zero_count: int
    negative_count: int
    stratum_means: Mapping[str, float]
    leave_one_stratum_out_means: Mapping[str, float]
    fixture_values: tuple[
        tuple[str, str, int, float, tuple[tuple[str, float], ...]], ...]


def descriptive_cell_summary(
    fixtures: Sequence[FixtureValue], *, cell: str,
) -> DescriptiveCellResult:
    _require(cell in CELLS, f"unknown cell {cell!r}")
    selected = select_final_sample(fixtures)
    values = [getattr(fixture, cell) for fixture in selected]
    mean = math.fsum(values) / FINAL_N
    median = _frozen_median(values)
    mad = _frozen_median([abs(value - median) for value in values])
    stratum_means = {
        stratum: math.fsum(
            getattr(fixture, cell) for fixture in selected
            if fixture.stratum == stratum) / PER_STRATUM
        for stratum in STRATA
    }
    leave_one_out = {
        stratum: math.fsum(
            getattr(fixture, cell) for fixture in selected
            if fixture.stratum != stratum) / (FINAL_N - PER_STRATUM)
        for stratum in STRATA
    }
    return DescriptiveCellResult(
        cell=cell,
        mean=mean,
        median=median,
        mad=mad,
        minimum=min(values),
        maximum=max(values),
        positive_count=sum(value > 0.0 for value in values),
        zero_count=sum(value == 0.0 for value in values),
        negative_count=sum(value < 0.0 for value in values),
        stratum_means=stratum_means,
        leave_one_stratum_out_means=leave_one_out,
        fixture_values=tuple(
            (fixture.case_id, fixture.stratum, fixture.eligible_rank,
             getattr(fixture, cell), tuple(
                 (row[0], row[1] if cell == "full_kv" else row[2])
                 for row in fixture.render_values))
            for fixture in selected),
    )


@dataclass(frozen=True)
class BehavioralCountResult:
    denominator: int
    full_kv_both_render_count: int
    value_only_both_render_count: int
    either_cell_both_render_count: int
    per_fixture: tuple[tuple[str, bool, bool, bool], ...]


def behavioral_recovery_counts(
    fixtures: Sequence[FixtureValue],
    render_rows: Iterable[Mapping[str, object]],
) -> BehavioralCountResult:
    """Validate and collapse upstream token-bound behavioral render rows."""
    selected = select_final_sample(fixtures)
    selected_ids = {fixture.case_id for fixture in selected}
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in render_rows:
        _require(isinstance(row, Mapping), "behavior row is not a mapping")
        case_id = row.get("case_id")
        _require(isinstance(case_id, str) and case_id in selected_ids,
                 "behavior row case_id is outside the selected sample")
        grouped.setdefault(case_id, []).append(row)
    _require(set(grouped) == selected_ids,
             "behavior rows do not cover the selected sample")
    per_fixture = []
    for fixture in selected:
        rows = grouped[fixture.case_id]
        _require(len(rows) == len(RENDER_IDS),
                 f"{fixture.case_id} behavior render count differs")
        render_ids = [row.get("render_id") for row in rows]
        _require(set(render_ids) == set(RENDER_IDS)
                 and len(set(render_ids)) == len(RENDER_IDS),
                 f"{fixture.case_id} behavior render IDs differ")
        ordered = sorted(rows, key=lambda row: str(row["render_id"]))
        for row in ordered:
            _require(row.get("render_origin") == "C",
                     f"{fixture.case_id} behavior render origin differs")
            for field in ("a_c_begins_target", "ff_begins_target",
                          "cc_begins_target", "fc_begins_target"):
                _require(isinstance(row.get(field), bool),
                         f"{fixture.case_id} {field} is not boolean")
            for arm in ("a_c", "ff", "cc", "fc"):
                field = f"{arm}_valid_generation"
                _require(isinstance(row.get(field), bool),
                         f"{fixture.case_id} {field} is not boolean")
                _require(row[field] is True,
                         f"{fixture.case_id} {arm} generation is invalid")
            _require(row["a_c_begins_target"] is True,
                     f"{fixture.case_id} fails behavioral A_C eligibility")
            _require(row["ff_begins_target"] is False,
                     f"{fixture.case_id} fails behavioral FF eligibility")
        full = all(bool(row["cc_begins_target"]) for row in ordered)
        value = all(bool(row["fc_begins_target"]) for row in ordered)
        per_fixture.append((fixture.case_id, full, value, full or value))
    return BehavioralCountResult(
        denominator=FINAL_N,
        full_kv_both_render_count=sum(row[1] for row in per_fixture),
        value_only_both_render_count=sum(row[2] for row in per_fixture),
        either_cell_both_render_count=sum(row[3] for row in per_fixture),
        per_fixture=tuple(per_fixture),
    )


@dataclass(frozen=True)
class FiellerResult:
    numerator_mean: float
    denominator_mean: float
    ratio_estimate: float
    upper: float
    alpha: float
    degrees_of_freedom: int
    critical_value: float
    coefficient_a: float
    coefficient_b: float
    coefficient_c: float
    discriminant: float
    bounded: bool


def fieller_ratio_upper(
    numerators: Sequence[float],
    denominators: Sequence[float],
    *,
    alpha: float = 0.05,
) -> FiellerResult:
    """Ordinary paired-fixture two-sided Fieller inversion companion."""
    _require(len(numerators) == len(denominators) == FINAL_N,
             "Fieller inputs must contain exactly 48 paired fixtures")
    _require(0.0 < alpha < 1.0, "Fieller alpha outside (0,1)")
    x = [_finite_number(value, "Fieller numerator")
         for value in numerators]
    d = [_finite_number(value, "Fieller denominator")
         for value in denominators]
    xbar = math.fsum(x) / FINAL_N
    dbar = math.fsum(d) / FINAL_N
    dx = [value - xbar for value in x]
    dd = [value - dbar for value in d]
    variance_x = math.fsum(value * value for value in dx) / (FINAL_N - 1)
    variance_d = math.fsum(value * value for value in dd) / (FINAL_N - 1)
    covariance = math.fsum(
        left * right for left, right in zip(dx, dd, strict=True)
    ) / (FINAL_N - 1)
    mean_variance_x = variance_x / FINAL_N
    mean_variance_d = variance_d / FINAL_N
    mean_covariance = covariance / FINAL_N
    degrees = FINAL_N - 1
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, degrees))
    critical_squared = critical * critical
    coefficient_a = dbar * dbar - critical_squared * mean_variance_d
    coefficient_b = -2.0 * (
        xbar * dbar - critical_squared * mean_covariance)
    coefficient_c = xbar * xbar - critical_squared * mean_variance_x
    discriminant = (coefficient_b * coefficient_b -
                    4.0 * coefficient_a * coefficient_c)
    ratio = math.copysign(math.inf, xbar) if dbar == 0.0 else xbar / dbar
    bounded = bool(
        math.isfinite(coefficient_a) and coefficient_a > 0.0 and
        math.isfinite(discriminant) and discriminant >= 0.0)
    if bounded:
        upper = (-coefficient_b + math.sqrt(discriminant)) / (
            2.0 * coefficient_a)
        if not math.isfinite(upper):
            bounded = False
            upper = math.inf
    else:
        upper = math.inf
    return FiellerResult(
        numerator_mean=xbar,
        denominator_mean=dbar,
        ratio_estimate=ratio,
        upper=upper,
        alpha=alpha,
        degrees_of_freedom=degrees,
        critical_value=critical,
        coefficient_a=coefficient_a,
        coefficient_b=coefficient_b,
        coefficient_c=coefficient_c,
        discriminant=discriminant,
        bounded=bounded,
    )


def clopper_pearson_interval(successes: int, n: int, alpha: float
                             ) -> tuple[float, float]:
    _require(not isinstance(successes, bool) and not isinstance(n, bool)
             and isinstance(successes, int) and isinstance(n, int),
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
    _require(not isinstance(successes, bool) and not isinstance(n, bool)
             and isinstance(successes, int) and isinstance(n, int),
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
