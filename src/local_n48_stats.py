"""Pure frozen analysis core for local coherent-state N48 v3.

The independent sampling unit is a conversation (one ``candidate_id``).
The two fixed carrier conditions are repeated measurements that are averaged
inside a conversation; they never increase N.  This module performs no file
discovery, model execution, selection, or outcome persistence.

Required row fields are ``candidate_id``, ``stratum``, ``carrier_id``, plus
exact C/W answer scores for B, E_C, E_W, and A_C.  The VP C/W scores are the
only optional fields because the frozen protocol permits a geometrically
unavailable applied placebo; the C/W pair must be jointly present or absent in
each candidate/carrier row. The caller must supply the exact 48 IDs from the
hash-bound treatment release; this pure core deliberately performs no file
discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Callable, Iterable, Mapping, Sequence

from scipy import stats


DESIGN_ID = "coherent-state-local-mlx-n48-v3"
STRATA = (
    "arithmetic_capacity_budget",
    "categorical_set_membership",
    "conjunction_all_conditions",
    "interval_schedule_overlap",
    "ordered_priority_exception",
    "referent_alias_resolution",
    "sequential_tiebreak",
    "threshold_eligibility",
)
CARRIER_IDS = ("fixed_a", "fixed_b")
FINAL_N = 48
ROWS_PER_CONVERSATION = 2
FINAL_ROW_COUNT = FINAL_N * ROWS_PER_CONVERSATION
PER_STRATUM = 6
CLIP_LOWER = -0.5
CLIP_UPPER = 0.5
HOEFFDING_ALPHA = 0.04
TAIL_ALPHA = 0.01
T_CONFIDENCE = 0.95

_REQUIRED_NUMERIC_FIELDS = (
    "l_c_e_c",
    "l_w_e_c",
    "l_c_e_w",
    "l_w_e_w",
    "l_c_b",
    "l_w_b",
    "l_c_a_c",
    "l_w_a_c",
)
_VP_NUMERIC_FIELDS = ("l_c_vp", "l_w_vp")


class LocalN48StatsError(ValueError):
    """Input differs from the frozen local-v3 analysis contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalN48StatsError(message)


def _finite_number(value: object, label: str) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)),
             f"{label} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} is nonfinite")
    return result


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float
    confidence: float


@dataclass(frozen=True)
class MetricSummary:
    """Conversation-level descriptive/model-based companion summary."""

    n_conversations: int
    mean: float
    sample_sd: float
    standard_error: float
    t_critical: float
    t_ci_95: ConfidenceInterval
    per_carrier_means: Mapping[str, float]


@dataclass(frozen=True)
class AvailableCaseMetricSummary:
    """Descriptive applied-placebo summary with explicit geometry missingness."""

    n_condition_rows_available: int
    n_condition_rows_missing: int
    n_complete_conversations: int
    n_partial_conversations: int
    n_unavailable_conversations: int
    complete_conversation_mean: float | None
    complete_conversation_sample_sd: float | None
    complete_conversation_standard_error: float | None
    complete_conversation_t_ci_95: ConfidenceInterval | None
    per_carrier_counts: Mapping[str, int]
    per_carrier_means: Mapping[str, float | None]
    complete_candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class HoeffdingBounds:
    alpha: float
    radius: float
    mean_z: float
    lcb: float
    ucb: float
    unclamped_lcb: float
    unclamped_ucb: float


@dataclass(frozen=True)
class TailPrevalenceBound:
    threshold: float
    successes: int
    n_conversations: int
    alpha: float
    confidence: float
    clopper_pearson_ucb: float


@dataclass(frozen=True)
class ConversationEffect:
    candidate_id: str
    stratum: str
    x_by_carrier: tuple[tuple[str, float], ...]
    raw_x: float
    clipped_z: float


@dataclass(frozen=True)
class LocalN48Analysis:
    design_id: str
    n_input_rows: int
    n_conversations: int
    independent_unit: str
    carrier_ids: tuple[str, ...]
    conversation_effects: tuple[ConversationEffect, ...]
    primary_raw: MetricSummary
    primary_hoeffding: HoeffdingBounds
    raw_over_half: TailPrevalenceBound
    companion_metrics: Mapping[str, MetricSummary]
    applied_placebo_metrics: Mapping[str, AvailableCaseMetricSummary]


def _validate_alpha(alpha: float, label: str) -> float:
    value = _finite_number(alpha, label)
    _require(0.0 < value < 1.0, f"{label} is outside (0,1)")
    return value


def _hoeffding_radius(*, n: int, alpha: float,
                       lower: float, upper: float) -> float:
    _require(not isinstance(n, bool) and isinstance(n, int) and n > 0,
             "Hoeffding n must be a positive integer")
    alpha = _validate_alpha(alpha, "Hoeffding alpha")
    lower = _finite_number(lower, "Hoeffding lower")
    upper = _finite_number(upper, "Hoeffding upper")
    _require(lower < upper, "Hoeffding support is invalid")
    return (upper - lower) * math.sqrt(
        math.log(1.0 / alpha) / (2.0 * n))


def recompute_hoeffding_stdlib(
    clipped_values: Sequence[float],
    *,
    alpha: float = HOEFFDING_ALPHA,
    lower: float = CLIP_LOWER,
    upper: float = CLIP_UPPER,
) -> HoeffdingBounds:
    """Independently recompute the frozen bound using only stdlib arithmetic.

    This helper intentionally does not call the production summarization or
    radius helpers.  It is an independent check over the 48 conversation
    values, not over the 96 carrier-condition rows.
    """

    _require(len(clipped_values) == FINAL_N,
             "stdlib Hoeffding recomputation requires exactly 48 conversations")
    _require(not isinstance(alpha, bool) and isinstance(alpha, (int, float)),
             "stdlib Hoeffding alpha is not numeric")
    alpha_value = float(alpha)
    _require(math.isfinite(alpha_value) and 0.0 < alpha_value < 1.0,
             "stdlib Hoeffding alpha is outside (0,1)")
    _require(not isinstance(lower, bool) and isinstance(lower, (int, float)) and
             not isinstance(upper, bool) and isinstance(upper, (int, float)),
             "stdlib Hoeffding support is not numeric")
    lo, hi = float(lower), float(upper)
    _require(math.isfinite(lo) and math.isfinite(hi) and lo < hi,
             "stdlib Hoeffding support is invalid")
    values: list[float] = []
    for index, value in enumerate(clipped_values):
        _require(not isinstance(value, bool) and isinstance(value, (int, float)),
                 f"stdlib Hoeffding value {index} is not numeric")
        number = float(value)
        _require(math.isfinite(number),
                 f"stdlib Hoeffding value {index} is nonfinite")
        _require(lo <= number <= hi,
                 f"stdlib Hoeffding value {index} is outside support")
        values.append(number)
    mean = statistics.fmean(values)
    radius = (hi - lo) * math.sqrt(
        -math.log(alpha_value) / (2 * len(values)))
    raw_lcb, raw_ucb = mean - radius, mean + radius
    return HoeffdingBounds(
        alpha=alpha_value,
        radius=radius,
        mean_z=mean,
        lcb=max(lo, raw_lcb),
        ucb=min(hi, raw_ucb),
        unclamped_lcb=raw_lcb,
        unclamped_ucb=raw_ucb,
    )


def clopper_pearson_upper(successes: int, n: int,
                           *, alpha: float = TAIL_ALPHA) -> float:
    """Exact one-sided binomial upper bound used for raw ``X > 0.5``."""

    _require(not isinstance(successes, bool) and isinstance(successes, int) and
             not isinstance(n, bool) and isinstance(n, int),
             "Clopper-Pearson counts must be integers")
    _require(n > 0 and 0 <= successes <= n,
             "Clopper-Pearson counts are invalid")
    alpha = _validate_alpha(alpha, "Clopper-Pearson alpha")
    if successes == n:
        return 1.0
    result = float(stats.beta.ppf(
        1.0 - alpha, successes + 1, n - successes))
    _require(math.isfinite(result) and 0.0 <= result <= 1.0,
             "Clopper-Pearson result is invalid")
    return result


def _metric_summary(
    conversation_values: Sequence[float],
    values_by_carrier: Mapping[str, Sequence[float]],
) -> MetricSummary:
    _require(len(conversation_values) == FINAL_N,
             "metric summary requires exactly 48 conversations")
    values = [_finite_number(value, "conversation metric")
              for value in conversation_values]
    mean = math.fsum(values) / FINAL_N
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / math.sqrt(FINAL_N)
    t_critical = float(stats.t.ppf(0.975, FINAL_N - 1))
    _require(math.isfinite(t_critical), "t critical value is nonfinite")
    half_width = t_critical * standard_error
    carrier_means: dict[str, float] = {}
    _require(set(values_by_carrier) == set(CARRIER_IDS),
             "metric carrier IDs differ from fixed_a/fixed_b")
    for carrier_id in CARRIER_IDS:
        carrier_values = [_finite_number(value, f"{carrier_id} metric")
                          for value in values_by_carrier[carrier_id]]
        _require(len(carrier_values) == FINAL_N,
                 f"{carrier_id} metric count differs from 48")
        carrier_means[carrier_id] = math.fsum(carrier_values) / FINAL_N
    return MetricSummary(
        n_conversations=FINAL_N,
        mean=mean,
        sample_sd=sample_sd,
        standard_error=standard_error,
        t_critical=t_critical,
        t_ci_95=ConfidenceInterval(
            lower=mean - half_width,
            upper=mean + half_width,
            confidence=T_CONFIDENCE,
        ),
        per_carrier_means=carrier_means,
    )


def _available_case_summary(
    effects: Sequence[ConversationEffect],
    row_lookup: Mapping[tuple[str, str], Mapping[str, object]],
    function: Callable[[Mapping[str, object]], float],
) -> AvailableCaseMetricSummary:
    per_carrier_values: dict[str, list[float]] = {
        carrier_id: [] for carrier_id in CARRIER_IDS}
    complete_values: list[float] = []
    complete_ids: list[str] = []
    partial = 0
    unavailable = 0
    for effect in effects:
        pair_values: list[float] = []
        for carrier_id in CARRIER_IDS:
            row = row_lookup[(effect.candidate_id, carrier_id)]
            if not all(field in row for field in _VP_NUMERIC_FIELDS):
                continue
            value = function(row)
            _require(math.isfinite(value),
                     f"{effect.candidate_id}:{carrier_id}:VP metric is nonfinite")
            pair_values.append(value)
            per_carrier_values[carrier_id].append(value)
        if len(pair_values) == ROWS_PER_CONVERSATION:
            complete_values.append(math.fsum(pair_values) / ROWS_PER_CONVERSATION)
            complete_ids.append(effect.candidate_id)
        elif pair_values:
            partial += 1
        else:
            unavailable += 1

    available = sum(len(values) for values in per_carrier_values.values())
    n_complete = len(complete_values)
    mean = (math.fsum(complete_values) / n_complete
            if n_complete else None)
    sample_sd: float | None = None
    standard_error: float | None = None
    interval: ConfidenceInterval | None = None
    if n_complete >= 2:
        sample_sd = statistics.stdev(complete_values)
        standard_error = sample_sd / math.sqrt(n_complete)
        critical = float(stats.t.ppf(0.975, n_complete - 1))
        _require(math.isfinite(critical), "VP t critical value is nonfinite")
        half_width = critical * standard_error
        interval = ConfidenceInterval(
            lower=float(mean) - half_width,
            upper=float(mean) + half_width,
            confidence=T_CONFIDENCE,
        )
    counts = {carrier_id: len(per_carrier_values[carrier_id])
              for carrier_id in CARRIER_IDS}
    carrier_means: dict[str, float | None] = {}
    for carrier_id in CARRIER_IDS:
        values = per_carrier_values[carrier_id]
        carrier_means[carrier_id] = (
            math.fsum(values) / len(values) if values else None)
    return AvailableCaseMetricSummary(
        n_condition_rows_available=available,
        n_condition_rows_missing=FINAL_ROW_COUNT - available,
        n_complete_conversations=n_complete,
        n_partial_conversations=partial,
        n_unavailable_conversations=unavailable,
        complete_conversation_mean=mean,
        complete_conversation_sample_sd=sample_sd,
        complete_conversation_standard_error=standard_error,
        complete_conversation_t_ci_95=interval,
        per_carrier_counts=counts,
        per_carrier_means=carrier_means,
        complete_candidate_ids=tuple(complete_ids),
    )


def analyze_treatment_records(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_candidate_ids: Sequence[str],
) -> LocalN48Analysis:
    """Validate and analyze the frozen 8x6x2 local-v3 treatment records."""

    materialized = list(rows)
    _require(len(materialized) == FINAL_ROW_COUNT,
             "expected exactly 96 rows (48 conversations x two carriers)")
    _require(all(isinstance(row, Mapping) for row in materialized),
             "treatment row is not a mapping")
    expected_ids = list(expected_candidate_ids)
    _require(
        len(expected_ids) == FINAL_N
        and all(isinstance(candidate_id, str) and bool(candidate_id)
                for candidate_id in expected_ids)
        and len(set(expected_ids)) == FINAL_N,
        "expected treatment-release candidate IDs are not 48 unique strings",
    )

    normalized: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(materialized):
        candidate_id = row.get("candidate_id")
        stratum = row.get("stratum")
        carrier_id = row.get("carrier_id")
        _require(isinstance(candidate_id, str) and bool(candidate_id),
                 f"row {index} candidate_id is empty")
        _require(isinstance(stratum, str) and stratum in STRATA,
                 f"row {index} has unknown stratum {stratum!r}")
        _require(isinstance(carrier_id, str) and carrier_id in CARRIER_IDS,
                 f"row {index} has unknown carrier_id {carrier_id!r}")
        pair = (candidate_id, carrier_id)
        _require(pair not in seen_pairs,
                 f"duplicate candidate/carrier row {candidate_id}:{carrier_id}")
        seen_pairs.add(pair)
        values: dict[str, float] = {}
        for field in _REQUIRED_NUMERIC_FIELDS:
            _require(field in row, f"row {index} is missing {field}")
            values[field] = _finite_number(
                row[field], f"{candidate_id}:{carrier_id}:{field}")
        vp_present = [field in row for field in _VP_NUMERIC_FIELDS]
        _require(vp_present[0] == vp_present[1],
                 f"row {index} has an incomplete C/W placebo score pair")
        if all(vp_present):
            for field in _VP_NUMERIC_FIELDS:
                values[field] = _finite_number(
                    row[field], f"{candidate_id}:{carrier_id}:{field}")
        normalized.append({
            "candidate_id": candidate_id,
            "stratum": stratum,
            "carrier_id": carrier_id,
            **values,
        })
    _require(len(seen_pairs) == FINAL_ROW_COUNT,
             "candidate/carrier pair count differs from 96")

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in normalized:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)
    _require(len(grouped) == FINAL_N,
             "expected exactly 48 unique candidate IDs")
    _require(set(grouped) == set(expected_ids),
             "observed candidate IDs differ from treatment release")
    stratum_counts = {stratum: 0 for stratum in STRATA}
    ordered_groups: list[tuple[str, str, list[dict[str, object]]]] = []
    for candidate_id, candidate_rows in grouped.items():
        _require(len(candidate_rows) == ROWS_PER_CONVERSATION,
                 f"{candidate_id} does not have exactly two carrier rows")
        carrier_ids = {str(row["carrier_id"]) for row in candidate_rows}
        _require(carrier_ids == set(CARRIER_IDS),
                 f"{candidate_id} does not have exactly fixed_a/fixed_b")
        strata = {str(row["stratum"]) for row in candidate_rows}
        _require(len(strata) == 1,
                 f"{candidate_id} has inconsistent strata")
        stratum = next(iter(strata))
        stratum_counts[stratum] += 1
        ordered_groups.append((candidate_id, stratum, candidate_rows))
    for stratum in STRATA:
        _require(stratum_counts[stratum] == PER_STRATUM,
                 f"{stratum} has {stratum_counts[stratum]} conversations, expected six")
    stratum_order = {stratum: index for index, stratum in enumerate(STRATA)}
    ordered_groups.sort(key=lambda item: (stratum_order[item[1]], item[0]))

    effects: list[ConversationEffect] = []
    raw_by_carrier: dict[str, list[float]] = {
        carrier_id: [] for carrier_id in CARRIER_IDS}
    row_lookup: dict[tuple[str, str], dict[str, object]] = {}
    for candidate_id, stratum, candidate_rows in ordered_groups:
        by_carrier = {str(row["carrier_id"]): row for row in candidate_rows}
        xs: list[float] = []
        x_pairs: list[tuple[str, float]] = []
        for carrier_id in CARRIER_IDS:
            row = by_carrier[carrier_id]
            x = float(row["l_c_e_c"]) - float(row["l_c_b"])
            _require(math.isfinite(x),
                     f"{candidate_id}:{carrier_id}:x is nonfinite")
            xs.append(x)
            x_pairs.append((carrier_id, x))
            raw_by_carrier[carrier_id].append(x)
            row_lookup[(candidate_id, carrier_id)] = row
        raw_x = math.fsum(xs) / ROWS_PER_CONVERSATION
        _require(math.isfinite(raw_x), f"{candidate_id}:raw X is nonfinite")
        effects.append(ConversationEffect(
            candidate_id=candidate_id,
            stratum=stratum,
            x_by_carrier=tuple(x_pairs),
            raw_x=raw_x,
            clipped_z=min(CLIP_UPPER, max(CLIP_LOWER, raw_x)),
        ))

    raw_values = [effect.raw_x for effect in effects]
    clipped_values = [effect.clipped_z for effect in effects]
    primary_raw = _metric_summary(raw_values, raw_by_carrier)
    clipped_mean = math.fsum(clipped_values) / FINAL_N
    radius = _hoeffding_radius(
        n=FINAL_N, alpha=HOEFFDING_ALPHA,
        lower=CLIP_LOWER, upper=CLIP_UPPER)
    raw_lcb, raw_ucb = clipped_mean - radius, clipped_mean + radius
    primary_hoeffding = HoeffdingBounds(
        alpha=HOEFFDING_ALPHA,
        radius=radius,
        mean_z=clipped_mean,
        lcb=max(CLIP_LOWER, raw_lcb),
        ucb=min(CLIP_UPPER, raw_ucb),
        unclamped_lcb=raw_lcb,
        unclamped_ucb=raw_ucb,
    )
    independent = recompute_hoeffding_stdlib(clipped_values)
    for field in ("radius", "mean_z", "lcb", "ucb",
                  "unclamped_lcb", "unclamped_ucb"):
        _require(math.isclose(
            getattr(primary_hoeffding, field), getattr(independent, field),
            rel_tol=0.0, abs_tol=1e-15),
            f"independent Hoeffding recomputation differs at {field}")

    successes = sum(value > CLIP_UPPER for value in raw_values)
    tail = TailPrevalenceBound(
        threshold=CLIP_UPPER,
        successes=successes,
        n_conversations=FINAL_N,
        alpha=TAIL_ALPHA,
        confidence=1.0 - TAIL_ALPHA,
        clopper_pearson_ucb=clopper_pearson_upper(
            successes, FINAL_N, alpha=TAIL_ALPHA),
    )

    metric_functions: list[tuple[str, Callable[[Mapping[str, object]], float]]] = [
        ("absolute_l_c_b", lambda row: float(row["l_c_b"])),
        ("absolute_l_w_b", lambda row: float(row["l_w_b"])),
        ("absolute_l_c_e_c", lambda row: float(row["l_c_e_c"])),
        ("absolute_l_w_e_c", lambda row: float(row["l_w_e_c"])),
        ("absolute_l_c_e_w", lambda row: float(row["l_c_e_w"])),
        ("absolute_l_w_e_w", lambda row: float(row["l_w_e_w"])),
        ("absolute_l_c_a_c", lambda row: float(row["l_c_a_c"])),
        ("absolute_l_w_a_c", lambda row: float(row["l_w_a_c"])),
        ("specificity_correct_minus_wrong",
         lambda row: float(row["l_c_e_c"]) - float(row["l_c_e_w"])),
        ("wrong_history_movement_from_fresh",
         lambda row: float(row["l_c_e_w"]) - float(row["l_c_b"])),
        ("correct_target_damage",
         lambda row: float(row["l_c_a_c"]) - float(row["l_c_b"])),
        ("margin_damage",
         lambda row: (
             float(row["l_c_a_c"]) - float(row["l_w_a_c"])
             - float(row["l_c_b"]) + float(row["l_w_b"]))),
        ("fresh_margin",
         lambda row: float(row["l_c_b"]) - float(row["l_w_b"])),
        ("correct_graft_margin",
         lambda row: float(row["l_c_e_c"]) - float(row["l_w_e_c"])),
        ("wrong_graft_margin",
         lambda row: float(row["l_c_e_w"]) - float(row["l_w_e_w"])),
        ("graft_margin_specificity",
         lambda row: (
             float(row["l_c_e_c"]) - float(row["l_w_e_c"])
             - float(row["l_c_e_w"]) + float(row["l_w_e_w"]))),
        ("margin_recovery_from_fresh",
         lambda row: (
             float(row["l_c_e_c"]) - float(row["l_w_e_c"])
             - float(row["l_c_b"]) + float(row["l_w_b"]))),
    ]

    companions: dict[str, MetricSummary] = {}
    for name, function in metric_functions:
        conversation_values: list[float] = []
        by_carrier_values: dict[str, list[float]] = {
            carrier_id: [] for carrier_id in CARRIER_IDS}
        for effect in effects:
            pair_values: list[float] = []
            for carrier_id in CARRIER_IDS:
                value = function(row_lookup[(effect.candidate_id, carrier_id)])
                _require(math.isfinite(value),
                         f"{effect.candidate_id}:{carrier_id}:{name} is nonfinite")
                pair_values.append(value)
                by_carrier_values[carrier_id].append(value)
            collapsed = math.fsum(pair_values) / ROWS_PER_CONVERSATION
            _require(math.isfinite(collapsed),
                     f"{effect.candidate_id}:{name} is nonfinite")
            conversation_values.append(collapsed)
        companions[name] = _metric_summary(
            conversation_values, by_carrier_values)

    vp_functions: tuple[
        tuple[str, Callable[[Mapping[str, object]], float]], ...
    ] = (
        ("placebo_movement_from_fresh",
         lambda row: float(row["l_c_vp"]) - float(row["l_c_b"])),
        ("correct_minus_placebo",
         lambda row: float(row["l_c_e_c"]) - float(row["l_c_vp"])),
        ("placebo_margin",
         lambda row: float(row["l_c_vp"]) - float(row["l_w_vp"])),
        ("correct_minus_placebo_margin",
         lambda row: (
             float(row["l_c_e_c"]) - float(row["l_w_e_c"])
             - float(row["l_c_vp"]) + float(row["l_w_vp"]))),
    )
    placebo = {
        name: _available_case_summary(effects, row_lookup, function)
        for name, function in vp_functions
    }

    return LocalN48Analysis(
        design_id=DESIGN_ID,
        n_input_rows=FINAL_ROW_COUNT,
        n_conversations=FINAL_N,
        independent_unit="conversation/candidate_id",
        carrier_ids=CARRIER_IDS,
        conversation_effects=tuple(effects),
        primary_raw=primary_raw,
        primary_hoeffding=primary_hoeffding,
        raw_over_half=tail,
        companion_metrics=companions,
        applied_placebo_metrics=placebo,
    )


__all__ = [
    "AvailableCaseMetricSummary",
    "CARRIER_IDS",
    "CLIP_LOWER",
    "CLIP_UPPER",
    "DESIGN_ID",
    "FINAL_N",
    "HOEFFDING_ALPHA",
    "LocalN48Analysis",
    "LocalN48StatsError",
    "STRATA",
    "analyze_treatment_records",
    "clopper_pearson_upper",
    "recompute_hoeffding_stdlib",
]
