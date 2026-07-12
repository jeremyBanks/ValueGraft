#!/usr/bin/env python3
"""Build the deterministic, nonadditive end-to-end accounting summary.

This is deliberately a pure combiner.  It performs no provider, transcript, or
credential access.  Every evidence input is explicit, schema-checked, and bound
by its repository-relative path, byte length, and SHA-256 digest.  Monetary
arithmetic is performed with :class:`decimal.Decimal`; the emitted strings are
exact decimal spellings rather than binary floating-point approximations.

The output separates four things that must not be collapsed into one number:

* metered provider consumption;
* subscription workload telemetry;
* token-only public-API counterfactuals; and
* unknown cash paid / subscription allocation.

In particular, this program never emits a grand cash total.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
MILLION = Decimal("1000000")
LONG_CONTEXT_THRESHOLD = 272_000
PRICING_RETRIEVED_ON = "2026-07-12"


# Standard-speed, text-token-only public API prices in USD per million tokens.
# The strings are part of the frozen evidence contract and become Decimals only
# at the arithmetic boundary.
CLAUDE_RATE_TABLE: dict[str, dict[str, str]] = {
    "claude-fable-5": {
        "uncached_input": "10",
        "cache_write_5m": "12.5",
        "cache_write_1h": "20",
        "cache_read": "1",
        "output": "50",
    },
    "claude-opus-4-8": {
        "uncached_input": "5",
        "cache_write_5m": "6.25",
        "cache_write_1h": "10",
        "cache_read": "0.5",
        "output": "25",
    },
    # Introductory Sonnet 5 pricing in effect through 2026-08-31.  The frozen
    # snapshots and this retrieval date are within that interval.
    "claude-sonnet-5": {
        "uncached_input": "2",
        "cache_write_5m": "2.5",
        "cache_write_1h": "4",
        "cache_read": "0.2",
        "output": "10",
    },
}


OPENAI_RATE_TABLE: dict[str, dict[str, str | bool]] = {
    "gpt-5.4": {
        "input": "2.5",
        "cached_input": "0.25",
        "output": "15",
        "cache_write_premium_applies": False,
    },
    "gpt-5.5": {
        "input": "5",
        "cached_input": "0.5",
        "output": "30",
        "cache_write_premium_applies": False,
    },
    "gpt-5.6-sol": {
        "input": "5",
        "cached_input": "0.5",
        "output": "30",
        "cache_write_premium_applies": True,
    },
    "gpt-5.6-terra": {
        "input": "2.5",
        "cached_input": "0.25",
        "output": "15",
        "cache_write_premium_applies": True,
    },
    "gpt-5.6-luna": {
        "input": "1",
        "cached_input": "0.1",
        "output": "6",
        "cache_write_premium_applies": True,
    },
}


PRICING_SOURCES = {
    "retrieved_on": PRICING_RETRIEVED_ON,
    "anthropic": {
        "url": "https://platform.claude.com/docs/en/about-claude/pricing",
        "basis": (
            "standard-speed text-token rates; Sonnet 5 introductory rates in "
            "effect through 2026-08-31"
        ),
    },
    "openai": {
        "model_catalog_url": "https://developers.openai.com/api/docs/models",
        "model_urls": {
            "gpt-5.4": "https://developers.openai.com/api/docs/models/gpt-5.4",
            "gpt-5.5": "https://developers.openai.com/api/docs/models/gpt-5.5",
            "gpt-5.6-sol": (
                "https://developers.openai.com/api/docs/models/gpt-5.6-sol"
            ),
            "gpt-5.6-terra": (
                "https://developers.openai.com/api/docs/models/gpt-5.6-terra"
            ),
            "gpt-5.6-luna": (
                "https://developers.openai.com/api/docs/models/gpt-5.6-luna"
            ),
        },
        "basis": (
            "standard text-token rates; requests with more than 272,000 input "
            "tokens use 2x input and 1.5x output rates; GPT-5.6 cache writes "
            "use 1.25x uncached-input rate"
        ),
    },
}


CLAUDE_USAGE_FIELDS = (
    "uncached_input_tokens",
    "cache_write_5m_tokens",
    "cache_write_1h_tokens",
    "cache_write_unclassified_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "context_input_tokens",
    "output_tokens",
    "total_tokens",
)

CODEX_TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "uncached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "non_reasoning_output_tokens",
    "total_tokens",
)


class EndToEndSummaryError(RuntimeError):
    """Raised when the supplied evidence cannot satisfy the summary contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EndToEndSummaryError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EndToEndSummaryError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise EndToEndSummaryError(f"{label} must be an array")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EndToEndSummaryError(f"{label} must be nonempty text")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EndToEndSummaryError(f"{label} must be a nonnegative integer")
    return value


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or value is None or isinstance(value, float):
        raise EndToEndSummaryError(f"{label} must be an exact decimal value")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise EndToEndSummaryError(f"{label} must be an exact decimal value") from exc
    if not parsed.is_finite() or parsed < 0:
        raise EndToEndSummaryError(f"{label} must be finite and nonnegative")
    return parsed


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _money(tokens: int, rate_per_million: str, multiplier: Decimal = Decimal(1)) -> Decimal:
    return Decimal(tokens) * Decimal(rate_per_million) * multiplier / MILLION


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _source_binding(path: Path, raw: bytes, *, display_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(display_root.resolve()).as_posix()
    except ValueError as exc:
        raise EndToEndSummaryError(
            f"accounting input is outside the display root: {resolved}"
        ) from exc
    return {
        "path": display,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _load_json(path: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EndToEndSummaryError(f"cannot read {label}: {path}") from exc
    try:
        parsed = json.loads(raw, parse_float=Decimal, parse_int=int)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EndToEndSummaryError(f"{label} is not valid JSON: {path}") from exc
    return _mapping(parsed, label), raw


def _token_vector(value: Any, label: str) -> dict[str, int]:
    row = _mapping(value, label)
    tokens = {field: _integer(row.get(field), f"{label}.{field}") for field in CODEX_TOKEN_FIELDS}
    _require(
        tokens["cached_input_tokens"] + tokens["uncached_input_tokens"]
        == tokens["input_tokens"],
        f"{label} cached + uncached input does not equal input",
    )
    _require(
        tokens["reasoning_output_tokens"] + tokens["non_reasoning_output_tokens"]
        == tokens["output_tokens"],
        f"{label} reasoning + non-reasoning output does not equal output",
    )
    _require(
        tokens["input_tokens"] + tokens["output_tokens"] == tokens["total_tokens"],
        f"{label} input + output does not equal total",
    )
    return tokens


def _sum_vectors(vectors: Iterable[Mapping[str, int]]) -> dict[str, int]:
    result = {field: 0 for field in CODEX_TOKEN_FIELDS}
    for vector in vectors:
        for field in CODEX_TOKEN_FIELDS:
            result[field] += vector[field]
    return result


def _claude_usage(value: Any, label: str) -> dict[str, int]:
    row = _mapping(value, label)
    usage = {
        field: _integer(row.get(field), f"{label}.{field}")
        for field in CLAUDE_USAGE_FIELDS
    }
    _require(
        usage["cache_write_5m_tokens"]
        + usage["cache_write_1h_tokens"]
        + usage["cache_write_unclassified_tokens"]
        == usage["cache_creation_input_tokens"],
        f"{label} classified cache writes do not equal cache creation",
    )
    _require(
        usage["uncached_input_tokens"]
        + usage["cache_creation_input_tokens"]
        + usage["cache_read_input_tokens"]
        == usage["context_input_tokens"],
        f"{label} context input components do not sum",
    )
    _require(
        usage["context_input_tokens"] + usage["output_tokens"]
        == usage["total_tokens"],
        f"{label} context input + output does not equal total",
    )
    return usage


def _sum_claude_usage(rows: Iterable[Mapping[str, int]]) -> dict[str, int]:
    result = {field: 0 for field in CLAUDE_USAGE_FIELDS}
    for row in rows:
        for field in CLAUDE_USAGE_FIELDS:
            result[field] += row[field]
    return result


def _build_claude(document: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        document.get("schema") == "claude_project_usage_snapshot_v1",
        "Claude snapshot schema is not claude_project_usage_snapshot_v1",
    )
    requests = _mapping(document.get("requests"), "Claude requests")
    by_model = _list(requests.get("by_model"), "Claude requests.by_model")
    _require(by_model, "Claude requests.by_model is empty")
    seen: set[str] = set()
    priced_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, int]] = []
    request_count = 0
    total_cost = Decimal(0)
    synthetic: dict[str, Any] | None = None

    component_map = (
        ("uncached_input", "uncached_input_tokens"),
        ("cache_write_5m", "cache_write_5m_tokens"),
        ("cache_write_1h", "cache_write_1h_tokens"),
        ("cache_read", "cache_read_input_tokens"),
        ("output", "output_tokens"),
    )
    for index, raw_row in enumerate(by_model):
        row = _mapping(raw_row, f"Claude by_model[{index}]")
        model = _text(row.get("model"), f"Claude by_model[{index}].model")
        _require(model not in seen, f"duplicate Claude model row: {model}")
        seen.add(model)
        count = _integer(
            row.get("request_count"), f"Claude by_model[{index}].request_count"
        )
        usage = _claude_usage(row.get("usage"), f"Claude {model} usage")
        request_count += count
        usage_rows.append(usage)
        if model == "<synthetic>":
            _require(
                all(value == 0 for value in usage.values()),
                "synthetic Claude row is allowed only when every token field is zero",
            )
            synthetic = {
                "model": model,
                "request_count": count,
                "tokens": usage,
                "pricing_classification": "zero_token_synthetic_row_not_priced",
            }
            continue
        _require(model in CLAUDE_RATE_TABLE, f"unknown nonzero Claude model: {model}")
        _require(
            usage["cache_write_unclassified_tokens"] == 0,
            f"Claude {model} has unclassified cache-write tokens",
        )
        rates = CLAUDE_RATE_TABLE[model]
        components: dict[str, Any] = {}
        model_total = Decimal(0)
        for component, token_field in component_map:
            amount = _money(usage[token_field], rates[component])
            components[component] = {
                "tokens": usage[token_field],
                "rate_usd_per_million_tokens": rates[component],
                "amount_usd": _decimal_text(amount),
            }
            model_total += amount
        total_cost += model_total
        priced_rows.append(
            {
                "model": model,
                "request_count": count,
                "tokens": usage,
                "components": components,
                "total_usd": _decimal_text(model_total),
            }
        )

    aggregate_usage = _claude_usage(requests.get("usage"), "Claude aggregate usage")
    _require(
        _sum_claude_usage(usage_rows) == aggregate_usage,
        "Claude by-model usage does not sum to aggregate usage",
    )
    _require(
        request_count == _integer(requests.get("request_count"), "Claude request_count"),
        "Claude by-model request counts do not sum to aggregate request_count",
    )
    money = _mapping(document.get("money"), "Claude money")
    _require(money.get("cash_spend_usd") is None, "Claude cash spend must remain unknown")
    coverage = _mapping(document.get("coverage"), "Claude coverage")
    _text(coverage.get("status"), "Claude coverage.status")
    return {
        "workload": {
            "measurement_class": "exact_source_record_for_frozen_cli_prefix",
            "request_count": request_count,
            "tokens": aggregate_usage,
            "coverage_status": coverage["status"],
        },
        "standard_speed_token_only_api_counterfactual": {
            "measurement_class": "counterfactual_public_api_list_price_not_cash",
            "pricing_retrieved_on": PRICING_RETRIEVED_ON,
            "per_model": priced_rows,
            "zero_token_synthetic": synthetic,
            "total_usd": _decimal_text(total_cost),
            "exclusions": [
                "subscription pricing and subsidy",
                "tax, credits, commitments, and volume discounts",
                "non-token tool charges",
            ],
        },
        "cash": {
            "amount_usd": None,
            "measurement_class": "unknown",
            "source_classification": money.get("cash_spend_classification"),
        },
    }


def _price_codex_row(row: Mapping[str, Any], index: int) -> tuple[dict[str, Any], Decimal, Decimal]:
    label = f"Codex owned aggregate row[{index}]"
    model = _text(row.get("model"), f"{label}.model")
    _require(model in OPENAI_RATE_TABLE, f"unknown Codex model: {model}")
    context_class = _text(row.get("input_context_class"), f"{label}.input_context_class")
    _require(context_class in {"standard", "long"}, f"{label} has unknown context class")
    count = _integer(row.get("request_event_count"), f"{label}.request_event_count")
    tokens = _token_vector(row.get("tokens"), f"{label}.tokens")
    rates = OPENAI_RATE_TABLE[model]
    input_multiplier = Decimal(2) if context_class == "long" else Decimal(1)
    output_multiplier = Decimal("1.5") if context_class == "long" else Decimal(1)
    cached = _money(tokens["cached_input_tokens"], str(rates["cached_input"]), input_multiplier)
    ordinary_uncached = _money(tokens["uncached_input_tokens"], str(rates["input"]), input_multiplier)
    premium_applies = bool(rates["cache_write_premium_applies"])
    upper_uncached = ordinary_uncached * (Decimal("1.25") if premium_applies else Decimal(1))
    output = _money(tokens["output_tokens"], str(rates["output"]), output_multiplier)
    lower = cached + ordinary_uncached + output
    upper = cached + upper_uncached + output
    rendered = {
        "model": model,
        "effort": _text(row.get("effort"), f"{label}.effort"),
        "input_context_class": context_class,
        "request_event_count": count,
        "tokens": tokens,
        "multipliers": {
            "input": _decimal_text(input_multiplier),
            "output": _decimal_text(output_multiplier),
            "long_context_rule": f"> {LONG_CONTEXT_THRESHOLD} input tokens",
        },
        "components": {
            "cached_input": {
                "amount_usd": _decimal_text(cached),
                "rate_usd_per_million_tokens": str(rates["cached_input"]),
            },
            "uncached_input_lower_ordinary": {
                "amount_usd": _decimal_text(ordinary_uncached),
                "rate_usd_per_million_tokens": str(rates["input"]),
            },
            "uncached_input_upper_all_cache_writes": {
                "amount_usd": _decimal_text(upper_uncached),
                "base_rate_usd_per_million_tokens": str(rates["input"]),
                "cache_write_premium_multiplier": "1.25" if premium_applies else "1",
            },
            "output_including_reasoning_once": {
                "amount_usd": _decimal_text(output),
                "rate_usd_per_million_tokens": str(rates["output"]),
                "reasoning_output_tokens_are_output_subset_not_extra_charge": True,
            },
        },
        "api_equivalent_lower_usd": _decimal_text(lower),
        "api_equivalent_upper_usd": _decimal_text(upper),
    }
    return rendered, lower, upper


def _build_codex(document: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        document.get("schema") == "codex_project_usage_snapshot_v1",
        "Codex snapshot schema is not codex_project_usage_snapshot_v1",
    )
    _require(document.get("measurement_class") == "reconstructed", "Codex measurement class changed")
    gate = _mapping(document.get("closeout_gate"), "Codex closeout_gate")
    _require(gate.get("status") in {"PASS", "BLOCKED"}, "Codex closeout gate status is invalid")
    _require(
        isinstance(gate.get("reconstructed_total_authorized"), bool),
        "Codex closeout gate lacks authorization boolean",
    )
    reconstructions = _mapping(document.get("reconstructions"), "Codex reconstructions")
    primary = _mapping(reconstructions.get("primary"), "Codex primary reconstruction")
    primary_tokens = _token_vector(primary.get("totals"), "Codex primary totals")
    no_graph = _token_vector(
        primary.get("no_graph_collapse_upper_bound"),
        "Codex no-graph-collapse upper bound",
    )
    _require(
        no_graph["total_tokens"] >= primary_tokens["total_tokens"],
        "Codex no-graph-collapse upper bound is below primary total",
    )
    aggregate = _mapping(document.get("owned_request_aggregate"), "Codex owned_request_aggregate")
    _require(
        aggregate.get("schema") == "codex_owned_request_usage_aggregate_v1",
        "Codex owned request aggregate schema changed",
    )
    aggregate_tokens = _token_vector(aggregate.get("tokens"), "Codex owned aggregate tokens")
    _require(aggregate_tokens == primary_tokens, "Codex owned aggregate tokens differ from primary")
    context_rule = _mapping(
        aggregate.get("input_context_class_rule"),
        "Codex owned aggregate input_context_class_rule",
    )
    _require(
        context_rule.get("input_token_field") == "last_token_usage.input_tokens"
        and context_rule.get("standard") == f"input_tokens <= {LONG_CONTEXT_THRESHOLD}"
        and context_rule.get("long") == f"input_tokens > {LONG_CONTEXT_THRESHOLD}",
        "Codex input context boundary differs from the frozen 272,000-token rule",
    )
    rows = _list(
        aggregate.get("by_model_effort_input_context_class"),
        "Codex owned aggregate rows",
    )
    priced_rows: list[dict[str, Any]] = []
    row_vectors: list[dict[str, int]] = []
    row_count = 0
    lower_total = Decimal(0)
    upper_total = Decimal(0)
    model_totals: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"request_event_count": 0, "lower": Decimal(0), "upper": Decimal(0)}
    )
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"Codex owned aggregate row[{index}]")
        rendered, lower, upper = _price_codex_row(row, index)
        priced_rows.append(rendered)
        row_vectors.append(rendered["tokens"])
        count = rendered["request_event_count"]
        row_count += count
        lower_total += lower
        upper_total += upper
        model_bucket = model_totals[rendered["model"]]
        model_bucket["request_event_count"] += count
        model_bucket["lower"] += lower
        model_bucket["upper"] += upper
    _require(
        _sum_vectors(row_vectors) == aggregate_tokens,
        "Codex priced rows do not sum to owned aggregate tokens",
    )
    _require(
        row_count == _integer(aggregate.get("request_event_count"), "Codex aggregate request_event_count"),
        "Codex priced row request counts do not sum",
    )
    money = _mapping(document.get("money"), "Codex money")
    _require(money.get("cash_spend_usd") is None, "Codex cash spend must remain unknown")
    per_model = [
        {
            "model": model,
            "request_event_count": values["request_event_count"],
            "api_equivalent_lower_usd": _decimal_text(values["lower"]),
            "api_equivalent_upper_usd": _decimal_text(values["upper"]),
        }
        for model, values in sorted(model_totals.items())
    ]
    return {
        "workload": {
            "measurement_class": "reconstructed",
            "request_event_count": row_count,
            "tokens": primary_tokens,
        },
        "pending_tail_gate_verbatim": deepcopy(gate),
        "token_sensitivity": {
            "primary_reconstruction_verbatim": deepcopy(primary.get("totals")),
            "no_graph_collapse_upper_bound_verbatim": deepcopy(
                primary.get("no_graph_collapse_upper_bound")
            ),
            "pricing_not_extended_to_no_graph_upper_bound": (
                "The no-graph token vector has no corresponding owned-request rows, "
                "so no request-class-aware dollar counterfactual is invented."
            ),
        },
        "conditional_on_primary_reconstruction_standard_api_equivalent": {
            "measurement_class": "counterfactual_conditional_on_reconstructed_primary_not_cash",
            "pricing_retrieved_on": PRICING_RETRIEVED_ON,
            "lower_usd": _decimal_text(lower_total),
            "upper_usd": _decimal_text(upper_total),
            "lower_assumption": "every uncached token uses the ordinary input rate",
            "upper_assumption": (
                "every GPT-5.6 uncached token is a cache write at 1.25x; "
                "pre-5.6 models receive no write premium"
            ),
            "long_context_rule": (
                f"input_context_class=long means >{LONG_CONTEXT_THRESHOLD} input "
                "tokens and applies 2x input / 1.5x output"
            ),
            "per_model": per_model,
            "per_model_effort_context": priced_rows,
            "reasoning_output_accounting": (
                "reasoning_output_tokens is a subset of output_tokens and is not "
                "charged a second time"
            ),
            "exclusions": [
                "subscription pricing and subsidy",
                "tool-call charges",
                "regional-processing uplift",
                "batch, flex, priority, and service-tier adjustments",
            ],
        },
        "cash": {
            "amount_usd": None,
            "measurement_class": "unknown",
            "source_classification": money.get("cash_spend_classification"),
        },
    }


def _build_runpod(
    reconciliation: Mapping[str, Any], nonpod: Mapping[str, Any]
) -> dict[str, Any]:
    _require(
        reconciliation.get("schema") == "runpod_project_reconciliation_v1",
        "RunPod reconciliation schema changed",
    )
    provider = _mapping(
        reconciliation.get("provider_account_pod_consumption"),
        "RunPod provider account pod consumption",
    )
    _require(provider.get("measurement_class") == "exact_source_record", "RunPod provider total is not exact")
    project = _mapping(reconciliation.get("project_attribution"), "RunPod project attribution")
    floor = _mapping(
        project.get("independently_attributed_lower_bound"), "RunPod project floor"
    )
    gap = _mapping(
        project.get("unattributed_provider_window_group"), "RunPod attribution gap"
    )
    provider_amount = _decimal(provider.get("amount_usd_credits"), "RunPod provider amount")
    floor_amount = _decimal(floor.get("amount_usd_credits"), "RunPod project floor amount")
    gap_amount = _decimal(gap.get("amount_usd_credits"), "RunPod gap amount")
    _require(provider_amount == floor_amount + gap_amount, "RunPod floor + gap does not equal provider amount")
    provider_ms = _integer(provider.get("billed_time_ms"), "RunPod provider billed_time_ms")
    floor_ms = _integer(floor.get("billed_time_ms"), "RunPod floor billed_time_ms")
    gap_ms = _integer(gap.get("billed_time_ms"), "RunPod gap billed_time_ms")
    _require(provider_ms == floor_ms + gap_ms, "RunPod floor + gap billed time does not reconcile")
    provider_count = _integer(provider.get("pod_count"), "RunPod provider pod_count")
    floor_count = _integer(floor.get("pod_count"), "RunPod floor pod_count")
    gap_count = _integer(gap.get("pod_count"), "RunPod gap pod_count")
    _require(provider_count == floor_count + gap_count, "RunPod floor + gap pod count does not reconcile")

    _require(
        nonpod.get("schema") == "runpod_nonpod_billing_snapshot_v1",
        "RunPod non-Pod snapshot schema changed",
    )
    resources = _mapping(nonpod.get("resources"), "RunPod non-Pod resources")
    serverless = _mapping(resources.get("serverless_endpoints"), "RunPod serverless")
    volumes = _mapping(resources.get("network_volumes"), "RunPod network volumes")
    serverless_amount = _decimal(serverless.get("amount_usd"), "RunPod serverless amount")
    volume_amount = _decimal(volumes.get("amount_usd"), "RunPod network volume amount")
    nonpod_amount = _decimal(nonpod.get("combined_amount_usd"), "RunPod non-Pod combined amount")
    _require(nonpod_amount == serverless_amount + volume_amount, "RunPod non-Pod components do not sum")
    cash = _mapping(
        reconciliation.get("cash_deposit_or_payment_history"), "RunPod cash history"
    )
    _require(cash.get("status") == "unknown" and cash.get("amount_usd") is None, "RunPod cash classification must remain unknown")
    return {
        "provider_account_window": {
            "pods": {
                "measurement_class": "exact_source_record",
                "consumed_credits_usd": _decimal_text(provider_amount),
                "billed_time_ms": provider_ms,
                "billed_hours_rounded_12dp": provider.get("billed_hours_rounded_12dp"),
                "pod_count": provider_count,
                "project_attribution_not_implied": True,
            },
            "serverless_endpoints": {
                "measurement_class": "exact_source_record",
                "amount_usd": _decimal_text(serverless_amount),
                "row_count": _integer(serverless.get("row_count"), "RunPod serverless row_count"),
            },
            "network_volumes": {
                "measurement_class": "exact_source_record",
                "amount_usd": _decimal_text(volume_amount),
                "row_count": _integer(volumes.get("row_count"), "RunPod volume row_count"),
            },
            "covered_categories_combined_consumed_credits_usd": _decimal_text(
                provider_amount + nonpod_amount
            ),
            "covered_categories_combination": "Pods + serverless endpoints + network volumes",
        },
        "project_attribution": {
            "independently_attributed_lower_bound": {
                "measurement_class": "lower_bound",
                "consumed_credits_usd": _decimal_text(floor_amount),
                "billed_time_ms": floor_ms,
                "billed_hours_rounded_12dp": floor.get("billed_hours_rounded_12dp"),
                "pod_count": floor_count,
            },
            "unattributed_provider_window_gap": {
                "measurement_class": "unknown",
                "consumed_credits_usd": _decimal_text(gap_amount),
                "billed_time_ms": gap_ms,
                "pod_count": gap_count,
            },
        },
        "cash": {
            "amount_usd": None,
            "measurement_class": "unknown",
            "reason": cash.get("reason"),
        },
    }


def _build_openrouter(document: Mapping[str, Any]) -> dict[str, Any]:
    _require(document.get("schema") == "openrouter_key_usage_snapshot_v2", "OpenRouter snapshot schema changed")
    counters = _mapping(document.get("usage_counters"), "OpenRouter usage counters")
    usage = _decimal(counters.get("usage"), "OpenRouter key lifetime usage")
    monthly = _decimal(counters.get("usage_monthly"), "OpenRouter monthly usage")
    weekly = _decimal(counters.get("usage_weekly"), "OpenRouter weekly usage")
    daily = _decimal(counters.get("usage_daily"), "OpenRouter daily usage")
    account = _mapping(document.get("account_context_nonadditive"), "OpenRouter account context")
    account_usage = _decimal(account.get("total_usage"), "OpenRouter account lifetime usage")
    account_credits = _decimal(account.get("total_credits"), "OpenRouter account total credits")
    _require(usage <= account_usage, "OpenRouter key usage exceeds account lifetime usage")
    classification = _mapping(document.get("classification"), "OpenRouter classification")
    _require(
        classification.get("measurement") == "exact_source_record_for_current_key",
        "OpenRouter current-key measurement classification changed",
    )
    _require(classification.get("cash_paid_usd") is None, "OpenRouter cash paid must remain unknown")
    return {
        "exact_source_records": {
            "current_key": {
                "lifetime_credits_used": _decimal_text(usage),
                "monthly_credits_used": _decimal_text(monthly),
                "weekly_credits_used": _decimal_text(weekly),
                "daily_credits_used": _decimal_text(daily),
                "measurement_class": "exact_source_record_for_current_key",
                "project_attribution_not_implied": True,
            },
            "authenticated_account_lifetime_context_nonadditive": {
                "total_usage_credits": _decimal_text(account_usage),
                "total_credits": _decimal_text(account_credits),
                "measurement_class": "exact_source_record",
                "project_attribution_not_implied": True,
            },
        },
        "project_attribution_envelopes_not_estimates": {
            "within_supplied_current_key": {
                "lower_credits": "0",
                "upper_credits": _decimal_text(usage),
            },
            "within_authenticated_account_lifetime": {
                "lower_credits": "0",
                "upper_credits": _decimal_text(account_usage),
            },
            "warning": (
                "These are set-inclusion envelopes, not an estimate of project use; "
                "they do not establish that all project traffic used this key or account."
            ),
        },
        "request_count": None,
        "tokens": None,
        "project_attributable_credits": None,
        "cash": {
            "amount_usd": None,
            "measurement_class": "unknown",
            "source_project_attribution": classification.get("project_attribution"),
        },
    }


def _build_local(document: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        document.get("schema") == "experimental_compute_and_residual_provider_audit_v1",
        "local/residual audit schema changed",
    )
    local = _mapping(document.get("local_mac_experimental_inference"), "local Mac audit")
    timed = _mapping(
        local.get("directly_timed_experimental_run_wall_time_lower_bound"),
        "local Mac timed lower bound",
    )
    _require(timed.get("measurement_class") == "lower_bound", "local Mac timing is not a lower bound")
    wall_seconds = _decimal(timed.get("wall_seconds"), "local Mac wall seconds")
    wall_hours = _decimal(timed.get("wall_hours"), "local Mac wall hours")
    _require(
        wall_seconds / Decimal(3600) == wall_hours,
        "local Mac wall seconds and hours do not agree exactly",
    )
    additional = _mapping(
        local.get("additional_proven_work_without_additive_timing"),
        "local Mac additional untimed work",
    )
    residual = _mapping(document.get("residual_providers_and_access"), "residual providers")
    chatgpt = _mapping(residual.get("chatgpt_subscription"), "ChatGPT subscription")
    claude = _mapping(residual.get("claude_subscription"), "Claude subscription")
    gemini = _mapping(residual.get("gemini"), "Gemini residual")
    hugging_face = _mapping(residual.get("hugging_face"), "Hugging Face residual")
    for label, row, money_field in (
        ("ChatGPT", chatgpt, "invoice_or_project_allocation_usd"),
        ("Claude", claude, "invoice_or_project_allocation_usd"),
        ("Gemini", gemini, "cash_usd"),
        ("Hugging Face", hugging_face, "hub_subscription_cash_usd"),
    ):
        _require(row.get("measurement_class") == "unknown", f"{label} residual is not unknown")
        _require(row.get(money_field) is None, f"{label} residual cash must be null")
    scope = _mapping(document.get("scope"), "local/residual scope")
    return {
        "research_window_start_utc": scope.get("research_window_start_utc"),
        "directly_timed_experimental_run_wall_time_floor": {
            "measurement_class": "lower_bound",
            "artifact_count": _integer(timed.get("artifact_count"), "local timed artifact_count"),
            "wall_seconds": _decimal_text(wall_seconds),
            "wall_hours": _decimal_text(wall_hours),
            "interpretation": timed.get("interpretation"),
        },
        "additional_proven_work_without_additive_timing": {
            "measurement_class": additional.get("measurement_class"),
            "explicit_mlx_model_artifact_count": _integer(
                _mapping(
                    additional.get("explicit_mlx_model_artifacts_missing_wall_seconds"),
                    "untimed MLX artifacts",
                ).get("artifact_count"),
                "untimed MLX artifact_count",
            ),
            "previously_audited_alpha_artifact_count": _integer(
                _mapping(
                    additional.get(
                        "previously_audited_local_alpha_sweep_artifacts_missing_model_and_wall_fields"
                    ),
                    "untimed alpha artifacts",
                ).get("artifact_count"),
                "untimed alpha artifact_count",
            ),
            "warning": additional.get("warning"),
        },
        "electricity_and_hardware_amortization": {
            "amount_usd": None,
            "measurement_class": "unknown",
        },
        "residual_cash_unknowns": {
            "chatgpt_subscription": deepcopy(chatgpt),
            "claude_subscription": deepcopy(claude),
            "gemini": deepcopy(gemini),
            "hugging_face": deepcopy(hugging_face),
        },
    }


def _summarizer_is_luna(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return value.get("provider") == "codex" and value.get("model") == "gpt-5.6-luna"


def _build_luna_audit(
    conversation: Mapping[str, Any],
    daily: Mapping[str, Any],
    overall: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        conversation.get("version") == 1
        and isinstance(conversation.get("notes"), list),
        "conversation summary manifest schema/version changed",
    )
    _require(
        daily.get("version") == 1 and isinstance(daily.get("days"), Mapping),
        "daily summary manifest schema/version changed",
    )
    _require(
        overall.get("version") == 2 and isinstance(overall.get("rollups"), Mapping),
        "overall summary manifest schema/version changed",
    )
    conversation_rows = _list(conversation["notes"], "conversation manifest notes")
    daily_rows = list(_mapping(daily["days"], "daily manifest days").values())
    overall_rows = list(_mapping(overall["rollups"], "overall manifest rollups").values())
    for label, rows in (
        ("conversation", conversation_rows),
        ("daily", daily_rows),
        ("overall", overall_rows),
    ):
        for index, row in enumerate(rows):
            mapped = _mapping(row, f"{label} manifest row[{index}]")
            _text(mapped.get("note"), f"{label} manifest row[{index}].note")
            summarizer = mapped.get("summarizer")
            _require(isinstance(summarizer, Mapping), f"{label} manifest row[{index}] lacks summarizer")
    counts = {
        "conversation": sum(_summarizer_is_luna(row.get("summarizer")) for row in conversation_rows),
        "daily": sum(_summarizer_is_luna(row.get("summarizer")) for row in daily_rows),
        "overall": sum(_summarizer_is_luna(row.get("summarizer")) for row in overall_rows),
    }
    total = sum(counts.values())
    return {
        "model": "gpt-5.6-luna",
        "retained_output_manifest_records": {
            **counts,
            "total": total,
            "measurement_class": "exact_count_in_supplied_manifests",
            "interpretation": (
                "This is a lower bound on Luna-produced summary outputs that "
                "survived into the supplied manifests. Overwritten outputs, failed "
                "attempts, retries, and ephemeral reruns leave no record here."
            ),
        },
        "telemetry": {
            "request_count": None,
            "tokens": None,
            "standard_api_equivalent_usd": None,
            "cash_paid_usd": None,
            "measurement_class": "unknown",
            "warning": (
                "Retained output records are not provider requests and are not "
                "converted into request, token, API-equivalent, or cash totals."
            ),
        },
    }


def build_end_to_end_summary(
    *,
    claude_snapshot: Path,
    codex_snapshot: Path,
    runpod_reconciliation: Path,
    runpod_nonpod_snapshot: Path,
    openrouter_snapshot: Path,
    local_residual_audit: Path,
    conversation_manifest: Path,
    daily_manifest: Path,
    overall_manifest: Path,
    display_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Load, validate, bind, and combine the nine explicit evidence inputs."""

    paths = {
        "claude_snapshot": Path(claude_snapshot),
        "codex_snapshot": Path(codex_snapshot),
        "runpod_reconciliation": Path(runpod_reconciliation),
        "runpod_nonpod_snapshot": Path(runpod_nonpod_snapshot),
        "openrouter_snapshot": Path(openrouter_snapshot),
        "local_residual_audit": Path(local_residual_audit),
        "conversation_manifest": Path(conversation_manifest),
        "daily_manifest": Path(daily_manifest),
        "overall_manifest": Path(overall_manifest),
    }
    documents: dict[str, Mapping[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        document, raw = _load_json(path, label)
        documents[label] = document
        bindings[label] = _source_binding(path, raw, display_root=display_root)

    summary: dict[str, Any] = {
        "schema": "end_to_end_accounting_summary_v1",
        "accounting_contract": {
            "total_project_cash_usd": None,
            "total_project_cash_measurement_class": "unknown",
            "grand_total_computed": False,
            "nonadditivity": [
                "RunPod Pod, serverless, and network-volume provider categories are additive; the two non-Pod categories are exact zero in the supplied window.",
                "RunPod project floor and attribution gap partition the provider Pod total; neither is added again.",
                "Claude and Codex public-API equivalents are counterfactuals for subscription workload, not observed cash and not additive to subscription invoices.",
                "Codex primary and no-graph totals are sensitivity alternatives, never additive.",
                "OpenRouter current-key usage is contained within account-lifetime usage; the two are not additive.",
                "Luna retained summary records can overlap the Codex workload ledger and have no standalone request/token/cash telemetry.",
                "Mac wall time, agent episode time, and RunPod rental time overlap and are different units; no compute-hour total is formed.",
            ],
        },
        "pricing": {
            "sources": deepcopy(PRICING_SOURCES),
            "claude_usd_per_million_tokens": deepcopy(CLAUDE_RATE_TABLE),
            "openai_usd_per_million_tokens": deepcopy(OPENAI_RATE_TABLE),
        },
        "claude_cli": _build_claude(documents["claude_snapshot"]),
        "codex_cli": _build_codex(documents["codex_snapshot"]),
        "runpod": _build_runpod(
            documents["runpod_reconciliation"],
            documents["runpod_nonpod_snapshot"],
        ),
        "openrouter": _build_openrouter(documents["openrouter_snapshot"]),
        "local_and_residual": _build_local(documents["local_residual_audit"]),
        "luna_summary_archive": _build_luna_audit(
            documents["conversation_manifest"],
            documents["daily_manifest"],
            documents["overall_manifest"],
        ),
        "source_commitments": bindings,
    }
    summary["summary_sha256"] = hashlib.sha256(
        _canonical_json(summary).encode("utf-8")
    ).hexdigest()
    return summary


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise EndToEndSummaryError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic end-to-end accounting summary from frozen inputs."
    )
    parser.add_argument("--claude-snapshot", type=Path, required=True)
    parser.add_argument("--codex-snapshot", type=Path, required=True)
    parser.add_argument("--runpod-reconciliation", type=Path, required=True)
    parser.add_argument("--runpod-nonpod-snapshot", type=Path, required=True)
    parser.add_argument("--openrouter-snapshot", type=Path, required=True)
    parser.add_argument("--local-residual-audit", type=Path, required=True)
    parser.add_argument("--conversation-manifest", type=Path, required=True)
    parser.add_argument("--daily-manifest", type=Path, required=True)
    parser.add_argument("--overall-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_end_to_end_summary(
        claude_snapshot=args.claude_snapshot,
        codex_snapshot=args.codex_snapshot,
        runpod_reconciliation=args.runpod_reconciliation,
        runpod_nonpod_snapshot=args.runpod_nonpod_snapshot,
        openrouter_snapshot=args.openrouter_snapshot,
        local_residual_audit=args.local_residual_audit,
        conversation_manifest=args.conversation_manifest,
        daily_manifest=args.daily_manifest,
        overall_manifest=args.overall_manifest,
    )
    write_summary(args.output, summary)
    print(
        "END-TO-END ACCOUNTING SUMMARY "
        f"cash=UNKNOWN sha256={summary['summary_sha256']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
