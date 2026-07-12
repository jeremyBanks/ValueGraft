from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from scripts.accounting.build_end_to_end_summary import (
    EndToEndSummaryError,
    _price_codex_row,
    build_end_to_end_summary,
    write_summary,
)


CLAUDE_FIELDS = (
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


def _claude_usage(
    *,
    uncached: int = 0,
    write_5m: int = 0,
    write_1h: int = 0,
    read: int = 0,
    output: int = 0,
) -> dict[str, int]:
    creation = write_5m + write_1h
    context = uncached + creation + read
    return {
        "uncached_input_tokens": uncached,
        "cache_write_5m_tokens": write_5m,
        "cache_write_1h_tokens": write_1h,
        "cache_write_unclassified_tokens": 0,
        "cache_creation_input_tokens": creation,
        "cache_read_input_tokens": read,
        "context_input_tokens": context,
        "output_tokens": output,
        "total_tokens": context + output,
    }


def _add_claude(rows: list[dict[str, int]]) -> dict[str, int]:
    return {field: sum(row[field] for row in rows) for field in CLAUDE_FIELDS}


def _tokens(
    *,
    uncached: int,
    cached: int = 0,
    output: int = 0,
    reasoning: int = 0,
) -> dict[str, int]:
    return {
        "input_tokens": uncached + cached,
        "cached_input_tokens": cached,
        "uncached_input_tokens": uncached,
        "output_tokens": output,
        "reasoning_output_tokens": reasoning,
        "non_reasoning_output_tokens": output - reasoning,
        "total_tokens": uncached + cached + output,
    }


def _add_tokens(rows: list[dict[str, int]]) -> dict[str, int]:
    return {field: sum(row[field] for row in rows) for field in rows[0]}


def _claude_document() -> dict:
    opus = _claude_usage(
        uncached=1_000_000,
        write_5m=1_000_000,
        write_1h=1_000_000,
        read=1_000_000,
        output=1_000_000,
    )
    synthetic = _claude_usage()
    return {
        "schema": "claude_project_usage_snapshot_v1",
        "coverage": {"status": "complete_frozen_prefix"},
        "requests": {
            "request_count": 3,
            "usage": _add_claude([opus, synthetic]),
            "by_model": [
                {
                    "model": "claude-opus-4-8",
                    "request_count": 1,
                    "usage": opus,
                },
                {
                    "model": "<synthetic>",
                    "request_count": 2,
                    "usage": synthetic,
                },
            ],
        },
        "money": {
            "cash_spend_usd": None,
            "cash_spend_classification": "subscription cash unknown",
        },
    }


def _codex_document(rows: list[dict] | None = None) -> dict:
    if rows is None:
        rows = [
            {
                "model": "gpt-5.6-sol",
                "effort": "ultra",
                "input_context_class": "standard",
                "request_event_count": 1,
                "tokens": _tokens(
                    uncached=900_000,
                    cached=100_000,
                    output=100_000,
                    reasoning=60_000,
                ),
            }
        ]
    totals = _add_tokens([row["tokens"] for row in rows])
    no_graph = dict(totals)
    no_graph["input_tokens"] += 100
    no_graph["uncached_input_tokens"] += 100
    no_graph["total_tokens"] += 100
    gate = {
        "status": "PASS",
        "classification": "reconstructed_agreement_coverage_pass",
        "reconstructed_total_authorized": True,
        "scope": "fixture scope",
        "blocked_reasons": [],
    }
    return {
        "schema": "codex_project_usage_snapshot_v1",
        "measurement_class": "reconstructed",
        "closeout_gate": gate,
        "reconstructions": {
            "primary": {
                "totals": totals,
                "no_graph_collapse_upper_bound": no_graph,
            }
        },
        "owned_request_aggregate": {
            "schema": "codex_owned_request_usage_aggregate_v1",
            "request_event_count": sum(row["request_event_count"] for row in rows),
            "tokens": totals,
            "input_context_class_rule": {
                "input_token_field": "last_token_usage.input_tokens",
                "standard": "input_tokens <= 272000",
                "long": "input_tokens > 272000",
            },
            "by_model_effort_input_context_class": rows,
        },
        "money": {
            "cash_spend_usd": None,
            "cash_spend_classification": "subscription cash unknown",
        },
    }


def _runpod_reconciliation() -> dict:
    return {
        "schema": "runpod_project_reconciliation_v1",
        "provider_account_pod_consumption": {
            "measurement_class": "exact_source_record",
            "amount_usd_credits": "10",
            "billed_time_ms": 3_600_000,
            "billed_hours_rounded_12dp": "1",
            "pod_count": 2,
        },
        "project_attribution": {
            "independently_attributed_lower_bound": {
                "measurement_class": "lower_bound",
                "amount_usd_credits": "8",
                "billed_time_ms": 2_880_000,
                "billed_hours_rounded_12dp": "0.8",
                "pod_count": 1,
            },
            "unattributed_provider_window_group": {
                "measurement_class": "attribution_unknown",
                "amount_usd_credits": "2",
                "billed_time_ms": 720_000,
                "pod_count": 1,
            },
        },
        "cash_deposit_or_payment_history": {
            "status": "unknown",
            "amount_usd": None,
            "reason": "no payment ledger",
        },
    }


def _runpod_nonpod() -> dict:
    return {
        "schema": "runpod_nonpod_billing_snapshot_v1",
        "combined_amount_usd": "0",
        "resources": {
            "serverless_endpoints": {"amount_usd": "0", "row_count": 0},
            "network_volumes": {"amount_usd": "0", "row_count": 0},
        },
    }


def _openrouter() -> dict:
    return {
        "schema": "openrouter_key_usage_snapshot_v2",
        "usage_counters": {
            "usage": "0.1",
            "usage_monthly": "0.08",
            "usage_weekly": "0.07",
            "usage_daily": "0.01",
        },
        "account_context_nonadditive": {
            "total_usage": "2",
            "total_credits": "3",
        },
        "classification": {
            "measurement": "exact_source_record_for_current_key",
            "cash_paid_usd": None,
            "project_attribution": "not implied",
        },
    }


def _local_residual() -> dict:
    return {
        "schema": "experimental_compute_and_residual_provider_audit_v1",
        "scope": {"research_window_start_utc": "2026-07-04T20:05:15Z"},
        "local_mac_experimental_inference": {
            "directly_timed_experimental_run_wall_time_lower_bound": {
                "measurement_class": "lower_bound",
                "artifact_count": 2,
                "wall_seconds": "3600",
                "wall_hours": "1",
                "interpretation": "retained-record floor",
            },
            "additional_proven_work_without_additive_timing": {
                "measurement_class": "lower_bound_for_artifact_presence_time_unknown",
                "explicit_mlx_model_artifacts_missing_wall_seconds": {
                    "artifact_count": 3
                },
                "previously_audited_local_alpha_sweep_artifacts_missing_model_and_wall_fields": {
                    "artifact_count": 4
                },
                "warning": "do not add artifact counts to timed run counts",
            },
        },
        "residual_providers_and_access": {
            "chatgpt_subscription": {
                "invoice_or_project_allocation_usd": None,
                "measurement_class": "unknown",
                "reason": "no invoice allocation",
            },
            "claude_subscription": {
                "invoice_or_project_allocation_usd": None,
                "measurement_class": "unknown",
                "reason": "no invoice allocation",
            },
            "gemini": {
                "cash_usd": None,
                "tokens": None,
                "measurement_class": "unknown",
                "reason": "no telemetry",
            },
            "hugging_face": {
                "hub_subscription_cash_usd": None,
                "download_bytes": None,
                "measurement_class": "unknown",
                "reason": "no invoice",
            },
        },
    }


def _conversation_manifest() -> dict:
    return {
        "version": 1,
        "description": "fixture",
        "notes": [
            {
                "note": "notes/conversation-one.md",
                "summarizer": {
                    "provider": "codex",
                    "model": "gpt-5.6-luna",
                    "reasoning": "medium",
                },
            },
            {
                "note": "notes/conversation-two.md",
                "summarizer": {"provider": "claude", "model": "sonnet"},
            },
        ],
    }


def _daily_manifest() -> dict:
    return {
        "version": 1,
        "description": "fixture",
        "days": {
            "20260711": {
                "note": "notes/20260711.md",
                "summarizer": {
                    "provider": "codex",
                    "model": "gpt-5.6-luna",
                    "reasoning": "medium",
                },
            },
            "20260712": {
                "note": "notes/20260712.md",
                "summarizer": {
                    "provider": "codex",
                    "model": "gpt-5.6-luna",
                    "reasoning": "medium",
                },
            },
        },
    }


def _overall_manifest() -> dict:
    return {
        "version": 2,
        "description": "fixture",
        "rollups": {
            "notes/202607.md": {
                "note": "notes/202607.md",
                "summarizer": {
                    "provider": "codex",
                    "model": "gpt-5.6-luna",
                    "reasoning": "medium",
                },
            },
            "notes/README.md": {
                "note": "notes/README.md",
                "summarizer": {"provider": "promote"},
            },
        },
    }


def _write_inputs(tmp_path: Path, documents: dict[str, dict] | None = None) -> dict[str, Path]:
    docs = {
        "claude_snapshot": _claude_document(),
        "codex_snapshot": _codex_document(),
        "runpod_reconciliation": _runpod_reconciliation(),
        "runpod_nonpod_snapshot": _runpod_nonpod(),
        "openrouter_snapshot": _openrouter(),
        "local_residual_audit": _local_residual(),
        "conversation_manifest": _conversation_manifest(),
        "daily_manifest": _daily_manifest(),
        "overall_manifest": _overall_manifest(),
    }
    if documents:
        docs.update(documents)
    paths: dict[str, Path] = {}
    for label, document in docs.items():
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        paths[label] = path
    return paths


def _build(tmp_path: Path, documents: dict[str, dict] | None = None) -> tuple[dict, dict[str, Path]]:
    paths = _write_inputs(tmp_path, documents)
    summary = build_end_to_end_summary(**paths, display_root=tmp_path)
    return summary, paths


def test_arithmetic_source_commitments_and_nonadditivity_contract(tmp_path: Path) -> None:
    summary, paths = _build(tmp_path)

    # Opus: 5 + 6.25 + 10 + 0.5 + 25 dollars for five one-MTok components.
    claude = summary["claude_cli"]["standard_speed_token_only_api_counterfactual"]
    assert claude["total_usd"] == "46.75"
    assert claude["per_model"][0]["total_usd"] == "46.75"

    # Sol: 0.1M cached * .5 + 0.9M ordinary * 5 + 0.1M output * 30.
    codex = summary["codex_cli"][
        "conditional_on_primary_reconstruction_standard_api_equivalent"
    ]
    assert codex["lower_usd"] == "7.55"
    assert codex["upper_usd"] == "8.675"
    assert summary["codex_cli"]["pending_tail_gate_verbatim"] == _codex_document()[
        "closeout_gate"
    ]
    assert summary["codex_cli"]["token_sensitivity"][
        "no_graph_collapse_upper_bound_verbatim"
    ]["total_tokens"] == 1_100_100

    assert summary["runpod"]["provider_account_window"][
        "covered_categories_combined_consumed_credits_usd"
    ] == "10"
    assert summary["runpod"]["project_attribution"][
        "independently_attributed_lower_bound"
    ]["consumed_credits_usd"] == "8"
    assert summary["runpod"]["project_attribution"][
        "unattributed_provider_window_gap"
    ]["consumed_credits_usd"] == "2"

    contract = summary["accounting_contract"]
    assert contract["total_project_cash_usd"] is None
    assert contract["total_project_cash_measurement_class"] == "unknown"
    assert contract["grand_total_computed"] is False
    assert len(contract["nonadditivity"]) >= 7

    for label, path in paths.items():
        raw = path.read_bytes()
        assert summary["source_commitments"][label] == {
            "path": path.name,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }


def test_long_context_boundary_uses_strictly_greater_than_272k() -> None:
    standard = {
        "model": "gpt-5.6-sol",
        "effort": "xhigh",
        "input_context_class": "standard",
        "request_event_count": 1,
        "tokens": _tokens(uncached=272_000, output=10, reasoning=4),
    }
    long = {
        "model": "gpt-5.6-sol",
        "effort": "xhigh",
        "input_context_class": "long",
        "request_event_count": 1,
        "tokens": _tokens(uncached=272_001, output=10, reasoning=4),
    }
    standard_priced, _, _ = _price_codex_row(standard, 0)
    long_priced, _, _ = _price_codex_row(long, 1)
    assert standard_priced["multipliers"] == {
        "input": "1",
        "output": "1",
        "long_context_rule": "> 272000 input tokens",
    }
    assert long_priced["multipliers"] == {
        "input": "2",
        "output": "1.5",
        "long_context_rule": "> 272000 input tokens",
    }


def test_changed_long_context_boundary_fails_closed(tmp_path: Path) -> None:
    document = _codex_document()
    document["owned_request_aggregate"]["input_context_class_rule"]["standard"] = (
        "input_tokens <= 271999"
    )
    paths = _write_inputs(tmp_path, {"codex_snapshot": document})
    with pytest.raises(EndToEndSummaryError, match="272,000-token rule"):
        build_end_to_end_summary(**paths, display_root=tmp_path)


def test_cache_write_range_applies_only_to_gpt56_and_reasoning_is_not_doubled() -> None:
    sol = {
        "model": "gpt-5.6-sol",
        "effort": "ultra",
        "input_context_class": "standard",
        "request_event_count": 1,
        "tokens": _tokens(uncached=1_000_000, output=1_000_000, reasoning=800_000),
    }
    old = {
        **sol,
        "model": "gpt-5.5",
    }
    sol_row, sol_lower, sol_upper = _price_codex_row(sol, 0)
    old_row, old_lower, old_upper = _price_codex_row(old, 1)
    assert sol_lower == Decimal("35")
    assert sol_upper == Decimal("36.25")
    assert old_lower == old_upper == Decimal("35")
    assert sol_row["components"]["output_including_reasoning_once"] == {
        "amount_usd": "30",
        "rate_usd_per_million_tokens": "30",
        "reasoning_output_tokens_are_output_subset_not_extra_charge": True,
    }
    assert old_row["components"]["uncached_input_upper_all_cache_writes"][
        "cache_write_premium_multiplier"
    ] == "1"


@pytest.mark.parametrize("ledger", ["claude", "codex"])
def test_unknown_nonzero_model_fails_closed(tmp_path: Path, ledger: str) -> None:
    overrides: dict[str, dict] = {}
    if ledger == "claude":
        document = _claude_document()
        document["requests"]["by_model"][0]["model"] = "claude-unknown"
        overrides["claude_snapshot"] = document
    else:
        row = {
            "model": "gpt-unknown",
            "effort": "high",
            "input_context_class": "standard",
            "request_event_count": 1,
            "tokens": _tokens(uncached=1),
        }
        overrides["codex_snapshot"] = _codex_document([row])
    paths = _write_inputs(tmp_path, overrides)
    with pytest.raises(EndToEndSummaryError, match="unknown"):
        build_end_to_end_summary(**paths, display_root=tmp_path)


def test_luna_counts_are_retained_output_floor_not_request_telemetry(tmp_path: Path) -> None:
    summary, _ = _build(tmp_path)
    audit = summary["luna_summary_archive"]
    assert audit["retained_output_manifest_records"] == {
        "conversation": 1,
        "daily": 2,
        "overall": 1,
        "total": 4,
        "measurement_class": "exact_count_in_supplied_manifests",
        "interpretation": (
            "This is a lower bound on Luna-produced summary outputs that survived "
            "into the supplied manifests. Overwritten outputs, failed attempts, "
            "retries, and ephemeral reruns leave no record here."
        ),
    }
    assert audit["telemetry"]["request_count"] is None
    assert audit["telemetry"]["tokens"] is None
    assert audit["telemetry"]["standard_api_equivalent_usd"] is None
    assert audit["telemetry"]["cash_paid_usd"] is None
    assert "not provider requests" in audit["telemetry"]["warning"]


def test_output_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    summary_one, paths = _build(tmp_path)
    summary_two = build_end_to_end_summary(**paths, display_root=tmp_path)
    assert summary_one == summary_two

    output = tmp_path / "summary.json"
    write_summary(output, summary_one)
    with pytest.raises(EndToEndSummaryError, match="refusing to overwrite"):
        write_summary(output, summary_two)


def test_source_mutation_changes_source_and_summary_commitments(tmp_path: Path) -> None:
    summary_one, paths = _build(tmp_path)
    conversation = json.loads(paths["conversation_manifest"].read_text())
    conversation["description"] = "changed but schema-valid"
    paths["conversation_manifest"].write_text(
        json.dumps(conversation, indent=2) + "\n", encoding="utf-8"
    )
    summary_two = build_end_to_end_summary(**paths, display_root=tmp_path)
    assert summary_one["source_commitments"]["conversation_manifest"] != summary_two[
        "source_commitments"
    ]["conversation_manifest"]
    assert summary_one["summary_sha256"] != summary_two["summary_sha256"]
