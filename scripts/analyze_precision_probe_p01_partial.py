#!/usr/bin/env python3
"""Post-run analysis of the terminal partial precision-probe p01 execution.

This analyzer is intentionally separate from the frozen complete-run analyzer.
It accepts exactly the observed, scientifically useful partial shape:

* passing NF4 and bfloat16 technical packages on one matched runtime;
* complete e01 outcomes for NF4 repeats 1/2 and bfloat16 repeat 1;
* an absent or explicitly incomplete bfloat16 repeat 2; and
* a non-PASS terminal runner manifest/completion marker.

Every complete raw package is reconstructed from its lossless package before
the complete 34-arm grid is checked.  Runner-produced compact JSON and render
ledgers are reproduced byte-for-byte from the reconstructed raw.  The output
is post-run, partial, and descriptive: it computes no p-value, confidence
interval, equivalence result, formal-v12 decision, or population claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "scripts", ROOT / "src"):
    value = str(import_root)
    if value not in sys.path:
        sys.path.insert(0, value)

# These imports deliberately reuse only independently tested parsers, grid
# validators, and pure render/compact constructors.  The terminal-partial
# contract below is new and fail-closed; the complete-run analyzer is unchanged.
import analyze_precision_probe_p01 as strict  # noqa: E402
import run_precision_probe_p01 as runner  # noqa: E402


PROTOCOL_ID = strict.PROTOCOL_ID
ANALYSIS_SCHEMA = "precision_probe_p01_postrun_partial_descriptive_analysis_v1"
ANALYSIS_STATUS = "POST_RUN_PARTIAL_DESCRIPTIVE"
EXPECTED_COMPLETE_KEYS = {
    ("nf4", "e01", 1),
    ("nf4", "e01", 2),
    ("bf16", "e01", 1),
}
EXPECTED_INCOMPLETE_KEY = ("bf16", "e01", 2)
TERMINAL_NONPASS = {"ERROR", "PARTIAL"}
REQUIRED_SOURCE_BINDINGS = {
    "preregistration", "identity_fixture", "technical_fixture",
    "case_e01", "case_e02", "case_e03", "runner",
    "precision_subject_loader", "artifact_packager",
    "coherent_canary_case", "coherent_canary_runtime",
    "coherent_canary_technical", "coherent_canary_tokens",
    "coherent_canary_controls", "coherent_canary_schema",
    "coherent_state_tokens",
}
GENERATION_CELLS = ("FF", "FC", "FW", "CC", "WW")
MATCHED_RUNTIME_FIELDS = (
    "model_id", "requested_revision", "resolved_snapshot", "architecture",
    "attention_backend", "kv_dtype", "eos_ids", "geometry",
    "repository_commit", "dependency_versions", "gpu_uuid",
)
MATCHED_SUBJECT_BINDING_FIELDS = (
    "repository", "dependencies", "host", "checkpoint", "g0",
    "tokenizer", "model", "eos", "placement",
)


AnalysisError = strict.AnalysisError
require = strict.require


def _single(paths: Sequence[Path], label: str) -> Path:
    require(len(paths) == 1, f"p01 partial run requires exactly one {label}")
    return paths[0]


def _resolve_run_path(run_dir: Path, declared: Any, label: str) -> Path:
    require(isinstance(declared, str) and declared, f"{label} path is absent")
    normalized = Path(declared).as_posix()
    matches: list[Path] = []
    for path in run_dir.rglob(Path(declared).name):
        relative = path.relative_to(run_dir).as_posix()
        if (normalized == relative or normalized.endswith("/" + relative)
                or relative.endswith("/" + normalized)):
            matches.append(path)
    require(len(matches) == 1, f"{label} path cannot be resolved uniquely")
    return matches[0]


def _verify_file_binding(run_dir: Path, row: Any, label: str) -> Path:
    require(isinstance(row, Mapping) and
            set(row) == {"path", "sha256", "size_bytes"},
            f"{label} binding shape differs")
    path = _resolve_run_path(run_dir, row.get("path"), label)
    require(path.is_file() and row.get("sha256") == strict.file_sha256(path) and
            row.get("size_bytes") == path.stat().st_size,
            f"{label} binding bytes differ")
    return path


def _verify_package_binding(
    run_dir: Path,
    binding: Any,
    package_row: Mapping[str, Any],
    label: str,
) -> Path:
    expected_fields = {
        "path", "manifest_path", "manifest_sha256", "manifest_size_bytes",
        "chunk_count", "original_sha256", "original_size_bytes",
        "compressed_sha256", "compressed_size_bytes", "verification_status",
    }
    require(isinstance(binding, Mapping) and set(binding) == expected_fields,
            f"{label} package binding shape differs")
    package_dir = _resolve_run_path(run_dir, binding.get("path"), label)
    require(package_dir.is_dir() and package_dir == package_row["package_dir"],
            f"{label} package path differs")
    manifest_path = _resolve_run_path(
        run_dir, binding.get("manifest_path"), f"{label} manifest")
    require(manifest_path == package_dir / "manifest.json",
            f"{label} package manifest path differs")
    manifest = strict.load_object(manifest_path)
    require(binding.get("verification_status") == "VERIFIED" and
            binding.get("manifest_sha256") == strict.file_sha256(manifest_path) and
            binding.get("manifest_size_bytes") == manifest_path.stat().st_size and
            binding.get("chunk_count") == manifest.get("chunk_count") and
            binding.get("original_sha256") ==
            manifest.get("original", {}).get("sha256") and
            binding.get("original_size_bytes") ==
            manifest.get("original", {}).get("size_bytes") and
            binding.get("compressed_sha256") ==
            manifest.get("compression", {}).get("compressed_sha256") and
            binding.get("compressed_size_bytes") ==
            manifest.get("compression", {}).get("compressed_size_bytes") and
            package_row["package_manifest_sha256"] ==
            binding.get("manifest_sha256") and
            package_row["raw_artifact_sha256"] ==
            binding.get("original_sha256"),
            f"{label} package receipt differs from reconstructed bytes")
    return package_dir


def _derived_package_binding(
    package_row: Mapping[str, Any], run_dir: Path
) -> dict[str, Any]:
    package_dir = package_row["package_dir"]
    manifest_path = package_dir / "manifest.json"
    manifest = strict.load_object(manifest_path)
    return {
        "path": package_dir.relative_to(run_dir).as_posix(),
        "manifest_path": manifest_path.relative_to(run_dir).as_posix(),
        "manifest_sha256": strict.file_sha256(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "chunk_count": manifest["chunk_count"],
        "original_sha256": manifest["original"]["sha256"],
        "original_size_bytes": manifest["original"]["size_bytes"],
        "compressed_sha256": manifest["compression"]["compressed_sha256"],
        "compressed_size_bytes": manifest["compression"][
            "compressed_size_bytes"],
        "verification_status": "VERIFIED",
    }


def verify_outcome_derived_content(
    raw: Mapping[str, Any],
    raw_package_binding: Mapping[str, Any],
    compact_path: Path,
    render_path: Path,
) -> None:
    """Recompute runner-derived artifacts from independently reconstructed raw."""
    document = dict(raw)
    document["raw_package_binding"] = dict(raw_package_binding)
    actual_compact = strict.load_object(compact_path)
    expected_compact = runner.compact_outcome(document)
    require(actual_compact == expected_compact,
            "complete outcome compact content differs from reconstructed raw")
    try:
        actual_render = render_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise AnalysisError(f"cannot read outcome render ledger: {exc}") from exc
    require(actual_render == runner.render_outcome_ledger(document),
            "complete outcome render content differs from reconstructed raw")


def _verify_technical_render(
    raw: Mapping[str, Any], raw_package_binding: Mapping[str, Any], path: Path,
) -> None:
    document = dict(raw)
    document["raw_package_binding"] = dict(raw_package_binding)
    require(path.read_text(encoding="utf-8") ==
            runner.render_technical_ledger(document),
            "technical render content differs from reconstructed raw")


def _validate_source_bindings(
    common: Any,
) -> dict[str, Any]:
    require(isinstance(common, Mapping) and
            set(common) == REQUIRED_SOURCE_BINDINGS,
            "p01 source-binding name set differs")
    rows: dict[str, Any] = {}
    for name in sorted(common):
        row = common[name]
        require(isinstance(row, Mapping) and
                set(row) == {"path", "sha256", "size_bytes"} and
                isinstance(row.get("path"), str),
                f"source binding differs: {name}")
        declared = Path(str(row["path"]))
        require(not declared.is_absolute() and ".." not in declared.parts,
                f"source binding path is unsafe: {name}")
        path = (ROOT / declared).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and
                row.get("sha256") == strict.file_sha256(path) and
                row.get("size_bytes") == path.stat().st_size,
                f"source binding no longer matches bytes: {name}")
        rows[name] = dict(row)
    return {
        "status": "PASS",
        "binding_names": sorted(rows),
        "canonical_sha256": strict.sha256_bytes(strict.canonical_bytes(rows)),
        "bindings": rows,
    }


def _validate_raw_source_bindings(
    raw: Mapping[str, Any], common: Mapping[str, Any], label: str,
) -> None:
    observed = raw.get("bindings")
    require(isinstance(observed, Mapping), f"{label} source bindings are absent")
    expected = dict(common)
    if raw.get("schema") == strict.OUTCOME_RAW_SCHEMA:
        case_id = raw.get("case_id")
        require(case_id in {"e01", "e02", "e03"},
                f"{label} case binding selector differs")
        expected["case"] = common[f"case_{case_id}"]
    require(dict(observed) == expected,
            f"{label} source bindings differ from run manifest")


def validate_partial_layout(
    complete_keys: set[tuple[str, str, int]],
    incomplete_keys: Sequence[tuple[str, str, int]],
) -> None:
    require(complete_keys == EXPECTED_COMPLETE_KEYS,
            "p01 partial complete-outcome set differs: "
            f"{sorted(complete_keys)}")
    require(len(incomplete_keys) <= 1 and
            all(key == EXPECTED_INCOMPLETE_KEY for key in incomplete_keys),
            "p01 partial incomplete-outcome set differs")
    require(EXPECTED_INCOMPLETE_KEY not in complete_keys,
            "bfloat16 e01 repeat 2 unexpectedly completed")


def _verify_completion_inventory(
    run_dir: Path, completion_path: Path, completion: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = completion.get("artifact_inventory")
    require(isinstance(inventory, list) and inventory,
            "partial completion inventory is empty")
    actual = {
        path.relative_to(run_dir).as_posix(): path
        for path in run_dir.rglob("*")
        if path.is_file() and path != completion_path
    }
    mapped: dict[str, Mapping[str, Any]] = {}
    for row in inventory:
        require(isinstance(row, Mapping) and
                set(row) == {"path", "sha256", "size_bytes"} and
                isinstance(row.get("path"), str),
                "partial completion inventory row differs")
        declared = str(row["path"])
        matches = [relative for relative in actual
                   if declared == relative or declared.endswith("/" + relative)]
        require(len(matches) == 1 and matches[0] not in mapped,
                f"partial inventory path cannot be resolved: {declared}")
        mapped[matches[0]] = row
    require(set(mapped) == set(actual),
            "partial completion inventory is not exhaustive")
    for relative, path in actual.items():
        row = mapped[relative]
        require(row.get("sha256") == strict.file_sha256(path) and
                row.get("size_bytes") == path.stat().st_size,
                f"partial completion inventory bytes differ: {relative}")
    require(completion.get("recovery_raw_files") == [],
            "partial completion retained unreceipted recovery raws")
    return {
        "status": "PASS",
        "file_count": len(actual),
        "inventory_sha256": strict.sha256_bytes(strict.canonical_bytes(
            [{"path": relative,
              "sha256": mapped[relative]["sha256"],
              "size_bytes": mapped[relative]["size_bytes"]}
             for relative in sorted(mapped)]
        )),
    }


def _validate_extension_decision(
    run_dir: Path,
    manifest: Mapping[str, Any],
    complete_by_key: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    decision_path = _single(sorted(run_dir.glob(
        "precision-probe-p01-extension-decision_*.json")),
        "extension decision")
    _verify_file_binding(run_dir, manifest.get("extension_decision"),
                         "extension decision")
    decision = strict.load_object(decision_path)
    strict._require_formal_boundary(decision, "extension decision")
    inputs = decision.get("inputs")
    require(decision.get("schema") == strict.DECISION_SCHEMA and
            isinstance(inputs, Mapping),
            "partial extension decision schema/inputs differ")
    times = inputs.get("nf4_e01_outcome_wall_time_seconds")
    require(inputs.get("nf4_e01_completion_statuses") ==
            ["COMPLETE", "COMPLETE"] and
            isinstance(times, list) and len(times) == 2,
            "partial extension decision NF4 receipts differ")
    raw_times = [float(complete_by_key[("nf4", "e01", repeat)]["raw"][
        "outcome_wall_time_seconds"]) for repeat in (1, 2)]
    require(all(isinstance(value, (int, float)) and
                not isinstance(value, bool) and math.isfinite(float(value)) and
                math.isclose(float(value), raw, rel_tol=0, abs_tol=1e-12)
                for value, raw in zip(times, raw_times)),
            "partial extension timings differ from complete NF4 raws")
    elapsed = inputs.get("provider_elapsed_seconds")
    rate = inputs.get("hourly_cost_usd")
    cap = inputs.get("provider_wall_cap_seconds")
    require(all(isinstance(value, (int, float)) and
                not isinstance(value, bool) and math.isfinite(float(value)) and
                float(value) > 0 for value in (elapsed, rate, cap)),
            "partial extension provider inputs differ")
    slower = max(raw_times)
    deadline = min(7200.0, float(cap), 4.0 * 3600.0 / float(rate))
    forecast = float(elapsed) + 1.20 * (6.0 * slower + 900.0)
    allowed = forecast <= deadline
    require(math.isclose(float(inputs.get("slower_complete_nf4_e01_seconds")),
                         slower, rel_tol=0, abs_tol=1e-12) and
            math.isclose(float(inputs.get("four_dollar_rate_ceiling_seconds")),
                         4.0 * 3600.0 / float(rate), rel_tol=0,
                         abs_tol=1e-9) and
            math.isclose(float(inputs.get("effective_hard_deadline_seconds")),
                         deadline, rel_tol=0, abs_tol=1e-9) and
            math.isclose(float(inputs.get("forecast_seconds")), forecast,
                         rel_tol=0, abs_tol=1e-9) and
            decision.get("formula") ==
            "elapsed + 1.20 * (6*t + 900) <= min(7200, cap, 4*3600/rate)" and
            decision.get("score_or_generation_accessible_to_decision") is False and
            decision.get("outcome_information_surface") ==
            ["completion_status", "outcome_wall_time_seconds"],
            "partial extension decision derivation differs")
    require(allowed is False and
            decision.get("extension_allowed") is False and
            decision.get("decision") == "BASE_E01_ONLY" and
            decision.get("selected_case_repeats") == {"e01": 2} and
            manifest.get("selected_case_repeats") == {"e01": 2},
            "partial execution did not freeze the base-e01-only schedule")
    return {
        "status": "PASS",
        "path": decision_path.relative_to(run_dir).as_posix(),
        "sha256": strict.file_sha256(decision_path),
        "decision": "BASE_E01_ONLY",
        "forecast_seconds": forecast,
        "effective_hard_deadline_seconds": deadline,
    }


def _continuous(analysis: Mapping[str, Any]) -> dict[str, float]:
    value_n = analysis["N_regional_estimands"]["R2_boundary"]["value_only"]
    full_n = analysis["N_regional_estimands"]["R2_boundary"]["full_KV"]
    value_p = analysis["P_R2_estimands"]["value_only"]
    damage = analysis["phase_a_oracle_to_fresh_damage"]
    return {
        "D_value_N_R2_focal": float(value_n["D_focal"]),
        "D_value_N_R2_nonfocal": float(value_n["D_nonfocal"]),
        "SEL_value_N_R2_signed": float(value_n["SEL_signed_p01"]),
        "Hplus_value_N_R2": float(value_n["Hplus"]),
        "D_full_KV_N_R2_focal": float(full_n["D_focal"]),
        "D_value_P_R2_focal": float(value_p["D_focal"]),
        "N_minus_P_value_D_shift": float(
            analysis["N_minus_P_value_D_shift"]),
        "oracle_to_fresh_focal_margin_damage": float(
            damage["focal"]["margin"]),
        "oracle_to_fresh_focal_correct_target_logprob_damage": float(
            damage["focal"]["correct_target_logprob"]),
        "oracle_to_fresh_nonfocal_margin_damage": float(
            damage["nonfocal"]["margin"]),
        "oracle_to_fresh_nonfocal_correct_target_logprob_damage": float(
            damage["nonfocal"]["correct_target_logprob"]),
    }


def _generation_pattern(
    analysis: Mapping[str, Any], probe: str,
) -> dict[str, Any]:
    rows = analysis.get("all_primary_generations")
    require(isinstance(rows, list), "primary generation list is absent")
    selected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if (isinstance(row, Mapping) and row.get("schedule") == "N" and
                row.get("region") == "R2_boundary" and
                row.get("cell") in GENERATION_CELLS and
                row.get("probe") == probe):
            cell = str(row["cell"])
            require(cell not in selected, f"duplicate generation cell: {cell}")
            selected[cell] = row
    require(set(selected) == set(GENERATION_CELLS),
            f"generation cell coverage differs for {probe}")
    cells: dict[str, Any] = {}
    for cell in GENERATION_CELLS:
        row = selected[cell]
        record = {
            "decoded_content": row.get("decoded_content"),
            "content_ids": row.get("content_ids"),
            "stop_reason": row.get("stop_reason"),
            "cap_hit": row.get("cap_hit"),
        }
        require(isinstance(record["decoded_content"], str) and
                isinstance(record["content_ids"], list),
                f"generation record differs for {probe}/{cell}")
        record["canonical_sha256"] = strict.sha256_bytes(
            strict.canonical_bytes(record))
        cells[cell] = record
    fresh = cells["FF"]
    return {
        "probe": probe,
        "cells": cells,
        "change_from_fresh": {
            cell: (cells[cell]["decoded_content"], cells[cell]["content_ids"]) !=
            (fresh["decoded_content"], fresh["content_ids"])
            for cell in ("FC", "FW", "CC", "WW")
        },
    }


def _terminal_documents(run_dir: Path) -> tuple[Path, dict[str, Any], Path,
                                                 dict[str, Any]]:
    completion_path = _single(sorted(run_dir.glob(
        "precision-probe-p01-completion_*.json")), "completion marker")
    manifest_path = _single(sorted(run_dir.glob(
        "precision-probe-p01-run-manifest_*.json")), "run manifest")
    completion = strict.load_object(completion_path)
    manifest = strict.load_object(manifest_path)
    require(completion.get("schema") == strict.COMPLETION_SCHEMA and
            completion.get("status") in TERMINAL_NONPASS,
            "p01 completion is not terminal non-PASS")
    require(manifest.get("schema") == strict.RUN_MANIFEST_SCHEMA and
            manifest.get("status") in TERMINAL_NONPASS,
            "p01 manifest is not terminal non-PASS")
    require(completion.get("status") == manifest.get("status"),
            "partial completion/manifest terminal statuses differ")
    strict._require_formal_boundary(completion, "partial completion")
    strict._require_formal_boundary(manifest, "partial manifest")
    bound_manifest = _verify_file_binding(
        run_dir, completion.get("run_manifest"), "completion run manifest")
    require(bound_manifest == manifest_path,
            "completion bound a different run manifest")
    return completion_path, completion, manifest_path, manifest


def _validate_matched_technical_views(
    technical_rows: Mapping[str, Mapping[str, Any]],
    declared_gate: Mapping[str, Any],
) -> dict[str, Any]:
    views: dict[str, dict[str, Any]] = {}
    for regime in strict.REGIMES:
        raw = technical_rows[regime]["raw"]
        runtime = raw["runtime_fingerprint"]
        bindings = raw["subject_attestation"]["bindings"]
        missing = [
            f"runtime.{field}" for field in MATCHED_RUNTIME_FIELDS
            if field not in runtime
        ] + [
            f"bindings.{field}" for field in MATCHED_SUBJECT_BINDING_FIELDS
            if field not in bindings
        ]
        require(not missing,
                f"{regime} matched technical view is incomplete: {missing}")
        views[regime] = {
            "runtime": {
                field: runtime[field] for field in MATCHED_RUNTIME_FIELDS
            },
            "bindings": {
                field: bindings[field]
                for field in MATCHED_SUBJECT_BINDING_FIELDS
            },
        }
    differences = strict.recursive_differences(views["nf4"], views["bf16"])
    require(not differences,
            f"independent matched technical views differ: {differences}")
    hashes = {
        # The frozen runner's matched-view commitment uses canonical JSON plus
        # one trailing newline; reproduce that byte contract independently.
        regime: strict.sha256_bytes(
            strict.canonical_bytes(views[regime]) + b"\n")
        for regime in strict.REGIMES
    }
    require(declared_gate.get("nf4_matched_view_sha256") == hashes["nf4"] and
            declared_gate.get("bf16_matched_view_sha256") == hashes["bf16"],
            "declared matched-runtime view hashes differ from technical raws")
    return {
        "status": "PASS",
        "differing_json_pointers": [],
        "view_sha256": hashes["nf4"],
        "runtime_fields": list(MATCHED_RUNTIME_FIELDS),
        "subject_binding_fields": list(MATCHED_SUBJECT_BINDING_FIELDS),
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    require(run_dir.is_dir(), f"p01 run directory is absent: {run_dir}")
    completion_path, completion, manifest_path, manifest = _terminal_documents(
        run_dir)
    inventory = _verify_completion_inventory(run_dir, completion_path, completion)

    require(manifest.get("model") ==
            "Qwen/Qwen3-30B-A3B-Instruct-2507" and
            manifest.get("revision") ==
            "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
            "partial run subject differs")
    provider = manifest.get("provider")
    require(isinstance(provider, Mapping) and
            isinstance(provider.get("effective_hard_deadline_seconds"),
                       (int, float)) and
            float(completion.get("provider_elapsed_seconds", math.inf)) <=
            float(provider["effective_hard_deadline_seconds"]) and
            float(completion.get("estimated_provider_cost_usd", math.inf)) <= 4.0,
            "partial run exceeded its provider ceiling")
    source_bindings = _validate_source_bindings(manifest.get("bindings"))
    common_bindings = source_bindings["bindings"]

    documents = strict.packaged_p01_documents(run_dir)
    technical_gates = strict._technical_gate_summary(run_dir, documents)
    final_technical_documents = [
        row for row in documents
        if row["raw"].get("schema") == strict.TECHNICAL_RAW_SCHEMA and
        row["raw"].get("status") != "CHECKPOINT"
    ]
    require(len(final_technical_documents) == 2 and
            all(row["raw"].get("status") == "PASS"
                for row in final_technical_documents),
            "p01 partial final technical package set differs")
    technical_rows: dict[str, Mapping[str, Any]] = {}
    for row in documents:
        raw = row["raw"]
        if (raw.get("schema") == strict.TECHNICAL_RAW_SCHEMA and
                raw.get("status") == "PASS"):
            regime = str(raw.get("regime"))
            require(regime in strict.REGIMES and regime not in technical_rows,
                    "duplicate final technical package")
            technical_rows[regime] = row
            _validate_raw_source_bindings(
                raw, common_bindings, f"{regime} technical raw")
    require(set(technical_rows) == set(strict.REGIMES),
            "final technical package coverage differs")

    complete_by_key: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    incomplete_rows: list[Mapping[str, Any]] = []
    for row in documents:
        raw = row["raw"]
        if raw.get("schema") != strict.OUTCOME_RAW_SCHEMA:
            continue
        if raw.get("status") == "CHECKPOINT":
            continue
        key = strict.identify_outcome(raw)
        if raw.get("status") == "COMPLETE":
            require(key not in complete_by_key, f"duplicate complete outcome: {key}")
            complete_by_key[key] = row
        else:
            incomplete_rows.append(row)
    incomplete_keys = [strict.identify_outcome(row["raw"])
                       for row in incomplete_rows]
    validate_partial_layout(set(complete_by_key), incomplete_keys)

    regimes = manifest.get("regimes")
    require(isinstance(regimes, Mapping) and set(regimes) == set(strict.REGIMES),
            "partial manifest regime set differs")
    receipt_by_key: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for regime in strict.REGIMES:
        regime_row = regimes[regime]
        require(isinstance(regime_row, Mapping) and
                regime_row.get("technical", {}).get("status") == "PASS" and
                isinstance(regime_row.get("outcomes"), list),
                f"{regime} manifest technical/outcome record differs")
        require((regime == "nf4" and
                 regime_row.get("status") == "OUTCOMES_FINISHED") or
                (regime == "bf16" and
                 regime_row.get("status") in {
                     "TECHNICAL_PASS", "OUTCOMES_FINISHED"}),
                f"{regime} partial manifest regime status differs")
        technical_receipt = regime_row["technical"]
        raw_technical = technical_rows[regime]["raw"]
        package_dir = _verify_package_binding(
            run_dir, technical_receipt.get("raw_package"),
            technical_rows[regime], f"{regime} technical")
        require(technical_receipt.get("recovery_raw_path") is None and
                technical_receipt.get("persistence_error") is None and
                technical_receipt.get("runtime_fingerprint") ==
                raw_technical.get("runtime_fingerprint") and
                technical_receipt.get("subject_bindings") ==
                raw_technical.get("subject_attestation", {}).get("bindings"),
                f"{regime} technical receipt differs from raw")
        render_path = _verify_file_binding(
            run_dir, technical_receipt.get("renders"),
            f"{regime} technical render")
        _verify_technical_render(
            raw_technical, technical_receipt["raw_package"], render_path)
        require(package_dir == technical_rows[regime]["package_dir"],
                f"{regime} technical package resolution differs")

        for receipt in regime_row["outcomes"]:
            require(isinstance(receipt, Mapping) and
                    receipt.get("regime") == regime and
                    receipt.get("case_id") in {"e01", "e02", "e03"} and
                    isinstance(receipt.get("repeat_index"), int),
                    f"{regime} manifest outcome receipt differs")
            key = (regime, str(receipt["case_id"]),
                   int(receipt["repeat_index"]))
            require(key not in receipt_by_key,
                    f"duplicate manifest outcome receipt: {key}")
            receipt_by_key[key] = receipt
    require(set(receipt_by_key) == EXPECTED_COMPLETE_KEYS and
            all(row.get("completion_status") == "COMPLETE"
                for row in receipt_by_key.values()),
            "manifest complete-outcome receipt set differs")

    matched_gate = manifest.get("matched_runtime_gate")
    require(isinstance(matched_gate, Mapping) and
            matched_gate.get("status") == "PASS" and
            matched_gate.get("differing_fields") == [] and
            completion.get("matched_runtime_gate") == matched_gate and
            technical_gates.get("status") == "PASS" and
            technical_gates.get("same_host") is True,
            "partial matched-runtime gate differs")
    independently_matched = _validate_matched_technical_views(
        technical_rows, matched_gate)

    outcome_analyses: dict[str, Any] = {}
    package_verification: dict[str, Any] = {}
    for key in sorted(complete_by_key):
        regime, case_id, repeat = key
        row = complete_by_key[key]
        raw = row["raw"]
        receipt = receipt_by_key[key]
        strict._require_formal_boundary(raw, f"{regime}/{case_id}/r{repeat}")
        _validate_raw_source_bindings(
            raw, common_bindings, f"{regime}/{case_id}/r{repeat}")
        technical_raw = technical_rows[regime]["raw"]
        require(raw.get("runtime_fingerprint") ==
                technical_raw.get("runtime_fingerprint") and
                raw.get("subject_attestation") ==
                technical_raw.get("subject_attestation"),
                f"{regime}/{case_id}/r{repeat} subject differs from technical")
        require(receipt.get("recovery_raw_path") is None and
                math.isclose(float(receipt.get("outcome_wall_time_seconds")),
                             float(raw.get("outcome_wall_time_seconds")),
                             rel_tol=0, abs_tol=1e-12),
                f"{regime}/{case_id}/r{repeat} timing receipt differs")
        package_dir = _verify_package_binding(
            run_dir, receipt.get("raw_package"), row,
            f"{regime}/{case_id}/r{repeat}")
        compact_path = _verify_file_binding(
            run_dir, receipt.get("compact"),
            f"{regime}/{case_id}/r{repeat} compact")
        render_path = _verify_file_binding(
            run_dir, receipt.get("renders"),
            f"{regime}/{case_id}/r{repeat} renders")
        verify_outcome_derived_content(
            raw, receipt["raw_package"], compact_path, render_path)
        analysis = strict.analyze_outcome(raw)
        label = f"{regime}:{case_id}:r{repeat}"
        outcome_analyses[label] = {
            "regime": regime,
            "case_id": case_id,
            "repeat": repeat,
            "package": {
                "path": package_dir.relative_to(run_dir).as_posix(),
                "manifest_sha256": row["package_manifest_sha256"],
                "raw_artifact_sha256": row["raw_artifact_sha256"],
            },
            "continuous_descriptive": _continuous(analysis),
            "generation_patterns": {
                probe: _generation_pattern(analysis, probe)
                for probe in ("focal", "nonfocal")
            },
            "full_grid_analysis": analysis,
        }
        package_verification[label] = {
            "raw_package": "VERIFIED",
            "arm_grid": "31_PRIMARY_PLUS_3_PLACEBO_ATTEMPTS_VERIFIED",
            "compact_consistency": "PASS",
            "render_consistency": "PASS",
        }

    incomplete_summary: dict[str, Any]
    if incomplete_rows:
        row = incomplete_rows[0]
        raw = row["raw"]
        strict._require_formal_boundary(raw, "bf16/e01/r2 incomplete")
        _validate_raw_source_bindings(raw, common_bindings,
                                      "bf16/e01/r2 incomplete")
        require(raw.get("status") != "COMPLETE" and
                not (isinstance(raw.get("phase_a"), Mapping) and
                     isinstance(raw.get("treatment"), Mapping)),
                "bf16 e01 repeat 2 is not genuinely incomplete")
        require(raw.get("runtime_fingerprint") ==
                technical_rows["bf16"]["raw"].get("runtime_fingerprint") and
                raw.get("subject_attestation") ==
                technical_rows["bf16"]["raw"].get("subject_attestation"),
                "incomplete bf16 repeat subject differs from technical")
        compact_paths = sorted(run_dir.glob(
            "*outcome-bf16-e01-repeat2-compact_*.json"))
        require(not compact_paths,
                "incomplete bf16 repeat unexpectedly has a compact artifact")
        render_paths = sorted(run_dir.glob(
            "*outcome-bf16-e01-repeat2-renders_*.md"))
        require(len(render_paths) <= 1,
                "incomplete bf16 repeat has duplicate render ledgers")
        render_status = "ABSENT"
        if render_paths:
            binding = _derived_package_binding(row, run_dir)
            document = dict(raw)
            document["raw_package_binding"] = binding
            require(render_paths[0].read_text(encoding="utf-8") ==
                    runner.render_outcome_ledger(document),
                    "incomplete bf16 render differs from reconstructed raw")
            render_status = "PASS"
        incomplete_summary = {
            "key": list(EXPECTED_INCOMPLETE_KEY),
            "status": raw.get("status"),
            "error_type": raw.get("error", {}).get("type"),
            "package": {
                "path": row["package_dir"].relative_to(run_dir).as_posix(),
                "manifest_sha256": row["package_manifest_sha256"],
                "raw_artifact_sha256": row["raw_artifact_sha256"],
            },
            "phase_a_present": isinstance(raw.get("phase_a"), Mapping),
            "treatment_present": isinstance(raw.get("treatment"), Mapping),
            "render_consistency": render_status,
            "compact_present": False,
        }
    else:
        incomplete_summary = {
            "key": list(EXPECTED_INCOMPLETE_KEY),
            "status": "ABSENT",
            "package": None,
            "render_consistency": "NOT_APPLICABLE",
            "compact_present": False,
        }

    extension = _validate_extension_decision(
        run_dir, manifest, complete_by_key)

    nf4_left = complete_by_key[("nf4", "e01", 1)]["raw"]
    nf4_right = complete_by_key[("nf4", "e01", 2)]["raw"]
    left_payload = {"phase_a": nf4_left["phase_a"],
                    "treatment": nf4_left["treatment"]}
    right_payload = {"phase_a": nf4_right["phase_a"],
                     "treatment": nf4_right["treatment"]}
    nf4_differences = strict.recursive_differences(left_payload, right_payload)
    nf4_stability = {
        "status": "PASS" if not nf4_differences else "FAIL",
        "stable": not nf4_differences,
        "repeat1_canonical_sha256": strict.sha256_bytes(
            strict.canonical_bytes(left_payload)),
        "repeat2_canonical_sha256": strict.sha256_bytes(
            strict.canonical_bytes(right_payload)),
        "differing_json_pointers": nf4_differences,
        "differing_field_count": len(nf4_differences),
        "note": "Execution repeats are a determinism check, not independent cases.",
    }

    nf4_r1 = outcome_analyses["nf4:e01:r1"]
    bf16_r1 = outcome_analyses["bf16:e01:r1"]
    nf4_values = nf4_r1["continuous_descriptive"]
    bf16_values = bf16_r1["continuous_descriptive"]
    matched_deltas = {
        name: nf4_values[name] - bf16_values[name]
        for name in sorted(nf4_values)
    }
    generation_concordance: dict[str, Any] = {}
    for probe in ("focal", "nonfocal"):
        nf4_pattern = nf4_r1["generation_patterns"][probe]
        bf16_pattern = bf16_r1["generation_patterns"][probe]
        generation_concordance[probe] = {
            "literal_cells_equal": {
                cell: nf4_pattern["cells"][cell] ==
                bf16_pattern["cells"][cell]
                for cell in GENERATION_CELLS
            },
            "change_vectors_equal": (
                nf4_pattern["change_from_fresh"] ==
                bf16_pattern["change_from_fresh"]),
            "nf4_change_from_fresh": nf4_pattern["change_from_fresh"],
            "bf16_change_from_fresh": bf16_pattern["change_from_fresh"],
        }

    return {
        "schema": ANALYSIS_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "analysis_status": ANALYSIS_STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir),
        "post_run_analysis": True,
        "terminal_partial_run": True,
        "runner_terminal_status": manifest["status"],
        "formal_v12_decision_eligible": False,
        "v12_reentry_authorized": False,
        "component_reuse_does_not_inherit_v12_eligibility": True,
        "semantic_evidence_eligible": False,
        "fixed_fixture_exploratory_only": True,
        "complete_outcome_keys": [list(key)
                                  for key in sorted(complete_by_key)],
        "required_bf16_repeat2_disposition": incomplete_summary,
        "terminal_files": {
            "manifest": {
                "path": manifest_path.relative_to(run_dir).as_posix(),
                "sha256": strict.file_sha256(manifest_path),
            },
            "completion": {
                "path": completion_path.relative_to(run_dir).as_posix(),
                "sha256": strict.file_sha256(completion_path),
            },
            "completion_inventory": inventory,
        },
        "source_bindings": source_bindings,
        "extension_decision": extension,
        "technical_gates": technical_gates,
        "matched_runtime_gate": dict(matched_gate),
        "independently_reconstructed_matched_runtime": independently_matched,
        "package_and_derived_verification": package_verification,
        "outcomes": outcome_analyses,
        "nf4_repeat_stability": nf4_stability,
        "matched_repeat1_descriptive": {
            "nf4": nf4_values,
            "bf16": bf16_values,
            "nf4_minus_bf16": matched_deltas,
            "generation_concordance": generation_concordance,
            "interpretation_scope": (
                "One engineered e01 fixture under a bundled weight/kernel "
                "runtime axis; continuous and literal-generation description only."
            ),
        },
        "inference": {
            "p_values_computed": False,
            "confidence_intervals_computed": False,
            "equivalence_test_computed": False,
            "formal_release_decision_computed": False,
            "population_interaction_claim_authorized": False,
            "small_cross_runtime_contrast_interpretation_permitted": False,
            "note": (
                "P01 terminated before bfloat16 repeat 2.  The matched repeat-1 "
                "comparison is post-run, partial, fixed-fixture, and descriptive; "
                "it does not identify semantic attribution, efficacy, 4-bit KV "
                "behavior, or population quantization dependence."
            ),
        },
    }


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True,
                          ensure_ascii=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return ROOT / "results/precision_probe_p01_analysis" / (
        "precision-probe-p01-postrun-partial-descriptive_"
        f"Qwen3-30B-A3B-Instruct-2507_{stamp}.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze_run(args.run_dir)
    output = args.output.resolve() if args.output else default_output()
    print(
        f"ANALYZE {ANALYSIS_STATUS} run={args.run_dir.resolve()} -> {output}",
        flush=True,
    )
    write_exclusive(output, result)
    print(json.dumps({
        "analysis_status": ANALYSIS_STATUS,
        "output": str(output),
        "output_sha256": strict.file_sha256(output),
        "runner_terminal_status": result["runner_terminal_status"],
        "nf4_repeat_stable": result["nf4_repeat_stability"]["stable"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
