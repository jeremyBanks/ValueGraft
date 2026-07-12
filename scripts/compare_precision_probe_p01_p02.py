#!/usr/bin/env python3
"""Fail-closed comparison of independent P01 and P02 analysis artifacts.

The inputs are the analyzer outputs, never runner compact ledgers.  This tool
accepts only the observed terminal-partial P01 boundary and the observed
MATCHED_COMPLETE P02 boundary.  It reports exact fixed-fixture values and
literal equality only.  It deliberately defines no tolerance, threshold,
ratio, p-value, confidence interval, semantic claim, or quantization claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA = "precision_probe_p01_p02_exact_comparison_v1"
P01_SCHEMA = "precision_probe_p01_postrun_partial_descriptive_analysis_v1"
P02_SCHEMA = "precision_probe_p02_independent_analysis_v1"
P01_PROTOCOL = "precision-probe-p01"
P02_PROTOCOL = "precision-probe-p02"
CASE_ID = "e01"
REGIMES = ("nf4", "bf16")
PROBES = ("focal", "nonfocal")
CELLS = ("FF", "FC", "FW", "CC", "WW")
CHANGE_CELLS = ("FC", "FW", "CC", "WW")
EXPECTED_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
EXPECTED_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
P01_PREREG_SHA256 = (
    "5619c5ced93f2e60564fcc2c98e24fd9bbf687f93b6cab6132f5be2105cda374"
)
P02_PREREG_SHA256 = (
    "dbcece8f189a0574b776149cf597dc1ce8a3b592981440062c749ddb404e221d"
)
EXPECTED_DEPENDENCIES = {
    "accelerate": "1.14.0",
    "bitsandbytes": "0.49.2",
    "huggingface-hub": "0.36.2",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.12.1",
    "transformers": "4.57.6",
}
EXPECTED_GEOMETRY = {
    "attention_heads": 32,
    "experts_per_layer": 128,
    "head_dim": 128,
    "hidden_size": 2048,
    "kv_heads": 4,
    "layers": 48,
    "moe_intermediate_size": 768,
    "rope_theta": 10000000.0,
    "vocab_size": 151936,
}
FORMAL_BOUNDARY = {
    "formal_v12_decision_eligible": False,
    "v12_reentry_authorized": False,
    "component_reuse_does_not_inherit_v12_eligibility": True,
    "semantic_evidence_eligible": False,
}

P01_TOP_KEYS = {
    "analysis_status",
    "complete_outcome_keys",
    "component_reuse_does_not_inherit_v12_eligibility",
    "created_at_utc",
    "extension_decision",
    "fixed_fixture_exploratory_only",
    "formal_v12_decision_eligible",
    "independently_reconstructed_matched_runtime",
    "inference",
    "matched_repeat1_descriptive",
    "matched_runtime_gate",
    "nf4_repeat_stability",
    "outcomes",
    "package_and_derived_verification",
    "post_run_analysis",
    "protocol_id",
    "required_bf16_repeat2_disposition",
    "run_directory",
    "runner_terminal_status",
    "schema",
    "semantic_evidence_eligible",
    "source_bindings",
    "technical_gates",
    "terminal_files",
    "terminal_partial_run",
    "v12_reentry_authorized",
}
P02_TOP_KEYS = {
    "all_lossless_packages",
    "component_reuse_does_not_inherit_v12_eligibility",
    "continuation_decisions",
    "created_at_utc",
    "fixed_fixture_exploratory_only",
    "formal_v12_decision_eligible",
    "inference",
    "matched_repeat1_descriptive",
    "matched_runtime_gate",
    "outcomes",
    "outer_pod_receipt",
    "package_and_derived_verification",
    "primary_dataset",
    "protocol_id",
    "repeat2_rider_status",
    "repeat_stability_and_interpretation_guard",
    "run_directory",
    "runner_terminal_status",
    "schema",
    "semantic_evidence_eligible",
    "source_bindings",
    "technical_gates",
    "terminal_files",
    "v12_reentry_authorized",
}
P01_OUTCOME_KEYS = {
    "case_id",
    "continuous_descriptive",
    "full_grid_analysis",
    "generation_patterns",
    "package",
    "regime",
    "repeat",
}
P02_OUTCOME_KEYS = {"analysis", "case_id", "primary", "regime", "repeat"}
P02_ANALYSIS_KEYS = {
    "E1_gross_compaction_damage",
    "E2_literal_generated_behavior",
    "E3_graft_contrasts",
    "case_id",
    "regime",
    "repeat_index",
    "secondary_controls",
}
P01_FULL_GRID_KEYS = {
    "N_minus_P_value_D_shift",
    "N_regional_estimands",
    "P_R2_estimands",
    "all_primary_generations",
    "available_placebo_control_count",
    "case_id",
    "phase_a_oracle_to_fresh_damage",
    "placebo_controls",
    "primary_D_value_N_R2",
    "primary_Hplus_value_N_R2",
    "primary_SEL_signed_p01",
    "primary_nonfocal_D_value_N_R2",
}
GENERATION_RECORD_KEYS = {
    "canonical_sha256", "cap_hit", "content_ids", "decoded_content",
    "stop_reason",
}
SCALAR_NAMES = (
    "E1_focal_margin_damage",
    "E1_nonfocal_margin_damage",
    "N_R2_D_value_focal",
    "N_R2_D_value_nonfocal",
    "N_R2_D_full_KV_focal",
    "N_R2_D_full_KV_nonfocal",
    "P_R2_D_value_focal",
    "P_R2_D_value_nonfocal",
    "N_minus_P_value_shift_focal",
    "N_minus_P_value_shift_nonfocal",
)
P02_SCALAR_NAMES = {
    "E1_focal_margin_damage",
    "E1_focal_minus_nonfocal",
    "E1_nonfocal_margin_damage",
    "E3_N_R2_D_full_KV_focal",
    "E3_N_R2_D_full_KV_nonfocal",
    "E3_N_R2_D_value_focal",
    "E3_N_R2_D_value_nonfocal",
    "secondary_N_minus_P_shift_focal",
    "secondary_N_minus_P_shift_nonfocal",
    "secondary_P_R2_D_value_focal",
    "secondary_P_R2_D_value_nonfocal",
}
P02_SCALAR_TO_NORMALIZED = {
    "E1_focal_margin_damage": "E1_focal_margin_damage",
    "E1_nonfocal_margin_damage": "E1_nonfocal_margin_damage",
    "E3_N_R2_D_value_focal": "N_R2_D_value_focal",
    "E3_N_R2_D_value_nonfocal": "N_R2_D_value_nonfocal",
    "E3_N_R2_D_full_KV_focal": "N_R2_D_full_KV_focal",
    "E3_N_R2_D_full_KV_nonfocal": "N_R2_D_full_KV_nonfocal",
    "secondary_P_R2_D_value_focal": "P_R2_D_value_focal",
    "secondary_P_R2_D_value_nonfocal": "P_R2_D_value_nonfocal",
    "secondary_N_minus_P_shift_focal": "N_minus_P_value_shift_focal",
    "secondary_N_minus_P_shift_nonfocal": "N_minus_P_value_shift_nonfocal",
}


class ComparisonError(RuntimeError):
    """An analysis artifact or its comparison boundary differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} is not an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    require(isinstance(value, list), f"{label} is not a list")
    return value


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    row = _mapping(value, label)
    require(set(row) == expected, f"{label} keys differ")
    return row


def _number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} is not a finite number",
    )
    return float(value)


def _sha(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{label} is not a lowercase SHA-256",
    )
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha_fields(value: Any, label: str = "artifact") -> None:
    """Reject malformed SHA-bearing analyzer fields recursively."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_label = f"{label}.{key}"
            if key == "sha256" or key.endswith("_sha256"):
                _sha(child, child_label)
            elif key.endswith("_sha256s"):
                rows = _sequence(child, child_label)
                for index, item in enumerate(rows):
                    _sha(item, f"{child_label}[{index}]")
            else:
                _validate_sha_fields(child, child_label)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_sha_fields(child, f"{label}[{index}]")


def load_artifact(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except Exception as exc:
        raise ComparisonError(f"cannot load {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} root is not an object")
    return value, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _validate_binding(value: Any, label: str) -> dict[str, Any]:
    row = _exact_keys(value, {"path", "sha256", "size_bytes"}, label)
    require(isinstance(row["path"], str) and row["path"], f"{label} path differs")
    size = row["size_bytes"]
    require(isinstance(size, int) and not isinstance(size, bool) and size > 0,
            f"{label} size differs")
    return {"path": row["path"], "sha256": _sha(row["sha256"], label),
            "size_bytes": size}


def _validate_formal_boundary(document: Mapping[str, Any], label: str) -> None:
    for key, expected in FORMAL_BOUNDARY.items():
        require(document.get(key) is expected, f"{label} formal boundary differs: {key}")
    require(document.get("fixed_fixture_exploratory_only") is True,
            f"{label} fixed-fixture boundary differs")


def _validate_inference(document: Mapping[str, Any], label: str) -> None:
    row = _mapping(document.get("inference"), f"{label} inference")
    forbidden = {
        "p_values_computed",
        "confidence_intervals_computed",
        "equivalence_test_computed",
        "formal_release_decision_computed",
        "population_interaction_claim_authorized",
    }
    if label == "P01":
        forbidden.add("small_cross_runtime_contrast_interpretation_permitted")
    else:
        forbidden.update({"ratios_computed", "quantization_dependence_claim_authorized"})
    require(all(row.get(key) is False for key in forbidden),
            f"{label} inference boundary differs")
    require(isinstance(row.get("note"), str) and row["note"],
            f"{label} inference note is absent")


def _validate_source_bindings(document: Mapping[str, Any], label: str) -> None:
    row = _mapping(document.get("source_bindings"), f"{label} source bindings")
    require(row.get("status") == "PASS", f"{label} source bindings did not pass")
    names = _sequence(row.get("binding_names"), f"{label} binding names")
    require(names == sorted(names) and len(names) == len(set(names)),
            f"{label} binding names differ")
    _sha(row.get("canonical_sha256"), f"{label} source-binding digest")
    bindings = _mapping(row.get("bindings"), f"{label} bindings")
    require(set(bindings) == set(names), f"{label} binding inventory differs")
    prereg = _mapping(bindings.get("preregistration"), f"{label} preregistration")
    expected_prereg = P01_PREREG_SHA256 if label == "P01" else P02_PREREG_SHA256
    require(prereg.get("sha256") == expected_prereg,
            f"{label} preregistration SHA differs")


def _validate_generation_record(value: Any, label: str) -> dict[str, Any]:
    row = _exact_keys(value, GENERATION_RECORD_KEYS, label)
    decoded = row["decoded_content"]
    ids = row["content_ids"]
    stop = row["stop_reason"]
    cap = row["cap_hit"]
    require(isinstance(decoded, str), f"{label} decoded content differs")
    require(
        isinstance(ids, list)
        and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0
                for item in ids),
        f"{label} token IDs differ",
    )
    require(isinstance(stop, str) and stop, f"{label} stop reason differs")
    require(isinstance(cap, bool), f"{label} cap flag differs")
    expected_hash = _canonical_sha256({
        "decoded_content": decoded,
        "content_ids": ids,
        "stop_reason": stop,
        "cap_hit": cap,
    })
    require(row["canonical_sha256"] == expected_hash,
            f"{label} canonical generation SHA differs")
    return {
        "decoded_content": decoded,
        "content_ids": list(ids),
        "stop_reason": stop,
        "cap_hit": cap,
        "canonical_sha256": row["canonical_sha256"],
    }


def _normalize_generation_pattern(
    value: Any, label: str, *, p02: bool,
) -> dict[str, Any]:
    expected = {"probe", "cells", "change_from_fresh"}
    if p02:
        expected |= {
            "four_bit_change_vector",
            "generation_hash_tuple",
            "literal_decoded_content_tuple",
            "literal_generation_tuple",
        }
    row = _exact_keys(value, expected, label)
    probe = row["probe"]
    require(probe in PROBES, f"{label} probe differs")
    cell_rows = _mapping(row["cells"], f"{label} cells")
    require(set(cell_rows) == set(CELLS), f"{label} cell set differs")
    cells = {
        cell: _validate_generation_record(cell_rows[cell], f"{label}/{cell}")
        for cell in CELLS
    }
    changes = {
        cell: (
            cells[cell]["decoded_content"], cells[cell]["content_ids"]
        ) != (cells["FF"]["decoded_content"], cells["FF"]["content_ids"])
        for cell in CHANGE_CELLS
    }
    require(dict(row["change_from_fresh"]) == changes,
            f"{label} change-from-fresh vector differs")
    if p02:
        require(row["literal_generation_tuple"] == [cells[cell] for cell in CELLS],
                f"{label} literal generation tuple differs")
        require(row["literal_decoded_content_tuple"] ==
                [cells[cell]["decoded_content"] for cell in CELLS],
                f"{label} decoded tuple differs")
        require(row["generation_hash_tuple"] ==
                [cells[cell]["canonical_sha256"] for cell in CELLS],
                f"{label} generation hash tuple differs")
        require(row["four_bit_change_vector"] ==
                [changes[cell] for cell in CHANGE_CELLS],
                f"{label} four-bit change vector differs")
    return {
        "cell_order": list(CELLS),
        "decoded_content": [cells[cell]["decoded_content"] for cell in CELLS],
        "content_ids": [cells[cell]["content_ids"] for cell in CELLS],
        "stop_reasons": [cells[cell]["stop_reason"] for cell in CELLS],
        "cap_hit": [cells[cell]["cap_hit"] for cell in CELLS],
        "canonical_sha256": [cells[cell]["canonical_sha256"] for cell in CELLS],
        "change_vector_cell_order": list(CHANGE_CELLS),
        "change_vector": [changes[cell] for cell in CHANGE_CELLS],
    }


def _normalize_placebo(value: Any, label: str, *, repeat1: bool) -> dict[str, Any]:
    row = _mapping(value, label)
    if not repeat1:
        require(
            set(row) == {"scores", "status"}
            and row.get("status") == "NOT_ATTEMPTED_BY_PROTOCOL_REPEAT2"
            and row.get("scores") is None,
            f"{label} repeat-2 placebo boundary differs",
        )
        return {"status": row["status"], "scores_present": False}
    require(set(row) == {"diagnostics", "movement_from_fresh", "scores", "status"},
            f"{label} keys differ")
    diagnostics = _mapping(row["diagnostics"], f"{label} diagnostics")
    require(
        row.get("status") == "PLACEBO_UNAVAILABLE"
        and diagnostics.get("status") == "PLACEBO_UNAVAILABLE"
        and row.get("scores") is None
        and row.get("movement_from_fresh") is None,
        f"{label} missing-control status differs",
    )
    layer = diagnostics.get("failed_layer_index")
    row_index = diagnostics.get("failed_row_index")
    require(
        isinstance(layer, int) and not isinstance(layer, bool) and layer >= 0
        and isinstance(row_index, int) and not isinstance(row_index, bool)
        and row_index >= 0,
        f"{label} failure coordinate differs",
    )
    source_hashes = _mapping(
        diagnostics.get("source_result_hashes"), f"{label} source-result hashes"
    )
    return {
        "status": row["status"],
        "scores_present": False,
        "movement_from_fresh": None,
        "failed_layer_index": layer,
        "failed_row_index": row_index,
        "canonical_diagnostics_sha256": _sha(
            diagnostics.get("canonical_diagnostics_sha256"),
            f"{label} diagnostics SHA",
        ),
        "source_result_hashes": dict(source_hashes),
    }


def _p01_outcome(value: Any, regime: str, repeat: int) -> dict[str, Any]:
    label = f"P01 {regime} repeat {repeat}"
    row = _exact_keys(value, P01_OUTCOME_KEYS, label)
    require(
        row["case_id"] == CASE_ID and row["regime"] == regime
        and row["repeat"] == repeat,
        f"{label} identity differs",
    )
    package = _exact_keys(
        row["package"], {"manifest_sha256", "path", "raw_artifact_sha256"},
        f"{label} package",
    )
    require(isinstance(package["path"], str) and package["path"],
            f"{label} package path differs")
    full = _exact_keys(row["full_grid_analysis"], P01_FULL_GRID_KEYS,
                       f"{label} full grid")
    require(full["case_id"] == CASE_ID, f"{label} grid case differs")
    damage = _mapping(full["phase_a_oracle_to_fresh_damage"], f"{label} E1")
    n_r2 = _mapping(_mapping(full["N_regional_estimands"], f"{label} N").get(
        "R2_boundary"), f"{label} N/R2")
    n_value = _mapping(n_r2.get("value_only"), f"{label} N/R2 value")
    n_full = _mapping(n_r2.get("full_KV"), f"{label} N/R2 full KV")
    p_value = _mapping(
        _mapping(full["P_R2_estimands"], f"{label} P/R2").get("value_only"),
        f"{label} P/R2 value",
    )
    scalars = {
        "E1_focal_margin_damage": _number(
            _mapping(damage.get("focal"), f"{label} E1 focal").get("margin"),
            f"{label} E1 focal margin",
        ),
        "E1_nonfocal_margin_damage": _number(
            _mapping(damage.get("nonfocal"), f"{label} E1 nonfocal").get("margin"),
            f"{label} E1 nonfocal margin",
        ),
        "N_R2_D_value_focal": _number(n_value.get("D_focal"), f"{label} N Dv focal"),
        "N_R2_D_value_nonfocal": _number(
            n_value.get("D_nonfocal"), f"{label} N Dv nonfocal"),
        "N_R2_D_full_KV_focal": _number(
            n_full.get("D_focal"), f"{label} N Dkv focal"),
        "N_R2_D_full_KV_nonfocal": _number(
            n_full.get("D_nonfocal"), f"{label} N Dkv nonfocal"),
        "P_R2_D_value_focal": _number(p_value.get("D_focal"), f"{label} P Dv focal"),
        "P_R2_D_value_nonfocal": _number(
            p_value.get("D_nonfocal"), f"{label} P Dv nonfocal"),
    }
    scalars["N_minus_P_value_shift_focal"] = (
        scalars["N_R2_D_value_focal"] - scalars["P_R2_D_value_focal"]
    )
    scalars["N_minus_P_value_shift_nonfocal"] = (
        scalars["N_R2_D_value_nonfocal"] - scalars["P_R2_D_value_nonfocal"]
    )
    require(full["primary_D_value_N_R2"] == scalars["N_R2_D_value_focal"]
            and full["primary_nonfocal_D_value_N_R2"] ==
            scalars["N_R2_D_value_nonfocal"]
            and full["N_minus_P_value_D_shift"] ==
            scalars["N_minus_P_value_shift_focal"],
            f"{label} stored scalar bindings differ")
    patterns = _mapping(row["generation_patterns"], f"{label} generations")
    require(set(patterns) == set(PROBES), f"{label} generation probe set differs")
    generations = {
        probe: _normalize_generation_pattern(
            patterns[probe], f"{label} {probe} generations", p02=False)
        for probe in PROBES
    }
    placebos = _mapping(full["placebo_controls"], f"{label} placebos")
    require(full["available_placebo_control_count"] == 0,
            f"{label} available placebo count differs")
    placebo = _normalize_placebo(
        placebos.get("R2_boundary"), f"{label} R2 placebo", repeat1=True
    )
    continuous = _mapping(row["continuous_descriptive"], f"{label} continuous")
    bindings = {
        "D_full_KV_N_R2_focal": "N_R2_D_full_KV_focal",
        "D_value_N_R2_focal": "N_R2_D_value_focal",
        "D_value_N_R2_nonfocal": "N_R2_D_value_nonfocal",
        "D_value_P_R2_focal": "P_R2_D_value_focal",
        "N_minus_P_value_D_shift": "N_minus_P_value_shift_focal",
        "oracle_to_fresh_focal_margin_damage": "E1_focal_margin_damage",
        "oracle_to_fresh_nonfocal_margin_damage": "E1_nonfocal_margin_damage",
    }
    require(all(continuous.get(source) == scalars[target]
                for source, target in bindings.items()),
            f"{label} continuous summary differs from full grid")
    return {
        "estimands": scalars,
        "generations": generations,
        "placebo": placebo,
        "package": dict(package),
    }


def _p02_outcome(value: Any, regime: str, repeat: int) -> dict[str, Any]:
    label = f"P02 {regime} repeat {repeat}"
    row = _exact_keys(value, P02_OUTCOME_KEYS, label)
    require(
        row["case_id"] == CASE_ID and row["regime"] == regime
        and row["repeat"] == repeat and row["primary"] is (repeat == 1),
        f"{label} identity differs",
    )
    analysis = _exact_keys(row["analysis"], P02_ANALYSIS_KEYS, f"{label} analysis")
    require(
        analysis["case_id"] == CASE_ID and analysis["regime"] == regime
        and analysis["repeat_index"] == repeat,
        f"{label} analysis identity differs",
    )
    damage = _mapping(analysis["E1_gross_compaction_damage"], f"{label} E1")
    e3 = _mapping(analysis["E3_graft_contrasts"], f"{label} E3")
    n_value = _mapping(e3.get("N_R2_D_value"), f"{label} N/R2 value")
    n_full = _mapping(e3.get("N_R2_D_full_KV"), f"{label} N/R2 full KV")
    secondary = _mapping(analysis["secondary_controls"], f"{label} controls")
    p_value = _mapping(secondary.get("P_R2_D_value"), f"{label} P/R2 value")
    shift = _mapping(secondary.get("N_minus_P_value_shift"), f"{label} N-P")
    scalars = {
        "E1_focal_margin_damage": _number(
            _mapping(damage.get("focal"), f"{label} E1 focal").get("margin"),
            f"{label} E1 focal margin",
        ),
        "E1_nonfocal_margin_damage": _number(
            _mapping(damage.get("nonfocal"), f"{label} E1 nonfocal").get("margin"),
            f"{label} E1 nonfocal margin",
        ),
        "N_R2_D_value_focal": _number(n_value.get("D_focal"), f"{label} N Dv focal"),
        "N_R2_D_value_nonfocal": _number(
            n_value.get("D_nonfocal"), f"{label} N Dv nonfocal"),
        "N_R2_D_full_KV_focal": _number(
            n_full.get("D_focal"), f"{label} N Dkv focal"),
        "N_R2_D_full_KV_nonfocal": _number(
            n_full.get("D_nonfocal"), f"{label} N Dkv nonfocal"),
        "P_R2_D_value_focal": _number(p_value.get("D_focal"), f"{label} P Dv focal"),
        "P_R2_D_value_nonfocal": _number(
            p_value.get("D_nonfocal"), f"{label} P Dv nonfocal"),
        "N_minus_P_value_shift_focal": _number(
            shift.get("focal"), f"{label} N-P focal"),
        "N_minus_P_value_shift_nonfocal": _number(
            shift.get("nonfocal"), f"{label} N-P nonfocal"),
    }
    require(
        scalars["N_minus_P_value_shift_focal"] ==
        scalars["N_R2_D_value_focal"] - scalars["P_R2_D_value_focal"]
        and scalars["N_minus_P_value_shift_nonfocal"] ==
        scalars["N_R2_D_value_nonfocal"] - scalars["P_R2_D_value_nonfocal"],
        f"{label} N-minus-P derivation differs",
    )
    patterns = _mapping(analysis["E2_literal_generated_behavior"], f"{label} E2")
    require(set(patterns) == set(PROBES), f"{label} generation probe set differs")
    generations = {
        probe: _normalize_generation_pattern(
            patterns[probe], f"{label} {probe} generations", p02=True)
        for probe in PROBES
    }
    placebo = _normalize_placebo(
        secondary.get("placebo"), f"{label} placebo", repeat1=(repeat == 1)
    )
    require(secondary.get("FF_fresh_serialized_identity") is True,
            f"{label} serialized FF identity differs")
    return {"estimands": scalars, "generations": generations, "placebo": placebo}


def _validate_terminal_files(document: Mapping[str, Any], label: str) -> None:
    terminal = _mapping(document.get("terminal_files"), f"{label} terminal files")
    for name in ("manifest", "completion"):
        binding = _exact_keys(terminal.get(name), {"path", "sha256"},
                              f"{label} {name}")
        require(isinstance(binding["path"], str) and binding["path"],
                f"{label} {name} path differs")
        _sha(binding["sha256"], f"{label} {name} SHA")
    inventory = _mapping(terminal.get("completion_inventory"),
                         f"{label} completion inventory")
    require(inventory.get("status") == "PASS"
            and isinstance(inventory.get("file_count"), int)
            and inventory["file_count"] > 0,
            f"{label} completion inventory differs")


def _validate_runtime(document: Mapping[str, Any], label: str) -> dict[str, Any]:
    gates = _mapping(document.get("technical_gates"), f"{label} technical gates")
    require(gates.get("status") == "PASS", f"{label} technical gate did not pass")
    if label == "P01":
        require(gates.get("same_host") is True
                and gates.get("stable_runtime_fields_equal") is True,
                "P01 technical same-runtime gate differs")
    regime_rows = _mapping(gates.get("regimes"), f"{label} technical regimes")
    require(set(regime_rows) == set(REGIMES), f"{label} technical regime set differs")
    compact: dict[str, Any] = {}
    for regime in REGIMES:
        row = _mapping(regime_rows[regime], f"{label} {regime} technical")
        status_key = "g2_status" if label == "P01" else "status"
        require(row.get(status_key) == "PASS", f"{label} {regime} technical failed")
        host = _mapping(row.get("host"), f"{label} {regime} host")
        fingerprint = _mapping(row.get("runtime_fingerprint"),
                               f"{label} {regime} runtime fingerprint")
        require(
            fingerprint.get("schema") == "precision_probe_p01_runtime_fingerprint_v1"
            and fingerprint.get("protocol_id") == P01_PROTOCOL
            and fingerprint.get("regime") == regime
            and fingerprint.get("model_id") == EXPECTED_MODEL
            and fingerprint.get("requested_revision") == EXPECTED_REVISION
            and fingerprint.get("resolved_snapshot") == EXPECTED_REVISION
            and fingerprint.get("architecture") == "Qwen3MoeForCausalLM"
            and fingerprint.get("attention_backend") == "eager"
            and fingerprint.get("kv_dtype") == "torch.bfloat16"
            and fingerprint.get("eos_ids") == [151643, 151645]
            and fingerprint.get("geometry") == EXPECTED_GEOMETRY
            and fingerprint.get("dependency_versions") == EXPECTED_DEPENDENCIES,
            f"{label} {regime} exact runtime differs",
        )
        expected_weight_runtime = (
            "bitsandbytes-nf4-double-quant-bf16-compute"
            if regime == "nf4" else "bf16"
        )
        require(fingerprint.get("weight_runtime") == expected_weight_runtime,
                f"{label} {regime} weight runtime differs")
        require(
            isinstance(host.get("gpu_uuid"), str)
            and re.fullmatch(r"GPU-[0-9a-f-]+", host["gpu_uuid"]) is not None
            and host.get("gpu_name") == "NVIDIA A100 80GB PCIe"
            and isinstance(host.get("driver_version"), str)
            and re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", host["driver_version"])
            is not None
            and isinstance(host.get("memory_total_mib"), int)
            and not isinstance(host["memory_total_mib"], bool)
            and host["memory_total_mib"] >= 81920
            and host.get("compute_capability") == [8, 0]
            and host.get("torch_cuda_version") == "13.0"
            and isinstance(host.get("platform"), str) and host["platform"],
            f"{label} {regime} admitted host facts differ",
        )
        require(
            isinstance(fingerprint.get("repository_commit"), str)
            and re.fullmatch(r"[0-9a-f]{40}", fingerprint["repository_commit"])
            is not None,
            f"{label} {regime} repository commit differs",
        )
        require(host.get("gpu_uuid") == fingerprint.get("gpu_uuid"),
                f"{label} {regime} GPU UUID binding differs")
        compact[regime] = {
            "host": {
                key: host.get(key) for key in (
                    "gpu_uuid", "gpu_name", "driver_version", "memory_total_mib",
                    "compute_capability", "torch_cuda_version", "platform",
                )
            },
            "runtime": {
                key: fingerprint.get(key) for key in (
                    "model_id", "requested_revision", "resolved_snapshot",
                    "architecture", "attention_backend", "kv_dtype", "eos_ids",
                    "geometry", "dependency_versions", "repository_commit",
                    "weight_runtime", "fingerprint_sha256",
                )
            },
        }
    require(compact["nf4"]["host"] == compact["bf16"]["host"],
            f"{label} regimes did not use one host")
    gate = _mapping(document.get("matched_runtime_gate"), f"{label} matched gate")
    require(gate.get("status") == "PASS", f"{label} matched-runtime gate failed")
    if label == "P01":
        require(gate.get("differing_fields") == []
                and gate.get("nf4_matched_view_sha256") ==
                gate.get("bf16_matched_view_sha256"),
                "P01 matched-runtime declaration differs")
        independent = _mapping(
            document.get("independently_reconstructed_matched_runtime"),
            "P01 independently reconstructed runtime",
        )
        require(independent.get("status") == "PASS"
                and independent.get("differing_json_pointers") == [],
                "P01 independent matched-runtime check differs")
        view_sha = gate["nf4_matched_view_sha256"]
    else:
        require(gate.get("same_host") is True
                and gate.get("differing_json_pointers") == [],
                "P02 matched-runtime declaration differs")
        view_sha = gate.get("view_sha256")
    _sha(view_sha, f"{label} matched-runtime view SHA")
    return {
        "matched_runtime_status": "PASS",
        "matched_runtime_view_sha256": view_sha,
        "regimes": compact,
    }


def _validate_p01(
    document: Mapping[str, Any], artifact: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(document, P01_TOP_KEYS, "P01 root")
    require(document.get("schema") == P01_SCHEMA
            and document.get("protocol_id") == P01_PROTOCOL,
            "P01 schema/protocol differs")
    _validate_formal_boundary(document, "P01")
    _validate_inference(document, "P01")
    _validate_source_bindings(document, "P01")
    _validate_terminal_files(document, "P01")
    require(
        document.get("analysis_status") == "POST_RUN_PARTIAL_DESCRIPTIVE"
        and document.get("post_run_analysis") is True
        and document.get("terminal_partial_run") is True
        and document.get("runner_terminal_status") == "ERROR",
        "P01 documented terminal-partial boundary differs",
    )
    expected_keys = [
        ["bf16", CASE_ID, 1], ["nf4", CASE_ID, 1], ["nf4", CASE_ID, 2]
    ]
    require(document.get("complete_outcome_keys") == expected_keys,
            "P01 complete outcome boundary differs")
    disposition = _mapping(
        document.get("required_bf16_repeat2_disposition"),
        "P01 bfloat16 repeat-2 disposition",
    )
    require(
        disposition.get("key") == ["bf16", CASE_ID, 2]
        and disposition.get("status") == "ERROR"
        and disposition.get("error_type") == "KeyboardInterrupt"
        and disposition.get("compact_present") is False
        and disposition.get("phase_a_present") is False
        and disposition.get("treatment_present") is False
        and disposition.get("render_consistency") == "PASS",
        "P01 bfloat16 repeat-2 partial boundary differs",
    )
    package = _exact_keys(
        disposition.get("package"),
        {"manifest_sha256", "path", "raw_artifact_sha256"},
        "P01 incomplete package",
    )
    require(isinstance(package.get("path"), str) and package["path"],
            "P01 incomplete package path differs")
    extension = _mapping(document.get("extension_decision"), "P01 extension")
    require(extension.get("status") == "PASS"
            and extension.get("decision") == "BASE_E01_ONLY",
            "P01 base-e01-only extension boundary differs")
    _sha(extension.get("sha256"), "P01 extension SHA")
    outcomes = _mapping(document.get("outcomes"), "P01 outcomes")
    expected_outcomes = {"nf4:e01:r1", "nf4:e01:r2", "bf16:e01:r1"}
    require(set(outcomes) == expected_outcomes, "P01 outcome set differs")
    parsed = {
        "nf4:r1": _p01_outcome(outcomes["nf4:e01:r1"], "nf4", 1),
        "nf4:r2": _p01_outcome(outcomes["nf4:e01:r2"], "nf4", 2),
        "bf16:r1": _p01_outcome(outcomes["bf16:e01:r1"], "bf16", 1),
    }
    for key in ("estimands", "generations", "placebo"):
        require(parsed["nf4:r1"][key] == parsed["nf4:r2"][key],
                f"P01 NF4 repeat payload differs in {key}")
    stability = _mapping(document.get("nf4_repeat_stability"),
                         "P01 NF4 repeat stability")
    require(
        stability.get("status") == "PASS" and stability.get("stable") is True
        and stability.get("differing_json_pointers") == []
        and stability.get("differing_field_count") == 0
        and stability.get("repeat1_canonical_sha256") ==
        stability.get("repeat2_canonical_sha256"),
        "P01 NF4 repeat stability differs",
    )
    matched = _mapping(document.get("matched_repeat1_descriptive"),
                       "P01 matched repeat 1")
    for regime, key in (("nf4", "nf4:r1"), ("bf16", "bf16:r1")):
        summary = _mapping(matched.get(regime), f"P01 {regime} matched summary")
        checks = {
            "D_full_KV_N_R2_focal": "N_R2_D_full_KV_focal",
            "D_value_N_R2_focal": "N_R2_D_value_focal",
            "D_value_N_R2_nonfocal": "N_R2_D_value_nonfocal",
            "D_value_P_R2_focal": "P_R2_D_value_focal",
            "N_minus_P_value_D_shift": "N_minus_P_value_shift_focal",
            "oracle_to_fresh_focal_margin_damage": "E1_focal_margin_damage",
            "oracle_to_fresh_nonfocal_margin_damage": "E1_nonfocal_margin_damage",
        }
        require(all(summary.get(source) == parsed[key]["estimands"][target]
                    for source, target in checks.items()),
                f"P01 {regime} matched summary differs")
    runtime = _validate_runtime(document, "P01")
    verification = _mapping(document.get("package_and_derived_verification"),
                            "P01 verification")
    require(set(verification) == expected_outcomes
            and all(row.get("raw_package") == "VERIFIED"
                    and row.get("compact_consistency") == "PASS"
                    and row.get("render_consistency") == "PASS"
                    for row in verification.values()),
            "P01 package/derived verification differs")
    return {
        "input_artifact": dict(artifact),
        "boundary": {
            "status": "POST_RUN_PARTIAL_DESCRIPTIVE",
            "terminal_partial_run": True,
            "runner_terminal_status": "ERROR",
            "complete_outcome_keys": expected_keys,
            "bf16_repeat2": {
                "status": disposition["status"],
                "error_type": disposition["error_type"],
                "phase_a_present": False,
                "treatment_present": False,
            },
            **FORMAL_BOUNDARY,
            "fixed_fixture_exploratory_only": True,
        },
        "runtime_facts": runtime,
        "repeat1": {
            "nf4": {key: parsed["nf4:r1"][key]
                    for key in ("estimands", "generations", "placebo")},
            "bf16": {key: parsed["bf16:r1"][key]
                     for key in ("estimands", "generations", "placebo")},
        },
        "repeat_stability": {
            "nf4": dict(stability),
            "bf16": {
                "status": "NOT_AVAILABLE_DOCUMENTED_TERMINAL_PARTIAL",
                "repeat2_status": "ERROR",
            },
        },
    }


def _validate_p02(
    document: Mapping[str, Any], artifact: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(document, P02_TOP_KEYS, "P02 root")
    require(document.get("schema") == P02_SCHEMA
            and document.get("protocol_id") == P02_PROTOCOL,
            "P02 schema/protocol differs")
    _validate_formal_boundary(document, "P02")
    _validate_inference(document, "P02")
    _validate_source_bindings(document, "P02")
    _validate_terminal_files(document, "P02")
    require(document.get("runner_terminal_status") == "COMPLETE"
            and document.get("primary_dataset") == "matched repeat 1 only",
            "P02 terminal/primary boundary differs")
    outer = _exact_keys(document.get("outer_pod_receipt"), {"receipt", "status"},
                        "P02 outer receipt")
    receipt = _exact_keys(
        outer.get("receipt"),
        {"artifact_count", "expected_commit", "path", "sha256", "terminal_status"},
        "P02 outer receipt binding",
    )
    require(
        outer.get("status") == "PASS" and receipt.get("terminal_status") == "PASS"
        and isinstance(receipt.get("artifact_count"), int)
        and receipt["artifact_count"] > 0
        and isinstance(receipt.get("path"), str) and receipt["path"],
        "P02 outer receipt did not establish hard eligibility",
    )
    _sha(receipt["sha256"], "P02 outer receipt SHA")
    packages = _exact_keys(
        document.get("all_lossless_packages"),
        {"count", "manifest_sha256s", "status"},
        "P02 lossless-package inventory",
    )
    hashes = _sequence(packages.get("manifest_sha256s"), "P02 package hashes")
    require(packages.get("status") == "VERIFIED"
            and packages.get("count") == len(hashes) and len(hashes) > 0
            and list(hashes) == sorted(hashes) and len(hashes) == len(set(hashes)),
            "P02 lossless-package inventory differs")
    continuation = _mapping(document.get("continuation_decisions"),
                            "P02 continuation decisions")
    require(
        continuation.get("status") == "PASS"
        and continuation.get("initial_matched_repeat1_authorized") is True
        and continuation.get("initial_repeat2_authorized") is True
        and continuation.get("bf16_repeat2_authorized") is True
        and document.get("repeat2_rider_status") == "MATCHED_COMPLETE",
        "P02 MATCHED_COMPLETE authorization differs",
    )
    expected_decisions = {
        "initial_schedule": ("MATCHED_R1_AND_BOTH_R2", True),
        "bf16_technical_admission": ("LOAD_BF16", True),
        "bf16_repeat1": ("RUN_BF16_REPEAT1", True),
        "bf16_repeat2": ("RUN_BF16_REPEAT2", True),
    }
    decision_rows = _mapping(continuation.get("documents"), "P02 decisions")
    require(set(decision_rows) == set(expected_decisions),
            "P02 continuation decision set differs")
    require(all(
        decision_rows[name].get("decision") == expected[0]
        and decision_rows[name].get("authorized") is expected[1]
        for name, expected in expected_decisions.items()
    ), "P02 continuation decisions differ")
    outcomes = _mapping(document.get("outcomes"), "P02 outcomes")
    expected_outcomes = {
        "nf4:e01:r1", "nf4:e01:r2", "bf16:e01:r1", "bf16:e01:r2"
    }
    require(set(outcomes) == expected_outcomes, "P02 outcome set differs")
    parsed = {
        f"{regime}:r{repeat}": _p02_outcome(
            outcomes[f"{regime}:e01:r{repeat}"], regime, repeat
        )
        for regime in REGIMES for repeat in (1, 2)
    }
    stability_guard = _mapping(
        document.get("repeat_stability_and_interpretation_guard"),
        "P02 repeat-stability guard",
    )
    require(stability_guard.get("E2_E3_blocked_by_generation_or_hash_instability")
            is False, "P02 E2/E3 was blocked by repeat instability")
    stability_rows = _mapping(stability_guard.get("within_regime_repeat_stability"),
                              "P02 within-regime stability")
    require(set(stability_rows) == set(REGIMES), "P02 repeat regime set differs")
    for regime in REGIMES:
        for key in ("estimands", "generations"):
            require(parsed[f"{regime}:r1"][key] == parsed[f"{regime}:r2"][key],
                    f"P02 {regime} repeat payload differs in {key}")
        row = _mapping(stability_rows[regime], f"P02 {regime} stability")
        deltas = _mapping(row.get("scalar_repeat2_minus_repeat1"),
                          f"P02 {regime} repeat deltas")
        require(
            row.get("status") == "PASS"
            and row.get("differing_json_pointers") == []
            and row.get("differing_field_count") == 0
            and row.get("generation_hashes_and_text_stable") is True
            and row.get("repeat1_sha256") == row.get("repeat2_sha256")
            and set(deltas) == P02_SCALAR_NAMES
            and all(value == 0.0 for value in deltas.values()),
            f"P02 {regime} repeat stability differs",
        )
    matched = _mapping(document.get("matched_repeat1_descriptive"),
                       "P02 matched repeat 1")
    scalar_summary = _mapping(matched.get("E1_E3_and_secondary_scalars"),
                              "P02 matched scalars")
    for regime in REGIMES:
        summary = _mapping(scalar_summary.get(regime), f"P02 {regime} scalars")
        require(set(summary) == P02_SCALAR_NAMES,
                f"P02 {regime} scalar name set differs")
        require(all(summary[source] == parsed[f"{regime}:r1"]["estimands"][target]
                    for source, target in P02_SCALAR_TO_NORMALIZED.items()),
                f"P02 {regime} matched scalar summary differs")
    nf4_minus_bf16 = _mapping(scalar_summary.get("nf4_minus_bf16"),
                              "P02 within-protocol deltas")
    require(set(nf4_minus_bf16) == P02_SCALAR_NAMES,
            "P02 within-protocol scalar set differs")
    require(all(
        nf4_minus_bf16[source] ==
        scalar_summary["nf4"][source] - scalar_summary["bf16"][source]
        for source in P02_SCALAR_NAMES
    ), "P02 stored within-protocol deltas differ")
    e2_summary = _mapping(matched.get("E2_literal_generated_behavior"),
                          "P02 matched E2")
    for probe in PROBES:
        row = _mapping(e2_summary.get(probe), f"P02 matched E2 {probe}")
        for regime in REGIMES:
            generation = parsed[f"{regime}:r1"]["generations"][probe]
            require(row.get(f"{regime}_literal_generation_tuple") == [
                {
                    "decoded_content": decoded,
                    "content_ids": ids,
                    "stop_reason": stop,
                    "cap_hit": cap,
                    "canonical_sha256": sha,
                }
                for decoded, ids, stop, cap, sha in zip(
                    generation["decoded_content"], generation["content_ids"],
                    generation["stop_reasons"], generation["cap_hit"],
                    generation["canonical_sha256"],
                )
            ] and row.get(f"{regime}_generation_hash_tuple") ==
            generation["canonical_sha256"]
            and row.get(f"{regime}_change_vector") == generation["change_vector"],
                    f"P02 matched E2 {probe}/{regime} differs")
        require(row.get("literal_tuples_equal") is (
            parsed["nf4:r1"]["generations"][probe] ==
            parsed["bf16:r1"]["generations"][probe]
        ) and row.get("change_vectors_equal") is (
            parsed["nf4:r1"]["generations"][probe]["change_vector"] ==
            parsed["bf16:r1"]["generations"][probe]["change_vector"]
        ), f"P02 matched E2 equality flag differs: {probe}")
    placebo_summary = _mapping(matched.get("placebo_status"),
                               "P02 matched placebos")
    require(all(placebo_summary.get(regime) == parsed[f"{regime}:r1"]["placebo"]
                or placebo_summary.get(regime) ==
                outcomes[f"{regime}:e01:r1"]["analysis"]["secondary_controls"]["placebo"]
                for regime in REGIMES),
            "P02 matched placebo summary differs")
    runtime = _validate_runtime(document, "P02")
    commits = {
        runtime["regimes"][regime]["runtime"]["repository_commit"]
        for regime in REGIMES
    }
    require(len(commits) == 1 and receipt.get("expected_commit") in commits,
            "P02 outer receipt commit differs from admitted runtime")
    verification = _mapping(document.get("package_and_derived_verification"),
                            "P02 verification")
    require(set(verification) == expected_outcomes
            and all(row.get("raw_package") == "VERIFIED"
                    and row.get("compact_consistency") == "PASS"
                    and row.get("render_consistency") == "PASS"
                    for row in verification.values()),
            "P02 package/derived verification differs")
    return {
        "input_artifact": dict(artifact),
        "boundary": {
            "hard_eligibility": "PASS",
            "runner_terminal_status": "COMPLETE",
            "outer_pod_receipt_status": "PASS",
            "matched_runtime_status": "PASS",
            "repeat2_rider_status": "MATCHED_COMPLETE",
            "primary_dataset": "matched repeat 1 only",
            **FORMAL_BOUNDARY,
            "fixed_fixture_exploratory_only": True,
        },
        "runtime_facts": runtime,
        "repeat1": {
            regime: {key: parsed[f"{regime}:r1"][key]
                     for key in ("estimands", "generations", "placebo")}
            for regime in REGIMES
        },
        "repeat_stability": {regime: dict(stability_rows[regime])
                             for regime in REGIMES},
    }


def _subtract(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    require(set(left) == set(SCALAR_NAMES) and set(right) == set(SCALAR_NAMES),
            "scalar subtraction name set differs")
    return {name: _number(left[name], name) - _number(right[name], name)
            for name in SCALAR_NAMES}


def _generation_equalities(
    left: Mapping[str, Any], right: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for probe in PROBES:
        a = _mapping(left[probe], f"left {probe} generations")
        b = _mapping(right[probe], f"right {probe} generations")
        cellwise = {}
        for index, cell in enumerate(CELLS):
            cellwise[cell] = {
                "decoded_content_equal": a["decoded_content"][index] ==
                b["decoded_content"][index],
                "content_ids_equal": a["content_ids"][index] == b["content_ids"][index],
                "stop_reason_equal": a["stop_reasons"][index] ==
                b["stop_reasons"][index],
                "cap_hit_equal": a["cap_hit"][index] == b["cap_hit"][index],
                "canonical_sha256_equal": a["canonical_sha256"][index] ==
                b["canonical_sha256"][index],
            }
        result[probe] = {
            "five_cell_payload_equal": a == b,
            "decoded_content_tuple_equal": a["decoded_content"] == b["decoded_content"],
            "content_id_tuple_equal": a["content_ids"] == b["content_ids"],
            "stop_reason_tuple_equal": a["stop_reasons"] == b["stop_reasons"],
            "generation_hash_tuple_equal": a["canonical_sha256"] ==
            b["canonical_sha256"],
            "change_vector_equal": a["change_vector"] == b["change_vector"],
            "cellwise": cellwise,
        }
    return result


def _placebo_equalities(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, bool]:
    keys = (
        "status", "scores_present", "movement_from_fresh", "failed_layer_index",
        "failed_row_index", "canonical_diagnostics_sha256", "source_result_hashes",
    )
    return {f"{key}_equal": left.get(key) == right.get(key) for key in keys} | {
        "full_payload_equal": dict(left) == dict(right)
    }


def _runtime_equalities(p01: Mapping[str, Any], p02: Mapping[str, Any]) -> dict[str, Any]:
    host_fields = (
        "gpu_uuid", "gpu_name", "driver_version", "memory_total_mib",
        "compute_capability", "torch_cuda_version", "platform",
    )
    runtime_fields = (
        "model_id", "requested_revision", "resolved_snapshot", "architecture",
        "attention_backend", "kv_dtype", "eos_ids", "geometry",
        "dependency_versions", "repository_commit", "weight_runtime",
        "fingerprint_sha256",
    )
    result = {"regimes": {}}
    for regime in REGIMES:
        a = p01["regimes"][regime]
        b = p02["regimes"][regime]
        result["regimes"][regime] = {
            "host_fields": {field: a["host"].get(field) == b["host"].get(field)
                            for field in host_fields},
            "runtime_fields": {
                field: a["runtime"].get(field) == b["runtime"].get(field)
                for field in runtime_fields
            },
        }
    result["same_gpu_uuid_across_protocols"] = (
        p01["regimes"]["nf4"]["host"]["gpu_uuid"] ==
        p02["regimes"]["nf4"]["host"]["gpu_uuid"]
    )
    result["same_driver_version_across_protocols"] = (
        p01["regimes"]["nf4"]["host"]["driver_version"] ==
        p02["regimes"]["nf4"]["host"]["driver_version"]
    )
    return result


def _a8_evaluation(
    p01: Mapping[str, Any], p02: Mapping[str, Any], equalities: Mapping[str, Any],
) -> dict[str, Any]:
    p02_values = {regime: p02["repeat1"][regime]["estimands"] for regime in REGIMES}
    p01_values = {regime: p01["repeat1"][regime]["estimands"] for regime in REGIMES}
    p02_generations = {
        regime: p02["repeat1"][regime]["generations"] for regime in REGIMES
    }
    p01_generations = {
        regime: p01["repeat1"][regime]["generations"] for regime in REGIMES
    }
    gross = {
        regime: {
            "p01": p01_values[regime]["E1_focal_margin_damage"],
            "p02": p02_values[regime]["E1_focal_margin_damage"],
            "p02_positive": p02_values[regime]["E1_focal_margin_damage"] > 0.0,
            "p02_minus_p01": p02_values[regime]["E1_focal_margin_damage"] -
            p01_values[regime]["E1_focal_margin_damage"],
        }
        for regime in REGIMES
    }
    placebo = {
        regime: {
            "p01": p01["repeat1"][regime]["placebo"],
            "p02": p02["repeat1"][regime]["placebo"],
            "status_equal": p01["repeat1"][regime]["placebo"]["status"] ==
            p02["repeat1"][regime]["placebo"]["status"],
        }
        for regime in REGIMES
    }
    graft_target_checks = {}
    for regime in REGIMES:
        focal = p02_generations[regime]["focal"]
        graft_values = focal["decoded_content"][1:]
        graft_target_checks[regime] = {
            "graft_cell_order": list(CHANGE_CELLS),
            "decoded_content": graft_values,
            "exact_target": "partner beta",
            "all_graft_cells_fail_exact_target": all(
                value != "partner beta" for value in graft_values
            ),
        }
    dkv = {
        regime: {
            "p01": p01_values[regime]["N_R2_D_full_KV_focal"],
            "p02": p02_values[regime]["N_R2_D_full_KV_focal"],
            "exactly_equal": p01_values[regime]["N_R2_D_full_KV_focal"] ==
            p02_values[regime]["N_R2_D_full_KV_focal"],
        }
        for regime in REGIMES
    }
    rules = [
        {
            "id": "A8-new-admitted-host",
            "evaluation": "OBSERVED",
            "evidence": {
                "p01_gpu_uuid": p01["runtime_facts"]["regimes"]["nf4"]["host"]["gpu_uuid"],
                "p02_gpu_uuid": p02["runtime_facts"]["regimes"]["nf4"]["host"]["gpu_uuid"],
                "gpu_uuid_differs": not equalities["runtime"]["same_gpu_uuid_across_protocols"],
                "p02_hard_eligibility": p02["boundary"]["hard_eligibility"],
            },
        },
        {
            "id": "A8-gross-focal-margin-damage",
            "evaluation": "SIGN_EVALUATED_MAGNITUDE_NOT_THRESHOLD_CLASSIFIED",
            "evidence": {
                "regimes": gross,
                "all_p02_values_positive": all(row["p02_positive"] for row in gross.values()),
                "large_magnitude_classification": "NOT_EVALUATED_NO_PREDECLARED_THRESHOLD",
            },
        },
        {
            "id": "A8-placebo-remains-unavailable",
            "evaluation": "OBSERVED_MISSING_CONTROL",
            "evidence": placebo,
        },
        {
            "id": "A8-graft-cells-fail-exact-target",
            "evaluation": "EXACT_STRING_CHECK",
            "evidence": graft_target_checks,
        },
        {
            "id": "A8-exact-five-cell-literal-tuples",
            "evaluation": "EXACT_PAYLOAD_CHECK",
            "evidence": {
                regime: equalities["generations_by_regime"][regime]
                for regime in REGIMES
            },
        },
        {
            "id": "A8-nf4-fresh-anomaly",
            "evaluation": "EXACT_PAYLOAD_CHECK",
            "evidence": {
                "p01_nf4_focal_FF": {
                    key: p01_generations["nf4"]["focal"][key][0]
                    for key in ("decoded_content", "content_ids", "stop_reasons",
                                "canonical_sha256")
                },
                "p02_nf4_focal_FF": {
                    key: p02_generations["nf4"]["focal"][key][0]
                    for key in ("decoded_content", "content_ids", "stop_reasons",
                                "canonical_sha256")
                },
                "exactly_equal": all(
                    equalities["generations_by_regime"]["nf4"]["focal"]["cellwise"]["FF"].values()
                ),
            },
        },
        {
            "id": "A8-bf16-literal-and-repeat-stability",
            "evaluation": "OBSERVED",
            "evidence": {
                "cross_protocol_five_cell_payload_equal":
                    equalities["generations_by_regime"]["bf16"]["focal"][
                        "five_cell_payload_equal"
                    ],
                "p02_repeat_status": p02["repeat_stability"]["bf16"]["status"],
                "p02_repeat_generation_stable":
                    p02["repeat_stability"]["bf16"]["generation_hashes_and_text_stable"],
            },
        },
        {
            "id": "A8-D-full-KV-sign-split",
            "evaluation": "SIGN_AND_EXACT_VALUE_CHECK_ONLY",
            "evidence": {
                "regimes": dkv,
                "p02_nf4_negative_bf16_positive":
                    dkv["nf4"]["p02"] < 0.0 and dkv["bf16"]["p02"] > 0.0,
                "quantization_dependence_claim_authorized": False,
            },
        },
        {
            "id": "A8-E3-scalar-reading",
            "evaluation": "DESCRIPTIVE_EXACT_COMPARISON_ONLY",
            "evidence": {
                "estimand_equalities": equalities["estimands_by_regime"],
                "thresholds_added": False,
                "ratios_added": False,
                "p_values_added": False,
                "confidence_intervals_added": False,
            },
        },
        {
            "id": "A8-repeat2-governs-stability",
            "evaluation": "OBSERVED_MATCHED_COMPLETE",
            "evidence": p02["repeat_stability"],
        },
        {
            "id": "A8-formal-and-claim-boundaries-remain-closed",
            "evaluation": "OBSERVED",
            "evidence": {
                **FORMAL_BOUNDARY,
                "quantization_dependence_claim_authorized": False,
                "efficacy_claim_authorized": False,
                "population_claim_authorized": False,
            },
        },
        {
            "id": "A8-technical-failure-conditional",
            "evaluation": "CONDITION_NOT_TRIGGERED",
            "evidence": {
                "p02_hard_eligibility": p02["boundary"]["hard_eligibility"],
                "matched_repeat1_present": True,
                "repeat2_rider_status": p02["boundary"]["repeat2_rider_status"],
            },
        },
        {
            "id": "A8-no-value-driven-grid-extension",
            "evaluation": "ANALYZER_OUTPUT_GRID_CHECKED_BROADER_OPERATIONS_NOT_EVALUABLE",
            "evidence": {
                "p02_case_ids": [CASE_ID],
                "p02_regimes": list(REGIMES),
                "p02_repeats": [1, 2],
                "repeat2_rider_status": "MATCHED_COMPLETE",
                "no_same_day_P03": "NOT_EVALUABLE_FROM_P01_P02_ANALYZER_OUTPUTS",
                "no_result_driven_dollar_or_minute_extension":
                    "NOT_EVALUABLE_FROM_P01_P02_ANALYZER_OUTPUTS",
            },
        },
    ]
    return {
        "source": (
            "notes/20260712A8-sol-p02-conditional-replication-expectations-"
            "and-launch-decision.md"
        ),
        "expectation_count": len(rules),
        "all_expectations_explicitly_addressed": True,
        "evaluations": rules,
    }


def compare_documents(
    p01_document: Mapping[str, Any],
    p02_document: Mapping[str, Any],
    p01_artifact: Mapping[str, Any],
    p02_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    p01_binding = _validate_binding(p01_artifact, "P01 input artifact")
    p02_binding = _validate_binding(p02_artifact, "P02 input artifact")
    _validate_sha_fields(p01_document, "P01")
    _validate_sha_fields(p02_document, "P02")
    p01 = _validate_p01(p01_document, p01_binding)
    p02 = _validate_p02(p02_document, p02_binding)

    within = {
        "p01_nf4_minus_bf16": _subtract(
            p01["repeat1"]["nf4"]["estimands"],
            p01["repeat1"]["bf16"]["estimands"],
        ),
        "p02_nf4_minus_bf16": _subtract(
            p02["repeat1"]["nf4"]["estimands"],
            p02["repeat1"]["bf16"]["estimands"],
        ),
    }
    p02_minus_p01 = {
        "by_regime": {
            regime: _subtract(
                p02["repeat1"][regime]["estimands"],
                p01["repeat1"][regime]["estimands"],
            ) for regime in REGIMES
        },
        "within_protocol_delta_difference": _subtract(
            within["p02_nf4_minus_bf16"], within["p01_nf4_minus_bf16"]
        ),
    }
    estimand_equalities = {
        regime: {
            name: p01["repeat1"][regime]["estimands"][name] ==
            p02["repeat1"][regime]["estimands"][name]
            for name in SCALAR_NAMES
        } for regime in REGIMES
    }
    generation_equalities = {
        regime: _generation_equalities(
            p01["repeat1"][regime]["generations"],
            p02["repeat1"][regime]["generations"],
        ) for regime in REGIMES
    }
    placebo_equalities = {
        regime: _placebo_equalities(
            p01["repeat1"][regime]["placebo"],
            p02["repeat1"][regime]["placebo"],
        ) for regime in REGIMES
    }
    equalities = {
        "estimands_by_regime": estimand_equalities,
        "within_protocol_deltas": {
            name: within["p01_nf4_minus_bf16"][name] ==
            within["p02_nf4_minus_bf16"][name]
            for name in SCALAR_NAMES
        },
        "generations_by_regime": generation_equalities,
        "placebos_by_regime": placebo_equalities,
        "runtime": _runtime_equalities(
            p01["runtime_facts"], p02["runtime_facts"]),
    }
    equalities["all_compared_estimands_equal"] = all(
        value for regime in estimand_equalities.values() for value in regime.values()
    )
    equalities["all_five_cell_generation_payloads_equal"] = all(
        generation_equalities[regime][probe]["five_cell_payload_equal"]
        for regime in REGIMES for probe in PROBES
    )
    equalities["all_placebo_payloads_equal"] = all(
        placebo_equalities[regime]["full_payload_equal"] for regime in REGIMES
    )

    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_status": "PASS",
        "comparison_basis": "matched repeat 1; exact fixed-fixture description",
        "p01": p01,
        "p02": p02,
        "within_protocol_deltas": within,
        "p02_minus_p01": p02_minus_p01,
        "exact_equalities": equalities,
        "frozen_A8_expectations": _a8_evaluation(p01, p02, equalities),
        "inference_boundary": {
            "thresholds_added": False,
            "ratios_computed": False,
            "p_values_computed": False,
            "confidence_intervals_computed": False,
            "equivalence_test_computed": False,
            "quantization_dependence_claim_authorized": False,
            "semantic_transfer_claim_authorized": False,
            "efficacy_claim_authorized": False,
            "population_claim_authorized": False,
            "formal_v12_decision_eligible": False,
            "v12_reentry_authorized": False,
            "note": (
                "Exact one-fixture, cross-host conditional-replication comparison; "
                "no causal, semantic, efficacy, quantization, or population inference."
            ),
        },
    }


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
               + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p01", required=True, type=Path)
    parser.add_argument("--p02", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    p01, p01_binding = load_artifact(args.p01, "P01 analysis artifact")
    p02, p02_binding = load_artifact(args.p02, "P02 analysis artifact")
    result = compare_documents(p01, p02, p01_binding, p02_binding)
    write_exclusive(args.output, result)
    print(
        f"PASS p01={p01_binding['sha256']} p02={p02_binding['sha256']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as exc:
        print(f"FAIL: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)
