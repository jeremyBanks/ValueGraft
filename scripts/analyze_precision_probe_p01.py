#!/usr/bin/env python3
"""Independently reconstruct and analyze precision-probe p01 outcomes.

The paid runner persists each outcome as a lossless gzip/base64 package.  This
script reads those packages rather than trusting runner-side summaries, checks
the complete frozen arm grid, recomputes the descriptive p01 estimands, checks
within-runtime repeat stability, and binds every file in the run directory.

P01 is an exploratory fixed-fixture screen.  This script deliberately computes
no p-values, confidence intervals, v12 stopping rule, or formal release label.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "precision-probe-p01"
ANALYSIS_SCHEMA = "precision_probe_p01_independent_analysis_v1"
PACKAGE_SCHEMA = "lossless-json-gzip-base64-package/v1"
OUTCOME_RAW_SCHEMA = "precision_probe_p01_outcome_raw_v1"
TECHNICAL_RAW_SCHEMA = "precision_probe_p01_technical_raw_v1"
RUN_MANIFEST_SCHEMA = "precision_probe_p01_run_manifest_v1"
COMPLETION_SCHEMA = "precision_probe_p01_completion_v1"
DECISION_SCHEMA = "precision_probe_p01_outcome_blind_extension_decision_v1"
PHASE_A_SCHEMA = "coherent_state_decision_canary_v12_phase_a_raw_v1"
TREATMENT_SCHEMA = "coherent_state_decision_canary_v12_treatment_raw_v1"
REGIMES = ("nf4", "bf16")
REGIONS = ("R1_content", "R2_boundary", "R3_anchor")
PRIMARY_CELLS = ("FF", "FC", "FW", "CF", "CC", "CW", "WF", "WC", "WW")
P_CELLS = ("CC", "WW", "FC", "FW")
EXPECTED_PRIMARY = {
    *(("N", region, cell) for region in REGIONS for cell in PRIMARY_CELLS),
    *(("P", "R2_boundary", cell) for cell in P_CELLS),
}
BASE_KEYS = {
    (regime, "e01", repeat)
    for regime in REGIMES for repeat in (1, 2)
}
EXTENSION_KEYS = {
    (regime, case_id, 1)
    for regime in REGIMES for case_id in ("e02", "e03")
}


class AnalysisError(RuntimeError):
    """A p01 package, score, grid, binding, or repeat differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except Exception as exc:
        raise AnalysisError(f"cannot parse {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def decode_float32_bits(value: Any, label: str) -> float:
    require(
        isinstance(value, str)
        and len(value) == 8
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not lowercase little-endian float32 bits",
    )
    result = struct.unpack("<f", bytes.fromhex(value))[0]
    require(math.isfinite(result), f"{label} is nonfinite")
    return float(result)


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def reconstruct_package(package_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a historical lossless package and return its exact JSON object."""
    manifest_path = package_dir / "manifest.json"
    manifest = load_object(manifest_path)
    require(manifest.get("schema") == PACKAGE_SCHEMA,
            f"package schema differs: {package_dir}")
    chunks = manifest.get("chunks")
    require(isinstance(chunks, list) and chunks,
            f"package chunk list differs: {package_dir}")
    compressed_parts: list[bytes] = []
    offset = 0
    for expected_index, descriptor in enumerate(chunks, start=1):
        require(isinstance(descriptor, Mapping), "package descriptor is not an object")
        require(descriptor.get("index") == expected_index,
                "package chunk indices are not contiguous")
        name = descriptor.get("name")
        require(isinstance(name, str) and Path(name).name == name,
                "package chunk name is unsafe")
        path = package_dir / name
        data = path.read_bytes()
        require(len(data) == descriptor.get("file_size_bytes") and
                sha256_bytes(data) == descriptor.get("file_sha256"),
                f"package chunk file differs: {path}")
        chunk = json.loads(data)
        require(isinstance(chunk, dict) and
                chunk.get("payload_encoding") == "base64" and
                chunk.get("index") == expected_index and
                chunk.get("name") == name and
                chunk.get("compressed_offset_bytes") == offset and
                chunk.get("compressed_segment_size_bytes") ==
                descriptor.get("compressed_segment_size_bytes") and
                chunk.get("compressed_segment_sha256") ==
                descriptor.get("compressed_segment_sha256"),
                f"package chunk wrapper differs: {path}")
        try:
            segment = base64.b64decode(chunk["payload"], validate=True)
        except Exception as exc:
            raise AnalysisError(f"invalid package base64: {path}") from exc
        require(len(segment) == descriptor.get("compressed_segment_size_bytes") and
                sha256_bytes(segment) ==
                descriptor.get("compressed_segment_sha256"),
                f"package compressed segment differs: {path}")
        compressed_parts.append(segment)
        offset += len(segment)
    compressed = b"".join(compressed_parts)
    compression = manifest.get("compression")
    original = manifest.get("original")
    require(isinstance(compression, Mapping) and
            len(compressed) == compression.get("compressed_size_bytes") and
            sha256_bytes(compressed) == compression.get("compressed_sha256"),
            f"compressed package commitment differs: {package_dir}")
    try:
        raw_bytes = gzip.decompress(compressed)
    except Exception as exc:
        raise AnalysisError(f"cannot decompress package: {package_dir}") from exc
    require(isinstance(original, Mapping) and
            len(raw_bytes) == original.get("size_bytes") and
            sha256_bytes(raw_bytes) == original.get("sha256"),
            f"raw package commitment differs: {package_dir}")
    try:
        raw = json.loads(raw_bytes)
    except Exception as exc:
        raise AnalysisError(f"packaged raw JSON is invalid: {package_dir}") from exc
    require(isinstance(raw, dict), "packaged raw JSON root is not an object")
    return manifest, raw


def target_summary(record: Any, label: str) -> dict[str, Any]:
    require(isinstance(record, Mapping), f"{label} target record is absent")
    token_ids = record.get("target_token_ids")
    token_bits = record.get("token_logprob_float32_bits")
    require(isinstance(token_ids, list) and token_ids and
            all(isinstance(value, int) and not isinstance(value, bool)
                for value in token_ids),
            f"{label} target IDs differ")
    require(isinstance(token_bits, list) and len(token_bits) == len(token_ids),
            f"{label} token-logprob coverage differs")
    token_values = [
        decode_float32_bits(value, f"{label}.token[{index}]")
        for index, value in enumerate(token_bits)
    ]
    mean = decode_float32_bits(record.get("mean_logprob_float32_bits"),
                               f"{label}.mean")
    numeric = record.get("mean_logprob")
    require(isinstance(numeric, (int, float)) and not isinstance(numeric, bool) and
            math.isfinite(float(numeric)) and float32(float(numeric)) == mean,
            f"{label} numeric mean differs from its float32 commitment")
    return {
        "target_token_ids": list(token_ids),
        "token_logprobs": token_values,
        "token_logprob_float32_bits": list(token_bits),
        "mean_logprob": mean,
        "mean_logprob_float32_bits": record["mean_logprob_float32_bits"],
    }


def score_summary(record: Any, label: str) -> dict[str, Any]:
    require(isinstance(record, Mapping), f"{label} score record is absent")
    correct = target_summary(record.get("correct"), f"{label}.correct")
    counter = target_summary(record.get("counterfactual"),
                             f"{label}.counterfactual")
    margin = decode_float32_bits(record.get("margin_float32_bits"),
                                 f"{label}.margin")
    recomputed = float32(correct["mean_logprob"] - counter["mean_logprob"])
    require(margin == recomputed,
            f"{label} margin differs from float32 mean subtraction")
    generation = record.get("generation")
    require(isinstance(generation, Mapping) and
            isinstance(generation.get("decoded_content"), str) and
            isinstance(generation.get("content_ids"), list),
            f"{label} generation record differs")
    return {
        "probe": record.get("probe"),
        "correct_text": record.get("correct_text"),
        "counterfactual_text": record.get("counterfactual_text"),
        "correct": correct,
        "counterfactual": counter,
        "margin": margin,
        "margin_float32_bits": record["margin_float32_bits"],
        "generation": {
            "decoded_content": generation["decoded_content"],
            "content_ids": list(generation["content_ids"]),
            "stop_reason": generation.get("stop_reason"),
            "cap_hit": generation.get("cap_hit"),
        },
    }


def score_pair(record: Any, label: str) -> dict[str, Any]:
    require(isinstance(record, Mapping) and set(record) == {"focal", "nonfocal"},
            f"{label} does not contain exactly focal/nonfocal")
    return {
        probe: score_summary(record[probe], f"{label}.{probe}")
        for probe in ("focal", "nonfocal")
    }


def token_contrast(left: Mapping[str, Any], right: Mapping[str, Any],
                   *, recorded_margin_difference: float) -> dict[str, Any]:
    """Decompose a margin contrast without pretending target tokens are aligned."""
    left_correct = left["correct"]
    right_correct = right["correct"]
    left_counter = left["counterfactual"]
    right_counter = right["counterfactual"]
    require(left_correct["target_token_ids"] == right_correct["target_token_ids"],
            "correct target IDs differ across contrast cells")
    require(left_counter["target_token_ids"] == right_counter["target_token_ids"],
            "counterfactual target IDs differ across contrast cells")
    correct_delta = [a - b for a, b in zip(
        left_correct["token_logprobs"], right_correct["token_logprobs"])]
    counter_delta = [a - b for a, b in zip(
        left_counter["token_logprobs"], right_counter["token_logprobs"])]
    correct_contribution = [value / len(correct_delta) for value in correct_delta]
    counter_contribution = [-value / len(counter_delta) for value in counter_delta]
    reconstructed = sum(correct_contribution) + sum(counter_contribution)
    return {
        "correct_target_token_ids": left_correct["target_token_ids"],
        "counterfactual_target_token_ids": left_counter["target_token_ids"],
        "correct_target_token_logprob_differences": correct_delta,
        "counterfactual_target_token_logprob_differences": counter_delta,
        "correct_target_per_token_mean_contributions": correct_contribution,
        "counterfactual_target_per_token_mean_contributions": counter_contribution,
        "first_target_choice": {
            "correct_token_id": left_correct["target_token_ids"][0],
            "counterfactual_token_id": left_counter["target_token_ids"][0],
            "correct_logprob_difference": correct_delta[0],
            "counterfactual_logprob_difference": counter_delta[0],
            "contribution_to_reconstructed_margin_difference":
                correct_contribution[0] + counter_contribution[0],
        },
        "token_reconstructed_margin_difference": reconstructed,
        "recorded_float32_margin_difference": recorded_margin_difference,
        "reconstruction_minus_recorded": reconstructed - recorded_margin_difference,
        "note": (
            "Correct and counterfactual target tokens are separate alternatives. "
            "Their per-token mean contributions are reported separately; no "
            "cross-target token alignment is assumed."
        ),
    }


def contrast(cells: Mapping[str, Mapping[str, Any]], *, correct_cell: str,
             wrong_cell: str, fresh: Mapping[str, Any]) -> dict[str, Any]:
    correct = cells[correct_cell]
    wrong = cells[wrong_cell]
    d_focal = correct["focal"]["margin"] - wrong["focal"]["margin"]
    d_nonfocal = correct["nonfocal"]["margin"] - wrong["nonfocal"]["margin"]
    result = {
        "contrast": f"{correct_cell}-{wrong_cell}",
        "D_focal": d_focal,
        "D_nonfocal": d_nonfocal,
        # The p01 text writes the signed formula.  Keep the historical v12
        # absolute-penalty convention beside it so neither gets silently conflated.
        "SEL_signed_p01": d_focal - d_nonfocal,
        "SEL_absolute_nonfocal_penalty_historical": d_focal - abs(d_nonfocal),
        "Hplus": (correct["focal"]["correct"]["mean_logprob"] -
                  wrong["focal"]["correct"]["mean_logprob"]),
        "counterfactual_target_movement": (
            correct["focal"]["counterfactual"]["mean_logprob"] -
            wrong["focal"]["counterfactual"]["mean_logprob"]),
        "U": correct["focal"]["margin"] - fresh["focal"]["margin"],
        "Uplus": (correct["focal"]["correct"]["mean_logprob"] -
                  fresh["focal"]["correct"]["mean_logprob"]),
        "correct_cell_focal": correct["focal"],
        "wrong_cell_focal": wrong["focal"],
        "correct_cell_nonfocal": correct["nonfocal"],
        "wrong_cell_nonfocal": wrong["nonfocal"],
    }
    result["focal_token_decomposition"] = token_contrast(
        correct["focal"], wrong["focal"],
        recorded_margin_difference=d_focal)
    result["nonfocal_token_decomposition"] = token_contrast(
        correct["nonfocal"], wrong["nonfocal"],
        recorded_margin_difference=d_nonfocal)
    return result


def movement_from_fresh(scores: Mapping[str, Any],
                        fresh: Mapping[str, Any]) -> dict[str, Any]:
    return {
        probe: {
            "margin_delta": scores[probe]["margin"] - fresh[probe]["margin"],
            "correct_target_logprob_delta": (
                scores[probe]["correct"]["mean_logprob"] -
                fresh[probe]["correct"]["mean_logprob"]),
            "generation": scores[probe]["generation"],
        }
        for probe in ("focal", "nonfocal")
    }


def analyze_outcome(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and summarize one reconstructed runtime/case/repeat outcome."""
    phase = raw.get("phase_a")
    treatment = raw.get("treatment")
    require(isinstance(phase, Mapping) and phase.get("schema") == PHASE_A_SCHEMA,
            "nested Phase-A payload differs")
    require(isinstance(treatment, Mapping) and
            treatment.get("schema") == TREATMENT_SCHEMA,
            "nested treatment payload differs")
    case_id = raw.get("case_id")
    require(isinstance(case_id, str) and
            phase.get("case_id") == case_id == treatment.get("case_id"),
            "outcome/Phase-A/treatment case IDs differ")

    phase_scores_raw = phase.get("scores")
    require(isinstance(phase_scores_raw, Mapping), "Phase-A scores are absent")
    phase_scores = {
        key: score_summary(value, f"phase_a.scores.{key}")
        for key, value in phase_scores_raw.items()
    }
    required_phase = {
        f"{prefix}_{probe}"
        for prefix in ("A_C", "A_W", "FF")
        for probe in ("focal", "nonfocal")
    }
    require(set(phase_scores) == required_phase,
            "Phase-A score selector coverage differs")

    arms = treatment.get("arms")
    require(isinstance(arms, list) and len(arms) == 34 and
            treatment.get("primary_arm_count") == 31 and
            treatment.get("placebo_control_count") == 3,
            "treatment does not contain 31 primary plus three placebo arms")
    primary: dict[tuple[str, str, str], dict[str, Any]] = {}
    placebo: dict[str, Mapping[str, Any]] = {}
    for row in arms:
        require(isinstance(row, Mapping), "treatment arm is not an object")
        selector = (row.get("schedule"), row.get("region"), row.get("cell"))
        if selector in EXPECTED_PRIMARY:
            require(row.get("arm_kind") == "primary" and selector not in primary,
                    f"primary arm kind/duplication differs: {selector}")
            primary[selector] = score_pair(row.get("scores"), f"arm{selector}")
        elif (row.get("arm_kind") == "placebo_control" and
              row.get("schedule") == "N" and row.get("region") in REGIONS and
              row.get("cell") == "V_PLACEBO"):
            region = str(row["region"])
            require(region not in placebo, f"duplicate placebo arm: {region}")
            placebo[region] = row
        else:
            raise AnalysisError(f"unexpected treatment arm selector: {selector}")
    require(set(primary) == EXPECTED_PRIMARY and set(placebo) == set(REGIONS),
            "treatment selector coverage differs")
    fresh = score_pair(treatment.get("fresh_scores"), "treatment.fresh_scores")
    for region in REGIONS:
        require(primary[("N", region, "FF")] == fresh,
                f"shared fresh baseline differs at {region}")

    n_estimands: dict[str, Any] = {}
    generations: list[dict[str, Any]] = []
    for region in REGIONS:
        cells = {cell: primary[("N", region, cell)] for cell in PRIMARY_CELLS}
        n_estimands[region] = {
            "full_KV": contrast(cells, correct_cell="CC", wrong_cell="WW",
                                fresh=fresh),
            "value_only": contrast(cells, correct_cell="FC", wrong_cell="FW",
                                   fresh=fresh),
            "key_only": contrast(cells, correct_cell="CF", wrong_cell="WF",
                                 fresh=fresh),
            "crossed_source_CW_minus_WC": contrast(
                cells, correct_cell="CW", wrong_cell="WC", fresh=fresh),
        }
    p_cells = {cell: primary[("P", "R2_boundary", cell)] for cell in P_CELLS}
    p_estimands = {
        "full_KV": contrast(p_cells, correct_cell="CC", wrong_cell="WW",
                            fresh=fresh),
        "value_only": contrast(p_cells, correct_cell="FC", wrong_cell="FW",
                               fresh=fresh),
    }
    for (schedule, region, cell), scores in sorted(primary.items()):
        for probe in ("focal", "nonfocal"):
            generations.append({
                "schedule": schedule, "region": region, "cell": cell,
                "probe": probe, **scores[probe]["generation"],
            })

    placebo_results: dict[str, Any] = {}
    for region in REGIONS:
        row = placebo[region]
        status = row.get("control_status")
        require(status in {"AVAILABLE", "PLACEBO_UNAVAILABLE"} and
                isinstance(row.get("diagnostics"), Mapping) and
                row["diagnostics"].get("status") == status,
                f"placebo status/diagnostics differ at {region}")
        if status == "AVAILABLE":
            scores = score_pair(row.get("scores"), f"placebo.{region}")
            placebo_results[region] = {
                "status": status,
                "movement_from_fresh": movement_from_fresh(scores, fresh),
                "scores": scores,
                "diagnostics": dict(row["diagnostics"]),
            }
        else:
            require("scores" not in row,
                    f"unavailable placebo unexpectedly has scores at {region}")
            placebo_results[region] = {
                "status": status, "movement_from_fresh": None,
                "scores": None, "diagnostics": dict(row["diagnostics"]),
            }

    damage: dict[str, Any] = {}
    for probe in ("focal", "nonfocal"):
        oracle = phase_scores[f"A_C_{probe}"]
        compact = phase_scores[f"FF_{probe}"]
        damage[probe] = {
            "margin": oracle["margin"] - compact["margin"],
            "correct_target_logprob": (
                oracle["correct"]["mean_logprob"] -
                compact["correct"]["mean_logprob"]),
            "oracle_generation": oracle["generation"],
            "fresh_generation": compact["generation"],
        }

    primary_value = n_estimands["R2_boundary"]["value_only"]
    return {
        "case_id": case_id,
        "phase_a_oracle_to_fresh_damage": damage,
        "N_regional_estimands": n_estimands,
        "P_R2_estimands": p_estimands,
        "primary_D_value_N_R2": primary_value["D_focal"],
        "primary_nonfocal_D_value_N_R2": primary_value["D_nonfocal"],
        "primary_SEL_signed_p01": primary_value["SEL_signed_p01"],
        "primary_Hplus_value_N_R2": primary_value["Hplus"],
        "N_minus_P_value_D_shift": (
            primary_value["D_focal"] - p_estimands["value_only"]["D_focal"]),
        "placebo_controls": placebo_results,
        "available_placebo_control_count": sum(
            row["status"] == "AVAILABLE" for row in placebo_results.values()),
        "all_primary_generations": generations,
    }


def identify_outcome(raw: Mapping[str, Any]) -> tuple[str, str, int]:
    require(raw.get("schema") == OUTCOME_RAW_SCHEMA and
            raw.get("protocol_id") == PROTOCOL_ID,
            "packaged outcome schema/protocol differs")
    require(raw.get("formal_v12_decision_eligible") is False and
            raw.get("v12_reentry_authorized") is False and
            raw.get("component_reuse_does_not_inherit_v12_eligibility") is True and
            raw.get("semantic_evidence_eligible") is False,
            "packaged outcome v12 ineligibility labels differ")
    regime = raw.get("regime")
    case_id = raw.get("case_id")
    repeat = raw.get("repeat_index", raw.get("repeat"))
    require(regime in REGIMES, "packaged outcome regime differs")
    require(case_id in {"e01", "e02", "e03"},
            "packaged outcome case differs")
    require(isinstance(repeat, int) and not isinstance(repeat, bool) and repeat >= 1,
            "packaged outcome repeat differs")
    return str(regime), str(case_id), int(repeat)


def packaged_p01_documents(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(run_dir.rglob("manifest.json")):
        try:
            header = load_object(manifest_path)
        except AnalysisError:
            continue
        if header.get("schema") != PACKAGE_SCHEMA:
            continue
        manifest, raw = reconstruct_package(manifest_path.parent)
        if raw.get("protocol_id") != PROTOCOL_ID:
            continue
        rows.append({
            "package_dir": manifest_path.parent,
            "package_manifest_sha256": file_sha256(manifest_path),
            "raw_artifact_sha256": manifest["original"]["sha256"],
            "raw": raw,
        })
    require(rows, f"no p01 lossless packages found under {run_dir}")
    return rows


def package_rows(
    run_dir: Path,
    documents: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = documents if documents is not None else packaged_p01_documents(run_dir)
    for generic in source:
        raw = generic["raw"]
        # The run also packages one technical-gate document per regime.  Only
        # outcome packages carry case/repeat keys and enter the estimands.
        if (raw.get("schema") != OUTCOME_RAW_SCHEMA or
                raw.get("status") != "COMPLETE"):
            continue
        key = identify_outcome(raw)
        rows.append({**dict(generic), "key": key})
    require(rows, f"no p01 lossless outcome packages found under {run_dir}")
    return rows


def _require_formal_boundary(document: Mapping[str, Any], label: str) -> None:
    require(document.get("protocol_id") == PROTOCOL_ID and
            document.get("formal_v12_decision_eligible") is False and
            document.get("v12_reentry_authorized") is False and
            document.get("component_reuse_does_not_inherit_v12_eligibility")
            is True and document.get("semantic_evidence_eligible") is False,
            f"{label} protocol/formal boundary differs")


def _technical_gate_summary(
    run_dir: Path, documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in documents
            if row["raw"].get("schema") == TECHNICAL_RAW_SCHEMA and
            row["raw"].get("status") == "PASS"]
    require(len(rows) == 2, "p01 requires exactly two passing final technical packages")
    by_regime: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw = row["raw"]
        _require_formal_boundary(raw, "technical package")
        regime = raw.get("regime")
        require(regime in REGIMES and regime not in by_regime,
                "technical package regime coverage differs")
        subject = raw.get("subject_attestation")
        runtime = raw.get("runtime_fingerprint")
        require(isinstance(subject, Mapping) and
                isinstance(subject.get("bindings"), Mapping) and
                isinstance(subject.get("runtime_fingerprint"), Mapping) and
                subject["runtime_fingerprint"] == runtime,
                f"{regime} technical subject/runtime binding differs")
        bindings = subject["bindings"]
        require(bindings.get("regime") == regime and
                bindings.get("protocol_id") == PROTOCOL_ID,
                f"{regime} loader binding differs")
        require(isinstance(runtime, Mapping) and
                runtime.get("regime") == regime and
                runtime.get("model_id") ==
                "Qwen/Qwen3-30B-A3B-Instruct-2507" and
                runtime.get("requested_revision") ==
                "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe" and
                runtime.get("attention_backend") == "eager" and
                runtime.get("kv_dtype") == "torch.bfloat16" and
                runtime.get("eos_ids") == [151643, 151645],
                f"{regime} runtime fingerprint differs")

        g0 = bindings.get("g0")
        weights = bindings.get("weights")
        kv = bindings.get("kv")
        host = bindings.get("host")
        loading = bindings.get("loading_info")
        require(isinstance(g0, Mapping) and g0.get("passes") is True and
                g0.get("topology", {}).get("checkpoint_key_count") == 18_867 and
                g0.get("topology", {}).get("eligible_linear_count") == 18_672 and
                g0.get("topology", {}).get("eligible_logical_elements") ==
                29_909_581_824,
                f"{regime} G0 topology gate differs")
        require(isinstance(kv, Mapping) and
                kv.get("observed_real_forward") is True and
                kv.get("layers") == 48 and
                kv.get("dtype") == "torch.bfloat16" and
                kv.get("expected_shape", [None, None, None, None])[:2] == [1, 4] and
                kv.get("expected_shape", [None, None, None, None])[-1:] == [128],
                f"{regime} real-forward KV gate differs")
        require(isinstance(host, Mapping) and
                host.get("gpu_name") == "NVIDIA A100 80GB PCIe" and
                isinstance(host.get("gpu_uuid"), str),
                f"{regime} host binding differs")
        require(isinstance(loading, Mapping) and loading and
                all(value == [] for value in loading.values()),
                f"{regime} model loading-info gate differs")
        require(isinstance(weights, Mapping), f"{regime} weight gate is absent")
        if regime == "nf4":
            sentinels = weights.get("sentinels")
            require(weights.get("linear4bit_modules") == 18_672 and
                    weights.get("expert_linear4bit_modules") == 18_432 and
                    weights.get("expert_logical_coverage") == {
                        "observed": 28_991_029_248,
                        "expected": 28_991_029_248,
                    } and
                    weights.get("total_logical_quantized_coverage") == {
                        "observed": 29_909_581_824,
                        "expected": 29_909_581_824,
                        "checkpoint_total": 30_532_122_624,
                    } and weights.get("quant_type") == "nf4" and
                    weights.get("double_quantization") is True and
                    weights.get("compute_dtype") == "torch.bfloat16" and
                    weights.get("ordinary_linears") == ["lm_head"] and
                    isinstance(sentinels, list) and len(sentinels) == 9 and
                    all(float(item.get("max_abs_error", 0)) > 0 and
                        float(item.get("mean_abs_error", 0)) > 0
                        for item in sentinels),
                    "NF4 full-coverage/sentinel gate differs")
        else:
            require(weights.get("regime") == "bf16" and
                    weights.get("logical_parameter_elements") ==
                    30_532_122_624 and
                    weights.get("floating_parameter_dtype") ==
                    "torch.bfloat16" and
                    weights.get("quantization_modules") == [],
                    "bf16 full-precision weight gate differs")

        identity = raw.get("generated_forced_identity")
        deterministic = raw.get("deterministic_repeats")
        replacement = raw.get("fresh_self_replacement")
        require(isinstance(identity, Mapping) and identity.get("status") == "PASS" and
                isinstance(deterministic, Mapping) and
                set(deterministic) == {"correct_history_N", "fresh_destination"} and
                all(item.get("status") == "PASS"
                    for item in deterministic.values()) and
                isinstance(replacement, Mapping) and
                replacement.get("status") == "PASS" and
                len(replacement.get("regions", [])) == 9 and
                all(item.get("status") == "PASS"
                    for item in replacement["regions"]),
                f"{regime} G2 identity/surgery gate differs")
        by_regime[str(regime)] = {
            "package": {
                "path": row["package_dir"].relative_to(run_dir).as_posix(),
                "manifest_sha256": row["package_manifest_sha256"],
                "raw_artifact_sha256": row["raw_artifact_sha256"],
            },
            "runtime_fingerprint": dict(runtime),
            "host": dict(host),
            "weights": dict(weights),
            "kv": dict(kv),
            "g2_status": "PASS",
        }
    require(set(by_regime) == set(REGIMES), "technical regime set differs")
    nf4_runtime = by_regime["nf4"]["runtime_fingerprint"]
    bf16_runtime = by_regime["bf16"]["runtime_fingerprint"]
    stable_fields = (
        "model_id", "requested_revision", "resolved_snapshot", "architecture",
        "attention_backend", "kv_dtype", "eos_ids", "geometry",
        "repository_commit", "dependency_versions", "gpu_uuid",
    )
    differing = [field for field in stable_fields
                 if nf4_runtime.get(field) != bf16_runtime.get(field)]
    require(not differing, f"matched-runtime fields differ: {differing}")
    require(by_regime["nf4"]["host"]["gpu_uuid"] ==
            by_regime["bf16"]["host"]["gpu_uuid"],
            "NF4/bf16 did not run on the same GPU host")
    return {
        "status": "PASS",
        "same_host": True,
        "stable_runtime_fields_equal": True,
        "regimes": by_regime,
    }


def _bound_file_matches(row: Any, path: Path, label: str) -> None:
    require(isinstance(row, Mapping) and
            set(row) == {"path", "sha256", "size_bytes"} and
            row.get("sha256") == file_sha256(path) and
            row.get("size_bytes") == path.stat().st_size,
            f"{label} binding differs")


def _validate_completion_and_manifest(
    run_dir: Path, *, observed_outcomes: set[tuple[str, str, int]],
    outcome_timings: Mapping[tuple[str, str, int], float],
) -> dict[str, Any]:
    completion_paths = sorted(run_dir.glob("precision-probe-p01-completion_*.json"))
    manifest_paths = sorted(run_dir.glob("precision-probe-p01-run-manifest_*.json"))
    require(len(completion_paths) == len(manifest_paths) == 1,
            "p01 run requires exactly one completion and one run manifest")
    completion_path = completion_paths[0]
    manifest_path = manifest_paths[0]
    completion = load_object(completion_path)
    manifest = load_object(manifest_path)
    require(completion.get("schema") == COMPLETION_SCHEMA and
            completion.get("status") == "COMPLETE",
            "p01 runner completion is not COMPLETE")
    require(manifest.get("schema") == RUN_MANIFEST_SCHEMA and
            manifest.get("status") == "COMPLETE",
            "p01 run manifest is not COMPLETE")
    _require_formal_boundary(completion, "runner completion")
    _require_formal_boundary(manifest, "run manifest")
    _bound_file_matches(completion.get("run_manifest"), manifest_path,
                        "runner manifest")
    inventory = completion.get("artifact_inventory")
    require(isinstance(inventory, list) and inventory,
            "runner completion inventory is empty")
    raw_inventory: list[Mapping[str, Any]] = []
    for row in inventory:
        require(isinstance(row, Mapping) and
                set(row) == {"path", "sha256", "size_bytes"} and
                isinstance(row.get("path"), str),
                "runner completion inventory row differs")
        raw_inventory.append(row)
    actual = {
        path.relative_to(run_dir).as_posix(): path
        for path in run_dir.rglob("*")
        if path.is_file() and path != completion_path
    }
    inventory_by_path: dict[str, Mapping[str, Any]] = {}
    for row in raw_inventory:
        matches = [relative for relative in actual
                   if row["path"] == relative or
                   row["path"].endswith("/" + relative)]
        require(len(matches) == 1 and matches[0] not in inventory_by_path,
                f"runner inventory path cannot be resolved uniquely: "
                f"{row['path']}")
        inventory_by_path[matches[0]] = row
    require(set(inventory_by_path) == set(actual),
            "runner completion inventory is not exhaustive")
    for relative, path in actual.items():
        row = inventory_by_path[relative]
        require(row.get("sha256") == file_sha256(path) and
                row.get("size_bytes") == path.stat().st_size,
                f"runner inventory bytes differ: {relative}")
    require(completion.get("recovery_raw_files") == [],
            "runner completion retained recovery raw files")

    provider = manifest.get("provider")
    require(isinstance(provider, Mapping) and
            isinstance(provider.get("effective_hard_deadline_seconds"),
                       (int, float)) and
            float(completion.get("provider_elapsed_seconds", math.inf)) <=
            float(provider["effective_hard_deadline_seconds"]) and
            float(completion.get("estimated_provider_cost_usd", math.inf)) <= 4.0,
            "runner completion exceeded its provider ceiling")
    require(manifest.get("model") ==
            "Qwen/Qwen3-30B-A3B-Instruct-2507" and
            manifest.get("revision") ==
            "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
            "run manifest subject differs")
    bindings = manifest.get("bindings")
    required_bindings = {
        "preregistration", "identity_fixture", "technical_fixture",
        "case_e01", "case_e02", "case_e03", "runner",
        "precision_subject_loader", "artifact_packager",
        "coherent_canary_case", "coherent_canary_runtime",
        "coherent_canary_technical", "coherent_canary_tokens",
        "coherent_canary_controls", "coherent_canary_schema",
        "coherent_state_tokens",
    }
    require(isinstance(bindings, Mapping) and
            required_bindings <= set(bindings),
            f"run source bindings are incomplete: "
            f"{sorted(required_bindings-set(bindings or {}))}")

    regimes = manifest.get("regimes")
    require(isinstance(regimes, Mapping) and set(regimes) == set(REGIMES) and
            all(isinstance(regimes[regime], Mapping) and
                regimes[regime].get("status") == "OUTCOMES_FINISHED" and
                regimes[regime].get("technical", {}).get("status") == "PASS"
                for regime in REGIMES),
            "run manifest regime/technical status differs")
    manifest_outcomes: set[tuple[str, str, int]] = set()
    for regime_name in REGIMES:
        rows = regimes[regime_name].get("outcomes")
        require(isinstance(rows, list),
                f"{regime_name} manifest outcome list is absent")
        for row in rows:
            require(isinstance(row, Mapping) and
                    row.get("regime") == regime_name and
                    row.get("completion_status") == "COMPLETE" and
                    row.get("recovery_raw_path") is None,
                    f"{regime_name} manifest outcome receipt differs")
            key = (regime_name, row.get("case_id"), row.get("repeat_index"))
            require(key not in manifest_outcomes and
                    key[1] in {"e01", "e02", "e03"} and
                    isinstance(key[2], int),
                    f"duplicate/invalid manifest outcome receipt: {key}")
            package = row.get("raw_package")
            require(isinstance(package, Mapping) and
                    package.get("verification_status") == "VERIFIED" and
                    isinstance(package.get("path"), str),
                    f"{key} raw package receipt differs")
            package_dir = run_dir / Path(package["path"]).name
            require(package_dir.is_dir() and
                    package.get("manifest_sha256") ==
                    file_sha256(package_dir / "manifest.json"),
                    f"{key} raw package binding differs")
            for artifact_name in ("compact", "renders"):
                binding_row = row.get(artifact_name)
                require(isinstance(binding_row, Mapping) and
                        isinstance(binding_row.get("path"), str),
                        f"{key} {artifact_name} binding is absent")
                artifact_path = run_dir / Path(binding_row["path"]).name
                _bound_file_matches(binding_row, artifact_path,
                                    f"{key} {artifact_name}")
            manifest_outcomes.add(key)
    require(manifest_outcomes == observed_outcomes,
            "manifest selected outcomes differ from final packages")
    matched = manifest.get("matched_runtime_gate")
    require(isinstance(matched, Mapping) and matched.get("status") == "PASS" and
            completion.get("matched_runtime_gate") == matched,
            "run manifest matched-runtime gate is not PASS")

    selected = manifest.get("selected_case_repeats")
    expected_selected = ({"e01": 2, "e02": 1, "e03": 1}
                         if observed_outcomes & EXTENSION_KEYS else {"e01": 2})
    require(selected == expected_selected,
            "run manifest selected case/repeat set differs from packages")
    decision_binding = manifest.get("extension_decision")
    decision_paths = sorted(run_dir.glob(
        "precision-probe-p01-extension-decision_*.json"))
    require(len(decision_paths) == 1,
            "p01 run requires exactly one frozen extension decision")
    decision_path = decision_paths[0]
    _bound_file_matches(decision_binding, decision_path, "extension decision")
    decision = load_object(decision_path)
    _require_formal_boundary(decision, "extension decision")
    inputs = decision.get("inputs")
    require(decision.get("schema") == DECISION_SCHEMA and
            isinstance(inputs, Mapping),
            "extension decision schema/inputs differ")
    statuses = inputs.get("nf4_e01_completion_statuses")
    times = inputs.get("nf4_e01_outcome_wall_time_seconds")
    require(statuses == ["COMPLETE", "COMPLETE"] and
            isinstance(times, list) and len(times) == 2 and
            all(isinstance(value, (int, float)) and
                not isinstance(value, bool) and math.isfinite(float(value)) and
                float(value) >= 0 for value in times),
            "extension decision completion/timing inputs differ")
    raw_times = [outcome_timings[("nf4", "e01", repeat)]
                 for repeat in (1, 2)]
    require(all(math.isclose(float(declared), float(observed),
                             rel_tol=0, abs_tol=1e-12)
                for declared, observed in zip(times, raw_times)),
            "extension decision timings differ from final NF4 e01 raws")
    elapsed = inputs.get("provider_elapsed_seconds")
    rate = inputs.get("hourly_cost_usd")
    cap = inputs.get("provider_wall_cap_seconds")
    require(all(isinstance(value, (int, float)) and
                not isinstance(value, bool) and math.isfinite(float(value)) and
                float(value) > 0 for value in (elapsed, rate, cap)),
            "extension decision provider scalar inputs differ")
    slower = max(float(value) for value in times)
    hard_deadline = min(7200.0, float(cap), 4.0 * 3600.0 / float(rate))
    forecast = float(elapsed) + 1.20 * (6.0 * slower + 900.0)
    allowed = forecast <= hard_deadline
    require(math.isclose(float(inputs.get("slower_complete_nf4_e01_seconds")),
                         slower, rel_tol=0, abs_tol=1e-12) and
            math.isclose(float(inputs.get("four_dollar_rate_ceiling_seconds")),
                         4.0 * 3600.0 / float(rate), rel_tol=0,
                         abs_tol=1e-9) and
            math.isclose(float(inputs.get("effective_hard_deadline_seconds")),
                         hard_deadline, rel_tol=0, abs_tol=1e-9) and
            math.isclose(float(inputs.get("forecast_seconds")), forecast,
                         rel_tol=0, abs_tol=1e-9),
            "extension decision derived timing fields differ")
    require(decision.get("selected_case_repeats") == expected_selected and
            decision.get("extension_allowed") is allowed and
            allowed is (expected_selected != {"e01": 2}) and
            decision.get("decision") ==
            ("EXTEND_E02_E03" if allowed else "BASE_E01_ONLY") and
            decision.get("formula") ==
            "elapsed + 1.20 * (6*t + 900) <= min(7200, cap, 4*3600/rate)" and
            decision.get("score_or_generation_accessible_to_decision") is False and
            decision.get("outcome_information_surface") == [
                "completion_status", "outcome_wall_time_seconds"],
            "frozen extension decision differs from observed execution")
    return {
        "completion": {
            "path": completion_path.relative_to(run_dir).as_posix(),
            "sha256": file_sha256(completion_path),
        },
        "manifest": {
            "path": manifest_path.relative_to(run_dir).as_posix(),
            "sha256": file_sha256(manifest_path),
        },
        "extension_decision": {
            "path": decision_path.relative_to(run_dir).as_posix(),
            "sha256": file_sha256(decision_path),
            "decision": decision.get("decision"),
        },
        "matched_runtime_gate": dict(matched),
        "selected_case_repeats": dict(selected),
        "source_binding_names": sorted(bindings),
        "provider_elapsed_seconds": completion["provider_elapsed_seconds"],
        "estimated_provider_cost_usd": completion[
            "estimated_provider_cost_usd"],
    }


def recursive_differences(left: Any, right: Any, path: str = "") -> list[str]:
    """Return every differing JSON pointer; values remain in bound packages."""
    if type(left) is not type(right):
        return [path or "/"]
    if isinstance(left, dict):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            child = f"{path}/{escaped}"
            if key not in left or key not in right:
                result.append(child)
            else:
                result.extend(recursive_differences(left[key], right[key], child))
        return result
    if isinstance(left, list):
        result = []
        for index in range(max(len(left), len(right))):
            child = f"{path}/{index}"
            if index >= len(left) or index >= len(right):
                result.append(child)
            else:
                result.extend(recursive_differences(left[index], right[index], child))
        return result
    return [] if left == right else [path or "/"]


def binding_inventory(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(run_dir).as_posix()
        rows.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    require(rows, "p01 run directory is empty")
    return rows


def summarize_values(values_by_case: Mapping[str, float]) -> dict[str, Any]:
    ordered = {key: float(values_by_case[key]) for key in sorted(values_by_case)}
    values = list(ordered.values())
    require(values, "cannot summarize an empty case set")
    return {
        "n": len(values),
        "values_by_case": ordered,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sign_count": {
            "positive": sum(value > 0 for value in values),
            "zero": sum(value == 0 for value in values),
            "negative": sum(value < 0 for value in values),
        },
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    packaged_documents = packaged_p01_documents(run_dir)
    technical_gates = _technical_gate_summary(run_dir, packaged_documents)
    packages = package_rows(run_dir, packaged_documents)
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in packages:
        key = row["key"]
        require(key not in by_key, f"duplicate outcome package: {key}")
        by_key[key] = row
    observed = set(by_key)
    require(BASE_KEYS <= observed,
            f"mandatory base outcome packages are absent: {sorted(BASE_KEYS-observed)}")
    extension_observed = observed & EXTENSION_KEYS
    require(not extension_observed or extension_observed == EXTENSION_KEYS,
            "outcome-blind extension is not all-or-none across runtimes/cases")
    allowed = BASE_KEYS | EXTENSION_KEYS
    require(observed <= allowed,
            f"unexpected outcome runtime/case/repeat keys: {sorted(observed-allowed)}")
    completion_record = _validate_completion_and_manifest(
        run_dir, observed_outcomes=observed,
        outcome_timings={
            key: float(row["raw"]["outcome_wall_time_seconds"])
            for key, row in by_key.items()
        })

    outcome_rows: dict[str, Any] = {}
    for key in sorted(by_key):
        regime, case_id, repeat = key
        raw_runtime = by_key[key]["raw"].get("runtime_fingerprint")
        require(raw_runtime == technical_gates["regimes"][regime][
            "runtime_fingerprint"],
            f"{regime}/{case_id}/repeat{repeat} outcome runtime differs "
            "from its passing technical gate")
        analysis = analyze_outcome(by_key[key]["raw"])
        label = f"{regime}:{case_id}:r{repeat}"
        outcome_rows[label] = {
            "regime": regime, "case_id": case_id, "repeat": repeat,
            "package": {
                "path": by_key[key]["package_dir"].relative_to(run_dir).as_posix(),
                "manifest_sha256": by_key[key]["package_manifest_sha256"],
                "raw_artifact_sha256": by_key[key]["raw_artifact_sha256"],
            },
            "analysis": analysis,
        }

    repeat_stability: dict[str, Any] = {}
    for regime in REGIMES:
        left = by_key[(regime, "e01", 1)]["raw"]
        right = by_key[(regime, "e01", 2)]["raw"]
        left_model_facing = {"phase_a": left["phase_a"],
                             "treatment": left["treatment"]}
        right_model_facing = {"phase_a": right["phase_a"],
                              "treatment": right["treatment"]}
        differences = recursive_differences(left_model_facing, right_model_facing)
        repeat_stability[regime] = {
            "stable": not differences,
            "left_canonical_sha256": sha256_bytes(canonical_bytes(left_model_facing)),
            "right_canonical_sha256": sha256_bytes(canonical_bytes(right_model_facing)),
            "differing_json_pointers": differences,
            "differing_field_count": len(differences),
        }

    cross_runtime: dict[str, Any] = {}
    case_repeats = [("e01", 1), ("e01", 2)]
    if extension_observed:
        case_repeats.extend((("e02", 1), ("e03", 1)))
    for case_id, repeat in case_repeats:
        nf4 = outcome_rows[f"nf4:{case_id}:r{repeat}"]["analysis"]
        bf16 = outcome_rows[f"bf16:{case_id}:r{repeat}"]["analysis"]
        label = f"{case_id}:r{repeat}"
        cross_runtime[label] = {
            "D_value_N_R2_nf4": nf4["primary_D_value_N_R2"],
            "D_value_N_R2_bf16": bf16["primary_D_value_N_R2"],
            "Delta_runtime_nf4_minus_bf16": (
                nf4["primary_D_value_N_R2"] - bf16["primary_D_value_N_R2"]),
            "Hplus_nf4": nf4["primary_Hplus_value_N_R2"],
            "Hplus_bf16": bf16["primary_Hplus_value_N_R2"],
        }

    cross_runtime_case_summary = summarize_values({
        case_id: cross_runtime[f"{case_id}:r1"][
            "Delta_runtime_nf4_minus_bf16"]
        for case_id in (("e01", "e02", "e03")
                        if extension_observed else ("e01",))
    })

    aggregate: dict[str, Any] = {}
    for regime in REGIMES:
        # Repeats remain literal repeatability observations.  For the optional
        # multi-case description, e01 contributes its first (identical if G4
        # passes) execution once rather than double-weighting that fixture.
        values = {
            "e01": outcome_rows[f"{regime}:e01:r1"]["analysis"][
                "primary_D_value_N_R2"]
        }
        if extension_observed:
            for case_id in ("e02", "e03"):
                values[case_id] = outcome_rows[f"{regime}:{case_id}:r1"][
                    "analysis"]["primary_D_value_N_R2"]
        aggregate[regime] = summarize_values(values)

    all_stable = all(row["stable"] for row in repeat_stability.values())
    return {
        "schema": ANALYSIS_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir),
        "formal_v12_decision_eligible": False,
        "v12_reentry_authorized": False,
        "fixed_fixture_exploratory_only": True,
        "extension_executed": bool(extension_observed),
        "outcome_keys": [list(key) for key in sorted(observed)],
        "input_file_inventory": binding_inventory(run_dir),
        "runner_completion_and_manifest": completion_record,
        "technical_gates": technical_gates,
        "outcomes": outcome_rows,
        "within_runtime_repeat_stability": repeat_stability,
        "all_mandatory_repeats_stable": all_stable,
        "small_cross_runtime_contrast_interpretation_permitted": all_stable,
        "cross_runtime_primary_contrasts": cross_runtime,
        "cross_runtime_primary_case_summary": cross_runtime_case_summary,
        "per_runtime_case_summary": aggregate,
        "inference": {
            "p_values_computed": False,
            "confidence_intervals_computed": False,
            "equivalence_test_computed": False,
            "population_interaction_claim_authorized": False,
            "note": (
                "P01 is a fixed-fixture descriptive screen of a bundled weight/"
                "kernel runtime axis.  It does not identify semantic attribution, "
                "4-bit KV behavior, efficacy, or population generality."
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
        "precision-probe-p01-independent-analysis_"
        f"Qwen3-30B-A3B-Instruct-2507_{stamp}.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    require(run_dir.is_dir(), f"p01 run directory is absent: {run_dir}")
    result = analyze_run(run_dir)
    output = args.output.resolve() if args.output else default_output()
    print(f"ANALYZE precision-probe-p01 run={run_dir} -> {output}", flush=True)
    write_exclusive(output, result)
    print(json.dumps({
        "output": str(output),
        "output_sha256": file_sha256(output),
        "repeat_stable": result["all_mandatory_repeats_stable"],
        "extension_executed": result["extension_executed"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
