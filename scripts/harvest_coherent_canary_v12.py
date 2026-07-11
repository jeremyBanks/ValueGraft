#!/usr/bin/env python3
"""Independently harvest one coherent-state canary v12 treatment artifact.

This harvester performs no model forward and no tokenizer reconstruction.  It
checks the persisted treatment artifact against its bound Phase-A release and
source files, decodes the authoritative little-endian float32 evidence, and
recomputes the preregistered per-case contrasts.  Runner decimal summaries and
runner status labels are deliberately not used.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DESIGN_ID = "coherent-state-decision-canary-v12"
RUN_SCHEMA = "coherent_state_decision_canary_v12_treatment_run_v1"
TREATMENT_SCHEMA = "coherent_state_decision_canary_v12_treatment_raw_v1"
PHASE_A_REPORT_SCHEMA = (
    "coherent_state_decision_canary_v12_phase_a_validation_v1")
PHASE_A_RUN_SCHEMA = "coherent_state_decision_canary_v12_phase_a_run_v1"
PHASE_A_PAYLOAD_SCHEMA = "coherent_state_decision_canary_v12_phase_a_raw_v1"
TECHNICAL_REPORT_SCHEMA = (
    "coherent_state_decision_canary_v12_technical_validation_v1")
REPORT_SCHEMA = "coherent_state_decision_canary_v12_treatment_harvest_v1"
SUBJECTS = {
    "local-apparatus": {
        "model_id": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "semantic_evidence_eligible": False,
    },
    "exact-subject": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "semantic_evidence_eligible": True,
    },
}
REGIONS = ("R1_content", "R2_boundary", "R3_anchor")
PRIMARY_CELLS = ("FF", "FC", "FW", "CF", "CC", "CW", "WF", "WC", "WW")
P_CELLS = ("CC", "WW", "FC", "FW")
PLACEBO_CELL = "V_PLACEBO"
EXPECTED_PRIMARY_SELECTORS = {
    *(("N", region, cell) for region in REGIONS for cell in PRIMARY_CELLS),
    *(("P", "R2_boundary", cell) for cell in P_CELLS),
}
EXPECTED_BINDINGS = {
    "case", "preregistration", "revision4_manifest",
    "revision4_blind_review", "revision4_paired_review",
    "technical_validation_report", "phase_a_validation_report",
    "phase_a_raw_artifact",
}


class HarvestError(RuntimeError):
    """Persisted scientific evidence differs from the v12 contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HarvestError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except Exception as exc:
        raise HarvestError(f"cannot parse {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def bound_file(row: Any, repo_root: Path, label: str) -> Path:
    require(isinstance(row, Mapping) and isinstance(row.get("path"), str) and
            is_sha256(row.get("sha256")), f"{label} binding differs")
    path = resolve_path(row["path"], repo_root)
    require(path.is_file(), f"{label} bound file is absent")
    require(file_sha256(path) == row["sha256"], f"{label} bound file hash differs")
    return path


def decode_float32_bits(value: Any, label: str) -> float:
    require(isinstance(value, str) and len(value) == 8 and all(
        character in "0123456789abcdef" for character in value),
        f"{label} is not eight lowercase float32 hex characters")
    result = struct.unpack("<f", bytes.fromhex(value))[0]
    require(math.isfinite(result), f"{label} is nonfinite")
    return float(result)


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def float32_bits(value: float) -> str:
    return struct.pack("<f", float32(value)).hex()


def target_summary(record: Any, label: str) -> dict[str, Any]:
    require(isinstance(record, Mapping), f"{label} target record is absent")
    token_ids = record.get("target_token_ids")
    token_bits = record.get("token_logprob_float32_bits")
    require(isinstance(token_ids, list) and bool(token_ids) and all(
        isinstance(value, int) for value in token_ids),
        f"{label} target IDs differ")
    require(isinstance(token_bits, list) and len(token_bits) == len(token_ids),
            f"{label} token-logprob coverage differs")
    token_values = [decode_float32_bits(value, f"{label} token {index}")
                    for index, value in enumerate(token_bits)]
    mean_bits = record.get("mean_logprob_float32_bits")
    mean = decode_float32_bits(mean_bits, f"{label} mean")
    # The model-facing scorer records the actual float32 reduction as mean_bits.
    # The float64 token fmean is diagnostic only; it is not substituted for the
    # scorer's observed reduction order.
    token_fmean = statistics.fmean(token_values)
    require(abs(mean - token_fmean) <= max(1e-5, abs(mean) * 2e-6),
            f"{label} mean is inconsistent with its per-token float32 evidence")
    return {
        "target_token_ids": list(token_ids),
        "token_count": len(token_ids),
        "mean_logprob": mean,
        "mean_logprob_float32_bits": mean_bits,
        "token_fmean_float64_diagnostic": token_fmean,
    }


def score_summary(record: Any, label: str) -> dict[str, Any]:
    require(isinstance(record, Mapping), f"{label} score is absent")
    correct = target_summary(record.get("correct"), f"{label}.correct")
    counterfactual = target_summary(
        record.get("counterfactual"), f"{label}.counterfactual")
    margin = float32(
        correct["mean_logprob"] - counterfactual["mean_logprob"])
    margin_bits = float32_bits(margin)
    require(record.get("margin_float32_bits") == margin_bits,
            f"{label} persisted margin bits differ from recomputation")
    return {
        "correct": correct,
        "counterfactual": counterfactual,
        "margin": margin,
        "margin_float32_bits": margin_bits,
    }


def score_pair(record: Any, label: str) -> dict[str, Any]:
    require(isinstance(record, Mapping) and
            set(record) == {"focal", "nonfocal"},
            f"{label} focal/nonfocal score set differs")
    return {
        probe: score_summary(record[probe], f"{label}.{probe}")
        for probe in ("focal", "nonfocal")
    }


def verify_source_bindings(raw: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    bindings = raw.get("bindings")
    require(isinstance(bindings, Mapping) and set(bindings) == EXPECTED_BINDINGS,
            "treatment binding set differs")
    paths = {name: bound_file(bindings[name], repo_root, name)
             for name in EXPECTED_BINDINGS}
    documents = {
        name: load_object(path) for name, path in paths.items()
        if name != "preregistration"
    }
    case_id = raw.get("case_id")
    subject = raw.get("subject")
    require(subject in SUBJECTS, "treatment subject differs")
    semantic_eligible = SUBJECTS[subject]["semantic_evidence_eligible"]

    case = documents["case"]
    require(case.get("design_id") == DESIGN_ID and case.get("case_id") == case_id,
            "bound case design/ID differs")
    manifest = documents["revision4_manifest"]
    manifest_rows = [row for row in manifest.get("cases", [])
                     if isinstance(row, Mapping) and row.get("case_id") == case_id]
    require(len(manifest_rows) == 1 and
            manifest_rows[0].get("input_file_sha256") == file_sha256(paths["case"]),
            "manifest does not bind the exact case")

    blind = documents["revision4_blind_review"]
    require(blind.get("schema") ==
            "coherent_state_decision_canary_v12_blind_review_v1" and
            blind.get("aggregate", {}).get("overall_verdict") == "PASS" and
            blind.get("shared_carrier_anchor_review", {}).get("verdict") == "PASS",
            "blind/carrier review is not PASS")
    paired = documents["revision4_paired_review"]
    paired_rows = [row for row in paired.get("paired_case_reviews", [])
                   if isinstance(row, Mapping) and row.get("case_id") == case_id]
    require(paired.get("schema") ==
            "coherent_state_decision_canary_v12_paired_diversity_review_v1" and
            paired.get("overall_verdict") == "PASS" and
            paired.get("cross_case_diversity_review", {}).get("verdict") == "PASS" and
            len(paired_rows) == 1 and paired_rows[0].get("verdict") == "PASS",
            "paired/diversity review is not PASS for the case")

    technical = documents["technical_validation_report"]
    technical_runtime = technical.get("checks", {}).get(
        "runtime_fingerprint", {})
    technical_fingerprint = technical_runtime.get("evidence", {}).get(
        "fingerprint_sha256")
    require(technical.get("schema") == TECHNICAL_REPORT_SCHEMA and
            technical.get("design_id") == DESIGN_ID and
            technical.get("status") == "PASS" and
            technical.get("subject") == subject and
            technical.get("semantic_release_eligible") is semantic_eligible and
            technical_runtime.get("passed") is True and
            is_sha256(technical_fingerprint),
            "bound subject technical validation is not valid for this mode")

    phase_report = documents["phase_a_validation_report"]
    phase_runtime = phase_report.get("checks", {}).get(
        "runtime_fingerprint", {})
    phase_fingerprint = phase_runtime.get("evidence", {}).get(
        "fingerprint_sha256")
    phase_status = phase_report.get("status")
    allowed_phase_statuses = (
        {"PRETREATMENT_PASS"} if semantic_eligible else
        {"PRETREATMENT_PASS", "ESTIMAND_INADEQUATE"})
    inadequacy = phase_report.get("inadequacy_reasons")
    require(phase_report.get("schema") == PHASE_A_REPORT_SCHEMA and
            phase_report.get("design_id") == DESIGN_ID and
            phase_report.get("case_id") == case_id and
            phase_report.get("subject") == subject and
            phase_status in allowed_phase_statuses and
            phase_report.get("semantic_release_eligible") is semantic_eligible and
            phase_runtime.get("passed") is True and
            phase_fingerprint == technical_fingerprint and
            not phase_report.get("invalidity_reasons") and
            isinstance(inadequacy, list) and
            ((phase_status == "PRETREATMENT_PASS" and not inadequacy) or
             (phase_status == "ESTIMAND_INADEQUATE" and bool(inadequacy))),
            "bound Phase-A validation is not eligible for this subject mode")
    report_raw = phase_report.get("raw_artifact")
    require(isinstance(report_raw, Mapping) and
            report_raw.get("sha256") == bindings["phase_a_raw_artifact"]["sha256"],
            "Phase-A report raw-artifact hash differs from treatment binding")
    report_raw_path = bound_file(report_raw, repo_root, "Phase-A report raw artifact")
    require(file_sha256(report_raw_path) == file_sha256(paths["phase_a_raw_artifact"]),
            "Phase-A report and treatment bind different raw bytes")

    phase_raw = documents["phase_a_raw_artifact"]
    phase_payload = phase_raw.get("phase_a")
    require(phase_raw.get("schema") == PHASE_A_RUN_SCHEMA and
            phase_raw.get("design_id") == DESIGN_ID and
            phase_raw.get("case_id") == case_id and
            phase_raw.get("subject") == subject and
            phase_raw.get("treatment_scores_present") is False and
            isinstance(phase_payload, Mapping) and
            phase_payload.get("schema") == PHASE_A_PAYLOAD_SCHEMA and
            phase_payload.get("design_id") == DESIGN_ID and
            phase_payload.get("case_id") == case_id and
            phase_payload.get("treatment_scores_present") is False,
            "bound Phase-A raw artifact differs")
    phase_bindings = phase_raw.get("bindings")
    require(isinstance(phase_bindings, Mapping),
            "Phase-A raw source bindings are absent")
    for name in (
            "case", "preregistration", "revision4_manifest",
            "revision4_blind_review", "revision4_paired_review",
            "technical_validation_report"):
        require(isinstance(phase_bindings.get(name), Mapping) and
                phase_bindings[name].get("sha256") == bindings[name]["sha256"],
                f"Phase-A/treatment {name} binding continuity differs")

    return {
        "paths": paths,
        "bindings": {name: {"path": str(paths[name]),
                             "sha256": bindings[name]["sha256"]}
                     for name in sorted(bindings)},
        "phase_raw": phase_raw,
        "phase_a_status": phase_status,
        "phase_a_inadequacy_reasons": list(inadequacy),
        "runtime_fingerprint_sha256": phase_fingerprint,
    }


def verify_runtime(raw: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    runtime = raw.get("runtime_fingerprint")
    phase_runtime = evidence["phase_raw"].get("runtime_fingerprint")
    require(isinstance(runtime, Mapping) and runtime == phase_runtime,
            "treatment/Phase-A runtime fingerprints are not identical")
    subject = raw.get("subject")
    require(subject in SUBJECTS, "treatment subject differs")
    expected = SUBJECTS[subject]
    require(runtime.get("requested_model") == expected["model_id"] and
            runtime.get("requested_revision") == expected["revision"] and
            runtime.get("dtype") == "torch.bfloat16" and
            runtime.get("attention_backend") == "eager" and
            runtime.get("subject_spec", {}).get("key") == subject,
            "treatment runtime is not the selected pinned subject")
    declared = runtime.get("fingerprint_sha256")
    payload = {key: value for key, value in runtime.items()
               if key != "fingerprint_sha256"}
    recomputed = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    require(declared == recomputed == evidence["runtime_fingerprint_sha256"],
            "runtime fingerprint commitment/release continuity differs")
    return {
        "subject": subject,
        "requested_model": expected["model_id"],
        "requested_revision": expected["revision"],
        "dtype": runtime["dtype"],
        "attention_backend": runtime["attention_backend"],
        "fingerprint_sha256": declared,
        "semantic_evidence_eligible": expected["semantic_evidence_eligible"],
    }


def verify_target_consistency(cell_scores: Mapping[tuple[str, str, str], Any],
                              placebo_scores: Mapping[str, Any],
                              fresh_scores: Mapping[str, Any]) -> None:
    signatures: dict[str, set[tuple[tuple[int, ...], tuple[int, ...]]]] = {
        "focal": set(), "nonfocal": set()}
    all_pairs = list(cell_scores.values()) + [fresh_scores]
    all_pairs.extend(value for value in placebo_scores.values() if value is not None)
    for pair in all_pairs:
        for probe in signatures:
            signatures[probe].add((
                tuple(pair[probe]["correct"]["target_token_ids"]),
                tuple(pair[probe]["counterfactual"]["target_token_ids"]),
            ))
    require(all(len(values) == 1 for values in signatures.values()),
            "target token IDs differ across treatment arms")


def contrast(cells: Mapping[str, Mapping[str, Any]], *,
             correct_cell: str, wrong_cell: str,
             fresh: Mapping[str, Any]) -> dict[str, float]:
    correct = cells[correct_cell]
    wrong = cells[wrong_cell]
    d_focal = correct["focal"]["margin"] - wrong["focal"]["margin"]
    d_nonfocal = correct["nonfocal"]["margin"] - wrong["nonfocal"]["margin"]
    return {
        "D_focal": d_focal,
        "D_nonfocal": d_nonfocal,
        "SEL": d_focal - abs(d_nonfocal),
        "Hplus": (correct["focal"]["correct"]["mean_logprob"] -
                  wrong["focal"]["correct"]["mean_logprob"]),
        "U": correct["focal"]["margin"] - fresh["focal"]["margin"],
        "Uplus": (correct["focal"]["correct"]["mean_logprob"] -
                  fresh["focal"]["correct"]["mean_logprob"]),
    }


def movement_from_fresh(scores: Mapping[str, Any],
                        fresh: Mapping[str, Any]) -> dict[str, Any]:
    return {
        probe: {
            "margin_delta": (scores[probe]["margin"] -
                             fresh[probe]["margin"]),
            "correct_target_logprob_delta": (
                scores[probe]["correct"]["mean_logprob"] -
                fresh[probe]["correct"]["mean_logprob"]),
        }
        for probe in ("focal", "nonfocal")
    }


def harvest(treatment_path: Path, *, repo_root: Path = ROOT) -> dict[str, Any]:
    raw = load_object(treatment_path)
    subject = raw.get("subject")
    require(raw.get("schema") == RUN_SCHEMA and
            raw.get("design_id") == DESIGN_ID and
            isinstance(raw.get("case_id"), str) and
            raw.get("phase_a_scores_present") is False,
            "treatment run schema/design/case/blinding differs")
    require(subject in SUBJECTS, "treatment run subject differs")
    semantic_eligible = SUBJECTS[subject]["semantic_evidence_eligible"]
    require(raw.get("semantic_evidence_eligible") is semantic_eligible and
            raw.get("apparatus_integration_only") is (not semantic_eligible),
            "treatment run semantic/apparatus eligibility differs")
    treatment = raw.get("treatment")
    require(isinstance(treatment, Mapping) and
            treatment.get("schema") == TREATMENT_SCHEMA and
            treatment.get("design_id") == DESIGN_ID and
            treatment.get("case_id") == raw.get("case_id") and
            treatment.get("phase_a_scores_present") is False,
            "nested treatment payload schema/design/case/blinding differs")
    evidence = verify_source_bindings(raw, repo_root)
    runtime = verify_runtime(raw, evidence)

    arms = treatment.get("arms")
    require(isinstance(arms, list) and len(arms) == 34 and
            treatment.get("arm_count") == 34 and
            treatment.get("primary_arm_count") == 31 and
            treatment.get("placebo_control_count") == 3,
            "treatment requires 31 primary plus three placebo arm records")
    primary_raw: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    placebo_raw: dict[str, Mapping[str, Any]] = {}
    for row in arms:
        require(isinstance(row, Mapping), "treatment arm is not an object")
        selector = (row.get("schedule"), row.get("region"), row.get("cell"))
        if selector in EXPECTED_PRIMARY_SELECTORS:
            require(row.get("arm_kind") == "primary",
                    f"primary arm kind differs: {selector}")
            require(selector not in primary_raw, f"duplicate primary arm {selector}")
            primary_raw[selector] = row
        elif (row.get("schedule") == "N" and row.get("region") in REGIONS and
              row.get("cell") == PLACEBO_CELL):
            require(row.get("arm_kind") == "placebo_control",
                    f"placebo arm kind differs: {selector}")
            region = str(row["region"])
            require(region not in placebo_raw, f"duplicate placebo arm {region}")
            placebo_raw[region] = row
        else:
            raise HarvestError(f"unexpected treatment arm selector: {selector}")
    require(set(primary_raw) == EXPECTED_PRIMARY_SELECTORS and
            set(placebo_raw) == set(REGIONS),
            "treatment arm selector coverage differs")

    fresh_scores = score_pair(treatment.get("fresh_scores"), "fresh_scores")
    cell_scores = {
        selector: score_pair(row.get("scores"), f"arm{selector}")
        for selector, row in primary_raw.items()
    }
    for region in REGIONS:
        require(cell_scores[("N", region, "FF")] == fresh_scores,
                f"{region} FF differs from the shared fresh baseline")

    placebo_results: dict[str, Any] = {}
    placebo_score_pairs: dict[str, Any] = {}
    for region, row in placebo_raw.items():
        status = row.get("control_status")
        diagnostics = row.get("diagnostics")
        require(status in {"AVAILABLE", "PLACEBO_UNAVAILABLE"} and
                isinstance(diagnostics, Mapping) and
                diagnostics.get("status") == status,
                f"{region} placebo availability/diagnostics differ")
        if status == "AVAILABLE":
            scores = score_pair(row.get("scores"), f"placebo.{region}")
            placebo_score_pairs[region] = scores
            placebo_results[region] = {
                "status": status, "scores": scores,
                "movement_from_fresh": movement_from_fresh(
                    scores, fresh_scores),
                "diagnostics": dict(diagnostics),
            }
        else:
            require("scores" not in row,
                    f"{region} unavailable placebo unexpectedly has scores")
            placebo_score_pairs[region] = None
            placebo_results[region] = {
                "status": status, "scores": None,
                "movement_from_fresh": None,
                "diagnostics": dict(diagnostics),
            }
    verify_target_consistency(cell_scores, placebo_score_pairs, fresh_scores)

    descriptive: dict[str, Any] = {"N": {}, "P": {"R2_boundary": {}}}
    for region in REGIONS:
        descriptive["N"][region] = {
            cell: cell_scores[("N", region, cell)] for cell in PRIMARY_CELLS}
    descriptive["P"]["R2_boundary"] = {
        cell: cell_scores[("P", "R2_boundary", cell)] for cell in P_CELLS}

    regional_n_estimands = {
        region: {
            "full_KV": contrast(
                descriptive["N"][region], correct_cell="CC",
                wrong_cell="WW", fresh=fresh_scores),
            "value_only": contrast(
                descriptive["N"][region], correct_cell="FC",
                wrong_cell="FW", fresh=fresh_scores),
        }
        for region in REGIONS
    }

    n_cells = descriptive["N"]["R2_boundary"]
    p_cells = descriptive["P"]["R2_boundary"]
    estimands = {
        "N": {
            "full_KV": contrast(
                n_cells, correct_cell="CC", wrong_cell="WW", fresh=fresh_scores),
            "value_only": contrast(
                n_cells, correct_cell="FC", wrong_cell="FW", fresh=fresh_scores),
        },
        "P": {
            "full_KV": contrast(
                p_cells, correct_cell="CC", wrong_cell="WW", fresh=fresh_scores),
            "value_only": contrast(
                p_cells, correct_cell="FC", wrong_cell="FW", fresh=fresh_scores),
        },
    }
    yardstick = {}
    for family in ("full_KV", "value_only"):
        d_n = estimands["N"][family]["D_focal"]
        d_p = estimands["P"][family]["D_focal"]
        absolute = abs(d_n - d_p)
        yardstick[family] = {
            "D_focal_N": d_n,
            "D_focal_P": d_p,
            "absolute_N_minus_P": absolute,
            "three_times_absolute_N_minus_P": 3.0 * absolute,
            "per_case_N_positive": d_n > 0,
            "per_case_N_at_least_three_times_difference": d_n >= 3.0 * absolute,
            "aggregate_rule_note": (
                "The preregistered rule is mean_i(D_N) >= "
                "3 * mean_i(abs(D_N-D_P)); this per-case record is a component, "
                "not an independent decision."),
        }

    return {
        "schema": REPORT_SCHEMA,
        "design_id": DESIGN_ID,
        "case_id": raw["case_id"],
        "subject": subject,
        "semantic_evidence_eligible": semantic_eligible,
        "apparatus_integration_only": not semantic_eligible,
        "phase_a_status": evidence["phase_a_status"],
        "phase_a_inadequacy_reasons": evidence[
            "phase_a_inadequacy_reasons"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "treatment_artifact": {
            "path": str(treatment_path), "sha256": file_sha256(treatment_path)},
        "verified_bindings": evidence["bindings"],
        "runtime": runtime,
        "fresh_scores": fresh_scores,
        "R2_primary_estimands": estimands,
        "regional_N_estimands": regional_n_estimands,
        "schedule_yardstick_components": yardstick,
        "descriptive_cell_scores": descriptive,
        "placebo_controls": placebo_results,
        "available_placebo_control_count": sum(
            row["status"] == "AVAILABLE" for row in placebo_results.values()),
        "runner_status_labels_ignored": True,
        "runner_decimal_scores_and_aggregates_ignored": True,
        "inference": {
            "p_values_computed": False,
            "note": (
                ("This local result validates apparatus integration only; it is "
                 "not semantic evidence. " if not semantic_eligible else "") +
                "This is one fixed engineered case. Aggregate stopping logic is "
                "applied only after four independently harvested primary cases, "
                "or six if the frozen ambiguity extension runs."),
        },
    }


def write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) +
            "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def default_output(repo_root: Path, case_id: str, subject: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (repo_root / "results/coherent_canary_v12_harvest" /
            f"coherent-canary-v12-harvest-{case_id}-{subject}-{stamp}.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = harvest(args.treatment, repo_root=args.repo)
    output = args.output or default_output(
        args.repo, report["case_id"], report["subject"])
    print(f"HARVEST case={report['case_id']} treatment={args.treatment} -> {output}",
          flush=True)
    write_exclusive_json(output, report)
    print(json.dumps({
        "case_id": report["case_id"],
        "output": str(output),
        "treatment_sha256": report["treatment_artifact"]["sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
