#!/usr/bin/env python3
"""Lean independent Phase-A validator for coherent-state canary v12.

The validator reads only pre-treatment evidence.  It imports no model-facing
case/runtime module and rejects treatment-shaped fields before inspecting
scores.  Model outcomes are recomputed from little-endian float32 mean bits;
runner verdicts and decimal margins are ignored.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence


DESIGN_ID = "coherent-state-decision-canary-v12"
RUN_SCHEMA = "coherent_state_decision_canary_v12_phase_a_run_v1"
PHASE_SCHEMA = "coherent_state_decision_canary_v12_phase_a_raw_v1"
REPORT_SCHEMA = "coherent_state_decision_canary_v12_phase_a_validation_v1"
EXPECTED_SCORE_KEYS = {
    "A_C_focal", "A_W_focal", "FF_focal",
    "A_C_nonfocal", "A_W_nonfocal", "FF_nonfocal",
}
EXPECTED_BINDINGS = {
    "case", "preregistration", "revision4_manifest",
    "revision4_blind_review", "revision4_paired_review",
    "technical_validation_report",
}
BANNED_KEYS = {
    "treatment", "treatment_scores", "source_rows", "graft_scores",
    "arm_grid", "region_effects", "layer_effects", "treatment_contrasts",
}
SUBJECTS = {
    "local-apparatus": {
        "model_id": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "geometry": {
            "layers": 28, "attention_heads": 16, "kv_heads": 8,
            "head_dim": 128, "rope_theta": 1_000_000.0,
        },
    },
    "exact-subject": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "geometry": {
            "layers": 48, "attention_heads": 32, "kv_heads": 4,
            "head_dim": 128, "rope_theta": 10_000_000.0,
        },
    },
}


class PhaseAValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PhaseAValidationError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except Exception as exc:
        raise PhaseAValidationError(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def decode_float32_bits(value: Any, label: str) -> float:
    require(isinstance(value, str) and len(value) == 8 and all(
        character in "0123456789abcdef" for character in value),
        f"{label} is not eight lowercase float32 hex characters")
    result = struct.unpack("<f", bytes.fromhex(value))[0]
    require(math.isfinite(result), f"{label} is nonfinite")
    return result


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def reject_treatment_fields(value: Any, path: str = "raw") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in BANNED_KEYS:
                raise PhaseAValidationError(
                    f"treatment-shaped field is forbidden: {path}.{key_text}")
            reject_treatment_fields(child, f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_treatment_fields(child, f"{path}[{index}]")


def _bound_file(row: Any, repo_root: Path, label: str) -> tuple[Path, dict[str, Any]]:
    require(isinstance(row, Mapping) and isinstance(row.get("path"), str) and
            is_sha256(row.get("sha256")), f"{label} binding differs")
    declared = Path(row["path"])
    path = declared if declared.is_absolute() else repo_root / declared
    require(path.is_file(), f"{label} file is absent")
    require(file_sha256(path) == row["sha256"], f"{label} file hash differs")
    return path, load_object(path)


def verify_bindings(raw: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    bindings = raw.get("bindings")
    require(isinstance(bindings, Mapping) and set(bindings) == EXPECTED_BINDINGS,
            "Phase-A binding set differs")
    case_path, case = _bound_file(bindings["case"], repo_root, "case")
    manifest_path, manifest = _bound_file(
        bindings["revision4_manifest"], repo_root, "manifest")
    case_id = raw.get("case_id")
    require(case.get("case_id") == case_id and
            case.get("design_id") == DESIGN_ID,
            "bound case ID/design differs")
    manifest_rows = [row for row in manifest.get("cases", [])
                     if row.get("case_id") == case_id]
    require(len(manifest_rows) == 1 and
            manifest_rows[0].get("input_file_sha256") == file_sha256(case_path),
            "manifest does not bind the exact case")
    reviews = []
    for label in ("revision4_blind_review", "revision4_paired_review"):
        path, document = _bound_file(bindings[label], repo_root, label)
        reviews.append((path, document))
    by_schema = {document.get("schema"): document for _, document in reviews}
    require(set(by_schema) == {
        "coherent_state_decision_canary_v12_blind_review_v1",
        "coherent_state_decision_canary_v12_paired_diversity_review_v1",
    }, "Phase-A review schemas differ")
    blind = by_schema["coherent_state_decision_canary_v12_blind_review_v1"]
    paired = by_schema[
        "coherent_state_decision_canary_v12_paired_diversity_review_v1"]
    require(blind.get("aggregate", {}).get("overall_verdict") == "PASS" and
            blind.get("shared_carrier_anchor_review", {}).get("verdict") == "PASS",
            "blind/carrier review is not PASS")
    paired_rows = [row for row in paired.get("paired_case_reviews", [])
                   if row.get("case_id") == case_id]
    require(paired.get("overall_verdict") == "PASS" and
            len(paired_rows) == 1 and paired_rows[0].get("verdict") == "PASS" and
            paired.get("cross_case_diversity_review", {}).get("verdict") == "PASS",
            "paired/diversity review is not PASS for the case")
    prereg_path = Path(bindings["preregistration"]["path"])
    if not prereg_path.is_absolute():
        prereg_path = repo_root / prereg_path
    require(prereg_path.is_file() and
            file_sha256(prereg_path) == bindings["preregistration"]["sha256"],
            "preregistration binding differs")
    technical_path, technical = _bound_file(
        bindings["technical_validation_report"], repo_root,
        "technical validation report")
    require(technical.get("schema") ==
            "coherent_state_decision_canary_v12_technical_validation_v1" and
            technical.get("design_id") == DESIGN_ID and
            technical.get("status") == "PASS" and
            technical.get("subject") == raw.get("subject"),
            "technical validation report status/subject differs")
    runtime_evidence = technical.get("checks", {}).get(
        "runtime_fingerprint", {}).get("evidence", {})
    require(is_sha256(runtime_evidence.get("fingerprint_sha256")),
            "technical validation fingerprint evidence differs")
    if raw.get("subject") == "exact-subject":
        require(technical.get("semantic_release_eligible") is True,
                "exact technical report is not semantic-release eligible")
    return {
        "case": {"path": str(case_path), "sha256": file_sha256(case_path)},
        "manifest": {"path": str(manifest_path),
                     "sha256": file_sha256(manifest_path)},
        "reviews": [{"path": str(path), "sha256": file_sha256(path)}
                    for path, _ in reviews],
        "preregistration": {
            "path": str(prereg_path), "sha256": file_sha256(prereg_path)},
        "technical_validation_report": {
            "path": str(technical_path), "sha256": file_sha256(technical_path),
            "fingerprint_sha256": runtime_evidence["fingerprint_sha256"],
            "subject": technical["subject"],
        },
    }


def verify_runtime(raw: Mapping[str, Any], *,
                   binding_evidence: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    fingerprint = raw.get("runtime_fingerprint")
    require(isinstance(fingerprint, Mapping), "runtime fingerprint is absent")
    subject = fingerprint.get("subject_spec")
    require(isinstance(subject, Mapping) and subject.get("key") in SUBJECTS,
            "runtime subject differs")
    key = str(subject["key"])
    expected = SUBJECTS[key]
    require(fingerprint.get("requested_model") == expected["model_id"] and
            fingerprint.get("requested_revision") == expected["revision"] and
            fingerprint.get("dtype") == "torch.bfloat16" and
            fingerprint.get("attention_backend") == "eager" and
            fingerprint.get("geometry") == expected["geometry"],
            "runtime model/revision/dtype/backend/geometry differs")
    declared = fingerprint.get("fingerprint_sha256")
    payload = {name: value for name, value in fingerprint.items()
               if name != "fingerprint_sha256"}
    require(declared == hashlib.sha256(canonical_bytes(payload)).hexdigest(),
            "runtime fingerprint commitment differs")
    technical = binding_evidence["technical_validation_report"]
    require(raw.get("subject") == key == technical.get("subject") and
            declared == technical.get("fingerprint_sha256"),
            "runtime subject/fingerprint differs from technical release")
    special_ids = fingerprint.get("protocol_tokenizer_attestation", {}).get(
        "all_special_ids")
    require(isinstance(special_ids, list) and all(
        isinstance(value, int) for value in special_ids),
        "runtime tokenizer special IDs are absent")
    return key, dict(fingerprint)


def score_means(record: Any, label: str) -> dict[str, Any]:
    require(isinstance(record, Mapping), f"{label} score is absent")
    correct = record.get("correct")
    counter = record.get("counterfactual")
    require(isinstance(correct, Mapping) and isinstance(counter, Mapping),
            f"{label} target records are absent")
    correct_ids = correct.get("target_token_ids")
    counter_ids = counter.get("target_token_ids")
    require(isinstance(correct_ids, list) and bool(correct_ids) and
            isinstance(counter_ids, list) and bool(counter_ids),
            f"{label} target IDs are absent")
    correct_mean = decode_float32_bits(
        correct.get("mean_logprob_float32_bits"), f"{label} correct mean")
    counter_mean = decode_float32_bits(
        counter.get("mean_logprob_float32_bits"), f"{label} counterfactual mean")
    for target_name, target in (("correct", correct), ("counterfactual", counter)):
        bits = target.get("token_logprob_float32_bits")
        require(isinstance(bits, list) and len(bits) == len(
            target["target_token_ids"]), f"{label} {target_name} token coverage differs")
        for index, value in enumerate(bits):
            decode_float32_bits(value, f"{label} {target_name} token {index}")
    return {
        "correct_ids": correct_ids, "counter_ids": counter_ids,
        "correct_mean": correct_mean, "counter_mean": counter_mean,
        "margin": float32(correct_mean - counter_mean),
    }


def generation_ok(record: Mapping[str, Any], target_ids: Sequence[int],
                  special_ids: set[int]) -> bool:
    generation = record.get("generation")
    if not isinstance(generation, Mapping):
        return False
    content = generation.get("content_ids")
    eos = generation.get("eos_ids")
    return (
        isinstance(content, list) and bool(content) and
        content[:len(target_ids)] == list(target_ids) and
        generation.get("stop_reason") == "model_eos" and
        generation.get("cap_hit") is False and isinstance(eos, list) and
        generation.get("stop_candidate_id") in eos and
        not special_ids.intersection(int(value) for value in content)
    )


def verify_scores(raw: Mapping[str, Any], special_ids: set[int]) -> dict[str, Any]:
    scores = raw.get("scores")
    require(isinstance(scores, Mapping) and set(scores) == EXPECTED_SCORE_KEYS,
            "Phase-A requires exactly six scores")
    values = {name: score_means(record, name) for name, record in scores.items()}
    competency = {
        "A_C_focal_margin_positive": values["A_C_focal"]["margin"] > 0,
        "A_W_focal_margin_negative": values["A_W_focal"]["margin"] < 0,
        "A_C_nonfocal_margin_positive": values["A_C_nonfocal"]["margin"] > 0,
        "A_W_nonfocal_margin_positive": values["A_W_nonfocal"]["margin"] > 0,
        "A_C_focal_generation": generation_ok(
            scores["A_C_focal"], values["A_C_focal"]["correct_ids"], special_ids),
        "A_W_focal_generation": generation_ok(
            scores["A_W_focal"], values["A_W_focal"]["counter_ids"], special_ids),
        "A_C_nonfocal_generation": generation_ok(
            scores["A_C_nonfocal"], values["A_C_nonfocal"]["correct_ids"],
            special_ids),
        "A_W_nonfocal_generation": generation_ok(
            scores["A_W_nonfocal"], values["A_W_nonfocal"]["correct_ids"],
            special_ids),
    }
    ac = values["A_C_focal"]
    ff = values["FF_focal"]
    damage = {
        "margin": ac["margin"] - ff["margin"],
        "correct_target_logprob": ac["correct_mean"] - ff["correct_mean"],
        "positive_margin_damage": ac["margin"] - ff["margin"] > 0,
        "positive_correct_target_damage":
            ac["correct_mean"] - ff["correct_mean"] > 0,
    }
    return {"values": values, "competency": competency,
            "fresh_damage_diagnostic": damage,
            "estimand_adequate": all(competency.values())}


def verify_forced_support(raw: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    support = raw.get("forced_carrier_support")
    require(isinstance(support, Mapping) and set(support) == {"C_N", "W_N"},
            "forced carrier support branch set differs")
    evidence = load_object(
        repo_root / "data/coherent_canary_v12/fixed_text_token_evidence_v2.json")
    expected_ids = evidence["texts"]["engineered_carrier_content"]["token_ids"]
    results = {}
    for branch in ("C_N", "W_N"):
        row = support[branch]
        require(isinstance(row, Mapping) and row.get("token_ids") == expected_ids,
                f"{branch} forced carrier token IDs differ")
        bits = row.get("token_logprob_float32_bits")
        require(isinstance(bits, list) and len(bits) == len(expected_ids),
                f"{branch} forced carrier support coverage differs")
        logprobs = [decode_float32_bits(value, f"{branch} carrier token {index}")
                    for index, value in enumerate(bits)]
        results[branch] = {"token_count": len(bits),
                           "mean_nll": -sum(logprobs) / len(logprobs)}
    require(support["C_N"]["token_ids"] == support["W_N"]["token_ids"],
            "C/W forced carrier token IDs differ")
    return results


def validate_phase_a_raw(raw_path: Path, *, repo_root: Path) -> dict[str, Any]:
    raw = load_object(raw_path)
    phase = raw.get("phase_a")
    structural_failures = []
    checks: dict[str, Any] = {}

    def structural(name: str, function):
        try:
            checks[name] = {"passed": True, "evidence": function()}
        except (PhaseAValidationError, KeyError, TypeError, ValueError) as exc:
            checks[name] = {"passed": False, "reason": str(exc)}
            structural_failures.append(name)

    structural("schema_and_blinding", lambda: (
        require(raw.get("schema") == RUN_SCHEMA and
                raw.get("design_id") == DESIGN_ID,
                "Phase-A run schema/design differs"),
        require(raw.get("subject") in SUBJECTS and
                isinstance(raw.get("case_id"), str),
                "Phase-A subject/case differs"),
        require(raw.get("treatment_scores_present") is False,
                "Phase-A run declares treatment scores"),
        require(isinstance(phase, Mapping) and
                phase.get("schema") == PHASE_SCHEMA and
                phase.get("design_id") == DESIGN_ID and
                phase.get("case_id") == raw.get("case_id") and
                phase.get("treatment_scores_present") is False,
                "nested Phase-A schema/design/case/blinding differs"),
        require(set(phase) == {
            "schema", "design_id", "case_id", "plans", "executions",
            "forced_carrier_support", "scores", "visible_messages",
            "treatment_scores_present",
        }, "nested Phase-A field set differs"),
        require(set(phase.get("executions", {})) == {"C_N", "W_N", "F"},
                "Phase-A execution set differs"),
        reject_treatment_fields(raw),
    ))
    binding_holder: dict[str, Any] = {}

    def binding_check():
        evidence = verify_bindings(raw, repo_root)
        binding_holder.update(evidence)
        return evidence

    structural("bindings", binding_check)
    runtime_holder: dict[str, Any] = {}

    def runtime_check():
        require(bool(binding_holder), "binding evidence is unavailable")
        subject, fingerprint = verify_runtime(
            raw, binding_evidence=binding_holder)
        runtime_holder["subject"] = subject
        runtime_holder["fingerprint"] = fingerprint
        return {"subject": subject,
                "fingerprint_sha256": fingerprint["fingerprint_sha256"]}

    structural("runtime_fingerprint", runtime_check)
    structural("forced_carrier_support", lambda: verify_forced_support(
        phase if isinstance(phase, Mapping) else {}, repo_root))

    score_result = None
    if not structural_failures:
        try:
            special_ids = set(runtime_holder["fingerprint"][
                "protocol_tokenizer_attestation"]["all_special_ids"])
            score_result = verify_scores(phase, special_ids)
            checks["scores"] = {"passed": True, "evidence": score_result}
        except (PhaseAValidationError, KeyError, TypeError, ValueError) as exc:
            checks["scores"] = {"passed": False, "reason": str(exc)}
            structural_failures.append("scores")

    if structural_failures:
        status = "INVALID_TECHNICAL"
        inadequacy = []
    else:
        assert score_result is not None
        inadequacy = [name for name, passed in
                      score_result["competency"].items() if not passed]
        status = "PRETREATMENT_PASS" if not inadequacy else "ESTIMAND_INADEQUATE"
    subject = runtime_holder.get("subject")
    return {
        "schema": REPORT_SCHEMA,
        "design_id": DESIGN_ID,
        "case_id": raw.get("case_id"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_artifact": {"path": str(raw_path), "sha256": file_sha256(raw_path)},
        "subject": subject,
        "status": status,
        "semantic_release_eligible": status == "PRETREATMENT_PASS" and
                                     subject == "exact-subject",
        "checks": checks,
        "fresh_damage_diagnostic": (
            None if score_result is None else
            score_result["fresh_damage_diagnostic"]),
        "inadequacy_reasons": inadequacy,
        "invalidity_reasons": structural_failures,
        "runner_status_and_decimal_margins_ignored": True,
        "limitations": [
            "No subject-model forward is rerun by this validator.",
            "Fresh damage is diagnostic and never gates Phase-A eligibility.",
            "A local-apparatus pass cannot release semantic treatment.",
        ],
    }


def write_unique_json(path: Path, value: Mapping[str, Any]) -> None:
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


def default_output(repo_root: Path, case_id: str, subject: str | None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (repo_root / "results/coherent_canary_validation" /
            f"coherent_canary_v12_phase_a_{case_id}_{subject or 'unknown'}_{stamp}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_phase_a_raw(args.raw, repo_root=args.repo_root)
    output = args.output or default_output(
        args.repo_root, str(report.get("case_id")), report.get("subject"))
    write_unique_json(output, report)
    print(f"{report['status']} -> {output}")


if __name__ == "__main__":
    main()
