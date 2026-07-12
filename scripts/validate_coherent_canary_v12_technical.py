#!/usr/bin/env python3
"""Lean independent validator for the v12 technical-control artifact.

This validator intentionally imports no model-facing ``coherent_canary_*``
module.  It is meant to catch plausible serialization, coverage, arithmetic,
and branch-selection mistakes without rerunning a subject model.  Runtime
``status``/``passes`` labels and decimal summaries are ignored.

Required raw sections are documented by the focused synthetic tests in
``tests/test_validate_coherent_canary_v12_technical.py``.
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
from typing import Any, Mapping


DESIGN_ID = "coherent-state-decision-canary-v12"
RAW_SCHEMA = "coherent_state_decision_canary_v12_technical_raw_v1"
REPORT_SCHEMA = "coherent_state_decision_canary_v12_technical_validation_v1"
MIN_PATH_MOVEMENT = 1e-4
ULP_COUNTS = (1, 2, 4, 8, 16, 32, 64)
REGIONS = ("R1_content", "R2_boundary", "R3_anchor")
MODES = ("K+V", "K-only", "V-only")
NATURAL_CELLS = {"A_g", "A_a", "F", "T_g", "T_a"}
SOURCE_PATHS = {
    "preregistration":
        "COHERENT-STATE-DECISION-CANARY-V12-PREREGISTRATION.md",
    "identity_fixture":
        "data/coherent_canary_v12/generated_forced_identity_fixture.json",
    "technical_fixture":
        "data/coherent_canary_v12/technical_control_fixture.json",
    "fixed_text_evidence":
        "data/coherent_canary_v12/fixed_text_token_evidence_v2.json",
}
SUBJECTS = {
    "local-apparatus": {
        "model_id": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "geometry": {
            "layers": 28, "attention_heads": 16, "kv_heads": 8,
            "head_dim": 128, "rope_theta": 1_000_000.0,
        },
        "devices": ["cpu"],
    },
    "exact-subject": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "geometry": {
            "layers": 48, "attention_heads": 32, "kv_heads": 4,
            "head_dim": 128, "rope_theta": 10_000_000.0,
        },
        "devices": ["cuda:0"],
    },
}


class TechnicalValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TechnicalValidationError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


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
        raise TechnicalValidationError(f"invalid JSON {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


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


def verify_source_bindings(raw: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    bindings = raw.get("source_bindings")
    require(isinstance(bindings, Mapping) and set(bindings) == set(SOURCE_PATHS),
            "technical source-binding set differs")
    observed = {}
    for key, relative in SOURCE_PATHS.items():
        row = bindings[key]
        require(isinstance(row, Mapping) and row.get("path") == relative and
                is_sha256(row.get("sha256")),
                f"{key} source binding differs")
        path = repo_root / relative
        require(path.is_file(), f"required source is absent: {relative}")
        digest = file_sha256(path)
        require(digest == row["sha256"], f"{key} source hash differs")
        if key != "preregistration":
            document = load_object(path)
            require(document.get("design_id", DESIGN_ID) == DESIGN_ID,
                    f"{key} design differs")
        observed[key] = {"path": relative, "sha256": digest}
    return observed


def verify_runtime_fingerprint(raw: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    fingerprint = raw.get("runtime_fingerprint")
    require(isinstance(fingerprint, Mapping), "runtime fingerprint is absent")
    subject_spec = fingerprint.get("subject_spec")
    require(isinstance(subject_spec, Mapping), "runtime subject spec is absent")
    subject_key = subject_spec.get("key")
    require(subject_key in SUBJECTS, "runtime subject key differs")
    expected = SUBJECTS[str(subject_key)]
    require(fingerprint.get("requested_model") == expected["model_id"] and
            fingerprint.get("requested_revision") == expected["revision"],
            "runtime model/revision differs")
    require(subject_spec.get("model_id") == expected["model_id"] and
            subject_spec.get("revision") == expected["revision"],
            "runtime literal subject spec differs")
    require(fingerprint.get("dtype") == "torch.bfloat16" and
            fingerprint.get("attention_backend") == "eager" and
            fingerprint.get("geometry") == expected["geometry"],
            "runtime dtype/backend/geometry differs")
    layers = fingerprint.get("attention_layers")
    require(isinstance(layers, list) and
            [row.get("layer") for row in layers] ==
            list(range(expected["geometry"]["layers"])) and
            all(row.get("backend") == "eager" for row in layers),
            "runtime eager layer coverage differs")
    require(fingerprint.get("parameter_devices") == expected["devices"] and
            fingerprint.get("training") is False and
            fingerprint.get("all_parameters_frozen") is True,
            "runtime device/eval/frozen state differs")
    loading = fingerprint.get("loading_info")
    require(isinstance(loading, Mapping) and all(
        loading.get(key) == [] for key in
        ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
        "runtime loading information is adverse")
    for key in ("model_inventory_sha256", "protocol_tokenizer_inventory_sha256",
                "weight_tensors_sha256", "loaded_parameter_topology_sha256"):
        require(is_sha256(fingerprint.get(key)), f"runtime {key} is absent")
    require(isinstance(fingerprint.get("parameter_count"), int) and
            fingerprint["parameter_count"] > 0,
            "runtime parameter count is invalid")
    release = fingerprint.get("release_binding")
    require(isinstance(release, Mapping), "runtime release binding is absent")
    for key in ("authorization_sha256", "sealed_inventory_sha256",
                "model_snapshot_contract_sha256",
                "subject_contract_entry_sha256",
                "protocol_tokenizer_contract_entry_sha256"):
        require(is_sha256(release.get(key)), f"runtime release {key} is absent")
    for key in ("apparatus_commit", "authorization_commit"):
        value = release.get(key)
        require(isinstance(value, str) and len(value) == 40 and all(
            character in "0123456789abcdef" for character in value),
            f"runtime release {key} is invalid")
    declared = fingerprint.get("fingerprint_sha256")
    payload = {key: value for key, value in fingerprint.items()
               if key != "fingerprint_sha256"}
    require(declared == hashlib.sha256(canonical_bytes(payload)).hexdigest(),
            "runtime fingerprint commitment differs")
    return str(subject_key), dict(fingerprint)


def identity_view(record: Any, label: str) -> dict[str, Any]:
    require(isinstance(record, Mapping), f"{label} identity record is absent")
    prefix = record.get("prefix")
    generation = record.get("generation")
    rows = record.get("content_row_hashes")
    require(isinstance(prefix, Mapping) and isinstance(generation, Mapping) and
            isinstance(rows, list) and bool(rows),
            f"{label} identity branch is incomplete")
    view = {
        "prefix_token_ids": prefix.get("token_ids"),
        "prefix_logical_positions": prefix.get("logical_positions"),
        "prefix_physical_positions": prefix.get("physical_positions"),
        "prefix_calls": prefix.get("calls"),
        "prefix_row_hashes": prefix.get("row_hashes"),
        "prefix_last_logits_sha256": prefix.get("last_logits_sha256"),
        "prefix_physical_end": prefix.get("physical_end"),
        "prefix_logical_end": prefix.get("logical_end"),
        "content_ids": generation.get("content_ids"),
        "content_logical_positions": generation.get("logical_positions"),
        "content_physical_positions": generation.get("physical_positions"),
        "token_logprob_float32_bits":
            generation.get("token_logprob_float32_bits"),
        "stop_candidate_id": generation.get("stop_candidate_id"),
        "stop_candidate_logprob_float32_bits":
            generation.get("stop_candidate_logprob_float32_bits"),
        "eos_ids": generation.get("eos_ids"),
        "stop_reason": generation.get("stop_reason"),
        "cap_hit": generation.get("cap_hit"),
        "decoded_content": generation.get("decoded_content"),
        "per_layer_content_rows": rows,
        "content_start": record.get("content_start"),
        "content_end": record.get("content_end"),
    }
    content = view["content_ids"]
    bits = view["token_logprob_float32_bits"]
    require(isinstance(content, list) and bool(content) and len(content) <= 64 and
            isinstance(bits, list) and len(bits) == len(content),
            f"{label} identity content/bit coverage differs")
    for index, value in enumerate(bits):
        decode_float32_bits(value, f"{label} identity token {index}")
    decode_float32_bits(view["stop_candidate_logprob_float32_bits"],
                        f"{label} identity stop candidate")
    require(isinstance(view["decoded_content"], str) and
            bool(view["decoded_content"]),
            f"{label} identity decoded content is empty")
    require(view["content_start"] == view["prefix_physical_end"] and
            view["content_end"] == view["content_start"] + len(content),
            f"{label} identity content interval differs")
    return view


def verify_identity(raw: Mapping[str, Any], *,
                    require_normal_stop: bool) -> dict[str, Any]:
    section = raw.get("generated_forced_identity")
    require(isinstance(section, Mapping), "generated/forced identity is absent")
    repeats = section.get("separate_branches")
    require(isinstance(repeats, list) and len(repeats) == 2,
            "identity requires exactly two repeats")
    common = []
    normal_stops = []
    for index, repeat in enumerate(repeats):
        require(isinstance(repeat, Mapping), f"identity repeat {index} is invalid")
        generated = identity_view(repeat.get("generated"), f"repeat {index} generated")
        forced = identity_view(repeat.get("forced"), f"repeat {index} forced")
        equivalence_fields = set(generated) - {"stop_reason", "cap_hit"}
        require({key: generated[key] for key in equivalence_fields} ==
                {key: forced[key] for key in equivalence_fields},
                f"identity repeat {index} generated/forced evidence differs")
        generated_normal = (
            generated["stop_reason"] == "model_eos" and
            generated["cap_hit"] is False and
            generated["stop_candidate_id"] in generated["eos_ids"])
        forced_normal = (
            forced["stop_reason"] == "model_eos" and
            forced["cap_hit"] is False and
            forced["stop_candidate_id"] in forced["eos_ids"])
        normal_stops.append(generated_normal and forced_normal)
        common.append({"generated": generated, "forced": forced})
    require(common[0] == common[1], "identity repeats differ")
    if require_normal_stop:
        require(all(normal_stops), "identity did not stop normally")
    return {
        "repeat_count": 2,
        "content_ids": common[0]["generated"]["content_ids"],
        "normal_stop": all(normal_stops),
    }


REPEAT_FIELDS = ("token_ids", "logical_positions", "physical_positions", "calls",
                 "snapshot_hashes", "last_logits_sha256")


def verify_deterministic_repeat(raw: Mapping[str, Any]) -> dict[str, Any]:
    section = raw.get("deterministic_repeats")
    require(isinstance(section, Mapping), "deterministic repeat is absent")
    observed = {}
    for label in ("correct_history_N", "fresh_destination"):
        group = section.get(label)
        require(isinstance(group, Mapping) and
                isinstance(group.get("records"), list) and
                len(group["records"]) == 2,
                f"deterministic repeat {label} coverage differs")
        views = []
        for record in group["records"]:
            require(isinstance(record, Mapping) and
                    all(field in record for field in REPEAT_FIELDS),
                    f"deterministic repeat {label} is incomplete")
            views.append({field: record[field] for field in REPEAT_FIELDS})
        require(views[0] == views[1],
                f"deterministic same-schedule repeat differs: {label}")
        observed[label] = {"token_count": len(views[0]["token_ids"])}
    return observed


def verify_self_replacement(raw: Mapping[str, Any]) -> dict[str, Any]:
    section = raw.get("fresh_self_replacement")
    require(isinstance(section, Mapping), "fresh self-replacement is absent")
    direct_rows = section.get("direct_snapshot_hashes")
    direct_logits = section.get("direct_last_logits_sha256")
    records = section.get("regions")
    require(isinstance(direct_rows, list) and bool(direct_rows) and
            is_sha256(direct_logits) and isinstance(records, list),
            "fresh self-replacement direct evidence is incomplete")
    expected = {(region, mode) for region in REGIONS for mode in MODES}
    observed = set()
    for record in records:
        require(isinstance(record, Mapping), "self-replacement record is invalid")
        key = (record.get("region"), record.get("mode"))
        require(key in expected and key not in observed,
                f"self-replacement selector is invalid/duplicate: {key}")
        observed.add(key)
        require(record.get("continued_snapshot_hashes") == direct_rows and
                record.get("continued_last_logits_sha256") == direct_logits,
                f"self-replacement continuation differs: {key}")
        require(isinstance(record.get("boundary_before_hashes"), list) and
                isinstance(record.get("boundary_after_hashes"), list) and
                record.get("continuation_calls"),
                f"self-replacement lineage is incomplete: {key}")
        insertion = record.get("insertion")
        expected_flags = {
            "K+V": (True, True), "K-only": (True, False),
            "V-only": (False, True),
        }[str(key[1])]
        require(isinstance(insertion, Mapping) and
                (insertion.get("use_keys"), insertion.get("use_values")) ==
                expected_flags,
                f"self-replacement channel flags differ: {key}")
    require(observed == expected, "self-replacement 3x3 coverage differs")
    return {"coverage": sorted(f"{region}/{mode}" for region, mode in observed)}


def verify_path_control(raw: Mapping[str, Any]) -> dict[str, Any]:
    section = raw.get("path_control")
    require(isinstance(section, Mapping), "path control is absent")
    baseline = section.get("gradient_baseline")
    require(isinstance(baseline, Mapping), "path-control baseline is absent")
    zero = decode_float32_bits(baseline.get("baseline_margin_float32_bits"),
                               "path baseline margin")
    attempts = section.get("attempts")
    require(isinstance(attempts, list) and 1 <= len(attempts) <= len(ULP_COUNTS),
            "path-control attempt coverage differs")
    require([row.get("ulp_count") for row in attempts] ==
            list(ULP_COUNTS[:len(attempts)]),
            "path-control ULP attempt order differs")
    recomputed = []
    first_pass = None
    for row in attempts:
        plus = row.get("plus")
        minus = row.get("minus")
        require(isinstance(plus, Mapping) and isinstance(minus, Mapping),
                "path-control directional score is absent")
        plus_margin = decode_float32_bits(plus.get("margin_float32_bits"),
                                          "path plus margin")
        minus_margin = decode_float32_bits(minus.get("margin_float32_bits"),
                                           "path minus margin")
        plus_move = plus_margin - zero
        minus_move = zero - minus_margin
        passed = plus_move >= MIN_PATH_MOVEMENT and minus_move >= MIN_PATH_MOVEMENT
        if passed and first_pass is None:
            first_pass = int(row["ulp_count"])
        plus_summary = row.get("plus_row_diagnostics_summary")
        minus_summary = row.get("minus_row_diagnostics_summary")
        require(isinstance(plus_summary, Mapping) and
                isinstance(minus_summary, Mapping) and
                all(isinstance(summary.get("row_count"), int) and
                    summary["row_count"] > 0 and
                    isinstance(summary.get("selected_count"), int) and
                    0 <= summary["selected_count"] <= summary["row_count"] and
                    is_sha256(summary.get("sha256"))
                    for summary in (plus_summary, minus_summary)) and
                isinstance(plus.get("insertion"), Mapping) and
                isinstance(minus.get("insertion"), Mapping),
                "path-control row/insertion evidence is incomplete")
        recomputed.append({
            "ulp_count": int(row["ulp_count"]),
            "plus_movement": plus_move, "minus_movement": minus_move,
            "passes": passed,
        })
    if first_pass is None:
        require(len(attempts) == len(ULP_COUNTS),
                "failed path control omitted frozen attempts")
    else:
        require(attempts[-1].get("ulp_count") == first_pass,
                "path control did not stop at first passing ULP count")
    return {"status": "PASS" if first_pass is not None else "FAIL",
            "chosen_ulp_count": first_pass, "attempts": recomputed}


def mean_bits(record: Any, expected_id: int, label: str) -> float:
    require(isinstance(record, Mapping) and
            record.get("target_token_ids") == [expected_id],
            f"{label} target ID differs")
    mean = record.get("mean_logprob_float32_bits")
    if mean is not None:
        return decode_float32_bits(mean, f"{label} mean")
    bits = record.get("token_logprob_float32_bits")
    require(isinstance(bits, list) and len(bits) == 1,
            f"{label} single-token bit coverage differs")
    return decode_float32_bits(bits[0], f"{label} token")


def natural_generation_ok(record: Mapping[str, Any], target_id: int,
                          special_ids: set[int]) -> bool:
    generation = record.get("generation")
    if not isinstance(generation, Mapping):
        return False
    content = generation.get("content_ids")
    eos_ids = generation.get("eos_ids")
    return (
        isinstance(content, list) and bool(content) and content[0] == target_id and
        generation.get("stop_reason") == "model_eos" and
        generation.get("cap_hit") is False and
        isinstance(eos_ids, list) and
        generation.get("stop_candidate_id") in eos_ids and
        not special_ids.intersection(int(value) for value in content)
    )


def verify_natural(raw: Mapping[str, Any]) -> dict[str, Any]:
    try:
        section = raw.get("natural_calibration")
        require(isinstance(section, Mapping), "natural calibration is absent")
        cells = section.get("raw")
        require(isinstance(cells, Mapping) and set(cells) == NATURAL_CELLS,
                "natural calibration does not contain exactly five cells")
        approve_ids = cells["A_g"].get("correct", {}).get("target_token_ids")
        deny_ids = cells["A_g"].get("counterfactual", {}).get(
            "target_token_ids")
        require(isinstance(approve_ids, list) and len(approve_ids) == 1 and
                isinstance(deny_ids, list) and len(deny_ids) == 1 and
                approve_ids != deny_ids, "natural target IDs are invalid")
        approve, deny = int(approve_ids[0]), int(deny_ids[0])
        tokenizer_attestation = raw.get("runtime_fingerprint", {}).get(
            "protocol_tokenizer_attestation", {})
        special_ids = tokenizer_attestation.get("all_special_ids")
        require(isinstance(special_ids, list) and all(
            isinstance(value, int) for value in special_ids),
            "natural special-token IDs are invalid")
        margins = {}
        for name, cell in cells.items():
            require(isinstance(cell, Mapping), f"natural cell {name} is invalid")
            correct = mean_bits(cell.get("correct"), approve,
                                f"natural {name} approve")
            wrong = mean_bits(cell.get("counterfactual"), deny,
                              f"natural {name} deny")
            margins[name] = correct - wrong
        green_denominator = margins["A_g"] - margins["F"]
        amber_denominator = margins["F"] - margins["A_a"]
        denominators_positive = green_denominator > 0 and amber_denominator > 0
        rho_green = ((margins["T_g"] - margins["F"]) / green_denominator
                     if green_denominator > 0 else None)
        rho_amber = ((margins["F"] - margins["T_a"]) / amber_denominator
                     if amber_denominator > 0 else None)
        green_generation = natural_generation_ok(
            cells["A_g"], approve, set(special_ids))
        amber_generation = natural_generation_ok(
            cells["A_a"], deny, set(special_ids))
        checks = {
            "green_oracle_margin_positive": margins["A_g"] > 0,
            "amber_oracle_margin_negative": margins["A_a"] < 0,
            "denominators_positive": denominators_positive,
            "green_generation": green_generation,
            "amber_generation": amber_generation,
            "rho_green_at_least_half": rho_green is not None and rho_green >= 0.5,
            "rho_amber_at_least_half": rho_amber is not None and rho_amber >= 0.5,
        }
        return {
            "status": "PASS" if all(checks.values()) else "ADVERSE",
            "margins": margins,
            "green_denominator": green_denominator,
            "amber_denominator": amber_denominator,
            "rho_green": rho_green, "rho_amber": rho_amber,
            "checks": checks,
        }
    except (TechnicalValidationError, KeyError, TypeError, ValueError) as exc:
        return {"status": "INVALID", "reason": str(exc)}


def validate_technical_raw(raw_path: Path, *, repo_root: Path) -> dict[str, Any]:
    raw = load_object(raw_path)
    require(raw.get("schema") == RAW_SCHEMA and raw.get("design_id") == DESIGN_ID,
            "technical raw schema/design differs")
    checks: dict[str, Any] = {}
    failures = []

    def run_check(name: str, function):
        try:
            checks[name] = {"passed": True, "evidence": function()}
        except (TechnicalValidationError, KeyError, TypeError, ValueError) as exc:
            checks[name] = {"passed": False, "reason": str(exc)}
            failures.append(name)

    run_check("source_bindings", lambda: verify_source_bindings(raw, repo_root))
    subject_holder: dict[str, Any] = {}

    def runtime_check():
        subject, fingerprint = verify_runtime_fingerprint(raw)
        subject_holder["key"] = subject
        subject_holder["fingerprint"] = fingerprint
        return {"subject": subject,
                "fingerprint_sha256": fingerprint["fingerprint_sha256"]}

    run_check("runtime_fingerprint", runtime_check)
    run_check("generated_forced_equivalence", lambda: verify_identity(
        raw, require_normal_stop=False))
    run_check("generated_forced_identity", lambda: verify_identity(
        raw, require_normal_stop=True))
    run_check("deterministic_repeat", lambda: verify_deterministic_repeat(raw))
    run_check("fresh_self_replacement", lambda: verify_self_replacement(raw))

    path_result: dict[str, Any]
    try:
        path_result = verify_path_control(raw)
        path_pass = path_result["status"] == "PASS"
        checks["path_control"] = {"passed": path_pass, "evidence": path_result}
        if not path_pass:
            failures.append("path_control")
    except (TechnicalValidationError, KeyError, TypeError, ValueError) as exc:
        path_result = {"status": "FAIL", "reason": str(exc)}
        checks["path_control"] = {"passed": False, "reason": str(exc)}
        failures.append("path_control")

    natural = verify_natural(raw)
    natural_valid = natural["status"] != "INVALID"
    checks["natural_calibration_valid"] = {
        "passed": natural_valid, "evidence": natural}
    if not natural_valid:
        failures.append("natural_calibration_valid")

    status = "PASS" if not failures else "FAIL"
    subject_key = subject_holder.get("key")
    return {
        "schema": REPORT_SCHEMA,
        "design_id": DESIGN_ID,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_artifact": {"path": str(raw_path), "sha256": file_sha256(raw_path)},
        "subject": subject_key,
        "checks": checks,
        "path_control": path_result,
        "natural_calibration": natural,
        "status": status,
        "semantic_release_eligible": status == "PASS" and
                                     subject_key == "exact-subject",
        "runner_status_labels_ignored": True,
        "failures": failures,
        "limitations": [
            "No subject-model forward is rerun by this validator.",
            "Raw logits, gradients, and cache hashes remain bound observations; "
            "the validator checks their internal consistency and arithmetic.",
            "The validator does not prove full-vocabulary argmax selection or "
            "backend-kernel correctness.",
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


def default_output(repo_root: Path, subject: str | None) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = subject or "unknown-subject"
    return (repo_root / "results/coherent_canary_validation" /
            f"coherent_canary_v12_technical_validation_{slug}_{timestamp}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--repo-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_technical_raw(args.raw, repo_root=args.repo_root)
    output = args.output or default_output(args.repo_root, report.get("subject"))
    write_unique_json(output, report)
    print(f"{report['status']} -> {output}")


if __name__ == "__main__":
    main()
