from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_coherent_canary_v12_technical.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_coherent_canary_v12_technical", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def bits(value: float) -> str:
    return struct.pack("<f", value).hex()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_bindings(repo: Path) -> dict:
    rows = {}
    for key, relative in MODULE.SOURCE_PATHS.items():
        path = repo / relative
        write_json(path, {"design_id": MODULE.DESIGN_ID, "key": key})
        rows[key] = {"path": relative, "sha256": MODULE.file_sha256(path)}
    return rows


def fingerprint(subject="exact-subject") -> dict:
    expected = MODULE.SUBJECTS[subject]
    value = {
        "requested_model": expected["model_id"],
        "requested_revision": expected["revision"],
        "dtype": "torch.bfloat16",
        "attention_backend": "eager",
        "geometry": expected["geometry"],
        "attention_layers": [
            {"layer": index, "backend": "eager"}
            for index in range(expected["geometry"]["layers"])
        ],
        "subject_spec": {
            "key": subject, "model_id": expected["model_id"],
            "revision": expected["revision"],
        },
        "parameter_devices": expected["devices"],
        "training": False,
        "all_parameters_frozen": True,
        "loading_info": {key: [] for key in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")},
        "model_inventory_sha256": "1" * 64,
        "protocol_tokenizer_inventory_sha256": "2" * 64,
        "weight_tensors_sha256": "3" * 64,
        "loaded_parameter_topology_sha256": "4" * 64,
        "parameter_count": 123,
        "protocol_tokenizer_attestation": {
            "all_special_ids": [9],
        },
        "release_binding": {
            "apparatus_commit": "a" * 40,
            "authorization_commit": "b" * 40,
            "authorization_sha256": "5" * 64,
            "sealed_inventory_sha256": "6" * 64,
            "model_snapshot_contract_sha256": "7" * 64,
            "subject_contract_entry_sha256": "8" * 64,
            "protocol_tokenizer_contract_entry_sha256": "9" * 64,
        },
    }
    value["fingerprint_sha256"] = hashlib.sha256(
        MODULE.canonical_bytes(value)).hexdigest()
    return value


def identity_record() -> dict:
    return {
        "prefix": {
            "token_ids": [10, 11],
            "logical_positions": [0, 1],
            "physical_positions": [0, 1],
            "calls": [{"kind": "prefill", "start": 0, "end": 2}],
            "row_hashes": [{"layer": 0, "k": "a", "v": "b"}],
            "last_logits_sha256": "a" * 64,
            "physical_end": 2, "logical_end": 2,
        },
        "generation": {
            "content_ids": [20, 21],
            "logical_positions": [2, 3],
            "physical_positions": [2, 3],
            "token_logprob_float32_bits": [bits(-0.1), bits(-0.2)],
            "stop_candidate_id": 99,
            "stop_candidate_logprob_float32_bits": bits(-0.3),
            "eos_ids": [99],
            "stop_reason": "model_eos",
            "cap_hit": False,
            "decoded_content": "neutral record",
        },
        "content_row_hashes": [{"layer": 0, "k": "c", "v": "d"}],
        "content_start": 2, "content_end": 4,
    }


def deterministic_record() -> dict:
    return {
        "token_ids": [1, 2], "logical_positions": [0, 1],
        "physical_positions": [0, 1],
        "calls": [{"kind": "prefill"}],
        "snapshot_hashes": [{"layer": 0, "k": "a", "v": "b"}],
        "last_logits_sha256": "b" * 64,
    }


def self_replacement() -> dict:
    direct = [{"layer": 0, "k": "a", "v": "b"}]
    records = []
    flags = {"K+V": (True, True), "K-only": (True, False),
             "V-only": (False, True)}
    for region in MODULE.REGIONS:
        for mode in MODULE.MODES:
            records.append({
                "region": region, "mode": mode,
                "continued_snapshot_hashes": direct,
                "continued_last_logits_sha256": "c" * 64,
                "boundary_before_hashes": direct,
                "boundary_after_hashes": direct,
                "continuation_calls": [{"kind": "prefill"}],
                "insertion": {"use_keys": flags[mode][0],
                              "use_values": flags[mode][1]},
                "status": "RUNNER_LABEL_IGNORED",
            })
    return {"direct_snapshot_hashes": direct,
            "direct_last_logits_sha256": "c" * 64,
            "regions": records, "status": "RUNNER_LABEL_IGNORED"}


def path_control() -> dict:
    return {
        "gradient_baseline": {"baseline_margin_float32_bits": bits(0.0)},
        "chosen_ulp_count": 2,
        "status": "RUNNER_LABEL_IGNORED",
        "attempts": [
            {
                "ulp_count": 1,
                "plus": {"margin_float32_bits": bits(0.00005),
                         "insertion": {}},
                "minus": {"margin_float32_bits": bits(-0.00005),
                          "insertion": {}},
                "plus_row_diagnostics_summary": {
                    "row_count": 1, "selected_count": 1, "sha256": "a" * 64},
                "minus_row_diagnostics_summary": {
                    "row_count": 1, "selected_count": 1, "sha256": "b" * 64},
                "passes": True,
            },
            {
                "ulp_count": 2,
                "plus": {"margin_float32_bits": bits(0.0002),
                         "insertion": {}},
                "minus": {"margin_float32_bits": bits(-0.0002),
                          "insertion": {}},
                "plus_row_diagnostics_summary": {
                    "row_count": 1, "selected_count": 1, "sha256": "a" * 64},
                "minus_row_diagnostics_summary": {
                    "row_count": 1, "selected_count": 1, "sha256": "b" * 64},
                "passes": False,
            },
        ],
    }


def target_record(token_id: int, value: float) -> dict:
    return {"target_token_ids": [token_id],
            "token_logprob_float32_bits": [bits(value)],
            "mean_logprob_float32_bits": bits(value)}


def natural_cell(margin: float, first_token: int) -> dict:
    return {
        "correct": target_record(1, margin),
        "counterfactual": target_record(2, 0.0),
        "generation": {
            "content_ids": [first_token], "stop_reason": "model_eos",
            "cap_hit": False, "stop_candidate_id": 9, "eos_ids": [9],
        },
        "margin": -999.0,
    }


def natural() -> dict:
    return {
        "status": "RUNNER_LABEL_IGNORED",
        "raw": {
            "A_g": natural_cell(1.0, 1),
            "A_a": natural_cell(-1.0, 2),
            "F": natural_cell(0.0, 1),
            "T_g": natural_cell(0.6, 1),
            "T_a": natural_cell(-0.6, 2),
        },
    }


def raw_fixture(repo: Path, *, subject="exact-subject") -> tuple[Path, dict]:
    identity = identity_record()
    repeat = deterministic_record()
    raw = {
        "schema": MODULE.RAW_SCHEMA, "design_id": MODULE.DESIGN_ID,
        "source_bindings": source_bindings(repo),
        "runtime_fingerprint": fingerprint(subject),
        "generated_forced_identity": {"status": "LIE", "separate_branches": [
            {"generated": deepcopy(identity), "forced": deepcopy(identity)},
            {"generated": deepcopy(identity), "forced": deepcopy(identity)},
        ]},
        "deterministic_repeats": {
            "correct_history_N": {"status": "LIE", "records": [
                repeat, deepcopy(repeat)]},
            "fresh_destination": {"status": "LIE", "records": [
                deepcopy(repeat), deepcopy(repeat)]},
        },
        "fresh_self_replacement": self_replacement(),
        "path_control": path_control(),
        "natural_calibration": natural(),
        "status": "RUNNER_LIES_FAIL",
    }
    path = repo / "raw.json"
    write_json(path, raw)
    return path, raw


def test_exact_valid_raw_passes_and_ignores_runner_labels(tmp_path):
    raw_path, _ = raw_fixture(tmp_path)
    report = MODULE.validate_technical_raw(raw_path, repo_root=tmp_path)
    assert report["status"] == "PASS"
    assert report["semantic_release_eligible"] is True
    assert report["path_control"]["chosen_ulp_count"] == 2
    assert report["natural_calibration"]["status"] == "PASS"
    assert report["runner_status_labels_ignored"] is True


def test_local_pass_is_not_semantic_release_eligible(tmp_path):
    raw_path, _ = raw_fixture(tmp_path, subject="local-apparatus")
    report = MODULE.validate_technical_raw(raw_path, repo_root=tmp_path)
    assert report["status"] == "PASS"
    assert report["semantic_release_eligible"] is False


def test_little_endian_bits_and_first_path_pass_are_authoritative(tmp_path):
    assert MODULE.decode_float32_bits("cdccccbd", "known") == pytest.approx(-0.1)
    raw_path, raw = raw_fixture(tmp_path)
    raw["path_control"]["chosen_ulp_count"] = 1
    raw["path_control"]["attempts"][0]["passes"] = True
    write_json(raw_path, raw)
    report = MODULE.validate_technical_raw(raw_path, repo_root=tmp_path)
    assert report["status"] == "PASS"
    assert report["path_control"]["chosen_ulp_count"] == 2


def test_identity_or_self_replacement_coverage_error_fails(tmp_path):
    raw_path, raw = raw_fixture(tmp_path)
    raw["generated_forced_identity"]["separate_branches"][0]["forced"][
        "generation"]["content_ids"] = [999]
    raw["fresh_self_replacement"]["regions"].pop()
    write_json(raw_path, raw)
    report = MODULE.validate_technical_raw(raw_path, repo_root=tmp_path)
    assert report["status"] == "FAIL"
    assert "generated_forced_identity" in report["failures"]
    assert "fresh_self_replacement" in report["failures"]


def test_natural_adverse_is_reported_but_not_a_plumbing_failure(tmp_path):
    raw_path, raw = raw_fixture(tmp_path)
    raw["natural_calibration"]["raw"]["T_g"]["correct"] = target_record(1, 0.1)
    write_json(raw_path, raw)
    report = MODULE.validate_technical_raw(raw_path, repo_root=tmp_path)
    assert report["natural_calibration"]["status"] == "ADVERSE"
    assert report["status"] == "PASS"
    assert report["semantic_release_eligible"] is True


def test_natural_malformed_is_invalid_and_fails_technical_record(tmp_path):
    raw_path, raw = raw_fixture(tmp_path)
    del raw["natural_calibration"]["raw"]["F"]
    write_json(raw_path, raw)
    report = MODULE.validate_technical_raw(raw_path, repo_root=tmp_path)
    assert report["natural_calibration"]["status"] == "INVALID"
    assert report["status"] == "FAIL"


def test_source_hash_mismatch_and_unique_output_fail_closed(tmp_path):
    raw_path, raw = raw_fixture(tmp_path)
    raw["source_bindings"]["technical_fixture"]["sha256"] = "0" * 64
    write_json(raw_path, raw)
    report = MODULE.validate_technical_raw(raw_path, repo_root=tmp_path)
    assert report["status"] == "FAIL"
    output = tmp_path / "result.json"
    MODULE.write_unique_json(output, report)
    with pytest.raises(FileExistsError):
        MODULE.write_unique_json(output, report)
