from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_coherent_canary_v12_technical.py"
SPEC = importlib.util.spec_from_file_location(
    "run_coherent_canary_v12_technical", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

VALIDATOR_SCRIPT = ROOT / "scripts" / "validate_coherent_canary_v12_technical.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_coherent_canary_v12_technical_pipeline", VALIDATOR_SCRIPT)
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
assert VALIDATOR_SPEC.loader is not None
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)


def bits(value: float) -> str:
    return struct.pack("<f", value).hex()


def full_fingerprint() -> dict:
    expected = VALIDATOR.SUBJECTS["local-apparatus"]
    value = {
        "requested_model": expected["model_id"],
        "requested_revision": expected["revision"],
        "dtype": "torch.bfloat16", "attention_backend": "eager",
        "eos_ids": [9],
        "geometry": expected["geometry"],
        "attention_layers": [{"layer": index, "backend": "eager"}
                             for index in range(expected["geometry"]["layers"])],
        "subject_spec": {"key": "local-apparatus",
                         "model_id": expected["model_id"],
                         "revision": expected["revision"]},
        "parameter_devices": ["cpu"], "training": False,
        "all_parameters_frozen": True,
        "loading_info": {key: [] for key in (
            "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")},
        "model_inventory_sha256": "1" * 64,
        "protocol_tokenizer_inventory_sha256": "2" * 64,
        "weight_tensors_sha256": "3" * 64,
        "loaded_parameter_topology_sha256": "4" * 64,
        "parameter_count": 10,
        "protocol_tokenizer_attestation": {"all_special_ids": [9]},
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
        VALIDATOR.canonical_bytes(value)).hexdigest()
    return value


def identity_branch() -> dict:
    return {
        "prefix": {
            "token_ids": [10], "logical_positions": [0],
            "physical_positions": [0], "calls": [{"kind": "prefill"}],
            "row_hashes": [{"layer": 0}], "last_logits_sha256": "a" * 64,
            "physical_end": 1, "logical_end": 1,
        },
        "generation": {
            "content_ids": [20], "logical_positions": [1],
            "physical_positions": [1],
            "token_logprob_float32_bits": [bits(-0.1)],
            "stop_candidate_id": 9,
            "stop_candidate_logprob_float32_bits": bits(-0.2),
            "eos_ids": [9], "stop_reason": "model_eos", "cap_hit": False,
            "decoded_content": "record",
        },
        "content_row_hashes": [{"layer": 0}],
        "content_start": 1, "content_end": 2,
    }


def replacement_result() -> dict:
    direct = [{"layer": 0}]
    flags = {"K+V": (True, True), "K-only": (True, False),
             "V-only": (False, True)}
    rows = []
    for region in VALIDATOR.REGIONS:
        for mode in VALIDATOR.MODES:
            rows.append({
                "region": region, "mode": mode, "status": "PASS",
                "continued_snapshot_hashes": direct,
                "continued_last_logits_sha256": "c" * 64,
                "boundary_before_hashes": direct,
                "boundary_after_hashes": direct,
                "continuation_calls": [{"kind": "prefill"}],
                "insertion": {"use_keys": flags[mode][0],
                              "use_values": flags[mode][1]},
            })
    return {"status": "PASS", "direct_snapshot_hashes": direct,
            "direct_last_logits_sha256": "c" * 64, "regions": rows}


def target(token_id: int, value: float) -> dict:
    return {"target_token_ids": [token_id],
            "token_logprob_float32_bits": [bits(value)],
            "mean_logprob_float32_bits": bits(value)}


def natural_result() -> dict:
    def cell(margin, first):
        return {"correct": target(1, margin),
                "counterfactual": target(2, 0.0),
                "generation": {"content_ids": [first],
                               "stop_reason": "model_eos", "cap_hit": False,
                               "stop_candidate_id": 9, "eos_ids": [9]}}
    return {"status": "PASS", "raw": {
        "A_g": cell(1.0, 1), "A_a": cell(-1.0, 2), "F": cell(0.0, 1),
        "T_g": cell(0.6, 1), "T_a": cell(-0.6, 2)}}


def write(path: Path, value) -> Path:
    if isinstance(value, dict):
        path.write_text(json.dumps(value) + "\n")
    else:
        path.write_text(str(value))
    return path


def arguments(tmp_path: Path):
    prereg = write(tmp_path / "prereg.md", "frozen text\n")
    identity = write(tmp_path / "identity.json", {"messages": []})
    technical = write(tmp_path / "technical.json", {
        "middle_end_msg": 3,
        "correct": [{"role": "system", "content": "x"}],
        "wrong": [{"role": "system", "content": "y"}],
    })
    carrier = write(tmp_path / "carrier.json", {"carrier": "fixed"})
    return MODULE.parse_args([
        "--subject", "local-apparatus",
        "--repo", str(tmp_path),
        "--output-dir", str(tmp_path / "results"),
        "--prereg", str(prereg),
        "--identity-fixture", str(identity),
        "--technical-fixture", str(technical),
        "--carrier-fixture", str(carrier),
        "--hourly-cost-usd", "1.25",
    ])


def fake_result():
    return SimpleNamespace(
        calls=[{"call_id": 0, "kind": "prefill"}],
        executed_token_ids=[1, 2],
        logical_positions=[10, 11],
        physical_positions=[0, 1],
        physical_end=2,
        logical_end=12,
        snapshot="snapshot",
        last_logits="logits",
    )


def install_success_mocks(monkeypatch, *, path_status="PASS", natural_status="PASS",
                          events=None):
    def prepare(**kwargs):
        if events is not None:
            events.append("prepare")
        assert kwargs["spec"].key == "local-apparatus"
        fingerprint = full_fingerprint()
        return {
            "model": "model", "tokenizer": "tokenizer",
            "runtime_fingerprint": fingerprint,
            "repository": {"status": "FROZEN"},
            "dependencies": {"torch": "pinned"},
            "device": {"device_type": "cpu"},
            "model_snapshot": Path("/cache/model"),
            "protocol_tokenizer_snapshot": Path("/cache/tokenizer"),
            "model_inventory": {"sha256": "a" * 64},
            "protocol_tokenizer_inventory": {"sha256": "b" * 64},
            "model_snapshot_contract": {"status": "FROZEN"},
        }

    monkeypatch.setattr(MODULE, "prepare_pinned_subject", prepare)
    monkeypatch.setattr(MODULE, "build_role_native_plan", lambda *a, **k: "N-plan")
    monkeypatch.setattr(MODULE, "build_fresh_destination_plan",
                        lambda *a, **k: "fresh-plan")
    monkeypatch.setattr(MODULE, "execute_replay_plan", lambda *a, **k: fake_result())
    monkeypatch.setattr(MODULE, "execute_fresh_plan", lambda *a, **k: fake_result())
    monkeypatch.setattr(MODULE, "snapshot_hashes",
                        lambda snapshot: [{"snapshot_sha256": "c" * 64}])
    monkeypatch.setattr(MODULE, "tensor_sha256", lambda tensor: "d" * 64)
    monkeypatch.setattr(MODULE, "run_generated_forced_identity",
                        lambda *a, **k: {
                            "status": "PASS", "repeat_count": 2,
                            "separate_branches": [
                                {"generated": identity_branch(),
                                 "forced": identity_branch()},
                                {"generated": identity_branch(),
                                 "forced": identity_branch()},
                            ]})
    monkeypatch.setattr(MODULE, "run_fresh_self_replacement",
                        lambda *a, **k: replacement_result())
    monkeypatch.setattr(MODULE, "run_path_control", lambda *a, **k: {
        "status": path_status,
        "gradient_baseline": {"baseline_margin_float32_bits": bits(0.0)},
        "attempts": [{
            "ulp_count": count,
            "plus": {"margin_float32_bits": bits(
                0.0002 if path_status == "PASS" and count == 1 else 0.0),
                "insertion": {}},
            "minus": {"margin_float32_bits": bits(
                -0.0002 if path_status == "PASS" and count == 1 else 0.0),
                "insertion": {}},
            "plus_row_diagnostics": [{}], "minus_row_diagnostics": [{}],
        } for count in ([1] if path_status == "PASS" else
                        [1, 2, 4, 8, 16, 32, 64])],
    })
    monkeypatch.setattr(MODULE, "run_natural_calibration", lambda *a, **k: {
        **natural_result(), "status": natural_status})


def test_success_writes_complete_lean_raw_record_and_prints_before_load(
        tmp_path, monkeypatch):
    events = []
    install_success_mocks(monkeypatch, events=events)
    real_print = print

    def observed_print(*args, **kwargs):
        events.append("print")
        real_print(*args, **kwargs)

    monkeypatch.setattr("builtins.print", observed_print)
    output, document, code = MODULE.run(arguments(tmp_path))
    persisted = json.loads(output.read_text())
    assert events.index("print") < events.index("prepare")
    assert code == 0
    assert document["status"] == persisted["status"] == "PASS"
    assert persisted["subject"] == "local-apparatus"
    assert set(persisted["bindings"]) == {
        "preregistration", "identity_fixture", "technical_fixture",
        "carrier_fixture"}
    assert all(len(row["sha256"]) == 64 for row in persisted["bindings"].values())
    for label in ("correct_history_N", "fresh_destination"):
        repeat = persisted["deterministic_repeats"][label]
        assert repeat["repeat_count"] == 2
        assert repeat["records"][0] == repeat["records"][1]
        assert repeat["records"][0]["token_ids"] == [1, 2]
        assert repeat["records"][0]["logical_positions"] == [10, 11]
        assert repeat["records"][0]["physical_positions"] == [0, 1]
    assert len(persisted["fresh_self_replacement"]["regions"]) == 9
    assert len(persisted["path_control"]["attempts"]) == 1
    assert set(persisted["natural_calibration"]["raw"]) == {
        "A_g", "A_a", "F", "T_g", "T_a"}
    assert persisted["estimated_cost_usd"] is not None
    assert "runtime_fingerprint" in persisted["provenance"]
    monkeypatch.setattr(VALIDATOR, "SOURCE_PATHS", {
        "preregistration": "prereg.md",
        "identity_fixture": "identity.json",
        "technical_fixture": "technical.json",
        "fixed_text_evidence": "carrier.json",
    })
    validated = VALIDATOR.validate_technical_raw(output, repo_root=tmp_path)
    assert validated["status"] == "PASS"
    assert validated["semantic_release_eligible"] is False


def test_path_failure_is_saved_and_skips_natural_calibration(tmp_path, monkeypatch):
    install_success_mocks(monkeypatch, path_status="FAIL")
    calls = []
    monkeypatch.setattr(MODULE, "run_natural_calibration",
                        lambda *a, **k: calls.append(True))
    output, document, code = MODULE.run(arguments(tmp_path))
    assert code == 2
    assert document["status"] == "FAIL"
    assert document["natural_calibration"]["status"] == \
        "SKIPPED_PATH_CONTROL_FAIL"
    assert not calls
    assert json.loads(output.read_text())["status"] == "FAIL"


def test_ordinary_exception_is_atomically_saved(tmp_path, monkeypatch):
    install_success_mocks(monkeypatch)

    def fail(*args, **kwargs):
        raise ValueError("identity exploded")

    monkeypatch.setattr(MODULE, "run_generated_forced_identity", fail)
    output, document, code = MODULE.run(arguments(tmp_path))
    persisted = json.loads(output.read_text())
    assert code == 1
    assert document["status"] == persisted["status"] == "ERROR"
    assert persisted["error"]["type"] == "ValueError"
    assert persisted["error"]["message"] == "identity exploded"
    assert "ValueError: identity exploded" in persisted["error"]["traceback"]


def test_repeat_difference_is_caught_and_saved(tmp_path, monkeypatch):
    install_success_mocks(monkeypatch)
    count = {"value": 0}

    def differing(*args, **kwargs):
        result = fake_result()
        count["value"] += 1
        result.executed_token_ids = [count["value"]]
        return result

    monkeypatch.setattr(MODULE, "execute_replay_plan", differing)
    output, document, code = MODULE.run(arguments(tmp_path))
    assert code == 1
    assert document["status"] == "ERROR"
    assert "deterministic repeat differs" in document["error"]["message"]
    assert json.loads(output.read_text())["status"] == "ERROR"


def test_subject_is_explicit_and_limited_to_two_frozen_choices():
    with pytest.raises(SystemExit):
        MODULE.parse_args([])
    with pytest.raises(SystemExit):
        MODULE.parse_args(["--subject", "other"])
    assert MODULE.parse_args(["--subject", "exact-subject"]).subject == \
        "exact-subject"
