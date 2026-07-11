from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_coherent_harvest_v6",
    ROOT / "scripts/validate_coherent_harvest.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def backend_fingerprint() -> dict:
    config_row = lambda scope: {
        "scope": scope,
        "config_class": "Qwen3MoeConfig",
        "_attn_implementation": "eager",
        "_attn_implementation_internal": "eager",
        "resolved_implementation": "eager",
    }
    doc = {
        "requested_implementation": "eager",
        "model_config": config_row("model_config"),
        "text_config": config_row("text_config"),
        "text_config_is_model_config": True,
        "expected_layer_count": 48,
        "layers": [{
            "layer_index": index,
            "module_name": f"model.layers.{index}.self_attn",
            "module_class": "Qwen3MoeAttention",
            "module_config_class": "Qwen3MoeConfig",
            "module_config__attn_implementation": "eager",
            "module_config__attn_implementation_internal": "eager",
            "resolved_implementation": "eager",
        } for index in range(48)],
    }
    doc["sha256"] = MODULE._backend_payload_sha256(doc)
    return doc


def stage(raw: dict, *, coverage: int, threshold=None, aggregate=None) -> dict:
    doc = {
        "status": "PASS",
        "passes": True,
        "prerequisites": [],
        "threshold": threshold,
        "comparison": "<=" if threshold is not None else None,
        "expected_coverage": coverage,
        "observed_coverage": coverage,
        "metric_names": [],
        "raw": raw,
        "failure_evidence": None,
        "started_at": "2026-07-11T00:00:00Z",
        "completed_at": "2026-07-11T00:00:01Z",
    }
    if aggregate is not None:
        doc["observed_aggregate"] = aggregate
    return doc


def zero_layers() -> list[dict]:
    return [{"layer": index, "k_max_abs": 0.0, "v_max_abs": 0.0}
            for index in range(48)]


def schedule_row() -> dict:
    return {
        "status": "PASS", "passes": True, "tolerance": 5e-4,
        "base_measurement_complete": True,
        "continuation_measurement_complete": True,
        "per_layer": zero_layers(),
        "continuation_per_layer": zero_layers(),
        "cache_k_max_abs": 0.0, "cache_v_max_abs": 0.0,
        "last_logits_max_abs": 0.0,
        "selected_margin_abs_shift": 0.0,
        "continuation_logits_max_abs": 0.0,
        "continuation_k_max_abs": 0.0,
        "continuation_v_max_abs": 0.0,
        "observed_aggregate": 0.0,
    }


def partition(length: int) -> list[int]:
    widths = []
    while length:
        width = min(length, 4096)
        widths.append(width)
        length -= width
    return widths


def complete_pass_gates() -> dict:
    backend = backend_fingerprint()
    synthetic_rows = []
    for length, partitions in zip(
            MODULE.SYNTHETIC_LENGTHS, MODULE.SYNTHETIC_PARTITIONS):
        synthetic_rows.append({
            **schedule_row(), "length": length,
            "reference_partition": list(partitions[0]),
            "alternative_partition": list(partitions[1]),
        })
    gap = {
        **schedule_row(),
        "logical_positions": list(range(32)) + list(range(8192, 8224)),
        "physical_cache_positions": list(range(64)),
        "full_attention_over_physically_prior_rows": True,
    }
    case_rows = []
    for order, cid in enumerate(MODULE.FROZEN_ORDER, 1):
        count = MODULE.CASE_CONTINUATION_POSITIONS[cid]
        tokens = list(range(count))
        positions = list(range(count))
        widths = {"system": 1, "history": count - 2, "request_header": 1}
        case_rows.append({
            **schedule_row(), "conversation_id": cid,
            "order_position": order,
            "source_path": f"data/synthetic/{cid}.json",
            "token_count": count,
            "raw_source_file_sha256": "1" * 64,
            "canonical_parsed_source_sha256": "2" * 64,
            "tokenizer_vocabulary_sha256": "3" * 64,
            "chat_template_sha256": "4" * 64,
            "summary_request_sha256": "5" * 64,
            "complete_prefix_token_ids": tokens,
            "complete_position_ids": positions,
            "complete_prefix_token_sha256": MODULE._sha256_ints(tokens, "tokens"),
            "complete_position_array_sha256": MODULE._sha256_ints(
                positions, "positions"),
            "conceptual_block_widths": widths,
            "system_end": 1,
            "request_header_start": count - 1,
            "system_equal": True,
            "request_header_equal": True,
            "ordinary_resolved_call_widths": partition(count),
            "message_block_resolved_call_widths": partition(count),
        })
    donor_rows = [{
        "order_position": order,
        "target_id": cid,
        "donor_id": MODULE.WRONG_DONORS[cid],
        "subject_native": False,
        "correct_prefix_tokens": 50,
        "wrong_prefix_tokens": 50,
        "structural_slots_equal": True,
        "special_ids_excluded": True,
        "replacement_coverage_exact": True,
        "changed_position_count": 7,
    } for order, cid in enumerate(MODULE.FROZEN_ORDER, 1)]
    gates = {
        "schema": 2,
        "amendment_id": MODULE.AMENDMENT_ID,
        "design_id": MODULE.DESIGN_ID,
        "status": "PASS", "passes": True, "failures": [],
        "max_technical_logical_position": 9509,
        "stage_order": [
            "attention_backend", "synthetic_schedule_fixtures",
            "committed_case_schedule_fixtures", "generated_replay_identity",
            "snapshot_rebuild_identity", "physical_causal_mask_identity",
            "future_mutation_identity", "position_structure",
            "intervention_propagation", "calibration_construction",
            "external_donor_construction", "retired_G_delta",
        ],
        "attention_backend": stage({
            "fingerprint": backend,
            "subject": {
                "resolved_revision": MODULE.MODEL_REVISION,
                "dtype": MODULE.PARAMETER_DTYPE,
                "device": "cuda:0", "context_limit": 32768,
                "max_technical_logical_position": 9509,
                "context_coverage_passes": True,
            },
        }, coverage=48),
        "synthetic_schedule_fixtures": stage({
            "contiguous": synthetic_rows, "logical_gap": gap,
        }, coverage=7, threshold=5e-4, aggregate=0.0),
        "committed_case_schedule_fixtures": stage({
            "rows": case_rows, "frozen_order": list(MODULE.FROZEN_ORDER),
            "coverage_exact": True,
        }, coverage=12, threshold=5e-4, aggregate=0.0),
        "generated_replay_identity": stage({
            "token_logprob_max_abs": 0.0, "k_max_abs": 0.0,
            "v_max_abs": 0.0, "per_layer": zero_layers(),
        }, coverage=1, threshold=1e-4, aggregate=0.0),
        "snapshot_rebuild_identity": stage({
            "logits_max_abs": 0.0, "k_max_abs": 0.0, "v_max_abs": 0.0,
        }, coverage=1, threshold=1e-4, aggregate=0.0),
        "physical_causal_mask_identity": stage({
            "logits_max_abs": 0.0, "k_max_abs": 0.0, "v_max_abs": 0.0,
        }, coverage=1, threshold=1e-4, aggregate=0.0),
        "future_mutation_identity": stage({
            "earlier_logits_max_abs": 0.0, "earlier_cache_max_abs": 0.0,
        }, coverage=1, threshold=1e-4, aggregate=0.0),
        "position_structure": stage({
            "common_summary_start": True, "wrong_prefix_length_equal": True,
            "post_summary_nonempty": True,
            "altered_structure_failure_injection": {"rejected": True},
            "wrong_position_failure_injection": {"rejected": True},
            "physical_cache_positions": list(range(32)),
        }, coverage=1),
        "intervention_propagation": stage({
            "fresh_self_replacement": True,
            "correct_insert_and_non_summary_preservation": True,
            "wrong_insert_and_non_summary_preservation": True,
            "per_arm_fork_at_summary_boundary": True,
            "independently_recomputed_identical_tail_lengths": True,
            "downstream_sensitivity": True, "recomputed_tail_changed": True,
            "pre_tailed_failure_injection": {"rejected": True},
            "sensitivity_attempts": [{"epsilon": 0.1}],
        }, coverage=1),
        "calibration_construction": stage({
            "passes": True, "model_forwards": 0,
            "semantic_scoring_performed": False,
        }, coverage=2),
        "external_donor_construction": stage({
            "rows": donor_rows, "mapping": MODULE.WRONG_DONORS,
            "frozen_order": list(MODULE.FROZEN_ORDER),
            "n_unique_donor_ids": 12, "n_unique_donor_hashes": 12,
        }, coverage=12),
        "retired_G_delta": stage({
            "executed": False, "authorizes_run": False,
        }, coverage=1),
    }
    gates["attention_backend"].update({
        "observed_backend": "eager", "fingerprint": backend,
    })
    return gates


def identity() -> dict:
    return {
        "schema": 2,
        "design_id": MODULE.DESIGN_ID,
        "amendment_id": MODULE.AMENDMENT_ID,
    }


def seal(path: Path, doc: dict) -> None:
    doc = {key: value for key, value in doc.items() if key != "payload_sha256"}
    doc["payload_sha256"] = MODULE._canonical_payload_sha256(doc)
    path.write_text(json.dumps(
        doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def semantic_tree(root: Path, *, corrupt_binding: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    backend = backend_fingerprint()
    apparatus = {"aggregate_sha256": "a" * 64}
    harvest = {
        "path": "results/technical.harvest_validation.json",
        "raw_sha256": "b" * 64,
        "payload_sha256": "c" * 64,
    }
    authorization = {
        "result_commit": "d" * 40,
        "run_dir": "results/technical",
        "gate_payload_sha256": "e" * 64,
        "raw_sha256": {"gate": "f" * 64},
        "apparatus_inventory": apparatus,
        "harvest": harvest,
    }
    binding = {
        "technical_result_commit": authorization["result_commit"],
        "technical_run_dir": authorization["run_dir"],
        "gate_payload_sha256": authorization["gate_payload_sha256"],
        "raw_sha256": authorization["raw_sha256"],
        "harvest_path": harvest["path"],
        "harvest_raw_sha256": harvest["raw_sha256"],
        "harvest_payload_sha256": harvest["payload_sha256"],
        "apparatus_aggregate_sha256": apparatus["aggregate_sha256"],
        "backend_exact": True,
        "static_fingerprint_exact": True,
    }
    if corrupt_binding:
        binding["gate_payload_sha256"] = "0" * 64
    fingerprint = {
        **identity(),
        "frozen_order": list(MODULE.FROZEN_ORDER),
        "wrong_donors": MODULE.WRONG_DONORS,
        "attention_backend": "eager",
        "attention_backend_fingerprint": backend,
        "semantic_authorization": binding,
    }
    checks = {
        "terminal_payloads_exact": True,
        "committed_directory_bytes_exact": True,
        "result_commit_is_ancestor": True,
        "result_commit_on_trunk": True,
        "harvest_attestation_committed_exact": True,
        "independent_harvest_revalidation_passed": True,
        "apparatus_inventory_exact": True,
        "static_data_fingerprint_exact": True,
        "backend_attestation_exact": True,
    }
    seal(root / "manifest.json", {
        **identity(), "status": "COMPLETE", "phase": "COMPLETE",
        "technical_only": False, "resume_probe_verified": True,
        "n_conversations": 6, "fingerprint": fingerprint,
        "attention_backend": "eager",
        "attention_backend_fingerprint": backend,
        "semantic_authorization": authorization,
        "authorization_checks": checks,
    })
    seal(root / "resume_probe.json", {
        **identity(), "status": "VERIFIED", "resume_probe_verified": True,
    })
    for position, cid in enumerate(MODULE.FROZEN_ORDER[:6], 1):
        arms = {arm: {"conversation_margin": 0.0} for arm in MODULE.ARMS}
        seal(root / f"conv_{position:02d}_{cid}.json", {
            **identity(), "stage": "scored", "status": "scored",
            "order_position": position, "conversation_id": cid,
            "conversation": {}, "summary": {}, "sources": {},
            "destination": {}, "arm_scores": arms,
            "conversation_outcomes": {arm: 0.0 for arm in MODULE.ARMS},
            "gates": {"technical_pass": True}, "runtime": {},
            "fingerprint": fingerprint,
        })
    paths = sorted(path.name for path in root.glob("*.json"))
    rows = []
    for relative in paths:
        path = root / relative
        doc = json.loads(path.read_text())
        rows.append({
            "path": relative, "bytes": path.stat().st_size,
            "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "payload_sha256": doc["payload_sha256"],
        })
    index = {
        **identity(), "status": "PASS",
        "required_apparatus_payload_paths": paths, "artifacts": rows,
    }
    index_path = root / "terminal_artifact_index.json"
    index_path.write_text(json.dumps(
        index, sort_keys=True, separators=(",", ":")) + "\n")
    receipt = {
        **identity(), "status": "PASS",
        "index_path": index_path.name, "index_bytes": index_path.stat().st_size,
        "index_raw_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }
    (root / "terminal_receipt.json").write_text(json.dumps(
        receipt, sort_keys=True, separators=(",", ":")) + "\n")
    (root / "job.log").write_text("COHERENT_STATE_JOB_DONE\n")


def technical_tree(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    gates = complete_pass_gates()
    refs = {}
    for stage_name in (
            "committed_case_schedule_fixtures",
            "external_donor_construction"):
        lifecycle = gates.pop(stage_name)
        relative = f"technical_stage_{stage_name}.json"
        path = root / relative
        seal(path, {
            **identity(), "status": "PASS",
            "kind": "technical_gate_stage_sidecar",
            "stage_name": stage_name, "lifecycle": lifecycle,
        })
        sidecar = json.loads(path.read_text())
        refs[stage_name] = {
            "path": relative, "byte_count": path.stat().st_size,
            "raw_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "payload_sha256": sidecar["payload_sha256"],
        }
    gates["stage_refs"] = refs
    backend = backend_fingerprint()
    fingerprint = {
        **identity(), "attention_backend": "eager",
        "attention_backend_fingerprint": backend,
    }
    static = {"code_commit": "a" * 40}
    apparatus = {"aggregate_sha256": "b" * 64}
    gate_doc = {
        **identity(), "status": "PASS", "completed_at": "2026-07-11T00:00:00Z",
        "model": MODULE.MODEL_ID, "revision": MODULE.MODEL_REVISION,
        "dtype": MODULE.PARAMETER_DTYPE, "technical_only": True,
        "attention_backend": "eager", "attention_backend_fingerprint": backend,
        "fingerprint": fingerprint, "fingerprint_static": static,
        "apparatus_inventory": apparatus, "geometry": {"layers": 48},
        "context_limit": 32768, "gates": gates,
    }
    unique_name = "production_kernel_gate_20260711T000000000000Z.json"
    seal(root / unique_name, gate_doc)
    seal(root / "production_kernel_gate.json", gate_doc)
    gate_sealed = json.loads((root / "production_kernel_gate.json").read_text())
    seal(root / "manifest.json", {
        **identity(), "status": "TECHNICAL_PASS", "phase": "TECHNICAL_COMPLETE",
        "technical_only": True, "model": MODULE.MODEL_ID,
        "revision": MODULE.MODEL_REVISION, "dtype": MODULE.PARAMETER_DTYPE,
        "attention_backend": "eager", "attention_backend_fingerprint": backend,
        "fingerprint": fingerprint, "fingerprint_static": static,
        "apparatus_inventory": apparatus, "geometry": {"layers": 48},
        "context_limit": 32768,
        "production_kernel_gate_path": "production_kernel_gate.json",
        "production_kernel_gate_attempt_path": unique_name,
        "production_kernel_gate_payload_sha256": gate_sealed["payload_sha256"],
    })
    paths = sorted(path.name for path in root.glob("*.json"))
    rows = []
    for relative in paths:
        path = root / relative
        doc = json.loads(path.read_text())
        rows.append({
            "path": relative, "bytes": path.stat().st_size,
            "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "payload_sha256": doc["payload_sha256"],
        })
    index = {
        **identity(), "status": "PASS",
        "required_apparatus_payload_paths": paths, "artifacts": rows,
    }
    index_path = root / "terminal_artifact_index.json"
    index_path.write_text(json.dumps(
        index, sort_keys=True, separators=(",", ":")) + "\n")
    receipt = {
        **identity(), "status": "PASS", "index_path": index_path.name,
        "index_bytes": index_path.stat().st_size,
        "index_raw_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }
    (root / "terminal_receipt.json").write_text(json.dumps(
        receipt, sort_keys=True, separators=(",", ":")) + "\n")
    (root / "job.log").write_text("COHERENT_STATE_TECHNICAL_DONE\n")


def test_independent_v6_science_validation_accepts_complete_exact_fixture():
    MODULE._validate_v6_pass_gates(complete_pass_gates())


@pytest.mark.parametrize("mutation,match", [
    (lambda gates: gates["synthetic_schedule_fixtures"]["raw"]["contiguous"][0].__setitem__(
        "cache_k_max_abs", 1e-3), "stored cache_k_max_abs"),
    (lambda gates: gates["committed_case_schedule_fixtures"]["raw"]["rows"][9].__setitem__(
        "token_count", 9508), "committed-case identity"),
    (lambda gates: gates["external_donor_construction"]["raw"]["rows"][0].__setitem__(
        "subject_native", True), "external donor row"),
])
def test_independent_v6_science_validation_recomputes_raw_evidence(mutation, match):
    gates = complete_pass_gates()
    mutation(gates)
    with pytest.raises(ValueError, match=match):
        MODULE._validate_v6_pass_gates(gates)


def test_semantic_complete_validates_bound_prior_authorization_and_envelope(
        tmp_path: Path):
    semantic_tree(tmp_path)
    observed = MODULE.validate(tmp_path, "complete")
    assert observed["status"] == "PASS"
    assert observed["n_scored"] == 6


def test_semantic_complete_rejects_mixed_authorization_binding(tmp_path: Path):
    semantic_tree(tmp_path, corrupt_binding=True)
    with pytest.raises(ValueError, match="fingerprint/prior authorization"):
        MODULE.validate(tmp_path, "complete")


def test_technical_harvest_validates_full_sidecar_and_terminal_contract(
        tmp_path: Path):
    technical_tree(tmp_path)
    observed = MODULE.validate(tmp_path, "technical")
    assert observed["status"] == "PASS"
    assert observed["production_gate"] == "PASS"


def test_technical_harvest_rejects_sidecar_binding_tamper(tmp_path: Path):
    technical_tree(tmp_path)
    path = tmp_path / "technical_stage_external_donor_construction.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="byte count differs|hashes differ"):
        MODULE.validate(tmp_path, "technical")
