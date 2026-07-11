from __future__ import annotations

import copy
import importlib.util
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
        "physical_positions": list(range(64)),
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
    return gates


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
