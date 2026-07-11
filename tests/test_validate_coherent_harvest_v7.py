from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_coherent_harvest_v7",
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


def hash_rows(seed: str) -> list[dict]:
    return [{
        "layer": str(index),
        "k_sha256": hashlib.sha256(f"{seed}:k:{index}".encode()).hexdigest(),
        "v_sha256": hashlib.sha256(f"{seed}:v:{index}".encode()).hexdigest(),
    } for index in range(48)]


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


def target_score(token_logprobs: list[float], *, token_base: int = 100) -> dict:
    return {
        "text": "fixture target",
        "token_ids": list(range(token_base, token_base + len(token_logprobs))),
        "token_logprobs": token_logprobs,
        "mean_logprob": sum(token_logprobs) / len(token_logprobs),
        "probe_suffix_ids": [90],
        "logical_position_ids": list(range(10, 10 + len(token_logprobs))),
        "physical_cache_positions": list(range(5, 5 + len(token_logprobs))),
    }


def arm_score(*, offset: float = 0.0, plant_id: str = "fixture-plant") -> dict:
    correct = target_score([-1.0 + offset, -2.0 + offset])
    counterfactual = target_score(
        [-2.0 + offset, -3.0 + offset], token_base=200)
    margin = correct["mean_logprob"] - counterfactual["mean_logprob"]
    row = {
        "plant_id": plant_id, "category": "referent", "probe": "Which?",
        "correct": correct, "counterfactual": counterfactual,
        "margin": margin,
    }
    return {"plants": [row], "conversation_margin": margin}


def partition(length: int) -> list[int]:
    widths = []
    while length:
        width = min(length, 4096)
        widths.append(width)
        length -= width
    return widths


def science_fingerprint() -> dict:
    return {
        "schema": 2, "design_id": MODULE.DESIGN_ID,
        "amendment_id": MODULE.AMENDMENT_ID,
        "code_commit": "a" * 40,
        "apparatus_inventory": {"aggregate_sha256": "b" * 64},
        "input_inventory": {"files": [], "aggregate_sha256": "c" * 64},
    }


def _real_case_identity(cid: str) -> dict:
    path = ROOT / "data" / "synthetic" / f"{cid}.json"
    raw = path.read_bytes()
    parsed = json.loads(raw)
    tokenizer = MODULE._validation_tokenizer()
    messages = parsed["messages"]
    correct = MODULE._generation_prefix_ids(
        tokenizer, list(messages) + [
            {"role": "user", "content": MODULE.SUMMARY_REQUEST}])
    fresh = MODULE._generation_prefix_ids(
        tokenizer, [messages[0],
                    {"role": "user", "content": MODULE.SUMMARY_REQUEST}])
    marker = int(tokenizer.encode(
        "<|im_start|>", add_special_tokens=False)[0])
    starts = [index for index, token in enumerate(correct) if token == marker]
    system_end = starts[1]
    suffix = fresh[system_end:]
    request_start = len(correct) - len(suffix)
    widths = {
        "system": system_end, "history": request_start - system_end,
        "request_header": len(correct) - request_start,
    }
    positions = list(range(len(correct)))
    return {
        "raw_source_file_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_parsed_source_sha256": MODULE._canonical_json_sha256(parsed),
        "parsed_source": parsed,
        "recorded_author": (parsed.get("meta") or {}).get("author"),
        "exact_model_revision": MODULE.MODEL_REVISION,
        "tokenizer_vocabulary_sha256": MODULE._canonical_json_sha256(
            tokenizer.get_vocab()),
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode()).hexdigest(),
        "summary_request_sha256": hashlib.sha256(
            MODULE.SUMMARY_REQUEST.encode()).hexdigest(),
        "complete_prefix_token_ids": correct,
        "complete_prefix_token_sha256": MODULE._sha256_ints(correct, "correct"),
        "fresh_prefix_token_ids": fresh,
        "token_count": len(correct),
        "continuation_logical_position": len(correct),
        "expected_continuation_logical_position": len(correct),
        "continuation_position_matches_frozen": True,
        "complete_position_ids": positions,
        "complete_position_array_sha256": MODULE._sha256_ints(
            positions, "positions"),
        "system_end": system_end, "request_header_start": request_start,
        "conceptual_block_widths": widths,
        "ordinary_resolved_call_widths": MODULE._chunk_widths(len(correct)),
        "message_block_resolved_call_widths": [
            piece for key in ("system", "history", "request_header")
            for piece in MODULE._chunk_widths(widths[key])],
        "boundary_token_ids": {
            "im_start": marker, "system_end_token": correct[system_end],
            "request_header_start_token": correct[request_start],
            "final_prefix_token": correct[-1],
        },
        "message_start_positions": starts,
        "blocks_nonempty": True, "blocks_ordered_nonoverlapping": True,
        "blocks_cover_prefix": True, "system_equal": True,
        "request_header_equal": True,
    }


def complete_pass_gates(*, real_cases: bool = False) -> dict:
    from coherent_state_calibration import validate_calibration_constructions
    calibration = validate_calibration_constructions(
        MODULE._validation_tokenizer())
    backend = backend_fingerprint()
    synthetic_rows = []
    for length, partitions in zip(
            MODULE.SYNTHETIC_LENGTHS, MODULE.SYNTHETIC_PARTITIONS):
        synthetic_rows.append({
            **schedule_row(), "length": length,
            "reference_partition": list(partitions[0]),
            "alternative_partition": list(partitions[1]),
            "token_ids_sha256": MODULE._sha256_ints([
                MODULE.FROZEN_FIXTURE_POOL[i % len(MODULE.FROZEN_FIXTURE_POOL)]
                for i in range(length)], "synthetic"),
        })
    gap = {
        **schedule_row(),
        "logical_positions": list(range(32)) + list(range(8192, 8224)),
        "physical_cache_positions": list(range(64)),
        "full_attention_over_physically_prior_rows": True,
        "length": 64, "reference_partition": [32, 32],
        "alternative_partition": [32] + [1] * 32,
        "continuation_logical_position": 8224,
        "logical_as_cache_position_rejected": True,
        "token_ids_sha256": MODULE._sha256_ints([
            MODULE.FROZEN_FIXTURE_POOL[i % len(MODULE.FROZEN_FIXTURE_POOL)]
            for i in range(64)], "gap_tokens"),
        "logical_positions_sha256": MODULE._sha256_ints(
            list(range(32)) + list(range(8192, 8224)), "gap_positions"),
    }
    case_rows = []
    for order, cid in enumerate(MODULE.FROZEN_ORDER, 1):
        count = MODULE.CASE_CONTINUATION_POSITIONS[cid]
        tokens = list(range(count))
        positions = list(range(count))
        widths = {"system": 1, "history": count - 2, "request_header": 1}
        case_row = {
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
        }
        if real_cases:
            case_row.update(_real_case_identity(cid))
        case_rows.append(case_row)
    donor_rows = []
    for order, cid in enumerate(MODULE.FROZEN_ORDER, 1):
        correct_ids = list(range(10))
        structural = [0, 1, 8, 9]
        content = [2, 3, 4, 5, 6, 7]
        changed = list(content)
        replacement = {
            "start": 2, "end": 8, "length": 6,
            "target_ids": correct_ids[2:8],
            "donor_pool_ids": [20, 21, 22, 23, 24, 25],
            "replacement_ids": [20, 21, 22, 23, 24, 25],
            "contains_special_token": False,
            "target_message_index": 1, "donor_message_index": 1,
            "role": "user", "cycles": 1,
        }
        replacement.update({
            "target_ids_sha256": MODULE._sha256_ints(
                replacement["target_ids"], "target"),
            "source_pool_sha256": MODULE._sha256_ints(
                replacement["donor_pool_ids"], "pool"),
            "replacement_sha256": MODULE._sha256_ints(
                replacement["replacement_ids"], "replacement"),
        })
        donor_rows.append({
            "order_position": order, "target_id": cid,
            "donor_id": MODULE.WRONG_DONORS[cid], "subject_native": False,
            "target_path": f"data/synthetic/{cid}.json",
            "donor_path": f"data/synthetic/{MODULE.WRONG_DONORS[cid]}.json",
            "target_sha256": "1" * 64, "donor_sha256": "2" * 64,
            "target_canonical_sha256": "3" * 64,
            "donor_canonical_sha256": "4" * 64,
            "donor_recorded_author": "opus",
            "correct_prefix_tokens": len(correct_ids),
            "wrong_prefix_tokens": len(correct_ids),
            "correct_prefix_ids": correct_ids,
            "correct_prefix_sha256": MODULE._sha256_ints(correct_ids, "correct"),
            "structural_position_count": len(structural),
            "structural_positions": structural,
            "structural_positions_sha256": MODULE._sha256_ints(
                structural, "structural"),
            "content_position_count": len(content),
            "content_positions": content,
            "content_positions_sha256": MODULE._sha256_ints(content, "content"),
            "changed_position_count": len(changed),
            "changed_positions": changed,
            "changed_positions_sha256": MODULE._sha256_ints(changed, "changed"),
            "replacement_count": 1, "replacements": [replacement],
            "structural_tokens_equal": True,
            "system_request_header_retained_tail_unchanged": True,
            "changes_confined_to_declared_content_positions": True,
            "correct_wrong_length_equal": True,
            "replacement_spans_non_overlapping": True,
            "replacement_coverage_exact": True,
        })
        reconstructed = list(correct_ids)
        reconstructed[2:8] = replacement["replacement_ids"]
        donor_rows[-1]["wrong_prefix_sha256"] = MODULE._sha256_ints(
            reconstructed, "wrong")
    donor_raw = {
        "status": "PASS", "passes": True,
        "requested_revision": MODULE.MODEL_REVISION,
        "resolved_tokenizer_revision": MODULE.MODEL_REVISION,
        "expected_coverage": 12, "observed_coverage": 12,
        "rows": donor_rows, "mapping": MODULE.WRONG_DONORS,
        "frozen_order": list(MODULE.FROZEN_ORDER),
        "n_unique_donor_ids": 12, "n_unique_donor_hashes": 12,
    }
    if real_cases:
        from validate_coherent_external_donors import validate_with_tokenizer
        donor_raw = validate_with_tokenizer(
            MODULE._validation_tokenizer(), ROOT / "data" / "synthetic",
            resolved_revision=MODULE.MODEL_REVISION)
        # Production uses the repository-relative CLI argument.
        for row in donor_raw["rows"]:
            row["target_path"] = row["target_path"].replace(
                f"{ROOT}/", "")
            row["donor_path"] = row["donor_path"].replace(
                f"{ROOT}/", "")
        donor_raw.pop("canonical_payload_sha256", None)
    donor_raw["canonical_payload_sha256"] = MODULE._canonical_json_sha256(
        donor_raw)
    static_fingerprint = science_fingerprint()
    gates = {
        "schema": 2,
        "amendment_id": MODULE.AMENDMENT_ID,
        "design_id": MODULE.DESIGN_ID,
        "status": "PASS", "passes": True, "failures": [],
        "max_technical_logical_position": 9509,
        "stage_order": [
            "static_provenance", "attention_backend", "synthetic_schedule_fixtures",
            "committed_case_schedule_fixtures", "generated_replay_identity",
            "snapshot_rebuild_identity", "physical_causal_mask_identity",
            "future_mutation_identity", "position_structure",
            "intervention_propagation", "calibration_construction",
            "external_donor_construction", "retired_G_delta",
        ],
        "static_provenance": stage({
            "fingerprint_static_sha256": MODULE._canonical_json_sha256(
                static_fingerprint),
            "apparatus_inventory_sha256": MODULE._canonical_json_sha256(
                static_fingerprint["apparatus_inventory"]),
            "input_inventory_sha256": MODULE._canonical_json_sha256(
                static_fingerprint["input_inventory"]),
            "code_commit": static_fingerprint["code_commit"],
            "model": MODULE.MODEL_ID, "revision": MODULE.MODEL_REVISION,
            "dtype": MODULE.PARAMETER_DTYPE, "attention_backend": "eager",
            "technical_only": True,
        }, coverage=1),
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
            "fixture_provenance": {
                "literal": MODULE.FROZEN_FIXTURE_LITERAL,
                "pool_token_ids": MODULE.FROZEN_FIXTURE_POOL,
                "pool_sha256": MODULE._sha256_ints(
                    MODULE.FROZEN_FIXTURE_POOL, "pool"),
                "margin_token_ids": MODULE.FROZEN_MARGIN_IDS,
                "continuation_token_id": MODULE.FROZEN_CONTINUATION_ID,
                "pool_contains_special_token": False,
            },
            "contiguous": synthetic_rows, "logical_gap": gap,
            "passes": True, "failures": [],
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
            "per_layer": zero_layers(),
        }, coverage=1, threshold=1e-4, aggregate=0.0),
        "physical_causal_mask_identity": stage({
            "logits_max_abs": 0.0, "k_max_abs": 0.0, "v_max_abs": 0.0,
            "per_layer": zero_layers(),
        }, coverage=1, threshold=1e-4, aggregate=0.0),
        "future_mutation_identity": stage({
            "earlier_logits_max_abs": 0.0, "earlier_cache_max_abs": 0.0,
            "per_layer": zero_layers(),
        }, coverage=1, threshold=1e-4, aggregate=0.0),
        "position_structure": stage({
            "common_summary_start": True, "wrong_prefix_length_equal": True,
            "post_summary_nonempty": True,
            "altered_structure_failure_injection": {"rejected": True},
            "wrong_position_failure_injection": {"rejected": True},
            "source_summary_start": 10, "physical_summary_start": 4,
            "physical_summary_end": 6, "system_end": 2,
            "request_logical_start": 8, "logical_next_position": 13,
            "logical_gap": 6,
            "prefix_position_ids": [0, 1, 8, 9],
            "summary_position_ids": [10, 11],
            "post_summary_position_ids": [12],
            "context_position_ids": [0, 1, 8, 9, 10, 11, 12],
            "physical_cache_positions": list(range(7)),
            "correct_prefix_ids": list(range(10)),
            "wrong_prefix_ids": [0, 1, 20, 21, 22, 23, 24, 25, 8, 9],
            "correct_prefix_sha256": MODULE._sha256_ints(
                list(range(10)), "position.correct"),
            "wrong_prefix_sha256": MODULE._sha256_ints(
                [0, 1, 20, 21, 22, 23, 24, 25, 8, 9], "position.wrong"),
            "wrong_structural_position_ids": [0, 1, 8, 9],
            "wrong_content_position_ids": [2, 3, 4, 5, 6, 7],
            "wrong_structural_positions": 4, "wrong_content_positions": 6,
            "wrong_structural_positions_sha256": MODULE._sha256_ints(
                [0, 1, 8, 9], "position.structural"),
            "wrong_content_positions_sha256": MODULE._sha256_ints(
                [2, 3, 4, 5, 6, 7], "position.content"),
            "context_ids": [101, 102, 103, 104, 105, 106, 107],
            "context_ids_sha256": MODULE._sha256_ints(
                [101, 102, 103, 104, 105, 106, 107], "position.context"),
            "summary_ids": [105, 106],
            "summary_ids_sha256": MODULE._sha256_ints(
                [105, 106], "position.summary"),
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
            "summary_span": {"start": 10, "end": 20},
            "boundary_lengths": {
                "fresh": 20, "self": 20, "correct": 20, "wrong": 20},
            "full_lengths": {"fresh": 30, "correct": 30, "wrong": 30},
            "hashes": {
                "fresh_boundary": hash_rows("fresh_boundary"),
                "self_boundary": hash_rows("fresh_boundary"),
                "correct_boundary": hash_rows("correct_boundary"),
                "wrong_boundary": hash_rows("wrong_boundary"),
                "fresh_before_summary": hash_rows("before"),
                "correct_before_summary": hash_rows("before"),
                "wrong_before_summary": hash_rows("before"),
                "correct_source_summary": hash_rows("correct_summary"),
                "wrong_source_summary": hash_rows("wrong_summary"),
                "correct_inserted_summary": hash_rows("correct_summary"),
                "wrong_inserted_summary": hash_rows("wrong_summary"),
                "full_fresh": hash_rows("full_fresh"),
                "full_correct": hash_rows("full_correct"),
                "full_wrong": hash_rows("full_wrong"),
                "fresh_post_summary": hash_rows("tail_fresh"),
                "correct_post_summary": hash_rows("tail_correct"),
                "wrong_post_summary": hash_rows("tail_wrong"),
            },
        }, coverage=1),
        "calibration_construction": stage(calibration, coverage=2),
        "external_donor_construction": stage({
            **donor_raw,
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
        arms = {arm: arm_score(offset=index / 10)
                for index, arm in enumerate(MODULE.ARMS)}
        calibration_details = {
            arm: arm_score(offset=index / 20, plant_id="calibration")
            for index, arm in enumerate(("G_fresh", "G_correct", "G_wrong"))
        }
        calibration_outcomes = {
            arm: score["conversation_margin"]
            for arm, score in calibration_details.items()
        }
        rendered_conversation = json.loads((
            ROOT / "data" / "synthetic" / f"{cid}.json").read_text())
        case_identity = _real_case_identity(cid)
        actual_schedule = {
            **identity(), **schedule_row(), "conversation_id": cid,
            "semantic_scoring_performed": False,
            **{key: case_identity[key] for key in (
                "complete_prefix_token_ids", "complete_prefix_token_sha256",
                "complete_position_ids", "complete_position_array_sha256",
                "fresh_prefix_token_ids", "token_count",
                "continuation_logical_position", "system_end",
                "request_header_start", "conceptual_block_widths",
                "ordinary_resolved_call_widths",
                "message_block_resolved_call_widths", "system_equal",
                "request_header_equal", "blocks_nonempty",
                "blocks_ordered_nonoverlapping", "blocks_cover_prefix")},
        }
        seal(root / f"conv_{position:02d}_{cid}.json", {
            **identity(), "stage": "scored", "status": "scored",
            "order_position": position, "conversation_id": cid,
            "conversation": rendered_conversation,
            "pre_score_schedule_equivalence": actual_schedule,
            "summary": {}, "sources": {},
            "destination": {}, "arm_scores": arms,
            "conversation_outcomes": {
                arm: score["conversation_margin"] for arm, score in arms.items()},
            "calibration": {
                "arm_details": calibration_details,
                "outcomes": calibration_outcomes,
            },
            "calibration_outcomes": calibration_outcomes,
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
    gates = complete_pass_gates(real_cases=True)
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
        "code_commit": "a" * 40,
        "apparatus_inventory": {"aggregate_sha256": "b" * 64},
        "attention_backend_fingerprint": backend,
        "summary_request_sha256": hashlib.sha256(
            MODULE.SUMMARY_REQUEST.encode()).hexdigest(),
        "subject_metadata": {
            "tokenizer_vocab_sha256": MODULE._canonical_json_sha256(
                MODULE._validation_tokenizer().get_vocab()),
            "chat_template_sha256": hashlib.sha256(str(
                MODULE._validation_tokenizer().chat_template).encode()).hexdigest(),
        },
        "input_inventory": {"files": [{
            "path": f"data/synthetic/{cid}.json",
            "bytes": (ROOT / "data" / "synthetic" / f"{cid}.json").stat().st_size,
            "sha256": hashlib.sha256((
                ROOT / "data" / "synthetic" / f"{cid}.json").read_bytes()).hexdigest(),
        } for cid in sorted(set(MODULE.FROZEN_ORDER).union(
            MODULE.WRONG_DONORS.values()))]},
    }
    static = dict(fingerprint)
    gates["static_provenance"]["raw"] = {
        "fingerprint_static_sha256": MODULE._canonical_json_sha256(static),
        "apparatus_inventory_sha256": MODULE._canonical_json_sha256(
            fingerprint["apparatus_inventory"]),
        "input_inventory_sha256": MODULE._canonical_json_sha256(
            fingerprint["input_inventory"]),
        "code_commit": fingerprint["code_commit"],
        "model": MODULE.MODEL_ID, "revision": MODULE.MODEL_REVISION,
        "dtype": MODULE.PARAMETER_DTYPE, "attention_backend": "eager",
        "technical_only": True,
    }
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


def test_independent_v7_science_validation_accepts_complete_exact_fixture():
    MODULE._validate_v7_pass_gates(
        complete_pass_gates(), fingerprint=science_fingerprint(), repo_root=None,
        static_fingerprint=science_fingerprint(), verify_sources=False)


@pytest.mark.parametrize("mutation,match", [
    (lambda gates: gates["synthetic_schedule_fixtures"]["raw"]["contiguous"][0].__setitem__(
        "cache_k_max_abs", 1e-3), "stored cache_k_max_abs"),
    (lambda gates: gates["committed_case_schedule_fixtures"]["raw"]["rows"][9].__setitem__(
        "token_count", 9508), "committed-case identity"),
    (lambda gates: gates["external_donor_construction"]["raw"]["rows"][0].__setitem__(
        "subject_native", True), "external donor canonical|external donor row"),
])
def test_independent_v7_science_validation_recomputes_raw_evidence(mutation, match):
    gates = complete_pass_gates()
    mutation(gates)
    with pytest.raises(ValueError, match=match):
        MODULE._validate_v7_pass_gates(
            gates, fingerprint=science_fingerprint(), repo_root=None,
            static_fingerprint=science_fingerprint(), verify_sources=False)


def test_independent_donor_validation_rejects_special_token_counterexample():
    gates = complete_pass_gates()
    donor = gates["external_donor_construction"]["raw"]
    row = donor["rows"][0]
    replacement = row["replacements"][0]
    special = int(MODULE._validation_tokenizer().all_special_ids[0])
    replacement["donor_pool_ids"][0] = special
    replacement["replacement_ids"][0] = special
    replacement["source_pool_sha256"] = MODULE._sha256_ints(
        replacement["donor_pool_ids"], "mutated donor pool")
    replacement["replacement_sha256"] = MODULE._sha256_ints(
        replacement["replacement_ids"], "mutated replacement")
    reconstructed = list(row["correct_prefix_ids"])
    for item in row["replacements"]:
        reconstructed[item["start"]:item["end"]] = item["replacement_ids"]
    changed = [index for index, pair in enumerate(zip(
        row["correct_prefix_ids"], reconstructed)) if pair[0] != pair[1]]
    row.update({
        "changed_positions": changed,
        "changed_position_count": len(changed),
        "changed_positions_sha256": MODULE._sha256_ints(changed, "changed"),
        "wrong_prefix_sha256": MODULE._sha256_ints(reconstructed, "wrong"),
    })
    donor["canonical_payload_sha256"] = MODULE._canonical_json_sha256({
        key: value for key, value in donor.items()
        if key != "canonical_payload_sha256"})
    with pytest.raises(ValueError, match="replacement bounds"):
        MODULE._validate_v7_pass_gates(
            gates, fingerprint=science_fingerprint(), repo_root=None,
            static_fingerprint=science_fingerprint(), verify_sources=False)


def test_independent_donor_reconstruction_matches_committed_sources():
    tokenizer = MODULE._validation_tokenizer()
    artifact = json.loads(next((ROOT / "results" / "coherent_state_ladder").glob(
        "coherent_external_donors_gapped_v7_*.json")).read_text())
    rows = {row["target_id"]: row for row in artifact["rows"]}
    for cid in MODULE.FROZEN_ORDER:
        donor_id = MODULE.WRONG_DONORS[cid]
        target = json.loads((ROOT / "data" / "synthetic" / f"{cid}.json").read_text())
        donor = json.loads((ROOT / "data" / "synthetic" / f"{donor_id}.json").read_text())
        expected = MODULE._reconstruct_donor_replacements(
            tokenizer, target, donor, cid)
        observed = rows[cid]
        assert observed["correct_prefix_ids"] == expected["correct_ids"]
        assert observed["structural_positions"] == expected["structural"]
        assert observed["content_positions"] == expected["content"]
        assert observed["changed_positions"] == expected["changed"]
        assert observed["replacements"] == expected["replacements"]
        assert observed["wrong_prefix_sha256"] == MODULE._sha256_ints(
            expected["wrong_ids"], f"{cid}.wrong")


def test_semantic_complete_validates_bound_prior_authorization_and_envelope(
        tmp_path: Path):
    semantic_tree(tmp_path)
    observed = MODULE.validate(tmp_path, "complete")
    assert observed["status"] == "PASS"
    assert observed["n_scored"] == 6


@pytest.mark.parametrize("mutation,match", [
    (lambda doc: doc["arm_scores"]["A_full"]["plants"][0]["correct"].__setitem__(
        "mean_logprob", -99.0), "target mean_logprob differs"),
    (lambda doc: doc["arm_scores"]["G_fresh"]["plants"][0].__setitem__(
        "margin", -99.0), "plant margin differs"),
    (lambda doc: doc["arm_scores"]["G_correct"].__setitem__(
        "conversation_margin", -99.0), "conversation_margin differs"),
    (lambda doc: doc["conversation_outcomes"].__setitem__(
        "G_wrong", -99.0), "conversation outcome differs"),
    (lambda doc: doc["calibration"]["arm_details"]["G_correct"]["plants"][0]
     ["counterfactual"].__setitem__("mean_logprob", -99.0),
     "target mean_logprob differs"),
    (lambda doc: doc["calibration_outcomes"].__setitem__(
        "G_wrong", -99.0), "calibration outcome differs"),
])
def test_semantic_checkpoint_recomputes_every_decision_aggregate(
        tmp_path: Path, mutation, match):
    semantic_tree(tmp_path)
    path = tmp_path / f"conv_01_{MODULE.FROZEN_ORDER[0]}.json"
    doc = json.loads(path.read_text())
    mutation(doc)
    with pytest.raises(ValueError, match=match):
        MODULE._validate_checkpoint(
            doc, path, scored=True,
            expected_fingerprint=doc["fingerprint"])


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


def test_failure_harvest_preserves_terminal_pass_rejected_by_independent_validator(
        tmp_path: Path):
    technical_tree(tmp_path)
    (tmp_path / "job.log").write_text(
        "MODEL_READY\nFATAL: independent technical harvest validation rejected "
        "terminal PASS\n")
    observed = MODULE.validate(tmp_path, "failure")
    assert observed["status"] == "PASS"
    assert observed["terminal_integrity_verified"] is True
    assert observed["technical_pass_rejected_at_harvest"] is True
