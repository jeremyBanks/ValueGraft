"""Exact technical-fixture orchestration for v12 (no model loading or release)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import torch

from coherent_canary_path_control import run_bidirectional_path_control
from coherent_canary_runtime import (
    append_block_to_snapshot, continue_fresh_plan, execute_fresh_plan,
    execute_prefix_block, execute_replay_plan, extract_rows, force_content_q1,
    greedy_generate_q1, probe_suffix_ids, replace_rows,
    require_generated_forced_identity, score_target_q1, snapshot_hashes,
    tensor_sha256,
)
from coherent_canary_schema import R1, R2, R3
from coherent_canary_tokens import build_fresh_destination_plan, build_role_native_plan
from coherent_state_tokens import generation_prefix_ids


class CanaryTechnicalError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryTechnicalError(message)


def source_messages(history: list[dict], middle: int) -> list[dict]:
    from coherent_canary_schema import (
        ANCHOR_ASSISTANT, ANCHOR_USER, ENGINEERED_CARRIER_CONTENT,
        ENGINEERED_CARRIER_REQUEST,
    )
    return list(history[:middle]) + [
        {"role": "user", "content": ENGINEERED_CARRIER_REQUEST},
        {"role": "assistant", "content": ENGINEERED_CARRIER_CONTENT},
        {"role": "user", "content": ANCHOR_USER},
        {"role": "assistant", "content": ANCHOR_ASSISTANT},
    ] + list(history[middle:])


def compact_messages(history: list[dict], middle: int) -> list[dict]:
    return source_messages(history, middle)[:1] + source_messages(history, middle)[middle:]


def _probe_record(model, tokenizer, snapshot, messages, context_ids,
                  logical_end: int, probe: str, correct_text: str,
                  wrong_text: str, eos_ids: Sequence[int]) -> dict:
    suffix = probe_suffix_ids(tokenizer, messages, context_ids, probe)
    correct_ids = tokenizer.encode(correct_text, add_special_tokens=False)
    wrong_ids = tokenizer.encode(wrong_text, add_special_tokens=False)
    correct = score_target_q1(
        model, snapshot, suffix_ids=suffix, target_ids=correct_ids,
        logical_context_end=logical_end)
    wrong = score_target_q1(
        model, snapshot, suffix_ids=suffix, target_ids=wrong_ids,
        logical_context_end=logical_end)
    prefix = append_block_to_snapshot(
        model, snapshot, suffix, logical_start=logical_end,
        label="probe_user_and_assistant_header")
    generation = greedy_generate_q1(
        model, prefix.snapshot, prefix.last_logits,
        logical_start=prefix.logical_end, eos_ids=eos_ids)
    return {
        "probe": probe, "suffix_ids": suffix,
        "correct_text": correct_text, "counterfactual_text": wrong_text,
        "correct": correct, "counterfactual": wrong,
        "margin": correct["mean_logprob"] - wrong["mean_logprob"],
        "probe_prefix_trace": {
            "calls": prefix.calls, "token_ids": prefix.executed_token_ids,
            "logical_positions": prefix.logical_positions,
            "physical_positions": prefix.physical_positions,
            "row_hashes": snapshot_hashes(prefix.snapshot),
        },
        "generation": {
            key: value for key, value in asdict(generation).items()
            if key != "snapshot"
        },
    }


def run_generated_forced_identity(model, tokenizer, fixture: dict,
                                  eos_ids: Sequence[int]) -> dict:
    prefix_ids = generation_prefix_ids(tokenizer, fixture["messages"])
    branches = []
    for repeat in range(2):
        generated_prefix = execute_prefix_block(
            model, prefix_ids, label="identity_generation_prefix")
        generated = greedy_generate_q1(
            model, generated_prefix.snapshot, generated_prefix.last_logits,
            logical_start=generated_prefix.logical_end, eos_ids=eos_ids)
        forced_prefix = execute_prefix_block(
            model, prefix_ids, label="identity_generation_prefix")
        forced = force_content_q1(
            model, forced_prefix.snapshot, forced_prefix.last_logits,
            content_ids=generated.content_ids,
            logical_start=forced_prefix.logical_end, eos_ids=eos_ids)
        branches.append(require_generated_forced_identity(
            generated_prefix, generated, forced_prefix, forced,
            content_start=generated_prefix.physical_end))
    _require(branches[0] == branches[1], "identity repeat differs")
    return {"status": "PASS", "repeat_count": 2, "evidence": branches}


def run_fresh_self_replacement(model, plan) -> dict:
    direct = execute_fresh_plan(model, plan)
    rows = []
    for region in (R1, R2, R3):
        start, end = plan.physical_regions.interval(region)
        fresh_rows = extract_rows(direct.snapshot, start, end)
        for mode, use_keys, use_values in (
                ("K+V", True, True), ("K-only", True, False),
                ("V-only", False, True)):
            boundary = execute_fresh_plan(model, plan, stop_at=end)
            replaced, insertion = replace_rows(
                boundary.snapshot, fresh_rows, start,
                use_keys=use_keys, use_values=use_values)
            completed = continue_fresh_plan(model, plan, replaced, start_at=end)
            _require(snapshot_hashes(completed.snapshot) ==
                     snapshot_hashes(direct.snapshot),
                     f"fresh self-replacement rows differ for {region}/{mode}")
            _require(tensor_sha256(completed.last_logits) ==
                     tensor_sha256(direct.last_logits),
                     f"fresh self-replacement logits differ for {region}/{mode}")
            rows.append({
                "region": region, "mode": mode, "status": "PASS",
                "fresh_selected_row_hashes": snapshot_hashes(fresh_rows),
                "boundary_before_hashes": snapshot_hashes(boundary.snapshot),
                "boundary_after_hashes": snapshot_hashes(replaced),
                "insertion": insertion,
                "continuation_calls": completed.calls,
                "continued_snapshot_hashes": snapshot_hashes(completed.snapshot),
                "continued_last_logits_sha256": tensor_sha256(
                    completed.last_logits),
            })
    return {
        "status": "PASS", "regions": rows,
        "direct_calls": direct.calls,
        "direct_snapshot_hashes": snapshot_hashes(direct.snapshot),
        "direct_last_logits_sha256": tensor_sha256(direct.last_logits),
    }


def _transplant_complete(model, plan, source_rows, region: str):
    start, end = plan.physical_regions.interval(region)
    boundary = execute_fresh_plan(model, plan, stop_at=end)
    replaced, insertion = replace_rows(
        boundary.snapshot, source_rows, start, use_keys=True, use_values=True)
    return continue_fresh_plan(model, plan, replaced, start_at=end), insertion


def run_natural_calibration(model, tokenizer, fixture: dict,
                            eos_ids: Sequence[int]) -> dict:
    middle = fixture["middle_end_msg"]
    green_history, amber_history = fixture["correct"], fixture["wrong"]
    green_plan = build_role_native_plan(tokenizer, green_history, middle_end_msg=middle)
    amber_plan = build_role_native_plan(tokenizer, amber_history, middle_end_msg=middle)
    fresh_plan = build_fresh_destination_plan(
        tokenizer, green_history, middle_end_msg=middle)
    _require(fresh_plan == build_fresh_destination_plan(
        tokenizer, amber_history, middle_end_msg=middle),
        "natural calibration fresh plans differ")
    green = execute_replay_plan(model, green_plan)
    amber = execute_replay_plan(model, amber_plan)
    fresh = execute_fresh_plan(model, fresh_plan)
    approve_text = fixture["correct_target"]
    deny_text = fixture["counterfactual_target"]
    approve_id = tokenizer.encode(approve_text, add_special_tokens=False)
    deny_id = tokenizer.encode(deny_text, add_special_tokens=False)
    _require(len(approve_id) == len(deny_id) == 1, "calibration targets differ")
    source_start, source_end = green_plan.regions.interval(R2)
    green_rows = extract_rows(green.snapshot, source_start, source_end)
    amber_rows = extract_rows(amber.snapshot, source_start, source_end)
    tg, tg_insert = _transplant_complete(model, fresh_plan, green_rows, R2)
    ta, ta_insert = _transplant_complete(model, fresh_plan, amber_rows, R2)
    compact = compact_messages(green_history, middle)
    raw = {
        "A_g": _probe_record(model, tokenizer, green.snapshot,
            source_messages(green_history, middle), green_plan.token_ids,
            len(green_plan.token_ids), fixture["probe"], approve_text, deny_text,
            eos_ids),
        "A_a": _probe_record(model, tokenizer, amber.snapshot,
            source_messages(amber_history, middle), amber_plan.token_ids,
            len(amber_plan.token_ids), fixture["probe"], approve_text, deny_text,
            eos_ids),
        "F": _probe_record(model, tokenizer, fresh.snapshot, compact,
            fresh_plan.token_ids, fresh_plan.logical_positions[-1] + 1,
            fixture["probe"], approve_text, deny_text, eos_ids),
        "T_g": _probe_record(model, tokenizer, tg.snapshot, compact,
            fresh_plan.token_ids, fresh_plan.logical_positions[-1] + 1,
            fixture["probe"], approve_text, deny_text, eos_ids),
        "T_a": _probe_record(model, tokenizer, ta.snapshot, compact,
            fresh_plan.token_ids, fresh_plan.logical_positions[-1] + 1,
            fixture["probe"], approve_text, deny_text, eos_ids),
    }
    summary = summarize_natural_calibration(
        raw, approve_id=approve_id, deny_id=deny_id,
        special_ids=getattr(tokenizer, "all_special_ids", ()))
    return {**summary, "raw": raw,
            "green_insertion": tg_insert, "amber_insertion": ta_insert}


def _natural_generation_ok(record: dict, target_ids: Sequence[int],
                           special_ids: Sequence[int]) -> bool:
    generation = record["generation"]
    content = generation.get("content_ids", [])
    eos = set(generation.get("eos_ids", []))
    special = set(int(value) for value in special_ids)
    return (
        bool(content) and content[:len(target_ids)] == list(target_ids) and
        generation.get("stop_reason") == "model_eos" and
        generation.get("cap_hit") is False and
        generation.get("stop_candidate_id") in eos and
        not special.intersection(content)
    )


def summarize_natural_calibration(raw: dict, *, approve_id: Sequence[int],
                                  deny_id: Sequence[int],
                                  special_ids: Sequence[int]) -> dict:
    _require(set(raw) == {"A_g", "A_a", "F", "T_g", "T_a"},
             "natural calibration does not contain exactly five cells")
    margins = {key: float(raw[key]["margin"]) for key in raw}
    _require(all(torch.isfinite(torch.tensor(value)).item()
                 for value in margins.values()),
             "natural calibration margin is nonfinite")
    green_denominator = margins["A_g"] - margins["F"]
    amber_denominator = margins["F"] - margins["A_a"]
    denominators_positive = green_denominator > 0 and amber_denominator > 0
    rho_green = ((margins["T_g"] - margins["F"]) / green_denominator
                 if green_denominator > 0 else None)
    rho_amber = ((margins["F"] - margins["T_a"]) / amber_denominator
                 if amber_denominator > 0 else None)
    green_generated = _natural_generation_ok(
        raw["A_g"], approve_id, special_ids)
    amber_generated = _natural_generation_ok(
        raw["A_a"], deny_id, special_ids)
    checks = {
        "green_oracle_margin_positive": margins["A_g"] > 0,
        "amber_oracle_margin_negative": margins["A_a"] < 0,
        "denominators_positive": denominators_positive,
        "green_generated_exact_target_and_normal_stop": green_generated,
        "amber_generated_exact_target_and_normal_stop": amber_generated,
        "rho_green_at_least_half": rho_green is not None and rho_green >= 0.5,
        "rho_amber_at_least_half": rho_amber is not None and rho_amber >= 0.5,
    }
    return {
        "status": "PASS" if all(checks.values()) else "ADVERSE",
        "checks": checks, "margins": margins,
        "green_denominator": green_denominator,
        "amber_denominator": amber_denominator,
        "rho_green": rho_green, "rho_amber": rho_amber,
    }


def run_path_control(model, tokenizer, fixture: dict) -> dict:
    plan = build_fresh_destination_plan(
        tokenizer, fixture["correct"], middle_end_msg=fixture["middle_end_msg"])
    suffix = probe_suffix_ids(
        tokenizer, compact_messages(fixture["correct"], fixture["middle_end_msg"]),
        plan.token_ids, fixture["probe"])
    correct_ids = tokenizer.encode(
        fixture["correct_target"], add_special_tokens=False)
    counterfactual_ids = tokenizer.encode(
        fixture["counterfactual_target"], add_special_tokens=False)
    _require(len(correct_ids) == len(counterfactual_ids) == 1,
             "path-control targets must each be one production token")
    return run_bidirectional_path_control(
        model, plan, region=R2, suffix_ids=suffix,
        correct_id=correct_ids[0], counterfactual_id=counterfactual_ids[0])
