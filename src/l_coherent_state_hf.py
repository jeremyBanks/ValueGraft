"""Qwen3-0.6B production-path ladder for coherent-summary-state apparatus."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from coherent_state_cases import (
    correct_source_messages,
    fresh_source_messages,
    wrong_source_messages,
)
from coherent_state_hf import (
    compare_rows,
    delta_deranged_snapshot,
    move_key_rows,
    replace_summary_rows,
    row_hashes,
)
from coherent_state_runtime import (
    ARM_NAMES,
    arm_snapshot,
    build_fresh_destination,
    capture_forced_summary,
    capture_generated_summary,
    score_arm,
    score_target,
    validate_generated_replay,
)
from coherent_state_store import checkpoint_path, read_checkpoint, save_render
from coherent_state_tokens import probe_layout, rendered_assistant_content_ids
from kvlib_hf import prefill, rebuild_cache, snapshot_cache


MODEL = "Qwen/Qwen3-0.6B"
REQUEST = "Write a short context summary. Output only the summary."
SUMMARY = "The approved label remains on file."


def fake_conv(cid: str, label: str, marker: str) -> dict:
    return {
        "id": cid,
        "messages": [
            {"role": "system", "content": "Maintain the user's test record."},
            {"role": "user", "content":
             f"The approved label is {label}. Remember that exact label."},
            {"role": "assistant", "content":
             f"Understood; the approved label is {label}."},
            {"role": "user", "content": f"Unrelated tail marker: {marker}."},
            {"role": "assistant", "content": "Tail marker noted."},
        ],
        "sections": {"middle_end_msg": 3},
    }


def engineered_gradient_control(model, tokenizer, fresh_snapshot, layout,
                                plant, targets):
    """Take a summary-V-only ascent step on a frozen downstream margin."""
    correct_layout = probe_layout(
        tokenizer, layout.messages, layout.context_ids,
        plant["probe"], targets[plant["id"]]["correct"])
    wrong_layout = probe_layout(
        tokenizer, layout.messages, layout.context_ids,
        plant["probe"], targets[plant["id"]]["counterfactual"])
    if correct_layout.suffix_ids != wrong_layout.suffix_ids:
        raise RuntimeError("positive-control probes do not share a prefix")
    if len(correct_layout.target_ids) != 1 or len(wrong_layout.target_ids) != 1:
        raise RuntimeError("positive-control A/B targets must each be one token")

    grad_values = []
    grad_snapshot = []
    for k, v in fresh_snapshot:
        vg = v.detach().clone().requires_grad_(True)
        grad_values.append(vg)
        grad_snapshot.append((k.detach(), vg))
    cache = rebuild_cache(grad_snapshot, DynamicCache)
    suffix = correct_layout.suffix_ids
    ids = torch.tensor([suffix], device=model.device)
    pos = torch.arange(len(layout.context_ids),
                       len(layout.context_ids) + len(suffix),
                       device=model.device)[None]
    out = model(input_ids=ids, past_key_values=cache, position_ids=pos,
                use_cache=True, logits_to_keep=1)
    lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    objective = (lp[correct_layout.target_ids[0]] -
                 lp[wrong_layout.target_ids[0]])
    objective.backward()
    grads = [v.grad for v in grad_values]
    if any(g is None for g in grads):
        raise RuntimeError("downstream margin did not backpropagate to summary V")
    s0, s1 = layout.summary_start, layout.summary_end
    sq = sum(float(g[..., s0:s1, :].float().square().sum()) for g in grads)
    n = sum(g[..., s0:s1, :].numel() for g in grads)
    rms = (sq / n) ** 0.5
    if not rms > 0:
        raise RuntimeError("downstream summary-V gradient is zero")

    baseline = score_arm(
        model, tokenizer, fresh_snapshot, layout.messages, layout.context_ids,
        [plant], targets)["conversation_margin"]
    attempts = []
    for epsilon in (0.01, 0.03, 0.1, 0.3, 1.0):
        treated = []
        for (k, v), g in zip(fresh_snapshot, grads):
            v2 = v.clone()
            v2[..., s0:s1, :] += (
                epsilon * g[..., s0:s1, :] / rms).to(v2.dtype)
            treated.append((k.clone(), v2))
        margin = score_arm(
            model, tokenizer, treated, layout.messages, layout.context_ids,
            [plant], targets)["conversation_margin"]
        attempts.append({"epsilon": epsilon, "margin": margin,
                         "gain": margin - baseline})
        if margin > baseline + 1e-5:
            return {"passes": True, "baseline_margin": baseline,
                    "gradient_rms": rms, "attempts": attempts}
    raise RuntimeError(f"engineered downstream V control did not move: {attempts}")


def run_loaded_kernel_gates(model, tokenizer, *, identity_tolerance: float,
                            rotation_tolerance: float,
                            placebo_quantization_tolerance: float = 0.05,
                            placebo_moment_tolerance: float = 0.02) -> dict:
    """Production-config gate run on the already loaded model before semantics."""
    cfg = getattr(model.config, "text_config", model.config)
    theta = float((getattr(cfg, "rope_parameters", None) or {})["rope_theta"])
    device = model.device

    # Native-shift identity: same token sequence, same relative positions, one
    # explicit absolute offset. Values should be unchanged; post-RoPE keys should
    # match after exact inverse rotation at every layer.
    seq = tokenizer("alpha beta gamma delta epsilon", add_special_tokens=False).input_ids
    ids = torch.tensor([seq], device=device)
    pos0 = torch.arange(len(seq), device=device)[None]
    pos37 = pos0 + 37
    cache0, _ = prefill(model, ids, position_ids=pos0)
    cache37, _ = prefill(model, ids, position_ids=pos37)
    snap0, snap37 = snapshot_cache(cache0), snapshot_cache(cache37)
    shifted_back = move_key_rows(snap37, -37, theta)
    native_rows = compare_rows(snap0, shifted_back)
    native_k = max(x["k_max_abs"] for x in native_rows)
    native_v = max(x["v_max_abs"] for x in native_rows)
    zero_rows = compare_rows(snap0, move_key_rows(snap0, 0, theta))
    zero_k = max(x["k_max_abs"] for x in zero_rows)
    roundtrip_rows = compare_rows(
        snap0, move_key_rows(move_key_rows(snap0, 37, theta), -37, theta))
    roundtrip_k = max(x["k_max_abs"] for x in roundtrip_rows)
    # This compares different absolute-position executions, so both K and V
    # inherit the position-shift kernel floor. It is distinct from same-prefix
    # generated/replay identity, which keeps the stricter identity tolerance.
    if zero_k != 0 or native_k > rotation_tolerance or native_v > rotation_tolerance:
        raise RuntimeError(
            f"loaded K/V shift gate failed: zero={zero_k} nativeK={native_k} "
            f"nativeV={native_v}")
    if roundtrip_k > rotation_tolerance:
        raise RuntimeError(f"loaded K roundtrip failed: {roundtrip_k}")

    # Snapshot/rebuild continuation identity through the production kernel.
    next_id = tokenizer(" z", add_special_tokens=False).input_ids[:1]
    if not next_id:
        raise RuntimeError("empty continuation token in loaded gate")
    nxt = torch.tensor([next_id], device=device)
    nxtpos = torch.tensor([[len(seq)]], device=device)
    with torch.no_grad():
        out1 = model(input_ids=nxt, past_key_values=cache0,
                     position_ids=nxtpos, use_cache=True)
        out2 = model(input_ids=nxt, past_key_values=rebuild_cache(snap0, DynamicCache),
                     position_ids=nxtpos, use_cache=True)
    rebuild_logits = float((out1.logits.float() - out2.logits.float()).abs().max())
    if rebuild_logits > identity_tolerance:
        raise RuntimeError(f"snapshot/rebuild logits differ: {rebuild_logits}")

    # Generated source-of-record and independent one-token replay.
    gen_messages = [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content":
         "Reply with the single word OK and then stop. Do not explain."},
    ]
    generated = capture_generated_summary(model, tokenizer, gen_messages,
                                           max_tokens=192)
    replay = capture_forced_summary(
        model, tokenizer, gen_messages, generated.summary_ids,
        source_kind="loaded_gate_replay")
    replay_identity = validate_generated_replay(
        generated, replay, identity_tolerance)
    generated.cache = None
    replay.cache = None

    # Exact summary span, self-replacement, downstream scoring, and placebo.
    target = fake_conv("c10", "A", "target-tail")
    donor = fake_conv("c02", "B", "donor-tail")
    fresh_messages = fresh_source_messages(target, REQUEST)
    summary_ids = rendered_assistant_content_ids(tokenizer, fresh_messages, SUMMARY)
    correct = capture_forced_summary(
        model, tokenizer, correct_source_messages(target, REQUEST), summary_ids,
        source_kind="loaded_gate_correct")
    wrong = capture_forced_summary(
        model, tokenizer, wrong_source_messages(target, donor, REQUEST), summary_ids,
        source_kind="loaded_gate_wrong")
    layout, _trace, fresh_snapshot, fresh_rows = build_fresh_destination(
        model, tokenizer, target, SUMMARY, summary_ids, REQUEST)
    correct.cache = None
    wrong.cache = None
    plant = {"id": "loaded-gate", "category": "referent",
             "probe": "Which label is approved? Answer with only A or B."}
    targets = {"loaded-gate": {"correct": "A", "counterfactual": "B",
                                "basis": "loaded production gate"}}
    fresh_score = score_arm(model, tokenizer, fresh_snapshot, layout.messages,
                            layout.context_ids, [plant], targets)
    base_hashes_before = row_hashes(fresh_snapshot)
    consumed_branch = list(fresh_snapshot)
    consumed_score = score_target(
        model, tokenizer, consumed_branch, layout.messages, layout.context_ids,
        plant["probe"], targets["loaded-gate"]["correct"],
        consume_snapshot=True)
    if consumed_branch:
        raise RuntimeError("bounded branch ownership was not transferred")
    base_hashes_after = row_hashes(fresh_snapshot)
    expected_correct = fresh_score["plants"][0]["correct"]
    bounded_lp_diff = max(abs(a - b) for a, b in zip(
        consumed_score["token_logprobs"], expected_correct["token_logprobs"]))
    if base_hashes_before != base_hashes_after or bounded_lp_diff > identity_tolerance:
        raise RuntimeError(
            f"bounded scoring changed base or scores: lp={bounded_lp_diff}")
    self_snapshot = replace_summary_rows(
        fresh_snapshot, fresh_rows, layout.summary_start,
        use_keys=True, use_values=True)
    self_score = score_arm(model, tokenizer, self_snapshot, layout.messages,
                           layout.context_ids, [plant], targets)
    f_lps = [lp for row in fresh_score["plants"]
             for side in ("correct", "counterfactual")
             for lp in row[side]["token_logprobs"]]
    s_lps = [lp for row in self_score["plants"]
             for side in ("correct", "counterfactual")
             for lp in row[side]["token_logprobs"]]
    noop = max(abs(a - b) for a, b in zip(f_lps, s_lps))
    if noop > identity_tolerance:
        raise RuntimeError(f"loaded tokenwise no-op failed: {noop}")
    irrelevant_no_state = replace_summary_rows(
        fresh_snapshot, wrong.rows, layout.summary_start,
        use_keys=False, use_values=False)
    irrelevant_score = score_arm(
        model, tokenizer, irrelevant_no_state, layout.messages,
        layout.context_ids, [plant], targets)
    i_lps = [lp for row in irrelevant_score["plants"]
             for side in ("correct", "counterfactual")
             for lp in row[side]["token_logprobs"]]
    irrelevant_noop = max(abs(a - b) for a, b in zip(f_lps, i_lps))
    if irrelevant_noop > identity_tolerance:
        raise RuntimeError(f"irrelevant-source no-state control failed: {irrelevant_noop}")
    delta_snapshot, delta_diag = delta_deranged_snapshot(
        fresh_snapshot, correct.rows, layout.summary_start, 20_260_711)
    if (any(x.fixed_points for x in delta_diag) or
            max(x.max_multiset_diff for x in delta_diag) != 0):
        raise RuntimeError("loaded placebo derangement invariant failed")
    applied_quant = max(max(x.applied_delta_max_abs_error,
                            x.applied_multiset_diff) for x in delta_diag)
    applied_moment = max(max(x.applied_mean_diff,
                             x.applied_covariance_diff) for x in delta_diag)
    if (applied_quant > placebo_quantization_tolerance or
            applied_moment > placebo_moment_tolerance):
        raise RuntimeError(
            f"loaded placebo bf16 mismatch: quant={applied_quant} "
            f"moment={applied_moment}")
    delta_score = score_arm(model, tokenizer, delta_snapshot, layout.messages,
                            layout.context_ids, [plant], targets)

    return {
        "passes": True,
        "zero_rotation_max_abs": zero_k,
        "roundtrip_rotation_max_abs": roundtrip_k,
        "native_shift_k_max_abs": native_k,
        "native_shift_v_max_abs": native_v,
        "snapshot_rebuild_logits_max_abs": rebuild_logits,
        "generated_replay": replay_identity,
        "generated_token_count": len(generated.summary_ids),
        "tokenwise_self_replacement_max_abs": noop,
        "bounded_scoring_tokenwise_max_abs": bounded_lp_diff,
        "bounded_scoring_base_unchanged": base_hashes_before == base_hashes_after,
        "irrelevant_source_no_state_tokenwise_max_abs": irrelevant_noop,
        "placebo_fixed_points": sum(x.fixed_points for x in delta_diag),
        "placebo_intended_multiset_max_diff": max(
            x.max_multiset_diff for x in delta_diag),
        "placebo_applied_quantization_max_abs": applied_quant,
        "placebo_applied_moment_max_abs": applied_moment,
        "fresh_margin": fresh_score["conversation_margin"],
        "delta_margin": delta_score["conversation_margin"],
    }


def run_ladder() -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    cfg = getattr(model.config, "text_config", model.config)
    theta = float((getattr(cfg, "rope_parameters", None) or {})["rope_theta"])
    loaded_gates = run_loaded_kernel_gates(
        model, tokenizer, identity_tolerance=1e-5, rotation_tolerance=1e-3)

    # Actual greedy generation mutates the source cache; an independent forced
    # replay must reproduce its rows and token likelihoods exactly.
    gen_messages = [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content":
         "Reply with the single word OK and then stop. Do not explain."},
    ]
    generated = capture_generated_summary(
        model, tokenizer, gen_messages, max_tokens=192)
    replay = capture_forced_summary(
        model, tokenizer, gen_messages, generated.summary_ids,
        source_kind="ladder_replay")
    generated_replay = validate_generated_replay(generated, replay, 1e-5)
    generated.cache = None
    replay.cache = None

    target = fake_conv("c10", "A", "target-tail")
    donor = fake_conv("c02", "B", "donor-tail")
    fresh_messages = fresh_source_messages(target, REQUEST)
    summary_ids = rendered_assistant_content_ids(
        tokenizer, fresh_messages, SUMMARY)
    correct = capture_forced_summary(
        model, tokenizer, correct_source_messages(target, REQUEST), summary_ids,
        source_kind="ladder_correct_forced")
    wrong = capture_forced_summary(
        model, tokenizer, wrong_source_messages(target, donor, REQUEST),
        summary_ids, source_kind="ladder_wrong_forced")
    layout, fresh_trace, fresh_snapshot, fresh_rows = build_fresh_destination(
        model, tokenizer, target, SUMMARY, summary_ids, REQUEST)
    correct.cache = None
    wrong.cache = None

    self_replaced = replace_summary_rows(
        fresh_snapshot, fresh_rows, layout.summary_start,
        use_keys=True, use_values=True)
    self_exact = all(torch.equal(a, b) and torch.equal(c, d)
                     for (a, c), (b, d) in zip(fresh_snapshot, self_replaced))
    if not self_exact:
        raise RuntimeError("fresh self-replacement changed a tensor")

    correct_delta = layout.summary_start - correct.summary_start
    wrong_delta = layout.summary_start - wrong.summary_start
    moved = move_key_rows(correct.rows, correct_delta, theta)
    roundtrip = move_key_rows(moved, -correct_delta, theta)
    rotation_rows = compare_rows(correct.rows, roundtrip)
    rotation_max = max(row["k_max_abs"] for row in rotation_rows)
    if rotation_max > 1e-3:
        raise RuntimeError(f"rotation roundtrip too large: {rotation_max}")

    plant = {"id": "ladder", "category": "referent",
             "probe": "Which label is approved? Answer with only A or B."}
    targets = {"ladder": {"correct": "A", "counterfactual": "B",
                           "basis": "engineered ladder record"}}
    outcomes = {}
    placebo_diagnostics = None
    for arm in ARM_NAMES[1:]:
        snap, diag = arm_snapshot(
            arm, fresh_snapshot, correct.rows, wrong.rows,
            layout.summary_start, correct_delta, wrong_delta, theta, 20_260_711)
        score = score_arm(model, tokenizer, snap, layout.messages,
                          layout.context_ids, [plant], targets)
        outcomes[arm] = score["conversation_margin"]
        if arm == "D_delta":
            placebo_diagnostics = diag
    if not all(torch.isfinite(torch.tensor(v)) for v in outcomes.values()):
        raise RuntimeError("non-finite production-path arm outcome")
    if not placebo_diagnostics or any(x["fixed_points"] for x in placebo_diagnostics):
        raise RuntimeError("delta placebo derangement failed")
    if max(x["max_multiset_diff"] for x in placebo_diagnostics) > 1e-6:
        raise RuntimeError("delta placebo changed its row multiset")

    positive = engineered_gradient_control(
        model, tokenizer, fresh_snapshot, layout, plant, targets)

    # A killed run after rendering must load identical text and never rerender.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fp = {"model": MODEL, "test": "resume"}
        path = checkpoint_path(root, 1, "c10")
        saved = save_render(path, fingerprint=fp, order_position=1,
                            conversation=target, reply_records=[])
        loaded = read_checkpoint(path, fp, "rendered")
        resume_exact = saved == loaded and loaded["conversation"] == target
    if not resume_exact:
        raise RuntimeError("render resume was not exact")

    return {
        "status": "PASS", "model": MODEL,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "dtype": str(next(model.parameters()).dtype),
        "generated_summary_token_count": len(generated.summary_ids),
        "generated_replay": generated_replay,
        "forced_summary_ids": summary_ids,
        "fresh_trace": asdict(fresh_trace),
        "self_replacement_exact": self_exact,
        "rotation_roundtrip_max_abs": rotation_max,
        "arm_outcomes": outcomes,
        "placebo": {
            "n_diagnostics": len(placebo_diagnostics),
            "max_multiset_diff": max(
                x["max_multiset_diff"] for x in placebo_diagnostics),
            "max_mean_diff": max(x["mean_diff"] for x in placebo_diagnostics),
            "max_covariance_diff": max(
                x["covariance_diff"] for x in placebo_diagnostics),
            "fixed_points": sum(x["fixed_points"] for x in placebo_diagnostics),
            "applied_quantization_max_abs": max(
                max(x["applied_delta_max_abs_error"], x["applied_multiset_diff"])
                for x in placebo_diagnostics),
            "applied_moment_max_abs": max(
                max(x["applied_mean_diff"], x["applied_covariance_diff"])
                for x in placebo_diagnostics),
        },
        "engineered_downstream_positive_control": positive,
        "render_resume_exact": resume_exact,
        "loaded_kernel_gates": loaded_gates,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    print(f"RUN coherent_state_ladder model={MODEL} -> {args.output}", flush=True)
    try:
        result = run_ladder()
    except Exception as exc:
        result = {"status": "FAIL", "error_type": type(exc).__name__,
                  "error": str(exc), "traceback": traceback.format_exc(),
                  "failed_at": datetime.now(timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
