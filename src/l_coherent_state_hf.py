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
)
from coherent_state_runtime import (
    ARM_NAMES,
    arm_snapshot,
    build_fresh_destination,
    capture_forced_summary,
    capture_generated_summary,
    score_arm,
    validate_generated_replay,
)
from coherent_state_store import checkpoint_path, read_checkpoint, save_render
from coherent_state_tokens import probe_layout, rendered_assistant_content_ids
from kvlib_hf import rebuild_cache


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


def run_ladder() -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    cfg = getattr(model.config, "text_config", model.config)
    theta = float((getattr(cfg, "rope_parameters", None) or {})["rope_theta"])

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
        },
        "engineered_downstream_positive_control": positive,
        "render_resume_exact": resume_exact,
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
