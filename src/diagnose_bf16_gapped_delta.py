"""Persist the BF16 eager gapped-delta placebo invariant diagnostic.

This is an additive technical diagnostic, not a semantic experiment.  It uses
the production gapped destination construction and ``G_delta`` arm dispatcher
on the exact Qwen3-0.6B ladder fixture.  The complete diagnostic is persisted
before a frozen invariant failure causes a nonzero exit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import huggingface_hub
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from coherent_state_cases import correct_source_messages, fresh_source_messages
from coherent_state_hf import row_hashes, sha256_ids
from coherent_state_runtime import (
    build_gapped_fresh_boundary,
    capture_forced_prefix_ids,
    capture_forced_summary,
    gapped_arm_boundary,
)
from coherent_state_tokens import (
    matched_wrong_prefix_ids,
    rendered_assistant_content_ids,
)
from l_coherent_state_hf import MODEL, REQUEST, SUMMARY, fake_conv


BACKEND = "eager"
DTYPE = torch.bfloat16
PLACEBO_SEED = 20_260_711
QUANTIZATION_MAX_LIMIT = 0.05
MOMENT_MAX_LIMIT = 0.02


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results/coherent_state_diagnostics") / (
        f"bf16_gapped_delta_{MODEL.split('/')[-1]}_{stamp}.json")


def persist(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def config_record(model) -> dict[str, Any]:
    config = model.config.to_dict()
    encoded = json.dumps(config, sort_keys=True, default=str).encode()
    text_config = getattr(model.config, "text_config", model.config)
    return {
        "model_id": MODEL,
        "checkpoint_revision": getattr(model.config, "_commit_hash", None),
        "name_or_path": getattr(model.config, "_name_or_path", None),
        "requested_attention_backend": BACKEND,
        "resolved_attention_backend": getattr(
            model.config, "_attn_implementation", None),
        "resolved_text_attention_backend": getattr(
            text_config, "_attn_implementation", None),
        "requested_dtype": str(DTYPE),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "device": str(model.device),
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "architecture": config.get("architectures"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "head_dim": config.get("head_dim"),
    }


def aggregate_diagnostics(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    if not diagnostics:
        raise RuntimeError("G_delta returned no derangement diagnostics")
    maxima = {
        "raw_multiset_max_abs": max(
            row["max_multiset_diff"] for row in diagnostics),
        "raw_mean_max_abs": max(row["mean_diff"] for row in diagnostics),
        "raw_covariance_max_abs": max(
            row["covariance_diff"] for row in diagnostics),
        "applied_delta_quantization_max_abs": max(
            row["applied_delta_max_abs_error"] for row in diagnostics),
        "applied_multiset_max_abs": max(
            row["applied_multiset_diff"] for row in diagnostics),
        "applied_mean_max_abs": max(
            row["applied_mean_diff"] for row in diagnostics),
        "applied_covariance_max_abs": max(
            row["applied_covariance_diff"] for row in diagnostics),
        "fixed_points": sum(row["fixed_points"] for row in diagnostics),
    }
    maxima["applied_quantization_max_abs"] = max(
        maxima["applied_delta_quantization_max_abs"],
        maxima["applied_multiset_max_abs"])
    maxima["applied_moment_max_abs"] = max(
        maxima["applied_mean_max_abs"],
        maxima["applied_covariance_max_abs"])
    return maxima


def run(payload: dict[str, Any], output: Path, device: str) -> bool:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        local_files_only=True,
        dtype=DTYPE,
        attn_implementation=BACKEND,
    ).to(device)
    model.eval()
    payload["model"] = config_record(model)
    persist(output, payload)

    if payload["model"]["parameter_dtype"] != str(DTYPE):
        raise RuntimeError("loaded parameter dtype differs from frozen BF16")
    if payload["model"]["resolved_attention_backend"] != BACKEND:
        raise RuntimeError("loaded attention backend differs from frozen eager")

    target = fake_conv("c10", "A", "target-tail")
    donor = fake_conv("c13", "B", "donor-tail")
    fresh_messages = fresh_source_messages(target, REQUEST)
    summary_ids = rendered_assistant_content_ids(
        tokenizer, fresh_messages, SUMMARY)
    correct = capture_forced_summary(
        model, tokenizer, correct_source_messages(target, REQUEST),
        summary_ids, source_kind="diagnostic_gapped_correct")
    matched = matched_wrong_prefix_ids(tokenizer, target, donor, REQUEST)
    if matched.correct_ids != correct.prefix_ids:
        raise RuntimeError("matched-wrong baseline differs from correct source")
    wrong = capture_forced_prefix_ids(
        model, tokenizer, matched.wrong_ids, summary_ids,
        source_kind="diagnostic_gapped_wrong",
        summary_start=correct.summary_start)
    layout, _trace, fresh_boundary, fresh_rows = build_gapped_fresh_boundary(
        model, tokenizer, target, SUMMARY, summary_ids, REQUEST,
        correct.prefix_ids)

    payload["fixture"] = {
        "source": "exact Qwen3-0.6B production-ladder gapped fixture",
        "request": REQUEST,
        "summary_text": SUMMARY,
        "target_conversation": target,
        "external_wrong_donor": donor,
        "summary_token_ids": list(summary_ids),
        "summary_token_sha256": sha256_ids(summary_ids),
        "correct_prefix_token_count": len(correct.prefix_ids),
        "correct_prefix_sha256": correct.prefix_sha256,
        "wrong_prefix_token_count": len(wrong.prefix_ids),
        "wrong_prefix_sha256": wrong.prefix_sha256,
        "wrong_prefix_length_equal": len(correct.prefix_ids) == len(wrong.prefix_ids),
        "source_summary_start": correct.summary_start,
        "physical_summary_start": layout.physical_summary_start,
        "physical_summary_end": layout.physical_summary_end,
        "logical_gap": layout.source_summary_start - layout.physical_summary_start,
        "placebo_seed": PLACEBO_SEED,
        "actual_dispatch_path": (
            "coherent_state_runtime.gapped_arm_boundary('G_delta') -> "
            "coherent_state_hf.delta_deranged_snapshot"),
    }
    payload["source_row_hashes"] = {
        "correct": correct.row_hashes,
        "wrong": wrong.row_hashes,
        "fresh_summary": row_hashes(fresh_rows),
    }
    payload["fresh_boundary_hashes_before"] = row_hashes(fresh_boundary)
    persist(output, payload)

    delta_boundary, diagnostics = gapped_arm_boundary(
        "G_delta", fresh_boundary, correct.rows, wrong.rows,
        layout.physical_summary_start, PLACEBO_SEED)
    aggregate = aggregate_diagnostics(diagnostics)

    end = layout.physical_summary_end
    preservation = {
        "all_keys_bit_exact_to_fresh": True,
        "all_pre_summary_values_bit_exact_to_fresh": True,
        "all_post_summary_values_bit_exact_to_fresh": True,
    }
    for (fresh_k, fresh_v), (delta_k, delta_v) in zip(
            fresh_boundary, delta_boundary, strict=True):
        preservation["all_keys_bit_exact_to_fresh"] &= torch.equal(
            fresh_k, delta_k)
        preservation["all_pre_summary_values_bit_exact_to_fresh"] &= \
            torch.equal(fresh_v[..., :layout.physical_summary_start, :],
                        delta_v[..., :layout.physical_summary_start, :])
        preservation["all_post_summary_values_bit_exact_to_fresh"] &= \
            torch.equal(fresh_v[..., end:, :], delta_v[..., end:, :])

    checks = {
        "zero_fixed_points": aggregate["fixed_points"] == 0,
        "raw_multiset_exact": aggregate["raw_multiset_max_abs"] == 0,
        "applied_quantization_within_frozen_limit": (
            aggregate["applied_quantization_max_abs"] <=
            QUANTIZATION_MAX_LIMIT),
        "applied_moments_within_frozen_limit": (
            aggregate["applied_moment_max_abs"] <= MOMENT_MAX_LIMIT),
        "non_summary_and_keys_preserved": all(preservation.values()),
    }
    payload["g_delta"] = {
        "arm": "G_delta",
        "technical_diagnostic_not_semantic_outcome": True,
        "frozen_limits": {
            "applied_quantization_max_abs": QUANTIZATION_MAX_LIMIT,
            "applied_moment_max_abs": MOMENT_MAX_LIMIT,
            "comparison": "less_than_or_equal",
        },
        "aggregate": aggregate,
        "checks": checks,
        "boundary_preservation": preservation,
        "per_layer_head_diagnostics": diagnostics,
        "delta_boundary_hashes": row_hashes(delta_boundary),
    }
    passed = all(checks.values())
    payload["status"] = "passed" if passed else "failed_frozen_invariant"
    payload["passes_frozen_invariants"] = passed
    payload["completed_at"] = utc_now()
    persist(output, payload)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or default_output_path()
    print(
        f"RUN bf16-gapped-delta model={MODEL} backend={BACKEND} "
        f"dtype={DTYPE} -> {output}", flush=True)
    payload: dict[str, Any] = {
        "schema": 1,
        "kind": "technical_bf16_gapped_delta_diagnostic_not_semantic_outcome",
        "status": "running",
        "passes_frozen_invariants": False,
        "started_at": utc_now(),
        "apparatus_git": {
            "commit": git_value("rev-parse", "HEAD"),
            "tree_status": git_value("status", "--short"),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "requested_device": args.device,
            "mps_available": torch.backends.mps.is_available(),
        },
    }
    persist(output, payload)
    try:
        passed = run(payload, output, args.device)
    except Exception as exc:
        payload["status"] = "execution_error"
        payload["passes_frozen_invariants"] = False
        payload["completed_at"] = utc_now()
        payload["failure"] = {
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        persist(output, payload)
        print(output, flush=True)
        raise SystemExit(2)
    print(output, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
