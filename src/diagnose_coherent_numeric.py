"""Reproduce numerical-position diagnostics for the coherent-state apparatus.

This is not a semantic experiment.  It compares a single stored cache under
position movement, an independent native-RoPE oracle captured before rotation,
and deliberately incorrect movement.  Results are written with model and time in
the filename so every diagnostic is preserved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from coherent_state_hf import compare_rows, move_key_rows
from kvlib_hf import prefill, rebuild_cache, snapshot_cache


MODEL = "Qwen/Qwen3-0.6B"
TEXT = "alpha beta gamma delta epsilon"
DELTA = 37


def distribution_metrics(reference_logits: torch.Tensor,
                         test_logits: torch.Tensor,
                         target_ids: tuple[int, int]) -> dict:
    ref_lp = torch.log_softmax(reference_logits.float(), dim=-1)
    test_lp = torch.log_softmax(test_logits.float(), dim=-1)
    ref_p, test_p = ref_lp.exp(), test_lp.exp()
    a, b = target_ids
    return {
        "logits_max_abs": float((test_logits.float() - reference_logits.float()).abs().max()),
        "logprobs_max_abs": float((test_lp - ref_lp).abs().max()),
        "target_a_logprob_abs": float((test_lp[a] - ref_lp[a]).abs()),
        "target_b_logprob_abs": float((test_lp[b] - ref_lp[b]).abs()),
        "target_pair_margin_abs": float(abs(
            (test_lp[a] - test_lp[b]) - (ref_lp[a] - ref_lp[b]))),
        "kl_reference_to_test": float((ref_p * (ref_lp - test_lp)).sum()),
        "total_variation": float(0.5 * (ref_p - test_p).abs().sum()),
        "top1_equal": bool(reference_logits.argmax() == test_logits.argmax()),
    }


def tensor_oracle_metrics(helper_rows, oracle_rows) -> dict:
    per_layer = []
    for li, ((kh, vh), (ko, vo)) in enumerate(zip(helper_rows, oracle_rows)):
        diff = kh.float() - ko.float()
        ah = kh.float().reshape(-1, kh.shape[-1])
        ao = ko.float().reshape(-1, ko.shape[-1])
        per_layer.append({
            "layer": li,
            "k_max_abs": float(diff.abs().max()),
            "k_relative_l2": float(diff.norm() / (ko.float().norm() + 1e-30)),
            "k_min_row_cosine": float(torch.nn.functional.cosine_similarity(
                ah, ao, dim=-1).min()),
            "v_max_abs": float((vh.float() - vo.float()).abs().max()),
        })
    return {
        "k_max_abs": max(x["k_max_abs"] for x in per_layer),
        "k_max_layer_relative_l2": max(x["k_relative_l2"] for x in per_layer),
        "k_min_row_cosine": min(x["k_min_row_cosine"] for x in per_layer),
        "v_max_abs": max(x["v_max_abs"] for x in per_layer),
        "per_layer": per_layer,
    }


def run_dtype(dtype: torch.dtype, tokenizer) -> dict:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=dtype, local_files_only=True)
    model.eval()
    captured: list[dict] = []
    hooks = []
    def capture_hook(box):
        def hook(_module, _inputs, output):
            box["pre_rope_k"] = output.detach().clone()
        return hook
    for layer in model.model.layers:
        box: dict = {}
        captured.append(box)
        hooks.append(layer.self_attn.k_norm.register_forward_hook(
            capture_hook(box)))

    token_ids = tokenizer(TEXT, add_special_tokens=False).input_ids
    ids = torch.tensor([token_ids], device=model.device)
    pos = torch.arange(len(token_ids), device=model.device)[None]
    cache0, _ = prefill(model, ids, position_ids=pos)
    rows0 = snapshot_cache(cache0)
    for hook in hooks:
        hook.remove()

    cache_shifted, _ = prefill(model, ids, position_ids=pos + DELTA)
    rows_shifted = snapshot_cache(cache_shifted)
    theta = float(model.config.rope_parameters["rope_theta"])
    shifted_back = move_key_rows(rows_shifted, -DELTA, theta)
    native_rows = compare_rows(rows0, shifted_back)

    helper_rows = move_key_rows(rows0, DELTA, theta)
    oracle_rows = []
    source_oracle_rows = []
    for box, (stored_k, stored_v) in zip(captured, rows0):
        pre_k = box["pre_rope_k"].transpose(1, 2)
        source_cos, source_sin = model.model.rotary_emb(pre_k, pos)
        dest_cos, dest_sin = model.model.rotary_emb(pre_k, pos + DELTA)
        _, source_k = apply_rotary_pos_emb(
            pre_k, pre_k, source_cos, source_sin)
        _, dest_k = apply_rotary_pos_emb(pre_k, pre_k, dest_cos, dest_sin)
        source_oracle_rows.append((source_k, stored_v.clone()))
        oracle_rows.append((dest_k, stored_v.clone()))
    source_oracle_diff = compare_rows(rows0, source_oracle_rows)

    roundtrip = move_key_rows(helper_rows, -DELTA, theta)
    roundtrip_rows = compare_rows(rows0, roundtrip)

    feed_ids = tokenizer(" z", add_special_tokens=False).input_ids
    target_a = tokenizer(" A", add_special_tokens=False).input_ids
    target_b = tokenizer(" B", add_special_tokens=False).input_ids
    if not feed_ids or len(target_a) != 1 or len(target_b) != 1:
        raise RuntimeError("diagnostic feed/A/B tokenization changed")
    feed = torch.tensor([[feed_ids[0]]], device=model.device)
    target_pair = (target_a[0], target_b[0])

    def score(rows, position: int) -> torch.Tensor:
        with torch.no_grad():
            out = model(
                input_ids=feed,
                past_key_values=rebuild_cache(rows, DynamicCache),
                position_ids=torch.tensor([[position]], device=model.device),
                use_cache=True,
            )
        return out.logits[0, -1].detach()

    base_logits = score(rows0, len(token_ids))
    helper_logits = score(helper_rows, len(token_ids) + DELTA)
    oracle_logits = score(oracle_rows, len(token_ids) + DELTA)
    wrong_logits = score(rows0, len(token_ids) + DELTA)
    value_shift_rows = [
        (k0.clone(), vs.clone())
        for (k0, _), (_, vs) in zip(rows0, rows_shifted)
    ]
    value_shift_logits = score(value_shift_rows, len(token_ids))

    result = {
        "dtype": str(dtype),
        "native_independent_prefill": {
            "k_max_abs": max(x["k_max_abs"] for x in native_rows),
            "v_max_abs": max(x["v_max_abs"] for x in native_rows),
            "per_layer": native_rows,
        },
        "source_hook_vs_stored": {
            "k_max_abs": max(x["k_max_abs"] for x in source_oracle_diff),
            "v_max_abs": max(x["v_max_abs"] for x in source_oracle_diff),
        },
        "helper_vs_model_native_destination": tensor_oracle_metrics(
            helper_rows, oracle_rows),
        "helper_roundtrip": {
            "k_max_abs": max(x["k_max_abs"] for x in roundtrip_rows),
            "v_max_abs": max(x["v_max_abs"] for x in roundtrip_rows),
            "per_layer": roundtrip_rows,
        },
        "functional_global_shift_helper_vs_source": distribution_metrics(
            base_logits, helper_logits, target_pair),
        "functional_helper_vs_native_oracle_same_destination": distribution_metrics(
            oracle_logits, helper_logits, target_pair),
        "failure_injection_missing_k_rotation": distribution_metrics(
            base_logits, wrong_logits, target_pair),
        "value_only_independent_prefill_position_sensitivity": distribution_metrics(
            base_logits, value_shift_logits, target_pair),
    }
    if not all(math.isfinite(v) for section in result.values()
               if isinstance(section, dict) for v in section.values()
               if isinstance(v, (float, int))):
        raise RuntimeError("non-finite diagnostic")
    del model
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    payload = {
        "schema": 1,
        "kind": "technical_numeric_diagnostic_not_semantic_outcome",
        "model": MODEL,
        "text": TEXT,
        "delta": DELTA,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "results": [run_dtype(dtype, tokenizer) for dtype in (
            torch.float32, torch.float16, torch.bfloat16)],
    }
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("results/coherent_state_diagnostics") / (
            f"numeric_Qwen3-0.6B_{stamp}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
