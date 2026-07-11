"""Measure BF16 full-prefill versus split-prefill schedule sensitivity.

This is a technical diagnostic, not a semantic experiment.  It evaluates the
exact cached Qwen3-0.6B checkpoint under both SDPA and eager attention.  Every
comparison is run with the automatically constructed causal mask and with an
independently constructed four-dimensional causal mask.  A failure is written
to the output artifact before the process exits nonzero.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import platform
from pathlib import Path
import sys
import traceback
from typing import Any

import huggingface_hub
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"
FIXTURE_TEXT = "alpha beta gamma delta epsilon"
CUT = 2
BACKENDS = ("sdpa", "eager")
TARGET_TEXTS = (" A", " B")


@dataclass
class Execution:
    logits: torch.Tensor
    rows: list[tuple[torch.Tensor, torch.Tensor]]
    live_cache: Any | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results/coherent_state_diagnostics") / (
        f"bf16_attention_schedule_Qwen3-0.6B_{stamp}.json")


def persist(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def causal_4d_mask(*, q_len: int, kv_len: int, past_len: int,
                   dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Construct a physical causal mask without Transformers mask helpers."""
    if q_len <= 0 or kv_len != past_len + q_len or past_len < 0:
        raise ValueError(
            f"invalid causal-mask geometry q={q_len} kv={kv_len} past={past_len}")
    mask = torch.full(
        (1, 1, q_len, kv_len), torch.finfo(dtype).min,
        dtype=dtype, device=device)
    for query_index in range(q_len):
        mask[..., query_index, :past_len + query_index + 1] = 0
    return mask


def snapshot_cache(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if not hasattr(cache, "layers"):
        raise RuntimeError("Transformers DynamicCache no longer exposes layers")
    rows = []
    for layer_index, layer in enumerate(cache.layers):
        if layer.keys is None or layer.values is None:
            raise RuntimeError(f"cache layer {layer_index} was not populated")
        rows.append((layer.keys.detach().clone(), layer.values.detach().clone()))
    if not rows:
        raise RuntimeError("cache snapshot was empty")
    return rows


def selected_margin(logits: torch.Tensor, target_ids: tuple[int, int]) -> float:
    a, b = target_ids
    return float((logits[..., a].float() - logits[..., b].float()).item())


def cache_difference(
        reference: list[tuple[torch.Tensor, torch.Tensor]],
        candidate: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise RuntimeError("cache layer counts differ")
    per_layer = []
    for layer_index, ((ref_k, ref_v), (got_k, got_v)) in enumerate(
            zip(reference, candidate, strict=True)):
        if ref_k.shape != got_k.shape or ref_v.shape != got_v.shape:
            raise RuntimeError(
                f"cache layer {layer_index} shapes differ: "
                f"K {tuple(ref_k.shape)} vs {tuple(got_k.shape)}, "
                f"V {tuple(ref_v.shape)} vs {tuple(got_v.shape)}")
        per_layer.append({
            "layer": layer_index,
            "k_max_abs": float((ref_k.float() - got_k.float()).abs().max()),
            "v_max_abs": float((ref_v.float() - got_v.float()).abs().max()),
        })
    return {
        "k_max_abs": max(row["k_max_abs"] for row in per_layer),
        "v_max_abs": max(row["v_max_abs"] for row in per_layer),
        "per_layer": per_layer,
    }


def execution_difference(reference: Execution, candidate: Execution,
                         target_ids: tuple[int, int]) -> dict[str, Any]:
    if reference.logits.shape != candidate.logits.shape:
        raise RuntimeError(
            f"logit shapes differ: {tuple(reference.logits.shape)} vs "
            f"{tuple(candidate.logits.shape)}")
    ref = reference.logits.float()
    got = candidate.logits.float()
    ref_last = ref[0, -1]
    got_last = got[0, -1]
    ref_lp = torch.log_softmax(ref_last, dim=-1)
    got_lp = torch.log_softmax(got_last, dim=-1)
    ref_p, got_p = ref_lp.exp(), got_lp.exp()
    ref_margin = selected_margin(ref_last, target_ids)
    got_margin = selected_margin(got_last, target_ids)
    cache = cache_difference(reference.rows, candidate.rows)
    return {
        "logits_max_abs": float((ref - got).abs().max()),
        "last_logits_max_abs": float((ref_last - got_last).abs().max()),
        "selected_margin_reference": ref_margin,
        "selected_margin_candidate": got_margin,
        "selected_margin_abs_shift": abs(ref_margin - got_margin),
        "kl_reference_to_candidate": float(
            (ref_p * (ref_lp - got_lp)).sum()),
        "total_variation": float(0.5 * (ref_p - got_p).abs().sum()),
        "top1_reference": int(ref_last.argmax()),
        "top1_candidate": int(got_last.argmax()),
        "top1_equal": bool(ref_last.argmax() == got_last.argmax()),
        **cache,
    }


def schedule_difference(full: Execution, split: Execution,
                        target_ids: tuple[int, int], cut: int) -> dict[str, Any]:
    if full.logits.shape[1] <= cut:
        raise RuntimeError("full execution does not contain split suffix")
    suffix = Execution(logits=full.logits[:, cut:], rows=full.rows)
    result = execution_difference(suffix, split, target_ids)
    result["full_query_length"] = int(full.logits.shape[1])
    result["split_query_lengths"] = [cut, int(split.logits.shape[1])]
    return result


def forward(model, input_ids: torch.Tensor, position_ids: torch.Tensor,
            cache_position: torch.Tensor, *, past=None,
            attention_mask: torch.Tensor | None = None) -> Execution:
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            past_key_values=past,
            position_ids=position_ids,
            cache_position=cache_position,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=0,
        )
    return Execution(
        logits=output.logits.detach().clone(),
        rows=snapshot_cache(output.past_key_values),
        live_cache=output.past_key_values,
    )


def execute_schedule(model, token_ids: list[int], *, split: bool,
                     explicit_mask: bool) -> Execution:
    device = model.device
    dtype = next(model.parameters()).dtype
    ids = torch.tensor([token_ids], device=device)
    positions = torch.arange(len(token_ids), device=device)[None]
    cache_positions = torch.arange(len(token_ids), device=device)
    if not split:
        mask = None
        if explicit_mask:
            mask = causal_4d_mask(
                q_len=len(token_ids), kv_len=len(token_ids), past_len=0,
                dtype=dtype, device=device)
        return forward(
            model, ids, positions, cache_positions, attention_mask=mask)

    first_mask = None
    if explicit_mask:
        first_mask = causal_4d_mask(
            q_len=CUT, kv_len=CUT, past_len=0,
            dtype=dtype, device=device)
    first = forward(
        model, ids[:, :CUT], positions[:, :CUT], cache_positions[:CUT],
        attention_mask=first_mask)

    second_len = len(token_ids) - CUT
    second_mask = None
    if explicit_mask:
        # Construct this independently from the full and first-chunk masks.
        second_mask = causal_4d_mask(
            q_len=second_len, kv_len=len(token_ids), past_len=CUT,
            dtype=dtype, device=device)
    return forward(
        model, ids[:, CUT:], positions[:, CUT:], cache_positions[CUT:],
        past=first.live_cache, attention_mask=second_mask)


def config_record(model, requested_backend: str) -> dict[str, Any]:
    config = model.config.to_dict()
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    text_config = getattr(model.config, "text_config", model.config)
    return {
        "requested_attention_backend": requested_backend,
        "resolved_attention_backend": getattr(
            model.config, "_attn_implementation", None),
        "resolved_text_attention_backend": getattr(
            text_config, "_attn_implementation", None),
        "commit_hash": getattr(model.config, "_commit_hash", None),
        "name_or_path": getattr(model.config, "_name_or_path", None),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "device": str(model.device),
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "config": config,
    }


def run_backend(model_id: str, backend: str, tokenizer, token_ids: list[int],
                target_ids: tuple[int, int], device: str) -> dict[str, Any]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=backend,
    ).to(device)
    model.eval()
    record: dict[str, Any] = {
        "backend": backend,
        "started_at": utc_now(),
        "model": config_record(model, backend),
        "mask_modes": {},
    }
    try:
        mode_executions: dict[str, tuple[Execution, Execution]] = {}
        for mode_name, explicit_mask in (
                ("automatic", False), ("independent_explicit_4d", True)):
            full_1 = execute_schedule(
                model, token_ids, split=False, explicit_mask=explicit_mask)
            split_1 = execute_schedule(
                model, token_ids, split=True, explicit_mask=explicit_mask)
            full_2 = execute_schedule(
                model, token_ids, split=False, explicit_mask=explicit_mask)
            split_2 = execute_schedule(
                model, token_ids, split=True, explicit_mask=explicit_mask)
            mode_executions[mode_name] = (full_1, split_1)
            record["mask_modes"][mode_name] = {
                "mask_shapes": (
                    None if not explicit_mask else {
                        "full": [1, 1, len(token_ids), len(token_ids)],
                        "split_first": [1, 1, CUT, CUT],
                        "split_second": [
                            1, 1, len(token_ids) - CUT, len(token_ids)],
                    }),
                "full_vs_split": schedule_difference(
                    full_1, split_1, target_ids, CUT),
                "repeatability": {
                    "full_first_vs_identical_repeat": execution_difference(
                        full_1, full_2, target_ids),
                    "split_first_vs_identical_repeat": execution_difference(
                        split_1, split_2, target_ids),
                },
            }

        automatic_full, automatic_split = mode_executions["automatic"]
        explicit_full, explicit_split = mode_executions[
            "independent_explicit_4d"]
        record["automatic_vs_independent_explicit_4d"] = {
            "full": execution_difference(
                automatic_full, explicit_full, target_ids),
            "split": execution_difference(
                automatic_split, explicit_split, target_ids),
        }
        record["completed_at"] = utc_now()
        record["status"] = "complete"
        return record
    finally:
        del model
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or default_output_path()
    print(
        f"RUN bf16-attention-schedule model={args.model} "
        f"backends={','.join(BACKENDS)} -> {output}", flush=True)

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True)
    token_ids = tokenizer(
        FIXTURE_TEXT, add_special_tokens=False).input_ids
    targets = tuple(
        tokenizer(text, add_special_tokens=False).input_ids
        for text in TARGET_TEXTS)
    if len(token_ids) != 5 or any(len(ids) != 1 for ids in targets):
        raise RuntimeError(
            f"diagnostic tokenization changed: fixture={token_ids}, targets={targets}")
    target_ids = (targets[0][0], targets[1][0])
    payload: dict[str, Any] = {
        "schema": 1,
        "kind": "technical_bf16_attention_schedule_diagnostic_not_semantic_outcome",
        "status": "running",
        "started_at": utc_now(),
        "model_id": args.model,
        "fixture": {
            "text": FIXTURE_TEXT,
            "token_ids": token_ids,
            "cut": CUT,
            "query_schedules": [[len(token_ids)], [CUT, len(token_ids) - CUT]],
            "selected_token_texts": list(TARGET_TEXTS),
            "selected_token_ids": list(target_ids),
            "split_cache_path": "live_cache_returned_by_first_chunk",
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
            "seed": 0,
        },
        "backends": [],
        "failures": [],
    }
    persist(output, payload)
    for backend in BACKENDS:
        try:
            result = run_backend(
                args.model, backend, tokenizer, token_ids, target_ids,
                args.device)
            payload["backends"].append(result)
        except Exception as exc:  # preserve complete failure provenance
            payload["failures"].append({
                "backend": backend,
                "failed_at": utc_now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            })
        persist(output, payload)

    payload["completed_at"] = utc_now()
    payload["status"] = "failed" if payload["failures"] else "complete"
    persist(output, payload)
    print(output, flush=True)
    if payload["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
