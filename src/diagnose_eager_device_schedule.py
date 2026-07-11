"""Reproduce Amendment-4's L=64 eager schedule fixture across local devices.

This is an additive, non-authorizing device diagnostic.  It runs the exact
cached Qwen3-0.6B checkpoint in BF16 with eager attention, comparing the frozen
``[64]`` reference schedule against ``[32, 32]``.  It also appends the frozen
common q=1 continuation to both caches.  Results are written incrementally and
a threshold failure is fully serialized before the program exits nonzero.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

import huggingface_hub
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"
EXPECTED_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ATTENTION_BACKEND = "eager"
DTYPE = torch.bfloat16
FIXTURE_LITERAL = "alpha beta gamma delta epsilon"
EXPECTED_POOL = [7141, 13440, 21619, 9477, 31204]
LENGTH = 64
REFERENCE_PARTITION = [64]
ALTERNATIVE_PARTITION = [32, 32]
MARGIN_TOKEN_TEXT = [" A", " B"]
EXPECTED_MARGIN_TOKEN_IDS = [362, 425]
CONTINUATION_TOKEN_ID = 7141
TOLERANCE = 5e-4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results/coherent_state_diagnostics") / (
        f"eager_device_schedule_Qwen3-0.6B_{stamp}.json")


def atomic_persist(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_ids(ids: list[int]) -> str:
    h = hashlib.sha256()
    for token_id in ids:
        h.update(int(token_id).to_bytes(8, "little", signed=True))
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cache_rows(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if not hasattr(cache, "layers"):
        raise RuntimeError(
            f"unsupported cache without layers: {type(cache).__name__}")
    rows = []
    for layer_index, layer in enumerate(cache.layers):
        if layer.keys is None or layer.values is None:
            raise RuntimeError(f"cache layer {layer_index} is empty")
        rows.append((
            layer.keys.detach().clone(), layer.values.detach().clone()))
    if not rows:
        raise RuntimeError("cache has no populated layers")
    return rows


def cache_per_layer_difference(
        reference: list[tuple[torch.Tensor, torch.Tensor]],
        alternative: list[tuple[torch.Tensor, torch.Tensor]], *,
        only_last_row: bool = False) -> dict[str, Any]:
    if len(reference) != len(alternative):
        raise RuntimeError(
            f"cache layer mismatch: {len(reference)} != {len(alternative)}")
    per_layer: list[dict[str, Any]] = []
    for layer_index, ((ref_k, ref_v), (alt_k, alt_v)) in enumerate(
            zip(reference, alternative, strict=True)):
        if ref_k.shape != alt_k.shape or ref_v.shape != alt_v.shape:
            raise RuntimeError(
                f"cache shape mismatch at layer {layer_index}: "
                f"K {tuple(ref_k.shape)} != {tuple(alt_k.shape)}, "
                f"V {tuple(ref_v.shape)} != {tuple(alt_v.shape)}")
        if only_last_row:
            ref_k, alt_k = ref_k[..., -1:, :], alt_k[..., -1:, :]
            ref_v, alt_v = ref_v[..., -1:, :], alt_v[..., -1:, :]
        per_layer.append({
            "layer": layer_index,
            "k_max_abs": float(
                (ref_k.float() - alt_k.float()).abs().max().item()),
            "v_max_abs": float(
                (ref_v.float() - alt_v.float()).abs().max().item()),
        })
    return {
        "scope": "newly_appended_row" if only_last_row else "all_aligned_rows",
        "k_global_max_abs": max(row["k_max_abs"] for row in per_layer),
        "v_global_max_abs": max(row["v_max_abs"] for row in per_layer),
        "per_layer": per_layer,
    }


def logits_difference(
        reference: torch.Tensor, alternative: torch.Tensor,
        margin_ids: list[int]) -> dict[str, Any]:
    if reference.shape != alternative.shape:
        raise RuntimeError(
            f"logit shape mismatch: {tuple(reference.shape)} != "
            f"{tuple(alternative.shape)}")
    ref = reference.float()[0, -1]
    alt = alternative.float()[0, -1]
    ref_margin = float((ref[margin_ids[0]] - ref[margin_ids[1]]).item())
    alt_margin = float((alt[margin_ids[0]] - alt[margin_ids[1]]).item())
    return {
        "last_logits_max_abs": float((ref - alt).abs().max().item()),
        "selected_margin_reference": ref_margin,
        "selected_margin_alternative": alt_margin,
        "selected_margin_abs_shift": abs(ref_margin - alt_margin),
        "top1_reference": int(ref.argmax().item()),
        "top1_alternative": int(alt.argmax().item()),
        "top1_equal": bool(ref.argmax().item() == alt.argmax().item()),
    }


def backend_fingerprint(model) -> dict[str, Any]:
    cfg = getattr(model.config, "text_config", model.config)
    model_fields = {
        "model_config__attn_implementation": getattr(
            model.config, "_attn_implementation", None),
        "text_config__attn_implementation": getattr(
            cfg, "_attn_implementation", None),
        "model_config_attn_implementation": getattr(
            model.config, "attn_implementation", None),
        "text_config_attn_implementation": getattr(
            cfg, "attn_implementation", None),
    }
    model_fields = {
        key: None if value is None else str(value)
        for key, value in model_fields.items()
    }
    resolved = {value for value in model_fields.values() if value is not None}
    if resolved != {ATTENTION_BACKEND}:
        raise RuntimeError(f"model backend is not unambiguously eager: {resolved}")
    layers = []
    for name, module in model.named_modules():
        class_name = type(module).__name__
        if not (hasattr(module, "q_proj") and hasattr(module, "k_proj") and
                "Attention" in class_name):
            continue
        fields = {
            field: (None if getattr(module, field, None) is None else
                    str(getattr(module, field)))
            for field in ("_attn_implementation", "attn_implementation")
        }
        local = {value for value in fields.values() if value is not None}
        if local and local != {ATTENTION_BACKEND}:
            raise RuntimeError(
                f"layer {name} backend is not eager: {sorted(local)}")
        layers.append({
            "layer_index": int(getattr(module, "layer_idx", len(layers))),
            "module_name": name,
            "module_class": class_name,
            "module_fields": fields,
            "resolved_implementation": ATTENTION_BACKEND,
        })
    layers.sort(key=lambda row: row["layer_index"])
    expected_layers = int(getattr(cfg, "num_hidden_layers", -1))
    if [row["layer_index"] for row in layers] != list(range(expected_layers)):
        raise RuntimeError(
            f"attention enumeration incomplete: {len(layers)} != {expected_layers}")
    record = {
        "requested_implementation": ATTENTION_BACKEND,
        "model_fields": model_fields,
        "expected_layer_count": expected_layers,
        "layers": layers,
    }
    record["sha256"] = sha256_json(record)
    return record


def timed_forward(model, *, input_ids: list[int], start: int,
                  past_key_values=None) -> tuple[Any, float]:
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    positions = torch.arange(
        start, start + len(input_ids), device=device).unsqueeze(0)
    cache_positions = torch.arange(
        start, start + len(input_ids), device=device)
    if device.type == "mps":
        torch.mps.synchronize()
    before = time.perf_counter()
    with torch.inference_mode():
        output = model(
            input_ids=ids,
            position_ids=positions,
            cache_position=cache_positions,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - before
    return output, elapsed


def execute_reference(model, token_ids: list[int]) -> tuple[Any, dict[str, float]]:
    output, elapsed = timed_forward(
        model, input_ids=token_ids, start=0)
    return output, {"prefill_seconds": elapsed}


def execute_alternative(
        model, token_ids: list[int]) -> tuple[Any, dict[str, float]]:
    first, first_elapsed = timed_forward(
        model, input_ids=token_ids[:32], start=0)
    second, second_elapsed = timed_forward(
        model, input_ids=token_ids[32:], start=32,
        past_key_values=first.past_key_values)
    return second, {
        "first_chunk_seconds": first_elapsed,
        "second_chunk_seconds": second_elapsed,
        "prefill_total_seconds": first_elapsed + second_elapsed,
    }


def run_device(
        model_id: str, revision: str, device_name: str, token_ids: list[int],
        margin_ids: list[int], persist_progress: Callable[[str, Any], None]
        ) -> dict[str, Any]:
    started = time.perf_counter()
    record: dict[str, Any] = {
        "device_requested": device_name,
        "status": "RUNNING",
        "started_at": utc_now(),
    }
    persist_progress(device_name, record)
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
        dtype=DTYPE,
        attn_implementation=ATTENTION_BACKEND,
    ).to(device_name)
    model.eval()
    load_seconds = time.perf_counter() - load_started
    try:
        device = next(model.parameters()).device
        commit_hash = getattr(model.config, "_commit_hash", None)
        if commit_hash != revision:
            raise RuntimeError(
                f"model revision mismatch: {commit_hash!r} != {revision!r}")
        if next(model.parameters()).dtype != DTYPE:
            raise RuntimeError(
                f"parameter dtype mismatch: {next(model.parameters()).dtype}")
        backend = backend_fingerprint(model)
        record.update({
            "device_resolved": str(device),
            "parameter_dtype": str(next(model.parameters()).dtype),
            "model_revision": commit_hash,
            "attention_backend": backend,
            "model_load_seconds": load_seconds,
        })
        persist_progress(device_name, record)

        reference, ref_timing = execute_reference(model, token_ids)
        alternative, alt_timing = execute_alternative(model, token_ids)
        initial_reference_rows = cache_rows(reference.past_key_values)
        initial_alternative_rows = cache_rows(alternative.past_key_values)
        initial = {
            "cache": cache_per_layer_difference(
                initial_reference_rows, initial_alternative_rows),
            "final_logits": logits_difference(
                reference.logits, alternative.logits, margin_ids),
        }
        record["prefill_comparison"] = initial
        record["timing"] = {
            "reference": ref_timing,
            "alternative": alt_timing,
        }
        persist_progress(device_name, record)

        continued_reference, ref_cont_seconds = timed_forward(
            model, input_ids=[CONTINUATION_TOKEN_ID], start=LENGTH,
            past_key_values=reference.past_key_values)
        continued_alternative, alt_cont_seconds = timed_forward(
            model, input_ids=[CONTINUATION_TOKEN_ID], start=LENGTH,
            past_key_values=alternative.past_key_values)
        continued_ref_rows = cache_rows(continued_reference.past_key_values)
        continued_alt_rows = cache_rows(continued_alternative.past_key_values)
        continuation = {
            "token_id": CONTINUATION_TOKEN_ID,
            "logical_position": LENGTH,
            "physical_cache_position": LENGTH,
            "cache_new_row": cache_per_layer_difference(
                continued_ref_rows, continued_alt_rows, only_last_row=True),
            "logits": logits_difference(
                continued_reference.logits, continued_alternative.logits,
                margin_ids),
        }
        record["common_q1_continuation"] = continuation
        record["timing"]["common_q1_continuation_seconds"] = {
            "reference": ref_cont_seconds,
            "alternative": alt_cont_seconds,
        }

        checks: dict[str, float] = {
            "prefill_k_global_max_abs": initial["cache"]["k_global_max_abs"],
            "prefill_v_global_max_abs": initial["cache"]["v_global_max_abs"],
            "prefill_last_logits_max_abs": initial["final_logits"][
                "last_logits_max_abs"],
            "prefill_selected_margin_abs_shift": initial["final_logits"][
                "selected_margin_abs_shift"],
            "continuation_k_global_max_abs": continuation["cache_new_row"][
                "k_global_max_abs"],
            "continuation_v_global_max_abs": continuation["cache_new_row"][
                "v_global_max_abs"],
            "continuation_last_logits_max_abs": continuation["logits"][
                "last_logits_max_abs"],
        }
        record["threshold_evaluation"] = {
            "tolerance": TOLERANCE,
            "checks": {
                key: {"observed": value, "pass": value <= TOLERANCE}
                for key, value in checks.items()
            },
            "complete_pass": all(value <= TOLERANCE for value in checks.values()),
        }
        record["elapsed_seconds"] = time.perf_counter() - started
        record["completed_at"] = utc_now()
        record["status"] = (
            "PASS" if record["threshold_evaluation"]["complete_pass"]
            else "FAIL")
        persist_progress(device_name, record)
        return record
    finally:
        del model
        gc.collect()
        if device_name == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=EXPECTED_REVISION)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--devices", nargs="+", default=["cpu", "mps"],
        choices=["cpu", "mps"],
        help="MPS is recorded as unavailable rather than treated as a failure.")
    args = parser.parse_args()
    output = args.output or default_output_path()
    print(
        f"RUN eager-device-schedule model={args.model}@{args.revision} "
        f"devices={','.join(args.devices)} -> {output}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=True)
    pool = tokenizer(
        FIXTURE_LITERAL, add_special_tokens=False).input_ids
    margin_ids = [
        tokenizer(text, add_special_tokens=False).input_ids
        for text in MARGIN_TOKEN_TEXT
    ]
    if pool != EXPECTED_POOL:
        raise RuntimeError(f"fixture pool mismatch: {pool} != {EXPECTED_POOL}")
    if margin_ids != [[token_id] for token_id in EXPECTED_MARGIN_TOKEN_IDS]:
        raise RuntimeError(
            f"selected-token mismatch: {margin_ids} != "
            f"{[[x] for x in EXPECTED_MARGIN_TOKEN_IDS]}")
    token_ids = [pool[index % len(pool)] for index in range(LENGTH)]
    special_ids = set(tokenizer.all_special_ids)
    if special_ids.intersection(pool):
        raise RuntimeError("fixture pool includes a special token")

    payload: dict[str, Any] = {
        "schema": 1,
        "kind": "non_authorizing_amendment4_eager_device_schedule_diagnostic",
        "status": "RUNNING",
        "started_at": utc_now(),
        "model": {
            "id": args.model,
            "requested_revision": args.revision,
            "dtype": str(DTYPE),
            "attention_backend": ATTENTION_BACKEND,
        },
        "fixture": {
            "amendment": "COHERENT-STATE-PREREGISTRATION-AMENDMENT-4",
            "authorizes_production_or_semantics": False,
            "literal": FIXTURE_LITERAL,
            "pool_token_ids": pool,
            "pool_decoded": [tokenizer.decode([token_id]) for token_id in pool],
            "pool_sha256": sha256_ids(pool),
            "special_token_ids_excluded": True,
            "length": LENGTH,
            "constructed_token_ids": token_ids,
            "constructed_token_ids_sha256": sha256_ids(token_ids),
            "reference_query_partition": REFERENCE_PARTITION,
            "alternative_query_partition": ALTERNATIVE_PARTITION,
            "logical_positions": [0, LENGTH - 1],
            "physical_cache_positions": [0, LENGTH - 1],
            "selected_margin": {
                "texts": MARGIN_TOKEN_TEXT,
                "token_ids": EXPECTED_MARGIN_TOKEN_IDS,
            },
            "common_continuation": {
                "token_id": CONTINUATION_TOKEN_ID,
                "decoded": tokenizer.decode([CONTINUATION_TOKEN_ID]),
            },
            "unchanged_tolerance": TOLERANCE,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "git_commit_at_start": git_value("rev-parse", "HEAD"),
            "git_status_at_start": git_value(
                "status", "--porcelain", "--untracked-files=all"),
        },
        "requested_devices": args.devices,
        "devices": [],
        "failures": [],
    }
    atomic_persist(output, payload)

    device_records: dict[str, dict[str, Any]] = {}

    def persist_progress(device_name: str, record: dict[str, Any]) -> None:
        device_records[device_name] = record.copy()
        payload["devices"] = [
            device_records[name] for name in args.devices
            if name in device_records
        ]
        atomic_persist(output, payload)

    for device_name in args.devices:
        if device_name == "mps" and not torch.backends.mps.is_available():
            record = {
                "device_requested": "mps",
                "status": "UNAVAILABLE",
                "recorded_at": utc_now(),
                "reason": "torch.backends.mps.is_available() is false",
            }
            persist_progress(device_name, record)
            continue
        try:
            run_device(
                args.model, args.revision, device_name, token_ids,
                EXPECTED_MARGIN_TOKEN_IDS, persist_progress)
        except Exception as exc:
            failure = {
                "device": device_name,
                "failed_at": utc_now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
            payload["failures"].append(failure)
            prior = device_records.get(device_name, {})
            prior.update({"status": "ERROR", "error": failure})
            persist_progress(device_name, prior)

    completed = [row for row in payload["devices"] if row["status"] in {"PASS", "FAIL"}]
    payload["complete_pass"] = bool(completed) and all(
        row["status"] == "PASS" for row in completed)
    payload["status"] = (
        "ERROR" if payload["failures"] else
        "PASS" if payload["complete_pass"] else "FAIL")
    payload["completed_at"] = utc_now()
    atomic_persist(output, payload)
    print(output, flush=True)
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
