#!/usr/bin/env python3
"""Narrow J-lens readouts around a compaction boundary.

This is an isolated prototype. It deliberately avoids importing or modifying
the live serving shim. The grafted-post-token state is enabled only for models
whose Hugging Face cache exposes standard per-layer keys/values.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


DEFAULT_LENS_REPO = "neuronpedia/jacobian-lens"
DEFAULT_LENS_REVISION = "qwen-n1000"
DEFAULT_QWEN36_LENS = (
    "qwen3.6-27b/jlens/Salesforce-wikitext/"
    "Qwen3.6-27B_jacobian_lens_n1000.pt"
)


SUMMARY_REQUEST = (
    "Context is about to be condensed. Write a context summary for an AI "
    "coding agent that will continue this work seeing only the summary plus "
    "recent messages. Preserve the task, concrete state, decisions, errors, "
    "and next steps. Be specific with names and numbers. No commentary before "
    "or after."
)


DEFAULT_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a careful coding assistant. Maintain precise task state "
            "and avoid inventing details."
        ),
    },
    {
        "role": "user",
        "content": (
            "We are fixing a small package called rivermark. Important: in "
            "this task the word bank always means a river bank, never a "
            "financial institution. The failing test is about driftwood being "
            "sorted by distance from the waterline."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Understood. I will preserve that bank means river bank here, and "
            "that the relevant object is driftwood near the waterline."
        ),
    },
    {
        "role": "user",
        "content": (
            "I checked src/rivermark/sort.py. The bug is that wet driftwood is "
            "treated as lower priority even though the spec says it should be "
            "inspected first. The next step is to update the priority key and "
            "run tests/test_sort.py."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "State noted: update src/rivermark/sort.py so wet driftwood on "
            "the river bank is inspected first, then run tests/test_sort.py."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-filename", default=DEFAULT_QWEN36_LENS)
    p.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
    p.add_argument("--messages-json")
    p.add_argument("--output", default="outputs/qwen36_boundary_probe.json")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--layers", default="quarter")
    p.add_argument("--max-new-summary-tokens", type=int, default=96)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument(
        "--probe-user",
        default=(
            "Continue the task. What exact file should be changed next, and "
            "what test should be run? Answer in one sentence."
        ),
        help=(
            "User message appended after the compaction boundary for the "
            "three-state full/fresh/grafted post-boundary probe."
        ),
    )
    p.add_argument(
        "--probe-target",
        default=(
            "Update src/rivermark/sort.py so wet driftwood is inspected first, "
            "then run tests/test_sort.py."
        ),
        help=(
            "Contentful continuation teacher-forced under full, fresh, and "
            "grafted caches for downstream intervention readouts."
        ),
    )
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return p.parse_args()


def load_messages(path: str | None) -> list[dict[str, str]]:
    if not path:
        return [dict(m) for m in DEFAULT_MESSAGES]
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data["messages"]
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in data]


def dtype_from_name(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def model_input_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    return next(model.parameters()).device


def render_text(tokenizer: Any, messages: list[dict[str, str]], gen_prompt: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": gen_prompt}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_ids(tokenizer: Any, messages: list[dict[str, str]], gen_prompt: bool) -> list[int]:
    text = render_text(tokenizer, messages, gen_prompt)
    return tokenizer(text, add_special_tokens=False).input_ids


def build_b_messages(
    messages: list[dict[str, str]], summary_text: str, tail_start_msg: int
) -> list[dict[str, str]]:
    note = (
        "[Context note] Earlier parts of this conversation were compacted. "
        "Summary of what came before:\n\n" + summary_text
    )
    return [
        messages[0],
        {"role": "assistant", "content": note},
        *messages[tail_start_msg:],
    ]


def token_tensor(model: torch.nn.Module, ids: list[int]) -> torch.Tensor:
    return torch.tensor([ids], device=model_input_device(model))


def generate_summary(
    model: torch.nn.Module,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> dict[str, Any]:
    conv_ids = render_ids(tokenizer, messages, False)
    req_messages = messages + [{"role": "user", "content": SUMMARY_REQUEST}]
    req_ids = render_ids(tokenizer, req_messages, True)
    input_ids = token_tensor(model, req_ids)
    eos = model.config.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad,
            eos_token_id=eos,
        )
    all_ids = out[0].tolist()
    gen_ids = all_ids[len(req_ids) :]
    if eos is not None:
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        gen_ids = [t for t in gen_ids if t not in eos_ids]
    return {
        "text": tokenizer.decode(gen_ids, skip_special_tokens=True).strip(),
        "conv_ids": conv_ids,
        "req_ids": req_ids,
        "gen_ids": gen_ids,
        "old_ids": req_ids + gen_ids,
        "conv_end": len(conv_ids),
        "s_start": len(req_ids),
        "s_end": len(req_ids) + len(gen_ids),
    }


def choose_layers(spec: str, n_layers: int, fitted_layers: list[int]) -> list[int]:
    if spec == "all":
        candidates = fitted_layers
    elif spec == "quarter":
        candidates = [
            n_layers // 4,
            n_layers // 2,
            (3 * n_layers) // 4,
            max(0, n_layers - 2),
        ]
    else:
        candidates = [int(x) for x in spec.split(",") if x.strip()]
    fitted = set(fitted_layers)
    out = []
    for layer in candidates:
        if layer in fitted and layer not in out:
            out.append(layer)
    if not out:
        raise ValueError(f"no requested layers are fitted; requested={candidates}")
    return out


def sample_summary_positions(start: int, end: int) -> list[int]:
    if end <= start:
        return []
    raw = [start, start + (end - start) // 2, end - 1]
    out = []
    for pos in raw:
        if start <= pos < end and pos not in out:
            out.append(pos)
    return out


def topk_from_logits(tokenizer: Any, logits: torch.Tensor, k: int) -> list[dict[str, Any]]:
    vals, idx = torch.topk(logits.float().cpu(), k)
    rows = []
    for score, token_id in zip(vals.tolist(), idx.tolist()):
        rows.append(
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "score": float(score),
            }
        )
    return rows


def capture_topk_for_positions(
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    input_ids: list[int],
    layers: list[int],
    positions: list[int],
    top_k: int,
) -> list[dict[str, Any]]:
    from jlens.hooks import ActivationRecorder

    device_ids = torch.tensor([input_ids], device=lens_model.input_device)
    final_layer = lens_model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    norm_positions = [p if p >= 0 else len(input_ids) + p for p in positions]
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=record_at) as rec:
        lens_model.forward(device_ids)
        activations = {i: rec.activations[i].detach() for i in record_at}

    results = []
    for pos in norm_positions:
        if pos < 0 or pos >= len(input_ids):
            continue
        token_id = int(input_ids[pos])
        row = {
            "position": int(pos),
            "token_id": token_id,
            "token": tokenizer.decode([token_id]),
            "layers": {},
        }
        for layer in layers:
            residual = activations[layer][0, pos].float().unsqueeze(0)
            transported = lens.transport(residual, layer)
            logits = lens_model.unembed(transported)[0]
            row["layers"][str(layer)] = topk_from_logits(tokenizer, logits, top_k)
        results.append(row)
    return results


def find_subsequence(haystack: list[int], needle: list[int], lo: int = 0) -> int | None:
    if not needle:
        return None
    n = len(needle)
    for i in range(max(0, lo), len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


def snapshot_standard_kv(model: torch.nn.Module, ids: list[int]) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], Any]:
    with torch.no_grad():
        out = model(input_ids=token_tensor(model, ids), use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return snapshot_list_cache(cache), out.logits[:, -1, :]
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError(
            "past_key_values has neither .layers nor .key_cache/.value_cache; "
            f"type={type(cache).__module__}.{type(cache).__name__}"
        )
    snap_layers = []
    for layer in layers:
        snap_layers.append(snapshot_cache_layer(layer))
    if not any(layer_has_kv(entry) for entry in snap_layers):
        raise TypeError(
            "cache has .layers but no populated K/V entries; "
            f"type={type(cache).__module__}.{type(cache).__name__}"
        )
    return {
        "kind": "layer_cache",
        "cache_class": cache.__class__,
        "layers": snap_layers,
    }, out.logits[:, -1, :]


def clone_tensor(x: torch.Tensor | None) -> torch.Tensor | None:
    return None if x is None else x.clone()


def clone_tensor_list(xs: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
    return [clone_tensor(x) for x in xs]


def snapshot_cache_layer(layer: Any) -> dict[str, Any]:
    return {
        "layer_class": layer.__class__,
        "layer_class_name": f"{type(layer).__module__}.{type(layer).__name__}",
        "keys": clone_tensor(getattr(layer, "keys", None)),
        "values": clone_tensor(getattr(layer, "values", None)),
        "conv_states": clone_tensor(getattr(layer, "conv_states", None)),
        "recurrent_states": clone_tensor(getattr(layer, "recurrent_states", None)),
        "has_previous_state": bool(getattr(layer, "has_previous_state", False)),
    }


def layer_has_kv(entry: dict[str, Any]) -> bool:
    k = entry.get("keys")
    v = entry.get("values")
    return k is not None and v is not None and k.numel() > 0 and v.numel() > 0


def restore_layer_state(layer: Any, entry: dict[str, Any]) -> None:
    k = clone_tensor(entry.get("keys"))
    v = clone_tensor(entry.get("values"))
    if k is not None and v is not None:
        layer.keys = k
        layer.values = v
        if hasattr(layer, "is_initialized"):
            layer.is_initialized = True
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length = int(k.shape[-2])

    conv = clone_tensor(entry.get("conv_states"))
    if conv is not None and hasattr(layer, "conv_states"):
        if hasattr(layer, "lazy_initialization") and not getattr(layer, "is_conv_states_initialized", False):
            layer.lazy_initialization(conv_states=conv)
        layer.conv_states.copy_(conv) if getattr(layer, "conv_states", None) is not None else setattr(layer, "conv_states", conv)
        if hasattr(layer, "is_conv_states_initialized"):
            layer.is_conv_states_initialized = True

    recurrent = clone_tensor(entry.get("recurrent_states"))
    if recurrent is not None and hasattr(layer, "recurrent_states"):
        if hasattr(layer, "lazy_initialization") and not getattr(layer, "is_recurrent_states_initialized", False):
            layer.lazy_initialization(recurrent_states=recurrent)
        layer.recurrent_states.copy_(recurrent) if getattr(layer, "recurrent_states", None) is not None else setattr(layer, "recurrent_states", recurrent)
        if hasattr(layer, "is_recurrent_states_initialized"):
            layer.is_recurrent_states_initialized = True

    if hasattr(layer, "has_previous_state"):
        layer.has_previous_state = bool(entry.get("has_previous_state", conv is not None or recurrent is not None))


def snapshot_list_cache(cache: Any) -> dict[str, Any]:
    """Snapshot Qwen3-Next-style caches with key_cache/value_cache lists.

    Qwen3.6 uses a hybrid architecture: full-attention layers have ordinary
    K/V tensors, while linear-attention layers carry recurrent state instead.
    For V-Graft we blend only populated attention-layer value_cache tensors and
    preserve the fresh recurrent/conv states unchanged.
    """

    return {
        "kind": "list_cache",
        "cache_class": cache.__class__,
        "key_cache": clone_tensor_list(cache.key_cache),
        "value_cache": clone_tensor_list(cache.value_cache),
        "conv_states": clone_tensor_list(getattr(cache, "conv_states", [])),
        "recurrent_states": clone_tensor_list(getattr(cache, "recurrent_states", [])),
        "layer_types": list(getattr(cache, "layer_types", [])),
    }


def rebuild_standard_cache(snap: dict[str, Any], model: torch.nn.Module | None = None) -> Any:
    if snap["kind"] == "layer_cache":
        cache = DynamicCache(config=model.config) if model is not None else DynamicCache()
        if len(cache.layers) < len(snap["layers"]):
            for i, entry in enumerate(snap["layers"]):
                k, v = entry.get("keys"), entry.get("values")
                if k is not None and v is not None:
                    cache.update(clone_tensor(k), clone_tensor(v), i)
        else:
            for layer, entry in zip(cache.layers, snap["layers"]):
                restore_layer_state(layer, entry)
        return cache

    if snap["kind"] == "list_cache":
        if model is None:
            raise TypeError("list-cache rebuild requires model.config")
        cache = snap["cache_class"](config=model.config)
        cache.key_cache = clone_tensor_list(snap["key_cache"])
        cache.value_cache = clone_tensor_list(snap["value_cache"])
        if hasattr(cache, "conv_states"):
            cache.conv_states = clone_tensor_list(snap["conv_states"])
        if hasattr(cache, "recurrent_states"):
            cache.recurrent_states = clone_tensor_list(snap["recurrent_states"])
        return cache

    raise TypeError(f"unknown cache snapshot kind: {snap.get('kind')}")


def cache_seq_len(snap: dict[str, Any]) -> int:
    if snap["kind"] == "layer_cache":
        for entry in snap["layers"]:
            k = entry.get("keys")
            if k is not None and k.numel() > 0:
                return int(k.shape[-2])
        return 0
    if snap["kind"] == "list_cache":
        for k in snap["key_cache"]:
            if k is not None and k.numel() > 0:
                return int(k.shape[-2])
        return 0
    raise TypeError(f"unknown cache snapshot kind: {snap.get('kind')}")


def cache_debug_summary(snap: dict[str, Any]) -> dict[str, Any]:
    if snap["kind"] == "layer_cache":
        shapes = [
            None if entry.get("keys") is None else [int(x) for x in entry["keys"].shape]
            for entry in snap["layers"][:8]
        ]
        populated = [i for i, entry in enumerate(snap["layers"]) if layer_has_kv(entry)]
        linear = [
            i
            for i, entry in enumerate(snap["layers"])
            if entry.get("conv_states") is not None or entry.get("recurrent_states") is not None
        ]
        return {
            "kind": snap["kind"],
            "layers": len(snap["layers"]),
            "populated_kv_layers": populated[:16],
            "populated_kv_layer_count": len(populated),
            "populated_linear_state_layers": linear[:16],
            "populated_linear_state_layer_count": len(linear),
            "seq_len": cache_seq_len(snap),
            "first_key_shapes": shapes,
            "layer_class_prefix": [entry.get("layer_class_name") for entry in snap["layers"][:16]],
        }
    if snap["kind"] == "list_cache":
        populated = [
            i
            for i, k in enumerate(snap["key_cache"])
            if k is not None and k.numel() > 0
        ]
        shapes = {
            str(i): [int(x) for x in snap["key_cache"][i].shape]
            for i in populated[:8]
        }
        return {
            "kind": snap["kind"],
            "layers": len(snap["key_cache"]),
            "populated_attention_layers": populated[:16],
            "populated_attention_layer_count": len(populated),
            "seq_len": cache_seq_len(snap),
            "first_key_shapes": shapes,
            "layer_types_prefix": snap.get("layer_types", [])[:16],
        }
    return {"kind": snap.get("kind")}


def blend_values(
    b_snap: dict[str, Any],
    old_snap: dict[str, Any],
    pairs: list[tuple[int, int]],
    alpha: float,
) -> dict[str, Any]:
    new_idx = torch.tensor([n for n, _ in pairs])
    old_idx = torch.tensor([o for _, o in pairs])

    if b_snap["kind"] != old_snap["kind"]:
        raise TypeError(f"cannot blend different cache kinds: {b_snap['kind']} vs {old_snap['kind']}")

    out = copy.copy(b_snap)

    if b_snap["kind"] == "layer_cache":
        layers = []
        for li, entry in enumerate(b_snap["layers"]):
            next_entry = copy.copy(entry)
            k = entry.get("keys")
            v = entry.get("values")
            if k is None or v is None or v.numel() == 0:
                next_entry["keys"] = clone_tensor(k)
                next_entry["values"] = clone_tensor(v)
                next_entry["conv_states"] = clone_tensor(entry.get("conv_states"))
                next_entry["recurrent_states"] = clone_tensor(entry.get("recurrent_states"))
                layers.append(next_entry)
                continue
            old_v = old_snap["layers"][li].get("values")
            if old_v is None or old_v.numel() == 0:
                next_entry["keys"] = clone_tensor(k)
                next_entry["values"] = clone_tensor(v)
                next_entry["conv_states"] = clone_tensor(entry.get("conv_states"))
                next_entry["recurrent_states"] = clone_tensor(entry.get("recurrent_states"))
                layers.append(next_entry)
                continue
            v2 = v.clone()
            in_bounds = (new_idx < v2.shape[-2]) & (old_idx < old_v.shape[-2])
            if bool(in_bounds.any()):
                ni = new_idx[in_bounds].to(v2.device)
                oi = old_idx[in_bounds].to(old_v.device)
                vf = v2.index_select(-2, ni).float()
                vo = old_v.index_select(-2, oi).to(v2.device).float()
                v2.index_copy_(-2, ni, ((1 - alpha) * vf + alpha * vo).to(v2.dtype))
            next_entry["keys"] = clone_tensor(k)
            next_entry["values"] = v2
            next_entry["conv_states"] = clone_tensor(entry.get("conv_states"))
            next_entry["recurrent_states"] = clone_tensor(entry.get("recurrent_states"))
            layers.append(next_entry)
        out["layers"] = layers
        return out

    if b_snap["kind"] == "list_cache":
        values = clone_tensor_list(b_snap["value_cache"])
        for li, v in enumerate(values):
            old_v = old_snap["value_cache"][li]
            if v is None or old_v is None or v.numel() == 0 or old_v.numel() == 0:
                continue
            v2 = v.clone()
            in_bounds = (new_idx < v2.shape[-2]) & (old_idx < old_v.shape[-2])
            if not bool(in_bounds.any()):
                values[li] = v2
                continue
            ni = new_idx[in_bounds].to(v2.device)
            oi = old_idx[in_bounds].to(old_v.device)
            vf = v2.index_select(-2, ni).float()
            vo = old_v.index_select(-2, oi).to(v2.device).float()
            v2.index_copy_(-2, ni, ((1 - alpha) * vf + alpha * vo).to(v2.dtype))
            values[li] = v2
        out["key_cache"] = clone_tensor_list(b_snap["key_cache"])
        out["value_cache"] = values
        out["conv_states"] = clone_tensor_list(b_snap.get("conv_states", []))
        out["recurrent_states"] = clone_tensor_list(b_snap.get("recurrent_states", []))
        return out

    raise TypeError(f"unknown cache snapshot kind: {b_snap.get('kind')}")


def alignment_pairs_by_exact_tokens(
    new_ids: list[int],
    old_ids: list[int],
    new_range: tuple[int, int],
    old_range: tuple[int, int],
) -> list[tuple[int, int]]:
    (nlo, nhi), (olo, ohi) = new_range, old_range
    sm = difflib.SequenceMatcher(a=new_ids[nlo:nhi], b=old_ids[olo:ohi], autojunk=False)
    pairs = []
    for block in sm.get_matching_blocks():
        if block.size <= 0:
            continue
        for i in range(block.size):
            pairs.append((nlo + block.a + i, olo + block.b + i))
    return pairs


def capture_forced_token_from_cache(
    model: torch.nn.Module,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    snap: dict[str, Any],
    suffix_ids: list[int],
    next_position: int,
    layers: list[int],
    top_k: int,
    forced_token_id: int | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    from jlens.hooks import ActivationRecorder

    cache = rebuild_standard_cache(snap, model)
    dev = model_input_device(model)
    if suffix_ids:
        pos = torch.arange(next_position, next_position + len(suffix_ids), device=dev)[None]
        with torch.no_grad():
            out = model(
                input_ids=token_tensor(model, suffix_ids),
                past_key_values=cache,
                position_ids=pos,
                use_cache=True,
                logits_to_keep=1,
            )
        logits = out.logits[:, -1, :]
        next_position += len(suffix_ids)
    else:
        with torch.no_grad():
            out = model(
                input_ids=torch.empty((1, 0), dtype=torch.long, device=dev),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        logits = out.logits[:, -1, :]
    argmax_token_id = int(torch.argmax(logits, dim=-1).item())
    token_id = argmax_token_id if forced_token_id is None else int(forced_token_id)

    final_layer = lens_model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=record_at) as rec:
        model(
            input_ids=torch.tensor([[token_id]], device=dev),
            past_key_values=cache,
            position_ids=torch.tensor([[next_position]], device=dev),
            use_cache=True,
            logits_to_keep=1,
        )
        activations = {i: rec.activations[i].detach() for i in record_at}

    row = {
        "label": label,
        "position": int(next_position),
        "token_id": token_id,
        "token": tokenizer.decode([token_id]),
        "forced_token": forced_token_id is not None,
        "argmax_token_id": argmax_token_id,
        "argmax_token": tokenizer.decode([argmax_token_id]),
        "next_token_top": topk_from_logits(tokenizer, logits[0], top_k),
        "layers": {},
    }
    for layer in layers:
        residual = activations[layer][0, 0].float().unsqueeze(0)
        logits_l = lens_model.unembed(lens.transport(residual, layer))[0]
        row["layers"][str(layer)] = topk_from_logits(tokenizer, logits_l, top_k)
    return row


def capture_forced_sequence_from_cache(
    model: torch.nn.Module,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    snap: dict[str, Any],
    suffix_ids: list[int],
    next_position: int,
    forced_ids: list[int],
    layers: list[int],
    top_k: int,
    label: str,
) -> dict[str, Any]:
    from jlens.hooks import ActivationRecorder

    cache = rebuild_standard_cache(snap, model)
    dev = model_input_device(model)
    if suffix_ids:
        pos = torch.arange(next_position, next_position + len(suffix_ids), device=dev)[None]
        with torch.no_grad():
            out = model(
                input_ids=token_tensor(model, suffix_ids),
                past_key_values=cache,
                position_ids=pos,
                use_cache=True,
                logits_to_keep=1,
            )
        logits = out.logits[:, -1, :]
        next_position += len(suffix_ids)
    else:
        raise TypeError("forced sequence capture requires a non-empty generation suffix")

    final_layer = lens_model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    rows = []
    for rel_pos, token_id in enumerate(forced_ids):
        argmax_token_id = int(torch.argmax(logits, dim=-1).item())
        with torch.no_grad(), ActivationRecorder(lens_model.layers, at=record_at) as rec:
            out = model(
                input_ids=torch.tensor([[int(token_id)]], device=dev),
                past_key_values=cache,
                position_ids=torch.tensor([[next_position]], device=dev),
                use_cache=True,
                logits_to_keep=1,
            )
            activations = {i: rec.activations[i].detach() for i in record_at}
        row = {
            "relative_position": int(rel_pos),
            "position": int(next_position),
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "argmax_token_id": argmax_token_id,
            "argmax_token": tokenizer.decode([argmax_token_id]),
            "forced_token_in_next_top": any(
                int(item["token_id"]) == int(token_id)
                for item in topk_from_logits(tokenizer, logits[0], top_k)
            ),
            "next_token_top": topk_from_logits(tokenizer, logits[0], top_k),
            "layers": {},
        }
        for layer in layers:
            residual = activations[layer][0, 0].float().unsqueeze(0)
            logits_l = lens_model.unembed(lens.transport(residual, layer))[0]
            row["layers"][str(layer)] = topk_from_logits(tokenizer, logits_l, top_k)
        rows.append(row)
        logits = out.logits[:, -1, :]
        next_position += 1
    return {
        "label": label,
        "forced_text": tokenizer.decode(forced_ids),
        "forced_token_count": len(forced_ids),
        "rows": rows,
    }


def run() -> dict[str, Any]:
    args = parse_args()
    messages = load_messages(args.messages_json)

    import jlens

    torch_dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    lens = jlens.JacobianLens.from_pretrained(
        args.lens_repo,
        filename=args.lens_filename,
        revision=args.lens_revision,
    )
    layers = choose_layers(args.layers, lens_model.n_layers, lens.source_layers)

    summary = generate_summary(model, tokenizer, messages, args.max_new_summary_tokens)
    tail_start_msg = max(1, len(messages) - 2)
    b_messages = build_b_messages(messages, summary["text"], tail_start_msg)
    b_ids = render_ids(tokenizer, b_messages, False)
    full_probe_messages = messages + [{"role": "user", "content": args.probe_user}]
    b_probe_messages = b_messages + [{"role": "user", "content": args.probe_user}]
    full_probe_ids = render_ids(tokenizer, full_probe_messages, False)
    b_probe_ids = render_ids(tokenizer, b_probe_messages, False)
    probe_target_ids = tokenizer(args.probe_target, add_special_tokens=False).input_ids

    summary_text_ids = tokenizer(summary["text"], add_special_tokens=False).input_ids
    b_summary_start = find_subsequence(b_ids, summary_text_ids)
    if b_summary_start is None:
        b_summary_positions = [-1]
        b_summary_range = None
    else:
        b_summary_range = (b_summary_start, b_summary_start + len(summary_text_ids))
        b_summary_positions = sample_summary_positions(*b_summary_range)
    b_probe_summary_start = find_subsequence(b_probe_ids, summary_text_ids)
    b_probe_summary_range = None
    if b_probe_summary_start is not None:
        b_probe_summary_range = (b_probe_summary_start, b_probe_summary_start + len(summary_text_ids))

    old_summary_positions = sample_summary_positions(summary["s_start"], summary["s_end"])
    result: dict[str, Any] = {
        "model": args.model,
        "lens": {
            "repo": args.lens_repo,
            "filename": args.lens_filename,
            "revision": args.lens_revision,
            "source_layers": lens.source_layers,
            "sampled_layers": layers,
        },
        "summary_text": summary["text"],
        "token_counts": {
            "conversation": len(summary["conv_ids"]),
            "summary_request_context": len(summary["req_ids"]),
            "summary_tokens": len(summary["gen_ids"]),
            "old_with_summary": len(summary["old_ids"]),
            "fresh_compacted": len(b_ids),
            "full_probe": len(full_probe_ids),
            "fresh_probe": len(b_probe_ids),
        },
        "probe_user": args.probe_user,
        "probe_target": {
            "text": args.probe_target,
            "token_count": len(probe_target_ids),
            "tokens": [tokenizer.decode([tid]) for tid in probe_target_ids],
        },
        "states": {},
        "graft": {"attempted": False, "available": False},
    }

    result["states"]["old_context_final_token"] = capture_topk_for_positions(
        lens_model,
        lens,
        tokenizer,
        summary["conv_ids"],
        layers,
        [len(summary["conv_ids"]) - 1],
        args.top_k,
    )
    result["states"]["write_time_summary"] = capture_topk_for_positions(
        lens_model,
        lens,
        tokenizer,
        summary["old_ids"],
        layers,
        old_summary_positions,
        args.top_k,
    )
    result["states"]["fresh_compacted_summary"] = capture_topk_for_positions(
        lens_model,
        lens,
        tokenizer,
        b_ids,
        layers,
        b_summary_positions,
        args.top_k,
    )

    try:
        result["graft"]["attempted"] = True
        old_snap, _ = snapshot_standard_kv(model, summary["old_ids"])
        full_probe_snap, _ = snapshot_standard_kv(model, full_probe_ids)
        b_probe_snap, _ = snapshot_standard_kv(model, b_probe_ids)
        if b_probe_summary_range is None:
            raise TypeError("could not locate summary text in fresh compacted probe IDs")
        pairs = alignment_pairs_by_exact_tokens(
            b_probe_ids,
            summary["old_ids"],
            b_probe_summary_range,
            (summary["s_start"], summary["s_end"]),
        )
        if not pairs:
            raise TypeError("no exact summary-token pairs for graft alignment")
        alpha0_probe_snap = blend_values(b_probe_snap, old_snap, pairs, 0.0)
        grafted_probe_snap = blend_values(b_probe_snap, old_snap, pairs, args.alpha)

        full_probe_gen_ids = render_ids(tokenizer, full_probe_messages, True)
        b_probe_gen_ids = render_ids(tokenizer, b_probe_messages, True)
        if full_probe_gen_ids[: len(full_probe_ids)] != full_probe_ids:
            raise TypeError("full probe generation prompt did not extend full probe prefix")
        if b_probe_gen_ids[: len(b_probe_ids)] != b_probe_ids:
            raise TypeError("fresh probe generation prompt did not extend fresh probe prefix")
        full_suffix = full_probe_gen_ids[len(full_probe_ids) :]
        b_suffix = b_probe_gen_ids[len(b_probe_ids) :]

        full_post = capture_forced_token_from_cache(
            model,
            lens_model,
            lens,
            tokenizer,
            full_probe_snap,
            full_suffix,
            len(full_probe_ids),
            layers,
            args.top_k,
            label="full_context",
        )
        forced_token_id = int(full_post["token_id"])
        fresh_post = capture_forced_token_from_cache(
            model,
            lens_model,
            lens,
            tokenizer,
            b_probe_snap,
            b_suffix,
            len(b_probe_ids),
            layers,
            args.top_k,
            forced_token_id=forced_token_id,
            label="fresh_compacted",
        )
        grafted_post = capture_forced_token_from_cache(
            model,
            lens_model,
            lens,
            tokenizer,
            grafted_probe_snap,
            b_suffix,
            len(b_probe_ids),
            layers,
            args.top_k,
            forced_token_id=forced_token_id,
            label="grafted_compacted",
        )
        full_sequence = capture_forced_sequence_from_cache(
            model,
            lens_model,
            lens,
            tokenizer,
            full_probe_snap,
            full_suffix,
            len(full_probe_ids),
            probe_target_ids,
            layers,
            args.top_k,
            label="full_context",
        )
        fresh_sequence = capture_forced_sequence_from_cache(
            model,
            lens_model,
            lens,
            tokenizer,
            b_probe_snap,
            b_suffix,
            len(b_probe_ids),
            probe_target_ids,
            layers,
            args.top_k,
            label="fresh_compacted",
        )
        alpha0_sequence = capture_forced_sequence_from_cache(
            model,
            lens_model,
            lens,
            tokenizer,
            alpha0_probe_snap,
            b_suffix,
            len(b_probe_ids),
            probe_target_ids,
            layers,
            args.top_k,
            label="alpha0_grafted_compacted",
        )
        grafted_sequence = capture_forced_sequence_from_cache(
            model,
            lens_model,
            lens,
            tokenizer,
            grafted_probe_snap,
            b_suffix,
            len(b_probe_ids),
            probe_target_ids,
            layers,
            args.top_k,
            label="grafted_compacted",
        )
        result["graft"].update(
            {
                "available": True,
                "pairs": len(pairs),
                "alpha": args.alpha,
                "cache_old_summary": cache_debug_summary(old_snap),
                "cache_full_probe": cache_debug_summary(full_probe_snap),
                "cache_fresh_probe": cache_debug_summary(b_probe_snap),
                "cache_grafted_probe": cache_debug_summary(grafted_probe_snap),
                "alignment": {
                    "old_summary_range": [summary["s_start"], summary["s_end"]],
                    "fresh_probe_summary_range": list(b_probe_summary_range),
                },
            }
        )
        result["post_boundary_probe"] = {
            "design": (
                "Same probe user message after the boundary. Full context is "
                "prefilled normally. Fresh compacted uses summary+tail text. "
                "Grafted compacted starts from the same fresh compacted text "
                "but V-grafts aligned summary-token value-cache entries from "
                "the write-time summary path. J-lens rows force the first "
                "full-context argmax token in all three states while retaining "
                "each state's own next-token candidates."
            ),
            "forced_token_source": "full_context_argmax",
            "forced_token_id": forced_token_id,
            "forced_token": full_post["token"],
            "states": {
                "full_context": full_post,
                "fresh_compacted": fresh_post,
                "grafted_compacted": grafted_post,
            },
            "forced_target_sequences": {
                "full_context": full_sequence,
                "fresh_compacted": fresh_sequence,
                "alpha0_grafted_compacted": alpha0_sequence,
                "grafted_compacted": grafted_sequence,
            },
        }
        result["states"]["grafted_post_token"] = [grafted_post]
    except Exception as exc:
        result["graft"].update(
            {
                "available": False,
                "skip_reason": f"{type(exc).__name__}: {exc}",
            }
        )

    return result


def main() -> None:
    out = run()
    path = Path(parse_args().output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(path)


if __name__ == "__main__":
    # parse_args is intentionally called inside run; this keeps import cheap for
    # syntax checks. main reparses only for the output path.
    args0 = parse_args()
    data = run()
    out_path = Path(args0.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(out_path)
