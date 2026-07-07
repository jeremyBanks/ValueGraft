#!/usr/bin/env python3
"""Narrow J-lens readouts around a compaction boundary.

This is an isolated prototype. It deliberately avoids importing or modifying
the live serving shim. The grafted-post-token state is enabled only for models
whose Hugging Face cache exposes standard per-layer keys/values.
"""

from __future__ import annotations

import argparse
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
    {
        "role": "user",
        "content": (
            "Before continuing, summarize the work so far so another agent can "
            "resume from a compacted context."
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
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError("past_key_values has no .layers; standard K/V snapshot unavailable")
    snap = []
    for layer in layers:
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        if k is None or v is None:
            raise TypeError("cache layer lacks .keys/.values; standard K/V snapshot unavailable")
        snap.append((k.clone(), v.clone()))
    return snap, out.logits[:, -1, :]


def rebuild_standard_cache(snap: list[tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    cache = DynamicCache()
    for i, (k, v) in enumerate(snap):
        cache.update(k.clone(), v.clone(), i)
    return cache


def blend_values(
    b_snap: list[tuple[torch.Tensor, torch.Tensor]],
    old_snap: list[tuple[torch.Tensor, torch.Tensor]],
    pairs: list[tuple[int, int]],
    alpha: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    new_idx = torch.tensor([n for n, _ in pairs])
    old_idx = torch.tensor([o for _, o in pairs])
    out = []
    for li, (k, v) in enumerate(b_snap):
        v2 = v.clone()
        vf = v2[..., new_idx, :].float()
        vo = old_snap[li][1][..., old_idx, :].float()
        v2[..., new_idx, :] = ((1 - alpha) * vf + alpha * vo).to(v2.dtype)
        out.append((k, v2))
    return out


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


def capture_first_generated_token_from_cache(
    model: torch.nn.Module,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    snap: list[tuple[torch.Tensor, torch.Tensor]],
    suffix_ids: list[int],
    next_position: int,
    layers: list[int],
    top_k: int,
) -> dict[str, Any]:
    from jlens.hooks import ActivationRecorder

    cache = rebuild_standard_cache(snap)
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
        _, logits = snapshot_standard_kv(model, [])
    token_id = int(torch.argmax(logits, dim=-1).item())

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
        "position": int(next_position),
        "token_id": token_id,
        "token": tokenizer.decode([token_id]),
        "layers": {},
    }
    for layer in layers:
        residual = activations[layer][0, 0].float().unsqueeze(0)
        logits_l = lens_model.unembed(lens.transport(residual, layer))[0]
        row["layers"][str(layer)] = topk_from_logits(tokenizer, logits_l, top_k)
    return row


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

    summary_text_ids = tokenizer(summary["text"], add_special_tokens=False).input_ids
    b_summary_start = find_subsequence(b_ids, summary_text_ids)
    if b_summary_start is None:
        b_summary_positions = [-1]
        b_summary_range = None
    else:
        b_summary_range = (b_summary_start, b_summary_start + len(summary_text_ids))
        b_summary_positions = sample_summary_positions(*b_summary_range)

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
        b_snap, _ = snapshot_standard_kv(model, b_ids)
        if b_summary_range is None:
            raise TypeError("could not locate summary text in fresh compacted IDs")
        pairs = alignment_pairs_by_exact_tokens(
            b_ids,
            summary["old_ids"],
            b_summary_range,
            (summary["s_start"], summary["s_end"]),
        )
        if not pairs:
            raise TypeError("no exact summary-token pairs for graft alignment")
        grafted = blend_values(b_snap, old_snap, pairs, args.alpha)
        b_gen_ids = render_ids(tokenizer, b_messages, True)
        suffix = b_gen_ids[len(b_ids) :]
        post = capture_first_generated_token_from_cache(
            model,
            lens_model,
            lens,
            tokenizer,
            grafted,
            suffix,
            len(b_ids),
            layers,
            args.top_k,
        )
        result["graft"].update({"available": True, "pairs": len(pairs), "alpha": args.alpha})
        result["states"]["grafted_post_token"] = [post]
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
