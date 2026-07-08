#!/usr/bin/env python3
"""Compare raw logit-lens readouts with the repaired J-lens snapshots."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jlens_boundary_probe"))

from boundary_probe import choose_layers, dtype_from_name, model_input_device, render_ids, topk_from_logits  # noqa: E402
from qwen_topic_probe import PROMPT_CASES, messages, phrase_matches  # noqa: E402


DEFAULT_JLENS_JSON = "topic_sensitivity_probe/outputs/qwen36_topic_probe_static_fix.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--layers", default="24,36,48,53,55,58,60,62")
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--jlens-json", default=DEFAULT_JLENS_JSON)
    p.add_argument("--output", default="topic_sensitivity_probe/outputs/raw_logit_compare.json")
    p.add_argument("--report", default="topic_sensitivity_probe/outputs/raw_logit_compare.md")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return p.parse_args()


def model_layer_count(model: torch.nn.Module) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise RuntimeError("could not determine decoder layer count")


def final_norm(model: torch.nn.Module) -> torch.nn.Module | None:
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    return None


def lm_head(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "embed_out"):
        return model.embed_out
    raise RuntimeError("could not find output projection")


def layer_hidden(hidden_states: tuple[torch.Tensor, ...], layer: int) -> torch.Tensor:
    # HF decoder hidden_states are embeddings at index 0, then output after each
    # layer at index layer + 1. That convention is the raw logit-lens target.
    idx = layer + 1
    if idx >= len(hidden_states):
        raise IndexError(f"layer {layer} unavailable in hidden_states length {len(hidden_states)}")
    return hidden_states[idx]


def top_tokens_for_hidden(
    model: torch.nn.Module,
    tok: Any,
    hidden: torch.Tensor,
    top_k: int,
    apply_norm: bool,
) -> list[dict[str, Any]]:
    h = hidden
    norm = final_norm(model)
    if apply_norm and norm is not None:
        h = norm(h)
    logits = lm_head(model)(h).float()
    return topk_from_logits(tok, logits, top_k)


def load_jlens_reference(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for case in data.get("lens", {}).get("cases", []):
        by_pos = {}
        for pos in case.get("positions", []):
            by_pos[str(pos["position"])] = pos
        out[case.get("case_id")] = by_pos
    return out


def comparable_rows(doc: dict[str, Any], focus_layers: list[str]) -> list[dict[str, Any]]:
    rows = []
    for case in doc["cases"]:
        for pos in case.get("positions", []):
            for layer in focus_layers:
                raw_norm = pos["layers"].get(layer, {}).get("raw_with_final_norm", [])
                jlens = pos.get("jlens_layers", {}).get(layer, [])
                raw_tokens = [x["token"] for x in raw_norm[:8]]
                jlens_tokens = [x["token"] for x in jlens[:8]]
                rows.append({
                    "case_id": case["case_id"],
                    "token": pos["token"],
                    "layer": int(layer),
                    "raw_norm": raw_tokens,
                    "jlens": jlens_tokens,
                })
    return rows


def write_report(path: Path, doc: dict[str, Any]) -> None:
    lines = [
        "# Raw Logit-Lens Comparison",
        "",
        f"- Model: `{doc['model']}`",
        f"- Elapsed seconds: `{doc['elapsed_sec']:.1f}`",
        f"- Layers: `{', '.join(str(x) for x in doc['layers'])}`",
        "",
        "This is a compact comparison between a conventional raw logit lens and the repaired J-lens prompt-token snapshots. Raw readouts are shown after applying the model's final normalization before unembedding; the JSON also includes the unnormalized raw readout.",
        "",
        "## Focus Rows",
        "",
        "| case | prompt token | layer | raw logit lens | J-lens |",
        "|---|---|---:|---|---|",
    ]
    focus_layers = ["48", "55", "58", "60", "62"]
    for row in comparable_rows(doc, focus_layers):
        if row["layer"] not in {48, 55, 58, 60, 62}:
            continue
        raw = ", ".join(repr(x) for x in row["raw_norm"])
        jlens = ", ".join(repr(x) for x in row["jlens"])
        lines.append(f"| `{row['case_id']}` | `{row['token']}` | {row['layer']} | {raw} | {jlens} |")

    lines.extend([
        "",
        "## Short Interpretation",
        "",
        "Raw logit lens can recover some coarse lexical associations, especially name pieces and place words. J-lens is cleaner for the specific observation we care about here: the phrase-final Chinese prompt token surfaces event/protest-style associations even when the generated answer redirects to reform/development language.",
        "",
        "So this is not a result that only exists because of J-lens, but J-lens makes the contrast easier to see and explain.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    t0 = time.time()
    output = Path(args.output)
    report = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype_from_name(args.dtype),
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    torch.set_grad_enabled(False)

    n_layers = model_layer_count(model)
    layers = choose_layers(args.layers, n_layers, list(range(n_layers)))
    jlens_ref = load_jlens_reference(Path(args.jlens_json))

    cases = []
    for case in PROMPT_CASES:
        print(f"RAW {case.case_id}", flush=True)
        ids = render_ids(tok, messages(case.user), False)
        matches = phrase_matches(tok, ids, case.phrase)
        positions = sorted(set(
            pos
            for start, phrase_ids in matches
            for pos in (start, start + len(phrase_ids) - 1)
        ))
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([ids], device=model_input_device(model)),
                use_cache=False,
                output_hidden_states=True,
            )
        rows = []
        for pos in positions:
            row = {
                "position": pos,
                "token_id": int(ids[pos]),
                "token": tok.decode([int(ids[pos])]),
                "context_window": tok.decode(ids[max(0, pos - 12): min(len(ids), pos + 13)]),
                "layers": {},
                "jlens_layers": {},
            }
            ref_pos = jlens_ref.get(case.case_id, {}).get(str(pos), {})
            row["jlens_layers"] = ref_pos.get("layers", {})
            for layer in layers:
                hidden = layer_hidden(out.hidden_states, layer)[0, pos]
                row["layers"][str(layer)] = {
                    "raw_with_final_norm": top_tokens_for_hidden(model, tok, hidden, args.top_k, True),
                    "raw_without_final_norm": top_tokens_for_hidden(model, tok, hidden, args.top_k, False),
                }
            rows.append(row)
        cases.append({
            "case_id": case.case_id,
            "phrase": case.phrase,
            "positions": rows,
        })

    doc = {
        "model": args.model,
        "dtype": args.dtype,
        "elapsed_sec": time.time() - t0,
        "layers": layers,
        "top_k": args.top_k,
        "jlens_json": args.jlens_json,
        "cases": cases,
    }
    output.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    write_report(report, doc)
    print(f"Wrote {output}", flush=True)
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
