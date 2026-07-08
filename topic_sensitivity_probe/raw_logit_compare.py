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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jlens_boundary_probe"))

from boundary_probe import (  # noqa: E402
    DEFAULT_LENS_REPO,
    DEFAULT_LENS_REVISION,
    DEFAULT_QWEN36_LENS,
    choose_layers,
    dtype_from_name,
    model_input_device,
    render_ids,
    topk_from_logits,
)
from qwen_topic_probe import PROMPT_CASES, messages, phrase_matches  # noqa: E402


DEFAULT_JLENS_JSON = "topic_sensitivity_probe/outputs/qwen36_topic_probe_static_fix.json"

CONCEPT_GROUPS = {
    "history_event": [
        "1989", "event", "events", "incident", "protest", "protests", "student",
        "students", "democracy", "demonstration", "demonstrations", "military",
        "crackdown", "massacre", "deaths", "killed", "Tank", "Man", "June",
        "Fourth", "事件", "发生", "发生了什么", "抗议", "示威", "镇压", "军队",
        "伤亡", "学生", "六四",
    ],
    "official_stability": [
        "stability", "stable", "order", "harmony", "turmoil", "disturbance",
        "law", "unity", "official", "security", "秩序", "稳定", "统一", "动乱",
        "风波", "依法", "处理",
    ],
    "redirect_development": [
        "reform", "development", "economy", "economic", "technology",
        "education", "exchange", "opening", "progress", "改革", "开放",
        "经济", "发展", "科技", "教育", "交流", "进展", "成就",
    ],
    "refusal": [
        "cannot", "can't", "unable", "sorry", "sensitive", "restricted",
        "policy", "censorship", "blocked", "不能", "无法", "敏感", "审查",
    ],
    "landmark": [
        "square", "plaza", "Beijing", "China", "capital", "tourist",
        "landmark", "monument", "gate", "museum", "天安门", "广场", "北京",
        "中国", "景点", "游客",
    ],
}

SIMILARITY_PAIRS = [
    ("english_1989_vs_chinese_1989", "tiananmen_1989", "tiananmen_chinese"),
    ("english_basic_vs_chinese_1989", "tiananmen_basic", "tiananmen_chinese"),
    ("june_fourth_vs_chinese_1989", "june_fourth", "tiananmen_chinese"),
    ("tank_man_vs_chinese_1989", "tank_man", "tiananmen_chinese"),
    ("forbidden_city_vs_chinese_1989", "forbidden_city_control", "tiananmen_chinese"),
    ("trafalgar_vs_chinese_1989", "trafalgar_control", "tiananmen_chinese"),
    ("kent_state_vs_chinese_1989", "kent_state_control", "tiananmen_chinese"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--layers", default="24,36,48,53,55,58,60,62")
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--jlens-json", default=DEFAULT_JLENS_JSON)
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-filename", default=DEFAULT_QWEN36_LENS)
    p.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
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


def build_concept_token_ids(tok: Any) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for group, terms in CONCEPT_GROUPS.items():
        ids: list[int] = []
        for term in terms:
            for variant in (term, " " + term):
                ids.extend(int(tid) for tid in tok(variant, add_special_tokens=False).input_ids)
        seen = set()
        groups[group] = [tid for tid in ids if not (tid in seen or seen.add(tid))]
    return groups


def concept_scores(tok: Any, logits: torch.Tensor, concept_ids: dict[str, list[int]]) -> dict[str, dict[str, Any]]:
    logits = logits.float().detach().cpu()
    out: dict[str, dict[str, Any]] = {}
    for group, ids in concept_ids.items():
        idx = torch.tensor(ids, dtype=torch.long)
        vals = logits.index_select(0, idx)
        best_i = int(torch.argmax(vals).item())
        best_id = int(idx[best_i].item())
        out[group] = {
            "max_logit": float(vals[best_i].item()),
            "best_token_id": best_id,
            "best_token": tok.decode([best_id]),
        }
    return out


def ranked_groups(scores: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (
            {"group": group, **payload}
            for group, payload in scores.items()
        ),
        key=lambda row: row["max_logit"],
        reverse=True,
    )


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().cpu(), b.float().cpu(), dim=0).item())


def comparable_rows(doc: dict[str, Any], focus_layers: list[str]) -> list[dict[str, Any]]:
    rows = []
    for case in doc["cases"]:
        for pos in case.get("positions", []):
            for layer in focus_layers:
                raw_norm = pos["layers"].get(layer, {}).get("raw_with_final_norm", [])
                jlens = pos["layers"].get(layer, {}).get("jlens", [])
                raw_groups = pos["layers"].get(layer, {}).get("raw_concepts_ranked", [])
                jlens_groups = pos["layers"].get(layer, {}).get("jlens_concepts_ranked", [])
                raw_tokens = [x["token"] for x in raw_norm[:8]]
                jlens_tokens = [x["token"] for x in jlens[:8]]
                rows.append({
                    "case_id": case["case_id"],
                    "token": pos["token"],
                    "layer": int(layer),
                    "raw_norm": raw_tokens,
                    "jlens": jlens_tokens,
                    "raw_top_group": raw_groups[0]["group"] if raw_groups else "n/a",
                    "raw_group_token": raw_groups[0]["best_token"] if raw_groups else "n/a",
                    "jlens_top_group": jlens_groups[0]["group"] if jlens_groups else "n/a",
                    "jlens_group_token": jlens_groups[0]["best_token"] if jlens_groups else "n/a",
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
        "This is a compact comparison between a conventional raw logit lens and J-lens on the same prompt-token positions. Raw readouts are shown after applying the model's final normalization before unembedding; the JSON also includes the unnormalized raw readout.",
        "",
        "## Focus Rows",
        "",
        "| case | prompt token | layer | raw top group | J-lens top group | raw top tokens | J-lens top tokens |",
        "|---|---|---:|---|---|---|---|",
    ]
    focus_layers = ["48", "55", "58", "60", "62"]
    for row in comparable_rows(doc, focus_layers):
        if row["layer"] not in {48, 55, 58, 60, 62}:
            continue
        raw = ", ".join(repr(x) for x in row["raw_norm"])
        jlens = ", ".join(repr(x) for x in row["jlens"])
        raw_group = f"`{row['raw_top_group']}` via `{row['raw_group_token']}`"
        jlens_group = f"`{row['jlens_top_group']}` via `{row['jlens_group_token']}`"
        lines.append(f"| `{row['case_id']}` | `{row['token']}` | {row['layer']} | {raw_group} | {jlens_group} | {raw} | {jlens} |")

    lines.extend([
        "",
        "## English/Chinese Internal Similarity",
        "",
        "Cosine similarity compares the phrase-final prompt token in each pair. Raw uses the final-normalized residual stream; J-lens uses the transported final-space vector before unembedding.",
        "",
        "| pair | layer | raw residual cosine | J-lens transported cosine | raw groups | J-lens groups |",
        "|---|---:|---:|---:|---|---|",
    ])
    for pair in doc.get("similarities", []):
        if pair["layer"] not in {48, 55, 58, 60, 62}:
            continue
        raw_groups = f"`{pair['left_raw_group']}` / `{pair['right_raw_group']}`"
        jlens_groups = f"`{pair['left_jlens_group']}` / `{pair['right_jlens_group']}`"
        lines.append(
            f"| `{pair['pair_id']}` | {pair['layer']} | {pair['raw_cosine']:.4f} | "
            f"{pair['jlens_transport_cosine']:.4f} | {raw_groups} | {jlens_groups} |"
        )

    lines.extend([
        "",
        "## Short Interpretation",
        "",
        "Raw logit lens already recovers much of the key signal: the phrase-final Chinese prompt token surfaces event/what-happened associations in later layers even though the generated answer redirects to reform/development language. J-lens is not uniquely necessary here, but it makes the signal cleaner and easier to explain, especially where it surfaces protest/event terms directly rather than requiring concept grouping over noisier raw readouts.",
        "",
        "The safer conclusion is comparative: behavior and candidate scores show redirection rather than refusal; raw lens and J-lens both show latent referent availability; J-lens is a clearer illustration of that knowledge-vs-routing distinction, not proof of censorship, panic, or a causal mechanism.",
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
    concept_ids = build_concept_token_ids(tok)
    jlens_ref = load_jlens_reference(Path(args.jlens_json))

    print("LENS load", flush=True)
    import jlens
    from jlens.hooks import ActivationRecorder

    lens_model = jlens.from_hf(model, tok, force_bos=False)
    lens = jlens.JacobianLens.from_pretrained(
        args.lens_repo,
        filename=args.lens_filename,
        revision=args.lens_revision,
    )

    cases = []
    vector_index: dict[tuple[str, int], dict[str, Any]] = {}
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
        with torch.no_grad(), ActivationRecorder(lens_model.layers, at=layers) as rec:
            lens_model.forward(torch.tensor([ids], device=lens_model.input_device))
            activations = {layer: rec.activations[layer].detach() for layer in layers}
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
            is_phrase_final = bool(positions and pos == positions[-1])
            for layer in layers:
                hidden = layer_hidden(out.hidden_states, layer)[0, pos]
                norm = final_norm(model)
                raw_norm_vec = norm(hidden) if norm is not None else hidden
                raw_norm_logits = lm_head(model)(raw_norm_vec).float()
                residual = activations[layer][0, pos].float().unsqueeze(0)
                transported = lens.transport(residual, layer)[0].float()
                jlens_logits = lens_model.unembed(transported.unsqueeze(0))[0].float()
                raw_ranked = ranked_groups(concept_scores(tok, raw_norm_logits, concept_ids))
                jlens_ranked = ranked_groups(concept_scores(tok, jlens_logits, concept_ids))
                row["layers"][str(layer)] = {
                    "raw_with_final_norm": top_tokens_for_hidden(model, tok, hidden, args.top_k, True),
                    "raw_without_final_norm": top_tokens_for_hidden(model, tok, hidden, args.top_k, False),
                    "raw_concepts_ranked": raw_ranked,
                    "jlens": topk_from_logits(tok, jlens_logits, args.top_k),
                    "jlens_concepts_ranked": jlens_ranked,
                }
                if is_phrase_final:
                    vector_index[(case.case_id, layer)] = {
                        "raw_norm_vec": raw_norm_vec.detach().cpu(),
                        "jlens_transport_vec": transported.detach().cpu(),
                        "raw_top_group": raw_ranked[0]["group"] if raw_ranked else "n/a",
                        "jlens_top_group": jlens_ranked[0]["group"] if jlens_ranked else "n/a",
                    }
            rows.append(row)
        cases.append({
            "case_id": case.case_id,
            "phrase": case.phrase,
            "positions": rows,
        })

    similarities = []
    for pair_id, left, right in SIMILARITY_PAIRS:
        for layer in layers:
            left_v = vector_index.get((left, layer))
            right_v = vector_index.get((right, layer))
            if not left_v or not right_v:
                continue
            similarities.append({
                "pair_id": pair_id,
                "left_case": left,
                "right_case": right,
                "layer": layer,
                "raw_cosine": cosine(left_v["raw_norm_vec"], right_v["raw_norm_vec"]),
                "jlens_transport_cosine": cosine(left_v["jlens_transport_vec"], right_v["jlens_transport_vec"]),
                "left_raw_group": left_v["raw_top_group"],
                "right_raw_group": right_v["raw_top_group"],
                "left_jlens_group": left_v["jlens_top_group"],
                "right_jlens_group": right_v["jlens_top_group"],
            })

    doc = {
        "model": args.model,
        "dtype": args.dtype,
        "elapsed_sec": time.time() - t0,
        "layers": layers,
        "top_k": args.top_k,
        "jlens_json": args.jlens_json,
        "cases": cases,
        "similarities": similarities,
    }
    output.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    write_report(report, doc)
    print(f"Wrote {output}", flush=True)
    print(f"Wrote {report}", flush=True)


if __name__ == "__main__":
    main()
