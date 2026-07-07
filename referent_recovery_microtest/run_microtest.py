"""Tiny referent-recovery K/V graft gate.

This is deliberately self-contained and lives outside the main experiment
pipeline. It reuses the validated HF cache snapshot / rebuild / K+V graft
helpers from src/, but constructs its own tiny cases and writes results only
under referent_recovery_microtest/outputs/.

Default model: Qwen/Qwen3-0.6B from the local HF cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arms_hf import hf_prefill_ids, rope_base  # noqa: E402
from kv_graft import graft, make_rope_fn  # noqa: E402
from kvlib_hf import rebuild_cache, tf_logprobs  # noqa: E402


POLICIES = {
    "v_only_075": (0.0, 0.75),
    "k_only_075": (0.75, 0.0),
    "coupled_075": (0.75, 0.75),
    "coupled_100": (1.0, 1.0),
}


@dataclass
class Case:
    case_id: str
    target_rule: str
    old_context: str
    summary: str
    tail: str
    probe: str
    gold: str


def numbered_rules(target_rule: str, target_body: str, distractor_name: str) -> str:
    """Return a compact bespoke payload where exactly one rule is probed."""
    rules = []
    for i in range(1, 11):
        if f"Rule {i}" == target_rule:
            body = target_body
        else:
            marker = f"ZX{i:02d}-{distractor_name.upper()}"
            body = (
                f"When visiting a helper named `{distractor_name}{i}`, replace "
                f"the first numeric literal with `__z(\"{marker}\")`, then "
                "append a trailing audit call `__audit(node.kind)`. This rule is "
                "not interchangeable with adjacent rules."
            )
        rules.append(f"Rule {i}: {body}")
    return "\n\n".join(rules)


def git_distraction(case_word: str, repeats: int = 16) -> str:
    para = (
        "The engineering review then drifted into Git transport internals. "
        "One engineer argued that packfile negotiation should treat gzip "
        "inflation as a transport-layer scheduling problem, because a slow "
        "decompressor can starve object enumeration even when network bandwidth "
        "is idle. Another engineer objected that reserved object-type values "
        "are not a mathematical limit but an architectural contract with old "
        "clients. They compared side-band packet framing, delta-base offsets, "
        "thin packs, object headers, and back-pressure around receive-pack. "
        f"The running example was `{case_word}`, but no AST, JavaScript, Z85, "
        "or minifier rule details were discussed in this debate."
    )
    return "\n\n".join(f"Git transport tangent {i + 1}. {para}" for i in range(repeats))


def build_cases() -> list[Case]:
    case1_rule = (
        "When applying this rule to `normalizeUser`, preserve the exported "
        "function signature exactly, discard the old branch body, and make the "
        "body a single return statement: `return __z(\"PELICAN-34\") && "
        "input.flag === true;`. Do not introduce helper declarations."
    )
    case1_gold = (
        "export function normalizeUser(input: UserInput) {\n"
        "  return __z(\"PELICAN-34\") && input.flag === true;\n"
        "}"
    )

    case2_rule = (
        "When applying this rule to `foldMetric`, keep the function name and "
        "parameter list, remove all arithmetic, and return the sentinel call "
        "`__metric(\"LANTERN-18\", sample.id)` directly."
    )
    case2_gold = (
        "export const foldMetric = (sample: MetricSample) => {\n"
        "  return __metric(\"LANTERN-18\", sample.id);\n"
        "};"
    )

    case3_rule = (
        "When applying this rule to `emitPatch`, keep the async modifier, delete "
        "the intermediate object construction, and return "
        "`await __patch(\"ORCHID-42\", source.path, source.kind);` as the only "
        "statement inside the function."
    )
    case3_gold = (
        "export async function emitPatch(source: PatchSource) {\n"
        "  return await __patch(\"ORCHID-42\", source.path, source.kind);\n"
        "}"
    )

    specs = [
        ("case01", "Rule 7", "normalizeUser", "atlas", case1_rule, case1_gold),
        ("case02", "Rule 4", "foldMetric", "beacon", case2_rule, case2_gold),
        ("case03", "Rule 9", "emitPatch", "cipher", case3_rule, case3_gold),
    ]

    cases: list[Case] = []
    for cid, rule_name, fn_name, word, rule_body, gold in specs:
        payload = numbered_rules(rule_name, rule_body, fn_name)
        old_context = (
            "Project anchor: we are implementing a bespoke TypeScript AST "
            "minifier whose transformation rules are intentionally nonstandard. "
            "The numbered rules below are authoritative and override normal "
            "compiler intuition.\n\n"
            "AST transformation payload:\n\n"
            f"{payload}\n\n"
            f"{git_distraction(word)}\n"
        )
        summary = (
            "Earlier context defined a custom TypeScript AST minifier with ten "
            "numbered transformation rules. The exact mechanics were too detailed "
            "to retain here. "
            f"{rule_name} specifically concerned `{fn_name}` and a private "
            "sentinel-style replacement, but this summary does not include the "
            "actual sentinel string or replacement code. A long later discussion "
            "about Git packfiles and reserved object types followed."
        )
        tail = (
            "Recent tail: we are back in the minifier implementation. The next "
            "answer must use the project-specific numbered rule, not a generic "
            "TypeScript cleanup."
        )
        probe = (
            f"\n\nProbe: Apply {rule_name} exactly to this TypeScript snippet. "
            "Output only the transformed code.\n\n"
            f"{probe_snippet(fn_name)}\n\nTransformed code:\n"
        )
        cases.append(Case(cid, rule_name, old_context, summary, tail, probe, gold))
    return cases


def probe_snippet(fn_name: str) -> str:
    if fn_name == "normalizeUser":
        return (
            "export function normalizeUser(input: UserInput) {\n"
            "  if (input.flag === true) {\n"
            "    return true;\n"
            "  }\n"
            "  return false;\n"
            "}"
        )
    if fn_name == "foldMetric":
        return (
            "export const foldMetric = (sample: MetricSample) => {\n"
            "  const adjusted = sample.value + sample.offset;\n"
            "  return adjusted * 2;\n"
            "};"
        )
    return (
        "export async function emitPatch(source: PatchSource) {\n"
        "  const draft = { path: source.path, kind: source.kind };\n"
        "  return await sendPatch(draft);\n"
        "}"
    )


def ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False).input_ids


def score_mean_logprob(model, snapshot, prefix_len: int, probe_ids: list[int],
                       gold_ids: list[int]) -> float:
    cache = rebuild_cache(snapshot, DynamicCache)
    feed = probe_ids + gold_ids[:-1]
    pos = torch.arange(prefix_len, prefix_len + len(feed),
                       device=model.device)[None]
    vals = tf_logprobs(model, cache, feed, gold_ids, position_ids=pos)
    return float(sum(vals) / max(1, len(vals)))


def process_case(model, tok, rope_fn, case: Case) -> dict:
    old_ids = ids(tok, case.old_context)
    summary_block_ids = ids(tok, "[Context note]\n" + case.summary + "\n\n")
    tail_ids = ids(tok, case.tail + "\n")
    probe_ids = ids(tok, case.probe)
    gold_ids = ids(tok, case.gold)

    full_ids = old_ids + tail_ids
    old_with_summary_ids = old_ids + summary_block_ids
    compact_ids = summary_block_ids + tail_ids

    if len(gold_ids) < 2:
        raise ValueError(f"{case.case_id}: gold tokenization too short")

    full_snap, _ = hf_prefill_ids(model, full_ids)
    old_summary_snap, _ = hf_prefill_ids(model, old_with_summary_ids)
    compact_snap, _ = hf_prefill_ids(model, compact_ids)

    pairs = [(i, len(old_ids) + i) for i in range(len(summary_block_ids))]

    lp_a = score_mean_logprob(model, full_snap, len(full_ids), probe_ids, gold_ids)
    lp_b = score_mean_logprob(model, compact_snap, len(compact_ids), probe_ids, gold_ids)

    policies = {}
    for name, (alpha_k, alpha_v) in POLICIES.items():
        grafted = graft(
            (compact_snap, old_summary_snap),
            pairs,
            alpha_K=alpha_k,
            alpha_V=alpha_v,
            positions_fresh=None,
            positions_write=None,
            rope_fn=rope_fn,
        )
        lp_e = score_mean_logprob(model, grafted, len(compact_ids), probe_ids, gold_ids)
        denom = lp_a - lp_b
        gc = (lp_e - lp_b) / denom if abs(denom) > 1e-8 else None
        policies[name] = {
            "alpha_K": alpha_k,
            "alpha_V": alpha_v,
            "lp_E": lp_e,
            "gap_closure": gc,
        }
        del grafted

    del full_snap, old_summary_snap, compact_snap
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "case_id": case.case_id,
        "target_rule": case.target_rule,
        "token_counts": {
            "old": len(old_ids),
            "summary_block": len(summary_block_ids),
            "tail": len(tail_ids),
            "full_prefix": len(full_ids),
            "compact_prefix": len(compact_ids),
            "probe": len(probe_ids),
            "gold": len(gold_ids),
            "grafted_pairs": len(pairs),
        },
        "lp_A": lp_a,
        "lp_B": lp_b,
        "gap_A_minus_B": lp_a - lp_b,
        "policies": policies,
    }


def aggregate(results: list[dict]) -> dict:
    out = {}
    for name in POLICIES:
        vals = [
            r["policies"][name]["gap_closure"]
            for r in results
            if r["policies"][name]["gap_closure"] is not None
            and r["gap_A_minus_B"] > 0
        ]
        out[name] = {
            "valid_n": len(vals),
            "mean_gap_closure": sum(vals) / len(vals) if vals else None,
            "wins_over_B": sum(
                1 for r in results if r["policies"][name]["lp_E"] > r["lp_B"]
            ),
        }
    return out


def write_markdown(path: Path, model_name: str, device: str, results: list[dict],
                   agg: dict) -> None:
    valid_gaps = [r for r in results if r["gap_A_minus_B"] > 0]
    any_positive_policy = any(
        row["mean_gap_closure"] is not None and row["mean_gap_closure"] > 0
        for row in agg.values()
    )
    if not valid_gaps:
        verdict = (
            "No usable gate: full context did not beat compacted context, so the "
            "case design/model pair does not create a measurable recovery target."
        )
    elif any_positive_policy:
        verdict = (
            "Promising gate: full context beats compacted context, and at least "
            "one graft policy moves the correct continuation toward the full-context "
            "ceiling."
        )
    else:
        verdict = (
            "Negative gate at this scale: full context beats compacted context, "
            "so the task creates a real floor/ceiling gap, but every tested graft "
            "policy moved below the compacted baseline. Do not scale this exact "
            "harness shape without changing the sparse-summary design, alpha range, "
            "model scale, or target construction."
        )
    lines = [
        "# Referent-Recovery Microtest Results",
        "",
        f"- Model: `{model_name}`",
        f"- Device: `{device}`",
        f"- Cases: `{len(results)}`",
        "",
        "This is a cheap falsification gate for the referent-recovery harness idea.",
        "It is not a publishable result by itself.",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Aggregate",
        "",
        "| policy | valid n | mean gap closure | wins over B |",
        "|---|---:|---:|---:|",
    ]
    for name, row in agg.items():
        mgc = row["mean_gap_closure"]
        mgc_s = "n/a" if mgc is None else f"{mgc:.3f}"
        lines.append(f"| `{name}` | {row['valid_n']} | {mgc_s} | {row['wins_over_B']} |")
    lines.extend(["", "## Cases", ""])
    for r in results:
        lines.append(f"### {r['case_id']} ({r['target_rule']})")
        lines.append("")
        lines.append(
            f"- Tokens: full prefix `{r['token_counts']['full_prefix']}`, "
            f"compact prefix `{r['token_counts']['compact_prefix']}`, "
            f"summary graft pairs `{r['token_counts']['grafted_pairs']}`"
        )
        lines.append(f"- `lp_A`: `{r['lp_A']:.4f}`")
        lines.append(f"- `lp_B`: `{r['lp_B']:.4f}`")
        lines.append(f"- `A-B`: `{r['gap_A_minus_B']:.4f}`")
        lines.append("")
        lines.append("| policy | lp_E | gap closure |")
        lines.append("|---|---:|---:|")
        for name, p in r["policies"].items():
            gc = p["gap_closure"]
            gc_s = "n/a" if gc is None else f"{gc:.3f}"
            lines.append(f"| `{name}` | {p['lp_E']:.4f} | {gc_s} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("SC_RR_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--device", default=os.environ.get("SC_RR_DEVICE", "auto"))
    args = ap.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    dtype = torch.float16 if device == "mps" else torch.float32

    outdir = Path(__file__).resolve().parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    rope_fn = make_rope_fn(rope_base(model), layout="half")

    cases = build_cases()[: args.limit]
    results = []
    for case in cases:
        print(f"== {case.case_id} {case.target_rule}", flush=True)
        results.append(process_case(model, tok, rope_fn, case))

    agg = aggregate(results)
    doc = {
        "model": args.model,
        "device": device,
        "dtype": str(dtype),
        "elapsed_sec": time.time() - t0,
        "policies": {k: {"alpha_K": v[0], "alpha_V": v[1]} for k, v in POLICIES.items()},
        "aggregate": agg,
        "results": results,
    }
    json_path = outdir / "microtest_results.json"
    md_path = outdir / "microtest_results.md"
    json_path.write_text(json.dumps(doc, indent=2))
    write_markdown(md_path, args.model, device, results, agg)
    print(json.dumps(agg, indent=2), flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
