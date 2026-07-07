"""Explore small referent-recovery task shapes.

This is not a paper experiment. It is a cheap local search over candidate
problem designs that might be worth scaling later. Each case follows the same
forced-summary protocol as run_microtest.py:

  old context -> force sparse summary -> extract summary-token K/V
  fresh sparse summary -> graft K/V -> teacher-force gold answer

The question here is design triage: which task shapes produce a usable A>B
gap, and do any show a positive graft movement at tiny scale?
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
    "v005": (0.0, 0.05),
    "v010": (0.0, 0.10),
    "v025": (0.0, 0.25),
    "v050": (0.0, 0.50),
    "k005": (0.05, 0.0),
    "k010": (0.10, 0.0),
    "k025": (0.25, 0.0),
    "kv005": (0.05, 0.05),
    "kv010": (0.10, 0.10),
    "kv025": (0.25, 0.25),
    "kv050": (0.50, 0.50),
}


@dataclass
class ShapeCase:
    case_id: str
    family: str
    label: str
    old_context: str
    summary: str
    tail: str
    probe: str
    gold: str
    rationale: str


def distraction(topic: str, repeats: int = 10) -> str:
    para = (
        f"The later engineering debate was about {topic}. The team discussed "
        "implementation tradeoffs, migration constraints, ordering guarantees, "
        "backward compatibility, rollout risks, benchmark interpretation, and "
        "why noisy proxy metrics can mislead a design review. The debate was "
        "dense and technical, but it did not restate the private label mapping "
        "from the earlier payload."
    )
    return "\n\n".join(f"Distraction block {i + 1}. {para}" for i in range(repeats))


def make_old_context(title: str, payload: str, topic: str) -> str:
    return (
        f"Project anchor: {title}. The following payload defines private labels "
        "for this project. These labels are authoritative and override ordinary "
        "meanings.\n\n"
        f"{payload}\n\n"
        f"{distraction(topic)}\n"
    )


def sparse_summary(labels: list[str], kind: str, extra: str = "") -> str:
    joined = ", ".join(labels)
    suffix = f" {extra}" if extra else ""
    return (
        f"Earlier context defined private {kind} labels: {joined}. The summary "
        "kept the label names so later turns can refer to them, but omitted the "
        "exact label-to-meaning relations and the concrete mechanics."
        f"{suffix} A later technical tangent followed."
    )


def build_cases() -> list[ShapeCase]:
    cases: list[ShapeCase] = []

    # 1. Sense labels: hidden relation is a familiar semantic interpretation.
    labels = {
        "Nimbus": "checkout funnel experiment",
        "Hydra": "multi-account permission collapse",
        "Lantern": "offline refund receipt flow",
        "Rook": "search relevance rollback",
    }
    payload = "\n".join(f"- {k}: in this project, means {v}." for k, v in labels.items())
    summary = sparse_summary(list(labels), "semantic interpretation")
    for label, gold in [("Nimbus", labels["Nimbus"]), ("Hydra", labels["Hydra"])]:
        cases.append(ShapeCase(
            f"sense-{label.lower()}", "sense_label", label,
            make_old_context("we are naming ambiguous product workstreams", payload,
                             "release-note taxonomy and deployment windows"),
            summary,
            "Recent tail: we are choosing the right project meaning for a label.\n",
            f"Resolve the project-specific meaning of {label}. Answer with only the noun phrase.\nAnswer:",
            gold,
            "Lower-entropy version of the Pokemon/Nimbus style: recover sense, not an exact arbitrary string.",
        ))

    # 2. Policy choices: hidden relation is a normal implementation policy.
    labels = {
        "Citrine": "render the view as a compact comparison table",
        "Violet": "send the request through the manual escalation queue",
        "Marble": "drop the optional analytics field before export",
        "Copper": "prefer the cached preview over a fresh screenshot",
    }
    payload = "\n".join(f"- Policy {k}: {v}." for k, v in labels.items())
    summary = sparse_summary(list(labels), "policy")
    for label, gold in labels.items():
        cases.append(ShapeCase(
            f"policy-{label.lower()}", "policy_choice", label,
            make_old_context("we are defining private UI and data-handling policies",
                             payload, "database index naming and migration windows"),
            summary,
            "Recent tail: the next answer must use the private policy label, not generic product advice.\n",
            f"What is the implementation policy for {label}? Answer with only the action phrase.\nAnswer:",
            gold,
            "A candidate task where the missing relation is familiar and action-like.",
        ))

    # 3. Bug-fix labels: hidden relation is a concrete engineering fix.
    labels = {
        "Quartz": "add pointer-events: none to the transparent overlay",
        "Basil": "invalidate the member-count cache after role changes",
        "Orion": "retry the idempotent webhook after a 409 response",
        "Juniper": "normalize the timezone before grouping invoices",
    }
    payload = "\n".join(f"- Incident {k}: resolved by this fix: {v}." for k, v in labels.items())
    summary = sparse_summary(list(labels), "incident")
    for label, gold in [("Quartz", labels["Quartz"]), ("Basil", labels["Basil"])]:
        cases.append(ShapeCase(
            f"bugfix-{label.lower()}", "bug_fix", label,
            make_old_context("we are recording incident labels and exact fixes",
                             payload, "Git object storage and packfile negotiation"),
            summary,
            "Recent tail: we are writing the postmortem patch note from the private incident label.\n",
            f"State the fix for Incident {label}. Answer with only the fix phrase.\nAnswer:",
            gold,
            "Code-adjacent but not requiring exact full code generation.",
        ))

    # 4. Low-entropy code transforms: hidden relation is a standard transform.
    labels = {
        "Coral": ("convert snake_case identifiers to camelCase", "userName"),
        "Slate": ("sort records by timestamp descending", "timestamp descending"),
        "Amber": ("redact email addresses before logging", "redact email addresses"),
        "Pine": ("deduplicate rows by stable account id", "deduplicate by account id"),
    }
    payload = "\n".join(f"- Rule {k}: {rule}." for k, (rule, _) in labels.items())
    summary = sparse_summary(list(labels), "transformation rule")
    for label, (rule, gold) in labels.items():
        prompt = (
            f"Apply Rule {label}. "
            + {
                "Coral": "Input: user_name. Output:",
                "Slate": "Which ordering should be used? Answer:",
                "Amber": "What should happen to email addresses before logging? Answer:",
                "Pine": "Rows should be deduplicated by what key? Answer:",
            }[label]
        )
        cases.append(ShapeCase(
            f"transform-{label.lower()}", "low_entropy_transform", label,
            make_old_context("we are defining small named code transformations",
                             payload, "formatter benchmarks and editor integration"),
            summary,
            "Recent tail: we are applying one named transformation rule.\n",
            prompt,
            gold,
            "Still code-ish, but the answer is a familiar low-entropy transformation.",
        ))

    # 5. Field-order formats: hidden relation is a schema/format choice.
    labels = {
        "Frame 18": "tenant | region | feature | checksum",
        "Frame 27": "region | tenant | checksum",
        "Frame 31": "user | action | timestamp | nonce",
        "Frame 44": "namespace | key | version | digest",
    }
    payload = "\n".join(f"- {k}: field order is {v}." for k, v in labels.items())
    summary = sparse_summary(list(labels), "wire-format")
    for label, gold in [("Frame 18", labels["Frame 18"]), ("Frame 44", labels["Frame 44"])]:
        cases.append(ShapeCase(
            f"format-{label.lower().replace(' ', '-')}", "format_order", label,
            make_old_context("we are defining private wire-format field orders",
                             payload, "stream compression and object header checks"),
            summary,
            "Recent tail: we are implementing one private frame format.\n",
            f"Write the exact field order for {label}. Answer with only the field order.\nAnswer:",
            gold,
            "A compact structured answer; may be too exact, but less arbitrary than full code.",
        ))

    return cases


def ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False).input_ids


def score_mean_logprob(model, snapshot, prefix_len: int, probe_ids: list[int],
                       gold_ids: list[int]) -> float:
    cache = rebuild_cache(snapshot, DynamicCache)
    feed = probe_ids + gold_ids[:-1]
    pos = torch.arange(prefix_len, prefix_len + len(feed), device=model.device)[None]
    vals = tf_logprobs(model, cache, feed, gold_ids, position_ids=pos)
    return float(sum(vals) / max(1, len(vals)))


def process_case(model, tok, rope_fn, case: ShapeCase) -> dict:
    old_ids = ids(tok, case.old_context)
    summary_ids = ids(tok, "[Context note]\n" + case.summary + "\n\n")
    tail_ids = ids(tok, case.tail)
    probe_ids = ids(tok, "\n" + case.probe)
    gold_ids = ids(tok, " " + case.gold)

    full_ids = old_ids + tail_ids
    old_with_summary_ids = old_ids + summary_ids
    compact_ids = summary_ids + tail_ids

    full_snap, _ = hf_prefill_ids(model, full_ids)
    old_summary_snap, _ = hf_prefill_ids(model, old_with_summary_ids)
    compact_snap, _ = hf_prefill_ids(model, compact_ids)
    pairs = [(i, len(old_ids) + i) for i in range(len(summary_ids))]

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
            "delta_vs_B": lp_e - lp_b,
            "gap_closure": gc,
        }
        del grafted

    best_name = max(policies, key=lambda n: policies[n]["delta_vs_B"])
    best = policies[best_name]
    out = {
        "case_id": case.case_id,
        "family": case.family,
        "label": case.label,
        "gold": case.gold,
        "rationale": case.rationale,
        "token_counts": {
            "old": len(old_ids),
            "summary": len(summary_ids),
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
        "best_policy": best_name,
        "best_delta_vs_B": best["delta_vs_B"],
        "best_gap_closure": best["gap_closure"],
        "policies": policies,
    }
    del full_snap, old_summary_snap, compact_snap
    return out


def family_summary(results: list[dict]) -> dict:
    fams = sorted({r["family"] for r in results})
    out = {}
    for fam in fams:
        rows = [r for r in results if r["family"] == fam]
        usable = [r for r in rows if r["gap_A_minus_B"] > 0]
        best_gcs = [r["best_gap_closure"] for r in usable if r["best_gap_closure"] is not None]
        best_deltas = [r["best_delta_vs_B"] for r in usable]
        out[fam] = {
            "n": len(rows),
            "usable_A_gt_B": len(usable),
            "mean_A_minus_B": sum(r["gap_A_minus_B"] for r in rows) / len(rows),
            "mean_best_delta_vs_B": sum(best_deltas) / len(best_deltas) if best_deltas else None,
            "mean_best_gap_closure": sum(best_gcs) / len(best_gcs) if best_gcs else None,
            "positive_best_cases": sum(1 for r in usable if r["best_delta_vs_B"] > 0),
            "best_policies": {r["case_id"]: r["best_policy"] for r in rows},
        }
    return out


def write_report(path: Path, doc: dict) -> None:
    lines = [
        "# Problem Shape Exploration",
        "",
        f"- Model: `{doc['model']}`",
        f"- Device: `{doc['device']}`",
        f"- Cases: `{len(doc['results'])}`",
        f"- Elapsed seconds: `{doc['elapsed_sec']:.1f}`",
        "",
        "This is a local design-search pass. It asks which small problem shapes",
        "are worth turning into a real referent-recovery harness.",
        "",
        "## Family Summary",
        "",
        "| family | n | A>B | mean A-B | mean best E-B | mean best GC | positive cases | best policies |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for fam, row in doc["family_summary"].items():
        best_delta = row["mean_best_delta_vs_B"]
        best_gc = row["mean_best_gap_closure"]
        policies = ", ".join(f"{k}:{v}" for k, v in row["best_policies"].items())
        lines.append(
            f"| `{fam}` | {row['n']} | {row['usable_A_gt_B']} | "
            f"{row['mean_A_minus_B']:.3f} | "
            f"{'n/a' if best_delta is None else f'{best_delta:.3f}'} | "
            f"{'n/a' if best_gc is None else f'{best_gc:.3f}'} | "
            f"{row['positive_best_cases']} | {policies} |"
        )
    lines.extend(["", "## Case Details", ""])
    for r in doc["results"]:
        lines.append(f"### {r['case_id']} ({r['family']})")
        lines.append("")
        lines.append(f"- Label: `{r['label']}`")
        lines.append(f"- Gold: `{r['gold']}`")
        lines.append(f"- Rationale: {r['rationale']}")
        lines.append(f"- `lp_A`: `{r['lp_A']:.4f}`")
        lines.append(f"- `lp_B`: `{r['lp_B']:.4f}`")
        lines.append(f"- `A-B`: `{r['gap_A_minus_B']:.4f}`")
        lines.append(
            f"- Best policy: `{r['best_policy']}` "
            f"(`E-B` {r['best_delta_vs_B']:.4f}, "
            f"GC {r['best_gap_closure'] if r['best_gap_closure'] is not None else 'n/a'})"
        )
        lines.append("")
        lines.append("| policy | E-B | gap closure |")
        lines.append("|---|---:|---:|")
        for name, pol in r["policies"].items():
            gc = pol["gap_closure"]
            lines.append(
                f"| `{name}` | {pol['delta_vs_B']:.4f} | "
                f"{'n/a' if gc is None else f'{gc:.3f}'} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("SC_RR_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--device", default=os.environ.get("SC_RR_DEVICE", "auto"))
    ap.add_argument("--case-limit", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    outdir = Path(__file__).resolve().parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, local_files_only=True)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    rope_fn = make_rope_fn(rope_base(model), layout="half")

    cases = build_cases()
    if args.case_limit:
        cases = cases[: args.case_limit]
    results = []
    for case in cases:
        print(f"== {case.case_id} [{case.family}] {case.label}", flush=True)
        results.append(process_case(model, tok, rope_fn, case))

    doc = {
        "model": args.model,
        "device": device,
        "dtype": str(dtype),
        "elapsed_sec": time.time() - t0,
        "policies": {k: {"alpha_K": v[0], "alpha_V": v[1]} for k, v in POLICIES.items()},
        "family_summary": family_summary(results),
        "results": results,
    }
    json_path = outdir / "problem_shape_exploration.json"
    md_path = outdir / "problem_shape_exploration.md"
    json_path.write_text(json.dumps(doc, indent=2))
    write_report(md_path, doc)
    print(json.dumps(doc["family_summary"], indent=2), flush=True)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
