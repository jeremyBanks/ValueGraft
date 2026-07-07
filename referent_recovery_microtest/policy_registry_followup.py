"""Follow-up prospecting for private policy / transform registries.

The first design search suggested two promising shapes:

  1. private policy label -> familiar action phrase
  2. named low-entropy transform -> short familiar operation

This runner widens those shapes while keeping the protocol cheap and local.
It deliberately maps structure rather than proving anything: task family,
policy category, summary style, and alpha policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
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
    "v002": (0.0, 0.02),
    "v005": (0.0, 0.05),
    "v010": (0.0, 0.10),
    "v020": (0.0, 0.20),
    "v030": (0.0, 0.30),
    "k002": (0.02, 0.0),
    "k005": (0.05, 0.0),
    "k010": (0.10, 0.0),
    "k020": (0.20, 0.0),
    "kv002": (0.02, 0.02),
    "kv005": (0.05, 0.05),
    "kv010": (0.10, 0.10),
    "kv020": (0.20, 0.20),
    "k005_v020": (0.05, 0.20),
    "k010_v020": (0.10, 0.20),
}


@dataclass(frozen=True)
class RegistryEntry:
    label: str
    category: str
    action: str
    family: str


@dataclass
class RegistryCase:
    case_id: str
    family: str
    category: str
    summary_style: str
    label: str
    action: str
    old_context: str
    summary: str
    tail: str
    probe: str
    gold: str


POLICY_ENTRIES = [
    RegistryEntry("Citrine", "ui_rendering", "render the view as a compact comparison table", "policy_choice"),
    RegistryEntry("Copper", "ui_rendering", "prefer the cached preview over a fresh screenshot", "policy_choice"),
    RegistryEntry("Pearl", "ui_rendering", "show the empty state before the loading spinner", "policy_choice"),
    RegistryEntry("Violet", "routing", "send the request through the manual escalation queue", "policy_choice"),
    RegistryEntry("Indigo", "routing", "route account deletion requests to privacy review", "policy_choice"),
    RegistryEntry("Umber", "routing", "retry the idempotent webhook after a 409 response", "policy_choice"),
    RegistryEntry("Marble", "privacy_export", "drop the optional analytics field before export", "policy_choice"),
    RegistryEntry("Saffron", "privacy_export", "redact email addresses before writing logs", "policy_choice"),
    RegistryEntry("Nickel", "privacy_export", "hash account IDs before analytics upload", "policy_choice"),
    RegistryEntry("Cobalt", "cache_policy", "invalidate the member-count cache after role changes", "policy_choice"),
    RegistryEntry("Jade", "cache_policy", "reuse the cached entitlement list for read-only checks", "policy_choice"),
    RegistryEntry("Hazel", "cache_policy", "refresh the local index after permission edits", "policy_choice"),
    RegistryEntry("Topaz", "release_review", "require reviewer signoff before enabling the flag", "policy_choice"),
    RegistryEntry("Garnet", "release_review", "stage the rollout to internal accounts first", "policy_choice"),
    RegistryEntry("Opal", "release_review", "block release until rollback notes are attached", "policy_choice"),
]


TRANSFORM_ENTRIES = [
    RegistryEntry("Coral", "identifier_transform", "userName", "low_entropy_transform"),
    RegistryEntry("Iris", "identifier_transform", "user_profile_card", "low_entropy_transform"),
    RegistryEntry("Slate", "ordering_transform", "timestamp descending", "low_entropy_transform"),
    RegistryEntry("Cedar", "ordering_transform", "tenant id", "low_entropy_transform"),
    RegistryEntry("Pine", "dedupe_transform", "deduplicate by account id", "low_entropy_transform"),
    RegistryEntry("Flint", "validation_transform", "trim whitespace", "low_entropy_transform"),
    RegistryEntry("Amber", "privacy_transform", "redact email addresses", "low_entropy_transform"),
    RegistryEntry("Azure", "normalization_transform", "UTC", "low_entropy_transform"),
]


def policy_payload(entries: list[RegistryEntry]) -> str:
    lines = [
        "Private policy registry. Each label names exactly one implementation policy.",
        "These mappings are project-local and must be used literally.",
    ]
    for entry in entries:
        lines.append(f"- {entry.label} [{entry.category}]: {entry.action}.")
    return "\n".join(lines)


def transform_payload(entries: list[RegistryEntry]) -> str:
    detail = {
        "Coral": "Input example `user_name` becomes `userName`.",
        "Iris": "Input example `user-profile-card` becomes `user_profile_card`.",
        "Slate": "Sort rows by the newest timestamp first.",
        "Cedar": "Group rows using their tenant id.",
        "Pine": "Remove duplicate rows by stable account id.",
        "Flint": "Trim surrounding whitespace before validation.",
        "Amber": "Replace email addresses with a redacted marker before logging.",
        "Azure": "Normalize displayed times to UTC.",
    }
    lines = [
        "Private transform registry. Each label names exactly one small data/code transform.",
        "The labels are project-local and must override ordinary guesses.",
    ]
    for entry in entries:
        lines.append(f"- Rule {entry.label} [{entry.category}]: {detail[entry.label]}")
    return "\n".join(lines)


def distraction(repeats: int = 8) -> str:
    para = (
        "The later discussion shifted into a dense review of Git packfile "
        "transport, decompressor scheduling, reserved object-type values, "
        "packet framing, rollback windows, and benchmark methodology. The team "
        "debated implementation details at length, but did not repeat any "
        "private registry mappings."
    )
    return "\n\n".join(f"Distraction {i + 1}. {para}" for i in range(repeats))


def make_summary(entries: list[RegistryEntry], family_name: str, style: str) -> str:
    if style == "label_only":
        labels = ", ".join(e.label for e in entries)
        return (
            f"Earlier context defined private {family_name} labels: {labels}. "
            "The summary keeps the label names but omits the label-to-action "
            "relations. A later technical tangent followed."
        )
    if style == "typed":
        typed = ", ".join(f"{e.label} ({e.category})" for e in entries)
        return (
            f"Earlier context defined private {family_name} labels and their "
            f"coarse categories: {typed}. The summary keeps labels and categories "
            "but omits the exact action or transform associated with each label. "
            "A later technical tangent followed."
        )
    raise ValueError(style)


def transform_probe(entry: RegistryEntry) -> str:
    prompts = {
        "Coral": "Apply Rule Coral to `user_name`. Output only the transformed identifier.\nAnswer:",
        "Iris": "Apply Rule Iris to `user-profile-card`. Output only the transformed filename stem.\nAnswer:",
        "Slate": "Apply Rule Slate. Which ordering should be used? Answer with only the ordering.\nAnswer:",
        "Cedar": "Apply Rule Cedar. Rows should be grouped by what key? Answer with only the key.\nAnswer:",
        "Pine": "Apply Rule Pine. What operation should be performed? Answer with only the operation phrase.\nAnswer:",
        "Flint": "Apply Rule Flint. What normalization happens before validation? Answer with only the action.\nAnswer:",
        "Amber": "Apply Rule Amber. What should happen before logging? Answer with only the action.\nAnswer:",
        "Azure": "Apply Rule Azure. What timezone should displayed times use? Answer with only the timezone.\nAnswer:",
    }
    return prompts[entry.label]


def build_cases(summary_styles: list[str]) -> list[RegistryCase]:
    cases: list[RegistryCase] = []
    groups = [
        ("policy", "policy registry", POLICY_ENTRIES, policy_payload(POLICY_ENTRIES)),
        ("transform", "transform registry", TRANSFORM_ENTRIES, transform_payload(TRANSFORM_ENTRIES)),
    ]
    for group_name, family_name, entries, payload in groups:
        old_context = (
            f"Project anchor: we are maintaining a private {family_name}. "
            "The following payload is authoritative and later work depends on "
            "using the exact project-local mapping.\n\n"
            f"{payload}\n\n{distraction()}\n"
        )
        for style in summary_styles:
            summary = make_summary(entries, family_name, style)
            for entry in entries:
                tail = (
                    f"Recent tail: continue using the private {family_name}. "
                    "Answer from the project-local mapping, not from generic advice.\n"
                )
                if entry.family == "policy_choice":
                    probe = (
                        f"What is the implementation policy for {entry.label}? "
                        "Answer with only the action phrase.\nAnswer:"
                    )
                else:
                    probe = transform_probe(entry)
                cases.append(RegistryCase(
                    case_id=f"{group_name}-{style}-{entry.label.lower()}",
                    family=entry.family,
                    category=entry.category,
                    summary_style=style,
                    label=entry.label,
                    action=entry.action,
                    old_context=old_context,
                    summary=summary,
                    tail=tail,
                    probe=probe,
                    gold=entry.action,
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


def process_case(model, tok, rope_fn, case: RegistryCase) -> dict:
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
        policies[name] = {
            "alpha_K": alpha_k,
            "alpha_V": alpha_v,
            "lp_E": lp_e,
            "delta_vs_B": lp_e - lp_b,
            "gap_closure": (lp_e - lp_b) / denom if abs(denom) > 1e-8 else None,
        }
        del grafted

    best_name = max(policies, key=lambda n: policies[n]["delta_vs_B"])
    best = policies[best_name]
    out = {
        "case_id": case.case_id,
        "family": case.family,
        "category": case.category,
        "summary_style": case.summary_style,
        "label": case.label,
        "gold": case.gold,
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


def summarize_group(rows: list[dict]) -> dict:
    usable = [r for r in rows if r["gap_A_minus_B"] > 0]
    best_gcs = [r["best_gap_closure"] for r in usable if r["best_gap_closure"] is not None]
    best_deltas = [r["best_delta_vs_B"] for r in usable]
    return {
        "n": len(rows),
        "usable_A_gt_B": len(usable),
        "positive_best_cases": sum(1 for r in usable if r["best_delta_vs_B"] > 0),
        "mean_A_minus_B": sum(r["gap_A_minus_B"] for r in rows) / len(rows),
        "mean_best_delta_vs_B": sum(best_deltas) / len(best_deltas) if best_deltas else None,
        "mean_best_gap_closure": sum(best_gcs) / len(best_gcs) if best_gcs else None,
        "best_policy_counts": dict(Counter(r["best_policy"] for r in rows)),
    }


def grouped_summary(results: list[dict]) -> dict:
    groups: dict[str, dict[str, list[dict]]] = {
        "family": defaultdict(list),
        "family_style": defaultdict(list),
        "family_category": defaultdict(list),
        "summary_style": defaultdict(list),
    }
    for r in results:
        groups["family"][r["family"]].append(r)
        groups["family_style"][f"{r['family']} / {r['summary_style']}"].append(r)
        groups["family_category"][f"{r['family']} / {r['category']}"].append(r)
        groups["summary_style"][r["summary_style"]].append(r)
    return {
        group_name: {k: summarize_group(v) for k, v in sorted(group.items())}
        for group_name, group in groups.items()
    }


def policy_surface(results: list[dict]) -> dict:
    out = {}
    for policy in POLICIES:
        vals = []
        pos = 0
        for r in results:
            if r["gap_A_minus_B"] <= 0:
                continue
            delta = r["policies"][policy]["delta_vs_B"]
            vals.append(delta)
            pos += int(delta > 0)
        out[policy] = {
            "usable_n": len(vals),
            "positive_cases": pos,
            "mean_delta_vs_B": sum(vals) / len(vals) if vals else None,
        }
    return out


def policy_surface_by_family(results: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["family"]].append(r)
    return {family: policy_surface(rows) for family, rows in sorted(groups.items())}


def write_report(path: Path, doc: dict) -> None:
    lines = [
        "# Policy Registry Follow-Up",
        "",
        f"- Model: `{doc['model']}`",
        f"- Device: `{doc['device']}`",
        f"- Cases: `{len(doc['results'])}`",
        f"- Elapsed seconds: `{doc['elapsed_sec']:.1f}`",
        "",
        "This is a noisy local search over candidate benchmark shapes. It is useful",
        "for picking promising regions, not for making claims.",
        "",
    ]

    for section, rows in [
        ("By Family", doc["grouped_summary"]["family"]),
        ("By Family And Summary Style", doc["grouped_summary"]["family_style"]),
        ("By Family And Category", doc["grouped_summary"]["family_category"]),
    ]:
        lines.extend([
            f"## {section}",
            "",
            "| group | n | A>B | positive | mean A-B | mean best E-B | mean best GC | best-policy counts |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        for name, row in rows.items():
            lines.append(
                f"| `{name}` | {row['n']} | {row['usable_A_gt_B']} | "
                f"{row['positive_best_cases']} | {row['mean_A_minus_B']:.3f} | "
                f"{fmt(row['mean_best_delta_vs_B'])} | {fmt(row['mean_best_gap_closure'])} | "
                f"{row['best_policy_counts']} |"
            )
        lines.append("")

    lines.extend([
        "## Policy Surface",
        "",
        "| policy | usable n | positive cases | mean E-B |",
        "|---|---:|---:|---:|",
    ])
    for policy, row in doc["policy_surface"].items():
        lines.append(
            f"| `{policy}` | {row['usable_n']} | {row['positive_cases']} | "
            f"{fmt(row['mean_delta_vs_B'])} |"
        )
    lines.append("")

    for family, surface in doc["policy_surface_by_family"].items():
        lines.extend([
            f"## Policy Surface For `{family}`",
            "",
            "| policy | usable n | positive cases | mean E-B |",
            "|---|---:|---:|---:|",
        ])
        for policy, row in surface.items():
            lines.append(
                f"| `{policy}` | {row['usable_n']} | {row['positive_cases']} | "
                f"{fmt(row['mean_delta_vs_B'])} |"
            )
        lines.append("")

    lines.extend(["", "## Top Cases", ""])
    top = sorted(doc["results"], key=lambda r: r["best_delta_vs_B"], reverse=True)[:12]
    lines.extend([
        "| case | family | category | style | best | E-B | GC | A-B | gold |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ])
    for r in top:
        lines.append(
            f"| `{r['case_id']}` | `{r['family']}` | `{r['category']}` | "
            f"`{r['summary_style']}` | `{r['best_policy']}` | "
            f"{r['best_delta_vs_B']:.3f} | {fmt(r['best_gap_closure'])} | "
            f"{r['gap_A_minus_B']:.3f} | {r['gold']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("SC_RR_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--device", default=os.environ.get("SC_RR_DEVICE", "auto"))
    ap.add_argument("--summary-style", choices=["label_only", "typed", "both"], default="both")
    ap.add_argument("--case-limit", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    styles = ["label_only", "typed"] if args.summary_style == "both" else [args.summary_style]
    cases = build_cases(styles)
    if args.case_limit:
        cases = cases[:args.case_limit]

    outdir = Path(__file__).resolve().parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, local_files_only=True)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    rope_fn = make_rope_fn(rope_base(model), layout="half")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"== {i}/{len(cases)} {case.case_id}", flush=True)
        results.append(process_case(model, tok, rope_fn, case))

    doc = {
        "model": args.model,
        "device": device,
        "dtype": str(dtype),
        "elapsed_sec": time.time() - t0,
        "policies": {k: {"alpha_K": v[0], "alpha_V": v[1]} for k, v in POLICIES.items()},
        "grouped_summary": grouped_summary(results),
        "policy_surface": policy_surface(results),
        "policy_surface_by_family": policy_surface_by_family(results),
        "results": results,
    }
    json_path = outdir / "policy_registry_followup.json"
    md_path = outdir / "policy_registry_followup.md"
    json_path.write_text(json.dumps(doc, indent=2))
    write_report(md_path, doc)
    print(json.dumps(doc["grouped_summary"]["family"], indent=2), flush=True)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
