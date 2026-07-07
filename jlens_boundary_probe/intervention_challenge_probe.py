#!/usr/bin/env python3
"""Sparse-summary intervention probes for ValueGraft/J-lens examples.

The batch probe uses summaries that already contain most answers. This script
uses deliberately sparse summaries and no retained tail by default. The goal is
to test whether aligned write-time value grafts carry any recoverable state
when the visible compacted prompt preserves labels but omits key details.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from boundary_probe import (
    DEFAULT_LENS_REPO,
    DEFAULT_LENS_REVISION,
    DEFAULT_QWEN36_LENS,
    choose_layers,
    dtype_from_name,
    parse_alpha_sweep,
)
from intervention_batch_probe import ProbeCase, run_case
from multi_demo_scan import CODING_MESSAGES, HOME_MESSAGES, SUPPORT_MESSAGES, Demo
from plain_conversation_probe import BLOCK_PARTY_MESSAGES
from pokemon_probe import POKEMON_MESSAGES


CHALLENGE_DEMOS: dict[str, Demo] = {
    "block_party_sparse": Demo(
        "block_party_sparse",
        BLOCK_PARTY_MESSAGES,
        (
            "Riverside block party compressed memory: Maple, Robin, Blue, "
            "Green, Red, B-410, P-771, Orchid, big oak table, Friday, "
            "Saturday, Crane, Mateo, and Jules are local terms with prior "
            "decisions. Preserve their resolved meanings, stale/current "
            "status, and constraints."
        ),
        ["Maple", "B-410", "P-771", "Blue", "Green", "Crane"],
        "sparse label-only event summary",
    ),
    "checkout_sparse": Demo(
        "checkout_sparse",
        CODING_MESSAGES,
        (
            "Checkout incident compressed memory: Mercury, Falcon, Raven, "
            "Patch 17, R3, Nova, and red button are project-local terms with "
            "prior resolved meanings, current/stale decisions, and safety "
            "constraints. Preserve the old decisions."
        ),
        ["Mercury", "Falcon", "Raven", "Patch 17", "R3", "Nova"],
        "sparse label-only incident summary",
    ),
    "home_sparse": Demo(
        "home_sparse",
        HOME_MESSAGES,
        (
            "Weekend trip compressed memory: Cedar, Basil, Delta, Orange, "
            "big cooler, and soft coolers are local trip terms with prior "
            "resolved meanings and stale/current decisions. Preserve the "
            "old constraints."
        ),
        ["Cedar", "Basil", "Delta", "Orange", "big cooler", "soft coolers"],
        "sparse label-only household summary",
    ),
    "pokemon_sparse": Demo(
        "pokemon_sparse",
        POKEMON_MESSAGES,
        (
            "Pokemon Emerald compressed memory: Soup, Vacuum, Ghost, Ghost2, "
            "Breloom, Hariyama, Dex, Makuhita, Castform, Rayquaza, Bagon, "
            "Ultra Balls, and graveyard are run-local terms with prior "
            "decisions and current/stale statuses. Preserve the run state."
        ),
        ["Soup", "Vacuum", "Ghost", "Ghost2", "Dex", "Makuhita", "Castform"],
        "sparse label-only Pokemon summary",
    ),
    "support_sparse": Demo(
        "support_sparse",
        SUPPORT_MESSAGES,
        (
            "Support handoff compressed memory: Atlas, Sage, T-88, T-104, "
            "amber, and purple are ticket-local terms with prior resolved "
            "meanings, current/stale statuses, and safety constraints. "
            "Preserve the handoff state."
        ),
        ["Atlas", "Sage", "T-88", "T-104", "amber", "purple"],
        "sparse label-only support summary",
    ),
}


CHALLENGE_CASES: list[ProbeCase] = [
    ProbeCase(
        "sparse_block_permit",
        "block_party_sparse",
        "Which permit number is current for the insurance form, and which one is stale? Answer in one sentence.",
        "Use P-771 as current; B-410 is stale.",
        ("P-771", "current", "B-410", "stale"),
        "Sparse summary names both permit numbers but omits which is current.",
    ),
    ProbeCase(
        "sparse_block_maple",
        "block_party_sparse",
        "What is Maple for in the Riverside plan? Answer in one sentence.",
        "Maple is the library's Maple Room for storage and volunteer check-in.",
        ("Maple", "Maple Room", "storage", "volunteer check-in"),
        "Sparse summary names Maple but omits its room/storage/check-in meaning.",
    ),
    ProbeCase(
        "sparse_checkout_falcon",
        "checkout_sparse",
        "Should rollback use Falcon or Raven, and why? Answer in one sentence.",
        "Use Raven for rollback; Falcon was rejected because it drops subscription coupons.",
        ("Raven", "Falcon", "rejected", "subscription coupons"),
        "Sparse summary names Falcon and Raven but omits accepted/rejected mapping.",
    ),
    ProbeCase(
        "sparse_checkout_patch",
        "checkout_sparse",
        "Which hotfix label is live, and which patch label is stale? Answer in one sentence.",
        "R3 is the current live hotfix label; Patch 17 is stale.",
        ("R3", "current live", "Patch 17", "stale"),
        "Sparse summary names Patch 17 and R3 but omits live/stale mapping.",
    ),
    ProbeCase(
        "sparse_home_delta",
        "home_sparse",
        "What does Delta mean in the trip plan? Answer in one sentence.",
        "Delta is ferry route Delta 6 at 7:40, not the airline.",
        ("Delta", "Delta 6", "7:40", "airline"),
        "Sparse summary names Delta but omits route/time disambiguation.",
    ),
    ProbeCase(
        "sparse_pokemon_ghost2",
        "pokemon_sparse",
        "Can Ghost2 be brought forward, and what happened to Ghost? Answer in one sentence.",
        "Ghost2 is alive and made it to Victory Road; Ghost is in the graveyard.",
        ("Ghost2", "alive", "Victory Road", "Ghost", "graveyard"),
        "Sparse summary names Ghost/Ghost2/graveyard but omits dead-vs-alive mapping.",
    ),
    ProbeCase(
        "sparse_pokemon_dex",
        "pokemon_sparse",
        "What does Dex owe from the Ruby save? Answer in one sentence.",
        "Dex owes the Makuhita-for-Castform trade from Ruby.",
        ("Dex", "Makuhita", "Castform", "Ruby"),
        "Sparse summary names trade labels but omits the owed trade relation.",
    ),
    ProbeCase(
        "sparse_support_escalation",
        "support_sparse",
        "Which escalation should be cited, and which one is stale? Answer in one sentence.",
        "Use T-104 as the active escalation; T-88 is stale.",
        ("T-104", "active escalation", "T-88", "stale"),
        "Sparse summary names both tickets but omits active/stale mapping.",
    ),
    ProbeCase(
        "sparse_support_amber",
        "support_sparse",
        "Which recovery path should be used, and which path is forbidden? Answer in one sentence.",
        "Use amber read-only recovery; do not run purple.",
        ("amber", "read-only recovery", "purple"),
        "Sparse summary names amber and purple but omits safe/forbidden mapping.",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-filename", default=DEFAULT_QWEN36_LENS)
    p.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
    p.add_argument("--cases", default="all")
    p.add_argument("--output", default="outputs/qwen36_intervention_challenge_probe.json")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--layers", default="16,32,48,62")
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--alpha-sweep", default="0,0.1,0.25,0.5,0.75,1")
    p.add_argument("--tail-messages", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return p.parse_args()


def selected_cases(spec: str) -> list[ProbeCase]:
    if spec == "all":
        return CHALLENGE_CASES
    names = {x.strip() for x in spec.split(",") if x.strip()}
    out = [case for case in CHALLENGE_CASES if case.case_id in names]
    missing = names - {case.case_id for case in out}
    if missing:
        raise ValueError(f"unknown cases: {sorted(missing)}")
    return out


def main() -> None:
    args = parse_args()
    cases = selected_cases(args.cases)

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

    result: dict[str, Any] = {
        "model": args.model,
        "design": (
            "Sparse summaries retain labels but omit key detail; no retained "
            "tail by default. The goal is a stricter intervention test where "
            "fresh compacted text is intentionally under-informative."
        ),
        "lens": {
            "repo": args.lens_repo,
            "filename": args.lens_filename,
            "revision": args.lens_revision,
            "source_layers": lens.source_layers,
            "sampled_layers": layers,
        },
        "top_k": args.top_k,
        "alpha": args.alpha,
        "alpha_sweep": parse_alpha_sweep(args.alpha_sweep, args.alpha),
        "tail_messages": args.tail_messages,
        "cases": {},
    }

    for case in cases:
        print(f"RUN {case.case_id}", flush=True)
        result["cases"][case.case_id] = run_case(
            case,
            CHALLENGE_DEMOS[case.demo_name],
            model,
            lens_model,
            lens,
            tokenizer,
            layers,
            args.top_k,
            args.alpha,
            args.alpha_sweep,
            args.tail_messages,
        )
        print(f"DONE {case.case_id}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(out_path)


if __name__ == "__main__":
    main()
