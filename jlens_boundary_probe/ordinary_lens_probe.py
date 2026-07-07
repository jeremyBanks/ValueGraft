#!/usr/bin/env python3
"""Ordinary-regime J-lens corroboration (role PRESERVED + retained tail).

This is the honest, subtle-but-real companion to ``strong_example_probe.py``. It
reuses the *exact* three-state ValueGraft + J-lens machinery from
``intervention_batch_probe.run_case`` (full-context / fresh-compacted /
aligned-graft, plus a shifted negative control and an alpha sweep). The ONLY
things that change relative to the strong probe are the *examples* and the
*regime*:

  * The authored ``summary`` KEEPS the role/referent in the text (it is NOT
    stripped). The disambiguating keywords survive compaction, so the
    fresh-compacted state already carries the meaning.
  * A short tail is RETAINED (``--tail-messages 2`` by default), so the
    establishing user turn is preserved after the boundary as well.

This is deliberately the ORDINARY regime -- the one where a prior agent found a
small but POSITIVE aligned-vs-shifted closure. It is NOT the sparse
"role-lost" regime of the strong probe (where the role is destroyed and only the
graft can restore it). Because the role is already present, the aligned graft
should nudge the readout only subtly, and crucially MORE than the shifted
(misaligned) control -- the alignment-sensitive signature we want to feature as
honest corroboration.

We use a handful of ``sense`` and ``referent`` plants from OUR synthetic
conversations (``data/synthetic/``). Each yields a focused ``Demo`` (system +
the verbatim establishing user turn + a short neutral ack) and a ``ProbeCase``
whose ``probe_target`` is a hinge-first phrasing of the gold role.

Because it delegates to the same ``run_case`` and writes the same schema, the
existing validators (``validate_three_state_probe.py``,
``validate_intervention_artifact.py``) and analysis
(``analyze_intervention_batch.py``) apply unchanged.

Self-test locally with ``--dry-run`` (no model / no lens / no tokenizer): it
constructs and prints every probe spec so the ordinary examples can be eyeballed
-- in particular it verifies the ROLE IS PRESENT in each summary.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# As in strong_example_probe: keep --dry-run runnable with no ML deps. We define
# field-compatible Demo / ProbeCase dataclasses here and import the real
# run_case lazily inside main past the dry-run gate.

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


@dataclass(frozen=True)
class Demo:
    name: str
    messages: list[dict[str, str]]
    summary: str
    anchors: list[str]
    shape: str


@dataclass(frozen=True)
class ProbeCase:
    case_id: str
    demo_name: str
    probe_user: str
    probe_target: str
    focus_phrases: tuple[str, ...]
    rationale: str


ACK = "Understood -- noted, I'll keep that in mind."


@dataclass(frozen=True)
class OrdinarySpec:
    """An ordinary-regime probe drawn from one synthetic-conversation plant.

    Unlike ``StrongSpec``, the ``summary`` PRESERVES the role: every phrase in
    ``role_keywords`` must appear in ``summary`` (the role survives compaction).
    ``hinge`` phrases must appear in ``target`` so the lens readout can be
    focused at the semantic hinge.
    """

    case_id: str
    conv_id: str          # synthetic conversation file stem, e.g. "c01"
    plant_id: str         # plant id inside that conversation
    anchor: str           # salient anchor token kept in the summary
    summary: str          # role-PRESERVING compaction (keeps the referent)
    target: str           # hinge-first gold role phrasing (forced sequence)
    hinge: tuple[str, ...]  # disambiguating phrases; must appear in target
    role_keywords: tuple[str, ...]  # role phrases that MUST appear in summary
    rationale: str


# ---------------------------------------------------------------------------
# 4 ordinary-regime examples: 2 sense + 2 referent. Every summary KEEPS the
# role. These are the honest, alignment-sensitive corroboration cases.
# ---------------------------------------------------------------------------
ORDINARY_SPECS: list[OrdinarySpec] = [
    OrdinarySpec(
        case_id="nimbus_sense_ordinary",
        conv_id="c01",
        plant_id="c01-sense-1",
        anchor="Nimbus",
        summary=(
            "Founder is planning the Chartkite product launch with a tiny team. "
            "Pricing, positioning, and onboarding have been discussed. Terminology "
            "note preserved: 'Nimbus' here always means the self-serve signup "
            "funnel work (the new flow from landing page to first login), not the "
            "infra team's cloud-hosting migration. Continue supporting launch prep."
        ),
        target="The self-serve signup funnel, not the cloud migration.",
        hinge=("self-serve", "signup funnel"),
        role_keywords=("self-serve", "signup"),
        rationale=(
            "'Nimbus' is overloaded (cloud migration vs self-serve signup funnel). "
            "The summary KEEPS the signup-funnel sense and the establishing turn is "
            "retained, so fresh already carries the role; the aligned graft should "
            "move the hinge readout only subtly, but more than the shifted control."
        ),
    ),
    OrdinarySpec(
        case_id="hydra_sense_ordinary",
        conv_id="c04",
        plant_id="c04-sense-1",
        anchor="Hydra",
        summary=(
            "Author is preparing an academic paper submission with their advisor. "
            "Venue, evaluation, and experimental write-up have been discussed. "
            "Terminology note preserved: 'Hydra' here always means the multi-task "
            "comparison model used as the main baseline, not the GPU cluster the "
            "lab runs jobs on. Continue supporting the paper."
        ),
        target="The multi-task comparison model, not the GPU cluster.",
        hinge=("multi-task", "comparison model"),
        role_keywords=("multi-task", "comparison"),
        rationale=(
            "'Hydra' is overloaded (GPU cluster vs multi-task comparison model). "
            "The summary KEEPS the comparison-model sense and the tail is retained; "
            "aligned graft should nudge the hinge readout subtly above the shifted "
            "control."
        ),
    ),
    OrdinarySpec(
        case_id="marisol_referent_ordinary",
        conv_id="c01",
        plant_id="c01-referent-2",
        anchor="Marisol",
        summary=(
            "Founder is planning the Chartkite launch. Advisor Marisol's proposal "
            "has been adopted: instead of a big public splash, run an invite-only "
            "beta with about thirty design partners first, opening the doors only "
            "once each partner has shipped one real report. Continue folding this "
            "into the launch plan."
        ),
        target="An invite-only beta with about thirty design partners, opening only after each ships a real report.",
        hinge=("invite-only", "design partners"),
        role_keywords=("invite-only", "design partners"),
        rationale=(
            "Referent plant: the concrete proposal is KEPT in the summary and the "
            "establishing turn is retained. The aligned graft should reinforce the "
            "design-partner/report specifics slightly more than the shifted control."
        ),
    ),
    OrdinarySpec(
        case_id="venue_referent_ordinary",
        conv_id="c04",
        plant_id="c04-referent-1",
        anchor="EMNLP",
        summary=(
            "Author is preparing a paper submission with their advisor. Venue "
            "decision preserved: after weighing the ACL main track, the EMNLP "
            "findings track, and the TACL journal, they committed to the EMNLP "
            "findings track given timeline and scope. Continue the write-up toward "
            "that target."
        ),
        target="The EMNLP findings track, the second of the three options.",
        hinge=("EMNLP", "findings"),
        role_keywords=("emnlp", "findings"),
        rationale=(
            "Referent plant: the chosen venue is KEPT in the summary and the tail is "
            "retained. Aligned graft should reinforce the EMNLP-findings choice "
            "slightly above the shifted control."
        ),
    ),
]


def _load_conversation(data_dir: Path, conv_id: str) -> dict[str, Any]:
    path = data_dir / f"{conv_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"conversation file not found: {path}")
    return json.loads(path.read_text())


def _find_plant(conv: dict[str, Any], plant_id: str) -> dict[str, Any]:
    for plant in conv.get("plants", []):
        if plant.get("id") == plant_id:
            return plant
    raise KeyError(f"plant {plant_id!r} not found in conversation {conv.get('id')!r}")


def _find_establishing_message(conv: dict[str, Any], middle_user: str) -> tuple[int, str]:
    for i, msg in enumerate(conv.get("messages", [])):
        if msg.get("role") == "user" and msg.get("content", "").strip() == middle_user.strip():
            return i, msg["content"]
    for i, msg in enumerate(conv.get("messages", [])):
        if msg.get("role") == "user" and middle_user.strip()[:60] in msg.get("content", ""):
            return i, msg["content"]
    return -1, middle_user


def _system_message(conv: dict[str, Any]) -> dict[str, str]:
    msgs = conv.get("messages", [])
    if msgs and msgs[0].get("role") == "system":
        return {"role": "system", "content": msgs[0]["content"]}
    return {"role": "system", "content": "You are a helpful assistant."}


@dataclass(frozen=True)
class BuiltCase:
    spec: OrdinarySpec
    plant: dict[str, Any]
    demo: Demo
    case: ProbeCase
    establishing_index: int
    establishing_text: str


def build_case(spec: OrdinarySpec, data_dir: Path) -> BuiltCase:
    """Construct the Demo + ProbeCase for one ordinary example (no ML deps)."""

    conv = _load_conversation(data_dir, spec.conv_id)
    plant = _find_plant(conv, spec.plant_id)

    est_index, est_text = _find_establishing_message(conv, plant["middle_user"])
    system_msg = _system_message(conv)

    messages = [
        system_msg,
        {"role": "user", "content": est_text},
        {"role": "assistant", "content": ACK},
    ]

    demo = Demo(
        name=spec.case_id,
        messages=messages,
        summary=spec.summary,
        anchors=[spec.anchor],
        shape=f"ordinary role-preserved example from {spec.conv_id} (retained tail)",
    )

    case = ProbeCase(
        case_id=spec.case_id,
        demo_name=spec.case_id,
        probe_user=plant["probe"],
        probe_target=spec.target,
        focus_phrases=spec.hinge,
        rationale=spec.rationale,
    )

    return BuiltCase(
        spec=spec,
        plant=plant,
        demo=demo,
        case=case,
        establishing_index=est_index,
        establishing_text=est_text,
    )


def selected_specs(spec_arg: str) -> list[OrdinarySpec]:
    if spec_arg == "all":
        return ORDINARY_SPECS
    names = {x.strip() for x in spec_arg.split(",") if x.strip()}
    out = [s for s in ORDINARY_SPECS if s.case_id in names]
    missing = names - {s.case_id for s in out}
    if missing:
        raise ValueError(f"unknown cases: {sorted(missing)}")
    return out


def _dry_run_checks(built: BuiltCase) -> dict[str, Any]:
    spec = built.spec
    plant = built.plant
    summary_l = spec.summary.lower()
    target_l = spec.target.lower()

    # Role-PRESERVED invariant: every role keyword present in the summary.
    role_present = {k: (k.lower() in summary_l) for k in spec.role_keywords}
    # Also report the plant's own keywords for cross-check.
    plant_keywords = [str(k) for k in plant.get("keywords", [])]
    plant_kw_present = {k: (k.lower() in summary_l) for k in plant_keywords}

    hinge_in_target = {h: (h.lower() in target_l) for h in spec.hinge}
    anchor_in_summary = spec.anchor.lower() in summary_l

    return {
        "role_keywords_present_in_summary": role_present,
        "plant_keywords_present_in_summary": plant_kw_present,
        "anchor_present_in_summary": anchor_in_summary,
        "hinge_present_in_target": hinge_in_target,
        "establishing_turn_found_in_conversation": built.establishing_index >= 0,
        "establishing_message_index": built.establishing_index,
        "context_message_count": len(built.demo.messages),
        "summary_char_len": len(spec.summary),
        "ok": (
            all(role_present.values())
            and anchor_in_summary
            and all(hinge_in_target.values())
            and built.establishing_index >= 0
        ),
    }


def dry_run(specs: list[OrdinarySpec], data_dir: Path, tail_messages: int) -> int:
    print("=" * 72)
    print("ORDINARY-LENS PROBE  --  DRY RUN (no model / no lens / no tokenizer)")
    print(f"data dir: {data_dir}   tail_messages={tail_messages} (role KEPT + retained tail)")
    print("=" * 72)
    all_ok = True
    for spec in specs:
        built = build_case(spec, data_dir)
        checks = _dry_run_checks(built)
        all_ok = all_ok and checks["ok"]
        plant = built.plant
        print()
        print(f"### CASE {spec.case_id}   [{spec.conv_id} / {spec.plant_id} / {plant.get('category')}]")
        print(f"  anchor (kept)     : {spec.anchor!r}")
        print(f"  role keywords     : {list(spec.role_keywords)}  -> present_in_summary={checks['role_keywords_present_in_summary']}")
        print(f"  hinge token(s)    : {list(spec.hinge)}")
        print(f"  gold role (plant) : {plant.get('gold')}")
        print(f"  probe_target      : {spec.target!r}")
        print(f"  probe_user        : {plant.get('probe')}")
        print(f"  role-KEPT summary : {spec.summary}")
        print(f"  establishing turn : conv msg #{built.establishing_index} "
              f"({'verbatim' if checks['establishing_turn_found_in_conversation'] else 'FALLBACK'})")
        print(f"    -> {built.establishing_text[:140]}...")
        print(f"  rationale         : {spec.rationale}")
        print(f"  CHECKS            : role_present={all(checks['role_keywords_present_in_summary'].values())} "
              f"anchor_in_summary={checks['anchor_present_in_summary']} "
              f"hinge_in_target={checks['hinge_present_in_target']} "
              f"context_msgs={checks['context_message_count']}")
        print(f"  => {'OK' if checks['ok'] else 'PROBLEM'}")
    print()
    print("=" * 72)
    print(f"DRY RUN {'PASSED' if all_ok else 'FAILED'}: "
          f"{len(specs)} ordinary-regime probe spec(s) constructed "
          f"(role PRESERVED in every summary).")
    print("=" * 72)
    return 0 if all_ok else 1


def parse_args() -> argparse.Namespace:
    try:
        from boundary_probe import (
            DEFAULT_LENS_REPO,
            DEFAULT_LENS_REVISION,
            DEFAULT_QWEN36_LENS,
        )
    except Exception:
        DEFAULT_LENS_REPO = "neuronpedia/jacobian-lens"
        DEFAULT_LENS_REVISION = "qwen-n1000"
        DEFAULT_QWEN36_LENS = (
            "qwen3.6-27b/jlens/Salesforce-wikitext/"
            "Qwen3.6-27B_jacobian_lens_n1000.pt"
        )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    p.add_argument("--lens-filename", default=DEFAULT_QWEN36_LENS)
    p.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--cases", default="all")
    p.add_argument("--output", default="outputs/qwen36_ordinary_lens.json")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--layers", default="8,16,24,32,40,48,56,62")
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--alpha-sweep", default="0,0.25,0.5,0.75,1")
    # Ordinary regime KEEPS a short retained tail (role present after boundary).
    p.add_argument("--tail-messages", type=int, default=2)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.add_argument("--dry-run", action="store_true",
                   help="construct + print probe specs without loading the model")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    specs = selected_specs(args.cases)

    if args.dry_run:
        raise SystemExit(dry_run(specs, data_dir, args.tail_messages))

    # --- heavy imports only past the dry-run gate ---
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from boundary_probe import choose_layers, dtype_from_name, parse_alpha_sweep
    from intervention_batch_probe import run_case
    import jlens

    built_cases = [build_case(spec, data_dir) for spec in specs]

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
            "Ordinary-regime J-lens corroboration: role/referent PRESERVED in the "
            "summary plus a retained tail (--tail-messages 2). Same three-state "
            "ValueGraft + J-lens intervention as intervention_batch_probe.run_case: "
            "full-context vs fresh-compacted vs aligned-graft, with a shifted "
            "negative control and an alpha sweep. Because the role already survives "
            "compaction, the aligned graft is expected to move the hinge readout "
            "only subtly -- but more than the shifted control (alignment-sensitive, "
            "the honest small-positive corroboration). Sense + referent plants from "
            "data/synthetic."
        ),
        "regime": "ordinary (role preserved, retained tail)",
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

    for built in built_cases:
        case = built.case
        print(f"RUN {case.case_id}", flush=True)
        try:
            case_result = run_case(
                case,
                built.demo,
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
            case_result["ordinary_example"] = {
                "conversation_id": built.spec.conv_id,
                "plant_id": built.spec.plant_id,
                "plant_category": built.plant.get("category"),
                "anchor_preserved": built.spec.anchor,
                "hinge_phrases": list(built.spec.hinge),
                "role_keywords_kept": list(built.spec.role_keywords),
                "gold_role": built.plant.get("gold"),
                "plant_keywords": [str(k) for k in built.plant.get("keywords", [])],
                "establishing_message_index": built.establishing_index,
                "regime": "ordinary (role preserved, retained tail)",
            }
            result["cases"][case.case_id] = case_result
            print(f"DONE {case.case_id}", flush=True)
        except Exception as exc:  # noqa: BLE001
            result["cases"][case.case_id] = {
                "case_id": case.case_id,
                "demo_name": case.demo_name,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"ERROR {case.case_id}: {type(exc).__name__}: {exc}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(out_path)


if __name__ == "__main__":
    main()
