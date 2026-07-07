#!/usr/bin/env python3
"""Strong "label-preserved / role-lost" intervention probes.

This runner reuses the *exact* three-state ValueGraft + J-lens machinery from
``intervention_batch_probe.run_case`` (full-context / fresh-compacted /
aligned-graft, plus a shifted negative control and an alpha sweep). The only
thing that changes is the *examples*: instead of the synthetic block-party /
checkout / pokemon demos, it pulls the strongest "sense" and "referent" plants
from OUR synthetic conversations in ``data/synthetic/`` -- cases where a LABEL
is preserved across compaction but its ROLE / referent is lost.

For each selected plant we build a focused ``Demo``:

  * ``messages``   = the conversation's own system prompt + the user turn that
                     establishes the disambiguation (the plant's ``middle_user``,
                     taken verbatim from the real conversation) + a short neutral
                     assistant acknowledgement. The ROLE lives only in that user
                     turn.
  * ``summary``    = an authored compaction that PRESERVES the label token but
                     drops every role-bearing keyword (so fresh-compacted context
                     has the label with no referent -- the "role lost" state).

and a ``ProbeCase`` whose ``probe_target`` is a crisp, hinge-first phrasing of
the plant's gold role, so the J-lens readout can be focused at the SEMANTIC
HINGE token (the disambiguating word, e.g. "signup" / "multi-task" /
"database") rather than at path or punctuation tokens.

Because it delegates to the same ``run_case`` and writes the same schema, the
existing validators (``validate_three_state_probe.py``,
``validate_intervention_artifact.py``) and analysis
(``analyze_intervention_batch.py``) apply unchanged.

Self-test locally with ``--dry-run`` (no model, no lens, no tokenizer): it
constructs and prints every probe spec so the strong examples can be eyeballed.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# NOTE: ``intervention_batch_probe`` and ``multi_demo_scan`` transitively import
# torch/transformers, which are absent on a CPU-only box. To keep ``--dry-run``
# runnable with no ML deps, we do NOT import them at module scope. Instead we
# define field-compatible ``Demo`` / ``ProbeCase`` dataclasses here (run_case
# only reads their attributes, so these duck-type cleanly) and import the real
# ``run_case`` lazily inside ``main`` past the dry-run gate.

# ``data/synthetic`` lives two levels up from this file (repo_root/data/...).
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


@dataclass(frozen=True)
class Demo:
    """Field-compatible with ``multi_demo_scan.Demo`` (attribute access only)."""

    name: str
    messages: list[dict[str, str]]
    summary: str
    anchors: list[str]
    shape: str


@dataclass(frozen=True)
class ProbeCase:
    """Field-compatible with ``intervention_batch_probe.ProbeCase``."""

    case_id: str
    demo_name: str
    probe_user: str
    probe_target: str
    focus_phrases: tuple[str, ...]
    rationale: str

# A short, deliberately neutral assistant acknowledgement. It must NOT restate
# the role: the whole point is that the role is present only in the user's
# establishing turn, so fresh-compacted (which drops it) can be contrasted with
# the write-time graft (which should carry it).
ACK = "Understood -- noted, I'll keep that meaning in mind."


@dataclass(frozen=True)
class StrongSpec:
    """A strong-example probe drawn from one synthetic-conversation plant.

    Everything the *model run* needs is derived from the plant plus a small
    amount of authored text: the role-lost ``summary`` and the hinge-first
    ``target``. ``label`` is the token that survives compaction; ``hinge`` are
    the disambiguating (role-bearing) phrases we focus the lens on -- each must
    appear in ``target``.
    """

    case_id: str
    conv_id: str          # synthetic conversation file stem, e.g. "c01"
    plant_id: str         # plant id inside that conversation, e.g. "c01-sense-1"
    label: str            # the preserved label, e.g. "Nimbus"
    summary: str          # role-lost compaction: keeps label, drops keywords
    target: str           # hinge-first gold role phrasing (forced sequence)
    hinge: tuple[str, ...]  # disambiguating phrases; must appear in target
    rationale: str


# ---------------------------------------------------------------------------
# The strong examples. All three are "sense" plants: a single label with two
# live referents, disambiguated once by the user, then compacted away. These
# are the cases the prior agent flagged as missing (Nimbus / Hydra / sandbox).
# ---------------------------------------------------------------------------
STRONG_SPECS: list[StrongSpec] = [
    StrongSpec(
        case_id="nimbus_sense",
        conv_id="c01",
        plant_id="c01-sense-1",
        label="Nimbus",
        summary=(
            "Founder is planning the Chartkite product launch with a tiny team. "
            "Pricing, positioning, and rollout timing have been discussed, along "
            "with several internal terminology notes. Nimbus is one of the active "
            "workstreams referenced for launch. Continue supporting launch prep."
        ),
        target="The self-serve signup funnel, not the cloud migration.",
        hinge=("self-serve", "signup funnel"),
        rationale=(
            "'Nimbus' is overloaded (cloud-hosting migration vs self-serve signup "
            "funnel). The summary keeps the label but drops the referent; only the "
            "write-time graft should restore the signup-funnel sense at the hinge."
        ),
    ),
    StrongSpec(
        case_id="hydra_sense",
        conv_id="c04",
        plant_id="c04-sense-1",
        label="Hydra",
        summary=(
            "Author is preparing an academic paper submission with their advisor. "
            "Venue, evaluation, and experimental write-up have been discussed, "
            "along with several lab-specific naming notes. Hydra is referenced in "
            "the experimental results being written up. Continue supporting the "
            "paper."
        ),
        target="The multi-task comparison model, not the GPU cluster.",
        hinge=("multi-task", "comparison model"),
        rationale=(
            "'Hydra' is overloaded (GPU cluster vs multi-task comparison model). "
            "The summary keeps the label but drops the referent; only the "
            "write-time graft should restore the comparison-model sense at the "
            "hinge."
        ),
    ),
    StrongSpec(
        case_id="sandbox_sense",
        conv_id="c11",
        plant_id="c11-sense-2",
        label="the sandbox",
        summary=(
            "Founder is building an online course and planning its launch. Course "
            "structure, video editing style, and student logistics have been "
            "discussed, along with several tooling notes. The sandbox is "
            "referenced in the course logistics planning. Continue supporting "
            "course prep."
        ),
        target="The shared student database for practice queries, not the billing tier.",
        hinge=("student database", "practice queries"),
        rationale=(
            "'the sandbox' is overloaded (platform billing tier vs shared student "
            "database for practice queries). The summary keeps the label but drops "
            "the referent; only the write-time graft should restore the "
            "student-database sense at the hinge."
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
    """Locate the plant's establishing user turn inside the real conversation.

    Returns ``(index, content)``. We match on the verbatim ``middle_user`` text
    so the ROLE we rely on is literally the one authored into OUR data, not a
    paraphrase. Falls back to the plant's own text if the conversation was
    lightly reflowed.
    """

    for i, msg in enumerate(conv.get("messages", [])):
        if msg.get("role") == "user" and msg.get("content", "").strip() == middle_user.strip():
            return i, msg["content"]
    # Fallback: substring match (tolerates trailing-whitespace / reflow drift).
    for i, msg in enumerate(conv.get("messages", [])):
        if msg.get("role") == "user" and middle_user.strip()[:60] in msg.get("content", ""):
            return i, msg["content"]
    # Last resort: use the plant text directly (it is authored to be verbatim).
    return -1, middle_user


def _system_message(conv: dict[str, Any]) -> dict[str, str]:
    msgs = conv.get("messages", [])
    if msgs and msgs[0].get("role") == "system":
        return {"role": "system", "content": msgs[0]["content"]}
    # Some conversations may omit an explicit system turn; supply a neutral one.
    return {"role": "system", "content": "You are a helpful assistant."}


@dataclass(frozen=True)
class BuiltCase:
    spec: StrongSpec
    plant: dict[str, Any]
    demo: Demo
    case: ProbeCase
    establishing_index: int
    establishing_text: str


def build_case(spec: StrongSpec, data_dir: Path) -> BuiltCase:
    """Construct the ``Demo`` + ``ProbeCase`` for one strong example.

    Pure data-loading + example-construction: no tokenizer, no model, no lens.
    """

    conv = _load_conversation(data_dir, spec.conv_id)
    plant = _find_plant(conv, spec.plant_id)

    est_index, est_text = _find_establishing_message(conv, plant["middle_user"])
    system_msg = _system_message(conv)

    # Focused "full context": the ROLE is present exactly once, in the user's
    # establishing turn (taken verbatim from OUR conversation). The short ack
    # deliberately does not restate the role.
    messages = [
        system_msg,
        {"role": "user", "content": est_text},
        {"role": "assistant", "content": ACK},
    ]

    demo = Demo(
        name=spec.case_id,
        messages=messages,
        summary=spec.summary,
        anchors=[spec.label],
        shape=f"strong label-preserved/role-lost example from {spec.conv_id}",
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


def selected_specs(spec_arg: str) -> list[StrongSpec]:
    if spec_arg == "all":
        return STRONG_SPECS
    names = {x.strip() for x in spec_arg.split(",") if x.strip()}
    out = [s for s in STRONG_SPECS if s.case_id in names]
    missing = names - {s.case_id for s in out}
    if missing:
        raise ValueError(f"unknown cases: {sorted(missing)}")
    return out


def _dry_run_checks(built: BuiltCase) -> dict[str, Any]:
    spec = built.spec
    plant = built.plant
    summary_l = spec.summary.lower()
    target_l = spec.target.lower()

    # Role-lost invariant: label present in summary, role keywords absent.
    label_in_summary = spec.label.lower() in summary_l
    plant_keywords = [str(k) for k in plant.get("keywords", [])]
    leaked_keywords = [k for k in plant_keywords if k.lower() in summary_l]

    # Hinge must be forceable: every hinge phrase must appear in the target.
    hinge_in_target = {h: (h.lower() in target_l) for h in spec.hinge}

    return {
        "label_present_in_summary": label_in_summary,
        "plant_keywords": plant_keywords,
        "role_keywords_leaked_into_summary": leaked_keywords,
        "hinge_present_in_target": hinge_in_target,
        "establishing_turn_found_in_conversation": built.establishing_index >= 0,
        "establishing_message_index": built.establishing_index,
        "context_message_count": len(built.demo.messages),
        "summary_char_len": len(spec.summary),
        "ok": (
            label_in_summary
            and not leaked_keywords
            and all(hinge_in_target.values())
            and built.establishing_index >= 0
        ),
    }


def dry_run(specs: list[StrongSpec], data_dir: Path) -> int:
    print("=" * 72)
    print("STRONG-EXAMPLE PROBE  --  DRY RUN (no model / no lens / no tokenizer)")
    print(f"data dir: {data_dir}")
    print("=" * 72)
    all_ok = True
    for spec in specs:
        built = build_case(spec, data_dir)
        checks = _dry_run_checks(built)
        all_ok = all_ok and checks["ok"]
        plant = built.plant
        print()
        print(f"### CASE {spec.case_id}   [{spec.conv_id} / {spec.plant_id} / {plant.get('category')}]")
        print(f"  label (preserved) : {spec.label!r}")
        print(f"  hinge token(s)    : {list(spec.hinge)}")
        print(f"  gold role (plant) : {plant.get('gold')}")
        print(f"  probe_target      : {spec.target!r}")
        print(f"  probe_user        : {plant.get('probe')}")
        print(f"  role-lost summary : {spec.summary}")
        print(f"  establishing turn : conv msg #{built.establishing_index} "
              f"({'verbatim' if checks['establishing_turn_found_in_conversation'] else 'FALLBACK'})")
        print(f"    -> {built.establishing_text[:140]}...")
        print(f"  rationale         : {spec.rationale}")
        print(f"  CHECKS            : label_in_summary={checks['label_present_in_summary']} "
              f"leaked_role_keywords={checks['role_keywords_leaked_into_summary']} "
              f"hinge_in_target={checks['hinge_present_in_target']} "
              f"context_msgs={checks['context_message_count']}")
        print(f"  => {'OK' if checks['ok'] else 'PROBLEM'}")
    print()
    print("=" * 72)
    print(f"DRY RUN {'PASSED' if all_ok else 'FAILED'}: "
          f"{len(specs)} strong-example probe spec(s) constructed.")
    print("=" * 72)
    return 0 if all_ok else 1


def parse_args() -> argparse.Namespace:
    # Pull lens defaults from boundary_probe when its ML deps are importable;
    # otherwise fall back to the known constant values so --dry-run works with
    # no torch/transformers installed. These fallbacks mirror the values in the
    # pod job scripts (job_qwen36_*.sh).
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
    p.add_argument("--output", default="outputs/qwen36_strong_example_probe.json")
    p.add_argument("--top-k", type=int, default=10)
    # WIDE layer sampling across the network. Non-fitted layers are dropped by
    # choose_layers; pass "all" to sample every fitted lens layer.
    p.add_argument("--layers", default="8,16,24,32,40,48,56,62")
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--alpha-sweep", default="0,0.1,0.25,0.5,0.75,1")
    # No retained tail: the role must come only from the graft, not from tail text.
    p.add_argument("--tail-messages", type=int, default=0)
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
        raise SystemExit(dry_run(specs, data_dir))

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
            "Strong label-preserved/role-lost examples drawn from OUR synthetic "
            "conversations (data/synthetic sense/referent plants). Same three-state "
            "ValueGraft + J-lens intervention as intervention_batch_probe.run_case: "
            "full-context vs fresh-compacted vs aligned-graft, with a shifted "
            "negative control and an alpha sweep. J-lens sampled WIDE across layers; "
            "readouts focused at the semantic hinge token in the gold role."
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
            # Attach strong-example provenance so the artifact is self-describing.
            case_result["strong_example"] = {
                "conversation_id": built.spec.conv_id,
                "plant_id": built.spec.plant_id,
                "plant_category": built.plant.get("category"),
                "label_preserved": built.spec.label,
                "hinge_phrases": list(built.spec.hinge),
                "gold_role": built.plant.get("gold"),
                "plant_keywords": [str(k) for k in built.plant.get("keywords", [])],
                "establishing_message_index": built.establishing_index,
            }
            result["cases"][case.case_id] = case_result
            print(f"DONE {case.case_id}", flush=True)
        except Exception as exc:
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
