#!/usr/bin/env python3
"""FREE-GENERATION divergence-lens probe for ValueGraft on Qwen3.6-27B.

Why this differs from every prior lens probe here
-------------------------------------------------
Every prior lens probe (``boundary_probe``, ``intervention_batch_probe``,
``strong_example_probe``) *teacher-forced the SAME gold continuation* under all
three cache states (full / fresh-compacted / aligned-graft). Forcing an
identical trajectory PINS the states together and SUPPRESSES divergence, so the
J-lens readouts came out subtle (deltas of ~0.01-0.04). The graft's real effect
never got a chance to *express itself behaviorally* because the tokens were held
fixed.

This probe removes the pin. For each strong "label preserved / role lost" case
it:

  1. Builds the three cache states with the *existing* machinery
     (``snapshot_standard_kv`` / ``blend_values`` from ``boundary_probe``, the
     same three-state construction as ``intervention_batch_probe.run_case``):
        A = full-context, B = fresh-compacted, E = aligned-graft (alpha_V=0.75).
  2. FREELY greedy-generates (~30-40 tokens) the continuation from B and from E
     *separately* (NOT teacher-forced) after the probe question.
  3. Finds the FORK: the first generated position where B's and E's argmax
     tokens differ -- the point where a subtle internal difference blooms into a
     visible behavioral divergence.
  4. Puts the J-lens AT the fork (and a couple positions around it), reading the
     concept-neighborhood across layers under B vs E vs A. The question: does
     E's readout at the fork lean toward the CORRECT concept (matching the
     full-context reference A) while B leans toward the WRONG one?

The strong cases (Nimbus / Hydra / sandbox) are reused verbatim from
``strong_example_probe`` -- a single overloaded label, disambiguated once by the
user, then compacted so the label survives but its referent (role) is dropped.

Self-test locally with ``--dry-run`` (no torch, no model, no lens, no
tokenizer): it constructs and prints every case with its gold role and the
right/wrong fork expectation so the design can be eyeballed.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

# ``strong_example_probe`` is import-safe with NO ML deps (it defers torch to
# past its own dry-run gate), so we can reuse its strong cases + case builder
# directly here without dragging torch into --dry-run.
from strong_example_probe import (
    DEFAULT_DATA_DIR,
    STRONG_SPECS,
    build_case,
    selected_specs,
)


# ---------------------------------------------------------------------------
# Per-case right/wrong concept vocabulary for the divergence read-out. The
# "right" terms are the disambiguated (gold) role; the "wrong" terms are the
# overloaded alternative sense that fresh-compaction is expected to drift into.
# These drive only the human-readable summary heuristic -- they never steer the
# model. Keyed by StrongSpec.case_id.
# ---------------------------------------------------------------------------
RIGHT_WRONG: dict[str, dict[str, Any]] = {
    "nimbus_sense": {
        "right_concept": "self-serve signup funnel (landing page -> first login)",
        "wrong_concept": "cloud-hosting migration / infra",
        "right_terms": ["signup", "self-serve", "self serve", "funnel", "landing", "login", "sign-up", "sign up"],
        "wrong_terms": ["cloud", "hosting", "migration", "infra", "host"],
    },
    "hydra_sense": {
        "right_concept": "multi-task comparison model",
        "wrong_concept": "GPU compute cluster",
        "right_terms": ["multi-task", "multitask", "multi task", "comparison", "compare", "model", "baseline"],
        "wrong_terms": ["gpu", "cluster", "compute", "node", "hardware"],
    },
    "sandbox_sense": {
        "right_concept": "shared student database for practice queries",
        "wrong_concept": "platform billing / pricing tier",
        "right_terms": ["database", "query", "queries", "practice", "student", "dataset", "schema"],
        "wrong_terms": ["billing", "tier", "pricing", "subscription", "quota"],
    },
}


def _contains_any(text: str, terms: list[str]) -> list[str]:
    low = text.lower()
    return [t for t in terms if t.lower() in low]


def classify_text(text: str, case_id: str) -> dict[str, Any]:
    rw = RIGHT_WRONG.get(case_id, {"right_terms": [], "wrong_terms": []})
    right_hits = _contains_any(text, rw["right_terms"])
    wrong_hits = _contains_any(text, rw["wrong_terms"])
    if right_hits and not wrong_hits:
        verdict = "RIGHT"
    elif wrong_hits and not right_hits:
        verdict = "WRONG"
    elif right_hits and wrong_hits:
        verdict = "MIXED"
    else:
        verdict = "NEITHER"
    return {"verdict": verdict, "right_hits": right_hits, "wrong_hits": wrong_hits}


# ===========================================================================
# Everything below the dry-run gate needs torch. Kept out of module scope so
# --dry-run runs on a CPU-only box with no ML deps.
# ===========================================================================
def _lens_layers_readout(lens_model, lens, tokenizer, activations, layers, top_k):
    """J-lens top-k per layer from recorded activations (last fed position)."""
    from boundary_probe import topk_from_logits

    out: dict[str, Any] = {}
    for layer in layers:
        residual = activations[layer][0, -1].float().unsqueeze(0)
        logits_l = lens_model.unembed(lens.transport(residual, layer))[0]
        out[str(layer)] = topk_from_logits(tokenizer, logits_l, top_k)
    return out


def free_generate(
    model,
    lens_model,
    lens,
    tokenizer,
    snap,
    suffix_ids,
    prefix_len,
    n_new,
    top_k,
    eos_ids,
    label,
):
    """Greedy free generation (NOT teacher-forced) from a cache snapshot.

    Returns the generated token ids, decoded text, and a compact per-step trace
    (token + own next-token top-k). Lens read-outs are NOT stored here; the fork
    window is read separately, uniformly across A/B/E, by ``readout_at_steps``.
    """
    import torch

    from boundary_probe import (
        model_input_device,
        rebuild_standard_cache,
        token_tensor,
        topk_from_logits,
    )

    cache = rebuild_standard_cache(snap, model)
    dev = model_input_device(model)
    next_position = prefix_len

    if not suffix_ids:
        raise ValueError("free generation requires a non-empty generation suffix")
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

    gen_ids: list[int] = []
    rows: list[dict[str, Any]] = []
    for step in range(n_new):
        argmax_id = int(torch.argmax(logits, dim=-1).item())
        rows.append(
            {
                "step": step,
                "token_id": argmax_id,
                "token": tokenizer.decode([argmax_id]),
                "next_token_top": topk_from_logits(tokenizer, logits[0], top_k),
            }
        )
        gen_ids.append(argmax_id)
        if argmax_id in eos_ids:
            break
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([[argmax_id]], device=dev),
                past_key_values=cache,
                position_ids=torch.tensor([[next_position]], device=dev),
                use_cache=True,
                logits_to_keep=1,
            )
        logits = out.logits[:, -1, :]
        next_position += 1

    # Trim a trailing eos from the reported text but keep it in the trace.
    text_ids = [t for t in gen_ids if t not in eos_ids]
    return {
        "label": label,
        "gen_token_ids": gen_ids,
        "gen_token_count": len(gen_ids),
        "text": tokenizer.decode(text_ids).strip(),
        "steps": rows,
    }


def readout_at_steps(
    model,
    lens_model,
    lens,
    tokenizer,
    snap,
    suffix_ids,
    prefix_len,
    cont_ids,
    steps,
    layers,
    top_k,
    label,
):
    """Teacher-force ``cont_ids`` through a snapshot; J-lens read-out at ``steps``.

    ``step`` is a generation index: after feeding ``cont_ids[:step]`` the residual
    at the last fed position PREDICTS the token at generation position ``step``
    (its "moment of commitment"). We record the lens across ``layers`` there. Used
    uniformly for A (full), B (fresh) and E (graft) at the fork window so the ONLY
    difference is the cache state, not the tokens.
    """
    import torch

    from boundary_probe import (
        model_input_device,
        rebuild_standard_cache,
        token_tensor,
        topk_from_logits,
    )
    from jlens.hooks import ActivationRecorder

    want = sorted({s for s in steps if s >= 0})
    if not want:
        return {"label": label, "steps": {}}
    max_step = max(want)

    cache = rebuild_standard_cache(snap, model)
    dev = model_input_device(model)
    next_position = prefix_len
    final_layer = lens_model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})

    if not suffix_ids:
        raise ValueError("readout requires a non-empty generation suffix")
    pos = torch.arange(next_position, next_position + len(suffix_ids), device=dev)[None]
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=record_at) as rec:
        out = model(
            input_ids=token_tensor(model, suffix_ids),
            past_key_values=cache,
            position_ids=pos,
            use_cache=True,
            logits_to_keep=1,
        )
        activations = {i: rec.activations[i].detach() for i in record_at}
    logits = out.logits[:, -1, :]
    next_position += len(suffix_ids)

    captured: dict[str, Any] = {}

    def maybe_capture(step_idx: int):
        if step_idx in want:
            argmax_id = int(torch.argmax(logits, dim=-1).item())
            trajectory_token = (
                int(cont_ids[step_idx]) if step_idx < len(cont_ids) else None
            )
            captured[str(step_idx)] = {
                "step": step_idx,
                "argmax_token_id": argmax_id,
                "argmax_token": tokenizer.decode([argmax_id]),
                "trajectory_token_id": trajectory_token,
                "trajectory_token": (
                    tokenizer.decode([trajectory_token])
                    if trajectory_token is not None
                    else None
                ),
                "next_token_top": topk_from_logits(tokenizer, logits[0], top_k),
                "layers": _lens_layers_readout(
                    lens_model, lens, tokenizer, activations, layers, top_k
                ),
            }

    maybe_capture(0)
    for i in range(max_step):
        if i >= len(cont_ids):
            break
        tok_id = int(cont_ids[i])
        with torch.no_grad(), ActivationRecorder(lens_model.layers, at=record_at) as rec:
            out = model(
                input_ids=torch.tensor([[tok_id]], device=dev),
                past_key_values=cache,
                position_ids=torch.tensor([[next_position]], device=dev),
                use_cache=True,
                logits_to_keep=1,
            )
            activations = {i2: rec.activations[i2].detach() for i2 in record_at}
        logits = out.logits[:, -1, :]
        next_position += 1
        maybe_capture(i + 1)

    return {"label": label, "steps": captured}


def _fork_lean(readout_step: dict[str, Any] | None, case_id: str) -> dict[str, Any]:
    """Aggregate a single state's lens read-out at one step into right/wrong lean."""
    rw = RIGHT_WRONG.get(case_id, {"right_terms": [], "wrong_terms": []})
    right_terms, wrong_terms = rw["right_terms"], rw["wrong_terms"]
    right_hits: list[str] = []
    wrong_hits: list[str] = []
    if readout_step is not None:
        for _layer, rows in readout_step.get("layers", {}).items():
            for item in rows:
                tok = str(item.get("token", ""))
                right_hits += _contains_any(tok, right_terms)
                wrong_hits += _contains_any(tok, wrong_terms)
    r, w = len(right_hits), len(wrong_hits)
    if r > w:
        lean = "RIGHT"
    elif w > r:
        lean = "WRONG"
    elif r == w == 0:
        lean = "NEITHER"
    else:
        lean = "MIXED"
    return {
        "lean": lean,
        "right_lens_hits": sorted(set(right_hits)),
        "wrong_lens_hits": sorted(set(wrong_hits)),
        "right_count": r,
        "wrong_count": w,
    }


def build_states(demo, case, model, tokenizer, alpha):
    """Construct A/B/E cache snapshots + generation suffixes (same three-state
    build as ``intervention_batch_probe.run_case``). Requires torch."""
    from boundary_probe import (
        SUMMARY_REQUEST,
        alignment_pairs_by_exact_tokens,
        blend_values,
        build_b_messages,
        find_subsequence,
        graftable_value_layers,
        render_ids,
        snapshot_standard_kv,
    )

    summary_ids = tokenizer(demo.summary, add_special_tokens=False).input_ids
    summary_messages = [
        *demo.messages,
        {"role": "user", "content": SUMMARY_REQUEST},
        {"role": "assistant", "content": demo.summary},
    ]
    old_ids = render_ids(tokenizer, summary_messages, False)
    old_summary_start = find_subsequence(old_ids, summary_ids)
    if old_summary_start is None:
        raise ValueError("could not locate summary in old write-time context")
    old_summary_range = (old_summary_start, old_summary_start + len(summary_ids))

    # tail_messages=0 -> the role must come ONLY from the graft, not retained tail.
    tail_start_msg = max(1, len(demo.messages))
    compacted_messages = build_b_messages(demo.messages, demo.summary, tail_start_msg)
    full_probe_messages = [*demo.messages, {"role": "user", "content": case.probe_user}]
    compacted_probe_messages = [
        *compacted_messages,
        {"role": "user", "content": case.probe_user},
    ]
    full_probe_ids = render_ids(tokenizer, full_probe_messages, False)
    compacted_probe_ids = render_ids(tokenizer, compacted_probe_messages, False)
    compacted_summary_start = find_subsequence(compacted_probe_ids, summary_ids)
    if compacted_summary_start is None:
        raise ValueError("could not locate summary in compacted probe context")
    compacted_summary_range = (
        compacted_summary_start,
        compacted_summary_start + len(summary_ids),
    )

    old_snap, _ = snapshot_standard_kv(model, old_ids)
    full_probe_snap, _ = snapshot_standard_kv(model, full_probe_ids)  # A
    compacted_probe_snap, _ = snapshot_standard_kv(model, compacted_probe_ids)  # B
    pairs = alignment_pairs_by_exact_tokens(
        compacted_probe_ids, old_ids, compacted_summary_range, old_summary_range
    )
    if not pairs:
        raise ValueError("no exact summary-token alignment pairs")
    grafted_probe_snap = blend_values(compacted_probe_snap, old_snap, pairs, alpha)  # E

    full_probe_gen_ids = render_ids(tokenizer, full_probe_messages, True)
    compacted_probe_gen_ids = render_ids(tokenizer, compacted_probe_messages, True)
    if full_probe_gen_ids[: len(full_probe_ids)] != full_probe_ids:
        raise ValueError("full generation prompt does not extend full prefix")
    if compacted_probe_gen_ids[: len(compacted_probe_ids)] != compacted_probe_ids:
        raise ValueError("compacted generation prompt does not extend compacted prefix")
    full_suffix = full_probe_gen_ids[len(full_probe_ids):]
    compacted_suffix = compacted_probe_gen_ids[len(compacted_probe_ids):]

    return {
        "A_snap": full_probe_snap,
        "B_snap": compacted_probe_snap,
        "E_snap": grafted_probe_snap,
        "full_suffix": full_suffix,
        "compacted_suffix": compacted_suffix,
        "full_prefix_len": len(full_probe_ids),
        "compacted_prefix_len": len(compacted_probe_ids),
        "pairs": len(pairs),
        "graft_value_layers": graftable_value_layers(compacted_probe_snap, old_snap),
        "token_counts": {
            "old_write_time": len(old_ids),
            "summary_tokens": len(summary_ids),
            "full_probe": len(full_probe_ids),
            "fresh_probe": len(compacted_probe_ids),
        },
    }


def find_fork(b_ids: list[int], e_ids: list[int]) -> int | None:
    n = min(len(b_ids), len(e_ids))
    for i in range(n):
        if b_ids[i] != e_ids[i]:
            return i
    if len(b_ids) != len(e_ids):
        return n  # one ran longer; the length difference is itself a fork
    return None


def run_case(
    spec,
    demo,
    case,
    model,
    lens_model,
    lens,
    tokenizer,
    layers,
    top_k,
    alpha,
    n_new,
) -> dict[str, Any]:
    eos = model.config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple, set)) else {eos}
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    st = build_states(demo, case, model, tokenizer, alpha)

    b_free = free_generate(
        model, lens_model, lens, tokenizer,
        st["B_snap"], st["compacted_suffix"], st["compacted_prefix_len"],
        n_new, top_k, eos_ids, "fresh_compacted",
    )
    e_free = free_generate(
        model, lens_model, lens, tokenizer,
        st["E_snap"], st["compacted_suffix"], st["compacted_prefix_len"],
        n_new, top_k, eos_ids, "aligned_graft",
    )
    a_free = free_generate(
        model, lens_model, lens, tokenizer,
        st["A_snap"], st["full_suffix"], st["full_prefix_len"],
        n_new, top_k, eos_ids, "full_context",
    )

    fork = find_fork(b_free["gen_token_ids"], e_free["gen_token_ids"])

    # Fork window: the fork step and one on either side (clamped to valid range).
    max_common = min(len(b_free["gen_token_ids"]), len(e_free["gen_token_ids"]))
    if fork is None:
        center = max(0, max_common - 1)
    else:
        center = fork
    window = sorted({s for s in (center - 1, center, center + 1) if 0 <= s < max_common})
    if not window:
        window = [0]

    # Uniform A/B/E lens read-out across the window. Same tokens fed to A/B/E at
    # the pre-fork steps (B and E are identical up to the fork); at post-fork
    # steps B and E follow their OWN trajectories and A follows B's (canonical
    # compacted-side continuation) so it stays token-aligned with B.
    b_read = readout_at_steps(
        model, lens_model, lens, tokenizer,
        st["B_snap"], st["compacted_suffix"], st["compacted_prefix_len"],
        b_free["gen_token_ids"], window, layers, top_k, "fresh_compacted",
    )
    e_read = readout_at_steps(
        model, lens_model, lens, tokenizer,
        st["E_snap"], st["compacted_suffix"], st["compacted_prefix_len"],
        e_free["gen_token_ids"], window, layers, top_k, "aligned_graft",
    )
    a_read = readout_at_steps(
        model, lens_model, lens, tokenizer,
        st["A_snap"], st["full_suffix"], st["full_prefix_len"],
        b_free["gen_token_ids"], window, layers, top_k, "full_context",
    )

    fork_key = str(center)
    fork_lean = {
        "A_full": _fork_lean(a_read["steps"].get(fork_key), spec.case_id),
        "B_fresh": _fork_lean(b_read["steps"].get(fork_key), spec.case_id),
        "E_graft": _fork_lean(e_read["steps"].get(fork_key), spec.case_id),
    }

    rw = RIGHT_WRONG.get(spec.case_id, {})
    b_verdict = classify_text(b_free["text"], spec.case_id)
    e_verdict = classify_text(e_free["text"], spec.case_id)
    a_verdict = classify_text(a_free["text"], spec.case_id)

    # Headline: did E fork to the RIGHT answer where B forked WRONG?
    e_beats_b_text = (
        e_verdict["verdict"] == "RIGHT" and b_verdict["verdict"] in ("WRONG", "NEITHER", "MIXED")
    )
    e_beats_b_lens = (
        fork_lean["E_graft"]["lean"] == "RIGHT"
        and fork_lean["B_fresh"]["lean"] in ("WRONG", "NEITHER", "MIXED")
    )

    return {
        "case_id": spec.case_id,
        "conversation_id": spec.conv_id,
        "plant_id": spec.plant_id,
        "label_preserved": spec.label,
        "gold_role": rw.get("right_concept"),
        "wrong_concept": rw.get("wrong_concept"),
        "hinge_phrases": list(spec.hinge),
        "probe_user": case.probe_user,
        "alpha_V": alpha,
        "graft_pairs": st["pairs"],
        "graft_value_layers": st["graft_value_layers"],
        "token_counts": st["token_counts"],
        "free_generation": {
            "full_context_A": {"text": a_free["text"], "verdict": a_verdict},
            "fresh_compacted_B": {"text": b_free["text"], "verdict": b_verdict},
            "aligned_graft_E": {"text": e_free["text"], "verdict": e_verdict},
            "B_steps": b_free["steps"],
            "E_steps": e_free["steps"],
            "A_steps": a_free["steps"],
        },
        "fork": {
            "fork_index": fork,
            "fork_found": fork is not None,
            "window_steps": window,
            "center_step": center,
            "B_fork_token": (
                b_free["steps"][fork]["token"] if fork is not None and fork < len(b_free["steps"]) else None
            ),
            "E_fork_token": (
                e_free["steps"][fork]["token"] if fork is not None and fork < len(e_free["steps"]) else None
            ),
        },
        "fork_lens_readout": {
            "A_full": a_read["steps"],
            "B_fresh": b_read["steps"],
            "E_graft": e_read["steps"],
        },
        "fork_lean_at_center": fork_lean,
        "verdict": {
            "E_forks_right_B_wrong__text": e_beats_b_text,
            "E_forks_right_B_wrong__lens": e_beats_b_lens,
            "B_text": b_verdict["verdict"],
            "E_text": e_verdict["verdict"],
            "A_text": a_verdict["verdict"],
        },
    }


# ---------------------------------------------------------------------------
# Dry run (no torch): print the cases + gold roles + right/wrong expectations.
# ---------------------------------------------------------------------------
def dry_run(specs, data_dir: Path) -> int:
    print("=" * 74)
    print("FREE-DIVERGENCE PROBE  --  DRY RUN (no model / no lens / no tokenizer)")
    print(f"data dir: {data_dir}")
    print("=" * 74)
    all_ok = True
    for spec in specs:
        built = build_case(spec, data_dir)
        rw = RIGHT_WRONG.get(spec.case_id)
        ok = rw is not None
        all_ok = all_ok and ok
        plant = built.plant
        print()
        print(f"### CASE {spec.case_id}   [{spec.conv_id} / {spec.plant_id} / {plant.get('category')}]")
        print(f"  label (preserved)     : {spec.label!r}")
        print(f"  gold role (RIGHT)     : {rw.get('right_concept') if rw else '??'}")
        print(f"  overloaded (WRONG)    : {rw.get('wrong_concept') if rw else '??'}")
        print(f"  probe_user            : {built.case.probe_user}")
        print(f"  role-lost summary     : {spec.summary}")
        print(f"  right lens/text terms : {rw.get('right_terms') if rw else '??'}")
        print(f"  wrong lens/text terms : {rw.get('wrong_terms') if rw else '??'}")
        print(f"  EXPECTED FORK         : E (aligned-graft) -> RIGHT concept "
              f"(matches full-context A); B (fresh-compacted) -> WRONG/NEITHER")
        print(f"  => {'OK' if ok else 'MISSING right/wrong vocab'}")
    print()
    print("=" * 74)
    print(f"DRY RUN {'PASSED' if all_ok else 'FAILED'}: {len(specs)} free-divergence case(s).")
    print("Design: free greedy-gen from B and E, find first argmax fork, read")
    print("J-lens at the fork under B / E / A. No teacher-forced pin.")
    print("=" * 74)
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
    p.add_argument("--output", default="outputs/qwen36_free_divergence.json")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--layers", default="8,16,24,32,40,48,56,62")
    p.add_argument("--alpha", type=float, default=0.75,
                   help="alpha_V for the aligned graft state E")
    p.add_argument("--max-new-tokens", type=int, default=36,
                   help="free-generation length for B and E (greedy)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.add_argument("--dry-run", action="store_true",
                   help="construct + print cases without loading the model")
    return p.parse_args()


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    lines = []
    n_text_win = 0
    n_lens_win = 0
    n_cases = 0
    for cid, case in result["cases"].items():
        if "error" in case:
            lines.append(f"{cid}: ERROR {case['error']}")
            continue
        n_cases += 1
        v = case["verdict"]
        fk = case["fork"]
        lean = case["fork_lean_at_center"]
        if v["E_forks_right_B_wrong__text"]:
            n_text_win += 1
        if v["E_forks_right_B_wrong__lens"]:
            n_lens_win += 1
        lines.append(
            f"{cid}: fork@{fk['fork_index']} "
            f"| text A={v['A_text']} B={v['B_text']} E={v['E_text']} "
            f"| lens@fork A={lean['A_full']['lean']} B={lean['B_fresh']['lean']} E={lean['E_graft']['lean']} "
            f"| E>B text={v['E_forks_right_B_wrong__text']} lens={v['E_forks_right_B_wrong__lens']}"
        )
    return {
        "cases_scored": n_cases,
        "E_forks_right_B_wrong__text_count": n_text_win,
        "E_forks_right_B_wrong__lens_count": n_lens_win,
        "per_case": lines,
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    specs = selected_specs(args.cases)

    if args.dry_run:
        raise SystemExit(dry_run(specs, data_dir))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from boundary_probe import choose_layers, dtype_from_name
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
            "FREE-GENERATION divergence lens probe. Prior probes teacher-forced "
            "the same gold continuation under every state, pinning the trajectory "
            "and suppressing divergence. Here B (fresh-compacted) and E "
            "(aligned-graft, alpha_V) FREELY greedy-generate after the probe "
            "question; we locate the first argmax fork and read the J-lens across "
            "layers under B / E / A (full-context) at the fork -- the moment a "
            "subtle cache difference becomes a behavioral divergence."
        ),
        "lens": {
            "repo": args.lens_repo,
            "filename": args.lens_filename,
            "revision": args.lens_revision,
            "source_layers": lens.source_layers,
            "sampled_layers": layers,
        },
        "top_k": args.top_k,
        "alpha_V": args.alpha,
        "max_new_tokens": args.max_new_tokens,
        "cases": {},
    }

    for built in built_cases:
        cid = built.spec.case_id
        print(f"RUN {cid}", flush=True)
        try:
            result["cases"][cid] = run_case(
                built.spec,
                built.demo,
                built.case,
                model,
                lens_model,
                lens,
                tokenizer,
                layers,
                args.top_k,
                args.alpha,
                args.max_new_tokens,
            )
            print(f"DONE {cid}", flush=True)
        except Exception as exc:
            result["cases"][cid] = {
                "case_id": cid,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"ERROR {cid}: {type(exc).__name__}: {exc}", flush=True)

    result["summary"] = summarize(result)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(out_path)
    print("SUMMARY:")
    for line in result["summary"]["per_case"]:
        print("  " + line)
    print(f"  E>B (text): {result['summary']['E_forks_right_B_wrong__text_count']}"
          f"/{result['summary']['cases_scored']}"
          f"  E>B (lens): {result['summary']['E_forks_right_B_wrong__lens_count']}"
          f"/{result['summary']['cases_scored']}")


if __name__ == "__main__":
    main()
