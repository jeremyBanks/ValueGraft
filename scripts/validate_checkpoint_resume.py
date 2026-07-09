#!/usr/bin/env python3
"""COMMITTED CPU validation of per-conversation checkpoint RESUME correctness
under the RELATIVE competence-floor mode, WITH an empty-alignment conversation in
the window (incident #38 fix; Fable review 07-09 rank-1/rank-2).

Why this exact case: the relative pre-scan computes lp_A for EVERY plant of EVERY
conv -- including an empty-ALIGNMENT conv (one whose A<->B graft alignment is
empty) -- but the main scoring loop returns EARLY on an empty-align conv, so it
contributes ZERO per_cat/task_excluded rows. If a checkpoint's scan_lpa were
derived from those rows it would be [] for that conv, so a RESUMED relative-mode
run would DROP its lp_A from the pooled median-k*MADN floor -> silently gate
task-competence differently -> change raw_EB and the conversation-clustered CI.
This is the one forbidden failure (a silent number change on resume). The fix
stores the pre-scan's OWN per-conv lp_A in the checkpoint; this script proves the
resumed floor + by_category_robust + traces are BYTE-IDENTICAL to uninterrupted,
and (negative control) that corrupting that stored lp_A to [] -- i.e. the OLD
derive-from-rows behavior -- WOULD have drifted the result.

Runs on CPU with the cached tiny real model Qwen/Qwen3-0.6B (a real Qwen3 chat
template + QK-norm), SC_BATCHED_RENDER=0 so a re-rendered conv is byte-identical
(this test isolates the FLOOR path, not batched-decode composition). No GPU.

Exit: 0 = PASS, 1 = FAIL (a real regression), 2 = SKIP (torch/model unavailable).

Usage:  SC_BATCHED_RENDER=0 uv run python scripts/validate_checkpoint_resume.py
"""
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("SC_BATCHED_RENDER", "0")   # byte-exact re-render
os.environ.setdefault("HF_HUB_OFFLINE", "1")      # cached weights only
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

MODEL = "Qwen/Qwen3-0.6B"
MARKER = "<<<EMPTY_ALIGN_TEST_CONV>>>"   # planted in one scenario's system prompt


def _skip(msg):
    print(f"SKIP: {msg}")
    sys.exit(2)


try:
    import torch  # noqa: F401
    import cross_arch_probe as P
except Exception as e:  # noqa: BLE001
    _skip(f"torch/cross_arch_probe unavailable: {type(e).__name__}: {e}")


def install_empty_align_hook():
    """Force EXACTLY the marked conversation to be empty-alignment in the SCORING
    loop (build_alignment -> []), while leaving the relative PRE-SCAN (which never
    calls build_alignment) to compute its lp_A normally. build_token_context runs
    once per conv immediately before build_alignment in the main loop, so we read
    the current conv's messages there and flip a flag the alignment hook honors."""
    orig_btc = P.build_token_context
    orig_ba = P.build_alignment
    state = {"marked": False}

    def btc(tok, family, msgs, summary_text, tsm):
        state["marked"] = any(MARKER in (m.get("content") or "") for m in msgs)
        return orig_btc(tok, family, msgs, summary_text, tsm)

    def ba(*a, **k):
        return [] if state["marked"] else orig_ba(*a, **k)

    P.build_token_context = btc
    P.build_alignment = ba


COMMON = dict(
    conv_limit=3, alpha_v=0.75, alpha0_tol=5e-3, change_tol=1e-3,
    max_gold_tok=80, trust_remote_code=True, conv_start=0,
    placebo_mode=None, alpha_sweep=False, seed=42, strong_prior=True,
    champion_scan=0, champion_regions=None, native_render=True,
    native_max_reply=24, native_temp=0.0, headroom_floor=0.3,
    task_lpa_floor=-8.0, task_competence_mode="relative",
    task_competence_k=3.0, ablate_qk_norm_flag=False, ablate_lambda_values=None,
)


def run(out_dir, scenarios, fresh=False):
    if fresh:
        os.environ["SC_CHECKPOINT_FRESH"] = "1"
    else:
        os.environ.pop("SC_CHECKPOINT_FRESH", None)
    return P.run_model(MODEL, REPO / "data", Path(out_dir), None,
                       scenarios=scenarios, **COMMON)


def floor_and_result(doc):
    """The competence FLOOR actually applied + the scientific blocks that a
    drifted floor would change."""
    return {
        "active_floor": (doc.get("gates") or {}).get("task_competence_active_floor"),
        "by_category_robust": doc.get("by_category_robust"),
        "raw_EB": doc.get("raw_EB"),
        "traces": doc.get("traces"),
        "n_task_excluded": (doc.get("gates") or {}).get("n_task_excluded"),
        "empty_alignment_convs": (doc.get("gates") or {}).get("empty_alignment_convs"),
    }


def canon(x):
    return json.dumps(x, sort_keys=True, default=str)


def show(tag, doc):
    fr = floor_and_result(doc)
    ref = (doc.get("by_category_robust") or {}).get("referent") or {}
    print(f"  [{tag}] status={doc.get('status')} floor={fr['active_floor']} "
          f"n_traces={len(doc.get('traces') or [])} "
          f"empty_align={fr['empty_alignment_convs']} "
          f"referent_raw_EB={ref.get('raw_EB_mean')} "
          f"referent_ci={ref.get('raw_EB_ci')}")


def main():
    install_empty_align_hook()
    scen_all = json.loads((REPO / "data" / "scenarios.json").read_text())
    scen = copy.deepcopy(scen_all[:3])
    # mark the FIRST conv as the empty-alignment one; it is KEPT (its checkpoint is
    # reused) on resume -- exactly the conv whose stored scan_lpa must be right.
    scen[0]["system"] = MARKER + "\n" + scen[0].get("system", "")
    empty_id = scen[0]["id"]

    work = Path(tempfile.mkdtemp(prefix="ckpt_resume_val_"))
    try:
        d = work / "run"
        print("== FRESH (uninterrupted), relative floor, conv0 empty-align ==")
        doc_fresh = run(d, scen, fresh=True)
        show("fresh", doc_fresh)
        ck_dir = P.checkpoint_dir(d, MODEL)
        assert doc_fresh.get("status") == "OK", \
            f"fresh must be OK, got {doc_fresh.get('status')}: {doc_fresh.get('reason')}"
        eac = (doc_fresh.get("gates") or {}).get("empty_alignment_convs") or []
        assert empty_id in eac, \
            f"test did not exercise the empty-align branch: empty_alignment_convs={eac}"
        assert doc_fresh.get("traces"), "expected scored traces from the 2 normal convs"
        # the empty-align conv's checkpoint must carry NON-empty scan_lpa (the fix);
        # if it were [] the resume floor would drift.
        empty_ck_path = P.checkpoint_path(d, MODEL, 0, empty_id)
        eck = json.loads(empty_ck_path.read_text())
        assert eck["payload"]["per_cat"] and all(
            not eck["payload"]["per_cat"][c] for c in eck["payload"]["per_cat"]), \
            "empty-align conv should contribute no scoring rows"
        assert len(eck["scan_lpa"]) > 0, \
            "FIX BROKEN: empty-align conv checkpoint scan_lpa is empty (floor will drift)"
        print(f"  empty-align conv {empty_id}: 0 scoring rows, "
              f"scan_lpa has {len(eck['scan_lpa'])} lp_A (pooled into the floor)")

        print("\n== RESUME (all convs checkpointed) == fresh? ==")
        doc_resume = run(d, scen, fresh=False)
        show("resume", doc_resume)
        ok1 = canon(floor_and_result(doc_fresh)) == canon(floor_and_result(doc_resume))
        print(f"  resume == fresh (floor + by_category_robust + traces): {ok1}")

        print("\n== KILL-SIM: delete a NORMAL conv's checkpoint, resume ==")
        # delete conv_002 (a normal conv); conv0 (empty-align) stays checkpointed
        # so its stored scan_lpa MUST be pooled correctly for the floor to match.
        victim = sorted(ck_dir.glob("conv_*.json"))[-1]
        victim.unlink()
        print(f"  deleted {victim.name}")
        doc_kill = run(d, scen, fresh=False)
        show("kill-sim", doc_kill)
        ok2 = canon(floor_and_result(doc_fresh)) == canon(floor_and_result(doc_kill))
        print(f"  kill-sim == fresh (floor + by_category_robust + traces): {ok2}")

        print("\n== NEGATIVE CONTROL: corrupt empty-align scan_lpa -> [] "
              "(the OLD derive-from-rows behavior) MUST drift ==")
        run(d, scen, fresh=True)                      # re-establish clean checkpoints
        ep = P.checkpoint_path(d, MODEL, 0, empty_id)
        bad = json.loads(ep.read_text())
        bad["scan_lpa"] = []                          # simulate the pre-fix bug
        ep.write_text(json.dumps(bad))
        doc_bad = run(d, scen, fresh=False)
        show("neg-control", doc_bad)
        drifted = canon(floor_and_result(doc_fresh)) != canon(floor_and_result(doc_bad))
        print(f"  neg-control DIFFERS from fresh (proves the bug is real + caught): "
              f"{drifted}  (floor {doc_bad.get('gates',{}).get('task_competence_active_floor')} "
              f"vs {doc_fresh.get('gates',{}).get('task_competence_active_floor')})")

        print("\n" + "=" * 60)
        all_ok = ok1 and ok2 and drifted
        if all_ok:
            print("PASS: resume==fresh WITH an empty-align conv under relative floor;"
                  " the negative control confirms the fix is load-bearing.")
            return 0
        print("FAIL: resume drifted from fresh (or the negative control did not "
              "drift) -- checkpoint resume is NOT byte-identical.")
        return 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
