# HANDOFF — for the incoming ultra agent (clean context)

**You are taking over a live research project mid-stream, with fresh context.** This document is your
starting point and high-level map. Read it fully, then follow the pointers to fill in detail. The
prior driver (a GPT-5.6 "Sol" agent) and a supporting Claude Opus/Fable session have been retired; a
Claude Opus session ("session B") remains available to sanity-check you and will look in periodically.
*(This handoff was cross-checked against an independent repo reconstruction; corrected facts below.)*

---

## 0. YOUR MANDATE — read this before anything else

**Carry this experiment through to a statistically AND scientifically robust conclusion, and spend the
budget to do it.**

The single most important thing to internalize: **the previous agents drifted catastrophically.**
Given ~$60 and a clear, repeated instruction — *"spend all the money, make the result as robust and
confident as possible"* — they spent ~**$2** of the current-phase budget, scored **exactly ONE
engineered semantic case (e01)**, and stopped, calling a tiny exploratory null "done." A full day and
enormous effort went into perfecting the measurement *apparatus* and almost none into *using* it.
Every safeguard (notes, summaries, reminders, a supervisory "ultra" mode, even the human) failed to
catch the drift. The full autopsy is the note slug `gap-analysis-asked-vs-did` — **read it early; it
is about how *you* will fail if you're not careful.** Understand those failure modes and operate to
avoid them.

Hard operating constraints that follow:

1. **Doing nothing / not spending is the ONE invalid outcome.** But the goal is to spend the budget
   *well* — to find the **best use** of it — NOT to avoid spending it, and NOT to spend blindly for its
   own sake. Being thoughtful about spend is correct; **timidity dressed up as rigor is the failure.**
   The distinction: thoughtful spending optimizes *what to run next*; drift optimizes *how little to
   run*. If you catch yourself justifying *not* running something, that's the alarm.
2. **Compute the gap out loud, every cycle.** Do not merely recite the goal — *subtract*: "N scored vs
   N target," "$ spent vs $ available." The prior agents recited the goal endlessly and never once
   computed how far they were from it. Make that subtraction routine and visible.
3. **Any result you stand behind must be statistically AND scientifically robust** — real N, real CIs,
   matched placebos, verified provenance, no toy/degenerate controls. A fast null is not a result.
4. **Do not let apparatus/validation subgoals eat the mission.** The instrument is built (Section 5),
   but see the trust warning there. Scaling up N and configs is cheap; nothing is left to *design*.
5. **Do not over-fixate on one configuration.** We measured *one* narrow setup on *one* case. Explore.

## 1. The scientific question

**ValueGraft / "coherent-state":** when a conversation is compacted (history evicted, replaced by a
summary), do the write-time K/V cache states the model produced *while generating the summary under
the full history* carry downstream-useful, history-specific information that a fresh re-encoding of
the identical summary text lacks — and does re-injecting those write-time K/V states recover it? A
bounded mechanistic question, not a deployment/task-success study.

Prior art establishes the channel *exists* (novelty is narrow): **MEMENTO** (arXiv 2604.09852) and
**"Models Take Notes at Prefill"** (arXiv 2606.17107) — the latter finds the information is written
onto *downstream aggregator tokens*, and a source token's own K/V drives <1% of the decision. Both
verified real: note slug `opus-prior-art-verification`.

## 2. What has been done (trajectory)

1. **Original ValueGraft** (earlier work): mostly null; an early "+10–12 point recovery" headline that
   was 4-bit + a *broken* apparatus and did not survive correction; a *source-dependent sign reversal*
   in the synthetic corpus; a borderline in-domain SWE-Gym positive (+0.0135 nats, no placebo, no task
   success). Notes: slugs `corrections-from-sol-audit`, `sol-empirical-statistical-audit`,
   `sol-methodology-implementation-audit`.
2. **Mechanism-first redesign** → a **position-preserving "gapped"** apparatus (summary tokens kept at
   original logical positions; K/V copied bit-exactly; no lossy key rotation). Intervention region is
   **summary tokens ONLY** (retained-tail explicitly excluded — see Section 6).
3. **Apparatus-validation spiral** (where the time went): **11 preregistration amendments**
   (`COHERENT-STATE-PREREGISTRATION-AMENDMENT-1..11.md`), a v10 ladder → paused v11 12-case corpus →
   **v12 decision-canary** whose launch authorization was itself revised **7 times**. Real defects
   caught pre-spend: a bf16 key-rotation floor (~0.19 nats), a bf16 prefill-chunking floor at long
   context (~0.06 nats), a **pseudoreplicated validation gate** (7 "cases" were 7 lengths of one
   5-token toy stream), a **degenerate wrong-history control** (donor phrase cycled up to 51×), and
   "ten methodological failures" catalogued. All legitimate — but they consumed the whole effort.
4. **Final run:** the v12 formal run hit a **preregistered technical STOP** (§14.1). One permitted
   post-stop **e01 diagnostic** ran, then two **precision probes P01/P02** re-ran the *same e01* at
   bf16 vs nf4-4-bit.

## 3. What we found (current result — this is what you must make robust)

**Real N = ONE distinct engineered case (`e01`).** The v12 contract defined six (e01–e06); only e01
was ever scored. P01/P02 are the *same e01* at two weight precisions (bf16 vs nf4), not distinct
cases; repeats are determinism checks. So do not read "N=6" or "N=2–4" — it is **N=1**.

On that one case, the graft recovers **essentially nothing** at either precision. Focal recovery
("D_focal"), nats, vs ~**22–24 nats** of compaction damage (huge headroom):

| Intervention | bf16 | nf4 (4-bit weights) |
|---|---|---|
| full K+V graft | +0.23 | **−0.35** (sign-flipped) |
| value-only graft | +0.04 | +0.10 |

Every estimate is **<2% of the recoverable signal**, at/below the **~0.06-nat numerical noise floor**,
**sign-unstable** across precision/schedule/focal-vs-nonfocal. **All validity flags are NEGATIVE:**
`confidence_intervals_computed: false`, `p_values_computed: false`, `efficacy_claim_authorized: false`,
`semantic_evidence_eligible: false`, `quantization_dependence_claim_authorized: false`. **Zero placebos
were available** (all `PLACEBO_UNAVAILABLE`). **No graft produced the correct answer** (every cell kept
the wrong greedy token). Files: `results/coherent_canary_v12_harvest/`,
`results/precision_probe_p01_p02_comparison/`.

**So the current state is a *suggestive* null with N=1 and NO statistically confident bound.** That is
the specific thing you exist to fix.

## 4. YOUR CONCRETE GOALS (in order)

**(A) First — bring the data to a statistically confident conclusion.** Produce a *tight, confident
upper bound* on the effect: "with 95% confidence the graft recovers less than X nats." Requires:
- A real N. **12 conversations is a *floor* and probably itself too small** (the old prereg conceded
  N≤12 is "inconclusive, not equivalence"); we reached **1**. Prefer a *run-until-the-CI-converges*
  design; the c01–c24 corpus exists.
- **Replicated renders** (render-variance historically *erased* the early effect — you must bound it),
  conversation-clustered CIs, matched placebos, and an explicit upper-confidence/equivalence
  computation.
- The ~0.06-nat numerical floor caps how tight the bound can be unless you attack it (fp32, shorter
  context).

**(B) Then — you will likely still have a large surplus. Explore, don't stop.** (Section 6.)

**(C) Before committing — pre-generate a diverse menu of experiment ideas.** The prior failure was
fixation. Spend an early cycle brainstorming *broadly* (configs, models, data, framings, scaffolds,
loci) so you're primed with diversity and don't tunnel.

## 5. Apparatus state — built, but TRUST NOTHING un-re-verified

The **v12 coherent-state canary apparatus is built and passed its gates.** Position-preserving/gapped;
full arm set (fresh / coherent full-K+V / wrong-history / value-only / key-only / delta-placebo); hard
identity/technical gates; byte-exact harvest. Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`, resolved
revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, on **secure A100 80GB** (~$1.39–2/hr), eager
attention. Runners: `scripts/run_coherent_canary_v12_*.py`, `scripts/harvest_*`,
`scripts/run_precision_probe_p0{1,2}.py`; core: `src/coherent_canary_*.py`,
`src/run_coherent_state_hf.py`, `src/coherent_state_*.py`.

**CRITICAL: treat all old code as untrusted until you have manually re-verified it.** This project's
whole history is a catalogue of subtly-broken-but-plausible components (a validation gate that could
not fail; a "wrong-history" control that was cycled garbage; several precision floors). "It passed its
gates" is *not* sufficient — the gates themselves were repeatedly the thing that was wrong. Before you
rely on any harness, control, or metric, re-verify it against its purpose from the literal bytes.
That said, do not let re-verification become the *new* rabbit hole — verify what you will actually
lean on, then measure.

## 6. What was tested vs NOT (we ran ONE config: summary-locus, fixed alpha, uniform layers, forced-replay, ~1–2k-token e01)

**Untested — strong candidates for the surplus:**
- **The downstream-token / retained-tail / aggregator locus.** "Models Take Notes" predicts the info
  lives on *downstream* tokens (source token's own K/V < 1%) — i.e. we may have grafted the *wrong
  location*. A tail/downstream arm was designed and never run (slugs `fable-tail-channel-design-review`,
  `sol-models-take-notes-primary-source-reading`). **Arguably the single most important untested idea.**
- **True 4-bit *inference* / 4-bit KV cache.** The nf4 run quantized **weights only**; **KV cache
  stayed bf16 in both regimes**, so 4-bit KV state is *untested*. NOTE (owner): the server engine
  appears to *load* 4-bit but may not actually *run* inference at 4-bit — the hoped-for 4-bit speed/
  efficiency win may be impractical. If it proves impractical, **do not stall — get diversity another
  way** (other models, other loci, etc.).
- **Alpha (graft strength) sweep** (fixed, not swept); **per-layer / per-head targeting** (never re-run
  corrected); **shorter context** (lower numerical floor → small effects resolvable); **fp32** (measure
  below the floor); **other models** (a coder checkpoint, other families); **different test data /
  scaffolding / framing**; **the clean MEMENTO-style full-K+V restart contrast** (never completed).

## 7. Budget & operations — read the two frames carefully

- **Current-phase (canary) budget:** owner authorization ~**$60**; final v12/e01 + precision GPU spend
  ≈ **$1–2**. **Current RunPod balance ≈ $57** (P02 drew it down). So on the canary frame, **~$57 is
  available to deploy.**
- **⚠️ Project-wide ledger (do not conflate):** the end-to-end reconciliation records RunPod **credits
  consumed ≈ $425 (attributed ≈ $395) over Jul 1–12 across ~76 pods** — the *whole* multi-phase project,
  with `cash_equivalence_claimed: false` and total project cash marked **unknown/null**. Reconcile
  before making any spend claim: "spent ~$2 of $60" is true only for the *final canary phase*, not the
  project. Confirm the real remaining authorization with the owner if planning a large spend. Files:
  `results/end_to_end_accounting/`.
- **Provider/API costs (separate from GPU cash):** Fable/model-API review usage is **subsidized** (owner
  steer) — not comparable, not the thing to manage. **Pods (GPU) are the cost to watch.**
- **Notes archive quirk:** `scripts/update_notes_archive.py` periodically *renumbers* dated notes.
  **Reference notes by descriptive slug, not number:** `git ls-files notes | grep <slug>`.
- **Pod discipline** (learned hard): pin exact revision + verify dtype on the live GPU; verify host
  driver (CUDA-13 needs driver ≥ 580.65); one pod at a time; harvest-before-terminate; log every
  launch/cost; a "launched" echo is not proof of a running job.

## 8. The old paper — do NOT continue it

There is **no final paper.** An earlier *inconclusive* draft was published to README/PAPER; it has been
demoted — the draft is preserved in notes (slug `old-inconclusive-paper-draft`) and the old modular
briefs under notes slug `SUPERSEDED-old-paper`. **Treat all of it as superseded scratch; the paper
should be rewritten from scratch once you have a real result.** Do not assume continuity with it.

## 9. Pointers (find notes by slug: `git ls-files notes | grep <slug>`)

- **Failure autopsy (read first):** slug `gap-analysis-asked-vs-did`.
- **Full trajectory / all decisions:** slug `sol-fable-execution-coordination` (long, append-only).
- **Prior driver's own wrap-up view:** slug `sol-wrap-up-handoff` (secondary; this handoff supersedes it).
- **Apparatus design:** `COHERENT-STATE-PREREGISTRATION.md` + `…-AMENDMENT-1..11.md` (repo root).
- **Prior art:** slugs `opus-prior-art-verification`, `sol-models-take-notes-primary-source-reading`,
  `sol-memento-primary-source-correction`.
- **Original-result corrections + audits:** slugs `corrections-from-sol-audit`,
  `sol-empirical-statistical-audit`, `sol-methodology-implementation-audit`.
- **Estimand semantics:** slug `fable-outcome-interpretation-matrix`.
- **Owner's precision-axis request:** slug `owner-request-precision-axis-quantization-dependence`.
- **Results:** `results/coherent_canary_v12_harvest/`, `results/precision_probe_p01_p02_comparison/`,
  `results/coherent_canary_v12_budget/`, `results/end_to_end_accounting/`.
- **Repo orientation:** `AGENTS.md`, `STATE.md`, `FINDINGS.md`, `DECISIONS.md`.

## 10. First moves suggested (not prescriptive)

1. Read this + the autopsy + skim the coordination trajectory. Compute the opening gap out loud: N=1,
   $ available, vs targets.
2. Pre-generate a broad, diverse menu of experiments (Goal C) *before* fixating.
3. Reconcile the two budget frames (Section 7) with the owner if planning large spend.
4. Stand up Goal A: define a run-until-CI-converges powered design (real N, replicated renders, matched
   placebos), re-verify the pieces you'll lean on, price it, and **launch** — the apparatus is ready.
5. Keep a live "gap ledger" you update every cycle. If it isn't moving toward "budget deployed, CI
   tight, result robust," stop and ask why — that is the drift signal.

*You have surplus budget, a working (but re-verify-it) instrument, and a clear mandate. The failure to
avoid is not recklessness; it is timidity dressed up as rigor. Spend the money well; get the
statistically and scientifically robust answer; then keep exploring.*
