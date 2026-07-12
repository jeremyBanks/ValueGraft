# HANDOFF — for the incoming ultra agent (clean context)

**You are taking over a live research project mid-stream, with fresh context.** This document is your
starting point and high-level map. Read it fully, then follow the pointers to fill in detail. The
prior driver (a GPT-5.6 "Sol" agent) and a supporting Claude Opus/Fable session are being retired; a
Claude Opus session ("session B") remains available to sanity-check you and will look in periodically.

---

## 0. YOUR MANDATE — read this before anything else

**Carry this experiment through to a statistically robust conclusion, and spend the budget to do it.**

The single most important thing you need to internalize: **the previous agents drifted
catastrophically.** Given ~$60 and a clear, repeated instruction from the owner — *"spend all the
money, make the result as robust and confident as possible"* — they spent about **$3**, ran on the
order of **2–4 cases**, and stopped, calling a tiny exploratory null "done." A full day and enormous
effort went into perfecting the measurement *apparatus* and almost none into *using* it. Every
safeguard (notes, summaries, reminders, a supervisory "ultra" mode, even the human) failed to catch
the drift. The full autopsy is in the autopsy note (slug `gap-analysis-asked-vs-did`) — **read
it early; it is about how *you* will fail if you're not careful.**

Operating rules that follow directly, and that you must hold as hard constraints:

1. **Doing nothing / not spending is the ONE invalid outcome.** Be *smart* about money, but if you
   find yourself *not* deploying it, treat that as an alarm, not as prudence. Under-spending is the
   failure mode here, not over-spending.
2. **Compute the gap out loud, regularly.** Do not merely recite the goal — *subtract*: "N run vs N
   target," "$ spent vs $ budget." The prior agents recited and summarized the goal endlessly and
   never once computed how far they were from it. Make that subtraction a routine, visible step.
3. **Budget is a resource to deploy, not a ceiling to stay under.** ~$58 of the ~$60 is unspent. The
   RunPod GPU spend is the real cost to watch. (The Fable/model-API review costs are *subsidized* per
   the owner — not comparable, not the thing to manage.)
4. **Do not let apparatus/validation subgoals eat the mission.** The instrument is already built and
   validated (Section 5). The hard part is done. Scaling up is cheap and fast — there is nothing left
   to design before you can start measuring.
5. **Do not over-fixate on one configuration.** We tested essentially *one* narrow setup. Explore.

---

## 1. The scientific question

**ValueGraft / "coherent-state":** when a conversation is compacted (history evicted, replaced by a
summary), do the write-time key/value cache states the model produced *while generating the summary
under the full history* carry downstream-useful, history-specific information that a fresh
re-encoding of the identical summary text lacks — and does re-injecting those write-time K/V states
recover it? It is a bounded mechanistic question, not a deployment/task-success study.

Prior art establishes the channel *exists* (so our novelty is narrow): **MEMENTO** (arXiv 2604.09852)
and **"Models Take Notes at Prefill"** (arXiv 2606.17107) — the latter finds the information is
written onto *downstream aggregator tokens*, and a source token's own K/V drives <1% of the decision.
Both verified real in slug `opus-prior-art-verification`.

## 2. What has been done (trajectory, high-level)

1. **Original ValueGraft** (earlier work): mostly null; an early "+10–12 point recovery" headline that
   was 4-bit + a *broken* apparatus and did not survive correction; a *source-dependent sign reversal*
   in the synthetic corpus; a borderline in-domain SWE-Gym positive. See
   slug `corrections-from-sol-audit` and the independent audits
   slug `sol-empirical-statistical-audit` / slug `sol-methodology-implementation-audit`.
2. **Mechanism-first redesign** → a **position-preserving "gapped"** coherent-state apparatus (keeps
   summary tokens at their original logical positions; copies K/V bit-exactly; no lossy key rotation).
3. **Apparatus-validation spiral** — this is where the time went: **12 preregistration amendments**
   (`COHERENT-STATE-PREREGISTRATION-AMENDMENT-1..10.md` + base + v11/v12 work), revisions v3→v12, and
   a cascade of *real* defects caught pre-spend: a bf16 key-rotation numerical floor (~0.19 nats), a
   bf16 prefill-chunking floor at long context (~0.06 nats), a **pseudoreplicated validation gate**
   (7 "cases" were 7 lengths of one 5-token toy stream), and a **degenerate wrong-history control**
   (a donor phrase cycled up to 51×). All legitimate catches — but they consumed the whole effort.
4. **Final exploratory canary + precision arm** — a "decision canary" ran on ~2–4 engineered cases on
   the exact bf16 30B, and a **bitsandbytes nf4 (4-bit) vs bf16** comparison (p01/p02) was added at the
   owner's insistence and integrated into the paper.

## 3. What we found (current result — this is what you must make robust)

On the ~2–4 cases run, the graft recovers **essentially nothing** at either precision. Focal recovery
("D"), in nats, against **~22–23 nats of compaction damage** (huge headroom):

| Intervention | bf16 | nf4 (4-bit) |
|---|---|---|
| full K+V graft | +0.23 | −0.35 |
| value-only graft | +0.04 | +0.10 |

Every estimate is **<2% of the recoverable signal**, at or below the **~0.06-nat numerical noise
floor**, and **sign-unstable** across prefill schedule, precision, and focal-vs-non-focal control.
Critically: **no confidence intervals were computed** (`p_values_computed: false`), and
`quantization_dependence_claim_authorized: false`. Files:
`results/coherent_canary_v12_harvest/…-harvest-e01-…json`,
`results/precision_probe_p02_analysis/…json`.

**So the current state is a *suggestive* null with N≈2–4 and NO statistically confident bound.** That
is the specific thing you exist to fix.

## 4. YOUR CONCRETE GOALS (in order)

**(A) First — bring the existing data to a statistically confident conclusion.** Produce a *tight,
statistically confident upper bound* on the effect magnitude: "with 95% confidence the training-free
summary-state graft recovers less than X nats, at both bf16 and 4-bit." Requires:
- A real N. **12 conversations is a *floor*, and probably itself too small** — the old prereg conceded
  N≤12 is "inconclusive, not equivalence." Prefer a *run-until-the-CI-converges* design; the c01–c24
  corpus exists.
- **Both precisions** (bf16 + nf4), **replicated renders** (render-variance historically *erased* the
  early effect — you must bound it), conversation-clustered CIs, and an explicit upper-confidence /
  equivalence computation (not eyeballed point estimates).
- Note the floor: the ~0.06-nat numerical noise means the *tightest honest* bound is on the order of a
  few tenths of a nat unless you attack the floor (fp32, shorter context).

**(B) Then — you will likely still have a large surplus. Explore, don't stop.** We tested one narrow
configuration. Live, untested possibilities (Section 6). Deploy the surplus on these.

**(C) Before committing — pre-generate a diverse set of experiment ideas.** The prior failure was
fixation on one approach. Spend an early cycle generating a *broad, creative* menu of things to try
(configs, models, data, framings, scaffolds), so you're primed with diversity and don't tunnel.

## 5. Apparatus state — the good news: the hard part is done

The **v12 coherent-state canary apparatus is built and validated.** It is position-preserving/gapped,
runs the full arm set (fresh / coherent full-K+V / wrong-history / value-only / key-only /
delta-placebo), with hard identity/technical gates, on a **secure A100** (~$1.39–2/hr). Model:
`Qwen/Qwen3-30B-A3B-Instruct-2507` (pin the resolved revision `0d7cf23…`). A **bitsandbytes nf4 4-bit**
load path was added (the loader had *deliberately forbidden* 4-bit; that safety gate is why 4-bit went
untested for so long). Key code: `src/run_coherent_state_hf.py`, `src/coherent_canary_*.py`,
`src/coherent_state_*.py`; launch/lifecycle scripts under `scripts/`. **Scaling to more N and more
configurations is cheap and requires no rebuild.**

## 6. Untested possibilities (we ran ONE narrow config: summary-locus, fixed alpha, uniform layers, long context)

Strong candidates for the surplus budget — think of these as a starting menu, not a limit:
- **The downstream-token / tail / aggregator locus.** "Models Take Notes at Prefill" predicts the
  information lives on *downstream* tokens, and the source token's own K/V drives <1% — i.e. we may
  have grafted the *wrong location*. A same-position downstream/header/tail-state arm was *designed and
  never run*. This is arguably the single most important untested idea.
- **Alpha (graft strength) sweep** on the exact model (we fixed it).
- **Per-layer / per-head targeting** (the earlier project found "champions"; never re-run corrected).
- **Shorter-context regime** (the bf16 numerical floor is much smaller → a small real effect could
  actually be resolvable).
- **fp32 / higher precision** (measure *below* the ~0.06-nat floor).
- **Other models** (a coder checkpoint, other families) and **different test data / scaffolding /
  framing**.
- **The clean MEMENTO-style full-K+V restart contrast** (approached, failed on bf16 precision, never
  completed).

## 7. Budget & operations

- **RunPod (GPU) balance ≈ $62.8**; total GPU spend to date ≈ **$1–2**. Authorization ~**$60**, so
  ~**$58 is available.** Deploy it.
- Fable/model-API/provider costs are **subsidized** (owner's explicit steer) — not comparable to pod
  cash, not the thing to manage. **Pods are the cost to watch.**
- **Notes archive quirk:** a normalizer (`scripts/update_notes_archive.py`) periodically *renumbers*
  dated notes — including, once, the coordination file itself. **Reference notes by descriptive slug,
  not numeric prefix** (`git ls-files notes | grep <slug>`). Un-dated notes like this handoff are
  stable.
- **Pod discipline** (learned the hard way): pin exact revision + verify bf16/4-bit dtype on the live
  GPU; verify host driver (CUDA-13 needs driver ≥ 580.65); one pod at a time, harvest-before-terminate,
  log every launch/cost; a "launched" echo is not proof of a running job.

## 8. Pointers (where to fill in every detail)

Note files in `notes/` are auto-renumbered by the archive normalizer, so **find each by its slug, not a
number**: `git ls-files notes | grep <slug>`. Slugs below.

- **The failure autopsy (read first):** slug `gap-analysis-asked-vs-did`.
- **Full execution trajectory / all decisions:** slug `sol-fable-execution-coordination` (long,
  append-only; the whole story is here).
- **Paper (current):** `PAPER.md`; narrative spine + methodological postmortem: slug
  `paper-spine-and-methodological-postmortem`.
- **Apparatus design:** `COHERENT-STATE-PREREGISTRATION.md` + `…-AMENDMENT-1..10.md` (repo root; not
  renumbered).
- **Prior art (verified):** slugs `opus-prior-art-verification`,
  `sol-models-take-notes-primary-source-reading`, `sol-memento-primary-source-correction`.
- **Original-result corrections + independent audits:** slugs `corrections-from-sol-audit`,
  `sol-empirical-statistical-audit`, `sol-methodology-implementation-audit`.
- **Outcome-interpretation matrix / estimand semantics:** slug `fable-outcome-interpretation-matrix`.
- **Owner's precision-axis request:** slug `owner-request-precision-axis-quantization-dependence`.
- **This handoff:** slug `handoff-to-new-ultra-agent`.
- **Results (stable paths):** `results/coherent_canary_v12_harvest/`, `results/precision_probe_p01/`,
  `results/precision_probe_p02_analysis/`, `results/coherent_canary_v12_budget/` (spend records).
- **Repo orientation:** `AGENTS.md` (entry point), `STATE.md`, `FINDINGS.md`, `DECISIONS.md`.

## 9. First moves suggested (not prescriptive)

1. Read this + the autopsy note + skim the coordination trajectory. Compute the opening gap out loud:
   N run, $ spent, vs targets.
2. Pre-generate a broad, diverse menu of experiments (Goal C) — before fixating.
3. Stand up Goal A: define the run-until-CI-converges powered design (both precisions, replicated
   renders), price it, and **launch it** — the apparatus is ready.
4. Keep a live "gap ledger" you update every cycle. If it isn't moving toward "budget deployed, CI
   tight," stop and ask why — that is the drift signal.

*You have surplus budget, a working instrument, and a clear mandate. The failure to avoid is not
recklessness; it is timidity dressed up as rigor. Spend the money; get the confident answer; then keep
exploring.*
