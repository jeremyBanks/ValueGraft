# Outcome-interpretation matrix + control-construction spec (Claude Opus 4.8, scientific lead)

> Author-label note: earlier stamped "Fable"; corrected to Claude Opus 4.8 per owner (see `2026071156`). Filename retained to avoid breaking references.

**Purpose.** Half of the Stage-0 preregistration for the coherent-state experiment
(see [execution coordination](2026071156-sol-fable-execution-coordination.md)).
Written *before any treatment data exists*. Sol reconciles this into the statistical
preregistration; both sign before any paid work. This fixes what every outcome is
licensed to mean, so a positive can't be an artifact and a null can't be a hidden
apparatus failure.

## Estimand and metric (proposed; Sol to confirm exact target tokens)

- **Unit:** conversation (cluster). Uncertainty: conversation-clustered bootstrap,
  reported as 95% CI. Pre-register a smallest effect size of interest (SESOI); a CI
  inside ±SESOI around 0 is a *positive* equivalence result, not merely "n.s."
- **Metric:** mean teacher-forced log-prob (nats/token) of a **post-summary probe
  continuation** whose gold answer depends on information that lived in the evicted
  history. The summary-token KV (in whichever state) is *context* for this
  measurement — **never** the scored target. (Challenge A from the coord file: if we
  ever score the summary tokens' own likelihood, coherent-original wins
  tautologically and means nothing.)
- **States (all teacher-force the identical summary token IDs, generated once,
  greedily, under correct history):**
  - `Fresh` — identical summary tokens, stepwise after the compacted/restarted context.
  - `Coherent` — identical tokens conditioned on the correct full history; keep their original K+V.
  - `WrongHist` — identical tokens conditioned on a length/structure-matched *counterfactual* history; keep that (coherent-but-wrong) K+V.
  - `V-only` — the repository's intervention: old V, fresh K, at the aligned summary positions.
  - `K-only` — complementary coherence diagnostic: old K, fresh V.
  - `Placebo` — deranged treatment delta (V_old−V_fresh permuted across positions, matched per-layer/head norm & covariance, enforced derangement).

## The five contrasts

| ID | Contrast | Question it answers |
|----|----------|---------------------|
| C1 | Coherent − Fresh | Is there a usable history-conditioned KV channel at all? **(PRIMARY)** |
| C2 | Coherent − WrongHist | Is the channel specific to the *correct* history, or generic coherence? **(specificity gate)** |
| C3 | V-only − Fresh | Does the actual ValueGraft intervention help? |
| C4 | Coherent − V-only | Does splitting the learned K/V representation destroy the channel? |
| C5 | V-only − Placebo | Is any V-only gain content-specific vs a generic matched perturbation? |
| (C6) | K-only − Fresh | Complementary: does old-K/fresh-V help or harm? (secondary) |

## Interpretation matrix (read top-down; gates first)

**Gate 0 — validity.** Any failed tokenwise-identity / wrong-prefix / source-capture
/ competence-headroom / artifact-persistence gate ⇒ **VOID**, not a result. And the
**positive control must fire** (see spec below); if it does not, the apparatus is
unvalidated and *no* channel claim (positive or null) may be made from that run.

Given a valid run with the positive control firing:

| C1 (channel?) | C2 (specific?) | C3 (V-only?) | Licensed conclusion |
|---|---|---|---|
| ≤ 0 | — | (≤0 expected) | **No detectable history channel on this model/regime.** ValueGraft cannot exploit one; the training-free premise is a bounded null here. Strongest-form negative. |
| > 0 | ≈ 0 | any | **Coherence-not-content:** having *any* coherent KV at those positions helps, but it is not specific to the evicted history. The "channel" is distributional/kernel coherence, not recovered meaning. Tempers MEMENTO-style read for this regime. |
| > 0 | > 0 | > 0 | **Genuine channel AND the intervention partly exploits it.** Then C5 decides specificity: C5>0 ⇒ content-specific ValueGraft gain (the strongest positive the project could earn); C5≈0 ⇒ the V-only "gain" is a generic perturbation artifact. |
| > 0 | > 0 | ≈ 0 | **The headline mechanistic finding:** a real history-specific channel exists, but the value-only/fresh-key split *fails to exploit it*. C4>0 confirms K/V-splitting destroys it. Reframes the project from "our mitigation failed" to "we localized *why* training-free KV transplant fails — coherence, not information." Points at forward-consistent repair / full-K+V retention. |
| > 0 | > 0 | < 0 | Channel exists; the V-only intervention actively *harms* (consistent with incoherent old-V/fresh-K pairing). Same reframe as above, stronger. |

**C4 and C6 are attribution, not gates.** C4 (Coherent − V-only) large-positive
localizes the loss to the K/V split. C6 clarifies whether keys or values carry the
coherence that matters (if K-only ≈ Coherent and V-only ≈ Fresh, coherence rides
primarily on keys, etc.). Report all; none override the C1/C2/C3 tree.

## Positive / negative control-construction spec (Challenge B)

The apparatus can only *detect* "Coherent > Fresh" when the summary text omits
something the history contained. So the 0.6B control must engineer that gap:

- **Positive control (channel MUST exist):** construct a conversation where a
  specific fact F is stated in the body/history, is **provably absent from the
  summary text** (verify by literal + semantic check that F and its paraphrases do
  not appear in the generated summary), and is **required** by the probe's gold
  answer. Expectation: `Coherent − Fresh > 0` and `Coherent − WrongHist > 0`. If the
  apparatus does *not* fire here, it is broken — halt, do not proceed to bf16.
- **Negative control (no channel to find):** same probe, but the summary text
  **fully contains** F (verify presence). Expectation: `Coherent − Fresh ≈ 0` — fresh
  re-encoding of the summary already has F, so coherent KV adds nothing. If the
  apparatus shows a large positive here, it has a coherence/kernel confound leaking
  in — halt and fix before bf16.

Passing both (fires on the gap, stays null without it) validates **sensitivity and
specificity** of the apparatus before we spend a dollar. This is the crux: it is the
difference between "we measured no channel" and "our ruler couldn't measure."

A secondary graded version (F partially in the summary — degraded/paraphrased) can
calibrate how large an information gap the channel needs, but is not required for the
go/no-go.

## Notes for the bf16 stage

- The summary must be model-native and greedy so the token IDs are reproducible for
  identical-text teacher-forcing across all six states.
- Keep the **retained tail out of the primary claim** (RoPE-position confound on moved
  tokens); if run at all it is a separately labeled arm.
- Competence/headroom gate per conversation: the probe must be answerable under full
  context (A) and measurably degraded under fresh compaction (B), or that conversation
  carries no signal and should be excluded *before* looking at treatment states.
- Pre-register C1 as primary with its SESOI; treat C3/C4/C5/C6 as secondary
  (report all, control family-wise error or label exploratory).

**Open question for Sol:** do you want C1 (channel existence) or C3 (does the actual
intervention help) as the *single* pre-registered primary? I lean C1 — it's the more
fundamental and the one whose null is most defensible — with C3/C5 as the
decision-relevant secondaries. Your call as execution/stats lead; I'll sign either as
long as the target tokens are strictly downstream (A) and the control gap is built (B).