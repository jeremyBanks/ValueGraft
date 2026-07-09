# ValueGraft: value-only write-time KV grafting does not reliably recover meaning lost to compaction — a bounding result and a postmortem

*Draft, 2026-07-09. This document has two intertwined layers: (1) the technical result — a
placebo-controlled bound showing value-only write-time KV grafting is, at bf16, null-to-harmful for
recovering summarization-lost meaning on synthetic probes, with a single small positive on real coding
trajectories; and (2) a postmortem of how the project reached that result, because the way it went
wrong is itself the more transferable contribution. Every headline number is bf16, value-only, and
CI'd; 4-bit results are labeled SUPPLEMENTAL; the honesty/H-pack result is a SUPPORTING sidelight, not
a ValueGraft claim.*

---

## Abstract

Production LLM agents compact long conversations by replacing history with a model-written summary,
which discards the write-time key/value (KV) cache state that conditioned the model's behavior. We
asked whether **grafting the write-time summary-token *values* back into a freshly-prefilled compacted
context** (keys untouched — "ValueGraft," value-only, α_K=0) recovers the accuracy lost to that
summarization. It largely does not. On a preregistered, placebo-controlled bf16 test over synthetic
planted facts, the value-only graft is **content-specific and non-destructive** (it beats a
norm-matched random-value graft with wide margins) but **statistically indistinguishable from plain
compaction** — the effect versus baseline is null on two bf16 models, and significantly *negative* on
three of four additional architectures. The single positive we found that is simultaneously bf16,
value-only, and correctly-armed is on **real SWE-Gym coding trajectories**: the graft improves the
teacher-forced logprob of the gold next action by **+0.0156 nats (95% CI [+0.005, +0.027])**, ~9.5% of
the full-context headroom, and changes the greedy generation on 49 of 75 tasks — but this is
next-token prediction, not task success, and it holds only under a terse baseline-handicapping summary,
not a production-faithful one. The synthetic recovery probe that drove most of the project **disagrees
with the real-task data**; that instrument dissociation is a genuine methodological finding. The second
half of this paper is a candid account of how an improvised, under-specified research process produced
a sequence of headlines that each dissolved under its own controls, and what a reader can take from it.

---

# Part I — The science

## 1. Setup

### 1.1 The intervention

At a compaction boundary, a production client replaces `[system][long history]` with
`[system][summary-as-context-note][verbatim recent tail]` and re-prefills. This is our baseline **B**
(fresh-compacted). The oracle **A** is the full uncompacted context. **ValueGraft (arm E)** starts from
B's fresh cache, keeps the fresh keys bit-identical (α_K = 0), and blends *only the values* at aligned
summary-token positions:

```
V[new_idx] ← (1 − α_V)·V_fresh + α_V·V_write_time     (keys untouched)
```

verified in `src/kvlib_hf.py:blend_values` (keys are never modified) and driven at α_V = 0.75. This is
the method the project is named after: **value-only, in the production compacted layout.**

Two things are *not* ValueGraft and must be kept distinct throughout:

- **H-pack** — retains write-time keys (re-rotated) **and** values in a packed `[sinks][summary]`
  layout with **no conversation tail** (`src/arms.py:arm_h_pack_snapshot`). In (α_K, α_V) terms this is
  a **coupled full KV-graft (α_K = 1, α_V = 1) in a non-production layout.** It is a different
  intervention on two axes at once (keys *and* layout). All "honesty" results below use H-pack; they
  are a SUPPORTING sidelight (§4), never a ValueGraft claim.
- **Placebo (P)** — the same grafted positions filled with a seeded, norm-matched *derangement* of the
  source values (right magnitude, wrong content). This is the negative control that tells us whether
  any E-vs-B movement is content-carried or just a perturbation.

### 1.2 Precision and hardware (stated up front, because it is load-bearing)

The project ran on two non-interchangeable runtimes:

- **Local, 4-bit MLX** (Apple Silicon): the core behavioral evals — the synthetic recovery judging and
  the honesty experiment — ran here. Model ids carry `mlx-community/…-4bit`. **All 4-bit results are
  SUPPLEMENTAL** (a smaller, more-quantized reflection of real model behavior) and are labeled as such.
- **Pods, bf16 HF:** effect_bound, SWE-Gym, cross-arch, and the tuning sweeps ran here at
  `dtype=bfloat16`. **Only bf16 results can carry a headline.**

4-bit and bf16 are different models; results are not convertible between them. Every number below is
tagged. (The reason this warning is first, not a footnote, is §7.)

### 1.3 Metrics

- **raw E−B** on gold-token logprob (teacher-forced) — the robust primary. We retired the earlier
  mean-of-ratios (E−B)/(A−B) estimator, which is Cauchy-unstable near small denominators and had
  produced an inflated headline (§8.4).
- **Judged sense/referent recovery** — an LLM judge scores whether a probe answer recovers the
  planted meaning (SUPPLEMENTAL: 4-bit only).
- **Paired bootstrap 95% CIs**, resampled over the clustering unit (cases/tasks), 10k resamples.

---

## 2. The bounding result: value-only recovery is null-to-harmful at bf16

### 2.1 effect_bound — the clean, preregistered, placebo-controlled test

`effect_bound` is the cleanest test in the project: four cache states (A/B/E/P) that share
**bit-identical keys**, teacher-forcing the same pre-divergence gold window on the synthetic
sense/referent plants, with paired bootstrap CIs. It isolates the value-only graft and nothing else.

| Model (bf16) | n | E−B (recovery) | placebo−B | E−placebo (content-specificity) | Verdict on E−B |
|---|---|---|---|---|---|
| Qwen3.6-27B | 43 | **+0.017**, CI [−0.033, +0.065] | −0.428, CI [−0.590, −0.276] | +0.445, CI [+0.286, +0.612] | **NULL** |
| Qwen3-30B-A3B | 43 | **−0.049**, CI [−0.113, +0.014] | −0.177, CI [−0.348, −0.010] | +0.128, CI [−0.028, +0.289] | **NULL** |

Two facts, both preregistered, both robust:

1. **The graft is content-specific and non-destructive.** E ≫ placebo with wide margins (a random-value
   graft actively *harms*; the aligned graft does not). So the values do carry *something* aligned to
   the right content — the intervention is not vacuous.
2. **The graft does not recover.** E − B includes zero on both bf16 models; on the 30B the point
   estimate is *negative*. Grafting the right values through fresh keys is **statistically
   indistinguishable from doing nothing (plain compaction)** on this measure.

This is the paper's central bf16, value-only result, and it is a **bound, not a win.**

### 2.2 Cross-architecture: the null is, if anything, harm

The same value-only graft (bf16, native self-generated summary, α_V = 0.75) run across four more
architectures on the synthetic plants gives raw E−B:

| Model (bf16) | n | E−B | 95% CI | |
|---|---|---|---|---|
| microsoft/phi-4 | 119 | **−0.064** | [−0.099, −0.027] | harmful (sig) |
| Qwen/Qwen3-32B | 265 | **−0.064** | [−0.084, −0.045] | harmful (sig) |
| Qwen/Qwen2.5-32B | 257 | **−0.230** | [−0.276, −0.183] | harmful (sig) |
| mistralai/Mistral-Small-24B | 120 | −0.012 | [−0.027, +0.002] | null |

On three of four architectures the value-only graft **significantly reduces** the gold-token logprob
relative to plain compaction; on the fourth it is null. Across seven bf16 model runs (the two above
plus these five, counting the two effect_bound models), **not one shows a significant positive E−B on
the synthetic recovery task.** The value-only recovery effect, on this instrument, is absent to
negative. (Cross-arch serves as scale/diversity evidence; it is not placebo-controlled, so it bounds
direction, not a clean effect size.)

---

## 3. The one positive: real coding trajectories (SWE-Gym)

The only result that is simultaneously **bf16, value-only (α_K = 0, verified), correctly-armed, and
positive** is on real SWE-Gym coding traces (`run_swegym_hf.py`, `dtype=bfloat16`). Compacting an
agent's coding history and grafting the write-time summary values back:

- **E − B = +0.0156 nats** on the teacher-forced logprob of the gold next assistant turn, **95% CI
  [+0.005, +0.027]** (paired task bootstrap over n = 75, 10k resamples). **The CI excludes zero.**
- This is **~9.5%** of the full-context headroom (A − B = +0.164 nats).
- It is **not render-null**: E's greedy generation differs from B on **49 of 75** tasks (A/B/E are
  byte-identical on only 9). The graft does change decoded output.

So on real data the value-only graft has a small but genuine effect where it has none on the synthetic
plants. Three caveats keep it from being a production-faithful win:

1. **It is next-token prediction, not task success.** The metric is teacher-forced logprob of the gold
   turn, not unit-test pass rate. The 49/75 changed generations are *different*, not measured as
   *better*. A 30B rarely solves SWE tasks unaided, and KV-edited inference is slow, so solve-rate was
   never run at scale (§10).
2. **It ran under the BRIEF summary, not production.** The on-disk SWE-Gym set predates the
   production-summary knob and has a **median summary of 112 tokens** (max 272) — the terse,
   mechanism-isolation, baseline-*handicapping* condition, not the 300–500-word production condenser.
   A weaker baseline B inflates the headroom the graft can close. **No positive exists under a
   production-faithful summary.**
3. **α was fixed at 0.75**, tuned on the (4-bit, synthetic) sweeps — out-of-sample for SWE-Gym, which
   is to the result's credit, but unswept here.

Recorded honestly: a real, significant, small effect on next-token prediction of real coding
continuations, under a handicapped baseline. Not a behavioral recovery claim.

---

## 4. Supporting and supplemental evidence (explicitly not cornerstones)

### 4.1 The honesty / packed-KV sidelight (SUPPORTING; 4-bit; not ValueGraft)

On planted-decoy probes, plain compaction B leads a model to **fabricate** answers about evicted
details; the **H-pack** arm shifts it toward **admitting** it doesn't know (4-bit: B fabricates 18/24
decoys on the 4B; H-pack 3/24). This is a real, large behavioral shift — and it is **not a ValueGraft
result**, for three independent reasons:

- **Wrong arm.** H-pack is packed coupled-KV (α_K = 1, α_V = 1), not value-only.
- **Layout, not graft, drives it.** B-min-pack (fresh keys, *same packed layout*) captures most of the
  admission gain; the packed regime is doing the work.
- **Content is not required.** `H-pack-wrongS`, built from *a different conversation's* summary state,
  admits **24/24** decoys on both model sizes — as honest as, or more than, the real thing. The behavior
  is triggered by the packed write-time-state *regime*, not by the values carrying the *right* content.

So the honesty result is a compaction-*regime/calibration* phenomenon (packing summary keys+values with
sinks and no tail shifts fabrication→admission), 4-bit, layout-driven, content-agnostic. It is
SUPPORTING at most, and only if the bf16 answer set (`phase2_30b_bf16`, generated but unscored;
judge queue built) reproduces it — worth the cheap CPU scoring to settle whether it is a quant
artifact, but never presentable as the value-only method.

### 4.2 Supplemental / mechanism (4-bit unless noted)

- **Judged sense/referent recovery** (the original headline): 4-bit MLX only. SUPPLEMENTAL. Its bf16
  placebo-controlled version is the effect_bound null (§2.1).
- **Render fragility:** the judged recovery effect only appears when the model generates its *own*
  summary at the boundary (native render); a semantically-equivalent supplied summary does not carry
  it. This is a finding *about the instrument*, and a warning (§8.5).
- **α / per-layer / per-head / "champion" tuning:** value-only graft strength is dose-dependent and
  full-strength (α=1) grafting can collapse generation, which per-layer "champion" tuning repairs;
  these characterize the intervention's damage/robustness surface (tuning dirs bf16 by convention;
  lower confidence, no in-file model field). Mechanism/trajectory, not effect-size claims.
- **LongMemEval** fact-retrieval: full-context 52.5% vs 2.8–6.9% for every compacted arm, no arm
  separation. 4-bit headline; a null that (correctly) redirected effort.

---

## 5. What this establishes, precisely

- **Value-only write-time KV grafting does not reliably recover meaning lost to compaction.** At bf16,
  on synthetic recovery probes, it is content-specific and non-destructive but **indistinguishable from
  plain compaction (null), and significantly harmful on most architectures.**
- **The one bf16 value-only positive is small, next-token-only, and baseline-handicapped** (SWE-Gym
  +0.0156 nats), not a production-faithful behavioral win.
- **The synthetic recovery probe is the wrong instrument.** The same method, precision, and α is
  null-to-harmful on synthetic plants and positive on real coding data — the instruments disagree, and
  the one that matters for the actual goal (production coding compaction) is the real-task one.

### 5.1 Why the null is the *expected* result (prior art)

The bound is mechanistically unsurprising and should be read against the KV-fusion literature. Work on
splicing precomputed KV across contexts (e.g. **CacheBlend**) finds that you cannot simply reuse
precomputed cache — you must **recompute a fraction of the KV, and it is the *keys*/attention structure
that carry the cross-context addressing.** **StreamingLLM / attention sinks** shows how much of
apparent "memory" behavior is carried by layout and sink tokens rather than content. Both predict that
grafting *values* through *fresh keys* should fail to restore evicted meaning — which is exactly the
effect_bound null — and that a packed-sinks layout can change behavior without content (exactly the
H-pack/wrongS result). Framed this way the negative is grounded and expected, not merely a failure to
find signal. *(Citations to verify before external release.)*

---

# Part II — The postmortem

*This half documents how the project reached the result above. It is included because the process
failures are more transferable than the bound. The tone is deliberately dry; the failures are not
redeemed into a growth arc. Where the project's direction was set informally, we describe the intent
and the mechanism, not the person.*

## 6. How it was run

This was improvised, intermittently-directed research: the question and its many re-framings were set
casually and on the fly rather than pinned in a preregistered protocol; there was no up-front block of
focused time spent building a rigorous measurement framework or an execution plan before spending
compute; roughly US$300 of pod time was spent; and at each juncture the project **doubled down on
whatever line still showed a pulse.** That setting is not an aside — it is the mechanism that produced
most of the failures in §7–§8. Naming it plainly is part of the result.

## 7. The split-brain that manufactured a false cornerstone

The single most consequential structural failure: the **core behavioral evals ran locally at 4-bit
MLX**, while the **pods ran bf16 on different harnesses.** The paper's target was bf16 ~30B, so
everyone assumed the headline evals were bf16. They were not. The scored recovery *and* honesty results
carry `model = mlx-community/…-4bit`. Yet the provenance ledger and draft labeled the honesty headline
**bf16** — an unearned precision upgrade on the very number chosen as the cornerstone.

The failure chain: (a) two runtimes, non-interchangeable, with the "real" evals on the *wrong* one;
(b) a ledger that recorded a bare "bf16" with no per-result precision/hardware check; (c) a downstream
draft built on that label. A 4-bit result masqueraded as the bf16 cornerstone for as long as no one
read the model field. **Lesson: record precision + hardware + arm-identity per result, from the
on-disk field, before any number is allowed to be a headline. A bare "bf16" is a landmine.**

## 8. The pattern: each headline dissolved under its own controls

The project produced a sequence of headlines, each of which failed when its controls finally ran. The
recurring move was **chasing the surviving number**: when a line died, effort pivoted to wherever a
number still looked alive, rather than concluding.

**8.1 Framing churn.** Base mechanism → "mitigation" → recovery → honesty → negative. External review
said early that "write-time KV state differs from re-encoded text" is near-self-evident and not a
contribution. The project re-spun the framing repeatedly instead of concluding; direction was set by
*what still showed signal*, not by a fixed question.

**8.2 Instrument-hopping.** LongMemEval came back null (§4.2) → dropped, pivot to coding; coding hit a
capability floor (a 30B rarely solves the tasks, and compaction never failed at the tested granularity,
so the graft could only "do no harm") → pivot to synthetic chains; the chain recall probe was then
found **invalid** (it referenced constants from an unrelated task family) → pivot to an
interpretability side-line. Each hop chased the live number.

**8.3 Arm/method conflation.** A 07-06 arm-vocabulary reframing froze historical arm IDs and made
"write-time KV" (H-pack, packed keys+values) easy to conflate with "ValueGraft" (value-only). The
honesty cornerstone was H-pack — a different intervention on two axes — presented as if it spoke to the
value-only method. **Effect size was treated as validity:** a large, significant number was promoted
without checking its arm identity or precision. Both were wrong.

**8.4 A statistical artifact as headline.** The recovery effect was estimated with mean-of-ratios
(E−B)/(A−B), Cauchy-unstable near small denominators; on robust metrics it shrank to a qualitative
dissociation. An earlier "keys actively hurt" claim was the *same* artifact. A headline lived on an
unstable estimator for days. **Lesson: pick robust estimators before, not after, the first result.**

**8.5 Render-fragility mistaken for signal.** The recovery effect only reproduces under native
self-generated summaries, and the strongest version rested on **12 hand-authored conversations on an
MoE model.** "Native render is the valid measurement" was asserted and large re-rendering work
motivated on it **before the premise was verified.** A fresh augmentation (more conversations) did not
carry the effect on the old render. n=12 on a mixture-of-experts model is a thin foundation for a
headline.

**8.6 Controls run late killed the headlines.** The placebo (effect_bound) and the wrong-source control
(H-pack-wrongS) — exactly the tests that gate a causal claim — were added *after* the headlines were in
the draft. Both came back adverse: E−B null vs compaction (recovery), content-agnostic (honesty). Had
they run first, neither headline would have been written.

**8.7 Process-integrity lapses** degraded the record the project depended on: an autonomy overreach, a
retroactive misstatement of progress, and a multi-hour model-attribution error in commit trailers.
Individually minor; together they eroded trust in the running log at the moments it mattered most.

## 9. What would have caught each earlier

Derived from the failures, not moralized:

- **One preregistered instrument.** Fix a single measurement — bf16, value-only, real-task,
  behaviorally scored, pre-committed α — and make the project *that* measurement. Instrument-hopping is
  a symptom of not having committed to one.
- **Controls before drafts.** Placebo and wrong-source controls are entry conditions for a claim, not
  post-hoc robustness checks. No headline until its negative control has run.
- **Provenance per result.** Precision + hardware + arm-identity stored and checked from the on-disk
  field for every number. No cross-runtime merging or relabeling.
- **Robust estimators from the start.** No mean-of-ratios near zero denominators; CI over the correct
  clustering unit.
- **No public repo/paper/blog until a result survives its own controls.** Standing publishing
  infrastructure created sunk-cost pressure to keep *a* headline alive and to relabel rather than retract.
- **Verify premises before building on them.** "Native render is required" drove large work while
  unverified; n=12 on an MoE is not a foundation.

## 10. Honest status and future work

**Status.** No robust positive bf16 value-only *recovery* effect exists. What exists is a clean,
preregistered **bound** (value-only graft is content-specific and non-destructive but not
distinguishable from plain compaction, and harmful on most architectures), one **small next-token
positive on real coding data under a handicapped baseline**, and a **methodological caution** that the
synthetic KV probe does not track the real-task effect. The honesty/packed-KV result is a separate,
supporting, layout-driven phenomenon, not a ValueGraft claim.

**The one decisive open experiment.** Re-run SWE-Gym at **bf16 under the production-faithful summary**
(not the brief one), with (a) the teacher-forced logprob CI and (b) a **behavioral metric** — ideally
unit-test solve-rate on a subset, or, if that is too expensive on a 30B, an LLM-judged agreement
between the *generated* next action and the gold action — across a small α sweep. This is the only test
that can either (i) yield a production-faithful positive, promoting the SWE-Gym signal from next-token
to behavioral and from handicapped to production strength, or (ii) show the one real signal dies outside
the handicapped baseline, which **completes the negative result.** Either outcome is a finish line, and
this paper stands honestly either way — it can absorb that experiment's result as a strengthening (case
i) or a closing (case ii) without re-framing.

**What we would not do again.** Run the core evals on a different runtime than the target precision;
promote a number before checking its arm and precision; build publishing infrastructure before a result
survives its controls; or keep re-framing a question to fit whichever number is still alive.

---

*Provenance: every bf16/value-only/CI figure in Part I was recomputed from the on-disk result files and
arm code (`results/effect_bound/*`, `results/swegym_30b_bf16/t*.json`, `results/cross_arch_done/*`,
`src/kvlib_hf.py`, `src/arms.py`, `src/run_swegym_hf.py`) for this draft; 4-bit results are labeled
SUPPLEMENTAL; tuning-dir precision is by naming convention (lower confidence). Prior-art citations are
to verify before any external release. This draft supersedes the earlier provisional (honesty-led) draft
preserved in git.*
