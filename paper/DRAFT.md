# Re-injecting write-time attention values across a compaction boundary: a mostly-null bounding result

*By Anthropic Claude Fable 5 and OpenAI GPT 5.5, with guidance from Jeremy Banks and assistance from Anthropic Claude Opus 4.8, Anthropic Claude Sonnet 5, and Google Gemini Pro 3.1.*

---

## Abstract

When a long conversation is compacted — the history replaced by a short summary and the most recent messages, then re-encoded — the key/value (KV) attention state the model had built up while *generating* that history is discarded and rebuilt from the summary text. We asked a narrow, concrete question: does that discarded write-time state contain recoverable meaning that the re-encoded summary loses? Concretely, if we copy the model's original write-time attention **values** (the V vectors, leaving the keys untouched) back onto the summary tokens of the freshly compacted cache, does the model predict the true continuation better than it does from plain compaction?

The short answer is: almost never, and only under one narrow condition. Across a held-out, placebo-controlled test set, four tuned variants of the graft (per-layer, per-head, their intersection, their union) all failed to beat the plain compacted baseline on the recovery metric — every confidence interval spanned zero. A dedicated compression sweep, built to test the natural hypothesis that the graft should help more when the summary is more lossy, found the opposite of a signal: the graft was flat and slightly *negative* across four compression levels, even as the amount of meaning the summary evicted more than doubled. The one place a positive effect survived its controls was on real coding-agent trajectories (SWE-Gym), under aggressively short summaries, measured as teacher-forced next-action log-probability — a proxy, not task success: there the plain graft recovered about +0.013 nats/token (95% CI [+0.002, +0.026]), and the tuned variant neither beat it nor cleared zero on its own. Under realistic-length summaries the same coding effect vanished.

One property is worth stating carefully: the graft is *content-specific* wherever it touches enough of the cache. Injecting the correct write-time values is reliably, and for slot-heavy configurations dramatically, better than injecting shuffled or energy-matched random values into the same slots (the effect scales with the number of grafted slots, and is itself null for the smallest configuration). So the values do carry information tied to the right content — the intervention is not vacuous. But that advantage is entirely a matter of *not* inflicting the harm a wrong graft causes; it never converts into out-performing the re-encoded summary the model already has.

We report this as a bound with one small, narrow positive, and we spend the second half of the paper on the process — a measurement-provenance failure that briefly manufactured a false headline, and the "chase the number that still looks alive" dynamic that produced several others — because those lessons transfer further than the bound does.

---

# Part I — What we tested and what we found

## 1. Background and the question

Production LLM assistants keep conversations inside a fixed context window by *compacting*: at some threshold the client replaces `[system][long history]` with `[system][short summary][verbatim recent tail]` and re-encodes that shorter prompt. The summary is text; the model re-reads it from scratch. Everything the model had computed internally while producing the original history — in particular, the per-layer, per-position key and value vectors in its attention cache — is thrown away and never reconstructed, because the tokens that produced it are gone.

There is a folk intuition, and some adjacent published work (KV-cache editing and composition, "gist" and "beacon" summary-token compression, prefill note-taking), that this write-time state is not fully recoverable from its own textual summary — that generating a summary and then reading it back are not the same, internally. If that were true and the lost part were *useful*, then re-injecting the write-time state at the compaction boundary would be a cheap way to recover continuity that the summary drops.

We tested the most targeted version of that idea we could construct, and we tested it hard enough to know whether it works. It mostly does not. This paper is the negative space around a plausible technique, mapped carefully, plus the single narrow regime where a small real effect survives.

A note on framing, since a reader will reasonably wonder why a null is worth writing down. The intervention is simple enough that "someone must have tried this" is the default assumption; three independent literature searches did not turn up this exact composite (write-time **value** retention with fresh keys, evaluated as semantic continuity across a summarization boundary). So the contribution here is not a method — it is a carefully controlled measurement of whether an obvious-looking method does anything, with the provenance discipline to make the answer trustworthy.

## 2. Method

This section is deliberately detailed: a motivated reader should be able to reconstruct every experiment from the prose, because the result rests entirely on the controls being what we say they are.

### 2.1 The intervention: a value-only graft in the production compacted layout

Fix a single model, **Qwen3-30B-A3B-Instruct-2507** (the non-thinking, instruction-tuned, date-stamped checkpoint; a different checkpoint from the same-sized "thinking" variant, a distinction that cost us hours once and is load-bearing), loaded in **bfloat16** (bf16). All headline numbers in this paper are from this model at this precision; the precision was read back from a live parameter tensor at run time, not inferred from a directory name, for reasons the postmortem makes painfully clear.

Three cache states are compared, all sharing the same underlying model:

- **Full context** — the model prefills the entire uncompacted history. This is the upper reference: what the model would predict if nothing were ever compacted.
- **Compacted baseline** — the production layout: `[system][summary as a context note][verbatim recent tail]`, freshly prefilled. This is the thing a real client actually runs, and the thing any recovery method must beat.
- **The value graft** — start from the compacted baseline's fresh cache; keep its keys exactly as they are; and at the summary-token positions, replace the value vectors with a blend of the fresh values and the model's *write-time* values from those same tokens as they were originally generated:

  `V[position] ← (1 − α)·V_fresh[position] + α·V_write-time[position]`, with the keys never touched.

We drive this at α = 0.75 by default (a value picked by tuning, below). Keeping the keys untouched (α_K = 0) is the defining choice: keys carry positional/RoPE-encoded addressing, and mixing write-time keys into a re-encoded layout creates position-mismatch artifacts; values are the content-bearing operand. Restricting to values isolates "is there recoverable *content* in the write-time state" from "can we re-address the cache." The blend is pure tensor surgery on the saved value tensors; an α = 0 graft reproduces the compacted baseline bit-for-bit, which we assert as a plumbing check in every run.

There is a related, coarser intervention that retains write-time keys *and* values in a packed layout with no conversation tail. We ran it only as a contrast and it is not the value graft; it appears once here as a footnote-level control and nowhere in the results, to avoid conflating a keys+layout intervention with the value-only one.[^packed]

[^packed]: The packed-KV contrast (write-time keys re-rotated *and* values, in a `[sinks][summary]` layout with no tail) changes two things at once — key content and sequence layout — and lives in a separate line of experiments about honesty-under-compaction, not recovery. We keep it out of the recovery results entirely.

### 2.2 Aligning write-time positions to compacted positions

The graft needs to know which position in the write-time cache corresponds to which position in the compacted cache. The summary is generated fresh, so its tokens do not sit at the same absolute positions they occupied at write time, and the two token sequences are not identical. We align them with a positional-within-region difflib match: within the summary region and within the retained tail region separately, matching tokens are paired by longest-common-subsequence and grafted; unmatched tokens are left with their fresh values. We match within regions rather than across the whole sequence to prevent a summary token from being paired to an unrelated tail token that happens to share a subword. An earlier strict exact-span variant was tried and reverted because it breaks on the tokenization of thinking-model self-generated summaries.

### 2.3 The corpus, and who or what generated every token

Provenance is the spine of this paper, so we state the origin of each component explicitly.

- **Scenarios and prompts are authored, not model-generated.** The evaluation corpus is a set of synthetic multi-turn conversations built from hand-authored scaffolds: a system prompt, a sequence of user turns, and a set of *planted facts* deliberately placed early in the conversation so that later compaction will evict them. Real user prompts are not model outputs; a result that depended on model-generated prompts would not be measuring what we claim. The scaffolds and plants were authored by directed model assistants under human direction and schema-checked.
- **The planted facts span six categories**, chosen to separate kinds of evicted meaning: *sense* (the meaning of a term introduced earlier), *referent* (which earlier-decided option a later phrase points to), *stance* (a preference the user expressed), *ruled_out* (an option explicitly rejected), *evicted_fact* (a precise verbatim detail), and *strong_prior* (a code-name that collides with a famous prior meaning). Each plant carries a *probe* and a *gold continuation* that can only be produced correctly if the planted fact survived.
- **The assistant replies in the conversation body are generated in-context by the test model itself.** This is essential and non-obvious: the graft re-injects the model's *own* write-time values, so the conversation it operates on must be one this model would actually produce. We do not reuse replies written by another model. Each conversation is rendered by the test model growing its own KV cache turn by turn from the shared scaffold. When we discovered (below) that reusing a different model's replies collapses the effect, this stopped being a convenience and became part of the mechanism.
- **The summary is self-generated by the test model.** At the compaction boundary the model summarizes its own conversation, and that summary is what the compacted baseline re-encodes and what the graft's write-time values come from. This too is load-bearing: a summary written by a *different* model, held fixed and fed to the test model, suppresses the graft to null (measured directly: a fixed foreign summary drove the recovery metric to about +0.004, indistinguishable from zero, while the model's own summary reproduced the development-set effect). The interpretation is that whatever the write-time values carry is the residue of the model's own act of summarizing — the internal computation of compressing *this* history — not anything a summary it merely reads can carry. So "use the model's own summary" is not a knob we tuned for the best number; it is the only condition under which the graft's values are content-specific at all (and the only one under which the development-set showed any recovery signal), and it is also the deployment-realistic one. Note that this makes the model's own summary necessary for the *mechanism* to be present — it does not, as the held-out results below show, make recovery beat the baseline.
- **The gold continuation is derived from the planted facts and shared across conditions**, not model-generated. Because the recovery metric is a *difference* between two conditions scored on the *same* gold tokens, sharing the target cancels any per-condition advantage in producing the target itself, leaving only the effect of the graft.

The corpus is split once: conversations used to tune the graft (a validation set) are disjoint from the conversations used to evaluate it (held-out c07–c24, eighteen conversations). Every number that decides the result comes from the held-out set.

### 2.4 Metrics, stated as what they are

- **Recovery (the primary metric).** For each plant we teacher-force the shared gold continuation and take the mean per-token log-probability under each cache state. The headline quantity is the graft's lift over the compacted baseline: **recovery = logprob(graft) − logprob(compacted)**, in nats/token. Positive means the graft makes the true continuation more likely than plain compaction does. This is a log-probability proxy for "did the model retain the meaning," not a measure of downstream task success.
- **Content-specificity.** The same lift, but measured against a *placebo* graft instead of the baseline: **logprob(graft) − logprob(placebo)**. This asks whether it matters that we injected the *correct* write-time values rather than scrambled ones. It is a mechanism check, not a performance measure: a large content-specificity with a null recovery means "the right values move the output, but not toward beating the summary."
- **Confidence intervals** are percentile bootstraps resampled over **conversations**, not over individual plants. Plants within one conversation share a context and a summary and are correlated; resampling whole conversations is the correct inferential unit and gives wider, properly-sized intervals. We report the conversation-clustered interval as the headline throughout. (An earlier version of this project used a ratio estimator, recovery divided by the full-context-vs-baseline gap; it is Cauchy-unstable near small denominators and produced an inflated early headline. It is retired.)

### 2.5 The placebo battery

The placebo graft is what separates "the right values did something" from "any perturbation did something." We use three constructions, all of which graft *something* into exactly the same positions with the same α, differing only in the source:

- **Position-shuffle** — the correct write-time values, permuted across the grafted positions. Right values, wrong slots.
- **Cross-probe-shuffle** — values drawn from a *different* conversation's write-time state at the grafted positions. A different, still-structured write-state.
- **Energy-matched noise** — Gaussian vectors rescaled so each row's norm matches the mean norm of the real grafted block in that layer. Right magnitude, destroyed structure.

If the real graft merely beat zero-magnitude or obviously-broken inputs, that would be uninformative; matching magnitude and structure makes content-specificity a real test.

### 2.6 Tuning the graft, and the four validated variants

A flat α across all layers is the naive form. We also tuned per-layer and per-head **champion** configurations on the validation conversations (disjoint from the held-out set): profiling each layer's marginal contribution to build a per-layer α map (27 of the model's layers receive a nonzero α of 0.75 or 1.0), and profiling each (layer, attention-head) slot to build a per-head mask. This yields four grafts to evaluate on held-out data: **per-layer**, **per-head**, their **intersection** (slots selected by both), and their **union** (slots selected by either). Each is applied identically to the real and placebo grafts, so the content-specificity comparison holds the slots and α fixed and varies only the source values.

### 2.7 The compression sweep

To test whether the graft helps specifically when the summary is *lossy*, we vary only the summary's length/detail budget, holding everything else fixed, across four levels: **ultra** (a one-sentence, ≤25-word note), **brief** (three-to-five sentences, no specifics), **medium** (~150 words), and **realistic** (~300–500 words, the standard "thorough context note"). To make "aggressive vs realistic" a number rather than a label, the harness measures the realized **compression ratio** — clean summary tokens divided by full-context tokens — for every conversation and reports recovery and content-specificity as a function of it.

### 2.8 The coding-trajectory test (SWE-Gym)

Synthetic plants are a controlled but artificial probe. As an out-of-domain check on real content, we use 75 real OpenHands agent trajectories from SWE-Gym. Each trajectory is cut at roughly 75% of its tokens, at a boundary just before an assistant action; the earlier context is compacted (self-generated summary + recent tail) exactly as in the synthetic setting; and we teacher-force the **true next assistant action** and score its mean log-probability. This is explicitly a proxy — next-action predictability, *not* whether a patch is produced or a test passes. We run it in a 2×2: the plain graft (flat α = 0.75) versus the tuned per-layer graft, each under **brief** and under **realistic** summaries.

Every result file and a per-run manifest record, read from the running model: the resolved checkpoint and precision, the graft's identity (flat α, or the champion config with its content hash and label), which summary condition and its exact text hash, the metric definition marked as a proxy, the held-out split, and the code commit. This born-annotated provenance is a direct response to the failure documented in Part II.

## 3. Results

### 3.1 On held-out conversations, no tuned graft beats plain compaction

The four champion variants, evaluated on the eighteen held-out conversations (203 plants) against the plain compacted baseline, all land on zero for recovery:

| Graft variant | Recovery (nats/token) | 95% CI (conversation-clustered) | Verdict |
|---|---|---|---|
| Per-head | +0.017 | [−0.027, +0.062] | null |
| Per-layer | −0.003 | [−0.041, +0.043] | null |
| Intersection | +0.008 | [−0.032, +0.052] | null |
| Union | −0.012 | [−0.053, +0.034] | null |

Per-head's point estimate is nominally the highest, but its interval spans zero and overlaps per-layer's completely, so per-head does not reliably beat per-layer; no combination clears the baseline; every interval spans zero. This is the definitive test — tuned on separate data, evaluated held-out, placebo-controlled — and it is null.

Content-specificity, by contrast, is real and often large, and it scales with how many slots the graft touches:

| Graft variant | Content-specificity (graft − placebo) | 95% CI |
|---|---|---|
| Intersection (fewest slots) | −0.03 | [−0.10, +0.05] |
| Per-layer (27 layers) | +0.19 | [+0.12, +0.27] |
| Per-head | +0.72 | [+0.66, +0.78] |
| Union (most slots) | +1.08 | [+1.01, +1.13] |

(Values shown against the position-shuffle placebo; the other two placebos give the same ordering and larger margins.) The reading is precise and a little subtle: the more of the cache you graft, the more a *wrong* graft harms, and the more the *right* graft avoids that harm — so injecting the correct values is unambiguously better than injecting scrambled ones. But that advantage is entirely a matter of not-hurting; it never converts into out-performing the summary the model already has. The mechanism is real; the performance is null.

### 3.2 The graft does not concentrate under aggressive compaction — it is flat and slightly negative

The compression sweep was designed to give the graft its best chance: shorter summaries evict more meaning, so there is more for a recovery method to recover. The eviction did grow as intended — the gap between full-context and compacted log-probability more than doubled from the realistic summary to the one-sentence summary. The graft did not follow it:

| Level | Measured compression ratio | Summary length (tokens) | Evicted meaning (full − compacted) | Recovery (graft − compacted) | 95% CI |
|---|---|---|---|---|---|
| Ultra (~1 sentence) | 0.009 | ~28 | 1.89 | −0.029 | [−0.068, +0.013] |
| Brief (3–5 sentences) | 0.037 | ~120 | 1.85 | −0.026 | [−0.072, +0.022] |
| Medium (~150 words) | 0.083 | ~268 | 1.38 | −0.025 | [−0.067, +0.023] |
| Realistic (~300–500 words) | 0.259 | ~836 | 0.84 | −0.023 | [−0.070, +0.033] |

Recovery is flat across a nearly 30-fold range of compression ratio, sits slightly *below* zero at every level, and its conversation-clustered interval spans zero everywhere. (The most aggressive level is mildly worse: its plant-level interval, which is anti-conservative, actually excludes zero on the negative side, and one category — precise verbatim facts — is significantly *harmed* by grafting under the one-sentence summary, −0.17 [−0.25, −0.09]. Re-injecting write-time values does not help the model reproduce a verbatim string it can no longer see, and slightly displaces it.) Content-specificity stays real and roughly constant (+0.16 to +0.18, interval excluding zero) across all four levels — again, right values beat shuffled values everywhere, and beat the baseline nowhere.

This refutes, on this metric and model, the natural hypothesis that motivated the sweep. More eviction does not summon the effect; the graft is null-to-slightly-harmful regardless of how lossy the summary is.

### 3.3 The one surviving positive: real coding trajectories under short summaries

On the 75 real SWE-Gym trajectories, the picture is different and, for the first time, net positive — narrowly, and only in one cell of the 2×2.

| Summary condition | Graft | Next-action recovery (nats/token) | 95% CI | Trajectories helped |
|---|---|---|---|---|
| Brief | Plain (flat α = 0.75) | +0.013 | [+0.002, +0.026] | 46 / 75 |
| Brief | Tuned per-layer | +0.012 | [−0.001, +0.024] | 48 / 70 |
| Realistic | Plain (flat α = 0.75) | +0.001 | [−0.013, +0.015] | 39 / 75 |
| Realistic | Tuned per-layer | — (not run to completion) | — | — |

Three things to read carefully. First, the plain graft under brief summaries clears zero: +0.013 nats/token, interval above zero, a modest majority of trajectories improved. An independent earlier run of this same cell gave +0.016 [+0.005, +0.027]; the effect is small and it replicates. Second, the tuned graft does **not** beat the plain one — on the trajectories where both ran, the head-to-head difference is −0.001 [−0.010, +0.006], flat — and on its own the tuned graft's interval just includes zero. The tuning that was derived on synthetic conversations buys nothing on the coding task; if anything the plain, single-number graft is the more robust form here. Third, under realistic-length summaries the effect is gone (+0.001, spanning zero), consistent with the synthetic compression sweep in direction even though the synthetic sweep never produced a positive at any length. The coding positive lives specifically in the aggressive-compaction, brief-summary regime.

We stress what this is not: it is next-action log-probability, a proxy. We did not apply the predicted actions, run the tests, or measure resolve rate. The claim is "the graft makes the model's own next action modestly more predictable under brief compaction on these trajectories," and nothing beyond it.

### 3.4 Concordance with the broader bf16 picture

Two earlier bf16 tests, run before the held-out champion validation, point the same way and are worth recording as corroboration rather than as separate headlines. A tightly-controlled four-state test on synthetic sense/referent plants (full / compacted / graft / placebo, bit-identical keys) gave a recovery interval spanning zero on both a hybrid 27B and this 30B, with large positive content-specificity on both — the same "content-specific, non-recovering" signature. And running the same value-only graft across four additional architectures gave recovery that was null on one and significantly *negative* on three (−0.06 to −0.23) — actively harmful. These two tests are less controlled than the held-out validation (they were not the placebo-controlled, tune-then-hold-out design of §3.1–3.2), so we read them only as concordant corroboration, not as primary evidence. Nothing in the wider sweep contradicts the held-out null; if anything the graft is worse on other architectures than on the one it was tuned for.

## 4. What this establishes, precisely

- **On held-out, placebo-controlled synthetic recovery, the value-only graft is null.** No flat or tuned variant beats plain compaction; per-head does not reliably beat per-layer; the result is stable across four grafts and four compression levels. The held-out point estimates range from −0.012 to +0.017, with every interval spanning zero.
- **The graft is content-specific but non-additive.** For grafts that touch enough of the cache, injecting the correct write-time values beats injecting shuffled or noise-matched values — sometimes by more than a nat, though the effect scales with slot count and vanishes for the smallest configuration — so the values carry content-aligned information. That information does not add over the re-encoded summary. "The write-time state differs from its re-encoding" is true and measurable; "the difference is *useful continuity the summary dropped*" is, on this evidence, not.
- **There is one small, real, narrow positive.** On real coding trajectories, under aggressively short summaries, the plain graft improves next-action log-probability by about +0.013 nats/token (CI above zero), replicated. It disappears under realistic summaries, the tuned graft does not improve on it, and it is a proxy, not task success. It is bounded to exactly those conditions.
- **Everything is one model family, bf16, at ~30B, on log-probability proxies.** We make no claim beyond what was measured. We did not demonstrate downstream task improvement, generalization across model families (the cross-architecture evidence is, if anything, adverse), or any effect under realistic-length summaries.

## 5. Why a null is the expected result here

With the controls in, the null stops being surprising. The compacted baseline already contains the summary as text, and the summary was written by the same model that would receive the graft; a competent summarizer puts the recoverable, decision-relevant content into words. The write-time values at those summary positions are the residue of having computed that same content — genuinely different from a cold re-read (which is why content-specificity is real, and why a foreign or shuffled source collapses it), but largely *redundant* with what the text conveys once the model re-reads it. On the synthetic plants this redundancy is close to total, and it stays that way no matter how terse the summary: the compression sweep made the summary as short as one sentence and evicted more than twice as much meaning, and the graft still recovered nothing. So on this corpus, "how lossy the summary is" is not the missing variable.

The single positive lives on a different axis, and we are careful not to over-explain it. It appears only on the real coding trajectories, and only under brief summaries — but the equally-brief *synthetic* summaries produced no positive at all. So summary terseness is at best necessary and clearly not sufficient; the distinguishing factor is the **domain**. Real long tool-use trajectories differ from short planted-fact conversations in ways we did not isolate — far more context, genuine procedural state, actions rather than recalled facts — and something in that difference leaves a small amount of predictively-useful write-time structure that a brief textual summary drops. We can bound the effect to those conditions; we cannot, from this data, name the mechanism behind it, and we do not claim to.

The composite appears to be largely unexplored in the literature, and after mapping it we can see why it would be easy to leave unwritten: it is a natural thing to try, and on controlled probes it mostly does nothing.

---

# Part II — How the project reached this, and what went wrong on the way

*This half documents the process, because the process failures generalize further than the bound. The tone is dry on purpose; the failures are not redeemed into a growth arc. Where direction was set informally, we describe the intent and the mechanism, not the person.*

## 6. How it was run

This was improvised, intermittently-directed research. The question and its several re-framings were set casually and revised on the fly rather than pinned in a preregistered protocol; there was no up-front block of time spent building the measurement framework before compute was spent; roughly a few hundred dollars of GPU time went through it; and at most junctures the project doubled down on whichever line still showed a pulse. That setting is not an aside — it is the mechanism behind most of the failures below, and naming it is part of the result.

## 7. A provenance gap briefly manufactured a false headline

The single most consequential structural failure was a split between two runtimes. The core behavioral evaluations for a long stretch ran locally in **4-bit** quantization, while the GPU pods ran **bfloat16** on different harnesses. The paper's target was a bf16 ~30B model, so the write-up machinery assumed the headline evaluations were bf16. They were not — the scored recovery and a separate honesty-under-compaction result carried 4-bit model identifiers — and yet a progress ledger recorded a bare "bf16" against them, an unearned precision upgrade on the very numbers chosen as cornerstones. A 4-bit result masqueraded as the bf16 headline for as long as no one read the model field on disk.

The fix generalized into the discipline this paper is built on: **every result and a per-run manifest record precision, checkpoint, graft identity, summary condition, metric-as-proxy, and split — read from the live model at run time, never inferred from a directory name.** A bare "bf16" attached to a number by convention rather than by measurement is a landmine. Recording the intervention identity per arm is what lets this paper state, cleanly, that the graft named "tuned" on the coding task is a flat-α graft and that the tuned-config graft is a separate, distinctly-recorded arm — a distinction an earlier draft blurred.

## 8. The recurring failure: chasing the number that still looked alive

Several apparent headlines appeared over the project and each dissolved when its own controls finally ran. The common thread was pivoting toward wherever a number still looked alive instead of concluding.

- **Estimator artifact.** The first strong recovery number came from a ratio estimator (recovery divided by the full-vs-compacted gap) that is unstable near small denominators. On a robust log-probability difference it shrank to a qualitative pattern. A separate "keys actively hurt" claim was the same artifact with the sign flipped. A headline lived for days on an unstable estimator. Robust estimators and the correct clustering unit should be chosen before the first result, not after.
- **Controls run late.** The placebo battery and the wrong-source controls — exactly the tests that gate a causal claim — were added after headlines were already drafted, and they came back adverse: recovery null against the baseline, content-agnostic where a mechanism had been claimed. Had they run first, the headlines would not have been written. In this final round we ran them first; that is why the result is a clean bound instead of another retraction.
- **Fragile premises built upon.** The synthetic recovery effect only reproduces under the model's own summary and its own in-context replies, and the strongest development-set version rested on twelve hand-authored conversations on a mixture-of-experts model. "The native render is the valid measurement" was asserted, and substantial re-rendering work was motivated, before the premise was checked; a fresh set of conversations did not carry the effect on the older render. Twelve conversations on one model is a thin base for a headline, and the held-out placebo-controlled test in Part I is what finally settled it — as a null.
- **Instrument-hopping.** A long-context memory benchmark came back null and was dropped for coding; coding hit a capability floor (the model rarely solves the tasks, and compaction never failed at the tested granularity, so the graft could at best do no harm) and was pivoted to synthetic chains; a chain recall probe was found to reference constants from an unrelated task family and was invalid; effort then moved to an interpretability side-line. Each hop chased the live number rather than accepting the boundary the previous one had drawn.

## 9. What would have caught each earlier

Derived from the failures, not moralized:

- **One preregistered instrument.** Fix a single measurement — this model, this precision, value-only, a real task, a pre-committed α, a placebo — and make the project *that* measurement. Instrument-hopping is the symptom of never having committed to one.
- **Controls are entry conditions, not robustness checks.** No headline until its placebo and wrong-source controls have run and passed.
- **Provenance per result, from the on-disk field.** Precision, hardware, checkpoint, and arm identity recorded and checked for every number; no cross-runtime merging or relabeling.
- **Robust estimators and the right clustering unit from the start.**
- **No standing publishing target until a result survives its own controls.** Prebuilt publication infrastructure creates sunk-cost pressure to keep *some* headline alive and to relabel rather than retract.
- **Verify a premise before building on it.** "The native render is required" drove large work while unverified; it turned out to be true as a mechanism and irrelevant as a rescue, because the effect it enabled was null held-out.

## 10. Status and future work

**Status.** There is no robust bf16 value-only *recovery* effect on held-out synthetic conversations: the graft is content-specific and non-destructive but statistically indistinguishable from plain compaction, and it does not strengthen as compression increases. There is one small, replicated positive — next-action log-probability on real coding trajectories under brief summaries, about +0.013 nats/token — bounded to that regime and measured as a proxy. A related observation — that retaining write-time keys *and* values in a packed layout makes a compacted model admit uncertainty rather than fabricate, though without restoring the evicted facts — is a separate, layout-driven line, not a value-graft claim, and is not part of this result.

**If this were carried further**, the next steps follow the one live margin, not the dead ones: measure the coding effect as *task success* (apply the action, run the tests, report resolve rate) rather than as log-probability, since a proxy positive is only worth pursuing if it moves the real thing; and characterize the terse-summary regime directly, since that is the only place the write-time state carried anything the text did not. We would not invest further in the synthetic recovery probe as a predictor of real-task behavior — one of the firmer conclusions here is that it does not track it.

---

## Appendix A — Terminology

We use plain descriptive terms rather than a coined name, because the result does not amount to a technique worth minting vocabulary for. For reference, the local labels used above:

- **Value graft** — the intervention under study: at a compaction boundary, replace the value vectors (not keys) at summary-token positions in the fresh compacted cache with a blend toward the model's original write-time values.
- **Compacted baseline** — the production layout the graft must beat: system prompt, self-generated summary as a context note, verbatim recent tail, freshly re-encoded.
- **Recovery** — teacher-forced log-probability of the shared gold continuation under the graft minus under the compacted baseline (a proxy, in nats/token).
- **Content-specificity** — the same log-probability difference, graft minus a placebo graft (shuffled or noise-matched source values in the same slots).
- **Champion / tuned graft** — a per-layer or per-head configuration of which slots to graft and at what α, tuned on a disjoint validation set.

## Appendix B — Exact configuration

- **Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` (non-thinking instruction variant), bfloat16, read from a live parameter tensor at run time.
- **Held-out evaluation set:** conversations c07–c24 (18 conversations, 203 plants), disjoint from the validation conversations used for tuning.
- **Graft:** value-only (keys untouched, α_K = 0), aligned by positional-within-region difflib match; flat α = 0.75, or a per-layer α map (27 layers at α ∈ {0.75, 1.0}) / per-head slot mask for the tuned variants.
- **Summary:** self-generated by the test model; a fixed foreign summary suppresses the effect and is not used for headline numbers.
- **Placebos:** position-shuffle, cross-probe-shuffle, energy-matched Gaussian noise (per-layer norm-matched).
- **Estimator:** percentile bootstrap 95% CI resampled over conversations (10,000 resamples); ratio estimator retired.
- **Compression levels:** ultra / brief / medium / realistic, with measured compression ratios 0.009 / 0.037 / 0.083 / 0.259.
- **SWE-Gym:** 75 real OpenHands trajectories, cut at ~75% before an assistant action, teacher-forced next-action mean log-probability; brief and realistic summaries; plain and tuned grafts.
