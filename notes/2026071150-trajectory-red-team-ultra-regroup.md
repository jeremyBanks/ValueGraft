# Independent trajectory red-team after ultra-depth regroup

**Reviewer:** Independent OpenAI Codex subagent; the runtime model identifier was
not exposed to Sol  
**Requested by:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11  
**Scope:** Read-only red-team of the project trajectory, current paired-v11 plan,
corpus contract and drafts, schedule primitives/tests, relevant incidents, audits,
and result artifacts. The review starts from the owner's original practical
question rather than sunk cost.

## Executive verdict

A clean N=12 v11 study is not the highest-decision-value next study.

The current plan is scientifically conscientious but overengineered for an
untested mechanism. It spends substantial effort constructing twelve
5,000–8,000-token, token-exact triple histories before learning whether the exact
30B model shows even a large, directionally coherent summary-state effect on one
case.

The recommendation is:

1. Pause expansion of the current v11 corpus.
2. Preserve completed drafts as engineering fixtures.
3. Run a $0 tiny-model end-to-end decomposition.
4. Then run a 1–3-case exact-30B canary, capped around $3–$5.
5. Author an independent confirmatory N=12 corpus only if that canary shows a
   large, interpretable channel.
6. Begin the corrected negative/methodological paper now in parallel.
7. Do not prioritize live coding-agent treatment evaluation yet; under the
   remaining budget it would be an engineering smoke, not credible performance
   evidence.

This is not an argument that N=12 is poorly designed as a confirmatory study. It
is an argument that confirmation is premature.

## Severity-ranked findings

### 1. Critical: the all-twelve-before-any-forward-pass rule is poor decision economics

The contract's strongest anti-adaptation rule—freeze and review all twelve long
cases before even a local semantic forward pass—protects a hypothetical
confirmatory claim but maximizes wasted work if:

- summary rows carry no usable signal;
- the production summaries include the target and eliminate compaction damage;
- forced counterfactual summaries are badly off-distribution;
- the new runner fails on the exact model;
- P/O schedule interactions swamp the effect;
- the completed corpus fails diversity review.

A discovery canary should precede confirmation. The 1–3 canary cases must not
later count toward the confirmatory N, and no confirmatory configuration should
be tuned on them. That cleanly separates exploration from confirmation without
constructing twelve elaborate cases blindly.

### 2. Critical for the specificity claim: `GMC` fixes an off-support mediator

The correct summary is freely generated under the correct history, then forced
under the counterfactual history. Formally, this is a controlled direct effect at
fixed summary text, so it is not mathematically meaningless. But it may violate
practical overlap badly.

If the summary says or implies the correct decision, forcing it after a history
that establishes the opposite creates a focal contradiction. `C-W` can then
reflect:

- history-summary consistency versus inconsistency;
- forced-token surprisal;
- generic conflict resolution;
- the focal hidden fact itself.

Focal selectivity does not solve this: the contradiction is itself focal and
therefore can selectively move the focal target.

Merely reporting forced-summary NLL is insufficient. Stronger options are:

- require the focal answer to be absent semantically from the summary before
  outcome scoring;
- require adequate summary likelihood under both histories;
- use both naturally generated summaries in a balanced history × summary-text
  crossover;
- or use a target-independent common carrier string as an engineered channel
  assay.

The balanced design is especially valuable: generate `s_correct` and `s_wrong`,
then force each under both histories. A credible history-state effect should
reverse direction with history while summary text is held fixed, not appear only
when the correct summary is paired with its native history.

### 3. Critical for practical interpretation: P is replay, not live conversation state

`turn_aligned_replay` is accurately named in the new module, but it is not how a
live agent cache was written.

P prefills every historical assistant message as one block. In a real session:

- user/tool messages are prefills;
- assistant replies are written token-by-token during generation;
- call boundaries and query shapes differ throughout the conversation.

The project has now directly established that query shape changes deterministic
bf16 state at effect-scale magnitude. Therefore the generated-versus-prefilled
distinction cannot be waved away.

A positive v11 result would support:

> A history-conditioned summary-row channel after an explicitly defined replay
> schedule.

It would not establish that the same effect exists in a live agent's native
cache. Testing that requires retaining the subject model's actual conversation
cache during native generation.

P and O are useful benchmark conditions, but neither is the ecological serving
condition.

### 4. High: the emerging corpus appears to reproduce a shared scaffold

The completed c01, c04, and c05 drafts vary in domain and length, but all are long
"working decision record" conversations with the same basic rhythm:

- user supplies operational facts;
- assistant produces polished exhaustive planning prose;
- two selected facts are established early;
- later discussion walks through budgets, schedules, risks, rights, and action
  lists;
- retained material is deliberately generic about the focal facts.

All three identify the author as Sol/gpt-5.6. C04 and c05 explicitly list no focal
downstream-reference messages. Independent passes by the same model can provide
fresh attention, but they do not create strong author/model diversity.

This does not make the cases invalid. It narrows them to an engineered
planning-dialogue benchmark and creates exactly the surface-template risk the
contract says to reject.

Run a blind diversity/scaffold review now, on the first completed cases, instead
of after all twelve are authored. If they fail, stop expanding this family.

### 5. High: summary leakage and headroom are not adequately gated

The summary request explicitly asks for "decisions that still affect next steps."
The referent plants are often explicit decisions; the sense plants are operational
definitions. A capable summarizer may include them.

If the visible summary already supplies the target:

- F may have little damage;
- C-F can become a ceiling-limited numerical comparison;
- the experiment no longer chiefly tests information absent from text.

The current contract says target inclusion will be "measured and reported," not
used as a design criterion. That is acceptable for a broad same-text state
question, but not for the stronger "state preserves information omitted by the
summary" claim.

Before treatment outcomes, classify summaries with a blinded semantic-leakage
audit and establish:

- full-history competence;
- compacted damage/headroom;
- whether each focal answer is explicit, paraphrastically recoverable, or absent.

A mitigation experiment without observed damage cannot answer whether anything
was repaired.

### 6. High: the outcome can improve while model loss gets worse

The proposed margin

`correct-target mean logprob - counterfactual-target mean logprob`

is useful, but an increased margin can arise by suppressing the counterfactual
while leaving the correct target unchanged or worse.

The design needs:

- full-history correct and counterfactual oracles;
- fresh-compacted baselines;
- correct-target log probability separately;
- counterfactual-target log probability separately;
- generated answer correctness;
- recovery relative to observed A-F damage.

A positive margin alone should not be called "less loss" or improved performance.

### 7. High: full-KV success would not vindicate naive value copying

The coherent full-KV condition tests whether a usable state channel exists. The
owner's original intervention was naive value copying.

These must remain separate conclusions:

- `C(K+V) > F`, `V-only <= F`: the channel exists, but naive V-only mixing fails.
- `V-only > F` and separates from controls: evidence for the original
  intervention.
- `C(K+V) <= F`: no usable summary-row channel detected in this assay.

V-only and `C-V` should therefore be prespecified, not treated as optional
diagnostics after a positive full-KV result. K-only can remain secondary.

### 8. High: N=12 cannot sustain the proposed intersection-heavy rhetoric

At N=12 conversation clusters, requiring positive lower 95% bounds for both `GF`
and `GMC` under both P and O creates a four-way conjunction before category,
focal-selectivity, V-only, and leakage analyses.

That rule is conservative, but a failure will be almost uninterpretable:

- no effect;
- insufficient power;
- one-schedule numerical interaction;
- counterfactual off-support;
- one category null;
- corpus heterogeneity.

Choose one primary replay schedule and one primary aggregate contrast family.
Treat O as a prespecified sensitivity analysis and report the interaction.
Dual-schedule survival is stronger evidence, but should not be the sole binary
success criterion.

More importantly, neither P nor O proves live-session robustness, so demanding
both is costly without solving the ecological limitation.

### 9. Medium-high: exact per-message token matching is justified but disproportionately expensive

Exact message widths are useful for matched P call boundaries and summary
positions. But optimizing long prose for exact token counts has already produced
unnatural controls once and creates substantial opportunity for hidden editing
artifacts.

For the confirmatory study, mechanical exactness plus blind review is defensible.
For a discovery canary, use much shorter histories. A 1,000–2,000-token causal
assay can test whether summary rows carry a history-specific channel before paying
the complexity tax of 5,000–8,000-token compaction realism.

### 10. Medium-high: the implementation is still only primitives

The new schedule module correctly derives P from message starts, O from
4,096-token chunks, and records explicit schedules. Its tests verify widths,
positions, failure behavior, and mocked generated/forced traces.

They do not yet validate:

- a real end-to-end model execution;
- the gapped destination;
- full-KV, V-only, K-only insertion;
- correct/wrong crossover;
- target scoring;
- persistence and independent harvest;
- exact production revision/backend lineage;
- live generated-versus-forced equality on the real stack.

Given the project's incident history, the distance from these primitives to a
trusted paid N=12 runner is material.

## Why the smaller canary has higher decision value

The recent preprint already provides evidence that downstream cache rows can
carry computed conclusions. The novel uncertainty here is narrower:

- Are generated summary rows such carriers?
- Does transplanting their coherent K+V help after compaction?
- Does V-only transplant preserve that benefit?
- Does the effect survive ordinary numerical schedule sensitivity?

A 1–3-case exact-model decomposition can answer whether there is a large signal
worth confirming. It cannot establish prevalence or a population effect, but that
is not its role.

The canary should include, at minimum:

- `A_correct`: full correct history;
- `A_wrong`: full counterfactual history;
- `F`: fresh compacted same-summary state;
- `C`: coherent correct-history full K+V;
- `W`: coherent counterfactual-history full K+V;
- `V`: correct-history values with fresh keys;
- optionally K-only;
- one P/O schedule sensitivity replay, not dual-schedule inference.

Require descriptive directional coherence:

- both full histories answer their own target correctly;
- compaction creates measurable damage;
- C raises the correct target itself, not only the margin;
- W moves toward the counterfactual target;
- focal movement exceeds disturbance on the unchanged plant;
- generated and forced same-schedule paths agree;
- schedule interaction is small relative to the observed channel.

If those conditions do not appear on 1–3 capable, damaged cases, that is not a
universal scientific null. It is a rational stopping result under this budget.

## Minimal decision tree

### Stage 0 — now, $0

- Pause further long-corpus expansion.
- Blind-review the first completed cases for shared scaffold and decoded quality.
- Freeze the exact scientific estimand before more implementation.
- Begin correcting the paper in parallel; do not make the paper wait for v11.

### Stage 1 — local, $0

Build a short engineered carrier assay on 0.6B:

- coherent correct/wrong histories;
- common target-independent summary/carrier tokens;
- full-KV, V-only, fresh, and wrong-state arms;
- full-history oracles;
- target-component and margin outputs;
- real end-to-end persistence and harvest.

This is an apparatus positive control, not evidence about Qwen3-30B or production
summarization.

### Stage 2 — exact 30B canary, approximately $3–$5

Run 1–3 cases once, preserving every state and render. No retries to rescue an
ambiguous result.

- Large coherent full-KV signal with history-direction reversal: proceed.
- Full-KV positive, V-only null/harmful: stop optimizing naive V; the important
  result is channel-exists/intervention-fails.
- No channel despite competence and damage: stop the synthetic mechanism program
  under the current budget.
- No headroom or severe summary leakage: corpus design failed; allow one redesign
  cycle only.
- Schedule interaction comparable to the effect: precision-limited terminus
  unless a higher-precision run receives separate funding.

### Stage 3 — independent confirmation only after Stage 2

If the canary is strong, build a new independent N=12 benchmark. Do not count the
canary cases. At that point the current corpus contract—simplified around one
primary schedule and corrected counterfactual logic—becomes appropriate.

With roughly $28.82 of the nominal total remaining after conservative Fable
estimates, cap additional experiment compute near $10 until the canary is read.
Preserve the rest for failures, final synthesis, and required paper review.

### Stage 4 — live agents, later funding or after a strong channel

A live coding-agent treatment study is not currently the highest-value next step
because:

- no treatment has cleared a coherent-state channel gate;
- the existing general Qwen subject had a task capability floor;
- Qwen3-Coder has not passed the production ladder;
- actual native-cache common-prefix forking is not implemented;
- two to six tasks would demonstrate plumbing, not performance.

If the canary is strong, a 2–4-task Qwen3-Coder common-prefix fork is worthwhile
as an engineering pilot. It must be described as feasibility only. A credible
task-effect estimate requires a later budget.

## What should be written now

The negative/methodological paper should begin now. The existing record already
supports a substantial contribution:

- early apparent positives were render-, source-, and estimator-sensitive;
- the scalar coding proxy failed independent replication;
- a selected layer map produced a small likelihood lead but no task-success
  evidence;
- schedule/query shape changes deterministic bf16 cache state at the scale of
  the hypothesized effect;
- synthetic equivalence fixtures and token-exact controls can pass while being
  scientifically unrepresentative;
- the fail-closed process caught several invalid designs before paid semantic
  execution.

A future canary or v11 result can be added. It should not hold the current honest
paper hostage.

## Evidence that would change this recommendation

Completing a confirmatory N=12 v11 becomes justified if an exact-30B canary shows
all of:

1. clear full-history competence and compaction damage;
2. a sizeable `C-F` gain driven by increased correct-target likelihood;
3. counterfactual state moving toward the counterfactual target, not merely
   producing generic degradation;
4. focal selectivity over the unchanged plant;
5. acceptable summary overlap or a balanced history × summary crossover;
6. V-only behavior measured separately;
7. schedule interaction clearly smaller than the state-channel effect;
8. no severe leakage or decoded-corpus failure.

A live-agent pilot becomes justified if, additionally:

- Qwen3-Coder passes its exact ladder;
- full context succeeds and ordinary compaction demonstrably hurts on
  capability-matched tasks;
- native incremental cache forking is observed to work;
- environments and deterministic tests survive a two-task smoke.

Absent those observations, the most sensible endpoint is the corrected
negative/methodological paper—not another broad rescue search.
