# Ultra-depth regroup: decision on the next experiment

**Decision owner:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11  
**Status:** binding strategy decision before any semantic v11/v12 model forward

## Bottom line

Do **not** complete the twelve-case paired-v11 corpus now. Preserve its six
draft/checkpoint files as non-executable engineering work, but stop expanding
that family. The current contract describes a defensible confirmatory benchmark;
it does not establish that building that benchmark is the best next use of time
or money. Confirmation is premature because the exact 30B subject has never
shown even a large, clean same-text summary-state effect under the corrected
apparatus.

The next experiment will be a separate, explicitly exploratory **decision
canary**. Its stimuli can never enter a later confirmatory sample. It will first
exercise the complete apparatus locally, then spend at most a few dollars on the
exact bf16 30B checkpoint. Only a large, history-directional, focal-selective
signal authorizes a new independent confirmatory corpus. In parallel, begin the
corrected negative/methodological paper from evidence that is already stable;
the paper must not be held hostage by another rescue search.

This decision follows my own reconstruction of the estimand and apparatus,
informed but not determined by the independent reviews in:

- `notes/2026071187-trajectory-red-team-ultra-regroup.md`;
- `notes/2026071188-causal-statistical-blackboard-ultra-regroup.md`.

## What I personally rechecked

1. The first two realistic v10 schedule fixtures fail exact equality by amounts
   comparable to or larger than the hypothesized semantic effect. The c10 origin
   diagnostic isolates deterministic query-shape arithmetic in the tested local
   bf16 path, not future-token leakage.
2. The old seven-length schedule gate was one periodic five-token input, and the
   old wrong-history control was repetitive corruption. Neither can license the
   intended scientific claim.
3. Four exact-width c02/c10 counterfactuals passed mechanical geometry and still
   failed decoded review. Exact token geometry is necessary for matched schedule
   contrasts but is not evidence of semantic validity.
4. The current v11 implementation consists of source-schedule primitives and
   tests, not a trusted end-to-end runner. It has not exercised a real gapped
   destination, the full arm set, persistence, independent harvesting, or exact
   30B lineage.
5. The completed c01/c04/c05 drafts are careful, but they share an engineered
   planning-dialogue scaffold and one author/model voice. They are suitable
   discovery fixtures after review, not evidence of broad ecological coverage.
6. `turn_aligned_replay` prefills each historical assistant response as a block.
   A live assistant writes response content token by token. Because this project
   has now observed query-shape effects at the estimand scale, that difference is
   not a harmless implementation detail.
7. The proposed summary request asks for decisions affecting next steps. A good
   summary may explicitly preserve the two plants, removing the damage/headroom
   needed for a mitigation claim.
8. The proposed focal margin can improve by suppressing the wrong target while
   the correct target becomes no more likely or even less likely. Margin alone
   is not "less loss."
9. MEMENTO already reports a same-text retained-versus-recomputed memento-KV
   accuracy difference in trained models. *Models Take Notes at Prefill* reports
   causal conclusion storage on downstream aggregator/delimiter positions in
   ordinary models. These priors make a channel plausible, but also narrow the
   unanswered question and warn against testing summary content rows as though
   they were the only possible carrier.

## The question that remains worth answering

The literature substantially answers the broad existence question: a generated
text span's retained KV state can carry useful information absent from a fresh
encoding. This repository's remaining high-value questions are narrower:

1. Under an ordinary, untrained instruction checkpoint, do summary/boundary rows
   carry **history-specific state that causally changes a later decision while
   visible carrier text is held fixed**?
2. Does retaining those rows improve a damaged compacted context relative to a
   fresh same-token state?
3. Does the owner's original training-free **value-only** intervention preserve
   any of that useful channel, or do keys/interactions/downstream aggregation
   make naive value copying ineffective?
4. Where is the carrier: summary content, the natural closing boundary, or an
   explicitly added small downstream anchor?

These are distinct claims. Full-KV success does not vindicate value-only
copying. A clean history contrast does not itself establish deployment utility.
A small engineered carrier effect does not establish live-agent performance.

## Replacement: exploratory decision canary

Create a new additive experiment version rather than weakening the frozen v11
contract. V11 remains a preserved, unexecuted confirmatory design candidate.

### Discovery stimuli

Use two deliberately separate strata, analyzed separately rather than pooled as
if they were one population:

1. **Engineered carrier stratum:** several short, independently worded paired
   histories (roughly 1,000--2,000 tokens) in which one rule/fact changes a
   derived decision. Use a common target-neutral carrier compatible with both
   histories. This is the clean causal channel assay and apparatus positive
   control, not an ecological conversation benchmark.
2. **Conversation stratum:** one to three decoded-reviewed planning/dialogue
   discovery cases, including at most one existing v11 draft and at least one
   structurally different independently authored case. Generate normal compact
   summaries and measure leakage, competence, and damage. These are case studies,
   not an inferential N.

Canary cases and every decision made from them are permanently exploratory. If
confirmation is authorized, its cases, authorship passes, targets, and analysis
are new and frozen without reference to canary outcomes beyond choosing the
already prespecified branch of the decision tree.

### Primary source schedule

Do not use current P as the sole primary protocol. Implement a more faithful
**role-native stepwise replay** schedule:

- system/user/tool additions are prefilled as the corresponding message block;
- historical assistant content is forced one token at a time;
- structural close/open tokens are appended in the order a persistent chat
  cache would receive them;
- the summary is generated or forced one token at a time.

For a fixed transcript, forcing an assistant's observed token IDs through the
same one-token calls computes the same cache states as selecting those IDs during
generation; token authorship remains imported and must be disclosed. This is
still replay, not a genuinely live agent trajectory, but it removes the largest
known query-shape mismatch in P.

Correct and counterfactual histories must have identical roles, per-message
token widths, call widths, positions, and carrier IDs under this schedule. The
current turn-aligned P or ordinary O schedule may be a prespecified sensitivity
condition on a small subset. Neither is a co-primary gate and neither may rescue
a failed primary schedule.

### Balanced visible-text design

For engineered cases, use one common target-neutral carrier and verify that its
forced NLL is adequate and similar under both histories.

For ordinary-summary cases, do not rely only on the summary naturally generated
under the correct history. Generate `s_correct` and `s_wrong`, then force each
exact token sequence through both matched histories. This history-by-summary-text
crossover separates a state effect within fixed visible text from an asymmetric
"native summary versus contradictory forced summary" effect. Before target
scoring, blind-label each summary/plant `target-neutral`, `partial`, or `explicit`
and retain the forced-token NLLs. Explicit/low-overlap cells remain reportable
but cannot carry the clean omitted-information claim.

### Prespecified retained regions

The canary is the proper place for a small localization factorial, frozen before
outcomes:

1. generated/forced summary content rows only;
2. content plus the canonical assistant closing boundary rows computed before
   eviction;
3. content plus closing boundary plus a short fixed target-neutral anchor
   appended before eviction.

The third region is an engineered method, not evidence for naive content-only
copying. A content null with a boundary/anchor positive would be an informative
localization result and a new design direction, not a rescue of the old claim.

### State arms

At minimum retain and score, from persisted source rows:

- full-history correct and counterfactual oracles;
- fresh compacted state;
- correct and counterfactual full K+V state;
- correct and counterfactual value-only state using identical fresh keys;
- K-only and a norm/magnitude-matched nonsemantic perturbation as secondary
  decomposition controls.

If destination scoring is cheap relative to source capture, prefer the frozen
3x3 key-source by value-source grid (`fresh/correct/wrong`) over an arbitrary
handful of combinations. It exposes K/V interactions without additional long
source passes. It remains exploratory and must not create post-hoc primary
contrasts.

### Outcomes and causal contrasts

Persist raw target-token log probabilities. Report separately:

- correct-target mean log probability;
- counterfactual-target mean log probability;
- their margin;
- deterministic generated answer;
- full-history competence;
- fresh compaction damage;
- forced-summary NLL and leakage class.

The clean semantic contrasts are same-schedule and same-shape:

- full-KV correct history minus full-KV counterfactual history;
- value-only correct history minus value-only counterfactual history.

Fresh comparisons are utility contrasts. They intentionally compare retained
state with a compacted restart path and therefore include execution-policy
differences. They cannot alone identify semantic state.

Every focal history intervention is also scored on an unchanged control fact.
The focal movement must exceed non-focal disturbance. A favorable margin is not
called reduced loss unless the correct target itself also becomes more likely.

### Technical gates

Before paid semantic scoring, the local end-to-end runner must observe:

- exact checkpoint/tokenizer/config binding in its local test subject;
- exact source/destination token and logical-position coverage;
- identical correct/wrong call geometry;
- generated-versus-forced identity under the same schedule;
- self-replacement identity;
- engineered perturbation downstream sensitivity;
- correct row/region confinement for K, V, and joint replacements;
- durable save/resume of every render, carrier, row hash, raw score, and trace;
- an independent harvester that reconstructs the arm lineage and primary
  contrasts from committed artifacts.

The 0.6B local run licenses apparatus completeness only. It is not a semantic
negative control for the 30B model. The paid canary must include one independently
specified downstream-note positive control on the exact 30B subject so a failed
summary result is not confused with a dead intervention/readout path.

## Decision and stopping rules

1. **Technical failure:** repair at most two bounded implementation defects with
   no semantic interpretation. A repeated failure triggers a fresh design audit,
   not a long grind or a waived gate.
2. **Inadequate estimand:** if the model lacks full-history competence, the
   carrier is grossly off-support, or ordinary summaries leave no compaction
   damage, do not interpret treatment efficacy. Permit at most one prospectively
   documented stimulus redesign cycle.
3. **No channel:** if the exact-model positive control works but correct versus
   counterfactual full-KV state shows no large, directionally coherent,
   focal-selective movement across the engineered and usable conversation
   cases, stop the summary-state program under the current budget. This is a
   bounded decision result, not proof of universal absence.
4. **Full-KV only:** if joint K+V shows a channel but value-only does not, stop
   optimizing naive value grafts. Report that an implicit channel exists in the
   assay but the proposed training-free split does not exploit it. A direct
   full-KV-minus-value-only contrast is required before claiming the two differ.
5. **Boundary/anchor only:** if content rows are null but closing/anchor rows are
   selective, pivot to the explicitly engineered carrier method. Do not label it
   a success of content-only value copying.
6. **Value-only promising:** only a value-only result that improves utility,
   follows matched history, exceeds non-focal disturbance, and raises the
   correct target authorizes an independent confirmatory benchmark. Descriptive
   canary direction is authorization, not a paper claim.
7. **Numerically fragile:** if schedule-condition movement is comparable to the
   semantic contrast, terminate broad causal rhetoric at a schedule-specific or
   precision-limited finding unless separately funded evidence resolves it.

No canary p-value or confidence interval will be presented as confirmation. A
future purposively authored N=12 benchmark would itself detect only large effects
and support a fixed-benchmark claim, not general population prevalence.

## Budget and execution order

The conservative ledger attributes roughly `$31.18` of the approximately `$60`
ceiling to reported Fable provider usage estimates, although the actual
incremental cash charge is unverified. Treat the remaining roughly `$28.82` as a
hard conservative envelope.

1. `$0`: finish this design review; preserve the paused corpus; implement and
   test the new end-to-end exploratory runner; run the local technical ladder;
   start the corrected paper scaffold.
2. Paid canary: launch no pod until a local sealed artifact and independent
   harvester pass. Use one observed one-case forecast, then cap the complete
   exact-30B canary at `$5`. Save and commit every render/state/result. Stop the
   pod before analysis/writing.
3. Reserve at least `$15` for final synthesis/review and failures. Do not spend
   the remainder on twelve-case confirmation without a strong canary and a
   revised budget forecast. If confirmation would cross the owner-authorized
   ceiling, return with the observed canary and a concrete funding decision.

This cap is a ceiling, not a target. No paid compute has yet been observed for
the coherent-state experiment.

## Real coding-agent evaluation

A live-agent treatment study is conditional, not the next step. Existing clean
chain tasks had a compacted ceiling; SWE-bench tasks had a model capability
floor; SWE-Gym is an offline teacher-likelihood/action proxy rather than executed
success. Running more tasks in those regimes will not repair the identification
problem.

If a usable value-only or full-KV treatment clears the canary and later
confirmation, the practical first study is a common-prefix fork:

1. use an open-weight coding model that passes its exact build ladder;
2. run a real tool-using trajectory to a frozen compaction point while retaining
   the actual incremental cache;
3. snapshot the repository/environment and generate one summary;
4. branch identical sandboxes into full context, ordinary fresh compaction,
   retained full-KV, and retained value-only conditions;
5. continue with all tool traffic and renders saved;
6. score deterministic tests, completion, steps/tokens, and safety failures.

Two to six capability-matched tasks are an engineering feasibility pilot only.
Approximately twenty or more varied task clusters would be needed before a
practical performance claim, and even that count needs power planning from
observed pilot variance. Before any treatment task, the model must pass a
no-treatment gate: full context succeeds and ordinary compaction measurably
hurts. The task must be neither ceiling nor floor.

## Paper consequence

Begin the corrected paper in parallel. Its stable spine does not require a
positive canary:

- apparent recovery effects were fragile to render, metric, source provenance,
  and precision;
- value-only bf16 headline tests were null or tiny proxies;
- a coding likelihood lead did not become live task success;
- schedule/query shape changes deterministic bf16 cache state at the effect
  scale;
- exact validators can pass pseudoreplicated or semantically invalid fixtures;
- multiple invalid mechanisms were caught before paid semantic execution;
- prior art establishes that latent KV channels can exist, sharply bounding what
  this project can still claim as novel.

The paper should lead with the honest negative/bounding and methodological
result unless new evidence earns a different framing. Fable should lead the
fresh narrative draft once the canary branch is resolved; Sol retains final
authority over factual, causal, and methodological claims.

## What would change this decision

I will authorize a new independent confirmatory corpus only after observing on
the exact 30B checkpoint:

1. working technical and downstream-note positive controls;
2. full-history competence and nontrivial fresh-compaction damage;
3. a sizeable same-text correct-versus-counterfactual state effect;
4. movement toward the source history's target rather than generic degradation;
5. focal selectivity over the unchanged control fact;
6. correct-target likelihood improvement for a utility claim;
7. separately resolved full-KV and value-only behavior;
8. an effect materially larger than schedule sensitivity;
9. acceptable carrier support/leakage or a balanced crossover that exposes the
   limitation.

Until those observations exist, more long-corpus authoring is not rigor. It is
premature confirmation work.
