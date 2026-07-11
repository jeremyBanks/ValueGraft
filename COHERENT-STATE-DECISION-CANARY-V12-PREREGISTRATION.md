# Coherent-state decision canary v12 — preregistration

**Design ID:** `coherent-state-decision-canary-v12`  
**Author/decision owner:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11  
**Status:** **DRAFT UNDER INDEPENDENT AUDIT — NO MODEL FORWARD AUTHORIZED**

This is a new additive exploratory experiment. It does not amend, waive, or
reinterpret v10, and it does not execute the frozen paired-v11 confirmatory
contract. V10 remains permanently non-authorizing. V11 remains paused with its
drafts non-executable.

The strategy record is in `notes/2026071189-sol-ultra-regroup-decision.md`; the
Fable review and Sol's disposition are in `notes/2026071190` and
`notes/2026071192`. This document freezes the literal experiment that follows
from those decisions. Until its status is changed additively to `FROZEN`, it
authorizes tokenizer-only construction and unit tests, but no local or paid
model forward.

## 1. Purpose and claim boundary

The canary decides whether a new independent confirmatory experiment has enough
expected value to propose. It cannot establish an efficacy or population claim.
Its stimuli, outcomes, thresholds, and any implementation choices learned from
them are permanently excluded from a later confirmatory sample.

The questions are:

1. Does a standard untrained instruction checkpoint write history-specific
   information into a fixed visible summary/boundary carrier?
2. Does retaining that state improve a damaged compacted continuation relative
   to a fresh same-token state?
3. Does value-only retention preserve a useful channel, separately from full
   K+V retention?
4. Is the state localized to summary content, the natural assistant boundary,
   or a deliberately appended neutral downstream anchor?

The canary is not evidence about a production continuously batched server, a
live organic coding agent, arbitrary models, arbitrary summaries, or broad
conversation traffic. The primary protocol is this repository's exact q=1
incremental loop over fixed token IDs.

## 2. Exact subjects and environments

### Local apparatus subject

- Model: `Qwen/Qwen3-0.6B` from the already cached local Hugging Face snapshot.
- Device: CPU.
- Dtype: `torch.bfloat16`.
- Attention: resolved eager on every layer.
- Purpose: technical completeness only. No semantic result from this subject
  can authorize or discourage the exact-model canary.
- The resolved revision/hash, tokenizer files, Transformers/PyTorch versions,
  model config, layer geometry, and backend enumeration must be persisted.

### Exact scientific subject

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Revision: `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.
- Dtype: bf16; no quantization.
- Attention: eager requested and resolved/attested for every decoder layer.
- Temperature: 0; greedy summary generation.
- Device: one paid NVIDIA GPU environment capable of loading the exact model.
- Transformers and repository commit must be identical between the sealed
  preflight packet and run. No unrecorded dependency upgrade is allowed.

The pod gate is not inherited from a local pass. A required exact-stack subset
runs and persists before any semantic treatment score.

## 3. Stimulus strata and independence

### 3.1 Engineered carrier stratum

Freeze six complete paired cases before any treatment outcome:

- primary cases `e01`--`e04`;
- ambiguity-reserve cases `e05`--`e06`, run only under Section 14's trigger.

E01--e03 contain roughly 1,000--2,000 production-tokenizer tokens before the
carrier request. E04 contains roughly 4,000--6,000. Reserve cases span different
lengths within 1,000--6,000. All six are frozen before e01 treatment unblinding.

Each case has:

- one correct history `C`;
- one minimally counterfactual history `W`;
- exactly one focal derived decision changed by `W`;
- exactly one non-focal control decision/fact left byte-identical;
- one focal probe with exact correct/counterfactual target phrases;
- one non-focal probe with its own fixed target pair;
- a retained tail after the frozen compaction boundary that is byte-identical
  and coherent under C and W;
- one fixed target-neutral carrier text compatible with C and W.

At least three independent authoring sessions spanning at least two model
families contribute, with no session authoring more than two cases. Authors do
not see model outcomes. A blind diversity review rejects a shared fill-in-the-
nouns scaffold.

These six fixed cases are not random population draws. Counts, target rows,
regions, schedules, and probes within a case do not increase N.

### 3.2 Conditional conversation stratum

This stratum runs only after a clear engineered full-KV pass. Freeze three
discovery cases before the first conversation treatment score:

- one predesignated existing v11 draft may be adapted additively after complete
  blind and target-aware review;
- at least two cases must be structurally different and independently authored;
- the three may not share one planning-dialogue scaffold or author session;
- each has one focal changed plant and one byte-identical non-focal control;
- each has a correct/minimally-counterfactual history with exact geometry and a
  retained tail.

Conversation authors are told only which already-frozen decision branch opened;
they do not see per-case, arm, layer, region, or schedule results. All cases are
exploratory and excluded from confirmation.

## 4. Decoded and mechanical eligibility

For every paired history:

- roles, message count/order, system text, retained tail, and non-allowlisted
  messages are byte-identical;
- every changed message has identical production-tokenizer width;
- canonical full-prefix length, every message start, every q=1 assistant span,
  and every call width are identical;
- the counterfactual establishes exactly the frozen alternative and repairs all
  focal downstream references;
- the non-focal chain is byte-identical;
- no repeated padding, clipped response, fake external action, special-token
  literal, dangling reference, or conspicuous balancing prose is permitted.

Before a case is executable it must pass:

1. production-tokenizer mechanical validation;
2. randomized blind singleton review of both histories;
3. target-aware paired review;
4. cross-case scaffold/diversity review;
5. third-reviewer adjudication for any disagreement.

Review failures produce new additive candidates and hashes. Never edit a
reviewed snapshot in place or delete a failed candidate.

## 5. Carrier texts and maximal visible carrier

### Engineered neutral carrier

Every engineered case uses this exact assistant content as the common carrier:

> The prior discussion established the operating context and resolved one local
> decision. Continue from this handoff, preserve the existing constraints, and
> answer later questions from the state available here. No unresolved action is
> introduced by this note.

It must be blind-verified not to state, imply, or favor either focal target or
the non-focal target.

### Conversation summaries

For each conversation case, the subject greedily generates summaries under C
and W from this exact request:

> Write a compact handoff note of 180–240 words for another assistant who will
> continue this conversation without seeing it. Identify the participants and
> their roles, current goals, decisions that still affect next steps,
> constraints, unresolved questions, and the next actions. Prioritize what the
> next assistant needs to act; compress settled background details when they are
> not needed. Use plain prose or concise bullets. Do not add commentary before
> or after the note.

Also freeze one externally authored target-neutral summary containing only
shared non-focal information and equally coherent under C and W. Its author sees
the two histories but no subject state or outcome. Blind reviewers label every
summary/plant `target-neutral`, `partial`, or `explicit` before treatment
unblinding.

### Fixed downstream anchor exchange

Every source and destination arm contains this exact visible exchange after the
carrier assistant message closes:

- user: `Acknowledge receipt of this handoff without adding or repeating any factual detail.`
- assistant: `Acknowledged.`

The exchange is present even when its rows are fresh. It must not contain or
favor any target. The maximal source carrier is:

1. assistant summary/carrier content;
2. the exact canonical assistant close suffix;
3. the exact canonical anchor user message, assistant header/content, and close.

## 6. Retained regions

All regions use identical visible tokens and positions. They vary only the rows
replaced by source rows; all subsequent rows are causally recomputed.

- `R1_content`: carrier assistant content only.
- `R2_boundary`: R1 plus the canonical suffix that closes that assistant
  message, ending before the anchor user's message begins.
- `R3_anchor`: R2 plus the complete fixed anchor exchange.

`R2_boundary` is the sole stop/go region. R1 and R3 are descriptive localization
conditions. They cannot rescue a failed R2 decision or become primary from their
outcomes.

## 7. Role-native q=1 source schedule N

The primary source protocol is `N = role_native_q1_replay`:

1. Render the entire alternating transcript canonically using the Qwen dummy-
   user/non-final method; derive boundaries from exact special-token markers,
   never from separately tokenized text concatenation.
2. Prefill the initial system/user material and assistant generation header in
   the exact canonical order.
3. Force each historical assistant content token in a separate one-token model
   call with explicit logical `position_ids` and contiguous physical
   `cache_position`.
4. Prefill the canonical assistant close plus the next user message and next
   assistant header as the next message addition.
5. Continue through the history, carrier request, generated/forced carrier,
   canonical close, and anchor exchange.

For fixed IDs, q=1 forcing computes the same state as q=1 greedy selection
conditional on those IDs. It does not make imported authored text subject-native
and does not emulate another serving engine's batching policy.

Correct and wrong N branches must have exactly identical call widths and
positions. The exact generated-C carrier replayed under C must be bit-exact or
within the frozen same-stack numerical identity threshold on every saved row and
token log probability; the threshold must be justified by a repeated same-
schedule measurement before this document becomes FROZEN.

## 8. Schedule sensitivity P

`P = turn_aligned_replay` uses the committed v11 derivation: one conceptual
block per canonical message, internally split only above 4,096 tokens, with the
carrier forced q=1. P runs on the same exact 30B stack for every engineered
primary case's R2 full-KV correct/wrong contrast. P is a numerical yardstick,
not a replicate or alternative primary.

For history contrast `D`, define `Q = abs(D_N - D_P)`. A clear N decision
requires `abs(D_N) >= 3 * Q` and the prespecified direction under N. The 3x rule
is an exploratory robustness heuristic. O/ordinary-4096 is deferred and may not
be added after outcomes to rescue a result.

## 9. Fresh gapped destination

For a fixed visible carrier:

- build compact physical storage containing the system, the carrier request,
  maximal visible carrier, fixed anchor exchange, and retained tail;
- assign the carrier/anchor the exact logical positions of the corresponding
  full-history source rows;
- keep `cache_position` contiguous in compact physical storage;
- force fresh carrier content q=1 and append close/anchor in the same exact call
  pattern used for the source region;
- fork at each exact region boundary, replace only selected rows, then causally
  recompute all following visible tokens and the retained tail;
- append each probe and teacher-force its exact target phrases from separately
  rebuilt immutable arm snapshots.

The fresh destination comparison is a deployment-utility contrast. Its compact
prefix differs from the long source trajectory by construction; it is not the
sole semantic control.

## 10. Source states and arm grid

For each history/carrier/schedule/region persist source K/V rows from:

- `F`: fresh compacted carrier;
- `C`: correct-history carrier;
- `W`: minimally counterfactual-history carrier.

At destination, run the complete 3x3 grid with key source first and value source
second:

`FF, FC, FW, CF, CC, CW, WF, WC, WW`.

The preregistered primary families are:

- full-KV: `CC` versus `WW`;
- value-only: `FC` versus `FW`.

`CF/WF` (K-only), crossed `CW/WC`, and interactions are secondary. `FF` is the
fresh utility baseline. Every replacement must be bit-exact in selected rows and
bit-identical to fresh outside them before downstream recomputation.

Add one deterministic nonsemantic V control per case/region. For every layer and
selected token row, match the L2 norm of the `V_C - V_W` delta, then apply a
frozen seeded orthogonal/permutation construction that is independent of target
labels. The exact algorithm and seed must be in the FROZEN version and tested
for zero target access.

## 11. Oracles, probes, and raw outcomes

Score two full-history oracles with the same visible maximal carrier/anchor:

- `A_C`: correct full history;
- `A_W`: counterfactual full history.

For each arm and probe persist:

- exact target token IDs;
- per-token and mean log probability for the C target;
- per-token and mean log probability for the W target;
- margin `Y = mean_lp(C_target) - mean_lp(W_target)`;
- deterministic generated answer and stop reason;
- complete teacher-forcing feed IDs/positions;
- immutable source/destination/arm hashes.

Never call a larger margin "less loss" unless the C-target mean log probability
also improves. Never discard the two log-probability components.

## 12. Per-case contrasts

For state family `X` (`R=full-KV`, `V=value-only`) and primary R2/N:

- `D_focal^X = Y_focal(X_C) - Y_focal(X_W)`;
- `D_nonfocal^X = Y_nonfocal(X_C) - Y_nonfocal(X_W)`;
- `SEL^X = D_focal^X - abs(D_nonfocal^X)`;
- `U^X = Y_focal(X_C) - Y_focal(FF)`;
- `LP_C^X = mean_lp_C(X_C) - mean_lp_C(FF)`.

The semantic claim direction is `D_focal > 0` and `SEL > 0`. Utility requires
`U > 0`; favorable correct-target movement requires `LP_C > 0`. The wrong-state
arm moving only by generic disruption fails selectivity.

For conversation native-summary crossover, compute these contrasts separately
within `s_C`, `s_W`, and the target-neutral summary. The neutral-summary cell is
the clean semantic comparison; native-summary cells describe state plus
history/text congruence and are never substituted for it.

## 13. Pre-treatment gate and unblinding order

Phase A may compute only:

- mechanical/review eligibility;
- summary text and exact token IDs;
- leakage labels;
- forced-token NLL/support under C and W;
- `A_C`, `A_W`, and `FF` competence/damage outcomes;
- technical identities and positive-control outcomes.

An independent validator writes and commits one of:

- `PRETREATMENT_PASS`;
- `ESTIMAND_INADEQUATE` with literal reasons;
- `INVALID_TECHNICAL`.

Treatment contrasts cannot be computed or exposed until this artifact passes.
Any single allowed stimulus redesign is triggered only by Phase-A evidence and
creates a new additive hash. No redesign may inspect a treatment source row,
arm score, layer effect, or region effect.

Pre-treatment adequacy requires:

- both full-history oracles favor their own focal target;
- the non-focal oracle remains correct under C/W;
- fresh compaction shows nonzero focal damage relative to A_C for a utility
  interpretation;
- carrier/summary generation ends normally with no embedded special token or
  cap hit;
- the target-neutral carrier is blind-approved;
- forced carrier NLLs are finite and the frozen support criterion passes.

The exact support statistic/threshold must be fixed from tokenizer-only or
pre-treatment calibration evidence before FROZEN status; no arbitrary historical
0.30-nat case exclusion is inherited.

## 14. Exact-model gates and positive control

Before semantic treatment scoring on the pod, persist and independently validate:

1. exact model/revision/tokenizer/dtype/backend/code/inventory binding;
2. deterministic same-schedule repeat on the gate fixture;
3. generated-versus-forced q=1 identity;
4. fresh self-replacement identity for R1/R2/R3;
5. selected-row exactness and non-selected-row preservation for K, V, and K+V;
6. canonical close/open rendering identity;
7. gapped physical/logical position coverage and tail recomputation;
8. deterministic perturbation sensitivity;
9. durable save/resume and independent lineage reconstruction.

Then run two label-balanced variants of a downstream-note causal positive control
adapted from the released prefill-note design through the **same gapped
destination and row replacement functions**. For each direction, the full-KV
note transplant must move the target margin toward its source conclusion and
recover at least 0.5 of the oracle-versus-stale margin gap. Failure aborts before
any canary treatment score. This threshold is an apparatus sensitivity gate,
not a semantic effect-size precedent.

## 15. Decision rules

All rules refer to R2/N and the four engineered primary cases unless stated.

### Clear engineered pass

Proceed to the conversation stratum only if:

- all technical, pre-treatment, and positive-control gates pass;
- at least three of four cases have `D_focal^R > 0` and `SEL^R > 0`;
- mean `LP_C^R > 0`;
- the aggregate full-KV N contrast clears the 3x N-versus-P yardstick.

### Clear engineered stop

Stop conversation execution and the summary-state program under the current
budget if the positive control passes but either:

- mean `D_focal^R <= 0`; or
- fewer than two of four cases have both `D_focal^R > 0` and `SEL^R > 0`.

Report the accepted residual risk: engineered short/mid-length nulls do not
logically exclude a channel unique to longer organic histories.

### Single ambiguity extension

Run frozen reserve e05/e06, once, only if:

- mean `D_focal^R > 0` and exactly two of four primary cases have positive D and
  SEL; or
- at least three have positive D and SEL but the aggregate schedule ratio is
  below 3x.

The six-case terminal decision applies the same proportional sign threshold
(at least four of six) and 3x yardstick. No second extension, arm change, target
change, or threshold change is allowed.

### Full-KV versus value-only branches

- Full-KV pass/value-only stop: conversation discovery may proceed for the
  narrower full-KV claim. No value-only confirmation or live-agent treatment is
  authorized. Difference language requires the direct paired R-minus-V contrast.
- Value-only promising: in addition to the full-KV branch, at least three of
  four primary cases must have positive `D_focal^V`, `SEL^V`, `U^V`, and
  `LP_C^V`, and the aggregate V contrast must clear the 3x yardstick. This only
  authorizes conversation discovery.
- R1/R3-only positivity cannot change the R2 branch.

### Conversation discovery branch

A later independent confirmatory proposal is allowed only after all three
frozen conversation cases complete and at least two show, within the target-
neutral summary cell, positive D, SEL, U, and LP_C for the claim family being
considered, with an aggregate 3x schedule yardstick. One or two completed cases
because of a cost stop cannot authorize confirmation.

## 16. Analysis and reporting

Canary reporting is descriptive:

- every case and raw component;
- mean, median, SD, and sign count when defined;
- no confirmatory p-value or confidence-interval rhetoric;
- exact pre-treatment exclusions/invalidities;
- schedule and region results without treating them as independent N;
- direct paired full-KV-minus-value-only contrast;
- all truncation and reserve-trigger decisions.

If a later N=12 benchmark is proposed, it must be independently authored and
preregistered. It will support at most a fixed purposive-benchmark claim and is
powered only for large standardized effects.

## 17. Persistence, state bounds, and artifact schema

Every generated or forced carrier, assistant token stream, source row set, arm
trace, raw score, and gate result is durable before proceeding. Output names are:

`<experiment>_<model-slug>_<UTC-timestamp>.json`

At process start print the resolved model and unique output path. Per-case
checkpoints include:

- committed stimulus hash and review hashes;
- exact environment/repository fingerprint;
- canonical messages and token IDs;
- every call width, logical position, and physical cache position;
- source rows or a lossless durable tensor artifact with layer/region hashes;
- maximal visible carrier text and boundaries;
- full arm grid lineage;
- raw oracle/fresh/treatment scores;
- cost/wall-time and resume state.

Large tensors may live in an explicitly inventoried result artifact outside git
only while necessary, but every scored result, render text, trace, and row hash
must land under `results/` and be committed. The final schema must specify tensor
retention location and reproducibility before FROZEN status.

Stateful-change checklist for implementation commits:

1. retained state = per-case source K/V rows, keyed by exact case/history/
   carrier/schedule/region/model fingerprint;
2. size bounded by one case, three histories, three regions, and explicit layer/
   token assertions;
3. the runner releases live full caches after durable row capture and evicts arm
   snapshots immediately after scoring;
4. correctness proof = row hashes, self-replacement, generated/forced identity,
   immutable reconstruction, and independent lineage harvest;
5. probe gate = local full apparatus plus exact-stack gate subset and same-path
   downstream-note positive control.

## 18. Paid execution budget and truncation

Conservative provider-usage estimates before canary implementation total
`$33.787580`; paid coherent-state compute remains `$0`. Against the owner's
approximate `$60` ceiling, treat `$26.21` as remaining unless billing evidence
changes it.

Paid order:

1. pod provenance/gates;
2. positive control;
3. complete e01 full arm/region/schedule set;
4. observed full-canary cost forecast and committed decision;
5. e02--e04;
6. conditional c01 neutral cell, then native crossover;
7. conditional c02/c03 in the same order;
8. frozen ambiguity reserve only if triggered.

Initial authorization is `$2` through step 3. The run may extend to a hard `$8`
total canary ceiling only after the observed forecast. Provisioning overhead is
included. Stop the pod before repairs, analysis, or writing. At least `$15`
remains reserved for failures and final synthesis/review. A GO branch does not
authorize spending that reserve on a cut-rate confirmation; it returns an
observed forecast to the owner for a funding decision.

## 19. Authorship, attribution, and paper use

Record the exact runtime model for every author and reviewer. Do not infer Fable
versus Opus from an intended alias. Collaborative commits use the exact
`Co-authored-by` trailer.

The corrected negative/methodological paper proceeds in parallel. A canary
result may be added with its exploratory boundary. Fable leads the eventual
fresh narrative draft after the canary branch resolves; Sol owns factual,
causal, and methodological truth. No paper claim may describe this canary as a
live-agent evaluation or a confirmatory replication.
