# Independent audit of the canary preregistration requirements

**Author:** Independent Codex subagent  
**Runtime model:** The runtime model identifier was not exposed to Sol  
**Date:** 2026-07-11  
**Scope:** Read-only audit of the strategy and closure notes before the additive
canary preregistration was frozen. Sol retains final decision authority.  
**Execution status:** No model forward was run. This audit reports no semantic
outcome.

## Verdict

The canary strategy is defensible, but the strategy-level closure note is not by
itself sufficient to authorize a model forward. The following points must be
resolved literally and numerically in the additive preregistration.

## Blocking gaps

### 1. Freeze the literal stimuli, not just the design

The preregistration must bind:

- exactly four primary engineered cases;
- exactly two reserve extension cases, authored and reviewed before initial
  outcomes;
- exact correct/counterfactual histories, roles, probes, targets,
  unchanged-control probes, carrier text, and anchor text;
- tokenizer IDs, positions, per-message widths, changed-token allowlists, and
  SHA-256 hashes; and
- author/model/session provenance and blind-review records.

Authoring extension cases after seeing the first four is outcome-adaptive.
E05--E06 should exist in a sealed reserve manifest beforehand.

### 2. Define the engineered case structure

State whether each case has:

- exactly one changed focal plant plus one unchanged control fact; or
- two focal plants with two counterfactual histories.

The reviewed notes use both idioms. All contrasts and sign counts depend on this
choice.

### 3. Fix the conversation-stage N and selection rule

"One to three," "case 1," and "any additional case" are not a freeze. Specify:

- exact case IDs and count;
- which existing v11 draft, if any, is used;
- which cases are newly authored;
- whether a second case is triggered only by a cost forecast, never by the
  first case's outcome; and
- the exact conversation-stage GO/STOP criterion.

The current closure says conversation cases must show clean history effect and
utility but provides no count or threshold.

### 4. Specify the positive control completely

Freeze its literal history, carrier, targets, row region, state arms, and pass
threshold. "Downstream-note positive control passes" is undefined. It must use
the same gapped destination, replacement code, R2 rows, target scorer, and
harvester as treatment.

A defensible pass rule would require a prespecified positive history-direction
margin, favorable correct-target movement, and deterministic answer direction,
not merely any nonzero logit difference.

### 5. Choose one exact arm matrix

"At minimum," "prefer," and "optional" are unacceptable after the freeze. A
clean choice is:

- oracles: `A_correct`, `A_wrong`;
- the full 3x3 destination grid with key source and value source each in
  `{fresh, correct, wrong}`; and
- one frozen norm-matched V perturbation placebo.

If cost requires less, enumerate the reduced set explicitly. Define V-only as
fresh K plus source V, K-only as source K plus fresh V, and state exactly which
arms run under P and which regions.

### 6. Define the outcomes mathematically

For every target sequence, freeze teacher-forced mean token log probability.
Then define, for state family `X`:

- `M`: correct-target mean log probability minus counterfactual-target mean log
  probability;
- `D_i^X`: focal `M(correct-state) - M(wrong-state)`;
- `Q_i^X`: correct-target log-probability movement between those states;
- `N_i^X`: absolute movement on the unchanged-control margin;
- `SEL_i^X = D_i^X - N_i^X`;
- `U_i^X = M(correct-state) - M(fresh)`; and
- `Uplus_i^X`: correct-target log-probability versus fresh.

Using absolute non-focal disturbance prevents a favorable-sign disturbance
from artificially inflating selectivity.

### 7. Make the engineered decision tree exhaustive

The existing branches omit, among others:

- four of four directional cases but schedule ratio below 3;
- three of four directional cases but correct-target movement nonpositive;
- positive sign count but nonpositive aggregate mean;
- value-only strong while full-KV misses; and
- ambiguity after the two-case extension.

Recommended terminal logic:

- **PASS:** positive mean `D`, at least 3/4 cases with `D>0` and `SEL>0`,
  positive mean `Q`, and schedule criterion passed.
- **STOP:** mean `D<=0` or at most 1/4 selective-directional cases.
- **AMBIGUOUS:** every other valid outcome; run the sole pre-frozen two-case
  extension.
- **Post-extension PASS:** same criteria with at least 4/6
  selective-directional cases.
- **Post-extension STOP:** everything else.

Fresh utility should be an additional requirement for a mitigation branch, not
for the narrower existence-of-channel branch.

### 8. Freeze the schedule-ratio formula

"Three times schedule movement" currently lacks a metric, aggregation, and
state-family definition. A cancellation-resistant definition is:

```text
S_X = mean_i abs(D_i,N^X - D_i,P^X)
```

and the criterion:

```text
mean_i D_i,N^X > 0
and
mean_i D_i,N^X >= 3 * S_X
```

Apply this separately to full-KV and value-only if both can authorize work. P
should run on every engineered case for R2, using the identical carrier and
relevant arms. R1/R3 remain descriptive.

### 9. Define N and P literally

Freeze:

- exact canonical chat rendering;
- which structural tokens are prefilled as blocks;
- which assistant tokens are passed one at a time;
- whether close tokens are included in assistant q=1 forcing;
- exact summary/carrier and anchor call segmentation;
- logical positions and cache positions;
- EOS and stop handling; and
- exact P block boundaries.

Bind the stack to `Qwen/Qwen3-30B-A3B-Instruct-2507`, revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, bf16, eager attention, plus exact
Transformers/PyTorch/CUDA and repository commit.

### 10. Freeze region row intervals

Include the identical literal anchor in every arm. Define exact half-open
intervals for:

- R1: summary/carrier content;
- R2: R1 plus canonical assistant closing boundary; and
- R3: R2 plus anchor.

State that R2 is the sole decision region, anchor rows remain fresh under R1/R2,
and every row after the selected replacement interval is causally recomputed.

### 11. Implement a real pre-treatment release boundary

The intended sequence should be machine-enforced:

1. Generate summaries and compute only oracles, fresh baselines, competence,
   damage, leakage, and carrier NLL.
2. Persist and independently harvest these metrics.
3. Commit a hash-bound eligibility verdict.
4. Require that committed verdict as input to the treatment command.
5. Only then construct or score correct/wrong retained-state treatment arms.

A redesign may occur once, only before step 4. Treatment outputs should not
merely be hidden in a combined file; they should not yet exist.

### 12. Resolve eligibility thresholds

"Competent," "nontrivial damage," and "grossly off-support" need literal rules.
The reviewed notes rightly rejected the inherited uncalibrated 0.30-nat
exclusion, so either:

- freeze justified numeric cutoffs; or
- make NLL and headroom continuous diagnostics, invalidate only
  nonfinite/structurally impossible cases, and reserve utility claims for cases
  with positive observed `A-F` damage.

Full-history competence can be defined without an arbitrary magnitude:
correct-history and counterfactual-history oracles must each produce the
appropriate deterministic answer and a correctly signed target margin.

### 13. Correct the execution-order contradiction

The closure currently orders the ambiguity extension after conversation cases.
That is impossible because conversation execution is conditional on resolving
the engineered verdict.

Correct order:

1. pod gates;
2. positive control;
3. E01 full-arm forecast;
4. E02--E04;
5. engineered verdict;
6. E05--E06 only if ambiguous;
7. terminal engineered verdict; and
8. conditional conversation stage.

### 14. Clarify the money rules

State explicitly:

- the `$8` ceiling includes provisioning, failed starts, all four engineered
  cases, the possible two-case extension, and any conversation work;
- the `$3` extension limit is inside, not additional to, `$8`;
- no case begins unless its forecasted complete cost fits the remaining cap;
- incomplete cases are persisted but scientifically non-contributing;
- repairs occur only with the pod stopped; and
- the forecast formula, hourly price, safety buffer, and billing source are
  recorded.

### 15. Resolve the full-KV procedural gate explicitly

The closure makes conversation execution conditional on a full-KV channel.
Therefore a value-only-positive/full-KV-negative result currently stops, despite
value-only being the owner's original intervention and the blackboard
recommending that R not gate V.

Either preserve that conservative rule explicitly or permit an independently
passing V branch. Do not leave this to interpretation after results.

## Claim boundaries that should be explicit

The preregistration should state that:

- the canary is exploratory and permanently excluded from confirmation;
- its authored cases are fixed fixtures, not random population draws;
- imported assistant histories forced through q=1 are replay, not native
  live-agent trajectories;
- the semantic contrast is a controlled-mediator, fixed-visible-text effect,
  not the natural total effect of changing history;
- fresh comparisons are utility contrasts and include execution-policy
  differences;
- full-KV success does not validate value-only copying;
- R2 success is content-plus-boundary retention, not summary-content-only
  success;
- R3 success is an engineered-anchor result;
- the 3x schedule rule is an exploratory heuristic, not a universal numerical
  law;
- positive-control success validates the tested path but does not establish a
  natural semantic channel;
- failure is a bounded stopping result for this checkpoint, carrier, schedule,
  and budget, not evidence of universal absence; and
- no canary p-value, interval, or sign count supports a confirmatory efficacy
  claim or live-agent claim.

## Provenance note

Several reviewed notes retained stale internal filenames after archive
renumbering. The decision is now `2026071192`, the independent reviews are
`2026071190` and `2026071191`, and Fable's review is `2026071193`. The additive
preregistration should cite current paths and committed hashes.

This audit changed no experiment file or result and ran no model forward.
