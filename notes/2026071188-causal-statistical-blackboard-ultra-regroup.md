# Causal and statistical blackboard review of paired v11

**Author:** Independent Codex subagent review  
**Runtime model:** The runtime model identifier was not exposed to Sol  
**Date:** 2026-07-11  
**Status:** Independent read-only methodological regroup. No model outcome is
reported and this note is not a preregistration.

## Verdict

The v11 direction is salvageable, but its cleanest evidence is not
`correct - fresh`. It is the same-schedule, same-shape comparison between
correct and minimally counterfactual histories, with focal selectivity.

## Severity-ranked findings

### 1. Critical — `R_correct - F` and `V_correct - F` are utility contrasts, not clean semantic causal contrasts

`F` is produced through the gapped destination path, while correct-source rows
come from a long-history source execution. The c10 diagnostic established that
query shape alone changes deterministic bf16 state. Therefore:

- `R_correct - F` asks whether full correct-history summary K/V improves
  deployment outcome over fresh same-text encoding.
- `V_correct - F` asks the corresponding practical value-only question.
- Neither alone identifies semantic information; each includes
  source-versus-destination execution-trajectory differences.

The cleanest semantic controls are:

- same-schedule `R_correct,s - R_wrong-p,s`;
- same-schedule `V_correct,s - V_wrong-p,s`.

Correct and wrong histories have identical message widths, positions, schedule
calls, summary IDs, and destination construction. Their token-content difference
is the intended treatment.

### 2. Critical — focal selectivity is required

A counterfactual history can generically disturb forced-summary state. Each
`wrong_referent` arm must be scored on both referent and sense probes, and
likewise for `wrong_sense`.

For state family `X in {R,V}`, define:

```text
U_i^X = (1/2) sum_p [Y_ip(X_C) - Y_ip(F)]

D_i^X = (1/2) sum_p [Y_ip(X_C) - Y_ip(X_Wp)]

N_i^X = (1/2) * {
  [Y_i,sense(X_C) - Y_i,sense(X_Wref)]
  + [Y_i,ref(X_C) - Y_i,ref(X_Wsense)]
}

SEL_i^X = D_i^X - N_i^X
```

Here `U` is utility, `D` is focal historical direction, and `SEL`
distinguishes focal semantic movement from generic damage. A target-specific
claim should require all three to be positive.

### 3. Critical — forcing the P-generated summary through W is valid only for a narrow controlled-mediator estimand

Let `S* = g(C,P)`. Forcing `S*` after W identifies:

> The effect of changing source history on write-time state while holding the
> realized visible summary tokens fixed.

It does not identify the natural effect of a counterfactual conversation,
because W might naturally generate different tokens.

If `S*` states or clearly implies the correct focal fact, W creates a
history-summary contradiction. Forced-summary NLL measures that surprisal but
does not remove it as a causal explanation.

Before target scoring, summaries should receive a blind per-plant label:
`target-neutral`, `partial`, or `explicit`. Clean semantic-specificity inference
should be limited to prospectively defined target-neutral summaries or else
explicitly described as including congruence/contradiction effects. No
outcome-based replacement is permissible.

Forcing the P summary through correct-history O is less problematic: it is a
well-defined same-text schedule sensitivity condition. If an O-native greedy
summary differs, however, the result is not the total effect of an O-native
workflow.

### 4. High — P should be primary; O need not be co-primary

P and O are different deterministic protocols, not noisy replicates. Neither is
literally a live agent schedule: imported assistant messages were not generated
q=1 by the subject.

Recommendation:

- Freeze P as the protocol-primary schedule.
- Make every claim explicitly P-specific.
- Treat O as a prespecified sensitivity/replication condition.
- O may be deferred if it materially increases cost or implementation risk.
- An O result must never rescue a failed P claim.
- Requiring positive lower bounds under both P and O needlessly changes the
  hypothesis to schedule invariance and sharply reduces N=12 power.

### 5. High — full-KV must not procedurally gate value-only

Run and score both from the beginning.

- Full-KV `R` asks whether coherent summary state carries a useful selective
  channel.
- Value-only `V` directly tests the owner's original mitigation idea.
- A practical V claim can stand if V clears its own `U`, `D`, and `SEL`
  criteria, even if R is inconclusive.
- R is required only for the mechanistic phrase “V captures part of an
  established coherent full-KV channel.”

Conversely, “R significant, V nonsignificant” does not prove value-only loses
the channel. That requires the direct paired `R-V` contrast to clear zero.

### 6. High — N=12 supports only a fixed-benchmark claim

The conversations are purposively authored, not random draws. Four authoring
passes improve diversity but do not create twelve independent draws from real
coding-agent traffic. Conversation t intervals have a working exchangeability
interpretation, not literal broad-population coverage.

At N=12, a directional t test has useful power only for roughly large
standardized effects, around 0.8 SD. A null remains inconclusive. Report:

- all twelve case values;
- mean, SD, median, and sign count;
- two-sided 95% t intervals;
- bootstrap only as sensitivity;
- leave-one-case-out and leave-one-author-pass-out results.

A sign-flip test is not design-exact because correct/counterfactual labels were
not randomized; label it a symmetry-based sensitivity analysis, not a
randomization test.

### 7. High — inherited competence/headroom gates are too arbitrary

The `0.30`-nat threshold has no external calibration. Do not exclude individual
cases based on headroom. Retain all prospectively eligible cases and report:

- `Y_A`;
- `A-F`;
- correct and counterfactual log-probability components;
- continuous headroom;
- summary leakage.

If aggregate `mean(Y_A) <= 0`, the model cannot answer the benchmark and
substantive inference should be withheld. `A-F` is informative but not a formal
upper bound.

### 8. Moderate — margin improvement is not automatically “less loss”

The correct-minus-counterfactual margin can rise solely because the
counterfactual target becomes less likely. Report both components. A claim of
reduced loss should require favorable correct-target log probability, at least
as a prespecified supporting outcome.

## Recommended claim structure

For each state family `X in {R,V}`, define a conjunctive claim requiring:

- `mean(U^X) > 0`: useful versus fresh;
- `mean(D^X) > 0`: moves in the correct focal direction versus matched
  counterfactual history;
- `mean(SEL^X) > 0`: focal movement exceeds non-focal disturbance.

Each is an intersection-union claim: all three components must clear. No
correction is needed among its components because failure of any component
defeats that claim.

There are nevertheless two claim families, R and V. Either:

- designate one as the sole confirmatory claim; or
- compute one global p-value per family as the maximum component p-value and
  apply Holm across the R and V families.

The second is preferred: it preserves the independent scientific value of both
without making one gate the other.

`R-V`, K-only, O-schedule results, leakage strata, category splits, NLL
correlations, and omission controls should remain secondary unless separately
allocated alpha.

## Minimum useful canary

The smallest decision-useful canary is six conversations spanning the authoring
passes. Two cases are only an apparatus test.

Use P only and score these eight arms per case:

- `A_full`;
- `F`;
- `R_correct`;
- `R_wrong_referent`;
- `R_wrong_sense`;
- `V_correct`;
- `V_wrong_referent`;
- `V_wrong_sense`.

Score every arm on both probes. Preserve raw correct/counterfactual token log
probabilities.

Stopping rule:

1. Any identity, lineage, replay, position, persistence, EOS, or exact-width
   failure yields `INVALID_TECHNICAL`.
2. Fewer than four of six summaries with both focal facts target-neutral, or
   aggregate `mean(Y_A) <= 0`, yields `ESTIMAND_INADEQUATE`; do not interpret
   efficacy.
3. Continue to the frozen N=12 only if at least one of R or V has positive
   sample means for `U`, `D`, and `SEL`, and at least four of six case values
   positive for each.
4. Otherwise stop as `NO_PROMISING_SIGNAL`, explicitly a bounded decision
   result, not evidence of absence.
5. N=6 can never support the paper's efficacy claim.

If the projected complete N=12 run remains around the earlier $8–$20 estimate,
running all twelve directly is preferable to an efficacy canary. The canary is
useful mainly if infrastructure or the true forecast remains uncertain.

## Recommended preregistration skeleton

1. Exact checkpoint, revision, dtype, backend, tokenizer, P protocol, and D
   construction.
2. Frozen authored corpus and author-pass provenance.
3. One P-generated summary `S*` per case; exact replay identity.
4. Pre-outcome summary leakage adjudication and eligibility rule.
5. Arms listed above; O and K-only status explicitly secondary or deferred.
6. Raw target log probabilities and margin definition.
7. Conversation-level `U`, `D`, `N`, and `SEL` for both R and V.
8. Primary claim family and multiplicity procedure.
9. No case exclusion, threshold change, or target replacement after scoring
   begins.
10. Technical invalidity, competence, canary, and terminal-N rules.
11. T interval plus case table, sign count, bootstrap, and leave-one-case/author
    sensitivities.
12. Explicit claim boundary: fixed authored benchmark under P, not live-agent
    traffic or universal serving schedules.

## What a live-agent pilot would add

The authored assay gives internal causal control. It does not test:

- subject-native assistant-token generation schedules;
- organic tool traces and unplanned facts;
- real compaction timing;
- downstream task completion;
- divergence of actions and environments after compaction.

A practical follow-up requires an open-weight agent whose KV cache is
accessible. Generate one real coding trajectory to a frozen compaction point,
snapshot the repository/environment, generate the summary natively, then branch
from the identical state into fresh, full-KV, and value-only conditions.
Continue each in isolated identical sandboxes with all renders and tool traffic
saved. Six tasks would be a feasibility pilot; approximately twenty or more
varied tasks would be needed before making a practical performance claim.

The decisive recommendation is: **center v11 on matched-history and
focal-selectivity contrasts; treat fresh as the utility baseline, not the sole
semantic control.**
