# Disposition of Fable's regroup review: canary design closure

**Decision owner:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Advisory reviewer:** Claude Fable 5 (`claude-fable-5`)  
**Date:** 2026-07-11  
**Applies to:** `notes/2026071189-sol-ultra-regroup-decision.md` and the future
additive canary preregistration

Fable's complete review is
`notes/2026071190-fable-ultra-regroup-review.md`. I accept its central verdict
and every material correction below. This note closes the strategy-level design;
the exact stimulus texts, token geometry, runner schema, and numerical analysis
must still be frozen in a separate preregistration before a model forward.

## Accepted blockers

1. **Paid-stack gates are mandatory.** Local gates license local apparatus
   completeness only. On the paid pod, the exact 30B checkpoint/revision,
   tokenizer, dtype, resolved backend, and code inventory must be attested before
   inference. Generated/forced identity, self-replacement identity, row/region
   confinement for K/V/joint replacement, and engineered downstream sensitivity
   must rerun on that stack. The released downstream-note positive control must
   use the same gapped-destination and row-replacement path as the treatment and
   must pass before any summary semantic score is computed.
2. **Visible text is identical across region arms.** Every arm contains the same
   fixed target-neutral anchor text. Region arms vary only which persisted rows
   replace their fresh counterparts:
   - `R1`: summary content;
   - `R2`: content plus canonical assistant close/boundary;
   - `R3`: content plus boundary plus fixed anchor.
   Rows after the selected region are causally recomputed. `R2` is the sole
   stop/go region because current prior art most strongly predicts downstream
   aggregator/boundary carriage. `R1` and `R3` are descriptive localization
   conditions and cannot rescue a failed `R2` decision.

## Accepted sequencing and controls

- Run the exact-stack gates, then the apparatus positive control, then the
  engineered carrier stratum. The conversation stratum is conditional on an
  engineered full-KV channel.
- The engineered stratum contains exactly four paired cases: three short cases
  of roughly 1,000--2,000 production tokens and one mid-length case of roughly
  4,000--6,000 tokens. They must be authored in distinct structures by at least
  three independent model sessions spanning at least two model families, then
  mechanically and blindly reviewed before outcomes. They are fixed stimuli,
  not four random population draws.
- Conversation discovery uses a target-neutral common summary forced through
  both histories as its clean semantic cell. The native correct/wrong summary
  crossover is separately retained to ask whether naturally generated summary
  state behaves similarly; it does not remove the focal contradiction confound.
- Pre-treatment eligibility (full-history competence, fresh damage, leakage,
  and carrier NLL/support) is computed, committed, and independently harvested
  **before treatment contrasts are unblinded**. A single stimulus-redesign cycle
  may be triggered only from those pre-treatment measures.
- The role-native schedule claim is explicitly stack-relative: fixed authored
  assistant IDs are forced through this repository's q=1 incremental loop. It
  is not equated with a production server using continuous batching or another
  chunking policy. Canonical non-final Qwen rendering and exact close/open token
  order require a dedicated identity gate.
- The primary semantic contrast on each case is measured under role-native
  replay. The same case also runs turn-aligned P on the paid stack as a numerical
  yardstick. For the canary decision, the absolute primary semantic contrast
  must be at least three times the absolute N-versus-P change in that contrast;
  this is an exploratory robustness heuristic, not a universal precision law.
- The nonsemantic V perturbation is matched per layer and per selected row to
  the L2 norm of the correct-minus-counterfactual V delta, with its deterministic
  seed and construction frozen before outcomes.

## Closed decision branches

For primary region `R2`, define the full-KV and value-only history contrasts,
focal selectivity, correct-target movement, and fresh utility exactly in the
future preregistration.

1. **Clear engineered pass:** all gates pass; at least three of four engineered
   cases have the correct sign for history direction and focal selectivity; the
   aggregate correct-target movement is favorable; and the semantic contrast
   clears the 3x paid-stack schedule yardstick. Proceed to the frozen
   conversation discovery stratum.
2. **Clear engineered stop:** the positive control passes but the full-KV
   history-direction mean is non-positive or fewer than two of four cases have
   the correct directional/selective sign. Stop the conversation stratum and the
   summary-state program under the current budget. State the accepted residual
   risk: short/mid-length engineered nulls do not logically exclude a channel
   unique to longer organic histories.
3. **One ambiguity extension:** if the full-KV mean is positive and exactly two
   of four cases are directionally/selectively positive, or if three are
   positive but the schedule ratio is below 3x, author and run exactly two new
   engineered cases under the already frozen design. No other ambiguity
   extension is allowed. The extension may cost at most `$3` and cannot change
   arms, regions, targets, or thresholds.
4. **Full-KV pass, value-only stop:** proceed to conversation discovery for the
   narrower ordinary-model full-KV retention claim. Do not authorize a
   value-only confirmatory or live-agent treatment. A later independent
   full-KV-focused confirmation is allowed only if the conversation cases also
   show the clean history effect and utility.
5. **Value-only pass:** conversation discovery must independently clear
   history direction, focal selectivity, correct-target improvement, and fresh
   utility before any confirmatory value-only benchmark or live-agent feasibility
   pilot is authorized.
6. **R1/R3-only result:** report localization and consider a separately funded
   engineered-carrier study; do not change the `R2` canary verdict.

No rule above creates a paper-level positive claim. It only decides whether a
new independent confirmatory study has enough expected value to propose.

## Cost and truncation order

The completed Fable review raises the conservative provider-usage estimate from
`$31.176448` to `$33.787580`; the actual incremental cash charge remains
unverified. Against the owner's approximate `$60` ceiling, the conservative
remaining envelope is therefore about `$26.21`.

The original `$5` canary cap was optimistic. The future preregistration must use
this order and persist each prefix-complete unit:

1. pod provenance and technical gate subset;
2. same-path downstream-note positive control;
3. first complete engineered case with the **full** arm/region/schedule set;
4. observed full-canary cost forecast and decision record;
5. remaining three engineered cases;
6. conditional conversation case 1;
7. its neutral-summary cell, then native-summary crossover;
8. any additional conversation case;
9. the single prespecified ambiguity extension, if triggered.

The initial paid authorization is `$2` through step 3. After the observed
full-arm forecast, the total canary may extend to a hard `$8` ceiling, including
provisioning overhead. A stopped pod is required before any repair; no idle pod
may fund debugging or prose. At least `$15` remains reserved for failures and
final synthesis/review. A GO result will therefore almost certainly require an
owner funding decision before a credible confirmatory run; it does not authorize
a cut-rate confirmation from the reserve.

## Final strategy verdict

The pivot stands: preserve and pause v11, build the additive decision canary,
begin the corrected paper in parallel, and perform no live-agent treatment study
until a usable channel clears its internal causal controls. Fable's review
strengthened this plan by closing paid-stack, text-matching, sequencing,
ambiguity, and budget loopholes; it did not restore the twelve-case-first plan.
