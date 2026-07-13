# Fable review — local v3 eligibility after v2 development canary

**Date:** 2026-07-12
**Reviewer:** Claude Fable 5 through the Claude CLI.
**Boundary:** treatment-blind v2 Phase-A values only; no graft/treatment outcome
had been constructed or scored.

## Review supplied verbatim

Verdict, grounded in the frozen v2 design and the committed canary record (`results/coherent_state_local_n48_v2_phase_a_canary_20260713T005159Z`, which confirms your numbers: the only focal failure on both carriers is `margin_damage_at_least_5` at 0.578/0.850, with `Dplus` 13.83/13.60, all identities bit-exact, and nonfocal failing only the `begins_target` form while margins stayed positive).

**Defensible, not post-hoc rationalization — conditional on three things you've already half-committed to.** The distinction that saves this: you are not changing the estimand and you are not changing a rule to admit a candidate whose *outcome* you've seen. The frozen v2 primary was always `L_C(E_C) − L_C(B)`; no treatment outcome exists (`treatment_outcomes_computed: false`). The `Dmargin>=5` gate fails for a structural, mechanism-level reason visible in treatment-blind data alone: when both target and countertarget are floor-censored in B (−13.8 and −21.6 nat/token), margin damage is unidentifiable regardless of 13.8 nats of correct-target headroom. A gate that cannot in principle be satisfied by fixtures with tiny fresh alternatives is a broken instrument, and removing it does not bias the primary — `Dplus>=5` still guarantees the recovery headroom the estimand needs, and specificity is still tested where it belongs, at E_C vs E_W. Note also that both revisions (dropping Dmargin, relaxing begins-with to contains) operate in the *conservative* direction on the pipeline: their old failure mode was attrition of otherwise-valid candidates, not admission of favorable ones. That is what distinguishes this from the outcome-chasing pattern you're worried about.

**Yes, exclude the development candidate — this is mandatory, not optional.** Its Phase-A evidence informed the rule change, so it cannot be selected under rules it shaped, even though it would pass them and even though no treatment outcome was computed. The exclusion must be unconditional and stated in the v3 prereg as such (it is out even if it passes every v3 gate), with its committed record cited as the development canary.

**The target-containing nonfocal criterion is adequate in concept, under-specified as written.** "Decoded greedy contains exact target" invites the next dispute unless mechanically pinned down. The observed generations ("The shelf word … is Nimbus." / "Nimbus.") would pass any reasonable version, so specify it now, blind to future candidates.

Mandatory corrections — nothing else:

1. **Freeze v3 before any further forward pass.** Commit and push the v3 prereg (new design ID or explicit amendment) before Phase-A runs on any rank of any stratum, including rank 2 of threshold. Prospective labeling only counts if the commit precedes the data.
2. **Update the stratum termination rule explicitly.** Threshold now requires six eligible among ranks 2–10; the other seven strata keep ranks 1–10. State it; don't leave it implied.
3. **Pin the containment gate mechanically.** Match by exact token-ID subsequence of the declared answer-content IDs (or word-boundary string match after a stated normalization — case, whitespace, punctuation) within the first 16 greedy tokens; apply the identical rule to countertarget exclusion. As written, substring matching can false-positive on the target embedded in a longer word and the countertarget rule is asymmetric.
4. **Demote Dmargin, don't delete it.** Remove the eligibility threshold; keep persisting and reporting margin and margin damage as the mandatory companions v2 already lists. Deleting the measurement would look like burying it.
5. **Add one claim-boundary sentence to v3.** The canary shows fresh B can retain a large *positive* margin toward C while both alternatives are floored. So the headline recovery claim is about absolute correct-target logprob, not C-vs-W discrimination, and any fixture with positive fresh margin must be reported as such. Without this sentence the result invites over-reading, and that's a bigger post-hoc risk than the gate change itself.

Two things I checked and am deliberately *not* asking you to change, to honor "no apparatus wishlist": keeping begins-exact for the focal A_C/A_W gates is fine — its failure mode routes candidates to the next frozen rank (attrition, conservative), and the canary shows focal oracle greedy does begin with the bare target; and the 16-token greedy cap matches what the canary harness already used. If focal begins-exact starts failing verbosely on unseen ranks, that is a new fact for a new prospective amendment, not something to pre-fix now.

## Main-agent disposition

Accepted exactly. V2 remains unchanged and its threshold-rank-1 development
candidate is excluded unconditionally from v3. V3 changes only treatment-blind
eligibility mechanics, keeps Dmargin as a reported companion, and leaves the
primary estimand, fixed carrier conditions, arms, controls, bound, and frame
unchanged.
