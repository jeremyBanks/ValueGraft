# Powered v13 adversarial review and corrective disposition

**Date:** 2026-07-12  
**Status:** correction record after the first draft and Fable freeze review;
supersedes the statistical/release portions of the earlier design disposition.

## Why this correction exists

The first v13 draft used a sequential Welch--Satterthwaite t UCB at N=24/32/48.
Fable judged that core sound, while correctly blocking freeze on Phase-A cost,
probe placement, VP feasibility, and render/claim wording. A separate adversarial
Codex review and the literal simulation found a deeper error that Fable missed:
Welch--Satterthwaite is approximate under heteroscedastic normals, and the
draft's zero-observed-variance branch was catastrophically unsafe for a rare-
responder distribution the protocol itself claimed to stress-test.

Concrete counterexample: if raw recovery is 5 nats with probability 0.06 and
zero otherwise, the true mean is 0.30, yet all first 24 observations are zero
with probability `0.94^24 ≈ 22.7%`; the old rule returned a zero-width UCB and
would have stopped. The committed first-pass simulation independently observed
joint noncoverage of 19.4% for its rare-responder mixture and 10.3% for its
skewed mixture. Those are negative design results, not ignorable diagnostics.

The first-pass artifact remains at:
`results/coherent_state_powered_v13_validation/powered-v13-stat-simulation_local_20260712T161932Z.json`.

## Adversarial blockers disposition

1. **Invalid 95% claim — accepted.** The primary is now a joint finite-sample
   two-part bound: Hoeffding UCBs on cell means clipped to `[-0.5,0.5]`, plus an
   alpha-allocated prevalence bound for any raw effect above 0.5. Raw-nat t and
   bootstrap intervals are mandatory but explicitly model-based.
2. **No power threshold — accepted.** N=48 is the sole analysis. Before release,
   it must show at least 80% joint resolution for independent worst-variance
   bounded endpoints with true clipped mean .05 and at least 95% for mean-zero,
   SD-.25 endpoints. The corrected 200k-trial simulation observed 86.8% and
   99.999%, respectively.
3. **Circular release — accepted.** Release is staged: static parent-rooted
   Phase-A authorization, then a parent-rooted treatment manifest whose launch
   receipt records the containing release commit. No file hashes its own commit.
4. **Undefined probes — accepted.** The full A_C/A_W/FF context, separate focal
   and nonfocal user probe forks, answer header, teacher-forced target span,
   greedy position, and 16-token cap are now literal.
5. **Placebo overclaim/underspecification — accepted.** VP has two named event
   classes, bounded partial-cycle candidates, exact ordering/numerics/zero rules,
   relaxed pre-outcome geometry thresholds, and only qualifies value-only. WW
   remains full-KV's matched semantic control; full-KV is never called
   nonsemantic-placebo-complete.
6. **Render variance confounded by origin — accepted.** Both primary carrier
   renders now originate independently under C. Their difference estimates
   same-origin stochastic render variability; exact IDs are still forced under
   C/W/F.
7. **Unbounded/ambiguous sample law — accepted.** Each stratum has a finite
   enumerated pool and one committed OS-random without-replacement permutation.
   At most ten candidates are screened; fewer than six eligible is terminal.
8. **Phase-A budget omitted — accepted.** Paid screening/pilots have a hard $12
   cap, an early balanced yield/cost projection, mandatory selected-row bundle
   reuse, a $30 core cap, $4.50 host cap, and $8 cleanup reserve.
9. **Secondary/surplus multiplicity — accepted.** The primary alpha is now
   exactly `.02+.02+.01=.05`. Behavioral output is a finite-panel count.
   Surplus remains unauthorized until a separate additive preregistration.
10. **Host/resume unspecified — accepted.** The host subset/fields/1e-5 score
    tolerance and label precedence are fixed. Partial cases quarantine and
    recompute; only terminal cases skip.

## Fable blockers disposition

- **B1 screening cost:** accepted and expanded as above.
- **B2 probe position:** accepted.
- **B3 VP feasibility:** accepted; a bounded state-only fallback family is now
  the primary VP algorithm, e01 availability gates scaled Phase A, and every
  selected render's availability is known before treatment.
- **B4 produces vs encodes:** accepted. The estimand says encodes, and both
  renders are now on-policy C-origin, which also removes the confound.

Fable's affirmation of the old t guarantee is rejected based on the explicit
counterexample and simulation. Advice is not authority.

## Corrected statistical result before any outcome

The corrected simulation artifact is:
`results/coherent_state_powered_v13_validation/powered-v13-bounded-stat-simulation_local_20260712T165226Z.json`.

Observed in 200,000 trials per scenario:

- maximum joint noncoverage across bounded uniform, skew-beta, t3, rare-
  responder, and contamination scenarios: `0.002495`;
- worst-variance mean-.05 two-cell joint resolution: `0.86834`;
- mean-zero SD-.25 joint resolution: `0.99999`;
- primary Hoeffding radius per clipped cell: `0.2018668859`;
- zero-responder 99% UCB at N=48: `0.0914824243`;
- all preregistered simulation gates: pass.

These simulations audit implementation and power; the mathematical coverage
comes from bounded concentration plus the union-bound alpha ledger, not from a
finite simulation being treated as proof.

## Still not authorized

The corrected document remains DRAFT. No paid work is authorized until an
independent reviewer approves the corrected literal formula/release/control
text, the fixture generator and complete content-review pool exist, local
analysis/release/harness gates pass, and a dedicated Stage-A authorization
commit is made. Primary treatment additionally requires the completed paid
treatment-blind Phase A and separate Stage-B release.
