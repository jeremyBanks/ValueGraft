# Value-steering configuration: design decisions and rationale

*Product of a design discussion (user + Claude), 2026-07-05 night. This
document is the reference for how graft/steering configurations are chosen,
reported, and explored going forward.*

## Decisions

1. **Primary approach stays simple: one global α per model** (currently
   mid-band α=0.25 at 4B; global α=0.75 at 30B, both holdout-validated).
   The headline claims ride on this and only this. Rationale: the core
   effect must be demonstrable with an intervention that fits in one
   sentence; anything cleverer invites "how was that chosen?" and carries
   overfitting risk that the primary result does not need.

2. **The 57-slot mask (posslots) is demoted to an exploration section** —
   despite beating the incumbent on holdout (+0.038 vs +0.024, 10/10 wins).
   It is 192 validation-derived binary choices fished from a profile matrix
   that is ~70% unstructured (and partly noise at n=10); it has not yet
   passed its wrong-conversation contamination guard; and it complicates
   the story disproportionately to its contribution. It gets reported
   honestly (including the guard outcome) as "per-slot calibration looks
   promising," not adopted as the method.

3. **The calibration interface for new models is the factored form:**
   α(layer, head) = a_layer × m_head, head factors normalized to mean 1.0.
   - Costs nothing beyond the profile pass we already run (rank-1 / additive
     fit of the measured 48×n_kv effect matrix).
   - On head-homogeneous models it self-collapses to per-layer tuning
     (empirically true for every Qwen tested: head means explain 1.4% of
     slot variance at 30B).
   - Its expressiveness limit (no layer×head interactions) is a deliberate
     inductive bias: interactions are precisely where our overfitting
     occurred (4B "head 2" story died on holdout).
   - It travels to head-heterogeneous architectures (Gemma-class) where the
     head factors get their first real content; the sliding-window layer
     problem there is separate (cache semantics, not head structure).

4. **Negative coefficients: bounded exploration, not primary.** The
   experiment ladder for the exploration section, all evaluated once on
   holdout: (a) marginal per-slot profile [done]; (b) clamped joint fit
   (α ≥ 0, trust-region bounded); (c) signed joint fit, ridge-regularized
   and bounded (e.g. α ∈ [−0.25, 1.25]). Comparing (b) vs (c) measures
   whether negative degrees of freedom carry real value at our sample
   sizes. Cheap enough to run locally at 4B.

5. **Marginal ≠ joint caveat** goes in the write-up: per-slot profiles
   measure each slot grafted alone; slots write to overlapping residual
   subspaces, so joint behavior can differ. The mask's holdout success
   suggests weak interference at this dose, but joint fits are the honest
   instrument for any serious per-slot claim.

## Context: the discussion behind decisions 4-5

**The user's argument (paraphrased, not verbatim).** The per-slot blend
coefficients don't act independently — they scale directions in a shared
system, and those directions may be correlated rather than orthogonal. In a
correlated linear system, reaching the right point can *require* some
coefficients to be strongly negative (the classic case: two nearly-parallel
directions whose difference is what's needed). A coefficient that looks
nonsensical in isolation ("why would you ever blend negatively?") can be
exactly what the joint solution needs. Therefore clamping coefficients at
zero — however sensible it looks slot-by-slot — restricts the reachable
space and could distort the overall solution more than the seemingly
nonsensical negative values would.

**Assessment: geometrically correct.** Linearizing the intervention, each
slot's α scales a direction (v_old − v_fresh) whose downstream effect passes
through the output projection into the shared residual stream; heads
demonstrably write into overlapping subspaces. Optimal coefficients over
non-orthogonal directions routinely include negative entries; clamping
restricts solutions to the directions' convex cone rather than their span,
and if the ideal correction lies outside that cone, clamping strictly
worsens the achievable fit. Supporting evidence that the negative half-space
is *meaningful* (not just noise): inverted steering (α=−1) produces
systematic away-from-context interpretations (the Pokémon demo), i.e.
coherent motion along these directions in both signs.

**Why we nonetheless don't adopt unconstrained negatives (the
counter-argument that carried the decision):**
- *Nonlinearity bounds everything.* The linear picture holds only near the
  operating point; every measured α-curve degrades beyond moderate |α|
  (4B collapses past ~0.35; 30B declines past ~1.0). Downstream layernorms
  and attention softmaxes do not tolerate values far off the training
  manifold. Whatever the unconstrained geometry requests, admissible
  solutions live in a small trust region.
- *Correlated directions are exactly where estimation variance explodes.*
  Collinearity produces huge canceling coefficient pairs that fit the
  profiling sample and generalize terribly — and we fit from ~10
  conversations. Clamping (NNLS-style) is a standard, effective variance
  control precisely in this regime. The statistical argument overrides the
  geometric one at our sample sizes.
- The compromise adopted: signed fits are explored, but *bounded and
  regularized*, and only within the exploration section (decision 4). If
  (c) beats (b) on holdout, the geometry argument wins in practice and the
  write-up says so.
