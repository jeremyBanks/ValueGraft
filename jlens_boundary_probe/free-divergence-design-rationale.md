# Free-generation divergence probe — how it fits the whole experiment

*Design rationale + honesty guardrails, so this probe strengthens the paper
without pulling it off its honest footing. Written before running (07-07).*

## Where it sits in the overall arc
The experiment has three evidence layers on ONE claim (write-time value
grafting recovers compaction-lost semantic continuity):
1. BEHAVIORAL (the star): the sense/referent/stance dissociation + exhibits
   (Ruben, Lena Cho, countdown). Solid.
2. MECHANISTIC / lens (secondary): the weakest layer. The J-lens corroborated
   in AGGREGATE but produced NO vivid per-example figure — because we measured
   under TEACHER-FORCING, which pins the trajectory and suppresses the very
   divergence we wanted to see.
3. K/V exploration (Phase 2): keys don't help; value is the operative axis.

This probe targets layer 2 — the paper's weakest part — with a methodological
FIX, not a new claim: let grafted (E) and fresh-compacted (B) states FREELY
generate, find the FORK where a subtle internal difference blooms into a
behavioral divergence, and put the lens THERE.

## What it genuinely adds vs. what we already have
- We ALREADY have the behavioral fork (Ruben: E→right answer, B→wrong). So the
  fork itself is not new.
- The probe's ONLY new contribution is INTERPRETABILITY: locating the fork
  INTERNALLY — which layer/token the representations part ways, and whether E's
  readout at the fork leans toward the CORRECT concept (matching full-context A).
  That is "seeing the mechanism act," not a new effect.

## Honesty guardrails (non-negotiable — this is the seductive-exhibit risk)
1. ILLUSTRATIVE, NOT QUANTITATIVE. Free-gen divergence is a different, greedy-
   decoding-dependent measurement, NOT comparable to our gap-closure numbers.
   It is a qualitative exhibit, never an effect-size claim.
2. DOES NOT MAKE THE EFFECT BIGGER, only more VISIBLE. The aggregate effect is
   MODEST (+10-12pp). A vivid single-case fork must be contextualized as ONE
   case against that modest aggregate — never implied to be the typical magnitude.
3. THREE-WAY, not two-way: must show E forks toward A (correct), not merely
   "differs from B." Differing from B is trivial; moving toward the truth is the
   point.
4. NEGATIVE OUTCOME IS FINE AND LIKELY-ISH. If greedy B and E don't cleanly
   fork (the graft's nudge is sub-threshold for greedy decoding), or the lens at
   the fork is muddy, REPORT THAT honestly. Do NOT force a fork or cherry-pick.
   We already learned (the strong-example smoke) that chasing dramatic lens
   exhibits hits a wall — this is a NEW angle (fork location vs hinge readout),
   a legitimate second attempt, held to the same discipline.
5. CONSISTENCY: 27B (the lens model, matching the paper's lens-on-27B framing),
   reuse validated machinery, alpha_V=0.75 (the paper's dose).

## If it works / if it doesn't
- WORKS (clean internal fork toward the correct concept): upgrades the paper's
  §6 from "no vivid per-example figure" to "here is the internal fork" — a
  MEANINGFUL claim change that must go through the full honesty pipeline
  (Fable conceptual gut-check + critics + Fable readability). Contextualized as
  one illustrative case, not a magnitude revision.
- DOESN'T (subtle/muddy/no clean fork): the paper's current honest framing
  STANDS unchanged, and we note in FINDINGS that even free-generation didn't
  surface a vivid internal exhibit — which further supports "the effect is real
  but genuinely subtle," itself an honest strengthening.

## Scope
Phase-1-paper improvement (mechanistic illustration), SEPARATE from Phase 2
K/V. Runs sequentially on the one pod after the K/V per-layer probe, swapping
to 27B; pod terminates after validation. Cost-conscious, one pod at a time.
