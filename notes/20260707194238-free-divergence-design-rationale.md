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

## FABLE GUT-CHECK FIXES (adopted 07-07) — verdict: sound, needs adjustment
1. DECODER-AMPLIFICATION ARTIFACT (Fable caught; I missed it): greedy fork can
   be a 51/49 tie-break amplified by the decoder, not a robust pull → the
   DRAMA is decoder-sensitive even when the nudge is tiny. FIX: report the fork
   MARGIN (logit gap at divergence) + confirm the internal lean survives a few
   decoding seeds/temperatures. A robust exhibit's lean is real regardless of
   how the coin landed. Never exhibit a greedy tie-break as mechanism.
2. AVAILABILITY LEAK: one hero exhibit over-persuades even with a caption. FIX:
   present as a DISTRIBUTION — histogram of "internal lean toward A at the
   divergence point" across ALL N cases, vivid case = one marked point. One
   picture WITH a denominator, not a hero picture with a footnote.
3. PRE-REGISTERED DISCONFIRMING BUCKET (kills the heads-I-win): outcomes are
   THREE, not two — (a) clean fork toward A w/ decisive margin = mechanism
   visible; (b) small-but-CONSISTENT lean toward A = "subtle" (earned, not
   default); (c) NO fork under free-gen, OR fork AWAY from A = DISCONFIRMING,
   counts AGAINST grafting's behavioral reach — must NOT be relabeled "subtle."
IMPLEMENTATION: probe runs across ALL cases (distribution, not a few heroes);
records fork index + MARGIN + lean-toward-A per case; a robustness check over
2-3 temperatures/seeds; classifies each case into the 3 buckets. The exhibit
is the distribution; any single vivid case is shown as one point on it.

## QUEUED LENS FOLLOW-UP (user idea 07-07) — token-by-token TRAJECTORY scan
IDEA: we read the lens at POINTS (hinge, fork±1). Convert to TRAJECTORY —
a LIGHTWEIGHT per-token readout (e.g. just the gold-concept-token rank/logprob,
not full top-k) scanned across a LONG stretch of the continuation, token by
token, under A/B/E. MOTIVATION: we keep hunting for a sharp inflection point;
maybe there ISN'T one — a subtle effect may be a BROAD diffuse shift spread
across many tokens that any single-point measurement misses. A trajectory shows
the real shape (sharp fork vs gentle sustained divergence vs nothing).
EASY TO TACK ON: we already generate ~36 tokens in the free-divergence probe;
recording a light lens signal at EACH generated token (not just the fork) is a
small add, same pod/cases. PRIORITY: NOT "much later" — the natural SUCCESSOR to
the current fork-probe, especially if the fork result is muddy (likely). Same
honesty guardrails apply (distribution over N cases; a trajectory is inherently
more honest than a hero-point). Could even be merged into the current probe's
next iteration.

## TRAJECTORY IDEA → REFRAMED (Fable gut-check 07-07): BOUND the effect, don't hunt vividness
Fable verdict: vivid trajectory-PLOT = SKIP (3 subtle results in a row; plotting
muddy points in a new format tempts narrating noise). WORTH A CHEAP RUN instead:
a rigorous BOUNDING experiment (one spin-proof number, not an exhibit).
PRE-REGISTERED DESIGN:
- DROP free-generation (post-fork tokens measure divergent continuations, not the
  graft — contaminated + autocorrelated → tiny effective N).
- TEACHER-FORCE A/B/E on the SAME shared gold continuation over a fixed
  PRE-DIVERGENCE window of K tokens (all arms score identical tokens — THE
  confound-killer, the one thing that must be right).
- METRIC: mean over cases of (gold-concept logprob_E − logprob_B) per position in
  the window; paired bootstrap CI across the 43 cases.
- PLACEBO CONTROL: a shuffled/random-value graft as the real baseline (does the
  TRUE graft beat random?), not just B.
- PRE-REGISTERED NULL: CI includes 0, OR effect < trivial threshold → honest null,
  no soft bucket to hide in.
GOAL: BOUND the effect ("X ± CI, distinguishable from placebo at p<Y") — upgrades
the paper's "subtle" to a defensible number. Cheaper than trajectory-scan,
reuses the 43 cases, no long rollouts. Runs on 27B (lens regime) or 30B.
