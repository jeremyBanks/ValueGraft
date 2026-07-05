# Phase 2 results — mitigation experiments (both scales)

Follows phase2-design.md. All machinery identity-validated (LH ladder, HP-0 at
both scales). Judged by Sonnet (1,152 Phase-2 verdicts + sweeps).

## 1. H-pack vs B-min-pack (fabrication; primary contrast)

Fabricated : admitted on unknowable items (evicted facts / decoy probes):

| arm                       | 4B evicted | 4B decoy | 30B evicted | 30B decoy |
| ------------------------- | ---------- | -------- | ----------- | --------- |
| A (full context)          | 2:0 (C=22) | 15:9     | 0:0 (C=24)  | 16:8      |
| B (production compaction) | 15:9       | 18:6     | 16:8        | 19:5      |
| B-min-pack (fresh encode) | 6:18       | 4:20     | 5:19        | 10:14     |
| **H-pack (write-time)**   | 2:22       | 3:21     | **1:23**    | **3:21**  |
| H-pack-wrongS             | 4:20       | 0:24     | 2:22        | 0:24      |
| H-gap                     | 1:23       | 2:22     | 4:20        | 2:22      |

Findings:

- **vs production compaction the effect is dramatic at both scales** (30B: 19:5
  → 3:21 decoys; 16:8 → 1:23 evicted).
- **Decomposition:** most of it is the packed _layout_ (bare sinks+summary
  context makes the model cautious — B-min-pack is already far more honest than
  templated B). The **write-time encoding adds a component that grows with
  scale**: matched-pair decoy fabrication 4→3 at 4B (nil) but **10→3 at 30B**.
  wrongS is also honest (0:24), so state-oddness caution remains a partial
  competing mechanism; the matched pair is the honest estimate of the
  encoding-specific part.
- Packed arms do not improve recall (referent/sense floored — they have no tail
  by design). Honesty, not memory, is what this intervention buys.
- Full context does not protect against inventing never-discussed specifics (A
  fabricates decoys 15:9 / 16:8) — confabulation under confident context is the
  default behavior these packed contexts suppress.

## 2. Tuned ValueGraft (alpha sweep, validation→holdout)

Protocol: tune on c01–c06+n01–n04, report once on c07–c12+n05–n08.

| scale | winner (validation) | holdout Δ vs B | wins  | 95% CI         | gap closure |
| ----- | ------------------- | -------------- | ----- | -------------- | ----------- |
| 4B    | mid-band α=0.25     | +0.017         | 10/10 | [0.012, 0.024] | ~10%        |
| 30B   | global α=0.75       | +0.033         | 9/10  | [0.014, 0.057] | **~24%**    |

Dose-response by scale: 4B peaks low (fine grid rises to ~0.35; α≥0.75 harms;
mid-layer gating helps). 30B peaks at α≈0.75–1.0, tolerates α=1.25 (+0.026
holdout — extrapolating past the old values stays positive), and needs no
gating. Certified by negative controls at both scales (shuffled /
wrong-conversation grafts collapse: 30B −1.91/−1.99 vs B −1.48).

## 3. Deployment summary

- **SelfGist/H-pack**: retain ~4+|S| cache entries (≈10 MB at 30B with a terse
  summary), keys re-rotated to packed positions (exact, validated). Cost: one
  summary generation at compaction time. Buys: large reduction in
  post-compaction confabulation. Requires the cache to still exist at compaction
  time (a choice, not a recovery — fits long-running local agents/single-host
  serving).
- **ValueGraft α***: one extra prefill + a vectorized V-blend at compaction.
  Buys: ~10–24% of the continuation-quality gap on compaction-dependent content
  (scale-dependent dose; tune per model).
- The two compose in principle (graft the tail, pack the summary) — untested.

## Open items

30B Phase-2 judging artifacts in results/judge_batches_p2_30b/; sweeps in
results/alpha_sweep_{4b,30b}/. Not yet done: H-pack+tail hybrid, G/SoftGraft,
coding-trace and benchmark scouting (tasks #12/#13).
