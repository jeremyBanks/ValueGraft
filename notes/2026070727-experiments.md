# EXPERIMENTS.md — portfolio of ways to measure graft/compaction impact

_Standing menu of experiment VARIETY. Three independent axes — mix any task ×
any metric × any design. Rated: cost (pods $), effort (my time), promise.
Findings land in FINDINGS.md. Keep this fed with new ideas._

## AXIS A — TASK TYPES (what conversation gets compacted)

A1. Synthetic plant convs (HAVE) — early fact/sense/referent/stance, evicted,
probed. F1 came from here. A2. Story/narrative continuation (NEW, cheap) —
characters/facts set early; continue after compaction; score CONTRADICTION rate
vs planted facts. Natural-length, auto-scorable, realistic. HIGH PROMISE. A3.
Long-document QA (NEW) — real long doc (article/story), discuss it, evict early
portion, ask early-dependent question. Realistic, uses real text not our
authored convs → generalizability. A4. Cross-dependent reasoning chain (NEW) —
step 1's _result_ needed at step 5 (unlike our independent-exercise chains);
compaction between → does graft carry the intermediate? Directly tests
"workspace" carry. A5. Multi-session persona/memory (HAVE partial: LongMemEval)
— persona early, referenced late. A6. Persistent style/format instruction (NEW,
cheap) — formatting rule given early, must hold post-compaction across many
turns; auto-check compliance. A7. Real agent trajectories replay (HAVE: SWE-Gym
stage 2) — extend n; natural-length agent context. A8. Roleplay/dialogue
consistency (NEW) — persona traits set early; measure drift.

## AXIS B — MEASUREMENTS (how to score) — most are METRIC UPGRADES, work on ANY task

B1. Binary task success (HAVE) — blunt, ceiling-prone. Deprioritize as sole
metric. B2. Teacher-forced continuation logprob (HAVE, stage1/2) — sensitive,
our best mechanism metric. B3. Judged MEANING recovery (HAVE, F1) — the semantic
dissociation. Works. B4. **Distance-to-oracle / gap-closure** (NEW, cheap, HIGH)
— continuous: for each metric, what fraction of the full-context→compacted gap
does graft close? Unifies everything on one 0-100% scale. We have the TF
machinery. B5. **Distributional KL-to-oracle** (NEW, cheap, HIGH) —
KL(grafted-next-token || full-context-next-token) vs KL(compacted ||
full-context). How close does graft move the OUTPUT DISTRIBUTION to the oracle?
Far more sensitive than binary; per-token. B6. **Self-contradiction rate** (NEW,
cheap, auto) — does post-compaction output contradict pre-compaction statements?
Automatic, no judge. Pairs great with A2/A8. B7. Confidence CALIBRATION (NEW) —
does the model KNOW what it lost? Compare stated confidence vs correctness
across arms (the honesty/feeling-of-knowing axis, quantified). B8. Pairwise
preference (NEW) — judge prefers grafted vs compacted continuation, blind.
Cheap, intuitive, publishable figure. B9. **Interpretability readout (jlens)**
(HAVE subdir!) — read the residual stream: IS the evicted concept present
internally after grafting? Direct mechanistic evidence, not behavioral
inference. Validates F1's mechanism from the inside.

## AXIS C — DESIGNS/CONTRASTS

C1. K vs V independent grafting (planned, DECISIONS 06:55) — different profiles
expected. C2. Alpha dose-response (HAVE for TF) — extend to new metrics. C3.
Per-layer / per-head (HAVE). C4. Ablation: summary-tokens-only vs tail-only vs
both graft anchors. C5. Recency-graded: does graft help more for content evicted
longer ago? C6. Oracle-distribution matching: treat full-context as target,
measure每 arm's distance (ties to B4/B5).

## TOP PICKS TO RUN (cheap, pod-light, high-promise)

1. B4/B5 gap-closure + KL-to-oracle on EXISTING synthetic data — continuous
   sensitive metric, might reveal effect where binary was blind. Reuses TF
   machinery. START NOW.
2. A2 story-continuation + B6 contradiction-rate — new realistic task,
   auto-scored, natural-length eviction. Build + small run.
3. B9 jlens interpretability — read whether graft actually inserts the concept
   internally. Uses the existing subdir. Validates mechanism directly.
4. A4 cross-dependent chains — fixes the "chains too independent" flaw with
   genuine step-dependency.

## NEXT CONCRETE STEPS (07-07 11:10, ready to run)

- **B4 category gap-closure** (t1 pod, ~$3, HIGH): per probe, capture TF-logprob
  of gold continuation under A/B/E, compute (E−B)/(A−B), STRATIFY by
  sense/referent/stance. Independent judge-free corroboration of F1. Runner:
  adapt run_tune_hf score() to iterate probes×categories. Stored data has
  answers not logprobs → needs the run. READY.
- **A2 story-contradiction**: subagent building src/story_tasks.py now.
- **B9 jlens**: read jlens_boundary_probe/ + wire concept-presence readout
  (mechanistic F1 validation).
- **F1 hardening**: widen A-ceiling sample; strict-RECOVERED robustness cut; 4B
  category replication (cheap).
