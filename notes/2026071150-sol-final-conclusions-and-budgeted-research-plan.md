# Final integrated conclusions and budgeted research plan

**Author:** Sol — OpenAI Codex, GPT-5 family, extra-high reasoning  
**Attribution note:** The owner referred to this session as “GPT-5.6 Sol”; the runtime exposed GPT-5-family identity but not a verifiable minor version.  
**Date:** 2026-07-10/11 (America/Toronto)  
**Status:** Final and most complete note from Sol's independent review. It supersedes the topline interpretations in the three preceding Sol notes where they differ, while those notes preserve the detailed evidence.

Related records:

- `notes/2026071003-sol-empirical-statistical-audit.md`
- `notes/2026071004-sol-methodology-implementation-audit.md`
- `notes/2026071005-sol-independent-scientific-consultation.md`

## Executive decision

The original intuition was substantially right, but the repository has not shown that its particular intervention reliably exploits it.

The most defensible answer is:

> Generation-time KV state can preserve task-relevant information that is absent from, or is not recoverable by re-encoding, the same compacted text. Recent literature demonstrates this directly. This project has not yet shown that training-free post-prefill replacement of fresh values with aligned historical values reliably converts that information into improved semantic or agent performance. Its principal synthetic null is source-heterogeneous, three compression conditions are invalid, and its coding evidence is an offline teacher-imitation signal rather than an executed task result.

This is not a recommendation to abandon the question. It is a recommendation to stop spending on broad value-graft searches and instead perform a small number of causally decisive experiments in strict sequence.

The highest-value next experiment is not another alpha, head, layer, or architecture sweep. It is an exact same-text, same-position, same-kernel comparison between:

1. summary tokens conditioned on the original history, retaining coherent original K+V;
2. the identical summary tokens teacher-forced stepwise after a fresh restart;
3. the identical tokens conditioned on a matched wrong history;
4. the old-V/fresh-K intervention;
5. a treatment-delta-matched placebo.

That design separates four questions the current project repeatedly conflated:

- Is useful history-dependent information present in the state?
- Is it specific to the correct hidden history?
- Does coherent full-state retention expose it?
- Does the old-V/fresh-K transplant preserve or destroy it?

## Integrity corrections that must govern future work

These are blocking, not stylistic caveats.

1. **Synthetic body provenance.** The c07-c24 held-out body is not Qwen3-30B-native. c07-c12 were rendered by Qwen3-4B 4-bit; c13-c24 were externally authored. Qwen3-30B generated summaries but did not generate those conversation bodies.
2. **Source reversal.** The held-out per-layer effect is about +0.120 on c07-c12 and -0.050 on c13-c24, while the pooled value is about -0.003. The pooled null is an arithmetic mixture, not a homogeneous universal null.
3. **Compression reconstruction defect.** Ultra, brief, and medium summaries were generated under level-specific requests but had their supposed write-time states reconstructed under the realistic request. Only realistic is valid as the intended intervention.
4. **Region ambiguity.** Current synthetic and SWE grafts affect aligned retained-tail and summary tokens. They do not isolate a summary-state mechanism.
5. **Placebo mismatch.** Gaussian and shuffled source-state controls are not matched to the norm/covariance of the actual treatment delta. They do not establish hidden-history semantic specificity.
6. **SWE outcome scope.** SWE-Gym results are next-action teacher likelihood and coarse demonstration-action match. No patch was applied and no test was run.
7. **SWE data provenance.** The historical actions came from successful GPT-4o/Claude trajectories, not the Qwen subject. Generator identity is not available per row.
8. **Missing SWE renders.** Exact generated summaries, token IDs, generation prefixes, contexts, alignments, and complete fingerprints were not saved. Old SWE results cannot be reused as exact cache-state replays.
9. **Metric wording.** Raw E-B is a log-probability difference in nats/token and is not mathematically bounded.
10. **Build gates.** Existing identity work is a real strength, but average-logprob smoke checks do not prove tokenwise production equivalence for every model, dtype, backend, and batching regime.

The current README should not be promoted externally as the final paper until these points are reflected accurately.

## What the existing record is still useful for

The prior work is not discarded.

- The 173 unique SWE trajectories estimate effect heterogeneity and variance and can support strict rescoring of already saved generations.
- The frozen SWE layer champion can be carried forward as one prespecified candidate. It should not seed another tuning search.
- c01-c24 are useful regression fixtures and apparatus-debugging cases. They are not a clean native confirmatory population.
- Saved synthetic render text avoids expensive regeneration for corrected local replays, although it does not preserve the old cache tensors.
- The source reversal identifies a concrete factor to test with paired renders.
- The incident archive, build ladder, `kvlib`, atomic checkpointing, unique-output convention, pod launcher, and monitor gates materially reduce future risk.
- The existing OpenHands shim and local task adapters provide most of an agentic harness, but need a capability-matched subject model, common-prefix forking, and stricter environment/test handling.

## Budget assumptions

Treat every budget below as a **maximum**, not a target to exhaust. Stop when a prerequisite fails.

At review time, [RunPod's official pricing page](https://www.runpod.io/pricing) listed community A100 80 GB pods around $1.39-$1.49/hour and H100 80 GB pods around $2.89-$2.99/hour. Availability, storage, tax, failed starts, and degraded pods create real overhead. The plans below use an effective planning rate of about **$1.80 per productive A100 hour**, leaving roughly 15-20% operational reserve:

| Total budget | Approximate productive A100-hours after reserve |
|---:|---:|
| $50 | 27 hours |
| $100 | 55 hours |
| $200 | 110 hours |

An H100 should be used only if a three-item calibration shows more than roughly 2.1x throughput for the actual harness. Otherwise its price premium buys less science than more A100 time.

For continuity with the existing mechanistic record, use the exact bf16 `Qwen/Qwen3-30B-A3B-Instruct-2507` checkpoint. For real coding agents, the best first candidate is [`Qwen/Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct): its official card identifies a 30.5B-total/3.3B-active, 48-layer model with four KV heads, long native context, and agentic tool-calling support. Its geometry is close enough to the existing Qwen harness to make porting plausible, but **no result counts until the exact production build ladder passes**.

Cost estimates remain uncertain by at least 35-50% until three real tasks measure dollars per common prefix and per continuation. Every stage therefore has a dollar cap and a stop rule.

## The common scientific ladder

Every nonzero budget includes this ladder; higher budgets include the lower-budget logic rather than replacing it.

### Gate 0 — Read, freeze, and repair

1. Read the pending frozen 75-trajectory SWE champion confirmation when it lands. Do not retune on it.
2. Fix the wrong-request compression reconstruction.
3. Save every request, visible token ID, absolute position, summary token, alignment, render, cache description, model/tokenizer revision, runtime, code commit, and environment fingerprint.
4. Assert exact tokenwise reconstruction and maximum-logit differences, not only average likelihood.
5. Predeclare one primary estimand, task inclusion rule, smallest effect size of interest, and stopping rule.

No scaled run starts until this gate passes on the exact production checkpoint.

### Gate 1 — Exact state-channel experiment

For each model-native context, create identical visible tokens, absolute positions, suffix, and stepwise decomposition across:

- **F — fresh restart:** identical summary tokens teacher-forced stepwise after the compacted prefix.
- **R — retained coherent state:** the same tokens conditioned on the full original history, retaining original K+V.
- **W — wrong history:** identical summary tokens conditioned on a length/content-matched counterfactual history.
- **V — current ValueGraft:** original V paired with fresh K.
- **K — key-only diagnostic:** rerotated original K paired with fresh V.
- **D — matched placebo:** a true derangement of observed treatment deltas, matched within layer/head for row norm and empirical covariance.

Primary contrasts:

- state-channel effect: `R - F`;
- hidden-history specificity: `R - W`;
- naive ValueGraft effect: `V - F`;
- coherence benefit: `R - V`;
- perturbation-matched specificity: `V - D`.

Save perturbation norms, cosine similarity, output KL, next-token and next-3-to-5-action likelihood, and generated behavior.

If `R-F` is null in a capable/damaged regime, there is no reason to optimize a transplant there. If `R-F` is positive but `V-F` is null or negative, the project has a strong and useful answer: the channel exists, but incoherent V-only mixing fails to preserve it.

### Gate 2 — Region factorial

Test:

- no retention;
- summary-only;
- retained-tail-only;
- both;
- matched placebo for each region.

The interaction between summary and tail is part of the result. A tail-only benefit would change the story from latent summary semantics to retention of identifiers, paths, recent tool state, or procedural context.

### Gate 3 — Capability and compaction sensitivity

Before any live-agent treatment run, the chosen model/task stratum must show:

- full-context success or meaningful graded progress on roughly 30-70% of discovery tasks;
- a measurable loss under the frozen production-style compaction baseline;
- naturally long model-native prefixes that reach the trigger;
- deterministic and trustworthy tests.

If full context almost always fails, there is a capability floor. If compacted baseline almost always succeeds, there is a ceiling. Both make the mitigation question unidentifiable.

### Gate 4 — Matched causal treatment before live scale

Only a candidate that beats fresh restart and is separated from W/D placebos proceeds to live agent evaluation. This candidate may be coherent K+V retention or selective downstream recomputation; it does not have to be the original value-only graft.

## $0: the 32 GB Mac program

The local machine can support a rigorous mechanism-and-replay program, not a large confirmatory agent study.

### Z0 — Extract more from existing artifacts

Machine time: seconds to minutes after analysis code exists.

1. Re-score saved SWE generations at four strictness levels:
   - tool only;
   - tool + path + operation;
   - full normalized command;
   - for `str_replace`, normalized `old_str` and `new_str` exact/edit-similarity scoring.
2. Report exact discordant counts, paired exact tests, task-clustered intervals, and repo strata. Call the result demonstration-action match, never correctness.
3. Stratify synthetic E-B by summary coverage: fact explicitly present, paraphrastically recoverable, or absent. Cross this only with source block and primary category; do not fit a many-predictor model to 18 clusters.
4. Produce an artifact ledger: model/revision, precision, body author, summary author/request, grafted region, valid/invalid status, render availability, and inferential unit.

This resolves whether the coarse +7-point SWE action signal survives when edit content matters.

### Z1 — Build the causal apparatus on a tiny model

Use Qwen3-0.6B or another tiny compatible checkpoint for engineering only. Implement F/R/W/V/K/D and require:

- fresh-versus-fresh identity;
- alpha-zero identity;
- identical visible tokens and absolute positions;
- identical stepwise kernel/decomposition;
- true wrong-history and delta derangements;
- saved and numerically matched per-layer/head delta norms;
- a nonzero treatment that visibly changes logits.

Because this adds retained serving state, its implementation commit must answer the repository's five-part stateful-change checklist.

### Z2 — Local 30B 4-bit causal canary

Use only the already-cached `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`. The bf16 30B checkpoint is outside the Mac's practical memory envelope.

Start with six banked conversations balanced across the divergent source blocks, then expand to 12 only if the apparatus passes.

- Primary categories: referent and sense.
- Primary discovery contrasts: `R-F` and `R-W`.
- Secondary: `V-F`, `R-V`, `V-D`, output KL, and perturbation norms.
- Report unconditional results and headroom strata; never condition the headline on favorable probes.

Existing local records show roughly 220-475 seconds per conversation for earlier, simpler 30B 4-bit runs. The new multi-arm experiment may be slower. Benchmark one conversation before scheduling the rest; run one MLX process at a time.

Discovery interpretation:

- `R-F > 0`, `R-W > 0`, `V-F <= 0`: coherent state carries specific information, V-only mixing fails.
- `V-F > 0`, `V-D <= 0`: geometric/regularizing effect, not demonstrated hidden-history specificity.
- neither `R-F` nor `R-W` positive: stop local tuning and report no usable state channel in this quantized setup.

N=6 or 12 is not a significance study.

### Z3 — Free agentic proxy

Two useful zero-dollar steps are possible:

1. **Existing-generation rescoring** from Z0, which needs no inference.
2. **New multi-turn offline replay** on the local SWE-Gym parquet: choose a cut, generate and save a new local-model summary, and score the next 3-5 teacher actions while teacher-forcing intervening teacher actions/tool responses. This measures whether an effect persists beyond one action. It remains foreign-trajectory imitation, not executed agent success, and is a new 4-bit experiment rather than a reproduction of the bf16 runs.

A one- or two-task local OpenHands smoke can validate environment forking and logging, but should not be reported as performance evidence. The existing chain tasks have a compacted-baseline ceiling; old real SWE-bench attempts had a full-context capability floor.

### Z4 — Conditional local follow-ups

Only after Z2 is informative:

- run the summary/tail factorial with matched deltas;
- rerun ultra/brief/medium/realistic using the exact generation request and saved summary IDs;
- optionally render 3-6 identical scaffolds with local Qwen3-4B and Qwen3-30B to pilot a paired source effect.

The best $0 outcome is a clean decomposition, not another small positive.

## $50 maximum: mechanism plus agentic viability

This tier can answer whether the channel exists under exact controls and whether a practical coding-agent study is viable. It cannot support a task-performance claim.

| Stage | Cap |
|---|---:|
| Production ladder, persistence smoke, three-item runtime calibration | $5 |
| Exact F/R/W/V/K/D causal study on about 12 model-native contexts | $8 |
| Port Qwen3-Coder and pass its exact production ladder | $8 |
| Six-task full-versus-compacted capability/damage discovery | $10 |
| If all gates pass: 4-6 common-prefix live continuations; wrong-history control on two | $12 |
| Operational reserve | $7 |

If the state-channel or agent capability gate fails, do not spend the live-agent allocation. Redirect it to completing the matched-control null and corrected compression cells.

Interpret a 4-6-task agent run only as an engineering canary: can prefixes be banked, environments forked, tests trusted, and arms completed at predictable cost?

## $100 maximum: credible discovery and a small frozen agent test

| Stage | Cap |
|---|---:|
| Expanded exact causal factorial, roughly 20 contexts | $15 |
| Qwen3-Coder port plus model-native multi-turn replay screen, about 16 prefixes | $12 |
| Full/compacted discovery on 8-10 tasks; freeze task stratum, trigger, candidate, and metric | $18 |
| Held-out paired full/fresh/treatment continuations on approximately 18-25 tasks; wrong-history on six | $40 |
| Operational reserve | $15 |

This is an estimation study. A 10-point binary resolve-rate difference will not be settled at N around 20. Use normalized deterministic test score as primary and resolve rate as secondary.

Predeclare a smallest effect of interest such as +0.10 normalized test score. A nonbinding futility check may occur after 10 held-out tasks: stop only when the interval's upper end is already below the smallest meaningful effect. Do not declare success early.

If the agent capability/damage gate fails, redirect the held-out allocation to a paired source factorial: identical scenario skeletons rendered by Qwen3-30B, Qwen3-4B, and an external model, scored under the repaired causal apparatus. That directly resolves the largest ambiguity in the current synthetic result.

## $200 maximum: a moderately informative live-agent study

| Stage | Cap |
|---|---:|
| Production ladder and exact causal study, roughly 24-30 contexts | $18 |
| Qwen3-Coder port plus 20-prefix native replay screen | $17 |
| Full/compacted discovery and cost calibration on 10-12 tasks | $25 |
| Frozen held-out full/fresh/treatment study on approximately 40-50 paired tasks | $105 |
| Same-text wrong-history control on 10-12 held-out tasks | $10 |
| Operational reserve | $25 |

After 20 held-out tasks, permit only a nonbinding futility analysis. Final success requires the frozen treatment-minus-baseline interval to exclude zero at terminal N, no endpoint/configuration changes, and separation from matched controls.

Power remains limited:

- With paired binary discordance around 0.30, detecting a 20-point resolve-rate improvement at conventional 80% power takes roughly 59 tasks.
- Detecting 15 points takes roughly 105 tasks.
- With paired standard deviation near 0.25, 40-50 tasks can estimate a normalized test-score shift near 0.10 much more efficiently.

Thus $200 can detect a large binary effect and provide useful graded-outcome evidence. It cannot establish a modest resolve-rate improvement.

If coherent retention passes the causal gate but the original V-only graft does not, the live treatment should be coherent K+V retention or a small selective-recomputation repair. Scientific loyalty is to the question, not to the first intervention.

## A practical real-agent evaluation

The repository already has much of the plumbing:

- `src/serve_shim.py` exposes the subject model to an OpenAI-compatible agent client;
- `src/e1_agent.py` and `scripts/e1_driver.sh` run OpenHands episodes;
- `src/swebench_tasks.py` provisions and scores repository tasks;
- the pod launcher, monitoring, results sync, and incident gates are reusable.

The new experiment should change the unit from independently started episodes to a **banked common-prefix fork**.

### Subject and task source

Use Qwen3-Coder-30B-A3B-Instruct in bf16 on one A100 80 GB after full ladder validation. The old general Qwen checkpoint failed all tested real SWE-bench tasks even with full context, so it cannot identify a recovery effect.

Screen task sources in this order:

1. curated, containerized easy [SWE-bench Verified](https://www.swebench.com/SWE-bench/guides/datasets/) tasks, especially the human-rated under-15-minute stratum;
2. [SWE-smith](https://swesmith.com/) tasks on small, reliably installable repositories;
3. small real PR reproductions with deterministic tests;
4. purpose-built multi-file microrepositories only as a calibration tier.

SWE-bench Verified contains expert-screened solvable issues and difficulty metadata; SWE-smith supplies large numbers of automatically generated testable bugs. Neither guarantees fit for this particular model/harness, so A/B discovery remains mandatory.

Prefer official Docker environments or prebuilt images. If nested Docker is unavailable on the GPU pod, run environments on a separate x86 CPU host and keep only inference on the GPU pod. If using the existing pure-Python adapter, first repair and test its historical test-ID and environment shortcomings against known gold patches.

### Common-prefix procedure

For each task:

1. Start from the exact base commit and verified test environment.
2. Let the subject model investigate with normal tools until the first frozen compaction trigger, before the task is solved.
3. Save the complete model-native event stream, exact token IDs, incremental cache, generated summary, tool outputs, workspace commit/diff, running-process assumptions, wall time, and cost.
4. Clone the workspace/container snapshot and transcript into paired branches.
5. Continue:
   - **A:** full history with its incremental cache;
   - **B:** fresh production-style summary/tail restart;
   - **E:** the one frozen candidate that passed the causal gate;
   - **P:** same visible text with wrong-history/matched state, on a prespecified subset.
6. Give every branch the same tool permissions, time/token budget, decoding policy, and test access. Randomize branch execution order.
7. Continue until agent finish or a 20-30 minute branch cap, then run the exact deterministic evaluator.

Serving A incrementally matters. Re-prefilling its ever-growing full history on every turn would impose an artificial compute disadvantage and invalidate cost comparisons.

Restrict tasks to those without unrecorded persistent process state, or snapshot that state explicitly. A filesystem clone alone is insufficient if a database/server process matters.

### Inclusion and anti-selection rules

- Discovery tasks may be used to select a task stratum and one global compaction policy using only A and B.
- Held-out task inclusion may use task metadata, environment validity, and whether the frozen trigger was reached before completion.
- Never inspect E before deciding whether a held-out task counts.
- Do not select tasks because E helped them.
- Cluster all inference by underlying task. Multiple seeds, if used, are within-task repeats.
- Use temperature zero under the repository's evaluation policy; if sampling is later introduced, predeclare seeds and share them across arms.

### Outcomes

Primary:

- normalized deterministic test score, including fail-to-pass gains and pass-to-pass regressions.

Secondary:

- full resolve rate;
- number of regressions;
- progress toward the gold patch without requiring exact patch match;
- repeated failed actions/tool loops;
- wall time, tokens, tool calls, and GPU dollars;
- next 3-5 model-native action likelihoods at the fork;
- patch size and edit validity.

The continuous test score extracts more information than binary resolve rate at these budgets. Resolve remains the outcome that matters operationally, but should not be the only one.

### Cheap agentic ladder

1. **Offline native replay:** score 3-5 subsequent actions at banked forks. No execution claim.
2. **Two-task engineering smoke:** prove transcript/workspace/cache forking and scoring.
3. **Six-task A/B capability canary:** establish that the selected model can act and compaction hurts.
4. **Four-to-six-task four-arm canary:** estimate cost and surface state/environment bugs.
5. **Task-held-out terminal study:** 18-25 tasks at $100 or 40-50 at $200.

This is practical because the expensive early investigation occurs once per task. Every arm starts from the identical prefix and repository state.

## Candidate treatment priority

If the causal ladder permits them, test candidates in this order:

1. coherent original summary K+V retention;
2. coherent state plus a small downstream recomputation window;
3. summary-only versus tail-only coherent retention;
4. boundary/summary-end downstream-note token retention;
5. the frozen SWE layer champion;
6. the original scalar old-V/fresh-K graft.

The order reflects causal plausibility and current evidence, not implementation convenience.

## What not to spend money on

- another broad architecture sign map;
- more alpha/head/layer fishing on the mixed synthetic corpus;
- treating the invalid compression sweep as a starting point without fixing it;
- more imported single-next-action SWE-Gym scoring without execution or multi-turn replay;
- unmatched Gaussian/source-vector placebos;
- live agent tasks before observing both full-context capability and compaction damage;
- a large new corpus before the tiny-model and three-item production gates pass;
- simultaneous GPU jobs;
- an H100 without measuring that its throughput advantage exceeds its price ratio;
- any run that does not save the exact render/state/provenance needed for reuse.

## Final recommendation

Do not authorize $200 at once.

1. Complete the $0 artifact rescoring and causal apparatus.
2. Run the six-context local 4-bit F/R/W/V/D canary.
3. If it is mechanically sound and scientifically informative, authorize only the first $50 stage.
4. Continue to $100 only if Qwen3-Coder shows both capability and an A-B compaction gap.
5. Continue to $200 only if one treatment separates from fresh restart and matched controls and the common-prefix agent canary has predictable costs.

The likely high-value outcomes are all publishable if framed honestly:

- **R helps, V does not:** latent continuity exists, naive value grafting destroys coherence.
- **R and V help, W/D do not:** the original idea survives a strong causal test.
- **R helps offline but not live:** state carries information that the agent does not use effectively.
- **No R effect in a capable/damaged regime:** ordinary untrained summaries for this checkpoint do not form a useful hidden channel despite the broader literature.
- **Agent E improves normalized tests or resolution:** the first direct evidence that retained state mitigates real compaction damage.

The project should now optimize for discriminating among those outcomes, not for recovering a positive headline.
