# Schedule, shape, precision, and position sensitivity — repository forensic audit

**Author:** Carver — OpenAI GPT-5.6 Sol, xhigh reasoning effort

**Date:** 2026-07-11

**Scope:** Read-only forensic trace of prior observations involving batched versus
stepwise execution, 4-bit prefill length, RoPE/position movement, SDPA versus
eager attention, device sensitivity, and the current natural `c10` schedule
failure.

**Status:** Evidence audit, not a preregistration and not an experiment result.

## Executive conclusion

The repository contains good evidence for a **family of deterministic numerical
execution sensitivities**, but it does not establish one universal root cause.
Three mechanisms must remain separate:

1. **Call-shape/backend/device trajectory sensitivity:** changing query
   partition, decode batching, attention backend, or device can change computed
   states despite identical token IDs and positions.
2. **Quantized projection length sensitivity:** the MLX 4-bit observation begins
   at layer 0, before attention can be responsible, and is therefore a different
   mechanism from the current `c10` failure.
3. **Lossy positional transformation:** re-rotating already stored post-RoPE
   keys is algebraically exact in real arithmetic but not functionally exact at
   finite precision.

The most important new synthesis is the discontinuity inside the same current
local apparatus:

- `Qwen3-0.6B`, CPU, bf16, eager attention passed **7/7 synthetic schedule
  fixtures at aggregate discrepancy exactly `0.0`**, including 4,096-, 4,097-,
  and 8,193-token fixtures.
- The natural 8,430-token `c10` prefix under the same model/backend/device
  produced K/V maxima `16.125/5.125`, last-logit maximum `0.59375`, selected
  margin shift `0.060546875`, and continuation-logit maximum `0.84375`.
  Layer 0 K/V were exact; layer 1 first differed at K/V
  `0.0625/0.0009765625`.

Therefore **length alone is falsified as the explanation**. Content and/or the
exact 23-token query boundary interacts with execution shape. Query-shape-
dependent bf16 attention rounding is the leading theory, but it was not yet an
observed root cause at the time of this audit. A subtle causal-mask or
cache-shape bug remained plausible until the frozen three-way diagnostic.

## Evidence-strength labels

- **A — committed machine evidence:** A committed raw result contains the
  relevant measurements and enough configuration/provenance to audit the
  observation.
- **A− — committed machine evidence with a material provenance limitation:**
  The numbers are preserved, but a relevant code/backend/runtime fact must be
  reconstructed from adjacent code or documentation.
- **B — committed code and contemporaneous prose only:** The observation is
  recorded and the producing code is preserved, but raw stdout/result evidence
  is absent or does not identify the exact runtime configuration.
- **C — summary-only evidence:** The observation survives in a generated
  transcript summary or retrospective prose, but the underlying result could
  not be unambiguously located. It is corroborating operational evidence, not a
  result suitable for a paper claim.

## Observation and provenance table

| Phenomenon | Strength | Exact committed evidence | What was observed | Relationship to current `c10` |
|---|---:|---|---|---|
| MLX one-token forward versus batched prefill | B | [`DECISIONS.md`](../DECISIONS.md), line 31, and [`src/l0_identity.py`](../src/l0_identity.py); observation/source commit `9135af10` | On the local fp16/4-bit MLX path, recomputing a position in a one-token call did not reproduce the batched-prefill logit; contemporaneous prose records max absolute difference about `0.5`. The L0 code consequently reuses the original prefill logits when testing cache round-trip identity. No raw result or stdout is committed. | Same broad execution-decomposition family, but not a demonstrated shared root. It changes batched Q to q=1 under MLX quantized kernels; `c10` uses HF bf16 eager CPU and large split prefills. |
| MLX 4-bit prefill-length sensitivity | B | [`src/lh_rerotate_identity.py`](../src/lh_rerotate_identity.py), source commit `6a3be96`; [`DECISIONS.md`](../DECISIONS.md), line 29, commit `0eae5fc7` | The same prefix tokens at the same positions, once alone and once as the prefix of a longer prefill, produced layer-0 key differences up to about `4.0`; equal-length calls were reported bit-identical. The exact runtime model is not auditable from a result artifact: the script defaults to 4B but permits an environment override. | **Separate root.** Layer-0 divergence precedes attention and implicates quantized projection/GEMM sequence-length behavior. `c10` layer 0 is exact and first differs at layer 1. This is an analogy, not an explanation of `c10`. |
| Incremental-cache split-prefill sensitivity | B | [`DECISIONS.md`](../DECISIONS.md), lines 81 and 88; corrective commit `a9aa85d4` | An early CPU-fp32 “bit-exact” incremental-cache claim was corrected. Retrospective prose records about `1e-4` fp32 logit drift across split prefills, while 6/6 bf16 greedy continuations remained token-identical in the tested worst-case splits. The earlier MPS discrepancy was itself described as a mixture of cross-process nondeterminism and a broken test control. No raw validation artifact was committed in the correction. | Same broad decomposition family only. It supports “repeatable at a fixed schedule does not imply invariant across schedules,” but does not establish the cause or magnitude of `c10`. |
| Batched native-render decode versus per-token decode | C | Batched implementation in [`src/cross_arch_probe.py`](../src/cross_arch_probe.py), commit `68b2b7e`; launcher [`scripts/job_batchtest.sh`](../scripts/job_batchtest.sh), commit `ec48633`; generated archive note [`notes/2026070872-conversation-user-opus48.md`](2026070872-conversation-user-opus48.md) | The archive says batched decoding was about 2.3× faster but changed bf16 greedy text. It reports referent lift about `+0.0485` with a zero-spanning interval and `evicted_fact` about `−0.305`, versus the per-token referent midpoint near `+0.10`. I could not locate an unambiguous committed batchtest result or log containing those exact aggregate claims. | A downstream manifestation of batch/padding/trajectory sensitivity, not a demonstrated common root. Autoregressive token differences compound, and left-padded batch execution differs from `c10`'s same-token teacher-forced comparison. The conservative decision to use per-token rendering remains justified. |
| RoPE position movement and finite-precision re-rotation | A | [`results/coherent_state_diagnostics/numeric_Qwen3-0.6B_20260711T065027Z.json`](../results/coherent_state_diagnostics/numeric_Qwen3-0.6B_20260711T065027Z.json) and [`src/diagnose_coherent_numeric.py`](../src/diagnose_coherent_numeric.py), commit `83bca9d8` | For a position delta of 37, helper-versus-model-native destination target-margin error was `0.0` in fp32, `0.0029296875` in fp16, and `0.1015625` in bf16; K maxima were about `0.0000339`, `0.03125`, and `0.25`. Round-trip K maxima were about `0.0000305`, `0.015625`, and `0.125`. Omitting rotation was catastrophic, with target-margin errors about `1.57–1.69`. | **Separate phenomenon.** This is finite-precision error from algebraically moving an already stored post-RoPE key. It is not caused by query partitioning, although both live in the broader family of finite-precision state non-equivalence. |
| Exact 30B bf16 SDPA, five tokens in one call versus `2+3` | A− | [`results/coherent_state/coherent_state_gapped_v3_Qwen3-30B-A3B-Instruct-2507_20260711T080431Z/manifest.json`](../results/coherent_state/coherent_state_gapped_v3_Qwen3-30B-A3B-Instruct-2507_20260711T080431Z/manifest.json) and [`production_kernel_gate.json`](../results/coherent_state/coherent_state_gapped_v3_Qwen3-30B-A3B-Instruct-2507_20260711T080431Z/production_kernel_gate.json), artifact commit `0becf81e`; recorded run-code commit `9cdb2302`; interpretation in [`COHERENT-STATE-PREREGISTRATION-AMENDMENT-4.md`](../COHERENT-STATE-PREREGISTRATION-AMENDMENT-4.md), commit `22f5988f` | On an A100 with exact Qwen3-30B-A3B-Instruct-2507 revision and bf16 parameters, the five-token `[5]` versus `[2,3]` comparison gave K `1.625`, V `0.4501953125`, and logits `1.125` against `5e-4`. The run failed before any semantic outcome. The artifact itself does not contain a full backend fingerprint; SDPA is recovered from the exact code/default and Amendment 4. | Likely the same **operational class** as the controlled local SDPA result below, but the low-level cause is not proven. MoE expert-route amplification was never measured. It is not the same established root as `c10`, which occurs on dense 0.6B eager CPU. |
| Local bf16 SDPA versus eager, five tokens in one call versus `2+3` | A | Production-fidelity [`results/coherent_state_diagnostics/bf16_attention_schedule_Qwen3-0.6B_20260711T085849Z.json`](../results/coherent_state_diagnostics/bf16_attention_schedule_Qwen3-0.6B_20260711T085849Z.json), [`src/diagnose_bf16_attention_schedule.py`](../src/diagnose_bf16_attention_schedule.py), commit `c146e545`; the `085802Z` artifact is explicitly preliminary | On exact Qwen3-0.6B CPU bf16, SDPA with automatic and independently explicit 4D masks produced K/V `1.0/1.0`, logits `0.4375`, and margin shift `0.109375`; eager automatic/explicit comparisons were all exactly `0.0`. Identical repeats reproduced. | Demonstrates a backend-by-query-shape interaction for this exact five-token fixture. The unchanged result under explicit mask weakens a simple mask-construction explanation. It does **not** prove that SDPA is universally sensitive or eager universally invariant; `c10` directly falsifies that extrapolation. |
| Eager CPU versus eager MPS, 64 tokens in one call versus `32+32` | A− | [`results/coherent_state_diagnostics/eager_device_schedule_Qwen3-0.6B_20260711T092109Z.json`](../results/coherent_state_diagnostics/eager_device_schedule_Qwen3-0.6B_20260711T092109Z.json) and [`src/diagnose_eager_device_schedule.py`](../src/diagnose_eager_device_schedule.py), commit `270f71b6` | CPU was exact `0.0`. MPS gave K `0.375`, V `0.62109375`, logits `0.1875`, margin shift `0.046875`; its common continuation gave K `0.375`, V `0.625`, and logits `0.28125`. The artifact records the diagnostic source as untracked at run start; source and artifact were committed together later. | A separate device implementation effect. `c10` occurs on CPU, so MPS is not required. It proves that the label “eager” alone does not name one universal numerical implementation. |
| Current v10 synthetic eager schedule battery | A | [`results/coherent_state_ladder/coherent_state_ladder_gapped_v10_Qwen3-0.6B_20260711T123246Z__stage_synthetic_schedule_fixtures.json`](../results/coherent_state_ladder/coherent_state_ladder_gapped_v10_Qwen3-0.6B_20260711T123246Z__stage_synthetic_schedule_fixtures.json), commit `858b3049` | Same 0.6B bf16 eager CPU apparatus passed 7/7 at aggregate `0.0`: lengths 5, 64, 900, 4,096, 4,097, 8,193, plus the 64-token logical-gap/q1 fixture. The 8,193-token comparison was `[4096,4096,1]` versus `[32,4096,4065]`. | Crucial negative evidence. Eager CPU schedule sensitivity is not universal, and prefix length alone cannot explain `c10`. Repetitive synthetic token content and the tested boundaries may simply fail to excite the divergent trajectory. |
| Natural `c10` eager schedule failure | A | [`results/coherent_state_ladder/coherent_state_ladder_gapped_v10_Qwen3-0.6B_20260711T123246Z__stage_committed_case_schedule_fixtures.json`](../results/coherent_state_ladder/coherent_state_ladder_gapped_v10_Qwen3-0.6B_20260711T123246Z__stage_committed_case_schedule_fixtures.json), commit `4ad714f5`; frozen follow-up [`COHERENT-STATE-SCHEDULE-ORIGIN-DIAGNOSTIC-PREREGISTRATION.md`](../COHERENT-STATE-SCHEDULE-ORIGIN-DIAGNOSTIC-PREREGISTRATION.md), commit `649b41e`; diagnostic implementation commit `43f1679` | Same 8,430 token IDs and positions under ordinary `[4096,4096,238]` versus message-aligned `[23,4096,4096,92,123]` produced K `16.125`, V `5.125`, logits `0.59375`, margin shift `0.060546875`, continuation logits `0.84375`, continuation K/V `1.0/1.59375`. Structural checks passed. Layer 0 K/V were exact; layer 1 first differed at K/V `0.0625/0.0009765625`. | Root was **not yet observed** when this audit was performed. Query-shape-dependent bf16 attention rounding was the leading theory. No completed `results/c10_schedule_origin/` artifact existed at audit time. |

## What is known, narrowed, and still theory

### Known from committed machine evidence

- Identical token IDs and positions do not guarantee identical K/V or logits
  across different execution decompositions.
- The sensitivity can be deterministic: exact repeats can agree while two
  different schedules disagree.
- Attention backend and device materially change whether a particular schedule
  contrast is exact.
- SDPA is not required for schedule sensitivity: natural `c10` fails under
  attested eager CPU execution.
- MoE routing is not required for schedule sensitivity: natural `c10` fails on
  a dense 0.6B model.
- Conversely, eager CPU does not always diverge: the entire v10 synthetic
  schedule battery, including 8,193 tokens, is exactly equal.
- The `c10` divergence appears after the first attention computation: layer 0
  stored K/V are exact, while layer 1 differs.
- Post-RoPE key movement is not functionally exact at finite precision, even
  when cosine similarity is extremely high.

### Narrowed or disfavored, but not logically eliminated

- **Length-only explanation:** falsified by the exact 8,193-token synthetic
  fixture and failing 8,430-token natural fixture.
- **Gross token-ID, position-ID, or initial-projection mismatch:** disfavored by
  complete source/position records, structural checks, and layer-0 equality.
- **Simple automatically generated causal-mask error:** disfavored in the
  five-token SDPA diagnostic because an independently explicit 4D mask produced
  the same discrepancy.
- **SDPA-only explanation:** falsified as a universal explanation by eager
  `c10`.
- **MPS-only or GPU-only explanation:** falsified by eager CPU `c10`.
- **MoE expert-routing as a necessary cause:** falsified by dense 0.6B `c10`.

### Still theory at audit time

- **Query/block-shape-dependent bf16 attention rounding** is the leading `c10`
  explanation because A/B tokens and positions match, layer 0 matches, and the
  first difference follows attention. It remained a theory until the frozen
  three-way diagnostic.
- **A subtle causal-mask or cache-shape bug** remained plausible. Such a bug can
  begin after attention while leaving layer-0 K/V exact and may be triggered by
  a particular content/boundary combination that the repetitive synthetic
  fixtures do not exercise.
- **Content-dependent numerical amplification** is plausible: non-repetitive
  natural attention scores can cross softmax, normalization, activation, or
  later routing boundaries that a five-token cycle does not. This was not
  directly instrumented.
- **MoE routing amplification** is plausible for the 30B SDPA magnitude but was
  never measured. No artifact records per-token expert-selection agreement.
- **Common-mode cancellation in semantic arm contrasts** is possible but must
  be measured. A large state difference does not prove a large difference in
  `G_correct-G_fresh` or `G_correct-G_wrong`; neither does it prove cancellation.

## Consequences for earlier mechanistic claims and artifacts

### 1. Literal “exact key re-rotation” claims are invalid

[`FINDINGS.md`](../FINDINGS.md), lines 214–221, introduced in commit
`2e5612ad`, says the K-graft is “not an approximation” and keys can be moved
“exactly.” [`notes/2026070516-phase2-results.md`](2026070516-phase2-results.md)
says packed keys were re-rotated “exact, validated.” Those claims confuse exact
RoPE composition in real arithmetic with finite-precision implementation and
downstream equivalence.

The licensed claim is narrower: the helper applies the correct algebraic
position transformation and can achieve very high cosine similarity. At bf16,
however, the committed diagnostic observes `0.25` K maximum and a `0.1015625`
target-pair margin error against the native same-destination oracle.

### 2. H-pack identifies a composite policy, not a clean semantic-state channel

The `B-min-pack` versus `H-pack` comparison controls visible packed layout, but
it replaces native fresh packed keys with write-time keys after lossy
re-rotation and uses generation-time/stepwise source state rather than a
schedule-matched batched re-encoding. Therefore the behavioral effect remains a
description of **packed, transformed write-time KV retention**. It does not
cleanly isolate hidden semantic information written by history.

The statement in [`FINDINGS.md`](../FINDINGS.md), lines 119–120, that the
decomposition earns “write-time KV specifically” requires this qualification.
It does not need deletion; its mechanism must be narrowed.

### 3. K-only “closure” is only about the tested transformed-key policy

The robust-metric correction in [`FINDINGS.md`](../FINDINGS.md), lines 420–433,
commit `2afd8764`, already retracts the dramatic “keys hurt” conclusion and
finds K-only effects near zero. The position diagnostic adds another limit:
the tested K-only arm used re-rotated keys, so a null cannot show that native old
keys contain no useful addressing information.

Licensed: the particular approximate packed K-graft did not improve the tested
task. Unlicensed: keys are intrinsically inert, the key channel is universally
closed, or native same-position old keys cannot help.

### 4. Write-time versus reread state was not isolated when source schedules differ

Generation-time summary K/V are created stepwise after the original history.
Fresh summary K/V are commonly created by a batched reread. The accumulated
schedule evidence shows that identical tokens and positions can acquire
different states solely from execution decomposition.

Consequently, `E-B` or `H-pack-B-min-pack` cannot alone attribute a state or
behavioral difference to history-conditioned semantics. Correct-versus-wrong
sources created under the same schedule can still support source/content
specificity, and identically shaped downstream scoring remains a valid control,
but neither isolates the original causal question. The same-position,
same-decomposition gapped assay was designed to close this gap.

### 5. The eager “rescue” is valid only for its exact fixtures

Amendment 4 carefully says that the local result identifies backend and query
schedule as variables “in the tested fixture.” Later language such as “eager
rescue empirically confirmed” or “eager directly removes the observed source”
is too broad.

The natural `c10` failure means that a 7/7 exact synthetic eager battery cannot
authorize long natural prefixes. No eager-based v4–v10 artifact can use the
synthetic battery to repair the permanent `c10` v10 failure.

### 6. SDPA and MoE claims must remain scoped

The controlled 0.6B diagnostic genuinely implicates SDPA in the five-token
fixture. The 30B SDPA failure is directionally consistent, and MoE expert-route
flips could amplify it. But route flips were not logged, and dense eager `c10`
proves that neither SDPA nor MoE is necessary for large schedule divergence.

Claims such as “SDPA was the bug,” “MoE routing is the smoking gun,” or “eager
solves the problem” exceed the evidence.

### 7. Fixed-schedule determinism claims can remain, with precise wording

The statement that `live1 == live2` within one environment can remain if it
means exact repeated pipeline executions. These observations are compatible
with deterministic schedule dependence.

The correct wording is **repeatable at a fixed schedule and environment**, not
decomposition-invariant. The incremental-cache v1 bit-exact claim was already
properly corrected in commit `a9aa85d4`.

### 8. The original same-shape guard is necessary but incomplete

Identically shaped downstream teacher-forcing protects cross-arm logit
comparisons from the observed MLX 4-bit layer-0 prefill-length floor. It does
not schedule-match the provenance of the K/V being injected.

“Same-shaped scoring” and “same-shaped source-state creation” are different
controls. Earlier notes sometimes treated the former as if it established the
latter.

### 9. Batched-render aggregates should not be promoted without their raw artifact

The batched-render sensitivity is useful operational corroboration, and forcing
`SC_BATCHED_RENDER=0` for a sign-sensitive sweep was conservative. The specific
`+0.0485` and `−0.305` aggregates should not become paper evidence unless the
original result/log is found and tied to its exact code/model/runtime.

### 10. Schedule sensitivity does not automatically void all earlier behavior

Same-schedule arm contrasts remain descriptive of the exact policies that were
run. `c10` does not retroactively change recorded 4-bit answers or scores. What
fails is the stronger mechanistic inference that cross-schedule differences
must reflect history-conditioned semantics rather than the source execution
trajectory.

## Minimal discriminating checks

### Check 1 — frozen three-way `c10` origin diagnostic

Use exact Qwen3-0.6B revision, CPU bf16 eager, exact `c10` rows `0..4095`, and
two bit-identical repeats per branch in order `A1,A2,B1,B2,C1,C2`:

- **A:** c10 rows `0..22` in one 23-token call.
- **B:** original c10 rows `0..4095` in one 4,096-token call.
- **C:** same 4,096-token shape and same c10 rows `0..22`, with rows
  `23..4095` replaced by exact same-index c02 prefix IDs.

Compare A/B and B/C on the first 23 rows at every layer, with a required B/C
tail mutation/cache-difference positive control.

The first matching interpretation is:

| Observation | Licensed diagnosis |
|---|---|
| Any A/B layer-0 difference | Construction/position/cache/projection divergence; do not interpret later layers. |
| Any B/C difference in rows `0..22` | Causally future content influenced early rows; investigate mask/backend behavior. |
| B/C first 23 exact; A/B first differs at layer ≥1 | Query/block-shape-dependent bf16 execution rounding in the tested fixture. |
| A/B and B/C first 23 both exact | The first boundary is not the origin; run only the predeclared fallback. |

This check localizes the apparatus failure. It does not estimate a semantic
effect and cannot authorize a semantic run.

### Check 2 — predeclared full-prefix fallback only if the first 23 rows are exact

On exact c10 full prefix, compare partitions:

- ordinary `[4096,4096,238]`;
- isolated-first-boundary `[23,4073,4096,238]`.

Repeat each branch twice and locate the earliest divergent row/layer/component
over all 8,430 rows. Do not introduce additional adaptive branches after seeing
the three-way result.

### Check 3 — estimand-level schedule stability

Root localization is not the scientific decision. Under a separately frozen
design, build the same semantic arms under canonical and alternative source
schedules, insert them into identical destinations, and measure schedule shifts
in at least:

- `G_correct-G_fresh`;
- `G_correct-G_wrong`.

Report the schedule-by-arm difference-in-differences and the schedule-robust
intersection of conclusions. Do not assume the artifact is common-mode and do
not assume its K/V maximum translates linearly to the target contrasts.

Stable contrasts may justify a redesigned, schedule-frozen assay. Shifts at the
plausible effect scale mean the mechanism is not identifiable at this precision
and implementation, regardless of whether the low-level diagnosis is called
rounding, backend selection, or cache-shape sensitivity.

## Final assessment

The repository's early numerical warnings were directionally valuable: call
shape matters, quantized kernels are shape-sensitive, positional key movement
is not free, and fixed-schedule repeatability is not cross-schedule identity.
The mistake was occasionally treating these as one interchangeable “kernel
noise” mechanism or treating a small eager fixture as a general repair.

The honest synthesis is narrower and stronger:

- there is a reproducible family of finite-precision execution sensitivities;
- the exact causal mechanism is configuration-specific;
- the 4-bit layer-0 floor, RoPE re-rotation error, SDPA five-token failure,
  MPS eager failure, and eager natural-`c10` failure must not be conflated;
- natural `c10` permanently falsifies v10's long-prefix equivalence condition;
- the three-way origin diagnostic can name the local apparatus failure;
- only estimand-level schedule stability can determine whether the original
  semantic question remains measurable.
