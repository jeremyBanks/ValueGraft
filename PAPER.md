# Transplanting KV-Cache State Across Conversation Compaction: Inconclusive Evidence and a Methodological Failure Catalogue

**Jeremy Banks · Anthropic Claude Fable 5 · OpenAI GPT-5.6 Sol (extra-high reasoning)**

*Working-paper status: under factual, methodological, and readability review. The published repository landing page remains unchanged until every review gate passes.*

---

## Abstract

When a long LLM conversation is compacted — replaced by a text summary so that work can continue in a fresh context — everything the model computed while generating the original conversation is discarded along with the text. Recent work provides evidence, in trained settings, that generation-time key/value (KV) state can carry task-relevant information that re-encoding the visible text does not recover. This project asked a narrower, practical question: can a **training-free, post-hoc transplant** of old KV state (in particular, old value vectors under fresh keys) into a compacted context reliably mitigate compaction damage on an ordinary instruction model?

The answer we can support is: **the project did not establish practical benefit, and the clean literal proposal remains unresolved.** Across four evidence strata that must not be pooled, trustworthy performance evidence was inconclusive, heterogeneous, or control-incomplete. The strongest surviving performance lead is a small out-of-fitting likelihood effect on a coding-trajectory proxy (+0.0135 nats/token, nominal 95% bootstrap CI [+0.0083, +0.0191]) with no matched placebo, no executed action, and no task-success signal. The final exact-state mechanism experiment formally stopped at a preregistered technical gate; a single permitted post-stop diagnostic produced a weak, schedule-sensitive, placebo-uncontrolled value-only trace with no behavioral recovery. No experiment simultaneously combined subject-native conversation generation, actual incremental-state capture, a naturally triggered compaction boundary, matched controls, and a behavioral endpoint. We report these measurements and limitations, and what we believe is the project's most durable contribution: a catalogue of ten methodological failures. Some moved numerical readouts at the scale of the hoped-for effect; others invalidated controls, provenance, or repairability. Several survived elaborate hash, test, and release machinery before being caught.

---

## 1. Introduction

Conversation compaction is now routine infrastructure: when an agent's context fills, the transcript is summarized, the summary replaces the history, and the agent continues. The visible information loss is obvious. The less obvious loss is computational: the KV cache the model built while *generating* its own turns — attention state that may encode intermediate conclusions, bindings, and commitments not fully explicit in the text — is thrown away and rebuilt by re-encoding the summary from scratch.

Recent preprints give direct evidence that generation-time state can matter. MEMENTO (§2) reports a 15.3-point accuracy drop when a trained dual-stream model is forced to restart from re-prefilled text instead of retaining its full memento KV. But MEMENTO's channel is *trained* and retains *full* KV written with the original context visible. The question this repository set out to answer is narrower:

> Can old KV state, captured from an ordinary instruction model with no retraining, be transplanted post hoc into a *different* (compacted) context — old values under fresh keys, at matched positions — and measurably restore what compaction destroyed?

The intended endpoint was always practical mitigation: the project owner's goal from the outset was to reduce compaction damage in real agent workloads, with mechanism experiments as the gate to a paired agent evaluation, not as the destination.

This paper reports that the gate was not cleared. We organize the evidence by stratum (§3), because the four bodies of evidence here differ in subject model provenance, source-state procedure, intervention region, controls, and formal status, and pooling them would manufacture confidence none of them individually supports. We then report the mechanism redesign that was meant to resolve the question cleanly, why it formally stopped before its treatment arm (§7), what the one permitted diagnostic showed and failed to show (§8), and the failure catalogue (§9). We close with what is and is not established, and what a defensible agent evaluation would require (§10–11).

Three things this paper does **not** claim, stated up front:

- It does not claim the state channel is absent, or that transplantation can never work. Most of our confidence intervals permit modest effects; a formal stop is not a null result.
- It does not claim equivalence anywhere. "No detected average lift" is the strongest negative reading we use.
- It does not claim any agent-level or task-success effect, positive or negative. No valid paired live-agent evaluation of a treatment that cleared the mechanism gate was run (§11); earlier exploratory agent runs had separate capability, compaction, and harness confounds and are not efficacy evidence.

---

## 2. Background and related work

**MEMENTO: Teaching LLMs to Manage Their Own Context** ([arXiv:2604.09852](https://arxiv.org/abs/2604.09852)) trains a model to manage its own context with custom in-place block masking, retaining full "memento" KV state written while the original block was visible. On AIME24 with Qwen3-8B, normal memento attention scored 66.1% versus 50.8% after restart/re-prefill — a −15.3-point ablation (with the disclosure that the normal condition used 64 repetitions and the restart condition 8). This is strong evidence from one recent preprint that generation-time KV state can carry information beyond restart text in a *trained, full-KV* setting; it is not an independent replication and does not validate the untrained, partial, post-hoc transplant studied here.

**Models Take Notes at Prefill: KV Cache Can Be Editable and Composable** ([arXiv:2606.17107](https://arxiv.org/abs/2606.17107)), a recent single-author preprint, causally locates conclusions on downstream aggregator/delimiter tokens across model families: in its tasks, the KV of the field containing the answer-relevant content drives under 1% of the decision, and downstream recomputation restores it. If that localization pattern holds more broadly, a transplant confined to summary-region rows plausibly targets the wrong locus entirely — the information may live downstream of the rows we replaced. It is, however, not a direct test of our carrier construction or of agent settings.

**CacheBlend** ([arXiv:2405.16444](https://arxiv.org/abs/2405.16444)) reuses cached text chunks while *selectively recomputing* tokens to restore cross-context interactions. It provides a useful systems contrast to uncompensated transplantation; it does not directly test our intervention or establish why our measurements were weak.

**Cache-to-Cache** ([arXiv:2510.03215](https://arxiv.org/abs/2510.03215)) learns a projection and gating mechanism to fuse a source model's cache into a target model. It is directly adjacent evidence that cache-state transfer can be useful with learned alignment, but differs from our within-model, training-free replacement.

**Learning to Compress Prompts with Gist Tokens** ([arXiv:2304.08467](https://arxiv.org/abs/2304.08467)) trains models to encode prompts into reusable gist-token state. It is learned latent compression, not post-hoc state salvage, and marks the "with training, this is achievable" end of the design space.

**Novelty claim, narrowed.** The broad premise — state beyond text — is prior work and is not ours. In a scoped search centered on learned compaction, cache composition, and latent cache communication, we did not find this exact combination: a *training-free, post-generation* transplant of old-state components (values, keys, or both) across a *text-compaction* boundary on an unmodified instruction model. We do not claim a systematic literature review. The contribution is the evaluation record and methodological catalogue, not the broad state-channel premise.

---

## 3. The question, the estimands, and the causal ladder

The claim ladder the project used, from weakest to strongest:

1. **State difference and possible channel** — bitwise differences between generation-time and re-encoded state are expected under finite-precision computation; whether they carry useful information beyond text is a separate empirical question.
2. **History specificity** — state written by *different histories* under *identical visible text* produces measurably different downstream readouts or output distributions.
3. **Component localization** — the difference is attributable to specific components (values vs. keys, specific layers/regions).
4. **Intervention utility** — transplanting the component recovers a meaningful fraction of compaction damage under controls (placebo, selectivity).
5. **Agent utility** — the intervention improves real task outcomes in paired agent runs.

This project's contested territory is rungs 2–4. Rung 5 was never reached, by design: it was gated on rung 4 and the gate did not clear.

The four evidence strata, which differ in nearly every methodological dimension and are never pooled in this paper:

| Stratum | What actually ran | Strongest licensed reading |
|---|---|---|
| Legacy synthetic (§4) | Foreign/mixed-source conversation bodies; 30B self-generated summaries; prefill-*reconstructed* source state; combined summary+tail value graft | No detected average lift in that exact exploratory apparatus; no equivalence, and no clean generation-state null |
| SWE-Gym/OpenHands proxy (§5) | OpenHands-derived historical trajectories described at acquisition as successful; 30B self-generated brief summaries; actual generation-mutated source snapshot; combined summary+tail value graft | Small out-of-fitting demonstrated-next-action likelihood effect for one selected layer map; no matched placebo, executed action, or task-success result |
| Coherent-state v10/v11 (§6.1) | Local technical fixtures, failed controls, paused draft corpus | Methodological evidence only; no semantic treatment result and no sample |
| Formal v12 + e01 (§6–8) | Exact 30B bf16 canary; fixed authored correct/wrong histories; same fixed carrier text; N/P forced replay | Formal technical stop; one later non-authorizing N=1 diagnostic gives a weak, schedule-sensitive, placebo-uncontrolled value-only hint |

Statistical conventions used throughout: a confidence interval spanning zero is reported as "no detected average lift," never as zero or equivalence; overlapping marginal intervals are never treated as a paired-difference test; fitting, internal evaluation, and external confirmation samples are kept separate; and the v12/e01 material is N=1 descriptive evidence carrying no p-value, interval, or population claim.

---

## 4. Legacy synthetic evidence

### 4.1 What ran, exactly

The legacy apparatus must travel with every legacy number, because it differs from the later redesign in ways that matter:

- **Subject/scorer:** the headline JSONs record `Qwen/Qwen3-30B-A3B-Instruct-2507`; surrounding project records attribute revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, Torch bf16 on A100, but those runtime fields were not born-annotated in every promoted file.
- **Conversation bodies:** c07–c12 were generated by `mlx-community/Qwen3-4B-Instruct-2507-4bit` (temperature 0.7, seeds 1006–1011, with truncation/tail repairs recorded in metadata). c13–c24 were externally authored by assistant sessions whose stored aliases are `sonnet`, `opus`, and `fable`; exact provider, runtime model/version, and effort are not preserved in those files. The 30B subject did **not** generate these bodies. System prompts, user turns, planted facts, probes, and gold answers were AI-assisted synthetic authorship under human direction — not hand-authored, and not produced by the experimental subject.
- **Summaries:** generated by the 30B subject itself.
- **Source state:** the harness discarded the actual incremental generation snapshot and *reconstructed* old state by prefill — a provenance defect the redesign later fixed.
- **Intervention:** the value graft covered retained-tail **plus** summary regions, not summary only; alignment was difflib positional-within-region.
- **Selection lineage:** the per-layer map reconstructs from its committed profile. The evaluated 109-slot per-head map does not: the cited committed profile selects 121 slots, with only 80 overlapping, and no nonempty subset of its ten rows reproduces the evaluated map. The later pod job likely re-derived an unpreserved profile; this is a reconstruction, not a recovered derivation. Intersection and union reconstruct from the embedded head map plus the layer map, so all four interventions are known, but selection and held-out separation are not independently auditable for the three head-derived variants.
- **Known provenance gaps:** headline manifests often record `git_commit: null`; the head-derived fitting profile/config/log is missing.

### 4.2 Held-out average: no detected lift

Four fixed graft variants, evaluated on 18 conversations outside the recorded intended fitting corpus and containing 203 scored planted targets, gave these graft-minus-compacted estimates (nats/token; nominal conversation-clustered 95% intervals). The per-layer variant has reproducible selection lineage; the three head-derived variants do not, as detailed above. Each plant averages 1, 3, or 4 phrasing-level probe scores before entering the plant-weighted aggregate; the artifacts contain 531 phrasing-level scores in total.

| Variant | Mean | 95% CI |
|---|---:|---:|
| Per-head | +0.017 | [−0.027, +0.062] |
| Per-layer | −0.003 | [−0.041, +0.043] |
| Intersection | +0.008 | [−0.032, +0.052] |
| Union | −0.012 | [−0.053, +0.034] |

This supports exactly one sentence: *no detected average lift in the apparatus that ran.* The upper bounds still permit modest positive effects; nothing here is a definitive null, a refutation, or an equivalence result — and because the source state was prefill-reconstructed, this is not a clean test of generation-time state either.

The post-hoc source blocks differed in sign and magnitude:

| Variant | c07–c12: 4B-rendered (6 conversations / 57 plants) | c13–c24: assistant-authored, stored aliases only (12 conversations / 146 plants) |
|---|---:|---:|
| Per-head | +0.071 [−0.031, +0.171] | −0.004 [−0.044, +0.039] |
| Per-layer | **+0.120 [+0.055, +0.181]** | **−0.050 [−0.076, −0.028]** |
| Intersection | **+0.114 [+0.035, +0.176]** | **−0.033 [−0.067, −0.004]** |
| Union | +0.064 [−0.025, +0.154] | **−0.041 [−0.080, −0.004]** |

This post-hoc split is descriptive, not a causal source-family comparison or a direct interaction test. The two blocks also differ in length, authoring process, plant construction, and similarity to the Qwen-rendered tuning corpus, and only six clusters support the positive block. The pooled no-detected-lift estimate therefore hides opposing block estimates; source family, length, construction, tuning-distribution similarity, or some combination may explain them. The split neither proves model nativeness nor licenses a universal negative claim.

### 4.3 Placebos, read narrowly

The correct old-state graft often outperformed position-shuffled and Gaussian source controls. The safe interpretation is only that aligned source values were **non-exchangeable** — often less disruptive than grossly mismatched full vectors. These controls did not match the actual treatment delta (V_old − V_fresh): Gaussian rows matched source energy rather than the delta, shuffles were not proven strict derangements, and correct old rows may simply have been geometrically closer to fresh rows. We therefore do not use these placebos as evidence of content-specific semantic information or of "a real mechanism."

### 4.4 Retired legacy claims

Several earlier headline claims from this stratum are void or retired, and we list them so they are not re-cited:

- **The four-level compression treatment conclusion is void.** Ultra/brief/medium cells generated summary text under level-specific requests but reconstructed old state under the default realistic request; only the realistic cell was internally consistent. Compression ratios remain descriptions of text. No flat-across-30× or severity conclusion is licensed.
- **The +10–12 percentage-point judged-recovery headline is retired.** It came from a local 4-bit-weight MLX stack with fp16 KV cache, brief/adversarial summaries, conversations c01–c12, and was render-fragile: a clean rerender moved judged sense from +8.7 to +1.0 points and referent from +9.7 to +5.2, with intervals spanning zero. Its failure to reproduce at bf16 does not identify whether weight quantization/runtime, corpus, or the corrected apparatus explains the difference; a clean precision interaction remains untested.
- **The packed "H-pack" construction is not evidence for value-only recovery.** It changed fabrication/admission behavior but simultaneously altered layout, transformed K, and V; it restored 0/24 evicted facts, and the earlier "38/48 accurate" claim was unreproducible.
- **Cross-architecture rows support no law.** Heterogeneous gates and provenance do not support a QK-norm, dense/MoE, keys-neutral, or shared-direction generalization.

---

## 5. The coding-domain likelihood proxy (SWE-Gym/OpenHands)

The strongest surviving performance lead in the repository is a **likelihood proxy on historical coding-agent trajectories**, and we are careful to describe it as exactly that.

**Setup.** The source paper describes the released 491 SWE-Gym/OpenHands trajectories as successful, rejection-sampled rollouts from `gpt-4o-2024-08-06` and `claude-3-5-sonnet-20241022`; the local ignored/untracked parquet preserves only a `messages` column, so row-level generator attribution and the immutable upstream revision are unavailable. The 30B subject did not generate the trajectory bodies. It generated a brief compaction summary; unlike the legacy stratum, the source state used by the graft was the *actual generation-mutated snapshot*. The intervention combined summary and retained-tail values. A layer map (α = 1 on layers 12–17 and 30–35) was selected on 41 fresh-pool trajectories, evaluated on 57 disjoint fresh trajectories, and partially confirmed on 45 of a planned 75 original-pool trajectories. The metric is teacher-forced mean token log-probability of the *historically demonstrated next action*.

**The fixed scalar under the brief summary.** Two runs scored the same original 75 IDs under the same brief-summary request but different prefill schedules, so they are not repeats and must not be averaged. The legacy single-call run gives +0.0156 nats/token, nominal 95% bootstrap CI [+0.0049, +0.0271]; the later 4,096-token-chunked run gives +0.0133 [+0.0016, +0.0263]. Summary-token counts differ for 64/75 IDs (t0001: 84 versus 131); the paired chunked-minus-single-call graft-effect contrast is −0.0023 [−0.0110, +0.0065], an apparatus contrast rather than an isolated schedule effect because the summaries and downstream states were regenerated.

The schedule-matched disjoint 98-trajectory chunked pool gives −0.0017 [−0.0136, +0.0089]. Pooling it only with the chunked original 75 gives +0.0048 [−0.0036, +0.0130] across 173 unique trajectories. The disjoint-minus-original chunked contrast is −0.0150 [−0.0319, +0.0015], so the heterogeneous point estimates do not constitute a detected pool difference. Source, task, index, or other composition differences remain unidentified from the preserved rows. A separate realistic/default-summary scalar result is reported below and must not be pooled with these brief-summary runs.

**Selected-map results (map minus compacted baseline, nats/token):**

- Fresh evaluation (n = 57): **+0.0117**, 95% bootstrap CI [+0.0063, +0.0172].
- Original-pool partial confirmation (n = 45): **+0.0158** [+0.0061, +0.0260].
- Descriptive pool of all 102 rows not used to fit this map: **+0.0135** [+0.0083, +0.0191], 77/102 positive.

The 102 rows are out-of-fitting *for this layer map*, but they are not a pristine preregistered external sample: the research program had inspected the original pool in earlier scalar runs, confirmation stopped at a budget-capped prefix of 45/75, and many analyses preceded this one. Intervals in this section are therefore nominal and pool-conditional, not multiplicity-adjusted confirmation intervals.

**Why this is not an agent result.** Every one of the following caveats binds:

- The target is one historically demonstrated next action, not necessarily the unique correct action.
- No action was applied, no patch tested, no repository task solved.
- The selected map had **no matched placebo** restricted to its selected layers, so map-specific content cannot be separated from map-specific disturbance.
- Against a fixed graft on the fresh 57, the selected map's advantage was inconclusive (+0.0041, CI spanning zero).
- A coarse structural match on tool/path/command was 53% (selected) versus 54% (baseline) on the fresh set; across all 102 out-of-fitting rows the graft fixed three baseline misses and broke three baseline successes — behaviorally a wash at this resolution.
- Confirmation stopped at 45/75 trajectories.
- Bootstrap treated trajectories as independent; a repository/task-cluster sensitivity analysis was not completed.
- A fixed scalar under realistic summaries showed no detected average lift (+0.0006 [−0.0136, +0.0146], n = 75), and the selected map was never tested there.
- Dataset provenance is partially unpreserved: the local `swegym.parquet` hashes `ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1`, but its immutable upstream revision, row-level generator attribution, and per-trajectory generated summary text/hashes were not retained.

Several alternatives fit the selected-map likelihood movement. Layer selection may change generic confidence or calibration; the combined tail+summary intervention does not identify where the signal lives; a teacher demonstration measures imitation likelihood rather than correctness; pool composition may explain the heterogeneous scalar result; and without a map-matched delta control, content-specific recovery cannot be separated from layer-specific perturbation. The structural-match wash supplies no corroborating action-level movement.

The licensed sentence is: *a selected-map, demonstrated-next-action likelihood lead of about +0.013 nats/token on rows not used to fit the map, without a matched placebo, executed action, or task-success endpoint.* It is a reason the question stays open; it is not coding-agent improvement.

---

## 6. The coherent-state redesign

### 6.1 v10/v11: methodological evidence only

The coherent-state v10/v11 effort — a preregistered assay with local technical fixtures and an authored corpus pipeline — produced failed controls and a paused draft corpus, and contributes methodological evidence only (§9). The planned twelve-case v11 corpus build was deliberately paused because no exact-model effect had yet justified confirmatory machinery. There is no semantic treatment result and no sample from this stratum.

### 6.2 v12: the same-visible-text design

The final canary, v12, asked the cleanest version of the question. Two histories — one **correct**, one **minimally counterfactual** — each produce KV state for the *same fixed neutral carrier text*, and a **fresh** arm re-encodes that same carrier text in a position-preserving gapped destination. Within a fixed replay schedule and arm construction, the visible text downstream of the histories is identical, so arm differences must be mediated by the retained state. The primary semantic contrast is correct-history versus wrong-history state (D); correct-history versus fresh is the preregistered within-assay recovery contrast (U), not evidence of practical utility. Full K+V and value-only transplants are separate families, and the intervention operates at matched positions with no re-rotation of quantized keys (a hard-won requirement; see §9, item 4).

The carrier was not a naturally generated summary. The exact request was `Write the fixed neutral handoff note for the next assistant. Output only that note.` The forced assistant note was:

> The prior discussion established the operating context and relevant decision criteria. Continue from this handoff, preserve the existing constraints, and answer later questions from the state available here. No unresolved action is introduced by this note.

The exact anchor user turn was `Acknowledge receipt of this handoff without adding or repeating any factual detail.`, followed by fixed assistant content `Acknowledged.` Every arm then processed a byte-identical retained-tail exchange, causally recomputed after the R1/R2/R3 transplant boundary, before the probes. The fresh destination therefore contained the original system message, carrier request and assistant note, anchor user and assistant turns, and retained-tail messages. Logical positions matched the source while physical storage remained contiguous. Two replay schedules produced the historical source state: **N** force-fed historical assistant content token by token (q = 1), and **P** prefilled complete historical messages; both then forced the identical carrier. Neither schedule is native continuous live-agent generation — a limitation, not a footnote, because schedule turned out to matter (§8).

Three readout regions were defined: R1 (carrier content), R2 (adds the carrier close plus the anchor user turn and assistant-generation header), and R3 (adds the fixed acknowledgment content). **R2 alone was primary.** The N-schedule grid crossed key source × value source over {Fresh, Correct, Wrong}: FF, FC, FW, CF, CC, CW, WF, WC, WW; P used the primary R2 subset. All cells, regions, and schedules are correlated views of **one case** — they add perspectives, not sample size.

The test case, e01, was an engineered fixed transcript authored by an independent Codex subagent. A later additive provenance correction binds the literal author/reviewer sessions to `gpt-5.6-sol`, `xhigh`, via session IDs and rollout hashes. Its sole causal input changes from a 6:40 rollback interval to 10:40; downstream statements were coherently updated to apply the explicit rule, so the two histories also state the resulting `partner beta` versus `staff ring` decision. An unchanged nonfocal probe (correct `21 days`, counter-answer `30 days`) serves as the selectivity control. Because the focal answer was explicitly resolved and repeated before the carrier, this fixture can detect retained resolved or lexical target state; it is not a test of recovering an unstated computation.

**Exact subject and key runtime fields** (complete fingerprint in the raw artifact): `Qwen/Qwen3-30B-A3B-Instruct-2507`, revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, bf16, eager attention, on an accepted A100 80GB PCIe host; technical-run commit `3b3539f6`, Phase-A commit `139d621e`, treatment-execution commit `cbdfa481`; runtime fingerprint `b6158b989b0b467c46589f3ecbe69acba88645d4cd193dd23d925e2e4cd69484`.

---

## 7. The formal stop

### 7.1 A prose/code conflict at the path-control gate

V12's preregistration required a passing path control before any treatment: minimal persisted edits (measured in units of last place, ULP) must demonstrably propagate to the readout. The frozen prose said to stop at the **first** ULP count at which both persisted edits differed measurably from fresh. The sealed code, however, continued until both edits also moved in the **intended signed directions**. The exact cells exposed the divergence:

| ULP | Oriented + movement | Oriented − movement | Sealed-code verdict |
|---:|---:|---:|---|
| 1 | 0.0 | +0.5 | fail |
| 2 | +0.25 | −0.25 | fail |
| 4 | +0.5 | +0.25 | pass |

Under the literal written rule, the procedure stops at ULP 2 — both edits measurably differ from fresh there — and the gate **fails**, because the movements are not both correctly oriented. The ULP 4 "pass" exists only under the code's stronger continuation rule, and therefore survives only as implementation-defined intervention/readout sensitivity, not as a preregistered pass. The conflict was discovered and dispositioned **before treatment**: the literal prose branch governs. **Formal v12 is terminal.** It can produce no four- or six-case aggregate, no confirmation, and no conversation- or agent-level claim.

The status sequence is therefore: (1) design and rule frozen; (2) path-control prose/code conflict observed; (3) literal preregistered branch applied and formal experiment stopped; (4) one unchanged treatment execution separately authorized as non-authorizing diagnostic evidence (§8). The diagnostic does not retroactively reopen the formal arm.

### 7.2 Adverse natural calibration

Independently, the frozen natural calibration was adverse. The green transplant recovered −3.6% and the amber transplant 29.8% of their respective full-oracle margin gaps, both below the preregistered 50% reference; the green transplant slightly worsened its already-green fresh margin, and both transplanted cells still freely generated `approve`. The same-visible-text green-minus-amber transplant contrast nevertheless moved 6.5 margin units in the source-history direction, so this was adverse evidence for large bidirectional answer recovery, not a zero-channel result. A path control or calibration establishes only the behavior it measures; neither establishes semantic validity.

---

## 8. The e01 diagnostic

Exactly one unchanged e01 treatment execution was permitted after the stop, explicitly classified **diagnostic-only**: its receipt's `phase_a_release_status` field was `PRETREATMENT_PASS`, and the receipt records no formal or expansion eligibility. It is one engineered case. Nothing below carries a p-value, an interval, or a population claim.

### 8.1 Results

For state source `X`, let `Y(X) = mean_lp(correct target | X) − mean_lp(wrong target | X)`. Then `D = Y(X_C) − Y(X_W)`; `SEL = D_focal − |D_nonfocal|`; `H+ = mean_lp(correct target | X_C) − mean_lp(correct target | X_W)`; `U = Y(X_C) − Y(FF)`; and `U+ = mean_lp(correct target | X_C) − mean_lp(correct target | FF)`, where `X_C` and `X_W` use correct- and wrong-history source state and `FF` is fully fresh. Positivity of `H+`/`U+` is a criterion, not part of their definition. All values are forced-continuation log-probability movements in nats.

| Schedule / region | Family | D focal | D nonfocal | SEL | H+ | U | U+ |
|---|---|---:|---:|---:|---:|---:|---:|
| N / R1 | full K+V | +0.154 | −0.022 | +0.132 | +0.023 | +0.007 | +0.260 |
| N / R1 | value-only | −0.126 | −0.114 | −0.240 | −0.274 | −0.154 | −0.199 |
| **N / R2** | **full K+V** | **+0.097** | **+0.245** | **−0.148** | **−0.131** | **+0.175** | **+0.025** |
| **N / R2** | **value-only** | **+0.175** | **−0.00007** | **+0.174** | **+0.710** | **+0.182** | **+0.946** |
| N / R3 | full K+V | −0.010 | +0.142 | −0.153 | −0.086 | −0.275 | −0.166 |
| N / R3 | value-only | −0.005 | +0.186 | −0.192 | +0.257 | +0.002 | +0.361 |
| P / R2 | full K+V | +0.039 | −0.001 | +0.038 | −0.446 | −0.297 | −0.334 |
| P / R2 | value-only | +0.021 | +0.018 | +0.003 | +0.229 | −0.073 | +0.280 |

### 8.2 Reading the table honestly

- **Full K+V failed its own criteria.** At the primary N/R2 cell, nonfocal movement (+0.245) exceeded focal movement (+0.097), and the correct target worsened (H+ = −0.131).
- **Value-only N/R2 is the one internally favorable cell**: focal movement +0.175 with essentially zero nonfocal movement (−0.00007), and the correct target improving.
- **The effect is schedule-sensitive.** Value-only D focal collapsed from +0.1745 under N to +0.0210 under P. Worse, at token level under P, the *first* `partner`-versus-`staff` choice token moved in the **wrong** direction (−0.1875); only the conditional second token (+0.2296) made the phrase mean slightly positive. Since neither N nor P is native generation, we cannot say which — if either — reflects live-agent state.
- **The surrounding cells do not corroborate.** R1, R3, key-only, and crossed cells form no coherent pattern around the favorable cell.
- **The recovered fraction is tiny.** Fresh re-encoding of the carrier cost 22.298 nats of focal margin and 22.291 nats of correct-target log-probability relative to the full-history oracle. The best value-only cell recovered 0.81% and 4.24% of that damage, respectively.
- **Semantic attribution is not identified.** The focal decision and target words were explicitly present before the carrier. With no working placebo and strong schedule/region/K-V interaction, the favorable cell is also compatible with deterministic lexical or resolved-state residue whose downstream effect depends on replay and boundary geometry.
- **There was no behavioral recovery.** The result records 31 schedule/region cells but only 29 distinct scored branches because the direct-fresh `FF` branch was reused across N/R1, N/R2, and N/R3. Every distinct branch freely generated the same unrelated/wrong answers (`Ring 3`, `30 days`). R2 alone was primary; R1/R3 were descriptive. No transplant changed what the model actually said.
- **The placebo is missing, not null.** All three planned norm-matched bf16 region controls were unavailable. At their shared first nonzero row — layer 1, fresh-destination physical row 76 — none of deterministic attempts 0–1023 produced an applied-bf16 perturbation satisfying the frozen relative-norm and cosine tolerances. This bounded failure does not prove that no representable perturbation exists; it means zero placebos were available in this run.
- **The placebo cannot be repaired locally.** The run persisted hashes and shapes, exact token/position traces, target IDs and log-probabilities, complete free-generation text and records, scores, and runtime provenance — but not K/V tensor element values. Reconstructing the exact state would require rerunning the model (§9, item 9).

Integrity of the diagnostic itself is well evidenced: treatment-fresh score objects matched Phase A canonically; the 8,897,066-byte raw record reconstructed byte-for-byte to SHA-256 `f6622d978c80d8f4f874c208ef9f6d9546ab3f1bf7676655c9a0b4662718d082`; independent pod and local harvests matched after only allowlisted path/timestamp normalization; and no unexpected post-run difference was found.

### 8.3 The strongest safe sentence

> In one execution, one fixed engineered case produced a favorable focal-over-single-nonfocal forced-logprob contrast in the N/R2 value-only cell under identical visible text.

We classify this as a weak, uncontrolled, schedule-sensitive mechanistic hint. Forward-run repeatability was not tested, and one nonfocal probe does not establish general selectivity. It does not establish semantic specificity, utility, robustness, generality, or behavioral recovery. Schedule-, boundary-, or key-interaction-dependent numerical or lexical residue remains a sufficient alternative explanation.

---

## 9. The methodological failure catalogue

We consider this the paper's most durable contribution. Each entry is a failure that occurred in this project, its consequence, and a scoped guardrail suggested by the experience. The three groups below have different epistemic status; an observed numerical hazard is not the same contribution as an operational incident. Several passed elaborate hash, test, review, and release machinery — the machinery validated the wrong literal assumption.

### 9.1 Measurement hazards observed on the tested stack

| # | Failure | Consequence | Guardrail |
|---|---|---|---|
| 1 | **Query shape/schedule is part of the computed bf16 state.** On local `Qwen/Qwen3-0.6B` (CPU, bf16, eager), ordinary versus coarse replay of the 8,430-token c10 prefix moved a fixed margin by 0.060546875 nats; c02 moved 0.1318359375. In the isolating A/B comparison, the first 23 token IDs and positions were identical but query width changed 23→4096. The first stored K/V divergence appeared at layer 1 (row 22), consistent with an origin in layer-0 attention. A separate equal-shape B/C control changed causally later content while protected rows remained bit-exact, ruling out the tested future-content leakage. | Replay schedule alone can change state and readout. The local magnitude is not a bound for 30B/A100. | Treat schedule as an experimental variable (`QUERY_SHAPE_ROUNDING`); measure treatment and control under bit-identical schedules. The general finite-precision risk is documented by [PyTorch's numerical-accuracy guidance](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html); our observation localizes it in this harness. |
| 4 | **Position correction was numerically consequential in this apparatus.** In the local 0.6B diagnostic, helper-versus-native target-margin error was 0 in fp32, 0.0029296875 in fp16, and 0.1015625 in bf16; corresponding maximum key errors were approximately 3.39×10⁻⁵, 0.03125, and 0.25. | Position handling masquerading as treatment effect. | Preserve native positions where possible. If re-rotation is unavoidable, start from sufficient-precision pre-rotation keys and calibrate against native destination computation; do not label an uncalibrated result exact. |
| 5 | **Toy fixtures certify plumbing, not schedule equivalence or the production numerical regime.** Short/periodic plumbing checks passed, while the first realistic local c10/c02 schedule fixtures failed. | False confidence transferred from nonrepresentative fixtures. | Gate production claims on representative exact-stack checks (model, dtype, length, hardware, schedule), after separate plumbing tests. |

### 9.2 Control, inference, and reproducibility failures

| # | Failure | Consequence | Guardrail |
|---|---|---|---|
| 2 | **Pseudoreplicated, nonrepresentative validation inputs.** An apparent 7/7 zero-difference validation used seven lengths of one five-token periodic stream — seven parameter settings, not seven representative inputs. | A fragile equivalence assumption appeared broadly verified. | Every gate must be able to fail for its intended reason; validate on representative, non-degenerate inputs. |
| 3 | **Mechanical geometry mistaken for semantic validity.** A wrong-history control cycled short donor text up to 51 times; later exact-width counterfactuals preserved tokenizer geometry but failed full decoded review. | "Controls" that were not meaningful alternative histories. | Decode and review every control as text; token-geometry equivalence is necessary, never sufficient. |
| 6 | **A control constructor can fail to yield an available production-dtype control.** For each of three e01 regions, attempts 0–1023 yielded no applied-bf16 perturbation satisfying the frozen norm and cosine tolerances at the shared first nonzero row. | The critical control was absent at analysis time; the bounded failed search does not prove global nonrepresentability. | Demonstrate that the constructor yields qualifying controls in the production dtype before the run; treat "no placebo" as missing evidence. |
| 7 | **Prose/code agreement is itself a preregistration gate.** Hash-frozen code did not resolve ambiguous stopping semantics (§7.1). | A pass/fail verdict that depended on which artifact you believed. | Diff the literal decision rule in prose against the literal branch in code before sealing; ambiguity discovered later must be dispositioned conservatively. |
| 9 | **State needed for later controls was not preserved.** E01 retained rich score, token, trace, generation, and runtime records, but no K/V elements; hashes cannot reconstruct K/V rows. | The missing placebo cannot be repaired without a full rerun. | Budget a bounded tensor bundle for control-relevant geometry — independently estimated here at ~39.84 MiB, versus ~480.94 MiB for five full snapshots. |

### 9.3 Operational and project-sequencing incidents

| # | Incident | Consequence | Guardrail |
|---|---|---|---|
| 8 | **Container identity is not host identity.** The same Secure A100 type and image surfaced NVIDIA drivers 580.159.03 and 550.90.12; pinned CUDA initialized on only one. | Irreproducible runtime; wasted provisioning. It explains no scientific result above. | Admission-gate the actual GPU, driver, and memory before bootstrap. |
| 10 | **Confirmation implementation ran ahead of an observed exact-model signal.** The v11 pipeline and draft-corpus work were built before such a signal; the planned paid twelve-case execution was then paused before running. E01 did not meet the later re-entry standard either. | Sunk implementation and authoring effort created pressure toward unjustified execution. | Preregister and design early, but sequence large implementation and paid collection behind an observed, gated signal; write the re-entry standard before you want it. |

The unifying lesson is not that rigor failed. Hashes were checked, partitions held, releases were sealed — and several of these failures happened anyway, because rigorous checks of *execution* can still validate the wrong *estimand* when the literal fixture, dtype, control, or stopping rule does not test the generalization being claimed. The check you need is the one aimed at the assumption you did not know you were making.

---

## 10. What is and is not established

**Established by this repository (within the stated scope):**

- No detected pooled average lift for the legacy synthetic value-graft apparatus on its exact mixed target corpus (§4.2), alongside descriptive post-hoc source-block estimates with opposite signs, intervals permitting modest effects, and a prefill-reconstruction provenance defect.
- A small, out-of-fitting, selected-map likelihood lead on demonstrated next actions in coding trajectories (§5), control-incomplete and behaviorally unresolved.
- A formal technical stop of the exact-state canary under its literal preregistered rule (§7), before any formally eligible treatment outcome.
- One N=1, diagnostic-only, placebo-missing, schedule-sensitive value-only trace under identical visible text, with no behavioral recovery (§8).
- Ten concrete failure modes with guardrails for activation-state experimentation (§9).

**Not established, in either direction:**

- Whether generation-time state on this model carries recoverable history-specific semantic content at all (the redesign stopped before answering).
- Whether any training-free transplant family can recover a practically meaningful fraction of compaction damage.
- Any effect — positive, negative, or null — on real agent task outcomes.
- Any cross-architecture generalization.
- Whether weight quantization or KV-cache dtype changes the effect.

**Also deliberately absent:** an overall cost figure. The end-to-end money/token audit is still open and will be reported separately, distinguishing cash, subscription usage, provider credits, list-price equivalents, estimates, lower bounds, and unknowns.

---

## 11. Why no final paired live-agent evaluation ran, and what one would require

No valid paired live-agent evaluation of a treatment that cleared the mechanistic ladder was run, and none is authorized. A randomized paired trial of one frozen bundled intervention could still answer a black-box efficacy question — whether that exact system changes task outcomes — without proving its mechanism. It could not attribute any movement to recovered semantic cache content. This project chose mechanism clearance as its launch policy. Given the unstable schedule-sensitive candidate, missing matched control, absence of a task source that had simultaneously cleared capability and compaction-damage gates, and the closed data-collection budget, another agent trial had poor expected information value. That is a project decision, not a causal law.

Earlier exploratory agent episodes explain the caution:

- On clean synthetic chain tasks, full context, ordinary compaction, α = 0.75, and the layer champion each solved 4/4; α = 1 solved 3/4. The compacted baseline's 4/4 ceiling left no damage to repair.
- On the SWE-bench-derived pilot set, tested arms failed and the full-context/oracle arm also failed: a subject/scaffold capability floor, not graft evidence.
- In tau-banking, natural sessions were too short. At the default 12,000-token threshold compaction did not fire; at `compact_at=1500` / `tail=500`, completed sessions were only about 1,600–2,080 tokens and evicted roughly 100–500 tokens, too little substantive early content for a meaningful contrast.

A defensible future **black-box efficacy** evaluation would require at minimum:

1. A separate qualification set, never reused for treatment evaluation, to establish that the subject can solve the task distribution and that ordinary compaction causes measurable damage.
2. Native generation-state capture at an actual compaction boundary, with one deployable treatment frozen before opening the locked evaluation set.
3. Paired forks from identical repository, container, task, seed, and pre-boundary state — standard compaction versus treatment — with full-context runs as reference where feasible.
4. A preregistered sample size, stopping rule, timeout/failure policy, and intent-to-treat analysis; evaluation tasks are not dropped because their individual A/B contrast is inconvenient.
5. Equal tool, token, time, and retry budgets; randomized paired-arm execution order to reduce infrastructure/order bias.
6. Repository-level deterministic tests as the primary endpoint, scored blind to arm when any judgment remains.
7. Full preservation of transcripts, summaries, captured state, patches, tests, resource budgets, and every compaction event.
8. Inference clustered by task/repository, reported as a feasibility result unless powered otherwise.

To support **mechanistic attribution** as well, it would additionally require bit-identical schedules across treatment and controls, a same-visible-text wrong-history placebo, and a delta-matched perturbation control. A task-success difference without those controls would still answer the bundled-system question, but not the semantic-state hypothesis.

The formal re-entry requirements are recorded in `notes/2026071288-sol-data-collection-stop-and-future-reentry.md`. Paid data collection for this project is finished.

---

## 12. Limitations

Beyond the per-stratum caveats above: the principal experiments concern Qwen3, mostly one checkpoint; heterogeneous historical side experiments on other architectures do not establish generalization. The exact redesign and diagnostic were bf16 only. The early 4-bit-weight MLX hint used fp16 KV cache and a now-retired, confounded apparatus, so neither weight-quantization dependence nor KV-cache-dtype dependence has been tested cleanly. A bitsandbytes NF4 arm would change weight representation and kernels while ordinarily leaving KV state floating-point; it would be a useful matched runtime/weight-quantization axis, not an isolated test of “4-bit KV.” It was not run because formal v12 stopped before eligible treatment, its diagnostic lacked placebos, and paid collection had closed; a new precision axis belongs behind the same re-entry gates rather than being attached post hoc to an N=1 diagnostic. Neither v12 replay schedule is native continuous generation, and the one suggestive cell was schedule-sensitive — the construct validity of forced replay for live-agent state is untested. The legacy stratum's source state was reconstructed rather than captured. The strongest positive lead lacks its matched placebo; the diagnostic lacks any placebo. Several provenance elements are irrecoverable: legacy `git_commit: null` manifests, the SWE-Gym upstream revision and row-level trajectory-generator identities, per-trajectory summary text, and all K/V tensor values from e01. Neither the SWE out-of-fitting structural-match endpoint nor e01 free generation moved in the treatment's favor; for the transplant evidence retained here, no task-success endpoint was measured.

---

## Author contributions

- **Jeremy Banks** — the research question, direction, and funding; repeated methodological corrections; final scope and taste decisions. The mitigation-first framing of the project was his intent from the beginning.
- **Anthropic Claude Fable 5** — initial final-paper drafting and synthesis; earlier strategic/diagnostic consultation recorded in the notes archive.
- **OpenAI GPT-5.6 Sol (extra-high reasoning)** — primary final analysis, execution, interpretation, and scientific decision authority, including the facts brief this draft is bound to.

Additional model contributions, attributed only where runtime evidence supports them: **Claude Opus 4.8** (paper-spine and methodological-postmortem analysis), and assistant sessions stored only under the aliases **`sonnet`, `opus`, and `fable`** (authorship of legacy conversation bodies c13–c24; exact provider/runtime/version not preserved there). The e01 transcript and two later review passes are bound by session IDs and rollout hashes to **OpenAI GPT-5.6 Sol, extra-high reasoning**.

## Acknowledgments

**OpenAI GPT-5.5 (extra-high reasoning)** provided independent external review of prior report revisions. Local body generation for c07–c12 used `mlx-community/Qwen3-4B-Instruct-2507-4bit`.

---

## Appendix A: Data and runtime provenance

### A.1 Legacy synthetic stratum

**Token provenance.** There was no single shared system prompt: each conversation's literal system prompt, experiment-authored user turns, planted manipulation, probes, and gold continuation are stored in `data/synthetic/c07.json` through `c24.json`. They were AI-assisted synthetic authorship under human direction, not authentic user conversations and not generated by the 30B subject. The exact authoring model/session IDs and configurations for this experiment-authored text were not preserved; literal text is tracked, but its author-model provenance is unavailable. The files define 206 plants; the pre-score contamination rule excluded `c08-sense-1`, `c10-referent-1`, and `c11-sense-2` because `contaminated_early` was nonempty. The remaining 203 scored plants comprise 35 referent, 34 sense, 36 stance, 36 ruled-out, 36 evicted-fact, and 26 strong-prior targets.

C07–c12 assistant replies were generated in context by `mlx-community/Qwen3-4B-Instruct-2507-4bit` at temperature 0.7 with categorical sampling and no top-p filter, seeds 1006–1011, and a 420-token reply cap; 132 capped replies were counted and trimmed to a prior sentence/paragraph boundary when available. Generation used one growing MLX KV cache. After each reply, the cache was truncated to the canonical prefix and the reply was re-prefilled in canonical non-final form to work around the Qwen chat-template change at a final assistant message. `src/fix_tails.py` then repaired 26 tail replies that leaked plant keywords using the same 4B model: up to six resamples with per-attempt seed `base_seed*977 + message_index*31 + attempt`, then truncation or a canned fallback. Metadata retains per-conversation counts, not the successful attempt for each repaired turn. It also omits the resolved model revision, MLX/MLX-LM versions, generation hardware/runtime, and body-generation code commit, so the tracked literal bodies are inspectable but their generation is not exactly rerunnable. C13–c24 bodies were authored by assistant sessions whose files retain only `sonnet`/`opus`/`fable` aliases. Thus none of the bodies are native 30B-subject generation.

The 30B subject generated its own summary under this exact request:

> Please write a thorough context note summarizing our conversation so far, for someone who will continue this conversation without seeing it. Cover: decisions made and what was chosen over what; open threads and next steps; definitions, names, and terms we introduced and what they mean; constraints and preferences either of us stated; approaches or options we tried and ruled out, and why. Be redundant and specific; use retrieval-friendly wording. Write it as flowing prose or bullet points, roughly 300-500 words. Do not add commentary before or after the note itself.

The conversation plus summary request was prefilled in sequential 4,096-token chunks plus a remainder; summary decoding was greedy one token per forward call (temperature 0), to EOS or a maximum 900 generation steps, seed 17. The helper unconditionally removed the final returned token as though it were EOS; c11, c18, c21, and c24 reached the cap, retained 899 summary tokens, and end mid-item or mid-sentence. The actual generation-mutated snapshot was discarded; the historical source cache was later reconstructed by prefilling the conversation plus request and saved summary tokens. Gold continuations were AI-assisted experiment-author targets derived from each plant, not generated by the subject, shared across arms, and capped at 80 tokens.

**Arms and scoring.** The scored context omitted each conversation file's final stored assistant message (`messages[:-1]`). `A` is that remaining history. `B` begins with the original system message, then one assistant message with the literal wrapper `[Context note] Earlier parts of this conversation were compacted. Summary of what came before:\n\n{summary}`, then the messages from the stored `middle_end_msg` tail boundary through the penultimate stored message. `E` starts from `B` and replaces aligned value rows in the summary and tail regions according to the frozen map; keys remain fresh. Alignment runs `difflib.SequenceMatcher(autojunk=False)` separately within the summary and tail regions, retains only matching blocks of at least eight tokens, and excludes destination positions below four and destination special tokens. A later strict exact-span aligner was tried and reverted; the reported runs use this difflib path. Each plant score first averages mean-token log-probability over its 1, 3, or 4 phrasing probes; `raw_EB = lp_E − lp_B`.

Map selection used a different endpoint and corpus from headline validation. C01–c06 used the same 4B sampled synthetic-body pipeline as above, with seeds 1000–1005. N01–n04 were not authentic natural chats: they were model–model simulations with experiment-authored personas and opening turns, the same 4B checkpoint playing both assistant and a separately prompted user simulator, seeds 2000–2003, 16 user turns, and the final four messages held out. For c01–c06 the selection endpoint was teacher-forced likelihood of the held-out final assistant content. For n01–n04 it was the shifted canonical suffix of the final four messages — two user/assistant exchanges, including markup and user-turn tokens except the suffix's first token. Selection used the actual generation-mutated summary snapshot; headline validation discarded that state and reconstructed the source by prefilling the conversation, summary request, and saved summary text. The maps were therefore transferred across a source-state procedure change as well as to a different corpus and endpoint.

The per-layer map has a reproducible derivation from its committed 48-layer profile: after ranking, the top approximately 21% received α = 1, the next approximately 35% received α = 0.75 only while their marginal remained positive, and the rest received zero. That profile predates the 4,096-token chunking change and used a single prefill call; headline validation used 4,096-token chunks. The evaluated per-head map contains 109 α = 1 slots, but its derivation profile was not preserved. Reapplying the claimed positive-mean rule to the cited committed ten-row head profile selects 121 slots; overlap is 80, with 29 evaluated-only and 41 committed-profile-only slots, and exhaustive search over all 1,023 nonempty row subsets reproduces the evaluated map zero times. The later head-scan job likely re-derived a fresh pod-local profile under chunked prefill because the launcher did not upload `results/`, then retained only validation outputs with embedded maps; this explains the discrepancy but cannot prove exact derivation or held-out separation. Intersection and union do reconstruct exactly from the embedded 109-slot map and 27-layer map: intersection retains the 74 head slots inside selected layers; union has 143 slots after adding every KV head in selected layers. Both use uniform α = 1, so the layer map's 0.75/1.0 values are not preserved. Exact evaluated maps are embedded in the four headline JSONs. The tuning directories also lack born-annotated model/dtype manifests; their 30B-bf16 identity is reconstructed from path and launch records.

For headline validation, the task-competence floor was mean-token `lp_A ≥ −8`; category headroom required mean `lp_A − lp_B ≥ 0.3`. Among the 203 contamination-cleared plants, the task-competence gate excluded zero further plants, zero categories were floored, and zero conversations had empty alignments. Long A/B/source prefills were forwarded sequentially in 4,096-token chunks plus the final remainder; each target continuation was teacher-forced in one batched call. Results are conditional on that query partition. Intervals use 10,000 percentile-bootstrap replicates, seed 42, resampling whole conversations while retaining the plant-weighted estimand.

**Subject/runtime provenance.** The promoted JSONs record subject/scorer ID `Qwen/Qwen3-30B-A3B-Instruct-2507`. Revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, bf16, and A100 runtime are reconstructed from surrounding project records rather than born-annotated in all four headline files; code commit is absent or null. The model has 48 layers, 32 query heads, 4 KV heads (GQA ratio 8), head dimension 128, QK norm, and RoPE theta 10,000,000. The 4B body-generator metadata gaps stated above remain unresolved.

### A.2 SWE-Gym/OpenHands likelihood stratum

**Source and cut.** The local ignored/untracked `swegym.parquet` is 10,379,409 bytes, SHA-256 `ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1`, 491 rows, and contains only `messages`. The source paper describes these as successful rejection-sampled OpenHands trajectories from GPT-4o and Claude 3.5 Sonnet, but success flags, row-level generator IDs, upstream revision, stable task/repository IDs, and subject-native generation are absent from the local file. All rows share one 4,758-character system prompt, now preserved literally in `results/swegym_paper_reanalysis/swegym-shared-system-prompt_provenance_20260712T054055Z.json` (UTF-8 content SHA-256 `1120aa8819abb372428afb82f6a5f49d1d243e4bf58cb27fd481809acd339e84`). Within each stored trajectory, that system instruction and the initial user task come from the SWE-Gym/OpenHands task scaffold; later user-role messages are tool/environment results or automated continuation instructions emitted by that scaffold; assistant-role actions and prose come from the historical rollout model. The demonstrated target is one of those imported assistant turns, not a Qwen-generated or uniquely correct action. Canonical-render length had to be 6,000–15,000 tokens. The harness chose the median eligible assistant boundary lying 60–85% through the rendered trajectory and required the preceding context to end with a user turn. For likelihood scoring, the target was the first 600 tokens of that demonstrated assistant turn; the structural `gold_action` was parsed from its full untruncated content before this truncation. The retained tail began at the message boundary nearest 75% of the pre-target context.

The 30B subject generated its own summary under this exact brief request (SHA-256 `d32f3f90eed98c08bdd04597a651689c011224022a2c20f80cf35d0210887c2e`):

> Please write a very brief context note (3-5 sentences, no lists) giving only the big picture of our conversation so far: what the project is and roughly where we are. Do not include specific decisions, names, numbers, or details. No commentary before or after.

In the later apparatus, the trajectory plus summary request was prefilled in sequential 4,096-token chunks plus a remainder; summary decoding was greedy q = 1 to EOS or a maximum 900 tokens, seed 17. The July 5 fixed-scalar run predates chunking and used a single long prefill call, which is why it is reported separately rather than averaged with the later run. In both paths the source was the actual generation-mutated summary snapshot. `B` used the same literal assistant wrapper as A.1, then appended the tail beginning at the message boundary nearest 75% of the pre-target context. The fixed scalar `E` replaced aligned summary+tail values at α = 0.75. The selected-map arm used α = 1 on layers 12–17 and 30–35. Both used the exact difflib alignment contract in A.1. Eight contiguous six-layer regions were profiled on the tune split; the union of regions with positive tune-split mean (regions 2 and 5) was frozen in `data/champion_configs/swegym_tuned_20260710T145330Z.json`, SHA-256 `6faa2d7227d46c86f0c614bef808d0424d4f124203a9dc7f7558e2b07e994d53`.

The original pool was formed by scanning parquet indices upward from zero and taking the first 75 rows that passed the cut function; its exact IDs are stored in the production manifest. The fresh pool scanned upward from index 214 with a requested cap of 150, but only 98 remaining rows passed; all 98 were used. Split assignment on those 98 used `int(sha256("20260710:<dataset-index>").hexdigest()[:8], 16) % 2`, even for tune and odd for evaluation: 41 tune and 57 evaluation rows. The later original-pool confirmation followed the original eligible-ID order and stopped for budget after a 45-row prefix of the planned 75; it was neither randomized nor complete.

The continuous score is the paired difference in teacher-forced mean token log-probability of the demonstrated next action. In the later apparatus, long A/B/source prefills were forwarded sequentially in 4,096-token chunks plus the final remainder; the July 5 legacy run used a single long call. Each target was scored in one batched teacher-forcing call, so every result is conditional on its stated query partition. For the structural endpoint, each primary arm fed the generation-header suffix in one forward call, then generated greedily q = 1 for at most 200 tokens or until EOS; the parser checks only tool/path/command agreement and does not execute the action or verify edit content. Result rows retain the parsed `action_match` object but not target text/token IDs, and retain only `gen[:500]`: for B, E-tuned, and E-champion, 138/294 fresh-pool fitting-plus-evaluation outputs are exactly 500 stored characters, including 81/171 held-out-evaluation outputs; 54/135 confirmation outputs are also exactly 500 characters. These generations may be truncated. The structural counts are recomputable from saved booleans but cannot be independently re-parsed from complete generations. Paper statistics are recomputed by `scripts/analyze_swegym_paper_metrics.py`: 10,000 trajectory-bootstrap replicates, seed 0, every component sorted numerically by dataset index. The bootstrap treats trajectory rows as independent because stable task/repository cluster IDs were not saved.

**Subject/runtime provenance.** Tune/evaluation manifests record `Qwen/Qwen3-30B-A3B-Instruct-2507`, revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, bf16 with no quantization on A100 80GB PCIe; Linux 6.8.0-111-generic x86_64/glibc 2.35, Python 3.11.10, Torch 2.4.1+cu124, and Transformers 4.57.6. Their `code.git_commit` and `code.git_dirty` fields are both null; NVIDIA driver, CUDA runtime, and resolved attention backend are not recorded. The initial original run lacks equivalent born-annotated provenance. Per-trajectory summary text and token IDs were not preserved.

### A.3 Formal v12 and e01

Model `Qwen/Qwen3-30B-A3B-Instruct-2507`, revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`; bf16; eager attention; 48 layers, 32 query heads, 4 KV heads (GQA ratio 8), head dimension 128, QK norm, RoPE theta 10,000,000. Host: accepted A100 80GB PCIe, NVIDIA driver 580.159.04, Linux 6.8.0-100. Software: Python 3.12.11, Torch 2.12.1+cu130 (CUDA 13.0, cuDNN 92000), Transformers 5.0.0, Accelerate 1.14.0, safetensors 0.8.0, huggingface-hub 1.22.0, and sentencepiece 0.2.1. Deterministic algorithms were false and float32 matmul precision was `highest`. Frozen v12 preregistration SHA-256 `fc02e86d007369121accbee69471383200fa3f566183e2ed96102e2cf7428b17`; technical-run commit `3b3539f6`, Phase-A commit `139d621e`, treatment-execution commit `cbdfa481`; runtime fingerprint `b6158b989b0b467c46589f3ecbe69acba88645d4cd193dd23d925e2e4cd69484`.

The literal e01 case is `data/coherent_canary_v12/revision2/session_d/e01.json`; its authorship and review lineage are bound in the runtime-provenance correction listed in Appendix B. A separate OpenAI GPT-5.6 Sol extra-high session authored every literal correct/wrong-history turn, probe, and target phrase in e01; the 30B subject generated none of them. The common carrier request, carrier content, and anchor were separately experimenter-authored and frozen, then force-fed identically across arms.

Replay schedule N forces historical assistant content token by token (q = 1); P prefills complete historical messages; both force the identical fixed carrier and then causally recompute the retained tail (§6.2). For each probe, its user turn plus assistant-generation header was appended in one canonical structural prefill. Correct and counterfactual target scoring and free generation forked from that same immutable prefix; preceding target tokens were teacher-forced q = 1. R1/R2/R3 used R2 as primary. The N grid is FF/FC/FW/CF/CC/CW/WF/WC/WW (key source first, value source second); P/R2 evaluated CC, WW, FC, and FW. Free generations used q = 1 greedy argmax, a 64-content-token cap, and EOS stopping. Status: formal arm terminal at the path-control stop; e01 execution diagnostic-only (receipt field `phase_a_release_status=PRETREATMENT_PASS`, with no formal or expansion eligibility). Three norm-matched bf16 region controls were planned. At their shared first nonzero row — layer 1, fresh-destination physical row 76 — none of attempts 0–1023 yielded an applied-bf16 perturbation satisfying the frozen norm and cosine tolerances; this bounded search failure is not proof of global nonrepresentability. The run persisted hashes and shapes, exact token/position traces, target IDs and log-probabilities, complete free-generation text and records, scores, and runtime provenance; K/V tensor elements were not persisted.

### A.4 Reproducibility boundary

The legacy and SWE paper statistics are exactly recomputable from committed score rows using the scripts in Appendix B. Exact forward inference is not. Legacy headline artifacts omit a trustworthy code commit; the three head-derived variants also lack their originating fitting profile/config/log, and the 4B body-generation records omit checkpoint revision, runtime, and code commit. SWE tune/evaluation manifests have null code bindings, while the parquet, generated summary texts, target token IDs, and complete free generations are not tracked. The SWE structural counts can be read from stored parser outputs but not independently re-parsed end to end. E01 differs: its 8,897,066-byte raw score record is byte-for-byte reconstructible from the committed package and re-harvests to the paper table. But its K/V tensor values were never saved, so a new placebo or control requires rerunning the model. The ~39.84 MiB control bundle and ~480.94 MiB five-snapshot figure are future-design storage estimates, not existing artifacts.

### A.5 Local schedule diagnostic

`Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`, with tokenizer `Qwen/Qwen3-30B-A3B-Instruct-2507` revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`; 28 layers, 16 query heads, 8 KV heads (GQA ratio 2), head dimension 128, QK norm, RoPE theta 1,000,000; macOS 26.2 arm64, Python 3.12.11, Torch 2.12.1, Transformers 5.0.0, CPU bf16 eager; four intra-op and ten inter-op threads; deterministic algorithms false; float32 matmul precision `highest`; code commit `cfde9bcc13f90261e93d3c5348f2cb75e31e7608`. On the 8,430-token c10 prefix, fixed-margin movement was 0.060546875; c02 moved 0.1318359375. The first stored K/V divergence appeared at layer 1, consistent with an origin in layer-0 attention under query-width change 23→4096.

## Appendix B: Artifact map

| Claim | Primary artifact(s) |
|---|---|
| Legacy literal conversations and token provenance | `data/synthetic/c01.json` through `c24.json`; `data/natural/n01.json` through `n04.json`; `src/compose.py`; `src/compose_natural.py`; summary/B-context definitions in `src/arms_common.py` |
| Legacy per-layer selection | `results/tune_layer_30b_bf16/`; `data/champion_configs/layers_30b_bf16.json`; `src/run_tune_hf.py` |
| Broken legacy head-map lineage | `scripts/analyze_legacy_head_map_lineage.py`; `results/legacy_head_map_lineage/legacy-head-map-lineage_Qwen3-30B-A3B-Instruct-2507_20260712T055444Z.json`; `notes/2026071294-sol-legacy-head-map-lineage-audit.md`; job/transfer mechanism in `scripts/job_headscan_bf16.sh` and `scripts/launch_pod.sh` |
| Legacy pooled headline rows | `results/champion_validate/headscan_30b_bf16_heads_shuffle_pos_20260709T194156Z.json`; `results/champion_validate/headscan_30b_bf16_layers_shuffle_pos_20260709T194156Z.json`; `results/champion_validate/headscan_30b_bf16_intersection_shuffle_pos_20260709T194156Z.json`; `results/champion_validate/headscan_30b_bf16_union_shuffle_pos_20260709T194156Z.json` |
| Legacy source split | `src/analyze_legacy_source_split.py`; `results/legacy_source_stratification/legacy-source-stratification_Qwen3-30B-A3B-Instruct-2507_20260712T052531Z.json`; `notes/2026071293-sol-legacy-source-stratification-reanalysis.md` |
| SWE fitting/evaluation/partial confirmation | `results/swegym_tune_20260710T145330Z_brief/`; `results/swegym_champeval_20260710T145330Z_brief/`; `results/swegym_confirm_20260711T011101Z_brief/`; frozen map `data/champion_configs/swegym_tuned_20260710T145330Z.json` |
| SWE shared system prompt | `results/swegym_paper_reanalysis/swegym-shared-system-prompt_provenance_20260712T054055Z.json` |
| SWE realistic-summary scalar | `results/swegym_30b_bf16_prod/` |
| SWE canonical paper statistics and schedule correction | `scripts/analyze_swegym_paper_metrics.py`; `results/swegym_paper_reanalysis/swegym_paper_metrics_Qwen3-30B-A3B-Instruct-2507_20260712T060222Z.json`; `notes/2026071295-sol-swegym-prefill-schedule-correction.md` |
| V12 design, arms, gates | `COHERENT-STATE-DECISION-CANARY-V12-PREREGISTRATION.md` §§3–17 |
| E01 literal case and review lineage | `data/coherent_canary_v12/revision2/session_d/e01.json`; `results/coherent_canary_validation/coherent_canary_revision4_full_manifest_Qwen3-30B-A3B-Instruct-2507_20260711T205951Z.json`; `results/coherent_canary_reviews/reviews/revision4_blind_singleton_independent_codex_20260711T210013Z.json`; `results/coherent_canary_reviews/reviews/revision4_paired_diversity_independent_codex_20260711T210013Z.json`; `results/coherent_canary_validation/coherent_canary_v12_codex_runtime_provenance_correction_gpt-5.6-sol_20260711T214629Z.json` |
| Accepted exact technical run and natural calibration | `results/coherent_canary_v12_technical/coherent-canary-v12-technical_exact-subject_20260712T024940208896Z.json`; `results/coherent_canary_v12_technical/coherent-canary-v12-technical_exact-subject_20260712T024859Z_receipt.json`; `results/coherent_canary_validation/coherent_canary_v12_technical_validation_exact-subject_20260712T024859Z.json`; `results/coherent_canary_v12_technical/coherent-canary-v12-natural-calibration-interpretation_gpt-5.6-sol_20260712T0210Z.md` |
| Formal stop: prose/code conflict and disposition | `results/coherent_canary_v12_budget/coherent_canary_v12_path_control_spec_conflict_disposition_20260712T0334Z.md` |
| Exact Phase A | `results/coherent_canary_v12_phase_a/coherent-canary-v12-phase-a-e01_exact-subject_20260712T030639073087Z.json`; `results/coherent_canary_v12_phase_a/coherent-canary-v12-phase-a-e01_exact-subject_20260712T030622Z_receipt.json`; `results/coherent_canary_validation/coherent_canary_v12_phase_a_e01_exact-subject_20260712T030622Z.json` |
| E01 treatment lifecycle and eligibility status | `results/coherent_canary_v12_treatment/coherent-canary-v12-treatment-e01-exact-subject-20260712T040158Z_receipt.json`; `results/coherent_canary_v12_budget/coherent_canary_v12_e01_treatment_lifecycle_and_cost_20260712T0419Z.md` |
| Byte-lossless package of persisted 8,897,066-byte e01 raw JSON (no K/V elements) | `results/coherent_canary_v12_treatment_package/coherent-canary-v12-treatment-e01_exact-subject_20260712T040237879370Z/manifest.json` and `chunk-000001.json` |
| E01 diagnostic table and integrity | `results/coherent_canary_v12_harvest/coherent-canary-v12-harvest-e01-exact-subject_20260712T040158Z.json`; `results/coherent_canary_v12_harvest/coherent-canary-v12-harvest-e01-exact-subject-local-replay-20260712T0418Z.json`; `results/coherent_canary_v12_postrun_audit/coherent-canary-v12-postrun-audit-e01-exact-subject_20260712T0418Z.json` |
| E01 interpretation and token decomposition | `notes/2026071287-sol-e01-diagnostic-final-interpretation.md`; `scripts/analyze_coherent_canary_v12_e01_tokens.py`; `results/coherent_canary_v12_analysis/coherent-canary-v12-e01-token-decomposition_Qwen3-30B-A3B-Instruct-2507_20260712T045020Z.json`; `notes/2026071289-sol-e01-token-level-decomposition.md` |
| E01 core runner at treatment execution | `scripts/run_coherent_canary_v12_treatment.py`; `src/coherent_canary_case.py`; `src/coherent_canary_runtime.py`; `src/coherent_canary_technical.py`; `src/coherent_canary_tokens.py`; `src/coherent_canary_controls.py`; treatment-execution commit `cbdfa481` |
| Historical e01 orchestration/packaging/audit snapshot | `scripts/historical/coherent_canary_v12_e01/`; exact per-file SHA-256 manifest `scripts/historical/coherent_canary_v12_e01/MANIFEST.sha256` |
| Local query-shape diagnostic | `results/coherent_state_ladder/coherent_state_ladder_gapped_v10_Qwen3-0.6B_20260711T123246Z__stage_committed_case_schedule_fixtures.json`; `results/c10_schedule_origin/c10_schedule_origin_Qwen3-0.6B_20260711T174844462255Z.json` |
| Key re-rotation numeric diagnostic | `src/diagnose_coherent_numeric.py`; `results/coherent_state_diagnostics/numeric_Qwen3-0.6B_20260711T065027Z.json`; code commit `83bca9d8` |
| Exploratory agent pilots | `results/agent_clean_run/chain-*`; SWE-bench-derived rows in `results/agent_clean_run/`; `DECISIONS.md` entries 2026-07-07 08:15–10:45 |
| Retired judged/H-pack claims | `uv run python scripts/reproduce.py --only sense`; `uv run python scripts/reproduce.py --only honesty`; source verdicts under `results/judge_semantic*/` and `results/phase2_30b_scored.json` |
| Data-collection stop and re-entry requirements | `notes/2026071288-sol-data-collection-stop-and-future-reentry.md` |
| Quantization-axis disposition | `notes/2026071296-sol-quantization-axis-disposition.md` |
| End-to-end accounting requirement | `results/coherent_canary_v12_budget/end_to_end_token_and_cost_audit_requirement_20260712T0307Z.md` |
| Methods/provenance checklist (questions, not answers) | `METHODS-PROVENANCE-REQUIREMENTS.md` |

Raw e01 record: 8,897,066 bytes, SHA-256 `f6622d978c80d8f4f874c208ef9f6d9546ab3f1bf7676655c9a0b4662718d082`. The receipt's original raw path is no longer present; reconstruct and verify it from the tracked package, then reproduce the harvest and token decomposition:

```bash
uv run python scripts/historical/coherent_canary_v12_e01/treatment_packaging/artifact_packager.py verify \
  results/coherent_canary_v12_treatment_package/coherent-canary-v12-treatment-e01_exact-subject_20260712T040237879370Z \
  --output /tmp/e01-treatment.json
uv run python scripts/harvest_coherent_canary_v12.py \
  --treatment /tmp/e01-treatment.json --repo . --output /tmp/e01-harvest.json
uv run python scripts/analyze_coherent_canary_v12_e01_tokens.py \
  --package-dir results/coherent_canary_v12_treatment_package/coherent-canary-v12-treatment-e01_exact-subject_20260712T040237879370Z \
  --output /tmp/e01-token-decomposition.json
```

Zero-GPU paper-table recomputation:

```bash
uv run python src/analyze_legacy_source_split.py --output /tmp/legacy-source-split.json
uv run python scripts/analyze_swegym_paper_metrics.py \
  --timestamp 20260712T000000Z --output /tmp/swegym-paper-metrics.json
uv run pytest -q tests/test_analyze_legacy_source_split.py \
  tests/test_analyze_swegym_paper_metrics.py \
  tests/test_analyze_coherent_canary_v12_e01_tokens.py
```

## References

1. MEMENTO: Teaching LLMs to Manage Their Own Context. [arXiv:2604.09852](https://arxiv.org/abs/2604.09852).
2. Models Take Notes at Prefill: KV Cache Can Be Editable and Composable. [arXiv:2606.17107](https://arxiv.org/abs/2606.17107).
3. CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. [arXiv:2405.16444](https://arxiv.org/abs/2405.16444).
4. Learning to Compress Prompts with Gist Tokens. [arXiv:2304.08467](https://arxiv.org/abs/2304.08467).
5. Cache-to-Cache: Direct Semantic Communication Between Large Language Models. [arXiv:2510.03215](https://arxiv.org/abs/2510.03215).
6. Training Software Engineering Agents and Verifiers with SWE-Gym. [arXiv:2412.21139](https://arxiv.org/abs/2412.21139).
7. PyTorch numerical accuracy documentation. [pytorch.org](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
