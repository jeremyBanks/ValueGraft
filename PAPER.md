# Transplanting KV-Cache State Across Conversation Compaction: Inconclusive Evidence and a Methodological Failure Catalogue

**Jeremy Banks · Anthropic Claude Fable 5 · OpenAI GPT-5.6 Sol (extra-high reasoning)**

*Working-paper status: under factual, methodological, and readability review. The published repository landing page remains unchanged until every review gate passes.*

---

## Abstract

When a long LLM conversation is compacted — replaced by a text summary so that work can continue in a fresh context — everything the model computed while generating the original conversation is discarded along with the text. Recent work provides evidence, in trained settings, that generation-time key/value (KV) state can carry task-relevant information that re-encoding the visible text does not recover. This project asked a narrower, practical question: can a **training-free, post-hoc transplant** of old KV state (in particular, old value vectors under fresh keys) into a compacted context reliably mitigate compaction damage on an ordinary instruction model?

The answer we can support is: **the project did not establish practical benefit, and the clean literal proposal remains unresolved.** Across four evidence strata that must not be pooled, trustworthy performance evidence was mostly null, harmful, or control-incomplete. The strongest surviving performance lead is a small out-of-fitting likelihood effect on a coding-trajectory proxy (+0.0135 nats/token, nominal 95% bootstrap CI [+0.0083, +0.0190]) with no matched placebo, no executed action, and no task-success signal. The final exact-state mechanism experiment formally stopped at a preregistered technical gate; a single permitted post-stop diagnostic produced a weak, schedule-sensitive, placebo-uncontrolled value-only trace with no behavioral recovery. A clean natural-compaction test using actual generation-state from the ordinary instruction model was never completed. We report these measurements and limitations, and what we believe is the project's most durable contribution: a catalogue of ten methodological failures. Some moved numerical readouts at the scale of the hoped-for effect; others invalidated controls, provenance, or repairability. Several survived elaborate hash, test, and release machinery before being caught.

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

- **Subject/scorer:** `Qwen/Qwen3-30B-A3B-Instruct-2507`, resolved revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, Torch bf16 on A100.
- **Conversation bodies:** c07–c12 were generated by `mlx-community/Qwen3-4B-Instruct-2507-4bit` (temperature 0.7, seeds 1006–1011, with truncation/tail repairs recorded in metadata). c13–c24 were externally authored by assistant sessions labeled Claude Sonnet, Claude Opus, and Claude Fable; exact runtime receipts are not uniformly exposed. The 30B subject did **not** generate these bodies. System prompts, user turns, planted facts, probes, and gold answers were AI-assisted synthetic authorship under human direction — not hand-authored, and not produced by the experimental subject.
- **Summaries:** generated by the 30B subject itself.
- **Source state:** the harness discarded the actual incremental generation snapshot and *reconstructed* old state by prefill — a provenance defect the redesign later fixed.
- **Intervention:** the value graft covered retained-tail **plus** summary regions, not summary only; alignment was difflib positional-within-region.
- **Known provenance gaps:** headline manifests often record `git_commit: null`.

### 4.2 Held-out average: no detected lift

Four selected graft variants, evaluated on 18 held-out conversations containing 203 planted targets, gave these graft-minus-compacted estimates (nats/token; nominal conversation-clustered 95% intervals). Each plant averages 1, 3, or 4 phrasing-level probe scores before entering the plant-weighted aggregate; the artifacts contain 531 phrasing-level scores in total.

| Variant | Mean | 95% CI |
|---|---:|---:|
| Per-head | +0.017 | [−0.027, +0.062] |
| Per-layer | −0.003 | [−0.041, +0.043] |
| Intersection | +0.008 | [−0.032, +0.052] |
| Union | −0.012 | [−0.053, +0.034] |

This supports exactly one sentence: *no detected average lift in the apparatus that ran.* The upper bounds still permit modest positive effects; nothing here is a definitive null, a refutation, or an equivalence result — and because the source state was prefill-reconstructed, this is not a clean test of generation-time state either.

That pooled result is strongly heterogeneous by conversation-body source:

| Variant | c07–c12: 4B-rendered (6 conversations / 57 plants) | c13–c24: Claude-authored (12 conversations / 146 plants) |
|---|---:|---:|
| Per-head | +0.071 [−0.031, +0.170] | −0.004 [−0.044, +0.038] |
| Per-layer | **+0.120 [+0.055, +0.181]** | **−0.050 [−0.076, −0.028]** |
| Intersection | **+0.114 [+0.035, +0.175]** | **−0.033 [−0.067, −0.004]** |
| Union | +0.064 [−0.027, +0.152] | **−0.041 [−0.080, −0.003]** |

This post-hoc split is descriptive, not a causal source-family comparison. The two blocks also differ in length, authoring process, plant construction, and similarity to the Qwen-rendered tuning corpus, and only six clusters support the positive block. It shows that the pooled null is not homogeneous: the effect may depend on source family, length, construction, tuning-distribution similarity, or some combination. It neither proves model nativeness nor licenses a universal null.

### 4.3 Placebos, read narrowly

The correct old-state graft often outperformed position-shuffled and Gaussian source controls. The safe interpretation is only that aligned source values were **non-exchangeable** — often less disruptive than grossly mismatched full vectors. These controls did not match the actual treatment delta (V_old − V_fresh): Gaussian rows matched source energy rather than the delta, shuffles were not proven strict derangements, and correct old rows may simply have been geometrically closer to fresh rows. We therefore do not use these placebos as evidence of content-specific semantic information or of "a real mechanism."

### 4.4 Retired legacy claims

Several earlier headline claims from this stratum are void or retired, and we list them so they are not re-cited:

- **The four-level compression treatment conclusion is void.** Ultra/brief/medium cells generated summary text under level-specific requests but reconstructed old state under the default realistic request; only the realistic cell was internally consistent. Compression ratios remain descriptions of text. No flat-across-30× or severity conclusion is licensed.
- **The +10–12 percentage-point judged-recovery headline is retired.** It came from a local 4-bit MLX stack, brief/adversarial summaries, conversations c01–c12, and was render-fragile: a clean rerender moved judged sense from +8.7 to +1.0 points and referent from +9.7 to +5.2, with intervals spanning zero.
- **The packed "H-pack" construction is not evidence for value-only recovery.** It changed fabrication/admission behavior but simultaneously altered layout, transformed K, and V; it restored 0/24 evicted facts, and the earlier "38/48 accurate" claim was unreproducible.
- **Cross-architecture rows support no law.** Heterogeneous gates and provenance do not support a QK-norm, dense/MoE, keys-neutral, or shared-direction generalization.

---

## 5. The coding-domain likelihood proxy (SWE-Gym/OpenHands)

The strongest surviving performance lead in the repository is a **likelihood proxy on historical coding-agent trajectories**, and we are careful to describe it as exactly that.

**Setup.** The source paper describes the released 491 SWE-Gym/OpenHands trajectories as successful, rejection-sampled rollouts from `gpt-4o-2024-08-06` and `claude-3-5-sonnet-20241022`; the committed local parquet preserves only a `messages` column, so row-level generator attribution and the immutable upstream revision are unavailable. The 30B subject did not generate the trajectory bodies. It generated a brief compaction summary; unlike the legacy stratum, the source state used by the graft was the *actual generation-mutated snapshot*. The intervention combined summary and retained-tail values. A layer map (α = 1 on layers 12–17 and 30–35) was selected on 41 fresh-pool trajectories, evaluated on 57 disjoint fresh trajectories, and partially confirmed on 45 of a planned 75 original-pool trajectories. The metric is teacher-forced mean token log-probability of the *historically demonstrated next action*.

**The naive fixed scalar result.** The same original 75 trajectories were rendered twice; averaging the two measurements within trajectory gives +0.01449 nats/token, nominal 95% bootstrap CI [+0.0041, +0.0257]. A disjoint 98-trajectory index pool gives −0.00169 [−0.0135, +0.0092]. Across 173 unique trajectories, counting each trajectory once, the estimate is +0.00532 [−0.00274, +0.01313]. The new-minus-original pool difference is −0.01618 [−0.03205, −0.00107]. Thus the result closest to the original naive scalar proposal did not generalize across pools; source, task, index, or other composition differences are not identifiable from the preserved rows.

**Selected-map results (map minus compacted baseline, nats/token):**

- Fresh evaluation (n = 57): **+0.0117**, 95% bootstrap CI [+0.0063, +0.0172].
- Original-pool partial confirmation (n = 45): **+0.0158**, interval above zero.
- Descriptive pool of all 102 rows not used to fit this map: **+0.0135** [+0.0083, +0.0190], 77/102 positive.

The 102 rows are out-of-fitting *for this layer map*, but they are not a pristine preregistered external sample: the research program had inspected the original pool in earlier scalar runs, confirmation stopped at a budget-capped prefix of 45/75, and many analyses preceded this one. Intervals in this section are therefore nominal and pool-conditional, not multiplicity-adjusted confirmation intervals.

**Why this is not an agent result.** Every one of the following caveats binds:

- The target is one historically demonstrated next action, not necessarily the unique correct action.
- No action was applied, no patch tested, no repository task solved.
- The selected map had **no matched placebo** restricted to its selected layers, so map-specific content cannot be separated from map-specific disturbance.
- Against a fixed graft on the fresh 57, the selected map's advantage was inconclusive (+0.0041, CI spanning zero).
- A coarse structural match on tool/path/command was 53% (selected) versus 54% (baseline) on the fresh set; across all 102 out-of-fitting rows the graft fixed three baseline misses and broke three baseline successes — behaviorally a wash at this resolution.
- Confirmation stopped at 45/75 trajectories.
- Bootstrap treated trajectories as independent; a repository/task-cluster sensitivity analysis was not completed.
- A fixed scalar under realistic summaries was null, and the selected map was never tested there.
- Dataset provenance is partially unpreserved: the local `swegym.parquet` hashes `ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1`, but its immutable upstream revision, row-level generator attribution, and per-trajectory generated summary text/hashes were not retained.

Several alternatives fit the selected-map likelihood movement. Layer selection may change generic confidence or calibration; the combined tail+summary intervention does not identify where the signal lives; a teacher demonstration measures imitation likelihood rather than correctness; pool composition may explain the heterogeneous scalar result; and without a map-matched delta control, content-specific recovery cannot be separated from layer-specific perturbation. The structural-match wash supplies no corroborating action-level movement.

The licensed sentence is: *a selected-map, demonstrated-next-action likelihood lead of about +0.013 nats/token on rows not used to fit the map, without a matched placebo, executed action, or task-success endpoint.* It is a reason the question stays open; it is not coding-agent improvement.

---

## 6. The coherent-state redesign

### 6.1 v10/v11: methodological evidence only

The coherent-state v10/v11 effort — a preregistered assay with local technical fixtures and an authored corpus pipeline — produced failed controls and a paused draft corpus, and contributes methodological evidence only (§9). The planned twelve-case v11 corpus build was deliberately paused because no exact-model effect had yet justified confirmatory machinery. There is no semantic treatment result and no sample from this stratum.

### 6.2 v12: the same-visible-text design

The final canary, v12, asked the cleanest version of the question. Two histories — one **correct**, one **minimally counterfactual** — each produce KV state for the *same fixed neutral carrier text*, and a **fresh** arm re-encodes that same carrier text in a position-preserving gapped destination. Because the visible text downstream of the histories is identical across arms, any downstream difference must come through state. The primary semantic contrast is correct-history versus wrong-history state (D); correct-history versus fresh is a utility contrast (U). Full K+V and value-only transplants are separate families, and the intervention operates at matched positions with no re-rotation of quantized keys (a hard-won requirement; see §9, item 4).

The carrier was not a naturally generated summary. The exact request was `Write the fixed neutral handoff note for the next assistant. Output only that note.` The forced assistant note was:

> The prior discussion established the operating context and relevant decision criteria. Continue from this handoff, preserve the existing constraints, and answer later questions from the state available here. No unresolved action is introduced by this note.

The exact anchor user turn was `Acknowledge receipt of this handoff without adding or repeating any factual detail.`, followed by fixed assistant content `Acknowledged.` Every arm then processed a byte-identical retained-tail exchange, causally recomputed after the R1/R2/R3 transplant boundary, before the probes. The fresh destination therefore contained the original system message, carrier request and assistant note, anchor user and assistant turns, and retained-tail messages. Logical positions matched the source while physical storage remained contiguous. Two replay schedules produced the historical source state: **N** force-fed historical assistant content token by token (q = 1), and **P** prefilled complete historical messages; both then forced the identical carrier. Neither schedule is native continuous live-agent generation — a limitation, not a footnote, because schedule turned out to matter (§8).

Three readout regions were defined: R1 (carrier content), R2 (adds the carrier close plus the anchor user turn and assistant-generation header), and R3 (adds the fixed acknowledgment content). **R2 alone was primary.** The N-schedule grid crossed key source × value source over {Fresh, Correct, Wrong}: FF, FC, FW, CF, CC, CW, WF, WC, WW; P used the primary R2 subset. All cells, regions, and schedules are correlated views of **one case** — they add perspectives, not sample size.

The test case, e01, was an engineered fixed transcript authored by an independent Codex subagent. A later additive provenance correction binds the literal author/reviewer sessions to `gpt-5.6-sol`, `xhigh`, via session IDs and rollout hashes. Its sole causal input changes from a 6:40 rollback interval to 10:40; downstream statements were coherently updated to apply the explicit rule, so the two histories also state the resulting `partner beta` versus `staff ring` decision. An unchanged nonfocal probe (correct `21 days`, counter-answer `30 days`) serves as the selectivity control. Because the focal answer was explicitly resolved and repeated before the carrier, this fixture can detect retained resolved or lexical target state; it is not a test of recovering an unstated computation.

**Exact subject and runtime** (full stack in Appendix A): `Qwen/Qwen3-30B-A3B-Instruct-2507`, revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, bf16, eager attention, on an accepted A100 80GB PCIe host; scientific commit `cbdfa481fe08de62bf8178d09ca040716d610f21`; runtime fingerprint `b6158b989b0b467c46589f3ecbe69acba88645d4cd193dd23d925e2e4cd69484`.

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

Exactly one unchanged e01 treatment execution was permitted after the stop, explicitly classified **diagnostic-only**: its Phase A integrity check was `PRETREATMENT_PASS`, and its receipt records no formal or expansion eligibility. It is one engineered case. Nothing below carries a p-value, an interval, or a population claim.

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
- **There was no behavioral recovery.** All 31 executed diagnostic cells, whatever their state source, freely generated the same unrelated/wrong answers (`Ring 3`, `30 days`). R2 alone was primary; R1/R3 were descriptive. No transplant changed what the model actually said.
- **The placebo is missing, not null.** All three planned norm-matched bf16 placebos were unavailable at the same first nonzero early row after 1,024 construction attempts each — the constructions were mathematically valid but not bf16-representable. Zero available placebos is missing evidence, not a passed control.
- **The placebo cannot be repaired locally.** Only hashes, shapes, traces, and scores were persisted — not K/V tensor element values — so reconstructing the exact state would require rerunning the model (§9, item 9).

Integrity of the diagnostic itself is well evidenced: treatment-fresh score objects matched Phase A canonically; the 8,897,066-byte raw record reconstructed byte-for-byte to SHA-256 `f6622d978c80d8f4f874c208ef9f6d9546ab3f1bf7676655c9a0b4662718d082`; independent pod and local harvests matched after only allowlisted path/timestamp normalization; and no unexpected post-run difference was found.

### 8.3 The strongest safe sentence

> One fixed engineered case exhibited deterministic, focal-selective forced-logprob movement in the exact N/R2 value-only cell under identical visible text.

We classify this as a weak, uncontrolled, schedule-sensitive mechanistic hint. It does not establish semantic specificity, utility, robustness, generality, or behavioral recovery. Deterministic schedule-, boundary-, or key-interaction-dependent numerical or lexical residue remains a sufficient alternative explanation.

---

## 9. The methodological failure catalogue

We consider this the paper's most durable contribution. Each entry is a failure that occurred in this project, the consequence it had or would have had, and the guardrail we now consider mandatory for activation-state experiments. They are not all the same kind of contribution: items 1, 4, and 5 are observed numerical/execution hazards; items 2, 3, 6, 7, and 9 invalidate controls, inference, or repairability; items 8 and 10 are operational/project-sequencing incidents. Several passed elaborate hash, test, review, and release machinery — the machinery validated the wrong literal assumption.

| # | Failure | Consequence | Guardrail |
|---|---|---|---|
| 1 | **Query shape/schedule is part of the computed bf16 state.** On local `Qwen/Qwen3-0.6B` (CPU, bf16, eager), ordinary versus coarse replay of the 8,430-token c10 prefix moved a fixed margin by 0.060546875 nats; c02 moved 0.1318359375. In the isolating A/B comparison, the first 23 token IDs and positions were identical but query width changed 23→4096, and first divergence localized to layer-0 attention. A separate equal-shape B/C control changed causally later content while protected rows remained bit-exact, ruling out the tested future-content leakage. | Replay schedule alone can change state and readout. The local magnitude is not a bound for 30B/A100. | Treat schedule as an experimental variable (`QUERY_SHAPE_ROUNDING`); measure treatment and control under bit-identical schedules. The general finite-precision risk is documented by [PyTorch's numerical-accuracy guidance](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html); our observation localizes it in this harness. |
| 2 | **Pseudoreplicated, nonrepresentative validation inputs.** An apparent 7/7 zero-difference validation used seven lengths of one five-token periodic stream — seven parameter settings, not seven representative inputs. | A fragile equivalence assumption appeared broadly verified. | Every gate must be able to fail for its intended reason; validate on representative, non-degenerate inputs. |
| 3 | **Mechanical geometry mistaken for semantic validity.** A wrong-history control cycled short donor text up to 51 times; later exact-width counterfactuals preserved tokenizer geometry but failed full decoded review. | "Controls" that were not meaningful alternative histories. | Decode and review every control as text; token-geometry equivalence is necessary, never sufficient. |
| 4 | **Position correction is numerically consequential.** bf16 re-rotation of already-quantized keys shifted target margins at the effect scale. | Position handling masquerading as treatment effect. | Preserve native positions where possible. If re-rotation is unavoidable, start from sufficient-precision pre-rotation keys and calibrate against native destination computation; do not label an uncalibrated result exact. |
| 5 | **Toy fixtures certify plumbing, not schedule equivalence or the production numerical regime.** Short/periodic plumbing checks passed, while the first realistic local c10/c02 schedule fixtures failed. | False confidence transferred from nonrepresentative fixtures. | Gate production claims on representative exact-stack checks (model, dtype, length, hardware, schedule), after separate plumbing tests. |
| 6 | **Controls can be valid in mathematics and unavailable after casting.** The norm-matched e01 placebo construction failed bf16 representability (0/3 after 1,024 attempts each). | The critical control silently absent at analysis time. | Prove control representability in the production dtype before the run; treat "no placebo" as missing evidence. |
| 7 | **Prose/code agreement is itself a preregistration gate.** Hash-frozen code did not resolve ambiguous stopping semantics (§7.1). | A pass/fail verdict that depended on which artifact you believed. | Diff the literal decision rule in prose against the literal branch in code before sealing; ambiguity discovered later must be dispositioned conservatively. |
| 8 | **Container identity is not host identity.** The same Secure A100 type and image surfaced NVIDIA drivers 580.159.03 and 550.90.12; pinned CUDA initialized on only one. | Irreproducible runtime; wasted provisioning. | Admission-gate the actual GPU, driver, and memory before bootstrap. (An operations/reproducibility finding; it explains no scientific error above.) |
| 9 | **State needed for later controls was not preserved.** E01 persisted hashes/shapes/traces/scores; hashes cannot reconstruct K/V rows. | The missing placebo cannot be repaired without a full rerun. | Budget a bounded tensor bundle for control-relevant geometry — independently estimated here at ~39.84 MiB, versus ~480.94 MiB for five full snapshots. |
| 10 | **Paid collection and implementation scale ran ahead of an observed signal.** The v11 twelve-case corpus build was paused because no exact-model effect justified executing it; e01 did not meet the re-entry standard either. | Sunk cost pressure toward running an unjustified confirmation. | Preregister and design early, but sequence paid collection and large implementation behind an observed, gated signal; write the re-entry standard before you want it. |

The unifying lesson is not that rigor failed. Hashes were checked, partitions held, releases were sealed — and several of these failures happened anyway, because rigorous checks of *execution* can still validate the wrong *estimand* when the literal fixture, dtype, control, or stopping rule does not test the generalization being claimed. The check you need is the one aimed at the assumption you did not know you were making.

---

## 10. What is and is not established

**Established by this repository (within the stated scope):**

- No detected pooled average lift for the legacy synthetic value-graft apparatus on its exact mixed target corpus (§4.2), with source-stratified sign reversal, intervals permitting modest effects, and a prefill-reconstruction provenance defect.
- A small, out-of-fitting, selected-map likelihood lead on demonstrated next actions in coding trajectories (§5), control-incomplete and behaviorally unresolved.
- A formal technical stop of the exact-state canary under its literal preregistered rule (§7), before any formally eligible treatment outcome.
- One N=1, diagnostic-only, placebo-missing, schedule-sensitive value-only trace under identical visible text, with no behavioral recovery (§8).
- Ten concrete failure modes with guardrails for activation-state experimentation (§9).

**Not established, in either direction:**

- Whether generation-time state on this model carries recoverable history-specific semantic content at all (the redesign stopped before answering).
- Whether any training-free transplant family can recover a practically meaningful fraction of compaction damage.
- Any effect — positive, negative, or null — on real agent task outcomes.
- Any cross-architecture generalization.

**Also deliberately absent:** an overall cost figure. The end-to-end money/token audit is still open and will be reported separately, distinguishing cash, subscription usage, provider credits, list-price equivalents, estimates, lower bounds, and unknowns.

---

## 11. Why no final paired live-agent evaluation ran, and what one would require

No valid paired live-agent evaluation of a treatment that cleared the mechanistic ladder was run, and none is authorized. A randomized paired trial of one frozen bundled intervention could still answer a black-box efficacy question — whether that exact system changes task outcomes — without proving its mechanism. It could not attribute any movement to recovered semantic cache content. This project chose mechanism clearance as its launch policy. Given the unstable schedule-sensitive candidate, missing matched control, absence of a task source that had simultaneously cleared capability and compaction-damage gates, and the closed data-collection budget, another agent trial had poor expected information value. That is a project decision, not a causal law.

Earlier exploratory agent episodes explain the caution:

- On clean synthetic chain tasks, full context, ordinary compaction, α = 0.75, and the layer champion each solved 4/4; α = 1 solved 3/4. The compacted baseline's 4/4 ceiling left no damage to repair.
- On the SWE-bench-derived pilot set, tested arms failed and the full-context/oracle arm also failed: a subject/scaffold capability floor, not graft evidence.
- In tau-banking, natural sessions were too short. High thresholds did not compact; low thresholds evicted too little or entered an amputation regime. No meaningful compaction-boundary treatment contrast resulted.

A defensible future **black-box efficacy** evaluation would require at minimum:

1. A no-treatment canary first: full-context versus ordinary-compaction capability/damage measurement, on tasks the subject model can actually solve and where compaction genuinely fires.
2. One treatment frozen before any agent outcome is seen.
3. Paired forks from identical repository/container/task/seed state — standard compaction versus treatment — with full-context runs as reference where feasible.
4. Repository-level deterministic tests as the primary endpoint.
5. Full preservation of transcripts, summaries, patches, tests, tool/token budgets, and every compaction event.
6. Inference clustered by task/repository, reported as a feasibility result, not a powered product claim.

To support **mechanistic attribution** as well, it would additionally require native generation-state capture, bit-identical schedules across treatment and controls, a same-visible-text wrong-history placebo, and a delta-matched perturbation control. A task-success difference without those controls would still answer the bundled-system question, but not the semantic-state hypothesis.

The formal re-entry requirements are recorded in `notes/2026071288-sol-data-collection-stop-and-future-reentry.md`. Paid data collection for this project is finished.

---

## 12. Limitations

Beyond the per-stratum caveats above: the principal experiments concern Qwen3, mostly one checkpoint; heterogeneous historical side experiments on other architectures do not establish generalization. Neither v12 replay schedule is native continuous generation, and the one suggestive cell was schedule-sensitive — the construct validity of forced replay for live-agent state is untested. The legacy stratum's source state was reconstructed rather than captured. The strongest positive lead lacks its matched placebo; the diagnostic lacks any placebo. Several provenance elements are irrecoverable: legacy `git_commit: null` manifests, the SWE-Gym upstream revision and trajectory-generator identities, per-trajectory summary text, and all K/V tensor values from e01. Neither the SWE out-of-fitting structural-match endpoint nor e01 free generation moved in the treatment's favor; for the transplant evidence retained here, no task-success endpoint was measured.

---

## Author contributions

- **Jeremy Banks** — the research question, direction, and funding; repeated methodological corrections; final scope and taste decisions. The mitigation-first framing of the project was his intent from the beginning.
- **Anthropic Claude Fable 5** — initial final-paper drafting and synthesis; earlier strategic/diagnostic consultation recorded in the notes archive.
- **OpenAI GPT-5.6 Sol (extra-high reasoning)** — primary final analysis, execution, interpretation, and scientific decision authority, including the facts brief this draft is bound to.

Additional model contributions, attributed only where runtime evidence supports them: **Claude Opus 4.8** (paper-spine and methodological-postmortem analysis), and assistant sessions labeled **Claude Sonnet, Claude Opus, and Claude Fable** (authorship of legacy conversation bodies c13–c24; exact runtime receipts not uniformly exposed). The e01 transcript was authored by an independent Codex subagent whose exact runtime model and effort were not exposed.

## Acknowledgments

**OpenAI GPT-5.5 (extra-high reasoning)** provided independent external review of prior report revisions. Local body generation for c07–c12 used `mlx-community/Qwen3-4B-Instruct-2507-4bit`.

---

## Appendix A: Data and runtime provenance

**Formal v12 / e01 subject and stack.** Model `Qwen/Qwen3-30B-A3B-Instruct-2507`, revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`; bf16; eager attention; 48 layers, 32 query heads, 4 KV heads, head dimension 128, RoPE theta 10,000,000. Host: accepted A100 80GB PCIe, NVIDIA driver 580.159.04, Linux 6.8.0-100. Software: Python 3.12.11, Torch 2.12.1+cu130, Transformers 5.0.0. Frozen v12 preregistration SHA-256 `fc02e86d007369121accbee69471383200fa3f566183e2ed96102e2cf7428b17`; scientific commit `cbdfa481fe08de62bf8178d09ca040716d610f21`; runtime fingerprint `b6158b989b0b467c46589f3ecbe69acba88645d4cd193dd23d925e2e4cd69484`. Replay: N forces historical assistant content token by token (q = 1); P prefills complete historical messages; both force the identical fixed carrier thereafter. Readout: forced-continuation log-probabilities over R1/R2/R3, R2 primary; N grid FF/FC/FW/CF/CC/CW/WF/WC/WW (key source first, value source second); P on the primary R2 subset. Estimands and sign conventions as defined in §8.1. Status: formal arm terminal at the path-control stop; e01 execution diagnostic-only (`PRETREATMENT_PASS`, no formal or expansion eligibility). Controls: three norm-matched bf16 placebos planned, zero available (representability failure, 1,024 attempts each). Persistence: hashes/shapes/traces/scores only; no K/V tensor values.

**Legacy synthetic stratum.** Subject/scorer as above (same model and revision), Torch bf16 on A100. Bodies: c07–c12 by `mlx-community/Qwen3-4B-Instruct-2507-4bit` (T = 0.7, seeds 1006–1011, repairs in metadata); c13–c24 authored by assistant sessions labeled Sonnet/Opus/Fable. Summaries: 30B self-generated. Source state: prefill-reconstructed (defect). Intervention: value graft over retained tail + summary; difflib positional-within-region alignment. Held-out sample: 18 conversations / 203 plants; conversation-clustered bootstrap intervals. Known gaps: `git_commit: null` in headline manifests; body-author runtime receipts not uniformly exposed.

**SWE-Gym/OpenHands stratum.** Historical successful trajectories; 30B self-generated brief summaries; actual generation-mutated source snapshots; combined summary+tail value graft; selected map α = 1, layers 12–17 and 30–35. Fitting n = 41 (fresh pool); evaluation n = 57 (disjoint fresh); partial confirmation n = 45 of 75 (original pool). Metric: teacher-forced mean token log-probability of the demonstrated next action; bootstrap over trajectories treated as independent (cluster sensitivity analysis not completed). Local `swegym.parquet` SHA-256 `ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1`; upstream revision, trajectory-generator identities, and per-trajectory summary text/hashes not preserved.

**Local diagnostic (failure catalogue item 1).** `Qwen/Qwen3-0.6B`, CPU, bf16, eager; 8,430-token c10 prefix; fixed-margin movements 0.060546875 (c10) and 0.1318359375 (c02); divergence localized to layer-0 attention under query-width change 23→4096.

## Appendix B: Artifact map

| Claim | Primary artifact(s) |
|---|---|
| Formal stop: prose/code conflict and disposition | `results/coherent_canary_v12_budget/coherent_canary_v12_path_control_spec_conflict_disposition_20260712T0334Z.md` |
| E01 treatment lifecycle and eligibility status | `results/coherent_canary_v12_budget/coherent_canary_v12_e01_treatment_lifecycle_and_cost_20260712T0419Z.md` |
| E01 diagnostic scores (table in §8.1) | `results/coherent_canary_v12_harvest/coherent-canary-v12-harvest-e01-exact-subject-20260712T040158Z.json` |
| E01 post-run integrity audit | `results/coherent_canary_v12_postrun_audit/coherent-canary-v12-postrun-audit-e01-exact-subject-20260712T0418Z.json` |
| E01 interpretation and token-level decomposition | `notes/2026071287-sol-e01-diagnostic-final-interpretation.md`, `notes/2026071289-sol-e01-token-level-decomposition.md` |
| V12 design, arms, gates | `COHERENT-STATE-DECISION-CANARY-V12-PREREGISTRATION.md` §§3–17 |
| Launch/packaging/audit tooling for e01 | `scripts/historical/coherent_canary_v12_e01/` |
| Legacy row provenance and exact analysis commands | `notes/2026070936-provenance-ledger.md`; `notes/2026071115-sol-final-paper-review.md`; per-result manifests under `results/` |
| Data-collection stop and re-entry requirements | `notes/2026071288-sol-data-collection-stop-and-future-reentry.md` |
| Methods/provenance checklist (questions, not answers) | `METHODS-PROVENANCE-REQUIREMENTS.md` |

Raw e01 record: 8,897,066 bytes, SHA-256 `f6622d978c80d8f4f874c208ef9f6d9546ab3f1bf7676655c9a0b4662718d082`, byte-for-byte reconstructible from the tracked packaging; independent pod/local harvests matched after allowlisted path/timestamp normalization only.

## References

1. MEMENTO: Teaching LLMs to Manage Their Own Context. [arXiv:2604.09852](https://arxiv.org/abs/2604.09852).
2. Models Take Notes at Prefill: KV Cache Can Be Editable and Composable. [arXiv:2606.17107](https://arxiv.org/abs/2606.17107).
3. CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion. [arXiv:2405.16444](https://arxiv.org/abs/2405.16444).
4. Learning to Compress Prompts with Gist Tokens. [arXiv:2304.08467](https://arxiv.org/abs/2304.08467).
