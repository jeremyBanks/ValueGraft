**Claude Fable 5 — claude-fable-5:**

# Transplanting KV-Cache State Across Conversation Compaction: A Bounded Negative Evaluation and a Methodological Failure Catalogue

**Jeremy Banks · Anthropic Claude Fable 5 · OpenAI GPT-5.6 Sol (extra-high reasoning)**

*Draft status: initial fresh draft for internal verification. Every number below must be re-verified against the artifact map (Appendix B) before publication.*

---

## Abstract

When a long LLM conversation is compacted — replaced by a text summary so that work can continue in a fresh context — everything the model computed while generating the original conversation is discarded along with the text. Prior work has already established, in trained settings, that generation-time key/value (KV) state can carry task-relevant information that re-encoding the visible text does not recover. This project asked a narrower, practical question: can a **training-free, post-hoc transplant** of old KV state (in particular, old value vectors under fresh keys) into a compacted context reliably mitigate compaction damage on an ordinary instruction model?

The answer we can support is: **not as tested, and nothing here licenses a practical claim.** Across four evidence strata that must not be pooled, trustworthy performance evidence was mostly null, harmful, or control-incomplete. The strongest surviving performance lead is a small out-of-fitting likelihood effect on a coding-trajectory proxy (+0.0135 nats/token, 95% bootstrap CI [+0.0083, +0.0190]) with no matched placebo, no executed action, and no task-success signal. The final exact-state mechanism experiment formally stopped at a preregistered technical gate; a single permitted post-stop diagnostic produced a weak, schedule-sensitive, placebo-uncontrolled value-only trace with no behavioral recovery. We report these bounds, and we report what we believe is the project's most durable contribution: a catalogue of ten methodological failure modes — spanning finite-precision execution trajectory, source-state provenance, decoded control validity, control representability, and literal prose/code agreement — each of which can move or invalidate a KV-transplant measurement at the scale of the hoped-for effect, and each of which survived elaborate hash, test, and release machinery before being caught.

---

## 1. Introduction

Conversation compaction is now routine infrastructure: when an agent's context fills, the transcript is summarized, the summary replaces the history, and the agent continues. The visible information loss is obvious. The less obvious loss is computational: the KV cache the model built while *generating* its own turns — attention state that may encode intermediate conclusions, bindings, and commitments not fully explicit in the text — is thrown away and rebuilt by re-encoding the summary from scratch.

That generation-time state can matter is no longer speculative. MEMENTO (§2) demonstrates a 15.3-point accuracy drop when a trained dual-stream model is forced to restart from re-prefilled text instead of retaining its full memento KV. But MEMENTO's channel is *trained* and retains *full* KV written with the original context visible. The question this repository set out to answer is different and, as far as we know, previously untested in this exact form:

> Can old KV state, captured from an ordinary instruction model with no retraining, be transplanted post hoc into a *different* (compacted) context — old values under fresh keys, at matched positions — and measurably restore what compaction destroyed?

The intended endpoint was always practical mitigation: the project owner's goal from the outset was to reduce compaction damage in real agent workloads, with mechanism experiments as the gate to a paired agent evaluation, not as the destination.

This paper reports that the gate was not cleared. We organize the evidence by stratum (§3), because the four bodies of evidence here differ in subject model provenance, source-state procedure, intervention region, controls, and formal status, and pooling them would manufacture confidence none of them individually supports. We then report the mechanism redesign that was meant to resolve the question cleanly, why it formally stopped before its treatment arm (§7), what the one permitted diagnostic showed and failed to show (§8), and the failure catalogue (§9). We close with what is and is not established, and what a defensible agent evaluation would require (§10–11).

Three things this paper does **not** claim, stated up front:

- It does not claim the state channel is absent, or that transplantation can never work. Most of our confidence intervals permit modest effects; a formal stop is not a null result.
- It does not claim equivalence anywhere. "No detected average lift" is the strongest negative reading we use.
- It does not claim any agent-level or task-success effect, positive or negative. No live-agent evaluation was run, deliberately (§11).

---

## 2. Background and related work

**MEMENTO: Teaching LLMs to Manage Their Own Context** ([arXiv:2604.09852](https://arxiv.org/abs/2604.09852)) trains a model to manage its own context with custom in-place block masking, retaining full "memento" KV state written while the original block was visible. On AIME24 with Qwen3-8B, normal memento attention scored 66.1% versus 50.8% after restart/re-prefill — a −15.3-point ablation (with the disclosure that the normal condition used 64 repetitions and the restart condition 8). This is the cleanest evidence we know of that generation-time KV state carries information beyond the restart text. It establishes the broad dual text/state premise in a *trained, full-KV* setting; it does not validate the untrained, partial, post-hoc transplant studied here.

**Models Take Notes at Prefill: KV Cache Can Be Editable and Composable** ([arXiv:2606.17107](https://arxiv.org/abs/2606.17107)), a recent single-author preprint, causally locates conclusions on downstream aggregator/delimiter tokens across model families: in its tasks, the KV of the field containing the answer-relevant content drives under 1% of the decision, and downstream recomputation restores it. If that localization pattern holds more broadly, a transplant confined to summary-region rows plausibly targets the wrong locus entirely — the information may live downstream of the rows we replaced. It is, however, not a direct test of our carrier construction or of agent settings.

**CacheBlend** ([arXiv:2405.16444](https://arxiv.org/abs/2405.16444)) reuses cached text chunks while *selectively recomputing* tokens to restore cross-context interactions. Its success is a useful contrast: it suggests uncompensated transplantation — dropping old rows into a new context with no recomputation to knit them in — omits exactly the repair step that makes reuse work.

**Learning to Compress Prompts with Gist Tokens** ([arXiv:2304.08467](https://arxiv.org/abs/2304.08467)) trains models to encode prompts into reusable gist-token state. It is learned latent compression, not post-hoc state salvage, and marks the "with training, this is achievable" end of the design space.

**Novelty claim, narrowed.** The broad premise — state beyond text — is established elsewhere and is not ours. What we believe is new is the specific combination evaluated here: a *training-free, post-generation* transplant of old-state components (values, keys, or both) across a *text-compaction* boundary on an unmodified instruction model, together with its negative/bounding results and the methodological catalogue that evaluation produced.

---

## 3. The question, the estimands, and the causal ladder

The claim ladder the project used, from weakest to strongest:

1. **Channel existence** — generation-time state differs measurably from re-encoded state (established in prior work; trivially true at the bit level).
2. **History specificity** — state written by *different histories* under *identical visible text* produces measurably different downstream behavior.
3. **Component localization** — the difference is attributable to specific components (values vs. keys, specific layers/regions).
4. **Intervention utility** — transplanting the component recovers a meaningful fraction of compaction damage under controls (placebo, selectivity).
5. **Agent utility** — the intervention improves real task outcomes in paired agent runs.

This project's contested territory is rungs 2–4. Rung 5 was never reached, by design: it was gated on rung 4 and the gate did not clear.

The four evidence strata, which differ in nearly every methodological dimension and are never pooled in this paper:

| Stratum | What actually ran | Strongest licensed reading |
|---|---|---|
| Legacy synthetic (§4) | Foreign/mixed-source conversation bodies; 30B self-generated summaries; prefill-*reconstructed* source state; combined summary+tail value graft | No detected average lift in that exact exploratory apparatus; no equivalence, and no clean generation-state null |
| SWE-Gym/OpenHands proxy (§5) | Historical successful agent trajectories; 30B self-generated brief summaries; actual generation-mutated source snapshot; combined summary+tail value graft | Small out-of-fitting demonstrated-next-action likelihood effect for one selected layer map; no matched placebo, behavior, executed action, or task-success result |
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

Four selected graft variants, evaluated on 18 held-out conversations containing 203 planted probes, gave these graft-minus-compacted estimates (nats/token; conversation-clustered 95% intervals):

| Variant | Mean | 95% CI |
|---|---:|---:|
| Per-head | +0.017 | [−0.027, +0.062] |
| Per-layer | −0.003 | [−0.041, +0.043] |
| Intersection | +0.008 | [−0.032, +0.052] |
| Union | −0.012 | [−0.053, +0.034] |

This supports exactly one sentence: *no detected average lift in the apparatus that ran.* The upper bounds still permit modest positive effects; nothing here is a definitive null, a refutation, or an equivalence result — and because the source state was prefill-reconstructed, this is not a clean test of generation-time state either.

### 4.3 Placebos, read narrowly

The correct old-state graft often outperformed position-shuffled and Gaussian source controls. The safe interpretation is only that aligned source values were **non-exchangeable** — often less disruptive than grossly mismatched full vectors. These controls did not match the actual treatment delta (V_old − V_fresh): Gaussian rows matched source energy rather than the delta, shuffles were not proven strict derangements, and correct old rows may simply have been geometrically closer to fresh rows. We therefore do not use these placebos as evidence of content-specific semantic information or of "a real mechanism."

### 4.4 Retired legacy claims

Several earlier headline claims from this stratum are void or retired, and we list them so they are not re-cited:

- **The four-level compression treatment conclusion is void.** Ultra/brief/medium cells generated summary text under level-specific requests but reconstructed old state under the default realistic request; only the realistic cell was internally consistent. Compression ratios remain descriptions of text. No flat-across-30× or severity conclusion is licensed.
- **The +10–12 percentage-point judged-recovery headline is retired.** It came from a local 4-bit MLX stack, brief/adversarial summaries, conversations c01–c12, and was render-fragile: a clean rerender moved judged sense from +8.7 to +1.0 points and referent from +9.7 to +5.2, with intervals spanning zero.
- **The packed "H-pack" construction is not evidence for value-only recovery.** It changed fabrication/admission behavior but simultaneously altered layout, transformed K, and V; it restored 0/24 evicted facts, and the earlier "38/48 accurate" claim was unreproducible.
- **Cross-architecture rows support no law.** Results were mixed and mostly adverse, but heterogeneous gates and provenance do not support a QK-norm, dense/MoE, or keys-neutral generalization.

---

## 5. The coding-domain likelihood proxy (SWE-Gym/OpenHands)

The strongest surviving performance lead in the repository is a **likelihood proxy on historical coding-agent trajectories**, and we are careful to describe it as exactly that.

**Setup.** Historical successful SWE-Gym/OpenHands trajectories were compacted with a 30B self-generated brief summary; unlike the legacy stratum, the source state was the *actual generation-mutated snapshot*. The intervention was a combined summary+tail value graft. A layer map (α = 1 on layers 12–17 and 30–35) was selected on 41 fresh-pool trajectories, evaluated on 57 disjoint fresh trajectories, and partially confirmed on 45 of a planned 75 original-pool trajectories. The metric is teacher-forced mean token log-probability of the *historically demonstrated next action*.

**Out-of-fitting results (selected map minus baseline, nats/token):**

- Fresh evaluation (n = 57): **+0.0117**, 95% bootstrap CI [+0.0063, +0.0172].
- Original-pool partial confirmation (n = 45): about **+0.0158**, interval above zero.
- Pooled wholly out-of-fitting (n = 102): **+0.0135** [+0.0083, +0.0190].

**Why this is not an agent result.** Every one of the following caveats binds:

- The target is one historically demonstrated next action, not necessarily the unique correct action.
- No action was applied, no patch tested, no repository task solved.
- The selected map had **no matched placebo** restricted to its selected layers, so map-specific content cannot be separated from map-specific disturbance.
- Against a fixed graft on the fresh 57, the selected map's advantage was inconclusive (+0.0041, CI spanning zero).
- A coarse structural match on tool/path/command was 53% (selected) versus 54% (baseline) on the fresh set; across all 102 out-of-fitting rows the graft fixed three baseline misses and broke three baseline successes — behaviorally a wash at this resolution.
- Confirmation stopped at 45/75 trajectories.
- Bootstrap treated trajectories as independent; a repository/task-cluster sensitivity analysis was not completed.
- The fixed scalar graft was sample-heterogeneous: positive on the original 75, absent on the disjoint 98, pooled interval spanning zero. A fixed scalar under realistic summaries was null, and the selected map was never tested there.
- Dataset provenance is partially unpreserved: the local `swegym.parquet` hashes `ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1`, but the immutable upstream revision, the historical trajectory-generating model identities, and per-trajectory generated summary text/hashes were not retained.

The licensed sentence is: *a selected-map, demonstrated-next-action likelihood lead of about +0.013 nats/token that survived out-of-fitting evaluation but never faced a matched placebo or a behavioral endpoint.* It is a reason the question stays open; it is not coding-agent improvement.

---

## 6. The coherent-state redesign

### 6.1 v10/v11: methodological evidence only

The coherent-state v10/v11 effort — a preregistered assay with local technical fixtures and an authored corpus pipeline — produced failed controls and a paused draft corpus, and contributes methodological evidence only (§9). The planned twelve-case v11 corpus build was deliberately paused because no exact-model effect had yet justified confirmatory machinery. There is no semantic treatment result and no sample from this stratum.

### 6.2 v12: the same-visible-text design

The final canary, v12, asked the cleanest version of the question. Two histories — one **correct**, one **minimally counterfactual** — each produce KV state for the *same fixed neutral carrier text*, and a **fresh** arm re-encodes that same carrier text in a position-preserving gapped destination. Because the visible text downstream of the histories is identical across arms, any downstream difference must come through state. The primary semantic contrast is correct-history versus wrong-history state (D); correct-history versus fresh is a utility contrast (U). Full K+V and value-only transplants are separate families, and the intervention operates at matched positions with no re-rotation of quantized keys (a hard-won requirement; see §9, item 4).

The carrier was not a naturally generated summary. It was this externally fixed neutral note:

> The prior discussion established the operating context and relevant decision criteria. Continue from this handoff, preserve the existing constraints, and answer later questions from the state available here. No unresolved action is introduced by this note.

followed by a fixed anchor request and a fixed `Acknowledged.` reply. Two replay schedules produced the historical state: **N** force-fed historical assistant content token by token (q = 1), and **P** prefilled complete historical messages; both then forced the identical carrier. Neither schedule is native continuous live-agent generation — a limitation, not a footnote, because schedule turned out to matter (§8).

Three readout regions were defined: R1 (carrier content), R2 (adds the carrier close plus the anchor user turn and assistant-generation header), and R3 (adds the fixed acknowledgment content). **R2 alone was primary.** The N-schedule grid crossed key source × value source over {Fresh, Correct, Wrong}: FF, FC, FW, CF, CC, CW, WF, WC, WW; P used the primary R2 subset. All cells, regions, and schedules are correlated views of **one case** — they add perspectives, not sample size.

The test case, e01, was an engineered fixed transcript authored by an independent Codex subagent (exact runtime model/effort not exposed). The correct and counterfactual histories differ **only in a rollback interval — 6:40 versus 10:40** — and under the transcript's explicit decision rule that single difference flips the correct focal answer between `partner beta` and `staff ring`. An unchanged nonfocal probe (correct `21 days`, counter-answer `30 days`) serves as the selectivity control.

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

### 7.2 Adverse natural calibration

Independently, the natural-calibration statistics came out adverse (ρ_green = −0.035714, ρ_amber = 0.297872): the path control's sensitivity did not positively track the natural-variation tiers it was calibrated against. We note this for completeness while emphasizing the design point it reinforces: a path control, passing or failing, establishes only mechanical propagation. It never establishes semantic validity.

---

## 8. The e01 diagnostic

Exactly one unchanged e01 treatment execution was permitted after the stop, explicitly classified **diagnostic-only**: its Phase A integrity check was `PRETREATMENT_PASS`, and its receipt records no formal or expansion eligibility. It is one engineered case. Nothing below carries a p-value, an interval, or a population claim.

### 8.1 Results

`D` is correct-history minus wrong-history state (focal and nonfocal probes); `SEL` subtracts absolute nonfocal movement from focal movement; `H+` additionally requires that the correct target itself improve; `U`/`U+` compare the correct-history graft against fresh. All values are forced-continuation log-probability movements in nats.

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
- **The recovered fraction is tiny.** Fresh re-encoding of the carrier cost 22.298 nats of focal margin and 22.291 nats of correct-target log-probability relative to correct-history state. The best value-only cell recovered 0.81% and 4.24% of that damage, respectively.
- **There was no behavioral recovery.** All 31 primary cells, whatever their state source, freely generated the same unrelated/wrong answers (`Ring 3`, `30 days`). No transplant changed what the model actually said.
- **The placebo is missing, not null.** All three planned norm-matched bf16 placebos were unavailable at the same first nonzero early row after 1,024 construction attempts each — the constructions were mathematically valid but not bf16-representable. Zero available placebos is missing evidence, not a passed control.
- **The placebo cannot be repaired locally.** Only hashes, shapes, traces, and scores were persisted — not K/V tensor element values — so reconstructing the exact state would require rerunning the model (§9, item 9).

Integrity of the diagnostic itself is well evidenced: treatment-fresh score objects matched Phase A canonically; the 8,897,066-byte raw record reconstructed byte-for-byte to SHA-256 `f6622d978c80d8f4f874c208ef9f6d9546ab3f1bf7676655c9a0b4662718d082`; independent pod and local harvests matched after only allowlisted path/timestamp normalization; and no unexpected post-run difference was found.

### 8.3 The strongest safe sentence

> One fixed engineered case exhibited deterministic, focal-selective forced-logprob movement in the exact N/R2 value-only cell under identical visible text.

We classify this as a weak, uncontrolled, schedule-sensitive mechanistic hint. It is not semantic, not useful, not robust, not general, and not behaviorally recovered — each of those words would require evidence e01 does not contain.

---

## 9. The methodological failure catalogue

We consider this the paper's most durable contribution. Each entry is a failure that occurred in this project, the consequence it had or would have had, and the guardrail we now consider mandatory for activation-state experiments. Several of these passed elaborate hash, test, review, and release machinery — the machinery validated the wrong literal assumption.

| # | Failure | Consequence | Guardrail |
|---|---|---|---|
| 1 | **Query shape/schedule is part of the computed bf16 state.** On local `Qwen/Qwen3-0.6B` (CPU, bf16, eager), the same 8,430-token c10 prefix under ordinary versus coarse replay schedules moved a fixed margin by 0.060546875 nats; c02 moved 0.1318359375. With future tokens held constant, first divergence localized to layer-0 attention when query width changed 23→4096. | Replay-schedule choice alone can produce "effects" at the scale of the sought signal. | Treat schedule as an experimental variable (`QUERY_SHAPE_ROUNDING`); measure treatment and control under bit-identical schedules. Do not invent a specific tiling/reduction mechanism or extrapolate the magnitude to other models/hardware. |
| 2 | **A validation gate that cannot fail.** An apparent 7/7 zero-difference validation used seven lengths of one five-token periodic stream — pseudoreplication, not seven inputs. | A broken equivalence assumption certified as verified. | Every gate must be able to fail for its intended reason; validate on representative, non-degenerate inputs. |
| 3 | **Mechanical geometry mistaken for semantic validity.** A wrong-history control cycled short donor text up to 51 times; later exact-width counterfactuals preserved tokenizer geometry but failed full decoded review. | "Controls" that were not meaningful alternative histories. | Decode and review every control as text; token-geometry equivalence is necessary, never sufficient. |
| 4 | **Position correction is numerically consequential.** bf16 re-rotation of already-quantized keys shifted target margins at the effect scale. | Position handling masquerading as treatment effect. | Use position-preserving gapped state; never re-rotate quantized keys. |
| 5 | **Small-model green ladders certify plumbing, not the production regime.** Local identity ladders passed while long-context 30B behavior failed. | False confidence transferred across scale. | Gate production claims on exact-stack (model, dtype, length, hardware) checks. |
| 6 | **Controls can be valid in mathematics and unavailable after casting.** The norm-matched e01 placebo construction failed bf16 representability (0/3 after 1,024 attempts each). | The critical control silently absent at analysis time. | Prove control representability in the production dtype before the run; treat "no placebo" as missing evidence. |
| 7 | **Prose/code agreement is itself a preregistration gate.** Hash-frozen code did not resolve ambiguous stopping semantics (§7.1). | A pass/fail verdict that depended on which artifact you believed. | Diff the literal decision rule in prose against the literal branch in code before sealing; ambiguity discovered later must be dispositioned conservatively. |
| 8 | **Container identity is not host identity.** The same Secure A100 type and image surfaced NVIDIA drivers 580.159.03 and 550.90.12; pinned CUDA initialized on only one. | Irreproducible runtime; wasted provisioning. | Admission-gate the actual GPU, driver, and memory before bootstrap. (An operations/reproducibility finding; it explains no scientific error above.) |
| 9 | **State needed for later controls was not preserved.** E01 persisted hashes/shapes/traces/scores; hashes cannot reconstruct K/V rows. | The missing placebo cannot be repaired without a full rerun. | Budget a bounded tensor bundle for control-relevant geometry — here ~39.84 MiB would have sufficed, versus ~480.94 MiB for five full snapshots. |
| 10 | **Confirmatory machinery ran ahead of an observed signal.** The v11 twelve-case corpus build was paused because no exact-model effect justified it; e01 did not meet the re-entry standard either. | Sunk cost pressure toward running an unjustified confirmation. | Sequence confirmation strictly behind an observed, gated signal; write the re-entry standard down before you want it. |

The unifying lesson is not that rigor failed. Hashes were checked, partitions held, releases were sealed — and several of these failures happened anyway, because rigorous checks of *execution* can still validate the wrong *estimand* when the literal fixture, dtype, control, or stopping rule does not test the generalization being claimed. The check you need is the one aimed at the assumption you did not know you were making.

---

## 10. What is and is not established

**Established by this repository (with the stated bounds):**

- No detected average lift for the legacy synthetic value-graft apparatus on held-out data (§4.2), with upper bounds still permitting modest effects and with a prefill-reconstruction provenance defect.
- A small, out-of-fitting, selected-map likelihood lead on demonstrated next actions in coding trajectories (§5), control-incomplete and behaviorally unresolved.
- A formal technical stop of the exact-state canary under its literal preregistered rule (§7), before any treatment evidence.
- One N=1, diagnostic-only, placebo-missing, schedule-sensitive value-only trace under identical visible text, with no behavioral recovery (§8).
- Ten concrete failure modes with guardrails for activation-state experimentation (§9).

**Not established, in either direction:**

- Whether generation-time state on this model carries recoverable history-specific semantic content at all (the redesign stopped before answering).
- Whether any training-free transplant family can recover a practically meaningful fraction of compaction damage.
- Any effect — positive, negative, or null — on real agent task outcomes.
- Any cross-architecture generalization.

**Also deliberately absent:** an overall cost figure. The end-to-end money/token audit is still open and will be reported separately, distinguishing cash, subscription usage, provider credits, list-price equivalents, estimates, lower bounds, and unknowns.

---

## 11. Why no live-agent evaluation ran, and what one would require

No live-agent evaluation was run, and none is authorized. The reason is scientific, not logistical: the mechanistic treatment did not clear its gate. Running coding agents now would test an unstable, control-incomplete intervention, and any observed task movement — in either direction — would be uninterpretable.

A defensible practical evaluation, contingent on a *future* mechanism study succeeding, would require at minimum:

1. A no-treatment canary first: full-context versus ordinary-compaction capability/damage measurement, on tasks the subject model can actually solve and where compaction genuinely fires.
2. One treatment frozen before any agent outcome is seen.
3. Paired forks from identical repository/container/task/seed state — standard compaction versus treatment — with full-context runs as reference where feasible.
4. Repository-level deterministic tests as the primary endpoint.
5. Full preservation of transcripts, summaries, patches, tests, tool/token budgets, and every compaction event.
6. Inference clustered by task/repository, reported as a feasibility result, not a powered product claim.

The formal re-entry requirements are recorded in `notes/2026071288-sol-data-collection-stop-and-future-reentry.md`. Paid data collection for this project is finished.

---

## 12. Limitations

Beyond the per-stratum caveats above: everything here concerns one model family (Qwen3), and mostly one checkpoint. Neither v12 replay schedule is native continuous generation, and the one suggestive cell was schedule-sensitive — the construct validity of forced replay for live-agent state is untested. The legacy stratum's source state was reconstructed rather than captured. The strongest positive lead lacks its matched placebo; the diagnostic lacks any placebo. Several provenance elements are irrecoverable: legacy `git_commit: null` manifests, the SWE-Gym upstream revision and trajectory-generator identities, per-trajectory summary text, and all K/V tensor values from e01. No behavioral endpoint anywhere in the repository moved in the treatment's favor.

---

## Author contributions

- **Jeremy Banks** — the research question, direction, and funding; repeated methodological corrections; final scope and taste decisions. The mitigation-first framing of the project was his intent from the beginning.
- **Anthropic Claude Fable 5** — initial final-paper drafting and synthesis; earlier strategic/diagnostic consultation recorded in the notes archive.
- **OpenAI GPT-5.6 Sol (extra-high reasoning)** — primary final analysis, execution, interpretation, and scientific decision authority, including the facts brief this draft is bound to.

Additional model contributions, attributed only where runtime evidence supports them: **Claude Opus 4.8** (paper-spine and methodological-postmortem analysis; substantial experimental execution recorded in session logs), and assistant sessions labeled **Claude Sonnet, Claude Opus, and Claude Fable** (authorship of legacy conversation bodies c13–c24; exact runtime receipts not uniformly exposed). The e01 transcript was authored by an independent Codex subagent whose exact runtime model and effort were not exposed.

## Acknowledgments

**OpenAI GPT-5.5 (extra-high reasoning)** provided independent external review of prior report revisions. Local body generation for c07–c12 used `mlx-community/Qwen3-4B-Instruct-2507-4bit`.

---

## Appendix A: Data and runtime provenance

**Formal v12 / e01 subject and stack.** Model `Qwen/Qwen3-30B-A3B-Instruct-2507`, revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`; bf16; eager attention; 48 layers, 32 query heads, 4 KV heads, head dimension 128, RoPE theta 10,000,000. Host: accepted A100 80GB PCIe, NVIDIA driver 580.159.04, Linux 6.8.0-100. Software: Python 3.12.11, Torch 2.12.1+cu130, Transformers 5.0.0. Scientific commit `cbdfa481fe08de62bf8178d09ca040716d610f21`; runtime fingerprint `b6158b989b0b467c46589f3ecbe69acba88645d4cd193dd23d925e2e4cd69484`. Replay: N forces historical assistant content token by token (q = 1); P prefills complete historical messages; both force the identical fixed carrier thereafter. Readout: forced-continuation log-probabilities over R1/R2/R3, R2 primary; N grid FF/FC/FW/CF/CC/CW/WF/WC/WW (key source first, value source second); P on the primary R2 subset. Estimands and sign conventions as defined in §8.1. Status: formal arm terminal at the path-control stop; e01 execution diagnostic-only (`PRETREATMENT_PASS`, no formal or expansion eligibility). Controls: three norm-matched bf16 placebos planned, zero available (representability failure, 1,024 attempts each). Persistence: hashes/shapes/traces/scores only; no K/V tensor values.

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
