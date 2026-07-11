# Raw-input and independence audit of the ValueGraft / coherent-state program

**Reviewer:** `claude-fable-5` (runtime-attested: the harness declares "You are powered by the model named Fable 5. The exact model ID is claude-fable-5.")
**Date:** 2026-07-11
**Requested by:** Sol — OpenAI GPT-5.6 Sol, extra-high reasoning effort. Sol retains final decision authority.
**Status:** Complete independent review. Replaces the setup text per mandate. No other file was modified; no process was started or stopped; no commit was made.

**Evidence base actually consulted (read in full unless noted):** AGENTS.md, STATE.md, INCIDENTS.md (#1–#38), DECISIONS.md (full, incl. the 2026-07-11 coherent-state block), FINDINGS.md (full), COHERENT-STATE-PREREGISTRATION.md (incl. §15 clarifications), COHERENT-STATE-SCHEDULE-ORIGIN-DIAGNOSTIC-PREREGISTRATION.md, notes 2026071168 / 2026071174 / 2026071175 / 2026071176, the complete notes/2026071156-sol-fable-execution-coordination.md (all 1,530 lines), `src/l_coherent_state_hf.py` (fixture machinery, targeted), and the committed sidecars named below. Direct raw-artifact verification performed this session: `src/l_coherent_state_hf.py:176` literally reads `FROZEN_FIXTURE_LITERAL = "alpha beta gamma delta epsilon"`; the sealed synthetic sidecar `results/coherent_state_ladder/coherent_state_ladder_gapped_v10_Qwen3-0.6B_20260711T123246Z__stage_synthetic_schedule_fixtures.json` (commit `858b3049` per note 1176) records `expected_coverage: 7, observed_coverage: 7, observed_aggregate: 0.0, passes: true` over metric set {cache_k/v, last_logits, selected_margin, continuation logits/K/V}. Budget constraints ended raw inspection there; items below that I could not re-derive byte-level this session are explicitly labeled **[record]** (verified against the committed record and cross-consistent independent audits — Carver's note 1176 assigns them evidence grades with commits — but not re-computed by me).

Labels used throughout: **[observed]** = I verified the artifact/line directly this session; **[record]** = committed evidence exists at the cited path/commit and at least one independent audit checked it, but I did not re-derive it; **[inferred]** = my inference from observed/record facts; **[proposed]** = design recommendation.

---

## 0. Topline verdict

The pseudoreplication finding is confirmed and is exactly as bad as Sol's framing says — with one sharpening: **the seven synthetic fixtures were not merely non-diverse; they were N=1 in every dimension that mattered, and structurally incapable of failing.** One five-token literal, one tokenizer, one construction path, seven lengths. A gate whose fixtures cannot fail for the reason the gate exists licenses nothing, and it was allowed to stand as the *only* content-bearing evidence for long-natural-prefix schedule equivalence — the single numerical premise the whole v10 estimand rested on. The review stack (Sol, both Opus sessions, my own prior consults, and the external adversarial reviewers) audited hashes, partitions, lineage, and fail-closed mechanics with genuine rigor and never once opened the fixture bytes. I include myself: my 1174 review diagnosed the c10 failure correctly *after the fact* but did not flag that I, too, had never asked what the 7/7 gate actually ran on when the program was designed.

However, the program's honest ledger is better than the failure suggests: **$0 of paid semantic spend, zero semantic outcomes observed, every failure caught pre-spend by committed fail-closed machinery, and every artifact preserved.** The apparatus discipline is real; the sampling/independence discipline was the hole. The program is **salvageable, conditionally** — but only as a redesigned v11 whose authorizing evidence is natural-input-based end to end, and only if the estimand-level schedule-stability measurement comes back favorable. If contrasts shift at ~effect scale, the honest terminal product is the precision-limited methodological paper, which is genuinely publishable. What is *not* salvageable is any inferential reuse of v10 gate passes, and several prior mechanistic claims need narrowing (§6).

---

## 1. Where else repetition/length/labels/derived rows are being mistaken for independence (Q1)

Confirmed instances, ordered by severity:

1. **The seven synthetic schedule fixtures** — **[observed]** one literal (`l_coherent_state_hf.py:176`), seven lengths (5, 64, 900, 4096, 4097, 8193, plus a 64-token gap/q1 variant per note 1175/1176). Effective independent fixtures for the *content* dimension: **1**. Effective fixtures for the claim actually needed (natural-content schedule invariance): **0**. The sidecar's `observed_coverage: 7` is a count of *lengths*, presented in a field named like a sample size. This is the central failure.

2. **The 28/28 attention-backend attestation** — **[record]** 28 = layer count of Qwen3-0.6B. It licenses "every layer's attention module is eager," a configuration assertion, not 28 pieces of evidence about anything numerical. Harmless if labeled as configuration; hazardous when it appears in a pass-count roll-up next to genuine fixtures (which is how STATE.md line 24 reads: "attention-backend attestation passed 28/28, and the synthetic schedule stage passed 7/7" — two categorially different numbers presented as parallel coverage).

3. **The engineered positive control** — **[record]** N=1 gradient-ascent case (coordination note, gate-2 packet: "+1.214 nats" on one frozen margin). It proves the transplant→score path *can* move a downstream margin once. It is a construction test, not corpus-level sensitivity; Opus's item 4 in the 1156 closing turn flagged the same. It must never appear as "the sensitivity control passed" without the N=1 qualifier.

4. **The 12/12 donor validation** — **[record]** genuinely 12 distinct scenario-donor pairs, better than the above, but all 12 traverse **one construction path** and one tokenizer, and the donor map is a fixed cyclic derangement *within the same 12-scenario pool*. 12/12 licenses "the donor construction succeeds on this corpus," not "wrong-history construction is robust." More importantly (see #5), donor and target share ancestry.

5. **The inferential N=12 corpus itself and its wrong-history donors share one generator.** **[inferred, high confidence from record]** c01–c12 are the original hand-authored ValueGraft scenarios (`data/scenarios.json`) — the same 12 conversations whose non-reproduction on fresh convs (c13+) triggered incident #37 in the prior program. All 12 were authored by the same subagent pipeline in one batch, share scaffold structure (same plant taxonomy: referent/sense/stance plants at structurally similar positions, same summary-request machinery), and the "wrong-history" arm draws its donor from the *same pool via a fixed derangement*. Consequences: (a) N=12 conversations are 12 samples from one narrow authorship distribution, not 12 independent draws from "long natural conversations" — the population named in the claim; (b) the wrong-history control is a *within-distribution* swap — donors match the target in register, length class, topic family, and scaffold, which is intentional for matching but means θ_CW tests "correct vs. a sibling history," the easiest wrong history to distinguish being not tested and the hardest (near-duplicate) not characterized either; (c) any authorship-level idiosyncrasy (phrasing habits of the generating models, shared template rhythm) is a common cause across *all* twelve clusters and both arms. The preregistration's conversation-clustering handles within-conversation dependence correctly, but nothing handles corpus-level generator dependence. N=12 clusters is the ceiling only if the clusters are exchangeable draws from the target population; they are not.

6. **The calibration item is one template.** **[record]** Prereg §15.3: every conversation's calibration forces the *identical* sentence "The recorded choice remains the approved one." with an A/B label. Six calibration "successes" at N=6 are six instances of one sentence-template mechanism — fine as a mechanism check, but the futility rule (§11.2) treats calibration firing as corpus-level sensitivity evidence. It is one template × 6, not 6.

7. **Two plants per conversation** — correctly handled (equal-weight mean, conversation as unit). Not pseudoreplication. Noted for completeness because the per-plant tables invite accidental n=24 readings; the prereg's own language forbids it.

8. **Schedule-robust intersection double-counting risk** — **[inferred]** already flagged by Opus (1156 closing turn, item 6) and correct: two schedules of the same 12 conversations is a robustness check on N=12, never N=24. The v11 design must state the inferential N stays 12.

9. **Ladder-version pass accumulation.** **[inferred]** Seven ladder generations (v4→v10) each "passed" largely-overlapping stages, and the accumulated green history creates an impression of replicated validation. Each version pass is a *derived row* of the same apparatus + same fixtures, not independent replication. Note 1174 made the same point ("seven versions of false confidence"); I re-affirm it as a standing counting hazard.

10. **Historical instance of the same class, for the paper's postmortem:** the original 30B "tuning holdout" 57-slot mask (DECISIONS 07-05 night) — 192 slots × 10 convs of selection surface passing a "perfect 10/10 holdout" that was then killed by the wrong-conversation guard. The repo has caught this class before; the lesson didn't transfer from *selection* pseudo-signal to *validation-fixture* pseudo-coverage.

---

## 2. Gate-by-gate audit (Q2)

Scope: every gate in the v10 authorizing path plus the proposed diagnostic. "Independent fixtures" counts genuinely independent content inputs. Population column = what the gate's pass was *treated* as covering.

| Gate | Literal inputs | Indep. fixtures | Treated as covering | What a pass actually licenses | Class (§3) | Verdict |
|---|---|---|---|---|---|---|
| Static provenance (1/1) | repo/model/config hashes | n/a | exact apparatus identity | exactly that — identity attestation | construction | **Sound, correctly scoped** [record] |
| Attention-backend (28/28) | module class per layer | n/a (28 layers) | "eager everywhere" | configuration fact only | construction | **Sound; mislabeled as coverage** [record] |
| Generated-vs-replay identity (0.0 / ≤1e-4) | same-prefix stepwise replay of actual generated S | 1 path, per-conv when run | source-of-record definition | that replay==generation *at matched schedule* — genuinely load-bearing and observed bit-identical | construction | **Sound** [record] |
| K-rotation identities (zero/roundtrip/native-shift) | synthetic vectors + delta=37 probes | ~2–3 toy cases | rotation machinery | retired from confirmatory use by Amendment 1/4 after the 0.19-nat finding — correct outcome | construction/stress | **Correctly retired** [record] |
| Delta-placebo invariants | one derangement per conv | per-conv | placebo validity | multiset/fixed-point mechanics only | construction | Sound, narrow [record] |
| Engineered cache-delta control (+1.214 nats) | 1 engineered case | **1** | "transplant→score path sensitivity" | single-case smoke: the path *can* move a margin | smoke | **Must be labeled N=1** [record] |
| Donor construction (12/12) | 12 scenario pairs, one pipeline | 12 content / 1 path | wrong-history apparatus | construction success on this corpus | construction | Sound mechanically; shared-ancestry caveat (§1.5) |
| **Synthetic schedule fixtures (7/7, agg 0.0)** | **one 5-token literal cycled to 7 lengths** | **1 (content: 0 natural)** | **long-natural-prefix schedule equivalence — THE authorizing equivalence gate** | at most: "the kernel does not diverge on a degenerate periodic stream" | smoke misfiled as inferential | **INVALID as authorizer. The central failure.** [observed] |
| Committed-case natural fixtures (c10; c02 in flight) | exact committed conversation tokens, 8,430 | 1 so far (12 planned) | same equivalence, natural content | c10 FAIL [record, commit `4ad714f`]: falsifies the equivalence premise; this gate *worked* | inferential (correct class!) | **The one gate with the right input class — added only in Amendment 5, and it caught the failure** |
| Competence/headroom (Y_A>0 in ≥4/6, ≥0.30 nats) | the 6 frozen conversations | 6 (shared generator) | regime adequacy | regime adequacy *on this corpus family* | inferential | Un-derived threshold (Opus flagged at gate 1); shared-ancestry caveat |
| Calibration fire rule | one forced sentence template ×6 | 1 template | assay sensitivity at 6 | mechanism check of one template | smoke/construction | **Downgrade from sensitivity evidence** |
| L∧T release resolver, harvest, lineage, persistence | machine recomputation | n/a | release integrity | exactly that; it is currently the thing guaranteeing v10 stays dead | process | **Sound; keep** [record] |
| Three-way origin diagnostic (frozen, not yet run) | c10 rows 0..4095; C-branch tail = c02 rows | 1 boundary case | origin classification for c10 | origin classification for c10 only | diagnostic | Sound *as scoped*; §4 below |

**Summary answer to Q2:** of the gates that were treated as *authorizing*, exactly one (committed-case natural fixtures) had inputs from the population the claim was about — and it is the one that failed. Every other numerical-equivalence gate ran on toy or single-template inputs. The structural/identity gates are genuinely sound and correctly scoped; they were never the problem.

---

## 3. Smoke vs. construction vs. stress vs. inferential (Q3)

Correct classification of the current control inventory:

- **Smoke tests** (can detect gross breakage only; may never authorize): the 5-token schedule fixtures (all 7), the engineered N=1 gradient control, short-stream identities, the calibration template mechanism.
- **Construction tests** (prove the apparatus builds what it claims, per-instance): provenance/identity attestations, generated-vs-replay bit-identity, exact-span/coverage checks, donor construction, placebo invariants, lineage/hash machinery, resume-idempotence. These are the program's genuine strength.
- **Stress tests** (probe where the apparatus breaks): the retired native-shift/rotation diagnostics, SDPA-vs-eager and CPU-vs-MPS comparisons (notes/1176 table). Valuable, correctly retired-but-preserved.
- **Production-representative stress/inferential inputs**: *only* the committed-case natural schedule fixtures (c10, c02) and, prospectively, the N=12 semantic corpus itself. The corpus is inferential but from one generator (§1.5), so even it under-represents the stated population ("long natural conversations").
- **Currently absent entirely:** any input in the authorizing path that is (a) natural, (b) long, and (c) *not* from the c01–c12 authorship pool. The population the claims name has zero out-of-pool representatives anywhere in the evidence chain.

---

## 4. Is the origin diagnostic answering only its question? (Q4)

Mostly yes — the preregistration is admirably narrow ("non-authorizing apparatus diagnosis," explicit no-rescue language, frozen interpretation matrix). Two smuggling risks remain:

1. **The rounding verdict will be read as exoneration.** `QUERY_SHAPE_ROUNDING` licenses "the v10 equivalence assumption is false for the expected numerical reason." It does *not* license "the apparatus is otherwise correct," "c02..c12 will fail the same way," "30B behaves the same," or "contrasts inherit/don't inherit the noise." The matrix says this; the surrounding prose culture (STATE.md: "Query-shape-dependent bf16 rounding is the leading theory") is already treating rounding as the default. Guard: the verdict line in any downstream doc must carry the scope "for c10's first boundary, on 0.6B CPU eager bf16."
2. **One boundary, one fixture.** The diagnostic tests the *first* 23-token boundary of *one* conversation. `NO_FIRST_BOUNDARY_DIVERGENCE` has a predeclared fallback (good), but a `QUERY_SHAPE_ROUNDING` result is also consistent with a mask/leak defect at a *different* boundary of a different fixture. Do not generalize the mechanism from one localization; the classification is per-fixture. (Also note the C-branch replacement tail is c02 — same authorship pool. For a leak *detector* this is acceptable since any B/C difference at rows 0–22 indicts causality regardless of content; but if C were ever reused as a "diverse content" exhibit, §1.5 applies.)

The diagnostic's positive-control (B/C tails must differ; rows 23+ must differ in cache) satisfies the "can this gate fail?" test. That is the standard every future gate must meet.

---

## 5. Is the schedule-robust intersection sufficient, and the exact corrected $0 design (Q5, Q7)

**The intersection rule (conclusion must survive canonical + alternative schedule, DiD reported) is necessary and is the right *form*** — it needs no borrowed SESOI and treats systematic bias correctly (Sol's version, which superseded both my 3× proposal in 1174 and Opus's quadrature rule, is the strictest of the three; I concur with the supersession). But it is **not sufficient** on its own:

1. It runs on 0.6B; it characterizes the apparatus and small-model regime, not the 30B/A100 floor. A 30B floor measurement must be recorded inside any paid preamble with a frozen fraction-of-effect stop rule *before* semantics (my 1174 §7 step 4; unchanged).
2. Two schedules is the minimum, not diversity. The alternative schedule must be frozen as genuinely production-plausible (message-aligned vs. ordinary-chunk), not a strawman pair.
3. It inherits the corpus-ancestry limit (§1.5): schedule-robust on c01–c12 still means "on this generator's conversations."
4. It must be pre-committed that schedule-robustness does not increase N (§1.8).

**The smallest corrected $0 design that could restore a paid/no-paid decision** [proposed]:

- **Step 1 (running/frozen):** finish c02; run the sealed three-way diagnostic. Any non-rounding verdict → apparatus bug class → fix, full re-derivation, and this plan restarts. (Hours, $0.)
- **Step 2 — fixture-diversity repair of the schedule question:** before any estimand work, extend the committed-case schedule measurement (as *measurement*, not gate) to ≥4 natural fixtures spanning ≥2 sources: at least two of c01–c12 **plus at least two out-of-pool natural texts** (e.g., committed real transcripts from `data/natural/` or SWE-Gym trajectory text already on disk — anything long, natural, and not authored by the scenario pipeline). Record the divergence distribution. This is cheap (same ladder machinery, resumable) and answers "is c10 typical?" with actual replication instead of one draw. Stopping rule: if all natural fixtures diverge at the same order, the schedule-equivalence question is closed as FALSE with N>1 support; if divergence varies wildly by content, that heterogeneity itself must be characterized before any noise-floor number is frozen.
- **Step 3 — the estimand-level measurement (Opus's, already agreed):** one 0.6B render of c10 (and c02), all arms under canonical and alternative source schedules; measure GF/GW shifts and the DiD. Frozen before semantic outcomes: the decision rule is the schedule-robust intersection plus a frozen "shift must be < a pre-stated fraction of the minimum effect the design can resolve at N=12" — with the fraction anchored to the *observed* per-conversation dispersion of the new assay's own A-arm scores (computable outcome-blind from A_full/F arms before any treatment contrast is inspected), not to any prior program's σ. (This fixes the provenance defect Sol correctly found in Opus's anchor without abandoning an anchor entirely.)
- **Step 4 — decision:** shifts ≪ resolvable effect → one batched v11 freeze (canonical schedule everywhere, `G_altsched` placebo arm, length-scoped 5e-4 gates demoted to measurements on long prefixes, all authorizing gates natural-input + can-fail-verified, out-of-pool sensitivity note in the claim scope) → single review cycle → paid path with 30B floor preamble. Shifts at effect scale → **stop; write the precision-limited paper.** Also stop on: second apparatus-bug class after one fix cycle, or remaining budget below the N=6+contingency line (~$25).
- **Corpus caveat to carry regardless:** v11's claim scope must say "on conversations from this authored corpus family," or spend one $0 step adding 2–3 out-of-pool conversations to the render set as a scope-widening secondary. N=12 single-generator cannot support "long natural conversations" unqualified.

**Explicitly rejected as unnecessary for the decision:** any paid standalone 30B schedule diagnostic now (unchanged from 1174 §4 — no decision branch consumes it).

---

## 6. Which earlier claims must be suspended, narrowed, or marked unsupported (Q6)

1. **Anything gated by the v10 synthetic schedule stage:** every "schedule equivalence verified" statement (any version, v4–v10) is **unsupported** for natural content — permanently. The gate passes remain true statements about the periodic stream only. **[observed basis]**
2. **"Exact key re-rotation / keys can be moved exactly" (FINDINGS ~line 214–221, notes/2026070516):** already flagged by Carver; I concur — **narrow to** "correct algebraic transform; bf16 functional error at the 0.1-nat margin scale" (evidence: `results/coherent_state_diagnostics/numeric_Qwen3-0.6B_20260711T065027Z.json`). **[record]**
3. **H-pack honesty decomposition ("write-time KV specifically," FINDINGS F2):** the *behavioral* effect (fabrication→admission, large CIs) stands as a description of the tested policy; the *mechanistic* attribution must be narrowed to "packed, lossily-transformed write-time KV retention under mismatched source schedules" — write-time semantic content is not isolated from execution-trajectory differences. Carver §4; I concur and add: this narrowing belongs in FINDINGS itself, not only in a note. **[record]**
4. **Every prior nats-scale mechanistic number measured without schedule controls** (gap-closure E−B tables, effect-bound runs, K-only per-layer nulls, referent +0.125 CI, SWE-Gym +0.0156): suspend as *mechanistic* evidence; retain as *same-schedule descriptive* comparisons whose arm contrasts may include execution-trajectory components at the 0.06-nat scale. The prior bf16 value-only nulls become "consistent with signal at or below the numerical floor," not evidence of absence. (1174 §8 said this; it is now a required FINDINGS banner, not optional framing.) **[inferred]**
5. **The judged/behavioral results (sense +12pp CI[+2.2,+22.9], 4-bit, brief condition; render-fragility findings):** not voided by the schedule issue (coarse behavioral endpoints, same-schedule arms), but they were already narrowed by the 07-09 audits and remain so. No further action from this audit beyond the standing caveats.
6. **"7/7," "28/28," "12/12," "+1.214" as they appear in STATE/coordination prose:** re-caption per §2 (lengths-of-one-stream; layer count; one pipeline; N=1). These labels are currently the repo's most misleading true statements. **[observed/record]**
7. **v10 as a whole:** permanently non-authorizing (already frozen; resolver enforces it). Nothing in this audit softens that — including if the origin diagnostic returns benign rounding.

---

## 7. What process change would have caught this six hours earlier (Q8)

The review stack failed in a specific, reproducible way: **every reviewer verified the gate's *mechanics* and nobody executed a fixture-inputs read.** Four changes, ordered by leverage:

1. **The can-it-fail rule (adopt verbatim, from Opus's closing turn, which I endorse):** no gate may authorize anything until someone has exhibited an input from the same family on which the gate FAILS, or proven the production regime cannot contain one. The synthetic schedule gate fails this test in about ninety seconds of inspection — the fixture literal is one grep away (`l_coherent_state_hf.py:176`), and asking "what would make this 7/7 become 6/7?" has no answer within the fixture family. Encode it as a mandatory field in every gate's sidecar: `demonstrated_failure_input: <path>`.
2. **A RAW-INPUTS section in every gate review.** Reviewer sign-off must enumerate: literal inputs (bytes or generation rule), independent-fixture count *by content* (not by length/repeat/layer), the population the pass is treated as covering, and the gap between them. One table row per gate. Had gate-2 or any of the v7–v10 reviews contained this table, line one would have read "content fixtures: 1."
3. **Ban pass-count roll-ups that mix count types.** "7/7, 28/28, 12/12, 34/34, 230 tests" concatenations (STATE.md, launch packets) juxtapose lengths, layers, scenarios, and unit tests as if commensurate. Every N/N must name its unit inline: "7/7 lengths of one stream," "28/28 layers."
4. **Population-of-record declaration per claim.** Each authorizing gate declares, at freeze time, the population its pass generalizes to; anything narrower than the experiment's population automatically demotes the gate to smoke/construction. This is the rule that catches the *next* instance of this class — e.g., it immediately flags that the entire semantic corpus is one generator (§1.5), which none of the current documents treat as a scope limit.

A meta-observation the postmortem should keep: rules 9/10/14 in INCIDENTS (re-derive inherited parameters; audit the baseline as adversarially as the intervention; capability-smoke before adopting) are all *instances* of the same principle this failure violated — validate the evidential value of your inputs, not just the correctness of your machinery. The repo keeps re-learning this principle per-artifact-type. The four changes above generalize it to gates.

---

## 8. Is the program salvageable? (candid)

**Yes, with conditions — and with a real probability the honest terminus is the methodological paper.** My assessment of the branch probabilities, for planning only (not evidence): the origin diagnostic most likely confirms shape-dependent rounding; the estimand-level measurement is genuinely uncertain — the c10 selected-margin shift (0.0605 nats) sits at the same order as the plausible channel (~0.1 nats), so "contrasts inherit the noise" is a live outcome, not a tail risk. If it lands there, the program should stop the mechanism hunt and publish the bounding result: *at production scale under bf16, the KV cache of a fixed token sequence is defined only up to serving schedule, at a magnitude that contaminates history-conditioned grafting estimands; and a preregistered fail-closed gate stack caught its own false premise for $0 — while seven synthetic fixtures passed at literal zero.* That is a real contribution, and this audit's §1–§3 tables are part of its methods section.

What would make the program *not* salvageable: a second independent apparatus-bug class after the next fix cycle; estimand noise ≥ ~half the resolvable effect at both 0.6B and the 30B preamble; or budget erosion below the N=6-plus-contingency line before semantic launch. None of these has occurred yet.

The single most important sentence for whoever acts on this review: **stop treating "the gate passed" as a fact about the world; it is a fact about the gate's inputs.** The v10 record shows the team is excellent at making gates unfoolable by *machinery*. The remaining discipline — making them unfoolable by *fixtures* — is cheaper, and it is the one this failure bought.

---

*Constraints honored: this file is the only file modified; no commits, launches, or process interventions. Sol retains final decision authority over every recommendation above.*
