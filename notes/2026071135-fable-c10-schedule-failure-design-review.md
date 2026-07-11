# Fable design review: the c10 schedule-gate failure and what to do next

**Reviewer:** `claude-fable-5` (runtime-observed model identifier; the harness reports
"You are powered by the model named Fable 5. The exact model ID is claude-fable-5.")
**Date:** 2026-07-11
**Role:** scientific/narrative adviser. Sol retains final decision authority.
**Read in full before writing:** AGENTS.md, STATE.md, COHERENT-STATE-PREREGISTRATION.md,
Amendments 1–11 and Clarification 11A, `src/l_coherent_state_hf.py` (schedule-fixture and
gate machinery), `src/coherent_state_runtime.py`, `src/coherent_state_hf.py`, the sealed c10
sidecar (`..._stage_committed_case_schedule_fixtures.json`), and the concluding turns of
`notes/2026071156-sol-fable-execution-coordination.md` (including Opus's contrast-cancellation
argument and its humility flag).

Verified directly from the sealed artifact: c10, 8430 tokens, ordinary partition
`[4096,4096,238]` vs message-aligned `[23,4096,4096,92,123]`; `cache_k_max_abs=16.125`,
`cache_v_max_abs=5.125`, `last_logits_max_abs=0.59375`, `selected_margin_abs_shift=0.060546875`,
continuation logits/K/V `0.84375/1.0/1.59375`; layer 0 K and V exactly `0.0`; layer 1
K `0.0625`, V `0.0009765625`; layer 2 K already `2.0`. All structural booleans
(system_equal, request_header_equal, coverage, continuation position 8430) pass.

---

## 0. Topline

The c10 FAIL is not primarily a gate failure. It is a **finding**: on this platform, at
natural production scale, *"the exact incremental K/V of a text" is not a well-defined
object at the precision this experiment requires.* The cache is defined only up to the
prefill call schedule, and members of that equivalence class differ downstream by ~0.06
nats on a selected-token margin — the same order as the hypothesized channel (~0.1 nats)
and a fifth of the 0.30-nat headroom gate. Every prior manifestation of this root cause
(the 0.19-nat rotation sensitivity, the 0.109-nat SDPA query-shape shift, now the 0.0605-nat
chunking shift) is the same fact wearing different clothes: **bf16 execution-path noise at
long context lives at the scale of the effect being hunted.**

The consequences, in order:

1. Do not spend anything on the frozen v10 gate. Correct call by both auditors; I concur
   and add a sharper reason below (§4).
2. The next dollar of value is $0 and local: finish c02, pause the ladder, run the
   (improved) three-way first-23-row test, then the contrast-level stability measurement
   Opus proposed. Those two results choose the redesign.
3. The redesign should **stop trying to prove schedule equivalence and instead
   canonicalize the schedule and measure schedule sensitivity at the estimand level** —
   including a new, cheap, per-conversation *schedule-placebo arm* that directly bounds
   the numerical noise floor of the primary contrast (§5). That converts an impossible
   invariant into a quantified control.
4. The honest-negative/methodological paper is a real option, not a consolation prize —
   but taking it before the two $0 measurements would be premature. If the contrasts are
   stable, the mechanism experiment survives and the noise story becomes its strongest
   methods section.

---

## 1. What c10 actually falsifies — and what survives

**Falsified (empirically, on this platform):**

- The apparatus premise that prefill call partitioning is numerically neutral at `5e-4`
  for natural long token streams under bf16 eager attention. Layer-0 K/V are bit-equal
  (pure projections of identical embeddings at identical positions — this cleanly rules
  out tokens, positions, embeddings, and cache construction). Layer-1 K differs by
  exactly `0.0625` = one bf16 ULP at magnitude [8,16) — consistent with a single-rounding
  difference in layer-0 attention output (shape-dependent GEMM tiling/accumulation for
  Q=4096 vs Q=23 blocks), then chaotic amplification through 28 layers to 16.125. This is
  the expected signature of accumulation-order divergence, not of a logic bug.
- The evidentiary value of the synthetic fixtures for this question. Seven versions of
  the apparatus passed repetitive cycled-pool fixtures at exactly `0.0` — including at
  L=8193 — and those passes were **unrepresentative**: degenerate content lets rounding
  differences coincide or cancel. A passing gate on an unrepresentative fixture is worse
  than no gate; it manufactured seven versions of false confidence. Amendment 5's move to
  exact committed-token fixtures is the single decision that caught this before paid
  spend — credit the process, and note it was one design choice away from never noticing.
- v10's authorization path. One failed row makes L PASS impossible; the conjunctive
  resolver is frozen; the version is dead as an authorizing instrument. This is the
  contract working as designed.

**Not falsified — and worth saying precisely:**

- The channel hypothesis itself. No semantic arm has ever been scored, at any scale, in
  the gapped design. There is *no evidence for or against* history-specific summary
  state; there is evidence that the v10 measuring instrument could not have resolved it.
- The position-preserving compaction layout, the donor/wrong-history control structure,
  the calibration design, the bit-copy intervention machinery. All of the c10 row's
  structural checks passed; the apparatus did exactly what it claims.
- A causal-mask/future-token leak is not yet excluded (the synthetic L=8193 pass at 0.0
  argues against a gross leak, but degenerate content weakens that argument). The
  three-way test settles it (§3).
- Schedule equivalence on other platforms (A100/CUDA eager) is untested — but the
  amplification mechanism is platform-generic, and the prior 30B fixture failures (v3
  SDPA at 1.625 on a five-token stream) set the prior firmly toward FAIL there too.

**One sharpening of Opus's cancellation argument.** Opus is right that the *destination*
prefix construction is shared by all arms and its split-vs-whole discrepancy is common-mode
to first order. But the schedule confound is **not** common-mode where it matters most: it
lives inside the treatment definition. `G_correct`'s summary rows come from a source cache
built by ordinary 4096-chunk prefill; `G_fresh`'s summary rows are encoded after the
message-aligned two-island destination prefill. The failing fixture is literally the
ordinary-vs-message-aligned comparison — i.e., it measures a schedule difference **between
the two production arms of the primary contrast**, as the coordination summary correctly
states. Opus's caveat (b) noticed this and called the source chunking "arguably part of the
ground-truth write-time state." That defense only works **after** you declare a canonical
serving schedule — otherwise "ordinary 4096 chunks" is an arbitrary apparatus choice among
numerically inequivalent options, none of which is more "ground truth" than another. §5
makes that declaration explicit and turns the residual arbitrariness into a measured
placebo. Opus's proposed contrast-stability measurement remains exactly the right next
empirical step; my disagreement is only with "should cancel to first order" as a prior.

---

## 2. The ladder: continue, stop after c02, or interrupt?

**Recommendation: let c02 finish (it is mid-flight; killing it wastes nearly-complete
evidence and its completion gives a second natural-case magnitude sample), then pause the
ladder and run the three-way test. Resume the remaining ten cases only as idle-capacity
background, off the critical path.**

Information-value accounting:

- Case 1 (c10) established existence. Case 2 (c02) establishes it is systematic rather
  than a c10 pathology, and gives a second magnitude draw. Marginal value of cases 3–12
  is a distribution/figure for the paper — real but small, and reproducible later under
  the redesigned harness (the sidecar machinery is committed and resumable). If c02
  unexpectedly PASSES, that is high-value surprise (case-dependence) and changes the
  diagnostic priority — another reason to let it finish.
- The ladder cannot authorize anything anymore; its remaining value is purely diagnostic,
  and the single local machine is the scarce resource (serial-jobs rule). The three-way
  test and the contrast-stability measurement are the decision-relevant computations;
  everything downstream (redesign choice, any paid spend, the paper framing) waits on
  them and on nothing produced by cases 3–12.
- The ladder's later stages (replay, mask identity, future-mutation at `1e-4`) use short
  or synthetic fixtures, so running the ladder to completion would *not* answer the leak
  question at natural scale — the three-way test answers it directly and more cheaply.

---

## 3. The three-way first-23-row test: decisive with five additions

The proposed design — (A) 23-token call, (B) original 4096-token call, (C) same 4096-token
call with tokens 23–4095 replaced; compare the first 23 cached rows across all layers — is
sound, and its interpretation logic is correct. One point worth making explicit because it
strengthens the test: with a correct causal mask, B=C on rows 0–22 should be **bit-exact**,
not merely small. Masked positions get softmax weight exactly 0.0 in fp32, and 0.0·v
contributes exactly 0.0 to the accumulation regardless of v; identical shapes mean
identical tiling. So any nonzero B−C on rows 0–22 at any layer is a genuine future-token
influence, full stop. Improvements, all cheap:

1. **Determinism repeats.** Run each branch twice; require bit-identical repeats before
   interpreting any cross-branch comparison. Prior diagnostics saw exact repeatability at
   short length; reconfirm at 4096 or the whole test is uninterpretable. Record and pin
   the BLAS/threading environment (`OMP_NUM_THREADS` etc.) — CPU thread count changes
   accumulation order and must match the ladder's environment for the result to transfer.
2. **Positive control on C.** Verify rows 23+ *do* differ between B and C (they must,
   since content differs). If C were accidentally identical to B everywhere, "B=C on rows
   0–22" would be vacuous. This is the repo's validate-before-trusting rule applied here.
3. **Diverse replacement tokens for C**, drawn from a committed donor (e.g., the frozen
   c13 pool), not a repetitive cycle — maximizes leak sensitivity given that degenerate
   content already masked one effect.
4. **Record K and V, per layer, with the first-divergence (layer, row) coordinates** and
   bit-exactness booleans, in a committed sidecar following the ladder schema.
5. **Predeclared fallback:** if A=B=C bit-exact on rows 0–22 (possible — the c10 maxima
   are over all 8430 rows and the divergence could originate past the first boundary),
   extend with a scan comparing `[4096,…]` vs `[23,4073,…]` full-prefix executions to
   locate the first divergent (row, layer). Declare this now so it isn't an adaptive
   choice later.

Interpretation table (unchanged in substance): any layer-0 difference anywhere →
construction bug; B≠C on rows 0–22 → mask/attention leak (apparatus bug class — fix,
new version, re-derive everything, and re-test the c10 fixture, which might then pass);
B=C and A≠B at layer ≥1 → shape-dependent rounding confirmed (expected outcome, prior
~0.8+).

**Then run Opus's measurement.** On 0.6B, locally, for c10 (and c02 once complete): build
the arms with correct-source rows computed under both schedules and measure how much
`G_correct−G_fresh` and `G_correct−G_wrong` shift. This is the decisive quantity — the
noise floor **on the estimand**, not on intermediate tensors. It is ~a day of work and $0,
and it selects between "redesign and proceed" and "precision-limited negative paper"
better than any other spend available at any price.

---

## 4. Paid 30B diagnostic before redesign: **no**

Test the proposal against the only criterion that matters: *which decision changes with
each possible result?*

- If the redesign canonicalizes schedules and adds estimand-level placebo controls (§5),
  it does not depend on 30B/A100 schedule equivalence at all — a PASS or FAIL both change
  nothing about the design.
- If the c10 fixture FAILS on 30B/A100 (the likely outcome — the amplification mechanism
  is platform-generic, and the platform's own five-token SDPA fixture already failed at
  1.625 in v3), we learn what we already assume.
- The one branch where a PASS matters — "keep a v10-like design gated on exact-model
  schedule equivalence" — is a branch I recommend pruning regardless, because it rests
  the entire science on a knife-edge numerical property of a specific kernel library
  version. Even a genuine PASS today is brittle against a CUDA/torch point release, and
  the claim scope would narrow to near-uselessness.

So: **do not buy information that no branch of the decision tree consumes.** The 30B
numerical noise floor *is* worth measuring — but as a recorded diagnostic inside the
redesigned experiment's paid technical preamble (where a pod is already up and it costs
minutes), not as a standalone pod now.

If the owner nevertheless wants it early, the bounded form the auditor sketched is right,
and I'd freeze it as: **Question:** on the exact production checkpoint, A100, CUDA, eager
bf16, does the frozen c10 fixture (identical committed token IDs, `[4096,4096,238]` vs
`[23,4096,4096,92,123]`, same margin/continuation protocol) satisfy `5e-4`, and what is
the per-layer divergence and margin-shift profile if not? **Stopping rule:** one run,
each branch once plus one determinism repeat, harvest, terminate; no retry, no threshold
change, no second fixture. **Artifacts:** the existing committed-case sidecar schema plus
full provenance, committed before interpretation. **Status:** permanently non-authorizing,
named in its own additive amendment before launch, prospectively interpreted exactly as
the auditor stated (PASS → licenses only "exact-platform validation controls" as a design
*option*; FAIL → schedule canonicalization is mandatory; neither resurrects v10).
**Maximum spend: $6** (one community A100 hour plus buffer). But my recommendation is to
fold it into the eventual paid preamble instead.

---

## 5. The cleanest redesign

Design principle: **stop assuming what c10 falsified. Define the state relative to a
declared serving schedule; make every arm-to-arm comparison schedule-matched where
possible; and where schedule differences are intrinsic to the phenomenon, measure their
numerical contribution with a placebo arm instead of gating on an impossible identity.**

Concretely (call it v11, one batched revision, not another single-day amendment chain):

1. **Canonical schedule = message-aligned serving schedule, everywhere.** Prefill every
   prefix (correct source, wrong source, destination islands) message-by-message —
   system as one call, each message block as its own call (4096-chunked within a block
   only when a block exceeds 4096), request/header as one call. This is also the more
   *production-realistic* source path: a live chat session accumulates its cache
   incrementally per turn; nobody re-prefills 8.4k tokens in monolithic 4096 chunks
   before summarizing. The current ordinary-chunk source capture was the less realistic
   choice as well as the schedule-mismatched one. (Even more realistic — the true live
   session cache accumulated across the render itself, with generated replies written
   token-by-token — is worth considering: at 9.5k tokens the 30B KV cache is ~1 GiB,
   trivially retainable for one conversation. If implementation cost is acceptable, the
   live-session cache is the most defensible "state the model actually wrote"; the
   message-aligned re-prefill is the acceptable second-best and much simpler to freeze.)
2. **Consequence for GW:** correct and wrong sources have identical length, identical
   structural slots, and now identical call boundaries — `G_correct−G_wrong` becomes
   fully schedule-matched in shape. It is the cleanest surviving contrast and should be
   promoted rhetorically to the mechanistic headline.
3. **Consequence for GF:** the fresh destination prefix is intrinsically shorter — that
   length difference is the treatment (attending to history vs not) and cannot and should
   not be removed. What canonicalization removes is the *arbitrary* component (chunking
   of identical content). GF is then honestly interpreted as the **deployment-relevant**
   contrast: what a real position-preserving compaction system would actually gain or
   lose, numerical path effects included — because a deployed system faces exactly those.
4. **New arm — the schedule placebo `G_altsched`:** summary rows from the *same correct
   history* under a frozen *alternative* schedule (e.g., the current ordinary 4096
   chunking), inserted into the identical destination. Then
   `|Y(G_altsched) − Y(G_correct)|` is a direct, per-conversation, estimand-level
   measurement of pure numerical schedule noise, with content held fixed. Preregister the
   claim rule: the co-primary intersection counts only if both CI lower bounds clear zero
   **and** the mean effects exceed a frozen multiple (I'd propose 3×) of the mean
   absolute schedule-placebo shift. This converts the dead `5e-4` identity gate into a
   quantified sensitivity control that the estimand itself certifies. Cost: one extra
   source capture and one extra arm score per conversation — marginal. (`G_Vcorrect`/
   `G_Kcorrect` inherit the same placebo bound; `G_delta` stays retired.)
5. **Length-scoped equivalence gates.** Keep the `5e-4` schedule-identity gates only
   where they are empirically satisfiable — short streams (the 146-token destination
   split, the five-token fixtures, stepwise-vs-stepwise identities) — and demote
   long-natural-prefix schedule comparison from gate to *recorded measurement* (the c10
   fixture keeps running and its divergence profile gets reported, not thresholded).
   Calibrate the length threshold from the ladder data already in hand plus c02.
   General lesson to encode: never freeze a tolerance measured on a 5-token fixture and
   apply it at 8,430 tokens; calibrate any numerical tolerance on the production-scale
   distribution before freezing.
6. **Contrast-stability spot-check as a gate.** For the first two paid conversations
   only, compute all arms under both canonical and alternative schedules and require the
   GF/GW shifts to be below the same frozen fraction of effect size. Two cases bound the
   cost; failing it stops before further semantic spend with a clean precision-limited
   verdict.
7. **Keep unchanged:** the position-preserving gapped layout, frozen order and external
   donor map, bit-copy insertion + lineage validation, calibration with the Amendment-2
   variant rule, N=6→12 with the no-efficacy rule, render preservation, the L∧T
   conjunctive release machinery (rebound to v11 artifacts), and the fail-closed
   persistence lifecycle. None of these caused this failure; several are why the failure
   was caught at $0.

This design answers the original question — does correct-history write-time summary state
carry downstream-usable, history-specific information that fresh identical-text encoding
loses — without assuming any numerical equivalence natural tokens have falsified, and it
makes the claim's schedule-relativity explicit instead of implicit.

---

## 6. Essential gates vs over-engineering

Do not weaken (each earned by a real caught defect): exact checkpoint/revision/dtype
attestation; per-layer eager-backend attestation; bit-copy insertion hashes and the
Amendment-8/9 lineage graph; generated-vs-replay identity at matched schedule (observed
bit-identical — it is the definition of the source of record); position/structure and
island-equality gates (they localized this very failure to numerics by all passing);
fail-closed persistence and independent harvest recomputation; the L∧T release
conjunction; render preservation.

Impossible or misdirected invariants (retire or repurpose):

- **The `5e-4` absolute schedule-equivalence requirement on long natural prefixes.** It
  tests a property this platform provably lacks. Keep the measurement, drop the gate,
  replace with §5's placebo/contrast-level controls. This is the one clearly impossible
  invariant in the stack.
- **Synthetic repetitive schedule fixtures as authorizing evidence.** Keep one or two as
  smoke tests; they can never again stand in for natural-token evidence, having produced
  seven versions of false assurance.
- The residual `0.02` packed-rotation diagnostics stay correctly retired-but-visible.

Process observation, offered carefully: eleven amendments were frozen in a single day,
each triggering a full version bump, fresh ladder, and fresh independent reviews. The
gates are not the problem — they caught every failure before paid spend, an unambiguous
success — but the *increment size* is now the dominant cost, and the marginal defects
caught have shifted from science-invalidating to bookkeeping. Batch the redesign into one
v11 with one review cycle. Freeze discipline should bind hardest at outcome-adjacent
choices (arms, thresholds, stopping rules) and be allowed coarser grain at apparatus
plumbing.

---

## 7. Decision tree under the remaining budget

Paid experiment spend to date: $0. Owner ceiling ≈ $60; ~$9.57 conservatively attributed
to consultations → ~$50 of experiment headroom.

1. **Now ($0, hours):** let c02 finish; pause ladder; run the improved three-way test
   with determinism repeats.
   - Layer-0 difference → construction bug: fix, v11, full re-derivation. (Unlikely.)
   - B≠C on rows 0–22 → mask leak: apparatus bug class; fix, v11; re-run the c10 fixture
     — if it then passes, schedule equivalence may be rescuable and §5 simplifies.
     (Low probability.)
   - B=C, A≠B at layer ≥1 → rounding confirmed (expected) → step 2.
2. **($0, ~a day):** contrast-stability + schedule-placebo measurement on 0.6B for
   c10/c02 (Opus's measurement, extended with the `G_altsched` construction).
   - Estimand-level shifts comparable to raw shifts (~0.06 nats) and to plausible
     effects → the channel is unresolvable above the bf16 floor at this scale on this
     evidence → go to 5, unless the owner explicitly wants the fp32-or-short-context
     confirmatory redesign (fp32 at 30B needs multi-GPU — likely outside budget; a
     shortened-context corpus trades away compaction realism — name the trade, owner's
     call).
   - Shifts ≪ effect scale (say <0.01 nats) → estimand robust → step 3.
3. **Redesign v11 as §5, one batch:** fresh ladder (natural-case fixtures now as
   measurements plus the new placebo machinery), one review cycle, clean commit.
4. **Paid execution (~$40–45 available):** one technical run (~$5–8) whose preamble
   *records* the 30B/A100 schedule-divergence and placebo noise floor; hard stop before
   semantics if the measured 30B floor exceeds the frozen fraction of minimum meaningful
   effect. Then semantic N=6 (~$15–20), frozen 6→12 rule, N=12 (~$15) — inside budget
   with margin. All existing stop rules intact.
5. **Abandon tripwires → write the honest paper.** Abandon the mechanism experiment when
   any of: (a) a second apparatus-bug class emerges after one fix cycle; (b) step-2
   estimand noise ≥ ~half of plausible effect; (c) the paid preamble's 30B floor
   similarly large; (d) remaining budget < ~$25 before semantic launch (insufficient for
   N=6 plus contingency). The negative/methodological paper is then the deliverable, and
   it is genuinely publishable (§8).

---

## 8. What this does to the paper

The most honest narrative has changed shape, and — importantly — it is *stronger*, not
weaker, for it. Two layers:

**If v11 runs and the intersection clears:** the claim is a schedule-relative,
history-specific coherent summary-state channel under position-preserving compaction, on
the exact checkpoint/backend, with the numerical noise floor *measured and cleared by a
preregistered margin* (placebo arm + contrast-stability gate) rather than assumed away.
GW is the mechanistic headline; GF the deployment-relevant one. The c10 failure becomes
the methods section that makes the result credible.

**If it never clears or is never run:** the headline is the bounding/methodological
result, and it should be written without apology: *at production scale (30B-class, bf16,
~8k context), the KV cache of a fixed token sequence is defined only up to serving
schedule, and schedule members differ downstream by ~0.06 nats — the same order as the
mechanistic effects reported in the KV-manipulation literature, including this project's
own earlier findings.* Three independent manifestations of one root cause (0.19-nat
rotation sensitivity → position-preserving redesign; 0.109-nat SDPA query-shape shift →
eager backend; 0.0605-nat prefill-chunking shift → this review) form a coherent, citable
result with direct implications for every paper that transplants, edits, or compares
cache states and implicitly assumes they are well-defined. The second citable result is
methodological: a preregistered, fail-closed, adversarially-validated gate stack caught a
false numerical premise for $0 of paid spend across four would-be launches, and the one
fixture class that caught it was exact production tokens — synthetic fixtures passed at
literal zero while the premise was false.

**Relationship to prior ValueGraft results:** state it plainly and additively. The
earlier judged behavioral effects (the 4-bit, brief-condition sense/referent results)
were coarse-grained behavioral measurements and are not retroactively voided, but every
prior *nats-scale mechanistic* number was measured without schedule-noise controls and
now inherits an explicit caveat; the prior bf16 value-only nulls are consistent with a
signal at or below the numerical floor rather than proof of absence. This continues the
existing honest trajectory (the 07-09 no-valid-cornerstone audit) rather than reversing
it, and per the framing-provenance correction, none of this is narrated as review
rescuing a wrong thesis — it is the preregistered instrument reporting its own resolution
limit before any confirmatory claim was made. That is the story of a methodology working.

---

*Constraints honored: no pods launched, no processes stopped, no code edited, no commits.
This file is the only file modified.*
