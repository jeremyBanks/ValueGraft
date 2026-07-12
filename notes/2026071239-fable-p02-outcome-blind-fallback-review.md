# P02 outcome-blind fallback review — independent adviser assessment

**Author (exact runtime identity):** Claude Fable 5, model id `claude-fable-5`, running as an
independent scientific-design adviser in a fresh session.
**Date:** 2026-07-12
**Inputs read (only):** AGENTS.md; PRECISION-PROBE-P01-PREREGISTRATION.md; my prior A3 review
(`notes/20260712A3-...`); Sol's prelaunch audit (`notes/20260712A4-...`);
`src/precision_probe_p01_loader.py` (structure/constants); `scripts/run_precision_probe_p01.py`
(grid/extension/decision boundary only); INCIDENTS.md #47–#48. **No P01 outcome artifact,
manifest, package, scratch, or log was opened, listed, or searched.** All reasoning below uses
only the owner-supplied non-outcome facts (timing boundaries, gate passes, cap arithmetic).

---

## 1. Verdict: is a P02 worth running?

**Yes — a deliberately trimmed P02 is worth running, conditionally.** P01, as frozen, cannot
answer its own screening question: its primary estimand is the cross-runtime contrast
`Delta_runtime = D^V_NF4 − D^V_bf16` on the **same host**, and the observed timing arithmetic
(§6) shows the same-host bf16 pair cannot complete inside the 7,200 s cap. What P01 will deliver
is a true-NF4 technical record plus NF4-only outcomes — genuinely valuable, but with no matched
comparator. The archived Transformers-5 bf16 e01 record is confounded on Transformers major
version, host, and driver simultaneously; the P01 preregistration itself demoted it to
"secondary historical context, not the paired comparator," and that judgment was correct. Using
it now as the comparator would quietly reverse a frozen design decision because the money ran out.

The condition: P02 is worth it **only if the paper needs any matched cross-regime sentence.** If
the owner would accept the weaker formulation — "the apparatus and its headline observables run
under true NF4 (matched-runtime, NF4-only); the archived bf16 record from a prior stack is
directionally consistent context" — then P01's existing artifacts suffice at zero additional
dollars, and that is a legitimate, honest paper paragraph. But for ~$2.5–3.5 (§6), a trimmed
matched pair removes a permanent confound from the paper's only precision-axis paragraph and
additionally buys a cross-host NF4 stability check (P01-NF4 vs P02-NF4, descriptive) and a
cross-Transformers-version bf16 replication attempt (P02-bf16 vs archived bf16, descriptive) as
free riders. That is good information per dollar. **Recommendation: run the trimmed P02.**

What P02 must NOT be: a rerun of the full 34-arm grid. The full grid is what broke P01's budget,
and §9 argues most of it never had an inferential role.

## 2. Smallest defensible execution set

One fresh secure A100-80GB pod, same admission gates, same isolated Transformers 4.57.6 stack,
exact pinned revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, slow tokenizer, eager
attention, KV bf16 throughout, e01 fixture only, **NF4 then bf16 sequentially on the same host**
(per the owner's frozen constraint and because a cross-pod NF4(P01)↔bf16(P02) comparison would
reintroduce exactly the host confound the design exists to remove — so yes, NF4 must be rerun).

Per regime, per repeat:

- **Phase A, full (3 source executions × focal/nonfocal probes):** correct-history oracle,
  wrong-history oracle, fresh compaction. All three are structurally required — oracle-correct
  and fresh anchor estimand E1 (damage); oracle-wrong is the value/key source for every W-graft.
  Observed cost ≈ 9 min; not trimmable without losing sources.
- **Treatment, 7 arms (vs P01's 34):**
  - N/R2: **FC, FW** (primary value-only contrast), **CC, WW** (full-KV strongest-dose pair),
    **FF** (fresh-self null-graft anchor — the closest available placebo-shaped control given
    incident #46; costs ~1 arm-minute).
  - P/R2: **FC, FW** (schedule-sensitivity check on the primary contrast only — e01's 8.3×
    schedule collapse makes this the one known fragility axis worth 2 arms; first tier dropped
    under time pressure).
- **Dropped, logged per the no-silent-caps rule:** all R1/R3 regions (18 arms), key-only CF/WF
  and crossed CW/WC (4 arms), P-schedule CC/WW (2 arms), and **all placebo attempts (3 arms)** —
  construction failed at bf16 after 1,024 attempts (incident #46) and NF4 does not change cache
  dtype, so re-attempting is predictably futile spend (my A3 §g1 said this before P01; P01 kept
  them anyway).

**Repeats:** two full trimmed repeats per regime (not targeted-cell repeats), because a full
trimmed repeat now costs ~15 min and whole-payload G4-style comparison is a strictly stronger
determinism check than cherry-picked cells. But repeats are the first casualty in the drop
ladder (§6): **completing the matched pair outranks completing the repeats.**

**Reused within the run:** model load + G1/G2 gates once per regime (not per repeat); checkpoint
shards downloaded once, shared by both regimes; case/tokenizer fixtures byte-identical.
**Reused from P01:** the entire loader/runner/analyzer/launchd-watchdog apparatus (now hardened
by incidents #47–#48), the technical-gate expectations (18,672/18,432 module counts, sentinel
experts, KV shape/dtype checks), and the observed timing model. **The one apparatus change:**
`run_treatment_case` is monolithic and complete-grid; P02 needs a parameterized arm list or a
thin new driver over the same per-arm internals. Bound it (small, tested, two-review), and note
it also moots Sol's audit residual #2 (mid-treatment loss) since the grid is now 7 arms with
per-repeat persistence. Freeze the P02 preregistration **before reading any P01 value**; the
second NF4 repeat finishing does not block the freeze as long as its contents stay unread.

## 3. The expensive-cell justification, item by item

- **Full Phase A per repeat: KEEP** — sources are prerequisites; E1 lives here; 9 min.
- **FC and FW: KEEP** — the contrast is undefined without both.
- **Value-only AND K+V: KEEP both pairs** — value-only is the historical primary; full-KV is
  the maximum-dose cell for the behavioral-null endpoint E2 (if any cell changes a generated
  answer, CC/WW is where). Four arms, ~3.5 min.
- **FF anchor: KEEP** — one arm buys an in-run null-graft control in each regime, the only
  placebo-shaped object available.
- **Schedule P: KEEP FC/FW only, drop-tier 1** — 2 arms for the known fragility axis; cut first.
- **Placebo attempts: DROP** — evidence-based futility (incident #46).
- **R1/R3, key-only, crossed: DROP** — descriptive freight with no predeclared claim slot; 22 of
  P01's 34 arms fall here and they are what sank the budget.
- **Two full repeats: KEEP as goal, sacrificial under the ladder.**

## 4. Predeclared primary estimands and outcome classifications (no P01 values used)

For regime r ∈ {NF4, bf16} within P02, `Y_r(X) = mean_lp(correct|X) − mean_lp(counterfactual|X)`:

- **E1 (damage robustness), primary:** `A_r = Y_r(oracle-correct) − Y_r(fresh)`, focal.
  Classified **reproduced** iff sign(A_NF4) = sign(A_bf16) and the ratio A_NF4/A_bf16 ∈ [0.5, 2];
  otherwise **not reproduced**. (Well-powered at N=1 because the bf16-class magnitude is ~22 nats.)
- **E2 (behavioral null), primary:** per cell (fresh, FF, FC, FW, CC, WW; N/R2), a binary
  any-generated-answer-change indicator vs the fresh baseline, per regime. Classified
  **identical pattern** iff the 6-bit vectors match across regimes; any mismatch is a
  **screening flag** — never an efficacy or dependence claim.
- **E3 (graft contrast), descriptive-primary:** `D^V_r = Y_r(FC) − Y_r(FW)` and
  `D^KV_r = Y_r(CC) − Y_r(WW)` at N/R2; report `Delta = D^V_NF4 − D^V_bf16` with exactly three
  predeclared descriptors: per-regime sign, cross-regime sign agreement, and ratio bucket
  (<0.5×, 0.5–2×, >2×). **Predeclared and binding: no magnitude of Delta on this one engineered
  fixture, with no placebo and known 8.3× schedule fragility, licenses the phrase "quantization
  dependence" or any population/interaction claim. E3 divergence of any size licenses at most a
  new multi-case preregistration.**
- **Secondary, descriptive, no claim slots:** P/R2 FC−FW and the N−P shift; FF movement;
  nonfocal D and SEL; per-token contributions; P02-NF4 vs P01-NF4 (cross-host); P02-bf16 vs
  archived Transformers-5 bf16 (cross-version).

Post-hoc cell selection is prevented structurally: the paper's precision paragraph gets exactly
three claim slots (E1, E2, E3-descriptive), written into the P02 prereg as template sentences
before launch. Every other number is labeled descriptive in the frozen analysis plan and has no
sentence slot to occupy.

## 5. Repeat-stability gate

As P01 G4: after both regimes, normalized whole-payload comparison across within-regime repeats
(paths/timing excluded). If a regime's repeats differ, report the exact fields and do not
interpret any cross-regime contrast smaller than the observed repeat instability; E1/E2 survive
nondeterminism, E3 does not. No third repeat.

## 6. Timing arithmetic, ceiling, and outcome-blind stopping rules

Observed P01 costs (artifact-boundary timing only): runner start ≈ 258 s provider-elapsed;
NF4 technical (load + G1/G2) ≈ 12 min; Phase A ≈ 9 min; 34-arm treatment ≈ 29 min ⇒ ≈ 51 s/arm.

Projected P02 (trimmed, 7 arms ⇒ treatment ≈ 6 min; repeat ≈ 15 min):

| Stage | min |
|---|---|
| Allocation/setup | 5 |
| NF4 load + G1/G2 | 12 |
| NF4 repeat 1, repeat 2 | 15 + 15 |
| bf16 load + G1/G2 | 12 |
| bf16 repeat 1, repeat 2 | 15 + 15 |
| **Total** | **89** |

With a 20% buffer: ≈ 107 min. **Ceiling: provider cap 9,000 s (2.5 h) AND cash cap $4.00,
whichever first**, same launchd watchdog and 180 s reserve (rules 38/39 machinery unchanged).
At the observed ~$1.39/h class of rate, 9,000 s ≈ $3.48 ≤ $4; expected spend ≈ $2.1–2.5. I
deliberately do not keep 7,200 s: the trimmed design fits it on paper (107 < 117 min usable) but
the margin is thin against an unmodeled bf16 load/throughput surprise, and the marginal ~$0.70
of headroom is cheap insurance. Within the $4–6 preference either way.

**Outcome-blind continuation ladder** (inputs: monotonic wall clock, completion status, rate —
same information surface as P01's extension rule, but pointed at *shrinking*, which is the rule
P01 lacked). Let `t1` = observed NF4 repeat-1 wall time, `L` = observed NF4 load+gate time,
`cap*` = cap − 180 s:

1. After NF4 r1: run NF4 r2 only if `elapsed + 1.20×(t1 + L + 2·t1) ≤ cap*` (r2 now, then bf16
   load/gates ≈ L, then two bf16 repeats each assumed ≤ t1). Else skip to bf16.
2. After bf16 r1: run bf16 r2 only if `elapsed + 1.20×t1 ≤ cap*`.
3. Tier-1 trim: if either check fails by ≤ 2×(P-arm time), instead first drop the P/R2 arms from
   all remaining executions and re-evaluate.
4. **Abort-before-bf16 rule:** if after NF4 r1 even one bf16 repeat cannot fit —
   `elapsed + 1.20×(L + t1) > cap*` — **stop, package, terminate before loading bf16.** An
   NF4-only P02 duplicates P01's failure mode and buys nothing; keep the money. This is the one
   scenario where a subsequent P03 with a 3 h cap (≈ $4.2–5) becomes methodologically necessary
   and the owner's "a little more" clause applies.

Predeclared priority order: **NF4 r1 > bf16 r1 > NF4 r2 > bf16 r2 > P-schedule arms.** The
matched pair is the experiment; repeats and schedule checks are riders. Every ladder decision
and its timing inputs are durably recorded before the next stage begins, exactly as P01's
extension decision was.

Technical gates: G1/G2 verbatim from P01 (module counts, sentinels, KV dtype/shape, identity/
surgery ladder) per regime; any failure ends that regime's outcome work and the run reports
apparatus evidence only.

## 7. Regime order: keep fixed NF4-first

Not randomized, not counterbalanced. With one sequence, randomization has zero inferential
yield; counterbalancing requires a second pod-pair (≈2× cost) to estimate an order effect it
still could not identify at n=1 per order. NF4-first retains the operational logic P01 validated:
the less-predictable regime sets the timing forecast before the comparator loads, and the
continuation ladder in §6 depends on observing `t1` from the risky regime first. Both regimes
read the same downloaded bf16 shards, so setup cost is order-symmetric. Same-host drift over a
~90-minute window is addressed by determinism (greedy, temperature 0), per-regime G2 ladders,
the FF anchor present in both regimes, and the G4 repeat comparison — not by order manipulation,
which cannot fix it anyway.

## 8. What each P02 result licenses in the paper — and what it does not

- **E1 reproduced in both regimes:** "The oracle-vs-fresh compaction damage reproduces in sign
  and magnitude class under true NF4 weight quantization in a matched same-host runtime." Not
  licensed: anything about 4-bit KV caches, other quantizers (GPTQ/AWQ/MLX), other cases or
  models, or the cause of the early MLX result (though it lowers the weight-precision
  explanation's posterior).
- **E1 not reproduced (NF4 damage vanishes/inverts):** a runtime-axis apparatus observation
  (per P01's own matrix, not automatically an apparatus failure if G1/G2 passed); licenses a
  methodological sentence and a follow-up design, no semantic claim.
- **E2 identical (expected: all-null):** "No graft cell changed any generated answer in either
  regime" — the behavioral null is not bf16-specific. Not licensed: general mitigation failure
  claims beyond this fixture.
- **E2 differs (an answer changes in one regime only):** a screening flag licensing exactly one
  thing — a new multi-case, control-designed preregistration. Explicitly not an efficacy result,
  not "quantization dependence."
- **E3 sign-agrees, ratio 0.5–2×:** one descriptive sentence: the small FC−FW contrast is not
  regime-specific in this fixture. No evidentiary upgrade — still N=1, still placebo-less.
- **E3 diverges by any amount:** "In this single fixture the value-only contrast differed by X
  nats between regimes." Not licensed: "the effect is precision-dependent" — indistinguishable
  from kernel-level numerical fragility given the fixture's own 8.3× schedule collapse.
- **bf16 arm fails to echo the archived Transformers-5 bf16 pattern (descriptive):** adverse
  robustness evidence about the e01 residue across Transformers versions — reportable and
  independently useful; no precision claim.
- **G1/G2 failure or timing abort:** apparatus/operations evidence only; nothing scientific;
  no relabeling of P01 or v12.

No P02 outcome touches formal v12, e01's diagnostic-only status, or general efficacy. All
packages (raw lossless, compact, render ledgers, attestations, receipts) preserved and committed
per the P01 persistence contract, pod terminated immediately after verified pull.

## 9. P01 post-mortem: what was overbuilt vs essential (outcome-blind, non-defensive)

**The central error is arithmetical and was checkable before launch.** The frozen base design
was 4 × (full 34-arm grid + Phase A) ≈ 4 × 38 min = 152 min of outcome work against a 120-minute
cap that also had to absorb ~12 min of NF4 technical work, a second load and gate ladder for
bf16, and setup. The base design and the cap were jointly infeasible under any assumption where
NF4 per-arm cost is even comparable to historical bf16 — no NF4 slowdown was needed to break it.
The elaborate outcome-blind extension rule guarded the *growth* direction (e02/e03) with real
care, while the *base* commitment had no shrink ladder and no feasibility check of the form
"4 × grid + 2 × load ≤ cap." P02's §6 ladder is that missing rule.

**Unnecessarily expensive in P01:**
- The complete 34-arm grid per execution, ×4. Roughly 22 of 34 arms (R1/R3, key-only, crossed,
  P-schedule CC/WW) had no predeclared claim slot — pure descriptive freight at ~51 s/arm,
  ~75 min total across four executions. My A3 review recommended the trimmed set before P01 was
  frozen; the preregistration chose the full grid for literal comparability with the historical
  e01 record. Comparability of *inferential cells* required matching those cells, not the whole
  grid.
- Placebo re-attempts (3 arms × 4 executions) after incident #46 had already established
  infeasibility at bf16 and the dtype argument predicted the same at NF4.
- Two repeats **of the full grid**. The determinism check was right; repeating 34 arms instead
  of the inferential subset doubled the freight too.

**Essential and worth every dollar:**
- The true-NF4 topology work: catching the Transformers-5 partial-quantization trap and
  positively verifying 18,432/18,432 expert linears. This is the single most valuable artifact
  P01 produced and permanently de-risks every future run on this axis.
- The isolated 4.57.6 environment, the G1 module/sentinel/KV gates, the per-regime G2
  identity/surgery ladder, NF4-first ordering, the durable packaging/receipt discipline, and the
  outcome-blind decision machinery *as machinery* (P02 reuses it, re-aimed).
- The launch/watchdog hardening bought by incidents #47–#48 — paid once, reused forever.

Net: P01's dollars bought a working, gated, honest NF4 apparatus and its timing model; they did
not buy the comparison. P02 buys the comparison with the apparatus P01 already paid for, at
~7/34 of the arm cost, inside a cap the observed numbers say it fits with margin.
