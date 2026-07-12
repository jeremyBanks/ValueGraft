# Fable review — powered coherent-state successor design

**Reviewer:** independent scientific-methods review (Fable), 2026-07-12.
**Scope:** design of the powered successor and allocation of the remaining $57.13 only. Apparatus history and prior prose not re-litigated. Evidence base: the owner's bundle (e01 result, v12 capabilities, cost observations, fixture inventory).

## 0. Framing: two claims plus an honesty valve

Build the successor to support exactly two population-style claims and one pre-registered escape hatch:

- **Claim D (damage):** "Compaction of histories drawn from recipe R destroys at least X nats at the focal probe" — one-sided 95% **lower** confidence bound on mean damage. Note that the famous 22–24 nats is currently N=1; this claim is not yet established and comes nearly free in the same runs.
- **Claim T (transplant futility / equivalence):** "Full-KV transplant of correct-history cache at the compacted locus restores at most δ of that damage" — one-sided 95% **upper** confidence bound on mean recovery fraction.
- **Heterogeneity valve (co-primary):** a pre-registered rule under which the headline becomes "not equivalent — a subpopulation of fixtures recovers substantially," reported instead of the mean bound. A design that can only output a mean UCB assumes the null structurally; this valve is what makes the design able to conclude either way.

All inference is scoped to the **fixture-generation recipe R**, not to conversations in the wild. Engineered fixtures are a designed benchmark; the statistics quantify sampling variability under R, nothing more. Say this in every artifact. **e01 is outcome-seen and iterated; it is excluded from N** and reported separately as the discovery case.

## 1. Estimand, unit, controls, decision rule (Q1)

**Sampling unit.** One base conversation per fixture; unit = fixture = conversation. No two fixtures may share a base conversation (if any do, cluster at the base conversation). Arms, schedules, and execution replicates are nested measurements, never units. Each fixture is collapsed to one number per cell (mean over its execution replicates) **before** any interval is computed. At N≤24, collapse-then-t is more trustworthy than cluster-robust sandwiches or cluster bootstrap, and it satisfies the owner's clustered-uncertainty requirement by construction. This makes the P01/P02 pseudo-replication mistake structurally impossible.

**Per-fixture outcomes.**
- D_i = damage: nats lost at the focal probe, compacted vs intact, mean of 2 executions.
- Δ_i,c = recovery in cell c: damage reduction under intervention c, mean of 2 executions. Signed; e01 showed sign instability, so do not truncate at zero.
- ρ_i,c = Δ_i,c / D_i, the recovery fraction — primary scale, with absolute nats co-reported.

**Inclusion gate (pre-registered, treatment-blind).** ρ is defined only where D_i ≥ 5.0 nats. Justification: with margin δ = 5%, the smallest effect that must be resolved is 0.25 nats ≈ 4× the 0.06-nat schedule floor. The gate uses only no-intervention arms, so it cannot select on treatment response. Sub-floor fixtures still feed Claim D's distribution and are reported.

**Primary cell family (K = 3).** Full-KV × correct-source × focal probe × q=1 schedule, at each of {carrier, boundary, anchor}. Everything else — value-only, wrong-source, nonfocal, P-schedule — is secondary or control. The headline is the **max of three simultaneous Bonferroni UCBs**: "even the most favorable pre-specified transplant restores ≤ …". Without a pre-registered family, the 3×2×2×3 grid guarantees a forking-paths disaster.

**Margin and tightness cap.** δ = 0.05 (5% of per-fixture damage). Do not promise or attempt bounds below ~1%: the 0.06-nat floor plus bf16 representability make sub-1% recovery fractions numerically uninterpretable. State this cap in the pre-registration so nobody chases it later.

**Controls feasible in bf16.** The old placebo failed because sparse additive deltas fell below bf16 representability. The fix is categorical: **placebos must move whole cache entries, never add deltas** — exchanges of existing bf16 values are exactly representable in any float format by construction.
1. **Identity placebo:** re-insert bit-identical KV from an independent execution of the same history. Directly estimates the pipeline noise floor; expected ≈ 0.
2. **Shuffled placebo:** correct-source KV with token positions permuted within the region (fixed per-fixture seed). Identical value distribution, destroyed content — the mechanical-size-matched placebo the old construction failed to be.
3. **Wrong-source arm** (already in apparatus): semantic-specificity control; correct − wrong is the secondary specificity contrast.
4. **Nonfocal probe:** negative outcome control.

Placebo gates: identity within ±0.18 nats (3× floor); shuffled placebo not systematically matching correct-source (if shuffle ≈ correct, any movement is content-free).

**Decision rule and sequential looks.** One-sided α = 0.05 per claim, spent over at most 3 looks at N = 12, 18, 24 evaluable fixtures, front-loaded conservative: α₁ = 0.005, α₂ = 0.010, α₃ = 0.035. At look k, per-cell UCB = ρ̄_c + t_{N−1, 1−α_k/3} · s_c/√N. Looks happen at pod-session boundaries so "run until CI converges" is implemented as pre-committed alpha spending, not peeking.
- **Early stop for equivalence:** all three UCBs < δ/2 (stricter early bar), heterogeneity valve silent, placebo gates pass. Remaining fixture budget flows to exploration.
- **Switch branch (effect is real):** any primary cell with lower bound > 0 and point estimate > δ → abandon equivalence, switch to characterizing the effect (locus, layer, dose). This is a success outcome, not a failure.
- **Otherwise continue; hard cap N = 24.** Report spending-adjusted simultaneous UCBs at whatever N stopped, plus Claim D's LCB (same machinery, one-sided lower, its own α = 0.05).

**Heterogeneity valve (own α = 0.05).** Trips if (i) between-fixture variance in ρ significantly exceeds within-fixture replicate variance (one-way random-effects comparison), or (ii) any single fixture's ρ CI (Bonferroni α/N) lies wholly above δ. Either → headline is the fixture-level distribution and the identity of recovering fixtures, and exploration budget is redirected to them.

**Secondary behavioral endpoint.** Greedy-answer flips (wrong → correct target) as a binary per fixture: with 0 flips in 24, the exact Clopper–Pearson 95% UCB on flip probability is ≈ 11.7% — a concrete, honest bound on "never rescues the answer."

**Min/max N.** Min = 12; below that a 3-cell simultaneous UCB against δ/2 is unreachable unless s is implausibly small. Max = 24 (budget and fixture supply). Sanity check at N=12 under e01-like behavior (ρ̄ ≈ 0.5%, s ≈ 1%): UCB ≈ 0.5% + 3.6·(1%/√12) ≈ 1.5% < 2.5% → early stop plausible. If s ≈ 5%, no early stop — the design correctly forces N=24 and likely trips the valve.

**Sensitivity analyses (no alpha):** anytime-valid empirical-Bernstein bound on ρ winsorized to [−0.25, 1]; absolute-nat versions; leave-one-fixture-out.

## 2. Fixture supply: e02–e06, e07+, c-bank (Q2)

- **Use e02–e06 in the primary N**, under one constraint: validation (§3) touches only their no-intervention and placebo arms, so they remain treatment-outcome-unseen and inclusion is clean. Their canary origin makes them plausibly less diverse than fresh fixtures — record them as a stratum and check them in leave-one-out.
- **Author e07–e26 (20 new fixtures)** under a written recipe R frozen before any treatment run: stratified over topic domain, answer type (entity / number / yes-no), distractor structure, and context length within apparatus limits; mechanical validity gates (render hashes, decoded-review coherence, paired-counterfactual minimality) applied pre-treatment; expect ~10–15% attrition → ~23–24 usable including e02–e06. Authoring costs CPU/agent time, ~zero GPU dollars.
- **Do not pool c01–c24 into the primary.** Incomplete provenance means R is undefined for them, and salvaging clipped/incoherent decodes forces selection-by-inspection — selecting on exactly the text properties that could drive damage and recovery. At most, keep the best-provenance 4–6 as a clearly labeled exploratory robustness set.
- **No iteration on outcomes:** a fixture failing gates is replaced by the next pre-authored one in order; no fixture is edited after any treatment result exists anywhere.

## 3. Smallest pre-scale validation, spiral-proofed (Q3)

v12 is already validated. Only the successor's deltas need checking: batched multi-fixture execution behind one model load, the two new placebo arms, and the new recipe. One session, two fixtures (e02, e03), pre-committed gates:

- **V-1** hashes/checkpoints match v12 expectations.
- **V-2** batched pathway reproduces the single-fixture pathway on e02 within 0.06 nats per arm.
- **V-3** identity placebo within ±0.18 nats on both fixtures.
- **V-4** shuffled placebo executes with no representability collapse (deltas actually land).
- **V-5** D ≥ 5 nats on ≥1 of 2. If both fail, the problem is fixture supply/recipe, not apparatus — reassess authorship, do not touch code.
- **V-6** within-fixture replicate SD ≤ 0.10 nats.

**Anti-spiral rule:** at most one fix iteration, then one re-run of the same gates. Two consecutive failures → stop and write up; no further GPU. Gates may not be renegotiated after seeing results. Later sessions re-check only V-1 (hash drift) at negligible cost. Validation budget ≈ $3.

## 4. Allocation of $57.13 (Q4)

Observed cost model: provisioning $0.81 + treatment allocation $0.43 + model prep ≈ $0.30 → **~$1.55 per pod session**. e01's full grid ran 421 s ≈ $0.16 of compute; the successor's per-fixture battery (damage ×2, three primary cells ×2, identity ×2, shuffle, wrong-source, nonfocal ≈ 13 arm-executions) is the same order → plan **$0.35/fixture** with a 2× safety factor already inside it.

| Line | Content | Budget |
|---|---|---|
| Validation session | §3 gates on e02–e03 | $3.00 |
| Session 1 (look 1) | to N=12 evaluable, full battery | $7.00 |
| Session 2 (looks 2–3) | to N=24 | $8.00 |
| Host/render variance | 4 fixtures × primary cells on a second fresh pod + 2 render replicates | $4.00 |
| Exploration portfolio (§5) | capped probes 1–5 | $24.00 |
| Reserve | pod failures, reruns; untouched until the end | $11.13 |
| **Total** | | **$57.13** |

Early stop at N=12 moves Session 2's ~$8 to exploration/reserve; the switch/heterogeneity branch instead redirects exploration money into characterization. The host-variance line is not optional: with everything on one pod, session is confounded with everything (driver, kernel selection, nondeterminism). Gate: cross-host per-cell differences ≤ 2× within-host replicate SD; render replicates hash-identical or within the floor; otherwise report host variance as a component and widen intervals honestly. Never plan the reserve to zero.

## 5. Exploration menu, ranked (Q5)

Each probe hard-capped; every probe writes a results note even if boring; the ranking is revised once, in writing, after look 1 — not continuously.

1. **Downstream / retained-tail aggregator transplant** — $8, 6–8 fixtures, 2–3 cells. The closest prior predicts the information has already migrated downstream of the compacted span, in which case a null at the compacted locus is exactly what it predicts and Goal A's bound is compatible with full recoverability elsewhere. Highest information value per dollar; run first.
2. **Layer/depth (alpha) targeting** — $5, 3–4 fixtures. Whether movement concentrates by depth is mechanistically informative in both directions and cheap.
3. **MEMENTO-style restart contrast** — $5, 4 fixtures. Anchors Claim D against the natural engineering alternative (restart with summary) and makes the damage number decision-relevant.
4. **fp32 / short-context floor verification** — $3, 1–2 fixtures. Confirms the 0.06-nat floor is numerical and bounds what bf16 hides. (fp32 activations for 30B may not fit 80GB — use shortest contexts or fp32-KV-only.)
5. **True low-precision KV cache** — $3. Deployment-relevant; tests precision-artifact explanations from the other side.
6. **Second model family — defer.** Highest cost per bit until the within-model story is settled; only from surplus/reserve after 1–4.

## 6. Fatal errors in "just run 24 cases and bootstrap" (Q6)

1. **Wrong resampling unit.** Bootstrapping fixture×cell×replicate rows fakes N in the hundreds when there are 24 clusters — the P01/P02 mistake with extra steps.
2. **Bootstrap undercoverage at N=24.** Percentile bootstrap on a skewed small-sample mean undercovers; for an equivalence UCB, undercoverage manufactures false "bounded" claims — precisely the artifact a skeptic will hunt for.
3. **No noise-floor calibration.** Without identity placebos, 0.3% "recovery" and pipeline noise are indistinguishable; bootstrap quantifies variance, not the bias-like 0.06-nat floor.
4. **Optional stopping.** "Run until the CI looks tight" with naive intervals inflates error without bound; looks must be pre-committed with spent alpha.
5. **Unmanaged multiplicity.** A ≥12-cell grid guarantees some cell "moves" or some bound looks tight by chance; no pre-registered primary family means no headline claim.
6. **Ratio blow-ups and forking paths.** Small-D_i fixtures explode ρ; post-hoc exclusion is unprincipled unless the damage-floor gate is pre-registered and treatment-blind.
7. **Population overclaim.** 24 engineered fixtures sample no natural population; unscoped bootstrap CIs quantify sampling from a population that was never sampled. Scope to recipe R or the claim is unsound.
8. **Session confound.** One pod, one load, one host → host/driver variance completely unbounded; the replication requirement exists for a reason.
9. **Averaged-away heterogeneity.** Three strongly-recovering fixtures out of 24 can coexist with a mean UCB < δ; without the valve, the most interesting possible outcome is suppressed by design. (Also: e01 must not be one of the 24 — it is outcome-seen and was iterated.)

## 7. Pre-registration freeze list (before the first treatment arm)

Recipe R text; hashes for e02–e26; gates V-1…V-6; D-floor = 5 nats; δ = 0.05; the 3-cell primary family; look schedule and alpha spend (0.005/0.010/0.035 at 12/18/24); heterogeneity valve; placebo gates; host-variance gate; flip-count endpoint; budget table above; unique successor name (not v12, not bound by v12 stop rules, all prior renders/results preserved untouched); analysis code committed with fixed seeds.

**Bottom line.** The budget is sufficient for a genuinely powered design with money left over: ~$18 buys Claim D's lower bound and Claim T's simultaneous 95% UCB at N up to 24 with real placebos, sequential looks, and cross-host replication; ~$24 buys a ranked exploration portfolio led by the one hypothesis (downstream aggregation) that could overturn the interpretation; $11 stays in reserve. The design can honestly end in "transplant restores ≤ ~1.5–3% of damage under recipe R," "a subpopulation recovers — here it is," or "the information lands downstream" — and each of those is a publishable sentence.
