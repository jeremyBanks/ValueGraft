**Claude Fable 5 — claude-fable-5:**

# Independent interpretation of the v12 e01 post-ambiguity diagnostic

**Scope.** Narrow review of one completed exploratory diagnostic. Files read: the pre-treatment
spec-conflict disposition (`results/coherent_canary_v12_budget/..._path_control_spec_conflict_disposition_20260712T0334Z.md`),
the e01 lifecycle/cost record (`..._e01_treatment_lifecycle_and_cost_20260712T0419Z.md`), the full
harvest (`results/coherent_canary_v12_harvest/coherent-canary-v12-harvest-e01-exact-subject-20260712T040158Z.json`),
the post-run audit (`results/coherent_canary_v12_postrun_audit/coherent-canary-v12-postrun-audit-e01-exact-subject-20260712T0418Z.json`),
and frozen preregistration §§7–12, 15–16, 20. Governing facts I hold fixed: formal v12 is terminally
stopped under the written ULP path-control rule (fail at ULP 2); e01 is diagnostic-only and cannot
authorize e02+, any aggregate, confirmation, or live agents; natural calibration was adverse; artifact
integrity and cross-host fresh-score equality passed; all primary greedy answers stayed wrong; all
three placebo arms were `PLACEBO_UNAVAILABLE`.

**Measurement trust.** I treat the numbers as real measurements: treatment-fresh and Phase-A canonical
hashes are byte-identical for both probes, pod/local normalized harvests hash-identical, raw
reconstruction byte-exact, FF cells identical across regions and equal to `fresh_scores` (an internal
consistency check the data passes). The question is purely interpretive.

## 1. The primary R2/N contrast: two families, two different stories

**Full-KV (CC vs WW) is a disruption signature, not transfer.** D_focal = +0.097, but D_nonfocal =
+0.245 (the untargeted probe moved 2.5× more, so SEL = −0.148) and Hplus = −0.131: the correct
target's log probability got *worse* under the correct-history state. The margin gain comes entirely
from the counterfactual target dropping 0.228 nats more than the correct one. §11's rule — never call
a larger margin "less loss" unless the correct-target LP also improves — is exactly the trap this cell
would fall into. Under the preregistered reading, full-KV at R2/N shows nothing semantic.

**Value-only (FC vs FW) shows the complete preregistered signature, and it is the only cell that
does.** D_focal = +0.175, SEL = +0.175 (D_nonfocal ≈ −7×10⁻⁵ — the nonfocal probe is essentially
untouched, margins matching to four decimals), Hplus = +0.710, U = +0.182, Uplus = +0.946. All five
directional criteria of §12 are satisfied simultaneously.

**But the decomposition undercuts the strong reading.** From the raw cells: under correct-history
values the focal *counterfactual* target also rose (+0.536 nats, FC vs FW), not just the correct
target (+0.710). The genuinely differential semantic component is the 0.17-nat margin gap; the
larger shared movement looks like generic redistribution of probability toward the focal answer
space. Worse, the wrong-history value arm also beats fresh on the correct target: FW − FF = +0.236
nats. So "importing *any* history-written values" produces roughly a quarter of the celebrated
+0.946 Uplus without any correct semantics in it. A nonsemantic component of the same order as the
effect is directly visible inside the data.

Note the estimands are not five independent confirmations: SEL derives from D_focal and D_nonfocal;
Hplus/Uplus and D/U share components. There are effectively ~2–3 semi-independent observations here.

## 2. N versus P: the effect does not survive its own schedule yardstick per-case

D_focal shrinks from +0.097 (N) to +0.039 (P) for full-KV and from +0.175 (N) to +0.021 (P) for
value-only — a 3–8× collapse when the imported history is written by turn-aligned prefill instead of
q=1 replay. Both families record `per_case_N_at_least_three_times_difference: false`: the N-vs-P
difference is the same order as the effect itself. Two readings exist and e01 cannot distinguish
them: (a) the mechanistically interesting one — token-by-token generation-time writing preserves
history-dependent state that chunked prefill loses, which is the project's core hypothesis; (b) the
deflationary one — at 0.02–0.2-nat scale, execution-path numerics (kernel/batching differences
between call patterns) contribute at the same order as the "signal." The one durable crumb for (a)
is that value-only Hplus stays positive under both schedules (+0.71 N, +0.23 P) — direction is
schedule-robust even though magnitude is not. The 3× rule is a frozen exploratory heuristic (§8),
and per §16 this is one component, not a decision; but as evidence it says the effect is not
schedule-stable at this case's scale.

## 3. R1/R3 localization: regional incoherence, not a clean mechanism map

Value-only is positive only at R2 (boundary: carrier content + close + anchor prompt/header); at R1
(content) every value-only estimand is *negative* (D_focal −0.126, Hplus −0.274), and at R3 focal
effects are null (D_focal −0.005). Meanwhile full-KV flips character by region: positive-selective at
R1 (D_focal +0.154, SEL +0.132, Hplus +0.023) but disruption-shaped at R2. A real value-borne channel
carrying evicted-history semantics has no obvious reason to appear in values-at-R2 but reverse in
values-at-R1 while full-KV does the opposite. At n = 1 this pattern is more consistent with
case-idiosyncratic perturbation geometry than with a coherent localization story. I would not draw
any mechanism map from it.

## 4. Scale against Phase-A damage and behavior: recovery is negligible and behaviorally invisible

Phase-A damage on the focal probe is 22.29 nats of correct-target LP and 22.30 nats of margin (the
full-history oracle was near-certain, ≈ −6×10⁻⁶ mean LP; the fresh compact state sits at −22.29).
The best treatment cell (FC) recovers +0.946 nats of correct-target LP — **4.2% of the damage** —
and +0.182 nats of margin — **0.8%**. Every focal margin in every cell remains between −5.6 and
−6.3: the model prefers the wrong answer by ~250:1 in log-space everywhere, all greedy answers stayed
wrong, and the nonfocal probe also prefers its counterfactual in all cells. The adverse natural
calibration matters here: in a regime this saturated, sub-nat LP movements are exactly where
nonspecific perturbation effects live, and nothing behavioral can corroborate them.

## 5. The missing placebos: structurally informative, interpretively crippling

All three placebo arms failed identically at layer 1, row 76: pre-cast constraints were satisfied to
~1e-16, but every one of 1024 deterministic attempts produced an *applied* bf16 delta with relative
L2 error ~0.09–0.10 against the 0.05 cap (the target ‖d‖ was ~0.002 — at bf16 granularity relative
to the fresh values, the equal-norm orthogonal vector is unrepresentable). Two consequences:

- **Informative:** of the scanned early rows, 42/43, 67/68, 70/71 had *exactly zero* C-vs-W value
  difference, and the first nonzero row's delta sat at quantization scale. The C/W state difference
  is extremely sparse and tiny in early layers — expected mechanistically (identical visible tokens;
  history enters only via attention), and it means the placebo construction failed for a structural
  reason, not an implementation fluke. Per §10 this is adverse control availability, and the frozen
  spec correctly does not let it invalidate the primary arms.
- **Crippling:** the placebo was the *only* control separating "correct-history semantic content in
  the values" from "any perturbation of this norm profile at these rows moves the focal probe."
  SEL controls generic disruption via the nonfocal probe, but a perturbation localized to rows the
  focal probe attends would be focal-selective without carrying semantics. With placebo unavailable,
  that explanation is uncontrolled — and the observed FW − FF = +0.24 actively feeds it.

## Verdict

**(b) — a real but tiny mechanistic hint, and only barely above (a).** It is not meaningless noise:
the value-only R2/N cell satisfies all preregistered directional criteria simultaneously with genuine
focal selectivity (nonfocal margins identical to 4 decimals), the direction of Hplus survives the
schedule change, the family that shows it (values carry it, keys add nothing but disruption) is the
same family the earlier v10 work independently identified, and the measurement chain is verified.
But it is one engineered case, adversely calibrated, recovering ≤4% of the damage with zero
behavioral effect, failing its own per-case schedule yardstick, with its decisive control absent and
a visible nonsemantic component (FW beating FF; the counterfactual target rising almost as much as
the correct one) of the same order as the differential effect.

**Strongest alternative explanation:** nonspecific, near-bf16-scale perturbation of R2 value rows in
a saturated wrong-preference regime, amplified by schedule-dependent execution numerics — any
sufficiently structured perturbation of the rows the focal probe attends shifts sub-nat probability
mass toward the focal answer space (both targets), and on this single case the differential
0.17-nat residue happened to land in the correct direction. Nothing in e01 excludes this; the
missing placebo is precisely the control that would have.

## Recommended next step (not a continuation of frozen v12)

Formal v12 stays terminal; nothing below treats e01 as permission. In order of value per dollar:

1. **Zero-GPU re-analysis of persisted e01 artifacts** (source rows and all scores are hashed and on
   disk): per-token decomposition of the +0.71 focal gain across the two target tokens; the
   FC/FW/FF three-way decomposition for both targets and both probes; a quantitative writeup of the
   placebo bf16-infeasibility (distribution of ‖d‖ across all rows/layers, not just the scanned
   prefix). Costs nothing and sharpens every claim above.
2. **Repair the placebo construction offline** before any new run: exempt rows whose ‖d‖ is below a
   bf16-representability floor (treat as zero-delta), or construct the perturbation directly on the
   bf16 grid, or match an aggregate rather than per-row norm profile. Validate against the persisted
   e01 rows; no GPU needed for design.
3. **Only then, optionally:** a small, newly preregistered diagnostic (per the Sol disposition's own
   contemplated path) testing *only* the value-only R2 contrast — FC vs FW vs FF vs repaired
   placebo — on a few new engineered cases with **non-adverse calibration** (correct target not ~22
   nats underwater, so a 0.5–1-nat movement could register behaviorally), under both N and P.
   Decision logic: placebo ≈ 0 with replicated FC − FW > 0 upgrades the hint; placebo comparable to
   FC − FW kills it. Descriptive sign-count reporting only.

Stopping entirely and reporting e01 as bounded descriptive evidence is also defensible; this result
does not compel further spend, and the $0.67 remaining under the frozen ceiling is not a reason.

## Paper-safe wording

Exact paragraph (safe to use verbatim):

> In the single post-ambiguity diagnostic case e01 — run after formal v12 was terminally stopped
> under the written path-control rule, and incapable of reopening it — the value-only R2/N contrast
> moved in the preregistered semantic direction on every estimand (D_focal = +0.17, SEL = +0.17,
> Hplus = +0.71, U = +0.18, Uplus = +0.95 nats) with near-zero nonfocal movement, while the full-KV
> contrast improved its margin only through larger counterfactual-target suppression (Hplus = −0.13,
> SEL = −0.15) and therefore does not qualify as reduced loss under the preregistered reading. These
> movements recover at most 4% of the ~22.3-nat Phase-A compaction damage, changed no greedy answer,
> shrank 3–8-fold under the P schedule (failing the per-case component of the 3× yardstick), and
> have no placebo calibration because all three preregistered placebo arms were unavailable at bf16
> granularity; the wrong-history value arm itself beat fresh on the correct target (+0.24 nats),
> demonstrating a nonsemantic component of the same order as the differential effect. We report this
> as a weak, single-case, uncontrolled directional hint — consistent with, but not evidence for, a
> value-borne semantic channel — that authorizes nothing under the frozen decision rules.

One-sentence version:

> A single diagnostic case showed a small, internally consistent, value-only directional effect
> (≤0.95 nats against 22.3 nats of damage, no behavioral change, no available placebo control) that
> we report as an uncontrolled mechanistic hint, not evidence of a semantic channel, with formal v12
> remaining terminally stopped.
