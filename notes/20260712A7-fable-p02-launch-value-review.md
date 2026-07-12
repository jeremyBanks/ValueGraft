# P02 launch-value review — independent adviser assessment

**Author (exact runtime identity as observed):** Claude Fable 5, model id `claude-fable-5`, per
the harness environment block of this session. The requested identity (a "Fable" launch-value
review) matches the observed runtime identity; there is no discrepancy to report.
**Date:** 2026-07-12
**Inputs read (only):** `PRECISION-PROBE-P02-PREREGISTRATION.md`,
`notes/20260712A5-fable-p02-outcome-blind-fallback-review.md`,
`notes/20260712A6-sol-p01-independent-partial-analysis.md`. The A6 machine-readable artifact was
not opened; A6's table is treated as the verified fact record per the owner's instruction. No
agents spawned, nothing browsed, no archives inspected. FINDINGS.md treated as stale for this axis.

**Standing facts taken as given:** P02 design/analysis frozen before any P01 outcome value was
accessed; apparatus has passed scientific-contract tests; three bounded operational fixes pending
(C-210 graceful interrupt with kill by C-150, verified nonempty unreceipted capture before
deletion, refreshed deadline checks); no pod active; balance ≈ $58.81; cap $3.90 scientific /
$4.00 absolute, expected ≈ $2–3.5.

---

## 1. What P02 actually adds now

Sol's A6 is correct that P02's original novelty claim is gone: P01 already delivered one
same-host matched repeat-1 pair plus an exact NF4 determinism repeat. Honest accounting of the
remaining increment:

1. **A prospectively preregistered replication of the entire matched observation.** Everything
   the paper can currently say on the precision axis rests on a *post-hoc descriptive analysis of
   an interrupted run*. P02 was frozen before P01 values were opened, so its result — whatever it
   is — carries prospective-prereg status. That is a qualitative epistemic upgrade no amount of
   P01 reanalysis can buy.
2. **A second, independent-host test of P01's single most surprising observation:** the NF4 fresh
   baseline producing a bizarre wrong answer ("Atlas 4.8…") that all four N/R2 grafts flip to the
   bf16-universal "Ring 3". P01's NF4 r1=r2 byte-identity proves within-host determinism only; it
   says nothing about whether the anomaly is host/instance-idiosyncratic. This is the highest
   value-of-information item in the run.
3. **A stability check on the descriptively conspicuous `D^KV` sign flip** (−0.346 NF4 vs +0.233
   bf16). Reproduce → a stronger (still descriptive) screening flag worth carrying into any
   multi-case prereg; fail to reproduce → correctly demoted to numerical instability near zero.
4. **Free riders:** cross-host NF4 stability (P02-NF4 vs P01-NF4), cross-version bf16 context,
   and one more bounded placebo construction attempt per §5.2.

What P02 does **not** add: a placebo control (the repeat-1 attempt will very probably fail again
— P01 failed deterministically at layer 1, rows 76/77, and nothing in P02 changes the geometry),
any population inference, any path to the phrase "quantization dependence," or any effect on
formal v12. E1 gross damage (~22–24 margin units in both regimes) will almost certainly
reproduce; its confirmation is cheap but low-surprise.

## 2. Outcome sensitivity of the paper-level conclusion

The current paper-level conclusion (from A6): no semantic recovery; very large gross compaction
damage in both regimes; source contrasts tiny relative to damage, with nonfocal exceeding focal
(anti-selectivity); an NF4-specific generic behavioral disruption on this one fixture; missing
placebo blocks semantic attribution.

**Outcomes that would materially change what the paper says:**

- **NF4 fresh anomaly does not reproduce** (P02 NF4 fresh also generates "Ring 3", or a third
  answer): the "runtime-specific behavioral disruption" observation collapses to a
  host-idiosyncratic curiosity and must be reported as unstable. This deletes or heavily
  rewrites the most interesting sentence in the precision paragraph. Genuinely decision-relevant.
- **Any E2 tuple/change-vector divergence from the P01 pattern in either regime:** the pattern
  is unstable across hosts; the paper reports a screening flag and the follow-up design changes.
- **E1 sign or magnitude-class failure:** would indicate an apparatus or host problem rather
  than science; downgrades to operations evidence but demands investigation before any paper.
- **Technical gate failure or no matched repeat-1 pair:** apparatus evidence only; the paper's
  precision paragraph then rests solely on the P01 partial, explicitly labeled post hoc.

**Outcomes that would not change it:** E1/E3 magnitude wobble within repeat-instability bands
(§6 comparison rule); the `D^KV` flip reproducing or not (one descriptive sentence either way);
placebo failing again (already the reported state); all secondary values. Full concordance
changes no content but upgrades status from "observed once, post hoc" to "observed twice,
independently, once prospectively" — which is exactly what a screening paper needs.

## 3. Does the missing placebo undermine the value enough to stop?

No. The placebo's job is to separate semantic state transfer from generic intervention effects —
it is load-bearing only when attributing a *positive* effect. P01's actual pattern is the
opposite: tiny contrasts, anti-selectivity, identical wrong answers. The endpoints P02 replicates
(E1 damage, E2 literal tuples, E3 signed contrasts) are all well-defined without a placebo, and
the prereg already predeclares `PLACEBO_UNAVAILABLE` as explicitly reported missing-control
evidence with weakened semantic attribution (§10). The one real cost: if the NF4 graft-flip
reproduces, we still cannot say whether *any* similar-norm cache perturbation would flip NF4
fresh behavior — the FF anchor is a duplicated baseline, not an executed surgery, so it does not
control for the intervention machinery. That caps the ceiling of claims, but the ceiling was
already capped by the frozen claim boundary. Not grounds to stop.

## 4. Is the prospective replication invalid now that P01 was opened?

I looked for a disqualifier and found none, on four checks:

- **Design contamination:** none possible — the freeze demonstrably preceded P01 access
  (prereg status line, §9, §11; A6 confirms packages stayed unopened until commit).
- **Implementation drift after unblinding:** the three pending fixes are operational and
  value-blind on their face (interrupt handling, capture-before-delete, deadline refresh); §11
  explicitly permits stricter/earlier-stopping changes. Condition: the diff must touch no
  analysis, scoring, arm-selection, or cell code. If it does, stop and re-review.
- **Outcome-conditioned launch:** deciding to launch partly because P01 looked surprising does
  not bias P02's own frozen endpoints. The mild meta-level selection concern is neutralized by
  the persistence contract: P02 is reported in full whatever it shows.
- **Interpretive latitude:** analysts now know P01's values, so P01-vs-P02 concordance could be
  characterized flatteringly after the fact. Cheap mitigation, recommended: before launch,
  commit a short additive dated note (permitted by §11) stating the P01-anchored expectations —
  e.g., "expected NF4 fresh tuple = [Atlas-answer, Ring 3 ×4]; expected bf16 tuple = [Ring 3
  ×5]; expected placebo failure at layer 1" — so concordance is judged against pre-stated
  expectations. The prereg itself already correctly labels P02-vs-P01 as cross-host descriptive
  context only (§8); the paper must keep that label.

One honest caveat survives: this is now a *conditional* replication (run in a world where P01
was interesting), and the paper should say so in one clause. That is a disclosure, not an
invalidity.

## 5. Recommendation: GO, with stopping rule

**GO.** Bounded downside: absolute worst case $4.00 against $58.81 (≈7% of balance); realistic
worst case ≈ $1.5–2 for an NF4-only timing abort that is still packaged apparatus/cross-host
evidence. The upside is the cheapest available upgrade of the paper's only precision-axis
paragraph from post-hoc partial description to prospectively preregistered matched replication,
aimed at the one observation (NF4 fresh anomaly + graft-flip) most plausibly host-idiosyncratic.
Both confirmation and disconfirmation change what the paper says or how strongly it can say it.
Nothing cheaper buys this, and a multi-case follow-up should be conditioned on P02's outcome
anyway.

**Preconditions (all before pod creation):**
1. The three operational fixes pass their tests; diff verified to touch no analysis/arm/scoring
   code.
2. The additive P01-anchored expectations note from §4 above is committed.

**Stopping rule (frozen §7 adopted verbatim, restated):** cap `C = min(9,000 s, $3.90-equivalent)`
provider-clock; stage launches only if its forecast fits `C − 300`; after NF4 repeat 1, proceed
to bf16 only if `e + 1.25(Lhat + t) ≤ C − 300`, else **stop and package before loading bf16** —
an NF4-only P02 duplicates P01 and buys nothing; repeat-2 riders only under the frozen double-t
check; graceful interrupt at `C − 180` (C-210, kill by C-150); forced pull/DELETE by `C − 60`
with verified nonempty capture before deletion; no value ever authorizes more work; any
settlement overage recorded, never retried. **Operational addendum for Sol's discretion:** if
the run dies before any scientific stage begins with spend under ~$0.50 (e.g., allocation
failure), one relaunch is reasonable; any later failure means stop, package what exists, and
reassess — no same-day P03, no new spend without a new frozen protocol.

Sol retains final authority; this note is advisory.
