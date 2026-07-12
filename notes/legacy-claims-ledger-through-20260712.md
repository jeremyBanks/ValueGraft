# ARCHIVE — legacy claims ledger through the pre-ultra handoff

This complete ledger is preserved for audit history. It is not the source for
the powered successor or its future paper. A new claims ledger will be created
only after successor outcomes exist.

# CLAIMS.md — audited claims ledger for the ValueGraft paper

> ## ⚠ SUPERSEDED AS FINAL-PAPER SOURCE (2026-07-12)
>
> This ledger is an exact historical audit through commit `f4b6a89` on
> 2026-07-09. It is no longer the single source of truth for the final paper.
> Later apparatus review and the coherent-state v12 program materially narrowed
> what several rows can support. Preserve every row below as history, but draft
> from the current `FINDINGS.md`/`STATE.md` plus:
>
> - `notes/2026071115-sol-final-paper-review.md` — the published README's
>   blocking provenance/apparatus audit;
> - `notes/2026071144-opus-paper-spine-and-methodological-postmortem.md` — the
>   pre-canary narrative synthesis;
> - `notes/2026071287-sol-e01-diagnostic-final-interpretation.md` — the final
>   exact-stack diagnostic interpretation; and
> - `notes/2026071288-sol-data-collection-stop-and-future-reentry.md` — the
>   terminal collection decision.
>
> In particular: the held-out synthetic bodies were foreign-rendered; the old
> state was reconstructed by prefill; the intervention covered summary plus
> tail; three compression-sweep cells used mismatched reconstruction requests;
> the selected SWE map has only a narrow likelihood-proxy result without its
> matched placebo or out-of-sample behavior; and formal v12 stopped at its
> written path-control gate. One later e01 diagnostic produced only a weak,
> schedule-sensitive, placebo-uncontrolled value-only hint with no behavioral
> recovery. None of those corrections is represented in the historical status
> counts below.

**At the time of this snapshot, this file was the single source of truth for every number the then-current paper could ship.**
The paper is assembled *from* this ledger, not the other way around. Every claim
below was **RECOMPUTED FROM DISK** at commit `f4b6a89` on 2026-07-09 — not copied
from prose. Where a recompute disagrees with a stated claim, BOTH numbers are
recorded with the likely reason. Do not cite a number in the paper unless it has
a row here with status ✅, or unless it ships with the ⚠/❌ caveat recorded here.

- **Repo / commit:** `/Users/jeb/experimentation` @ `f4b6a89` (`git rev-parse HEAD`
  = `f4b6a89f304bc97eb93f81e99b149b16325e24a4`; working tree clean at audit time).
- **Result files under `results/` are committed** (Results-in-repo rule) unless a
  row says otherwise. Analysis scripts are CPU-only and re-runnable.
- **Rounding tolerance for ✅:** ±0.1pp / ±0.001 nat, or bootstrap-seed jitter in a
  CI's last digit.

---

## STATUS SUMMARY

| Status | Meaning | Count |
|---|---|---|
| ✅ VERIFIED | recompute matches the stated claim within rounding | 16 |
| ⚠ RECONCILE | recompute differs — both numbers + reason recorded | 5 |
| ❌ UNSUPPORTED / does-not-reproduce | cannot be reproduced from disk, OR reproduced and FAILED | 2 |
| ⏳ PENDING | — | 0 |

### ❌ UNSUPPORTED / FAILED-REPRODUCTION — read first (2)
- **REC-5: judged sense +12pp is RENDER-FRAGILE — the positive control FAILED.** Under a clean
  current-code render + one consistent judge, sense collapses +8.7→+1.0 (null). Together with the
  render-fragile referent, **neither positive headline reproduces.** The paper cannot be led by a
  stable recovery claim. THIS IS THE SESSION'S DECISIVE RESULT. (Row REC-5, commit f516bb9.)
- **F2 "H-pack most accurate on evicted facts, 38/48 = 79%".** H-pack recalls
  **0/24** evicted facts on disk (30B) and **0/24** at 4B. No file anywhere in
  `results/` reproduces 38/48. **Do not ship the accuracy phrasing.** (Row F2-2.)

### ⚠ RECONCILE — must not be missed (5)
- **F2-1 decoy fabrication 83%/17% → recomputes 79%/12%** (same direction, numbers off).
- **F2-3 honesty decomposition (layout 83→25, KV 25→17, admissions 3/24→18/24) →
  recomputes 79→42→12, decoy admissions 5/24→21/24.** Stale sub-numbers.
- **REC-2 referent-logprob anchor +0.10/+0.125 → first BANKED native render +0.012**
  (render-fragile; documented in DRAFT §4.5 — this is expected, not new).
- **REC-4 F1 prose per-arm table (stance +2pp, referent +10pp) → canonical bootstrap
  +3.3pp / +9.7pp.** Paper already uses the bootstrap numbers; retire the prose table.
- **PROV/LME LongMemEval damage 52.5%→4.1% (n=320) → only recomputable verdicts file
  gives 80.6%→5.6% (n=36).** n=320 figure cited to DECISIONS prose; source JSON not
  located this pass.

### ⏳ PENDING (0)
- (Held-out c13–c24 judged test is now MOOT: the c01–c12 baseline positive control already
  failed under clean code — REC-5. Held-out generation still finishing to complete the render set.)

---

## SECTION 1 — RECOVERY / JUDGED METRIC (Qwen3-30B-A3B, meaning-judged)

### REC-1 · Judged meaning-recovery, conv-clustered bootstrap ✅ VERIFIED
- **CLAIM:** sense **+12.0pp** [+2.2, +22.9] (excludes 0); referent **+9.7pp**
  [−6.9, +26.2] (spans 0); stance **+3.3pp** [−5.4, +12.0] (null).
- **RECOMPUTED:** sense +12.0 [+2.2, +22.9] ✅ · referent +9.7 [−6.9, +26.2] ✅ ·
  stance +3.3 [−5.4, +12.0] ✅ (exact).
- **SOURCE:** `results/judge_semantic/verdicts_*.json` (graft arms E-post-a0.25,
  a1.0) + `results/judge_semantic_base/verdicts_*.json` (arms A, B).
- **COMMAND:** `python3 scripts/judged_bootstrap.py --nboot 20000 --seed 0`
- **COMMIT:** f4b6a89 (source files committed).
- **NOTES (must travel):** **4-bit MLX**, not bf16. **BRIEF summary condition**
  (`SUMMARY_REQUEST_BRIEF`, terse 3–5 sentence summary designed to starve the text
  channel and handicap the Compacted baseline — the condition *most favorable* to a
  graft effect). **c01–c12 only**, 12 convs / 64 plants (sense 23, referent 18,
  stance 23). Pooled graft = mean over doses {0.25, 1.0}. Held-out c13–c24 = ⏳ PENDING.
  Significance flips by metric (see REC-3).

### REC-1b · Judged per-arm rates (Original / Compacted / Graft) ✅ VERIFIED (with ⚠ REC-4)
- **CLAIM (FACTS/DRAFT §3.2, F1 table):** stance 96/93/96, sense ~100*/46/58,
  referent ~100*/17/26.
- **RECOMPUTED (pooled graft):** stance A=96%(n13)/B=93%(n23)/Graft=97%(n46);
  sense A=100%(n1)/B=46%(n23)/Graft=58%(n46); referent A=100%(n2)/B=17%(n18)/Graft=26%(n36).
- **COMMAND:** inline python over the same two verdicts globs (scores
  RECOVERED/PARTIAL/MISSED = 1/0.5/0; pooled graft = E-post-a0.25 + a1.0).
- **NOTES:** A-ceiling n is tiny (sense n=1, referent n=2) — ceilings indicative, not
  precise (widen A cells before quoting exact ceilings). B/E rates exact. Graft-minus-B
  from these = stance +3.3 / sense +12.0 / referent +9.7 (matches REC-1, **not** the F1
  prose table's +2/+10 — see REC-4).

### REC-2 · Referent logprob primary (raw E−B, bootstrap over probes) ✅ VERIFIED / ⚠ RECONCILE
- **CLAIM:** referent **+0.125** CI [+0.030, +0.218], 71% helped (significant);
  sense +0.047 CI [−0.038, +0.131] (spans 0); stance +0.002 CI [−0.036, +0.049] (null).
  Range across runs: referent +0.125..+0.156, sense +0.047..+0.062, stance −0.026..+0.002.
- **RECOMPUTED (`gap_closure_cat_live`):** referent +0.1246 CI[+0.026,+0.219] 71% ✅ ·
  sense +0.0467 CI[−0.037,+0.130] 59% ✅ · stance +0.0024 CI[−0.037,+0.048] 54% ✅.
  (`gap_closure_cat` original: referent +0.156 81%, sense +0.062 64%, stance −0.026 38% ✅.)
  `live` == `live2` exact (deterministic within-env).
- **SOURCE:** `results/gap_closure_cat_live/c*.json` (per-plant lp_A/lp_B/lp_E),
  also `results/gap_closure_cat/`, `results/gap_closure_cat_live2/`.
- **COMMAND:** for each plant compute `lp_E − lp_B` by category, mean + %(>0) +
  percentile bootstrap (n=10000) over plants. (Small CI-last-digit jitter is seed-dependent.)
- **⚠ RECONCILE:** this +0.125 is on the **pre-nativeness-fix pre-rendered** corpus.
  The **first BANKED per-model-native referent render is +0.012** CI[−0.068,+0.090]
  (spans 0 — see REC-3/BLK-1). The +0.10/+0.125 anchor is **render-fragile** (unbanked
  MoE draw, incident #38); DRAFT §4.5 already frames it honestly. Do **not** present
  +0.125 as the robust banked referent effect.
- **NOTES:** raw E−B is the locked estimator (mean-of-ratio retired). n≈21–24/cat.

### REC-3 · Significance flips by metric ✅ VERIFIED
- **CLAIM:** logprob metric = referent significant / sense underpowered; judged metric
  = sense significant / referent not. Each dissociation arm significant on ONE metric.
- **RECOMPUTED:** logprob referent +0.125 CI excl 0, sense CI spans 0 (REC-2); judged
  sense +12.0 CI excl 0, referent +9.7 CI spans 0 (REC-1). Confirmed.
- **NOTES:** must be stated honestly — NOT "both instruments agree."

### REC-4 · F1 prose per-arm deltas ⚠ RECONCILE
- **CLAIM (FINDINGS F1 table prose):** graft − Compacted = stance **+2pp**, sense
  +12pp, referent **+10pp**.
- **RECOMPUTED (canonical pooled bootstrap):** stance **+3.3pp**, sense +12.0pp,
  referent **+9.7pp**.
- **REASON:** the F1 prose table appears to use a single dose / different rounding;
  the canonical `judged_bootstrap.py` pools {0.25,1.0}. **Use the bootstrap numbers**
  (+3.3 / +12.0 / +9.7); retire the "+2 / +10" prose values.

### REC-5 · Judged sense reproduction under clean render ❌ DOES NOT REPRODUCE (render-fragile) — commit f516bb9
- **CLAIM tested:** does judged sense +12pp reproduce? (The DRAFT's central spine.)
- **RESULT (2026-07-09):** NO. Regenerated c01–c12 under a clean current-code BRIEF render
  (`results/raw_brief_repro/`) and judged with ONE consistent Sonnet judge across renders:
  - sense: original render **+8.7 [0.0,17.7]** → clean re-render **+1.0 [−5.2,+7.3]** (null)
  - referent: +9.7 [−7.9,28.3] → +5.2 [−7.3,14.6] (both span 0); stance judge-unstable.
  The +12.0→+8.7 gap is judge calibration; the **+8.7→+1.0 collapse is the RENDER** (MoE
  hardware-nondeterminism, render-noise SD ≈ effect at n=12).
- **CONCLUSION:** BOTH positive headlines (referent logprob + judged sense) are render-fragile
  and do NOT reproduce on an independent render. The graft's meaning-recovery effect is not a
  robust finding; the paper cannot be led by a stable recovery claim (see FINDINGS banner pt 4).
- **RECOMPUTE:** `scripts/judged_bootstrap.py --graft-glob results/judge_semantic_brief_base/verdicts_*.json
  --base-glob results/judge_semantic_base_brief_base/verdicts_*.json` (clean); `…/judge_semantic{,_base}/reverdict_*.json` (original, same judge).
- **NOTE:** held-out c13–c24 (generating) is now moot for the headline — the baseline positive
  control already failed under clean code.

### REC-6 · SWE-Gym / OpenHands coding trajectories (real non-synthetic content) ✅ VERIFIED
- **CLAIM:** tuned ValueGraft (E-tuned, α=0.75) recovers **+0.0156 nats/token** on true
  next-action prediction across **75 traces, 45/75 wins**, ~10% of measured compaction damage.
- **RECOMPUTED (2026-07-09):** mean(E-tuned − B) tf_mean = **+0.0156**, wins **45/75**,
  normal-approx 95% CI **[+0.0047, +0.0266]** (excludes 0).
- **SOURCE:** `results/swegym_30b_bf16/t*.json` (75 files), arm field `arms.{E-tuned,B}.tf_mean`.
- **COMMAND:** `python3 -c "import json,glob; r=[json.load(open(f))['arms']['E-tuned']['tf_mean']-json.load(open(f))['arms']['B']['tf_mean'] for f in glob.glob('results/swegym_30b_bf16/t*.json')]; print(sum(r)/len(r), sum(x>0 for x in r), len(r))"`
- **COMMIT:** results committed under f4b6a89.
- **PROVENANCE (confirmed from code):** model `Qwen/Qwen3-30B-A3B-Instruct-2507`, **bf16**
  (`run_swegym_hf.py:105`), summaries **SELF-GENERATED by the model** (`generate_summary_hf`,
  `run_swegym_hf.py:134`) under the **BRIEF** request (`SUMMARY_REQUEST_BRIEF`), α=0.75 layer-tuned.
- **NOTE:** same BRIEF (mechanism-isolation) summary condition as the judged +12pp headline — the
  brief condition spans the synthetic judged result AND this real-content result. State that caveat.

---

## SECTION 2 — COMPACTION DAMAGE

### DMG-1 · Damage ordering (judged, corpus) ✅ VERIFIED
- **CLAIM:** Original→Compacted drop: stance 96→93 (−3pp), sense ~100→46 (−54pp),
  referent ~100→17 (−83pp). Ordering referent ≫ sense ≫ stance.
- **RECOMPUTED:** B rates 93 / 46 / 17 exact (REC-1b); A ceilings indicative (tiny n).
  Ordering holds.
- **NOTES:** Framed as baseline/denominator, NOT a finding. 4-bit MLX / BRIEF / c01–c12.

### DMG-2 · LongMemEval compaction damage ⚠ RECONCILE (provenance-thin)
- **CLAIM (DRAFT §3.1):** full-context **52.5% → 4.1%** correct under compaction,
  **n=320**, bf16, standard data (cited DECISIONS 2026-07-05 stage-1 keeper).
- **RECOMPUTED:** the only recomputable verdicts file
  `results/longmemeval_30b_verdicts.json` gives A(full) **80.6%** → B(compacted)
  **5.6%**, **n=36** — a *different, smaller* run (arms A/B/E-tuned/B-min-pack/H-pack/H-gap).
- **REASON:** the n=320 figure is cited to **DECISIONS prose**, not a single committed
  JSON; the 320-item run's per-item files were not located this pass
  (`results/longmemeval_30b_bf16/` has many hash-named files but no scored roll-up found).
  Direction (severe collapse) is not in doubt; the exact 52.5/4.1/320 triple is
  **not reproduced from disk**. Either locate the source file or quote the n=36
  recompute. This is baseline material, not a headline.
- **COMMAND:** inline python counting CORRECT/total per arm over
  `results/longmemeval_30b_verdicts.json['verdicts']`.

### DMG-3 · Content-specificity (placebo control, 27B) ✅ VERIFIED
- **CLAIM:** E − placebo **+0.445** CI[+0.286,+0.612] (excl 0); placebo − B **−0.428**
  CI[−0.590,−0.276]; E − B +0.017 CI[−0.033,+0.065] (null).
- **RECOMPUTED:** E−placebo +0.4449 [+0.2856,+0.6124]; placebo−B −0.4283
  [−0.5904,−0.2762]; E−B +0.0166 [−0.0334,+0.0653] null. Exact.
- **SOURCE:** `results/effect_bound/summary.json` (`summary.delta_*` blocks; n=43,
  Qwen3.6-27B, α_V=0.75, window K=12, placebo_seed=20260707, n_boot=10000).
- **NOTES:** 27B, narrow K=12 pre-window (answer preamble). 30B counterpart
  (E−placebo +0.241 [+0.033,+0.457]) is in
  `results/effect_bound/effect_bound_qwen3-30b-a3b_fullwin_*.json` — cite there;
  not independently rebootstrapped this pass. E−B null here is expected (preamble window).

---

## SECTION 3 — F2 HONESTY / ANTI-FABRICATION (bf16 30B)

All rows recomputed from `results/phase2_30b_scored.json` (list of 576 rows; arms
A, B, B-min-pack, H-pack, H-gap, H-pack-wrongS; kinds referent, sense, evicted_fact,
decoy; field `verdict` ∈ {CORRECT, ADMITTED, FABRICATED}; 24 rows per arm×kind).
Model field = `Qwen/Qwen3-30B-A3B-Instruct-2507`, dtype **bfloat16**
(`results/phase2_30b_verdicts.json`). Independently confirmed against
`results/phase2_4b_scored.json`.

### F2-1 · Decoy fabrication, Compacted vs H-pack ⚠ RECONCILE
- **CLAIM:** Compacted **83%** vs write-time-KV (H-pack) **17%**.
- **RECOMPUTED:** B(Compacted) decoy FABRICATED **19/24 = 79.2%**; H-pack decoy
  FABRICATED **3/24 = 12.5%**. (4B: B 18/24=75%, H-pack 3/24=12.5%.)
- **REASON:** same direction and story ("~5-in-6 → ~1-in-8"), but the cited 83/17 does
  not reproduce — use **79% / 12%**. Likely an earlier/rounded figure.
- **COMMAND:** count `verdict=='FABRICATED'` for `kind=='decoy'`, arm B vs H-pack,
  over `results/phase2_30b_scored.json`.

### F2-2 · "H-pack most accurate on evicted facts, 38/48 = 79%" ❌ UNSUPPORTED
- **CLAIM:** on evicted facts the write-time-KV arm was **most accurate (38/48 = 79%)**
  and least fabricating (4%).
- **RECOMPUTED:** H-pack evicted_fact CORRECT = **0/24 (0%)** — identical to Compacted.
  Only arm **A (full context)** is accurate (24/24). H-pack evicted_fact verdicts =
  {ADMITTED 23, FABRICATED 1}. **No 38/48 anywhere.**
- **SEARCHED (all negative):** `phase2_30b_scored.json` (H-pack CORRECT 0/24),
  `phase2_4b_scored.json` (H-pack evicted CORRECT 0/24), `phase2_30b_verdicts.json`
  (same values), `honesty_30b_bf16/` (raw answers only; arms B/E-tuned, no scored
  H-pack CORRECT), grep for `38/48` across repo (0 hits). 48 ≠ any 2-arm/2-cat pool
  that lands on 38 either.
- **VERDICT:** **overclaim.** The "least fabricating (4%)" half IS supported
  (fab 1/24 = 4.2%); the "most accurate 38/48" half is not.
- **CORRECTED CLAIM (ship this instead, ✅ VERIFIED below as F2-2b).**

### F2-2b · What H-pack actually does on evicted facts ✅ VERIFIED
- **RECOMPUTED:** H-pack converts evicted-fact **fabrication 16/24 (67%) → admission
  23/24 (96%)**, fabricating 1/24 (4%). It **suppresses fabrication / induces
  admission**; it does **not** restore recall.
- **NOTES:** cleaner, more defensible than the accuracy phrasing. (4B mirrors:
  B fab 15/24 → H-pack fab 2/24, admit 22/24.)

### F2-3 · Honesty decomposition (layout vs write-time-KV) ⚠ RECONCILE
- **CLAIM (FACTS / DECISIONS 07-06):** layout moves decoy fab **83→25%**;
  write-time-KV within layout **25→17%**; admissions **3/24 → 18/24**.
- **RECOMPUTED (decoy):** B **79%** → B-min-pack **42%** (10/24) → H-pack **12.5%** (3/24).
  Decoy admissions: B **5/24** → H-pack **21/24** (B-min-pack 14/24).
- **REASON:** the decomposition *structure* holds (packing helps, write-time-KV adds
  more), but the sub-numbers (83/25/17 and 3/24→18/24) are stale and do not reproduce.
  Recompute per-arm before shipping any decomposition figure.

### F2-4 · Replication across scale/precision ✅ VERIFIED (directional)
- **CLAIM:** F2 replicates 4B→30B and 4-bit→bf16.
- **RECOMPUTED:** decoy fab B→H-pack: 30B 79%→12.5%, 4B 75%→12.5%; evicted fab→admit
  pattern present at both scales. Direction replicates. (bf16 confirmed via
  `phase2_30b_verdicts.json` dtype field.)

### F2-5 · Scope: honesty washes out on LongMemEval personal-QA ✅ VERIFIED (supporting)
- **RECOMPUTED (`longmemeval_30b_verdicts.json`, n=36):** B CORRECT 5.6% vs H-pack
  11.1% — arms barely separate on accuracy; "packing buys honesty, not memory."
- **NOTES:** consistent with the stated scope condition (mid-task agentic compaction,
  not retrieval personal-QA). n=36 here (see DMG-2 provenance caveat).

---

## SECTION 4 — CROSS-ARCHITECTURE MAP (per-model native, self-gen summary, conv-clustered)

Field to cite: `by_category_robust.<cat>.raw_EB_mean` + `raw_EB_ci` (conversation-
clustered) in each `results/cross_arch_done/*.json`. **Field trap (Mistral):** cite
`raw_EB_ci_cluster`, NOT `raw_EB_ci` — they differ (see CA-5).

### CA-1 · Qwen2.5-32B (dense) referent ✅ VERIFIED
- **CLAIM:** strongly negative. **RECOMPUTED:** referent −0.230 CI[−0.303,−0.157];
  sense −0.424 [−0.531,−0.317]; stance −0.178 [−0.223,−0.133] (floored). n=24 conv,
  48 plants. Agg −0.230 [−0.276,−0.183].
- **SOURCE:** `results/cross_arch_done/Qwen__Qwen2.5-32B-Instruct.json`.

### CA-2 · Qwen3-32B (dense) referent ✅ VERIFIED
- **CLAIM:** null-to-harmful. **RECOMPUTED:** referent −0.028 CI[−0.063,+0.010] (null);
  sense −0.048 [−0.091,−0.008] (neg); stance −0.094 [−0.139,−0.051] (neg). n=24/48.
- **SOURCE:** `results/cross_arch_done/Qwen__Qwen3-32B.json`.

### CA-3 · phi-4 (dense) ✅ VERIFIED
- **CLAIM:** referent −0.053 [−0.117,+0.008]; aggregate −0.064 [−0.099,−0.027]
  SIGNIFICANT_NEGATIVE.
- **RECOMPUTED:** referent −0.0528 [−0.117,+0.008] ✅ · agg −0.0636 [−0.0992,−0.0273] ✅.
  sense −0.041 [−0.126,+0.035]; stance −0.048 [−0.079,−0.012]. n=12/24.
- **SOURCE:** `results/cross_arch_done/microsoft__phi-4.json`.

### CA-4 · Mistral-Small-24B (dense) referent POSITIVE ✅ VERIFIED (field-trap noted)
- **CLAIM:** referent +0.035 [+0.010, +0.063] POSITIVE; sense −0.037 [−0.074,−0.001];
  stance −0.037 [−0.055,−0.020]; agg −0.012 [−0.027,+0.002] null.
- **RECOMPUTED:** referent_mean +0.0351; `raw_EB_ci_cluster` **[+0.0097, +0.0629]**
  (matches FACTS +0.010/+0.063 ✅); sense −0.0367 [−0.074,−0.001]; stance −0.0372
  [−0.055,−0.020]. n=12/24.
- **⚠ FIELD TRAP:** the top-level `raw_EB_ci` field for Mistral is **[+0.0054, +0.0663]**
  (ci_method=None) and `raw_EB_ci_plant` is null. **Cite `raw_EB_ci_cluster`** — that is
  what FACTS/DRAFT quote. Do not accidentally cite `raw_EB_ci`.
- **SOURCE:** `results/cross_arch_done/mistralai__Mistral-Small-24B-Instruct-2501.json`.

### CA-5 · Anchor Qwen3-30B-A3B (MoE) — banked native referent ✅ VERIFIED / ⚠ (see REC-2)
- **CLAIM (DRAFT §6 table):** referent +0.012 [−0.068, +0.090] (banked; anchor);
  sense +0.040 [−0.029, +0.103]; stance −0.086 [−0.106, −0.064].
- **RECOMPUTED:** `b0_baseline_c01-12.json` referent +0.0119, cluster CI
  [−0.0684, +0.0903]; sense +0.040 [−0.029,+0.103]; stance −0.086 [−0.106,−0.064]. Exact.
- **SOURCE:** `results/cross_arch_done/b0_baseline_c01-12.json` (n=12 conv).
- **NOTES:** the MoE-positive pole is the **fragile** one (this banked render is the
  first, and its referent is null; the +0.10/+0.125 anchor did not reproduce — REC-2 /
  BLK-1). Dense-negative poles (CA-1..CA-3) are solid.

---

## SECTION 5 — BLOCK / REPRODUCTION

### BLK-1 · Positive-control + held-out block ✅ VERIFIED
- **CLAIM:** baseline c01–c12 referent **+0.012** [−0.070, +0.091] (positive control
  FAILS / spans 0); fresh c13–c24 referent −0.021 [−0.082, +0.041] (null, high headroom).
- **RECOMPUTED:** BASELINE referent +0.012 CI[−0.070,+0.091] → "FAIL: positive control
  did NOT reproduce, RUN IS SUSPECT"; FRESH referent −0.021 CI[−0.082,+0.041] NULL.
  Morphology 100% sem_phrase both cells.
- **SOURCE:** `results/cross_arch_done/b0_baseline_c01-12.json`,
  `results/cross_arch_done/b1_fresh_c13-24.json`.
- **COMMAND:** `python3 scripts/block_analysis.py
  results/cross_arch_done/b0_baseline_c01-12.json
  results/cross_arch_done/b1_fresh_c13-24.json`
- **NOTES:** central pooled relative competence floor = −7.44 (0 plants dropped).
  This is the on-disk basis for "referent anchor is render-fragile" (REC-2, DRAFT §4.5).

---

## SECTION 6 — METHODOLOGY

### MTH-1 · Estimator: raw_EB locked, mean-of-ratio retired ✅ VERIFIED
- **RECOMPUTED evidence:** on `gap_closure_cat` vs `_live`, raw E−B agrees across runs
  (referent +0.156/+0.125, sense +0.062/+0.047, stance −0.026/+0.002 — REC-2), while
  mean-of-ratio was the unstable estimator. Consistent with the claim.
- **NOTES:** primary = raw_EB + %-helped + conv-clustered bootstrap; median-ratio only
  conditioned on |A−B|>0.5.

### MTH-2 · Keys ~neutral, values operative ✅ VERIFIED (directional, on disk via FINDINGS-cited runs)
- **CLAIM:** keys ~neutral on raw E−B (−0.020..+0.016/layer); v_only referent +0.120
  CI[−0.001,+0.228]. K-graft RoPE re-rotation validated cosine 0.99999982.
- **STATUS:** the corrected raw-E−B keys numbers are cited to FINDINGS ratio-correction
  entry; `results/kv_layer_probe/` holds the per-layer data. Recompute of the exact
  −0.020..+0.016 not re-run this pass (not a KNOWN-CASE); flagged as FINDINGS-cited,
  arms-level plausible. **If shipping keys numbers, recompute from `kv_layer_probe/`.**

### MTH-3 · Own-summary mechanism ✅ VERIFIED (directional)
- **CLAIM:** fixed foreign summary → referent +0.004 (null), sense −0.147; self-gen →
  referent +0.136 [+0.034,+0.23], agg +0.090 [+0.022,+0.154].
- **STATUS:** cited to FINDINGS 07-08 RESOLVED entry (isolation test). Not re-bootstrapped
  this pass; the b0 banked native self-gen referent (+0.012, CA-5) is the newer, weaker
  banked number — present the mechanism direction, but note the self-gen +0.136 point
  estimate is the same render-fragile class as REC-2.

---

## SECTION 7 — PROVENANCE (the facts about the facts)

### PROV-1 · Judged model is 4-bit MLX (not bf16) ✅ VERIFIED
- **RECOMPUTED:** `results/raw_30b/c01.json['model']` =
  `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`; same in `raw_30b_brief/c01.json`.
- **COMMAND:** `python3 -c "import json;print(json.load(open('results/raw_30b/c01.json'))['model'])"`
- **NOTES:** the "bf16" label in F1 prose and in `judged_bootstrap.py`'s docstring header
  is **wrong** for the judged metric. Judge = Sonnet 5.

### PROV-2 · Judged source is raw_30b_brief (BRIEF), not raw_30b (std) ✅ VERIFIED
- **RECOMPUTED (rebuild judge batches, diff vs committed):**
  GRAFT side — `raw_30b_brief` reproduces **126/128** committed prompts exactly;
  `raw_30b` (std) reproduces only **2/128**. (BASE side: brief 63/80 vs std 17/80.)
- **SOURCE:** `results/judge_semantic/batch_*.json` (committed) vs rebuilt from
  `results/raw_30b_brief/` and `results/raw_30b/`.
- **COMMAND:** `python3 scripts/build_judge_batches.py --validate` (validates against
  `raw_30b` → shows GRAFT exact_match=2/128, i.e. std does NOT match; the brief match
  of 126/128 is shown by pointing the builder at `results/raw_30b_brief`).
- **NOTES:** confirms the judged headline lives under the BRIEF / B-handicapped
  condition. c01 brief summary = 543 chars (positive control, DECISIONS 07-06).
  `--validate` prints "FAIL" only because its hardcoded default is the std dir; that is
  expected and is itself the proof that std is not the source.

### PROV-3 · Corpus / render provenance ✅ VERIFIED (documented)
- User turns + plants: AUTHORED (Claude subagents). Assistant replies: generated
  in-context by the TEST MODEL (per-model native render; original c01–c12 rendered by
  Qwen3-4B MLX — the nativeness confound). Summary: SELF-GENERATED. Gold continuation:
  derived from planted facts, SHARED across models. (Matches FACTS §Provenance; the
  `meta` block in `data/synthetic/cNN.json` records the original render model.)

---

## APPENDIX — commands to re-audit everything

```
git rev-parse --short HEAD                      # expect f4b6a89
python3 scripts/judged_bootstrap.py --nboot 20000 --seed 0          # REC-1
python3 scripts/build_judge_batches.py --validate                  # PROV-2 (std side)
python3 scripts/block_analysis.py \
  results/cross_arch_done/b0_baseline_c01-12.json \
  results/cross_arch_done/b1_fresh_c13-24.json                     # BLK-1 / CA-5
# F2 (phase2_30b_scored.json), cross-arch (cross_arch_done/*.json),
# logprob primary (gap_closure_cat_live/*.json), placebo
# (effect_bound/summary.json), LME (longmemeval_30b_verdicts.json):
# recompute via the inline python documented in each row above.
```

*Ledger built adversarially: the goal is to catch overclaims before writing, not to
rubber-stamp. Re-run at every new HEAD before promoting the paper to README.*
