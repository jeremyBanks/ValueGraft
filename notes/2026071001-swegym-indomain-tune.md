# Brief-SWE-Gym: CONFIRM-then-OPTIMIZE (reconciled combined design; free step DONE)

## 2026-07-10 UPDATE — tune finished; data exhausted; the decisive confirm is the original pool

**Tune result (idx 215-486, N=98).** The naive scalar graft did NOT replicate here
(E-tuned − B = −0.0017, spans 0). But the in-domain-tuned per-layer champion (layers
12-17 + 30-35, α=1.0, selected on the tune-41 subset) **cleared baseline held-out**:
E-champion − B = **+0.0117 CI[+0.0063,+0.0172]** on eval-57 (44/57 improved).

**FREE diagnostics (0 spend, from the persisted champeval scores):**
- Full 98: E-champion − B = **+0.0111 CI[+0.0069,+0.0155]** (73/98); tune-41 +0.0103,
  eval-57 +0.0117 — **not split-specific** (the "eval leans favorable" worry is overblown:
  over 200 random half-splits, **200/200 have positive mean**, 5-95th pctile [+0.0071,+0.0149]).
- Head-to-head on the full 98: E-champion − E-tuned = **+0.0128 CI[+0.0040,+0.0230]** — the
  champion **beats** the scalar (the head-to-head only spanned 0 on the thin eval-57 alone).
So the champion is more robust than the eval-57-only read suggested — WITHIN the idx≥215 pool.

**DATA-HEADROOM number (the constraint the owner asked me to check FIRST): EXHAUSTED.**
Ran the real `find_cut` filter over all 491 parquet rows: exactly **173 pass** the
6k–15k-token budget/cut filter, and **all 173 are already used** (75 at idx 1-213 + 98 at
idx 215-486). **Zero unused budget-passing trajectories remain.** A brand-new disjoint set
is impossible without relaxing the filter (which yields out-of-band, shorter/longer,
lower-quality trajectories).

**TOP RECOMMENDATION (~$2-3 of the $11): apply the FIXED champion to the ORIGINAL 75
(idx 1-213).** These were scored only with scalar / synthetic-champion arms, never the
SWE-Gym champion, and were **entirely disjoint from this champion's selection** (which used
only idx≥215). They are in-band budget-passers. Re-scoring them with the fixed champion is
the one clean, in-band, fully out-of-sample N=75 replication the exhausted data still
allows — the decisive test of whether +0.011 is real or specific to the idx≥215 pool. It
also re-tests the scalar α-sweep + placebo + discrete metric on this independent set. The
champion config was reconstructed and byte-verified (sha256
`6faa2d72…` matches the champeval intervention record) and committed, since the pod's copy
was lost on termination.

**Why not the alternatives.** (a) *New disjoint set* — impossible, data exhausted. (b)
*Fresh-seed re-split of the 98* — FREE (CPU), already done above (200/200 positive); no GPU
needed, and it's partly circular since selection used a subset of the same 98. (c) *Relax
the budget filter* — unlocks only out-of-band trajectories (a different regime than the one
tuned/measured), a weaker and confounded test; keep as an optional way to burn leftover
credit, not the primary. (d) *Compression dose-response* — changes the question (cross-regime
generalization) rather than resolving the sign.

**EXACT LAUNCH COMMAND (the confirm; ~$2-3, do not launch — owner fires):**
```bash
SC_HF_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 SC_LOAD_DTYPE=bfloat16 SC_SUMMARY=brief \
SC_SWE_MIN_IDX=0 SC_SWE_N=75 \
SC_CHAMPION_CONFIG=data/champion_configs/swegym_tuned_20260710T145330Z.json \
SC_E_ALPHAS="0.5,0.75,1.0" SC_SWE_PLACEBO=1 \
scripts/launch_pod.sh swegymconfirm scripts/job_swegym_confirm_bf16.sh
```
Writes `results/swegym_confirm_<STAMP>_brief` (UNIQUE; never reuses existing dirs); prints
E-champion−B, E-champion−E-tuned, the α-sweep, content-specificity, and the discrete metric
over the 75. **Optional** (to also spend the remaining ~$7 on an out-of-band robustness
check): add a second launch of the same job with the budget filter relaxed — but that needs
a small harness knob (SC_SWE_MIN_TOK/SC_SWE_MAX_TOK) not yet added; flag if wanted.

**One-line paper-conclusion effect:** if E-champion−B clears 0 on the original 75, §3.3/§10
change from "one fragile, thin-N, possibly-pool-specific positive" to "a modest in-domain
per-layer-tuned graft that replicates across two disjoint trajectory pools (while the naive
scalar does not)"; if it spans 0, the champion positive was pool-specific and the paper
stays at its clean bounding-null with the scalar proxy as the only (fragile) signal.

---


**Status:** built + committed; the paid run is queued, NOT launched (owner fires it).
The free step 0 is DONE (numbers below). Reconciles the α-sweep/profiling design with
an independent prioritization: the paper's only fragile claim is the brief-SWE-Gym
positive (+0.013, N=75, lower bound one resample from zero), so the priority is to
**resolve that positive's sign** (confirm) before optimizing — via independent new N,
a near-free discrete task-relevant metric, and the α-sweep in the same run.

## FREE STEP 0 — DONE (0 spend); results
Backfilled the discrete next-action-match metric on both existing 75-trajectory runs
(`src/backfill_swegym_action_match.py`, sidecars written; originals not mutated). The
two runs cover the **same 75 trajectory indices** (test-retest, not independent N).

| Existing run | continuous E−B (logprob) | action-match B | action-match graft | paired discrete (graft−B) |
|---|---|---|---|---|
| `swegym_30b_bf16` | +0.0156 [+0.0049, +0.0271] | 37% (28/75) | 39% (29/75) | +0.013 [−0.040, +0.067] |
| `swegym_30b_bf16_brief` | +0.0133 [+0.0016, +0.0263] | 33% (25/75) | 36% (27/75) | +0.027 [−0.027, +0.080] |
| **pooled (per-trajectory averaged, denoised)** | **+0.0145 [+0.0040, +0.0257]** | — | — | — |

Reading: the continuous positive **holds and tightens slightly** when the two
measurements of the same trajectories are averaged (+0.0145, lower bound +0.0040 —
still one resample from zero). The discrete metric points the **same direction** (the
graft emits the correct next tool/path/command slightly more often, net +1 and +2
"fixes minus breaks"), but is **underpowered** at N=75 (paired CIs span 0). Both say
the same thing: the effect is real-but-fragile and the resolver is **independent N**,
which only new disjoint trajectories provide. (Every gold turn here happened to be a
real action, 75/75.)

**Status of the paid run:** queued, NOT launched. Do not touch running pod state files.

## The gap this closes
The value graft's *only* net-positive cell is brief-SWE-Gym: flat scalar α=0.75 gives
next-action recovery **+0.013 [+0.002, +0.026]** (clears 0, N=75). The lower bound is one
cluster-resample from zero; everything else in the paper is comfortably null. So the
single highest-value move is to **resolve that positive's sign** — which pure optimization
assumes away. **Confirm before optimize.** Then, in the same run, also learn whether 0.75
is in-domain-optimal and whether the effect is content-specific and behaviorally real.

## Combined design (ONE run, unique stamped dir)
Hold fixed: Qwen3-30B-A3B-Instruct-2507, bf16, **brief** self-generated summaries, the
teacher-forced next-action mean-logprob metric (a **proxy**), the same ~75% pre-action cut.

1. **Independent new N (the confirm).** Score ~150 trajectories **disjoint** from the
   existing 75: `SC_SWE_MIN_IDX=214` (the existing runs consumed all find_cut-passing
   indices in [1,213]) → guaranteed-disjoint tail. This adds independent N to the α=0.75
   arm, tightening its CI to firm the positive or collapse it to a clean null. **Parquet
   reality:** it holds **491** trajectories, but only ~35% pass the find_cut budget filter
   (6k–15k tokens with a valid cut), and the existing 75 already took the passers in
   [1,213]; the disjoint tail (idx 214–490, 277 candidates) yields **~90–100 usable**, not
   a full 150. `SC_SWE_N=150` is a ceiling; the run takes what passes. ~75 → ~170 total
   shrinks the CI half-width by roughly a third.
2. **α-sweep in the same run.** Value graft at α ∈ {0.25, 0.5, 0.75, 1.0, 1.5} as
   score-only arms `E-a{α}` (share the trajectory's prefill → the extra α are cheap).
   Answers "is 0.75 in-domain-optimal, or does another α recover more?"
3. **Placebo per α.** `P-a{α}` = same slots, source values position-shuffled (seeded).
   Content-specificity = `E-a{α} − P-a{α}`.
4. **Discrete next-action-match metric** (addresses the paper's biggest caveat without
   Docker/tests). Parse the free generation of each *generating* arm (baseline, α=0.75,
   champion) for the emitted **tool / file-path / command** and score whether it matches
   the true next action when the baseline does NOT (and vice-versa). Recorded per
   trajectory inline (`arms.<name>.action_match`) plus `gold_action` at the top level.
5. **Held-out discipline + provenance.** Deterministic `sha256(seed:idx)` tune|eval split
   per trajectory; α selected / any layer champion built on **tune**, reported on the
   **disjoint eval**; trajectory-clustered bootstrap CIs; unique stamped dir; born-annotated
   manifest recording `min_idx`, the arm interventions, and the discrete secondary metric.
6. **Per-region profiling = OPTIONAL stretch** (`SC_STAGE=full`), only if budget remains
   after the core: 8 fractional-depth region marginals `R{k}` on the tune split → build a
   SWE-Gym-tuned per-layer champion (union of positive-tune regions) → a second pass
   evaluates it **held-out** (`E-champion` vs scalar vs baseline, continuous + discrete).

### Overfitting caveat (stated honestly)
Even with the new N, per-split counts are modest (~85 each after the tune/eval split).
Selecting α (one global scalar) is robust to this; a *layer* champion has many degrees of
freedom, so its held-out eval is the one that matters — an in-sample win that does not
survive the eval split is overfitting, and the analysis flags it. The failed
synthetic-champion transfer mildly suggests the graft may be **config-insensitive** here
(only the presence of the graft matters, not α or layer-set), in which case the honest
result is "flat — nothing beats the naive scalar." That is a real, reportable outcome.

## Loss lesson applied
Every run writes to a **unique, stamped** directory (`results/swegym_tune_<STAMP>_brief`,
`results/swegym_champeval_<STAMP>_brief`) and a stamped champion config
(`data/champion_configs/swegym_tuned_<STAMP>.json`). The job **never** writes to
`results/swegym_30b_bf16_{brief,prod}` and never reuses a prior run's filenames — the
prod-champion data was lost precisely because a re-run reused those paths and a cp-backup
created a revert path. No in-place overwrite of any prior run.

## Staged cost estimate (one A100 80GB @ ~$1.39/hr; balance $25.30)
The α-sweep/placebo/region arms are **score-only** (teacher-forced logprob, no 200-token
generation); only the ~6 primary arms generate (and carry the discrete metric). The
disjoint core scores ~90–100 new trajectories.

- **Stage `sweep`** (core: disjoint new N + α-sweep + placebo + discrete, one pass): ~90–100
  trajectories × ~60–90 s ≈ **1.5–2.5 h ≈ $2–3.5**. This is the priority — it resolves the
  +0.013 sign, tells us if 0.75 is optimal, gives content-specificity, and the behavioral
  discrete metric. Comfortably inside $25.
- **Stage `full`** (adds region profiling in the same first pass + a second held-out
  champion-eval pass over the same disjoint set): ~2 passes ≈ **3.5–5 h ≈ $5–7** total.
  Still well inside $25, so `full` is affordable if the owner wants the profiling stretch.

(Estimates; the run prints per-trajectory wall time so the first few calibrate it.)

## How to run — exact launch commands (ready to fire when funded)

**Core (priority) — disjoint new N + α-sweep + placebo + discrete metric:**
```bash
SC_HF_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
SC_LOAD_DTYPE=bfloat16 \
SC_SUMMARY=brief \
SC_STAGE=sweep \
SC_SWE_MIN_IDX=214 \
SC_SWE_N=150 \
SC_E_ALPHAS="0.25,0.5,0.75,1.0,1.5" \
SC_SWE_PLACEBO=1 \
scripts/launch_pod.sh swegymtune scripts/job_swegym_tune_bf16.sh
```

**Full — the above PLUS per-region profiling + held-out champion eval (affordable stretch):**
```bash
SC_HF_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
SC_LOAD_DTYPE=bfloat16 \
SC_SUMMARY=brief \
SC_STAGE=full \
SC_SWE_MIN_IDX=214 \
SC_SWE_N=150 \
SC_E_ALPHAS="0.25,0.5,0.75,1.0,1.5" \
SC_SWE_PLACEBO=1 \
SC_PROFILE_REGIONS=8 \
scripts/launch_pod.sh swegymtune scripts/job_swegym_tune_bf16.sh
```

(`launch_pod.sh swegymtune ...` writes `.pod_swegymtune_state.json`; all knobs incl.
`SC_SWE_MIN_IDX` are forwarded by `launch_pod.sh`. `swegym.parquet` is staged locally and
rsyncs to the pod. Both `SC_STAGE` values write UNIQUE stamped dirs
`results/swegym_tune_<STAMP>_brief` / `results/swegym_champeval_<STAMP>_brief` — never the
existing `swegym_30b_bf16_{brief,prod}`. Re-run analysis anytime for free:
`python3 src/analyze_swegym_tune.py --run results/swegym_tune_<STAMP>_brief
[--champion-eval results/swegym_champeval_<STAMP>_brief]`.)

## Free step 0 — reproduce anytime (0 spend)
```bash
uv run python src/backfill_swegym_action_match.py --parquet swegym.parquet \
  --dirs results/swegym_30b_bf16 results/swegym_30b_bf16_brief
```
Writes `action_match_backfill.json` sidecars (originals untouched) and prints the pooled +
discrete numbers in the table above.

## How to read the paid run
The analysis prints, for the held-out **eval** split unless noted:
- **scalar baseline** `E-tuned − B` continuous (all/tune/eval) + the **discrete
  next-action-match** rates (baseline vs graft) and paired difference — the +0.013 to
  confirm, now with independent N.
- **α curve**: per-α `E-a − B` (in-sample ALL and held-out EVAL) + content-specificity
  `E-a − P-a`. Then the **α selected on TUNE** and its **held-out EVAL** CI vs scalar 0.75.
- **region marginals** on TUNE, the champion layer-set built, and (full mode) the
  **held-out champion**: `E-champion − B` (continuous + discrete) and paired vs scalar.
- **Decision rules**: the positive is *confirmed* if the α=0.75 arm's larger-N EVAL CI
  still clears 0; a tuned α or layer champion *wins* only if its held-out CI clears 0 **and**
  beats scalar; if head-to-head CIs span 0, the in-domain tuning added nothing and the naive
  scalar is the robust form — report that plainly.

## Paper update text (for review — NOT inserted; owner finalizes)

### §3.3 — add after the proxy-caveat sentence (behavioral corroboration from the free step)
> A behavioral cross-check on the same 75 trajectories, using the free generation rather than
> the log-probability, points the same way but is underpowered. Parsing each arm's emitted
> next action (tool, file path, command) and scoring it against the true next action, the
> graft matches the correct action slightly more often than the compacted baseline (39% vs
> 37%, and 36% vs 33% across the two runs), with a small positive excess of trajectories the
> graft gets right and the baseline gets wrong; the paired difference is positive but its
> interval spans zero at this sample size. This is a weak, same-direction corroboration that
> the log-probability gain corresponds to occasionally producing the correct next action —
> not a claim of behavioral significance, which N=75 cannot support.

### §10 — replace the future-work paragraph with confirm-then-optimize
> **Confirm the one live effect, then optimize it where it lives.** The single fragile claim
> in this paper is the brief-summary coding positive: at N=75 its interval clears zero only
> narrowly, and pooling our two measurements of those trajectories tightens the estimate to
> about +0.0145 nats/token without moving the lower bound far from zero. The first priority
> is therefore not optimization but *sign resolution* — scoring a set of new coding
> trajectories disjoint from the original, which adds the independent sample the current
> estimate lacks and will either firm the effect or collapse it to a clean, complete null.
> In the same measurement one can also learn whether the graft strength we used out of habit
> is in fact the in-domain optimum (a strength sweep with a shuffled-source placebo for
> content-specificity) and whether a graft profiled per-layer directly on coding data beats
> the naive single-strength graft under a strict tune/evaluate split — noting that the
> failure of our conversation-tuned configurations to transfer mildly anticipates a flat
> result, in which case only the presence of the graft matters. We add a behavioral
> next-action-match metric (does the grafted model emit the correct tool, file, and command
> when the baseline does not) to make this partly independent of the log-probability proxy
> without test execution; a fuller version would measure task success directly, which the
> capability floor of a 30B model on these repairs makes uninformative at any affordable
> scale. The narrowness is the point: the value is in characterizing the one live margin
> precisely, not in reviving the general claim the rest of the paper bounds to null.

## Files
- `src/swegym_action_match.py` — the discrete next-action-match extractor/scorer (shared by
  harness, backfill, analysis).
- `src/backfill_swegym_action_match.py` — free step 0: backfills the discrete metric on
  existing dirs (sidecars, originals untouched) + prints the pooled test-retest estimate.
- `src/run_swegym_hf.py` — `SC_SWE_MIN_IDX` disjoint-set control; inline discrete
  action-match on every generating arm + top-level `gold_action`; score-only α-sweep /
  placebo / region-profile arms; deterministic tune|eval split; manifest records it all.
- `src/analyze_swegym_tune.py` — CPU held-out analysis: continuous + discrete scalar
  baseline, α curve (tune-select→eval-report), content-specificity, region marginals,
  champion build, held-out champion eval (continuous + discrete).
- `scripts/job_swegym_tune_bf16.sh` — combined staged job (`sweep` core = disjoint N +
  α-sweep + placebo + discrete; `full` adds profiling + held-out champion eval); unique
  stamped dirs, never reuses `swegym_30b_bf16_{brief,prod}`.
- `scripts/launch_pod.sh` — forwards the new envs (`SC_STAGE`, `SC_SWE_MIN_IDX`,
  `SC_E_ALPHAS`, `SC_SWE_PLACEBO`, `SC_PROFILE_REGIONS`, `SC_PROFILE_ALPHA`).
- `results/swegym_30b_bf16{,_brief}/action_match_backfill.json` — free-step sidecars.
