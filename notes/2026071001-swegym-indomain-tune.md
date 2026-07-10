# In-domain graft optimization on brief-SWE-Gym (design; queued, NOT launched)

**Status:** built + committed, NOT launched (balance $0.30; fires only when funded).
Do not launch; do not touch running pod state files.

## The gap this closes
The value graft's *only* net-positive cell is brief-SWE-Gym: flat scalar α=0.75 gives
next-action recovery **+0.013 [+0.002, +0.026]** (clears 0). But every champion we built
was tuned on the **synthetic conversation** corpus — the null regime — and transferred
out-of-domain to SWE-Gym adding nothing (≈ scalar; head-to-head flat). `run_swegym_hf.py`
had only ever run a single α (default 0.75) plus an optional *transferred* champion. We
have **never** swept α or profiled per-layer **on coding trajectories**. This optimizes the
graft in the one regime where it works, and asks: can an in-domain-tuned graft beat the
naive +0.013, and by how much?

## Design
Hold fixed: Qwen3-30B-A3B-Instruct-2507, bf16, **brief** self-generated summaries, the
teacher-forced next-action mean-logprob metric (a **proxy**, not resolve rate), the same
~75% pre-action cut. Vary the graft.

1. **α-sweep** (cheap, first): value graft at α ∈ {0.25, 0.5, 0.75, 1.0, 1.5} on every
   trajectory, as score-only arms `E-a{α}`. Answers "is 0.75 optimal on code, or does
   another α recover more of the gap?"
2. **Placebo per α**: `P-a{α}` = same grafted slots, source values position-shuffled
   (seeded, per-trajectory). Content-specificity = `E-a{α} − P-a{α}` — is the coding effect
   content-carried here, not generic perturbation.
3. **Per-region layer profiling** (heavier): partition the 48 layers into 8 contiguous
   fractional-depth regions; graft each region alone at α=1.0 as `R{k}`, giving each
   region's marginal next-action recovery. A CPU step builds a **SWE-Gym-tuned per-layer
   champion** = union of the regions with positive **TUNE-split** marginal, then a second
   harness pass **evaluates it held-out** on the eval split (`E-champion` vs scalar
   `E-tuned` vs `B`).
4. **Held-out discipline** (critical at N≈75): every trajectory carries a deterministic
   `split` = tune|eval from `sha256(seed:idx)` parity. α is *selected* and the layer
   champion *built* on **tune**; both are *reported* on the **disjoint eval** split. The
   in-sample (all-trajectory) α curve is also printed, clearly labelled as descriptive, not
   the held-out claim.
5. **Provenance**: unique stamped output dir per run; each result records the arm
   intervention (α or champion path+hash+label), the split, and a born-annotated manifest.

### Overfitting caveat (stated honestly)
Only ~75 trajectories pass the cut, so each split is ~35–40. Selecting α (one global
scalar) on ~38 and confirming on ~37 is thin but legitimate; a *layer* champion has many
degrees of freedom, so its held-out eval is the one that matters — an in-sample win that
does not survive the eval split is overfitting, and the analysis reports it as such. The
failed synthetic-champion transfer mildly suggests the graft may be **config-insensitive**
here (α and layer-set barely matter, only the presence of the graft), in which case the
result is "flat — nothing beats the naive scalar." That is a real, reportable outcome.

## Loss lesson applied
Every run writes to a **unique, stamped** directory (`results/swegym_tune_<STAMP>_brief`,
`results/swegym_champeval_<STAMP>_brief`) and a stamped champion config
(`data/champion_configs/swegym_tuned_<STAMP>.json`). The job **never** writes to
`results/swegym_30b_bf16_{brief,prod}` and never reuses a prior run's filenames — the
prod-champion data was lost precisely because a re-run reused those paths and a cp-backup
created a revert path. No in-place overwrite of any prior run.

## Staged cost estimate (one A100 80GB @ ~$1.39/hr)
The α-sweep, placebo, and region arms are **score-only** (teacher-forced logprob, no
200-token generation), so they add little on top of the per-trajectory prefills. Rough:

- **Stage `sweep`** (α-sweep + placebo, one pass, analysis): ~75 trajectories × ~60–90 s ≈
  **1.0–1.5 h ≈ $1.5–2.5**. High-value, do first — it alone answers "is 0.75 optimal / can
  we beat +0.013 / is it content-specific."
- **Stage `full`** (adds region profiling in the same first pass + a second held-out
  champion-eval pass): ~2 passes ≈ **2.5–3.5 h ≈ $4–5** total.

(Estimates; the run prints per-trajectory wall time so the first few calibrate it.)

## How to run — exact launch commands (ready to fire when funded)

**Stage 1 — α-sweep + placebo (cheapest, do first):**
```bash
SC_HF_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
SC_LOAD_DTYPE=bfloat16 \
SC_SUMMARY=brief \
SC_STAGE=sweep \
SC_E_ALPHAS="0.25,0.5,0.75,1.0,1.5" \
SC_SWE_PLACEBO=1 \
scripts/launch_pod.sh swegymtune scripts/job_swegym_tune_bf16.sh
```

**Full — α-sweep + placebo + per-region profiling + held-out champion eval:**
```bash
SC_HF_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
SC_LOAD_DTYPE=bfloat16 \
SC_SUMMARY=brief \
SC_STAGE=full \
SC_E_ALPHAS="0.25,0.5,0.75,1.0,1.5" \
SC_SWE_PLACEBO=1 \
SC_PROFILE_REGIONS=8 \
scripts/launch_pod.sh swegymtune scripts/job_swegym_tune_bf16.sh
```

(`launch_pod.sh swegymtune ...` writes `.pod_swegymtune_state.json`; the new knobs are
forwarded by `launch_pod.sh`. `swegym.parquet` must be staged locally so it rsyncs to the
pod. Re-running analysis afterward is free: `python3 src/analyze_swegym_tune.py --run
results/swegym_tune_<STAMP>_brief [--champion-eval results/swegym_champeval_<STAMP>_brief]`.)

## How to read it
The analysis prints, for the held-out **eval** split unless noted:
- **scalar baseline** `E-tuned − B` (all/tune/eval) — the +0.013 to beat.
- **α curve**: per-α `E-a − B` (in-sample ALL, and held-out EVAL) + content-specificity
  `E-a − P-a`. Then the **α selected on TUNE** and its **held-out EVAL** CI vs scalar 0.75.
- **region marginals** on TUNE, the champion layer-set built, and (full mode) the
  **held-out champion**: `E-champion − B` and the paired `E-champion − E-tuned`.
- **Decision rules**: a tuned α or the layer champion *wins* only if its held-out EVAL CI
  clears 0 **and** beats scalar; if the head-to-head CI spans 0, the in-domain tuning added
  nothing and the naive scalar is the robust form — report that plainly.

## Drafted future-work paragraph for the paper's §10 (for review — not yet inserted)

> **Optimize the graft where it works.** The one regime where value grafting is net-positive
> — real coding trajectories under brief summaries — is the one regime we never tuned for.
> The only grafts we optimized were fit on the synthetic conversation corpus, where the
> effect is null, and they transferred to the coding task adding nothing over a flat α=0.75.
> The obvious next experiment is an in-domain one: sweep the graft strength and profile the
> per-layer contribution directly on held-out coding trajectories, with a shuffled-source
> placebo and a strict tune/evaluate split, and ask whether a graft fit to this regime beats
> the naive +0.013 nats/token. We designed and built this optimization but did not run it,
> and we flag two honest possibilities in advance. It may find a better operating point,
> tightening a real if narrow effect. Or it may come back flat — the graft largely
> insensitive to α and layer choice, so that only the presence of the graft matters and the
> naive scalar is already the best form — which the failure of the synthetic-tuned champion
> to transfer mildly anticipates. Either way the honest version of this result requires
> measuring **task success** (apply the action, run the tests, report resolve rate), not the
> log-probability proxy: an optimized proxy effect is only worth reporting as a technique if
> it moves the real thing. The narrowness is the point — the value of this direction is in
> characterizing the one live margin precisely, not in resurrecting the general claim the
> rest of the paper bounds to null.

## Files
- `src/run_swegym_hf.py` — score-only α-sweep / placebo / per-region-profile arms;
  deterministic per-trajectory tune|eval split; manifest records the optimization surface.
- `src/analyze_swegym_tune.py` — CPU held-out analysis: α curve + tune-select→eval-report,
  content-specificity, region marginals, champion build, held-out champion eval.
- `scripts/job_swegym_tune_bf16.sh` — staged job (`sweep` default; `full` adds profiling +
  held-out champion eval); unique stamped dirs.
- `scripts/launch_pod.sh` — forwards the new envs (`SC_STAGE`, `SC_E_ALPHAS`,
  `SC_SWE_PLACEBO`, `SC_PROFILE_REGIONS`, `SC_PROFILE_ALPHA`).
