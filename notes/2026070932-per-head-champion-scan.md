# Per-head champion scan @ bf16 on the primary 30B MoE

**Author:** Fable (design + code), 2026-07-09. Runnable spec + committed harness.
Coordinator (Claude) launches the pods with the exact commands below.

---

## TOPLINE

Our current champion value-graft is **per-LAYER** (`layers_30b_bf16.json`: an
alpha-map, all kv-heads of each selected layer). This experiment asks, at **bf16
full resolution on the PRIMARY model** (`Qwen/Qwen3-30B-A3B-Instruct-2507`, a MoE,
48 layers x 4 KV heads): can a **per-HEAD** champion (a subset of `(layer,
kv-head)` slots) do BETTER, and does a **combination** (intersection / union of
the per-head and per-layer selected slots) beat both? We **DERIVE** the per-head
selection on the tuning **VAL** set (c01-c06,n01-n04) only, then **VALIDATE** all
four configs — per-head, per-layer, intersection, union — on the **disjoint
held-out** recovery plants (c07-c24) with the **same** placebo-controlled,
conversation-clustered `E - placebo` CI used for the per-layer champion, so the
numbers are directly comparable. **We do not pre-commit to per-head winning** —
the held-out `content_specificity.e_minus_placebo` CI is the sole arbiter, and if
per-head ties or loses we report exactly that.

**Launch (one A100 80GB pod, unattended):**
```bash
cd /Users/jeb/experimentation
MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507" python3 scripts/preflight.py \
    --job scripts/job_headscan_bf16.sh --launcher scripts/launch_pod.sh
MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507" \
SC_HF_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507" \
SC_TUNE_TAG="30b_bf16" SC_LOAD_DTYPE="bfloat16" \
SC_CONV_START="6" SC_CONV_LIMIT="18" \
  bash scripts/launch_pod.sh headscan scripts/job_headscan_bf16.sh "NVIDIA A100 80GB PCIe"
```

---

## THE QUESTION AND WHY IT IS A LEGITIMATE DOUBLE-CHECK

The per-layer champion grafts every kv-head of each selected layer. That is
coarse: within a "good" layer some kv-heads may carry the semantic value signal
and others may be neutral or slightly harmful; and a "dead" layer may host one
useful head. A per-head champion can be **more precise** (fewer slots, less
injected noise) or **more complete** (rescue a head in an unselected layer). If a
per-head selection genuinely improves the held-out, placebo-controlled recovery,
we should adopt it — "leave no better champion on the table." If it does not, the
per-layer champion stands and we have ruled out a plausible refinement rigorously.

## THE MECHANISM / MACHINERY (verified against the harness)

- **Derive** — `src/run_tune_hf.py` `PHASE=head` already profiles every `(layer,
  kv-head)` slot: blend that ONE slot's old V into the compacted baseline at
  alpha=1, teacher-force the held-out continuation, score `- B`. Runs on VAL
  (c01-c06,n01-n04). Output `results/tune_head_<tag>/<conv>.json` with keys
  `{"B", "L<l>H<h>":score,...}` (192 slots at 48x4). This is the exact HF/bf16-30B
  port of the local-4B `src/head_profile.py`, already in-repo — no new derive code
  needed.
- **Select** — `src/head_select.py` (new): mean each slot's marginal `dEB = score
  - B` over VAL convs, apply ONE pre-registered rule, emit a `head_map` champion.
- **Apply** — `src/kvlib_hf.py:blend_values(..., head_map=...)` blends ONLY the
  selected kv-heads' V rows (`v2[:, h, new_idx, :] = mixed[:, h]`); keys are never
  touched (alpha_K = 0 structurally). `cross_arch_probe.py` resolves a `head_map`
  config (`:2136`) and applies the SAME `graft_head_map` to the real graft E
  (`:2559`) AND the placebo (`:2593`) — confirmed. So E vs placebo differ ONLY in
  source values, at the identical slots/positions/alpha.
- **Validate** — `cross_arch_probe.py` on held-out c07-c24: `raw_EB` (E-B),
  `placebo`, and the decisive paired `content_specificity.e_minus_placebo`
  (conversation-clustered bootstrap 95% CI), identical to `job_champion_bf16.sh`.

## SELECTION RULE + THE THREE CONFIGS

**Pre-registered rule (default `SC_HEAD_SELECT=positive`):** select every `(layer,
kv-head)` slot whose **mean VAL marginal dEB > 0**, grafted at
`SC_HEAD_ALPHA=1.0`. This is the direct per-head analogue of the per-layer
derivation (which also ranks by single-unit VAL marginal). Alternatives
(`topn`/`topfrac`) exist via env but only ONE rule is ever run, so only one
held-out test is spent (see overfitting guard).

Emitted to `data/champion_configs/`:
- `heads_30b_bf16.json` — the per-head champion (`head_map` + `alpha`).
- `heads_x_layers_30b_bf16_intersection.json` — per-head slots **restricted to
  layers the per-layer champion also selected** (both families agree the layer
  carries content). Conservative combination.
- `heads_x_layers_30b_bf16_union.json` — per-head slots **UNION every kv-head of
  every per-layer-champion layer**. Inclusive combination.

**Combination granularity / honest caveat:** both combined configs are `head_map`
form at a single uniform alpha (1.0). The champion schema forbids carrying an
`alpha_map` and a `head_map` together, so the union/intersection graft their SLOTS
at one alpha and cannot reproduce the per-layer champion's 0.75-vs-1.0 gradation.
The per-layer champion is therefore validated SEPARATELY with its own alpha-map
for the side-by-side. Which of intersection/union we would adopt is decided by the
held-out CI, reported honestly — not pre-chosen.

## WHY THIS IS OVERFITTING-SAFE

1. **Derive on VAL, validate on DISJOINT held-out.** Selection sees c01-c06,n01-n04
   only; every reported CI is on c07-c24, never selected on. Content-specificity
   cannot be a tuning artifact.
2. **Multiple-comparisons control.** The 192 single-slot marginals are used ONLY to
   RANK (they are the VAL profile). Exactly ONE selection rule fires, producing a
   fixed, small config set (heads / intersection / union), and each config gets
   exactly ONE held-out placebo-controlled validation. The selection budget is
   thus the handful of configs here, NOT 192 — stated plainly rather than hidden.
3. **Paired placebo cancels difficulty.** The decisive number is per-plant `lp_E -
   lp_E_placebo` through the SAME slots/positions/alpha, so a possibly-overfit
   config is compared against ITS OWN placebo — overfit inflates both equally and
   cancels; only genuine content-specificity survives.
4. **Three placebos** (`shuffle_pos` primary, `shuffle_probe` strongest content
   control, `gauss` energy-matched). A real champion must beat all three.

## HOW TO RUN

The single launch command is in TOPLINE. The job (`scripts/job_headscan_bf16.sh`)
does, in one unattended pod session:
- STAGE 1 derive: `run_tune_hf.py PHASE=head` on VAL at bf16.
- STAGE 1b select: `head_select.py` -> the three configs above (VAL-only).
- STAGE 2 validate: `cross_arch_probe.py`, placebo-controlled, held-out c07-c24,
  for EACH of {heads, layers, intersection, union} x {shuffle_pos, shuffle_probe,
  gauss}. Shared `--out-dir` => the expensive render is generated ONCE (champion
  config is in the SCORE fingerprint only, never the render fingerprint) and
  reused across all 12 runs; each result copied to a unique
  `results/champion_validate/headscan_<tag>_<config>_<mode>_<UTC>.json`.
- FINAL SUMMARY: a side-by-side table of `E-B` and `E-placebo [lo,hi]` for all four
  configs x three placebos, flagging every `CI>0`, and naming the best config on
  the primary placebo.

bf16 30B loads on the pod's stock torch 2.4.1 — **NO torch upgrade** (that was only
a 4-bit/compressed-tensors issue). Only `transformers>=4.57,<5` is pinned.

## HOW TO READ THE RESULT (answers the question)

In each result JSON, `content_specificity.e_minus_placebo = {mean, lo, hi, n,
n_clusters}` on the held-out set. The job prints them side by side.

- **Does per-head beat per-layer?** Compare the `heads` row vs the `layers` row on
  the primary placebo (`shuffle_pos`): higher `E-placebo` mean AND `lo > 0` = a
  better, content-specific champion. If `heads.lo <= 0` while `layers.lo > 0`, or
  the CIs overlap heavily with a lower/equal mean, **per-head does NOT beat
  per-layer** — report that and keep the per-layer champion.
- **Best overall champion** = whichever of {heads, layers, intersection, union} has
  the **highest `E-placebo` mean with `lo > 0`** on `shuffle_pos`, corroborated by
  `shuffle_probe`/`gauss` (positive, ideally CI>0) and by `raw_EB` (E-B) > 0 so it
  recovers meaning at all. If the intersection wins, the refinement is "prune the
  per-layer graft to its content-carrying heads"; if the union wins, "add the
  extra useful heads the per-layer champion missed"; if `layers` wins, the coarse
  champion was already optimal.
- **Null outcome is a real outcome:** every config can be positive vs B yet
  indistinguishable from its placebo (CI spans 0) — that is the honest "per-head
  refinement buys nothing content-specific" verdict, reported as-is.

## FILES

- `src/head_select.py` (new) — VAL-only selection + intersection/union configs.
- `scripts/job_headscan_bf16.sh` (new) — derive -> select -> held-out placebo
  validate -> side-by-side summary.
- `scripts/job_swegym_bf16.sh` (new, SECONDARY) — see below.
- Derive already in `src/run_tune_hf.py` (`PHASE=head`); apply/validate already in
  `src/cross_arch_probe.py` (`head_map` path) and `src/kvlib_hf.py:blend_values`.
- Emitted at runtime (commit after the run): `data/champion_configs/heads_30b_bf16.json`,
  `heads_x_layers_30b_bf16_intersection.json`, `heads_x_layers_30b_bf16_union.json`,
  and the per-head VAL profile `results/tune_head_30b_bf16/*.json`.

## SECONDARY: SWE-Gym anchor strengthening (spec, lower effort)

`scripts/job_swegym_bf16.sh` replicates the **+0.0156 SWE-Gym anchor** (E-tuned vs
B, `SC_SUMMARY=brief`, `SC_E_ALPHA=0.75`) at higher N (default 150 vs prior 75)
with a **trajectory-bootstrap 95% CI** on the paired `E-tuned - B` tf-logprob
difference; CI lower bound > 0 => the anchor replicates. `src/run_swegym_hf.py` has
no source-corrupted placebo arm (that would need a harness extension), so B is the
honest within-trajectory control here — stated in the job. **Prereq:**
`swegym.parquet` must be staged locally (launch_pod rsyncs it) and pyarrow installs
on the pod. Launch:
```bash
cd /Users/jeb/experimentation
MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507" python3 scripts/preflight.py \
    --job scripts/job_swegym_bf16.sh --launcher scripts/launch_pod.sh
MODELS="Qwen/Qwen3-30B-A3B-Instruct-2507" \
SC_HF_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507" \
SC_LOAD_DTYPE="bfloat16" \
  bash scripts/launch_pod.sh swegym scripts/job_swegym_bf16.sh "NVIDIA A100 80GB PCIe"
```

## RISKS / CAVEATS

- **Head-marginal composition is nonlinear** (the harness warns single-region raw
  EBs don't sum to the joint). Selecting positive-marginal slots is a candidate,
  not a guarantee the joint graft helps — which is exactly why the held-out placebo
  CI, not the VAL profile, is the arbiter.
- **`positive` rule could select many slots** if most heads are marginally positive,
  making per-head close to a dense graft. The summary reports the slot count;
  `SC_HEAD_SELECT=topn`/`topfrac` are available if a sparser candidate is wanted
  (but re-running a second rule spends a second held-out test — do so only
  deliberately and report the budget).
- **swegym.parquet is not in the repo** and was absent locally at authoring time;
  the swegym job hard-fails loud if it is missing rather than silently skipping.
- The per-layer champion is ALSO being validated on `.pod_champbf16_state.json`;
  this job re-validates it in-band purely for a same-render, same-held-out
  side-by-side. The two should agree within CI — a useful cross-check.
