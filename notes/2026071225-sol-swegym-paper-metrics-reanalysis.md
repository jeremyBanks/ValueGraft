# Zero-cost SWE-Gym paper-metric recomputation

> **SUPERSEDED 2026-07-12:** This note treated two different prefill schedules
> as repeated measurements and averaged them. That was methodologically invalid.
> See `notes/2026071295-sol-swegym-prefill-schedule-correction.md` and the v2
> canonical artifact stamped `20260712T060222Z`. This historical note is retained
> to preserve the correction trail and must not be cited for fixed-scalar pooling.

**Author:** Sol — OpenAI GPT-5.6 Sol, extra-high reasoning  
**Date:** 2026-07-12  
**Scope:** CPU-only recomputation from committed JSON score artifacts; no GPU, paid API, or external-model call.

## Authoritative artifact

The reusable analysis is `scripts/analyze_swegym_paper_metrics.py`. Its paper-facing output is:

- `results/swegym_paper_reanalysis/swegym_paper_metrics_Qwen3-30B-A3B-Instruct-2507_20260712T051816Z.json`
- output SHA-256: `7f1b96509931e462fa6311768873599761fb32cfc6e1cc985a9d77cbf6941a04`
- analysis-script SHA-256 recorded in the output: `edae7126b55f3d8cec8f54df47c7d304ea7f65146665b38f407ba8d596fa9066`
- bootstrap: ordinary trajectory-level percentile bootstrap, 10,000 replicates, Python `random.Random`, seed 0; the exact quantile indices and pool-concatenation order are recorded in the JSON.

The output inventories every consumed score/manifest/config path and its SHA-256, verifies all observed subject-model IDs, verifies the frozen selected-map hash in all 143 rows that carry `E-champion`, proves the 41 fitting IDs do not overlap either out-of-fitting component, and fails closed on the expected 75/98/45 counts.

## Independently recomputed results

All continuous quantities are paired per-trajectory differences in saved teacher-forced mean log-probability, in nats per target token.

| Selected-map contrast | N | Mean | Bootstrap 95% CI | Positive rows |
|---|---:|---:|---:|---:|
| Fresh eval, selected map − baseline | 57 | +0.011742 | [+0.006293, +0.017155] | 44/57 |
| Original-pool partial confirm, selected map − baseline | 45 | +0.015836 | [+0.006094, +0.025990] | 33/45 |
| Pooled wholly out-of-fitting, selected map − baseline | 102 | +0.013548 | [+0.008335, +0.019087] | 77/102 |
| Fresh eval, selected map − fixed scalar | 57 | +0.004113 | [−0.004742, +0.012889] | 32/57 |
| Pooled 102, selected map − fixed scalar | 102 | +0.001679 | [−0.005643, +0.008825] | 54/102 |

The pooled out-of-fitting structural demonstration-action match is exactly null: baseline 53/102, selected map 53/102, with **3 fixes and 3 breaks**; paired difference 0.0000, bootstrap CI [−0.04902, +0.04902]. The fresh 57 contribute 2 fixes/3 breaks; the partial original 45 contribute 1 fix/0 breaks. This metric is parser-level agreement with the recorded demonstrated action, not unique correctness or executed task success.

For the fixed scalar α = 0.75 graft:

| Fixed-scalar estimate | N | Mean | Bootstrap 95% CI |
|---|---:|---:|---:|
| Original run 1 | 75 | +0.015643 | [+0.004912, +0.027094] |
| Original run 2 | 75 | +0.013331 | [+0.001565, +0.026337] |
| Original pool, two repeats averaged within trajectory | 75 | +0.014487 | [+0.004029, +0.025690] |
| Later disjoint-index pool | 98 | −0.001689 | [−0.013558, +0.008931] |
| Unique pooled trajectories | 173 | +0.005324 | [−0.002756, +0.013120] |

The pool contrast, later 98 minus repeat-averaged original 75, is −0.016176 with an independently resampled two-pool bootstrap interval [−0.032282, −0.000743]. This supports **sample/pool heterogeneity**; it does not identify whether the cause is domain variation, index/source confounding, or another unrecorded difference.

## Interpretation and limits

The selected map has a small continuous out-of-fitting lead versus the compacted baseline across the completed 57+45 rows. It does **not** reliably beat the fixed scalar head-to-head, and its pooled structural match does not move at all. The fixed scalar is positive in the repeated original 75, absent in the later 98, and null after pooling 173 unique trajectories. These facts support the paper's narrow “likelihood lead, behaviorally unresolved” description; they do not support coding-agent improvement or semantic recovery.

The result remains control-incomplete: no selected-map-matched placebo exists. The original confirmation stopped at 45 of 75 and is a stopped prefix, not a randomized completed sample. The historical first original run lacks a manifest and row-level dtype annotation. The saved score rows omit the generated summary text, target text/token IDs, K/V tensors, stable task-cluster IDs, and row-level upstream generator identity. Accordingly this is an exact recomputation of committed **scores**, not a reproduction of inference or a task-clustered analysis.

## Reproduction and validation

```bash
uv run python scripts/analyze_swegym_paper_metrics.py
uv run pytest -q tests/test_analyze_swegym_paper_metrics.py
```

Observed focused test result: `3 passed`.

Two earlier outputs in the same directory, stamped `20260712T051445Z` and `20260712T051600Z`, are preserved as preliminary row-order sensitivities. The first globally sorted the two selected-map pool components; the second exposed that some fixed-scalar components inherited lexicographic file insertion order. Point estimates are identical and interval endpoints differ only through seeded finite-bootstrap Monte Carlo ordering. The `051816Z` artifact is authoritative: every component is sorted numerically by saved dataset index, then pooled in the explicitly recorded component order.
