# SWE-Gym fixed-scalar prefill-schedule correction

**Author:** OpenAI GPT-5.6 Sol, extra-high reasoning  
**Date:** 2026-07-12

## What was wrong

The first paper reanalysis averaged two α = 0.75 fixed-scalar measurements on the same 75 trajectory IDs as though they were repeats under one apparatus. They were not. `results/swegym_30b_bf16/` was run and committed before commit `317a0dd1a13a25b614e867a39270ee176e73926f` introduced 4,096-token chunked prefill. `results/swegym_30b_bf16_brief/` ran on July 10 under the later chunked path. The disjoint 98-row pool also used chunked prefill.

This difference is not ignorable here: the project independently observed that query shape can alter deterministic bf16 state. The two 75-ID runs also produced different summary lengths for 64/75 IDs; t0001 changed from 84 to 131 summary tokens. Their within-ID graft-effect contrast, chunked minus legacy single-call, was −0.002312 nats/token with nominal bootstrap 95% CI [−0.010999, +0.006501]. This is an apparatus contrast, not an isolated causal estimate of schedule, because summary text/state and every downstream score were regenerated.

## Corrected fixed-scalar results

All intervals below use the same 10,000-replicate trajectory bootstrap, seed 0.

| Condition | N | Mean E − B (nats/token) | Nominal 95% CI | Positive |
|---|---:|---:|---:|---:|
| Legacy single-call original pool | 75 | +0.015643 | [+0.004912, +0.027094] | 45/75 |
| Chunked original pool | 75 | +0.013331 | [+0.001565, +0.026337] | 46/75 |
| Chunked disjoint pool | 98 | −0.001689 | [−0.013558, +0.008931] | 53/98 |
| Schedule-matched chunked pool | 173 | +0.004823 | [−0.003642, +0.013045] | 99/173 |

The schedule-matched disjoint-minus-original pool contrast is −0.015020, nominal 95% CI [−0.031931, +0.001510]. It spans zero. The prior interval [−0.032282, −0.000743] excluded zero only because the invalid cross-schedule average was used as the original-pool comparator. We may describe heterogeneous point estimates across pools, but not a detected pool difference.

The selected-layer-map results are unchanged: their 41/57 fitting/evaluation split and 45-row partial confirmation all came from the later chunked apparatus. The correction concerns only the fixed-scalar original/disjoint pooling and the interpretation of the two original 75-ID runs.

## Canonical artifact

- `results/swegym_paper_reanalysis/swegym_paper_metrics_Qwen3-30B-A3B-Instruct-2507_20260712T060222Z.json`
- SHA-256 `3b577f44af4aa46c6569201b9405c93e3b815c7af128efdde7c783d1193657e4`
- Schema `swegym-paper-metrics-reanalysis/v2`

Reproduce:

```bash
uv run python scripts/analyze_swegym_paper_metrics.py
uv run pytest -q tests/test_analyze_swegym_paper_metrics.py
```

The older v1 artifacts remain tracked for auditability but are explicitly superseded.
