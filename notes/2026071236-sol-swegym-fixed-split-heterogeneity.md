# SWE-Gym fixed-scalar hash-split heterogeneity audit

**Author:** OpenAI GPT-5.6 Sol (extra-high reasoning)

**Date:** 2026-07-12
**Status:** zero-GPU, post-hoc descriptive audit prompted by M1 in the final skeptical Fable review

## Bottom line

The fixed value-graft at scalar α = 0.75 had materially different point estimates on the two pre-existing hash halves of the later 98-row, 4,096-token-chunked SWE-Gym pool:

| Hash subset | n | Fixed scalar E−B (nats/target token) | Ordinary trajectory-bootstrap 95% CI | Positive / negative rows |
|---|---:|---:|---:|---:|
| `tune` | 41 | −0.014643 | [−0.037651, +0.004817] | 21 / 20 |
| `eval` | 57 | +0.007629 | [−0.004208, +0.019280] | 32 / 25 |
| `eval − tune` | — | **+0.022272** | **[−0.000396, +0.046494]** | — |

This verifies the arithmetic behind Fable's observation: the point-estimate swing for the *same fixed intervention* is about 0.022 nats/token, larger than the selected map's descriptive pooled-102 mean of +0.0135. But the correct two-subset bootstrap interval barely spans zero. This audit therefore does **not** detect a tune/eval difference at the ordinary nominal 95% level. It documents post-hoc subset heterogeneity whose scale is consequential for interpreting small likelihood movements.

## Exact method

- Metric per saved trajectory row: `E-tuned.tf_mean − B.tf_mean`, in natural-log units per demonstrated target token.
- Input measurement: the later chunked profile run under the brief summary request, `results/swegym_tune_20260710T145330Z_brief/`.
- The deterministic split was already written into every row: parity of the first eight hexadecimal digits of `sha256("20260710:<dataset-index>")`, with even = `tune` and odd = `eval`.
- The profile and champion-evaluation directories contain exactly identical parsed fixed-scalar E−B deltas for all 98 IDs. This analysis counts those scalar measurements once, from the profile directory.
- Each subset interval is the existing canonical ordinary percentile bootstrap of the arithmetic mean: 10,000 replicates, Python `random.Random(0)`, sampling trajectory rows with replacement, using sorted dataset-index order.
- The `eval − tune` interval independently resamples 41 tune rows and 57 evaluation rows on each replicate, then records `mean(eval) − mean(tune)`.
- No multiplicity adjustment was applied. This comparison was requested after inspection of the paper and review, so it is explicitly post hoc.

The earlier independent analysis program `src/analyze_swegym_tune.py` reproduces both subset means and marginal intervals. The extended canonical analyzer now records the subsets, the independent contrast, the map relationship, and the interpretive boundary under schema `swegym-paper-metrics-reanalysis/v3`.

## Relationship to the selected layer map

The relationship is real but narrower than a casual reading of “selected on the half where the scalar looked worst” could suggest.

| Subset | Fixed scalar − B | Selected map − B | Selected map − fixed scalar |
|---|---:|---:|---:|
| Tune41, **selection-exposed** | −0.014643 [−0.037651, +0.004817] | +0.010273 [+0.003498, +0.016716] | +0.024916 [+0.007961, +0.044996] |
| Eval57, **out of fitting for the map** | +0.007629 [−0.004208, +0.019280] | +0.011742 [+0.006293, +0.017155] | +0.004113 [−0.004742, +0.012889] |

The layer map was built on the tune41 rows, but **not by selecting against the fixed scalar**. The selection rule kept the contiguous layer regions whose separate `R{k} − B` tune means were positive: R2 (layers 12–17) and R5 (layers 30–35). The fixed α = 0.75 tune/eval contrast was not the selection criterion. Thus:

1. It is accurate to say that map selection used the same hash half on which the fixed-scalar point estimate was negative.
2. It is inaccurate to imply that the negative fixed-scalar result itself caused or optimized the map selection.
3. The tune41 map numbers are selection-exposed descriptions, not validation evidence.
4. On the out-of-fitting eval57 rows, the selected map's +0.011742 movement exceeded the fixed scalar's +0.007629 by only +0.004113, with the paired interval spanning zero. The selected map therefore did not show a detected advantage over the fixed scalar on that split.
5. Comparing the +0.022272 split swing with the pooled-102 selected-map mean is scale context only. They are different estimands, and the pooled-102 result combines eval57 with a separate original-pool confirmation prefix of 45.

## Interpretation boundary

This result strengthens the paper's caution about small teacher-forced likelihood movements. It does not by itself establish that the intervention effect truly differs between random halves: both marginal subset intervals span zero, and the direct interval spans zero by about 0.0004 nats/token. It also does not identify a cause. Dataset-index/task composition, target properties, summary realization, correlated tasks, or chance could contribute. This audit uses the canonical row-level trajectory bootstrap; task/repository-cluster sensitivity is a separate analysis and should not be silently conflated with it.

The strongest safe sentence is:

> In a post-hoc split of the later chunked 98-row pool, the same fixed scalar graft averaged −0.0146 nats/token on the 41 map-fitting rows and +0.0076 on the 57 held-out rows; their +0.0223 eval-minus-tune contrast had a nominal row-bootstrap 95% interval of [−0.0004, +0.0465].

This belongs, if used, beside—not in place of—the stronger controls on interpretation: selected-map versus fixed-scalar on eval57 was inconclusive; structural behavior was a wash; the brief summary was intentionally detail-stripping; and no matched selected-map placebo or task-success endpoint ran.

## Reproducible artifact

- Extended analyzer: `scripts/analyze_swegym_paper_metrics.py`
- Focused tests: `tests/test_analyze_swegym_paper_metrics.py` (7 passed, including the exact 10,000-replicate interval endpoints)
- New v3 artifact: `results/swegym_paper_reanalysis/swegym_paper_metrics_Qwen3-30B-A3B-Instruct-2507_20260712T064340Z.json`
- V3 artifact SHA-256: `576ee4133dd6902c2caf3140280a3f79c0f0ab6cbb295f7631f6c69a6f97197c`
- Analyzer SHA-256 recorded inside the artifact: `e8fcdf9f66f0f29944b5ef5241f298aa72cf0011e4ad0d4754c0a2264cc1b6ec`
- Superseded only by extension, not overwritten: canonical v2 artifact `results/swegym_paper_reanalysis/swegym_paper_metrics_Qwen3-30B-A3B-Instruct-2507_20260712T060222Z.json`, SHA-256 `3b577f44af4aa46c6569201b9405c93e3b815c7af128efdde7c783d1193657e4`
- New GPU spend: $0.00. External model calls: 0.
