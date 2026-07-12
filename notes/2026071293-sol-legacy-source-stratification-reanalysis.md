# Legacy synthetic source-stratification reanalysis

**Author:** Sol — OpenAI GPT-5.6 Sol, extra-high reasoning  
**Date:** 2026-07-12 UTC  
**Status:** observed zero-GPU recomputation; exploratory, not a causal source-family comparison

## Why this was necessary

The legacy headline table pools c07–c24 even though the conversation bodies came
from two materially different sources: c07–c12 were rendered by
`mlx-community/Qwen3-4B-Instruct-2507-4bit`, while c13–c24 were authored by
assistant sessions whose files retain only `sonnet`/`opus`/`fable` aliases. A pooled interval alone makes this evidence
look more homogeneous than it is.

I added `src/analyze_legacy_source_split.py`, which reads the exact four saved
headline result files, verifies the complete c07–c24 conversation set, preserves
the original plant-weighted `raw_EB = lp_E - lp_B` estimand, and repeats the
original conversation-clustered percentile bootstrap (10,000 replicates, seed
42). The machine-readable output records every input path and SHA-256:

`results/legacy_source_stratification/legacy-source-stratification_Qwen3-30B-A3B-Instruct-2507_20260712T052531Z.json`

SHA-256: `df4c1a74e110e0add4c74e621fe8a0920d78e95829a869612e00cc2ebc18e196`.

## Observed results

| Configuration | c07–c12: 4B-rendered (6 conversations / 57 plants) | c13–c24: assistant-authored, stored aliases only (12 conversations / 146 plants) | Pooled c07–c24 |
|---|---:|---:|---:|
| Per-head | +0.071 [−0.031, +0.171] | −0.004 [−0.044, +0.039] | +0.017 [−0.027, +0.062] |
| Per-layer | **+0.120 [+0.055, +0.181]** | **−0.050 [−0.076, −0.028]** | −0.003 [−0.041, +0.043] |
| Intersection | **+0.114 [+0.035, +0.176]** | **−0.033 [−0.067, −0.004]** | +0.008 [−0.032, +0.052] |
| Union | +0.064 [−0.025, +0.154] | **−0.041 [−0.080, −0.004]** | −0.012 [−0.053, +0.034] |

Values are nats/token with nominal 95% bootstrap intervals, conditional on the
realized conversation bodies. Minor rounding in this table is from the exact
JSON values.

## Interpretation

The pooled null is a correct arithmetic description of the exact mixed target
population, but it is not a homogeneous null. The largest configuration changes
from positive in the six 4B-rendered conversations to negative in the twelve
Claude-authored conversations.

This split does **not** identify a causal model-nativeness effect. The blocks also
differ in conversation length, authoring procedure, plant construction, and
similarity to the Qwen-rendered tuning corpus; only six clusters support the
positive block. The defensible reading is that the legacy result is strongly
corpus/source-conditional and may reflect tuning-distribution similarity,
length, construction, source family, or some combination. It neither proves
nativeness nor licenses a universal negative claim. The final paper must show
this heterogeneity alongside the pooled table.
