# E01 token-level decomposition of the two-token focal targets

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning  
**Date:** 2026-07-12  
**Status:** zero-GPU descriptive re-analysis of the committed raw artifact

The committed e01 package preserves per-token float32 log probabilities for
both two-token focal targets. I reconstructed the 8,897,066-byte raw JSON from
the committed gzip/base64 package, reverified its SHA-256, and decomposed the
primary R2 correct-history-minus-wrong-history contrasts. The generated result
is:

`results/coherent_canary_v12_analysis/coherent-canary-v12-e01-token-decomposition_Qwen3-30B-A3B-Instruct-2507_20260712T045020Z.json`

Recompute with:

```bash
uv run python scripts/analyze_coherent_canary_v12_e01_tokens.py
```

## Result

Each row below shows the semantic margin difference separately for the first
and second target token, followed by their mean. The first token directly
contrasts the initial choice between `partner` and `staff`; the second token is
conditional on having forced the first token of each phrase.

| Schedule | Family | Token 1 | Token 2 | Mean |
|---|---|---:|---:|---:|
| N | full K+V, CC-WW | +0.218750 | -0.024277 | +0.097237 |
| N | value-only, FC-FW | +0.156250 | +0.192841 | +0.174545 |
| P | full K+V, CC-WW | -0.031250 | +0.109041 | +0.038896 |
| P | value-only, FC-FW | **-0.187500** | **+0.229593** | **+0.021047** |

The N/value-only cell is internally consistent across both target tokens. The
P/value-only phrase average is not: its first-token choice moves in the wrong
semantic direction, and a positive conditional second-token contribution
slightly outweighs it. Full K+V under P has the same qualitative split.

This sharpens, but does not change, the final interpretation. The already-small
P mean should not be described as a schedule-robust replication of the N
direction. At the actual initial answer choice it reverses. A later conditional
token can compensate in a teacher-forced phrase score even when free generation
would never enter that phrase, which is especially relevant because every
primary arm freely generated the same unrelated `Ring 3` answer.

The script computes these token rows directly from stored float32 token
log-probabilities. The main harvester uses separately persisted float32-rounded
phrase means, so correct-target component values can differ at about 1e-6 while
the semantic phrase-margin contrasts agree to the displayed precision.

This analysis adds no independent case, control, or inference. It further
supports the existing classification of e01 as a schedule-sensitive,
behaviorally null, uncontrolled mechanistic hint.
