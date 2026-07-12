# Graft alignment-dose recovery audit

**Author:** OpenAI GPT-5.6 Sol, extra-high reasoning
**Date:** 2026-07-12

## Verdict

The legacy headline alignment dose is exactly recoverable from committed repository
artifacts. The SWE alignment dose is not fully recoverable from a clean clone. A
useful partial SWE reconstruction is possible only because the ignored local
`swegym.parquet` is still present and exactly matches the hash recorded in the paper.

This distinction is substantive:

- The legacy graft was not sparse. Across c07--c24, every summary and tail region was
  one full contiguous difflib block and 100% of eligible non-special source positions
  aligned.
- The locally reconstructible SWE tail was also not sparse or selected from isolated
  boilerplate fragments. Every one of 173 unique reconstructed trajectory tails was
  one full exact block with 100% eligible-row coverage.
- The SWE generated summary texts/token IDs were never saved. Exact summary alignment,
  exact total dose, and the summary-versus-tail split therefore remain unavailable from
  the committed record.
- Summary-only and tail-only treatment ablations were never run. Alignment row mass
  does not identify which region caused a likelihood movement.

No model weights were loaded, no model forward was called, and no GPU was used.

## Canonical audit unit

- Script: `scripts/audit_graft_dose_recovery.py`
  - SHA-256: `f0f493ed6941912646bcf1dae1433eee51f054810f9755f4e6d9c31bb012e2d1`
- Focused tests: `tests/test_audit_graft_dose_recovery.py`
  - SHA-256: `b802b827356a100b1c9a432c99b4abbc1250a04b15665dbe58880f57c2a3b717`
- Machine-readable result:
  `results/graft_dose_recovery/graft-dose-recovery_Qwen3-30B-A3B-Instruct-2507_20260712T064641Z.json`
  - SHA-256: `b2beff202fbd04ecea105ee296ec47f890512d0727ca1700183a3691696be19d`
  - Size: 313,381 bytes
  - Schema: `graft-dose-recovery-v1`
  - Repository commit observed at execution:
    `d37a2bb3bceae761744bb024d7263e869b115083`

The JSON preserves every per-conversation legacy result, every unique reconstructed SWE
tail geometry, all distributions and bounds, all relevant input file hashes, the exact
tokenizer revision, runtime provenance, and the worktree status observed before output.

## Method

Tokenizer-only reconstruction used:

- `Qwen/Qwen3-30B-A3B-Instruct-2507`
- revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`
- `difflib.SequenceMatcher(autojunk=False)` through the repository's active
  positional-within-region alignment implementation
- minimum retained matching-block width: eight tokens
- destination sink exclusion: positions below four
- destination special-token exclusion

The reported `aligned_positions` count is the number of token-position pairs supplied
to `blend_values`. The map-specific layer/KV-head coverage is reported separately; a
single aligned token position can write many layer/head value-row vectors.

Quartiles use linear interpolation at ranks `(N - 1) * p`, matching NumPy's default
percentile convention used during the exploratory audit.

### Legacy exact path

The audit reconstructed the two regions for each of the 18 tracked checkpoints under:

`results/champion_validate/_work_headscan/Qwen__Qwen3-30B-A3B-Instruct-2507/`

For every conversation it independently rebuilt token geometry from the saved literal
conversation and self-generated `summary_text`, then required:

1. reconstructed total pairs equal every saved per-plant `n_pairs`;
2. reconstructed summary width equal every saved `summary_len_tokens`;
3. both regions form one complete contiguous difflib block;
4. every eligible non-special source position is included; and
5. totals and summary lengths equal all four promoted headline JSONs.

All checks passed. The legacy input manifest contains 24 tracked files (18 checkpoints,
four headline results, and two source files) with aggregate SHA-256
`57d749b6b6a782042ac42a62736637f2e7c6cf8de6a483b059911171e89beede`.
The JSON records the constituent hashes and the aggregate construction rule.

### SWE partial local path

The committed SWE rows save `summary_tokens`, cut metadata, scores, and truncated free
generations. They do **not** save:

- pair counts or indices;
- summary/tail pair counts;
- exact tail region bounds;
- per-trajectory generated summary text or IDs;
- `old_ids` or `b_ids`; or
- the source trajectory messages.

The still-present ignored parquet was observed at 10,379,409 bytes with SHA-256
`ea4bf37de020e165c5210bedddeef523d8834a89a35a8c65fec24f76f0eae4f1`,
exactly matching the value recorded in the paper. It contains 491 rows and only the
`messages` column. It is not tracked by Git.

Using that local file, the audit reconstructed the tail independently of the missing
summary text. Two radically different summary placeholders produced the same relative
tail token stream and alignment; only the absolute destination offset changes with
summary length. All saved full/context token counts and cuts matched across 173 unique
trajectory rows. This makes the tail counts locally exact, but not portable to a clean
clone.

Exact summary alignment remains unknowable. For each SWE row it is bounded by zero and
the saved `summary_tokens` source width. The machine artifact therefore reports total
alignment as `[tail_pairs, tail_pairs + summary_tokens]`, not as a recovered point value.

## Exact legacy results

All numbers below are per conversation unless explicitly called pooled.

| Quantity | N | Min | Q25 | Median | Mean | Q75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Total aligned positions | 18 | 1,091 | 1,222.75 | 1,252.5 | 1,549.61 | 1,722.5 | 2,787 |
| Summary aligned positions | 18 | 714 | 759.75 | 828.5 | 824.56 | 896.25 | 899 |
| Tail aligned positions | 18 | 335 | 430.25 | 453 | 725.06 | 903 | 1,889 |
| Coverage of all B rows | 18 | 93.06% | 93.67% | 93.89% | 94.73% | 96.08% | 97.48% |
| Summary share of aligned positions | 18 | 32.22% | 47.72% | 63.00% | 57.39% | 67.57% | 72.85% |
| Tail share of aligned positions | 18 | 27.15% | 32.43% | 37.00% | 42.61% | 52.28% | 67.78% |

Pooled over the 18 conversations:

- total aligned positions: 27,893;
- summary: 14,842 (53.2105%);
- tail: 13,051 (46.7895%);
- intended source rows including structural tokens: 28,405;
- coverage including structural tokens: 98.1975%; and
- coverage of eligible non-special source rows: exactly 100% in every conversation.

Map-specific active layer-by-KV-head slots out of 48 × 4 = 192:

| Variant | Active slots | Fraction | Pooled value-row vectors written |
|---|---:|---:|---:|
| Per-head | 109 | 56.7708% | 3,040,337 |
| Per-layer | 108 | 56.2500% | 3,012,444 |
| Intersection | 74 | 38.5417% | 2,064,082 |
| Union | 143 | 74.4792% | 3,988,699 |

The per-layer row comprises 68 active slots at α = 0.75 and 40 at α = 1. The
other three variants use α = 1 at all active slots.

## Partial SWE results and bounds

The selected-map held-out evaluation set (57 trajectories) had:

- exact locally reconstructed tail positions: median 1,466, mean 1,640.65,
  range 137--8,543;
- saved generated-summary source width: median 113, mean 116.39,
  range 88--237;
- pooled exact tail positions: 93,517;
- pooled summary-width upper bound: 6,634;
- total alignment bound: 93,517--100,151 positions; and
- pooled tail share lower bound: **93.3760%**.

The tail represented a median 20.92% of the original context but ranged from 1.18% to
73.57% because the nearest-message boundary can be far from the nominal 75% point when
a single message is very large.

Other brief-summary selected-map strata:

| Stratum | N | Exact tail positions | Summary-width upper bound | Tail pooled lower bound |
|---|---:|---:|---:|---:|
| Fresh fitting half | 41 | 68,395 | 4,758 | 93.4958% |
| Held-out evaluation | 57 | 93,517 | 6,634 | 93.3760% |
| Partial confirmation | 45 | 80,382 | 5,207 | 93.9163% |

These are lower bounds because any summary rows that failed to align would make the tail
share larger. They do not make exact total dose recoverable.

## Boilerplate objection

The specific possibility that difflib created a near-empty graft by retaining only a few
high-frequency or boilerplate fragments was not observed:

- Legacy: every summary and tail was one full contiguous matching block.
- SWE tail: every locally reconstructed tail was one full contiguous matching block.
- `autojunk=False` disables difflib's frequent-token junk heuristic.

This does not prove that recent tail content is semantically valuable, nor does it assess
the missing SWE summaries. It establishes only that the measured alignment was not a
fragmentary boilerplate selection in the regions that can be audited.

## Materiality and unresolved mediation

The legacy no-detected-average-lift result cannot be explained by an intervention that
touched only a handful of positions. The aligned-position dose was extensive.

For SWE, the more material mundane alternative is tail recomputation. Under the brief
summary, at least 93.38% of pooled aligned row mass in the held-out evaluation was recent
verbatim tail. A small likelihood movement could therefore be wholly mediated by
full-history-conditioned tail values, with no contribution from summary-region state.

Row coverage cannot identify effect mediation. No summary-only or tail-only ablation was
run. The paper may state the locally observed tail dominance with its untracked-input
caveat, but it must not present an exact SWE summary/total alignment count or claim that
summary state caused the likelihood movement.

## Reproduce and verify

Full zero-GPU audit, requiring the tokenizer to be cached locally:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/audit_graft_dose_recovery.py --local-files-only
```

Focused tests:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  uv run pytest -q tests/test_audit_graft_dose_recovery.py
```

Observed test result: `4 passed, 2 warnings in 10.46s`. The warnings were SWIG-type
deprecation warnings during tokenizer-related imports; no audit assertion warned or
failed.

If `swegym.parquet` is absent, the script still emits the exact legacy section and marks
SWE `UNAVAILABLE_IN_CLEAN_CLONE`. If the parquet exists with any other hash, it fails
closed. Outputs are timestamped and the writer refuses to overwrite an existing file.
