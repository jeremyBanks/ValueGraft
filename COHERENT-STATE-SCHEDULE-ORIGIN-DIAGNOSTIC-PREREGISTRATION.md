# Coherent-state schedule-origin diagnostic — preregistration

**Status:** frozen before execution; no semantic outcome has been observed.

**Diagnostic identity:** `coherent-state-c10-schedule-origin-v1`

**Scientific status:** non-authorizing apparatus diagnosis. This document does
not amend or rescue `coherent-state-gapped-v10`. The completed `c10` failure
permanently prevents the frozen v10 local ladder from satisfying Amendment 11's
`L AND T` release condition.

## 1. Motivation and observed input

The exact v10 CPU ladder compared one natural 8,430-token `c10` prefix at the
same token IDs, logical positions, and contiguous physical positions under
ordinary call widths `[4096, 4096, 238]` and message-aligned widths
`[23, 4096, 4096, 92, 123]`. Every structural check passed, but the schedules
produced cache K/V maxima `16.125/5.125`, final-logit maximum `0.59375`, selected
margin shift `0.060546875`, and continuation-logit maximum `0.84375` against the
frozen `5e-4` threshold. Layer-0 cached K/V were bit-identical; layer 1 first
diverged at K/V `0.0625/0.0009765625`.

This diagnostic distinguishes the leading explanation—query-shape-dependent
bf16 attention rounding—from a causal-mask/future-token leak or an earlier
input/position/cache-construction error. It does not estimate a semantic effect.

## 2. Frozen subject and source

- Model: `Qwen/Qwen3-0.6B`.
- Revision: `c1899de289a04d12100db370d81485cdf75e47ca`.
- Device/dtype/backend: CPU, `torch.bfloat16`, eager attention at every layer.
- Source A/B: the exact committed `data/synthetic/c10.json`, rendered through
  the same production-tokenizer and source-layout construction as the failed
  ladder row. Its first 4,096 token IDs and positions `0..4095` are frozen.
- Source C replacement: the exact committed `data/synthetic/c02.json` correct
  prefix. C retains c10 token IDs at rows `0..22` and replaces rows `23..4095`
  with c02 token IDs from the same row indices. This is a diverse, natural,
  committed replacement—not a repeated synthetic cycle.
- Record exact raw-file, canonical-source, tokenizer-vocabulary, chat-template,
  rendered-token, position-array, model-revision, and diagnostic-code hashes.

Any mismatch in model, revision, dtype, device, backend, token counts, source
hashes, the shared first 23 IDs/positions, or the required changed tail is a
terminal `INVALID` result.

## 3. Frozen branches and repeats

Run each branch twice in the order `A1, A2, B1, B2, C1, C2` without changing
model state, thread settings, or environment:

- **A:** one eager forward over c10 rows `0..22` (`Q=K=23`).
- **B:** one eager forward over original c10 rows `0..4095`
  (`Q=K=4096`).
- **C:** one eager forward over the hybrid stream described above, also
  `Q=K=4096`.

Every branch starts with no cache and uses identical logical and physical
positions for its covered rows. Persist the relevant threading/BLAS/PyTorch/
platform environment. Snapshot every layer's K/V with dtype, shape, raw
SHA-256, and per-row comparison evidence.

Each branch's two repeats must be bit-identical over its complete cache. Repeat
failure is terminal `NONDETERMINISTIC`; no cross-branch interpretation is
licensed.

## 4. Frozen comparisons and controls

1. Compare A with B on rows `0..22`, per K/V, layer, and row. Persist bit-exact
   booleans, maxima, and the earliest divergent `(layer, row, component)`.
2. Compare B with C on rows `0..22` identically. Because B and C have the same
   tensor shapes and differ only in causally future tokens, these rows must be
   bit-identical under a correct causal implementation.
3. Positive-control the mutation: B and C token IDs must differ within
   `23..4095`, and their cached K or V rows after row 22 must differ in at least
   one layer. Otherwise the diagnostic is terminal `INVALID`.
4. Layer-0 A/B equality is separately required before a rounding
   interpretation because layer-0 stored K/V precede layer-0 attention output.

All reductions are recomputed from raw tensors before the unique sealed result
is written. The output path is unique by diagnostic, model slug, and UTC
timestamp; it is logged at process start and committed before interpretation.

## 5. Frozen interpretation matrix

Apply the first matching terminal row after structural, repeat, and positive
controls pass:

| Observation | Diagnostic verdict | Licensed conclusion |
|---|---|---|
| Any A/B difference in layer-0 rows `0..22` | `CONSTRUCTION_DIVERGENCE` | Input, position, cache construction, or initial projection differs; do not interpret later layers. |
| Any B/C difference in rows `0..22` at any layer | `FUTURE_TOKEN_INFLUENCE` | Causally future content influenced earlier cached rows; investigate mask/backend correctness before redesign. |
| B/C rows `0..22` bit-identical; A/B rows first diverge at layer ≥1 | `QUERY_SHAPE_ROUNDING` | Query/block shape changes bf16 attention results despite equal causal inputs; the v10 long-prefix equivalence assumption is false for the expected numerical reason. |
| A/B and B/C rows `0..22` both bit-identical | `NO_FIRST_BOUNDARY_DIVERGENCE` | The failed full-prefix divergence begins later; this test does not localize it. |

No verdict authorizes semantic execution, changes the frozen c10 FAIL, or
licenses a claim about the existence or absence of a coherent-state channel.

## 6. Predeclared fallback and next decision

If the verdict is `NO_FIRST_BOUNDARY_DIVERGENCE`, the sole adaptive fallback is
a same-code scan of the exact c10 full prefix comparing partitions
`[4096, 4096, 238]` and `[23, 4073, 4096, 238]`, locating the earliest divergent
row/layer while preserving identical tokens and positions. No semantic targets
are scored.

If the verdict is `QUERY_SHAPE_ROUNDING`, the next decision-bearing `$0` work is
an estimand-level schedule-stability measurement on c10 and, if completed, c02:
construct correct-source summary rows under a canonical message-aligned schedule
and the frozen ordinary alternative, insert both into the identical gapped
destination, and measure the induced shifts in `G_correct-G_fresh` and
`G_correct-G_wrong`. Its arms, thresholds, and stopping interpretation must be
separately frozen before execution.

No standalone paid v10 technical run will launch. Exact 30B schedule sensitivity
may be recorded later inside a redesigned experiment's paid technical preamble,
where its result controls a prospectively frozen estimand-level noise gate.

## 7. Local-process sequencing

The already-running v10 ladder may finish its in-flight `c02` committed case to
obtain a second natural magnitude. It is then reversibly paused before `c01`
completes so this diagnostic takes the local CPU critical path. Remaining v10
cases are non-authorizing background diagnostics and may resume only after the
decision-bearing `$0` measurements no longer need the machine.

