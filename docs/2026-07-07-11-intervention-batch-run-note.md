# Intervention Batch Run Note

Status: run plan and interpretation contract  
Started: 2026-07-07

This batch is a correction to the earlier two-state lens exploration. The
earlier sweep compared write-time summary state with fresh compacted summary
state. That can show that the same visible summary token has different
state-conditioned readouts, but it cannot show whether ValueGraft moves the
compacted run toward the original long-context run.

The batch artifact must therefore include at least these states for each case:

- **Full context**: the original conversation plus the probe question.
- **Fresh compacted**: the preauthored summary plus retained tail plus the same
  probe question.
- **Aligned grafted compacted**: the same visible text as fresh compacted, with
  aligned write-time summary-token value-cache entries blended into the fresh
  compacted cache.
- **Shifted graft control**: the same visible text as fresh compacted, with the
  same old value entries shifted to the wrong summary-token positions.
- **Alpha controls**: at minimum alpha-zero, intermediate values, and
  alpha-one, so any effect is not reported as a single arbitrary setting.

The current implementation is V-only. It leaves fresh keys and the rest of the
fresh compacted state in place. That makes this narrower than the full
ValueGraft terminology with independent `alpha_K` and `alpha_V`, but it is a
useful intervention probe because it isolates one cache-state axis.

## What Would Count As Compelling

The strongest qualitative evidence would combine all of the following:

- The fresh compacted run is measurably farther from full context than the
  aligned grafted run at the relevant target span.
- The shifted control fails to produce the same movement, or moves in an
  obviously less coherent direction.
- The readout difference is visible on content-bearing spans, not only
  punctuation, whitespace, path separators, or local continuation syntax.
- The example can be explained using the scenario text without special
  pleading or hidden filtering.
- Ordinary next-token candidates do not already explain the whole effect.

## What Would Count As A Useful Null

A null or weak result is still informative if it is clean:

- Fresh and grafted readouts remain nearly identical across the stronger cases.
- Grafted changes mostly appear on structural tokens or generic continuation
  artifacts.
- Aligned and shifted grafts behave similarly.
- Alpha-one or shifted controls produce apparent next-token rescues while lens
  closure worsens.
- The retained tail or summary text already contains the answer so clearly that
  there is little semantic state left for the graft to reveal.

In that case, the honest conclusion is not that ValueGraft has no effect. It
is that this J-lens setup did not find a strong, shareable mechanistic example
for the effect under the tested conditions.

## Reporting Rules

- Do not present old-vs-fresh readout separation as intervention evidence.
- Do not hide shifted-control failures or alpha settings that look worse.
- Do not overfocus on single-token argmax rescues; use them only beside the
  forced-token lens readouts.
- Quote enough scenario and target text for a reader to understand each
  example without reading the raw JSON.
- Prefer fewer, better examples over many near-identical tables.
