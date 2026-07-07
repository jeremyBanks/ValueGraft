# J-Lens Boundary Probe Worklog

Status: internal handoff note  
Last updated: 2026-07-07

This directory is a qualitative side investigation for ValueGraft. It is not
the main experiment harness, and none of these scripts should silently change
the current ValueGraft protocol. The purpose is to inspect the model state near
a compaction boundary so the quantitative results have a more concrete
mechanistic story.

## Current Question

The main ValueGraft question is behavioral: can carrying write-time summary
state across compaction make a compacted conversation behave more like the
original long-context conversation?

The J-lens question is narrower:

```text
When the same summary token is seen in two states,
  write-time: after the original conversation that produced the summary
  fresh: re-encoded later from only the compacted summary text
does a vocabulary readout of the residual stream show different semantics?
```

This is not meant to prove that compaction loses information. That is already
the baseline condition this project is trying to mitigate. The J-lens work is
useful because it can show examples of what kind of state might be available
to preserve or graft.

## Current State

The broad J-lens sweep has been run for Qwen3.6-27B with the downloaded
Neuronpedia/Anthropic Jacobian lens.

Committed artifacts:

- `full_layer_sweep.py`: all-summary-token, all-fitted-layer sweep.
- `job_qwen36_full_layer_sweep.sh`: RunPod wrapper for that sweep.
- `outputs/qwen36_full_layer_sweep_summary.json`: compact committed summary.
- `full_layer_sweep_handoff.md`: run-specific handoff for the full sweep.

Local ignored artifacts:

- `outputs/qwen36_full_layer_sweep.json`: full raw sweep output, about 63 MB.
  The repository blocks files over 4 MB, so keep this local/ignored unless it
  is moved to external storage.
- `*.log`, pod state JSON files, and other run byproducts.

The broad sweep covered 1,175 aligned summary tokens across 63 fitted layers,
for 74,025 token-layer rows. The pod that produced the validated artifact has
been terminated.

## Script Map

- `boundary_probe.py`: first narrow smoke probe around a compaction boundary.
- `pokemon_probe.py`: hand-authored Pokemon demo with private labels such as
  `Ghost`, `Ghost2`, `Vacuum`, and `Dex`.
- `plain_conversation_probe.py`: ordinary block-party demo with private labels
  such as `Maple`, `B-410`, and `Crane`.
- `multi_demo_scan.py`: wider scan across multiple summary formats.
- `next_token_readout_comparison.py`: control comparing J-lens top-k readouts
  to ordinary next-token probabilities.
- `swegym_next_action_probe.py`: qualitative SWE-style probe over true next
  assistant actions from real trajectories.
- `full_layer_sweep.py`: broad sweep over every aligned summary token and every
  fitted lens layer.

## Output Map

- `outputs/qwen36_pokemon_probe_v2.json`: clearest Pokemon matched-wrapper run.
- `outputs/qwen36_plain_probe.json`: clearest ordinary planning run.
- `outputs/qwen36_multi_demo_scan.json`: multi-format token scan.
- `outputs/qwen36_next_token_readout_comparison.json`: next-token control.
- `outputs/qwen36_swegym_next_action_probe_*.json`: SWE-style trajectory
  variants. Some early variants are diagnostic only because generated summaries
  leaked into tool-call/action text.
- `outputs/qwen36_full_layer_sweep_summary.json`: compact summary of the broad
  full-layer sweep.

## Method Notes

The basic comparison uses matched wrappers. The summary text is held literal
and aligned token-by-token. The difference is whether the summary appears after
the old conversation that produced it, or whether it is freshly encoded in the
same local wrapper without the old conversation.

The readout is a vocabulary-space projection from residual-stream states using
the downloaded J-lens. It is not the model's KV cache, and it is not direct
behavior. Treat it as qualitative telemetry.

Layer choice matters. Earlier probes sampled layers 16, 32, 48, and 62.
The full sweep shows that layer 48 is a strong mid/late layer for
write-time-vs-fresh divergence while staying relatively distinct from ordinary
next-token probabilities. Layer 62 can be very readable, but it is much more
continuation-like.

Top-k readout change is not automatically semantic. Raw high-change rows often
include punctuation, whitespace, Markdown table syntax, JSON braces, subword
fragments, and local continuation artifacts. Useful reporting should group
tokens into spans and should filter or separately label structural tokens.

## Reporting Principles

The J-lens notes should support the main ValueGraft story without becoming the
main evidence.

Use these claims:

- The same visible summary token can have different readout neighborhoods
  under write-time context versus fresh re-encoding.
- Some examples recover private or operational meanings that are not obvious
  from the token's generic lexical prior.
- Mid-layer readouts can differ substantially from next-token probabilities,
  so the probe is not always just a disguised next-token list.
- SWE-style action probes are promising, but they must be span-first because
  file paths, XML-ish tool syntax, and commands make raw token rankings noisy.

Avoid these claims:

- Do not present J-lens examples as behavioral proof that ValueGraft works.
- Do not imply that compaction harm is a novel finding.
- Do not treat raw highest-divergence token rows as automatically meaningful.
- Do not compare contaminated SWE summary variants as if they were clean
  compacted-context runs.

## Known Good Examples

Strong examples to reuse with enough surrounding context:

- `B-410`: write-time layer-48 readouts surface `obsolete`, `outdated`,
  `deprecated`, and `expired`, while next-token logits mostly predict the code
  continuation and fresh readouts become civic-code-like.
- `Maple`: write-time readouts show definition/reference semantics; fresh
  readouts drift toward streets, places, and generic names.
- `Ghost`/`Ghost2`: write-time readouts expose the dead-vs-survived Ralts
  contrast; fresh readouts become generic Ghost/Pokemon/name semantics.
- `Dex`: write-time readouts expose trade obligation; fresh readouts drift
  toward Pokedex/DexNav/progress meanings. This example needs next-token
  caveats because the local text also contains `trade`.
- SWE-Gym path spans: getmoto `responses.py`, Dask `base.py`, and MONAI
  `reproduce_error.py` show the method can land on operational next-action
  spans, but the presentation needs span grouping.

## Run Discipline

- Before running a pod script, check local syntax with `python3 -m py_compile`
  and `bash -n`. Use ShellCheck too if it is available.
- Do not terminate a pod until the artifact has been pulled locally and
  validated. One completed full-sweep artifact was lost by terminating during
  `rsync`; do not repeat that.
- Keep full raw artifacts local/ignored when they exceed repository limits.
  Commit compact summaries, notes, and scripts.
- Commit explicit file paths only. Avoid broad staging while other agents may
  be working in this repository.
- Treat generated reports as secondary. Raw JSON and the scripts are the
  source of truth.

## Immediate Next Work

1. Write a clear full-layer sweep report from
   `outputs/qwen36_full_layer_sweep_summary.json`, using the local raw JSON for
   deeper example drill-down if needed.
2. Make the report human-readable: include actual quoted local context,
   side-by-side write-time/fresh/readout snippets, and explain why each example
   matters.
3. Separate broad aggregate findings from illustrative examples.
4. Do not use `pack` or other discarded terminology in public-facing prose.
5. For coding examples, report spans rather than isolated subword tokens.

## Useful Commands

```sh
git status --short --branch
jq '{tokens: .total_summary_tokens, rows: .total_grid_rows, layers: (.lens.sampled_layers | length)}' \
  jlens_boundary_probe/outputs/qwen36_full_layer_sweep_summary.json
jq '.global.layer_summary | max_by(.mean_rank_weighted_distance)' \
  jlens_boundary_probe/outputs/qwen36_full_layer_sweep_summary.json
python3 -m py_compile jlens_boundary_probe/full_layer_sweep.py
bash -n jlens_boundary_probe/job_qwen36_full_layer_sweep.sh
```

