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

## Process Correction: 2026-07-07 Missing Intervention State

The first shareable ValueGraft/J-lens explainer overclaimed the relationship
between the J-lens examples and the graft intervention. The broad sweep only
compared two states:

1. write-time summary tokens processed with the old conversation present;
2. the same visible summary tokens freshly re-encoded in a compacted context.

That two-state comparison is a useful diagnostic for a state gap, but it is
not evidence about whether ValueGraft closes the gap. The earlier
`boundary_probe.py` had an optional grafted post-token path, but the saved
Qwen3.6 artifact skipped it with `cache layer lacks .keys/.values`. Therefore
the report did not have the intervention-state data it needed.

The correction is to collect a post-boundary intervention probe. The minimum
valid artifact must include:

- full context with the same appended probe user message;
- fresh compacted context with the same visible `summary + tail` and probe;
- grafted compacted context with the same visible text as fresh compacted, but
  with aligned write-time summary value-cache entries blended into the fresh
  cache;
- each state's own ranked next-token candidates;
- J-lens readouts for the same forced first token, chosen from the full-context
  argmax, so token identity does not confound the readout comparison;
- explicit cache/debug metadata showing whether the graft path was actually
  available, how many aligned pairs were grafted, and which cache structure was
  used.

Do not rewrite public-facing conclusions from this side investigation until
that three-condition artifact exists and an independent read-only audit has
confirmed that it addresses the missing-intervention-state problem.

## Process Correction: 2026-07-07 Schema Surprise

During the first writeup pass for the full-layer sweep, I made an analysis
process error. I queried the JSON output with ad hoc `jq` expressions, was
surprised by the structure, and then adjusted the queries defensively instead
of immediately stopping to verify the producer schema.

That was the wrong response. This directory is part of a research project, and
the artifact format is ours. If the structure is surprising, the correct
sequence is:

1. Open the producing script.
2. State the expected schema explicitly.
3. Validate the artifact against that schema.
4. Only then interpret the data.
5. Record the mistake and the resolution in the repo state.

What actually happened:

- `full_layer_sweep.py` intentionally writes `demos` as an object keyed by
  demo name, not as an array. My first query treated each demo value as though
  it also contained a `demo_id`, so the demo name appeared as `null`. That was
  a query mistake, not missing data.
- A later query used sloppy boolean/pipe structure while filtering contexts.
  The resulting jq error was another analysis-query mistake, not evidence that
  `token_summary` contained mixed row types.
- A direct schema audit showed that the raw artifact is internally consistent:
  every demo has `token_summary == summary_tokens`, 63 layer summaries, and
  `grid_scores == summary_tokens * 63`.
- The compact summary also matches the raw totals: 7 demos, 1,175 summary
  tokens, and 74,025 token-layer rows.

The correction is now encoded in `validate_full_layer_sweep.py`. Run it before
interpreting or reporting on the sweep:

```sh
python3 jlens_boundary_probe/validate_full_layer_sweep.py \
  jlens_boundary_probe/outputs/qwen36_full_layer_sweep.json \
  --summary jlens_boundary_probe/outputs/qwen36_full_layer_sweep_summary.json
```

Expected output for the current artifacts:

```text
VALID: demos=7 layers=63 summary_tokens=1175 grid_rows=74025
VALID: compact summary matches raw totals and schema
```

New rule for this directory: do not "work around" surprising output structure.
If an artifact shape is unexpected, stop analysis, validate the schema, and
record the resolution before drawing conclusions.

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
- `validate_full_layer_sweep.py`: strict schema validator for the raw
  full-layer sweep artifact and its compact summary.

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

## Shareable Explainer

`interpreting_valuegraft_examples.md` is now the current standalone
human-readable writeup for this side investigation. It leads with `B-410`,
shows next-token candidates beside J-lens readouts, uses the clarified
ValueGraft terminology (`alpha_K`, `alpha_V`, old-context path, fresh path),
demotes Pokemon to an intuition sidebar, and keeps the J-lens claims scoped to
qualitative residual-stream readouts rather than behavioral proof.

That document is the best starting point for public-facing explanation. Older
files such as `semantic_readout_blog_draft.md`, `full_layer_sweep_report.md`,
and the per-demo notes are useful audit/history material, but should not be
treated as the current prose baseline without re-review.

## Run Discipline

- Before running a pod script, check local syntax with `python3 -m py_compile`
  and `bash -n`. Use ShellCheck too if it is available.
- Before interpreting full-layer sweep results, run
  `validate_full_layer_sweep.py` on the raw artifact and compact summary.
- Do not terminate a pod until the artifact has been pulled locally and
  validated. One completed full-sweep artifact was lost by terminating during
  `rsync`; do not repeat that.
- Keep full raw artifacts local/ignored when they exceed repository limits.
  Commit compact summaries, notes, and scripts.
- Commit explicit file paths only. Avoid broad staging while other agents may
  be working in this repository.
- Treat generated reports as secondary. Raw JSON and the scripts are the
  source of truth.
- If the output format surprises you, do not keep querying until something
  works. Inspect the producer, update or run a validator, and record the
  resolution.

## Immediate Next Work

1. If revising the explainer, keep the B-410 lead and next-token-control
   framing unless stronger examples are validated.
2. For coding examples, report spans rather than isolated subword tokens.
3. If adding new examples, validate the artifact schema first and pull rows
   from named fields rather than ad hoc shape guesses.
4. Do not use `pack` or other discarded terminology in public-facing prose.
5. Treat the J-lens as readout evidence that must be paired with behavioral
   validation before making performance claims.

## Useful Commands

```sh
git status --short --branch
jq '{tokens: .total_summary_tokens, rows: .total_grid_rows, layers: (.lens.sampled_layers | length)}' \
  jlens_boundary_probe/outputs/qwen36_full_layer_sweep_summary.json
jq '.global_layer_summary | max_by(.mean_rank_weighted_distance)' \
  jlens_boundary_probe/outputs/qwen36_full_layer_sweep_summary.json
python3 -m py_compile jlens_boundary_probe/full_layer_sweep.py
bash -n jlens_boundary_probe/job_qwen36_full_layer_sweep.sh
```
