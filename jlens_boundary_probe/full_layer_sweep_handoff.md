# Full-Layer J-Lens Sweep Handoff

Date: 2026-07-07  
Status: implementation committed; pod created; broad sweep not launched yet.

## Purpose

The earlier J-lens examples were useful but too hand-picked. They mostly
sampled named anchors such as `Dex`, `Maple`, `B-410`, and `Crane`, and most
scripts sampled only four heuristic layers: `16, 32, 48, 62`.

The current goal is to get a broader view:

```text
for every aligned summary token:
  for every fitted J-lens layer:
    compare write-time readout vs fresh-summary readout
    rank token-layer positions by amount of change
```

This should let us distinguish three things:

- broad layer behavior rather than a few attractive examples;
- raw high-change rows, including punctuation and formatting noise;
- human-useful semantic rows after filtering/grouping.

The point is not to replace the quantitative ValueGraft experiments. This is
an interpretability support probe: it helps us see where write-time summary
state differs from fresh re-encoding, and how that difference changes by layer.

## Current Files

- `full_layer_sweep.py`: new broad sweep script.
- `job_qwen36_full_layer_sweep.sh`: pod job wrapper for Qwen3.6-27B.
- Expected output: `outputs/qwen36_full_layer_sweep.json`.

The implementation was committed as:

```text
1d37814 Add full-layer J-lens sweep
```

Local checks already run:

```text
python3 -m py_compile jlens_boundary_probe/full_layer_sweep.py
bash -n jlens_boundary_probe/job_qwen36_full_layer_sweep.sh
```

ShellCheck was requested as a good preflight idea for pod scripts, but it is
not installed locally on this machine at the time of writing.

## Pod State

An A100 80 GB PCIe pod was unavailable. A fallback A100-SXM4-80GB pod was
created instead:

```text
pod id: 2vasci152rpelv
state file: jlens_boundary_probe/.pod_jlens_fullsweep_state.json
```

As of this note, the code has not yet been uploaded and the job has not yet
been launched.

## What The Script Records

For each demo in `multi_demo_scan.py`, the script builds matched wrappers:

- write-time: old conversation plus summary request plus summary;
- fresh: system/local wrapper plus the same literal summary.

For each aligned summary token and each requested fitted layer, it computes:

- write-time J-lens top-k;
- fresh J-lens top-k;
- Jaccard distance between those two top-k sets;
- rank-weighted distance between those two top-k sets;
- whether the top-1 readout changed;
- next-token-vs-lens overlap for each state at that same position/layer.

The output keeps:

- `grid_scores`: compact metrics for every token-layer row;
- `layer_summary`: aggregate behavior by layer;
- `token_summary`: aggregate behavior by summary token;
- `top_token_layers_raw`: highest-change rows with full top-k readouts;
- `top_token_layers_semantic`: highest-change rows after dropping pure
  whitespace/punctuation tokens.

This balances breadth with file size: we keep metrics for the whole grid, but
only store full top-k readout lists for the highest-change rows.

## Interpretation Cautions

Raw top-change rows may be dominated by punctuation, Markdown table separators,
JSON syntax, wordpiece fragments, or summary formatting. That is expected. The
broad sweep is useful partly because it shows this noise directly.

The useful analysis should separate:

- raw highest-change token-layer rows;
- semantic-filtered rows;
- named anchors we already understand;
- layer trends, especially whether mid layers separate write-time/fresh
  semantics more than late layers;
- next-token-like rows, where a J-lens readout is mostly continuation pressure.

The next-token control already showed that layer 62 can resemble ordinary
next-token probabilities, while layer 48 was much more distinct. This sweep
should test whether that pattern generalizes across all summary tokens and all
fitted layers.

## Next Steps

1. Poll the new pod for SSH endpoint.
2. Install `rsync`/`git` remotely if needed.
3. Upload `jlens_boundary_probe/`.
4. Launch `job_qwen36_full_layer_sweep.sh` detached with full logging.
5. Monitor until output exists or failure is clear.
6. Pull `outputs/qwen36_full_layer_sweep.json`.
7. Terminate the pod promptly.
8. Commit the raw output if its size is reasonable.
9. Write a report that starts with broad observations before selecting
   illustrative examples.

## Open Questions For Analysis

- Which layers have the largest write-time/fresh divergence on average?
- Are semantic rows concentrated in a layer band, or are they scattered?
- Do named-entity and identifier tokens behave differently from ordinary
  words, punctuation, and table/JSON syntax?
- How often are high-change J-lens rows also high-overlap with next-token
  logits?
- Does the sweep recover the known examples (`B-410`, `Dex`, `Maple`,
  `Crane`) without hand selection?
- Are there new examples that are better than the previous hand-picked ones?
