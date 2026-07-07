# SWE-Gym Next-Action Readout Notes

This note tracks the qualitative J-lens probe that applies the same semantic
readout idea to SWE-Gym next-action targets. The goal is not to run an agent
loop. The goal is to take a real trajectory cut, force the true next assistant
action, and compare token readouts under the full context versus a compacted
summary-plus-tail context.

## Method

For each trajectory:

1. Reload the SWE-Gym messages from `swegym.parquet`.
2. Use the same broad cut rule as the Stage 2 SWE-Gym experiment: choose an
   assistant action between roughly 60% and 85% of the full trajectory tokens.
3. Treat messages before that action as the context and the chosen assistant
   message as the true next action.
4. Generate a compact summary from the pre-action context.
5. Build a compacted prompt with the summary plus a recent tail.
6. Teacher-force the exact next action after both prompts.
7. Run the Jacobian lens over every forced action token, then rank full-context
   versus compacted-context top-k readout divergence.

This directly targets the coding shape we care about: tool choice, command
choice, path choice, and line-range/file-local action choice.

## Output Map

- `outputs/qwen36_swegym_next_action_probe_t0005.json`: first clean MONAI run.
- `outputs/qwen36_swegym_next_action_probe_t0001_t0004.json`: early token-level
  run for getmoto and Dask.
- `outputs/qwen36_swegym_next_action_probe_spanaware_t0001_t0004_t0005.json`:
  span-aware rerun. Useful for span aggregation, but trajectories 1 and 4 used
  action-shaped generated "summaries", so treat those two as diagnostic only.
- `outputs/qwen36_swegym_next_action_probe_cleanprompt_t0001_t0004.json`:
  stronger no-tool summary prompt. Trajectory 1 is clean; trajectory 4 still
  continues into tool-call text after an initially useful summary.
- `outputs/qwen36_swegym_next_action_probe_trimmed_t0004.json`: trajectory 4
  with summary text trimmed at tool/chat markers.

## Summary Generation Lesson

SWE-Gym traces have a strong action prior. If we append a summary request to an
agent trajectory, Qwen3.6 sometimes continues the task with a tool call instead
of writing a compact note. That happened for trajectories 1 and 4 in the first
span-aware rerun.

The probe now does three things to reduce that failure mode:

- The summary request says explicitly that this is not a request to continue the
  task.
- The prompt pre-fills `Summary:` before generation.
- The generated text is trimmed at tool/chat markers such as `<function=`,
  `<parameter=`, and `<think>`.

That is not a scientific result about grafting. It is an implementation lesson
for this qualitative probe: if the compacted context accidentally includes the
next action, the readout comparison is contaminated.

## Trajectory 1: getmoto File View

Clean output:
`outputs/qwen36_swegym_next_action_probe_cleanprompt_t0001_t0004.json`

Target action:

```text
<function=str_replace_editor>
<parameter=command>view</parameter>
<parameter=path>/workspace/getmoto__moto__4.1/moto/rds/responses.py</parameter>
<parameter=view_range>[584, 600]</parameter>
</function>
```

The compact summary is proper prose. It describes the RDS `describe_db_clusters`
issue, the getmoto repository, inspected files, and the relevant lines in
`moto/rds/models.py` and `moto/rds/responses.py`.

The span readout now correctly captures the path even though tokenization merges
the preceding `>` with `/workspace`. Useful spans:

- `str_replace_editor`: mean span divergence 0.651.
- `/workspace/getmoto__moto__4.1/moto/rds/responses.py`: mean 0.561.
- `responses.py`: mean 0.568.
- `[584, 600]`: mean 0.407.

This is a good example of the probe landing on the actual next-action object:
not just "view some file", but this file and this line range. It is still noisy,
but it has the right shape.

## Trajectory 4: Dask Grep

Best output: `outputs/qwen36_swegym_next_action_probe_trimmed_t0004.json`

Target action:

```text
<function=execute_bash>
<parameter=command>grep -n 'normalize_token' /workspace/dask__dask__2022.6/dask/base.py</parameter>
</function>
```

The trimmed summary identifies the task as implementing deterministic hashing
for Enum types in Dask tokenization, notes that `dask/base.py` contains
`normalize_token`, and says the next step is to search registrations in
`base.py`.

Useful spans:

- `execute_bash`: mean span divergence 0.579.
- `grep -n 'normalize_token' /workspace/dask__dask__2022.6/dask/base.py`: mean
  0.536.
- `/workspace/dask__dask__2022.6/dask/base.py`: mean 0.548.
- `base.py`: mean 0.413.

This is the closest coding analogue to the earlier named-entity demos: the
important token is not a fictional label but an operational string that chooses
the next search command and file.

## Trajectory 5: MONAI Reproduce Script

Clean output: `outputs/qwen36_swegym_next_action_probe_t0005.json`

Target action:

```text
<function=execute_bash>
<parameter=command>python3 /workspace/Project-MONAI__MONAI__0.8/reproduce_error.py</parameter>
</function>
```

The generated compact summary correctly identifies the task as fixing MONAI's
`Evaluator` `mode` parameter, the repository path, the file under edit
(`monai/engines/evaluator.py`), and the current state of the attempted fix.

Useful spans from the later span-aware diagnostic run:

- `execute_bash` and `command` tokens show the action frame.
- `python3` and the `/workspace/...` path show the command to run.
- `Project-MONAI__MONAI__0.8` identifies the repo instance.
- `reproduce_error.py` identifies the verification script.

The strongest human-readable differences are around the path and verification
script tokens. Around `/workspace`, the later-layer readout in both contexts
still knows it is a path, but the full-context side is more tied to the specific
repository/action continuation, while the compacted side is more generic
path/project syntax. Around `reproduce_error.py`, both contexts decode the
filename structure, but the full-context path shows more test/debug associations
in middle layers and less generic completion/end-marker pressure.

This is not as clean as the Pokemon and block-party examples. Tool syntax and
path fragments introduce lots of punctuation and tokenization noise. Still, the
probe is doing the right kind of thing: it found the exact operational tokens in
the next action and exposed measurable readout differences on them.

## Immediate Takeaway

The method is viable for SWE-style examples, with two caveats.

First, summary generation must be controlled. For agent traces, the model may
try to continue acting instead of summarizing, so the summary path needs
prompting and trimming safeguards.

Second, ranking should be span-first, not raw-token-first:

- Prefer path/tool/command anchors over raw highest-divergence punctuation.
- Group token fragments into phrase-level spans such as full paths and command
  names.
- Report one compact comparison per span instead of many subword-token rows.

That would make the coding examples much easier to read and would better match
the real question: whether the full-context state makes the next action's
operational tokens more situated than a fresh compacted summary does.
