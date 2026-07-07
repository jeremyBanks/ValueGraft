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

## Trajectory 5: MONAI Reproduce Script

Raw output: `outputs/qwen36_swegym_next_action_probe_t0005.json`

Target action:

```text
<function=execute_bash>
<parameter=command>python3 /workspace/Project-MONAI__MONAI__0.8/reproduce_error.py</parameter>
</function>
```

The generated compact summary correctly identifies the task as fixing MONAI's
`Evaluator` `mode` parameter, the repository path, the file under edit
(`monai/engines/evaluator.py`), and the current state of the attempted fix.

The scan hits the useful target tokens:

- `execute_bash` and `command` tokens show the action frame.
- `python3` and the `/workspace/...` path show the command to run.
- `Project-MONAI__MONAI__0.8` identifies the repo instance.
- `reproduce_error.py` identifies the verification script.

The strongest human-readable differences are around the path and verification
script tokens. Around `/workspace`, the later-layer readout in both contexts
still knows it is a path, but the full-context side is more tied to the
specific repository/action continuation, while the compacted side is more
generic path/project syntax. Around `reproduce_error.py`, both contexts decode
the filename structure, but the full-context path shows more test/debug
associations in middle layers and less generic completion/end-marker pressure.

This is not as clean as the Pokemon and block-party examples. Tool syntax and
path fragments introduce lots of punctuation and tokenization noise. Still,
the probe is doing the right kind of thing: it found the exact operational
tokens in the next action and exposed measurable readout differences on them.

## Immediate Takeaway

The method is viable for SWE-style examples. The next improvement should be
better ranking, not a different core probe:

- Prefer path/tool/command anchors over raw highest-divergence punctuation.
- Group token fragments into phrase-level spans such as full paths and command
  names.
- Report one compact comparison per span instead of many subword-token rows.

That would make the coding examples much easier to read and would better match
the real question: whether the full-context state makes the next action's
operational tokens more situated than a fresh compacted summary does.
