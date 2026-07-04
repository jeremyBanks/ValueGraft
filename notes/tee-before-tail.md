---
name: tee-before-tail
description: "When filtering long command output (tail/grep), tee the full stream to a scratch file first so earlier output stays reviewable"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bda7fb9f-f447-4890-904b-dde750ff3370
---

When running commands whose output gets filtered (e.g. `cmd | tail -30`), first `tee` the full stream to a temp/scratch file: `cmd 2>&1 | tee "$SCRATCH/cmd.log" | tail -30`.

**Why:** filtering discards the earlier parts of the stream; if something goes wrong you can't go back and inspect. Also `tail` buffers until EOF, which makes background-task output files useless for progress monitoring.

**How to apply:** default habit for long-running or background commands; user says discretionary, not a hard rule. Also bracket long commands with `date` timestamps to detect pathological runtimes (same session's related request).
