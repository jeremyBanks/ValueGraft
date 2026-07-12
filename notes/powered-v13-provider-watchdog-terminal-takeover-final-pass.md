# Powered-v13 provider watchdog terminal-takeover final pass

Date: 2026-07-12

Reviewer: independent Codex QA shard `v13_provider_validator_qa`

Reviewed HEAD: `0303b0352337d633d5de16d90c76b642bc2a0578`

Worktree: `/Users/jeb/experimentation-worktrees/v13-provider-validator-terminal-final`

Verdict: **PASS**

## Scope

This was the last read-only confirmation of the powered-v13 Stage-T provider
watchdog, watchdog CLI, independent technical terminal validator, and their
focused/store tests. It re-ran the complete prior test matrix, the new durable
post-harvest takeover test, and independent executions of every counterexample
from the two preceding BLOCK audits. No source file was changed; this note is
the only intended commit content.

## Load-bearing post-harvest takeover

The former failure is repaired. A legitimate `TERMINATING` record now receives
`WATCHDOG_RESUMED_TERMINATION` without changing its lifecycle state. The loop
selects `STOP_RESUME_TERMINATION`, skips the ordinary provider observation and
job probe, sees the existing harvest, and proceeds directly to idempotent
deletion plus same-cycle direct-404/inventory verification.

In addition to the unit test, an independent real-process loss-boundary check
did the following:

1. started a child watcher that held the nonblocking owner `flock`;
2. observed exactly one successful harvest and the durable `TERMINATING`
   record;
3. blocked the child inside its first deletion call;
4. confirmed a second live writer was rejected by the owner lock;
5. killed the child with `SIGTERM`, causing OS lock release;
6. started a successor whose job-probe and harvest callbacks would fail the
   check if called; and
7. logged the successor provider order.

Observed result:

```text
child messages            = harvest, delete_entered
live owner rejected       = true
child exit                = -15
durable status            = TERMINATING
successor provider order  = DELETE, GET, inventory
final status              = TERMINATED_HARVESTED
termination reason exact  = preserved
harvest evidence exact    = preserved
HARVEST_FINISHED events   = 1
resume events             = 1
```

There was no second harvest, no job probe, and no provider GET before resumed
deletion. The original `job_complete` termination reason and the complete
harvest mapping were byte-value equal before and after takeover. Direct GET
returned 404 and the same cleanup invocation observed the pod absent from the
complete active inventory.

## Prior counterexamples replayed

All eight prior executed counterexamples behaved correctly on this exact HEAD:

1. **Stale pre-I/O deadline at `$2/hour`:** post-probe action was
   `STOP_DEADLINE`; harvest was reduced to 56 seconds; cleanup finished with
   10 seconds of hard-deadline headroom after 2,690 provider seconds, or
   `$1.494444444444444444444444444` at the frozen rate.
2. **Prefilled harvest/status forgery:** a pretermination record containing a
   forged reason and harvest failed validation.
3. **COMPLETE under provider ambiguity:** returned `STOP_JOB_COMPLETE`.
4. **DEAD under provider ambiguity:** returned `STOP_PROCESS_DEATH`.
5. **Active-case ancestor escape:** an outside directory reached through
   `active/<identity>` symlink was rejected.
6. **Store-valid path versus kind ordering:** the canonical binding order
   `[arm_FF_checkpoint, arm_FF_artifact]` was accepted and all eight records
   validated.
7. **Negative PID and malformed boot UUID:** both were rejected by exact lock
   validation.
8. **Impossible UTC:** `2026-99-99T99:99:99Z` was rejected as non-calendar
   time.

The inherited checks also retain literal `Decimal("1.50")`/3,300-second
recomputation, rate-increase stop, bounded harvest failure, atomic file plus
directory fsync, one-writer ownership, clean duplicate-init exit 2, exact
technical sequence/parent/artifact/terminal hashes, and store-compatible
independent receipt promotion.

## Test results

Executed with the repository Python 3.12 environment and `PYTHONPATH=src`:

```text
PYTHONPATH=src /Users/jeb/experimentation/.venv/bin/python -m pytest -q \
  tests/test_powered_v13_watchdog.py \
  tests/test_run_powered_v13_stage_t_watchdog.py \
  tests/test_validate_powered_v13_technical_terminal.py \
  tests/test_powered_v13_store.py

93 passed, 1 skipped in 0.86s
```

The single skip is the pre-existing platform-conditioned store test, not a
watchdog or terminal-validator failure. The exact reviewed provider-watchdog
and technical-terminal-validator surface passes this final repair audit.
