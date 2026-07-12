# Powered-v13 provider watchdog and terminal-validator repair final audit

Date: 2026-07-12

Reviewer: independent Codex QA shard `v13_provider_validator_qa`

Reviewed HEAD: `2aafea84d4477c2a50bec094a63ab779dc4efbe4`

Worktree: `/Users/jeb/experimentation-worktrees/v13-provider-validator-final`

Verdict: **BLOCK**

## Scope

This was a read-only repair re-audit of the provider watchdog, its CLI, the
independent technical terminal validator, and their focused/store tests. Every
executed counterexample from the audit at `73bee60f5be21ef322ea2cecb587fa876fb2f39a`
was replayed on the exact reviewed HEAD. Owner locking, supervisor takeover,
record/directory fsync, and duplicate-init CLI behavior were checked
separately. No source file was changed; this note is the only intended commit
content.

## Remaining blocker: takeover cannot resume durable termination

The new lifecycle coupling correctly rejects forged cleanup evidence in an
`ALLOCATED` or `WATCHING` record. It also makes the critical supervisor cut
unresumable at this HEAD.

A watcher that finishes and durably records harvest has the legitimate state:

```text
status = TERMINATING
termination_reason = <original stop reason>
harvest = <validated harvest evidence>
```

If it then dies before direct-404/inventory confirmation, the independent
supervisor calls `watch()` on that record. `_watch_locked()` unconditionally
appends `WATCHDOG_STARTED` with `status="WATCHING"` while retaining the reason
and harvest (`src/powered_v13_watchdog.py:571-578`). `validate_record()` then
correctly rejects the resulting inconsistent pretermination state. The exact
execution produced:

```text
terminating_takeover = REJECTED
exception = V13WatchdogError
message = pre-termination watchdog record contains cleanup evidence
```

The owner lock is released by process death, so locking is not the failure.
The state transition is. A takeover of `TERMINATING` must preserve the original
reason and harvest, must not probe or harvest again, and must resume only dual
deletion confirmation. The current test named for watcher exit covers a
`WATCHING` record before harvest, not this post-harvest/preconfirmation loss
boundary. Because this is the exact point where independent supervision is
needed to prevent paid-pod leakage, the Stage-T gate remains blocked.

## Prior blockers replayed successfully

### Literal deadline and rate-derived cutoff

The exact prior `$2/hour` fake-clock case began one second before the delete
trigger and charged every declared maximum latency: 10-second provider GET,
15-second job probe, bounded harvest, and three 10-second cleanup calls. The
post-probe clock was resampled, action changed from the old `CONTINUE` to
`STOP_DEADLINE`, and harvest was dynamically reduced to 56 seconds. Observed:

```text
bounded_provider_seconds = 2700
finish_epoch = hard_deadline - 10 seconds
provider elapsed = 2690 seconds
computed spend = $1.494444444444444444444444444
```

The literal `Decimal("1.50")`, 3,300-second minimum rule, conservative floor,
prior-spend recomputation, rate-increase stop, and immutable deadline fields
remain intact.

### Forged prefilled cleanup state

Prefilling harvest alone, a termination reason alone, or both into an
`ALLOCATED` record now fails with `pre-termination watchdog record contains
cleanup evidence`. The former zero-harvest-call terminalization counterexample
no longer reaches `watch()`.

### Terminal job under provider ambiguity

The exact crossed decisions now return:

```text
COMPLETE + provider RuntimeError -> STOP_JOB_COMPLETE
DEAD     + provider RuntimeError -> STOP_PROCESS_DEATH
```

Definitive job terminal state is no longer masked by a simultaneous provider
GET failure.

### Active-chain path and binding order

Moving `active/<identity>` outside the store and replacing it with a directory
symlink now fails with `active case path contains a symlink`. The validator
also accepts the deliberately store-valid/canonical ARM binding order
`[arm_FF_checkpoint, arm_FF_artifact]`, whose path ordering differs from sorted
kind ordering, and validates all eight preterminal records.

### Exact lock process and UTC fields

The exact mutations now fail:

- `runner_pid=-7`: `runner PID lies outside 1..2147483647`;
- boot token containing 36 hyphens: active-lock identity/process rejection;
- `2026-99-99T99:99:99Z`: not a real UTC timestamp.

The validator keeps its independent implementation and exact technical
sequence, parent hashes, artifact sizes/hashes, terminal evidence, and receipt
compatibility with the store.

## Ownership, durability, and CLI checks

- The nonblocking `flock` owner file prevents a second live writer from
  touching the record. After release, a `WATCHING` record can be acquired and
  completed by a later watcher. The post-harvest `TERMINATING` cut is the sole
  takeover failure described above.
- Instrumented record creation and replacement observed fsync order
  `[file, directory, file, directory]`: both exclusive creation and atomic
  replacement fsync the record bytes and parent directory.
- Duplicate CLI `init` now exits 2, prints the stable
  `STAGE-T WATCHDOG ERROR` prefix, emits no traceback, and preserves the first
  record.
- Same-cycle direct 404 plus complete-inventory absence remains required;
  accumulated evidence from different cycles does not terminalize cleanup.
- The real harvest helper remains bounded and records timeout/spawn failure
  before deletion.

## Tests and independent executions

Executed with the repository Python 3.12 environment and `PYTHONPATH=src`:

```text
PYTHONPATH=src /Users/jeb/experimentation/.venv/bin/python -m pytest -q \
  tests/test_powered_v13_watchdog.py \
  tests/test_run_powered_v13_stage_t_watchdog.py \
  tests/test_validate_powered_v13_technical_terminal.py \
  tests/test_powered_v13_store.py

92 passed, 1 skipped in 1.09s
```

Independent no-source-edit executions replayed the `$2/hour` deadline,
prefilled harvest/reason, COMPLETE/DEAD under provider ambiguity, active-case
ancestor escape, path-versus-kind ordering, negative PID, malformed boot UUID,
impossible UTC, post-harvest takeover, fsync sequence, and duplicate-init CLI.

The eight original counterexamples are repaired, but the executed
post-harvest/preconfirmation takeover failure is operationally load-bearing.
Repair and exact-commit re-audit are required before paid allocation.
