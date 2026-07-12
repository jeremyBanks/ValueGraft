# Powered-v13 provider watchdog and terminal-validator audit

Date: 2026-07-12

Reviewer: independent Codex QA shard `v13_provider_validator_qa`

Reviewed HEAD: `73bee60f5be21ef322ea2cecb587fa876fb2f39a`

Worktree: `/Users/jeb/experimentation-worktrees/v13-provider-validator-audit`

Verdict: **BLOCK**

## Scope and boundary

This was a read-only correctness review of:

- `src/powered_v13_watchdog.py`
- `scripts/run_powered_v13_stage_t_watchdog.py`
- `scripts/validate_powered_v13_technical_terminal.py`
- `tests/test_powered_v13_watchdog.py`
- `tests/test_run_powered_v13_stage_t_watchdog.py`
- `tests/test_validate_powered_v13_technical_terminal.py`

The review checked the literal `$1.50`/3,300-provider-second boundary, the
rate-derived cutoff, job and watcher failures, bounded harvest-before-delete,
same-cycle direct-404 plus complete-inventory absence, record mutation,
distinct-process/PID validation, the exact technical sequence and external
artifact hashes, path/symlink handling, and CLI behavior. No source file was
changed. This note is the only intended commit content.

## Blocking findings

### 1. The deadline loop does not enforce cleanup inside the literal cap

`watch()` samples `now` before the provider GET and job probe
(`src/powered_v13_watchdog.py:544-554`) and uses that stale value for the
deadline decision. Those operations are independently allowed 10 and 15
seconds in the production adapter. A cycle sampled one second before
`delete_trigger_epoch` can therefore decide `CONTINUE` after the trigger has
already passed. The next cycle still performs another provider GET and job
probe before starting the fixed 60-second harvest, followed by DELETE, direct
GET, and inventory calls. The nominal 120-second lead has no remaining margin.

A deterministic fake-clock execution at the exact `$2/hour` rate-derived
boundary produced:

```text
bounded_provider_seconds = 2700
first action              = CONTINUE
termination_reason        = deadline
dual-confirmation finish  = hard_deadline + 20 seconds
elapsed through cleanup   = 2720 seconds
rate * elapsed / 3600     = $1.511111111111111
```

In that exact-latency simulation DELETE was accepted only at the hard deadline
and the protocol-required dual confirmation finished 20 seconds later. Any
ordinary overhead moves DELETE itself beyond the deadline. The record also
timestamps deletion verification with the pre-call epoch, so it does not
preserve the actual confirmation completion time. The core duration formula is
otherwise correct: it floors `(1.50 - prior_spend) * 3600 / created_rate`, takes
the minimum with 3,300, rejects a window of at most the cleanup lead, and
recomputes all literal fields on read. The loop around that formula is the
failure.

The end-to-end total also still trusts caller-supplied
`prior_stage_t_spend_usd` and `provider_clock_started_epoch`; neither is bound
to a provider ledger or allocation timestamp in these files. A later launcher
must supply and independently bind those values for the cap to include a
rejected first allocation and acquisition time.

### 2. Definitive job completion/death is masked by provider ambiguity

`guard_decision()` handles a non-404 provider error before examining a
definitive job terminal state (`src/powered_v13_watchdog.py:396-418`). Both of
these independent checks returned `HOLD_PROVIDER_AMBIGUOUS`:

```text
job_state=COMPLETE, provider_error=RuntimeError -> HOLD_PROVIDER_AMBIGUOUS
job_state=DEAD,     provider_error=RuntimeError -> HOLD_PROVIDER_AMBIGUOUS
```

Thus known completion or process death can rent until the provider observation
recovers or the deadline fires instead of immediately attempting bounded
harvest and deletion. The focused fault matrix covers provider ambiguity only
with a running job, so it did not expose this precedence error.

### 3. Record validation allows a forged harvest to bypass harvest execution

`validate_record()` validates field shapes but, at this HEAD, does not couple
`status`, `termination_reason`, `harvest`, deletion flags, or event kinds into
an exact lifecycle. An `ALLOCATED` record containing a syntactically successful
harvest and `termination_reason="forged"` validates. On a later `COMPLETE`
observation, `watch()` sees non-null `harvest`, skips the harvest callback,
deletes the pod, and terminalizes as harvested while preserving the forged
reason. The independent execution observed zero harvest calls and
`TERMINATED_HARVESTED`.

Single-writer replacement is file-atomic and fsyncs the temporary file, but
there is no watcher ownership/CAS/file lock. Concurrent ordinary and emergency
watchers can both run harvest/delete and overwrite each other's journals.
Directory fsync is also absent after initial creation and replacement. These
are additional durability/supervision concerns even after lifecycle coupling
is repaired.

### 4. The terminal validator follows an active-chain ancestor symlink outside the store

The validator rejects a symlink at the final `records` path but does not check
the `active/<identity>` ancestors or require the resolved record directory to
remain beneath the resolved store root
(`scripts/validate_powered_v13_technical_terminal.py:288-299`). Moving the
valid case directory outside the root and replacing `active/<identity>` with a
directory symlink was accepted; the validator returned an eight-record chain
and its expected chain SHA-256 from the external target. Direct bound-artifact
and terminal-evidence paths do receive component-by-component symlink and
containment checks; the active record chain needs the same rule.

### 5. The validator rejects a store-valid artifact-binding order

The store canonicalizes `artifact_bindings` by path, but `_bindings()` requires
the observed *kind list* to equal sorted expected kinds
(`scripts/validate_powered_v13_technical_terminal.py:187-208`). A store-valid
ARM record whose lexical paths placed checkpoint before artifact contained:

```text
["arm_FF_checkpoint", "arm_FF_artifact"]
```

The independent validator rejected it with `artifact kind set/order differs`.
Kind equality should be checked as an exact unique set while retaining the
store's canonical path ordering. Current tests pass only because their fixture
path names incidentally sort in kind order.

### 6. The claimed distinct-PID/exact-lock validation is weaker than the store contract

The validator checks only `os.getpid() != runner_pid` and equality with the
lock value; it does not validate `runner_pid` as a plain positive bounded PID.
A canonical lock with `pid=-7`, supplied with `runner_pid=-7`, passed the full
eight-record validation. Its `_START_TOKEN` regex also accepts any 36-character
mixture of lowercase hex and hyphens rather than the store's UUID-shaped boot
ID. A token of
`linux-procfs-v1:` + 36 hyphens + `:123` passed. Neither lock could pass
`powered_v13_store.validate_lock()`.

The same independent-contract drift exists for time fields: the validator uses
a shape-only UTC regex and accepted `2026-99-99T99:99:99Z` in the active lock,
whereas the store parses and rejects impossible timestamps. Record `sequence`
is compared with `==` rather than checked as a plain integer, so Boolean
sequence values can likewise compare equal to 0 or 1 in a forged chain. The
CLI does launch a separate process in the happy-path test and the exact
identity/lock PID equality is checked, but this is not yet an exact independent
reimplementation of the lock/record contract.

## Verified behavior

The following checks did pass on the reviewed bytes:

- `MAX_TOTAL_SPEND_USD` is the exact `Decimal("1.50")`,
  `MAX_PROVIDER_SECONDS` is the integer 3,300, and duration is the smaller of
  the literal time cap and the floor of the rate/prior-spend budget.
- A higher live provider rate stops; a lower rate cannot extend the frozen
  deadline. Provider identity mismatch and direct 404 stop.
- Probe exits 0/20/21 map to RUNNING/COMPLETE/DEAD; other exits, timeout, and
  spawn failure map to AMBIGUOUS.
- The production harvest helper passes a 60-second timeout and converts timeout
  and spawn failure to durable failed-harvest evidence; deletion still follows
  a recorded harvest failure.
- `cleanup_once()` requires direct per-pod 404 and inventory absence from the
  same invocation. A two-cycle injection that accumulated 404 in one cycle and
  inventory absence in the next remained `TERMINATING` even though both
  historical flags were true.
- Provider active-inventory rows must be a complete list of unique string IDs
  at the adapter boundary. A malformed or unavailable inventory cannot
  terminalize cleanup.
- The validator independently enforces the exact preterminal kind sequence
  `STARTED, FOUNDATION_LOAD, ARM_FF, ARM_CC, ARM_WW, ARM_FC, ARM_FW, ARM_VP`,
  exact filenames/sequences, identity, bounds, completed-arm prefixes, parent
  hashes, unique artifact paths, payload-to-binding hashes, external sizes and
  SHA-256s, terminal evidence, evidence-chain SHA-256, and exclusive receipt.
- Direct artifact and terminal-evidence symlinks, `..` traversal, root escape,
  missing files, corruption, missing arms, same validator/runner PID, and
  receipt overwrite are rejected by the covered paths.
- The independent validator does not import `powered_v13_store`, and its
  happy-path receipt is accepted by the store and supports terminal promotion.

## Watcher supervision and CLI gaps

`emergency-cleanup` is a useful resume entry point and a record left in
`WATCHING` can be reopened. However, no reviewed file installs or verifies an
independent OS supervisor, proves automatic restart after real watcher death,
or prevents emergency cleanup from racing a still-live watcher. The existing
"watcher exit" test calls `watch()` twice sequentially in one test process; it
does not kill a watcher or verify supervisor ownership/restart. Given the
documented prior monitor-death incident, the production launcher still needs a
fault-injected distinct-process terminal chain before paid allocation.

The validator CLI has a stable success JSON and returns 2 for its handled
validation/I/O failures. The watchdog CLI's expected exclusive-init collision
is not handled by `main()`: a second `init` emits a Python traceback and exits
1 because raw `FileExistsError` is outside the `V13WatchdogError` catch. The
focused CLI test asserts only nonzero. Watch/status/emergency provider paths
also lack subprocess-level CLI tests, and the reused `pod.api()` prints expected
HTTP 404 diagnostics to stdout before the watchdog's JSON summary. These are
not the central scientific blockers, but machine-consumed CLI output and error
codes are not yet uniform.

## Tests and independent checks

Executed with the repository's Python 3.12 environment and `PYTHONPATH=src`:

```text
PYTHONPATH=src /Users/jeb/experimentation/.venv/bin/python -m pytest -q \
  tests/test_powered_v13_watchdog.py \
  tests/test_run_powered_v13_stage_t_watchdog.py \
  tests/test_validate_powered_v13_technical_terminal.py
33 passed in 0.37s

PYTHONPATH=src /Users/jeb/experimentation/.venv/bin/python -m pytest -q \
  tests/test_powered_v13_store.py
50 passed, 1 skipped in 0.61s
```

Additional deterministic checks, executed without source edits, covered:

1. stale pre-probe deadline sampling with all declared provider/job/harvest
   latencies;
2. provider ambiguity crossed with COMPLETE and DEAD job states;
3. inconsistent nonterminal harvest/reason lifecycle state;
4. timeout and spawn-failure harvest evidence;
5. split-cycle versus same-cycle deletion proof;
6. active-case ancestor symlink escape;
7. store-valid path order versus validator kind order;
8. negative PID, malformed process token, and impossible UTC lock fields; and
9. repeated watchdog `init` CLI behavior.

Because the cap/cleanup, terminal-job, lifecycle, path, ordering, and exact-lock
counterexamples all execute on the exact reviewed bytes, this HEAD does not
satisfy the Stage-T authorization gate. Repair these cases and rerun an
exact-commit independent audit before any paid allocation.
