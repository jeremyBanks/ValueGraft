# Powered-v13 Stage-T concrete lifecycle adapter audit

Date: 2026-07-12

Reviewer: independent Codex QA shard `v13_stimuli_qa`

Reviewed HEAD: `e2104a1bf49f14371680f4a3fedac28fde9316c9`

Verdict: **BLOCK**

## Scope

This was a read-only cross-audit of the exact integrated Stage-T lifecycle,
its fixed production CLI, launchd/watchdog path, and the composed Stage-T
runner/release/subject boundaries. The review looked only for reachable false
PASS, invalid harvest, paid-Pod leak/retry, or command/file/path mismatch. No
provider, network, allocation, or experiment call was made. This note is the
only intended repository change.

The integrated adapter did repair the previously reported false-COMPLETE,
ambiguous-create cleanup, launchd API-key binding, unobserved CUDA claim, and
stale-RUNNING/PID defects. Three later composition blockers remain.

## Blocking findings

### 1. The runner's receipt directory contains the HF token and is rejected

`OpenSshTransport.sync` copies the fixed launch receipt to
`/workspace/powered-v13-stage-t/external`, verifies it while it is alone, and
then copies `hf-token` into that same directory
(`src/powered_v13_lifecycle.py:789-819`). The detached command later passes
that directory verbatim as `--receipt-directory`
(`src/powered_v13_lifecycle.py:898-903`).

The exact runner calls `open_exact_subject`, which re-runs the fixed Stage-T
checkout verifier (`src/powered_v13_subject.py:320-349`). That verifier requires
the receipt directory to contain exactly one entry with the fixed receipt
basename (`src/powered_v13_release.py:1118-1147`). At runner start the directory
contains two entries. The exact integrated runner therefore fails before
subject load. HF credentials must live in a separate fixed directory/path;
the receipt directory must remain receipt-only.

### 2. The allowed bootstrap can make the one receipt stale before runner open

The release verifier permits a receipt age of only 300 seconds
(`src/powered_v13_release.py:74,1326-1327`). `command_run` verifies the receipt
before allocation (`scripts/run_powered_v13_stage_t_lifecycle.py:216-226`). The
same bytes are then reused after endpoint admission, exact checkout sync,
Python/Torch installation, and the 30B snapshot download. Those operations are
allowed up to the 1,800-second sync/bootstrap bound before the runner starts.

`open_exact_subject` performs a fresh real-clock verification at runner start,
with no clock or tolerance override. Thus an otherwise authorized bootstrap
that takes more than five minutes necessarily fails as stale. The paid path
needs a release-authorized fresh post-bootstrap receipt step and must preserve
and bind those exact bytes; weakening subject freshness is not a repair.

### 3. COMPLETE is published before the terminal receipt exists

The detached wrapper atomically publishes `job-state.json` as `COMPLETE` and
only afterward writes `receipts/job-terminal.json` and exits
(`src/powered_v13_lifecycle.py:925-928`). The production probe accepts
`COMPLETE` without requiring the PID to be dead
(`scripts/run_powered_v13_stage_t_lifecycle.py:135-175`). A watcher can
therefore begin harvest and deletion in that interval.

The harvester requires at least one file in each category, not the exact
terminal receipt. The already-written `job-start.json` satisfies the receipt
category, so harvest and lifecycle can report success even if deletion prevents
`job-terminal.json` from ever being written. The terminal receipt must be made
durable before the atomic COMPLETE state publication, with a regression for
that ordering and the required receipt set.

## Verification

Executed at the reviewed HEAD with the repository Python environment:

```text
PYTHONPATH=src ../v13-subject-loader/.venv/bin/python -m pytest -q \
  tests/test_powered_v13_lifecycle.py \
  tests/test_powered_v13_watchdog.py \
  tests/test_run_powered_v13_stage_t_watchdog.py \
  tests/test_powered_v13_stage_t.py

88 passed in 2.51s
```

The existing release counterexamples independently confirmed both composed
receipt constraints:

```text
PYTHONPATH=src ../v13-subject-loader/.venv/bin/python -m pytest -q \
  tests/test_powered_v13_release.py::test_any_extra_receipt_directory_entry_is_rejected \
  tests/test_powered_v13_release.py::test_absent_and_stale_receipts_are_rejected

3 passed in 4.36s
```

`py_compile` passed for the lifecycle core, lifecycle CLI, watchdog CLI, and
integrated Stage-T runner. `git diff --check` was clean before this note.

No paid work is authorized by this audit.
