_This chunk covers two read-only status refreshes by the codex (gpt-5.5) agent
on the ValueGraft experiment, followed by a user correction about the agent's
refresh methodology and the agent's self-diagnosis of that failure mode._

**Participants:** User and gpt-5.5-xhigh.

The first refresh (2026-07-06) found the project had shifted again: the
SWE-bench evaluation path was effectively retired as too difficult for the model
under test, with plain SWE-bench at 0/17 and `swbo:` oracle-validation mode at
0/7 (some rows with degraded scoring) — both now treated as boundary/failure
evidence rather than arm-comparison evidence. Current useful evidence had moved
to `chain:*` tasks: chain A passed 4/4, chain B 1/1, `E:a1.0` 1/2, and `E:a0.75`
on `chain:s4` was in flight (at least one example passed). The synthetic
legacy/brief benchmark remained a strong but non-production-faithful proxy: A
13/13, B 5/12, `E:a0.75` 11/12, `E:a1.0` 3/3, `E:cfg=layers` 11/15. Budget
posture had shifted to prioritizing getting results before further spend, and
tau2/vendor-benchmark was scouted as a possible future standard benchmark but
not yet in use. Operationally, `matrix_V1` was stuck in a "shim down" state for
hours with health checks on port 8021 resetting; active runner processes were
`chain:s4 E:a0.75:c12000` on 8023 and synthetic `t2:s31 E:a0.75` on 8013, while
8022 returned empty, 8024 was down, and 8013/8023 timed out under load. The
driver at this point supported chain tasks, had a 1800s cap, widened
invalid-error matching, and an `E1_NATIVE` export. Branch `trunk` was 75 commits
ahead of `origin/trunk`, worktree clean.

The user then flagged that the agent's refreshes were missing a large volume of
changed notes. The agent's self-diagnosis (carried into 2026-07-07): its refresh
method had relied on tailing files like `STATE.md` that are not structured for
tail-reading — stale "current focus" sections remain near the top while
superseding content is appended elsewhere — and it had summarized from truncated
tool output (10k–50k tokens) without recognizing the truncation as incomplete.
It had not diffed from the last commit it had actually seen despite 20+ new
commits accumulating, and had overweighted easily-measured live-process/score
data over the note/commit changes that actually encoded plan pivots. It had also
failed to separate "latest committed plan" from "currently running artifacts,"
which matters because old runners keep executing after the underlying plan has
moved on.

**Handoff state / protocol change.** The agent committed to a revised refresh
protocol for this repo going forward: (1) record the last-seen commit; (2) run
`git log previous..HEAD --name-status`; (3) read the actual changed hunks
(`git show`/diffs) for every changed note/status file — `STATE.md`,
`DECISIONS.md`, `INCIDENTS.md`, `HANDOFF.md`, `SHIFT-LOG.md`, and any
plan/writeup docs; (4) only then separately summarize live processes/results;
(5) explicitly report which notes changed and what was learned from them. Future
refreshes on this project should follow this diff-based reconstruction method
rather than tail/head sampling of status files.
