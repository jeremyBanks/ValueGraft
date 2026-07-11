_This conversation refreshed the ValueGraft experiment’s live state and
corrected the refresh methodology: repository narrative changes must be
reconstructed from targeted diffs before interpreting runners and scores._

**Participants:** User and gpt-5.5-xhigh.

**Handoff State.** As of 2026-07-06, the worktree was clean and
`trunk...origin/trunk [ahead 75]`. SWE-bench plain results (`0/17`) and `swbo:`
oracle validation (`0/7`) are boundary/failure evidence rather than
arm-comparison evidence; SWE-bench is effectively retired for this model. The
active evidence track is `chain:*`: A `4/4`, B `1/1`, `E:a1.0` `1/2`, with
`E:a0.75` on `chain:s4` still running and at least ex2 passed. Legacy synthetic
results remain strong—A `13/13`, B `5/12`, `E:a0.75` `11/12`, `E:a1.0` `3/3`,
and `E:cfg=layers` `11/15`—but are a mechanism-oriented proxy, not
production-faithful headline evidence. Tau2/vendor benchmarks were only a
possible later direction.

Operationally, `matrix_V1` remained stuck on `shim down`; health checks on port
`8021` reset. Active processes included `chain:s4 E:a0.75:c12000` on `8023` and
synthetic `t2:s31 E:a0.75` on `8013`; `8022` returned `{}`, `8024` was down, and
`8013/8023` timed out under load. The current driver supported chain tasks, a
1800-second cap, widened invalid-error matching, and `E1_NATIVE` export. Budget
posture was to obtain current results before further spending.

**Methodology Correction.** The refresh had over-weighted live process status,
score counts, and file tails while missing narrative updates in `STATE.md`,
`DECISIONS.md`, `INCIDENTS.md`, `HANDOFF.md`, `SHIFT-LOG.md`, and
planning/writeup documents. The durable protocol is: record the previously seen
commit; inspect `git log previous..HEAD --name-status`; read targeted
`git show`/changed hunks for every relevant note or status file; then separately
inspect processes and results. Always distinguish the latest committed plan from
running or stale artifacts, and explicitly report which notes changed and what
they imply.

## Conversation sources

- `019f3007-bab0-7e50-b019-2625d1538f63`
