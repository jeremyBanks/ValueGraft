_This chunk covers two read-only status refreshes by the codex (gpt-5.5) agent
on the ValueGraft experiment, followed by a user correction about the agent's
refresh methodology and the agent's self-diagnosis of that failure mode._

**Participants:** User and gpt-5.5-xhigh.

**Handoff state / protocol change.** The agent committed to a revised refresh
protocol for this repo going forward: (1) record the last-seen commit; (2) run
`git log previous..HEAD --name-status`; (3) read the actual changed hunks
(`git show`/diffs) for every changed note/status file — `STATE.md`,
`DECISIONS.md`, `INCIDENTS.md`, `HANDOFF.md`, `SHIFT-LOG.md`, and any
plan/writeup docs; (4) only then separately summarize live processes/results;
(5) explicitly report which notes changed and what was learned from them. Future
refreshes on this project should follow this diff-based reconstruction method
rather than tail/head sampling of status files.
