_This conversation covers the completion of the chain-tier arm comparison table,
its mixed verdict (a confirmed tuning result but a confirmed null on compaction
damage), operational troubleshooting of the run infrastructure, and the
resulting pause of the agent-track experiment pending a three-way decision from
the user._

**Participants:** User, claude-opus-4-8, and claude-fable-5.

The session opened with the user asking about tau2-bench's status. The assistant
reiterated that tau2 (`banking_knowledge` domain, with core tau domains as
negative control) remains the designated confirm-phase standard once two
conditions are met: the chain-tier table shows a real graft effect, and the user
funds the next phase — with a further
~$3 pilot gate (session length vs. compaction threshold, reward determinism, user-simulator cost) before any real spend. The chain tier was being run instead because it answers the prior question — does the graft do anything at all on solvable agent tasks — at lower cost (~$1.50/episode)
and without an unpiloted benchmark's unverified assumptions, following the
established lesson of piloting before committing to a task source.

**Budget status at the time:** $200 loaded across four
$50 installments; ~$163.50 spent,
~$36.50 remaining, later replenished by a user top-up to ~$65 and drifting down
to ~$43.77 by session end as prepaid billing settled. The assistant gave a rough
three-way decomposition of the $163.50 spent: roughly a third into banked
defensible results (stage-1 damage quantification, stage-2 real-trace recovery,
honesty replication, calibrations/guards, tuning story), a third into
agent-track "tuition" (contaminated runs, mis-settings, dead shims, idle time,
the SWE-bench capability-floor discovery), and a third into legitimate
infrastructure now producing clean data (pod spin-ups, chain-tier construction).

**Operational incidents during the run.** Two infrastructure bugs stalled the
chain-arm grid for hours: an earlier unquoted shell loop
(`for pair in "C2 8021"...`) had created follower scripts with spaces in their
filenames, making them unlaunchable and silently killing the priority lanes; and
a `tail -f`-based follower design failed to advance because two followers shared
the same spec file. Both were fixed by rebuilding consumers as plain sequential
runners, with the champion (per-layer-tuned) arm prioritized first per the
user's hypothesis that it would cure the graft-collapse instability.

**Chain arm grid results (20/20 cells, 4 chains × 5 arms: Original, Compacted,
graft@0.75, graft@1.0, champion):**

- graft@1.0 (full-strength grafting) collapsed to 0/4 on seed s1 but passed 4/4
  on s3/s4/s6 — confirming graft@1.0 is unstable on some chains.
- The champion (per-layer-tuned) arm passed 4/4 on every seed, including s1 —
  confirming the user's hypothesis that layer-tuning cures the α=1.0
  instability. This is a clean, confirmed positive result.
- Compacted passed 4/4 on every seed, matching Original everywhere — meaning the
  chains show no detectable compaction damage at the binary pass/fail level.
- A deeper look found no rescue for that null: per-exercise breakdown showed no
  late-session degradation, and the effort proxy (`agent_events`) showed no
  clean separation between Compacted and Original (36–70 vs 44–58, overlapping).
- The recall probe, intended to catch subtler damage, was found to be invalid
  for chain tasks — it was hardcoded to check for tax-rate/backoff-ms constants
  from the earlier synthetic (t1/t2) task design, which chain exercises don't
  use. This is logged as another instance of an instrument built for one task
  type being silently inherited by another. The recall column for the chain tier
  is void.

**Standing verdict:** the chain tier delivered one genuine confirmed result
(layer-tuning eliminates the full-strength-graft collapse) and one honest
confirmed null (these chains are too compaction-robust to show damage for
grafting to repair, at every currently valid resolution). This closes out the
chain-arm phase of the agent track as a mixed-but-informative result rather than
a clean demonstration of compaction damage being repaired.

**Current state and required next action.** All three pods were terminated (burn
now $0/hr) and the pod-monitoring, validation-gate, and heartbeat loops were
retired since there is no active run to babysit. The complete 105-row enriched
chain table is committed to the repo. The banked, non-agent-track findings
remain unaffected and strong: mechanism results (write-time cache state carries
recoverable continuity), the honesty effect (compaction induces fabrication,
grafting restores appropriate uncertainty, replicated at bf16), and the full
tuning story including the graft-collapse cure. Twenty-two incidents and twenty
rules are documented from the cost of getting the instrument trustworthy.

The user must now choose one of three directions before further agent-track
spend resumes: (1) fund the tau2 confirm-phase pilot
(~$3) leading to a full run (~$30) for a standard-benchmark demonstration with a
built-in evictable-vs-unevictable dissociation; (2) build harder,
cross-exercise-dependent chains with a corrected recall probe (~half a day,
cheaper, less externally legible); or (3) close the agent track here and write
up on the banked mechanism-and-honesty findings alone, treating the agent-task
story as an honest, unfinished thread. No experiments are running; the system is
parked awaiting this decision, with full cold-start documentation in place for
resuming from any of the three branches.
