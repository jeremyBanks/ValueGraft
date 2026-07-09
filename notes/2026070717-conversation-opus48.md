_This chunk documents the closure of the chain-arm agent-track experiment: the
recall probe was found invalid for chain exercises, the chain table produced one
confirmed real result and one confirmed null, all compute pods were terminated
to stop billing, and the program was left fully parked pending a three-way
decision on how to proceed with the agent-task line of evidence._

**Participants:** claude-opus-4-8.

**Chain arm results and instrument failure.** Analysis of the sub-binary/chain
exercise table surfaced that the recall probe used to detect subtle compaction
damage was invalid for this tier: its hardcoded questions referenced constants
(tax rate in basis points, backoff base in ms) belonging to the earlier t1/t2
synthetic-constraint tasks, not the constants used in chain exercises. This is
the same class of instrument-mismatch bug flagged previously (a component built
for one task type silently inherited by another, per existing rule 9) — the
recall column for the chain tier is void and cannot be used as evidence either
way. The effort proxy (`agent_events` counts) also showed no clean separation
between Compacted (36–70) and Original (44–58) conditions, so it offered no
additional signal.

With those two channels ruled out, the remaining valid finding is binary
pass/fail per exercise, which yielded a mixed verdict: (1) confirmed real — the
champion (tuned) grafting configuration cures the α=1.0 collapse seen with naive
full-strength grafting, passing 4/4 on exercise s1 versus 0/4 for the untuned
graft, replicating the fine-tuning hypothesis; (2) confirmed null — at every
currently measurable resolution, these chain exercises show no compaction damage
for grafting to repair; Compacted passes as reliably as Original, meaning the
current chains are too compaction-robust to demonstrate a recovery effect. The
chains were judged to have hit a difficulty/robustness ceiling flagged several
exchanges earlier, now confirmed rather than hypothetical.

**Decision and path forward (unresolved, awaiting user direction).** Two
candidate fixes were identified for producing a clean demonstration of
compaction damage that grafting repairs on agent-style tasks: (a) build harder,
cross-exercise-dependent chains (e.g., exercise 4 requires content evicted from
exercise 1), estimated at roughly half a day of work and comparatively cheap but
less rigorous/prestigious; or (b) adopt the tau2 confirm-phase design, judged
the more principled option because it has a built-in evictable-vs-system-message
dissociation and vendor-certified task difficulty, at an estimated
~$3 pilot cost followed by ~$30 for a full run. A third option is to stop the
agent-task line entirely and write up the program on the strength of the
already-banked mechanism and behavioral findings. No direction had been chosen
as of this chunk; spending on the current chain design was explicitly halted and
the decision was left open for the user or a future agent to make.

**Operational actions taken to close out the run.** All three compute pods
(previously billing ~$4.17/hr combined) were terminated once the chain-arm
analysis was complete, since any of the three forward options either requires
fresh provisioning or no pods at all — bringing burn to $0/hr. Re-provisioning
is estimated at roughly 15 minutes once a direction is chosen. Balance stood at
approximately
$43.77 (prepaid, drifting down as billing settles, no overspend risk) after an earlier note of ~$63.
The pod-monitoring and validation-gate background loops were retired as obsolete
once pods were gone, and residual watchdog/monitor alerts referencing the
terminated pod were treated as expected trailing echoes, not new problems. The
90-minute heartbeat monitor was also retired on the reasoning that "keep
working" is not the correct default while holding for a user decision on an idle
system; only local-cost monitors (memory, disk) remain active.

**Banked results considered defensible regardless of the agent-track outcome:**
the full chain arm table (105 enriched rows) is committed to the repo; mechanism
results and stage-1 damage quantification; stage-2 real-trace recovery; the
honesty-effect replication at bf16 (compaction induces fabrication, grafting
restores appropriate uncertainty); the complete layer-tuning story, including
the graft-collapse cure; and both prior contamination guards. Across the
program, 22 documented incidents and 20 rules are cited as having made the
instrument trustworthy at real operational cost, and all state is described as
committed so that either the user or a fresh agent can resume cleanly from
cold-start docs once a direction (tau2, harder chains, or write-up) is chosen.
No pods are running and no further spending is occurring pending that decision.
