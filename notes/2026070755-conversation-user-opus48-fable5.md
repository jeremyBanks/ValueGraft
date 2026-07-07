_This chunk covers the tau2 benchmark decision (deferred pending chain-tier
results and funding), a full budget accounting, and completion of the chain-arm
experiment grid, which confirmed the per-layer-tuned "champion" graft config
eliminates the full-strength grafting collapse but also exposed a
compaction-robustness ceiling problem in the current chain set._

**Participants:** User, claude-opus-4-8, and claude-fable-5.

On the tau2 question, the plan of record is to treat tau2's `banking_knowledge`
benchmark as the confirm-phase standard source (with core tau domains as
negative control) only if the chain-tier table shows a real graft effect and the
user funds the next phase, and only after a roughly
$3 pilot validates that sessions exceed the compaction threshold, the reward function is deterministic, and user-simulator cost is tolerable. If the chain results come back null or funding doesn't materialize, tau2 does not run without explicit user sign-off. The rationale for using the chain tier first rather than tau2 now: the current question is whether the graft does anything at all on solvable agent tasks, which the chain tier (four validated chained-exercise sessions, ~$1.50/episode,
no user-simulator cost, difficulty independently verified) can answer far more
cheaply than an unpiloted benchmark; tau2 was only identified as a candidate a
few hours before this discussion, and switching to it before piloting would
repeat the earlier pattern of committing to a task source before verifying it
can carry the experiment.

A budget accounting was given on request: $200 total loaded across four
$50 installments, ~$163.50 spent,
~$36.50 remaining at that point (later replenished to ~$64.92 via a user
top-up). Of the spend, roughly $45–55 produced banked, defensible results
(damage quantification, real-trace recovery, honesty replication,
calibration/guard work, the tuning story); roughly $55–70 was "agent-track
tuition" (contaminated runs, mis-set tiers, the adversarial-summarizer episode,
dead shims, idle-lane time, and the SWE-bench capability discovery, which
yielded 7 informative failures against ~15 wasted episodes on a dataset never
vendor-certified); roughly $35–45 was legitimate infrastructure/diagnostics (pod
spin-ups, downloads, probes, and construction of the now-working chain tier).
The user's response was to let the chain-arm run play out before making further
budget decisions.

During the chain-arm run, two infrastructure bugs stalled the grid for hours
before being found and fixed. First, an earlier unquoted shell loop
(`for pair in "C2 8021"...`) had expanded incorrectly, creating follower scripts
with literal spaces in their filenames (e.g. `lane_C2 8021.sh`), making them
un-launchable — this was the root cause of repeated priority-lane deaths and the
grid stalling; the fix was to rebuild the three lane consumers from a single
clean template on fresh ports rather than debug the broken names in place.
Second, once lanes were restarted as flaky `tail -f` followers, they again
failed to advance (partly due to two followers sharing one spec file); these
were replaced with straightforward sequential runners split across tunnels, with
the champion-config arm prioritized first per the user's standing hypothesis
that per-layer tuning would cure the full-strength-grafting (α=1.0) instability
seen on seed s1 (0/4, a clean, reproducible collapse) versus its full pass on
seed s3.

The completed 20/20 chain-arm grid (4 seeds × 5 arms: Original, Compacted,
graft·0.75, graft·1.0, champion) confirmed the user's hypothesis at the
pass/fail level: the champion (per-layer-tuned) arm scored 4/4 on every seed,
including s1, where full-strength grafting had collapsed to 0/4 — establishing
that naive full-replacement grafting is unstable and layer-tuning eliminates
that instability. (A separate display bug in the summary script, caused by a
`-c` splitter matching the "c" inside "cfg", had briefly made the champion
column appear blank despite the underlying data being complete.) The deflating
counterpart: Compacted also scored 4/4 across all seeds, meaning the binary
pass/fail shows no compaction damage for grafting to repair on this chain set —
a ceiling problem, since if compaction isn't hurting at this granularity there's
nothing for the graft to visibly fix, and the champion's win reads as "doesn't
harm" rather than "repairs damage." Next planned step, not yet executed: examine
the finer-grained signals already captured in the data — per-exercise
degradation (particularly late in chains where context pressure peaks) and
recall-probe performance — to check whether a real compaction effect is hidden
beneath the binary chain-level pass/fail. If Compacted genuinely shows no
degradation at that finer grain either, the pre-declared fallback is that these
chains are too compaction-robust and harder chains (more exercises, tighter
thresholds, cross-exercise dependencies) would be needed.
