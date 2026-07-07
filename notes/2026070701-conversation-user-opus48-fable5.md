_This conversation covers the resolution of the chain-arm experiment: an
infrastructure bug that had stalled data collection, the full arm-comparison
table completing, and an honest mixed verdict that leaves the project at a
genuine decision point. Assistant model switched from claude-fable-5 to
claude-opus-4-8 partway through (daily sequence 002 onward)._

**Participants:** User, claude-opus-4-8, and claude-fable-5.

**Budget context.** Total loaded to date: $200 (four
$50 installments). At the point spend was tallied, ~$163.50 had been spent,
split roughly evenly three ways: durable banked results
(mechanism/stage-1/stage-2/honesty findings,
~~$45–55), costly agent-track lessons now encoded as incidents and rules (~$55–70),
and legitimate infrastructure/diagnostics (~~
$35–45). The user deferred further budget decisions pending the chain-arm table's outcome. Balance fluctuated afterward (as low as ~$36.50,
later ~$43.77–$65 as top-ups landed and idle billing settled) but was not the
binding constraint by the end of this stretch.

**tau2 sequencing decision.** The user asked why tau2 (`banking_knowledge`
benchmark) wasn't in use yet. Decision: tau2 remains the designated
confirm-phase benchmark (with core tau domains as negative control), gated on
(1) the chain tier showing a real graft effect, (2) the user funding the next
phase, and (3) a
~$3 pilot verifying session length exceeds the compaction threshold, reward determinism, and acceptable user-simulator cost. Rationale given: chains answer the prior yes/no question ("does the graft do anything on agent tasks the model can solve") more cheaply (~$1.50/episode)
and were already mid-flight when tau2 surfaced as an option two hours earlier;
committing to an unpiloted benchmark mid-race would repeat a previously
identified failure pattern (commit-before-verify).

**Infrastructure fault found and fixed.** The chain-arm grid had stalled for
hours. Root cause: an earlier shell loop (`for pair in "C2 8021"...`) expanded
unquoted, producing follower scripts with spaces in their filenames (e.g.
`lane_C2 8021.sh`), which were consequently un-launchable — this silently killed
the priority lanes and halted grid completion. Diagnosed and fixed by rebuilding
the three lane consumers from a single clean template and replacing flaky
`tail -f` followers with straightforward sequential runners, with
champion-config rows prioritized first per the user's standing hypothesis.

**Chain-arm grid result (20/20 cells, 4 seeds × 5 arms: Original, Compacted,
graft·0.75, graft·1.0, champion/per-layer-tuned):**

- Original: 4/4 on all seeds (validation intact).
- Compacted: 4/4 on all seeds — no binary evidence of compaction damage.
- graft·1.0 (full-strength replacement): unstable — 0/4 on seed s1, 4/4 on
  s3/s4/s6, confirming a real collapse mode on some chains.
- graft·0.75: 4/4 on all seeds.
- Champion (per-layer-tuned config): 4/4 on all seeds, including the s1 case
  where graft·1.0 collapsed to 0/4.

**Interpretation.** Two conclusions, one confirmed positive and one confirmed
null:

- Confirmed: layer-tuning cures the full-strength-graft instability (champion
  4/4 vs. graft·1.0's 0/4 on s1) — validates the user's fine-tuning hypothesis
  and stands as a clean, genuine result.
- Confirmed null: at every resolution checked — binary pass/fail, per-exercise
  breakdown, and the agent-events effort proxy (Compacted 36–70 vs. Original
  44–58, overlapping, no separation) — these chain tasks show no compaction
  damage for grafting to repair. Compacted performs as well as Original
  throughout.
- Void: the recall probe was invalid for chain tasks — it was hardcoded for the
  earlier synthetic task's constants (tax rate in basis points, backoff base in
  ms), which don't appear in chain exercises, so its answers are
  uninterpretable. This is another instance of a component built for one task
  type being inherited by another without adaptation.

**Net assessment of the agent-task track.** SWE-bench was too hard (model
capability floor); the current chain tier is too easy / compaction-robust to
show damage. Neither has yet produced a clean demonstration of compaction damage
that grafting repairs on real agent tasks. The mechanism-and-honesty findings
(cache-state continuity, honesty/fabrication effect under compaction, tuning
story, contamination guards) remain solid, replicated, and are considered
defensible as the paper's spine regardless of how the agent track resolves.

**Operational wind-down.** After recording the verdict, all pods were terminated
(idle billing eliminated, $0/hr burn) and background monitors tied to the
now-idle experiment (pod loop, validation-gate loop, 90-minute heartbeat) were
retired, since they were firing against an intentionally paused board.
Local-only monitors (memory, disk) remain active at no cost. State, including 22
documented incidents and 20 operating rules, is committed to the repo for
cold-start resumption.

**Open decision (blocking, awaiting user choice):** three options were presented
for how to proceed on the agent-task track, and no work will resume until one is
chosen:

1. Fund the tau2 confirm phase (~$3 pilot, then ~$30 full run) for a
   standard-benchmark, vendor-certified difficulty result with a built-in
   evictable-vs-unevictable control.
2. Build harder, cross-exercise-dependent chains with a corrected
   (chain-appropriate) recall probe — roughly half a day of work, cheaper but
   less benchmark-prestigious.
3. Close out the agent-task track as-is and write up the program on the banked
   mechanism and honesty findings alone, treating those as sufficient.

No pods are running and no spend is occurring pending this decision.
