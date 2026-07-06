# INCIDENTS.md — everything that went wrong (07-05→07-06), facts vs theories

*Written 08:25 07-06 at user direction. Purpose: a future agent must not be
misled by contaminated data or repeat these failures. Each item labeled
KNOWN (verified) vs THEORY (plausible, unverified). Chronological.*

## 1. Monitor false alarms (night)
KNOWN: parallel monitors fired simultaneous SSHes → false UNREACHABLE;
bash-isms (`declare -A`) died under harness `sh`; self-matching `pgrep -f`
(pattern matched its own invoking shell). All fixed (single loop, script
file, `[b]racketed` patterns).

## 2. Silent job deaths (night)
KNOWN: p1 1b runner died repeatedly with NO traceback: causes found =
(a) OOM from snapshot clone (16GB dup) — FIXED (optional snapshot);
(b) ALL LME-S haystacks > 85K cap → runner exits cleanly with 0 results
(exit 0, "INCOMPLETE") — full-protocol A needs >80GB card. LME then
abandoned by user.
KNOWN: honesty runner "HONESTY_DONE" with 0 results — pod lacked
data/synthetic (pre-launcher manual sync); empty glob → loop never ran.
FIX: count-validated completion markers everywhere; 5-min dead-job +
new-error alarms, no exemptions.

## 3. ssh-detach race (recurring, ≥4 incidents)
KNOWN: `ssh host 'nohup x & echo done'` can kill the child when the
session closes. Verified-working pattern: subshell + all fds redirected +
sleep + pgrep-verify: `(nohup x > log 2>&1 < /dev/null &); sleep 3;
pgrep -f "[x]pattern"`. Monitors' children ALSO die when the monitor
exits (tunnels, runners launched from Monitor shells) — launch long-lived
processes from the main Bash tool only.

## 4. THE SESSION-STATE LEAK (the big one)
KNOWN: shim session key = hash(first user msg) only; per-session state
(frozen-boundary summary cache incl. snapshot, cfg_head_map, compaction
counters) was shared between ARMS of the same task:seed when they ran on
the same pod. Design assumption "one task per pod" (in the docstring) was
silently violated by the shared-lane matrix. FIXED: key includes
mode+config suffix.
KNOWN: night-matrix arm comparisons are therefore suspect; which specific
rows were affected is NOT yet computed correctly (first log-based
forensics double-counted relaunched-reader lines — RETRACTED; correct
method = unique score.json files joined to per-lane execution order).
THEORY (likely wrong): the leak explained the α=0.75 anomaly — the
anomaly's rows mostly PREDATE cfg rows on their lanes.
UNAFFECTED (KNOWN): single-lane/round-1 runs, all TF-metric work (guards,
profiles), honesty suite, stage-1/2 — none used shared shim lanes.

## 5. α=0.75 agent anomaly
KNOWN: night matrix E@0.75 6/22 vs other-α 16/20 vs cfg=layers 9/9.
KNOWN: NOT resolved by session isolation (morning probe 0/6 — but that
probe is itself VOID, see #6).
THEORY A: real dose effect specific to agent setting (TF-optimal ≠
agent-optimal). THEORY B: mode-parse/serving bug affecting default-E
requests under some conditions (see #6). Clean r-lane data will decide.

## 6. e1 shim served wrong mode, then died (morning)
KNOWN: after its morning reload, e1's shim log shows agent requests
served as "[A] ... -" (uncompacted A-mode) during runs that requested B/E;
its isolation probe FAILED repeatedly (connection reset); process later
died. ALL e1 morning data (8-run anomaly probe) VOID. Tasks themselves
verified solvable (reference solution passes t1:s20).
THEORY: mode-parse regression or partial-code reload on e1 (its src was
rsynced mid-flight several times); the fresh r-pods run launcher-shipped
code and PASSED isolation probes (r4 verified serving mode-specific debug
fields). Root cause not yet found — do NOT reuse e1-style hand-reloaded
shims; always launcher-fresh + probe-gated.

## 7. Ceiling effect (instrument insensitivity) — REVISED 10:30 07-06
UPDATE (KNOWN): clean-instrument B = 2/7 on the same task family — the
"ceiling" did NOT reproduce post-fix; it was plausibly a session-leak
artifact (B inheriting warm summaries from earlier arms on shared
sessions). Treat the night ceiling as a property of the LEAKY instrument.
Original entry (context):
KNOWN: single-constraint seeded tasks pass under plain compaction ~90%
(constraints survive summaries/tail echoes) at c9000/c6000/c4500 — B~A
means NO room to show recovery. Round-1's dramatic B-fail/E-pass (4
unseeded runs, pre-cache shim) did NOT generalize.
FIX: sensitivity gate now pre-registered (DECISIONS 08:05): B pass-rate
≥75% on gate sample → abort+harden (t3 5-constraint task built for this;
t4 8-constraint + compact_at 3000 is the next escalation).

## 8. Dissociation probe instrument bug
KNOWN: recall extraction captured wrong SDK events → 0/82 "recalls"
including 6/6 A-condition (oracle can't fail recall → probe invalid).
FIXED (capture final assistant message); night dissociation data VOID.

## 9. Cost/ops misc
KNOWN: orphan pod burned ~1h unregistered (launcher register step raced;
now registers before launch); H200 killed pre-spend when LME abandoned;
RunPod REST 500s on new-pod creation for ~hours (capacity, not us);
launcher previously had a transient syntax error (fixed; bash -n gate
added); tune_configs.json wasn't shipped by early launcher versions.

## Standing rules distilled (enforce, don't re-learn)
1. Docstring assumptions → asserted invariants or scale-up checklist.
2. Instrument-sensitivity smoke BEFORE scaling any new eval (does the
   baseline actually fail?).
3. Probe-gate every serving endpoint before trusting it (mode-specific
   debug fields must round-trip).
4. Launch long-lived processes from main shell, never from monitors;
   verified-detach pattern only.
5. Completion markers must embed output counts.
6. Kill by PID + pgrep-verify; `[b]racketed` patterns.
7. Data from a quarantined instrument is VOID, not "probably fine".
8. Quick forensics that contradicts ground-truth files gets retracted,
   not reported.

## 10. cfg=layers 500s in the clean run (07-06 ~09:10)
KNOWN: serve_shim cfg parser referenced `sess` before creation →
UnboundLocalError → HTTP 500 on every cfg= request since the morning
deploy; agents erroed out; those clean-run rows were ARTIFACTS (deleted +
requeued post-fix). Night cfg rows (9/9) ran a code path where... NOT
fully explained — treat night cfg data per session-leak rules anyway.
KNOWN: detected by the new-error alarm within one 5-min pass. Fix
verified by syntax + redeploy; lanes restarted (1-2 in-flight rows may
score as casualties — check for suspicious False rows at 09:05-09:20 and
re-run if found).

## 11. icache unbounded VRAM growth (07-06 ~12:30)
KNOWN: incremental session cache stored per-session GPU snapshots with no
eviction → accumulated across episodes → r4 shim died silently (VRAM
exhaustion class), detected by dead-job alarm in ≤10 min. The "one task
per pod" assumption violated by MY OWN new code — third occurrence of the
class. FIX: one-entry cache (all other sessions' icache evicted on
store). Bit-exactness unaffected (eviction only). Rule 1 reaffirmed: this
assumption must become an assert, not a memory.

## 12. Dead-shim row burn (07-06 ~11:20) — the second big one
KNOWN: matrix runners had NO per-row shim-health check; while r1/r3/r4
shims were dead (ccache VRAM leak — the SAME unbounded-session-snapshot
bug as #11, present in the summary cache SINCE ITS DEPLOY last night),
runners burned ~102 spec rows as instant connection-failure "False"
scores, including 18 false A-failures (the tell: Original can't fail 78%).
FIX: (a) ccache one-entry bound + assert (same as icache); (b) runner now
health-gates before every row (waits, not burns); (c) forensic purge:
burn = connection errors in agent.log OR no E1_AGENT_DONE — 102 purged,
16 genuine kept, all purged rows re-runnable (score-skip cleared).
THEORY: most/all of today's shim deaths (r3, r4 pre-icache, possibly e1's
weirdness) trace to the ccache leak — it was the day's root pathogen.

## 13. icache VRAM-arithmetic deaths → optimization withdrawn (07-06 ~13:50)
KNOWN: even with entry-count bounds, model (61G) + A-session snapshot
(~5G) + summary snapshot + per-request transient clones exceeded 80G
during long real-repo episodes → r-shims died repeatedly (~12:09).
Validity-at-source held: the in-flight real episode (pylint-7080) produced
invalid.marker, NOT a false score — first proof the burn class is truly
dead. DECISION: incremental cache WITHDRAWN for the day's run (stable
no-cache shims on all 4 lanes); returns only after VRAM-headroom-guard
redesign + soak test at confirm phase. Also: r2 discovered (earlier
width-retry success), synced, added as 4th lane; registry deduped.
THEORY→KNOWN update for #12: resource arithmetic, not just leak, was the
recurring shim-death mechanism all day.

## 14. Mis-wired tunnel: 8023 → r3 instead of r2 (07-06, found ~15:05)
KNOWN: local tunnel 8023 was created against r3's endpoint; r2's lane
unknowingly sent all rows to r3's shim (r2 GPU idle; r3 double-loaded —
throughput loss, no validity impact: same code+isolation semantics served
correctly). Canary gate correctly FAILED against the wrong pod — the gate
design caught the mis-wire before v2 was trusted. Rewired + verified;
canary gate PASS (MISS→HIT, pred 59.2GB); r2 admitted with icache v2.
LESSON: tunnel creation must verify endpoint identity (health + a
pod-unique marker), not just connectivity — added to fix backlog.

## 15. Adversarial summarizer as production baseline (caught 17:30 07-06, by user question)
KNOWN: every agent episode from E0 (07-05 night) through the first humane
relaunch ran compaction with SUMMARY_REQUEST_BRIEF — a prompt DESIGNED to
exclude specifics ("Do not include specific decisions, names, numbers, or
details"), built for mechanism-isolation experiments where summary text
must NOT carry facts (so KV grafts' contribution is identifiable). Using
it as the production-compaction baseline made B adversarially weak vs any
real condenser. Affected strata: ALL synthetic agent rows + tier-0 real
rows (internally consistent — all arms equally handicapped — but not
production-faithful; labeled, kept, never headline). NOT affected: TF
experiments (used the thorough variant), honesty/packed experiments
(brief was correct-by-design there), stage-1/2. ROOT CAUSE: a
context-specific design choice silently inherited across an experiment
boundary — nobody re-derived the choice when the E-track's purpose
changed from mechanism to deployment-realism. FIX: SUMMARY_REQUEST_PROD
(task/state-with-paths/decisions/next-steps, 300-500w); variant recorded
in every response (audit C2).

## 16. Tier-0 compaction settings: over-corrected pressure (caught 16:00 07-06, by user question)
KNOWN: compact_at 9000 / tail 2500 gave real agents ~2.9K effective
working memory on 17-27K-token investigations — ~10x tighter than
production practice, chosen (last night) to guarantee compaction fired on
SMALL synthetic tasks and never re-derived for real repos. Observed
consequence: 51 compacted calls in one episode; forensic autopsy
(forensics-flask5063-tier0.md) shows the working-memory-amputation
signature (4x path re-guessing, evicted self-todo, divergent plan
re-derivations). Tier-0 B failures conflate "compaction damages context"
with "we amputated working memory below task viability". FIX: humane tier
12K/6K targeting 2-6 TRUE recompactions (M1 counter), verified from
sc_debug per episode.

## Design-confound rules distilled (join the standing rules)
9. Every design parameter inherited across an experiment-purpose boundary
   must be RE-DERIVED for the new purpose (summary prompt, thresholds,
   timeouts — list them explicitly at each phase change).
10. The baseline arm must be a GOOD-FAITH implementation of production
    practice — an experiment showing "X beats a strawman" is worthless;
    audit the baseline as adversarially as the intervention.
11. Any per-request/per-process config that can vary MUST be recorded in
    the data it produces (self-reporting instruments), never only in
    narrative docs.

## 17. Local plumbing unmonitored (caught by Codex read-only pass, 19:00 07-06)
KNOWN: lanes sat in "waiting: shim down" for hours after local tunnels
died — pods healthy, runners correctly waiting, ZERO alarms: pod-side
monitoring existed, results-side existed, but the LOCAL links (tunnels,
runner wait-states) had no watcher; the one lane-progress monitor had
self-retired at spec completion that morning. The validity audit
legitimately missed it (out of scope). FIX: local-plumbing watch in the
5-min loop (stuck-waiting lanes ≥12 min; forwarder-less local ports).
RULE 12: maintain a monitoring COVERAGE MAP — enumerate every link in the
chain (pod proc → shim health → tunnel → runner → driver → score → sync)
and name the watcher for each; any link without one is a standing gap.
Cross-model read-only passes are cheap and catch what scoped audits
don't.

## 18. Liveness≠progress (why the waiting lanes went unnoticed, 07-06)
KNOWN: all my board checks counted alive processes; waiting runners are
alive, so half-stalled looked healthy. The only rate-watcher self-retired
at its spec completion (~06:00) and its stall half was never replaced.
FIX: throughput-floor alarm (runners alive + 0 scores in 60 min → alarm).
RULE 13: health checks must measure OUTPUT RATE against expectation,
never merely process existence. "N lanes running" is not a status.

## Coverage map (rule 12 EXECUTED, 20:15 07-06 — the table that should have existed this morning)
| link | failure mode | watcher | latency |
|---|---|---|---|
| pod exists/billed | orphan burning | pod-count vs pods.list (podcheck) | 5m |
| shim process | death | single-strike dead-proc alarm | 5m |
| shim serving correctly | wrong mode/config | isolation probe at lane start + CONFIG banner + per-request DBG log | at start/always |
| tunnel | dies/mis-wired | forwarder check + identity verify on establish | 5m |
| runner | dead | (outcome watcher covers) | ≤100m |
| runner | waiting forever | stuck-waiting check | ~10m |
| episode | hung >60m | perl alarm; scored as timeout | hard bound |
| episode | never ran | validity-at-source (no score) | immediate |
| PER-LANE output | silent stall/all-errors | per-lane score-rate alarm | 100m |
| GLOBAL output | total stall | throughput floor | 60m |
| score integrity | impossible data | A-failure + burst alarms; dropped_ids flags | 5m |
| results→repo | sync breaks | GAP — no watcher (cp -n in loop; failure silent) |
| balance | runaway spend | GAP — no low-balance alarm (prepaid cap only) |
| local disk/mem | exhaustion | disk+memory watchdogs | 10m/2m |
Two gaps found by doing the exercise: (1) repo-sync failure would be
silent — mitigation: sync errors now matter only at analysis (reads
scratchpad directly as fallback); accepted, documented. (2) no
low-balance alarm — added below.

## 19. Dataset difficulty unvalidated before committing the experiment to it (07-06, evening)
KNOWN: SWE-bench-Lite was adopted as primary real-task source on
runnability evidence alone (adapter fail/pass validation) — never a
CAPABILITY check (can THIS model solve ANY of it, in ANY setting?). 0/2
easy-tier full-capability validations + 0 passes across all attempts;
signal cost scales 1/p, making the plain setting economically
unmeasurable long before statistical questions arise. Two full days of
agent-pipeline work targeted a dataset the subject model may not be able
to touch. RULE 14: before adopting any task source, run a CAPABILITY
SMOKE — a handful of full-capability (oracle-best-case) episodes to
estimate p — BEFORE building comparisons on it. Difficulty-to-model
matching is a precondition, not a tuning detail.
RULE 15 (from the same evening): validation queues should mix SOURCES
(interleaved), so a single dataset's difficulty miss doesn't stall the
whole program.
