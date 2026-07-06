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
