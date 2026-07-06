# HANDOFF.md — shift protocol + live state (07-06 03:45)
## Shift contract (user-directed): Claude and Codex alternate ~30-min
shifts. ONLY the on-shift agent acts (pods, launches, kills); the other
sleeps. Each shift ends with an append to SHIFT-LOG.md: what you did,
what's running, what needs watching, anything broken. Claude state docs
(STATE/DECISIONS) remain Claude-written; Codex appends to SHIFT-LOG.md and
its own files only. Kill discipline: kill by PID then pgrep-verify. Launch
discipline: nohup + subshell + </dev/null + sleep + pgrep-verify (ssh
detach race is real). NEVER trust a completion marker without an output
count.
## Live system (03:45)
- 6 pods: p2 honesty (6/12 done, ~25 min/conv); p4 4B-block→guard queued;
  e1/e2/e4/w1 shims RELOADING 30B (~ready by 03:55) — 4 local
  transition.sh scripts waiting to tunnel (8010/8011/8013/8014) and start
  matrix specs 1/2/4/w (scratchpad/spec*.txt → matrix_<pod>.log).
- Matrix runner skips scored runs; scores land in
  scratchpad/e1_runs/<task>_<mode>/score.json.
- Monitors live: podloop v3 (5-min pulls+crash detect, 30-min health),
  matrix progress watch, e3 capacity retrier, heartbeat 90min, memory+disk.
- Balance ~$100; burn ~$8/hr. RunPod 500s on NEW pod creation (capacity).
- Pipeline: explore (tonight) → confirm (A/B/champion, s10-s49) → sealed
  final (s50-s99; DECISIONS 02:50). Champion picked from matrix table.
- Morning report contract in STATE.md. Failure ledger in DECISIONS.md.

## 12:15 07-06 addendum — hardened-stack rules (supersede where conflicting)
Shims: ALWAYS single-pod restart commands (loops fail silently — incidents
ledger); SC_INCR_CACHE=1 is validated bit-exact; both session caches are
one-entry-bounded with asserts. Runner (e1_matrix.sh) health-gates rows;
driver writes NO score for invalid episodes (invalid.marker instead).
Trust only rows with score.json + E1_AGENT_DONE. streamcheck alarms on
A-arm failures (near-impossible) and burst scoring. Read INCIDENTS.md
#11-12 before touching shim/serving code; fill the AGENTS.md
stateful-change checklist in any such commit.
