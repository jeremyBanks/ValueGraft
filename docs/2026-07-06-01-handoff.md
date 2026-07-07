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

## 19:30 07-06 — FULL SUCCESSOR HANDOFF (supersedes all above where conflicting)

READ ORDER: INCIDENTS.md (all 17 + rules) → STATE.md top block →
DECISIONS.md 07-06 entries → PIPELINE-AUDIT.md → this file.

LIVE SYSTEM:
- Pods (4, ~$5.6/hr): r1=45557@38.128.232.57, r3=30237@213.173.105.10,
  r4=15419@213.173.105.10, r2=11448@38.128.233.55. All shims alive,
  config: tail_keep=6000 summary=prod no-icache (r4: tail 2500 legacy for
  synthetic stratum). Boot banner "CONFIG ..." in each shim.log verifies.
- Local tunnels: 8021→r1, 8022→r3, 8023→r2, 8013→r4. VERIFY IDENTITY
  after any re-establish (incident 14): health + CONFIG banner match.
- Runners: matrix_H1/H2/H3 (real tasks, spec_H1-3, humane tier) +
  matrix_SYN (synthetic remainder). Health-gated (wait, never burn);
  driver refuses scores for never-ran episodes; timeouts scored.
- Monitors: podloop v3 (5-min pulls + podcheck single-strike death +
  streamcheck incl. local-plumbing watch), heartbeat 90min, memory+disk.
- Analysis: pre-registered rules in DECISIONS (fit criteria 10:15,
  screening 13:20, secondaries 13:35+17:15, autonomy rule 11:00/11:05,
  covariates 18:20). Scores → results/agent_clean_run/ (auto-sync);
  quarantines fenced. Balance ~$66. Canonical names in writeup-guidelines.
RESUME: watch matrix_H*.log fill; when 40 real rows done → verdict per
autonomy rule → confirm phase or stop-and-report. icache v2 (validated,
flag-off) only post-soak. Sealed final eval seeds s50-99 untouched.

## 08:45 07-07 addendum (Opus 4.8) — CURRENT
Supersedes prior handoff blocks. See STATE.md 08:45 block for live state.
Key deltas since last handoff: model is now Opus (provenance correction
filed); SWE-bench retired; chain arm table complete (mixed verdict);
tau2 integrated + banking pilot running (bm25, keyless). Docs were stale
(incident 25) — now current. If resuming: read STATE 08:45 block, check
scratchpad/tau_banking.log + the 5-min watcher, then continue tau or
review jlens per queued directives.
