---
name: quiet-monitors
description: Monitors must stay silent on routine progress; only emit on completion/error/state-change, with 2min→24min backoff sleep
metadata:
  type: feedback
---

Monitor/watch loops must NOT notify on every tick. The user gets pinged whenever
a Monitor produces OUTPUT, so echoing a status line each sleep-cycle spams them.

**How to apply (user, 2026-07-07):**
- Stay SILENT during routine progress. Only `echo` (→ notify) on: COMPLETION,
  ERROR, or a genuinely meaningful STATE CHANGE. Not on "5/12 → 6/12" ticks.
- BACKOFF the sleep: start ~120s (2min), grow ~1.5x each cycle, CAP at ~1440s
  (24min). Frequent early checks, sparse later — don't poll every 4min forever.
- Track last-reported state in a var; compare and only speak on change.
- Completion/error still fire immediately (the point of the monitor).
- **LAUNCH-VERIFICATION (paired requirement, incident 28):** after launching any
  pod/long job, SYNCHRONOUSLY verify it reached REAL WORK within ~1-2 min — GPU
  memory climbed (model loaded; 1 MiB = NOT loaded) AND log advanced past setup
  into real progress — BEFORE trusting it and walking away. A spawned process /
  "launched" echo / pgrep match is NOT proof it's working. And the quiet watcher
  must itself SPEAK early if work never started (don't let a crash-at-launch stay
  silent under backoff while the pod bills for nothing).

Related: [[model-selection-by-fitness]].

- **IDLE-POD WATCHDOG (user 07-08):** run a standing monitor that checks ALL pods
  (list via RunPod API each cycle) every ~5min and alerts ONLY when a pod is
  genuinely idle — GPU util <10% AND no job process running. Downloads show 0%
  GPU but are real work → match snapshot_download/probe processes as JOB so they
  don't false-alarm. Catches chaining-gaps + finished-job idle = money burning.
  Pods idling between jobs is a recurring cost leak (a chained watcher's poll
  latency left one idle); this backstops it.
