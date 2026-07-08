<!-- ═══════════════════════════════════════════════════════════════════════════
     PREVENTION MAP — read this first. The disease on this project is "the rule
     was WRITTEN but not APPLIED under momentum." Prose you must remember fails
     exactly when you're moving fast. So every recurring failure class below is
     now guarded by a MECHANISM that re-reads reality and FAILS CLOSED — not by a
     paragraph you have to recall. If you try to do the wrong thing, the mechanism
     stops you. Do not rely on remembering; rely on the interlock firing.
     ═══════════════════════════════════════════════════════════════════════════ -->

# THE PREVENTION LAYER (docs + process + interlocks + monitoring as ONE system)

| If you do this…                                   | …this MECHANISM catches you                                             | How it fails CLOSED                                              | Kills (incident)          |
|---------------------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------------|---------------------------|
| Launch >1 pod / spend >$5 / irreversible          | `launch_pod.sh` → `preflight.py --verify` (green token gate)            | `exit 5`, no pod launches without a fresh matching token        | assumed-precondition scaling |
| Launch a model id that's a typo / multimodal      | preflight **check B** (hub resolve + text-only arch)                    | RED, `exit 1` before any spend                                  | 5-of-16 multimodal, unloadable id |
| Launch the WRONG CHECKPOINT (right family)         | preflight **check B2** (known-wrong-twin registry)                      | RED unless `SC_ALLOW_CHECKPOINT="id:reason"` logged             | #34 thinking-vs-Instruct (hours) |
| rsync a source path that doesn't exist            | preflight **check C** (manifest vs disk)                               | RED; kills rsync's silent `\|\| true` drop                       | "fix silently didn't apply" |
| Trust a launch that hung / crashed at start       | `launch_pod.sh` **post-launch real-work check** + bounded ssh          | hang → nonzero exit; no-proc/crash → `exit 1`, loud            | #3, #28, #35 launcher hang |
| Trust a monitor you never validated               | `monitor_selftest.sh` (fault-injection incl. happy-path)               | any wrong classification → `exit 1`, "MONITOR NOT CLEARED"      | #35, #36 + monitor quartet |
| Call a pod healthy when it's absent/unparseable   | `classify_pod.sh` invariant 1 (**unknown == alarm**)                   | MISSING/UNREACHABLE/UNKNOWN all return PROBLEM (rc 1)          | #35 never-launched pod    |
| Report the probe's result as the final run        | `classify_pod.sh` invariant 2 (**unique terminal marker only**)        | DONE requires `WIDE SWEEP DONE`; probe → INTERIM, n shown      | #36 probe-as-DONE         |
| Miss an errored result because the log looked clean| `classify_pod.sh` invariant 3 (**result status is authoritative**)     | status=ERROR/UNSUPPORTED → ERROR regardless of log             | error-as-interim bug      |
| Cry "stalled" on a slow render                    | `classify_pod.sh` invariant 4 (**stall is GPU-aware**)                 | GPU busy → RENDERING regardless of log; only idle-GPU+stale-log → STALLED, idle-GPU → alarm | stall-false-positive bug  |
| Call a pod healthy after losing ONE key signal    | `classify_pod.sh` invariant 6 (**partial signal-loss == alarm**)       | reachable but proc OR gpu unreadable → SIGNAL-LOSS (rc 1)      | GPU-signal-loss read as "rendering" |
| Let a blind/blank endpoint sit SILENT forever     | `exp_watch.sh` boot-persistence → `classify_pod.sh` BOOT-TIMEOUT       | BOOTING past the grace window (state file) → PROBLEM (rc 1)    | endpoint-blind never alarmed |
| Feed the classifier a grep narrower than its own  | monitors build the remote error grep from `PC_ERROR_SIGNATURES`        | single-source pattern; self-test asserts monitors interpolate it | Traceback/OOM/CUDA-error never reached classifier |
| Commit a shell script with a latent bug           | `lint.sh` (shellcheck --severity=warning, all `*.sh`)                  | any finding → `exit 1`; do not commit dirty                    | `declare -A`, stdin-eating loop |
| Scale before you have ONE real number             | early-signal probe (`SC_PROBE_CONVS`) + canary tripwire (below)        | a run with no early-signal path is NOT READY to launch          | 5h black-box fan-outs     |

**The order of operations, every scaled run:** `preflight.sh` GREEN → `monitor_selftest.sh` CLEARED
→ launch (auto post-launch check) → canary/probe yields a real number → fan out → monitor alarms on any PROBLEM.
Each arrow is a mechanism, not a memory.

---

# HARD RULE: A MONITOR IS UNTRUSTED UNTIL IT PASSES FAULT-INJECTION (incl. happy-path)

The most-repeated bug of the last sessions: monitors that LIE — built on unverified
assumptions about how the system actually emits signals (endpoint-blind, stall on a quiet
render, DONE on the probe's write, ERROR result read as interim; incidents #35/#36).
**A monitor's classification logic may NOT be relied on until it has passed a fault-injection
self-test that (a) produces NO false alert on the happy path AND (b) fires on every failure mode.**

- The classifier is a PURE function with no I/O: `scripts/classify_pod.sh` (`classify_pod`).
  Monitors (`exp_watch.sh`) SOURCE it — they must NOT re-implement state logic inline (that is
  how every monitor bug was born; inline logic can't be fault-tested).
- Before trusting a monitor: `bash scripts/monitor_selftest.sh` → must print "MONITOR CLEARED".
  It drives `classify_pod` with a fixture for every state (row 0 = happy path = no alert) and
  asserts the exact class; a wrong classification exits nonzero.
- **Absence is an alarm. Unknown is an alarm. "It didn't say anything" is a failure state**, never
  a benign line. Result STATUS (not just log grep) is authoritative for errors. A terminal
  classification (DONE) requires a marker UNIQUE to termination. Adding a monitor signal = adding
  a self-test case that proves it fires on the real emission before you trust it.

---

# HARD RULE: PROCESS TRIPWIRES — concrete conditions, not vibes

The behavioral rules ("verify first", "consult Fable when stuck") failed because they were vibes.
These are the checkable tripwires that replace them. When a tripwire condition is TRUE, the action
is MANDATORY — it is not a judgement call.

- **CANARY-BEFORE-FANOUT.** Tripwire: about to start unit #2. Action: unit #1 must first have
  produced a real, finite scored number through the full mechanism (not "it ran"). No number → do
  not fan out. (The probe `SC_PROBE_CONVS` is this by construction; honour it.)
- **VERIFY-THE-NUMBER.** Tripwire: about to report/believe a result. Action: read the actual number
  AND its `n` yourself (`status=OK` ≠ correct; n≈6 = probe, n≈48 = full run). A result that is "too
  clean" or "too catastrophic" is an audit trigger, not a headline (rule 19).
- **CONSULT-FABLE.** Tripwire: a fix has not converged in **≤2 attempts**, OR a subagent is looping,
  OR a result is confusing, OR you feel the urge to stop and ask the USER for direction. Action:
  consult Fable for the strategic/diagnostic view and KEEP working — do not grind for hours, do not
  offload the thinking to the user. (Fable caught the wrong-model + nativeness classes narrow
  debugging missed.) The user gets decisions/outcomes, not "here's where I'm confused."
- **SHARED-CODE-FIRST.** Tripwire: two independent apparatuses fail IDENTICALLY. Action: look in the
  shared code/input first, not either harness (would have found #33 in minutes).
- **COMMIT/PUSH HYGIENE.** Tripwire: a unit of work is done. Action: commit AND push to origin/trunk
  with EXPLICIT paths (never `git add -A` — it can stage secret keys). Trunk only; never rewrite
  history (correct additively). Never `rm`/overwrite uncommitted work.
- **NO FALSE CONFIDENCE.** Tripwire: about to write "works / fixed / robust / will work". Action:
  don't. Report only what you OBSERVED (past tense) and name what you did NOT verify. Success is
  declared retroactively from an observed number.
- **NEVER ACT ON INFERENCE.** Tripwire: about to terminate a pod / kill / rm / spend without an
  explicit instruction. Action: ASK or WAIT. Reversibility does not excuse it.

---

# HARD RULE: PRE-FLIGHT GATE BEFORE ANY SCALED SPEND OR FAN-OUT

## The named failure this prevents
**Assumed-precondition scaling.** Scaling up (fan-out) or spending on a precondition
that was *assumed* rather than *observed*. Two cognitive slips combine:
1. **Intent filed as effect** — "I fixed the path" / "the model list is right" is stored
   as if the fix *applied* and the file *arrived*. The loop is never closed with a real
   read of the actual state.
2. **Scale-blindness** — launching 16 pods *feels like the same single action* as
   running one, so no threshold trips and no check fires.
The symptoms (wrong checkpoint for hours, MLX for a corpus, killed at 25/27 off a size
proxy, rsync dropped a path, 16 pods on unloadable ids) are all one shape: **clever /
expensive move made while a boring precondition was unobserved.**

## Why writing the lesson down has not worked
The lesson lived as a *memory to recall*. Recall fails exactly under momentum. The fix is
not willpower — it is a **fail-closed interlock in the execution path** that does not
depend on remembering: the launcher physically refuses to run without a fresh GREEN token
for the exact job + launcher + model set. See `scripts/preflight.py` /
`scripts/launch_pod.sh` (`exit 5` when not green).

## THE TRIGGER — when the gate is mandatory
Run the gate GREEN first for **any action that crosses the "second unit" line**:
- spawns **more than one** unit (pod / job / model / parallel worker), **or**
- costs **more than ~$5**, **or**
- is **hard to reverse** (destructive, or hours of compute you can't ctrl-C back).
The **first unit is the gate**: one cheap/tiny/local round-trip that produces a real,
finite number is the price of admission to units 2..N. If it's exactly one cheap unit,
that unit *is* the probe — otherwise, gate first. When unsure, you are over the line.

## THE PRE-FLIGHT GATE — what it asserts (all local, seconds, no GPU)
Run: `MODELS=… ANCHORS=… BREADTH=… bash scripts/preflight.sh <job-script>`
It exits nonzero on the first failure and only writes `.preflight_ok.json` when every
check passes (valid 6h, pinned to this job+launcher fingerprint + the verified model set):
- **B — models resolve & are loadable.** Every id downloads `config.json` from the hub
  (catches typos / nonexistent checkpoints) and is a **text-only causal LM** (rejects
  multimodal wrappers: `vision_config`, VL/ConditionalGeneration archs). *This alone would
  have caught 5-of-16 multimodal + the unloadable model in minutes.*
- **C — the launcher's rsync sources all exist locally.** rsync's `|| true` silently drops
  a missing path; the gate makes that fail closed. *This is the "rsync dropped a path" /
  "fix silently didn't apply" bug.*
- **D — every `data/…` file the code opens exists at the read path.**
- **E — the launch mechanism parses end-to-end** (job script + every `src/*.py`).
- **F — secrets & state**: HF token present, ssh key present, git tree committed.

## EARLY SIGNAL AS A DEFAULT (not a thing added when nagged)
Every long/expensive run must surface a **real scored number early**, by construction:
- **Tier 1 round-trip before fan-out:** launch **one** cheap unit (tiny model / few convs)
  through the *full real mechanism* and confirm it yields a **sane finite number** — not
  "it ran". Only then fan out to the rest.
- The per-run **probe** (`SC_PROBE_CONVS`, already in `job_sweep.sh`) scores a few units
  first so a broken/out-of-distribution run per architecture shows in ~40min, not 5h.
Treat a run with no early-signal path as **not ready to launch.**

## THE ONE MECHANISM THAT STOPS RECURRENCE
`launch_pod.sh` fails closed: **no pod launches without a fresh green token matching this
job+launcher whose verified set contains every model being launched.** An unverified model
is rejected *even if a token exists* — you cannot sneak one into a fan-out. Override is a
single explicit, logged escape hatch (`SC_SKIP_PREFLIGHT="reason"`) for a deliberate
one-off. The interlock works when you are deep in momentum and *think* you already checked,
because it re-reads the actual state instead of trusting your memory of it.

---

# HARD RULE: OBSERVABILITY — DETECT FAILURES AS THEY HAPPEN (production SRE)

## The framing (owner, 07-08, emphatic): THESE ARE SOLVED PROBLEMS — apply them
When running anything at scale or unattended, think like a **Site Reliability / production
software engineer**, NOT with ad-hoc hacks. **Detecting systems that are behaving anomalously
or throwing errors is a SOLVED PROBLEM** with decades of established **production software
reliability engineering (SRE)** practice. Manually SSH-ing pod-by-pod to check, "I'll just look
at it myself", and — worst — *waiting for the owner to notice the breakage and nag* are the
ABSENCE of engineering. Pre-flight checks alone are NOT the answer (they can't predict every
failure). The complement is runtime fault detection. The established patterns below cover
essentially every failure mode this project has hit. Use their real names; reach for them first:

- **OBSERVABILITY** — instrument the system so its internal state is externally visible without
  manual poking: **metrics, structured logs, traces.**
- **ERROR REPORTING / ERROR TRACKING** (Sentry-style) — every failure (uncaught exception,
  non-zero exit code, `status=ERROR/UNSUPPORTED`, crash, OOM) **surfaces automatically the moment
  it happens.** It must NEVER sit silently in a log until someone looks.
- **HEALTH CHECKS — liveness & readiness probes** per unit (Kubernetes-style): is it alive, is it
  *making progress*, is it producing sane output?
- **METRICS + ANOMALY DETECTION** — track key signals (progress rate, GPU/resource utilization,
  result-in-expected-range) and flag deviations: stalled (no progress), idle-GPU-while-running,
  out-of-range result.
- **ALERTING** — fire a notification off the error/anomaly signals **automatically**
  (PagerDuty-style) so a human learns of a failure in minutes without watching.
- **CANARY / EARLY SIGNAL** — validate one small unit before fanning out (canary deployment).
- **FAIL-CLOSED ADMISSION CONTROL** — the pre-flight gate above.

## The rule
Before ANY scaled or unattended run, the **observability is built FIRST, by construction** — not
bolted on when the owner nags. **A run whose failures do not self-report is NOT READY to launch.**
- Every job self-reports structured status (OK/ERROR/UNSUPPORTED + reason).
- A standing **health monitor watches EVERY unit and ALERTS on any bad state**: crashed, errored,
  stalled, idle-GPU, died, unreachable, signal-loss, stuck-booting — silent when healthy. BOTH
  monitors — `scripts/exp_watch.sh` (expected-set) and `scripts/pod_health.sh` (self-discovering) —
  **SOURCE the fault-tested pure classifier `scripts/classify_pod.sh`; neither classifies inline**
  (inline logic is how every monitor lied and cannot be fault-tested). Both build their remote error
  grep from the classifier's single-source `PC_ERROR_SIGNATURES`, both feed log-age for GPU-aware
  stall, and both are covered by `scripts/monitor_selftest.sh`. `exp_watch.sh` additionally persists
  per-pod BOOTING duration so a never-resolving blind endpoint escalates to an alarm (does not sit
  silent). Trust neither until `monitor_selftest.sh` prints "MONITOR CLEARED".
- The owner learns of a failure from an **alert**, NEVER by having to ask "is it broken?".

## The anti-pattern (what NOT to do — this is the pattern the owner keeps catching)
- Manually SSH-ing pod-by-pod to see if things work — reactive, incomplete, doesn't scale.
- Assuming a run is healthy because it "started".
- Treating monitoring/alerting as optional or bolt-on. **It is part of building the run.**
These are hacks. The professional move is **instrumented, automated fault detection.** Solved problem.

---

# HARD RULE: shellcheck EVERY shell script — NON-OPTIONAL (owner, 07-08)
`bash -n` only checks syntax; it passes broken scripts (`declare -A` on macOS bash 3.2 ran;
an `ssh` inside a `while read` loop silently ate stdin so the health check only saw ONE pod).
**shellcheck is a standard linter that catches exactly these. Use it on every `*.sh`, always.**
- Before committing ANY shell script: `scripts/lint.sh` (runs shellcheck --severity=warning on
  all of scripts/*.sh; exits nonzero on any finding). Fix findings; do not commit dirty.
- Do NOT hand-roll ad-hoc checks in place of the linter. Standard tooling over hacks — same
  principle as the observability rule above. These are solved problems.
- `bash -n` and a manual smoke run are NECESSARY but NOT SUFFICIENT; shellcheck is required too.

---

# HARD RULE: never act on the user's resources/work without EXPLICIT instruction (07-08)
Do NOT terminate pods, kill processes, `rm`, or spend based on INFERENCE about what the user
"probably" wants. Twice today I acted unilaterally: killed near-complete work off a proxy, and
terminated a running pod the user had NOT told me to kill (right after they said "I don't want you
to stop"). Inferring intent about their money/work/compute and acting on it is a top-severity
failure. If not explicitly instructed: ASK, or WAIT. Reversibility does not excuse it.

---

# HARD RULE: a Fable-planned solution requires a Fable REVIEW of the implementation (owner, 07-08)
When a solution is designed per a Fable plan/recommendation, then AFTER it is implemented a **fresh
Fable review** must verify the implementation actually matches the plan and is correct — BEFORE the
solution is trusted or relied on. Implementation-without-review of a Fable-planned solution is NOT done.
- Use a FRESH Fable (independent eyes); the agent that implemented it self-certifying is insufficient
  — that is exactly the implement-and-assume-correct / false-confidence gap this closes.
- Give the reviewing Fable: the original plan, the ACTUAL diff/implementation (not a summary of it),
  and current facts. Ask it to confirm the plan is faithfully + correctly implemented, and flag gaps.
- Applies to: the holistic prevention system, the QK-norm ablation, any harness/monitor/gate change
  built to a Fable design.
