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
- A standing **health monitor watches EVERY unit and ALERTS on any bad state**
  (`scripts/pod_health.sh` + the health/alerting monitor): crashed, errored, stalled, idle-GPU,
  died, unreachable — silent when healthy.
- The owner learns of a failure from an **alert**, NEVER by having to ask "is it broken?".

## The anti-pattern (what NOT to do — this is the pattern the owner keeps catching)
- Manually SSH-ing pod-by-pod to see if things work — reactive, incomplete, doesn't scale.
- Assuming a run is healthy because it "started".
- Treating monitoring/alerting as optional or bolt-on. **It is part of building the run.**
These are hacks. The professional move is **instrumented, automated fault detection.** Solved problem.
