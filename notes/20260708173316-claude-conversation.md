_This shard covers the resolution of the early-signal challenge (probe pass and
bug fixes landing), a second instance of the same premature-conclusion pattern
(multimodal models), a further root-cause reframing distinguishing
"assumed-precondition scaling" from missing observability, construction of
paired detection/prevention systems, a documentation-structure correction, and
the operational recovery to a single clean gated relaunch._

## Probe pass and bug fixes land

The two responses committed in the prior shard (log check, probe-pass mechanism)
paid off within minutes: checking live pod logs found every running pod failing
on `FileNotFoundError: data/scenarios.json`, confirming the concern that a
systemic bug could otherwise have burned the full multi-hour, multi-pod compute
budget before being noticed. Root cause was the same rsync fix from earlier — it
moved `synthetic`/`natural` into `data/` but not the top-level
`scenarios.json`/`model_geometry.json`, and rsync drops the `data/` prefix on
top-level files, landing them outside the path the harness reads. This was
fixed, and the probe-pass mechanism (score 3 conversations first, ~40 minutes)
was implemented and committed so every future model run produces a real scored
signal early rather than only at the 3-5 hour finish line. Already-provisioned
pods were repaired (redeploy + restart, no re-provision needed) rather than
terminated, since the models were already downloaded.

## Repeated premature-conclusion pattern on multimodal models

Verifying the repaired pods surfaced a second genuine early-signal catch:
Mistral-Small-3.2-24B is a multimodal wrapper (`Mistral3Config`, not a causal
LM), caught in ~2 minutes at model load rather than after a full render. It was
swapped for the text-only Mistral-Small variant. Prompted by this, all 16
planned model IDs were checked against HuggingFace configs, revealing 5 of 16
are multimodal wrapper classes (`ForConditionalGeneration`), including both
models in the Gemma-4 primary de-confound pair and both in the Qwen3.6 pair —
since the harness loads via `AutoModelForCausalLM`, all 5 would return
UNSUPPORTED, reducing the pre-registered two-pair design to one pair (Qwen3).
This was routed to Fable as science-critical while the 11 confirmed
text-loadable models continued on their own retry loop.

Fable's recommendation was accepted: one confirmed pair (Qwen3) plus the
11-model QK-norm regression is independently publishable; recovering Gemma-4
(cross-vendor, higher value) via text-only loading is worth a time-boxed,
positive-control-gated attempt; recovering Qwen3.6 (same-vendor, lower value) is
not worth the effort. Report the exclusion as a dated, pre-outcome pre-reg
deviation note.

Acting on this recommendation exposed a repeat of the same failure mode already
being addressed in the session: concluding the 5 multimodal models were
categorically "UNSUPPORTED" from the class name alone, without testing whether
they run in text-only mode. The correction: they very likely do run text-only,
since the harness's `AutoModelForCausalLM` call simply doesn't auto-map the
wrapper class — this is a loading-code fix, not a fundamental blocker. The one
legitimate remaining risk is that the graft (which edits KV-cache tensors) needs
a positive-control check to confirm it targets the correct text-decoder tensors
inside the wrapper, which is a five-minute verification, not a blocker. Fable
independently reached the same conclusion. The plan was corrected accordingly:
actually load Gemma-4 text-only and verify the graft, rather than assuming it
can't be used.

## Root-cause reframing: assumed-precondition scaling and observability

A subsequent attempt to verify the live pods were healthy found
empty/inconsistent logs, traced to multiple concurrent launch mechanisms (retry
loop, repair pass, manual relaunches) overwriting the same log files and racing
— a recurrence of the earlier w1 stacking/racing incident. All concurrent
launchers were stopped and pod state was read from the cloud API (source of
truth) instead of local logs, revealing 3 pods freshly created by the retry
loop's last round, not yet running jobs.

The recurring failure across the session was reframed: the issue is not
insufficient upfront testing (which is inherently incomplete, since it can't
anticipate every failure mode) but a missing observability layer — error
reporting, health checks, metrics/anomaly detection, and alerting that fire
automatically the moment something goes wrong in production, regardless of
whether the failure mode was anticipated in advance. Fable separately named the
specific root cause as "assumed-precondition scaling": treating a stated intent
(e.g., "I fixed the path") as if it were verified effect, and treating a 16-pod
launch as equivalent in risk to a 1-pod launch, so no check ever trips at the
point where scale multiplies the cost of an unverified assumption.

Two complementary systems were built in response:

- **Detection (observability):** `pod_health.sh`, classifying each pod's state
  (rendering / errored / stalled / died / GPU-idle / done), paired with a
  standing alerting monitor that fires only on bad states and stays silent when
  healthy.
- **Prevention (fail-closed gate):** Fable's pre-flight gate
  (`scripts/preflight.py` plus a launcher interlock) refusing any fleet-scale
  launch unless a fresh check confirms every model loads as a real causal LM and
  every required data file resolves at its expected path. Run against the 11
  text models, the gate returned green in seconds; it also caught a separate
  real bug, `job_gate.sh` having the wrong checkpoint hardcoded (thinking
  variant of Qwen3-30B-A3B instead of Instruct-2507), matching an earlier costly
  bug from prior sessions.

Both systems, plus the "assumed-precondition scaling" diagnosis, were committed
and formalized as hard rules, expressed in standard SRE terminology (error
reporting/tracking, liveness/readiness health checks, metrics and anomaly
detection, alerting, canaries) and framed as solved problems the project should
default to rather than reaching for ad hoc fixes.

## Documentation restructuring

The reliability/observability guidance was initially written into a CLAUDE.md
file, which was identified as inconsistent with the project's actual
conventions: the project uses AGENTS.md as its orientation document with
purpose-specific docs for detailed content, not a single Claude-specific
catch-all file. This was corrected: content moved to a new `RELIABILITY.md`,
CLAUDE.md removed, and AGENTS.md now points to RELIABILITY.md. A `notes/`
directory inadvertently swept into a broad `git add -A` during this cleanup was
left untouched per explicit instruction, since it is unrelated user-managed
content recoverable via history if needed.

## Recovery to a single clean gated relaunch

With detection and prevention systems committed, the reconciliation of the 3
live pods (read via API rather than logs) confirmed they correspond to w1/w2/w7
(Qwen3-30B-A3B, Qwen3-32B, Mistral text) and were reused; a single clean
launcher script was written to provision the remaining 8 of the 11 text models
sequentially through the pre-flight gate, avoiding the earlier pattern of
concurrent/manual launch mechanisms racing each other.

A pgrep check for lingering launcher processes initially returned matches that
were confirmed to be false positives (the pgrep pattern matched its own
command-line invocation, not an actual running launcher); a stricter check for
real script invocations confirmed zero launchers running before relaunching.

The first launch attempt through the new script failed at runtime: it had been
syntax-checked but not runtime-tested, and used bash's `declare -A` (associative
arrays), which is unsupported on macOS's default bash 3.2. This was identified
as a distinct instance of not fully verifying a fix before relying on it. The
script was corrected to rely on the existing grep-based dedup check alone (no
associative array needed), the buggy background task was killed, and the fixed
script was read in full before relaunching, rather than assumed correct. The
corrected script — sequential gated launches per model with grep-based dedup,
retry on HTTP 500s, and no concurrent launch churn — was committed and launched;
jobs run in parallel once started. State at the end of this shard: this single
clean gated launch flow is running and being verified in progress, with the
health monitor as the intended sole source of truth going forward, and Gemma-4
text-only load/graft recovery still pending as a time-boxed side task per
Fable's guidance.
