_The gated relaunch is verified working end-to-end, after which shellcheck is
adopted project-wide and used to catch a real observability bug; a further round
of infrastructure failures (an SSH-detach hang, a gate-token invalidation, and
an unpinned-dependency break) is diagnosed and fixed, ending with the tier-1
canary rule catching the last bug before any fleet-scale fan-out._

**Participants:** User and claude-opus-4-8.

## Gated launcher verified end-to-end

The single clean gated launch flow completed its first real cycle: w1 passed the
pre-flight gate and launched successfully, and the launcher proceeded to walk
sequentially through the 11 text models with no concurrent-launch churn, reusing
the 3 already-live pods (w1/w2/w7) and provisioning the remaining 8 with
retry-on-500. This confirmed all three reliability layers built earlier in the
session were functioning together: the fail-closed pre-flight gate (prevention),
the pod-health alerting monitor (detection), and the probe-pass mechanism (early
signal at ~40 minutes instead of 3–5 hours).

## Shellcheck adopted as a mandatory hard rule

Prompted by a direct question about whether static analysis was in use, it was
confirmed that scripts had only been syntax-checked (`bash -n`), which does not
catch runtime-only failures like the `declare -A` bash-version issue from the
prior conversation. `shellcheck` was installed and run across all scripts. It
immediately caught a real bug in `pod_health.sh`: inside a `while read` loop, an
`ssh` call was consuming the loop's piped stdin (SC2095), so the "watch every
pod" health monitor was silently only ever processing the first pod in its list
— the monitoring tool was not covering what it claimed to cover. This was fixed,
along with cd-guards, an unused variable, and quoting issues found in both live
and defunct scripts across the repository. Shellcheck-clean was established as a
non-optional requirement for all shell scripts, enforced via `scripts/lint.sh`,
and recorded in RELIABILITY.md; the full lint pass was brought to green and
committed.

## SSH-detach hang blocking the sequential launcher

A health-check tick showed the launcher stuck on w1 (never advancing to w2) with
w1 itself showing what initially looked like 3 racing processes and a stale
UNSUPPORTED result. Investigation distinguished the two: the "3 procs" was a
normal probe-run process tree, and the UNSUPPORTED result was stale from an
earlier race already addressed. The actual bug was that `launch_pod.sh`'s final
`nohup … &` SSH invocation was hanging because its stdin was not redirected,
blocking the sequential launcher indefinitely on w1 and preventing it from ever
reaching w2–w16. This is the same class of SSH-detach hang encountered earlier
in the session. The fix (`</dev/null` on the SSH command plus `disown`) was
applied, shellchecked, and verified to actually resolve the hang — the launcher
was confirmed advancing all the way through w2–w16 without stalling.

A side effect of editing `launch_pod.sh` after the pre-flight gate had already
written its authorization token was that the fail-closed gate then correctly
refused all subsequent launches ("not green"), since the launcher's fingerprint
had changed. This was resolved by re-writing the gate token against the updated
launcher; a single-model gate check was run in isolation to confirm it now
passed (exit 0) rather than inferring success from ambiguous, partially-stale
log tails.

## Pod-health monitor found to be silently blind on endpoints

While confirming launches were completing, the health monitor was found to be
misreporting live, reachable pods as still in INIT state. Root cause: it
resolved pod endpoints from the raw cloud API, which was returning blank ports
for these pods, while the pods were in fact reachable — the correct source of
truth is the endpoint recorded in each pod's launch log. This was identified
explicitly as a case where a monitoring tool giving false readings is worse than
having no tool, since it produces false confidence. `pod_health.sh` was fixed to
resolve endpoints from launch logs instead of the API, after which it correctly
reported true per-pod state.

## Tier-1 canary catches an unpinned transformers major-version break

Once endpoint resolution was fixed, the health monitor immediately surfaced a
genuine new failure: w1's model load began failing with a `RuntimeError` during
weight loading. Reviewing this against the project's own recently-recorded
tier-1-canary rule (verify one model produces a clean end-to-end result before
fanning out to the full fleet) it was noted that the fan-out to 11 models had
already been started without first confirming any single model against the
freshly-repaired infrastructure — a violation of the rule the session had just
written into RELIABILITY.md. The fan-out was paused to get the complete error.

Root cause: the job script's `pip install -U "transformers>=4.57.0"` had picked
up **transformers 5.13.0**, a new major version with breaking changes
("automatic conversion of the weights" error), rather than the 4.57.x line the
model previously loaded under — an unpinned-dependency bug. `transformers<5` was
pinned in all job scripts and committed. The tier-1 canary was then run as
intended: w1 alone, redeployed with the pinned dependency, launched end-to-end
before any further fan-out, to confirm both that the model loads and that the
pipeline produces a sane probe result — this canary run was in progress and
being watched at the end of this conversation, with the health monitor (now
fixed) as the mechanism reporting its outcome.
