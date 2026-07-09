_This chunk covers a discipline correction on consulting Fable before locking in
scaling decisions, Fable's redesign of the render-cost fix (batched decode +
KV-cache snapshot-replay), a batch-size OOM and its resolution, a bf16-noise
regression that forced reverting to the validated per-token render for the
sweep, and — most consequentially — the discovery of a long cascade of
infrastructure bugs in the 16-model wide sweep that led to building a full
reliability-engineering layer (pre-flight gate, health monitoring, mandatory
shellcheck) before finally launching a single-model canary run._

**Participants:** User and claude-opus-4-8.

Participants: user (Franklin) and Claude Opus 4.8 (claude-opus-4-8) via Claude
Code.

**Consultation discipline.** Early in the chunk the user flagged that a scaling
decision (conv counts, reply length, deep/shallow split) had been made without
consulting Fable, repeating a pattern already documented as a hard rule. The
agent corrected course and routed the scaling question to Fable before banking
any plan.

**Fable's scaling verdict.** Fable determined the agent's initial instincts
(cutting conversation counts, shortening replies) were the wrong levers —
shortening replies trades render time for statistical power exactly where power
is already thin. The actual cost sits in the per-token decode loop (sequential,
batch=1). Fable's recommended fix: batch the decode across a model's
conversations (~10x candidate speedup) plus render-once/snapshot-replay across
arms (rendering up to the compaction boundary once, then replaying variant arms
from the KV snapshot — a free optimization the agent had missed). Fable also
approved a deep/shallow anchor design (24 anchors deep, 12 breadth) conditional
on using continuous normalized effect size as the regression outcome rather than
a hard sign threshold, precision-weighting the regression by each model's
bootstrap SE, and keeping render depth equal within de-confound pairs. Minimum
viable was set at 12 conversations/model (24 anchors).

**Design validation.** Before the scaling work landed, the native-render
verification (referent CI [+0.012, +0.195], mid +0.10) completed and reproduced
the original pre-rendered effect, confirming the per-model-native redesign (own
in-context replies, shared gold, self-generated summary) is sound. The
referent/sense/stance dissociation reproduced, and the `ruled_out` validity gate
correctly floored and excluded a low-headroom category rather than misreading it
— validating the gating logic on real data.

**Render scaling implementation and setbacks.** Batched decode was built and
CPU-verified byte-identical to the per-token path on a small model. GPU testing
was repeatedly blocked by pod infrastructure problems: SSH detach/launch
patterns that had worked before (inline `nohup … &`) stopped working reliably on
a heavily-reused pod (11793), which was eventually judged degraded and
terminated in favor of a fresh pod. A first fresh-pod attempt crashed instantly
because `launch_pod.sh` did not rsync `data/scenarios.json` to the render
scaffold — the same gap that had been hand-patched on the prior pod — and was
fixed permanently in the launcher. A subsequent GPU run of batched decode showed
a real but modest speedup (~5x) before hitting a CUDA OOM: batching all 12
conversations' KV caches at once (~16GB) on top of the ~60GB model exceeded the
80GB card. Fix: a batch-size cap (`SC_NATIVE_BATCH`, chunking to ~4
conversations with OOM-retry that halves further if needed), CPU-verified as
producing byte-identical chunk boundaries.

**Batched-vs-per-token decision.** A batch-capped re-run completed at ~2.3x
speedup (not the hoped-for ~4x) but the referent effect weakened and its CI
widened to span zero (+0.0485, CI [-0.053, +0.150] vs the validated +0.10 CI
excluding zero), and `evicted_fact` swung to -0.31. Consulted with Fable: the
referent shift was judged noise (overlapping CIs at n=24), but the evicted_fact
swing was flagged as the real tell — a 0.28-point swing landing on a CI that
flips sign is too large for benign bf16 rounding and indicates batching can
systematically reroute decision-relevant tokens, threatening the sign-map claim
that is the paper's central result. Fable's ruling: run the sweep on per-token
(validated, clean) rather than batched, since the batched method's only
advantage (~2.3x pod-hours) doesn't translate to wall-clock savings under hard
parallelization, so it wasn't worth risking fidelity to save cost. A cheap
off-ramp was proposed (fp32 logits for the argmax only, to stabilize
matmul-reduction-order-induced tie-breaks) as a candidate for later, but not
adopted for the immediate sweep.

**Budget check.** Before firing the 16-model sweep, the agent checked spend
against the ~$80 cap. Budget-conscious 12-conv-per-model sweep: ~$48. Full
24-conv-on-anchors/12-on-breadth version: ~$60, tight against a ~$65-70 balance.
The user chose the full/tighter version to preserve anchor statistical power,
accepting reduced margin. The user also stated an explicit priority ordering
going forward: when new results land, get Fable's review first for possible
minor tweaks, but do not let render-perfectionism delay starting to collect the
other-model (cross-architecture) data, which is the actual deliverable.

**Pre-registration finalization.** Before firing the sweep, the agent fixed a
bug where QK-norm detection read config keys that don't reflect how
Qwen3/Gemma/OLMo-2 actually implement QK-norm (as `q_norm`/`k_norm` modules, not
config flags) — the detector had been silently returning False for all models.
Fixed by re-detecting from the loaded model's modules after load (since the
config-based check runs before load, intentionally, so geometry survives a load
failure). This unblocked finalizing and freezing the pre-registered hypothesis
(H1: QK-norm presence predicts a positive graft sign) before any sweep results
existed, preserving its confirmatory status.

**Wide-sweep infrastructure cascade.** Launching the 16-model per-token sweep
surfaced a long sequence of real bugs, each caught and fixed in turn: (1)
`launch_pod.sh` didn't forward `MODELS`/`SC_CONV_LIMIT`/other `SC_*` env vars to
the remote job, so every pod's job died immediately; (2) the per-token job was
wrapped in a fixed `timeout 3600s`, insufficient for ~5-hour anchor renders —
fixed to scale timeout with conversation count; (3) repeated manual re-launches
caused multiple job processes to race on the same GPU, causing an apparent OOM
that was actually resource contention, not a real batching problem; (4) the user
directly questioned whether the design meant waiting 3-5 hours per model with no
early signal of breakage — this triggered building an early-signal/probe
mechanism (score 3 convs first, ~40 min to a real sanity result, before
committing to the full multi-hour render); (5) peeking at running pod logs
immediately surfaced that every pod was failing on `FileNotFoundError` for
`scenarios.json` — the earlier rsync fix hadn't actually taken because rsync
dropped the `data/` path prefix, landing the file at the wrong location; this
was fixed at the root; (6) checking actual HuggingFace model classes revealed 5
of the planned 16 models are multimodal (`ForConditionalGeneration`) rather than
plain causal LMs, including both models in a primary de-confound pair, which the
harness's `AutoModelForCausalLM` loader could not handle — Fable was consulted
and judged that recovering the cross-vendor pair member (not the same-vendor
one) was worth a time-boxed, positive-control-gated spike, while an
11-text-model regression plus the surviving single de-confound pair was judged
an honest, publishable result on its own even without recovery; (7) further
churn from multiple concurrent launch/retry mechanisms overwriting each other's
logs made pod state unreadable from logs, forcing a switch to querying the
provider API directly as the source of truth; (8) the agent's first attempt to
conclude the multimodal models were simply unusable was itself an unverified
assumption (per user pushback) — the wrapper classes likely still support
text-only loading, with the real constraint being a graft compatibility check on
the text-decoder tensors inside the wrapper, not an outright block.

**Reliability engineering buildout.** In response to the accumulated pattern of
failures only surfacing when someone manually checked, the user reframed the
core problem as a missing observability layer, not insufficient upfront testing.
The agent and Fable jointly built: a fail-closed pre-flight gate
(`scripts/preflight.py` / `job_gate.sh`) that verifies model loadability, file
presence, launcher correctness, and env/timeout/race-safety before any spend is
allowed — proven to catch the multimodal-model problem in ~20 seconds; a
continuous pod health-check and alerting monitor (`pod_health.sh`) classifying
every pod's state (rendering/errored/stalled/died/GPU-idle/done) and firing only
on bad states; and a hard-rule doc capturing this discipline. This was initially
misplaced into a Claude-specific config file that doesn't match the project's
actual documentation structure (purpose-specific docs referenced from AGENTS.md)
and was corrected into a dedicated RELIABILITY.md, referenced from AGENTS.md,
with CLAUDE.md removed. The gate itself was found to have a stale/wrong
checkpoint baked in (the thinking variant of a model instead of the intended
Instruct checkpoint) and was corrected before use.

**Shellcheck adopted as a mandatory rule.** Prompted by the user, the agent
installed and ran shellcheck across all shell scripts, which caught a real bug
in the newly-built health monitor itself (`ssh` inside a `while read` loop
consuming piped stdin, causing the "watch every pod" monitor to silently watch
only the first pod) plus a `declare -A` runtime failure on macOS bash 3.2 that
`bash -n` syntax-checking had missed. All scripts were brought to
shellcheck-clean and this was recorded as a non-optional hard rule in
RELIABILITY.md, enforced via `scripts/lint.sh`.

**Further launcher bugs and a canary-discipline lapse.** Fixing the pre-flight
gate and relaunching still hit two more issues: a hung `launch_pod.sh` SSH call
(missing `</dev/null` stdin redirect) that blocked the sequential launcher from
ever advancing past the first model, fixed at the root; and a stale pre-flight
token invalidated by the launcher-script edit itself (the fail-closed interlock
correctly refusing to proceed until re-verified). After finally getting all 16
models launching, the agent recognized it had violated its own newly-written
tier-1-canary rule by fanning out to 11 models before proving even one worked
end-to-end on real infrastructure — at which point a live model-load failure
surfaced: `transformers` had been auto-upgraded to an incompatible 5.13.0
(breaking weight-conversion changes) because the job script's version pin
(`>=4.57.0`) didn't cap the major version, and a subsequent attempt to force a
downgrade also silently failed to take (compounded by unreliable SSH on that
pod). All fan-out pods were terminated to stop billing (~$3.57/hr for
non-functional runs) rather than continue guessing.

**Tier policy and canary restart.** Given the day's infrastructure
unreliability, the user opted to prioritize reliability over cost and directed a
switch from RunPod's "community" tier to "secure" tier. This was recorded in
DECISIONS.md as a policy: use community tier when cheap and healthy (favors high
parallelism), cap secure-tier usage at 3 pods when community shows
capacity/reliability problems, and always canary on secure. The transformers
dependency bug was hardened with a fail-closed check (force-reinstall to the
known-good pinned version, verify the installed version is below 5.x, and
hard-exit refusing to run the model if the pin didn't take, rather than silently
proceeding on a bad dependency as before). A single secure pod was then
provisioned and, at the point this chunk ends, a single-model canary run
(Qwen3-30B-A3B, 3 conversations) was in progress with a monitor watching four
checkpoints in sequence: transformers version verified below 5.x, model load
succeeds, render progresses, and the 3-conversation referent number reproduces
roughly +0.10. The stated commitment is that only a full pass on all four
checkpoints authorizes fanning back out to the remaining models (via community
tier if healthy, otherwise capped secure); any checkpoint failure will be
reported directly rather than patched ad hoc. No wide-sweep results have yet
been obtained as of the end of this chunk.
