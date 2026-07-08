_This shard covers the RunPod-based cloud scale-up execution — parallelizing
across up to five pods, discovering and validating a per-KV-slot tuning result
before demoting it in favor of simplicity, running stage 1 (LongMemEval) to a
damage-quantification conclusion, catching and correcting an evidence-quality
mistake (citing an illustrative demo as data), redesigning the benchmark slate
around continuation/behavioral metrics instead of fact-retrieval QA, and
beginning the OpenHands end-to-end coding-agent track._

**Participants in this Conversation.**

User; `claude-fable-5` (Claude Code `2.1.200`).

## Parallelization and infrastructure hardening

The user approved multi-pod parallelization with up to a 20% cost premium for
speed, then pushed further, asking whether the pod count could go higher and
what the actual failure modes were. The assistant clarified that RunPod's
platform imposes no meaningful pod-count ceiling; the real constraint was
operational — hand-rolled SSH launches were the source of every incident so far
(a stray runner colliding with a relaunch, a CPU-offload stall from VRAM
contention, monitor false alarms from simultaneous SSH probes). The fix was a
hardened, idempotent launcher (provision → verify GPU/VRAM residency → sync →
launch with atomic resumable writes → self-register with the watchdog) plus a
consolidated single monitor loop: 5-minute silent result pulls (bounding data
loss to ≤5 minutes) with 30-minute tiered health checks
(stall/traceback/milestone detection), replacing the earlier noisy, redundant
monitors. A pre-commit hook was added to block staged files over 4MB, and the
repo was audited to confirm datasets/weights/credentials stay untracked while
docs, source, and JSON result artifacts are committed. The user set the actual
worst-case framing: financial exposure is capped by prepaid balance, so the real
risks are idle-tail waste and un-synced result loss on pod death, both mitigated
by the frequent auto-pull. The user approved scaling to nine pods, established a
90-minute heartbeat backstop wakeup, and set a standing rule that gates should
throttle **spending**, not **effort** — specific approaches can be abandoned,
but the overall experimental goal cannot; when stuck, the assistant is directed
to consult a second-opinion model (Codex CLI, not currently on PATH, noted as an
overnight gap) via CLI at highest settings for a differently-trained
perspective.

## Tuning track: per-slot discovery, demotion, and factorization discussion

Pod-4's tuning chain (α sweep → per-layer profile → per-KV-head profile →
holdout) produced a striking result: at 30B/bf16, α≥1.0 ("extrapolation,"
pushing values away from the fresh/compacted-context reading) beat α=0.75 in the
validation sweep, explained via the
classifier-free-guidance/contrastive-decoding analogy — subtracting a biased
"read against the summary" signal sharpens the true-context direction.
Separately, the 192-slot (48 layers × 4 KV heads) per-head profile revealed that
although the aggregate profile looked diffuse (no dominant band), a specific
scattered subset of 57 "positive" slots, grafted alone, decisively beat the
incumbent global-blend rule on holdout: +0.038 nats vs +0.024 nats, winning all
10 held-out conversations vs 7/10. The assistant flagged this as still needing
its pre-registered contamination guard (wrong-conversation values through the
same 57 slots must fail, or the finding is an artifact) before being trusted.

The user, after understanding the magnitude, made a firm methodology call: this
per-slot mask should **not** become the primary/promoted approach — it risks
overfitting and unnecessarily complicates the narrative — but should be
documented as a secondary exploration section. The user proposed a simpler
alternative: a rank-1 factorization (one coefficient per layer, one per head,
multiplied together), reasoning from remembered linear-algebra intuition that
correlated/entangled directions might require negative coefficients to reach
certain solutions, and that clamping at zero could be discarding real degrees of
freedom. The assistant validated this reasoning as formally correct in the
linearized regime (transformer heads write into overlapping/non-orthogonal
subspaces via superposition, so optimal joint corrections can require negative
coefficients) but showed empirically, via an SVD check on the measured 30B
profile matrix, that (a) head factors are nearly flat (1.4% of variance) —
confirming the user's intuition that heads don't matter for these models — but
(b) the matrix is far from rank-1 (top singular component only 37% of variance;
rank-1 R²=0.31), meaning ~70% of slot-to-slot variation is
idiosyncratic/noise-like rather than layer×head-factorable. A follow-up local 4B
experiment (clamped non-negative fit vs signed/unconstrained fit) showed the
clamped fit reliably beat the signed fit (10/10 wins, mean +0.0074 vs 6/10 wins,
mean +0.0036) and both underperformed the simple hand-tuned mid-band rule —
empirically settling the discussion in favor of the statistical
(variance/collinearity) argument over the geometric one, in a result the user
found amusingly "aggressive" in how decisively it refuted their own hypothesis.
All of this reasoning — the user's original geometric framing, the assistant's
counter-argument, and the resolving data — was written into a dedicated
design-notes document (not folded silently into other docs) per explicit
instruction, with the factored (layer×head) parameterization retained as the
calibration interface planned for architecturally head-heterogeneous models like
Gemma, where head factors are expected to actually deviate from 1.0.

An evidence-grading rule was also codified per user direction: small-scale/4-bit
local results are hypothesis generators only — nulls or boundaries observed
there don't prune the hypothesis space for larger models, and even positive
small-scale findings are "candidates for validation," not claims, until
confirmed at 30B/bf16 on standard data.

## Stage 1 results and a self-corrected efficiency failure

Stage 1 (320-question subsampled LongMemEval, six arms, 13 Sonnet judge batches)
completed: full-context accuracy 52.5% collapsed to ~3–7% under compaction
across all five non-oracle arms, with essentially no differentiation between
arms on fact-retrieval accuracy, and flat fabrication:admission ratios across
arms (E-tuned's graft did not increase fabrication relative to plain compaction
— a safety-relevant null). The user immediately identified this as an
inefficient use of compute: the local pilot data had already predicted that
QA-framed fact retrieval washes out arm differences at 30B, yet the cloud run
was scaled to n=320 across six arms anyway, costing roughly $8–10 more than
necessary. The assistant accepted the critique directly, attributing it to not
applying an interim-analysis gate, and immediately wrote a corrective rule into
DECISIONS.md: cloud replications of locally-predicted nulls should run at
minimum viable n with an interim gate, not full n and full arm count.

This critique cascaded into a larger redesign discussion. The user objected to
continuing to test precise fact-retrieval, since both prior evidence and
intuition already suggested the intervention would not help there. This led the
assistant to reconsider the two remaining benchmark stages (stage 1b
full-haystack LongMemEval, and stage 3 LoCoMo) since both were QA/fact-retrieval
framed — the same dimension already shown to be a predictable null. Proposed and
(in-progress) adopted replacement slate: a cheap A/B-only stage-1b "mini" run
(~40 questions) purely for a citable standard-protocol damage number; deepening
stage 2 (more real SWE-Gym/R2E-Gym trajectories) since that's where measurable
recovery has actually appeared; a bf16 rerun of the honesty/fabrication
decoy-probe suite (previously only run at 4-bit); continuing the Gemma
exploration (unaffected, since it uses continuation-based metrics, not QA); and
dropping LoCoMo-as-QA. In the same exchange the user caught and corrected a
serious evidence-hygiene error: the Pokémon inversion demo (a hand-built n=1
illustration made to explain the concept to a non-technical friend) had been
cited by the assistant alongside real controlled experiments as evidence for
"where the effect lives." The assistant acknowledged the category error, deleted
the demo artifacts from the `results/` tree (retained in git history, with any
future public-facing use requiring explicit relabeling and separate approval),
and recorded a standing rule that illustrative demos never belong in the results
tree or in evidence enumerations.

## OpenHands end-to-end coding-agent track (stage E)

Following up on the earlier-floated "nuclear option" (a standard-harness,
execution-based test of the intervention), the user pushed to parallelize this
work across pods for earlier signal. The assistant laid out the real
architecture: OpenHands (agent harness with a pluggable condenser hook for
compaction) driving real coding tasks against runnable SWE-Gym unit-test suites
(objective, binary pass/fail outcome, no LLM judge needed), with a custom
OpenAI-compatible serving shim wrapping the existing HF transformers/graft
machinery hosted on a pod, and the harness plus Docker sandboxes running locally
(avoiding docker-in-pod reliability problems) while all LLM inference routes to
pod-hosted shims over HTTP — making N pods = N concurrent task streams
naturally. A staged plan was proposed: E0 (shim build + one-task smoke,
~$3), E1 (pilot, ~10–12 compaction-forcing tasks, B-vs-E interleaved across 2–3 pods, ~$15–20,
pre-approved), and E2 (the real n≈30 run, ~$60) gated on E1 showing clean
plumbing and pre-registered process-metric separation (repeated-failed-command
counts, evicted-file re-reads, steps-to-completion) — with the gate explicitly
only throttling spend, not effort, per the user's standing rule.

The user then directly confronted a sequencing failure: despite establishing
that the coding track was a must-do and should be started immediately, the
assistant had let several smaller unblocking tasks (warm-pod handoffs, a Gemma
ladder retry) preempt starting the shim build, and called this out as the
conversation "losing its spark." The assistant acknowledged the sequencing error
plainly and started the E0 shim build as the uninterrupted priority, with the
creative-probing mandate ("try many cheap evaluation angles in parallel, not
execute one plan solemnly") written into the charter. By the shard's close, the
E0 shim was committed, smoke-tested, and its pod (e1) was loading the 30B model;
the runbook for interleaving B/E-mode tasks across additional shim pods was
committed; and Gemma's pod (g1) had been relaunched with a properly scoped
ladder after an earlier launch failure, with the hybrid-architecture rotation
prerequisite recorded as a finding in its own right (compaction surgery on
interleaved sliding-window/global-attention models requires layer-type-aware key
rotation).

## State at shard boundary

Running concurrently: pod-1/pod-2 finishing stage 1 (completed 350/350 by end of
shard, then immediately handed stage-1b-mini and the bf16 honesty suite
respectively), pod-4 mid-4B-bf16 calibration block (queued next: slot-mask
contamination guard, then Mistral pre-tuning), pod-g1 running the Gemma-3-4B
identity ladder and profile sweep, and the new shim pod (e1) loading the 30B for
the OpenHands E0 smoke test. Local box: idle except analysis and the
shim/OpenHands build. Balance was last logged around $77 with burn near $4/hr
(rising toward higher rates as the fan-out proceeds). All state/decision
documents (STATE.md, DECISIONS.md, AGENTS.md, cloud-plan.md, and the newly
created value-steering design-notes document) were kept current throughout,
explicitly written for a cold-start successor agent, motivated by the user's
disclosed risk of hitting a weekly usage quota and potentially handing off
mid-run to a different model (GPT-5.5) via the documented state. The shard ends
with the user checking in ("How is it going") as the assistant is mid-build on
the E0 shim with five machines active.

---
