_This chunk covers a single extended overnight session on the
value-steering/compaction-recovery project: closing out stage-1 benchmark
results, a design discussion redirecting slot-mask and factored-parameter tuning
to a secondary "exploration" track, a major pivot away from fact-retrieval QA
benchmarks toward agent-coding evaluation (the "E-track"), a scale-up from 4 to
7+ RunPod GPU instances, and a sequence of monitoring/reliability failures and
fixes culminating in a formal handoff protocol alternating shifts with a
Codex-driven agent._

**Participants:** User and claude-fable-5.

**Extrapolation and incoherence characterization.** The extended α bracket
({1.0,1.25,1.5,2.0,3.0}) was queued on pod-4 and later confirmed a smooth
decline from the 0.75 peak with no cliff. Incoherence from corrupted cache
values at 30B was found to be almost never token-level gibberish (that only
appeared at 4B under full shuffled grafts) but semantic: fluent-but-ungrounded
"amnesia" answers, confident denial of true evicted facts, and topic bleed from
donor conversations — a more dangerous failure mode since fluency gives no
signal of context corruption. A Pokémon-themed four-condition demo (full context
/ deleted context / technique-applied / inverted-technique) was built to
illustrate the mechanism for a non-technical audience; the user later corrected
that this demo is illustrative only (n=1, hand-constructed, greedy-decoded) and
must never be cited as evidence — it was removed from `results/` and a rule was
recorded (in DECISIONS.md) that demos never live in the results tree or appear
in evidence lists. The user asked for the demo to be fleshed out into a richer,
longer standalone illustrative conversation for future reuse (done as
`demos/pokemon-demo-conversation.md`, explicitly quarantined).

**Slot-mask and factored-tuning exploration.** Per-slot profiling on the 30B (48
layers × 4 KV-heads = 192 slots) found 57 slots individually helpful; a
"posslots" rule (graft only those 57 at full strength) beat the incumbent
global-blend rule on holdout (+0.038 vs +0.024 nats, 10/10 vs 7/10 conversations
won). The user directed that this finding be demoted from primary approach to a
secondary "exploration" section, to avoid complicating the primary claim, but
should still be pursued, documented, and included in the eventual write-up. The
user proposed a rank-1 factored parameterization (one value per layer, one per
head, combined multiplicatively) as a low-parameter alternative; a check on the
actual 30B profile matrix showed head effects are near-flat (head-mean explains
1.4% of variance, confirming the user's intuition) but the matrix is far from
rank-1 (top component only ~37% of variance, R²≈0.31), meaning ~70% of
slot-to-slot variation is idiosyncratic/noisy given only 10 profiling
conversations. The user separately raised a geometric argument
(correlated/non-orthogonal directions can require negative coefficients to reach
an optimal joint solution) for allowing signed (unclamped) coefficients; the
assistant agreed the geometry was valid in principle but argued statistics
(collinearity/overfitting risk with only 10 conversations) and nonlinearity
(trust-region bounds from observed α degradation) would likely dominate. An
empirical clamped-vs-signed holdout test on the 4B model confirmed the
statistical concern won in practice: clamped factored fit (+0.0074, 10/10 wins)
beat signed factored fit (+0.0036, 6/10 wins), with both underperforming a
simple hand-tuned mid-band rule (+0.0173, 10/10) — flagged as a
small-scale/4-bit result only, not proof signed fits can't help at larger scale.
An evidence-grading rule was formalized: local/4-bit/small-model results are
hypothesis generators only; nulls at small scale never prune hypotheses for
larger models; positive results only earn "candidate for validation" status;
30B-bf16-on-standard-data is where claims are earned.

**Stage-1 results and course correction.** The full stage-1 LongMemEval run (320
questions × 6 arms, all 13 judge batches) landed: full-context accuracy 52.5% vs
compacted arms all in the 2.8–6.9% range, with no meaningful separation between
arms (B, E-tuned, B-min-pack, H-pack, H-gap) on fact-retrieval accuracy, and
grafting did not increase fabrication rates relative to plain compaction. The
user assessed this as an inefficient use of compute, noting the project's own
prior local data had already predicted that QA/fact-retrieval framing would wash
out arm differences at 30B, and that roughly half the ~$18 stage-1 spend bought
information already known. This triggered a broader redesign: stage-1b
(115K-token haystacks) and stage-3 (LoCoMo) were both originally planned as more
fact-retrieval QA, which would repeat the same predicted null at higher cost. A
hold was placed on all new pod provisioning pending realignment. The revised
slate: stage-1b reduced to a cheap "mini" (A/B only, ~40 questions,
standard-protocol citability number only); LoCoMo dropped from the plan; effort
redirected toward where recovery effects had actually shown up — deepening
stage-2 (more real SWE-Gym/R2E-Gym agent traces), a 30B-bf16
honesty/fabrication-decoy suite (replicating a 4-bit finding of dramatic
fabrication reduction under the packed/graft arm), Gemma architecture-transfer
exploration, and the coding-agent end-to-end track. A general rule was adopted:
when local evidence predicts a null, cloud replication should run at minimum
viable n rather than full n.

**Pivot to agent-coding evaluation (E-track).** This became the project's
central new direction. Architecture: OpenHands (agent harness) and Docker
sandboxes run locally (avoiding unreliable docker-in-pod), while all LLM
inference is served by an OpenAI-compatible shim hosted on RunPod GPUs,
implementing compaction+graft server-side at the point where compaction
naturally fires (triggered via a condenser hook rather than prompt-sniffing).
Objective scoring uses pytest pass/fail on SWE-Gym-style tasks (no judge
needed), plus process metrics (repeated failed commands, re-reading evicted
files, steps-to-completion) and a "dissociation probe" (asking the agent to
recite the exact constraint post-compaction) to test whether grafts restore
behavioral compliance without factual recall — directly extending the earlier
Pokémon "sense not facts" observation into a controlled, objective, real-agent
metric. A staged plan (E0 smoke → E1 pilot ~12 tasks → E2 full run ~30–50 tasks)
was adopted, with E2 originally gated on pre-registered pass/fail criteria; the
user later clarified that spending gates should never mean giving up on the
experiment's overall goal — approaches can be killed, but the mission (getting a
real answer) must continue via alternative angles, redesign, or cross-model
consultation, only spending should ramp down when yield is low. Verified
mechanism: the shim's compaction path reuses the exact validated experimental
code (same summary generation, alignment, and blend_values at α=0.75), confirmed
via live request logs showing correct compaction flags and graft position
counts. A throughput optimization (freezing the compaction boundary at first
trigger and caching the summary, matching production compact-once-then-continue
semantics, roughly doubling B/E throughput) was designed, locally smoke-tested
(catching and fixing a boundary-margin bug), verified bit-exact, and staged for
deployment at the next queue changeover without mixing semantics mid-comparison.

Early E1 round-1 results: compacted-only (B) agents failed both tasks (0/2),
grafted (E) agents passed both (2/2) — flagged as promising but statistically
weak at n=2 pending rounds 2–3 confirmation (full sign-test significance would
require the pattern to hold across all rounds, roughly p≈0.03 if the pattern
held to 6/6 vs 0/6).

**Scale-out and infrastructure.** Following user pressure to increase
parallelism (the "limit of 4 pods" was determined to be an operational
bottleneck, not a platform cap; RunPod supports single-digit A100 allocations,
with spot-availability the main friction, addressable via secure-tier pricing at
~20% premium the user explicitly approved), the assistant built a hardened,
idempotent pod-launcher (provision → verify GPU/VRAM → sync → launch with
resumable atomic writes → self-register with the watchdog) and scaled to a
target of up to nine pods, later described as a "paced" plan (3–5 pods
sustained, not blitzed) after the user pushed back on over-aggressive spend
framing, and then scaled further to seven live pods (four shim pods running the
matrix, one wildcard/creative pod, one honesty-suite pod, one calibration pod)
at the user's continued insistence on more parallelism. A background retrier was
added to keep attempting an additional pod (e3) every 30 minutes after repeated
RunPod 500 errors, without blocking other work.

An expanded "matrix" experiment design was built per user request for
creative/exploratory breadth beyond the core B-vs-E comparison: dose-response
arms (α ∈ {−0.5 (anti-graft/inversion), 0.5, 0.75, 1.0, up to 2.0 on the
wildcard pod}), shuffled-graft negative controls, compaction-threshold variants
(6K vs 9K), and the recall-dissociation probe applied to every run, plus a
dedicated wildcard pod running looser exploratory variations
(summary-less/vector-only grafts, alpha-annealing schedules, double-graft
compounding) with results logged to a separate journal.

**Parameter-tuning ladder and holdout discipline.** The user requested a formal
tuning program be added as a first-class part of the results: a ladder of B (no
intervention) → E single-α → E per-layer coarse tuning → E slot-mask, evaluated
as matrix arms using configs derived from the pod-4 bf16 profiles, with a
champion/challenger promotion discipline (promotion only after repeated wins on
fresh tasks, ≥25% of runs continuing as baseline-guard checks after any
promotion, all transitions dated in DECISIONS.md). A three-phase evaluation
pipeline was formalized: explore (current matrix, seeds s1–s5, promotions may
use up to s49) → confirm (side-by-side comparison of no-compaction vs
plain-summary vs champion vs single-α on ~15–20 fresh tasks per arm) → final (a
single sealed run against seeds s50–s99, held out from all tuning/exploration,
pre-registered before any ladder results existed, reported however it lands).
This confirm phase is expected to launch automatically once the matrix crowns a
champion.

**Cost and risk framing.** The user pressed for an explicit worst-case analysis
of scaling pod count; the assistant's answer: total financial exposure is
hard-capped by prepaid balance (no possibility of runaway billing), the main
realistic risks are idle-tail waste (a few dollars per incident) and un-synced
results lost if a pod dies before a pull (addressed by moving to 5-minute
automated result pulls, later further tightened), with no scenario risking
permanent data loss for already-committed work or silent scientific corruption.
Local (4-bit MLX, unlimited/free) vs pod (A100 bf16, ~$1.39/hr+) tradeoffs were
characterized: pod enables full-precision, huge-context (115K+ tokens
demonstrated), and larger models; local wins on cost-free exploratory sweeps;
same-model-same-precision throughput favors pod by roughly 5–10x, with local
4-bit vs pod bf16 closer to parity on decode-heavy work but pod 3–5x faster on
long prefill.

**Monitoring/reliability failures and fixes.** A recurring theme was inadequate
failure detection: several jobs (a Gemma launch, a probe-data job, a LongMemEval
mini run) failed silently or stalled without alerting, in one case for ~35
minutes, prompting repeated user correction. Fixes implemented over the session:
consolidation of a 30-minute alerting watchdog and a 5-minute silent
result-puller into one 5-minute-tiered loop (pulls + failure-unreachability
check every 5 min, deeper stall/traceback/milestone checks every 30 min); a
90-minute backstop heartbeat in case the assistant itself goes idle; a
pre-commit hook blocking any staged file over 4MB; explicit dead-process
detection (job process dead without a completion marker) and new-error-line
detection added as instant triggers within the 5-minute loop; job scripts
required to end with count-validated completion markers (`DONE (n)` vs
`INCOMPLETE (n)`) so silent-empty runs can no longer masquerade as success; and
removal of the shim pod's prior exemption from dead-job alarms, which had
created a blind spot during a multi-pod restart window that the user had to flag
manually. Root causes diagnosed during the session included: OOM from an
unconditional KV-cache snapshot clone during grafting even when not needed
(fixed to be conditional), a hard 85K-token cap that made every full-protocol
LongMemEval haystack (110–120K tokens) physically exceed single-A100-80GB bf16
capacity for a 30B model — resolved by moving that specific mini-run to a rented
H200 (141GB) — and repeated Gemma integration failures (six attempts) traced
successively to dual RoPE bases, strict chat-template role-alternation rules,
assistant-note ordering, a Qwen-specific assumption buried in shared summary
code, and finally a hybrid sliding-window attention mask-construction bug in the
scoring path, at which point Gemma work was parked with a precise resume point
rather than continuing to burn pod time on it. A new operating rule was adopted:
for any new tokenizer/model integration, run a full local dry-run of the actual
context-construction code path (not just template rendering) before spending any
pod time, since all early Gemma failures were tokenizer-level and reproducible
locally in seconds.

**Multi-agent handoff protocol.** Given the user's finite weekly usage quota,
STATE.md was rewritten to be readable by a cold-start successor agent with zero
shared context, and AGENTS.md was updated to note that Claude and a separately
invoked Codex-CLI agent are both available on the machine; the assistant is
strongly encouraged to consult a different-model agent (via CLI, ideally at the
highest available effort/model tier, potentially resuming an extended session)
when stuck, since a differently-trained model can surface angles the primary
agent misses — though the current environment did not have `codex` on PATH at
check time. The user separately arranged for a second agent (apparently
Codex-driven, per a summary line the assistant identified as not its own) to
operate read-only on the repository except for creating new files/folders in
non-conflicting locations, which it commits directly without touching the
assistant's staging; the assistant adjusted its own git habits (avoiding
`git add -A`) to avoid conflicting with this. By the end of the chunk, the user
proposed alternating 30-minute (then shorter) work shifts between the assistant
and the Codex agent overnight, contingent on an "impeccable" handoff document;
the assistant fixed the dead-job-alarm blind spot, verified all matrix lanes
were confirmed running with health-checked tunnels and first scores expected
within ~10 minutes at an anticipated aggregate rate of ~30–35 scored coding
runs/hour, logged status to a SHIFT-LOG, and went to sleep for a stated
~25-minute scheduled interval, planning to verify on wake that scores were
landing before handing back to the next shift.
