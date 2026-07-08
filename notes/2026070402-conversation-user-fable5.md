_This note covers a multi-day local experiment on whether write-time KV-cache
state, preserved or transplanted across conversation compaction, restores
conversational meaning lost when production systems re-encode a text summary
from scratch, followed by a mid-run pivot from a mechanism-focused design to a
mitigation-focused one, and the start of a real-dataset (LongMemEval) validation
pass with a cloud scale-up plan queued for user review._

**Participants:** User and claude-fable-5.

Work ran on a local Apple Silicon machine using MLX, primarily Qwen3-4B
(development) and Qwen3-30B-A3B (production), with Sonnet-based subagents used
for scenario authoring, judging, and analysis review, reserving the local
subject model for dialogue generation only (judging and scoring are near-modal
against the subject model's own text, so mixing generators would add noise to
the core logprob metric). All surgery machinery (gapped-cache retention, value
transplant, key re-rotation) was validated against an identity-test ladder
(L0–L4, later extended with LH-1/LH-2 for re-rotation) before any result was
trusted, and negative controls (wrong-conversation grafts, shuffled-value
grafts) were run to rule out generic smoothing effects.

**Original design and early results.** The initial brief defined five compaction
arms (A oracle, B production-style text compaction, C gapped KV retention, D
no-summary ablation, E value transplant) scored on continuation log-probability
("continuation gap": A−B measures how much compaction hurts next-token
prediction, closure toward A is the improvement metric) and on planted probe
questions (referent, sense-disambiguation, stance, ruled-out, evicted-fact).
Three companion documents were read and committed early: follow-up explorations
(Arms G/H, "ValueGraft"/"SelfGist" naming), a phase-2 scale-up/coding-agent
extension plan, and an external-review amendments document (B-causal control,
negative controls, six-class leakage labeling, contradiction probes,
metric-hierarchy flip to probe-accuracy-as-headline, literature positioning
against "Models Take Notes at Prefill" arXiv 2606.17107 and other neighbors). A
12-conversation synthetic corpus (120 hand-planted probes) plus 8 natural
conversations were generated, repaired for tail-content leakage (115/120 plants
ended clean), and run through the full arm ladder at 4B, then replicated at 30B.
Sonnet-based judge batches (moved off Haiku per explicit user instruction to
prefer Sonnet) scored probe accuracy; a "brief/shadow" terse-summary condition
was added to guarantee plants were absent from the summary, sharpening the probe
contrast.

**Mid-run refocus.** Partway through, the user (relaying input from an external
reviewing model) redirected the project: the mechanism claim ("write-time KV
differs from re-encoded text") was judged close to self-evident and not the
valuable question. The reoriented goal became strictly practical: can a small,
deployable cache-state intervention reduce compaction damage relative to
text-only summarization, with success requiring (a) a documented A>B compaction
failure, (b) a small intervention beating B, (c) ruling out
leakage/graft-control/fluency explanations, and (d) plausible deployability. Per
explicit instruction, the response was staged in two phases: Phase 1 closed out
the in-flight 30B run to an interpretable stopping point (with
STATE.md/DECISIONS.md/RESULTS.md synced and committed) rather than expanding the
old arm matrix; Phase 2 designed one narrow contrast, chosen as **H-pack vs
B-min-pack** (identical packed summary tokens, only write-time vs fresh encoding
differs) scored primarily on fabrication-vs-honest-admission over evicted facts
plus a newly authored, collision-audited decoy probe set, with low-α ValueGraft
as a secondary comparison and Arm G/SoftGraft explicitly deferred.

**Key results.** At 4B: light value blending (α≈0.25) reliably helped
continuation NLL while heavy blending (α≈1) hurt; H-gap beat B-min on
continuation prediction (+0.093 nats, most conversations); C (gapped retention)
did not pay for itself on fluency. At 30B, patterns partly inverted and
strengthened: H-gap beat B-min in 12/12 conversations (+0.128 nats, CI excludes
zero), and full-strength value grafting (α=1) became net-positive (~29% gap
closure) rather than harmful, suggesting larger models tolerate key/value
decoupling better. Natural (non-synthetic) conversations showed almost none of
these effects, because their continuations depend much less on evicted content —
establishing that intervention benefit is conditional on continuation-dependence
on evicted material, not a blanket effect. The clearest deployment-relevant
finding was on fabrication: packed write-time summary state (SelfGist/H-pack)
sharply reduced confident fabrication of never-discussed decoy facts relative to
standard text compaction, with the effect strongest and most cleanly
judge-confirmed at 30B (roughly 19:5 fabricated:admitted ratio favorable
comparison, decomposed into a layout-driven "admits ignorance" component present
even without write-time encoding, plus an encoding-specific component that grows
with scale). A subsequent finer α-sweep (guarded by a validation/holdout split,
per external reviewer suggestion, rather than tuning on the reporting set) found
the optimal blend differs by scale: 4B favored a low, middle-layer-gated α≈0.25
(~10% gap closure on held-out conversations), while 30B favored a much higher α
(up to 1.25, i.e., extrapolating past the original values) for ~24% held-out gap
closure. Known unresolved gap: no intervention recovered accurate recall of
specific evicted facts — this is reported as a clean, stated null rather than
reframed away.

Deployment/practicality framing was refined per user prompts: SelfGist only
requires retaining the summary's small KV footprint (roughly 8 MB at 30B for a
terse summary, quantifiable and comparable to an image-sized attachment; ~1.2 GB
for a full 12K-token context, so ~0.7% retained), making it compatible with
request-attached "opaque compaction handle" style APIs even for
otherwise-stateless serving; ValueGraft requires the entire old cache resident
at compaction time, restricting it to server-side/live-process use rather than
post-loss recovery, and the write-up's limitations section states this
distinction explicitly. A companion note (`opaque-compaction-handle-notes.md`)
further shaped the overhead section's framing around handle lifecycle,
provider-side storage cost, and version/tokenizer-pinning constraints; a "wrong
provenance/injected KV as new trust surface" caveat was also logged.

**Handoff and repo practices.** Per user request, `STATE.md` (volatile
snapshot), `DECISIONS.md` (methodology/runtime facts and deviations), and
`AGENTS.md` (stable orientation/rules) are maintained as the handoff mechanism
and were kept synced at each phase transition; an external review of the repo
caught and the assistant fixed several staleness/inconsistency bugs (a
scoring-path mismatch between local and applied judge verdicts for
stance/ruled-out probes, stale STATE.md claims about 30B progress, missing
DECISIONS.md entries, and an incomplete per-conversation appendix in
analyze.py). Long-running jobs are launched detached with unbuffered output and
monitored, GPU-bound jobs are run serially rather than in parallel (per user
guidance, since concurrent GPU jobs only added memory risk without throughput
gain), and any piped/filtered command output is first teed to a scratch file
before tailing/grepping so the full stream stays reviewable (an adopted
practice, saved to persistent memory). Frequent small commits are preferred over
a curated history, excluding large binaries (model weights stay outside the
repo).

**Target artifact and provenance.** Following input from an external reviewing
model, the intended output was reframed from an academic paper to a HuggingFace
community blog post plus the GitHub repo, explicitly framed as a "did anyone try
this, or is it obviously not worth writing up" findability-first piece rather
than a formal claim. Provenance is to be stated plainly in the body: designed
and largely executed by Claude Fable 5, with adversarial review/reframing input
from GPT-5.5, directed and sanity-checked by the user, who has been told to
personally read the draft in full before anything goes public since it carries
claims under their name. A literature pass (Sonnet subagent, properly sourced
with verified IDs) found no exact prior match; nearest neighbors (SamKV's
blending formula for RAG chunks, "Models Take Notes at Prefill" for
position-portability physics without a compaction setting, Compressed Context
Memory as a trained raw-KV compressor without a text-summary baseline) each miss
at least one leg of the combination, supporting the working names SelfGist and
ValueGraft as collision-free.

**Current state and next steps.** The core local experimental program (original
arms, Phase 2 mitigation contrast, α-sweep at both scales) is complete and
committed, along with a first-draft blog post pending the literature section
(since filled in) and user review. Per the latest instruction, active work is
validating findings against a standard external dataset rather than only
synthetic/natural corpora: LongMemEval was selected (real multi-session chat
with answers in earlier, evictable sessions), instances were built with the
answer-bearing session in the compacted-out region, and a 4B batch (48
questions) completed and was judged (fixing a data-type crash on integer gold
answers along the way); a 30B batch was in progress, encountering escalating
per-question latency (memory-pressure related, mitigated by relaunching with
larger event-batching windows and avoiding concurrent local model work), with a
fallback plan to cap the 30B sample at ~24–30 questions if timing does not
stabilize. SWE-Gym/OpenHands-SFT-Trajectories was identified as a viable,
directly replayable coding-trace dataset for a possible follow-on realism test,
and a from-scratch cloud rental plan (guardrailed spend, step-by-step for a user
unfamiliar with GPU rental platforms, addressing SSH/API-key access) was queued
as a deliverable for the user's return, per their request to have such a plan
ready in a few hours alongside continued autonomous refinement of the
LongMemEval evaluation.
