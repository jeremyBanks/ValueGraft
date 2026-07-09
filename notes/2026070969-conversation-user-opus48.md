_This chunk continues the mainline project conversation started with the
QK-norm-ablation-to-reproduction pivot and incident #38 (the lost 6.3-hour
block-reproduction render). It covers the recovery build (per-conversation
checkpointing plus permanent render persistence), followed by a shift into
planning a broad, budget-gated model-enrichment sweep: which additional model
architectures to attempt, in what order, at what confidence, and what an
additional $50 top-up (bringing the total to ~$100) would concretely buy._

**Participants:** User and claude-opus-4-8.

**Budget and initial scope decision.** The user asked for a budget check. Actual
figures: 5 pods running at ~~$7.17/hr against an account balance of
$36.11 (~5 hours runway), insufficient to let the in-flight 24-conversation runs (gate, ablation, pair-B, Qwen2.5) finish, let alone complete the full planned sweep (~$70–100
more needed). The user made top-up contingent on Fable's agreement. Fable
recommended a $45 top-up allocated to: landing the in-flight runs (~$15),
completing the QK-norm ablation plus one replication and landing pair-B/Qwen2.5
(~~$30); it recommended dropping Wave 3 (GLM/gpt-oss/Nemotron) entirely,
deferring the OLMo re-run until its floor rule was resolved on-disk (no GPU
cost), and running Yi/phi-4/Mixtral only as shallow fill on already-idle pods,
never provisioning new pods for them. The user approved this "causal-core" scope
but directed that everything Fable proposed cutting become a pure opportunistic
tail — pursued only if the causal-core work finishes and budget remains, never
planned around. Funds were topped up and the plan executed: Yi came up as w4,
Qwen2.5 (w6) had to be relaunched after failing to provision, and stale-log
artifacts on repurposed pods (old models' "DONE" markers lingering after
repurposing) repeatedly produced false alarms that were traced back to log
staleness rather than real failures — established as a recurring pattern to
check for on repurposed pods.

**QK-norm ablation attempt and failure.** The ablation (disabling QK-norm
modules and comparing to non-ablated referent) was confirmed to be correctly
configured (log-verified "ablated N QK-norm modules... forward pass now runs
WITHOUT QK-norm"), but removing all QK-norm modules broke the models'
generation: Qwen3-32B (w4) errored with all plants task-excluded (competence
floor), and Qwen3-30B-A3B (expm) produced degenerate/empty generations. This was
a pre-flagged risk, not a surprise, and confirmed H1 (that QK-norm predicts the
graft's sign) was likely unrescuable in binary form. Yi (w4) also came back
UNSUPPORTED in the harness (architecture not loadable) and was repurposed rather
than debugged, consistent with Yi being low-priority breadth per Fable's scope.
(This UNSUPPORTED verdict was later revisited — see below.)

**Salvage attempt and correction of leading-question bias.** Fable proposed
salvaging the causal test via a graded λ dose-response (λ ∈ {1.0, 0.5, 0.25,
0.0}, perturbing only the QK-norm read-out on a frozen, clean-generated corpus
rather than degrading the model itself), estimated at ~$16. The user then
questioned whether Fable's continued endorsement of ablation work reflected
genuine analysis or was an artifact of leading prompts ("how do we salvage the
ablation" presupposes salvage is worthwhile). This was recognized as valid:
prior Fable consultations had been anchored/leading. Re-asked as an open,
un-anchored question with full trajectory context, Fable reversed its prior
guidance: it judged H1 already empirically falsified (Mistral's no-QK-norm
result was positive and excluded zero, contradicting H1's prediction), the
ablation causally unrescuable (model-collapse and genuine sign-flip are
indistinguishable via λ), and recommended reporting H1 as a pre-registered null
at $0 cost. Fable identified the actual threat to the paper as the headline
effect (+0.10 referent dissociation) resting on only 12 hand-authored
conversations (c01–c12); when the corpus was doubled with fresh scenarios
(c13–c54) in earlier testing, the effect did not carry (referent ≈ +0.009), and
STATE.md's claim that native-render would fix this was asserted but never
verified on disk. Fable's recommendation: the highest-value experiment is
reproducing the headline on held-out fresh conversations (c13–c54, native
render, anchor model), which simultaneously tightens the CI, tests
corpus-overfit, and settles the native-render question. This was recorded as
Incident #37 (never bake a conclusion into a subagent prompt; always ask open,
un-anchored questions and periodically re-ask without anchoring) with a
corresponding RELIABILITY.md rule.

**Pivot execution and Fable sanity-check.** The user directed resuming the
existing full-context Fable conversation thread for execution sanity-checks (not
just decisions), which surfaced five real errors in the initial pivot plan: (1)
a fresh-only null is uninterpretable without a matched c01–c12 baseline run on
the same apparatus — mandatory, not optional; (2) the already-running canary run
(Qwen3-30B-A3B, native render, greedy/deterministic, c01–c24) was in fact
already the correct block-design experiment (baseline + fresh in one homogeneous
run), so no new run was needed; (3) splitting the render across pods would have
invalidated the competence floor unless computed centrally over pooled per-plant
traces; (4) running the full 42-conversation extension upfront was over-spending
— the correct design was 24 fresh convs first (n=12 initially via canary,
extending to c25–c36 only if the n=12 cell spans zero, a pre-committed stopping
rule); (5) per-cell headroom (A−B) and plant morphology should be reported to
catch a false-null. A held-out conversation selector
(`SC_CONV_START`/`SC_CONV_LIMIT`, selecting exactly c13–c36 with zero overlap
with the tuned corpus) and a `block_analysis.py` script (central pooled floor,
conv-clustered CIs, positive-control gate, pre-committed extend rule) were built
and validated, with the analysis script's floor computation cross-checked
against the pre-registered Mistral floor value.

During this pivot, a related over-broad rule from a prior incident was
corrected: the standing rule is not "never terminate pods" but specifically
"don't kill wanted work by inferring a change of intent" — terminating
idle/surplus/failed pods for a clear operational (budget) reason remains the
agent's judgment call. Applying this, surplus pods (w4, w5) were terminated
cleanly via API after harvesting phi-4's result, once it was confirmed the
canary run alone covered the c01–c24 block design, reducing to 4 pods (canary,
expm, expo, w6) at $5.56/hr.

**Idle-capacity principle established.** The user twice flagged idle GPU time as
waste to avoid; the standing practice adopted was to proactively fill idle
capacity with useful, preemptible work rather than parking it — judged case by
case, dropped immediately if higher-priority work appears. Applied concretely:
expm was set to render the second fresh-conversation half (c25–c36) in parallel
with canary's c01–c24; separately, idle wait-time was used to run a
conversation-clustered bootstrap CI on the previously uncorrected judged +12pp
SENSE metric. Result: the judged SENSE arm held at CI [+2.2, +22.9]pp (excludes
zero), while the judged REFERENT arm did not reach significance (+9.7pp, CI
[−6.9, +26.2], n=18). Combined with the logprob metric (referent significant,
sense underpowered), this established that the referent/sense dissociation is
real but metric-dependent — each arm significant on only one of the two
instruments — recorded as a required honest caveat for the paper.

**Incident #38 — lost block-reproduction run.** After ~6.3 hours of rendering
all 24 conversations (fresh conversations ran longer than originals, ~16 min
each), the canary run was killed ~10 minutes into its final scoring pass by
`MODEL_TIMEOUT`, a hardcoded formula (`convs×900+1800`) never validated against
actual per-conversation duration for larger fresh conversations. Because the
cross-architecture harness renders entirely in memory and writes a single result
JSON only at the end, the full ~6.3-hour render was unrecoverable. This was
identified as a regression: earlier phases of the project used per-conversation
result files with an auto-pulling watchdog (≤30 min max loss window), and this
incremental-save discipline is documented in AGENTS.md but was not carried into
the newer cross-architecture harness. Fable independently converged on the same
root cause and flagged this as the project's second resource-sizing incident
(after incident #30, disk), noting that reliability investment had gone into
failure detection but not capacity planning or checkpointing. A hasty first
recovery attempt (re-launching split runs on canary/expm) was executed sloppily
and left pod state uncertain; both pods then became ssh-degraded. Rather than
fight degraded pods, both were terminated cleanly via API, and the recovery was
deliberately slowed to avoid repeating past incidents (#32/#33) caused by
panicked mid-recovery re-engineering.

**Architectural fix and hard rule.** The fix, built deliberately (not under
recovery pressure) and gated on Fable review before being trusted per the user's
standing GPU-code-review requirement: per-conversation incremental checkpointing
(render interruptible/resumable, losing at most one conversation) combined with
full-render persistence. The user directly asked whether full conversation
renders — not just summaries — were being saved to disk; investigation confirmed
only self-gen summaries were saved, not the native-reply text. Since generation
(not scoring) is the expensive step (~16 min/conv) and the generated text is
small and committable (unlike KV tensors), persisting it makes every future
re-test — different α values, per-head/independent-KV tuning, the champion
depth-band scan, or any future idea — a cheap forward pass over saved text
rather than a full re-render. This was elevated to an absolute, standing rule in
AGENTS.md: save every render to disk and commit it, for all harness work going
forward, framed as preventing a repeat of incident #38 and making each rendered
model a permanent, reusable asset. The checkpointing agent was tasked with
building both resumable checkpointing and full-render persistence together,
pending Fable review.

**Per-layer/champion tuning status clarified.** Status confirmed: per-layer
VALUE tuning (mid-band hump around L12–L22, champion configs at 4B/30B) is
banked and complete; key-grafting is concluded closed. Per-head tuning and
independent-KV tuning exist as built-but-unrun capability, judged low-value to
pursue further. The one piece judged worth keeping is a "rescue test" using
`SC_CHAMPION_SCAN`: scan where the graftable value signal lives on both a
positive-sign model (Qwen3-30B-A3B) and a negative-sign model (Qwen2.5 or
Mistral), then graft only the positive model's identified depth-band onto the
negative model to test whether its sign flips — converting the
architecture-specificity finding from an observation into a testable mechanism
("the depth-band profile, not the model, drives the sign"). This is gated: it
only runs after the reproduction result is settled, and only piggybacks on
renders already paid for (near-zero marginal GPU cost) — not run ahead of or
independent of the reproduction. The render-persistence rule is what makes this
scan (and any future re-analysis) cheap going forward.

**Explicit process/publishing commitments recorded as binding.** (1) All pods
will be terminated before paper-writing begins — data collection and writing are
sequential; (2) the paper will be drafted jointly by the user, Claude, and
Fable, with heavy emphasis on Fable's involvement; (3) Claude may autonomously
write the paper and add it to the repository/README once genuinely confident,
but will not independently publish externally without the user's direct
involvement; (4) a recurring practice was adopted of periodically asking Fable
an open-ended, un-anchored question — whether the overall trajectory still makes
sense, not just execution details — at future checkpoints, as the direct
antidote to incident #37's leading-prompt bias.

**Documentation and housekeeping.** AGENTS.md was updated near its top to point
agents to `notes/` (generated daily and overall summaries, starting at
`notes/README.md`) as background context before strategic decisions or on a cold
pickup. STATE.md, found badly stale, was rewritten to reflect the actual pivot
and trimmed from ~821 to 47 lines by removing a superseded historical tail (git
preserves the removed content). An unreferenced scratch file
(`scratch_align_equiv.py`) was removed from repo root; the user noted other
top-level docs may now be obsolete but this archival sweep was deliberately
deferred to a wrap-up phase rather than done mid-flight. Following the
model-portfolio planning discussion (below), MODEL-QUEUE.md — previously still
describing the abandoned QK-norm queue — was fully rewritten as the enrichment
plan of record (tiered candidate list, the UNSUPPORTED-has-three-flavors
finding, transformers-5.x-deferred status, the multimodal-backbone-extraction
approach, and landscape sources), and DECISIONS.md received eight new entries
capturing the session's planning decisions (rescue-test priority, rigor-first
overnight sequencing, attempt-many-cheap-validate-before-render strategy,
easy-fruit-first ordering, the +$50 allocation, the cross-model render-transfer
exploration, and the pivot scope/policies). A verification pass confirmed all of
this session's decisions are durably committed across INCIDENTS.md, AGENTS.md,
DECISIONS.md, MODEL-QUEUE.md, and STATE.md rather than living only in chat.

**Model-enrichment portfolio planning.** With render-persistence turning each
future model into a permanent, reusable asset, the user asked for a full
accounting of which model architectures to target, their load confidence, their
diversity value, and a graceful priority order in case of a budget cutoff. An
initial prioritized list was drafted (Llama-3.1, OLMo-2-32B, Gemma-2-27B,
Mixtral-8x7B, GLM-4-32B, gpt-oss-20b, Nemotron-Super-49B), then revised after
the user asked for a proper look at the best open-weight releases across 2025
and 2026 rather than working from memory; a web search was run to ground this in
current data.

Two corrections emerged from that discussion. First, multimodal 2026-generation
flagships (Gemma 3/4, Llama 4, likely Qwen 3.6) are not a hard exclusion but a
difficulty tier: these ship as wrapper classes (e.g.,
`Gemma3ForConditionalGeneration`) bundling a standard-transformer language-model
backbone, a vision encoder, and a projector; the harness's
`AutoModelForCausalLM` loader fails only because it expects a plain causal-LM
class, not because the underlying decoder is unusual. Reaching into the wrapper
(`.language_model`) and running the graft on just the text backbone is feasible,
if fiddlier than a native text-only load — the user separately flagged that many
families also ship a genuine text-only sibling checkpoint alongside the
multimodal release (e.g., Qwen vs Qwen-VL, Mistral vs Pixtral, small text-only
Gemma sizes), which is cheaper to check first and should be tried before
attempting backbone extraction. Second, the harness's pin to transformers 4.57.1
(adopted earlier as a workaround for a 5.x `RuntimeError` on weight-conversion)
is a deliberate, deferred choice rather than a fundamental blocker, and is
likely fixable per-model later; it matters because some newest-generation
architectures may require 5.x, so unlocking it eventually gates access to that
frontier tier.

Separately, the harness's `UNSUPPORTED` verdict (previously reported for Yi
without explanation) was clarified as the harness's own catch-all with three
distinct possible causes — OOM (size), model-load failure (transformers/arch
incompatibility), or an unhandled architecture feature the graft code doesn't
yet support — of materially different severity and fixability. It was
acknowledged that Yi's specific reason was never actually read from the logs
before its pod was terminated, so the true cause is now unknown and
re-attempting it would surface the answer cheaply.

This produced a settled ordering principle, adopted as the enrichment strategy:
cheap load/smoke-test validation (minutes of pod time) precedes any real render
spend (dollars), so many candidate architectures can be attempted cheaply and
only the winners get rendered. Candidates were tiered by difficulty — Tier A
(easy, standard text arch, new vendors: meta/Llama, allenai/OLMo-2,
google/Gemma-2-text, ibm-granite, cohere/Command-R, tii/Falcon-3,
deepseek-distill-on-Qwen-or-Llama, mistralai/Mixtral); Tier B (medium, odd
architectures possibly needing a transformers-version tweak: z.ai/GLM-4,
openai/gpt-oss-20b, nvidia/Nemotron, sarvam); Tier C (hard: multimodal-only
models requiring backbone extraction, or transformers-5.x-only releases, or too
large) — with the explicit rule that harder tiers go strictly last so time isn't
burned chasing difficult architectures while easier, high-confidence vendor
diversity remains unattempted.

**Budget top-up to ~$100 and value clarification.** The user added $50, bringing
the total budget to roughly
$100. Clarifying what this buys (asked twice, answered with increasing concision): ~$15–20
for the core reproduction (gates everything — the fresh-conversation headline
verdict),
~$10 to finish the two arch-pole runs (Qwen3-32B pair-B, Qwen2.5), and ~$60–70
(~50 pod-hours, ~$6–8/model) for roughly 8–12 new model renders — the difference
between "the graft works on Qwen and a couple others" and a ~10–13 model,
~8-vendor architecture-specificity map, which is framed as the paper's genuinely
novel contribution. The user confirmed that each render also directly yields
that model's effectiveness result (referent/sense/stance sign, magnitude, CI)
and — once render-persistence lands — the champion/depth scan, the rescue test,
and any further per-layer/alpha tuning essentially for free (forward passes over
saved text, seconds not minutes), so the render is the only truly expensive step
and everything else compounds on top of it. The added funds are earmarked for
enrichment only, spent top-down through the tiered list, after the core
reproduction is confirmed solid — not ahead of it.

**Cross-model render-compatibility exploration (queued, capped).** The user
proposed, as a low-stakes end-of-list exploration, testing whether the rendered
conversation from one model can be reused with another model — e.g., feeding
model B the text model A generated and measuring compatibility. It was clarified
that this must be a text-only transfer: actual KV/value vectors cannot cross
models (differing hidden dimensions, layer counts, tokenizers), but model B can
read model A's native replies and self-gen summary as foreign-but-coherent text
and compute its own write-time values and graft over it. This was judged more
than a curiosity: it directly probes the project's existing "self-native" scope
claim — if B's graft still works on A's foreign text, the effect is about
coherent context generally; if it collapses, the effect is genuinely
self-native-specific, sharpening rather than undermining the current claim
either way. Near-free once renders are saved (reuse A's text, forward-pass B,
graft, score). Recorded as worth a handful of off-diagonal cells (not a full N×N
matrix, to bound diminishing returns), strictly after the core reproduction and
the primary enrichment sweep.

**Handoff state at end of chunk.** The checkpointing-and-render-persistence
build remains in progress and is still the gating item — it must land, then
receive Fable review confirming it changes zero numbers, before the
interruptible block-reproduction re-run is launched on fresh pods. The two
architecture-pole runs (expo/Qwen3-32B pair-B, w6/Qwen2.5) remain healthy and
continue rendering, unaffected by incident #38. Total budget is now ~$100
(original top-ups plus the additional $50), with spending order fixed as: core
reproduction first (gates everything), arch poles to completion, then — only if
the reproduction holds — the tiered enrichment sweep (cheap-validate then
render, easy tiers first, transformers-5.x and multimodal-backbone extraction
deferred to the end), then the capped rescue-test and cross-model compatibility
explorations as final, near-free additions on top of banked renders. The agreed
overnight arc remains: land checkpointing/render-persistence → Fable review →
interruptible block re-run (the reproduction verdict, banking its renders) →
Fable un-anchored trajectory check → if the reproduction holds, spend remaining
budget on the tiered model sweep (banking every render, running cheap champion
scans) → deliver a morning report covering the reproduction verdict first, then
architecture-specificity signatures and whatever additional models were banked.
