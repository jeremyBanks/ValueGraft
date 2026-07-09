_This chunk covers a budget-driven pivot from a QK-norm ablation causal test to
a fresh-conversation reproduction of the headline finding, a serious incident in
which the block-reproduction run was lost to a timeout after ~6.3 hours of
rendering (incident #38), and the resulting architectural fix to persist
per-conversation renders permanently to disk._

**Participants:** User and claude-opus-4-8.

Participants: user and Claude (Opus 4.8), with Fable consulted repeatedly
throughout as a secondary reviewer.

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
catch a false-null (a fresh summary that trivially preserves content, giving
nothing to recover). A held-out conversation selector
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

**Idle-capacity principle established.** The user twice flagged idle GPU time
(an expm pod held for a possible later extension, and general "what do we do
with slack time") as waste to avoid; the standing practice adopted was to
proactively fill idle capacity with useful, preemptible work rather than parking
it — judged case by case, dropped immediately if higher-priority work appears.
Applied concretely: expm was set to render the second fresh-conversation half
(c25–c36) in parallel with canary's c01–c24, so the full n=24 fresh cell would
be ready without sequential waiting; separately, idle wait-time was used to run
a conversation-clustered bootstrap CI on the previously uncorrected judged +12pp
SENSE metric. Result: the judged SENSE arm held at CI [+2.2, +22.9]pp (excludes
zero), while the judged REFERENT arm did not reach significance (+9.7pp, CI
[−6.9, +26.2], n=18). Combined with the logprob metric (referent significant,
sense underpowered), this established that the referent/sense dissociation is
real but metric-dependent — each arm significant on only one of the two
instruments — which was recorded as a required honest caveat for the paper
rather than a clean "both instruments agree" result.

**Incident #38 — lost block-reproduction run.** After ~6.3 hours of rendering
all 24 conversations (fresh conversations ran longer than originals, ~16 min
each), the canary run was killed ~10 minutes into its final scoring pass by
`MODEL_TIMEOUT`, a hardcoded formula (`convs×900+1800`) that was never validated
against actual per-conversation duration for larger fresh conversations. Because
the cross-architecture harness renders entirely in memory and writes a single
result JSON only at the end, the full ~6.3-hour render was unrecoverable — no
incremental checkpoint existed. This was identified as a regression: earlier
phases of the project used per-conversation result files with an auto-pulling
watchdog (≤30 min max loss window), and this incremental-save discipline is
documented in AGENTS.md but was not carried into the newer cross-architecture
harness. Fable independently converged on the same root cause and flagged this
as the project's second resource-sizing incident (after incident #30, disk),
noting that reliability investment had gone into failure detection but not
capacity planning or checkpointing. A hasty first recovery attempt (re-launching
split runs on canary/expm) was executed sloppily and left pod state uncertain;
both pods then became ssh-degraded (hung sessions, lingering processes that
wouldn't die cleanly). Rather than fight degraded pods, both were terminated
cleanly via API (consistent with the "terminate+reprovision a degraded pod
rather than fight it" guidance), and the recovery was deliberately slowed down
to avoid repeating past incidents (#32/#33) caused by panicked mid-recovery
re-engineering.

**Architectural fix and hard rule.** The fix in progress at the end of this
chunk: build per-conversation incremental checkpointing (render
interruptible/resumable, losing at most one conversation rather than a full
run), to be CPU-validated as producing byte-identical numbers and reviewed by
Fable before being trusted, per the user's standing requirement that
GPU-affecting code changes get Fable review before being relied on. During this
discussion the user asked directly whether full conversation renders (not just
summaries) were being saved to disk, prompting recognition that generation (not
scoring) is the expensive step (~16 min/conv), so persisting the generated
render text (small, committable, unlike KV tensors) makes every future re-test —
different α values, per-head/independent-KV tuning, the champion depth-band scan
— a cheap forward pass over saved text rather than a full re-render. This was
elevated to an absolute, standing rule recorded in AGENTS.md: save every render
to disk and commit it, for all harness work going forward, framed explicitly as
preventing a repeat of incident #38 and as making each rendered model a
permanent, reusable asset. The checkpointing agent was tasked with building both
resumable per-conversation checkpointing and full-render persistence together,
pending Fable review before being trusted.

**Per-layer/champion tuning status clarified.** In response to a user question
about whether per-layer/champion-configuration tuning had been abandoned, status
was confirmed: per-layer VALUE tuning (mid-band hump around L12–L22, champion
configs at 4B/30B) is banked and complete; key-grafting was concluded closed
(keys don't recover referent, uniformly or per-layer). Per-head tuning and
independent-KV tuning exist as built-but-unrun capability and were judged
low-value to pursue further (per-head already ties manual tuning at 4B; keys
already null). The one piece judged worth keeping is a "rescue test" using the
existing `SC_CHAMPION_SCAN` capability: scan where the graftable value signal
lives on both a positive-sign model (Qwen3-30B-A3B) and a negative-sign model
(Qwen2.5 or Mistral), then graft only the positive model's identified depth-band
onto the negative model to test whether its sign flips — converting the
architecture-specificity finding from an observation into a testable mechanism
("the depth-band profile, not the model, drives the sign"), which would
materially strengthen the paper's answer to "why does it reverse." This is
gated: it only runs after the reproduction result is settled, and only
piggybacks on renders already paid for (near-zero marginal GPU cost) — it is not
to be run ahead of, or independent of, the reproduction. The user separately
expressed continued strong interest in testing a variety of additional model
architectures overnight for enrichment purposes, explicitly subordinate to
reproduction/statistical rigor as the top priority; this was recorded as:
scientific rigor (reproduction + stats) first and unconditional, model-variety
enrichment only afterward and only as budget allows, made newly affordable by
the render-saving rule since each banked render supports unlimited free
downstream re-analysis.

**Explicit process/publishing commitments recorded as binding.** In response to
user clarification: (1) all pods will be terminated before paper-writing begins,
not left running during it — data collection and writing are sequential, not
overlapping; (2) the paper will be drafted jointly by the user, Claude, and
Fable (with heavy emphasis on Fable's involvement, consistent with its
established value throughout this project), written fresh against current
requirements once results are in; (3) Claude may autonomously write the paper
and add it to the repository/README once genuinely confident, but will not
independently publish externally to the internet (e.g., broader publication
venues) without the user's direct involvement — that decision and action remain
the user's; (4) a recurring practice was adopted of periodically asking Fable an
open-ended, un-anchored question — whether the overall trajectory still makes
sense, not just execution details — at future checkpoints, as the direct
antidote to incident #37's leading-prompt bias.

**Documentation and housekeeping.** AGENTS.md was updated near its top to point
agents to `notes/`, which now contains generated daily and overall summaries
starting at `notes/README.md`, as useful background/context before strategic
decisions or on a cold pickup. STATE.md, found badly stale (still describing the
abandoned QK-norm/go-wide plan), was rewritten to reflect the actual pivot and
later trimmed from ~821 to 47 lines by removing a superseded historical tail
(git preserves the removed content). An unreferenced scratch file
(`scratch_align_equiv.py`) was removed from repo root. The user noted, without
directing immediate action, that other top-level files/docs may now be obsolete
and could be archived into `notes/` if unused; this was deliberately deferred to
a wrap-up phase (existing task #37) rather than done mid-flight, to avoid
disturbing in-progress pipelines, with the plan to archive obsolete top-level
docs and leave only live docs
(FINDINGS/STATE/DECISIONS/INCIDENTS/PREREGISTRATION/RELIABILITY/AGENTS/CLAUDE/paper)
in root once GPU work concludes.

**Handoff state at end of chunk.** Architecture-pole runs (expo/Qwen3-32B
pair-B, w6/Qwen2.5) were healthy and continuing to render, unaffected by
incident #38. The block-reproduction run (canary c01–c24) was terminated and not
yet relaunched; the deliberate checkpointing-and-render-persistence build was in
progress, gated on Fable review before being trusted, with the plan to then
provision fresh pods and re-run the block experiment interruptibly. The user
signed off for the night, and the agreed overnight arc was: land
checkpointing/render-persistence → Fable review → interruptible block re-run
(the reproduction verdict, banking its renders) → Fable un-anchored trajectory
check → if the reproduction holds, spend remaining budget rendering additional
model architectures (banking every render, running cheap champion scans) →
deliver a morning report covering the reproduction verdict first, then
architecture-specificity signatures and any additional models banked.
