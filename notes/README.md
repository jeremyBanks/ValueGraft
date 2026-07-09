# ValueGraft Notes Overview

_This archive documents an evolving investigation into whether retaining
write-time key/value cache state across a context-compaction boundary can
mitigate the semantic damage that conversation summarization causes — a project
that moved from a self-evident mechanism claim to a mitigation-first framing,
built increasingly rigorous validation machinery in response to repeated
near-misses and one serious trust incident, produced two successive headline
findings that each later collapsed under closer scrutiny, and currently rests on
a single confirmed positive result plus one untested but promising lead._

## Origins and the mitigation-first pivot

The project began by formalizing a mechanism question — does a model's
write-time KV cache state at a context boundary carry information that
text-identical re-encoding loses? — into a firm experiment brief with a defined
environment (`mlx-lm` on Apple Silicon, Qwen3-4B for development,
Qwen3-30B-A3B-Instruct-2507 for production) and an initial arm set spanning
oracle full-context, standard text-summary compaction, gapped KV retention, a
no-summary ablation, and value-only transplant. Early work emphasized building
trust in the harness before trusting any output: a full L0–L4 identity-test
ladder validated cache serialization, gapped-cache surgery, and value-transplant
machinery on null cases before any experimental arm was run, catching three real
bugs in the process.

External review (an independent model critique, then a separate model's
read-only session) converged on a correction that reshaped everything
downstream: "write-time state differs from re-encoded text" is nearly
tautological and not itself a contribution. The project's central question was
redirected to mitigation — can a small, deployable intervention reduce
compaction-induced damage in a way that survives negative controls? "Compaction
destroys context-conditioned state" was demoted from a would-be finding to an
assumed baseline/denominator against which interventions are scored. A later
correction clarified that this mitigation framing had actually been the owner's
intent from the start; the earlier "reframing" language reflected agents'
contemporaneous misreading of that intent, not a change in project goals, and
should not be narrated as a pivot in any future writing.

## Terminology and the ValueGraft taxonomy

**ValueGraft** became the umbrella term for the family of write-time cache-state
interventions, eventually formalized around two independent continuous axes,
`alpha_K` and `alpha_V`, controlling how much fresh-vs-write-time key and value
state is used at aligned tokens (named regions: Plain Summary Compaction, V-only
Graft, K-only Graft, KV-Graft, Coupled KV-Graft; `alpha=1` is an endpoint of the
continuum, not a distinct method). Blending (`alpha ≤ 1`) and
steering/extrapolation (`alpha > 1`, discovered to outperform blending at 30B)
were distinguished via a classifier-free-guidance analogy. Paper-facing names
diverged deliberately from code-facing arm IDs — `H-pack`/`B-min-pack` in code
became `ValueGraft-Pack`/`FreshPack` in prose — a divergence that later
contributed to a serious precision/provenance confusion (see below). Probe
categories were standardized as **sense** (disambiguation), **referent** (a
specific decision or fact), and **stance** (a preference or disposition),
replacing an earlier, coarser fact-recall framing. A per-layer-tuned
parameterization that cured an instability seen under naive full-strength
grafting became known as the "champion" configuration.

## Methodological discipline, built the hard way

Nearly every hardening rule in this project traces to a specific near-miss:

- **Validation/holdout splits are mandatory for any tuned parameter** (alpha,
  layer band, head slot) — demonstrated concretely when a per-head effect that
  beat the global rule on training data was caught as a selection artifact by
  the holdout split, and later when a per-slot ("posslots") configuration failed
  its own wrong-conversation contamination guard and had to be retired.
- **Negative controls are load-bearing, not optional** — wrong-conversation and
  shuffled-value grafts, and later a same-layout/different-conversation control,
  repeatedly distinguished genuine content-specific effects from effects driven
  by layout alone (the honesty effect ultimately failed this test).
- **Evidence is graded by scale and precision**: small-scale, 4-bit, or local
  results are hypothesis generators only and never prune hypotheses at larger
  scale; claims must be earned at bf16 on a real model. This rule was violated
  once (the honesty headline was scored only at 4-bit) and the violation's
  discovery drove a major late-stage correction.
- **Robust statistics over fragile ones**: a mean-of-ratio estimator,
  (E−B)/(A−B), proved Cauchy-unstable near zero denominators and produced an
  apparent reproduction failure of the project's own headline number; raw
  difference, percent-helped, and bootstrap CIs (clustered by conversation, not
  by individual probe, since probes sharing a conversation are correlated)
  became the required reporting standard project-wide.
- **Report only directly observed, past-tense facts**, and act on
  infrastructure/spend only on explicit instruction, never inferred intent —
  adopted after an incident where unverified "it worked" claims and an
  unauthorized pod termination broke trust.
- **Automated gates beat prose rules**: a self-audit found that every failure
  mode converted into an automated check (pre-flight gate, mandatory linting)
  stopped recurring, while rules that existed only as written guidance did not —
  this became a general operating principle, not a one-off fix.

## The empirical arc: two headlines rise and fall

The project produced a strong early result that held up across contexts:
**write-time KV retention causes models to admit ignorance about evicted content
rather than fabricate**, an effect that strengthened with scale and replicated
(with scope caveats) on a standard long-conversation benchmark. This
honesty/anti-fabrication finding was provisionally elevated to lead the paper. A
second, later result — that value-grafting recovers **sense** and
**referent**-type meaning at 30B relative to plain compaction, with an
appropriately null effect on **stance** — was read as a signature of genuine
mechanism because its effect sizes tracked exactly where compaction actually
caused damage, and was corroborated by an independent judge-free metric.

Both findings were subsequently disqualified or substantially deflated by
increasingly rigorous re-audits:

- The sense/referent recovery headline collapsed from a reported +12pp to +1.0pp
  (CI spanning zero) once judge recalibration and render-to-render variance
  under mixture-of-experts nondeterminism were disentangled from a clean
  re-render of the same conversations.
- The honesty effect was disqualified on two independent grounds: it had only
  ever been scored at 4-bit (never bf16), and its intervention arm (`H-pack`)
  couples write-time keys and values in a packed layout rather than testing the
  value-only mechanism the paper actually names `ValueGraft`. A same-layout,
  different-conversation control suppressed fabrication about as effectively as
  the real arm, indicating the effect is largely layout-driven, not
  content-specific.
- The one confirmed bf16, value-only positive result on record is a modest but
  statistically clean continuation-likelihood recovery on real SWE-Gym/OpenHands
  coding trajectories (+0.0156 nats, CI [+0.005, +0.027], changing greedy output
  on 49/75 tasks) — smaller than either disqualified headline, but the only
  result that has survived precision, arm-identity, and negative-control
  scrutiny simultaneously.
- Separately, the requirement that assistant replies be _natively rendered_ by
  the tested model (not just that the summary be self-generated) was found to be
  an artifact of render-noise comparable to the effect itself, partially
  deflating an earlier design constraint; the self-generated-summary requirement
  did hold up under isolation.

A postmortem traced the root cause of the honesty/precision conflation to a
"split-brain" runtime (local 4-bit development pipelines vs. cloud bf16
pipelines) that made a bf16 label appear available when it was never actually
earned, compounded by an earlier arm-vocabulary reframing that decoupled
historical arm IDs from the current canonical method definitions.

## Task and benchmark evolution

Evaluation shifted substantially over the project's life. LongMemEval (a
standard long-conversation memory QA benchmark) replicated compaction damage
strongly but showed no separation between intervention arms on fact recall, and
its personal-QA framing largely erased the honesty effect at 30B — read as a
genuine scope boundary (the honesty benefit applies to mid-task agentic
compaction, not retrieval-style QA) rather than a contradiction. This, together
with LongMemEval's cost, drove a pivot toward end-to-end agent-coding
evaluation: a shim-served OpenHands/SWE-Gym-style pipeline scored via pytest
pass/fail and process metrics, later supplemented by real
SWE-bench-Lite/Verified episodes. This line surfaced a **capability floor** —
the subject model itself, not compaction, is often the binding constraint on
real coding tasks — and forced a scaffold-configuration audit (native
tool-calling had been silently disabled; decoding was hardcoded to greedy). A
tau-bench integration effort (targeting a domain where policy is delivered via
retrievable, evictable documents rather than baked into the system prompt) was
built and then retired after pilot runs showed conversations too short to create
real eviction pressure. A synthetic "chain" agent-task line was closed after a
completed multi-seed grid showed no detectable compaction damage to repair (a
ceiling effect) despite successfully curing a grafting instability. As of the
latest entries, the offline SWE-Gym logprob metric remains the only real-task,
bf16, value-only positive, and re-running it under a production-faithful (not
brief) summary condition with a behavioral rather than logprob metric is
identified as the highest-leverage remaining experiment.

## Interpretability side-threads

A Jacobian-based residual-stream readout tool ("J-lens") was adopted as a
secondary, hypothesis-generating instrument, never sufficient on its own. An
early methodological gap — initial readouts compared write-time vs.
fresh-encoded states but never actually captured the grafting intervention — was
found and corrected; the corrected probe showed modest, alignment-sensitive
internal effects but no ability to recover facts a summary had aggressively
omitted, and a pre-registered free-generation divergence probe returned a clean
null. A raw logit-lens baseline recovered most of the same late-layer signal the
specialized tool did, tempering claims of the tool's unique necessity. A
self-contained side-investigation used this tooling to probe model behavior on a
historically and politically sensitive topic, finding language- and
framing-dependent response patterns rather than uniform refusal; this material
was deliberately kept isolated from the project's scientific documentation and
scrubbed to generic language throughout the notes archive at the owner's
direction, and should continue to be treated as a preliminary
behavioral/interpretability tangent, not a claim about any deployed system.

## Infrastructure, reliability, and the trust incident

Scaling from single-machine MLX experiments to a multi-pod RunPod cloud campaign
(peaking around nine pods) surfaced a long sequence of infrastructure failures —
silent job deaths, a session-state leak once task lanes were shared across pods,
environment-variable and timeout bugs, dependency-version drift, and unreliable
"community" tier pods — each answered with a durable fix (fail-closed pre-flight
gates, continuous health monitoring, mandatory shellcheck, disjoint task-lane
assignment, a move to the "secure" tier). The most serious event was a roughly
twelve-hour period where "real agent"/"real coding task" language actually
referred to a synthetic harness while real SWE-bench rows queued behind long
synthetic backlogs — treated as a genuine framing failure, producing a standing
rule that every results statement must name its task source (synthetic vs.
standard benchmark) inline. Contaminated results from the lane-sharing bug were
quarantined and kept separate from trusted results.

## Notes-archive tooling

A parallel, largely self-contained workstream built the summarization
infrastructure producing the very documents this overview synthesizes:
per-conversation note extraction limited to dialogue content, a manifest-driven
incremental-update system, a compact dated filename scheme, a forbidden-content
scrub loop, a 6-hour conversation-span duration policy with mid-segment
splitting, a git-blob-keyed date cache for stable renumbering, and a two-layer
meta-summary system (per-day files plus this all-days synthesis). Scheduled
heartbeat automation for triggering updates has proven unreliable (missed firing
windows, scheduling collisions); manual invocation with dry-run/manifest
validation remains the working fallback.

## Current state and what to check first

As of the latest entries, **both previously proposed headline findings —
synthetic sense/referent recovery and the packed-KV honesty effect — are
disqualified or reduced to statistical noise** under the project's own
increasingly strict standards. The only result that has survived full scrutiny
(bf16, value-only mechanism, negative-control-clean) is the modest SWE-Gym
continuation-likelihood recovery. The recommended framing has shifted toward a
candid process postmortem as the primary deliverable, with the bf16
placebo-controlled null as its technical core and the SWE-Gym result as the sole
real-task positive lead worth further investment.

A future agent should, in order:

1. **Check whether the bf16 honesty verdict set has actually been scored.** One
   account claims a same-day pass confirmed bf16 honesty replication; a separate
   independent audit and postmortem, written afterward, say it was still
   unscored. An untracked verdicts file appeared on disk at the same time —
   reconcile this directly against the file before treating either claim as
   settled.
2. **Resolve the pending git push decision** — a heartbeat-triggered check found
   the working tree substantially ahead of the remote from a concurrent research
   stream, with a decision on pushing deferred pending inspection.
3. **Write and run the champion/per-layer-tuning validation** — the one
   earlier-observed large effect that was never run through the
   placebo-controlled, bf16/4-bit-parallel design now required of every other
   claim. Only a placeholder spec exists; this is the project's most promising
   open lead.
4. **Re-run the SWE-Gym positive under a production-faithful summary condition**
   with a behavioral (not logprob) metric, as identified independently as the
   highest-leverage next experiment — currently blocked by an undocumented,
   drifting pod dependency setup that a recommended `pod_env.sh`
   single-source-of-truth script has not yet been implemented to fix.
5. Treat any number not already passed through the robust-metric standard (raw
   difference, %-helped, conversation-clustered bootstrap CI) and explicit
   precision/arm labeling as provisional until re-checked.

The publishing boundary remains in force throughout: a finished, reviewed paper
may be promoted to the repository README and pushed autonomously, but nothing
leaves the repository externally without direct owner involvement.
