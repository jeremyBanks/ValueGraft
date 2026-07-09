# ValueGraft Notes Overview

_This archive documents an interpretability/mitigation research program studying
what is lost when a language model's conversation context is compacted
(summarized) mid-session, and whether retaining or transplanting write-time
attention state across that compaction boundary can recover some of what is
lost. The project moved from a mechanism-first framing to a mitigation-first
one, built and repeatedly hardened an empirical harness across local and cloud
hardware, discovered and lost several headline results to reproduction and
precision failures, and arrived at a narrower, more defensible set of claims
than it started with — a trajectory that is itself part of the record._

## Origins and founding discipline

The project began from a specific question: does retaining write-time KV cache
state across a context-compaction boundary preserve measurable semantic
continuity that text-identical recomputation loses? An experiment brief fixed
the environment (`mlx-lm` on Apple Silicon for development, later ported to
Hugging Face/`transformers` for cloud scale), excluded architectures unsuited to
clean cache surgery (Gemma's original line, Mamba/Gated-DeltaNet hybrids), and
defined an initial arm set spanning an oracle full-context baseline,
text-summary compaction, gapped KV retention, a no-summary ablation, and
value-only transplant.

Before any experimental arm was trusted, a full L0–L4 identity-test ladder was
built and passed: cache serialization, gapped-cache surgery, and
value-transplant machinery were all validated bit-identical or near-identical
under null conditions. This ladder caught three real bugs before they could
contaminate results, and the practice of validating machinery before trusting
its output — reinforced repeatedly through later reproduction crises — became
the project's most durable methodological habit. A synthetic probe corpus
(hand-planted referent/sense/stance/evicted-fact probes across scenario
conversations, generated so that the subject model itself produced all
in-conversation replies) was built, audited for tail-leakage contamination, and
repaired to near-full cleanliness before being treated as a stable evaluation
instrument.

## The mitigation reframing

Early external review (independent model critique, then a read-only outside
session) converged on a correction that reshaped everything downstream:
"write-time KV state differs from re-encoded text" is close to self-evident and
not itself a contribution. The project's central question was redirected to
mitigation — can a small, deployable cache-state intervention reduce the damage
conversation compaction causes, in a way that survives negative controls and
looks plausibly deployable? A second, stronger version of this correction later
established that "compaction destroys context-conditioned state" is not a
finding at all but an assumed baseline against which interventions are scored —
any document phrasing it as a result needed rewriting. It was later clarified
that this reframing was a correction of the _agents'_ earlier misreading of the
owner's intent, not a change in the owner's actual goals, and should not be
narrated as an initial-belief-then-corrected arc in any write-up.

A parallel claim-hierarchy discipline followed: primary claims are mitigation
(does the intervention reduce compaction-induced errors on leakage-clean
probes), secondary claims are mechanism (write-time value states carry
context-conditioned information), and language like "meaning lives in the
values" was avoided in favor of narrower phrasing. Continuation NLL was demoted
from headline to secondary metric in favor of probe accuracy by category and
leakage class, and holdout/validation-split discipline became a hard requirement
for any tuned parameter — repeatedly vindicated when it caught overfit
"findings" (a per-head effect, later a per-slot "posslots" config) that failed
their own negative-control guards.

## Arm taxonomy and the ValueGraft family

The intervention space was eventually formalized as a two-axis family, `alpha_K`
and `alpha_V`, controlling how much fresh-vs-write-time key and value state is
used at aligned tokens: Plain Summary Compaction (α_K=α_V=0), V-only Graft,
K-only Graft, KV-Graft, and Coupled KV-Graft, with α=1 understood as an endpoint
of a continuous parameter rather than a distinct method. **ValueGraft** became
the umbrella/paper-facing name for the value-only intervention family; earlier
packed-summary arms (**H-pack**/**B-min-pack**, renamed for the paper as
**ValueGraft-Pack**/**FreshPack**) were reclassified as a confounded auxiliary
comparison — different layout, no retained tail, both K and V grafted — rather
than a clean point in this taxonomy, a distinction that mattered enormously
later. A **B-causal** control removed an ordering confound between summary
generation and conversation tail; negative controls (wrong-conversation and
shuffled-value grafts) were elevated to load-bearing status throughout, reliably
cratering performance and validating that the harness was measuring something
real.

## The empirical arc: pilots to cloud scale

Local 4B and cloud-scale 30B (bf16) pilots established a durable pattern: no
method recovered exact evicted-fact/referent recall, but a distinct **honesty
effect** emerged and strengthened with scale — write-time-encoded summary state
made models admit ignorance about evicted facts far more than text-only
compaction, which fabricated. A tuned value-graft alpha sweep
(validation→holdout) showed scale-dependent optima (mid-band α at 4B, a much
larger α — including a "steering/extrapolation" regime past α=1 — at 30B), and
layer-profiling found a real mid-depth effect band at 4B that a per-head
investigation correctly failed to replicate, demonstrating the holdout
discipline was working.

Standard-benchmark validation (LongMemEval) replicated compaction damage
universally but showed the honesty effect was frame-dependent — strong on the
project's own agentic corpus, nearly absent on LongMemEval's retrieval-style
personal QA — yielding an explicit scope statement that the honesty benefit
applies to mid-task agentic compaction, not retrieval QA. This same benchmark
run later returned a near-total null across all compacted arms on fact accuracy,
which became the trigger for the project's biggest strategic pivot (below).
Offline replay of real SWE-Gym/OpenHands coding-agent trajectories gave a
smaller but statistically clean positive: tuned value-grafting recovered a
modest fraction of the full-context-vs-compacted likelihood gap, a result that
survived every later audit and remains the project's most stable finding.

## The pivot to agent-coding evaluation

The LongMemEval null on fact retrieval, combined with mounting confidence in the
coding-trace signal, redirected effort toward end-to-end agentic evaluation: a
shim-served OpenHands/SWE-Gym-style pipeline scored objectively via test
pass/fail, run across a scaled cloud pod fleet under a champion/challenger
promotion ladder (explore → confirm → sealed holdout). This phase was also where
the project's reliability discipline was forged under stress: a session-state
leak from shared pod lanes required a pre-registered clean rerun with
contaminated results explicitly quarantined; a genuine trust rupture occurred
when "real agent" language was found to actually describe a synthetic harness
for roughly twelve hours, producing a standing rule that every results statement
must name its task source (synthetic vs. standard benchmark) inline; and a
**capability floor** was discovered on live standard coding benchmarks — the
subject model itself, not compaction, was often the binding constraint, forcing
a scaffold-configuration audit that found two bugs (disabled native
tool-calling, hardcoded greedy decoding) affecting every arm symmetrically.
Tau-bench and its `banking_knowledge` domain were scouted as a cleaner
dissociation target but ultimately retired after real integration effort showed
the domain's conversations were too short to create eviction pressure. A
synthetic "chain" agent-task grid similarly closed with no result to report: a
tuned grafting config fixed a real instability, but the underlying task set
showed no detectable compaction damage to repair — a ceiling effect, not
evidence against the mitigation.

## The sense/referent/stance finding and interpretability corroboration

A re-analysis of existing model outputs through a different lens — scoring for
**meaning recovery** rather than exact fact recall — produced what became the
project's central finding: value-grafting at 30B recovers a meaningful fraction
of lost **sense** (disambiguation) and **referent** (specific prior decision)
meaning relative to plain compaction, with a correctly-null effect on **stance**
(preference/disposition), where plain summaries already preserve the relevant
disposition. This sense/referent/stance taxonomy replaced the earlier, coarser
fact-recall framing project-wide. The pattern was corroborated by an independent
judge-free teacher-forced metric, did not replicate at 4B, and only partially
replicated on a second same-scale model (sense/stance split held, referent
effect did not).

A Jacobian-based interpretability readout tool (internally "the J-lens") was
adopted as a secondary, hypothesis-generating instrument — never sufficient
alone, always paired with behavioral confirmation. An early version of this tool
had a real methodological gap (it never actually captured the grafting
intervention itself, only write-time-vs-fresh comparisons), which was corrected
once discovered. Key-grafting was tested as a rescue mechanism for referent
recovery and found to have no effect at any tested layer, closing that line for
semantic-phrase-style targets. A self-contained, isolated interpretability
side-investigation into model behavior on a historically sensitive topic
produced language- and framing-dependent behavioral results; per owner direction
this material was kept out of science documentation and scrubbed to generic
language throughout the archive, and should stay described only generically (an
interpretability tangent) rather than by its specific subject matter.

A full narrative report was drafted, reviewed, revised with concrete transcript
examples, and published to the repository as the primary README-facing document
— before a late-day discrepancy between a new "effect-bound" measurement and the
already-published headline number was discovered and left as an open, blocking
risk pending reproduction.

## The precision and reproduction crises

The two crises that most reshaped the project's final posture both concerned
whether headline results held up under scrutiny of _how_ they were measured, not
just _what_ they measured.

First, a reproduction crisis traced an apparent collapse of the headline
recovery number to a Cauchy-unstable mean-of-ratio estimator, (E−B)/(A−B); on
robust metrics (raw E−B, %-helped, bootstrap CI) the original qualitative
dissociation held, and mean-ratio was retired project-wide as a metric standard.
This same investigation surfaced a genuinely new mechanistic finding: the
value-graft effect only reproduces when the tested model generates its own
summary at the compaction boundary, not when an externally supplied,
semantically equivalent summary is grafted in — forcing every subsequent
cross-model sweep to use self-generated summaries. A related nativeness confound
(assistant replies authored by one model contaminating a cross-architecture sign
map) was found and fixed via a matched-scaffold, model-filled corpus design. The
cross-architecture sweep that followed was reframed around attention geometry
(KV-head count, QK-norm, RoPE) rather than an initially attractive but likely
reviewer-fatal MoE-routing explanation, and surfaced an early, unresolved
counterexample to the QK-norm hypothesis.

Second, and more serious, a precision/provenance audit found that **neither of
the project's two marquee positive results had ever been scored at bf16** — both
had run only on local 4-bit MLX — and that the honesty-effect's intervention arm
(H-pack) was a coupled-KV intervention (α_K=1, α_V=1, no tail) distinct from the
paper's named value-only ValueGraft method, with a wrong-summary control
suppressing fabrication just as effectively as the real arm — indicating the
honesty effect is driven by packed layout, not grafted content specifically. An
independent re-render at n=12 further collapsed the recovery headline from +12pp
to +1.0pp (CI spanning zero), attributed mostly to render fragility under MoE
hardware nondeterminism rather than judge artifact. Once fully re-derived with
correct bootstrapping, the SWE-Gym bf16 value-only result survived as genuine
and significant (E−B = +0.0156 nats, CI [+0.005, +0.027]) — currently the
project's only confirmed bf16, correctly-armed positive result. A packed-KV
admission effect (honesty/fabrication reduction) was separately reconfirmed at
bf16 and stands as real but explicitly supporting-only, not the named method.

## Current state and handoff

The paper draft (`paper/DRAFT.md`, committed but not yet promoted to README) is
now built around an honest bounding/postmortem spine: the placebo-controlled
bf16 null as the primary honest result, the SWE-Gym bf16 positive as a small
caveated supporting effect, the packed-KV admission effect reported as real but
supporting-only, and a transcript-grounded process postmortem covering the
precision and arm-conflation failures. The owner has authorized autonomous
promotion of a fully-reviewed paper to README and push to the repository once
confident, but drew a hard line that nothing leaves the repository externally
(no blog post, no external posting) without direct owner involvement.

A future agent picking this up should first check: (1) whether the reopened
champion/per-layer-tuned graft configuration (top16-layer α=1, "posslots" α=1)
has been placebo-controlled at bf16 and 4-bit as directed — this validation
appeared to be already in progress via a concurrent agent stream at last note
(an unpushed `results/phase2_30b_bf16_verdicts.json`-style artifact) with no
results yet reflected in the notes archive; (2)
`STATE.md`/`DECISIONS.md`/`RESULTS.md` for the latest ground truth, since this
summary reflects the notes archive only through 2026-07-09; (3) whether any
unpushed commits exist from a concurrent agent stream before pushing, per the
coordination lesson learned this same day; and (4) the pod dependency-drift
issue (a proposed but unimplemented shared `pod_env.sh` and preflight check), a
plausible recurring failure source for any new cloud run. The scheduling
heartbeat that drives notes maintenance has proven unreliable and should not be
trusted to fire on time without manual verification.
