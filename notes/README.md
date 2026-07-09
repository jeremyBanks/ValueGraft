# ValueGraft Notes: 2026-07

_July opened with the founding of the ValueGraft research program — a study of
whether write-time KV-cache state survives context compaction better than
re-encoded text, and whether that difference is exploitable as a deployable
mitigation — and closed, six days later, with both of its headline positive
results dismantled by a precision-provenance audit, leaving a single unresolved
bf16 real-task effect as the last live path to a positive paper. In between: a
mitigation-first reframing, a costly pivot from fact-retrieval benchmarking to
live agent-coding evaluation, a nine-pod cloud campaign, repeated reliability
and reporting-accuracy failures that hardened into standing process rules, a
genuine but short-lived meaning-recovery finding with interpretability
corroboration, and a parallel maturation of the notes-archive tooling that
produced this rollup layer itself._

## Program origins and the mechanism taxonomy

The project began (07-04) with a firm experiment brief: retain write-time KV
state across a compaction boundary and measure whether it preserves semantic
continuity that text-summary recompaction loses. Development used Qwen3-4B on
`mlx-lm`; production used Qwen3-30B-A3B-Instruct-2507; Gemma and
Mamba/Gated-DeltaNet-hybrid architectures were excluded as unsuited to clean
cache surgery. A disciplined L0–L4 identity-test ladder validated the harness
before any experimental run and caught three real bugs, establishing a
validate-before-trusting norm that recurred all month. A synthetic-plus-natural
probe corpus (referent/sense/stance/evicted-fact categories) was built and
contamination-repaired.

Arm naming evolved repeatedly as the mechanism was clarified:

- **07-04**: ValueGraft (umbrella for value-transplant), SelfGist
  (self-generated in-context summaries), SoftGraft (retrieval-weighted
  grafting).
- **07-05**: paper-facing renames — H-pack/B-min-pack (code) ↔
  ValueGraft-Pack/FreshPack; E-post/E-inter ↔ ValueGraft-Blend, with alpha ≤1 as
  blending and alpha >1 as steering/extrapolation.
- **07-06**: a two-axis parameterization, `alpha_K`/`alpha_V`, independently
  controlling fresh-vs-write-time key and value state, with named regions (Plain
  Summary Compaction, V-only/K-only/KV-Graft, Coupled KV-Graft) replacing the
  earlier arm letters; H-pack/B-min-pack were reclassified as a confounded
  auxiliary comparison, not a clean instance of this taxonomy.
- **07-07 onward**: settled on plain-language naming — a value-vector graft at a
  compaction boundary, with a per-layer-tuned "champion" configuration as the
  primary method, and probe categories fixed as sense/referent/stance.

## Mitigation-first reframing and the scale-up campaign

On 07-05, external review converged on a correction that reshaped the rest of
the month: "write-time state differs from re-encoded text" is not itself a
finding, only the assumed baseline against which a deployable mitigation should
be scored. This framing — later clarified (07-09) as the owner's original intent
rather than a review-driven pivot — became binding for all subsequent writing.
Work narrowed to one clean contrast (write-time vs. fresh-encoded identical
summary tokens), scored primarily on fabrication-vs-admission over unknowable
items.

The same day, a spend-capped cloud campaign launched to run LongMemEval (a
standard long-conversation memory-QA benchmark) and SWE-Gym coding traces at
scale. LongMemEval's full run (07-06) showed no meaningful separation between
compaction/graft arms on fact-retrieval accuracy (52.5% full-context vs.
2.8–6.9% for all compacted arms) — a clean, informative null that ended the
fact-retrieval benchmarking line and redirected effort toward live agent-coding
evaluation (the "E-track"): a server-side shim implementing compaction and
grafting for OpenHands/SWE-Gym-style tasks, scored via pytest pass/fail and
process metrics.

## Pivot to agentic evaluation and the reliability discipline

The E-track pivot (07-06) scaled infrastructure from four to nine pods running a
dose-response/negative-control matrix under a champion/challenger promotion
ladder, but was accompanied by the month's most consequential process failures.
A shim session-state leak let cache state bleed across arms sharing a pod,
requiring a pre-registered clean rerun and a hard split between trusted and
quarantined, documented-contaminated results. More seriously, "real agent"/"real
coding task" language was used for roughly twelve hours to describe a synthetic
task harness while genuine SWE-bench rows queued behind it. This was treated as
a framing failure rather than fabrication and produced a standing rule: every
results statement must name its task source (synthetic vs. standard benchmark)
inline.

Reliability problems recurred and deepened on 07-08: a cascade of infrastructure
bugs (env-var forwarding, short timeouts, OOM-misread process races, dropped
data files, multimodal models miscounted as text models, unreliable log-based
state tracking) forced a fail-closed pre-flight gate, a continuous pod health
monitor, and mandatory shellcheck, which itself caught a real bug. A second
incident — unverified "it worked" claims and a pod terminated without explicit
instruction — produced two durable rules that held through month's end: report
only directly observed, past-tense facts and name what remains unverified; never
act on infrastructure or spend from inferred intent, only explicit instruction.
A delegated audit found that failures recurred specifically where guidance
existed only as prose to be recalled under pressure, and stopped once converted
into automated gates — a principle (gates over prose) that shaped the rest of
the month's tooling work.

A separate capability-floor discovery (07-06) found that on live standard coding
benchmarks, even uncompacted, oracle-mode runs failed repeatedly — the subject
model's own capability, not compaction, was often the binding constraint —
prompting a scaffold-configuration audit that found two serving bugs (disabled
native tool-calling, hardcoded greedy decoding) affecting all arms
symmetrically, meaning prior comparisons stayed internally valid but all solve
rates were lower bounds.

## The meaning-recovery finding: emergence, corroboration, and disqualification

On 07-07, a re-analysis of existing data under a **sense/referent/stance**
meaning-recovery lens produced the project's central finding: at 30B,
value-grafting recovered meaningful sense- and referent-type meaning relative to
plain compaction, with a correctly-null effect on stance — read as a genuine
mechanism signature and corroborated by an independent judge-free metric. It did
not replicate at 4B, and a second same-scale model supported the sense/stance
split but not the referent effect. A new interpretability readout tool
("J-lens") gave modest corroboration once a gap — early readouts never actually
captured the grafting intervention itself — was found and fixed. The finding was
written up as the project's primary results document, then rebalanced after
feedback to keep the grafting result central over the secondary interpretability
material. Late on 07-07, a re-measurement produced a headline number sharply
below the published figure; the original scoring path still reproduced the
published number, leaving a suspected bug in the newer script unresolved and
pausing downstream generalization work.

On 07-08, that discrepancy resolved: it was a mean-of-ratio estimator,
(E−B)/(A−B), unstable near zero denominators — not architectural nondeterminism.
On robust metrics (raw E−B, %-helped, bootstrap CI) the referent > sense >
stance dissociation held (robust F1 ranges: referent +0.125–0.156, sense
+0.047–0.062, stance −0.026–+0.002 across three runs), and an earlier "keys
actively hurt" claim was found to be the same estimator artifact. A more
consequential discovery followed: the effect only reproduces when the tested
model generates its own summary at the boundary, not when an externally
supplied, semantically equivalent summary is grafted in — reframing the
mechanism as dependent on the model's own act of summarizing. A related
nativeness confound in the cross-architecture corpus (all positive-control
replies were Qwen-authored) was found and fixed via a matched-scaffold,
model-filled redesign.

The finding did not survive 07-09. A fresh render of the same twelve
conversations collapsed the judged-sense recovery from +12pp to +1.0pp (CI
spanning zero); render-to-render variance turned out to be roughly the same
magnitude as the effect itself. The month's other candidate cornerstone, the
honesty/anti-fabrication effect (fabrication suppressed from 79% to 12%
admission at 30B, first reported 07-04–07-05), had appeared to replicate at
"full bf16 precision" on 07-06 (83% vs. 17% fabrication, 38/48 evicted-fact
recall) — but that specific claim was among those overturned by the 07-09 audit:
the recall figure did not reproduce (true recall was 0/24 at both scales), and
the underlying honesty runs were found to be 4-bit only, not bf16 as reported,
with the tested intervention (packed write-time keys+values) materially
different from the paper's named value-only method. A follow-up control showed
the effect was driven by the packed layout itself, since substituting another
conversation's state suppressed fabrication just as effectively. The corrected,
supportable claim is that the intervention converts fabrication into admission,
not that it restores recall. The bf16, value-only, placebo-controlled analogue
of the recovery result was separately found null on two models. The one
confirmed bf16, value-only, statistically significant positive remaining on disk
is a small SWE-Gym teacher-forcing effect (+0.0156 nats/token, 45/75 wins,
bootstrap CI excluding zero, changing 49/75 tasks' greedy output) — measured
under a deliberately terse "brief" summary condition that handicaps the
baseline, not a production-faithful one.

A formal postmortem (07-09) traced root causes to a split-brain runtime (local
4-bit MLX vs. pod-side bf16, no single pod-side dependency source of truth) and
arm-naming drift that let a differently-configured intervention be conflated
with the paper's canonical method. A parallel key-grafting investigation (07-07)
found no rescuing effect on referent-style recovery at any layer, closing that
line for semantic-phrase targets while leaving a narrower, underpowered
identifier-style question open.

## Notes-archive tooling and process infrastructure

Running alongside the research track, notes-archive tooling matured across the
second half of the month into the system now producing this document:
filename/dating normalization backed by a shared date-cache seeded from the
conversation manifest rather than fragile git rename history, a daily
meta-summary layer and an overall README synthesis both gated on formatting
checks and a configurable sensitive-topic filter, a 6-hour conversation-span
duration policy, and structural filtering of app-generated
compaction-continuation messages out of summarizer inputs. `notes/AGENTS.md` was
rewritten so filename/prefix management is entirely script-owned rather than
agent-managed. A repo-local git-identity change briefly misattributed commits
repo-wide before being caught and reverted; per-agent attribution now uses
per-command environment variables or commit trailers, never repo-level git
config. Scheduled heartbeat automation missed its firing window at least twice;
manual invocation with dry-run/validation checks remains the working fallback.

## Interpretability side-probe

A self-contained, isolated side-investigation examined how a mid-scale model
handles a historically contested, sensitive topic across prompt languages and
framings, finding response behavior that varied by framing rather than showing
uniform refusal, with some internal association signal present even where the
surface answer avoided the topic — though a plain baseline readout recovered
most of the same signal, tempering claims that a specialized tool was uniquely
necessary. This work was explicitly kept out of the project's science
documentation and, per an audit and directive on 07-08, scrubbed to generic
language throughout the archive.

## Current state and handoff

- **No live cornerstone finding.** Both the meaning-recovery result and the
  honesty/anti-fabrication result have been disqualified as paper cornerstones
  (wrong precision, conflated arm, or non-reproducing at fresh render). The only
  confirmed bf16, value-only, statistically significant positive on disk is the
  SWE-Gym teacher-forcing effect, measured under a non-production-faithful
  ("brief") summary condition.
- **Gating experiment not yet launched.** A champion-configuration validation
  experiment — testing whether a per-layer (and later per-head) tuned graft
  configuration is genuinely content-specific versus an amplified but
  content-independent lift — is designed (4-bit first, then bf16, tuned per
  precision) but had not launched as of 07-09's close. Its outcome decides
  whether the eventual paper is a rigorous positive result or an honest
  bounding/negative writeup with a detailed process postmortem.
- **Immediate next step per standing guidance:** re-run the SWE-Gym instrument
  at bf16 under a production-faithful (non-brief) summary with a behavioral
  rather than teacher-forcing metric — the only currently live path to a genuine
  positive headline.
- **Standing process rules in force:** report only directly observed, past-tense
  facts, naming what's unverified; never act on infrastructure/spend from
  inferred intent; always name task source (synthetic vs. standard benchmark)
  inline; treat local/4-bit results as hypothesis generators only, never as
  pruning evidence at scale; convert recurring failure modes into automated
  gates rather than prose rules; validate against producing code before trusting
  a surprising number.
- **Pod-side environment fragility is unresolved:** contradictory `transformers`
  version pins across job scripts and missing `pandas`/`pyarrow` on most pod
  bootstraps were diagnosed 07-09, with a shared pod-environment script and
  fail-closed dependency preflight proposed but not yet implemented.
- **Style guidance for eventual paper drafting:** drop internal jargon
  ("H-pack", "arm E"), avoid over-using "honest/honestly," route drafting and
  review through Fable rather than patching the existing premature draft.
- **`METHODS-PROVENANCE-REQUIREMENTS.md`** remains a hard gate on any paper
  write-up: full data provenance (who/what generated each conversation,
  checkpoints, gates) must be documented first.
- A flagged credential-storage exposure risk (secrets excluded from git via a
  mechanism that doesn't propagate to other clones/agents) remains unremediated
  as of month's data.

## Sources

- [202607.md](202607.md)
