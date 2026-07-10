# ValueGraft Notes: 2026-07

_Over six UTC days, ValueGraft grew from a founding hypothesis about KV-cache
continuity across context compaction into a multi-front experimental program —
synthetic probe corpora, real coding-agent trajectories, a cross-architecture
generalization sweep, and a borrowed interpretability instrument — before its
two headline "recovery" results collapsed under independent reproduction,
leaving one modest, robust real-task effect and an honest
bounding-and-postmortem framing as the state the month closes on._

## Founding hypotheses and experimental architecture

The project opened by asking whether write-time KV-cache state carried across a
context-compaction boundary preserves semantic continuity (referent resolution,
word sense, stance) that pure-text recomputation destroys (**H1**), and whether
any such benefit lives mainly in cached **values** rather than position-stamped
**keys** (**H2**), which would let a "fresh keys, old values" transplant
substitute for full state retention. The instrument, not any single technique,
was treated as the contribution: a probe suite of
referent/sense/stance/evicted-fact questions, since downstream task-accuracy
work on cache manipulation already existed. Work targeted `mlx-lm` locally
(Qwen3-4B for iteration, Qwen3-30B-A3B for reportable results), deliberately
excluding architectures with linear-state layers where key/value comparison is
ill-defined. Six original arms (full-context oracle, standard text-summary
compaction, gapped KV retention, a no-summary ablation, value transplant, and
later a minimal-summary retention arm) were built behind an identity-test ladder
requiring bit-identical cache serialization before any arm could be trusted. An
early contamination audit found and mostly repaired tail-leakage of planted
facts into probes, and a causal-order-matched control (B-causal) was added after
external review flagged an ordering confound between arms.

## From mechanism to mitigation, and the ValueGraft naming arc

An external review on 07-05 argued the original question — does write-time cache
state differ from re-encoded text — was close to self-evident and not itself a
contribution. The user endorsed this as a genuine correction and redirected the
project toward a mitigation question: can a small, practical intervention reduce
the behavioral damage that compaction causes? A later clarification (07-09)
fixed the record on this: the mitigation-first framing had been the owner's
intent from the start, and the intervening "mechanism as finding" period was the
agents' misreading, not a pivot — future summaries should not narrate this as a
reframing.

Naming went through several iterations before settling. Early paper-only aliases
(ValueGraft-Pack, FreshPack) were later abandoned once their underlying
comparison (H-pack vs. B-min-pack, i.e. packed write-time retention) was
reclassified as a confounded auxiliary experiment — it varies keys and values
together and drops the retained tail, so it doesn't belong in the main method
taxonomy. The mechanism itself converged on a clean two-parameter formalism:
`K_final = (1-α_K)·K_fresh + α_K·K_write_time_rerotated`,
`V_final = (1-α_V)·V_fresh + α_V·V_write_time`, with **ValueGraft** as the
umbrella family name and V-only grafting (α_K=0) as the arm actually used in
most live experiments. "Replace" was rejected as a distinct method — α=1 is just
an endpoint on a continuum, not a separate category. By month's end (07-09), the
owner overruled even this: given that no ValueGraft variant had yet been shown
to robustly work, the next paper draft was directed to drop the branded term
entirely in favor of a plain, glossary-defined phrase, strip internal arm
jargon, and scope every claim exactly to what was tested.

## Two empirical threads that diverged: honesty vs. recovery

Two distinct positive results emerged early and were tracked separately for the
rest of the month, with different fates.

The **honesty/anti-fabrication effect** — retained or grafted state causing the
model to admit ignorance of evicted facts rather than confabulate confidently —
appeared first at 4B, strengthened at 30B, replicated at full bf16 precision
essentially verbatim (83% fabrication under plain compaction vs. 17% under the
write-time-KV arm on unknowable content), and held up across a LongMemEval
standard-benchmark check at 4B (though it nearly vanished at 30B under that
benchmark's personal-QA framing, narrowing the deployment claim to agentic-style
tasks). Late-month provenance auditing (07-09) added an important correction:
this result was earned using **packed retention of write-time keys and values**,
not the value-only grafting method the project is nominally named after, and a
follow-up check showed the effect is layout-driven rather than content-specific
— a summary state from a _different_ conversation suppressed fabrication just as
well as the matching one. The honesty effect is real and robust, but it is not
evidence for ValueGraft's value-only mechanism specifically.

The **sense/referent/stance dissociation** — grafting recovers word-sense and
referent information that compaction damages, while correctly doing nothing for
stance a good summary already preserves — was established on 07-07 as the
project's intended headline: +12pp on sense, +10pp on referent, near-null +2pp
on stance at 30B, corroborated by an independent judge-free logprob metric and
surviving a robustness cut, though it did not replicate at 4B. This result did
not survive the month. A disciplined reproduction on 07-09, rendering the same
original conversations under clean current code, saw the +12pp sense result
collapse to +1.0pp with a confidence interval spanning zero; decomposing the
drop showed most of it came from render-to-render nondeterminism (MoE hardware
variance) rather than judge calibration. A subsequent precision-and-arm audit
found neither headline had ever been scored at bf16 on the actual value-only
method — the recovery result was 4-bit only, and its bf16 placebo-controlled
counterpart was null on two models. A related overclaim ("38/48 accurate"
evicted-fact recall) also failed to reproduce (true recall: 0/24); the
defensible claim is that packed retention converts fabrication into admission,
not that it restores recall. Both effects are now understood as render-fragile
at the tested sample size, not measurement artifacts per se, but not currently
trustworthy as reported.

One real-task result survived the audit: the SWE-Gym coding-trace replay effect
(+0.0156 nats teacher-forced log-probability, 95% CI [0.005, 0.027], 49/75 wins)
was reconfirmed on 07-09 as genuinely bf16 and value-only — the one remaining
positive result under the project's own precision/method bar, though still a
log-probability proxy rather than a behavioral success metric, and measured only
under a terse "brief" summary condition.

## The coding-agent task line

A parallel thread tried to demonstrate compaction damage and grafting recovery
on live agentic task completion rather than probes. An overnight synthetic
matrix (07-06) produced a genuine per-layer-tuned champion configuration passing
9/9 gated episodes, but the run was contaminated by a shim session-state leak
that forced quarantining tainted data and rebuilding a clean, pre-registered
design. A more serious framing failure surfaced the same day: despite repeated
references to "real agent" tasks, live episodes had actually been running on
synthetic tasks for roughly 12 hours, producing a standing rule that every
results statement must name its task source inline. Moving to real
SWE-bench-Lite instances exposed a capability floor — the 30B subject model
failed under full, uncompacted context on the easiest available instances, while
a Sonnet subagent solved the same instance in 52 seconds with no project context
— pushing the line toward oracle-mode retrieval, two silent
scaffold-configuration fixes (disabled native tool calling, hardcoded greedy
decoding against the model card), and a new "chained-exercise" difficulty tier.
That chain-tier closed on 07-07 with a real but narrower result: the tuned
champion configuration cured a genuine full-strength-grafting instability (4/4
on every seed vs. 0/4 for naive α=1.0), but the uncompacted baseline also scored
4/4 throughout, so the task set demonstrated no compaction damage for grafting
to visibly repair. A parallel attempt to adopt tau2-bench's `banking_knowledge`
domain as a standard benchmark with genuinely evictable policy was integrated
end-to-end but retired the same day — sessions never grew long enough to force
real eviction, and all arms scored reward 0.00 with no separation. A pipeline
audit around the same time fixed three validity bugs biasing the uncompacted
control (a wall-clock timeout, missing per-request configuration provenance, and
a token-ID filter that could silently drop scoring tokens for any SWE-bench
instance).

## Interpretability corroboration: the J-lens

A collaborator's handoff introduced a Jacobian-based residual-stream readout
(J-lens) as a secondary instrument. Early qualitative demos showed the same
visible summary token reads differently depending on write-time vs. fresh-time
processing, but user-driven scrutiny caught a serious gap mid-investigation:
none of this work had actually captured the _grafted_-intervention state, only
compared old-context vs. fresh — showing a state gap exists but nothing about
whether grafting closes it. Rebuilding the capture machinery for a genuine
four-condition probe (full context, fresh compacted, an alpha-zero sanity
control, aligned graft, shifted-alignment control) produced a more honest,
narrower claim: grafting reduces _reinterpretation error_ around facts a summary
still names, but doesn't recover facts a summary omitted outright, and argmax
token-rescue alone is not sufficient evidence of a good graft. A pre-registered
free-generation divergence probe came back clean-negative. Throughout, a
companion check against plain logit lens found the older, simpler technique
already recovers most of the same signal — the J-lens adds clarity, not unique
access. This framing was also applied to an isolated, genuinely exploratory
interpretability side-check into narrative routing on a historically sensitive
topic, which is recorded generically and was scrubbed from tracked science files
per the owner's direction.

## Cross-architecture generalization sweep

The month's most infrastructure-heavy thread (concentrated 07-08) tried to test
whether attention-geometry properties (QK-norm, KV-head count, GQA ratio)
predict the sign of the graft effect across ~16 models spanning 9 vendors. It
opened with a reproduction crisis: a live re-run failed to match saved numbers,
traced not to MoE nondeterminism (ruled out by byte-identical same-hardware
runs) but to a `transformers` version drift and a Cauchy-unstable mean-of-ratio
estimator, replaced with robust metrics (raw difference, percent-helped,
median-ratio, bootstrap CIs). A positive-control failure then produced a durable
finding: the graft effect depends on the model having generated its own summary
at compaction, not merely on equivalent content being present — self-generated
summaries became a mandatory, toggle-enforced design constraint. Fable's review
flagged an "MoE flips the sign" framing as reviewer-fatal and reoriented the
sweep around attention geometry with a same-generation dense de-confound. A
separate nativeness confound (assistant replies had been generated by one model
in-context, potentially manufacturing an artifact) was resolved with a
matched-scaffold, model-filled design where each model renders its own replies
against a shared scenario, canceling generic nativeness effects by construction.
A long tail of infrastructure failures (OOMs, missing env forwarding, undersized
timeouts, multimodal-wrapped models miscounted as plain causal LMs) drove a
reliability layer (fail-closed pre-flight gates, health monitoring, mandatory
shellcheck) and two rules now written into AGENTS.md: verify the exact model
checkpoint before other debugging, and consult Fable after one or two failed
debugging attempts rather than after hours of narrow effort. By 07-08's end,
three concurrent models were testing the QK-norm hypothesis with an ablation
control; an early non-QK-norm dense model showed a positive primary-measure
effect but a different pattern on secondary measures — an unresolved
complication. This run's outcome is not reported in the 07-09 sources, which
pivoted to the reproduction collapse instead; its status is open at month's end.

## Process and infrastructure lessons

Several operational rules were established and reinforced across the month, most
now codified in AGENTS.md or standing memory rather than left as prose: verify a
job reached real computation before trusting it; never kill long-running work on
an output-size proxy; tee long output before filtering; add-and-commit by
explicit pathspec, never `add -A`; report only directly observed facts,
separately naming what's unverified, with no predictive "fixed" claims; and
never act on inferred intent for infrastructure or spend. A delegated audit
(07-08) found that recurring failure themes correlate specifically with guidance
that exists only as prose reminders — every theme converted into an automated
gate stopped recurring, which motivated the reliability-layer investment. A
separate, largely independent effort (07-08–07-09) built an incremental
notes-summarization pipeline for this archive itself: manifest-tracked
per-source hashing, mandatory italicized-summary-paragraph openings,
deterministic participant lines, and segment-grouping rules — the same machinery
that produced the daily notes synthesized into this document. Its scheduled
heartbeat automation is unreliable (missed firing windows), so manual invocation
remains the working fallback.

## Current state and handoff

- **Robust, standing result:** the honesty/anti-fabrication effect (packed
  write-time K+V retention, not value-only grafting) replicates across scale and
  precision but is layout-driven, not content-specific — usable as a finding,
  but not as evidence for the ValueGraft value-only mechanism.
- **Not currently trustworthy:** the sense/referent/stance recovery headline —
  its 4-bit reproduction collapsed toward null (+12pp→+1.0pp, CI spans zero) and
  it was never validated at bf16 on the actual value-only method (bf16
  counterpart was null on two models). Treat as an open question, not a result,
  until re-established.
- **One live positive result:** SWE-Gym coding-trace replay, bf16, value-only,
  +0.0156 nats (CI excludes zero) — a log-probability proxy, not a behavioral
  metric, under a terse summary condition only.
- **Decisive pending experiment:** a placebo-controlled, per-quantization
  champion-validation harness (paired, conversation-clustered CI vs.
  corrupted-value controls, 4-bit then bf16) is built and queued as the one
  remaining path to a legitimate positive headline; a per-head vs. per-layer
  champion comparison and a compression-severity sweep are built alongside it
  but not yet run.
- **Coding-agent task line:** chain-tier closed with a real tuning fix but no
  demonstrated damage to recover; tau2 is retired as structurally unusable; no
  forward direction (harder cross-dependent chains vs. writing up existing
  strength) has been chosen.
- **Cross-architecture sweep:** QK-norm-vs-sign hypothesis test was mid-run with
  an unresolved complication as of 07-08; not mentioned again by 07-09, so its
  outcome is unknown pending direct follow-up.
- **Paper direction:** write the honest bounding/negative result with a candid
  methodological postmortem now (no further compute needed for that); do not
  coin a branded technique name; scope every claim to
  bf16/Qwen3-30B-A3B/log-probability-proxy/aggressive-compaction as tested;
  Fable drafts with full framing latitude, then multi-pass
  adversarial/proofreading/fact-check review, contingent on the
  champion-validation outcome.
- **Infrastructure debt:** pod dependency management still lacks a single source
  of truth (proposed `pod_env.sh` + fail-closed preflight not yet built);
  notes-heartbeat automation still unreliable.

## Sources

- [202607.md](202607.md)
