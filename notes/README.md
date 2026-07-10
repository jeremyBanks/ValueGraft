# ValueGraft Notes: 2026-07

_The month opened with the launch of ValueGraft — an investigation into whether
write-time KV-cache state carried across a context-compaction boundary preserves
semantic continuity that text-only recomputation loses — and spent four weeks
building the experimental and terminological machinery to test that idea,
producing several genuine empirical effects along the way. It closed with a
serious reckoning: independent reproduction and a precision/provenance audit
collapsed both of the project's headline "recovery" results, traced the
surviving honesty effect to the wrong quantization and the wrong intervention
arm, and forced a deliberate pivot toward an honest bounding/postmortem framing,
with one narrow real-task positive and several validation experiments still
unresolved at month's end._

## Founding hypotheses and the mitigation reframing

The project began (07-04) with two hypotheses: that retained write-time KV state
preserves measurable semantic continuity that identical-text recomputation
cannot (H1), and that this benefit lives mainly in cached values rather than
position-stamped keys (H2). A harness was built on `mlx-lm` (Qwen3-4B for
iteration, Qwen3-30B-A3B for reportable results), with an identity-test ladder
gating every downstream claim and six initial arms (A oracle, B production-style
compaction, C gapped retention, D no-summary ablation, E value transplant, plus
H/B-min self-generated-summary variants) isolating individual factors.

Early in 07-05, external review argued the founding question — whether cached
state differs from recomputed text at all — was close to self-evident and not
itself a contribution, prompting a pivot toward a mitigation-first framing: can
a practical cache-state intervention reduce the behavioral damage text-only
compaction causes? By 07-09, the owner corrected the record on this:
mitigation-first framing had been the intent from the start, and the earlier
mechanism-as-finding framing was an agent misunderstanding that review
corrected, not a genuine pivot in project goals. This distinction matters for
any future retelling of the project's history.

## Terminology: the ValueGraft family and the α_K/α_V formalism

Arm naming went through several rounds of confusion and correction before
settling on a clean two-axis formalism (07-06): **ValueGraft** as the umbrella
name for the intervention family, with `K_final` and `V_final` each a α-weighted
blend of fresh and write-time state. This cleanly separated a _layout_ axis
(packed vs. gapped vs. tail-retained) from a _state-source_ axis (fresh vs.
write-time K/V), and identified the live coding arms as **V-only Graft** (α_K=0,
α_V tuned). "Replace" was rejected as a separate method name — α=1 is just an
endpoint of the continuous parameterization.

A critical distinction, only fully clarified on 07-09, is that **H-pack is not
ValueGraft**: H-pack is a packed, coupled-KV (keys _and_ values) auxiliary arm
with no retained tail, materially different from the paper's named value-only,
aligned-position method. Two paper-facing renames (H-pack→ValueGraft-Pack,
B-min-pack→FreshPack) from 07-05 were later understood to have obscured this
difference, and the project settled on treating H-pack results as useful
supporting evidence, not part of the core method family.

## Empirical arc: effects found, then complicated

Three empirical threads dominated the month, and each went through a "found,
then walked back or narrowed" arc:

- **Honesty/fabrication effect.** First observed at 4B/30B on 07-05
  (retained/grafted-state arms admit ignorance of evicted facts where text-only
  compaction fabricates), it replicated at full bf16 precision on 07-06 (83%
  fabrication under plain compaction vs. 17% under the write-time-KV arm). But
  the 07-09 precision/provenance audit found this had actually been run only on
  the local 4-bit MLX backend at the time of the paper's writing, and that the
  specific intervention arm (H-pack) is coupled-KV, not value-only. A control
  using a _different_ conversation's summary state suppressed fabrication just
  as effectively as the real one — indicating the effect is driven by the packed
  layout/regime, not by matching content. Scoring the true bf16 data confirmed
  the packed-layout fabrication reduction still holds (decoy fabrication
  −62.5pp, evicted-fact −20.8pp, CI-significant) as a real, if narrower, finding
  — but it cannot serve as evidence for the value-only ValueGraft method
  specifically.
- **Sense/referent/stance dissociation.** Established 07-07 as the project's
  central result — grafting recovers ~+12pp on sense disambiguation and ~+10pp
  on referent recovery while correctly showing near-null (+2pp) on stance,
  tracking exactly where compaction did damage. It was corroborated by an
  independent logprob metric and survived a robustness cut, though it did not
  replicate at 4B. An independent re-render of the same conversations under
  clean current code (07-09) collapsed the sense effect to +1.0pp with a CI
  spanning zero, attributable mostly to MoE render nondeterminism rather than
  judge calibration — establishing that render-to-render noise is comparable in
  size to the claimed effect, and that neither of the project's two positive
  "meaning recovery" headlines currently reproduces.
- **Tuned per-layer/per-head "champion" configurations.** A per-layer-tuned
  config hit a 9/9 pass rate on live coding episodes (07-06) and later cured a
  genuine full-strength-grafting instability on synthetic chain tasks (07-07),
  but a competing 57-slot mask failed its wrong-conversation guard and was
  retired as a content-independent artifact. The chain-task win, however, rode
  on top of a ceiling effect: the uncompacted baseline also scored 4/4 on every
  seed, so there was no damage for the tuning fix to visibly repair. Whether the
  champion configuration's advantage is genuinely content-specific or an
  amplified version of an already-known generic placebo effect remained the
  single live open question at month's end, pending a placebo-controlled bf16
  validation.

Across the month, the one result that has consistently survived scrutiny is a
**SWE-Gym real coding-trace effect**: a small but statistically clean
teacher-forced-logprob improvement (+0.0156 nats, ~45–49 of 75 wins, CI
excluding zero), first found 07-05 and still the project's only qualifying bf16,
value-only, statistically significant positive as of 07-09 — though measured
under a deliberately terse "brief" summary condition that handicaps the
baseline, not yet tested under a production-faithful one.

## Real-task and standard-benchmark work

Moving from synthetic scenarios to real agentic tasks surfaced a capability
floor: the 30B subject model failed real SWE-bench-Lite instances outright under
full context with no compaction at all (07-06), while a Sonnet subagent solved
the same task in seconds — isolating the constraint to the subject
model/scaffold rather than task difficulty. This drove a pivot to oracle-mode
retrieval, fixes to two silent scaffold defects (disabled native tool calling,
hardcoded greedy decoding contrary to the model card), and a "chained-exercise"
difficulty tier between synthetic tasks and full SWE-bench. LongMemEval, a
standard multi-session benchmark, produced an honest null throughout — no method
recovered evicted recall at either scale, though the honesty benefit there was
found to be frame-dependent (strong at 4B, nearly vanishing at 30B on
personal-QA framing). A parallel effort to adopt tau2-bench's
`banking_knowledge` domain (07-06/07) succeeded technically but was retired as
structurally unable to force real eviction within realistic session lengths.

## Interpretability tangent and the cross-architecture sweep

A borrowed interpretability tool (the "J-lens," a Jacobian-based residual-stream
readout) was used throughout 07-07 to probe grafted states directly.
Mid-investigation, the team caught its own serious blind spot — early demos had
compared old-context vs. fresh-compacted states but never actually captured the
grafted-intervention state — and rebuilt the capture machinery to add a genuine
four-condition probe. The resulting, more honest claim: value grafting reduces
reinterpretation error around facts a summary still names, but does not recover
facts a summary aggressively omitted. A companion comparison found raw logit
lens already recovers much of the same signal, so the specialized tool's
contribution is clarity rather than unique access — a conclusion also reached,
independently, in an unrelated interpretability tangent examining
narrative-routing behavior on a historically sensitive topic (kept deliberately
generic in the archive).

A separate cross-architecture generalization effort (07-08) went through its own
reproduction crisis: an unstable mean-of-ratio estimator was retired in favor of
robust metrics; a positive-control failure established that the effect depends
on the model having generated its own summary, not just semantically equivalent
content being present; and a nativeness confound (assistant replies generated by
one model in another's evaluation) was resolved via a matched-scaffold,
per-model-native redesign. A 16-model, 9-vendor wide sweep testing whether
QK-norm predicts the sign of the grafting effect was mid-flight as of 07-08,
with a preliminary complication (a dense, non-QK-norm model showing a positive
primary effect) unresolved; the 07-09 collapse of the core recovery headline
does not appear to have directly touched this thread, leaving its status
genuinely open.

## Infrastructure, incidents, and standing process rules

The month accumulated a dense set of incidents and the rules written in
response, several now codified in AGENTS.md rather than private memory: a shim
session-state leak bled cache state between concurrently run arms; a sustained
framing failure had "real agent" episodes actually queued and unrun for ~12
hours, producing a rule that every results statement must name its task source
(synthetic vs. standard benchmark) inline; a multi-hour non-reproduction traced
to debugging against the wrong model checkpoint (thinking vs. non-thinking
variant) produced a rule to verify exact checkpoints before other debugging; and
repeated instances of "we have data for X" turning out to reference the wrong
precision or arm produced a provenance-ledger discipline requiring model, dtype,
intervention, and metric to be read from artifacts, never inferred. A recurring
meta-lesson, drawn from a delegated audit late in the month: failure themes that
exist only as prose guidance keep recurring, while every theme converted into an
automated gate (fail-closed pre-flight checks, structural impossibility of
"burned rows," provenance manifests) stopped recurring.

Three independent literature searches (07-08) converged on the same conclusion:
the project's exact composite mechanism — preserving write-time values while
recomputing keys, evaluated for reinterpretation fidelity rather than task
accuracy — appears novel, though every individual ingredient has scattered prior
art, and the clearest gap in existing KV-cache literature is evaluation
methodology itself.

A largely independent thread rebuilt the notes-archive's own tooling:
incremental per-source-stream summarization keyed by content-hash manifests, a
two-layer day/month meta-summary system, a shared sensitive-topic filter, and a
fix for git-identity misattribution. Automation heartbeats intended to rerun
this pipeline proved unreliable and required manual fallback.

## Current state / handoff

- No on-disk result currently satisfies "bf16, value-only ValueGraft,
  statistically significant positive" except the narrow SWE-Gym
  teacher-forced-logprob effect (+0.0156 nats), itself measured under a summary
  condition that handicaps the baseline.
- The champion (per-layer/per-head tuned graft) validation against a
  content-corrupted placebo, a compression-level sweep, and a
  `SC_CHAMPION_CONFIG` SWE-Gym arm were all launched but unresolved at month's
  end (`results/champion_validate/`, `results/swegym_30b_bf16_prod/`,
  `phase2_30b_bf16_verdicts.json` remain untracked in-progress artifacts) —
  their outcome is the one variable that could still move the paper from a
  bounding/negative result toward a modest real positive.
- Paper direction, per Fable's pivot decision (07-09): lead with the
  bf16-confirmed honesty/anti-fabrication effect (framed as supporting evidence,
  not the named method), elevate render-fragility itself into a co-headline
  methodological contribution, and do not attempt to rescue the "ValueGraft
  recovers meaning" claim. Naming discipline going forward: coin no branded term
  unless the eventual result is genuinely reusable.
- The cross-architecture QK-norm sweep (16 models) was still mid-flight as of
  07-08 with an unresolved complication; its status was not updated on 07-09 and
  should be checked directly before any further reliance on it.
- Publication remains gated on direct owner involvement and scoped to a private
  repository only, with no arXiv/Zenodo submission planned.
- Notes-archive tooling is functional but its scheduled automation is
  unreliable; manual invocation with dry-run validation is the working practice.

## Sources

- [202607.md](202607.md)
