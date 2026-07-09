# ValueGraft Notes Overview

_A research program investigating whether preserving write-time KV-cache state
across conversation-compaction boundaries can measurably improve semantic
continuity — and whether such preservation can be turned into a deployable
mitigation — moved from a mechanism-first pilot through a hard reframing toward
mitigation, into multi-scale empirical validation on standard benchmarks and
real coding-agent traces, into an interpretability corroboration side-track, and
finally into a cross-architecture generalization sweep whose infrastructure and
methodology are still being hardened._

## Origins and the core method

The project began by asking whether retaining a language model's write-time
KV-cache state across a context-compaction boundary preserves information that
text-only recomputation loses. The chosen mechanism decouples keys and values:
at the compaction boundary, a model's summary can be re-encoded fresh (standard
practice) or its original write-time value tensors can be grafted onto
freshly-computed keys, at a tunable blend strength. This intervention family was
eventually named **ValueGraft**, parameterized by two continuous knobs,
`alpha_K` and `alpha_V` (0 = fresh/plain compaction, 1 = full grafting, >1 =
"steering"/extrapolation past the original values). Early ad hoc arm letters
(A–H, E-post/E-inter, H-pack/B-min-pack) were retired in favor of this two-axis
scheme for reporting, though original code-level identifiers were preserved for
provenance and must never be redescribed as clean instances of the K/V taxonomy
— they confound layout and tail-retention with state source.

Development used `mlx-lm` on Apple Silicon (Qwen3-4B for iteration,
Qwen3-30B-A3B-Instruct-2507 for production), later ported to a validated
HF/transformers stack for cloud scale-out. Gemma and Gated-DeltaNet/Mamba-hybrid
architectures were excluded early as unsuitable for clean cache surgery; MLA
architectures were later found structurally blocked for the same reason. A
disciplined identity-test ladder (L0–L4, later extended) was built before any
experimental arm ran, and caught several real bugs (kernel mismatches,
chat-template instability, alignment bugs) before they could contaminate results
— this practice of validating machinery before trusting output recurred
throughout the project's history and became a standing norm.

## The mitigation reframing

Early results treated "write-time state differs from re-encoded text" as if it
were itself a finding. External review (independent model critique, then a
read-only session) converged on the correction that this claim is close to
self-evident and not a contribution by itself. The project was redirected to a
**mitigation-first framing**: the central question became whether a small,
deployable cache-state intervention reduces the damage that conversation
compaction causes, evaluated against negative controls, not whether write-time
and recomputed state merely differ. A stronger version of this correction
followed shortly after: "compaction destroys context-conditioned state" is not a
result at all, only the assumed baseline/denominator against which interventions
are scored — no document may report it as a finding. This claim hierarchy
(primary: mitigation; secondary: mechanism; boundary: costs/limits) has governed
all analysis and writing since.

Methodological hygiene tightened correspondingly: probe accuracy (by category
and by a six-class leakage taxonomy) was promoted to the headline metric over
continuation NLL, which proved noisy; negative controls (wrong-conversation and
shuffled-value grafts) became load-bearing rather than optional; a **B-causal**
control was introduced to remove a summary-ordering confound between arms; and
any tuned parameter (alpha, layer band, head slot) is required to pass a strict
validation→holdout split — a discipline that concretely caught and voided an
apparent per-head effect that turned out to be a selection artifact.

## Empirical arc

Results accumulated in layers, each qualifying the last:

- **Mechanism-level signal** (4B/30B pilots): referent- and sense-disambiguation
  probes showed retained-state arms outperforming plain text summarization; a
  light value-graft reliably improved continuation likelihood; transplanted
  values carried measurable word-sense margin.
- **A robust, scale-strengthening honesty/anti-fabrication effect**:
  write-time-encoded summary state makes models admit ignorance about evicted
  facts far more than text-only compaction, which instead fabricates. This held
  from 4-bit pilots through full-precision replication and strengthened with
  model scale, but is scope-bound — it appears strongly in mid-task agentic
  compaction and nearly vanishes on retrieval-style personal-QA benchmarks
  (LongMemEval), where a larger model's own refusal calibration already covers
  the gap. Packed-summary intervention buys honesty specifically, not memory:
  referent/sense recall is not improved by packing.
- **Scale-dependent tuning optimum**: the best blend strength shifts materially
  between 4B (mid-range alpha) and 30B (a much larger alpha, including the
  steering regime beyond 1.0), with no layer-gating needed at the larger scale.
- **Real coding-agent evidence**: an offline SWE-Gym next-action proxy over real
  trajectories showed a modest but statistically clean recovery (~10% of
  measured compaction damage). Live agentic evaluation on real SWE-bench tasks
  later hit a hard capability floor for the subject model (zero passes across
  all attempts, confirmed against a stronger model solving the same instance in
  under a minute), which triggered a pivot to a self-authored chained-exercise
  capability tier and, separately, scouting of tau2-bench as a
  standard-benchmark confirm phase. The chain-tier grid ultimately showed the
  tuned grafting configuration curing full-strength blend collapse, but the
  compacted baseline itself proved too compaction-robust on that task family to
  demonstrate a repair effect — an honestly-recorded ceiling problem, not a
  success. Tau2-bench's `banking_knowledge` domain was formally retired as
  non-viable: sessions were structurally too short to force genuine
  early-content eviction.
- **A later, more careful re-analysis (F1)** established that value-grafting
  recovers meaning specifically on sense- and referent-disambiguation probes
  (~+10–12pp over plain compaction) with a near-null effect on stance (where a
  good summary already preserves disposition), tracking where compaction
  actually caused damage. This was corroborated by an independent judge-free
  logprob metric but did not replicate at 4B scale — recorded as a genuine scope
  bound.
- **A late estimator bug** was found and fixed: an unstable mean-of-ratio metric
  had distorted several F1-era conclusions, including an apparent "keys actively
  hurt" finding that turned out to be an artifact — keys are neutral, value
  remains the operative axis. Robust metrics (raw E−B, %-helped, median-ratio)
  are now the locked standard for all analysis.

Throughout, negative controls (shuffled/wrong-conversation grafts, alpha-zero
sanity checks) consistently collapsed performance as expected, and
referent-recall recovery remained an honest, persistent null across every method
and scale tested.

## Interpretability corroboration (J-lens)

A side-investigation applied a Jacobian-lens technique (J-lens, a refinement of
logit-lens) to probe internal representations at the compaction boundary. Early
qualitative demos were compelling but a methodological failure was discovered
mid-track: the first shareable probe never actually captured the
grafted-intervention state, only two-condition (write-time vs. fresh)
comparisons, because it silently failed to snapshot grafted cache for a
non-standard architecture. After correction to a proper four-condition probe
(full / fresh / alpha-zero sanity / grafted), effects were found real but
modest, and a pre-registered free-generation divergence probe returned a clean
negative (no discrete internal "fork" toward the correct concept in 43 cases).
The project's settled position is that J-lens is a hypothesis-generation and
corroboration tool, not primary evidence — two-condition comparisons must never
be presented as intervention evidence, and the paper's central claim rests on
the behavioral dissociation, not the lens. A later comparison found that raw
logit-lens already recovers most of the same late-layer signal, so the
specialized lens is not uniquely necessary, only cleaner.

A structurally separate, self-contained interpretability tangent examined model
behavior around a sensitive historical topic (language- and framing-dependent
responses and internal lens readouts). It is documented as a
behavioral/interpretability audit, not a policy-circumvention exercise, and is
unrelated to the ValueGraft experimental line; per standing notes-hygiene policy
it is described generically here and in project docs rather than by topic.

## Cross-architecture generalization

The most recent major thread tests whether the effect generalizes beyond Qwen3
architectures. Its execution surfaced a sequence of real bugs, each initially
misdiagnosed: a wrong-model-checkpoint error, a nativeness confound (the
original 12-conversation corpus used Qwen-authored in-context replies for all
models, conflating "effect" with "reply nativeness" — fixed by requiring each
model to generate its own native replies and summaries from shared scaffolding),
and a long cascade of pod/launcher infrastructure defects (path bugs, timeout
mismatches, dependency drift, monitoring blind spots). A positive-control
failure produced a genuine mechanistic finding along the way: the graft effect
depends on the model having generated its own summary at the compaction
boundary, not on a semantically equivalent externally-supplied one. The sweep's
central hypothesis was reframed from an overclaim ("MoE reverses the effect,"
which conflates feed-forward and attention mechanisms) to a defensible,
pre-registered one: attention geometry — QK-norm presence, GQA ratio, head
dimension — predicts the sign of the effect (H1: QK-norm predicts a positive
sign).

The repeated infrastructure failures prompted a durable process response:
reporting must state only observed facts in past tense and name what remains
unverified (no predictive or "it works" claims), and no action affecting
infrastructure or spend may be taken without explicit instruction. An audit of
prior incidents concluded that only failures converted into automated gates
(pre-flight checks, health monitors, linting) actually stopped recurring —
written rules alone did not — and this is now a governing principle for future
process fixes.

## Provenance, novelty, and documentation hygiene

Three independent literature searches converged on the same conclusion: the
specific composite approach — write-time value retention with fresh keys,
evaluated via referent/sense/stance semantic-continuity metrics across a
compaction boundary — is not directly anticipated in the literature through
mid-2026, though every individual ingredient (key/value asymmetry, cross-context
KV reuse, cache splicing, summary retention) has prior art, and hosted-provider
compaction APIs (OpenAI, Anthropic) already ship adjacent opaque-compaction
primitives without public confirmation of any similar value-tensor mechanism.
This supports the paper's novelty claim with appropriate caveats.

A separate provenance-integrity incident was found and corrected without
rewriting git history: several hours of commits attributed to one model had in
fact been authored by a different model after an automatic quota fallback; the
correct boundary was recovered from transcripts and documented, and
commit-identity trailers are now required to update immediately on any model
handoff. A `METHODS-PROVENANCE-REQUIREMENTS.md` document now blocks final paper
write-up until full experimental data provenance is documented, a direct
consequence of the nativeness-confound discovery. The `notes/` archive itself
was standardized into a single dated-filename convention with a reusable
normalization script, and a reusable conversation-transcript summarization
pipeline (the one producing this document's daily inputs) was built as a
first-class component.

## Current state and what to check first

As of the most recent notes (2026-07-08), the project has: a locked-in
robust-metric standard for all F1-era analysis; a confirmed mechanistic finding
that grafting requires self-generated (not externally supplied) summaries; a
pre-registered attention-geometry hypothesis (H1) undergoing active
cross-architecture ablation, with one preliminary complication (a positive
effect observed in a non-QK-norm dense model) not yet resolved; a hardened
cloud/pod reliability layer (pre-flight gates, health monitoring, tier policy);
and a real-task evaluation line that has moved off SWE-bench (capability floor)
and off tau2-bench (structurally unviable) toward chain-tier tasks, whose
repair-effect signal remains inconclusive due to a compaction-robustness
ceiling.

A future agent should first read `notes/AGENTS.md`, then `STATE.md`,
`DECISIONS.md`, and `FINDINGS.md` for the authoritative current status, and
treat this document only as historical orientation. Specific items likely still
open: the QK-norm ablation result and its implications for H1; reconciliation of
the uncommitted `cross_arch_probe.py` changes and `results/cross_arch_done/`
directory against the latest sweep state; the unresolved credential-storage
exposure risk flagged by the incident audit; and the three-way decision on the
agent-task line (harder chained tasks, an alternative standard benchmark, or
writing up existing chain-tier/coding-trace results as-is).
