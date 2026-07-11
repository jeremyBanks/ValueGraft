_This conversation documents the invalidation of v10 as an authorizing
experiment, the discovery of schedule-dependent bf16 divergence and defective
controls, and a subsequent pivot toward a smaller, decision-focused
decomposition assay before any full confirmatory corpus or paid fan-out. The
current project priority is to validate whether a coherent, history-specific
summary-state channel exists at all, while preserving strict technical and
content gates._

**Participants:** User and gpt-5.6-sol-xhigh.

**Handoff State.** V10 is permanently non-authorizing. Its paid run was
cancelled before launch, no semantic result was collected, and the old ladder is
paused after durable failures on c10 and c02; c01 has no substantive completion.
C10 showed a fixed-margin shift of approximately `0.06055` nats, maximum K/V
divergence `16.125`, and continuation-logit change `0.84375`. C02 independently
showed a `0.1318359375` margin shift, K/V maxima `6.5/5.6875`, final-logit
maximum `0.5`, and continuation-logit maximum `0.46875`. Earlier mechanistic
observations require a schedule-noise caveat, although coarse behavioral
observations are not automatically void.

The frozen c10 origin diagnostic completed with `QUERY_SHAPE_ROUNDING`. Branches
with identical 4,096-token shapes but different causally future content were
bit-identical in the first 23 rows, while identical prefixes processed with
23-token and 4,096-token query shapes diverged. This rules out the suspected
future-token leak for c10 and identifies call-shape-dependent bf16 arithmetic as
the local cause. The underlying phenomenon is established prior art; the
potentially useful contribution is an assay-specific warning that execution-path
numerics can masquerade as semantic state in KV/activation grafting. The
observed magnitude is established only for the tested Qwen3-0.6B CPU/bf16/eager
configuration, not the intended 30B A100 setup.

The methodological implication is fundamental: at finite precision, a KV cache
is determined not only by weights, tokens, positions, and dtype, but also by
call boundaries, query shape, kernel, backend, hardware, and library version. P
and O must therefore be treated as distinct explicit replay protocols rather
than assumed-equivalent executions. P is turn-aligned replay of imported
assistant text, not native live conversational state; a positive result must be
described as a schedule-conditioned summary-row channel, not evidence about
ordinary live-agent caches.

The original synthetic gate was invalid as scientific validation. Its seven
fixtures were seven lengths of one five-token periodic stream, creating
pseudoreplication and failing to represent natural-language diversity. Exact
zero on that stream is informative only for that trajectory and does not
establish general schedule invariance. The old `G_wrong` control was also
invalid: it cycled short donor token IDs into target-length slots, producing
incoherent repetitive histories. The old `message_block` schedule was
mislabelled as message-aligned; it concatenated the whole history before
chunking rather than replaying actual message boundaries. None of these controls
may authorize future work.

Attribution is settled. Sol/gpt-5.6-sol designed and froze the cyclic fixture
and approved its implementation. Claude Opus 4.8 had the literal fixture in
review scope but failed to challenge its representativeness.
Fable/claude-fable-5 did not choose the fixture and identified its degeneracy
only after inspecting the construction. Fable’s later corrective review was
stopped after $2.34 without producing a usable note and is not a current
dependency. The hard natural-case gate nevertheless worked as intended: it
stopped paid semantic execution and exposed the schedule problem before
contamination. The durable lesson is that every authorizing gate requires
raw-input inspection, independent fixture count, content-diversity review,
production-representativeness review, and an explicit statement of the claim it
can support.

The banked c10/c02 counterfactual candidates were mechanically exact but failed
blind naturalness and target-aware factual review. All four remain unauthorized;
the corrected validator reports geometry `MECHANICAL_PASS`, content review
`FAIL`, and execution authorization `false`. The validator’s own provenance and
stale-review-reporting defects were corrected additively. The candidates are
historical feasibility witnesses only. The banked 30B corpus is unsuitable as
primary confirmatory material: 226/264 assistant replies hit the 320-token cap,
raw token IDs and resolved revision were not preserved, and the text does not
constitute native live-cache state. It may be retained as model-authored surface
text after coherence review, but not as preserved native conversation state.

The replacement v11 effort was paused after three mechanically exact drafts were
committed and pushed as explicitly `DRAFT_UNREVIEWED`: c01 at 6,221 tokens, c04
at 6,688, and c05 at 6,862. Additional authoring checkpoints exist locally for
c02, c03, and c06. No v11 model forward pass has occurred. The drafts appear to
share a long planning/decision-record scaffold and were produced by the same
authoring model, so blind diversity and coherence review is required before
further expansion. Existing drafts should be preserved as engineering fixtures,
not treated as momentum toward a confirmatory study.

**Design pivot.** The latest red-team and statistical reviews, together with
independent reconstruction, conclude that completing twelve long cases before
observing the exact model is not the highest-decision-value next step. The
immediate program should be a separate discovery/decomposition assay, with its
cases permanently excluded from any later confirmatory N. The corrected
methodological or negative paper should begin in parallel and should not depend
on a future positive result.

The cleanest semantic contrasts are same-schedule, same-shape comparisons
between correct and plant-specific minimally counterfactual histories.
Correct-versus-fresh remains a practical utility contrast because the fresh arm
uses a different gapped-destination execution path; it is not by itself a clean
semantic causal contrast. Each counterfactual must change only the focal
referent or overloaded sense and all necessary downstream references, while
preserving message boundaries, canonical widths, positions, summary IDs, and
schedule calls. Every wrong-history arm must be scored on both focal plants so
generic disruption can be separated from focal movement.

The forced correct summary is a controlled mediator, not a natural
counterfactual workflow. It estimates the effect of changing source history
while holding visible summary tokens fixed, but can create history-summary
contradiction when the summary states the correct fact. Summaries therefore
require a blinded pre-outcome leakage label such as target-neutral, partial, or
explicit. Forced-summary NLL must be reported, but does not remove the
contradiction concern. A balanced history-by-summary crossover, or a
target-independent carrier assay, is needed if a strong semantic-specificity
claim is retained.

The recommended state families are:

- `R`: coherent full K+V summary-state insertion, testing whether a usable state
  channel exists.
- `V`: value-only insertion, directly testing the original mitigation idea.
- Fresh, correct-history, and plant-specific counterfactual-history arms, all
  with the same summary IDs and matched schedules.

For each family, report utility versus fresh, focal
correct-versus-counterfactual movement, and focal selectivity against the
non-focal plant. Correct-target and counterfactual-target log probabilities must
be reported separately; a larger margin alone does not establish improved
competence. Full-KV success must not automatically authorize value-only claims.
A positive full-KV result with a null value-only result would support “a
coherent channel exists but naive V-only copying fails,” not the original
intervention hypothesis.

P should be the protocol-primary schedule. O, ordinary 4,096-token chunking,
should be a prespecified sensitivity condition rather than a co-primary
requirement unless the budget and implementation make dual scheduling
inexpensive. Neither schedule is native live-agent execution. N=12, if
eventually run, supports a fixed authored benchmark claim rather than broad
population generalization; all case values, sign counts, intervals, author-pass
sensitivity, and leave-one-case-out analyses should be reported.

**Required sequence.** First, freeze further long-corpus expansion and preserve
the three drafts as unreviewed engineering fixtures. Second, complete a `$0`
local end-to-end decomposition on a short or first-two-case apparatus, including
real model execution, persistence, independent harvest, full-KV, V-only, fresh,
and coherent counterfactual arms. The local result is a technical and decision
diagnostic, not evidence about the 30B numerical floor. It must verify exact
model/template identity, schedule derivation, deterministic repeats,
generated/forced replay identity, positions and lineage, summary leakage and
headroom, finite scores, and durable artifacts.

Third, if the local apparatus is valid and shows a large interpretable channel,
run a separate exact-30B canary on one to three cases, approximately `$3–$5`,
not counted toward any later confirmatory N. Require full-history competence,
measurable compaction damage, correct-target improvement under coherent state,
counterfactual-direction movement under wrong history, focal selectivity,
acceptable summary leakage, generated/forced identity, persistence and
independent-harvest integrity, and schedule interaction smaller than the
observed channel. A null is bounded under this assay and budget; it is not a
universal null. Severe leakage, absent headroom, incoherence, invalid controls,
or schedule interaction at effect scale should terminate the mechanism program
under the current budget.

Only a strong canary should authorize a newly authored independent N=12 corpus.
That corpus must be authored as matched correct/counterfactual triplets from the
beginning, with complete non-clipped replies, diverse structures and topics,
coherent retained tails, exact canonical geometry, blind randomized naturalness
review, target-aware factual review, cross-case scaffold review, and a third
adjudicator for disagreements. No summary, state, NLL, or outcome may be visible
during authoring or review.

A live coding-agent treatment evaluation is deferred. It requires native
assistant-token generation, common-prefix cache forking, isolated environments,
and a capability-matched model that passes its own technical ladder. A small
number of tasks would be an engineering smoke test rather than credible
performance evidence. It becomes worthwhile only after a coherent-state channel
clears the discovery assay and the native-cache infrastructure exists.

The project’s intended document form is a paper-style corrected negative or
methodological report, with any later canary or v11 result added as a separate
empirical section. The durable current state is: v10 is dead as an authorizing
instrument; c10 is classified as query-shape rounding; the old ladder is paused;
the old synthetic and wrong-history controls are invalid; v11 expansion is
paused; three drafts are committed but unreviewed; no v11 forward pass or paid
semantic run has occurred; Fable is nonblocking; and the next decision is
whether the `$0` decomposition and, if justified, the separate exact-30B canary
should replace blind completion of the full N=12 corpus.

## Conversation sources

- `019f4f15-7584-7b03-9760-138202ff7c80`
- `019f51d6-0db5-7180-8b50-47f9997efd8e`
- `019f51d6-29ef-7ef0-828f-b5b12236faae`
- `019f515b-176d-70a3-9fff-adf1ad835f72`
- `019f5242-746f-7540-92b2-9d86fb7b1bf9`
- `019f5255-f803-7b23-a704-43cd57d57d88`
- `019f5254-c794-7290-bc92-24f7b2bf55ef`
- `019f5259-71a4-7c82-a898-765888e3dade`
- `019f5261-4513-7f91-8a71-25352239708b`
- `019f5267-8b36-77f0-bb1a-b24cb81c0c47`
- `019f5286-6b5e-7fd0-8352-980a83e287d5`
- `019f5287-c1cb-7ef1-8dbc-9397ce31da62`
