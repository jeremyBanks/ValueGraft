_This conversation documents the collapse of v10 as an authorizing experiment
after natural c10 and c02 fixtures exposed large schedule-dependent bf16
divergence, followed by discovery of invalid controls and a redesign toward
schedule-robust, independently reviewed matched histories. The latest review
recommends pausing the costly N=12 build and testing the mechanism with a
smaller decomposition canary first._

**Participants:** User and gpt-5.6-sol-xhigh.

**Core result.** V10 is permanently non-authorizing; its paid run was cancelled
before launch and no semantic result was collected. Natural c10 showed a
fixed-margin shift of about `0.06055` nats, maximum K/V divergence `16.125`, and
continuation-logit change `0.84375`; c02 independently showed a `0.1318359375`
margin shift, K/V maxima `6.5/5.6875`, final-logit maximum `0.5`, and
continuation-logit maximum `0.46875`. The old ladder stopped durably after c02
(`c10`, `c02` failed; c01 had no substantive completion). Earlier mechanistic
results require a schedule-noise caveat; coarse behavioral observations are not
automatically void.

The frozen c10 origin diagnostic completed with `QUERY_SHAPE_ROUNDING`. Branches
with the same 4,096-token shape but altered causally future content were
bit-identical in the first 23 rows, while the same prefix processed with
23-token versus 4,096-token query shapes diverged. This rules out the suspected
future-token leak for c10 and identifies call-shape-dependent bf16 arithmetic as
the local cause. The phenomenon is established prior art; the potentially useful
contribution is an assay-specific warning that execution-path numerics can
masquerade as semantic state in KV/activation grafting. It does not rescue v10,
and the observed magnitude is established only for the tested Qwen3-0.6B
CPU/bf16/eager configuration, not the intended 30B A100 setup.

The synthetic gate was invalid as scientific validation: seven “fixtures” were
seven lengths of one five-token periodic stream, creating pseudoreplication and
failing to represent natural-language diversity. The supposed `G_wrong` control
also cycled short donor token IDs into target-length slots, producing incoherent
repetitive histories; the old `message_block` schedule likewise concatenated the
whole history before chunking rather than replaying actual message boundaries.
These controls must not be reused. The hard empirical gate itself was correct
and prevented paid or semantic execution. Attribution is settled:
Sol/gpt-5.6-sol designed and froze the cyclic fixture and approved its
implementation; Claude Opus 4.8 had the fixture in review scope and failed to
challenge its representativeness; Fable/claude-fable-5 did not choose it and
identified the degeneracy only after inspecting the literal construction.
Fable’s later corrective review was stopped after $2.34 without producing a
usable note and is not a current dependency.

The banked c10/c02 counterfactual candidates were mechanically exact but failed
blind naturalness and target-aware factual review. All four remain unauthorized;
the corrected validator reports geometry `MECHANICAL_PASS`, content review
`FAIL`, and execution authorization `false`. Validator provenance and
review-reporting defects were independently corrected. The banked 30B corpus is
unsuitable as primary confirmatory material: 226/264 assistant replies hit the
320-token cap, raw token IDs and resolved revision were not preserved, and the
text does not constitute native live-cache state.

The prior redesign proposed a fresh externally authored paired corpus of 12
diverse conversations, each with referent and overloaded-sense plants, coherent
correct and plant-specific minimally counterfactual histories, identical message
boundaries and canonical widths, and blind plus target-aware review before any
model outcome was exposed. Its schedules were `P` (true turn-aligned replay),
`O` (ordinary 4,096-token chunks), and `D/F` (fixed gapped destination), with
forced identical summaries across arms. The intended estimands were
history-versus-fresh (`GF`) and history-versus-matched-counterfactual (`GMC`)
under both schedules, with focal selectivity, component log-probabilities,
summary NLL, and competence diagnostics retained. P/O cache equality is not
required; deterministic repeats and explicit schedule effects are required.

That N=12 plan is now paused for first-principles review. Three mechanically
exact v11 drafts were committed and pushed as explicitly `DRAFT_UNREVIEWED`
(`c01` 6,221 tokens, `c04` 6,688, `c05` 6,862); additional authoring checkpoints
exist locally for c02/c03/c06; no v11 model forward pass has occurred. The
drafts appear to share a long planning/decision-record scaffold and were
produced by the same authoring model, so blind diversity review is required
before further expansion.

The latest red-team assessment concludes that a clean N=12 confirmation is not
currently the highest-decision-value next step. The recommended sequence is:

- Freeze further long-corpus expansion and retain existing drafts as engineering
  fixtures.
- Complete a `$0` local end-to-end decomposition on a short or first-two-case
  apparatus, including real model execution, persistence, harvest, full-KV,
  V-only, fresh, and counterfactual arms.
- If technically valid, run a separate 1–3-case exact-30B canary, approximately
  `$3–$5`, not counted toward any later confirmatory N.
- Build the independent N=12 corpus only if the canary shows a large,
  interpretable channel.
- Begin the corrected negative/methodological paper in parallel rather than
  making it depend on a future positive result.
- Defer live coding-agent treatment evaluation until a coherent-state channel
  and the required native-cache infrastructure exist; a few tasks would
  currently be only an engineering smoke test.

The canary must address several previously under-specified issues. Copying only
generated summary content rows may miss information stored in a natural closing
delimiter or downstream aggregator, so content-only,
content-plus-closing-boundary, and a small fixed-anchor condition should be
considered prospectively. The forced correct summary under a counterfactual
history is an off-support mediator and can measure history-summary contradiction
rather than semantic memory. A balanced history-by-summary crossover, or a
target-independent carrier assay, is needed if the stronger specificity claim is
retained. Summary leakage and compacted-state headroom must be audited before
interpreting any improvement. Correct-target and counterfactual-target log
probabilities must be reported separately; a larger margin alone does not
establish improved competence. Full coherent K+V, V-only, and optionally K-only
interventions must be distinguished, since a positive K+V result would not
validate the original naive value-copying intervention.

The canary should require full-history competence, measurable compaction damage,
a correct-target increase under coherent state, counterfactual-direction
movement under the wrong history, focal selectivity, generated/forced replay
identity, persistence and independent-harvest integrity, and schedule
interaction smaller than the observed channel. A null at this stage would be a
bounded decision result, not a universal null. A positive canary would justify a
new independent confirmatory corpus; a V-only null would support “channel
exists, naive V-only intervention fails”; absent headroom, severe leakage,
incoherence, or schedule interaction at effect scale, the mechanism program
should stop under the current budget.

The proposed P schedule remains explicitly replay, not live conversational
state: it prefills historical assistant messages rather than reproducing
token-by-token native generation. Any positive result must therefore be
described as a history-conditioned summary-row channel under an explicit replay
schedule, not as proof that an ordinary live agent cache carries the same state.
Native-cache evaluation is a separate future study requiring common-prefix
forking and a different control strategy.

The immediate handoff state is therefore: v10 remains permanently
non-authorizing; the old ladder is paused; the c10 diagnostic is complete with
`QUERY_SHAPE_ROUNDING`; v11 expansion is paused; three drafts are committed but
unreviewed; no v11 forward pass or paid semantic run has occurred; Fable is not
blocking progress; and the next required decision is whether the smaller local
decomposition and exact-30B canary should replace blind completion of the full
N=12 corpus.

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
