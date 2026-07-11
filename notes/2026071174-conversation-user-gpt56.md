_This conversation documents the collapse of v10 as an authorizing experiment
after natural c10 and c02 fixtures showed large schedule-dependent bf16
divergence, followed by forensic discovery of invalid controls and a cautious
redesign toward schedule-robust, independently reviewed matched histories._

**Participants:** User and gpt-5.6-sol-xhigh.

**Core result.** V10 is permanently non-authorizing; its paid run was cancelled
before launch and no semantic result was collected. Natural c10 showed a
fixed-margin shift of about `0.06055` nats, maximum K/V divergence `16.125`, and
continuation-logit change `0.84375`; c02 independently showed a `0.1318359375`
margin shift, K/V maxima `6.5/5.6875`, and final-logit maximum `0.5`. The old
ladder stopped durably after c02 (`c10`, `c02` failed; c01 had no substantive
completion). Prior mechanistic results require a schedule-noise caveat; coarse
behavioral observations are not automatically void.

The frozen c10 A/B/C origin diagnostic completed with `QUERY_SHAPE_ROUNDING`:
branches with the same 4,096-token shape but altered causally future content
were bit-identical in the first 23 rows, while the same prefix processed with
23-token versus 4,096-token query shapes diverged. This rules out the suspected
future-token leak for c10 and identifies call-shape-dependent bf16 arithmetic as
the local cause. It does not rescue v10. The general phenomenon is established
prior art; the potentially useful contribution is an assay-specific warning that
execution-path numerics can masquerade as semantic state in KV/activation
grafting.

A major process failure was identified in the synthetic gate: seven “fixtures”
were merely seven lengths of one five-token periodic stream, so the result was
pseudoreplication and could not validate natural-language schedule invariance. A
second failure was the supposed `G_wrong` control, which filled target-length
message slots by cycling short donor token IDs, producing grotesquely
repetitive, incoherent histories. A third label was false: the old
`message_block` schedule concatenated the whole history and chunked it, rather
than replaying actual message boundaries. These controls are invalid and must
not be reused. The hard empirical gate itself was correct: it caught the
failures before paid or semantic execution.

Attribution was explicitly settled: Sol/gpt-5.6-sol made and froze the cyclic
fixture design and approved its implementation; Claude Opus 4.8 had the fixture
in review scope and failed to challenge its representativeness;
Fable/claude-fable-5 did not choose the fixture and only identified its
degeneracy after seeing the literal construction. Fable’s later attempted
corrective review was stopped after spending $2.34 without producing a usable
note; Fable is no longer a dependency or gate. Future reviews should receive
only the minimum decision-specific evidence bundle, with summaries as optional
orientation and explicit instructions not to reopen unrelated settled questions.

The banked c10/c02 counterfactual candidates were mechanically exact but failed
both blind naturalness and target-aware factual review. All four remain
unauthorized; the validator now reports the decisive two-part status: geometry
`MECHANICAL_PASS`, content review `FAIL`, execution authorization `false`.
Independent audits also corrected validator provenance/reporting defects. The
banked 30B corpus is unsuitable as primary confirmatory material: 226/264
assistant replies hit the 320-token cap, raw token IDs and resolved revision
were not preserved, and the text does not constitute native live-cache state.

The preferred redesign is a fresh, externally authored paired corpus rather than
repairing c02/c10. Freeze 12 complete, diverse conversations before any model
forward pass, each with referent and overloaded-sense plants, a correct history
plus plant-specific minimally counterfactual history, coherent retained tails,
identical message boundaries and canonical token widths, and no clipping.
Require exact-tokenizer mechanical validation, blind randomized singleton
review, target-aware factual review, cross-case diversity review, and
adjudication before authorization. No summary, KV state, NLL, or outcome may be
exposed during authoring or review.

The proposed schedules and arms are:

- `P`: true turn-aligned replay derived from canonical message boundaries.
- `O`: ordinary consecutive 4,096-token chunks over identical source tokens.
- `D/F`: fixed gapped destination used for the fresh compacted baseline.
- Source arms: `C_P`, `C_O`, `W_ref,P`, `W_ref,O`, `W_sense,P`, `W_sense,O`,
  plus `F`.

Generate one greedy correct-history summary under `P`, then force identical
summary IDs through every arm. Require positive conversation-level lower bounds
for both history-vs-fresh (`GF`) and history-vs-matched-counterfactual (`GMC`)
under both schedules; retain focal selectivity, P/O interactions, component
log-probabilities, summary NLL, and competence diagnostics. N remains 12
conversations, not inflated by schedules or plants. Do not require P/O cache
equality; require deterministic repeats and interpret them as distinct
prospective conditions.

Current v11 state: expansion was paused for a first-principles re-review after
three mechanically exact but explicitly `DRAFT_UNREVIEWED` cases were committed
and pushed (`c01` 6,221 tokens, `c04` 6,688, `c05` 6,862). Additional authoring
checkpoints exist locally for c02/c03/c06; no v11 model forward pass has
occurred. A new additive schedule primitive has nine focused tests and 27
combined new/legacy tests passing. The final decision remains with the primary
agent after adversarial subagent review; subagents provide challenges, not
authority.

The smallest proposed next step is a `$0` local technical diagnostic on the
first two frozen cases, with no efficacy interpretation: validate P/O/D
construction, deterministic repeats, generated/forced summary identity, position
and lineage preservation, non-summary cache identity, persistence, and an
engineered scoring positive control. Expected local runtime is approximately
4–10 hours. Only after that would a 30B canary and, conditionally, a full run be
considered; the estimated exact-30B workload is roughly 6–14 A100 hours, with a
$25 compute hard cap and an approximate $8–$20 expected range at the observed
rate.

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
