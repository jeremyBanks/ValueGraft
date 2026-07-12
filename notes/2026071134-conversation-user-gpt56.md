_This conversation covers the collapse of v10’s schedule-invariance assumption,
forensic discovery of multiple invalid controls, and a deliberate pivot toward a
smaller, independently reviewed v12 canary before any paid semantic experiment.
The latest state is materially improved but not execution-ready: stimulus
reviews pass, while the causal runtime, persistence, release, and harvest layers
remain incomplete._

**Participants:** User and gpt-5.6-sol-xhigh.

**Established results.** Natural c10 and c02 both failed the v10
cache-equivalence gate on Qwen3-0.6B CPU bf16 eager attention. c10 shifted the
selected margin by approximately `0.0605` nats; c02 shifted it by
`0.1318359375`, with cache/logit divergence appearing at layers 1 and 4
respectively. The frozen A/B/C diagnostic classified the cause as
`QUERY_SHAPE_ROUNDING`: equal-shaped branches showed no future-token leakage,
while 23-token and 4096-token query shapes diverged. Thus schedule dependence is
a real apparatus confound, not a causal-mask defect; P and O must be treated as
distinct protocols. V10 is permanently non-authorizing, its ladder remains
paused, no semantic result was collected, and paid experiment compute remains
`$0`.

The synthetic “7/7 exact-zero” gate was invalid as general validation: it used
seven lengths of one repetitive five-token stream, constituting
pseudoreplication and failing to represent natural inputs. The original
`G_wrong` control was also invalid because short donor token pools were cycled
to fill target slots, producing incoherent repetition. A purported
message-aligned schedule was actually one large history chunked at 4096, not
turn-by-turn replay. These findings are recorded as process incidents; future
gates must audit literal bytes, independent fixture count, content diversity,
production representativeness, and the exact claim licensed by each fixture.

The corrective control design uses coherent, plant-specific minimally
counterfactual histories, identical canonical widths and P/O schedules, forced
common summary/carrier tokens, focal and non-focal scoring, and same-schedule
correct-versus-counterfactual contrasts. Four banked c02/c10 candidates passed
mechanical geometry but failed blind and target-aware content review because the
inherited conversations themselves contained clipped turns, broken factual
chains, and an introduced Cedar contradiction. They remain historical
diagnostics only. Existing 30B text banks are also unsuitable as primary
evidence: most replies hit the 320-token cap, provenance is incomplete, and
replayed text is not native live-cache state.

**Strategic pivot.** The planned twelve-case v11 confirmation was paused before
further model work. Independent red-team and statistical reviews concluded that
constructing twelve long cases before observing any exact-model signal had poor
decision value. The replacement is a six-case, tokenizer-only authored v12
canary with two pre-sealed reserve cases, permanently excluded from later
confirmation. It tests a narrower claim: whether an ordinary untrained 30B model
leaves history-specific decision state at a fixed neutral handoff boundary, and
separately whether coherent full K+V or value-only transplantation can use it.
Fresh comparisons are utility contrasts containing schedule differences; matched
correct/wrong histories and focal selectivity are the semantic controls. Replay
is explicitly not live-agent native state, and any positive canary only
authorizes a later independent study.

The v12 stimulus set is now at revision 4. All 12 anonymous histories passed
blind singleton review; all six paired causal contrasts and cross-case diversity
passed; prior scaffold, padding, stale-index, and chronology defects were
corrected additively. The current source manifest contains literal
token/event/position arrays and observed round trips. No subject-model forward
pass has occurred. The exact model/revision, role-native N replay, P schedule,
R1/R2/R3 whole-call region boundaries, generated/forced identity requirements,
oracle schedule, 64-token stopping rule, natural-control recovery formula, and
post-bf16 placebo tolerances have been frozen before outcomes.

**Current implementation state.** Pure planning, tokenization, stimulus
validation, review-packet tooling, and initial fake-model runtime tests are
working. The runtime now exercises exact N/fresh event execution,
logical-versus-packed positions, bounded row extraction, K/V replacement,
downstream recomputation, and q=1 target scoring in deterministic tests.
However, the independent execution audit found that v12 still lacks the real
subject-model runner, bounded persistent store, treatment/eligibility release
modes, differentiable positive-control executor, independent harvester, and
budget/preflight wrappers. The preregistration therefore remains nonauthorizing;
no GPU launch or paid run is permitted until these are implemented and locally
validated.

The required execution order is: finalize and bind stimuli/reviews; complete the
runtime, differentiable control, persistence, harvester, and release machinery;
run the full `$0` local 0.6B apparatus; rerun critical gates on the paid 30B
stack; only then perform the exact-model canary. The canary is exploratory, not
confirmatory and cannot support a paper efficacy claim. Expected local execution
was estimated at roughly 4–10 hours on the Mac; a paid run was previously
projected at `$8–$20` with an `$8` hard ceiling, though no paid case has started
and the current implementation has not yet reached that decision point.

Operationally, subagents are being used as bounded adversarial reviewers and
implementers, not as decision authorities. Fable’s broad-context resume failed
economically; a later narrowly scoped review cost `$2.611132` and identified two
accepted blockers. The standing rule is to provide only the minimum
decision-specific evidence bundle, use summaries for optional orientation, and
reserve Fable for high-value synthesis rather than allowing it to block
progress. Final scientific and go/no-go judgment remains with the primary agent.

The handoff priority is to finish and independently audit the v12 execution
architecture, then run only the local technical apparatus. Do not resume
v10/v11, authorize paid semantic work, reinterpret mechanical review passes as
scientific evidence, or expand the corpus until the runtime, positive control,
release boundary, and harvester produce a valid local result.

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
- `019f52a0-5302-7682-9495-2525ceccefb5`
- `019f52a0-3529-7dd2-aae0-bf4d914a4356`
- `019f52ac-b70e-7832-8000-a951b40284a5`
- `019f52ac-d2c4-7b60-b4ab-8e61b36d95a5`
- `019f52c2-44c1-70a3-bda2-1e696225bb30`
- `019f52cd-5ccd-7ce1-b35f-eb75a64f7dc6`
- `019f52cd-84c8-77c3-94af-292faaf0ec19`
- `019f52d6-eeea-7701-90c6-339182371740`
- `019f52d7-1776-7050-92cd-60d6802e4d15`
- `019f52dc-8980-7240-b39f-266378f6e1eb`
- `019f52e0-b583-74d2-8aca-e1f68e4c9a5f`
- `019f52e3-33c2-7b70-a653-a978ee58bead`
- `019f52e4-3d8f-7c00-8b7c-7d9a50d4d992`
- `019f52ea-caca-7c90-8b02-909f3985a127`
- `019f52f1-fcb3-76e3-b110-201a67a87f82`
- `019f52f2-1a9d-78c1-919f-940756696ab2`
- `019f52f2-4644-77a3-8df5-99a94ea0a379`
- `019f52fb-46e6-7bc1-a929-63dab10b0be4`
- `019f52fb-6caa-79c2-922f-59ae90a63c13`
- `019f52fb-878c-79a3-bb5e-301ae792cb55`
