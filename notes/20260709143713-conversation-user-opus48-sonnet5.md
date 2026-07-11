_This conversation overturned a proposed honesty-led ValueGraft paper after
provenance audits showed the behavioral headlines were 4-bit and/or the wrong
intervention, then identified a properly controlled bf16 champion-configuration
experiment as the decisive remaining test. The intended deliverable is now a
coherent academic paper combining the validated result with a precise,
non-triumphal postmortem, followed by extensive Fable-led review and
in-repository promotion only after confidence is earned._

**Participants:** User, claude-opus-4-8, and claude-sonnet-5.

**Scientific correction.** The original recovery and honesty headlines were not
valid ValueGraft cornerstone results: scored behavioral data used local 4-bit
MLX; H-pack was packed coupled-KV (write-time keys and values), not the named
aligned value-only ValueGraft; and its fabrication reduction was largely
layout-driven and content-agnostic. The placebo-controlled bf16 value-only
`effect_bound` test was null-to-harmful across two models, while
cross-architecture recovery was negative or null. A real but small bf16
value-only SWE-Gym signal remains (+0.0156 nats, CI [+0.005,+0.027], changed
greedy output on 49/75 tasks), but it used a brief/non-production summary and is
not sufficient alone. Existing bf16 H-pack answers were subsequently scored: the
packed-KV admission effect replicated (+62.5pp decoy, +20.8pp evicted, both
significant), but remains supporting mechanism evidence rather than ValueGraft
evidence.

**Reopened experiment.** Per-layer/champion tuning was not covered by the
placebo-controlled null: `effect_bound` used scalar α=0.75, whereas tuned bf16
configurations showed larger raw gains, especially position-tuned α=1
(`posslots`, +0.054; 6/6 wins), without placebo controls. A new decisive
validation therefore uses the champion per-layer/per-position value-only
configuration versus a placebo using identical layers, α values, and positions
but corrupted source values. The criterion is paired, conversation-clustered
`E − placebo`: CI lower bound above zero establishes content-specific champion
recovery; a CI spanning zero leaves the null intact. Tuning is derived on
c01–c06 plus n01–n04 and validated on disjoint held-out c07–c24 (18 clusters,
roughly 200 plants), so the main test is out-of-sample; the paired placebo also
cancels generic configuration overfitting.

**Implementation and execution.** Fable/subagent work committed the champion
harness and jobs in `5998be4`, including per-layer config support, paired
content-specificity CIs, value-only enforcement, CPU tests, bf16 and 4-bit
workflows, and `notes/2026070979-champion-validation-experiment.md`. The bf16
A100 run is active and healthy at the latest checkpoint (c06/18), with
Transformers 4.57.6, champion graft active on E and placebo, and held-out
validation configured correctly. A separate 4-bit pod is being launched
first/alongside it using an Intel AutoRound-compatible Qwen3-MoE Int4 model; the
earlier GPTQ attempt failed because `gptqmodel` upgraded Transformers to 5.x,
but this was treated as a fixable environment-ordering problem, not a scientific
blocker. The 4-bit run is supplemental; bf16 remains primary. Both pods are
isolated.

**Writing and review state.** The provisional negative/bounding draft
(`e2bb83d`) is not final because it predates champion validation. It should be
rewritten after the new results, with Fable given full authorship latitude and
all evidence/context supplied but independently verified. The agreed form is a
readable academic paper with a technical bounding result and a dry, specific
postmortem of provenance errors, arm conflation, quantization confusion, late
controls, render fragility, statistical pitfalls, and improvised experimental
framing—not a motivational narrative. Fable should perform multiple coherence
iterations, then distinct adversarial-statistics, proofreader,
readability/focus, and fresh-reviewer passes. The paper must explain the method
plainly, avoid internal H-pack jargon unless essential, distinguish
bf16/value-only from supplemental 4-bit or packed-KV results, and let the new
champion outcome determine whether the central framing is positive or null.

**Handoff state.** No external publication has occurred. Once the experiment,
paper rewrite, review iterations, and repository checks are complete, the agent
is authorized to promote the finished paper to `README.md`, commit, and push
autonomously; anything outside the repository remains out of scope.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `a251f9cd9db06b8b7`
- `ae7cf96ea279748dc`
- `ac3918af2ffc97d0e`
- `af3935f188941e093`
- `a501dd78aedbef8aa`
- `a04a586e32f5194dc`
- `a3bb78279d88ee582`
- `aab84c6fd6b22b31f`
- `a55a4dc63308f4889`
- `ad5eac6bd788cdbcf`
- `af6f5c8198c95dc73`
- `a295603ac9efcd435`
- `ad25b18802e0e004e`
- `acb2c5ecce0e32153`
- `ac8c4927379b23177`
- `a9e3abb595c36c42e`
