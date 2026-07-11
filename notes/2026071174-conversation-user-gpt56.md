_This conversation documents the invalidation of v10, the discovery of
schedule-dependent bf16 divergence and defective controls, and a pivot from
blind v11 expansion to a smaller v12 carrier-state canary. No subject-model
forward or paid experiment has yet occurred; the project remains in apparatus
validation._

**Participants:** User, gpt-5.6-sol-xhigh, and gpt-5.6-sol-ultra.

**Handoff State.** V10 is permanently non-authorizing, its paid run was
cancelled before launch, and no semantic result was collected. The old ladder is
paused after c10 and c02 failed; c01 has no substantive completion. C10 showed
an approximately `0.06055`-nat fixed-margin shift, maximum K/V divergence
`16.125`, and continuation-logit change `0.84375`. C02 independently showed a
`0.1318359375` margin shift, K/V maxima `6.5/5.6875`, final-logit maximum `0.5`,
and continuation-logit maximum `0.46875`. These results are established for the
tested Qwen3-0.6B CPU/bf16/eager configuration, not the intended 30B A100 setup.

The frozen c10 origin diagnostic completed with `QUERY_SHAPE_ROUNDING`.
Equal-shaped 4,096-token branches whose future content differed were
bit-identical in the first 23 rows, while identical prefixes processed under
23-token and 4,096-token query shapes diverged. This rules out the suspected
future-token leak in c10 and identifies call-shape-dependent bf16 arithmetic as
the local cause. The durable methodological claim is narrower: at finite
precision, KV state depends on call boundaries, query shape, kernel, backend,
hardware, and library version. P and O must therefore be explicit replay
protocols, not assumed-equivalent executions. P is turn-aligned replay of
imported assistant text, not native live-agent state.

The old synthetic gate was invalid as general scientific validation. Its seven
fixtures were seven lengths of one five-token periodic stream, creating
pseudoreplication and allowing exact equality on that trajectory without
establishing natural-input schedule invariance. The old `G_wrong` control was
also invalid because it cycled short donor token IDs into target-length slots,
producing incoherent repetitive histories. The coarse `message_block` schedule
was mislabelled as message-aligned: it concatenated the full history before
chunking rather than replaying actual message boundaries. These controls cannot
authorize future work.

The hard natural-case gate nevertheless worked: it stopped paid semantic
execution before contamination and exposed schedule divergence and control
defects. The durable review rule is that every authorizing gate requires
raw-input inspection, independent fixture count, content-diversity and
production-representativeness review, exact claim boundaries, and fail-closed
technical, provenance, and content checks. Attribution of the cyclic-fixture
design is settled as Sol/gpt-5.6-sol’s primary design error, with Claude Opus
4.8 sharing responsibility for the independent-review failure; Fable did not
choose the fixture and identified its degeneracy when later shown the
construction.

The banked c10/c02 counterfactual candidates were mechanically exact but failed
blind naturalness and target-aware factual review. All four remain historical
feasibility witnesses only. The banked 30B corpus is unsuitable as primary
confirmatory material: 226/264 assistant replies hit the 320-token cap, raw
token IDs and resolved revision were not preserved, and the text is
model-authored surface text rather than preserved native live-cache state. It
may be retained after coherence review, but not used as clean primary evidence.

The earlier v11 replacement effort is paused. Three mechanically exact drafts
remain committed as explicitly `DRAFT_UNREVIEWED` engineering fixtures: c01 at
6,221 tokens, c04 at 6,688, and c05 at 6,862. Additional local checkpoints
existed for c02, c03, and c06, but no v11 forward pass occurred. The shared
planning-record scaffold and same-author provenance require blind diversity and
coherence review before any expansion; these drafts are not momentum toward a
confirmatory N.

**Design Pivot.** The project now uses a separate v12 carrier-state canary
rather than completing twelve long v11 cases before observing the exact model.
The canary tests whether an ordinary untrained 30B model leaves history-specific
decision state at an identical neutral handoff boundary, and separately tests
coherent full-KV versus the original value-only intervention. Its cases are
fixed stress tests, not statistical replicates, and canary cases are permanently
excluded from any later confirmatory corpus.

The primary semantic comparison is same-schedule, same-shape correct history
versus a coherent, plant-specific minimally counterfactual history. Fresh
compacted state is a utility contrast and includes destination-path execution
differences; it is not by itself a clean semantic causal contrast. Each
wrong-history arm must be scored on both focal plants so focal movement can be
separated from generic disturbance. Forced summaries are a controlled-mediator
design: they hold visible summary tokens fixed while changing source history,
but may create history-summary contradiction. Summary leakage must therefore be
classified before outcomes, and forced-summary NLL must be reported.

The current state-family plan distinguishes:

- Full coherent K+V insertion (`R`), testing whether a usable summary-state
  channel exists.
- Value-only insertion (`V`), testing the original mitigation idea directly.
- Fresh compacted state (`F`), correct-history state, and plant-specific
  counterfactual-history state.
- Nested carrier regions: R1 content, R2 content plus canonical close/boundary,
  and R3 including a fixed role-native bridge/anchor.

R2 is the primary decision region; R1 and R3 are descriptive extensions. Every
arm must contain identical visible carrier, close, bridge, anchor, and probe
text. Regions end only at complete call boundaries. The same visible text and
identical logical geometry are required across source histories and schedules. P
is the primary replay schedule; O, ordinary 4,096-token chunking, is a
prespecified sensitivity condition. Neither is native live-agent execution.

The canary’s primary history-specific estimands use correct-minus-counterfactual
comparisons under identical schedules, with focal selectivity against the
unchanged plant. Fresh comparisons report practical utility separately.
Correct-target and counterfactual-target log probabilities must be reported
separately; a larger margin alone does not establish improved competence.
Full-KV success does not authorize value-only claims. A positive R with null V
means a coherent channel may exist while naive value copying fails.

The v12 stimulus process has been repeatedly corrected. Revision-1 fixtures were
rejected because the retained tail inflated advertised pre-carrier lengths, five
cases lacked the intended causal depth, and the long case had chronology
defects. Revision-2 corrected the causal boundary and region geometry;
revision-3 repaired the long case and its stale control indices; revision-4
passed the blind singleton review and paired/diversity review for all six causal
pairs and twelve histories. The cases are mechanically valid, independently
reviewed, and still non-executable until the final metadata, manifest, and
apparatus bindings are frozen.

The v12 runtime architecture now includes tokenizer-only N/P replay planning,
fresh compact reconstruction, R1–R3 region extraction, K/V/K+V replacement,
causal downstream recomputation, target scoring, generated-versus-forced
identity logic, gradient-based bidirectional path control, deterministic value
placebos, an append-only store foundation, tensor-bundle support, and a separate
harvester. Focused fake-model and pure-function suites pass, but they do not
establish real Qwen causal behavior, gapped-mask correctness, bf16 gradient
equivalence, or authentic model outputs.

Several independent audits found and corrected real implementation defects: a
planner had split structural additions into the wrong call shapes; an R2
boundary initially cut through a call; a generated/forced fixture lacked
checkpoint binding; float32 bits were encoded with inconsistent endianness in
the harvester; review status was reported stale; tensor geometry and N/P
interval equality were under-bound; and loader inventory and
subject-specification paths could be caller-controlled. These are treated as
authorization blockers until corrected, tested, and independently harvested
rather than as minor bookkeeping issues.

**Current Blockers.** V12 is not frozen and no subject-model forward is
authorized. The independent harvester still requires production-grade
completion: little-endian raw-bit decoding, independent recomputation of
technical/path-control and natural-calibration outcomes, lossless gradient
evidence, actual selected/outside tensor-row lineage, probe-prefix binding to
scored states, exact receipt cardinality and parent-chain validation,
phase-separated eligibility versus treatment artifacts, review-schema
validation, placebo coverage, and machine-enforced treatment blindness.

The execution path also requires a production runner, exact local and 30B
loaders, technical/Phase-A/treatment release modes, bounded raw artifacts,
durable tensor persistence and resume checks, budget wrappers, and a clean
committed repository inventory. The loader must bind immutable model snapshots,
complete safetensors inventories, exact geometry, tokenizer/chat-template
identity, EOS policy, dtype, eager backend, device placement, dependency
versions, and clean Git state. The local apparatus uses Qwen3-0.6B CPU/bf16; the
exact subject is Qwen3-30B-A3B-Instruct-2507 at revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, bf16, eager attention, 48 layers, 4
KV heads, and head dimension 128. Protocol tokenization remains pinned to the
30B Instruct tokenizer.

**Required Sequence.** First complete and independently validate the harvester,
loader, tensor artifacts, phase-separated release state machine, technical
runner, and exact raw schemas. Then freeze the final v12 manifest and code
inventory. Run the `$0` local apparatus only after the frozen technical gates
pass; this local run is a technical diagnostic, not evidence about the 30B
numerical floor. Only a valid local apparatus can justify a bounded exact-30B
canary, expected at approximately `$3–$5` for one to three cases and excluded
from later confirmation. A strong canary would authorize a newly authored
independent confirmatory corpus; a null, severe leakage, absent headroom,
invalid control, or schedule interaction at effect scale would terminate the
current mechanism program under the stated budget.

No subject-model forward has occurred, no paid coherent-state compute has been
spent, and no pod is active. The intended document remains a paper-style
corrected negative or methodological report, which should begin in parallel
rather than waiting for a future positive result. A live coding-agent evaluation
remains deferred until a coherent state channel and native cache-forking
infrastructure clear their respective technical gates.

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
- `019f5308-3355-70b0-870b-0f9fd0117845`
- `019f5308-1975-7511-92ae-fd96e8fb151c`
- `019f5307-f779-7992-a5bf-8821fd6d05b9`
- `019f530f-5d11-7d90-96b2-9c0ca1cd5c7a`
