# Local coherent-state N48 v1 — frozen design

**Design ID:** `coherent-state-local-mlx-n48-v1`  
**Status before selection:** `DESIGN_FROZEN_NO_TREATMENT_OUTCOMES`  
**Subject:** `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` at revision
`e9675aa3ca5f900ccef55267914466d55ab325fa`  
**Backend:** Apple MLX; mixed 4-bit weights with 8-bit MoE gates; bf16 runtime
tensors; value-only intervention.

## Question and scope

For a deliberately sensitivity-enriched paired conversation corpus, does
replacing freshly encoded visible-token values with the same tokens' write-time
values from the correct full history improve the log probability of the exact
history-dependent answer? The study bounds this training-free intervention on
one exact local model/backend. It is not a prevalence estimate for ordinary
conversation, a bf16/full-KV result, or a task-success study.

## Frame, unit, and selection

The sampling frame is the existing deterministic powered-v13 recipe: 4,096
candidates in each of eight rule strata, 32,768 total. A fresh 128-bit seed per
stratum produces a complete PCG64 permutation. The seed record, literal order,
and the first ten candidate IDs per stratum are committed before any candidate
text or treatment outcome is inspected.

One paired C/W fixture is one unit. The first six eligible candidates in frozen
rank order in each stratum form fixed N=48. Text/mechanical/review failures and
oracle/fresh ineligibility advance only to the next frozen rank. If a stratum
has fewer than six eligible candidates among ranks 1–10, v1 terminates without
changing the frame or threshold.

Before treatment, independent reviewers inspect complete C and W histories for
coherence, opposite target entailment, minimal single-input change, retained-
tail neutrality, and absence of the answer in forbidden visible text. Rejected
text and reasons remain archived. The subject model is not a reviewer.

## Two carrier renders

Each candidate receives exactly two accepted stochastic carrier renders under
the complete C history. The exact request is:

> Write a 40–60 word handoff that mentions only shared background and the
> existence—not the content—of prior decision criteria. Do not state or imply
> any option, answer, name, number, value, rule branch, outcome, or nonfocal
> fact. Output only the handoff.

Sampling uses temperature `0.7`, top-p `0.95`, no top-k, and at most 80 content
tokens. Seeds are derived from
`SHA256(canonical_json([design_id, candidate_id, render_id, attempt]))`, first
eight bytes unsigned big-endian masked to 63 bits. At most three attempts per
render. An accepted carrier ends normally, has 40–60 whitespace words and
40–80 subject-tokenizer content tokens, contains no special token, digit,
target/countertarget, changed value, nonfocal fact, or fixture forbidden phrase,
and passes independent target-aware review as compatible with both C and W.
All attempts are saved unchanged.

The exact accepted carrier IDs are forced under W to create the wrong-history
donor and freshly encoded in the compacted destination. C and W source prefixes
differ; visible destination tokens and target scoring are identical within a
render. The two render outcomes are averaged within fixture and never counted
as N.

## Context and arms

The source contains the fixture's evicted C or W history, carrier request and
carrier, a neutral acknowledgment, and its byte-identical retained tail. The
packed fresh destination contains the original system message, exact carrier
request/carrier, acknowledgment, and retained tail; the earlier focal history
is absent. Values are position-unrotated, so correct and wrong source value rows
for exact visible tokens can be mapped to their packed destination token twins.
Post-RoPE keys are never transplanted.

Required states per render:

- `A_C`, `A_W`: untouched full-history oracle contexts.
- `B`: packed fresh visible context.
- `B_sham`: B's own values routed through the exact assignment path; must be
  bit-exact to B.
- `E_C`: fresh keys plus correct-C write-time values at exact visible twins.
- `E_W`: fresh keys plus wrong-W write-time values at the same destinations.
- `VP`: fresh keys plus the deterministic displacement-matched whole-V-row
  placebo selected from state geometry alone.

The placebo reuses the audited state-only v13 constructor. Its `content` class
is the carrier content and its `structural` class is the remaining aligned
carrier/acknowledgment/tail region. Classes never mix. It uses the frozen
SHA-sorted candidate order, moved counts, norm-ratio, layer-count, and cosine
criteria without access to probes, targets, logits, or outcomes. Unavailable
controls remain unavailable. Positive history-specific language requires
`E_C` to exceed both `E_W` and `VP`; the numerical E_C−B bound is still reported
when a placebo is unavailable, labeled control-limited.

## Scores and eligibility

From the identical answer position append the declared focal probe and
teacher-force separately the C target and W countertarget. Score only their
content token IDs. Persist token log probabilities and:

- `L_C`, `L_W`: arithmetic mean target log probabilities;
- margin `M = L_C - L_W`;
- correct-target damage `Dplus = L_C(A_C) - L_C(B)`;
- margin damage `Dmargin = M(A_C) - M(B)`.

Treatment-blind eligibility requires, in both renders: finite scores; A_C
favors C and A_W favors W; fresh B does not greedily begin with the full C
target; `Dplus >= 5.0` and `Dmargin >= 5.0` nat/token; nonfocal visible-fact
oracle/fresh checks pass; normal uncapped oracle generation; carrier review and
forced-replay identity pass. Only A/B validity is exposed before the fixed 48
IDs/renders are committed. E_C, E_W, and VP target outcomes remain uncomputed.

## Estimand and confidence rule

For render `r` of fixture `i`:

`x_ir = L_C(E_C) - L_C(B)`.

The fixture value is `X_i = (x_i1 + x_i2) / 2`; the primary bounded value is
`Z_i = clip(X_i, -0.5, 0.5)`. The headline is the equal-weight mean over the
eight balanced strata.

The mean endpoint uses family alpha `.04`; its distribution-free one-sided
Hoeffding radius at N=48 is
`sqrt(log(1/.04)/(2*48)) = 0.18311186883717767`. Report both
`max(-.5, mean(Z)-radius)` and `min(.5, mean(Z)+radius)`, with the upper endpoint
answering the requested negative bound. Alpha `.01` separately bounds the
prevalence of raw fixture effects `X_i > 0.5`; together the two statements have
at least 95% simultaneous coverage. Raw mean/t interval, stratified bootstrap,
render differences, damage, specificity `L_C(E_C)-L_C(E_W)`, placebo movement,
margin, greedy answers, and nonfocal outcomes are mandatory companions, not
substitutes for the finite-sample bound.

No interim treatment peeking changes N, selection, or stopping. Checkpoints and
progress reports every four completed fixtures report only valid independent N,
elapsed wall time, resource state, and failures—not aggregate treatment effect.

## Surplus order

After N48 collection is immutable: (1) downstream/retained-tail value locus;
(2) verbatim doubled carrier/summary with both-copy and second-copy-only value
grafts into matched destinations; (3) only then layer/depth/read-demand variants.
