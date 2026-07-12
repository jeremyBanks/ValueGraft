# Coherent-state powered successor v13 — preregistration

**Design ID:** `coherent-state-powered-successor-v13`  
**Date:** 2026-07-12  
**Decision owner:** primary Codex agent acting under the owner's powered-successor handoff  
**Status:** **DRAFT — NO PAID WORK OR PRIMARY TREATMENT AUTHORIZED**

This is a new protocol. It does not amend or inherit authorization from v10,
v12, P01, or P02. Historical artifacts are evidence about failure modes and
costs only. Authorization has three acyclic stages: draft; static freeze that
authorizes a capped paid treatment-blind Phase A; and a separate treatment
release. Section 18 defines the transitions.

## 1. Purpose and claim boundary

The motivating question is whether K/V cache state written while a model
encodes a compact handoff under a full conversation carries useful
history-specific information that a fresh encoding of the identical visible
handoff lacks, and whether transplanting that state recovers information lost
by compaction.

The primary claim is deliberately narrower:

> For target-damaged fixtures drawn from the literal engineered recipe in this
> protocol, under the pinned Qwen subject, role-native q=1 replay schedule, and
> maximal carrier-boundary locus, what finite-sample upper bound can be placed
> on correct-target log-probability recovery from (a) correct-history full-KV
> and (b) correct-history value-only transplantation?

The joint primary result has two parts: simultaneous distribution-free UCBs on
the mean recovery clipped to `[-0.5, 0.5]` nat/token in the two named cells,
and a bound on the prevalence of fixtures whose best raw recovery exceeds
`0.5` nat/token. Raw-nat means and model-based intervals are mandatory
companions but are not mislabeled finite-sample guarantees. This two-part result
does not establish that write-time information is absent elsewhere. In
particular it does not cover downstream retained-tail rows, a maximal-state
restart, other layers/alphas, another serving schedule, another checkpoint,
natural traffic, or a live agent.

The finite named panel, the conditional recipe population, and legacy e02--e06
are three distinct targets and are never pooled silently.

## 2. Exact subject and environment

Primary subject:

- model: `Qwen/Qwen3-30B-A3B-Instruct-2507`;
- resolved revision: `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`;
- weights and K/V: bf16, no weight or KV quantization;
- attention: eager requested and resolved on all 48 decoder layers;
- expected geometry: 48 layers, 32 attention heads, 4 KV heads, head dimension
  128, RoPE theta 10,000,000;
- semantic scoring and greedy answers: temperature 0;
- carrier corpus generation only: seeded sampling in Section 5;
- primary device class: admitted secure A100 80GB;
- host driver, CUDA runtime, GPU UUID/name/memory, PyTorch, Transformers,
  tokenizer/template, model inventory, loaded parameter count, repository
  commit, and dependency lock hashes are persisted and independently checked.

Any mismatch aborts. A new model/config requires its own design and build
ladder; it cannot be substituted during a provider shortage.

Local technical development may use the cached `Qwen/Qwen3-0.6B` bf16/eager
subject. Local outcomes never enter semantic N or change the exact-subject
thresholds.

## 3. Sampling frame, strata, and independent unit

### 3.1 Target recipe

The target is an equal-weight mixture of eight fixed engineered strata:

1. threshold eligibility;
2. sequential decision with tie-break;
3. conjunction / all-conditions rule;
4. interval or schedule overlap;
5. arithmetic capacity or budget;
6. categorical membership / set logic;
7. ordered priority with one exception;
8. referent or alias resolution requiring a derived exact answer.

Each stratum has a separately authored parameterized conversation template and
a frozen parameter distribution. Templates differ in discourse order, rule
shape, domain, target type, and explicitness; within-template parameterization
is intentional and is part of the claim boundary. The recipe is not described
as a sample of all conversations or all templates.

Each candidate contains:

- one correct history `C` and one minimally counterfactual history `W`;
- one declared changed input before the compaction boundary;
- a focal target under C and countertarget under W, each 1--4 production-
  tokenizer tokens;
- one independent nonfocal fact/decision whose evidence, target, and declared
  countertarget are byte-identical; its evidence must remain visible in the
  system or retained tail after compaction;
- at least two genuine distractors;
- a byte-identical coherent retained tail beginning with a user turn and ending
  with an assistant turn;
- an explicit or unstated-derived subtype;
- no subject-model output, treatment result, old case text, or outcome in the
  authoring prompt.

Each stratum has a finite enumerated parameter pool of at least 4,096 tuples.
Before any candidate text or gate is inspected, one 128-bit OS-random seed per
stratum is committed. NumPy PCG64 applies that seed to a without-replacement
permutation of the pool. That literal permutation, generator version, parameter
pools, and template definitions live under `data/coherent_state_powered_v13/`
and are bound by the static manifest. Candidate IDs are their immutable
permutation ranks. A literal-history collision aborts the recipe rather than
resampling. Order never changes after static freeze.

### 3.2 Independent unit and selection

One paired base conversation fixture is one independent unit. Two carrier
renders, arms, targets, loci, schedules, layers, executions, and hosts are
nested measurements and never increase N.

Within each stratum, at most the first ten candidates in the frozen permutation
may advance through content and Phase-A gates. The first six eligible candidates
are the primary sample. If any stratum yields fewer than six eligible candidates
among those ten, the terminal pre-treatment result is
`RECIPE_INFEASIBLE_PRETREATMENT`; no alternate template, extra index, or primary
treatment is allowed under v13.

The primary sample therefore has exactly six eligible fixtures per stratum and
`N=48`. N=24 and N=32 are durability/progress checkpoints only; no treatment
statistic is computed or unmasked there. The single inferential analysis is at
N=48. Fixture execution order is frozen in an interleaved eight-stratum block
order and is unrelated to pod boundaries.

The realized content/Phase-A acceptance rate and every rejection reason are
reported. Inference is to the conditional recipe `R | eligible`, not the
unfiltered generator.

### 3.3 Legacy sensitivity

`e01` is outcome-seen and is technical/pilot material only. The five outcome-
unseen v12 cases `e02`--`e06` remain candidate legacy sensitivity cases pending
literal successor revalidation. If all pass, all five are rebound without
editing and run as one fixed panel if budget permits. They are never mixed into
the new recipe result, and no subset is selected.

## 4. Content eligibility before any subject forward

Every candidate must pass all of the following on literal bytes:

1. complete schema and exact author/template/parameter/seed provenance;
2. exclusive-create file and SHA-256 binding;
3. identical C/W role sequence, message count, system, retained tail, and every
   non-allowlisted message;
4. exactly the declared changed messages differ, all before compaction;
5. equal production-tokenizer width for each changed message;
6. identical canonical length, message starts, assistant q1 spans, event widths,
   N-plan geometry, and fresh-destination geometry;
7. exact decoded round trip, no embedded special literal, clipping, repeated
   padding, empty message, fake external action, or call above 4,096 tokens;
8. targets 1--4 tokens and target/countertarget lengths differing by at most
   one token;
9. randomized blind singleton review of every C and W history;
10. independent target-aware paired review of derivation, minimality,
    downstream repair, nonfocal independence, retained-tail neutrality, and
    leakage;
11. cross-template review confirming the eight templates are substantively
    distinct while recording intentional within-template structure;
12. third-reviewer adjudication by another model family for any disagreement.

Failed candidates are archived under their original hashes. Before static
freeze, a repair creates a new additive candidate and regenerates the literal
permutation/manifest; reviewed files are never edited in place. After static
freeze, only the first ten already bound indices may run and no repair or
replacement is allowed. Qwen is not an author or judge.

## 5. Genuine carrier renders

Each fixture has exactly two accepted independently seeded stochastic handoffs,
both generated on-policy under full history C. This targets the motivating
deployment path and makes the within-fixture difference a genuine same-origin
render replicate rather than a render-plus-origin contrast. Exact IDs from both
renders are separately forced under C, W, and F.

Exact prompt:

> Write a 40–60 word handoff that mentions only shared background and the
> existence—not the content—of prior decision criteria. Do not state or imply
> any option, answer, name, number, value, rule branch, outcome, or nonfocal
> fact. Output only the handoff.

Sampling is q=1, temperature `0.7`, top-p `0.95`, top-k disabled, maximum 80
content tokens, normal EOS required. Attempt seeds derive from
`SHA256(design_id | case_id | render_index | attempt_index)` and are converted
to a recorded nonnegative 63-bit PyTorch generator seed. At most three attempts
per render are allowed, in order.

An accepted render must:

- contain 40--60 whitespace-delimited words, 40--80 production-tokenizer
  content tokens, and end normally; both limits are deliberate and binding;
- contain no embedded special token or cap hit;
- contain no digit, target/countertarget, changed value, declared focal or
  nonfocal fact, or case-declared forbidden phrase under Unicode-casefolded
  matching; forbidden lists include digit, punctuation, and spelled-out forms
  of every number/name/value;
- be certified by an independent target-aware reviewer as compatible with both
  C and W and free of focal/nonfocal information.

Every rejected and accepted attempt is persisted before another attempt:
render index, seed, sampler/RNG/library versions, prompt, complete history, text,
token IDs, per-token log probabilities, stop token/reason, boundaries,
positions, and hashes. Rejection advances the seed; text is never edited.

For each accepted text, the exact same IDs are forced q=1 under C and W and
freshly encoded in F. Within a render, the selected carrier/post-compaction
tokens and logical positions are identical across C/W/F; the evicted C/W
histories are intentionally different. Generated-to-forced identity under C
must be bit-exact for IDs, positions, token log-probability bits, selected K/V
rows, and EOS witness. The two render results are averaged within fixture
before inference. Their difference is reported as nested render variability;
it is not semantic N.

## 6. Visible context, schedule, and primary locus

Every source history appends:

1. the exact carrier request from Section 5;
2. the accepted carrier assistant content;
3. user: `Acknowledge receipt of this handoff without adding or repeating any factual detail.`;
4. assistant: `Acknowledged.`;
5. the fixture's byte-identical retained tail.

Fresh compaction retains the original system message and the exact items 1--5,
at their original logical positions with the evicted-history gap. All arms use
the same visible text within a render.

`A_C` is the untouched full C-history source continued through the selected
carrier, acknowledgment, retained tail, and probe. `A_W` is the corresponding
untouched W-history source. `FF` and all graft arms begin from the compacted
system-plus-items-1--5 destination. Focal and nonfocal scores are separate forks
from the same completed base state: append one byte-identical case-declared user
probe, append the canonical assistant generation header, then teacher-force the
C target and W countertarget as two separate branches beginning at the first
assistant-content position. `L_C`/`L_W` are arithmetic mean token log
probabilities over only those declared content tokens; message framing is not
scored. Greedy generation starts at the identical answer position, is capped at
16 content tokens, must end with normal EOS for an oracle gate, and is recorded
separately from teacher forcing. The focal and nonfocal probe texts, targets,
countertargets, token IDs, answer position, and cap are bound in every fixture.

The primary source schedule `N` is role-native q=1 replay:

- initial system/user material plus assistant header is prefilled;
- historical assistant content is forced one token per model call;
- each canonical assistant close plus next user plus next assistant header is
  one structural call, split only above 4,096 tokens;
- carrier content and `Acknowledged.` are q=1;
- explicit logical `position_ids` and contiguous physical `cache_position` are
  recorded;
- no automatic hidden chunking is allowed.

The primary locus is `R2_carrier_boundary`: carrier assistant content plus its
complete canonical close, the acknowledgment user message, and the
acknowledgment assistant generation header, ending immediately before the first
`Acknowledged.` content token. It is the same v12-style maximal carrier
boundary, implemented in the new namespace with dynamic carrier text.

All rows after R2, including `Acknowledged.` and the retained tail, are causally
recomputed after intervention. The alternate turn-aligned P schedule is an
audit condition on the first primary fixture in each stratum and never a
replicate or substitute primary.

## 7. Frozen arm set and controls

For each render, build one immutable fresh R2 boundary and score these six
states:

- `FF`: fresh K, fresh V;
- `CC`: correct-history K and V;
- `WW`: wrong-history K and V;
- `FC`: fresh K, correct-history V;
- `FW`: fresh K, wrong-history V;
- `VP`: fresh K, nonsemantic whole-row-permuted correct-history V.

`CC` and `FC` are the two primary treatment cells. `WW` and `FW` are matched
history-specificity controls at identical token IDs and logical positions.
`VP` is an applied disruption placebo, not a semantic donor.

### V-row placebo

Cached K rows are post-RoPE and may never be permuted across positions. VP keeps
fresh K and moves complete bf16 V rows only. The two event classes are (a)
carrier assistant content R1 and (b) the structural suffix R2 minus R1
(canonical carrier close, acknowledgment user message, and acknowledgment
assistant header).

For each class of width `m`, SHA256 of
`design_id|case_id|render|class` defines a permutation of its row indices. The
frozen moved-row counts are `[2,4,8,16,32,64,m]`, retaining only distinct values
between 2 and m. For each ordered count pair and rotation direction `+1,-1`,
rotate the selected rows by one cycle and leave unselected rows fresh. Apply the
same token-index map at every layer. Evaluate at most the first 98 distinct
candidates in lexicographic `(content_count, structural_count, direction)`
order after SHA index permutation. Selection may inspect only C/W/F row values
and geometry, never a probe, target logit, continuation, or outcome.

All norms and cosines are accumulated on CPU float64 in layer-major,
class-major, token-major, head-major, dimension-major order. An active layer has
finite `||C_l-F_l||_F > 0`; zero-real-delta layers, including possible layer 0,
are excluded from ratios rather than divided by zero. A candidate is accepted
only if at least 24 layers are active, every moved row changes bytes in at least
one active layer, aggregate active-layer displacement ratio
`||VP-F||/||C-F||` is in `[0.75,1.33]`, median active-layer ratio is in
`[0.50,2.00]`, and absolute active-layer aggregate cosine with `C-W` is at most
`0.20`. A zero/nonfinite denominator or cosine rejects the candidate.

If none passes, VP is `PLACEBO_UNAVAILABLE`; no tolerance changes. Realistic
exact-subject e01 geometry must demonstrate availability before scaled Phase A,
and VP availability is computed outcome-blind for every selected render before
treatment release. The availability rate and every rejected diagnostic are
reported. VP qualifies **only the value-only cell**: `VALUE_PLACEBO_COMPLETE`
requires both renders available in at least 90% of primary fixtures. Full-KV
has the decoded-valid same-position WW history control but no nonsemantic
full-KV placebo, and is never described as placebo-complete. Primary numerical
bounds remain computable when VP is unavailable, with the control limitation in
the conclusion label. No positive/history-specific interpretation is allowed
unless correct-source movement also exceeds wrong-history movement and the VP
distribution in the preregistered fresh confirmation sample.

### Exact identity

An independent fresh execution supplies R2 rows that are reinserted at the same
positions for K, V, and K+V. Continued cache, logits, scores, and generated IDs
must be bit-exact to untouched fresh. A behavioral tolerance is forbidden.

## 8. Scores and per-fixture estimands

For each probe record persist:

- mean per-token log probability of the C target `L_C`;
- mean per-token log probability of the W countertarget `L_W`;
- margin `M = L_C - L_W`;
- complete token-level log probabilities and float32 bit strings;
- greedy generated content IDs/text, stop reason, and cap flag.

Never describe margin improvement as correct-target recovery unless `L_C` also
improves. No component is discarded.

For render `r` of fixture `i`:

- correct-target damage: `Dplus_ir = L_C(A_C) - L_C(FF)`;
- margin damage: `Dmargin_ir = M(A_C) - M(FF)`;
- full-KV primary recovery: `X_R_ir = L_C(CC) - L_C(FF)`;
- value-only primary recovery: `X_V_ir = L_C(FC) - L_C(FF)`;
- full specificity: `H_R_ir = L_C(CC) - L_C(WW)`;
- value specificity: `H_V_ir = L_C(FC) - L_C(FW)`;
- placebo movement: `P_ir = L_C(VP) - L_C(FF)` when available.

The raw fixture values `X_R_i` and `X_V_i` are arithmetic means over its two
accepted same-origin renders. Signed values are never truncated at zero. The
bounded primary mean endpoints are
`Z_R_i = clip(X_R_i, -0.5, 0.5)` and
`Z_V_i = clip(X_V_i, -0.5, 0.5)`. Clipping is part of the estimand, not data
cleaning. The separate tail endpoint preserves visibility of effects above the
cap. Raw correct-target effects, focal margin, nonfocal target/margin,
specificity, damage, placebo, within-fixture render difference, and
ratio-of-means are mandatory companion components.

## 9. Treatment-blind Phase-A eligibility

Phase A may expose only content/review evidence, carrier attempts and leakage
review, technical identities, A_C/A_W/FF scores, forced-token support, source
and fresh R2 tensor bundles/hashes, VP construction availability without any
continuation/probe, and runtime/cost. It cannot construct, score, persist, or
reveal CC/WW/FC/FW/VP continuation or probe outcomes. Persisting the exact
selected source/fresh rows is mandatory and is treatment-input reuse, not an
outcome.

A content-eligible fixture becomes primary recovery-eligible only if, for both
accepted renders:

1. A_C greedily begins with the complete exact C target and has `M(A_C)>0`;
2. A_W greedily begins with the complete exact W target and has `M(A_W)<0`;
3. the nonfocal oracle under C/W and fresh compaction greedily begins with the
   exact nonfocal target and favors it over its countertarget;
4. `Dplus_ir >= 5.0` nats/token and `Dmargin_ir >= 5.0` nats;
5. all target scores are finite and no required generation is empty or capped;
6. both target-neutral render reviews and generated-to-forced identities pass.

This defines `R | eligible`. It is a sensitivity-enriched benchmark and not a
prevalence estimate for ordinary conversations. Candidate eligibility and
selection are committed in a treatment release manifest before any primary
outcome. If six eligible fixtures per stratum are not available within the
first ten frozen candidates, no primary treatment begins.

## 10. Primary finite-sample confidence rule

There are `H=8` fixed strata, six fixtures per stratum, and N=48. Because the
sample is balanced, the equal-weight recipe mean is the ordinary mean across
all 48 fixture values. The frozen without-replacement sampling order and bounded
endpoints permit Hoeffding's sampling-without-replacement bound; using the
independent bounded form is conservative. No normality, nonzero observed
variance, or asymptotic approximation enters the primary guarantee.

The joint family alpha `0.05` is allocated once at the sole final analysis:

- `0.04` to the two clipped-mean cells, `gamma_R=gamma_V=0.02`;
- `0.01` to the large-responder prevalence endpoint.

For `c in {R,V}`, with `Z_ci in [-0.5,0.5]`:

```
mean_Z_c = (1/48) * sum_i Z_ci
U_Z_c = min(0.5,
            mean_Z_c + sqrt(log(1/gamma_c) / (2*48)))
U_Z_primary = max(U_Z_R, U_Z_V)
```

The range is exactly one nat, hence no range multiplier is omitted. Missing or
nonfinite cells, duplicate fixtures/renders, wrong render count, unbalanced
strata, an altered sample order, or N other than 48 fails closed. Constant/all-
zero observations retain the positive Hoeffding radius; they never create a
zero-width interval.

Large response is
`H_i = 1{max(X_R_i, X_V_i) > 0.5 nat/token}`. Let `S=sum_i H_i`. If `S=0`,
the without-replacement zero-event probability is no larger than its
with-replacement counterpart; AM--GM then gives the conservative average-
prevalence bound
`U_H = 1 - 0.01^(1/48)`, even with unequal stratum prevalences. If `S>0`, use
the distribution-free Hoeffding bound
`U_H = min(1, S/48 + sqrt(log(1/0.01)/(2*48)))`; no binomial-iid claim is made.

The union bound gives at least 95% joint coverage for both clipped cell means
and the responder-prevalence component under the frozen sampling law. Cell
correlation is harmless.

The operational resolution thresholds are `delta_clip=0.35` nat/token and
`delta_tail=0.10`. The clipped threshold is about three to six times the
observed 0.06--0.13-nat schedule sensitivity, at most 7% of the minimum eligible
damage, and a 1.42 odds multiplier for one token. It is a scoped resolution
target, not universal practical irrelevance. The actual UCBs are always
reported.

N=24 and N=32 reveal only completion/cost/control counts. Treatment scores stay
masked. At N=48:

- joint resolution requires `U_Z_primary <= 0.35` and `U_H <= 0.10`;
- cell-specific clipped bounds remain valid and are reported even if the joint
  criterion fails;
- any raw positive/history-specific claim requires a separately frozen fresh
  sample and cannot be manufactured from the primary alpha.

## 11. Raw-nat, render, placebo, and behavioral companions

Mandatory companions, none substituted for the finite-sample primary bound:

- raw `X_R/X_V` means and a stratified fixture-cluster bootstrap with 100,000
  within-stratum resamples, both renders retained as one cluster;
- Welch--Satterthwaite and ordinary t UCBs labeled **model-based nominal**, not
  guaranteed 95%; zero observed variance returns no nominal interval;
- stratum-specific and leave-one-stratum-out means, median, MAD, sign count,
  min/max, and complete fixture/render values;
- pooled within-fixture variance from the two same-C-origin render replicates;
- raw focal margin/correct/countertarget components, nonfocal movement,
  specificity, damage, and VP movement/availability;
- ratio-of-means `E(X)/E(Dplus)` by Fieller inversion labeled model-based and
  returning `+infinity` when denominator support is inadequate.

Behavioral eligibility requires A_C to generate the exact C target and FF not
to do so. Freeze one fixture-level endpoint:
`B_i = 1{CC or FC begins with the exact C target}`. It is reported as a named
finite-panel count. A marginal binomial interval, if shown, is explicitly
model-based because stratum flip probabilities may differ; it is not part of
the joint 0.05 family. Greedy answers are generated only for A_C, A_W, FF, CC,
and FC; wrong/placebo arms remain score-only.

## 12. Analysis validation before release

The final algorithm must be implemented twice or independently recomputed and
pass before treatment release:

1. machine proof that the alpha ledger is exactly `0.02+0.02+0.01=0.05` and
   that the clipped range is exactly one;
2. at least 200,000 trials under bounded two-point, uniform, beta, skewed,
   rare-responder, and contaminated distributions, with cell correlations 0,
   0.5, and 0.9; empirical noncoverage may exceed 0.05 only by a preregistered
   two-sided 99% Monte Carlo binomial tolerance around 0.05;
3. a zero-variance/rare-responder case that the old draft falsely resolved must
   retain the positive Hoeffding radius and correct tail bound;
4. power grids over clipped means, variance, and tail prevalence. Freeze gate:
   N=48 joint-resolution probability must be at least 0.80 for two independent
   worst-variance endpoints on `{-0.5,+0.5}` with true mean `0.05` and zero
   `>0.5` responders, and at least 0.95 when both true clipped means are zero
   with SD at most 0.25;
5. damage near the 5-nat gate correlated with raw recovery, verifying that the
   code labels the estimand conditional rather than claiming unfiltered R;
6. host offsets and identity/placebo failures abort without adding N;
7. golden cases for constant/all-zero scores, NaN, missing cells, row order,
   duplicate renders/fixtures, wrong stratum counts, raw effects beyond both
   clipping limits, zero and nonzero responder counts, and Fieller unbounded
   denominators;
8. byte-for-byte agreement between production and independent final-bound
   recomputation.

The prior nominal-t first-pass simulation is retained as negative design
evidence; it observed severe undercoverage on skewed/rare-responder mixtures and
cannot authorize treatment. Simulation failures revise the draft before
outcomes; they never tune thresholds afterward.

## 13. Exact production gates

Before any paid Phase A on the primary host, the static release must bind:

1. successor release/case/review/selector/code/lock/model binding, plus deliberate
   one-token, position, source, hash, and control corruptions that fail;
2. exact host/model/revision/dtype/backend/geometry/tokenizer/template/dependency
   attestation once per load;
3. relevant L0/L1/L3 deterministic replay and continuation identities;
4. stochastic sampled-to-forced q1 identity with fixed seed;
5. exact fresh self-replacement at dynamic R2 for K, V, and K+V;
6. selected-row exactness and non-selected-row preservation;
7. a signed bidirectional R2 bf16 path control on the frozen technical fixture,
   using exactly four `nextafter` ULP steps at the frozen maximal-gradient
   coordinates and requiring plus/minus margin movements of at least `1e-4` in
   opposite directions;
8. the complete deterministic VP search and its expected unavailable path;
9. checkpoint corruption and hard-kill tests with no more than one active
   case at risk;
10. warm-order equivalence: a case standalone, after another case, and in
    reversed order must have exact plan/row/score bits;
11. literal technical/e01/long-geometry fixture hashes, path-control coordinates
    and comparator semantics, warm-order fields, kill points, and rejection
    codes.

The static freeze then authorizes, in this order and only within Section 16's
Phase-A cap: one e01 short production timing/VP canary; one 4.5k-token geometry
timing canary; then treatment-blind candidate carrier generation, A_C/A_W/FF
screening, and source-row persistence. Scaled screening requires observed VP
availability on e01. Before treatment release, every selected render must have
its VP availability status computed without a probe or continuation.

The independent-host audit replays both saved renders of the eligible-rank-1
fixture from each stratum (eight fixtures) after the primary run. It requires a
different GPU UUID and fresh provider allocation but the same model revision,
dtype, backend, dependency hashes, and admitted GPU class. Token IDs, plan/call
traces, selected-row hashes, and greedy IDs must match exactly. Every persisted
target log probability and margin must differ by at most `1e-5` nat. Failure
does not void the internally valid primary-host result; it makes
`PORTABILITY_UNRESOLVED` the highest portability label and scopes all numerical
claims to the primary runtime fingerprint.

The old non-gating natural calibration is not inherited because it gates no v13
claim.

## 14. Warm-session state and persistence

One prepared model is retained per admitted host. Cases execute sequentially at
batch size one; variable histories are never tensor-batched.

Retained state is keyed by
`(design_id, release_hash, runtime_fingerprint, case_id, render_id, schedule, history)`.
At most one case and one render foundation are live. Code asserts bounds on one
active case, three histories, six arms, maximum 7,000 live tokens, maximum 256
selected rows, 48 layers, and a measured per-case deadline. Every reference is
evicted in `finally` after terminal persistence or error.

Per case/render persistence order:

1. exclusive-create `STARTED` record;
2. accepted/rejected render artifact, read-back and hash;
3. plans and Phase-A/oracle scores;
4. immutable lossless selected C/W/F R2 tensor bundles plus foundation binding;
5. each arm checkpoint synchronously before the next model operation;
6. terminal case artifact, independent validation, then eviction.

The append-only session index contains hashes/bindings, not embedded case
payloads. Every generated text/token stream, score, call/position trace,
selected-row hash, control diagnostic, environment record, partial/invalid
result, and provider receipt lands in `results/coherent_state_powered_v13/`
under a unique model-and-UTC name. Rejected or invalid artifacts go to a
quarantine subdirectory and are never deleted.

Lossless selected-row tensor bundles are mandatory for technical/control
fixtures and every Phase-A-eligible semantic render. They are bounded by 256
rows x 48 layers and permit treatment reuse without rerunning historical source
replay on the same compatible runtime. Every render is mandatory text/token
evidence.

A hard kill or process error quarantines every partial artifact for the one live
case. Partial arms are never resumed into a terminal case and never count as N;
the entire case/render is recomputed from its validated Phase-A bundles. A
terminal immutable case is skipped only after exact release/runtime/case/render
binding validates. This is the sole resume policy.

## 15. Staged release and unblinding boundary

There is no self-hashing commit or circular amendment.

### Stage A — static Phase-A release

A static manifest binds only experiment-bearing bytes: this preregistration and
additive disposition; template/pool/generator/permutation and content-review
files; selector/alpha and analysis code; successor schema/planner/runtime/
control/runner code; reused low-level primitives; dependency locks; model
contract; literal gate fixtures; and the exact **parent commit** containing all
those bytes. Mutable orientation docs and unrelated history are excluded.

After Section 18's unpaid gates pass, a dedicated commit changes status to
`STATIC_FROZEN_PHASE_A_AUTHORIZED`. Its manifest references its immutable parent
tree/commit rather than the commit that contains itself. It authorizes only the
capped treatment-blind operations in Sections 13 and 16. No CC/WW/FC/FW/VP
continuation or probe may execute.

### Stage B — primary treatment release

Phase-A artifacts and the exact first six eligible fixtures per stratum are
committed before release. A unique treatment manifest lists 48 case hashes, 96
accepted-render hashes, all rejected-attempt/review hashes, Phase-A score and
tensor-bundle hashes, VP-availability statuses, selector/analysis hashes, and
the immutable `phase_a_root_commit` that is its immediate parent. The commit
adding this manifest contains no other experiment change. The launch receipt,
created after that commit exists, records the release-commit hash and manifest
SHA-256; the manifest never claims its own commit hash.

The remote checks out the recorded release commit, verifies its parent equals
`phase_a_root_commit`, rehashes every listed byte, then persists separate live
host/runtime attestations. An absent, duplicate, stale, modified, or wrong-
parent release fails. Only then may treatment run. Terminal-case reuse follows
Section 14; partial cases quarantine and recompute.

## 16. Provider and compute gate

The last observed provider state before this draft was `$57.1287946692`, no
active pods, at `2026-07-12T14:39:21Z`. Refresh immediately before allocation.

Hard buckets before observed release:

- at most `$12.00` total for admission, exact gates, e01/long pilots, render
  attempts, balanced yield measurement, and all treatment-blind Phase A;
- at most `$30.00` projected for the 48-case treatment core after mandatory
  source-bundle reuse;
- at most `$4.50` for the independent-host audit;
- at least `$8.00` left unallocated for failed acquisition, cleanup, artifact
  recovery, and analysis. It is not called a CI reserve because no later look
  exists.

After e01/long pilots and the first two candidates in every stratum complete,
compute observed provider seconds per candidate, accepted-render yield,
content/Phase-A eligibility yield, source-bundle bytes, and arm time. Use the
maximum observed balanced-batch per-candidate time and the 90% one-sided exact
lower confidence bound on overall eligibility yield to project the candidates
needed, capped by ten per stratum. If the yield lower bound is zero, projection
is infinite. Continue screening only if:

`spent_phaseA + projected_remaining_phaseA + projected_core + 4.50 + 8.00 <= live_balance`.

Scaled Phase A also requires:

- gate durable completion within 900 provider seconds;
- e01 two-render/six-arm timing canary within 1,200 seconds;
- arm median at most 18 seconds and maximum at most 25 seconds;
- 4.5k-token two-render extrapolation at most 4,000 seconds;
- checkpoint quarantine/recompute, eviction, and warm equivalence pass;
- e01 VP available under the frozen search;
- projected treatment core at most `$30` after measured source-bundle reuse.

One fresh admitted-host timing measurement may distinguish a slow host from a
design overrun before unseen outcomes. The gate controls waste; it does not
authorize shrinking N after seeing treatment results. Exceeding a candidate,
time, Phase-A dollar, or yield bound produces
`RECIPE_INFEASIBLE_PRETREATMENT` or `BUDGET_INFEASIBLE_PRETREATMENT`, preserves
all artifacts, and exposes no treatment.

Keep the admitted core host warm through the treatment-release review for at
most 45 idle minutes or `$1.10`, whichever occurs first. If release is not ready,
harvest bundles and terminate; a new host must rebuild or independently verify
compatible bundles under the post-screen projection. No network volume is
created for the core under the current session-economics decision. Harvest and
verify before deletion.

## 17. Conclusion labels and surplus order

Terminal labels are mutually prioritized:

- **RECIPE_INFEASIBLE_PRETREATMENT** or **BUDGET_INFEASIBLE_PRETREATMENT:** a
  bounded static/Phase-A gate failed before any primary treatment;
- **INVALID_TECHNICAL:** a decision-bearing technical/release identity failed;
- **JOINT_BOUND_RESOLVED_VALUE_CONTROL_COMPLETE:** the joint bound resolved and
  `VALUE_PLACEBO_COMPLETE` also holds;
- **JOINT_BOUND_RESOLVED:** at N=48, `U_Z_primary <= 0.35` and `U_H <= 0.10`;
- **CLIPPED_MEAN_BOUND_ONLY:** both clipped cell UCBs are at most 0.35 but the
  large-responder prevalence bound exceeds 0.10;
- **CELL_BOUND_ONLY:** one named clipped cell UCB is at most 0.35 and the other
  is not;
- **BOUND_NOT_RESOLVED_AT_N48:** neither joint nor cell criteria resolve; report
  the valid numerical UCBs rather than calling the experiment failed;
- **PORTABILITY_UNRESOLVED:** the primary host is valid but the frozen second-
  host audit fails; this suffix overrides any portability wording but not the
  internally valid primary-host numerical label.

Regardless of label, report the actual per-cell means/UCBs, damage, specificity,
placebo, tail, flips, renders, strata, rejections, and costs. A null at R2 is
never phrased as no hidden cache information.

The Opus note whose slug is `speculation-why-graft-fails-and-mitigations`
pre-ranks downstream/retained-tail, verbatim second-carrier trail, read-demand,
and maximal-state tests. It supplies hypotheses, not authorization. No surplus
treatment, outcome-driven winner selection, or confirmation claim is authorized
by this document. After primary artifacts are safe, a separate additive
preregistration must freeze literal loci/trails/bridges, arms, sample supply,
estimands, selector/tie-break, multiplicity, and success rules. Until then the
work is fixed-panel exploration only.

M2' value-only remains the first-ranked rung because copied values are not
RoPE-positioned.
A full-KV second-copy-to-first-copy transplant is forbidden unless it preserves
the later logical positions or a separately validated key-position transform
has an effect floor well below the decision scale. No result from this menu can
retroactively change the primary R2 selector or UCB.

Only observed post-primary surplus beyond the `$8` failure/cleanup reserve may
fund the separately frozen exploration. Paid pods terminate before paper
analysis/writing.

## 18. Authorization conditions

### DRAFT to STATIC_FROZEN_PHASE_A_AUTHORIZED

Before any paid work:

1. the archived Fable freeze review and independent adversarial
   statistics/release review have additive main-agent dispositions;
2. a fresh independent reviewer approves the corrected finite-sample formula,
   sampling law, probe construction, staged release, placebo algorithm, and
   Phase-A economics;
3. the corrected analysis simulation/golden artifact passes every Section 12
   threshold and an independent implementation agrees byte-for-byte;
4. complete template/pool/generator/permutation and content-review manifests
   pass locally for every candidate allowed by the ten-per-stratum cap;
5. local build-ladder, corruption, checkpoint-quarantine, and warm-order tests
   pass;
6. the Stage-A verifier has demonstrated acceptance plus deliberate token,
   position, source, hash, control, parent-commit, and duplicate-manifest
   rejection cases;
7. one dedicated authorization commit contains only the status/manifest needed
   by Section 15 and changes the status to
   `STATIC_FROZEN_PHASE_A_AUTHORIZED`.

### Phase A to TREATMENT_RELEASED

Before any primary treatment:

1. exact e01 and long-geometry gates/timings pass within the Phase-A cap;
2. the e01 VP is observed available and its tensor evidence independently
   validates;
3. balanced yield/cost projection passes after two candidates per stratum;
4. exactly six eligible fixtures per stratum, 96 accepted render artifacts,
   mandatory source/fresh bundles, all rejections, and outcome-blind VP statuses
   are durable and committed;
5. the live balance inequality and `$30` core projection pass;
6. an independent validator builds the acyclic Stage-B manifest, demonstrates
   every rejection path, and the release-only commit/launch receipt satisfy
   Section 15.

No informal message, old seal, green status string, or provider allocation can
waive these conditions.
