# Coherent-state powered successor v13 — preregistration

**Design ID:** `coherent-state-powered-successor-v13`  
**Date:** 2026-07-12  
**Decision owner:** primary Codex agent acting under the owner's powered-successor handoff  
**Status:** **DRAFT — NO PAID WORK OR PRIMARY TREATMENT AUTHORIZED**

This is a new protocol. It does not amend or inherit authorization from v10,
v12, P01, or P02. Historical artifacts are evidence about failure modes and
costs only. Authorization has four acyclic stages: draft; a separately frozen
technical-only engineering canary with semantic `N=0`; static freeze that
authorizes capped paid treatment-blind Phase A; and a separate treatment
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
Every canonical parameter tuple first receives a stable candidate ID equal to
SHA-256 of the design ID, stratum ID, template version, and canonical tuple
bytes. Candidate IDs therefore do not depend on sampling order. Carrier-attempt
seeds and every other candidate-specific random seed derive from this stable ID,
never from a permutation rank. Before any candidate text or gate is inspected,
one 128-bit OS-random permutation seed per stratum is committed. NumPy PCG64
applies that seed to a without-replacement permutation of the stable-ID pool.
Permutation rank is recorded separately from candidate ID. The literal
permutation, generator version, parameter pools, template definitions, and
canonical-ID algorithm live under `data/coherent_state_powered_v13/` and are
bound by the static manifest. A tuple-ID or literal-history collision aborts
the recipe rather than resampling. Order never changes after static freeze.

All SHA inputs in this protocol use canonical UTF-8 JSON arrays with
`ensure_ascii=false` and separators `(',', ':')`; no informal delimiter
concatenation is allowed. A tuple's final element is its canonical JSON object
with sorted keys. A 128-bit permutation seed is recorded as exactly 32 lowercase
hex digits, converted by `int(seed_hex, 16)`, and passed directly to
`numpy.random.Generator(numpy.random.PCG64(seed_int)).permutation(pool_size)`
under the manifest-pinned NumPy version. The complete integer permutation is
persisted, so later replay never depends on another NumPy version.

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

Eligibility is a fixed potential attribute of each stable-ID tuple under its
predeclared render seeds and exact runtime. The independent random permutation
is therefore exchangeable over the finite eligible population. The unordered
first six eligible tuples are a simple random sample without replacement from
that population. Conditioning on the predeclared feasibility event that the
sixth eligible tuple appears by rank ten is also invariant to eligible-tuple
labels, so it does not change that uniform subset law. No rank-derived seed or
rank-varying treatment is permitted. The finite eligible unit includes its two
frozen render streams; the estimand is not an expectation over fresh render
RNG.

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
8. a production-tokenizer skeleton width of 940--1,180 tokens after subtracting
   only the dynamic carrier-content span from the complete canonical source
   through the final retained-tail assistant; the compiler proves the exact
   subtraction against its frozen sentinel carrier and framing;
9. targets 1--4 tokens and target/countertarget lengths differing by at most
   one token; focal target and countertarget token sequences, and nonfocal
   target and countertarget sequences, must be distinct and neither sequence
   may be a prefix of the other;
10. randomized blind singleton review of every C and W history;
11. independent target-aware paired review of derivation, minimality,
    downstream repair, nonfocal independence, retained-tail neutrality, and
    leakage;
12. cross-template review confirming the eight templates are substantively
    distinct while recording intentional within-template structure;
13. third-reviewer adjudication by another model family for any disagreement.

Failed candidates are archived under their original hashes. Component banks and
templates may be repaired only before permutation seeds are committed. Once a
seed is committed or any ranked candidate is materialized, no template, pool,
rule, candidate, or permutation repair/replacement is allowed under v13; a
mechanical or review failure is simply an ineligible frozen candidate. If the
pool/compiler itself is invalid, v13 terminates pre-treatment rather than
adapting its frame after inspection. Reviewed files are never edited in place.
Qwen is not an author or judge.

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
content tokens, normal EOS required. Attempt digest is
`SHA256(canonical_json([design_id, stable_candidate_id, render_index,
attempt_index]))`. Its first eight digest bytes are interpreted unsigned
big-endian and bitwise-ANDed with `(2^63)-1` to obtain the recorded nonnegative
63-bit PyTorch generator seed. At most three attempts per render are allowed,
in order.

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
assistant header). Their literal SHA/input/map keys are exactly lowercase
`content` and `structural`, respectively.

For each class of width `m`, compute one digest per integer row index `j` as
`SHA256(canonical_json([design_id, stable_candidate_id, render_id, class,
j]))` and sort indices by `(digest_bytes, j)` ascending. This literal order is
the class permutation. The frozen moved-row counts are
`[2,4,8,16,32,64,m]`, retaining only first occurrences whose values lie in
`[2,m]`. For a count `k`, take the first `k` permuted indices `p`. Direction
`+1` assigns correct-history donor row `C[p[t]]` to destination
`p[(t+1) mod k]`; direction `-1` assigns it to `p[(t-1) mod k]`. All
unselected V rows remain fresh. Classes never mix, and the same token-index map
is applied at every layer. Evaluate the at-most-98 candidates in nested order:
content counts in their retained list order, structural counts in their
retained list order, then directions `(+1,-1)`. Selection may inspect only
C/W/F row values and geometry, never a probe, target logit, continuation, or
outcome.

Each bf16 scalar is cast to IEEE float64 before subtraction. For layer `l`,
flatten the two classes in class-major, token-major, head-major,
dimension-major order and define `a_l=VP_l-F_l`, `b_l=C_l-F_l`, and
`d_l=C_l-W_l` in float64. Every dot product and sum of squares uses Python
`math.fsum` over the frozen layer-major then flattened-coordinate order. An
active layer has finite `norm(b_l)>0`; zero-real-delta layers, including
possible layer 0, are excluded rather than divided by zero. Its displacement
ratio is `r_l=norm(a_l)/norm(b_l)`. The aggregate ratio is
`sqrt(fsum_l ||a_l||^2)/sqrt(fsum_l ||b_l||^2)`. The aggregate cosine is
`fsum_l dot(a_l,d_l) / sqrt(fsum_l ||a_l||^2 * fsum_l ||d_l||^2)` over active
layers. Sort `(r_l, layer_index)` ascending for the median; with an even count,
use `math.fsum` of the two central ratios divided by two.

The first candidate in the frozen search order is accepted only if at least 24
layers are active, every moved destination row differs bytewise from fresh in
at least one active layer, the aggregate ratio is in `[0.75,1.33]`, the median
ratio is in `[0.50,2.00]`, and the absolute aggregate cosine is at most `0.20`.
Every interval is inclusive. A nonfinite input/difference/accumulator, zero
`a`, `b`, or `d` aggregate denominator, or undefined cosine rejects the
candidate.

If none passes, VP is `PLACEBO_UNAVAILABLE`; no tolerance changes. Realistic
exact-subject e01 geometry must demonstrate availability before scaled Phase A,
and VP availability is computed outcome-blind for every selected render before
treatment release. The availability rate and every rejected diagnostic are
reported. VP qualifies **only the value-only cell**: `VALUE_PLACEBO_COMPLETE`
requires both renders available in at least 44 of 48 primary fixtures (the
integer realization of at least 90%). Full-KV
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

For every new candidate semantic fixture, Phase A may expose only
content/review evidence, carrier attempts and leakage review, technical
identities, A_C/A_W/FF scores, forced-token support, source and fresh R2 tensor
bundles/hashes, VP construction availability without any continuation/probe,
and runtime/cost. It cannot construct, score, persist, or reveal
CC/WW/FC/FW/VP continuation or probe outcomes for those candidates. Persisting
the exact selected source/fresh rows is mandatory and is treatment-input reuse,
not an outcome.

There are exactly two noncandidate technical exceptions. The frozen,
outcome-seen e01 canary may execute all six arms and its already observed probe
in a separate `technical_e01` namespace. The frozen 4.5k-token outcome-free
geometry fixture may exercise the same six cache-construction, recomputation,
persistence, and fixed neutral-probe paths in `technical_long`; its probe fact
is byte-identical and directly visible in every history and is not a semantic
plant. These exceptions measure end-to-end arm time, identity, persistence, VP
availability, and length scaling. Their numerical values cannot enter N,
eligibility, a selector, a threshold, a template choice, or a scientific claim.
Only predeclared pass/fail technical diagnostics and timings may gate scaled
Phase A. No outcome-unseen legacy or new semantic case receives an exception.

The two exceptions may first run once under the separately released Stage-T
engineering canary in Section 15. That run has semantic `N=0`, is excluded from
every candidate, eligibility, selector, treatment, and inferential artifact,
and may measure only exact-subject load/identity, fixed technical path
correctness, wall time, VRAM, bundle I/O, deterministic VP construction, arm
time, harvest, and deletion. It requests no production permutation entropy,
does not import the production-pool materializer, and cannot inspect or create
any ranked v13 candidate text.

A content-eligible fixture becomes primary recovery-eligible only if, for both
accepted renders:

1. A_C greedily begins with the complete exact C target and has `M(A_C)>0`;
2. A_W greedily begins with the complete exact W target and has `M(A_W)<0`;
3. the nonfocal oracle under C/W and fresh compaction greedily begins with the
   exact nonfocal target and favors it over its countertarget;
4. `Dplus_ir >= 5.0` nats/token and `Dmargin_ir >= 5.0` nats/token;
5. all target scores are finite and no required generation is empty or capped;
6. FF does not greedily begin with the complete C target;
7. the complete canonical source through the final retained-tail assistant and
   before any probe is 900--1,400 production-tokenizer tokens, inclusive;
8. both target-neutral render reviews and generated-to-forced identities pass.

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

- raw `X_R/X_V` means and a stratified fixture-cluster bootstrap with exactly
  100,000 replicates. The seed digest is SHA-256 of canonical JSON
  `["coherent-state-powered-successor-v13","analysis",
  "stratified_cluster_bootstrap_v1"]`, namely
  `cd12a03d4b107b93598407b5a2e3b49708629e5b75357641a66b50efa982bb75`;
  its first-eight-byte 63-bit seed is `5553677475614063507`, passed to pinned
  NumPy PCG64. In every replicate, draw six fixture indices with replacement
  independently within each stratum, use the same index draws for both cells,
  retain both renders inside their fixture mean, and compute the equal-weight
  48-fixture raw mean. Report the one-sided 95th percentile using NumPy
  `quantile(method='higher')`, plus the replicate-stream hash;
- one-sided 95% Welch--Satterthwaite stratified and ordinary 48-fixture t UCBs
  labeled **model-based nominal**, not guaranteed 95%. The stratified variance
  is `sum_h (1/8)^2*s_h^2/6` with the usual component-wise Satterthwaite df;
  the ordinary UCB is `mean+t_.95,47*s/sqrt(48)`. Either returns no interval
  when its estimated variance is zero;
- stratum-specific and leave-one-stratum-out means, median, MAD, sign count,
  min/max, and complete fixture/render values;
- pooled within-fixture variance from the two same-C-origin render replicates:
  for each cell, average the 48 within-fixture sample variances
  `(render1-render2)^2/2` and report its square root;
- raw focal margin/correct/countertarget components, nonfocal movement,
  specificity, damage, and VP movement/availability;
- ratio-of-means `E(X)/E(Dplus)` by ordinary paired-fixture Fieller inversion,
  labeled model-based. For each cell use its 48 recovery values and the 48
  two-render-average damage values, ordinary mean covariance divided by 48,
  and two-sided `t_.975,47`. Invert
  `(xbar-r*dbar)^2 <= t^2*(Vx-2*r*Cxd+r^2*Vd)`; report the upper root when the
  quadratic is bounded and `+infinity` when the denominator coefficient is
  nonpositive, the discriminant is nonfinite/negative, or support is otherwise
  unbounded.

Behavioral eligibility requires both A_C renders to begin with the complete C
target and both FF renders not to do so. For each of the 48 eligible fixtures,
freeze `B_Ri=1` only if CC begins with the complete C-target token sequence in
both renders, `B_Vi=1` analogously for FC, and `B_i=max(B_Ri,B_Vi)`. Report all
three finite-panel counts with denominator 48; a missing/capped required greedy
generation is a technical failure, not a changed denominator. A marginal
binomial interval, if shown, is explicitly model-based because stratum flip
probabilities may differ; it is not part of the joint 0.05 family. Greedy
answers are generated only for A_C, A_W, FF, CC, and FC; wrong/placebo arms
remain score-only. Behavioral collapse accepts only upstream independently
validated release/case/render/target-token-bound rows with C render origin and
explicit nonempty, normal-EOS, uncapped generation validity for A_C, FF, CC,
and FC.

## 12. Analysis validation before release

The final algorithm must be implemented twice or independently recomputed and
pass before treatment release:

1. machine proof that the alpha ledger is exactly `0.02+0.02+0.01=0.05` and
   that the clipped range is exactly one;
2. at least 200,000 repeated stratified samples without replacement from a
   fixed finite population of 4,096 units per stratum under each crossing of
   bounded two-point, uniform, beta, skewed, rare-responder, and contaminated
   families with cell-dependence targets 0, 0.5, and 0.9; dependence is Pearson
   correlation of the two clipped endpoints after flattening the equal-weight
   eight-stratum population, and every realized correlation must lie within
   0.03 absolute of its target; exact finite-population clipped means and
   responder prevalences are the truths; realized dependence and stratum
   heterogeneity are recorded; empirical noncoverage may exceed 0.05 only up
   to the exact two-sided 99% Binomial(`trials`, 0.05) upper acceptance count,
   not a normal approximation;
3. a zero-variance/rare-responder case that the old draft falsely resolved must
   retain the positive Hoeffding radius and correct tail bound;
4. 200,000-trial power grids with these literal constructions: independent
   `{-0.5,+0.5}` cells at clipped means
   `{-0.10,0,0.05,0.10,0.15}` and zero raw responders; independent
   `{-s,+s}` cells at `s in {0,0.125,0.25,0.375,0.5}` and zero responders; and
   shared any-cell responder prevalence
   `p in {0,0.0025,0.005,0.01,0.02,0.05,0.10}`, with both raw cells equal to
   `5` for responders and otherwise `-0.5*p/(1-p)` so the population clipped
   mean is zero before finite-count rounding. Record exact realized finite-
   population means, SDs, and prevalences at every point. Freeze gate:
   N=48 joint-resolution probability must be at least 0.80 for two independent
   worst-variance endpoints on `{-0.5,+0.5}` with true mean `0.05` and zero
   `>0.5` responders, and at least 0.95 when both true clipped means are zero
   with SD at most 0.25 and zero `>0.5` responders; nonzero responder
   prevalences remain in the reported grid but are not subject to those two
   headline power gates;
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
timing canary using only the technical-long exception in Section 9; then
treatment-blind candidate carrier generation, A_C/A_W/FF screening, and
source-row persistence. Scaled screening requires observed VP availability on
e01. Before treatment release, every selected render must have its VP
availability status computed without a probe or continuation.

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
binding validates on the same runtime fingerprint.

A primary batch is bound to one `primary_batch_id`, GPU UUID, and runtime
fingerprint. Results from different primary batches or fingerprints are never
combined. If the host is lost before all 48 terminal cases, that entire batch
is quarantined and its scientific scores remain uninspected; only the technical
loss time and cost may be read. Before any primary score is unmasked, at most
one full-batch restart on a newly admitted host is allowed if the refreshed
worst-case projection can fund all 48 cases, the host audit, and the $8 reserve.
The restart gets a new batch-specific launch receipt and runtime attestation,
rebuilds every source/fresh foundation from the release-bound text/token
artifacts, and recomputes all 48 cases; old-host terminal cases and tensor
bundles cannot be reused. The Stage-B manifest's original Phase-A bundle hashes
remain immutable audit evidence rather than executable restart inputs. Before
any restarted arm score, the new batch receipt must list and hash-bind every
regenerated C/W/F foundation and its exact case/render/release/runtime source
mapping, and an independent verifier must accept the complete regenerated
foundation set. If that projection or binding fails, any score was inspected,
or the restart host is also lost, the terminal result is
`INVALID_TECHNICAL_INCOMPLETE_PRIMARY` with no primary bound. This is the sole
cross-host restart policy.

## 15. Staged release and unblinding boundary

There is no self-hashing commit or circular amendment.

### Stage T — technical engineering canary release

A technical manifest binds the exact subject loader/attestation, technical-only
runner, bundle/placebo/store/runtime dependencies, fault-injected provider
watchdog and cleanup path, dependency locks, and literal hash-allowlisted e01
and technical-long inputs at one immutable `technical_root_commit`. Production
pool, seed, permutation, ranked fixture, candidate review, selector, and
analysis artifacts are absent from both the inventory and import graph.

After the first Section 18 transition passes, a dedicated commit changes the
literal status from DRAFT to `TECHNICAL_CANARY_AUTHORIZED`. It contains only
that status transition and one technical manifest; the manifest names its
immutable parent rather than its own future commit. A post-commit receipt binds
the authorization commit and manifest bytes. The remote uses a clean detached
checkout, verifies the exact parent inventory and two-path child diff, and
fails on absent, duplicate, stale, modified, attached, dirty, wrong-parent,
extra-diff, wrong-status, or wrong-receipt input before model load.

Stage T authorizes at most two bounded allocation attempts but only one admitted
host and only the two Section 9 exceptions. A rejected host is positively
deleted before the second request; there is no restart after model loading or
technical execution begins. Stage T ends at the earlier of `$1.50` total
provider spend or 3,300 provider seconds, including acquisition, setup, idle,
harvest, and cleanup. It has no semantic N, no production entropy, and no warm
hold after completion. All artifacts are harvested and the pod is deleted and
independently observed absent. The `$1.50` is spent inside the existing `$12`
pilot/Phase-A bucket; the protected `$8` failure/cleanup reserve is unchanged.

### Stage A — static Phase-A release

A static manifest binds only experiment-bearing bytes: this preregistration and
additive disposition; template/pool/generator/permutation and content-review
files; selector/alpha and analysis code; successor schema/planner/runtime/
control/runner code; reused low-level primitives; dependency locks; model
contract; literal gate fixtures; and the exact **parent commit** containing all
those bytes. Mutable orientation docs and unrelated history are excluded.

After Section 18's full static gates pass, a dedicated commit changes status
from `TECHNICAL_CANARY_AUTHORIZED` to
`STATIC_FROZEN_PHASE_A_AUTHORIZED`. Its manifest references its immutable
parent tree/commit rather than the commit that contains itself. It authorizes
only the capped treatment-blind operations in Sections 13 and 16. No
CC/WW/FC/FW/VP continuation or probe may execute for a new semantic candidate;
only the two frozen technical carveouts in Section 9 may exercise those arms.

That authorization commit contains only the status transition and Stage-A
manifest. After it exists, a separate Stage-A launch receipt records the exact
authorization-commit hash and manifest SHA-256. The remote checks out that
commit detached, requires a clean tree, verifies that its parent is the
manifest's `static_root_commit`, and rehashes every experiment-bearing path from
that recorded parent tree rather than from the status-changing child. It also
verifies that the child diff contains exactly the new manifest plus the single
allowed preregistration status transition, verifies the child's literal status,
then binds the live host/runtime attestation to the receipt. Absent, duplicate,
stale, modified, dirty-tree, wrong-parent, extra-diff, wrong-status, or wrong-
receipt input fails before model execution.

### Stage B — primary treatment release

Phase-A artifacts and the exact first six eligible fixtures per stratum are
committed before release. A unique treatment manifest lists 48 case hashes, 96
accepted-render hashes, all rejected-attempt/review hashes, Phase-A score and
tensor-bundle hashes, VP-availability statuses, selector/analysis hashes, and
the immutable `phase_a_root_commit` that is its immediate parent. The commit
adding this manifest changes the literal status to `TREATMENT_RELEASED` and
contains no bytes other than that status transition and manifest. The launch
receipt, created after that commit exists, records the release-commit hash and
manifest SHA-256; the manifest never claims its own commit hash.

The remote checks out the recorded release commit, verifies its parent equals
`phase_a_root_commit`, requires a clean tree and literal `TREATMENT_RELEASED`
status, and rehashes every listed experiment/Phase-A byte from that parent tree.
It separately verifies that the child diff contains exactly the treatment
manifest plus the single allowed preregistration status transition, verifies
the launch receipt, then persists separate batch/host/runtime attestations. An
absent, duplicate, stale, modified, dirty-tree, extra-diff, wrong-status, wrong-
receipt, or wrong-parent release fails. Only then may treatment run. Terminal-
case reuse and a possible whole-batch restart follow Section 14; partial cases
quarantine and recompute.

## 16. Provider and compute gate

The latest provider checkpoint before the Stage-T amendment observed
`$57.1287946692`, no active pods, at `2026-07-12T18:26:48Z`. Refresh immediately
before allocation.

Hard buckets before observed release:

- at most `$12.00` total for admission, exact gates, e01/long pilots, render
  attempts, balanced yield measurement, and all treatment-blind Phase A. The
  one Stage-T technical canary is a `$1.50` and 3,300-provider-second sub-cap of
  this bucket, leaving at most `$10.50` if fully spent;
- at most `$30.00` projected for the 48-case treatment core after mandatory
  source-bundle reuse;
- at most `$4.50` for the independent-host audit;
- at least `$8.00` left unallocated for failed acquisition, cleanup, artifact
  recovery, and analysis. It is not called a CI reserve because no later look
  exists.

After e01/long pilots and the first two candidates in every stratum complete,
compute observed provider dollars/seconds per candidate, accepted-render yield,
content/Phase-A eligibility yield separately by stratum, source-bundle bytes,
and arm time. Yield is descriptive and cannot reduce the cost forecast. For
each unresolved stratum, assume every remaining frozen rank through ten must be
screened; a stratum with six eligible cases has zero remaining slots. Multiply
the sum of those slots by the maximum observed balanced-batch per-candidate
dollars, then add any still-unspent literal pilot/gate allowance, to obtain
`projected_remaining_phaseA`. Record `phaseA_start_balance`, the refreshed
provider-ledger `phaseA_spent`, and current unspent `live_balance`. Continue
screening only if both:

`phaseA_spent + projected_remaining_phaseA <= 12.00`

`projected_remaining_phaseA + projected_core + 4.50 + 8.00 <= live_balance`.

The first inequality enforces the whole Phase-A bucket. The second compares
only future costs with the current remaining balance, so already spent dollars
are not double-counted. A projection is recomputed after every candidate and
may decrease only because a frozen rank completed or a stratum reached six
eligible fixtures, never because pooled yield looked favorable.

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
rank, time, Phase-A dollar, or worst-case projection bound produces
`RECIPE_INFEASIBLE_PRETREATMENT` or `BUDGET_INFEASIBLE_PRETREATMENT`, preserves
all artifacts, and exposes no treatment.

The Stage-T host is harvested and deleted immediately after its bounded
technical measurements; it is never the core warm host. Keep the later admitted
core host warm through the treatment-release review for at
most 45 idle minutes or `$1.10`, whichever occurs first. If release is not ready,
harvest bundles and terminate; a new host must rebuild or independently verify
compatible bundles under the post-screen projection. No network volume is
created for the core under the current session-economics decision. Harvest and
verify before deletion.

## 17. Conclusion labels and surplus order

Terminal labels are mutually prioritized:

- **RECIPE_INFEASIBLE_PRETREATMENT** or **BUDGET_INFEASIBLE_PRETREATMENT:** a
  bounded static/Phase-A gate failed before any primary treatment;
- **INVALID_TECHNICAL_INCOMPLETE_PRIMARY:** no single runtime-bound batch
  completed all 48 fixtures under Section 14's restart rule, so no primary
  bound exists;
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

### DRAFT to TECHNICAL_CANARY_AUTHORIZED

Before the first paid minute:

1. the Stage-T manifest and import audit prove that only hash-allowlisted e01
   and technical-long inputs are reachable, semantic N is zero, production
   entropy is never requested, and production-pool modules cannot import;
2. the exact model ID/revision, bf16 dtype, eager backend, tokenizer/template,
   dependency, GPU, driver, and CUDA attestation plus L0/L1/L3 code passes local
   fake-runtime and corruption tests; the manifest freezes the on-host order as
   attestation and build ladder before any timing/VP/arm measurement, with
   immediate abort and cleanup on failure;
3. the bundle/VP/store path needed by the canary has passed independent review,
   exact byte-corruption tests, and CPU-as-execution, arbitrary-VP, partial-
   resume, and cross-runtime rejection cases;
4. the provider watchdog passes happy-path and fault-injection tests for the
   `$1.50`, 3,300-second, process-death, harvest, deletion-404, and active-
   inventory gates;
5. the technical release verifier accepts the exact parent/two-path child and
   rejects wrong status, parent, inventory, hash, diff, checkout, and receipt;
6. a dedicated authorization commit contains only the Stage-T manifest and
   literal DRAFT-to-`TECHNICAL_CANARY_AUTHORIZED` transition.

The technical run may then observe only the metrics in Sections 9 and 15. Any
semantic or production-pool artifact is a fatal protocol violation, not a
pilot result.

### TECHNICAL_CANARY_AUTHORIZED to STATIC_FROZEN_PHASE_A_AUTHORIZED

Before any paid treatment-blind semantic Phase A:

1. the archived Fable freeze review and independent adversarial
   statistics/release review have additive main-agent dispositions;
2. a fresh independent reviewer approves the corrected finite-sample formula,
   sampling law, probe construction, staged release, placebo algorithm, and
   Phase-A economics;
3. the corrected analysis simulation/golden artifact passes every Section 12
   threshold and an independent implementation agrees byte-for-byte;
4. complete template/pool/generator/permutation and content-review manifests
   pass locally for every candidate allowed by the ten-per-stratum cap;
5. local plus Stage-T build-ladder, corruption, checkpoint-quarantine, and
   warm-order tests pass, and the observed technical timings preserve the live
   `$12/$30/$4.50/$8` inequalities;
6. the Stage-A verifier has demonstrated acceptance plus deliberate token,
   position, source, hash, control, parent-commit, and duplicate-manifest
   rejection cases, and the detached remote checkout/clean-tree/status/receipt
   procedure has passed locally;
7. one dedicated authorization commit contains only the status/manifest needed
   by Section 15 and changes the technical status to
   `STATIC_FROZEN_PHASE_A_AUTHORIZED`.

### Phase A to TREATMENT_RELEASED

Before any primary treatment:

1. exact e01 and long-geometry gates/timings pass within the Phase-A cap;
2. the e01 VP is observed available and its tensor evidence independently
   validates;
3. the worst-case per-stratum rank-through-ten cost projection passes after two
   candidates per stratum and after every later candidate;
4. exactly six eligible fixtures per stratum, 96 accepted render artifacts,
   mandatory source/fresh bundles, all rejections, and outcome-blind VP statuses
   are durable and committed;
5. the live balance inequality and `$30` core projection pass;
6. an independent validator builds the acyclic Stage-B manifest, demonstrates
   every rejection path, and the release-only commit changes the literal status
   to `TREATMENT_RELEASED` with its launch receipt satisfying Section 15.

No informal message, old seal, green status string, or provider allocation can
waive these conditions.
