# Coherent-state powered successor v13 — preregistration

**Design ID:** `coherent-state-powered-successor-v13`  
**Date:** 2026-07-12  
**Decision owner:** primary Codex agent acting under the owner's powered-successor handoff  
**Status:** **DRAFT — NO PAID OR PRIMARY TREATMENT OUTCOME AUTHORIZED**

This is a new protocol. It does not amend or inherit authorization from v10,
v12, P01, or P02. Historical artifacts are evidence about failure modes and
costs only. This document becomes `FROZEN` only after its literal statistical,
fixture, control, release, and analysis artifacts pass the reviews and tests in
Section 18.

## 1. Purpose and claim boundary

The motivating question is whether K/V cache state written while a model
produces a compact handoff under a full conversation carries useful
history-specific information that a fresh encoding of the identical visible
handoff lacks, and whether transplanting that state recovers information lost
by compaction.

The primary claim is deliberately narrower:

> For target-damaged fixtures drawn from the literal engineered recipe in this
> protocol, under the pinned Qwen subject, role-native q=1 replay schedule, and
> maximal carrier-boundary locus, what is the one-sided 95% upper confidence
> bound on mean correct-target log-probability recovery from (a) correct-history
> full-KV and (b) correct-history value-only transplantation?

The primary result is the maximum simultaneous UCB across those two cells. It
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
- one independent nonfocal fact/decision whose evidence and target are
  byte-identical;
- at least two genuine distractors;
- a byte-identical coherent retained tail beginning with a user turn and ending
  with an assistant turn;
- an explicit or unstated-derived subtype;
- no subject-model output, treatment result, old case text, or outcome in the
  authoring prompt.

The literal template definitions, parameter pools, generator version, and
candidate seed rule live under `data/coherent_state_powered_v13/` and are bound
by the release manifest. Candidate seeds derive from
`SHA256(design_id | stratum | candidate_index)`. Duplicate literal histories
are rejected. Candidate order cannot change after a gate result.

### 3.2 Independent unit and selection

One paired base conversation fixture is one independent unit. Two carrier
renders, arms, targets, loci, schedules, layers, executions, and hosts are
nested measurements and never increase N.

Within each stratum, candidates advance in frozen index order through content
and Phase-A gates. The first eight eligible candidates are frozen before any
primary treatment outcome:

- eligible positions 1--6: primary sample;
- eligible position 7: downstream discovery reserve;
- eligible position 8: downstream confirmation reserve.

If a stratum has fewer than eight eligible candidates, new candidate indices
continue under the already frozen generator while all treatment remains
machine-blocked. No replacement or authoring occurs after any primary treatment
outcome is exposed.

The primary sample therefore has six eligible fixtures per stratum, maximum and
planned `N=48`. Balanced looks use the first 3, 4, and 6 eligible fixtures per
stratum: `N=24`, `N=32`, and `N=48`. Fixture order within a look is frozen in an
interleaved eight-stratum block order and is unrelated to pod boundaries.

The realized content/Phase-A acceptance rate and every rejection reason are
reported. Inference is to the conditional recipe `R | eligible`, not the
unfiltered generator.

### 3.3 Legacy sensitivity

`e01` is outcome-seen and is technical/pilot material only. All five outcome-
unseen v12 cases `e02`--`e06` are rebound without editing and run as one fixed
legacy sensitivity panel if budget permits. They are never mixed into the new
recipe CI, and no subset is selected from them.

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

Failed candidates are archived under their original hashes. Repairs create new
candidate indices; reviewed files are never edited in place. Qwen is not an
author or judge.

## 5. Genuine carrier renders

Each fixture has exactly two accepted stochastic subject-generated handoffs:

- render 1 originates under full history C;
- render 2 originates under full history W.

Exact prompt:

> Write a 40–60 word handoff that mentions only shared background and the
> existence—not the content—of prior decision criteria. Do not state or imply
> any option, answer, name, number, value, rule branch, outcome, or nonfocal
> fact. Output only the handoff.

Sampling is q=1, temperature `0.7`, top-p `0.95`, top-k disabled, maximum 80
content tokens, normal EOS required. Attempt seeds derive from
`SHA256(design_id | case_id | render_index | attempt_index)` and are converted
to a recorded nonnegative 63-bit PyTorch generator seed. At most three attempts
per origin are allowed, in order.

An accepted render must:

- contain 40--80 content tokens and end normally;
- contain no embedded special token or cap hit;
- contain no digit, target/countertarget, changed value, declared focal or
  nonfocal fact, or case-declared forbidden phrase under Unicode-casefolded
  matching;
- be certified by an independent target-aware reviewer as compatible with both
  C and W and free of focal/nonfocal information.

Every rejected and accepted attempt is persisted before another attempt:
origin, seed, sampler/RNG/library versions, prompt, complete history, text,
token IDs, per-token log probabilities, stop token/reason, boundaries,
positions, and hashes. Rejection advances the seed; text is never edited.

For each accepted text, the exact same IDs are forced q=1 under C and W and
freshly encoded in F. Within a render, visible tokens and logical positions are
identical across histories. Generated-to-forced identity on the origin history
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
fresh K and permutes complete bf16 V token rows within frozen event classes,
using the same fixed-point-free cyclic permutation at every layer. Candidate
shifts are a SHA-derived frozen order over all nonzero shifts in each class.
Selection may inspect only tensor geometry and C/W/F row values, never a probe,
target logit, continuation, or treatment outcome.

The first candidate is accepted only if:

- applied bytes differ from FC and FF at every layer with a nonzero real
  displacement;
- no token row remains fixed in a class of size greater than one;
- aggregate Frobenius displacement norm ratio
  `||VP-F|| / ||C-F||` lies in `[0.80, 1.25]`;
- the median per-layer norm ratio lies in `[0.80, 1.25]`;
- absolute aggregate cosine with the semantic `C-W` V delta is at most `0.10`.

If no candidate passes, VP is `PLACEBO_UNAVAILABLE` for that render; no
tolerance is changed and no substitute is invented. Exact-subject pilot
geometry must demonstrate at least one available VP before primary treatment.
The availability rate across primary renders is reported. If fewer than 90% of
eligible primary fixtures have two available VP renders, the main treatment UCB
may still be computed but the result is explicitly **not matched-placebo-
complete** and cannot receive the strongest conclusion label in Section 17.

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

The fixture values `X_R_i` and `X_V_i` are arithmetic means over its two
accepted renders. Signed values are never truncated at zero. Focal margin,
nonfocal target/margin, specificity, damage, placebo, render-origin interaction,
and ratio-of-means are secondary components.

## 9. Treatment-blind Phase-A eligibility

Phase A may expose only content/review evidence, carrier attempts and leakage
review, technical identities, A_C/A_W/FF scores, forced-token support, and
runtime/cost. It cannot construct, score, persist, or reveal CC/WW/FC/FW/VP
probe outcomes.

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
outcome. If eight eligible fixtures per stratum are not available, no primary
treatment begins.

## 10. Primary sequential confidence rule

There are `H=8` fixed strata with equal weights `w_h=1/8`. At look k each
stratum contributes `n_h` fixture values per cell, with
`n_h in {3,4,6}` for `N in {24,32,48}`.

For cell `c in {R,V}`:

```
mu_hat_c = sum_h w_h * mean_hc
var_hat_c = sum_h w_h^2 * s_hc^2 / n_h
df_c = var_hat_c^2 /
       sum_h ((w_h^2 * s_hc^2 / n_h)^2 / (n_h - 1))
```

Terms with zero variance contribute zero to the denominator. If every stratum
variance is zero, the UCB equals the observed mean. Missing/nonfinite cells,
invalid degrees of freedom, duplicate fixtures, render duplication, or
unbalanced looks fail closed.

One-sided family alpha `0.05` is spent over looks as:

- N=24: `alpha_1 = 0.005`;
- N=32: `alpha_2 = 0.010`;
- N=48: `alpha_3 = 0.035`.

Within a look, divide alpha equally across the two primary cells:

```
gamma_kc = alpha_k / 2
U_kc = mu_hat_c + t(df_c, 1 - gamma_kc) * sqrt(var_hat_c)
U_primary_k = max(U_kR, U_kV)
```

Under independent normal errors within the frozen strata, the union bound over
all six look/cell events gives at least 95% simultaneous coverage for the
selected look. Correlation between cells is harmless. This assumption and its
stress tests are reported; the t result is not called distribution-free.

The absolute operational-resolution target is `delta_nat = 0.25` nat/token.
It is approximately two to four times the previously observed 0.06--0.13-nat
schedule sensitivity and corresponds to a 1.284 odds multiplier for a one-token
target. It is a stopping threshold, not a universal practical-importance claim.

At each scheduled look:

- if `U_primary_k <= 0.25`, stop the primary batch and report the observed
  bound;
- otherwise continue to the next frozen look;
- at N=48 stop and report, whether the bound resolved or remained above 0.25.

No lower-bound switch consumes this alpha. A large positive result is reported
as observed and can redirect only the preregistered untouched-reserve
exploration; any confirmatory positive claim requires a separately frozen fresh
sample. Pod/session boundaries never create looks. Completed overshoot cases
remain durable but are masked from the current look.

## 11. Robustness, heterogeneity, and behavioral endpoints

Primary companion analyses, without replacing the frozen UCB, are:

- stratified fixture-cluster bootstrap (100,000 resamples within strata), with
  both renders retained as one cluster;
- stratum-specific means and leave-one-stratum-out results;
- median, MAD, sign count, min/max, and complete fixture/render table;
- t interval after deleting no observations;
- bounded-score Hoeffding sensitivity for preregistered clipped endpoints at
  `[-5,5]` and `[-1,1]`, explicitly labeled as clipped estimands;
- Fieller one-sided ratio-of-means UCB `E(X)/E(Dplus)`; return `+infinity` if
  denominator support is inadequate or the confidence set is unbounded.

Rare response is a separate fixture endpoint:

`H_i = 1{max(X_R_i, X_V_i) > 0.5 nat/token}`.

Report exact Clopper--Pearson prevalence bounds for `P_R(H=1)` at each look
using its own explicitly labeled marginal alpha ledger. This tail result never
cancels the mean UCB and cannot be promoted as jointly 95% with the primary
unless its alpha is subtracted from the primary family in a later additive
amendment made before outcomes.

Behavioral eligibility requires A_C to generate the exact C target and FF not
to do so. One fixture-level flip endpoint is frozen:

`B_i = 1{CC or FC changes the damaged fresh answer to begin with the exact C target}`.

The any-cell definition prevents post-hoc cell selection. Exact binomial bounds
use the scheduled-look alpha rather than the fixed-N 0/24 shortcut. Greedy
answers are generated only for A_C, A_W, FF, CC, and FC; wrong/placebo arms are
score-only unless a later outcome-blind audit requires generation.

## 12. Analysis validation before release

The complete selector and stopping algorithm must be implemented twice or
independently recomputed and pass:

1. at least 200,000 simulated trials under independent normal stratum outcomes,
   with primary-cell correlations 0, 0.5, and 0.9;
2. skewed lognormal/beta, t3, two-point rare-responder, and single-outlier
   contamination stress tests, reporting—not concealing—t undercoverage;
3. damage near the 5-nat gate correlated with recovery;
4. power/stop grids over means, variances, and responder prevalence;
5. host offsets and identity/placebo failures that abort without adding N;
6. golden cases for constant scores, zero variance, NaN, missing cells, row
   order, duplicate renders, duplicate fixtures, look overshoot, zero flips,
   and Fieller unbounded denominators;
7. machine verification that spent alpha across primary look/cell events sums
   to exactly 0.05;
8. byte-for-byte agreement between the production analysis and an independent
   final-bound recomputation.

Simulation failures do not license threshold tuning after outcomes. Revise the
draft now or report the method's limitation.

## 13. Exact production gates

Before primary outcomes on each host/config:

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
8. observed VP availability at exact-subject realistic geometry with a bounded
   tensor evidence bundle;
9. checkpoint corruption and hard-kill resume tests with no more than one active
   case at risk;
10. warm-order equivalence: a case standalone, after another case, and in
    reversed order must have exact plan/row/score bits;
11. one e01 short production timing canary and one 4.5k-token geometry timing
    canary, both durable and independently read back.

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
4. immutable source/fresh foundation binding;
5. each arm checkpoint synchronously before the next model operation;
6. terminal case artifact, independent validation, then eviction.

The append-only session index contains hashes/bindings, not embedded case
payloads. Every generated text/token stream, score, call/position trace,
selected-row hash, control diagnostic, environment record, partial/invalid
result, and provider receipt lands in `results/coherent_state_powered_v13/`
under a unique model-and-UTC name. Rejected or invalid artifacts go to a
quarantine subdirectory and are never deleted.

Lossless tensor bundles are mandatory for technical/control-validation fixtures
and optional for every semantic case once exact model/code/input/row hashes are
durable. Every render is mandatory text/token evidence.

## 15. Release and unblinding boundary

The successor seal binds only experiment-bearing bytes:

- this preregistration and any additive amendment;
- literal template/parameter/generator and candidate files;
- content/review manifests;
- Phase-A carrier/eligibility manifest;
- selector/alpha manifest;
- successor schema/planner/runtime/control/analysis/runner code;
- relevant reused low-level source files;
- dependency lockfiles and model snapshot contract;
- exact execution commit.

Mutable orientation docs and unrelated repository history are not sealed.

Primary treatment is impossible until an independent validator commits a unique
release manifest containing the exact 48 primary and 16 reserve case/render
hashes and all gates. The remote job checks out the exact commit, rehashes every
bound byte, and refuses an absent, duplicate, stale, or modified release.
Resume accepts only terminal immutable per-case artifacts with matching release,
runtime, selector, case, and render fingerprints.

## 16. Provider and compute gate

The last observed provider state before this draft was `$57.1287946692`, no
active pods, at `2026-07-12T14:39:21Z`. Refresh immediately before allocation.

Planning buckets:

- `$1.50` admission, exact gate, and short/long pilots;
- `$18.00` primary through N=24;
- `$16.70` conditional continuation through N=48;
- `$4.50` independent-host audit;
- `$12.00` locked failure/CI reserve;
- remainder unallocated until observed cost releases it.

After pilots, scale only if all were observed:

- gate durable completion within 900 provider seconds;
- short two-render/six-arm case within 1,200 seconds;
- arm median at most 18 seconds and maximum at most 25 seconds;
- 4.5k-token two-render extrapolation at most 4,000 seconds;
- checkpoint reconstruction, eviction, and warm equivalence pass;
- projected primary core at most `$35`;
- `projected core + $4.50 host audit + $12 reserve <= current balance`.

One fresh admitted-host timing measurement may distinguish a slow host from a
design overrun before unseen outcomes. The gate controls waste; it does not
authorize shrinking N after seeing treatment results.

Keep the admitted core host warm through the case batch and treatment-release
review unless observed idle cost exceeds a fresh acquisition plus verified
restart risk. No network volume is created for the core under the current
session-economics decision. Harvest and verify before deletion.

## 17. Conclusion labels and surplus order

After a scheduled look:

- **BOUND_RESOLVED:** primary simultaneous UCB at most 0.25, all technical gates
  pass, and at least 90% of fixtures are matched-placebo-complete;
- **BOUND_RESOLVED_CONTROL_LIMITED:** UCB at most 0.25 but placebo completeness
  is below 90%; report the numerical bound and missing-control limitation;
- **POSITIVE_LEAD:** a primary mean is positive with compelling assumption-
  robust evidence, but no confirmatory positive population claim is made from
  this one sample;
- **INCONCLUSIVE_AT_CAP:** N=48 UCB remains above 0.25 without a confirmed
  positive effect;
- **INVALID_TECHNICAL:** a decision-bearing technical/release identity fails;
- **PORTABILITY_UNRESOLVED:** the primary host is valid but the frozen second-
  host replay fails its tolerance.

Regardless of label, report the actual per-cell means/UCBs, damage, specificity,
placebo, tail, flips, renders, strata, rejections, and costs. A null at R2 is
never phrased as no hidden cache information.

First surplus experiment, already split without treatment access, directly
tests the mechanisms proposed in the Opus speculation note whose slug is
`speculation-why-graft-fails-and-mitigations`. That note supplies hypotheses,
not evidence:

1. on eligible reserve position 7 in every stratum, compare (M1) downstream
   request/header and retained-tail rows, (M2') a position-preserving verbatim
   second carrier copy as a note-inducing trail, and (M3) a frozen read-demand
   bridge at the probe;
2. include a maximal-visible-state restart as a channel ceiling and an inert
   trail of matched length as the M2' negative control;
3. freeze one winning or null-covering condition using discovery outcomes;
4. confirm it once on untouched eligible reserve position 8 in every stratum;
5. then, if budget remains, shorter-context/schedule precision, layer/alpha,
   second checkpoint/model, true KV quantization, and natural/self-summary work
   in that priority order after separate additive preregistration.

M2' value-only is the first rung because copied values are not RoPE-positioned.
A full-KV second-copy-to-first-copy transplant is forbidden unless it preserves
the later logical positions or a separately validated key-position transform
has an effect floor well below the decision scale. No result from this menu can
retroactively change the primary R2 selector or UCB.

If primary stops at N24/N32, unused continuation funds become exploration
budget while the `$12` reserve remains locked until all data and host audits are
safe. Paid pods terminate before paper analysis/writing.

## 18. Conditions to change DRAFT to FROZEN

Before this document can authorize paid or unseen primary treatment:

1. a fresh focused Fable review writes a serious assessment into `notes/` using
   this literal draft, the design-review disposition, and the statistical audit;
2. the main agent dispositions every blocking Fable item additively;
3. independent statistical and harness reviewers approve the final literal
   implementation against Sections 10--15;
4. the analysis simulation/golden-test artifact passes;
5. the complete template/generator/candidate and content-review manifests pass;
6. local build-ladder, corruption, checkpoint, and warm-order tests pass;
7. the exact release verifier has demonstrated both acceptance and deliberate
   rejection cases;
8. a separately committed amendment changes `Status` to `FROZEN` and binds the
   release artifact hashes.

No informal message, old seal, green status string, or provider allocation can
waive these conditions.
