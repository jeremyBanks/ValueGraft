# Local coherent-state N48 v3 — frozen design

**Design ID:** `coherent-state-local-mlx-n48-v3`
**Status:** `DESIGN_FROZEN_NO_V3_FORWARD_PASS`
**Subject:** `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`, revision
`e9675aa3ca5f900ccef55267914466d55ab325fa`
**Backend:** Apple MLX, mixed 4-bit/8-bit weight conversion, bf16 runtime
tensors, value-only intervention.

V1 and v2 are archived intact. V1 terminated before treatment because its
stochastic prompt/filter was contradictory. V2's only treatment-blind Phase-A
candidate proved the inherited margin/nonfocal gates mismatched the local
correct-target endpoint. That threshold-eligibility rank-1 candidate is an
unconditional development exclusion from v3 because its A/B evidence informed
this design. No E_C, E_W, or VP target outcome existed in v1/v2.

## Question and claim boundary

For a sensitivity-enriched paired conversation corpus, do value states written
for a fixed externally authored, contextually neutral token sequence forced
under full history C versus W contain history-specific information that improves
the exact hidden answer after identical visible text is freshly encoded without
that history?

This is an off-policy forced-carrier experiment on one exact model/backend. It
does not establish behavior of naturally generated summaries or production
compaction. The headline recovery is absolute correct-target log probability,
not C-vs-W discrimination. Fresh B may retain a positive C-vs-W margin while
assigning negligible probability to both alternatives; every such case and all
margin/margin-damage values remain reported.

## Frozen frame, unit, and candidate order

The base frame is the already-frozen deterministic 32,768-member powered-v13
recipe: 4,096 candidates in each of eight rule strata. OS-random seeds, complete
PCG64 permutations, ranks 1–10, literal fixtures, base reviews, and two fixed
carrier conditions were all committed before any treatment outcome.

One paired C/W fixture is one independent unit. V3 selects the first six valid
candidates by frozen rank in each stratum:

- `threshold_eligibility`: ranks **2–10** only; rank 1 is excluded development;
- the other seven strata: ranks **1–10**;
- categorical rank 1 retains its earlier outcome-blind base-text FAIL and is
  skipped independently of Phase A.

If threshold has fewer than six valid ranks among 2–10, or any other stratum
has fewer than six among its permitted ranks, v3 terminates rather than changing
text, thresholds, or the frame. Fixed N is 48, six per stratum.

## Fixed carrier conditions

The exact carrier bank is
`data/coherent_state_local_n48_v2/fixed-carriers-v1.json`, adopted byte-for-byte
with SHA-256 `41f0d41c6ddb21d16aaf877195f3dbfd2571382aeb4716c579b0f6b5a605fb44`.
The two externally authored generic texts passed 160/160 pure mechanical checks
and 160/160 independent semantic reviews across all 80 frozen fixtures.

They are two fixed carrier-text levels, not random renders. Their effects are
averaged within conversation before inference; they never increase N. Per-text
effects are mandatory fixed-condition sensitivity results, and opposite signs
limit any claim to the carrier exhibiting it.

## Source, destination, and forcing

For each C/W history, insert one fixed carrier before a neutral acknowledgment
and the byte-identical retained tail. Force exact carrier IDs q=1 under C and W;
persist per-token forced log probabilities. The packed fresh destination keeps
the original system, carrier request/text, acknowledgment, and retained tail,
with focal history absent. Persist fresh carrier log probabilities. These are
diagnostics only, never selectors.

Value rows for exact visible token twins map from source logical positions to
packed fresh destinations. Values are position-unrotated. Post-RoPE keys are
never transplanted. Independent same-schedule fresh executions, full cache/
next-logit identity, and a B-own-values assignment sham must be bit-exact.

## Arms and applied controls

Per candidate/carrier condition:

- `A_C`, `A_W`: untouched full-history oracle contexts;
- `B`: packed fresh visible context;
- `B_sham`: B's own values through the exact assignment path, bit-exact;
- `E_C`: fresh keys plus correct-C write-time values at exact visible twins;
- `E_W`: fresh keys plus wrong-W write-time values at the same destinations;
- `VP`: fresh keys plus deterministic displacement-matched whole-V-row
  placebo selected from C/W/B state geometry without targets or outcomes.

The audited state-only v13 placebo constructor is reused unchanged. `content`
is the carrier-content span; `structural` is other aligned request,
acknowledgment, and retained-tail rows. Classes never mix. Frozen SHA order,
moved-row counts, displacement-ratio interval, active-layer minimum, and cosine
threshold remain unchanged. Unavailable remains unavailable. Positive history-
specific language requires E_C to exceed both E_W and VP; E_C−B is still
reported as control-limited if VP is unavailable.

## Scores and v3 treatment-blind validity

At the identical answer position, append the focal probe and teacher-force C
target and W countertarget separately. Score only declared answer-content IDs.
Persist token log probabilities, `L_C`, `L_W`, margin `M=L_C-L_W`, correct-
target damage `Dplus=L_C(A_C)-L_C(B)`, and margin damage
`Dmargin=M(A_C)-M(B)`.

For **both** carrier conditions, a v3-valid candidate must have:

1. finite complete scores and passing fresh-repeat/sham identities;
2. A_C greedy content begins with the complete exact C-target token sequence
   and has positive margin;
3. A_W greedy content begins with the complete exact W-target token sequence
   and has negative C-minus-W margin;
4. B greedy content does not begin with the complete C-target sequence;
5. `Dplus >= 5.0` nat/token;
6. for each of nonfocal A_C, A_W, and B, the complete exact nonfocal target
   token sequence occurs contiguously within the first 16 greedy content tokens,
   the complete countertarget sequence does not occur contiguously there, and
   target-minus-countertarget margin is positive.

Exact subsequence matching uses token IDs, not decoded substrings. Dmargin is
persisted and reported but has no eligibility threshold. Phase A exposes only
these A/B validity results; E_C/E_W/VP target outcomes remain uncomputed until
the 48 IDs and both carrier conditions are committed in a treatment release.

## Estimand and confidence rule

For carrier `q` of fixture `i`,
`x_iq=L_C(E_C_iq)-L_C(B_iq)`. The conversation effect is
`X_i=(x_i,fixed_a+x_i,fixed_b)/2`; `Z_i=clip(X_i,-0.5,0.5)`. The headline is the
equal-weight mean over the balanced eight strata.

The bounded mean uses alpha `.04`, with N=48 one-sided Hoeffding radius
`sqrt(log(1/.04)/(2*48)) = 0.18311186883717767`. Report both lower and upper
endpoints; the upper answers the requested negative bound. Alpha `.01` gives an
exact one-sided Clopper–Pearson upper bound on prevalence of raw `X_i>0.5`, for
at least 95% simultaneous coverage. Raw mean/t interval, per-carrier effects,
damage, margin/margin damage, specificity, placebo movement, greedy answers,
nonfocal outcomes, and forcing diagnostics are mandatory companions.

No treatment peeking changes N or selection. Progress checkpoints every four
completed fixtures report valid N, elapsed wall time, resource state, and
failures, not aggregate treatment effect.

## Surplus order

After N48 is immutable: separate carrier-only versus retained-tail loci; then
the owner's verbatim doubled-carrier test with both-copy and second-copy-only
value grafts; then read-demand or layer/depth variants.
