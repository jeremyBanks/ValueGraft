# Coherent summary state: Amendment 4 — eager-backend production equivalence

**Frozen:** 2026-07-11, after the gapped-v3 30B technical failure and before
any 30B semantic arm outcome from the position-preserving apparatus. This file
is additive. The original preregistration, Amendments 1–3, every failed run, and
every intermediate diagnostic remain part of the audit record.

## 1. Observations motivating this amendment

The exact gapped-v3 production checkpoint loaded in bf16 on an A100 and failed
before calibration, `A_full`, or any treatment outcome was scored. The committed
artifact is:

`results/coherent_state/coherent_state_gapped_v3_Qwen3-30B-A3B-Instruct-2507_20260711T080431Z/`

For the frozen five-token one-shot-versus-2+3 zero-gap comparison it recorded
maximum K, V, and final-logit differences of `1.625`, `0.4501953125`, and
`1.125`, against the unchanged `5e-4` limit. The gate failed closed and the pod
was retired. The exact model was `Qwen/Qwen3-30B-A3B-Instruct-2507`, revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, with 48 layers, bf16 parameters,
and the Transformers default SDPA attention path. Later 30B gates did not run.
There is no 30B semantic result to preserve or adapt to.

The committed local diagnostic introduced in commit `c146e545189d41db85bc0c75004d9605890b27e3`
repeated the same five-token schedule comparison on the exact cached
`Qwen/Qwen3-0.6B` checkpoint in CPU bf16. Under SDPA it recorded maximum
K/V/logit differences `1.0`/`1.0`/`0.4375` and a fixed selected-token-margin
shift of `0.109375` nat. Supplying independently constructed explicit causal
masks did not change those values. Under eager attention the corresponding
automatic-mask and explicit-mask comparisons were all exactly `0.0`; identical
repeats were also exact. These observations identify the attention backend and
query schedule as an execution variable in the tested fixture. They do not by
themselves prove a general mechanism for every SDPA implementation, model,
device, precision, or sequence length.

A separate local bf16 eager development pass reached the exploratory `G_delta`
placebo and raised before serializing its offending values, exposing a persistence
defect. Commit `e162f4cf40050f960edc4beda0bd40b0a1d9cb10` then preserved a
standalone reproduction through the existing `gapped_arm_boundary("G_delta")`
path. Its non-semantic artifact (SHA-256
`fc89962aadea389fa530ef82ef8fc31d02d02daa7e027c4f1b066fb8fa13d483`)
records applied-quantization maximum `0.125` against `0.05`, applied-mean error
`0.0357143879` and covariance error `0.429353714` against the `0.02` moment
limit, while the raw multiset difference and fixed-point count remain zero and
keys/non-summary rows remain bit-exact. The original raise-before-write defect
and the reproducible replacement artifact are both preserved. They are technical
development evidence, not a production result or authorizing semantic artifact.
No frozen limit is changed in response.

Independent review concluded that a whole-assay eager backend is the smallest
testable revision that removes the observed arm-correlated SDPA query-shape
effect without loosening a threshold. The alternative of forcing every
post-system token through q=1 SDPA calls was rejected before semantic outcomes
because it changes the source schedule, greatly increases dispatch count, and
does not preserve the ordinary large-prefill serving trajectory it was meant to
represent.

## 2. Frozen backend and version identity

Every subject-model forward in this design must use Hugging Face
`attn_implementation="eager"`. This includes native conversation rendering,
summary generation, all source capture and replay, wrong-history construction,
fresh/destination construction, calibration, tail recomputation, target scoring,
technical fixtures, and continuations. Mixing an eager source with an SDPA
destination, or vice versa, is prohibited.

The requested backend string alone is not evidence. Immediately after loading,
the apparatus must enumerate every one of the 48 decoder attention modules and
assert that its resolved implementation is eager. The ordered per-layer record
must include layer index, module class, resolved implementation, and any model-
or module-level implementation fields used to resolve it. The complete ordered
record and its SHA-256 are part of the run fingerprint, manifest, ladder, and
production-gate artifact. A missing, ambiguous, non-eager, or heterogeneous
layer fails closed.

Artifacts governed by all four additive amendments use exactly:

- `amendment_id = COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4`
- `design_id = coherent-state-gapped-v4`
- `schema = 2`

No v1–v3 ladder or production artifact can authorize v4. The checkpoint,
revision, bf16 dtype, geometry, tokenizer, donor map, frozen case order, visible
text, logical-position design, summary-only intervention, and all applicable
Amendments 1–3 requirements remain unchanged.

## 3. Arms, estimands, and retirement of `G_delta`

The v4 scored arms are exactly:

- `A_full`
- `G_fresh`
- `G_correct`
- `G_wrong`
- `G_Vcorrect`
- `G_Kcorrect`

`G_delta` is retired, and the exploratory `GCD = G_correct-G_delta` contrast is
removed. The v3 implementation, tests, development observations, and any prior
artifacts are retained rather than rewritten or deleted. Retirement is more
conservative than changing its bf16 moment or quantization limits, and it does
not remove a confirmatory control: `G_wrong` remains the exact-position,
content-specific negative control.

The unchanged co-primary intersection remains:

```text
GF = mean_i(Y_i,G_correct - Y_i,G_fresh)
GW = mean_i(Y_i,G_correct - Y_i,G_wrong)
```

Both are confirmatory only if every v4 technical gate below passes on the exact
30B bf16 eager subject before any semantic outcome. In particular, GF remains
co-primary only under a passing exact-model schedule-equivalence gate; the v3
SDPA execution did not authorize GF. The endpoint remains N=12, and Amendment
3's no-efficacy-at-N=6 rule remains in force.

The exploratory contrasts are `G_Vcorrect-G_fresh`,
`G_Kcorrect-G_fresh`, and `A_full-G_fresh`. No replacement exploratory arm or
contrast may be selected after inspecting outcomes.

## 4. Frozen schedule-equivalence fixtures

All fixtures use one frozen pool of ordinary, non-special token IDs obtained by
tokenizing the literal ASCII string `alpha beta gamma delta epsilon` with the
production tokenizer. The required pool is exactly
`[7141, 13440, 21619, 9477, 31204]`; a tokenizer mismatch fails closed. The pool
is cycled in order to reach the declared length `L`; it is never sampled or
selected from model outputs. The fixed selected-token margin is token `362`
(` A`) minus token `425` (` B`), and the common q=1 continuation token is the
first pool token, `7141` (`alpha`). The implementation artifact must persist the
exact literal, complete pool, margin, and continuation token IDs and decoded
text, special-token exclusion check, pool SHA-256, and the complete constructed
token-ID SHA-256 for each length. These values are implementation provenance
fixed before the paid attempt, not adaptable inputs.

### 4.1 Contiguous-position fixtures

For each row, compare the reference schedule with the alternative schedule on
identical token IDs at identical logical and physical positions:

| `L` | Reference query partition | Alternative query partition |
|---:|---|---|
| 5 | `[5]` | `[2, 3]` |
| 64 | `[64]` | `[32, 32]` |
| 900 | `[900]` | `[32, 868]` |
| 4096 | `[4096]` | `[32, 4064]` |
| 4097 | `[4096, 1]` | `[32, 4065]` |
| 8193 | `[4096, 4096, 1]` | `[32, 4096, 4065]` |

The partitions deliberately straddle the apparatus's 4096-token prefill chunk
boundary and include the 900-token summary cap and a long-prefix scale. They are
technical fixtures, not sampled conversation lengths.

For every row, persist and require at most `5e-4` for:

1. maximum absolute K and V difference at every layer over all aligned rows;
2. maximum absolute final-token logits difference;
3. absolute shift in the frozen token-362-minus-token-425 log-probability margin;
   and
4. after appending the same fixed q=1 continuation token to both caches, maximum
   absolute continuation-logit difference and every-layer maximum absolute
   difference in the newly appended K and V row.

The artifact records per-layer values as well as each global maximum. Top-1
agreement, KL, and total variation may be recorded diagnostically but cannot
substitute for any required maximum or margin comparison.

### 4.2 Logical-gap fixture

Use exactly 64 physical tokens from the same frozen pool. Tokens 0–31 have
logical positions `0..31`; tokens 32–63 have logical positions `8192..8223`.
Physical `cache_position` is contiguous `0..63` in both executions.

Compare:

- reference: a 32-token first island followed by the 32-token second island in
  one query call, query partition `[32, 32]`;
- alternative: the same first island followed by the same second-island tokens
  in 32 q=1 calls, query partition `[32, 1 × 32]`.

The two executions must satisfy the same every-layer K/V, final-logit, fixed-
margin, and common-q=1-continuation requirements as section 4.1, each at the
unchanged `5e-4` limit. The artifact must additionally prove identical token
IDs, identical logical positions, contiguous physical storage positions, full
attention over all physically prior rows, and rejection of logical positions
mistakenly supplied as `cache_position`.

## 5. Unchanged identity limits

The `5e-4` limit applies only to the one-shot/split execution comparisons in
section 4. The following existing identity limits remain `1e-4` and are not
recalibrated:

- generated source-of-record versus independent same-token replay, including
  token log-probabilities and every-layer summary K/V;
- snapshot/rebuild continuation identity;
- automatic masking versus an independently constructed explicit 4D physical
  causal mask;
- future-token mutation invariance for earlier logits and earlier appended K/V;
- fresh self-replacement and alpha-zero/no-op paths; and
- all other Amendment-1 identity checks already frozen at `1e-4`.

Declared bit copies remain bit-exact: inserted `G_correct` and `G_wrong` summary
rows must hash-match their source rows. Neither `5e-4` nor `1e-4` permits a hash
mismatch, dtype conversion, or key rotation on a confirmatory arm.

## 6. Exact pre-semantic gate order

The v4 apparatus runs the following order. A later stage cannot authorize or
repair an earlier failure.

1. **Open the durable sink.** Atomically create the unique technical artifact
   with status `RUNNING`, exact design/amendment IDs, code provenance, intended
   output path, thresholds, fixture declarations, and model request before model
   execution.
2. **Model and backend provenance.** Assert the exact checkpoint revision,
   bf16 live parameters, geometry, CUDA residency, tokenizer/chat-template
   hashes, clean non-null commit, and the eager implementation for every
   attention layer. Persist the per-layer backend fingerprint.
3. **Frozen schedule fixtures.** Run section 4.1 in its table order, then the
   section 4.2 logical-gap fixture. Persist every measurement before judging the
   aggregate stage.
4. **Replay and rebuild identities.** Run generated/source-of-record versus
   independent one-token replay, then snapshot/rebuild continuation, each at
   `1e-4`.
5. **Causal-mask identities.** Run automatic versus independent explicit 4D
   physical masking, then future-token mutation invariance, each at `1e-4`.
6. **Position and structure gates.** Assert system-island equality,
   request/header equality, island non-overlap, common summary start `P_i`,
   common downstream positions, contiguous physical storage, logical-gap-only
   `position_ids`, and the wrong-position failure injection.
7. **Intervention and propagation gates.** Assert bit-exact correct/wrong row
   insertion, no key movement, per-arm fork at the summary boundary, pre-tail
   hashes showing only declared summary-row changes, identical tail inputs with
   tail recomputed in every arm, causal tail sensitivity, and rejection of a
   reused fresh-precomputed tail.
8. **Frozen-control validation.** Revalidate Amendment 2's two unique
   calibration variants and exact-length construction, then Amendment 3's twelve
   disjoint external donors with the production tokenizer and complete hashes.
9. **Terminal technical decision.** Atomically mark the technical artifact PASS
   or FAIL, write its final hash and a linked terminal manifest/failure record,
   and only then exit the technical process. No calibration margin, `A_full`,
   natural target, or treatment arm is scored in this technical attempt.

The 0.6B ladder must execute the same bf16 eager backend, fixture declarations,
ordering, thresholds, backend assertions, and applicable structure/intervention
gates. A fresh unique v4 ladder artifact, full test suite, trusted-monitor
self-test, independent code/science review, exact clean launch commit, and exact
production-tokenizer donor validation are required before the paid technical
attempt.

## 7. Persistence before failure

Every gate owns a predeclared result field initialized before its computation.
After each safely computable subcheck, the apparatus atomically persists its raw
measurements, per-layer values, threshold, boolean result, exception state, and
input hashes before it may raise. It must accumulate all later diagnostics that
remain safe and scientifically interpretable after an individual subcheck fails;
it must not continue into calibration or semantic scoring, nor run a dependent
check whose prerequisites failed. The terminal aggregate failure is raised only
after those safe diagnostics and the linked `failure.json` have been persisted.

This rule applies to PASS and FAIL, including fixture, backend, mask, replay,
position, bit-copy, calibration-construction, donor, and retired diagnostic
paths. No evidence may exist only in stdout or an exception string. The v3
`G_delta` raise-before-write defect is preserved as a process finding and is not
repeated or hidden by the arm's retirement.

## 8. Paid-attempt and stopping rules

Exactly one further paid v4 **technical-only** attempt is authorized initially.
It uses the exact 30B revision, bf16, eager backend, and exact clean reviewed
commit. It may not score or reveal any semantic outcome, even if all gates pass
in the loaded process.

- If any backend assertion, section-4 fixture maximum or margin, `1e-4`
  identity, structure, intervention, calibration-construction, donor, artifact,
  wrong-model, or spend gate fails, stop the v4 30B line. Preserve and report the
  complete failure; do not raise a tolerance, alter a fixture, switch to the q=1
  SDPA candidate, demote GF after seeing outcomes, or buy another technical try
  under this amendment.
- If every gate passes, the PASS artifact may authorize a later semantic run of
  the unchanged v4 apparatus, subject to the existing budget, lifecycle,
  6→12, artifact, render, and regime rules. GF and GW then remain the frozen
  co-primary intersection. A technical PASS is not evidence that either semantic
  effect is positive.

Any software or infrastructure failure that prevents the declared fixtures from
being evaluated is a preserved failed attempt, not permission to infer a PASS.
A genuinely non-scientific launch failure may be handled only under the existing
reliability policy; it cannot change the one completed exact-model technical
evaluation allowed here.

## 9. Claim scope and required disclosure

This backend choice was made after observing the v3 **technical** SDPA failure and
the local backend diagnostic, but before any 30B semantic outcome. It must be
described as a post-preregistration, pre-outcome apparatus amendment, not as part
of the original frozen plan.

If the semantic run is authorized and positive, the licensed claim is limited to
the exact Qwen3-30B-A3B-Instruct-2507 revision in bf16 under the frozen Hugging
Face eager-attention execution and position-preserving compaction layout. It does
not establish the effect under default SDPA, FlashAttention, packed/reset
positions, another serving stack, another checkpoint, or a real coding agent.
The failed 30B SDPA fixture and the local 0.6B SDPA/eager diagnostic must be
reported alongside the eager result. The evidence supports an association with
the tested backend/schedule paths; it does not by itself establish which kernel
operation caused the divergence.

If v4 fails technically, the result is an apparatus/backend boundary, not a null
semantic effect. If v4 passes technically but the semantic intersection does not
clear, the original bounded inconclusive/null interpretation rules apply. No
technical diagnostic may be promoted into evidence for a history-specific
semantic channel.
