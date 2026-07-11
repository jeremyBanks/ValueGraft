# Coherent summary state: Amendment 10 — independent authorization and serial-terminal closure

**Frozen:** 2026-07-11, after independent code and scientific reviews of the
unexecuted Amendment-9 apparatus, and before any v10 30B technical or semantic
outcome. This amendment is additive. It changes no arm, tolerance, case order,
donor map, estimand, stopping rule, or claim scope.

## Why this amendment is necessary

Adversarial v9 fixtures demonstrated four remaining fail-closed defects:

1. semantic COMPLETE harvest accepted a fabricated, nonexistent prior technical
   result because it shape-checked copied authorization fields without resolving
   and independently revalidating the referenced commit/tree/attestation;
2. an N=6 COMPLETE result could carry the nonterminal `EXTEND_TO_12` decision,
   and an N=12 COMPLETE result could follow an N=6 stop;
3. bit-identical K/V digests could coexist with nonzero stored K/V replay maxima;
   and
4. the raw-snapshot waiver did not independently require the actual source to be
   a normally terminated, special-token-free incremental generation with an exact
   trace and bounded summary length.

These corrections make existing authorization, stopping, and source-generation
rules independently enforceable. They do not alter those rules.

## 1. Version identity

Artifacts governed by all ten additive amendments use exactly:

- `amendment_id = COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4-5-6-7-8-9-10`
- `design_id = coherent-state-gapped-v10`
- `schema = 2`

No v9 artifact can authorize v10.

## 2. Semantic harvest independently resolves prior technical authorization

A semantic COMPLETE artifact may not authorize itself by copying plausible fields
or producer booleans. Independent harvest must resolve the referenced technical
result commit and repository directory, require the commit to be on the attested
trunk ancestry, read the exact committed technical result bytes, revalidate its
terminal envelope and independent technical PASS, and bind the committed harvest
attestation and apparatus inventory to the semantic launch.

The semantic authorization's launch commit must equal the semantic fingerprint's
code commit. The technical result commit, run directory, gate/raw/payload hashes,
harvest path and hashes, apparatus aggregate, backend, and static fingerprint must
all match the independently resolved evidence. A nonexistent directory, fabricated
commit, uncommitted result, changed apparatus, or copied-but-unresolved PASS fails
semantic harvest.

## 3. Serial decisions distinguish terminal stop from extension

At N=6, semantic COMPLETE is valid only for the frozen terminal decisions
`STOP_TECHNICAL`, `STOP_REGIME`, or `STOP_FUTILITY`. `EXTEND_TO_12` is nonterminal:
the process must continue to the frozen twelve cases and cannot seal an N=6
COMPLETE result.

At N=12, both the persisted and independently recomputed immutable N=6 decision
must be exactly `EXTEND_TO_12`; an N=6 stop followed by additional cases is invalid.
The N=12 decision remains `FINAL_N12`. Negative regressions cover both invalid
directions.

## 4. Bit-identical replay is numerically self-consistent

When actual and replay K/V tensor digests are bit-identical, every persisted
per-layer K and V max-absolute replay difference and both aggregate K/V maxima
must equal exactly `0.0`. A nonzero value at or below the broader `1e-4` identity
tolerance is internally inconsistent with the stronger byte witness and fails
harvest. Token log-probability comparison retains its frozen `1e-4` bound unless
separately proven bit-identical.

## 5. The actual source is verified as normal incremental generation

For every semantic conversation, independent harvest requires:

- `summary.generation_trace` to be exactly equal to `sources.correct_actual.trace`;
- trace token IDs, start, end, and coverage to match the saved prefix and summary;
- the actual generation trace to have `ended_on_eos = true`, while the separate
  forced replay is labeled non-generative and has `ended_on_eos = false`;
- summary length to satisfy `0 < T < 900` under the frozen generation cap;
- no saved summary token to be a production-tokenizer special ID; and
- exact logical position arrays and contiguous physical cache-position arrays for
  the complete actual and replay prefix+summary state.

Missing or mismatched generation traces, false normal termination, cap-length
output, embedded special tokens, or altered physical positions invalidate the
raw-tensor archival waiver and stop semantic execution.

## 6. Release gate

Before any paid v10 attempt: generate and commit a fresh v10 production-tokenizer
donor artifact; complete and commit a fresh v10 0.6B bf16 eager CPU ladder; run
the full unit and monitor fault-injection suites; obtain fresh independent code,
science, and cross-family reviews on the exact clean launch commit; and push that
commit. The first paid run remains technical-only, and semantic execution remains
separately conditional on an independently resolved, committed technical PASS.
