# C10 schedule-origin diagnostic result

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11

## Result

The preregistered diagnostic completed with the non-authorizing classification
`QUERY_SHAPE_ROUNDING`.

Artifact:

`results/c10_schedule_origin/c10_schedule_origin_Qwen3-0.6B_20260711T174844462255Z.json`

Payload SHA-256:

`3c91f3d8c8a587d0ed9be7b181718e43a4f3a5004682294907da3a5d9d45a55a`

The artifact is sealed, reports `status=COMPLETE` and `authorizing=false`, and
binds the exact model/tokenizer revisions, CPU bf16 eager backend at every
layer, source hashes, script hash, thread environment, and launch commit
`cfde9bcc13f90261e93d3c5348f2cb75e31e7608`.

## Frozen branches

- **A:** c10 rows 0–22 in one 23-token query.
- **B:** c10 rows 0–4095 in one 4,096-token query.
- **C:** the same 4,096-token shape as B, with c10 rows 0–22 unchanged and
  causally future rows 23–4095 replaced by committed c02 tokens.

Each branch ran twice. Complete cache snapshots and logits were bit-identical
within A, B, and C.

## Observed comparisons

| Comparison | Result | K max | V max | First divergence |
|---|---:|---:|---:|---|
| A vs B, protected rows 0–22 | not bit-exact | 0.75 | 0.875 | layer 1, row 22, K |
| B vs C, protected rows 0–22 | bit-exact | 0.0 | 0.0 | none |
| B vs C, changed rows after 22 | not bit-exact | 374.0 | 118.5 | layer 0, row 23, K |

Layer-0 A/B rows were exact. Thus identical tokens and positions first changed
only after attention execution, not at embedding, projection, position, or
initial cache construction. B and C used equal tensor shapes and different
causally masked future content; their protected earlier rows remained exact at
all 28 layers. The positive control verified that the changed tail produced
large downstream cache differences beginning exactly at the first changed row.

## Licensed interpretation

On this pinned Qwen3-0.6B CPU bf16 eager configuration, processing the same
causal prefix as a short query or as the leading rows of a much larger query
changes deterministic arithmetic. Causally future token content did not affect
the protected rows. This is consistent with query/block-shape-dependent bf16
matrix execution and rules out the tested future-token/mask-leak explanation.

This does **not** establish the precise low-level kernel reduction or tiling
path, generalize to the exact 30B/A100 backend, or make the resulting numerical
differences negligible at the semantic estimand. It also does not rescue v10:
the frozen schedule-equivalence premise remains false and the v10 release
conjunction remains impossible.

The source is a realistic generated-conversation fixture from
`data/synthetic/c10.json`, not an organic human or live subject-native
conversation. The preregistration's occasional shorthand “natural” should be
read as corrected by this provenance statement.

## Consequence for the replacement design

The next experiment must not use equality between ordinary long chunks and
turn-aligned replay as a gate. It should prospectively define:

- `P = turn_aligned_replay` from every exact canonical message boundary;
- `O = ordinary_4096` over the identical prefix;
- `D = gapped_destination_canonical` for the compacted boundary.

Within-schedule repeats and generated-versus-forced summary replay must be
deterministic. P and O are distinct repeated conditions whose arm contrasts and
interaction are measured. A semantic effect may be claimed only for the exact
schedule on which it was observed; schedule robustness requires prospective
support in both.

The completed causal-mask control removes the diagnostic blocker on building
that replacement apparatus. It does not authorize semantic execution before a
new clean paired corpus, decoded reviews, mechanical validator, preregistration,
and build ladder exist.
