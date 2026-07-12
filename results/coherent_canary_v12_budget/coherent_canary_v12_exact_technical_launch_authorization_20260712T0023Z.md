# V12 exact technical launch authorization

**Recorded by:** Sol — GPT-5.6 Sol, extra-high reasoning

**Time:** 2026-07-12T00:23Z

**Scope:** exact Qwen3-30B technical gate only

**Paid v12 pod compute observed before launch:** `$0`

## Frozen apparatus

- apparatus commit: `ec5f118414e877dd947e589614f719b9df7754d0`
- authorization commit: `37459cffc3956505b02889a2955a04aa3da1d5c6`
- sealed inventory: `598fc2f5ae90619936153a767d8b9fe87b8842190eba266d322737dbb1400787`
- focused static gate: `162/162` passed

## Local evidence required by the reviewed amendment

The complete local raw diagnostic is
`results/coherent_canary_v12_technical/coherent-canary-v12-technical_local-apparatus_20260712T002004823813Z.json`
(SHA-256 `5e7623b76abcefed9b494eca1db8b9cecb885cd2db82b1c6948666ceefe44ed5`).
Its independent report is
`results/coherent_canary_validation/coherent_canary_v12_technical_validation_local-apparatus_20260712T002212Z.json`
(SHA-256 `aa429e6217b1710cb7c3d5c79f6b8d053b7a364ac3d8c7033c6c833ac18dde1d`).

Observed conditions:

- generated/forced equivalence passed over both 64-token repeats;
- the two complete branch records were identical;
- strict normal EOS was the sole validator failure;
- deterministic role-native replay passed;
- deterministic fresh replay passed;
- all nine fresh self-replacement cells passed;
- the bidirectional path control passed at the first valid frozen setting,
  two bf16 ULPs in each direction;
- natural calibration was structurally valid and `ADVERSE`, which is reported
  rather than a plumbing failure under the preregistration.

Claude Fable 5 reviewed this exact advancement rule in
`notes/2026071175-fable-v12-local-cap-amendment-review.md`, found no blocker, and
returned `GO`.  The local technical report remains `FAIL`, is not semantic-release
eligible, and is not reclassified.

## Budget and stopping boundary

The owner budget clarification at
`results/coherent_canary_v12_budget/coherent_canary_v12_owner_budget_clarification_20260711T2349Z.md`
separates subsidized Claude CLI estimates from actual pod charges.  This launch
uses the initial `$2` actual-pod tranche.  The overall actual v12 pod ceiling is
`$8`, including provisioning and failed starts, but this authorization extends
only through the exact technical gate and its reported natural calibration.

The exact runner must use the original frozen identity fixture, 64-token cap,
model ID and revision, bf16 eager backend, and strict validator.  Any exact
normal-stop, equivalence, deterministic-repeat, self-replacement, path-control,
binding, or structural-validation failure stops advancement.  A successful
exact technical report authorizes consideration of exact Phase A only; this note
does not authorize Phase A, treatment, semantic release, or a claim.
