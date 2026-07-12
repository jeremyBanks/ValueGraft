# Disposition of Fable's Stage-T runner-boundary review

Date: 2026-07-12

## Decision

The central recommendation is accepted. Stage T will execute and persist the
already-outcome-seen e01 probe and one carrier-visible neutral technical-long
probe for all six arms. Their artifacts state `semantic_n: 0` and
`inferential_use: FORBIDDEN`; numerical values cannot enter eligibility,
selection, a threshold, a template decision, an estimand, or a scientific
claim. This closes the actual continuation/probe production path and makes the
arm timings representative rather than timing boundary construction alone.

The literal probe contracts now live in `src/powered_v13_technical.py`. The
e01 probe and two targets are transcribed from the sole hash-allowlisted e01
bytes and checked against those bytes. The long probe asks about `continuity`,
a word in the literal carrier that remains byte-visible in C, W, and F; its
countertarget `criteria` is visible in that same carrier. Each literal, probe
block, context-message set, input core, and combined technical case is
hash-bound. The old ambiguous `probe_or_score_present: false` field is replaced
by `production_probe_reachable: false` and `score_values_present: false`: the
planning object contains the frozen technical probe contract but no production
probe bank and no observed score.

## Corrections to the advisory review

1. The suggested technical-long filler fact is invalid for the fresh arm. The
   filler is inserted into the evicted middle and is absent from F. The adopted
   neutral fact therefore comes from the carrier, which is visible in all
   histories and destinations.
2. L1 cannot compare event-by-event q1 replay with one monolithic prefill and
   demand bit identity. Prior work found that changing call width can change
   bf16/quantized kernel numerics, and the v13 schedule deliberately makes q1
   calls. L1 will split/reconcatenate or serialize/rebuild the same cache and
   continue with the exact same event schedule and call widths. It does not
   compare different decompositions.
3. The technical store sequence requires six named arm checkpoints, but by
   itself it does not prove that an arm artifact contains a probe result. The
   reason to execute probes is the preregistered carveout plus production-path
   correctness and representative timing, not a claim that the store schema
   alone logically forces scoring.
4. An unavailable e01 VP search may still be preserved as a valid terminal
   Stage-T diagnostic rather than corrupting or abandoning the case. It blocks
   later scaled Stage A, because the preregistration separately requires
   observed e01 VP availability before scaling. It is therefore nonfatal to
   artifact integrity but fatal to progression unless a prospective amendment
   changes the design.

## Frozen minimum runner order

1. Start the independent provider watchdog immediately after allocation.
2. Verify the detached, clean Stage-T release checkout and post-commit receipt
   before model load.
3. Attest the exact subject, revision, bf16/eager runtime, tokenizer/template,
   dependency lock, GPU/driver/CUDA, topology/content sentinels, and VRAM.
4. On technical e01, run L0 deterministic same-call replay; L1 exact cache
   round-trip/null split-and-reconcat followed by the same continuation calls;
   L3 uninterrupted versus R2-stop/rebuild/same-schedule continuation; fresh
   K-only, V-only, and K+V self-replacement; and saved-bundle read-back and CUDA
   materialization identity.
5. Terminalize technical e01 in exact store order: foundation load, FF, CC, WW,
   FC, FW, VP, independently validated terminal evidence.
6. On technical-long, repeat deterministic replay and R2-stop/rebuild
   continuation at 4.5k geometry, then the same six-arm terminal sequence.
7. Harvest and hash all complete, partial, invalid, and timing artifacts; delete
   immediately; require both per-pod 404 and absence from complete active
   inventory; reconcile provider seconds/dollars and zero active pods off-host.

Every arm artifact binds exact token/logical/physical traces, boundary and
inserted/outside-row hashes, probe suffix and target IDs, float32 score bits,
greedy/forced identity evidence, wall time, and peak VRAM. Each checkpoint is
written and read back before the next model operation. A distinct validator
process recomputes the preterminal chain and emits the receipt the store checks.

Immediate cleanup follows any release, attestation, ladder, finite-value,
geometry, store, bundle, per-case deadline, import-boundary, or watchdog stop
failure. Slow timings are recorded and may reject later feasibility; they do
not retroactively invalidate an otherwise exact technical artifact. No new
terminal kind, score threshold, production input, second host, warm hold,
network volume, or runner-side competing provider clock is added.

## Status

The review is accepted with the corrections above. The literal technical probe
binding and tests are implemented locally; release, subject-loader, runner,
validator, import-closure, and exact Stage-T authorization remain required
before allocation.
