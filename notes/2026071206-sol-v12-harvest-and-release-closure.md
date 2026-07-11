# V12 harvest and release closure

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning

**Date:** 2026-07-12 (America/Toronto)

## Status

This note records a pre-outcome apparatus correction. No subject-model forward
was run and no semantic outcome was observed. The preregistration remains
`DRAFT UNDER INDEPENDENT AUDIT`; nothing here authorizes execution by itself.

## Why the first harvester draft is not acceptable

The first independent-harvester implementation genuinely reconstructs the Qwen
canonical message stream, N/P/fresh call geometry, R1/R2/R3 boundaries, target
IDs, probe-prefix token IDs, and the frozen decision formulas. Its focused
synthetic tests passed 8/8. A personal audit and an independent Codex audit then
found that those tests did not establish a production-valid release boundary:

1. The runtime serializes authoritative float32 log-probability bits with
   `struct.pack("<f", ...)`, while the harvester and its synthetic fixture used
   big-endian `">f"`. The test agreed with itself but not with production.
2. One technical, one eligibility, and one treatment receipt cannot represent
   the store's case-specific releases. A four-case run needs one reusable
   technical receipt plus four eligibility receipts and four treatment chains;
   an ambiguity extension needs two more case pairs.
3. The combined input lets a single invocation read Phase-A and treatment cells.
   That does not machine-enforce pre-treatment blindness, even if it computes
   Phase A first in Python source order.
4. Receipt checking currently trusts terminal labels and does not independently
   validate the checkpoint chain, exact fingerprint, source/review bindings,
   case identity, release receipts, result binding, or git ancestry.
5. Categorical `key_source`/`value_source` labels do not prove tensor lineage.
   The independent boundary must reconstruct selected insertion and outside-row
   preservation from lossless tensor evidence or explicitly refuse the claim.
6. Technical path-control and natural-calibration verdicts are trusted strings,
   and the current raw score record omits enough feed/position/cache evidence to
   bind a supplied log probability to the claimed immutable probe state.
7. Review validation searches serialized bytes for `"PASS"`; it must validate
   the exact two review schemas, complete coverage, packet bindings, and frozen
   verdict fields.
8. The frozen per-case/per-region deterministic placebo is absent from the raw
   coverage contract.

These are authorization blockers, not reasons to discard the useful independent
layout and formula code.

## Correct release topology

The implementation will use three physically separate, append-only evidence
phases:

### 1. Technical release

The technical runner may see only the literal identity/path/natural fixture. It
persists exact runtime provenance, repeated generated/forced identity,
self-replacement, K/V/K+V confinement, the bidirectional path trace, and the
five natural-control cells. A separate validator reads raw float bits and tensor
evidence, recomputes every gate, and promotes a technical checkpoint to terminal
`PASS` or `FAIL`. Natural calibration is reported `PASS`/`ADVERSE`; it cannot
turn a failed path gate into a pass or invalidate an otherwise passing technical
gate.

### 2. Per-case pre-treatment release

For one engineered case, the eligibility runner may execute only full-history
oracles, fresh `FF`, forced-carrier support, generation/stop checks, and the
committed mechanical/review predicates. It writes no treatment source row,
graft score, layer effect, region effect, or treatment contrast. An independent
validator recomputes the exact Phase-A predicates and commits a case-specific
`PRETREATMENT_PASS`, `ESTIMAND_INADEQUATE`, or `INVALID_TECHNICAL` receipt.

### 3. Per-case treatment release and harvest

Treatment creation requires the exact committed technical `PASS` receipt and
that case's exact committed `PRETREATMENT_PASS` receipt. Each treatment chain is
case-specific and binds its raw result. The final harvester verifies every
chain, source/review/runtime binding, tensor lineage, score feed, and formula
without importing runner/layout constructors. It rejects reserve cases unless a
separately committed four-case `AMBIGUOUS4` trigger predates their release.

The terminal engineered report is therefore a pure function of committed
technical evidence, committed per-case Phase-A evidence, and committed
case-atomic treatment artifacts. There is no combined raw file that can expose
treatment before eligibility.

## Lossless tensor evidence with bounded size

For each case, persist a safetensors artifact containing:

- the fresh destination's complete cache at the maximal R3 boundary, once;
- C and W selected rows through R3 under N;
- C and W selected rows through R3 under P.

R1 and R2 are exact prefixes of those maximal row sets. Because the fresh R3
boundary occurs before the retained tail, this artifact is bounded and much
smaller than a full long-history cache. For every arm, the runner records the
post-replacement boundary snapshot hashes. The independent validator loads the
fresh boundary, applies the declared K/V selection itself, and recomputes the
expected post-replacement hashes. This proves selected-row source identity,
unused-channel freshness, and outside-region preservation without persisting a
second complete cache for every arm. It also re-hashes every referenced
safetensors file at validation time.

The raw record must bind the exact physical destination interval, source logical
interval, layer count, dtype, shapes, and tensor names. A later continuation is
bound by its exact call/token/logical/physical trace and final immutable
probe-prefix cache hashes.

## Authoritative numerical representation

All float32 model outcomes use exactly eight lowercase hexadecimal characters
representing the four bytes produced by `struct.pack("<f", value)`. Independent
decoding uses `struct.unpack("<f", bytes.fromhex(bits))`. Decimal values are
diagnostic only. Tests require at least one known runtime-produced bit pattern so
a test fixture cannot silently choose a different byte order again.

## Required next implementation units

1. Add lossless bounded safetensors persistence and independent reconstruction
   tests.
2. Add raw per-token q=1 bit records, exact generation/feed traces, and
   path-control bit records.
3. Implement the independent technical validator and terminal technical
   receipt.
4. Implement the physically separate Phase-A runner/validator and per-case
   receipt.
5. Replace the first combined harvester schema with per-case treatment inputs,
   exact checkpoint-chain validation, review-schema validation, placebo
   coverage, and reserve-trigger enforcement.
6. Only after a final file/hash inventory is committed may the preregistration
   gain an additive `FROZEN` declaration and any local subject-model forward
   begin.

## Interpretation

This correction delays new data but increases confidence for a concrete reason:
the first independent implementation already caught a real self-consistent test
fixture error before production. Passing synthetic tests is construction
evidence; authorization requires evidence that the test boundary matches the
actual serialization, release topology, and tensor path.
