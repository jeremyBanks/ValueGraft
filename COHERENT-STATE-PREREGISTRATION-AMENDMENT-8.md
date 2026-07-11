# Coherent summary state: Amendment 8 — composed destination schedule and snapshot provenance

**Frozen:** 2026-07-11, after independent code and science reviews of the
unexecuted Amendment-7 apparatus, and before any v8 30B technical or semantic
outcome. This amendment is additive. It changes no arm, tolerance, case order,
donor map, estimand, stopping rule, or claim scope.

## Why this amendment is necessary

The v7 reviews found four classes of evidence that the producer checked but the
independent harvest validator did not fully reconstruct: donor replacements and
special-token exclusion; the frozen calibration prefixes; intervention
sensitivity from its recorded attempts; and the static apparatus/input
fingerprint. A separate review demonstrated that persisted semantic aggregates
could be changed without independent rejection. All of those validator gaps are
closed here with counterexample regressions.

The reviews also found one uncomposed schedule risk. The exact per-render source
fixture covered the actual long source prefix, while the gapped destination tests
separately covered logical gaps and short end-to-end intervention paths. No gate
combined the actual render length, the actual compacted destination, the actual
logical-position gap, the production system/request call split, and every token
of the saved summary.

Finally, the jointly agreed plan said generation-time snapshots would be saved.
The implementation retained and used the true incremental snapshot but archived
hashes and replay evidence, not raw K/V tensor bytes. This amendment makes that
distinction explicit and replaces any implicit waiver with a stricter,
fail-closed, independently validated provenance contract.

## 1. Version identity

Artifacts governed by all eight additive amendments use exactly:

- `amendment_id = COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4-5-6-7-8`
- `design_id = coherent-state-gapped-v8`
- `schema = 2`

No v7 ladder, donor artifact, technical result, harvest attestation, or semantic
artifact can authorize v8.

## 2. Exact actual-destination schedule fixture

After a separately authorized semantic process durably saves a freshly generated
summary, but before it constructs a wrong source, arm, target, calibration, or
semantic outcome, it must run a second per-render schedule fixture:

1. reconstruct the exact compacted destination prefix from the rendered
   conversation, frozen request, saved summary text and IDs, and the exact correct
   source prefix;
2. use identical compacted prefix tokens, identical gapped logical positions,
   identical contiguous physical cache positions, and the complete saved summary
   in both branches;
3. compare the production system/request prefix split against ordinary consecutive
   chunks of at most 4096 tokens, forcing every summary token stepwise in both;
4. persist both full tokenwise log-probability traces and, for all 48 layers, the
   maximum K and V difference for every individual summary row;
5. require every tokenwise, rowwise, layerwise, and aggregate difference to be at
   most `5e-4`; and
6. persist complete tokens, positions, hashes, partitions, raw rowwise values,
   aggregates, identity, conversation ID, and PASS/FAIL before applying the verdict.

The independent validator reconstructs the two exact schedules from the saved
conversation and summary; requires the full trace and 48-by-summary-length row
coverage; recomputes all scalar aggregates; and rejects missing, reordered,
mispositioned, wrongly partitioned, non-finite, or self-inconsistent evidence.

## 3. Raw-snapshot archival waiver is conditional, not assumed

For the pinned 30B checkpoint (48 layers, 4 KV heads, head dimension 128, bf16),
one summary's K+V slice is `98,304 × T` bytes. At the frozen 900-token cap this is
`88,473,600` bytes (84.375 MiB) per source and 1012.5 MiB for twelve correct
sources; correct, wrong, and fresh slices can approach 2.97 GiB. Git LFS is not
configured and the repository rejects files larger than 4 MiB. Raw K/V tensors
therefore are explicitly **not archived**, and the paper may not claim that they
are independently reusable from the repository.

That omission is permissible only if every semantic conversation satisfies all
of the following before any scoring:

1. the generated summary text, IDs, prefix, tokenwise generation trace, and exact
   per-layer generation-time summary K/V hashes are durably saved;
2. a separate stepwise forced replay uses the identical prefix and summary IDs,
   passes the existing numeric identity bound of `1e-4`, **and has bit-identical
   per-layer K and V hashes for the complete summary slice**;
3. the identity artifact contains both actual and replay hashes, records the exact
   scoring-source materialization, and sets the raw-tensor waiver only when the
   bit-exact comparison passes;
4. an uninterrupted run records that live incremental-generation rows supplied
   scoring; a resumed run may instead use a stepwise reconstruction only after it
   bit-exactly matches the previously saved actual-generation hashes, and must label
   that materialization explicitly; and
5. each gapped arm persists component-level source and inserted-span hashes. The
   independent validator must prove the exact lineage graph: `G_fresh` takes fresh
   K/V; `G_correct` takes actual correct-source K/V; `G_wrong` takes wrong-history
   K/V; `G_Vcorrect` takes fresh K and correct-source V; and `G_Kcorrect` takes
   correct-source K and fresh V. It also requires non-summary prefix preservation,
   exact branch lengths, the full frozen arm set, and unambiguous source labels.

The tensor-hash definition is frozen as SHA-256 over, in order: the tensor dtype
string, the tuple-shaped dimension string, and the contiguous CPU tensor bytes.
This is the exact `sha256_tensor` procedure used by the apparatus. The saved
digests are deliberately non-reconstructive witnesses: because raw tensor bytes
are not archived, harvest cannot recompute the actual generation-time digest
from repository bytes. It instead independently reconstructs the exact replay
inputs, verifies every recorded hash equality and source-to-insertion lineage
edge, and rejects any internal or cross-artifact inconsistency. Generation and
forced replay share the frozen production primitives; "independent" refers to
the later harvest validator, not to a second inference implementation.

Approximate numeric equality alone does not waive raw archival. Any row-hash
mismatch, missing lineage edge, ambiguous materialization, or failed independent
validation stops semantic execution. A failure is preserved and reported; it
cannot be converted into a looser threshold after outcomes are visible.

## 4. Remaining independent reconstruction closure

The v8 independent validator additionally reconstructs and rejects tampering in:

- every donor source-derived replacement pool, cycling rule, reconstructed wrong
  prefix, and special-token exclusion;
- the exact frozen calibration messages, prefixes, positions, labels, and hashes;
- intervention sensitivity booleans from the persisted numeric attempts;
- the launch commit's apparatus paths and bytes, all eight amendment bytes, input
  inventory, model/runtime/static fingerprint bindings, and top-level duplicates;
  and
- every target mean, plant margin, arm conversation margin, conversation outcome,
  calibration detail/outcome, and cross-arm plant identity from tokenwise evidence.

A producer boolean, plausible aggregate, or terminal status cannot substitute
for these recomputations. Snapshot digests are the one intentional exception to
raw-value recomputation: under §3 they serve only as relational witnesses whose
equality and complete lineage graph are independently checked, never as evidence
that harvest regenerated unavailable tensor bytes.

## 5. Release gate

Before any paid v8 attempt: finish and commit the exact v8 production-tokenizer
donor artifact and complete v8 0.6B bf16 eager CPU ladder; run the full unit and
monitor fault-injection suites; obtain fresh independent code, science, and
cross-family reviews of the exact clean launch commit; and push that commit. The
paid run remains technical-only first. Semantic execution remains separately
conditional on a committed, independently harvest-valid technical PASS with an
unchanged apparatus inventory. A technical PASS does not relax any per-render
source schedule, destination schedule, snapshot-lineage, calibration, competence,
or stopping gate.
