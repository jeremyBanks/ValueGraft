# Coherent summary state: Amendment 5 — attested two-process authorization

**Frozen:** 2026-07-11, after the first local gapped-v4 development ladder was
started and after its post-code/science audit, but before any execution of the
exact 30B checkpoint under eager attention. This amendment is additive. The
original preregistration, Amendments 1–4, every failed run, and every
development diagnostic remain part of the audit record.

## 1. Status and reason for this amendment

The first/current v4 ladder execution is development evidence only, regardless
of whether it ultimately passes, fails, or is interrupted. It does not authorize
a paid technical attempt or a semantic run. Review of the implemented v4 path
found authorization-contract gaps that can be closed without observing an
exact-30B eager result, changing an estimand, or loosening a limit:

- the synthetic schedule battery did not exercise the exact token streams and
  production partition boundaries of all twelve committed cases;
- the declared maximum technical logical position omitted the q=1 continuation
  at logical position `8224` after the gapped fixture's last token at `8223`;
- eager-backend attestation must inspect the resolved internal field as well as
  the public implementation field at every configuration level;
- measurements, persistence, validation, and safe post-failure continuation
  need one exhaustive, machine-checkable lifecycle;
- donor validation must be recomputed inside the exact-model gate rather than
  trusted from an earlier artifact;
- harvest must independently recompute numeric, coverage, and terminal-file
  integrity; and
- semantic execution must be a separate process cryptographically bound to a
  previously committed technical PASS made by an identical apparatus.

No exact-30B eager technical or semantic observation was available when these
requirements were frozen. A fresh v5 local ladder, full review, and clean launch
commit are required. No v4 ladder or review can be relabeled as v5 evidence.

## 2. Frozen v5 identity and unchanged scientific design

Artifacts governed by all five additive amendments use exactly:

- `amendment_id = COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4-5`
- `design_id = coherent-state-gapped-v5`
- `schema = 2`

The exact checkpoint remains
`Qwen/Qwen3-30B-A3B-Instruct-2507` at revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, with live bf16 parameters and
eager attention on all 48 decoder layers. The frozen case order, external-donor
map, tokenizer and chat template, request, seeds, caps, logical-position layout,
summary-only intervention, and source data remain unchanged.

The scored arms remain exactly:

- `A_full`
- `G_fresh`
- `G_correct`
- `G_wrong`
- `G_Vcorrect`
- `G_Kcorrect`

`G_delta` remains retired. The co-primary intersection remains `GF` and `GW` at
N=12, with Amendment 3's no-efficacy-at-N=6 rule unchanged. The exploratory
contrasts and every frozen interpretation rule remain those in Amendment 4.
The schedule limits remain `5e-4`; all identity limits remain `1e-4`; declared
row copies remain bit-exact. No v5 requirement changes a tolerance or chooses a
result-dependent estimand.

## 3. Exact eager-backend attestation

The requested load argument is not an attestation, and a model-level config
fallback is insufficient. Before any other model-backed gate, persist one
ordered backend record that inspects all of the following independently:

1. `model.config._attn_implementation` and
   `model.config._attn_implementation_internal`;
2. when distinct, `model.config.text_config._attn_implementation` and
   `model.config.text_config._attn_implementation_internal`; and
3. for each of the 48 ordered decoder attention modules, that module's own
   config object's `_attn_implementation` and
   `_attn_implementation_internal`.

Every inspected value must resolve unambiguously to the exact string `eager`.
Missing module config, a missing field, a value inherited only from a separate
model config, mixed values, duplicate or missing layer indices, a non-attention
module substituted into the enumeration, or anything other than 48 ordered
records fails closed. Each layer record contains its index, fully qualified
module name, module class, module-config class, both raw fields, and the resolved
value. The gate persists the complete ordered record plus its canonical JSON
SHA-256. The same record is included in the technical fingerprint and must be
recomputed after the semantic process loads its model.

Every subject-model forward remains eager, including all technical fixtures,
rendering, generation, source capture, replay, destination construction,
calibration, arm construction, tail recomputation, and scoring.

## 4. Frozen schedule-equivalence fixtures

Amendment 4's six contiguous synthetic fixtures and one synthetic logical-gap
fixture remain unchanged. The logical-gap stream occupies logical positions
`0..31` and `8192..8223`; its common q=1 continuation occupies logical position
`8224`. Therefore `max_technical_logical_position` is exactly `8224`, not
`8193`. Model context coverage must include position `8224`.

The fixed literal, token pool, token-362-minus-token-425 technical margin,
continuation token, partitions, and `5e-4` measurements in Amendment 4 remain
binding. These are execution diagnostics, not natural targets or semantic
outcomes.

### 4.1 All-twelve committed-case exact-token fixtures

Add one fixture for each case in the exact frozen order:

```text
c10, c02, c01, c04, c07, c11, c05, c09, c06, c12, c08, c03
```

Each fixture reads its committed source file at
`data/synthetic/<conversation_id>.json`. It validates the native conversation,
then constructs the correct summary source with the production
`SUMMARY_REQUEST` and the production tokenizer/chat template. The exact stream
is the output of the production generation-prefix renderer for:

```text
all committed conversation messages
+ final user SUMMARY_REQUEST
+ template-generated assistant header
```

No generated reply, generated summary, plant probe, correct target,
counterfactual target, calibration margin, or treatment-arm score may be
computed by this fixture.

For each case define the three conceptual blocks without re-tokenizing a
prefix:

1. `system = correct_prefix_ids[0:system_end]`, where `system_end` is the second
   exact `<|im_start|>` position in the correct generation prefix;
2. create the fresh generation prefix from exactly the committed system message
   plus the same final user `SUMMARY_REQUEST`; let
   `request_header_suffix = fresh_prefix_ids[system_end:]`; require it to equal
   the exact suffix of `correct_prefix_ids`; then set
   `request_header_start = len(correct_prefix_ids) - len(request_header_suffix)`;
3. `history = correct_prefix_ids[system_end:request_header_start]`; and
   `request_header = correct_prefix_ids[request_header_start:]`.

Require all three blocks to be non-empty, ordered, non-overlapping, and to cover
the prefix exactly. Require the system and request/header equalities used by the
production gapped layout. Message boundaries are found only by scanning exact
`<|im_start|>` token IDs; no standalone text re-tokenization may locate them.

Compare identical full-prefix token IDs at identical logical positions
`0..P_i-1` and identical contiguous physical positions `0..P_i-1` under:

- **ordinary production prefill:** apply the production prefill helper to the
  complete exact prefix, whose resolved forward-call widths are consecutive
  chunks of at most `4096`; and
- **message-block production partition:** invoke the same prefill helper
  separately for `system`, then `history`, then `request_header`, preserving one
  cache. Within each block the same helper's consecutive `4096`-token chunking
  applies.

Persist both the three conceptual widths and every resolved forward-call width.
Neither partition may use q=1 replay or omit the history block. For every case,
persist and require at most `5e-4` for the Amendment-4 schedule measurements:
every-layer aligned K/V maxima, final-token logits maximum, the frozen selected-
token margin shift, and—after the same frozen q=1 continuation at logical and
physical position `P_i`—continuation-logit maximum and every-layer new-row K/V
maxima. Persist all per-layer raw values and aggregate maxima.

Each case record must also contain:

- conversation ID, frozen-order position, source path, raw source-file SHA-256,
  canonical parsed-source SHA-256, and recorded author metadata;
- exact model revision, tokenizer-vocabulary hash, chat-template hash, request
  hash, complete prefix-token SHA-256, token count, and complete position-array
  SHA-256;
- `system_end`, `request_header_start`, conceptual block widths, resolved call
  partitions, boundary-token IDs, and the exact system/request equality booleans;
  and
- its threshold, every raw measurement, recomputed aggregate maximum, status,
  and failure evidence if applicable.

The aggregate fixture passes only with exact coverage of all twelve IDs in the
frozen order, twelve distinct source paths and hashes where the files differ,
no missing measurement, and twelve passing case records. A synthetic fixture
cannot substitute for a committed-case fixture, and one case cannot stand in
for another.

## 5. Exact technical gate order

The v5 technical process executes the following order. Every stage and every
predeclared subcheck is created in the durable sink before its computation.
Later evidence cannot repair an earlier failure.

1. **Open the sink.** Create the unique attempt artifact with status `RUNNING`,
   v5 identity, exact intended paths, thresholds, the full stage/subcheck
   schema, technical-only mode, model request, and static apparatus/data hashes.
2. **Static provenance.** Assert a clean non-null launch commit, exact amendment
   files and hashes, exact apparatus-file inventory and aggregate hash, exact
   model/revision request, tokenizer/chat-template hashes, input-data hashes,
   frozen order/map, and technical-only CLI mode.
3. **Load and attest the subject.** Assert exact live revision, bf16 parameters,
   CUDA residency, geometry, context coverage through logical position `8224`,
   and section 3's complete 48-layer eager record.
4. **Synthetic schedule fixtures.** Run Amendment 4 section 4.1 in frozen table
   order, followed by its logical-gap fixture and q=1 continuation at `8224`.
5. **Committed-case schedule fixtures.** Run section 4.1 of this amendment in
   exact frozen case order.
6. **Replay and rebuild identities.** Run generated source-of-record versus
   independent one-token replay, followed by live-snapshot versus rebuilt-cache
   continuation, at `1e-4`.
7. **Mask and causal identities.** Run automatic versus independently
   constructed explicit 4D physical causal masking, followed by future-token
   mutation invariance, at `1e-4`.
8. **Position and structure gates.** Assert system/request equality, island
   non-overlap, common summary starts and downstream positions, contiguous
   physical storage, logical-gap-only `position_ids`, exact-length wrong-source
   construction, and both wrong-position and altered-structure failure
   injections.
9. **Intervention and propagation gates.** Assert bit-exact selected row
   insertion, unselected/non-summary preservation, per-arm fork exactly at the
   summary boundary, fresh self-replacement, independently recomputed identical
   tails, downstream sensitivity, and rejection of a pre-tailed or reused-tail
   branch.
10. **Calibration construction.** Recompute the two distinct Amendment-2
    construction variants, exact-length/structural invariants, and their input
    hashes. Compute no calibration margin or natural target score.
11. **All-twelve donor construction.** Inside this same loaded technical gate,
    reload every target and its frozen external donor from the fingerprinted
    files and recompute section 7 below with the production tokenizer.
12. **Aggregate and terminalize.** Recompute every subcheck verdict from raw
    measurements, reject semantic fields, write PASS only if all required stages
    pass, otherwise write FAIL, then create the terminal integrity inventory and
    exit the process. Technical PASS is not a semantic result.

The local 0.6B v5 ladder executes this ordering and every applicable assertion
with bf16 eager attention. It must produce a new unique v5 artifact. A complete
test suite, trusted-monitor self-test, independent code review, independent
science review, exact production-tokenizer validation, and clean launch commit
are required after implementation and before any paid v5 technical attempt.

## 6. Measure, persist, validate, and safe continuation

Every declared stage and subcheck uses exactly these lifecycle states:
`PENDING`, `RUNNING`, `PASS`, `FAIL`, `ERROR`, or `SKIPPED_DEPENDENCY`.
`PENDING` and `RUNNING` are never terminal. `SKIPPED_DEPENDENCY` is terminal
only for an overall FAIL and must name the failed prerequisite and why execution
would not be scientifically interpretable or operationally safe.

For every subcheck the process must perform, in order:

1. **Declare:** persist its input identities, expected coverage, metric names,
   threshold and comparison operator, prerequisites, and status `PENDING`.
2. **Begin:** atomically persist status `RUNNING` before allocating or executing
   its model work.
3. **Measure:** compute only predeclared raw values. After each independently
   meaningful row, layer, fixture, or case, atomically persist the raw value,
   input hash, observed coverage, and any exception. No verdict is inferred yet.
4. **Persist:** atomically write the complete currently available raw payload to
   the unique attempt artifact and read it back successfully.
5. **Validate:** recompute finiteness, coverage, aggregates, hashes, and the
   boolean comparison from the persisted raw payload, not transient tensors.
6. **Close:** atomically persist `PASS`, `FAIL`, or `ERROR`, observed aggregate,
   threshold, validation result, completion time, and structured failure
   evidence. Read it back before moving to another stage.

An exception is a measurement, not an excuse to omit a record: persist exception
type, message, traceback, last completed unit, intended and observed coverage,
and input hashes. NaN, infinity, missing layers, missing cases, missing hashes,
or a disagreement between a stored aggregate and the aggregate recomputed from
raw values is FAIL.

After a subcheck fails, the process continues only into a predeclared check that
is independent of the failed output, uses a known-valid immutable input/cache,
cannot expose a semantic outcome, and can run without unsafe memory pressure or
state corruption. It may continue remaining independent rows within the same
fixture battery and independent static/tokenizer validations. It must mark a
dependent replay, cache reconstruction, intervention, or propagation check
`SKIPPED_DEPENDENCY` rather than using a failed or partially mutated cache. CUDA
OOM, loss of model residency, backend ambiguity, corrupted persistence, model
or tokenizer mismatch, and failed input-integrity validation prohibit further
model-backed checks; only static integrity and terminalization remain safe.

No failure path may enter rendering, summary generation for a real case,
calibration scoring, `A_full`, any treatment arm, a plant probe, a natural
target, analysis, or semantic continuation. The aggregate technical verdict is
computed only after all safe independent evidence has been closed. A terminal
artifact is immutable; resumption creates a new unique attempt rather than
changing a terminal record.

## 7. Donor validation recomputed inside the gate

The technical gate must not authorize itself by copying the earlier donor-
validation artifact. For all twelve target→donor pairs, in frozen order, it
reloads the exact target and donor files bound in the static fingerprint and
recomputes `matched_wrong_prefix_ids` with the production tokenizer and request.

For every pair persist target/donor IDs, paths, raw file hashes, recorded donor
author, subject-native=`false`, correct-prefix and wrong-prefix token counts and
hashes, complete structural/content position hashes and counts, and every
replacement's target/donor message indices, role, span, source-pool hash,
replacement hash, length, and cycle count. Require twelve unique donors,
target/donor disjointness, exact correct/wrong length equality, exact structural
token equality, at least one changed content token, no special token in a
replacement, complete non-overlapping replacement coverage of every declared
target evicted-content slot, unchanged system/request/header/retained-tail
tokens, and the exact frozen donor map. Recompute every assertion from the
persisted values before closing the stage.

This inside-gate result has its own canonical payload SHA-256 and is part of the
technical gate payload hash. An earlier external artifact may be cited as
development evidence but cannot satisfy this gate.

## 8. Terminal integrity and independent harvest validation

### 8.1 Terminal payloads and file hashes

Every terminal apparatus JSON payload other than the integrity index and
receipt contains `payload_sha256`, defined as SHA-256 over its canonical JSON
(`sort_keys=true`, compact separators, UTF-8) with only its own
`payload_sha256` field omitted. The process recomputes this value after
read-back. Mutable timestamps and paths remain in the hashed payload; nothing
else is excluded.

After all terminal payloads and the terminal manifest are immutable, write a
unique `terminal_artifact_index.json` containing the relative path, byte count,
and raw-file SHA-256 of every JSON payload required for the attempt, including
the unique gate-attempt file, canonical gate alias, manifest, and `failure.json`
when present. The index also records the exact required apparatus-payload path
set and rejects an unindexed terminal payload; the index and receipt themselves
are the only excluded envelope files. Then write `terminal_receipt.json`
containing the index filename, byte count, and raw-file SHA-256. Neither the
index nor receipt claims a self-hash. Harvest recomputes their bytes directly.
The job log's final technical marker is written only after the receipt is
durable.

### 8.2 Independent harvest checks

The harvest validator remains independent of the apparatus modules and embeds
the frozen v5 constants. It parses every JSON and independently requires:

- exact v5/model/revision/dtype/backend identities and complete terminal status;
- a complete 48-layer eager attestation, including both public and internal
  fields at model, text-config, and per-attention-module config levels;
- exact synthetic fixture coverage and maxima recomputed from finite raw
  per-layer/per-metric values;
- exactly twelve committed-case fixture records in frozen order, with exact
  file/token/position hashes, conceptual and resolved partitions, 48-layer K/V
  coverage, continuation coverage, and every stored aggregate recomputed;
- exact calibration-construction and twelve-pair donor-validation coverage,
  uniqueness, disjointness, structure/content invariants, and hashes;
- PASS iff every required raw comparison satisfies its unchanged threshold and
  every required hash/equality assertion passes; FAIL evidence otherwise;
- no forbidden semantic field, conversation checkpoint, render marker,
  calibration outcome, arm outcome, natural target, or scored analysis in a
  technical harvest;
- every payload's canonical `payload_sha256`, every raw file byte count and hash
  in the terminal index, the receipt's index byte count and hash, the exact
  required apparatus-payload path set, and no unindexed terminal payload other
  than the index and receipt envelopes; and
- a terminal technical job marker only after the receipt has validated.

A status string, stored aggregate, apparatus-produced `passes=true`, prior donor
artifact, or log message cannot substitute for these recomputations. Harvest
writes its own validation result, including raw manifest/index/receipt hashes,
before pod termination. A mismatch is a failed harvest and cannot authorize
semantics.

## 9. Separate, cryptographically bound semantic mode

The technical executable defaults to technical-only and contains no reachable
semantic continuation. Its process exits after a terminal PASS or FAIL and
receipt. It must not render conversations, generate real-case summaries,
calibrate margins, score `A_full`, construct a scored treatment arm, or write a
conversation checkpoint. A technical process still computes no semantic
outcome even when every gate passes.

Semantic execution is a separate explicit mode/process and remains disabled
unless given a prior v5 technical attestation. Before rendering or scoring, it
must verify all of the following:

1. the prior technical gate and manifest are terminal PASS/
   `TECHNICAL_COMPLETE`, independently harvest-valid, and their payload, raw
   file, index, and receipt hashes recompute exactly;
2. the entire prior technical result directory is present byte-for-byte in a
   named git commit on `trunk`; `git ls-tree` resolves every attested path to the
   bytes being verified; that result commit is an ancestor of the semantic
   launch commit;
3. the current apparatus-file inventory and aggregate SHA-256 exactly equal the
   inventory attested by the technical PASS. This inventory includes every
   source, analysis, storage, validation, job, monitor, and lifecycle file that
   can affect model loading, cache construction, technical authorization,
   rendering, arm scoring, analysis, persistence, harvest, or pod lifecycle;
4. the amendment ID and every amendment-file hash, exact checkpoint/revision,
   bf16 dtype, geometry, model/tokenizer/chat-template hashes, summary request,
   source/target/donor file hashes, frozen case order, donor map, caps, seeds,
   tolerances, arm list, and all other data/static-fingerprint fields are
   byte-for-byte identical to the technical PASS;
5. after loading, the semantic process recomputes section 3's complete backend
   record and it is byte-for-byte identical to the technical record; and
6. the semantic manifest records the technical result commit, attested relative
   paths, gate payload SHA, raw gate/manifest/index/receipt SHA values, apparatus
   aggregate SHA, and the exact equality result for every bound field.

Any mismatch, missing commit, uncommitted attestation, non-ancestor result
commit, apparatus change, data change, backend change, or unverifiable hash
fails before a render or score. Re-running a gate inside an otherwise unbound
semantic process cannot replace the committed prior PASS. A code change after
technical PASS requires a new v5 ladder, review, exact-30B technical run,
harvest, and committed PASS before semantics.

## 10. Paid-attempt and stopping rules

Exactly one initial paid v5 technical-only attempt is authorized after the new
ladder and reviews pass. It uses the exact 30B revision, bf16, eager backend,
clean reviewed apparatus, frozen fixtures, and this amendment. It produces no
semantic outcome.

- Any technical, persistence, integrity, coverage, backend, committed-case,
  donor, harvest, wrong-model, or spend failure terminates that attempt as FAIL.
  Preserve all safe evidence. Do not raise a threshold, remove a case, adapt a
  partition, or infer a semantic null.
- A fully validated technical PASS may be harvested, committed, and then used
  only through section 9's separate semantic authorization. It is evidence that
  the frozen apparatus cleared its technical gates, not evidence that GF or GW
  is positive.

The existing budget, pod lifecycle, render preservation, N=6→12, resume, and
artifact rules remain binding. A launch/infrastructure failure is handled only
under the existing reliability policy and cannot change a scientific rule.

## 11. Claim scope

Amendment 4's claim scope is unchanged. A later semantic result, if authorized,
is limited to the exact Qwen3-30B-A3B-Instruct-2507 revision in bf16 under the
frozen Hugging Face eager-attention apparatus and position-preserving layout. It
does not establish an effect under SDPA, FlashAttention, packed/reset positions,
another serving stack, another checkpoint, another task domain, or a real coding
agent. The v3 SDPA failure, local backend diagnostics, v4 development evidence,
and every v5 authorization step must be disclosed. A v5 technical failure is an
apparatus/backend boundary, not a semantic null; a technical PASS is not a
semantic effect.
