# Schedule-robust ValueGraft estimand design audit

**Author:** Lagrange (`gpt-5.6-sol-xhigh`)

**Date:** 2026-07-11

**Status:** Independent, read-only design audit. This note records observed facts,
inferences, and a proposed experiment. It is not a preregistration and reports no
new model outcome.

## Executive conclusion

The schedule-origin diagnostic should remain mandatory evidence for causal and
construction validity, but it should not act as a project-wide mutex. While the
single local model process runs serially, agents can freeze the next design,
implement and test branch-independent schedule plumbing, define persistence and
analysis schemas, and prepare the paid apparatus. Only semantic interpretation,
branch-dependent methodological choices, and paid fan-out need wait for the
diagnostic result.

The smallest direct local test needs five compacted arms on each of the two frozen
natural cases, `c10` and `c02`: fresh `F`; correct-history state under a true
turn-aligned replay schedule `C_P`; the same correct prefix and summary tokens under
ordinary 4096-chunk prefill `C_O`; and exact-length wrong-history counterparts
`W_P` and `W_O`. It should score the already frozen first referent and first sense
plants and directly report the schedule interactions in `C-F` and `C-W`. It must
not model schedule sensitivity as independent, centered, zero-mean, or
common-mode noise.

One important correction is required before freezing this test. The existing
schedule labeled `message_block` or `message-aligned`, including c10 widths
`[23,4096,4096,92,123]`, is not message-by-message. It is a coarse three-conceptual-
block schedule: system, the complete 8,284-token concatenated history, and the
request/header. The history block is then mechanically split at 4096. A defensible
canonical schedule must be newly derived from every actual message boundary and
called **turn-aligned replay**, not “live-session state.” The historical assistant
messages were imported text rather than tokens generated live by this subject, so
even a true turn-aligned forced replay is not the state the subject would have
written during a native conversation.

## Epistemic labels used below

- **Observed** means directly read from committed code, documents, or artifacts.
- **Inferred** means a conclusion derived from those observed facts.
- **Proposed** means a design recommendation that has not been executed.

## 1. What the existing schedules actually are

### Observed

The schedule constructor scans every `<|im_start|>` boundary, but then discards the
internal history boundaries when constructing the evaluated partition. It forms
exactly three conceptual slices:

1. `correct[:system_end]`;
2. `correct[system_end:request_header_start]`, the entire concatenated history;
3. `correct[request_header_start:]`, the summary request and assistant header.

It then independently 4096-chunks each of those three widths. This is literal in
[`_case_schedule_layout`](../src/l_coherent_state_hf.py#L674-L721). The independent
harvester reconstructs the same three-key loop—`system`, `history`,
`request_header`—rather than one block per message
([`validate_coherent_harvest.py`](../scripts/validate_coherent_harvest.py#L2439-L2458)).
Amendment 5 also defines the schedule explicitly as system, then the complete
history, then request/header
([`COHERENT-STATE-PREREGISTRATION-AMENDMENT-5.md`](../COHERENT-STATE-PREREGISTRATION-AMENDMENT-5.md#L135-L159)).

The committed c10 artifact records:

- 8,430 prefix tokens;
- conceptual widths `system=23`, `history=8284`, `request_header=123`;
- ordinary schedule `O=[4096,4096,238]`;
- coarse three-block schedule `B=[23,4096,4096,92,123]`;
- 47 recorded `<|im_start|>` positions.

The machine evidence is in the one-line committed
[`c10 schedule sidecar`](../results/coherent_state_ladder/coherent_state_ladder_gapped_v10_Qwen3-0.6B_20260711T123246Z__stage_committed_case_schedule_fixtures.json).
Recomputing calls from its frozen message starts gives the true turn-aligned replay
schedule:

```text
P = [23, 59, 350, 48, 383, 44, 362, 63, 402, 58, 225, 54, 413,
     39, 405, 63, 422, 53, 424, 57, 401, 45, 411, 52, 418, 60,
     417, 77, 414, 32, 396, 62, 408, 46, 190, 22, 23, 40, 16,
     25, 69, 29, 255, 30, 422, 123]
```

These 46 calls cover the same 8,430 tokens. No individual turn exceeds 424 tokens.
`P` therefore differs materially from the five-call `B` schedule.

The current summary-capture functions do not accept an explicit partition.
`capture_generated_summary`, `capture_forced_summary`, and
`capture_forced_prefix_ids` each pass the whole prefix to the shared ordinary
`prefill` path
([`coherent_state_runtime.py`](../src/coherent_state_runtime.py#L221-L290)).
Consequently, existing capture APIs cannot truthfully produce a `C_P`, `C_B`, or
explicit `C_O` artifact with independently validated schedule provenance.

The fresh gapped destination is different in kind. It has compact physical storage
but discontiguous logical positions. Its implementation correctly makes one system
call and one request call, then forces the summary stepwise
([`build_gapped_fresh_boundary`](../src/coherent_state_runtime.py#L371-L415)).

### Inferred

The label `message-aligned` is false for `B`. The implementation is better named
`coarse_system_history_request` or `three_block_history_chunked`.

The fact that the code preserved all message-start positions does not make the
evaluated schedule message-aligned. Those starts were provenance evidence; only
the system and final request boundaries controlled actual forward calls.

A fresh alternative `F_O` must not be invented by packing its logically separated
system and request islands into an ordinary contiguous logical stream. That would
change the position-preserving layout rather than only its call schedule. The
scientific comparison should hold the one valid fresh destination `F` fixed and
vary the correct and wrong **source** schedules.

Turn-aligned `P` is still a replay schedule, not a native live-session cache. A
live system would normally prefill user turns and generate assistant replies
token-by-token. The committed conversation bodies were not generated by this
subject along that path. Calling `P` “the state the model actually wrote” would
therefore overclaim provenance.

## 2. Proposed smallest direct `$0` test

### 2.1 Purpose

**Proposed:** Directly measure whether source prefill schedule changes the actual
ValueGraft contrasts on the two natural cases that are already on the critical
path. This is an estimand-sensitivity diagnostic, not evidence that a semantic
channel exists and not an estimate of the 30B/A100 numerical floor.

### 2.2 Frozen reusable fixtures

**Observed:** The case order starts with `c10,c02`; their exact wrong donors are
`c13,c14`; and the frozen primary categories are referent and sense
([`coherent_state_cases.py`](../src/coherent_state_cases.py#L19-L31)). The case
constructor selects the first plant in each category
([`coherent_state_cases.py`](../src/coherent_state_cases.py#L57-L71)), while the
target loader requires one committed correct/counterfactual pair for every frozen
plant ([`coherent_state_cases.py`](../src/coherent_state_cases.py#L155-L189)).

**Proposed:** Reuse without alteration:

- `data/synthetic/c10.json` and `data/synthetic/c02.json`;
- donor conversations `c13` and `c14`;
- the exact production tokenizer, template, summary request, subject revision,
  dtype, eager backend, plant selection, targets, logical positions, and destination
  construction already bound by v10;
- all committed c10/c02 schedule evidence as diagnostic provenance, but no K/V
  tensor from it—the sidecar contains no reusable semantic summary state.

### 2.3 Prospective schedules

**Proposed:** Freeze three names unambiguously:

- `P = turn_aligned_replay`: system as one call; each historical message's exact
  canonical token block as one call, 4096-chunked only if that individual message
  exceeds 4096; final summary request plus assistant header as one call.
- `O = ordinary_4096`: consecutive calls of at most 4096 over the complete
  contiguous source prefix.
- `D = gapped_destination_canonical`: the existing system/request two-island
  fresh construction followed by q=1 summary forcing. `D` never becomes a packed
  `O` stream.

Retain `B = coarse_system_history_request` only as the already observed c10/c02
diagnostic schedule. It need not become a semantic arm. If inexpensive, `C_B` may
be recorded as a secondary bridge to the old artifact, but it must not replace
either prospectively meaningful `P` or `O` and must not enter the primary rule.

### 2.4 Newly generated versus forced/reused state

**Proposed:** For each case:

1. Generate exactly one greedy summary after the correct source under `P`.
   Preserve its exact incremental `C_P` summary rows as source of record.
2. Independently force the same summary IDs after the same correct `P` prefix and
   require matched-schedule generated/replay identity.
3. Force those same IDs after the identical correct prefix under `O`, producing
   `C_O`.
4. Construct the existing exact-length wrong prefix, which changes only declared
   evicted-content slots, then force the same summary IDs under `P` and `O`,
   producing `W_P` and `W_O`.
5. Force those same summary IDs through the single canonical gapped destination
   `D`, producing `F` and the common immutable branch boundary.
6. Construct `A_full` under `P` only as a competence/headroom reference. It is not
   part of the schedule interaction.

No additional conversation render, donor, target, summary text, V-only arm,
K-only arm, or derangement arm is needed for this narrow test.

### 2.5 Scored arms and outcomes

**Proposed:** Score five compacted arms per case:

| Arm | Inserted summary K/V source | Destination and continuation |
|---|---|---|
| `F` | Fresh same-token rows from `D` | Fixed `D`; common tail/probe |
| `C_P` | Correct history under `P` | Identical to `F` |
| `C_O` | Same correct prefix/IDs under `O` | Identical to `F` |
| `W_P` | Exact-length wrong history under `P` | Identical to `F` |
| `W_O` | Same wrong prefix/IDs under `O` | Identical to `F` |

The existing scorer's per-plant outcome is the correct-target mean token log
probability minus its counterfactual-target mean token log probability
([`score_target` and `score_arm`](../src/coherent_state_runtime.py#L680-L745)).
Call this margin `Y_{i,p}(arm)` for case `i` and plant `p`.

Compute and persist every case-by-plant cell:

```text
GF_P = Y(C_P) - Y(F)
GF_O = Y(C_O) - Y(F)

GW_P = Y(C_P) - Y(W_P)
GW_O = Y(C_O) - Y(W_O)

delta_C  = Y(C_O) - Y(C_P)
delta_W  = Y(W_O) - Y(W_P)
Delta_GF = GF_O - GF_P = delta_C
Delta_GW = GW_O - GW_P = delta_C - delta_W
```

`Delta_GF` directly measures correct-source schedule sensitivity because `F` is
fixed. `Delta_GW` directly measures the schedule-by-history-specificity
interaction because both correct and wrong sources are evaluated under both
schedules. Report signed values, absolute values, and all underlying token
log-probabilities. Conversation means may be secondary summaries but must never
replace the four individual case-by-plant cells for each contrast.

### 2.6 Why this does not assume zero-mean schedule drift

**Inferred:** No schedule term is added in quadrature, averaged away, or assumed
independent of arm content. `delta_C` and `delta_W` are measured separately and
their signed difference is the `GW` sensitivity estimand. A systematic schedule
bias is therefore visible rather than relabeled as random noise. This follows the
later correction in the coordination dialogue, which withdrew the proposed
variance-inflation model and endorsed dual-schedule inference
([`2026071156-sol-fable-execution-coordination.md`](2026071156-sol-fable-execution-coordination.md#L1384-L1435)).

## 3. Acceptance, stopping, and paid-work interpretation

### 3.1 Local technical completion rules

**Proposed:** Mark the local test `INVALID`, preserving all completed evidence,
for any of:

- unresolved construction divergence or future-token influence in the origin
  diagnostic;
- non-bit-identical required deterministic repeats;
- generated `C_P` versus forced `C_P` replay mismatch at the frozen tolerance;
- wrong-prefix length, structural-token, donor-map, or changed-slot failure;
- token, logical-position, physical-position, cache-length, lineage, inserted-row,
  or non-summary-row hash failure;
- any target mismatch, missing target token, nonfinite log probability, or
  incomplete plant/arm cell;
- failure to durably preserve the generated render and trace before semantic
  scoring.

Do not stop after c10 merely because a signed schedule interaction is large or a
contrast changes sign. c02 is the only independent second natural fixture already
available, and completing it is necessary to distinguish a c10-specific case from
a repeated pattern. No cross-case average may rescue a failed technical invariant.

### 3.2 No manufactured local equivalence claim

**Observed:** The program explicitly stated that an honest SESOI was not available
at `N<=12`
([`2026071156-sol-fable-execution-coordination.md`](2026071156-sol-fable-execution-coordination.md#L119-L129));
the later coordination dialogue rejected importing a different experiment's
dispersion and rejected treating schedule sensitivity as independent zero-mean
noise ([same dialogue](2026071156-sol-fable-execution-coordination.md#L1384-L1405)).

**Inferred:** Two cases on Qwen3-0.6B CPU cannot establish statistical equivalence
or the numerical floor of Qwen3-30B-A3B on an A100. Therefore this local test
should have no invented `0.01`, `3x`, variance-inflation, sign-consistency, or
equivalence “PASS” threshold. It ends with a complete table of exact paired
schedule interactions and a technical-validity verdict.

### 3.3 What does and does not justify 30B work

**Proposed:** A causally valid origin result plus a complete, technically valid
five-arm local factorial justifies only a bounded 30B **dual-schedule** experiment.
It does not authorize a canonical-only claim and does not establish that the 30B
estimand is robust.

The same `P/O` factorial must be part of the 30B design. The final confirmatory
claim should require the co-primary conclusion to survive both prospectively
fixed schedules: clustered confidence-interval lower bounds above zero for both
`GF` and `GW` under both `P` and `O`, with the two difference-in-differences and
their intervals reported. This schedule-robust intersection requires no claim
that schedule drift is centered or random.

The first c10/c02 30B cases may be immutable members of the frozen `N=6` set,
rather than an efficacy-peeking gate. Technical failures may stop immediately;
semantic magnitudes should follow the existing no-early-efficacy adaptation rule.

Paid work is not scientifically justified if:

- construction or future-token influence remains unresolved;
- any summary-source, replay, position, intervention, persistence, or independent-
  harvest invariant fails;
- the experiment can afford only `P` or only `O`, making the proposed robust
  intersection impossible;
- the paid apparatus silently reuses `B` while calling it per-message or live
  serving.

### 3.4 Work that should proceed in parallel

**Proposed:** While the origin result is pending, proceed with work valid under all
diagnostic branches:

- freeze this arm/schedule/outcome schema before seeing semantic outcomes;
- implement one shared validated scheduled-prefix primitive and route generated
  and forced captures through it;
- derive `P` from exact rendered message boundaries without re-tokenizing text
  fragments, and assert exact coverage/order/non-overlap;
- unit-test `P`, `O`, wrong-source equal widths, fresh `D`, lineage, resume,
  unique-output, and negative-validator cases on short fixtures;
- implement result derivation and independent recomputation from raw token
  log-probabilities;
- prepare but do not launch the 30B dual-schedule job.

The Mac's standing serial-ML rule still applies: parallel work means agent/code/
document/test progress while one model process owns compute, not two simultaneous
0.6B jobs.

## 4. Required saved-render record

**Proposed:** Each c10/c02 checkpoint must be written before scoring and contain:

- byte-identical conversation JSON plus raw/canonical hashes and case/donor IDs;
- exact model, revision, tokenizer vocabulary/template/request hashes, dtype,
  device, backend, software versions, thread settings, seed, and unique output
  path logged at process start;
- exact generated summary text, IDs, logical positions, EOS evidence, per-token
  generation log-probabilities, and generation trace;
- every `P`, `O`, and `D` call width, token range, logical/physical range, boundary
  position, and complete prefix token/position hash;
- `C_P`, replayed `C_P`, `C_O`, `W_P`, `W_O`, and `F` per-layer row hashes and
  matched-schedule replay measurements;
- exact wrong-prefix structural/content positions and donor replacement evidence;
- destination boundary, inserted-span, non-summary-span, completed-tail, and
  branch-lineage hashes;
- every correct and counterfactual target token, per-token log probability,
  plant margin, arm margin, and derived signed contrast/interaction;
- terminal technical and scientific-completion status with failure evidence.

If raw K/V tensors exceed the repository artifact limit, use only the already
reviewed conditional replay waiver: the live `C_P` source remains authoritative,
and an independently reconstructed substitute is permitted only after exact
matched-schedule replay evidence. Never silently downgrade “saved exact state” to
hashes without naming the waiver.

## 5. Inventory of prior coarse-schedule mislabeling

This inventory distinguishes definite false equivalences from looser terminology.
It does not propose editing history; later documents should correct the labels
additively.

### Definite or consequential conflations

| Location | Existing wording | Why it is misleading |
|---|---|---|
| [`COHERENT-STATE-PREREGISTRATION-AMENDMENT-5.md`](../COHERENT-STATE-PREREGISTRATION-AMENDMENT-5.md#L153-L159) | “message-block production partition,” defined as system, then whole history, then request/header | Calls coarse `B` a production message-block schedule even though it ignores every internal message boundary. |
| [`src/l_coherent_state_hf.py`](../src/l_coherent_state_hf.py#L674-L741) | “exact production message blocks,” `message_block_resolved_call_widths`, and “ordinary-vs-message-block” | The actual computation creates only three conceptual widths. The nearby saved `message_start_positions` are not used to partition the forward calls. |
| [`src/l_coherent_state_hf.py`](../src/l_coherent_state_hf.py#L865-L873) | “production ordinary chunking” versus “exact system/history/request message-block chunking” | “Exact” applies to three broad regions, not messages; “production” is unsupported as a live-serving claim. |
| [`scripts/validate_coherent_harvest.py`](../scripts/validate_coherent_harvest.py#L2439-L2458) | reconstructs `message_block` by iterating only system/history/request | The independent validator faithfully validates `B`, but its field name reinforces the false message-level interpretation. |
| [`COHERENT-STATE-SCHEDULE-ORIGIN-DIAGNOSTIC-PREREGISTRATION.md`](../COHERENT-STATE-SCHEDULE-ORIGIN-DIAGNOSTIC-PREREGISTRATION.md#L14-L17) | calls `[23,4096,4096,92,123]` “message-aligned widths” | The literal widths are `B`, not true message-aligned `P`. |
| [`COHERENT-STATE-SCHEDULE-ORIGIN-DIAGNOSTIC-PREREGISTRATION.md`](../COHERENT-STATE-SCHEDULE-ORIGIN-DIAGNOSTIC-PREREGISTRATION.md#L106-L112) | proposes a “canonical message-aligned schedule” without distinguishing it from the preceding `B` label | A new preregistration must state whether this means true `P` or the old coarse `B`; using `B` would not implement the stated redesign. |
| [`DECISIONS.md`](../DECISIONS.md#L423-L432) | calls `[23,4096,4096,92,123]` “message-aligned” | Repeats the false equivalence in the main methodology decision record. |
| [`notes/2026071175-sol-schedule-discontinuity-and-claim-boundary.md`](2026071175-sol-schedule-discontinuity-and-claim-boundary.md#L31-L35) | labels the same widths “message-aligned schedule” | Repeats the false label in a scientific interpretation note. |
| [`notes/2026071176-carver-schedule-sensitivity-forensic-audit.md`](2026071176-carver-schedule-sensitivity-forensic-audit.md#L76-L77) | labels the c10 alternative “message-aligned” | Repeats the artifact label rather than the code's literal three-region construction. |
| [`notes/2026071174-fable-c10-schedule-failure-design-review.md`](2026071174-fable-c10-schedule-failure-design-review.md#L14-L19) and [lines 92–104](2026071174-fable-c10-schedule-failure-design-review.md#L92-L104) | calls `B` “message-aligned” and the failure “literally” ordinary versus message-aligned | This conflicts with the same review's later, correct definition of a new message-by-message schedule. |
| [`notes/2026071174-fable-c10-schedule-failure-design-review.md`](2026071174-fable-c10-schedule-failure-design-review.md#L228-L240) | defines canonical “message-aligned serving schedule” as message-by-message, then contrasts it with a true live cache | The recommendation itself defines `P` sensibly, but reuses the same label previously assigned to `B`. Without new schedule names, readers can wrongly believe the failed five-call fixture already tested this proposed 46-call c10 schedule. |

### Ambiguous references that should be clarified but do not explicitly claim
message-by-message execution

| Location | Existing wording | Audit reading |
|---|---|---|
| [`COHERENT-STATE-PREREGISTRATION-AMENDMENT-7.md`](../COHERENT-STATE-PREREGISTRATION-AMENDMENT-7.md#L10-L17) and [lines 60–67](../COHERENT-STATE-PREREGISTRATION-AMENDMENT-7.md#L60-L67) | “message-block partition/chunks” while also recording exact message boundaries | Mechanically consistent with `B`, but easily read as one call per recorded message. |
| [`notes/2026071156-sol-fable-execution-coordination.md`](2026071156-sol-fable-execution-coordination.md#L747-L752) | “ordinary versus message-block chunking” | Does not explicitly say per-message, but inherits the misleading field name. |
| [`notes/2026071156-sol-fable-execution-coordination.md`](2026071156-sol-fable-execution-coordination.md#L1228-L1237) and [lines 1255–1261](2026071156-sol-fable-execution-coordination.md#L1255-L1261) | reports `B` as “message-block,” once beside “production-split” | The widths are accurate; the name is not. The text does not independently prove either schedule reflects live serving. |
| [`notes/2026071163-sol-v7-v8-adversarial-review.md`](2026071163-sol-v7-v8-adversarial-review.md#L43-L55) | “actual long correct-history prefix under ordinary and message-block chunking” | “Actual” refers to tokens, not necessarily serving provenance, but the schedule label remains ambiguous. |

The remaining code occurrences are schema-name propagation rather than independent
scientific claims, but they should be renamed in any new design rather than copied:

- producer accesses of `message_block_resolved_call_widths` at
  [`src/l_coherent_state_hf.py`](../src/l_coherent_state_hf.py#L800-L812) and
  [lines 900–906](../src/l_coherent_state_hf.py#L900-L906);
- independent-harvest reconstruction/access at
  [`scripts/validate_coherent_harvest.py`](../scripts/validate_coherent_harvest.py#L785-L800),
  [lines 2454–2459](../scripts/validate_coherent_harvest.py#L2454-L2459), and
  [lines 2615–2633](../scripts/validate_coherent_harvest.py#L2615-L2633);
- technical test-fixture fields at
  [`tests/test_l_coherent_state_gapped.py`](../tests/test_l_coherent_state_gapped.py#L210-L243)
  and [`tests/test_validate_coherent_harvest_v10.py`](../tests/test_validate_coherent_harvest_v10.py#L325-L335),
  [lines 638–655](../tests/test_validate_coherent_harvest_v10.py#L638-L655),
  [lines 706–720](../tests/test_validate_coherent_harvest_v10.py#L706-L720), and
  [lines 1144–1153](../tests/test_validate_coherent_harvest_v10.py#L1144-L1153).

Those references consistently preserve the old coarse `B` schema. They do not
show that any test or validator ever derived one call per recorded message.

`STATE.md` lines 25–32 report only the two raw partitions and do not call `B`
message-aligned; that is the preferable historical wording
([`STATE.md`](../STATE.md#L25-L32)).

## Bottom line

**Observed:** v10 directly established a large deterministic disagreement between
`O` and coarse `B` on c10. It did not evaluate a true per-turn schedule and did not
measure a schedule-robust ValueGraft estimand.

**Inferred:** The old failure remains valid evidence of execution-path sensitivity,
but it cannot select `B` as a canonical serving state or be described as evidence
about a true live-session cache.

**Proposed:** Continue origin validation and branch-independent implementation in
parallel; freeze `P/O/D` explicitly; run the five-arm c10/c02 factorial; preserve
all renders and raw score evidence; and require dual-schedule inference on the
actual 30B subject before any channel claim.
