# Coherent summary state: Amendment 1 — position-preserving compaction

**Frozen:** 2026-07-11, before any semantic arm outcome from the production
checkpoint existed. This file is additive. It does not edit or supersede the
audit record in `COHERENT-STATE-PREREGISTRATION.md`; where the two conflict for
the next run, this dated amendment governs.

## 1. Why an amendment is necessary

The first paid production-config attempt stopped inside its technical gate,
before `Runner` construction and before any downstream target was scored. The
preserved artifact is:

`results/coherent_state/coherent_state_Qwen3-30B-A3B-Instruct-2507_20260711T062724Z/`

It resolved the exact checkpoint and loaded the expected bf16 model on an A100,
then measured zero-position K movement as exactly `0.0`. A different diagnostic
independently prefilling the same five tokens at positions `0..4` and `37..41`
measured inverse-shifted K/V maxima `2.810546875`/`0.810546875`, above the frozen
`0.02` limit, and failed closed. The run was harvested and the pod terminated.

That diagnostic did not isolate post-RoPE K movement. It compared two complete
48-layer bf16 trajectories. Position-dependent rounding can propagate through
attention, residual streams, and MoE routing; the large V difference is decisive
because V is not rotated by the proposed surgery at all. The original failure is
valid evidence that whole-model bf16 translation equivariance cannot be assumed,
but not evidence that a stored K rotation was coded with the wrong sign or layout.

Subsequent reproducible 0.6B diagnostics are preserved under
`results/coherent_state_diagnostics/`. They show a more important limitation:
even though the CPU helper rotation and a model-native destination-K oracle had
relative K error about `0.002` and minimum row cosine `0.999993` in bf16, their
target-pair log-probability margins differed by `0.1015625` nats in the frozen
technical example. Small tensor-relative error, KL, and unchanged top-1 therefore
do not bound the selected-token margin used by this experiment. The helper already
upcasts stored bf16 K to float32 for rotation; “rotate in float32” is not an unused
rescue.

No threshold is raised to accommodate these observations. Packed/reset-position
full-K/V arms that rerotate quantized post-RoPE K are retired from confirmatory use.
Their failed artifacts and diagnostics remain reportable technical findings.

## 2. Revised question and claim boundary

The confirmatory question is now:

> When a compaction boundary removes stored tokens but preserves the logical
> position counter, does the K/V state that the model actually wrote while
> generating a summary under the correct history carry downstream-useful,
> history-specific information that an identical summary encoded at the same
> positions without that history lacks?

A positive result licenses a claim only for **position-preserving compaction**:
storage may be compacted, but serving infrastructure must decouple physical cache
length from logical RoPE positions. It does not establish that post-RoPE K can be
rerotated reliably after a client packs or resets positions. A future packed
extension may capture pre-RoPE K and apply the loaded model's native destination
rotary operator, but that is outside this run.

The model, revision, native renders, frozen order, target pairs, downstream margin,
temperature, sample ceiling, analysis method, headroom rule, and 6→12 decision rule
remain unchanged unless explicitly amended below.

## 3. Exact logical layout

For conversation `i`, let `P_i` be the natural start position of the summary that
the correct full-history source actually generates. Let `S_i` be its exact greedy
summary token IDs and `n_i = |S_i|`.

### Correct source

The correct source remains the ordinary full target native conversation, including
the retained tail, followed by the exact summary request and assistant-generation
header. Its prefix positions are `0..P_i-1`; it greedily generates `S_i` at
`P_i..P_i+n_i-1`. The actual incremental cache is the source of record.

### Position-matched wrong source

The wrong source begins from the correct source's exact prefix token stream and
position vector. Only content-token slots inside the evicted-history messages are
replaced. System tokens, role/wrapper/boundary tokens, target retained-tail tokens,
summary-request tokens, and the assistant-generation header remain byte-identical
and at identical indices.

For each target evicted message in ordinal order:

1. take the ordinary content-token IDs from the corresponding-role message in the
   already-frozen donor conversation;
2. reject special/chat-boundary IDs and require a nonempty donor pool;
3. fill the target message's `L` content slots with donor IDs
   `donor[j mod len(donor)]` for `j=0..L-1` (truncate when the donor is longer,
   cycle when shorter);
4. do not alter any target structural slot.

If the donor has fewer evicted messages of the required role, corresponding-role
donor messages cycle by ordinal. The original `WRONG_DONOR` map remains frozen.
The exact target slots, donor pools, mappings, resulting IDs, decoded diagnostic
text, and hashes are persisted before scoring.

Consequently the correct and wrong prefixes have identical token count `P_i`,
identical structural positions, identical request/header suffix, and summary start
`P_i`. The exact `S_i` IDs are counterfactually forced at those same positions.
This is an artificial exact-position unrelated-history control. It supports
“specific to the correct native history versus this exact-position donor-slot
history,” not “the planted fact alone caused the difference.”

### Fresh source and compacted destination

The compacted prefix physically stores only the target system message plus the
exact summary request and assistant-generation header. Its logical position vector
has two contiguous islands:

- system block: its original positions beginning at `0`;
- request/header block: the exact suffix positions it occupied at the end of the
  correct prefix, ending at `P_i-1`.

The omitted interval is a logical gap representing evicted history. The request /
header suffix must be exact-ID-equal to the suffix of the correct prefix, and the
two islands must not overlap.

Fresh stepwise encoding forces `S_i` at `P_i..P_i+n_i-1`. Logical `position_ids`
carry those positions, while `cache_position` is always the contiguous physical
storage index `0..stored_length-1`; a logical gap must never be passed as cache
storage indices.

The immutable compacted base ends at the summary boundary. For each arm, construct
one transient branch, make the declared summary-row intervention, and only then
append the exact target retained-tail IDs beginning at `P_i+n_i`. Probe and target
tokens continue monotonically after the tail. Thus the intervention touches only
summary rows, while tail K/V may differ as legitimate causal descendants of that
intervention. Replacing summary rows in a full cache whose tail was already
computed under fresh state is prohibited: that would freeze away propagation
through the retained tail and answer a narrower direct-effect question.

Every scored compacted arm uses the same physical token IDs and exact logical
position vector. Physical cache storage remains contiguous in every branch.

## 4. Arms and estimands

- `A_full`: unchanged full-history competence/headroom reference.
- `G_fresh`: gapped compacted destination with freshly encoded summary K/V;
  its tail is appended from the summary boundary in its own branch.
- `G_correct`: `G_fresh` with correct actual-generation summary K and V copied
  bit-for-bit into the physical summary slots. No rotation or dtype conversion.
- `G_wrong`: same destination with exact-position wrong-source summary K/V copied
  bit-for-bit.
- `G_Vcorrect`: fresh K plus correct V, copied without transformation.
- `G_Kcorrect`: correct K plus fresh V, copied without transformation.
- `G_delta`: fresh K plus the existing fixed-point-free derangement of
  `V_correct - V_fresh`, with the same invariant diagnostics.

The co-primary intersection is:

```text
theta_GF = mean_i(Y_i,G_correct - Y_i,G_fresh)
theta_GW = mean_i(Y_i,G_correct - Y_i,G_wrong)
```

The claim of a correct-history coherent channel requires both contrasts to point
positive and both conversation-clustered 95% interval lower bounds to exceed zero.
At `N<=12`, failure to clear is inconclusive, not equivalence.

`G_Vcorrect-G_fresh`, `G_Kcorrect-G_fresh`, `G_correct-G_delta`, and `A_full-G_fresh`
remain exploratory/mechanistic. No packed V-only contrast becomes confirmatory in
this run because fresh and natural source values otherwise arise at different
absolute positions; the local bf16 diagnostic showed that selected-token margins
can be position-sensitive even when K is held fresh.

## 5. Gates before any downstream outcome

All original applicable gates remain. Packed K rotation gates are replaced, not
relaxed, by the following:

1. exact revision, bf16 dtype, geometry, CUDA residency, and clean non-null commit;
2. correct actual-generation versus independent replay at identical positions:
   IDs identical and token-logprob/K/V maximum difference at most `1e-4`;
3. correct/wrong prefix lengths identical; all structural/request/header slots and
   positions identical; only declared evicted content slots differ;
4. correct, wrong, fresh, and destination summary starts all equal `P_i`; exact
   `S_i` IDs and all downstream position vectors identical;
5. the gapped request/header suffix exactly matches the correct prefix suffix and
   ends at `P_i-1`; logical islands do not overlap;
6. physical DynamicCache `cache_position` remains contiguous while explicit
   `position_ids` carry the gap; every appended token attends every physically
   prior entry, no phantom gap entry exists, and no future physical entry is
   exposed by a logical-position mask;
7. `G_correct` inserted summary K/V hashes bit-match the actual generated source
   rows; `G_wrong` bit-matches the frozen wrong source; no call to `move_key_rows`
   occurs on a confirmatory arm;
8. fresh self-replacement and alpha-zero paths are exact tokenwise no-ops within
   `1e-4`; snapshot/rebuild continuation identity remains within `1e-4`;
9. wrong-rotation failure injection remains in the technical diagnostic suite so
   the retired packed limitation stays visible, but it cannot authorize a packed
   arm;
10. every branch is forked at the summary boundary; pre-tail hashes prove that only
    declared summary rows changed, then the identical tail is recomputed per arm;
    a fresh-precomputed tail reused across treatment arms fails closed;
11. placebo, bounded-memory ownership, summary-cap, raw-render, atomic persistence,
    and forced-restart gates remain unchanged;
12. the exact-label calibration is reconstructed with the same position islands
    and exact-length wrong construction.

The production loaded gate must write a complete `production_kernel_gate.json`
for both PASS and FAIL, including per-layer diagnostics, before raising. No
`A_full`, calibration target, or treatment target may be scored before all global
and first-case position/structure gates pass.

The 0.6B ladder must demonstrate:

- bit-exact coherent insertion;
- gapped self-replacement identity;
- identical gapped position vectors across arms;
- explicit causal-mask coverage over physically stored entries, including failure
  injection where logical positions are incorrectly supplied as `cache_position`;
- per-arm tail recomputation from the intervention boundary, with a test that a
  changed summary state can causally change tail state while non-summary
  intervention inputs remain identical;
- exact-length donor-slot construction and special-token exclusion;
- failure on injected wrong position vectors or altered structural slots;
- downstream sensitivity of the existing engineered control.

## 6. Sequential rule and interpretation

The first-six frozen order and donor map remain. The existing competence/headroom
and exact-label calibration gates remain, with `F` interpreted as `G_fresh`.

The 6→12 rule becomes: stop for futility only if both observed primary means
`theta_GF <= 0` and `theta_GW <= 0` and the prespecified exact-label calibration
fails. Otherwise extend once to the frozen twelve ceiling. Technical or regime-gate
failure always stops immediately. No other adaptive choice is permitted.

Interpretation:

| Observation | Licensed reading |
|---|---|
| `G_correct` clears both co-primary contrasts | A downstream-usable, correct-history-specific channel exists under position-preserving compaction. |
| `G_correct > G_fresh` but not `G_wrong` | Generic native/coherence advantage; correct-history specificity not established. |
| V-only decomposition positive inside a coherent channel | Values carry a usable component at matched logical positions. |
| K-only decomposition positive inside a coherent channel | Keys carry a usable component without position transformation. |
| Both means nonpositive while calibration fires | No detected natural-summary channel in this bounded position-preserving regime; intervals govern. |
| Calibration also fails | Inadequate sensitivity; no substantive absence claim. |

The final paper must report the failed packed bf16 attempt, the numerical diagnostic,
this pre-outcome amendment, and the deployment distinction between packed/reset and
position-preserving serving.
