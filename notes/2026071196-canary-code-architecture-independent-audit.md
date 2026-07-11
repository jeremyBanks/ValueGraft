# Independent implementation-architecture audit for the decision canary

**Author:** Independent Codex subagent  
**Runtime model:** The runtime model identifier was not exposed to Sol  
**Date:** 2026-07-11  
**Scope:** Read-only review of the current runtime, gapped destination, v11
schedule primitives, store, release/harvest stack, and tests. No file was edited
during the audit and no model forward was run.

## Architecture verdict

Use a new design ID and namespace, for example
`coherent-state-carrier-canary-v12`. Do not import v10 arm sets, schemas,
wrong-history constructors, gate schemas, or release contracts. The stable
low-level tensor/token primitives are usable, but the orchestration and
scientific schema must be new.

Two literals must be frozen before implementation:

1. Define the "anchor" as a complete target-neutral bridge turn, preserving
   role alternation:

   - fixed user bridge request;
   - fixed assistant acknowledgment, forced q=1;
   - canonical close.

   Then `R3` means carrier content through the end of that bridge turn. This
   allows a later user probe naturally. A bare user anchor leaves no clean
   role-native place for the later probe.

2. Define the same-path positive-control recipe and threshold. A natural
   engineered history effect is not a positive control because it can genuinely
   be null. I recommend a detached, gradient-designed R2 state that is
   reinserted through the public treatment path and must move a downstream
   frozen target margin in both prescribed directions.

I would not authorize paid execution until those are explicit.

## New modules

- `src/coherent_canary_schema.py`
  - design/model/revision constants;
  - `ReplayEvent`, `ReplayPlan`, `CarrierRegions`, `CarrierCapture`,
    `DestinationPlan`;
  - frozen arm, region, schedule, stage, and artifact field sets.

- `src/coherent_canary_tokens.py`
  - independently derives exact canonical message deltas;
  - identifies assistant open/content/close separately;
  - constructs role-native events;
  - constructs the compact destination as source system island plus exact
    request/carrier/anchor suffix;
  - derives nested R1/R2/R3 logical and physical spans.

- `src/coherent_canary_runtime.py`
  - executes role-native plans;
  - generates or forces the carrier q=1;
  - captures maximal R3 rows once and slices R1/R2;
  - builds fresh gapped boundaries at each region end;
  - replaces K, V, or both;
  - causally recomputes every row after the selected region;
  - scores raw downstream target log probabilities.

- `src/coherent_canary_controls.py`
  - per-layer/per-row seeded norm-matched V perturbation;
  - pod gate fixtures;
  - detached same-path downstream-margin positive control.

- `src/coherent_canary_store.py`
  - atomic, sealed, additive artifacts;
  - immutable terminal files and exact reconstruction checks;
  - no v10 constants or arm validation.

- `src/run_coherent_canary_hf.py`
  - separate modes: `technical`, `eligibility`, `treatment`;
  - no mode may reach a later stage without a committed release attestation.

- `scripts/validate_coherent_canary_harvest.py`
  - independent token/event/layout/lineage reconstruction;
  - must not import the canary runtime or runner.

- `scripts/validate_coherent_canary_release.py` and
  `scripts/launch_coherent_canary_release.sh`
  - bind technical PASS to eligibility PASS to treatment authorization.

Corresponding focused tests should mirror those module boundaries.

## Reuse versus forbid

Reuse directly:

- `arms_common.canonical_ids_any`, `render_hf`.
- `coherent_state_tokens.generation_prefix_ids`,
  `rendered_assistant_content_ids`, `probe_layout`, `teacher_forcing_feed`.
- `coherent_state_hf.sha256_ids`, `sha256_tensor`, `row_hashes`,
  `compare_rows`, `append_ids_stepwise`, `generate_greedy_incremental`.
- `coherent_state_hf.extract_summary_rows` only behind a canary wrapper named
  `extract_region_rows`, with its own R3 bound.
- `coherent_state_hf.replace_summary_rows` only behind
  `replace_region_rows`, which audits arbitrary region lineage.
- `kvlib_hf.prefill`, `snapshot_cache`, `rebuild_cache`, `tf_logprobs`.
- `coherent_state_runtime.validate_position_schedule`,
  `snapshot_physical_length`, `eos_ids`.
- `coherent_state_runtime.eager_backend_fingerprint`.
- Pure sealing/hash helpers from `coherent_state_integrity`, if imported without
  its v10 envelopes.

Do not reuse:

- `matched_wrong_prefix_ids` or any donor/cycling logic.
- `derive_prefix_schedules` as the primary schedule: it block-prefills
  historical assistant messages.
- `SourceCapture`: its summary-only fields cannot represent R1-R3 correctly.
- `summary_destination_layout`, `gapped_destination_layout`: both hard-code the
  old content-only boundary and message ordering.
- `gapped_arm_boundary`, `arm_snapshot`, old arm enums.
- `delta_deranged_snapshot`: it does not implement the newly frozen
  per-layer/per-row norm match.
- `coherent_state_store` schemas or stage promotion.
- `run_loaded_gapped_gates`, its pseudoreplicated schedule fixtures, old donor
  gates, or v10 calibration.
- v10 harvester, terminal envelope, semantic-release overlay, or launch scripts.

## Role-native replay

Construct the event plan from canonical completed-message deltas, never by
standalone tokenization or naive prefix re-rendering.

For every historical assistant message:

1. append exact `<|im_start|>assistant\n` as a structural block;
2. force every exact rendered content token in q=1 calls;
3. append exact canonical `<|im_end|>\n` as a structural block.

System, user, and tool messages are appended as their exact canonical message
blocks. If any block exceeds 4,096, either fail the canary stimulus or split it
explicitly and record every resolved call; never rely on `prefill`'s invisible
auto-chunking.

For the carrier:

1. append summary-request user block;
2. append assistant generation header;
3. generate or force carrier content q=1;
4. append canonical close exactly once;
5. append the complete fixed bridge turn under the same role-native rules.

Qwen's EOS is also the assistant close token. `generate_greedy_incremental`
observes EOS without appending it, so the close block must append it once;
duplicate or missing close is a high-risk failure.

Correct and counterfactual histories must have identical:

- roles and message count;
- token width per message;
- event kind and event width;
- logical and physical positions;
- carrier and bridge IDs.

## R1/R2/R3 and destination

Capture one maximal contiguous source span per history:

- `R1 = [carrier_content_start, carrier_content_end)`;
- `R2 = [carrier_content_start, carrier_close_end)`;
- `R3 = [carrier_content_start, bridge_turn_end)`.

The fresh compact destination is the exact source system block plus the exact
request/carrier/bridge suffix. Its physical cache is packed, but logical
positions are copied from the corresponding source indices. Thus source and
destination region rows occupy identical logical positions and require no key
rotation.

For each arm:

1. build fresh destination only through that region's end;
2. self-replacement must be bit-exact;
3. replace selected K/V channels;
4. recompute every later event causally:
   - R1 recomputes close and bridge;
   - R2 recomputes bridge;
   - R3 begins probe continuation after the retained bridge;
5. verify the branch boundary remained immutable and every unselected
   row/channel remained bit-exact.

All arms contain identical visible carrier, close, bridge, and probe text.
R1/R3 may not change visible text relative to R2.

Capture correct and wrong source R3 once, move only bounded rows to CPU, derive
R1/R2 slices, and release the full source cache. Raw cache tensors should not be
committed; exact messages, token IDs, event plans, traces, region hashes, and
reconstruction contract are the scientific artifact. On resume, replay and
require exact saved row hashes before scoring.

## Exact-30B gate subset

Before any treatment score on the paid stack:

1. Exact subject binding:
   - model and tokenizer revision;
   - all floating parameters bf16 and CUDA-resident;
   - 48 layers, 32 attention heads, 4 KV heads, head dimension 128, RoPE theta
     10,000,000;
   - eager backend attested on every layer;
   - context limit;
   - clean trunk commit and full apparatus inventory.

2. Role-native template gate:
   - exact assistant open/content/close decomposition;
   - generated versus forced q=1 carrier identity;
   - compare token log probabilities and all R1/R2/R3 K/V rows;
   - include the bridge turn, not content alone.

3. Gapped-path gate:
   - logical/physical position coverage;
   - snapshot/rebuild continuation;
   - fresh self-replacement at all three region ends;
   - K-only, V-only, and joint confinement at primary R2;
   - exact selected insertion and bit-exact preservation elsewhere;
   - descendants change after a deliberate perturbation;
   - base boundaries remain immutable.

4. Same-path positive control:
   - construct a trainable R2 cache state from the fresh technical fixture with
     model parameters frozen;
   - optimize a downstream correct-minus-counterfactual target margin using a
     prespecified epsilon sequence;
   - detach the edited rows and destroy the graph/live destination;
   - reconstruct a new fresh gapped destination;
   - reinsert the released rows through the exact public R2 replacement
     function;
   - recompute the bridge and score downstream targets;
   - require `+delta` to raise and `-delta` to lower the frozen margin by a
     preregistered minimum;
   - persist raw target log probabilities, row hashes, delta norms, confinement
     audits, and every attempted epsilon.

This licenses the intervention/readout path only. It is not evidence of a
natural history channel.

The primary same-stack schedule yardstick should run only R2 full-KV
correct/wrong under role-native N and old turn-aligned P. Define it as:

```text
|D_N| >= 3 * |D_N - D_P|
```

R1/R3 or P cannot rescue failed role-native R2.

## Checkpoint schema

Use immutable prefix-complete files beneath a unique run directory:

- `manifest.json`
- `gates/<stage>.json`
- `cases/<order>_<id>/stimulus.json`
- `cases/<order>_<id>/source_<history>_<carrier>.json`
- `cases/<order>_<id>/destination.json`
- `cases/<order>_<id>/eligibility.json`
- `cases/<order>_<id>/arms/<schedule>_<region>_<k-source>_<v-source>.json`
- `cases/<order>_<id>/case_analysis.json`
- terminal index and receipt.

Every source artifact should include literal messages, canonical IDs, every
event and call width, logical/physical positions, carrier text/IDs, EOS/cap
status, tokenwise NLL, R1-R3 spans and hashes, model/code provenance, and
`raw_tensor_archived: false`.

Every arm artifact should include visible-text hash, named K/V source, source
artifact payload hashes, selected and nonselected segment hashes, descendant
hashes, exact target IDs, raw token log probabilities, deterministic answer,
and elapsed time.

Eligibility must be a separate process and committed/independently harvested
before treatment authorization. It contains full-history competence, fresh
damage, leakage, carrier support/NLL, and nothing derived from
correct-versus-wrong treatment contrasts.

## Independent harvester

The harvester should use only the standard library plus the pinned tokenizer.
It must not import runner constructors.

It independently:

- rerenders literal messages with the dummy-user canonical method;
- reconstructs every role-native event and q=1 call;
- derives the source-system plus suffix destination mapping;
- derives R1/R2/R3 spans;
- confirms all visible-text hashes are identical;
- validates per-layer row shape/hash coverage and expected K/V hash mixing;
- validates confinement from before/region/after hashes;
- reconstructs probe and target token IDs;
- recomputes means, margins, focal selectivity, correct-target movement,
  schedule yardstick, and stopping rule from raw token log probabilities;
- verifies technical, eligibility, and treatment commit ancestry and release
  attestations.

It must ignore the runner's aggregate analysis.

## Highest-risk traps

- Naive re-rendering around final assistant messages can silently add/remove
  Qwen's empty reasoning wrapper.
- EOS/`<|im_end|>` can be duplicated or omitted.
- `prefill` silently chunks calls above 4,096, invalidating recorded geometry.
- Logical positions can accidentally be used as physical `cache_position`.
- Source extraction indices are logical=physical; gapped destination indices
  are not.
- R2 close must be one complete structural call; never cut a region inside it.
- R1/R2 recomputation must start from the treated boundary, not from a cached
  fresh suffix.
- Mechanical exact-width validity does not establish decoded semantic validity.
- The norm-matched control must be seeded and constructed before outcomes;
  preferably orthogonalize its per-row random direction to the
  correct-minus-wrong V delta before rescaling.
- Do not use treatment outcomes for eligibility, redesign, leakage
  classification, or carrier selection.
- The 3x yardstick concerns the semantic contrast, not cache-max differences.
- Full-KV success does not license a value-only claim.
- A gradient-designed positive control proves only that the apparatus can
  transmit a cache intervention.

## Final recommendation

This is implementable with the existing stable low-level machinery, but not
safely by extending the current v11 schedule module or v10 runner.
