# Powered-v13 harness implementation audit

Date: 2026-07-12  
Scope: read-only implementation audit of the current v12 apparatus and the
new v13 planner/recipe foundation. No model forward was run. This note does not
authorize Phase A or treatment.

## Bottom line

The successor should reuse the tested cache mechanics, not the v12 experiment
or release wrappers. The reusable core is small and good: explicit-position
replay, cache snapshot/rebuild, bounded row extraction/replacement, causal
continuation, target forks, tensor hashing, exclusive-create persistence, and
the safetensors round-trip pattern. V13 needs its own sampler, bundle schema,
placebo adapter, Phase-A/treatment split, state machine, exact-subject release
binding, and Stage-A/B verifier.

The most important implementation invariant is:

> Sample a carrier from a role-native q=1 **C-history** prefix, persist its exact
> IDs, prove sampled-to-forced identity from a separately executed identical
> prefix, and prove those same IDs/positions/R1 rows are the ones used by the
> complete C/W/F plans. Never decode, strip, and retokenize as the identity
> bridge.

## Exact reuse versus fork map

| Requirement | Reuse directly | V13 work required |
|---|---|---|
| Dynamic C/W/F plans | `powered_v13_schema.py`, `powered_v13_tokens.py`; the canonical rendering helpers in `coherent_state_tokens.py`/`arms_common.py` | Add a carrier-generation-prefix plan. It must use historical-assistant q1 calls, not `execute_prefix_block`'s one-shot prefill. Assert sampled IDs round-trip as contextual assistant-content IDs and are byte-identical in C/W/F R1/R2. |
| Cache execution | `coherent_canary_runtime.py`: `snapshot_cache`, `rebuild_cache`, `extract_rows`, `replace_rows`, `execute_replay_plan`, `execute_fresh_plan`, `continue_fresh_plan`, hashing, and low-level target fork | Add/factor a continuation from a partially generated source plan and a general capped q1 generator. The existing greedy generator rejects every cap except 64; v13 needs 16 for probes and 80 for carrier attempts. |
| Stochastic carrier | The CPU-generator/top-p pattern in `_native_pick_token` (`cross_arch_probe.py`) is useful only as a pattern | New v13 sampler: exact 63-bit digest seed, temperature .7/top-p .95/top-k disabled, CPU `torch.Generator`, per-token float32 bits, positions/calls/RNG versions, EOS witness, and exclusive persistence before another attempt. Existing canary generation is greedy. |
| Generated/forced identity | `force_content_q1` and `require_generated_forced_identity` | Extend the witness to three-way sampled branch = separate forced-C branch = complete C-plan selected rows. The C branch can then continue from the sampled cache; W is independently forced and F freshly encoded. |
| Probe scoring | `append_block_to_snapshot`, `score_target_from_prefix_q1`, float32-bit serialization | New `score_probe` record that constructs one exact user+assistant-header prefix and forks C target, W countertarget, and max-16 greedy generation from it. It must accept manifest-bound contextual target IDs and record base/prefix hashes. Do not reuse v12's labels or 64-token cap. |
| Selected R2 persistence | The atomic safetensors, metadata/index, external-hash, tamper checks, and reconstruction pattern in `coherent_canary_artifacts.py` | Fork the descriptor/constants. Persist one full fresh boundary through R2 plus exact C/W selected R2 rows (N; P only where required), keyed by release/runtime/case/render/schedule/history. The loader is CPU-only; treatment must explicitly move one validated foundation to the model device before continuation. V12 reconstruction was audit-only and cannot execute a CPU `DynamicCache` on CUDA. |
| Six arms | `replace_rows` + `continue_fresh_plan` | New treatment-only module with literal order `FF, CC, WW, FC, FW, VP`; every arm starts from a fresh clone of the same persisted boundary and is synchronously persisted before the next model operation. FF should also continue from that boundary, with direct-fresh equality separately gated. |
| VP | Tensor/hash helpers; no v12 semantic control | Do **not** reuse `norm_matched_value_placebo_row`: it is a different rowwise orthogonal control. The concurrently authored pure `powered_v13_placebo.py` has the right isolation boundary (C/W/F bf16 values only, no model/probe). Add a lossless adapter between runtime `[1,H,T,D]` layer snapshots and its `[L,T,H,D]` content/structural tensors; bind and recheck selected map and per-layer result hashes at treatment. |
| Path control | `collect_fresh_region_margin_gradients` and `bf16_gradient_ulp_edit_row` are reusable mechanics | Fork the v13 wrapper. V12 searches ULP counts until one passes; v13 requires exactly four nextafter steps at contract-bound coordinates and opposite movement >=1e-4. Persist public/differentiable baseline equality and detached reinsertion evidence. |
| Exact subject | Snapshot resolution/inventory, config/weight/tokenizer attestation, MoE conversion/content sentinels, dependency and device probes in `coherent_canary_loader.py` | Do not call `prepare_pinned_subject` or `attest_loaded_subject`: both re-verify the old v12 seal. Factor/fork a v13 subject loader that first validates the Stage-A/B receipt, then performs the same model checks, and includes release hash, GPU UUID/driver/CUDA, dependency hashes, and tokenizer/template hashes in the runtime fingerprint. |
| Persistence/resume | `atomic_create_json`, fsync/read-back/hash helpers; p01's provider clock, stage timing, lossless checkpoint-before-next-stage, and recovery-raw pattern | Fork a compact v13 store. Avoid v12's copied whole-payload promotion chain. Use one immutable file per render attempt/foundation/arm plus an append-only directory of index rows. A stale active-case lock is quarantined only after checking the actual PID/process. Terminal reuse requires exact batch/release/runtime/case/render hashes; partial cases recompute and never contribute N. |
| Phase blindness | V12 treatment runner's delayed import is a useful test pattern | Put semantic Phase A and treatment in physically separate modules. Phase A must never import treatment and may call only the pure VP selector, never VP continuation/scoring. Import treatment only after Stage B verifies. Put e01/long full-arm exceptions in a separate technical entry point with hardcoded fixture hashes and `semantic_n=0`. |
| Release | Git byte-reading and exclusive receipt patterns in `coherent_canary_store.py` and the historical receipt tools | New `powered_v13_release.py`: detached exact HEAD, clean tree, one parent, exact parent named by manifest, canonical inventory rehashed from the parent tree, literal one-line status transition, child diff of exactly status+one manifest, exact post-commit launch receipt, freshness/uniqueness, and deliberate duplicate/wrong-parent/extra-diff failures. Stage B additionally checks 8x6 fixtures, two renders each, all attempt/review/Phase-A/bundle/VP bindings. |

## Minimal module graph

Keep the model-facing graph narrow:

1. Existing `powered_v13_schema`, `recipe`, `tokens`, `stimuli`, permutation,
   statistics, and cycle-accounting modules remain pure foundations.
2. `powered_v13_runtime.py`: carrier prefix, seeded q1 sampling, source
   continuation, max-16 score/generation forks, and exact identity records;
   delegates cache operations to `coherent_canary_runtime`.
3. `powered_v13_bundle.py`: bounded R2 safetensors and device materialization.
4. `powered_v13_placebo.py`: pure outcome-blind selector (already being authored).
5. `powered_v13_store.py`: immutable case artifacts, active-case lock,
   quarantine, terminal validation, and session index rows.
6. `powered_v13_release.py` and `powered_v13_subject.py`: Stage-A/B git/receipt
   verification followed by exact runtime attestation.
7. `powered_v13_phase_a.py`: attempts/reviews, C/W/F foundation, A_C/A_W/FF,
   eligibility, bundles, and outcome-blind VP status. It cannot import item 8.
8. `powered_v13_treatment.py`: bundle reload, six arms, focal/nonfocal forks,
   and terminal case record; imported only after Stage B.
9. One thin session runner plus independent validators for Phase A, release,
   terminal cases, and final analysis. Technical-e01/technical-long dispatch is
   a separate, hash-allowlisted runner mode.

The runner should retain only the prepared model/tokenizer and one live
case/render foundation. Assert: batch size one, one active case, one active
render, three histories, six serial arms, <=7,000 live tokens, <=256 selected
rows, exactly 48 layers, and a literal per-case deadline. Evict all cache and
tensor references in `finally`, run GC/CUDA cache cleanup, and write only
bindings—not embedded case payloads—to the session index.

## Likely failure modes to test explicitly

1. **Schedule confound:** generating the carrier with a single prefill and then
   forcing it with role-native q1 creates a different state despite identical
   text. The generation prefix must use the production schedule.
2. **Decode/re-encode drift:** sampled IDs may not equal contextual IDs of the
   decoded string. Reject unless exact contextual round-trip holds; never
   `.strip()` the identity-bearing text.
3. **Wrong R2 indexing:** source positions are logical/full-cache indices;
   fresh positions are packed physical indices. Persist both intervals and
   assert identical selected token IDs and widths.
4. **CPU bundle execution:** safetensors load on CPU. A reconstructed CPU cache
   passed to the CUDA model will fail or induce an uncontrolled conversion.
   Move/attest one foundation explicitly.
5. **In-place VP rotation:** donors must always be read from immutable C, not
   from the accumulating VP tensor. Test row-index-coded cycles in both
   directions and both event classes.
6. **Probe fork contamination:** focal and nonfocal, both targets, and greedy
   generation must each clone the same immutable completed state/prefix.
7. **Nominal blindness only:** importing a combined Phase-A/treatment module
   leaves treatment reachable. Add an import-trace test and reject any
   treatment schema/key in Phase-A artifacts.
8. **Dirty release checkout:** Phase-A output under the checkout prevents the
   required clean Stage-B verification. Write remote artifacts/bundles outside
   the git checkout, harvest tracked sidecars, then fetch the release-only
   commit while the model remains loaded.
9. **Review-loop deadlock:** target-aware review is part of render acceptance.
   Persist one attempt, await a hash-bound independent review receipt under the
   45-minute/$1.10 idle bound, and only then accept or advance the attempt.
   Generating all three first silently changes the sequential acceptance rule.
10. **False resume:** a terminal case from another GPU UUID/batch cannot be
    reused in the primary batch. A new-host restart regenerates and verifies all
    96 foundations before any arm score; it never mixes old terminal cases.

## Efficient build order and test reuse

1. Freeze/test release and store first: synthetic git repositories for every
   Stage-A/B accept/reject case; corruption and SIGKILL at every durable
   boundary. Reuse the atomic/store, treatment-release, and p01 interruption
   test patterns.
2. Add carrier-prefix/sampler/scoring runtime on fake models. Extend the
   existing canary runtime/token tests for q1 boundaries, exact bits,
   max-16/max-80 caps, round-trip failure, and sampled/forced/full-plan
   three-way identity.
3. Finish the v13 bundle adapter and VP integration. Port all safetensors
   tamper/no-overwrite/lineage tests; add CPU-to-device and content-versus-
   structural offset tests.
4. Implement Phase A alone and prove by import tracing and payload inspection
   that treatment is unreachable. Test first-six/rank-ten selection and call
   `powered_v13_cycle` after every candidate and every major paid gate.
5. Implement treatment from saved bundles, then standalone/forward/reverse
   warm-order equality and full hard-kill recovery. Only afterward implement
   the e01/long allowlisted timing runner.
6. Factor the exact loader under the new release verifier; run L0/L1/L3,
   sampled identity, self-replacement, exact-4-ULP path, VP available/unavailable,
   and long geometry before scaled Phase A.
7. Finally add the host-audit projection: compare invariant runtime fields while
   requiring a different GPU UUID, exact plans/rows/greedy IDs, and <=1e-5
   score/margin differences. This runner consumes saved carriers and never
   contributes semantic N.

This order keeps the expensive subject behind tested release, persistence, and
blindness boundaries, while reusing the portions of v12 that actually earned
trust.
