# Independent review — powered-v13 Stage-T technical canary runner and build-ladder contract

Reviewer runtime identity: Claude Fable 5 (`claude-fable-5`), fresh session
2026-07-12, no prior authorship of any v13 module or protocol text. Review
boundary: the Stage-T technical canary execution contract only. Estimand,
randomization, historical v12 validity, paper questions, and the separately
reviewed release verifier / exact subject loader are out of scope except where
Stage T structurally depends on them.

Evidence read: preregistration §9, §13–16, §18; the technical-canary review
disposition note; `src/powered_v13_technical.py` (all);
`src/coherent_canary_runtime.py` (all); `src/powered_v13_watchdog.py` (all);
`src/powered_v13_bundle.py` public API and descriptor validation;
`src/powered_v13_store.py` technical sequence and record-evidence schemas.

## 1. Central decision: Stage T should execute and persist both probes

The alternative — construction and continuation identities plus timings only,
no probe numbers — is not defensible, for four independent reasons:

1. **The store already forces it.** `TECHNICAL_SEQUENCE` in
   `powered_v13_store.py:73` is frozen as `STARTED, FOUNDATION_LOAD, ARM_FF,
   ARM_CC, ARM_WW, ARM_FC, ARM_FW, ARM_VP, TERMINAL_TECHNICAL`, and
   `_required_completed_arms` requires all six `PRIMARY_ARMS` before
   `TERMINAL_TECHNICAL`. A probe-free runner can never produce a terminal
   technical case; it ends in quarantine or requires a new terminal kind —
   new apparatus that the closed apparatus list forbids.
2. **Arm time without the probe is not arm time.** In
   `coherent_canary_runtime.py`, an arm's cost is dominated by continuation
   plus `append_block_to_snapshot` (probe prefix) plus per-token teacher
   forcing in `score_target_q1`/`score_target_from_prefix_q1`. Stage T exists
   to make the §16 gates (arm median ≤ 18 s, max ≤ 25 s, 4.5k extrapolation)
   forecastable before Stage A. Timings of half an arm forecast nothing, and
   the first-ever probe execution would then happen inside the Stage-A paid
   window — recreating exactly the sequencing deadlock the disposition
   amendment was written to remove.
3. **The preregistration already licenses it.** §9 states the outcome-seen e01
   exception "may execute all six arms and its already observed probe" in
   `technical_e01`, and `technical_long`'s "fixed neutral-probe" fact is
   byte-identical and visible in every history. Unblinding risk is nil by
   construction: e01's outcome is historical, and the long probe cannot
   distinguish C/W/F semantically.
4. **"Fixed technical path correctness" (§9 Stage-T sentence) is unprovable
   without executing the path.** The probe/scoring primitives are exactly the
   production primitives; a Stage T that never calls them certifies nothing
   about them.

**Persistence rule:** probe numbers (per-token log-probability float32 bits,
means, greedy IDs, stop witnesses) are persisted inside the arm artifacts in
the `technical_e01`/`technical_long` namespaces, marked
`inferential_use: forbidden`. At Stage T they gate nothing except predeclared
pass/fail diagnostics: finiteness, determinism on repeat, generated/forced
identity, and chain/hash validity. No threshold is applied to any margin
value. This satisfies §9's "numerical values cannot enter N, eligibility, a
selector, a threshold, a template choice, or a scientific claim."

## 2. Exact probe binding without opening production inputs

The current input module is deliberately probe-free
(`probe_or_score_present: False`, `powered_v13_technical.py:288`) and returns
only histories and plans. Binding must be by **literal frozen constants in the
technical module itself**, never by import:

- **e01:** copy the exact historical v12 e01 probe string and its two
  already-observed target strings (correct and wrong-focal) into
  `powered_v13_technical.py` as literals (`E01_PROBE`, `E01_TARGET_C`,
  `E01_TARGET_W`), exactly as `TECHNICAL_CARRIER` is a literal today. Record
  their SHA-256s in the module and the Stage-T manifest, with a provenance
  comment naming the v12 artifact they were transcribed from. No v12 protocol
  module, production pool, or recipe import.
- **technical-long:** one literal fixed neutral probe (`TECHNICAL_LONG_PROBE`)
  whose target is the byte-visible filler fact (e.g. a fixed neutral question
  answered by the literal filler sentence), plus a literal target string.
  Frozen the same way. Since the fact is byte-identical in every history and
  arm, its scores carry no semantic contrast by construction.
- **Runtime derivation:** probe suffix token IDs come from
  `probe_suffix_ids(tokenizer, visible_context_messages, context_ids, PROBE)`
  under the attested tokenizer — deterministic, prefix-extension-checked, and
  persisted (IDs + SHA-256) in each arm artifact. Target IDs are tokenized
  from the literals and persisted the same way.
- **Assertion restructure (mandatory):** the single field
  `probe_or_score_present: False` becomes contradictory the moment probes are
  bound. Keep the probe-free plan core (and its `technical_input_sha256`)
  intact, add a separate `technical_probe` block with its own hash, and
  replace the old field with two precise assertions:
  `production_probe_reachable: False` and `score_values_present: False`. The
  Stage-T manifest freezes the probe-block hashes alongside `E01_SHA256`.

## 3. Minimum build-ladder identities (on-host, before any timing/arm/VP measurement, per §18-2)

Run on `technical_e01` material first; every failure is immediate abort plus
cleanup.

- **L0 — determinism:** execute one fixed forward block twice; require
  identical last-logits SHA-256 and per-layer row hashes.
- **L1 — chunked vs monolithic:** execute the fresh plan event-by-event
  (`execute_fresh_plan`) versus one contiguous prefill
  (`execute_prefix_block`) over the identical (token IDs, logical, physical)
  span; require bit-identical rows and last logits. The identity claim covers
  exactly the span compared.
- **L3 — boundary continuation:** `execute_fresh_plan` with `stop_at` at the
  R2 physical end, snapshot, `continue_fresh_plan`; require bit-identity
  (row hashes, last-logits SHA, greedy next ID) with the uninterrupted run.
- **Technical replay identity — fresh self-replacement:** extract F's own R2
  rows, `replace_rows` into the F boundary for K-only, V-only, and K+V,
  continue; each must be bit-identical to the L3 continuation. This is the
  identity that makes the FC/FW arms interpretable.
- **Bundle round-trip:** `save_bundle` → read-back → `load_verified_bundle` →
  materialize → rows bit-identical to the captured snapshots. This doubles as
  the bundle-I/O measurement.

On `technical_long`, repeat only the determinism repeat and the L3 boundary
continuation at 4.5k — those are the length-scaling correctness claims; the
rest is geometry-independent and already proven on e01. Nothing beyond this
list is required before this engineering run can be trusted; broader sampler,
ranked-content, and warm-order gates correctly wait (disposition correction 2).

## 4. Six-arm checkpoints, store usage, and ordering

Per case: `STARTED` → build and verify technical input → C/W source replays
and F fresh execution → R2 extraction → bundle save and read-back →
`FOUNDATION_LOAD` (materialize **from the saved bundle**, so bundle reuse is
what the arms actually exercise; since the technical sequence has no
`FOUNDATION_BUNDLE` record, the creation artifacts are bound into the
`FOUNDATION_LOAD` record's `foundation_bundle/descriptor/receipt` hashes) →
`ARM_FF, ARM_CC, ARM_WW, ARM_FC, ARM_FW, ARM_VP` → `TERMINAL_TECHNICAL` with
the independent terminal-validation receipt.

Each arm record is written, hash-bound, and fsynced synchronously before the
next model operation (§14 order). Minimum arm artifact: arm ID; executed token
IDs with logical/physical positions and call-trace hashes; probe suffix IDs +
SHA; target IDs; per-token logprob float32 bits; greedy/forced identity result
where applicable; R2 row hashes before/after replacement (FC/FW); VP receipt
or documented VP-unavailable search evidence (ARM_VP — unavailability is the
§13-8 expected path, a valid artifact, not an abort); wall-clock start/end;
peak-VRAM delta.

Run order across the canary:

1. Watchdog record built at allocation; watchdog supervises from a process
   independent of the runner (its fault-injection tests already exist).
2. Release verifier (detached checkout, parent inventory, two-path child
   diff) — before model load.
3. Attestation: model ID/resolved revision, bf16, eager backend,
   tokenizer/template hashes, dependency locks, GPU/driver/CUDA, expected
   VRAM after load. Abort on any mismatch.
4. Build ladder (§3 above) on e01.
5. Full `technical_e01` case to terminal.
6. Long-geometry identity pair, then full `technical_long` case to terminal.
7. Harvest (tar + hash `results/`), then delete; dual 404 + inventory-absent
   confirmation.

e01 completes before long begins so the highest-value evidence is terminal if
the watchdog fires mid-long; long partials quarantine with timings still
readable, which is an acceptable Stage-T outcome.

## 5. Independent validation minimum

1. In-run: the store's independent terminal validator receipt per case
   (already implemented), plus read-back-and-hash of every persisted artifact
   before the next model operation.
2. Post-harvest, off-host: recompute both evidence chains; re-verify bundle
   descriptors/manifests via the CPU verification (non-materializing) path;
   validate the terminal watchdog record (harvest evidence, per-pod 404,
   inventory-absent) against a refreshed provider ledger showing total
   Stage-T delta ≤ $1.50 and zero active pods.
3. Diagnostic-only cross-check: e01 greedy IDs and margin direction versus
   the historical v12 observation, recorded as information (the runtime
   differs from the v12 era); mismatch is neither an abort nor a gate.

## 6. Minimum measurements

All UTC-stamped start/end pairs: acquisition and cold-start; attestation
duration; model load; per case — input build/verify, C/W source replay, F
fresh execution, R2 extraction, bundle save/read-back/load/materialize
(seconds and bytes), per-arm wall time, VP search time; peak VRAM after load
and per stage (`torch.cuda.max_memory_allocated` with resets between stages);
technical-long 4.5k replay/continuation time; harvest duration;
deletion-confirmation latency; total provider seconds and ledger dollars.
These are compared **descriptively** to the §16 gates in the post-run report;
the 18 s/25 s arm gates belong to the later Stage-A canary and must not be
promoted into Stage-T abort conditions.

## 7. Abort conditions (fail-closed)

Abort with immediate harvest-then-delete: release-verifier failure (before
load); attestation mismatch; any build-ladder identity failure; any
`V13TechnicalError` / `V13BundleError` / `V13StoreError` /
`CanaryRuntimeError`; nonfinite logprob; live-cache (7,000) or selected-row
(256) bound violation; per-case deadline; a cheap runtime assert that no
production-pool module name is in `sys.modules` at case start (the static
import audit remains primary); and every watchdog `STOP_*` (deadline, rate
increase, provider identity, job death). Harvest is attempted once and never
delays deletion beyond the 120 s lead. Ambiguous provider state holds and
never reallocates. **Not** aborts: slow arms versus §16 gates, VP
unavailability on e01, the e01-vs-v12 diagnostic mismatch, and harvest failure
(recorded; deletion proceeds).

## 8. Contradictions and unnecessary apparatus

1. **Blocking contradiction:** `probe_or_score_present: False` in the input
   core versus the contract's required probe execution. Resolve per §2
   (separate literal probe block, two precise assertions) before the runner is
   written; do not resolve it by importing a probe from production or v12
   modules, and do not resolve it by skipping probes.
2. **Structural confirmation, not contradiction:** the frozen six-arm
   `TECHNICAL_SEQUENCE` already commits Stage T to full arm execution. Do not
   add a timings-only terminal kind.
3. **Wording tension in §9:** the Stage-T sentence's measurement list ("…arm
   time, harvest, and deletion") does not name probe values, while the same
   section licenses the e01 probe and the neutral-probe path. Pin the
   interpretation in the Stage-T manifest — probe numbers persisted as
   non-gating technical diagnostics — rather than reopening the
   preregistration beyond the already-planned Stage-T amendment.
4. **Keep excluded apparatus excluded:** no new score schema, no thresholds or
   CI machinery on technical numbers, no second admitted host, no warm hold,
   no restart after model load, no network volume. The watchdog is the single
   owner of the provider clock (rate-derived bound within $1.50, 120 s delete
   lead); do not add a parallel runner-side dollar clock.
5. **Stop clock:** the hour-7.8758 deadline is feasible only if the remaining
   work is exactly the closed list — probe-literal binding plus module
   restructure with tests, runner assembly from the already-reviewed pieces,
   and the separately reviewed loader/release path. Any additional gate must
   displace one, per the disposition.

## Verdict

**PASSABLE PLAN**, conditional on three mandatory pre-run items:

1. Bind both probes as literal frozen constants with the restructured
   assertions and manifest-frozen hashes (§2). If this cannot land as
   specified, the correct call is BLOCK, because every alternative (probe
   import, probe skip, new terminal kind) violates either the import boundary
   or the frozen store sequence.
2. Freeze the runner order, measurement list, and abort list of §§3–7 in the
   Stage-T manifest so the on-host order is attestation → ladder → e01 case →
   long case → harvest → delete, with probe numbers marked non-inferential.
3. The exact subject loader and release verifier pass their separate review;
   Stage T inherits them and adds nothing to them.

With those in place, executing and persisting both probes is not only
permitted but the minimum defensible contract: it is the only design that
reaches `TERMINAL_TECHNICAL` under the frozen sequence, produces arm timings
that actually forecast the §16 gates, and certifies the production probe path
before the first paid semantic token.
