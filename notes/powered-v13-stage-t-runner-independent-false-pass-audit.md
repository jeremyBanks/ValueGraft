# Powered v13 Stage-T runner independent false-PASS audit

## Boundary and verdict

I independently audited the integrated Stage-T runner and CLI at
`e2104a1bf49f14371680f4a3fedac28fde9316c9`. The audited runner, CLI,
independent terminal validator, and their focused tests were byte-identical at
`47958898af99fc48fba29774e388fd6f7cb4efe7`. The review was restricted to
reachable false PASS and invalid-scientific-artifact paths.

**Runner verdict: PASS.** I found no reachable false PASS or invalid
scientific artifact path in `src/powered_v13_stage_t.py`,
`scripts/run_powered_v13_stage_t.py`, or
`scripts/validate_powered_v13_technical_terminal.py`.

**Concrete launch-path qualification:** the full provider path is PASS only
after integrating lifecycle repairs
`3fc699ea115fef5140ac9ae7a10d488e178cf0af` and
`cd4ba0f6d09126b566c513f1de45cc6a8c829a03`, then regenerating all final
commit-bound inventory, manifest, launch receipt, and authorization evidence.
The original `e2104a1` lifecycle could publish completion before a terminal
receipt and mixed the HF token with the subject receipt while its prebootstrap
receipt could age out. Those were concrete launch blockers, not runner false
PASSes, and the cited additive commits fail closed on them.

## Observed evidence

- Exact-subject open precedes case/model work and binds the fixed model ID and
  revision, clean authorized checkout, pinned dependency/runtime inventory,
  BF16/eager geometry, one exact A100 80GB PCIe CUDA device, GPU UUID, and the
  fresh subject launch receipt into the runtime and case identities.
- `technical_e01` must pass the full frozen L0/L1/L3 production-path ladder
  before measured Stage-T work. `technical_long` requires 4,000--5,000 tokens
  and performs exact same-event repeat plus R2 stop/resume comparisons over
  calls, executed token IDs, physical/logical positions, Q1 logprob records,
  snapshot hashes, final logits, and terminal positions.
- Foundation capture persists and read-verifies C/W selected rows and the fresh
  F boundary. Bundle load, descriptor commitments, tensor hashes, CUDA
  materialization, and deterministic VP receipt are checked before arm work.
- The exact frozen arm order is `FF`, `CC`, `WW`, `FC`, `FW`, `VP`.
  Reconstruction is from the immutable F boundary, and the bundle layer checks
  exact selected inserts plus unchanged prefixes. Each completed arm executes
  continuation, correct/counterfactual Q1 scoring, greedy generation, forced
  replay identity, canonical JSON/nonfinite rejection, exclusive artifact
  write/readback, an exclusive checkpoint, and then a hash-bound store append.
- A mathematically unavailable VP produces a bound `PLACEBO_UNAVAILABLE` arm
  with no probe. It can close Stage-T technical evidence but explicitly blocks
  Stage-A progression for `technical_e01`; the long case remains diagnostic.
  Thus technical terminal validity is not misreported as applied-control
  availability.
- Terminalization recomputes the exact STARTED, FOUNDATION_LOAD, and six-arm
  chain. A distinct process independently checks canonical records, identity,
  bounds, order, external file hashes, parent hashes, terminal evidence, live
  runner PID binding, and safe non-symlink paths. The store then independently
  validates the receipt/evidence and finalizes a terminal-only index.
  `RUN_COMPLETE.json` is exclusive-written only after both cases finalize.
- The concrete wrapper compatibility path, with the two cited repairs, keeps
  the detached wrapper PID live, requires one uniquely named run directory,
  `RUN_COMPLETE` PASS with the exact batch, zero-return supervisor evidence,
  a canonical batch-bound COMPLETE terminal receipt published before COMPLETE
  job state, complete bounded harvest hashes, and dual provider deletion.
  A stale RUNNING file whose wrapper PID is dead is classified DEAD.

## Verification and limits

The exact CPU command covered the runner, independent validator, subject,
bundle, store, ladder, technical fixtures, lifecycle, watchdog, import audit,
and technical release tests. It observed **273 passed, 1 skipped** in 30.24 s;
the skip was the platform-specific Linux `/proc` integration path on macOS.
`py_compile` and the lifecycle-specific 120-test suite had already passed for
the two lifecycle repair commits.

No provider host was allocated, no model weights were loaded, no GPU outcome
was observed, and no numerical or scientific effect claim was verified in this
audit. Final authorization remains invalid until generated from the final
integrated clean commit; no stale receipt, manifest, or inventory hash should
be reused.
