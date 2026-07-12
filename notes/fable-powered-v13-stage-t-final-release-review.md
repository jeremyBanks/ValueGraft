# Fable review — powered-v13 Stage-T final release path

**Reviewer:** Claude Fable, high effort, fresh read-only CLI consultation  
**Date:** 2026-07-12  
**Reviewed tree:** scientific/lifecycle code through `eef1686`, with the DRAFT
Section-15 dual-receipt clarification and final Stage-T contract/report present
in the working tree.  
**Boundary:** only reachable false PASS/COMPLETE, unauthorized input, invalid
artifact, paid-Pod leak/duplication, deterministic launch failure, or violation
of the `$1.50` / 3,300-provider-second Stage-T cap.

## Verdict at review time: BLOCK (two timing/cap items)

Fable found the scientific and release-integrity chain otherwise coherent. It
specifically traced the repaired terminal-receipt-before-state ordering,
terminal harvest validation, exact batch/return-code binding, credential and
receipt-directory separation, fresh post-bootstrap receipt, import-closure
recomputation, ambiguous-create reconciliation, emergency cleanup, single-host
rule, and final lifecycle COMPLETE condition.

### 1. Setup receipt age was rechecked too late

The initial receipt passed the normal 300-second freshness gate locally before
allocation, but `OpenSshTransport.sync` invoked the same normal verifier again
on the remote after Pod boot, SSH admission, clone, and rsync. The sparse
`--filter=blob:none` clone could make the full-tree release schema scan lazily
fetch many blobs. A valid setup receipt could therefore age out at the remote
pre-bootstrap gate, after admitting the only allowed host, deterministically
turning a sound launch into a cleaned `POST_ADMISSION_FAILURE` with no retry.

Required minimal repair:

- make the remote setup gate a fixed, non-caller-selectable path that verifies
  detached clean HEAD, authorization child/parent, manifest/inventory, exact
  setup-receipt bytes/hash/schema/commit/manifest bindings, and import-report
  recomputation, but does not claim the already locally checked setup receipt
  is still a fresh subject-launch receipt;
- leave the normal subject verifier and its 300-second rule unchanged for the
  fresh receipt created after bootstrap;
- remove the partial-clone blob filter or prefetch the full commit so release
  verification has a predictable object set;
- state this distinction in the still-DRAFT Section 15.

### 2. Provider seconds were capped per attempt, not cumulatively

The watchdog bounded each record by 3,300 seconds and cumulative dollars, but
its record had no prior-provider-seconds field. If the first billed allocation
was rejected and a second was permitted, the second could receive another full
3,300-second allowance at a sufficiently low hourly price. Dollar accounting
remained cumulative, but the preregistered earlier-of `$1.50` or 3,300 seconds
is a Stage-T-wide cap.

Required minimal repair:

- persist caller-supplied and observed cumulative Stage-T provider seconds;
- charge every positively identified allocation from its create clock through
  positive deletion;
- build each subsequent watchdog record with
  `min(3300 - prior_seconds, dollar-derived remaining seconds)`;
- fail closed if the remaining interval cannot preserve the cleanup lead;
- expose the cumulative observed seconds in lifecycle evidence and tests.

## Checks that passed review

- The job wrapper can publish terminal state only after an atomic, fsynced,
  canonical job-terminal receipt; harvest requires the receipt's exact schema,
  batch ID, status, and return-code pairing.
- Lifecycle COMPLETE additionally requires watchdog `job_complete`, successful
  harvest, and both direct 404 and complete active-inventory absence.
- The Hugging Face token is outside receipt and harvest directories.
- The fresh subject receipt is created by the fixed release function after
  dependency/model download, is the only entry in its directory, is verified
  normally before model load, and is copied/hash-checked into the harvest.
- The setup and fresh receipt hashes are distinct and both preserved.
- Static Stage T contains only the four technical execution roots; production
  pool/permutation/analysis paths are forbidden and import audit is recomputed.
- Zero-active-pod preflight, nonce-bound ambiguous-create recovery, dual
  deletion, one admitted host, ordinary/emergency watchdog paths, and no warm
  hold all fail closed.

## Residual operational risks, not blockers

- Harvest has 60 seconds to transfer a bounded result tree; a slow link can
  produce `TERMINATED_HARVEST_FAILED` and delete the Pod, losing the canary but
  not creating false evidence.
- The total 3,300-second window remains tight for boot, setup, a large model
  download, load, two cases, harvest, and cleanup.
- A port mapping can appear before SSH is ready; the single SSH admission path
  may reject such a host safely.
- A simultaneous operator crash before watchdog ownership and provider
  ambiguity remains an irreducible cleanup case; the zero-pod preflight and
  nonce reconciliation reduce but cannot mathematically eliminate it.

## Dual-receipt assessment

The dual-receipt design is coherent and preserves the subject's 300-second
freshness guarantee, provided the setup/subject distinction above is explicit.
The first receipt authorizes paid setup and is freshness-checked locally. The
second is created only after setup from the same immutable authorization and
manifest, then passes the unmodified normal verifier immediately before model
load. Download is not model load. No technical arm executes between setup and
fresh receipt creation.

Advice is not authority. The main agent owns the repairs, final evidence, and
authorization decision.
