# Powered-v13 staged-release repair final re-audit

**Date:** 2026-07-12

**Audited commit:** `91346e249dfc6e94a92027878c422117edea9fae`

**Integrated repair commit:**
`2f688a2da0a9aa10c6e3b25d06afebe90b10211b`

**Scope:** read-only final re-audit of
`src/powered_v13_release.py`, `tests/test_powered_v13_release.py`, and
`tests/test_powered_v13_technical_release.py` against the four blockers in
`notes/powered-v13-staged-release-final-audit.md`, plus SHA-1/SHA-256 Git
operation, replacement/graft and inherited-Git-environment attacks, fixed
production clocks, exact Stage-T/Stage-A lineage, duplicate/cross-stage
manifest rejection, detached-clean checkout, and receipt binding. No source,
release contract, manifest, receipt, preregistration, or paid resource was
changed.

## Verdict

**PASS**

At exact audited HEAD, the three audited files were byte-identical to the
integrated repair commit. The four reproduced authorization-boundary defects
were closed, their positive and adversarial tests passed, and no replacement
blocker was found in this scope.

This PASS is a verdict on the repaired release-verification library and its
tests. It is not a Stage-T or Stage-A authorization and does not authorize paid
work.

## Original blocker dispositions

### 1. Git replacement/graft and ambient repository selection — closed

- Every verifier Git subprocess removes inherited `GIT_*` variables, then sets
  `GIT_NO_REPLACE_OBJECTS=1` and `GIT_LITERAL_PATHSPECS=1`. A direct mocked
  environment probe observed only those two `GIT_*` entries after poisoning
  `GIT_DIR`, `GIT_WORK_TREE`, object, index, namespace, and injected-config
  variables.
- Exact-commit admission rejects any `refs/replace/` entry and any extant or
  symlinked `.git/info/grafts` metadata before reading commit semantics.
- The synthetic attack that maps an unrelated authorization OID to a valid
  authorization through `git replace` failed closed. The legacy graft probe
  and inherited repository-selection environment probe also failed or were
  neutralized as intended.
- Exact OID admission accepts only lowercase 40- or 64-hex commit IDs. Default
  SHA-1 synthetic releases and a full SHA-256 Stage-T authorization, receipt,
  detached checkout, and verification path passed.
- The audited repository itself used SHA-1, had no replacement refs or graft
  file, and `git fsck --no-dangling --no-reflogs --strict` emitted no error.

### 2. Caller-forged generic release specification — closed

- The generic engine now accepts an exact internal `_StageKey` only. `_spec_for`
  requires the exact enum type and maps its two members through an immutable
  `MappingProxyType` table to the frozen Stage-T and Stage-A specifications.
- Generic build, authorization, receipt, and checkout functions no longer
  accept a caller-created `_ReleaseSpec`.
- The public entry points expose no `stage`, `stage_key`, `spec`, or capability
  selector. Stage-T and Stage-A wrappers each select their fixed internal key.
- A reproduced weak `_ReleaseSpec` and a string impersonating the technical
  key were rejected by the focused suite.

### 3. Caller-weakable receipt time controls — closed

- Public Stage-T/Stage-A receipt creators expose no `created_at` argument.
- Public receipt and full-checkout verifiers expose no `now`, maximum-age, or
  future-skew argument. They sample timezone-aware UTC internally after the
  authorization and receipt bindings are read.
- Production bounds are fixed at 300 seconds maximum age and 5 seconds future
  skew. Deterministic clock injection remains only on explicitly private
  test helpers.
- Public real-clock receipt/checkout paths passed for both stages. Receipts
  privately constructed in years 2000 and 2100 were rejected by the public
  Stage-T checkout as stale and too far in the future, respectively.

### 4. Parent-tree duplicate/cross-stage invisibility — closed

- The verifier scans the complete immutable parent and authorization-child
  trees for the known canonical Stage-T and Stage-A manifest schemas, including
  the superseded Stage-A v1 schema.
- A Stage-T root must contain no Stage-T or Stage-A manifest; its child must
  contain exactly the selected Stage-T manifest and no Stage-A manifest.
- A Stage-A root must contain exactly one prior Stage-T manifest, no Stage-A
  manifest, and must include that Stage-T manifest in its fixed inventory
  contract. Its child must preserve that exact Stage-T path and contain exactly
  the selected Stage-A manifest.
- Stage-A manifest schema v2 binds the prior Stage-T authorization commit,
  Stage-T static root, manifest path, and manifest SHA-256. The verifier finds
  the manifest-addition history, reruns the exact fixed Stage-T authorization
  verifier, requires one valid lineage, and requires the current manifest bytes
  to match it.
- Focused tests rejected pre-existing Stage-T duplicates, pre-existing or
  cross-stage Stage-A manifests, duplicate child manifests, omission of the
  prior Stage-T manifest from the Stage-A inventory, mutation of that manifest,
  and tampering with the recorded prior authorization.

## Composed checkout and receipt boundary

The re-audit also observed the following composed behavior:

- Both stages bind a canonical manifest to an exact immutable parent tree and
  fixed canonical inventory contract, recomputing blob mode, byte count, and
  SHA-256.
- An authorization is one exact, one-parent commit whose diff is only the new
  mode-100644 manifest plus the fixed preregistration status-line transition.
- Full checkout requires exact HEAD equality, detached HEAD, and a clean tree
  including untracked files. Branch, wrong-HEAD, tracked-dirty, and
  untracked-dirty probes failed closed.
- The receipt directory must contain exactly one fixed-basename, regular,
  non-symlink file. Receipt bytes must be canonical JSON plus LF and bind the
  authorization commit, static root, manifest path, manifest SHA-256, payload
  hash, stage identity, and fresh timestamp. Missing, duplicate, extra-entry,
  symlink, stale, and rebound/tampered receipt probes failed closed.
- Positive complete Stage-T and Stage-A synthetic checkouts returned PASS only
  from exact detached, clean authorization commits with matching fresh
  receipts.

## Tests and exact provenance

- Focused command:
  `PYTHONPATH=src uv run --locked pytest -q tests/test_powered_v13_release.py tests/test_powered_v13_technical_release.py`
  — **50 passed in 43.86 seconds**.
- Broader command:
  `PYTHONPATH=src uv run --locked pytest -q tests/test_powered_v13*.py`
  — **514 passed, 1 skipped, 10 warnings in 247.07 seconds**. The skip was the
  expected non-Linux production-procfs gate on macOS. Warnings were SWIG and
  multiprocessing/fork deprecations, not release-test failures.
- Audited file SHA-256 values:
  - `src/powered_v13_release.py`:
    `2e33a7052f43c4b2336e491cf498eebecdfc462e68a1305b33748c93e55b1487`
  - `tests/test_powered_v13_release.py`:
    `9a218182799e0cb46d9b977d50756061c8d76a79fbdabaaf4fefb5079a4d2c0b`
  - `tests/test_powered_v13_technical_release.py`:
    `2d062d583567aaa8c569377fa33c1292a00d2b02a0ab6c1481d765eedcec2c1e`

## Live authorization boundary

At audited HEAD, the v13 preregistration literal status remained
`DRAFT — NO PAID WORK OR PRIMARY TREATMENT AUTHORIZED`. Both fixed real
inventory-contract paths were absent, no tracked real Stage-T/Stage-A release
manifest or launch receipt was observed, and a complete-tree schema scan found
zero Stage-T and zero Stage-A manifests. Therefore this PASS closes the code
repair audit only; an exact later release, receipt, production-wrapper call
path, and all other frozen gates remain required before paid work.
