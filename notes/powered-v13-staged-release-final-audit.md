# Powered-v13 staged-release final audit

**Date:** 2026-07-12

**Audited commit:** `d3809e88bb8f22edf2f9331514627c2be88b1711`

**Scope:** read-only adversarial audit of `src/powered_v13_release.py`,
`tests/test_powered_v13_release.py`, and
`tests/test_powered_v13_technical_release.py` against preregistration Sections
15 and 18. No implementation file, real release manifest, inventory contract,
authorization commit, or paid resource was created.

## Verdict

**BLOCK**

The fixed Stage-T and Stage-A wrappers correctly encode the intended literal
status transitions, but the composed gate is not yet an immutable,
non-caller-weakable authorization boundary. Four independently reproduced
defects remain.

## Blocking defects

### 1. Git replacement objects can make an unrelated authorization OID pass

Every Git command runs as plain `git ...` through `_git`
(`src/powered_v13_release.py:144-166`). The verifier neither invokes Git with
`--no-replace-objects` nor rejects replacement refs/grafts.

A throwaway SHA-1 repository demonstrated the consequence:

1. create a valid Stage-T authorization commit;
2. create a different commit whose literal tree is only the unmodified DRAFT
   parent tree;
3. install `refs/replace/<different-commit>` pointing to the valid
   authorization commit;
4. detach HEAD at the different commit;
5. create a receipt naming that different commit; and
6. run `verify_stage_t_checkout`.

The full checkout returned `PASS` and bound authorization OID
`87c62c58380f6d8515b0cad89f0c043f5dc8e561`. With replacement processing
disabled, that OID's literal tree was
`f9b01f30588460206b536d4d6f2f6efe9f02a722`; ordinary Git commands used the
replacement tree `a322617a057d3db10ae1d7ecfc7ca8c2100b1813`.

Thus exact detached HEAD and receipt commit equality do not currently prove
the bytes of the named immutable commit. All verification Git invocations must
disable replacement/graft semantics, and the production gate should reject
replacement metadata rather than silently interpreting it. Inherited Git
repository-selection environment variables should also be neutralized or
explicitly attested.

### 2. Python-visible generic specifications can authorize an arbitrary weaker stage

`_ReleaseSpec` and the generic engines `_build_manifest`,
`_verify_authorization_commit`, `_create_launch_receipt`, and
`_verify_checkout` accept an arbitrary caller-created spec
(`src/powered_v13_release.py:82-119`, `261-315`, `681-750`, `895-930`, and
`1118-1157`). The leading underscore is a naming convention, not an execution
boundary, and no generic engine asserts that its spec is exactly the frozen
Stage-T or Stage-A singleton.

A throwaway probe constructed a spec with:

- preregistration path `PROTOCOL.md`;
- transition `OLD\n` to `NEW\n`;
- a two-path caller-authored inventory contract; and
- receipt basename `weak.json`.

The generic builder, authorization verifier, receipt creator, and checkout
verifier returned `PASS WEAK PROTOCOL.md 2`.

A production runner can therefore select a weaker release law from the same
module despite the fixed wrappers. Generic code must map an internal fixed
stage discriminator to sealed constants (and reject every other object), and
the eventual production runner/import audit must prove it calls only the exact
stage wrapper. Test-only dependency injection cannot remain selectable on the
production call surface.

### 3. Production-facing receipt time controls are caller-weakable

The public receipt creators accept `created_at`; the public checkout verifiers
accept `now`, `max_receipt_age_seconds`, and `future_skew_seconds`
(`src/powered_v13_release.py:933-968` and `1160-1203`). These values directly
control the freshness decision in `_verify_launch_receipt`
(`src/powered_v13_release.py:986-1042`).

A Stage-T receipt created at `2000-01-01T00:00:00.000000Z` was accepted in the
2026 audit environment by passing the same year-2000 value as `now`; the full
checkout returned `PASS` with age `0.0`. Arbitrarily large caller age/skew
bounds provide a second equivalent bypass.

Production creation and verification must obtain time from the fixed trusted
clock path and use fixed bounds. A private test core may inject a clock, but the
production wrappers must expose no timestamp or tolerance override.

### 4. Duplicate and cross-stage release artifacts already in the parent tree are invisible

`_reject_duplicate_changed_manifests` scans only the two-path child diff
(`src/powered_v13_release.py:652-678`). It does not inspect the immutable parent
or complete child tree for a pre-existing manifest of the stage being
authorized, or for a later-stage manifest that should be impossible at that
status.

Two throwaway probes passed:

- a DRAFT parent contained a separately valid earlier Stage-T manifest, and
  its fixed-path inventory contract included that manifest; a new Stage-T
  authorization added a second manifest and passed, printing
  `preexisting_manifest_valid True` and `new_inventory_contains_old True`;
- a DRAFT parent inventory contained JSON artifacts identifying both the
  Stage-T and Stage-A manifest schemas; the Stage-T authorization still passed.

The direct Stage-A-wrapper-on-Stage-T-child test does reject, and an extra
manifest added in the same child diff is rejected. That is insufficient for
the Section-15 absent/duplicate/cross-stage condition. The verifier needs a
stage-aware scan of the immutable parent and authorization child: Stage T must
start with no Stage-T or later-stage authorization manifest and end with
exactly the selected Stage-T manifest; Stage A must start with no Stage-A or
later-stage manifest and end with exactly the selected Stage-A manifest while
preserving the one required prior-stage lineage.

## What passed

- The real preregistration path is fixed to
  `COHERENT-STATE-POWERED-SUCCESSOR-V13-PREREGISTRATION.md`.
- At audited HEAD, the literal DRAFT status occurs exactly once and the Stage-T
  and Stage-A target status lines occur zero times.
- The wrappers fix DRAFT to `TECHNICAL_CANARY_AUTHORIZED` and then
  `TECHNICAL_CANARY_AUTHORIZED` to
  `STATIC_FROZEN_PHASE_A_AUTHORIZED`; the Stage-A regression tests use the
  corrected prior status.
- Each stage uses a fixed inventory-contract path and requires canonical JSON,
  exact stage/design identity, sorted unique paths, inclusion of the contract
  and preregistration, and exact manifest equality to that contract.
- Parent-tree inventory modes, byte counts, and SHA-256 hashes are recomputed
  from Git blobs. The immediate parent and tree binding are recomputed.
- A normal authorization child is required to have one parent and exactly the
  manifest addition plus preregistration modification. The preregistration
  blob must equal the parent blob with only the one fixed literal line
  replacement, with mode preserved.
- Receipt creation is exclusive; the full checkout scans an otherwise single-
  entry directory, fixes the basename, rejects leaf symlinks/non-files, and
  binds receipt payload, authorization, parent, manifest path, and manifest
  SHA-256.
- The normal full gate requires exact detached HEAD and a Git-clean tracked and
  untracked worktree.
- A throwaway Git SHA-256 repository passed end to end with 64-hex commit and
  tree OIDs. The default SHA-1 synthetic suites also passed, so no SHA-1-only
  length assumption was observed.

## Tests and literal audit counts

- Focused repository suite: **33 passed** in **15.56 seconds**.
- Throwaway probes: **5** classes executed:
  1. Git replacement-object substitution — unsafe acceptance reproduced;
  2. caller-created weak `_ReleaseSpec` — unsafe acceptance reproduced;
  3. caller-controlled ancient receipt clock — unsafe acceptance reproduced;
  4. pre-existing duplicate/cross-stage manifest artifacts — unsafe acceptance
     reproduced;
  5. Git SHA-256 object format — expected acceptance observed.
- Audited file SHA-256 values:
  - `src/powered_v13_release.py`:
    `25b68fa93484040a411ac8998075c3babd9c8336797eca3a2edb86612df2682c`
  - `tests/test_powered_v13_release.py`:
    `9a66f051d38d767ab3d5c84fe092ebab2055942acc8df48ffd958348a2f1d4d2`
  - `tests/test_powered_v13_technical_release.py`:
    `8c42239c76da305828b4d095d3c1550e6f01bda0c24622982f793c64d5c39d01`
  - preregistration:
    `a6dec01999fcb0609760bc81a5204049f1db1958deff252b25dcfce966fe78f0`

Both real inventory-contract paths are absent at this DRAFT HEAD, as intended.
This audit **authorizes no paid work** and does not authorize Stage T or Stage A.
