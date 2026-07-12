# Powered-v13 Stage-A release verifier audit

Date: 2026-07-12

Scope: unpaid pure-Git Stage-A release foundation. No real release manifest,
inventory contract, status transition, launch receipt, model load, or paid work
was created.

## First-pass rejection

The first implementation passed its synthetic tests but was rejected by an
independent conventional review for three live-release defects:

1. the preregistration path and old/new status bytes were caller-supplied, so
   the generic verifier did not enforce the literal v13 transition;
2. manifest completeness was caller-declared, so a required
   experiment-bearing path could be omitted while every listed hash passed;
3. receipt uniqueness checked only a caller-supplied list, so an undisclosed
   second receipt was invisible.

## Repair and exact-commit re-audit

Commit `e3cdbb58a1eaaa76a1ac67770e853e90bf968fb2` repaired all three:

- code pins the real v13 preregistration path and exact LF-terminated
  `DRAFT` to `STATIC_FROZEN_PHASE_A_AUTHORIZED` bytes;
- a strict canonical contract at the fixed parent-tree path
  `data/coherent_state_powered_v13/stage-a-inventory-contract.json` enumerates
  the complete inventory, must include itself and the preregistration, and the
  release manifest inventory must equal it exactly;
- receipt creation and verification use one fixed basename in an otherwise
  empty real directory and reject any extra entry, directory, or symlink.

A fresh read-only reviewer then audited the exact integrated commit and
returned **PASS**. The actual preregistration contained the old line exactly
once and the new line zero times. The focused 23-test suite passed, including
fixed status/path, omitted or altered/noncanonical contract, contract hash,
wrong parent/extra diff, dirty or attached checkout, undisclosed extra receipt,
wrong basename, directories, and symlinks. A separate wrong-single-basename
probe also failed closed.

Audited hashes:

- `src/powered_v13_release.py`:
  `ddeb1b0a2a787407c4c1664880ab590b9c4f3e934c23ae439ad783ec41da01b5`
- `tests/test_powered_v13_release.py`:
  `425c5927d74c0e4c9ef6f2d5935f66fc8f6e388cd64b5e051dceb482c19ddcf8`

## Expected hold

The real inventory-contract file is intentionally absent. Manifest creation
therefore fails closed today. It will be created only after all Stage-A
experiment-bearing code, data, locks, literal fixtures, and review inputs are
finished and reviewed. This PASS qualifies the verifier foundation; it does
not authorize the future status-changing commit or any paid Phase A.
