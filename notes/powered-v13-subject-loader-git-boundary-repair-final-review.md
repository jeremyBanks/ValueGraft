# Powered-v13 subject-loader Git-boundary repair final review

**Date:** 2026-07-12

**Audited HEAD:** `7674bf6ac8f6519c28c0e17ca4b35dda67ca7d15`

**Repair commit:** `b410fd88d3bb9b842ee421c0dc670af53786e103`

## Verdict

**PASS**

The repair closes both concrete bypasses recorded in
`notes/powered-v13-subject-loader-final-review.md`.

- Every subject-side Git operation now reaches the single `_git` boundary.
  The only other subprocess helper is called with fixed `uv` and `nvidia-smi`
  commands. `_git` removes every inherited `GIT_*` variable, then fixes
  `GIT_NO_REPLACE_OBJECTS=1` and `GIT_LITERAL_PATHSPECS=1` before invoking Git
  with the intended repository as its working directory.
- The static-root inventory-contract `cat-file` read in
  `_fixed_stage_t_release` uses `_git`. Its revision is a verifier-returned
  hexadecimal commit ID and its path is the fixed, normalized Stage-T
  inventory path. Ambient repository/object-directory selection and a decoy
  `refs/replace` therefore cannot redirect or replace this object read.
- The post-load `rev-parse`, detached-HEAD check, and complete porcelain status
  check all use the same fixed boundary against the passed checkout. A dirty
  intended checkout fails even when ambient `GIT_DIR` and `GIT_WORK_TREE`
  select a clean detached decoy.

The exact regression combines the prior replacement-object and dirty-decoy
conditions: it observed the original real-repository blob and rejected the
dirty real checkout. No subject or release implementation changed after the
repair; the audited subject/test files match `b410fd8`.

## Tests

- Exact regression:
  `PYTHONPATH=src ../v13-subject-loader/.venv/bin/python -m pytest -q tests/test_powered_v13_subject.py::test_subject_git_boundary_ignores_ambient_repo_and_replacement_redirection`
  — **1 passed in 0.19s**.
- Focused subject/release suite:
  `PYTHONPATH=src ../v13-subject-loader/.venv/bin/python -m pytest -q tests/test_powered_v13_subject.py tests/test_powered_v13_release.py tests/test_powered_v13_technical_release.py`
  — **86 passed, 2 warnings in 74.44s**.

This PASS is limited to the repaired subject-loader Git boundary. It does not
authorize paid work or replace the preregistered Stage-T release conditions.
