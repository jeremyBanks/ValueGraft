# Powered-v13 subject-loader final review

**Date:** 2026-07-12

**Audited commit:** `fbe654ec08110a21d78587b4f0164c100ca51f75`

**Subject implementation commit:**
`73bee60f5be21ef322ea2cecb587fa876fb2f39a`

**Scope:** read-only independent review of `src/powered_v13_subject.py`,
`data/coherent_state_powered_v13/subject-contract-v1.json`, the repaired
Stage-T release API, and `tests/test_powered_v13_subject.py`. The review covered
release-before-load ordering, no v12 authorization reuse, public policy seams,
immutable contract and dependency hashes, exact snapshot/model/topology/content
and tokenizer checks, A100/CUDA/bf16/eager admission, one-live-handle lifecycle,
post-load release recheck, forbidden production imports, and failure eviction.
No implementation, contract, release artifact, or paid resource was changed.

## Verdict

**BLOCK**

The model/runtime and lifecycle attestation is extensive, but the production
subject boundary performs three security-bearing Git reads outside the repaired
release verifier without scrubbing inherited `GIT_*` variables or disabling
replacement objects. Two read-only temporary-repository counterexamples
reproduced unsafe acceptance:

1. a valid repaired Stage-T release whose real static-root inventory omitted
   the subject loader was accepted after a decoy repository and replacement
   tree supplied a forged inventory-path list; and
2. the post-load recheck returned `PASS` and `clean_tree: True` while the real
   subject checkout contained an untracked dirty file, because it inspected a
   clean detached decoy instead.

Thus `open_exact_subject(repo, receipt_directory)` has an ambient
caller-controlled repository-selection bypass despite exposing no explicit
policy argument. Stage-T does not yet reliably authorize the bytes that the
subject loader uses, and the load is not reliably bracketed by checks of the
same real checkout.

## Blocking defect: subject-side Git is caller-redirectable

The repaired `powered_v13_release.verify_stage_t_checkout` sanitizes every Git
subprocess and rejects replacement refs/grafts. The subject wrapper correctly
calls that fixed public verifier first. Immediately afterward, however,
`_fixed_stage_t_release` rereads the static-root inventory contract with direct
`subprocess.run(["git", "cat-file", ...])` at lines 326--330. That subprocess
inherits `GIT_DIR`, `GIT_WORK_TREE`, `GIT_OBJECT_DIRECTORY`, namespace,
replacement, config, index, and pathspec environment.

The post-load `_recheck_stage_t_binding` repeats the defect. Its `rev-parse`,
`symbolic-ref`, and `status` calls at lines 505--517 inherit the same ambient
Git selection. `_run_text` at lines 264--279 does not provide a sanitized
environment.

### Counterexample A: forged required Stage-T inventory

A temporary real repository was given a valid repaired Stage-T authorization
and fresh receipt. Its true static-root inventory contract contained only:

```text
COHERENT-STATE-POWERED-SUCCESSOR-V13-PREREGISTRATION.md
data/coherent_state_powered_v13/technical-canary-inventory-contract.json
```

It therefore intentionally omitted `src/powered_v13_subject.py`, the v13
subject contract, factual code, locks, and legacy model-bit contract, and should
have been rejected by `_validate_release`.

A clone containing the same objects installed
`refs/replace/<real-static-root>` pointing to a commit whose contract listed all
11 `_REQUIRED_STAGE_T_PATHS`. The process environment then selected the clone
through `GIT_DIR` and `GIT_WORK_TREE` before calling:

```python
observed = subject._fixed_stage_t_release(real_repo, receipt_directory)
```

The repaired release verifier ignored the poisoned environment and correctly
verified the real repository. The subsequent subject-side `git cat-file`
honored the decoy replacement. Exact observed output was:

```text
REAL_OMITS_SUBJECT True
DECOY_REPLACEMENT 0ce72efb19440d823aeefc78106e32cdac171f0e->1165dbc5240eee5aee97a0ab7aeeb8f586fd2d74
SUBJECT_OBSERVED_REQUIRED True
SUBJECT_OBSERVED_COUNT 11
```

This is not a hash collision or a forged release-verifier result. It composes a
real fixed-verifier PASS with the subject's later unsanitized read of a different
Git object graph.

### Counterexample B: dirty real checkout passes post-load recheck

Two temporary repositories shared the same exact commit. The real subject
checkout then received an untracked file; the decoy remained clean and was
detached at that commit. With ambient `GIT_DIR`/`GIT_WORK_TREE` selecting the
decoy, the production recheck was called against the real path:

```python
(real_repo / "untracked-during-load.txt").write_text("dirty\n")
os.environ["GIT_DIR"] = str(decoy_repo / ".git")
os.environ["GIT_WORK_TREE"] = str(decoy_repo)
result = subject._recheck_stage_t_binding(
    real_repo,
    {"authorization": {"authorization_commit": exact_commit}},
    subject._load_contracts(real_repo),
)
```

Exact observed output was:

```text
ACTUAL_STATUS '?? untracked-during-load.txt'
RECHECK_RESULT {'status': 'PASS', 'detached_head': '791caee6385d9f27cb10b902b0997b0737a38c66', 'clean_tree': True, ...}
```

The contract reread used the real filesystem and still matched, which makes the
counterexample narrower: only the Git identity/detached/clean assertions were
redirected, and that was sufficient for an unsafe PASS.

### Required repair and regression gates

- Route every subject-side Git invocation through one fixed helper that removes
  inherited `GIT_*`, sets `GIT_NO_REPLACE_OBJECTS=1` and literal pathspecs, and
  rejects replacement refs and graft metadata for the selected real repo.
- Prefer returning the already verified static-root inventory paths from the
  release boundary, or expose a fixed safe release recheck, rather than
  reimplementing object reads with a weaker Git path.
- Recheck the same exact real repository after load and reject replacement or
  graft metadata introduced during loading. If receipt bytes are intentionally
  not age-refreshed, that boundary should remain explicit and hash-bound.
- Add production-function regressions for both counterexamples: a decoy object
  replacement must not satisfy required inventory, and a dirty real checkout
  must fail even when ambient Git variables select a clean detached decoy.
- Verify failure releases the one-live reservation and CUDA objects before any
  retry.

## What passed

### Fixed public surface and release order

- `open_exact_subject` exposes only `repo` and `receipt_directory`; `__all__`
  exports no seam, model selector, revision, policy, hash, clock, callback, or
  tolerance override.
- The production path calls the fixed public
  `verify_stage_t_checkout(repo, authorization_commit, manifest_path,
  receipt_directory)` API with values bound by the canonical Stage-T receipt.
  No v12 repository authorization is called or accepted.
- Torch, Transformers, and historical factual code are lazy imports. Tests
  observed Stage-T release before dependency probing, dependency probing before
  runtime import, A100/CUDA admission before snapshot resolution, and snapshot
  resolution before model load.
- Pool, recipe, and permutation modules are absent from the production import
  graph. No production-pool text is imported by this loader.

### Immutable contract and exact subject checks

- The v13 contract was canonical JSON plus LF and loaded successfully. Its
  SHA-256 was
  `1bba591ce70c00569a174acebdcf8318f8c09a224bbc19e96a0f510de43a6963`.
- The exact HEAD hashes for `pyproject.toml`, `uv.lock`, all three factual-code
  files, and the complete legacy model-bit contract matched the literals in the
  v13 contract. The observed locked Python, uv, and seven distribution versions
  also matched.
- The legacy v12 artifact is used only as a hash-bound factual inventory for the
  exact 30B checkpoint and tokenizer. Its schema/design, whole-file SHA-256,
  exact-subject entry hash, and protocol-tokenizer entry hash are fixed by the
  v13 contract.
- The code fixes model ID/revision, snapshot repository location, file
  inventory and SHA-256, bf16 checkpoint tensor count/topology/parameter count,
  name-to-shard mapping, Transformers-5 packed topology and conversion recipe,
  and nine cross-layer/expert/projection content comparisons to exact checkpoint
  tensors.
- Model load is fixed to local snapshot bytes, `trust_remote_code=False`, bf16,
  eager attention, `cuda:0`, no quantization/offload, frozen evaluation mode,
  empty loading diagnostics, all parameters on CUDA 0, and the exact class,
  geometry, MoE configuration, and parameter count.
- Tokenizer file hashes/sizes, wrapper/backend classes, backend serialization,
  vocabulary, chat template, special-token IDs, EOS/pad IDs, and embedding
  domain are checked.
- CUDA admission requires one valid-UUID A100 80GB PCIe, exact nvidia-smi and
  Torch memory/name/capability agreement, driver at least 580.65, Torch
  2.12.1+cu130, CUDA 13.0, and cuDNN 92000.

The local audit host had the exact tokenizer/config/index snapshot but not the
30B weight shards and was not an A100 host. Therefore full real-weight loading
and GPU residency were not observed in this audit; those remain production
admission facts, while synthetic failure tests exercised the code paths.

### Lifecycle and failure behavior

- One process-global reservation is acquired before snapshot resolution/load.
  A second handle cannot resolve or load while one model remains live.
- `close()` makes the handle unusable and drops its model/tokenizer/runtime
  references. An external model alias deliberately keeps the global slot until
  the actual model is garbage-collected; after collection, reopen succeeds.
- Load and post-load attestation failures clear the reservation, call garbage
  collection/CUDA cache cleanup, return no handle, and permit a later verified
  retry. No reusable unverified handle was observed in the exercised failures.

## Tests and exact provenance

- Focused subject plus repaired-release command:
  `PYTHONPATH=src uv run --locked pytest -q tests/test_powered_v13_subject.py tests/test_powered_v13_release.py tests/test_powered_v13_technical_release.py`
  — **85 passed, 2 warnings in 82.01 seconds** (35 subject tests and 50 release
  tests).
- Relevant broad command:
  `PYTHONPATH=src uv run --locked pytest -q tests/test_powered_v13*.py`
  — **528 passed, 1 skipped, 10 warnings in 239.01 seconds**. The skip was the
  expected non-Linux procfs test on macOS. Warnings were SWIG and
  multiprocessing/fork deprecations.
- The passing subject tests replace both production release checks with private
  seams. They contain no `GIT_DIR`, `GIT_WORK_TREE`, replacement-ref, real dirty
  checkout, or production `_fixed_stage_t_release` counterexample, explaining
  why the unsafe production path remained green.
- Audited file SHA-256 values:
  - `src/powered_v13_subject.py`:
    `bdcf4d65e2cb3dad64127801dad153e0d8a348972dfaa41cd5d7dc165b3ac66f`
  - `data/coherent_state_powered_v13/subject-contract-v1.json`:
    `1bba591ce70c00569a174acebdcf8318f8c09a224bbc19e96a0f510de43a6963`
  - `tests/test_powered_v13_subject.py`:
    `8fde532d3b0836d9f49255961ebc0b39a765df13223031bc478f0e3d354b8b96`

## Authorization boundary

At audited HEAD, the v13 preregistration remained
`DRAFT — NO PAID WORK OR PRIMARY TREATMENT AUTHORIZED`, and the real Stage-T
inventory contract was absent. This BLOCK authorizes no model load, GPU
allocation, entropy, production-pool access, or paid work.
