# Powered-v13 permutation-boundary audit and disposition

Date: 2026-07-12
Scope: unpaid compact-frame randomization only. No production entropy was
requested, no production seed or permutation file exists, and no in-pool
conversation text was materialized.

## Disposition

The first implementation draft was **rejected before freeze**. An independent
read-only audit reproduced caller-forgeable seed bindings, NumPy-dependent
later verification, a noncanonical raw-`u4` permutation hash, authorization of
ranks 11--4096, permissive schemas, and a roughly 16.3 MB redundant literal
artifact. Those were release-blocking defects, not cosmetic findings.

The replacement implementation now has the following boundary:

1. The seed writer starts only from a clean, pushed integration root `C0` and
   verifies the exact selection/randomization input inventory against
   `git show C0:path`.
2. It requests exactly eight separate `secrets.token_bytes(16)` values,
   exclusive-writes only the fixed seed path, and exits without constructing a
   permutation or text. The content-state says `SEEDS_FROZEN...`; it does not
   claim that its own future commit already exists.
3. The next step requires a pushed dedicated `C1` whose only changed path is
   that seed file and whose parent is the manifest-bound `C0`.
4. The permutation writer reads the exact seed bytes from `git show C1:path`,
   proves the worktree copy and every `C0` inventory blob, constructs PCG64
   permutations under the seed-pinned NumPy version, and exclusive-writes only
   the fixed literal path.
5. Construction replay is separate from durable structural verification. The
   former requires the exact NumPy version and replays all 32,768 indices. The
   latter imports no NumPy eagerly and validates the persisted integers,
   candidate mapping, IDs, hashes, and global uniqueness without RNG replay.
6. Every permutation/index/ID aggregate uses SHA-256 of canonical UTF-8 JSON
   arrays. Raw file bindings remain ordinary SHA-256 of the exact file bytes.
7. Strict JSON loading rejects duplicate keys, validators reject every extra or
   missing field recursively, and seed/literal paths are fixed under
   `data/coherent_state_powered_v13/`.
8. Each stratum persists all 4,096 indices and all 4,096 ranked stable IDs, but
   repeats canonical tuples only for ranks 1--10. The pretty artifact is about
   2.98 MB in the synthetic full-frame test, below the 4 MB preflight bound.
9. Only a resolver that validates both actual files may mint a materialization
   receipt. The recipe itself independently rejects rank 11 before reaching the
   history materializer. The receipt names `seed_manifest_sha256`, the separate
   40-character `seed_git_commit`, and `literal_permutation_sha256`.

## Independent golden and tests

For test-only seed `000102030405060708090a0b0c0d0e0f` under NumPy 2.5.1,
the first ten threshold-stratum indices are
`[3873,1811,1140,3884,1876,3037,1328,3852,1620,3559]`; the canonical index-array
SHA is `2946e202e58eee1d0e2b3be6904bdbfe55b2d2beb9355f5de5e997d3bd4714f0`
and the ranked-ID-array SHA is
`7c5c26ee50fa88b221368b4e65f02e24acdb9bfdbfc4a36c078706855d609e8c`.
The independent audit supplied these values before the replacement tests were
written; the new implementation agrees exactly.

The focused suite has 17 tests, including a real synthetic bare-remote Git
sequence `C0 -> seed-only C1 -> literal output`, exact eight-by-16-byte entropy
calls, duplicate/extra/forged-field rejection, NumPy-free structural replay,
full 8x4,096 bijection/global-ID validation, rank-10 acceptance/rank-11
pre-text rejection, compact-size bound, and exclusive-create non-overwrite.

Implementation hashes at disposition:

- `src/powered_v13_permutation.py`:
  `26634e8cb03264467c08fb50838982f6cfc1f1ce7b4db102d270f3a40257dde1`
- seed writer:
  `cd9da951ef4c569a9aa19e75f8198e6acbbd16df83e14156950ceccdd1de61f9`
- literal writer:
  `6b986a9733985da1ab0b73885ca036bf2ea9f73cca1bde902dbdfc6ad25a35a3`
- focused tests:
  `2993175abb6de118a3313bd379faba9c0519e54acebed9986013308bb147b37c`

## Remaining boundary

This is not permission to request the real seeds. The repaired implementation
still needs an independent re-audit at its committed hash, and the tokenizer,
content-review, harness, release, corruption, and build-ladder gates must pass
before `C0`. Any failure after real seeds freezes the frame and makes the
candidate ineligible or terminates v13; it is not repaired by resampling.
