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
2. It requests `secrets.token_bytes(16)` exactly once for each of the eight
   strata. Any duplicate aborts after the eighth request without resampling.
   It exclusive-writes only the fixed seed path and exits without constructing
   a permutation or text. The content-state says `SEEDS_FROZEN...`; it does not
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
9. Ranked expansion has one public entry point and no caller-supplied path,
   hash, or commit parameter. It requires a clean pushed `trunk` at a dedicated
   literal-only `C2`, proves its parent is the dedicated seed-only `C1`, proves
   `C1`'s parent is the manifest-bound `C0`, reads both fixed files through
   `git show`, matches their worktree bytes, rehashes every `C0` inventory blob,
   and structurally validates the literal before resolving ranks 1--10.
10. The recipe's ranked compiler and receipt type are private and additionally
    capability-gated. The receipt separately records `seed_manifest_sha256`,
    `seed_git_commit`, `literal_permutation_sha256`, and
    `literal_git_commit`. Persisted fixture validation requires the exact
    authorization fields, status, candidate ID, integer rank at most ten, two
    lowercase SHA-256 values, and two real-shape Git commit IDs; development
    sentinels carry the same binding fields with null values and reject extras.

## Independent golden and tests

For test-only seed `000102030405060708090a0b0c0d0e0f` under NumPy 2.5.1,
the first ten threshold-stratum indices are
`[3873,1811,1140,3884,1876,3037,1328,3852,1620,3559]`; the canonical index-array
SHA is `2946e202e58eee1d0e2b3be6904bdbfe55b2d2beb9355f5de5e997d3bd4714f0`
and the ranked-ID-array SHA is
`7c5c26ee50fa88b221368b4e65f02e24acdb9bfdbfc4a36c078706855d609e8c`.
The independent audit supplied these values before the replacement tests were
written; the new implementation agrees exactly.

The focused permutation suite has 24 test cases, including a real synthetic
bare-remote Git sequence `C0 -> seed-only C1 -> literal-only C2`, exact
eight-by-16-byte entropy calls plus duplicate abort without resampling,
duplicate/extra/forged-field rejection, NumPy-free structural replay, full
8x4,096 bijection/global-ID validation, rank-10 acceptance/rank-11 pre-Git
rejection, compact-size bound, exclusive-create non-overwrite, and deliberate
uncommitted, unpushed, dirty, non-dedicated-C1, non-dedicated-C2,
manifest-parent, and C0-inventory failures.
The success path monkeypatches the private compiler with a non-text sentinel;
no in-pool history is materialized. The recipe suite separately exercises
invented-receipt capability rejection and strict persisted authorization
schemas for both ranked and development records.

Implementation hashes at disposition:

- `src/powered_v13_permutation.py`:
  `470daa8e794dd077cadb3487568af7bc4c8b68b822b871839a8c5abf87e3b8ce`
- `src/powered_v13_recipe.py`:
  `94913b6800de0b17371c22d6c15083af8e2418f555cb074e7266836e6573c0cd`
- seed writer:
  `1d7bf230ad46f57577147684ba336ecbde2b8e95de57a55db6dc395161d16cc1`
- literal writer:
  `bf098259719c1833236b526b6893a4f7337f689f9f086d066a9bf830c0cc86f9`
- focused tests:
  `a1b17cd1574b6591156e107aede727b1a0921aad893be7dd9e3e1af708063fce`
- recipe tests:
  `c51b2427cf6415bc305d4d199dbc2cb27a7e2eb6311f7c65f4a97f0adfb5b73a`

## Remaining boundary

This is not permission to request the real seeds. The repair commit still needs
independent review at its committed hash, and the tokenizer, content-review,
harness, release, corruption, and build-ladder gates must pass before `C0`.
Any failure after real seeds freezes the frame and makes the candidate
ineligible or terminates v13; it is not repaired by resampling.
