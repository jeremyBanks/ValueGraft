# Powered v13 independent bound reimplementation audit

**Date:** 2026-07-12  
**Status:** PASS on seven synthetic golden cases and malformed-input checks; no
treatment outcomes inspected.  
**Baseline commit:** `3caccd57b69a80b31ba3af6cfad709dc54b0e8f5`

## Scope and separation

`scripts/recompute_powered_v13_bound_independent.py` is a stdlib-only
recomputation of the fixed-N v13 primary rule. It does not import
`powered_v13_stats`, NumPy, or SciPy. It independently collapses two C-origin
renders, selects ranks 1--6 from each of eight strata, clips both cells to
`[-0.5, 0.5]`, applies the per-cell alpha `.02` Hoeffding UCB, applies the
alpha `.01` zero/nonzero responder rules, and emits the numerical conclusion
label. Its CLI reads render rows as JSON and writes deterministic canonical
JSON.

The comparison test invokes that script in a separate process, constructs the
same projection from `src/powered_v13_stats.py`, and requires byte-for-byte
canonical JSON equality. An AST check limits the independent script's imports
to named Python standard-library modules and rejects any production-module
import.

The first dyadic-only comparison passed but was not sufficient: arbitrary
binary64 rows could differ by one ULP because production used NumPy reduction
while the independent path used Python arithmetic. Before retaining the audit,
both implementations were changed to freeze the same explicit arithmetic law:
`STRATA` order, then eligible rank/case ID, then `CELLS` order, with
`math.fsum` for every render, fixture, stratum, and final mean. The production
and independent alpha ledgers now prove `1/50 + 1/50 + 1/100 = 1/20` with
`Fraction` before emitting the protocol's float fields. Both paths also reject
JSON booleans as numeric outcomes or eligible ranks.

## Golden results

All numeric differences were exactly zero. Hashes are SHA-256 over canonical
UTF-8 JSON including its final newline.

| Case | Collapsed input / selected N | Full/value UCB | Responders / tail UCB | Numerical label | Canonical SHA-256 |
|---|---:|---:|---:|---|---|
| constant zero | 48 / 48 | `0.20186688594189123` / `0.20186688594189123` | `0` / `0.09148242434831322` | `JOINT_BOUND_RESOLVED` | `d92573326debf06a6e3eba32e53efa5034a991e2a0a19b9f4caa52552fc47178` |
| one 5-nat responder | 48 / 48 | `0.21228355260855788` / `0.20186688594189123` | `1` / `0.2398550737398722` | `CLIPPED_MEAN_BOUND_ONLY` | `1bf6ba2ecba896a74578edb17e6b60152f8de6f9f3e9efb42dab35b3cd8d0159` |
| mixed strata | 48 / 48 | `0.38871584427522454` / `0.21130699010855788` | `1` / `0.2398550737398722` | `CELL_BOUND_ONLY` | `cde740a1cd8291884997c127cfbd4e1717676c51a19bf90be0e863305ba678d4` |
| row-shuffled mixed | 48 / 48 | same as mixed | same as mixed | `CELL_BOUND_ONLY` | `cde740a1cd8291884997c127cfbd4e1717676c51a19bf90be0e863305ba678d4` |
| rank overshoot | 64 / 48 | `0.20186688594189123` / `0.20186688594189123` | `0` / `0.09148242434831322` | `JOINT_BOUND_RESOLVED` | `efc9c5672dfdf9ccd080eabb9737909584e9680db69959f1fb4f538877ce8031` |
| non-dyadic | 48 / 48 | `0.2951307236899457` / `0.21352439101127596` | `0` / `0.09148242434831322` | `JOINT_BOUND_RESOLVED` | `e64557104e592b39e9885dea5f6929a27c09adc85631c78b15143f53742b7825` |
| catastrophic cancellation | 48 / 48 | `0.3685335526085579` / `0.20186688594189123` | `32` / `0.8856884070732055` | `CELL_BOUND_ONLY` | `022a6a1df68e5593327720523978ad4d906cdd1c72c2eb7961e6302a70f13fdc` |

The mixed case contains independent dyadic render perturbations, one raw value
below the lower clipping limit, one above the upper limit, and an exact `0.5`
boundary value. It makes the full-KV cell unresolved and the value-only cell
resolved, thereby exercising `CELL_BOUND_ONLY`. Shuffling every render row
left the complete canonical output byte-identical.

The overshoot case supplies ranks 1--8 in every stratum and puts 5-nat effects
only in ranks 7 and 8. No rank above six entered the selected case IDs. After
removing the diagnostic `n_collapsed_input` field (`64` rather than `48`), its
entire projection was byte-identical to the constant-zero projection.

The non-dyadic and catastrophic-cancellation vectors were selected so the old
NumPy mean differs from the frozen `math.fsum` mean. They therefore fail if
either path silently reverts to the prior backend reduction while still
requiring byte agreement under the new arithmetic convention. Separate
corruption cases verify that boolean outcome and rank fields fail closed in
both paths.

## Verification

Command:

```text
PYTHONPATH=src .venv/bin/pytest -q tests/test_powered_v13_independent_recompute.py tests/test_powered_v13_stats.py
```

Observed result:

```text
.........................                                                [100%]
25 passed in 0.56s
```

Runtime was Python `3.12.11` with pytest `9.1.1`.

Artifact hashes at verification time:

- production core: `2c468df1ea5083d93c999b17b8074a8de2afc8f1e0dccee496f717d8e99e6d18`;
- independent script: `0b3de1dee0661d1bc853243f431aa91b09f207e9556779b1b8cc5028f33858c7`;
- comparison test: `c6f4c7fce9711ba09791b027d13bc23a483d8f1f5bfc4a30ad3d39e31eaa5356`.

The preregistration was being corrected concurrently by another workstream, so
this audit does not claim or bind a transient working-copy hash for it.

## Boundary

The independent output's `numeric_conclusion_label` covers only
`JOINT_BOUND_RESOLVED`, `CLIPPED_MEAN_BOUND_ONLY`, `CELL_BOUND_ONLY`, or
`BOUND_NOT_RESOLVED_AT_N48`. Higher-precedence pre-treatment/technical labels,
`VALUE_PLACEBO_COMPLETE`, and the portability suffix depend on external
artifacts and are deliberately not inferred by this numerical recomputation.
The production statistical core itself exposes resolution booleans rather than
a terminal label string; the comparison derives the corresponding numerical
label from those booleans under the frozen protocol precedence.

The production statistics were intentionally changed only to freeze arithmetic,
make the alpha assertion exact, and fail closed on booleans. The preregistration
was not modified by this audit.
