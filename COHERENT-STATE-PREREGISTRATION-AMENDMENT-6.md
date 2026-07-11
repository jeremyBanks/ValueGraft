# Coherent summary state: Amendment 6 — exact committed-case position ceiling

**Frozen:** 2026-07-11, during local implementation review of Amendment 5 and
before any execution of the exact 30B checkpoint under eager attention. This is
an additive correction. No 30B eager technical or semantic outcome exists.

## Why the ceiling must change

Amendment 5 added exact-token schedule fixtures for all twelve committed source
files but incorrectly retained `8224` as the maximum technical logical position.
That value covers the synthetic logical-gap fixture only: its final token is at
`8223` and its common continuation is at `8224`.

The already-frozen production tokenizer and request give the following exact
committed-case prefix lengths, which are also the logical positions of their
common q=1 continuations:

| Case | Continuation position |
|---|---:|
| c10 | 8430 |
| c02 | 8385 |
| c01 | 8855 |
| c04 | 8595 |
| c07 | 8600 |
| c11 | 9381 |
| c05 | 8556 |
| c09 | 9195 |
| c06 | 8876 |
| c12 | **9509** |
| c08 | 8525 |
| c03 | 8913 |

These counts were obtained from the exact production-tokenizer construction
already preserved in the committed external-donor validation artifact. The v6
gate must recompute them from the fingerprinted source files and tokenizer; the
table is a frozen expectation, not a substitute for that recomputation. A count
or order mismatch fails closed.

## Corrected requirement

`max_technical_logical_position` is exactly **`9509`**. Model context coverage
must include logical position `9509`. The synthetic logical-gap continuation
remains exactly `8224`; its fixture is otherwise unchanged. Every committed-case
fixture retains its exact token stream, partitions, per-layer measurements,
and unchanged `5e-4` limit from Amendment 5.

No tolerance, arm, estimand, donor, case, partition, stopping rule, or claim
scope changes. The correction only makes the declared context bound cover the
fixtures Amendment 5 already required.

## Version identity

Artifacts governed by all six additive amendments use exactly:

- `amendment_id = COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4-5-6`
- `design_id = coherent-state-gapped-v6`
- `schema = 2`

No v5 implementation, ladder, review, donor artifact, technical attempt, or
semantic artifact can authorize v6. A fresh v6 ladder, production-tokenizer
validation, full tests, independent code/science review, clean launch commit,
and the complete Amendment-5 attestation contract are required before paid
execution.
