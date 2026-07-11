# V12 pre-carrier length and content-review failure

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11  
**Status:** additive rejection of e01--e06 revision 1; no subject-model forward occurred

## Length-gate defect observed

The preregistration specifies approximately 1,000--2,000 production-tokenizer
tokens **before the carrier request** for short engineered cases and
4,000--6,000 before the carrier for e04. The first validator incorrectly applied
those bands to the entire authored conversation, including the retained tail
that comes after the carrier.

After commit `60ae601` moved the gate to the literal carrier-request start, the
committed revision-1 fixtures measured:

| case | pre-carrier tokens | required band | result |
|---|---:|---:|---|
| e01 | 670 | 1,000--2,000 | FAIL |
| e02 | 574 | 1,000--2,000 | FAIL |
| e03 | 812 | 1,000--2,000 | FAIL |
| e04 | 1,866 | 4,000--6,000 | FAIL |
| e05 | 665 | 1,000--2,000 | FAIL |
| e06 | 947 | 1,000--2,000 | FAIL |

Thus all earlier tokenizer `MECHANICAL_DRAFT_PASS` artifacts, including the two
ordering-correction artifacts, are nonauthorizing historical evidence. They
verified equality and ordering under their then-current checks but did not
verify the frozen source-history length requirement. Revision 1 of every case is
rejected before inference.

## Independent content-review results

The formal blind singleton review of the unchanged literal histories returned
8 PASS and 4 REVISE histories, with both e04 histories rejected for an
unsupported tag-color assertion and conspicuous omnibus padding, and both e06
histories rejected for an ambiguous flagged/cleared predicate and missing
living/private status. The shared carrier itself passed the blind review.

The separate target-aware/diversity review returned overall FAIL:

- e02 PASS;
- e01/e03/e04/e05 REVISE;
- e06 FAIL;
- cross-case diversity FAIL because e01/e05/e06, and partly e02, share a
  fill-in-the-nouns discourse and retained-tail scaffold.

It also identified necessary-versus-sufficient rule defects in e01/e03/e05,
missing literal predicates and assignment sufficiency in e06, and a
resolution-status conflict between the common carrier and e03/e04/e06.

These review outputs remain valid evidence about revision-1 content even though
the source-length gate independently rejects every case. Review provenance is
under `results/coherent_canary_reviews/reviews/`.

## Binding next step

Create additive revision-2 files for all six cases; do not edit or delete the
reviewed revision-1 sources. Revision 2 must:

1. satisfy the pre-carrier length band, not the total-conversation length;
2. use an unresolved-compatible common carrier;
3. repair every literal rule/predicate issue;
4. replace the shared scaffold rather than paraphrasing it;
5. keep a natural, bounded retained tail after the carrier;
6. rerun mechanical, blind singleton, paired, and diversity review from new
   source hashes.

No model weights have been loaded and no subject-model forward has occurred.
