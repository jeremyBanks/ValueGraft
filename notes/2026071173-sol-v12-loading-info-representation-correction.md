# V12 loading-info representation correction

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning

## Observed failure

The first local technical execution loaded all 311 model tensors and then stopped
before its first model forward.  The attestation reported
`model loading reported missing_keys: set()`.  Transformers 5.0.0 supplied empty
sets for loading-info fields, while the frozen loader accepted only values equal
to an empty list.  The failed raw artifact is preserved at
`results/coherent_canary_v12_technical/coherent-canary-v12-technical_local-apparatus_20260711T235018354046Z.json`.

This was a representation-compatibility defect in the release gate, not an
experimental observation: no logits, cache intervention, or comparison had run.

## Bounded correction

The loader now requires exactly `missing_keys`, `unexpected_keys`,
`mismatched_keys`, and `error_msgs`, accepting only empty `list`, `tuple`, or
`set` containers for their values.  It rejects an absent diagnostic, every
nonempty diagnostic, and other empty objects such as dictionaries.  The runtime
fingerprint normalizes accepted containers to lists, so recorded provenance
remains canonical.

The apparatus points to a new, separately named authorization file.  The repair
commit is therefore a new apparatus commit, followed by a new additive
authorization commit.  The original authorization and failed execution remain
unchanged in Git history.

## Validation before reauthorization

The loader-focused suite passed `14/14`, including actual Transformers-5-shaped
`set/set/set/list` diagnostics, missing-field rejection, wrong-container
rejection, and nonempty rejection for each diagnostic.  The complete focused
v12 suite then passed `158/158`, with only the same two SWIG deprecation warnings.
No subject-model forward was performed by these tests.
