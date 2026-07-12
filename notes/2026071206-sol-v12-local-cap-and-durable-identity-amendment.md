# V12 local cap result and durable identity amendment

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning

## Observed result

The third local technical execution was the first to reach subject-model
inference.  Qwen3-0.6B did not emit a member of the frozen EOS set within the
64-content-token cap on the identity prompt frozen for the 30B exact subject.
The runner raised `generated identity branch did not stop normally` after 8.64
seconds.  The raw error is preserved at
`results/coherent_canary_v12_technical/coherent-canary-v12-technical_local-apparatus_20260712T000134494813Z.json`.

This observation says that the local proxy failed the frozen termination
predicate.  It does **not** establish generated/forced cache inequality: the
strict function checked termination before it compared content rows, and the
exception prevented the generated 64-token stream and forced branch from being
written to the raw artifact.  No semantic treatment outcome was observed.

## Integrity defect and bounded correction

Discarding the generated stream on a failed predicate violated the repository's
SAVE EVERY RENDER rule.  The corrected path:

1. constructs and retains both generated and forced branch records before
   applying the strict predicate;
2. independently compares prefix tokens/calls/positions/K/V/logits, content
   tokens/positions/log-probability bits/K/V rows, and the final unappended
   candidate before checking normal termination;
3. records an ordinary identity `FAIL` with its evidence instead of converting
   that scientific/technical outcome into a top-level exception;
4. runs the remaining deterministic-repeat, self-replacement, path-control, and
   natural-calibration diagnostics; and
5. keeps the independent validator's normal-EOS requirement in the strict gate.

The validator now reports generated/forced equivalence separately from the full
identity predicate.  A cap hit can therefore show equivalence evidence while the
strict identity check and overall technical report remain `FAIL`.

## Local-proxy advancement decision

The local model is an apparatus proxy and has no inferential role.  The fixture
it failed explicitly names the 30B exact subject.  Two independent bounded
reviews and my own analysis agreed that adapting a new, easier local prompt after
seeing this outcome would be less clean than retaining the original prompt and
reporting the proxy's termination limitation.

Accordingly, the local cap hit alone does not substitute for or relax any exact
gate.  It may permit the bounded **exact technical gate only** if a rerun shows:

- the independent generated/forced equivalence check passes;
- both local repeats are identical;
- normal termination is the sole identity failure;
- every remaining local technical diagnostic is present and passes its plumbing
  predicate (natural calibration may retain its preregistered adverse status).

This exception does not authorize exact Phase A, treatment, semantic release, or
any claim.  The exact Qwen3-30B-A3B-Instruct-2507 run must use the original
fixture and cap, emit normal EOS, repeat bit-exactly, pass generated/forced
equivalence, and pass the full exact technical validator before any later exact
stage.  The amendment changes only how a noninferential local proxy failure is
preserved and whether it blocks trying the strict exact gate.

## Validation before reauthorization

Four focused regression suites passed `31/31`, covering strict cap rejection
after equality checks, durable branch evidence, continuation of later technical
diagnostics, and independent equivalence-pass/normal-stop-fail classification.
The complete focused v12 suite passed `162/162`, with only the same two SWIG
deprecation warnings.  These tests used deterministic fake models; no additional
subject-model forward was performed.

## Fable review and disposition

Claude Fable 5 independently reviewed the observed artifact, preregistration,
fixture, amendment, runtime, runner, and validator.  Its full assessment is in
`notes/2026071175-fable-v12-local-cap-amendment-review.md`; it found no blocker
and returned **GO** under the four stated local conditions, while confirming
that the exact gate remains strict.

Before freeze I removed the advisory string-matched
`equivalence_checks_completed` flag it identified as fragile; advancement rests
only on the independent validator pair.  I also moved durable construction of
the completed generated record ahead of the forced branch and now preserve that
record if ordinary forced-branch validation fails.  Unexpected comparison or
forced-branch exceptions are recorded as an identity apparatus `ERROR`, then
promoted to a top-level error only after the available identity evidence has
been attached to the document.  The remaining limitations Fable listed are
accepted and disclosed: no real-model normal-EOS path has yet passed, the
validator cap is a frozen literal, and the advancement exception remains an
explicit procedural decision rather than a code path that could weaken release.
