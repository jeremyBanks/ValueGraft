# Coherent summary state: Amendment 2 — calibration independence and system identity

**Frozen:** 2026-07-11, after the Amendment-1 0.6B technical ladder and before
any semantic arm outcome from the 30B production checkpoint. This is additive;
the original preregistration, Amendment 1, the failed packed run, and all
intermediate ladder artifacts remain unchanged.

## Why this amendment is necessary

A read-only audit found two pre-outcome specification gaps.

First, the exact-label calibration is not six independent controls. Its authored
context has only two deterministic variants: Label A approved or Label B approved.
The conversation ID selects the label but otherwise does not change the calibration
tokens. In the frozen first-six order the assignments are A/A/A/A/B/A, so the old
“four of six” rule could fire entirely from five duplicate executions of the A
variant. Counting those repeats as independent sensitivity successes would be
pseudo-replication.

Second, the gapped-layout constructor required exact request/header suffix identity
but did not itself assert that the compact system island exactly matched the correct
natural source. Production callers supplied the same system upstream, but the
claimed invariant was not fail-closed at its owning boundary.

Neither issue was discovered from a 30B semantic result. No such outcome exists.

## Revised calibration rule

The calibration remains frozen to the first six conversations and retains the same
two label contexts, exact summary, arms, and target margin. Repeated executions are
technical replications, not independent calibration units.

For each unique label variant separately, compute:

```text
GF_label = mean(G_correct - G_fresh among repeats of that label)
GW_label = mean(G_correct - G_wrong among repeats of that label)
```

The sensitivity calibration fires only if **both unique variants** (A-approved and
B-approved) have `GF_label > 0` and `GW_label > 0`. The one B execution in the first
six is sufficient to evaluate that deterministic variant, but it is not presented
as replicated evidence. Outcomes from conversations 7–12 cannot rescue or dilute
this gate.

The exact-length wrong calibration source is constructed as a frozen token stream.
It must have the same prefix length and summary start as the correct source; all
changed IDs must lie in the declared first-record content block; every other ID must
be exact; no changed ID may be special/chat-structural. The changed positions,
target/replacement IDs, complete prefixes, decoded diagnostic, and equality gates
are persisted before scoring.

## System-island gate

The gapped layout must assert exact token-ID equality between the entire compact
system island and the corresponding prefix of the correct natural source. A changed
system token is an injected failure. This joins the existing exact request/header
suffix, non-overlap, summary-position, and physical-cache gates.

## Zero-gap technical tolerance provenance

The first Amendment-1 float32 0.6B ladder run compared a five-token contiguous
prefill with a 2+3 split execution and observed maximum K/V/logit differences of
approximately `1.30e-4`, `1.73e-4`, and `4.86e-5`, respectively. The original
`1e-5` identity threshold therefore failed before the engineered arm outcomes were
computed. This is a comparison of different prefill kernel shapes, not a copied-row
identity check. A separate zero-gap split tolerance of `5e-4` was then frozen and
two unique ladder runs passed; the initial failure and both passes are committed.

The generated/replay, snapshot/rebuild, explicit-mask, causal-future, and bit-copy
gates retain their stricter identity limits. On the exact 30B bf16 model, the
zero-gap split comparison must still be recorded and must be at most `5e-4`; failure
stops before production semantic scoring. The limit will not be raised in response
to a 30B failure.

## Version identity

Artifacts governed by both additive amendments use:

- `amendment_id = COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2`
- `design_id = coherent-state-gapped-v2`
- `schema = 2`

The prior `gapped-v1` ladder artifacts remain valid evidence of the apparatus
development path but cannot authorize the production run. A new unique 0.6B v2
ladder artifact and a fresh code review are required before paid execution.
