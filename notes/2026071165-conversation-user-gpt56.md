_This conversation covers the v10 experiment’s release-gating redesign,
validation progress, and a correction to transcript-summary time-boundary
handling. The experiment remains pre-paid and semantically blocked, while
selective note regeneration is now underway._

**Participants:** User and gpt-5.6-sol-xhigh.

**Experiment state.** v10 preserves the original scientific assay and identity,
with Amendment 11 adding only a machine-enforced release overlay: local 0.6B CPU
validation and the paid 30B technical check may run concurrently, but semantic
execution requires independently validated `L AND T`. Fable (`claude-fable-5`)
was consulted as an independent adviser; its conclusion agreed that the proposed
retained-tail arm is invalid for v10 because it violates causal ordering/RoPE
position constraints and lacks matched wrong-history controls. The tail
hypothesis is reserved for a separately designed v11 follow-up, while v10 nulls
must be interpreted narrowly as no detected summary-region channel.

The corrected release layer received independent science and code GO
assessments, with targeted suites passing (reported 50 and 34 tests), unchanged
35-file v10 apparatus inventory, deep ladder reconstruction, remote
pre-inference gating, and post-run receipt validation. Full suite, monitor
self-test, exact pushed-commit review, gate-holder authorization, and preflight
remain required. No paid pod has launched. The local ladder completed synthetic
coverage 7/7 with aggregate discrepancy `0.0`, then entered 12 committed cases;
`c10` subsequently failed the strict equivalence threshold, although the
diagnostic run continued collecting remaining cases. This failure is apparatus
evidence and cannot authorize semantic work.

The earlier estimate was approximately 1.5–2.5 days to a validated result and
reviewed paper if gates passed: roughly 18–26 hours for committed-case checks,
followed by reviews, a 2–5-hour paid technical gate, 3–6 hours for N=6
semantics, possible additional N=12 work, and 6–12 hours for paper synthesis.
Real-agent evaluation remains conditional on finding both a channel and usable
treatment. Intermediate scientific and documentation state should be committed
and pushed frequently.

**Transcript-summary correction.** The six-hour note-duration policy remains
correct: when a conversation exceeds six hours, choose the largest inter-message
gap in the four-to-five-hour window after chunk start, with a fallback before
six hours. The incremental continuation path incorrectly appended to existing
notes without reapplying this splitter; a dry run found a 12.52-hour note that
should become approximately 4.18, 4.63, and 3.39 hours. Rollups may remain
longer. The repair is committed, regression tests pass, and Luna is running
selective regeneration from a frozen input snapshot, affecting only the overlong
source record and rollups whose input hashes change. Concurrent ladder artifacts
are being left untouched.

## Conversation sources

- `019f4f15-7584-7b03-9760-138202ff7c80`
- `019f50f5-5a05-7083-ac9c-f8f62e8a9fc0`
- `019f5110-ae49-7932-8b7b-1a24b5b50165`
- `019f5110-93b5-7a90-8307-9964c9b49787`
- `019f5124-6ba0-7502-adc9-c7738cb8ef5f`
- `019f5124-45a1-70c0-a6e6-9197c6098f50`
- `019f514f-f264-7cc1-96f6-f6071736cb32`
- `019f516d-39e2-7822-aff2-202b8fdb3c90`
- `019f516d-6a84-7f73-99bf-14b84aed4ae7`
- `019f51c1-dc68-7c23-aa37-4e5ddd449938`
- `019f51cc-0694-70f2-8a32-157d7a5bcef6`
- `019f51cb-e1e7-7d90-b159-c7844538ea0d`
- `019f51d6-29ef-7ef0-828f-b5b12236faae`
- `019f51d6-0db5-7180-8b50-47f9997efd8e`
