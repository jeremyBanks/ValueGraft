_This conversation covers the transition from a nearly launch-ready v10
ValueGraft experiment to a decisive technical halt after the natural c10
validation case exposed large schedule-dependent bf16 divergence. It also
records a selective repair of overlong conversation summaries and the resulting
redesign/diagnostic state._

**Participants:** User and gpt-5.6-sol-xhigh.

**Experiment state.** v10’s apparatus and release interlock passed independent
reviews, with parallel technical execution prospectively authorized only under a
machine-enforced local-ladder AND paid-technical-PASS condition. The local 0.6B
bf16 eager-CPU ladder reached 7/7 synthetic fixtures with zero discrepancy, but
c10 (8,430 natural-token case) failed dramatically: ordinary chunks
`[4096,4096,238]` versus message-aligned `[23,4096,4096,92,123]` produced K/V
maxima of 16.125/5.125, final-logit discrepancy 0.59375, and continuation
discrepancies up to 1.59375. Inputs, tokens, positions, cache structure, and
artifact hashes matched; layer-0 K/V were identical and divergence began at
layer-1, implicating query-shape-dependent bf16 attention rounding amplified
through depth rather than tokenization or position corruption.

This failure voids v10 semantic authorization and invalidates any paid technical
PASS for v10. No paid pod has launched and experiment spend remains $0. The
earlier 1.5–2.5-day completion estimate is superseded. The remaining cases may
establish whether c10 is representative, but the next priority is a cheap
three-way first-layer diagnostic: compare a 23-token call, the original
4096-token call, and a modified 4096-token call with later tokens changed. Equal
latter outputs with differing 23-token outputs would confirm shape-dependent
rounding; changes in earlier rows would indicate leakage or construction error.
A narrowly preregistered 30B/A100 c10 diagnostic may follow, but it is
diagnostic only and cannot resurrect v10.

**Methodological conclusions.** The proposed native retained-tail arm remains
rejected: it would move state across causal order and RoPE positions, lack a
matched wrong-history control, and require a new v11 design. Any v10 null would
only bound summary-region channels, not downstream retained-tail information.
The correct follow-up is a separately controlled, same-position downstream-tail
study.

**Summary-pipeline repair.** Incremental continuation updates were found to
append without reapplying the six-hour splitter, producing a 13.40-hour note.
The invariant is: split overlong conversations at the largest gap in the
4–5-hour window, falling back before six hours; rollups may span longer. The fix
and regression tests passed. Only the affected source was selectively rebuilt
into three notes spanning 4.18h, 4.63h, and 4.28h; manifests, hashes,
normalization, and rollups were verified, with focused tests 21/21.

**Current handoff.** The archive repair is complete. The experiment is paused at
technical diagnosis: c02 remains healthy and is being preserved as a second
natural-case check, while no paid execution or semantic work may proceed. The
local ladder’s c10 failure is terminal for v10 regardless of later diagnosis;
any replacement assay requires a fresh amendment, identity, apparatus
validation, reviews, and authorization.

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
