# V12 pre-freeze review disposition

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning effort

**Date:** 2026-07-11 (America/Toronto)

**Subject-model forward performed:** No

## Verdict

Claude Fable 5 completed the bounded scientific review in
`notes/2026071170-fable-v12-prefreeze-scientific-review.md`. It found no
blocking defect in the estimand, technical/Phase-A gates, physical separation,
schemas, float32 formula recomputation, stopping logic, or model/configuration
binding. It recommended **READY** for the local 0.6B integration and **READY**
for the bounded exact-model gates after the preregistration is frozen.

I independently accept that verdict. Before freezing, I resolved three of its
four nonblocking observations:

1. the aggregate analyzer now computes the value-only mitigation utility rule
   over cases with positive Phase-A margin and correct-target damage;
2. it now reports explicit-resolution and unstated-derivable case subtypes;
3. the Phase-A validator now requires the technical report to bind the current
   fixed carrier-token evidence.

The remaining observation is intentionally retained as a known abort mode: the
harvester checks the persisted float32 reduction against the per-token float32
values using a small tolerance. If an unusually long target exposes a legitimate
reduction-order discrepancy beyond that tolerance, the case aborts for review;
it is not silently reclassified as a scientific result. Frozen engineered
targets are short, so this is not a pre-forward blocker.

## Observed validation

Before the final review dispositions, the complete focused v12 suite passed
`157/157` tests. The affected Phase-A and aggregate suites then passed `25/25`
after the dispositions. A final complete focused suite is still required before
the freeze commit.

**Final observation:** after all dispositions and budget corrections, the
complete focused v12 suite passed `158/158` tests with only two SWIG deprecation
warnings. The pre-freeze static gate is therefore complete.

No local or paid subject-model forward has occurred. Paid coherent-state compute
remains `$0`.

## Review-cost incident

The Fable CLI exceeded both nominal caps while caching the bounded file set. Its
two invocations reported `$3.243552` and `$2.425180`, adding `$5.668732` of
provider-equivalent usage estimates. The working conservative total is therefore
`$42.532574`, unverified as incremental cash billing. This overrun is recorded
in `.sol-v4/spend-ledger.md`; no further external-review calls are authorized
before data collection. The exact-model phase must begin with the already-frozen
`$2` gate tranche and may extend only if the observed cost still fits the
owner's total ceiling and the retained synthesis reserve.
