# Powered-v13 measured N=48 feasibility rejection

**Decision time:** 2026-07-12 after active goal second 24,886
(`6.9128` hours), before the Stage-T hour-`7.8758` stop boundary.  
**Decision:** reject N=48 execution under the powered-v13 apparatus; do not
authorize Stage A or another Stage-T retry.

## Measured gap

- Required independent conversation target: **48**.
- Independently scored semantic fixtures: **1** (`e01`).
- Remaining population-unit gap: **47**.
- Valid powered-v13 technical canaries: **0**.
- Valid powered-v13 semantic cases: **0**.

No conversation-level confidence interval, p-value, equivalence result, or
population upper bound was obtained. This rejection is operational; it is not
evidence for or against write-time K/V information.

## What execution measured

The first Stage-T authorization used two allocations, both rejected and
positively deleted after a local response-schema mismatch. The explicit
one-allocation infrastructure retry corrected that interface and positively
admitted an exact secure A100 80GB host, one GPU, 81,920 MiB, CUDA available,
driver `580.159.04`, and exact Torch GPU identity. It then failed after
`HOST_ADMITTED` and before `SYNCED`; `JOB_STARTED` remained false. Its
one-shot/no-fallback authority is consumed.

Across all Stage-T attempts, conservative lifecycle accounting is 106 provider
seconds and `$0.04092777777777777777777777777`. The latest provider read showed
zero active Pods and balance `$57.1090060887`. Budget was not the limiting
factor. The limiting fact is that the required end-to-end technical path never
completed, so there is no representative unit time, disk, VRAM, artifact, or
loss-bound measurement from which to authorize or project a 48-conversation
batch.

## Why construction stops

The live decision fixed Stage-T construction to hour `7.8758` specifically to
prevent another apparatus spiral while independent N remained one. Two released
attempt paths exhausted their explicit authority without reaching the canary.
The second failure retained only a generic `V13LifecycleError`, not the failing
setup subcommand or stderr, so even its exact cause is not recoverable from the
evidence. Another amendment/retry would revise the stop rule without a measured
scientific unit and would repeat the failure pattern the rule was introduced to
prevent.

Therefore:

1. no further powered-v13 paid allocation is authorized;
2. Stage A and the N=48 treatment release are not authorized;
3. retained-tail and doubled-summary surplus experiments are not reached;
4. the existing scientific conclusion remains limited to the prior one-fixture
   observations; and
5. all raw failure evidence remains part of the audit record.

A future separately authorized study would need a minimal end-to-end canary
that preserves exact failing-command diagnostics before it can make a credible
N=48 cost/time projection. No such future spend or scope is assumed here.
