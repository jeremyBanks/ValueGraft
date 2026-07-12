# Powered-v13 Stage-T infrastructure retry 1

**Status:** **DRAFT — NO PROVIDER ALLOCATION AUTHORIZED**

This amendment is an outcome-transparent outer infrastructure retry boundary.
It does not change the scientific canary. The only scientific payload remains
the exact original Stage-T authorization commit
`766db12ff12de52f5826238f7446435bef047ab1` and its canonical manifest
`results/coherent_state_powered_v13/releases/powered-v13-stage-t-manifest_20260712T2240Z.json`
(SHA-256 `c774d65fa9bf0a0ab5f9bdbd9cc09ea43da04ba7d78f4ec9c5cbb07a583ae98f`).
The retry has semantic N=0 and cannot contribute an inferential observation.

The original authorization exhausted two allocations before admission because
the real RunPod response represented secure-cloud identity under
`machine.secureCloud` and `machine.gpuTypeId`, while the local validator required
a fake-test top-level `cloudType`. The deterministic schema repair is commit
`79ab77f10d743e97153af8485532bcf5592092ef`. This amendment authorizes no
scientific or protocol change. The one-attempt controller repair is exact commit
`ffc90ce6e39b7fd08ad3dd77cfb7fb51fe53f6d2`; the retry must invoke it with
`max_allocation_attempts=1`.

If a later exact two-path authorization changes only the literal status line
above to `INFRA_RETRY_AUTHORIZED`, it may authorize exactly one additional
provider allocation. There is no fallback allocation and no second retry.
Carry-in is 58 provider seconds, conservative lifecycle spend
`$0.02239444444444444444444444444`, and observed provider balance delta
`$0.0197885805`. Cumulative ceilings remain 3,300 provider seconds and `$1.50`.
At the pinned `$1.39/hour` request, the one retry is additionally bounded by
3,242 seconds and `$1.251772222222222222222222222`.

The rejection incident is bound by
`notes/powered-v13-stage-t-admission-rejected-provider-schema.md` (SHA-256
`0388d036b1dcd51ea7ee8575e967751f626259c1576d2d4d15ef1af2879ed257`).
The raw evidence directory is
`results/coherent_state_powered_v13/stage-t-admission-rejected_20260712T224225Z/`.
Its exact tracked files and hashes are fixed by the canonical inventory contract
and the retry manifest; they may not be omitted or replaced.

The authorization child must have the static root as its only parent and change
exactly two paths: this amendment by the one status-line substitution, and one
new canonical retry manifest. A fresh create-if-absent receipt, detached exact
authorization checkout, clean tree, and exact manifest/inventory verification
are required before an allocation. The original setup receipt is exhausted and
cannot authorize this retry. The outer retry authorization is the sole authority
for creating a provider allocation. It permits creation and use of one fresh
inner Stage-T receipt solely to bind the immutable `766db12` scientific payload
at the existing subject gate; neither that inner receipt nor the exhausted
original authorization independently authorizes any allocation.

The paid entry point consumes the outer authorization exactly once before it
delegates, even if the delegate later fails. It writes one canonical,
create-if-absent, fsynced record keyed by the full outer authorization hash
under the fixed account-home state root (resolved from the operating-system
account database, not the caller's `HOME` environment)
`.local/state/valuegraft/powered-v13-stage-t-infra-retry-1-consumed/`.
An existing record permanently rejects reuse. The session directory basename
must equal the authorization-bound primary batch ID.
