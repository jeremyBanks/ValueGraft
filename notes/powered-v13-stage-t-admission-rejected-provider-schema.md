# Powered-v13 Stage-T admission rejection — RunPod response schema

**Observed:** 2026-07-12 22:43–22:44 UTC  
**Authorization commit:** `766db12ff12de52f5826238f7446435bef047ab1`  
**Batch:** `stage-t-766db12-20260712T224225Z`  
**Disposition:** `ADMISSION_REJECTED`; no host admitted, no dependencies/model
downloaded, no technical arm run, semantic N remains zero.

## Exact outcome

The bounded lifecycle made its two authorized allocation attempts. Both were
secure-cloud `NVIDIA A100 80GB PCIe` requests at `$1.39/hour`. Attempt 1 was
positively deleted after 29 provider seconds; only after direct 404 and complete
active-inventory absence did attempt 2 begin. Attempt 2 was also positively
deleted after 29 provider seconds. Final lifecycle accounting:

- observed Stage-T provider seconds: **58**;
- observed Stage-T spend: **$0.02239444444444444444444444444**;
- admitted Pod: **none**;
- final provider inventory: **empty**;
- both harvests: unavailable/failed because rejection preceded remote artifact
  directory creation;
- both watchdogs: terminal with direct 404 and active-inventory absence.

A post-deletion provider read at `2026-07-12T22:46:48.869424Z` observed balance
`$57.1090060887` and no active Pods, a balance decrease of `$0.0197885805` from
the prelaunch read. The lifecycle's `$0.0223944…` value is deliberately
conservative because its clock includes positive-deletion verification time.

The raw setup receipt, create responses, Pod states, watchdog records/logs, and
lifecycle record are preserved under
`results/coherent_state_powered_v13/stage-t-admission-rejected_20260712T224225Z/`.

## Reproduced cause

`validate_admission()` required the allocation response to contain the fake-test
shape `cloudType == "SECURE"`. The real RunPod create response omitted that
top-level field. It instead recorded:

```text
machine.secureCloud = true
machine.gpuTypeId = "NVIDIA A100 80GB PCIe"
gpuCount = 1
```

Replaying the persisted create response through the validator reproduced:

```text
V13LifecycleError: allocated host is not secure cloud
```

The provider request itself had been pinned to secure cloud, CUDA 13.0, one GPU,
and the exact A100 SKU. This was a local provider-response schema mismatch, not
observed evidence of a wrong GPU, driver, CUDA runtime, model, or scientific
result. The adapter did not persist the successful SSH/Torch probe outputs
before the later allocation-response check, so those properties are not claimed
as observed for either rejected host.

## Protocol consequence

Section 15 authorized at most two allocation attempts. They are exhausted. The
authorization and setup receipt must not be reused for a third request. Any
retry requires a new, explicit, outcome-transparent design/release that binds
the corrected real-provider schema and carries forward both `$0.0223944…` and
58 seconds against any cumulative cap. The scientific Stage-T canary remains
unrun.
