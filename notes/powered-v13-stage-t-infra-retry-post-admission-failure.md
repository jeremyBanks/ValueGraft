# Powered-v13 Stage-T infrastructure retry — post-admission failure

**Observed:** 2026-07-12 23:27–23:28 UTC  
**Outer authorization:** `5d3f4957236e215049292d72012e6382a9bde762`  
**Immutable scientific payload:** `766db12ff12de52f5826238f7446435bef047ab1`  
**Batch:** `stage-t-infra-retry-1-5d3f4957236e-20260712T232626Z`  
**Disposition:** outer authority consumed; no fallback; technical canary unrun;
semantic N remains zero for Stage T and one fixture overall.

## Observed outcome

The sole authorized allocation created RunPod `t8ymnp8y1ccy4p` at
`$1.39/hour`. The corrected real-provider response gate passed. The admitted
host evidence recorded:

- secure-cloud nested provider attestation: true;
- exact GPU: `NVIDIA A100 80GB PCIe`;
- memory: 81,920 MiB;
- one GPU and CUDA available;
- system-container Torch GPU name: exact match;
- driver: `580.159.04`;
- GPU UUID: `GPU-1250b4a7-23d8-f6da-582f-0006f37e4b59`.

The lifecycle then moved directly from `HOST_ADMITTED` to
`POST_ADMISSION_FAILURE`. It never recorded `SYNCED` or `JOB_STARTED`. The
persisted error type is only `V13LifecycleError`; the bounded-command helper did
not preserve the failing subcommand or stderr. Therefore the exact setup
subcause is **unknown**. No model load, technical arm, or scientific result is
claimed.

The OS-owned watchdog attempted harvest, accepted DELETE, observed direct 404,
and observed complete active-inventory absence. Its terminal status was
`TERMINATED_HARVEST_FAILED` because no remote harvest was available. A later
provider read at `2026-07-12T23:28:37Z` again observed zero active Pods, balance
`$57.1090060887`, and spend limit `$80`.

## Accounting and authority

The attempt lasted 48 provider seconds. Conservative cumulative Stage-T
accounting, including the first schema-rejection run, is now:

- provider seconds: **106**;
- lifecycle spend: **$0.04092777777777777777777777777**;
- incremental retry lifecycle spend: **$0.01853333333333333333333333333**;
- observed provider-balance change since retry preflight: **$0** at the stated
  post-delete read (billing may not yet have posted).

The canonical consumption record is preserved with SHA-256
`bf6b168fbcfed94d3386fe9821fede0a964885b97db906226ba3ba63b454f22e`.
It binds the full outer/inner authorization, manifest, and receipt hashes,
carry-ins, batch, and session. It permanently prevents another invocation of
this authority. The protocol authorized no fallback and no second
infrastructure retry.

## Consequence

This is an operational feasibility failure, not evidence about K/V grafting.
The response-schema repair was positively exercised; the remaining setup path
failed without exact diagnostic provenance. Under the Stage-T stop rule,
further apparatus construction and paid retry are closed. The raw session,
both receipts, consumption record, provider state, and post-delete snapshot are
preserved under
`results/coherent_state_powered_v13/stage-t-infra-retry-post-admission-failure_20260712T232626Z/`.
