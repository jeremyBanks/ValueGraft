_This conversation covers the recovery of a reproducible exact-technical GPU
workflow, successful technical and treatment-blind Phase-A validation on a
qualified A100 host, and the conservative decision to stop formal v12 after a
preregistration/path-control ambiguity while permitting one exploratory e01
diagnostic. The exploratory treatment run is currently active, with no result or
receipt yet available._

**Participants:** User and gpt-5.6-sol-ultra.

**Handoff State.** Two earlier Secure RunPod attempts failed before scientific
comparison: one due to a false-positive Transformers 5 packed-MoE topology
check, and one due to an incorrect commit SHA followed by an incompatible host
driver. The loader issue was corrected and independently validated: Transformers
5 converts 18,432 raw expert tensors into 96 packed tensors while preserving the
48×128×3 expert grid and total 30,532,122,624 parameters. The infrastructure
issue was confirmed as Secure-pool host variability: observed drivers included
580.159.03, 550.90.12, 580.126.20, and 580.159.04 beneath the same container
image. Future GPU work must gate and record GPU identity, driver, CUDA, cuDNN,
Torch build, dtype, backend, platform, and kernel before paid work.

The durable provisioning fix requests CUDA-13-compatible capacity, independently
requires an A100 80GB PCIe and driver ≥580.65.06, verifies the actual runtime
before model download, and fails closed on incompatibility. Lifecycle safeguards
now include finite API timeouts, explicit cleanup ownership for precreated pods,
no automatic replacement after ambiguous failures, receipt-driven artifact
pulling, exact pinned-head verification, and a provider-clock ceiling. The
intended documentation form is a neutral critical-incident/postmortem note with
evidence, causal limits, prevention rules, and explicit stop conditions. A
further process lesson is that the provider was adopted from an AI
recommendation without a suitability review against
reproducible-scientific-compute requirements.

**Technical Results.** The local 0.6B apparatus initially failed normal-EOS
termination, but after the approved local-only amendment it passed
generated/forced equivalence, repeats, replay, self-replacement, and path
controls; natural calibration was valid but adverse. This result authorized only
bounded exact technical work, never semantic release. The exact 30B technical
gate subsequently passed with independent validation, bit-exact identity
evidence, repeated controls, corrected MoE attestation, and
`semantic_release_eligible=true`. The accepted technical artifact was pulled,
hash-verified, committed, and the pod terminated.

A same-host technical requalification was required after the first Phase-A
attempt used a different kernel from the passing report. The requalification
passed on a healthy Secure A100 PCIe host using the CUDA-13 stack, platform
`Linux-6.8.0-100-generic-x86_64-with-glibc2.35`, Torch `2.12.1+cu130`,
Transformers 5.0.0, bf16, and eager attention. Phase A then ran on the same
host/GPU environment and independently returned `PRETREATMENT_PASS`; treatment
scores were absent. It established exact-subject competence, fresh-compaction
damage, carrier support, and the required focal/nonfocal controls, but it was
not treatment evidence.

The natural calibration remains `VALID BUT ADVERSE`: it failed the preregistered
requirement of at least 50% recovery in both directions, although the amber
transplantation recovered substantial correct-target log-probability and
demonstrated a non-null history-dependent signal. It remains one short,
one-token fixture and does not establish generality, selectivity, value-only
efficacy, or categorical utility.

**Scientific Decision.** Formal v12 is conservatively stopped because the
written path-control rule mandates stopping at ULP2, where one direction moved
incorrectly, while the sealed implementation continued to ULP4 and passed there.
The pre-freeze code review establishes that the implementation behavior was
intentional, but selecting it as the formal result would conflict with the
literal preregistration. Therefore the formal status is `INVALID_TECHNICAL`; the
ULP4 result is exploratory only. No e02–e06, aggregate v12 claim, conversation
study, confirmation claim, or live-agent evaluation is authorized.

One unchanged e01 treatment run was permitted as a predeclared exploratory
diagnostic. It must remain diagnostic-only and formally ineligible, cannot
rescue formal v12, and cannot authorize aggregate expansion. The exact
scientific apparatus, e01 bytes, model revision, technical and Phase-A reports,
runtime fingerprint, and treatment inputs are hash-bound to verifier-valid
commit `cbdfa481fe08de62bf8178d09ca040716d610f21`. The treatment execution
bundle requires 31 primary records plus 3 norm-matched placebo controls,
independent harvesting, a cross-host fresh-score witness, and lossless packaging
of oversized raw output.

**Current Run.** A qualifying pod `t09se1chhqpw5e` was allocated at
`$1.39/hour`, matched A100 80GB PCIe, driver `580.159.04`, 81,920 MiB, and the
CUDA-13 placement requirement. The pinned verifier and runtime checks passed,
the treatment runner launched on GPU `GPU-b2a…`, and the model is resident at
approximately 61 GiB while actively computing. At the latest checkpoint it had
run for about 11 minutes 47 seconds, with no error, receipt, or treatment
result. The independent watchdog remains authoritative, with deletion required
no later than 2,300 seconds after allocation and an early deletion trigger 120
seconds before that bound. No replacement allocation is authorized.

After a terminal artifact appears, the required sequence is receipt-driven pull,
raw reconstruction if chunked, byte/hash verification, independent validation,
commit and push, then pod deletion confirmation. A successful exploratory result
must still be reported as diagnostic-only; any error, invalidity, packaging
failure, or watchdog termination must be preserved without retrying
automatically.

**Accounting.** A final end-to-end accounting audit is required after research,
paper work, and model-assisted review stop. It must separately report actual
cash transactions, RunPod consumed credits, subscription workload, nominal
list-price equivalents, local inference workload, and unknowns, with each
measurement labeled exact, reconstructed, estimated, lower-bound, or unknown.
Current provisional snapshots include roughly 5.38 billion combined Claude/Codex
tokens and about $418.89 in recorded RunPod consumed credits, but these are
live, source-specific snapshots rather than final totals. Claude list-price
equivalents are not cash charges; Codex and Claude usage occurred through
subscriptions. RunPod payment/deposit history, long-context pricing, OpenRouter
activity, subscription invoices, and final post-quiescence token reconstruction
remain unresolved. The final ledger must deduplicate resumed and forked
sessions, exhaust provider pagination, reconcile every pod, and be committed as
reproducible Markdown plus machine-readable data.

## Conversation sources

- `019f4f15-7584-7b03-9760-138202ff7c80`
- `019f5372-cd62-7f82-902f-271830bd73fc`
- `019f5372-f2dc-7460-b579-4d4f73d61305`
- `019f5308-1975-7511-92ae-fd96e8fb151c`
- `019f53c9-f9ec-7b23-b37d-61a7e7f5e285`
- `019f53ca-0b88-7571-94f3-4f3f6c9e3c37`
- `019f53dc-a9b7-77d3-9d7d-3f4e5be9207a`
- `019f53dc-bf4b-7a31-8ce3-27abaffe1708`
- `019f53e9-6702-7851-8c1c-ce8469c2cbe8`
- `019f53e9-775c-7792-9809-e2d5e312ca24`
- `019f53f9-4e57-7070-9782-baca39f3212d`
- `019f53f9-2858-7871-b169-67b9c0ce3bd5`
- `019f53ff-159b-7dc0-a4d1-0ecdb89d74a2`
- `019f53ff-319b-7ab1-a648-1dc772fdfa34`
- `019f5410-3297-7f02-8217-7e86c5e2e5eb`
- `019f5416-e88e-7d73-90eb-f0a984a5031d`
- `019f5425-ca75-7682-a99e-b3fcebb91e95`
- `019f5425-a728-72e3-a5a4-76eb29a6edb1`
- `019f5433-673c-7cb3-bd81-05027d4198e0`
- `019f542d-0027-7893-a3ab-0093b192cac0`
- `019f544b-2625-7860-9dab-b3f30e16aa62`
- `019f544b-363a-7ea0-b556-67e5c40440ef`
- `019f5455-4d90-7be3-9605-2bbe166f0a58`
- `019f5455-6528-71f3-a64c-a190a65348b4`
- `019f545f-f94c-7383-b3bf-a02aaf53493e`
- `019f546b-5486-7a03-9398-f11c1a88bd40`
