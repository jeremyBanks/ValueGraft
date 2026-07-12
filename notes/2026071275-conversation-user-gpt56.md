_This conversation covers the terminal validation, failure analysis, and
scientific interpretation of the v12 apparatus, culminating in a formally
stopped protocol and one bounded e01 diagnostic. The current handoff is paper
drafting and final cost/token reconciliation; no further GPU or treatment work
is authorized._

**Participants:** User and gpt-5.6-sol-ultra.

**Handoff State.** Formal v12 is stopped under the written path-control rule:
ULP2 was the mandated stop because both edits were measurable but one moved in
the wrong direction. The sealed implementation continued to ULP4, which is
retained only as implementation-defined plumbing evidence and cannot rescue the
formal branch. Exactly one unchanged e01 treatment run was allowed as a
post-ambiguity, diagnostic-only experiment; it is not formal v12 evidence and
authorizes no e02–e06, aggregate, confirmation, conversation, or live-agent
work. Data collection is complete, no pod is active, and the accepted artifacts
and lifecycle records are committed and pushed.

The exact 30B technical gate and treatment-blind Phase A both passed on a
qualified same-host stack: Qwen3-30B-A3B-Instruct-2507 at revision `0d7cf239…`,
bf16/eager attention, A100 80GB PCIe, driver `580.159.04`, Linux `6.8.0-100`,
Torch `2.12.1+cu130`, Transformers `5.0.0`. Phase A returned
`PRETREATMENT_PASS`; treatment fields were absent. The e01 diagnostic passed
runner, independent-validation, receipt, cross-host fresh-score identity, raw
reconstruction, and local replay checks. Its raw artifact was losslessly
packaged because the complete treatment JSON exceeded the repository hook limit.
Exact K/V tensor values were not retained, only hashes, shapes, traces, and
scores, so future placebo reconstruction requires a new model run.

**Scientific Result.** The strongest e01 pattern was a small value-only effect
at R2/N: `D_focal=+0.174545`, `D_nonfocal=-0.000070`, `SEL=+0.174476`,
`Hplus=+0.710260`, recovering only 0.81% of fresh margin damage and 4.24% of
correct-target log-probability damage. Full K+V failed selectivity and
correct-target movement. The value-only signal shrank to `D=+0.021046` under
schedule P, did not reproduce coherently in R1/R3, and all 31 primary cells
generated the same incorrect outputs (`Ring 3` and `30 days`). All three
norm-matched placebo constructions were unavailable, so placebo count is missing
evidence, not a null result. Natural calibration was adverse
(`rho_green=-0.035714`, `rho_amber=0.297872`). The defensible conclusion is a
weak, uncontrolled, schedule-sensitive mechanistic hint in one explicitly
resolved fixture—not semantic recovery, useful mitigation, or a general channel.

A token-level reanalysis found that the alternate P schedule’s small positive
phrase-average masked a wrong-direction first target token, with the second
token compensating. This further weakens the apparent effect. The final
interpretation must preserve the distinction between forced log-probability
movement and behavioral/task recovery, and must not treat correlated regions,
schedules, or arm decompositions as independent replications.

**Infrastructure/Postmortem.** Secure RunPod hosts varied materially despite the
same container and GPU class: observed drivers included `580.159.03`,
`550.90.12`, and `580.159.04`; the CUDA-13 stack failed on the 550 host. This
established that container pinning alone did not define a reproducible runtime.
The workflow now records and gates driver, GPU, platform, runtime, and exact
commit. Provider selection was also adopted from an AI recommendation without an
explicit reproducibility/qualification review; the durable lesson is to qualify
infrastructure against scientific invariants before committing the workflow. The
packed-MoE loader failure was separately diagnosed as a legitimate Transformers
5 representation conversion (`18,867` raw tensors to `531` packed tensors), not
a wrong checkpoint.

**Accounting.** A final end-to-end audit remains required after paper and review
work stop. Provisional snapshots reported roughly 3.21B Claude tokens, 2.17B
Codex tokens, and 5.38B combined tokens, but source cutoffs differ and both
reconstructions must be rerun after quiescence. RunPod recorded consumption was
approximately `$418.89` in the latest provisional snapshot, with the final
settled total still pending; recent named v12/e01 deltas include `$0.2322` for
exact technical work and `$0.4342` for the diagnostic treatment unit. Separate
actual cash/deposits, provider credits, subscription workload, API-list-price
equivalents, OpenRouter usage, and unknowns; never collapse them into one total.
Preserve exact, reconstructed, estimated, lower-bound, and unknown
classifications, deduplicate resumed/forked sessions, reconcile all RunPod
billing pages, and commit machine-readable plus Markdown ledgers.

**Paper Handoff.** The existing README is unsafe as an incremental base and
should be replaced with a fresh science-first paper. Keep four evidence strata
separate: original synthetic experiments, SWE-Gym/OpenHands trajectory work,
v10/v11 methodological failures, and formal v12/e01. Correct stale claims about
summary-only intervention, compression, content-specificity, placebo validity,
replication, model provenance, and novelty. Include exact artifact
paths/commits/configurations, the ULP2 prose/code conflict, adverse calibration,
zero placebo availability, e01’s diagnostic status, infrastructure postmortem,
and related work. Use conservative language: no equivalence or definitive null
claims, no population inference from v12’s descriptive N=1, and no live-agent
performance claim. The intended form is a fresh academic/paper-style narrative
with a compact postmortem component, not a continuation of the old README’s
positive framing.

The drafting brief has been committed with the corrected evidence hierarchy,
provenance requirements, prior-art positioning, and unsafe-claim list. Claude
Fable 5 has been assigned the initial narrative draft from that bounded brief;
its draft is to be preserved verbatim before scientific editing. Final review
must pass factual/numeric provenance, methods-checklist coverage, and fresh-eyes
readability checks.

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
- `019f548e-e46f-7df2-8fcd-178d2962d024`
- `019f548f-0d8e-76f0-ae6f-06be9df34784`
