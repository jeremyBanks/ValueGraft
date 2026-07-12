# V12 exact-subject e01 Phase A — bounded launch authorization

**Decision owner:** Sol — GPT-5.6 Sol, extra-high reasoning

**Scope:** one treatment-blind Phase-A execution for frozen case `e01`; no
treatment process or treatment score is authorized

## Scientific basis

The exact technical runner and independent validator passed. The natural
calibration remains correctly labeled `ADVERSE`, but component recomputation
showed source-dependent movement rather than a null: the same-visible-text
`T_g-T_a` margin contrast was +6.5, and the amber transplant recovered 87.0%
of the correct `deny` token's missing oracle log-probability while failing to
suppress `approve` enough to flip the answer. The full interpretation and
decision are in
`results/coherent_canary_v12_technical/coherent-canary-v12-natural-calibration-interpretation_gpt-5.6-sol_20260712T0210Z.md`.

Two independent read-only Codex audits and Sol separately recommended the same
staged action: run e01 Phase A only because it measures full-history
competence, generated target compliance, unchanged-control competence,
carrier support, and fresh damage without exposing treatment. No Fable verdict
is inferred from the earlier capped implementation-review failures; the frozen
preregistration and its completed Fable review already specified the natural
calibration as adverse-but-nongating.

## Frozen scientific binding

- apparatus A7: `8cfc6e2c2dd53505ddc4f6089953f807c72ac306`
- authorization B7: `59afc9a671c106bcf09e3481b3cdf94a140485ca`
- sealed inventory SHA-256:
  `86a26693fb99f0c3ce080399abbc8814f76bcf70d708ecf17b0d1f4291f4a5b1`
- technical raw/report result commit:
  `018afb141efe269c678d88db97a25228dfa81671`
- technical raw SHA-256:
  `ddc5faf7f24dd6ca70bf5833351c38b930d851bb9b0535cb45814dc32413b990`
- technical validator SHA-256:
  `46fa73d864d7f5a06b2053dd38837d918fdcdf9ae88c4709c07070a3d8a81022`
- technical runtime fingerprint:
  `58987aaf5935dcce469ea41d6329dea0d229e2c7652fd2aba7aa553bf58f0607`
- e01 SHA-256:
  `6a2ad7ae0bf094fa5727e76fb710aaeba7bc72082bd93082a5e9a42e09126090`

The production frozen-repository verifier passed at the pre-authorization head
after all technical result-only additions. This authorization file is itself
a result-only addition. The launch binds the exact commit adding this file,
requires it to be an ancestor of `origin/trunk`, and checks out that exact SHA.

## External orchestration boundary

The observed launcher detach defect and technical pull-loop stdin defect are
not repaired in sealed files because that would invalidate A7/B7. Instead, the
scientific runner/validator bytes remain frozen while an ignored orchestration
job invokes them with exact literal inputs and produces a receipt.

- ignored job SHA-256:
  `50925c77fa3381d441e14773167630eefbc2f217b83171647fe510539e92d111`
- ignored receipt-driven puller SHA-256:
  `33924731545fd732a6be5452d0839b82ce617794bff32b2bd991402527dd1c00`
- tracked launcher SHA-256:
  `1c3bf8015a60f9d2c3a6403525ac90192822670280a9f407d1e8dc03871dece1`
- preflight: GREEN, mechanism hash `cd2b1091829e6552`
- focused Phase-A/provisioning tests: `37/37` passed
- ignored job and puller: Bash syntax and ShellCheck passed

The job rejects before model loading unless the observed platform is exactly
`Linux-6.8.0-107-generic-x86_64-with-glibc2.35`, the value bound by the
technical runtime fingerprint. It recreates Python 3.12.11, Torch
2.12.1+cu130/CUDA 13.0, Transformers 5.0.0, the exact `/workspace/hf-cache`
snapshot path, dependencies, frozen release verifier, e01 hash, and technical
report hash before invoking the tracked Phase-A runner. The known tracked
launcher may keep its local SSH process open until the job exits; this is
operationally monitored and does not change the detached remote job or model
execution.

## Spend and lifecycle boundary

- Active RunPod pods immediately before authorization: `0`.
- RunPod balance immediately before authorization: `$62.8738965671`.
- Exact-v12 attempts 1--3 observed balance deltas: `$0.3293793481` at the
  prior snapshot; later provider settlement is handled by the final billing
  endpoint audit.
- This Phase-A-only unit has an operational ceiling of `$0.50`, including a
  rejected platform host. A new host is not automatically retried.
- Require Secure Cloud A100 80GB PCIe, provider CUDA-13 filter, driver
  `>=580.65.06`, at least 80,000 MiB, and rate no higher than `$2/hour`.
- Once the actual rate and rental time are observed, terminate before the
  `$0.50` unit ceiling; at the expected `$1.39/hour`, the full-rental limit is
  about 21.5 minutes.

No pod is retained for analysis. Every receipt-listed artifact is pulled,
hash-checked, committed, and pushed before termination whenever the endpoint
remains reachable. A kernel/platform mismatch, setup/provenance mismatch,
runtime-fingerprint mismatch, runner error, or structural invalidity produces
no retry without a new observed decision.

## Terminal decision after the artifact

- `INVALID_TECHNICAL`: stop; no treatment.
- `ESTIMAND_INADEQUATE`: stop treatment; consider only the already-frozen
  Phase-A-evidence redesign route.
- `PRETREATMENT_PASS`: preserve and review fresh margin and correct-target
  damage, then make a separate treatment-value/spend decision.

This authorization never permits a treatment import, source-row treatment
score, arm grid, case expansion, threshold change, or calibration redesign.
