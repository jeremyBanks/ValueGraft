# V12 e01 — same-host technical requalification plus Phase A

**Decision owner:** Sol — GPT-5.6 Sol, extra-high reasoning

## Decision

Authorize one healthy Secure A100-80GB-PCIe pod to run:

1. the complete frozen exact-v12 technical gate;
2. independent technical validation and durable artifact preservation;
3. after that exact report is pulled, hash-checked, committed, and pushed,
   e01 Phase A in a distinct Python process and fresh repository clone on the
   **same pod and GPU UUID**;
4. independent Phase-A validation, pull, commit/push, and immediate deletion.

No treatment is authorized.

## Why requalification is the proportional correction

The prior passing technical report literally binds Linux kernel
`6.8.0-107`. A healthy location-constrained host exposed kernel
`6.8.0-100` and was correctly rejected before model work. Kernel identity is
part of the implemented runtime fingerprint even though the scientific subject
contract fixes model/revision, bf16, eager attention, and a capable NVIDIA GPU,
not one Linux patch release.

Renting successive hosts to find one historical kernel is now lower-value than
running the full technical gate on the actual healthy stack. The complete gate
rechecks exact weights/topology/content, generated/forced identity,
deterministic repeats, all self-replacements, bidirectional path sensitivity,
natural calibration, persistence, and independent validation. Phase A then
requires byte-identical runtime fingerprint and the same GPU UUID. This is
stronger evidence for the executing host than weakening or deleting the
platform check.

The prior PCIe technical PASS and natural `ADVERSE` result remain preserved and
reported. The new run is not selected because of its calibration outcome; any
valid new calibration is reported under the same frozen interpretation, and a
technical failure stops before Phase A.

## Frozen scope and orchestration

- exact model/revision: `Qwen/Qwen3-30B-A3B-Instruct-2507` at
  `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`;
- GPU: exact Secure `NVIDIA A100 80GB PCIe`, CUDA-13 provider filter, driver
  `>=580.65.06`, at least 80,000 MiB;
- scientific technical job: tracked
  `scripts/job_coherent_canary_v12_technical.sh`;
- technical raw and independent report must satisfy the existing strict PASS
  and `semantic_release_eligible=true` rules;
- e01 bytes remain SHA-256
  `6a2ad7ae0bf094fa5727e76fb710aaeba7bc72082bd93082a5e9a42e09126090`;
- Phase A uses the existing tracked runner/validator, with an ignored
  orchestration job hard-binding the newly committed technical report,
  platform string, exact launch head, and GPU UUID;
- technical and Phase-A processes use separate model loads and fresh repository
  clones while sharing only the pinned model cache and physical pod/GPU;
- every render/generation remains inside its raw artifact.

Only a data center other than failed `CA-MTL-3` may be requested. A read-only
positive stock query is required immediately before a request; received HTTP
500/no-ID responses may be retried up to three times, but the first allocation
ends the request sequence. Any degraded/incompatible/ambiguous allocated host
is deleted and stops this route.

## Spend and provider-clock limits

Three Phase-A provisioning windows conservatively total `$0.1783833333`.
Based on the observed passing technical rental (~10.3 minutes) and the audited
Phase-A estimate (~7--8 likely, ~10--11.25 conservative), a healthy same-host
sequence should add about `$0.4--$0.5` at `$1.39/hour`.

Set a hard `$0.80` ceiling for the entire e01 provisioning/requalification/
Phase-A unit, including the three prior windows. The new pod therefore has at
most about 26.8 billable minutes at `$1.39/hour`; operational targets are
tighter:

- healthy endpoint/admission by `+3:00` from rental;
- technical terminal artifact by approximately `+10:30`, hard stop/salvage at
  `+12:30` if absent;
- technical artifacts committed/pushed and Phase A relaunched by about
  `+15:00`;
- Phase-A terminal artifact expected by `+22:00`;
- begin final salvage by `+24:00` and confirm deletion by `+26:00`.

Rate above `$1.39/hour`, wrong GPU/data center, technical invalidity/failure,
artifact-integrity failure, inability to bind the same GPU UUID, Phase-A
structural error, or provider-clock deadline stops without another allocation.
No pod remains running for analysis or writing.

## Terminal science rule

- technical not strict PASS: no Phase A;
- Phase A `INVALID_TECHNICAL`: stop;
- Phase A `ESTIMAND_INADEQUATE`: stop treatment;
- Phase A `PRETREATMENT_PASS`: preserve and review; make a separate treatment
  decision.

Nothing here changes a stimulus, threshold, target, arm, treatment boundary, or
claim.
