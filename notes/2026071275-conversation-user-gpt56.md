_This conversation covers the diagnosis of two failed exact-technical pod
attempts and the discovery that Secure RunPod hosts provide inconsistent NVIDIA
driver environments despite identical container images. The project remains
pre-outcome: no exact subject forward, technical comparison, Phase A, or
treatment result has been obtained._

**Participants:** User and gpt-5.6-sol-ultra.

**Handoff State.** Attempt 1 loaded the 30B checkpoint but failed a
false-positive raw-versus-packed MoE topology attestation before any subject
forward; Transformers 5.0.0 intentionally converts 18,867 raw tensors into 531
packed tensors while preserving 30,532,122,624 parameters. The approved
correction validates the 48×128×3 expert grid, conversion-aware topology,
parameter conservation, loader diagnostics, and three bit-exact content
sentinels. Attempt 2 first failed its commit interlock due to an incorrectly
expanded SHA, then—after correction—failed before model download because the
host exposed driver 550.90.12 while the pinned plain PyTorch 2.12.1 resolved
CUDA 13 and reported CUDA unavailable. Both failures are preserved; the second
pod was terminated. Attempt 1 used driver 580.159.03 and successfully loaded the
model. Observed first-attempt balance cost was approximately $0.0383, with the
conservative rental estimate about $0.129.

The local 0.6B proxy diagnostic is complete but not release-eligible:
generated/forced equivalence, repeat determinism, replay, self-replacement, and
path controls passed; natural calibration was valid but adverse; only strict
normal-EOS termination failed. This authorizes at most the bounded exact
technical gate, never Phase A or treatment. The exact gate remains unchanged and
requires normal EOS, identical repeats, rebuilt generated/forced prefixes, and
bit-exact IDs, positions, K/V, logits, token log-probabilities, and final EOS
candidate evidence.

**Infrastructure Finding.** The effective runtime was not reproducible because
the host kernel driver varied beneath the pinned container and user-space
packages. This is a confirmed source of CUDA initialization inconsistency and a
plausible multiplier of prior contradictory diagnostics, but it does not explain
unrelated checkpoint, fixture, or cache-position errors. Future GPU runs must
record and gate GPU identity, driver, CUDA, cuDNN, Torch build, dtype, and
backend before installation, model download, or forward execution.

The current audit recommends a third bounded rental only after fresh additive
authorization. The preferred path is official `torch==2.12.1+cu126`, which is
compatible with drivers ≥525.60.13 and avoids relying on CUDA-13 forward
compatibility; alternatively, provisioning must request
`allowedCudaVersions=["13.0"]` and independently require driver ≥580.65.06
before any paid work. Never fall back to unfiltered capacity; reject and
terminate incompatible hosts immediately. A five-minute host-vetting limit and
one-hour rental limit were proposed, with no automatic fourth attempt after
another infrastructure failure. No repository files were changed by the final
audit subagents, so implementation status of the provisioning interlock and
CUDA-build amendment must be verified.

**Postmortem Rule.** Provider selection was inherited from an AI recommendation
without a suitability review against reproducible-scientific-compute
requirements. The durable lesson is to qualify infrastructure explicitly—driver
guarantees, CUDA compatibility, host variability, provenance, cost controls, and
preflight behavior—rather than treating an unfamiliar provider recommendation as
validated. The intended documentation form is a neutral
critical-incident/postmortem note with evidence, causal limits, prevention
rules, and explicit stop conditions.

## Conversation sources

- `019f4f15-7584-7b03-9760-138202ff7c80`
- `019f5372-cd62-7f82-902f-271830bd73fc`
- `019f5372-f2dc-7460-b579-4d4f73d61305`
- `019f5308-1975-7511-92ae-fd96e8fb151c`
- `019f53c9-f9ec-7b23-b37d-61a7e7f5e285`
- `019f53ca-0b88-7571-94f3-4f3f6c9e3c37`
- `019f53dc-a9b7-77d3-9d7d-3f4e5be9207a`
- `019f53dc-bf4b-7a31-8ce3-27abaffe1708`
