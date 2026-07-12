# Critical finding: host-driver drift and unvalidated provider selection

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning

**Owner contribution:** Jeremy Banks identified the likely higher-order effect
on diagnostic churn and supplied the provider-selection provenance.

**Recorded:** 2026-07-11/12 EDT/UTC

## The observed fact

Two RunPod **Secure Cloud** allocations requested the same GPU type
(`NVIDIA A100 80GB PCIe`) and the same container image
(`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`), but exposed
different physical-host drivers:

| Allocation | Host driver | Pinned Torch runtime | Observed outcome |
|---|---:|---:|---|
| exact v12 attempt 1 | `580.159.03` | CUDA `13.0` | CUDA available; exact 30B weights loaded |
| exact v12 attempt 2 | `550.90.12` | CUDA `13.0` | CUDA unavailable; stopped before model download/forward |

The container image did not pin the kernel driver. This means the project had
been treating a partially specified runtime as though it were reproducible.
The fact that both allocations were Secure, not Community, rules out the easy
explanation that only community hardware was heterogeneous.

RunPod's official Pod API supports `allowedCudaVersions`; if absent, any CUDA
version is acceptable. NVIDIA's CUDA 13.0 release notes require Linux driver
`>=580.65.06`. These are precisely the two conditions the previous provisioner
failed to encode:

- [RunPod create-Pod API: `allowedCudaVersions`](https://docs.runpod.io/api-reference/pods/POST/pods)
- [NVIDIA CUDA 13.0 minimum driver table](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)

## Why this is a major learning

Host-driver drift is a **proven** source of intermittent infrastructure
behavior. It can make identical repository bytes and container tags alternate
between working and failing. Repeated inconsistency can also create a secondary
human/agent cost: diagnostics branch, assumptions are revisited, attention is
fragmented, and important details are forgotten while the team is spun through
apparently contradictory evidence. The owner correctly identified this as a
plausible multiplier of the week's loose focus and confusion.

The causal boundary matters. This finding does not retroactively explain the
known wrong-checkpoint incident, periodic five-token fixture, malformed
counterfactual histories, or cache-position bugs. Those have direct causes.
Nor does driver variation alone void an old result without evidence that its
runtime differed materially. The defensible conclusion is narrower and still
important: the environment was under-specified, this caused at least one paid
failure, and it may have amplified other debugging loops.

## Provider-selection postmortem

The owner recalls choosing RunPod after asking Claude for a lightweight prepaid
GPU-rental recommendation. The provider was not independently familiar to the
owner and was not subjected to a formal suitability or credibility review
before the project built around it. Agents likewise did not stop to translate
the experiment's requirements into a provider-qualification checklist.

This does **not** establish that RunPod is disreputable or was a bad overall
choice; provider credibility has not been adjudicated here. The process defect
exists regardless: an AI recommendation was treated as sufficient selection
evidence. For scientific infrastructure, a recommendation should only generate
a candidate. Qualification must independently verify reproducibility controls,
host/runtime filters, artifact egress, billing behavior, availability,
observability, and failure handling.

## Correction selected

The frozen numerical environment remains Torch 2.12.1 / CUDA 13 / Transformers
5.0.0 / bf16 / eager. A CUDA-12.6 build of the same Torch release was identified
as a viable alternative, but it would change compiled CUDA and math libraries.
Because RunPod officially supports a CUDA-version admission filter and attempt
1 already loaded under native CUDA 13, the lower-risk correction is to preserve
that stack and fix admission:

1. `src/pod.py` sends `allowedCudaVersions: ["13.0"]`.
2. Before bootstrap, the launcher reads the actual GPU name, driver, and memory.
3. `src/pod_admission.py` requires the exact A100-80GB PCIe, driver
   `>=580.65.06`, and at least 80,000 MiB.
4. Incompatible allocated hosts are terminated immediately; there is no
   unfiltered fallback.
5. The exact wrapper derives `SC_EXPECTED_COMMIT` with `git rev-parse HEAD`,
   eliminating the separate hand-expanded-SHA failure, and bounds admission at
   three attempts.
6. Pinned Torch's own CUDA initialization remains the authoritative second
   runtime check before model download.

Claude Fable 5 independently recommended preserving the pinned CUDA-13 stack
and pre-vetting driver 580+ rather than changing Torch/CUDA or injecting
forward-compatibility libraries. Its full audit is preserved at
`results/coherent_canary_reviews/reviews/v12_attempt2_provisioning_audit_claude_20260712T0106Z.md`.

## Standing lesson

The effective experimental environment is:

```text
weights + tokenizer + code + Python packages + compiled CUDA libraries
+ physical GPU + host kernel driver + execution backend
```

Every term must be selected, checked, and recorded. "Same GPU" and "same
container" are insufficient. Provider choice and per-host admission are part of
methodology, not incidental operations.
