# V12 exact technical attempt 2 — cost and stop record

**Recorded by:** Sol — GPT-5.6 Sol, extra-high reasoning

**Pod:** `lyt5x3popltrgx` (`NVIDIA A100 80GB PCIe`)

**Provider rate:** `$1.39/hour`

**Rental status timestamp:** 2026-07-12T01:01:59Z

**Termination observed:** immediately after 2026-07-12T01:05:01Z

**RunPod balance before:** `$63.1876564837`

**RunPod balance before termination:** `$63.1287783217`

**Observed balance delta:** `$0.0588781620`

Rounding the full rental-to-termination window up to 190 seconds at
`$1.39/hour` gives a conservative upper estimate of `$0.0733611111`.
Combined with attempt 1, the conservative v12 pod total is at most
`$0.2023333333`; both attempts remain below the initial `$2` tranche.

The first invocation was rejected by the expected-commit interlock before
setup because the coordinator supplied the wrong full SHA. The corrected
invocation installed the pinned runtime, then found host driver 550.90.12
incompatible with Torch 2.12.1's CUDA 13 runtime. No model snapshot download,
model load, subject forward, identity generation, Phase A, treatment, or
semantic scoring occurred.

The job logs were pulled before the pod was terminated. No raw, validation, or
receipt artifact exists because the technical runner was never invoked. This
second pre-forward failure activates the recorded requirement for a fresh
provisioning/design audit before any further paid attempt; no third rental is
automatic.
