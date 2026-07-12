# V12 exact technical attempt 1 — cost and stop record

**Recorded by:** Sol — GPT-5.6 Sol, extra-high reasoning

**Pod:** `rv7x3vwmfyvpah` (`NVIDIA A100 80GB PCIe`)

**Provider rate:** `$1.39/hour`

**Rental status timestamp:** 2026-07-12T00:37:52Z

**Termination observed:** 2026-07-12T00:43:26Z

**RunPod balance before:** `$63.3160022124`

**RunPod balance after termination:** `$63.2777396809`

**Observed balance delta:** `$0.0382625315`

The provider balance delta is the closest observed billing value but may update
asynchronously.  Treating the entire 334-second status-to-termination window as
billable at `$1.39/hour` gives a conservative attempt upper estimate of
`$0.1289722222`.  Both values are far below the initial `$2` actual-pod tranche.

The exact runner loaded the frozen 30B weights and then failed loader attestation
with `loaded parameter names/shapes/dtypes differ from weight inventory`.  It
performed no subject-model forward, identity generation, Phase A, treatment, or
semantic scoring.  Raw, independent validation, job log, and receipt were
pulled, receipt-hash verified, and committed before termination.
