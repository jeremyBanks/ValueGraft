# V12 exact technical attempt 3 — cost and stop record

**Recorded by:** Sol — GPT-5.6 Sol, extra-high reasoning

**Pod:** `2k4s986sioi2p2` (`v12tech3-20260712t0149-1`)

**Provider configuration:** Secure Cloud `NVIDIA A100 80GB PCIe`, driver
`580.126.20`, 81,920 MiB; `$1.39/hour`

**Rental observed:** 2026-07-12T01:49:36Z

**Terminal technical result:** 2026-07-12T01:57:06Z

**Termination observed:** approximately 2026-07-12T01:59:55Z

**RunPod balance before attempt:** `$63.1124828884`

**RunPod balance after termination:** `$62.8802442338`

**Observed balance delta:** `$0.2322386546`

The provider balance delta is the closest balance-based billing observation but
may settle asynchronously. Treating the complete 619-second
rental-to-termination interval as billable at `$1.39/hour` gives a conservative
attempt bound of `$0.2390027778`. The raw runner's `$0.1895504044` estimate is
not the provider charge: it uses a conservative `$2/hour` argument but covers
only the runner's 341.19 seconds, excluding provisioning, setup, validation,
artifact transfer, and teardown.

Across exact-v12 technical attempts 1--3, observed balance deltas total
`$0.3293793481`. Combining the previously recorded conservative attempts-1--2
bound (`$0.2023333333`) with attempt 3's full-window bound gives
`$0.4413361111`. These are two alternative accounting views; they must not be
added together. Provider transaction/billing records will supersede the
balance reconstruction in the final whole-project audit.

The exact runner completed with exit 0; the independent validator reported
`PASS` and `semantic_release_eligible=true`. The raw artifact, validation,
receipt, job record, and durable generated/forced-identity checkpoint were
pulled, hash-checked, committed, and pushed before termination. A local
independent revalidation also returned `PASS` (working output retained under
the ignored analysis directory).

After termination, a direct lookup of pod `2k4s986sioi2p2` returned HTTP 404
and the active-pod list contained zero entries. No pod remained active when this
record was written.

This technical PASS does not itself authorize a semantic claim. Its natural
downstream-note calibration was valid but scientifically adverse
(`rho_green=-0.0357143`, `rho_amber=0.2978723`, both below the frozen `0.5`
reference). The preregistration classifies that calibration as reported rather
than a plumbing gate. The decision whether Phase A still has enough expected
information value is therefore being reviewed before any further rental.
