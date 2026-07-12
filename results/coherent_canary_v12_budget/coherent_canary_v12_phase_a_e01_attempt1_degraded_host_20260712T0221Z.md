# V12 e01 Phase A attempt 1 — degraded host stop

**Recorded by:** Sol — GPT-5.6 Sol, extra-high reasoning

**Pod:** `otv9b2lyptkmh2` (`v12phasea-e01-20260712t0217`)

**Cloud/GPU request:** Secure Cloud, A100 80GB PCIe, provider CUDA-13
filter

**Provider rate:** `$1.39/hour`

**Rental observed:** 2026-07-12T02:18:12Z

**Termination completed:** approximately 2026-07-12T02:21:21Z

The provider kept `desiredStatus=RUNNING` but returned `runtime=null`, an empty
public IP, and no port mappings throughout the 189-second observation window.
No SSH endpoint ever existed. Therefore no bootstrap, repository clone, model
download/load, model forward, generated token, Phase-A score, or artifact was
created. This is an infrastructure non-attempt, not a scientific result.

At roughly three minutes, Sol interrupted the still-armed local launcher. Its
failure trap successfully deleted the pod. A direct lookup then returned HTTP
404 and the active-pod list contained zero entries.

**Balance before:** `$62.8738965671`

**Balance after deletion:** `$62.8432299995`

**Immediate observed delta:** `$0.0306665676`

The balance may settle asynchronously. Charging the entire 189 seconds at
`$1.39/hour` gives a conservative window bound of `$0.072975`. The final
billing-endpoint audit supersedes this immediate reconstruction.

This is the already-documented degraded-host class: an allocated record that
never acquires runtime or an endpoint. The proportional response is
terminate-and-reprovision, not debug or harden the scientific apparatus. A
second host is not automatic; it requires a fresh observed balance/pod check
and a recorded decision within the existing `$0.50` Phase-A-only unit cap.
