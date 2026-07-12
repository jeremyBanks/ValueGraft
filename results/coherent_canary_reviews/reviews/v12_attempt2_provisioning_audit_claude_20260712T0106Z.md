# V12 attempt-2 provisioning audit — fresh strategic review

**Reviewer runtime model identity:** `claude-fable-5` (Claude Fable 5, Anthropic), fresh session, read-only review of the files listed below.

**Inputs read (only):** AGENTS.md; notes/2026071177-sol-v12-transformers-moe-loader-correction.md; both v12 budget cost/stop records; the attempt-2 infrastructure-failure artifact; scripts/job_coherent_canary_v12_technical.sh; scripts/launch_pod.sh; src/pod.py.

## VERDICT

**A third bounded exact-technical rental IS justified.** Method: **operational host-driver pre-vetting (reject drivers < 580) at the launcher level, keeping the pinned Torch 2.12.1 / CUDA 13 / Transformers 5.0.0 stack unchanged.** Do NOT change Torch/CUDA pins and do NOT attempt forward-compat injection. No refreeze of the frozen exact runner/loader apparatus is required; the launcher gains one additive operational gate (which does require a fresh preflight green, since preflight fingerprints job+launcher).

## BLOCKERS before attempt 3 (all cheap, all local)

1. **Driver pre-vet gate in `launch_pod.sh`:** extend the existing bootstrap `nvidia-smi` call (line 51) to query `driver_version`; if major < 580, terminate + reprovision (bounded loop, max ~3 tries), before rsync/upload/launch. Cost per rejected host ≈ 1–3 min at $1.39/h ≈ $0.02–$0.07. RunPod's create API cannot filter by host driver, so post-create/pre-launch is the earliest possible vet point.
2. **Mechanical EXPECTED_COMMIT derivation:** the coordinator must set `SC_EXPECTED_COMMIT="$(git rev-parse HEAD)"` — never hand-expand a short SHA. The interlock caught the typo (as designed) but it cost a launch cycle on a billed pod. Make this a scripted step in the launch invocation, not a transcription.
3. **Fresh preflight green + `bash -n`/ShellCheck** on the modified launcher; commit additively (a dated authorization note, per the additive-correction house rule) before spending.

## Why this is the lowest-risk method

- **The failure is a host lottery, not an apparatus defect.** Attempt 1's host (driver 580.159.03) initialized the identical pinned stack and loaded the 61 GB model; attempt 2's host (550.90.12, max CUDA 12.4 per the log's "found version 12040") cannot initialize CUDA 13 wheels at all. Same image, same job, different host. The variable is the host driver and nothing else.
- **Changing Torch/CUDA pins** (e.g. cu12 wheels) would trip the job's own dependency-drift FATAL, force edits to the frozen job script, invalidate the 165/165 local suite's relevance, and discard the one partial success we have (attempt 1's load under this exact stack). It trades a solved provisioning problem for fresh apparatus risk.
- **Forward-compat injection** (cuda-compat libs into a container we don't control) is an unpinned moving part and exactly the "grind on infrastructure" anti-pattern AGENTS.md forbids; reprovisioning is the sanctioned response to a bad host.
- **Threshold rationale:** 580.159.03 is observed-good; 550.90.12 is observed-bad; the range in between is untested. Fail closed: require major ≥ 580 rather than reasoning about intermediate drivers.

## Scientific comparability

Unimpaired. Both failures were **pre-forward**: no token, cache comparison, Phase A, treatment, or semantic observation exists to void or reconcile. The pinned numeric environment (torch 2.12.1 / CUDA 13 / transformers 5.0.0 / bf16 / eager) is unchanged; a driver floor is below the CUDA runtime ABI and does not alter numerics — it only lets the already-pinned runtime initialize. Requiring 580.x makes attempt 3 maximally comparable to attempt 1, the only environment in which the model has actually loaded. The receipt already records driver version via its `gpu` field, so provenance is preserved automatically.

## Spend and stopping rules

- **Spend to date:** observed balance deltas $0.0383 + $0.0589 ≈ **$0.097**; conservative upper bound **$0.2023**. Both far inside the $2 tranche; the ~$60 budget is untouched in any material way. Critically, both failures were cheap *because* the gates failed closed fast — the reliability apparatus performed exactly as designed. That is evidence the bounded-attempt process works, not evidence against continuing.
- **Attempt-3 stopping rules (proposed):**
  1. Attempt 3 stays inside the existing $2 tranche (expected cost ≈ attempt-1 scale plus download/load/forward time).
  2. Driver pre-vet reprovision loop bounded at 3 hosts; if 3 consecutive hosts fail the vet, stop and reassess availability rather than looping.
  3. If the loader attestation fails again on real weights (attempt-1 failure class recurring after the correction), terminate immediately, pull logs, diagnose locally — no on-pod iteration on billed time.
  4. Any *new* pre-forward failure class → stop, fresh design audit + Fable consult before attempt 4 (extending the standing "second failure → audit" rule that this document satisfies for attempt 3).

## Refreeze assessment

The exact runner/loader was already corrected and re-authorized after attempt 1 (MoE-aware attestation, revision 6; two independent Codex reviews; 17/17 loader suite, 165/165 focused suite). Attempt 2 never reached it and produced no information that impeaches it — the loader correction remains unexercised on real 30B weights, which is precisely attempt 3's purpose. **No refreeze of the frozen scientific apparatus is needed.** The launcher driver gate and the SHA-derivation rule are operational/provisioning changes outside the measurement path; record them additively (dated authorization note + preflight re-green), not as a preregistration amendment.

## Residual risks acknowledged

- The 580-floor is a one-sample known-good; a ≥580 host could still expose a new failure mode (e.g. NCCL/cuDNN quirks). Bounded by stopping rule 4.
- Driver vetting cannot be done pre-create, so each bad host costs a small terminate fee; bounded by stopping rule 2.
- The loader's sentinel bit-compare has never run against real weights; attempt 3 is the test, and stopping rule 3 caps its downside.
