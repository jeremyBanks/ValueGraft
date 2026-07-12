# V12 exact technical attempt 3 — bounded launch authorization

**Recorded by:** Sol — GPT-5.6 Sol, ultra reasoning

**Recorded at:** `2026-07-12T01:47:42Z`

## Frozen binding

- apparatus A7: `8cfc6e2c2dd53505ddc4f6089953f807c72ac306`
- authorization B7: `59afc9a671c106bcf09e3481b3cdf94a140485ca`
- sealed files: `68`
- sealed inventory SHA-256:
  `86a26693fb99f0c3ce080399abbc8814f76bcf70d708ecf17b0d1f4291f4a5b1`
- authorization-file SHA-256:
  `2dfaba52cde650cbbb23f21579efecbbadbc5e96dafe6a51b5cc44656f363469`
- production frozen verifier at B7: PASS, zero post-authorization additions
- complete focused suite on A7 bytes: `189/189` passed
- Bash syntax, ShellCheck, and Python compilation on A7 bytes: PASS

This file is the sole planned result-only descendant before launch. Its commit
must add only this unique Markdown path. The exact wrapper will derive that
commit's full SHA mechanically, require clean `trunk == origin/trunk`, run the
production frozen verifier locally, and bind the verifier's observed head to
the launch head. The remote job checks out the same exact commit on its local
`trunk` and repeats verification before model download.

## Critical apparatus hashes

- exact job: `531b07fe53276e68e29c755c8121a139556d46ecce548f88a1be5e2739972f76`
- exact puller: `e33bc6076bf78a4b85c9c231b1ba96ba786519094dbfc6bbbbe4e0c3a7c74e5e`
- exact wrapper: `dc03a2074c9906cac771e360d692c125150f8bb8b37d9ebf5c5004076ec08034`
- generic launcher: `1c3bf8015a60f9d2c3a6403525ac90192822670280a9f407d1e8dc03871dece1`
- loader: `ddc98ce1dc59a89f41ae4bf1ee40af87f26787e83dc48045c382078eddf08c3a`
- provider client: `8df4786b1d68a1e70ee620e29aced9173757894e95da319c2d6eb77db46a510f`
- host admission validator:
  `c625d70a0f2b3e557643b55e07d3095735ef0a976447bee66686d93fca878f7a`

## Preserved prior-attempt evidence

- Attempt 1 failure/receipt artifacts were preserved in commit
  `51c6b8da09fc0c666c38730e3617d6122a5d198c`.
- Attempt 1 cost record was preserved in commit
  `a7c8e5e2bfe9fdaf10e1565f34772c73bc9d20dd`; file SHA-256
  `84e89fa7ba99d48ac40a09cb6d1f96e090ab457fe2c8da5ce0c502acd779a67d`.
- Attempt 2 infrastructure failure and cost record were preserved in commit
  `560f5e6a39185e8bbc53ecef8342b90bad48a7ec`; cost-record SHA-256
  `1341e003690657e29959234e49a7863ed4c30d5520f8456871f03d91c04db9e8`.
- The completed Claude Fable 5 provisioning audit was preserved in commit
  `3c936920e79f9727a6a70fc192d68b52b2a7b43d`; file SHA-256
  `1340d9cc8a84b708d33f3db79a3f033eb0d9728567f3c962a9dfefa2e43d9cd9`.
- The host-driver/provider postmortem SHA-256 is
  `922364216661e97a32ef1400fdb8b13d3b7ea2995f671be81baa7062aad57e1a`.
- The final cross-model implementation disposition SHA-256 is
  `a7adf0cdaa291de9909a664e459a357b9bd5da7f387787facfb9d2069e15f623`.

Both attempts stopped before any subject forward, generated token, cache
comparison, Phase A arm, treatment, or semantic score. Attempt 1 established
that the pinned stack initializes and the exact 30B checkpoint loads on driver
`580.159.03`; it then exposed a false raw-versus-packed MoE attestation. Attempt
2 exposed driver `550.90.12`, on which pinned CUDA 13 could not initialize.

## Why a third technical attempt is methodologically legitimate

The loader correction changes only how the exact Transformers-5 packed MoE
representation is attested. The provisioning correction does not change model,
revision, tokenizer, Torch release, CUDA runtime, Transformers release, dtype,
attention backend, prompts, targets, or metrics. It asks the provider for CUDA
13 and independently requires the exact A100-80GB PCIe, driver
`>=580.65.06`, and at least 80,000 MiB before bootstrap. Actual driver identity
is recorded. All paired technical comparisons remain within one host.

The first admission implementation received a fresh independent audit. Its
cleanup/retry/credential/commit-race defects were corrected, then re-reviewed.
The final reviewer independently reran `24/24` provisioning tests and the shell
checks and returned GO after the A7/B7 release steps. A fresh post-fix Fable
review was attempted under strict caps but produced no verdict before its tool
handoff budgets; no endorsement is inferred. Sol owns the final decision.

## Authorized action

Only the exact technical job may run:

```text
scripts/launch_coherent_canary_v12_technical.sh <unique-attempt-3-base-name>
```

The wrapper requests Secure Cloud and `allowedCudaVersions=[13.0]`; it permits
one to three total allocation/admission attempts, uses a unique pod name and
state file for each, and never falls back to unfiltered capacity. The job
contains no Phase A or treatment command. A technical PASS requires runner and
independent-validator exit zero, validation status PASS,
`semantic_release_eligible=true`, and exactly one durable identity checkpoint.

## Spend boundary and observed starting state

- Attempt 1 observed balance delta: `$0.0382625315`.
- Attempt 2 observed balance delta: `$0.0588781620`.
- Combined observed deltas: `$0.0971406935`.
- Combined conservative bound: `$0.2023333333`.
- Active RunPod pods observed immediately before authorization: `0`.
- RunPod balance observed immediately before authorization: `$63.1124828884`.

The original `$2` exact-v12 tranche remains the operational bound for all three
technical attempts combined. At the last observed Secure A100 rate of
`$1.39/hour`, one hour plus brief rejected-host admissions remains within that
tranche after the conservative prior-attempt bound. If the allocated rate is
above `$2/hour`, or the one-hour rental boundary is reached without a terminal
artifact, stop, preserve available evidence, and terminate; no silent budget
expansion. Subsidized Claude/Codex CLI accounting is tracked separately and is
not pod spend.

## Mechanical stop rules

1. Local freeze, preflight, cleanliness, push, or head-binding failure: no
   allocation.
2. Exit 85 (explicit provider HTTP-500/no allocation) may advance within the
   three-attempt ceiling. If a live pod is ever observed for an 85 attempt, the
   assumption is falsified: stop all retrying and terminate after preserving
   evidence.
3. Exit 86 (incompatible/unattestable allocated host) may advance only because
   its cleanup DELETE succeeded. No unfiltered fallback.
4. Exit 87 (cleanup failed), a fully unreadable provider-status window, any
   bootstrap/credential/provenance failure, or any other/new pre-forward class:
   stop immediately and do not allocate another pod.
5. After detached handoff, no retry is automatic. An ambiguous post-launch
   check leaves exactly one registered pod for inspection. Preserve wanted
   artifacts before any termination decision.
6. CUDA initialization, exact topology/content sentinels, generated/forced
   identity, normal EOS, runtime/path controls, runner, or validator failure:
   pull and commit every available artifact, terminate, and audit locally. No
   paid on-host iteration.
7. A technical PASS authorizes evidence review and a separately recorded Phase
   A decision only. It does not authorize Phase A, treatment, expansion, or a
   scientific claim by itself.

Subject to the result-only commit, push, frozen verification, and fresh
preflight, Sol authorizes this bounded exact technical attempt.
