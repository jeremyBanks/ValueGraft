# Exact-v12 attempt-three implementation review and disposition

**Author and decision owner:** Sol — GPT-5.6 Sol, ultra reasoning

**Independent inputs:** Claude Fable 5 (`claude-fable-5`) provisioning audit;
two fresh read-only Codex/GPT-5.6 Sol subagent audits; local fault-injection and
focused test observations.

**Recorded:** 2026-07-11/12 EDT/UTC

## Top-line decision

The third exact technical attempt is **conditionally authorized** only after a
fresh A7/B7 apparatus freeze, one result-only launch-authorization record, a
passing production frozen verifier, and a fresh preflight. The numerical stack
is unchanged. The provisioner now narrows host admission to the stack it was
designed for and fails closed across ordinary allocation, admission, setup,
credential, cleanup, and provenance failures.

This is not an authorization for Phase A, treatment, or a semantic claim. It is
an authorization to finally exercise the corrected exact technical apparatus
on one qualifying host. Neither prior rental reached a subject forward.

## Honest record of the attempted fresh Fable implementation review

The standing reliability rule called for a fresh Fable review after implementing
the Fable-recommended host-admission remedy. Sol made three deliberately narrow,
budget-capped attempts. The CLI exposed the exact runtime identity
`claude-fable-5` in each attempt, and the model read the current implementation,
but budget/tool-handoff failures stopped every session before it wrote a verdict
or the reserved note. Reported nominal provider usage was `$1.261740`,
`$0.876495`, and `$1.053130`; an immediately interrupted mistaken prompt added
`$0.001423`, for `$3.192788` total. This is subsidized Claude CLI usage, not pod
compute.

Sol stopped rather than approach/cross the owner's roughly `$4` warning for a
routine focused review. **There is no fresh Fable implementation verdict, and
none is inferred or attributed.** The earlier completed Fable provisioning
audit remains valid strategic input: preserve CUDA 13, pre-vet driver 580+, use
bounded attempts, and stop on any new failure class. Fresh post-fix review was
instead supplied by independent Codex audit contexts, with Sol retaining final
decision authority as the owner requested.

## What the first independent implementation audit found

The first committed admission implementation correctly used RunPod's CUDA-13
filter and parsed the actual GPU/driver/memory before bootstrap. It was still a
NO-GO because the paid lifecycle was not fail-closed:

1. `pod.py terminate || true` could swallow a failed DELETE while the wrapper
   claimed termination and allocated another pod.
2. Bootstrap failures shared exit 86 with incompatible-host admission and could
   consume all three retries for one deterministic setup defect.
3. Transient status errors and several upload/launch failures bypassed cleanup,
   leaving an allocated pod billing.
4. The rsync containing the required Hugging Face token used `|| true`, and the
   remote move was also masked; the job could be launched only to die without
   credentials.
5. The wrapper captured the right full commit, but the job cloned the moving
   `origin/trunk` tip and required tip equality. An unrelated later notes commit
   could abort a scientifically unchanged job.
6. Authorization 6 could not open the changed/post-auth history. Three sealed
   note paths had also been renamed by the archive normalizer.
7. The original eight provisioning tests checked literals, not lifecycle
   behavior, and could not detect any of the above.

These were ordinary operational/scientific failure modes, not adversarial-host
hardening. Catching them locally prevented a plausible third paid failure and a
possible duplicate-billing leak.

## Corrections actually implemented

Commit `f7c72e5` made the lifecycle explicit:

- exit 85 is reserved for an explicit RunPod HTTP-500/no-allocation response;
  transport ambiguity is not retryable;
- exit 86 is reserved for an allocated host whose GPU compatibility cannot be
  established or does not meet the frozen requirement;
- exit 87 means cleanup DELETE failed and forbids any further allocation;
- every other failure is non-retryable;
- one pre-job EXIT trap owns and terminates only the pod created by that
  invocation; it disarms only after detached job handoff, because post-launch
  ambiguity may contain wanted work and must leave exactly one pod to inspect;
- status reads retry inside a finite endpoint wait, while wholly unreadable
  provider status stops as a new failure class;
- required credential transfer, remote presence, rename, and non-empty content
  are checked fail-closed;
- the exact expected commit must exist and remain an ancestor of cloned
  `origin/trunk`, then is checked out onto local branch `trunk`, satisfying the
  production verifier even if the remote tip later advances;
- the wrapper retries only 85/86, never bootstrap, credential, cleanup, launch,
  or provenance failures; and
- mocked shell-level fault injection now exercises status retry, unreadable
  status, absent endpoint, bootstrap failure, credential failure, deletion
  failure, ambiguous/no allocation, and wrapper retry taxonomy.

A fresh post-fix audit found one remaining blocker: REST/GQL calls had no socket
timeout, so create/status/DELETE could themselves hang indefinitely. It also
identified that the exact wrapper should mechanically run the authoritative
frozen-repository verifier before allocating, because the ordinary preflight
hash intentionally covers only the job and generic launcher.

Commit `5496b94` therefore:

- gives every RunPod REST/GQL call a positive finite timeout (30 seconds by
  default, validated; configurable with `SC_POD_API_TIMEOUT_S`);
- leaves timeouts/transport errors non-retryable and allocation-ambiguous;
- runs `verify_frozen_repository(Path.cwd())` locally before entering the
  allocation loop; and
- requires the verifier's head to equal the mechanically captured launch head.

The remote job independently repeats the same frozen verification before model
snapshot resolution/download. Local verification avoids paying to discover an
invalid freeze; remote verification binds the actual executed clone.

## Observed verification

On the final pre-freeze implementation bytes, Sol observed:

- provisioning/lifecycle suite: `24/24` passed;
- complete focused v12 suite plus provisioning: `189/189` passed;
- loader-focused tests included in that suite: `17/17` passed;
- Bash syntax: clean for the exact wrapper, generic launcher, exact job,
  puller, and preflight wrapper;
- ShellCheck: clean for the same shell paths; and
- Python compilation: clean for `src/pod.py`, `src/pod_admission.py`, and
  `scripts/preflight.py`.

The only test output outside PASS was the recurring SWIG deprecation warnings.
These tests do not substitute for the real 30B topology, sentinel, identity,
and forward gates; those remain exactly the purpose of the paid technical run.
A fresh post-fix reviewer separately reran the `24/24` provisioning suite and
the shell checks, inspected commit `5496b94`, and returned GO after the listed
release housekeeping with no remaining scoped launch-blocking code fault.

## Scientific boundary

The pinned user-space numerical environment remains:

```text
Qwen/Qwen3-30B-A3B-Instruct-2507@0d7cf239...
+ Torch 2.12.1 / CUDA 13 / Transformers 5.0.0
+ bf16 / eager attention / exact frozen protocol
```

Provisioning now asks for CUDA 13 and rejects drivers below NVIDIA's documented
CUDA-13 floor `580.65.06`; it does not swap Torch wheels or inject compatibility
libraries. Driver identity can still differ above the floor. That is acceptable
for this within-host technical gate because the actual driver is recorded and
all paired comparisons occur on the same host. Any later semantic execution
must retain that paired-within-host structure and bind its actual driver; raw
numerical evidence must not be blended across drivers without a separate
comparability argument.

## Mandatory boundary before spending

1. Update the loader to select authorization 7, correct the three normalized
   note paths, and require every launch/admission/review file in the sealed
   inventory.
2. Rerun the `189/189` focused suite and all shell/Python checks on the exact A7
   bytes.
3. Commit and push apparatus A7.
4. Generate authorization 7 from `git show A7:<path>` bytes, canonical and
   lexically sorted; commit B7 immediately after A7, adding only that JSON.
5. Add one unique `results/coherent_canary_*` launch-authorization note. No
   notes, rollups, renames, merges, edits, or empty commits may occur after B7.
6. Push, run the production frozen verifier locally, and observe its head equal
   the launch head.
7. Run a fresh preflight for the exact model/job/launcher, then invoke only
   `scripts/launch_coherent_canary_v12_technical.sh`.

## Exact paid stop rules

- At most three total allocation/admission attempts, with no unfiltered
  fallback. Exit 85 or 86 may advance to the next uniquely named attempt only
  after the previous attempt has either allocated nothing or positively
  completed DELETE.
- Exit 87, an unreadable provider status window, bootstrap/credential failure,
  provenance failure, or any other new pre-forward class stops immediately and
  receives a fresh audit; no automatic next rental.
- Once detached handoff occurs, the wrapper never allocates another pod. Any
  ambiguous post-launch check leaves exactly that one registered pod for prompt
  inspection, artifact pull if present, and an explicit lifecycle decision.
- Torch CUDA initialization, exact model topology/content sentinel, generated
  or forced identity, normal-EOS, runtime/path, or independent validation
  failure stops the run. Preserve logs/results, terminate after preservation,
  and do not iterate on the paid host.
- A technical PASS authorizes inspection and a separate Phase-A decision only.
  It does not automatically authorize Phase A, treatment, expansion, or a
  claim.

## Final disposition

The implementation is proportionate to the actual threat model. It does not
attempt cryptographic deployment or adversarial-host defense. It prevents the
specific ordinary failure modes observed or exposed by review, preserves the
known-good numerical stack, records the remaining host variable, and keeps
every retry bounded. Subject to the mandatory fresh freeze and preflight above,
Sol's verdict is **GO for one bounded exact technical attempt**.
