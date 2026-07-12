# V12 exact technical launch review disposition

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning

**Paid v12 pod compute observed:** `$0`

## Scope

The local proxy evidence met the four conditions in the reviewed cap amendment:
independent generated/forced equivalence passed, both repeats were identical,
normal EOS was the sole identity failure, and every other technical plumbing
check passed or was valid-and-adverse under its frozen reporting rule.  This
permits only a bounded exact 30B technical attempt.

The exact job is deliberately small.  It clones one expected trunk commit,
installs Python 3.12.11 and the five loader-pinned dependency versions, verifies
one CUDA device, resolves the frozen 30B revision, runs the exact technical
runner, independently validates the raw record, and writes raw, identity
checkpoint, validation, job-log, and receipt artifacts.  It contains no Phase A
or treatment command.

## Independent launch review

A bounded runtime review found the model ID, revision, bf16/eager settings,
dependency checks, CUDA check, pinned download, A100-80GB choice, and core gate
binding correct.  It found ordinary operational blockers before launch; all were
disposed as follows:

1. **Failure could exit zero:** the job now exits nonzero unless runner and
   validator exit zero, the report is `PASS`, semantic release eligibility is
   true, and exactly one durable identity checkpoint exists.
2. **Old raw reuse:** a timestamp marker now requires exactly one newly created
   exact raw artifact; committed prior results cannot be selected on retry.
3. **No off-pod durability:** a separate bounded pull script copies every exact
   raw/checkpoint/report/log/receipt, verifies receipt hashes locally, and is run
   before termination.  The coordinator then stages explicit result paths,
   commits, and pushes before any pod shutdown that would discard remote state.
4. **Stale launch binding:** this apparatus revision receives a new immediate
   authorization; a later additive result note will bind that actual pair and
   the final pushed launch commit before provisioning.
5. **Hourly rate not forwarded:** the in-job `$2.00/hour` value is retained and
   documented as a conservative artifact estimate.  The coordinator records the
   actual `costPerHr` from the RunPod API and monitors elapsed actual cost against
   the initial `$2` tranche and overall `$8` actual-pod ceiling.

The pod is not self-terminating because a passing exact report may make the
already-loaded model useful for the next separately authorized stage.  The
coordinator actively monitors job liveness, actual elapsed cost, and terminal
markers; on failure it pulls available evidence before termination, and on
success it first pulls, validates, commits, and reviews the technical result.

## Static observations

- both shell scripts passed `bash -n` and ShellCheck;
- the complete focused v12 suite passed `162/162` with only the same two SWIG
  deprecation warnings;
- the job and pull scripts are part of the next sealed inventory;
- no paid pod was provisioned and no additional subject-model forward was run
  during this launch review.
