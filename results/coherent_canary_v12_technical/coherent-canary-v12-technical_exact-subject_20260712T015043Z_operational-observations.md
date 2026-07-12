# Exact-v12 attempt 3 — operational observations

**Recorded by:** Sol — GPT-5.6 Sol, extra-high reasoning

**Scientific artifact status:** preserved and independently validated `PASS`

This record separates two newly observed orchestration defects from the valid
scientific result. Neither changed model inputs, cache state, model outputs, or
the persisted raw/validation bytes. Both must be corrected and tested before
the affected helper is described as reliable in a later release.

## 1. The advertised detached launch did not return promptly

The exact job started normally, survived the SSH session, wrote its log, and
completed. However, the local launch call did not return after
`job-launched`. Remote inspection while the model job was running showed a
shell process (PID 608, PPID 1) waiting for the actual job process (PID 610).
The likely ordinary shell cause is the command grammar at
`scripts/launch_pod.sh:143`: the trailing `&` backgrounds the preceding
`cd && chmod && ... nohup ...` AND-list, leaving an intermediate shell that
waits for the model job. Redirecting the job's file descriptors was therefore
not sufficient to make the SSH invocation return promptly.

Observed impact: monitoring had to occur from a separate shell, and the
launcher returned only after the job completed. The job itself was not killed
or corrupted. Before reuse, the launch grammar should make setup commands
foreground statements and background only the final `nohup` command; a live
integration test must observe both a prompt SSH return and a still-running
child.

## 2. The receipt-driven pull loop lost its remaining input

The tracked puller successfully copied the receipt and raw artifact, then
failed its terminal local verification because the validation artifact was
absent. The loop at `scripts/pull_coherent_canary_v12_technical.sh:89-97`
invokes `ssh` without `-n` while the loop itself reads artifact paths from a
here-string. That SSH process consumed the loop's remaining standard input, so
later receipt-listed paths were never iterated.

Observed impact: no remote artifact was lost. Sol separately fetched every
receipt-listed path (raw, validation, job record, and the durable identity
checkpoint), then checked each exact byte length and SHA-256 against the
receipt. All four matched:

- raw: `ddc5faf7f24dd6ca70bf5833351c38b930d851bb9b0535cb45814dc32413b990`;
- validation: `46fa73d864d7f5a06b2053dd38837d918fdcdf9ae88c4709c07070a3d8a81022`;
- job record: `4166e4d8e50a5372b72b5af12cffa16d73b5ee36988f89dd0cad2be96e2689c7`;
- identity checkpoint:
  `0c8fcc6d07314505b4ca81f52b159a343256af854b2d328ef67a61de0020ba7d`.

The proportional correction is to detach SSH from the loop's stdin (for
example, `ssh -n`) and add a regression test with multiple artifact paths.

## Freeze consequence

Both helper files are part of the sealed apparatus inventory. Editing them now
would intentionally invalidate the runtime fingerprint bound by the passing
exact technical gate. Consequently, this record preserves the observations
without changing frozen bytes. Any further paid semantic execution must either
use a recorded external orchestration workaround that leaves the scientific
apparatus unchanged, or create a new apparatus authorization and repeat every
gate that its rules require. No claim that these two helpers are already fixed
is made here.
