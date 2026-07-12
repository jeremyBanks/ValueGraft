# e01 exact-treatment post-run audit

This ignored directory contains an independent, fail-closed checker prepared
before looking at the treatment outcome. It performs no discovery of treatment
files and has no default treatment or pod-harvest path.

## Checks

1. Resolve the Phase-A raw artifact from the treatment's hash binding (an
   explicit local `--phase-a` override is accepted only when its bytes have the
   bound hash).
2. Compare raw `treatment.fresh_scores.focal/nonfocal` with raw Phase-A
   `scores.FF_focal/FF_nonfocal` using canonical JSON bytes. A difference in
   any field, including diagnostic decimal fields, is `INVALID` and skips the
   harvester.
3. Reconstruct the treatment raw artifact byte-for-byte inside a new ignored
   work directory and verify both its byte hash and canonical-object hash.
4. Verify that `scripts/harvest_coherent_canary_v12.py` is byte-identical to
   both its `HEAD` blob and the pre-run frozen SHA-256 recorded in the checker,
   then run that tracked script in a subprocess.
5. Compare the pod and local harvest objects. The only normalized fields are:

   - `/completed_at_utc`
   - `/treatment_artifact/path`
   - `/verified_bindings/<binding>/path`

   Any other difference is `INVALID`. The report records every raw difference
   pointer, classifies legitimate path/timestamp differences, and records
   hashes for inputs, reconstruction, harvester, subprocess logs, local output,
   and normalized pod/local objects.

The report itself is written as `audit_report.json`, with a separate
`audit_report.sha256.json` sidecar so the report can have an external digest
without a self-referential hash.

## Invocation after the outcome is safely persisted

Use a unique, nonexistent work directory:

```bash
python3 .sol-v4/treatment_postrun_audit/postrun_audit.py \
  --repo /Users/jeb/experimentation \
  --treatment /explicit/path/to/e01-treatment-raw.json \
  --pod-harvest /explicit/path/to/e01-pod-harvest.json \
  --work-dir /Users/jeb/experimentation/.sol-v4/treatment_postrun_audit/run-UTCSTAMP
```

If a pod binding contains a pod-absolute Phase-A path, pass the local committed
copy with `--phase-a`; its SHA-256 still must equal the treatment binding.

Exit codes are `0` for `PASS`, `2` for scientific `INVALID`, and `1` for an
audit/setup `ERROR`.

## Deliberate limitations

- This is a single-case artifact/harvest equivalence check, not a second model
  scorer or a statistical analysis.
- It trusts the frozen Phase-A and treatment serialization schemas and the
  tracked harvester's scientific interpretation. Its purpose is to catch the
  newly identified cross-phase fresh-score continuity gap and pod/local
  reproducibility differences.
- Path values are permitted to differ because pod and local repository roots
  differ; their bound file hashes are not normalized.
- The live source trees are not snapshotted atomically by this checker. Inputs
  should already be immutable and committed before invocation.
