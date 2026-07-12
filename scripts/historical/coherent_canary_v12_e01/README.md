# Historical e01 execution and verification tools

This directory preserves the exact text bytes used to provision, run, package,
and independently audit the coherent-canary-v12 e01 diagnostic. The tools were
originally created in the ignored `.sol-v4/` workspace and were hash-bound in
the committed launch/release records, but had not themselves been tracked.

The copied bytes match their original SHA-256 values in `MANIFEST.sha256`.
Several scripts intentionally contain historical absolute paths and frozen
one-run assumptions. They are audit artifacts, not current launch entry points,
and convey no authorization to allocate a pod or run another v12 case.

The committed treatment package under
`results/coherent_canary_v12_treatment_package/` preserves the original raw
JSON byte-for-byte. `treatment_packaging/artifact_packager.py` can reconstruct
and verify that package; `treatment_postrun_audit/postrun_audit.py` preserves the
independent audit implementation; and `treatment_execution/` preserves the
bounded launcher, receipt, puller, provider watchdog, and their tests.

## Important preservation limit

The raw result contains exact token/position traces, float32 score bits,
generation text, runtime provenance, tensor shapes, and state/row hashes. It
does not contain the K/V tensor element values. The hashes can verify a future
rerun but cannot reconstruct the original rows. Although
`src/coherent_canary_artifacts.py` implements a bounded lossless safetensors
bundle, the e01 runner never invoked it. The original tensor state disappeared
when the pod was deleted.

A future study should save the fresh destination through R3 plus C/W source
rows through R3 under N/P before teardown. For the observed e01 geometry that
payload was independently estimated at about 39.84 MiB, versus about 480.94
MiB for five full cache snapshots. This is a future-design requirement, not a
claim that e01's tensors remain recoverable.
