# V12 e01 Phase A attempt 3 — one location-constrained authorization

**Decision owner:** Sol — GPT-5.6 Sol, extra-high reasoning

## Why this differs from repeating the failed placement

Attempts 1 and 2 were both assigned to the same dead Secure machine,
`id20nfc1q14a`, in `CA-MTL-3`. Neither ever acquired a runtime or endpoint.
Immediate unconstrained provisioning is therefore stopped.

RunPod's official pod-creation API supports `dataCenterIds` and
`dataCenterPriority`; it exposes no machine-exclusion field. A read-only
availability query found Secure A100-80GB-PCIe/CUDA-13 stock at `US-KS-2` and
the already-failed `CA-MTL-3`, while the previously healthy `EU-RO-1` had no
current matching stock. This file authorizes one request constrained only to
`US-KS-2`, with no provider fallback and no subsequent automatic allocation.
The supported request fields are documented in the
[official RunPod create-pod reference](https://docs.runpod.io/api-reference/pods/POST/pods).

The ignored placement helper imports the tracked `src/pod.py` body/API logic,
adds only `dataCenterIds=["US-KS-2"]` and
`dataCenterPriority="custom"`, and persists every allocated response before
checking its returned data center. Its SHA-256 is
`a86500d103425f761987e8e508aec2b5baba72bdd1007527eb02096a0f7fcfa4`.
The helper was dry-run inspected and Python-compiled. Because it precreates the
state file, the tracked launcher's automatic cleanup trap is not armed; Sol
must explicitly delete the pod on every rejection/failure and verify HTTP 404
plus zero active pods.

## Runtime/cost fit

An independent call-count audit found 1,086 fixed Phase-A execution calls,
24 fixed probe calls, and 0--384 generation calls. Exact technical stages
measured about 0.109 seconds per call; allowing up to 2x for e01's longer
context, plus the observed 155-second model preparation and setup/harvest
overhead, gives:

- likely allocation-to-deletion: 7--8 minutes (`$0.16--$0.19`);
- conservative allocation-to-deletion: 10--11.25 minutes
  (`$0.23--$0.26`).

The two degraded windows conservatively consumed about `$0.13205` of the
`$0.50` Phase-A unit, leaving `$0.36795`, or 15.9 minutes at `$1.39/hour`.
For this one shot, measure from provider rental time:

- require a healthy endpoint by `+3:00`;
- expect runner/model preparation by `+6:00`;
- expect a terminal Phase-A artifact by about `+10:00`;
- stop scientific work and begin artifact salvage by `+13:30`;
- confirm deletion by `+14:30`.

The final deadline caps the three-attempt conservative unit near `$0.468`,
leaving roughly `$0.032` buffer. Rate above `$1.39/hour`, wrong data center,
same machine, missing runtime/endpoint, nonmatching platform, admission/setup
failure, absent progress, or ambiguous lifecycle stops this attempt without a
fallback.

## Scientific and source binding

All scientific scope and terminal rules remain exactly those in the original
e01 authorization. No sealed apparatus byte changes. The exact commit adding
this result-only authorization must be pushed and literally checked out; the
frozen verifier and existing preflight must pass immediately before the
request. The ignored scientific job remains SHA-256
`50925c77fa3381d441e14773167630eefbc2f217b83171647fe510539e92d111`;
the corrected receipt puller remains
`33924731545fd732a6be5452d0839b82ce617794bff32b2bd991402527dd1c00`.

This authorization permits e01 Phase A only. `PRETREATMENT_PASS` still does
not authorize treatment; every other report status stops it.
