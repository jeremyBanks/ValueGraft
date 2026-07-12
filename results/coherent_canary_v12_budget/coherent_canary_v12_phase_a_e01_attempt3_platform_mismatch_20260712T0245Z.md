# V12 e01 Phase A attempt 3 — healthy host, bound-platform mismatch

**Recorded by:** Sol — GPT-5.6 Sol, extra-high reasoning

**Pod:** `es2sc9kg3c2kle` (`v12phasea-e01-20260712t0243`)

**Machine/data center:** `3is1tqstotnl`, `US-KS-2`

**Observed admission:** Secure `NVIDIA A100 80GB PCIe`, driver `580.159.04`,
81,920 MiB, `$1.39/hour`

**Rental:** 2026-07-12T02:43:35Z

**Deletion:** approximately 2026-07-12T02:45:35Z

The provider placement was healthy and acquired SSH promptly. The ignored
Phase-A job then observed host kernel `6.8.0-100-generic`; the passing exact
technical report's runtime fingerprint binds
`Linux-6.8.0-107-generic-x86_64-with-glibc2.35`. The job intentionally exited
before dependency installation, repository cloning, model download/load,
model forward, generation, or Phase-A scoring. No scientific artifact exists.

Sol manually deleted the precreated pod. Direct lookup returned HTTP 404 and
the active-pod list was empty.

**Immediate observed delta:** `$0.0220443204`

**Full 120-second window bound:** `$0.0463333333`

**Phase-A provisioning unit across three attempts:** immediate balance delta
`$0.1548273167`; conservative full-window bound `$0.1783833333`. Billing may
settle asynchronously; the final per-pod endpoint audit supersedes these
snapshots.

This was not a scientifically incompatible GPU. It was a different host kernel
from the one accidentally/literally included in the prior runtime fingerprint.
Continuing to rent hosts in search of kernel `107` would optimize for a
historical placement rather than validate the actual execution environment.
The next proportional route, if separately authorized, is a fresh complete
technical gate on one healthy exact A100-PCIe host, followed by e01 Phase A on
that same pod/GPU in a distinct process bound to the new report. No old
technical threshold or result is discarded.
