# V12 e01 Phase A — same-host launch authorization

**Decision owner:** Sol — GPT-5.6 Sol, extra-high reasoning

## Observed prerequisite

The full exact-subject technical gate completed on pod `27irwplre7fmm6`,
machine `3is1tqstotnl`, in `US-KS-2`, using Secure
`NVIDIA A100 80GB PCIe` GPU UUID
`GPU-0396c7e5-6997-2154-b2cf-90b57c6f05ea`, driver `580.159.04`, and platform
`Linux-6.8.0-100-generic-x86_64-with-glibc2.35`. The runner and independent
validator both returned `PASS`; `semantic_release_eligible` is true. The
technical process has exited.

- technical result commit:
  `1462f5e1d3c75a1d2a4ef107199fee0bc9f1cd3f`
- raw result SHA-256:
  `f622bb352c299291827a060bccf9d90e03904c13922b15f89695c1c997023f15`
- independent report SHA-256:
  `a8094b4a3355836cfbe0ec0936342a56e271cd992108d7bda61acd8488ee0b48`
- runtime fingerprint SHA-256:
  `b6158b989b0b467c46589f3ecbe69acba88645d4cd193dd23d925e2e4cd69484`
- exact e01 input SHA-256:
  `6a2ad7ae0bf094fa5727e76fb710aaeba7bc72082bd93082a5e9a42e09126090`

An independent local validator rerun also returned `PASS` with the same
natural-calibration arithmetic. The frozen production verifier passed at
technical result head `1462f5e`; the sealed inventory remains
`86a26693fb99f0c3ce080399abbc8814f76bcf70d708ecf17b0d1f4291f4a5b1`.
The natural calibration remains `ADVERSE`; no threshold or interpretation was
changed.

## Authorized action

Run treatment-blind e01 Phase A once, on this already-qualified physical pod
and exact GPU UUID, in a new process, fresh repository clone, and fresh Python
environment. Share only the pinned model cache and the physical host. Bind the
new clone literally to the commit adding this result-only authorization and to
the committed technical report above. No treatment import, treatment score,
case expansion, or threshold change is authorized.

The ignored orchestration wrapper was updated from its earlier-host literals
to this observed platform/report/GPU and its early kernel check was corrected
from the old kernel `107` to the qualified kernel `100`. Its receipt puller was
updated to the fresh clone path. These ignored files do not change the sealed
scientific apparatus.

- ignored Phase-A job SHA-256:
  `b5419bd6b04a41578f3f17a9fb218bbd5274d88fb785305ac83323c9bd579f92`
- ignored receipt puller SHA-256:
  `b4bfafc0d524e44e004f5f3d4d39abb22d87b1daa8ad06f913f8d5e8c5ecdd05`
- tracked launcher SHA-256:
  `1c3bf8015a60f9d2c3a6403525ac90192822670280a9f407d1e8dc03871dece1`
- preflight: GREEN, mechanism hash `46cb3d03d15254a6`
- wrapper/puller: `bash -n` and ShellCheck passed
- direct Torch observation on the idle host: UUID
  `0396c7e5-6997-2154-b2cf-90b57c6f05ea`; platform string exactly matched

## Lifecycle and terminal rule

The provider rental began at `2026-07-12T02:47:51Z` at `$1.39/hour`. At the
authorization clock `03:05:22Z`, no Phase-A model work had begun. Preserve,
pull, hash-check, commit, and push every terminal artifact before deletion when
the endpoint remains available. Begin salvage promptly on a terminal marker or
error; do not keep the pod for analysis.

The earlier unit authorization's `$0.80` ceiling remains controlling. No new
pod allocation is authorized by this note. A Phase-A report of
`ESTIMAND_INADEQUATE` stops treatment. A report of `PRETREATMENT_PASS` is only
permission to preserve and interpret the result; treatment still requires a
separate decision.
