# V12 e01 diagnostic treatment — one guarded launch authorization

**Decision owner:** Sol — GPT-5.6 Sol, extra-high reasoning

## Scope and scientific status

This authorizes exactly one unchanged exact-subject `e01` treatment execution
and independent harvest. It does not reopen formal v12. Under the conservative
disposition already committed, the written §14.1 stop rule makes the formal v12
tree a technical failure, while the frozen implementation demonstrates only
implementation-defined ULP-4 path sensitivity. The treatment is therefore a
post-ambiguity exploratory diagnostic.

Its companion receipt must state:

- `diagnostic_only=true`;
- `formal_v12_decision_eligible=false`;
- `aggregate_expansion_authorized=false`.

No outcome can choose the stop-rule interpretation, rescue formal v12, enter a
four-/six-case aggregate, authorize `e02`--`e06`, or support confirmation,
conversation-stratum, or live-agent claims. The adverse natural calibration
must be reported beside the diagnostic regardless of its sign.

## Frozen scientific release evidence

All remaining scientific execution uses a clean `trunk` clone pinned literally
to the last verifier-valid head:

`cbdfa481fe08de62bf8178d09ca040716d610f21`

Sol independently observed the frozen repository verifier pass at that head:

- apparatus commit: `8cfc6e2c2dd53505ddc4f6089953f807c72ac306`;
- authorization commit: `59afc9a671c106bcf09e3481b3cdf94a140485ca`;
- sealed inventory SHA-256:
  `86a26693fb99f0c3ce080399abbc8814f76bcf70d708ecf17b0d1f4291f4a5b1`;
- authorization SHA-256:
  `2dfaba52cde650cbbb23f21579efecbbadbc5e96dafe6a51b5cc44656f363469`;
- 30 permitted post-authorization result additions.

The treatment runner's release-only check returned exact-subject
`PRETREATMENT_PASS`, estimand adequate, semantic-release flag true under the
sealed machinery, and runtime fingerprint
`b6158b989b0b467c46589f3ecbe69acba88645d4cd193dd23d925e2e4cd69484`.
The relevant input hashes are:

- case:
  `6a2ad7ae0bf094fa5727e76fb710aaeba7bc72082bd93082a5e9a42e09126090`;
- preregistration:
  `fc02e86d007369121accbee69471383200fa3f566183e2ed96102e2cf7428b17`;
- revision-4 manifest:
  `738c176fc08075ba905d4c48b70d095a8da0dca649f382e4d7596e4a406affb2`;
- blind review:
  `4b9eace0b4f4754ca0de6fcd6dc59852374ab5fd804f42bfb06d108da1d1fcda`;
- paired review:
  `1909224e4dab2265d9b033d7b016c1a9842a24dce9531cf17d240a54287a375b`;
- exact technical report:
  `a8094b4a3355836cfbe0ec0936342a56e271cd992108d7bda61acd8488ee0b48`;
- e01 Phase-A report:
  `aabac5f0aa64db8f0db49110583be9a17f8453e1a523f1cbc0e0312ec3c15410`;
- e01 Phase-A raw:
  `2cd7f6f190b9f4e9598844e45e5488779b72cd54a1fd51dd5b8a5b24154d672d`.

Sol observed 58 focused frozen-apparatus tests pass. Claude Fable 5's full
independent release review is preserved at
`notes/2026071283-fable-e01-treatment-release-review.md`; Sol's conservative
disposition and corrected accounting are in
`notes/2026071284-sol-fable-e01-review-disposition.md`.

## Exact runtime and host admission

The job must use:

- `Qwen/Qwen3-30B-A3B-Instruct-2507` at revision
  `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`;
- one Secure `NVIDIA A100 80GB PCIe`, at least 80,000 MiB;
- NVIDIA driver exactly `580.159.04`;
- kernel `6.8.0-100-generic` and platform
  `Linux-6.8.0-100-generic-x86_64-with-glibc2.35`;
- Python `3.12.11`, Torch `2.12.1+cu130`, CUDA `13.0`, Transformers
  `5.0.0`, Accelerate `1.14.0`, Safetensors `0.8.0`, and
  huggingface-hub `1.22.0`;
- BF16, eager attention on all 48 layers, and the exact released runtime
  fingerprint before treatment code is imported.

At `2026-07-12T03:59:31.855103Z`, the provider reported Low Secure A100-80GB
PCIe stock in `US-KS-2`, balance `$62.0207254421`, and zero active pods. This
authorization permits one request constrained to `US-KS-2`, with no provider
fallback and no automatic second allocation. A returned pod must report an
hourly rate at or below `$1.39`; any higher or unknown rate is deleted.

## Forecast and machine-enforced ceiling

The observed Phase-A core took 152.666 seconds for approximately 1,125 forward
calls. The treatment is estimated at approximately 2,932--3,092 forward calls,
or 6.6--7.0 minutes of core work, plus model preparation, dependency/setup,
pull, validation, packaging, commit, and deletion. The conservative unbuffered
whole-rental estimate is 27.5 minutes. With the preregistered 25% buffer this is
34.375 minutes, or `$0.7963541667` at `$1.39/hour`.

The canary's conservative provider-balance origin is `$63.3160022124`. The
prelaunch snapshot therefore records `$1.2952767703` spent and
`$1.1047232297` remaining under the `$2.40` ceiling. A detached provider-clock
watchdog starts its clock before the allocation request, polls at most every 15
seconds, begins deletion at 2,180 seconds, uses 10-second provider API timeouts,
and sets an absolute 2,300-second bound. At the maximum admitted rate, 2,300
seconds is `$0.8880555556`, leaving `$0.2166676741` of canary headroom. The
treatment runner separately receives a 1,500-second SIGINT limit with a
60-second kill fallback.

Any allocated failed start is part of this ceiling. No replacement allocation
is authorized here. A received provider HTTP 500 with no pod ID allocates
nothing, but still does not trigger an automatic retry. The outer guard owns
cleanup because the location helper precreates the pod before the tracked
launcher sees it. Ambiguous post-launch state is preserved for inspection, not
duplicated; the provider deadline remains active.

## Bound external orchestration

The following ignored orchestration bytes are fixed for this attempt:

- location helper:
  `a86500d103425f761987e8e508aec2b5baba72bdd1007527eb02096a0f7fcfa4`;
- guarded launcher:
  `43a8970785cb54ffcd489cfeee5f184b788ea94018e3b1ffda40f93bf7d7edd9`;
- provider watchdog:
  `0b0ea46a6f08ffb256098ebb631a96733842cf344af6d87450c7c0f18a576e93`;
- treatment job:
  `0cefac1b75eb72f54cb0626cc8fe27f544d08f5aa03b428b9ba167a47493303e`;
- receipt puller:
  `7cb1db3a48e03bc129fde326d4b9cf6ca6a74c05cf57e2f317ef2fe28294c05d`;
- receipt tools:
  `bcd11a96782f7c7f706988b6e4b0a0a949e1ab336d2312e0cf7493c90099a1fd`;
- execution tests:
  `5a43d83366aa3fdc8ce4e21659a28d7aadbc1f1230d8d2038825cacdcb94c5a3`.

Sol observed `bash -n`, ShellCheck, Python compilation, nine execution/receipt/
provider-guard tests, exact pinned-input checks, and the tracked preflight pass.
The green preflight mechanism ID is `79718cf9dd16a250`; its token SHA-256 is
`eb4b7129060eb44d59cd52c2d652c78a3617e7b232b6c762532305b6ae146fcc`,
with no warnings.

## Outcome-blind cross-host and artifact checks

Before any treatment estimand is inspected, a separately prepared checker must
compare canonical JSON bytes exactly:

- treatment `fresh_scores.focal` versus Phase A `FF_focal`;
- treatment `fresh_scores.nonfocal` versus Phase A `FF_nonfocal`.

Any field mismatch makes the diagnostic `INVALID` and skips local harvest
comparison. The checker then reruns the tracked harvester on reconstructed raw
bytes and permits pod/local differences only in completion timestamp, treatment
artifact display path, and verified-binding display paths. Binding hashes and
all scientific fields remain exact.

The pre-outcome checker bindings are:

- checker:
  `7076bf6b01acae3f85de1a3e5c150e4593bb7d2c4d3ed222e8c63b1f31a0b46a`;
- checker tests:
  `495b458ab58499bd0ca8e26f66e4efbde780d7242a995f131b0bfdf6c55220bb`;
- pre-run integrity record:
  `e9f7ce9f31eea62f10030aafd61b751afecf2b74e41d1f9c5a345f2ce58dddfb`;
- tracked harvester:
  `9dbe29ad84cd001e13c5144605f506d52a4f6b06d00353436a49fb67bbbfdfad`.

Sol and an independent red-team agent observed all six checker tests pass. The
red-team also perturbed a binding hash, an estimand, and an unrelated field
named `path`; each was rejected, while path/timestamp-only provenance changes
passed.

The intact raw must first be pulled and receipt-hash verified. If over the
repository's file-size limit, it must be packaged losslessly using packager
`6d88f6ebdea5f86e2910108d4f8641b5116ce2bb54c73b22d3cdc2e6cba3a714`,
whose tests hash to
`667a3d9a4083512684316033e017fcc4c55ae7c33e16d9a92fd09f4d60e742f1`.
Six package tests and real/synthetic integration round trips passed. The local
reconstruction must match the original raw SHA-256 byte-for-byte; the post-run
checker runs on that reconstruction. Chunks, manifest, pod harvest, local audit,
receipt, and job log must be committed and pushed before deletion.

## Accepted residual risk and terminal order

The sealed runner persists the raw only after the full 34-arm treatment function
returns or its outer `finally` executes. A host loss or hard kill inside an
unresponsive kernel can therefore lose completed in-memory arms. Adding per-arm
checkpoints would alter the sealed apparatus; for this one sub-dollar diagnostic,
that residual is disclosed and accepted.

Terminal order is: detect receipt; pull intact artifacts; verify receipt hashes;
package and reconstruct raw; run the outcome-blind cross-host/local-harvest
audit; commit and push every result; delete the pod; verify provider 404 and zero
active pods; then inspect and interpret e01. Any failure preserves evidence but
does not expand the experiment.
