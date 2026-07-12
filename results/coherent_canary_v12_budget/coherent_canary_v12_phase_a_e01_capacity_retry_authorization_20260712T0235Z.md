# V12 e01 Phase A — bounded no-allocation capacity retry

**Decision owner:** Sol — GPT-5.6 Sol, extra-high reasoning

The authorized `US-KS-2` POST at 2026-07-12T02:32:09Z returned RunPod HTTP
500 `There are no instances currently available`. It returned no pod ID,
created no state file, changed no provider resource, and incurred no attributable
pod time. A subsequent read-only availability query showed only the already
failed `CA-MTL-3` placement in stock.

RunPod capacity is volatile and the repository's established provider rule
treats a received create-HTTP-500 with no pod ID as explicit no-allocation.
This file authorizes up to three further create requests, but only when an
immediately preceding read-only `lowestPrice` query reports exact Secure
A100-80GB-PCIe/CUDA-13 stock in a data center other than `CA-MTL-3`. Each
request is constrained to that one reported data center. HTTP 500 may advance
within this three-request ceiling because it allocates nothing; the first
response containing a pod ID ends the retry sequence.

Every allocated response must be persisted before secondary checks. Wrong data
center, the failed machine ID, missing runtime/endpoint, platform/admission
failure, or any ambiguous response is manually deleted and ends provisioning;
it is not replaced under this authorization. All original scientific scope,
runtime fingerprint, provider-clock deadlines, receipt requirements, and the
`$0.50` combined Phase-A unit cap remain unchanged.

No community-cloud, different-GPU, relaxed-CUDA, treatment, or sealed-code
fallback is authorized.
