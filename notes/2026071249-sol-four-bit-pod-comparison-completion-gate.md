# Four-bit pod-comparison completion gate

**Author:** OpenAI GPT-5.6 Sol (extra-high reasoning)  
**Recorded:** 2026-07-12  
**Disposition:** satisfied by observed, committed pod evidence; required before final publication

The owner requires a real four-bit model comparison on GPU pods before this
project may be declared complete. I independently re-read the losslessly
packaged technical attestations, terminal receipt, and P01/P02 comparator on
2026-07-12. The requirement is satisfied by the P01/P02 NF4-versus-bfloat16
precision screen. This note is a completion-gate record, not a new scientific
claim and not authorization for another paid run.

## What actually ran

Both P01 and P02 loaded the exact checkpoint
`Qwen/Qwen3-30B-A3B-Instruct-2507` at revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` on separate NVIDIA A100 80 GB
PCIe pods. In each pod execution, the NF4 regime was compared with a bfloat16
weight regime under the same experiment fixture. P02 completed two repeats in
each regime; P01 completed both NF4 repeats and bfloat16 repeat 1 before its
terminal interruption.

The NF4 technical attestation on each pod recorded all of the following:

- `load_in_4bit: true`, `bnb_4bit_quant_type: nf4`, double quantization enabled,
  uint8 quantization storage, and bfloat16 linear compute;
- 18,672 bitsandbytes `Linear4bit` modules for 18,672 eligible linears:
  18,432 expert projections, 192 attention projections, and 48 routers;
- exact logical coverage of all 29,909,581,824 quantization-eligible weight
  elements; and
- a real forward pass with bfloat16 K/V cache tensors in all 48 layers.

The two pod hosts were physically distinct:

| Run | GPU UUID | Driver | Technical gate |
|---|---|---|---|
| P01 | `GPU-470c3e18-9f87-0a04-f3bd-e974d59e1903` | `580.159.04` | `PASS` |
| P02 | `GPU-8a42830e-71ab-fb23-351d-125ccbd5bdb2` | `580.159.03` | `PASS` |

P02's receipt reports runner `COMPLETE`, terminal `PASS`, runner and packaging
exit code 0, four outcome packages, and 42 lossless packages. The P01/P02 exact
comparator also reports `PASS`; all normalized common scientific fields matched
across the two hosts, and P02's within-regime repeats matched exactly.

## Audit anchors

- P01 packaged technical record:
  `results/precision_probe_p01/precision-probe-p01_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z/precision-probe-p01-technical-nf4-raw_Qwen3-30B-A3B-Instruct-2507_20260712T075751683593Z.lossless-package/`
- P02 packaged technical record:
  `results/precision_probe_p02/precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z/precision-probe-p02-technical-nf4-raw_Qwen3-30B-A3B-Instruct-2507_20260712T103106714544Z.lossless-package/`
- P02 terminal receipt:
  `results/precision_probe_p02/precision-probe-p02-receipt_Qwen3-30B-A3B-Instruct-2507_20260712T103011764456Z.json`
- Exact cross-pod comparator:
  `results/precision_probe_p01_p02_comparison/precision-probe-p01-p02-exact-comparison_Qwen3-30B-A3B-Instruct-2507_20260712T120014514457Z.json`
  (SHA-256
  `588229df5f4ca5c8613dc4f21564043214bfe889fc4dba176300ace0435442be`)

## Scientific boundary

This is a genuine **four-bit-weight model comparison on pods**, not a four-bit
KV-cache experiment. Both regimes used bfloat16 KV caches. NF4 versus bfloat16
also bundles weight representation with the corresponding bitsandbytes versus
ordinary linear-kernel implementations, so it does not isolate a causal
quantization effect. The screen used one engineered fixture, had no available
matched placebo, and produced no correct graft generation. Its defensible result
is exact reproducibility of the fixed NF4/bfloat16 computation across the two
observed A100 hosts—not semantic recovery, efficacy, population generalization,
or a live-agent benefit.

Final publication remains blocked if the paper or handoff omits this comparison,
misstates it as four-bit KV, or upgrades the result beyond that boundary.
