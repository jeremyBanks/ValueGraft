# Disposition of the requested 4-bit versus bf16 precision axis

**Author:** OpenAI GPT-5.6 Sol, extra-high reasoning  
**Date:** 2026-07-12

## Decision

The paper now scopes the exact redesign and e01 diagnostic to bf16 and explicitly leaves weight-quantization and KV-cache-dtype dependence open. I did not authorize a new 4-bit treatment run in this round.

This is not because the axis is uninteresting. The original positive-looking judged result came from a 4-bit-weight MLX checkpoint and failed to survive later bf16 work. That difference could reflect weight quantization, kernels/runtime, corpus, summaries, rendering, or the many defects in the early apparatus. A clean precision interaction would be informative.

It is also important to name the axis correctly. The early MLX subject had 4-bit model weights but an fp16 KV cache, as recorded in `notes/2026070510-results.md`. A standard bitsandbytes NF4 load likewise quantizes weights; it does not make the KV cache 4-bit. An NF4-versus-bf16 comparison would therefore test a bundled weight-representation/runtime interaction while holding the intended cache-state surgery fixed. A distinct KV-cache-dtype experiment would require explicitly casting/storing K/V at the desired precision and validating that intervention separately.

## Why it was not appended now

1. Formal v12 stopped at its technical path-control rule before any eligible treatment result.
2. The sole post-stop e01 execution is N=1, schedule-sensitive, and lacks all three planned matched perturbation controls.
3. There is therefore no cleared bf16 treatment estimand to use as one side of a clean precision interaction.
4. A 4-bit subject would need its own L0/L1/L3 ladder, production-dtype control-construction gate, exact schedule qualification, and matched placebo availability. Treating NF4 as a drop-in rerun would repeat the project's central mistake of assuming numerical equivalence across apparatuses.
5. Paid data collection was closed. Spending to add a post-hoc precision arm to a formally stopped assay has lower information value than first repairing the assay.

## Re-entry design

If the project resumes, qualify two frozen subject runtimes on the same cases and schedules:

- bf16 weights/compute with explicitly recorded KV dtype;
- NF4 weight quantization with the same explicitly recorded KV dtype and otherwise matched harness.

Require all technical ladders and all matched controls to pass independently in each runtime before opening outcomes. Analyze treatment, runtime, and their interaction; do not infer a KV-precision interaction unless KV dtype itself is the randomized axis. If KV-cache precision is of interest, add a separate bf16-versus-fp16-versus-lower-precision cache-storage experiment with native-position controls and exact tensor provenance.

This preserves the owner's substantive concern without pretending the early 4-bit result identified its cause or that a new run could repair the stopped bf16 assay retroactively.
