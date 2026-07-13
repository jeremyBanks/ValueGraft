# FINDINGS.md — current observed facts and unknowns

The pre-takeover findings ledger is archived in `notes/` under the slug
`legacy-findings-through-20260712`. This root file contains only observations
that remain safe inputs to the powered successor.

## What is observed

- **Independent semantic N is one fixture (`e01`).** P01 and P02 repeated the
  same fixed computation and do not add population units.
- On e01, fresh compaction caused `22.298429` nats of focal margin damage and
  `22.290991` nats of correct-target-logprob damage.
- The best v12 value-only cell recovered `0.81%` of margin damage and `4.24%`
  of correct-target-logprob damage. Full-K+V was nonselective at the primary
  cell. Results varied by schedule, region, and precision.
- No primary cell changed the generated wrong answer to the correct target.
- Every frozen exact-bf16 matched-placebo construction was unavailable after
  applied-dtype checks. This is missing control evidence, not a null placebo.
- P01/P02 observed exact equality of their specified same-fixture fields across
  two A100 UUIDs and adjacent driver patches, including within-regime repeats.
  This is fixed-computation reproducibility evidence on those hosts only.
- The P01/P02 NF4 axis quantized eligible weights while KV remained bf16 and
  also changed linear-kernel implementation. It is not a causal test of 4-bit
  KV state.
- Realistic-prefix schedule/query-shape changes moved selected margins by about
  `0.06`--`0.13` nat in the tested local fixtures. Schedule is an experimental
  condition/numerical floor, not a replicate.
- Existing old conversation banks have mixed provenance and decoded-quality
  defects. Some may remain useful after literal audit, but none is presumed
  valid for a successor primary sample.

## What is not established

- No population confidence interval, p-value, equivalence result, or confident
  upper bound exists.
- No statistically supported semantic-transfer, mitigation-efficacy,
  quantization-causality, live-agent, or broad model/population claim exists.
- No result establishes absence of write-time information outside the tested
  carrier/boundary rows.
- A mean near zero in one fixture does not rule out heterogeneity, a localized
  layer/alpha effect, a downstream request/header or retained-tail channel, or
  another model/regime.
- There is no final paper-ready scientific conclusion yet.

## Load-bearing evidence

- `results/coherent_canary_v12_harvest/`
- `results/coherent_canary_v12_postrun_audit/`
- `results/precision_probe_p01_analysis/`
- `results/precision_probe_p02_analysis/`
- `results/precision_probe_p01_p02_comparison/`
- `notes/` slugs `gap-analysis-asked-vs-did-and-upper-bound` and
  `20260712-handoff-to-new-ultra-agent`

New findings enter this file only after their raw artifacts, sampling unit,
controls, uncertainty, and claim boundary have been independently checked.

## 2026-07-12 direct-local technical findings (semantic N=0)

- The fully cached local MLX conversion resolved to
  `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` revision
  `e9675aa3ca5f900ccef55267914466d55ab325fa`. Its literal config records 48
  layers, 32 attention heads, four KV heads, head dimension 128, default 4-bit
  group-64 weights, and 8-bit group-64 MoE gate overrides. Runtime model leaves
  included 386 uint32 packed arrays and 965 bf16 arrays.
- L0 and L3 identity gates passed on that exact local model. The long c01
  technical run completed without swap. The old seven continuation arms took
  1--3 seconds each; 70 greedy probe generations dominated its 345-second wall
  time.
- The c01 correct value-graft movement was `+0.00671875` nat/token
  (`-1.50890625 - -1.515625`), exactly repeating an existing old c01 result.
  It is neither a new render nor a new conversation.
- An applied fresh-value sham routed 1,973 rows through the exact graft path.
  Across all 48 layers its cache was bit-exact to untouched fresh B and both
  held-out scores were exactly `-1.515625`. This rules out a generic assignment-
  path perturbation in that canary; it is an identity gate, not a nonsemantic
  placebo.
