# Precision probe P02 — outcome-blind matched weight-runtime screen

**Protocol ID:** `precision-probe-p02`

**Status:** FROZEN BEFORE ANY P01 OR P02 OUTCOME VALUE WAS ACCESSED

**Frozen by:** Sol — gpt-5.6-sol-xhigh

**Independent design input:** Claude Fable 5 (`notes/20260712A5-fable-p02-outcome-blind-fallback-review.md`) and an outcome-blind implementation audit

**Date:** 2026-07-12

## 1. Why this successor exists

P01 successfully established a true-NF4 Qwen3-MoE runtime and completed its exact technical gate. Its first unopened e01 raw outcome took approximately 38 minutes: about 9 minutes for Phase A and 29 minutes for the complete 34-arm treatment. That outcome-blind timing proved that P01's base commitment—two complete e01 repeats in each of two regimes, plus two technical ladders—cannot finish inside its frozen 7,200-second provider cap. P01 remains unchanged and bounded. A partial P01 is apparatus/timing evidence and NF4-only fixed-case evidence, not the matched comparison it preregistered.

P02 asks the same narrow precision-axis question with only the cells that have a predeclared interpretive role. It is frozen while P01 packages remain unopened. P01 timing and technical gate status informed this design; no P01 score, margin, token contribution, generated text, or arm ordering did.

## 2. Question and claim boundary

On one fixed e01 engineered case, under the same host and software stack, do these observables materially differ between:

1. bitsandbytes NF4 double-quantized **weights**, bfloat16 linear compute, bfloat16 KV cache; and
2. bfloat16 **weights**, bfloat16 linear compute, bfloat16 KV cache?

The manipulated axis necessarily combines weight representation with its linear kernels. P02 does not test 4-bit KV storage, isolate quantization error from kernel implementation, estimate a population effect, establish general efficacy, explain the old VOID MLX result, or reopen formal v12. It is an exploratory fixed-case matched screen.

## 3. Frozen subject and runtime

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Revision: `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.
- Case: `data/coherent_canary_v12/revision2/session_d/e01.json`, exact committed bytes.
- Runtime: isolated Python 3.12.11, Torch 2.12.1/CUDA 13.0, Transformers 4.57.6, Accelerate 1.14.0, bitsandbytes 0.49.2, huggingface-hub 0.36.2, safetensors 0.8.0, tokenizers 0.22.2.
- Tokenizer: pinned slow Qwen2 tokenizer and exact chat-template attestation.
- Attention: eager.
- Host: one admitted Secure A100 80 GB PCIe, one GPU, one UUID, driver at least 580.65.06, at least 80,000 MiB.
- Decoding: greedy/temperature 0 under the reused exact runtime.
- Regime order: fixed NF4 then bfloat16 on the same pod. One sequence cannot estimate an order effect; NF4-first measures the risky regime early enough to make an outcome-blind budget decision.

Archived P01 NF4 and historical Transformers-5 bfloat16 outcomes are descriptive cross-host/cross-version context only. Neither substitutes for a P02 regime.

## 4. Technical gates

Each regime must independently pass the P01 loader and technical contract before its outcome work counts:

- exact repository, dependency, host, checkpoint, tokenizer, EOS, model-class, topology, placement, and attention-backend bindings;
- exact checkpoint/meta key equality and all-shard availability;
- NF4: all 18,432 expert linears and all 18,672 eligible linears are `Linear4bit`/`Params4bit` with nested NF4 double-quant state, bfloat16 compute, expected storage, global config, and frozen sentinel coverage;
- bfloat16: all model parameters have the frozen bfloat16 coverage, with no quantized module/state;
- real forward attests bfloat16 KV dtype, shape, device, cache snapshot/rebuild, and forward API;
- generated-versus-forced identity, deterministic correct-history replay, deterministic fresh replay, and fresh self-replacement pass.

A failed technical gate stops that regime. A matched scientific description requires both regimes to pass and at least repeat 1 to complete in both. Otherwise the result is apparatus evidence only.

## 5. Exact outcome execution set

Each selected repeat is a complete, independently persisted outcome consisting of full Phase A plus a targeted treatment. Phase A is intentionally rerun for whole-payload repeat stability; its caches are **not** claimed to supply the later treatment, whose source executions remain separate.

### 5.1 Full Phase A

Run the reused exact Phase A for:

- correct-history oracle (`C_N`),
- wrong-history oracle (`W_N`), and
- fresh compacted destination (`F`),

with focal and nonfocal scoring/generation exactly as in the reused component. This supplies the gross oracle-to-fresh damage endpoint and the source-independent baseline.

### 5.2 Targeted treatment

The treatment still constructs its required source executions `C_N`, `W_N`, `C_P`, `W_P`, direct fresh, and the R2 fresh boundary. It then records the direct-fresh/`FF` baseline once and attempts these operations in this exact order:

1. `N/R2/FC` — fresh keys, correct-history values;
2. `N/R2/FW` — fresh keys, wrong-history values;
3. `N/R2/CC` — correct-history keys and values;
4. `N/R2/WW` — wrong-history keys and values;
5. `P/R2/FC` — turn-aligned correct-history values with fresh keys;
6. `P/R2/FW` — turn-aligned wrong-history values with fresh keys;
7. in repeat 1 only, one `N/R2/V_PLACEBO` construction attempt using the reused deterministic norm-matched orthogonal control.

`FF` is exactly the already-computed direct-fresh score and generation. It is a zero-increment duplicated-baseline record, not an independently executed surgery and not a placebo. Each repeat has six executed graft cells and one zero-increment fresh/FF anchor. Repeat 1 additionally has at most one executed placebo cell; repeat 2 does not repeat placebo construction. The placebo is attempted last, after all six primary cells are durably checkpointed, so failure cannot erase completed primary cells; it remains the lowest-priority operation and can still delay the later regime. Its projection budget remains exactly 1,024 deterministic attempts per first failing row; `PLACEBO_UNAVAILABLE` is missing control evidence, not a null effect and not grounds for another attempt.

Explicitly dropped: R1 and R3; key-only `CF/WF`; crossed `CW/WC`; P-schedule `CC/WW`; all R1/R3 placebo attempts; e02/e03; and every flexible post-result arm addition.

## 6. Repeats and eligibility

- **Primary matched dataset:** repeat 1 in NF4 and repeat 1 in bfloat16.
- **Repeat-stability rider:** repeat 2 in both regimes, if the frozen timing rule authorizes it.
- No third repeat.
- Repeat 2 never becomes required for the fixed-case matched description if both repeat-1 packages pass; it can only qualify numerical stability.
- A repeat comparison normalizes away paths, timestamps, and timing fields and compares the common Phase-A, fresh-baseline, and six-graft scientific payload. The repeat-1-only placebo is excluded. Any differing common field is reported. For each scalar estimand separately, compare the cross-runtime absolute delta with the maximum absolute within-regime repeat delta for that same scalar; a cross-runtime delta no larger than that repeat delta receives no numerical interpretation. Any hash or generated-text instability independently blocks E2/E3 interpretation.

## 7. Outcome-blind timing rule and hard cap

Scientific work is admitted against this provider-clock budget:

`C = min(9,000 seconds, floor($3.90 * 3,600 / hourly_rate))`

measured from the provider's creation timestamp. A stage may launch only when its frozen forecast finishes by `C - 300 seconds`. The sole graceful job interrupt occurs at `C - 180 seconds`. The OS-supervised local watcher begins its forced final pull and DELETE sequence no later than `C - 60 seconds`; an earlier complete receipt triggers immediate pull and deletion. `C` is the scientific-work cap, not a physically exact provider-deallocation timestamp. The $3.90 calculation reserves $0.10 inside the absolute $4 envelope for provider/API deletion-settlement latency; any settlement overage is recorded and never triggers a retry. At $1.39/hour, the 9,000-second scientific-work maximum is $3.475.

The approximate planning model is deliberately conservative: setup 5 minutes; each regime's load/technical ladder 12 minutes; each full targeted outcome 19–24 minutes because the complete treatment has large fixed source/boundary costs and **must not** be scaled as `29 minutes × selected_arms / 34`. Two technical ladders and four target outcomes are expected around 100–125 minutes before reserve, within the 145-minute stage-forecast ceiling (`C - 300`) but not guaranteed. The graceful interrupt point is 147 minutes (`C - 180`) when `C = 9,000`.

All continuation decisions receive only completion status and scalar monotonic timings. No path, score, margin, generation, token, or arm payload crosses the decision boundary.

After NF4 repeat 1, let:

- `e` = provider elapsed seconds,
- `L` = seconds from immediately before `prepare_precision_subject` through verified persistence of the final technical raw package and technical ledger, and `Lhat = max(L, 900 seconds)`,
- `t` = seconds from immediately before Phase-A execution through verified persistence of Phase A, treatment-foundation and per-arm checkpoints, and the final targeted-treatment raw package for NF4 repeat 1,
- `C` = the provider cap above.

`t` excludes compact-score parsing and Markdown outcome rendering. Those derived artifacts are forbidden until all continuation decisions are durably frozen; their bounded completion/persistence time is covered by the 300-second forecast reserve.

First require the matched bfloat16 repeat-1 forecast:

`e + 1.25 * (Lhat + t) <= C - 300`

If false, stop and package before loading bfloat16; an NF4-only P02 would duplicate P01 and cannot answer the matched question.

Authorize **both repeat-2 riders** only if:

`e + 1.25 * (t + Lhat + 2*t) <= C - 300`

This covers NF4 repeat 2, bfloat16 technical work, and two bfloat16 outcomes using NF4 timing as the conservative proxy and a 15-minute minimum load/gate estimate. If false, skip NF4 repeat 2 and therefore skip bfloat16 repeat 2. The exact six-graft cell set is never shrunk asymmetrically.

After bfloat16 technical work, repeat 1 runs only if `elapsed + 1.25*t <= C - 300`; failure is packaged as a timing abort. If repeat 2 was authorized, it runs after bfloat16 repeat 1 only if NF4 repeat 2 exists and `elapsed + 1.25*max(t, t_bf16) <= C - 300`; otherwise the completed unmatched NF4 repeat 2 remains a stability rider and primary eligibility still rests solely on the matched repeat-1 pair.

No P01 or P02 value can authorize more work. No surprising result adds a cell, case, repeat, dollar, or minute.

## 8. Frozen analysis

For regime `r`, let `Y_r(X)` be focal mean log-probability of the correct target minus that of the counterfactual target under cell `X`, using the reused exact float32-rounded arithmetic.

### E1 — gross compaction damage

Report continuously:

`A_r = Y_r(oracle-correct) - Y_r(fresh)`.

Report each signed value and `A_NF4 - A_bf16`. No ratio is reported. Describe sign and rough magnitude concordance in this fixed case without a pass threshold, confidence interval, power claim, or population inference.

### E2 — literal generated-behavior pattern

For each regime, report the five unique N-schedule generated-answer cells:

`fresh (= FF), FC, FW, CC, WW`.

Report the five-element literal generation tuple and hashes, plus a four-bit vector indicating whether each of `FC`, `FW`, `CC`, and `WW` differs from fresh. If the tuples/change vectors match, this is fixed-case behavioral concordance. If they differ, it is a screening flag for a new preregistration, not efficacy and not quantization dependence. Report an available placebo separately; it is not part of the tuple or four-bit vector.

### E3 — graft contrasts, descriptive only

Report:

- `D^V_r = Y_r(N/R2/FC) - Y_r(N/R2/FW)`,
- `D^KV_r = Y_r(N/R2/CC) - Y_r(N/R2/WW)`, and
- each signed NF4-minus-bfloat16 difference.

Report signed continuous values. Ratios near zero are unstable and receive no bucket or inferential label. No magnitude, sign change, or apparent interaction on this single engineered, placebo-limited, schedule-sensitive fixture licenses the phrase “quantization dependence.” Divergence licenses only a multi-case preregistration.

### Secondary/control reporting

- `P/R2/FC - P/R2/FW` and the N-versus-P shift;
- nonfocal damage and selectivity;
- FF/fresh identity;
- R2 placebo status and scores if available;
- per-token contributions;
- repeat stability;
- P02 NF4 versus P01 NF4 as cross-host/protocol context only;
- P02 bfloat16 versus archived Transformers-5 bfloat16 as cross-version context only.

No p-value, confidence interval, or generalization claim is calculated from this fixed case.

## 9. Persistence and blinding

Before any derived score access, persist and verify:

- runtime and subject attestations;
- technical identity checkpoints and final technical raw packages;
- each Phase-A raw package;
- a treatment-foundation checkpoint after source/fresh/boundary construction;
- an append-only recovery record after every completed selected arm;
- each final targeted-treatment raw package;
- outcome timing/status receipts;
- the frozen continuation decision;
- compact score/generation JSON and complete Markdown render ledgers for every completed selected outcome;
- terminal manifest, provider record, logs, and receipt inventory.

Operational timing code may inspect only scalar receipts. Human or analysis code does not parse P02 values until the matched repeat-1 decision is durably fixed. P01 outcome packages remain unopened until this preregistration is committed. Every partial/error artifact is preserved in the results namespace; nothing is deleted or silently reclassified.

## 10. Interpretation matrix

- Both technical gates and matched repeat 1 complete: report the fixed-case matched screen under E1–E3.
- E1/E2 concordant: the gross damage/behavioral pattern was observed in both matched weight/runtime regimes in this one case. No broader claim.
- E1/E2 discordant: a runtime-axis screening observation requiring a new study; not an efficacy or causal quantization result.
- E3 concordant or discordant: descriptive fixed-case numbers only.
- Placebo unavailable: weakened semantic attribution, explicitly reported.
- Technical mismatch, timing abort, or no matched repeat-1 pair: apparatus/operations evidence only.
- Any result leaves formal v12 terminal and unchanged.

## 11. Freeze rule

This file is immutable after the first P01 or P02 outcome value is accessed. Corrections are additive, dated deviations that cannot retroactively authorize a favorable interpretation. Implementation may become stricter or stop earlier; it may not add cases, cells, repeats, outcome access, budget, or claim strength without a new protocol frozen before values.
