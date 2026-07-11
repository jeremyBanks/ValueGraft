# Schedule discontinuity and the revised claim boundary

**Author:** Sol — OpenAI GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11  
**Status:** Working scientific assessment after the first natural v10 failure; the preregistered origin diagnostic and second natural fixture were not yet complete when this note was written.

## Bottom line

The current contrast is not adequately described as “synthetic passes, natural fails.” The synthetic gate tested seven lengths of one five-token periodic stream. It was a useful smoke test, but it was not a content-diverse validation of schedule invariance. Its exact-zero result therefore cannot authorize an experiment on realistic conversations.

Conversely, c10 does not yet establish a semantic, linguistic, or even purely numerical explanation. It establishes that one realistic 8,430-token fixture produces a large deterministic disagreement between two supposedly equivalent prefill schedules in the exact local production configuration. Until the origin diagnostic excludes future-token or cache/mask construction influence, the correct classification is **unexplained apparatus failure**.

This invalidates v10 as an authorizing ladder. It does not validate the intended ValueGraft mechanism, and it does not by itself show that natural language is intrinsically more schedule-sensitive than synthetic text.

## What was actually observed

Configuration: Qwen3-0.6B, CPU, BF16, PyTorch eager attention, temperature zero.

The committed synthetic battery used the periodic token stream corresponding to:

> alpha beta gamma delta epsilon

The five tokens were cycled to each requested length. All seven comparisons were bit-exact at every recorded outcome. The long comparisons included:

- 4,096 tokens: `[4096]` versus `[32, 4064]`
- 4,097 tokens: `[4096, 1]` versus `[32, 4065]`
- 8,193 tokens: `[4096, 4096, 1]` versus `[32, 4096, 4065]`

Thus, prefix length alone cannot explain the natural failure.

For natural fixture c10, the same 8,430 token IDs and logical positions were evaluated with:

- ordinary schedule: `[4096, 4096, 238]`
- message-aligned schedule: `[23, 4096, 4096, 92, 123]`

Observed maximum disagreements were:

- cache keys: `16.125`
- cache values: `5.125`
- final logits: `0.59375`
- fixed selected-margin shift: `0.060546875`
- continuation logits: `0.84375`
- continuation keys: `1.0`
- continuation values: `1.59375`

Layer 0 cache projections were exact. The first recorded divergence appeared at layer 1 (`K=0.0625`, `V=0.0009765625`), after layer 0 attention had executed with a different query shape. The key disagreement later peaked at `16.125` around layer 12.

The `16.125` value is not an inferential effect size and should not be called statistically significant. It is a maximum coordinate disagreement whose practical scale depends on the tensor distribution. The fixed margin shift is more directly decision-relevant, but it is still one fixture, not a population estimate.

## Why exact zero beside a large disagreement is possible—but not yet explained

Finite-precision computations are discontinuous after rounding. Two different reduction orders can land in the same BF16 bins for one input and in different bins for another. If the first rounded state remains identical, the subsequent deterministic computation can remain bit-identical. If one value crosses a rounding boundary, the initial difference can propagate and amplify through later attention and MLP layers.

That is a mathematically plausible account of the observed pattern:

1. the periodic stream remains in identical BF16 bins under both schedules;
2. c10 first crosses a bin after layer 0 attention;
3. the small layer-1 disagreement amplifies through the network.

However, this remains a theory. The same pattern could still be produced by a subtle causal-mask, cache-shape, or construction error whose effect is triggered by the 23-token boundary. Token hashes, position coverage, layer-0 equality, and repeatability make a gross token or position error unlikely, but they do not exclude a subtle attention-path error.

## The preregistered discriminant

The frozen `coherent-state-c10-schedule-origin-v1` diagnostic evaluates:

- **A:** the first 23 c10 tokens in a 23-token query;
- **B:** the first 4,096 c10 tokens in a 4,096-token query;
- **C:** the first 23 c10 tokens followed by c02 tokens through position 4,095, also in a 4,096-token query.

B and C have identical shapes and identical first 23 tokens but different future tokens. Each branch is repeated.

Interpretation:

- B/C disagreement within the causally protected first 23 positions indicates future-token influence or another construction/mask defect.
- B/C exact agreement in the first 23, together with A/B disagreement after layer 0, isolates query-shape-dependent numerical execution as the source.
- A/B layer-0 cache disagreement indicates a construction defect before attention.
- If the first boundary is exact, the preregistered full-prefix fallback finds the later first divergence.

This diagnostic classifies the origin of the schedule disagreement. It does **not** rescue v10 and does not establish that the semantic estimator is robust.

## Consequences for the experiment

The earlier ladder made a category error: exact equality on a low-entropy periodic path was treated as evidence for invariance over realistic paths. Future gates must separate three questions:

1. **Construction correctness:** identical tokens, positions, masks, and intended cache semantics.
2. **Execution-path sensitivity:** multiple heterogeneous realistic fixtures, not only one periodic stream, evaluated under the schedules the experiment will actually use.
3. **Estimand robustness:** the scientific contrasts themselves must agree across at least two valid schedules. Absolute cache or logit disagreement may cancel in a difference, or it may create systematic bias; that has to be measured rather than assumed.

The periodic stream can remain a cheap smoke test. It cannot again serve as the decisive invariance gate.

No paid semantic run should begin from the current v10 artifact. The next sequence is:

1. preserve the second natural fixture and stop the old ladder;
2. run and seal the origin diagnostic;
3. generate and save a realistic local render once;
4. measure the ValueGraft contrasts under both a canonical and an alternate valid schedule;
5. proceed to paid work only if the inference is schedule-robust under a criterion frozen before looking at semantic outcomes.

## Relationship to prior literature

The general numerical phenomenon is established rather than novel. PyTorch documents that floating-point operations are not associative and that mathematically identical computations need not be bitwise identical across implementations or batched forms ([Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)). Thinking Machines gives a directly relevant account of batch-size and chunked-prefill invariance: a token's result can change when the reduction shape changes unless kernels deliberately preserve invariant reduction order ([Defeating nondeterminism in LLM inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)). A NeurIPS 2025 study further reports that BF16 inference variation under batching and hardware changes can reach generated behavior ([NeurIPS abstract](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f80094a824ba5912d4a2de169c404a40-Abstract-Conference.html)).

The potentially publishable contribution is narrower: an assay-specific demonstration that prefill partition can contaminate a history-conditioned KV-state grafting estimand, together with controls that distinguish this contamination from semantic state. We must not claim discovery of floating-point non-invariance, universality across models/backends, or behavioral divergence from c10 alone.

## Confidence statement

High confidence:

- the synthetic and c10 results differ exactly as reported;
- length alone is not the separator;
- the prior synthetic gate was insufficient;
- v10 cannot authorize semantic conclusions.

Unresolved:

- whether the c10 divergence is query-shape rounding or a subtler attention/cache defect;
- whether c02 reproduces the disagreement;
- whether schedule effects cancel or bias the actual ValueGraft contrasts;
- whether a schedule-robust paid experiment remains practical within the available budget.

