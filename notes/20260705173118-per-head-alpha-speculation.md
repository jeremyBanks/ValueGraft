# Per-Head Alpha Speculation

These are speculative notes, not a plan or commitment.

## Thought

The current ValueGraft experiments use a single global blend weight:

```text
V_new = (1 - alpha) * V_fresh + alpha * V_old
```

That global `alpha` applies across the whole model: all layers, all KV heads,
all aligned positions, and all dimensions inside each value vector.

This may be too blunt. Cached value tensors are structured by layer and KV head,
roughly:

```text
layer -> [batch, kv_head, position, head_dim]
```

So a more natural form might be:

```text
V_new[layer, head, pos, :] =
  (1 - alpha[layer, head]) * V_fresh[layer, head, pos, :]
  + alpha[layer, head] * V_old[layer, head, pos, :]
```

The global alpha found by tuning may be a compromise across heads and layers
with different tolerances:

- some heads may benefit from strong old-value grafting,
- some heads may be damaged by it,
- some layers may carry more context-conditioned interpretation,
- other layers may mostly add lexical/local or final-decision information.

The model-size difference already observed in the alpha sweeps makes this seem
plausible. At 4B, low/mid-layer gated low alpha worked best. At 30B, a larger
global alpha was tolerated and performed better. That could reflect not only a
model-level difference, but also different distributions of useful or fragile
heads.

## Why this is coherent

Attention heads are not interchangeable. Even with grouped-query attention, the
cached values are still separated by KV head. A single scalar alpha forces all
KV heads to accept the same amount of old state, even if only a subset of heads
is useful for preserving the relevant interpretation.

This suggests several possible granularities, from least to most flexible:

```text
global alpha
layer-band alpha          # early / middle / late
per-layer alpha
per-KV-head alpha
per-layer-per-KV-head alpha
per-position/head gate
```

Per-dimension alpha is probably too flexible and too easy to overfit. Per layer
or per-KV-head seems like a more plausible next granularity.

## Cautions

This would introduce many more tuning degrees of freedom, especially with a
small corpus. Any serious version would need a validation/holdout split and
strong negative controls.

Because Qwen uses GQA, the number of KV heads is small, which makes per-KV-head
alpha more tractable than per-query-head tuning. But it is still an extra
optimization layer and should not be mistaken for a mechanism result unless it
generalizes.

## Possible framing

If mentioned later, the modest version is:

> The tuned global alpha is likely a coarse compromise. Since cached values are
> organized by layer and KV head, a practical ValueGraft system might use
> per-layer or per-head blend weights, or a learned/heuristic gate, rather than
> one scalar for the entire model.

This belongs in future-work/speculation, not the current core claims.

---

## Empirical follow-up (2026-07-05, added by Claude)

Ran the diagnostic version at 4B: per-layer graft profile (α=1, one layer at a
time, all 20 conversations). Results (results/layer_profile_4b*):

- Clear structure: graftable signal concentrates in a mid-depth band (L12–L22 of
  36; peak L17 = +12.3 milli-nats from a single layer). Late layers (L24+) are
  uniformly NEGATIVE — grafting them hurts. Early layers are noise. This
  explains why mid-band gating beat global α at 4B.
- Profile-derived rule (positive-validation-delta layers, 2 DOF), evaluated once
  on holdout: α=0.75 on derived layers = +0.0182 (8/10) vs the hand-tuned
  mid-band champion's +0.0173 (10/10) — a tie. α=1.0 on derived layers still
  hurts at 4B (−0.003).
- Conclusion: at this scale the calibration procedure REPRODUCES manual tuning
  rather than beating it; its value is automation + potential transfer. Queued
  for the cloud stage: does profile-then-graft transfer across families/scales
  (procedure, not layer indices)?

## Hybrid-attention corollary (Gemma-class models)

In sliding-window hybrids (e.g. Gemma 3: ~5 local layers per global layer),
full-history conditioning can only live in the sparse global-attention layers —
the profile predicts its mass piles up exactly there. If true: mechanism
confirmation + a practical win (retain/graft only ~1/6 of layers; SelfGist state
shrinks proportionally). Requires per-layer-type surgery handling; a follow-up
study, not a replication target.
