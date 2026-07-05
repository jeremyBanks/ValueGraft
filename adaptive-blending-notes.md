# Adaptive Blending Notes

These are quick notes on a possible refinement of the current ValueGraft /
SoftGraft direction.

## Core idea

The current implemented value-graft arm uses a fixed global blend:

```text
V_new = (1 - alpha) * V_fresh + alpha * V_old
```

where `alpha` is the same everywhere for a given run, such as `0.25`, `0.5`,
`0.75`, or `1.0`.

That tests whether old write-time values help at all, but it is crude. Full
replacement can be harmful because old values may be partially incompatible
with the freshly encoded compacted context. Low-alpha blending helps preserve
the fresh context while allowing some old interpretation to influence the
model.

The refinement is adaptive blending:

```text
V_new = (1 - gate_i) * V_fresh + gate_i * V_old_or_retrieved
```

where `gate_i` varies by token, layer, head, or position.

## Why blend rather than replace

- Old value state may carry the original context-conditioned interpretation.
- But old state may also be stale, mismatched, or incoherent with fresh keys.
- Strong replacement can create confident wrongness or general degradation.
- A gate lets the model trust old state only when there is evidence that it is
  relevant.

## Possible gates

### Exact-span confidence

Use more old value when the new token belongs to a long exact-matching span
from the old conversation. Use little or none for short/common matches.

This is closest to the current E-post setup, but makes alpha local instead of
global.

### Attention confidence

Use the model's own attention as the correspondence signal. For a compacted
token, score its query against old keys. If attention sharply points to one
old region, blend more. If attention is diffuse, blend less.

This avoids treating raw value-vector cosine distance as meaningful. QK
attention is the model's trained retrieval mechanism.

### Top-match margin

Blend more when the best old position is much stronger than the second-best.
Blend less when many old positions look similarly plausible.

### Head or layer agreement

Blend more when multiple heads or layers retrieve the same old span. Blend
less when they disagree.

### Layer gating

Try applying old values only in selected layer bands. Middle layers may carry
more semantic interpretation; early layers may be too lexical, and late layers
may be too decision-specific.

## Relation to current experiment

Already implemented:

- Fixed-alpha E-post value grafting at exact token matches.
- Negative controls using shuffled and wrong-conversation values.

Not yet implemented:

- Per-token/head/layer adaptive gates.
- Attention-retrieved values for paraphrased compacted text.
- Confidence weighting based on retrieval sharpness or agreement.

This means the adaptive idea is not part of the current pilot results. It is
closest to the planned G / SoftGraft direction.

## Better framing

The practical question is not whether old KV state differs from fresh KV
state. The practical question is whether we can use a small, deployable cache
intervention to reduce compaction damage.

Adaptive blending is one possible route:

> Freshly encode the compacted context, then inject old value state only where
> there is high confidence that the old state corresponds to the new token or
> summary phrase.

## Caution

Avoid using raw cosine similarity in value space as the main notion of
"closeness." Value space does not necessarily have an interpretable geometry
on its own. Attention scores are a safer starting point because they are part
of the model's learned retrieval machinery.
