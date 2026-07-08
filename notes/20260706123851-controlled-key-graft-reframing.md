# Controlled KV-Graft Reframing

This note records the revised terminology and experimental-control rule for
ValueGraft-style interventions. The purpose is to make future comparisons
scientifically interpretable while preserving the meaning of experiments already
in flight.

## Core Model

**ValueGraft** should refer to the family of interventions that reuse or blend
cached key/value state across a compaction boundary.

For an aligned token in the compacted context, the general form is:

```text
K(alpha_K) = (1 - alpha_K) * K_fresh
             + alpha_K * K_write_time_rerotated

V(alpha_V) = (1 - alpha_V) * V_fresh
             + alpha_V * V_write_time
```

`K_fresh` and `V_fresh` come from ordinary re-encoding of the compacted context.
`K_write_time_rerotated` and `V_write_time` come from the state the model wrote
when the summary token was generated. If a write-time key is used at a different
position, it must first be re-rotated into the compacted position.

`alpha_K` and `alpha_V` may be constants or structured policies, including
per-layer, per-head, token/span-dependent, or tuned matrices. A bare `alpha`
should only be used when it is clear from context that the same setting applies
to both K and V, or when the experiment only has one active alpha parameter. For
new design notes, prefer explicit `alpha_K` and `alpha_V`.

## Named Regions

These are not separate algorithms so much as regions of the same parameter
space:

| Name                     | Key policy                  | Value policy              |
| ------------------------ | --------------------------- | ------------------------- |
| Plain Summary Compaction | `alpha_K = 0`               | `alpha_V = 0`             |
| V-only Graft             | `alpha_K = 0`               | `alpha_V` varied or tuned |
| K-only Graft             | `alpha_K` varied or tuned   | `alpha_V = 0`             |
| KV-Graft                 | `alpha_K` varied or tuned   | `alpha_V` varied or tuned |
| Coupled KV-Graft         | `alpha_K = alpha_V = alpha` | same shared alpha         |

`alpha = 1` is not a separate method. It is the parameter setting where that
side uses write-time state exactly. `alpha = 0` means that side uses fresh
compacted-context state exactly. Values above 1 are extrapolation settings and
should be described as such.

Do not introduce a separate method name for the `alpha = 1` case. If needed, say
`alpha_V = 1`, `alpha_K = 1`, or `alpha_K = alpha_V = 1`.

## Current Experiment State

The in-flight coding experiments using `E`, `E:a0.75`, `E:a1.0`, and
`E:cfg=layers` should be interpreted as **V-only Graft** experiments:

```text
alpha_K = 0
alpha_V = constant or tuned, depending on the arm
```

Those experiments do not need to be renamed or interrupted. Existing internal
identifiers, specs, result directories, and logs should remain stable for
provenance. In reports, translate them into the clearer conceptual vocabulary.

The historical `H-pack` arm should be interpreted as a packed/minimal-layout
KV-retention experiment, not as the clean V-only versus KV comparison. It used a
packed layout with sinks plus summary-token state, while the live coding setup
uses a production-shaped compacted context with summary plus recent tail. That
layout/tail difference is a confound for claims about key policy.

The historical packed arms are still useful evidence about the behavior of that
packed layout. They should not be used as if they isolate the effect of fresh
keys versus write-time keys in the same compacted context.

## Controlled Comparisons

For any future comparison, the intended experimental axis must be the axis that
actually changes.

To test key policy:

- hold `alpha_V` fixed
- hold summary text fixed
- hold summary token IDs fixed
- hold compacted-context layout fixed
- hold retained tail fixed
- hold token positions fixed
- hold continuation prompt and decoding fixed
- vary only `alpha_K` or the key policy

To test value policy:

- hold `alpha_K` fixed
- hold the same context, layout, tail, positions, prompt, and decoding fixed
- vary only `alpha_V` or the value policy

To test a coupled KV policy:

- state that both K and V are changing by design
- keep summarization, layout, tail, token positions, prompt, and decoding fixed
- compare against appropriate one-sided and plain-compaction controls when
  possible

The central rule is simple:

```text
Do not change summarization, layout, tail retention, token positions, or prompt
shape unless that is the experimental axis being claimed.
```

## Corrected Future Arm Family

A future clean implementation should build all arms inside one shared compacted
context:

1. Generate the summary once from the original conversation prefix.
2. Build the compacted prompt once, with the same summary and same retained
   tail.
3. Freshly prefill that exact compacted prompt to obtain `K_fresh` and
   `V_fresh`.
4. Align identical summary tokens, and any other explicitly chosen identical
   spans, between the write-time trace and the compacted prompt.
5. Re-rotate write-time keys for those aligned tokens into the compacted
   positions.
6. Construct arms by applying the chosen `alpha_K` and `alpha_V` policies.
7. Compare arms only when all non-target axes are identical.

Examples:

```text
Plain Summary Compaction:
  alpha_K = 0, alpha_V = 0

V-only Graft:
  alpha_K = 0, alpha_V = tuned

K-only Graft:
  alpha_K = tuned, alpha_V = 0

Coupled KV-Graft:
  alpha_K = alpha_V = tuned

Independent KV-Graft:
  alpha_K = tuned_K, alpha_V = tuned_V
```

This framing lets us ask cleaner questions:

- Does value grafting help when keys are fresh?
- Does key grafting help when values are fresh?
- Do key and value grafting interact?
- Is the best policy scalar, layer-specific, head-specific, token-specific, or
  some structured combination?

## Naming Guidance

Use method names that describe the active policy:

| Historical name | Recommended report name                      |
| --------------- | -------------------------------------------- |
| `B`             | Plain Summary Compaction                     |
| `E` / `E:a...`  | V-only Graft, with stated `alpha_V` policy   |
| `E:cfg=layers`  | Layer-tuned V-only Graft                     |
| `B-min-pack`    | Packed Fresh-KV Control                      |
| `H-pack`        | Packed KV-Graft, auxiliary layout experiment |

Do not reuse `H-pack` for a future matched-context key experiment if the
packed/minimal-layout implementation detail has been removed. That future arm
should be named by its actual policy, for example K-only Graft, KV-Graft, or
Coupled KV-Graft, with the relevant `alpha_K` and `alpha_V` settings stated.

The old identifiers can remain in code and result files where changing them
would create provenance risk. The write-up vocabulary should use the clearer
policy names.
