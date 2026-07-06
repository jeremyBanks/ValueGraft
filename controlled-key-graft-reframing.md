# Controlled Key-Graft Reframing

This note records a required tightening of the ValueGraft terminology and
experimental design. The goal is to separate exploratory arms that changed
multiple things at once from the cleaner causal comparison we now actually
want.

## Core Correction

The main scientific question is not whether a packed layout, a different tail
policy, or a different summary presentation helps. Those can be useful
engineering choices or auxiliary controls, but they are not the key causal
axis.

For a clean comparison, the intended difference between arms must be the axis
we claim to be testing. In particular, for a key-policy comparison, the only
intended difference between the two state-preserving variants should be:

```text
How are the cached keys for the same summary tokens constructed?
```

Everything else should be held fixed:

- same source conversation
- same generated summary text
- same summary token IDs
- same compacted context layout
- same recent tail
- same token positions for the summary and tail
- same continuation prompt
- same value vectors, or the same value-blending rule
- same decoding and evaluation settings

If an arm drops the tail while another keeps it, or if one arm uses a different
layout, then that comparison is not an isolated test of key handling.

## Revised Vocabulary

Use **ValueGraft** as the umbrella name for cache-state interventions across a
compaction boundary. Under that umbrella, describe variants by their key and
value policies, not by incidental layout names.

### Plain Summary Compaction

Freshly re-encode the compacted prompt:

```text
K = fresh compact-context keys
V = fresh compact-context values
```

This is the ordinary summary baseline.

### V-Graft

Use fresh compact-context keys, but blend the values for aligned tokens toward
their write-time values:

```text
K = fresh compact-context keys
V = write-time/blended values
```

At `alpha = 0`, V-Graft is identical to plain summary compaction. At
`alpha = 1`, it uses the write-time values exactly. At intermediate alpha, it
interpolates between fresh values and write-time values. At alpha greater than
1, it extrapolates past the write-time value in the same value direction.

### KV-Graft

Use write-time keys and write-time values for the same aligned tokens. If the
tokens are placed at different positions in the compacted prompt, the cached
keys must be position-corrected by RoPE re-rotation:

```text
K = write-time keys, re-rotated to the compact-context positions
V = write-time/blended values, using the same value rule as the matched V-Graft arm
```

For the cleanest first key-policy contrast against V-Graft, use the same alpha
in both arms. Then the remaining difference is whether the keys are freshly
computed from the compact context or preserved from write time and re-rotated
into place. `alpha = 1` is not a distinct method; it is simply one parameter
setting where the value side is entirely write-time.

### Alpha Policy

Alpha is a value-policy parameter. It controls how much of the value vector is
drawn from the fresh compact-context encoding versus the write-time encoding:

```text
V_blend(alpha) = (1 - alpha) * V_fresh + alpha * V_write_time
```

Alpha may be a single constant, a per-layer setting, a per-head setting, a
token/span-dependent setting, or a tuned matrix. Those are all valid
experiments as long as the active comparison is named clearly and all unrelated
axes are controlled.

Important boundary cases:

- `alpha = 0` turns off value grafting. In V-Graft, because the keys are also
  fresh, this is identical to plain summary compaction.
- `alpha = 1` uses the write-time values exactly. This is not a separate
  method; it is one point in the same alpha parameterization.
- `alpha > 1` extrapolates past the write-time value in the same value
  direction.
- In KV-Graft, `alpha = 0` would still leave a keys-only intervention if the
  keys are taken from write time. It is therefore not a full no-op unless the
  key policy is also fresh.

When comparing key policies, the alpha policy must be identical across the
arms. When comparing alpha policies, the key policy and all context/layout
details must be identical across the arms.

## What Was Wrong With The Old `H-pack` Contrast

The historical `H-pack` arm was useful as an exploratory packed-cache arm, but
it should not be treated as the clean conceptual comparison against V-Graft.
It changed more than key handling.

In particular, the implemented packed arm used a minimal packed layout: sinks
plus summary-token state. That differs from the production-shaped compacted
context used by the live coding shim, which keeps a summary plus a recent
tail. Dropping the tail changes the task-visible context and therefore
confounds the comparison.

So the old packed results should be preserved under their historical
identifiers for provenance, but interpreted narrowly:

- They can inform us about packed summary-only cache retention.
- They can inform us about honesty or caution effects in that layout.
- They cannot, by themselves, answer whether write-time keys outperform fresh
  keys when all other context and value choices are held fixed.

## Corrected Controlled Comparison

The corrected key-policy comparison should be built in one shared compacted
context:

1. Generate the summary once from the original conversation prefix.
2. Build the compacted prompt once: summary plus the same retained tail.
3. Freshly prefill that exact compacted prompt to obtain the plain compaction
   baseline.
4. Align identical summary tokens, and any other explicitly chosen identical
   token spans, between the write-time trace and the compacted prompt.
5. Construct V-Graft by blending only the values at those aligned positions,
   leaving fresh keys in place.
6. Construct KV-Graft by using the same values as V-Graft, but taking the
   corresponding keys from the write-time cache after re-rotating them to the
   same compacted positions.
7. Compare V-Graft and KV-Graft only when summary text, token positions,
   values, tail, prompt shape, and decoding settings are all identical.

For example, at `alpha = 1`:

```text
V-Graft(alpha=1):  K_fresh + V_write_time
KV-Graft(alpha=1): K_write_time_rerotated + V_write_time
```

That isolates the key policy at that alpha setting. The same rule applies
across the sweep:

```text
V-Graft(alpha):  K_fresh + V_blend(alpha)
KV-Graft(alpha): K_write_time_rerotated + V_blend(alpha)
```

The alpha policy may be scalar, per-layer, per-head, or otherwise structured.
It must not vary between V-Graft and KV-Graft in a key-policy comparison.

The same control principle applies in the other direction: if the experiment
is an alpha-policy comparison, then key policy, summary text, token layout,
tail retention, and decoding must remain fixed.

## Naming Guidance Going Forward

The code and existing result artifacts may keep historical identifiers such as
`E`, `H-pack`, or `B-min-pack` where changing them would risk provenance or
interrupt active runs. But future descriptions should not elevate those names
into conceptual categories.

Recommended conceptual names:

| Conceptual name | Key policy | Value policy | Context/layout requirement |
|---|---|---|---|
| Plain Summary Compaction | fresh | fresh | matched compacted context |
| V-Graft(alpha) | fresh | blended by alpha | matched compacted context |
| KV-Graft(alpha) | write-time, re-rotated if moved | blended by same alpha | same matched compacted context |
| Packed Fresh-KV Control | fresh | fresh | packed/minimal layout; auxiliary only |
| Packed KV-Graft | write-time, re-rotated if moved | write-time | packed/minimal layout; auxiliary only |

The word "packed" should describe a layout condition, not the headline method.
If a packed layout is studied, it should be named as such and treated as a
separate layout experiment or auxiliary control.

## Implication For Current Interpretation

We should not discard the exploratory packed-arm results, but we should stop
using them as if they cleanly answer the V-Graft versus KV-Graft question. The
right interpretation is:

```text
Old packed arms: evidence about packed summary-only retention and its behavior.
Matched V/KV arms: required evidence about whether preserving keys matters.
```

This is a basic experimental-control issue. Now that the relevant options are
clearer, the next implementation should minimize variation rather than
continue comparing arms that differ in both cache state and visible context.
