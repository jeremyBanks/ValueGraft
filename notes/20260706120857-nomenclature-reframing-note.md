# Nomenclature Reframing Note

This note records a proposed naming cleanup for future descriptions of the
experiment. It is not a request to rename code, result directories, historical
logs, or active arm identifiers. During active experimentation, internal names
such as `E`, `H-pack`, `B-min-pack`, and result directory names should remain
stable for provenance. The names below should be treated as public-facing or
write-up-facing aliases until the experiment is no longer in flight.

## The Problem With the Current Names

The current names mix two different axes:

1. **Layout:** where the retained tokens live in the cache.
   - compact summary + tail
   - packed summary-only layout
   - gapped/original-position layout

2. **State source:** whether keys and values are freshly encoded or preserved
   from write time.
   - fresh K + fresh V
   - fresh K + write-time/blended V
   - write-time K + write-time V

Names like `H-pack` emphasize the layout axis. That is why they are confusing.
Both `B-min-pack` and `H-pack` are "packed"; packing is not the experimental
contrast. The meaningful contrast is which parts of the KV state are fresh
versus preserved from write time.

## Conceptual Model

Use the shorthand:

- **K / keys:** attention addresses. They affect whether later tokens attend to
  a prior token. In RoPE models, cached keys include positional rotation.
- **V / values:** attention payloads. They determine what information is read
  out if a later token attends to that prior token.

When a summary token was originally written at position `p` but is moved to a
packed position `q`, preserving its write-time key requires a positional
correction:

```text
rotated write-time key = RoPE(q) * W_k(h_write_time)
fresh key              = RoPE(q) * W_k(h_fresh_compact_context)
```

These are not equivalent unless the write-time hidden state and the fresh
compact-context hidden state are identical. The whole hypothesis is that they
are often not identical. Key rotation is therefore not just an implementation
detail; it is the operation that lets us preserve a write-time key while placing
it in a new packed layout.

Values do not need RoPE re-rotation. Blending or replacing values is literally
changing the value payload while leaving the key/address side alone.

## Recommended Names

Use **ValueGraft** as the umbrella for methods that preserve or reuse write-time
KV state across a compaction boundary. Within that family, name the variants by
which parts of KV state they preserve.

| Historical name  | Recommended name              | Keys                                  | Values                    | Layout                 |
| ---------------- | ----------------------------- | ------------------------------------- | ------------------------- | ---------------------- |
| `A`              | Full-Context Oracle           | fresh from full context               | fresh from full context   | full context           |
| `B`              | Plain Summary Compaction      | fresh                                 | fresh                     | compact summary + tail |
| `E` / `E-tuned`  | V-Graft                       | fresh                                 | blended write-time V      | compact summary + tail |
| `E:a0.75`        | V-Graft Blend                 | fresh                                 | interpolated write-time V | compact summary + tail |
| `E:a1.0`         | V-Graft Replace               | fresh                                 | write-time V              | compact summary + tail |
| `E:cfg=layers`   | Layer-Tuned V-Graft           | fresh                                 | layer-tuned write-time V  | compact summary + tail |
| `E:cfg=posslots` | Slot-Masked V-Graft           | fresh                                 | slot-masked write-time V  | compact summary + tail |
| `B-min-pack`     | Fresh-KV Control              | fresh                                 | fresh                     | packed summary         |
| `H-pack`         | KV-Graft                      | write-time K, re-rotated              | write-time V              | packed summary         |
| `H-gap`          | Gapped KV-Graft               | write-time K                          | write-time V              | gapped summary         |
| `H-pack-wrongS`  | Wrong-Source KV-Graft Control | wrong-source write-time K, re-rotated | wrong-source write-time V | packed summary         |

## Why `Pack` Should Not Be the Headline Name

`Pack` describes only the fact that summary tokens are placed into a compact
contiguous cache layout. That layout detail matters methodologically, but it is
shared by both sides of the important contrast:

- `B-min-pack` uses packed layout with freshly encoded K/V.
- `H-pack` uses the same packed layout with write-time K/V.

So `H-pack` should not be explained as "the packed one". The clearer explanation
is:

> **KV-Graft** preserves both keys and values from the summary's write-time
> state. Because those keys were written at their original positions, they are
> re-rotated when moved into the packed compact layout.

The matched control is:

> **Fresh-KV Control** uses the same summary text and the same packed layout,
> but recomputes both keys and values from the compact context.

## V-Graft vs KV-Graft

The main distinction should be stated directly:

```text
V-Graft:  fresh keys + write-time/blended values
KV-Graft: write-time keys + write-time values
```

More concretely:

- **V-Graft** builds the compacted context normally, keeps its freshly computed
  keys, and blends or replaces only the values at aligned token positions.
- **KV-Graft** keeps the summary token K/V states from write time. Its keys are
  position-corrected by re-rotation when the summary is moved into a packed
  layout.
- **Fresh-KV Control** uses the same packed summary layout as KV-Graft, but
  obtains both K and V by ordinary fresh re-encoding.

This taxonomy avoids the earlier confusion where `H-pack` sounded like a layout
variant of `V-Graft`. It is not. It is the K+V preservation branch of the
family.

## Suggested Prose

Use something close to this:

> We use **ValueGraft** for a family of training-free cache-state interventions
> across compaction boundaries. **V-Graft** preserves write-time value state
> while leaving freshly computed keys in place. **KV-Graft** preserves both keys
> and values from the summary's write-time state; when evaluated in a compact
> layout, its keys are re-rotated to the packed positions. The **Fresh-KV
> Control** uses the same summary text and packed layout as KV-Graft, but
> recomputes both keys and values from the compact context.

## Cautions

- Do not rename code paths, existing result files, or historical logs while the
  experiment is still active.
- Do not let the naming imply that every variant succeeded. In particular,
  Slot-Masked V-Graft failed its wrong-conversation contamination guard and
  should be described as an exploratory variant/control, not as a headline
  positive result.
- Do not describe KV-Graft as "just V-Graft with alpha = 1". V-Graft alpha
  controls values only. KV-Graft preserves both keys and values.
