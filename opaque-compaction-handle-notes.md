# Opaque Compaction Handle Notes

These are deployment-framing notes, not a prescriptive design. The point is
to describe one way to think about the practical overhead of retaining
summary KV/state after compaction.

## The practical tension

SelfGist / H-pack-style compaction is not purely stateless. It preserves a
small piece of model-native state: the KV entries for the summary tokens as
they were written while the full conversation was still present.

That is much smaller than preserving the full old context cache, but it is
still large compared with plain text.

For the 30B Qwen3-A3B model, rough fp16 KV size is:

```text
48 layers * 2 (K,V) * 4 KV heads * 128 dim * 2 bytes
= 98,304 bytes/token
≈ 96 KiB/token
```

So a summary-state sidecar costs roughly:

```text
100 summary tokens  ≈ 9.4 MiB
300 summary tokens  ≈ 28 MiB
500 summary tokens  ≈ 47 MiB
700 summary tokens  ≈ 66 MiB
```

That is tiny compared with a full 12K-token KV cache, but huge compared with
the text summary itself. As a raw client-uploaded blob, this is probably not
an attractive stateless API payload.

## Opaque handle framing

A more plausible API/product framing is not:

```text
client sends summary text + tens of MB of KV with every request
```

but:

```text
provider stores the compacted state server-side
client receives and later presents an opaque handle
```

This resembles existing API patterns such as prompt-cache handles,
server-side session IDs, or opaque reasoning/thinking tokens: the client does
not inspect or transport the internal state directly, but can refer to it in
later calls.

One possible lifecycle:

```text
1. Conversation approaches compaction boundary.
2. Provider runs compaction while full history/cache is still available.
3. Provider generates a visible summary.
4. Provider also stores packed summary KV/state near the model.
5. Provider returns:
   - summary text, or a summary id
   - opaque compaction_handle
6. Later requests include:
   - compaction_handle
   - recent visible tail
   - new user turn
7. Provider reconstructs compacted context as:
   - retained packed summary state
   - freshly encoded recent tail
   - fresh new turn
```

The visible summary remains the auditable text channel. The handle represents
the model-native state associated with that summary.

## Harness/API primitive framing

Another way to phrase this is that compaction should be an explicit operation
the harness can call, not a hidden transport detail. The harness already
decides when to summarize or truncate context; an opaque-state version could
fit into that existing control point.

For example:

```text
compact_from(boundary, instructions) -> {
  summary_text,
  compaction_token
}
```

where:

- `boundary` is the message/token boundary before which history should be
  compacted.
- `instructions` are the summarization/compaction instructions.
- `summary_text` is the ordinary visible summary the harness would already
  insert.
- `compaction_token` is an opaque provider-side handle to the latent state
  produced while the full pre-compaction history/cache was still available.

Future requests could then remain mostly text-shaped:

```text
system/developer messages
summary_text
<compaction_token>
recent_tail_messages
new_user_turn
```

The token's semantics would be something like:

```text
Apply the provider-side latent state associated with this summary at this
compaction boundary.
```

It could be represented as a special cache handle attached to the summary
message, or as a synthetic opaque token placed immediately after the summary.
The exact surface is a product/API choice. The important point is that the
harness does not need to inspect tensors; it only needs to preserve and pass
the handle in the compacted context.

This also gives a natural fallback story. If a provider or model does not
support latent compaction handles, the harness simply keeps `summary_text` and
drops `compaction_token`, reducing to ordinary text-only compaction.

## How to describe the overhead

There are several different costs, and they should not be collapsed into one
number.

### Client bandwidth

With a raw KV sidecar, bandwidth is likely unacceptable for common stateless
HTTP usage.

With an opaque server-side handle, client bandwidth is small: the client sends
the handle plus ordinary text.

### Server storage

The provider pays storage for the retained summary state. For 30B fp16 KV,
this is roughly tens of MiB per compacted conversation checkpoint, depending
on summary length.

This is still far less than storing the entire long-context KV cache.

### Compute

At compaction time, the provider must generate the summary while full history
is present and retain/pack the summary state.

On later turns, the provider can recompute the recent tail normally and attach
the stored summary state. This may be cheaper than reprocessing the full old
history, but not as stateless as ordinary text-only compaction.

### Cache lifetime

The handle probably needs expiration, revocation, and billing semantics, much
like prompt caching. It may be useful only for a bounded session window.

## Important caveats

- The handle is model-specific, tokenizer-specific, and probably
  chat-template-specific.
- It may not survive model upgrades without regeneration.
- It has privacy/security implications because it encodes information derived
  from prior conversation state, not just visible text.
- The user-visible summary is no longer the entire persisted state, which
  complicates audit and debugging.
- APIs should probably disclose when latent compaction state is active.
- Compression or quantization of the retained state might be necessary, but
  would need separate validation.

## Possible writeup framing

This should be presented as a bounded deployment possibility, not a solved
interface design:

> Raw summary KV is too large to ship as a normal stateless request payload.
> But if an API already supports provider-side cached prefixes, session
> continuation, or opaque reasoning/cache tokens, then SelfGist-like
> compaction could be exposed as an opaque compaction handle: the summary
> remains visible text, while the provider stores a compact model-native state
> object for that summary.

The useful comparison is:

```text
text-only compaction:
  summary text + recent tail, all re-encoded fresh

opaque-state compaction:
  summary text + provider-side summary-state handle + recent tail
```

This is no longer purely stateless, but it is not full conversation state
either. It is a middle point: a compact, server-resident continuation artifact
created at compaction time.

For agent harnesses, the practical description might be:

> When context gets too long, the harness asks the provider to compact from a
> chosen boundary. The provider returns a human-readable summary plus an
> opaque compaction token. The harness inserts both where it would normally
> insert a summary, keeps the recent tail as text, and continues. The
> experiment's in-memory cache surgery is testing whether such a token would
> be worth having; serialization, expiration, and handle transport follow
> established cached-state API patterns.
