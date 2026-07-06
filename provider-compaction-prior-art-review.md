# Hosted compaction surfaces and prior-art implications

Date: 2026-07-05

This note is a focused prior-art and product-surface review for the
conversation-compaction work in this repository. It is written after discovering
that hosted frontier APIs now expose compaction and opaque-state primitives that
are much closer to our proposed production shape than the earlier project notes
assumed.

The most important framing correction is this: the project should not be
presented as trying to prove that re-encoding a summary loses useful
context-conditioned state. That is the premise and the baseline damage we are
measuring. The research question is whether a practical state-preserving
intervention can reduce the damage of ordinary text-only compaction, especially
in agentic or coding settings where compaction happens mid-task.

## Bottom line

OpenAI has the closest documented public interface to the production shape we
had been imagining: Responses supports automatic compaction through
`context_management`, a standalone `/responses/compact` endpoint, and an opaque
encrypted `compaction` item that is passed forward in later requests. OpenAI
also separately documents that prompt caching stores attention key/value tensors
derived during prefill. The docs do not say that the compaction item contains
raw KV tensors, cached value vectors, ValueGraft-style blends, or anything
mechanistically equivalent to our experiments. But the outside shape is close
enough that we must cite it and avoid claiming novelty for the API pattern.

Anthropic also has first-class server-side compaction, but its public compaction
surface looks more like a typed summary block. Anthropic separately exposes
opaque thinking signatures and documents KV-backed prompt caching. That makes
the platform direction similar, while leaving much less public evidence that
Anthropic compaction itself carries latent state beyond the visible summary.

Gemini has several adjacent primitives: implicit context caching, encrypted
thought signatures, Managed Agents automatic context compaction, Live API
context-window compression, and session-resumption handles. I did not find a
general text API endpoint analogous to OpenAI `/responses/compact`, but Gemini's
interfaces clearly support server-managed state and opaque reasoning continuity.

Open source currently shows abstraction and round-tripping around provider
compaction items, not a clear public implementation of summary-boundary
KV/value-state preservation. The most relevant open-source item I found was
`lmctx`, which preserves OpenAI/Anthropic compaction artifacts as typed parts and
provider-raw blobs so they can be passed back correctly.

The literature overlap is stronger than our earlier notes suggested. In
particular, "Models Take Notes at Prefill" establishes editable and composable
KV caches as a general mechanism, "Fast KV Compaction via Attention Matching"
does true latent-space KV compaction, and "Parallel Context Compaction" studies
production-style agent summarization/compaction. These do not appear to collapse
our exact experiment, but they strongly constrain the novelty claim.

The safest contribution claim is something like:

> Hosted APIs already expose compaction/state objects, and prior work already
> shows that KV cache state is editable, composable, and compressible. This
> project isolates a specific compaction-boundary question: when a conversation
> is reduced to a visible summary plus retained tail, can preserving or
> reconstructing write-time cached attention value state associated with the
> compacted representation reduce the behavioral damage of text-only
> compaction?

## What counts as fact vs inference

I use three labels below:

- **Documented:** directly stated in provider docs, SDK type definitions, a
  paper abstract/body, or source code.
- **Inference:** a reasonable interpretation of the documented interface, but
  not guaranteed by the provider or paper.
- **Unknown:** something we should not claim without further evidence.

This distinction matters because hosted providers intentionally do not expose
the internal representation of their compaction artifacts. The public API can
look like our proposed shape without using our mechanism.

## Hosted provider surfaces

### OpenAI

**Documented interface.**

OpenAI Responses now has explicit compaction support:

- `context_management` can enable automatic compaction with a
  `compact_threshold`.
- `/responses/compact` can be called directly to compact an input window.
- The compacted window contains a typed `compaction` item with encrypted
  content.
- The compaction item is passed forward as part of later stateless requests, or
  carried implicitly when using server-side response chaining.
- The docs describe the compaction item as carrying prior state and reasoning
  forward with fewer tokens.
- The SDK has generated types such as `ResponseCompactionItem`,
  `ResponseCompactionItemParam`, `CompactedResponse`, and a
  `compaction_trigger` input item.

OpenAI also has adjacent opaque-state machinery:

- Reasoning models can return encrypted reasoning content for stateless
  continuation.
- The conversation-state guide describes carrying prior output items forward,
  including opaque reasoning items.
- Prompt caching has a `prompt_cache_key` for routing/cache affinity.
- The prompt-caching guide explicitly describes cached attention key/value
  tensors produced during prefill, including extended retention of those tensors
  in GPU-local storage.

**Inference.**

OpenAI's public surface is extremely close to the "summary plus opaque
compaction handle" shape we had sketched. The existence of encrypted reasoning
items and encrypted compaction items shows that OpenAI is comfortable with
client-carried opaque state in otherwise stateless API flows. The prompt caching
docs show that OpenAI production systems already retain KV tensors internally.

This makes our production shape realistic. It is not some awkward bolt-on; it is
aligned with how the platform is already moving.

**Unknown.**

The public docs do not reveal what is inside the encrypted compaction item. It
could contain a text summary, a reasoning-state artifact, a model-private latent
representation, references to server-side state, a compressed serialized form of
multiple artifacts, or something else. The SDK phrase "encrypted content of the
compaction summary" is compatible with a summary-only object; the guide's
language about carrying prior state and reasoning is broader. Neither is enough
to claim raw KV preservation or value-vector blending.

So the correct public claim is not "OpenAI is doing ValueGraft." The correct
claim is "OpenAI exposes a typed encrypted compaction-state object that solves a
similar product-interface problem, and OpenAI also documents KV-backed prompt
caching elsewhere."

### Anthropic

**Documented interface.**

Anthropic's Messages API has beta server-side context compaction:

- Requests opt in with a beta header and `context_management` edits of type
  `compact_20260112`.
- The compaction operation can trigger around a token threshold.
- It creates a `compaction` block containing a summary.
- Later requests include that block, and the API can drop earlier content before
  the compaction block.
- The beta type supports optional custom summarization instructions and a
  `pause_after_compaction` mode.

Anthropic also documents adjacent state machinery:

- Context editing is a server-side operation that changes the prompt before the
  model sees it, while the client can keep the full transcript.
- Extended thinking uses opaque signatures. In omitted-thinking mode, the
  visible thinking can be empty while the signature carries encrypted thinking
  state needed for continuation.
- Prompt caching stores KV representations and cryptographic hashes in memory,
  scoped to workspace/organization.

**Inference.**

Anthropic has a mature product model for compaction and for opaque signed or
encrypted continuation state. The public compaction block, however, appears more
text-summary-shaped than OpenAI's encrypted compaction item. Anthropic's
thinking signatures prove the platform has opaque state pass-through where it is
needed, but they are not themselves compaction artifacts.

**Unknown.**

I did not find public evidence that Anthropic's compaction block contains hidden
KV-like latent state beyond the summary content. It may still do internal work
that is not visible in the transcript, especially when server-side state is used,
but the public docs do not establish that.

### Google / Gemini

**Documented interface.**

Gemini has several neighboring primitives:

- Implicit context caching for repeated input prefixes, with cached-token usage
  metadata.
- Thought signatures: encrypted representations of internal reasoning state that
  must be preserved across turns in stateless flows.
- Stateful interaction modes where the server manages conversation state.
- Managed Agents automatic context compaction for long-running agent sessions.
- Live API context-window compression, sliding-window behavior, and
  session-resumption handles.

**Inference.**

Gemini is clearly converging on provider-managed long-context state, opaque
reasoning continuity, and automatic compaction in agentic settings. Thought
signatures are particularly relevant because they are an explicit encrypted
state artifact that travels with the transcript.

**Unknown.**

I did not find a general Gemini text endpoint equivalent to OpenAI
`/responses/compact`, where a client asks for a compacted replacement transcript
and receives a typed opaque compaction object. Managed Agents and Live API
compaction appear more automatic and product-specific. I also did not find
documentation connecting Gemini compaction to KV tensors.

## Provider-surface comparison

| Provider | Public compaction surface | Opaque state in transcript? | Public KV link? | What we can safely say |
| --- | --- | --- | --- | --- |
| OpenAI | Responses automatic compaction and `/responses/compact` | Yes, encrypted `compaction` item | Prompt caching docs explicitly discuss KV tensors, but not compaction internals | Very close to our proposed production interface; mechanism unknown |
| Anthropic | Beta `compact_20260112` context edit producing compaction block | Thinking signatures are opaque; compaction block itself appears summary-shaped | Prompt caching docs discuss KV representations | Product compaction exists; opaque state exists elsewhere; hidden compaction state not documented |
| Gemini | Managed Agents automatic compaction; Live API compression; implicit caching | Thought signatures and session handles | Context caching implied, but I did not find public KV details for compaction | Strong adjacent state-management primitives; no general compact endpoint found |

## Implications for our production-shape idea

The "summary plus opaque handle" idea is viable, but it is not purely
speculative anymore. OpenAI has essentially validated the external pattern:
a compaction operation can produce an opaque item that travels with a compacted
conversation and lets a later request continue with less visible text.

That changes the tone of our writeup. We should not pitch the interface itself
as a novel product invention. We should pitch our work as investigating one
mechanistic route that could make such compaction artifacts better: preserving
or reconstructing write-time cached attention value state at the point where the
full context is still available.

The user-facing API should probably be described as a typed state item rather
than a literal "token" in the text stream. The existing platforms use exactly
that pattern: typed output/input items, typed content blocks, signatures, or
handles. A string token inserted after the summary is a useful mental model, but
the production object would more likely be a signed/encrypted structured item
with model, tokenizer, prompt-template, tool-schema, and provenance metadata.

The hash idea remains natural:

- The compaction artifact can be bound to the canonical pre-compaction context,
  the summary instructions, the retained-tail policy, the model/version, and the
  tool/schema environment.
- A digest can help validate that a handle is being used with the intended
  summary/tail context.
- A provider-internal hash can help route identical compactions to the same
  cache storage and improve cache friendliness.
- The client-visible digest should probably be tenant-scoped or HMACed to avoid
  cross-tenant equality leaks.

The important distinction is that a hash verifies provenance; it does not carry
state. The state carrier is still the encrypted compaction item, server-side
object reference, or other opaque artifact.

## What this means for novelty

The old novelty claim was too broad if stated as "KV caches have semantic state"
or "APIs could expose compaction handles." Both now have strong prior art or
product evidence.

The more defensible novelty is narrower and more useful:

1. Use text-only conversation compaction as the production baseline.
2. Treat the compaction damage as already expected, not as the discovery.
3. Compare that baseline to training-free state-preserving variants that keep or
   blend cached attention value state around the compacted summary/tail.
4. Measure whether the intervention reduces compaction-induced behavioral harm
   on tasks that resemble real agent continuity.

This is not the same as latent KV compression in general, exact prefix caching,
RAG chunk cache reuse, or editable user memory. It is specifically a
summary-boundary experiment.

## Open-source findings

I searched GitHub for provider compaction terms including `responses.compact`,
`compact_20260112`, `compact_threshold`, `compaction_trigger`, and related
phrases.

The main relevant non-provider project I found was `lmctx`:

- It describes itself as a context kernel for LLM APIs.
- It models text, files, tools, thinking, and compaction as typed context parts.
- It includes OpenAI Responses compact adapters and Anthropic compact examples.
- For OpenAI, it round-trips the `compaction` item by preserving provider raw
  data and encrypted content as a blob.
- For Anthropic, it supports the beta compaction edit flow.

This is meaningful interface prior art. It shows that external tooling is
already adapting to provider compaction artifacts as first-class state. But I did
not see evidence that `lmctx` implements KV surgery, value-vector blending,
summary write-time value preservation, or an open-model equivalent of our arms.

The official OpenAI and Anthropic SDKs also matter here. They are not research
systems, but they define public typed surfaces for compaction and should be
cited as interface evidence.

This search is not exhaustive. Provider APIs are moving quickly, and code search
misses private repos, notebooks, renamed abstractions, and implementations that
do not use the provider terms. Still, I did not find an open-source project that
appears to directly do our specific summary-boundary cache-state experiment.

## Literature findings

### Models Take Notes at Prefill

"Models Take Notes at Prefill: KV Cache Can Be Editable and Composable" is the
strongest mechanistic neighbor.

The paper argues that during prefill, models write field-conditioned conclusions
onto downstream tokens in the KV cache. It then demonstrates cache editing and
composition: precompiled cache blocks can be moved, edited with errata, and
spliced into later contexts while closely matching full recompute behavior.

Implication for us:

- We should not claim that editable/composable KV state is novel.
- We should not claim that position-portable cache blocks are novel in general.
- We can use this paper as mechanistic support for why write-time value state
  might matter after compaction.
- Our differentiator is the conversation-summary boundary and the comparison
  against ordinary text-only summary compaction, not the mere fact that KV state
  can be meaningful.

This paper also makes the practical API story more plausible: if cached state is
portable and editable under controlled conditions, then an opaque compaction
artifact is not a strange product object. It is a natural form of memoized
computation.

### Fast KV Compaction via Attention Matching

"Fast KV Compaction via Attention Matching" is direct latent-compaction prior
art. It constructs shorter compacted KV caches that preserve attention behavior,
with per-KV-head matching and efficient closed-form subproblems. It also
includes online compaction examples.

Implication for us:

- This paper occupies the "latent KV compaction" space much more directly than
  many older memory papers.
- It is especially relevant to any per-head alpha or nonuniform blending idea,
  because it treats attention behavior at the head level.
- It compares against token-space summarization as a lossy deployment baseline.
- It does not appear to focus on the same visible-summary-plus-tail
  conversation boundary or on preserving the write-time state of the summary
  representation generated under the full old context.

In a writeup, this should be cited as a close method neighbor and a reason to be
precise: our method is not "KV compaction" in the broadest sense. It is a
summary-conditioned state-preservation intervention.

### Parallel Context Compaction for Long-Horizon LLM Agent Serving

"Parallel Context Compaction for Long-Horizon LLM Agent Serving" is close on the
production problem. It studies long-running agents whose histories grow beyond
the context window. It treats LLM summarization as the standard compaction
mechanism, points out that it is lossy and blocking, and proposes parallel
summarization to improve latency, volume control, and predictability.

Implication for us:

- This is strong evidence that conversation compaction is an active serving
  problem, not just a toy harness concern.
- It reinforces that summary-only compaction is the right production baseline.
- It does not attempt to preserve cached attention state. Its summaries remain
  human-readable token-space artifacts.

This should be cited in motivation and related work, especially for the agentic
serving context.

### KVLink, CacheBlend, SamKV, and chunk-cache reuse

KVLink precomputes document KV caches independently and later concatenates or
reuses them, with positional adjustment and trainable link tokens to recover
performance. CacheBlend and related systems also target reuse of independently
encoded chunks or cached context for serving efficiency. The project notes also
mention SamKV as especially formula-adjacent because it uses a KV blending
operation.

Implication for us:

- These systems are adjacent whenever we discuss cache reuse, position
  correction, and blending.
- Their usual setting is RAG/document chunk reuse or latency reduction, not
  conversation compaction after a lossy summary.
- They weaken broad novelty claims about blending or reusing caches, but do not
  obviously answer the summary-boundary question.

### Learned latent compression and memory systems

Cartridges, Compressed Context Memory, AutoCompressor, ICAE, Activation Beacon,
gist-token methods, and related work all matter because they ask models to carry
context in nonstandard latent or compressed forms. MemGPT and framework-level
agent memories matter because they motivate compaction as an operational
problem.

Implication for us:

- H-pack/SelfGist-like arms should be framed as training-free natural-language
  cousins of learned compression/gist approaches, not as isolated inventions.
- ValueGraft-style arms should be framed as cache-state preservation at a
  production compaction boundary, not as a general replacement for learned
  latent memory.

## What we should change in future writeups

Do say:

- "Hosted APIs now expose compaction and opaque state artifacts."
- "OpenAI's Responses compaction is the closest public API analogue."
- "Anthropic and Gemini expose adjacent compaction, caching, and opaque
  reasoning-state mechanisms."
- "Prior work already shows that KV caches can be edited, composed, reused, and
  compacted."
- "Our experiment asks whether preserving write-time cached attention value
  state can reduce the damage of text-only conversation compaction."

Do not say:

- "No one has thought of opaque compaction handles."
- "The novelty is that KV caches contain semantic information."
- "OpenAI/Anthropic/Gemini are doing ValueGraft."
- "A small proxy effect proves practical coding-agent usefulness."
- "A small proxy effect is evidence against practical usefulness."

The current results should be described carefully: statistically significant
positive movement on fragile proxy tasks is promising, especially because those
tasks may understate real agent impact. But the proxy effect is not enough by
itself to claim a practical coding-agent win.

## Possible black-box checks against hosted APIs

These are not required for the current experiment, but they would help position
the work:

1. Run the same long synthetic or coding-agent trace through OpenAI
   `/responses/compact`.
2. Inspect the returned compacted window: item types, retained visible text,
   token counts, and whether the opaque compaction item is present.
3. Continue from full context, manual summary, and provider compaction on probes
   that target evicted facts, decoys, and already-failed actions.
4. Treat provider compaction as a product baseline, not as evidence about its
   internals.
5. If possible, repeat an analogous check on Anthropic server-side compaction
   and Gemini Managed Agents/Live API compaction.

This would not reveal whether providers use KV-like state internally, but it
would tell us whether provider-native compaction behaves more like a text-only
summary or like a stronger opaque-state continuation in the cases we care about.

## Open questions

- What exactly is serialized inside OpenAI's encrypted compaction item?
- Is OpenAI compaction deterministic for identical input/context-management
  settings, or intentionally stochastic/private?
- How large are compaction items in practice, and how are they counted for
  billing/context accounting?
- Does the compaction item include only summary text, reasoning state, a latent
  state object, server-side references, or a mixture?
- Does Anthropic's compaction block carry any hidden state beyond the visible
  summary when used with server-side state?
- Can Gemini's Managed Agents compaction be inspected or compared in a
  controlled way?
- Can provider compaction objects be safely bound to source-context hashes
  without leaking cross-tenant equality information?
- How do prompt caching and compaction handles interact when the same compacted
  conversation recurs across branches?
- Would per-head or per-layer state preservation materially outperform the
  current global-alpha blend in our setting?

## Practical interpretation for this repo

The provider findings make the deployment story stronger but the novelty story
narrower.

They make deployment stronger because major providers already expose API shapes
that can carry opaque state through stateless-looking requests. The idea that a
compaction operation could return a state object bound to a summary is now
plainly realistic.

They narrow novelty because the product interface and the broad principle of
meaningful cached state are no longer defensible as new. The repo's distinctive
value has to be in the experiment:

- treating ordinary summary compaction as the baseline to improve;
- preserving cached attention value state at the compaction boundary;
- validating the cache surgery with identity tests;
- measuring whether the intervention reduces concrete behavioral damage;
- and extending that measurement toward real coding-agent traces.

That is still a real contribution if the effect survives better workloads. It is
just a more precise contribution than the one the early framing implied.

## Sources checked

Provider docs and SDKs:

- OpenAI compaction guide:
  <https://developers.openai.com/api/docs/guides/compaction.md>
- OpenAI conversation state:
  <https://developers.openai.com/api/docs/guides/conversation-state.md>
- OpenAI reasoning guide:
  <https://developers.openai.com/api/docs/guides/reasoning.md>
- OpenAI prompt caching:
  <https://developers.openai.com/api/docs/guides/prompt-caching.md>
- OpenAI Python SDK generated Responses resources and types:
  <https://github.com/openai/openai-python/tree/main/src/openai>
- Anthropic compaction:
  <https://platform.claude.com/docs/en/build-with-claude/compaction.md>
- Anthropic context editing:
  <https://platform.claude.com/docs/en/build-with-claude/context-editing.md>
- Anthropic prompt caching:
  <https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md>
- Anthropic extended thinking:
  <https://platform.claude.com/docs/en/build-with-claude/extended-thinking.md>
- Gemini caching:
  <https://ai.google.dev/gemini-api/docs/caching>
- Gemini thinking:
  <https://ai.google.dev/gemini-api/docs/thinking>
- Gemini Managed Agents quickstart:
  <https://ai.google.dev/gemini-api/docs/managed-agents-quickstart>
- Gemini Live API session management:
  <https://ai.google.dev/gemini-api/docs/live-api/session-management>

Open source:

- lmctx:
  <https://github.com/Yuki-Imajuku/lmctx>

Papers:

- Models Take Notes at Prefill: KV Cache Can Be Editable and Composable:
  <https://arxiv.org/abs/2606.17107>
- Fast KV Compaction via Attention Matching:
  <https://arxiv.org/abs/2602.16284>
- Parallel Context Compaction for Long-Horizon LLM Agent Serving:
  <https://arxiv.org/abs/2605.23296>
- KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse:
  <https://arxiv.org/abs/2502.16002>
