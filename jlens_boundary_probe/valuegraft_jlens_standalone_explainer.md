# ValueGraft, Seen Through The J-Lens

Status: standalone explanatory draft
Date: 2026-07-07
Audience: readers who know the basics of transformer attention, but not the
details of this repository or mechanistic interpretability

## Short Version

When a long LLM conversation is compacted, a system usually replaces old turns
with a text summary and then re-encodes `summary + recent tail` from scratch.
That preserves some visible words, but it discards the cached attention state
that was written when those words were interpreted in the original context.

**ValueGraft** is a training-free intervention that asks whether we can reduce
that loss by carrying selected old-context key/value state across the
compaction boundary. The main evaluated shape so far is **V-Graft**: build the
ordinary compacted transcript with fresh keys, then blend write-time value
tensors into exact-aligned summary and tail tokens. Other variants can also
blend keys, with key re-rotation when positions change.

The J-lens work in this directory is not a separate thesis. It is an
interpretability spotlight on the same KV-cache hypothesis. It asks whether the
same summary tokens have different readable internal neighborhoods when they
are processed after the old conversation versus freshly re-encoded after
compaction. The `B-410` example is the compact version of the story: the same
printed permit code reads as stale/obsolete in the old-context path and as a
generic civic code in the fresh path.

## The Compaction Problem

A compacted conversation can keep the right words while changing the
context-conditioned state around those words.

Here, "state" means the hidden activations and cached attention context induced
while the summary is processed. It does not mean an explicit belief, memory, or
database entry.

The key question is not whether summarization loses information. Of course it
does. The research question is narrower and more useful:

```text
Given the same visible compacted transcript, can preserving selected
write-time attention state make the compacted conversation behave more like
the original full-context conversation?
```

In ordinary API-style conversation compaction, a long transcript is replaced by
a shorter summary plus a recent tail. The summary text is then re-encoded in a
new context. That is cheap and similar to how API context compaction is usually
implemented, but it discards the key/value state that existed while the summary
was written under the full conversation.

That matters because many compact summaries contain short local labels:
permit codes, branch names, file paths, nicknames, patch numbers, and private
aliases. The words may survive, while the local status attached to those words
becomes weaker: stale vs current, rejected vs live, room vs street, person vs
game mechanic.

## The ValueGraft Intervention

In standard transformer attention, each token position contributes keys and
values. Later tokens use queries to match against keys, then mix the
corresponding values. Keys are the addressable side of memory; values are the
content returned when attention lands there.

ValueGraft names a family of interventions that reuse or blend cached
key/value state across a compaction boundary:

```text
fresh compacted transcript
+ selected old-context key/value state for the summary and retained tail
```

The experiment has two traces for the same summary text:

1. **Write-time trace.** The model generates or processes the summary while the
   old conversation is still present. We save the key/value tensors written for
   the summary tokens under that full context.
2. **Fresh compacted trace.** The system builds the normal compacted transcript,
   typically `summary + recent tail`, and pre-fills it from scratch. This gives
   the ordinary fresh keys and values that a production-style compacted request
   would use.

ValueGraft then aligns identical summary tokens, and any explicitly chosen
identical tail spans, between those traces. At aligned compacted positions, it
can replace or blend the freshly computed cache entries with the entries written
under the old context.

The general parameterization is:

```text
K_final = (1 - alpha_K) * K_fresh
        + alpha_K       * K_write_time_rerotated

V_final = (1 - alpha_V) * V_fresh
        + alpha_V       * V_write_time
```

Keys need the extra "rerotated" qualifier because transformer keys are tied to
position through RoPE-style rotation. If a write-time key is moved to a new
compacted position, it has to be rotated into that position. Values do not have
the same positional rotation.

The important variants are:

| Name | Key policy | Value policy | Meaning |
| --- | --- | --- | --- |
| Plain Summary Compaction | `alpha_K = 0` | `alpha_V = 0` | The normal text-only baseline: fresh K and fresh V. |
| V-Graft | `alpha_K = 0` | `alpha_V` varied or tuned | Keep fresh attention addresses, blend old-context value content. |
| K-only Graft | `alpha_K` varied or tuned | `alpha_V = 0` | Test whether old-context addressing helps by itself. |
| KV-Graft | `alpha_K` varied or tuned | `alpha_V` varied or tuned | Blend or substitute both sides. |
| Coupled KV-Graft | `alpha_K = alpha_V = alpha` | same shared alpha | Move both sides together under one strength parameter. |

An alpha of `0` means "use the fresh compacted-context state." An alpha of `1`
means "use the write-time state exactly" for that channel. Intermediate values
blend; values above `1` are extrapolations. A full replacement setting is not a
separate algorithm. It is one point in this parameter space.

The current in-flight coding/continuation experiments are best understood as
V-Graft experiments: `alpha_K = 0`, with `alpha_V` constant or tuned. That
choice keeps the compacted transcript's fresh key geometry intact while testing
whether old-context value content improves continuity. Future controlled
branches can isolate K-only and full KV policies under the same summary, tail,
positions, prompt, and decoding setup.

## What We Measure

The behavioral experiments compare arms built from the same source
conversation:

- **Full Context:** the original long conversation, used as the ceiling.
- **Plain Summary Compaction:** the production-style baseline, freshly encoding
  `summary + tail`.
- **ValueGraft variants:** the same visible compacted transcript, but with
  selected write-time key/value state grafted into aligned summary or tail
  positions.
- **Negative controls:** wrong-conversation or shuffled grafts, used to detect
  improvements that come from generic smoothing rather than aligned content.

The central measurement is not "can the model remember everything that was
evicted?" It generally cannot, and ValueGraft is not a magic recall channel for
facts absent from the visible summary. The stronger target is mitigation:

- less fabrication when the compacted context no longer contains enough
  evidence;
- better continuation likelihood on the true next assistant text;
- better next-action likelihood on coding-agent traces;
- improved behavior specifically when the continuation depends on compact
  labels whose meanings were established before compaction.

When possible, results are interpreted as gap closure between full context and
plain summary compaction: how much of the loss from compaction does a grafted
arm recover? This framing matters because it asks whether ValueGraft reduces
the impact of compaction, not whether compaction damage exists.

## What The J-Lens Measures

The Jacobian lens, or J-lens, reads a residual-stream activation by transporting
it into the final-layer basis with an averaged Jacobian map, then decoding with
the model's vocabulary unembedding. It produces a ranked list of vocabulary
tokens associated with what that internal state is disposed to verbalize.

That gives us a way to ask:

```text
When the same summary token is evaluated in old-context and fresh paths,
do the residual-stream readouts point to different concepts?
```

This is one step downstream of the KV-cache hypothesis. The J-lens does not
open the cache and label individual key or value vectors. Instead, it reads the
residual-stream representation that results after the model has processed a
token in a particular context. If old-context cache state helps bind `B-410` to
"stale permit" during summary writing, and fresh re-encoding binds it more
weakly or differently, the J-lens can make that contrast visible in a way a
human can inspect.

Several limits matter:

- The J-lens reads residual-stream state rather than the KV cache directly.
- A token list is only a readout; downstream performance still has to be tested.
- The method works best for single-token or short verbalizable concepts.
- Late layers can look like ordinary next-token prediction.
- Paths, commands, identifiers, and Markdown syntax need span-level grouping.

For the qualitative examples below, layer 48 is the default display layer. In
the full sweep over 1,175 summary tokens and 63 fitted layers, layer 48 gave a
strong average old/fresh separation while staying relatively distinct from the
ordinary next-token list. Layer 62 can be more immediately readable, but it is
also more continuation-like.

The current sweep used fixed summary texts in matched wrappers. That keeps the
token sequence stable for comparison. It is a model of the compaction boundary,
not a claim that each displayed summary was freshly sampled during the sweep.

## Next-Token Control: What The Lens Adds

The readout becomes interesting only when it differs from ordinary continuation
pressure. The B-410 row is strong because the top next-token candidates are
mainly about completing an identifier, while the layer-48 J-lens candidates
name the identifier's old-context status.

Other rows are weaker and should be described that way. In the `Maple` example
below, the local text says `Maple means...`, so next-token prediction already
sees a definition cue. That row still shows old/fresh state contrast, with a
less clean next-token control.

The artifacts used here store ranked next-token scores, not calibrated
probabilities. This document therefore reports them as ranked candidates rather
than as probabilities. Every core token-level example below includes those
candidates beside the J-lens candidates. Otherwise it is too easy to mistake a
plain continuation effect for an internal-state readout.

## How To Read The Example Tables

The tables normalize away leading whitespace in displayed tokens and omit a few
non-English or formatting-heavy tokens when they do not help the reader. The
raw artifacts keep the full token lists.

In the tables, `old-context` means the write-time trace: the summary token is
processed while the old conversation is still available. `fresh` means the
same visible summary token is re-encoded in the compacted transcript.

| Field | Meaning |
| --- | --- |
| Visible snippet | The literal summary text around the inspected token or span. |
| Anchor | The token or short span being inspected. |
| Actual next | The next token in the fixed summary, when the artifact records or directly exposes it. |
| Next-token candidates | Ordinary top continuations after the anchor, ranked by the model's raw next-token scores. |
| J-lens readout | Vocabulary tokens from the residual-stream readout. |
| Interpretation | What the contrast suggests, including any example-specific caveat. |

## Example 1: `B-410`, A Stale Permit Code

Visible snippet:

```text
... lane. Red means cancel.
- Permit: B-410 is stale. Use P-771.
```

Anchor: `B` in `B-410`
Layer: 48
Actual next token: `-`

| Path | Next-token candidates | J-lens readout |
| --- | --- | --- |
| old-context | `-`, `4`, `-st`, punctuation | `obsolete`, `outdated`, `deprecated`, `expired` |
| fresh | `-`, `2`, `PD`, `1`, `3`, `4` | `municipal`, `City`, `Civic`, `License`, `permit` |

The next-token lists are doing local syntax. Both paths mostly know that the
next character should continue a code. The J-lens readout is doing something
more diagnostic: the old-context path attaches the stale/obsolete status to the
identifier itself, while the fresh path falls back toward civic-code semantics.

This matters because later questions often target the identifier, not the
adjective. A user may ask "which permit goes on the form?" or "is B-410 still
usable?" ValueGraft is aimed at preserving the state that binds `B-410` to
"stale, do not use."

## Example 2: `Maple`, A Local Alias

Visible snippet:

```text
Riverside block party state:
- Maple means the library's Maple Room for storage and volunteer check-in.
```

Anchor: first `Maple`
Layer: 48
Actual next token: `means`

| Path | Next-token candidates | J-lens readout |
| --- | --- | --- |
| old-context | `=`, `is`, `refers`, `means`, `Room` | `refers`, `=`, `referring`, `denotes` |
| fresh | `Street`, `St`, `Ave`, `street`, `Drive` | `Street`, `street`, `neighborhood`, `park`, `City` |

The old-context path treats `Maple` as a local defined referent. The fresh path
leans toward ordinary named-place priors: streets, parks, neighborhoods, city
names.

Because the text says `Maple means`, the ordinary next-token list already has
a definition cue. The row is best read as a state-contrast example: the
old-context readout is about local reference; the fresh readout is about the
model's default associations for `Maple` outside this conversation.

## Example 3: `Dex`, A Trade Partner Rather Than A Pokedex

Visible snippet:

```text
- Dex trade: spare Makuhita for Dex's Castform.
```

Anchor: `Dex`
Layer: 48
Actual next token: `trade`

| Path | Next-token candidates | J-lens readout |
| --- | --- | --- |
| old-context | `owes`, `trade`, `owed`, local trade words | `promised`, `partnered`, `promise`, `exchange` |
| fresh | `Nav`, `entry`, `completion`, tracker words | `completion`, `tracker`, `stats`, `bonus`, `Collector` |

The old-context path treats `Dex` as a person involved in an exchange. The
fresh path drifts toward Pokemon-interface meanings: Pokedex progress, DexNav,
completion tracking.

This row has a clear caveat. The visible snippet itself contains `trade`, so
the next-token list already carries some of the intended meaning. The J-lens
contrast is still helpful because it separates "Dex as trade partner" from
"Dex as game progress system," but it should be a secondary example.

## Example 4: Ordinary Planning Labels

The household checklist examples are useful because they are mundane. They are
about local labels and changed instructions, not puzzles.

Visible snippets:

```text
- Delta = ferry route, not airline; booked Delta 6 at 7:40.
- Orange = lockbox tag color; orange key opens kayak shed.
- Big cooler instruction is stale; bring two soft coolers.
```

Selected token rows from the full-layer sweep:

**`Delta`**
Layer: 47
Actual next token: `=`

| Path | Next-token candidates | J-lens readout |
| --- | --- | --- |
| old-context | `=`, `means`, `refers`, `ferry` | `refers`, `=`, `represents`, `means` |
| fresh | `=`, `Lake`, `Air`, `Airlines`, `Hotel` | `Delta`, `airport`, `airline`, `Sky`, `River` |

Reading: local route label vs airline/name priors.

**`Orange`**
Layer: 50
Actual next token: `=`

| Path | Next-token candidates | J-lens readout |
| --- | --- | --- |
| old-context | `=`, `lock`, `tag`, `refers`, `key` | `refers`, `signifies`, `represents`, `symbol`, `denotes` |
| fresh | `=`, `juice`, `County`, `Juice`, `Grove` | `Orange`, `orange`, `citrus`, `color`, `Juice` |

Reading: local tag role vs ordinary color/fruit priors.

**`cooler`**
Layer: 36
Actual next token: `instruction`

| Path | Next-token candidates | J-lens readout |
| --- | --- | --- |
| old-context | `is`, `rejected`, `=`, `replaced`, `removed` | `canceled`, `replaced`, `rejected`, `failed`, `refused` |
| fresh | `=`, `is`, `must`, `in` | `freezer`, `fridge`, `camping`, `cooler` |

Reading: stale instruction vs generic object semantics.

These are the kinds of meanings conversation summaries are full of: aliases,
exceptions, stale plans, and compact labels whose real meaning was established
earlier. A text-only summary can include the labels and glosses while still
producing a less situated readout around the labels themselves.

The `cooler` row is a useful caution. The ranked next-token candidates do not
cleanly predict the literal next token, `instruction`, in either path. The
J-lens contrast is still readable, but this is not as clean a control as
`B-410`.

## Example 5: `Falcon` And `Patch 17`, Coding-Adjacent Labels

Visible snippet:

```text
| Falcon | old rollback branch | rejected because it drops subscription coupons |
| Patch 17 | stale patch label | do not cite as live |
```

Selected token rows from the full-layer sweep:

**`Falcon`**
Layer: 43
Actual next token: `|`

| Path | Next-token candidates | J-lens readout |
| --- | --- | --- |
| old-context | `|`, `branch`, `old`, `rejected` | `rejected`, `obsolete`, `deprecated`, `failed`, `outdated` |
| fresh | `|`, `-`, `ry`, `9` | `Falcon`, `Flight`, `Aviation`, `eagle`, `Aerospace` |

Reading: rejected branch status vs bird/aviation priors.

**Separator after `Patch 17`**
Layer: 41
Actual next token: `stale`

| Path | Next-token candidates | J-lens readout |
| --- | --- | --- |
| old-context | `stale`, `old`, `previous`, `patch`, `prior` | `outdated`, `obsolete`, `old`, `expired`, `legacy` |
| fresh | `hot`, `fix`, `tax`, `invoice`, `coupon` | `patch`, `repair`, `fixes`, `revision`, `testing` |

Reading: stale label status vs generic patch/version semantics.

This is the bridge to coding agents. Real coding work is packed with compact
operational labels: branch names, issue IDs, file paths, commands, patch
numbers, failing tests, line ranges. The context-conditioned information is
often the status of those labels: current, rejected, failed, stale, already
tried, next to inspect.

The Markdown table also exposes a reporting hazard. Some high-divergence rows
are punctuation, spaces, table separators, or subword fragments. Public
examples should group tokens into spans such as `Patch 17` and its status
cell. The current row is reported precisely as a single token near that span;
raw token rows are too noisy for final human-facing coding examples.

## Short Sidebar: `Ghost2`

Visible snippet:

```text
Replacement Ralts Ghost2 is alive and made it to Victory Road.
```

Anchor: `2` in `Ghost2`
Layer: 44
Actual next token: `is`

| Path | Next-token candidates | J-lens readout |
| --- | --- | --- |
| old-context | `survived`, `is`, `made`, `survives`, `has` | `survived`, `successfully`, `surviving`, `survives`, `retained` |
| fresh | `is`, `was`, `joined`, `started`, `replaced` | `replacement`, `replaced`, `aka`, `renamed`, `second` |

The old-context path carries survival. The fresh path emphasizes replacement
and naming. This is intuitive and memorable, which makes it useful for
explaining the phenomenon.

The local phrase says `Ghost2 is alive`, so ordinary next-token prediction
already sees survival language. This is a vivid illustration of the same
pattern, though the control is less clean than B-410.

## Preliminary Coding Bridge: SWE-Style Action Spans

This section is not part of the core example table above. The artifacts here
are span-divergence probes over true next assistant actions, and they do not
yet include a matched next-token-candidate table for each span. They are
useful for deciding what to instrument next, but they should not be presented
as the same kind of controlled example as `B-410`, `Delta`, or `Falcon`.

We also sampled J-lens readouts on true next assistant actions from SWE-Gym
style trajectories. These are teacher-forced readouts of the known next action,
rather than agent benchmark scores.

In one getmoto trajectory, the true next action was:

```text
str_replace_editor view /workspace/getmoto__moto__4.1/moto/rds/responses.py [584, 600]
```

The span-aware probe found notable full-context vs compacted-context
divergence on:

| Span | Mean span divergence |
| --- | ---: |
| `str_replace_editor` | 0.651 |
| `/workspace/getmoto__moto__4.1/moto/rds/responses.py` | 0.561 |
| `responses.py` | 0.568 |
| `[584, 600]` | 0.407 |

Here, mean span divergence is a top-k readout-change score averaged over a
span. Higher means the old-context and compacted readout lists differ more; it
is a comparative readout score, not task accuracy.

In a Dask trajectory, the true next action was:

```text
execute_bash grep -n 'normalize_token' /workspace/dask__dask__2022.6/dask/base.py
```

The probe found divergence on:

| Span | Mean span divergence |
| --- | ---: |
| `execute_bash` | 0.603 |
| `grep -n 'normalize_token' /workspace/dask__dask__2022.6/dask/base.py` | 0.600 |
| `/workspace/dask__dask__2022.6/dask/base.py` | 0.624 |
| `base.py` | 0.538 |

This is promising mainly as a pointer to the right unit of analysis. The next
coding-lens pass should report the same ingredients used above, but grouped by
span: actual next text, ranked next-token candidates around the span, and
J-lens readouts for the old-context and fresh paths. Token-level examples are
often unreadable because paths and tool-call formats split into many pieces.

These rows do not show that ValueGraft improves agent coding. They show that
the readout method can localize differences on operational spans that matter
for agent actions.

## What This Does And Does Not Show

The current J-lens examples support a narrow qualitative claim:

```text
The same fixed summary tokens, evaluated under an old-context path and a
fresh compacted path, can have visibly different residual-stream readout
neighborhoods.
```

In the best examples, the old-context readout points toward the local status or
role of an identifier, while the fresh readout points toward the model's
default associations outside the conversation. That is compatible with the kind
of state ValueGraft is designed to preserve.

This is the intended relationship to the KV work: ValueGraft proposes a way to
carry selected old-context key/value state across compaction; the J-lens
examples make it easier to see why such state might matter by showing how the
same visible summary text can land in different readable internal
neighborhoods. The lens work demonstrates the potential target of preservation,
while the arm experiments decide whether preserving it actually improves model
behavior.

The examples do not settle the behavioral question. For that, the main
experiment still needs arm comparisons, guardrails, held-out tasks, and clear
failure accounting. The J-lens is most useful as a way to generate and explain
mechanistic hypotheses, then compare those hypotheses against behavioral data.

## Method And Artifact Trail

Subject model for this J-lens side investigation:

- `Qwen/Qwen3.6-27B`

Lens:

- Neuronpedia/Anthropic Jacobian-lens weights for Qwen3.6-27B

Primary local artifacts:

- `jlens_boundary_probe/outputs/qwen36_next_token_readout_comparison.json`
- `jlens_boundary_probe/outputs/qwen36_full_layer_sweep_summary.json`
- local ignored raw sweep:
  `jlens_boundary_probe/outputs/qwen36_full_layer_sweep.json`
- SWE-style artifacts:
  `jlens_boundary_probe/outputs/qwen36_swegym_next_action_probe_cleanprompt_t0001_t0004.json`
  and related variants

Validator:

```sh
python3 jlens_boundary_probe/validate_full_layer_sweep.py \
  jlens_boundary_probe/outputs/qwen36_full_layer_sweep.json \
  --summary jlens_boundary_probe/outputs/qwen36_full_layer_sweep_summary.json
```

Expected output for the current artifacts:

```text
VALID: demos=7 layers=63 summary_tokens=1175 grid_rows=74025
VALID: compact summary matches raw totals and schema
```

The broad sweep covered seven constructed demos, 1,175 aligned summary tokens,
and 63 fitted layers, giving 74,025 token-layer rows. The examples in this
document are selected for readability from that sweep and from the next-token
control artifact.

## Related Work And Prior Art

This note sits at the intersection of two bodies of work: interpretability
readouts for internal states, and systems or methods that reuse, edit, or
compress cached attention state. The overlap is still young. The
interpretability work helps us look at what may be present near a compaction
boundary; the KV-cache work constrains what can be claimed as new.

### Interpretability Readouts

The logit lens is the simplest ancestor of the J-lens used here. It projects an
intermediate residual-stream vector through the model's final unembedding and
asks which vocabulary tokens are already linearly accessible at that layer.
This is useful because it gives an immediate vocabulary-shaped view into hidden
states, but it is crude: intermediate layers are not naturally in the final
layer's basis, and late-layer next-token pressure can dominate the readout.

The tuned lens improves on that idea by learning layer-specific translators
from intermediate residual states to the final prediction space. It is a better
tool for reading latent next-token predictions, but it is still primarily a
vocabulary-projection method. It does not directly read KV-cache entries, and
it does not by itself establish whether a readout is causally responsible for a
behavioral difference.

The Jacobian lens, introduced by Gurnee et al. (2026), transports a
residual-stream vector into the final-layer basis using an averaged
input-output Jacobian, then decodes through the model's unembedding. The
associated Anthropic paper frames these transported, verbalizable directions as
a functional "workspace" for concepts the model can report or use across
contexts. For our purposes, the important part is narrower: the J-lens provides
a principled vocabulary readout for residual-stream states at specific layers
and positions.

This project uses the public Neuronpedia/Anthropic J-lens weights for
Qwen3.6-27B. That choice matters: the examples in this document are not
model-general evidence, and they are not measurements from the main
ValueGraft behavioral model. They are qualitative readouts from a side probe
whose role is to make the state-preservation hypothesis easier to inspect.

The J-lens also inherits real limitations. It works best for concepts that can
be named by single vocabulary tokens or short token neighborhoods; paths,
commands, and multi-token relations need span-level grouping. Its false-positive
rate is not fully characterized, so lens-visible differences should be treated
as hypotheses or qualitative support until paired with behavioral validation.
That is why this document keeps next-token candidates beside the J-lens
readouts and avoids treating the readout as proof.

### KV Cache, Latent Context, And Compaction

The transformer architecture defines attention in terms of queries, keys, and
values: a later token's query scores earlier keys and mixes the corresponding
values. During inference, those per-position keys and values are cached so the
model does not recompute the whole prefix at every decode step. ValueGraft
operates on this cached attention state at a conversation-compaction boundary.

"Models Take Notes at Prefill" is the closest mechanistic prior art for our
claim that cached state can carry more than performance bookkeeping. Li (2026)
argues that prefill writes field-conditioned conclusions onto downstream cache
state, and demonstrates that cached blocks can be edited, moved, and composed
while closely matching full recompute in controlled settings. That strongly
constrains novelty: ValueGraft should not claim that KV caches containing useful
semantic computation is new. Our narrower question is whether write-time state
associated with a human-readable conversation summary can reduce the damage of
ordinary text-summary compaction.

"Fast KV Compaction via Attention Matching" is the closest latent-compaction
neighbor. Zweiger et al. (2026) construct shorter keys and values that preserve
attention behavior, with per-KV-head matching and efficient subproblems. This
is directly relevant to any future per-layer or per-head `alpha_K`/`alpha_V`
tuning, because it treats compaction at the attention-head level rather than as
a single global operation. The difference is that Attention Matching creates a
compact latent cache, while ValueGraft keeps a visible natural-language summary
and asks whether selected old-context state should be attached to it.

KV reuse systems such as KVLink and CacheBlend address a related serving
problem: avoiding full prefill when chunks recur across requests. KVLink
precomputes document caches independently, adjusts positions at inference, and
uses trainable link tokens to help independently encoded chunks interact.
CacheBlend reuses precomputed chunk caches even when they are not simple
prefixes, selectively recomputing a small subset of tokens to recover
cross-chunk conditioning. These systems weaken broad novelty claims about
cache reuse, RoPE/position adjustment, and blending. They mostly target RAG or
document-chunk reuse, not conversation-summary replacement after a long
dialogue has been condensed.

Learned latent-compression methods form another nearby family. Gist tokens
train models to compress prompts into reusable special tokens. AutoCompressors
train models to turn long contexts into compact summary vectors used as soft
prompts. Compressed Context Memory continually compresses accumulating
key/value context for online interaction. Cartridges train a small offline KV
cache for a corpus using self-study, then reuse that cache for many later
queries. These systems all show that models can be trained or adapted to carry
context through nonstandard latent forms. ValueGraft is different because it is
training-free and centered on the existing production pattern of replacing old
dialogue with a human-readable summary.

Hosted APIs now expose product surfaces in the same broad area. OpenAI's
Responses compaction returns a compacted window that includes an encrypted
opaque compaction item carrying prior state forward. Anthropic exposes
server-side compaction as a typed compaction block containing a summary. Gemini
has context-caching and encrypted thought-signature mechanisms that preserve
reasoning continuity across calls. These public interfaces do not show that any
provider is doing ValueGraft internally. They do show that opaque or
semi-opaque state-carrying artifacts are a natural extension of current API
design, rather than an exotic deployment shape.

### What Remains Distinct

Taken together, the prior art says we should be careful. We should not claim
that internal states contain meaning, that KV caches are editable, that cache
blocks can be reused, or that latent context compression is new. The more
specific ValueGraft question is:

```text
When conversation history is replaced by a visible summary, does preserving
some old-context key/value state for that summary make the compacted
conversation behave more like the original long-context conversation?
```

The J-lens work in this directory is a qualitative companion to that behavioral
question. It gives concrete examples of what context-conditioned information
may be present around summary tokens before and after fresh re-encoding.

### References

- Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
  [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
- nostalgebraist. 2020.
  [Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).
  LessWrong.
- Belrose, Nora, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith,
  Danny Halawi, Stella Biderman, and Jacob Steinhardt. 2023.
  [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112).
  arXiv:2303.08112.
- Gurnee, Wes, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski,
  Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang,
  Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen,
  Joshua Batson, and Jack Lindsey. 2026.
  [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/).
  Transformer Circuits Thread.
- Anthropic. 2026.
  [jacobian-lens reference implementation](https://github.com/anthropics/jacobian-lens).
- Neuronpedia. 2026.
  [Jacobian Lens - Qwen3.6-27B](https://www.neuronpedia.org/qwen3.6-27b/jlens).
- Li, Bojie. 2026.
  [Models Take Notes at Prefill: KV Cache Can Be Editable and Composable](https://arxiv.org/abs/2606.17107).
  arXiv:2606.17107.
- Zweiger, Adam, Xinghong Fu, Han Guo, and Yoon Kim. 2026.
  [Fast KV Compaction via Attention Matching](https://arxiv.org/abs/2602.16284).
  arXiv:2602.16284.
- Yang, Jingbo, Bairu Hou, Wei Wei, Yujia Bao, and Shiyu Chang. 2025.
  [KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse](https://arxiv.org/abs/2502.16002).
  arXiv:2502.16002.
- Yao, Jiayi, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang,
  Kuntai Du, Shan Lu, and Junchen Jiang. 2025.
  [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444).
  EuroSys 2025; arXiv:2405.16444.
- Mu, Jesse, Xiang Lisa Li, and Noah Goodman. 2023.
  [Learning to Compress Prompts with Gist Tokens](https://arxiv.org/abs/2304.08467).
  NeurIPS 2023.
- Chevalier, Alexis, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023.
  [Adapting Language Models to Compress Contexts](https://aclanthology.org/2023.emnlp-main.232/).
  EMNLP 2023.
- Kim, Jang-Hyun, Junyoung Yeom, Sangdoo Yun, and Hyun Oh Song. 2023.
  [Compressed Context Memory For Online Language Model Interaction](https://arxiv.org/abs/2312.03414).
  ICLR 2024.
- Eyuboglu, Sabri, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley,
  Emily Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, and
  Christopher Re. 2025.
  [Cartridges: Lightweight and general-purpose long context representations via self-study](https://arxiv.org/abs/2506.06266).
  arXiv:2506.06266.
- OpenAI. 2026.
  [Compaction](https://developers.openai.com/api/docs/guides/compaction).
  OpenAI API documentation.
- Anthropic. 2026.
  [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction).
  Claude Platform documentation.
- Google AI for Developers. 2026.
  [Thought signatures](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures)
  and [context caching](https://ai.google.dev/gemini-api/docs/caching).
- Qwen Team. 2026.
  [Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model](https://qwen.ai/blog?id=qwen3.6-27b).
