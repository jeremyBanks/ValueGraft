# Seeing What ValueGraft Is Trying To Preserve

Status: standalone explanatory draft
Date: 2026-07-07
Audience: readers who know the basics of transformer attention, but not the
details of this repository or mechanistic interpretability

## The Idea In One Example

A compacted conversation often keeps the right text while losing part of the
state that made the text mean something locally.

The cleanest example in the current J-lens sweep is a permit code. The summary
contains this line:

```text
- Permit: B-410 is stale. Use P-771.
```

We inspect the token `B` in `B-410` under two paths:

- **Old-context path:** the same summary token sequence is evaluated after the
  original conversation prefix.
- **Fresh path:** the same summary token sequence is re-encoded in a compacted
  wrapper without that old conversation prefix.

The ordinary next-token candidates mostly continue the code:

| Path | Top next-token candidates after `B` |
| --- | --- |
| old-context | `-`, `4`, `-st`, punctuation |
| fresh | `-`, `2`, `PD`, `1`, `3`, `4` |

At layer 48, the Jacobian-lens readout looks very different:

| Path | J-lens readout near `B` |
| --- | --- |
| old-context | `obsolete`, `outdated`, `deprecated`, `expired` |
| fresh | `municipal`, `City`, `Civic`, `License`, `permit` |

This is the whole motivation in miniature. The visible text is still there in
both paths. The old-context path makes the identifier itself read as a stale
permit. The fresh path makes it read more like a generic civic code. ValueGraft
asks whether compaction can carry some of that old-context state forward,
rather than asking a later model pass to reconstruct everything from text.

This example is qualitative telemetry, not a behavioral result. It is useful
because it makes the target of the intervention concrete.

## What ValueGraft Means Here

In standard transformer attention, each token position contributes keys and
values. Later tokens use queries to match against keys, then mix the
corresponding values. Keys are the addressable side of memory; values are the
content returned when attention lands there.

In ordinary API-style conversation compaction, a long transcript is replaced by
a shorter summary plus a recent tail. The summary text is then re-encoded in a
new context. That is cheap and product-shaped, but it discards the key/value
state that existed while the summary was written under the full conversation.

ValueGraft names a family of interventions at that boundary:

```text
fresh compacted transcript
+ selected old-context key/value state for the summary and retained tail
```

In the broader design, keys and values can have separate interpolation
parameters, `alpha_K` and `alpha_V`. An alpha of `0` leaves that channel fresh.
An alpha of `1` fully substitutes the old-context state for that channel.
Intermediate values blend; values above `1` are extrapolations. The current
J-lens note is not evaluating which alpha is best. It explains why there may
be useful state to preserve at all.

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

Several limits matter.

- The J-lens reads residual-stream state, not the KV cache directly.
- A token list is not behavior, a belief, or proof of downstream performance.
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

## Why This Is Not Just Next-Token Prediction

The readout becomes interesting only when it differs from ordinary continuation
pressure. The B-410 row is strong because the top next-token candidates are
mainly about completing an identifier, while the layer-48 J-lens candidates
name the identifier's old-context status.

Other rows are weaker and should be described that way. In the `Maple` example
below, the local text says `Maple means...`, so next-token prediction already
sees a definition cue. That row still shows old/fresh state contrast, but it is
not as clean a next-token control.

This distinction should stay visible in any public figure: show next-token
candidates beside J-lens candidates. Otherwise it is too easy to mistake a
plain continuation effect for an internal-state readout.

## How To Read The Example Tables

The tables normalize away leading whitespace in displayed tokens and omit a few
non-English or formatting-heavy tokens when they do not help the reader. The
raw artifacts keep the full token lists.

| Field | Meaning |
| --- | --- |
| Visible snippet | The literal summary text around the inspected token or span. |
| Anchor | The token or short span being inspected. |
| Next-token candidates | Ordinary top continuations after the anchor. |
| J-lens readout | Vocabulary tokens from the residual-stream readout. |
| Interpretation | What the contrast suggests, including caveats. |

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

What to notice:

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

What to notice:

The old-context path treats `Maple` as a local defined referent. The fresh path
leans toward ordinary named-place priors: streets, parks, neighborhoods, city
names.

This row is useful but not as clean as B-410. Because the text says `Maple
means`, the ordinary next-token list already has a definition cue. The row is
best read as a state-contrast example: the old-context readout is about local
reference; the fresh readout is about generic lexical priors.

## Example 3: `Dex`, A Trade Partner Rather Than A Pokedex

Visible snippet:

```text
- Dex trade: spare Makuhita for Dex's Castform.
```

Anchor: `Dex`
Layer: 48

| Path | Next-token candidates | J-lens readout |
| --- | --- | --- |
| old-context | `owes`, `trade`, `owed`, local trade words | `promised`, `partnered`, `promise`, `exchange` |
| fresh | `Nav`, `entry`, `completion`, tracker words | `completion`, `tracker`, `stats`, `bonus`, `Collector` |

What to notice:

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

Selected rows from the full-layer sweep:

| Anchor | Layer | Old-context J-lens | Fresh J-lens | Reading |
| --- | ---: | --- | --- | --- |
| `Delta` | 47 | `refers`, `=`, `represents`, `means` | `Delta`, `airport`, `airline`, `Sky`, `River` | Local route label vs airline/name priors. |
| `Orange` | 50 | `refers`, `signifies`, `represents`, `symbol`, `denotes` | `Orange`, `orange`, `citrus`, `color`, `Juice` | Local tag role vs ordinary color/fruit priors. |
| `cooler` | 36 | `canceled`, `replaced`, `rejected`, `failed`, `refused` | `freezer`, `fridge`, `camping`, `cooler` | Stale instruction vs generic object semantics. |

What to notice:

These are the kinds of meanings conversation summaries are full of: aliases,
exceptions, stale plans, and compact labels whose real meaning was established
earlier. A text-only summary can include the labels and glosses while still
reconstructing a weaker state around the labels themselves.

## Example 5: `Falcon` And `Patch 17`, Coding-Adjacent Labels

Visible snippet:

```text
| Falcon | old rollback branch | rejected because it drops subscription coupons |
| Patch 17 | stale patch label | do not cite as live |
```

Selected rows from the full-layer sweep:

| Anchor | Layer | Old-context J-lens | Fresh J-lens | Reading |
| --- | ---: | --- | --- | --- |
| `Falcon` | 43 | `rejected`, `obsolete`, `deprecated`, `failed`, `outdated` | `Falcon`, `Flight`, `Aviation`, `eagle`, `Aerospace` | Rejected branch status vs bird/aviation priors. |
| `Patch 17` span | 41 | `outdated`, `obsolete`, `old`, `expired`, `legacy` | `patch`, `repair`, `fixes`, `revision`, `testing` | Stale label vs generic patch/version semantics. |

What to notice:

This is the bridge to coding agents. Real coding work is packed with compact
operational labels: branch names, issue IDs, file paths, commands, patch
numbers, failing tests, line ranges. The useful state is often the status of
those labels: current, rejected, failed, stale, already tried, next to inspect.

The Markdown table also exposes a reporting hazard. Some high-divergence rows
are punctuation, spaces, table separators, or subword fragments. Public
examples should group tokens into spans such as `Patch 17`; raw token rows are
too noisy for humans.

## Example 6: `Ghost2`, A Vivid But Secondary Example

Visible snippet:

```text
Replacement Ralts Ghost2 is alive and made it to Victory Road.
```

Anchor: `2` in `Ghost2`
Layer: 44

| Path | Next-token candidates | J-lens readout |
| --- | --- | --- |
| old-context | `survived`, `is`, `made`, `survives`, `has` | `survived`, `successfully`, `surviving`, `survives`, `retained` |
| fresh | `is`, `was`, `joined`, `started`, `replaced` | `replacement`, `replaced`, `aka`, `renamed`, `second` |

What to notice:

The old-context path carries survival. The fresh path emphasizes replacement
and naming. This is intuitive and memorable, which makes it useful for
explaining the phenomenon.

It should not lead the evidence. The local phrase says `Ghost2 is alive`, so
ordinary next-token prediction already sees survival language. It is a good
illustration, not the cleanest control.

## Coding Bridge: SWE-Style Action Spans

We also sampled J-lens readouts on true next assistant actions from SWE-Gym
style trajectories. These are teacher-forced readouts of the known next action,
not agent benchmark scores.

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

What to notice:

This is promising mainly as a pointer to the right unit of analysis. Coding
examples should be span-first: full file paths, tool names, commands, symbols,
line ranges, and named tests. Token-level examples are often unreadable because
paths and tool-call formats split into many pieces.

These rows do not show that ValueGraft improves agent coding. They show that
the readout method can localize differences on operational spans that matter
for agent actions.

## What The Examples Support

The current J-lens examples support a narrow qualitative claim:

```text
The same fixed summary tokens, evaluated under an old-context path and a
fresh compacted path, can have visibly different residual-stream readout
neighborhoods.
```

In the best examples, the old-context readout points toward the local status or
role of an identifier, while the fresh readout points toward generic lexical
priors. That is exactly the kind of state ValueGraft is designed to preserve.

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

## Related Work

This note sits at the intersection of three literatures.

First, the transformer attention framing comes from Vaswani et al. (2017):
attention maps queries against keys and returns weighted sums of values.
ValueGraft operates on that cached key/value state at a compaction boundary.

Second, prior work already shows that KV caches can carry useful computation,
be edited, be reused, or be compacted. Li (2026), in "Models Take Notes at
Prefill," is the closest mechanistic neighbor: it argues that prefill writes
field-conditioned conclusions into downstream cache state and demonstrates
editable, composable cached blocks. Zweiger et al. (2026) study latent KV
compaction directly through attention matching. KVLink (Yang et al., 2025) and
CacheBlend (Yao et al., 2025) reuse or blend precomputed caches for chunked
contexts, mostly in retrieval-style settings. These papers mean we should not
claim novelty for "KV state is meaningful" or "cache blocks can be reused." The
narrow question here is conversation compaction: can the state associated with
a human-readable summary help after the old transcript has been replaced?

Third, the readout method comes from the logit-lens lineage. The original logit
lens projects intermediate residual states through the unembedding;
tuned-lens work learns layer-specific translators; the Jacobian lens introduced
by Gurnee et al. (2026) uses an averaged Jacobian transport into the final-layer
basis. We use the J-lens as interpretability telemetry for residual-stream
states, not as a direct KV-cache measurement.

References:

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
- Qwen Team. 2026.
  [Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model](https://qwen.ai/blog?id=qwen3.6-27b).
