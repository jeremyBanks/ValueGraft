# Interpreting ValueGraft Through Concrete Examples

Status: standalone explanatory draft  
Date: 2026-07-07  
Audience: readers with basic transformer familiarity, but not necessarily
mechanistic interpretability background

## Short Version

Conversation compaction usually preserves text, not the full internal state
the model had when that text was written. ValueGraft is an attempt to preserve
some of that state at the compaction boundary.

The examples in this note use a Jacobian-lens readout, or J-lens, to make that
idea visible. We compare the same literal summary text in two states:

- **Write-time state:** the model sees the summary after the original
  conversation that produced it.
- **Fresh state:** the model sees the same summary text re-encoded later,
  without the original conversation.

The most important example is `B-410`. The visible summary says:

```text
Permit: B-410 is stale. Use P-771.
```

At the `B` in `B-410`, ordinary next-token prediction mostly wants to continue
the code with punctuation and digits. The write-time J-lens readout instead
surfaces words like `obsolete`, `outdated`, `deprecated`, and `expired`. The
fresh readout treats the span more like a generic civic permit code:
`municipal`, `City`, `Civic`, `License`.

That is the point in miniature. The text contains the same identifier in both
states, but the write-time state appears to carry a more situated meaning. The
claim is not that this proves ValueGraft improves behavior. The claim is that
this gives a concrete way to see the kind of context-conditioned state
ValueGraft is trying to preserve.

## What ValueGraft Is Trying To Preserve

In a transformer, each token position contributes key and value tensors to the
attention mechanism. Roughly, keys help later tokens decide which earlier
positions are relevant, and values carry the content that later tokens mix in
when they attend to those positions. This is the standard key/value attention
framing from the Transformer architecture.

When a model writes a summary after a long conversation, the summary tokens are
not just bare text. They are produced while the model is still conditioned on
the full conversation. A compact label such as `B-410`, `Maple`, or `Falcon`
may therefore be represented with local role information: stale permit, library
room, rejected rollback branch.

When the same summary is later re-encoded from text alone, the model still sees
the words. But it may reconstruct a more generic representation: permit-like
identifier, street name, bird/aviation word, patch/version label.

ValueGraft explores whether we can carry some write-time key/value state across
that boundary instead of relying only on text re-encoding. In the broader
formulation, this can include different choices about whether to blend keys,
values, or both, and how strongly to blend them. This note does not evaluate
those parameter choices. It explains what kind of state might be worth
preserving.

## What The J-Lens Adds

The J-lens is an interpretability tool for reading intermediate residual-stream
states. Given a token position and layer, it produces a ranked list of
vocabulary tokens that are associated with what the model could verbalize from
that internal state. It is related to logit-lens and tuned-lens methods, but
uses an averaged Jacobian map designed to compensate for layer-to-layer
representational changes.

This matters because a raw J-lens list is easy to misread. It is not a belief
scanner. It is not behavior. It is not the KV cache. It is a vocabulary-shaped
view into residual-stream state.

Used carefully, it can still answer a useful question:

```text
When the same summary token is read in write-time and fresh states,
does the internal readout point toward the same meaning?
```

For our current Qwen3.6-27B probes, layer 48 is the best default display layer.
The full-layer sweep found that layer 48 had the largest average
write-time/fresh separation while staying relatively distinct from ordinary
next-token probabilities. Layer 62 can be more readable in some examples, but
it is also much more next-token-like, so it is weaker evidence for a distinct
semantic readout.

## How To Read The Tables

Each example uses the same comparison:

| Field | Meaning |
| --- | --- |
| Visible summary text | The literal compacted text seen in both states. |
| Anchor | The token or span being inspected. |
| Actual next token | The literal token after the anchor, when applicable. |
| Next-token candidates | Ordinary top next-token predictions. This is a control. |
| Write-time J-lens | Vocabulary readout when the summary is seen after the old conversation. |
| Fresh J-lens | Vocabulary readout when the same summary is re-encoded without the old conversation. |
| Interpretation | What the contrast suggests, and what it does not show. |

The examples below abbreviate top-k lists for readability. The full artifacts
are listed at the end.

## Example 1: `B-410`

This is the cleanest example so far because the next-token distribution is
mostly busy continuing an identifier, while the write-time J-lens readout
surfaces the status of that identifier.

Visible summary text:

```text
Permit: B-410 is stale. Use P-771 on the insurance form.
```

Anchor: `B` in `B-410`  
Display layer: 48  
Actual next token: `-`

| State | Next-token candidates | J-lens candidates |
| --- | --- | --- |
| write-time | `-`, `4`, `-st`, digits/punctuation | `obsolete`, `outdated`, `deprecated`, `expired` |
| fresh | `-`, `2`, `PD`, `1`, `3`, `4` | `municipal`, `City`, `Civic`, `License`, `permit` |

At the surface level, both states contain the same text. But the readouts
separate. The write-time state reads `B-410` as a stale/obsolete permit. The
fresh state reads it more like a generic municipal code.

This is stronger than looking at the explicit word `stale`. If we inspect the
word `stale`, fresh encoding also recovers words like `expired`, `outdated`,
and `obsolete`, because the text says that directly. The important anchor is
the identifier span. The question is whether `B-410` itself carries stale
status, not whether the word `stale` means stale.

Why this matters for ValueGraft: a later task might say "use the current permit"
or "which permit goes on the form?" The visible summary contains the needed
words, but the write-time state appears to bind the status to the identifier
more directly.

## Example 2: `Maple`

This is an ordinary planning example. It is less clean than `B-410`, because
the local text says `Maple means...`, so next-token prediction already has a
definition cue. It is still useful because the write-time and fresh states
point to different senses of the same word.

Visible summary text:

```text
Riverside block party state:
- Maple means the library's Maple Room for storage and volunteer check-in.
```

Anchor: `Maple`  
Display layer: 48  
Actual next token: `means`

| State | Next-token candidates | J-lens candidates |
| --- | --- | --- |
| write-time | `=`, `is`, `refers`, `means`, `Room` | `refers`, `=`, `referring`, `denotes` |
| fresh | `Street`, `St`, `Ave`, `street`, `Drive` | `Street`, `street`, `neighborhood`, `park`, `City` |

The write-time readout treats `Maple` as a defined local referent. The fresh
readout drifts toward ordinary place-name and street-name priors.

This should not be overclaimed. Because the summary itself says `Maple
means...`, ordinary next-token prediction is already informative. The useful
contrast is between a locally defined room label and a generic named-place
reading.

Why this matters for ValueGraft: compact summaries often introduce local
aliases. A text-only summary may keep the alias and its gloss, while still
reconstructing a weaker or more generic state around the alias itself.

## Example 3: `Delta`, `Orange`, And `cooler`

The household checklist examples show the same pattern without game jargon or
software jargon.

Visible summary text:

```text
- Delta = ferry route, not airline; booked Delta 6 at 7:40.
- Orange = lockbox tag color; orange key opens kayak shed.
- Big cooler instruction is stale; bring two soft coolers.
```

Selected full-layer sweep rows:

| Anchor | Layer | Write-time J-lens | Fresh J-lens | Interpretation |
| --- | ---: | --- | --- | --- |
| `Delta` | 47 | `refers`, `=`, `represents`, `means` | `Delta`, `airport`, `airline`, `Sky`, `River` | Local ferry route label vs airline/name priors. |
| `Orange` | 50 | `refers`, `signifies`, `represents`, `symbol`, `denotes` | `Orange`, `orange`, `citrus`, `color`, `Juice` | Local tag-color role vs ordinary color/fruit priors. |
| `cooler` | 36 | `canceled`, `replaced`, `rejected`, `failed`, `refused` | `freezer`, `fridge`, `camping`, `backpack`, `cooler` | Stale/replaced instruction vs generic object semantics. |

These examples are helpful because they are mundane. ValueGraft is not mainly
about exotic puzzles. The motivating case is normal conversation state: private
labels, changed plans, stale instructions, and local roles whose meaning is
obvious in context but underdetermined from a short summary.

## Example 4: `Falcon` And `Patch 17`

This example connects the same phenomenon to operational and coding-adjacent
contexts. The summary is a Markdown table about a software incident.

Visible summary text:

```text
| Falcon | old rollback branch | rejected because it drops subscription coupons |
| Patch 17 | stale patch label | do not cite as live |
```

Selected full-layer sweep rows:

| Anchor | Layer | Write-time J-lens | Fresh J-lens | Interpretation |
| --- | ---: | --- | --- | --- |
| `Falcon` | 43 | `rejected`, `obsolete`, `deprecated`, `failed`, `outdated` | `Falcon`, `Flight`, `Aviation`, `eagle`, `Aerospace` | Rejected branch status vs bird/aviation prior. |
| `Patch 17` span | 41 | `outdated`, `obsolete`, `old`, `expired`, `legacy` | `patch`, `repair`, `fixes`, `revision`, `testing` | Stale patch label vs generic version/repair semantics. |

This is where the practical coding relevance starts to become visible. Agentic
coding often turns on compact operational labels: branch names, issue IDs,
file names, patches, commands, and line ranges. The J-lens results suggest
that write-time state may carry which of those labels are current, rejected,
stale, or actionable.

The table format also shows a limitation. Markdown pipes, table separators,
and subword fragments produce many high-divergence rows that are not good
human examples. A useful automatic presentation must group tokens into spans
such as `Patch 17`, not report isolated punctuation.

## Example 5: `Ghost2`

The Pokemon example is vivid, but it should not lead the story. It works best
as an intuition pump once the ordinary examples have established the pattern.

Visible summary text:

```text
Replacement Ralts Ghost2 is alive and made it to Victory Road.
```

Anchor: `2` in `Ghost2`  
Selected layer: 44

| State | Next-token candidates | J-lens candidates |
| --- | --- | --- |
| write-time | `survived`, `is`, `made`, `survives`, `has` | `survived`, `successfully`, `surviving`, `survives`, `retained` |
| fresh | `is`, `was`, `joined`, `started`, `replaced` | `replacement`, `replaced`, `aka`, `renamed`, `second` |

The write-time state strongly carries the "survived replacement" meaning. The
fresh state keeps more of a replacement-marker reading.

This is not the cleanest next-token control because the local phrase says
`Ghost2 is alive`. Ordinary next-token prediction can therefore see much of
the relevant signal. The useful contrast is that write-time state is more
directly about survival, while fresh encoding is more about replacement/name
semantics.

## Coding Bridge: SWE-Style Action Spans

We also tried the same readout idea on true next actions from SWE-Gym style
trajectories. This is not ready to be a flagship example. It is still useful
as a bridge to the use case we ultimately care about.

In one getmoto trajectory, the true next action was:

```text
str_replace_editor view /workspace/getmoto__moto__4.1/moto/rds/responses.py [584, 600]
```

The span-aware probe found notable full-context vs compacted-context divergence
on:

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

This is encouraging, but the current coding examples are noisier than the
short summary examples. Tool-call syntax, XML-like wrappers, paths, commands,
and subword splits create many unhelpful token-level readouts. The right unit
for coding examples is the span: the full path, command, tool name, symbol, or
line range.

## What This Does Not Prove

The examples above do not prove that ValueGraft improves model behavior. They
show an interpretability pattern:

```text
same visible summary text
different conditioning state
different readable internal neighborhoods
```

That pattern is compatible with the ValueGraft hypothesis, but behavioral
experiments still have to decide whether preserving or blending write-time
KV state actually helps downstream tasks.

The examples also do not show that compaction harm is a new phenomenon.
Compaction losing information is the baseline problem. The interesting
question is whether we can reduce that loss by carrying state that text alone
does not reconstruct.

Finally, the J-lens has limitations:

- It reads residual-stream activations, not the KV cache directly.
- It returns vocabulary-token readouts, so multi-token concepts can be hard to
  see cleanly.
- Late layers can collapse toward ordinary next-token prediction.
- Raw highest-divergence rows can be punctuation, whitespace, table syntax, or
  tokenization artifacts.
- A plausible readout is not a causal intervention.

## What This Suggests

The examples make ValueGraft's target less abstract. The method is not trying
to preserve every hidden detail of the old conversation. It is trying to carry
the compacted summary's situated state: which labels are stale, which aliases
refer to local entities, which branch names are rejected, which path is the
next useful object.

That gives us a more concrete set of predictions:

- ValueGraft should help most when the continuation depends on compact labels
  whose meanings were established earlier.
- It should help less when the summary text explicitly says everything needed
  in a way fresh re-encoding already captures.
- It should be evaluated on behavior, not just readouts.
- A good qualitative figure should show next-token candidates beside J-lens
  readouts, so we do not confuse continuation pressure with internal state.
- Coding examples should be span-first.

## Audit Trail

Primary artifacts:

- `outputs/qwen36_next_token_readout_comparison.json`
- `outputs/qwen36_full_layer_sweep_summary.json`
- local ignored raw file: `outputs/qwen36_full_layer_sweep.json`
- SWE span artifacts:
  - `outputs/qwen36_swegym_next_action_probe_cleanprompt_t0001_t0004.json`
  - `outputs/qwen36_swegym_next_action_probe_trimmed_t0004.json`

Before interpreting the full-layer sweep, validate the raw and compact
artifacts:

```sh
python3 jlens_boundary_probe/validate_full_layer_sweep.py \
  jlens_boundary_probe/outputs/qwen36_full_layer_sweep.json \
  --summary jlens_boundary_probe/outputs/qwen36_full_layer_sweep_summary.json
```

Current expected output:

```text
VALID: demos=7 layers=63 summary_tokens=1175 grid_rows=74025
VALID: compact summary matches raw totals and schema
```

The full-layer sweep found 1,175 summary tokens across 63 fitted layers, for
74,025 token-layer rows. It supports layer 48 as the best default display layer
for these qualitative examples.

## Prior Art And Context

The attention terminology comes from the Transformer architecture introduced
by Vaswani et al. (2017), where attention maps queries against key-value pairs
and outputs a weighted sum of values.

The readout method is related to a line of vocabulary-projection methods:
nostalgebraist's logit lens (2020), Belrose et al.'s tuned lens (2023), and
the Jacobian lens introduced by Gurnee, Lindsey, and collaborators in
`Verbalizable Representations Form a Global Workspace in Language Models`
(2026). The Jacobian-lens paper is especially relevant because it emphasizes
both the usefulness and the limits of using J-lens readouts as a window into
intermediate residual-stream content.

Our implementation used Qwen3.6-27B as the subject model and the public
Neuronpedia Jacobian-lens weights for Qwen3.6-27B. Qwen3.6-27B is useful here
because it is a strong open model that can run in our experimental setup, but
the examples in this note should not be treated as model-general without
replication.

References:

- Vaswani et al., 2017, `Attention Is All You Need`: https://arxiv.org/abs/1706.03762
- nostalgebraist, 2020, `interpreting GPT: the logit lens`: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- Belrose et al., 2023, `Eliciting Latent Predictions from Transformers with the Tuned Lens`: https://arxiv.org/abs/2303.08112
- Gurnee, Lindsey, and collaborators, 2026, `Verbalizable Representations Form a Global Workspace in Language Models`: https://transformer-circuits.pub/2026/workspace/
- Neuronpedia J-lens page for Qwen3.6-27B: https://www.neuronpedia.org/qwen3.6-27b/jlens
- Qwen team, 2026, `Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model`: https://qwen.ai/blog?id=qwen3.6-27b

