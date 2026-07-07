# ValueGraft Under the Lens

Status: standalone report draft
Date: 2026-07-07
Data artifact: `outputs/qwen36_boundary_three_state_probe.json`
Validator: `validate_three_state_probe.py`

## Overview

ValueGraft is a cache-state intervention for conversation compaction. In an
ordinary compacted conversation, the model receives a summary plus recent tail
and freshly encodes that text. In a ValueGraft-style compacted conversation,
the visible text is still the summary plus recent tail, but selected cache
entries are blended from the state that was written when the summary was
generated under the original long context.

This report asks a narrow mechanistic question:

```text
Among compacted variants with identical visible text, does a value-state graft
move the model's internal readout toward the full-context run?
```

The probe is deliberately small. It uses Qwen3.6-27B on a controlled coding
scenario about a package named `rivermark`, then inspects the model with the
Jacobian lens, or J-lens. The target continuation is:

```text
Update src/rivermark/sort.py so wet driftwood is inspected first, then run tests/test_sort.py.
```

The important result is not simply that one next-token argmax changes. The
important result is that aligned value grafting and a deliberately misaligned
shifted control behave differently under the J-lens. In this single probe:

- alpha-zero exactly matches fresh compaction;
- aligned `alpha_V = 0.75` moves layer-48 readouts closer to the full-context
  state and rescues the next-token argmax at ` inspected`;
- `alpha_V = 1.0` rescues more local argmaxes but worsens the internal
  readout distance;
- the shifted-value control also rescues one local argmax, but its J-lens
  readouts move much farther from full context.

Next-token behavior is useful, but too thin by itself. The lens readout gives
a second view: not just "what token came next?" but "what
concept-neighborhood is active at this position?"

## The Compaction Problem

A summary can preserve words without preserving the state that those words had
when they were written.

In a long conversation, the model may resolve local meanings: a name, a file,
a disambiguated sense of a word, an object that needs to be changed, or a
constraint that came from several turns earlier. When the conversation is
compacted, the summary may mention the right text, but the model then encodes
that summary in a shorter context. The text is visible, but the write-time
key/value states from the original context are gone.

ValueGraft tests whether some of that write-time state can be carried forward.
The question is not whether the summary text matters. Of course it does. The
question is whether the same summary text has a better downstream state when
selected old-context values are blended into the freshly encoded compacted
cache.

## The Intervention

For each aligned summary-token position, the probe has two cache traces:

- `V_fresh`: the value state produced by freshly encoding the compacted
  summary;
- `V_write_time`: the value state written when the summary was produced under
  the original context.

This artifact tests a V-only graft:

```text
K_final = K_fresh
V_final = (1 - alpha_V) * V_fresh + alpha_V * V_write_time
```

The implementation aligns 96 summary-token positions and changes value entries
in 16 value-cache layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51,
55, 59, and 63. Fresh keys and non-value state are left as in the compacted
run.

That detail matters. This is not a hidden full-context replay. The visible
text, fresh positions, fresh keys, and non-value state remain the compacted
run. The intervention is only a value-state blend at aligned summary-token
positions.

The run records:

| Condition | Visible text | Cache state |
| --- | --- | --- |
| Full context | Original conversation plus probe question | Normal full-context cache |
| Fresh compacted | Summary plus recent tail plus probe question | Fresh compacted cache |
| Alpha-zero control | Same as fresh compacted | Graft path with `alpha_V = 0` |
| Aligned ValueGraft | Same as fresh compacted | Aligned write-time values blended into fresh values |
| Shifted control | Same as fresh compacted | Same old values shifted to wrong summary-token positions |

The alpha sweep records `alpha_V = 0`, `0.25`, `0.5`, `0.75`, and `1.0`.
The artifact filename contains an older "three-state" label because the first
version stored only the full, fresh, and grafted paths; this version also
contains the alpha sweep and shifted control.

## What the J-Lens Adds

The J-lens maps a residual-stream activation into a vocabulary basis using an
averaged Jacobian transport, then decodes that transported vector with the
model's unembedding. It emits vocabulary tokens, so its output looks like a
next-token table, but it is not the model's ordinary next-token distribution.

In this report:

- **next-token candidates** show what token the model is about to emit before
  the forced target token;
- **J-lens candidates** show vocabulary concepts read from the residual-stream
  state at that forced token.

The distinction is crucial. A path token like `river` may be easy for every
condition to predict as the next token. That does not tell us whether the
internal state around `rivermark` is about a file path, a river bank, a test
command, or generic string continuation. The J-lens can expose those
neighborhoods.

The lens is still only telemetry. It does not label individual KV vectors and
it does not replace task metrics. Its value here is that it makes the cache
intervention inspectable at the positions where the continuation is being
forced.

## Scenario

The compacted task state used by the probe is the exact generated summary
below. The final line ends mid-path in the artifact; that truncation is part of
the compacted state being tested.

```text
Task: Fix `rivermark` package where "bank" refers to a river bank. The failing
test in `tests/test_sort.py` concerns sorting driftwood by distance from the
waterline.

Current State:
- Bug identified in `src/rivermark/sort.py`: Wet driftwood is currently
  treated as lower priority.
- Specification requirement: Wet driftwood must be inspected first (higher
  priority).
- Next Step: Update the priority key logic in `src
```

The probe question is:

```text
Continue the task. What exact file should be changed next, and what test should be run? Answer in one sentence.
```

The probe uses the following intended continuation. It is teacher-forced one
token at a time so every condition can be inspected at identical token
positions; this is a measurement setup, not a claim that the model would
freely generate the whole sentence.

```text
Update src/rivermark/sort.py so wet driftwood is inspected first, then run tests/test_sort.py.
```

For a target token `t`, the next-token table is recorded immediately before
forcing `t`; the J-lens table is recorded after `t` has been fed and the
residual stream at that position exists. That timing is why the two tables can
look related while still measuring different things.

## Aggregate Results

The validator confirms that all conditions share the same 23 forced target
tokens. Alpha-zero matches fresh exactly on token IDs and scores. The aligned
and shifted grafts both use 96 summary-token value pairs.

Layer closure is measured as the improvement in mean Jaccard distance between
a condition's J-lens top-k set and the full-context top-k set:

```text
closure(layer, condition)
  = mean_t distance(full_context_t, fresh_compacted_t)
  - mean_t distance(full_context_t, condition_t)
```

The mean is over the 23 forced target tokens at that layer. Positive means the
condition is closer to full context than fresh compaction. Negative means
farther. An argmax rescue means fresh compaction missed the full-context
next-token argmax and the condition recovered it. An argmax regression means
fresh matched full context and the condition moved away. Changed argmax counts
all positions where the condition's next-token argmax differs from fresh.

| Condition | Argmax rescues | Argmax regressions | Changed argmax | Layer-48 closure | Layer-62 closure |
| --- | ---: | ---: | ---: | ---: | ---: |
| `alpha_V = 0` | 0 | 0 | 0 | 0.0000 | 0.0000 |
| `alpha_V = 0.25` | 0 | 0 | 0 | 0.0019 | -0.0097 |
| `alpha_V = 0.5` | 0 | 0 | 0 | 0.0129 | -0.0193 |
| `alpha_V = 0.75` | 1 | 0 | 1 | 0.0337 | -0.0146 |
| `alpha_V = 1.0` | 2 | 0 | 2 | -0.0184 | -0.0265 |
| Shifted `alpha_V = 0.75` | 1 | 0 | 1 | -0.0903 | -0.0560 |

The best balance in this probe is aligned `alpha_V = 0.75`. It produces one
next-token rescue and the strongest layer-48 closure. Full value replacement
gets two argmax rescues, but moves the internal readout farther from full
context at layers 48 and 62. The shifted control gets one argmax rescue, but
its layer-48 closure is strongly negative.

For aligned `alpha_V = 0.75`, the sampled layers are:

| Layer | Full vs fresh | Full vs aligned graft | Fresh vs aligned graft | Closure |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.1488 | 0.1430 | 0.0657 | 0.0058 |
| 32 | 0.1899 | 0.1788 | 0.0676 | 0.0111 |
| 48 | 0.3882 | 0.3546 | 0.1838 | 0.0337 |
| 62 | 0.3359 | 0.3506 | 0.1507 | -0.0146 |

The readout is not uniformly positive. Layers 16, 32, and 48 move closer to
full context. Layer 62 moves slightly farther away. That is consistent with
late-layer lens readouts being more tied to immediate continuation pressure.

## Reading the Examples

The example tables below are intentionally not just score summaries. They show
the vocabulary neighborhoods that the model exposes through the J-lens.

Some rows contain subword fragments, duplicate stems, or formatting tokens.
That is normal for a vocabulary-space lens. For readability, the tables below
omit non-ASCII tokens and a few uninformative formatting fragments when they
do not change the interpretation. The raw artifact keeps the complete lists.

## Example 1: `river` in `rivermark`

Target span:

```text
Update src/rivermark/sort.py ...
```

Forced token: `river`, index 3.

This is the cleanest first example because next-token prediction is not the
interesting signal. Every condition predicts the path token `river` or a close
variant before the token is forced. The question is whether the hidden state at
that position looks like the full-context state or merely like generic path
completion.

### Next-Token Candidates

| Condition | Argmax before forcing | Top candidates |
| --- | --- | --- |
| Full context | `river` | `river`, `rior`, `iver`, `river`, `River`, `runner` |
| Fresh compacted | `river` | `river`, `river`, `rior`, `River`, `River`, `iver` |
| Aligned `alpha_V = 0.75` | `river` | `river`, `river`, `rior`, `iver`, `River`, `River` |
| Shifted `alpha_V = 0.75` | `river` | `river`, `runner`, `river`, `rim`, `running`, `River` |

The next-token table mostly says that the spelling of `src/rivermark` is easy
once the answer has begun. It does not say whether the model has carried
forward the river-bank meaning that made this package name relevant.

### J-Lens Readout at Layer 62

| Condition | Layer-62 J-lens readout |
| --- | --- |
| Full context | `mark`, `mark`, `bank`, `marks`, `-mark`, `_mark` |
| Fresh compacted | `tests`, `/tests`, `mark`, `then`, `tests`, `/run` |
| Alpha-zero control | `tests`, `/tests`, `mark`, `then`, `tests`, `/run` |
| `alpha_V = 0.5` | `tests`, `/tests`, `then`, `mark`, `tests`, `/run` |
| Aligned `alpha_V = 0.75` | `mark`, `bank`, `tests`, `mask`, `/tests`, `/run` |
| `alpha_V = 1.0` | `mark`, `bank`, `wood`, `mark`, `marks`, `markdown`, `mask` |
| Shifted `alpha_V = 0.75` | `mark`, `bank`, `_mark`, `-mark`, `marks`, `mark` |

Fresh compaction pulls the layer-62 readout toward action-plan neighbors:
`tests`, `/tests`, `then`, `/run`. Full context is more about the path and its
referent: `mark`, `bank`, and related path fragments. Aligned `alpha_V = 0.75`
moves `mark` and `bank` upward relative to fresh, even though the next-token
argmax did not need help.

The shifted control is a caution. It also surfaces `mark` and `bank` on this
row, which means the row is not enough by itself. Its role is different: it
shows why we need aggregate closure and shifted controls instead of selecting
the most vivid token table. Across the full target sequence, the shifted
condition is much farther from full context than fresh compaction.

## Example 2: `inspected`

Target span:

```text
... wet driftwood is inspected first ...
```

Forced token: ` inspected`, index 13.

This row connects a behavioral change to an internal readout. Fresh compaction
prefers the stem ` priorit`; aligned grafting at `alpha_V = 0.75` returns the
argmax to ` inspected`, matching full context.

### Next-Token Candidates

| Condition | Argmax before forcing | Top candidates |
| --- | --- | --- |
| Full context | ` inspected` | `inspected`, `priorit`, `given`, `treated`, `sorted`, `assigned` |
| Fresh compacted | ` priorit` | `priorit`, `inspected`, `given`, `assigned`, `treated`, `sorted` |
| Alpha-zero control | ` priorit` | `priorit`, `inspected`, `given`, `assigned`, `treated`, `sorted` |
| `alpha_V = 0.25` | ` priorit` | `priorit`, `inspected`, `given`, `assigned`, `treated`, `sorted` |
| `alpha_V = 0.5` | ` priorit` | `priorit`, `inspected`, `given`, `assigned`, `treated`, `sorted` |
| Aligned `alpha_V = 0.75` | ` inspected` | `inspected`, `priorit`, `given`, `assigned`, `treated`, `sorted` |
| `alpha_V = 1.0` | ` inspected` | `inspected`, `priorit`, `given`, `treated`, `assigned`, `sorted` |
| Shifted `alpha_V = 0.75` | ` priorit` | `priorit`, `inspected`, `given`, `treated`, `sorted`, `processed` |

This is a real local rescue, and the shifted condition does not get the same
argmax here. The stronger claim, though, comes from pairing that local rescue
with the lens table.

### J-Lens Readout at Layer 48

| Condition | Layer-48 J-lens readout |
| --- | --- |
| Full context | `priority`, `priorit`, `instead`, `precedence`, `prioritize` |
| Fresh compacted | `priority`, `priorit`, `prioritize`, `precedence`, `faster` |
| Alpha-zero control | `priority`, `priorit`, `prioritize`, `precedence`, `faster` |
| `alpha_V = 0.5` | `priority`, `priorit`, `precedence`, `prioritize`, `faster` |
| Aligned `alpha_V = 0.75` | `priority`, `priorit`, `precedence`, `prioritize`, `faster` |
| `alpha_V = 1.0` | `priority`, `priorit`, `precedence`, `prioritize` |
| Shifted `alpha_V = 0.75` | `priority`, `priorit`, `before`, `prioritize`, `precedence` |

The J-lens readout is not simply saying "the next word is inspected." It is
showing a priority/ordering neighborhood around the phrase. Full context has
`instead` and `precedence` high in the list; fresh compaction has
`prioritize` and `faster`; aligned grafting shifts the ordering toward
`precedence` while also producing the next-token rescue.

### J-Lens Readout at Layer 62

| Condition | Layer-62 J-lens readout |
| --- | --- |
| Full context | `first`, `before`, `First`, `first`, `highest`, `with`, `by` |
| Fresh compacted | `first`, `First`, `first`, `before`, `with`, `highest`, `-first` |
| Aligned `alpha_V = 0.75` | `first`, `First`, `with`, `first`, `before`, `highest`, `by` |
| Shifted `alpha_V = 0.75` | `first`, `before`, `First`, `first`, `highest`, `with`, `-first` |

Layer 62 is highly readable but less discriminating. All conditions are in the
"first/before/highest" neighborhood because the local target text and the
summary already make the priority relation explicit. This is why the aggregate
analysis emphasizes layer 48 rather than treating the most human-readable
late-layer rows as the whole result.

## Example 3: `wet`

Target span:

```text
... so wet driftwood ...
```

Forced token: ` wet`, index 9.

This row illustrates the kind of semantic field the lens exposes. The
next-token candidates are mostly local grammar and adjective choice; the
J-lens candidates are about the physical scene around the object.

### Next-Token Candidates

| Condition | Argmax before forcing | Top candidates |
| --- | --- | --- |
| Full context | ` that` | `that`, `wet`, `the`, `it`, `dry`, `its` |
| Fresh compacted | ` wet` | `wet`, `that`, `the`, `its`, `it`, `dry` |
| Aligned `alpha_V = 0.75` | ` wet` | `wet`, `that`, `the`, `its`, `it`, `dry` |
| Shifted `alpha_V = 0.75` | ` wet` | `wet`, `that`, `the`, `dry`, `it`, `its` |

Full context prefers `that` here, while the compacted conditions prefer `wet`.
That next-token difference is not the main evidence because the intended
continuation is forced for inspection. The lens readout below is the more
informative part of the row.

### J-Lens Readout at Layer 62

| Condition | Layer-62 J-lens readout |
| --- | --- |
| Full context | `drift`, `drifting`, `river`, `drifted`, `float`, `drain` |
| Fresh compacted | `drift`, `river`, `drifting`, `drifted`, `bank`, `float`, `priority` |
| Aligned `alpha_V = 0.75` | `drift`, `river`, `drifting`, `bank`, `drifted`, `drain`, `float` |
| `alpha_V = 1.0` | `drift`, `river`, `drifting`, `bank`, `drifted`, `drain` |
| Shifted `alpha_V = 0.75` | `drift`, `drifting`, `drifted`, `river`, `float`, `wood` |

Here the readout is about drift, river, bank, floating, and drainage. Aligned
grafting raises `bank` and `drain` relative to fresh and keeps the row in the
river/driftwood neighborhood. The shifted row remains plausible, but it
emphasizes `wood` and local continuation more than the aligned row.

This example is useful precisely because it is not a clean next-token rescue.
It shows the qualitative kind of state difference that the aggregate Jaccard
closure score is trying to summarize.

## Example 4: `is`

Target span:

```text
... driftwood is inspected ...
```

Forced token: ` is`, index 12.

This row shows why argmax rescues are insufficient. A condition can recover
the full-context next-token argmax at one position while being a worse internal
match across the sequence.

### Next-Token Candidates

| Condition | Argmax before forcing | Top candidates |
| --- | --- | --- |
| Full context | ` is` | `is`, `has`, `on`, `gets`, `near`, `receives` |
| Fresh compacted | ` has` | `has`, `is`, `on`, `receives`, `gets`, `from` |
| Aligned `alpha_V = 0.75` | ` has` | `has`, `is`, `on`, `receives`, `gets`, `from` |
| `alpha_V = 1.0` | ` is` | `is`, `has`, `on`, `receives`, `gets`, `from` |
| Shifted `alpha_V = 0.75` | ` is` | `is`, `has`, `on`, `gets`, `receives`, `sorts` |

Full replacement and shifted values both recover the full-context argmax at
this position. If the analysis only counted local argmax rescues, both would
look better than aligned `alpha_V = 0.75`.

### J-Lens Readout at Layer 62

| Condition | Layer-62 J-lens readout |
| --- | --- |
| Full context | `inspected`, `priorit`, `treated`, `sorted`, `inspect`, `assigned` |
| Fresh compacted | `inspected`, `treated`, `priorit`, `assigned`, `sorted`, `given` |
| Aligned `alpha_V = 0.75` | `inspected`, `treated`, `priorit`, `assigned`, `given`, `sorted` |
| `alpha_V = 1.0` | `inspected`, `treated`, `priorit`, `assigned`, `given`, `sorted` |
| Shifted `alpha_V = 0.75` | `inspected`, `priorit`, `treated`, `processed`, `sorted`, `handled` |

This local row looks semantically reasonable in several conditions. The reason
it belongs in the report is that it prevents an overly simple interpretation:
recovering one next-token argmax can happen under a misaligned perturbation.
The aggregate readout says the shifted condition is much farther from full
context than fresh compaction at every sampled layer.

## Example 5: `wood` in `driftwood`

Target span:

```text
... driftwood is inspected ...
```

Forced token: `wood`, index 11.

This is a small example rather than a showcase. It is included because it
explains how the aggregate layer-48 movement can be made of many modest
rank-order changes rather than one dramatic token replacement.

### Next-Token Candidates

| Condition | Argmax before forcing | Top candidates |
| --- | --- | --- |
| Full context | `wood` | `wood`, `woods`, `wood`, `Wood`, `WOOD`, `water` |
| Fresh compacted | `wood` | `wood`, `woods`, `wood`, `Wood`, `WOOD`, `wo` |
| Aligned `alpha_V = 0.75` | `wood` | `wood`, `woods`, `wood`, `water`, `WOOD`, `Wood` |
| Shifted `alpha_V = 0.75` | `wood` | `wood`, `wood`, `on`, `is`, `has`, `gets` |

Every condition predicts `wood`, so the next-token table is almost entirely a
spelling check.

### J-Lens Readout at Layer 48

| Condition | Layer-48 J-lens readout |
| --- | --- |
| Full context | `priorit`, `priority`, `prioritize`, `priority`, `precedence` |
| Fresh compacted | `priority`, `priorit`, `prioritize`, `priority`, `precedence` |
| Alpha-zero control | `priority`, `priorit`, `prioritize`, `priority`, `precedence` |
| `alpha_V = 0.5` | `priority`, `priorit`, `prioritize`, `priority`, `precedence` |
| Aligned `alpha_V = 0.75` | `priorit`, `priority`, `prioritize`, `priority`, `precedence` |
| `alpha_V = 1.0` | `priorit`, `priority`, `prioritize`, `priority`, `precedence` |
| Shifted `alpha_V = 0.75` | `priority`, `priorit`, `priority`, `prioritize`, `precedence` |

All states are in the priority neighborhood because the summary explicitly
says wet driftwood should have higher priority. The distinction is rank rather
than identity: full context puts the `priorit` stem first, fresh puts
`priority` first, and aligned `alpha_V = 0.75` restores the full-context
ordering.

### J-Lens Readout at Layer 62

| Condition | Layer-62 J-lens readout |
| --- | --- |
| Full context | `is`, `has`, `gets`, `receives`, `on`, `near`, `inspected` |
| Fresh compacted | `has`, `receives`, `gets`, `on`, `is`, `sorts`, `ranks` |
| Aligned `alpha_V = 0.75` | `has`, `receives`, `gets`, `on`, `is`, `sorts`, `ranks` |
| Shifted `alpha_V = 0.75` | `has`, `gets`, `receives`, `on`, `is`, `sorts`, `ranks` |

Layer 62 does not favor aligned grafting here. That is part of the result:
the mid-layer readout improves under aligned moderate blending, while later
continuation-like readouts are mixed.

## What the Controls Show

The alpha sweep and shifted control are the strongest part of this artifact.

Alpha-zero is a plumbing check. It exactly matches fresh compaction, including
J-lens token IDs and scores. That means the mere act of snapshotting,
rebuilding, and replaying the cache is not creating the changes.

The alpha sweep is non-monotonic. Small blending (`0.25`, `0.5`) shifts the
layer-48 readout slightly toward full context without changing next-token
argmaxes. Moderate blending (`0.75`) gives the strongest layer-48 closure and
one next-token rescue. Full replacement (`1.0`) rescues two local argmaxes but
worsens layer-48 and layer-62 closure.

The shifted control is the most important negative control. It uses old value
states, but at the wrong aligned positions. If old values were merely a generic
helpful perturbation, shifted values should look roughly like aligned values.
They do not:

| Condition | Layer-16 closure | Layer-32 closure | Layer-48 closure | Layer-62 closure |
| --- | ---: | ---: | ---: | ---: |
| Aligned `alpha_V = 0.75` | 0.0058 | 0.0111 | 0.0337 | -0.0146 |
| Shifted `alpha_V = 0.75` | -0.0957 | -0.0836 | -0.0903 | -0.0560 |

The shifted condition can still look good on selected rows. That is why the
aggregate table matters. Across the target sequence, shifted values move the
internal readout farther from full context at every sampled layer.

## Interpretation

The safest interpretation is:

1. The intervention is active. Aligned value grafting changes downstream
   prediction and internal readouts while alpha-zero remains identical to
   fresh.
2. The effect is alignment-sensitive. Shifting old values to the wrong
   summary-token positions produces a much worse internal match to full
   context.
3. Moderate blending is better than full replacement in this example.
   `alpha_V = 1.0` can produce local next-token wins while degrading the
   J-lens readout.
4. The J-lens adds information beyond next-token candidates. Some rows have no
   interesting next-token difference but do have readable internal differences.
5. The result is still narrow. It is one constructed probe, not a task-level
   success result and not an effect-size estimate.

This is the kind of artifact that can make ValueGraft legible. The behavioral
benchmark tells us whether the method helps. The lens probe helps us see what
kind of internal state changes when it helps, and warns us when a local
next-token win may be caused by a worse perturbation.

## Scope and Next Work

This artifact studies one model, one generated summary, one target
continuation, and V-only grafting. The target is teacher-forced so every
condition can be inspected at the same positions. The J-lens readout is
residual-stream telemetry, not direct access to individual KV vectors.

The natural next steps are:

- run the same alpha/shifted-control design on several existing trajectory
  prediction examples;
- add a wrong-conversation graft control;
- test K-only and independent K/V policies under the same visible text;
- connect lens movement to behavioral task metrics;
- choose examples where the key concept is not already explicit in the
  summary text, so the lens has a harder job.

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.
  N., Kaiser, L., & Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). arXiv:1706.03762.
- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
  [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). arXiv:2104.09864.
- Li, B. (2026). [Models Take Notes at Prefill: KV Cache Can Be Editable and Composable](https://arxiv.org/abs/2606.17107). arXiv:2606.17107.
- nostalgebraist. (2020). [Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). LessWrong.
- Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney,
  L., Biderman, S., & Steinhardt, J. (2023). [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112). arXiv:2303.08112.
- Gurnee, W., Sofroniew, N., Pearce, A., Piotrowski, M., Kauvar, I., Chen,
  R., Soligo, A., Bogdan, P., Ong, E., Wang, R., Thompson, B., Abrahams, D.,
  Kantamneni, S., Ameisen, E., Batson, J., & Lindsey, J. (2026).
  [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/). Transformer Circuits Thread.
- Anthropic. (2026). [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens). Reference implementation for the Jacobian lens.
