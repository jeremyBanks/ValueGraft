# ValueGraft Under the Lens: A Corrected Three-State Readout

Status: corrected standalone draft
Date: 2026-07-07
Primary artifact: `outputs/qwen36_boundary_three_state_probe.json`
Validator: `validate_three_state_probe.py`

## Abstract

ValueGraft is a family of cache-state interventions for conversation
compaction. Ordinary compaction keeps visible text, typically a summary plus a
recent tail, but re-encodes that text in a new context. ValueGraft asks whether
some of the key/value state written under the original long context can be
carried across the boundary so the compacted conversation behaves more like the
uncompacted one.

This note corrects an earlier interpretability mistake. A two-state J-lens
comparison between old-context and fresh summary encodings can show that
context-conditioned residual-stream state exists, but it cannot show the effect
of ValueGraft. The corrected probe uses four states over the same forced target
tokens: `full_context`, `fresh_compacted`, `alpha0_grafted_compacted`, and
`grafted_compacted`. In this single constructed coding-style example, a V-only
graft changes downstream next-token and J-lens readouts relative to the fresh
compacted condition. The most concrete result is one next-token argmax rescue:
for the token ` inspected`, the fresh compacted state predicts ` priorit`,
while the grafted compacted state returns to the full-context argmax
` inspected`. The layer-level readout is mixed: grafted states move modestly
toward full context at layers 16, 32, and 48, but slightly away at layer 62.

This is proof of method, not a general effect-size estimate.

## What ValueGraft Is Testing

In transformer self-attention, each token position contributes keys and values.
Later tokens produce queries, compare those queries to earlier keys, and mix
the corresponding values. The standard Transformer formulation introduced this
query/key/value attention language; modern autoregressive inference stores
previous keys and values as a KV cache so later decode steps can attend to
earlier positions without recomputing the whole prefix.

Conversation compaction creates a specific problem for that mechanism. Suppose
the model has a long conversation, generates or receives a summary, and then
continues with only the summary plus recent turns. The visible words may be
adequate, but the key/value states for those words are now freshly written in a
shorter context. Any context-conditioned interpretation that existed when the
summary was produced under the full conversation has to be reconstructed from
text.

ValueGraft parameterizes interventions at that boundary:

```text
K_final = (1 - alpha_K) * K_fresh
        + alpha_K       * K_write_time_rerotated

V_final = (1 - alpha_V) * V_fresh
        + alpha_V       * V_write_time
```

`alpha_K = 0, alpha_V = 0` is ordinary text-only compaction. A V-only graft has
`alpha_K = 0` and `alpha_V > 0`: it keeps fresh attention addresses while
blending in value content written under the old context. A K-only graft would
set `alpha_V = 0` and vary `alpha_K`. A full KV graft can vary both. Keys need
the rerotation qualifier because RoPE-style position encoding rotates queries
and keys by position; values are not RoPE-rotated in the same way.

The corrected J-lens artifact here is V-only:

```text
alpha_K = 0
alpha_V = 0.75
```

The artifact records the policy as:

```text
V-only summary-token value-cache blend; fresh keys and linear-attention
recurrent state preserved
```

It aligns 96 summary-token positions and changes value entries in 16 layers.

## Why the Earlier Two-State Readout Was Not Enough

The broad J-lens sweep compares two paths:

- **write-time summary:** the summary token is processed after the original
  conversation that produced it.
- **fresh summary:** the same literal summary token is re-encoded in the
  compacted context.

That comparison is useful. It can show that the same visible summary text has
different residual-stream readouts depending on whether it was processed under
the old context or freshly. It helps explain what kind of state ValueGraft may
be trying to preserve.

But it omits the intervention. It does not answer:

```text
Does the grafted compacted state move the model away from fresh compaction
and toward the full-context behavior?
```

The corrected probe therefore has to compare at least these states:

| State | Visible context | Cache state |
| --- | --- | --- |
| `full_context` | Original conversation plus the same probe question | Normal full-context cache |
| `fresh_compacted` | Summary plus retained tail plus the same probe question | Normal freshly encoded compacted cache |
| `alpha0_grafted_compacted` | Same as fresh compacted | Graft code path with `alpha_V = 0`; should match fresh |
| `grafted_compacted` | Same as fresh compacted | Fresh compacted cache with aligned summary value states blended from write-time context |

The alpha-0 state is not a scientific arm by itself. It is a machinery check:
if alpha-0 differs from fresh, then the cache rebuild path is changing the
measurement and the artifact is not trustworthy.

## Probe Design

The corrected probe uses Qwen3.6-27B with the Anthropic/Neuronpedia Jacobian
lens weights for that model. It builds a small coding-style scenario around a
package named `rivermark`, where `bank` means river bank, not financial bank.
The compacted summary says:

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
Continue the task. What exact file should be changed next, and what test should
be run? Answer in one sentence.
```

The forced target sequence is:

```text
Update src/rivermark/sort.py so wet driftwood is inspected first, then run
tests/test_sort.py.
```

For each state, the probe teacher-forces that target one token at a time. At
each position it records:

- the model's next-token top-k before the target token is forced;
- the argmax next token before forcing;
- J-lens top-k readouts at sampled layers 16, 32, 48, and 62.

The forced target makes the internal readout comparable across states. It is
not claiming the model would freely generate the entire sentence unaided.

## Validation Checks

The artifact passes `validate_three_state_probe.py` in strict mode. The
validator checks that:

- `graft.available` is true;
- all required states exist;
- all four forced target sequences have the same token IDs;
- graft provenance fields exist;
- the graft changes a positive number of value layers;
- the alpha-0 graft matches fresh compacted exactly on next-token and per-layer
  top-k token IDs.

The alpha-0 result is especially important. It means the observed fresh-vs-graft
differences are not merely artifacts of snapshotting and rebuilding the cache.

## Result Summary

The target contains 23 tokens. The V-only graft changes the next-token argmax
at one target position and creates no argmax regressions under the validator's
definition.

| Metric | Value |
| --- | ---: |
| Aligned summary-token pairs | 96 |
| Changed value-cache layers | 16 |
| Changed value slots | 1,536 |
| Forced target tokens | 23 |
| Argmax rescues | 1 |
| Argmax regressions | 0 |
| Argmax changed by graft | 1 |

Layer-level J-lens movement is measured as mean Jaccard distance between top-k
token sets. Lower full-vs-state distance means closer to full context. Positive
closure means the grafted state moved closer to full context than fresh did.

| Layer | Full vs fresh | Full vs grafted | Fresh vs grafted | Fresh vs alpha-0 | Closure |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.1488 | 0.1430 | 0.0657 | 0.0000 | 0.0058 |
| 32 | 0.1899 | 0.1788 | 0.0676 | 0.0000 | 0.0111 |
| 48 | 0.3882 | 0.3546 | 0.1838 | 0.0000 | 0.0337 |
| 62 | 0.3359 | 0.3506 | 0.1507 | 0.0000 | -0.0146 |

This is not a clean "graft always helps" story. It is a narrower and more
honest result: the intervention is active, alpha-0 is inert, and the grafted
state sometimes moves toward the full-context state on interpretable readouts.

## Concrete Example: The `inspected` Token

The strongest single row occurs at target index 13, where the forced token is
` inspected`.

| State | Argmax before forcing | Next-token top candidates |
| --- | --- | --- |
| `full_context` | ` inspected` | `inspected`, `priorit`, `given`, `treated`, `sorted`, `assigned`, `highest`, `handled` |
| `fresh_compacted` | ` priorit` | `priorit`, `inspected`, `given`, `assigned`, `treated`, `sorted`, `higher`, `highest` |
| `alpha0_grafted_compacted` | ` priorit` | `priorit`, `inspected`, `given`, `assigned`, `treated`, `sorted`, `higher`, `highest` |
| `grafted_compacted` | ` inspected` | `inspected`, `priorit`, `given`, `assigned`, `treated`, `sorted`, `processed`, `highest` |

This is the cleanest behavioral sign in the artifact. With the same visible
compacted text, the V-only graft changes the next-token argmax from the fresh
compacted answer back to the full-context answer. The alpha-0 control stays
identical to fresh.

At layer 48, the J-lens readout also shifts slightly toward the full-context
ranking:

| State | Layer-48 J-lens top candidates |
| --- | --- |
| `full_context` | `priority`, `priorit`, `instead`, `precedence`, `prioritize` |
| `fresh_compacted` | `priority`, `priorit`, `prioritize`, `precedence`, `faster` |
| `alpha0_grafted_compacted` | `priority`, `priorit`, `prioritize`, `precedence`, `faster` |
| `grafted_compacted` | `priority`, `priorit`, `precedence`, `prioritize`, `faster` |

At layer 62, all states strongly express "first/before" ordering, and the
difference is mostly rank order rather than concept identity. This matches the
broader next-token control: late-layer J-lens readouts are more
continuation-like, so layer 62 should be treated cautiously.

## Concrete Example: The `rivermark` Span

The target contains the file path `src/rivermark/sort.py`. At the token
`river`, all states correctly put `river` at next-token argmax, so there is no
next-token rescue. But layer 62 shows a readable internal-state shift:

| State | Layer-62 J-lens top candidates at `river` |
| --- | --- |
| `full_context` | `mark`, `bank`, `marks`, `-mark`, `_mark` |
| `fresh_compacted` | `tests`, `/tests`, `mark`, `then`, `tests`, `/run` |
| `alpha0_grafted_compacted` | `tests`, `/tests`, `mark`, `then`, `tests`, `/run` |
| `grafted_compacted` | `mark`, `bank`, `tests`, `mask`, `/tests`, `/run` |

This is qualitatively suggestive, not decisive. The grafted state brings
`mark` and `bank` upward relative to fresh, which fits the scenario's
river-bank disambiguation. But file paths and subword tokens are noisy, and
the next-token distribution already handles the literal token. This example is
best used as an interpretability illustration, not as a behavioral result.

## What the J-Lens Adds

The Jacobian lens reads residual-stream activations by transporting them into a
final-layer vocabulary basis with an averaged Jacobian, then decoding through
the model's unembedding. It is a principled relative of the logit lens and
tuned lens: all three are vocabulary-space readouts of internal states, but the
J-lens is designed to surface concepts an activation is disposed to verbalize
across contexts, not merely the next token in this context.

For this project, the J-lens is useful because it can make the compaction
boundary visible. It can show that:

- old-context and fresh summary encodings can carry different readable state;
- a grafted compacted state can differ from fresh compacted under the same
  visible text;
- mid-layer readouts are often more distinct from ordinary next-token
  probabilities than very late-layer readouts.

But the J-lens does not directly label KV-cache entries, and it does not
replace behavioral evaluation. It reads residual-stream state after attention
and MLP computation. A good report should therefore say "the lens suggests"
unless the same pattern is confirmed by behavioral task metrics.

## Prior-Art Context

This work sits near two literatures.

First, it depends on standard transformer attention and KV caching. Vaswani et
al. introduced scaled dot-product attention and the query/key/value framing.
RoPE, introduced by Su et al., explains why moved keys require position-aware
handling. Recent KV-cache work is especially close: Li's "Models Take Notes at
Prefill" shows that prefill can write field-conditioned conclusions onto
downstream cache states, and that those states can be edited and composed. That
substantially constrains any novelty claim here. The distinctive ValueGraft
question is not "can KV caches be edited?" but whether write-time cache state
can preserve semantic continuity across conversation compaction and reduce the
practical damage of replacing old context with a summary.

Second, the readout method belongs to the vocabulary-lens family. The logit
lens applies the unembedding directly to intermediate states; the tuned lens
learns per-layer translators; the J-lens uses an averaged Jacobian transport.
Our use is modest: we use the lens to inspect candidate examples and generate
mechanistic hypotheses, then require behavioral or controlled intervention
tests before treating those hypotheses as results.

## Limitations

This corrected artifact fixes the missing-intervention comparison, but it is
still a single small probe.

- It uses one constructed coding-style scenario, not a broad benchmark.
- It uses a V-only graft with `alpha_V = 0.75`; it does not compare K-only,
  coupled KV, or independent `alpha_K`/`alpha_V` policies.
- The target is teacher-forced for measurement, so the sequence readout is a
  controlled diagnostic rather than a free-generation result.
- The summary was capped at 96 generated tokens and ends mid-fragment, though
  the key file/test facts appear earlier in the summary.
- J-lens readouts are residual-stream telemetry, not direct KV-cache labels.
- The layer-62 result is mixed and slightly farther from full context after
  grafting.
- There is no wrong-graft or shuffled-graft negative control in this artifact.

The right conclusion is therefore: this probe demonstrates that the corrected
three-state method can detect a real intervention effect under controlled
conditions. It does not establish generality or production impact.

## Next Steps

The next useful work is not more prose polish over this single example. It is
to run the same validated three-state design over multiple examples and tasks:

- add wrong-graft or shuffled-graft controls;
- predeclare the gap-closure metric before looking at examples;
- run V-only, K-only, coupled KV, and independent KV variants under the same
  visible text and same summary;
- apply the method to existing next-action trajectory examples where the
  behavioral target is already known;
- keep broad old-vs-fresh J-lens sweeps as diagnostic scouting, not as
  intervention evidence.

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
