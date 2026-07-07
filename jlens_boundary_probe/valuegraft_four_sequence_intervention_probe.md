# ValueGraft Under the Lens

Status: standalone report draft
Date: 2026-07-07
Primary artifact: `outputs/qwen36_boundary_three_state_probe.json`
Validator: `validate_three_state_probe.py`

## Summary

ValueGraft is a cache-state intervention for conversation compaction. A model
normally continues from compacted text by freshly encoding the summary and
recent tail. ValueGraft instead keeps the same visible compacted text while
blending selected key/value cache entries written when the summary was produced
under the original long context.

This report studies one small coding-style probe with Qwen3.6-27B and a
Jacobian-lens readout. The target continuation is:

```text
Update src/rivermark/sort.py so wet driftwood is inspected first, then run tests/test_sort.py.
```

The most readable result is at the token ` inspected`. Full context predicts
` inspected`; fresh compaction predicts ` priorit`; aligned ValueGraft at
`alpha_V = 0.75` returns the argmax to ` inspected`. The alpha-zero control
matches fresh exactly, so the cache plumbing itself is not creating the
change.

The richer control run adds an alpha sweep and a shifted-value negative
control. The aligned `alpha_V = 0.75` run improves the layer-48 lens distance
to full context. The shifted control, which injects the same old value states
at the wrong summary-token positions, worsens the layer-48 distance even though
it also changes one next-token argmax. That is a useful sign: alignment matters
for the internal readout, and next-token rescues alone are too weak to carry
the interpretation.

## Intervention

For each summary-token position, the probe has two cache traces:

- a fresh compacted trace, produced from the visible compacted context;
- a write-time trace, produced when the same summary text was written after
  the original context.

This artifact tests a V-only graft:

```text
K_final = K_fresh
V_final = (1 - alpha_V) * V_fresh + alpha_V * V_write_time
```

The implementation aligns 96 summary-token positions and changes value entries
in 16 value-cache layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51,
55, 59, and 63. Fresh keys and all non-value cache state are preserved.

The main run uses `alpha_V = 0.75`. The control run also records
`alpha_V = 0`, `0.25`, `0.5`, and `1.0`, plus a shifted control where the old
summary-token values are cyclically shifted by one token before injection.

## Probe Design

The compacted scenario concerns a package named `rivermark`. The important
facts are that `bank` means river bank, the relevant file is
`src/rivermark/sort.py`, wet driftwood is being prioritized incorrectly, and
the relevant test is `tests/test_sort.py`.

The probe question is:

```text
Continue the task. What exact file should be changed next, and what test should be run? Answer in one sentence.
```

The target continuation is teacher-forced one token at a time. Before each
forced token, the probe records:

- the model's next-token top candidates;
- whether the next-token argmax matches full context;
- J-lens top candidates at layers 16, 32, 48, and 62.

The comparison conditions are:

| Condition | Visible text | Cache state |
| --- | --- | --- |
| Full context | Original conversation plus probe question | Normal full-context cache |
| Fresh compacted | Summary plus recent tail plus probe question | Freshly encoded compacted cache |
| Alpha-zero control | Same as fresh compacted | Graft path with `alpha_V = 0` |
| Aligned ValueGraft | Same as fresh compacted | Aligned write-time values blended into summary-token positions |
| Shifted control | Same as fresh compacted | Write-time values shifted to the wrong summary-token positions |

## Results

The validator confirms that all conditions use the same 23 forced target
tokens, the alpha-zero control matches fresh exactly on token IDs and scores,
and the grafted paths use 96 aligned summary-token pairs.

Layer closure below is the improvement in mean Jaccard distance to full
context, using J-lens top-k token sets. Positive values mean closer to full
context than fresh compaction at that layer. Negative values mean farther.

| Condition | Argmax rescues | Argmax regressions | Changed argmax | Layer-48 closure | Layer-62 closure |
| --- | ---: | ---: | ---: | ---: | ---: |
| `alpha_V = 0` | 0 | 0 | 0 | 0.0000 | 0.0000 |
| `alpha_V = 0.25` | 0 | 0 | 0 | 0.0019 | -0.0097 |
| `alpha_V = 0.5` | 0 | 0 | 0 | 0.0129 | -0.0193 |
| `alpha_V = 0.75` | 1 | 0 | 1 | 0.0337 | -0.0146 |
| `alpha_V = 1.0` | 2 | 0 | 2 | -0.0184 | -0.0265 |
| Shifted `alpha_V = 0.75` | 1 | 0 | 1 | -0.0903 | -0.0560 |

The best overall balance in this small probe is `alpha_V = 0.75`. It produces
one next-token rescue and the strongest layer-48 closure. Full replacement
(`alpha_V = 1.0`) rescues two argmax positions, but its layer-48 and layer-62
readouts move farther from full context. The shifted control also rescues one
argmax position, but its readouts are much farther from full context. That
combination argues for reading the result through both next-token behavior and
internal readout, not either one alone.

For the primary aligned `alpha_V = 0.75` condition, all sampled layers are:

| Layer | Full vs fresh | Full vs aligned graft | Fresh vs aligned graft | Closure |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.1488 | 0.1430 | 0.0657 | 0.0058 |
| 32 | 0.1899 | 0.1788 | 0.0676 | 0.0111 |
| 48 | 0.3882 | 0.3546 | 0.1838 | 0.0337 |
| 62 | 0.3359 | 0.3506 | 0.1507 | -0.0146 |

Layers 16, 32, and 48 move closer to full context. Layer 62 moves slightly
farther away; this is consistent with late-layer readouts being more tied to
immediate continuation pressure.

## Token-Level Comparisons

### ` inspected`

At target index 13, the forced token is ` inspected`.

| Condition | Argmax before forcing | Top candidates |
| --- | --- | --- |
| Full context | ` inspected` | `inspected`, `priorit`, `given`, `treated` |
| Fresh compacted | ` priorit` | `priorit`, `inspected`, `given`, `assigned` |
| Alpha-zero control | ` priorit` | `priorit`, `inspected`, `given`, `assigned` |
| `alpha_V = 0.25` | ` priorit` | `priorit`, `inspected`, `given`, `assigned` |
| `alpha_V = 0.5` | ` priorit` | `priorit`, `inspected`, `given`, `assigned` |
| Aligned `alpha_V = 0.75` | ` inspected` | `inspected`, `priorit`, `given`, `assigned` |
| `alpha_V = 1.0` | ` inspected` | `inspected`, `priorit`, `given`, `treated` |
| Shifted `alpha_V = 0.75` | ` priorit` | `priorit`, `inspected`, `given`, `treated` |

This is the cleanest local effect. The aligned graft changes the next-token
argmax in the same direction as full context; the shifted control does not.

### ` is`

At target index 12, the forced token is ` is`.

| Condition | Argmax before forcing | Top candidates |
| --- | --- | --- |
| Full context | ` is` | `is`, `has`, `on`, `gets` |
| Fresh compacted | ` has` | `has`, `is`, `on`, `receives` |
| Alpha-zero control | ` has` | `has`, `is`, `on`, `receives` |
| Aligned `alpha_V = 0.75` | ` has` | `has`, `is`, `on`, `receives` |
| `alpha_V = 1.0` | ` is` | `is`, `has`, `on`, `receives` |
| Shifted `alpha_V = 0.75` | ` is` | `is`, `has`, `on`, `gets` |

This row is a warning against using argmax rescues by themselves. The shifted
control recovers the full-context argmax here, but the shifted condition is
worse by the J-lens distance metrics. A local token-level win can come from a
badly aligned perturbation.

## Interpretation

This probe shows that aligned write-time value states can change downstream
prediction and residual-stream readouts under identical visible compacted
text. The alpha sweep suggests a non-monotonic pattern: moderate blending looks
better than either no graft or full replacement on the layer-48 readout. The
shifted control makes the result more informative by showing that injecting old
values at the wrong positions is not equivalent to aligned ValueGraft.

The result is still a single constructed example. It is useful because it
connects the cache intervention to an inspectable internal readout and a
token-level prediction change. Broader task metrics remain the main evidence
for whether the technique improves real coding work.

## How the J-Lens Is Used

The Jacobian lens maps residual-stream activations into a vocabulary basis
using an averaged Jacobian transport, then decodes them with the model's
unembedding. Here it is used as an inspection tool: it helps compare whether
different cache states make the same visible continuation look more or less
like the full-context state.

The lens does not label individual KV-cache vectors. It reads the
residual-stream state produced after the model uses the cache. That is exactly
why this probe pairs lens metrics with next-token candidates and alpha/shifted
controls.

## Scope

This artifact studies one short coding-style setup, one model, one generated
summary, one target continuation, and V-only grafting. It is a measurement
design and qualitative mechanistic probe. It should be expanded across more
tasks, summaries, models, and graft policies before being treated as an effect
size estimate.

The most important next additions are:

- run the same alpha/shifted-control design on several existing trajectory
  prediction examples;
- add a wrong-conversation graft control;
- test K-only and independent K/V policies under the same visible text;
- connect lens movement to the existing behavioral scoring pipeline.

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
