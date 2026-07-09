# ValueGraft Under the Lens

Status: standalone report draft Date: 2026-07-07 Data artifact:
`outputs/qwen36_boundary_three_state_probe.json` Validator:
`validate_three_state_probe.py`

## Overview

ValueGraft is a cache-state intervention for conversation compaction. In an
ordinary compacted conversation, the model receives a summary plus recent tail
and freshly encodes that text. In a ValueGraft-style compacted conversation, the
visible text is still the summary plus recent tail, but selected cache entries
are blended from the state that was written when the summary was generated under
the original long context.

This report asks a narrow mechanistic question: among compacted variants with
identical visible text, does a value-state graft move the model's internal
readout toward the full-context run?

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
  state and rescues the next-token argmax at `inspected`;
- `alpha_V = 1.0` rescues more local argmaxes but worsens the internal readout
  distance;
- the shifted-value control also rescues one local argmax, but its J-lens
  readouts move much farther from full context.

Next-token behavior is useful, but too thin by itself. The lens readout gives a
second view: not just "what token came next?" but "what concept-neighborhood is
active at this position?"

As a qualitative demonstration, however, this artifact is weaker than we would
want. The compacted summary and retained tail already state almost all of the
answer, so many token-level readouts look nearly identical across conditions.
The result is useful as a cache-intervention sanity check and a warning about
alignment controls; it is not a strong showcase of visible semantic recovery.

## The Compaction Problem

A summary can preserve words without preserving the state that those words had
when they were written.

In a long conversation, the model may resolve local meanings: a name, a file, a
disambiguated sense of a word, an object that needs to be changed, or a
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

- `V_fresh`: the value state produced by freshly encoding the compacted summary;
- `V_write_time`: the value state written when the summary was produced under
  the original context.

This artifact tests a V-only graft:

```text
K_final = K_fresh
V_final = (1 - alpha_V) * V_fresh + alpha_V * V_write_time
```

The implementation aligns 96 summary-token positions and changes value entries
in 16 value-cache layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55,
59, and 63. Fresh keys and non-value state are left as in the compacted run.

That detail matters. This is not a hidden full-context replay. The visible text,
fresh positions, fresh keys, and non-value state remain the compacted run. The
intervention is only a value-state blend at aligned summary-token positions.

The run records:

| Condition          | Visible text                                 | Cache state                                              |
| ------------------ | -------------------------------------------- | -------------------------------------------------------- |
| Full context       | Original conversation plus probe question    | Normal full-context cache                                |
| Fresh compacted    | Summary plus recent tail plus probe question | Fresh compacted cache                                    |
| Alpha-zero control | Same as fresh compacted                      | Graft path with `alpha_V = 0`                            |
| Aligned ValueGraft | Same as fresh compacted                      | Aligned write-time values blended into fresh values      |
| Shifted control    | Same as fresh compacted                      | Same old values shifted to wrong summary-token positions |

The alpha sweep records `alpha_V = 0`, `0.25`, `0.5`, `0.75`, and `1.0`. The
artifact filename contains an older "three-state" label because the first
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

The lens is still only telemetry. It does not label individual KV vectors and it
does not replace task metrics. Its value here is that it makes the cache
intervention inspectable at the positions where the continuation is being
forced.

## Scenario

The compacted task state used by the probe is the exact generated summary below.
The final line ends mid-path in the artifact; that truncation is part of the
compacted state being tested.

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

There are two different "cuts" in this setup:

- Conversation compaction: the fresh compacted prompt replaces the first
  user/assistant task exchange with the generated summary, then keeps the last
  user message and last assistant state note as recent tail.
- Summary length cap: summary generation was capped at 96 new tokens, which is
  why the generated summary itself ends after the fragment `src` with an
  unmatched opening backtick.

The concrete contexts are:

| Condition          | Prompt before the probe question                                                                            | Token count before target |
| ------------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------: |
| Full context       | System message plus all four original task messages                                                         |                       221 |
| Fresh compacted    | System message, generated summary in a context note, last user task message, last assistant state note      |                       261 |
| Aligned ValueGraft | Same visible text as fresh compacted, but summary-token value states blended from write time                |                       261 |
| Shifted control    | Same visible text as fresh compacted, but old summary-token values shifted to wrong summary-token positions |                       261 |

The write-time value source is not another visible prompt condition. It is the
cache state produced when the model generated the 96-token summary after the
original conversation and summary request. Those write-time summary-token values
are what the graft blends into the fresh compacted cache.

This matters for interpretation. The retained tail already says the exact file
and test, and the summary repeats the bug and requirement. That makes this a
conservative state-manipulation probe, but a poor qualitative demo: the visible
text leaves little for the graft to recover in a way that jumps out from token
tables.

The probe question is:

```text
Continue the task. What exact file should be changed next, and what test should be run? Answer in one sentence.
```

The probe uses the following intended continuation. It is teacher-forced one
token at a time so every condition can be inspected at identical token
positions; this is a measurement setup, not a claim that the model would freely
generate the whole sentence.

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

Layer closure is measured as the improvement in mean Jaccard distance between a
condition's J-lens top-k set and the full-context top-k set:

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

| Condition                | Argmax rescues | Argmax regressions | Changed argmax | Layer-48 closure | Layer-62 closure |
| ------------------------ | -------------: | -----------------: | -------------: | ---------------: | ---------------: |
| `alpha_V = 0`            |              0 |                  0 |              0 |           0.0000 |           0.0000 |
| `alpha_V = 0.25`         |              0 |                  0 |              0 |           0.0019 |          -0.0097 |
| `alpha_V = 0.5`          |              0 |                  0 |              0 |           0.0129 |          -0.0193 |
| `alpha_V = 0.75`         |              1 |                  0 |              1 |           0.0337 |          -0.0146 |
| `alpha_V = 1.0`          |              2 |                  0 |              2 |          -0.0184 |          -0.0265 |
| Shifted `alpha_V = 0.75` |              1 |                  0 |              1 |          -0.0903 |          -0.0560 |

The best balance in this probe is aligned `alpha_V = 0.75`. It produces one
next-token rescue and the strongest layer-48 closure. Full value replacement
gets two argmax rescues, but moves the internal readout farther from full
context at layers 48 and 62. The shifted control gets one argmax rescue, but its
layer-48 closure is strongly negative.

For aligned `alpha_V = 0.75`, the sampled layers are:

| Layer | Full vs fresh | Full vs aligned graft | Fresh vs aligned graft | Closure |
| ----: | ------------: | --------------------: | ---------------------: | ------: |
|    16 |        0.1488 |                0.1430 |                 0.0657 |  0.0058 |
|    32 |        0.1899 |                0.1788 |                 0.0676 |  0.0111 |
|    48 |        0.3882 |                0.3546 |                 0.1838 |  0.0337 |
|    62 |        0.3359 |                0.3506 |                 0.1507 | -0.0146 |

The readout is not uniformly positive. Layers 16, 32, and 48 move closer to full
context. Layer 62 moves slightly farther away. That is consistent with
late-layer lens readouts being more tied to immediate continuation pressure.

## Qualitative Readouts

The qualitative readouts in this artifact are mostly underwhelming. That is an
important result about the probe design, not a prose problem to hide.

The largest closure rows in this 23-token target are mostly path, punctuation,
or formatting-like positions. For example, the biggest layer-48 closure is on
the `.py` token in `src/rivermark/sort.py`; aligned grafting exactly matches the
full-context J-lens top-k set there, but the token is not a semantic hinge of
the task. The more human-readable rows, such as `river`, `wet`, `inspected`,
`is`, and `wood`, usually differ by small rank shifts inside nearly the same
neighborhood.

The clearest behavioral row is `inspected`:

| Condition                | Argmax before forcing | Top candidates                                                    |
| ------------------------ | --------------------- | ----------------------------------------------------------------- |
| Full context             | `inspected`           | `inspected`, `priorit`, `given`, `treated`, `sorted`, `assigned`  |
| Fresh compacted          | `priorit`             | `priorit`, `inspected`, `given`, `assigned`, `treated`, `sorted`  |
| Alpha-zero control       | `priorit`             | `priorit`, `inspected`, `given`, `assigned`, `treated`, `sorted`  |
| Aligned `alpha_V = 0.75` | `inspected`           | `inspected`, `priorit`, `given`, `assigned`, `treated`, `sorted`  |
| `alpha_V = 1.0`          | `inspected`           | `inspected`, `priorit`, `given`, `treated`, `assigned`, `sorted`  |
| Shifted `alpha_V = 0.75` | `priorit`             | `priorit`, `inspected`, `given`, `treated`, `sorted`, `processed` |

That is a real next-token rescue, but the corresponding J-lens row is modest.
The J-lens rows below are ASCII-filtered display excerpts; the raw artifact
keeps the complete ranked top-k lists, including non-English vocabulary items
and duplicate subword variants.

| Condition                | Layer-48 J-lens readout excerpt at `inspected`               |
| ------------------------ | ------------------------------------------------------------ |
| Full context             | `priority`, `priorit`, `instead`, `precedence`, `prioritize` |
| Fresh compacted          | `priority`, `priorit`, `prioritize`, `precedence`, `faster`  |
| Alpha-zero control       | `priority`, `priorit`, `prioritize`, `precedence`, `faster`  |
| Aligned `alpha_V = 0.75` | `priority`, `priorit`, `precedence`, `prioritize`, `faster`  |
| `alpha_V = 1.0`          | `priority`, `priorit`, `precedence`, `prioritize`            |
| Shifted `alpha_V = 0.75` | `priority`, `priorit`, `before`, `prioritize`, `precedence`  |

The aligned graft moves `precedence` above `prioritize`, closer to the
full-context ordering, but the table is not visually dramatic. It should be read
as one piece of the aggregate layer-48 closure, not as a standalone mechanistic
demonstration.

The `river` row is the best example of the J-lens adding something beyond the
next-token table, because every condition predicts the next path token:

| Condition                | Layer-62 J-lens readout excerpt at `river`                  |
| ------------------------ | ----------------------------------------------------------- |
| Full context             | `mark`, `mark`, `bank`, `marks`, `-mark`, `_mark`           |
| Fresh compacted          | `tests`, `/tests`, `mark`, `then`, `tests`, `/run`          |
| Alpha-zero control       | `tests`, `/tests`, `mark`, `then`, `tests`, `/run`          |
| Aligned `alpha_V = 0.75` | `mark`, `bank`, `tests`, `mask`, `/tests`, `/run`           |
| `alpha_V = 1.0`          | `mark`, `bank`, `wood`, `mark`, `marks`, `markdown`, `mask` |
| Shifted `alpha_V = 0.75` | `mark`, `bank`, `_mark`, `-mark`, `marks`, `mark`           |

Even this row is not a clean win for aligned grafting, because the shifted
control also surfaces `mark` and `bank`. It is useful mainly as a warning:
individual top-k rows can look meaningful even when the aggregate shifted
control is worse.

The right conclusion is therefore narrow. This probe shows that the graft path
is active, alpha-zero is a valid plumbing control, and aligned values behave
differently from shifted values. It does not provide the kind of compelling
human-readable example we were hoping for. A better qualitative probe should use
tokens like `B-410`, `Maple`, `Falcon`, `Patch 17`, `Ghost2`, or `Dex` from the
broader write-time/fresh sweep, then rerun them with the full three-state
intervention design. Those examples have larger semantic separation, but the
existing sweep for them is only two-state telemetry, not direct evidence that
grafting closes the gap.

## What the Controls Show

The alpha sweep and shifted control separate aligned state transfer from a
generic perturbation.

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

| Condition                | Layer-16 closure | Layer-32 closure | Layer-48 closure | Layer-62 closure |
| ------------------------ | ---------------: | ---------------: | ---------------: | ---------------: |
| Aligned `alpha_V = 0.75` |           0.0058 |           0.0111 |           0.0337 |          -0.0146 |
| Shifted `alpha_V = 0.75` |          -0.0957 |          -0.0836 |          -0.0903 |          -0.0560 |

The shifted condition can still look good on selected rows. That is why the
aggregate table matters. Across the target sequence, shifted values move the
internal readout farther from full context at every sampled layer.

## Interpretation

The safest interpretation is:

1. The intervention is active. Aligned value grafting changes downstream
   prediction and internal readouts while alpha-zero remains identical to fresh.
2. The effect is alignment-sensitive. Shifting old values to the wrong
   summary-token positions produces a much worse internal match to full context.
3. Moderate blending is better than full replacement in this example.
   `alpha_V = 1.0` can produce local next-token wins while degrading the J-lens
   readout.
4. The J-lens can add information beyond next-token candidates, but this
   particular probe is not a strong qualitative demonstration. The visible
   examples are mostly subtle rank shifts or path-token effects.
5. The result is still narrow. It is one constructed probe, not a task-level
   success result, not an effect-size estimate, and not yet the right showcase
   example.

This makes the intervention inspectable in principle: the same visible compacted
text can produce different internal readouts depending on whether aligned
write-time values are grafted. Behavioral benchmarks still decide whether the
method helps in practice. A better lens demonstration should be built from cases
where compaction preserves a label but loses its role, rather than from a short
target whose answer is already explicit in the summary and tail.

## Scope and Next Work

This artifact studies one model, one generated summary, one target continuation,
and V-only grafting. The target is teacher-forced so every condition can be
inspected at the same positions. The J-lens readout is residual-stream
telemetry, not direct access to individual KV vectors.

The natural next steps are:

- run the same alpha/shifted-control design on stronger qualitative examples
  from the write-time/fresh sweep, especially private labels and stale
  identifiers;
- add a wrong-conversation graft control;
- test K-only and independent K/V policies under the same visible text;
- connect lens movement to behavioral task metrics;
- choose examples where the key concept is not already explicit in the summary
  text, so the lens has a harder job.

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, L., & Polosukhin, I. (2017).
  [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
  arXiv:1706.03762.
- Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
  [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
  arXiv:2104.09864.
- Li, B. (2026).
  [Models Take Notes at Prefill: KV Cache Can Be Editable and Composable](https://arxiv.org/abs/2606.17107).
  arXiv:2606.17107.
- nostalgebraist. (2020).
  [Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).
  LessWrong.
- Belrose, N., Furman, Z., Smith, L., Halawi, D., Ostrovsky, I., McKinney, L.,
  Biderman, S., & Steinhardt, J. (2023).
  [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112).
  arXiv:2303.08112.
- Gurnee, W., Sofroniew, N., Pearce, A., Piotrowski, M., Kauvar, I., Chen, R.,
  Soligo, A., Bogdan, P., Ong, E., Wang, R., Thompson, B., Abrahams, D.,
  Kantamneni, S., Ameisen, E., Batson, J., & Lindsey, J. (2026).
  [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/).
  Transformer Circuits Thread.
- Anthropic. (2026).
  [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens).
  Reference implementation for the Jacobian lens.
