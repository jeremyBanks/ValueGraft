# ValueGraft Intervention Probe Findings

Status: working report  
Date: 2026-07-07  
Artifacts:

- `outputs/qwen36_intervention_batch_summary.json`
- `outputs/qwen36_intervention_challenge_summary.json`
- local raw artifacts, ignored because they are larger than normal repo limits:
  `outputs/qwen36_intervention_batch_probe.json` and
  `outputs/qwen36_intervention_challenge_probe.json`

## Summary

We ran two post-boundary intervention probes with Qwen3.6-27B and the public
Neuronpedia/Anthropic Jacobian-lens weights.

The first probe used ordinary, information-rich summaries plus the last two
conversation messages. It shows a small but real-looking pattern: low and
moderate value grafts move J-lens readouts slightly toward full context, while
a shifted-value control is consistently worse. That is useful evidence that
the cache surgery is doing something position-sensitive and that alignment
matters. It is not a compelling human-readable semantic rescue. The visible
summary and tail already contain most answers, and many of the largest token
movements are formatting, punctuation, number, or continuation effects.

The second probe was deliberately stricter. It used sparse summaries that kept
the labels but omitted key facts, and it retained no tail. This creates large
full-vs-fresh gaps. Under that setup, V-only grafting did not recover the
omitted facts. Closures are tiny, next-token rescues are absent, and stronger
alphas introduce regressions. This is a useful falsification of an overly
strong story: with this model, lens setup, and V-only summary-token graft,
ValueGraft does not behave like a hidden fact-recovery channel for label-only
summaries.

The combined interpretation is narrower and better:

- V-only grafting can slightly steer compacted-state readouts toward the
  full-context state when the visible compacted prompt already contains enough
  relevant information.
- Alignment matters; injecting the same values at the wrong summary-token
  positions is reliably worse in the ordinary batch.
- V-only grafting alone does not rescue details that the compacted text has
  aggressively omitted.
- Higher alpha is not automatically better. The ordinary batch often prefers
  `alpha_V = 0.25` or `0.5`; alpha-one is more disruptive.

This does not invalidate the broader ValueGraft behavioral result. It does
constrain the mechanistic story we should tell from the J-lens side probe.

## What Was Tested

Both probes used the same intervention family: a **V-Graft** over aligned
summary tokens.

For each case we constructed:

- **Full context:** the original conversation plus the probe question.
- **Fresh compacted:** a compacted prompt with summary text plus any retained
  tail, then the same probe question.
- **Alpha-zero control:** the graft path with `alpha_V = 0`, which must match
  fresh compaction exactly.
- **Aligned grafted compacted:** the same visible text as fresh compacted, but
  with aligned write-time summary-token value-cache entries blended into the
  fresh compacted cache.
- **Shifted control:** the same value entries shifted to wrong summary-token
  positions before injection.
- **Alpha sweep:** additional constant alpha values.

Keys were not grafted in these probes:

```text
K_final = K_fresh
V_final = (1 - alpha_V) * V_fresh + alpha_V * V_write_time
```

This is not the whole ValueGraft design space. It is the V-only branch:
`alpha_K = 0`, `alpha_V` varied. That distinction matters because earlier
conceptual discussion left room for independent `alpha_K` and `alpha_V`.

The probe then teacher-forced a target answer token by token. At each target
position it recorded:

- the ordinary next-token top candidates before forcing the target token;
- J-lens top-k readouts after the forced target token exists in the residual
  stream;
- closure toward the full-context J-lens readout.

Closure is measured as:

```text
closure = distance(full_context, fresh_compacted)
        - distance(full_context, condition)
```

Distance is Jaccard distance between top-k readout token sets at the same
target position and layer. Positive closure means the condition is closer to
full context than fresh compaction. Negative closure means it is farther away.

## Validation

The validator confirms the two artifacts are internally consistent.

| Artifact | Cases | Forced target tokens | Aligned summary-token pairs | Layers | Errors |
| --- | ---: | ---: | ---: | --- | --- |
| ordinary batch | 10 | 156 | 1,820 | 16, 32, 48, 62 | none |
| sparse challenge | 9 | 142 | 485 | 16, 32, 48, 62 | none |

For every case, alpha-zero exactly matches fresh compaction on token IDs,
argmaxes, and J-lens readout rows. That is an important sanity check: any
observed graft effect is not coming from simply running through a different
capture path.

One implementation note: in the sparse challenge, the compacted prefix by
itself is only `system + summary note`, with no retained user turn. Qwen's chat
template refuses to render that prefix alone because it has no user query. The
actual measured prompt appends the probe user question before rendering, so it
is valid. The artifact records the prefix-only render failure as metadata and
does not use it in the measurement.

## Ordinary Batch

The ordinary batch used preauthored summaries from the prior write-time/fresh
sweep and retained the last two original messages. The summaries were
information-rich. Examples include permit numbers, Maple's meaning, Falcon vs
Raven, Patch 17 vs R3, Delta, Ghost/Ghost2, Dex, T-104/T-88, and amber/purple.

This batch is useful for checking whether value injection is coherent when the
visible compacted prompt already contains the answer. It is a poor test of
whether the graft can recover missing information, because little information
is actually missing from the prompt.

Mean focus-token closure:

| Condition | L16 | L32 | L48 | L62 | Focus argmax rescues / regressions / changes |
| --- | ---: | ---: | ---: | ---: | --- |
| `alpha_V = 0.25` | 0.0328 | 0.0231 | 0.0124 | 0.0066 | 0 / 0 / 0 |
| `alpha_V = 0.5` | 0.0433 | 0.0273 | 0.0090 | 0.0111 | 0 / 0 / 0 |
| `alpha_V = 0.75` | 0.0357 | 0.0042 | -0.0336 | 0.0004 | 2 / 0 / 2 |
| `alpha_V = 1.0` | -0.0005 | -0.0239 | -0.0709 | -0.0353 | 4 / 1 / 5 |
| shifted `alpha_V = 0.75` | -0.0998 | -0.1363 | -0.1527 | -0.1108 | 5 / 7 / 13 |

The best result in this table is not a single dramatic rescue. It is the
shape of the controls:

- Low and moderate alphas have small positive closure without changing focus
  argmaxes.
- `alpha_V = 0.75` still has positive closure at layer 16 but turns negative at
  layer 48.
- `alpha_V = 1.0` is worse, especially at layers 48 and 62.
- The shifted control is clearly bad: negative closure at every sampled layer
  and many more argmax disruptions.

That pattern supports a modest mechanistic claim: aligned value-state blending
has a measurable direction, and misaligned value injection is not an innocent
control. It does not support claiming that the J-lens has shown a robust
semantic fix.

### Example: Shifted Values Are Not Harmless

In the permit case, the target answer is:

```text
Use P-771 on the insurance form, not B-410.
```

At the final digit of `P-771`, full, fresh, and aligned graft all keep the
next-token argmax as `1`. The shifted control changes the argmax to ` on`.
Layer-62 readouts also become number-like under the shifted control:

| Path | Layer-62 J-lens readout after `P-771` |
| --- | --- |
| full context | `on`, `,`, `for`, `;`, `and` |
| fresh compacted | `on`, `,`, `for`, `;`, `on` |
| aligned V-graft | `on`, `,`, `for`, `on`, `;` |
| shifted control | `on`, `0`, `9`, `8`, `4`, `6`, `2`, `3` |

This is not a semantic win for aligned grafting; aligned and fresh are almost
the same here. The useful point is the negative control. Moving the same value
states to wrong summary-token positions creates a qualitatively different and
less coherent readout.

In the Maple case, the shifted control similarly drifts into unrelated
operational words around the `check-in` span:

| Path | Layer-16 J-lens readout near `check-in` |
| --- | --- |
| full context | `.`, `,`, `-`, `at`, `on` |
| fresh compacted | `,`, `.`, `-`, `via`, `at` |
| aligned V-graft | `,`, `.`, `-`, `with`, `at`, `on` |
| shifted control | `weekend`, `folks`, `online`, `onsite`, `overnight` |

Again, this is mainly a shifted-control result. It shows that the graft path is
not equivalent to arbitrary noise or smoothing. It does not show that aligned
V-Graft recovers a hidden fact.

## Sparse Challenge

The sparse challenge was designed after the ordinary batch looked too easy.
Each summary kept local labels but intentionally omitted the key relation:

- both permit numbers, but not which was current;
- Maple, but not that it was the library's Maple Room for storage/check-in;
- Falcon and Raven, but not accepted vs rejected;
- Patch 17 and R3, but not live vs stale;
- Delta, but not ferry route Delta 6 at 7:40;
- Ghost and Ghost2, but not dead vs alive;
- Dex/Makuhita/Castform, but not the trade relation;
- T-88 and T-104, but not stale vs active;
- amber and purple, but not safe vs forbidden.

No recent tail was retained. This made the fresh compacted prompt
intentionally under-informative. Baseline full-vs-fresh focus-token distances
were large: for example, layer-48 distances were 0.898 for Maple, 0.780 for
Falcon/Raven, 0.955 for Delta, and 0.919 for amber/purple.

Mean focus-token closure:

| Condition | L16 | L32 | L48 | L62 | Focus argmax rescues / regressions / changes |
| --- | ---: | ---: | ---: | ---: | --- |
| `alpha_V = 0.25` | -0.0021 | 0.0056 | 0.0098 | -0.0040 | 0 / 0 / 2 |
| `alpha_V = 0.5` | -0.0010 | 0.0071 | 0.0106 | -0.0216 | 0 / 2 / 4 |
| `alpha_V = 1.0` | -0.0096 | -0.0062 | 0.0082 | -0.0363 | 0 / 3 / 5 |
| shifted `alpha_V = 0.25` | -0.0016 | 0.0037 | -0.0012 | -0.0128 | 0 / 0 / 2 |

The sparse challenge gives a cleaner negative result:

- There are no focus-token argmax rescues.
- Closure is near zero.
- Aligned and shifted conditions are often too close to distinguish.
- Higher alphas add regressions without revealing omitted facts.

This is exactly the failure mode that matters for interpretation. The old
write-time value states may carry context-conditioned information, but a
V-only summary-token graft does not appear to be enough to reconstruct
omitted relations when the summary itself has collapsed them to a label list.

### Example: Falcon/Raven Does Not Recover

The sparse checkout summary names `Falcon` and `Raven`, but omits which branch
was accepted. The target answer is:

```text
Use Raven for rollback; Falcon was rejected because it drops subscription coupons.
```

At the `Raven` target token, the next-token argmax shows the problem:

| Path | Next-token argmax |
| --- | --- |
| full context | ` Raven` |
| fresh compacted | ` Falcon` |
| aligned V-graft | ` Falcon` |
| shifted control | ` Falcon` |

The J-lens readouts also do not recover the missing relation. At `Falcon`,
layer 48 under full context strongly reads rejected-status words:

| Path | Layer-48 J-lens readout at `Falcon` |
| --- | --- |
| full context | `rejected`, `because`, `failed`, `unacceptable`, `discarded` |
| fresh compacted | `was`, `is`, `used`, `reserved`, `intended`, `serves` |
| aligned V-graft | `was`, `is`, `used`, `reserved`, `intended`, `serves` |
| shifted control | `was`, `is`, `reserved`, `used`, `intended`, `serves` |

This is a strong caution. The old-context/full path has the semantic status,
but the V-only graft does not restore it in the sparse compacted prompt.

### Example: Maple Moves Slightly, But Not Enough

The sparse Maple summary names `Maple` but omits the Maple Room detail. At the
`le` token in `Maple`, layer 48 shows a tiny movement:

| Path | Layer-48 J-lens readout |
| --- | --- |
| full context | `refers`, `is`, `represents`, `referring`, `specifically` |
| fresh compacted | `is`, `was`, `designated`, `serves`, `location` |
| aligned V-graft | `is`, `was`, `designated`, `serves`, `location`, `refers` |
| shifted control | `is`, `was`, `designated`, `serves`, `location`, `tree`, `trees` |

The aligned graft nudges `refers` into the list and avoids the tree drift that
appears in the shifted control, but this is far below the standard for a
compelling semantic recovery. It does not produce the Maple Room, storage, or
volunteer check-in relation in the readout or next-token behavior.

## Interpretation

The two probes answer different questions.

The ordinary batch asks: when the compacted prompt is already fairly
informative, does value grafting behave coherently? Tentatively, yes. Small
alphas slightly close the full-vs-fresh readout gap, and shifted injection is
much worse. This is a good sanity check for the intervention machinery and the
alignment control.

The sparse challenge asks: can V-only grafting recover facts that a summary has
reduced to bare labels? In this setup, no. The full-vs-fresh gap becomes much
larger, but the graft does not close it in a meaningful way.

This suggests a more disciplined story for the J-lens work:

1. The old-vs-fresh J-lens sweep remains useful background. It shows that the
   same visible summary tokens can have very different context-conditioned
   readouts.
2. The ordinary intervention batch shows that aligned value grafts are not
   arbitrary: they differ from shifted grafts and have an alpha-dependent
   effect.
3. The sparse challenge prevents overclaiming. V-only grafting should not be
   described as a way to smuggle entire omitted facts through a terse label
   summary.

That is consistent with the broader ValueGraft framing as mitigation, not
magic recall. It may preserve or improve state around a reasonably good
summary. It does not eliminate the need for the summary to actually carry
important facts.

## Consequences For Next Experiments

The immediate report should not use these J-lens intervention probes as the
main proof that ValueGraft works. They are better used as:

- evidence that the intervention has measurable, alignment-sensitive internal
  effects;
- a warning that full replacement and high alpha can be harmful;
- a clean negative against the strongest hidden-fact-recovery interpretation;
- motivation for independent key/value experiments.

The most natural next branch is not more prose around V-only examples. It is a
controlled K/V axis experiment:

- V-only: `alpha_K = 0`, sweep `alpha_V`;
- K-only: sweep `alpha_K`, `alpha_V = 0`, with correct re-rotation;
- coupled KV: `alpha_K = alpha_V`;
- independent KV: small grid or tuned low-rank profile over both.

The sparse challenge is especially useful for that next branch. If key
grafting matters for retrieval/addressing, the label-only summaries are where
it should have a chance to show up. If K-only and KV variants also fail there,
we learn that omitted relations are not practically recoverable from these
summary-token states under this setup.

We should also keep summary quality as an explicit axis. The sparse challenge
is intentionally under-specified; a production system should not generate
summaries that omit the active/stale mapping for critical labels. A realistic
claim for ValueGraft is therefore:

> Given a summary that preserves the relevant facts in text, can write-time
> cache state reduce reinterpretation error around those facts?

That is a smaller claim than "recover facts from a bad summary," but it is the
claim the current evidence supports.

## References

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
