# ValueGraft: Mitigating Conversation Compaction with Write-Time Value State

**Status:** focused second draft. The broader synthesis remains preserved in
`valuegraft-paper-draft.md`.  
**Date:** 2026-07-05  
**Authors:** Jeremy Banks; Anthropic Claude Fable 5; OpenAI GPT-5.5

## Abstract

Long-running LLM agents often compact their conversation history by replacing
old turns with a summary and re-encoding the shortened transcript. This keeps
visible text but discards the attention state written when the model originally
interpreted that text in full context. We test whether preserving part of that
write-time state can reduce the impact of compaction. We use **ValueGraft** as
the name for a family of training-free compaction-state interventions and study
two variants on Qwen3-4B-Instruct-2507 and Qwen3-30B-A3B-Instruct-2507:
**ValueGraft-Pack**, which carries a summary's write-time cache entries into a
packed context, and **ValueGraft-Blend**, which freshly encodes the compacted
context but blends aligned old cached value tensors into the fresh cache. The
interventions do not recover evicted factual recall. They produce two narrower
mitigation effects: ValueGraft-Pack reduces fabrication on unknowable questions
in agentic compaction frames, and tuned ValueGraft-Blend recovers a small but
consistent fraction of continuation and coding-agent next-action likelihood
lost to compaction. We use full-context vs text-compacted performance to
normalize effect sizes. On 75 OpenHands SWE-Gym traces, ValueGraft-Blend
improves true next-action likelihood by +0.0156 nats/token, about 10% of that
gap. The evidence supports a limited mitigation claim: write-time value state
can preserve some behavioral continuity across compaction boundaries, but it is
not a general memory-recovery mechanism.

## 1. Introduction

Conversation compaction is a practical necessity for long-running LLM agents.
When the context window fills, systems commonly replace old history with a text
summary and continue from `summary + recent tail`. The routine is cheap,
auditable, and compatible with ordinary stateless API calls.

But the operation changes more than the visible transcript. The retained tail
and the new summary are encoded after the old context has been removed. Any
context-conditioned computation previously written into the model's cached
attention state is discarded. A phrase like "the second approach" may still be
present in the tail, but its original interpretation was computed when the
earlier discussion was attendable.

We ask whether carrying forward a small amount of write-time attention state can
reduce the impact of text-only compaction.

We use **ValueGraft** as the umbrella name for two closely related
interventions.

**ValueGraft-Pack** generates a summary while the full conversation is still
available, then carries the summary's write-time key/value entries into a
compact packed context. Keys are re-rotated to their packed positions; values
are unchanged.

**ValueGraft-Blend** builds the ordinary compacted context and keeps its fresh
keys, but blends old value tensors into exact-aligned summary and tail
positions:

```text
V_final = (1 - alpha) * V_fresh + alpha * V_old
```

Here `V_old` is the value tensor written when the matching token was processed
with the full context still available. The implemented arms use literal token
alignment only; they do not perform fuzzy semantic retrieval.

Across the current evidence, these interventions do not restore facts that were
only present in the evicted history. Their measurable benefits are concentrated
in two failure modes: fabrication after compaction and loss of continuation or
next-action likelihood.

## 2. Terminology and Outcomes

We use **cached attention state** for the per-token key and value tensors stored
during transformer prefill or generation. We use **value state** for the value
side of that state. These are not scalar attention weights. The standard object
is usually called a KV cache, but this work is not about caching as a speed
optimization. It is about which attention state is present after compaction.

We distinguish three outcomes:

- **Recall:** answering questions whose evidence was in the evicted context.
- **Honesty:** admitting missing information rather than fabricating.
- **Continuity:** assigning higher likelihood to the true next continuation or
  next agent action.

The evidence so far supports honesty and continuity effects, not recall
recovery.

## 3. Related Work

Several nearby literatures already establish that cached attention state can be
useful beyond speed. Li (2026) argues that prefill writes conclusions into
downstream cached state and that those cached blocks can be edited or composed.
Zweiger et al. (2026) study latent-space KV compaction directly. KV reuse
systems such as KVLink (Yang et al., 2025), CacheBlend (Yao et al., 2025), and
SamKV (Cao et al., 2025) reuse or mix cache state for independently encoded
chunks. Learned compression methods such as gist tokens (Mu et al., 2023),
AutoCompressor (Chevalier et al., 2023), ICAE (Ge et al., 2024), Activation
Beacon (Zhang et al., 2024), Compressed Context Memory (Kim et al., 2024), and
Cartridges (Eyuboglu et al., 2025) explore non-textual ways to carry context.

The narrower setting here is conversation compaction: old dialogue is replaced
by a generated text summary and the system continues from a compacted transcript.
Text compression and memory systems are the baseline condition. The intervention
is to preserve write-time state associated with the summary or retained tail and
compare it to re-encoding the same visible compacted text from scratch.

Hosted provider APIs now expose related product surfaces, including opaque
compaction, reasoning, thought-signature, context-caching, and session
continuation artifacts (OpenAI, n.d.; Anthropic, n.d.; Google AI for
Developers, n.d.). Those interfaces show that production APIs can carry
non-textual continuation state, although public documentation does not show
whether providers use a ValueGraft-like mechanism. The evidence here is a
white-box open-model measurement of one such mechanism.

## 4. Methods

### 4.1 Shared Compaction Setup

Each example is first rendered with the model's chat template in a canonical
non-final form, so token spans can be compared across arms without Qwen's final
assistant-message template instability. We keep the first four tokens as
attention sinks. The retained tail begins at a message boundary near the final
quarter of the tokenized conversation; the material between the sinks and that
tail is the evicted region.

A summary is generated greedily by the same subject model while the full
pre-compaction conversation is still available. The summary request is appended
as a normal user turn, the model is prefixed through the full conversation plus
that request, and the generated summary tokens are decoded from the resulting
continuation. The cache snapshot from this generation run is saved. It contains
the key/value tensors written for the original conversation, the summary
request, and the summary tokens as they were generated under full context.

The production text-compaction baseline then constructs a new transcript:
system message, assistant context note containing the summary, and the retained
tail messages. That transcript is freshly encoded from scratch. The
interventions differ only in which cached attention state is carried across
this boundary.

### 4.2 Arms and Controls

The main evaluated arms are:

- **A:** full context, no compaction.
- **B:** production-style text compaction, `summary + tail`, freshly encoded
  from the shortened transcript.
- **FreshPack:** the first four sink tokens plus the exact generated summary
  token ids, packed contiguously and freshly encoded.
- **ValueGraft-Pack:** the same packed token sequence as FreshPack, but using the
  summary's write-time cache entries.
- **ValueGraft-Blend:** the normal B transcript with fresh keys and blended old
  value tensors at exact-aligned summary and tail positions.
- **Negative controls:** shuffled-value and wrong-conversation grafts.

ValueGraft-Pack vs FreshPack isolates the effect of write-time summary encoding
in a packed layout: identical tokens, identical packed positions, different
cache state. ValueGraft-Blend vs B tests whether old value payloads improve an
otherwise ordinary compacted transcript.

### 4.3 ValueGraft-Pack Construction

ValueGraft-Pack extracts the summary token span from the saved
summary-generation cache. For Qwen3 in this stack, keys are stored after RoPE
rotation and values are unrotated. To make the retained summary cache contiguous
and prefix-shaped, we move the summary span immediately after the four sink
tokens. The summary keys are re-rotated by the positional offset between their
write-time positions and their packed positions; the values are copied
unchanged. The resulting cache contains only the sinks and summary entries,
with the cache offset set to the end of the packed summary.

FreshPack is the matched text-only control for this operation. It uses the
same sink tokens and the same summary token ids at the same packed positions,
but obtains their key/value tensors by an ordinary fresh prefill. Any
ValueGraft-Pack vs FreshPack difference is therefore not due to summary wording,
token count, or packed position layout.

### 4.4 ValueGraft-Blend Construction

ValueGraft-Blend starts from the production text-compaction baseline B. We first
freshly prefill the compacted transcript, preserving its ordinary contiguous
positions and its fresh keys. We then align tokens from the compacted transcript
to tokens from the old full-context summary-generation run.

Alignment is exact-token matching, not semantic retrieval. The summary region
and tail region are matched separately because B places the summary before the
tail, while the old generation run places the summary after the original
conversation and summary request. Matching blocks shorter than eight tokens are
discarded to avoid common-token coincidences, and special tokens and sink
positions are excluded. For each accepted pair `(new_pos, old_pos)`, each
layer's value tensor is replaced by a linear blend:

```text
V_final[layer, new_pos] =
    (1 - alpha) * V_fresh[layer, new_pos]
  + alpha       * V_old[layer, old_pos]
```

The reported ValueGraft-Blend arm is the post-prefill version: blending happens
after B's compacted transcript has been encoded. Earlier exploratory arms also
tried interleaving the blend during prefill, but the current headline comparison
uses fresh keys and post-prefill blended values. The 4B setting used a
mid-layer-band gate with alpha=0.25; the 30B setting used a global alpha=0.75,
both selected on validation continuation likelihood before holdout evaluation.

### 4.5 Validation

Because cache surgery is sensitive to position, template, and kernel details,
every reported configuration had to pass identity tests before its results
counted. The ladder verifies cache
serialization, null surgery, ValueGraft-Blend alpha=0 equivalence to B,
old-context-equals-new-context equivalence to A, tokenization stability, and
key re-rotation for packed ValueGraft-Pack. Runtime traps found during the work
include Qwen chat-template instability, batched-vs-stepwise logit differences,
and sequence-length-dependent 4-bit kernel behavior.

Negative controls check whether gains can be explained by generic smoothing or
odd cache perturbations. Shuffled-value grafts use the correct conversation's
old values but attach them to the wrong aligned positions. Wrong-conversation
grafts use old values from another conversation. The wrong-summary pack control
uses packed summary state from another conversation. These controls test whether
an effect survives after content/state alignment is broken.

### 4.6 Data and Scoring

The evidence comes from four sources:

- **Synthetic planted conversations:** 12 approximately 9K-token conversations
  with planted referents, senses, stances, ruled-out approaches, and evicted
  facts.
- **Natural/free-form conversations:** 8 conversations with held-out
  continuations.
- **LongMemEval-S:** standard benchmark material restructured so the evidence
  session is evicted. Early runs cover n=48 at 4B and n=36 at 30B; a later
  30B-bf16 aggregate covers n=320.
- **SWE-Gym/OpenHands traces:** 75 real coding-agent trajectories scored by
  teacher-forced likelihood of the true next assistant action.

Probe-style tasks append a user question to each arm and greedily generate an
answer at temperature 0. Continuation and coding-trajectory tasks teacher-force
the held-out continuation or next assistant action and report mean per-token
log likelihood. Where possible, results are paired by conversation and reported
with wins, bootstrap confidence intervals, and gap closure `(arm - B) / (A -
B)`. Local 4B results are 4-bit MLX; cloud 30B results are bf16
HuggingFace/Transformers unless otherwise stated.

## 5. Results

### 5.1 Baseline Gap Used for Evaluation

We compare each intervention against text-only compaction by measuring how much
of the full-context vs compacted gap it reduces. The gap is a normalization
denominator for mitigation scores. Removing the evidence is expected to hurt
recall.

For calibration, the early LongMemEval-S runs show full context at 71-81%
correct while compacted variants fall to at most 11%. The later 30B-bf16
aggregate gives a larger standard-data estimate: 52.5% correct with full
context vs 4.1% with text compaction over n=320. On SWE-Gym/OpenHands
next-action prediction, the full-context vs compacted gap is 0.164 nats/token.
The reported intervention effects are measured against these gaps.

### 5.2 Same Text, Different State

The same summary text behaves differently depending on whether its cache entries
were written under full context. In an earlier gapped summary-state pilot, the
write-time cache variant beats a fresh-encoded minimal summary control by
+0.093 nats on the 4B pilot, winning 10/12 conversations. At 30B the contrast
grows to +0.128 nats, winning 12/12 conversations, CI [0.097, 0.158].

A separate micro-sense experiment isolates the mechanism. A sentence with
identical tokens and positions is evaluated with and without a
disambiguating context. The bare-context sense margin is -0.23 nats.
Transplanting only value vectors moves it to +0.84; transplanting K+V moves it
to +1.95; full context is +3.81. Cached value tensors therefore carry
context-conditioned interpretation in this controlled setting.

Wrong-conversation and shuffled-value grafts degrade performance rather than
improve it, arguing against a generic smoothing explanation.

### 5.3 ValueGraft-Pack Reduces Fabrication in Agentic Frames

ValueGraft-Pack's clearest benefit is honesty rather than recall. On unknowable
questions, production-style compaction often fabricates. Packed summary contexts
make the model more cautious, and write-time summary state adds a further
component, especially at 30B.

Fabricated:admitted counts on Phase 2 synthetic/decoy probes:

| Arm | 30B decoys | 30B evicted facts | 4B decoys | 4B evicted facts |
| --- | --- | --- | --- | --- |
| B, production compaction | 19:5 | 16:8 | 18:6 | 15:9 |
| FreshPack, fresh packed summary | 10:14 | 5:19 | 4:20 | 6:18 |
| ValueGraft-Pack, write-time packed summary | **3:21** | **1:23** | **3:21** | **2:22** |

The matched comparison is ValueGraft-Pack vs FreshPack. At 30B, write-time
state reduces decoy fabrication from 10 to 3. At 4B the matched difference is
small; most of the effect is already produced by the packed minimal layout.

The same distinction explains why the standard benchmark result is weaker. In 30B
LongMemEval personal-QA framing, the compacted baseline already tends to admit
missing history. The larger n=320 run confirms that honesty is flat across
arms, and grafting does not increase fabrication. The honesty effect therefore
appears frame-dependent: it matters most when the compacted context invites the
model to continue acting as a task participant.

### 5.4 ValueGraft-Blend Recovers a Small Continuation Signal

ValueGraft-Blend improves continuation likelihood by small but consistent
amounts when tuned on a validation split and evaluated once on holdout.

| Scale | Setting | Holdout gain vs B | Wins | CI | Gap closure |
| --- | --- | --- | --- | --- | --- |
| 4B | mid-band alpha=0.25 | +0.017 nats | 10/10 | [0.012, 0.024] | ~10% |
| 30B | global alpha=0.75 | +0.033 nats | 9/10 | [0.014, 0.057] | ~24% |

The dose response changes with scale. At 4B, high alpha is harmful and layer
gating helps. At 30B, a global alpha around 0.75 works best, and alpha above 1.0
declines smoothly rather than failing immediately. Per-slot tuning produced a
promising 30B holdout result, but it is not yet robust enough to be a headline
method.

In the closest coding-domain check so far, on 75 OpenHands SWE-Gym traces,
tuned ValueGraft-Blend improves true next-action likelihood by +0.0156
nats/token, wins 45/75, CI [0.005, 0.027], and recovers about 10% of the
full-context vs compacted gap. This remains an offline proxy rather than an
end-to-end task-success result, but it places the effect on real agent
trajectories.

### 5.5 Boundary: No Recall Recovery

The interventions do not recover evicted facts. On LongMemEval, all compacted
variants remain at or below 11% correct in the early runs, and the larger
Stage-1 aggregate gives the expected full-context vs compacted recall gap. In
synthetic probe cuts, referent recovery remains poor. ValueGraft-Pack mainly
changes whether the model fabricates or admits missing information;
ValueGraft-Blend mainly shifts likelihood toward the full-context continuation.

The current evidence therefore supports "less damaging compaction," not "latent
recall of deleted context."

## 6. Discussion

The findings separate two losses from compaction: visible information and
computation state. Summary text can only preserve what it says. Write-time value
state appears to preserve some context-conditioned interpretation of the text
that remains or of the summary generated under full context. That state is not
enough to answer arbitrary questions about evicted facts, but it can affect
uncertainty and continuation behavior.

A compacted coding agent may not need to recall every old line verbatim, but it
does need to avoid confidently repeating failed approaches, hallucinating
settled details, or drifting away from the trajectory implied by the previous
work. The current OpenHands result is only a next-action likelihood proxy, but
it is pointed at a relevant failure surface.

## 7. Limitations

The evidence is narrow. Most results are from one model family. The 4B local
and 30B cloud runs differ in scale and precision. Synthetic probes are clean but
not representative of all agent failures. LongMemEval shows that prompt frame
matters: in personal-QA framing, 30B already admits missing history and the
honesty intervention has little room to help. The coding evidence is offline
next-action prediction, not end-to-end task completion.

Several controls remain. A stronger text-only summary baseline is needed.
ValueGraft-Pack's honesty effect is partly layout-driven and partly write-time
state-driven; the matched pair isolates some of the latter, but not every
possible caution mechanism. Per-slot calibration should remain exploratory until
it passes wrong-conversation guards and larger holdout tests. One-off demos are
useful explanations but are not evidence; the sense-level evidence is the
controlled micro-sense experiment.

## 8. Conclusion

The current evidence supports the following conclusion:

ValueGraft-Pack and ValueGraft-Blend do not make compacted models remember
deleted facts. They show that write-time cached attention state can reduce some
behavioral harm from compaction. ValueGraft-Pack can make post-compaction models
less likely to fabricate unsupported details in agentic frames. ValueGraft-Blend
can recover a small but consistent fraction of continuation and next-action
likelihood lost to compaction.

This supports treating conversation compaction as a text-plus-state problem. The
summary is the visible artifact, but the computation that produced and
interpreted it may also be worth preserving.

## Code Availability

Code, experiment scripts, draft analysis, and reproducibility notes are available
at <https://github.com/jeremyBanks/ValueGraft>.

## References

- Anthropic. n.d. [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction). Claude Platform Docs. Accessed 2026-07-06.
- Cao, Ziyi, Qingyi Si, Jingbin Zhang, and Bingquan Liu. 2025. [Sparse Attention across Multiple-context KV Cache](https://arxiv.org/abs/2508.11661). arXiv:2508.11661. DOI: [10.48550/arXiv.2508.11661](https://doi.org/10.48550/arXiv.2508.11661).
- Chevalier, Alexis, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. [Adapting Language Models to Compress Contexts](https://aclanthology.org/2023.emnlp-main.232/). In *Proceedings of EMNLP 2023*, pages 3829-3846. DOI: [10.18653/v1/2023.emnlp-main.232](https://doi.org/10.18653/v1/2023.emnlp-main.232).
- Eyuboglu, Sabri, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, and Christopher Re. 2025. [Cartridges: Lightweight and general-purpose long context representations via self-study](https://arxiv.org/abs/2506.06266). arXiv:2506.06266. DOI: [10.48550/arXiv.2506.06266](https://doi.org/10.48550/arXiv.2506.06266).
- Ge, Tao, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. 2024. [In-context Autoencoder for Context Compression in a Large Language Model](https://arxiv.org/abs/2307.06945). ICLR 2024; arXiv:2307.06945. DOI: [10.48550/arXiv.2307.06945](https://doi.org/10.48550/arXiv.2307.06945).
- Google AI for Developers. n.d. [Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking) and [context caching](https://ai.google.dev/gemini-api/docs/caching). Accessed 2026-07-06.
- Kim, Jang-Hyun, Junyoung Yeom, Sangdoo Yun, and Hyun Oh Song. 2024. [Compressed Context Memory For Online Language Model Interaction](https://arxiv.org/abs/2312.03414). ICLR 2024; arXiv:2312.03414. DOI: [10.48550/arXiv.2312.03414](https://doi.org/10.48550/arXiv.2312.03414).
- Li, Bojie. 2026. [Models Take Notes at Prefill: KV Cache Can Be Editable and Composable](https://arxiv.org/abs/2606.17107). arXiv:2606.17107. DOI: [10.48550/arXiv.2606.17107](https://doi.org/10.48550/arXiv.2606.17107).
- Mu, Jesse, Xiang Lisa Li, and Noah Goodman. 2023. [Learning to Compress Prompts with Gist Tokens](https://arxiv.org/abs/2304.08467). NeurIPS 2023; arXiv:2304.08467. DOI: [10.48550/arXiv.2304.08467](https://doi.org/10.48550/arXiv.2304.08467).
- OpenAI. n.d. [Compact a response](https://platform.openai.com/docs/api-reference/responses/compact) and [Conversation state](https://platform.openai.com/docs/guides/conversation-state). OpenAI API documentation. Accessed 2026-07-06.
- Yang, Jingbo, Bairu Hou, Wei Wei, Yujia Bao, and Shiyu Chang. 2025. [KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse](https://arxiv.org/abs/2502.16002). arXiv:2502.16002. DOI: [10.48550/arXiv.2502.16002](https://doi.org/10.48550/arXiv.2502.16002).
- Yao, Jiayi, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and Junchen Jiang. 2025. [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444). *EuroSys 2025*; arXiv:2405.16444. DOI: [10.48550/arXiv.2405.16444](https://doi.org/10.48550/arXiv.2405.16444).
- Zhang, Peitian, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou. 2024. [Long Context Compression with Activation Beacon](https://arxiv.org/abs/2401.03462). arXiv:2401.03462. DOI: [10.48550/arXiv.2401.03462](https://doi.org/10.48550/arXiv.2401.03462).
- Zweiger, Adam, Xinghong Fu, Han Guo, and Yoon Kim. 2026. [Fast KV Compaction via Attention Matching](https://arxiv.org/abs/2602.16284). arXiv:2602.16284. DOI: [10.48550/arXiv.2602.16284](https://doi.org/10.48550/arXiv.2602.16284).

## Author Contributions and Provenance

Jeremy Banks directed the project and is the first author of record. Anthropic
Claude Fable 5 designed and implemented much of the experimental machinery and
executed substantial autonomous coding-agent work. OpenAI GPT-5.5 contributed
adversarial review, reframing, prior-art analysis, and this synthesis. Claude
Sonnet subagents performed logged answer judging. This should be read as
AI-assisted research directed by the human first author.

## Notes for a Later Full Paper

Material intentionally minimized here but preserved in the wider draft:

- Detailed hosted-provider API analysis.
- Full value-steering calibration discussion, including signed vs clamped fits.
- Deployment cost accounting.
- Extended related-work taxonomy.
- Process notes and collaboration history.
