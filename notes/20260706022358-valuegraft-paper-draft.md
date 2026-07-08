# ValueGraft: Preserving Write-Time Value State Across Conversation Compaction Boundaries

**Status:** provisional paper-style synthesis, written as if the investigation
were interrupted at the current evidence frontier.\
**Date:** 2026-07-05\
**Authors:** Jeremy Banks; Anthropic Claude Fable 5; OpenAI GPT-5.5

## Abstract

Long-running LLM agents commonly compact conversation history by replacing old
turns with a text summary and re-encoding the remaining transcript. This is a
practical necessity, but it discards the cached attention state that was written
when the model originally interpreted the retained text under the full
conversation. We evaluate whether preserving or grafting this write-time state
can reduce compaction-induced behavioral damage. We study two training-free
interventions on Qwen3-4B-Instruct-2507 and Qwen3-30B-A3B-Instruct-2507:
SelfGist/H-pack, which carries the summary's generation-time cache entries
forward in packed form, and ValueGraft, which freshly encodes the compacted
context but replaces or blends aligned cached value tensors with their
full-context counterparts. Across synthetic conversations, standard benchmark
material, and offline coding-agent traces, these methods do not restore evicted
factual recall. However, they do produce two scoped positive effects: write-time
summary state reduces fabrication on unknowable post-compaction questions in
agentic frames, and tuned ValueGraft recovers a small but consistent fraction of
continuation or next-action likelihood lost to compaction. A larger 30B-bf16
LongMemEval aggregate confirms severe compaction damage on standard data, but
also shows that QA-style framing can erase mitigation headroom by making all
compacted arms similarly cautious. The strongest coding-adjacent result so far
is a +0.0156 nat/token gain on OpenHands SWE-Gym trajectory prediction over 75
traces, about 10% of the full-context vs compacted gap. The results support a
limited claim: cached attention value state can be used as a mitigation signal
across compaction boundaries, but the current evidence does not show general
recall recovery or end-to-end task improvement.

## 1. Introduction

Conversation compaction is now a normal part of long-running LLM systems. A
client or harness takes an overlong transcript, asks the model or another model
to summarize the older portion, keeps a recent tail, and continues from the
shorter context. The visible text is preserved well enough for many purposes,
but the model's earlier computation over that text is not preserved. The same
sentence re-encoded after compaction may no longer be conditioned on the context
that originally resolved its references, senses, constraints, or failed paths.

This project began from a simple mechanism-level observation: transformer
inference already computes per-token key and value tensors during prefill and
generation. In ordinary serving these tensors are called a KV cache because they
are used to avoid recomputing attention state. The word "cache" can be
misleading here: under ordinary use, caching is supposed to preserve behavior
while improving speed. Our question is not whether caching itself changes model
behavior. It is whether, at a compaction boundary, the discarded cached
attention state contains useful context-conditioned information that can be
retained or reintroduced to make compaction hurt less.

The most conservative framing is mitigation-first:

> Given that text-only compaction loses context-conditioned computation, can a
> small state-preserving intervention reduce the behavioral damage?

We evaluate two interventions. **SelfGist/H-pack** generates a summary while the
full conversation is still attendable, then keeps the summary's write-time cache
entries, with keys re-rotated into a compact packed layout. **ValueGraft**
builds the ordinary compacted context, keeps its fresh keys, and blends old
full-context value tensors into aligned fresh value slots:

```text
V_final[layer, kv_head, position, :] =
  (1 - alpha) * V_fresh[layer, kv_head, position, :]
  + alpha * V_old[layer, kv_head, aligned_old_position, :]
```

The implemented ValueGraft arms use exact token alignment: the summary text was
generated in-context and is reused verbatim, and retained tail tokens are
literal text from the source conversation. This avoids fuzzy semantic matching.
Keys stay fresh in the compacted context; only value tensors are changed.

If we had to stop now, the evidence would support a real but bounded result. The
interventions do not recover hidden facts from the evicted context. They do not
solve compaction. They do, however, change behavior in content-specific and
negative-control-certified ways, and the positive effects are strongest in the
settings where compaction resembles an agent continuing a task rather than a
personal-QA benchmark eliciting refusal.

### 1.1 Terminology

Throughout this draft, **cached attention state** means the per-layer key and
value tensors stored for each token during transformer prefill/generation.
**Cached value tensors** or **value state** refer specifically to the V side of
that state. These are not scalar attention weights. We keep the term **KV
cache** when referring to the standard implementation object, but the
intervention is about preserving or modifying the state stored there, not about
ordinary caching as a speed optimization.

We use **old** or **write-time** state for tensors computed while the full
pre-compaction context was still attendable. We use **fresh** state for tensors
computed by re-encoding the compacted summary/tail context from scratch.

We separate three outcomes:

- **Recall:** correctly answering questions whose evidence was in the evicted
  context.
- **Honesty:** admitting missing information rather than fabricating an answer.
- **Continuity:** assigning higher likelihood to the true next continuation or
  next agent action after compaction.

The current interventions mainly improve honesty and continuity, not recall.

### 1.2 Conclusions If Cut Off Here

| Question                                                                            | Current answer                                                                    | Strength                        |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------- |
| Does text-only compaction damage behavior when the task depends on evicted context? | Yes. This is clear in synthetic probes, LongMemEval, and coding-trace likelihood. | Strong within tested settings   |
| Does write-time state affect behavior for identical visible text?                   | Yes. H-gap vs B-min and micro-sense tests show same-text/different-state effects. | Strong as mechanism evidence    |
| Does preserving summary write-time state recover evicted factual recall?            | No. Recall remains mostly lost.                                                   | Strong negative in current data |
| Does H-pack reduce fabrication?                                                     | Yes in agentic/synthetic frames; less or not at all in 30B personal-QA framing.   | Moderate, frame-dependent       |
| Does ValueGraft improve continuation or next-action likelihood?                     | Yes, by small but consistent amounts on holdout and coding traces.                | Moderate                        |
| Are per-head/per-slot policies ready for the headline method?                       | No. They are promising exploration, but too overfit-prone so far.                 | Strong methodological decision  |

## 2. Contributions

This draft makes four contributions, scoped to the evidence currently in hand:

1. It defines a compaction-boundary intervention, ValueGraft, that blends
   write-time cached attention value tensors into a freshly encoded compacted
   context while preserving fresh keys and contiguous positions.
2. It evaluates a paired family of baselines and controls that separate text
   summary quality, packed-layout effects, write-time summary encoding, and
   value-state grafting.
3. It reports evidence that write-time state can reduce post-compaction
   fabrication and recover a small fraction of continuation/next-action
   likelihood loss.
4. It identifies boundaries: evicted factual recall remains unrecovered,
   personal-QA benchmark framing can erase the honesty headroom, and per-slot or
   per-head calibration is not yet reliable enough to headline.

## 3. Related Work and Product Context

The broad claim that KV state is meaningful is no longer novel. The closest
mechanistic neighbor is Li (2026), which argues that prefill writes conclusions
onto downstream cached state and demonstrates editable, composable,
position-portable KV blocks. Zweiger et al. (2026) directly study latent KV
compaction and per-head attention matching. KVLink (Yang et al., 2025),
CacheBlend (Yao et al., 2025), and SamKV (Cao et al., 2025) study reuse or
blending of independently encoded chunks, usually for RAG or serving efficiency.
Learned latent compression methods such as gist tokens (Mu et al., 2023),
AutoCompressor (Chevalier et al., 2023), ICAE (Ge et al., 2024), Activation
Beacon (Zhang et al., 2024), Compressed Context Memory (Kim et al., 2024), and
Cartridges (Eyuboglu et al., 2025) ask models to carry context in compressed
non-text forms. Text-space compression and memory systems such as LLMLingua
(Jiang et al., 2023), RECOMP (Xu et al., 2024), MemGPT (Packer et al., 2023),
and framework-level agent summarization are the operational baseline.

The specific gap here is narrower: we study a conversation-compaction event
where old history is replaced by a generated visible summary plus retained tail,
then ask whether preserving or grafting write-time cached value state improves
behavior relative to text-only compaction.

Hosted provider APIs also now overlap with the proposed deployment shape. OpenAI
Responses exposes compaction through `context_management` and
`/responses/compact`, returning an encrypted `compaction` item that can be
passed forward (OpenAI, n.d.). Anthropic exposes beta server-side compaction
blocks and context-management controls (Anthropic, n.d.). Gemini exposes thought
signatures, context caching, and server-managed interaction state (Google AI for
Developers, n.d.). These public interfaces show that frontier providers are
already exposing opaque state and compaction artifacts, but they do not reveal
whether those systems use raw KV tensors, cached value vectors, or any
ValueGraft-like mechanism. Therefore we should not claim novelty for opaque
compaction handles or infer provider internals. The contribution here is the
open, controlled, white-box measurement of one possible mechanism.

## 4. Methods

### 4.1 Models and Runtime

The main model family is Qwen3-Instruct-2507:

- Local pilot: `Qwen3-4B-Instruct-2507`, 4-bit MLX weights with fp16 cache.
- Scale runs: `Qwen3-30B-A3B-Instruct-2507`, including 4-bit MLX targeted runs
  and bf16 HuggingFace/Transformers cloud runs.

All evaluation uses temperature 0. The project maintains a strict distinction
between local 4-bit results and cloud bf16 results; cross-precision comparisons
are stated as such.

The cache machinery was validated before use. For Qwen3 in this stack, keys are
stored post-RoPE, values are unrotated, and cache round trips are
generation-identical. The build ladder includes:

- L0 cache reconstruction identity.
- L1 null surgery identity.
- L2 trivial irrelevant eviction.
- L3 ValueGraft identities: alpha=0 reproduces the text baseline, and alpha=1
  with old context equal to new context reproduces the oracle.
- L4 tokenization stability audits.
- LH/HP tests for key re-rotation and packed H-pack identity.

Several runtime traps were found and logged: Qwen chat templates add an empty
`<think>` block to final assistant messages, so canonical rendering is required;
batched prefill logits must not be compared to decode-step logits; 4-bit
quantized kernels can be sequence-length-dependent, so cross-shape comparisons
must be bounded by this numerical floor.

### 4.2 Experimental Arms

Each arm starts from the same canonical rendering of a conversation. The first
four tokens are retained as attention sinks. A tail boundary is chosen at a
message boundary near the final quarter of the tokenized conversation; material
between the sinks and that boundary is the evicted region. A summary is then
generated greedily by the subject model while the full pre-compaction
conversation remains attendable. The resulting cache snapshot is the source of
the write-time state used by H-pack and ValueGraft.

The core arms are:

- **A, full context:** the original conversation; the oracle ceiling.
- **B, production text compaction:** a new transcript containing the system
  message, an assistant context note with the summary, and the retained tail;
  freshly encoded from scratch.
- **B-min / B-min-pack:** four sink tokens plus the exact generated summary
  token ids, packed contiguously and freshly encoded.
- **H-gap / H-pack:** summary cache entries generated under full context and
  carried forward; H-pack uses packed positions with re-rotated keys.
- **C, gapped retention:** sinks, original tail entries, and summary write-time
  entries retained with positional gaps.
- **E / ValueGraft:** B's compacted context, fresh keys, blended old value
  tensors at exact-aligned summary and tail positions.
- **Negative controls:** wrong-conversation grafts and shuffled-value grafts.

H-pack vs B-min-pack is the clean matched pair for summary write-time encoding:
same packed token sequence and positions; different cached attention state.
ValueGraft vs B measures whether old value payloads improve a normal compacted
context.

H-pack is built by extracting the generated-summary span from the saved
summary-generation cache. Since Qwen3 keys are already RoPE-rotated in cache,
moving the summary to a packed prefix requires re-rotating the keys by the
difference between their original write-time positions and their new packed
positions. Values are unrotated and are copied directly. The packed cache then
contains only the sink entries and the summary entries, and future tokens are
generated as if this were an ordinary contiguous prefix. B-min-pack is the
matched fresh-prefill control for exactly this packed token sequence.

ValueGraft instead keeps the production B transcript. After freshly pre-filling
B, it aligns B's summary and tail tokens to the old full-context
summary-generation token stream. Alignment is exact-token matching in the
summary and tail regions separately; blocks shorter than eight tokens are
dropped, and special tokens and sink positions are excluded. For every accepted
pair `(new_pos, old_pos)`, each selected layer's value tensor is replaced by
`(1 - alpha) * V_fresh[new_pos] + alpha * V_old[old_pos]`. Keys remain fresh.
The reported version is post-prefill ValueGraft: it edits the completed B cache
rather than feeding edited values forward during prefill. The current tuned
settings are mid-layer alpha=0.25 at 4B and global alpha=0.75 at 30B.

Wrong-conversation, shuffled-value, and wrong-summary controls test whether
improvements come from content-specific state rather than generic perturbation.
Wrong-conversation grafts use old values from another conversation;
shuffled-value grafts use the right conversation's values at wrong matched
positions; H-pack-wrongS uses packed summary entries from another conversation.

### 4.3 Data and Tasks

The evidence currently spans four settings.

**Synthetic planted conversations.** Twelve approximately 9K-token conversations
contain planted referents, ambiguous senses, stances, ruled-out approaches, and
evicted facts. A summary-leak audit classifies whether probes are answerable
from summary text. A terse "shadow summary" condition suppresses summary
leakage.

**Natural/free-form conversations.** Eight longer conversations provide held-out
continuations. These are useful for likelihood scoring but often have little
full-context vs compacted gap, limiting recoverable signal.

**LongMemEval-S.** Standard benchmark material is restructured so evidence
sessions fall in the evicted region and distractor sessions remain in the tail.
Early reported runs cover n=48 at 4B and n=36 at 30B. A later Stage-1 30B-bf16
aggregate over n=320 standard-data questions is now the larger anchor for
compaction damage and QA-frame arm equivalence.

**SWE-Gym/OpenHands trajectory prediction.** Seventy-five real OpenHands traces
are compacted mid-trajectory. The main offline coding metric is teacher-forced
log probability of the true next assistant action. A behavioral repeated-failed-
command heuristic was attempted but is not yet reliable because it also fires
for the oracle.

### 4.4 Metrics

We report:

- Probe accuracy by category and leakage class.
- Fabricated vs admitted-ignorance counts on unknowable evicted or decoy
  questions.
- Mean per-token log likelihood of held-out continuations or next actions.
- Gap closure `(arm - B) / (A - B)` when the A-B gap is meaningful.
- Paired wins and bootstrap confidence intervals over conversations.

The most important interpretive rule is that recall and honesty are different.
"Less fabrication" may mean better uncertainty calibration rather than better
memory. The project therefore reports correct / fabricated / admitted counts
separately where possible.

## 5. Results

### 5.1 Compaction Damage Replicates

The full context substantially outperforms text compaction whenever the task
depends on evicted material. In the early LongMemEval-S runs, full context
answers 71-81% of questions correctly, while every compacted variant in the
reported n=48/n=36 runs falls to at most 11% correct. The later 30B-bf16 Stage-1
aggregate gives a larger standard-data estimate: full context at 52.5% correct
vs compacted baseline at 4.1% over n=320. On SWE-Gym/OpenHands next-action
prediction, the A-B gap is reported as 0.164 nats/token. This establishes
headroom for mitigation but is not itself the main claim.

### 5.2 Identical Text, Different Write-Time State

The same summary text behaves differently depending on whether its cache entries
were written under the full context. In the 4B pilot, H-gap beats B-min by
+0.093 nats on continuation likelihood, 10/12 conversations. At 30B, the same
contrast grows to +0.128 nats, 12/12 conversations, CI [0.097, 0.158].

An isolated micro-sense experiment supports the mechanism. A sentence with
identical tokens and positions is evaluated in a disambiguating context and a
bare context. The fresh bare-context sense margin is -0.23 nats; transplanting
only value vectors moves it to +0.84; transplanting K+V moves it to +1.95; the
full-context oracle is +3.81. This does not prove a deployed mitigation by
itself, but it shows that cached value tensors can carry context-conditioned
interpretation.

Negative controls are important: shuffled and wrong-conversation value grafts
collapse performance rather than improve it. That makes it unlikely that the
positive E-style gains are just generic smoothing.

### 5.3 H-Pack Reduces Fabrication but Does Not Recover Recall

The strongest qualitative mitigation result is reduced fabrication on unknowable
questions. In Phase 2, H-pack is compared to production compaction and to a
packed fresh-encode control.

Fabricated:admitted counts on 30B:

| Arm                               | Decoys   | Evicted facts           |
| --------------------------------- | -------- | ----------------------- |
| Full context A                    | 16:8     | 0:0, with 24/24 correct |
| Production compaction B           | 19:5     | 16:8                    |
| B-min-pack, fresh packed summary  | 10:14    | 5:19                    |
| H-pack, write-time packed summary | **3:21** | **1:23**                |

At 4B the direction is similar:

| Arm                     | Decoys   | Evicted facts |
| ----------------------- | -------- | ------------- |
| Production compaction B | 18:6     | 15:9          |
| B-min-pack              | 4:20     | 6:18          |
| H-pack                  | **3:21** | **2:22**      |

The decomposition matters. Much of the honesty effect comes from the packed
minimal context itself: B-min-pack is already far more cautious than production
B. The matched encoding-specific component is clearest at 30B, where H-pack
reduces decoy fabrication from 10 to 3 relative to B-min-pack. Wrong-summary
H-pack is also cautious, so "state/layout oddness induces admission" remains a
partial competing explanation. The conservative claim is not that H-pack
remembers hidden facts. It is that packed write-time summary state can reduce
post-compaction fabrication in agentic frames.

LongMemEval bounds the claim. At 4B, H-pack reduces fabrication on LongMemEval-S
from 17 to 11 relative to B. At 30B, the effect vanishes: B already fabricates
rarely on personal-history questions, apparently because the model's refusal or
uncertainty calibration is triggered by the benchmark frame. The larger 30B-bf16
Stage-1 aggregate reinforces this: local evidence had already predicted
arm-equivalence under QA framing, and the cloud run confirmed that honesty was
flat across compacted arms while grafting did not increase fabrication (50 vs
51). This suggests the effect is most relevant where the compacted context
invites the model to continue acting as a task participant, not where the prompt
already sounds like retrieval from unavailable personal history.

### 5.4 ValueGraft Recovers a Small Fraction of Continuation Loss

ValueGraft gives a modest but consistent continuation benefit after tuning on a
validation split and reporting once on holdout.

| Scale | Reported setting    | Holdout gain vs B | Wins  | 95% CI         | Approx. gap closure |
| ----- | ------------------- | ----------------- | ----- | -------------- | ------------------- |
| 4B    | mid-band alpha=0.25 | +0.017 nats       | 10/10 | [0.012, 0.024] | ~10%                |
| 30B   | global alpha=0.75   | +0.033 nats       | 9/10  | [0.014, 0.057] | ~24%                |

The dose response changes with scale. At 4B, strong grafting is harmful and
mid-layer gating helps. At 30B, global alpha around 0.75-1.0 works best, and
alpha=1.25 remains positive though below the peak. This suggests larger models
may tolerate fresh-key/old-value mismatch better, but the evidence is still
within one model family.

The coding-adjacent result is smaller but important because it uses real agent
traces. On 75 OpenHands trajectories, tuned ValueGraft gains +0.0156 nats on
true next-action prediction, wins 45/75, CI [0.005, 0.027], about 10% of the A-B
gap. This does not yet imply better end-to-end issue resolution. It does show
that the effect survives beyond synthetic probes into a real coding-agent trace
distribution.

### 5.5 Per-Slot and Per-Head Calibration Is Not Yet a Headline Result

A 30B positive-profile 57-slot mask beat global alpha=0.75 on holdout: +0.0384
vs +0.0239, with 10/10 wins. This is promising, but it is deliberately not
adopted as the primary method. A 4B per-head story failed on holdout, and the
30B slot mask has not yet passed its wrong-conversation guard. The current
policy is therefore sound: report simple global alpha as the primary method and
reserve per-slot, factored, or signed calibration for exploration.

## 6. Discussion

The results are mixed in a way that is scientifically useful. The interventions
do not retrieve lost facts from the evicted context. That failure is consistent
across synthetic probes and LongMemEval. If a fact is absent from visible text
and only present in discarded history, neither ValueGraft nor H-pack should be
described as recovering it.

What does survive is subtler. The summary and tail tokens can carry
context-conditioned write-time state that affects uncertainty, continuation
style, and next-action likelihood. In H-pack, that state appears to make the
model less willing to fabricate unsupported specifics, especially in frames
where ordinary compaction makes the model behave as though it should still know
the whole task. In ValueGraft, blended value tensors provide a small directional
correction toward the full-context continuation distribution.

This distinction helps reconcile the apparently small effect sizes. A
0.015-0.033 nat/token gain is not a dramatic language-model improvement. But in
teacher-forced next-action prediction, especially after the summary already
contains much of the visible signal, a consistent paired gain can still indicate
a real reduction in compaction damage. The correct next test is not to inflate
the claim; it is to evaluate whether such small likelihood gains correspond to
fewer practical agent failures in end-to-end coding tasks.

The production shape is plausible. Major hosted APIs already expose compaction,
opaque reasoning, thought-signature, prompt-caching, and session-continuation
artifacts. A provider-side compaction operation could in principle return a
visible summary plus an opaque state item. But the provider-surface overlap also
narrowly constrains novelty: the paper should not claim to invent opaque
compaction handles or to discover that KV state is meaningful. It should claim a
specific open experiment on summary-boundary mitigation.

## 7. Limitations

The evidence is still limited.

First, most controlled results come from one model family. The local 4B and
cloud 30B runs differ in both scale and precision, so cross-scale trends are
suggestive rather than definitive.

Second, the synthetic probe suite is useful but narrow. It was designed to
stress referents, senses, stances, ruled-out options, and evicted facts. That
gives clean measurement but not a complete distribution of real agent failures.

Third, LongMemEval validation shows that frame matters. In personal-QA framing,
larger models may already admit missing history, leaving little room for honesty
interventions; the n=320 Stage-1 aggregate makes this more than a small-sample
caveat. Agentic contexts are the intended deployment target, but the current
coding evidence is offline next-action prediction rather than end-to-end task
success.

Fourth, the summary channel is a powerful baseline. Better summaries, extractive
memories, or task-specific compactors may erase some gains. A deliberately
strong text-only baseline remains an important future control.

Fifth, the mechanisms are not fully disentangled. In H-pack, packed layout,
state oddness, and write-time encoding all affect admission behavior. The
matched H-pack vs B-min-pack pair isolates part of the write-time component, but
wrong-summary honesty shows that caution can arise for reasons other than
content-correct latent state.

Sixth, per-slot calibration is underpowered. The 30B slot result is promising,
but without a guard pass and more data it is too easy to overfit a profile
matrix. The paper should keep the primary method simple.

Seventh, illustrative demos are not evidence. One-off examples are useful for
explaining the mechanism to readers, but the evidence for sense-level recovery
in this draft comes from the controlled micro-sense experiment, not from
hand-built demonstrations.

## 8. Practical Overhead

ValueGraft requires one ordinary compacted-context prefill plus a vectorized
value blend at compaction time, and it requires the old cache to remain
available until the operation is complete. It is therefore a server-side or
local-agent technique, not a method for reconstructing state after it has been
discarded.

H-pack/SelfGist stores only the summary's cache entries. For the 30B model, the
rough fp16 cache cost is about 96 KiB per token:

```text
48 layers * 2 (K,V) * 4 KV heads * 128 dim * 2 bytes
```

A 100-token terse summary is therefore roughly 10 MiB of state; a 500-token
summary is roughly 47 MiB. That is much smaller than a full long-context cache
but vastly larger than the summary text. In a hosted API, the natural interface
would be an opaque provider-managed compaction artifact with expiry and billing,
not a raw client-uploaded KV sidecar.

## 9. Conclusion

If the project stopped here, the conclusion would be:

Text-only conversation compaction discards useful write-time attention state.
Preserving that state does not recover hidden factual recall, but it can reduce
two important kinds of damage: fabrication after compaction, and loss of
continuation/next-action likelihood on compaction-sensitive tasks. The evidence
is strongest for a scoped mitigation claim, not a sweeping memory claim:
ValueGraft and H-pack are promising training-free interventions for preserving
some behavioral continuity across compaction boundaries, especially in agentic
contexts where the model is expected to continue work after old history has been
summarized away.

The next required evidence is straightforward: run a stronger text-only
baseline, complete the standard-protocol LongMemEval and coding-agent analyses,
test end-to-end task success, and replicate outside the Qwen3-2507 family.

## Author Contributions and Provenance

Jeremy Banks directed the project, made the framing, budget, and scaling
decisions, supplied interpretation and critique throughout, and is the first
author of record. Anthropic Claude Fable 5 designed and implemented much of the
experimental machinery, ran autonomous coding-agent work, wrote many of the
repository notes, and coordinated substantial experiment execution. OpenAI
GPT-5.5 contributed adversarial review, mitigation-first reframing, prior-art
analysis, and this provisional synthesis. Claude Sonnet subagents performed
logged answer judging. All AI-generated experimental and analytical work should
be treated as assisted research output directed by the human first author, not
as independent personal authorship in the human sense.

## References

- Anthropic. n.d.
  [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction),
  [Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows),
  and
  [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).
  Claude Platform Docs. Accessed 2026-07-06.
- Cao, Ziyi, Qingyi Si, Jingbin Zhang, and Bingquan Liu. 2025.
  [Sparse Attention across Multiple-context KV Cache](https://arxiv.org/abs/2508.11661).
  arXiv:2508.11661. DOI:
  [10.48550/arXiv.2508.11661](https://doi.org/10.48550/arXiv.2508.11661).
- Chevalier, Alexis, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023.
  [Adapting Language Models to Compress Contexts](https://aclanthology.org/2023.emnlp-main.232/).
  In _Proceedings of EMNLP 2023_, pages 3829-3846. DOI:
  [10.18653/v1/2023.emnlp-main.232](https://doi.org/10.18653/v1/2023.emnlp-main.232).
- Cim, Musa, Burak Topcu, Chita Das, and Mahmut Taylan Kandemir. 2026.
  [Parallel Context Compaction for Long-Horizon LLM Agent Serving](https://arxiv.org/abs/2605.23296).
  arXiv:2605.23296. DOI:
  [10.48550/arXiv.2605.23296](https://doi.org/10.48550/arXiv.2605.23296).
- Eyuboglu, Sabri, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily
  Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, and Christopher
  Re. 2025.
  [Cartridges: Lightweight and general-purpose long context representations via self-study](https://arxiv.org/abs/2506.06266).
  arXiv:2506.06266. DOI:
  [10.48550/arXiv.2506.06266](https://doi.org/10.48550/arXiv.2506.06266).
- Ge, Tao, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. 2024.
  [In-context Autoencoder for Context Compression in a Large Language Model](https://arxiv.org/abs/2307.06945).
  ICLR 2024; arXiv:2307.06945. DOI:
  [10.48550/arXiv.2307.06945](https://doi.org/10.48550/arXiv.2307.06945).
- Google AI for Developers. n.d.
  [Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking),
  [Context caching](https://ai.google.dev/gemini-api/docs/caching), and
  [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview).
  Accessed 2026-07-06.
- Jiang, Huiqiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023.
  [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models](https://arxiv.org/abs/2310.05736).
  EMNLP 2023; arXiv:2310.05736. DOI:
  [10.48550/arXiv.2310.05736](https://doi.org/10.48550/arXiv.2310.05736).
- Kim, Jang-Hyun, Junyoung Yeom, Sangdoo Yun, and Hyun Oh Song. 2024.
  [Compressed Context Memory For Online Language Model Interaction](https://arxiv.org/abs/2312.03414).
  ICLR 2024; arXiv:2312.03414. DOI:
  [10.48550/arXiv.2312.03414](https://doi.org/10.48550/arXiv.2312.03414).
- Li, Bojie. 2026.
  [Models Take Notes at Prefill: KV Cache Can Be Editable and Composable](https://arxiv.org/abs/2606.17107).
  arXiv:2606.17107. DOI:
  [10.48550/arXiv.2606.17107](https://doi.org/10.48550/arXiv.2606.17107).
- Mu, Jesse, Xiang Lisa Li, and Noah Goodman. 2023.
  [Learning to Compress Prompts with Gist Tokens](https://arxiv.org/abs/2304.08467).
  NeurIPS 2023; arXiv:2304.08467. DOI:
  [10.48550/arXiv.2304.08467](https://doi.org/10.48550/arXiv.2304.08467).
- OpenAI. n.d.
  [Compact a response](https://platform.openai.com/docs/api-reference/responses/compact),
  [Conversation state](https://platform.openai.com/docs/guides/conversation-state),
  and [Prompt caching](https://platform.openai.com/docs/guides/prompt-caching).
  OpenAI API documentation. Accessed 2026-07-06.
- Packer, Charles, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion
  Stoica, and Joseph E. Gonzalez. 2023.
  [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560).
  arXiv:2310.08560. DOI:
  [10.48550/arXiv.2310.08560](https://doi.org/10.48550/arXiv.2310.08560).
- Xu, Fangyuan, Weijia Shi, and Eunsol Choi. 2024.
  [RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://arxiv.org/abs/2310.04408).
  ICLR 2024; arXiv:2310.04408. DOI:
  [10.48550/arXiv.2310.04408](https://doi.org/10.48550/arXiv.2310.04408).
- Yang, Jingbo, Bairu Hou, Wei Wei, Yujia Bao, and Shiyu Chang. 2025.
  [KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse](https://arxiv.org/abs/2502.16002).
  arXiv:2502.16002. DOI:
  [10.48550/arXiv.2502.16002](https://doi.org/10.48550/arXiv.2502.16002).
- Yao, Jiayi, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang,
  Kuntai Du, Shan Lu, and Junchen Jiang. 2025.
  [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444).
  _EuroSys 2025_; arXiv:2405.16444. DOI:
  [10.48550/arXiv.2405.16444](https://doi.org/10.48550/arXiv.2405.16444).
- Zhang, Peitian, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng
  Dou. 2024.
  [Long Context Compression with Activation Beacon](https://arxiv.org/abs/2401.03462).
  arXiv:2401.03462. DOI:
  [10.48550/arXiv.2401.03462](https://doi.org/10.48550/arXiv.2401.03462).
- Zweiger, Adam, Xinghong Fu, Han Guo, and Yoon Kim. 2026.
  [Fast KV Compaction via Attention Matching](https://arxiv.org/abs/2602.16284).
  arXiv:2602.16284. DOI:
  [10.48550/arXiv.2602.16284](https://doi.org/10.48550/arXiv.2602.16284).

Repository source documents used for this synthesis:

- `RESULTS.md`
- `RESULTS-30B-addendum.md`
- `PHASE2-RESULTS.md`
- `LONGMEMEVAL-RESULTS.md`
- `DECISIONS.md`
- `value-steering-design-notes.md`
- `provider-compaction-prior-art-review.md`
- `semantic-continuity-experiment-brief.md`
- `amendments-from-external-review.md`
