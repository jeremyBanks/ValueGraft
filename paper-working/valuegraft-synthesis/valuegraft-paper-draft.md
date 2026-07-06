# ValueGraft: Preserving Write-Time Value State Across Conversation Compaction Boundaries

**Status:** provisional paper-style synthesis, written as if the investigation
were interrupted at the current evidence frontier.  
**Date:** 2026-07-05  
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
factual recall. However, they do produce two scoped positive effects:
write-time summary state sharply reduces fabrication on unknowable post-
compaction questions in agentic frames, and tuned ValueGraft recovers a small
but consistent fraction of continuation or next-action likelihood lost to
compaction. The strongest coding-adjacent result so far is a +0.0156 nat/token
gain on OpenHands SWE-Gym trajectory prediction over 75 traces, about 10% of the
full-context vs compacted gap. The results support a limited claim: cached
attention value state can be used as a mitigation signal across compaction
boundaries, but the current evidence does not show general recall recovery or
end-to-end task improvement.

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
generation. In ordinary serving these tensors are called a KV cache because
they are used to avoid recomputing attention state. The word "cache" can be
misleading here: under ordinary use, caching is supposed to preserve behavior
while improving speed. Our question is not whether caching itself changes model
behavior. It is whether, at a compaction boundary, the discarded cached
attention state contains useful context-conditioned information that can be
retained or reintroduced to make compaction hurt less.

The most conservative framing is mitigation-first:

> Given that text-only compaction loses context-conditioned computation, can a
> small state-preserving intervention reduce the behavioral damage?

We evaluate two interventions. **SelfGist/H-pack** generates a summary while the
full conversation is still attendable, then keeps the summary's write-time
cache entries, with keys re-rotated into a compact packed layout. **ValueGraft**
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

If we had to stop now, the evidence would support a real but bounded result.
The interventions do not recover hidden facts from the evicted context. They do
not solve compaction. They do, however, change behavior in content-specific and
negative-control-certified ways, and the positive effects are strongest in the
settings where compaction resembles an agent continuing a task rather than a
personal-QA benchmark eliciting refusal.

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
   personal-QA benchmark framing can erase the honesty headroom, and per-slot
   or per-head calibration is not yet reliable enough to headline.

## 3. Related Work and Product Context

The broad claim that KV state is meaningful is no longer novel. The closest
mechanistic neighbor is **Models Take Notes at Prefill**, which argues that
prefill writes conclusions onto downstream cached state and demonstrates
editable, composable, position-portable KV blocks. **Fast KV Compaction via
Attention Matching** directly studies latent KV compaction and per-head
attention matching. **KVLink**, CacheBlend-style systems, and SamKV-like
methods study reuse or blending of independently encoded chunks, usually for
RAG or serving efficiency. Learned latent compression methods such as gist
tokens, AutoCompressor, ICAE, Activation Beacon, Compressed Context Memory, and
Cartridges ask models to carry context in compressed non-text forms. Text-space
compression and memory systems such as LLMLingua, RECOMP, MemGPT-like memory,
and framework-level agent summarization are the operational baseline.

The specific gap here is narrower: we study a conversation-compaction event
where old history is replaced by a generated visible summary plus retained
tail, then ask whether preserving or grafting write-time cached value state
improves behavior relative to text-only compaction.

Hosted provider APIs also now overlap with the proposed deployment shape.
OpenAI Responses exposes compaction through `context_management` and
`/responses/compact`, returning an encrypted `compaction` item that can be
passed forward. Anthropic exposes beta server-side compaction blocks and opaque
thinking signatures. Gemini exposes thought signatures, context caching, managed
agent compaction, Live API compression, and resumption handles. These public
interfaces strongly suggest that frontier providers are exploring opaque state
and compaction artifacts internally, but they do not reveal whether those
systems use raw KV tensors, cached value vectors, or any ValueGraft-like
mechanism. Therefore we should not claim novelty for opaque compaction handles
or infer provider internals. The contribution here is the open, controlled,
white-box measurement of one possible mechanism.

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

The core arms are:

- **A, full context:** the original conversation; the oracle ceiling.
- **B, production text compaction:** summary plus recent tail, freshly encoded.
- **B-min / B-min-pack:** minimal packed summary text controls, freshly encoded.
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

### 4.3 Data and Tasks

The evidence currently spans four settings.

**Synthetic planted conversations.** Twelve approximately 9K-token
conversations contain planted referents, ambiguous senses, stances, ruled-out
approaches, and evicted facts. A summary-leak audit classifies whether probes
are answerable from summary text. A terse "shadow summary" condition suppresses
summary leakage.

**Natural/free-form conversations.** Eight longer conversations provide held-out
continuations. These are useful for likelihood scoring but often have little
full-context vs compacted gap, limiting recoverable signal.

**LongMemEval-S.** Standard benchmark material is restructured so evidence
sessions fall in the evicted region and distractor sessions remain in the tail.
Reported runs cover n=48 at 4B and n=36 at 30B in the earlier standard-benchmark
validation, plus a larger 30B-bf16 run in progress at the time of this draft.

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
depends on evicted material. In LongMemEval-S, full context answers 71-81% of
questions correctly, while every compacted variant in the reported n=48/n=36
runs falls to at most 11% correct. On SWE-Gym/OpenHands next-action prediction,
the A-B gap is reported as 0.164 nats/token. This establishes headroom for
mitigation but is not itself the main claim.

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

The strongest qualitative mitigation result is reduced fabrication on
unknowable questions. In Phase 2, H-pack is compared to production compaction
and to a packed fresh-encode control.

Fabricated:admitted counts on 30B:

| Arm | Decoys | Evicted facts |
| --- | --- | --- |
| Full context A | 16:8 | 0:0, with 24/24 correct |
| Production compaction B | 19:5 | 16:8 |
| B-min-pack, fresh packed summary | 10:14 | 5:19 |
| H-pack, write-time packed summary | **3:21** | **1:23** |

At 4B the direction is similar:

| Arm | Decoys | Evicted facts |
| --- | --- | --- |
| Production compaction B | 18:6 | 15:9 |
| B-min-pack | 4:20 | 6:18 |
| H-pack | **3:21** | **2:22** |

The decomposition matters. Much of the honesty effect comes from the packed
minimal context itself: B-min-pack is already far more cautious than production
B. The matched encoding-specific component is clearest at 30B, where H-pack
reduces decoy fabrication from 10 to 3 relative to B-min-pack. Wrong-summary
H-pack is also cautious, so "state/layout oddness induces admission" remains a
partial competing explanation. The conservative claim is not that H-pack
remembers hidden facts. It is that packed write-time summary state can reduce
post-compaction fabrication in agentic frames.

LongMemEval bounds the claim. At 4B, H-pack reduces fabrication on
LongMemEval-S from 17 to 11 relative to B. At 30B, the effect vanishes: B
already fabricates rarely on personal-history questions, apparently because the
model's refusal or uncertainty calibration is triggered by the benchmark frame.
This suggests the effect is most relevant where the compacted context invites
the model to continue acting as a task participant, not where the prompt already
sounds like retrieval from unavailable personal history.

### 5.4 ValueGraft Recovers a Small Fraction of Continuation Loss

ValueGraft gives a modest but consistent continuation benefit after tuning on a
validation split and reporting once on holdout.

| Scale | Reported setting | Holdout gain vs B | Wins | 95% CI | Approx. gap closure |
| --- | --- | --- | --- | --- | --- |
| 4B | mid-band alpha=0.25 | +0.017 nats | 10/10 | [0.012, 0.024] | ~10% |
| 30B | global alpha=0.75 | +0.033 nats | 9/10 | [0.014, 0.057] | ~24% |

The dose response changes with scale. At 4B, strong grafting is harmful and
mid-layer gating helps. At 30B, global alpha around 0.75-1.0 works best, and
alpha=1.25 remains positive though below the peak. This suggests larger models
may tolerate fresh-key/old-value mismatch better, but the evidence is still
within one model family.

The coding-adjacent result is smaller but important because it uses real agent
traces. On 75 OpenHands trajectories, tuned ValueGraft gains +0.0156 nats on
true next-action prediction, wins 45/75, CI [0.005, 0.027], about 10% of the
A-B gap. This does not yet imply better end-to-end issue resolution. It does
show that the effect survives beyond synthetic probes into a real coding-agent
trace distribution.

### 5.5 Per-Slot and Per-Head Calibration Is Not Yet a Headline Result

A 30B positive-profile 57-slot mask beat global alpha=0.75 on holdout:
+0.0384 vs +0.0239, with 10/10 wins. This is promising, but it is deliberately
not adopted as the primary method. A 4B per-head story failed on holdout, and
the 30B slot mask has not yet passed its wrong-conversation guard. The current
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
larger models may already admit missing history, leaving little room for
honesty interventions. Agentic contexts are the intended deployment target, but
the current coding evidence is offline next-action prediction rather than
end-to-end task success.

Fourth, the summary channel is a powerful baseline. Better summaries, extractive
memories, or task-specific compactors may erase some gains. A deliberately
strong text-only baseline remains an important future control.

Fifth, the mechanisms are not fully disentangled. In H-pack, packed layout,
state oddness, and write-time encoding all affect admission behavior. The
matched H-pack vs B-min-pack pair isolates part of the write-time component,
but wrong-summary honesty shows that caution can arise for reasons other than
content-correct latent state.

Sixth, per-slot calibration is underpowered. The 30B slot result is promising,
but without a guard pass and more data it is too easy to overfit a profile
matrix. The paper should keep the primary method simple.

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
contexts where the model is expected to continue work after old history has
been summarized away.

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

## References and Pointers

This draft is intentionally light on formal bibliography formatting. The
working source documents are:

- `RESULTS.md`
- `RESULTS-30B-addendum.md`
- `PHASE2-RESULTS.md`
- `LONGMEMEVAL-RESULTS.md`
- `DECISIONS.md`
- `value-steering-design-notes.md`
- `provider-compaction-prior-art-review.md`
- `semantic-continuity-experiment-brief.md`
- `amendments-from-external-review.md`

Key external items to cite formally in a final version:

- Models Take Notes at Prefill: KV Cache Can Be Editable and Composable.
- Fast KV Compaction via Attention Matching.
- Parallel Context Compaction for Long-Horizon LLM Agent Serving.
- KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse.
- OpenAI Responses compaction and prompt-caching documentation.
- Anthropic compaction, context-editing, prompt-caching, and extended-thinking
  documentation.
- Gemini context caching, thought-signature, Managed Agents, and Live API
  session-management documentation.
