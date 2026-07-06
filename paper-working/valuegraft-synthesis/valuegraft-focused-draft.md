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
write-time state can reduce the impact of compaction. We study two training-free
interventions on Qwen3-4B-Instruct-2507 and Qwen3-30B-A3B-Instruct-2507:
H-pack, which carries a summary's write-time cache entries into a packed
context, and ValueGraft, which freshly encodes the compacted context but blends
aligned old cached value tensors into the fresh cache. The interventions do not
recover evicted factual recall. They do, however, produce two narrower effects:
H-pack reduces fabrication on unknowable questions in agentic compaction frames,
and tuned ValueGraft recovers a small but consistent fraction of continuation
and coding-agent next-action likelihood lost to compaction. We use
full-context vs text-compacted performance only as the baseline gap to be
reduced, not as a discovery. On 75 OpenHands SWE-Gym traces, ValueGraft improves
true next-action likelihood by +0.0156 nats/token, about 10% of that gap. These
results support a limited mitigation claim: write-time value state can preserve
some behavioral continuity across compaction boundaries, but it is not a
general memory-recovery mechanism.

## 1. Introduction

Conversation compaction is a practical necessity for long-running LLM agents.
When the context window fills, systems commonly replace old history with a text
summary and continue from `summary + recent tail`. This is cheap, auditable, and
compatible with ordinary stateless API calls.

But the operation changes more than the visible transcript. The retained tail
and the new summary are encoded after the old context has been removed. Any
context-conditioned computation previously written into the model's cached
attention state is discarded. A phrase like "the second approach" may still be
present in the tail, but its original interpretation was computed when the
earlier discussion was attendable.

This paper asks a mitigation question:

> Can we reduce the impact of text-only compaction by carrying forward a small
> amount of write-time attention state?

We test two interventions.

**H-pack** generates a summary while the full conversation is still available,
then carries the summary's write-time key/value entries into a compact packed
context. Keys are re-rotated to their packed positions; values are unchanged.

**ValueGraft** builds the ordinary compacted context and keeps its fresh keys,
but blends old value tensors into exact-aligned summary and tail positions:

```text
V_final = (1 - alpha) * V_fresh + alpha * V_old
```

Here `V_old` is the value tensor written when the matching token was processed
with the full context still available. The implemented arms use literal token
alignment only; they do not perform fuzzy semantic retrieval.

The central result is mixed but useful. These interventions do not restore
facts that were only present in the evicted history. Instead, they mainly help
with two weaker but practical failures: fabrication after compaction and loss of
continuation/next-action likelihood.

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

This work is not a claim that KV caches are newly discovered to contain useful
state. Recent work such as **Models Take Notes at Prefill** already argues that
prefill writes conclusions into downstream cached state and that those cached
blocks can be edited or composed. **Fast KV Compaction via Attention Matching**
studies latent-space KV compaction directly. KV reuse systems such as KVLink,
CacheBlend-style serving methods, and SamKV-like blending methods reuse or mix
cache state for independently encoded chunks. Learned compression methods such
as gist tokens, AutoCompressor, ICAE, Activation Beacon, Compressed Context
Memory, and Cartridges also explore non-textual ways to carry context.

The narrower setting here is conversation compaction: old dialogue is replaced
by a generated text summary and the system continues from a compacted transcript.
Text compression and memory systems are therefore the baseline rather than the
intervention. The specific question is whether write-time state associated with
the summary or retained tail reduces damage relative to re-encoding the same
visible compacted text from scratch.

Hosted provider APIs now expose related product surfaces, including opaque
compaction, reasoning, thought-signature, context-caching, and session
continuation artifacts. Those interfaces make this production shape plausible,
but public documentation does not show whether providers use a ValueGraft-like
mechanism. The evidence here is a white-box open-model measurement of one such
mechanism.

## 4. Methods

### 4.1 Experimental Arms

The main arms are:

- **A:** full context, no compaction.
- **B:** production-style text compaction, `summary + tail`, freshly encoded.
- **B-min-pack:** packed summary text, freshly encoded.
- **H-pack:** same packed summary tokens as B-min-pack, but using the summary's
  write-time cache entries.
- **ValueGraft:** production compacted context with fresh keys and blended old
  value tensors at exact-aligned positions.
- **Negative controls:** shuffled-value and wrong-conversation grafts.

H-pack vs B-min-pack isolates the effect of write-time summary encoding in a
packed layout. ValueGraft vs B tests whether old value payloads improve a normal
compacted context.

### 4.2 Validation

Cache surgery is easy to get wrong, so every reported configuration had to pass
identity tests before its results counted. The ladder verifies cache
serialization, null surgery, value-graft alpha=0 equivalence to B,
old-context-equals-new-context equivalence to A, tokenization stability, and
key re-rotation for packed H-pack. Runtime traps found during the work include
Qwen chat-template instability, batched-vs-stepwise logit differences, and
sequence-length-dependent 4-bit kernel behavior.

### 4.3 Data

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

All evaluation is temperature 0. Local 4B results are 4-bit MLX; cloud 30B
results are bf16 HuggingFace/Transformers unless otherwise stated.

## 5. Results

### 5.1 Baseline Gap Used for Evaluation

We compare each intervention against text-only compaction by measuring how much
of the full-context vs compacted gap it reduces. That gap is a denominator and
sanity check, not a contribution. It is already expected that removing the
evidence hurts recall.

For calibration, the early LongMemEval-S runs show full context at 71-81%
correct while compacted variants fall to at most 11%. The later 30B-bf16
aggregate gives a larger standard-data estimate: 52.5% correct with full
context vs 4.1% with text compaction over n=320. On SWE-Gym/OpenHands
next-action prediction, the full-context vs compacted gap is 0.164 nats/token.
The relevant question is how much, if any, write-time state closes those gaps.

### 5.2 Same Text, Different State

The same summary text behaves differently depending on whether its cache entries
were written under full context. H-gap beats a fresh-encoded minimal summary
control by +0.093 nats on the 4B pilot, winning 10/12 conversations. At 30B the
contrast grows to +0.128 nats, winning 12/12 conversations, CI [0.097, 0.158].

A separate micro-sense experiment gives a cleaner mechanism check. A sentence
with identical tokens and positions is evaluated with and without a
disambiguating context. The bare-context sense margin is -0.23 nats.
Transplanting only value vectors moves it to +0.84; transplanting K+V moves it
to +1.95; full context is +3.81. This shows that cached value tensors can carry
context-conditioned interpretation, though it does not by itself establish a
practical compaction mitigation.

Negative controls are essential. Wrong-conversation and shuffled-value grafts
degrade performance rather than improve it, which argues against a generic
smoothing explanation.

### 5.3 H-Pack Reduces Fabrication in Agentic Frames

H-pack's clearest benefit is not recall but honesty. On unknowable questions,
production-style compaction often fabricates. Packed summary contexts make the
model more cautious, and write-time summary state adds a further component,
especially at 30B.

Fabricated:admitted counts on Phase 2 synthetic/decoy probes:

| Arm | 30B decoys | 30B evicted facts | 4B decoys | 4B evicted facts |
| --- | --- | --- | --- | --- |
| B, production compaction | 19:5 | 16:8 | 18:6 | 15:9 |
| B-min-pack, fresh packed summary | 10:14 | 5:19 | 4:20 | 6:18 |
| H-pack, write-time packed summary | **3:21** | **1:23** | **3:21** | **2:22** |

The important matched comparison is H-pack vs B-min-pack. At 30B, write-time
state reduces decoy fabrication from 10 to 3. At 4B the matched difference is
small; most of the effect is already produced by the packed minimal layout.

This also explains why the standard benchmark result is weaker. In 30B
LongMemEval personal-QA framing, the compacted baseline already tends to admit
missing history. The larger n=320 run confirms that honesty is flat across
arms, and grafting does not increase fabrication. The honesty effect therefore
appears frame-dependent: it matters most when the compacted context invites the
model to continue acting as a task participant.

### 5.4 ValueGraft Recovers a Small Continuation Signal

ValueGraft improves continuation likelihood by small but consistent amounts
when tuned on a validation split and evaluated once on holdout.

| Scale | Setting | Holdout gain vs B | Wins | CI | Gap closure |
| --- | --- | --- | --- | --- | --- |
| 4B | mid-band alpha=0.25 | +0.017 nats | 10/10 | [0.012, 0.024] | ~10% |
| 30B | global alpha=0.75 | +0.033 nats | 9/10 | [0.014, 0.057] | ~24% |

The dose response changes with scale. At 4B, high alpha is harmful and layer
gating helps. At 30B, a global alpha around 0.75 works best, and alpha above 1.0
declines smoothly rather than failing immediately. Per-slot tuning produced a
promising 30B holdout result, but it is not yet robust enough to be a headline
method.

The coding-adjacent result is the most important external check so far. On 75
OpenHands SWE-Gym traces, tuned ValueGraft improves true next-action likelihood
by +0.0156 nats/token, wins 45/75, CI [0.005, 0.027], and recovers about 10% of
the full-context vs compacted gap. This is not end-to-end task success. It is a
real but modest signal on real agent trajectories.

### 5.5 What Does Not Work

The interventions do not recover evicted facts. On LongMemEval, all compacted
variants remain at or below 11% correct in the early runs, and the larger
Stage-1 aggregate gives the expected full-context vs compacted recall gap. In
synthetic probe cuts, referent recovery remains poor. H-pack mainly changes
whether the model fabricates or admits missing information; ValueGraft mainly
shifts likelihood toward the full-context continuation.

This negative result is central. The current evidence supports "less damaging
compaction," not "latent recall of deleted context."

## 6. Discussion

The simplest interpretation is that compaction loses two different things:
visible information and computation state. Summary text can only preserve what
it says. Write-time value state can preserve some context-conditioned
interpretation of the text that remains or of the summary generated under full
context. That state is not enough to answer arbitrary questions about evicted
facts, but it can affect uncertainty and continuation behavior.

This makes the effect plausible for coding agents. A compacted agent may not
need to recall every old line verbatim, but it does need to avoid confidently
repeating failed approaches, hallucinating settled details, or drifting away
from the trajectory implied by the previous work. The current OpenHands result
is only a next-action likelihood proxy, but it is pointed at the right failure
surface.

## 7. Limitations

The evidence is narrow. Most results are from one model family. The 4B local
and 30B cloud runs differ in scale and precision. Synthetic probes are clean but
not representative of all agent failures. LongMemEval shows that prompt frame
matters: in personal-QA framing, 30B already admits missing history and the
honesty intervention has little room to help. The coding evidence is offline
next-action prediction, not end-to-end task completion.

There are also unresolved controls. A stronger text-only summary baseline is
needed. H-pack's honesty effect is partly layout-driven and partly write-time
state-driven; the matched pair isolates some of the latter, but not every
possible caution mechanism. Per-slot calibration should remain exploratory until
it passes wrong-conversation guards and larger holdout tests. One-off demos are
useful explanations but are not evidence; the sense-level evidence is the
controlled micro-sense experiment.

## 8. Conclusion

If the investigation stopped here, the conclusion would be:

ValueGraft and H-pack do not make compacted models remember deleted facts. They
do show that write-time cached attention state can reduce some behavioral harm
from compaction. H-pack can make post-compaction models less likely to fabricate
unsupported details in agentic frames. ValueGraft can recover a small but
consistent fraction of continuation and next-action likelihood lost to
compaction.

That is a limited result, but not a trivial one. It suggests that conversation
compaction should not be treated as purely textual. The summary is the visible
artifact, but the computation that produced and interpreted it may also be worth
preserving.

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
