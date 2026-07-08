# Refocusing and Reframing the Semantic Continuity Experiment

## Executive summary

The strongest framing for this project is not simply that re-encoding visible
text after compaction differs from preserving the original KV state. That is
true, but it can sound self-evident: hidden activations differ when the model
sees different histories.

The more valuable framing is mitigation-first:

> Can a compacted conversation preserve more of the model's prior interpretation
> by retaining or transplanting a small amount of write-time KV state, instead
> of relying only on a text summary?

Under this reframing, the project is not mainly a philosophical demonstration
that "meaning lives in KV." It is an intervention study: can cache-state methods
reduce the semantic damage caused by conversation compaction, under controlled
leakage checks and negative controls?

## Why refocus

The original theory remains useful, but the practical contribution should be
centered on reduction of harm:

- Text-only compaction is already known to be lossy in practice.
- It is unsurprising that a model's internal state changes when old context is
  removed.
- What is not obvious is whether any of the lost continuity can be recovered
  cheaply, selectively, and reliably.
- A mitigation result would be directly relevant to agent systems, coding
  assistants, long-running chats, and production context management.

The project should therefore avoid making the headline claim:

> KV cache contains semantic meaning that re-encoding loses.

and instead prefer:

> Write-time cache state can be used as a compact intervention that reduces
> semantic continuity loss after text-only compaction.

That version is more testable, less metaphysical, and more useful.

## Revised research questions

1. Does standard text-summary compaction degrade the model's behavior on probes
   whose answers depend on context-dependent interpretation rather than visible
   text alone?

2. Can retaining or transplanting a small amount of write-time state reduce that
   degradation compared with a text-only compacted baseline?

3. Which part of the retained state carries useful mitigation signal?

   - The summary's write-time encoding: `H` vs `B-min`.
   - The retained tail's old values: `E` vs `B`.
   - Old positional addresses and keys: `C` vs `E`, and `C` vs `B-causal`.
   - Attention-based correspondence for paraphrased compacted text: future `G`.

4. What are the limits?

   - Does the effect survive low-leakage probe classes?
   - Does it disappear under shuffled or wrong-conversation graft controls?
   - Is it robust beyond one model size and one corpus?
   - Is the cost small enough to matter in realistic systems?

## Suggested claim hierarchy

The writeup should distinguish three levels of claim.

### Primary claim: mitigation

The most important claim would be:

> A cache-state intervention reduces compaction-induced semantic errors on
> leakage-controlled probes, relative to a standard text-summary baseline.

This is the claim that makes the work useful.

Evidence that would support it:

- `E-post` at a low alpha beats `B` on clean probes.
- `H-gap` or a later `H-pack` beats `B-min` on clean probes.
- Gains are not reproduced by `E-shuffled` or `E-wrongconv`.
- The effect appears in categories such as referent, sense, stance, or ruled-out
  decisions, not only in continuation fluency.

### Secondary claim: mechanism

The mechanism claim should be phrased carefully:

> Some write-time value states carry context-conditioned information useful for
> later interpretation of compacted conversations.

This is stronger than "activations differ" but weaker and cleaner than "meaning
lives in values."

Evidence that would support it:

- Low-alpha value grafting improves clean probe behavior.
- K+V or gapped retention behaves differently from value-only grafting.
- Wrong-content and shuffled grafts fail.
- The micro-sense experiment shows that value swaps shift disambiguation margins
  in the expected direction.

### Boundary claim: limits and costs

A null or brittle result is still publishable if framed as a boundary:

> Write-time state contains recoverable signal, but naive transplantation is
> fragile; useful deployment likely requires careful gating, selective span
> choice, or retrieval-based correspondence.

This would still be valuable because it prevents overconfident cache-retention
stories and clarifies what simple methods cannot do.

## Reframed role of each arm

The arms should be presented as mitigation tests, not just ontology probes.

| Arm or contrast | Reframed purpose                                                                            |
| --------------- | ------------------------------------------------------------------------------------------- |
| `A`             | Oracle: no compaction damage.                                                               |
| `B`             | Production text-summary baseline.                                                           |
| `B-causal`      | Text-only control matching `C`'s tail-then-summary order.                                   |
| `E-post`        | Practical value-state mitigation with fresh keys and contiguous cache layout.               |
| `E-inter`       | More invasive value-state mitigation; useful if post-prefill grafting is too weak.          |
| `H` vs `B-min`  | Cleanest tiny-state test: same summary text, different write-time encoding.                 |
| `C`             | Upper-bound-ish retained-cache comparison, but confounded by gapped positions and old keys. |
| `D`             | Tests whether retained tail state alone carries signal without summary state.               |
| `E-shuffled`    | Negative control: alignment should matter.                                                  |
| `E-wrongconv`   | Negative control: content should matter.                                                    |
| Future `G`      | Generalizes value grafting when compacted text paraphrases rather than repeats.             |

The center of gravity should move toward `E`, `H`, and negative controls. `C` is
still informative, but it should not carry the main practical claim because it
is harder to deploy and varies more factors at once.

## Metrics under the new framing

The metric hierarchy should match the mitigation claim.

### Headline metric

Probe accuracy by category and leakage class should be the headline.

The most important cuts are:

- `absent-from-summary`
- `evicted-only`, mainly as a leakage and laundering control
- non-tail-visible plants

Useful categories are:

- referent resolution
- sense disambiguation
- stance and preference retention
- ruled-out approach avoidance

If a cache intervention only improves probes whose answers are explicit or
paraphrased in the summary, it is not evidence for mitigation beyond text.

### Supporting metric

Continuation logprob should be secondary.

It is useful as a broad degradation measure, but it can reflect fluency, style,
template fit, or order effects. It should support the behavioral probe story,
not replace it.

### Negative-control requirement

Any claimed improvement from value intervention should be interpreted only
relative to:

- correct graft vs shuffled graft
- correct graft vs wrong-conversation graft
- preferably low alpha vs high alpha curves

If wrong or shuffled grafts improve too, the result is not a semantic continuity
mitigation result.

## Practical deployment story

A mitigation-first paper or memo should explain what a system might actually do.

### Path 1: SelfGist

Generate a summary in the full conversation, then retain a small number of the
summary's write-time KV states as compact memory tokens.

Why it matters:

- It is simple.
- It requires retaining only summary tokens, not the whole conversation.
- `H` vs `B-min` is a clean test of whether the same summary is more useful when
  encoded in-context.

Main risk:

- The summary may not absorb enough latent context to matter.
- If it works only with gapped positions, a packed/re-rotated version is needed
  for real deployment.

### Path 2: ValueGraft

Build the compacted context normally, then blend old value states into aligned
positions where compacted text or retained tail text has exact token twins.

Why it matters:

- It keeps fresh keys and a contiguous cache.
- It avoids gapped-position attention.
- It can be represented as a small cache-edit operation after prefill.

Main risk:

- Fresh keys paired with old values may be incoherent.
- Strong alpha may hurt, making gating or low-dose blending necessary.

### Path 3: SoftGraft

Use attention over the old cache to retrieve value blends for new compacted
tokens, including paraphrases.

Why it matters:

- Exact token alignment is too restrictive for real summaries.
- Attention is the model's native retrieval mechanism, avoiding arbitrary
  value-space nearest-neighbor assumptions.

Main risk:

- Retrieval may be diffuse or lost-in-the-middle.
- It needs entropy gating and more validation before it can be trusted.

## Suggested revised abstract

Long-running LLM conversations are often compacted by replacing old history with
a text summary and re-encoding the retained turns from scratch. This changes not
only what text is visible, but also the model's internal interpretation of
retained text whose meaning depended on the discarded history. We study whether
this compaction damage can be reduced by preserving small amounts of write-time
KV cache state. Across controlled conversation probes, we compare standard
text-summary compaction with cache-state interventions: retaining in-context
summary states, transplanting old value states onto freshly keyed compacted
contexts, and gapped retention of original cache entries. Negative controls use
shuffled and wrong-conversation grafts, and all probe results are stratified by
summary leakage. The goal is not merely to show that hidden states differ after
compaction, but to test whether write-time state provides a practical, compact
mitigation for semantic continuity loss.

## Suggested revised title options

- Preserving Semantic Continuity Across Conversation Compaction
- Cache-State Mitigation for Long-Context Compaction Loss
- ValueGraft: Reducing Compaction Damage with Write-Time Value States
- SelfGist and ValueGraft: Compact KV-State Interventions for Conversation
  Continuity

## What would count as success

The strongest pilot result would look like this:

1. Standard compaction `B` fails clean referent, sense, stance, or ruled-out
   probes that full context `A` answers.
2. `H` or low-alpha `E-post` recovers a meaningful fraction of those failures.
3. `E-shuffled` and `E-wrongconv` do not recover them.
4. The effect appears in `absent-from-summary` plants, not only in explicit or
   paraphrased summary plants.
5. Continuation logprob is consistent with the probe result but not the sole
   evidence.

That would justify a claim that cache-state methods can reduce compaction
damage.

## What would still be useful if results are weak

Weak or mixed results are still informative under the refocused framing:

- If `H` does not beat `B-min`, in-context summaries may not function as
  training-free gist tokens in this setting.
- If low-alpha `E` helps but high-alpha `E` hurts, value states carry useful
  signal but require careful gating.
- If `C` hurts while `E` helps, gapped old-key retention may be too OOD even
  though values contain recoverable signal.
- If only continuation NLL improves but clean probes do not, the intervention
  may improve style or fluency rather than semantic continuity.
- If negative controls improve, the value-graft metric is contaminated and
  should not be interpreted as semantic recovery.

These outcomes would still shape the next experiment.

## Recommended next-step emphasis

Before expanding to more arms or larger models, the refocused project should
finish the current pilot cleanly:

1. Repair missing `B-causal` continuation scores.
2. Complete external judging and paraphrase leakage checks.
3. Aggregate probe accuracy by category, arm, and leakage class.
4. Put `H` vs `B-min`, low-alpha `E` vs `B`, and negative controls at the center
   of the interpretation.
5. Treat 30B or future `G` runs as follow-ups only if the clean pilot result
   shows a mitigation signal.

The key decision is whether the current evidence supports a practical
intervention story. If it does, scale and coding-agent extensions become much
more compelling. If it does not, the project should report the boundary
conditions honestly and avoid overclaiming from hidden-state non-equivalence.

## One-sentence reframing

This project should be framed as testing whether small, controlled cache-state
interventions can reduce semantic damage from conversation compaction, not
merely as showing that re-encoding text loses hidden state.
