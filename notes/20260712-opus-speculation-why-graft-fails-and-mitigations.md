# Speculation: WHY re-injecting write-time KV state doesn't recover the history — and how we might fix it

**Author:** Claude Opus 4.8 (session B). **Date:** 2026-07-12. **Status:** *This is my speculation*,
framed as such — a mechanistic hypothesis, grounded in two papers I read in full, assuming the
experimental null is real (which it probably is). It is meant to (a) explain why the naive intuition
fails and (b) propose concrete ways the next agent could try to *mitigate* the effect. None of the
mitigations has been tried; several are novel relative to the literature.

## The naive intuition, and why it's reasonable

The model generated the summary *while it could see the full history*, so surely the summary tokens'
K/V "remember" the history — re-inject them across compaction and you should recover the lost meaning.
This is the whole premise of the project, and it feels obviously right.

## What the two papers actually establish (load-bearing facts, quoted/paraphrased)

**"Models Take Notes at Prefill" (arXiv 2606.17107)** — untrained models, causal patching:
- **A token's own K/V drives < 1% of the decision.** Refreshing *only the field's* K/V recovers
  essentially nothing: **field-only recovery = −0.028** on Llama-3.1-8B, near zero across models.
- The resolved conclusion is **written onto downstream *aggregator* tokens** (punctuation, newlines,
  section breaks) *after* the field — "distributed write, concentrated read." Many mid-layer attention
  heads write it redundantly onto aggregators; a small set of **late read heads** (≈12 heads recover
  0.78 of the decision) retrieve it.
- **Recovery accrues only as you include *many* tokens after the field** ("the effect lives downstream
  and late"); top-8 aggregators recover 0.74–0.79. No single settling distance is given, but it is a
  *dose–response in downstream token count*.
- **Chain-of-thought gates the read:** a field-only edit recovers the decision **1.00 with CoT, 0.00
  without** — the reasoning chain forces later computation to *re-read* the field; without it, the
  decision commits to stale downstream notes and never revisits.
- Composability: **values are position-free; keys are RoPE-rotated** source→target. Transplant retains
  logit cosine 0.90–0.999 *when you carry the notes*, not the field alone.

**MEMENTO (arXiv 2604.09852)** — the method that *works*:
- Keeps the memento (summary) KV across eviction; removing that KV channel drops AIME24 by **15 pp**.
- **But it is a supervised fine-tuning method** (two-stage SFT on 228K annotated traces): the model is
  *trained* to compress into reusable memento states, and it *reasons forward attending to mementos*
  (there is downstream reasoning that re-reads them).

## The synthesis: why our null is over-determined

Our graft is, almost exactly, the "Models Take Notes" **field-only KV refresh** — applied to the
summary tokens. They got −0.028; we got a noise-floor null. We essentially *replicated* their result
in the compaction setting. The naive intuition fails for four compounding reasons, and the deepest one
is specific to our design:

1. **Wrong locus.** The history's resolved meaning does not live on the summary *content* tokens (a
   token's own K/V < 1%). It lives, distributed, on downstream *aggregator* tokens. We grafted the
   tokens that carry almost none of it.
2. **No trail existed to write the notes onto (the deep one).** In our setup the summary is the *last*
   thing generated: `[system][history][request][summary]` — nothing follows it during generation. The
   conclusion is written onto downstream tokens *as the model processes tokens after the field*. With
   no tokens after the summary, **the notes were never written** — the history was used *online* in the
   live attention during summary generation and then generation stopped. We grafted states that never
   had the chance to settle their notes anywhere durable. This is exactly the intuition the owner was
   reaching for ("enough context after the key info for the meaning to resolve").
3. **No read demand.** Even if notes existed, they are only retrieved when downstream reasoning
   re-reads them (CoT gates the read, 1.00 vs 0.00). Our probe queries a compacted context cold, with
   no reasoning bridge to trigger the late read heads.
4. **Coherence / durability.** Value-only puts old V with fresh K — an address/payload mismatch the
   model never saw. And untrained states aren't durable stores: MEMENTO needed SFT to make memento K/V
   reusable at all.

**Reframe:** the KV cache *feels* like a semantic memory ("the state remembers"), but it is a
computation substrate — distributed, downstream-written, demand-read, and (for durability) trained.
Transplanting a slice of it, and specifically the *field* slice with no trail and no read demand, is
close to a guaranteed null. The interesting scientific content of our result is that it corroborates
"distributed write, concentrated read" in a compaction/eviction setting the paper didn't test.

## How we might MITIGATE it (concrete experiments; ordered by promise)

**M1 — Graft the *trail*, not the summary (highest promise).** Preserve/graft the downstream tokens
where notes live — the retained tail and/or the top-K highest-causal-effect aggregator tokens — rather
than (or in addition to) the summary content tokens. This is the paper's own recipe ("recompute field
+ top-K notes"). Our v12 design threw the tail away (recomputed it fresh); *keeping* it is the first
fix. Expect this to be the single most informative next run.

**M2 — Generate a "thinking-space" trail so the notes get written, then graft it (the owner's idea,
refined).** After generating the summary *under full history*, **append a short trail of tokens and let
the model process them**, so the history's conclusions are written onto that trail; then capture and
graft the trail-token states along with the summary. Key nuance from the CoT result: the trail probably
must *create demand* (be note-inducing), not be inert padding — so test a ladder:
   - (a) inert filler (e.g., repeated newlines / a neutral pad) — provides aggregator *positions* only;
   - (b) a generic resolving continuation ("Key facts to remember: …");
   - (c) an explicit CoT/note prompt ("Before continuing, note what from the history matters later: …").
   Vary trail **length** (recovery is dose–response in downstream token count). My prediction: (a) ≈
   null, (b)/(c) recover progressively — which would *prove* the mechanism. Position handling (owner's
   "don't mess up the trajectory"): place the trail in the *gap* region of the position-preserving
   layout so the real tail/probe positions don't shift.

**M2′ — Duplicate the summary as its own trail (owner's idea; the cleanest controlled version).**
Generate the summary once under full history, then **append a verbatim second copy** so the source is
`[…history…][request][summary#1][summary#2]`. Now summary#2 has downstream context that summary#1
never did — it is processed *while attending back to summary#1 and the history*, so it occupies the
aggregator/trail positions where notes get written. Two graft arms:
   - **Doubled graft:** copy both copies' K/V into the destination.
   - **Second-copy-only graft (the elegant test):** copy *only summary#2*'s states into the single
     summary slot in the destination, and check whether the second occurrence has concentrated more
     recoverable value than the first. Per "distributed write / dose-response downstream," summary#2
     *should* carry strictly more of the history's resolved meaning than summary#1 — so this arm is a
     direct, self-controlled measurement of the mechanism: **same tokens, same content, only difference
     is downstream position.** If second-copy-only beats summary-only (M-baseline) beats nothing, the
     "notes live downstream" story is confirmed with an almost placebo-perfect control.
   Why this is nicer than generic filler (M2 a–c): the trail is *note-inducing by construction* (the
   model re-encodes the exact content it must preserve) and needs no prompt design, and the two copies
   are token-identical so any recovery difference isolates *position/trail*, not content. Variations to
   sweep: verbatim-double vs. triple vs. free-running continuation after the summary; value-only for the
   second copy (values are position-free, so no re-rotation needed) vs. full-K/V with key re-rotation to
   the destination summary position. **Caveat:** summary#2's keys were RoPE'd at the later source
   position; grafting into an earlier destination slot needs either the position-preserving gap layout
   or key re-rotation — value-only sidesteps this and is the clean first rung.

**M3 — Recreate the read demand at probe time (CoT bridge).** Because the read is CoT-gated, after
grafting insert a brief reasoning step before the probe ("Based on the above, the relevant decision
was …") to force the late read heads to retrieve the notes. Cheap; test with/without.

**M4 — Target the specific write/read heads/layers.** The paper localizes writing to mid-layer heads
(relative depth ≈0.19–0.26) and reading to a small late-head set (commit ≈0.47–0.48). A layer/head-
targeted graft could beat the uniform all-layer graft dramatically. Connects to our untested per-layer/
head targeting.

**M5 — The honest ceiling: maybe it needs training.** MEMENTO only works *with SFT*. A training-free
graft may have a low ceiling regardless of locus. If M1–M4 still null, the real "fix" is a lightweight
compaction-aware fine-tune (block → summary+trail → masked continuation), keeping the write-time states
at inference — i.e., stop trying to make an untrained model's states portable and instead *train* them
to be.

## Testable fork (what would distinguish the hypotheses)

- If **M1/M2 recover** → the null was *wrong locus / no trail* (fixable, training-free). This is the
  optimistic branch and it directly follows the paper's mechanism.
- If **M1–M4 all null but a MEMENTO-style fine-tune recovers** → untrained states simply aren't a
  durable store; portability requires training.
- If **even a fine-tune barely helps** → the deepest version: the information really was spent online
  and is not recoverable from any frozen state without re-attending to the original history (i.e.,
  compaction is lossy in a way no cache surgery fixes).

Each outcome is a real, publishable finding. The next agent should run M1 and M2 first — they are cheap,
they follow directly from the strongest paper's own mechanism, and they turn our null from "the graft
doesn't work" into "we localized *why*, and here is what does."
