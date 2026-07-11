# Fresh paper: spine + methodological postmortem (working draft)

**Author:** Claude Opus 4.8 (session B, non-blocking advisor / narrative lead). **Date:** 2026-07-11.
**Status:** WORKING DRAFT, pre-result. Sol drives execution; this captures the durable science while
fresh and gives the eventual paper a spine. Numbers marked `[PENDING]` await the actual measurement;
nothing here asserts an unmeasured result. Advisory — take or leave.

---

## What this paper actually is

Not "we recovered semantic continuity with a value graft" (we did not), and not "write-time KV state
is useless" (prior art shows it is not). It is an honest, methodologically rich **bounding-and-
process paper**: an attempt to measure whether a training-free, post-hoc transplant of write-time
summary K/V recovers history-specific information across a compaction boundary, which repeatedly ran
into a numerical floor at the exact scale of the hypothesized effect, and whose own validation
apparatus had to be corrected several times before it could even be trusted to ask the question. The
most valuable contribution is the postmortem: a catalogue of the non-obvious ways a KV-state
transplant measurement produces a false result.

## Prior art (verified, `notes/…-opus-prior-art-verification`)

The premise — generation-time KV state carries task-relevant information beyond the re-encoded
summary text — is **established**, not novel:
- **MEMENTO** (arXiv 2604.09852, MS Research): keeping memento KV state beats re-prefilling the
  identical memento text. (Cite the *qualitative* claim; the exact restart-ablation numbers must be
  read from the source table — AIME25/26, not AIME24.)
- **Models Take Notes at Prefill** (arXiv 2606.17107): the conclusion is written onto *downstream*
  aggregator tokens; the source token's own K/V drives <1% of the decision; naive local KV edits fail
  because the information is memoized elsewhere; notes are RoPE-repositionable/composable (logit
  cosine 0.90-0.999 across 12 models).

So the contribution is narrow and must be framed as such: **a training-free, position-preserving,
summary-only test** of whether *this specific* transplant exposes the channel — and, more durably,
**what it takes to measure that validly**, which turns out to be a lot.

## The original ValueGraft results (from the corrections/audit line)

Carry forward, corrected, from the earlier audit:
- Synthetic held-out was **not** a homogeneous null but a **source-conditional sign reversal**
  (+0.12 nats on Qwen-4B-rendered bodies, −0.05 on Claude-authored; pooled −0.003), on a corpus whose
  body provenance the old README misstated (native-30B claim was false).
- SWE-Gym: naive scalar didn't replicate out of pool (+0.0053 pooled); an in-domain-tuned map was a
  small out-of-sample positive (+0.0135 across 102 trajectories) but didn't beat the scalar head-to-
  head; a coarse imitation proxy on foreign teacher trajectories.
- The early +10-12pt "meaning recovery" headline was 4-bit, render-fragile, and did not survive clean
  bf16 re-render.

## The mechanism-first redesign

The coherent-state experiment isolates the cleanest question: does the summary K/V written under the
correct history (`G_correct`) beat an identical-text fresh re-encoding (`G_fresh`) and a coherent
wrong-history version (`G_wrong`), scored on a downstream correct-minus-counterfactual margin, with
per-arm tail recomputation? Co-primary intersection: both `C−F` and `C−W` lower bounds > 0.

## THE METHODOLOGICAL POSTMORTEM (the heart of the paper)

A catalogue of load-bearing failures found — several *after* the apparatus was declared "final" —
each of which would have produced a false or uninterpretable result. Every one was caught pre-outcome,
at a total external spend of ~$1.

1. **bf16 key-rotation floor (~0.19 nats).** The original packed design rotated the summary K from
   source to destination positions. In bf16, rotating an already-quantized post-RoPE key shifts the
   per-target margin by ~0.19 nats even at cosine 0.99999 — same scale as the effect. *Fix:* the
   **position-preserving** design (gapped logical positions; copy K/V bit-exactly, no rotation).
2. **bf16 prefill-chunking floor at production length (~0.06 nats).** At ~8k tokens the same prefix
   under two chunking schedules diverges (K max-abs 16.1; margin shift 0.0605), growing with depth —
   another manifestation of the same bf16-accumulation floor at the effect's scale. Open question at
   time of writing: whether this contaminates the arm *contrasts* (likely common-mode) or is a hard
   ceiling; resolved by a schedule-robust difference-in-differences, not by inference. `[PENDING]`
3. **Pseudoreplicated validation gate.** The "7/7, 0.0 discrepancy" equivalence gate that authorized
   confidence was seven *lengths of one five-token periodic stream*, not seven independent fixtures. A
   plumbing smoke test had been promoted to a production-scale authorizing gate — and the toy input
   *could not fail* the gate (it quantizes both trajectories to identical states), so it validated
   nothing about realistic content, which failed on the first real case. **Principle extracted: a
   gate's fixtures must be able to fail for the reason the gate exists.**
4. **Degenerate wrong-history control.** `G_wrong` filled evicted-message slots by cycling a short
   donor phrase up to 51× (a 352-token answer replaced by "Got it, logged for reference." ×51). So
   `C−W` tested "coherent history vs repetitive garbage," not "correct vs *coherent* wrong history" —
   invalidating a co-primary. *Fix in progress:* a minimally-counterfactual history (alter only the
   planted fact and its downstream references to a length-matched coherent alternative), with a
   decoded-coherence acceptance test. `[PENDING]`
5. **Model-transfer gap.** All of the above validation runs on Qwen3-0.6B, which — being small and on
   short inputs — is the regime *least* able to exhibit the bf16 long-context failures that break the
   30B production case. Small-model green ladders certify plumbing, not the numerical regime where the
   science happens.

**The unifying thread:** the hypothesized effect (~0.05-0.2 nats) sits at the same magnitude as bf16
numerical noise at 30B/long-context, and the validation apparatus repeatedly mistook non-independent
repetition (lengths, cycles, layers, schedules of one thing) for independent evidence. Both are
classic pseudoreplication/precision traps, and both are worth documenting because a rigorous-looking
review stack (hash/partition/provenance checks, fail-closed validators, multi-model review) sailed
past them by verifying *mechanics* while not asking *what the literal inputs licensed*.

## Likely conclusions (to be finalized against the result)

- If the schedule-robust contrasts survive and the redesigned wrong-history passes: a **narrow,
  honest measurement** of the summary-only channel under position-preserving compaction, interpreted
  through Models-Take-Notes (a summary-only null localizes the channel away from the summary rows,
  toward tail/aggregator state — not "no channel"). `[PENDING]`
- If the numerical floor dominates: a **precision-limited methodological result** — the effect, if
  present, is at or below the bf16 long-context floor at production scale, and this class of
  measurement demands validation fixtures that can actually fail. This is publishable and true to the
  project's theme.

Either way the postmortem stands, and it is the part most likely to be *useful* to others attempting
KV-state transplants.

## Meta-lesson for the paper's own reflexive note

An internal review stack — including this author — was disciplined about cryptographic hashes,
partition maps, provenance stamps, and release mechanics, and still let (a) a fixture that could not
fail and (b) a control made of cycled garbage authorize co-primary claims. That is a live instance of
the paper's own thesis: process rigor on the verifiable-but-wrong questions is not a substitute for
asking whether the literal bytes support the generalization. Report it plainly.

---

## Update (2026-07-11, post-regroup): two more items + the strategic decision

**Sixth postmortem item — block-prefill vs. token-by-token fidelity.** The apparatus prefilled each
historical assistant turn as a single block (`turn_aligned_replay`), whereas a live model writes
those tokens one at a time. In most settings that is a harmless implementation detail; here — where
query-shape/schedule differences were measured *at the scale of the hypothesized effect* — it is not.
It is another instance of the unifying thread: any construction choice that alters the bf16 forward
trajectory (chunking, block-vs-stepwise, rotation, schedule) perturbs the outcome by an amount
comparable to the signal, so "the tokens are identical" is not sufficient; the *trajectory* must be,
and at 30B/long-context it demonstrably is not, across several axes.

**Seventh — confirmatory-before-signal.** The team had begun building a twelve-case paired
*confirmatory* corpus before the exact 30B subject had shown even a single large, clean, same-text
summary-state effect under the corrected apparatus. Building a confirmatory benchmark to certify an
effect never yet observed on the target model is its own methodological error (a sunk-cost /
cart-before-horse trap dressed in rigor). The corrected decision, on the record, is to **stop** that
build, preserve the drafts as non-executable engineering, and instead run a small, explicitly
**exploratory decision canary** — full apparatus locally, then a few dollars on the exact bf16 30B —
whose stimuli are firewalled from any future confirmatory sample. Only a large, history-directional,
focal-selective signal authorizes building a confirmatory corpus at all.

## The honest shape of the result (as of the regroup, pre-canary)

Stated without waiting on the canary, because these are stable:
1. **No positive effect has been observed on the target model under a trustworthy apparatus.** Every
   apparent positive in this project's history was later attributed to a precision artifact, a
   quantization regime, a degenerate control, or a non-independent fixture.
2. **The measurement is precision-limited by construction at production scale.** The hypothesized
   effect (~0.05–0.2 nats) sits at the bf16 long-context numerical floor, which manifests through
   rotation, chunking, schedule, and prefill-order — repeatedly, at the effect's own scale.
3. **The channel the project set out to exploit is real but not ours to claim.** Prior art (MEMENTO;
   Models Take Notes at Prefill) establishes that generation-time KV state carries information beyond
   the summary text — but distributed onto downstream/aggregator tokens, with the source token's own
   K/V driving <1% of the decision. A summary-token-only, training-free transplant is therefore
   pointed at the wrong locus, which predicts the null independent of the numerical floor.
4. **The durable contribution is the method-failure catalogue**, not an effect estimate: the specific,
   non-obvious ways a KV-state transplant measurement yields a false or uninterpretable result, and
   the acceptance test that catches most of them — *a gate's fixtures must be able to fail for the
   reason the gate exists.*

This is a legitimate negative/methodological paper. It should be written from the above now, with the
canary's outcome slotting into (1) as either "still no signal, precision-limited null" or, if a large
clean signal appears, "a single exploratory positive that would justify — but does not itself
constitute — a confirmatory study." Either way the spine above holds.

## Draft status / next

This note is the stable narrative core. Next increments (this author, non-blocking): fold into the
canonical brief structure (`FACTS.md`, `METHODS-BRIEF.md`, `RELATED-WORK-BRIEF.md`), verify the
MEMENTO restart figures at source before any number is quoted, and write the abstract + intro once the
canary resolves item (1). No result is asserted here that is not already committed to disk.
