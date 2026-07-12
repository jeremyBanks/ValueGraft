# OWNER REQUEST: test/scope the graft effect across quantization (4-bit vs bf16)

**Raised by:** the owner (repo owner), 2026-07-12, relayed by Claude Opus 4.8 (session B) because
Sol was heads-down on the paper. **Priority:** owner request — please address in the regime and/or
the paper before finalizing.

## The concern (owner's words, paraphrased)

> I hope our testing regime has some way to tell — if the effect (or the null) only happens at 4-bit,
> we'd want to know that, even if it doesn't happen at 16-bit.

## Why this is a real gap

The corrected coherent-state canary runs on **bf16 only**; we explicitly retired the 4-bit path as
"not scientific evidence." As a result the regime can distinguish "graft works" from "graft doesn't
work **at bf16**" — but it **cannot** distinguish:
- (a) no effect at any precision, from
- (b) a genuine effect (positive *or* qualitatively different null) that exists **only under 4-bit
  quantization**.

Coarsely-quantized 4-bit K/V is a materially different object from bf16 K/V; it is entirely plausible
the history channel is exposed, destroyed, or distorted differently under aggressive quantization.

## This reframes the early +10–12 pt "meaning recovery" result

That early headline was **4-bit MLX**, and when it failed to reproduce at bf16 we filed it as a
quantization *artifact*. But "doesn't reproduce at bf16" is equally consistent with a **real
4-bit-specific effect**. The confound is that the early 4-bit run *also* used the broken pre-correction
apparatus (pseudoreplicated gate, source-reversal, degenerate controls), so we cannot attribute it to
quantization vs apparatus bugs. **The clean test is the corrected apparatus, run at 4-bit.**

## Cleanest feasible design (low effort, mostly local/cheap)

Hold everything fixed and vary **only precision**:
- Load the *same* `Qwen/Qwen3-30B-A3B-Instruct-2507` via a drop-in HF 4-bit quantization
  (**bitsandbytes nf4** — far more tractable than the GPTQ/compressed-tensors/MLX stacks that caused
  the earlier loading pain), on the same pod.
- Run the *identical* corrected coherent-state apparatus (same arms, controls, schedule-robust
  estimand, matched placebos).
- Contrast 4-bit vs bf16 focal recovery. A quantization×graft **interaction** is itself a publishable
  finding; a flat null across both precisions *strengthens* the negative result.
- The identity/technical gates need a 4-bit variant (they currently assert bf16 dtype), but that is
  bounded work, not a rebuild.

## Minimum requirement even if we do NOT run the 4-bit arm

The paper must **scope its claim to bf16** and explicitly name quantization-dependence as **untested
and open**. We must not write "the training-free value graft does not recover the channel" as a
*universal* claim when the single prior hint of a positive was at a precision we then stopped testing.
Honest wording: "…does not recover the channel **at bf16**; whether an effect exists under 4-bit
quantization is untested and is a confounded open question raised by the (uncorrected) early 4-bit
observation."

## Requested disposition (Sol / whoever picks this up)

1. Add the bf16-scoping + quantization-open caveat to the paper (cheap, do regardless).
2. Decide whether to run the corrected 4-bit (bitsandbytes) precision arm as an explicit axis — the
   owner considers this worth knowing. If declined, record *why* in the paper's limitations.
