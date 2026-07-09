# Postmortem — the honesty "cornerstone" was built on the wrong precision AND the wrong arm (2026-07-09)

*Detailed postmortem (kept in notes/, not the top-level succinct docs). Written by Claude at the
owner's encouragement, mid-incident. The deep provenance audit (notes/2026070976) will supersede any
factual claim here that it corrects — this is the "what went wrong and why" record.*

## What happened
Driving the paper to completion after the recovery headline failed reproduction, I (Claude) selected the
**honesty / anti-fabrication result** as the robust positive **cornerstone**. It has a large,
conversation-clustered-significant effect (decoy fabrication Compacted 79% → H-pack 12%, CI far from 0).
On that basis Fable re-spined the whole paper honesty-led. Two foundational problems then surfaced:

1. **Wrong precision.** The honesty result ran at **4-bit MLX (local)**, NOT bf16. The scored answers
   (`results/phase2_30b/*.json`) carry `model = mlx-community/...-4bit`. Yet the paper AND `CLAIMS.md §3`
   labeled it **bf16** — an unearned precision upgrade on the very headline. The owner's rule: **the
   cornerstone cannot be 4-bit** (4-bit = supporting evidence for scale/diversity/motivation only; 4-bit
   and bf16 are different models, not convertible).
2. **Wrong arm.** The honesty result uses the **H-pack** arm = **packed keys+VALUES**, which is a
   DIFFERENT intervention from the **value-only ValueGraft** (α_K=0, aligned positions) the paper is named
   after. H-pack was a deliberate Phase-2-pivot experiment (DECISIONS 07-05: "one clean contrast = H-pack
   vs B-min-pack") — NOT discarded — but the 07-06 (α_K, α_V) reframing froze the historical arm IDs, and
   H-pack sits at a corner of the space (packed keys+values) that is not the value-only method.

So the proposed cornerstone was **doubly disqualified: wrong precision AND wrong arm.** Meanwhile the only
on-disk result that is BOTH bf16 AND on the actual value-only method appears to be the small SWE-Gym
E-tuned +0.0156 nats (single render, untested for render-robustness).

## Root causes
1. **Split-brain runtime** (local MLX 4-bit vs pod HF bf16): the core behavioral evals (recovery + honesty)
   are MLX-only pipelines (`run_arms.py`, phase2) → they ran LOCAL at 4-bit; the pods ran bf16 on DIFFERENT
   harnesses (SWE-Gym, cross-arch, LME). The "bf16" the paper assumed was never true for the core evals.
2. **Arm-vocabulary reframing** (07-06 α_K/α_V, "historical IDs frozen in code") created a gap between the
   arm IDs in the data (H-pack) and the current canonical method (value-only). Easy to conflate "write-time
   KV" (H-pack, packed keys+values) with "ValueGraft" (value-only).
3. **Claude treated a large significant number as cornerstone-eligible without verifying (a) its precision
   and (b) that its arm is the current method.** The CLAIMS ledger recorded a bare "bf16" label with no
   precision-provenance check, and I built on it. Effect size is not validity.

## How it was caught
- The **adversarial review pass** traced the phase2 answers to the 4-bit render and flagged the bf16
  mislabel as a blocker.
- The **owner's recollection** that H-pack was a reframed/different arm from the current method.

## Lessons (durable)
1. A **cornerstone must be verified against the current experiment's canonical intervention (arm identity)
   AND its claimed precision/hardware**, from the on-disk model/dtype fields, BEFORE it is used as a headline.
2. The provenance ledger (CLAIMS.md) must record **precision (quant) + arm-identity + hardware explicitly
   per result**. A bare "bf16" is a landmine.
3. **Effect size ≠ validity.** A huge significant effect on the wrong arm at the wrong precision is
   supporting/adjacent evidence at best, never the cornerstone.
4. **Split-brain results are non-comparable and non-convertible** (local 4-bit ≠ pod bf16; 4B ≠ 30B).
   Never merge or relabel across the boundary.
5. **Sub-agent management:** when a correction is coming, tell running agents to HOLD so they don't
   over-commit to a superseded line (I did this poorly here — the paper-reviser kept going after the
   foundation was already in question).

## Status at time of writing
Construction HALTED. Deep provenance+validity audit running (notes/2026070976). Paper-reviser Fable told to
hold. The bf16 H-pack answer set (`results/honesty_30b_bf16/`, bf16 on a pod, unscored) will be analyzed as
SUPPORTING evidence (if valid) — not assumed cornerstone. The paper will be re-anchored on a bf16 +
value-only result if one exists, or honestly reframed as a cautionary/negative result if not. More data
analysis — or more experiments at bf16 on the value-only method — will be done if the audit says they're needed.
