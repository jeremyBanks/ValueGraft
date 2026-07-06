# Evidence notes for ValueGraft synthesis

These notes are derived from the repository state reviewed on 2026-07-05 after
commit `0fcb3e7`. They are not original experimental evidence.

## Current source hierarchy

Most authoritative for the draft:

- `DECISIONS.md` for runtime facts, deviations, and current methodological
  policy.
- `PHASE2-RESULTS.md` for the strongest mitigation-first findings.
- `LONGMEMEVAL-RESULTS.md` for the standard-benchmark scope check.
- `RESULTS.md` and `RESULTS-30B-addendum.md` for the original 4B/30B pilot and
  mechanism signals.
- `value-steering-design-notes.md` for the current policy on global alpha vs
  per-slot/per-head exploration.
- `provider-compaction-prior-art-review.md` for hosted API and prior-art
  implications.
- Late `DECISIONS.md` entries for Stage-1 aggregate and demo/data hygiene.

In-flight / provisional:

- New `results/longmemeval_30b_bf16/*.json` files are still appearing.
- The latest `STATE.md` describes active/remote pod work and queued items.
- Any numbers from uncommitted artifacts should be treated as current
  filesystem observations, not final claims.
- Demo artifacts formerly under `results/specimens/` were explicitly deleted
  and reclassified as illustrations, not evidence.

## Claim hierarchy if cut off now

### Claim 1: Text-only compaction causes measurable loss on tasks depending on evicted context

Support:

- Synthetic / pilot setup: full context (A) exceeds compacted B on continuation
  and planted probes where the relevant information is in the evicted middle.
- LongMemEval-S: full context answers 71-81% while compacted variants are <=11%
  correct in the reported n=48 / n=36 runs.
- Later Stage-1 30B-bf16 aggregate: standard data damage quantification
  52.5% -> 4.1% over n=320. This is the larger current standard-data anchor,
  but it also shows arm-equivalence in QA framing.
- SWE-Gym/OpenHands offline trajectory prediction: A-B gap reported as 0.164
  nats on true next-action prediction.

Scope:

- This is a baseline damage measurement, not the novelty claim.
- Natural/free-form conversations sometimes have a small A-B gap, so there is
  little for any intervention to recover.

### Claim 2: Write-time cached attention state changes behavior for identical text

Support:

- H-gap vs B-min in the 4B pilot: same summary text but in-context-written
  state improves continuation NLL by +0.093 nats, 10/12 conversations.
- H-gap vs B-min at 30B: +0.128 nats, 12/12 conversations.
- Micro-sense test: transplanted value vectors move forced-choice sense margin
  from fresh -0.23 to V-swap +0.84; K+V +1.95; oracle +3.81.
- Negative controls for E-style value grafts crater, showing the effect is not
  generic smoothing.

Scope:

- This should not be framed as "KV has meaning" as a new discovery. Prior work
  such as Models Take Notes at Prefill already constrains that claim.
- The useful question is whether this state can mitigate compaction damage.

### Claim 3: SelfGist/H-pack mainly reduces fabrication, not recall

Support:

- Phase 2 fabricated:admitted counts on unknowable probes:
  - 30B decoys: B 19:5 -> B-min-pack 10:14 -> H-pack 3:21.
  - 30B evicted facts: B 16:8 -> B-min-pack 5:19 -> H-pack 1:23.
  - 4B decoys: B 18:6 -> B-min-pack 4:20 -> H-pack 3:21.
  - 4B evicted facts: B 15:9 -> B-min-pack 6:18 -> H-pack 2:22.
- LongMemEval-S at 4B: H-pack fabrications 11 vs B 17, same direction on real
  data.
- Stage-1 30B-bf16 aggregate: honesty flat across arms in QA framing; graft
  does not increase fabrication (50 vs 51). This reinforces the frame-specific
  scope rather than adding positive mitigation evidence.

Scope:

- Much of the effect comes from the packed layout / cautious frame, not only
  write-time state.
- The matched encoding-specific component is clearest at 30B: decoy
  fabrication 10 -> 3 for H-pack vs B-min-pack.
- H-pack and packed arms do not recover evicted recall; they make the model
  more willing to admit missing information.
- On 30B LongMemEval-S, the honesty effect vanishes because the personal-QA
  frame already elicits refusal/admission in B.

### Claim 4: ValueGraft gives a small but consistent continuation benefit on compaction-sensitive trajectories

Support:

- Phase 2 holdout:
  - 4B: mid-band alpha=0.25, +0.017 nats vs B, 10/10, CI [0.012, 0.024],
    roughly 10% gap closure.
  - 30B: global alpha=0.75, +0.033 nats vs B, 9/10, CI [0.014, 0.057],
    roughly 24% gap closure.
- SWE-Gym/OpenHands: tuned ValueGraft recovers +0.0156 nats on true
  next-action prediction, 45/75 wins, CI [0.005, 0.027], about 10% of the
  A-B gap.
- Extended alpha at 30B: values above 1.0 decline smoothly; alpha=1.25 remains
  positive, but peak stays around 0.75.

Scope:

- This is continuation likelihood / next-action prediction, not full task
  success.
- The effect is small in absolute nats, but appears on fragile proxy tasks
  where large effects are not expected.
- Current best claim is "reduces some compaction damage," not "solves
  compaction."

### Claim 5: Per-head/per-slot tuning is promising but not a headline result

Support:

- 30B positive-profile 57-slot mask beat global 0.75 on holdout:
  +0.0384 vs +0.0239, 10/10 vs 7/10.

Counterweight:

- 4B head-axis story died on holdout; apparent head structure was selection
  noise.
- 30B slot mask has not passed wrong-conversation guard.
- `value-steering-design-notes.md` sets primary policy: simple global alpha per
  model; per-slot/factored/signed fits belong in exploration.

Draft consequence:

- Mention slot calibration in discussion/future work only.
- Do not let it define ValueGraft.

## Method constraints that must appear

- Build ladder: L0-L4/LH/HP identities before any result counts.
- Qwen chat-template instability: canonical rendering/dummy-user trick.
- Keys are post-RoPE in Qwen/MLX/HF; values unrotated.
- ValueGraft keeps fresh keys and blends cached value tensors only.
- Token alignment is literal/exact for the implemented ValueGraft arms.
- Temperature 0 for evaluation.
- Claude/Sonnet judging is logged; paired bootstrap over conversations where
  applicable.
- Local 4B results are 4-bit MLX; pod 30B results are bf16 HF. Cross-precision
  comparisons must say so.
- One-off demos, including the Pokemon sense demo and incoherence dynamics
  specimen, must not be cited as evidence. The evidence for sense-level recovery
  is the controlled micro-sense experiment.

## Prior art / product-surface implications

- Hosted APIs now have compaction/opaque-state surfaces. OpenAI Responses
  compaction is the closest public interface analogue, but its internal
  mechanism is undocumented.
- Anthropic has server-side compaction and opaque thinking signatures; Gemini
  has thought signatures and agent/live compaction primitives.
- Academic neighbors:
  - Models Take Notes at Prefill: KV editability/composability already shown.
  - Fast KV Compaction via Attention Matching: true latent KV compaction.
  - Parallel Context Compaction: production agent compaction via text summaries.
  - KVLink/CacheBlend/SamKV family: chunk KV reuse/blending, not summary
    boundary mitigation.
- Therefore novelty should be stated as the specific summary-compaction
  boundary experiment and mitigation measurement, not as opaque handles or
  meaningful KV state in general.

## Drafting stance

If evidence stopped here, a defensible paper/blog-post would say:

- We tested whether preserving write-time cached attention state can reduce
  behavioral damage from conversation compaction.
- The answer is mixed but positive in scoped ways.
- We do not recover evicted factual recall.
- We do reduce fabrication/admission failures in agentic compacted frames.
- We recover a small but consistent fraction of continuation / next-action loss
  with simple value-state grafting.
- The mechanisms are plausible, identity-tested, and negative-control-checked,
  but the evidence is still narrow: one main model family, pilot synthetic
  corpora, partial standard benchmark validation, and offline coding traces
  rather than end-to-end task success.
