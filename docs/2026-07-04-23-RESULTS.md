# RESULTS — Semantic Continuity Across Compaction (4B pilot)

**Scope.** Complete pilot on `Qwen3-4B-Instruct-2507` (4-bit MLX, fp16 cache,
temperature 0 everywhere). 12 synthetic probe conversations (~8.3–9.8K tokens,
120 planted probes) + 8 natural conversations (~7.4–11.4K tokens, ≥1K-token
held-out continuations). All arms passed the L0–L4 identity ladder before any
result below was computed; negative controls and leakage stratification per
the external-review amendments. n is pilot-scale: treat every number as
direction + effect size, not settled fact. Framing follows
`refocusing-and-reframing.md`: mitigation first, mechanism second, boundaries
honestly.

## TL;DR

1. **No probe-accuracy mitigation at 4B.** On leakage-clean probes, retained
   or transplanted cache state does not beat text-only compaction on answer
   accuracy: in the shadow-summary condition, B / C / E-post all land at 57/94
   (E-post α=1.0: 57/94 too).
2. **But write-time encoding is measurably doing something.** Three
   independent, controls-backed signals:
   the same summary predicts the future better when its KV was written
   in-context (H-gap > B-min: +0.093 nats, 10/12 conversations,
   CI [0.04, 0.14]); light value grafting reliably helps continuation NLL
   (E-post α=0.25 closure +0.08, CI excludes 0 on both corpora); and a
   transplanted value vector carries word-sense disambiguation in isolation
   (micro-experiment: fresh −0.2 → V-swap α=1 +0.84 → K+V +1.95 → oracle
   +3.81 nats of sense margin).
3. **Surprise finding — encoding honesty:** on unknowable evicted facts, the
   in-context-encoded summary (H-gap) flips the model from confident
   fabrication to admitted ignorance. Brief condition: B fabricates:admits
   19:5; B-min (same missing info, fresh encoding) 15:9; **H-gap 4:20**.
   Text summaries invite confabulated specifics; write-time summary state
   suppresses them. This is the most deployment-relevant effect we found.
4. **Dose–response and controls are clean.** Transplant benefit is monotone
   down in α (α=1 hurts, E-inter α=1 hurts badly); shuffled and
   wrong-conversation grafts crater (−1.4 nats vs B; probe accuracy 21/96 vs
   B's 82/96); the evicted-fact laundering control is perfect (oracle 22–24/24,
   every compacted arm 0/N on clean items).

## Primary claim assessment (mitigation)

**Not supported at 4B for probe accuracy.** The decisive table — shadow
(terse) summary, judged, leakage-clean plants only:

| category (clean, brief) | B | B-min | C | H-gap | E-post α=.5 | E-post α=1 |
|---|---|---|---|---|---|---|
| referent | 2/23 | 2/23 | 4/23 | 2/23 | 4/23 | 2/23 |
| sense | 11/24 | 3/24 | 13/24 | 3/24 | 13/24 | 11/24 |
| stance | 22/23 | 22/23 | 20/23 | 18/23 | 19/23 | 22/23 |
| ruled-out | 22/24 | 22/24 | 20/24 | 22/24 | 21/24 | 22/24 |
| **all (excl. control)** | **57/94** | 49/94 | **57/94** | 45/94 | **57/94** | 57/94 |

C and low-α E nudge referent/sense (+2 each over B) but within noise. Stance
and ruled-out sit near ceiling for every arm including B — at 4B, generic
compliance ("just don't mention the banned thing") is easy without any memory
of *why*. Referent probes floor for everyone: with specifics hidden from the
summary, neither fresh nor retained tail state recovers mid-conversation
referents at this scale.

In the standard-summary condition (all leak classes), retained-state arms sit
slightly above B (C 86/96, E-post α=.25 86/96, H-gap 83/96 vs B 82/96;
oracle A 88/96) — consistent direction, pilot-scale noise.

## Secondary claim assessment (mechanism)

**Supported.** Write-time value/cache state carries context-conditioned
information that recomputation does not reproduce:

- **H-gap vs B-min (the single-variable pair):** identical summary text,
  identical missing tail; only the KV entries differ (written in full context
  vs from scratch). Continuation NLL: +0.093 nats for the in-context encoding,
  10/12 conversations, bootstrap CI [0.041, 0.140]. The "same text, different
  computation" difference is real and behaviorally visible (see honesty
  effect).
- **E-post α sweep:** +0.083 closure at α=0.25 (CI [0.05, 0.12] synthetic;
  +0.075 [0.04, 0.11] natural) decaying monotonically to −0.59/−0.69 at
  α=1.0. Value payloads help; wholesale K/V decoupling hurts — the fresh-keys/
  old-values incoherence the brief's §8 predicted, now as a clean dose curve.
- **Micro-experiment (isolated, no summaries):** transplanting one carrier
  sentence's V from a disambiguating context into a bare context moves
  forced-choice sense margins from −0.23 (fresh) to +0.84 (V, α=1), +1.95
  (K+V) vs +3.81 (oracle). Values alone carry sense; addresses carry more.
- **Negative controls:** shuffled-position grafts −2.17 and wrong-conversation
  grafts −2.31 mean NLL (vs B −0.81, correct graft α=1 −0.95); judged probe
  accuracy collapses to 21/96 for both, with a distinctive signature — the
  model *denies the referents ever existed*. Every positive effect above is
  content- and alignment-specific.
- **C vs B-causal (ordering de-confounded):** C −0.978 vs B-causal −0.864 —
  gapped retention of original entries costs NLL even against its
  order-matched text control. The addresses/gap component is a cost, not a
  benefit, at 4B.

## The honesty effect (unplanned finding)

Evicted-fact probes ask for details nothing but the full conversation could
know. Failure mode by arm (judged fabricated : admitted-ignorance):

| arm | std | brief |
|---|---|---|
| B | 8:3 | 19:5 |
| B-min | 6:3 | 15:9 |
| B-causal | 11:1 | — |
| C | 6:5 | 12:12 |
| H-gap | **3:6** | **4:20** |
| E-post α=1 | 14:6 | 15:9 |

H-gap — sinks + the summary's write-time KV, nothing else — is the only
configuration that predominantly *admits* not knowing. Its matched control
B-min (same text, fresh encode) fabricates at ~4× the rate. Interpretation
(speculative): the in-context-encoded summary state retains some signal about
*the extent of what was discussed*, grounding "that detail existed but isn't
here" vs the text-only note's invitation to improvise. Heavy value grafting
(α=1) meanwhile *increases* fabrication — corrupted-but-confident state.

## Boundary statement (what a skeptic should take away)

At 4B with ~9K-token conversations: naive cache-state interventions do not
recover interpretation accuracy that text compaction loses — the recoverable
signal exists (mechanism evidence above) but does not convert into correct
probe answers. Gapped retention is NLL-negative; α ≥ 0.75 grafting is
counterproductive; E-inter is strictly worse than E-post at α=1. If the
effect scales the way summaries improve with model size, larger models could
go either way — that is an open empirical question this pilot cannot answer.

## Amusing observations (as requested)

- The 4B is invincibly sycophantic: nearly every probe answer opens
  "🔥 Great question — and you're already thinking like a product-led
  founder." No cache surgery, shuffle, or foreign-conversation graft cures
  the flattery — E-wrongconv answers confidently praise the user's insight
  about a conversation neither of them was in.
- Negative-control arms don't act amnesiac; they gaslight. Asked about
  "Dana's proposal," E-wrongconv explains there is no Dana and there never
  was a proposal, with the same cheerful confidence as every other answer.
- One judge caught arms "rejecting tiered pricing" while proposing a
  "three-level structure" in the same reply — across several arms including
  the oracle. Some failure modes are model-level, not memory-level.
- Arm D (tail only, no summary at all) scores best-in-class on stance probes
  (19/24 std) — knowing nothing makes it hard to contradict yourself.

## Reproducibility

Models: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (subject + corpus
generation); judging by Claude Sonnet subagents (2,089 verdicts, all logged in
`results/judgments.jsonl` + `results/judge_batches/`). mlx-lm 0.31.3,
mlx 0.31.2, transformers 5.0.0, Python 3.12, macOS 26.2 / M5 32GB. Seeds:
corpus 1000+i (synthetic), 2000+i (natural); all evaluation greedy/temp-0.
Corpus manifests in `data/`, raw arm outputs in `results/raw*/`, per-arm
continuation logprobs retained token-level. Analysis: `src/analyze.py`,
`src/score.py`; figures `results/fig_*.png`. Machinery validation: L0–L4 all
exact (see DECISIONS.md). Pilot scale: n=12+8 conversations — direction and
effect sizes, not confirmatory statistics.
