# Amendments from External Review — Companion Document 4

**Context:** an independent model reviewed the experiment plan (documents 1–3) and produced a detailed critique with a deeper literature search than ours. This document distills what survived our evaluation of that review. As with document 3, it was written **without visibility into your actual progress or results** — and, importantly, it arrives while you have work in progress. Nothing here asks you to discard or restart anything.

**How to use this document:** each item is tagged with *when it bites*:
- **[before-results]** — should be in place before you treat any cross-arm numbers as findings (usually a small addition to the harness, not a redo).
- **[analysis-time]** — affects how existing results are scored, sliced, or reported; can be applied retroactively to runs you've already done if raw outputs were logged.
- **[phase-2]** — only matters for the scale-up / coding phase.
- **[writeup]** — only matters when drafting the report or paper.

If an item conflicts with something you've already built and validated, your judgment wins; log the divergence in `DECISIONS.md` as usual.

---

## 1. New control arm: B-causal **[before-results]** — the single most important amendment

Arm C has a confound we built in without noticing. In C, the summary is generated at the end of the full conversation, so the retained state's causal order is *tail → summary*. In B, the rebuilt context is *summary → tail*. C-beats-B is therefore ambiguous between "write-time KV was preserved" and "summary-after-tail ordering is simply better."

**Fix:** add **B-causal** — a fresh prefill of `system + tail + summary` (same texts as B, order matched to C). Then:
- **C vs B-causal** is the clean mechanism claim (same causal order, only the encoding differs).
- **B vs A** remains the production-realism baseline.
- **B vs C** is reported only as a deployment-flavored contrast, never as the mechanism result.

Same logic, applied to H: **H-pack vs B-min** is the position-matched pair (same text, same packed positions, only write-time encoding differs). H-gap vs B-min additionally varies positional layout; report it, but H-pack carries the causal claim.

Cost: one extra text-level arm — no new surgery machinery. If you have completed runs, B-causal can be run afterward on the same materials and paired in.

## 2. Negative controls for value interventions **[before-results]**

The identity tests (L-ladder) prove the machinery does nothing when it should do nothing. They do not prove it isn't helping *for the wrong reason* — e.g., value-blending acting as distributional smoothing that improves NLL regardless of content. Before any E/G gain is called a finding, run at minimum:

- **Wrong-conversation graft:** transplant values from a *different* conversation's aligned-length positions. If this improves any metric, that metric is contaminated.
- **Shuffled-value graft:** correct conversation, values permuted across positions within layer/head. Tests whether alignment matters at all.
- Optional but valuable: **wrong-span same-conversation graft** (topical-but-misaligned), **K-only graft** (old keys re-rotated, fresh values — the complement of E; needs the re-rotation utility), and **K+V-old packed** (completes the factorization).

These are cheap (reuse the E machinery with different source indices) and they are what makes an E result credible to a hostile reader.

## 3. Arm G: define the GQA aggregation rule before coding it **[before-results, G only]**

The model has 32 query heads sharing 4 KV heads. G injects one blended old-value per *KV-head* slot, but retrieval produces up to 8 attention patterns per slot (one per query head in the group). The brief never said how they combine. Decide and document before implementing. Reasonable candidates: average the group's attention distributions; entropy-weighted average (sharper heads count more); lowest-entropy head only. Suggested default: entropy-weighted average, with per-slot entropy gating as already specified. This is a real degree of freedom, not a detail — different rules can retrieve different values on exactly the ambiguous-referent probes we care about.

## 4. Leakage classifier and reporting by class **[analysis-time; retroactive]**

Upgrade the leakage audit from "reclassify contaminated probes" to a six-class label on every planted item, assigned automatically against the actual summary text:

`explicit-in-summary / paraphrased-in-summary / absent-from-summary / contradicted-by-summary / tail-visible / evicted-only`

Report all probe results **by class**. The load-bearing claims live in *absent-from-summary* and *evicted-only*; any non-A success on *evicted-only* is presumptively leakage, judge artifact, or bug until explained. This can be applied retroactively to completed runs if summaries and manifests were saved.

## 5. New probe category: contradiction probes **[before-results if materials not yet frozen; else phase-2]**

Plant items where the summary asserts X but the old context's truth is Y (post-edit the summary for half the set). These measure whether grafted/retained latent state can **override visible text** — scientifically the sharpest question in the design, and practically double-edged: if values can silently outvote the visible context, that is both the effect we hypothesize and a deployment risk worth documenting. Score as contradiction-following rate per arm. If your synthetic materials are already frozen, defer to phase 2 rather than regenerate.

## 6. Metric hierarchy flip **[analysis-time; writeup]**

Probe accuracy (by category × class × arm) is the **headline**; continuation NLL is **secondary**. NLL rewards fluency and style continuity, which is not the claim. Keep normalized gap closure as the cross-arm summary statistic, with edge-case handling: if A ≈ B on a conversation, gap closure is unstable — report raw deltas there; if an arm exceeds A, cap in plots only, never in tables. Add first-token KL from A as a cheap local-divergence diagnostic. Statistical treatment: probes within a conversation are not independent — paired design with conversation-level (hierarchical) bootstrap, as the brief already specifies; this amendment just makes it non-optional.

## 7. One deliberately strong text-only baseline **[before-results if cheap; else phase-2]**

Pre-empt the obvious reviewer objection — "wouldn't a better summary fix this?" — by adding at least one strengthened text arm: a 2–3× summary budget, or an extractive bullet-memory that preserves named entities and constraints verbatim. If ValueGraft-family arms beat a deliberately strong text baseline, the result is much harder to dismiss. If the current runs are against the standard summary only, this can be a phase-2 addition, but say so explicitly in the writeup.

## 8. Sample-size framing **[writeup; phase-2]**

Current n (≈12 synthetic, ≈8 natural) is a **pilot** and should be named as such in any writeup. Publication-grade progression: pilot → powered synthetic (n≈50–100) → natural (n≈20–40) → coding traces. Do not let pilot-scale numbers carry paper-scale language.

## 9. Materials hygiene **[phase-2, unless trivially cheap now]**

Mitigations for generator/judge circularity — mixed-source synthetic materials (templates, hand-authored, a different model family for some fraction), and blinded judging (the judge never sees which arm produced an answer). Note: because all arms share the generator and comparisons are paired, this is an external-validity concern, not a validity threat to the contrasts themselves — which is why it's deferred rather than urgent.

## 10. Prior art and positioning **[writeup — important]**

An early-June 2026 paper is now the closest neighbor and must be engaged directly: **"Models Take Notes at Prefill: KV Cache Can Be Editable and Composable" (arXiv 2606.17107)**. It establishes, causally across model families, that prefill writes field-conditioned conclusions onto *downstream* tokens (the field's own KV drives <1% of the decision), and that cached "notes" are position-portable — RoPE-repositioned splices near-indistinguishable from full recompute for self-contained blocks.

Consequences:
- **It supports our premise.** Write-time downstream state demonstrably carries context-conditioned conclusions. H1 is no longer speculative; cite this as mechanism evidence. It also suggests a refinement worth trying if E's effect is weak: weight the graft toward high-attention-mass "aggregator" positions rather than uniformly.
- **It constrains our novelty claim.** "KV is editable/composable/portable" is taken. The defensible framing: *prior work optimizes KV reuse, editing, and composition, mostly for latency and cache economics; this work isolates whether compaction destroys the semantic continuity of text that remains visible, and whether write-time value states restore it.*
- **It sharpens the boundary.** Their splice≈recompute result holds for *self-contained* content (skills, documents) — content whose encoding barely depends on its surroundings. Our probes deliberately plant *context-dependent* content; that is exactly where splice and recompute should diverge, and the contrast between their result and ours (if we find one) is the interesting sentence of the paper.

Also cite as neighbors: KVEraser (learned span-erasure via steering states — the learned cousin of our eviction), Cartridges and Attention-Matching compaction (trained/synthetic KV state as reusable context), CacheBlend/KVLink (chunk-level cross-context reuse), gist/AutoCompressor/ICAE/Beacon (trained latent compression — H is their training-free, natural-language analog), StreamingLLM/H2O/SnapKV (eviction lineage), LLMLingua/RECOMP (text-compression baselines), Lost-in-the-Middle/RULER/SCBench (evaluation framing — vary plant positions across early/middle/late-evicted/tail), MemGPT/A-MEM (agent-memory motivation), and C2C/LatentMAS/KVCOMM (cross-model/agent latent transfer, for the discussion section).

Language discipline for the writeup: prefer "value states carry context-conditioned information useful for downstream interpretation" and "the same text induces different continuations depending on write-time cache state" over "meaning lives in the values" or "the model remembers in KV."

## 11. Phase-2 adjustment: offline trajectory prediction before task resolution **[phase-2]**

Refining document 3: before (or instead of) full SWE-bench task-resolution runs, evaluate on **offline trajectory prediction** — from a compacted real agent trace, predict the next action / next file touched / next command / next patch location, scored against the actual trajectory. Full task success is noisy and expensive; a cache intervention can improve semantic continuity while task outcome stays dominated by unrelated tool and search errors. Trajectory prediction gives cleaner signal per GPU-hour, and the F-arm span priorities for code remain as document 3 lists them (imports, signatures, failing test output, constraints, failed commands, exact paths, repeatedly-referenced symbols).

## 12. Suggested priority order, reconciled with in-progress work

If starting fresh, the reviewed core would be: **A, B, B-causal, H(-pack primary), E-post (+ negative controls), then E-inter**, with G and C conditional and F deferred to coding. Since you are mid-flight: finish what's in progress; then the cheapest high-value insertions are, in order, **B-causal (item 1), the two negative controls (item 2), the leakage classifier (item 4, retroactive), and the metric flip (item 6, retroactive)**. Everything else slots into phase 2 or the writeup. A tight, brutally-controlled A/B/B-causal/H/E result is worth more than every additional arm combined.
