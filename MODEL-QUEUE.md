# MODEL-QUEUE.md — enrichment plan (post-pivot, 2026-07-09)

Models now serve **ENRICHMENT** (the QK-norm regression is dead — incident #37): map the graft's effect
+ its architecture-specific reversal across as many vendors/archs as possible. **Gated behind the core
reproduction.** Work the queue top-down; render only models that pass a cheap load-smoke.

## THE QUEUE (run in this order)

**DONE / RUNNING:** Qwen3-30B-A3B (anchor, +0.10) · Qwen3-32B · Qwen2.5-32B · Mistral-Small-24B · phi-4
→ vendors so far: Qwen, mistralai, microsoft.

**TIER A — new vendors, easy load (do FIRST):**
| # | vendor | model | note |
|---|---|---|---|
| 1 | meta | Llama-3.1-8B-Instruct (or 3.3-70B) | reference arch, biggest gap |
| 2 | allenai | OLMo-2-0325-32B-Instruct | fully open; use `SC_TASK_COMPETENCE_MODE=relative` |
| 3 | google | Gemma-2-27B-it | text (pre-multimodal Gemma) |
| 4 | ibm-granite | Granite-3.x | new vendor |
| 5 | cohere | Command-R / Command-R7B | new vendor |
| 6 | tii | Falcon-3 | new vendor |
| 7 | deepseek | DeepSeek-R1-Distill-Qwen-32B | DeepSeek flavor on Qwen arch → trivial load |
| 8 | mistralai | Mixtral-8x7B | MoE |

**TIER B — odd arch, may need a version tweak (pre-flight-passed):**
z.ai GLM-4-32B-0414 · openai gpt-oss-20b · nvidia Nemotron-Super-49B · sarvam Sarvam-30B

**CODING CONTRAST (after most diversity):** Qwen2.5-Coder-32B vs our Qwen2.5-32B — base-vs-coder, same
arch → tests arch-driven vs training-driven effect. 1–2 pairs (optionally Codestral vs Mistral).

**TIER C — hardest, LAST:** multimodal → text-backbone (Gemma-3, Qwen3.6, Llama-4) or 5.x-only/too-big
(GLM-5, DeepSeek-V4, Kimi, MiniMax).

**SKIP:** Yi-1.5-34B (came back UNSUPPORTED, reason uncaptured — retry cheaply to learn why).

**GOAL:** 6–8 new vendors on the map (meta, allenai, google, ibm, cohere, tii, deepseek, z.ai).

## Rules of engagement
- **Attempt many, cheap:** load + 1–3-conv smoke = minutes; the RENDER (~$6–9) is the cost. Smoke gates; render only winners.
- **Save every render** (AGENTS.md hard rule): render once → effectiveness direct; champion/rescue/tuning/reproduce all cheap-or-free after, forever. Each render = a permanent asset.
- **Budget ~$100:** core reproduction (~$15–20, GATES all) → arch poles (~$10) → enrichment (~$60–70). Rigor first, always.

## Reference notes
- **UNSUPPORTED = 3 flavors, always read the reason:** OOM (size) / "load failed: \<exc\>" (arch/transformers) / unhandled-arch-feature (harness gap, patchable). Never call it "failed" blind.
- **transformers-5.x:** pods pin 4.57.1 (5.x threw a weight-conversion RuntimeError; pinned as a workaround, NOT fundamental — fixable later). Unlocks the newest 2026 archs. Park at the very END, test isolated, don't touch mid-run.
- **Multimodal = difficulty, not exclusion:** the wrapper's `.language_model` is a normal decoder we CAN graft (+ a vision tower we ignore). Check for a TEXT-ONLY SIBLING first (Qwen vs Qwen-VL, Mistral vs Pixtral); else extract the backbone (last/hardest).
- **Landscape (2025–26):** 2025 = DeepSeek V3/R1, Llama-3.x, Qwen2.5/3, Gemma-2, OLMo-2, GLM-4; 2026 = Gemma-4, Qwen3.6, Llama-4, GLM-5, DeepSeek-V4 (mostly multimodal/huge/5.x). Sources: computingforgeeks open-source-llm-comparison, HF blog daya-shankar/open-source-llms, sebastianraschka "a-dream-of-spring" (architectures), lmstudio.ai/models.
