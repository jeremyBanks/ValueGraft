# MODEL-QUEUE.md — model enrichment plan of record (rewritten 2026-07-09, post-pivot)

**Purpose changed:** the old QK-norm cross-arch REGRESSION is DEAD (H1 falsified — see DECISIONS/incident #37).
These models now serve **ENRICHMENT**: mapping the graft's effect + its ARCHITECTURE-SPECIFIC reversal
across as many vendors/architectures as we can. Diversity of vendor + arch lineage is the value; the map
(the graft's sign/signature varies and can reverse across the field) is the novel finding.

## Operating approach (owner, 07-09) — attempt MANY cheaply, render the WINNERS
- Getting a model to LOAD + pass a 1-3 conv smoke is CHEAP (minutes of pod time: download+load+probe), NOT
  a render. The RENDER (~$6-9, ~5-6 pod-hr) is the cost. So: cheap-load-smoke gates each; render only winners.
- SAVE EVERY RENDER (AGENTS.md hard rule): render once → effectiveness result direct; champion scan / rescue
  test / tuning / reproduction all cheap-or-free afterward, forever. Each render = a permanent reusable asset.
- Budget ~$100 (owner +$50): core reproduction first (~$15-20, GATES all) → arch poles (~$10) → enrichment
  (~$60-70 ≈ 8-12 models). Rigor gates every dollar; never breadth before rigor.

## "UNSUPPORTED" is not one thing — ALWAYS read the reason (diagnostic note)
Our harness stamps status=UNSUPPORTED + a REASON when it can't produce trustworthy numbers. Three very
different flavors: (1) OOM = too big for the GPU (size — quantize/smaller/bigger GPU); (2) "model load
failed: <exception>" = didn't load (arch/transformers incompat — the exception says what); (3) unhandled
arch feature (e.g. Gemma sliding-window) = loaded but graft code doesn't support it (a harness gap, patchable).
Yi-1.5-34B came back UNSUPPORTED but its reason was never captured (pod terminated) — re-attempt (cheap)
to learn which flavor. DON'T report UNSUPPORTED as "failed" without the reason.

## transformers-5.x (deferred, NOT fundamental — end of list)
Pods pin transformers==4.57.1 because 5.x threw a RuntimeError ("automatic weight conversion") on our
models; we PINNED as a workaround rather than debug it. Not a wall — likely a 5.0.0-era bug/API change
(fix: later 5.x patch, update our loader, or a skip-conversion flag). Matters because the newest 2026
archs may REQUIRE 5.x → getting it working unlocks Gemma-4/Qwen3.6/Llama-4-class. Park at the very END,
test ISOLATED per-model (risk of breaking the working pipeline). Do NOT touch mid-experiment.

## Multimodal is a DIFFICULTY tier, not an exclusion
Multimodal wrapper (e.g. Gemma3ForConditionalGeneration) = .language_model (a NORMAL text decoder we CAN
graft) + .vision_tower + projector. Our AutoModelForCausalLM path fails only because multimodal repos
register under a different auto-class — the decoder inside is standard. Two ways in: (a) CHECK FOR A
TEXT-ONLY SIBLING first (Qwen vs Qwen-VL, Mistral vs Pixtral, small Gemma text sizes — cheaper); (b)
EXTRACT the backbone (load the right class, grab .language_model, run graft on the decoder, ignore vision
— modest custom code, fiddly KV-surgery layer paths). Extraction is the LAST/hardest tier.

## ---- BASE (done/running) ----
Qwen3-30B-A3B-Instruct-2507 (Qwen·MoE, the +0.10 anchor) · Qwen3-32B (Qwen·dense) · Qwen2.5-32B (Qwen·dense,
referent−) · Mistral-Small-24B (mistralai·dense, referent+/sense−/stance−) · phi-4 (microsoft·dense, null).
Vendors covered: Qwen, mistralai, microsoft.

## ---- ENRICHMENT candidates, ORDERED (easy proven-load fruit FIRST) ----

### TIER A — easy (standard text arch, high load confidence; NEW vendors)
1. meta / **Llama-3.1-8B-Instruct** (or 3.3-70B) — the reference arch; biggest vendor gap; small=cheap
2. allenai / **OLMo-2-0325-32B-Instruct** — fully open, QK-norm; loads (use SC_TASK_COMPETENCE_MODE=relative)
3. google / **Gemma-2-27B-it** — TEXT (pre-multimodal Gemma); major vendor; confirm text-causal load
4. ibm-granite / **Granite-3.x** — new vendor, standard
5. cohere / **Command-R / Command-R7B** — new vendor
6. tii / **Falcon-3** — new vendor
7. deepseek / **DeepSeek-R1-Distill-Qwen-32B** (or -Llama-70B) — DeepSeek flavor on Llama/Qwen arch = trivial load
8. mistralai / **Mixtral-8x7B-Instruct** — MoE (repeat vendor, new arch)

### TIER B — medium (odd arch / may need transformers-version tweak) [pre-flight-passed in old queue]
9. z.ai / **GLM-4-32B-0414** · 10. openai / **gpt-oss-20b** (attn sinks) · 11. nvidia / **Nemotron-Super-49B** (DeciLM)
· 12. sarvam / **Sarvam-30B**

### TIER C — hard (LAST): multimodal→text-backbone, or 5.x-only, or too-big
google/Gemma-3-27B · Qwen/Qwen3.6 · meta/Llama-4-Scout (all multimodal → extract or find text sibling) ·
GLM-5 / DeepSeek-V4 / Kimi / MiniMax (huge / 5.x-only).

### SKIP: Yi-1.5-34B (UNSUPPORTED, reason uncaptured — retry cheaply to learn why); multimodal-only w/o a
text sibling until Tier C.

## Landscape references (2025-2026, for picking + updating)
2025: DeepSeek V3/R1 (MIT), Llama-3.x, Qwen2.5/3, Mistral, Gemma-2, OLMo-2, phi-4, GLM-4.
2026: Gemma-4, Qwen3.6, Llama-4, Mistral-Small-4, GLM-5/5.1, DeepSeek-V4 — many multimodal/huge/5.x-only.
Sources: computingforgeeks.com/open-source-llm-comparison, huggingface.co/blog/daya-shankar/open-source-llms,
magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight (architectures), lmstudio.ai/models.

## Goal: 6-8 NEW vendors on the architecture map (meta, allenai, google, ibm, cohere, tii, deepseek, z.ai).

## Coding-variant contrast (owner idea, 07-09): base-vs-coder = a controlled TRAINING-DOMAIN knob
Coding variants (Qwen-Coder, DeepSeek-Coder, Codestral, CodeLlama, CodeGemma, Granite-Code) are usually
the SAME architecture as their base — so ZERO arch diversity, but a clean CONTROLLED CONTRAST: base vs
coder, arch held FIXED, training domain varied. Tests: is the graft's semantic-continuity effect
ARCH-driven (robust to code fine-tuning) or TRAINING-driven (shifts)? Complementary to the arch map
(map varies arch; this holds arch, varies training). Cheapest high-value pairs = where we already have
the base: Qwen2.5-Coder-32B vs Qwen2.5-32B; Codestral vs Mistral-Small. Easy load (proven arch), cheap
(reuse pipeline). Caveat: our task is NL-conversation; a code-heavy model may render NL a bit worse
(measurable via competence gates; modern coders are base+code, usually OK). PRIORITY: mid — after
pure-new-vendor Tier A (it's a refinement, not a new vendor); 1-2 pairs is plenty.
