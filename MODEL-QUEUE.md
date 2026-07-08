# MODEL-QUEUE.md — prioritized cross-arch run order

Ordering rule (in priority): **(1) scientific value → (2) likelihood of loading clean (arch
familiarity) → (3) VENDOR DIVERSITY as the tiebreaker among equally-likely models** (spreading
vendors strengthens the cross-vendor generality of the geometry/QK-norm claim; don't cluster one
vendor). Run ≤3 secure pods concurrent per the tier policy (DECISIONS.md), more if community is
healthy. Do NOT start a wave until: canary passed (observed result) + observability fault-tested.
All 11 passed the pre-flight load-check; "risk" below = architecture-oddness, not id validity.

## WAVE 1 — the primary de-confound PAIR (must be same-vendor by design; top value)
1. **Qwen/Qwen3-30B-A3B-Instruct-2507**  (Qwen · MoE · QK-norm)  — CANARY / pair A
2. **Qwen/Qwen3-32B**  (Qwen · dense · QK-norm)  — pair B → COMPLETES the primary inference
   (does the sign flip MoE↔dense with attention geometry held fixed?)

## WAVE 2 — high-confidence breadth, ONE NEW VENDOR AT A TIME (for the geometry regression + H1)
Interleaved by vendor so cross-vendor signal appears early; QK-norm present/absent mixed:
3. **allenai/OLMo-2-0325-32B-Instruct**  (allenai · dense · QK-norm)  — NEW vendor + cross-vendor QK-norm
4. **mistralai/Mistral-Small-24B-Instruct-2501**  (mistralai · dense · no-QK-norm)  — NEW vendor
5. **01-ai/Yi-1.5-34B-Chat**  (01-ai · Llama arch · no-QK-norm)  — NEW vendor, very standard
6. **microsoft/phi-4**  (microsoft · dense · no-QK-norm)  — NEW vendor
7. **Qwen/Qwen2.5-32B-Instruct**  (Qwen · dense · no-QK-norm)  — the H1 within-Qwen contrast
   (Qwen3 QK-norm vs Qwen2.5 no-QK-norm, the non-fitted contrast H1 was built on) — repeat vendor
8. **mistralai/Mixtral-8x7B-Instruct-v0.1**  (mistralai · MoE · no-QK-norm)  — repeat vendor, MoE

## WAVE 3 — riskier architectures (new vendors, but odd archs → run LAST; expect some UNSUPPORTED)
9.  **zai-org/GLM-4-32B-0414**  (zai)
10. **openai/gpt-oss-20b**  (openai · attention sinks / unusual)
11. **nvidia/Llama-3_3-Nemotron-Super-49B-v1_5**  (nvidia · DeciLM heterogeneous — highest arch risk)

## EXCLUDED (pre-flight, multimodal wrappers — see PREREGISTRATION deviation note)
google/gemma-4-31B-it, gemma-4-26B-A4B-it (would have been a 2nd de-confound pair, google),
gemma-3-27b-it, Qwen3.6-35B-A3B, Qwen3.6-27B. Gemma-4 text-substack recovery = a possible later spike.

## Rationale
Required same-vendor pair first (Wave 1). Then likelihood-tier: standard archs (Wave 2) before odd
archs (Wave 3), and WITHIN the standard tier, new-vendor-first so we get allenai/mistral/01-ai/
microsoft coverage before repeating Qwen. If Wave 3 fails, the spine (Qwen3 pair) + a 5-vendor
breadth for the QK-norm regression are already banked. Anchors 1-2 (+7) get 24 convs; rest 12.
Vendor count if all run: Qwen, allenai, mistralai, 01-ai, microsoft, zai, openai, nvidia = 8 vendors.
