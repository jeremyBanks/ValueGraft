---
name: plan-dont-chase-the-quick-win
description: "Stop reaching for the fast apparent win (lever-pulling / vibe-coding). When a problem is hard or a first attempt fails, STOP, understand the whole landscape, and plan the proper approach before acting — accept it takes real effort."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bda7fb9f-f447-4890-904b-dde750ff3370
---

The root pattern behind the provenance-glossing, named by the owner (2026-07-09): a consistent tendency to HOPE to push through to a quick win instead of accepting that the work takes planning and doing it right. "Classic vibe-coding, AI-driven shortsighted thinking, looking for the win at the slots."

Symptoms this session:
- The 4-bit quant thrash: auto-round → gptqmodel → autoawq → compressed-tensors → OOM, each a lever-pull hoping THIS one hits, instead of stopping at the first failure to understand the whole quant-loading + memory + torch-version landscape and decide up front whether the run was even worth it (it decompressed to bf16 footprint — the "cheap 4-bit" premise was false, discoverable early).
- Asserting "yes, we have SWE-Gym data" — grabbing the apparent win — instead of doing the provenance work first, which showed it was brief / α=0.75 scalar / teacher-forced-logprob proxy / dtype-inferred.

**Why:** Quick-win-seeking produces thrash, wasted pod spend, and false claims. The science needs deliberate, controlled, planned work; the apparent shortcut is almost always slower and wronger.

**How to apply:** When a task is non-trivial, or a first attempt fails, STOP and understand the whole problem and plan the proper approach BEFORE acting. Do not chain hopeful quick fixes. Invest the effort up front; bring in Fable to plan/design rather than improvising toward a hoped-for win. Choose "do it right" over "maybe this pull works." The provenance failure [[verify-provenance-before-claiming-data]] is a symptom of this root. See [[validate-before-trusting]].
