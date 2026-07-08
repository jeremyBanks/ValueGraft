---
name: question-the-backend
description: "Question the tool/backend for a task; don't reflexively reuse an existing slow tool. For text generation use fast models (subagent mix/API), never local single-threaded generation."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bda7fb9f-f447-4890-904b-dde750ff3370
---

I reused an existing generator (`compose.py`) that renders synthetic-corpus filler
text via a **local 4-bit 4B MLX model**, token-by-token, re-prefilling a growing
14K context per turn — ~7-8 MIN PER CONVERSATION. The user: "using local MLX is
insane, that's the slowest of all possible options." Correct.

**Lesson:** when a task needs generation, QUESTION THE BACKEND before running —
don't reflexively reuse an existing tool just because it's there. The assistant
replies were only *context filler*; they didn't need a model at all, let alone a
slow local one.

**How to apply:**
- Text/content generation → use FAST models: a MIX of subagents (Fable/Opus/Sonnet,
  low effort, specific goals) writing in parallel, or an API — NOT local
  single-threaded generation. The mix also gives useful diversity + verification.
- For anything slow, ask "is this the fastest appropriate method?" A tool existing
  is not a reason to use it. Local MLX/Ollama is for quick sanity checks, not
  production generation.
- VERIFY each generator's output against the required schema (the user insisted).
- Related: this is the flip side of [[model-selection-by-fitness]] — pick by
  fitness AND speed, and don't let inertia pick a slow default.
