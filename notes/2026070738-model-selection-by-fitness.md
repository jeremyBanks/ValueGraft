---
name: model-selection-by-fitness
description: "Pick subagent/model by fitness for the task, not to save cost; Opus has headroom, Sonnet is for perspective-diversity not cheapness"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bda7fb9f-f447-4890-904b-dde750ff3370
---

Choose the model for a task by FITNESS, not cost-avoidance. The user has
generous Opus headroom, so do NOT reflexively downgrade subagents to Sonnet
to save money — that was a mistake I repeated.

**Why:** the user said (2026-07-07) "we don't need to use [Sonnet] for cost
reasons. We only need to use it if it's useful." Fable is expensive but
available; Opus has quite a bit of headroom.

**How to apply:**
- Default subagents to Opus (or the main-loop model) unless a task specifically
  wants something else. Don't route to Sonnet as a cost-safe default.
- Sonnet's real value = a DIFFERENT model's perspective (independent third/fourth
  eyes in a critic panel, adversarial check, or bake-off) — use it for diversity,
  not because it's cheap.
- Fable = strong writing; use it where writing quality is the point (it does best
  with CLEAN, minimal context — a drowned Fable loses the thread).
- Model bake-offs (e.g. Fable vs Opus fresh-eyes prose pass, then merge best-of-both)
  are a legitimate quality move when the output really matters.
- **Fable readability pass is a HARD requirement for any writing deliverable.** The
  main loop keeps getting filter-flipped to Opus even when the user sets Fable, so
  do NOT assume Fable is writing — explicitly spawn a `model: fable` SUBAGENT for the
  readability/prose pass (clean minimal context: just the doc). If the final prose
  never passed through a Fable subagent, the writing isn't finished. (user, 07-07)
- Cost still matters for pods/compute; this is specifically about model tier for
  agent/LLM calls, where fitness wins.

Related: [[verify-kills-by-pid]] (verify-don't-trust discipline applies to
subagent outputs too — check the file, not the claim).
