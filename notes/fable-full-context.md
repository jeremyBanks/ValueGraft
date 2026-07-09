---
name: fable-full-context
description: "When consulting Fable, give it full context and point it at notes/ conversation logs for trajectory"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bda7fb9f-f447-4890-904b-dde750ff3370
---

When consulting Fable (the strategic advisor subagent), give it FULL context, not just a
compressed summary of the moment. Explicitly encourage it to read the CONVERSATION LOGS in
the `notes/` folder (files like `notes/*conversation*.md`) for recent trajectory/context if it
wants to ground itself in how we got here.

**Why:** the owner observed that a risky plan (the binary full-QK-norm ablation, which broke the
model) got agreed to partly because Fable was working from a thin summary rather than the full
trajectory. Under-contextualized advice produces plausible-but-ungrounded plans.

**How to apply:** in every Fable consult prompt, (1) include the real state + history, not a lossy
recap, and (2) add "read the conversation logs in notes/ for trajectory if you want fuller context."
Grounded Fable > fast Fable. See [[question-the-backend]] and [[validate-before-trusting]].
