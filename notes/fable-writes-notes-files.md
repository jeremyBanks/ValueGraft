---
name: fable-writes-notes-files
description: "Every serious Fable consultation must WRITE its output to a notes/ file, not just return it in chat"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bda7fb9f-f447-4890-904b-dde750ff3370
---

For every serious Fable consultation, do NOT let the result live only in the agent-to-agent reply — it gets lost. Instead: **I create an empty notes/ file** (following the `YYYYMMDD<counter>-slug.md` naming convention, next counter after the latest), pass Fable a **ton of context** in the prompt, and instruct Fable to **write/edit its assessment directly into that file itself** (it owns the file) and return only the path + a short topline. Fable's substantive outputs (verdicts, assessments, experiment designs, trajectory reviews) are thereby saved for posterity in `notes/`.

**Why:** the owner observed (2026-07-09) that major Fable results were "just in communication between you guys and that's getting lost." Notes files are the durable record.

**How to apply:** default this for every serious/major Fable consult going forward — un-anchored trajectory reviews, design consults, assessments. Combine with [[fable-full-context]] (give full trajectory + point at notes/ logs) and the un-anchored consulting rule (don't lead the prompt). Pairs with the claims-ledger discipline ([[validate-before-trusting]]): capture provenance and Fable's reasoning durably before writing the paper.
