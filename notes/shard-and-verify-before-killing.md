---
name: shard-and-verify-before-killing
description: Shard multi-item generation across parallel subagents from the start; NEVER conclude a subagent is stalled from a proxy (output-file byte-size) — verify real progress before any destructive kill.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bda7fb9f-f447-4890-904b-dde750ff3370
---

Two mistakes in one incident (07-08), both worth not repeating:

**1. Shard independent multi-item work from the START — for QUALITY, not just speed.**
I ran a 27-scenario authoring job in ONE sequential subagent — immediately after
correctly sharding the sibling rendering job across a parallel mix. Same shape of work,
opposite (wrong) choice. If N items are independent (N scenarios, N files, N models),
fan them out to parallel subagents writing to SEPARATE outputs, then merge.
The speedup is the obvious reason; the BIGGER reason is QUALITY:
- **Author diversity.** One model writing all N converges on its own patterns (same
  rhythms, same structures). Several subagents across different models (Sonnet/Fable/
  Opus) in fresh contexts produce genuinely varied output — a feature when the artifact's
  purpose IS diversity (a corpus, a test set, examples).
- **Fresh attention per item.** A single context grinding through N degrades — by item
  ~20 it's fatigued and formulaic, reusing what it wrote. Small shards give each item
  full, uncrowded attention; the tail items of a long sequential run are the weakest.
So sharding generation is a quality decision, not merely a latency one. Default to it.

**2. NEVER kill a subagent off a PROXY signal — verify real progress first.** I judged
the subagent "stalled" from its 131-byte OUTPUT-FILE SIZE + an unchanged target file,
and killed it. It was actually ~25/27 done (on the last two items), holding work
in-context to write at the end; its last message even said "now the final two." I
destroyed near-complete work off a misread metric — right after preaching "verify,
don't assume."
- A small transcript/output file != no work (agents buffer, write at the end).
- Before ANY destructive action (kill, delete, overwrite), check REAL progress: the
  agent's recent activity timestamp, its last emitted message, actual artifacts — not a
  size/mtime proxy. If active recently, it's working; wait or ask.
- Destructive actions are irreversible; the bar for them is HIGHER than for a status
  read. Slow down specifically there.

Related: [[validate-before-trusting]] (verify by reading real signal, not narrating a
proxy), [[question-the-backend]] (the sibling rendering task that I *did* parallelize).

**Git: work ONLY on trunk — never branches (this project).** A subagent created a
feature branch and mainline work got committed onto it; the owner was emphatic: only
trunk, always. Tell every subagent to commit to trunk and never branch. Push to origin/trunk after every unit of work (commit + push) so nothing sits only
local and origin is always current. If a branch
appears, consolidate with a fast-forward (git branch -f trunk <head>; checkout trunk;
delete branch) — it is just moving head references, non-disruptive, no working-tree change.
