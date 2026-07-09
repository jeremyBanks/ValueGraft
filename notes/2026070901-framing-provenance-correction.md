# Framing-provenance correction (owner, 2026-07-09)

**Correction to the historical narrative in the daily summaries and overview.**

The generated summaries (notes/20260705.md, notes/README.md, and the underlying
conversation logs) describe the mitigation-first framing as a REDIRECTION — as
if the project began with "write-time KV state differs from re-encoded text" as
its intended contribution, and external review (independent critique + a
GPT-5.5 session) then corrected the project toward mitigation.

**The owner states that is not what happened from their side.** The
mitigation-oriented framing was the owner's intent all along; the
mechanism-as-finding framing was the AGENTS' misunderstanding of what the owner
had tried to communicate. The owner did not realize for some time that the
agents had misunderstood and run with the wrong framing. The 07-05 "reframing"
was therefore not a change in the owner's goals — it was the point where the
owner's original intent finally got through.

## Why this matters

- **Do not narrate the project's history as "we initially believed the
  mechanism difference was the finding, until review corrected us"** in the
  paper, in summaries, or in briefings — that attributes the agents' confusion
  to the project's design. If the paper narrates framing history at all, the
  accurate version is: the experiment was always aimed at whether a deployable
  intervention mitigates compaction damage; early drafts by agents misframed
  the self-evident baseline difference as a finding and were corrected.
- **Summaries are generated from conversation logs that embed the agents'
  contemporaneous (mis)understanding.** Treat the "reframing" narrative in
  notes/20260705.md and notes/README.md as reflecting agent belief at the time,
  not owner intent. This note is the additive correction (history is never
  rewritten; the generated summaries are left as-is).
- Standing rule reminder this reinforces: "compaction destroys
  context-conditioned state" is baseline/denominator, never a finding — that
  was the owner's position from the start.
