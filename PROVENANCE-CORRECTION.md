# PROVENANCE CORRECTION — model attribution error

**Issue (recorded 2026-07-07 ~08:45):** git commit trailers on this repo
credit "Claude Fable 5" on **314 commits**. The Fable→Opus 4.8 handoff
(Fable quota exhaustion) actually occurred earlier than the trailer was
updated — the trailer only switched to Opus 4.8 at commit 12972bf
(~08:10 07-07). Therefore an UNKNOWN TAIL of those 314 Fable-trailered
commits was in fact authored by **Claude Opus 4.8**, mislabeled as Fable.

**Why not rewritten:** the exact handoff commit is unknown (the model
change was noticed by the user verbally, with no marker in history);
rewriting 314 trailers to a guessed boundary would fabricate a precision
we don't have. This standing correction is the honest record instead.

**Correct attribution for the work as a whole:**
- Claude Fable 5 — majority of the session (design + execution) up to an
  imprecise point on 07-07.
- Claude Opus 4.8 — a late-session block on 07-07 (including much of the
  tau integration and this correction), mislabeled as Fable in commit
  trailers b6bd161 and earlier until 12972bf.
- Claude Sonnet 5 — subagents (judging, scouting, adapters, probes).
- GPT-5.5 (OpenAI) — adversarial/second-perspective review.
- Directed & funded by the user (Jeremy Banks).

**Meta-error:** I continued stamping "Fable" after becoming Opus and only
corrected when the user flagged it — a provenance-integrity failure in a
research audit trail, not merely a credit oversight. Logged as INCIDENTS #26.

## UPDATE — exact boundary recovered from transcript (2026-07-07 ~08:50)

Read-only analysis of the session transcript (assistant-message `model`
fields) gives the precise handoff — it was NOT unknown:

- **Single clean transition at 2026-07-07 02:11:01→02:11:13.** No
  back-and-forth; one switch.
- **claude-fable-5**: 3,323 assistant messages, 2026-07-04 20:09:13 →
  2026-07-07 02:11:01.
- **claude-opus-4-8**: 158 assistant messages, 2026-07-07 02:11:13 →
  ongoing.

So: commit trailers said "Fable" until ~08:10 07-07 (commit 12972bf), but
the ACTUAL model became Opus at 02:11 07-07. Commits made in the ~6-hour
window 02:11→08:10 that trailer "Fable" were in fact **Opus 4.8** work.
(Boundary now known precisely; history still NOT edited per user directive —
this note is the correction of record.)

## What triggered the switch (transcript evidence, 07-07)

NOT topic-triggered. The boundary (02:11:01 Fable → 02:11:13 Opus) sits
between two responses to the SAME routine "90-min heartbeat" notification,
mid-work on the chain arm table. Last Fable content = graft@1.0 collapse
analysis; first Opus content = debugging a stuck local port. No sensitive
topic, capability boundary, or subject change at the seam. Signature =
Fable-5 QUOTA EXHAUSTION → automatic fallback to Opus 4.8 at the next
turn. (User had independently noted Fable was "basically out of quota".)
