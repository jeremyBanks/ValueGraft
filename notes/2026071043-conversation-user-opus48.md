_This conversation finalized the interpretation of the follow-up SWE-Gym
experiment as a clean independent null and recorded the remaining close-out
workflow, including post-paper note regeneration and a separate speculative
pedagogical note._

**Participants:** User and claude-opus-4-8.

**Results.** The disjoint STAGE1 sample (~70 trajectories, indices ≥214)
overturned the original brief SWE-Gym proxy positive: scalar α=0.75 was −0.0035
(CI [−0.018, +0.010]) versus the earlier +0.013 on 75 trajectories. No α
improved over baseline; α=1.0 and 1.5 were harmful, while the discrete
action-match change (47%→56%, +6 fixes) had a CI touching zero. The result was
therefore treated as a complete null, with STAGE2 held-out evaluation retained
to measure the tuned champion directly rather than infer its failure.

Runtime estimates were corrected from ~1–1.5 hours to ~7–8 hours after
accounting for 23 arms per trajectory; the run remained healthy, GPU-active, and
within budget. STAGE1 completed and STAGE2 was running at 20/150, with an
expected ~2 additional hours and sufficient balance. Harvesting every cycle
protected against data loss.

**Close-out State.** The committed order is: finish and value the run; update
README §3.3 and §10 to the clean-complete-null empirical result and push;
archive any draft and regenerate all note paths and summaries, then commit and
push those notes; finally obtain a fresh Fable assessment producing a standalone
speculative/pedagogical note about why content-specific values may be
non-additive. The main paper remains a paper-style empirical bounding result and
should not incorporate that speculation.

The user explicitly required that note regeneration occur after the paper is
completed and pushed, and that both the regenerated notes and prior round notes
be committed and pushed. The repository was synchronized at `44f3cf2` before the
live experiment; the live output was to be harvested and included at close-out.

The complete task state was captured in
[`notes/2026071002-todo-list-state-capture.md`](/private/var/folders/cz/581svs_55z5fd6919b74vhnh0000gn/T/valuegraft-luna-summary-e7vqjbml/notes/2026071002-todo-list-state-capture.md)
and pushed in commit `7f7e2b7` to `origin/trunk`. It records all 30 tasks,
including eight pending tasks, current result state, and the later filename
normalization during note regeneration.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
