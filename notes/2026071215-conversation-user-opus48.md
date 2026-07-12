_The first exact-model canary (e01) validates the experimental setup and shows
strong history dependence, but treatment efficacy remains unmeasured because
treatment scoring is blocked by an unresolved path-control stop-rule conflict._

**Participants:** User and claude-opus-4-8.

On bf16 30B, the focal answer margin was +16.5 nats with correct history, −19.7
with wrong history, and −5.8 in the fresh post-compaction condition. Thus
evicted history strongly controls the answer, summary text alone fails to
preserve it, and the graft has roughly 22 nats of recovery headroom. The
non-focal control behaved as expected. These are pre-treatment reference results
only; `treatment_scores_present = False`, N=1, and no claim that the graft works
is supported yet.

**Handoff State.** Resolve the path-control/stop-rule conflict before scoring
coherent-state graft arms, then report treatment recovery against the
pre-treatment baseline. The project remains pre-spend following the locally
caught v12 generated-stop failure; current priority is non-blocking
collaboration, one cheap exact-model exploratory canary, and an honest
methodological paper. A scheduled archive refresh also fired and began
pulling/committing loose notes in the background.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
