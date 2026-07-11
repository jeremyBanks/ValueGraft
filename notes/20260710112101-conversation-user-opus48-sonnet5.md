_The overnight validation is complete and the paper is finalized as a
mostly-null, provenance-conscious bounding result: conversation recovery is null
across champion families and compression levels, while only a small SWE-Gym
brief scalar-graft proxy effect survives. A follow-up brief-SWE-Gym confirmation
and in-domain optimization run is now active, using the remaining budget to test
whether that fragile positive replicates and whether tuning improves it._

**Participants:** User, claude-opus-4-8, and claude-sonnet-5.

**Finalized paper and repository.** The paper is science-first with a postmortem
at the end, uses no coined method name, and reports conceptual-replication-grade
methods. It was adversarially reviewed, proofread, and cold-read; fixes
addressed the domain-versus-terseness mechanism error, content-specificity
overclaims, scale anchoring, terminology, missing-data disclosure, and scope
creep. The finished paper was promoted to root `README.md`, the draft archived
as `notes/2026070920-paper-draft-archived.md`, notes normalized, summaries
regenerated, and all changes pushed to `origin/trunk` at `d4cbf16` (README
promotion at `4b05806`, naming at `67ddae5`; paper fixes at `d94df0b`).

**Results.** The held-out four-way comparison (heads, layers, intersection,
union) is null for every family; no champion beats baseline. The synthetic
compression sweep is flat to slightly negative from ultra through realistic,
refuting concentration of recovery benefit under aggressive compaction.
Content-specificity is real for slot-heavy configurations but does not produce
net recovery improvement. The sole surviving positive is the plain scalar graft
on real SWE-Gym coding trajectories under brief summaries, approximately
`+0.013 nats/token` with a CI barely above zero, roughly 8% of the recoverable
gap; it is a teacher-forced next-action log-probability proxy, not task success.
The transferred conversation-tuned champion does not improve on it. A free
discrete-generation check points in the same direction but is underpowered:
correct-action matches were approximately 39% versus 37% and 36% versus 33%.

**Provenance correction.** The prod-SWE-Gym champion arm (75 trajectories) and
five brief-champion trajectories were not persisted. The loss occurred before
the budget cutoff through a filename collision involving the scalar backup, not
through the final harvest; the pod and transient logs are gone, and git contains
no recoverable copy. Persisted data still includes the complete scalar
brief/prod arms, the four-way comparison, compression curve, and 70/75
brief-champion trajectories. The paper documents the gap and excludes the absent
prod-champion arm rather than inferring it. The loss is scientifically
secondary, but the earlier claim that nothing was lost was too broad and must
not be repeated.

**Follow-up decision.** With `$25.30` available, an unled Fable assessment
(model `claude-opus-4-8`) recommended resolving the fragile positive before
optimizing it: run roughly 90–100 genuinely disjoint new brief-SWE-Gym
trajectories, add a discrete action-match metric, and use shared scoring for an
α-sweep. The design also includes placebo controls, held-out evaluation, and
coding-specific per-region profiling/champion construction. Existing repeated
runs are the same 75 trajectories, so combining them denoises but does not add
independent sample size (`+0.0145 [+.0040,+.0257]`).

**Operational state.** The first launch failed cheaply after about one minute
because an intentionally unset singular `SC_E_ALPHA` was forwarded as an empty
string and parsed with `float('')`; the parser was corrected to treat legitimate
optional empty variables as defaults, committed, and relaunched. The active run
is confirmed healthy on genuinely disjoint indices (`t0226+`), with α-sweep
scoring underway and approximately 13/150 processed at last check; expected
duration is about 1.5–2 hours. A backed-off monitor (`bbb11uez3`) harvests the
unique result directory every roughly 12 minutes and reports only completion,
crash, vanished processes, or runtime beyond approximately 3 hours. After
confirmation, the remaining budget (about $18, conditional on survival and cost)
is reserved for a coding compression dose-response; paper updates to §3.3 and
§10 will incorporate the final confirmation, α result, and discrete metric.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `a912f14b713ed6ced`
- `a27b526f2f3c478cd`
- `aeb77f5bc686606db`
- `a2442929773d0546c`
