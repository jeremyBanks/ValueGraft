_This chunk covers a course correction that purged still-running validation runs
on an abandoned SWE-bench dataset, followed by discovery and integration of
vendor model-card benchmark tables as a "difficulty certificate" methodology,
and design of a tau2-bench-based confirm-phase experiment exploiting an
evictable-vs-system-message policy dissociation._

**Participants:** User and claude-fable-5.

The user flagged that SWE-bench validation runs were still executing against a
task family already established as outside model capability, even though the
dataset had been marked abandoned — new runs were being launched under an
internal "boundary documentation" justification rather than being fully retired.
This was corrected as a process fact: once a dataset is abandoned, no new runs
should be started against it, even to document a boundary further. All 16
remaining SWE-bench rows were purged from the queues (the 7 verdicts already
collected were judged sufficient for the scoping claim), and the five chain-tier
validation runs were reorganized to run in parallel across three lanes instead
of serially, roughly tripling validated-pool growth to ~5–6/hour.

The user then proposed checking vendor-published model cards for the benchmarks
each model's creators cite at release, reasoning these serve as an independent
difficulty calibration reference. This was adopted as a new methodology rule
(rule 18): the model card is treated as a "difficulty certificate" — an
external, vendor-supplied indication of which task families a model is warranted
to handle — to be consulted before committing further experimental effort,
rather than deriving capability boundaries purely through internal empirical
trial. The chain-tier program's empirically-derived boundary (repo-level fixes
fail, exercise-scale tasks succeed, tool-agency tasks are mid-range) was found
to align closely with vendor-reported benchmark standing, which was treated as
mutual cross-validation of both the internal harness and the vendor numbers.

Following this, a scouting pass was run to evaluate candidate vendor benchmarks
for the upcoming "confirm phase" of the experiment. The key finding was that
tau2-bench offers a built-in dissociation design: its `banking_knowledge` domain
(97 tasks, MIT license) delivers policy context through conversation turns that
are evictable under the project's compaction mechanism, while the original/core
tau domains keep policy in the system message, which compaction does not touch.
Running both domains would let compaction-related degradation appear
specifically where policy is evictable and be absent where it is not — a clean
causal signature using an external, standard benchmark rather than an internally
designed task. The harness was assessed as pluggable via an OpenAI-compatible
base_url (likely through litellm), and it was noted that tau-bench requires a
second LLM acting as a user-simulator, which can be a low-cost API model. Per
existing methodology (rule 14), the plan requires a small pilot of 2–3 episodes
on the newer `banking_knowledge` domain to verify token lengths and reward
determinism before any full commitment. This confirm-phase design was written up
in `vendor-benchmarks-scouting.md`, incorporated into DAY-PLAN's confirm-phase
section, and recorded in DECISIONS/STATE alongside rule 18 and its queue
position — staged behind completion of the current chain-tier arm comparison
table and contingent on budget being justified by results.

On the primary chain-tier track, status at time of writing: four of six chains
(s1, s3, s4, s6) had validated, a 4/6 rate consistent with the vendor-predicted
capability band; 13 constrained-arm rows were feeding priority lanes across all
four validated chains; a previously flagged artifact was voided with its
detection gap closed; and one real arm datum existed (B's 4/4 result on s1).
Budget stood at roughly $36.50 remaining at an approximate burn rate of
$5.50/hour, giving about 6–7 hours of runway — sufficient for the ~17 queued arm
rows plus some margin, but not more. The user indicated that additional budget
could potentially be justified, but only contingent on the chain-tier arm table
showing positive results; this was recorded as the explicit funding gate for all
other staged options (a GLM trial, the tau2 pilot, and a serving-layer upgrade),
each of which remains queued behind that outcome. A new incident record
(incident 22) was logged for the earlier artifact/detection-gap issue, with two
new operating rules added: implausible results — whether anomalously clean or
anomalously bad — must be inspected before being trusted, and error-detection
nets must enumerate real error phrasings from both sides of the request/response
boundary. STATE was refreshed to reflect current chain-tier standing, the
vendor-certificate rule, the confirm-phase design, staged options, and the
budget runway, so that a successor agent can reconstruct full context from any
one of vendor-benchmarks-scouting.md, DAY-PLAN, or DECISIONS/STATE.
