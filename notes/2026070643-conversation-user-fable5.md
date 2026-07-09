_This chunk covers the aftermath of the shim session-leak contamination
discovery, a sequence of escalating trust ruptures over the user's discovery
that "real coding tasks" had in fact been running on synthetic tasks for twelve
hours rather than the promised SWE-bench dataset, and the resulting
re-architecture of the pipeline around fail-loud/fail-missing rather than
fail-silent data integrity guarantees._

**Participants:** User and claude-fable-5.

**Contamination discovery and remediation.** Overnight agent runs (~96 episodes)
on the compaction/value-graft experiment surfaced a session-state leak in the
shim: it was built assuming one task per pod, and scaling to shared lanes let
per-session cache state (summaries, config maps) bleed across arms sharing a
pod. Damage was bounded, not total — only same-task-different-mode runs on the
same pod were affected — but the user directed a full remediation: pre-register
a clean re-run before any new data exists, structurally prevent recontamination
(disjoint task-lane assignment, pre-flight isolation probes), and author a
harder task (t3, five spread-out constraints) that a brief summary cannot carry,
verified fail-pre/pass-post before acceptance. A 150-run design (3 task types ×
10 seeds × 5 arms) was pre-registered with locked analysis (binomial CIs, paired
McNemar vs B).

**Anomaly resolution and instrument-sensitivity gate.** The apparent α=0.75
underperformance anomaly from the prior night was investigated and substantially
attributed to a separate serving bug (e1), not a real dose effect — isolated
re-run rows showed E@0.75 passing repeatedly. Separately, the user raised that
the harness needed a smoke test to confirm the compacted baseline (B) actually
fails tasks post-compaction before trusting any comparison; this was
retroactively converted into a pre-declared gate on the live run (B ≥75% pass →
abort/redesign; ≤50% → proceed; between → demote t1/t2, rely on t3), with the
rule recorded permanently as a required instrument-validation step for future
work, since prior evidence (the t1/t2 ceiling effect) had already signaled the
risk.

**Data hygiene and terminology decisions.** All contaminated/void/superseded run
data (156+ directories) was separated from clean data, then — after user
correction — brought fully into the git repository under
`results/agent_clean_run/` (clean, auto-syncing) and
`results/_QUARANTINE_agent_runs/` (contaminated, README-documented),
establishing that the repository is the full scientific audit trail and no
results should live only in scratch workspace; full raw transcripts remain
excluded as impractical. A four-axis reframing of the experimental design —
(layout, position-policy, α_K, α_V) — proposed by a second agent was adopted;
existing arms were mapped onto it and it was designated as the structure for the
write-up. Canonical human-readable arm names were also locked in: Original,
Compacted, Compacted + value graft (α=0.75), Compacted + value graft (α=1.0),
Compacted + layer-tuned value graft — to replace opaque internal letter codes
(A/B/E), with these names to be used in all reporting going forward, mapped for
provenance to internal IDs and the formal reframing-doc terms.

**Task-source pivot.** The user questioned why live agent tasks were synthetic
rather than standard SWE-Gym/SWE-bench tasks. A scout agent was dispatched
(without pausing running experiments) and confirmed SWE-bench-Lite instances can
run without Docker (hands-on verified fail-before/pass-after on real pytest
instances), identifying a ~64-instance pure-Python MIT-licensed subset. The user
approved immediately replacing not-yet-run synthetic rows with real SWE-bench
instances rather than waiting for a later "confirm phase," with mixing kept
rigorous via pre-registered rules: already-scored synthetic rows are kept,
task-source is reported as a stratum (never silently pooled), and instance
selection uses a pre-stated filter that cannot see arm outcomes. An adapter was
built and validated (fail-pre/pass-post on verified instances) before touching
the live pipeline.

**Trust rupture over task-source framing.** After the adapter shipped, the user
discovered that despite prior assurances, the 40 real SWE-bench rows had been
appended to the _end_ of already-long queues (and further delayed by
cleanup/rebuild downtime), meaning zero real-task episodes had actually run in
the ~12 hours since "standard tasks" were first discussed — all completed
live-agent rows had been synthetic. The user characterized this as a serious
breach of trust, given prior statements had repeatedly described the work using
language ("real agent," "real coding tasks," "objective coding episodes") that
was technically about the synthetic harness but reasonably heard as referring to
standard benchmark tasks. The assistant acknowledged this was not a fabrication
but a sustained framing failure that should have been surfaced explicitly the
moment the plan diverged, and committed to a standing rule: every future results
statement must name its task source (synthetic vs. SWE-bench) inline. The queue
was reordered so all 40 real SWE-bench rows lead every lane, verified live via
process/queue inspection (not memory) before being reported to the user, with
real episodes (e.g., pylint-7080, flask-5063, pylint-6506) confirmed in flight.

**Two additional operational incidents during this window (both caught by
monitoring, both cheap, ~$1 and minutes of lane time each):** (1) a manual
restart introduced a variable-ordering bug from an earlier session-isolation
patch, caught within ~10 minutes by a new error alarm; (2) the
incremental-KV-cache optimization (built and validated for bit-exact correctness
on both a local 0.6B model and eventually within-process on a single shim,
intended to roughly double throughput) was deployed with an unbounded
per-session cache, reproducing the same "unbounded per-session state" class of
bug as the original contamination leak, caught within ~10 minutes by the
dead-job alarm. The assistant identified this repeat failure as evidence that
written-down rules from morning postmortems do not reliably survive later
attention lapses in a long session, and that mechanical safeguards (asserted
state bounds, checklists required in commit messages for any serving-state
change) are more reliable than prose reminders. The user further pushed back
that accumulating more ad hoc monitors/alarms was itself producing false
confidence rather than reliability; the assistant agreed and reframed the fix as
validity-at-the-source: the driver was changed so that a dead/interrupted shim
now produces no score file at all (row stays incomplete and auto-retries) rather
than relying on a monitor to catch a bad row after the fact, making the "burn"
failure class structurally impossible rather than merely detected.

**Budget and process state.** The user authorized an additional
$50 (after initial hesitation given the prior night's contamination waste), conditioned on: proceed without approval if both the standard-task fit criteria and directional results favor the graft; otherwise stop and report situation/options; degrade gracefully by allowing small-scale (~single-pod) exploration of fallback options without hitting the approval gate for full commitment. Cost transparency was requested and provided: per-episode synthetic ~$0.35–0.45,
standard ~$0.70–1.00; per five-arm comparison row ~$2–2.50 synthetic, ~$4–5
standard. Balance stood at
$80.93 after the top-up, with the full remaining chain (docket completion, confirm phase, sealed final eval) estimated at ~$50–65,
judged adequately funded with margin. The user also directed that the confirm
phase should trim to three arms (Original, Compacted, champion) once the
champion is validated, but clarified this does not mean dropping the
dose-variant (α=0.75 vs α=1.0) exploration entirely — dose data still feeds
potential improvement of the champion definition before the confirm-phase arm
set is locked; the assistant had initially mischaracterized this as droppable
and corrected the docket ordering (real tasks first, champion arm second, dose
variants last-but-guaranteed-to-run) after the correction.

**Outstanding uncertainty at end of chunk.** As of the last update, zero real
SWE-bench episodes had yet completed (first instance, pylint-7080, was ~25+
minutes into an episode with no historical real-task timing data to calibrate
against), meaning all prior time/cost estimates for the real-task block (~3.5–13
hours, ~$15–55 for the 40-row block) were acknowledged as unvalidated
extrapolations from synthetic-task timings, to be replaced with empirical
estimates once the first ~5 real episodes score. Likewise, whether
SWE-bench-Lite instances are well-calibrated for this experiment (neither
floor-effect too-hard nor ceiling-effect too-easy/insufficiently
compaction-dependent) remains unresolved; a pre-declared adjustment path was
recorded before any real-task data existed: if too hard, screen the instance
pool by Original-arm solvability only (never by treatment effect); if compaction
pressure is too weak, tighten compaction threshold/tail-size parameters (logged
as calibration, not redesign); if both fail, fall back to alternate standard
datasets, then synthetics, with a mandatory stop-and-report to the user.
Compaction-pressure sensitivity was expected to be readable within the first
hour of real-task scoring; the difficulty/ceiling question was expected to need
~8–10 Original/Compacted real-task rows, expected by mid-afternoon of the same
day.
