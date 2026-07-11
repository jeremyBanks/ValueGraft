_This conversation covers the recovery from contaminated agent-run comparisons,
the hardened clean-run protocol, and the transition from synthetic coding tasks
to Docker-free SWE-bench-Lite evaluation. The current scientific question
remains unresolved: whether value grafting improves real coding-agent outcomes
under genuine compaction pressure._

**Participants:** User, claude-fable-5, and claude-sonnet-5.

**Handoff State.** The original shared-shim matrix is demoted to exploratory
evidence: cross-arm comparisons are suspect because session state leaked across
modes; the morning anomaly probe and dissociation data are void. Clean,
unaffected results include TF calibration, contamination guards, stage 1/2
replay, honesty replication (83% versus 17% decoy fabrication), and real-trace
recovery (+0.0156, n=75). All run scores are now versioned:
`results/agent_clean_run/` contains valid clean scores, while
`results/_QUARANTINE_agent_runs/` contains contaminated, void, artifact, and
superseded scores with explanations. Transcripts remain excluded as impractical.
`INCIDENTS.md`, `STATE.md`, `HANDOFF.md`, `DECISIONS.md`, `AGENTS.md`, and
write-up guidelines were updated; the intended deliverable remains a rigorous
blog-style write-up with paper-like experimental documentation and explicit
caveats.

The clean agent design uses Qwen3-30B-A3B-Instruct-2507 at bf16 through the shim
and OpenHands, with pytest scoring and no separate judge. Five canonical arms
are: Original; Compacted; Compacted + value graft (α=0.75); Compacted + value
graft (α=1.0); and Compacted + layer-tuned value graft. Internal IDs remain
frozen for provenance. The formal framing uses
`(layout, position-policy, α_K, α_V)`, with current experiments primarily
varying α_V; per-layer tuning is the validated practical candidate, while finer
slot/head optimization is rejected or guard-failed.

The clean 150-run design was pre-registered with three synthetic task templates,
ten seeds, and five arms, using disjoint task-to-lane assignment, isolation
probes, atomic completion-only writes, paired analysis, binomial intervals, and
McNemar comparisons. Synthetic tasks are deliberately seeded so constants vary
and cannot be guessed from convention. Early valid synthetic evidence was
favorable but small: Original 3/3, Compacted 2/4, graft α=0.75 4/5, α=1.0 1/1;
the broader clean count reached 16–17 valid rows. This suggests a possible
effect but is not decision-grade.

Several infrastructure failures were identified and structurally addressed. The
original session leak came from violating a one-task-per-pod assumption. Later,
an unbounded summary/cache snapshot caused GPU deaths; 102 apparent score rows
were purged as connection-error artifacts, with 16 genuine rows retained. A
cache optimization then repeated the same lifecycle mistake, causing roughly $1
and 30 lane-minutes of loss but no scientific corruption. The corrected pipeline
refuses to emit scores without completion evidence, health-gates lanes, bounds
and asserts cache state, records cache generations, and requires stateful-change
checklists. Monitoring now includes result-stream sanity checks, but the durable
principle is source-level invalidation rather than accumulating reactive
watchers.

Incremental KV caching was validated locally with a 0.6B model, with the
decisive equivalence test required within-process and then on a real A100 before
broad rollout. It is enabled only at task boundaries and never partway through a
task: all five arms for a task:seed must share one cache generation. The
optimization is expected to provide roughly 2× speedup; it must remain off if
A100 equivalence fails. Model weights can use a datacenter-local network volume
to reduce reloads from roughly 15 minutes to about 3 minutes, though volumes are
datacenter-specific and secure-cloud constrained.

The task-source correction is important. The live-agent track initially ran only
invented synthetic tasks; no live SWE-bench task had completed when the
discrepancy was discovered. Stage 2’s 75 real-trace replay episodes were clean
but are not live execution. The queue was rewritten so 40 SWE-bench-Lite rows
lead the lanes, with eight real instances × five arms, while already-run
synthetic rows remain a separate stratum. The standard-task adapter was hands-on
validated on `pytest-dev__pytest-5227` and `pytest-dev__pytest-7432`: both
failed before the gold patch and passed afterward. It handles cached repository
clones, local virtual environments, exact pytest IDs, malformed parametrized
IDs, and old setuptools compatibility. The viable filtered pool currently
contains 41 instances across pytest, requests, flask, pylint, xarray, and
seaborn; sympy was excluded because its test identifiers do not fit the exact
pytest-node contract.

Mixing synthetic and standard tasks is permitted only under explicit rules:
retained synthetic rows stay; replacements affect only unrun rows; task source
is reported as a separate stratum and never silently pooled; instance selection
cannot inspect treatment outcomes; and conditional-on-Original solvability
analysis is predeclared. Standard tasks are now the primary live-agent
evaluation, with synthetics retained for controlled mechanism calibration and
dose/per-layer comparisons. The confirm phase may trim to Original, Compacted,
and whichever champion emerges, but α variants remain required exploration
because they can improve the champion before confirmation.

Budget after the additional $50 was approximately $80.93. Estimated remaining
cost was $25–30 for the current exploration, $15–20 for a three-arm confirm
phase, and $10–15 for a sealed final evaluation, subject to real-task timing.
The autonomy rule is explicit: proceed without another approval only if standard
tasks demonstrably exercise compaction with a sensitive baseline and the paired
graft contrast favors grafting under the preregistered analysis. Ambiguous
results complete the planned n but do not trigger new phases. If validation
fails, stop high spending and report options; small one-pod probes for
recalibration or fallback datasets remain authorized.

At the end of the transcript, live real-task evidence was still zero. The first
real episode, `pylint-dev__pylint-7080`, had exceeded the synthetic-derived time
estimate, confirming that prior 8–15-minute and cost projections were guesses
about real-agent episode length. The honest range was approximately 8–60 minutes
per episode, with the first five scored rows intended to replace estimates with
empirical timing. Two preregistered validity questions remain open: whether
Original can solve enough real tasks to avoid a floor effect, and whether
compaction actually evicts load-bearing context rather than leaving Original and
Compacted equivalent. Pressure metrics should be available within the first
hour; roughly 8–10 Original/Compacted rows were expected to provide a difficulty
read by mid-afternoon. If fit fails, the declared sequence is
Original-solvability screening, logged compaction-pressure adjustment, alternate
standard datasets, then synthetics as fallback, with high-cost continuation
requiring review.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `ae8ff8e41b919f900`
- `afdf17117960af31b`
- `a556c5c256fa23d1a`
