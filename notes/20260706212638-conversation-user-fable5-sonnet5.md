_The project abandoned further SWE-bench validation after seven collected
failures and redirected remaining budget to capability-gated chained exercises,
while adopting vendor-published benchmark tables as the difficulty-calibration
starting point. Benchmark scouting identified tau2-bench `banking_knowledge`
plus core tau domains as the strongest future confirm-phase design for testing
compaction-sensitive policy retention._

**Participants:** User, claude-fable-5, and claude-sonnet-5.

**Handoff State.** Sixteen queued SWE rows were purged; the seven existing
verdicts were deemed sufficient for the scoping claim. Five chain validations
were parallelized across three lanes, with four of six chains validated (s1, s3,
s4, s6), 13 constrained-arm rows available, and only one substantive arm datum
so far: B’s 4/4 on s1. The first multi-chain comparison table was expected
within a couple of hours. Approximately $36.50 remained at roughly $5.50/hour,
providing 6–7 hours of runway—enough for the queued arm rows but not additional
speculative work. The chain arm table is the evidence required to justify more
budget; GLM testing, tau2 work, and serving upgrades remain blocked behind it.

A new standing rule makes the vendor model card or benchmark table the first
difficulty certificate for future experiments. The observed boundary—repo-fixing
tasks infeasible, exercise-scale chains feasible, and tool-agency tasks
intermediate—matched vendor-reported capabilities and was treated as
cross-validation of the instrument. Implausible outcomes, whether unusually
clean or catastrophic, now require inspection before acceptance, and error nets
must enumerate realistic phrasings from both sides of the interface.

Benchmark scouting, including a switch from the main `claude-fable-5` flow to
`claude-sonnet-5` subagent assessments, ranked candidates as follows: tau2-bench
`banking_knowledge` first; classic tau2/tau domains as negative controls; BFCL
`memory_*` categories only with harness changes; BFCL core as too short; and
Aider-Polyglot as lacking persistent context. The proposed confirm experiment
compares `banking_knowledge`, where policy arrives through evictable user turns
and retrieved documents, against core tau domains, where policy remains in the
unevictable system message. Expected compaction damage only in the former would
provide a clean causal dissociation on standard, vendor-calibrated tasks.

`banking_knowledge` contains 97 MIT-licensed tasks and appears compatible with
the existing OpenAI-compatible shim through LiteLLM; it also requires a second
model to simulate the user. Because it is newer, gated behind an extra install,
and less battle-tested, the plan requires an empirical 2–3-episode pilot
confirming actual token lengths and deterministic reward behavior before
committing calendar time. Findings are recorded in
`vendor-benchmarks-scouting.md`, with the confirm design also reflected in
DAY-PLAN and DECISIONS/STATE.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `a068bbb44e3edafe7`
- `a65f4d26fd3f48e34`
