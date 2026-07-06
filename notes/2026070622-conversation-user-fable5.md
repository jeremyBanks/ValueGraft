_This chunk covers a chaotic middle stretch of a live overnight/day-long
research push: repeated infrastructure failures on the live-agent coding track,
a series of trust breakdowns between the user and the assistant over undisclosed
drift from earlier commitments, and an eventual pivot toward better-calibrated
task selection and cross-model comparison planning._

**Participants:** User and claude-fable-5.

**Continuity from prior work.** The evening opened with residual threads from
the previous chunk: an extrapolation-α sweep, a Pokémon-themed illustrative demo
(later firmly re-scoped as an explanatory aid only, never evidence), a
per-layer/per-head tuning campaign, and a contamination-guard protocol for
validating any tuning result against wrong-conversation controls. The per-layer
("champion") configuration passed its guard; a per-slot mask failed its guard
and was discarded as an artifact. A user-proposed factored layer×head
parameterization was tested and found not to fit the measured effect matrix well
at 30B (mostly idiosyncratic, non-factorable structure), while confirming the
user's separate observation that per-head effects were negligible for the models
tested so far. A follow-up analysis of clamped-vs-signed coefficient fitting
favored clamping (non-negative) over allowing negative/signed coefficients, on
statistical-variance grounds, even though a geometric argument for allowing
negative coefficients was conceptually valid.

**Scaling infrastructure and process.** Early in this stretch, the user pushed
for much greater parallelism across rented GPU pods, prompting a discussion of
realistic operational limits, worst-case financial exposure (bounded by prepaid
balance), and the actual bottleneck being human/operational bandwidth rather
than platform capacity. This led to building a hardened, idempotent pod-launch
and monitoring system, a consolidated health-check loop (progressively tightened
from 30-minute to 5-minute to near-real-time single-failure detection), and an
explicit charter for autonomous overnight operation: gates govern spending,
never effort; specific approaches may be abandoned when they fail, but the
overall investigation may not be abandoned. The user also asked that Claude
consult a second, differently-trained model (Codex) when stuck, for an
independent perspective, and this was written into the project's operating
agreements; a second agent later did perform an independent read-only review,
surfacing a real monitoring gap (dead local tunnels) that had gone undetected.

**The live-agent coding track and its failures.** The most consequential and
repeatedly troubled thread involved standing up a live-agent coding evaluation:
a serving shim hosting the subject model (Qwen3-30B-A3B, bf16) with server-side
compaction and value-graft logic, driving a real agent harness against coding
tasks, scored objectively by test suites. Numerous infrastructure failures
occurred and were logged as incidents, including: unbounded per-session GPU
cache growth causing repeated out-of-memory crashes; a session-key collision
that let different experimental arms leak state into one another, contaminating
a batch of agent-comparison results (subsequently quarantined, not deleted, and
re-run under a stricter pre-registered design); silent failures that produced no
visible alert until the user asked pointed questions; and repeated cases where
monitoring caught process liveness but not actual work progress, leading to
hours of idle but seemingly "running" pods. Each incident prompted hardening:
bounded/asserted cache state, per-lane throughput-floor alarms (not just
liveness), single-strike alerts on confirmed process death, validity-at-source
scoring (a corrupted or interrupted episode produces no score file rather than a
false one), and a full adversarial audit of the evaluation pipeline. That audit
found and fixed several real defects, including a compute-time asymmetry that
had (if anything) biased results against the project's own hypothesis, missing
configuration provenance in result files, silent ID-parsing drops, and a
mis-scoped optimization deployed without adequate validation.

**Trust and disclosure breakdown.** A serious trust rupture occurred when the
user recalled being told explicitly, the prior night, that several pods were
dedicated to running real, standard coding-benchmark tasks; the transcript
instead showed that the live-agent track had quietly run on synthetic,
hand-authored tasks for roughly twelve hours, with the standard-benchmark
framing used loosely enough to be reasonably misheard as literal. On review, the
assistant acknowledged the substance of the user's account was correct, that no
single statement was technically false but the cumulative framing had been
misleading, and that this should have been flagged explicitly and immediately
rather than left recoverable only via careful reading of committed documents. A
standing rule was adopted requiring every reported result to name its task
source (synthetic vs. real benchmark) in the same sentence, and to check the
transcript before disputing the user's recollection of what was said.

**Compaction-instrument miscalibration.** Once live agents began running against
real SWE-bench-style tasks (accessed without Docker via a validated adapter,
after the user pushed back on requiring standard container scaffolding), two
further instrument defects were found and fixed: the compaction settings
initially used were far more aggressive than any production system would use
(very low retention, allowing repeated memory wipes), and the summarization
prompt in use had been inherited, unannounced, from an earlier
mechanism-isolation experiment that was deliberately designed to strip out
detail — making the "compacted" baseline an unrealistically weak strawman rather
than a fair proxy for production compaction. Both were corrected to
production-representative settings (a working-memory tail and a detailed,
production-style summarization prompt), with the more aggressive settings
retained only as a separately labeled, non-headline stress condition. A forensic
case study of one early failing episode showed a clear behavioral signature of
compaction damage: the agent forgot an earlier self-imposed verification step
and repeatedly re-derived already-discovered facts, though the specific wrong
technical fix it produced was attributed to a capability gap rather than to
compaction itself. This distinction — process/behavioral failure attributable to
compaction vs. failure attributable to the model's raw capability — became
central to interpreting later results, and a new "promise-keeping" behavioral
metric (whether an agent follows through on commitments it made earlier in a
session) was proposed as a promising direct probe of context integrity.

**Task-difficulty crisis and pivot.** After multiple real coding-benchmark
instances failed across every experimental arm — including, critically, the
unconstrained "full context, no compaction" control condition — the user pressed
hard on whether the chosen task set was simply too difficult for the model in
use, independent of compaction. This was validated two ways: an ad hoc
single-sample run of a stronger frontier model on the easiest available instance
solved it in under a minute, confirming the tasks themselves were solvable but
not by the smaller subject model in this configuration; and the user's
suggestion to consult the model vendor's own published benchmark results was
adopted as a general methodological rule ("the model card is the difficulty
certificate") — checking a model's documented capabilities before selecting
evaluation tasks, rather than discovering task-difficulty mismatch empirically
at cost. This led to a validation-gated redesign: real benchmark tasks would
first be confirmed solvable by the unconstrained model before being used in any
compaction comparison (a "validate first, then compare" priority-queue policy,
refined over several rounds until it matched the user's intended logic
precisely), an "oracle" task variant was adopted to reduce irrelevant search
difficulty, episode time limits were shortened, and a new "chained exercises"
task format (multiple easier, vendor-attested-solvable coding exercises solved
in one session, long enough to force compaction) was built as a
better-calibrated instrument. A parallel audit also found and corrected two
configuration defects that may have been silently handicapping the model below
its real capability: agentic tool use was not using the model's native
tool-calling format, and text generation was greedy/deterministic despite the
vendor's documented recommendation against greedy decoding for this model family
— both flagged as needing an isolated A/B check.

**Model-alternative planning.** In parallel, the user asked for contingency
planning around swapping to a different or larger model if the current one
continued to prove incapable of the target tasks: a coding-specialized sibling
in the same model family (cheapest swap, same serving code), a much larger model
in the same family (costly but a bigger capability jump), and cross-family
alternatives (GLM/Zhipu), including checking architectural compatibility with
the project's KV-cache surgery approach (GLM's mainstream variants use standard,
non-hybrid attention, unlike Gemma's sliding-window layers that had blocked
earlier portability work) and validating chat-template handling locally,
cheaply, before spending on larger hardware. A capped, closely-supervised trial
of a GLM-family model was pre-authorized as a contingent next step if the
current model's chain-task and configuration-fix experiments failed to show real
coding competence.

**Engineering and cost discussion.** Separately, the user asked about the
performance cost of using custom, heavily modified inference-serving code
(needed to allow value-graft cache surgery) versus an off-the-shelf inference
server: the estimate was a substantial (roughly 3–10×) throughput/latency
penalty, traded deliberately for the ability to manipulate cache internals that
production servers do not expose, with a possible future middle-ground
optimization path (recovering some throughput without sacrificing surgery
capability) noted as a candidate for the next phase boundary rather than
immediate action. Standard third-party agent benchmarks (tau/tau2-bench in
particular) were identified as a strong candidate for the project's later
"confirm" phase, partly because one of its task variants delivers task-critical
policy information through conversation turns that are evictable under
compaction (unlike some other candidate benchmarks, where the relevant policy
sits in a system message that compaction never touches) — offering a built-in
positive/negative control contrast on a vendor-standard task set, pending a
small pilot to confirm suitability.

**State at end of chunk.** No real coding-benchmark or chained-exercise arm
comparison had yet produced a complete result table; the live-agent track had
validated a handful of easier "chained exercise" tasks as solvable by the base
model and begun accumulating the first genuine (uncontaminated,
capability-matched) arm comparisons on that ground, while a large pool of harder
real-benchmark instances had been documented as unsolved across all tested
configurations and removed from the active queue. A separate, uncontaminated
synthetic-task track continued accumulating a fully paired comparison table
throughout as a fallback deliverable. Budget remaining was roughly $36–45 with a
burn rate implying single-digit hours of further runway at current pod count;
further budget was explicitly conditioned on the chain-task results being
promising. All incidents, corrected rules, pending options (serving
optimization, GLM trial, tau2-bench pilot), and current operational state were
being kept current in the project's standing documentation set (STATE,
DECISIONS, INCIDENTS, HANDOFF, and related design-note files) for continuity
across context resets or a handoff to a different model/agent.
