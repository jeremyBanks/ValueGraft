_This conversation documents the recovery and redesign of a coding-agent
experiment after infrastructure, configuration, task-difficulty, and
compaction-calibration failures. The project pivoted from SWE-bench-Lite
comparisons to capability-gated, easier chained exercises, while preserving
synthetic coverage and staging a possible GLM-4.5-Air comparison._

**Participants:** User, claude-fable-5, and claude-sonnet-5.

**Measurement and instrumentation.** Episode records now retain binary outcome
plus SDK steps, token/compaction traces, wall time, recall-probe text, source
labels, diff lines/files, self-termination, timeout status, and configuration
provenance. Timeouts are valid within-budget failures when the episode ran, and
accidental fixes are distinguishable from declared completion. Process metrics
are secondary endpoints only; pass rate remains primary. Promise-keeping—whether
agents execute verification steps they committed to early—is an additional
pre-declared behavioral measure.

**Infrastructure lessons.** The withdrawn cache optimization failed because peak
memory was never budgeted: cloning a 5–8 GB cache alongside a 61 GB model,
activations, and interleaved caches exceeded an 80 GB GPU. Local validation
checked correctness on a tiny CPU model but not realistic VRAM arithmetic. A v2
design uses in-place cache extension, context-size admission limits,
predicted-memory guards, and degradation by dropping cache rather than killing
the process; it passed local correctness/guard tests and was canaried on one
lane. Health monitoring repeatedly confused process liveness with useful
progress. The current design prioritizes outcome monitoring: per-lane score-rate
alarms, invalid/error detection, impossible-data checks, and diagnostics only
after an outcome alarm. A local tunnel/wait-state gap survived the
science-focused audit and was added to the incident and coverage map.

**Compaction configuration.** The original tier used a 9K trigger, 2.5K tail,
and deliberately detail-free 300–500-token summary, causing severe
working-memory loss and pathological re-derivation. For production-calibrated
runs the selected regime became a 12K trigger, 6K verbatim tail, and detailed
summary containing files, changes, state, constraints, and next steps; expected
recompactions are 2–6 per episode. Summary length remains a future sensitivity
variable because a roughly 500–650-token summary compresses increasingly large
histories late in long episodes. In graft arms, source values come from the tail
and summary in their original full-context encoding; the tail is not duplicated
on the source side, while the new compacted layout is freshly encoded and then
grafted.

**Audit findings and contamination control.** A hostile pipeline audit found
three critical and two major issues, including asymmetric Original-arm
recomputation under a common wall-clock limit, missing configuration provenance,
silently dropped test IDs, and inaccurate compaction counts. These were fixed
and deployed; prior rows are labeled by configuration/validity stratum. The
audit also established durable rules: re-derive inherited parameters at purpose
boundaries, audit baselines as rigorously as interventions, make instruments
self-report configuration, validate each task source with its gold patch, and
perform capability smokes before expensive comparisons. Earlier invalid rows
from tunnel/shim failures are excluded rather than scored.

**Task-difficulty pivot.** Initial Docker-free SWE-bench tasks produced zero
passes: full-context validation failed on seaborn-3190 and pytest-5495, and
oracle-guided validation subsequently failed on pylint-7080, seaborn-3190,
pytest-11143, requests-5414, and others. Since validation cost scales
approximately as \(1/p\) with full-capability solve rate, the original dataset
was judged too difficult for affordable signal with the Qwen3-30B instruct
model. Plain real-task runs were stopped/replaced, a 30-minute cap was imposed,
and easiest-first ordering was adopted. A dynamic priority queue now sorts first
by whether an instance has passed Original validation, then by priority of
constrained-arm work; validated instances immediately outrank further
validation.

**Fallbacks and new task source.** SWE-bench Verified’s human-rated “under
15-minute fix” subset was merged and interleaved with the remaining easy
instances, while BugsInPy remained a secondary, less-validated option and
SWE-smith was rejected as Docker-only. The stronger fallback is a
chained-exercise tier: four seeded, easy coding exercises in one conversation,
with deterministic selection/order, individual scoring, and an embedded recall
probe. `chain_tasks.py` passed pre-solution failure and reference-solution
success checks, deterministic seed checks, token-collision checks, and
compilation. Original-arm capability smokes must pass before arm comparisons;
successful instances then receive immediate constrained-arm priority. A
synthetic lane is independently completing paired quads and is expected to
provide analyzable data even if real-task work fails.

**Scaffold/model audit.** The subject model may have been artificially
handicapped: native tool calling was disabled, and the shim’s decoder ignored
the configured temperature because its loop was hardcoded to argmax. These were
recorded as major instrument findings; all prior solve rates are lower bounds,
though arm symmetry preserves internal comparisons. A controlled A/B on the
failed pylint control was started with native tool calling and seeded
recommended sampling. The custom Python/Torch server is estimated at 3–10×
slower than mature serving stacks because of missing prefix reuse,
less-optimized attention, serial serving, and unbatched decoding. Future work
should retain KV surgery while adding flash attention, compiled/efficient
decode, asynchronous serving, and validated in-place cache reuse; this is first
priority at the next phase boundary, not during the current diagnostic run.

**Model alternatives and current state.** Sonnet-5, used as a subagent
calibration probe, solved the easiest oracle task in 52 seconds with nine tool
calls and passed 127 capture tests, confirming that the task format is solvable
and that the Qwen3-30B capability/scaffold combination is the main bottleneck.
GLM-4.5-Air and GLM-4-9B tokenizers passed context-construction dry runs. The
GLM-9B local serving smoke OOM’d due to insufficient margin and was not rerun;
the representative gate is a dedicated GLM-4.5-Air single-pass smoke. A possible
trial is pre-authorized for at most two expensive pod-hours: first
full-capability validation, then immediate arm variants if successful, with
tight supervision.

At the end of the transcript, five SWE validations had failed, chain:s1 was the
first full-capability chained episode in progress, s2/s3 were queued, the
scaffold A/B was awaiting a healthy shim, and synthetic paired data continued
accumulating. The intended near-term milestone is the first successful chain
validation followed by a clean Original-versus-constrained arm comparison; if
chains also fail under the corrected scaffold, the next decision is the GLM
trial or a synthetic-only result.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `a5b199fc910b8dfee`
- `a667ec05ec256753f`
- `ad30637d597792e87`
- `af37b240a577c61d7`
- `a8bf835589ee6b625`
