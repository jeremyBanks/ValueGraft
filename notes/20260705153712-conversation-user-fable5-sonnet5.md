_This conversation covers the transition from local validation to a budgeted,
parallel cloud evaluation centered on standard benchmarks, with Qwen3-30B as the
primary model and cross-model testing as a secondary benefit. It also records
the approved full-haystack LongMemEval protocol, agent-trace evaluation,
operational safeguards, budget, and current three-pod execution state._

**Participants:** User, claude-fable-5, and claude-sonnet-5.

**Research conclusions.** Local LongMemEval used standardized third-party
material but not the published protocol: sessions were subsampled into
≤16K-token contexts, evidence was deliberately placed in an evicted region, and
results are paired experimental contrasts rather than leaderboard-comparable
scores. Full-context accuracy was 71% (4B) and 81% (30B), while standard
compaction reduced every arm to ≤11%; H-pack reduced fabrication at 4B by
roughly 35% (11/48 versus 17/48), but showed no benefit at 30B on personal-QA
framing because refusal behavior left little fabrication headroom. The mechanism
remains supported by 12/12 paired wins at 30B and approximately 24%
continuation-gap closure, but recall recovery is not demonstrated. A newly
approved stage 1b will run approximately 100 full ~115K-token haystacks with the
benchmark’s own prompts and the same compaction arms, providing a
standard-protocol baseline while explicitly noting that Sonnet judging remains a
deviation.

The per-layer profile found a mid-depth hump (L12–L22, peak L17), with late
layers harmful. A validation-derived layer-selection procedure scored +0.0182 on
8/10 holdout conversations versus +0.0173 for the manually tuned champion; full
replacement remains too strong at 4B. This validates the calibration procedure
as a transfer hypothesis, not a universally optimal recipe. Gemma hybrid
attention is deferred from primary replication because sliding-window layers
confound comparison, but a profile-first Gemma 3 27B experiment remains a
contingent follow-up; MLA architectures remain a hard incompatibility without
new machinery.

**Plan and priorities.** The user corrected the earlier emphasis on cross-family
replication: rented hardware is primarily for statistical power, full benchmark
protocols, and real agent trajectories on the best-understood Qwen3 model. The
committed order is identity ladder/setup; LongMemEval controlled run (350
questions); approved full-protocol LongMemEval stage 1b; SWE-Gym/OpenHands
coding-agent traces (~75); a second published benchmark (LoCoMo or SCBench);
then, budget permitting, homemade leakage-audited probes, Mistral variety tests,
Llama-3.3-70B scale testing, and Gemma hybrid profiling. The intended document
form remains a blog-style draft, led by standard-benchmark evidence and using
synthetic probes only as mechanism support.

**Operations and budget.** The strategy is one A100 80GB pod at a time where
practical, with resumable per-item outputs, result synchronization, local
judging/analysis off-meter, and termination between stages. Parallel execution
was approved with up to roughly 20% cost premium: stage 1 is sharded across two
pods, and stage 2 runs on a third. Pod-1 is secure at $1.39/hr, pod-2 community
at $1.19/hr, and combined burn is approximately $3.80/hr. The user added $50,
bringing the available balance to $98.44; further $100 is conditional on very
promising results, an explicit recommendation, and fresh user agreement. A
previous three-hour swap-thrashing incident led to PID-specific process
verification, progress-based watchdogs, five-minute memory checks (alert below
15% free memory or above 25GB swap), and strict stale-runner cleanup. A
transient SSH watchdog alert was manually checked and found false; both pods
were healthy and progressing.

**Handoff State.** Cloud execution has explicit approval. Stage 0 passed on CUDA
after a full six-arm local smoke test; a symlink-copy issue was fixed by copying
dereferenced model bytes. Stage 1 is running across pods 1–2, with roughly 11
results on pod-1 and 10 on pod-2 at the last check, slightly ahead of
projection; the balance was $97.52. Stage 1b is queued after stage 1, with a
one-question smoke test first. Stage 2’s runner passed local smoke testing (n=1
showed A −0.510 versus B −0.586) and pod-3 is downloading the model before its
~75-trajectory run. Incremental Sonnet judging is planned once sufficient
stage-1 results arrive; the assistant model switched from claude-fable-5 to
claude-sonnet-5 for judging assessments. Core results were expected tonight,
stage 1b overnight, the second benchmark by tomorrow morning, and the full
budget permitting plan by tomorrow evening, subject to adapter builds, pod
availability, and incidents.

All coordination changes were committed to `cloud-plan.md`, `STATE.md`, and
`DECISIONS.md`; `AGENTS.md` remains unchanged. These files now capture benchmark
ordering and comparability caveats, shard assignments, pod state, credential
rules, budget/timeline, stage designs, and process-hygiene requirements.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `add36226b64075bc0`
- `a8da6fba093198de0`
