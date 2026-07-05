_This chunk covers recovery and completion of the interrupted 30B local
benchmark run, a per-layer graft-profiling investigation that found and
validated a mid-depth calibration procedure, and an extended cloud-plan planning
dialogue that repeatedly corrected the model-selection rationale before
execution began on a multi-pod RunPod deployment._

**Participants:** User and claude-fable-5.

**30B batch recovery.** The Qwen3-30B run that had stalled from the earlier
swap-thrash incident resumed cleanly after the PID-pattern kill bug fix (tracked
in memory as [[verify-kills-by-pid]]); a stale duplicate monitor was stopped,
and a new progress-based watchdog (keyed on new result files, alarming on 15
minutes of stall or process death) plus a separate memory-pressure watchdog
(5-minute cadence, alerts below 15% free or over 25GB swap) were added per user
request to catch both failure modes going forward. The 36-question batch
completed, was judged, and the 30B benchmark table was produced.

**Layer-profile investigation.** Following a user-provided note
(`per-head-alpha-speculation.md`), a per-layer graft profile was run (36 layers
× 20 conversations) on the 4B model. Results showed a consistent mid-depth band
(L17–L22 of 36, later refined to L12–L22 with peak at L17) where single-layer
grafts recovered meaningful fraction of the full tuned effect, while late layers
were actively harmful. Per user direction to consider whether this refinement
was worth pursuing on the more expensive GPU, a disciplined calibration
procedure was designed and validated: derive a graft/no-graft threshold from
validation-only conversations (a low-DOF rule, not per-layer fitting), then
evaluate once on a holdout set. The derived-layer-set holdout result (α=0.75,
+0.0182, 8/10) was statistically a tie with the hand-tuned mid-band champion
(+0.0173, 10/10), while full replacement even on good layers stayed too strong
at 4B. This validates the _procedure_ (automatic calibration reproducing manual
tuning) as the transferable artifact for cloud testing, rather than any specific
layer numbers. A follow-up hypothesis was logged for hybrid-attention
architectures (e.g., Gemma 3's sliding-window layers): profile mass should
concentrate on the sparse global-attention layers if the mechanism
(cross-context interpretation requires cross-context attention) is correct —
noted as future work, with Gemma excluded from the core replication because its
architecture dilutes the premise.

**Cloud-plan model-selection corrections.** Initial cloud planning drifted
toward prioritizing cross-family model diversity (Llama-3.3-70B as scale anchor,
Mistral Small 3.2 as second family, contingent Gemma 3 27B hybrid profile),
justified by architectural constraints ruling out most newer open models
(MLA-based DeepSeek/GLM, iRoPE Llama 4, sliding-window Gemma, linear-attention
hybrid Qwen variants all break the ValueGraft mechanism's K/V assumptions). The
user repeatedly corrected this framing: the actual reason for renting hardware
was to scale up testing volume and validate against established,
industry-standard benchmarks and data sets, not to swap in a different (and
specifically not an older/less capable) model for its own sake; model diversity
is a valid but secondary bonus goal. This correction produced a full plan
restructure, ordering stages as: full-scale LongMemEval (all 500 questions) →
real coding-agent trajectories (SWE-Gym/OpenHands, interpreting the user's
"genetic programming" phrasing as agentic programming, pending confirmation) → a
second published benchmark (LoCoMo or SCBench) → homemade synthetic probes
demoted to a small clearly-labeled stage 4 (only for leakage-audited fabrication
decoys, which standard benchmarks can't provide) → model-variety stages
(Mistral, Llama-3.3-70B) and Gemma hybrid profile demoted to budget-permitting
bonus stages. This ordering — standard benchmarks first, homemade material last
— is now the credibility-ordering principle for the eventual write-up.

**Benchmark protocol clarification.** In response to a direct question about how
meaningful the local LongMemEval result was, it was clarified that the local run
is not run under LongMemEval's standard protocol: sessions were subsampled to
≤16K-token contexts (far fewer distractors than the standard ~115K-token full
haystack) with evidence deliberately positioned for eviction, so results are a
controlled paired-arm contrast built from LongMemEval materials, not a
comparable LongMemEval score. Because the cloud model supports 262K context and
a full 115K-token haystack fits comfortably in bf16 KV cache, a "stage 1b" was
proposed to run the genuine standard protocol (full haystacks, the benchmark's
own judging prompts) for a citable baseline plus compaction-impact numbers
anchored to it. The user approved stage 1b explicitly.

**Recap of local LongMemEval findings** (as reported mid-conversation):
full-context accuracy 71% (4B) / 81% (30B); after standard summary-compaction
this collapses to ≤11% for every method including the project's own — recall of
evicted facts is not recovered by any current arm (an honest null). The
anti-fabrication effect is real but conditional: at 4B, the packed write-time
summary approach fabricated ~35% less than production-style compaction; at 30B
on this benchmark's personal-QA framing, the effect flattens because the larger
model already refuses when uncertain, leaving no fabrication headroom —
contrasted with the project's own synthetic agentic conversations, where the 30B
baseline fabricated at a much higher rate and the intervention cut that
substantially. The scope conclusion recorded: the anti-fabrication benefit
applies to mid-task agentic compaction framings (matching production
compaction), not retrieval-style QA.

**Credentials and gating.** A HuggingFace token was added (gitignored, verified
untracked, owner-only permissions) alongside the existing RunPod key, both held
under the same no-use-without-explicit-go rule. Gating check confirmed:
Qwen3-30B-A3B, Mistral variants, and both target datasets (LongMemEval, SWE-Gym)
are ungated; Gemma 3 27B was already approved; Llama-3.3-70B approval was
pending and arrived during the conversation, after which the user granted final
go-ahead.

**Hardware and budget.** Strategy settled as one A100 80GB pod at a time
(right-sized for Qwen3-30B-A3B's ~61GB bf16 footprint; not the most powerful
available card, since the workload is memory-bandwidth-bound, not
compute-bound), serial provision→run→rsync→terminate cycles to keep idle time
near zero, with a persistent network volume considered if more than two stages
run. Initial estimate: core stages (0–3) ≈ $32–38 within the existing ~$50
balance; full plan including 70B anchor ≈ $85 worst case. The user added a
further $50 (bringing the balance to ~$98–100) to avoid compromising scope, with
a conditional additional $100 contingent on stage 1–3 results being promising
and on model recommendation plus fresh user agreement at that time.

**Parallelization.** In response to a user request to consider speeding up
execution without materially increasing cost, work items (questions,
trajectories) were confirmed independent and atomically resumable, enabling
near-cost-neutral sharding (`SC_SHARD=k/N`) across multiple pods — fixed
overhead only ~$0.40/extra pod for model download. The user authorized
parallelizing to whatever degree was practical, accepting up to roughly a 20%
cost premium for faster results. Three pods were brought up: pod-1 (secure tier,
stage-1 shard 0/2), pod-2 (community tier, stage-1 shard 1/2), pod-3 (stage 2,
SWE-Gym coding traces). A process-hygiene near-miss matching the morning's
original incident shape (a stale unsharded runner beside the new sharded one)
was caught and killed by PID within a minute.

**Execution status at chunk end.** HF-transformers port of the surgery core
passed its identity ladder on CUDA
(~$0.35 spent). Stage 1 (LongMemEval, capped at 350 questions rather than the full 500 because measured per-question cost, ~120s/question, ran higher than estimated — full 500 would have cost ~$23)
is running sharded across pods 1–2; stage 1b (full-haystack standard protocol,
~100 questions) is approved and queued to follow on the same pods; stage 2
(SWE-Gym, ~75 trajectories) is running on pod-3. Combined burn ~$3.80/hr across
three pods, balance $97.52 after the $50 top-up landed. A watchdog "pods
unreachable" alert was investigated and confirmed a false alarm (transient SSH
probe failure); both pods verified healthy and on pace.

**Documentation and timeline commitments.** All four coordination documents
(`cloud-plan.md`, `STATE.md`, `DECISIONS.md`, `AGENTS.md`) were updated at two
points in the conversation to capture the strategy pivots, hardware rationale,
parallelization decision and near-miss, stage 1b approval with its
non-comparability caveat, and current operational state (pod addresses, shard
assignments, key paths, queue), explicitly so a successor agent could pick up
the plan without additional context. Timeline given (relative to the point of
estimation): stage 1 shards finish and are judged same night; stage 1b runs
overnight once stage 1 frees the pods; stage 2 results same night in parallel;
stage 3 (second benchmark) adapter built that evening, runs overnight, scored
the next morning; core-plus-second-benchmark synthesis by the next late morning;
bonus stages (homemade probes, Mistral variety, Llama-70B, Gemma hybrid) the
following afternoon/evening — full plan estimated at roughly 30 hours end-to-end
and $85–95 of the ~$100 budget, with 70B pod availability and the two adapter
builds (stages 2/3) flagged as the main schedule risks.
