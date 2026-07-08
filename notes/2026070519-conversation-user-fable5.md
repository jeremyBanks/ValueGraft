_This chunk covers recovery from a memory-thrashing incident on the local 30B
benchmark run, a user-directed strategic pivot of the cloud-compute plan toward
standard-benchmark evidence on the primary model rather than cross-model
variety, execution of a four-pod parallel cloud campaign (LongMemEval, SWE-Gym
trajectory replay, and a dedicated tuning track), and a validation-discipline
case study where an apparently striking per-head finding at 4B was killed on
holdout._

**Participants:** User and claude-fable-5.

**Incident and instrumentation.** The local Qwen3-30B benchmark batch had
stalled roughly three hours due to a bad `pkill` pattern that left two model
instances resident and thrashing swap; the user directly registered concern
about lack of proactive care. The fix (kill by PID with pgrep verification) was
already recorded in persistent memory. New standing instrumentation was added: a
progress-based watchdog (alarms on 15 minutes without a new result file,
independent of PID tracking) and, per user request, a periodic memory-pressure
watchdog (checks every 5 minutes, alerts on low free memory or high swap). The
30B batch then completed normally (36 questions), was judged, and produced the
local 30B benchmark table.

**Strategic pivot on cloud plan.** After a layer-profile experiment showed a
mid-depth band (roughly layers 12–22 of 36, peaking near layer 17) where
single-layer value grafting recovered a meaningful fraction of the tuned effect,
the user asked whether a profile-derived, low-degree-of-freedom calibration
procedure (rather than hand-tuned constants) could be validated and then used to
test transfer across models and architectures. This was accepted, with the
derived-layer-set rule (α=0.75 on profile-positive layers) validated on holdout
and found statistically tied with the hand-tuned mid-band champion — confirming
the _procedure_ works, not that it beats manual tuning outright.

This opened a broader planning discussion in which the user corrected the
drafted cloud plan's emphasis. The user's position, adopted as project policy:
the primary purpose of renting cloud GPU time is not to test different models
but to get statistically powered, industry-standard-benchmark evidence for the
model the project already understands well (Qwen3-30B-A3B, bf16, on a single
A100-80GB); testing model variety (Mistral, Llama, Gemma) is a legitimate but
secondary "bonus" objective, contingent on budget, and should never involve
downgrading to an older/weaker model just for architecture diversity when the
current model is already the strongest available. The plan was restructured
accordingly with the ordering: identity ladder → full-scale LongMemEval →
SWE-Gym agent-trace replay → a second published benchmark (LoCoMo/SCBench) →
homemade synthetic probes (demoted, supporting-evidence only) → model-variety
bonus stages (Mistral Small 3.2/Ministral-8B, contingent Llama-3.3-70B) →
contingent Gemma 3 27B hybrid-attention profile (requires new per-layer-type
surgery machinery, deferred to last). The user also raised, and the assistant
validated, why Llama-3.3-70B specifically was chosen (the only architecturally
"clean" dense 70B still using conventional full attention + RoPE + per-layer
K/V, since most newer open models — Llama 4's iRoPE, DeepSeek/GLM's MLA, Gemma's
sliding-window hybrid — break assumptions the KV-cache surgery relies on).

A related methodological clarification: the local controlled LongMemEval runs
(subsampled ≤16K contexts, evidence deliberately placed for eviction) are _not_
comparable to published LongMemEval leaderboard numbers; they are a paired-arm
experiment built from LongMemEval's materials. The user asked for something
closer to the standard protocol, and a "stage 1b" was approved: full ~115K-token
haystacks with the benchmark's own judging-style prompts, giving a citable
standard-protocol Qwen3-30B baseline plus compaction-impact and recovery numbers
anchored to it.

**Credentials, hardware, and cost.** The user supplied a RunPod key and a
Hugging Face key, both explicitly held un-used until separate go-ahead; both
were verified gitignored and permission-restricted. Hugging Face gating was
checked for every model in the plan: only Gemma 3 27B (already approved) and
Llama-3.3-70B (approval initially pending, later granted) required manual
license acceptance; Qwen3, Mistral variants, and both benchmark datasets are
ungated. Hardware strategy, explained to the user (unfamiliar with cloud GPU
rental): single A100-80GB pods, one workload at a time by default, secure tier
first with a switch to cheaper community tier once stability is confirmed,
serial provision-run-terminate discipline to avoid idle billing, at roughly
$1.39/hr (secure) or $1.19/hr (community). Initial budget was $50; after
reviewing stage cost estimates the user added $50 more (to $100 total)
explicitly "to not compromise," with the further $100 contingent on strong
results plus the assistant's recommendation and fresh user agreement — this
budget policy was recorded in the plan documents.

The user asked whether the campaign could be parallelized across pods without
materially raising total cost; this was approved (up to ~20% cost premium
acceptable) since GPU-hours are largely fungible across concurrent pods for
independent, resumable, atomically-written work items. A sharding scheme
(`SC_SHARD=k/N`) was implemented and multiple pods were brought up to split
LongMemEval shards and run SWE-Gym trajectory replay in parallel, followed by a
fourth pod dedicated to a tuning track. A minor process-hygiene near-miss (a
stale unsharded runner briefly coexisting with a new one, the same failure shape
as the morning incident) was caught and killed by PID within a minute; a new
rule was adopted to verify GPU memory residency (via `nvidia-smi`) after every
pod model load, after one pod silently CPU-offloaded due to a stray process
holding VRAM.

**Tuning track.** In response to the user's explicit request for per-head tuning
on the actual deployed (bf16) models rather than reusing constants tuned on the
local 4-bit stack, a dedicated pod ran a staged tuning pipeline: global-α
re-sweep, per-layer profile, and a per-KV-head profile (192 slots at 30B), all
restricted to a validation split, with candidate low-degree-of-freedom rules
evaluated exactly once on a holdout split (pre-registered adoption rule: replace
the incumbent α=0.75 if it wins with a CI excluding zero, adopt the bf16-tuned
value anyway if statistically tied since it is the correct dose for the deployed
precision, otherwise keep the incumbent). A notable finding during the α sweep:
at bf16 on the 30B model, α values above 1.0 (extrapolation past the "old" value
vectors, subtracting a fraction of the "fresh" ones) began outperforming
blending — interpreted as a contrastive-steering effect analogous to
classifier-free guidance, attributed to bf16 removing quantization noise that
previously destabilized extrapolation at 4-bit. An extended α bracket (up to
3.0) was queued to find where this effect peaks and where it collapses into
degenerate output. At 4B, a parallel local per-head profile was run out of
curiosity per the user's low-priority request; a striking-looking result (effect
concentrated in one KV head) failed decisively on holdout (win rate collapsed
from a promising validation signal to near coin-flip), reinforcing that the
validated structure at 4B is the layer axis only, not the head axis — documented
explicitly as a demonstration of why the holdout discipline exists. The
analogous 30B per-head evaluation was in progress at the end of this chunk. A
Mistral Small template adapter was under construction so pre-tuning for the
model-variety bonus stage could run on the same pod immediately after the Qwen
phases, without idling the GPU; several Mistral checkpoints lack a chat
template, narrowing the practical choice to Ministral-8B-Instruct-2410,
Mistral-Small-Instruct-2409, or Mistral-Nemo-Instruct-2407.

**Results reported to the user.** Stage 2 (SWE-Gym real coding-agent trajectory
replay, n=75, off-policy caveat noted) completed with a statistically
significant but modest recovery: full-context per-token confidence ~74% vs.
compacted ~62%, tuned graft recovering to ~63% (about 10% of the
compaction-induced damage), mean effect +0.0156–0.0166 nats, 95% CI excluding
zero, roughly 44–45/75 paired wins. This was explicitly framed as smaller than
the ~24% closure seen on the synthetic corpus and larger than the near-zero
effect on free-form chat, consistent with the standing thesis that recovery
scales with how much a continuation actually depends on evicted content. The
user accepted this as adequate given the instrument's acknowledged brittleness
(single arbitrary cut, off-policy replay, token-exact scoring), and the effect's
survival on this relatively unforgiving test was read as a lower bound rather
than the full picture, reinforcing the case for a follow-up end-to-end
OpenHands-based agent-loop evaluation as the default next step if budget allows.
Pre-registered fallback experiments for a null SWE-Gym result (dependence
stratification, span-level rescoring, targeted re-cuts around key trajectory
events, discrete/behavioral rescoring) were documented before results were seen,
along with an explicit acknowledgment (raised by the user) that the offline
trajectory-replay design itself may be too artificial regardless of outcome,
making the OpenHands end-to-end run potentially valuable even in a muddy-null
scenario, not just a clean-signal one. A standard, pluggable extension point for
this was identified: OpenHands' "condenser" abstraction and SWE-agent's
history-processor hooks support pluggable compaction policies at the harness
level, but the serving side (vLLM/SGLang) does not expose the KV-cache editing
needed, so a thin OpenAI-compatible shim around the existing HF/DynamicCache
machinery would be required — scoped as a follow-up project (~1 day engineering,
~$50–100 GPU) rather than in the current budget.

**Operational housekeeping.** Coordination documents (`cloud-plan.md`,
`STATE.md`, `DECISIONS.md`) were updated repeatedly at the user's insistence to
keep them current for a fresh agent, including the hardware/utilization
strategy, the parallelization decision, the stage-1b approval and its
non-comparability caveat, the tuning-adoption rule, and the incident/mitigation
history. Statistical-significance reasoning for the SWE-Gym effect size was
walked through in detail (paired-design standard error at n=75, detection floor
around +0.008 nats) at the user's request. A local disk-space concern was
resolved by deleting unrelated, re-downloadable local model files (all Ollama
models, and an itemized list of LM Studio models led by a 17 GB Gemma
checkpoint) after user confirmation, freeing roughly 100 GiB. A repeated
false-alarm problem in the multi-pod monitoring setup (simultaneous SSH probes
across pods tripping connection limits, and a bash-only script being invoked
under `sh`) was fixed by consolidating into a single staggered, retry-hardened,
file-state-based watchdog script, reducing spurious wake-ups per the user's
standing request to conserve their usage quota while keeping substantive updates
detailed. The user also flagged, as a non-urgent note for the record rather than
an action item, that the repository should be cleaned up before any broader
sharing of results — explicitly not to be acted on now — which was logged in
`STATE.md`.

At the end of this chunk: four cloud pods have been active (two sharded
LongMemEval, one SWE-Gym — completed and terminated — one tuning track), stage 1
(LongMemEval, 350 questions) is nearing completion, stage 1b (full-haystack
protocol) and stage 3 (second published benchmark, adapter not yet built) remain
queued, the 30B per-head tuning holdout evaluation is in progress, and account
balance was tracking near $79–95 of the authorized $100, on pace with estimates.
