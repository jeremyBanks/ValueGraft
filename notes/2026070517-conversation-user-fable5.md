_This chunk covers the resolution of the Qwen3-30B GPU crash, full execution of
the cloud phase (RunPod), a per-layer/per-head tuning campaign including an α>1
"value steering" finding, and ongoing benchmark stage results — spanning the
transition from crisis response through disciplined multi-pod parallel
execution._

**Participants:** User and claude-fable-5.

## Incident resolution and instrumentation

The morning's 30B GPU-memory crash (root-caused: `pkill -f` pattern missing an
argv flag, leaving two 30B instances thrashing swap for ~3 hours) was fully
resolved: the relaunched run recovered to normal ~2 min/question pace,
completing the 36-question local 30B batch and judging. Two standing watchdogs
were installed as a result: a progress-based monitor (alarms on 15 minutes
without a new result file or process death, independent of PID tracking) and a
memory-pressure monitor (5-minute cadence, alerts below 15% free memory or above
25GB swap, lists offending processes). A monitor script bug (bash-only syntax
run under `sh`) caused false "unreachable" alarms later in the day; this was
fixed by moving logic into a tested bash script file, after which false alarms
stopped.

## Cloud phase: scope and strategy pivot

After the 30B batch completed, a cloud execution plan (`cloud-plan.md`) was
drafted for RunPod-based scale-up, initially organized around cross-family model
replication (Qwen3, Mistral Small 3.2, Llama-3.3-70B) plus a Gemma 3 27B
hybrid-attention profiling stage. A significant course correction followed: the
cloud budget's primary purpose was redirected from model variety toward
statistical power and standard-benchmark validation on the already-characterized
Qwen3-30B model. The revised stage order became: full-scale LongMemEval (500
questions), real coding-agent trajectories (SWE-Gym/OpenHands traces), and a
second published benchmark (LoCoMo or SCBench), with the homemade synthetic
probe corpus and cross-model variety demoted to bonus/contingent stages. This
reordering was driven by the principle that industry-standard datasets carry
more credibility than internally constructed material, and that renting hardware
is best spent on evidence volume/quality at a well-understood model rather than
on justifying a different model.

A related clarification session addressed why Llama-3.3-70B remained in the plan
at all (as the only architecturally "clean" dense 70B with standard
full-attention + RoPE + GQA, needed as a scale anchor, since Qwen's own line
tops out at 32B dense or jumps to an unaffordable 235B MoE) and why Gemma was
excluded from the core replication (its sliding-window-heavy hybrid attention
breaks assumptions underlying the KV-cache surgery, since most layers never see
full-history context to preserve). Gemma 3 27B was retained only as a
contingent, profile-first exploratory stage — testing the hypothesis that
graftable signal in a hybrid model concentrates in its sparse global-attention
layers — requiring nontrivial new per-layer-type surgery code, and therefore not
addable trivially.

## Hardware, budget, and execution mechanics

The infrastructure decision was a single A100 80GB pod at a time (right-sized
for Qwen3-30B-A3B's ~61GB bf16 footprint; H100 rejected as poor value since the
workload is memory-bandwidth- not compute-bound), run serially with detached,
per-item-resumable batches, downloaded results, and pod termination between
stages to minimize idle billing. Budget was authorized at $50, then raised to
$100 total (with an explicit note that a further $100 is conditional on strong
results plus recommendation plus renewed agreement) after confirming RunPod's
$80 figure was a per-hour spend-limit setting, not a total cap. Execution was
later parallelized across up to four concurrent pods (approved at up to ~20%
cost premium for faster wall-clock), using a `SC_SHARD=k/N` deterministic
sharding scheme and atomic, resumable writes so that pod loss or preemption
costs little. A near-repeat of the morning's stale-process incident was caught
and killed within a minute during pod-1's relaunch; a related VRAM-residency
verification rule was added after pod-1 silently fell back to slow CPU-offloaded
inference due to a stray process holding GPU memory.

Both `.runpod_key` and `.huggingface_key` credentials were staged with explicit
holds (no use until direct approval), later approved. A Hugging Face gating
check found the entire core plan (Qwen3-30B, Mistral variants, both target
datasets) ungated; only Llama-3.3-70B and Gemma required manual license
approval, both later granted.

## Cloud execution results

Stage 0 (environment/model port to HuggingFace `transformers`, identity ladder)
passed on CUDA. Stage 1 (LongMemEval, capped at 350 of 500 questions to protect
remaining budget after per-question cost ran higher than estimated) ran sharded
across two pods. Stage 2 (SWE-Gym real coding-agent trajectory replay, 75
trajectories) completed: the tuned value-graft intervention recovered +0.0156
nats of a ~0.14-nat full-vs-compacted logprob gap (45/75 paired wins, 95% CI
[0.005, 0.027], roughly t≈2.8) — about 10% closure of measured damage,
translating to roughly 5–28x higher exact-action likelihood at typical action
lengths, a modest but statistically clean effect on the most demanding
(off-policy, single-cut, token-exact) instrument used in the project. This was
read as supporting rather than undermining the result's credibility, since a
stricter instrument still cleared significance. Pod-3 was terminated after stage
2 completed, at roughly $19 spent (of $79.59 remaining balance at that point).

Discussion clarified that offline trajectory replay has an off-policy limitation
(scoring prediction of another agent's recorded actions, not self-generated
continuations), partially mitigated by arm A's high absolute per-token accuracy
(~74%) showing the metric isn't dominated by policy-mismatch noise.
Pre-registered fallback experiments for a null/ambiguous stage 2 were documented
(dependence-stratified re-analysis, span-level rescoring, targeted re-cuts
around key trajectory events, and small-scale rollout-based behavioral counts)
before results were seen. A follow-on possibility — wiring the cache-state
intervention into a standard agent harness (OpenHands' condenser extension
point, or SWE-agent's history-processor hooks) via a thin OpenAI-compatible
serving shim around the existing HF cache machinery, enabling true end-to-end
task-success evaluation — was scoped (~1 day engineering, ~$50–100 GPU) and set
to "default-on unless the mechanism proves inert," per a discussion emphasizing
that this end-to-end test may be warranted regardless of whether the offline
proxy shows a clean signal, since the offline design's artificiality is itself a
risk.

Stage 1b (full ~115K-token haystack under LongMemEval's standard protocol and
judging prompts, ~100 questions) was approved and queued to give a
benchmark-comparable, citable baseline — clarifying that the originally reported
local/stage-1 LongMemEval numbers use a deliberately reduced/controlled
construction (subsampled ≤16K contexts, evidence placed for eviction) and are
not comparable to published leaderboard scores; that non-comparability caveat
was written explicitly into the results documentation.

## Tuning campaign

A dedicated tuning track (stage T, later run on a fourth pod) was added in
response to a request for finer-grained, model-specific calibration rather than
reusing the 4-bit-tuned α=0.75 dose on the bf16 cloud stack. The methodology: a
global-α resweep on a validation split only, a per-layer profile (measurement,
not tuning), and a per-KV-head profile (48 layers × 4 heads = 192 slots at 30B;
36×8=288 at a locally-run 4B check), from which a small number of
low-degree-of-freedom candidate rules (e.g., top-K head sets, positive-layer
sets, a single global α) were derived and each evaluated exactly once on a
held-out split, with adoption governed by a pre-agreed rule (adopt if it beats
the incumbent with CI excluding zero or ≥8/10 wins; adopt the bf16-native tuning
even on a tie, since it's the correct default for the deployed stack; otherwise
keep the incumbent and record the null).

Results: at 30B/bf16, the α sweep suggested values ≥1.0 (into "extrapolation,"
not blending) performed best — a finding initially read as surprising, explained
via the same principle as classifier-free guidance/contrastive decoding
(subtracting a fraction of a "biased-by-impoverished-context" fresh-value read
sharpens the signal that distinguishes full-history from summary-only
interpretation); an extended α bracket (up to 3.0) was queued to locate the
point where extrapolation collapses into degenerate output. The 30B per-layer
profile was diffuse (no strong band, unlike the crisp 4B mid-band hump at
L12–L22). At 4B, a per-head profiling pass initially suggested a single standout
KV-head-band effect, but this did not survive holdout evaluation (5/10 wins,
half the apparent effect) — a clear selection-artifact case that validated the
holdout discipline and left the existing mid-band-layer rule as 4B's champion
(10/10 holdout wins). The 30B per-head profile's own holdout verdict was pending
as of the end of this transcript. Terminology was formalized: interventions with
α≤1 are "blending," α>1 is "extrapolation" or "contrastive steering," together
termed "value steering" in the write-up, drawing an explicit parallel to
guidance-scale methods in diffusion models for reader legibility.

A cheap local scale-curve sweep (Qwen3 1.7B and 8B, from the older non-2507
line, explicitly caveat-labeled as a different sub-family from the 4B/30B-2507
anchors) was run as low-priority appendix material to visualize dose-response
across sizes; 0.6B was skipped as too weak to produce meaningful signal. This
local sweep was placed under a dedicated auto-kill memory-pressure guard (kills
only the sweep process at <12% free memory or >28GB swap) specifically to
prevent a repeat of the earlier swap-thrashing incident while cloud pods
continued billing independently; the user explicitly flagged this failure mode's
severity (local coordination hang while remote spend continues unchecked) as a
standing concern, and confirmed satisfaction with the
event-driven/quota-conserving monitor design pattern more broadly.

A Mistral-family chat-template adapter was under construction to enable
pre-tuning a Mistral model on the pod once Qwen phases finished, motivated by
the idea of arriving pre-calibrated for stage 5's model-variety runs rather than
treating tuning as a first-time cost per new model. Template probing found
`Mistral-Small-3.2`/`3.1` lack usable `chat_template` fields, while
`Ministral-8B-Instruct-2410`, `Mistral-Small-Instruct-2409`, and
`Mistral-Nemo-Instruct-2407` do; the target model choice among these was left
open at the end of the transcript.

## Operational hygiene

Local disk space became a concern after model downloads; a full audit found
Ollama and LM Studio model caches (totaling ~59 GiB, largest single item a 17GB
Gemma GGUF) unrelated to the experiment. All were deleted with user approval,
restoring the machine to ~100 GiB free; both were noted as trivially
re-downloadable if needed later. A brief false alarm from process-reaping (a
long-finished launcher shell being cleaned up by the harness) was correctly
identified as harmless since actual pod work runs detached remotely.
STATE.md/DECISIONS.md were updated repeatedly through the day per standing
instruction to keep them current for a hypothetical fresh agent, and a note was
recorded (not acted on) that the repository is currently disorganized and will
need a cleanup pass before any public sharing, explicitly deferred to
end-of-project so as not to disrupt in-flight pipelines.

As of the end of this transcript: stage 1 (LongMemEval, sharded across two pods)
and pod-4's tuning phases (per-head profile eval plus extended-α bracket) were
still in progress; stage 1b was queued to launch on freed pods; a Mistral
pre-tuning chain and stage 3 (second published benchmark) adapter remained to be
built/launched; running balance was tracking under the $100 budget (~$19 spent
through stage 2 alone, out of a $79.59 remaining balance at that checkpoint).
