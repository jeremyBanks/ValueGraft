_This conversation corrected the project’s central evidence story: the
native-render referent effect is render-fragile, self-generated summaries remain
a supported mechanism, and the paper must add production-faithful summaries
before making a positive claim. It also established durable reproducibility,
environment, and operational follow-up priorities while the brief-condition
held-out run and cross-model scans continue._

**Participants:** User and claude-opus-4-8.

**Evidence and framing.** The banked native baseline c01–c12 produced referent
raw_EB +0.012, CI [−0.070,+0.091], while fresh c13–c24 was −0.021, confirming
that the earlier +0.10 was not reproducibly established. Fable’s audit found the
apparatus mechanically sound and identified render variance—not a harness
regression—as the best explanation; another native draw was +0.116. The judged
sense result (+12pp, CI [+2.2,+22.9]) exists only on development conversations
and under the BRIEF summary condition, so it remains a hypothesis pending
held-out judging. BRIEF summaries deliberately handicap the text baseline; no
positive result is yet established under a production-faithful summary. The
intended paper is therefore an honest, report-style scientific paper: robust
compaction damage and honesty findings lead; sense recovery may upgrade the
paper after validation; referent-logprob is reported as fragile/underpowered
with metric and render-variance caveats.

The native premise must be split: dependence on the model’s own self-generated
summary is supported (+0.136 versus fixed-summary +0.004 in the cited isolation
test), whereas native rendering of replies as a validity requirement is not
established. The authored-to-native shift was confounded by MoE render variance,
and the decisive authored-cross-model control was never run. Fable proposed
cheap EXP-2 wrong-conversation/placebo re-grafts at the referent readout, EXP-3
rescoring native replies with a fixed summary, then—only if useful—EXP-1 with
roughly 5–6 additional banked native draws. Do not present native rendering as
proven necessary.

**Durable results and corrections.** Every available Qwen3-30B native render was
harvested and committed; 33 checkpoints are local and reusable for rescoring.
The claims ledger is committed at `5cb1a73` and records 16 verified, 5
reconciliation items, 1 unsupported, and 1 pending claim. The unsupported F2
assertion that H-pack restores evicted-fact accuracy (38/48) must not ship: disk
data supports fabrication falling from 67% to 12% and admission rising to 96%,
not recall restoration. The judged metric provenance is 4-bit local MLX, not
bf16, and uses BRIEF summaries with graft doses a0.25+a1.0; this was verified by
full batch comparison after earlier single-key inferences proved unreliable.

**Current experiments.** The user overrode the prior champion gate and
authorized scans. Qwen3-30B’s champion scan completed and produced a validated
10-configuration map using banked renders. Qwen3-32B and Mistral-Small-24B are
rendering c01–c12 with all renders to be banked before champion scoring;
Qwen2.5-32B failed to launch due to a RunPod HTTP 500, and phi-4 was terminated
after becoming unreachable. The local BRIEF generation/decider for c01–c24 is
running with per-conversation checkpoints and was at 11/24 in the latest status;
it must first establish whether the +12pp control reproduces, then test held-out
conversations. The next planned local arm is a production-faithful/std summary
judged run, queued after the brief run because the Mac GPU is serial. The user
wants both BRIEF synthetic stress-test results and realistic summaries retained.

SWE-Gym means teacher-forced next-action prediction over recorded trajectories,
with offline greedy generations for analysis only; it is not live OpenHands
execution, code execution, or agent rollout. A production-summary anchor run was
attempted on the warm 30B pod but stalled amid environment/import and SSH issues
and was terminated; cross-model gym capture remains desirable only when marginal
model-switch cost is low and the environment is provisioned reliably.

**Environment handoff.** A retrospective found the recurring operational root
cause: local MLX/uv/Python 3.12 versus pod HF/bf16/system Python 3.11, with
dependency installation copied across seven job scripts and no single pod
environment specification. It identified contradictory Transformers pins
(including `transformers==5.0.*` where another job documents that version as
incompatible), inconsistent pandas/pyarrow installation, repeated torchvision
and torch workarounds, and the lack of fail-closed dependency checks. The
recommended proportionate fix is one shared `scripts/pod_env.sh`, sourced by
every pod job, plus dependency validation in `preflight.py`; this is awaiting
authorization and should precede any further gym-on-warm-pod attempt. The latest
live state is two healthy rendering pods (32B and Mistral), no active 30B gym
pod, local MLX decider under memory pressure but progressing, and no immediate
need to kill the decider because checkpointing makes recovery possible.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `a72408e00d959a72d`
- `a9816998adec69806`
- `a07c3c3c0b423d4cf`
- `a08581a3ea0f385c3`
- `aa59207cd56f771c9`
- `a66941a265b0bcede`
- `aab64ce94fe815886`
- `ad3a0aaa7b0d3c741`
