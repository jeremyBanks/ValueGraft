_This chunk covers continued autonomous execution of the cross-architecture
KV-cache compaction study: disk-space cleanup to keep the local coordination
machine safe, monitor infrastructure hardening, the α>1 "value steering"
discovery framed via classifier-free-guidance analogy, completion and
quantification of the SWE-Gym stage-2 coding-trace results, and resolution of
the per-head tuning question (validated null at 4B, pending at 30B)._

**Participants:** User and claude-fable-5.

**Disk and infrastructure management.** With the multi-pod cloud campaign
running, local disk pressure was addressed by deleting all Ollama models
(immediate) and, on follow-up confirmation, all LM Studio models
(gemma-4-26B-A4B at 17G plus five smaller roleplay/chat models), freeing the
local machine to 100 GiB free from a starting point of concern. A memory/swap
watchdog was kept armed throughout to auto-kill the lowest-priority local sweep
processes if free memory or swap approached danger thresholds, addressing a
standing concern that local coordination-machine instability could leave paid
cloud pods running unattended. The consolidated pod-health monitor was found to
be using bash-only syntax under a `sh`-invoking harness (causing garbled
output); it was rewritten as a standalone, tested `podwatch.sh` script with
staggered SSH retries and silent-when-healthy behavior, intended to end a run of
false-alarm wakeups.

**Value-steering (α>1) discovery.** Pod-4's bf16 tuning sweep showed α≥1.0
outperforming the incumbent α=0.75 at 30B, initially flagged as
counterintuitive. The mechanism was reframed as extrapolation/contrastive
steering rather than blending: at α=1.25, V = 1.25·V_old − 0.25·V_fresh
subtracts a biased "read against the impoverished summary" signal to sharpen the
true-history direction, analogous to classifier-free guidance in diffusion
models. The write-up terminology was fixed as "value steering," with blending
(α≤1) and extrapolation (α>1) as named regimes. An extended-α bracket (1.0,
1.25, 1.5, 2.0, 3.0) was queued on pod-4 to find where extrapolation peaks and
where it degenerates toward shuffled-graft-style gibberish, explicitly logged as
low priority but worth doing given idle capacity.

**Tuning sweep design clarified as axis-aligned slices, not a full grid.** The
design was confirmed as three separate 1D-style slices — α alone, layer alone
(fixed α=1), head-slot alone (fixed α=1) — composed afterward into a handful of
low-degree-of-freedom candidate rules, each tested once against the incumbent on
the 10-conversation holdout split. This was adopted explicitly to avoid
overfitting a ~1,000-cell interaction space with insufficient data. Status
snapshot at this point: 4B fully sliced (layer axis shows a crisp mid-band hump
L12–L22; head axis profiled via a same-evening local playground); 30B (4-bit
local) fully sliced, global α=0.75 wins; 30B (bf16, pod-4) α-slice and diffuse
layer-slice done, head-slice completing; 1.7B/8B curiosity sweeps show no
benefit at any dose.

**4B per-head holdout result: validated null.** The local 4B head profile (36
layers × 8 KV heads = 288 slots) suggested an apparent single-head ("head-2
band") concentration of signal. On formal holdout evaluation this did not
survive: mid-band α=0.25 incumbent kept a perfect 10/10 win record (+0.0173
mean) versus top-16-slots (+0.0197 mean but only 6/10 wins, indicating a few
large wins carrying an overfit selection) and head-2-band (+0.0075, 5/10 — a
coin flip). Conclusion recorded: at 4B, real structure exists only on the layer
axis; apparent head-axis structure is a selection artifact that the holdout
discipline correctly caught. This result is held up as validating the
pre-registered holdout ritual and is now the skeptical frame applied to the
pending 30B per-head result.

**SWE-Gym stage-2 completed and quantified.** Final result on 75 real OpenHands
coding-agent trajectories: the tuned graft recovers +0.0156 nats (45/75 wins,
95% CI [0.005, 0.027]), roughly 10% of the measured compaction damage on real
agent traces, versus ~24% closure on the compaction-dependent synthetic corpus
and ~0% on free-form chat — a consistent story of recovery scaling with how much
the continuation actually depended on evicted content. In practical terms:
per-token confidence in the true next action is 73.7% (full context) vs 62.4%
(compacted) vs 63.4% (compacted+graft); over a ~100-token tool call the grafted
context is ~5× more likely to reproduce the exact true action than plain
compacted context (with the caveat that exact-sequence likelihood overstates
practical action-correctness). Effect size is modest (d≈0.33) but statistically
clean (t≈2.8); the user endorsed treating "statistically significant, modest
effect" as an acceptable outcome given the instrument (off-policy replay of
another agent's trajectory, single arbitrary cut, exact-token scoring) is an
unusually unforgiving lower-bound test — reinforcing that this result places the
project in the "default-on" scenario for the end-to-end OpenHands+condenser
follow-up once the current budget-authorized work concludes. Pod-3 was
terminated after stage 2 completed (with a brief false pod-3-unreachable alert
from the watchdog not yet knowing about the termination, resolved by removing it
from the check list). Balance after stage-2 close: $79.59, tracking under the
$100 authorized budget.

**Repository cleanliness noted for later, not acted on.** The user flagged that
the repository is currently disorganized and that this must be cleaned up before
any broad sharing, but explicitly asked that nothing be touched now to avoid
disrupting running pipelines; this was logged in STATE.md as an explicit
end-of-project task.

**Rough non-committal timeline given** (relative, not absolute): stage-1 shards
and stage-2 to complete "tonight," stage-1 judging and stage-1b launch following
same evening, stage-3 (LoCoMo/SCBench) adapter to be built and run overnight,
stage-1b and stage-3 judged results expected by "tomorrow morning," consolidated
results document and write-up revisions at "tomorrow midday," and bonus stages
(Mistral variety, possibly Llama-70B, Gemma hybrid profile) in the "tomorrow
afternoon–evening" window with final synthesis by "tomorrow night." Largest
named uncertainties: stage-1b's per-question time at 115K-token context could
push completion later by a couple of hours, 70B pod availability is unconfirmed,
and any additional incident tax is unbudgeted in the estimate.

**Handoff state at end of chunk.** Four cloud pods in play: pod-1 and pod-2
running stage-1 shards nearing completion; pod-3 terminated post-stage-2; pod-4
(secure) has completed its full Qwen tuning-phase chain (sweep, layer profile,
head profile) and is running the extended-α bracket plus 30B per-head holdout
evaluation, with the Mistral template dispatch being integrated into the tuning
runner so Mistral pre-tuning can launch on pod-4 immediately after. Local GPU
had completed the Qwen scale-curve sweeps (1.7B/8B, null results) and the 4B
per-head playground (also null, validated). Immediate next milestones: stage-1
shard completion triggering the Sonnet judging fan-out, stage-1b launch, and the
pending 30B per-head holdout verdict from pod-4.
