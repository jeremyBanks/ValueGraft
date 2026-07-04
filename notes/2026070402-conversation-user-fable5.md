_This note summarizes the semantic-continuity/compaction experiment as it
progressed from initial setup through a mid-project refocus toward a
mitigation-first framing, culminating in a completed local evidence package (4B
and 30B scales) plus follow-on validation against a standard benchmark and a
cloud-scaling plan. Model attribution: work was performed by Claude Fable 5
(claude-code, v2.1.200) throughout both days covered by this transcript._

**Participants:** User and claude-fable-5.

## Setup and infrastructure

The project began from a pre-written experiment brief (Arms A–E, later extended
by three companion documents: follow-up explorations Arms G–H with proposed
"ValueGraft"/"SelfGist" naming, a Phase 2 scale-up/coding-agent extension, and
an external-review amendments document). The user directed the agent to drive
the experiment autonomously on local hardware (Apple Silicon, MLX runtime),
using Qwen3 models at two scales — a 4B development model and a 30B-A3B
production model — with sub-agents kept to a small number concurrently to
control token spend.

Early environment work resolved an mlx-lm/transformers version incompatibility,
then built and validated a full identity-test ladder (L0–L4) proving the
KV-cache surgery machinery (gapped-cache retention, value transplantation) is
bit-identical or near-identical under null operations before any experimental
claims were trusted. Several latent bugs were caught this way: a
batched-vs-single-token kernel mismatch, an unstable `<think>` block in the Qwen
chat template, and a token-alignment routine that silently dropped tail
alignment when summary and tail swapped order between contexts.

## Operational lessons established early (durable, still applicable)

- Serialize GPU-bound background jobs rather than running them in parallel —
  parallel execution gave no throughput benefit and added memory risk.
- Tee long-running command output to a scratch file before filtering with
  `tail`/`grep`, so the full stream remains reviewable without bloating context
  (saved to persistent memory as "tee-before-tail").
- Bracket long commands with wall-clock timestamps so stalls are diagnosable.
- Use unbuffered output (`python -u`) for detached background jobs;
  block-buffered stdout otherwise hides genuine progress until process exit.
- Division of labor: the local subject model (4B/30B) generates all
  in-conversation dialogue text (since the continuation-prediction metric
  requires text the subject model would itself produce), while all meta-work —
  scenario/plant authoring, judging, spot-checks, analysis — is delegated to
  Claude subagents. Judging was upgraded from a local-model judge to
  Sonnet-class subagents (Haiku was explicitly ruled out per user instruction,
  since Sonnet was affordable), batched (~150 items, 2 concurrent) rather than
  per-row.
- Maintain three standing handoff documents at each phase transition: `STATE.md`
  (volatile snapshot of exact pipeline state), `DECISIONS.md`
  (methodology/runtime deviations and their rationale), and `AGENTS.md` (stable
  rules/traps/reading order) — added specifically so a new session or agent
  could resume cold. Commit frequently at each milestone rather than batching
  into few large commits, while avoiding committing large binaries (model
  weights stay outside the repo; results/data are small JSON).

## Initial experimental design and early signal

The design compacts long conversations five (later twelve) ways — A (oracle full
context), B (production-style text summary + recompute), C (retained KV with a
positional gap), D (no-summary ablation), E (value-only transplant, swept over
α) — and measures (1) continuation log-probability gap-closure relative to
oracle A, and (2) accuracy on planted probe questions (referent,
sense-disambiguation, stance, ruled-out, evicted-fact categories), using
synthetic (12) and natural (8) conversations with plants authored by a Claude
subagent.

An early isolated micro-experiment showed transplanted value vectors alone carry
word-sense disambiguation information into an impoverished context (+0.8 nats
toward the planted sense at α=1 vs +3.8 for full oracle context, monotone in α),
establishing the core hypothesis had at least some real basis before the full
pipeline ran.

A corpus quality problem was found and fixed mid-project: probe "tail" replies
were leaking plant answers, contaminating the evicted-fact and referent/sense
categories. A repair pass (resampling/truncating leaking replies) brought
clean-plant counts from 77/120 to 115/120, with the remaining 5 flagged leaks
confined to the (also-evicted) early section and therefore harmless.

## Companion-document arms committed but deferred

Two follow-up-exploration arms were added to the running variant set at low
marginal cost: H-gap (sinks + in-context-encoded summary only, no tail) and its
matched control B-min (same summary text, freshly encoded, no history) —
motivated as the single sharpest test of the hypothesis that write-time encoding
itself carries meaning. Arm G (retrieval-based value transplant / "SoftGraft,"
using the model's own attention as a correspondence mechanism for paraphrased
content) was designed but deferred as too costly relative to its expected
marginal value at this project stage.

The external-review amendments document contributed several before-results-grade
additions actually implemented: a new causal-order control arm B-causal (fresh
prefill of summary+tail in the same causal order as C, to separate ordering
effects from encoding effects); negative controls for value grafting
(wrong-conversation graft, shuffled-value graft) to rule out generic
distributional smoothing; a six-class leakage classification applied to every
planted probe; a metric-hierarchy flip making probe accuracy (by leakage class)
the headline and continuation NLL secondary; and a literature-positioning note
flagging "Models Take Notes at Prefill" (arXiv 2606.17107) as the closest
prior-art neighbor, which supports the project's premise (write-time KV carries
context-conditioned conclusions, position-portable via RoPE re-rotation) while
narrowing the novelty claim toward compaction-specific semantic continuity
rather than KV editability/portability in general.

## Mid-project refocus (major methodology pivot)

After the 4B evidence package was complete and the 30B run was partway through,
the user introduced a significant reframing (informed by outside review,
including from another model, later specified as OpenAI GPT-5.5): the original
design's central question — whether write-time KV state differs from or carries
more information than re-encoded text — was judged close to self-evident and not
the valuable question. The valuable question was redefined as: can a small,
practical cache-state intervention reduce the damage caused by conversation
compaction, in a way that is plausibly deployable, not confounded by leakage or
generic smoothing, and testable on a clean-failure/clean-improvement contrast
rather than a large arm matrix?

The agent executed this pivot in two phases per user instruction: Phase 1
brought the in-flight 30B run (already trimmed to the mitigation-relevant arm
subset) to a clean, documented stopping point with narrowed scoring, rather than
expanding the old matrix further. Phase 2 was a newly designed, narrower
experiment centered on one clean contrast: H-pack vs B-min-pack (identical
packed summary token sequence and positions, differing only in write-time vs
fresh-time encoding), scored primarily on a purpose-built v2 probe set targeting
fabrication-vs-admission on evicted facts (with collision-audited decoy items)
plus secondary referent/sense probes, retaining
wrong-conversation/shuffled-value negative controls. This required a new key
re-rotation utility (to pack summary KV at contiguous positions), validated by
new identity tests before use; debugging surfaced a rotation-convention mismatch
against MLX's RoPE implementation, resolved empirically, and also surfaced that
4-bit matmul kernels are sequence-length-dependent (a documented noise floor,
mitigated by same-shape comparisons).

## Results (both mitigation-relevant contrasts, both model scales)

At 4B scale, on continuation NLL: light-dose value grafting (α≈0.25) reliably
helped (+0.075–0.083 nats closure across synthetic and natural corpora, CIs
excluding zero), while heavy transplantation degraded performance monotonically;
H-gap beat matched control B-min consistently (+0.093 nats, 10/12
conversations); gapped retention (C) was mildly negative to neutral once the
ordering confound was removed via B-causal.

At 30B scale, on the synthetic corpus (n=12, full contrast set): H-gap beat
B-min in all 12 conversations (+0.128 nats, CI [0.097, 0.158]); the previously
harmful full-strength value graft (α=1.0) reversed to net-positive (+0.052,
10/12 conversations, ~29% of the A−B gap), indicating greater tolerance for K/V
decoupling at larger scale; C remained approximately neutral. On the natural
corpus at 30B, none of the synthetic contrasts replicated (the A−B gap itself
was small there, ~0.07 nats), establishing that these interventions help in
proportion to how much the actual continuation depends on evicted content — a
conditional rather than unconditional finding, later folded into the write-up's
limitations rather than treated as grounds to redefine the underlying claim, per
explicit user direction to stop re-litigating the claim framing and focus on
improving compaction behavior.

The Phase 2 fabrication-focused contrast produced the strongest and most
deployment-relevant result of the project: packed write-time summary state
(SelfGist / H-pack family) sharply reduced confident fabrication of
never-discussed decoy details relative to standard text compaction, most
dramatically at 30B (roughly 19:5 fabricated:admitted for B versus near-total
admission for H-family arms), decomposed into a layout component (bare packing
alone helps) and an encoding component (write-time vs fresh encoding) that grows
with scale.

A follow-up α-tuning sweep, run per user request with a validation/holdout split
to avoid overfitting to one corpus, used a finer low-range grid at 4B
(0.00–0.35) informed by the earlier finding that high α hurts small models; the
4B optimum was a mid-layer-band gate at α=0.25 (holdout +0.017 nats, 10/10
conversations, ~10% gap closure). At 30B the dose-response shape differed
qualitatively — late-layer gating and even extrapolation past unit α (α≈1.25)
outperformed the 4B-tuned range, with a holdout-validated optimum around α=0.75
(~24% gap closure), reinforcing that optimal intervention strength and layer
placement are scale-dependent and must be tuned per scale, not assumed to
transfer.

## Deployment/overhead analysis added at user request

At the user's request, the write-up was extended with a quantitative treatment
of practical deployment cost, framed around an "opaque compaction handle"
concept (a client-facing artifact than can restore the intervention without
exposing raw KV): per-token KV footprint (~96 KB/token at 30B-A3B, ~144 KB/token
at 4B in fp16); a SelfGist retained blob of roughly 8 MB for a terse summary at
30B (larger for thorough summaries, smaller under KV quantization, which was not
tested in the main runs); comparison against the ~1.2 GB full-context KV
footprint (SelfGist ships well under 1%); and the structural point that
ValueGraft requires the entire old cache resident at compaction time, making it
a server-side/local technique only, whereas SelfGist's small footprint is
plausible as a client-portable artifact. Constraints noted: validity is tied to
exact model/revision/tokenizer, and accepting client-supplied KV state
introduces a new, largely unauditable trust surface, suggesting a
signed/expiring cache-handle model rather than raw blob transfer for any
productionized version.

## Standard-benchmark validation and cloud-scaling groundwork (in progress at transcript end)

Following completion of the local program (mechanism findings, the mitigation
contrast, and the tuned dose-response at both scales), the user asked the agent
to continue autonomously toward two further goals while away from the session:
(1) validate findings against an established, realistic benchmark rather than
only synthetic/natural home-built corpora, and (2) produce a concrete,
low-friction plan for renting cloud GPU hardware, written for a user unfamiliar
with these platforms, with explicit spend guardrails.

For (1), the agent selected LongMemEval as the closest-fitting standard dataset
(multi-session chat histories with questions answered from earlier, evictable
sessions), constructing ≤16K-token instances with answer-bearing sessions placed
in the evicted region, and ran the same proven arm subset (A, B,
SelfGist/H-pack, B-min-pack, tuned ValueGraft) against a sample sized to local
compute (~48 questions at 4B, completed and Sonnet-judged; a comparable batch
launched at 30B, encountering a per-question slowdown traced to memory pressure
and mitigated via relaunch with larger event-batching intervals, with an
explicit fallback to cap the batch around n=24–30 if timing did not stabilize).
Separately, prior scouting (lower confidence, flagged for re-verification before
execution) had identified SWE-Gym/OpenHands-SFT-Trajectories as a viable,
directly replayable source of real coding-agent traces for the Phase 2
coding-agent extension, should that avenue be pursued later.

For (2), the agent was in the process of drafting the cloud-rental plan (target
audience: someone who has used AWS but is unfamiliar with GPU-rental workflows
or SSH-based remote execution) with hard budget guardrails, to be ready for user
review upon return; this document was not yet complete as of the final logged
message.

## Deliverables and writing form

The intended final public artifact was reframed, per outside input the user
relayed, from an academic paper to a findability-oriented technical blog post
(plus the GitHub repository as supporting material), targeting a reader who has
had the same idea and is searching for prior art. Requirements adopted:
plain-language problem framing alongside proper terminology in the
title/opening/README; explicit, specific provenance in the body (design and
execution primarily by Claude Fable 5, adversarial review and reframing input
from GPT-5.5, direction and sanity-checking by the user, stated plainly rather
than falsely modest); a related-work section naming specific search terms and
papers considered and ruled out; results reported regardless of direction
(including the clean null on recall of evicted specifics); and a "mechanism →
ablation/headline numbers (framed as % of gap recovered) → limitations → open
question to readers" structure, in a matter-of-fact, non-apologetic tone. A
literature pass (properly sourced, ~30 fetches with verified IDs) confirmed no
existing work combines all the elements of this design; nearest neighbors
(SamKV's blending formula for RAG chunks, "Models Take Notes at Prefill" for
position-portability physics without a compaction/summary setting, Compressed
Context Memory for the problem setting without a text-summary comparison
baseline) were each documented as missing at least one leg, and the proposed
technique names (ValueGraft, SelfGist) were confirmed collision-free. A first
complete blog draft was committed, with the related-work section filled in from
the literature pass, pending the user's own end-to-end review before any public
posting.
