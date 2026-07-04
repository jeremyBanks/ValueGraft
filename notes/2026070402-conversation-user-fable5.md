_This chunk covers the launch and early execution of the semantic-continuity
compaction experiment (Arms A–E core pipeline) on local MLX models, including
environment setup, harness validation, corpus generation and repair, several
operational process corrections, and the first interim (n=4) results from the 4B
dev-model batch, which was 10/19 conversations complete as this chunk ends._

**Participants:** User and claude-fable-5.

**Setup and validation.** The agent read
`semantic-continuity-experiment-brief.md`, resolved an `mlx-lm`/`transformers`
version incompatibility by pinning transformers to 4.x, and built out the core
surgery library plus the full identity-test ladder (L0–L4): cache
serialize/rebuild is bit-identical, the gapped-cache machinery (Arm C) is
bit-identical under null surgery and loses only 0.04 nats evicting an irrelevant
span, and the value-transplant machinery (Arm E) passes both null-transplant
identities exactly. Three bugs were caught by the ladder before they could
corrupt results: a batched-vs-single-token kernel mismatch, an unstable
`<think>` block in the Qwen chat template, and a difflib alignment bug that
dropped the tail alignment when summary/tail order swapped between contexts. All
planned arms were implemented: A (oracle), B (text compaction), C (gapped KV
retention), D (no-summary ablation), E-post (α sweep), E-inter (layer-by-layer
swap), and later H-gap/B-min (added early, before arm runs started, since they
ride the existing `arm_c_snapshot` machinery at negligible marginal cost).

**Two model sizes.** Qwen3-4B is the fast development model used to shake out
the full pipeline before the more expensive run; Qwen3-30B-A3B is the
primary/reportable model. 4B results are kept as a secondary replication at a
second scale rather than discarded.

**Corpus.** 12 synthetic scenarios (120 hand-planted probes across referent,
sense-disambiguation, stance, evicted-fact categories) plus 8 natural
conversations were composed by a Claude subagent for scenario/probe text, with
the local subject model generating only the in-conversation assistant replies
(needed because the held-out continuation used for the logprob metric must be
text the subject model would naturally produce). An initial contamination audit
found widespread tail leakage of planted facts (77/120 clean); a repair pass
(resampling/truncating leaking replies conversation-by-conversation, c01–c12)
brought this to 115/120 clean, with all five probe categories fully clean of
tail contamination in the final corpus (the remaining 5 flagged items are
harmless early-section leaks, since the early section is evicted anyway). Corpus
was then treated as final and the full 19-conversation, 12-arm-variant run (L5)
began on the 4B model.

**Process corrections adopted during this chunk, all logged as durable
practice:** (1) bracket long-running commands with wall-clock timestamps so
interrupted work can be compared against start time; (2) when two jobs are
GPU-bound and would run at roughly half speed in parallel, serialize them
instead — lower memory risk and earlier partial results; (3) always `tee`
command output to a scratch file before piping through `tail`/`grep`, so the
full stream stays reviewable without bloating context (saved to persistent
memory); (4) use `python -u` / unbuffered output for detached long-running jobs,
since block-buffered stdout was making a healthy process look stalled; (5)
commit frequently at milestones rather than batching into few large commits,
avoiding large binaries (model weights stay in the HF cache, outside the repo).

**Judging and generation division of labor.** Local small models are the
experimental subjects and generate only conversation dialogue text (this is
load-bearing: the continuation-prediction metric needs text near-modal for Arm
A, so switching the generator would inflate the noise floor and dilute the
A-vs-B gap the normalization depends on). All meta-work — scenario/probe
authoring, judging probe answers, spot-checking, analysis — was moved to Claude
subagents (Haiku/Sonnet-class) rather than the local judge model, batched as a
few calls over the undecided set rather than per-row, with the local judge kept
only as fallback.

**Companion documents committed to the repo (content not yet acted on, queued
behind the core A–E analysis):** the follow-up explorations document (Arm H
minimal in-context summary retention with its B-min control, Arm G
retrieval-based value transplant via cross-attention as the generalization of
Arm E, and naming suggestions ValueGraft/SelfGist/SoftGraft); the Phase 2
scale-up and coding-agent extension document (triggered only if local results
are meaningful; budget ~$100–200 cloud credits, choice between Strategy S — one
larger model for a scale point — and Strategy P — more statistical power at
local scale — decided by a short decision memo; coding-agent realism leg using
real SWE-bench-style trajectories with behavioral, machine-checkable probes);
and the amendments-from-external-review document, whose most important addition
is a new **B-causal** control arm (fresh prefill of system+tail+summary,
causal-order-matched to Arm C) needed because Arm C's
summary-generated-after-full-history vs Arm B's summary-before-tail ordering was
an unaddressed confound — C vs B-causal is now the clean mechanism claim, B vs A
remains the realism baseline, and B vs C is reported only as a
deployment-flavored contrast. Other amendments queued for
before-results/analysis-time application: negative controls for value grafts
(wrong-conversation graft, shuffled-value graft), a GQA aggregation rule for Arm
G (default: entropy-weighted average with per-slot entropy gating), a six-class
leakage classifier per probe
(explicit-in-summary/paraphrased-in-summary/absent-from-summary/contradicted-by-summary/tail-visible/evicted-only),
contradiction probes, a metric-hierarchy flip (probe accuracy is now the
headline metric, continuation NLL secondary, per amendment 6 — already
implemented in the scoring code), and positioning against nearest prior art
(arXiv 2606.17107, "Models Take Notes at Prefill," plus a list of other
neighbors) for the eventual writeup.

**Scaling discussion (deferred, contingent on local results).** If local results
are meaningful, the agent's recommended priority for a ~$200 cloud budget is
cross-family replication (e.g. Llama-3.3-70B or Llama-3.1-8B + one other family)
over pushing to a single larger model, since statistical power comes from
conversation count (n) and probe count rather than parameter count; bigger
models mainly buy external validity and removal of the local 4-bit-quantization
confound. The literature in this niche modally uses 7–14B models with a 70B
robustness subset in stronger papers, so the current 4B/30B setup is already
within community norms. This full plan is only to be written up formally if the
local run shows consistent positive signal; a tight null is still reportable.

**Interim results (n=4, keyword-scoring only, no judge pass yet, repaired
corpus).** Single-conversation (c01) numbers: A −0.77, B −1.11 NLL; D
(tail-only, no summary) closed 63% of the gap; E-inter α=1 closed 33%; C sat at
B's level; E-post was at-or-below B and degraded with increasing α; H-gap
(−1.47) beat its matched control B-min (−1.55) on the single sharpest same-text
contrast. At n=4 aggregated: on probes (now the headline metric), retained-state
arms lead where expected — referent probes C 6/8 and H-gap 6/8 vs B 4/8 (A 7/8,
D 0/8 confirming the no-summary control behaves correctly); sense probes H-gap
3/8 vs B 2/8; Arm E (value-only transplant) was not yet beating B on probes.
Continuation NLL was secondary and noisy: C and E-post α=1 sat below B, only
E-post α=0.25 was marginally positive, with wide per-conversation swings. Known
artifacts flagged for the judge/classifier passes rather than the keyword
scorer: stance probes are currently uninterpretable pre-judge (anti-keyword
scoring penalizes Arm A for quoting disliked terms while still complying), and
evicted-fact probes show clear summary leakage (B recovering 4/8
nominally-evicted facts), to be quarantined by the six-class leakage classifier.

**Handoff state at end of chunk.** The 4B full batch (19 conversations × 12 arm
variants) is in progress and healthy, at 10/19 conversations complete, running
at a steady pace of roughly 9–9.5 minutes per synthetic conversation (naturals
expected faster, ~4–5 min, being CONT-only). No fixed ETA was restated at this
point beyond the steady per-conversation pace; supplement arm runs (B-causal,
negative controls) and the classifier/judge passes are queued to run after the
batch completes, followed by the 30B production run.
