_This chunk covers the completion of the 4B-scale evidence package (naturals,
negative-control supplements, brief/shadow summary condition, B-causal
ordering-confound repair, and the full Sonnet-judged probe pass), the launch and
partial progress of the 30B run, a major project reframing toward a
mitigation-first blog-post artifact, an external code review and fixes, and the
user's Phase 1/Phase 2 restructuring directive, followed by early 30B results
and several queued brainstorming items._

**Participants:** User and claude-fable-5.

**4B package completion.** All 12 synthetic conversations' negative-control
supplements and the 8 natural (CONT-only) conversations finished, replicating
the synthetic pattern closely: E-post α=0.25 gives a reliable small positive
continuation-NLL effect (+0.075 natural vs +0.083 synthetic closure), heavier
value transplantation (α=1) hurts monotonically, and gapped retention (C) is
mildly negative on NLL. A brief/shadow summary condition was built and verified
(78-token summaries with zero plant keywords leaking), making it the powered
version of the interpretation-fidelity (probe) comparison. A B-causal arm was
added and run across all 12 conversations to remove a tail→summary text-ordering
confound, revealing that ordering itself partly explains C's apparent cost. The
complete synthetic CONT table showed H-gap beating B-min by +0.093 nats (the
cleanest "write-time encoding helps" signal) and C losing to B-causal even after
removing the ordering confound. All 14 Sonnet-judge batches (per user
instruction, Haiku was dropped in favor of Sonnet for all judging/subagent work)
completed (~1,200+ judgments), confirming the shadow condition held
(brief-condition summaries judged content-free) and surfacing a qualitative
pattern: retained-state arms (D, H-gap, low-α E) tend to admit ignorance on
evicted facts where B fabricates confidently — flagged as a potentially
deployment-relevant finding in its own right.

**Handoff documentation.** Per user request to ensure a fresh session could take
over, STATE.md (volatile snapshot, updated at each phase transition) and
AGENTS.md (stable rules/traps/reading order) were created and committed,
complementing the existing DECISIONS.md (why) and RESULTS.md. The user confirmed
this division of labor was sufficient. An external review caught four real
issues, all fixed and committed: `score.py local` mode diverged from
`score.py apply` on stance/ruled-out scoring (fixed to match the apply path's
judge-trusting logic); STATE.md had gone stale on the 30B download/run status;
DECISIONS.md was missing entries for the scoring-rule change, the Sonnet-judging
switch, B-causal repair, and 30B ladder facts; and analyze.py's per-conversation
appendix omitted the now-central H-gap/B-min/B-causal/α=0.25 arms (all now
included).

**Reframing toward a blog-post artifact.** Based on external-agent input
(attributed explicitly: design and execution by Claude Fable 5, adversarial
review/reframing input from OpenAI GPT-5.5, direction/sanity-checking by the
user), the intended final artifact shifted from an academic paper to a
HuggingFace community blog post plus GitHub repo, framed around genuine
uncertainty ("this seems obvious — am I missing prior art, or is it too trivial
to have been written up?"). Guidelines committed to the repo: state provenance
plainly in the body; optimize for findability by a future searcher
(plain-language framing up front, a simple consistent name for the technique, a
related-work section listing specific terms/papers searched and ruled out — gist
tokens, KV cache merging/CaM/KVMerger, Activation Beacon, compressed context
memory, etc.); report results regardless of valence; structure as mechanism →
ablation/headline numbers (framed as % of full-context/fresh-summary gap
recovered) → limitations → open question to readers; and a matter-of-fact,
non-apologetic tone. This is now the target document style for eventual
write-up, though academic-paper drafting, the memo, and further 30B
analysis-writing remain explicitly deferred until called for.

**Phase 1/Phase 2 restructuring.** The user determined the original experimental
focus (showing write-time KV state differs from re-encoded text / carries
context information) was a near-tautological mechanism observation, and
redirected the project's central question to a mitigation-first framing: can a
small, practical cache-state intervention reduce damage from conversation
compaction? The user directed two phases: Phase 1 — bring the already-in-flight
30B run to a clean, interpretable stopping point (no expansion of the old arm
matrix), update STATE.md/DECISIONS.md/RESULTS.md to reflect actual state, and
commit that as a discrete close-out; Phase 2 — design and execute one narrow,
clean mitigation-first contrast rather than a broad arm sweep. The chosen Phase
2 design is H-pack vs B-min-pack: identical packed summary tokens at identical
positions, differing only in write-time vs fresh-context encoding, scored
primarily on fabrication-vs-admission over evicted facts plus 24 newly authored,
collision-audited decoy probes (audited against actual conversation content and
re-audited after replacing 6 colliding decoys), with referent/sense accuracy as
a secondary metric and a wrong-summary donor control. Supporting machinery — key
re-rotation validated bit-exact against MLX's own RoPE via an LH identity-test
ladder (which also surfaced that 4-bit matmul kernels are
sequence-length-dependent, a documented noise floor mitigated by using
same-shape comparisons) — is built and committed; the Phase 2 4B run is queued
to launch once the GPU frees up (~1 hour of compute), with success criteria
requiring that B fails relative to A on a compaction-sensitive behavior, the
small intervention improves over B, and that improvement is not explained by
leakage, wrong/shuffled grafts, or generic fluency, plus at least plausible
deployability. G/SoftGraft and broad matrix expansion remain deferred pending
signal from this narrower test.

**Fine alpha-sweep request.** The user requested a finer ValueGraft alpha sweep
(0.00, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.35 rather than the coarse
0.25/0.5/0.75/1.0 grid), with the guardrail that alpha be tuned on a validation
split (proposed: c01–c06 + n01–n04) and reported only on held-out conversations,
to avoid overfitting to one corpus's idiosyncrasies. This was queued (task #11)
to run after the Phase 1 close-out and Phase 2 fabrication run, and machinery
makes it cheap since all alpha variants share one B prefill per conversation.

**Partial 30B results (through 10/12 main-pass conversations) meaningfully
revise the 4B-only picture and are provisional pending the
brief/shadow-condition pass and judging:** (1) the H-gap vs B-min encoding
advantage strengthens with scale (+0.126 nats at 7/7 vs 4B's +0.093 at 10/12);
(2) the alpha dose-response inverts — at 4B only low alpha helped and alpha=1
hurt, but at 30B alpha=1.0 now beats B (+0.053, 6/7 conversations, ~27% of the
A−B gap) and beats alpha=0.25, suggesting K/V-decoupling fragility may be a
small-model limitation rather than a general one, directly bearing on how the
alpha-sweep task should be scoped per model scale; (3) C (gapped retention)
shifts from harmful at 4B to roughly neutral at 30B (−0.03 vs B). Probe keyword
rates at 30B under the standard (non-shadow) summary condition remain
leakage-swamped, so the decisive interpretation-fidelity and fabrication tables
still depend on the queued brief-condition pass and judging. If the
honesty/fabrication effect replicates at 30B, the intended blog-post narrative
arc is: mechanism holds at both scales, and the deployable (contiguous,
no-position-surgery) intervention becomes net-positive specifically at larger
scale.

**Other queued brainstorming items (not commitments):** (1) the user's
suggestion to look into existing datasets containing agent/tool-call
trajectories (e.g., SWE-agent/OpenHands SWE-bench trajectory dumps) to assess
whether the current scaffolding could practically replay and evaluate compaction
impact on agentic/coding traces, and whether the effect is larger or smaller
there than in chat — logged as task #12, explicitly non-urgent; (2) review of
user-added `adaptive-blending-notes.md`, folded into task #11's scope covering
per-scale alpha grids.

**Standing execution order and process notes carried forward:** GPU work remains
strictly serial; judging/subagent work now uses Sonnet exclusively (never Haiku,
per explicit user instruction); STATE.md and DECISIONS.md must be kept current
at every phase transition as a non-negotiable habit going forward, per repeated
user reminders; the current queue order is: 30B main pass completion (at 10/12)
→ brief/shadow-condition pass at 30B → Phase 1 close-out commit (narrow scoring:
referent/sense + fabrication only, skipping stance/ruled-out which was at
ceiling at 4B) → Phase 2 H-pack vs B-min-pack fabrication run at 4B → fine alpha
sweep with validation/holdout split → brainstorming items as time permits. No
paper drafting, memo, or further academic-style artifacts are to proceed until
called for; the blog-post-style guidelines are the standing target format for
eventual write-up.
