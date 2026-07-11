_This conversation covers completion and interpretation of the 4B KV-cache
semantic-continuity experiment, the interim 30B scale run, and a pivot toward a
mitigation-first ValueGraft study targeting compaction damage rather than merely
demonstrating cache-state differences._

**Participants:** User, claude-fable-5, and claude-sonnet-5.

**Handoff State.** The 4B evidence package is complete: 2,520 score rows, 2,089
judge items, no missing verdicts, all B-causal repairs complete, and clean/brief
mitigation tables matching committed machine-readable results. The 30B targeted
run is currently at 10/12 synthetic conversations in its main pass; it uses the
trimmed mitigation-relevant arms (A/B/C/D/H-gap/B-min/E-post α∈{0.25,1.0}),
followed by the brief/shadow condition. After completion, the committed plan is
narrow scoring of referent/sense and evicted-fact fabrication/admission only, a
scale addendum only if interpretable, synchronization of STATE.md and
DECISIONS.md, and a clean Phase 1 close-out commit.

The 4B findings establish a dose-response and important caveats: E-post α=0.25
gives a small positive continuation-NLL effect, heavy grafting hurts, C is
mildly negative, and negative controls are catastrophic. H-gap beats B-min on
the same summary text, indicating an in-context encoding advantage; B-causal is
better than C, showing an ordering confound in the original comparison. Standard
summaries leak many target facts, so the brief/shadow condition—verified to
contain no plant keywords—is the decisive probe condition. Sonnet, not Haiku, is
the mandated judge model. Judge observations consistently identify
referent/sense failures, confident fabrication of unavailable numeric facts,
occasional honest admission, and subtle reintroduction of ruled-out options;
all-fail probes caused by overly specific gold answers must be flagged.

Interim 30B results from 7/12 conversations materially changed the scale
hypothesis: H-gap beat B-min by +0.126 nats in all seven cases versus +0.093 at
4B; α=1.0 beat B by +0.053 in six of seven, reversing the 4B high-alpha harm;
and C was near-neutral rather than clearly harmful. These are provisional and
leakage-swamped under the standard condition; the brief-condition judged tables
are required before claiming scale-dependent deployability or an honesty effect.

**Methodological Pivot.** The project now prioritizes: “Can a small, practical
cache-state intervention reduce compaction damage relative to text-only summary
compaction?” The selected Phase 2 primary contrast is H-pack versus B-min-pack:
identical packed token sequence and positions, differing only in write-time
versus fresh summary encoding. It is intended as a contiguous,
prefix-cache-friendly intervention rather than diagnostic surgery. The
evaluation emphasizes fabrication-versus-admission on evicted facts, with
reference resolution secondary, a wrong-conversation donor control, leakage
audits, and freshly authored collision-audited decoy probes. Key re-rotation
machinery and identity tests are implemented and validated against mlx; the work
also documented sequence-length-dependent 4-bit matmul noise, so same-shape
comparisons are required. G/SoftGraft remains deferred.

Phase 2 machinery, design documentation, decoys, and runner are built and
committed while the 30B run continues. The next GPU experiment is the 4B
H-pack/B-min-pack fabrication run, expected to take about one hour after GPU
availability. A later ValueGraft α sweep is queued with validation/holdout
discipline: initially α∈{0.00, 0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.35}, but
the 30B interim inversion means the eventual grid should be scale-specific and
extend upward at 30B. Proposed validation split is c01–c06 plus n01–n04, with
held-out conversations reserved for reporting.

Repository handoff discipline is explicit: AGENTS.md contains stable rules and
traps; STATE.md is the volatile operational snapshot; DECISIONS.md records
methodology, runtime facts, deviations, and interpretation decisions. Recent
fixes synchronized local fallback scoring with external judge application,
corrected the per-conversation appendix to include H-gap/B-min/B-causal and
α=0.25, and updated 30B state and decisions. These files must be refreshed and
committed at every phase transition.

The intended final artifact is a straight, findability-oriented Hugging Face
community blog post plus GitHub repository, not an academic paper. It should use
a simple consistent name such as “dual-context KV blending,” front-load
plain-language search terms (“summary tokens lose their original activations
after context compaction”), explain the mechanism, present the ablation matrix
and gap-recovery metrics, include latency/compute overhead, limitations, related
terminology, and an open question about whether the idea is already known.
Provenance should state that Anthropic Claude Fable 5 designed and largely
executed the work, OpenAI GPT-5.5 contributed adversarial review and reframing,
and the user directed, made scaling/budget decisions, and sanity-checked what
they could personally verify.

A later brainstorming task concerns established agent-code trajectory datasets,
particularly SWE-agent and OpenHands/SWE-bench traces. The eventual questions
are whether tool-call trajectories can be replayed through the scaffolding,
whether compaction harms agent retry/rule-following behavior more than chat, and
whether next-action/file/command prediction provides a cheap offline metric.
This is queued as task #12 and is not part of the immediate close-out.

## Conversation sources

- `bda7fb9f-f447-4890-904b-dde750ff3370`
- `a8fd1509db3de6c39`
- `a100f87abe1428764`
- `a7c60c827a7188114`
- `a2cb1f7e1fa5ad914`
- `afa0c3d06123ecabb`
- `a6d6f6f737e311078`
- `a4ef6c0bfcc487062`
- `add7bf0e5494a1800`
- `a94adf759a7433365`
- `a020460af4ff5ff2f`
- `aa912ba5220ddf3eb`
- `a8f82434271997146`
- `a9c176854d730e357`
- `afc278d53d351a4f7`
