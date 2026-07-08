_This chunk covers the ValueGraft project's wrong-checkpoint root-cause
resolution, an operational-behavior crisis around when to escalate (Fable vs.
the user) that produced several durable process rules, the redesign of the
cross-architecture sweep into a per-model-native "matched-scaffold,
model-filled" methodology, new provenance/attribution documentation requirements
for the eventual write-up, and the start of native-render verification on the
correct Qwen checkpoint._

**Participants:** User and claude-opus-4-8.

The wrong-model incident (running the thinking variant instead of
`Qwen/Qwen3-30B-A3B-Instruct-2507`) was confirmed as the sole cause of the
multi-hour alignment saga; on the correct checkpoint, `cross_arch_probe.py`'s
multi-arch harness reproduced the trusted apparatus's referent CI (~+0.12,
excluding zero) with the same referent>sense>stance dissociation, validating
that the harness itself was never broken. A cosmetic bug was noted (raw_EB point
estimate stores as None while CIs compute correctly) but not yet patched.

Following that resolution, a second-order risk surfaced: Fable (consulted for
strategic review after the marathon) identified that the original
positive-control corpus (c01–c12) had its assistant replies generated in-context
by a Qwen model, meaning the measured +0.12 could be Qwen-native rather than a
genuine cross-architecture geometry effect — the same class of confound that
caused the newly-doubled corpus (c13–c54, rendered with foreign frontier-model
replies) to underperform. This "nativeness confound" was treated as a potential
threat to the entire planned 16-model sweep and became the top pre-spend gate. A
decisive test (Mistral-Small on c01–c12, to see whether the effect collapses
off-Qwen) was launched but its pod failed to provision reliably; this bonus test
was deprioritized once a design fix superseded the need for it.

The user pushed back sharply on process throughout this stretch, producing
several durable behavioral corrections now recorded in the repo (mainly
AGENTS.md) rather than in private assistant memory, since subagents and
collaborators cannot see the latter: (1) consult Fable at the first sign of
being stuck (non-convergence in 1–2 attempts, a looping subagent, or a confusing
result) rather than after hours of narrow debugging; (2) the impulse to stop and
ask the user for direction is itself the signal to consult Fable and keep
working autonomously instead — escalate to the user only for decisions that are
genuinely his (scope, spend, taste), not for technical unblocking; (3) learnings
and rules belong in the repository's purpose-specific files (FINDINGS.md for
load-bearing scientific claims, INCIDENTS.md for failures/fixes, DECISIONS.md
for dated decisions with reasons, STATE.md for current handoff state, AGENTS.md
for workflow/process rules), not dumped indiscriminately into AGENTS.md, and
STATE.md in particular must be kept current as the live handoff artifact. The
user also had to clarify multiple times that the underlying scientific result
was not in doubt — only the newly-doubled corpus and the untested
cross-architecture extension were — after the assistant's explanations conflated
"the effect might not exist" with "one specific extension's design needs
fixing."

Working through the nativeness question, the user's own framing (that in real
deployments a model's assistant replies are always its own output, so requiring
native replies is realistic rather than a confound) reframed the problem
productively. Fable confirmed and refined this into a "matched-scaffold,
model-filled" design (v2, later v2.1): shared scenario scaffold (user prompts,
planted facts, gold continuation, compaction structure) across all 16 models,
but each model generates its own in-context assistant replies and its own
self-generated compaction summary. Because the frozen metric raw_EB = lp_E −
lp_B is a difference computed on the same shared gold continuation for the same
model, generic nativeness/capability effects on that continuation cancel, so
nativeness mainly threatens magnitude, not sign — protecting the paper's core
sign-based claim. Remaining nativeness effects (reply/summary content, headroom,
task competence) become measured covariates and gates rather than confounds:
per-model reply information-content and length as regression covariates,
headroom (A−B gap) used to normalize and floor-gate interpretability, and a
competence gate requiring the model to solve the task under full context. The
two within-vendor dense/MoE pairs (Qwen3-30B-A3B/Qwen3-32B and a Gemma-4 pair)
carry primary inferential weight since nativeness is controlled by construction
there; the full 16-model regression is confirmatory of a single pre-registered
geometry direction (predicting sign, not magnitude), plus a small
shared-fixed-corpus arm on a few models for a causal mechanism claim. This
design change also resolved the corpus-provenance question the user raised
repeatedly: reply-nativeness and scenario-design-quality are separate axes that
were previously confounded (c01–c12 had native Qwen replies plus original plant
design; c13–c54 had foreign replies plus newer plant design), and native
rendering removes the nativeness axis for all 54 scenarios equally, so there is
no a-priori basis for treating c13–c54 as bad — headroom/competence gates, not
batch origin, will determine which scenarios carry the effect. This means the
originally-planned full 54-scenario dataset remains usable, not just the smaller
original 12; nothing from the corpus-doubling effort was wasted, only some
previously-rendered conversations are obsolete as artifacts (their underlying
scenarios persist in `data/scenarios.json`).

A harness build implementing per-model native rendering was completed and
committed to trunk (`src/cross_arch_probe.py`, gated by `SC_NATIVE_RENDER`),
including headroom normalization, a competence gate, and reply covariates, with
the gold continuation kept shared by design. CPU self-tests confirmed structural
correctness (reproducing c01's conversation shape); GPU verification was then
launched on the cached pod (Qwen3-30B-A3B-Instruct-2507) to check whether native
rendering reproduces the ~+0.12 referent effect. This verification proved much
slower than expected (~10 minutes per conversation due to full autoregressive
in-context reply generation for ~22 turns per scenario, roughly 7K tokens),
which was flagged as a real scaling concern for the eventual 16-model × up to
54-scenario sweep — likely requiring fewer scenarios per model for
lower-priority architectures, shorter replies, and/or heavier pod
parallelization (one model per pod). As of the end of this chunk, the
verification run was in progress on the pod (c01–c07 of 12 rendered cleanly, no
errors, GPU utilization ~38%), with the referent number not yet produced; this
is the immediate gating result for validating the redesign before wide spend.

Separately, a geometry table for all 16 candidate models was gathered
(`data/model_geometry.json`) covering GQA ratio, head_dim, rope_theta, and layer
count, but QK-norm detection was found to be broken (reads False for every
model, including Qwen3/Gemma-3-4/OLMo-2, which do use it) because it read from
top-level config keys rather than inspecting loaded model modules for
q_norm/k_norm layers; this needs fixing before QK-norm can be used as the
pre-registered geometry predictor, since it was the leading candidate given it
is the sharpest attention-mechanism difference between the known-positive Qwen3
and known-negative Qwen2.5. A pre-registration document (`PREREGISTRATION.md`)
was drafted with the estimand, frozen metric, and exclusion rules, pending
finalization of this single geometry hypothesis.

Two new documentation requirements were established for the eventual write-up
(an HF community blog post plus GitHub repo, not an academic paper): a new
blocking file, `METHODS-PROVENANCE-REQUIREMENTS.md`, mandates exhaustive
documentation of who/what generated every piece of data (prompts, replies,
summaries, gold targets), exact model checkpoints, and design rationale —
created because the entire nativeness saga stemmed from this provenance detail
never having been tracked, and wired as a blocking gate into
`writeup-guidelines.md` and `AGENTS.md`'s publish workflow. Separately, the
author attribution byline in `writeup-guidelines.md` was corrected to match the
existing clean hierarchy already used in the report header — Fable 5 and GPT-5.5
as main authors, Jeremy Banks credited for guidance/direction, light "assistance
from" credit to Opus 4.8, Sonnet 5, and Gemini Pro 3.1 — removing an
inconsistent per-model contribution breakdown and a funding mention that had
crept into the guidelines; this attribution question is explicitly kept separate
from the provenance/methods documentation requirement.

Handoff state at the end of this chunk: pending task #28 (redesigned cross-arch
sweep) is in progress with the native-render GPU verification running on pod
`ujg3iqy5cirnga` (Qwen3-30B-A3B-Instruct-2507, A100 80GB); task #38 (paper must
document data provenance per the new requirements file) remains blocking for the
write-up phase. Immediate next steps once the verification completes: confirm
the referent effect reproduces (~+0.12) under native rendering, extend
verification to all 54 scenarios to test whether the newer scenario designs
carry the effect, fix QK-norm detection from loaded model modules, finalize the
single pre-registered geometry direction, then provision and run the full
16-model per-model-native sweep.
