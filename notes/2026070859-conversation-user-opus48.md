_This conversation covers the resolution of the wrong-checkpoint saga's
aftermath, a sustained sequence of process corrections about when to consult
Fable versus the user, the discovery and resolution of a nativeness confound
threatening the cross-architecture sweep's validity, and the redesign of the
experimental harness to a per-model-native rendering approach that preserves the
full 16-model ambition._

**Participants:** User and claude-opus-4-8.

The positive control was confirmed reproducing on the correct model checkpoint
(Qwen3-30B-A3B-Instruct-2507): referent CI [+0.034, +0.230], midpoint ~+0.13,
matching the trusted apparatus's prior +0.12, with the referent>sense>stance
dissociation intact and identity/alpha0 checks passing. This validated that the
multi-hour debugging saga earlier in the session was purely a wrong-checkpoint
problem, not a flaw in the harness or the underlying effect.

A significant process correction occurred around consultation discipline. The
user pointed out that hours were spent on narrow technical debugging before
consulting Fable for a strategic read, when a basic question ("are you running
the exact checkpoint the original result was measured on?") would have resolved
the issue quickly. This was recorded as a rule in AGENTS.md: when a fix hasn't
converged in 1–2 attempts, a subagent is looping, or a result is confusing,
consult Fable immediately rather than after extended debugging. The user further
noted that turning to the user for direction when stuck was itself a failure
mode — since Fable is available for autonomous strategic consultation without
interrupting the user, the correct sequence is to consult Fable first and
continue working, reserving user escalation for decisions that are genuinely the
user's to make (scope, spend, taste). This was also recorded in the repo (not
private memory, since private memory files are invisible to subagents and
collaborators — an earlier practice of recording learnings in private
`~/.claude` memory was corrected by moving them into AGENTS.md and other
repo-tracked files).

The user also flagged that project-management discipline had blurred — multiple
purpose-specific tracking files exist (FINDINGS.md for load-bearing scientific
claims, INCIDENTS.md for failures and fixes, DECISIONS.md for dated rationale,
MASTER-PLAN.md for the durable design, STATE.md for current handoff state,
AGENTS.md for workflow rules) and STATE.md in particular had gone stale, still
reflecting pre-resolution status. STATE.md was refreshed to current reality.

Fable identified a second, more consequential issue: the original
positive-control corpus (c01–c12) had assistant replies generated in-context by
a Qwen model, meaning the effect could be native to Qwen and could fail to
generalize — the same failure mode that had caused the newly-doubled corpus
(c13–c54, rendered with foreign frontier-model replies) to underperform. If
true, a cross-architecture sign map built on this corpus would track per-model
reply-nativeness rather than attention geometry, which would undermine the
planned 16-model sweep. The user asked several clarifying questions to separate
what was and wasn't at risk; the assistant clarified that (1) the core effect
and original corpus remain valid and unaffected, (2) only the newly-doubled
corpus's _rendered conversations_ (not their underlying scenario scaffolding)
were compromised by foreign replies, and (3) the cross-architecture geometry
claim was an unpublished, unverified extension, not an existing finding under
threat.

The user's expressed preference was to keep the full wide 16-model plan rather
than scale back, conditional on making it valid. The user also observed that in
real deployments, assistant replies in a conversation are always native to the
model that produced them, so requiring native replies is not an artificial
workaround but a realistic constraint — this reframing pointed to the resolving
design: rather than testing all models on one shared (Qwen-native) corpus, each
model should generate its own native assistant replies and summary from a shared
set of scenarios (user prompts, planted facts, gold continuations), while the
probe's gold continuation stays shared. Fable validated this as the correct
design ("matched-scaffold, model-filled," v2.1), noting that the graft metric (a
difference between grafted and baseline log-probabilities on the same shared
gold continuation) causes generic-nativeness effects to cancel, since both terms
are affected equally. Residual nativeness effects — reply content/capability,
headroom (whether a model's own summary evicts the referent at all), and task
competence — become measured covariates and gates rather than confounds:
reply-covariate regression, headroom normalization with a floor gate, and a
competence gate on full-context log-probability. Primary inference weight falls
on two within-vendor dense/MoE pairs (Qwen3, Gemma-4) where nativeness is
controlled by construction; the full 16-model regression becomes confirmatory of
one pre-registered geometry direction (predicting sign, not magnitude). This
design keeps the entire 54-scenario corpus usable, since native rendering
regenerates replies for every scenario, removing the earlier c01–c12-vs-c13–c54
split, which the assistant acknowledged had conflated reply-nativeness (a
design-independent problem, now fixed uniformly) with underlying scenario-design
quality (a separate, still-open question that native rendering tests fairly for
the first time).

The user required that experimental methodology — specifically data provenance
(who or what generated each piece of conversation content, exact checkpoints,
procedures, gates) — be documented far more thoroughly than in prior paper
drafts, calling the previous omission unacceptable and something that must not
recur, since the omitted nativeness detail turned out to be load-bearing for
validity. This was recorded as a blocking requirement in a new
`METHODS-PROVENANCE-REQUIREMENTS.md` file, referenced from
`writeup-guidelines.md` and `AGENTS.md`'s publish workflow, so the write-up
cannot proceed without addressing it. Separately, the user specified the
author-attribution byline should remain a clean hierarchy (Fable 5 and GPT-5.5
as main authors, Jeremy Banks providing guidance, light "assistance from" credit
to other models, no funding mention, no per-model contribution breakdown) — the
guidelines had drifted to include per-model itemization and a funding mention,
and were corrected to match the report's existing byline, with an explicit note
distinguishing this attribution byline from the separate, detailed
data-provenance methods section.

Implementation proceeded on the per-model native-rendering harness
(`src/cross_arch_probe.py`): an `SC_NATIVE_RENDER` mode was added that ports the
original corpus generator's in-context reply-generation approach to the HF/torch
path, producing native replies and self-generated summaries per model while
keeping the gold continuation shared, plus headroom, competence, and
reply-covariate gates. This passed CPU self-tests reproducing the original
conversation structure. A GPU verification run was launched on the cached pod
(Qwen3-30B-A3B-Instruct-2507) to confirm native rendering reproduces the ~+0.12
referent effect. This run proved considerably slower than expected —
approximately 10 minutes per conversation due to genuine autoregressive
generation cost (~7K tokens per conversation), putting the 12-scenario
verification at roughly two hours rather than the initially estimated 30–60
minutes, and flagging a real feasibility concern for the eventual 16-model × up
to 54-scenario sweep. The assistant characterized the bottleneck as inherent
generation cost rather than a fixable inefficiency (an initial suspicion of
unnecessary re-prefill was ruled out) and recorded a scaling plan for later
decision: reducing scenarios per model, shortening replies, and/or parallelizing
across more pods. As of the end of this transcript, the verification run was
still in progress (through roughly c07 of 12), with no referent number yet
produced; a Mistral-Small nativeness-control run on the original corpus was
abandoned as unnecessary after the design redesign (its intended role — showing
the effect fails on non-native replies — is now superseded by the
per-model-native design, and its pod provisioning was unreliable regardless).

Also completed during this stretch: a geometry table for all 16 candidate models
(GQA ratio, head dimension, RoPE theta, layer count, QK-norm flag) was gathered
and saved to `data/model_geometry.json`, but QK-norm detection was found to be
broken (reads false for all models, including Qwen3/Gemma/OLMo-2 which do use
it) because it was checked via config keys rather than loaded model modules —
this needs fixing before finalizing the pre-registered geometry hypothesis,
since QK-norm was the leading candidate predictor. A pre-registration draft
(`PREREGISTRATION.md`) was started, covering the estimand, frozen metric,
exclusion rules, and primary/secondary inference structure, with the specific
geometry direction still to be finalized pending the QK-norm fix and a further
Fable consultation.

**Handoff state.** The native-render verification on Qwen (pod ujg3iqy5cirnga)
was the immediate blocking gate: pending its referent result reproducing ~+0.12,
the plan is to fix QK-norm detection from loaded model modules, finalize the
single pre-registered geometry direction, extend verification to the full
54-scenario set (to test the previously-untested design quality of c13–c54 under
fair native rendering), and then provision and run the full 16-model
per-model-native sweep — factoring in the render-time scaling concern discovered
in this session. No fixed ETA was given for the verification's completion beyond
the revised ~2-hour estimate for the 12-scenario run, which had not yet
completed at the end of this transcript.
