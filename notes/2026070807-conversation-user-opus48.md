_This conversation covers a marathon debugging saga in the ValueGraft
cross-architecture research program: resolution of the wrong-checkpoint root
cause, discovery and resolution of a nativeness confound that threatened the
validity of the planned 16-model sweep, and a cluster of process corrections
about when to consult the Fable model versus the user, and how to record
learnings and project state._

**Participants:** User and claude-opus-4-8.

**Technical resolution.** The multi-arch harness (`cross_arch_probe.py`) was
verified on the correct checkpoint (Qwen3-30B-A3B-Instruct-2507), reproducing
the trusted apparatus's referent CI (+0.034 to +0.230, midpoint ~+0.13) and the
referent>sense>stance dissociation, confirming the harness itself was never the
problem — the entire earlier failure cascade traced to running a
thinking-variant checkpoint instead of the Instruct-2507 checkpoint the original
result was measured on.

**The nativeness confound and redesign.** Fable identified a further risk before
wide spending: the original corpus (c01–c12) had assistant replies generated
in-context by a Qwen model, while the newly-doubled corpus (c13–c54) had foreign
replies from a frontier-model mix — the same condition that suppressed recovery
in the new convs. This raised the possibility that the cross-architecture sign
map would track reply-nativeness rather than attention geometry. The user's
framing sharpened the fix: in real deployments, a model's assistant replies are
always its own, so requiring native replies isn't a workaround but the realistic
condition. This led to design v2.1, "matched-scaffold, model-filled": scenarios
(user prompts, planted facts, gold continuations) stay shared and reusable
across all 54 conversations, but each model generates its own assistant replies
and self-gen summary in-context at measurement time. Because the raw_EB metric
is a difference against a shared gold continuation for the same model,
continuation-nativeness cancels in the metric — meaning nativeness is expected
to affect magnitude, not sign, and the residual nativeness effects (reply
content/capability, headroom, task competence) become measured covariates and
gates rather than confounds. This design keeps the full 16-model, 54-scenario
ambition rather than forcing a fallback to a smaller validated set; per-model
native rendering was implemented (`SC_NATIVE_RENDER`) and a verification run on
Qwen was launched to confirm it reproduces the known-positive effect before any
wide spend. A separate, informal Mistral-on-Qwen-native-corpus control was
deprioritized once the redesign superseded its purpose (each model will get its
own native corpus in the real sweep); its pod also failed to provision.

During the render verification, a secondary issue surfaced: the per-model
geometry table (`data/model_geometry.json`) reads QK-norm as false for all 16
models due to a config-detection bug — QK-norm actually applies to Qwen3,
Gemma-3/4, and OLMo-2, and since it was the leading candidate for the single
pre-registered geometry hypothesis, detection needs to be fixed by inspecting
loaded model modules rather than config keys before the hypothesis is finalized.
The native render itself proved much slower than expected (~10 minutes per
conversation due to real autoregressive generation cost, not an inefficiency
bug), implying the 16-model × 24–54-scenario sweep will be expensive;
mitigations noted for later (fewer scenarios per model outside the paired
anchors, shorter replies, harder pod parallelization) were recorded but not yet
acted on.

**Process corrections (durable, now in-repo).** Several corrections were made
explicit in AGENTS.md rather than left to private memory, per the user's
directive that private memory files are invisible to subagents and collaborators
and therefore useless for durable process rules:

- Consult Fable at the first sign of being stuck (1–2 failed attempts, a looping
  subagent, or a confusing result), not after hours of narrow debugging — deep
  technical focus suppresses the instinct to step back and check "boring" things
  like which checkpoint is actually running.
- The urge to stop and ask the user for direction is itself the signal to
  consult Fable and keep working autonomously; escalate to the user only for
  decisions that are genuinely theirs (scope, spend, taste), not to offload
  being stuck.
- Learnings and rules belong in the repo's purpose-specific files, not private
  assistant memory — file roles were re-clarified: FINDINGS.md (load-bearing
  scientific claims + evidence), INCIDENTS.md (failures and fixes), DECISIONS.md
  (date/decision/reason log), MASTER-PLAN.md (durable design spine), STATE.md
  (current handoff state, which had gone stale and was refreshed), AGENTS.md
  (workflow rules and file map), README.md/REPORT.md (the deliverable blog
  post), writeup-guidelines.md (how to write it).
- The user pushed back sharply when it appeared the doubled corpus and possibly
  the whole result might be invalid; the resolution clarified that the original
  12-conversation result is unaffected, only the newly-rendered (not the
  newly-designed) c13–c54 conversations and the untested cross-architecture
  extension were in question — and that under native rendering there is no
  a-priori good/bad split between the original and expanded scenario sets, since
  reply-nativeness (the actual problem) is fixed uniformly for all 54 under the
  new design; whether the c13–c54 scenario _designs_ (independent of who wrote
  replies) carry the effect as well as the originals remains to be tested
  empirically once native rendering is verified.

**Paper/documentation requirements.** In direct response to the nativeness
episode — where an undocumented detail (who generated the assistant replies)
turned out to determine result validity — a new blocking requirement,
`METHODS-PROVENANCE-REQUIREMENTS.md`, was created: a checklist mandating that
the eventual write-up document exactly who/what generated every category of
content (prompts, replies, summaries, gold targets), exact checkpoint IDs,
procedures, gates, and design rationale, sufficient for reproducibility. This is
wired as a blocking gate into writeup-guidelines.md and AGENTS.md's publish
workflow, separate from the author-attribution byline. The byline itself was
reconciled between writeup-guidelines.md and the current report: a clean
hierarchy (Fable 5 and GPT-5.5 as main authors, Jeremy Banks as
guidance/direction, light "assistance from" credit to Opus 4.8, Sonnet 5, and
Gemini Pro 3.1), with no per-model contribution breakdown and no funding mention
— guidelines were rewritten to match rather than contradict the report.

**State at end of chunk.** The native-render verification (12 scenarios,
Qwen3-30B-A3B-Instruct-2507, pod ujg3iqy5cirnga) was still in the rendering
phase, having cleanly generated native replies through roughly c07 of 12 with no
errors, GPU utilization confirming active generation; no referent number had yet
been produced. This run is the gate for validating the v2.1 design before
committing to the full 16-model, per-model-native sweep. Also staged and ready
pending that gate: a draft pre-registration (`PREREGISTRATION.md`) covering
estimand, frozen metric, and exclusion rules, with the primary inference resting
on two within-vendor dense/MoE pairs (Qwen3, Gemma-4) and a confirmatory
16-model regression on one pre-registered geometry direction; and the (currently
buggy) geometry table for all 16 candidate models.
