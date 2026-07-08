_An extended debugging sequence resolves the cross-architecture ValueGraft
positive-control failure — root-caused to a wrong model checkpoint, not the
science — and pivots the sweep design to per-model-native rendering before the
16-model wide sweep launches and exposes a new wave of infrastructure bugs._

**Participants:** User and claude-opus-4-8.

## Corpus recovery and freeze

The conversation opens mid-recovery from an earlier destructive `rm` that
deleted six uncommitted corpus batch files. Recovery succeeded byte-identically
by re-running the subagents' underlying generator scripts rather than
re-authoring content. The corpus was merged (54 scenarios, 324 new plants, zero
non-ASCII, no banned names), and all 27 new conversations (c28–c54) were
rendered across six parallel batches and frozen — bringing the corpus to 54
conversations / 626 plants total. A git-branch mishap also surfaced here: a
debugger subagent had created `fix/selfgen-summary-think-alignment` and the
working tree had drifted onto it, causing several commits to land off trunk.
This was resolved with a fast-forward (`git branch -f trunk HEAD`, checkout,
delete branch) and codified as a hard rule: **work only ever happens on trunk**,
and any subagent that branches must be corrected immediately.

## The positive-control crisis and root cause

The debugger's alignment fix for the self-gen crash showed `status=OK` but
produced a failed positive control (referent CI spanning zero instead of the
expected +0.136, all categories shifted negative). This triggered an extended,
multi-stage misdiagnosis:

1. Initial hypothesis: the fix's think-block-stripping logic mis-grafted values.
   Code audit showed the design was actually sound (full think+answer content
   stayed in the write-time prefill; only position-mapping was answer-only).
2. Fable proposed a decisive **null self-graft test** (E := B must equal ~0 by
   construction) to distinguish a plumbing bug from genuine substance loss —
   this became the template "decisive O(1) test" pattern used repeatedly
   afterward.
3. A debugger subagent, tasked with running that test, instead spun for ~2 hours
   re-running full 40-minute positive-control runs without ever executing the
   cheap test — flagged explicitly as a "failing slow" anti-pattern.
4. Taking over directly, tracing revealed the _true_ upstream root cause: an
   earlier "alignment hardening" task had silently replaced the tolerant
   `build_alignment_difflib` (used to originally measure +0.156) with a strict
   `build_alignment_direct` at the shared `build_alignment` function used by
   **both** harnesses (`gap_closure_cat.py` and `cross_arch_probe.py`). The
   strict aligner couldn't handle the `<think>` block's 899-vs-538 token
   divergence on thinking-model self-gen. The one-line fix (revert the
   delegation) was applied.
5. Re-running still produced a negative referent — which turned out to be
   because **the wrong model checkpoint had been used throughout**:
   `Qwen/Qwen3-30B-A3B` (a thinking model) instead of
   `Qwen/Qwen3-30B-A3B-Instruct-2507` (the non-thinking model the +0.136
   baseline was actually measured on, and the harness's own documented default).
   Every symptom of the entire multi-hour saga — the `<think>` block, the
   token-count divergence, the alignment crash, the think-strip patch, the
   negative numbers — traced to this single wrong-model error.
6. On the correct checkpoint, the positive control **reproduced cleanly**:
   referent = +0.1246 (matching +0.136), with the expected referent > sense >
   stance ≈ null dissociation.

Lessons recorded: verify the exact model checkpoint against the known-good
baseline before any other debugging; when two independent harnesses fail
identically, suspect shared code first; and the user explicitly noted the
assistant should have consulted Fable hours earlier rather than grinding solo or
deferring to the user — "the urge to ask the user is the signal to consult Fable
and keep driving" became a recorded hard rule.

## Corpus dilution and the nativeness confound

Splitting the positive-control result by original (c01–c12) vs. new (c13–c54)
conversations showed the new conversations recovered almost nothing (+0.009
referent despite a substantial 1.12 pre-graft gap), diluting rather than
strengthening the corpus. Root cause: the new conversations' assistant replies
were authored by foreign frontier models (Sonnet/Opus/Fable/Codex) during this
session's fast rendering, rather than generated in-context by the test model
itself, as the original pipeline did.

Fable then identified a much larger risk: the entire positive control might be a
**Qwen-nativeness artifact** rather than evidence of a generalizable,
geometry-driven effect — since the original corpus's replies were themselves
Qwen-generated. If so, a cross-architecture sign map built on that shared corpus
would track "is this corpus native to model X" rather than attention geometry, a
confound that could have invalidated the marquee 16-model experiment before it
started.

The user's framing reframed this constructively: in real deployment, a model's
conversation history always consists of its own replies, so requiring native
replies isn't an artificial workaround — it matches reality. This led to the
adopted design, **"matched-scaffold, model-filled" (v2.1)**: scenarios (user
prompts, planted facts, gold continuations) stay shared across all 16 models,
but each model generates its own assistant replies and self-gen summary
in-context. The gold continuation stays shared, which is load-bearing — because
the metric (`raw_EB = lp_E − lp_B`) is a within-model difference on the same
shared probe, model-competence-on-the-probe cancels out and the sign is not
sensitive to whether the probe text is itself native to a given model. Remaining
nativeness effects (reply content/capability, headroom, task competence) were
converted into explicit measured covariates and gates (headroom floor
SC_HEADROOM_FLOOR=0.3, task-competence floor lp_A ≥ −8.0) rather than left as
confounds. This also resolved the corpus-dilution question: since native
rendering regenerates replies for all 54 scenarios uniformly, there is no
inherent "originals good / new scenarios bad" split — quality is determined
per-scenario by the gates, not by authorship batch.

Primary inference was anchored on two within-vendor dense/MoE pairs (Qwen3,
Gemma-4) where nativeness is controlled by construction, with the full 16-model
regression as confirmatory of one pre-registered geometry direction (predicting
sign, not magnitude).

## Build, validation, and scaling

A subagent implemented per-model in-context native rendering in
`cross_arch_probe.py` (`SC_NATIVE_RENDER`), preserving the shared gold
continuation and adding the headroom/competence/covariate gates. GPU
verification on the correct model reproduced the effect: referent CI [+0.012,
+0.195] (excludes zero), with `ruled_out` correctly floored by the headroom gate
rather than misread as harm — validating both the design and the gating logic on
real data.

This surfaced a severe cost problem: native rendering at ~10
minutes/conversation made a 16-model sweep infeasible at scale. Fable's
diagnosis (again sought proactively this time) identified the actual bottleneck
as sequential, unbatched per-token decoding, not the
render-once/replay-across-arms mechanism (already correctly implemented). The
prescribed fix was batched decode across a model's conversations plus a
batch-size cap to avoid OOM. A CPU-verified byte-identical implementation was
built, but the first GPU test OOM'd (12 conversations' KV caches simultaneously
exceeded the 80GB A100). A batch-size cap (chunks of 4, with automatic
degrade-on-OOM) fixed this and delivered a real but modest ~2.3× speedup rather
than the hoped 4–10×.

More importantly, the batched run's referent CI widened to span zero (+0.0485 vs
+0.10) and `evicted_fact` moved strongly negative (−0.31 vs −0.02). Fable's
verdict: the referent shift was likely noise (overlapping CIs, small n), but the
evicted_fact swing was too large to dismiss as benign bf16 rounding and could
plausibly bias which sign a model reports — an unacceptable risk given the
sweep's entire claim rests on sign. Recommendation: run the sweep on the
validated, slower per-token method now (wall-clock is roughly equal once
parallelized across one pod per model; batching mainly saves pod-hours, not
time), and pursue a cheap off-ramp (fp32 logits for the argmax only) as a
separate validation track rather than block the sweep.

## Wide sweep launch and immediate infrastructure debugging

With user direction to prioritize getting cross-architecture data flowing even
under some ambiguity, and after a budget check
(~$48 for a 12-conv version across 16 models, ~$60 for a version with 24-conv
depth on the 5 anchor models against a ~$65–70 balance and $80 cap — user chose
the fuller, tighter-margin version), the wide sweep was launched: one pod per
model, per-token native rendering, 24 convs on anchor models / 12 on breadth
models. The pre-registration was frozen (H1: QK-norm presence predicts a
positive graft sign) before any wide-sweep results could land, preserving its
confirmatory status. A QK-norm detection bug (had been silently reading False
for all 16 models because it checked config keys rather than actual loaded model
modules) was fixed just before freezing, since it was the leading candidate
predictor.

Provisioning immediately hit community A100 capacity exhaustion (not transient),
handled with a resilient background retry loop rather than manual babysitting.
Verifying the first-launched pod (rather than assuming it worked) caught two
separate sweep-breaking bugs in the launcher: environment variables (`MODELS`,
`SC_CONV_LIMIT`, etc.) were not being forwarded to the remote job process, and a
hardcoded `MODEL_TIMEOUT` of 3600s would have killed every per-token render job
partway through (anchor renders need ~5 hours). Both were fixed in
`job_sweep.sh` before wasting further pod-hours, with the timeout now scaled to
conversation count. The retry loop automatically picks up both fixes for the
remaining 15 pods as capacity frees.

## Process and documentation changes

Several durable process corrections were made and written into the repo (not
private memory, which subagents cannot see):

- **Only trunk** — no branches, ever; any subagent creating one gets corrected
  via fast-forward.
- **Consult Fable early**, specifically the moment a fix fails to converge in
  1–2 attempts, a subagent loops, or a result is confusing — not after hours of
  solo grinding, and not by deferring to the user instead (explicitly: turning
  to the user when stuck was called out as offloading, not diligence).
- **Commit and push after every unit of work.**
- File-purpose discipline was reaffirmed: FINDINGS.md for load-bearing results,
  INCIDENTS.md for failures/fixes, DECISIONS.md for dated decisions, STATE.md
  for current handoff state (kept refreshed), AGENTS.md for process rules/agent
  workflow.
- A new **`METHODS-PROVENANCE-REQUIREMENTS.md`** was created as a blocking
  requirement for the eventual writeup: every token's provenance (who/what
  generated prompts, replies, summaries, gold targets), exact checkpoint IDs,
  gating procedures, and design rationale must be documented, directly motivated
  by how much time was lost this session to an undocumented provenance detail
  (which model generated the assistant replies). A companion "must-state" list
  of scientific nuances (nativeness scope condition, per-model-native design,
  shared-gold difference metric, gates, alignment method/reversal, dissociation,
  keys-neutral, architecture boundaries) was added as an explicit review
  checklist.
- The paper's author-attribution byline was fixed to a clean hierarchy (Fable 5
  and GPT-5.5 as main authors, Jeremy Banks as guidance/direction, light
  "assistance from" credit to Opus 4.8/Sonnet 5/Gemini Pro 3.1, no funding
  mention, no per-model itemization), correcting writeup-guidelines.md, which
  had drifted to contradict the report's existing byline.
- The user, checking in periodically ("are you still working," "do you need me
  to keep checking in"), established that monitors must actively re-invoke the
  assistant on job completion/failure rather than requiring the user to prompt
  for status — this is now the operating assumption for the wide sweep.

## State at conversation boundary

The core ValueGraft effect is confirmed real and specific (referent recovery
~~+0.10–0.12, correct dissociation, validated on the correct model with native
rendering). The 16-model wide sweep is launched and running with a frozen
pre-registration, using the slower but validated per-token native-rendering
method. The first anchor pod (w1, Qwen) is running after two launcher bugs (env
forwarding, render timeout) were caught and fixed by direct verification rather
than assumption; the remaining 15 pods are queued behind a retry loop against
exhausted community GPU capacity. Budget is tight (~~ $60 of a ~$65–70 balance
under an $80 cap). Next immediate work is building the results-harvesting and
scientific-significance notification tooling so completed model runs can be
summarized meaningfully as they land, plus watching for further infrastructure
issues on the remaining pods.

---
