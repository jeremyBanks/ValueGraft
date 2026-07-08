_This shard covers the aftermath of the mainline report-finalization push, an
escalating measurement crisis in which the placebo-controlled effect-bound probe
appeared to contradict the paper's headline grafting result, and the diagnosis
and correction of a broken statistical estimator underlying the gap-closure
metric — ending mid-investigation into an "alignment/token matching" mechanism
the user flagged as alarming and possibly discarded long ago._

## K/V question closed, lens thread bounded with a null

The cross-check with the other agent's policy-registry microtest resolved
cleanly: the two lines of evidence agree everywhere they overlap (V-only
dominates for semantic-phrase recovery), and the microtest's "keys help" signal
was isolated to short-identifier targets never tested in the main sweep, with
the divergence itself statistically weak (winner's-curse from best-of-15-policy
selection, tiny cells). The user then explicitly ordered a stop to further
key-digging ("keys are a settled question, not something we need to investigate
anymore") — this was reaffirmed later in the shard and recorded as a standing
decision. The identifier-morphology follow-up task was dropped.

A rigorous placebo-controlled "effect-bound" experiment (teacher-forced logprob
lift over a fixed window, aligned graft vs. random-value graft, bootstrap CIs)
was built at Fable's suggestion to replace an earlier rejected idea (a vivid
free-generation trajectory scan, which Fable advised against as likely to invite
over-narration of noise). On 27B this produced a clean result: the aligned graft
beats a random placebo decisively (CI excludes zero), while the graft's
advantage over plain compaction (E−B) was null over the narrow measurement
window — consistent with 27B already being the "weak" model per the paper's
non-replication in §7.

## Operational incidents and rules from pod/job management

Several infrastructure failures recurred and were converted into standing rules
(numbered in INCIDENTS.md/AGENTS.md, now in the high-20s):

- A launched job can silently fail to start (bad `cd` path, missing packages,
  torchaudio/torch symbol mismatch) while looking "running" in a naive check —
  leading to a job that billed for ~time without doing any work. New rule:
  verify real work has started (GPU memory loaded, logs progressing past setup)
  before trusting a launch, and specifically verify the _correct_ model loaded
  (a later incident found a "30B" run had silently re-run 27B because a
  `--model` flag/env var was ignored).
- A completed job can look identical to a crashed one (process gone, GPU freed)
  — a completion must be distinguished by checking the log reached its expected
  endpoint and the output file exists, not inferred from process/GPU state
  alone.
- Quiet/silent monitors made a crashed job invisible longer; monitors should
  surface only on real work stalling, not stay silent indefinitely.
- Suppressing rsync/deploy stderr caused a second silent deploy failure (missing
  rsync binary on a fresh pod); deploy output must not be suppressed.
- A community pod's fixed 200G disk filled completely (three cached ~65GB model
  weights) and killed the pod outright, losing an in-progress download (no
  analysis-relevant data lost). Follow-up pods were provisioned with larger disk
  (400G) and disk-headroom checks were added before each download, plus eviction
  of cached model weights between models.
- New convention: experiment run scripts must accept explicit
  `--model`/`--output` and self-announce the resolved model and output path in
  logs, and outputs must use unique, self-describing filenames (model + date) to
  prevent one run's output from silently overwriting another's.

## Repo cleanup and report framing

At user request, a full repo-wide markdown cleanup was executed: obsolete
top-level docs, all `referent_recovery_microtest/` markdown, and the jlens
findings markdown were moved into `docs/` with `YYYY-MM-DD-HH-slug` prefixes
derived from each file's git first-commit date (via a delegated subagent,
git-recoverable); the `referent_recovery_microtest/` directory itself was
deleted after its markdown was preserved. Exactly nine top-level files were
retained as agent-essentials (AGENTS, DECISIONS, FINDINGS, INCIDENTS,
MASTER-PLAN, README, REPORT, STATE, writeup-guidelines). The
`jlens_boundary_probe/` code was deliberately held back from deletion while lens
experiments were still active.

External feedback recommended keeping the repo otherwise messy but adding a
minimal "Reproduce the core result" section (pinned model, one setup command,
one run command, expected numbers) as the actual deliverable for outside
scrutiny. This was adopted as a report-readiness task folded into the
cross-architecture harness design, with `cross_arch_probe.py` intended to double
as the public repro entry point. Per user request, a short community-note-style
framing paragraph (acknowledging no found prior art and inviting input) was
added by Fable at the very top of REPORT.md/README, before the title, followed
by a horizontal rule; the paper's title was also refreshed at user's request to
Fable's recommendation, "Value grafting: recovering lost semantic continuity
when a conversation is compacted." Standing rule established: REPORT.md is the
working file; README.md is only updated by explicit re-promotion after major
updates, not edited directly.

## Phase 4 mandate: rebalance toward the primary result

The user issued an explicit strategic correction: the lens (interpretability)
material had become disproportionately prominent (76 uses of the word "lens"
across three distinct named instruments — logit lens, tuned lens, J-lens — with
many ambiguous bare "the lens" references) relative to its actual payoff, which
turned out to be a near-total dud (0/43 clean forks in the pre-registered
free-generation probe). The user directed that the paper's center of gravity
return to the grafting technique itself as the primary novelty, with two levers
to strengthen it: running more experiments (not excluded — user clarified
"examples" was an additional option, not a replacement) and mining more
qualitative examples from existing data (cheap, no pod cost). The user also
directed that Fable, when doing conceptual/editing work, be explicitly briefed
on this "primary vs. secondary" focus rather than operating
direction-agnostically, and that "terminology consistency" (the overloaded
"lens" problem) be added as an explicit review dimension for future passes.

This produced a large, mostly-approved Phase 4 plan: a cross-architecture
generalization sweep (grafting benefit tested across architecture families
beyond Qwen) as the primary evidence-strengthening move, with a secondary,
capacity-permitting extension to compare per-layer "champion tuning" profiles
across successful architectures as a possible architectural fingerprint
(explicitly gated on first seeing which models succeed, and the user asked that
Fable be consulted on this before committing pod time — timing left open,
"sooner or later"). A separate, lower-priority requirement was recorded: at the
next paper pass, have Fable research (with tools) how the grafting technique's
applicability is limited by newer architectures — specifically Multi-head Latent
Attention (MLA, used by DeepSeek/Kimi) where no per-head value vector exists to
graft, versus sliding-window designs (Gemma) where the effect may merely
attenuate rather than be blocked. Fable's research (with live web search)
refreshed the target model list to mid-2026 SOTA (Qwen3.6-35B-A3B/27B,
Gemma-4-27B, Mistral-Small-4, GLM-4.7-Flash, OLMo-2-32B, Qwen2.5-32B as a
same-vendor control), verified via HF repo IDs, with MLA models (Kimi K2.x,
native DeepSeek-V3) documented as a principled architectural exclusion rather
than silently skipped.

Fable also caught a design confound before any pod spend: letting each model
generate its own compaction summary would confound "grafting works less" with
"this model writes lower-quality summaries." The fix, adopted per user
suggestion, was to hold the compacted summary fixed across all models, generated
once by Sonnet (chosen deliberately over a frontier model like Fable, to keep
summary quality in a realistic ~30B-tier ballpark without giving the compacted
condition an unrealistically capable summary, and chosen over any one target
model to avoid home-field advantage). Fable reframed the whole sweep as a
mechanistic falsification test — since the effect should in principle travel to
any standard-KV-cache architecture, every outcome (travels / attenuates /
doesn't travel) is a reportable, mechanism-relevant result, not just a
pass/fail.

All of STATE.md, DECISIONS.md, INCIDENTS.md, AGENTS.md, and MASTER-PLAN.md were
brought fully current at user request, since DECISIONS.md in particular had
fallen ~3 hours behind actual decisions made.

## The measurement crisis: broken estimator, not broken effect

While running the 27B effect-bound, a second run intended for 30B silently
re-ran 27B (model-selection flag bug, later fixed and turned into the
unique-output-naming rule). Once corrected, the real 30B effect-bound run
produced a result that appeared to directly contradict the paper's headline:
over a narrow K=12 pre-divergence window, E−B (graft vs. plain compaction) was
significantly negative; a re-run at full window length (K=48) was also
null/negative for E−B, and the graft-vs-placebo advantage also went null at full
window. This was treated as a serious threat to the paper's core claim, and the
cross-architecture sweep was explicitly paused pending resolution ("if our
apparatus is broken and incoherent, there's no point trying to build on top of
it").

Investigation proceeded in stages: first ruling out that the effect-bound probe
used a different/stale summary prompt or that a recent code change affected the
deterministic (temperature=0) generation path (both ruled out); confirming
environment drift (transformers 5.0.x at original paper-run time vs. 5.13.0
currently) as a plausible but ultimately red-herring explanation. Re-running the
original, paper-producing gap-closure script (`gap_closure_cat.py`, the actual
"apparatus of record," distinct from the newer effect-bound probe) live on 30B
gave numbers that themselves diverged substantially from the paper's saved F1
numbers, including the paper's clean stance-null dissociation failing to
reproduce (38% helped in saved data vs. 58% live). A second same-hardware run
was then run to isolate hardware/environment differences from genuine
instability, and it was found to be **byte-identical** to the first live run —
confirming the apparatus is fully deterministic within a fixed environment,
ruling out MoE routing nondeterminism as the cause (this hypothesis, floated
early, was explicitly wrong).

At the user's direction ("you need Fable's help"), Fable was brought in with the
full picture and diagnosed the actual root cause correctly and cheaply (no GPU
needed): the paper's headline gap-closure metric, computed as the **mean of a
per-case ratio** (E−B)/(A−B), is statistically unstable (Cauchy-like) whenever
the denominator (A−B, the compaction damage) is small for some cases — a few
near-zero-denominator cases can swing the aggregate mean dramatically, which is
exactly what produced the paper's dramatic "stance −0.31" figure from what was
actually just two near-zero-denominator probes. Recomputing all three available
runs (paper-saved, live1, live2) on robust statistics (raw E−B, median ratio,
%-helped) showed full agreement across runs and preserved the paper's
qualitative dissociation: referent and sense are positive, stance is ~null. The
conclusion: the underlying grafting effect is real and reproducible; only the
reporting estimator was broken. This was explicitly framed as a
numbers-correction to the paper, not a retraction.

Proper bootstrap confidence intervals were then computed directly from existing
saved data (no GPU) using raw E−B as the standard metric going forward
(mean-of-ratio retired as an estimator). Result: referent is a statistically
significant effect (95% CI [+0.030, +0.218], n=21, excludes zero); sense is
directionally positive but underpowered at current sample size (CI spans zero,
n=22); stance is a genuine null (tight CI around zero, n=24). This reframed the
paper's crisp three-way dissociation as more nuanced than originally presented
(only referent is currently significant on this metric), and exposed the paper's
small per-category sample size (n≈21–24) as the real, fixable limitation rather
than a validity problem — directly motivating the extended
cross-architecture/scale-up plan going forward as the way to gain statistical
power, not just breadth.

Applying the same robust-metric correction retroactively to the earlier K/V
(key-vs-value grafting) analysis — prompted by a direct user question about
whether "keys hurt" findings might rest on the same broken estimator — revealed
that the paper's dramatic claim that key-only grafting "actively hurts" (e.g.,
−0.283 for sense) was itself a ratio-estimator artifact; recomputed on raw E−B,
key grafting is essentially **neutral** (near zero, not harmful), while value
grafting's positive effect (+0.120 referent) is confirmed to survive on the
robust metric. The discussion distinguished this from reopening the keys
question: the qualitative conclusion ("value is the operative axis, keys don't
help") is unchanged and remains closed; only the magnitude/wording of "keys
hurt" vs. "keys are neutral" needed correcting. A broader task was opened to
systematically re-audit every gap-closure number in the paper (F1, the 27B
table, 4B results, K/V) on the robust metric before any further report claims.

## Cross-architecture sweep restart and a new alarm

With the estimator fixed, the cross-architecture sweep resumed on a fresh 400G
pod (the prior 200G community pod had filled its disk and died mid-download,
losing only an in-progress download, no analysis-critical data). At user
request, conversation count was increased from 6 to 12 per model (marginal cost
small since per-model wall-clock is download-dominated, ~20–30 min/model, mostly
download) to gain statistical power, requiring Sonnet-generated fixed summaries
for the additional c07–c12 conversations (generated and committed). The user
approved running the full 7-model sweep with a second pod for parallelism once
the pipeline validated on one model.

The first validation run, Qwen2.5-32B (dense-GQA, same vendor as the
corroborated baseline, using the fixed neutral Sonnet summary), tripped the
smoke gate: the graft's aggregate effect (raw E−B) came back significantly
_negative_ (−0.28, CI excludes zero) rather than positive — a real, non-noisy
result confirmed via CI and full-plant averaging, not a few-item fluke.
Compaction damage was confirmed genuine (A−B = +0.61, CI excludes zero), so the
setup wasn't degenerate. Two competing explanations were raised: (a) a genuine
architecture-specific negative effect on Qwen2.5, or (b) a flaw specific to
grafting onto an _externally supplied_ (Sonnet-written) fixed summary rather
than a model-self-generated one, possibly involving how conversation content is
aligned/matched against the summary text. An isolation test (same model,
self-generated vs. fixed summary) was launched to distinguish the two, alongside
a harness fix so that negative/null results are recorded as legitimate map
datapoints rather than discarded as "unsupported."

The shard ends as the user reacts with alarm to the assistant's passing
description of the fixed-summary alignment mechanism as relying on "spurious
exact-token matching" between Sonnet's paraphrased summary and the original
conversation — the user states this "token matching" approach was explicitly
discarded much earlier in the project in favor of concatenating the full summary
text directly (both appended at the end of context and in a fresh-context
arrangement), and demands the assistant verify by reading the actual
`build_alignment` code before concluding anything, explicitly flagging this as a
potential source of contamination across prior results and asking whether Fable
consultation is needed. The investigation into what `build_alignment` actually
does was in progress and unresolved at the end of the shard.

---
