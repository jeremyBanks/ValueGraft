_This conversation covers a long autonomous research push on the ValueGraft
compaction-recovery project: shipping and then substantially revising the
synthesis paper (adding real transcript exhibits, a related-work section, and
honest scoping of secondary findings), closing out the K/V-grafting exploration
with a clean negative, running a placebo-controlled "effect-bound" experiment on
the core claim that ultimately surfaced a serious internal inconsistency in the
measurement apparatus, and setting up (but pausing) a cross-architecture
generalization sweep._

**Participants:** User, claude-opus-4-8, and claude-fable-5.

Participants: Jeremy Banks (user) and Claude, alternating between Opus 4.8 and
Fable 5 (the session's provider-side model-selection filter repeatedly reset the
main loop to Opus even when Fable was selected).

Early in this chunk, the user caught an execution stall on the tau-bench
integration attempt and, separately, called the first version of the synthesis
report too vague, lacking concrete examples of the model's internal behavior
despite the project having rich transcript data available. In response, the
report was rewritten to foreground verbatim transcript exhibits (a
fabricated-consultant confabulation, a recovered-vs-invented editing-style pair,
a "stance" case where the model avoids a forbidden option without being able to
cite why), given a related-work/citations section pressure-tested against prior
art, and put through a multi-critic revision pipeline (accuracy,
over-justification, through-line, accessibility) followed by a two-writer prose
bake-off between Fable and Opus. The user established a standing operating rule
during this stretch: model selection should be by fitness for the task, not cost
(Opus has ample headroom; Sonnet is for deliberate second-perspective diversity;
Fable, though expensive, must be used — via explicit subagent invocation, since
the main loop keeps flipping to Opus — both for final readability passes and for
proactive high-level conceptual gut-checks on claims and framing before they're
locked in). This Fable-gut-check practice caught a real overclaim later in the
session (a proposed "scale/architecture-dependent" explanation for why a
referent-recovery result didn't replicate on a second model was rejected by the
user as unsupported, since the two models differ only modestly in scale and are
confounded on multiple axes at n=2; the eventual write-up instead reported a
single cross-model non-replication with cause left open).

After the paper shipped, the user directed a second phase exploring the K/V
(key/value) grafting space — separate per-layer, potentially per-head tuning of
α_K and α_V — motivated by the hypothesis that keys carry RoPE-based
positional/addressing information that might recover specific evicted decisions
("referent") where value-only grafting had floored. Key-grafting with exact RoPE
re-rotation was implemented and validated to floating-point precision. A coarse
sweep and then a per-layer probe both returned a clean negative: key-grafting
does not help, and in fact hurts, across all tested layers and categories; value
remains the operative axis. An independent microtest run by another agent in the
repo hinted keys might help for short, identifier-style targets (as opposed to
semantic-phrase targets), but that signal was judged statistically weak (small
cells, best-of-many-policy selection, tiny model) via a Fable-adjudicated
review; the claim was scoped accordingly rather than treated as a live lead, and
further key-digging was explicitly deprioritized by the user in favor of other
avenues. A short-identifier-recovery follow-up was noted as a possible cheap
"question-closer" but was also deprioritized.

The user then redirected effort toward strengthening the primary contribution
(the grafting technique itself), judging that the paper had become
disproportionately focused on the secondary interpretability-lens material
relative to its actual (modest) payoff — a concern reinforced when the user
separately flagged that the word "lens" was overloaded in the text, referring
ambiguously to at least two or three distinct instruments (plain logit lens,
tuned lens, J-lens) across ~76 uses. This was confirmed by inspection and logged
as a requirement for the next revision pass, along with a new standing review
dimension ("terminology consistency") to catch this class of issue, since the
existing single-dimension critics had all missed it. The user also asked that
Fable, when doing conceptual/editing work going forward, be explicitly briefed
on the intended focus (grafting as primary, lens work proportionate to its
secondary role) rather than working direction-agnostically, and clarified that
pursuing more experiments and mining more examples from existing data are
complementary strategies, not alternatives.

A "free-generation divergence" lens experiment (teacher-forcing was suspected of
pinning trajectories and suppressing the very effect being measured;
free-generation was proposed to find the point where behavior forks) was
designed with Fable's input, which caught a decoder-amplification artifact risk
and an unfalsifiability risk in the original design and prescribed fixes (report
fork margins and decoding-robustness across temperatures; pre-register a genuine
disconfirming bucket). The corrected N=43 probe ran and returned a clean
negative (zero cases showed a clear fork toward the correct concept), which was
interpreted as strengthening rather than undermining the paper's already-honest
"lens is a low-resolution corroborator" framing. A follow-on idea from the user
(scan the lens across a broader token range rather than at single points) was
reframed by Fable into a more defensible design: a placebo-controlled
"effect-bound" experiment comparing the real graft against a random-value graft
with bootstrapped confidence intervals, to produce a spin-proof quantitative
bound rather than another attempt at a vivid but potentially cherry-picked
exhibit.

Running that effect-bound experiment surfaced this chunk's most consequential
open problem. On 27B, the graft beat a random-value placebo decisively (CI
excludes zero) but showed no clear advantage over plain compaction on the
pre-divergence window. On 30B, results were inconsistent across window lengths,
and when the same gap-closure metric used to produce the paper's original
headline numbers (referent 81% helped, sense 64% helped) was recomputed using a
different ("effect-bound") implementation on the identical case set, it returned
close to the opposite result (referent 48%, sense 27% helped). Careful diagnosis
ruled out differences in alpha value and case selection — both scripts use
identical inputs — leaving unresolved disagreement between two implementations
that should agree. A live re-run of the original, paper-producing code
(`gap_closure_cat.py`) was launched to confirm whether it still reproduces the
published 81%/64% numbers; this is the explicit gating step before any further
work proceeds, per the user's directive that no work should build on top of a
measurement apparatus that may be broken or internally incoherent. If the
original code reproduces its published numbers live, the effect-bound probe is
the likely buggy outlier and will be retired or fixed, with the
placebo-comparison idea preserved but ported into the trusted codebase; if it
does not reproduce, the paper's core F1 claim needs re-examination before any
further extension (cross-architecture sweep,
champion-tuning-as-architectural-fingerprint comparison, etc.) is pursued.

Other operational threads from this chunk: a repository cleanup moved ~34
obsolete top-level markdown files (plus an entire now-subsumed
`referent_recovery_microtest/` directory) into a dated `docs/` archive
convention, leaving only nine essential agent-facing files at top level (AGENTS,
DECISIONS, FINDINGS, INCIDENTS, MASTER-PLAN, README, REPORT, STATE,
writeup-guidelines); REPORT.md was promoted to README.md as the public landing
page, with a short forum-post-style framing paragraph (written by Fable, at the
user's request) placed before the title, and a new title chosen by the user from
Fable's suggestions. A cross-architecture generalization sweep (testing whether
value-grafting transfers to other current ~30B-class open models — Qwen2.5,
Gemma-4, Mistral Small, GLM-4.7-Flash, OLMo-2 — with DeepSeek/Kimi-style MLA
architectures excluded as mechanistically incompatible with the graft's KV-cache
assumptions) was designed with Fable's input, including a fix for a
summary-generation confound (using a fixed, neutral Sonnet-written summary
across all models rather than each model's own summary) and a protocol for fair
per-model tuning (baseline pass at neutral α before any per-layer champion
tuning). This sweep, along with the user's suggestion to later compare
champion-tuning profiles across architectures as a possible "architectural
fingerprint," is queued but currently held pending resolution of the
apparatus-consistency problem above. Several recurring operational lessons were
logged as incidents/rules during this stretch: monitors should stay silent
during routine progress and only alert on completion/error/state-change, backing
off from a 2-minute to a 24-minute cadence (implemented at the user's explicit
request); a launched job must be verified to be doing genuine,
correctly-configured work (right model loaded, logs advancing) rather than
trusted from a "launched" message, since a crashed or misconfigured job can look
identical to a healthy one from process-liveness alone; output files must use
unique, self-describing names to prevent one run from silently overwriting
another's results; and community-cloud pods carry real environment risk (missing
rsync, torch/CUDA version mismatches causing silent CPU fallback) that cost
significant wall-clock across several incidents this session, each of which
produced a hardening rule against recurrence.
