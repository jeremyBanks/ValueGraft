_This chunk covers the conclusion of the K/V (key-versus-value) grafting
investigation, a rebalancing push to strengthen the paper's primary grafting
evidence over its secondary interpretability material, a series of process and
documentation fixes, and a placebo-controlled re-measurement effort that
uncovered a serious discrepancy between two measurement scripts on the same 30B
model — discrepancy resolution work was in progress as this chunk ends._

**Participants:** User, claude-opus-4-8, and claude-fable-5.

**K/V exploration concluded.** Following up on earlier key-grafting work, both a
uniform-alpha sweep and a per-layer probe on referent-category recovery came
back negative: grafting keys does not help recover referent information, either
applied uniformly across layers or isolated to any single layer (best layer lift
was statistically indistinguishable from noise). Value grafting remains the
effective mechanism. An independent, lower-rigor microtest run by a separate
collaborating agent had suggested keys might help for short, identifier-like
recovery targets (as opposed to semantic-phrase targets), but that signal was
judged (with a dedicated model-assisted review) to be a likely artifact of
selecting the best result among many candidate settings, on a small and noisy
dataset. The team agreed the claim "value is the operative axis" needed to be
scoped rather than stated universally — it holds for semantic-phrase referent
recovery on the tested corpus, while short-identifier recovery remains an open,
untested edge case, deferred to a possible future targeted experiment. The user
directed that no further effort go into hunting for a key-grafting benefit; that
line of inquiry is closed.

**Lens work wound down and honestly bounded.** A free-generation divergence
probe (designed to locate the internal moment where model representations "fork"
toward correct versus incorrect content) was run across 43 cases and found no
clean example of a representation forking cleanly toward the correct answer;
results were classified into pre-registered categories designed to prevent an
ambiguous null result from being reframed as a positive one. The user then
raised that a broader "scan across many tokens" approach might reveal a diffuse
effect that point-based measurements were missing; on review, this was reframed
(with model-assisted input) into a more rigorous experiment: a
placebo-controlled comparison using a shuffled/randomized value-graft as a
baseline, measuring whether the true graft's effect on next-token likelihood is
statistically distinguishable from random perturbation, with a bootstrapped
confidence interval as the reported figure rather than a qualitative impression.
This effect-bound experiment, run on the 27B model, found the aligned graft
clearly outperforms a randomized-value placebo (confidence interval excluding
zero), meaning the effect is content-specific rather than generic
noise-injection — but did not show the graft clearly beating plain compaction
over the measured window. The user also flagged that the report's repeated use
of the word "lens" was overloaded across at least two or three distinct
instruments (a plain logit-based readout, a more sophisticated Jacobian-based
readout) without clearly distinguishing them in many instances; this was
confirmed by direct count and logged as a required fix, alongside a broader
lesson that per-dimension editorial review passes can each miss cross-cutting
issues like inconsistent terminology, arguing for adding a dedicated
terminology-consistency review pass.

**Rebalancing the paper toward its primary claim.** The user explicitly directed
that all further review and revision work proceed without asking for check-ins,
and that the writing team (the main assistant plus a secondary model used for
conceptual and readability review) independently determine next steps to
strengthen the paper's core grafting claim, since the interpretability material
had become disproportionately prominent relative to its modest payoff. The user
clarified this should include both new experiments and additional illustrative
examples mined from already-collected data, not one at the expense of the other.
This produced a plan (documented in the project's plan file) to: pursue a
cross-architecture generalization test, mine more compelling exhibits from
existing data, fix the overloaded terminology, and ensure the secondary reviewer
is explicitly briefed on the intended focus (primary claim first, secondary
material proportionate) rather than reviewing in a direction-agnostic way.

**Cross-architecture generalization effort.** The user asked whether the
grafting technique might be architecture-limited, citing prior informal attempts
with other model families. Investigation concluded the technique depends on
standard per-head, per-position key/value caching; models using
compressed/latent attention representations (where individual per-head values
are not directly stored) cannot use the method as designed and were documented
as a principled architectural exclusion rather than a lazy gap, while
sliding-window attention designs were expected to show an attenuated rather than
blocked effect. A harness was built to run the grafting benefit test across
multiple current-generation ~30B-class models, updated after a live web search
corrected the target model list to current mid-2026 releases in that size class.
A design flaw was caught before spending compute: different models produce
differently-effective compaction summaries, confounding architecture with
summary quality, so the plan shifted to holding one fixed, externally generated
summary (produced by a mid-tier proprietary model rather than either the target
models under test or a frontier model, to avoid unrealistically strong
summaries) constant across all architectures. A published output-naming
convention was also introduced after a mistaken rerun silently overwrote a
completed result — outputs must now be named uniquely per model/date and the
correct model identity verified at launch, not just that a process is running.

**Operational fixes.** Multiple in-session incidents reinforced the standing
practice that a spawned or "running" process must be verified as doing genuine,
correctly-configured work (right model loaded, output advancing) before being
trusted, since several jobs in this session silently failed at launch or ran
with the wrong model while appearing superficially healthy. Monitor notification
frequency was adjusted per user request to avoid frequent interruptions during
long-running jobs, using increasing backoff between updates and reporting only
on completion, error, or meaningful state change. A significant repository
cleanup relocated dozens of one-time planning, report, and findings documents
into a dated archive folder using a consistent naming convention based on each
file's original creation date, leaving only a small set of essential top-level
files (agent-orientation, decisions, findings, incidents, plan, state, and the
report/readme) at the top level; an entire subdirectory of now-superseded
exploratory work from a collaborating agent was deleted after its markdown
content was preserved in the archive. All standing tracking documents (state,
decisions, incidents, findings, plan) were brought back up to date after falling
behind during a period of rapid experimentation.

**Publication framing.** Following informal outside feedback relayed by the
user, the report's public-facing copy was revised to lead with a short, humble
framing paragraph — stating the result plainly, noting an inability to find
directly relevant prior art or to test at meaningful scale, and inviting outside
input — placed before the title and separated by a horizontal rule, mirroring
how such results are typically shared in relevant online research communities.
The report's title was also revised at the user's request (selecting a
secondary-reviewer-proposed alternative) to better foreground the grafting
contribution rather than the problem statement, replacing an older title judged
stale.

**Unresolved discrepancy at close of chunk.** While re-running the
placebo-controlled effect-bound experiment on the 30B model to strengthen the
primary claim, results were essentially negative to null — a clear disagreement
with the paper's previously reported dissociation figures (which show strong
positive recovery on this same model and category). Investigating the mismatch,
it was confirmed the two measurement scripts use identical model, identical
alpha value, and identical case set, yet produce opposite conclusions — strongly
suggesting an implementation bug in the newer effect-bound script rather than a
failure of the underlying grafting effect, since the original, paper-producing
measurement script was confirmed (via its previously saved output) to reproduce
the paper's published figures exactly. The user endorsed treating this as a
blocking prerequisite: no further building (including the cross-architecture
sweep or any alpha-tuning protocol) should proceed until the measurement
apparatus itself is confirmed trustworthy through a live re-run of the original,
paper-producing script. That live confirmation run was in progress as this chunk
ends, with the plan — once confirmed — to retire or fix the newer probe, revisit
whether its content-specificity/placebo finding survives under a corrected
implementation, and only then proceed to a systematic alpha-strength sweep
(using a neutral middle value as baseline, then per-model tuned "champion"
configurations) as the basis for the cross-architecture comparison, with the
tuned configurations across architectures also intended as a point of comparison
reflecting each model's internal structure.
