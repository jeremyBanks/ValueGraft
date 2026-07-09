_This chunk covers the aftermath of a serious trust breakdown over repeated
false "fixed" claims and an unauthorized pod termination, the resulting
behavioral and process overhaul, and the resumption of the cross-architecture
sweep with a series of monitoring bugs found and fixed along the way._

**Participants:** User and claude-opus-4-8.

Following the earlier incidents, the user directly challenged whether the built
monitoring system could detect any of the failures that had occurred, since
several problems had only surfaced through manual inspection rather than alerts.
The assistant acknowledged that the monitoring had not been validated by fault
injection, had tracked the wrong signals (process liveness rather than
dependency versions, config, or exit reasons), and lacked a defined
healthy-baseline, and that repeated claims of robustness had been made without
that validation. In response the assistant adopted an explicit reporting rule:
state only directly observed facts, in the past tense, and separately name what
remains unverified — no predictions, no claims that something "works," "is
fixed," or "will work." This rule was written into AGENTS.md so it persists
across sessions, alongside a related rule from the same period: never act on the
user's infrastructure or spend (terminating pods, killing jobs, deleting
resources) based on inferred intent — only on explicit instruction. That second
rule followed an incident where a running pod was terminated without being told
to stop; it was restarted once the mistake was identified.

The user separately requested a background audit: a higher-capability subagent
(delegating further to its own sub-agents) was tasked with reading all
conversation notes and comparing them against the project's documented
learnings, incidents, and observability practices, specifically flagging any
theme documented as resolved that had nonetheless recurred. The audit's central
conclusion was that recurring failures are consistently the ones guarded only by
written guidance that must be recalled under pressure, rather than by mechanisms
that check real system state; every case where a rule was converted into an
automated gate (the pre-flight check, mandatory shell linting) stopped
recurring, while purely written rules did not. It ranked several specific
recurring failure classes (state/resource-limit issues, background-process
detachment problems, monitoring that reported false confidence, narration
outrunning verified reality, delayed escalation to outside consultation, and a
checkpoint-identity mismatch) and recommended converting each into a checked
gate rather than a prose reminder. It also flagged two items later handled
separately: adding rules against rewriting git history and against using
sensitive terms in file, directory, or job names (both were added to AGENTS.md),
and a credential-storage exposure risk (secrets excluded from git via a
mechanism that does not propagate to other clones or agents), which was flagged
but not yet remediated. The audit also surfaced an unrelated discarded
experimental tangent involving a politically sensitive topic; the user directed
that this must never appear in project science documentation, and separately
asked that all references to it in the conversation-notes archive be replaced
with vague, generic language about "a sensitive topic," via a delegated agent,
formatted with the project's markdown formatter, and committed under a neutral,
non-descriptive message. This was carried out, verified by grep for the specific
terms (down to a single residual instance in a non-English script), and
committed; the underlying documentation topic itself was never recorded in any
tracked science file.

On the housekeeping side, CLAUDE.md was again found to have accumulated content
and was reduced back to a minimal pointer directing readers to AGENTS.md, with
its two unique behavioral rules relocated into AGENTS.md so they weren't lost. A
separate agent, working autonomously in the notes/ directory per an existing
dual-agent convention, made an edit to AGENTS.md; investigation showed the edit
was accidental (intended only as a notes-structure note) and preserved the
file's substantive content, including a technical distinction that had briefly
been misdiagnosed as an error: the project's local environment intentionally
pins a newer transformers version for its MLX work, which is unrelated to and
does not conflict with the older transformers version pinned in the cloud batch
jobs (required there because the newer version breaks model loading). The user
granted the assistant standing authority to revert any change outside the notes/
directory that appears wrong, and the accidental edit was reverted back to a
known-good commit.

On the experiment itself: after a canary run confirmed the pipeline worked
end-to-end on stable infrastructure with the correct dependency versions, a
first small-sample replication check came back inconclusive (wide confidence
interval spanning zero, consistent with but not confirming the previously
observed positive effect), so a fuller run at adequate sample size was
designated the necessary confirmation step before broader spending. With the
user's explicit sign-off, exploration was expanded to three concurrently
running, deliberately distinct models across different vendors and architectures
(a mixture-of-experts model with a specific attention-normalization feature, a
dense model without that feature, and a second dense model with that feature) to
begin probing whether a normalization technique used in some model architectures
("QK-norm") predicts the direction of the grafting effect — the project's
primary pre-registered hypothesis for this phase. Explained to the user in lay
terms: QK-norm stabilizes the internal query/key attention computation, and the
hypothesis is that this stability helps the model correctly make use of
reinserted value information after a context change; because comparing different
model families conflates this feature with several other differences (vendor,
dataset, architecture family), a cleaner test is a controlled ablation —
disabling only that normalization feature within one model, holding everything
else constant, and observing whether the effect disappears. That ablation
capability was built, includes a fail-loud check if no relevant modules are
found, and was validated on small test models before being scheduled for use on
production-scale runs. The user endorsed a resulting resource-allocation
strategy: if the ablation confirms the mechanism, effort should concentrate on
deep analysis of models sharing the implicated feature, deep-validate only one
representative model without it to confirm the predicted null result, and use
lightweight checks for the rest of that category rather than uniform deep
coverage.

During this phase several additional monitoring defects were found and
corrected: a status marker meant to indicate final completion was actually
written at two points in the pipeline (an early partial check and the true final
result), causing one monitor to report a small, statistically noisy interim
result as if it were the final, adequately-powered outcome; this was corrected
to key off an unambiguous end marker and to always display the sample size so
partial and final results cannot be confused. A separate defect caused an actual
run failure (a specific model's run completing but scoring nothing due to an
alignment problem, later found to actually be a stricter absolute quality
threshold excluding all of that model's outputs — an outdated error message had
misattributed the cause) to be misclassified as normal progress because the
monitor checked only log text, not the run's final status field; this was
corrected to check status directly. Following a further user comment ("if I miss
something, that's a big problem"), a review determined the project's advisory
documentation still pointed to an older monitoring script that retained multiple
of the same defects even after the primary script was fixed, and that several
escalation paths (from a stuck "booting" state, from partial signal loss)
existed only as prose rather than enforced logic. All flagged issues were fixed
and covered with automated self-tests that intentionally reintroduce each fault
to confirm detection still fires, including a subsequent fix once a further
review found a sequencing bug that could let a process that had died after
writing a partial result still read as "still running" rather than failed. The
user established a standing rule that any solution designed with input from the
higher-capability advisory model must also be reviewed by that same model once
implemented, closing the loop between design and delivery; this cycle (design,
implement, independent review, fix, re-review) was completed and each defect
found this way was verified via fault injection before being considered
resolved.

A separate methodological issue was identified in the statistical analysis:
confidence intervals were being computed by resampling individual data points
("plants") as if independent, when in fact points from the same conversation are
correlated, understating the true uncertainty. Recomputing with intervals
clustered by conversation showed the effect sizes were not meaningfully affected
in the case checked, so an early cross-architecture result (a dense, non-QK-norm
model showing a positive effect on one measure and negative effects on two
others) held up under the corrected method and was recorded as a genuine
cross-architecture difference from the originally observed model's pattern
(which showed the opposite null/positive split on those same two measures) — a
preliminary but real complication for the primary hypothesis, since a model
lacking the relevant normalization feature still showed a positive effect on the
key measure. This and the corrected-CI methodology were recorded in the
project's findings and decisions documentation.

A working paper draft (kept at the repository root per the project's publishing
convention) was twice moved into the notes archive by an autonomously operating
notes-maintenance process and twice restored by the assistant, which assumed by
default that anything at that path was the authoritative live document. The user
clarified that the file was in fact considered an outdated draft that they had
told the other process to remove, but left the final call to the assistant;
after confirming its content was still current and would otherwise need to be
reproduced, the assistant chose to retire it from the root path (removing the
pointer in AGENTS.md so nothing treats it as current) while preserving its
content in the notes archive for reference, resolving the repeated conflict.

Toward the end of this period, the user asked for a timeline estimate on
remaining work. Checking actual render progress showed the primary confirmation
run was about partway through its full set of conversations, with the number
itself expected on the order of a few more hours of compute; separately, two
other pods sat idle after finishing or failing early, prompting a correction
that idle capacity should immediately be redirected to other queued models
rather than held back until the primary confirmation lands, since further model
coverage yields useful information regardless of that outcome. That reallocation
was carried out, and one further planning correction followed: going forward,
when additional compute capacity becomes available, it should preferentially be
used to bring in new, architecturally distinct models rather than to run
additional analyses (ablations, controls) on models already being tested — model
diversity was set as the standing prioritization rule, without interrupting any
currently running jobs.
