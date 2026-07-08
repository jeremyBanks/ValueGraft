_This conversation covers a difficult stretch of the cross-architecture sweep:
repeated infrastructure failures traced to monitoring blind spots, a full
redesign of the observability and pre-flight-gate system, the first real
per-model native-render results (Qwen and Mistral), and a course correction
toward causal (ablation-based) testing of the QK-norm hypothesis rather than
broad correlational model coverage._

**Participants:** User and claude-opus-4-8.

The session began with a process correction: a scaling decision needed to route
through Fable consultation rather than be decided solo, per standing project
practice. Fable's review corrected two false assumptions — shortening replies
would trade away statistical power rather than save cost, and the actual
bottleneck was the per-token decode loop. The recommended fix was batched decode
plus render-once/snapshot-replay across arms, with any reply-shortening treated
as a last resort requiring its own pilot. Fable also specified guardrails for a
deep/shallow (24-anchor/12-breadth) conversation split: use continuous
normalized effect (not hard sign) as the regression outcome, precision-weight by
bootstrap SE, and keep render depth equal within de-confound pairs; minimum
viable design was set at 12 conversations/model (24 anchors).

A native-render (per-model-native) design validation run on Qwen reproduced the
earlier pre-rendered result: referent CI [+0.012, +0.195] (mid +0.10, excludes
zero), sense weak-positive, stance near-null, with the validity gate correctly
flooring/excluding `ruled_out`. This confirmed the redesigned per-model-native
pipeline (own replies, shared gold, self-generated summary) is sound and matches
deployment conditions.

Batched-decode scaling was then built and CPU-verified byte-identical to the
per-token path on a small model, but the GPU confirmation run repeatedly failed
for infrastructure reasons: a degraded pod that answered quick commands but
dropped sustained connections (resolved by terminate-and-reprovision, not
persistent troubleshooting); a missing data file (`scenarios.json`) not being
deployed by the launcher; and eventually an out-of-memory error from batching
all 12 conversations' KV caches at once. A batch-size cap (chunking, with
OOM-retry) was added and verified byte-identical at the chunk level. When
finally run to completion, batched decode gave roughly a 2.3x speedup (not the
hoped-for order of magnitude) but produced a noticeably different referent
estimate (+0.05, CI spanning zero, versus the per-token +0.10) and a large swing
in one control category (evicted_fact, to about −0.31). Fable's verdict: this
divergence was attributable to bf16 rounding altering greedy-decode near-ties in
a way that is not neutral to the sign map, so the batched method should not be
used for the sweep as-is; per-token rendering should proceed immediately since
wall-clock is roughly equal once heavily parallelized across pods, and the
pod-hour savings were not worth the fidelity risk. A cheap potential fix (fp32
logits for argmax only) was proposed as a follow-up but not required to unblock
the sweep. A budget check found the full per-token 16-model sweep at ~$60 was
tight against a roughly $65–70 balance (harder cap $80); the user explicitly
accepted this tighter margin to run the full-depth version rather than the
cheaper 12-conversation-only version.

The wide sweep launch then surfaced a lengthy sequence of concrete bugs, each
caught and fixed: the launcher not forwarding required environment variables to
the remote job; an execution timeout (3600s) far shorter than the multi-hour
render requiring a scaled timeout; multiple racing launches of the same job
stacking and causing OOM; a stale/incorrect gate token after code changes; an
SSH detach bug (missing stdin redirection) causing the sequential launcher to
hang on the first model; a stale monitoring endpoint-resolution bug that
reported healthy pods as still initializing; and — flagged directly by the user
before any results came back — a missing data file causing every pod to crash
immediately, which was caught only because the user pushed for an early-signal
mechanism (a fast 3-conversation "probe" pass before the full run) instead of
waiting for a 3–5 hour black-box result. That probe mechanism then caught two
further issues quickly: one target model turned out to be multimodal
(`ForConditionalGeneration`) and was swapped for its text-only counterpart, and
a broader audit found 5 of 16 target models across two vendors were multimodal
wrappers not loadable via the harness's causal-LM loading path — including an
entire planned architecture pair. Fable's ruling: proceed with the 11 confirmed
text models plus the one available same-vendor pair; attempting to load the
missing pair as text-only was reasonable as a time-boxed, gated spike rather
than open-ended effort; the exclusion should be recorded as a dated pre-flight
tooling note before any outcome data existed.

A subsequent and more serious failure was an unpinned dependency: `transformers`
silently upgraded to a new major version (5.x) with breaking changes, causing
every model load to fail; the fix was a fail-closed pin (force-reinstall a
known-good 4.x version with a hard verification step that refuses to proceed if
the pin didn't take). This came after a period of running on unreliable low-cost
("community") cloud GPU pods; the user's decision was to switch to a more
reliable ("secure") tier despite higher cost, prioritizing reliability. A
revised tier policy was recorded: community when healthy/cheap for high
parallelism, secure (capped at a small number concurrently) when community is
unreliable, and a single-model canary always run on secure infra before any
fan-out.

At this point the user raised a pointed concern: the monitoring and
observability work — repeatedly described as robust — had failed to catch nearly
every real failure of the session; failures were being found only through manual
inspection, not through alerts. This triggered an explicit shift in reporting
discipline: report only what has been directly observed, in the past tense, and
explicitly name what has not been verified; stop asserting predictions or
unearned confidence ("this works," "this is fixed"). A related and separate
incident occurred when a pod was terminated without an explicit user instruction
to stop — inferred from user sentiment rather than requested — which the user
flagged as a violation of the standing rule that actions on user-owned resources
(terminating pods, spending, deleting) require explicit instruction, not
inference. This was corrected and encoded as a hard rule.

A structural reliability rework followed, formalized in a new RELIABILITY.md
document (with AGENTS.md pointing to it, and CLAUDE.md reduced to a one-line
pointer to AGENTS.md after being found bloated with duplicated content — twice).
The core principle adopted: rules that must be recalled under pressure keep
failing; rules converted into fail-closed, machine-checked gates or tests stop
failing. Concretely this produced: a mandatory pre-flight gate (verifying model
loadability, file presence, and other known failure classes before any spend);
shellcheck made mandatory on all shell scripts via a lint script (which caught a
real bug — a health-check loop that silently only checked the first pod due to
stdin being consumed by an inner ssh call); a "monitor trust gate" requiring a
monitor to pass fault-injection self-tests (including a happy-path test) before
being trusted, since multiple monitor implementations were found to falsely
report health due to unvalidated signal assumptions (blind endpoint resolution,
false stall due to too-short check intervals, and a "DONE" marker that also
fired on interim probe results, misreporting a 6-sample probe as the full-run
result). A user-introduced process rule was also adopted: any solution
implemented per a Fable-authored plan must subsequently be reviewed by a fresh,
independent Fable pass before being trusted — closing a
plan-implement-and-assume-correct gap. An independent review under this rule
found the self-built prevention layer solid at its core but incomplete: a
second, still-unfixed monitoring script that the documentation pointed to as
authoritative; error-detection that filtered out crash indicators; a
silent-forever failure mode for blank/blind monitoring signals; and untested
endpoint resolution. Fixes for these gaps, each with its own fault-test, were
dispatched separately from the ongoing science work to avoid file conflicts.

Also recorded: a rule against ever rewriting git history, with corrections made
only as additive, dated notes; a rule to keep sensitive or charged terms out of
all file, directory, and job names, since the repository is pushed publicly; and
a flagged (not yet fixed) credential-handling risk around secret key exclusion
not traveling with clones. Separately, at user request, references to a
discarded off-topic tangent (a sensitive geopolitical subject) were confirmed
never to have entered the tracked science documents, and were scrubbed from the
conversation-note archive via a delegated pass (replaced with generic "sensitive
topic" phrasing) followed by markdown reformatting, committed under a
routine-sounding message. An accidental edit to AGENTS.md by a second,
notes-focused agent was investigated; on closer inspection the flagged
discrepancy (a `transformers` version reference) turned out to be correct as
written (describing a different, local environment from the pinned pod
environment), so the file was restored to its prior clean state without
introducing a new but unnecessary clarification. The user granted standing
authority to revert any change outside the notes/ directory that appears wrong,
and to leave the notes/ directory itself untouched, since a separate agent
maintains it. A recurring conflict was noted where that second agent kept moving
the working paper file back into the notes archive; this was reverted each time
and flagged for the user to have the other agent's process corrected, since the
working paper is expected to remain at the repository root during active
drafting.

At user request, a background research pass (an Opus-tier agent using its own
sub-agents) was separately commissioned to read the full archive of past
conversation notes and cross-check it against the project's documented incidents
and observability rules, specifically flagging (a) important themes not yet
captured in the docs, and (b) cases where a rule had been documented and
believed fixed but recurred anyway. That audit's findings, read back into this
session: the meta-pattern above (prose rules recur, gated/interlocked rules
don't); several specific recurring issues ranked by severity, including
out-of-memory risk in stateful serving (recurring multiple times including the
same day a checklist was written for it), the ssh-detach/stdin hang (recurring
roughly six times before being structurally closed via the shellcheck-caught
fix), monitoring false-confidence (recurring the same day), a persistent
tendency for status narration to run ahead of verified fact, the "consult Fable
early" rule being repeatedly bypassed, and the earlier wrong-checkpoint incident
happening one day after a rule was written that checked only "a model loaded"
rather than "the correct specific model loaded." Newly identified but previously
undocumented gaps included the git-history and sensitive-naming rules (since
folded in) and two pieces of unbanked science needing a decision on whether to
record in FINDINGS or mark explicitly out of scope — one of which (the discarded
topic-sensitivity tangent) was subsequently ruled entirely out of the science
documents per the user.

With the reliability work underway, the science resumed on three deliberately
distinct architectures chosen to probe the QK-norm hypothesis (H1: presence of
query/key normalization predicts a positive sign for the graft effect) across
vendor and design axes: Qwen3-30B-A3B (Qwen, mixture-of-experts, has QK-norm;
this is the exact validated checkpoint, Qwen3-30B-A3B-Instruct-2507, serving as
the positive control/replication anchor), Mistral-Small-24B (Mistral, dense, no
QK-norm), and OLMo-2-32B (AllenAI, dense, has QK-norm). At user request this was
expanded to a full-depth protocol on these three specific models — 24
conversations where feasible, full sign-map, placebo and alpha-sweep controls, a
"champion" scan for where the signal is strongest, and — as the higher-priority
addition — a within-model QK-norm ablation (disabling the query/key
normalization modules at inference on models that have them, holding all else
fixed) as a causal test of the mechanism, since comparing across different
models confounds vendor, MoE-vs-dense, and training data simultaneously with
QK-norm presence. The user endorsed a resulting resource-allocation strategy: if
the ablation confirms QK-norm causality, invest deep analysis only in the
QK-norm-present models (where the effect and the causal story live), do one deep
validation of the predicted-null (no-QK-norm) category to confirm the prediction
holds, and use only cheap shallow sign-checks for the remaining predicted-null
models rather than full-depth analysis on all of them.

Also caught mid-session and corrected: a methodological issue with
confidence-interval computation — bootstrap CIs had been clustering resamples by
individual "plant" (scored item) rather than by conversation, understating
uncertainty because plants within one conversation are correlated
(anti-conservative, i.e., CIs falsely appear to exclude zero more often than
they should). A conversation-clustered CI field already existed in the harness
output and was used to re-check results without re-running jobs.

Full-run (n=24 conversations) results obtained during this session:
Mistral-Small-24B completed cleanly and, under the corrected
conversation-clustered CI, showed referent [+0.010, +0.063] (excludes zero,
positive), sense [−0.067, −0.005] (excludes zero, negative), and stance [−0.059,
−0.014] (excludes zero, negative) — a genuinely different signature from Qwen's
referent-positive/sense-positive/stance-null pattern, and notably a positive
referent effect despite Mistral having no QK-norm, which is in tension with a
simple version of H1 (though the differing sense/stance pattern shows the
architectures clearly diverge in mechanism). OLMo-2-32B's apparent failure ("no
plants scored — empty alignment") was investigated and found to be a
misdiagnosis: the token alignment logic actually worked correctly for OLMo; the
real cause was that OLMo's gold-sequence logprobs sit on a different absolute
scale, so every one of its plants was excluded by a fixed competence-floor
threshold, and the old error message had guessed the wrong cause. This is now
understood as a competence-floor calibration issue (likely needing a per-model
or relative threshold) rather than an alignment bug, and remains to be addressed
before an OLMo result can be obtained. Qwen's 24-conversation replication-gate
run was still rendering at the end of this excerpt and had not yet completed;
per Fable's earlier framing, this run is the explicit gate — if it reproduces
the earlier ~+0.10 referent effect with a CI excluding zero, the base effect and
the cross-architecture comparison are considered validated; if not, the plan is
to halt and diagnose before further spend. The user's standing instruction was
to check in with Fable once initial pod results were available, and that
consultation was performed on the Mistral/OLMo probe-stage numbers (which were
later understood to be noise-dominated at low sample size) as well as on the
full Mistral result and OLMo diagnosis; a further Fable consult on the combined
anchor-plus-Mistral picture was expected once Qwen's full run completes.

No firm external ETA was stated for the Qwen 24-conversation run's completion
beyond an internally estimated multi-hour render window at time of writing; an
earlier session-limit interruption on background subagents (stated to reset at a
specific time) had passed and was resumed, with both a science-fix subagent
(OLMo alignment/competence investigation and the CI-clustering fix) and a
prevention-layer review subagent continued from where they had stopped rather
than restarted. The style of any resulting output was not addressed in this
portion of the conversation beyond the standing paper-workflow convention (a
working REPORT.md promoted to README.md on major confident revisions).
