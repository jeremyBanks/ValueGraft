_This chunk covers the conclusion of the K/V key-grafting investigation, a
cross-check of that conclusion against a second agent's independent findings, an
infrastructure failure in verifying job execution, and the launch of a follow-on
lens-based investigation._

**Participants:** User and claude-opus-4-8.

The K/V per-layer probe completed with a clean negative result: no single layer
(of 12 sampled) rescues key-grafting on the referent category, with the best
layer showing only a +0.011 lift (within noise, 10/21 plants positive). Combined
with the earlier coarse sweep, this closed Phase 2 of the research program:
value-grafting is the effective mechanism for the tested regime, and
key-grafting does not help, whether applied uniformly or at any individual
layer. This result was recorded to the findings document and the phase was
marked complete.

At the user's request, this conclusion was checked against a second,
independently-run line of work by another agent (documented in a file the user
pointed to, previously untracked by this thread) exploring a related "key/value"
recovery microtest. That agent's screening runs surfaced a potential exception:
for short, identifier-like recovery targets (e.g., transforming a label into a
short code token), key-grafting sometimes outperformed value-only grafting,
whereas for longer semantic-phrase targets, value-only dominated — consistent
with this thread's own conclusion on the overlapping regime. The user asked
whether the other agent's positive finding was statistically well-supported. A
conceptual review model was asked to weigh in and judged the identifier-recovery
signal statistically weak (very small sample sizes, an underpowered exploratory
model, and a best-of-many-policies selection procedure prone to appearing
significant by chance), so it should not overturn the negative conclusion for
semantic-phrase recovery. However, the review model also judged that the
negative conclusion, as previously stated, was overclaimed as universal, since
it had only been tested for semantic-phrase-style targets. The conclusion was
revised: value-grafting is the effective axis specifically for semantic referent
recovery; whether key-grafting helps for short-identifier-style recovery remains
an open, mechanistically plausible question that was not tested at adequate
rigor. A dedicated follow-up experiment (a single pre-registered run at the
primary model scale, sweeping the full range of grafting policies rather than
reporting only the best-performing one, to avoid the same selection bias) was
queued as a next-phase task, to run after the current paper is finished and
shipped — not immediately, and not as an open-ended pursuit. Other key-grafting
directions (per-head tuning, broader alpha/layer search) were judged not worth
pursuing further, since the data already show keys hurting uniformly and at
every sampled layer for the tested regime.

Work then turned to a lens-based follow-on experiment intended to give the paper
a more visible illustration of the (acknowledged modest) grafting effect: rather
than measuring the internal readout at a fixed token under forced continuation
(which pins the trajectory and suppresses divergence), the new design
free-generates from the compared conditions and locates the point where their
outputs first diverge, then reads the lens at that fork. This design had already
been reviewed by a secondary model for conceptual soundness, which flagged and
the team corrected several risks: that a discrete-decoding fork can look
dramatic from a near-tied internal state, unrelated to whether the underlying
pull is robust (fix: report fork margins and check robustness across sampling
temperatures); that a single vivid example can create a misleading impression of
the true (modest) effect size if shown without the full distribution of cases
(fix: present all sampled cases, with any single illustrative case shown as one
point among many, not a standalone example); and that describing a null result
as merely "subtle" risks being unfalsifiable (fix: pre-register a third outcome
bucket where no divergence, or divergence in the wrong direction, counts
explicitly against the hypothesis). The corrected probe (43 cases combining two
task categories) was built and committed with these safeguards.

Launching this probe on a pod (a different, larger model than used for the main
behavioral results, matching the model for which the lens weights exist)
surfaced an operational failure: a background job was reported and believed to
be running for an extended period, but had actually aborted almost immediately
due to a working-directory mismatch between where code was deployed and where
the job script expected it — meaning the pod ran idle (incurring cost) while
producing no results, and this was not detected until the user asked for a
status check. Two further startup issues (a missing required package, and a
library version incompatibility) were found and fixed once the failure was
investigated. This was treated as a durable operational lesson, distinct from
the earlier established practice of not trusting a job simply because it was
launched or reported alive: the additional requirement is to confirm a job has
reached actual computation (e.g., visible log advancement past setup, or
resource usage matching real work) before treating it as running, and background
monitors should escalate rather than stay silent if expected progress stalls.
This rule was applied immediately afterward when the restarted job's early
indicators looked ambiguous (a stalled-looking download progress display);
further inspection confirmed the download was in fact proceeding normally by
checking underlying cache growth and network activity directly, rather than
trusting either the job's own progress display or assuming failure. Separately,
per an earlier explicit instruction, monitors are now expected to run with
increasing backoff between checks (starting around two minutes, growing to
roughly twenty-four minutes) and to stay quiet during normal progress, only
interrupting on completion, error, or stall.

The user also proposed a further, lower-priority follow-on for the lens work:
instead of reading the lens at a single located point, scan a lightweight
version of the readout token-by-token across a longer stretch of generated
continuation, to check whether the grafting effect is better described as a
diffuse trend across many tokens rather than a discrete point-like event. This
was judged inexpensive to add on top of infrastructure already in place for the
current probe, and was queued as the natural next step after the current
fork-based probe, ahead of the previously queued identifier-recovery follow-up,
to be taken up if (as anticipated) the fork-based result proves inconclusive
rather than the fork-based probe's use of a single vivid example if the
fork-based probe's own conceptual "isn't a lot of work."

As of the end of this chunk, the corrected lens free-divergence probe is running
for real (verified via active model-download progress) on the pod; the next
milestone is the three-outcome-bucket result from that probe, after which both
it and the scoped key-grafting conclusion are to be folded into the paper before
it is finished and shipped. The identifier-recovery follow-up experiment and the
newly proposed token-by-token trajectory scan remain queued as near-term,
low-cost follow-ons behind the paper's completion.
