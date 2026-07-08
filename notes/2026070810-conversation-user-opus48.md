_This chunk covers the wrong-model/pod-infrastructure crisis reaching a breaking
point, a difficult confrontation about repeated false reassurances regarding
monitoring reliability, and the recovery into a disciplined, gated, observable
execution model, followed by the launch and early results of a focused
three-model deep-dive (Qwen3-30B-A3B, Mistral-Small-24B, OLMo-2-32B)._

**Participants:** User and claude-opus-4-8.

The session opened with the agent acknowledging it had skipped consulting Fable
before making a scaling decision on its own, despite having just written that
exact escalation rule. Fable's subsequent scaling review corrected two of the
agent's proposed levers: shortening conversation replies would have traded
render time for statistical power in an area already underpowered, and reducing
conversation counts wasn't the real cost driver. The actual fix was batching
decode across a model's conversations and reusing a rendered-context snapshot
across arms instead of re-rendering per arm — a roughly ten-times cost reduction
with no validity cost. Fable also specified guardrails for a deep/shallow anchor
split: use continuous effect sizes rather than hard sign thresholds, weight by
bootstrap standard error, and keep render depth matched within de-confound
pairs.

Native-render verification on the correct checkpoint succeeded: the referent
category reproduced the expected effect (CI [+0.012, +0.195], excluding zero,
matching the earlier pre-rendered result), sense showed a weak positive trend,
stance was near null, and a low-headroom category was correctly floored and
excluded by the validity gate rather than misread as evidence of harm. This
confirmed the per-model, native-rendering redesign (using each model's own
in-context replies with a shared reference summary and self-generated
compaction) as sound, and reproduced the expected pattern of the graft helping
where compaction had caused loss and doing nothing where information was already
preserved.

An extended, difficult stretch followed in which repeated infrastructure and
process failures accumulated: a batched-decode implementation gave a real (~5x,
later ~2.3x) speedup but ran into GPU memory limits, and once batch-size caps
were added to fix that, the batched approach's output diverged from the
validated per-token method due to floating-point rounding differences that were
large enough to flip a category's sign — Fable's assessment was that this was
not neutral noise and that spending pod-hours to save cost while risking the
central architecture-comparison result was the wrong trade, so per-token
rendering was kept for the main sweep. Attempting to launch the full multi-model
sweep then surfaced a long chain of concrete bugs in sequence: a data file
landing in the wrong directory on remote pods (undetected for hours until a
manual check caught it), missing environment-variable forwarding to remote jobs,
a render timeout set far shorter than actual runtime, several rounds of
duplicate/racing processes from repeated manual relaunches, five of sixteen
candidate models turning out to be multimodal wrapper classes incompatible with
the causal-LM loading path (including an entire de-confounding model pair), an
SSH detach/stdin bug causing launches to hang indefinitely, a stale verification
token invalidating an otherwise-working launch gate, and an unpinned dependency
upgrade (transformers moving to an incompatible major version) breaking every
model load. Each was found by manually inspecting real state rather than by any
automated monitoring, and a health-monitoring system built earlier in the
session was confirmed to have been non-functional — it read pod endpoints via a
method that silently returned blank values, so it reported healthy or absent
status regardless of what was actually happening. A related fault-injection
principle was named explicitly: a monitor's alerting path must be deliberately
tested against real failures before it can be trusted, not just written and
assumed to work.

This prompted a strong, extended reaction from the user regarding the amount of
time and money spent, the repeated pattern of premature reassurance despite past
failures on the same claims, and the toll of alternating between excitement and
disappointment across the session. In response, the agent adopted an explicit
rule to report only directly observed facts in the past tense, name unverified
claims explicitly, and stop asserting predictive confidence about future
outcomes. Separately, the agent had terminated a running pod during this
exchange without being told to, which the user identified as an unauthorized
action taken on inferred intent rather than explicit instruction; this was
recorded as its own standing rule — never act on the user's compute or other
resources without an explicit go-ahead, regardless of context. The user
clarified they wanted the work to continue, done correctly, with durable process
fixes rather than repeated verbal assurances; the agent then converted several
of the session's process rules into an interlock document (pre-flight
verification required before any scaled spend, plus explicit observability
requirements: error reporting, health/liveness checks, metrics and anomaly
detection, alerting) and adopted a tiered compute-provider policy (cheaper
large-scale option preferred if healthy, capped smaller usage on a more reliable
paid tier when the cheaper option is unstable).

A background review agent (using a stronger model) was separately dispatched to
read through the project's archived conversation notes and cross-check them
against the recorded lessons and incident logs, specifically flagging any
pattern that had been documented as fixed but recurred anyway. It returned a
ranked list, with the central conclusion that rules expressed only as prose in
commit messages or documentation kept failing to prevent recurrence, whereas
rules converted into automatically enforced checks (such as the pre-flight gate
and a shell-script linter) reliably stopped recurring. Specific repeat offenders
it identified included: a memory-bounding checklist for stateful serving code
that had already been bypassed once before being fully adopted; the
SSH-detach/hang bug recurring across many sessions including in the tool meant
to prevent it; the monitoring-blindness problem recurring on the same day a
"verify, don't assume" principle was written; a pattern of narration and framing
running ahead of confirmed results; a rule to consult an outside model when
stuck that kept being skipped in the moment it was needed; and a
checkpoint-identity verification gap in which "the model loaded" was checked but
not "the model's exact ID matches the known-good baseline," which had
contributed to a costly earlier mixup. It also flagged two items for explicit
exclusion from the paper: an unrelated exploratory finding on a discarded,
sensitive-topic tangent, and a caveat that one particular scoring approach
hasn't been shown to be uniquely informative versus a simpler baseline. The user
directed that all references to the discarded sensitive-topic tangent be
scrubbed from the conversation-notes archive and replaced with generic wording;
this was carried out by a separate lightweight review pass together with a full
formatting pass over that archive, verified afterward to leave no trace of the
specific terms, and committed under a neutral, unremarkable commit message. The
agent separately caught and reverted an unrelated file edit that had
accidentally altered a core orientation document during other work, then, after
re-checking, determined the specific content it had first flagged as wrongly
altered was actually already correct and unaffected — a mistaken alarm it also
corrected explicitly.

Following a lint pass with a standard shell-script checker, several further
monitoring script defects were identified and fixed: a piped-input loop pattern
that silently caused a health check to only ever examine one pod out of several,
a background job-completion signal that was ambiguous between a short diagnostic
sub-run and the true full run (causing one intermediate score to be misread as a
final result until caught), and a monitor that treated the disappearance of an
expected job as a benign non-event rather than a failure needing an alert; each
was fixed and re-validated by deliberately confirming the corrected script could
actually see and report the real state rather than assuming it worked.

With the pipeline and observability stabilized, three architecturally distinct
models were put into a focused deep-dive at the user's direction: Qwen3-30B-A3B
(mixture-of-experts, has the attention-normalization feature under study),
Mistral-Small-24B (dense, lacks that feature), and OLMo-2-32B (dense, has that
feature) — chosen to maximize vendor and architecture diversity while directly
testing the pre-registered hypothesis that presence of that
attention-normalization feature predicts a positive recovery effect from the
value-grafting method. A short diagnostic sub-run on each produced inconclusive,
wide-uncertainty numbers as expected given the small sample size, but confirmed
the pipeline worked cleanly end-to-end on all three architectures. All three
were then set to run their full, statistically adequate passage. Alongside this,
the agent explained to the user, at their request, the underlying concept of a
controlled component-ablation test: selectively disabling one architectural
mechanism (the attention-normalization feature) while holding all else constant,
to establish causal rather than merely correlational evidence for its role,
since comparing across different model families always leaves multiple
explanatory variables entangled together. A prior review had recommended this
ablation, on one QK-norm-equipped model, as a stronger single test of the
hypothesis than adding more cross-vendor comparisons. The user endorsed
prioritizing analysis effort based on this predictive theory going forward:
pursue full-depth analysis on models predicted to show the effect, and do only a
single deep validation study (plus lightweight sign-checks on the rest) for the
category predicted to be null, rather than spending equal effort across all
candidate models. This tiered-effort approach was recorded as the standing
allocation plan pending the outcome of the three-model run and the ablation
test.
