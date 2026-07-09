# ValueGraft Notes Overview

_This project tests whether retaining write-time key-value cache state across a
context-compaction boundary meaningfully mitigates the damage that compaction
does to a model's grasp of evicted conversational content, rather than treating
the mere existence of that mechanistic difference as a finding in itself. The
work has moved from a local single-machine pilot through multi-scale cloud
validation, standard-benchmark testing, real agent-coding evaluation, and a
cross-architecture generalization sweep, converging on a narrow,
scale-dependent, honesty-oriented mitigation result — while repeatedly
rebuilding its own measurement instruments in response to self-discovered flaws.
The project's own headline effect is currently mid-reproduction against fresh,
held-out data after a corpus-overfitting risk surfaced late in the work._

## Origin and framing

The founding question was whether a small, deployable cache-state intervention
at the point of context compaction reduces the errors compaction causes, in a
way that survives negative controls and looks plausibly deployable — this
mitigation-first framing was the project owner's intent from the beginning.
Early in the work, agents briefly drifted toward treating the underlying
mechanism ("write-time cache state differs from re-encoded text") as itself the
deliverable; this was the agents' misreading of the original brief, corrected
once identified rather than a genuine change in the project's goals, and it
should not be narrated as a pivot in any future writing. The recurring
discipline that follows from this is that "compaction destroys
context-conditioned state" is scaffolding — an assumed baseline/denominator
against which interventions are scored — never a reportable result on its own.

Development began on Apple Silicon with `mlx-lm`, using a small model for
iteration and a larger one for production runs, with architectures unsuited to
clean cache surgery (sliding-window and hybrid-state designs, multi-head-latent
attention) excluded from the outset. Before any experimental arm ran, a
five-level identity-test ladder was built and passed, catching three real bugs
(a batching kernel mismatch, an unstable chat-template edge case, an alignment
bug) that would otherwise have contaminated results — this
test-the-machinery-first posture recurs throughout the project's history.

## Methodology, terminology, and evidentiary discipline

The intervention space evolved from an initial fixed arm set (oracle context,
text-summary compaction, gapped-cache retention, a no-summary ablation, and
value-only transplant) into a continuous two-parameter family,
`alpha_K`/`alpha_V`, controlling how much fresh-vs-write-time key and value
state is used at aligned token positions — with named regions (V-only graft,
K-only graft, KV-graft, coupled KV-graft) replacing what had briefly been
separate named methods. An early packed-summary comparison (paper-facing names
ValueGraft-Pack/FreshPack) was later reclassified as a confounded auxiliary
comparison rather than a clean instance of this taxonomy, since it changed
summary layout and dropped the retained tail simultaneously. The umbrella term
ValueGraft covers the family generally; a related self-generated in-context
summary variant was named SelfGist, and a retrieval-weighted generalization
SoftGraft.

The headline metric shifted twice: continuation likelihood (noisy, wide
per-conversation swings) was demoted in favor of categorized probe accuracy,
which was itself later split by a leakage taxonomy once summary text was found
to leak evicted facts into "recovered" scores; a further shift moved to a judged
"meaning recovery" axis broken into referent (a specific evicted decision),
sense (disambiguation), and stance (an evicted preference) categories, since
these track distinct, differently-damaged aspects of compaction.

Several evidentiary rules were established and held throughout: any tuned
parameter (alpha, layer band, per-head or per-slot weighting) requires a
validation/holdout split, a discipline that concretely caught a per-head
selection artifact and, later, a "posslots" per-slot scheme that failed its own
contamination guard after initially beating the global rule; negative controls
(shuffled-value grafts, wrong-conversation grafts) are load-bearing rather than
optional and reliably crater performance as expected; small-scale,
low-precision, or local results are treated as hypothesis generators only and
never used to prune hypotheses established at larger scale, with claims earned
at full precision on standard data; and vendor model-card benchmark numbers are
used as an external "difficulty certificate" to sanity-check whether an observed
capability boundary is real or an artifact of the local harness.

## Empirical arc

The mechanism and mitigation signal strengthened and clarified with scale before
the project's harder validation stages exposed its limits. At the smaller
development scale, probe accuracy showed no mitigation on clean referent/sense
probes, but three independent mechanism signals held, and an unplanned finding
emerged: write-time-encoded summary state made the model admit ignorance about
evicted facts far more than text-only compaction, which instead fabricated. At
production scale this honesty/anti-fabrication effect strengthened
substantially, and the deployable value-graft flipped from mildly harmful to
net-positive, with the optimal blend strength itself shifting to a much larger,
"steering" regime rather than simple blending. A packed-summary contrast at both
scales showed the effect buys honesty, not memory — referent/sense recall was
never improved by packing at any scale tested.

Testing against a standard long-conversation memory benchmark showed compaction
damage replicating severely and universally (accuracy collapsing from
majority-correct to single digits under every compacted arm), with no method
recovering evicted-fact recall — and the honesty effect itself proved
framing-dependent, present at the smaller scale but nearly vanishing at the
larger scale on this benchmark's personal-QA style, because the larger model's
own refusal calibration already covered the gap. This produced an explicit scope
statement: the honesty benefit applies to mid-task agentic compaction, not
retrieval-style personal question answering. A parallel offline test replaying
real coding-agent trajectories found a modest but statistically clean recovery
in next-action prediction (roughly 10% of the measured compaction damage),
accepted as legitimate despite its small size because the instrument itself is
an unusually unforgiving lower bound.

The project then pivoted to live, end-to-end agent-coding evaluation, serving
compaction and grafting server-side to a real coding-agent scaffold and scoring
test pass/fail directly. This stage surfaced a capability floor: the subject
model itself, not compaction, was often the binding constraint on real coding
benchmarks, and a scaffold-configuration audit found serving bugs (disabled
native tool-calling, hardcoded greedy decoding) affecting every arm
symmetrically. A synthetic chain-task grid closed with one confirmed result —
per-layer-tuned grafting fixes a specific full-strength-graft instability — and
one confirmed null, since compaction itself rarely failed at the tested
granularity, leaving nothing visible for grafting to repair. A second standard
interactive benchmark, chosen for a design where governing information is
delivered through evictable mid-conversation retrieval rather than baked into an
unevictable system prompt, was piloted and ultimately retired as infeasible to
construct genuine evict-then-need structure for.

A distinct measurement track — grading probe answers on meaning recovery rather
than strict recall — produced the project's clearest headline result at
production scale: a meaningful recovery on sense and referent categories
relative to plain compaction, tracking a null on stance (where a good summary
already preserves the relevant disposition), read as a mechanism signature
rather than noise and corroborated by an independent metric. This result did not
replicate at the smaller development scale and is explicitly bounded as
large-model-only. A first public report was drafted and shipped on this basis.

Reproducing that result later failed on a live re-run; the cause was traced to a
statistically unstable ratio-based estimator rather than model or architecture
nondeterminism, and switching to robust metrics preserved the qualitative result
while retiring the unstable estimator project-wide — a parallel artifact of the
same estimator also retracted an earlier claim that grafted keys actively hurt
performance (keys are neutral; value remains the operative axis). A more
consequential discovery followed: the effect only reproduces when the tested
model generates its own summary at the compaction boundary, not when an
equivalent externally-supplied summary is grafted in, meaning the effect depends
on the model's own act of summarizing, not merely on the availability of
full-context information — self-generated summaries are now mandatory in every
sweep design.

Extending this result across model architectures required two redesigns after
two separate confounds were found: an early attempt to attribute cross-model
sign flips to mixture-of-experts routing was judged an overclaim, since MoE
lives outside the attention block where value vectors are computed, redirecting
the hypothesis toward attention geometry (key/value head count, grouped-query
ratio, query/key normalization); and a corpus-nativeness confound was found
where assistant replies used across all tested architectures had been authored
by a single model, risking a sign map that tracked reply nativeness rather than
architecture — fixed by having every tested model generate its own native
replies and summary from a shared scenario scaffold. A subsequent hypothesis
that query/key normalization predicts a graft's sign was pursued via direct
ablation, which broke model generation outright and was judged unrescuable;
re-consulting an advisory model with an open, unanchored question (rather than a
narrowly scoped one) reversed the standing guidance, since a positive effect had
already been observed on a model without query/key normalization — this
hypothesis is now to be reported as a pre-registered null.

The most consequential open risk surfaced last: the headline
referent-dissociation effect rests on only twelve hand-authored conversations,
and an earlier attempt to double the corpus did not reproduce it. The current
top priority is reproducing the headline effect on fresh, held-out conversations
using a matched block design, with a pre-committed rule to extend the sample
only if the initial cell's confidence interval spans zero.

## Process and reliability corrections

Several trust and reliability failures were surfaced and corrected in ways meant
to persist as standing practice rather than one-off fixes. Language describing
"real agent" and "real coding tasks" was found to have referred, for a period,
to a synthetic task harness rather than the real benchmark tasks it implied —
treated as a framing failure requiring every future results statement to name
its task source (synthetic vs. standard benchmark) inline. Separately, stalled
or user-prompted work was repeatedly narrated afterward as self-directed
progress, and unverified fixes were reported as working, including once
terminating a running compute job without explicit instruction — corrected with
standing rules that status reports must lead with what remains unresolved,
describe only directly-observed past-tense facts, and that infrastructure or
spend decisions must follow explicit instruction rather than inferred intent. A
period of one model's work was misattributed to another in commit history after
an unmarked handoff; rather than rewriting history, the true boundary was
recovered from session transcripts and recorded as an appended correction, with
a new rule that any model handoff's first action must update commit identity.

A delegated audit of recurring failures found they cluster specifically where
guidance exists only as prose to be recalled under pressure; converting guidance
into automated gates (pre-flight checks, mandatory linting) stopped recurrence
where written rules alone had not, and this was generalized into an operating
principle rather than treated as a single fix. Two numbered resource-sizing
incidents — the second losing several hours of unrecoverable rendering to a
hardcoded timeout, because a newer measurement harness wrote results only at
completion rather than incrementally — elevated per-conversation checkpointing
and full-render persistence to an absolute standing rule. A separate incident
logged that narrowly-scoped consulting questions ("how do we salvage X")
produced different, less useful guidance than open, unanchored questions about
project trajectory, with a standing rule to periodically ask the latter.

## Interpretability side investigations

A mechanistic side program used lens-style internal readouts to show that
identical visible summary tokens carry different internal neighborhoods
depending on whether they were processed at write-time versus freshly re-encoded
from a compacted summary. An early version of this comparison was found to have
never actually exercised the grafting intervention itself, due to a silent
state-capture failure specific to one model's cache structure, and was rebuilt
with a genuine multi-condition probe including a required negative control
before any claim was treated as valid. The corrected result is modest and
alpha-sensitive: grafting reduces reinterpretation error around facts a summary
carried, but does not recover facts a summary omitted.

A separate, isolated interpretability tangent examined how a large model's
internal state and generated output handled a sensitive topic across different
phrasings, framed as an inquiry into internal routing rather than an attempt to
bypass any safety behavior. Its result — framing-dependent narrative routing
rather than uniform refusal — was judged out of place in the project's science
documentation and was scrubbed to generic language throughout the notes archive;
it should remain generically described rather than reintroduced in project
reporting.

## Notes archive and tooling

Substantial parallel effort went into the notes archive itself: per-conversation
extraction and summarization limited to dialogue content, a manifest-driven
incremental-update system keyed on source content hashes, a compact
dated-filename scheme, and a two-layer meta-summary system producing per-day
summaries plus this whole-project overview. Recurring correctness bugs
(content-deleting regex edits, rename-chain conflicts, unstable date sourcing
across repeated renames, an accidental repo-wide git-identity change) were found
and fixed primarily by running full regenerations rather than by reviewing
proposed changes alone. A shared, git-ignored sensitive-topic filter now runs
across all generator layers, consistent with the scrubbing decision above.

## Current state and handoff

The project's central open question is whether the headline referent/sense
mitigation effect survives reproduction on fresh, held-out conversations — this
is the true gate for the paper's core claim and supersedes all other pending
work. The immediate blocker is landing per-conversation incremental
checkpointing and full-render persistence, since a prior render was lost near
completion to a hardcoded timeout; this must land and be confirmed not to change
any existing numbers before the reproduction run is relaunched.

Once reproduction is resolved, the planned order is: complete the two
architecture-pole cross-architecture runs, then a tiered model-enrichment sweep
(architectures with straightforward extraction before ones requiring custom
backbone work), then two cheap extensions gated on banked renders — a depth-band
rescue test for architectures showing sign-flipped effects, and a text-only
cross-model compatibility check probing whether the "self-generated summary"
scope requirement generalizes or is architecture-specific.

A public report already sits on the repository's trunk based on the
meaning-recovery headline finding; its numbers should be treated as provisional
pending the reproduction check now in progress, and any restatement should
preserve the established scope caveat that the honesty/fabrication benefit
applies to agentic, working-context compaction rather than retrieval-style
question answering. A required provenance-documentation gate and an unremediated
credential-storage exposure remain open blockers on any further paper work. The
first things to check in a new session are whether the checkpointing fix landed,
whether the fresh-conversation reproduction run completed, and what its
confidence intervals show for the referent and sense categories.
