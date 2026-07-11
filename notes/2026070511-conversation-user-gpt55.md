_This conversation reviewed the current KV-cache compaction pilot and decisively
reframed the project around finding a practical mitigation for compaction
damage, culminating in committed notes on adaptive ValueGraft blending._

**Participants:** User and gpt-5.5-xhigh.

**Handoff State.** Read-only review found the 4B results mechanically consistent
with committed JSON: 2,520 score rows, 2,089 judge verdicts, complete B-causal
repair for 12 synthetic conversations, and matching headline tables. Important
follow-ups remain: local fallback scoring disagrees with external-judge
stance/ruled-out scoring; `STATE.md` is stale about 30B progress; `DECISIONS.md`
lacks recent methodological/runtime updates; and `analyze.py` omits newer arms
from its per-conversation appendix. Partial 30B artifacts existed, including
`c01`–`c08`, but were not fully reconciled in this review.

The central correction is that demonstrating context-conditioned differences in
write-time KV state is only a mechanism result. The valuable question is whether
a small, plausibly deployable cache-state intervention reduces the behavioral
damage caused by text-only compaction. The intended comparison is full context
(**A**, oracle) versus freshly re-encoded summary plus tail (**B**, production
baseline), with mitigation arms such as in-context summary KV (**H/SelfGist**)
and value blending (**E/ValueGraft**). Success requires improvement over B
toward A, while surviving leakage, shuffled, wrong-conversation, and
generic-fluency controls.

The next workflow was specified as two phases: first finish only the smallest
coherent portion of any already-active experiment, update `STATE.md`,
`DECISIONS.md`, and results documentation as warranted, and commit that
stopping-point state; then stop expanding the old matrix and execute a narrow
mitigation-first design. Suggested options include H-pack versus B-min with
identical summary text and positions, focused low-alpha E-post tests, a smaller
high-sensitivity probe set where A succeeds and B fails without summary/tail
leakage, and a focused honesty/uncertainty test based on H-gap’s apparent
reduction in fabrication. G/SoftGraft should be deferred unless simpler tests
justify it.

The current E-post arm uses fixed global blending,
`V_new = (1−alpha)V_fresh + alpha V_old`, at exact token matches. Adaptive
blending is not yet implemented: the proposed refinement is
`V_new = (1−gate_i)V_fresh + gate_i V_retrieved`, with gates varying by token,
layer, or head according to attention sharpness, top-match margin, head
agreement, and exact/paraphrased correspondence. Attention-based retrieval is
preferred over raw value-vector cosine similarity because QK scores reflect the
model’s trained retrieval mechanism. Blending is favored over replacement
because old state may preserve useful interpretation but may also be
incompatible with the freshly encoded compacted context.

A concise explanation for non-specialists was established: compaction replaces
full history with summary plus tail and re-encodes it; the experiment asks
whether retaining or softly injecting a small amount of the original KV state
makes that transition less damaging. The pilot shows real signal in old state,
but naive interventions mostly do not recover answer accuracy; the strongest
deployment-relevant observation is that original summary KV may increase
appropriate admissions of missing information rather than fabrication.

The adaptive-blending notes were written to `adaptive-blending-notes.md` and
committed alone as `347602c Add adaptive blending notes`. Existing untracked 30B
JSON files were deliberately left untouched. A single thread heartbeat was
scheduled to check back approximately 60 minutes after setup and again
approximately 240 minutes after setup; no repository changes were made by that
scheduling action.

## Conversation sources

- `019f3007-bab0-7e50-b019-2625d1538f63`
