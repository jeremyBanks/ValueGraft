_This transcript covers the resolution of the ValueGraft cross-architecture
sweep's positive-control crisis — a multi-hour debugging saga that traced back
to a wrong model checkpoint rather than a science failure — followed by
expansion of the model set and corpus, several operational near-misses, and a
final confirmed-real result with one open corpus-quality caveat._

**Participants:** User and claude-opus-4-8.

The session opened with a live re-run of the original F1 gap-closure code
failing to reproduce the saved numbers, which the assistant initially
misattributed to Mixture-of-Experts routing nondeterminism. The user pushed back
("this seems like bugs, we had stable results earlier") and asked for Fable's
input on a decision of this importance. Fable, working from data already on disk
(no GPU needed), identified the actual defect: the paper's headline gap-closure
metric, mean of (E−B)/(A−B), is Cauchy-unstable with small denominators, and the
dramatic swings (including the paper's "stance null" dissociation) were
estimator artifacts, not evidence the effect was fake. Two same-hardware reruns
confirmed the underlying apparatus is fully deterministic (byte-identical
results), and bootstrapped confidence intervals on the robust metric (raw E−B +
%-helped) showed referent as a statistically significant recovered effect (CI
excludes zero), sense as positive but underpowered, and stance as a genuine null
— a more nuanced but sound version of the original claim. This resolution
generalized to the earlier K/V-grafting work: the same broken ratio estimator
had made keys look like they "actively hurt" recovery, when on the robust metric
keys are actually neutral. The user confirmed this doesn't reopen the keys
question — "value is the operative axis, keys don't help" remains settled; only
the reported magnitudes need correction, not the conclusion.

With the estimator fixed, the project moved to the marquee experiment: a
cross-architecture generalization sweep testing whether the value-graft effect
travels across model families. Early results showed Qwen3-30B-A3B (MoE) grafting
positive (~~+0.12–0.14 referent) while Qwen2.5-32B (dense) grafted significantly
negative (~~−0.3), reproduced on both the cross-arch harness and the trusted
original apparatus — a genuine architecture-dependent sign reversal. A consulted
Fable design review flagged a critical trap before wide spending: since value
grafting is an attention-mechanism intervention and MoE only affects the FFN
block, attributing the sign flip to "MoE vs. dense" would be a
reviewer-vulnerable conflation; the real driver is more likely attention
geometry (KV-head count, GQA ratio, head-dim, RoPE/QK-norm details), and the
experiment needed to be redesigned to test that hypothesis directly — including
a same-generation dense/MoE de-confound pair (Qwen3-32B alongside Qwen3-30B-A3B)
and, once identified as available, an independent second dense/MoE pair from
Gemma 4 (26B-A4B MoE vs. 31B dense). The model set was substantially widened
over the course of the session, driven partly by explicit vendor-diversity
pushback from the user (the original list was Qwen-heavy), landing at 16 models
across 9 vendors (Qwen, Mistral, Google, AllenAI, Zhipu/GLM, OpenAI's gpt-oss,
Microsoft's Phi-4, 01.ai's Yi, NVIDIA's Nemotron), organized into
anchor/replication/geometry/breadth tiers, with a "go wide when you can,
tentative results ranked below more reliable ones" ordering rather than a hard
cutoff. Llama-2 was deliberately excluded (off the ~30B scale band, different
era/tokenizer) despite being the only available full multi-head-attention
(non-GQA) reference point; this tradeoff was surfaced but the user chose to
leave it out.

A parallel discovery reshaped the sweep's design further: a positive control run
through the redesigned cross-arch harness, using a fixed externally-authored
(Sonnet) summary, failed to reproduce the known +0.136 result on Qwen3-30B (came
back null). Isolation testing confirmed the fixed foreign summary — not the
harness or alignment code — was suppressing the graft; switching to
self-generated summaries restored the effect. This produced a mechanistic
finding: the graft appears to require the model's own act of generating its
summary, not merely reading an externally supplied one, since the recoverable
write-time continuity is tied to the model's own summarization act rather than
the text content alone. The user reasoned through and accepted this as
enrichment rather than a setback, noting the raw-E−B primary metric would still
be affected by summary detail/length (since it isn't gap-normalized like the
ratio), so per-model summary length/detail should be logged and reported
alongside results as an explicit covariate rather than ignored.

Substantial engineering hardening occurred alongside the science: the alignment
mechanism (matching summary token spans between write-time and compacted
contexts) was scrutinized after the user raised alarm over apparent "token
matching" logic reminiscent of a discarded old approach; investigation showed
the difflib-based alignment was a positional 1:1 mapper operating only within
same-text regions (not fuzzy cross-context matching) and had never compromised
results, but the user's instinct that it was overkill for what should be a
direct substring/span operation was validated — it was replaced with a simpler
direct span-offset map, verified byte-identical to the old method on existing
data. A cluster/conversation-level bootstrap was adopted as the correct unit of
statistical power (probes within one conversation are correlated; conversations
are the independent unit), which is why corpus expansion (more conversations,
not more probes per conversation) was pursued to sharpen confidence intervals —
this was Fable's original recommendation (12→27 conversations) and later doubled
again at the user's request (27→54) after a cost estimate (~$12–16 to double
tests across all models, since only compute scales with test count, not the
fixed download/load cost). Harness controls added during this phase included
placebo grafts, an identity-graft integrity check, an alpha dose-response sweep,
full raw-trace and model-hyperparameter capture (intended to support a
regression of effect sign on attention-geometry properties — the paper's real
target claim), and a per-layer "champion" scan. The user was explicit that any
champion-tuning results must be labeled prominently as a weak, non-exhaustive,
non-comparable quick scan — at most suggestive of future work, never presented
as an optimization or valid cross-model comparison. A later Fable consult
reframed the champion scan as potentially load-bearing rather than a side
figure: since a uniform graft's effect is a nonlinear composition over depth, a
"rescue test" (grafting only the positive-scanning depth regions on a
known-negative dense model) could causally demonstrate that the sign reversal is
a depth-composition/graftability-profile effect rather than an intrinsic
architectural property — this was folded into the harness with fractional-depth
binning (to make region definitions comparable across models with different
layer counts) and a per-layer value-alignment cosine measurement as a candidate
geometric explanation.

The user also directed that final paper writing may be delegated substantially
to Fable as first-draft author (with the existing review/iteration/human-edit
process still applied on top), and confirmed a standing requirement that a
shareable major paper revision receive one Codex (GPT-5.5, extra-high reasoning
effort) review — run directly by the assistant via the CLI, not requested of the
user. The Codex CLI was installed and confirmed authenticated during this
session, configured to default to gpt-5.5/xhigh.

The build phase toward the wide sweep saw several operational failures that were
explicitly recorded as lessons: (1) the local MLX-based conversation-rendering
pipeline was identified as an unnecessarily slow choice for generating filler
dialogue text (~90 minutes for the render) since that text needs no model
capability at all — it was replaced with a parallel mix of foundation-API
subagents (Sonnet, Opus, Fable) and Codex, cutting generation to minutes; (2) a
single sequential subagent tasked with authoring 27 new scenarios was killed by
the assistant based on a misleading small-output-file-size proxy, when the
subagent was actually near completion — this was owned as a process failure
(destructive actions require verifying real progress, not proxies) and led to a
sharper reframing that sharding generation work across multiple parallel authors
is a quality improvement (author diversity, fresh attention per item), not just
a speed one; (3) the recovery afterward hit a second failure when an
unconditional cleanup script deleted six recovered-but-uncommitted batch files
after a corpus-merge assertion failed on a padding bug — recovered without
content loss because the underlying generator script reproduced byte-identical
output, but recorded as a rule never to delete uncommitted work until its
consumer has both succeeded and committed; (4) a debugger subagent working on
the alignment crash created and worked on a separate git branch, which the
assistant hadn't caught, causing several commits (including the corpus freeze)
to accumulate off trunk before being fast-forwarded back — the user established
a standing rule of only ever working on trunk, and that commits should be pushed
after every unit of work.

The most consequential failure, surfaced when the user pressed to "fail faster,"
was that a debugger subagent had spent roughly two hours running three full
~40-minute positive-control reruns without executing the cheap decisive
diagnostic (a null self-graft test) that had been assigned to isolate
plumbing-bug-vs-substance-loss. The assistant stopped the subagent and took over
directly, eventually tracing the root cause through several intermediate false
leads (a "hardening" commit had silently replaced the tolerant difflib alignment
with a strict span-matcher across both the trusted and cross-arch codepaths,
which broke specifically on thinking-model self-generated summaries containing
`<think>` blocks) to the actual root cause: the assistant had been running the
wrong model checkpoint throughout — `Qwen3-30B-A3B` (a thinking model) instead
of `Qwen3-30B-A3B-Instruct-2507` (the non-thinking variant the original +0.136
result was measured on, and the harness's own coded default, which had been
manually overridden). Every downstream symptom — the think-block token-length
mismatch, the alignment crash, the failed positive control — stemmed from this
single wrong-model error. Running the trusted apparatus against the correct
checkpoint confirmed the effect fully reproduces: referent +0.1246 (71% helped)
on the original 12 conversations, essentially matching the known +0.136, with
the expected sense-positive/stance-null dissociation intact. This closed the
multi-hour crisis with the finding that the science was sound throughout and the
entire episode was self-inflicted tooling/checkpoint confusion.

One open issue remains at the point this transcript ends: splitting the same
correct-model run by conversation batch showed the original 12 conversations
carry the effect cleanly (referent +0.12) while the 42 newly authored
conversations (c13–c54) show a much weaker, near-null referent recovery (+0.009)
despite having a substantial pre-graft gap (A−B = 1.12, meaning there is real
evicted content to recover) — ruling out the benign explanation that the new
conversations simply have less damage to fix. This indicates the
model-mix-rendered corpus expansion is lower quality for this specific
measurement and, as currently constructed, dilutes rather than strengthens the
signal. The assistant's recommendation, not yet acted on, is to run the
cross-architecture sweep on the reliable original 12-conversation corpus for the
actual sign-map result, while treating the corpus augmentation's quality problem
as a separate matter to diagnose and fix before it's trusted for added
statistical power.
