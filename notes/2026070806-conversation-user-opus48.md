_This conversation covers the final stretch of paper-shipping followed by a
serious mid-sweep validation crisis (positive control failure → traced to a
self-inflicted alignment bug), a major design-review pivot with Fable that
reframed the cross-architecture sweep, and a substantial widening of the
model/corpus scope — punctuated by an infrastructure mishap (killing a
near-complete corpus-authoring subagent) that was reframed as a quality
improvement._

**Participants:** User and claude-opus-4-8.

## Alignment "token matching" scare — resolved as non-issue

The user raised alarm that `build_alignment`'s use of `difflib.SequenceMatcher`
was "spurious token matching" that might have compromised all prior results.
Investigation showed this fear was unfounded: difflib operates only within
matched regions (summary↔summary, tail↔tail) where identical text produces a
single contiguous 1:1 positional block — it never matches summary text against
arbitrary conversation content. Diagnostic on Qwen2.5 showed 0/1190 tokens
dropped (100% aligned) at `MIN_BLOCK=8`. However, the user's engineering
instinct that a fuzzy sequence-matcher was overkill for what should be a direct
substring/span operation was correct and adopted: the harness was refactored to
`build_alignment_direct`, an exact-span offset map that raises loudly on any
tokenization divergence instead of silently dropping unmatched runs. This was
equivalence-verified (12/12 conversations byte-identical to the old method on
fixed-summary data) before being deployed — establishing a "prove equivalence
before switching mid-stream" pattern.

## Positive control failure and the fixed-summary mechanistic finding

With the alignment scare resolved, the Qwen2.5-32B negative graft result
(~-0.30) was confirmed as genuine architecture behavior via the trusted,
harness-independent `gap_closure_cat.py` (referent -0.26, sense -0.38, stance
-0.25, all significant) — validating both the cross-arch harness and the user's
original hypothesis that some architectures reverse the effect.

The user then pressed on whether the harness itself had been validated against a
known positive result, not just agreement on a negative. This check (running
Qwen3-30B-A3B — the model with the known +0.156 F1 result — through the
cross-arch harness with the fixed Sonnet summary) failed: it returned near-null
(referent +0.004, sense -0.147) instead of the expected positive. An isolation
test (same model/harness, but self-generated rather than fixed summary)
recovered the expected result almost exactly (+0.136 referent, 81% helped,
matching F1's +0.156/81%). This established a genuine mechanistic finding: **the
value-graft only works when it re-injects the model's own generated summary; a
foreign/fixed summary suppresses the effect to null.** The interpretation is
that the graft recovers write-time state from the model's own act of
summarization, not generic full-context information. This reframed a validation
crisis into a richer paper claim, and the sweep design switched wholesale from
fixed to self-generated summaries. A caveat raised by the user (self-gen
summaries could vary in detail, confounding the gap size) was resolved by noting
the primary metric of raw E-B is somewhat sensitive to summary length/detail,
mitigated by reporting per-model pre-graft gap (A-B) alongside raw E-B and a
conditioned/scale-free ratio, plus logging summary token-count per conversation
for analysis.

Lesson recorded: negative-agreement between two methods is not sufficient
validation; a positive control (reproducing a known positive result) is
necessary and is what caught this bug. This is now a durable memory
(`validate-before-trusting`).

## Fable design review reframes the cross-architecture sweep

Before committing to a wide 12-model sweep, the user required a Fable design
consult, given the stakes ("major decision-making... I think I wanna go really
really wide"). Fable's critical insight: **value vectors live in the attention
block; MoE structure lives in the FFN, so MoE never touches W_V.** This means
attributing the Qwen3-MoE-positive vs Qwen2.5-dense-negative reversal to
"MoE-ness" is a likely reviewer-fatal claim — the real driver is more plausibly
attention geometry (KV-head count, head-dim, GQA ratio, QK-norm, RoPE), not MoE
per se. This reframed the entire sweep's organizing question from "does the
effect generalize" to "what attention-geometry property predicts the sign of the
effect."

Consequences of this reframe, all adopted:

- Add a same-generation dense model to each MoE anchor to de-confound generation
  from architecture (e.g., Qwen3-32B dense alongside Qwen3-30B-A3B MoE).
- Prioritize depth (rich controls, full traces, all attention hyperparameters
  logged) over raw model count, at least on a smaller set of anchors.
- Capture placebo graft, identity-graft integrity check, alpha dose-response,
  and full raw traces as a matter of course for the eventual sign-regression
  against attention hyperparameters.
- Treat the own-vs-foreign-summary contrast as a first-class controlled
  experiment, not an incidental observation.
- Corpus needs more conversations (Fable recommended widening from 12), since
  the conversation — not the probe — is the true unit of statistical
  independence for the cluster bootstrap; probes within one conversation are
  correlated.

A second Fable consult (specifically on a proposed per-layer "champion" scan)
produced a further insight: because the uniform graft's effect is a nonlinear
composition over depth (later layers consume values grafted into earlier ones),
the architecture-dependent sign might be explained by _where_ in the depth
profile the graftable value signal lives, rather than being an intrinsic
per-model property. The proposed "rescue test" — grafting only the
positive-scanning depth regions on an otherwise-negative dense anchor, and
checking whether the net effect flips positive — would provide causal evidence
for this depth-composition mechanism rather than just a descriptive fingerprint.
Fable also proposed two cheap byproducts: a per-layer write-vs-read
value-alignment cosine (a candidate geometric explanation per region) and a
region=all sanity check that must reproduce the uniform-graft number. The user
required this scan be explicitly documented as weak/tentative/non-exhaustive (a
quick free scan, not a real optimization or valid cross-model comparison) to
prevent overclaiming in the eventual paper.

## Corpus expansion and generation-backend correction

The original 12-conversation corpus (c01-c12) was augmented, first to 27
conversations (c13-c27) per Fable's power recommendation, then doubled again to
a target of 54 (c28-c54) at the user's request, once cost estimates showed
doubling the test corpus only added ~$12-16 to the eventual 14-16 model sweep
(since download/load costs, not per-plant compute, dominate per-model cost). A
new "strong_prior" plant category was added (codenamed facts like famous
fictional names standing in for mundane facts) to test whether the graft can
shift probability mass away from a well-known prior meaning toward the actual
conversation-established referent — a signed disambiguation test. Two more
usable existing categories (`ruled_out`, `evicted_fact`) were also folded into
the harness's `CATS` tuple, bringing total probes/model from 43 (referent+sense
only) to 115, later further increased with the corpus expansion.

The initial corpus render used the original local generator, `compose.py`, which
ran a local 4-bit MLX 4B model turn-by-turn — reasoned by the user to be
needlessly slow, since assistant reply text in these scenarios is just context
filler around planted facts and doesn't need any model, let alone a slow local
one. This was corrected by switching corpus generation entirely to
foundation-API subagents (Sonnet, Opus, Fable) plus Codex/GPT-5.5, mixed across
conversations for both speed and diversity. The user later explicitly confirmed
corpus authoring must never use locally-hosted models — only foundation API
calls and Codex — reserving local/self-hosted model use strictly for the ~30B
"subject" models being measured in the actual experiment. This is now an
explicit constraint recorded for the project. `compose.py`/mlx was also formally
scoped as a local-only corpus-generation convenience tool, never part of the
reproduction path — the reproduction entry point (task 33) must run only on
committed corpus data via transformers on Linux/GPU, with no mlx dependency, so
a third party reproducing results never needs Apple Silicon.

While recomposing the doubled corpus, one plant's codename ("Palantir") was
quietly swapped for a copyright-safer alternative ("Rivendell") at the user's
request, without any acknowledgment of the substitution in commit messages —
handled silently as instructed.

## Infrastructure mishap: killed near-complete parallel-authoring work

An early attempt to author the 27 additional scenarios (c28-c54) was
sequentially assigned to a single Opus subagent, inconsistent with the
parallel-mix pattern already used for conversation rendering. When the user
questioned the slow pace, the assistant checked a small output file, misread
near-zero output-file size as "stalled," and killed the subagent — only to
discover afterward that it had in fact nearly finished (reached c53/c54) but was
holding its output in-context to write once at the end, so the file-size proxy
was misleading. This was recorded as a durable lesson
(`split-and-verify-before-killing`): independent multi-item generation work
across parallel subagents from the outset for consistency with sibling tasks,
and never kill a subagent based on an indirect progress proxy like output file
size — verify actual liveness/progress first, since destructive actions require
a higher evidentiary bar than passive status checks. The user reframed the
incident positively: forcing genuine parallel batching (across Sonnet and Fable,
in fresh contexts) likely produces a higher-quality, more diverse corpus than
one fatigued single-context model would have produced writing all 27 scenarios
sequentially — parallel batching is a quality lever, not just a speed lever. Six
parallel batches were relaunched to redo the authoring, unaffected by (and
running in parallel with) the ongoing alignment-bug debugging on the pod.

## Alignment bug re-discovered on self-generated summaries (the gate catches a real defect)

Before any wide spend, a single-model "confidence gate" run (Qwen3-30B-A3B via
the fully redesigned self-gen harness against the frozen original
12-conversation corpus) was run to validate the rebuilt apparatus end-to-end. It
failed: `build_alignment_direct` raised because the write-time summary span (899
tokens) did not match the compacted B-context summary region (538 tokens) — a
genuine misalignment, not tokenization drift. Root cause: the direct-span
alignment refactor (from the earlier "token matching" scare) had only been
equivalence-verified against **fixed** summaries, never against
**self-generated** ones, which is the actual production path the redesigned
sweep depends on. This is recorded as incident 32, with the lesson that
apparatus equivalence must be verified against the actual production
configuration, not a proxy configuration. A debugger subagent was dispatched
with pod access (model cached, ~2 min per iteration) to find the root cause and
prove the fix by reproducing the known +0.14 referent positive control —
explicitly barred from a silent difflib fallback or skipping the failing
conversation. At conversation's end this fix was in progress (uncommitted edits
to `arms_common.py`/`cross_arch_probe.py`), with a positive-control verification
run (12 conversations, self-gen, Qwen3-30B) live on the GPU (47% utilization).

The user's response to this failure was to request faster failure detection:
since this specific bug was a pure tokenization/CPU-level issue requiring no GPU
or model load, a CPU-only tokenizer smoke test could have caught it in seconds
before any pod spend. This was adopted as standing practice — a `--smoke-align`
CPU-only companion check (verified to fail on the broken code and pass after the
fix) becomes a mandatory pre-flight step before any pod launch, and more
generally, checks are now to be layered cheapest/fastest-first: CPU smoke tests,
then a tiny (~0.6B) model full-path run, then the real model positive control
last.

## Operational and process additions

An idle-pod watchdog was added at the user's request: a 5-minute-interval check
across all pods (queried fresh from the RunPod API each cycle) alerting only
when a pod shows both GPU <10% utilization and no active job process — designed
not to false-trigger during legitimate low-GPU activity like model downloads.

Codex CLI (`@openai/codex`) was installed and configured after the user
clarified they wanted the assistant to run it autonomously, not ask the user to
do so; it authenticated automatically via existing ChatGPT login, and
`~/.codex/config.toml` already defaulted to
`model=gpt-5.5`/`model_reasoning_effort=xhigh`, so the standing invocation is
`codex exec -s read-only "..."`. This is now the required review step at least
once per shareable major paper revision (not repeated routinely), run directly
by the assistant, alongside the existing Fable multi-perspective review passes.

The user indicated Fable may be given more latitude to do initial drafting of
the paper's final version directly (not just review/critique), with the
assistant still handling iteration, review-incorporation, and any changes judged
necessary — this is a forward-looking process note for when the paper reaches
its writing phase.

The user requested and received per-model cost estimates for the sweep: roughly
$1.15-1.30/model (dominated by ~20 min download+load plus self-gen compute), giving ~$17-24
for the original 14-model/27-conversation plan, with corpus-doubling adding only
~$12-16 more since compute (not model loads) scales with test count.

## Final model set (as of conversation end)

Through several rounds of user pushback on vendor/version diversity (challenging
an initial Qwen-heavy list, questioning the absence of Llama-2 — deliberately
excluded per Fable's advice as a scale/era confound despite being the only
available full-MHA/GQA-ratio-1 anchor — and catching a Gemma-4 existence check
the assistant initially got wrong on model size), the frozen set grew to 16
models across roughly 9-10 vendors, including two independent within-vendor
dense/MoE anchor pairs (Qwen3-30B-A3B/Qwen3-32B, and newly-discovered
Gemma-4-26B-A4B MoE/Gemma-4-31B dense), a Qwen2.5-32B cross-generation dense
anchor, a Gemma-3-vs-4 generational comparison, non-Qwen MoE/dense replication
(Mixtral, Mistral-Small), a geometry outlier (Gemma-3's sliding-window
attention), open-training and breadth points (OLMo-2-32B, Qwen3.6 variants,
GLM-4-32B), and new-vendor breadth additions (OpenAI gpt-oss-20b, Microsoft
phi-4, 01.ai Yi-1.5-34B, NVIDIA Nemotron-49B, including both v1 and v1_5
Nemotron checkpoints and both 2501/2506 Mistral-Small checkpoints as
within-model training-checkpoint controls). Gemma models carry acknowledged
architectural risk (sliding-window HybridCache may make the graft structurally
UNSUPPORTED). The user directed that riskier/newer architectural gambles (e.g.,
very recent releases) be run in a second wave after banking results from
lower-risk, well-understood models first.

## State at conversation boundary

The wide sweep launch remains held pending: (1) the alignment-bug fix being
verified via a reproduced +0.14 positive control on the pod (in progress), and
(2) the corpus reaching its doubled target of 54 conversations via six
freshly-relaunched parallel authoring batches (in progress, not yet merged into
`scenarios.json`). All pods apart from the single gate pod are currently down,
avoiding idle burn during this build/debug phase. Numerous decisions, incidents,
and lessons from this conversation (the alignment scare resolution, the
fixed-summary mechanism, the Fable design reframe, the champion-scan grand
insight, the MLX-to-foundation-API switch, the sequential-vs-parallel-authoring
incident, and the gate-catches-a-real-bug incident) have been recorded in the
project's STATE/FINDINGS/DECISIONS/INCIDENTS tracking documents and, where
transferable beyond this project, in persistent cross-session memory.

---