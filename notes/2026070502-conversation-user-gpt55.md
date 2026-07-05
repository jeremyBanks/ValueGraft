_This conversation, conducted with a codex/gpt-5.5 agent operating under a
strict no-write policy except for isolated, explicitly-approved notes files,
tracked the KV-cache-compaction experiment across several days: orientation and
hand-off review, a critical reframing of the research question, brainstorming on
adaptive value-blending and production deployment shape, and a deep prior-art
investigation into hosted-provider compaction APIs._

**Participants:** User, gpt-5.5-xhigh, and gpt-5.5-high.

Orientation and review passes

The agent was repeatedly invoked in read-only mode to orient on repo state
without disturbing another agent actively working in the directory (`kvlib.py`
for cache snapshot/rebuild, `arms.py` for A/B/C/D/E/H arm construction,
`run_arms.py` as the driver, `supplement_arms.py` for B-causal/negative
controls, `score.py` for leakage-aware scoring). Across three review passes it
flagged several concrete issues later fed back to the working agent:
`score.py`'s local-fallback scoring path had drifted from the external-judge
path (local still folds in `kw_pass`, external trusts judge verdicts only);
`STATE.md` periodically fell stale relative to actual progress (30B run status,
LongMemEval activity); `analyze.py`'s per-conversation appendix omitted
newly-central arms (B-causal, B-min, H-gap, E-post-a0.25); the LongMemEval
runner writes final JSON non-atomically (risk of corrupt files on interruption)
and lacked a scoring/judging layer or an "A-success" gate before comparing arms;
and a hard-coded local HF cache path that would break portability. Checks
confirmed that reported headline numbers (2,520 score rows, 2,089 judge
verdicts, B-causal CONT repair) were internally consistent with committed JSON
before any of these issues were raised.

Reframing the research question

A central methodological correction emerged: the original framing — "does
write-time KV cache state preserve semantic continuity better than re-encoding
text" / "does re-encoding destroy context-conditioned state" — was judged
self-evident and not worth trying to prove; it is a premise, not a finding. The
refocused framing is: given that compaction destroys context-conditioned state,
can a small, practical cache-state intervention (value-vector blending,
packed/re-rotated summary KV retention, etc.) measurably reduce the resulting
behavioral damage, in a way that is deployable, cheap relative to full-context
retention, and not explained by leakage or generic smoothing (validated via
wrong-conversation/shuffled negative controls)? Practical implication: `C`
(gapped-cache retention) is now secondary/diagnostic rather than central;
`H-pack`/`SelfGist` (retained summary-only KV) and low-alpha `E-post` (value
grafting) become the primary mitigation contrasts; `G`/SoftGraft
(retrieval-weighted grafting) is a deferred generalization path. A copy-paste
instruction was drafted for the working agent: finish only the smallest coherent
chunk of in-flight work, commit, update STATE.md/DECISIONS.md/RESULTS.md, then
pivot fully to the mitigation-first design rather than continuing to expand the
old arm matrix.

A related open methodological question, only partially resolved: current results
showing "less fabrication" under H-pack/H-gap conflate calibrated honesty
(abstaining only when evidence is truly absent) with generic refusal/caution
(abstaining broadly regardless of evidence). The LongMemEval-style benchmark,
because it has genuinely answerable memory questions, was identified as a way to
test the missing half — whether the same interventions preserve or improve
accuracy when evidence is present, not just increase abstention when it's
absent. A calibration matrix (answerable-from-summary/tail vs. evicted/decoy
items) was proposed as the way to disambiguate.

Mechanism and blending ideas

Discussion clarified precise terminology: the correct term for what's being
manipulated is "cached attention value vectors" (or "KV-cache value tensors"),
not "activations" (too broad) or unqualified "KV values" (imprecise). In the
ValueGraft arm, keys remain fresh; only value vectors are blended/replaced at
aligned token positions using `V_final = (1-alpha)*V_fresh + alpha*V_old`, with
alpha as a single global scalar currently. A follow-on idea, captured as
speculative notes (not adopted into the active plan): tune alpha
per-layer/per-KV-head rather than globally, since a single global alpha is
likely a blunt compromise across heads with different retrieval/semantic roles;
a more advanced version (adaptive/confidence-weighted gating using
attention-based match confidence rather than raw value-space cosine similarity)
was discussed as the conceptual precursor to the planned `G`/SoftGraft
direction, not yet implemented.

Production deployment framing

Sizing showed that raw KV sidecars are impractically large for ordinary
stateless API payloads (~96 KiB/token for a 30B model, so even a modest summary
is tens of MiB) — far larger than the text they represent, so this is not a
lightweight bolt-on. The more realistic production shape is a provider-owned
opaque compaction handle/token, analogous to already-shipping patterns (OpenAI
encrypted reasoning content, Anthropic prompt caching, Gemini context caching):
an explicit
`compact_from(boundary, instructions) -> {summary_text, compaction_token}`
primitive, with the token as structured metadata (ideally including a
scoped/signed context digest for cache convergence) rather than a literal
in-content token, and with TTL/versioning/billing semantics similar to prompt
caching. This was scoped explicitly as a lightweight deployment/product note,
not a research branch — the experiment itself remains single-process and
in-memory, with the transport/handle design treated as an established-pattern
implementation detail.

Prior-art investigation

A user-flagged concern — that hosted frontier APIs (OpenAI Responses
`context_management`/`/responses/compact` with an opaque encrypted compaction
item; Anthropic's beta server-side compaction block; Gemini's encrypted "thought
signatures" and managed-agent auto-compaction) already look externally similar
to the proposed opaque-handle shape — triggered a deep, explicitly non-lazy
research pass, written up in a new committed document
(`provider-compaction-prior-art-review.md`). Key finding: documented interfaces
show all three labs exposing some form of opaque, provider-side continuity state
across compaction/long-running interactions, which suggests but does not
establish that they preserve anything resembling raw KV/cached-value state —
providers do not document doing the specific value-vector grafting/blending
mechanism under study. Literature search surfaced closer mechanistic neighbors
than initially assumed: "Models Take Notes at Prefill" (KV cache is directly
editable/composable — largely removes novelty from the general claim that KV
carries write-time state), "Fast KV Compaction via Attention Matching," and
"Parallel Context Compaction for Long-Horizon LLM Agent Serving" (same
deployment problem, but text/token-space rather than latent-state preservation).
Open-source search found interface-level adoption of provider compaction
artifacts (the `lmctx` project round-trips OpenAI/Anthropic opaque compaction
blobs) but no public implementation of KV value-vector grafting. Conclusion:
novelty should be relocated away from "KV contains meaning" or "opaque handles
are a novel API shape" (both are now reasonably well-precedented) and toward the
specific mitigation experiment — whether a compaction-time cache-state
intervention measurably reduces compaction damage relative to text-only
summarization, evaluated under leakage controls and, eventually, real
coding-agent tasks.

Interim empirical results and remaining work

At time of writing, a Stage 2 result showed tuned ValueGraft improving SWE-Gym
next-action log-probability by ~0.0156 nats across 75/75 traces (winning 45/75,
CI roughly [.005, .027]) — a small but statistically consistent positive effect
on a fragile proxy task. The explicit caveat recorded: proxy fragility means the
small magnitude is not evidence the real-world (coding-agent) effect is small;
it has not yet been tested on real coding tasks, which remains the decisive
follow-up. Operational notes from this period also include: RunPod and Hugging
Face API keys were stored in git-ignored local files (`.runpod_key`,
`.huggingface_key`, via `.git/info/exclude`, permissions 600, no whitespace); a
cloud/server stack review (`latest-cloud-stack-review.md`) flagged ignoring
`.pod_state.json`, making HF LongMemEval writes atomic/resume-safe,
strengthening `score_lme.py`, and sanity-checking HF `H-gap` suffix semantics
against MLX before further cloud spend. A recurring heartbeat/check-in schedule
(hourly, 45-minute idle threshold, to auto-draft and commit data-only reports)
was set up and later cancelled as premature given that runs were taking longer
than expected.
