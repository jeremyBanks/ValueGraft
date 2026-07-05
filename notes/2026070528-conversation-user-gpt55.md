_This chunk covers a mainline discussion of interpreting in-flight
ValueGraft/compaction experiment results, a scope correction on how the
compaction-damage baseline should be framed, an extensive prior-art
investigation into hosted-provider compaction APIs, a full paper-style synthesis
and revision cycle, arm renaming, publication-channel questions, and a
tangential thread about a prior CHI 2018 publication and arXiv policy — followed
by a repo-state refresh describing active E1 coding-agent infrastructure._

**Participants:** User and gpt-5.5-xhigh.

Participants in this chunk were Jeremy Banks (user) and Codex CLI (assistant,
model gpt-5.5, provider openai, codex_cli 0.142.5, effort xhigh) throughout.

**Interim results interpretation.** Early in the chunk, a per-trace Stage 2
result was reviewed: tuned ValueGraft improved SWE-Gym next-action prediction by
roughly +0.0156 nats across 75/75 traces, winning 45/75, CI approximately [.005,
.027]. This was characterized as a promising but still in-flight signal, not a
settled result. A recurring heartbeat check (`check-back-after-60-minutes`) was
cancelled at the user's request because the run was expected to take longer than
planned and would otherwise produce a false-positive "stalled" trigger.

**Framing correction (load-bearing for all future writing).** The user
established a firm, repeated correction: "compaction destroys/degrades
context-conditioned state" is not a research question or a contribution — it is
an assumed, self-evident premise, used only to establish a baseline/denominator
against which mitigation methods are measured. Any document phrasing that
presents this as a "finding" or "result" is incorrect and must be rewritten so
the baseline gap reads as evaluation scaffolding, not a contribution. This
correction was applied concretely later to the focused paper draft (retitled,
abstract and section renamed, e.g. "Compaction Damage Is Real but Not the Novel
Claim" became "Baseline Gap Used for Evaluation"). The sharper framing going
forward: given that compaction destroys context-conditioned state (assumed), can
a small, practical compaction-time state artifact reduce the resulting
behavioral damage, and does that gain survive real coding-agent tasks. A
small-but-statistically-significant improvement on the current best (fragile)
proxy task should not be read as evidence the real-world effect is small, since
the proxy has low dynamic range; the decisive open test remains real
coding-agent performance, not yet run.

**Terminology clarifications.** "Activations" was rejected as too broad; the
precise terms are cached attention value vectors / KV-cache value tensors /
write-time value state (for values) and cached key vectors (for keys). "KV
cache" remains the correct technical term despite sounding performance-only; the
clarification for writing is that normally the cache is behavior-preserving, and
this project's intervention deliberately changes which value tensors are stored,
not just how fast they're retrieved.

**Method descriptions (now written into paper methodology).** Two narrowed
experimental arms were clarified and later renamed for the paper (code/repo
naming was left untouched; renaming was documented as paper-only, done inside
the isolated synthesis directory):

- Formerly "H-pack," now **ValueGraft-Pack**: the summary is generated under
  full old-context visibility; the summary tokens' write-time KV cache is
  preserved and repositioned (packed) into a compact contiguous prefix; keys are
  RoPE-re-rotated to new positions, values unchanged. Compared against a matched
  fresh-encoded control, renamed **FreshPack** (formerly "B-min-pack"). Main
  observed effect: reduced fabrication / more honest admission of uncertainty,
  not recall recovery.
- Formerly plain "ValueGraft," now **ValueGraft-Blend**: standard compacted
  context (summary + retained tail) is freshly encoded with fresh keys; for
  exact-token-aligned positions, fresh value vectors are blended with old
  full-context value vectors via `V_final = (1-alpha)*V_fresh + alpha*V_old`,
  with alpha in [0,1] as blending and >1 as contrastive extrapolation. Keys
  remain fresh. Main observed effect: continuation/next-action likelihood
  improvement (the SWE-Gym proxy result above). "ValueGraft" was redefined as
  the umbrella family name for the paper, per explicit user instruction, with
  the two arms as named variants.

**Prior-art investigation (major work item).** Following a user directive to
treat the topic with high priority — triggered by discovering that OpenAI's
Responses API already has an explicit `context_management`/`compact_threshold`
mechanism and a `/responses/compact` endpoint returning an encrypted opaque
"compaction item," and that Anthropic has a beta server-side `compaction` block
— a deep, carefully evidence-graded investigation was conducted and written up
as a standalone repo document, `provider-compaction-prior-art-review.md` (commit
`7f9aca7`). Findings were explicitly stratified into documented interface,
documented semantics, and inference: OpenAI's compaction item is documented as
encrypted/opaque and "carrying forward key state and reasoning" but not
documented as literal KV/value-tensor preservation; Anthropic's public
compaction is text-summary-based, while its separate prompt-caching docs do
describe KV-based caching; Gemini has adjacent encrypted "thought signatures"
and context caching but no public general compact/summary primitive found.
Literature/open-source sweep found no exact prior implementation of the
summary-boundary value-state mitigation approach, but strong mechanistic
neighbors: "Models Take Notes at Prefill" (KV editability/composability), "Fast
KV Compaction via Attention Matching" (closest latent KV-compaction paper),
"Parallel Context Compaction for Long-Horizon LLM Agent Serving" (same
deployment problem, text-space not latent), KVLink, CacheBlend, SamKV (cache
reuse/blending, different boundary), and an open-source adapter project `lmctx`
that round-trips provider compaction artifacts as typed objects without doing KV
surgery. Conclusion: frontier labs plausibly explore this internal problem space
(their APIs suggest state-beyond-visible-text handling), but no public evidence
confirms they use this project's specific value-tensor mechanism; the defensible
novelty claim is the summary-boundary mitigation experiment and its evaluation,
not "KV is editable" or "opaque state handles" generally, both of which are
prior art.

**Production API shape critique.** The proposed production shape
(`compact_from(boundary, instructions) -> {summary_text, compaction_token}`,
opaque provider-side handle attached to a visible summary) was judged realistic
only as a provider-owned opaque state handle, analogous to OpenAI's encrypted
reasoning content, Anthropic's prompt-caching breakpoints, and Gemini's context
caching — not as a raw client-managed KV sidecar, which was judged a non-starter
for audit/privacy/debugging reasons. Recommended representation is structured
metadata (e.g., a `compaction_handle`/`source_digest`/opaque `state` object)
rather than a literal in-content text token, since the artifact is execution
state, not language. Viability depends on TTL/billing/model-version binding,
similar to prompt-cache semantics. The overall framing recommended for the
paper: this is a compaction-time cache artifact, not a new general memory layer.

**Paper drafting and repo/document workflow.** Per explicit user direction, an
isolated working directory `paper-working/valuegraft-synthesis/` was created
with a README marking it as personal derived analysis, not source-of-truth
experimental data, so other agents/processes would not treat it as
authoritative. Authorship was fixed as Jeremy Banks (first author), Anthropic
Claude Fable 5 (second author), and the assistant/OpenAI GPT-5.5 (third author)
— the user's earlier idea of listing AI models as primary authors was dropped.
Work proceeded through: an evidence ledger (`evidence_notes.md`), a wide first
full draft (`valuegraft-paper-draft.md`, committed and later revised for a
larger Stage-1 LongMemEval n=320 aggregate showing large compaction damage but
flat arm performance in QA framing, and for a demo/deleted-artifact data-hygiene
rule), and — per user request to cut unrelated tangents (provider API detail,
calibration philosophy, deployment overhead, provenance) — a second, tighter
file `valuegraft-focused-draft.md` (kept as a separate file, original wide draft
preserved unchanged). The focused draft underwent several rounds of revision:
removing the "compaction damage as a finding" framing per the correction above;
a general "AI-writing" editorial pass removing defensive/ghost-objection
phrasing and generic scaffolding; substantial expansion of the Methods section
(compaction boundary selection, summary generation, arm construction for
A/B/FreshPack/ValueGraft-Pack/ValueGraft-Blend, exact-token alignment, tuning,
negative controls, scoring) after the user flagged that methodology was
compressed to near-nothing; formal author-year citations with links replacing
bare title mentions, applied to both drafts; the
ValueGraft-Pack/ValueGraft-Blend/FreshPack renaming; and a final "Code
Availability" section linking the GitHub repo, added after the user confirmed no
blind review is expected.

**Git workflow correction (durable rule).** The user established an explicit
safer git protocol going forward: never pre-stage files speculatively; for
already-tracked files commit directly by explicit path (`git commit -- path`);
for new files, add the exact path only immediately before committing, verify the
staged set with `git diff --cached --name-status`, then commit by explicit
pathspec; never use `git add .`/`git add -A`/directory-wide staging in this
repo, given other processes are concurrently modifying unrelated files.

**Publication channel decisions.** A private GitHub repo,
`https://github.com/jeremyBanks/ValueGraft`, was created by the user; the full
local `trunk` branch was pushed there as the repo's initial and ongoing
publication channel (commits progressed from `be60b46` through later pushes,
most recently `cf3d67d Add code availability to focused draft`, with a separate
methodology-expansion commit `0cd6a4c` pushed on request). The user decided,
after discussion of arXiv's tightened 2026 endorsement policy and CS
review/position-paper moderation changes, and of Zenodo's differing role
(artifact/DOI archiving, not paper discovery), that GitHub alone will likely be
the only publication channel used; no arXiv or Zenodo submission is currently
planned. A tangential subthread covered a prior CHI 2018 paper the user
coauthored (DOI 10.1145/3173574.3174182) and whether/how it could be reposted to
arXiv; the user ultimately decided not to pursue this, given the administrative
overhead of coauthor consent for what is a peripheral authorship credit — this
does not affect the ValueGraft repo or its documents.

**Handoff state at chunk end (from final read-only refresh).** Worktree clean;
local `trunk` was 11 commits ahead of `origin/trunk` at
`57af41e Fast crash/error detection (5-min podcheck) + launcher syntax gates`
(not yet pushed at chunk close). Recent infrastructure-focused commits
(concurrent, non-paper work by other process/agent activity) included Gemma
tokenizer/template fixes (role-alternation merging, Gemma-safe context
construction, dry-run gating before spending pod compute), substantial
advancement of `src/serve_shim.py` (OpenAI-compatible `sc-A`/`sc-B`/`sc-E` modes
with server-side compaction and ValueGraft, with E0 reported as a successful
full end-to-end agent run through the shim with tests passing, and a verified
summary cache showing stable MISS/HIT/graft-count behavior, intended to deploy
only at E1 queue drain to avoid mid-run semantic drift), an active E1
coding-agent run (local OpenHands SDK driver executing mode B against
`localhost:8010`, tunneled to an `e1` pod, with an in-progress edit to
`src/pricing.py` and no final score yet), a safer LongMemEval full runner
(per-question OOM skip, `SC_MAX_FULL=110K` cap), and a safer pod launcher
(pre-launch registration, job-shell and Python AST syntax gates). One
methodology inconsistency was flagged as needing future cleanup: code comments
in `serve_shim.py` and `run_lme_full_hf.py` describe summarizing only the
"evicted/older region," but the actual implementation appears to still generate
the summary from the full message list before appending the tail separately —
this discrepancy between comment and behavior should be reconciled before final
reporting. The decisive open test — real coding-agent task performance (E1) —
was in progress but not yet resolved at the end of this chunk.
