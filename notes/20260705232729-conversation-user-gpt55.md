_The project matured from proxy evidence and prior-art analysis into a
disciplined ValueGraft study with an academic-paper-style writeup, clearer
method definitions, and active real coding-agent evaluation. Current work is
focused on serving-shim integration and E1 coding tasks; the decisive practical
result remains pending._

**Participants:** User and gpt-5.5-xhigh.

**Research framing.** The premise that text compaction loses context-conditioned
state is treated as an expected baseline, not a discovery or contribution. The
actual question is whether compaction-time preservation or reconstruction of
cached attention state reduces the resulting behavioral loss. Current proxy
evidence shows statistically significant but small positive effects; the proxy
is fragile and low-bandwidth, so its magnitude cannot be extrapolated directly
to coding performance. Real coding-agent runs are the required next test.

**Methods and terminology.** The narrowed comparison consists of two sibling
variants under the ValueGraft umbrella:

- `ValueGraft-Pack` (formerly H-pack): retain summary tokens’ write-time cached
  key/value state from summary generation, move it into a compact contiguous
  layout, re-rotate keys for new RoPE positions, and leave values unchanged.
- `ValueGraft-Blend` (formerly plain ValueGraft): freshly encode the ordinary
  summary-plus-tail context, retain fresh keys, align literally matching tokens
  between old and compacted contexts, and replace/blend fresh value vectors with
  old value vectors using `V_final = (1-alpha)V_fresh + alpha V_old`; alpha may
  exceed 1 for extrapolation.

“Cached value vectors,” “cached attention state,” and “KV-cache value tensors”
are the preferred technical terms. “KV cache” is precise but should be clarified
as behavior-relevant stored attention state, not merely a performance cache.
Exact alignment requires identical tokenization and matching summary/tail
regions; no fuzzy semantic matching is used. Per-slot/head tuning remains
exploratory, while the primary reported method should remain simple global alpha
per model unless guard checks justify otherwise.

**Prior art and production shape.** `provider-compaction-prior-art-review.md`
was created and committed as `7f9aca7`. Hosted APIs now expose compaction
artifacts or related opaque state surfaces, especially OpenAI Responses
compaction, Anthropic compaction blocks, and Google’s context/thought-signature
systems. Public documentation does not establish that any provider uses raw KV
preservation or the exact ValueGraft mechanism. Open-source investigation found
interface-level adapters such as `lmctx`, but no clear public implementation of
ValueGraft-style cache surgery. Academic neighbors include KV
editing/composition, latent KV compaction, cache blending/reuse, and agent
context compaction; none matches the summary-boundary mitigation experiment
exactly. The defensible novelty is therefore the intervention and evaluation,
not opaque handles, KV editability, or the general claim that state beyond
visible text can matter.

**Writing artifacts.** The isolated workspace
`paper-working/valuegraft-synthesis/` contains a wide reference draft, evidence
ledger, and a focused draft. The focused draft was repeatedly revised to remove
conversational/AI-generated prose, treat compaction loss only as an evaluation
denominator, expand reproducible methodology, adopt the
ValueGraft-Pack/ValueGraft-Blend naming, add author-year linked citations, and
include the GitHub code-availability URL. Relevant commits include `622dbc1`,
`be60b46`, `0cd6a4c`, `70081ee`, and `cf3d67d`; the repository was pushed
through `cf3d67d` before subsequent local work. The intended form is a concise,
genuine academic-style empirical paper or technical report, not a casual blog
post.

**Current handoff state.** The latest read-only refresh found a clean worktree
with local `trunk` 11 commits ahead of `origin/trunk`; latest local commit is
`57af41e`. Recent work primarily hardened Gemma templates, serving
infrastructure, pod launchers, and LongMemEval runners. The OpenAI-compatible
serving shim now exposes `sc-A`, `sc-B`, and `sc-E` modes with server-side
compaction and ValueGraft. E0 reportedly completed an end-to-end agent task with
tests passing. An E1 OpenHands run is active in mode B through the
localhost:8010 shim/tunnel; the current `t1_B` log shows edits to
`src/pricing.py`, but no final score exists yet. Summary-cache behavior was
observed as MISS/HIT/HIT with stable graft counts, with deployment intended
after the E1 queue drains to avoid changing semantics mid-run. Recheck that
comments and implementation agree on whether summaries represent only the
evicted region or are generated from the full message list before final
reporting. The previously configured hourly false-idle monitor was explicitly
cancelled; no replacement timing commitment was established.

## Conversation sources

- `019f3007-bab0-7e50-b019-2625d1538f63`
