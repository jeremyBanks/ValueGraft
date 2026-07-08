_This conversation covers a shift from exploratory theory-review into a
paper-writing and repo/paper-hygiene phase for the ValueGraft
compaction-mitigation project, culminating in a private GitHub repo, a synthesis
workspace with two paper drafts, and a naming/prior-art overhaul._

**Participants:** User, gpt-5.5-xhigh, and gpt-5.5-high.

**Reframing consolidated.** Early messages settled the project's central
framing: the claim "re-encoding retained text after compaction is not equivalent
to preserving write-time cache state" is treated as self-evident and explicitly
_not_ a research contribution — it is only a baseline/denominator used to size
compaction damage. The valuable question is whether a small, deployable
cache-state intervention can measurably reduce that damage. This correction was
made repeatedly and should be treated as a firm, standing constraint on all
future writeup work: never present the compaction-damage baseline as a finding
or contribution.

**Terminology fixes.** "Activations" was rejected as too broad; the correct
terms are cached attention value vectors / cached key vectors / write-time value
state, with "KV cache" acceptable as the standard mechanism name but requiring a
caveat that this is not a caching-as-speed-optimization study — the cache's
contents themselves are being modified.

**Two narrowed experimental arms.** The project narrowed from a large arm matrix
(A/B/C/D/E/H/B-min etc.) to two central comparisons: H-pack (renamed away from
that label per the final instruction below) — reuses the summary's write-time KV
state from when it was generated under full context, repositioned/re-rotated
into a packed compacted layout, compared against a fresh encoding of the same
packed summary (B-min-pack) — and ValueGraft — takes the ordinary
production-shaped compacted context (summary + tail), keeps freshly computed
keys, and linearly blends in old value vectors at exact-token-aligned positions
(`V_final = (1-alpha)*V_fresh + alpha*V_old`), compared against plain compaction
(B). H-pack's main signal is reduced fabrication/more honest abstention;
ValueGraft's main signal is a small but statistically significant
continuation/next-action-likelihood gain (e.g. +0.0156 nats on SWE-Gym traces,
45/75 wins, CI ~[.005,.027]). A 30B per-slot/positive-profile mask showed a
larger gain (+0.0384 vs +0.0239 global alpha) but was deliberately kept as
"exploration," not the headline method, per an explicit calibration-discipline
policy: keep the primary claim on simple global alpha, treat per-head/per-slot
tuning as secondary until guard checks pass. Negative/extrapolative alpha (>1,
contrastive steering) and negative coefficients were tested and did not beat the
simple mid-band rule.

**Honesty-vs-refusal distinction.** A key methodological gap was identified:
reduced fabrication could reflect either genuine calibrated honesty or generic
increased caution/refusal. Existing evicted-fact and decoy probes only test the
abstain side; LongMemEval-style answerable-question data is needed to test the
answer side. This calibration-matrix framing (answer when evidence present,
abstain when absent) was adopted as the way to distinguish the two explanations,
and is not yet fully resolved — early LongMemEval samples showed A answering
correctly while B and H-pack both abstained, which is consistent with caution
rather than proven calibration.

**Deployment/production framing.** Raw KV sidecar transport (tens of MiB per
compacted conversation) was judged too expensive for ordinary stateless API
payloads. The more realistic shape is a provider-side opaque compaction
handle/artifact (analogous to prompt-cache handles, encrypted
reasoning/thought-signature tokens, or session IDs), attached as metadata to a
visible summary rather than embedded as a literal text token, with an explicit
`compact_from(boundary, instructions) -> {summary_text, compaction_token}`
primitive. Research triggered by this (see below) found this is closer to
reality than assumed.

**Prior-art escalation.** A significant real-world discovery: current provider
APIs already expose structures that are shaped similarly to the paper's proposed
production mechanism — OpenAI's Responses API has an explicit
`context_management`/`compact_threshold` and a `/responses/compact` endpoint
returning an opaque encrypted compaction item; Anthropic has a beta compaction
feature emitting a typed `compaction` block with a text summary (their separate
prompt-caching docs do describe KV-based caching internally); Gemini has
adjacent context-caching and encrypted "thought signature" continuity plus
automatic compaction in its Managed Agents product, without a public
general-purpose `/compact` text primitive. This was investigated carefully to
distinguish documented interface facts from mere inference — none of the
provider docs confirm they preserve raw KV/value tensors or anything like
ValueGraft's blending mechanism; the overlap is in problem/interface shape, not
confirmed mechanism. Open-source search found only interface-level adapters
(e.g. `lmctx`, which round-trips opaque provider compaction artifacts) with no
public KV-surgery implementation matching this project. Academic literature
search surfaced strong mechanistic neighbors — "Models Take Notes at Prefill"
(KV cache editability/composability), "Fast KV Compaction via Attention
Matching" (closest latent KV-compaction paper), "Parallel Context Compaction for
Long-Horizon LLM Agent Serving" (same deployment problem, text-space not
latent-state), plus KVLink/CacheBlend/SamKV as cache-reuse/blending neighbors —
none matching the exact summary-boundary write-time-value-preservation
experiment. This full analysis was written to a new document,
`provider-compaction-prior-art-review.md`, committed at the repo root (commit
`7f9aca7`).

**Paper drafting workflow and hygiene rules established:**

- A dedicated, clearly self-labeled workspace was created at
  `paper-working/valuegraft-synthesis/` with a README explicitly marking it as
  personal derived analysis, not source-of-truth experimental data, so other
  agents working elsewhere in the repo would not be confused.
- Only files inside that directory are edited/committed by this synthesis work;
  all other in-flight experiment files/results are left untouched even as they
  change concurrently.
- Git workflow was tightened per explicit correction: never pre-stage files
  speculatively; for tracked files commit directly by path
  (`git commit -- path`); for new files, add only the exact path immediately
  before committing, verify the staged set, then commit — minimizing risk of
  touching unrelated concurrent work.
- Two paper drafts exist: `valuegraft-paper-draft.md` (the original wide
  synthesis, preserved untouched, includes provider-API and
  calibration-philosophy tangents) and `valuegraft-focused-draft.md` (a tighter
  ~300-line version with one narrative spine: baseline gap as denominator → two
  interventions → what they recover/don't → limitations, with side material
  trimmed to brief notes and a short related-work section retained for academic
  expectations).
- Multiple editorial passes followed: removing "compaction hurts recall" framed
  as a finding (retitled section to "Baseline Gap Used for Evaluation," reframed
  abstract and body so this is explicitly not a contribution); removing
  AI-writing tics (ghost-objection caveats, generic "this is important"
  scaffolding, defensive "not X but Y" phrasing); adding proper author-year
  citations with links/arXiv/DOI in place of title-only mentions, applied to
  both drafts; and substantially expanding the methodology sections (compaction
  boundary selection, summary generation, arm construction for
  A/B/B-min-pack/H-pack/ValueGraft, key re-rotation, exact-token
  alignment/blending, tuning, negative controls, scoring) since methodology had
  been condensed to near-nothing relative to results.
- The private GitHub repo `jeremyBanks/ValueGraft` was connected as `origin` and
  `trunk` pushed and kept in sync across the citation and methodology commits;
  tracked-history was checked for leaked secret patterns before pushing (none
  found; `.runpod_key`/`.huggingface_key` remain locally git-ignored via
  `.git/info/exclude`, not the tracked `.gitignore`).
- Authorship for the eventual paper: Jeremy Banks (first author), Claude Fable 5
  (second author), the GPT-5.5/Codex agent (third author).

**Naming correction (open at conversation boundary).** The final instruction of
this conversation: the names "H-pack" and "ValueGraft" read as unrelated
concepts despite being closely related variants of the same idea, and "H-pack"
reads like a leftover placeholder. The user wants "ValueGraft" to become the
umbrella name for the overall approach, with the two current variants renamed as
sibling methods/variants (e.g., lettered, numbered, or newly named) under that
umbrella. This is a paper-writing-only change — the assistant may rename in the
draft without touching source code or other repo files, and does not need to
keep code and paper terminology in sync. This renaming was in progress,
unresolved, at the end of the conversation.

---
