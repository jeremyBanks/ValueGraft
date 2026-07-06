# Write-up guidelines (final artifact: HuggingFace community blog post + repo)

Target: HF community blog post + this GitHub repo — NOT an academic paper.
Underlying gist, written straight: "This seems like an obvious thing to try,
but I couldn't find it in my literature search — am I missing the right
terminology, or is it so trivial/inconsequential that nobody wrote it up?
Either way, here's what I observed."

1. **Provenance (in the body, not the byline):** experiment design,
   implementation, and most of the writing by Anthropic's Claude Fable 5,
   directed by the user, who made scaling/budget decisions and commissioned
   an adversarial review from OpenAI's GPT-5.5 (source of the external-review
   amendments, the mitigation reframing, and the blog-post target framing).
   "Designed and largely executed by Claude Fable 5, with adversarial review
   by GPT-5.5, directed and sanity-checked by me." Be explicit about what the
   user personally verified vs. couldn't.
2. **Findability is the primary goal.** Front-load naive plain-language
   phrasings ("summary tokens lose their original activations after context
   compaction") in title/opening/README alongside proper terminology. Give
   the technique one simple name (e.g. "dual-context KV blending") used
   consistently. Related-work section: exact terms/papers searched and ruled
   out (gist tokens, KV cache merging / CaM / KVMerger, Activation Beacon,
   compressed context memory, CacheBlend, KVLink, StreamingLLM/H2O, "Models
   Take Notes at Prefill" etc.) with one line each on why they're not this.
   Do a real (web) literature pass before finalizing this section.
3. **Report results as they are** — clean negatives with the ablation table
   are equally valuable.
4. **Structure:** mechanism first → ablation matrix + headline numbers
   (framed as % of the full-context gap recovered) → limitations →
   open question to readers ("is this known? is the setup sound?").
   Include compute/latency overhead (double prefill for grafting; summary
   generation for SelfGist).
5. **Tone:** "here's a thing I measured." Not apologetic, not overclaiming;
   limitations matter-of-fact.

## Canonical arm names (user-set, 07-06 — use these in ALL human-facing text)

| internal id | canonical name | formal (reframing-doc) name |
|---|---|---|
| A | **Original** (no compaction) | — |
| B | **Compacted** | Plain Summary Compaction |
| E / E:a0.75 | **Compacted + value graft (α=0.75)** | V-only Graft, α_V=0.75 |
| E:a1.0 | **Compacted + value graft (α=1.0)** | V-only Graft, α_V=1.0 |
| E:cfg=layers | **Compacted + layer-tuned value graft** | Layer-tuned V-only Graft |
| E:shuf | Compacted + shuffled graft (negative control) | corrupted-alignment control |
| E:a-0.5 | Compacted + inverted graft (negative control) | α_V<0 extrapolation control |
| B-min-pack | Packed summary (fresh KV) control | Packed Fresh-KV Control |
| H-pack | Packed write-time-KV | Packed KV-Graft (layout differs — see reframing doc) |

Internal ids stay frozen in code/specs/result files for provenance; every
table, report, and figure translates. Reports/messages to the user should
use canonical names by default.

## Tier names (translate in all report text)
- internal "tier-0" → "aggressive compaction (stress setting)": compact_at
  9000, tail 2500, detail-free summary — mechanism-era settings, labeled
  stratum, never headline.
- internal "humane tier" → "production-calibrated compaction": compact_at
  12000, tail 6000, production-style summary, 2-6 recompactions/episode.
