>> ⛔ BLOCKING REQUIREMENT: the paper MUST fully document DATA PROVENANCE and experimental
>> methodology per **METHODS-PROVENANCE-REQUIREMENTS.md** (who/what generated every piece of
>> data — prompts authored, replies model-native, summary self-gen, gold from planted facts;
>> exact model checkpoint ids; procedures; gates; design rationale). EVERY prior writeup FAILED
>> this. A methods section that skips any item there is NOT DONE. This is a blocking review
>> failure, not optional. Check the paper against that file with Fable + critics + Codex.

# Write-up guidelines (final artifact: HuggingFace community blog post + repo)

Target: HF community blog post + this GitHub repo — NOT an academic paper.
Underlying gist, written straight: "This seems like an obvious thing to try,
but I couldn't find it in my literature search — am I missing the right
terminology, or is it so trivial/inconsequential that nobody wrote it up?
Either way, here's what I observed."

1. **Attribution (byline hierarchy — match the top of REPORT.md, keep it clean):**
   Fable 5 and GPT-5.5 are the two MAIN AUTHORS; Jeremy Banks provides guidance /
   direction / advising; the other models get a light "assistance from" thanks.
   Do NOT itemize what each model did specifically. Do NOT mention funding (it reads
   weird). Just the hierarchy, exactly as the report byline below.
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

## Attribution / byline (updated 07-08 — match the top of REPORT.md; do NOT itemize per model, do NOT mention funding)
- MAIN AUTHORS: Anthropic Claude Fable 5 and OpenAI GPT 5.5.
- GUIDANCE / direction / advising: Jeremy Banks.
- ASSISTANCE (light thanks, NOT itemized by task): Anthropic Claude Opus 4.8,
  Anthropic Claude Sonnet 5, Google Gemini Pro 3.1.
- NO funding mention. NO per-model breakdown of who did what.

The byline, exactly as it appears at the top of REPORT.md (this is the canonical form):
"By Anthropic Claude Fable 5 and OpenAI GPT 5.5, with guidance from Jeremy Banks and
assistance from Anthropic Claude Opus 4.8, Anthropic Claude Sonnet 5, and Google Gemini Pro 3.1."

NOTE: this author-attribution byline is SEPARATE from the DATA-PROVENANCE methods
documentation (METHODS-PROVENANCE-REQUIREMENTS.md) — that documents who/what generated
each piece of experimental DATA and IS required in detail in the methods section. The
byline is just the clean author hierarchy above; don't conflate the two.
