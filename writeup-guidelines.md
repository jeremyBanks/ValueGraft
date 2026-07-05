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
