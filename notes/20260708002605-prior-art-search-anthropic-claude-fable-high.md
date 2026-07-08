# Prior-Art & Novelty Assessment: Preserving Write-Time KV-Cache Values to Maintain Semantic Interpretation Across a Compaction Boundary

## TL;DR

- **The specific composite idea is largely novel, but each of its four
  mechanical ingredients has strong prior art.** No paper preserves original
  write-time VALUE states while recomputing KEYS specifically to test whether
  compaction changes the _semantic interpretation of retained tokens_, evaluated
  on referent/word-sense/stance metrics. That combination appears absent from
  the literature through mid-2026.
- **The closest single work is “Models Take Notes at Prefill: KV Cache Can Be
  Editable and Composable” (Bojie Li, Pine AI, arXiv 2606.17107, 2026)**, which
  shows values are position-free while keys are RoPE-re-rotated across a context
  boundary, uses the model’s own attention to transplant cached notes (no
  trained projection), and evaluates whether a transplanted/edited cache still
  _governs a decision_ — but it frames this as mechanism/efficiency, not as
  semantic-continuity of surviving tokens under compaction.
- **The evaluation dimension is the clearest gap.** Every
  KV-eviction/compression/compaction paper measures efficiency, perplexity,
  retrieval (needle-in-a-haystack), or aggregate QA/reasoning accuracy. No
  benchmark measures coreference re-binding, word-sense shift, or stance drift
  of _retained_ text after cache manipulation. The one adjacent work on stance
  (ConstraintRot / “Governance Decay,” arXiv 2606.22528, 2026) measures
  constraint _survival vs. dropping_, not reinterpretation of surviving
  constraints.

## Key Findings

1. **Asymmetric key/value treatment already exists — but for
   efficiency/positional correctness, not semantic continuity.** Multiple
   2025–2026 papers exploit that RoPE keys are position-dependent while values
   are position-free: “Models Take Notes at Prefill” (2606.17107), “Still:
   Amortized KV Cache Compaction” (2606.07878), KV Packet (2604.13226), MiniPIC
   (2606.13126), KVLink (2502.16002), AdapShot (2605.03644), and HERMES
   (2601.14724) all re-rotate cached keys to new positions while leaving values
   unrotated. Separately, AsymKV (2506.05410) and KVSlimmer (2603.00907) treat
   keys and values asymmetrically during _merging_ (compress homogeneous keys,
   preserve heterogeneous values). None of these frames the asymmetry as
   protecting semantic interpretation of retained tokens across a compaction
   boundary.
1. **No eviction/compression method studies semantic reinterpretation of
   retained tokens.** StreamingLLM, H2O, SnapKV, PyramidKV, Scissorhands, TOVA,
   FastGen, KIVI, and their successors all measure perplexity, throughput,
   memory, and task accuracy. Their concern is which _dropped_ tokens hurt
   downstream answers; they treat surviving tokens’ meaning as fixed.
1. **Summary-in-context with cache retention is done by Activation Beacon — but
   via learned/new representations, not preserved original write-time states.**
   Activation Beacon (2401.03462) generates condensed “beacon” activations while
   the full chunk is still attended, then retains those activations and discards
   the raw ones — but the retained states are _newly learned_ beacon-token KVs,
   not the summary text’s original write-time cache.
1. **Attention-based retrieval of old cached KV into new contexts is
   well-established** (Memorizing Transformers, InfLLM, LongMem, MemoryLLM).
   These preserve write-time KV pairs and retrieve them via the model’s own
   attention — matching the “attention-based transplant” concept mechanically —
   but they measure perplexity/retrieval, never semantic continuity, and they
   _append_ memory rather than transplanting it across a compaction boundary to
   replace re-encoded text.

## Details

### Area 1 — KV-cache eviction and compression

**StreamingLLM (Xiao et al., 2023/2024)** retains “attention sink” initial
tokens plus a recent window, evicting the middle. **H2O (Zhang et al., NeurIPS
2023, arXiv 2306.14048)** keeps “heavy hitter” tokens by cumulative attention
score, dynamically retaining ~20% of the cache. **SnapKV (Li et al., 2024)**
uses an observation window at the prompt suffix with pooling to select clustered
important KV positions per head; **PyramidKV (Cai et al., 2024)** and **Ada-KV**
vary the budget per layer/head; **Scissorhands, TOVA, FastGen** are related
attention-score heuristics; **KIVI** and Q-Hitter quantize.

- (a) Mechanically: rank tokens by attention and drop/keep whole KV pairs (both
  key and value together), then renormalize attention over the retained set.
- (b) Values: for retained tokens, the original write-time KV is kept verbatim;
  evicted tokens are permanently erased. There is no _transplant_ — the cache is
  never moved across a boundary.
- (c) Keys vs values: treated symmetrically (a token is kept or dropped in
  full). Exceptions are value-aware scoring (Devoto et al. 2024) and merging
  methods (below), but these still don’t recompute keys while keeping values.
- (d) Evaluation: LongBench, RULER, needle-in-a-haystack, perplexity,
  throughput. Two papers explicitly reach toward “semantics”: **ChunkKV
  (2502.00299)** and **SABlock (2510.22556)** argue token-level eviction
  “fragments” semantic units and that discarding function words/digits breaks
  segment meaning — but both still evaluate on LongBench accuracy, not
  referent/word-sense metrics. **SentenceKV (2504.00970)** and **SemantiCache
  (2603.14303)** retain sentence/semantic units but again measure
  LongBench/NIAH.
- (e) Difference from core idea: none preserve values while recomputing keys;
  none measure semantic reinterpretation of survivors.

### Area 2 — Cross-context KV reuse and blending

**CacheBlend (Yao et al., 2024/2025, arXiv 2405.16444)** reuses precomputed
chunk KV at non-prefix positions and _selectively recomputes a small subset of
tokens’ KV_ to restore cross-attention — its published recomputation ratio is
roughly 6% of tokens (per ProphetKV, arXiv 2602.02579: “the best prior method,
CacheBlend, requires a recomputation ratio of approximately 0.06”), and its own
abstract reports it “reduces time-to-first-token (TTFT) by 2.2–3.3x and
increases the inference throughput by 2.8–5x from full KV recompute without
compromising generation quality.” **Prompt Cache (Gim et al., MLSys 2024, arXiv
2311.04934)** precomputes reusable “prompt modules” with position placeholders
and splices them, adapting RoPE for discontinuous position IDs. **EPIC
(2410.15332)** does position-independent caching with LegoLink, recomputing a
few chunk-boundary tokens. **KVLink (Yang et al., 2025, arXiv 2502.16002)**
re-encodes positions of cached keys and inserts _trainable_ link tokens to
restore cross-document self-attention, cutting TTFT by up to 90%.
**Block-attention** methods and **KVCOMM (2510.12872)** estimate cross-context
KV offsets.

- (a)–(c): These reuse both keys and values of shared segments; the
  recomputation they do is _selective for both K and V_. CacheBlend specifically
  identifies its recompute set as “High-KV-Deviation (HKVD) tokens” — per
  CacheClip (arXiv 2510.10129), “the value matrix (V) from the recomputed second
  layer is compared against the precomputed version (which lacks inter-chunk
  attention) to identify top-k tokens with the largest V-value discrepancies.”
  RoPE re-rotation of keys (Prompt Cache, KVLink) does treat keys specially for
  _position_, keeping values unrotated — but the goal is positional correctness,
  not semantic preservation of retained meaning.
- (d) Evaluation: F1, QA accuracy, TTFT, throughput. CacheClip notes CacheBlend
  can degrade “context continuity” mid-range and split entities (e.g., a long
  number tokenized into pieces) — a semantic-integrity observation — but still
  measured by answer correctness on RULER.
- (e) Difference: closest in that they recompute _some_ K/V while reusing the
  rest, but the reuse is for efficiency across _documents/requests_, not to
  preserve interpretation of _retained_ text across a _compaction_ event, and
  none isolate value-preservation-with-key-recompute as the mechanism.

### Area 3 — Learned compression into latent/soft tokens

**Gist tokens (Mu et al., 2023)**, **AutoCompressors (Chevalier et al., 2023)**,
**ICAE (Ge et al., 2024)**, **Activation Beacon (Zhang et al., 2024, arXiv
2401.03462)**, **500xCompressor (Li et al., 2025)**, **xRAG (Cheng et al., 2024,
arXiv 2405.13792)**, and **Cartridges (Eyuboglu et al., 2025, arXiv
2506.06266)** all compress context into _newly learned_ representations.

- (b) Values: These do NOT preserve original write-time states — they train new
  soft tokens / beacon activations / cartridge KVs (Cartridges literally
  backprop into K/V vectors, i.e., prefix tuning). Activation Beacon is the
  nearest to “summary-in-context with cache retention”: it “directly
  compress[es] the activations (i.e. keys and values at every layer), rather
  than leveraging soft prompts to relay information,” achieving up to a
  128K-token context with ~8x KV-cache memory reduction and ~2x inference
  acceleration at an 8x condensing ratio, and is “trained purely with
  short-sequence data in just 10K steps.” Crucially, the retained state is the
  beacon’s _learned_ KV, not the summary text’s write-time cache.
- (d) Evaluation: perplexity, LongBench, NIAH, throughput/memory. None measure
  referent/word-sense/stance reinterpretation.
- (e) Difference: fundamentally opposite to the core idea — they _replace_
  original states with learned ones, whereas the core idea _preserves_ original
  write-time values. Confirmed: none measure semantic interpretation shifts of
  retained text.

### Area 4 — KV-cache editability / composability / splicing (most relevant)

**“Models Take Notes at Prefill: KV Cache Can Be Editable and Composable” (Bojie
Li, Pine AI, arXiv 2606.17107, 2026)** is the closest existing work.

- (a) Mechanically: establishes causally that at prefill the model writes
  “field-conditioned conclusions” onto downstream aggregator/delimiter tokens;
  the field’s own KV drives <1% of the decision. It then (i) _edits_ the cache
  via an append-only “erratum,” and (ii) _composes_ by transplanting a
  precompiled skill’s cached KV into a new context — RoPE-re-rotating the keys
  from source to target positions while leaving **values unrotated** (“values
  are position-free”), using the model’s own attention with no trained
  projection.
- (b) Values: preserved from write time (this is the crux — it caches post-RoPE
  keys and re-rotates them, values untouched).
- (c) Keys vs values: explicitly asymmetric — keys re-rotated, values kept.
- (d) Evaluation: logit cosine similarity to full-reprefill (0.90–0.999 across
  twelve models), decision-identity on a gated tool-decision (deny/approve)
  task, LoCoMo QA accuracy (transplant ≡ full recompute), and serving metrics
  (98.5% prefix-cache hit-rate, 53–398× lower p90 TTFT). This is a
  _decision-governance_ evaluation — “does the transplanted/edited cache still
  govern the decision” — which is the nearest thing in the literature to
  semantic-continuity, but it is framed as a mechanistic/efficiency result, not
  as coreference/word-sense/stance drift of retained tokens.
- (e) Difference from core idea: It transplants a _skill/module_ into a fresh
  context and edits _mutable fields_, not a _summary generated in-context whose
  cache is retained across a compaction boundary_; and it does not use
  linguistic semantic-continuity metrics. It even positions itself against
  CacheBlend/EPIC/KVLink as building “on prior caching work,” claiming novelty
  only in the _mechanism_ and _decision-governance lens_.

Other Area-4 work: **“Still: Amortized KV Cache Compaction in a Single Forward
Pass” (2606.07878)** trains a per-layer Perceiver “compactor” that operates in a
position-free frame (inverse-rotate keys, compact, re-rotate; “values are not
rotated”) — asymmetric key/value handling for _compaction_, but the compact KV
is _synthesized_, not preserved write-time, and it supports summarization but
evaluates on RULER (exceeding the strongest baseline by 8–22
points)/HELMET/LongBench. **KV cache merging** — CaM, D2O, MiniCache, KeepKV
(2504.09936), VAM, AsymKV (2506.05410), KVSlimmer (2603.00907) — merges
to-be-evicted states into retained ones; AsymKV/KVSlimmer treat keys/values
asymmetrically (compress homogeneous keys, preserve heterogeneous values) but
for _compression_, measured by perplexity/LongBench. **VAM** uniquely targets
“semantic degradation in the cache itself” by merging attention outputs into
value states, but still evaluates on standard long-text benchmarks. **“Fast KV
Compaction via Attention Matching” (2602.16284)** and Cartridges (2506.06266)
synthesize compact caches via attention-matching / self-study rather than
preserving originals.

### Area 5 — Cross-model / cross-agent KV transfer

**DroidSpeak (Liu et al., NSDI 2026, arXiv 2411.02820)** enables KV-cache reuse
across _different fine-tuned models of the same architecture_ by selectively
recomputing a few critical layers and reusing the rest (up to 4× throughput,
~3.1× faster prefill, negligible quality loss). **KVCOMM (2510.12872)** does
online cross-context KV communication for multi-agent systems via anchor-based
offset estimation. **KV-Embedding (2601.01046)** re-routes final-token KV states
as a prefix for embedding.

- (b)–(c): DroidSpeak reuses write-time KV across models but recomputes at
  _layer_ granularity (not key-vs-value asymmetry); measures F1/Rouge-L/code
  similarity + throughput. It studies “if/when sharing affects quality” — an
  efficiency-quality tradeoff, not semantic reinterpretation.
- (e) Difference: cross-model, not cross-compaction-boundary within one model;
  no value-preservation/key-recompute asymmetry; no semantic-continuity metric.

### Memory-augmented transformers (relevant to the attention-transplant concept)

**Memorizing Transformers (Wu et al., ICLR 2022, arXiv 2203.08913)** append
write-time (key,value) pairs to a non-differentiable external memory and
retrieve them via kNN-augmented attention, combining local and memory attention
through a learned gate. Their abstract reports that “an approximate kNN lookup
into a non-differentiable memory of recent (key, value) pairs improves language
modeling,” with performance improving “when we increase the size of memory up to
262K tokens,” and critically “gradients are not backpropagated into the external
memory.” **InfLLM (Xiao et al., NeurIPS 2024)** augments StreamingLLM’s
sink+window with block-level retrieval from a full KV cache. **LongMem** and
**MemoryLLM** are similar.

- (b) Values: preserved from write time (the defining feature — kNN over stored
  KV pairs).
- (d) Evaluation: perplexity, long-context retrieval, NIAH. Never semantic
  continuity of retained text.
- (e) Difference: they _retrieve/append_ old KV to extend context using the
  model’s own attention (matching the transplant concept mechanically and
  preserving write-time values), but they do NOT transplant a summary’s cache
  across a compaction boundary to _replace_ re-encoded summary text, and they
  never measure whether retrieval changes the interpretation of surviving
  tokens. They are the strongest prior art for “attention-based retrieval of
  preserved write-time KV,” and should be cited as such.

### The evaluation gap (confirmed via targeted search)

- **“Semantic Integrity Matters: Benchmarking and Preserving High-Density
  Reasoning in KV Cache Compression” (Liu et al., HKUST-GZ, arXiv 2502.01941)**
  introduces KVFundaBench and ShotKV, but “semantic integrity” means keeping
  few-shot demonstration blocks as indivisible “Semantic Units”; it measures
  CoT-reasoning accuracy, finding e.g. “GSM8K experiences a severe average
  performance drop exceeding 35%” at a 10% compression ratio while “retrieval
  tasks remain robust.” Not referent/sense/stance drift of survivors.
- **“Hold Onto That Thought: Assessing KV Cache Compression on Reasoning” (Liu
  et al., UMD/UChicago, arXiv 2512.12008, NeurIPS 2025)** benchmarks
  H2O/SnapKV/TOVA/etc. on eight reasoning datasets (FOLIO, DROP, GSM8K,
  MATH-500, ReClor, StrategyQA, CommonSenseQA, OpenBookQA), measuring answer
  accuracy and reasoning-trace length; it finds “H2O and our decoding-enabled
  variant of SnapKV are dominant strategies for reasoning models” and that
  “eviction strategies at low budgets can produce longer reasoning traces.” Not
  semantic reinterpretation.
- **ConstraintRot / “Governance Decay: How Context Compaction Silently Erases
  Safety Constraints in Long-Horizon LLM Agents” (Shiyang Chen, arXiv
  2606.22528, 2026)** is the only work on stance/constraint consistency under
  compaction: across 1,323 episodes, “violation rises from 0% with the policy in
  full context to 30% after compaction, reaching 59% for some models; when the
  constraint survives the summary, violation remains 0%, but when it is dropped,
  violation reaches 38%.” Critically, it attributes violations to constraints
  being _dropped_ vs. _surviving_ — a binary survival test, not reinterpretation
  of a _retained_ constraint. It also introduces a “Compaction-Eviction Attack”
  and a “Constraint Pinning” mitigation.
- No coreference-resolution or word-sense-disambiguation benchmark tied to
  KV-cache manipulation exists. A survey (“Rethinking KV Cache Compression,”
  arXiv 2503.24000) explicitly notes existing studies “lack an in-depth analysis
  of how KV cache compression affects the response quality of individual
  examples,” supporting the novelty thesis.

## Direct Novelty Assessment

**Verdict: the core idea partially exists in fragments but the integrated scheme
— and especially its evaluation axis — appears absent.**

- **Ingredient (1), preserve/transplant write-time VALUES with asymmetric
  key/value handling (keys recomputed/re-rotated, values kept):** ALREADY EXISTS
  as mechanism. “Models Take Notes at Prefill” (2606.17107) and “Still”
  (2606.07878) both explicitly keep values unrotated while re-rotating keys;
  merging work (AsymKV 2506.05410, KVSlimmer 2603.00907) treats keys/values
  asymmetrically. **This is not novel in isolation.**
- **Ingredient (2), measuring whether compaction changes the SEMANTIC
  INTERPRETATION of RETAINED tokens (referent/word-sense/stance):** APPEARS
  ABSENT. This is the strongest and most defensible novelty. The closest,
  ConstraintRot (2606.22528), tests constraint _survival_, not reinterpretation
  of survivors; reasoning benchmarks (2502.01941, 2512.12008) test aggregate
  accuracy.
- **Ingredient (3), generate a summary in-context while full history is attended
  and RETAIN its cache rather than re-encode:** PARTIALLY EXISTS. Activation
  Beacon (2401.03462) retains in-context-generated condensed activations, but
  they are _learned_ beacon KVs, not the summary text’s _preserved write-time_
  cache. No work retains a natural-language summary’s own write-time KV across a
  compaction boundary.
- **Ingredient (4), attention-based transplant using the model’s own attention
  (no trained projection):** ALREADY EXISTS. Memorizing Transformers
  (2203.08913), InfLLM, and “Models Take Notes at Prefill” all use native
  attention (kNN or direct splice) over preserved write-time KV. **Not novel in
  isolation.**

**Closest existing work:** “Models Take Notes at Prefill: KV Cache Can Be
Editable and Composable” (arXiv 2606.17107, 2026) — it matches ingredients (1)
and (4) tightly and touches (2) via decision-governance, but not (3) or a
linguistic semantic-continuity evaluation.

**The specific remaining gaps** (i.e., the novel contribution to claim): (a) a
semantic-continuity _evaluation_ of RETAINED tokens (coreference re-binding,
word-sense shift, stance drift) under cache manipulation — no such benchmark
exists; and (b) _combining_ value-preservation-with-key-recompute with a
retained-in-context-summary cache across a compaction boundary, motivated by and
measured against semantic continuity rather than efficiency.

## Recommendations

**Stage 1 — Position the contribution as the _evaluation +
asymmetric-preservation combination_, not any single mechanism.** The mechanical
pieces (value-preservation, key re-rotation, attention-based retrieval,
in-context summary) are individually published. The defensible novelty is: (a)
_deliberately_ preserving write-time VALUE states across a compaction boundary
while recomputing/re-rotating keys, _for the purpose of_ semantic continuity;
and (b) a semantic-continuity evaluation (referent resolution, word-sense,
stance/constraint) of _retained_ tokens. Lead with these.

**Stage 2 — Directly differentiate from the three nearest works in writing:**
(1) “Models Take Notes at Prefill” (2606.17107) — cite as the closest mechanism
(value-position-free, key re-rotation, attention transplant, decision-governance
eval) and distinguish on
summary-generated-in-context-then-retained-across-compaction and linguistic
semantic-continuity metrics vs. their deny/approve decision-identity. (2)
Activation Beacon (2401.03462) — distinguish preserved _original_ write-time KV
vs. their _learned_ beacon activations. (3) Memorizing Transformers / InfLLM —
distinguish transplant-to-replace-re-encoding vs. append-to-extend, and
semantic-continuity metric vs. perplexity.

**Stage 3 — Build the missing benchmark.** Since no
coreference/word-sense/stance-consistency benchmark exists for cache
manipulation, constructing one is itself a publishable contribution and the
strongest novelty anchor. Model it on ConstraintRot’s deterministic grading but
target _reinterpretation of surviving text_ (e.g., pronoun re-binding, sense
flips, established-stance drift) rather than survival/dropping.

**Benchmarks/thresholds that would change the assessment:**

- If a paper is found that evicts/compacts, _retains_ specific tokens, and then
  measures whether those retained tokens’ referents/senses/stances change → the
  evaluation-novelty claim collapses to “incremental.” (None found through
  mid-2026.)
- If a paper preserves values while recomputing keys _and_ attributes a semantic
  (not efficiency) benefit to that asymmetry → the mechanism-novelty claim
  weakens substantially. “Models Take Notes at Prefill” is the closest and
  should be monitored for follow-ups.
- If “Models Take Notes at Prefill” is revised to add coreference/word-sense
  evaluation → treat as direct prior art.

## Caveats

- **Recency/peer-review risk.** Several of the most relevant papers carry 2026
  arXiv IDs and are likely unreviewed preprints: “Models Take Notes at Prefill”
  (2606.17107, single-author from Pine AI with self-published code), “Still”
  (2606.07878), “Governance Decay”/ConstraintRot (2606.22528), KV Packet
  (2604.13226), MiniPIC (2606.13126), ProphetKV (2602.02579). Verify publication
  status before treating as established prior art. “Hold Onto That Thought”
  (2512.12008) is NeurIPS-2025-listed but likely workshop-level.
- **Date-plausibility flag.** A few arXiv IDs surfaced in search (e.g.,
  2605.xxxx, 2606.xxxx, 2607.01299) imply very recent submissions consistent
  with the stated July 2026 present that I could not always fetch in full —
  treat their specific numeric claims as provisional.
- **Definitional boundary.** “Semantic continuity of retained tokens” is my
  operationalization of the user’s intent (referent/word-sense/stance). If the
  intended metric is broader (e.g., any answer-quality change attributable to
  reinterpretation), then reasoning-focused benchmarks (2502.01941, 2512.12008)
  become partial prior art.
- **I could not exhaustively fetch every paper’s full text**; some mechanical
  claims (e.g., exact key-vs-value handling in some merging papers) rely on
  abstracts and secondary descriptions. The core conclusions rest on primary
  sources for the pivotal papers (2606.17107, 2401.03462, 2203.08913,
  2405.16444, 2311.04934).
