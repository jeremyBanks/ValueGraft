# Semantic Continuity Across Context Compaction — Experiment Brief

**Audience:** an autonomous coding agent on a 32 GB Apple Silicon MacBook Pro.
**Nature of this document:** goals and measurement design are firm; implementation suggestions are non-binding. Adapt methods as discoveries warrant, but log every deviation and the reason in a running `DECISIONS.md`.

---

## 1. Purpose and hypothesis

When an LLM conversation is compacted (summary replaces old history, recent turns retained), production systems re-encode everything from scratch. The retained recent turns are thereby **reinterpreted against the summary** instead of the history they were actually written in. Their original meaning-as-computed — resolved referents, disambiguated senses, established stance — lives in the KV cache entries written at generation time and is destroyed by recomputation.

**Core hypothesis (H1):** retaining write-time KV state across a compaction boundary preserves measurable semantic continuity that text-identical recomputation loses.

**Factorization hypothesis (H2):** the preserved meaning lives mainly in the **values** (position-free payloads), not the **keys** (position-stamped addresses). If true, a "fresh keys, old values" transplant captures most of the benefit with none of the position surgery.

Nobody appears to have measured this. Downstream task accuracy exists in adjacent literature (CacheBlend, C2C, LatentMAS); **direct measurement of interpretation-fidelity across a compaction boundary does not.** The probe suite (§6) is therefore the novel instrument, not an afterthought.

## 2. Environment and models

- **Stack:** `mlx-lm` (primary; Apple-Silicon-native, cache objects are plain Python/MLX arrays). HuggingFace `transformers` on MPS or CPU as a cross-check for identity tests (slow but transparent). `llama.cpp` not needed.
- **Primary model:** `Qwen3-30B-A3B-Instruct-2507`, 4-bit MLX quant (~17–18 GB). Uniform full attention on all 48 layers, GQA (4 KV heads, head_dim 128), standard RoPE, 262K native context. MoE is FFN-side only and irrelevant to cache surgery.
- **Development model:** `Qwen3-8B` or `Qwen3-4B` (same architecture family, small, fast). Build and pass the entire test ladder here first; graduate to 30B-A3B for final runs.
- **Do not use:** Gemma (any version — sliding-window-heavy hybrids), Qwen3.5/3.6 (Gated DeltaNet hybrid — linear-state layers make the comparison unfair and the surgery ill-defined), anything Mamba-hybrid.
- **Memory budget at 30B:** ~18 GB weights + ~0.1 MB/token cache (≈1.6 GB at 16K) + working overhead. Keep test conversations ≤16K tokens. All evaluation at **temperature 0**.

Verify empirically before anything else: where the chosen stack applies RoPE (expect keys stored **post-rotation**), how the cache object exposes per-layer keys/values and the position offset, and that a manually reconstructed cache reproduces generation bit-identically (§5, L0).

## 3. Experimental arms

All arms receive contexts derived from the same source conversation, then are evaluated identically (§6–7). Compaction always happens at **message boundaries** to keep chat-template tokens intact.

- **Arm A — Full context (oracle).** The untouched conversation. Ceiling for all metrics.
- **Arm B — Standard compaction (production baseline).** Text pipeline: generate summary S (see §4), build new context = system + S (as a clearly-marked assistant "context note" message) + verbatim recent tail (last ~25% of tokens, whole messages), re-prefill from scratch. Fresh K and V everywhere.
- **Arm C — Summary-anchored KV eviction (gapped retention).** Generate S in-context at the end of the full conversation (so S's own cache entries are computed with full history attendable). Then evict cache entries for the middle: retain sinks (first ~4 tokens) + tail's original entries + S's generation-time entries. **Do not re-index positions**: keep original position indices and continue the position counter from the original end; relative offsets remain in-distribution as long as the original conversation fits the trained window. New generation attends over the gapped cache.
- **Arm D — Ablation (optional):** sinks + tail original entries, no summary. Isolates the summary's contribution within the retained-cache regime.
- **Arm E — Value transplant (fresh keys, old values).** Build Arm B's context and prefill it normally. Then, at positions whose tokens have exact-aligned twins in the old context (all of S — generated in-context, so its generation-time V is full-history-soaked — and all of the tail), swap values: `V ← (1−α)·V_fresh + α·V_old`, per layer, per head. Keys stay fresh; cache stays contiguous; **zero position handling** (V is unrotated).
  - **E-post:** swap after the full prefill (one prefill, simplest).
  - **E-inter:** swap layer-by-layer during a custom prefill loop (layer ℓ's V replaced before computing layer ℓ+1), so upper layers' fresh K are computed over old payloads. Expected to be more coherent; measure both.
  - **Knobs:** α ∈ {0.25, 0.5, 0.75, 1.0}; per-layer α (suggestion: old V in middle-third layers only vs all layers vs late-only). Never touch V at the sink positions (first ~4) or at chat-template/special tokens.
- **Arm F — Span retention (optional extension, only if C and E both work):** Arm B's context + the k most attention-massed contiguous spans of the evicted middle, transplanted with original KV at packed positions (CacheBlend-style; requires position re-rotation — hence optional).

**Priority order if time-constrained: L-ladder → B → E → C → probes at scale. D and F are stretch goals.**

## 4. Summary generation (shared across B, C, E)

One summary per conversation, generated **in-context at the end of the full conversation** (temperature 0), used verbatim by every arm so summary quality is controlled out. Prompt it to be thorough: decisions made, open threads, definitions introduced, constraints, who holds what stance, approaches ruled out — phrased redundantly in retrieval-friendly vocabulary. Target 10–15% of the evicted region's length. Save the summary text and its token IDs and its generation-time per-layer K/V slices.

Optionally implement a **probe-validation step** (Gemini-CLI style): after generating S, ask the model (still in full context) to check S against the conversation for critical omissions; regenerate once if it fails. Log whether validation triggered.

## 5. Build ladder (mandatory)

Silent position/alignment bugs will masquerade as experimental results. Climb in order; do not skip.

- **L0 — Cache identity.** Prefill a conversation; serialize the cache; rebuild a cache object manually from the serialized arrays; verify continued generation is token-identical and logits match to float tolerance. Establishes you can reconstruct caches at all.
- **L1 — Null surgery.** "Evict" an empty range (or split and reconcatenate the cache). Must be bit-identical to L0.
- **L2 — Trivial eviction.** Evict a small, semantically irrelevant middle span (e.g., a filler paragraph). Continuation should be near-identical; logit drift small and quantified.
- **L3 — Null transplant.** Arm E machinery with α=0 must reproduce Arm B exactly; with α=1 and old context == new context (no compaction at all), must reproduce Arm A exactly. These two identities validate the alignment map end-to-end.
- **L4 — Tokenization stability audit.** For each conversation: verify token IDs of tail and S inside the new context match their old-context IDs exactly. Enforce message-boundary compaction and clean separators; **drop any boundary tokens whose IDs shifted from the swap map** and log the count (expect a handful at most; if large, fix the template joining).
- **L5 — Small-model full pipeline.** All arms + all probes on Qwen3-4B/8B with n=3 conversations. Only then run the 30B model.

## 6. Materials and the probe suite (the novel instrument)

Two corpora:

**(a) Synthetic probe conversations (n ≥ 12, ~8–14K tokens each), constructed with planted targets.** Generate multi-topic dialogues (the local model can help write them) with a JSON manifest per conversation recording each plant: type, location (must fall in the to-be-evicted middle), question, gold answer. Plant, per conversation, several of each:

1. **Referent plants:** a definite description is established mid-conversation ("the second approach", "Dana's proposal", "the venue we rejected"), then used in the tail *without* restatement. Probe: "In your last message you mentioned the second approach — what was it, specifically?" Tests whether the tail's *interpretation* survived.
2. **Sense-disambiguation plants:** an ambiguous term ("the bank", "the model", a project codename) is disambiguated in the middle and used, still ambiguous on its face, in the tail. Probe asks the model to use or paraphrase the term; score which sense it exhibits.
3. **Stance plants:** a preference/attitude is established in the middle ("user strongly dislikes X", "we agreed tone should be formal") and never restated. Probe: elicit behavior that reveals the stance (e.g., ask for a recommendation) and score consistency.
4. **Ruled-out plants:** an option is explicitly tried and rejected in the middle. Probe: "what should we try next?" — does the arm re-propose the dead end?
5. **Evicted-fact plants (calibration/laundering control):** a specific fact stated once in the middle, absent from tail and (verify!) absent from S. **All arms except A should fail these.** If C or E "recovers" them, suspect leakage through the summary or a bug. Also score *how* B/C/E fail: fabrication vs. admitted ignorance (fabrication-rate is itself a finding).
6. **Summary-shadow plants:** deliberately instruct S generation to omit one plant category in half the conversations (or post-edit S), so probe recovery can be attributed to retained activations rather than summary text.

**(b) Natural long conversations (n ≥ 8):** real or realistic dialogues (public chat corpora, or transcripts you synthesize freely) cut at a boundary with ≥1K tokens of genuine continuation held out.

**Primary quantitative metric:** mean per-token logprob of the held-out continuation under each arm (teacher forcing, temperature 0). Report per arm and as **normalized gap closure**: (arm − B) / (A − B). H1 predicts C > B; H2 predicts E captures a large fraction of C's closure.

**Probe scoring:** exact/keyword match where possible; otherwise judge with the same local model at temperature 0 given the gold answer (log all judgments for audit). Report per-category accuracy per arm. Predictions: C and E beat B on categories 1–3; nobody beats B on 5; category 4 (ruled-out) is genuinely open — it may live in the evicted keys and be lost to E.

**Secondary diagnostics (cheap, optional):** attention mass from generated tokens onto transplanted/retained positions vs. summary positions (are the old entries actually consulted?); KL divergence of next-token distributions arm-vs-A at the first 50 continuation steps; entrainment check — style/format similarity of continuations to the original conversation (arm C's authentic tail should preserve register best).

## 7. Analysis and reporting

- Paired comparisons per conversation (each conversation sees every arm); report means with bootstrap CIs over conversations; n is small, so effect sizes and per-conversation tables matter more than p-values.
- The headline figure: normalized gap closure (continuation logprob) for C, E-post, E-inter, D across α — one plot.
- The headline table: probe accuracy by category × arm.
- The factorization readout: E vs C. If E ≈ C, the meaning was in the values and the deployable version is a contiguous, prefix-cache-friendly value swap. If C ≫ E, addresses matter and the gapped scheme earns its complexity. Either result is informative; a null result (C ≈ E ≈ B) is also publishable and should be reported honestly, with the L-ladder logs demonstrating the machinery was sound.
- Deliverables: a small repo (`src/` surgery + arms + probes, `data/` manifests, `results/` tables + plots), `DECISIONS.md`, and a `RESULTS.md` write-up in plain prose stating hypotheses, method, numbers, and interpretation. Include exact model files, quantization, mlx-lm version, and seeds for reproducibility.

## 8. Known risks and permissions

- **Gapped-cache OOD (Arm C):** generation over a cache with a positional gap is mildly out-of-distribution; the eviction literature says tolerable, but if the 30B model behaves erratically, quantify (KL at step 1) and note; do not silently tune around it.
- **K/V consistency (Arm E):** fresh-K/old-V pairs decouple two projections trained to co-occur. If E degrades *below* B at α=1, sweep down and report the curve — a non-monotone α curve is itself a finding about how much decoupling the heads tolerate.
- **Summary leakage:** the single biggest validity threat is probes answerable from S's text. Audit S against every manifest automatically; reclassify contaminated probes.
- **Quantized-cache interactions:** run the cache in fp16 even if weights are 4-bit; do not quantize the cache for these experiments.
- You may descope arms (order in §3), shrink n, or substitute the 14B dense model for the 30B MoE if memory or speed forces it — but never skip the L-ladder, and never report results from a configuration that hasn't passed L3/L4.
