Architectural Paradigms in Key-Value Cache Management: Semantic Continuity,
Asymmetry, and State Transplantation The scaling of autoregressive large
language models (LLMs) has fundamentally shifted the primary inference
bottleneck from parameter computation to memory bandwidth and capacity, a
constraint driven overwhelmingly by the Key-Value (KV) cache. As models are
deployed in increasingly complex, long-horizon settings—spanning multi-turn
agentic workflows, repository-scale code generation, and interactive document
analysis—the memory footprint of the KV cache scales linearly with sequence
length, batch size, and layer depth. For example, a 70-billion-parameter model
processing a 128,000-token context can easily consume upwards of 40 GB of VRAM
solely for its KV cache, severely limiting concurrent user serving and driving
up operational costs. To mitigate this memory wall, a vast body of literature
has emerged proposing various forms of cache eviction, quantization, cross-layer
sharing, and latent compression. However, the overwhelming majority of these
optimizations treat the KV cache purely as a systems-level tensor optimization
problem. They prioritize throughput, memory reduction, and average-case
perplexity over the rigorous preservation of semantic continuity. A new
theoretical paradigm is emerging that challenges this monolithic treatment,
proposing highly targeted, asymmetric manipulations of the cache. This report
exhaustively analyzes the intersection of asymmetric KV treatment,
semantic-continuity preservation, in-context summary state retention, and
attention-based transplantation. It systematically distinguishes these advanced
theoretical constructs from existing adjacent fields and concludes with a direct
assessment of whether a unified architecture matching this highly specific
profile currently exists in the academic or industrial literature. Taxonomy of
Adjacent Context Management Paradigms To properly situate the concept of
asymmetric, attention-based state transplantation across a compaction boundary,
it is necessary to first delineate five major adjacent areas of KV-cache
research. These fields represent the current state-of-the-art in context
management, though they diverge significantly in their goals, underlying
mechanisms, and treatment of semantic integrity. KV-Cache Eviction and
Compression The most ubiquitous approach to managing unbounded context growth is
KV-cache eviction, which operates on the premise that attention distributions in
transformers are inherently sparse. A small fraction of tokens—often referred to
as "heavy hitters," attention sinks, or structural anchors—accumulates the vast
majority of attention mass, rendering the remaining tokens theoretically
expendable. Frameworks such as H2O, StreamingLLM, and SnapKV implement dynamic
eviction policies that score tokens based on accumulated attention weights and
discard those falling below a predefined threshold, effectively maintaining a
fixed-size cache budget. PyramidKV refines this by observing that the number of
crucial keys and values required for future generation decreases in deeper
layers, allowing for aggressive, layer-wise pruning during the prefill stage.
Similarly, CompressKV introduces the concept of Semantic Retrieval Heads (SRHs),
identifying specific attention heads responsible for capturing long-range
semantic evidence and using only their scores to dictate eviction. Further
extending this, systems like ThinKV introduce "thought-adaptive" eviction for
reasoning models, applying a hybrid quantization-eviction strategy that assigns
token precision by thought importance and progressively evicts tokens from less
critical logical branches as reasoning trajectories evolve. IntentKV introduces
cross-turn intent-aware pruning for agent inference, maintaining a session-level
query memory that scores history across multiple turns without relocating
surviving KV rows, thereby preserving logical slot identities. Interestingly,
aggressive KV eviction has also been shown to inadvertently induce "Accidental
Robustness" against jailbreak attacks through Malicious Semantic Eviction, where
the attack's own attention redirection causes its malicious tokens to be dropped
from the cache. Semantic Preservation Focus: The primary objective of these
methods is strictly computational efficiency and memory reduction. They
fundamentally fail to address deep semantic-continuity preservation. By treating
tokens as isolated, independent variables to be pruned, these methods routinely
induce "semantic fragmentation". They indiscriminately sever linguistic
structures, disrupting noun-phrase dependencies, multi-step logical derivations,
and established contextual constraints. Therefore, while they optimize cache
reuse for efficiency, they actively destroy semantic continuity across
compaction boundaries. Cross-Context KV Reuse and Blending A parallel track of
research focuses on reusing precomputed KV caches across different inference
requests to minimize the Time-to-First-Token (TTFT) during the compute-heavy
prefill phase. Technologies like vLLM's PagedAttention and SGLang's
RadixAttention allow multiple requests sharing an identical prompt prefix to map
to the same physical memory blocks, utilizing OS-style virtual memory paging and
radix trees. More advanced frameworks attempt to reuse non-contiguous cache
segments. CacheBlend, KVLink, and Prompt Cache propose mechanisms to reuse
precomputed KV caches of isolated text chunks (e.g., retrieved documents in a
Retrieval-Augmented Generation pipeline) even when they do not share an exact
prefix. CacheBlend achieves this by selectively recomputing the KV values of a
small subset of transition tokens to update the reused cache, addressing the
fact that standard KV caches are context-dependent and heavily influenced by
absolute positional encodings. RelayCaching pushes this further for multi-agent
LLM systems, recognizing that KV caches for identical content are highly
consistent across phases, while prefix-induced deviations are localized. It
selectively recomputes only these localized deviations, allowing downstream
agents to reuse upstream decoding caches. Additionally, systems like CacheGen
tackle distributed serving by encoding KV caches into compact bitstreams for
rapid network transmission, decoding them into full tensors upon arrival at a
new inference node. Semantic Preservation Focus: This category is entirely
efficiency-oriented, aiming to bypass redundant floating-point operations
(FLOPs) and network bottlenecks. It does not perform context compaction or
semantic summarization. Because these methods strive for exact or near-exact
mathematical equivalence to full recomputation, they preserve semantic
continuity by default, but they do so by avoiding compaction altogether rather
than gracefully carrying semantics across a compressed boundary. Learned Context
Compression into Latent/Soft Tokens To achieve extreme compression ratios beyond
the limits of token eviction, researchers have developed methods that compress
extensive textual contexts into a small number of continuous latent vectors, or
"soft tokens." Architectures utilizing gist tokens, AutoCompressors, Activation
Beacons, and ICAE fall into this category. In these systems, a specialized
encoder (or a modified forward pass of the LLM itself) squashes a long context
into a dense representation. For instance, Simplified Sparse Attention (SSA)
interleaves gist tokens during pretraining, forcing the model to pack
chunk-level information into these specific tokens via restricted causal masks.
During decoding, the query only scores against the gist tokens, selectively
unfolding the raw tokens of the most relevant chunks. Other approaches train
explicit encoder-decoder compressors, mapping a long token sequence to a shorter
sequence of latent embeddings that replace standard token embeddings in the
decoder. COMI (Coarse-to-fine Adaptive Context Compression) dynamically assigns
compression rates based on Marginal Information Gain (MIG), merging tokens into
latent representations based on information value distribution. Cartridges
similarly trains highly compact KV caches in latent space via prefix-tuning on
synthetic data, treating the compact cache as a trainable parameter rather than
a filtered subset of the original. Semantic Preservation Focus: These methods
explicitly attempt to preserve broad semantic meaning within a highly compressed
footprint. However, they rely heavily on trained projection modules, auxiliary
encoders, or specialized fine-tuning objectives. Furthermore, because the
representations are highly abstracted aggregations of the input, they frequently
suffer from "lossy" compression. They struggle with exact referent resolution or
strict constraint adherence compared to the raw text, as the nuanced, discrete
data points are smoothed into a continuous distribution. Crucially, they do not
utilize the model's native attention mechanism to transplant unaltered,
write-time KV states; they synthesize entirely new, lower-dimensional
representations. KV-Cache Editability and Composability Recent breakthroughs in
mechanistic interpretability have revealed that the KV cache is not merely a
static log of past inputs, but a dynamic, editable workspace. Research
surrounding "Models Take Notes at Prefill" demonstrates that LLMs memoize
field-conditioned conclusions onto downstream aggregator tokens during the
prefill phase. The actual key/value vectors of the raw data fields drive less
than 1% of the final causal decision; the model relies almost entirely on the
downstream "notes" it has written. This observation has birthed the concept of
KV-cache editability and splicing. Systems like KVEraser allow for localized
context erasing by replacing the KV states of a specific deleted interval with
learned steering states, leaving the suffix cache intact without requiring a
full recomputation of the sequence. Furthermore, because these memoized notes
are position-portable, precompiled skills or context blocks can be
RoPE-repositioned (Rotary Position Embedding) and spliced directly into new
contexts. The boundary seams are then repaired by recomputing a minimal number
of transition tokens, yielding a result virtually indistinguishable from a full
recomputation. Semantic Preservation Focus: This area directly manipulates
semantic continuity by proving that specific reasoning states can be isolated,
rotated, and transplanted. It is highly semantic in nature, but its current
application is heavily focused on post-hoc editing, unlearning (erasing harmful
context or stale tool observations), and static modular skill composition. It is
not currently applied to the continuous summarization and active compaction of
an unfolding agentic trajectory. While it avoids trained projection modules for
the splicing itself, systems like KVEraser still rely on trained steering
vectors to simulate the erasure. Cross-Model and Cross-Agent Latent Transfer As
multi-agent systems scale, the conventional method of passing information
between agents via generated text creates a severe decoding and re-encoding
bottleneck. Latent communication protocols bypass this by allowing agents to
directly exchange continuous internal representations—specifically, their KV
caches. Systems like PolyKV propose shared, asymmetrically-compressed KV cache
pools where multiple concurrent inference agents access a single latent memory,
vastly reducing the footprint of multi-agent execution. More complex is
Cache-to-Cache (C2C) communication and heterogeneous dense alignment protocols,
which enable the projection and fusion of a source model's KV cache directly
into the latent space of a target model with a different architecture. This
requires sophisticated alignment mechanisms because heterogeneous models differ
in layer depth, head dimensionality, channel geometry, and positional encoding
logic. Semantic Preservation Focus: The objective here is high-fidelity
knowledge transfer without the lossy intermediate step of text generation. While
it heavily involves KV-cache transplantation, it requires explicit neural cache
fusers or trained projection layers to map the latent spaces between different
architectures. This fundamentally distinguishes it from native, intra-model
attention-based state retrieval, placing it outside the scope of utilizing the
model's own frozen attention to navigate a compaction boundary.

| Paradigm                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Primary Mechanism                                    | Core Objective                                  | Semantic Preservation Focus                      | Relies on Trained Projections?      |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------ | ----------------------------------- |
| Eviction/Compression                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Token scoring and dropping (e.g., SnapKV, ThinKV)    | Memory reduction, TTFT speedup                  | Low (Prone to semantic fragmentation)            | No                                  |
| Context Reuse                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Exact prefix matching, localized selective recompute | Compute efficiency across requests              | High (Mathematically equivalent to full prefill) | No                                  |
| Latent Compression                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Gist tokens, Encoder-Decoder architectures           | Extreme context length reduction                | Medium (Abstracted/lossy semantics)              | Yes (Encoders/Fine-tuning)          |
| Editability/Splicing                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | RoPE-repositioning, seam-repair, KV steering         | Post-hoc editing, unlearning, skill composition | High (Manipulates memoized conclusions)          | Partial (Some use steering vectors) |
| Cross-Agent Transfer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Inter-model KV projection, Dense Alignment           | Bypassing text decoding bottlenecks             | High (Requires dense latent alignment)           | Yes (Neural cache fusers)           |
| The Measurement of Semantic Continuity                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                                                      |                                                 |                                                  |                                     |
| A critical distinction must be drawn between standard evaluations of context management and the rigorous measurement of semantic continuity. The vast majority of KV-cache compression papers evaluate their methods using automated perplexity, generalized QA accuracy (e.g., LongBench), or simple passkey retrieval tasks (e.g., Needle-in-a-Haystack).                                                                                                                                                                                                                                                                                                                                                                      |                                                      |                                                 |                                                  |                                     |
| These metrics are fundamentally inadequate for measuring the true degradation of intelligence in compressed systems. As demonstrated by the KVFundaBench framework and the "Semantic Integrity Matters" study, there is a sharp dichotomy in compression robustness: while sparse retrieval tasks remain remarkably resilient under heavy token eviction, "High-Density Reasoning" tasks suffer catastrophic failure. High-density reasoning—such as arithmetic derivations or complex coding tasks—requires the maintenance of Chain-of-Thought (CoT) coherence, where nearly every token serves as a critical logical link. Standard token-dropping mechanisms shatter these links, causing severe task-dependent degradation. |                                                      |                                                 |                                                  |                                     |
| The Evolution Toward Semantic Units                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |                                                      |                                                 |                                                  |                                     |
| Recognizing this, researchers have begun prioritizing "Semantic Units" over isolated tokens. ShotKV proves that preserving few-shot exemplars as atomic, indivisible semantic units dramatically improves reasoning robustness under aggressive compression, separating prefill preservation from dynamic decoding compression. Similarly, frameworks like ChunkKV and SemantiCache attempt to mitigate semantic destruction by segmenting the cache along natural language boundaries (phrases, sentences) and clustering semantically related tokens, rather than dropping them based on isolated attention scores.                                                                                                            |                                                      |                                                 |                                                  |                                     |
| However, even these advanced methods evaluate success based on generalized accuracy recovery. A true measurement of whether a cache manipulation changes the semantic interpretation of surviving tokens must move beyond string matching and explicitly assess:                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |                                                      |                                                 |                                                  |                                     |

- Referent Resolution: If a compaction boundary summarizes a character's
  background, does a pronoun used 5,000 tokens later still accurately bind to
  the correct entity attributes stored in the retained cache?
- Word-Sense Disambiguation: If an initial prompt strictly defines a common word
  with a highly specific, counter-intuitive definition for the scope of the
  task, does the model retain this constrained definition after the instruction
  has been compressed into a latent state?
- Established Stance/Constraints: Does an agent remember negative constraints
  (e.g., "Under no circumstances use the os module in Python") after the context
  undergoes repeated summarization cycles? The precise measurement of referent
  resolution, coreference tracking, and word-sense disambiguation across a
  synthesized boundary remains a nascent and largely unquantified frontier in
  current KV-cache literature. Current evaluation suites do not isolate the
  preservation of specific, write-time semantic values independent of the
  generative success of the sequence as a whole. In-Context Summarization and
  Generative Cache Retention In long-horizon agentic workflows—such as
  continuous software development or extensive data analysis—the context window
  inevitably saturates. As the trajectory lengthens, the accumulation of prior
  thoughts, tool calls, and observations leads to "context rot," where stale or
  erroneous tokens pollute the attention distribution and cause the model to
  lose track of its original intent. The standard industry response to this is
  context compaction via summarization. Scaffolding systems like Deep Agents,
  Lore, and various programmatic agents employ a "summarize-and-compact"
  routine. When a specific token threshold is breached (e.g., 85% of the context
  window), the LLM is prompted to generate a structured in-context summary of
  the preceding events, encompassing the original intent, key decisions,
  artifacts created, and next steps. The raw history is then flushed from the
  active prompt, and the generated summary replaces it, freeing up tokens for
  continued reasoning. The Re-encoding Penalty vs. State Retention In standard
  inference deployments, after the summary text is generated, it is treated as
  new user input and appended to the beginning of the subsequent prompt. This
  forces the model to perform a complete prefill operation to encode the summary
  from scratch. This approach has two massive flaws:
- Computational Waste: It requires an O(L^2) prefill computation to encode text
  the model itself just finished generating.
- Semantic Flattening: By forcing the summary back through the token embedding
  layer, all the deep, multi-layer latent representations (the "memoized notes"
  and rich hidden states) that the model built up while generating that summary
  are destroyed. The model must attempt to reconstruct those deep states purely
  from the flat text string. Advanced frameworks attempt to circumvent this
  rigidity. The SELFCOMPACT scaffolding, for instance, pairs an inline
  compaction tool with a lightweight rubric, allowing the agent to emit a
  <summarize> token to trigger compaction dynamically mid-derivation, rather
  than waiting for an arbitrary token threshold. From an infrastructure
  perspective, if engineered optimally, an inference engine can retain the KV
  cache that is generated during the autoregressive decoding of the summary
  itself. Because the summary is generated token-by-token, its KV cache is
  actively populated in the GPU's memory. Retaining this generative cache
  state—rather than parsing the text string back through a new prefill
  pass—bypasses the computational re-encoding penalty. The Insufficiency of the
  Summary Cache However, retaining the cache state of the summary simply
  preserves the semantics of the summary. It does not automatically retrieve or
  transplant the deep, write-time values of the original raw text that was
  summarized. The generated summary cache is, by definition, a heavily lossy
  distillation of the original context. Systems like Lore attempt to mitigate
  this by shifting from lossy summarization to "incremental distillation,"
  extracting highly specific knowledge entries (patterns, gotchas, exact file
  paths) and storing them in an external, search-enabled memory architecture.
  Yet, even here, when those specific facts are retrieved, they are injected as
  text, fundamentally losing their original, deep-layer write-time KV states. To
  maintain true semantic interpretation across the boundary, a mechanism is
  required to bridge the compressed summary directly to the original
  high-dimensional values of the raw text. Asymmetric Key and Value Treatment To
  understand how original write-time values might be preserved across a context
  boundary while utilizing freshly computed keys, it is imperative to dissect
  the asymmetric structural roles of Keys (K) and Values (V) within the
  transformer attention mechanism. In standard multi-head self-attention, the
  output of a layer is determined by:

This equation reveals a profound functional dichotomy. The Queries (Q) and Keys
(K) interact strictly within the softmax function to produce a probability
distribution—a set of scalar attention weights. Their role is purely geometric
and routing-oriented; they determine where the model looks. Conversely, the
Values (V) are linearly aggregated according to these weights. They carry the
actual representational content, the rich semantic payload that is transferred
forward through the residual stream to inform the next token generation.
Exploiting Asymmetry in the Literature Recent systems research has heavily
exploited this asymmetry, primarily to optimize memory compression and
quantization:

- Asymmetric Quantization Sensitivity: Research on algorithms like AsymKV, KIVI,
  and TurboQuant demonstrates that the transformer's output is vastly more
  sensitive to perturbations in the Key matrix than the Value matrix. Because
  Keys operate inside the exponential softmax function, slight quantization
  errors are magnified non-linearly, drastically altering the attention
  distribution. Values, being aggregated linearly, are highly tolerant to
  aggressive, lower-precision quantization. KIVI exploits this by applying
  per-channel quantization to Keys (to preserve outliers critical for routing)
  and per-token quantization to Values. TurboQuant pushes this further using
  data-oblivious polar coordinate rotations, spreading information evenly across
  dimensions before aggressive scalar quantization, yet maintaining asymmetric
  bit-widths (e.g., 4 bits for Keys, 2 bits for Values) to protect routing
  stability.
- Dimensionality Decoupling: Theoretical work on asymmetric attention suggests
  that the dimensionality of Queries and Keys (d_{\text{select}}) can be
  drastically reduced to O(\log N) because their only job is to rank N distinct
  patterns. Meanwhile, the dimensionality of the Values (d_{\text{value}}) must
  remain at the full model dimension to preserve complete representational
  content.
- Cross-Layer Fusion (FusedKV): Investigations into cross-layer KV cache sharing
  reveal a deep architectural asymmetry: values are predominantly derived from
  the bottom layers of the network (representing foundational, localized token
  semantics), while keys draw critical routing information from both the bottom
  and middle layers. FusedKV exploits this by deriving top-layer KV caches from
  a learnable, dimension-wise weighted fusion that treats keys and values
  distinctly. | Area of Optimization | Implementation Example | Key (K)
  Treatment | Value (V) Treatment | |---|---|---|---| | Quantization Precision |
  KIVI, AsymKV, TurboQuant | Higher precision (e.g., INT8 or INT4), per-channel
  grouping | Lower precision (e.g., INT4 or INT2), per-token grouping | |
  Dimensionality Reduction | Asymmetric Attention | Compressed to O(\log N)
  dimensions for ranking | Maintained at full d_{\text{model}} dimensions for
  representation | | Cross-Layer Sharing | FusedKV, FusedKV-Lite | Fused from
  bottom and middle layers for complex routing | Directly derived from bottom
  layers for foundational semantics | The Implications for State Transplantation
  Because Keys are bound to spatial geometry—heavily modulated by Rotary
  Position Embeddings (RoPE)—shifting a token's position during compaction
  typically destroys the validity of its Key. If a 10,000-token context is
  compacted into a 500-token summary, the relative distances between all
  subsequent queries and the historical keys are shattered. If the old keys are
  retained, the RoPE phases must be painstakingly re-applied or offset, a
  complex process prone to degradation outside of simple static splicing.
  However, because Values carry the semantic payload independent of the
  positional routing mechanism, an advanced theoretical architecture could
  selectively discard or recompute the Keys (to establish a new, valid geometric
  relationship with future tokens) while meticulously preserving the original,
  write-time Values. This asymmetric transplant would ensure that when the newly
  computed Keys are queried, the representations pulled forward are the exact,
  unadulterated, high-dimensional semantic constructs generated when the raw
  text was first processed, rather than a degraded representation filtered
  through a summary generation pass. While the theoretical groundwork for this
  exists in the documented asymmetry of K and V, current literature utilizes
  this asymmetry almost exclusively for differential quantization bit-widths or
  cross-layer parameter sharing. It is not currently utilized for synthesizing
  novel context boundaries through asymmetric recomputation, where original
  Values are paired with fresh Keys. Attention-Based Transplant vs. Trained
  Projections To bridge the gap between a highly compressed summary and the rich
  semantic detail of the original text, a system must possess a mechanism to
  retrieve historical states into the new compacted space. Many methodologies
  rely on trained projection modules to accomplish this. For example,
  AutoCompressors and cross-agent latent communication frameworks like C2C
  utilize explicit neural networks (e.g., MLPs or neural cache fusers) to
  compress, transform, and project cached states into new contexts. These
  projection modules must be trained via gradient descent, requiring extensive
  offline optimization, which severely limits their generalizability to frozen,
  off-the-shelf LLMs. An alternative, highly elegant approach is attention-based
  transplantation, which uses the model's native, frozen attention mechanics to
  retrieve old cached states into a new context.
- The Residual Stream as an Information Hub: Breakthrough theoretical work such
  as "The Residual Stream Is All You Need" demonstrates that the KV cache is,
  technically, mathematically redundant. Keys and values at every layer are
  deterministic projections of the residual stream, and any layer's KV pair can
  be bit-identically reconstructed from a single residual vector per token. This
  implies that a model's attention mechanism can natively route, project, and
  reconstruct past states if the residual stream is properly maintained,
  fundamentally proving that the network already possesses the internal hardware
  for deep state retrieval.
- Attention Matching and Latent Splicing: Methods like "Attention Matching"
  construct compact keys and values in latent space by directly optimizing them
  to reproduce the attention outputs and attention mass of the full context,
  without training a new neural network. Similarly, the "Models Take Notes at
  Prefill" framework demonstrates that localized KV states can be seamlessly
  spliced into new contexts by repositioning their RoPE phases and executing a
  brief attention-based seam-repair on the boundary tokens. This repair relies
  exclusively on the model's native attention to patch the cross-attention
  links, restoring coherence at the transition point. If a system were designed
  to use the model's own attention mechanism to query a localized subset of the
  original context (e.g., retrieving specific entities or constraints) and
  transplant those exact values into the active cache of the in-context summary,
  it would achieve zero-shot semantic continuity. It would do so without the
  need for auxiliary projection weights, relying entirely on the frozen
  parameters of the pre-trained model. Direct Assessment: Does the Unified
  Paradigm Exist? The user query posits a highly specific, unified architectural
  paradigm consisting of four strictly defined characteristics:
- Asymmetric Key/Value Treatment Across Contexts: Preserves or transplants
  original write-time KV-cache values across a context/compaction boundary,
  while keeping them paired with freshly computed keys to maintain valid routing
  geometry.
- Semantic-Continuity Measurement: Explicitly measures the preservation of deep
  semantic interpretation (referent resolution, word-sense disambiguation,
  strict constraints) rather than relying solely on speed, memory footprint, or
  next-token perplexity.
- In-Context Summary State Retention: Generates a summary in-context (while the
  full history is still attended) and directly retains its active cache state,
  fundamentally bypassing the need to re-encode the summary text from scratch.
- Attention-Based Transplant: Uses the model's native, frozen attention
  mechanics to retrieve and transplant the old cached state into the new
  compacted context, entirely avoiding trained projection modules or neural
  cache fusers. Conclusion Based on an exhaustive analysis of the provided
  academic and industry literature, a system matching this exact, unified core
  description is absent. While the individual mechanical prerequisites exist in
  isolated domains, their synthesis into a single operational framework has not
  been documented:
- Regarding Point 1 (Asymmetry): The literature robustly supports the asymmetric
  nature of Keys and Values, proving that Keys dictate routing and Values hold
  semantic representations. However, this asymmetry is currently leveraged
  almost exclusively for differential quantization (e.g., KIVI, TurboQuant) or
  cross-layer parameter sharing (e.g., FusedKV). The specific concept of
  computing fresh keys to bridge a compaction boundary while seamlessly pairing
  them with original, unadulterated write-time values is a brilliant theoretical
  corollary of current research, but it is not implemented in any cited
  context-compaction system.
- Regarding Point 2 (Semantic Metrics): The field is only just beginning to
  acknowledge the failure of standard metrics. Frameworks like KVFundaBench and
  ShotKV highlight the destruction of High-Density Reasoning (CoT coherence)
  caused by standard cache manipulation. Yet, these benchmarks are used to
  evaluate traditional token-dropping and prefill/decode separation, not the
  advanced asymmetric transplant mechanism described. True referent resolution
  metrics across synthetic boundaries remain absent.
- Regarding Point 3 (Summary Retention): Systems like SELFCOMPACT, Lore, and
  Deep Agents heavily utilize in-context summarization for long-horizon agent
  memory. Furthermore, state-of-the-art inference engines can theoretically
  preserve the KV cache of a generated sequence to avoid re-encoding. However,
  this is treated as an operational efficiency standard (continuous
  decoding/prefix caching) rather than a deliberate vehicle for transplanting
  historical values.
- Regarding Point 4 (Attention Transplant): The ability to splice, edit, and
  repair KV caches using the model's native attention mechanisms is proven by
  works like KVEraser and "Models Take Notes at Prefill". Yet, these are applied
  to localized edits and static skill composition, not to the dynamic retrieval
  of write-time values into an actively generated summary. In summary, the
  proposed architecture represents an exceptionally sophisticated, theoretical
  "holy grail" of context management. It correctly identifies the geometric
  fragility of Keys and the semantic richness of Values, proposing a method to
  decouple them to survive summarization boundaries. While the foundational
  physics of the Transformer attention mechanism (residual stream redundancy,
  K/V sensitivity divergence, latent state editability) fully support the
  viability of such a system, the comprehensive execution of this paradigm does
  not appear in the current body of research.
