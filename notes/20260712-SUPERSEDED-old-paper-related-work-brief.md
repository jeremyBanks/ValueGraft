# RELATED-WORK BRIEF (extracted from the 4 prior-art notes, 07-09)
(Structured digest by Explore subagent from notes/2026070853-55 + 2026070629.)

## Closest single work
"Models Take Notes at Prefill: KV Cache Can Be Editable and Composable" — Bojie Li,
Pine AI, arXiv 2606.17107 (2026). Edits/composes KV via append-only errata + skill
transplant, RoPE-re-rotating keys, values unrotated ("values are position-free").
Differs: transplants precompiled skills into fresh contexts, no in-context summary
retained across a compaction boundary, decision-identity eval not referent/sense/
stance. Cite as closest MECHANISM; distinguish on setting + evaluation axis.

## Categories and exemplars (each with the notes' stated distinction)
- KV eviction/compression (StreamingLLM, H2O 2306.14048, SnapKV, PyramidKV, ChunkKV
  2502.00299, SentenceKV 2504.00970, SemantiCache 2603.14303, etc.): drop/keep whole
  KV pairs for efficiency; none preserve values with fresh keys; none measure
  reinterpretation of retained text.
- Cross-context KV reuse/blending (CacheBlend 2405.16444, Prompt Cache 2311.04934,
  EPIC 2410.15332, KVLink 2502.16002, KV Packet 2604.13226, KVCOMM 2510.12872):
  efficiency-oriented reuse across documents/requests; not summary-boundary; no
  semantic-continuity metrics.
- Learned latent compression (gist tokens Mu 2023, AutoCompressors, ICAE, Activation
  Beacon 2401.03462, 500xCompressor, Cartridges 2506.06266, compressed context
  memory): train NEW soft tokens/beacon KV — opposite of preserving originals.
  Activation Beacon = nearest "summary-with-cache" but its state is LEARNED.
- Asymmetric K/V treatment (AsymKV 2506.05410, KVSlimmer 2603.00907, KIVI, FusedKV):
  asymmetry used for quantization/sharing, never for fresh-keys+original-values at a
  compaction boundary.
- Memory-augmented transformers (Memorizing Transformers 2203.08913, InfLLM, LongMem):
  strongest prior art for attention-based retrieval of preserved write-time KV — but
  APPEND to extend context, don't REPLACE re-encoded summary text; perplexity/NIAH
  metrics.
- Cache compaction synthesis (Fast KV Compaction via Attention Matching 2602.16284,
  Still 2606.07878): synthesize compact caches, not preserve originals.
- Cross-model transfer (DroidSpeak 2411.02820, C2C, XC-CACHE): trained projections,
  different problem.
- Evaluation gap: KVFundaBench/ShotKV 2502.01941, "Hold Onto That Thought" 2512.12008
  measure aggregate accuracy; ConstraintRot/"Governance Decay" 2606.22528 is the ONLY
  stance/constraint-under-compaction work but tests constraint SURVIVAL vs DROP, not
  reinterpretation of retained text. "Rethinking KV Cache Compression" survey
  2503.24000 explicitly notes the per-example response-quality gap.
- Provider APIs (Note 4): OpenAI /responses/compact + encrypted compaction item +
  context_management/compact_threshold; Anthropic compact_20260112 compaction block +
  pause_after_compaction; Gemini managed-agent compaction, thought signatures, Live
  API compression. Validate the "summary + opaque handle" product shape; mechanism
  unknown — never claim providers do ValueGraft, never claim API-pattern novelty.

## Convergent novelty verdict (3 searches)
Composite absent as an established line; every ingredient present in fragments. The
clearest gap = the EVALUATION AXIS: no benchmark measures coreference re-binding /
word-sense shift / stance drift of RETAINED text after cache manipulation. Caveats:
many closest works are unreviewed 2026 preprints (2606.17107, 2606.07878, 2606.22528…);
some full texts unfetchable (claims provisional).

## Recommended terminology (findability)
"semantic continuity of retained tokens"; "value-preservation-with-key-recompute";
"write-time cached attention value state"; "summary-boundary experiment";
"training-free". Avoid: "no one thought of opaque compaction handles"; "novelty =
KV caches contain semantic information"; "providers are doing ValueGraft".
