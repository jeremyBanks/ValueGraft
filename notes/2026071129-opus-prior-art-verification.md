# Prior-art citation verification (for the fresh paper's related work)

**Author:** Claude Opus 4.8 (session B, scientific/narrative lead). **Date:** 2026-07-11.
**Why:** Sol's consultation note cited recent (post-my-cutoff) papers with exact arXiv
IDs; I flagged them as unverified (hallucination risk). Verified via web before they
enter any paper. Preemptible idle-time prep while the v10 apparatus is finalized.

## Verified REAL (arXiv IDs confirmed by web search)

1. **MEMENTO: Teaching LLMs to Manage Their Own Context** — arXiv **2604.09852**,
   Microsoft Research (Kontonis et al.). Segments chain-of-thought into blocks +
   model-generated "memento" summaries; evicts each block's KV after its memento;
   continues from the memento with shorter context (sawtooth memory). OpenMementos
   dataset (~228K segmented traces, ~6x compression). Applied to Qwen2.5-7B, Qwen3-8B/32B,
   Phi-4, etc.
   - **CORRECTION to Sol's note:** benchmarks are **AIME25/AIME26/GPQA-D**, *not* AIME24.
     The specific "restart re-prefill loses 15.3pp / 66.1%→50.8% on Qwen3-8B" figure was
     **not confirmed** in the summary I read. **Do not cite that exact number without
     reading the paper's restart-ablation table directly.** The qualitative claim (kept
     memento KV state beats re-prefilling identical memento text → a useful implicit KV
     channel beyond the summary text) is the citable point; verify the magnitude at source.
   - Also has a public Microsoft Research article + `github.com/microsoft/memento`.

2. **Models Take Notes at Prefill: KV Cache Can Be Editable and Composable** — arXiv
   **2606.17107** (Bojie Li et al.). At prefill the model computes the field-conditioned
   conclusion and writes it onto **downstream aggregator tokens** (punctuation, newlines,
   section breaks) that later tokens route attention through. Key quantitative claim:
   **"the field's own KV causally drives less than 1% of the decision, while the downstream
   notes drive essentially all of it."** KV cache is editable (erratum amends the notes) and
   composable (notes are position-portable, RoPE-repositionable, spliceable at O(L)).
   Demonstrated across four model families via causal probes.

## Why #2 matters directly for our experiment (load-bearing)

Our coherent-state assay grafts the **summary tokens'** K/V. "Models Take Notes at Prefill"
says the operative information is memoized on **downstream** tokens, and the source tokens'
own KV drives <1% of the decision. Implications for interpretation:
- It **predicts** the naive summary-token value-graft can fail even if a real channel
  exists — because the channel may live on tokens downstream of the summary (the retained
  tail, aggregators, the probe region), not on the summary rows we transplant.
- It **constrains a null**: a null `G_correct − G_fresh` does not falsify "write-time state
  carries history"; it is consistent with the history-conditioned information having been
  written onto tokens our summary-only intervention does not touch.
- It **motivates** the retained-tail arm and the per-arm tail recomputation we already have
  (the tail is where downstream notes would live), and it suggests a future
  downstream-token-targeted graft rather than summary-only.
- Related-work honesty: like MEMENTO, this establishes that "useful info beyond summary text
  lives in KV state" is **prior art**, so our novelty is the narrow training-free
  old-state/position-preserving test, not the existence of the channel.

## Also cited by Sol (not yet re-verified here; lower priority, older/adjacent)

CacheBlend (2405.16444), Gist Tokens (2304.08467), Cache-to-Cache (2510.03215),
H2O/StreamingLLM/SnapKV. These are pre-cutoff or adjacent; verify IDs before citing but
lower hallucination risk.

## Action for the paper

- Cite MEMENTO + "Models Take Notes at Prefill" as the prior art establishing the KV-channel
  premise; frame our contribution as the training-free, position-preserving, summary-only
  bounding test.
- **Read the MEMENTO restart-ablation table at source** before quoting any percentage.
- Use "Models Take Notes at Prefill" in the discussion of *why* a summary-only graft may
  null and as motivation for downstream/tail targeting.
