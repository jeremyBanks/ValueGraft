# Final-paper related-work primary-source verification

**Author:** Sol — OpenAI GPT-5.6 Sol, extra-high reasoning

**Date:** 2026-07-12

**Scope:** read-only verification of the claims in `PAPER.md` §2 and the
SWE-Gym provenance sentence against the cited authors' primary papers. No
secondary source was used to authorize a paper claim.

## Findings

- **MEMENTO** (`arXiv:2604.09852`) supports the paper's bounded description.
  Its Table 3 reports Qwen3-8B AIME24 pass@1 of 66.1% under normal memento
  attention and 50.8% after restart/re-prefill, a −15.3 percentage-point
  difference. The normal number comes from 64 repetitions and restart from 8.
  The restart condition first generates identical memento text with block
  access, then discards and recomputes KV without the summarized block. The
  paper appropriately calls this strong evidence in a trained full-KV setting,
  not an independent replication of our transplant.

- **Models Take Notes at Prefill** (`arXiv:2606.17107`) supports the paper's
  description that the answer-relevant field's own KV drives under 1% of the
  decision in the tested tasks, while downstream aggregator/delimiter notes
  carry the conclusion. The primary source reports the causal mechanism across
  Qwen3, Llama-3.1, Gemma-2, and Mistral families. Our wording remains
  appropriately conditional when extrapolating this result to the carrier
  construction.

- **CacheBlend** (`arXiv:2405.16444`) supports the systems contrast: it combines
  precomputed chunk caches and selectively recomputes a subset of tokens to
  restore cross-context interactions. The paper does not use CacheBlend as
  direct evidence for our intervention.

- **Cache-to-Cache** (`arXiv:2510.03215`) supports the statement that it learns
  a projection and gating mechanism to fuse a source model's KV cache into a
  target model. It is adjacent learned cross-model transfer, not a test of
  training-free within-model replacement.

- **Learning to Compress Prompts with Gist Tokens** (`arXiv:2304.08467`)
  supports the statement that gist tokens are learned prompt compression whose
  cached representations are reusable; it is correctly separated from
  post-hoc state salvage.

- **SWE-Gym** (`arXiv:2412.21139`) supports the provenance description used in
  §5/A.2: the authors report 491 successful OpenHands trajectories obtained by
  rejection sampling from `gpt-4o-2024-08-06` and
  `claude-3-5-sonnet-20241022` under different temperatures. The local parquet
  still lacks row-level generator IDs and immutable upstream revision, so the
  paper correctly attributes this only to the source paper rather than claiming
  the local file proves it.

- PyTorch's numerical-accuracy documentation remains a general systems source,
  not evidence for the repository's measured query-shape magnitudes. The paper
  makes that boundary explicit.

## Disposition

No related-work correction was required. The novelty statement remains scoped
as a non-systematic search and does not claim the broad state-beyond-text premise
as original. Recheck only if a cited preprint version changes before external
publication; repository publication can cite the exact arXiv identifiers above.
