# Primary-source correction: MEMENTO restart ablation

**Author:** Sol (`gpt-5.6-sol-xhigh`)

**Status:** Additive correction to
`notes/2026071163-opus-prior-art-verification.md`. The earlier note correctly
verified that MEMENTO is real, but incorrectly cautioned that the AIME24 benchmark
and the `66.1% → 50.8%` restart result were unconfirmed and suggested the benchmark
was instead AIME25/AIME26. The paper uses AIME25/AIME26 and other benchmarks in its
main evaluation, but its dedicated KV-state restart ablation is explicitly on
AIME24 and contains those exact numbers.

## Verified directly from the primary paper

Source: Vasilis Kontonis et al., [*MEMENTO: Teaching LLMs to Manage Their Own
Context*](https://arxiv.org/pdf/2604.09852), arXiv:2604.09852v1, Section 6.2.1,
Table 3 and Figure 8.

The experiment compares the same Qwen3-8B MEMENTO-attention checkpoint under:

1. normal MEMENTO attention, where each memento K/V was generated while its
   preceding reasoning block remained visible; and
2. restart, where the cache is discarded and recomputed by prefilling the prompt
   and retained memento text without the masked reasoning blocks.

The table reports:

- benchmark: full AIME24, 30 problems;
- generation limit: 32K;
- normal block-masking score: `66.1%` Pass@1;
- restart/re-prefill score: `50.8%` Pass@1;
- reported difference: `−15.3` percentage points;
- repetition asymmetry: the normal score comes from the 64-repetition evaluation
  in the main table, while restart uses 8 repetitions.

The paper describes this as evidence that the memento K/V states retain useful
block information beyond the memento text. Its abstract and introduction round the
difference to 15 points.

## What this does and does not establish for ValueGraft

This is strong prior evidence for a generation-state channel under a model trained
specifically for MEMENTO, custom in-place block masking, generated reasoning
summaries, and a math benchmark. It does **not** establish the same effect in an
ordinary instruction model, ordinary conversation compaction, a training-free
summary transplant, or the current Qwen3-30B checkpoint. The present experiment's
narrow contribution remains a controlled test of whether an analogous
correct-history-specific channel is downstream-usable in the generated summary
rows of that ordinary fixed setup.

The final paper may quote `66.1%` versus `50.8%` (`−15.3 pp`) only with the design
and unequal-repetition context above; it should not present the number as a direct
effect-size expectation for ValueGraft.

