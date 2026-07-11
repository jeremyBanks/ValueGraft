# V12 path control must be bf16-representable by construction

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11  
**Status:** additive pre-forward correction; no subject-model forward occurred

The first technical path-control design normalized the complete multi-million-
element R2 K/V gradient to unit L2 and tried absolute epsilons only through 0.1.
For a distributed direction, typical per-element edits would be far below one
bf16 unit in the last place, especially for large-magnitude key values. The gate
could therefore fail because every edit rounded back to the original cache—not
because the public intervention/readout path was insensitive.

Revision 2 uses a quantization-aware construction. In each nonzero
layer/channel/token gradient row, it deterministically selects the lowest-index
maximum-gradient scalar and steps that stored bf16 value by a fixed number of
representable ULPs in the positive or negative gradient direction. The frozen
ULP sequence is `[1, 2, 4, 8, 16, 32, 64]`. Every attempted coordinate, gradient,
bf16 bit pattern, and actual tensor delta is persisted. Detached rows are still
destroyed/reconstructed/reinserted through the public R2 path before scoring.

This is deliberately an engineered plumbing control, not a natural semantic
perturbation. Its purpose is to prove that representably different released
rows can move a frozen downstream margin in both prescribed directions. It does
not estimate a natural history channel and cannot support a scientific positive
by itself.
