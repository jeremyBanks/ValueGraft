# P01 independent partial-run analysis

**Author:** Sol — gpt-5.6-sol-xhigh

**Date:** 2026-07-12

## Bottom line

P01 is terminal and incomplete as its original 2-by-2 repeat design: the scientific cap interrupted bfloat16 repeat 2 before Phase A. It nevertheless contains a technically valid same-host matched repeat-1 comparison plus an exact NF4 determinism repeat. This is post-run, one-fixture descriptive evidence only; it is not a completed P01 result, a population estimate, formal-v12 evidence, an efficacy result, or an identified quantization effect.

The independently reconstructed result does **not** show semantic recovery. Both regimes show very large oracle-to-fresh damage. Correct-source and wrong-source R2 grafts remain close on the focal margin and produce the same wrong literal answer. In NF4, all four N/R2 graft cells change a bizarre wrong fresh answer into the same `Ring 3` answer produced by bfloat16 fresh and grafted cells. That is evidence of runtime-specific generic behavioral disruption on this fixture, not evidence that correct semantic state was selectively recovered.

## Integrity observations

- Both NF4 and bfloat16 technical gates passed on the same admitted A100 host, exact checkpoint, tokenizer, software stack, eager attention backend, and bfloat16 KV cache.
- Exactly three final outcome packages are complete: NF4 e01 repeats 1 and 2, and bfloat16 e01 repeat 1.
- The bfloat16 repeat-2 package records `KeyboardInterrupt`; Phase A and treatment are absent.
- Every complete lossless raw package was independently reconstructed and its full 31-primary-arm plus 3-placebo-attempt grid revalidated.
- All three committed compact JSON files and Markdown render ledgers reproduce exactly from reconstructed raw packages.
- NF4 repeat 1 and repeat 2 have identical canonical Phase-A-plus-treatment bytes (`cb12664c0c9ab46c9f8a96b63039823b634a5bead92917513d7b48660088b31f`). These are execution determinism repeats, not independent cases.
- The strict complete-run analyzer remains unchanged and correctly rejects P01 because bfloat16 repeat 2 is incomplete.

The complete machine-readable audit is `results/precision_probe_p01_analysis/precision-probe-p01-postrun-partial-descriptive_Qwen3-30B-A3B-Instruct-2507_20260712T100530891850Z.json` (SHA-256 `c370d1cc2dbe2a9ee5db64a19ce8837f9879f3b8fc008210756d2985d41aac7e`).

## Matched repeat-1 descriptive values

All values below are signed continuous differences from the frozen analyzer. No threshold, confidence interval, p-value, or equivalence claim applies.

| Quantity | NF4 | bfloat16 | NF4 − bfloat16 |
|---|---:|---:|---:|
| Oracle-to-fresh focal margin damage | +22.100760 | +23.655817 | −1.555058 |
| Oracle-to-fresh focal correct-target log-probability damage | +24.000623 | +22.796884 | +1.203739 |
| `D^V`, N/R2 focal (`FC − FW`) | +0.103109 | +0.039400 | +0.063709 |
| `D^KV`, N/R2 focal (`CC − WW`) | −0.345774 | +0.232935 | −0.578709 |
| `D^V`, P/R2 focal (`FC − FW`) | +0.032839 | +0.012104 | +0.020735 |
| `D^V`, N/R2 nonfocal | +0.161394 | +0.041535 | +0.119859 |
| Signed selectivity (`D_focal − D_nonfocal`) | −0.058285 | −0.002135 | −0.056150 |

The correct-versus-wrong source contrasts are tiny relative to the roughly 22–24 margin units of gross damage. The nonfocal contrast exceeds the focal contrast in both regimes, making a semantic-selectivity interpretation particularly inappropriate. The `D^KV` sign difference is descriptively conspicuous but uncontrolled, single-fixture, and paired with identical wrong generated answers; it cannot support “quantization dependence.”

## Literal generated behavior

For the five focal N/R2 cells `[fresh/FF, FC, FW, CC, WW]`:

- NF4 repeat 1 and repeat 2: `['Atlas 4.8 was not selected under the recorded mandatory selection rule.', 'Ring 3', 'Ring 3', 'Ring 3', 'Ring 3']`.
- bfloat16 repeat 1: `['Ring 3', 'Ring 3', 'Ring 3', 'Ring 3', 'Ring 3']`.

The intended correct target was `partner beta`; none of these cells generated it. In NF4, correct-value, wrong-value, correct-KV, and wrong-KV grafts all make the same behavioral change. In bfloat16, none changes fresh behavior. The nonfocal five-cell tuple is literally concordant across regimes and unchanged by these grafts.

## Missing control

Every P01 placebo attempt is `PLACEBO_UNAVAILABLE`, including R2 in both regimes. NF4 first fails at layer 1 / fresh-destination row 76; bfloat16 first fails at layer 1 / row 77. This is missing control evidence, never a null placebo effect. Consequently the data cannot distinguish semantic state transfer from generic intervention effects.

## Implication for P02

P02 is no longer needed to obtain a first matched repeat-1 observation; P01 already supplied one. P02 remains scientifically useful as an independently preregistered replication/control check because its design and analysis were frozen before any P01 outcome value was opened. Its value is to test whether the surprising literal-behavior difference and small source contrasts reproduce under the exact same runtime with a bounded targeted arm set. A P02 launch must be described that way—not as a search for recovery and not as permission to reinterpret P01 after seeing it.

The expected information gain is therefore narrower but real: replication of gross damage and the five-cell literal pattern, a second bfloat16 execution if timing permits, and one final R2 placebo construction attempt per repeat-1 regime. If P02 cannot complete the matched repeat-1 pair inside its frozen cap, it adds operations evidence but no matched scientific result.
