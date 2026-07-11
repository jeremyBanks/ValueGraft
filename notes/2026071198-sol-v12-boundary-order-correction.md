# V12 carrier-boundary ordering correction

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11  
**Status:** second additive pre-forward correction; no subject-model forward occurred

## Observed defect

After correcting structural call shapes, I traced the full case ordering rather
than only the event widths. The tokenizer validator was still passing the entire
authored conversation to the carrier planner and appending the carrier/anchor at
the end. That placed the supposedly retained tail *before* the compaction
carrier. It also made `R3_anchor` extend to the end of the entire stream, silently
including every retained-tail row.

The intended experiment requires:

`evicted prefix -> carrier request/content -> anchor exchange -> retained tail`

The prior order did not instantiate the compaction boundary in the
preregistration and could not authorize any inference.

## Correction and observed verification

Commit `7cb98b4` makes `middle_end_msg` an explicit planner input, inserts the
carrier and anchor immediately after that frozen boundary, appends the
byte-identical retained tail afterward, and ends R3 exactly at the retained-tail
user-message start. New tests exercise literal ordering, the R3 endpoint, event
coverage, and matched correct/counterfactual geometry. The focused suite observed
30 passes.

All six literal cases then observed `MECHANICAL_DRAFT_PASS` under the corrected
order. The current all-six artifact is:

`results/coherent_canary_validation/coherent_canary_stimuli_boundary_order_Qwen3-30B-A3B-Instruct-2507_20260711T201434Z.json`

Its observed `(R1 start, R1 end, R2 end, R3 end, original authored-history
tokens)` tuples are:

- e01: `(695, 738, 740, 768, 1019)`
- e02: `(599, 642, 644, 672, 1037)`
- e03: `(837, 880, 882, 910, 1120)`
- e04: `(1891, 1934, 1936, 1964, 4043)`
- e05: `(690, 733, 735, 763, 1021)`
- e06: `(972, 1015, 1017, 1045, 1556)`

No model weights were loaded and no subject-model forward occurred.

## Provenance consequence

Both the original per-author validation artifacts and the intermediate
turn-addition artifact named in `notes/2026071197-sol-v12-turn-addition-schedule-correction.md`
are now nonauthorizing historical evidence. The latest boundary-order artifact
above is the only current mechanical schedule record. Literal case files and
their content-review packets did not change, so content-review provenance
remains valid. A future execution manifest must bind commit `7cb98b4` or a
descendant with identical planner inventory and the latest artifact hash.
