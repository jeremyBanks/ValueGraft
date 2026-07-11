# V12 turn-addition schedule correction

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11  
**Status:** additive pre-forward correction; no subject-model forward occurred

## Observed defect

While personally tracing the tokenizer-only replay planner against the
preregistered persistent-chat protocol, I found that the first implementation
split structural material into separate calls: system message, user message,
assistant header, assistant close, and the next user message were each separate
prefills. The preregistered schedule instead requires the actual turn-addition
shape:

1. initial system/user material plus the first assistant generation header in
   one prefill;
2. assistant content forced q=1;
3. the completed assistant close plus the next user message and next assistant
   header in one continuation prefill.

This distinction is scientifically material because the repository has already
observed deterministic query-shape arithmetic at the scale of the hypothesized
effect. The earlier planner therefore did not implement schedule N as stated.

## Correction and observed verification

Commit `b51db97` changed only the tokenizer/event planner and its focused tests.
The corrected event stream now constructs one initial generation-prefix call,
q=1 assistant-content calls, and one structural continuation call between
assistant responses. Focused tokenizer/control/review tests observed 28 passes.

All six committed literal case files were then revalidated through the corrected
planner. The validator observed `MECHANICAL_DRAFT_PASS` for all six and wrote:

`results/coherent_canary_validation/coherent_canary_stimuli_turn_addition_schedule_Qwen3-30B-A3B-Instruct-2507_20260711T201253Z.json`

No language-model weights were loaded and no subject-model forward was run.

## Provenance consequence

The earlier per-author and combined tokenizer-validation artifacts faithfully
record the older fragmented event geometry and are now **nonauthorizing
historical evidence**. They are retained rather than overwritten. The literal
stimulus texts did not change, so the already-sealed content-review packets
remain bound to the same source-file hashes. Any later frozen execution manifest
must cite the corrected all-six validation artifact above and must not cite an
earlier geometry hash as the executable schedule.
