# V12 retained regions must end at model-call boundaries

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11  
**Status:** additive pre-forward design correction; no subject-model forward occurred

## Defect found by causal execution tracing

After repairing turn-addition query shapes, the primary R2 definition still
ended immediately after the carrier's canonical close. Under the corrected
schedule, that close is computed in one structural forward together with the
fixed anchor-user message and the next assistant generation header.

That made R2 end **inside a model call**. A causal intervention cannot replace
the close rows and then recompute later rows from the same call: those later
rows were computed simultaneously, and splitting them into a new call would
change the query shape whose numerical importance this project has already
observed. The prior region was therefore not implementable under its own source
schedule without either stale descendants or a schedule change.

## Corrected event-aligned regions

Revision 2 uses only event boundaries:

- R1: carrier assistant content, ending after its q=1 tokens;
- R2: R1 plus the **entire** next structural call—carrier close, fixed anchor
  user message, and anchor-assistant generation header—ending before the
  q=1 acknowledgment content;
- R3: R2 plus the fixed `Acknowledged.` content forced q=1. The anchor close and
  retained-tail continuation remain visible in all arms but are recomputed.

Thus downstream recomputation always begins with a complete original event.
The primary claim boundary is correspondingly narrower and more honest: R2 is
not content-only or content-plus-close. It includes the target-neutral anchor
prompt/header rows, which are exactly the kind of downstream note positions
recent prior work suggests may carry conclusions.

The schema fields are now `content_end`, `anchor_prefix_end`, and
`anchor_content_end`; old `close_end`/`anchor_end` geometry is historical and
nonauthorizing. Focused tests observed 31 passes after the change. No source
stimulus outcome, subject cache, or model score existed when this correction was
made.
