# V12 schedule P closure

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11  
**Status:** additive pre-forward closure; no subject-model forward occurred

The independent execution red-team correctly found that the preregistration
named schedule P but did not define an executable complete v12 event stream.
The old v11 helper ended at summary generation and could not define the revised
anchor, event-aligned R2/R3, or retained-tail suffix.

V12 P is now frozen independently. Each complete historical message before the
carrier is block-prefilled (with explicit <=4096 pieces). The carrier request
and assistant header are the next block. From carrier content onward, P reuses
the exact N events: carrier q=1, the complete carrier-close/anchor-user/header
call, acknowledgment q=1, and retained-tail continuation. P therefore changes
only the imported-history write schedule. It neither changes visible tokens nor
creates a competing post-carrier protocol.

`build_turn_aligned_plan` constructs this stream and `validate_case` now requires
exact correct/counterfactual P geometry in addition to N geometry. Focused tests
observed 38 passes after the closure. The revised all-six mechanical artifact
must be regenerated after this change; the prior revision-2 artifact remains
valid for its recorded N geometry and prose hashes but cannot by itself
authorize P execution.
