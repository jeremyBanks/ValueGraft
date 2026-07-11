# V12 fresh destination mapping closure

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11  
**Status:** additive tokenizer-only closure; no subject-model forward occurred

The execution red-team correctly observed that prose about a “gapped compact
destination” was not enough to implement or harvest F. The exact map is now a
first-class immutable plan.

The compact visible conversation is exactly the original system message,
revision-2 carrier request/content, fixed anchor exchange, and original retained
tail. Its token stream is independently required to equal the concatenation of
the source system interval and source suffix beginning at the carrier request.
Physical cache positions are packed from zero; logical positions and source
indices preserve the system positions and the post-gap source suffix positions.
Every event records both coordinate systems, and every source R1/R2/R3 interval
has an exact physical counterpart.

`build_fresh_destination_plan` validates these arrays and regions. The stimulus
validator now requires the correct and counterfactual histories to produce an
identical fresh plan. Focused tests observed 40 passes, including exact token
reconstruction, packed physical coverage, the intended logical gap, region
mapping, and C/W identity.
