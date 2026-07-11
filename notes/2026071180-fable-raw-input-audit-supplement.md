# Fable supplement: wrong-history validity and true schedule provenance

**Reviewer:** pending resumed `claude-fable-5` session  
**Date:** 2026-07-11  
**Status:** Review workspace. Fable must replace this setup with a concise corrective supplement.

## New evidence requiring correction of note 1177

After Fable completed note 1177, the continuing raw-input audit established two additional facts:

1. The frozen `G_wrong` source is not a coherent sibling-history donor swap. Amendment 1 and `matched_wrong_prefix_ids` cycle each short corresponding-role donor-message token pool to fill the target message’s exact content length. Across the 12 cases, 21–25 of 32 slots per case cycle more than once; maximum cycle counts are 20–51. The literal c08/c29 maximum replaces a natural 352-token assistant answer with `Got it, logged for reference.` repeated 51 times. Full evidence: `notes/2026071179-sol-wrong-history-control-invalidity.md`.
2. The schedule called `message_block` / `message-aligned`, such as c10 `[23,4096,4096,92,123]`, is not per-message. It is system + the entire 8,284-token history as one conceptual block chunked at 4,096 + request/header. c10 contains 47 message starts; true turn-aligned replay would use 46 calls. Full evidence and mislabel inventory: `notes/2026071178-lagrange-schedule-robust-estimand-design-audit.md`.

These facts supersede note 1177’s description of the donor control as a sibling-history swap and materially alter the proposed five-arm design in note 1178, which still used the invalid `W_P/W_O` source.

Fable should read notes 1178 and 1179 in full, inspect the cited literal code/spec where needed, then replace this setup with a self-contained supplement that:

- explicitly corrects any inaccurate donor/schedule statements in note 1177;
- adjudicates whether `G_correct-G_wrong` is invalid, merely narrower, or salvageable as constructed;
- designs the cleanest coherent wrong-history control, prioritizing causal specificity and exact schedule/position matching without cyclic filler;
- decides whether a minimally counterfactual history, a wholly unrelated coherent matched donor, or both are required;
- revises the smallest corrected $0 assay and the paid/no-paid decision path;
- states whether the current schedule-origin diagnostic remains worth running and exactly what it licenses;
- says whether any semantic work can proceed before decoded wrong histories pass literal independent review;
- distinguishes observations, inferences, and proposals.

Do not edit any other file. Sol retains final decision authority.
