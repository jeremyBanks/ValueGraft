# V7 semantic aggregate validator — implementation provenance

**Author:** Sol — `gpt-5.6-sol-xhigh`

**Date:** 2026-07-11

**Implementation commit:** `7b6a99d4f8fd82cf19340310f7c7f277f5fffd51`

The semantic aggregate reconstruction added to
`scripts/validate_coherent_harvest.py` in commit `7b6a99d` was implemented and
tested by Sol during the independent v7 science audit. A shared-worktree commit
race included it in the concurrently prepared commit titled “coherent-state:
reconstruct donor evidence at harvest.” Repository policy forbids rewriting
history, so this additive note records the exact provenance instead.

The validator independently recomputes, from persisted token log-probabilities:

- every target mean log-probability;
- every correct-minus-counterfactual plant margin;
- every arm-level conversation margin;
- every `conversation_outcomes` value from its arm score; and
- all calibration detail and outcome aggregates.

It also requires identical plant identities, categories, probes, and order
across semantic arms and across calibration arms. Negative regressions mutate
each decision-bearing aggregation layer and require independent harvest to
reject it.

This closes only the semantic-aggregate reconstruction item from the science
audit. It does not claim to resolve the separately identified actual gapped-
destination schedule-composition gate or the snapshot-persistence decision.
