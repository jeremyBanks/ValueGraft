# The frozen wrong-history control is distributionally invalid

**Author:** Sol — OpenAI GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11  
**Status:** Raw-input audit finding before any coherent-state semantic run. No semantic outcome was generated with this control.

## Bottom line

The frozen `G_wrong` construction does preserve token count, structural tokens, summary positions, and destination shape. It does **not** preserve conversational coherence or a plausible wrong-history distribution. It repeatedly cycles short donor-message token pools to fill much longer target-message content slots. In the twelve frozen cases, 21–25 of 32 replaced message slots per case require more than one donor cycle; the most extreme slot repeats a seven-token acknowledgement 51 times.

Consequently, `G_correct - G_wrong` cannot cleanly identify correct-history-specific semantic state. A positive contrast could reflect state written after a coherent natural history versus state written after grotesquely repetitive corrupted text. Exact position matching does not remove that distributional confound.

This is a design failure, not an observed semantic result. The co-primary `GW` arm must be replaced before any semantic execution.

## Where the construction was frozen

[Amendment 1](../COHERENT-STATE-PREREGISTRATION-AMENDMENT-1.md) lines 77–100 explicitly freezes the rule:

1. take the token IDs from the corresponding-role donor message;
2. fill the target content slot using `donor[j mod len(donor)]`;
3. truncate a longer donor or cycle a shorter donor;
4. cycle donor messages by ordinal if necessary.

The implementation is [src/coherent_state_tokens.py](../src/coherent_state_tokens.py), `matched_wrong_prefix_ids`, especially the construction:

```python
replacement = [int(pool[j % len(pool)]) for j in range(end - start)]
```

Later amendments and validators persist and reconstruct cycle counts, hashes, structural equality, and special-token exclusion. Those checks prove that the corrupted construction was reproduced faithfully; they do not make its content a valid scientific control.

## Direct audit of all frozen pairs

The table below was recomputed from the committed `data/synthetic/c*.json` files with the exact production tokenizer revision and `matched_wrong_prefix_ids`. `Slots cycled` counts replaced message slots whose donor content had to repeat at least once. `Maximum cycles` is the largest number of repetitions required by any slot.

| Target | Donor | Prefix tokens | Replaced slots | Slots cycled | Maximum cycles |
|---|---|---:|---:|---:|---:|
| c10 | c13 | 8,430 | 32 | 23 | 26 |
| c02 | c14 | 8,385 | 32 | 24 | 20 |
| c01 | c15 | 8,855 | 32 | 25 | 24 |
| c04 | c16 | 8,595 | 32 | 21 | 29 |
| c07 | c17 | 8,600 | 32 | 22 | 30 |
| c11 | c18 | 9,381 | 32 | 23 | 23 |
| c05 | c25 | 8,556 | 32 | 22 | 20 |
| c09 | c26 | 9,195 | 32 | 22 | 25 |
| c06 | c27 | 8,876 | 32 | 21 | 22 |
| c12 | c28 | 9,509 | 32 | 23 | 43 |
| c08 | c29 | 8,525 | 32 | 24 | 51 |
| c03 | c30 | 8,913 | 32 | 24 | 47 |

The cycling is therefore not a rare boundary accommodation. It is the dominant construction for most replaced messages in every case.

## Literal worst-case example

For target c08 with donor c29, target message index 22 is a 352-token assistant response. The mapped donor pool at message index 22 contains seven tokens decoding to:

> Got it, logged for reference.

The control fills the 352-token target slot by repeating that sentence 51 times. The beginning of the resulting content is:

> Got it, logged for reference.Got it, logged for reference.Got it, logged for reference.Got it, logged for reference…

The target slot instead begins with a natural answer about migrating stored payment credentials. These are not distribution-matched histories.

## What remains valid

The implementation successfully enforces the mechanical properties it claims:

- correct and wrong prefixes have identical token counts;
- all chat-structural, system, retained-tail, request, and header tokens remain fixed;
- only declared evicted-content slots change;
- no replacement introduces chat special tokens;
- correct and wrong summaries begin at identical positions.

Those properties are valuable construction evidence. They are insufficient for the semantic claim assigned to `GW`.

## Claim consequence

The old arm could at most test:

> Does state obtained after the natural correct history outperform state obtained after a position-matched but severely corrupted repetitive prefix?

It cannot establish:

> Does the summary state specifically retain information from the correct history rather than a coherent wrong history?

Because `GW` was a co-primary contrast, the planned co-primary intersection is currently undefined as a valid test of history specificity. No amount of schedule robustness, hashing, or larger-model execution repairs this input-control defect.

## Required redesign properties

A replacement wrong-history source must be frozen before outcomes and must satisfy all of the following:

1. coherent natural text throughout the source;
2. exact position and call-schedule matching to the correct source;
3. identical summary token IDs forced at the same positions;
4. removal or counterfactual alteration of the target history facts without broad corruption;
5. no cyclic filler, templated repetition, or conspicuous distributional anomaly;
6. independent human/model review of the literal decoded wrong histories before execution;
7. negative controls showing that any effect is tied to the relevant counterfactual content rather than generic source degradation.

The cleanest candidate is a minimally counterfactual version of each target history: preserve the conversation and alter only the planted facts to token-length-matched alternatives. A fully unrelated but token/turn-length-matched coherent donor is a broader control, not a substitute for the minimal counterfactual. The final choice requires fresh preregistration and literal-content review.

## Reproduction command

The audit used the exact production tokenizer and, for every frozen target/donor pair, called `matched_wrong_prefix_ids`, then counted `replacement.cycles > 1` and the maximum `replacement.cycles`. The worst-case decoded example was selected by `max(replacements, key=lambda row: row.cycles)` for c08/c29. No model forward pass or semantic score was involved.
