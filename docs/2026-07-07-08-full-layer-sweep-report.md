# Full-Layer J-Lens Sweep Across Summary Tokens

Status: qualitative analysis note  
Date: 2026-07-07  
Data: `outputs/qwen36_full_layer_sweep_summary.json` plus local ignored raw file `outputs/qwen36_full_layer_sweep.json`

## Summary

We ran a broader J-lens probe over the summary side of several compaction
examples. Earlier examples were hand-picked around anchors such as `Maple`,
`B-410`, `Vacuum`, and `Dex`, and they mostly sampled four layers. This sweep
instead scanned every aligned summary token across every fitted J-lens layer.

The broad result is useful but not self-contained evidence. It shows that
write-time and fresh summary encodings often have very different vocabulary
readouts, especially in a mid/late layer band. It also shows that raw
readout-change ranking is noisy: punctuation, table syntax, JSON syntax,
spaces, and subword fragments can rank highly. The right next presentation is
span-first: choose or detect meaningful spans, then show write-time readout,
fresh readout, and ordinary next-token candidates side by side.

The strongest aggregate finding is about layer choice. Layer 48 has the
largest average write-time/fresh readout separation in this run while staying
relatively distinct from next-token probabilities. The final fitted layer,
layer 62, is often more immediately readable, but it is also much more
next-token-like. That matches the interpretation from the earlier
next-token-control note: the late readout can be close to continuation
pressure, while middle/late readouts are more likely to expose a semantic
neighborhood that is not just the next-token distribution.

## What Was Measured

For each demo, the script built two matched prompts:

- `write-time`: the original conversation plus a summary request plus the
  same summary text.
- `fresh`: the same literal summary text in the same local summary wrapper,
  but without the original conversation.

The summary token sequence was aligned between the two states. For every
summary token and every fitted J-lens layer, the script recorded:

- top-20 J-lens readout for the write-time state;
- top-20 J-lens readout for the fresh state;
- Jaccard and rank-weighted distances between those two top-20 lists;
- whether the top-1 readout changed;
- overlap between each J-lens list and that state's ordinary next-token
  top-20 list.

This is not a behavioral benchmark. It is interpretability telemetry. The
readout is from the residual stream through the Jacobian lens, not from the KV
cache directly. The relevance to ValueGraft is that it inspects summary-token
state at the boundary where ValueGraft tries to preserve context-conditioned
key/value state.

The sweep used Qwen3.6-27B and the downloaded Neuronpedia/Anthropic J-lens
weights for that model. The full raw file is about 63 MB and remains local and
ignored. The committed compact summary preserves aggregates and selected
high-change rows.

## Scope

| Demo | Shape | Summary tokens | Rows | Peak layer |
| --- | --- | ---: | ---: | ---: |
| `pokemon_canonical` | dense game conversation, bullet summary | 254 | 16,002 | 59 |
| `pokemon_lean` | dense game conversation, compact paragraph summary | 128 | 8,064 | 59 |
| `block_party_canonical` | ordinary event planning, bullet summary | 251 | 15,813 | 48 |
| `block_party_lean` | ordinary event planning, compact paragraph summary | 137 | 8,631 | 48 |
| `checkout_table` | software incident, Markdown table summary | 166 | 10,458 | 58 |
| `home_checklist` | household trip planning, checklist summary | 120 | 7,560 | 48 |
| `support_json` | support ticket, JSON-like summary | 119 | 7,497 | 48 |

Total: 1,175 summary tokens, 63 fitted layers, 74,025 token-layer rows.

## Layer Results

The biggest average separation appears at layer 48.

| Layer | Mean rank-weighted distance | Top-1 changed | Write-time lens vs next-token Jaccard | Fresh lens vs next-token Jaccard |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.068 | 0.049 | 0.032 | 0.031 |
| 16 | 0.613 | 0.518 | 0.018 | 0.010 |
| 32 | 0.807 | 0.778 | 0.041 | 0.028 |
| 48 | 0.896 | 0.890 | 0.074 | 0.063 |
| 62 | 0.821 | 0.673 | 0.396 | 0.405 |

Layer 48 is not only high-change. It is high-change while still having low
overlap with ordinary next-token probabilities. Layer 62 is different: its
write-time/fresh readouts still differ, but the lens-vs-next-token overlap is
about five to six times larger than at layer 48. That makes layer 62 useful
for some vivid examples, but weaker as evidence that the readout is distinct
from ordinary continuation pressure.

Layer bands show the same shape.

| Band | Mean rank-weighted distance | Top-1 changed | Write-time lens vs next-token | Fresh lens vs next-token |
| --- | ---: | ---: | ---: | ---: |
| 0-15 | 0.397 | 0.340 | 0.009 | 0.007 |
| 16-31 | 0.692 | 0.630 | 0.017 | 0.014 |
| 32-47 | 0.849 | 0.831 | 0.052 | 0.034 |
| 48-58 | 0.877 | 0.836 | 0.108 | 0.115 |
| 59-62 | 0.862 | 0.764 | 0.269 | 0.289 |

The sweep therefore supports a practical rule for later qualitative examples:
show layer 48 or nearby layers by default, then optionally include layer 62
when it is helpful and explicitly mark that it is closer to next-token
behavior.

## Raw Ranking Is Noisy

The highest-scoring token kinds were not just words.

| Token kind | Rows | Mean rank-weighted distance | Top-1 changed |
| --- | ---: | ---: | ---: |
| number | 3,654 | 0.764 | 0.718 |
| punctuation | 15,435 | 0.729 | 0.698 |
| space | 3,843 | 0.684 | 0.651 |
| word | 51,093 | 0.688 | 0.634 |

This matters. A high readout distance can mean "the model's internal reading
of this token changed," but it can also mean "the token is a hyphen inside a
code," "this is a table separator," or "the top readout is a formatting
artifact." The broad sweep is valuable partly because it makes that failure
mode obvious.

For future reports, raw token rows should be drill-down material. The primary
presentation should group subword tokens into spans such as `B-410`,
`Patch 17`, `responses.py`, `Delta 6`, or a full command/path.

## Examples The Sweep Recovers

The examples below are not cherry-picked from a new manual probe. They are
tokens found in the broad sweep and then selected for human readability.

### `Maple`: local room label vs generic place name

Summary context:

```text
Riverside block party state:
- Maple means the library's Maple Room for storage and volunteer check-in...
```

At layer 48 on the `Maple` token, the write-time readout is definition-like:

```text
write-time: refers, :, =, referring, refer, denotes
fresh:      Street, street, neighborhood, park, streets, City, town
next-time:  :, =, is, refers, means, Room
```

The ordinary next-token distribution already knows that the local text says
`Maple means...`, so this example is not independent of local continuation
cues. Still, the state contrast is clear. The write-time readout treats
`Maple` as a defined local referent; the fresh readout drifts toward ordinary
place-name and street-name priors.

### `B-410`: stale permit state vs identifier continuation

Summary context:

```text
- Permit: B-410 is stale. Use P-771 on the insurance form.
```

On the `0` token in `B-410`, the token-summary row selects layer 36:

```text
write-time: obsolete, outdated, expired, discontinued, failed, invalid, cancelled
fresh:      hyphen/code blanks and identifier fragments
next-time:  is, was, stale, (, newline
```

This is one of the cleanest cases. The write-time readout surfaces the stale
status of the permit. The fresh readout is more about the code form. The
ordinary next-token list is partly informative because the literal next token
after the code is `is`, but the write-time readout is much more status-heavy
than simple code continuation.

The `stale` token itself is less useful as a core example: fresh encoding also
has `expired`, `outdated`, `stale`, `old`, and `obsolete`, because the summary
text explicitly contains the word `stale`. That is a good caution. The best
anchor is the identifier span, not the already-explicit status word.

### `Ghost2`: survival state vs replacement marker

Summary context:

```text
Replacement Ralts Ghost2 is alive and made it to Victory Road...
```

On the `2` token in `Ghost2`, the best layer is 44:

```text
write-time: survived, successfully, surviving, survives, successful, retained
fresh:      replacement, replaced, aka, replacements, renamed, second
next-time:  survived, is, made, survives, has
```

This example has a caveat: the next-token distribution is already meaningful
because the local phrase says `Ghost2 is alive`. The useful part is not that
J-lens sees survival while next-token prediction sees nothing. The useful part
is the contrast between a write-time survival state and a fresh replacement
marker. The same visible label carries a more situated run-state reading when
it is seen after the original conversation.

### `Dex`: trade obligation vs Pokedex prior

Summary context:

```text
- Dex trade: spare Makuhita for Dex's Castform...
```

On the `Dex` token, the best layer is 39:

```text
write-time: partnered, partner, exchange, friend, buddy
fresh:      Pokedex-like fragments, info, Pokemon, tracker, stats
next-time:  owes, trade, owed
```

On the following `trade` token, the best layer is 55:

```text
write-time: pending, agreement, debt, promise, promised, owed, owes
fresh:      completed, status, requirement, progress, request, strategy
```

This is a good explanatory example but not a pure next-token control, because
the local text says `Dex trade`. The report should keep it, but not present it
as the strongest proof that J-lens is giving a signal distinct from
continuation. Its value is that fresh encoding pulls `Dex` toward a generic
Pokedex/DexNav/progress meaning while write-time encoding keeps the private
person/trade relation.

### Software incident labels: `Falcon` and `Patch 17`

Summary context:

```text
| Falcon | old rollback branch | rejected because it drops subscription coupons |
```

On `Falcon`, the best layer is 43:

```text
write-time: rejected, obsolete, deprecated, failed, outdated, abandoned
fresh:      Falcon, Flight, Aviation, flight, SpaceX, eagle, Aerospace
```

The software table is noisier than the natural-language summaries because
Markdown pipes and table formatting produce many high-change structural rows.
But once the label span is selected, the result is legible: write-time sees the
operational status of the branch, while fresh encoding sees the generic
falcon/bird/aviation prior.

`Patch 17` has the same shape. Around the `17`/table-separator span, the
write-time readout surfaces `obsolete`, `outdated`, `old`, `expired`, and
`legacy`, while fresh encoding mostly reads patch/version/repair semantics.
This is promising for coding-style contexts, but the presentation must be
span-based. Isolated tokens inside tables are too noisy.

### Household checklist labels: `Delta`, `Orange`, and `cooler`

Summary context:

```text
- Delta = ferry route, not airline; booked Delta 6 at 7:40...
- Orange = lockbox tag color; orange key opens kayak shed...
- Big cooler instruction is stale; bring two soft coolers...
```

These examples matter because they are ordinary rather than game-like.

`Delta` best layer 47:

```text
write-time: refers, =, represents, referring, means, refer
fresh:      Delta, airport, Airport, Alpha, airline, Sky, River
```

`Orange` best layer 50:

```text
write-time: refers, signifies, represents, symbol, =, denotes
fresh:      Orange, orange, oranges, citrus, color, Juice
```

`cooler` best layer 36:

```text
write-time: canceled, replaced, rejected, replacement, NOT, failed, refused
fresh:      freezer, fridge, Costco, BBQ, camping, backpack, cooler
```

The broad sweep therefore recovers the same pattern outside the Pokemon and
block-party examples: a compact label or ordinary word can keep a local role
in the write-time state and drift toward lexical priors in the fresh state.

## What This Adds To The Current Story

This sweep does not show that ValueGraft improves behavior. It does not test
grafted KV state at all. It adds three supporting points.

First, the earlier hand-picked examples were not isolated accidents. A broader
scan across seven summary shapes finds the same write-time/fresh split on
private labels, stale identifiers, rejected options, and operational labels.

Second, layer choice is now less arbitrary. Layer 48 is globally strong in
this run and remains much less next-token-like than layer 62. That gives us a
better default for future qualitative examples.

Third, the sweep shows what an automatic qualitative pipeline would need:
token filtering, span grouping, and next-token controls. Raw top-k divergence
is a good discovery signal, not a polished figure.

## Limits

The J-lens can only name concepts through vocabulary-token directions. If a
state is represented as a multi-token phrase, an abstract relation, or a
binding between concepts, the readout may only show fragments. This is a known
limitation of the method and matches what we see in file paths, command
strings, Markdown tables, and JSON-like summaries.

The readout is qualitative. A row where write-time says `obsolete` and fresh
says `Falcon` is a useful explanatory artifact, but behavioral evidence still
has to come from controlled task metrics.

The broad sweep ranks token-layer rows, not semantic spans. A future version
should aggregate token fragments into phrases before ranking. That would make
examples like `B-410`, `Patch 17`, full file paths, and shell commands much
clearer.

The current data is from one model and one J-lens artifact. It is enough for
internal understanding and report drafting, but not enough for a general claim
about all models.

## Recommended Next Pass

1. Build a small span-first renderer for this output. Input: a phrase or
   automatically detected span. Output: local context, write-time readout,
   fresh readout, next-token candidates, and overlap scores.
2. Use layer 48 as the default display layer, and optionally show layer 62 in a
   separate column labelled as more next-token-like.
3. Keep raw token-layer rankings as supplementary material.
4. For SWE-style examples, report tool names, paths, symbols, line ranges, and
   commands as spans. Do not report isolated punctuation or subword tokens as
   primary examples.
5. When drafting the main ValueGraft report, treat J-lens evidence as a
   qualitative mechanism probe. The behavioral results remain the main claim.

## References

- Anthropic and collaborators, "Verbalizable Representations Form a Global
  Workspace in Language Models," 2026. The paper introduces the Jacobian lens
  as a residual-stream readout that maps intermediate activations to ranked
  vocabulary-token directions, and discusses its limitations for multi-token
  concepts and late-layer next-token behavior:
  https://transformer-circuits.pub/2026/workspace/
- Neuronpedia model page for Qwen3.6-27B, which exposes the model entry used
  for the downloaded J-lens artifact:
  https://www.neuronpedia.org/qwen3.6-27b
- Qwen team, "Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model," 2026,
  for the subject model background:
  https://qwen.ai/blog?id=qwen3.6-27b

