# J-Lens Readouts vs Next-Token Probabilities

Status: qualitative control report\
Date: 2026-07-07\
Raw data: `outputs/qwen36_next_token_readout_comparison.json`

## Question

The J-lens readouts in the ValueGraft notes look like ranked vocabulary
distributions. That raises a necessary control question: are they mostly just
the model's ordinary next-token probabilities at the same position?

The answer from this small control is no, but with an important qualification.
Late-layer J-lens readouts can resemble next-token logits, especially when the
local summary text strongly determines the next word. Mid-layer readouts often
show a different signal: they expose semantic neighborhoods that are not simply
the literal continuation distribution.

This matters for the qualitative ValueGraft story. The J-lens examples are not
evidence that grafting improves behavior by themselves. They are a way to
inspect whether the same visible summary token has different internal semantics
under write-time context versus fresh re-encoding. The next-token control helps
keep that claim honest.

## What Was Compared

I selected five summary-anchor positions from the existing Pokemon and plain
conversation demos:

- `Vacuum` in the Pokemon summary
- `Dex` in the Pokemon summary
- `Maple` in the block-party summary
- `B` in `B-410` in the block-party summary
- `Crane` in the block-party summary

For each anchor, the script compared two matched states:

- **write-time:** the summary appears after the full old conversation;
- **fresh:** the same literal summary appears in the same local wrapper, but
  without the old conversation.

At each anchor position, it recorded:

- J-lens top-20 vocabulary readouts from Qwen3.6-27B at layers 48 and 62;
- the model's actual top-20 next-token logits at the same anchor position;
- top-20 overlap between the J-lens readout and next-token logits;
- simple term hits for private-context meanings and generic lexical meanings.

This is intentionally not a behavioral benchmark. It is a control on the
interpretability probe.

## Aggregate Result

Across 5 anchors, 2 states, and 2 sampled layers, the mean top-20 Jaccard
overlap between the J-lens readout and next-token logits was `0.256`.

Layer matters:

| Layer | Mean top-20 Jaccard |
| ----- | ------------------: |
| 48    |               0.076 |
| 62    |               0.436 |

The layer-62 result is unsurprising: very late residual-stream readouts are
closer to the model's final continuation distribution. The layer-48 result is
the more useful control. At that depth, the top vocabulary readout often has
almost no overlap with the ordinary next-token distribution, yet still exposes
the private or lexical meaning split seen in the earlier examples.

The write-time and fresh states had similar mean overlap:

| State      | Mean top-20 Jaccard |
| ---------- | ------------------: |
| write-time |               0.269 |
| fresh      |               0.243 |

So the main distinction is not that write-time states are globally more or less
next-token-like. The distinction is where the readout signal comes from. Some
anchors are continuation-dominated; others expose a semantic neighborhood that
ordinary next-token probabilities mostly miss.

Layer 48 is where the difference is most visible:

| Anchor and state    | Actual next token | Next-token top candidates      | Layer-48 J-lens top candidates                  | Jaccard |
| ------------------- | ----------------- | ------------------------------ | ----------------------------------------------- | ------: |
| `Dex`, write-time   | `trade`           | `owes`, `trade`, `owed`        | `promised`, `partnered`, `agreed`, `exchange`   |   0.053 |
| `Dex`, fresh        | `trade`           | `Nav`, `completion`, `entry`   | `completion`, `tracker`, `stats`, `Collector`   |   0.026 |
| `Maple`, write-time | `means`           | `=`, `refers`, `means`, `Room` | `refers`, `=`, `denotes`, `means`               |   0.111 |
| `Maple`, fresh      | `means`           | `Street`, `Ave`, `street`      | `Street`, `street`, `park`, `City`              |   0.053 |
| `B-410`, write-time | `-`               | `-`, `4`, `-st`                | `obsolete`, `outdated`, `deprecated`, `expired` |   0.000 |
| `B-410`, fresh      | `-`               | `-`, `2`, `PD`, `1`            | `municipal`, `City`, `Civic`, `permit`          |   0.000 |
| `Crane`, write-time | `is`              | `is`, `means`, `refers`        | `refers`, `is`, `represents`, `hired`           |   0.111 |
| `Crane`, fresh      | `is`              | `is`, `operator`, `rental`     | `crane`, `trucks`, `contractor`, `tower`        |   0.053 |

The table is abbreviated from the top-20 rows; the raw JSON contains the full
ranked lists.

## Examples

### `B-410`: Strong Additional Signal

Context:

```text
Permit: B-410 is stale. Use P-
```

At the anchor token `B`, the literal next token is just `-`. Ordinary next-token
logits mostly predict punctuation and code fragments:

```text
next-token top: -, <|im_end|>, 4, -, <nonbreaking hyphen>, -st, -, ...
```

Layer-48 J-lens under write-time context instead reads out the status of the
permit:

```text
write-time J-lens L48: obsolete, outdated, deprecated, obsolete, expired,
deprecated, ...
```

The top-20 overlap at layer 48 is exactly `0`. This is the clearest control
case. The J-lens readout is not merely reporting what token comes after `B`; it
is exposing a semantic feature of the current referent.

Fresh encoding flips to the generic civic-code reading:

```text
fresh J-lens L48: municipal, City, Municipal, city, Local, Civic, Town, License
```

This matches the original qualitative point: the same visible summary text keeps
the phrase `B-410 is stale`, but the write-time state more directly contains the
stale/obsolete interpretation.

### `Maple`: Local Continuation Helps, But Does Not Explain Everything

Context:

```text
Riverside block party state:
- Maple means the library's Maple Room for storage and volunteer...
```

Here the actual next token after `Maple` is `means`, so ordinary next-token
logits already contain local-definition words:

```text
write-time next-token top: :, =, is, refers, means, (, Room, =, denotes, library
```

That means this is not a pure win for J-lens. The next-token distribution is
informative because the local summary text itself says `Maple means...`.

Even so, layer-48 J-lens is more separated from next-token prediction than the
surface result suggests:

```text
write-time J-lens L48: refers, :, <Chinese "means">, =, <Chinese "means">,
referring, refer, denotes
fresh J-lens L48: Street, street, neighborhood, park, streets, City, town, Town
```

Layer-48 top-20 Jaccard is `0.111` in the write-time state and `0.053` in the
fresh state. The useful signal is the contrast between local-definition
semantics and generic place-name semantics, not the fact that the next word is
easy to predict.

### `Dex`: Next-Token Prediction Can Carry the Same Local Cue

Context:

```text
Dex trade: spare Makuhita for Dex's Cast...
```

At this anchor, the next literal token is `trade`. As a result, next-token
logits are themselves semantically meaningful:

```text
write-time next-token top: owes, ,, is, trade, (, owed, still, the, ...
```

The write-time J-lens readout is also the trade-obligation reading:

```text
write-time J-lens L48: promised, partnered, partner, partnership, agreed,
promise, planned, exchange
write-time J-lens L62: owes, owed, owe, trade, traded, trades, ...
```

This example is useful but not decisive for distinctness. Because the local text
is `Dex trade`, ordinary next-token probabilities already see much of the right
cue. The stronger point is the state contrast:

```text
fresh J-lens L48: completion, tracker, stats, Completion, bonus, completed
fresh J-lens L62: Nav, nav, completion, Nav, nav, dex, navigation, entry
```

Fresh encoding treats `Dex` more like Pokedex/DexNav progress. That state
contrast remains valuable, but the next-token control warns us not to present
this example as if next-token prediction had no access to the relevant meaning.

### `Crane`: Mixed Case

Context:

```text
Crane is the stage rental company, not equipment. Crane...
```

The next token after the first `Crane` is `is`, so next-token logits are mostly
local grammar plus some definition words:

```text
write-time next-token top: is, means, (, refers, the, ,, :, <|im_end|>, ...
fresh next-token top: is, was, operator, :, rental, means, <|im_end|>, 's
```

Layer-48 J-lens separates the two readings more visibly:

```text
write-time J-lens L48: <Chinese "is a">, refers, is, represents, hired,
company
fresh J-lens L48: crane, Crane, trucks, contractor, truck, tower, hire, Tower
```

Layer 62 partially collapses back toward next-token behavior:

```text
write-time J-lens L62: is, refers, means, Stage, stage, delivers, are
fresh J-lens L62: rental, is, operator, schedule, lease, needs, license
```

This supports a more careful presentation. The J-lens signal is strongest in
layers and examples where the readout is not reconstructing the local next-token
distribution.

### `Vacuum`: Mostly Continuation-Dominated In This Control

Context:

```text
Vacuum never battles gyms or major fights...
```

This sampled occurrence is less useful than the earlier `Vacuum` example. The
local text says `Vacuum never...`, and both next-token logits and J-lens
readouts are dominated by that local continuation:

```text
write-time next-token top: never, does, is, doesn, fights, should, must, will
write-time J-lens L62: never, does, fights, battles, is, doesn, handles, must
```

Layer-62 top-20 Jaccard is `0.667`, the highest overlap in the run. This is a
good negative warning for the writeup: not every readable J-lens row is evidence
of a distinct semantic readout. Some rows mostly mirror local continuation
pressure.

## Interpretation

The next-token control changes how we should present the J-lens work. The right
claim is:

```text
J-lens readouts can expose semantic neighborhoods at a token position that are
not reducible to the ordinary next-token distribution, especially at
intermediate layers.
```

That phrasing leaves room for the cases where the two signals are related.
Late-layer readouts and anchors with predictable next words can look much like
ordinary next-token probabilities.

For ValueGraft, this is exactly the useful interpretability role. The readouts
let us inspect a summary token as a state-bearing object rather than only as a
prompt prefix for the next token. In the best examples, the write-time state
surfaces private-context features such as `obsolete/outdated` for `B-410`, while
the next-token distribution is busy predicting punctuation or local syntax.

The control also tells us how to avoid overclaiming:

- Always show the local text around the anchor.
- Report the actual next token and next-token top candidates beside the J-lens
  candidates.
- Prefer layer-48 or similarly intermediate readouts when the question is
  semantic state, while treating layer-62 as closer to continuation behavior.
- Do not use examples like this sampled `Vacuum` row as core evidence for
  distinctness, even if they look readable.
- Treat examples like `Dex` and `Maple` as mixed: next-token prediction already
  carries some local cue, but the write-time/fresh J-lens contrast can still
  illuminate the semantic shift.

## Bottom Line

This control makes the J-lens application more precise. It shows that some
attractive examples are partly or mostly next-token effects, and it identifies
cases where the J-lens readout contains additional semantic signal. The best
current example is `B-410`: next-token prediction says hyphen/code continuation;
write-time layer-48 J-lens says obsolete/outdated/deprecated.

For the report, the clean presentation should include a side-by-side table:
context snippet, actual next token, next-token top candidates, write-time J-lens
readout, fresh J-lens readout, and top-k overlap. That will make the
interpretability evidence understandable to a reader without asking them to
trust a few cherry-picked vocabulary words.
