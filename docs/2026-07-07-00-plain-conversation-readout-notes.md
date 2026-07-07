# Plain Conversation J-lens Readout Notes

Status: qualitative illustration only. This is a companion to the Pokemon demo,
using an ordinary planning conversation rather than a game-domain memory trap.

## What Was Run

The probe in `plain_conversation_probe.py` uses a Riverside block-party planning
thread. The old context establishes several private meanings:

- `Maple` is the library's Maple Room, not trees or syrup.
- `Robin` is the human volunteer coordinator, not a bird or story character.
- `Blue` is the chosen rain plan.
- `B-410` is a stale permit number; `P-771` is current.
- `Orchid` is the dessert vendor, not flowers.
- the `big oak table` was rejected.
- `Friday` is an insurance deadline, not the party day.
- `Crane` is the stage rental company, not equipment.

The useful comparison is again the matched-wrapper state in
`outputs/qwen36_plain_probe.json`:

- `write_time_matched_summary_anchors`: summary after the full old
  conversation.
- `fresh_matched_summary_anchors`: the same literal summary in the same local
  wrapper, but without the old conversation.

The sampled model/lens setup is Qwen3.6-27B with the Neuronpedia Jacobian lens
at layers 16, 32, 48, and 62.

## Main Examples

### Maple

Canonical summary, matched wrapper:

| State | Layer 48 top readouts | Layer 62 top readouts |
| --- | --- | --- |
| write-time | `refers`, `=`, `referring`, `denotes` | `=`, `refers`, `is`, `means` |
| fresh | `Street`, `street`, `neighborhood`, `park`, `town`, `City` | `Street`, `Ave`, `Avenue`, `St`, `Streets` |

This is a strong ordinary-language analogue of the Pokemon `Vacuum` effect.
With old context, `Maple` is interpreted as a local label whose meaning has
been defined. Freshly encoded, it drifts toward a generic place/street name.

### Robin

Canonical summary, matched wrapper:

| State | Layer 48 top readouts | Layer 62 top readouts |
| --- | --- | --- |
| write-time | `is`, `aka`, parenthetical/person-ish continuations | `is`, `volunteers`, `volunteer`, `person` |
| fresh | `Robin`, `Mary`, `Emily`, `Rose`, `Green`, `volunteer`, `Oak` | `Hood`, `Robin`, `Oak`, `handles`, `manages` |

The old-context state points toward the human volunteer role. Fresh encoding
partly keeps the name/person framing, but also drifts into generic `Robin Hood`
and name-list associations.

### Blue

Canonical summary, matched wrapper:

| State | Layer 48 top readouts | Layer 62 top readouts |
| --- | --- | --- |
| write-time | `approved`, `option`, local decision punctuation | `chosen`, `chose`, `selected` |
| fresh | `Blue`, `tents`, `Rain`, `blue`, `Flag` | `tarp`, `tents`, `Tent`, `tent`, `Zone` |

Write-time `Blue` carries the chosen-plan decision. Fresh `Blue` moves toward
rain-event objects and color-plan associations. It is still in the event
domain, but the decision status is weaker.

### Green

Canonical summary, matched wrapper:

| State | Layer 48 top readouts | Layer 62 top readouts |
| --- | --- | --- |
| write-time | `rejected`, `failed`, `refused`, `unacceptable`, `because` | `rejected`, `tents`, `meant`, `means` |
| fresh | `Green`, `green`, `cancelled`, `option`, `would` | `means`, `canc...`, `would`, `was` |

This is less dramatic than `Blue`, but still useful: the old-context state
emphasizes rejection, while the fresh state is closer to generic option
semantics.

### B-410

Canonical summary, matched wrapper:

| State | Layer 48 top readouts | Layer 62 top readouts |
| --- | --- | --- |
| write-time | `obsolete`, `outdated`, `deprecated`, `expired` | permit-code continuation fragments |
| fresh | `municipal`, `City`, `city`, `Civic`, `Town` | code/number continuations |

The stale-number fact is visible in the write-time state. Fresh encoding knows
it is a civic/permit-ish code, but the stale/obsolete meaning is much weaker.

### P-771

Canonical summary, matched wrapper:

| State | Layer 48 top readouts | Layer 62 top readouts |
| --- | --- | --- |
| write-time | `new`, `newly`, `new`-like continuations | code continuation fragments |
| fresh | `ID`, `PX`, `PN`, code-like fragments | code continuation fragments |

The current-number side is subtler than the stale-number side. Write-time
leans toward new/current; fresh mostly sees an identifier.

### Orchid

Canonical summary, matched wrapper:

| State | Layer 48 top readouts | Layer 62 top readouts |
| --- | --- | --- |
| write-time | `refers`, `represents`, company/customer-ish meanings | `is`, `refers`, `means` |
| fresh | `Orchestra`, `Music`, `Dance`, `music`, `orchestr...` | name/wordpiece continuations |

This one is delightfully weird. Because the tokenization starts with `Orch`,
fresh encoding drifts into orchestra/music associations rather than the dessert
vendor. Old context pulls it toward a defined local referent.

### Big Oak Table

Canonical summary, matched wrapper:

| State | Layer 62 top readouts |
| --- | --- |
| write-time | `oak`, `Oak`, `oval`, `owl`, `ark` |
| fresh | `screen`, `cooler`, `tent`, `tarp`, `banner`, `oak` |

This is noisier, but the probe-question state was clearer: when asked
`Should we use the big table?`, both full and compacted contexts pull toward
`oak`, and the table token has `rejected` in the top readouts. It may be less
useful as a summary-anchor example than as a probe-token example.

### Friday

Canonical summary, matched wrapper:

| State | Layer 48 top readouts | Layer 62 top readouts |
| --- | --- | --- |
| write-time | `solely`, `ONLY`, `only`, `weekday`, `Friday` | `is`, `deadline`, `separate` |
| fresh | `Friday`, `Monday`, `Tuesday`, `weekend`, `Thursday`, `Saturday` | `timeline`, `night`, `schedule`, `deadline` |

Write-time carries the "Friday is only the deadline, keep it separate" warning.
Fresh encoding mostly sees an ordinary day/schedule token, with some deadline
signal still present from local text.

### Crane

Canonical summary, matched wrapper:

| State | Layer 48 top readouts | Layer 62 top readouts |
| --- | --- | --- |
| write-time | `is`, `refers`, `represents`, `hired` | `is`, `refers`, `means`, `Stage`, `stage`, `delivers` |
| fresh | `crane`, `Crane`, `trucks`, `contractor`, `tower`, `hire` | `rental`, `operator`, `schedule`, `lease`, `license` |

The fresh state is not wrong exactly, but it is more generic: rental/equipment
operator semantics. Write-time better preserves the local company/stage-rental
referent.

## Interpretation

This demo is less dramatic than Pokemon and that is a feature. The same pattern
appears in ordinary planning text:

- private local labels (`Maple`, `Orchid`, `Crane`) drift toward generic named
  entities or common senses when freshly encoded;
- old-context state more often carries local definition words such as
  `refers`, `means`, `is`, `chosen`, `rejected`, `deadline`, or `obsolete`;
- locally explicit phrases like `P-771` remain mostly code-like in both states,
  but old-context state better reflects whether the code is current or stale.

The qualitative story is now broader:

- Pokemon shows vivid referent/sense traps with memorable private meanings.
- The block-party probe shows a mundane summarization setting with the same
  kind of context-conditioned semantic drift.

Neither is evidence by itself. Together, they are useful explanatory artifacts:
they make the hypothesized mechanism easier to see before showing quantitative
results.

## Possible Next Variants

Good next demos would add one of:

- a work meeting thread where names overlap with tools, branches, or rooms;
- a family logistics thread with stale dates and reused nicknames;
- a small support-ticket thread with a customer name that is also a product
  name;
- an automatic-anchor scan that ranks tokens by write-time vs fresh top-k
  divergence instead of using hand-selected anchors.
