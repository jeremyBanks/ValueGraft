# Pokemon J-lens Readout Notes

Status: qualitative illustration only. This file describes a demo probe, not
experiment evidence and not a statistical result.

## What Was Run

The probe uses the canonical Pokemon Emerald conversation from
`demos/pokemon-demo-conversation.md`: Soup is the Mudkip starter, Vacuum is a
Zigzagoon/Pickup utility slot, Ghost is the original dead Ralts, Ghost2 is the
replacement Ralts that survived, Dex owes a Makuhita-for-Castform trade, and
Hariyama was explicitly ruled out.

The script `pokemon_probe.py` samples Qwen3.6-27B with the Neuronpedia Jacobian
lens at layers 16, 32, 48, and 62. The most useful output is the matched-wrapper
comparison in `outputs/qwen36_pokemon_probe_v2.json`:

- `write_time_matched_summary_anchors`: the summary is placed after the full old
  conversation, in a standard summary-request/assistant-summary wrapper.
- `fresh_matched_summary_anchors`: the same literal summary is placed in the
  same wrapper, but without the old conversation.

That matched wrapper matters. It reduces a formatting confound: the local text
around the summary is held constant, while old-context conditioning is present
or absent.

## Main Examples

### Ghost

The summary contains the token `Ghost` in text explaining that the original
Ralts died, while `Ghost2` survived.

Canonical summary, matched wrapper:

| State      | Layer 48 top readouts                                       | Layer 62 top readouts                            |
| ---------- | ----------------------------------------------------------- | ------------------------------------------------ |
| write-time | `died`, `destroyed`, `lost`, `survived`, `killed`, `failed` | `died`, `faint`, `is`, `faded`, `failed`, `went` |
| fresh      | `Ghost`, `ghost`, `/G`, `Magic`, `ghosts`                   | `/G`, `-type`, `/S`, `/P`, `/F`                  |

This is the cleanest qualitative example so far. With old-context conditioning,
the token is read as the private run-state referent: the Ralts that
died/fainted. Freshly encoded, it drifts toward generic Pokemon
Ghost-type/string semantics.

Lean summary, matched wrapper:

| State      | Layer 48 top readouts                                   | Layer 62 top readouts                |
| ---------- | ------------------------------------------------------- | ------------------------------------ |
| write-time | `survived`, `survives`, `succeeded`, `died`, `replaced` | `2`, `3`, `1`, `4`, `survived`       |
| fresh      | `Ghost`, `ghosts`, `ghost`, `spooky`, `Ghost`           | `-type`, `/G`, `type`, `types`, `/P` |

The lean wording sharpens a slightly different part of the distinction: the
write-time state strongly anticipates the `2` in `Ghost2` and survival contrast,
while the fresh state again looks like generic Ghost-type semantics.

### Ghost2

Canonical summary, matched wrapper:

| State      | Layer 48 top readouts                                          | Layer 62 top readouts                   |
| ---------- | -------------------------------------------------------------- | --------------------------------------- |
| write-time | `survived`, `successfully`, `survives`, `successful`, `safely` | `2`, `survived`, `3`, `1`, `survives`   |
| fresh      | `was`, `became`, `now`, `later`, `second`                      | `is`, `was`, `has`, `joined`, `arrived` |

The write-time state carries the alive-vs-dead contrast. Fresh encoding still
recognizes a named entity and replacement-ish context, but the survival
semantics are much weaker.

### Vacuum

The summary says Vacuum is the Zigzagoon/Pickup utility slot and not a fighter.

Canonical summary, matched wrapper:

| State      | Layer 48 top readouts                      | Layer 62 top readouts                                |
| ---------- | ------------------------------------------ | ---------------------------------------------------- |
| write-time | `aka`, `was`, `is`                         | `is`, `nickname`, `nick`, `Zig`, `nicknamed`, `/Z`   |
| fresh      | `Vacuum`, `vacuum`, `Vapor`, `Air`, `Dust` | `Cleaner`, `cleaner`, `Clean`, `cleaned`, `cleaners` |

This one is especially interpretable. The same literal token has
Zigzagoon/nickname semantics when written after the old conversation, but
appliance/cleaning semantics when freshly encoded.

Lean summary, matched wrapper:

| State      | Layer 48 top readouts                            | Layer 62 top readouts                                |
| ---------- | ------------------------------------------------ | ---------------------------------------------------- |
| write-time | `was`, `remains`, `is`, `aka`, `stayed`          | `is`, `Zig`, `/Z`, `(Z`, `zig`, `handles`            |
| fresh      | `vacuum`, `Vacuum`, `Air`, `suction`, `cleaning` | `cleaner`, `Cleaner`, `cleaners`, `Clean`, `cleaned` |

The lean variant repeats the same pattern even more plainly.

### Dex

The summary says Dex owes a Makuhita-for-Castform trade.

Canonical summary, matched wrapper:

| State      | Layer 48 top readouts                                      | Layer 62 top readouts                                    |
| ---------- | ---------------------------------------------------------- | -------------------------------------------------------- |
| write-time | `promised`, `partnered`, `agreed`, `promise`, `exchange`   | `owes`, `owed`, `owe`, `trade`, `traded`, `trades`       |
| fresh      | `completion`, `tracker`, `stats`, `completed`, `Collector` | `Nav`, `nav`, `completion`, `dex`, `navigation`, `entry` |

Again, old-context conditioning reads Dex as a person involved in an owed trade.
Fresh encoding drifts toward Pokedex/DexNav/progress-tracker meanings.

Lean summary, matched wrapper:

| State      | Layer 48 top readouts                                    | Layer 62 top readouts                                              |
| ---------- | -------------------------------------------------------- | ------------------------------------------------------------------ |
| write-time | `promised`, `promise`, `swapped`, `pending`, `agreed`    | `owes`, `owed`, `trade`, `traded`, `owe`, `trades`                 |
| fresh      | `completion`, `stats`, `numbers`, `completed`, `tracker` | `entry`, `completion`, `completed`, `count`, `entries`, `progress` |

The lean variant makes the person/trade vs index/progress split even clearer.

### Hariyama

Hariyama was explicitly rejected in favor of Breloom.

Canonical summary, matched wrapper:

| State      | Layer 48 top readouts                                       | Layer 62 top readouts                                   |
| ---------- | ----------------------------------------------------------- | ------------------------------------------------------- |
| write-time | `rejected`, `forbidden`, `banned`, `prohibited`-like tokens | name-continuation fragments such as `y`, `ya`, `Yam`    |
| fresh      | `backup`, `alternative`, `replacement`-like tokens          | name-continuation fragments such as `ama`, `ya`, `Hari` |

The lower-level name-continuation behavior is noisy, but layer 48 is useful:
write-time emphasizes rejection/prohibition, while fresh is closer to generic
candidate/alternative semantics.

### Ultra Balls

Ultra Balls is a useful sanity check because the phrase is locally explicit.

Canonical summary, matched wrapper:

| State      | Layer 62 top readouts             |
| ---------- | --------------------------------- |
| write-time | `Balls`, `Ball`, `balls`, `ball`  |
| fresh      | `Balls`, `balls`, `Ball`, `balls` |

This is not a dramatic private-context recovery example, and that is good. It
shows that the probe is not merely producing arbitrary differences everywhere.
Some tokens are locally pinned strongly enough that fresh encoding preserves the
important reading.

## Interpretation

The examples are not proof, but they are unusually legible. They show the same
literal summary tokens acquiring different lens-visible neighborhoods depending
on whether they were written after the old conversation or freshly encoded
without it.

The strongest cases are exactly the tokens whose meaning is private to the
conversation:

- `Ghost`: dead original Ralts vs generic Ghost-type semantics.
- `Ghost2`: surviving replacement vs weaker generic named-entity semantics.
- `Vacuum`: Zigzagoon/Pickup nickname vs appliance/cleaner semantics.
- `Dex`: friend who owes a trade vs Pokedex/DexNav/progress semantics.
- `Hariyama`: explicitly rejected option vs generic candidate/alternative.

This is a good qualitative companion to the quantitative experiments because it
does not argue that compaction loses information. Instead, it helps show what
kind of context-conditioned state the write-time representation appears to be
carrying, and why grafting that state might plausibly help a compacted model
continue with the intended private meanings.

## Next Demo To Build

The Pokemon demo is intentionally dense and memorable. A useful follow-up would
be a plainer conversation-summarization demo at roughly 3k-4k input tokens and
800-1k summary tokens. It should use ordinary ambiguous anchors, such as:

- a person/project name that is also a common noun;
- an explicitly rejected option;
- a stale numeric fact;
- a deadline or date with multiple possible referents;
- a tool or place name with a generic alternate sense.

That would make the same phenomenon easier to explain to readers who find the
Pokemon example too specialized.
