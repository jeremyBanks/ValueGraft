# Nomenclature Reframing Note

This note records a suggested naming cleanup for the paper/write-up. It is not
a request to rename code, result directories, or historical logs. Those names
should remain available for provenance. The goal is to make the public-facing
story sound like one coherent research program rather than a collection of
unrelated arm names.

During active experimentation, the safest policy is to treat these as
**paper-facing aliases**. Internal identifiers such as `E`, `H-pack`,
`B-min-pack`, and result directory names should probably stay unchanged until
the experiment is no longer in flight. Renaming code paths, arm labels, or
output directories mid-run is an unnecessary source of provenance mistakes.

## Core Suggestion

Use **ValueGraft** as the umbrella name for the overall approach:

> preserving or reusing write-time KV state across a compaction boundary so
> that compacted text is not interpreted only from a clean, post-hoc context.

Under this framing, the different experimental arms are variants of
ValueGraft, not separate inventions with unrelated names.

## Suggested Crosswalk

| Current / historical name | Suggested paper-facing name | Role |
|---|---|---|
| ValueGraft / E / E-tuned | ValueGraft Blend | Fresh compacted context plus blended write-time value states. |
| E:a1.0 | ValueGraft Replace | Use write-time values directly rather than a partial blend. |
| E:cfg=layers | Layer-Steered ValueGraft | Per-layer coefficient map; current champion candidate, pending clean validation. |
| E:cfg=posslots | Slot-Masked ValueGraft | Exploratory variant; failed contamination guard, so not a main finding. |
| H-pack | ValueGraft Pack | Packed summary cache entries written while the full context was available. |
| H-gap | ValueGraft Gap | Diagnostic, non-deployable gapped version of the packed-state idea. |
| B-min-pack | Fresh Pack Control | Same packed layout/text, freshly encoded; control for ValueGraft Pack. |
| B | Plain Summary Compaction | Production-like text-only summary compaction baseline. |
| A | Full-Context Oracle | No compaction baseline. |

## Why Change It

`H-pack` and `ValueGraft` sound like fundamentally different ideas even though
they are closely related: both ask whether write-time state preserves useful
semantic conditioning that fresh re-encoding loses. The old names mostly came
from arm labels and implementation details. They are useful internally, but
they make the paper harder to read.

The word `pack` is used because this variant takes state that was originally
written at one set of positions and places it into the compacted, contiguous
post-summary layout. In other words, the summary's cached state is "packed"
next to the remaining context instead of being left at its old gapped
positions. That is a layout distinction, not the core scientific claim.

This may still be confusing as a public name. If `ValueGraft Pack` sounds too
mechanical, alternatives include **ValueGraft Preserve**, **ValueGraft Summary
State**, or **Packed-State ValueGraft**. The important contrast is:

- the control freshly re-encodes the same summary text in the compact layout;
- the intervention keeps the summary's write-time state and adapts it to that
  compact layout.

## Proposed Writing Pattern

Introduce the family first:

> We call this family of interventions **ValueGraft**: training-free methods
> that carry write-time KV state across a compaction boundary.

Then introduce the two main branches:

> **ValueGraft Blend** modifies the compacted context's value vectors by
> interpolating or extrapolating from write-time values. **ValueGraft Pack**
> instead preserves the summary's write-time K/V entries directly in a packed
> cache layout.

Then discuss controls:

> **Fresh Pack Control** uses the same packed text and positions as
> ValueGraft Pack, but obtains its K/V state by ordinary fresh re-encoding.

This keeps the story unified while still making the methodological distinctions
clear.

## Caution

Do not let the naming imply that all variants succeeded. In particular,
Slot-Masked ValueGraft should be described as an exploratory variant that
failed its wrong-conversation contamination guard. That failure is useful
evidence about the tuning surface, but it should not be folded into the main
positive claim.
