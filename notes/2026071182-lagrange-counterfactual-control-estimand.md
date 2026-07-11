# Causal estimand after invalidation of the cyclic wrong-history control

**Author:** Lagrange (`gpt-5.6-sol-xhigh`)

**Date:** 2026-07-11

**Status:** Focused read-only methodological adjudication. This note defines a
proposed replacement control and estimand. It reports no model outcome and is not
itself a preregistration.

## Decision

The co-primary history-specificity control should be a **plant-specific minimally
counterfactual history**, not a whole-conversation unrelated donor. Each focal
plant receives one coherent source history that establishes the already frozen
counterfactual answer while preserving every non-focal fact possible. Correct and
counterfactual sources then force the same summary token IDs at the same logical
positions under the same schedule.

The experiment should retain two conceptual co-primary estimands:

1. correct history versus fresh same-text encoding, testing downstream utility
   beyond the visible summary text; and
2. correct history versus a coherent minimally counterfactual history, testing
   whether the state depends on the focal historical fact rather than generic
   source coherence.

An omission/neutral history is the most valuable additional control, but it asks a
different question and should be secondary. A coherent unrelated whole-
conversation donor is a broad stress control and cannot replace the focal
counterfactual in the co-primary claim.

## Basis for the decision

The frozen cyclic control is mechanically exact but distributionally invalid.
Across all twelve cases, most replaced slots repeat donor content, with maximum
cycle counts of 20–51; the literal worst slot repeats a seven-token acknowledgement
51 times. Therefore the old `G_correct-G_wrong` can distinguish a natural history
from corrupted repetitive text, not correct semantic history from coherent wrong
history. The byte-level evidence and claim consequence are recorded in
[`2026071179-sol-wrong-history-control-invalidity.md`](2026071179-sol-wrong-history-control-invalidity.md).

The independent lineage audit also found that the two selected plants share one
conversation, summary, donor pairing, and arm caches; the conversation is the
independent unit, not the plant. It further established that a single donor per
target cannot represent a distribution of arbitrary wrong histories
([`2026071181-carver-v10-gate-and-sample-lineage-audit.md`](2026071181-carver-v10-gate-and-sample-lineage-audit.md)).

Claude Opus 4.8's focused review endorsed a minimal counterfactual only subject to
literal decoded-history review: every fact-specific downstream reference must be
repaired, exact token-length matching must remain natural rather than padded, and
the counterfactual history must establish the same alternative used by the frozen
target. That review appears in the latest turns of
[`2026071156-sol-fable-execution-coordination.md`](2026071156-sol-fable-execution-coordination.md).

## Replacement source construction

For conversation `i` and focal plant
`p in {referent, sense}`, define `W_MC(i,p)` as follows:

1. Start from the complete correct source conversation.
2. Change the focal establishing fact so that the history establishes the exact
   `counterfactual` answer already frozen for plant `p` in
   `data/coherent_state_targets.json`.
3. Change every fact-specific downstream reference required to make that
   alternative globally coherent. A minimal causal intervention may therefore
   touch more than one token span; “minimal” means minimal semantic change, not
   necessarily one lexical substitution.
4. Preserve the other selected plant and every unrelated planted fact exactly.
5. Preserve all chat-structural tokens, roles, message count, message order, and
   summary request/header.
6. Preserve the exact token count of every message block, not only total prefix
   length. This keeps the turn-aligned call widths, summary start, logical
   positions, and physical cache positions matched.
7. Force the identical saved summary token IDs used by the correct source,
   stepwise at identical positions.

Each literal decoded `W_MC(i,p)` must be committed and accepted before any outcome
is observed. Acceptance requires both:

- blind review finds it to be a natural, internally coherent conversation that
  genuinely establishes the specified alternative, with no dangling or
  contradictory reference; and
- tokenizer reconstruction proves exact per-message length, structural-token,
  position, and schedule identity to the correct source.

Conspicuous padding, repeated filler, semantically empty token stuffing, or a
change to the non-focal selected plant fails the control. If a plant cannot meet
both coherence and exactness, it must be redesigned before outcomes. If it still
cannot qualify, apply a prospectively frozen case replacement/exclusion rule; do
not silently score only the remaining plant, because that changes the
conversation-level estimand.

## Exact estimands

Let:

- `i = 1,...,N` index conversations;
- `p` index the two selected plants, referent and sense;
- `s` index the prospectively fixed source schedule, such as turn-aligned replay
  `P` or ordinary chunks `O`;
- `Y_i,p(A,s)` be the correct-target minus counterfactual-target mean token-log-
  probability margin after arm `A` under schedule `s`.

The fresh destination has one fixed valid construction. Write `Y_i,p(F)` for it.
For each conversation and schedule define:

```text
GF_i,s = (1/2) * sum_p [Y_i,p(C,s) - Y_i,p(F)]

GMC_i,s = (1/2) * sum_p [Y_i,p(C,s) - Y_i,p(W_MC(i,p),s)]
```

`GF` is the deployment-relevant utility contrast: does correct-history-conditioned
summary state improve the fixed downstream margin beyond fresh encoding of the
same summary tokens?

`GMC` is the focal history-specificity contrast: does changing only the focal
historical fact to a coherent specified alternative change the downstream margin
in the predicted direction while summary text, positions, and schedule remain
fixed?

The old generic label `GW` should be retired for confirmatory prose. `GMC` names
the actual intervention and prevents a broad “wrong-history distribution” claim.

If dual-schedule robustness remains part of the design, the confirmatory
intersection requires positive conversation-clustered lower confidence bounds for
both `mean_i(GF_i,s)` and `mean_i(GMC_i,s)` under **each** prospectively fixed
schedule. The schedule difference-in-differences must be reported, but schedules
are repeated conditions and do not increase sample size.

Useful secondary decompositions include:

```text
MCF_i,s = (1/2) * sum_p [Y_i,p(W_MC(i,p),s) - Y_i,p(F)]
```

A negative `MCF` supports the interpretation that counterfactual-conditioned state
tracks the alternative answer rather than merely suffering generic damage. It is
diagnostic, not an additional co-primary endpoint.

## Inferential unit and aggregation

The independent unit is the **conversation**.

For every schedule and control:

1. compute correct and counterfactual target margins within each plant;
2. compute each paired arm contrast within plant;
3. if multiple variants of one control exist, average those variants within the
   plant first;
4. average referent and sense contrasts equally within the conversation;
5. form t/bootstrap intervals over the `N` conversation-level contrasts.

At the planned endpoint, `N=12`, not 24 plants, not 24 counterfactual histories,
and not a multiple of the number of schedules or controls. Plant/category estimates
are descriptive unless a separately powered analysis is preregistered.

## Adjudication of alternative controls

### Plant-specific minimally counterfactual history

**Role:** co-primary specificity control.

This is the narrowest causal manipulation aligned with the target margin. It
changes the historical answer of interest while holding the surrounding natural
conversation as constant as coherence permits. One frozen counterfactual per plant
is sufficient to define the fixed-benchmark estimand because each plant already has
one frozen target alternative. It does not establish robustness across all possible
wrong answers or histories.

Multiple minimally counterfactual variants are required only if the intended claim
is explicitly about robustness across alternatives. Such variants remain repeated
measurements: average them within plant, and do not count them as additional `N`.

### Omission/neutral history

**Role:** preferred secondary causal control.

Define `W_omit(i,p)` by removing the focal fact without asserting its opposite,
repairing downstream references, and satisfying the same decoded-coherence and
per-message-length requirements. Then `C-W_omit` asks whether correct state carries
the presence of the fact, whereas `C-W_MC` asks whether state distinguishes correct
from specifically alternative content.

Omission is especially useful if a counterfactual result could be explained by
history-summary contradiction, surprisal, or an actively opposing state. Requiring
both omission and counterfactual contrasts as co-primary would enlarge an already
demanding intersection without being necessary for the core claim. Report omission
as a prespecified secondary decomposition.

### Whole-conversation coherent unrelated donor

**Role:** optional broad stress control only.

Even a natural, exact turn/length-matched unrelated source changes topic, style,
nearly every latent fact, history-summary congruence, and relation to the retained
tail. A positive `C-U` therefore cannot isolate the focal fact. One unrelated donor
is also an arbitrary fixed stimulus, not a sample from the distribution of wrong
histories.

If a broader robustness statement is desired, use at least two independently
authored coherent donors per conversation and average them within conversation.
Otherwise a single donor may be reported only as a fixed-case diagnostic. It must
never replace `W_MC` in the co-primary estimand.

## Claim boundary

If the co-primary intersection clears, the supported claim is:

> On this fixed benchmark, checkpoint, backend, position-preserving layout, and
> prospectively fixed schedule(s), correct-history-conditioned summary K/V both
> improved the downstream correct-versus-counterfactual margin over fresh encoding
> of identical summary text and differed in the predicted direction from summary
> K/V conditioned on one coherent, minimally changed history establishing the
> specified alternative.

It would not establish:

- an average effect over arbitrary wrong histories or counterfactual alternatives;
- ecological validity for real agent transcripts or compaction systems;
- that an unrelated-history donor is harmless or representative;
- that plants, schedules, donors, or control variants are independent replicates;
- a universal latent-memory mechanism beyond the exact summary-state intervention.

This intersection separates downstream utility (`GF`) from focal semantic
specificity (`GMC`) without asking a globally corrupted or globally unrelated
history to carry a causal claim it cannot identify.
