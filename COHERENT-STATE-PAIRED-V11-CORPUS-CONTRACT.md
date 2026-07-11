# Coherent-state paired v11 — corpus authoring contract

**Status:** authoring contract frozen before any v11 model summary, cache, target
score, or semantic outcome. Draft histories produced under this contract remain
non-executable until the mechanical and decoded-review gates below pass.

**Author:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11

## 1. Purpose and separation from v10

V11 is a new additive experiment. It does not amend, waive, or reinterpret the
failed `coherent-state-gapped-v10` apparatus or its release gates. V10's cyclic
wrong-history arm and coarse schedule are forbidden inputs to v11.

The corpus must support a causal comparison between:

- one coherent correct conversation;
- a minimally counterfactual conversation changing only one frozen referent
  decision; and
- a minimally counterfactual conversation changing only one frozen overloaded
  sense.

The conversation body is an openly authored experimental input. It is not
called subject-native or organic. The future subject model must freely generate
the tested summary under the correct history; no corpus author sees that summary
or any model outcome.

## 2. Frozen corpus size and independence

- Exactly twelve conversations, `v11-c01` through `v11-c12`.
- The independent inferential unit is the conversation (`N=12`).
- Each conversation contains exactly two selected plants: one `referent` and
  one `sense`.
- Plants, histories, schedules, arms, targets, and reviewers do not increase N.
- All twelve conversations and their three literal variants must be frozen and
  reviewed before any v11 semantic outcome is produced, including on the local
  0.6B checkpoint.
- The future local apparatus diagnostic uses the first two frozen cases without
  changing them in response to their values.

## 3. Diversity requirements

The twelve cases must span distinct practical domains, names, entities, prose
rhythms, decision structures, message counts, plant locations, and distractor
chains. No shared conversation template may merely substitute nouns.

Planned domains:

1. software migration;
2. museum exhibition;
3. community garden;
4. documentary production;
5. conference logistics;
6. small manufacturing;
7. nonprofit grant program;
8. research-lab operations;
9. apartment renovation;
10. book launch;
11. course planning;
12. expedition logistics.

At least four independent authoring passes should contribute, with no pass
authoring more than three cases. A later reviewer must explicitly test whether
cases are surface variants of one scaffold.

## 4. Conversation structure

Each case must contain:

- one system message followed by alternating user and assistant messages;
- roughly 5,000–8,000 production-tokenizer tokens in the complete source
  prefix before summary generation;
- complete, responsive assistant turns, with no clipping, cap trimming,
  dangling preamble, fake file creation, or claim that an external action was
  performed;
- several unrelated factual decisions and unresolved next steps so the two
  plants are not the only salient facts;
- a frozen `middle_end_msg` boundary;
- both selected plants established and resolved strictly before that boundary;
- a retained tail after the boundary that is coherent under the correct and
  both counterfactual histories and does not explicitly identify either answer
  from either selected target pair;
- no early assertion that independently resolves a selected plant before its
  declared establishing chain;
- no special-token literal, chat-template control string, or hidden metadata in
  message content.

Message count, plant positions, distractor count, and turn lengths should vary
organically across cases. Individual messages should remain below 4,096 tokens.

## 5. Selected plants

### Referent plant

The history presents at least three realistic alternatives, then records one
unambiguous final selection. Freeze:

- a probe that asks which alternative was selected;
- the exact correct target phrase;
- one exact counterfactual target phrase corresponding to a real rejected
  alternative;
- every message that establishes, confirms, or fact-specifically refers to the
  selection.

The referent counterfactual changes the final selection to that exact rejected
alternative and repairs every later fact-specific reference. Merely listing the
alternative is insufficient; it must become the unambiguous final choice and the
correct answer must cease to be selected.

### Sense plant

The history introduces one term or nickname with at least two plausible local
meanings, then unambiguously establishes which meaning the user intends in this
conversation. Freeze:

- a probe that asks what the term meant here;
- the exact correct target phrase;
- one exact counterfactual target phrase corresponding to the other meaning;
- every message that establishes, confirms, or fact-specifically uses the
  mapping.

The sense counterfactual swaps the intended mapping and repairs every downstream
type or factual reference. The original meaning may remain described as the
other sense, but must no longer be established as the intended in-chat sense.

## 6. Matched-pair invariants

For each case, store three full literal message arrays: `correct`,
`wrong_referent`, and `wrong_sense`.

For each correct/counterfactual pair:

- roles, message count, message order, and all chat-structural tokens are
  identical;
- every non-allowlisted role/content pair is byte-identical;
- system and retained-tail messages are byte-identical;
- only messages required for the focal semantic intervention may change;
- the non-focal selected plant and every unrelated factual chain remain
  byte-identical;
- each changed message's content has the exact same token count under
  `Qwen/Qwen3-30B-A3B-Instruct-2507` revision
  `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`;
- complete canonical generation-prefix length and every message-start position
  are identical;
- exact turn-aligned replay (`P`) and ordinary-4096 (`O`) call widths are
  identical;
- token balancing must read naturally. Repetition, conspicuous padding,
  irrelevant qualifiers, invented rationale, and empty token stuffing fail.

Target phrases should be approximately matched in production-tokenizer length.
Outcome scoring will use mean token log probability, but avoid needless target-
length asymmetry at authoring time.

## 7. Draft artifact schema

Each case is one JSON file with at least:

```text
schema: coherent_state_paired_v11_case_draft_v1
status: DRAFT_UNREVIEWED
execution_ready: false
case_id, domain, title
authoring_provenance
middle_end_msg
variants: {correct, wrong_referent, wrong_sense}
plants: {referent, sense}
changed_message_allowlists
distractor_fact_inventory
tokenizer_binding
observed_mechanical_evidence
review: PENDING
warning
```

Each variant contains the complete literal `messages` array. Each plant contains
its ID, category, probe, correct target, counterfactual target, establishing
indices, downstream-reference indices, and a concise factual rationale. The
mechanical evidence records per-variant full-prefix hashes/counts, message-start
positions, corresponding per-message widths, P widths, O widths, changed-token
counts, and decoded round-trip status.

Drafts must never use `PASS`, `FROZEN`, `REVIEWED`, or `execution_ready=true`.

## 8. Summary request frozen for geometry

Mechanical prefix validation appends this exact future request to every variant:

> Write a compact handoff note of 180–240 words for another assistant who will
> continue this conversation without seeing it. Identify the participants and
> their roles, current goals, decisions that still affect next steps,
> constraints, unresolved questions, and the next actions. Prioritize what the
> next assistant needs to act; compress settled background details when they are
> not needed. Use plain prose or concise bullets. Do not add commentary before
> or after the note.

The request is production-like lossy compaction, not a targeted taboo or
instruction to hide either selected plant. Whether the future generated summary
explicitly contains a target is measured and reported; it is not shown to
authors and is not an authoring criterion.

## 9. Conjunctive review gates

No draft becomes frozen or executable without all of:

1. **Mechanical validator:** committed-byte provenance, exact tokenizer and
   template binding, decoded round trip, exact per-message/full-prefix/P/O
   geometry, changed-position confinement, unique coverage, sealed output, and
   false authorization for drafts.
2. **Blind randomized whole-history review:** every correct and counterfactual
   history is reviewed as a singleton for naturalness, complete responses,
   internal coherence, target recoverability, dangling references, padding, and
   editing artifacts. The reviewer is not told which variant or words changed.
3. **Target-aware paired review:** verifies the exact frozen alternative is
   established, the correct answer is no longer selected/intended, all focal
   downstream references are repaired, the intervention is minimal, and the
   non-focal plant/unrelated chains are unchanged.
4. **Diversity audit:** checks topic, structure, plant placement, tone, and
   lexical overlap across all twelve; rejects pseudoreplicated templates.
5. **Adjudication:** any disagreement is resolved by a third independent review.

Review failures produce new additive revisions with new hashes. Never edit a
reviewed snapshot in place or delete a failed candidate.

## 10. Stopping and anti-adaptation rules

- Corpus authoring and review may iterate on text and geometry because no v11
  model outcome exists.
- Once all twelve cases are frozen, no history, target, probe, boundary,
  inclusion rule, or summary request may change in response to local or paid
  semantic values.
- If a frozen case later fails a preregistered technical or full-history
  competence gate, retain and report it under the prospective rule; do not
  silently replace it after seeing graft outcomes.
- No model forward pass beyond tokenizer-only reconstruction is permitted until
  the corpus freeze record and the separate v11 scientific preregistration are
  committed.
