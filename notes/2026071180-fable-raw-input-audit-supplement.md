# Fable supplement attempt: no completed review

**Requested reviewer:** `claude-fable-5`

**Date:** 2026-07-11

**Recorded by:** Sol — OpenAI GPT-5.6 Sol, extra-high reasoning effort

**Status:** **INCOMPLETE — contains no Fable scientific conclusion and must not be cited as an endorsement.**

## Purpose

Fable's completed raw-input audit in `notes/2026071177-fable-raw-input-and-independence-audit.md` preceded two material byte-level findings:

1. the frozen `G_wrong` source uses heavily cycled donor-message token pools and is not a coherent wrong history (`notes/2026071179-sol-wrong-history-control-invalidity.md`);
2. the schedule labeled `message_block` / `message-aligned` is actually a coarse system/whole-history/request schedule, not per-turn replay (`notes/2026071178-lagrange-schedule-robust-estimand-design-audit.md`).

A corrective Fable supplement was requested so those new facts could supersede inaccurate donor/schedule descriptions in note 1177.

## What happened

The first attempt resumed Fable's very large prior session. It exhausted a `$4` cap re-caching the accumulated context and produced no review.

A fresh attempt was then limited to notes 1177–1179 and the three directly cited code/spec snippets. The owner clarified during execution that routine Fable reviews should receive minimal, decision-specific context and should not approach/exceed `$4`; large contexts are reserved for final synthesis. Sol interrupted the call at a reported `$2.343170`. Its local transcript shows only orientation and file-reading messages. It had not written the required note or emitted a scientific assessment.

The call was not resumed. No prose in this file is attributed to Fable.

## Current evidence and decisions

- Fable's earlier note 1177 remains useful for its completed general raw-input audit, but its description of the wrong-history donor and schedule must be read together with the later corrections in notes 1178 and 1179.
- Claude Opus 4.8 independently evaluated the new wrong-history finding in `notes/2026071156-sol-fable-execution-coordination.md`. It agreed that the cyclic arm is co-primary-invalidating and endorsed minimally counterfactual histories only conditionally: exact token-length equality, alignment to the frozen counterfactual target, complete downstream-reference repair, and blind decoded-coherence review are all mandatory.
- Sol retains final authority and has not yet frozen a replacement arm. No semantic work may use the cyclic `G_wrong` construction.

## Process correction

`AGENTS.md` now requires Fable prompts to use the minimum sufficient decision-specific evidence bundle, points to generated summaries only as optional orientation, explicitly excludes re-litigation of unrelated settled questions, and reserves comprehensive context for final paper synthesis/review.
