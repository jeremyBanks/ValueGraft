# V12 decision-canary engineered stimulus contract

**Design:** `coherent-state-decision-canary-v12`  
**Author:** Sol — GPT-5.6 Sol, extra-high reasoning effort  
**Date:** 2026-07-11  
**Status:** frozen authoring contract; tokenizer-only work; no model forward

This contract governs `e01`--`e06`. It freezes what authors may create before
any exact-model outcome. Passing this contract does not make a case executable;
mechanical validation plus blind, target-aware, and diversity reviews are still
required by the v12 preregistration.

## 1. Assignment and provenance

- Exactly six cases: primary `e01`--`e04`, sealed ambiguity reserve `e05`--`e06`.
- E01--e03: roughly 1,000--2,000 canonical production-tokenizer tokens before
  the carrier request.
- E04: roughly 4,000--6,000 tokens.
- E05/e06: each 1,000--6,000 tokens, with different lengths and structures.
- At least three independent authoring sessions spanning at least two model
  families; no session authors more than two cases.
- Store the exact runtime model ID, reasoning/effort if exposed, authoring
  session/source ID if available, and timestamp. Do not infer an intended alias.
- Authors may inspect this contract and tokenizer utilities. They must not see a
  local or paid semantic outcome, source K/V, arm result, layer scan, or target
  score. None exists at authoring time.

## 2. Case structure

Each case contains exactly two literal histories:

- `correct` (`C`);
- `wrong_focal` (`W`).

Each history starts with one system message, then alternates user/assistant.
Freeze `middle_end_msg`, which must point to a user message beginning the retained
tail. The focal and control chains finish before that boundary. The retained tail
is byte-identical and natural under C and W.

Each case has exactly:

- one **focal derived decision** changed by W;
- one **non-focal control decision/fact** byte-identical under C/W;
- one focal probe and exact C/W target phrases;
- one non-focal probe and exact target/countertarget phrases.

The focal target must require applying a stated rule, constraint, mapping, or
multi-fact relation. A near-verbatim lookup is insufficient. The minimally
counterfactual history changes one load-bearing input so the derived answer
reverses, and repairs every focal downstream reference. It must not change the
rule, the question, the control fact, or unrelated discussion.

The control probe should be answerable in both histories and should draw on a
different factual chain. Its margin is used only to measure generic disturbance;
it must not be logically downstream of the focal change.

## 3. Conversation quality

Histories should resemble plausible work rather than a benchmark template.
Across cases vary:

- domain and participant roles;
- number and placement of turns;
- length/rhythm of user and assistant messages;
- whether the focal rule is introduced before or after its factual input;
- distractor chains and retained-tail purpose;
- target vocabulary and decision type.

Every assistant response must be complete and responsive. Forbidden:

- clipped/capped endings or dangling preambles;
- fake file writes, calls, emails, commands, or external actions;
- repeated padding, near-synonym stuffing, or irrelevant qualifiers added only
  for tokenizer width;
- a shared scaffold with nouns substituted;
- repeated five-token or other periodic filler;
- hidden instructions, special-token literals, or chat-template strings;
- early statements that independently reveal the final focal answer before the
  declared chain;
- retained-tail text that states or implies either focal target.

The histories may be openly authored. Never call them organic, native, or drawn
from production traffic.

## 4. Exact matched geometry

Binding tokenizer/model:

- `Qwen/Qwen3-30B-A3B-Instruct-2507`;
- revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.

For every C/W pair:

- roles, message count/order, and all structural tokens are identical;
- system, retained tail, non-focal chain, and all non-allowlisted messages are
  byte-identical;
- each allowlisted changed message has exactly the same content-token count;
- every complete canonical message width is identical;
- complete generation-prefix length and every message start are identical;
- the future role-native event kinds/widths are identical;
- all individual messages are at most 4,096 tokens;
- decoded round trip succeeds;
- the two full canonical streams differ at at least one allowlisted content
  position and nowhere else.

Use natural rephrasing inside the genuinely focal changed messages. If exact
width cannot be achieved without awkwardness, redesign the case. Do not pad.

Target phrases should each be one to four tokens when natural and approximately
length-matched. Scoring uses mean token log probability, but needless length
asymmetry fails review.

## 5. Shared carrier and anchor

Authors do not change these strings.

Carrier assistant content:

> The prior discussion established the operating context and resolved one local
> decision. Continue from this handoff, preserve the existing constraints, and
> answer later questions from the state available here. No unresolved action is
> introduced by this note.

Anchor user message:

> Acknowledge receipt of this handoff without adding or repeating any factual
> detail.

Anchor assistant response:

> Acknowledged.

The carrier must be truthful but noncommittal under both histories. If a case
makes any phrase in it favor a target, redesign the case.

## 6. Required JSON schema

Path while drafting:

`data/coherent_canary_v12/drafts/<author-session>/<case-id>.json`

Required top-level structure:

```text
schema: coherent_state_decision_canary_v12_case_draft_v1
design_id: coherent-state-decision-canary-v12
case_id: e01..e06
status: DRAFT_UNREVIEWED
execution_ready: false
stratum: engineered
length_band: short|mid
domain, title
authoring_provenance
middle_end_msg
variants: {correct:{messages:[...]}, wrong_focal:{messages:[...]}}
focal: {...}
nonfocal_control: {...}
changed_message_allowlist
distractor_fact_inventory
retained_tail_purpose
tokenizer_binding
observed_mechanical_evidence
review: PENDING
warning
```

`focal` contains:

```text
plant_id, category=derived_decision
rule_or_relation
changed_input
probe
correct_target
counterfactual_target
establishing_message_indices
downstream_reference_indices
why_derived
why_counterfactual_reverses
```

`nonfocal_control` contains:

```text
plant_id, category=unchanged_control
probe
target
countertarget
establishing_message_indices
why_independent_of_focal
```

`changed_message_allowlist` lists only message indices whose content differs.
Every other role/content pair must be byte-identical.

`observed_mechanical_evidence` records, per variant:

- canonical prefix token count and SHA-256;
- literal-message SHA-256;
- content and complete canonical widths per message;
- message-start positions and SHA-256;
- future role-native event kinds/widths and SHA-256;
- changed token positions and confinement;
- decoded round trip;
- exact production tokenizer/revision.

Drafts never use `PASS`, `FROZEN`, `REVIEWED`, or `execution_ready=true`.

## 7. Assignment domains

To prevent convergent templates, authors take these broad assignments but choose
their own structures and details:

- `e01`: software/release or data operations;
- `e02`: public-service or nonprofit allocation;
- `e03`: physical logistics or field operations;
- `e04`: long mid-length case in research, manufacturing, or safety operations;
- `e05`: event, accessibility, or facilities decision;
- `e06`: publishing, education, or expedition decision.

Assignments are topic boundaries, not story templates. No two cases may share
the same rule form, target pair, participant rhythm, or retained-tail pattern.

## 8. Stop conditions for authors

Stop and preserve a WIP draft rather than forcing completion if:

- exact token geometry requires unnatural text;
- the focal answer becomes a lookup rather than a derived decision;
- W creates a contradiction outside the allowlist;
- the retained tail reveals a target;
- the non-focal control depends on the focal change;
- tokenizer/chat rendering cannot be reconstructed exactly;
- the assigned token band cannot be reached without filler.

Authors commit only their own new case files/checkpoints and report their exact
runtime model provenance. They do not review or approve their own cases.
