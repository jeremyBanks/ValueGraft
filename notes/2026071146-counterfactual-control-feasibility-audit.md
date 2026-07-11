# Plant-specific counterfactual-control feasibility audit

**Author:** Sol (`gpt-5.6-sol-xhigh`)

**Date:** 2026-07-11

**Status:** Independent, read-only design audit preserved at owner request. No
model forward pass or semantic outcome was produced. The literal candidate
counterfactuals below are **unreviewed feasibility witnesses, not frozen inputs
and not execution-ready controls**. They have not passed the required independent
blind naturalness/coherence review.

## Executive recommendation

Retire the invalid conversation-level cyclic `G_wrong` control from every future
semantic assay. Replace it with a plant-specific minimally counterfactual source,
`G_cf[i,p,s]`, where:

- `i` is the conversation;
- `p` is the selected plant (`referent` or `sense`);
- `s` is the prospectively frozen source schedule (`P` or `O`);
- only the evicted messages needed to establish that plant's already frozen
  counterfactual answer, plus every fact-specific downstream reference within the
  evicted block, are changed;
- system, retained tail, summary request, assistant generation header, every
  unapproved message, summary token IDs, summary logical positions, destination,
  and downstream probe remain fixed.

This is the cleanest replacement for the invalid external-donor/cyclic control.
A coherent unrelated donor could remain as a broader secondary control, but it
cannot substitute for the plant-aligned minimal counterfactual in the co-primary
history-specificity contrast.

The banked c10/c02 histories show that coherent-looking, target-aligned,
exact-per-message-token-length counterfactuals are constructible. They do not
establish feasibility for the future fresh exact-30B renders. The actual 30B
assistant replies may contain different downstream references or retained-tail
leakage, and those literal renders must be authored and reviewed separately
before any summary/state outcome exists.

## Why the replacement must be plant-specific

The two scored plants in a conversation have different frozen alternatives. One
shared wrong history cannot be minimally counterfactual for both without changing
two semantic chains and broadening the intervention. The clean construction is
therefore one wrong source per plant:

- `W_ref`: flips only the selected referent decision to the referent target's
  frozen alternative;
- `W_sense`: flips only the selected sense mapping to the sense target's frozen
  alternative.

Each `W_p` is scored against its own plant `p`. Scoring `W_ref` on the sense plant
and `W_sense` on the referent plant is useful as a descriptive specificity
diagnostic, but it does not create extra independent samples.

## Exact tokenizer and construction path checked

The feasibility check loaded:

```text
model/tokenizer: Qwen/Qwen3-30B-A3B-Instruct-2507
revision: 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe
tokenizer class observed: Qwen2Tokenizer
```

For each candidate, the check used the repository's actual
`generation_prefix_ids`/dummy-user canonical chat-template path. It compared the
complete correct and counterfactual generation prefixes, all `<|im_start|>`
positions, and each changed message's production-tokenizer content count. It did
not rely only on word counts or a generic tokenizer.

Observed results:

| Case / plant | Messages changed | Per-message content-token counts, correct → CF | Full canonical prefix | Message starts | Differing full-prefix token positions |
|---|---|---|---:|---|---:|
| c10 referent | 27, 28 | 72→72; 409→409 | 8,430→8,430 | identical | 27 |
| c10 sense | 31, 32 | 57→57; 403→403 | 8,430→8,430 | identical | 129 |
| c02 referent | 31, 32 | 68→68; 180→180 | 8,385→8,385 | identical | 120 |
| c02 sense | 11, 12, 14 | 60→60; 398→398; 381→381 | 8,385→8,385 | identical | 293 |

Because every message boundary is unchanged, the counterfactual and correct
histories have exactly the same `P` schedule. Because total length is unchanged,
they also have exactly the same `O` schedule.

## Literal c10 feasibility witnesses

### c10 first referent

Frozen correct target:

```text
an L-shape with a center island where the dining door was
```

Frozen counterfactual target:

```text
a U-shape with a peninsula
```

Only evicted messages 27 and 28 change. Earlier layout discussion lists options
but does not assert the final selection. Retained-tail message 37 says only “the
layout we decided on” and remains compatible with either decision.

#### Candidate message 27 — user — 72 tokens

```text
Layout verdict is in. Naomi drew up three schemes: (1) keeping the galley but widening it, (2) an L-shape with a center island where the dining door was, and (3) a U-shape with a peninsula. After living with the tape outlines on the floor for a week, we're going with the third one.
```

#### Candidate message 28 — assistant — 409 tokens

```text
**Brilliant decision — and a perfect example of how thoughtful, hands-on evaluation leads to the best outcome.**

Let’s now turn this into a **clear, official, and professionally documented layout verdict** — so it’s preserved, shared with your contractor, and referenced throughout the project. This is the *final* layout decision — and it should be treated as a binding, design anchor.

---

## ✅ Official Layout Decision: Final Kitchen Layout Approved

> 📂 **File Name**: `Kitchen_Floorplan_Verdict_v1.pdf`  
> 📝 **Date**: April 7, 2024  
> 🏡 **Project**: Kitchen Remodel – Final Layout Selection & Rationale  

---

### 🏁 Final Decision: **U-Shaped Kitchen with a Peninsula (Option 3)**

**Statement**:  
> *“The final kitchen layout is the **U-shaped configuration with a peninsula** located where the dining room door previously was. This layout is fully approved and will proceed as the foundation for all design, construction, and permitting decisions.”*

---

### 📌 Rationale & Why This Was Chosen

| Reason | Explanation |
|-------|-------------|
| 🔄 **Work Triangle Optimization** | The new U-shape creates a true, efficient work triangle (sink, stove, fridge) with natural flow and reduced crossover. |
| 🪑 **Peninsula as a Functional Hub** | The peninsula provides a **work surface**, **extra storage**, and **seating** — perfect for cooking, hosting, and gathering. It also breaks the linear flow of the galley, creating a more open, inviting space. |
| 📐 **Proven by Real-World Testing** | We lived with **tape outlines** of all three options on the floor for **7 days** — the U-shape felt most natural, balanced, and usable. The peninsula created a sense of *space* and *center*, which we both deeply valued.
```

The natural balancing edit “approved” → “fully approved” offsets the tokenizer
difference from the semantic substitutions. This is exactly the kind of edit a
blind reviewer must evaluate rather than accepting it merely because the count
matches.

### c10 first sense

Frozen correct target:

```text
the back deck rebuild bundled into the remodel
```

Frozen counterfactual target:

```text
the cedar closet lining idea upstairs
```

Only evicted messages 31 and 32 change. The retained-tail question “Should Cedar
wait...” is generic and remains compatible with either mapping.

#### Candidate message 31 — user — 57 tokens

```text
Shorthand warning: in our house 'Cedar' means two projects. My partner uses it for the back deck rebuild we're bundling into this remodel, but when I say Cedar with you I mean the cedar closet lining idea upstairs. Closet only, always, in this chat.
```

#### Candidate message 32 — assistant — 403 tokens

```text
Perfect — and **you just saved yourself from a major communication hiccup**.

This is **exactly the kind of subtle, house-specific shorthand that can turn into a disaster in a big project** — especially when multiple teams (contractors, architects, family) are involved.

Let’s now **officially document and clarify** the meaning of “Cedar” — so it’s instantly clear, protected, and prevents future confusion.

---

## ✅ Official Shorthand Clarification: "Cedar" Means Closet — Only

> 📂 **File Name**: `Kitchen_Terms_Clarification_v2.pdf`  
> 📝 **Date**: April 7, 2024  
> 🏡 **Project**: Kitchen Remodel — Preventing Miscommunication in Shared Language  

---

### 🔒 Shorthand Rule: **"Cedar" = Cedar Closet Lining (Only)**

**Statement**:  
> *“In all project communications, meetings, emails, and design notes, the term **'Cedar'** will be interpreted as the **cedar closet lining idea upstairs**, not any other project.*  
>  
> ✅ **Cedar (in this project)** = Closet only — including lining, trim, shelves, and installation.  
>  
> ❌ **Cedar ≠ back deck rebuild** — that is a separate, unrelated project (upstairs, in the master bedroom).  
>  
> 🚫 **No usage of 'Cedar' to refer to anything else** — including deck framing or finishes.  
>  
> This clarification applies to all team members, contractors, and future reference materials.*

---

### 📌 Why This Matters

| Risk | Explanation |
|------|-------------|
| 🚨 **Misalignment** | If a contractor or designer says “we’ll do cedar” and means the deck, you could be charged for closet work that wasn’t requested — or worse, work on the wrong project.
```

## Literal c02 feasibility witnesses

### c02 first referent

Frozen correct target:

```text
boxed lunches from the worker co-op bakery on Fifth
```

Frozen counterfactual target:

```text
a hot buffet catered by Cateright
```

Only evicted messages 31 and 32 change. Retained-tail message 37 refers only to
“the catering we picked,” so it remains compatible with both selections.

#### Candidate message 31 — user — 68 tokens

```text
Catering decision. Three options on the table: (1) food trucks parked in the courtyard, (2) a hot buffet from Cateright, or (3) boxed lunches from the worker co-op bakery on Fifth. I compared quotes and dietary coverage, and I'm going with the second one. Locking it in today.
```

#### Candidate message 32 — assistant — 180 tokens

```text
**Perfect — and a *deliberate, guest-friendly* choice.**  

You’ve just made one of the most *important, practical, and intentional* decisions in event logistics: **catering isn’t just about food — it’s about equity, access, and values.**

Let’s **officially document and formally celebrate this decision** — and show how it reflects the *principles of Riverbend*: **accessibility, local support, and human-centered planning**.

---

## ✅ Riverbend DevCon: Catering Decision — Finalized & Rationale

> 🎯 **We selected Option 2: A Hot Buffet Catered by Cateright**  
> ✅ *Because it aligns with Riverbend’s values of variety, convenience, and guest comfort.*

---

### 📌 Why This Choice Wins (Over the Other Two)
```

This candidate is token-exact but its residual general “equity/access/local
support” framing is precisely the sort of subtle naturalness/congruence issue the
required blind review may reject. It is preserved as an observed feasibility
witness, not endorsed as final prose.

### c02 first sense

Frozen correct target:

```text
the mentorship track pairing first-time speakers with veterans
```

Frozen counterfactual target:

```text
the sponsors' Beacon networking lounge
```

Messages 11 and 12 establish the mapping. Message 14 is also fact-specific: it
says the speaker-ratio plan supports “the Beacon mentorship track” and that ten
speakers are paired “via Beacon.” Those two references must change. The later
retained-tail request for a “program blurb for Beacon” is generic and remains
compatible with the counterfactual mapping.

#### Candidate message 11 — user — 60 tokens

```text
Naming clarification before it bites us: 'Beacon' means two things around here. We run a Beacon mentorship track pairing first-time speakers with veterans for talk prep, but when I say Beacon with you I mean the sponsors' networking lounge. That's the one I care about in this chat.
```

#### Candidate message 12 — assistant — 398 tokens

```text
Absolutely — thank you for the *clear, thoughtful clarification*.

You're absolutely right: **"Beacon" has dual meaning**, and in volunteer-driven events, *ambiguous naming can lead to confusion, misaligned effort, or even duplicated work*.

Let’s **officially resolve this once and for all** — so you and your team (and sponsors) are on the same page.

---

## 🚨 Naming Clarification: “Beacon” — The Official Breakdown

> ✅ This is *now* a living, shared understanding — no more guesswork.

### 🟦 Beacon (Sponsor Networking Lounge) — Your Priority
- **What it is**: A dedicated, branded lounge where sponsors can meet attendees for relaxed networking.
- **Goal**: Give sponsors and attendees a comfortable place for useful, low-pressure conversations.
- **Format**:  
  - Hosted introductions and open drop-in hours  
  - Coffee, seating, and quiet conversation areas  
  - Optional scheduled sponsor office hours
- **Operating plan**: Keep the lounge open between sessions, staff it with a host making introductions, provide coffee and comfortable seating, and reserve short windows for founders or hiring teams. The tone should stay conversational rather than sales-heavy, with signage so attendees can opt in without pressure.
- **Why it works**: Supports useful connections without disrupting sessions — *exactly what Riverbend needs*.

> 🔑 **Use this phrase in all internal comms**:  
> *"The Beacon sponsor lounge — for relaxed networking."*

---

### 🟨 Beacon (Mentorship Track) — Speaker-Focused
- **What it is**: A structured program pairing new speakers with experienced mentors for talk preparation.
- **Goal**: Help first-time speakers prepare stronger talks through structured practice and veteran support.
- **Crucially**: This is **not** the same as the sponsor lounge — so we must *never confuse them*, especially in planning.
```

#### Candidate downstream message 14 — assistant — 381 tokens

```text
This is one of the *most important and nuanced* decisions in speaker programming — and you’re asking the right question.

You want **fresh voices** to fuel innovation and diversity — but you also need **stability and quality** to ensure the event feels credible, valuable, and engaging for *all* attendees.

Let’s build a **data-informed, balanced, and sustainable speaker mix** — without overreaching or under-supporting new speakers.

---

## 🎯 Riverbend DevCon: Optimal Speaker Ratio (Newcomers vs. Experienced)

> ✅ Goal: Maximize diversity, reduce risk of shaky talks, and *build confidence* — without sacrificing innovation.

---

### 📊 Recommended Ratio: **50% Experienced, 50% New Voices**  
*(With strategic guardrails)*

#### Why this works:
- **Balances risk and innovation** — new voices bring fresh ideas, while experienced speakers provide reliability and grounding.
- **Supports the separate mentorship track** — pairing new speakers with veterans creates a feedback loop that improves quality *without* requiring new speakers to go it alone.
- **Reduces risk of "shaky" talks** — experienced speakers often help structure talks, handle Q&A, and manage nerves.
- **Builds trust** — attendees know *they can learn from someone who’s been there* — and that new voices are *supported*, not just "thrown in".

> 📌 Example:  
> - 70 talks total → 35 new, 35 experienced  
> - 10 new speakers are matched with 10 veterans through mentoring  
> - The remaining 25 new speakers are paired with *small peer groups* or *light peer support* (to avoid isolation)

---

### 🚩 How to Protect Against "Shaky" Talks (Pro Tips)
```

## Exact schedule matching

Define schedules without production-sounding ambiguity:

- `P = turn_aligned_replay`: system as one call; every historical canonical
  message block as one call; any individual message longer than 4,096 tokens
  chunked only within that message; final summary request plus assistant header
  as one call.
- `O = ordinary_4096`: consecutive calls of at most 4,096 over the complete
  contiguous source prefix.
- `D = gapped_destination_canonical`: the one fixed fresh system/request
  two-island destination followed by stepwise forcing of the saved summary. `D`
  is not repacked into `O`.

The old five-call coarse system/history/request schedule `B` is not
message-by-message and should not enter the replacement semantic assay.

Observed c10 schedules:

```text
P = [23, 59, 350, 48, 383, 44, 362, 63, 402, 58, 225, 54, 413,
     39, 405, 63, 422, 53, 424, 57, 401, 45, 411, 52, 418, 60,
     417, 77, 414, 32, 396, 62, 408, 46, 190, 22, 23, 40, 16,
     25, 69, 29, 255, 30, 422, 123]
O = [4096, 4096, 238]
```

Observed c02 schedules:

```text
P = [22, 59, 395, 40, 422, 43, 420, 46, 416, 54, 330, 65, 403,
     38, 386, 52, 404, 57, 361, 55, 400, 38, 417, 48, 370, 64,
     405, 60, 404, 43, 403, 73, 185, 40, 197, 40, 100, 39, 20,
     26, 16, 23, 394, 32, 357, 123]
O = [4096, 4096, 193]
```

## Exact arm and estimand structure

Generate exactly one greedy summary from the correct source under `P`. Preserve
its incremental summary rows as the source of record. Force those exact summary
IDs everywhere else.

Per conversation, construct seven compacted boundaries:

| Arm | Summary-row source | Destination / continuation |
|---|---|---|
| `F` | Fresh same-token rows from `D` | Fixed `D`; common tail and probe |
| `C_P` | Correct history under `P` | Identical to `F` |
| `C_O` | Same correct history/IDs under `O` | Identical to `F` |
| `W_ref,P` | Referent-specific minimal CF under `P` | Identical to `F` |
| `W_ref,O` | Same referent CF/IDs under `O` | Identical to `F` |
| `W_sense,P` | Sense-specific minimal CF under `P` | Identical to `F` |
| `W_sense,O` | Same sense CF/IDs under `O` | Identical to `F` |

For plant `p`, using the already frozen correct-minus-counterfactual downstream
target margin `Y`:

```text
GF_i,p,s = Y_i,p(C_s) - Y_i,p(F)
GW_i,p,s = Y_i,p(C_s) - Y_i,p(W_p,s)
```

Conversation-level values remain equal-weight means of referent and sense:

```text
GF_i,s = mean_p GF_i,p,s
GW_i,s = mean_p GW_i,p,s
```

The schedule-robust confirmatory intersection should require the co-primary
lower confidence bounds to clear zero under both schedules:

```text
GF_P > 0  AND  GW_P > 0  AND  GF_O > 0  AND  GW_O > 0
```

Also report the signed schedule interactions and every underlying plant cell:

```text
Delta_GF = GF_O - GF_P
Delta_GW = GW_O - GW_P
```

No zero-mean, independence, quadrature, variance-inflation, or common-mode
assumption is made. The two schedules are robustness conditions on the same
conversations; inferential `N` remains 12, not 24. The two plants also remain
within-conversation measurements, not independent samples.

## Required construction and review workflow

### 1. Correct render-only phase

Freshly render the exact-30B correct conversations at temperature zero. Commit
each complete conversation and reply record immediately. Stop before generating
any summary, K/V state, target score, or semantic arm outcome.

The current one-process render→summary→arms path is unsuitable for this control.
It must become a durable two-stage workflow with a human/model-independent
content-authoring pause.

### 2. Counterfactual authoring phase

For each selected plant, give an independent content author only:

- the committed correct render;
- the plant probe;
- its frozen correct and counterfactual targets;
- the exact evicted/retained boundary;
- the exact tokenizer/message-length requirement.

The author changes the smallest sufficient set of evicted messages and every
evicted fact-specific downstream reference. The author must not see a generated
summary, summary likelihood, K/V, plant margin, arm score, or any semantic
outcome.

### 3. Mechanical validator

The validator must fail closed unless all of the following hold:

- system, retained tail, request, assistant header, and all unapproved messages
  are byte-identical;
- every changed message index is strictly before `middle_end_msg`;
- no chat special token or template control string is introduced;
- complete canonical prefix lengths match;
- all canonical `<|im_start|>` positions match;
- every canonical per-message block width matches;
- exact `P` and `O` call-width arrays match;
- summary start and every summary logical position match;
- the changed-message allowlist, token IDs, hashes, complete decoded correct/CF
  histories, and target binding are durably committed;
- the correct source reconstructed from the CF artifact remains byte-identical to
  the saved correct render outside the approved changes.

Checking `tokenizer.encode(message["content"])` alone is insufficient. The full
canonical chat rendering must be checked because assistant-template
normalization can change a block.

### 4. Semantic validity and blind review

Use two independent reviewers per plant, with a third adjudicator for any
disagreement.

One target-aware factual auditor checks that:

- the history establishes exactly the already frozen alternative;
- the correct answer is no longer selected or assigned;
- every downstream reference is consistent;
- all other selected and unselected plants retain their original meanings;
- changes are the smallest coherent semantic chain, not broad rewriting.

One blind naturalness reviewer sees randomized singleton histories without arm
labels, schedule labels, or indication of which text was edited. The reviewer:

- answers the plant probe from the history;
- must recover the intended alternative;
- flags contradictions, dangling references, conspicuous padding, repetition,
  register shifts, or edited-text artifacts;
- separately answers the other selected plant to verify it remains unchanged.

Commit literal review prompts, randomized presentation order, verdicts, and
adjudication before any summary/state scoring. A failed case is redesigned. If a
valid exact-length history cannot be produced, the case is prospectively dropped
through an additive preregistration amendment; it is never silently excluded
after model outcomes.

### 5. State and scoring phase

Only after all counterfactuals and review attestations are committed:

1. generate `C_P` once;
2. force the same IDs under `P` and require generated/forced `C_P` identity;
3. force them through `C_O`, `W_ref,P`, `W_ref,O`, `W_sense,P`, and
   `W_sense,O`;
4. force them through fixed destination `D` to build `F`;
5. persist prefix/schedule provenance, complete row lineage, wrong-summary NLL
   per plant and schedule, token log probabilities, plant margins, conversation
   outcomes, and signed schedule interactions;
6. independently harvest and recompute all construction and inferential cells.

## Claim boundary

These histories are externally edited counterfactual interventions. They are not
subject-native alternative conversation generations. Blind review and minimal
editing reduce the old corruption/distribution confound but do not erase this
provenance difference.

The defensible claim is:

> correct-history summary state versus a coherent, position- and
> schedule-matched minimally edited counterfactual history under identical
> forced summary tokens.

It is not:

> two naturally generated alternative conversations.

The exact correct summary is intentionally counterfactually forced after the
alternative history. Wrong-summary surprisal remains a mandatory diagnostic and
an alternative explanatory channel; it does not become an outcome.

## Downstream redesign inventory

Historical frozen documents must not be rewritten. A new additive amendment must
explicitly supersede their active `G_wrong` provisions while preserving the old
text as provenance.

Required new-version surfaces include:

### Scientific contracts and data

- base coherent-state preregistration and active `G_wrong` provisions in
  Amendments 1, 3, 4, 5, 8, and 9, superseded additively;
- schedule-origin/estimand follow-up design, replaced with the plant-specific
  `P/O` factorial;
- new full-history plant-specific CF artifacts, changed-message manifests,
  canonical token hashes, and blind-review attestations;
- unchanged `data/coherent_state_targets.json`, now directly bound to each CF;
- c13–c30 external donors retired from the new apparatus, preserved only as
  historical evidence;
- calibration's separate minimal label counterfactual renamed/scoped so its
  `G_wrong` label is not conflated with the invalid semantic donor control.

### Core construction, execution, and analysis

- `src/coherent_state_cases.py`;
- `src/coherent_state_tokens.py`;
- `src/coherent_state_runtime.py`;
- `src/run_coherent_state_hf.py`;
- `src/coherent_state_store.py`;
- `src/analyze_coherent_state.py`;
- `src/coherent_state_calibration.py`;
- `src/l_coherent_state_hf.py`;
- `src/coherent_state_integrity.py`;
- `src/validate_coherent_external_donors.py`, retired/replaced by a
  plant-counterfactual validator;
- any numerical diagnostic that imports `WRONG_DONOR` or
  `matched_wrong_prefix_ids`.

### Independent harvest, release, jobs, and tests

- `scripts/validate_coherent_harvest.py`;
- `scripts/validate_semantic_release.py`;
- coherent-state technical, semantic, release, trusted-job, gate, and monitor
  wrappers that bind design IDs, inventories, stage names, or arm schemas;
- all case/token/runtime/store/analysis/ladder/driver/harvest/release tests that
  contain `G_wrong`, `WRONG_DONOR`, `matched_wrong_prefix_ids`, or
  `external_donor_construction`;
- `STATE.md`, `DECISIONS.md`, and later findings/paper language updated only
  after the new design is frozen and observed.

## Blocking conditions

1. **No actual 30B counterfactuals exist.** The four candidates above apply to
   committed banked histories, not future fresh exact-30B assistant replies.
2. **Independent blind review has not occurred.** Every literal candidate in
   this note is unreviewed and not execution-ready.
3. **Retained-tail leakage can invalidate a case.** A fresh assistant tail may
   restate the selected fact. Because retained-tail bytes must remain identical,
   such a case cannot be repaired by editing the tail; it must be prospectively
   redesigned or dropped.
4. **The other ten conversations are unverified.** Exact-length semantic swaps,
   especially long assistant elaborations and sense mappings, may fail natural
   review even if mechanical matching is possible.
5. **The schedule-origin causal diagnostic remains unresolved.** Repairing the
   wrong control does not establish source-schedule validity or authorize paid
   semantic execution.
6. **V10 is permanently non-authorizing.** This replacement requires a new
   additive scientific version, new ladder stages, apparatus inventory, harvest
   contract, release resolver binding, and final attestation. It cannot be
   installed by waiving or reinterpreting the failed v10 gates.
7. **The authored-control provenance remains a limitation.** Exact geometry and
   blind naturalness cannot turn externally edited assistant replies into native
   alternative generations.

## Final assessment

Plant-specific minimally counterfactual histories are a practical replacement
direction. On the literal banked c10/c02 inputs, four examples were observed to
simultaneously satisfy target alignment, exact content-token counts in every
changed message, exact full canonical prefix length, identical canonical message
starts, and therefore exact `P/O` schedule matching. The c02 sense example also
demonstrates why whole-history downstream-reference audit is load-bearing: an
additional assistant message had to change even though it was not the establishing
turn.

That feasibility result is necessary but not sufficient. The candidates remain
unreviewed, non-native, and non-execution-ready. The decisive next evidence is a
committed render-only → counterfactual-authoring → mechanical-validation → blind-
review pipeline applied to the actual fresh exact-30B histories before any
summary, K/V, or semantic score is produced.
