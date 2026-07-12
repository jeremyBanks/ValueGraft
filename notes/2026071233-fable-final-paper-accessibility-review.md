# Final-paper accessibility and publication-readiness review

**Reviewer runtime identity:** Anthropic Claude Fable 5 (exact runtime model ID: `claude-fable-5`).
**Review angle:** fresh first-time reader — an informed systems/LLM practitioner encountering the project cold — assessing clarity, narrative flow, and memorability of PAPER.md as of 2026-07-12. No scientific claims are re-litigated; the fixed floor (no practical benefit established; literal hypothesis unresolved; all stated caveats on the SWE proxy, fixed scalars, formal v12 stop, e01 diagnostic, legacy lineage, and open quantization axis) is preserved throughout. Scope: PAPER.md only; no repository inspection, no subagents, no browsing.

---

## Topline

The paper is scientifically disciplined and, by the end, trustworthy — but the first-time reader has to earn that trust through roughly 2,000 words of undefined project vocabulary before the paper starts helping them. The core problems are all fixable at the prose level: the abstract's second paragraph is a wall; key terms ("plant," "out-of-fitting," "values under fresh keys") are used before they are defined; §4.1 front-loads the weakest, most caveat-dense material as the reader's first contact with actual evidence; the e01 estimand block in §8.1 asks the reader to hold six derived quantities from a formula line; and there is no single practical-takeaway paragraph a practitioner can carry away. None of these require changing a number or a claim.

The paper's best assets — the §3 stratum table, §8.2's honest bullet reading, and the closing paragraph of §9 ("the check you need is the one aimed at the assumption you did not know you were making") — are under-leveraged. Promote them; they are what a reader will remember.

---

## Title and opening

The current title is accurate and honest but reads as two stapled clauses, and "Inconclusive Evidence" undersells the part of the paper that *is* conclusive (the catalogue). A question title is both more memorable and more precise about the epistemic state, and it cannot overclaim in either direction:

> **Can Old KV-Cache State Be Transplanted Across Conversation Compaction? Inconclusive Evidence and Ten Ways the Measurement Failed**

Alternative, if a declarative is preferred: *"Transplanting KV-Cache State Across Conversation Compaction: What We Could Not Establish, and Ten Measurement Failures Worth Keeping."*

**Proposed opening move** (for the abstract's first three sentences; the current first sentence is good and should survive): state the loss, the question, and the answer's shape before any apparatus:

> When a long LLM conversation is compacted into a text summary, the model's generation-time key/value state is discarded along with the text. We asked whether a training-free, post-hoc transplant of that old state — old value vectors under fresh keys — into the compacted context can measurably restore what compaction destroyed, on an unmodified instruction model. It cannot, on the evidence here: the project established no practical benefit, and the clean literal hypothesis remains unresolved.

(The third sentence's "it cannot, on the evidence here" is bounded by the clause that follows; if that still feels too strong, "We could not show that it does" is the safe fallback.) Then one sentence on the strongest surviving lead with its caveats, one on the stop and diagnostic, one on the catalogue.

---

## Abstract

**Blocker.** The second paragraph packs the four-strata warning, the SWE proxy with CI, the formal stop, the diagnostic, and the catalogue into one breath, and uses "evidence strata that must not be pooled" before a reader knows what a stratum is. Restructure as four short sentences in a fixed order: answer → strongest lead (plainly glossed) → stop and diagnostic → durable contribution. Two specific fixes:

1. **Gloss the headline magnitude once.** "+0.0135 nats/token" means nothing to most readers at first contact. One clause suffices: "about a 1.4% relative increase in per-token likelihood of the historically demonstrated next action" (e^0.0135 ≈ 1.0136). This adds scale without changing the claim, and makes "small" concrete.
2. **Gloss the intervention once.** "Old values under fresh keys" needs its one-line intuition at first use: fresh keys control where attention looks; old values are the content retrieved. That single sentence unlocks §§3–8 for a reader who hasn't lived in this codebase.

---

## Narrative structure and section order

The overall skeleton — question (§1–3), evidence by stratum (§4–8), failures (§9), ledger (§10–11) — is right, and the §3 stratum table is the best orientation device in the paper. Three flow problems:

**Blocker: §4.1 is the reader's first contact with evidence, and it is a nine-part apparatus bullet wall.** Nearly all of it is duplicated (correctly, and in more detail) in Appendix A.1. The principle that "the apparatus must travel with every legacy number" is served by a 4–5 sentence summary naming the load-bearing defects — prefill-reconstructed source state, non-subject conversation bodies, combined tail+summary graft, broken head-map lineage — with a pointer to A.1 for the rest. As written, a first-time reader's takeaway from §4.1 is "I am not equipped to read this paper," which is the wrong lesson from a section whose actual content is a clean null.

**High-value: the strata are presented weakest-first.** Chronological order is defensible, but the reader meets the most compromised evidence (§4) before the strongest lead (§5) and the cleanest design (§6–8). If reordering is off the table, add one signposting sentence at the top of §4: "This stratum is presented first for chronology; its headline is a null in a defective apparatus, and the paper's strongest surviving lead is §5." Cheap, and it tells the reader why they are reading.

**Minor: the "must not be pooled" refrain.** It appears in the abstract, §3 (twice), §5, and elsewhere. State it once as a convention in §3 and cross-reference; repetition reads as anxiety rather than rigor.

---

## Jargon and first-use definitions

**Blocker-level items** (each is one sentence to fix):

- **"Plant" / "planted target"** is never defined; it first carries weight in §4.2 ("203 scored planted targets"). Define at first use: a fact deliberately placed early in the conversation whose later recall is probed.
- **"Out-of-fitting"** is a project coinage; readers will expect "held-out" and wonder if the difference is meaningful (it is — §5 explains the pool was previously inspected). Define it where §5 first uses it, in exactly that contrast.
- **"Teacher-forced" and "q = 1"** — one parenthetical each at first use; "q = 1" especially reads as an internal flag.
- **Codename load.** e01, v10/v11/v12, c07–c24, N/P, R1–R3, FF/FC/…/WW, D/SEL/H+/U/U+ — each is defined somewhere, but the density peaks exactly at the paper's most important table (§8.1). A five-line glossary box at the top of §6 (or an inline recap caption on the §8.1 table) would pay for itself many times over.

**§8.1 specifically (blocker):** the formula block defining D, SEL, H+, U, U+ is correct but unreadable as a first pass. Add one plain-English line per estimand, e.g. "D: does correct-history state favor the correct answer more than wrong-history state does? SEL: is that movement specific to the focal question? H+: did the correct answer itself get more likely? U/U+: does correct-history state beat fresh re-encoding?" §8.2 already reads the table honestly — the definitions just need to meet the same standard.

---

## Tables

- **§3 stratum table:** excellent; consider referencing it from §1 so readers know an orientation map exists.
- **§4.2 pooled and block-split tables:** fine; the prose around the split is careful and clear.
- **§8.1:** with the plain-English estimand lines above, workable. The bolding of the primary row is good. Consider a one-word verdict column ("adverse / favorable / mixed") mirroring §8.2 — it is descriptive, not inferential, so it stays inside the floor.
- **§9 catalogue tables:** the out-of-order item numbers (1, 4, 5 / 2, 3, 6, 7, 9 / 8, 10) trip the reader even though the preamble explains why. Either renumber within groups and give the chronology mapping in one footnote, or move the explanation into each table caption. Also, `QUERY_SHAPE_ROUNDING` is an internal flag name meaningless to outsiders — describe the guardrail, not the variable.

---

## Failure catalogue (§9)

This is correctly billed as the durable contribution and mostly delivers. Two improvements: (1) the closing paragraph is the most memorable writing in the paper — echo its final line in the abstract or §10 so it is not buried; (2) several "Failure" cells run 60+ words before the reader learns the category — lead each cell with the short bolded lesson (most already do; items 1 and 4 don't).

---

## Conclusion and practical takeaway

**Blocker: there is no paragraph a practitioner can act on.** §10's ledger is precise and §11's requirements list is genuinely useful, but neither answers "what should someone building agent infrastructure do tomorrow?" Add a short **Practical takeaway** paragraph at the top of §10, fully inside the floor:

> Do not build post-hoc KV transplantation into a compaction pipeline on this evidence: no variant tested here demonstrated behavioral recovery or task-level benefit, and the one favorable diagnostic cell is N=1, uncontrolled, and schedule-sensitive. If you attempt state-preservation research, the ten guardrails in §9 are the transferable output — especially: treat replay schedule as an experimental variable, verify your placebo constructor works in the production dtype before the run, and preserve the tensor state your future controls will need.

§11's opening ("No valid paired live-agent evaluation … and none is authorized") is good and blunt; keep it.

---

## Consolidated lists

**Blockers (publication-readiness, all prose-level):**
1. Restructure the abstract's second paragraph; gloss the +0.0135 magnitude and the values-under-fresh-keys intervention in one sentence each.
2. Define "plant," "out-of-fitting," "teacher-forced," "q = 1" at first use.
3. Compress §4.1's apparatus wall to a summary plus pointer to Appendix A.1.
4. Add plain-English lines for the §8.1 estimands.
5. Add a practical-takeaway paragraph to §10.

**Optional polish:**
- Adopt the question title (or tighten the current one); use the proposed abstract opening.
- Signpost §4 as chronology-first / weakest stratum; cite the §3 table from §1.
- Glossary box for codenames at §6; verdict column in §8.1.
- Renumber or caption-explain §9 item ordering; replace `QUERY_SHAPE_ROUNDING` with a description.
- State the no-pooling convention once and cross-reference.
- Echo §9's closing line in the abstract or §10.
- "The facts brief this draft is bound to" (Author contributions) is internal jargon — explain in half a sentence or cut.
- The MEMENTO repetition-count parenthetical in §2 is awkwardly placed mid-sentence; move to a trailing sentence.

Nothing in this review requires touching a number, an interval, a claim boundary, or the appendices, which are strong as-is.
