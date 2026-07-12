# Final generic paper review — writing, structure, publication readiness

**Reviewer runtime identity:** `claude-fable-5` (Anthropic Claude Fable 5), as exposed by the harness environment. **Angle:** broad research-writing review — title, opening, organization, flow, proportionality, readability — not a re-audit of the numbers. The scientific floor (no practical benefit; literal hypothesis unresolved; SWE lead placebo-less; fixed-scalar CIs span zero; v12 stopped pre-treatment; e01 N=1 diagnostic-only; three head-map lineages broken) is taken as fixed and I verified the draft nowhere exceeds it.

## Overall verdict

This is a publishable, unusually honest negative-result/methods paper. The scientific discipline is exemplary: stratum separation (§3), the "strongest licensed sentence" device (§5, §8.3), the retired-claims list (§4.4), and Appendix A's token-level provenance are better than most published work. The writing problem is the inverse: the paper is so defensively caveated that its genuine contribution — the failure catalogue and the anatomy of how rigor machinery validated wrong assumptions — is buried under repeated hedging. It reads as a deposition, not a paper. The fixes below are mostly compression, sequencing, and signposting; nothing requires new analysis.

## Title and opening

**Current title** ("Transplanting KV-Cache State Across Conversation Compaction: Inconclusive Evidence and a Methodological Failure Catalogue") is accurate but limp: "Inconclusive Evidence" undersells the catalogue, which §9 itself calls the durable contribution, and the double-barrel reads as an apology. Recommended alternatives (all floor-safe):

1. **"When Rigor Validates the Wrong Assumption: A Failure Catalogue from Training-Free KV-Cache Transplantation Across Conversation Compaction"** — leads with the durable contribution and the paper's actual thesis sentence (§9's closing paragraph).
2. "Training-Free KV Transplants Across Compaction Did Not Clear the Bar: An Evaluation Record and Ten Methodological Failures."
3. Keep current title but replace "Inconclusive Evidence" with "A Negative Evaluation Record" — smallest change that removes the apologetic register.

**Abstract.** Too long (two dense paragraphs of caveat-chains) and it uses insider machinery before definition: "four evidence strata," "formal v12," "e01," "N/P schedule" mean nothing to a first-time reader. Restructure as: (1) question, (2) one-sentence answer, (3) the one surviving lead with its three missing controls, (4) the stop and the diagnostic in one sentence each, (5) the catalogue as the offered contribution. Cut the CI from the abstract or keep only "+0.013 nats/token"; the full interval belongs in §5. Delete "(§2)" cross-reference from the abstract — abstracts should be self-contained.

**§1.** Good. The blockquoted question is the right device. The "three things this paper does not claim" list is excellent and should stay — consider promoting it to the end of the abstract in compressed form.

## Organization and flow

- **§3's stratum table is load-bearing and well placed.** But its "Strongest licensed reading" column duplicates language repeated verbatim in §4–§8 and again in §10. Pick one canonical location (the table), and let §10 reference rather than restate.
- **§9 numbering is broken as presented: items appear as 1, 4, 5 / 2, 3, 6, 7, 9 / 8, 10.** The grouping-by-epistemic-status rationale is sound, but out-of-order numbers with silent gaps look like editing damage and force the reader to hunt. Either renumber within groups (M1–M3, C1–C5, O1–O2) with a mapping note, or add one sentence stating that numbers preserve the historical incident order across the three groups. **Blocking** — this is the flagship section and it currently reads as corrupted.
- **§7.2 (adverse natural calibration) is orphaned.** It appears after the stop decision but its relevance (independent adverse evidence) is stated in one compressed paragraph. One added transition sentence — why calibration is reported even though the formal arm was already terminal — would fix it.
- **§6.2's forced-carrier quotations** (the literal handoff note and anchor turn) belong in Appendix A.3; in the main text they stall the narrative at the moment the design is being motivated. Keep one clause ("a fixed experimenter-authored neutral handoff note") and move the literals.
- **§11's exploratory-agent bullet list** is the right content in the right place; the eight-requirement list is strong and concrete. Good section.
- **Appendices are excellent** and carry the METHODS-PROVENANCE-REQUIREMENTS checklist: token-level authorship (A.1–A.3), exact checkpoints and geometry (A.1/A.3), metric and alignment contracts, gates/exclusions with counts, and the reproducibility boundary (A.4). Note the checklist's embedded scientific claims (nativeness scope condition, "keys neutral," the dissociation) are stale relative to the floor; the paper correctly reports only the provenance facts and does not revive them — no change needed, but a reviewer should not "fix" the paper toward that file's claims.

## Proportionality

- §4 and §5 each spend roughly half their length on caveats, many structurally identical (no placebo, no execution, nominal intervals, provenance gaps). Consolidate the shared caveat grammar once — e.g., a short "How to read every interval in this paper" paragraph at the end of §3 (it half-exists in the statistical-conventions paragraph) — then keep only stratum-specific caveats in §4/§5. This alone could cut ~15% of the body without touching any claim.
- §8.1's full eight-row table plus §8.2's nine bullets is proportionate for the paper's only treatment observation; keep.
- The failure catalogue deserves *more* prose, not less: items 1, 7, and 9 are the memorable ones and each could carry one concrete sentence of narrative ("the check passed because…"). The closing paragraph of §9 is the best writing in the paper — consider echoing it in the abstract/conclusion.

## Readability

- Sentence length is the chronic issue: many sentences carry three qualifications and a parenthetical (e.g., §5's schedule paragraph, §4.1's lineage bullet). Split; each caveat gets its own sentence.
- Numbers are reported to inconsistent precision (+0.0135 vs +0.013 vs +0.174545 in FINDINGS-derived text). Standardize: four decimals in tables, two significant figures in prose.
- Define "nats/token" and "teacher-forced" once at first use for the systems-audience reader.
- §8.1's operator definitions (D, SEL, H+, U, U+) are correct but dense; a two-column "symbol — plain meaning" mini-table would halve reader effort.

## Blocking before repository publication

1. **§9 numbering/ordering** — renumber or explain (above).
2. **Abstract rewrite** — self-contained, jargon-deferred, catalogue-led (above).
3. **Consistency check §4.1 vs Appendix A.1 on head-map lineage wording** — both must match the audit note's disposition ("fixed evaluated interventions whose derivation profile was not preserved"); §4.1 currently says this correctly but uses "reconstruction, not a recovered derivation" while A.1 says "cannot prove exact derivation" — harmonize the phrase so external readers don't infer two different severities.
4. **Title decision** — any of the three options; the current one should not ship unmodified.
5. **Cost figure placeholder (§10)** — "will be reported separately" needs a concrete pointer (the tracked requirement artifact is already in Appendix B; cite it in §10).

## Optional polish

- Promote the §9 closing paragraph's thesis into a short standalone Conclusion section; the paper currently ends on limitations/appendices with no landing.
- §2: one sentence ordering the five related works by distance from this design would help non-specialists.
- §3's ladder and strata table could be merged visually (ladder rung ↔ stratum coverage).
- Consider a one-page "reader's map" box after the abstract: which section answers which question.
- Author-contribution phrasing "scientific decision authority" is unusual; fine if intentional, but confirm the owner wants it public.

Nothing in the draft exceeds the scientific floor; all recommended changes are editorial.
