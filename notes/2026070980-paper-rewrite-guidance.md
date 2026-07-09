# Paper rewrite guidance (owner feedback, 2026-07-09) — apply on the NEXT paper pass

The current `paper/DRAFT.md` (committed e2bb83d) is **premature and incoherent** and should be
**rewritten from scratch after the champion-validation experiment resolves**, NOT patched. It reads as
a postmortem yet also claims a meaningful result in places — that contradiction exists only because it
was written before the deciding experiment ran. Do not touch it until we have the bf16 champion-validation
result; then the paper's shape is determined:
- **Real / partial positive** (champion config beats its placebo, content-specific) → a PROPER, RIGOROUS
  ACADEMIC PAPER: the result is the focus, with an elaborate, rigorous methodology + result-analysis
  section, designed like a real academic paper. NOT a postmortem.
- **Null** → a clean bounding/negative paper (the postmortem angle can stay, but coherent, not mixed).

## Style / content fixes (owner-specified) — apply in the rewrite
1. **No internal jargon externally.** Drop `H-pack`, `arm E`, `arm A/B/C/D`, `E-post`, etc. A reader has
   no "arms A–D." Use plain, self-explanatory descriptions ("the compacted baseline," "the value graft,"
   "the tuned-config graft," "a random-value placebo graft"). Introduce any needed notation minimally and
   in externally-coherent terms.
2. **Do not over-explain H-pack.** It was never the main intent (it's packed keys+values, not the
   value-only method). At most a one-line mention as a related control; it should NOT have its own section.
3. **Stop overusing "honest"/"honestly."** It's an AI tell and over-serviced. BE honest in substance;
   don't keep *saying* the word. Remove the honesty-machinery repetition ("named so it stays dead,"
   "we report rather than hide," "standing honest-reporting notes," etc.).
4. **Remove weird/random internal-process comments** that make no sense to an external reader (repo-internal
   asides, ledger references, "the coordinator," etc.).
5. **Real academic structure + style** if there's a result — rigorous, elaborate, worth reading and even
   enjoyable; not a dry apology.

## Review process fix (owner-specified)
- The prior draft needed **more editing passes from a MORE OBJECTIVE reviewer** — one NOT loaded with our
  biased project context — checking OVERALL READABILITY, FLOW, FOCUS, and COHERENCE relative to
  **academic-paper expectations and style**. Add a dedicated fresh-eyes readability/flow/focus review
  (prompted for exactly that, minimal project bias) to the review stack, distinct from the
  adversarial-honesty and numeric-verification passes.
