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

## Paper-WRITING process (owner, reinforced 2026-07-09)
- **FABLE WRITES THE PAPER**, heavily involved — quite possibly the sole author of the prose. I (Claude)
  provide Fable ALL the information, pointers, guidance, and a *suggestion* for framing — but give it
  **FULL LATITUDE on how to frame it in every aspect**; it has full freedom to write whatever it wants.
- **Iterate many times for COHERENCE** — make sure the paper explains itself and hangs together; re-pass
  repeatedly, don't one-shot.
- **Multiple distinct Fable review roles at the END**: an adversarial pure reviewer, a proofreader, a
  readability/flow/focus reviewer, etc. — lots of separate passes, each with a single clear lens.

## Naming / terminology (owner, 2026-07-09) — DO NOT coin a term
- **Do NOT coin a new term** (like "ValueGraft") in the paper — especially NOT in the title, and NOT as
  persistent vocabulary. The technique does not robustly work (it's null on the primary metric; effect is
  indirect at best), and we are NOT telling anyone to do it. Coining a capitalized/camelCase name for a
  technique that doesn't work just **litters the namespace** — it's irresponsible to mint persistent
  vocabulary for a non-result.
- Internally we can keep using a term for convenience, but in the paper: use a **plain, lowercase,
  descriptive** phrase (NOT capitalized, NOT camelCase, not a brand). Define it ONCE in a **glossary** as
  *"the term we use in this paper to refer to X"* — explicitly a local convenience label, NOT a coinage.
- Rationale: we report a bounding/negative (or at best indirect) result; the framing must not smuggle in
  the implication that this is a named, recommended method. The name should read as descriptive, not
  aspirational-branding.
