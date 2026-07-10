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

## Naming / terminology (owner, 2026-07-09, refined) — coin ONLY if EARNED
- **The rule: don't coin a term unless you've earned it.** It's CONDITIONAL on the final result:
  - **IF the earned result is a real EFFECT that amounts to a TECHNIQUE someone might actually want to
    reuse / apply elsewhere** → we MAY use our camelCase coined term (e.g. "ValueGraft") as persistent
    vocabulary, potentially including the title. A real reusable technique earns a name.
  - **IF we're mostly reporting FAILURES** (null/indirect, or too narrow/small to recommend as a method) →
    use plain **sentence-case, ordinary descriptive vocabulary**; NO coinage, NO camelCase brand. Define
    the local label once in a glossary as "the term we use here," not a minted persistent term.
- Coining a name for a technique that hasn't earned it just **litters the namespace** — irresponsible.
- **Judgment call for the write-up (Fable decides at valuation time):** assess the FINAL earned effect.
  Current data = a real but small, narrow, compression-dependent effect (helps under aggressive compaction,
  null under realistic summaries; teacher-forced-logprob proxy). Whether that rises to "a technique worth
  naming" is a genuine call — **lean toward NOT coining unless the effect is clearly a reusable technique;
  when in doubt, sentence-case descriptive.** The decision follows the evidence, not the other way around.

## Scope, tone, and structure (owner, 2026-07-09) — real but NARROW; do NOT oversell
- **Do NOT oversell / over-generalize.** What we have DIRECTLY demonstrated is EXTREMELY NARROW: one model
  family (bf16 Qwen3-30B-A3B), a teacher-forced-logprob PROXY (not resolve-rate / not downstream task
  success), a specific corpus, and the effect appears only in the aggressive-compaction regime. Do NOT
  assert or imply the effect applies beyond exactly what was tested. Every claim bounded to its measured
  conditions; no "this suggests broadly that..." leaps.
- **But it IS a real effect — don't undersell it to nothing either.** We have real, statistically-supported
  evidence of a real (if small) effect: the graft helps under aggressive/brief compaction (SWE-Gym brief
  +0.0156 CI[+0.0047,+0.0266]; prod null +0.0006 CI spans 0 → compression-dependent). Present it as a
  genuine, modest, narrow positive — "stupid and small" but real. Honest calibration, both directions.
- **Structure = great science FIRST, elaborate postmortem AT THE BOTTOM.** The main body is a rigorous,
  well-crafted scientific paper: exactly what we tested, the controls (placebo battery, held-out,
  born-annotated provenance), and what we learned (the compression-dependence result; the champion null
  under realistic summaries). THEN, toward the end, an ELABORATE POSTMORTEM section (what went sideways,
  the provenance/measurement lessons, the quick-win/lever-pull failure modes, what we'd do differently).
  Ordering: mostly paper-parts first, mostly postmortem at the bottom — the science stands on its own; the
  postmortem is the reflective tail, not interleaved through the results.

## Methods detail / conceptual reproducibility (owner, 2026-07-09) — the BIG past failure
- Past papers were **far, far too thin** on methods and specific implementation/design choices — "so far
  away it's not even funny." This must change dramatically.
- **The aim: a reader could CONCEPTUALLY replicate our results from the paper ALONE** — not by copying the
  code, but because we explained, in high conceptual detail, exactly what we did and why. Every non-obvious
  design/implementation choice spelled out: the value-graft mechanism (value-only, α_K=0, aligned
  positions, per-layer alpha map), the alignment procedure (span-offset map), the placebo battery
  (shuffle_pos / shuffle_probe / gauss[energy-matched] / mean — how each corrupts the source), the
  self-gen summary condition (and WHY fixed summaries suppress the graft), the metric definitions
  (teacher-forced logprob = a proxy, raw_EB primary, content-specificity E−placebo), the held-out split
  (tune c01-06+n01-04 → validate c07-24), the conversation-clustered bootstrap CI, the champion tuning,
  and the compression levels (with measured ratios). A motivated reader reconstructs the experiment
  conceptually.
- **Calibration:** full conceptual-replication may be a slight overstatement / we may fall a bit short for
  practical reasons — that's OK. But the DIRECTION must be unmistakable: err heavily toward MORE methods
  detail, per-choice justification, and explicit definitions than feels necessary. This is the level the
  paper aims for. (Aligns with METHODS-PROVENANCE-REQUIREMENTS.md / task #38.)
- Match detail to importance: deep on methods + the choices that affect the result; don't pad the parts
  that don't matter. Right level of detail in the right places, heavy where it enables replication.
