# REPORT-PLAN.md — the write-up: spec, process, narrative

## Deliverable
A report between a technical blog post and an academic paper. Audience:
technically literate ML readers (not necessarily interp specialists).
Register: precise but accessible; compelling; earns trust through evidence,
not throat-clearing.

## The failure modes to actively guard against (user-specified)
1. COHERENT THROUGH-LINE — each section builds on the last; one story, not
   a pile of results.
2. RIGHT DETAIL LEVEL — enough that a skeptic can follow the reasoning and
   believe it; not so much it drowns.
3. ACCESSIBLE — a smart non-specialist can follow the arc; jargon defined
   on first use.
4. COMPELLING — the reader wants to keep going; the surprising core (sense
   lives in the cache, not the text) is foregrounded.
5. NO BIG GAPS — no unexplained leaps; caveats/limits stated, not hidden.
6. NOT OVER-JUSTIFIED (the critical one) — explain what the READER needs to
   understand and trust the result, NOT a defensive litany of every check
   we ran because we ran it. The 26 incidents are mostly PROCESS, not
   results — they belong in the report ONLY where a failure taught
   something scientific (the scope boundary, the ceiling effect). Cut
   justification that serves the conversation's history rather than the
   explanation's logic. AI over-justifies from context; the report must
   justify from necessity.

## The narrative through-line (the spine)
1. HOOK: When an AI conversation gets compacted (old turns → a summary),
   the model loses something a good summary should have preserved. What,
   exactly — and where does it live?
2. HYPOTHESIS: The lost thing is SEMANTIC continuity — the "sense" the
   model had built up — and it lives in the model's write-time KV-cache
   state, not recoverable by re-reading a text summary.
3. MECHANISM: Grafting write-time value vectors back at the compaction
   boundary recovers meaning that text compaction lost. (+ the honesty
   effect: compaction → confident fabrication; write-time state →
   appropriate uncertainty.)
4. THE REFINEMENT (the heart): It's not fact-recall, it's semantic
   richness. The dissociation — grafting recovers SENSE/REFERENT (meaning)
   but is null on STANCE (a summary already preserves preferences) — and
   recovers MEANING more than verbatim FORM (judge-metric > token-metric).
   The effect TRACKS the damage.
5. CORROBORATION: three independent measurements agree at 30B (lenient
   judge, strict judge, teacher-forced logprob); [if it lands] a
   logit-lens readout shows the concept re-appearing INSIDE the model.
6. TUNING: naive full-strength grafting can BREAK a task; per-layer tuning
   (guard-validated) fixes it — how to deploy safely.
7. HONEST BOUNDS: large-model effect (30B; does NOT hold at 4B — graft
   effects are scale-dependent). Demonstrating end-to-end agent-task
   benefit is hard: standard benchmarks miss the operative regime (too
   hard for the model, or sessions too short to force real eviction) —
   itself a finding about the benchmark landscape.
8. SO WHAT: deployment framing (opaque compaction handle); what a
   purpose-built benchmark would need; the K-vs-V and cross-scale open
   questions.

## Process (multi-pass, agent-assisted)
P0. Assemble source-of-truth: FINDINGS.md is canonical; every number in the
    report traces to it. (Do NOT invent/round beyond the data.)
P1. OUTLINE against the through-line above; get section-level logic right
    before prose.
P2. DRAFT section by section.
P3. ADVERSARIAL REVIEW — one agent PER failure mode, each ONLY hunting its
    dimension: (a) through-line/coherence, (b) gaps/unexplained-leaps,
    (c) OVER-JUSTIFICATION cutter (most important — flag every sentence
    that justifies from habit not necessity), (d) accessibility, (e)
    accuracy/honesty vs FINDINGS (no overclaim, caveats present).
P4. SYNTHESIZE revisions from the critics; repeat P3 on hot spots until
    clean.
P5. Final read for voice/compellingness.

## Kickoff
AFTER concept-readout lands (it's a load-bearing corroboration or an honest
"lens didn't show it" — either way the report needs to know). Then run P0→P5.
Provenance footer: Fable 5 (majority) + Opus 4.8 (continuation) + Sonnet 5
(subagents) + GPT-5.5 (adversarial review), directed & funded by the user.
See PROVENANCE-CORRECTION.md.
