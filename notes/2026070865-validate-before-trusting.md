---
name: validate-before-trusting
description: "Validate a measurement pipeline with a POSITIVE control before trusting it; mine on-disk data before spending compute; isolate variables; verify by reading code/data, don't narrate a plausible-but-wrong story"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bda7fb9f-f447-4890-904b-dde750ff3370
---

Hard-won during the ValueGraft cross-architecture work (07-07/08), where a chain of
premature conclusions each got caught — several by the user, not me.

**POSITIVE control before trusting a new pipeline.** A new harness that AGREES with
trusted code on a NEGATIVE case is NOT validated. The cross-arch harness matched the
trusted apparatus on Qwen2.5 (both negative) — but when finally run as a POSITIVE
control (reproduce a known +0.156), it gave null/negative. Negative-agreement was
worthless; the positive control caught that the fixed-summary design silently
suppressed the effect. ALWAYS reproduce a KNOWN POSITIVE result through a new
pipeline before trusting ANY of its numbers.

**Mine on-disk data before spending compute.** When the "F1 result doesn't reproduce"
scare hit, I reached for GPU re-runs and MoE-nondeterminism theories. Fable solved it
in zero GPU by comparing the TWO runs already on disk per-probe: the effect was stable,
only the mean-of-ratio ESTIMATOR was unstable. Check the data you already have first.

**Don't conclude from one datapoint or one plausible story — isolate variables.**
Every real answer came from a control that isolated ONE variable: fixed-vs-selfgen
summary (isolates the summary), trusted-gap_closure_cat-vs-harness (isolates the
harness), difflib-vs-direct alignment equivalence (isolates the alignment). One model
is never a verdict.

**Verify by READING code/data, not narrating.** I twice told a confident, plausible,
WRONG story: "MoE nondeterminism" (it was a bad estimator) and "spurious token matching
to the conversation" (the alignment is positional-within-region; I hadn't read the
function). The user's blunt pushes — "it seems like bugs," "why are we doing token
matching," "did we run a positive control" — repeatedly beat my plausible-sounding
guesses. Default to MORE skepticism of my own narration; read the function, run the
diagnostic, before explaining.

**CONSULT WHEN STUCK — Fable for DIAGNOSIS, not implementation.** When a result is
confusing or a fix isn't converging, reach for Fable's conceptual reasoning EARLY rather
than grinding solo or over-trusting a subagent's "status=OK". Fable helps figure out WHAT
is wrong (mechanism, hypotheses, the decisive test); it should NOT implement. The owner has
repeatedly had to remind me to consult. EARLY trigger (07-08, cost HOURS): the moment a
fix has NOT converged in ~1-2 attempts, OR a subagent is looping on the same failure, OR a
result is confusing — STOP and consult Fable for the STRATEGIC view BEFORE grinding further
or letting a subagent grind. Do not wait for hours of narrow technical debugging. Fable pulls
you up to "verify the boring things" (exact model, exact config) that deep debugging misses —
exactly the wrong-model checkpoint that a 2-hour saga should have caught in 5 minutes — treat being stuck as the trigger, not the last resort.

**Verify the EXACT model checkpoint before debugging a non-reproduction.** I lost HOURS
(alignment crashes, a difflib revert, a think-strip band-aid) debugging why a known +0.156
would not reproduce — when the real cause was running Qwen3-30B-A3B (a THINKING model, emits
<think>) instead of Qwen3-30B-A3B-Instruct-2507 (the NON-thinking checkpoint the +0.156 was
measured on, the harness default I had overridden). Same family, different checkpoint = the
whole saga. When a known-good result does not reproduce, FIRST confirm you are running the
IDENTICAL model id + config as the known-good run — not just the same family/size. Diff the
exact repo id against the one that produced the number BEFORE touching any other code.

**A broken result is often a richer result.** The fixed-summary failure pointed at a
mechanistic insight (the graft may need the model's OWN generated summary, not a foreign
one). Controls that falsify your method are how you find the real mechanism.


**FAIL FAST — push every check to the cheapest, earliest layer.** Incident 32: a
tokenization-only alignment bug took ~30 min + a model download + pod spend to surface,
when it needed NEITHER GPU NOR model — a CPU/tokenizer-only smoke would have caught it in
seconds, locally, before any spend. Order checks by cost: (1) CPU/tokenizer-only smokes
(pure-data logic: alignment, span layout, template rendering) run BEFORE any pod; (2) a
TINY-model (~0.6B, loads in seconds) full-path smoke BEFORE the big model; (3) the big-model
positive control last. Use REPRESENTATIVE inputs in the cheap smoke (the self-gen summary
was LONG ~900 tok — a short fixed summary would NOT have reproduced it; test the hard case).
A pre-flight that exits nonzero on misalignment gates every launch.

Related: [[quiet-monitors]] (launch-verify real work), [[model-selection-by-fitness]]
(Fable for clear-headed conceptual gut-checks).
