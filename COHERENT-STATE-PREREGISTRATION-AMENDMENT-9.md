# Coherent summary state: Amendment 9 — source-derived semantic evidence and stopping closure

**Frozen:** 2026-07-11, after independent code and scientific reviews of the
unexecuted Amendment-8 apparatus, and before any v9 30B technical or semantic
outcome. This amendment is additive. It changes no arm, tolerance, case order,
donor map, estimand, stopping rule, or claim scope.

## Why this amendment is necessary

Adversarial v8 fixtures demonstrated that semantic harvest could accept
arithmetically self-consistent but scientifically false evidence: fabricated
plant identities, probes, target texts and token IDs; an arbitrary destination
tail; an unvalidated actual-render `G_wrong` source; arithmetic-only calibration;
and a terminal COMPLETE run with no independently required analysis or stopping
decision. Snapshot digests also lacked explicit dtype and shape metadata, which
made the Amendment-8 raw-tensor waiver insufficiently self-describing.

These are independent-evidence failures, not changes to the planned experiment.
They are closed below before any paid outcome is observed.

## 1. Version identity

Artifacts governed by all nine additive amendments use exactly:

- `amendment_id = COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4-5-6-7-8-9`
- `design_id = coherent-state-gapped-v9`
- `schema = 2`

No v8 donor artifact, ladder, technical result, harvest attestation, semantic
checkpoint, or analysis can authorize v9.

## 2. Main semantic outcomes are source-derived

For every scored conversation, independent harvest must derive the first
referent and sense plants from committed `data/scenarios.json`, preserving exact
plant IDs, categories, probes, and order. It must bind the correct and
counterfactual texts to committed `data/coherent_state_targets.json`, render the
exact production-tokenizer target IDs, and validate the probe suffix, tokenwise
log-probability coverage, and every logical and physical scoring position.

The validator reconstructs the complete compacted destination after the summary,
including the assistant close and retained tail, full messages, context token
IDs, gapped logical positions, and contiguous physical cache positions. It also
reconstructs the full-history `A_full` messages/context and every scoring
continuation. Arithmetic agreement without these exact source bindings fails.

## 3. The actual-render wrong-history control is reconstructed

The static donor battery remains necessary but cannot substitute for validating
the decision-bearing semantic `G_wrong` source. For each freshly rendered case,
harvest independently rebuilds the frozen external-donor substitution from that
target conversation and donor file. It requires exact correct and wrong prefix
IDs, changed/structural/content partitions, replacement pools and cycling,
special-token exclusion, and prefix hashes. The stored wrong-source summary trace
must cover the exact saved summary, and its mean NLL is recomputed from the
tokenwise log-probabilities.

## 4. Calibration receives full semantic provenance

Calibration affects sensitivity interpretation and the six-to-twelve futility
decision, so arithmetic-only validation is prohibited. Harvest reconstructs the
deterministic A/B label assignment; correct, wrong, and fresh prefixes; frozen
ambiguous summary; compacted messages and layout; probes; target texts and
production-rendered IDs; logical and physical positions; and all three arms'
source-to-inserted K/V lineage. Missing label variants, fabricated targets,
wrong positions, or incomplete intervention lineage fail the checkpoint.

## 5. Snapshot witnesses are self-describing

Every retained K/V digest records the exact tensor dtype and shape alongside the
hash. Independent harvest requires 48 ordered layers, `torch.bfloat16`, and shape
`[1, 4, T, 128]`, with the exact expected summary length `T`, for actual, replay,
fresh, wrong, declared-source, and inserted spans. The hash definition and
non-reconstructive relational-witness limitation remain exactly as frozen in
Amendment 8. A digest without matching geometry cannot participate in the raw-
tensor archival waiver.

## 6. Terminal analysis and serial stopping are independently recomputed

A terminal semantic COMPLETE result must contain the frozen analysis artifacts,
not merely scored checkpoints:

- at N=6, exactly `analysis_n06.json`;
- at N=12, both the immutable N=6 analysis and `analysis_n12.json`, with the N=6
  decision equal to `EXTEND_TO_12`.

From the validated checkpoints, independent harvest recomputes all five
conversation-level contrast rows; Student-t intervals; deterministic 10,000-
sample, seed-20260711 conversation-bootstrap intervals; the technical gate; the
first-six regime gate; both unique calibration variants; the frozen six-row
futility rule; the serial decision; and the interpretation. Every analysis field
must match exactly. The manifest's conversation count, serial decision, and
interpretation must be exact copies of the appropriate terminal analysis.

An N=6 run may be COMPLETE only with its rule-consistent terminal decision. An
N=12 run without an immutable N=6 extension decision, or any missing/tampered
analysis, fails harvest and authorizes no claim.

## 7. Required adversarial regressions

Before release, tests must reject at least:

- fabricated plant, category, probe, target text, token ID, or scoring position;
- arbitrary or truncated destination tail and `A_full` context;
- fabricated wrong source, donor replacement, or wrong-summary NLL;
- arithmetic-only or wrongly labeled calibration evidence;
- snapshot rows with false/missing dtype or shape;
- missing or tampered N=6/N=12 analysis;
- a manifest decision or interpretation that differs from recomputation; and
- N=12 without an exact N=6 `EXTEND_TO_12` decision.

## 8. Release gate

Before any paid v9 attempt: generate and commit a fresh v9 production-tokenizer
donor artifact; complete and commit a fresh v9 0.6B bf16 eager CPU ladder; run
the full unit and monitor fault-injection suites; obtain fresh independent code,
science, and cross-family reviews on the exact clean launch commit; and push that
commit. The first paid run remains technical-only. Semantic execution remains
separately conditional on a committed, independently harvest-valid technical
PASS with unchanged apparatus bytes.
