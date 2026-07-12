# P02 result interpretation and paper disposition — independent review

**Author (exact runtime identity):** Claude Fable 5, model id `claude-fable-5`.
**Date:** 2026-07-12
**Role:** independent senior scientific advisor; advisory only. Sol retains final authority.
**Inputs verified directly this session:** `PRECISION-PROBE-P02-PREREGISTRATION.md`; notes A6, A7, A8; the P01 partial-descriptive analysis JSON and the P02 independent-analysis JSON (both interrogated field-by-field, not taken on trust). **Gap disclosed:** session budget was exhausted on artifact verification before a full re-read of FINDINGS.md, the 2026071290 writing brief, and PAPER.md; paper-disposition advice below is anchored on the preregistration, A6–A8, and the artifacts themselves. If the brief's precision-axis paragraph deviates from A6's characterization, reconcile toward the artifacts.

## 1. Verified result (from the artifacts, not from prior notes)

From `results/precision_probe_p02_analysis/precision-probe-p02-independent-analysis_Qwen3-30B-A3B-Instruct-2507_20260712T114104839688Z.json`:

- **Terminal PASS, full design complete.** `runner_terminal_status = COMPLETE`; outer pod receipt `terminal_status = PASS` (110 artifacts, expected commit `a40b1df…`); 42 lossless packages VERIFIED; all four outcome packages (`nf4:e01:r1/r2`, `bf16:e01:r1/r2`) VERIFIED with compact-consistency and render-consistency PASS; continuation decisions PASS with initial schedule `MATCHED_R1_AND_BOTH_R2`; `repeat2_rider_status = MATCHED_COMPLETE`. P02 therefore completed the full 2×2 that P01's cap interrupted.
- **Exact within-regime repeat identity.** NF4 r1 and r2 share canonical sha256 `c5757029…`; bf16 r1 and r2 share `27553188…`; `differing_field_count = 0` in both regimes; every scalar repeat-2-minus-repeat-1 delta is exactly 0.0; generation hashes/text stable; `E2_E3_blocked_by_generation_or_hash_instability = false`.
- **Distinct hardware, exact value concordance with P01.** P02 host: `GPU-8a42830e-…`, driver 580.159.03. P01 host: `GPU-470c3e18-…`, driver 580.159.04. Different GPU UUIDs and driver patch versions. Every matched-repeat-1 scalar I compared is identical to the P01 anchor at full float precision: E1 focal margin damage 22.100759506225586 (NF4) / 23.65581703186035 (bf16); N/R2 `D^V` 0.10310935974121094 / 0.03940010070800781; N/R2 `D^KV` −0.34577369689941406 / +0.23293495178222656; P/R2 `D^V` 0.03283882141113281 / 0.012104034423828125; nonfocal `D^V` 0.16139435768127441 / 0.041535139083862305; E1 nonfocal 14.532580137252808 / 15.45745587348938; focal-minus-nonfocal 7.568179368972778 / 8.198361158370972; N-minus-P shifts likewise.
- **E2 literal behavior reproduced exactly.** NF4 focal tuple `[‘Atlas 4.8 was not selected under the recorded mandatory selection rule.’, ‘Ring 3’, ‘Ring 3’, ‘Ring 3’, ‘Ring 3’]` (change vector all-true); bf16 focal tuple `[‘Ring 3’ ×5]` (no graft changes fresh). Nonfocal tuples: `‘30 days’ ×5` in both regimes, tuples and change vectors equal across regimes. All stops `model_eos`. No cell ever generates the correct target `partner beta`.
- **Placebo unavailable again, same geometry.** `PLACEBO_UNAVAILABLE` in both regimes, first failure at layer 1, fresh-destination row 76 (NF4) / 77 (bf16) — identical to P01. Diagnostics: `zero_delta_count = 67` of 68–69 completed rows, i.e., the construction saturates on degenerate (zero-delta) rows and fails at the first hard row. NF4 `max_applied_relative_norm_error` 0.0379 vs bf16 0.0039.
- **Analyzer guardrails engaged.** No CIs, p-values, or ratios computed; `quantization_dependence_claim_authorized = false`; `population_interaction_claim_authorized = false`; `semantic_evidence_eligible = false`; `formal_v12_decision_eligible = false`; fixed-fixture exploratory flag set.
- **One caveat on “exact.”** P02's canonical payload hashes differ from P01's (`cb12664c…` does not appear in the P02 file). This is expected — P01 packages canonicalize the full 34-arm grid, P02 the targeted set — so cross-run identity is established value-by-value on the common cells (all scalars, tuples, placebo geometry I checked), not by whole-payload byte equality. The paper should state the hash-inequality reason preemptively so a reviewer does not misread it as instability.

## 2. Every A8 frozen expectation, scored

1. Gross damage large and positive in both regimes — **confirmed, bit-exactly** (22.10 / 23.66).
2. Placebo construction unavailable — **confirmed**, at the same layer-1 rows 76/77.
3. Graft cells still fail to generate `partner beta` — **confirmed** (all outputs Ring 3 or the Atlas sentence).
4. NF4 fresh anomaly (A8 assigned this lower confidence) — **reproduced exactly**, token-for-token.
5. `D^KV` sign split (also lower confidence) — **reproduced exactly** (−0.3458 vs +0.2329).
6. Exact five-cell literal tuples in both regimes — **reproduced**; A8's predeclared reading "cross-host fixed-fixture behavioral concordance" applies.
7. No within-P02 repeat instability — **confirmed at exactly 0.0** with byte-identical repeat packages.
8. Predeclared limits (no semantic transfer, no quantization-dependence license, descriptive E3 only) — **all hold**; the analyzer enforces them mechanically.

A8's modal expectation was fully realized, including both items it flagged as most likely to fail. One thing A8 did not anticipate framing: it posed reproduce-vs-not, whereas the outcome was reproduce-*bit-exactly* — which changes what the replication means (next section).

## 3. What this establishes — and pointedly does not

**Establishes (deterministic fixed-fixture reproduction).** The entire matched observation — gross damage, tiny source contrasts, anti-selectivity, the NF4-only fresh anomaly and its flip-by-any-graft, even the placebo construction failure rows — is a deterministic property of (checkpoint bytes, code, fixture, pinned stack), invariant across two physical A100-80GB hosts with different GPU UUIDs and driver patch versions. This answers the highest-value question A7 identified: the NF4 anomaly is **not** host- or instance-idiosyncratic. It also demonstrates the pipeline is bitwise reproducible end-to-end — a genuinely strong methods/reproducibility statement few LLM-intervention papers can make.

**Does not establish (run noise, sample size).** Because reproduction was exact, P02 contributes **zero independent statistical information** about the phenomenon. It is a *reproduction* (portability of a deterministic computation), not a *replication with independent randomness*. n remains 1 fixture × 1 computation per regime. Corollary: the §6 repeat guard is vacuous as a noise yardstick here — with within-regime repeat delta exactly 0.0, *any* nonzero cross-regime delta trivially "exceeds repeat noise." `numerical_interpretation_permitted_by_repeat_guard = true` means only "not blocked by instability"; the paper must never cite it as "difference exceeds noise" in a statistical sense. The relevant variability for any effect claim is cross-fixture, which is unmeasured. Also do not generalize the determinism itself beyond what was sampled: same GPU model, same driver major (580.159.x), eager attention, greedy decoding. A different SKU or driver major could legitimately produce different bits.

**Bundled axis.** NF4-vs-bf16 remains a bundle of weight representation *and* linear kernels; cross-host concordance does nothing to unbundle it. "Quantization dependence" stays unlicensed — the descriptively conspicuous `D^KV` sign flip is now a *stable* single-fixture curiosity, still uncontrolled and still paired with identical wrong generated answers.

**Semantic attribution and efficacy.** Unchanged and negative-leaning: correct-vs-wrong source contrasts (~0.03–0.16) are two orders of magnitude below gross damage (~22–24); nonfocal `D^V` exceeds focal `D^V` in both regimes (anti-selectivity); every graft cell yields the same wrong answer; and the E1 damage decomposition shows the correct target's own log-probability collapses by ~22.8 (bf16) / ~24.0 (NF4) — the correct answer is destroyed, not merely outranked. With the placebo unavailable, semantic state transfer cannot be separated from generic intervention effects even in principle on this data. No efficacy, no semantic recovery, no population or agent-level claim of any kind.

**The NF4 flip, best available interpretation.** The most parsimonious reading of "any graft (correct, wrong, or fresh-key pairing) flips NF4 fresh from the Atlas sentence to the bf16-consensus Ring 3" is a knife-edge logit balance: NF4's fresh state sits near a decision boundary that bf16's does not, so any R2-row perturbation tips it into the basin bf16 occupies robustly. That is *generic disruption of a near-tie*, the opposite of semantic recovery.

## 4. Headline and paper disposition

**No headline changes.** P02 changes the *epistemic status* of the precision-axis paragraph, not its content: from "observed once, post hoc, in an interrupted run" to "prospectively preregistered, independently re-executed on a second host, reproduced exactly, with the full 2×2 completing." That is precisely the upgrade A7 said a screening paper needs.

**How P01 and P02 enter the paper:**
- P01 = discovery run: post-hoc partial descriptive analysis of a cap-interrupted run (bf16 r2 absent, `KeyboardInterrupt`), matched repeat-1 pair, labeled post hoc.
- P02 = prospectively preregistered **conditional** replication — frozen before any P01 value was opened, launched after P01 looked interesting (one-clause disclosure, per A7 §4) — full 2×2, terminal PASS, exact reproduction on distinct hardware.
- Present **one** table of matched values with an explicit note that P01 and P02 values are identical at full float precision; two tables would imply independent estimates that do not exist.
- Keep "PASS" strictly as an operations/terminal status word, never as a scientific verdict — "the replication passed" must not read as "the hypothesis was confirmed."

**Strongest defensible language (suggested):** "On this one engineered fixture, compaction destroys the planted fact in both matched weight/runtime regimes (≈22–24 log-probability units, dominated by collapse of the correct target itself); no graft condition recovers it; correct- and wrong-source grafts are nearly indistinguishable and produce identical wrong answers; and the one runtime-specific signature — an NF4-only wrong fresh answer that any cache graft flips to the bf16 consensus — reproduces bit-for-bit across two hosts with different GPUs and driver versions under a prospectively frozen protocol. Because reproduction is exact and the placebo control could not be constructed, these are deterministic fixed-fixture observations: they license no semantic-transfer, efficacy, quantization-dependence, or population claim." Also include one sentence of the form: "the exactness of the reproduction means P02 tests protocol integrity and hardware/driver invariance, not sampling variability; it does not increase n."

## 5. Further paid work?

**No.** There is no scientific reason for another paid run of this protocol: determinism guarantees a third execution returns the same bits (information value ≈ 0), and the frozen no-value-authorizes-more-work rule independently forbids it. The only genuinely informative follow-ups — multi-fixture runs (the sole route to real variability and generality), a redesigned control family that can actually construct on this geometry, or an unbundling design (e.g., dequantized-NF4 weights in bf16 linears) — each require a new frozen preregistration and none is needed for the current paper. Recommendation: zero further spend for this paper.

## 6. Checks Sol may be missing (all local, free, descriptive)

1. **Fresh-state logit-gap extraction.** From already-committed raw packages, extract the NF4 vs bf16 fresh-state first-token log-probability gap between the ‘Ring'-basin and ‘Atlas'-basin continuations. If NF4's gap is near zero and bf16's is large, the "generic disruption" sentence gains a clean mechanistic footnote (near-tie instability) at zero cost. If the saved logits don't support it, drop it — no new run.
2. **Common-cell payload diff.** Normalize P01 and P02 packages to the common cell set and diff all fields (including per-token contribution vectors), converting my value-level verification of exact concordance into whole-common-payload verification, and record why the canonical hashes differ (different arm grids).
3. **Placebo failure framing.** Report the failure as structural (67 of ~68 rows are zero-delta degenerate; first hard row fails bf16-validity in both regimes), not as bad luck — this justifies "a redesigned control is a design change," and the asymmetric norm-error diagnostics (0.038 NF4 vs 0.004 bf16) merit one descriptive clause at most.
4. **Repeat-guard wording audit.** Scan the paper draft for any sentence that uses the within-regime repeat identity or the guard's `true` as statistical support for the NF4−bf16 deltas; rewrite any such sentence to the determinism framing.

**Bottom line:** P02 is a clean, complete, terminal PASS whose exact bit-level concordance with P01 across distinct hardware upgrades the precision-axis paragraph's provenance from post-hoc description to prospectively preregistered deterministic reproduction — while adding no statistical strength, licensing no new claims, and closing the question of host-idiosyncrasy. Publish it that way; spend nothing more on it.
