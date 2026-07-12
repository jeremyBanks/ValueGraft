# Skeptical peer review of PAPER.md — final research paper

**Reviewer runtime identity:** `claude-fable-5` (Anthropic Claude Fable 5). This is the exact runtime model ID exposed to this session; it is stated as such and not inferred from any alias.

**Review angle:** Fresh, hostile-but-fair methods referee. I read only `PAPER.md` (all 478 lines, including Appendices A–B and references). I did not inspect the repository, artifacts, notes archive, or any prior review; I did not browse or spawn subagents. Everything below is judged from the paper's own text and internal arithmetic.

**Date:** 2026-07-12

---

## Summary judgment

This is one of the most honestly hedged negative/inconclusive papers I have refereed. The scientific floor it claims — no practical benefit established, literal hypothesis unresolved, one control-incomplete likelihood proxy, a formal stop before eligible treatment, an N=1 placebo-missing diagnostic, ten catalogued failures — is defensible and internally consistent; the reported numbers cross-check arithmetically (I verified the pooled legacy mean, the SWE 102-row and 173-row pools, and the e01 SEL/U/U+ identities from the reported components). The remaining problems are almost all framing residue, one factual-consistency error in §8, and two feasible analyses the paper's own headline depends on but does not perform. Recommendation: **major revision, mostly textual**, with two zero-GPU analysis obligations. Nothing here requires new data collection.

---

## True blockers

**B1. The abstract violates the paper's own pooling convention.** §3 promises that fitting/evaluation/confirmation samples are "distinguished and reported separately before any explicitly descriptive pooling," yet the abstract's headline number is exactly that descriptive pool (+0.0135 [+0.0083, +0.0191] over 102 rows), presented with a CI and the advocacy phrase "strongest surviving performance lead," without the label *descriptive, nominal, pool-conditional* and without its two counterweights: (a) map-minus-fixed-graft on the fresh 57 was inconclusive (+0.0041, CI spanning zero), and (b) the schedule-matched fixed-scalar pooled and pool-contrast intervals span zero, and the realistic-summary scalar was null. A reader who stops at the abstract takes away a significantly positive result the body then spends §5 dismantling. Fix: carry the "descriptive pool" label and at least the map-vs-fixed and realistic-summary nulls into the abstract sentence; replace "lead" with "difference" or "movement" here and in §5's licensed sentence.

**B2. §8.2's "Value-only N/R2 is the one internally favorable cell" is contradicted by the paper's own table.** N/R1 full K+V satisfies the same internal criteria: D focal +0.154 dominating D nonfocal −0.022 (SEL +0.132), H+ positive (+0.023), U and U+ positive (+0.007, +0.260). If "internally favorable" is scoped to the primary region R2, say so in the sentence; as written, with the adjacent bullets sweeping R1/R3, the claim is false. Note also that acknowledging N/R1 K+V as a second favorable-looking cell *strengthens* the paper's own noise interpretation — favorable patterns appear in multiple cells with no family consistency (K+V favorable at R1, value-only favorable at R2, both families adverse elsewhere) — so the fix costs nothing scientifically. Relatedly, "the surrounding cells do not corroborate" needs the same rescoping.

**B3. The cluster-independence assumption under the headline CI is testable with preserved data and was not tested.** The bootstrap treats 491 SWE-Gym trajectories as independent "because stable task/repository cluster IDs were not saved" (A.2). But the preserved `messages` column contains the initial user task, which in SWE-Gym/OpenHands scaffolds ordinarily names the repository and file paths. SWE-Gym draws from a small number of repositories; if the ~102 pooled rows cluster into a handful of repos, a cluster-resampled CI could easily span zero, which would demote the paper's single strongest surviving number to "no detected lift." This is a zero-GPU, analysis-only obligation that does not violate the closed data-collection budget. Either recover cluster IDs from the messages and report cluster-robust intervals, or demonstrate concretely why recovery is infeasible. Until then the abstract CI is not merely "nominal," it rests on an assumption the authors could check and didn't. The nonrandom 45/75 budget-capped confirmation prefix, taken in parquet-index order, compounds this: if parquet order correlates with repo or generator model, the prefix is cluster-skewed too.

**B4. No same-condition replicate exists anywhere on the production stack, and the paper does not put this next to the e01 numbers.** A.3 records `deterministic algorithms false` on the 30B/A100 stack, and §8.3 concedes "forward-run repeatability was not tested." Meanwhile the paper's own local diagnostic (§9 item 1) shows schedule/apparatus artifacts of 0.06–0.13 nats on a 0.6B model — the same order as the celebrated N/R2 value-only movements (+0.175 D focal), with the explicit caveat that the local magnitude "is not a bound for 30B/A100." So the favorable cell's magnitude is within the plausible range of apparatus variation the project itself demonstrated, and the run-to-run noise floor under nondeterministic kernels was never measured. §8.2 gestures at schedule sensitivity (N→P collapse) but never states plainly: *the diagnostic hint is of a magnitude comparable to demonstrated apparatus artifacts, against an unmeasured replicate noise floor.* That sentence, or its equivalent, belongs in §8.2/§8.3. (The fix is textual, since rerunning is closed; but its absence is a blocker because it is the single strongest objection to the only mechanistic residue the paper retains.)

**B5. §7.1's path-control table has no units and no readout definition.** "Oriented + movement +0.5" of *what*, on *which* readout? A referee cannot audit the ULP-2-vs-ULP-4 stop decision — the pivotal event that terminated the formal experiment — without knowing what the numbers are. Define the readout and units in place.

**B6. Venue-dependent: AI systems listed as authors.** "Anthropic Claude Fable 5" and "OpenAI GPT-5.6 Sol" appear on the author line. Most journals and conferences (COPE guidance, Nature/Science policies, NeurIPS/ICML CFPs) prohibit LLM authorship; arXiv moderation is more permissive but flags it. The contributions section is exemplary and can carry the attribution; the author line will block submission at most venues. Decide per target venue before circulating.

---

## Major objections short of blockers

**M1. Split-level variation under a *single* intervention rivals the claimed effect.** From the paper's own numbers: the fixed scalar on the schedule-matched disjoint 98-pool is −0.0017; the map on the eval-57 subset is +0.0117 with map-minus-fixed +0.0041, implying fixed ≈ +0.0076 on those 57; algebra then puts fixed ≈ −0.015 on the 41 tune-assigned rows of the same pool. That is a ~0.02 nats/token swing for the *same* fixed intervention between random hash-halves of one pool — larger than the +0.0135 headline. (Derived, hence approximate; the authors can compute it exactly.) It should be reported: it is direct evidence that between-subset variance under one treatment is on par with the claimed map effect, and it notes that the map was selected on precisely the half where the fixed scalar looked worst.

**M2. Ecological validity of the SWE result is near zero, and the paper doesn't say so crisply.** The only positive likelihood movement lives under a summary request that *deliberately strips all specifics* ("Do not include specific decisions, names, numbers, or details"); the realistic/default-summary scalar was null and the map was never tested there. So the best-case number exists only in the condition engineered to maximize damage and least resembling deployed compaction. §5 lists the realistic-null in a bullet; the implication — even the strongest surviving number has no demonstrated bearing on realistic compaction — deserves a sentence of its own, in §5 and §10.

**M3. The intervention dose is never reported.** Both the legacy and SWE grafts replace only difflib-aligned value rows (matching blocks ≥ 8 tokens). Nowhere does the paper report coverage: how many rows were actually replaced per conversation/trajectory, and the summary-vs-tail split. Without dose, a reader cannot tell whether "no detected lift" reflects a substantial intervention that did nothing or a near-empty intervention that touched a handful of boilerplate rows (difflib block-matching plausibly over-selects high-frequency code/boilerplate spans, which couples directly to a mean-token-logprob metric). If aligned positions are recomputable from preserved artifacts, report coverage distributions; if not, state that the dose is unrecoverable — that is itself a provenance finding for the catalogue.

**M4. The tail-mediated mundane channel deserves a name.** Both performance strata graft summary **plus retained-tail** values. Replacing tail values means giving verbatim-shared recent text its full-history-conditioned values — essentially CacheBlend-adjacent cache reuse — which could produce the entire likelihood movement with zero summary-region contribution and zero "salvage of compacted state." §5 says the combined intervention "does not identify where the signal lives," but the specific, boring alternative (the effect is a tail-recomputation artifact, not compaction mitigation) should be stated, and a tail-only ablation named as the first decomposition behind the re-entry gate.

**M5. "Independent" overstates the e01 fixture's review lineage.** The case was "authored by an independent Codex subagent," but the provenance correction binds author *and* reviewer sessions to `gpt-5.6-sol` — the same model that holds "primary final analysis, execution, interpretation, and scientific decision authority." Independence here is session-level, not model-level: fixture authorship, blind review, treatment interpretation, and the facts brief all trace to one model family. Soften "independent" to "separate-session" and note the monoculture as a limitation; a skeptic will otherwise do it for you.

**M6. §10 lists the SWE result under "Established."** Even with the trailing qualifiers, "established" is the wrong verb for a control-incomplete, selection-exposed, nominal-CI observation. "Observed" or "recorded" preserves the floor without the tension.

**M7. Residual causal vocabulary.** Scattered instances: "likelihood effect" (abstract) for a non-randomized selected contrast; "recovered 0.81%/4.24% of that damage" (§8.2) where "offset" is safer since nothing shows restoration of content rather than generic perturbation; "moved 6.5 margin units in the source-history direction" (§7.2) is an N=1 observation presented without the caveat scaffolding e01 gets — one clause would fix it. §1's "Recent work provides evidence" rests on a single non-replicated preprint whose 64-vs-8 repetition asymmetry the paper itself discloses in §2; qualify at first mention.

---

## Missing alternative explanations (beyond those already in the text)

1. Run-to-run nondeterminism as the source of the e01 favorable cell (B4) — the paper names lexical/resolved residue and schedule geometry but never the unmeasured replicate noise floor.
2. Repository/task clustering as the source of the SWE CI's apparent precision (B3) and of the original-vs-disjoint pool contrast; also generator-model composition (GPT-4o vs Claude 3.5 rows) across index-based pools, which the paper subsumes under "composition differences" without noting stylistic attribution from preserved messages might partially recover it.
3. Tail-recomputation (CacheBlend-adjacent) as the entire performance signal (M4).
4. Difflib alignment bias toward high-frequency spans interacting with a mean-token-logprob endpoint (M3).

---

## What is unusually strong

Credit where due, because much of this is rare:

- **The retirement section (§4.4).** Explicitly listing void and retired prior headlines, with reasons, so they cannot be re-cited. Almost no papers do this.
- **The stop was honored against interest.** The prose/code conflict (§7.1) was dispositioned to the conservative literal branch even though the sealed code's rule would have passed at ULP 4, and the post-stop diagnostic was firewalled as non-authorizing. This is preregistration behaving as intended.
- **Stratification discipline.** The four-strata table with "strongest licensed reading" per stratum, the explicit anti-pooling convention, and the "licensed sentence" pattern (§4.2, §5, §8.3) are a model for inconclusive-result reporting.
- **The failure catalogue (§9)** with epistemic-status grouping (measurement hazards vs. inference failures vs. operational incidents) and the closing observation that execution-rigor machinery validated wrong estimands. Item 1 (query schedule is part of the state) is a genuinely useful hazard result independent of the transplant question.
- **The reproducibility boundary (A.4)** honestly separates "statistics recompute from committed rows" from "inference is not rerunnable," names every unrecoverable artifact, and ships zero-GPU recomputation commands.
- **Internal numerical consistency.** Every pooled figure I recomputed from reported components checked out; the e01 metric identities hold across the table.
- **The refusal to run the agent trial (§11)**, with explicit expected-information reasoning and a written re-entry standard, instead of a salvage evaluation.

---

## Optional strengthening (non-blocking)

- O1. Mark the three unauditable head-derived variants in the §4.2 table itself (dagger/footnote), not only in prose.
- O2. Label SEL in the §8.1 table as a *single-nonfocal-probe* contrast; note the FF-branch reuse (29 distinct branches for 31 cells) in the table caption.
- O3. If SWE target token-level logprobs were not preserved (A.4 suggests not), state explicitly that a token-position decomposition of the +0.013 (e.g., concentration in header/format tokens of a 600-token target) is unrecoverable.
- O4. Renumber the failure catalogue sequentially within groups or note that original numbering is preserved for cross-reference; current 1/4/5–2/3/6/7/9–8/10 reads as an editing accident.
- O5. Standardize "e01"/"E01" capitalization.
- O6. §6.2's "arm differences must be mediated by the retained state" is correctly conditioned on fixed schedule/construction, but given §9 item 1, consider adding "as computed under this replay schedule" to close the loop for readers who reach §8's schedule collapse.

---

## Verdict

The paper does not revive any stale strong claim; its floor holds. The abstract's framing of the pooled SWE number (B1), the §8 "one favorable cell" inaccuracy (B2), the untested-but-testable cluster assumption under the headline CI (B3), the missing replicate-noise framing for e01 (B4), the unitless stop-decision table (B5), and the venue authorship question (B6) must be resolved before publication. B1, B2, B4, B5 are hours of textual work; B3 is a bounded zero-GPU analysis; B6 is a decision. With those addressed, this is a publishable — and unusually instructive — negative-and-inconclusive methods paper whose real contribution, as the authors correctly judge, is the failure catalogue.
