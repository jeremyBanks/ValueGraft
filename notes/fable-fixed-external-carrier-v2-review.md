# Fable review — fixed external carrier v2 after stochastic-carrier failure

**Date:** 2026-07-12  
**Reviewer:** Claude Fable 5 through the Claude CLI.  
**Evidence supplied:** frozen v1 prompt/rules, 0/48 accepted rank-1 attempts,
no treatment exposure, and the proposed fixed-external alternative.

## Review supplied verbatim

Before answering, I pulled the actual v1 failure data: all 48 rejections are `FORBIDDEN_SURFACE` hits, and the leaked surfaces are overwhelmingly innocuous scenario nouns — "traveling collection" (23), "transfer docket" (21), "river branch" (21), "east lock" (12) — with only 3 hits on anything resembling a decision label ("inactive"/"active"). Essentially zero leakage of targets, numbers, or rule branches. That diagnosis drives several answers below.

### 1. Is the fixed-external forced carrier defensible?

Yes — and for the stated question it is *cleaner* than subject-generated carriers, not a compromise. The question is whether write-time K/V values under full history carry history-specific information **beyond identical visible text**. A subject-generated carrier never delivers identical visible text for free: you either force the C-generated text under W anyway (so generation only added a selection step), or you accept different surfaces per condition (which breaks the question). Worse, subject-generated text is *selected by the model conditioned on history*, so accepted carriers retain a residual text-choice channel that no mechanical/semantic review can provably close. Fixed external text closes that channel by construction: any C-vs-W difference in the carrier-region state must arise at write time via attention over the history.

**Exact claim limitation you must state:** the result will show only that *K/V values written for a fixed, externally authored, contextually neutral token sequence, teacher-forced under history C vs W, do (or do not) retain decodable/graft-useful history-specific information beyond the identical visible text.* It says nothing about the model's own naturally generated summaries under production compaction: forcing q=1 is off-policy, the forced tokens may have very different likelihood under C vs W, and the resulting state may be atypical of on-policy handoffs. Ecological compaction claims are out of scope for v2 and must be labeled as such in the prereg.

### 2. Two fixed texts: replicates or conditions?

They are **two fixed levels of a carrier-text factor, not random render replicates**. Three stochastic seeds were draws from a distribution and licensed a within-conversation variance estimate; two hand-authored texts license neither variance estimation nor generalization to "carriers in general" (n=2 texts, fixed effects). They are still worth having — as a robustness check that the effect is not an artifact of one surface form.

Analysis: collapse **within conversation first** (pre-specify the rule — mean of the two texts as the primary outcome is fine), then analyze at conversation level. N stays 48; the two texts must never inflate N. Report per-text results as a secondary heterogeneity check; a sign disagreement between the two texts caps the claim at "at least one fixed carrier shows the effect." One consequence you flagged around implicitly: the already-frozen 32,768 frame and outcome-blind permutation were frozen against a 3-stochastic-seed render structure. Any frozen analysis object that references seeds/render exchangeability is no longer valid as-is and must be re-derived and re-frozen under the v2 label — reusing v1-frozen artifacts is only legitimate for elements invariant to the design change.

### 3. Revise the prompt, or switch?

Switch immediately — but record the correct postmortem, because the data do not say "the model can't do this." The v1 prompt *instructed* the model to mention "shared background," and the mechanical filter forbids precisely the nouns that constitute the shared background. The 0/48 is a prompt–filter contradiction, not model incapacity; a revised prompt ("use only fully generic language; name no object, place, document, or entity from the conversation") would likely lift acceptance substantially at ~1.2 s/attempt.

The reason to switch anyway is not the failure rate. It is that (a) even a 100% acceptance rate leaves the selection channel from Q1, so subject generation answers the core question strictly less cleanly; (b) prompt iteration against an acceptance filter is exactly the apparatus-drift / forking-paths pattern you are trying to stop; and (c) it adds a semantic-review pipeline that fixed texts don't need. Subject-generated carriers, if wanted at all, belong in a later exploratory arm after the primary question is answered — not on the critical path of the remaining budget.

### 4. Minimum safeguards for v2

1. **Pre-hoc carrier freezing.** Both texts authored by an independent party (not the subject model), sha256-committed with authorship provenance *before any forward pass touches them*. Compatibility with all 80 fixtures established by the existing blind review of rendered histories, not by author say-so.
2. **Forced-encoding identity gate.** Verify the carrier's token IDs are byte-identical across C, W, and B placements (same template position), and that q=1 forcing in B reproduces fresh-encode state exactly at the production dtype/geometry — this is the L0-style positive control that the forcing apparatus itself is inert.
3. **Pre-registered collapse rule.** The within-conversation combination of the two texts (Q2) frozen before phase A; per-text results reported as secondary; N = conversations only.
4. **Report forced-carrier NLL per condition.** Per-token logprobs are already logged for free; pre-specify reporting carrier NLL under C, W, B as a manipulation/surprise-asymmetry check — report only, no gating, no post-hoc exclusion rule.
5. **No in-place patching.** If either text fails any gate for any fixture, v2 is void in its entirety and a relabeled v3 is authored. Re-freeze (under the v2 label) every analysis artifact that referenced the old 3-seed render structure.

That is the whole list. Everything else in the v1 apparatus — N48 strata, teacher-forced focal target, the A/B/sham/E_correct/E_wrong/applied-placebo battery, conversation-level UCB — carries over unchanged and should not be touched.

## Main-agent disposition

Accepted. V1 closes before treatment as a prompt/filter contradiction, not as
evidence that the subject cannot produce summaries. V2 uses two fixed external
texts as two fixed carrier conditions, averages them within conversation, and
limits the claim to off-policy forced neutral sequences. The frame, literal
permutation, base fixtures, and blind base-history reviews remain valid because
they were frozen and performed without treatment outcomes and do not depend on
carrier exchangeability.
