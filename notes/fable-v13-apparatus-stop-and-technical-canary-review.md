# Independent Scientific-Program Review: K/V Write-Time History Signal Campaign

**Reviewer runtime identity:** `claude-fable-5` (Claude Fable 5, Anthropic). This review uses only the facts supplied in the request bundle; no tools were run and no repository state was inspected.

**Date:** 2026-07-12
**Balance at review:** $57.1287946692, no active pod, no top-up assumed.

---

## Verdict in one paragraph

The successor is not yet repeating the apparatus spiral, but is one session away from it. The defects found so far were real and validity-critical, so the ~2 unpaid hours were well spent; however, the gate list is now growing faster than it is closing, and several gates are physically unclosable without a GPU — a deadlock structure that is the signature of the previous failure. I recommend amending the draft **before any outcomes exist** to authorize a ≤$1.50 engineering-only canary, because the single largest threat to reaching N=48 is not apparatus validity but an unverified per-case cost model, and no amount of static work can measure that.

---

## 1. Apparatus spiral or information-critical foundation?

**Observed facts.** Independent reviews found fail-open defects (forged rank receipts, incomplete tokenizer binding and surface filters, generic release inventory, CPU bundle execution, arbitrary VP payloads) before freeze. The 200k finite-population validation passed with an independent implementation. No population bound has yet been produced by anyone across two agents. The "still unimplemented" list has roughly as many items as the "completed" list.

**Inference.** The work done so far passes the information-criticality test: each repaired defect was fail-open, meaning it would have silently invalidated paid data rather than crashing. That is exactly the class of defect worth catching unpaid. But two warning signs of a re-forming spiral are present:

1. The gate list is expanding (the draft now requires manifests, ladder tests, rejection tests, warm-order tests, *and* a review pass) rather than converging to a fixed enumerated set.
2. Some gates ("exact-runtime gates cannot truly be demonstrated without a GPU") block the only action that could close them. A preregistration that says "no paid work before all gates" plus "some gates need paid hardware" is a deadlock, not rigor.

**Conclusion:** foundation work to date — justified. The current *sequencing rule* — already pathological. Fix the rule, not the work ethic.

## 2. Minimum defensible unpaid gate set before the first capped paid technical canary

The discriminating question for each gate is: *does its absence corrupt semantic outcomes, or merely engineering measurements that will be discarded anyway?* A technical canary with hardcoded e01/technical-long, semantic N=0, and no in-pool text can only be contaminated through (a) wrong subject, (b) uncontrolled spend, or (c) accidental exposure of pool content.

**Must close before any paid minute (canary-blocking):**

- Exact subject loader and attestation (Qwen3-30B-A3B-Instruct-2507, bf16, eager). Without this, canary timing numbers describe the wrong subject and the money is wasted.
- Hard-kill / timeout / cost-cap runner control. This is the budget's only physical protection; a hung eager-attention prefill on a metered pod is the most likely way to burn the reserve.
- Pinned release commit for the canary code path (the existing Stage-A verifier suffices if it can pin; a full rejection-test matrix does not need to precede an engineering run).
- Deterministic bf16 V-row placebo path exercised (already implemented per the bundle — just confirm it is what ships).

**Deferrable until after measured e01/long timing, without semantic contamination:**

- Complete permutation and ranked content manifests, and the ranked first-ten mechanical/content review — these gate *candidate selection*, which the canary does not perform.
- Sampler q1 identity and sampled/forced/full-plan identity — gates in-pool generation, not the hardcoded fixture.
- Bundle/store independent second passes — required before any data enters confirmatory statistics, not before data that is preregistered as excluded.
- Build ladder, path control, warm-order tests — the canary itself *is* the first warm-order and path-control measurement.
- Phase-A/treatment physical separation — no treatment material exists in a canary.
- Real Stage-A inventory and full verifier rejection tests — required before Stage-A authorization, not before an engineering probe.

The contamination logic holds only if the canary touches zero pool text, requests no real entropy, and its outputs are preregistered as excluded from every confirmatory statistic. All three are already consistent with the proposed design.

## 3. Amend for a ≤$1.50 technical canary now, or finish static gates first?

**Recommendation: amend now. [Requires a preregistration amendment — status-only, before any outcomes exist.]**

**Scientific tradeoff.** The canary produces no semantic information, so it cannot bias the headline UCBs provided its exclusion is preregistered and no ranked/in-pool material is loaded. The amendment must be written before the canary runs (currently true — no outcomes exist), state semantic N=0, hardcode e01/technical-long, and bar entropy requests and pool inspection. Done this way, it is scientifically free.

**Budget tradeoff.** This is where the case is decisive. $1.50 is 2.6% of the balance, and it buys the one number the entire campaign's feasibility rests on: measured cold-start-to-first-token and per-case wall time for a 4.5k-token context under bf16/eager on a 30B MoE. Rough arithmetic: $30 treatment cap at ~$1.60–1.90/hr A100 buys ~16–18 GPU-hours for 96 treatment renders — roughly 10–11 minutes per case *including* amortized setup. Whether that is comfortable or catastrophic depends entirely on eager-attention prefill and VP overhead at 4.5k tokens, which no static test can estimate. Discovering a 2× cost miss inside Stage A (after $12) leaves no recovery room; discovering it for $1.50 leaves $55.63 and time to redesign (fewer nested renders, shorter carriers, or flash-attention justification). Charge the canary against the $8 reserve or the Stage-A $12, **not** the $2.63 slack — slack at $1.13 is functionally zero.

Finishing all static gates first has no offsetting benefit: the deferred gates protect semantic validity, which the canary does not touch, and several gates cannot close without exactly this run.

## 4. STOP/CONTINUE rule and next five actions

**Rule (proposed, mechanical):** Apparatus work CONTINUES only against this closed list: (1) subject loader + attestation, (2) runner hard-kill/cost-cap test, (3) canary prereg amendment, (4) post-canary: sampler identity, (5) bundle/store second passes, (6) manifests + ranked first-ten review, (7) Stage-A inventory + verifier rejection tests. **STOP** and escalate to the owner if: any *new* gate is proposed without removing one from this list; or more than **6 additional unpaid goal-hours** elapse without the canary having run; or more than **10 total additional goal-hours** elapse without a Stage-A authorization being issued or a written feasibility rejection of N=48. A third consecutive session ending with zero paid measurements while balance permits them is itself a STOP condition.

**Next five actions, in order:**

1. Write and commit the status-only preregistration amendment authorizing the ≤$1.50 engineering canary (semantic N=0, hardcoded e01/technical-long, excluded from all statistics, charged to reserve). **[Amendment]**
2. Close the two canary-blocking gates: subject loader/attestation and hard-kill/cost-cap runner test (local, unpaid).
3. Run the canary: cold start, attest subject, measure setup time, prefill/decode wall time at 4.5k, VP placebo overhead, VRAM headroom; kill pod; write results to notes.
4. Recompute the worst-case budget/yield projection for N=48 from measured numbers; if infeasible, produce the fallback design (e.g., one carrier render per fixture, or fewer strata) *before* Stage A, as its own amendment.
5. Close the semantic-critical gates (sampler identity, manifests, bundle/store passes, first-ten review, Stage-A inventory), then issue the unpaid Stage-A authorization.

## 5. The assumption most likely to strand the campaign

**The unmeasured per-case cost model under bf16/eager.** Every budget line ($12 / $30 / $4.50 / $8, slack $2.63) is derived from an assumed per-case time that has never been observed on this subject at this context length with eager attention and the VP path active. Eager attention at 4.5k tokens on a 30B model, plus a historically ~20-minute cold setup repeated across any pod churn, is the kind of factor that misses by 2–3×, and the plan has $2.63 of slack — less than two pod-hours of error across the whole campaign. A secondary, structural version of the same assumption: the sequencing currently presumes all gates can close unpaid, which is false by the draft's own admission; left unamended, that presumption strands the campaign at $57.13 with a perfect apparatus and N=0, which is precisely how the predecessor failed.

---

*Facts vs. inference: all statements under "Observed facts" and the balance/design parameters are taken verbatim from the supplied bundle. All cost-per-hour figures, per-case time arithmetic, and spiral diagnosis are reviewer inference and should be replaced by measured values after action 3. Recommendations requiring preregistration amendments are marked **[Amendment]** (actions 1 and, conditionally, 4).*
