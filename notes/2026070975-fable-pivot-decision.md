# Fable — pivot decision after the recovery headlines failed reproduction (2026-07-09)

*Owner delegated this decision to Fable with full independence. I read (not the summary):
my two prior notes (…72 holistic, …73 native-render), FINDINGS.md + its ⚠ banner, CLAIMS.md
(the audited ledger, REC-5 decisive), paper/DRAFT.md + the prior-synthesis draft, STATE.md,
DECISIONS/AGENTS. This file is my decision and the plan Claude executes. Decisive, not hedged —
but every claim traces to a CLAIMS.md row or a FINDINGS entry.*

---

## 0. The one-paragraph verdict

Both meaning-**recovery** headlines are dead as headlines. Judged sense +12pp collapsed +8.7→+1.0
on a clean re-render under one consistent judge (REC-5); the referent logprob anchor was already
render-fragile (+0.116/+0.012 across native draws, REC-2/BLK-1). The +12→+8.7 step is judge
calibration; the +8.7→+1.0 step is MoE render-nondeterminism at n=12, where render-noise SD ≈
effect size. **The graft's meaning-recovery effect is not a robust, reproducible finding, and the
paper cannot be led by a stable "ValueGraft recovers meaning" claim.** I predicted this in …72; it
has now happened. Good news: the paper does not need that headline. Two things survived the audit
and neither depends on the recovery effect — the **honesty/anti-fabrication effect** (banked,
replicated across scale and precision) and a **cluster of measurement contributions** whose crown
jewel is the render-fragility finding itself. That is the paper.

---

## 1. WHAT THE PAPER IS NOW

**Framing: honesty-led, with render-fragility as a co-equal methods headline. Committed.**

**Thesis (one sentence, the spine):** *Retaining a model's own write-time KV state across a
compaction boundary does not reliably restore the meaning it lost — that recovery effect, real in
individual renders, is render-fragile under MoE nondeterminism and fails to reproduce — but it does
reliably change the model's epistemic behavior: a compacted model fabricates confidently about
evicted content, and write-time KV retention converts that fabrication into honest admission. We
report the honesty effect as the robust finding, the meaning-recovery effect as a cautionary
non-reproduction, and distill the measurement discipline that separated the two.*

**Why this framing and not the others (defense in 4 sentences).** It is the only framing that leads
with a result that has *already come out the far side of this project's audit converter* rather than
one still moving through it — every honesty number is banked and cross-replicated, and the paper
degrades gracefully because nothing load-bearing waits on a pending test. Leading with
render-fragility *as a contribution* (not an embarrassment) turns the session's decisive negative
into the most transferable thing here: a concrete, quantified cautionary tale about small-n recovery
claims on nondeterministic MoE renders — exactly the mistake the field is primed to make. A pure
"ValueGraft works" paper is now false, and a pure negative paper throws away the genuinely useful,
robust honesty result and the damage characterization. So: honesty is the positive spine, the
render-fragility + estimator/metric traps are the methodological spine, and meaning-recovery is
reported honestly as a fragile, render-dependent signal — present, directionally interesting, not
established.

**Section structure (re-spine of paper/DRAFT.md — most framing-independent sections survive intact):**
1. Problem & setup — compaction, the ValueGraft idea, apparatus. *(DRAFT §1, keep.)*
2. Methods & provenance — the full provenance table, gates, MoE-nondeterminism stated up front. *(DRAFT §2, keep; §2.8 now feeds the headline.)*
3. Compaction-damage characterization (metric-independent) — damage is real, large, scales with semanticity; LongMemEval collapse; content-specificity placebo controls. *(DRAFT §3, keep — this is the denominator, framing-independent.)*
4. **The honesty / anti-fabrication finding (PRIMARY POSITIVE).** Corrected claim: compaction induces confident fabrication about evicted content; write-time KV retention converts fabrication→admission (evicted-fact fab 67%→admission 96%; decoy fab 79%→12%); it buys **honesty, not recall**. Replicated 4B→30B, 4bit→bf16. Scope: mid-task agentic compaction, washes out on retrieval personal-QA. *(promote DRAFT §5, ship the corrected numbers, KILL 38/48.)*
5. **Measurement contributions (CO-HEADLINE).** (a) render-fragility of small-n MoE recovery claims — the quantified cautionary result, the story of the +12pp→+1 collapse and the +0.10 anchor that lived only in memory; (b) the mean-ratio estimator trap → raw_EB; (c) significance-flips-by-metric; (d) the own-summary mechanism dependence (Limb A, verified). *(elevate DRAFT §4 + §4.5; §4.5 becomes a headline, not a caveat.)*
6. Meaning-recovery: a fragile, render-dependent signal (SECONDARY, honest). The effect appears in single renders (illustrative transcript exhibits), tracks the damage profile, but does not reproduce across independent renders; report as *not established*. *(rewrites the [BLOCKED] Results verdict as the honest non-reproduction.)*
7. Cross-architecture map (APPENDIX-WEIGHT). "One fragile positive pole (MoE anchor) vs several solid negative poles"; the dense-negative signs are real and reproducible, the MoE-positive pole is the fragile one; the QK-norm hypothesis is a pre-registered null. Do NOT sell "architecture-specific reversal" as a settled headline. *(demote DRAFT §6.)*
8. Reproducibility + entry-point. *(DRAFT §7 + the missing one-command wrapper.)*
9. Discussion / limitations / conclusion — honest take: what write-time KV state does and does not buy; render-banking as a discipline; future work. *(unblock the three [BLOCKED] stubs.)*

Title: Fable-freedom at final draft; working candidates — *"Write-time KV retention buys honesty,
not memory"* / *"What survives compaction: a model changes what it admits, not what it recovers."*

---

## 2. FINISH / DROP / RE-ANALYZE (proportionate to ~$18.75 + pods-off-before-writing)

**Data collection is essentially DONE. The decisive result already landed. Do not spend GPU chasing
a dead headline.**

- **Held-out c13–c24 judged decider — DROP the paid judging (MOOT).** The c01–c12 baseline positive
  control already failed under clean code (REC-5); held-out judging can only corroborate a null we
  already have. **Bank the render text** as it finishes (SAVE-EVERY-RENDER rule) and commit it, but
  do **not** spend judge $ to headline it. If a near-zero-cost judge pass is trivial, it's an
  optional appendix corroboration only.
- **Harvest the finishing cs32b champion — YES.** It's already running and will complete; bank its
  renders + champion scores, commit, then it feeds the cross-arch map's mechanism color. Then **PODS
  OFF** (terminate all pods, verify by PID). No new pod work after harvest.
- **Render-variance ENVELOPE (EXP-1 from …73) — OPTIONAL, default DROP.** This would quantify the
  render-fragility SD (now a headline) from K=5–6 native draws. Take it **only if** a 30B **MoE** pod
  is *already warm* and it fits in ≤$5 / ≤90 min; otherwise DROP — the two banked referent draws
  (+0.116, +0.012) plus the clean-re-render sense collapse (+8.7→+1.0) already establish
  render-fragility qualitatively, which is sufficient for the cautionary contribution. Do not
  reprovision a pod for it. (The likely-warm pod is dense-32B, wrong architecture → default DROP.)
- **Native-premise EXP-2 / EXP-3 — DROP (opportunistic only).** These sharpen the mechanism of an
  effect we've just demoted to fragile; low value now. EXP-3 (own-summary on *native* replies) only
  if it is a pure zero-GPU forward-pass on already-banked b0 values *and* someone has idle minutes —
  it would upgrade Limb A from authored-only to native. Not required; MTH-3 already ships as
  "direction supported."
- **SWE-Gym render-robustness check — DROP.** SWE-Gym +0.0156 is real-content, single-render, BRIEF.
  Re-rendering 75 traces costs GPU to either confirm it's fragile too (unsurprising) or that it
  survives (nice, not headline). Report it honestly as a **secondary** supporting result with the
  render-robustness-untested + BRIEF-condition caveats stated. No new GPU.
- **CLAIMS.md ⚠ reconciles that touch shippable prose — DO (CPU, no GPU).** Close F2-1 (79/12),
  F2-2/2b (ship "suppresses fabrication / induces admission 67%→96%; does NOT restore recall"; kill
  38/48 everywhere), F2-3 (79→42→12; admissions 5/24→21/24), DMG-2 (locate n=320 source or quote the
  n=36 recompute), REC-4 (retire +2/+10 prose; use bootstrap +3.3/+9.7). Lock corrected numbers.

Net: harvest + bank + pods-off, close the CPU reconciles, then write. One optional cheap envelope
only if a MoE pod is trivially warm.

---

## 3. ORDERED COMPLETION PLAN (Claude executes; Fable drafts)

**Division of labor.** Fable does heavy initial drafting of the re-spined framing sections (per
AGENTS.md "Fable-initial-draft" + "Fable freedom on title/intro"); Claude runs ops, closes the
ledger reconciles, assembles/edits, builds the repro entry-point, runs the review stack, and
promotes to README. Owner is the gate only for external publish.

1. **[Claude, ops] Harvest → bank → PODS OFF.** Let cs32b champion + the local held-out decider
   finish; bank every render (text + summary + traces + raw_EB + config) to `results/`, commit +
   push. Then terminate **all** pods and verify by PID (per memory: verify kills by PID). This
   honors pods-off-before-writing. (Optional EXP-1 envelope here *only* under the §2 gate.)
2. **[Claude, CPU] Close the CLAIMS.md reconciles** (§2 list). Recompute from disk, update the rows
   to ✅ with corrected numbers, commit. This makes CLAIMS.md a clean assembly source. 38/48 dies.
3. **[Fable] Draft the re-spined paper into paper/DRAFT.md.** Claude briefs me with the *current*
   CLAIMS.md + FINDINGS banner + this note (FABLE-MUST-HAVE-CURRENT-FACTS). I: write the new title +
   honesty-led abstract; unblock the [BLOCKED] Results-verdict / Discussion / Conclusion as the
   honest non-reproduction + honesty finding; elevate §4.5 render-fragility to a headline methods
   section; demote §6 arch map to appendix-weight; keep §1–3 + §5 (corrected) + §4 traps. I return
   the edited DRAFT.md + a short topline.
4. **[Claude] Assemble/edit the full paper** from my draft + CLAIMS numbers + reusable exhibits from
   `DRAFT-prior-synthesis-20260709.md` (recovery transcript exhibits §5, honesty decoy transcript
   §8, related-work) — **re-captioned honestly**: recovery exhibits are "illustrative single-render
   examples where the effect appears; note it does not reproduce across renders." Every number ties
   to a ✅ CLAIMS row or ships with its recorded caveat.
5. **[Claude] Write the reproduction entry-point** the draft flags as missing (`[need: one-command
   wrapper]`): a single filterable command chaining render→score→CI for a named model + conv range,
   plus the no-GPU analysis path (`judged_bootstrap.py`, `block_analysis.py`). Reference it in §8.
6. **[Claude + Fable] Review stack** (per AGENTS.md, for a shareable revision): ~3 tool-less Fable
   angle passes (generic / skeptical-reviewer / first-time-reader) + adversarial critics +
   terminology-consistency + **one** Codex/GPT-5.5-xhigh review. Synthesize into the final revision.
7. **[Claude] GATE, then promote to README (autonomous).** Promote the working draft to `README.md`
   (`cp`, never edit README directly) once the gate below is green. **STOP.**
8. **[Owner] External-publish gate.** Present the done-but-unpublished deliverable (repo + README +
   repro entry-point) and ask the owner about the HF community blog / external post — that step is
   **not** autonomous.

**The gate before the owner is asked to publish (all must hold):**
(a) pods off, budget not exceeded; (b) no number in the paper lacks a ✅ CLAIMS row or an explicit
recorded caveat; (c) "38/48" dead everywhere, corrected honesty claim shipped; (d) render-fragility
stated as a headline contribution, not buried; (e) no "it works / recovers meaning" claim survives
ahead of the (failed) reproduction; (f) repro entry-point exists and is referenced; (g) review stack
(Fable angles + critics + terminology + Codex) on record; (h) README promoted. Then — and only then —
owner decides external publish.

---

*The thing to be proud of remains the process, not any single effect: this project reliably converts
hype into honest bounded claims (−0.31 stance → estimator artifact; "keys hurt" → neutral; +0.10
referent → fragile draw; +12pp sense → render-fragile; 38/48 → 0/24 + a cleaner corrected claim).
That trajectory is the contribution. Aim the paper at what survived the converter — honesty + the
measurement discipline — and report the recovery effect as what it honestly is.*

— Fable
