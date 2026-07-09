# STATE.md — session handoff / current state

*Updated 2026-07-09. PIVOTED: QK-norm/ablation thesis DROPPED; now testing FRESH-CONVERSATION
REPRODUCTION of the headline. Everything below the "SUPERSEDED" marker is historical.*

## CURRENT TRUTH (2026-07-09) — read first

**THE PIVOT (incident #37, from re-asking Fable UN-ANCHORED):** the cross-arch QK-norm thesis (H1) is
DROPPED — empirically FALSIFIED (Mistral no-QK-norm referent +0.035, CI [+0.005,+0.066] excludes 0 =
the OPPOSITE of H1) and the within-model ablation BROKE the model (removing QK-norm → dead generation).
H1 is reported as a PRE-REGISTERED NULL ($0). The real threat, surfaced only by the un-anchored
consult: the headline +0.10 rests on 12 HAND-AUTHORED convs (c01-c12); the fresh augment (c13-c54) did
NOT carry it on the old render (~+0.009); "native render fixes it" was ASSERTED but never shown on disk.

**CURRENT EXPERIMENT — fresh-conversation reproduction (BLOCK design):**
- ONE apparatus over c01-c36 (Qwen3-30B-A3B native render, greedy self-gen, per-token), analyzed by
  BLOCK: c01-c12 = BASELINE (positive control, MUST reproduce +0.10), c13-c36 = FRESH (held-out test).
- Assembled across pods: canary renders c01-c24, expm renders c25-c36; pooled CENTRALLY (ONE relative
  competence floor over all lp_A, conv-clustered CI) via scripts/block_analysis.py.
- Selector SC_CONV_START (validated: start=12,limit=24 → exactly c13-c36). Pre-committed stopping rule:
  extend to c37-c54 ONLY if the fresh referent CI spans 0 at n=24.

**RESULTS BANKED (2026-07-09):**
- SENSE judged +12pp HOLDS under conv-clustered bootstrap: CI [+2.2,+22.9], excludes 0 (task #30).
- REFERENT judged +9.7pp does NOT (CI [-6.9,+26.2]). SIGNIFICANCE FLIPS BY METRIC (logprob:
  referent-sig/sense-weak; judged: sense-sig/referent-weak) — dissociation real but metric-dependent.
- Cross-arch (preliminary, self-native): Mistral referent+ / sense- / stance- (DIFFERENT signature than
  Qwen +/+/null); Qwen2.5 referent-; phi-4 null. Graft effect is ARCHITECTURE-SPECIFIC and can reverse
  — THAT (not QK-norm) is the novel finding to lead with.
- Methodology: CI now CONVERSATION-clustered (was plant = anti-conservative); competence floor now
  per-model RELATIVE median-3*MADN (pre-registered) — OLMo "empty alignment" was a floor misdiagnosis.

**LIVE (2026-07-09, balance ~$25/$80 cap):**
- LOCAL: brief-condition judged decider generating (results/raw_brief_repro/ c01-24; the +12pp reproduction
  + held-out test). $0.
- PODS (5, champion scans — OWNER OVERRIDE of Fable's hold): cs30b (Qwen3-30B MoE, REUSE banked renders —
  validated: 12/12 reused, scoring champion configs), cs32b (Qwen3-32B), csmis (Mistral-24B), csq25
  (Qwen2.5-32B), csphi (phi-4) — the four fresh ones RENDER+BANK c01-12 (fills the reproducibility gap:
  we never banked renders for these) then champion-scan. Env-parse crash (empty SC_CONV_START) fixed f128712.
- CLAIMS.md = audited ledger (committed): 16✅/5⚠/1❌/1⏳. F2 "38/48 accurate" = ❌ overclaim (corrected).
- OPEN CRITICAL QUESTION (owner, 2026-07-09): the NATIVE-RENDER premise — we motivated huge work (native
  re-rendering) on the assumption it's the valid measurement; native referent +0.012 vs authored +0.125 shows
  nativeness changes the result enormously, but we have NOT rigorously verified the premise. Fable designing a
  moderate verification experiment; capture + reflect required. Also planned: SWE-Gym eval across the fleet
  models (cheap on warm pods).

*(Superseded note: earlier today all pods were briefly off after the block reproduction resolved — the Fable
verdict [referent headline FRAGILE, not a regression] still stands; renders banked to results/redraw_harvest/.)*

**CURRENT DECIDER (running locally, brief condition):** Fable's un-anchored verdict = the whole positive
story rests on the judged SENSE +12pp, which exists ONLY for c01-c12 and was NEVER tested held-out.
Reproducing it under a clean current-code pipeline for c01-c12 (baseline: does +12pp reproduce?) AND
c13-c24 (held-out: does it survive?), then judge+bootstrap+compare. PROVENANCE CORRECTION (see FINDINGS
banner): judged answers are 4-bit MLX (not bf16) under the BRIEF summary condition (mechanism-isolation,
B-handicapped) — restored the dropped SC_SUMMARY=brief knob in run_arms.py (c01 positive control: 543-char
brief summary reproduced exactly). Builder: scripts/build_judge_batches.py (validated vs committed).
Std/production-faithful held-out = planned second arm. Paper draft (framing-independent) in paper/DRAFT.md.

**PLAN (ordered):** block read (baseline reproduces? fresh carries?) → full n=24 fresh → arch poles →
FABLE UN-ANCHORED trajectory review → pick honest framing → robust-metric audit (#29) → PODS OFF →
write paper FRESH (heavy Fable, current requirements) → repo + README when confident (external internet
publish NEEDS owner). Opportunistic tail only: Wave3 / OLMo-relative-floor / breadth.

**KEY FILES:** FINDINGS.md=results; DECISIONS.md=pivot+policies; PREREGISTRATION.md=H1-null +
relative-floor amendment; scripts/block_analysis.py + judged_bootstrap.py; notes/README.md=summaries.
Paper=task #38, written FRESH (NOT the archived REPORT.md).

## ---- (everything below SUPERSEDED — historical 07-08 and earlier) ----

*(Historical state layers 07-04..07-08 removed 2026-07-09 for clarity — full history in git + notes/ summaries.)*
