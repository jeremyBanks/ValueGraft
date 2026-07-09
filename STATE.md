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

**LIVE PODS (2, ~$2.78/hr):** expo=pair-B Qwen3-32B @24, w6=Qwen2.5-32B @24 (arch poles, running).
The block pods (canary c01-c24, expm) TIMED OUT and were terminated — INCIDENT #38: non-durable
in-memory render lost 6.3h. Block re-run PENDING: restore per-conv incremental checkpointing (being
built + Fable-reviewed) → fresh clean pods → re-run interruptibly.

**PLAN (ordered):** block read (baseline reproduces? fresh carries?) → full n=24 fresh → arch poles →
FABLE UN-ANCHORED trajectory review → pick honest framing → robust-metric audit (#29) → PODS OFF →
write paper FRESH (heavy Fable, current requirements) → repo + README when confident (external internet
publish NEEDS owner). Opportunistic tail only: Wave3 / OLMo-relative-floor / breadth.

**KEY FILES:** FINDINGS.md=results; DECISIONS.md=pivot+policies; PREREGISTRATION.md=H1-null +
relative-floor amendment; scripts/block_analysis.py + judged_bootstrap.py; notes/README.md=summaries.
Paper=task #38, written FRESH (NOT the archived REPORT.md).

## ---- (everything below SUPERSEDED — historical 07-08 and earlier) ----

*(Historical state layers 07-04..07-08 removed 2026-07-09 for clarity — full history in git + notes/ summaries.)*
