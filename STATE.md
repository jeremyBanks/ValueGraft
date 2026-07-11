# STATE.md — session handoff / current state

## CURRENT TRUTH (2026-07-11) — coherent-state v10 execution

The active experiment is the additive Amendment-1-through-10
`coherent-state-gapped-v10` assay. It asks whether the exact incremental K/V of a
model-generated summary carries correct-history-specific information that a fresh
same-token encoding loses under position-preserving compaction. It does **not** test
every possible write-time channel elsewhere in the cache.

- Exact production checkpoint: `Qwen/Qwen3-30B-A3B-Instruct-2507`, revision
  `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, bf16, eager attention.
- No paid pod is running and no paid v10 semantic outcome exists.
- RunPod balance was last observed at `$63.3160022124`; the owner authorized an
  approximately `$60` total ceiling. The fresh Fable design consultation reported a
  `$4.440664` provider usage estimate, conservatively tracked but not verified as an
  incremental cash charge.
- The v10 production-tokenizer donor artifact passed 12/12 and is committed at
  `results/coherent_state_ladder/coherent_external_donors_gapped_v10_Qwen3-30B-A3B-Instruct-2507_20260711T122320Z.json`.
- The exact local 0.6B bf16 eager CPU ladder is live as PID `55253`, output prefix
  `results/coherent_state_ladder/coherent_state_ladder_gapped_v10_Qwen3-0.6B_20260711T123246Z`.
  Static provenance passed 1/1, attention-backend attestation passed 28/28, and the
  synthetic schedule stage passed 7/7 with aggregate discrepancy exactly `0.0`
  against the frozen `5e-4` limit. The twelve committed-case schedule stage is now
  running from 0/12. Intermediate sidecars are committed at each durable milestone.
- The accidental historical v6 CPU ladder remains paused and cannot authorize v10.

Amendment 11 now separates scheduling from authorization. The healthy local ladder
continues, while one paid technical-only 30B attempt may run concurrently after the
remaining exact-commit launch gates. The paid result is provisional and cannot authorize
semantics alone. Before any semantic work, the external release resolver must independently
validate `L AND T`: the exact committed terminal local-ladder PASS and the exact committed,
independently harvested 30B technical PASS. A remote outer wrapper revalidates the same
packet before inference; a separate final release receipt is required before any semantic
result counts. The scientific assay remains schema-2 `coherent-state-gapped-v10` under
Amendments 1–10; Amendment 11 is a separately identified authorization overlay. Every render
must still be saved and committed.

The release layer was frozen and implemented without changing any of the 35 inventoried v10
apparatus files; the aggregate remains
`818a60623c4858f0796865a124d255897325136f147ad611c1810026c9352715`, byte-identical to
the ladder-launch commit. Independent code and science re-audits returned GO after the
resolver gained deep 28-layer raw/numeric/source recomputation, exact path/inventory checks,
remote pre-inference closure, exact technical binding, outer receipt, and post-run release
closure. The targeted suite passed 34/34; the prior full suite passed 230 tests and monitor
self-test passed 59/59. A fresh full-suite run on the final exact candidate remains required
before paid launch.

Claude Opus 4.8 raised a recent-prior-art concern that history-conditioned notes may
reside on downstream tokens. After direct geometry inspection, two independent audits,
and a deep `claude-fable-5` consultation, the decision is to keep v10 frozen. Native
retained-tail tokens occur before the summary request in the source but after the summary
in the destination, so the proposed one-arm tail transplant was neither same-position
nor rotation-free and lacked a wrong-history control. Claude accepted and withdrew it.
The binding null boundary is: **no detected downstream-usable channel carried by the
generated summary rows under this fixed assay**, never “no write-time state exists
elsewhere.” See `notes/2026071164-fable-tail-channel-design-review.md` and the concluding
turns of `notes/2026071156-sol-fable-execution-coordination.md`. A same-position
request/header or immediate-post-summary correct-versus-wrong assay is reserved as a
separately preregistered follow-up reusing saved renders.

After data collection, Fable leads a fresh paper draft; Sol owns factual/methodological
truth-checking. The final review stack remains the required multi-angle Fable passes,
critics, terminology and methods/provenance checks, and an independent Codex review.
Only a fully reviewed in-repo paper may be promoted to `README.md` and pushed.

*Updated 2026-07-09. PIVOTED: QK-norm/ablation thesis DROPPED; now testing FRESH-CONVERSATION
REPRODUCTION of the headline. Everything below the "SUPERSEDED" marker is historical.*

## CURRENT TRUTH (2026-07-09) — read first

### ⚠⚠⚠ 2026-07-09 LATE — NO VALID POSITIVE CORNERSTONE (deep audit: notes/2026070976; postmortem: notes/2026070977)
The paper had NO valid positive cornerstone under the owner's rule (**cornerstone must be bf16 AND the
value-only ValueGraft method**). Verdict = honestly a **NEGATIVE / CAUTIONARY / BOUNDING** paper.
- **Precision truth:** BOTH behavioral headlines are **4-bit MLX only** in every SCORED form: recovery
  (judged sense/referent: raw_30b, raw_brief_repro) AND honesty (phase2_30b/4b). Local=MLX-4bit; pods=bf16.
  A bf16 honesty answer set exists (honesty_30b_bf16) but was NEVER SCORED → no scored bf16 behavioral result.
- **Honesty ≠ ValueGraft:** its arm H-pack = coupled KV-graft (α_K=1,α_V=1) in a PACKED non-production
  layout — NOT value-only ValueGraft (E-post, α_K=0, aligned, production layout). AND not content-specific:
  H-pack-wrongS (a DIFFERENT conv's state) suppresses fabrication 24/24 too → a packed-layout calibration
  phenomenon, not value-carried meaning. Supporting at best.
- **The bf16 value-only tests:** effect_bound (placebo-controlled, value-only, on recovery plants) = NULL
  on TWO bf16 models (Qwen3.6-27B + Qwen3-30B-A3B). SWE-Gym E-tuned +0.0156 nats (bf16, value-only) but
  tiny/TF-only/render-null/un-CI'd. So no robust positive at bf16+value-only.
- **STATUS: HALTED.** Provisional honesty-led draft committed (e017631) is SUPERSEDED. Reframing to the
  honest negative/bounding result OR seeking fresh bf16 value-only data — owner deciding; Fable advising
  on how to continue (notes/2026070978). CLAIMS/FINDINGS need a full precision-provenance correction pass.
  4-bit = SUPPLEMENTAL only (weaker/more-artifacts); primary target = bf16 ~30B.

### ⭐ STANDING GUIDANCE + OPEN DIRECTIVES (owner, 2026-07-09) — DO NOT LOSE THESE
1. **Fable writes to a notes/ file for EVERY serious consult** — I pre-create the empty file, pass rich
   context, Fable writes its own assessment there, returns the path. (AGENTS.md "FABLE WRITES TO A NOTES FILE").
   Don't blindly follow Fable — get its perspective, owner decides.
2. **CLAIMS.md is the audited single-source-of-truth** — every shippable number recomputed from disk with
   file + exact command + commit + status. Assemble the paper FROM it. Never second-guess a claim ad-hoc again.
3. **NATIVE/SELF-RENDER PREMISE must be VERIFIED before the paper implies it's a requirement.** Readers will
   infer self-rendering is required; it's NOT a problem if it turns out only helpful (still more robust/interesting)
   — but we must interrogate it, not assert it. Verification experiment being designed (Fable). Capture + reflect.
4. **RE-RENDER + BANK every key model's conversations** for reproducibility, each tagged with model ID +
   quantization + size. Fleet banks 32B/Mistral/Qwen2.5/phi-4 (never banked before). 30B already banked.
5. **Champion scans** (owner override of Fable's hold): mechanism map per model; bank the renders.
6. **BRIEF-CONDITION CAVEAT is load-bearing:** the judged +12pp AND SWE-Gym +0.0156 both ran under the
   brief (mechanism-isolation, baseline-handicapping) summary. Fable's #1 recommendation: get a
   **production-faithful (std-summary) arm** for the headline — currently NO positive result exists under std.
7. **Push-notify major results.** SWE-Gym cross-model = cheap warm-pod add-on (LOWER priority).
8. Fable holistic assessment (2026-07-09): notes/2026070972-fable-holistic-assessment.md — argues honesty-led
   may be safer than sense-led (degrades gracefully); std-summary arm is highest-leverage missing piece.

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
