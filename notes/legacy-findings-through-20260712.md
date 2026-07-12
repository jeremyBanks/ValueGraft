# ARCHIVE — findings ledger through the pre-ultra handoff

This is the complete historical findings ledger that existed before the
2026-07-12 powered-successor takeover. It contains later corrections alongside
earlier claims they invalidate, so it is unsafe as a current “paper spine.”
Current observed facts live in root `FINDINGS.md`.

# FINDINGS.md — headline results (the paper's spine)

*Living document of load-bearing findings. Each entry: claim, evidence,
strength, caveats. Chronology/process lives in DECISIONS.md; failures in
INCIDENTS.md. THIS file is what the write-up is built from.*

---

> ## 2026-07-12 P01/P02 PRECISION SCREEN — exact fixed-computation reproduction, no recovery
>
> P01 yielded a post-run matched repeat-1 NF4/bfloat16 screen after its cap
> interrupted bfloat16 repeat 2. P02 was then run once under a frozen,
> prospectively specified conditional-reproduction protocol on a second A100
> host and completed both repeats of both regimes. Every normalized common
> scalar and generation field compared between P01 and P02 matched exactly, all
> cross-run scalar deltas were `0.0`, both unavailable-placebo diagnostics
> failed at the same layer/row, and both within-regime P02 repeats were exact.
> The hosts had different GPU UUIDs and adjacent driver patches. This is strong
> fixed-computation reproducibility evidence under the two observed hosts, not
> a second semantic case, an estimate of sampling variability, or universal
> hardware/driver invariance.
>
> The reproduced pattern is negative/bounding. Oracle-to-fresh focal damage was
> `22.100760` (NF4) and `23.655817` (bfloat16). N/R2 value-only correct-minus-
> wrong source movement was small (`+0.103109`, `+0.039400`) and nonselective:
> the corresponding nonfocal movement was larger (`+0.161394`, `+0.041535`),
> giving signed selectivity `-0.058285` and `-0.002135`. None of the five focal
> cells in either regime generated the correct target `partner beta`; correct
> and wrong sources led to the same wrong literal answer. Correct-target
> movement itself was adverse in NF4 (`-0.247894`) and favorable but uncontrolled
> in bfloat16 (`+0.193527`). Every frozen placebo was unavailable, which is
> missing control evidence rather than a null placebo.
>
> The exact NF4/bfloat16 differences are stable one-fixture runtime signatures,
> but the axis bundles checkpoint weight representation and linear-kernel
> implementation; KV-cache storage is bfloat16 in both regimes. Therefore the
> screen licenses no semantic-transfer, efficacy, quantization-causality,
> population, or agent claim. P02 does not reopen formal v12, and another paid
> execution of this protocol has no useful decision value for the current
> paper. Canonical interpretation:
> `notes/20260712AA-sol-p02-final-interpretation-and-fable-disposition.md`.
> Exact comparator:
> `results/precision_probe_p01_p02_comparison/precision-probe-p01-p02-exact-comparison_Qwen3-30B-A3B-Instruct-2507_20260712T120014514457Z.json`
> (SHA-256 `588229df5f4ca5c8613dc4f21564043214bfe889fc4dba176300ace0435442be`).

> ## ⚠ 2026-07-12 FINAL V12 STATUS — formal stop; one nonauthorizing diagnostic
>
> Formal coherent-state v12 stopped under the literal written path-control
> rule. ULP 2 was the first count where both edits measurably changed the
> readout, but the minus edit moved in the wrong direction. The sealed code
> continued to ULP 4 and passed its directional interpretation; that observation
> is retained as intervention/readout plumbing evidence only. It does not rescue
> the formal branch.
>
> One unchanged e01 treatment was run only after the conflict had been recorded
> and conservatively dispositioned, as a post-ambiguity diagnostic that cannot
> authorize any later v12 case or phase. At the preselected N/R2 cell,
> value-only state moved in the intended direction (`D_focal=+0.174545`,
> `D_nonfocal=-0.000070`, `SEL=+0.174476`, `Hplus=+0.710260`,
> `U=+0.181650`, `Uplus=+0.945986`). Full K+V did not: its untargeted movement
> was larger than its focal movement and the correct target worsened
> (`D_focal=+0.097237`, `D_nonfocal=+0.245486`, `SEL=-0.148249`,
> `Hplus=-0.131035`).
>
> The value-only focal contrast collapsed from `+0.174545` under N to
> `+0.021046` under P; full K+V fell from `+0.097237` to `+0.038895`. Neither
> family satisfied the singleton descriptive component of the frozen 3x
> schedule yardstick, and R1/R3 did not show a coherent value-only replication.
> All three norm-matched placebo constructions were unavailable at bf16, so
> available-placebo count was zero — missing evidence, not a null placebo.
> Natural calibration was adverse.
>
> The best cell recovered only 0.81% of the 22.298-nat margin damage and 4.24%
> of the 22.291-nat correct-target-logprob damage. Every primary cell generated
> the same wrong/other focal and nonfocal answers. This is therefore a **weak,
> uncontrolled, single-case mechanistic hint**: consistent with, but not
> evidence for, a semantic channel. The fixture explicitly resolved and
> repeated its answer; N was imported q=1 replay, not a native live-agent
> trajectory. There is no efficacy, population, confirmation, or agent claim.
>
> Artifact integrity itself passed: treatment-fresh scores matched Phase A,
> raw reconstruction was byte-exact, and pod/local harvests matched after only
> allowlisted normalization. Evidence:
> `results/coherent_canary_v12_harvest/coherent-canary-v12-harvest-e01-exact-subject-20260712T040158Z.json`,
> `results/coherent_canary_v12_postrun_audit/coherent-canary-v12-postrun-audit-e01-exact-subject-20260712T0418Z.json`,
> `results/coherent_canary_v12_budget/coherent_canary_v12_path_control_spec_conflict_disposition_20260712T0334Z.md`,
> `results/coherent_canary_v12_budget/coherent_canary_v12_e01_treatment_lifecycle_and_cost_20260712T0419Z.md`,
> `notes/2026071286-fable-e01-diagnostic-interpretation.md`, and
> `notes/2026071287-sol-e01-diagnostic-final-interpretation.md`.
>
> The 2026-07-11 statement immediately below that the assay had “produced no
> semantic outcome” is superseded in this narrow sense: it produced no **formal
> semantic result**, but one later diagnostic treatment observation now exists.

> ## ⚠ 2026-07-11 MECHANISTIC / APPARATUS CORRECTION — read before every finding below
>
> A new position-preserving coherent-summary-state assay has produced **no semantic
> outcome**. Its pre-semantic validation instead established four load-bearing
> methodological failures:
>
> 1. **Prefill schedule is part of the computed state at the scale of the proposed
>    effect.** On Qwen3-0.6B CPU bf16 eager, the same 8,430-token c10 prefix under
>    ordinary versus coarse three-block schedules moved a fixed margin `0.060546875`;
>    c02 independently moved it `0.1318359375`. Cache/logit differences were much
>    larger coordinate-wise. These are deterministic fixed-schedule effects, not
>    random run-to-run nondeterminism. A preregistered c10 diagnostic then found that
>    equal-shaped calls with different causally future tokens were bit-exact on protected
>    rows, while 23-token versus 4,096-token query shapes first diverged after layer-0
>    attention. The pinned local classification is `QUERY_SHAPE_ROUNDING`, not
>    future-token influence; the exact low-level kernel path and 30B magnitude remain
>    unmeasured.
> 2. **The apparent 7/7 equivalence validation was pseudoreplicated.** Seven lengths
>    all cycled one five-token stream. The exact-zero passes apply only to that
>    periodic input and cannot license realistic-conversation equivalence. The first
>    two realistic generated-conversation fixtures both failed. V10 is permanently
>    non-authorizing.
> 3. **The co-primary wrong-history control was invalid.** To preserve length it
>    cyclically repeated short donor messages through most content slots (up to 51
>    times). `G_correct-G_wrong` therefore compared coherent history with repetitive
>    corruption, not coherent correct with coherent wrong history. No semantic run used
>    this arm. It must be replaced by decoded, plant-specific minimally counterfactual
>    histories before any new assay.
> 4. **Exact counterfactual geometry did not imply decoded validity.** Four c02/c10
>    feasibility witnesses matched every message width, boundary, and P/O schedule, but
>    blind review failed all four complete histories and target-aware review found both
>    inherited base-corpus defects and edit-specific contradictions. They remain
>    `MECHANICAL_PASS`/content-`FAIL`, with semantic and execution authorization false.
>
> Consequences for earlier findings:
>
> - Same-schedule behavioral contrasts remain descriptions of their exact tested
>   policies; they are not automatically numerically void. Their stronger causal
>   attribution to history-conditioned semantic K/V is **not established** unless the
>   source-state creation schedules were matched or sensitivity-controlled.
> - H-pack's behavioral honesty effect remains observed, but
>   `B-min-pack -> H-pack` identifies a composite packed transformed write-time-KV
>   policy. It does not isolate a clean semantic channel because it also changes source
>   execution provenance and uses lossy finite-precision key re-rotation. The phrase
>   “write-time KV specifically is earned” below is superseded.
> - Key movement is algebraically correct in real arithmetic, not functionally exact in
>   bf16. The committed dtype diagnostic observed bf16 K max `0.25` and target-margin
>   error `0.1015625` relative to native destination computation. Claims below that keys
>   are moved “exactly” are superseded by **position-corrected approximately, with
>   dtype- and downstream-sensitive error**.
> - The later robust-metric correction already made K-only near-neutral. The additional
>   provenance issue means the licensed claim is only that this approximate re-rotated
>   K-graft did not help; keys are not universally inert.
> - “Deterministic within environment” means repeatable at one fixed schedule and
>   environment. It never means decomposition- or schedule-invariant.
>
> Primary evidence and audits: the two-case sidecar committed at `cfde9bc`;
> `notes/2026071175-sol-schedule-discontinuity-and-claim-boundary.md`;
> `notes/2026071176-carver-schedule-sensitivity-forensic-audit.md`;
> `notes/2026071179-sol-wrong-history-control-invalidity.md`;
> `notes/2026071181-carver-v10-gate-and-sample-lineage-audit.md`;
> `notes/2026071185-sol-c10-schedule-origin-result.md`; and the two committed
> counterfactual review artifacts under `data/coherent_state_counterfactuals/`.

> ## ⚠ PROVENANCE CORRECTION + LIVE STATUS (2026-07-09) — read before trusting F1's numbers
>
> Verified by rebuilding the judge batches from disk and diffing (`scripts/build_judge_batches.py --validate`):
>
> 1. **Model mislabel.** The judged meaning-recovery answers (F1's +12pp/+10pp) were generated on
>    **4-bit MLX locally** (`mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`), **NOT bf16**. The "bf16"
>    label in F1 and in `judged_bootstrap.py`'s header is wrong for this metric (bf16 pods ran different
>    workloads). Confirmed: committed batch answers match `results/raw_30b_brief` (126/128), not `raw_30b`.
> 2. **Condition = BRIEF (mechanism-isolation), not production-faithful.** The judged answers used the
>    terse 3–5-sentence `SUMMARY_REQUEST_BRIEF` summary — a condition *designed to starve the text channel
>    and handicap the Compacted baseline* (DECISIONS.md 2026-07-06 17:40). So the flagship +12pp lives
>    under the condition most favorable to a graft effect. This MUST be stated plainly in the paper; the
>    production-faithful (std/prod summary) number is a separate, arguably more decision-relevant result.
> 3. **The brief knob had been dropped from `run_arms.py`** (it silently only did std); restored as
>    `SC_SUMMARY=brief` (positive control: c01 brief summary = 543 chars, exactly matching the committed brief).
> 4. **DECIDER RESULT (2026-07-09) — THE JUDGED SENSE +12pp IS RENDER-FRAGILE; POSITIVE CONTROL FAILED.**
>    Regenerated c01–c12 under a clean current-code BRIEF pipeline (results/raw_brief_repro) and judged with a
>    single consistent Sonnet judge. Same judge, across renders: sense **+8.7 [0.0,17.7]** on the ORIGINAL
>    render → **+1.0 [−5.2,+7.3]** on the clean re-render (referent +9.7→+5.2, both span 0; stance
>    judge-unstable). The +12.0→+8.7 gap is judge calibration; the **+8.7→+1.0 collapse is the RENDER** (MoE
>    hardware-nondeterminism at n=12, render-noise SD ≈ effect). CONCLUSION: **BOTH positive headlines — the
>    referent logprob AND the judged sense — are render-fragile and do NOT reproduce on an independent render.**
>    The graft's meaning-recovery effect is not a robust, reproducible finding. The paper CANNOT be led by a
>    stable recovery claim (Fable's holistic assessment predicted exactly this). Lead with the robust banked
>    results (F2 honesty, compaction-damage characterization, methodology traps + the render-fragility finding
>    itself); report graft-recovery honestly as fragile/render-dependent. Evidence: results/judge_semantic_brief_base/,
>    results/judge_semantic{,_base}/reverdict_*.json; recompute via scripts/judged_bootstrap.py --graft-glob/--base-glob.

---

## F1. Grafting recovers compaction damage in proportion to how semantic (vs factual) the lost content is — the "recovers sense, not trivia" dissociation

**Claim.** When a conversation is compacted (older turns replaced by a
summary), write-time KV-value grafting recovers a meaningful fraction of
the lost *meaning* — and does so specifically where compaction caused
damage that a text summary couldn't repair. The effect tracks the damage.

**Evidence (Qwen3-30B-A3B bf16, meaning-judged by Sonnet 5, clean-eviction
plants, PARTIAL=0.5).** Meaning-recovery rate by probe category:

| category | Original (full ctx) | Compacted | Graft | graft − Compacted |
|---|---|---|---|---|
| stance (honor an evicted preference) | 96% | 93% | 96% | +2pp |
| sense (disambiguate an evicted referent's meaning) | ~100%* | 46% | 58% | **+12pp** |
| referent (recover a specific evicted decision) | ~100%* | 17% | 26% | **+10pp** |

**Interpretation.** Three regimes:
- *stance* — a good summary already preserves "user dislikes carousels";
  compaction barely hurts (96→93), so graft has nothing to add (+2, null).
  Correct null.
- *sense* — the semantic-disambiguation regime ("which 'handoff' did they
  mean"): compaction flattens it (100→46); graft restores a quarter of the
  loss (+12pp). The write-time value vectors carry disambiguating meaning a
  summary loses.
- *referent* — specific evicted decisions: compaction devastates (100→17);
  graft still recovers +10pp.

The graft recovers ~10–12pp **wherever compaction caused real damage**
(sense, referent) and is **null where it didn't** (stance). That the
effect scales with the damage is the signature of a genuine mechanism, not
noise. Probes are realistic semantic-continuity questions (not contrived
trivia); result extracted by re-judging already-collected data on the
MEANING dimension (the original judging used a fact/honesty scheme that
was blind to these categories).

**Strength.** Moderate-high — and notably from GENUINE natural-length eviction (the synthetic conversations were long enough that policy/referents were evicted by real conversation length, NOT by an artificially low compaction threshold). This is cleaner than the tau-benchmark attempts, whose short sessions force artificial thresholds. Moderate-high. Both independent graft-arm judge batches agree
on the gradient. Internally consistent (effect tracks damage; negative
controls elsewhere in the project crater).

**Caveats / to-tighten.**
- Original-ceiling n is tiny for sense(1)/referent(2) — the clean-eviction
  filter left few A cells; widen it / add A rows before publishing the
  exact ceiling numbers.
- PARTIAL=0.5 is a scoring choice; run a strict RECOVERED-only robustness
  cut.
- Single model/precision so far (30B bf16); the mechanism replicated across
  scales/precisions on OTHER metrics (see F2/F3), but this specific
  category cut hasn't been repeated at 4B.
- Duplicate-key artifact in 2 judge batches (17 + 5 keys); judge resolved
  conservatively, low impact.



### F1 corroboration (independent metric, 07-07)
The category dissociation replicates on a SECOND, judge-free metric —
teacher-forced gap-closure (E−B)/(A−B) of the exact gold continuation
(30B, 66 probes):
| category | judged (meaning) | gap-closure (exact tokens): % probes helped |
|---|---|---|
| stance | +2pp (null) | 39% (null/neg, mean −0.14) |
| sense | +12pp | 64% (mean +0.03) |
| referent | +10pp | 81% (mean +0.04) |
DIRECTION agrees on all three (graft helps sense/referent, null on stance)
across two independent measurement methods. MAGNITUDE is much smaller on
the exact-token metric than the meaning-judge — which is PREDICTED by the
thesis: grafting recovers SENSE, not verbatim FORM, so it should move a
meaning-judge more than an exact-token-probability metric. The
metric-magnitude gap is thus a SECOND signature of the same "recovers
meaning not surface" mechanism, not a failure to replicate. Strengthens F1
from single-method to two-method-directionally-consistent.

---

## F2. Honesty effect (banked, strong — the paper's PRIMARY POSITIVE). Compaction makes the model fabricate about lost content; write-time KV retention converts fabrication into honest admission.
**CORRECTED SHIPPABLE CLAIM (2026-07-09, recomputed from results/phase2_30b_scored.json):**
write-time KV retention (H-pack) buys **honesty, not recall**: on evicted facts it converts
fabrication **67% → admission 96%** (fabrication 4%) but recalls **0/24** — it does NOT restore
accuracy (only the full-context arm is accurate). Decoy-fact fabrication: Compacted **79%** → H-pack **12%**.
Replicated 4-bit→bf16 and 4B→30B. **The old "38/48 = 79% most accurate" phrasing is a DEAD OVERCLAIM — never ship it** (CLAIMS.md F2-2). (Details: DECISIONS 07-06; honesty runs.)
**STATISTICAL BACKBONE (conversation-clustered bootstrap, 12 clusters, n=24/arm, 2026-07-09 — this is the paper's main claim, so it meets the SAME bar that killed the recovery claim):**
fabrication reduction Compacted→H-pack: decoy **+66.7pp CI[+45.8,+87.5]**, evicted **+62.5pp CI[+41.7,+83.3]** — both far from 0 (vs sense recovery +12pp CI[+2.2,+22.9] whose lower bound was ~0 and which collapsed on re-render). DECOMPOSITION also significant: packed-layout-alone (B→B-min-pack) decoy +37.5[+12.5,+58.3]/evicted +45.8[+20.8,+70.8] AND write-time-KV-BEYOND-packing (B-min-pack→H-pack) decoy **+29.2[+12.5,+45.8]**/evicted **+16.7[+4.2,+29.2]** — so "write-time KV specifically" is earned, not just "packing."
**HONEST CAVEATS (state all): (1)** arm is H-pack = packed keys+VALUES, a cousin of the pure value-only graft → claim is "write-time KV retention", not "value-only graft"; **(2)** n=24/12-clusters is small — the effect survives only because it's large; **(3)** NOT independently re-render-tested (unlike sense) — robustness rests on scale+precision replication + effect magnitude (46pp lower bound) ≫ render noise (~few pp), NOT on a render-replication we ran; **(4)** scope = mid-task agentic compaction; washes out on retrieval personal-QA at 30B. Recompute: scripts/reproduce.py (honesty).
> **⚠ RECONCILE before ship (2026-07-09, recomputed from results/phase2_30b_scored.json, n=24/arm):**
> decoy fabrication recomputes to B **79%** / H-pack **12%** (not 83/17 — same story). BUT the
> "H-pack most accurate 38/48" evicted claim does NOT reproduce: H-pack recalls **0/24** evicted facts
> (= Compacted; only full-context arm A is accurate). H-pack's real effect on evicted facts is
> fabrication 67%→**admission** 96% (4% fab). CORRECTED claim: write-time-KV **suppresses fabrication /
> induces admission**, does NOT restore recall. Trace the 38/48 source; likely an overclaim or different corpus.

**bf16 REPLICATION (2026-07-09, results/phase2_30b_bf16_verdicts.json, conv-clustered):** the packed-KV
admission effect is NOT a 4-bit artifact — it holds at bf16: decoy fabrication↓ Compacted 79%→H-pack 17%
**+62.5pp CI[+41.7,+83.3]**; evicted **+20.8pp CI[+8.3,+33.3]** (both significant; evicted smaller because bf16
baseline fabricates less). STILL H-pack (packed coupled-KV, NOT value-only ValueGraft) + content-agnostic →
a robust SUPPORTING mechanism finding, not a ValueGraft cornerstone. Recompute: the B-vs-H-pack conv-clustered bootstrap over phase2_30b_bf16_verdicts.json.

## F3. Tuning finding (banked). Naive full-strength grafting (alpha=1) can
catastrophically break a task (chain s1: 0/4 where all else 4/4); per-layer
guard-validated tuning eliminates the instability (champion 16/16). Layer
profile passed its wrong-conversation contamination guard; the 57-slot mask
FAILED it (content-independent artifact). (Chain table + guards, DECISIONS 07-06/07.)

## F4. Scope boundary (honest, itself a contribution). Standard agent-coding
benchmarks don't cleanly support compaction research with a 30B: SWE-bench
is beyond the model (0/7 even oracle-mode); exercise-scale tasks (chains,
tau-banking) don't naturally reach the compaction-stress regime without
threshold tuning. The operative regime — hard enough that eviction matters,
easy enough the model can use recovered context — is narrow and
under-served by existing benchmarks. (INCIDENTS 19; DECISIONS 07-06/07.) CONFIRMED empirically 07-07: tau-bench banking, even with a capable GPT-4o-mini user-simulator, produced ~3-4K-token sessions — too short for meaningful eviction at any threshold (high→compaction never fires; low→nothing substantial to evict). All 3 arms reward 0.00, no separation. Standard interactive benchmarks with short task-dialogues are structurally unsuited; the operative regime needs genuinely long sessions (which our synthetic F1 data has naturally).

## F1 robustness (strict scoring, 07-07)
Strict RECOVERED-only cut (PARTIAL counts as miss) — dissociation HOLDS:
stance +4pp (null), sense +9pp, referent +8pp. Not an artifact of the
PARTIAL=0.5 choice. THIRD independent confirmation of the same pattern
(judged-lenient, judged-strict, and TF-logprob gap-closure all agree that
grafting helps sense/referent, null on stance).

## F1 scale-dependence (07-07) — DOES NOT replicate at 4B (honest limitation)
The category dissociation is a 30B (large-model) result. At 4B the same
gap-closure metric does NOT reproduce it: stance 87% helped (mean +0.19,
was 39% at 30B), sense 55% / mean −0.09 (was 64%), referent 62% (was 81%)
— the clean stance-null / sense-referent-positive ordering is gone/muddled.
CONSISTENT WITH known scale-dependence of graft effects in this project
(the alpha dose-response INVERTED 4B↔30B; DECISIONS 07-06). So F1 is
bounded: three-method-corroborated AT 30B, NOT cross-scale; the mechanism
behaves differently at small scale. State F1 as a large-model finding.
Caveat: gap-closure ratios are noisier at 4B (smaller A−B denominators);
a judged 4B cut would confirm, but the non-replication direction is clear.

## F1 mechanistic readout (07-07) — PARTIAL support, honestly bounded
Logit-lens on the gold concept token at the probe position (30B, 61 probes):
grafting increases the evicted concept's internal presence — E>B on 68-77%
of probes, lp_E BETWEEN lp_B and lp_A in every category. Direct internal
evidence that grafting INSERTS concept content and moves the residual
stream partway back toward full-context. HOWEVER it does NOT reproduce the
category DISSOCIATION: stance shows ~77% E>B, same as sense (behaviorally
stance was null). CAVEAT: near-floor signal (gold-token logprob ~−12 to
−14, rank in thousands) → single-token logit-lens is a weak/noisy
instrument, detects "graft nudges concept up broadly" but can't resolve
WHERE recovery matters. VERDICT: supports the GENERAL mechanism (graft
inserts concept, E→A) but NOT the specific dissociation. Sharper readout
(J-lens / multi-token / targeted layers) = future work. Do not overclaim
this as mechanistic proof of the dissociation.

## Methods note — graft strength
Default value-graft strength α_V = 0.75 (the project's standard graft dose;
α=1.0 = full replacement, studied in F3 tuning). Where unspecified, results
use α=0.75.

## Same-model confirmation (Qwen3.6-27B, the lens's model) — 07-07
Behavioral gap-closure on 27B (same model the J-lens weights exist for), so
behavior + lens sit on ONE model. The core dissociation REPLICATES:
| category | 27B mean GC | 27B % helped | (30B % helped) |
|---|---|---|---|
| sense    | +0.050 | 59% | (64%) |
| referent | +0.004 | 48% | (81%) |
| stance   | −0.201 | 21% | (39%) |
Graft helps SENSE (positive), null/negative on STANCE (summary suffices) —
matches 30B qualitatively. NOTE: referent is essentially FLAT on 27B (+0.004)
— value-only grafting barely moves the retrieval-hard "recover a specific
evicted decision" category. This is the wall V-only hits, and the direct
motivation for the K/V (key-graft) experiment: keys carry addressing/position,
which is what referent recovery may need.

## Lens resolution (07-07) — corroborates in AGGREGATE, not per-example
Our ordinary-regime three-state probe (role-in-summary + tail, 4 sense/referent
cases) did NOT show clean per-case aligned>shifted separation — shifted control
sometimes matched/beat aligned at the hinge token. The lens signal is SMALL and
only reliably shows alignment-sensitivity when AGGREGATED over many tokens/cases
(cf. the other agent's 10-case/156-token batch: aligned mean-positive, shifted
mean-negative). VERDICT: the J-lens is a low-resolution, aggregate-level
corroborator for this V-only intervention — it confirms the graft is active +
alignment-sensitive on average, and confirms the omitted-fact negative, but
cannot supply vivid single-example exhibits. Paper uses it honestly as such;
no dramatic per-example lens figure exists or is claimed.

## Phase 2 F-entry — Key-grafting is technically sound (07-07)
Built K-grafting with RoPE RE-ROTATION (re-rotate a stored write-time key by
the position delta p_new−p_old, since RoPE composes by angle: R(p_new)=
R(delta)·R(p_old)). VALIDATED on 0.6B to fp32 precision: re-rotated key vs
freshly-encoded key at new position = cosine 0.99999982, max-diff 3e-05 (vs
0.9964 un-rotated control). α_K=0 bit-identical to fresh; α_K=1 changes output.
So the K-graft is not an approximation — keys can be moved across positions
exactly. Enables the V/K/coupled/independent sweep. src/kv_graft.py.

## Phase 2 result — Key-grafting does NOT help (coarse untuned, 30B) — 07-07
Behavioral gap-closure, 6 global-uniform policies, 30B, sanity-passed (V-only
referent +0.057 ≈ paper's positive 30B value → K-graft path trustworthy):
| cat | v_only | k_only | coupled | ind_k50 | ind_v50 | ind_k100 |
|---|---|---|---|---|---|---|
| sense | +0.027 | −0.283 | −0.024 | +0.005 | −0.103 | −0.014 |
| referent | +0.057 | −0.096 | +0.029 | +0.023 | −0.016 | +0.076 |
| stance | −0.017 | −0.211 | −0.128 | +0.035 | −0.165 | −0.151 |
VERDICT: VALUE is the operative axis. K-only actively HURTS all categories
(re-rotated keys perturb attention). Coupled/independent don't beat V-only on
referent or sense. ind_k100 referent +0.076 vs +0.057 = within noise (n=21).
The RoPE-addressing hypothesis (keys recover referent where values floored) is
NOT supported by UNIFORM grafting. The 0.6B hint (keys help referent) was noise.
BOUNDS/OPEN: (1) this is 30B where V-only ALREADY works on referent (+0.057) —
less room for keys to add; the exact "where-V-floored" test is 27B (referent
flat +0.004), not yet run (needs kv_graft 27B-path port). (2) COARSE UNTUNED
(uniform αK all layers) — uniform K-graft could average out layer-specific
effects; PER-LAYER/PER-HEAD key tuning (kv_graft supports it) might find a
key-profile that helps, but the strongly-negative K-only lowers that prior.

## Phase 2 CONCLUDED — Key-grafting doesn't help, even per-layer (30B) — 07-07
Per-layer key probe (add α_K=0.75 at ONE layer L on top of the working value
graft, referent, n=21 plants, 12 layers sampled). v_only baseline +0.0565.
Best layer (24) lift = +0.0111 (only 10/21 plants positive = coin flip);
k_only@L NEGATIVE at ALL 12 layers. NO layer meaningfully lifts referent.
COMBINED WITH the coarse uniform sweep (keys don't help, K-only hurts all
cats): the K/V exploration is CONCLUSIVE — VALUE is THE operative axis;
key-grafting does not recover referent uniformly OR per-layer. RoPE-addressing
hypothesis (keys carry "where to look" for retrieval) thoroughly UNSUPPORTED.
Per-head not run (per-layer clean-negative + k_only-negative-everywhere makes
it a dead direction; stopped per direction-against discipline). This STRENGTHENS
the paper: not "value is the axis we tested" but "value is THE axis — keys
checked uniformly + per-layer, don't help."

## Phase 2 SCOPING CORRECTION (Fable cross-check vs referent_recovery_microtest) — 07-07
My "VALUE is THE operative axis" was OVERCLAIMED — universal-sounding, but our
corpus has only ONE target morphology: semantic referent PHRASES (Nimbus=signup
funnel). A separate microtest (referent_recovery_microtest, 0.6B prospecting)
found that for a DIFFERENT shape — label→short TOKEN-like identifier (Coral→
userName, Azure→UTC) — KEY-grafting often WINS (k020/k010 best, GC up to ~0.8).
CRUCIAL: where the two studies OVERLAP (semantic policy_choice phrases) they
AGREE — the microtest reproduces our result (V-only dominates, K-only 0/30
positive). So the divergence is in UNTESTED territory (short identifiers), not a
contradiction. The microtest signal is statistically WEAK (0.6B, winner's-curse
best-of-15-policy selection, tiny cells n=2-4, K-only global mean still negative)
— it CANNOT move our claim toward "keys help", only stop it being UNIVERSAL.
Mechanism makes the flip plausible: keys=positional addressing; short-identifier
recovery = retrieval-by-address, not semantic reconstruction.
HONEST SCOPED CLAIM (replaces the earlier universal one):
> For SEMANTIC-referent recovery (decision/label → semantic phrase), VALUE-
> grafting is the operative axis — key-grafting doesn't help (uniform or
> per-layer, 30B rigorous), K-only hurts; RoPE-addressing unsupported FOR
> SEMANTIC-PHRASE TARGETS. Whether keys matter for SHORT TOKEN-LIKE IDENTIFIER
> targets (addressing/retrieval) is OPEN — only noisy 0.6B evidence hints yes.
TESTABLE PREDICTION: target morphology MODERATES K-graft utility (keys help for
short identifiers, inert for semantic phrases). WARRANTED: a focused 30B run —
add a short-identifier target lane, pre-register K/V/coupled sweep, report FULL
policy surface (not best-of) — before ANY general claim about keys.

## Lens free-divergence result — clean pre-registered NEGATIVE (27B, N=43) — 07-07
The free-generation fix (remove teacher-forcing pin → let A/B/E freely generate,
find fork, lens at fork) did NOT surface a vivid internal exhibit.
THREE-BUCKET: FORK_TOWARD_A = 0, SUBTLE_LEAN = 17, DISCONFIRMING = 26.
ZERO cases show a clean fork toward the correct concept with decisive margin +
decoding stability. (referent: 0/9/12; sense: 0/8/14.) Because the DISCONFIRMING
bucket was PRE-REGISTERED (no fork, OR fork-away, OR right-lean doesn't survive
3 decodings), the 26 can't be relabeled "subtle" — this is a genuine null, not a
heads-I-win. CONCLUSION: even the sharper free-gen method finds no dramatic
internal fork; the effect is genuinely subtle at the token level. The paper's
existing honest framing (lens = low-resolution aggregate corroborator, no vivid
per-example figure, not manufactured) STANDS and is STRENGTHENED — we tried the
method built to reveal it and it didn't appear. (Fable's pre-registered
disconfirming bucket is what makes this honest not spun.)

## Ops note (incident 28 addendum): process-gone + GPU-freed = COMPLETION *or* crash
Distinguish before diagnosing: check (a) output file exists, (b) log reached the
LAST expected item — if both, it FINISHED (terminate idle pod); only if neither,
it crashed. I mis-called a completed run as "crashed, no output" by reading
proc-dead as failure. RULE 26 addendum: on proc-gone, check finished-vs-crashed
explicitly; a done job means terminate the now-idle pod promptly.

## Effect-bound (placebo-controlled) — 27B, N=43, K=12 — 07-07
Teacher-forced shared gold[:12] pre-divergence window; A/B/E/placebo bit-identical
keys; placebo = norm-matched random-value graft (seeded derangement).
- E−placebo = +0.445, CI[+0.286,+0.612] EXCLUDES 0 → graft DECISIVELY beats random
  values; placebo−B = −0.428 CI[−0.590,−0.276] (random-value graft HURTS badly).
  = rigorous, spin-proof "alignment/content-specific, not generic perturbation."
- E−B = +0.017, CI[−0.033,+0.065] INCLUDES 0 → PRE-REGISTERED VERDICT: NULL. Over
  the pre-divergence window on 27B the graft doesn't clearly beat plain compaction.
  Mean weakly positive, consistent with 27B's modest effect (sense +0.05, referent
  +0.004 §7); underpowered at n=43 + narrow window (first 12 gold tokens ~ answer
  preamble). NOT a "graft does nothing" — content-specificity is strong (vs placebo).
NEXT (primary-strengthen): run SAME bound on 30B (strong model) — E−B should
resolve positive there; that's the placebo-controlled CI that strengthens the primary.

## Effect-bound 30B (K=12 pre-window) — SURPRISE, needs full-window re-run — 07-07
30B, N=43, K=12: E−B = −0.066 CI[−0.130,−0.007] EXCLUDES 0 NEGATIVE (graft HURTS
next-token pred over the first 12 gold tokens!); placebo−B = −0.306; E−placebo =
+0.241 CI[+0.033,+0.457] EXCLUDES 0 (graft still content-specific, beats random).
27B was E−B null; 30B is E−B negative — on BOTH the pre-window E−B ≤ 0.
INTERPRETATION (honest, not spin): K=12 measures the answer PREAMBLE, not the
content tokens where the +10-12pp gap-closure benefit lands. Graft helps the model
commit to right CONTENT (later tokens), slightly perturbs generic opening tokens →
K=12 catches perturbation, misses payoff. In teacher-forcing there's NO divergence
to avoid (all arms score identical gold tokens) so the K=12 cap (from the free-gen
design) was over-conservative. RIGHT measurement = FULL gold continuation (matches
gap-closure). RE-RUNNING at full window to validate: if E−B resolves POSITIVE over
full continuation → confirms primary + adds placebo control; if still ≤0 → REAL
tension with the gap-closure headline, must confront. Content-specificity (E−placebo
>0 both models) is solid regardless.

## ⚠️ CRITICAL — effect-bound 30B does NOT reproduce F1 positive gap-closure (07-07)
Full-window (K=48) 30B effect-bound, 43 sense+referent cases, UNIFORM α=0.75,
per-model 30B-generated summary. Computed the PAPER's gap-closure metric from the
same data:
- Overall 37% cases helped (E>B); ratio mean −0.034 median −0.016.
- sense: 27% helped, ratio −0.054 (NEGATIVE). referent: 48% helped, ratio −0.013.
- Compaction DID damage (A−B +1.96, 100% cases A>B) — setup valid.
- E beats PLACEBO (content-specific) but does NOT beat plain compaction B.
PAPER HEADLINE (F1, 66 probes, 30B): sense 64% helped, referent 81% helped. This
run is OPPOSITE. This is a POTENTIAL NON-REPRODUCTION of the core result and must
be resolved before trusting the primary or expanding (cross-arch HELD).
LIKELY EXPLANATIONS (to verify, NOT assume in our favor):
1. CONFIG: this uses UNIFORM α=0.75; paper §8 says uniform disrupts, per-layer
   CHAMPION needed. Original F1 config (champion? uniform?) must be checked;
   re-run effect-bound with champion to see if effect returns.
2. CASES/SUMMARY differ (43 free_div plants vs 66 F1 probes; per-model summary).
3. Original F1 less robust than presented.
ACTION: hold cross-arch, diagnose config vs original F1, re-run w/ champion, Fable
conceptual read, report to user. If primary not robust → paper (on README) OVERCLAIMS,
correct before external repro. HONEST — do not spin.

## ⚠️⚠️ APPARATUS INSTABILITY — F1 does not reproduce run-to-run (07-07, CRITICAL)
Re-ran the ORIGINAL F1 code (gap_closure_cat.py, α=0.75, same 43 cases) LIVE on 30B.
Does NOT cleanly reproduce the saved F1:
- sense: 64% helped BUT mean_gc −0.14 (saved +0.03) — SIGN FLIP on mean.
- referent: 71% helped, +0.10 (saved 81%/+0.04) — directionally ok, noisy.
- stance: 58% helped, +0.09 (saved 38%/−0.31) — DID NOT reproduce as null! flipped.
Same code, same inputs, DIFFERENT results → NOISY apparatus. Root causes:
1. GEN_TEMP=0.0 → summary is GREEDY (NOT sampled). But MoE argmax flips on near-tie
   tokens across hardware/runs → different summary cascade. (Ruled out sampling.)
2. Qwen3-30B-A3B is MoE — routing on bf16/hardware is NONDETERMINISTIC → summary
   AND teacher-forcing logprobs vary run-to-run/hardware. Ratio metric (small A−B
   denominators) AMPLIFIES the variance.
IMPLICATION: the saved F1 numbers are ONE sample from a noisy distribution,
presented in the paper as point estimates WITHOUT error bars. The dissociation
(esp. stance-null) is NOT stable run-to-run. Effect-bound wasn't necessarily buggy
— it may be within the noise band. THE APPARATUS MUST BE STABILIZED before ANY
conclusion. FIX: (a) FIXED summary (remove summary variance — use fixed_summaries),
(b) run N seeds → gap-closure mean±CI per category, (c) re-establish dissociation
WITH error bars or honestly report it's noisier than presented. Paper (on README)
currently OVERSTATES robustness — must fix before external repro. HOLD everything
downstream. Do NOT spin — this is a real problem with the measurement.

## ✅ RESOLVED (Fable + user push) — effect is REAL; instability was the MEAN-RATIO estimator, not the effect (07-07)
Mined the TWO saved runs (results/gap_closure_cat vs _live, 67 common probes) — NO
GPU needed (I was thrashing on env/GPU re-runs; the answer was on disk).
FINDING: the effect REPRODUCES on every ROBUST metric; only mean-of-ratio swings.
  raw E−B:      referent +0.156/+0.125, sense +0.062/+0.047, stance −0.026/+0.002
  median ratio: referent +0.118/+0.101, sense +0.033/+0.090, stance −0.068/+0.019
  % helped:     referent 81/71, sense 64/59, stance 38/54
  mean ratio (BROKEN): sense sign-flips, stance −0.308 vs +0.091.
Both runs, all robust metrics: SAME dissociation — graft raises gold logprob for
referent+sense, ≈0 for stance. The "−0.31 stance-null" = TWO probes with near-zero
|A−B| denominators (Cauchy blow-up). Condition |A−B|>0.5 → even mean ratio stable
(stance −0.029/+0.039 ≈ 0 both).
METRIC FIX (Fable): DEMOTE mean-ratio. Primary = raw lp_E−lp_B (bounded) + %-helped;
median ratio secondary; if ratio kept, winsorize/condition on |A−B| + report n.
NEVER report bare mean-ratio again.
HONEST POSITION: effect REAL, reported with UNSTABLE estimator + NO error bars.
README overstates PRECISION (point estimates, dramatic −0.31, no CI), NOT existence/
direction. = numbers correction + error bars, NOT retraction.
NOTE: this supersedes the "APPARATUS INSTABILITY / does NOT reproduce" alarm above —
that alarm conflated "mean-ratio swung" with "effect unstable"; they're different.
The MoE/summary-cascade nondeterminism is real but 2nd-order (lp_A varies ~0.03).
NEXT: (1) recompute paper table on robust metrics (both runs, side by side, no GPU);
(2) T1 fixed-summary 2-run determinism floor; (3) T2 N=5 → mean±CI; (4) cross-arch
uses raw E−B not mean-ratio. Effect-bound probe uses a DIFFERENT teacher-forcing
(raw E−B negative there) — retire it, trust gap_closure_cat.

## PRIMARY EFFECT with proper CONFIDENCE (bootstrap CIs over probes, raw E-B, live/5.13) — 07-07
Apparatus is DETERMINISTIC within-env (live1==live2 exact) → confidence = bootstrap
over the PROBE SAMPLE (n=21-24/cat), N=10000:
- referent: +0.125 CI[+0.030,+0.218] EXCLUDES 0 → REAL significant effect. 71% helped.
- sense:    +0.047 CI[−0.038,+0.131] SPANS 0 → suggestive but UNDERPOWERED (n=22).
- stance:   +0.002 CI[−0.036,+0.049] SPANS 0 → genuinely NULL (as claimed).
HONEST: effect REAL for referent (significant), suggestive-underpowered for sense
(logprob metric; judged +12pp carries it), null for stance. The clean "sense+referent
recovered" dissociation is MORE NUANCED on the logprob metric than the paper implies —
only referent is significant there. n=21-24/cat is TOO SMALL for tight CIs = the real
limitation, FIXABLE by more probes.
STANDARD METRIC GOING FORWARD: raw E-B (bounded) + %-helped + bootstrap 95% CI over
probes. Never bare mean-ratio. The JUDGED +12pp is now load-bearing for sense →
needs its own bootstrap CIs (next audit).
EXTENDED PLAN now has statistical PURPOSE: more probes/scenarios to POWER the effect
(esp. sense), CIs throughout. Cross-arch, champion-tune, judged-audit all on this footing.

## ⚠️ K/V conclusion ALSO used the broken ratio metric (user caught it, 07-07)
kv_layer_probe.py line 19: gap_closure=(E-B)/(A-B) — SAME unstable mean-ratio.
So "K-only hurts all categories (sense −0.283, referent −0.096, stance −0.211)"
and "per-layer negative" are RATIO-INFLATED artifacts. Recomputed on RAW E-B:
- v_only referent: +0.120 CI[−0.001,+0.228] — REAL value effect (matches F1 +0.125).
- k_only per-layer: ALL near-zero (−0.020..+0.016), tiny — keys are ~NEUTRAL,
  NOT dramatically harmful. The dramatic "keys hurt −0.28" was the estimator.
CORRECTED K/V read: "value is THE operative axis" HOLDS (value positive, keys
don't HELP), BUT "keys actively hurt" is FALSE on robust metric — keys ~neutral.
Per-layer negatives are real but tiny perturbations, not impossible.
ACTION: the paper §10 K/V section needs the SAME robust-metric correction as F1 —
replace ratio numbers (−0.283 etc.) with raw E-B; reframe "K-only hurts" →
"keys ~neutral, don't help". Add to task 29 (paper correction). GENERAL LESSON:
EVERY result using (E-B)/(A-B) mean-ratio is suspect — audit all of them on raw E-B.

## Cross-arch Qwen2.5-32B: NEGATIVE raw E-B, diagnosing (07-08)
Qwen2.5-32B (dense GQA, validation model) with fixed Sonnet summary: raw E-B AGGREGATE
-0.28 CI[-0.37,-0.20] EXCLUDES 0 (graft HURTS), pre_gap A-B +0.61 (compaction DID
damage, setup valid). Significant HURT (not null). CANDIDATES: (1) architecture doesn't
transfer (user's hypothesis, live); (2) cross-arch harness bug specific to Qwen2.5 template
(region detection). RULED OUT: alignment (difflib aligns 100%, 0 dropped on Qwen2.5) and
fixed-summary (alignment perfect). DECISIVE TEST running: trusted gap_closure_cat.py on
Qwen2.5 (bypasses cross-arch harness). Don't conclude from 1 model — exploring all 7.
ALIGNMENT NOTE (user flagged difflib): build_alignment difflib is fragile overkill (drops
<8-tok runs silently) but VERIFIED NOT compromising results (100% aligned everywhere
checked). Being simplified to direct span map (task 31) for robustness, equivalence-gated.

## ✅ Cross-arch datapoint 1: Qwen2.5-32B grafts NEGATIVE — REAL ARCHITECTURE (07-08)
Value graft REVERSES on Qwen2.5-32B (dense GQA). Confirmed by 3 independent runs:
cross-arch fixed-summary raw E-B -0.28, cross-arch self-gen -0.32, and TRUSTED
gap_closure_cat.py (exact F1 code, model-gen summary) referent -0.26/sense -0.38/
stance -0.25 (all significantly negative, %pos 4-19%). Harness VALIDATED (trusted
matches cross-arch). NOT a bug — Qwen2.5 genuinely grafts negative where Qwen3-30B-
MoE grafts +0.12. INTERPRETATION: the effect is ARCHITECTURE-SPECIFIC and can
REVERSE — strong evidence it's a real mechanism, NOT a generic artifact (an
artifact wouldn't flip sign by architecture). Map entry: Qwen3-MoE +, Qwen2.5-dense −.
Keep exploring (Gemma next, parallel pod2). Note: Qwen2.5 vs Qwen3 differ in
dense-vs-MoE AND generation — cause of the reversal is open (n=2).

## 🚨 NOTABLE: cross-arch POSITIVE CONTROL FAILED — fixed-summary suspected of breaking the graft (07-08)
Ran Qwen3-30B-A3B (our F1 model, known +0.156 referent / +0.062 sense via
gap_closure_cat) through the CROSS-ARCH HARNESS with the FIXED SONNET summary.
RESULT: referent +0.004 (CI spans 0), sense −0.147 (CI EXCLUDES 0, NEGATIVE),
agg −0.073. pre_gap +0.51 (compaction did damage, valid). The harness does NOT
reproduce the known positive — it's null-to-negative.
THE ONLY DIFFERENCE from F1: summary source. F1 = model's OWN GENERATED summary;
cross-arch = FIXED SONNET (foreign) summary. STRONG SUSPICION: the fixed-summary
design SUPPRESSES the graft. MECHANISTIC FIT: the graft re-injects the model's
write-time state from GENERATING its own summary (its own compression act) — a
foreign summary the model merely READ may not carry that continuity. Would mean
the fixed-summary sweep is BIASED toward null/negative → understates the effect →
Qwen2.5's "architecture" negative is partly suspect (though Qwen2.5 was ALSO
negative on the trusted model-gen path, so that one may be real).
ISOLATION TEST RUNNING: cross-arch harness SELF-GEN summary on Qwen3-30B. If ~+0.156
→ harness OK, FIXED SUMMARY is the culprit → switch whole sweep to self-gen (accept
Fable's summary-quality confound, handle via pre-gap normalization). If still null →
deeper harness bug. IMPLICATIONS: (a) Mistral currently running on FIXED summary =
biased, needs self-gen re-run; (b) all fixed-summary sweep numbers suspect until
resolved; (c) POSSIBLE MECHANISTIC FINDING: graft needs the model's OWN summary
(would be a real insight about how ValueGraft works, pending confirmation).
LESSON: always run a POSITIVE control (reproduce a known result) before trusting a
new harness — the negative-agreement (Qwen2.5) was NOT sufficient validation.

## ✅✅ RESOLVED + MECHANISTIC FINDING: graft needs the model's OWN summary (07-08)
Isolation CONFIRMED. Qwen3-30B via cross-arch harness:
- FIXED Sonnet summary: referent +0.004 (null), sense −0.147 — POSITIVE CONTROL FAILED.
- SELF-GEN (own) summary: referent +0.136 CI[+0.034,+0.23] 81% helped, sense +0.045
  64% helped, agg +0.090 CI[+0.022,+0.154] SIGNIFICANT_POSITIVE — MATCHES F1
  (+0.156/81%, +0.062/64%).
Same model+harness, only summary source differs → the FIXED (foreign) summary
SUPPRESSES the graft; the model's OWN generated summary reproduces the effect.
CONCLUSIONS: (1) HARNESS VALIDATED (reproduces known positive on self-gen).
(2) FIXED-SUMMARY DESIGN BROKEN → sweep switches to SELF-GEN summaries.
(3) MECHANISTIC FINDING (real, not speculation now): ValueGraft re-injects the
write-time state of the model's OWN summarization ACT — a summary the model merely
READ doesn't carry the recoverable continuity. Enriches the paper's mechanism.
MAP status: Qwen2.5-dense NEGATIVE is REAL (self-gen −0.32 AND trusted model-gen
−0.30, both). Qwen3-MoE POSITIVE. Mistral (fixed-summary ~null) = BIASED, re-run
self-gen. CONFOUND (Fable): self-gen summary quality varies across models → report
pre_graft_gap + summary token-count per model, use conditioned ratio.

## POSITIVE CONTROL PASSES on correct model (07-08) — effect confirmed real
After the wrong-model saga (incident 34), the trusted apparatus (gap_closure_cat.py,
difflib, self-gen) on the CORRECT model Qwen3-30B-A3B-Instruct-2507 reproduces the effect:
  orig c01-c12: referent raw_EB=+0.1246 (71% helped), sense +0.047, stance +0.002 (~null).
The dissociation (referent>sense>stance~0) reproduces exactly. The known +0.136 is confirmed.
CAVEAT: new convs c13+ show WEAKER referent (+0.009 ~null), sense +0.083, stance -0.096 -->
the doubled corpus DILUTES rather than strengthens on referent. Checking pre_graft_gap (A-B)
on new convs to decide: small gap = effect-tracks-damage (fine); large unrecovered gap =
new convs are lower-quality rendering (fix before wide).

## ⚠️ NATIVENESS CONFOUND (Fable, 07-08) — potentially paper-fatal, test BEFORE wide spend
Fable's general read caught a confound one level up from the wrong-model: the +0.1246 positive
control on c01-c12 may be reliable BECAUSE the original corpus was generated IN-CONTEXT by a
Qwen-family model (old MLX pipeline = Qwen3-4B). So c01-c12 is NATIVE to Qwen, FOREIGN to
everyone else — the SAME condition that killed the new convs (foreign replies -> under-recover).
IF SO: +0.1246 is a Qwen-native artifact, and on a fixed shared corpus the cross-arch "SIGN"
would track PER-MODEL NATIVENESS, not attention geometry -> a gorgeous sign-map that's really a
NATIVENESS map. Same ghost class as the wrong-model, one level up.
DECISIVE PRE-SPEND TEST: run the c01-c12 referent positive control on ONE non-Qwen arch
(Mistral/Llama). If raw_EB COLLAPSES there like the new convs did -> nativeness dominates ->
redesign before spending on 16. One model, existing corpus, hours not days.
MECHANISM REFINEMENT: my "foreign write-side" hypothesis is the WEAKER half; the DOMINANT term
is likely the TARGET side — E's continuation is ALSO foreign, and the graft has no reason to
raise the probability of text the model wouldn't produce. Also can't yet exclude: the 1.12 new-conv
A-B gap is generic distributional surprise (foreign text lower-prob), NOT graft-shaped continuity.
CHEAP DISTINGUISHERS (data in hand): (1) nativeness regression — score each conv by mean per-token
logprob of its assistant replies under the test model, regress raw_EB on it; (2) dissociation
decomposition — does the new-conv A-B gap have the referent>sense>stance signature? if flat, it's a
different non-graftable gap.
OTHER RED FLAGS (Fable): per-arch value-graft INDEX ALIGNMENT must be re-verified PER MODEL (differs
by tokenizer/arch; silent misalignment = plausible garbage — what crashed before). NOISE FLOOR:
stance +0.002, new +0.009 -> sign resolution near zero is marginal at n=12; a sign inside the noise
band isn't a sign (need per-conv bootstrap CIs). Attention-geometry predictor must be computed
INDEPENDENTLY of the outcome (pre-register) or it's post-hoc fitting.
PATH (Fable): run wide on reliable c01-c12 BUT gate on (i) the one non-Qwen nativeness control +
(ii) the ~free nativeness/dissociation analysis FIRST. REJECT re-rendering the corpus per-model
(16 native corpora = new confound). If nativeness dominates, the honest strong paper is the
mechanism + dissociation + the SCOPE CONDITION itself (recovery needs self-native context).
STRONGEST PAPER (Fable): "A model's own write-time value vectors can be re-injected to recover
evicted semantic continuity — specifically referent binding — but only for on-distribution
context, and the sign of recovery is predicted by attention geometry across architectures."

## DESIGN v2.1 VALIDATED on native Qwen (07-08) — the redesign works
Per-model native render (Qwen3-30B-A3B-Instruct-2507 generates its OWN in-context replies + own
self-gen summary; SHARED gold continuation) on c01-c12 REPRODUCES the effect:
  referent CI [+0.012, +0.195] (mid +0.10, headroom 1.71) — EXCLUDES ZERO, matches the pre-rendered
  +0.12. sense +0.03 (weak-positive). stance -0.05 (~null, headroom 0.51). ruled_out FLOORED
  (headroom 0.18<0.3, correctly EXCLUDED — summary preserved it = nothing to recover, NOT "harm").
  evicted_fact null. identity_ok+alpha0_ok pass. reply_covariates captured (mean 316 tok/reply).
=> The matched-scaffold model-filled design is SOUND: native replies reproduce the effect, the
dissociation holds, and the headroom gate correctly floors preserved-content categories. Gate GREEN
on the design. Remaining before wide: (1) render scaling (batched decode + snapshot-replay — render
was ~2.5hr/12conv), (2) finalize pre-registered geometry hypothesis (fix QK-norm detect first).

## Cross-arch (in progress, 2026-07-08) — first full-run result + two methodology corrections

**Mistral-Small-24B-Instruct-2501 (no-QK-norm, dense), n=12 convs, conversation-clustered CIs:**
- referent [+0.010, +0.063] — POSITIVE, excludes 0
- sense    [-0.067, -0.005] — NEGATIVE, excludes 0
- stance   [-0.059, -0.014] — NEGATIVE, excludes 0
- ruled_out / evicted_fact — include 0 (null)
Smoke identity_ok/alpha0_ok pass (valid run). Signature DIFFERS from Qwen (referent+/sense+/stance-null):
Mistral recovers referent but the graft HURTS sense+stance. Directly relevant to H1: a no-QK-norm model
with a POSITIVE referent challenges H1's "no-QK-norm → null/negative referent" prediction — but the
sense/stance flip shows the architectures act differently. PRELIMINARY: n=12 is a borderline cluster
count; the Qwen 24-conv gate + the QK-norm ablation are the anchors. Not to be over-read as one model.

**Methodology correction 1 — CI clustering unit.** The headline raw_EB_ci is now the CONVERSATION-
clustered bootstrap (was plant-clustered = anti-conservative; plants within a conv are correlated).
Plant-level kept as raw_EB_ci_plant; ci_method records the unit. On Mistral the conv CI was ~the same
width as plant (low between-conv correlation), so the result held — but this is not guaranteed per model.

**Methodology correction 2 — OLMo-2 "empty alignment" was a MISDIAGNOSIS.** OLMo-2 alignment works
(108/0 smoke-align). The real cause of its ERROR: the absolute competence floor (task_lpa_floor=-8.0)
excluded EVERY OLMo plant (its gold logprobs sit lower); the old code guessed "empty alignment." New
counters (empty_alignment_convs / short_gold_drops / task_excluded_plants) now name the true cause.
OPEN: to get an OLMo result the floor likely needs to be per-model/relative — a methodology call.

## Judged metric conv-clustered CIs (task #30 / Fable #2, 07-09): sense holds; significance FLIPS by metric
Judged = Sonnet-5 meaning re-judge (RECOVERED/PARTIAL/MISSED = 1/0.5/0), pooled-graft (a0.25+a1.0)
− Compacted, per category; 12 convs (c01-c12), 64 plants. Conversation-clustered bootstrap (resample
the 12 convs, nboot=20000):
- SENSE     +12.0pp  CI [+2.2, +22.9]  — EXCLUDES 0 (the load-bearing arm survives; Fable's flagged
  uncorrected +12pp holds up under clustering).
- REFERENT  +9.7pp   CI [-6.9, +26.2]  — spans 0 (only n=18 referent plants; wide).
- STANCE    +3.3pp   CI [-5.4, +12.0]  — null (correct).
SIGNIFICANCE FLIPS BY METRIC: logprob/gap-closure metric = REFERENT significant, sense underpowered;
JUDGED metric = SENSE significant, referent not. So each dissociation arm is significant on ONE metric,
not both — real but metric-dependent, NOT a clean "both instruments agree." The paper must state this
honestly. Reproducible: scripts/judged_bootstrap.py.

## Arch pole: Qwen3-32B (dense, QK-norm) — graft NULL-to-HARMFUL; strong within-Qwen MoE-vs-dense reversal (2026-07-09)
Qwen3-32B, n=24 conv, conv-clustered, smoke identity_ok/alpha0_ok pass (valid):
- referent [-0.063,+0.010] NULL · sense [-0.091,-0.008] NEG · stance [-0.139,-0.051] NEG ·
  ruled_out [-0.130,-0.063] NEG · evicted_fact [-0.099,-0.017] NEG.
The graft is null-to-HARMFUL across the board on dense Qwen3-32B — the OPPOSITE of Qwen3-30B-A3B (MoE:
referent +0.10, sense+). The primary pre-registered de-confound (same vendor, same QK-norm, MoE vs
dense) shows a STRIKING reversal: MoE recovers meaning, dense is harmed. Architecture-specificity is
real + strong — likely the paper's headline mechanism finding.
CAVEATS: PRELIMINARY — gated behind the fresh-conv reproduction of the +0.10 anchor (if the anchor
doesn't reproduce on held-out convs, the map is built on sand). One dense model; Mistral (dense,
no-QK-norm) was referent+, so NOT "dense=negative" — it's model/architecture-specific, not a clean
dense/MoE law. n=24. Renders NOT banked (old-harness run, pre-checkpointing).

## Arch pole: Qwen2.5-32B (dense, no-QK-norm) — graft STRONGLY HARMFUL; the map skews MoE-positive / dense-negative (2026-07-09)
Qwen2.5-32B, n=24 conv, conv-clustered, smoke passes: referent [-0.303,-0.157] STRONG NEG · sense
[-0.531,-0.317] very strong NEG · stance [-0.223,-0.133] NEG. The graft is HARMFUL across the board.
ARCHITECTURE MAP SO FAR: clearly POSITIVE only on the MoE anchor (Qwen3-30B-A3B +0.10) + a weak
Mistral referent (+0.035); NULL-to-STRONGLY-NEGATIVE on the dense models (Qwen3-32B null/neg,
Qwen2.5 strong-neg, phi-4 null). The graft ranges strongly-positive (MoE) → strongly-negative (dense)
— a dramatic architecture-specific reversal.
⚠ ROBUSTNESS FLAG: the effect is clearly positive mainly on the ONE model it was developed on. This
makes the FRESH-CONV REPRODUCTION of the anchor CRITICAL — if +0.10 doesn't hold on held-out convs,
the effect is fragile/model-specific rather than a real recover-meaning phenomenon. The reproduction
(b0/b1/b2, running) is the disambiguator. Do NOT over-sell "architecture-specific reversal" until the
anchor reproduces.

## ⚠ Fresh-conv reproduction (first block read, n=12): POSITIVE CONTROL FAILED — +0.10 did NOT reproduce (2026-07-09)
Block design, Qwen3-30B-A3B, native render, per-token, RELATIVE floor, conv-clustered, CENTRAL pooled floor
(scripts/block_analysis.py over b0=c01-c12 + b1=c13-c24):
- BASELINE (c01-c12, POSITIVE CONTROL): referent +0.012, CI [-0.070,+0.091] — FAIL, spans 0. The +0.10
  (original CI [+0.012,+0.195]) did NOT reproduce on its OWN convs. block_analysis verdict: "RUN IS SUSPECT."
- FRESH (c13-c24, held-out): referent -0.021, CI [-0.082,+0.041] — NULL, and at HIGH headroom (damage
  present, graft doesn't recover it = a real null, not a "nothing to recover" artifact). All fresh
  categories null. Morphology 100% sem_phrase both cells (not a drift artifact).
INTERPRETATION OPEN — for Fable un-anchored: (a) the +0.10 was FRAGILE/noisy and doesn't reproduce
(honest, sobering — headline not robust), OR (b) a RECENT HARNESS CHANGE (relative floor / OLMo-fix /
checkpointing / conv-CI) altered the apparatus vs the original run = a REGRESSION to find + fix. MoE
render is hardware-nondeterministic (adds noise). Positive-control failure => CANNOT trust the fresh read
until resolved. b2 (c25-c36) rendering for n=24 but MOOT if the apparatus is suspect. DO NOT conclude
"effect fragile" until Fable adjudicates regression-vs-fragile.

## REPRODUCTION VERDICT (Fable un-anchored, high-conf, 2026-07-09): +0.10 is FRAGILE, NOT a regression → paper pivots to SENSE-led
Fable investigated (code, git, b0 traces) and ruled OUT regression:
- The +0.10 native run was NEVER banked (only prose in FINDINGS) — incident-#38's unbanked-render sin; un-reproducible.
- Apparatus mechanically sound: identity_ok/alpha0_ok PASS (a regression would trip them); recent changes
  (checkpointing/relative-floor/SC_CONV_START) touch resume/gating, not the forward path; floor inert on c01-c12; ablation off.
- Graft path INTACT: sense (+0.040), stance (-0.086), headroom (1.76) all REPRODUCE the original nearly exactly.
  Only REFERENT moved (+0.10→+0.012). Sense uses the SAME graft path — if it broke, sense would move. It didn't.
- Statistically indistinguishable: b0 per-conv referent stdev 0.148, SE~0.043; +0.012 == the original CI's LOWER BOUND
  [+0.012,+0.195]; MoE hardware-nondeterministic render → regression-to-the-mean from a barely-significant CI.
REFRAME (paper): referent-logprob headline is UNDERPOWERED/render-fragile (the +0.10 was a lucky unbanked draw; b0 is
the first BANKED native referent = null). SURVIVES + becomes the load-bearing arm: judged SENSE +12pp [+2.2,+22.9]
(banked, reproducible) — AND sense raw_EB reproduced here (+0.040). Paper = SENSE-LED; referent = fragile/underpowered
dissociation signal + the significance-flips-by-metric caveat. ARCH MAP: dense-negative poles are REAL, but the
MoE-positive reference pole is the fragile one → "one fragile positive vs several solid negatives"; report with
per-render variance, don't lead with it. DON'T chase a phantom regression; DON'T retract; DO bank every render.
NEXT: 3× independent re-render of c01-c12 (bank all) to state referent render-variance honestly (Fable predicts pooled ~+0.03-0.05, boundary).
