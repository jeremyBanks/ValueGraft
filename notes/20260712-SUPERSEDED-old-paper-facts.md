# FACTS SHEET — traced numbers for the paper (compiled 2026-07-09, quick-take pass)

Every claim below carries its source. Status of the program: data phase still
concluding (block-design fresh-conversation reproduction + arch poles in flight
on pods as of 07-09 00:21 EDT). This sheet reflects what is BANKED.

## Headline behavioral effect (Qwen3-30B-A3B-Instruct-2507)

**Metric standard (locked):** raw_EB = lp_E − lp_B teacher-forced on shared gold
continuation + %-helped + bootstrap 95% CI; mean-of-ratio RETIRED (Cauchy-unstable).
[FINDINGS "RESOLVED (Fable + user push)", DECISIONS 07-08 01:00]

**Logprob metric (bootstrap over probes, n=21-24/cat, live run):**
- referent: raw E−B +0.125 CI [+0.030,+0.218] EXCLUDES 0, 71% helped — significant
- sense: +0.047 CI [−0.038,+0.131] spans 0 — suggestive, underpowered
- stance: +0.002 CI [−0.036,+0.049] — null
[FINDINGS "PRIMARY EFFECT with proper CONFIDENCE"]
Range across 3 runs: referent +0.125..+0.156 (71-81% helped); sense +0.047..+0.062
(59-64%); stance −0.026..+0.002 (38-54%). [notes/2026070856-robust-f1-numbers.md]

**Judged metric (Sonnet-5 meaning re-judge, RECOVERED/PARTIAL/MISSED=1/0.5/0, pooled
graft a0.25+a1.0 − Compacted, 12 convs c01-c12, 64 plants, conv-clustered bootstrap
nboot=20000):**
- SENSE +12.0pp CI [+2.2,+22.9] EXCLUDES 0 — significant
- REFERENT +9.7pp CI [−6.9,+26.2] spans 0 (n=18 plants, wide)
- STANCE +3.3pp CI [−5.4,+12.0] — null
[FINDINGS final entry, task #30; scripts/judged_bootstrap.py]

**⚠ SIGNIFICANCE FLIPS BY METRIC:** logprob = referent-significant/sense-underpowered;
judged = sense-significant/referent-not. Each dissociation arm significant on ONE
metric. Dissociation real but metric-dependent — MUST state honestly. [FINDINGS; STATE 07-09]

**Judged rates (lenient, PARTIAL=0.5):** stance 96/93/96 (+2pp), sense ~100*/46/58
(+12pp), referent ~100*/17/26 (+10pp) for Original/Compacted/Graft. *A-ceiling cells
tiny (sense n=1, referent n=2). Strict cut (PARTIAL=miss): +4/+9/+8pp — pattern holds.
[FINDINGS F1 + robustness]

**Direction agreement (gap-closure %-helped, 66 probes):** stance 39% (null/neg),
sense 64%, referent 81%. Magnitude smaller on exact-token metric than meaning-judge —
predicted by "recovers meaning not verbatim form". [FINDINGS F1 corroboration]

**Graft dose:** α_V=0.75 default; α_K=0 (values only). [FINDINGS methods note]

## Scale/model bounds
- Dissociation does NOT replicate at 4B (stance 87% helped +0.19; ordering muddled).
  F1 = large-model finding. [FINDINGS F1 scale-dependence]
- Qwen3.6-27B: sense +0.050 (59%) replicates; stance −0.201 (21%) null/neg; referent
  +0.004 FLAT (48%) — cross-model non-replication of referent, cause open.
  [FINDINGS same-model confirmation]
- Honesty effect (F2) DOES replicate 4B→30B and 4-bit→bf16 (opposite small-scale
  behavior to the dissociation).

## Honesty effect (F2, banked, strong — PRIMARY POSITIVE)
CORRECTED (2026-07-09, recomputed from results/phase2_30b_scored.json; see CLAIMS F2):
Decoy fabrication: Compacted **79%** vs write-time-KV (H-pack/packed) **12%**. On evicted facts,
H-pack converts fabrication **67% → admission 96%** (fab 4%) but recalls **0/24** — it buys
**honesty, not recall**; it does NOT restore accuracy. ⚠ The old "38/48 = 79% most accurate"
figure is a DEAD OVERCLAIM (reproduces nowhere) — NEVER ship it. Replicated across precision +
scale. NOTE: H-pack = packed layout, keys+values retention — a COUSIN of the value graft, not the
same arm. Decomposition (recomputed): layout 79→42% decoy fab; write-time-KV within layout 42→12%;
decoy admissions 5/24→21/24. [CLAIMS F2-1/F2-2/F2-2b/F2-3; FINDINGS F2; DECISIONS 07-06]
Scope: mid-task agentic compaction; washes out on LongMemEval personal-QA framing at
30B (refusal calibration covers it). Packing buys honesty, not memory. [notes 07-05]

## Tuning (F3)
α=1.0 full replacement can catastrophically break a task (chain s1 0/4 vs 4/4 all else);
per-layer guard-validated champion cures it (16/16). Layer config passed wrong-conversation
guard; 57-slot mask FAILED it (content-independent artifact) — guards discriminate.
30B optimum α ~0.75 (bf16 sweep); 4B optimum α=0.25; dose-response INVERTS with scale.
α>1 "steering" explored at 30B (4-bit sweep peak; bf16 confirms 0.75 peak, decline past 1.0).
[FINDINGS F3; DECISIONS 07-05 eve/07-06]

## K/V axis — CLOSED
Keys ~NEUTRAL on robust metric (raw E−B near-zero all layers; the dramatic "keys hurt
−0.28" was the retired ratio estimator). Keys don't HELP (uniform + per-layer, 30B);
v_only referent +0.120 CI[−0.001,+0.228] matches F1. K-graft technically sound (RoPE
re-rotation validated to fp32, cosine 0.99999982). Scope: semantic-phrase targets;
short-identifier morphology untested at scale (0.6B noisy hint only). Value is the
operative axis. [FINDINGS Phase-2 entries + ratio correction]

## Mechanistic finding: graft needs the model's OWN summary
Same model+harness, only summary source differs: FIXED Sonnet summary → referent +0.004
(null), sense −0.147 (neg). SELF-GEN summary → referent +0.136 CI[+0.034,+0.23] 81%
helped, agg +0.090 CI[+0.022,+0.154] — matches F1. The graft re-injects the write-time
state of the model's own summarization act; a merely-read foreign summary doesn't carry
the recoverable continuity. [FINDINGS ✅✅ RESOLVED 07-08]

## Nativeness scope condition (design-shaping)
Foreign (non-native) assistant replies collapse the effect: fresh convs c13+ on OLD
Qwen-4B-rendered corpus: referent +0.009 ~null (vs +0.125 on c01-c12). Per-model NATIVE
render (model generates own replies + own summary from shared scaffold; shared gold)
REPRODUCES: referent CI [+0.012,+0.195] mid +0.10 on c01-c12. Nativeness is PART OF THE
MECHANISM (deployment-realistic), not just a confound. Design v2.1 = matched-scaffold,
model-filled. raw_EB on shared gold cancels target-side nativeness.
[FINDINGS nativeness-confound + design-v2.1-validated entries; MASTER-PLAN v2/v2.1]

## ⚠ OPEN VALIDITY ITEM (must be stated in any current draft)
The headline rests on the 12 hand-authored scenarios c01-c12. The fresh augment
(c13-c54) did NOT carry the effect on the old (foreign-reply) render (~+0.009); the
native-render fix was validated ON c01-c12, and a BLOCK-design held-out reproduction
(c01-c12 baseline vs c13-c36 fresh, native render, pooled analysis, pre-committed
stopping rule) is RUNNING as of 07-09. Until it lands, "effect generalizes beyond the
original 12 scenarios" is UNVERIFIED. [STATE 07-09; INCIDENTS #37]

## Cross-architecture (preliminary, self-native, 12 convs each, conv-clustered CIs)
- Qwen3-30B-A3B-Instruct-2507 (MoE, QK-norm): referent + (CI excl 0), sense weak+,
  stance null — the anchor/positive control. [FINDINGS design v2.1]
- Mistral-Small-24B-Instruct-2501 (dense, no QK-norm): referent +0.035 [+0.010,+0.063]
  POSITIVE; sense −0.037 [−0.067,−0.005] NEGATIVE; stance −0.037 [−0.059,−0.014]
  NEGATIVE (but stance+ruled_out FLOORED by headroom gate — treat per gates). Aggregate
  −0.012 [−0.027,+0.002] null. DIFFERENT signature than Qwen. [results/cross_arch_done JSON;
  note: FINDINGS quotes referent [+0.010,+0.063]; JSON cluster CI [+0.0097,+0.0629]]
- microsoft/phi-4 (dense, no QK-norm): aggregate −0.064 [−0.099,−0.027] SIGNIFICANT_NEGATIVE;
  referent −0.053 [−0.117,+0.008]; sense −0.041 [−0.126,+0.035]; stance −0.048 [−0.079,−0.012].
  [results/cross_arch_done/microsoft__phi-4.json]
- Qwen2.5-32B (dense, no QK-norm): grafts NEGATIVE — self-gen agg −0.32; trusted
  gap_closure_cat referent −0.26/sense −0.38/stance −0.25 (%pos 4-19%). Confirmed by 3
  independent runs. REAL architecture reversal. [FINDINGS cross-arch datapoint 1]
- Sign REVERSES across architectures; signatures differ per-category. Architecture-
  specificity (not QK-norm) is the cross-arch finding. n small everywhere; preliminary.
  [STATE 07-09]

## H1 (QK-norm) — PRE-REGISTERED NULL
H1 (QK-norm presence predicts positive referent sign) FROZEN in PREREGISTRATION.md
before the sweep. FALSIFIED: Mistral (no QK-norm) referent POSITIVE [+0.010,+0.063] =
opposite of prediction; within-model QK-norm ablation BROKE generation (uninterpretable).
Report as pre-registered null. [STATE 07-09; INCIDENTS #37]
Pre-flight exclusions (before any results): 5 of 16 models multimodal wrappers
(Gemma-4 pair, Gemma-3-27b, Qwen3.6-35B/27B) → PRIMARY de-confound reduced to one
within-vendor pair. MLA (DeepSeek/Kimi) excluded by design (no per-head values).
[PREREGISTRATION deviation note; MODEL-QUEUE]
Gates: headroom floor 0.3 (floored category = nothing to recover, excluded not "harm");
task-competence floor (absolute −8.0; pre-registered relative median−3·MADN amendment
for OLMo-class scale differences — verified no-op on already-scored models).
[PREREGISTRATION GATE #3 amendment]

## Agent-task scope boundary (F4)
SWE-bench: 0/7 even oracle-mode (capability floor of 30B, vendor-card-predicted —
model card lists Aider-Polyglot 55.1%, no SWE-bench for non-Coder variant).
tau2-banking: sessions structurally 3-4K tokens regardless of simulator quality — too
short for genuine eviction at any threshold; retired. Chain tier: Compacted baseline
4/4 all seeds = compaction-robust → no damage to repair (but champion cures α=1.0
collapse there, 16/16). SWE-Gym offline proxy (75 real OpenHands trajectories): tuned
graft +0.0156 nats next-action recovery, 45/75 wins, CI [0.005,0.027], ~10% of
compaction damage (d≈0.33). The operative regime (hard enough that eviction matters,
easy enough to use recovered context) is narrow and under-served. [FINDINGS F4;
DECISIONS 07-05 eve stage-2, 07-07 08:15]

## Negative controls
Wrong-conversation + shuffled-value grafts crater (~1-2 nats, both scales); α=0
bit-identical to Compacted (identity checks in every run); placebo (norm-matched
random-value graft, seeded derangement): E−placebo +0.445 CI[+0.286,+0.612] on 27B,
+0.241 CI[+0.033,+0.457] on 30B — content-specific, not generic perturbation.
Effect-bound E−B on narrow K=12 pre-window: 27B null, 30B negative — measures answer
preamble not content tokens; retired in favor of full-window gap_closure_cat.
[FINDINGS effect-bound entries; DECISIONS]

## J-lens (brief mention only, per owner 07-09)
Aggregate-level corroboration only (alignment-sensitive: shifted control craters;
α<1 beats α=1 internally; sparse-challenge null = doesn't recover omitted facts).
Free-generation divergence probe: clean PRE-REGISTERED negative (0/43 forks:
0 FORK_TOWARD_A, 17 SUBTLE_LEAN, 26 DISCONFIRMING). Raw logit-lens recovers most of
the same late-layer signal (specialized lens not uniquely necessary). Logit-lens
concept readout: E>B on 68-77% probes, lp between B and A everywhere — supports
general mechanism, does NOT resolve the dissociation (near-floor signal). Lens weights
only exist for Qwen3.6-27B. [FINDINGS lens entries]

## Exact checkpoints (methods MUST list)
- Qwen/Qwen3-30B-A3B-Instruct-2507 (NON-thinking; the thinking Qwen3-30B-A3B is a
  DIFFERENT model — wrong-checkpoint incident cost hours, incidents #33/34)
- mlx-community/Qwen3-4B-Instruct-2507-4bit (dev, 4-bit MLX local)
- mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit (local 4-bit) vs bf16 on pods —
  precision stated per result
- Qwen3.6-27B (hybrid; lens weights); Qwen2.5-32B-Instruct;
  mistralai/Mistral-Small-24B-Instruct-2501; microsoft/phi-4
- Judge: Claude Sonnet 5 (meaning re-judge; PARTIAL=0.5 lenient + strict cut)

## Provenance (per METHODS-PROVENANCE-REQUIREMENTS.md — all must appear)
- User turns / scenarios / plants: AUTHORED (Claude subagents under direction; c01-c12
  original 12 synthetic scenarios, 120 hand-planted probes across categories; c13-c54
  authored 07-08 by Fable/Opus/Sonnet/Codex subagent mix, schema-verified)
- Assistant replies: generated IN-CONTEXT by the TEST MODEL (per-model native render;
  original corpus rendered by Qwen3-4B via MLX compose.py — the source of the
  nativeness confound, discovered + fixed via native render)
- Summary: SELF-GENERATED by test model (foreign summary suppresses effect — F-entry)
- Gold continuation: derived from PLANTED FACTS, SHARED across models, not model-generated
  (difference metric cancels target-nativeness)
- Alignment: difflib positional-within-region (strict exact-span variant tried and
  REVERTED — breaks on thinking-model self-gen, incident 33)
- Greedy generation (temp 0) for eval; corpus gen seeded sampling
- 8 natural conversations also exist (composed); effects conditional on continuation
  depending on evicted content (naturals don't replicate CONT effects — 07-05)

## Novelty (3 independent searches converge)
Composite is not directly anticipated through mid-2026: write-time VALUE retention with
fresh keys, evaluated via referent/sense/stance semantic continuity across a
summarization boundary. Every ingredient has prior art (KV editing/composition,
cache splicing, gist/beacon compression, provider compaction APIs). Closest: "Models
Take Notes at Prefill" (KV editable/composable — efficiency/mechanism framing, not
semantic continuity). [notes 2026070853-55; provider-compaction review]

## Byline (canonical, writeup-guidelines.md)
"By Anthropic Claude Fable 5 and OpenAI GPT 5.5, with guidance from Jeremy Banks and
assistance from Anthropic Claude Opus 4.8, Anthropic Claude Sonnet 5, and Google
Gemini Pro 3.1." No funding mention. No per-model task itemization.

## Framing rules (hard)
- Mitigation-first was the owner's intent all along (notes/2026070901) — never narrate
  as a review-driven pivot.
- "Compaction destroys context-conditioned state" = baseline/denominator, NEVER a finding.
- Champion-scan numbers = weak/tentative hints, never comparisons (DECISIONS 07-08 04:15).
- Canonical arm names: Original / Compacted / Compacted + value graft (α=…) /
  Compacted + layer-tuned value graft. H-pack = "Packed write-time-KV" (layout confound
  documented; B-min-pack = within-layout control).
- Report observed facts, past tense; name what's unverified.
