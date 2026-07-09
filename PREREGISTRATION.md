# Pre-registration — cross-architecture value-graft sign map

> Committed BEFORE running the 16-model sweep (design v2.1 requirement). Confirmatory,
> not exploratory. Freeze this file; timestamp = its git commit. Any change after the
> sweep starts is a protocol deviation and must be reported as such.

## Estimand
For a model operating on its OWN native conversation (assistant replies generated
in-context by that model + its own self-gen compaction summary), does re-injecting
write-time VALUE vectors at the compaction boundary help or harm referent-continuity
recovery, and is the SIGN predicted by attention geometry?

## Metric (frozen)
`raw_EB = lp_E(graft) − lp_B(compacted)`, teacher-forced on the SHARED gold continuation
(from the planted facts, NOT model-generated). Per category. Robust: raw E−B + percentile
bootstrap 95% CI over CONVERSATIONS (cluster). `raw_EB_normalized = raw_EB / max(headroom, ε)`,
headroom = referent A−B gap. SIGN = sign of the normalized referent raw_EB.

## Inclusion / exclusion (frozen, applied per model before any sign is read)
- HEADROOM floor: a category with referent headroom (A−B) < 0.3 is FLOORED — excluded from
  the sign readout (no evicted meaning = nothing to recover ≠ "harm").
- TASK-COMPETENCE: a plant with per-token lp_A below the competence floor (model can't solve
  the task even with full context) is dropped. Report N excluded per model.
- GENERATION-QUALITY: coherence-screen the model-filled replies; a model whose native replies
  are degenerate/incoherent is flagged and its result treated as untrusted (report, don't hide).

## PRIMARY inference (high power, nativeness-controlled by construction)
The two WITHIN-VENDOR dense/MoE de-confound PAIRS:
  - Qwen3-30B-A3B (MoE) vs Qwen3-32B (dense)
  - Gemma-4-26B-A4B (MoE) vs Gemma-4-31B (dense)
Because value vectors are produced in the ATTENTION block and MoE lives in the FFN, the
geometry hypothesis predicts: **within each pair, the referent-recovery SIGN does NOT flip
between the MoE and dense variant** (same-vendor attention geometry ≈ held fixed; MoE/FFN
should not determine the value-graft sign). A sign FLIP within a pair falsifies "geometry
(attention), not MoE (FFN)."

## SECONDARY / confirmatory (the 16-model regression)
`referent_sign ~ geometry + headroom + reply_infocontent + reply_length`, geometry =
{n_kv_heads, head_dim, GQA ratio, QK-norm presence, RoPE theta, n_layers}. Report that geometry
survives the controls. n=16 is UNDERPOWERED for 5 free predictors, so:
- **COMMITTED DIRECTIONAL HYPOTHESIS (frozen before results, 07-08):**
  **H1: QK-norm presence predicts the SIGN of referent value-graft recovery — models WITH
  per-head QK-norm (q_norm/k_norm modules) have referent raw_EB ≥ 0 (help); models WITHOUT
  trend ≤ 0 (harm/null).** Direction: QK-norm present → positive.
  - Mechanistic rationale: QK-norm re-normalizes each head's query/key before the attention
    dot-product, which stabilizes the attention distribution over positions. A grafted
    write-time VALUE vector is only useful if the compacted-context query still attends to
    the grafted slot with the write-time geometry; QK-norm makes that attention pattern more
    scale-invariant / transferable across the A→B context change, so the re-injected value is
    read out constructively rather than mis-weighted. It is also the SHARPEST attention-geometry
    difference between the known-positive Qwen3 family (QK-norm) and the (earlier) negative
    Qwen2.5 (no QK-norm) — an existing, non-fitted contrast.
  - Test: one-sided; QK-norm coefficient on referent_sign has the predicted (positive) sign
    after the reply-content + headroom controls.
  - **BACKUP reliably-measured predictor (reported alongside, NOT the primary): GQA ratio**
    (n_heads/n_kv_heads) — reported as a secondary direction in case QK-norm module-detection
    (newly fixed 07-08; see below) proves unreliable on any family. If QK-norm detection is
    confirmed clean across the 16 (qk_norm_source = "module:*"), H1 is the primary; if detection
    is ambiguous for some models, fall back to GQA and report the switch as a deviation.
- Nativeness (mean logprob of each model's own corpus under itself) reported as a robustness
  covariate only — NOT the primary fix (collinear/underpowered alone).

## SUPPLEMENT (causal cross-check, not primary)
A shared-fixed-corpus arm on ~4–6 models (everyone on identical replies) to answer "holding
context fixed, does geometry flip the sign?" Reported as a mechanism supplement; expected to
be weaker/less-realistic than the native-per-model primary.

## What confirms / disconfirms
- CONFIRM: no sign-flip within either de-confound pair AND the pre-registered geometry
  direction predicts the cross-model sign after controls.
- DISCONFIRM: sign flips within a pair (MoE/FFN matters) OR the pre-registered geometry
  direction fails after controls.
- INCONCLUSIVE (report honestly): too many models floored/excluded, or the sign sits inside
  the bootstrap noise band near zero.

## GEOMETRY TABLE + a data-quality note (07-08)
Gathered data/model_geometry.json for all 16. Reliable varying predictors: GQA ratio (2..24),
head_dim (64..256), rope_theta (1e4..1e9), n_layers (24..80).
⚠️ QK-NORM DETECTION IS BROKEN: config-key detection returns False for ALL models, but Qwen3
family / Gemma-3,4 / OLMo-2 DO use QK-norm. It's an architectural feature, not a reliable config
flag. MUST detect from the LOADED MODEL's modules (presence of q_norm/k_norm layers) — the harness
model_hparams.qk_norm has the same bug and must be fixed before qk_norm is used as a predictor.
Until fixed, do NOT pre-register on qk_norm; either fix detection first, or pre-register on a
reliably-measured predictor (GQA ratio / head_dim / a composite). Finalize with Fable.

## PRE-FLIGHT DEVIATION NOTE (07-08, appended before ANY outcome/effect data — NOT fit to data)
An HF architecture pre-flight (querying each model's config `architectures` BEFORE running inference)
found 5 of the 16 models ship as MULTIMODAL wrappers (`*ForConditionalGeneration`, not
`*ForCausalLM`), incompatible with the `AutoModelForCausalLM` harness:
- google/gemma-4-31B-it, google/gemma-4-26B-A4B-it  (Gemma4ForConditionalGeneration) — the pre-registered
  Gemma-4 PRIMARY de-confound pair
- google/gemma-3-27b-it  (Gemma3ForConditionalGeneration)
- Qwen/Qwen3.6-35B-A3B, Qwen/Qwen3.6-27B  (Qwen3_5(Moe)ForConditionalGeneration)
These are EXCLUDED BY TOOLING, discovered at pre-flight, with NO effect estimates informing the decision.
CONSEQUENCE: PRIMARY inference drops from TWO within-vendor dense/MoE de-confound pairs to ONE (Qwen3-30B-A3B
MoE / Qwen3-32B dense). Cross-vendor generalization now rests on the pre-registered H1 (QK-norm) regression
across the 11 text models. This conjunction — one within-vendor de-confound + a cross-vendor geometry
regression — is the reported primary case.
RECOVERY ATTEMPT (pre-registered here, before results): a TIME-BOXED, positive-control-gated spike to load
the GEMMA-4 pair via its TEXT substack (Gemma exposes a first-class text decoder). GATE: (1) the text-substack
KV snapshot must byte-match a clean CausalLM load of the same text weights, (2) self-graft must reproduce
baseline generation. If BOTH pass, Gemma-4 is reported as the pre-registered pair RECOVERED via a disclosed
text-substack loading path (same comparison, not post-hoc). If either fails or the box expires, we fall back
to the one-pair-plus-regression case above. Qwen3.6 recovery is NOT attempted (same-vendor as Qwen3 = low
de-confound value). All 5 models appear in an exclusions table with the architecture class as the stated reason.

## GATE #3 AMENDMENT — RELATIVE (per-model) task-competence floor (pre-registered 2026-07-08, BEFORE it is applied to any scored result)

This amendment CHANGES the task-competence gate (inclusion/exclusion GATE #3 above)
for ALL models. It is committed BEFORE any result is scored under it, so it is a
deliberate pre-commitment, not a post-hoc knob. Any already-scored result would be
recomputed under this rule for comparability (see the on-disk comparability note below
showing the recompute is a verified no-op for the models scored so far).

**Problem being fixed.** The original GATE #3 used an ABSOLUTE floor
`task_lpa_floor = -8.0` on the per-token gold logprob under full context A (`lp_A`).
An absolute floor confounds cross-model comparison: it silently drops MORE plants from
models whose logprobs run lower (different vocab/tokenizer/scale), so different models
get scored on different subsets. Observed on-disk (per-token `lp_A`, this harness / the
trusted gap_closure_cat path):

| model | N | median lp_A | min lp_A | plants below −8.0 |
|---|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 | 67 | −2.76 | −4.91 | 0 |
| Mistral-Small-24B-2501 (cross-arch) | 120 | −1.70 | −4.03 | 0 |
| Qwen2.5-32B | 67 | −3.74 | −6.76 | 0 |
| Qwen3-4B | 67 | −4.35 | −6.33 | 0 |
| OLMo-2 | (not on disk) | — | — | **ALL** (per FINDINGS: floor excluded every plant) |

So −8.0 is INERT on Qwen/Mistral (excludes 0) yet total on OLMo-2 — the classic
different-subsets confound. OLMo's own `lp_A` distribution is NOT observable on disk (its
run errored under the floor). Minimal data that would settle OLMo directly: one OLMo run
that records per-plant `lp_A` even for gate-excluded plants (the `task_excluded` records
already store `lp_A`); that is exactly what a relative-mode re-run produces.

**The rule (frozen).** Replace the absolute floor with a WITHIN-MODEL ROBUST-OUTLIER
floor computed from the model's OWN gold-`lp_A` distribution (pooled over all of that
model's scored plants):

>   floor_model = median(lp_A) − K · MADN(lp_A),
>   MADN(lp_A) = 1.4826 · median(|lp_A − median(lp_A)|)   (normal-consistent MAD)
>   K = 3.0   (a plant is excluded iff lp_A < floor_model)

A plant is scored iff `lp_A ≥ floor_model`. If a model has fewer than 8 finite `lp_A`
values (too few to estimate a scale) the floor is undefined and NO plant is dropped
(permissive fallback). Applied IDENTICALLY across all models. This is `--task-competence-mode
relative` (`SC_TASK_COMPETENCE_MODE=relative`), K via `--task-competence-k`; absolute mode
stays the default until the sweep is re-run under this rule.

**Justification of each design choice.**
- *Relative/per-model (a):* the floor tracks each model's own logprob scale, so it does not
  confound cross-model comparison — a model whose whole distribution sits lower is not
  penalized for that shift, only for plants that are anomalous FOR IT.
- *Includes OLMo's plants (b):* median − K·MADN adapts to OLMo's (lower) scale, so OLMo's
  TYPICAL plants clear it; the spurious "no plants scored" error is removed.
- *Identical across models (c):* the same estimator + K = 3.0 + min-N = 8 for every model.
- *Still a genuine competence gate (d):* it does NOT disable the gate — it excludes plants
  whose `lp_A` is an extreme low outlier FOR THAT MODEL (the degenerate / can't-do-the-task
  plants the gate is meant to remove). MAD (not mean/SD) is used deliberately so the very
  low-outlier plants we want to exclude cannot inflate the spread and mask themselves.
- *K = 3.0* is the conventional extreme-outlier cutoff (≈3σ under normality), chosen a priori.
  On the observable models the exclusion count is INVARIANT for K ∈ [2.5, 4.0] (all give 0
  excluded), so the exact K is not doing hidden work here; it will matter only on a model
  with a genuine low-outlier tail, which is where a gate should act.

**Comparability impact (verified on-disk, no GPU).** Under this rule the per-model floors
are: Qwen3-30B −7.87, Mistral-24B −5.42, Qwen2.5-32B −11.30, Qwen3-4B −9.14. Each sits
below that model's minimum observed `lp_A`, so the rule excludes 0 plants on every model
scored so far — IDENTICAL to the current −8.0 (which also excludes 0). Therefore the already
-scored Qwen/Mistral included-plant sets and all their raw_EB numbers are UNCHANGED under
this amendment (the required recompute is a proven no-op for them). The rule changes outcomes
ONLY for models whose distribution the absolute floor mis-handles (OLMo-2), which is the point.
A pure lower-quantile rule (e.g. drop the bottom 5%) was REJECTED precisely because it would
mechanically drop ~3–6 plants from Qwen/Mistral that −8.0 keeps, changing their sets and
breaking backward-comparability; the robust-outlier floor does not.

**What confirms/disconfirms unchanged.** This is an inclusion-rule amendment only; the
estimand, metric, headroom gate, and the H1/geometry inference are untouched.
