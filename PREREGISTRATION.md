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
