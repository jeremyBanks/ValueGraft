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
- PRE-REGISTER ONE DIRECTIONAL HYPOTHESIS (a single named geometry variable or a single
  composite index), predicted BEFORE the sweep. **[TO FINALIZE before the run — via Fable
  mechanistic reasoning + the actual per-model geometry table; the leading candidate is
  QK-norm presence and/or GQA ratio, since QK-norm is the sharpest attention-geometry
  difference between the known-positive Qwen3 and the (earlier) negative Qwen2.5. This line
  MUST be replaced with the single committed directional prediction + its mechanistic
  rationale before the 16-run; committing it is the last pre-reg step.]**
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
