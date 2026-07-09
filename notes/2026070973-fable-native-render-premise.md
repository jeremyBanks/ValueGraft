# Fable — un-anchored reflection on the NATIVE-RENDER premise (2026-07-09)

*Consult run un-anchored; captured here per the standing "Fable writes to notes" practice.
This one predated the mechanism by minutes, so Claude transcribed Fable's returned assessment.*

## (a) Epistemic status: PARTIALLY supported — the specific evidence used to justify the expensive native-rendering work does NOT hold up. Split the premise into two limbs.

The premise bundles **two logically distinct claims** under one word "nativeness."

**Limb A — "the graft needs the model's OWN self-generated summary, not a foreign one."**
VERIFIED, replicated, mechanistically clean. Isolation test (same model + harness + authored replies,
only summary source varied): FIXED Sonnet summary → referent +0.004 (null), sense −0.147; SELF-GEN →
referent +0.136 CI[+0.034,+0.23], agg +0.090 CI[+0.022,+0.154] (FINDINGS.md:466–479; CLAIMS.md MTH-3).
Legitimate single-variable result. **Limb A is the real mechanistic finding and it is sound.**

**Limb B — "you must render the REPLIES natively; authored (4B) replies are an invalid foreign-content
confound, so +0.147 authored was inflated."** ASSUMED. Its headline justification is confounded with
render noise of the same magnitude as the effect.

Reconciliation (c01–c12, referent, raw_EB E−B nats, self-gen summary, recomputed from disk):

| render of c01–c12 | referent raw_EB | 95% CI | source |
|---|---|---|---|
| AUTHORED (Qwen3-4B, fixed corpus) | +0.147 | [+0.063,+0.231] excl 0 | results/gap_closure_cat/ |
| NATIVE draw 1 ("live", v2.1, 07-07) | +0.116 | [+0.014,+0.217] **excl 0** | results/gap_closure_cat_live/ |
| NATIVE draw 2 (b0 banked, 07-09) | +0.012 | [−0.068,+0.090] spans 0 | results/redraw_harvest/b0/ |

The current story — "nativeness moved +0.147 → +0.012 (null)" — **cherry-picks the low native draw and
ignores the high one on disk the whole time.** The *first* native render (+0.116, CI excludes 0)
essentially REPRODUCED authored (+0.147). Two independent native draws of identical convs: +0.116 and
+0.012 — a 0.10 swing from MoE hardware-nondeterministic routing. **Render-to-render SD of referent at
n=12 ≈ its own effect size.** So "nativeness killed the effect" is NOT established; render variance did,
and the effect is underpowered/render-fragile regardless of nativeness. (gap_closure_cat_live and _live2
are byte-identical = same render scored twice; so c01–c12 has exactly two independent native draws.)

**Two further gaps:**
1. **The decisive de-confound was designed but never run:** run the AUTHORED (4B) corpus on a NON-Qwen
   model with self-gen summary; if referent collapses → nativeness dominates (FINDINGS.md:495–502). On
   disk every cross-arch run is native+self-gen; the cheap authored-cross-model control was skipped.
2. **Negative controls point AGAINST "authored is a foreign-content artifact":** on authored render,
   E-wrongconv = −1.50, E-shuffled = −1.36 vs B (both harmful) while aligned E-post is positive
   (results/gap_closure.json, whole-continuation). A generic "any values help" artifact would make
   wrongconv help; it destroys → alignment-specific, real. (Caveat: whole-continuation readout, not the
   referent-category α=0.75 readout — EXP-2 closes exactly this.)

Independent corroboration: the placebo-controlled bounding estimator on AUTHORED (n=43) returned NULL
(E−B −0.049 [−0.113,+0.014]; placebo−B −0.177 excl 0; E−placebo +0.128 [−0.028,+0.289])
(results/effect_bound/…20260707T232845Z.json). So authored was control-bounded null BEFORE nativeness
entered → the +0.147→null shift cannot be blamed on nativeness.

### Strongest argument against "native = the valid measurement"
One native render (+0.116, excl 0) reproduced authored (+0.147, excl 0); the null came from a *different*
native draw of the same convs. The hours of native re-rendering did not demonstrably change the answer —
they added a ~0.10 render-noise term, and a low draw was interpreted as a substantive result. The
self-gen-summary limb did the scientific work; the native-*replies* limb rode along unearned. Native MoE
rendering may have *degraded* measurement precision without evidence it improved validity.

### What would change the verdict
- Toward CONFIRMED: authored referent sits consistently above a properly-sampled native envelope (≥5
  draws) AND authored E-wrongconv/placebo at the referent readout also recovers ~+0.147.
- Toward REFUTED: native envelope routinely reaches +0.10–0.15 (authored inside it), OR authored
  E-wrongconv/placebo at referent readout ≈ 0/negative while aligned = +0.147 (authored positive is real).

## (b) Experiment plan — ordered, reuse-first (warm Qwen3-30B + banked c01–c36 native renders + authored corpus). Bank every render.

**EXP-2 (minutes, ~0 new GPU) — foreign-content artifact test at the referent readout, BOTH render
types.** Re-graft already-banked values: aligned E-post-a0.75, E-wrongconv, placebo=gauss,
placebo=shuffle_pos; referent+sense; conv-clustered CI over c01–c12; on AUTHORED and NATIVE(b0).
- CONFIRM premise: authored wrongconv/placebo referent ≈ aligned (+0.10..+0.15) → authored positive is a
  generic artifact, native is the honest measurement.
- REFUTE premise: authored wrongconv/placebo referent ≈ 0/neg while aligned = +0.147 → authored positive
  is alignment-specific and REAL → native rendering removed no confound (added noise). (Prior from the
  −1.50 whole-continuation control: REFUTE is more likely.)

**EXP-3 (minutes) — the clean 2×2 summary axis on NATIVE replies.** Add the missing cell by re-scoring
b0 banked native replies with the FIXED Sonnet summary swapped in. Cells (replies × summary), ref+sense,
c01–c12: (authored,self)=+0.147; (authored,foreign)=+0.004; (native,self)=+0.012/+0.116; (native,foreign)=NEW.
- CONFIRM Limb A generalizes: (native,self) − (native,foreign) large & positive (~+0.14). REFUTE: gap
  vanishes on native → Limb A was authored-specific.

**EXP-1 (~1.5–2 GPU-hr, one warm pod) — native render-variance ENVELOPE.** Re-render c01–c12 natively
K=5–6 more times on warm Qwen3-30B (bank every render); score ref/sense/stance raw_EB per draw → n≈7–8
native estimates. Test whether authored +0.147 lies inside/outside the native envelope. This is the one
number nobody has measured: the render-noise SD.

**Order:** EXP-2 + EXP-3 immediately (free, warm pod/banked), read them, THEN commit ~2 GPU-hr to EXP-1.
Total moderate spend ~2–3 GPU-hr on one pod; everything else forward-passes.

## (c) Trajectory changes
1. **Stop citing "+0.147 authored → +0.012 native" as establishing nativeness.** It's single point
   estimates under a metric whose render-noise SD ≈ the effect; the +0.116 native draw that reproduces
   authored was dropped from the narrative. Reframe STATE.md/CLAIMS.md to: "referent raw_EB is
   render-fragile at n=12; native and authored are indistinguishable up to render variance (pending EXP-1)."
2. **Report the premise as two claims:** Limb A (self-gen summary) VERIFIED, keep as mechanism; Limb B
   (native replies required) ASSUMED, motivating evidence confounded — present EXP results, not assertion.
3. **Hard rule:** with a nondeterministic (MoE) render and noise-SD ≈ effect, NEVER compare render
   conditions on single draws — estimate the render-variance envelope FIRST, then test.
4. **Paper posture unchanged and correct:** SENSE-led (judged sense +12pp banked/reproducible), referent
   as fragile/underpowered dissociation with significance-flips-by-metric. This analysis *reinforces* that
   and removes a shaky sub-claim (nativeness made referent null) the paper should not lean on. Tightens, not costs.
