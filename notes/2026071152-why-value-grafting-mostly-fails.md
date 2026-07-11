# Why re-injecting write-time values mostly fails to recover anything

*A speculative, pedagogical note. Written after the paper (README.md) was finished, reasoning
from the result rather than adding to it. Everything here is a hypothesis. Where the paper's own
data already bears on a hypothesis I say so; where a claim is pure speculation I flag it; and for
each I try to say how it could be tested. None of this is a finding.*

---

## The thing to explain

The value graft copies the model's original write-time V vectors back onto the summary tokens of a
freshly re-encoded compacted cache (keys untouched) and asks whether the model then predicts the true
continuation better than it does from plain compaction. The paper's result, stated compactly:

1. On held-out synthetic recovery it is **null** across all four tuned variants and flat/slightly
   negative across a 30x compression sweep.
2. It is **content-specific but non-additive**: injecting the *correct* write-time values reliably
   beats injecting shuffled/noise-matched ones (sometimes by more than a nat), yet never beats the
   re-encoded summary the model already re-read.
3. The one survival is **real coding trajectories under brief summaries**, where a fixed-strength
   graft is sample-heterogeneous (positive on 75, null on a fresh 98) and only an *in-domain
   per-layer-tuned* graft holds a small pooled positive (~+0.013 nats/token). Middle layers help;
   the tuned config grafts bands 12-17 and 30-35 and drops early/late layers.

Six questions follow. I take them in order.

---

## H1 — Why re-injection mostly recovers nothing: the summary already carries the recoverable part, and V is largely a re-derivable function of the text

The single most parsimonious explanation, and the one the paper itself lands on in §5: **the value
vector at a summary token is, to first order, a function the model computes from the surrounding
tokens.** When the model re-reads the summary, it recomputes values at those positions from the same
text through the same weights. If the summary text pins down the content, the freshly-derived V is
already most of the way to the write-time V, and the graft is adding a small correction to a quantity
that is already near-correct.

Put differently: values are not an *independent* memory store sitting alongside the text. They are an
activation — a deterministic readout of the resident tokens (plus their left-context) under fixed
weights. The only part of the write-time V that the re-read *cannot* reconstruct is the part that
depended on context the summary dropped. And a competent summarizer, by construction, tries to put
the decision-relevant dropped context into words. So the residual — write-time V minus re-read V,
restricted to the part that is *useful* for the continuation — is squeezed from both sides: it is
small because V is mostly re-derivable, and what's left is mostly redundant with text the model
re-read anyway.

- **Data bears on this:** strongly. This is the paper's §5 reading and it is consistent with the null
  being robust to compression (H3) and with content-specificity being real but non-additive (H2). It
  also predicts, correctly, that a *foreign* summary collapses the effect — if V is a readout of the
  resident text, grafting values whose provenance is a different act of summarizing onto text the
  model didn't write should not help, and it doesn't.
- **Speculative part:** the claim that the *useful* residual specifically is near-zero (as opposed to
  the total residual). Content-specificity says the total residual is large. H2 is about why the
  useful part isn't.
- **Test:** decompose write-time V at summary positions into the component predictable from the
  re-read residual stream (regress V_write on the fresh hidden states / fresh V) and the orthogonal
  remainder. The prediction is that recovery, if it lives anywhere, lives in the orthogonal remainder
  and that its norm is small relative to V, and smaller still after projecting onto directions the
  unembedding/next-token gradient actually reads.

---

## H2 — Why content-specific yet non-additive: the graft can only *avoid harm*, because the summary already occupies the useful subspace

Content-specificity is large and scales with slot count; recovery is null. The clean way to reconcile
these: **the correct write-time V and the re-read V agree on the component that matters for the
continuation, and disagree mostly on components that don't.** A wrong graft (shuffle/noise) destroys
the agreeing component and injects garbage into the attention readout, which is actively harmful — and
the more slots you corrupt, the more harm, which is exactly the observed scaling of content-specificity
with slot count. The correct graft merely declines to inflict that harm. So "correct beats wrong" is
real and can be huge, while "correct beats the summary" is null, because on the useful axis correct
and summary are already the same vector.

Geometrically: think of the useful subspace (the directions the downstream layers and unembedding
read to produce the gold continuation) as already spanned by the re-read state. Write-time V adds
energy, but off that subspace. Adding an off-subspace vector doesn't move the readout; scrambling the
on-subspace vector does. Non-additivity is then not mysterious — it's what you'd expect from injecting
a signal that is either redundant (on-axis) or irrelevant (off-axis), with wrong grafts being the only
way to get *destructive* interference.

- **Data bears on this:** strongly. Content-specificity scaling with slot count, being null for the
  smallest configuration, and never converting to recovery is precisely this picture. The paper's own
  phrasing — "entirely a matter of not-hurting" — is H2 in words.
- **Speculative part:** the specific geometry (useful subspace already spanned). It's the natural model
  but not directly measured.
- **Test:** project both V_write and V_fresh onto the gradient of the gold-continuation logprob w.r.t.
  the value at each grafted position. Prediction: the projections are nearly equal (redundant on the
  useful axis), while the full vectors differ substantially (large content-specificity off it). If so,
  additivity was never geometrically available.

---

## H3 — Why more compression doesn't summon the effect: terseness removes *text*, not the redundancy between V and text

The natural hypothesis the sweep was built to test — "shorter summary → more evicted meaning → more
for the graft to recover" — assumes the graft recovers *evicted* meaning. But under H1 the graft only
supplies the part of write-time V that isn't re-derivable from the resident summary text, and it can
only supply it *at surviving summary positions*. Making the summary terser does two things that both
work against the graft:

1. It shrinks the number of grafted positions (fewer summary tokens to carry values), so there is less
   surface for any lift to act through.
2. The meaning it evicts is evicted precisely by *not being in the resident text* — and the graft's
   values live on the resident tokens. A value vector at "the summary" can't encode a fact that has no
   token to sit on. The RoPE-addressed key for the evicted content is gone; there is no slot whose
   value you could graft to bring it back.

So "how lossy the summary is" is the wrong axis: terseness increases the *demand* for recovery while
simultaneously removing the *substrate* (tokens/positions) through which the graft could deliver it.
The verbatim-fact harm at the one-sentence level (−0.17 on precise strings) is the sharp version — you
cannot reconstruct a deleted string by perturbing values on the tokens that remain, and the
perturbation slightly displaces even what the model could still guess.

- **Data bears on this:** strongly for the flatness; the sweep is the direct evidence. The "fewer
  positions" mechanism (point 1) is partly evidenced by content-specificity scaling with slot count
  elsewhere.
- **Speculative part:** attributing the flatness specifically to the substrate-removal rather than to
  H1's redundancy. These are complementary, not rival, and I can't separate them from the sweep alone.
- **Test:** hold the number of grafted positions fixed while varying summary detail (e.g. pad terse
  summaries to constant token count), to separate "less text" from "fewer slots." If recovery stays
  flat even at constant slot count, redundancy (H1) dominates; if it rises, substrate (H3.1) mattered.

---

## H4 — Why sample-heterogeneous and why in-domain layer-tuning stabilizes it: a small effect near the noise floor plus a bias/variance argument

The fixed-strength graft is +0.015 on 75 trajectories and −0.002 on a disjoint 98, a genuine
between-pool difference. Two non-exclusive stories:

- **H4a (near-floor sampling).** The true per-domain effect is a few thousandths of a nat/token — of
  the same order as between-trajectory variance. A single flat α is a blunt instrument: on any given
  sample it can land net-positive or net-null depending on the mix of trajectories where middle-layer
  injection happens to help versus where late-layer injection (which the flat graft also applies) hurts.
  The flat graft carries both the helpful and the harmful layers at once; its sign is then a sample
  property.
- **H4b (tuning removes the harmful layers, not just picks the lucky ones).** The in-domain champion
  keeps only bands with a *positive marginal* and grafts 12-17 and 30-35 while dropping early and very
  late layers. If early/late grafting is net-harmful (H5), the flat graft is (small help) + (small harm)
  and the tuned graft is (small help) alone. That both raises the mean *and* reduces variance — you've
  removed the layers whose sign flips across samples. That is exactly the observed signature: the tuned
  graft beats the fixed one specifically on the fresh pool where the fixed one fails, and ties it where
  the fixed one already works (because there the harmful layers happened to be dominated).

The reason tuning has to be **in-domain** is H6: the useful-layer profile is a property of the
content/task, and the synthetic champion was fit on a distribution where there was no positive to find
at all, so it can't transfer.

- **Data bears on this:** partly. The between-pool difference excludes zero, and the tuned graft's
  "wins where fixed fails, ties where it works" pattern is real in the tables. The decomposition into
  help-minus-harm is inferred, not measured.
- **Speculative part:** H4b's claim that the variance reduction comes specifically from dropping
  sign-flipping layers. Plausible and consistent, not proven.
- **Test:** measure per-layer-band marginal recovery *separately on each pool*. Prediction: the
  helpful bands (12-17, 30-35) are positive in both pools; the dropped bands are the ones whose sign
  differs between pools. If the between-pool difference is concentrated in the dropped bands, H4b holds.

---

## H5 — Why middle layers help while early and late layers hurt: layer roles and where the write/read gap can matter

This is the most mechanistically interesting sub-question and the most speculative. A layer-role
account:

- **Early layers** are close to the token/positional surface. Their value vectors are dominated by
  local, largely lexical features that the re-read reconstructs almost perfectly from the same tokens —
  so there's nothing to add — and any mismatch you introduce (write-time values computed under a
  *different* absolute-position and different left-context than the re-encoded layout) is a
  perturbation to a quantity the model is about to use for low-level composition. Injecting a slightly
  off early-layer value is like corrupting the input features: net harm, no upside, because the useful
  residual there is ~zero (H1 at its strongest) while the harm is nonzero.
- **Late layers** are close to the output and are largely *already-decided*: by the top of the stack
  the representation has been shaped into next-token-predictive form conditioned on the resident
  context. Grafting write-time values here overwrites the model's *current* near-output computation
  with a version formed under a context (the full pre-compaction history) that no longer exists. You're
  pasting an answer computed for a different question. That is plausibly harmful precisely because late
  values are high-leverage on the logits — a small wrong nudge moves the output directly.
- **Middle layers** are where the paper's small positive lives, and the story is that this is the band
  where two conditions coincide: (i) the representation is abstract enough that the write-time V encodes
  *procedural / relational* content that isn't a trivial re-read of the tokens (so there's a non-zero
  useful residual), yet (ii) it is not so close to the output that a small mismatch dominates the
  logits (so the harm is bounded). Middle layers are the classic locus of "what is being talked about
  and what has been established" — the level at which retained trajectory state (H6) would live if it
  lives anywhere.

There's a positional wrinkle that reinforces the early/late asymmetry. Keys are left untouched (α_K=0),
so all grafted values are read through *re-encoded* keys at *new* absolute positions. The write-time V
was produced in concert with write-time keys at old positions. Values don't carry RoPE directly, but
they were computed by a stack that had already attended through rotary-addressed keys, so a write-time
V is implicitly "addressed" to a positional context that no longer matches. Early layers, being closest
to where position is injected, are where that mismatch is most acute — another reason early grafting is
harm-without-upside.

- **Data bears on this:** partly. The champion grafting 12-17 and 30-35 and dropping early/very-late
  layers is direct evidence that the *useful* layers are in the interior; the held-out synthetic
  per-layer null and the cross-architecture negatives are consistent with "grafting the wrong layers
  hurts." The *mechanism* (why each band behaves as it does) is speculation built on standard
  layer-role intuitions.
- **Speculative part:** essentially all of the per-band interpretation. Note 30-35 is not "late" in a
  ~48-layer model — it is upper-middle — so the clean "late hurts" story applies to the very top, not
  to the upper-middle band the champion keeps. That nuance matters and I don't want to oversell a tidy
  three-zone picture.
- **Test:** a per-layer graft-vs-placebo *and* graft-vs-baseline profile, on coding trajectories,
  reported for every layer. Predictions: (i) early layers show near-zero content-specificity residual
  (V re-derivable) and negative recovery; (ii) the very top layers show large logit leverage and
  negative recovery; (iii) the two kept bands show positive recovery. Cross-checking with a
  logit-lens / activation-patching readout of *what* the kept-band values encode (procedural state vs.
  lexical) would test the "abstract middle" claim directly.

---

## H6 — Why real coding trajectories under aggressive compaction are the one place it does anything: procedural state is a write-time byproduct that text summaries under-serialize

The domain contrast is the sharpest fact in the paper: equally-brief *synthetic* summaries produce no
positive, but *coding* trajectories do. So terseness is necessary and not sufficient; domain is the
distinguishing variable. A hypothesis for what's different:

Synthetic plants are **declarative facts** — a sense, a referent, a stance. A fact is exactly the kind
of thing a summarizer is good at writing down in words, so its useful residual (H1) is near-zero: if the
model retained it at all, it's in the summary text. Real coding trajectories carry **procedural state** —
an implicit working model of the repo, which files were touched, what was tried and failed, the shape of
the current edit, the "where am I in this task" that accumulates over thousands of tokens of tool use.
That state is (a) high-dimensional and relational, (b) genuinely computed into the write-time activations
by the act of doing the task, and (c) *hard to serialize losslessly into a brief natural-language summary*
— a three-sentence summary of a 12k-token repair episode drops most of it. That is the profile of content
whose write-time V has a non-trivial useful residual: the model built the state internally, and the terse
summary couldn't carry it out in words. So the graft has something to add that the re-read genuinely lacks.

This also explains why it's confined to **brief** summaries (realistic-length summaries have room to
serialize enough procedural state that the residual closes again) and why it's **small and unstable**
even there (procedural state is only *partly* recoverable from write-time values at surviving positions;
much of it lived on tokens that were evicted, per H3). And it explains why in-domain tuning is required
(H4): the layer profile that carries procedural residual is a task property, absent in the synthetic
distribution where there was no residual to profile.

- **Data bears on this:** the domain contrast, the brief-only confinement, and the vanishing under
  realistic summaries are all in the paper and all consistent with H6. The *identity* of the extra
  content as "procedural state" is interpretation — the paper is explicit that it "cannot name the
  mechanism," and I am not naming it, only proposing a candidate.
- **Speculative part:** substantial. "Procedural state under-serialized by brief summaries" is a
  reasonable candidate but the paper's own two coding pools differ from each other in ways it couldn't
  isolate, so even within-domain the story is incomplete.
- **Test:** stratify coding trajectories by a proxy for procedural-state density (number of distinct
  files/tools touched before the cut, or the full-vs-brief-summary logprob gap on the next action) and
  check whether recovery concentrates in the high-density stratum. If the effect is procedural-state
  driven, it should ride that proxy — and that same proxy might explain the 75-vs-98 heterogeneity (H4)
  if the pools differ in state density. The stronger, expensive test is the one the paper flags as open:
  measure task success (apply the action, run tests) rather than next-action logprob.

---

## What ties it together

If H1 and H2 are right, the null is overdetermined: values are a re-derivable readout of resident text,
the useful subspace is already spanned by the re-read, so the correct graft can only decline to harm and
the wrong graft harms in proportion to how much cache it corrupts. H3 says compression can't rescue this
because terseness removes the substrate faster than it removes the redundancy. The single positive
requires content whose useful residual is *not* near-zero — H6's procedural state — plus a layer profile
that isolates the band carrying that residual from the bands where grafting is pure harm (H4/H5). That is
a narrow conjunction, which is why it shows up in exactly one regime and needs in-domain tuning to be
stable.

The honest summary: the mechanism (write-time V differs from its re-encoding, content-specifically) is
real and measured; the usefulness (that the difference is continuity the summary dropped) is null almost
everywhere, and the lone exception is small, proxy-measured, and consistent with — but not proof of —
procedural state that brief text summaries fail to carry. The most decisive single experiment against my
own hypotheses would be the gradient-projection test in H2: if V_write and V_fresh already coincide on
the gold-logprob gradient direction, then additivity was never on the table and every downstream
question is about why the coding domain is the rare case where they don't quite coincide.
