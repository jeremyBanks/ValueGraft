# Concept readout (B9): a mechanistic test of F1

## The question

F1 (FINDINGS.md) says write-time KV value-grafting *recovers the meaning* that
compaction destroyed, in proportion to how much damage compaction did
(sense/referent recover ~10–12pp; stance is a correct null). That evidence is
**behavioral** — a Sonnet meaning-judge, plus a teacher-forced gap-closure of the
gold continuation. Both infer "the concept came back" from the model's *output*.

This readout asks the sharper, mechanistic question: **does grafting actually
insert the evicted concept back into the model's internal state?** If yes, we
should be able to *read the concept out of the residual stream* under the grafted
arm — more than under compaction, approaching full context — without ever looking
at what the model says.

## The design (4 lines)

1. Per probe, build three arms exactly as gap_closure_cat.py: **A** = full
   context, **B** = compacted (summary), **E** = compacted + value-graft (α=0.75).
2. Feed the probe question; at the **probe position** (last token, whose
   next-token prediction is the answer) read the residual stream at **every
   layer** with a lens, project to vocab, and take the **logprob / rank of the
   gold concept token** (the disambiguating keyword compaction destroyed).
3. Metric per probe = concept-token logprob at the **best layer** (the layer
   where the concept is most readable under full context A — chosen on A so the
   choice is blind to E), reported under A / B / E, plus gap-closure (E−B)/(A−B).
4. **Prediction if F1 is mechanistic:** on sense/referent the gold concept is
   more present under E than B, approaching A; on stance (no compaction damage)
   the effect is null — the same dissociation F1 shows behaviorally, now read
   directly off the internal state.

## What "gold concept token" means

Each plant carries `keywords` (the disambiguating content words, e.g. Nimbus →
`["self-serve", "signup"]`). We take the first sub-token of each keyword (leading
space for the natural mid-sentence token) plus the first token of the gold phrase,
and average the lens logprob/rank across them. Single-vocab-token readout is what
a logit/J-lens can express cleanly.

## Why the probe position, and why per-layer

The last token of the rendered probe + generation prompt is the position whose
next-token distribution *is* the model's answer decision. A concept that has been
re-inserted into the state should be readable there. We scan **all layers**
because a concept can be present in the mid-stack residual stream well before it
surfaces in the final-layer logits — the logit lens' whole point — and because
grafting edits values whose effect on the residual stream is layer-dependent
(cf. the α=1.0 collapse being a per-layer phenomenon, F3). Choosing the best
layer *on arm A* keeps the layer choice from being tuned to flatter E.

## Lens choice: logit-lens (primary) vs J-lens (optional)

Per DECISIONS 07-07 (07:09 / 07:15):

- **logit-lens = the robust baseline, and the primary path here.** It is the
  well-understood 2020 method: apply the model's final RMSNorm then the
  unembedding to each layer's residual. No fitting, no extra dependency, no
  uncharacterized failure mode. This is what the CLI runs by default and what the
  headline numbers should come from.
- **J-lens = a principled *refinement* of the logit lens** (averaged-Jacobian
  transport into the final-layer basis before unembed). It is more general — it
  surfaces forward-looking / unspoken concepts a raw logit lens can miss — but the
  Nanda caveat is explicit: it is a **hypothesis-generation** tool with an
  **uncharacterized false-positive rate**, not a validated measurement. Our
  project's stance (07:15) is that we *may* lean on it harder than Nanda *because
  we hold independent behavioral ground truth* (the arms) to cross-check it — but
  it is never the sole evidence.

So the division of labor is: **logit-lens measures; J-lens generates hypotheses**
we then confirm behaviorally. The CLI exposes `--lens jlens` for cross-checking,
guarded by an import of the `jlens` package. Note the honest caveat baked into the
code: a plain J-lens forward pass ignores the KV cache, so it cannot by itself
distinguish B from E (the graft is a cache-level intervention). The J-lens path
therefore reads A vs B only and flags E as a placeholder; distinguishing E is
exactly where the cache-aware logit-lens path (which prefills the grafted snapshot
and reads its hidden states) is required. For a first pass, use `--lens logit`.

## Interpreting the output

Per-probe JSON records the full per-layer `lp_A/lp_B/lp_E` curves plus the
best-layer summary (logprob, rank, E−B, A−B, gap-closure). The printed table
aggregates by category. Read it as:

- **sense / referent:** `E − B` positive and a meaningful fraction of `A − B`,
  and "% probes E>B" well above 50% → the graft *inserted* the concept.
- **stance:** `E − B ≈ 0` (and `A − B ≈ 0`, since compaction did no damage) →
  correct null, matching F1.

The magnitude is expected to be *small* (as with the exact-token gap-closure in
the F1 corroboration): grafting restores *sense*, not verbatim surface form, so a
single-token lens logprob moves less than a meaning-judge. Direction and the
category dissociation are the load-bearing signal, not absolute logprob.

## Caveats

- Logit-lens with the final norm is a known-imperfect probe (it under-reads
  concepts that the final layers rotate into place); it is a *lower bound* on
  concept presence, which only makes a positive E>B result more conservative.
- Best-layer-on-A is one defensible selection rule; the JSON keeps full per-layer
  curves so a fixed-layer or best-on-B robustness cut is a re-analysis, not a
  re-run.
- Small-model null is expected and was used only to prove the code path (see
  self-test below); the effect, if real, needs the 30B where F1 was established.

## How this was validated locally

Self-tested on Qwen/Qwen3-0.6B (CPU), c01, 3 probes — proves the machinery runs
end-to-end and emits A/B/E numbers across all 28 layers with best-layer selection
and gap-closure. 0.6B is far too small to show the effect (gold-concept ranks sit
at ~26k–28k of the vocabulary; best layer collapses to layer 1) — exactly the
expected small-model null. The point of the self-test is the code path, not the
result.
