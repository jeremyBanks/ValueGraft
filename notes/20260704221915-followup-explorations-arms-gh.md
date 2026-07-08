# Follow-up Explorations — Companion to the Semantic-Continuity Brief

**Read this only after the initial investigation (Arms A–E) is complete or
clearly under control.** Everything here reuses the harness, materials, probe
suite, and L-ladder from the main brief. Same rules apply: goals and
measurements are firm, implementation is adaptable, deviations go in
`DECISIONS.md`.

---

## Arm H — Minimal in-context summary retention (run this first; it is the cheapest and sharpest)

### Idea

Generate the summary S at the end of the full conversation, as in Arm C, so S's
cache entries are computed while the entire history is attendable. Then retain
**only** the attention sinks (first ~4 tokens) and S's tokens with their
generation-time KV. Truncate everything else — no tail, no alignment maps, no
transplant machinery. Continue the conversation on top of that.

### The matched control (B-min)

Prefill a fresh context containing exactly: sinks/system + the identical summary
text. Same tokens as H, freshly encoded with no history behind them.

H vs B-min is a single-variable experiment: **the same summary, encoded with
understanding vs encoded from scratch.** Any probe difference is attributable to
precisely one thing — the meaning carried in S's write-time activations. This is
the most direct test of the program's core hypothesis that exists, with the
fewest moving parts. If H shows no signal over B-min anywhere, that is a major
(and reportable) constraint on everything else.

### Variants

- **H-gap:** keep S at its original position indices; continue the position
  counter from the original end. Simplest; positional gap between sinks and S is
  in-distribution as long as the original conversation fit the trained window.
- **H-pack:** re-rotate S's keys down to positions immediately after the sinks
  (values untouched — they carry no rotation). Contiguous, prefix-cache-shaped,
  the more deployable form. Requires the key re-rotation utility; validate it
  with an L-ladder identity first (re-rotate by zero offset must be
  bit-identical).

Run both; report divergence between them (it measures how much the gap itself
costs, independent of encoding).

### Predictions

- H > B-min on referent, sense-disambiguation, and stance probes, in proportion
  to how much conversational understanding S's encoding absorbed.
- H = B-min = failure on evicted-fact probes (leakage control still applies;
  verify against the manifest).
- H < C overall (no tail). The H-to-C gap, measured, is the marginal value of
  the authentic tail — a number the main brief wants anyway.

### Bonus interpretation

H is the training-free, natural-language analog of gist/Beacon-style compression
tokens: if it works, self-generated summaries _encoded in context_ function as
gist tokens without any training — a bridge between the compaction problem and
the learned-compression literature.

---

## Arm G — Retrieval-based value transplant (the generalization of Arm E)

### Motivation

Arm E requires verbatim twins: every transplanted position must be a token
sequence that literally existed in the old context. Real compaction summaries
paraphrase — "turned down" becomes "rejected" — and E has nothing to offer those
tokens. The naive fix, nearest-neighbor search in value space, is unsound twice
over: (1) value-space has no privileged basis (any invertible rotation of it,
absorbed into adjacent weights, leaves the model bit-identical), so cosine
distance there is not a licensed metric; and (2) the query problem — to search
for the right old state you need a probe vector, but the new token's own state
was computed against the impoverished context, so similarity search retrieves
matches to the impoverished understanding.

The resolution: the model already contains a trained similarity engine over
exactly this space — **attention**. Q·K matching is learned retrieval over
cached state. Use it as the correspondence mechanism.

### Mechanism

One extra cross-attention pass at compaction time (one-shot; this is a
compaction technique, not a runtime RAG mechanism):

1. Prefill the compacted context (Arm B's construction) normally.
2. Keep the old conversation's cache resident (K and V; this is temporary,
   discarded after the transplant).
3. For each selected new position, at each selected layer: take the position's
   query vectors (already computed during the prefill), score them against the
   **old** context's keys at the same layer/head, softmax, and form the
   attention-weighted blend of **old values**.
4. Inject: `V_new ← (1−α)·V_fresh + α·V_retrieved`, per layer, per head. Fresh
   keys stay untouched; the cache stays contiguous; values carry no rotation, so
   no position handling.

E is the degenerate case: where a verbatim twin exists, retrieval peaks on it
and G collapses to E. C2C is the trained generalization: a fuser with parameters
where G uses the model's own frozen machinery. G sits between — zero training,
learned metric.

### Design choices (suggested defaults, adapt freely)

- **Positions:** content tokens of S and the tail only. Never sinks, never
  chat-template/special tokens.
- **Layers:** start with the middle third (where the semantic load
  concentrates); then all-layers and late-only as ablations.
- **Gating beyond global α:** entropy-gate per position/head — if the retrieval
  distribution over the old context is high-entropy (no clear match), scale α
  toward zero for that slot. This should suppress exactly the hazy, smeared
  retrievals that would inject noise. Log retrieval entropy; it doubles as a
  diagnostic for where in the old context meaning was findable at all.
- **Sink exclusion on the retrieval side too:** mask the old context's sink
  positions out of the retrieval softmax, or the dump-site will absorb attention
  mass and dilute every blend with near-zero values.

### Validation additions to the L-ladder

- **LG-1 (approximate identity):** with old context == new context (no
  compaction), G at α=1 should approximately reproduce Arm A — each position's
  retrieval should peak on itself. It will not be exact (self-attention mass
  spreads); quantify the divergence and treat it as G's noise floor.
- **LG-2 (twin agreement):** on verbatim-twin positions, G should approximately
  reproduce E. Report the divergence; large divergence means retrieval is not
  finding the twins, which is a bug or a lost-in-the-middle effect worth knowing
  about.

### Predictions

- G ≈ E on verbatim conditions (LG-2).
- G > E on paraphrased-summary conditions — the entire point. Construct these by
  post-paraphrasing S (reword, keep content) so E's alignment map goes empty
  while G still has something to retrieve.
- G's characteristic failure: retrieval haze at depth. Where the old
  conversation's relevant span sits mid-context, lost-in-the-middle degrades the
  retrieval; entropy gating should show this as suppressed α precisely there.

### Cost

Roughly one extra prefill-equivalent attention pass over the old keys, once, at
compaction time; the old cache must be resident during the transplant and is
then freed. Well within the 32 GB budget at the brief's conversation lengths.

---

## Naming (suggestions for writeups, non-binding)

If results warrant a named technique, suggested vocabulary — chosen to slot into
the existing namespace (CacheBlend, KVLink, StreamingLLM, gist tokens):

- **ValueGraft** — Arm E and the family generally: transplanting write-time
  value states onto freshly-keyed context. The graft metaphor supplies working
  vocabulary for free: the α-curve measures _graft take_, degradation below
  baseline B is _rejection_, G's entropy gating is _tissue matching_.
- **SelfGist** — Arm H: a self-generated, in-context-encoded summary retained as
  training-free gist tokens.
- Arm G, if it needs its own name: **SoftGraft** (retrieval-weighted grafting).

A paper-shaped title: _"ValueGraft: preserving write-time semantics across
compaction boundaries via value-state transplantation."_

## The contrast matrix (why these arms together answer the question)

Each pairwise comparison isolates exactly one component of "meaning survival":

| Contrast                | Isolates                                                               |
| ----------------------- | ---------------------------------------------------------------------- |
| B-min vs H              | The summary's write-time **encoding** (same text, two computations)    |
| B vs E                  | The tail's write-time **values** (same context text, payloads swapped) |
| E vs C                  | The **addresses** (old keys + gap vs fresh keys, values matched)       |
| E vs G (paraphrase set) | The **correspondence mechanism** (identity vs learned retrieval)       |

Run in that order. H is an afternoon once the harness exists; G is the only one
requiring new machinery (the cross-attention pass, ~tens of lines). Together
with the main brief, a complete factorization of where conversational meaning
lives across a compaction boundary — encoding, payloads, addresses,
correspondence — each pinned by a single controlled comparison.
