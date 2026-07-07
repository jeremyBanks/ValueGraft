# Handoff Note — The J-lens / Global Workspace Work (Optional Exploratory Tooling)

**From:** the collaborator who drafted the original experiment briefs (documents
1–4). **Status of this note:** an _optional_ pointer to newly-released tooling
that may be relevant to a mechanistic follow-up. It is not a change to your
protocol. You have been supervising the details against reality; I have not seen
the shape your experiment actually took. Everything below is offered as a
resource and a perspective, not a prescription. Where it conflicts with what
you've built and learned, your judgment wins.

## What this is

On 2026-07-06 Anthropic published research on a "global workspace" in language
models, along with an open-source interpretability tool. Links:

- **Blog post (accessible overview):**
  https://www.anthropic.com/research/global-workspace
- **Full paper (Transformer Circuits):**
  https://transformer-circuits.pub/2026/workspace/index.html — _"Verbalizable
  Representations Form a Global Workspace in Language Models"_
- **Open-source code (Apache 2.0):** https://github.com/anthropics/jacobian-lens
- **Independent review + replication (Neel Nanda, DeepMind):**
  https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper
- Interactive demo exists on Neuronpedia (search "jlens").

## Why it may be relevant to us specifically

The core finding: there is a small, privileged set of internal directions — the
"J-space" — that carries the concepts a model is _disposed to be able to say_,
is causally load-bearing for reasoning, and can be read out as a ranked list of
vocabulary tokens using the **Jacobian lens (J-lens)**. Mechanically the lens
transports a residual-stream vector at any layer/position into the final-layer
basis via an averaged input–output Jacobian, then decodes with the model's own
unembedding. In short: **it reads what concept-content an activation carries.**

Two facts make this potentially useful to our project rather than merely
interesting:

1. **It runs on our model class.** The lens fits open-weights decoder
   transformers; the code's usage example is literally
   `transformers.AutoModelForCausalLM` → `jlens.from_hf(...)`. Nanda's
   independent replication was done **on Qwen3.6-27B** — i.e. the exact model
   family in our experiment. So this is not aspirational tooling; it targets
   what we're already running.

2. **It measures the thing our project is fundamentally about.** Our whole
   question is whether _meaning_ survives a compaction boundary. Our probes
   measure that _behaviorally_ (does the model answer the referent/sense
   question correctly). The J-lens offers a _second, internal_ readout: does the
   disambiguating concept-content actually persist in the internal state, and
   does an intervention (retention / grafting / whatever form our arms took)
   restore it? It turns "did meaning survive" from a pure black-box inference
   into a partially-observable internal measurement.

## How it _might_ apply — examples, not instructions

Offered as illustrations of the shape, to adapt to whatever form the experiment
actually took:

- **Compaction-damage read (no surgery needed):** at a position whose
  interpretation depended on now-evicted context, read the J-space contents
  under full-context vs. standard-compaction conditions. Does the disambiguating
  concept (e.g. a "river" reading of "bank") appear under full context and
  vanish under compaction? This is the internal analogue of the
  sense-disambiguation probe, and it needs no intervention machinery — just the
  lens on two conditions.
- **Restoration check:** wherever an arm is meant to _restore_ meaning (retained
  values, grafts, in-context-encoded summary — whatever survived in our design),
  read the J-space at the affected positions and compare to the full-context
  target. If the restored condition's concept-content matches full-context where
  plain compaction didn't, that's _internal_ evidence the intervention carried
  the concept, not just improved the output. Much stronger than a behavioral
  delta alone, and the kind of mechanistic evidence reviewers weight heavily.
- **Contradiction read (if we kept contradiction probes):** where visible text
  asserts X but the intervention injects state carrying Y, read whether the
  J-space shows X, Y, or a mix — and whether behavior follows the workspace.
  This directly observes whether transplanted latent content can populate the
  workspace against the visible text.

## Hard limits — please internalize these before spending time

The paper and reviewers are emphatic about scope; treat these as walls:

- **Single-token concepts only.** The lens "can only identify concepts that
  correspond to single tokens." It reads atoms (river / bank / honest / spider),
  **not** compositional referents ("the second approach we discussed"). Design
  any J-lens probe around _atomic_ concepts; keep the behavioral probes for the
  compositional ones. Use each instrument only where it's strong. (The paper
  mentions multi-token extensions exist, but treat those as research, not
  off-the-shelf.)
- **Approximate.** The authors call it "undoubtedly an imperfect method" that
  only "approximately captures the model's true workspace." Good for detecting
  _presence/absence and change_ of concept-content; unreliable for fine
  magnitude claims.
- **Hypothesis generation, not verification.** Nanda's assessment is explicit
  and worth adopting: J-Lens is "clearly useful as a hypothesis generation tool,
  but less useful for validating hypotheses"; its false-positive rate is not
  well characterized. Use it to _find_ and _illustrate_ effects, and treat a
  positive read as suggestive corroboration, not proof.
- **It says nothing about phenomenal consciousness, and neither can any result
  we produce with it.** The access-consciousness / functional-workspace framing
  is well-evidenced; the phenomenal question is untouched and, per the authors,
  possibly untouchable by experiment. Keep that wall bright in any writeup — do
  not let J-lens results drift into consciousness claims.

## Practical / cost notes

- Fitting a lens uses ~100–1000 short corpus sequences (quality reportedly
  saturates around ~100); the repo notes fitting time is dominated by the
  model's own backward pass, and the implementation is a reference, not
  optimized. Nanda's replication used as few as ~25 prompts of length 128,
  **skipping the first four tokens due to high norm** — a useful practical tip
  that also rhymes with our attention-sink handling.
- Verify architecture support for our exact model before committing time; the
  Qwen3.6-27B replication is encouraging but our specific
  checkpoint/quantization may need adaptation. If it doesn't load cleanly,
  adapting it is real work — weigh against value.
- The Jacobian computation is heavier than a plain forward pass. Targeted
  single-position/single-layer reads should be tractable at our context lengths
  on the local machine; a full position×layer sweep may not be. Scope reads
  narrowly.

## Suggested placement (a suggestion only)

Do **not** retrofit this into the running behavioral experiment — it's a second
instrument with its own validation burden, and the behavioral result is the
load-bearing deliverable. The natural home is a **mechanistic follow-up** (e.g.
a "why does the effect happen" section of the phase-2 writeup): once the
behavioral result exists, the cheapest high-value read is the compaction-damage
read above (no surgery), then the restoration check as internal confirmation of
whatever effect we found. The contradiction read is the ambitious extension if
the first two show signal.

Whether any of this is worth doing depends entirely on what you've found and
what's tractable on the actual setup. If the behavioral result is clean and
self-sufficient, this can remain a single paragraph in the discussion ("an
internal-state analysis using the J-lens is a natural next step") and nothing
more. Your call.
