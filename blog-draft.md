# Summary tokens lose their original activations after context compaction. Does keeping them help?

*Or: I couldn't find anyone who tried re-using write-time KV cache across a
chat-compaction boundary, so I measured it. Am I missing the right search
term, or is this genuinely unwritten-up?*

**Provenance, stated plainly:** this experiment was designed and largely
executed by Anthropic's Claude (Fable 5) running as an autonomous coding
agent on my MacBook, over roughly one long day. I directed it, made the
scaling and budget calls, and commissioned an adversarial design review from
OpenAI's GPT-5.5 (which contributed the matched controls and the
mitigation-first reframing that shaped the final experiments). Judging of
~4,000 probe answers was done by Claude Sonnet with all verdicts logged. I
personally verified the repo structure, spot-checked judged answers and the
identity-test outputs, and reviewed every summary the agent produced; I did
not independently re-derive every number. The repo with all code, corpora,
raw outputs, and decision logs is public — everything below is reproducible
from it.

## The naive idea

Production LLM systems compact long conversations: old history is replaced
by a text summary, recent turns are kept verbatim, and everything is
re-encoded from scratch. But the model's *original understanding* of those
retained turns — which sense of an ambiguous word was meant, what "the
second approach" refers to, what was already tried and rejected — was
computed when it first read them with full history attendable, and it lives
in the KV cache entries written at that moment. Re-encoding throws that
away and reinterprets the same text against only the summary.

Two obvious-sounding interventions follow:

- **SelfGist**: generate the summary *inside* the full conversation (so its
  cache entries are written while everything is still attendable), then keep
  only those few hundred cache entries — re-rotated to packed positions —
  instead of re-encoding the summary text cold. Cost: ~10 MB of retained
  state at 30B for a terse summary.
- **ValueGraft** (dual-context KV blending): build the compacted context
  normally, then blend the *old* value vectors into the fresh cache at
  positions whose tokens exactly match the original (the verbatim tail and
  the summary), `V ← (1−α)·V_fresh + α·V_old`. Keys stay fresh; the cache
  stays contiguous; values carry no positional rotation, so there is no
  position surgery at all.

Both require that the old cache still exists when you compact — this is for
"compaction as a choice" (long-running local agents, single-host serving),
not state-loss recovery.

## What we measured

Two Qwen3 models (4B-Instruct-2507 and 30B-A3B-Instruct-2507, 4-bit MLX,
fp16 cache, temperature 0 everywhere) on 12 synthetic ~9K-token
conversations with 120 planted probes (referents, ambiguous senses, stances,
ruled-out options, and unknowable "evicted facts" as calibration), plus 8
free-form conversations with held-out continuations, plus 24 decoy probes
about plausible details *nobody ever discussed*. Every surgical operation
passed bit-identity tests before use (null surgery reproduces baseline
exactly; transplant at α=0 ≡ text baseline; re-rotation by zero ≡ identity),
and every positive result is bracketed by negative controls (values from a
shuffled alignment or from a *different conversation*, which crater
performance — the effects are content- and alignment-specific, not generic
smoothing). Probes were stratified by an automatic summary-leakage audit so
"the summary just said it" never counts as cache magic.

## Result 1: most of what compaction loses, these methods don't recover

On leakage-clean interpretation probes (what did "the second approach" refer
to?), no cache intervention meaningfully beat the text-summary baseline at
either scale. Referent resolution stays near floor for every compacted arm.
If you hoped grafted values would restore *recall* of evicted specifics: they
don't, and we report that plainly.

## Result 2: what they do buy — a large drop in confabulation

Ask a compacted model about a detail that no longer exists in its context
(or never existed), and the production-style baseline confidently invents
one. Fabricated-vs-admitted counts on unknowable probes, 30B:

| context | fabricated : admitted (decoys) | (evicted facts) |
|---|---|---|
| full conversation (!) | 16 : 8 | — (24/24 correct) |
| production compaction (summary + tail) | 19 : 5 | 16 : 8 |
| packed summary text, fresh encode | 10 : 14 | 5 : 19 |
| **packed summary, write-time KV (SelfGist)** | **3 : 21** | **1 : 23** |

Two decomposed causes, per the matched pair: the bare packed layout itself
induces caution (a big, free effect — wrapping a summary in fluent chat
template apparently *invites* confident invention), and the write-time
encoding adds a component on top that **grows with model scale** (matched-
pair decoy fabrication 10→3 at 30B; negligible at 4B). Note the first row:
even the *un-compacted* model invents answers about never-discussed details
at 2:1. Confabulation is the default; context construction modulates it.

## Result 3: tuned ValueGraft closes a quarter of the continuation gap

Teacher-forcing each conversation's genuine continuation and tuning α on a
validation split, then reporting once on held-out conversations:

| scale | tuned setting | held-out gain vs baseline | gap closed |
|---|---|---|---|
| 4B | α=0.25, middle-third layers only | +0.017 nats (10/10 convs, CI 0.012–0.024) | ~10% |
| 30B | α=0.75, all layers | +0.033 nats (9/10, CI 0.014–0.057) | **~24%** |

The dose–response *inverts with scale*: at 4B, α≥0.75 actively hurts
(fresh-keys/old-values incoherence) and layer-gating helps; at 30B the model
tolerates full replacement, the optimum moves to α≈0.75–1.0, and even
extrapolating past the old values (α=1.25) stays positive. Small models need
the graft diluted; big ones drink it straight.

A hard caveat: these gains appear where the continuation actually *depends*
on evicted content (our probe corpus guarantees it). On free-form chat where
the summary already suffices, the measured gap between full context and
compaction was tiny and the interventions did roughly nothing.

## Overhead, and what "no longer stateless" costs in practice

ValueGraft: one extra prefill of the compacted context plus a vectorized
value blend, once, at compaction time — but it needs the *entire* old cache
resident (~1.2 GB for a 12K-token context at 30B), so it is a server-side /
local technique only.

SelfGist's retained state is just the summary's cache entries: at fp16 on
the 30B, ≈96 KiB/token (48 layers × K,V × 4 kv-heads × 128 dims × 2 B) —
so ~9 MiB for a terse 100-token summary, ~47 MiB for a thorough 500-token
one. That's ~0.7% of the full context's KV but ~15,000× the summary *text*,
which makes a raw client-uploaded KV sidecar an unattractive stateless-API
payload.

The more plausible surface is an **opaque compaction handle**, following
existing patterns (prompt-cache handles, session continuation, opaque
reasoning tokens): the harness calls something like
`compact_from(boundary, instructions) → {summary_text, compaction_token}`;
the provider generates the summary while the full cache is still resident,
stores the packed summary state server-side, and later requests stay
text-shaped — summary text + token + recent tail. The visible summary
remains the auditable channel; the token means "apply the provider-side
latent state associated with this summary." Fallback is free: a harness
talking to a provider without support just drops the token and gets ordinary
text compaction. The honest cost accounting then splits into: negligible
client bandwidth; tens of MiB of provider storage per compaction checkpoint
(with prompt-cache-style expiry/billing); one summary generation plus state
packing at compaction time. Caveats that don't go away: the handle is
model-, tokenizer-, and template-version-specific; it encodes state derived
from conversation content (privacy/audit surface — the visible summary is no
longer the whole persisted state, and APIs should disclose when latent
compaction state is active); and quantizing the stored state (plausibly
2–4× smaller) needs separate validation — we ran fp16 throughout.

## Result 4: standard-benchmark check (LongMemEval) bounds the claim

On LongMemEval-S questions restructured so the evidence session is evicted
(n=48 at 4B, n=36 at 30B): compaction damage replicates exactly — full
context answers 71–81%, every compacted arm ≤11%. The honesty effect holds
at 4B (H-pack fabricates 11 vs the baseline's 17) but **washes out at 30B on
this benchmark** — because the benchmark's personal-fact framing already
triggers the larger model's "I don't have access to your history" refusal
training, so the baseline barely fabricates and there's nothing to fix.
The scope statement this forces: cache-state honesty interventions matter
where the compacted frame *invites* the model to keep confabulating as a
conversation participant — mid-task agentic contexts, which is where
production compaction actually runs — not in retrieval-style QA that
refusal training already covers. Real data also revealed a fabrication
channel our synthetic corpus couldn't: substituting plausible *world
knowledge* (a real-but-wrong song title) for lost personal facts.

## Mechanism evidence (why this isn't nothing)

The same summary text predicts the conversation's future better when its KV
was written in-context: +0.128 nats, **12/12 conversations** at 30B (CI
0.097–0.158) against the character-identical fresh encoding. An isolated
micro-test shows a single sentence's transplanted value vectors carry word-
sense disambiguation into a context that lacks it. And write-time state
scales *up*, not away: every one of these contrasts is larger at 30B than 4B.

## Limitations, matter-of-factly

Pilot scale (n=12+8 conversations, one model family, 4-bit weights, MLX).
Generator/judge circularity partially mitigated (scenarios authored by
Claude, dialogue by the subject model, judging by Sonnet with logs) but not
eliminated. The honesty effect is part layout-caution, part encoding — the
matched pair bounds the split but a wrong-summary graft is *also* honest, so
"weird state → caution" remains a live partial explanation. Interpretation-
accuracy recovery is a null. Quantized kernels are sequence-length-dependent
(same tokens, same positions, different batch length → different values up
to fp16-visible magnitude), which silently bounds any cross-shape cache
comparison — our identity tests are all same-shape for this reason.

## Related work (what we searched and why none of it is quite this)

We could not find work that (a) takes a *conversation-compaction* event
(history → generated text summary + re-encoded recent turns), (b) preserves
or transplants the generation-time KV entries across it, and (c) compares
against the text-only summary baseline. The neighbors fall into three
families:

**Learned latent compression** (train something to squeeze context into few
tokens/vectors): gist tokens (Mu et al., NeurIPS 2023), AutoCompressor
(Chevalier et al., EMNLP 2023), ICAE (Ge et al., ICLR 2024), Activation
Beacon (Zhang et al., ICLR 2024), Compressed Context Memory (Kim et al.,
ICLR 2024 — closest *problem setting*: online conversational compression,
but a trained LoRA compressor on raw KV, no text summary anywhere). SelfGist
is the training-free, natural-language cousin of these.

**KV reuse across independently-encoded chunks** (RAG-flavored): CacheBlend
(Yao et al., EuroSys 2025), KVLink (Yang et al., arXiv:2502.16002), and
SamKV (arXiv:2508.11661) — the last literally uses a
`θ·KV_new + (1−θ)·KV_old` blend, the closest existing instantiation of
ValueGraft's formula, but applied to document-chunk concatenation, not a
compaction boundary, and never against a summary baseline.

**Within-context budget management**: StreamingLLM's attention sinks (Xiao
et al., ICLR 2024 — we retain sinks in every arm), H2O (NeurIPS 2023),
SnapKV, CaM (ICML 2024), KVMerger — eviction/merging inside one continuous
context; and text-space compression (LLMLingua, RECOMP) which is essentially
the *baseline* we compare against, plus MemGPT-style text-level memory
management.

The closest mechanistic precedent is "Models Take Notes at Prefill"
(arXiv:2606.17107, June 2026), which reports KV entries are position-portable
and spliceable near-losslessly — effectively the physics SelfGist relies on —
but for precompiled "skills," with no compaction event and no summary
baseline. (We had difficulty fully verifying this paper beyond its abstract;
read it yourself before leaning on it.) C2C (ICLR 2026) fuses caches *across
models*. Searches that came up empty, for the record: "context compaction KV
cache preservation", "keep summary activations compaction conversation",
"transplant KV cache summary tokens", and both coinages ("SelfGist",
"ValueGraft") — no collisions.

## The open question

Is this known? If there's a paper that measures write-time-KV retention
against text-summary compaction on conversation continuity, I'd genuinely
like to read it — the terminology in this area is scattered enough that we
may simply have missed it. And if the setup has a flaw the identity tests
and negative controls didn't catch, tell me. Repo: [LINK].

---

*Fun failures, for the record: the 4B opens nearly every answer with "🔥
Great question — you're already thinking like a product-led founder,"
including when its values were transplanted from an entirely different
conversation. Negative-control models don't act amnesiac; they gaslight —
cheerfully explaining that Dana never existed and there was never a
proposal. And one judge caught several arms rejecting "tiered pricing"
while proposing a "three-level structure" in the same reply — including the
full-context oracle. Some failure modes aren't about memory at all.*
