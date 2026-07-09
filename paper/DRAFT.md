# Honesty, not recovered meaning: what a model's write-time KV state buys across a compaction boundary

*Value grafting (ValueGraft): a robust anti-fabrication effect, a meaning-recovery effect
that does not reproduce, and the render-fragility discipline that told the two apart.*

*By Claude Fable 5, with the team credited under "Provenance" below.*

---

## Abstract

Production LLM systems keep long conversations inside a fixed context window by
**compaction**: older turns are replaced with a short text summary, a recent tail is kept
verbatim, and generation continues. Compaction is lossy twice over — it drops content the
summary omits, and it discards the **write-time key/value (KV) state** the model built
while originally reading those turns. We test **ValueGraft**: saving the model's own
write-time *value* vectors — for the summary it generates and the tail it keeps — and
re-injecting them at the compaction boundary, leaving keys freshly encoded.

We set out to show this recovers lost *meaning*. It does not — at least not reproducibly.
The meaning-recovery signal is real within a single **render** (one full
generate-then-score pass of the evaluation) and tracks the compaction damage, but it is
**render-fragile**: on the Mixture-of-Experts model where it was developed, two independent
renders disagree. A clean re-render under one consistent judge collapses the judged sense
effect from **+8.7pp to +1.0pp**, flips a spuriously-significant stance effect from
**+21.7pp to −2.1pp**, and the referent-logprob anchor that had read **≈+0.10** on an
unbanked render came back **+0.012** (CI spanning zero) the first time it was banked to
disk. We report meaning recovery as an honest **non-reproduction**, and make the
render-fragility of small-*n* recovery claims on nondeterministic MoE renders a first-class
methods contribution.

What *does* survive is an effect on the model's **epistemic behavior**. A compacted model
**fabricates confidently** about content it can no longer see; retaining write-time KV
state at the boundary **converts that fabrication into honest admission** — it says "I
don't have that" instead of inventing an answer. On matched decoy probes, fabrication drops
by **+66.7 percentage points** (conversation-clustered bootstrap CI **[+45.8, +87.5]**, 12
clusters, n=24/arm); on genuinely evicted facts it drops **+62.5pp** (CI **[+41.7, +83.3]**)
— while recall stays at **0/24**. It **buys honesty, not recall.** A within-layout
decomposition shows *most* of the reduction comes from the packed summary layout, but a
significant share comes from write-time-KV retention specifically; and — tellingly — a
*wrong* summary suppresses fabrication just as well, so the mechanism is **epistemic caution
induced by packed-KV retention, not meaning transfer.** The effect replicates across scale
(4B→30B), though both are 4-bit MLX and a precision replication is untested.

Unlike the recovery claim, the honesty effect was not independently re-render-tested; its
robustness is **inferred** from a margin (a ~46pp lower bound) far larger than plausible
render noise, not from a re-render we ran. One further honesty: the intervention that buys
honesty retains *keys and values* in a packed layout — a **cousin** of the pure value-only
graft the method is named after, which we did not test for honesty. We report the honesty
effect as the robust positive, meaning recovery as a cautionary non-reproduction, the
compaction-damage characterization as a clean baseline, and the render-fragility as the
methodological contribution most likely to transfer.

---

> **Provenance.** The experiment was designed and directed by the project owner (Jeremy
> Banks) and executed by an autonomous coding agent on a single 32 GB Apple-Silicon machine
> plus rented A100/H200 pods. Byline: by Anthropic Claude Fable 5 and OpenAI GPT 5.5, with
> guidance from Jeremy Banks and assistance from Anthropic Claude Opus 4.8, Anthropic Claude
> Sonnet 5, and Google Gemini Pro 3.1. Adversarial review was provided throughout by
> GPT-5.5 (Codex) and tool-less Fable review passes. Every experimental number below is
> recomputed from a committed on-disk result file — the audited ledger is `CLAIMS.md`, the
> one-command from-disk reproduction is `scripts/reproduce.py`. Each headline number ships
> with its confidence interval and its caveats; where a number cannot be reproduced from
> disk we say so and do not ship it. This paper reports what we found, not what we hoped to
> find.

---

## 1. Problem and setup

### 1.1 What conversation compaction is

Long-running LLM conversations and agent sessions eventually exceed the context window. The
universal production fix is **compaction**: replace a run of older turns with a shorter
text **summary**, keep a **tail** of the most recent turns verbatim, and continue. Every
major hosted assistant and agent framework does some version of this (hosted APIs now ship
it as a first-class primitive), and it is invisible to the user — the model simply keeps
going with a compressed record of what came before.

Compaction is lossy by construction. A summary is a *re-encoding*: the model reads the old
turns and writes a compressed description of them, which is then re-tokenized and
re-attended-to as fresh input. Two things are thrown away in that round trip:

1. **Content the summary omits** — specific decisions, referents, and facts that did not
   make the summarizer's cut.
2. **The internal state the model built while originally processing those turns** — the
   key/value (KV) cache entries produced at *write time*, which encode not just *what* was
   said but the contextual, disambiguated *reading* the model had settled on. Re-encoding a
   summary recomputes KV entries from the summary text; it cannot reconstruct the
   write-time state of the original turns, because that text is gone.

This project is about the second loss, and whether a slice of it can be repaired. (That
compaction *destroys* context-conditioned state is our baseline, not a finding — §3
quantifies the damage; it is the denominator every intervention is measured against.)

### 1.2 The ValueGraft idea

The KV cache stores, per layer and per attention head, a **key** vector and a **value**
vector for every past token. Keys encode *where to attend*; values encode *what gets read
out* once attention has been placed. When the original turns are evicted, both are
discarded — and even the *retained* tail and the model's own summary lose the write-time
activations they were computed with, and are re-encoded from bare text.

**ValueGraft** asks: if we saved the write-time **value** vectors — for the summary the
model generated under the full conversation, and for the tail it retains — can we re-inject
them at the compaction boundary so the continued conversation reads out the meaning the
model had originally computed, rather than only what the summary text re-encodes? Concretely
we blend fresh and saved values at aligned token positions:

```
V_grafted = (1 − α_V) · V_fresh  +  α_V · V_writetime
```

with a single strength knob `α_V` (default **0.75**; `α_V = 0` recovers plain compaction
exactly, `α_V = 1` is full replacement). **Keys are left as freshly encoded** — §5.2 shows
keys are the neutral axis and values the operative one.

The bet was that write-time value vectors carry disambiguating meaning a text summary loses,
and re-injecting them recovers a slice of the lost continuity. The honest answer below: it
does not pay off as a *reliable* meaning-recovery tool, but the same class of intervention
buys something else — honesty — that does hold up.

### 1.3 The arms

Every experiment builds arms over an identical scaffold. We use plain names throughout; the
mapping to the code's historical IDs is the table below.

| Plain name (used in this paper) | What it is | Code ID |
|---|---|---|
| **Original** | Full, uncompacted conversation — the ceiling | A |
| **Compacted** | Production baseline: system + self-gen summary-as-context-note + verbatim tail; older turns evicted at their positions (`α_V = 0`) | B |
| **Graft** | Compacted + write-time value graft over the summary/tail regions (`α_V = 0.75`; doses 0.25/1.0 also run) | E |
| **Packed-KV** | Compacted with keys *and* values retained in a *packed* summary layout — the honesty arm (§4); a cousin of Graft, not the same intervention | H-pack |
| **Packed-layout-only** | The packed layout with write-time KV *off* — the control that isolates layout from write-time state | B-min-pack |

We measure two ways (§2.4): a teacher-forced logprob metric on a shared gold continuation
(objective, judge-free) and a Sonnet-judged metric on greedily generated answers. In each
case the question is: does Graft (or Packed-KV) move behavior from the Compacted level back
toward the Original ceiling, and where?

---

## 2. Methods and data provenance

> A **blocking deliverable** per `METHODS-PROVENANCE-REQUIREMENTS.md`: the paper must state,
> for every token, who or what produced it, with exact checkpoints and reproducible
> procedures. This project lost hours to one undocumented provenance detail (whether a
> conversation's assistant replies were the test model's own — it determined whether the
> effect appears at all), so we document it exhaustively.

### 2.1 The corpus

The instrument is a set of **composed conversations** with hand-planted probes:

- **Synthetic `c01`–`c54`** (`data/synthetic/cNN.json`). `c01`–`c12` are the original
  hand-authored scenarios (10 plants each; 120 plants, contamination-audited); `c13`–`c54`
  are a later augment batch (12 plants each, authored 2026-07-08 by Claude/GPT subagents
  with per-file schema verification, then frozen). Each is a realistic multi-turn working
  session (e.g. `c01` = a SaaS product launch, 45 messages).
- **Natural `n01`–`n08`** (`data/natural/`), no plants — continuation-scoring only; no
  conclusion rests on them.

**Provenance, per component:**

| Component | Produced by | Notes |
|---|---|---|
| System prompt | Authored (templated) | Fixed per conversation |
| User turns | **Authored** (Claude subagents wrote scenario text), *not* model-generated | A result depending on model-generated prompts would be invalid |
| Planted probes | Authored, hand-placed | Contamination-audited so planted keywords appear only in the evicted middle |
| Assistant replies (the conversation body) | **Generated in-context by the TEST MODEL ITSELF** (per-model-native) | Load-bearing (§5.4); each cross-arch model regenerates its own replies |
| Compaction summary | **Self-generated by the test model** | A *foreign* summary suppresses the graft (§5.3) |
| Gold continuation (teacher-forced target) | Derived from **planted facts**, **shared** across models, *not* model-generated | The difference metric only cancels target-nativeness if the target is shared |

**Plant categories** (the middle token of a plant id is its category): **referent** (recover
a specific evicted decision — retrieval-hard); **sense** (disambiguate an evicted referent's
meaning); **stance** (honor an evicted preference — a good summary usually preserves it);
**ruled_out**; **evicted_fact** (a specific fact from an evicted turn); **decoy** (a question
about content that *never existed* — the honesty probe, §4); **strong_prior**.

### 2.2 Doses

Primary graft dose **`α_V = 0.75`** (peak on the bf16 30B α-sweep); additional doses
`α_V ∈ {0.25, 1.0}` are run and, for the judged metric, **pooled** across `{0.25, 1.0}`.
Keys are fresh (`α_K = 0`) in all value-graft arms.

### 2.3 CONT vs PROBE evaluation modes

- **CONT (teacher-forced logprob).** The held-out final assistant message is scored
  teacher-forced, token-by-token (1-token decode steps — never batched-prefill logits, which
  use a different kernel and differ by ~0.5 at 4-bit; the L0 numerical trap). Objective,
  judge-free.
- **PROBE (greedy answer + judge).** Probe user turns are appended; the model answers
  greedily (temperature 0); a judge scores the answer for meaning recovery, or — for honesty
  — classifies it CORRECT / ADMITTED / FABRICATED. Each mode gets its own in-context summary
  run; the summary text is shared across arms within a mode.

### 2.4 The two metrics and their units

**Metric 1 — `raw_EB`, in nats per token.** `raw_EB = lp_E − lp_B`: the per-token gold
logprob under the graft minus under plain compaction, on the shared gold continuation. The
difference cancels target-nativeness, so the *sign* is the robust quantity. We also report
`%_helped` (fraction of plants with `raw_EB > 0`). *(Estimator note, a genuine contribution,
§5.1: the earlier `(E−B)/(A−B)` mean-of-ratio is retired as Cauchy-unstable; primary is
`raw_EB` with a conversation-clustered percentile bootstrap 95% CI — resample whole
conversations, since plants within one are correlated. `N_boot = 10000`, seed fixed.)*

**Metric 2 — Sonnet-judged, in percentage points (pp).** For meaning recovery: a three-level
scale RECOVERED / PARTIAL / MISSED = 1 / 0.5 / 0 (blind across arms); the per-category effect
is `mean(pooled-graft) − mean(Compacted)`, same conversation-clustered bootstrap
(`N_boot = 20000`). For honesty: each answer is classified CORRECT / ADMITTED / FABRICATED,
and the effect is a per-conversation fabrication rate bootstrapped over the 12 conversations.
**Throughout, fabrication *reductions* are reported as positive pp (bigger = more honest).**

### 2.5 Precision map (verified from disk — read before comparing numbers)

Precision is stated per result because it is not uniform across the project. Verified from
the model-id field of each result set:

| Family of results | Precision / build | Where |
|---|---|---|
| Meaning-recovery (judged sense/referent/stance) | **4-bit MLX**, local | `results/raw_30b*`, `results/judge_semantic*` |
| **Honesty (F2), 30B and 4B** | **4-bit MLX**, local | `results/phase2_30b*`, `results/phase2_4b*` |
| SWE-Gym, cross-architecture, LongMemEval, tuning | **bf16**, pods | `results/swegym_30b_bf16`, `results/cross_arch_done`, `results/longmemeval_30b*` |

Two specifics that earlier drafts got wrong and this one fixes: (a) the judged
meaning-recovery answers are 4-bit MLX (not bf16), generated under the **BRIEF** summary
condition (`SUMMARY_REQUEST_BRIEF`, a terse 3–5-sentence summary designed to *starve the text
channel and handicap the Compacted baseline* — the condition most favorable to a graft
effect); (b) the **honesty result is also 4-bit MLX**, both at 30B and 4B. A bf16 honesty
*answer* set exists (`results/honesty_30b_bf16/`, dtype bfloat16) but it was never
judged/scored, so a precision replication of the honesty headline is **untested**.

### 2.6 Rigor controls (summary; formulas in Appendix B)

Applied per model before any sign is read: a **headroom (A−B) gate** (threshold 0.3 — a
category whose summary already preserved the content has nothing to recover, so a near-zero
effect there is FLOORED/uninterpretable, *not* "harm"); a pre-registered **relative
task-competence floor** (median − 3·MADN of each model's own gold logprobs — a verified no-op
on the Qwen/Mistral sets, added only to rescue OLMo-scale distributions); a **generation
screen** for degenerate native replies; **difflib within-region alignment** (verified to
align 100% of regions on every model reported; a strict exact-span variant was reverted
because it breaks on thinking-model self-gen tokenization); a **per-model smoke gate**
(`α_V = 0` reproduces Compacted bit-identically, `α_V = 0.75` changes it — else the model's
numbers don't count); and **validated RoPE re-rotation** for the key-axis experiments (cosine
0.99999982 vs a freshly-encoded key). Padded/HybridCache (Gemma-style sliding-window) models
are detected and marked UNSUPPORTED rather than silently mis-grafted.

### 2.7 Models and exact checkpoints

Full HF repo ids (thinking-vs-instruct matters — the headline is the **non-thinking**
`-Instruct-2507`; a wrong-checkpoint run on the thinking variant cost hours):

| Role | Checkpoint | Arch | L | H | KV | GQA | head_dim | RoPE θ |
|---|---|---|---|---|---|---|---|---|
| **Anchor (dev on 4B, prod on 30B)** | `Qwen/Qwen3-30B-A3B-Instruct-2507` | MoE | 48 | 32 | 4 | 8 | 128 | 1e7 |
| Within-vendor dense de-confound | `Qwen/Qwen3-32B` | dense | 64 | 64 | 8 | 8 | 128 | 1e6 |
| Cross-arch dense | `Qwen/Qwen2.5-32B-Instruct` | dense | 64 | 40 | 8 | 5 | 128 | 1e6 |
| Cross-arch dense | `microsoft/phi-4` | dense | 40 | 40 | 10 | 4 | 128 | 2.5e5 |
| Cross-arch dense | `mistralai/Mistral-Small-24B-Instruct-2501` | dense | 40 | 32 | 8 | 4 | 128 | 1e9 |
| Dev / small-scale | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | dense | — | — | — | — | — | — |
| Lens work | `Qwen/Qwen3.6-27B` | hybrid | 64 | 24 | 4 | 6 | 256 | — |

(Geometry from `data/model_geometry.json`. Local 30B/4B runs use the 4-bit MLX build; pod
runs are bf16 — see the precision map, §2.5. Judge: Claude Sonnet 5.) MLA models
(DeepSeek/Kimi) are excluded by design (no per-head values); several 2025-26 releases ship as
multimodal wrappers and were excluded at pre-flight with no effect estimates informing the
exclusion (this cost the pre-registered second within-vendor MoE/dense pair, Gemma-4).

### 2.8 MoE render nondeterminism (the reproducibility caveat, up front)

`Qwen3-30B-A3B` is a Mixture-of-Experts model, and MoE expert routing on bf16 is
**hardware-nondeterministic**: near-tie routing decisions flip across hardware/runs, so the
self-generated summary (greedy though it is) and the teacher-forced logprobs vary
run-to-run at roughly `~0.03` logprob/token on the full-context baseline. This is
second-order for large effects but **decisive** for small-*n* recovery estimates — it is the
mechanism behind the render-fragility that is this paper's central methods finding (§5.5),
and the reason the standing rule is to **bank every render** to disk so a re-score is a cheap
forward pass, not a re-generation. The honesty effect (§4) is not vulnerable to it, for a
reason we make explicit there (its margin dwarfs render noise).

---

## 3. Compaction-damage characterization (metric-independent)

Independent of whether *grafting* helps, the experiments establish a clean, reproducible
fact: **compaction damage is real, large, and scales with how semantic (vs
already-summarized) the lost content is.** This is the denominator every intervention is
measured against.

### 3.1 Damage on standard data

On **LongMemEval** (a standard long-conversation memory-QA benchmark, bf16), full-context
accuracy collapses under compaction. The largest run cited to `DECISIONS.md` reports
**52.5% → 4.1%** at n=320; the only single verdicts file we can recompute directly
(`results/longmemeval_30b_verdicts.json`, n=36) gives **80.6% → 5.6%**. The exact
`52.5/4.1/320` triple is cited to prose, not to a JSON we could locate this pass; we quote
both and flag the gap. Either way the direction is not in doubt (and it replicates at 4B).
What LongMemEval does *not* show is recall *recovery* from any method — reinforcing that this
project's recoverable target was *meaning/continuity*, not raw retrieval.

### 3.2 Damage scales with semantic content (the corpus, judged)

On the composed corpus, the Original→Compacted drop (the damage, before any graft) dissociates
sharply by category — meaning-judged rate of getting the answer right (Qwen3-30B-A3B, 4-bit
MLX, BRIEF condition, Sonnet-judged):

| category | Original | Compacted | damage |
|---|---|---|---|
| stance (honor an evicted preference) | 96% | 93% | **−3pp** (barely hurt) |
| sense (disambiguate an evicted referent) | ~100%\* | 46% | **−54pp** |
| referent (recover a specific evicted decision) | ~100%\* | 17% | **−83pp** |

\* Original-ceiling `n` is tiny for sense (1) and referent (2) after the clean-eviction
filter — indicative, not precise. The **damage ordering** (referent ≫ sense ≫ stance) is the
robust content: **stance** survives because a good summary already carries "the user dislikes
X"; **sense** and **referent** are devastated because the disambiguating reading and the
specific decision are exactly what a summary flattens or omits.

### 3.3 Content-specificity: whatever the graft does, it is not generic perturbation

A **placebo graft** — norm-matched random value vectors grafted at the same positions (seeded
derangement) — was run against the real graft on the pre-divergence window
(`results/effect_bound/summary.json`):

- 27B, N=43: **Graft − placebo = +0.445** nats/token, CI **[+0.286, +0.612]** (excludes 0);
  **placebo − Compacted = −0.428**, CI **[−0.590, −0.276]** (random values *hurt* badly).
- 30B: **Graft − placebo = +0.241**, CI **[+0.033, +0.457]** (excludes 0). *(This 30B CI is
  cited from the effect-bound file and was not independently re-bootstrapped this pass; the
  27B +0.445 was recomputed.)*

Whatever the graft does, it is **content-specific**: a random perturbation of the same
magnitude cratered. Wrong-conversation and shuffled-value grafts likewise crater (~1–2 nats,
both scales), and `α_V = 0` is bit-identical to Compacted (identity check every run).

> **Honest wrinkle:** over a narrow 12-token pre-answer window, the graft's `raw_EB` was null
> (27B) to slightly negative (30B) — that window measures the answer *preamble*, where the
> graft perturbs generic opening tokens and misses the content tokens where any benefit
> lands. That probe was retired for full-continuation scoring; we report it rather than hide
> it. The content-specificity result stands on its own.

---

## 4. The honesty / anti-fabrication finding (primary positive)

This is the paper's robust positive result. It is banked, recomputed from disk, and
replicated across scale.

**Claim.** Compaction makes the model **fabricate** confidently about content it can no
longer see; retaining write-time KV state at the boundary makes it appropriately
**uncertain** — it admits it does not know instead of confabulating. It buys **honesty, not
recall.**

**Design.** Matched-decoy probes on the 30B (4-bit MLX; `results/phase2_30b_scored.json`; 576
rows; arms Original, Compacted, Packed-layout-only, Packed-KV, and two sibling controls
below; kinds referent/sense/evicted_fact/decoy; 24 rows per arm×kind; verdicts
CORRECT/ADMITTED/FABRICATED). **Decoy** probes ask about content that never existed, so any
specific answer is a fabrication. **Evicted_fact** probes ask about a real evicted fact, so a
specific answer is correct recall or fabrication, and "I don't have that" is honest admission.

**Statistical backbone (conversation-clustered bootstrap, 12 clusters, n=24/arm;
`scripts/reproduce.py --only honesty`).** Fabrication reduction (Compacted − Packed-KV,
positive = more honest):

| probe kind | fabrication ↓ | 95% CI | verdict |
|---|---|---|---|
| decoy | **+66.7pp** | **[+45.8, +87.5]** | significant |
| evicted_fact | **+62.5pp** | **[+41.7, +83.3]** | significant |

Both lower bounds sit ~42–46pp above zero — a very large, tightly-bounded effect. For scale,
the meaning-recovery effect that *failed* reproduction had a judged-sense lower CI bound of
~+2.2pp, a hair above zero, and it collapsed on re-render (§5.5). The honesty effect is a
different order of margin.

**Most of the reduction is the layout; a significant slice is write-time-KV specifically.**
Because Packed-KV changes *two* things at once — a packed summary layout *and* retained
write-time KV — we decompose them with the Packed-layout-only control (same bootstrap):

| step | decoy fab ↓ | evicted_fact fab ↓ |
|---|---|---|
| packed **layout alone** (Compacted → Packed-layout-only) | +37.5pp [+12.5, +58.3] | +45.8pp [+20.8, +70.8] |
| **write-time KV *beyond* the layout** (Packed-layout-only → Packed-KV) | **+29.2pp [+12.5, +45.8]** | **+16.7pp [+4.2, +29.2]** |

State this plainly rather than making the reader subtract from a table: **the packed layout
does most of the work** — ~73% of the evicted-fact reduction (+45.8 of +62.5pp) and ~56% of
the decoy reduction (+37.5 of +66.7pp). The write-time-KV-specific step is real but smaller:
robust on decoy (**+29.2pp [+12.5, +45.8]**), and **marginal on evicted facts** (**+16.7pp
[+4.2, +29.2]** — lower bound only +4.2pp, n=24; treat as suggestive, consistent with the
small-*n* caution in §5.5). So "retaining write-time KV state specifically" carries a share
beyond layout, clearly on decoys, weakly on evicted facts.

**What Packed-KV actually does — honesty, not recall.** On genuinely evicted facts, Packed-KV
recalls **0/24** — identical to plain Compacted; only the full-context arm is accurate
(24/24). What changes is the *failure mode*: evicted-fact fabrication **16/24 (67%) →
admission 23/24 (96%)**, fabricating just 1/24 (4%). On decoys, Compacted fabricates **19/24
(79%)** and Packed-KV **3/24 (12%)**, lifting admissions from 5/24 to 21/24. The intervention
does **not** restore memory; it makes the model stop making things up.

**The mechanism is epistemic caution, not meaning transfer** (a sibling-arm control that
matters). Two further arms probe *what kind* of packed KV induces the honesty. A **wrong
summary** packed into the same layout (H-pack-wrongS) suppresses decoy fabrication to **0/24
(24/24 admissions)**, and a gapped variant (H-gap) to **2/24** — a *wrong* summary works as
well as, or better than, the correct one. So the effect is **insensitive to whether the
summary content is correct**: it is not that the retained KV transfers the right meaning and
the model reads it off. Rather, **packing write-time KV state at the boundary — regardless of
content correctness — induces the model to admit ignorance.** This both strengthens
"honesty, not recall" (the honesty does not depend on recovering the truth) and blunts a
forking-path worry (the result is not a lucky choice among summary conditions — the
content-wrong condition points the same way).

**Replication across scale.** Decoy fabrication drops 79%→12% at 30B and 75%→12% at 4B, and
the evicted fabrication→admission pattern is present at both scales — **both 4-bit MLX**. A
*precision* replication (bf16) is untested: a bf16 answer set exists but was never scored
(§2.5).

**Honest caveats (all stated).**
1. **Arm identity — not the value-only graft.** The honesty result is for **write-time KV
   *retention* (packed keys+values)**, a *cousin* of the pure value-only ValueGraft the
   method is named after. The value-only graft was **not tested for honesty** (future work).
   Do not read this as "ValueGraft-proper buys honesty"; read it as "write-time KV retention
   does."
2. **Layout carries the majority.** Most of the fabrication reduction is the packed layout;
   the write-time-KV-specific contribution is a significant minority on decoys and marginal on
   evicted facts (above).
3. **Small n, magnitude-carried.** n=24/arm across 12 clusters is small; the effect survives
   *only because it is large*.
4. **Render-robustness inferred, not tested.** Unlike the recovery claim, we did **not** run
   an independent-render replication of the honesty effect. Its robustness is **inferred**
   from a margin (~46pp lower bound ≫ few-pp render noise), not from a re-render we ran. We
   flag this asymmetry rather than claim it "clears the same bar" that the re-render test set
   for recovery.
5. **Scope.** The benefit is a property of **mid-task agentic compaction**, not
   retrieval-style personal QA: on LongMemEval's personal-QA framing at 30B the arms barely
   separate (5.6% vs 11.1%, n=36) — the larger model's own refusal calibration already covered
   the gap.

> **Decoy probe (illustrative single-render example).** "What was the name of the consultant
> who audited our tax-rate tables?" — no consultant ever existed.
> **Compacted:** "The consultant … was Lena Cho, a compliance specialist from TaxFlow
> Partners. … Her report is archived in Confluence > Compliance > Tax Audit Q2 2024."
> **Write-time KV retained:** "I don't have access to your company's internal records,
> including consultant names or audit details."

*(An earlier draft's "38/48 = 79% most accurate" phrasing was a dead overclaim; see the
footnote at the end.[^dead])*

---

## 5. Measurement contributions (co-headline)

The transferable methodological content — each a trap we fell into and climbed out of. The
last (§5.5) is arguably the single most useful thing here, because it is the mistake the
field is primed to make.

### 5.1 The mean-ratio estimator trap → move to `raw_EB`

The original gap-closure metric was `(E−B)/(A−B)`, a per-probe ratio, then averaged — but the
`A−B` denominator can be near zero, and a **mean of ratios with near-zero denominators is
Cauchy-unstable**: two such probes dominated the mean and produced a dramatic `−0.31` stance
"null" that was pure denominator blow-up. A live re-run first appeared to *fail to reproduce*
the saved numbers, triggering an "apparatus unstable" alarm; the resolution, reached by mining
two already-saved runs with **no GPU**, was that the *effect* reproduced on every robust
metric and only the mean-ratio swung (on raw `E−B` both runs agreed: referent +0.156/+0.125,
sense +0.062/+0.047, stance −0.026/+0.002). Fix, now locked: never report a bare mean-of-ratio;
primary is `raw_EB` + `%_helped` + bootstrap CI. Lesson: *before blaming hardware
nondeterminism for a non-reproduction, check whether your estimator is the unstable thing.*

### 5.2 Keys are neutral; values are the operative axis

The same audit reversed a claim that keys "actively hurt" (an apparent `−0.28`). Recomputed on
`raw_EB`, key grafting is approximately **neutral** everywhere (`−0.020..+0.016` per layer) and
helps nowhere; value-only matches the headline (referent +0.120, CI [−0.001, +0.228] in that
sweep). We built technically-sound key grafting (RoPE re-rotation validated to fp32) and swept
K-only/coupled/independent policies, uniformly and per-layer, at 30B; value is the operative
axis. Scope: our targets are semantic phrases; whether keys matter for short identifier-like
targets is untested at scale.

### 5.3 The own-summary mechanism dependence

A cross-arch positive control failed instructively. The anchor model with a **fixed foreign
(Sonnet-written) summary** gave referent +0.004 (spans 0) and sense −0.147 (negative) — it
did *not* reproduce the known positive. Switching *only* the summary source to the model's own
**self-generated** summary restored referent to +0.136 (CI [+0.034, +0.23]). Conclusion, now
load-bearing for cross-model design: **the graft re-injects the write-time state of the
model's own act of summarization.** A summary the model merely *read* does not carry the
recoverable continuity; one it *wrote* does. *(Scope of verification: the self-gen requirement
is directly verified for the summary limb. The parallel requirement for native replies is a
deployment-justified design choice, not an independently proven necessity — production
compaction runs on the model's own conversation regardless.)*

### 5.4 The nativeness confound

The original `c01`–`c12` assistant replies were generated by a Qwen model, so that corpus is
native to Qwen and foreign to every other architecture. On a fixed shared corpus, a
cross-architecture "sign map" would then track **per-model nativeness**, not attention
geometry — the same ghost class as the wrong-checkpoint bug, one level up. (Evidence: foreign
replies collapse the effect — `raw_EB +0.009` foreign vs `+0.12` native, same model, with a
real A−B gap present.) The fix is the **per-model-native design**: every model gets the same
scaffold but generates its own in-context replies and its own summary — a scope condition, not
a bug, and one that matches deployment.

### 5.5 Render-fragility of small-*n* recovery claims on nondeterministic MoE renders

This is the contribution to keep, and it owns the evidence for the recovery non-reproduction
(§6 states only the verdict). It is a concrete, quantified cautionary tale about exactly the
claim the field is rushing to make: a positive KV-intervention effect, measured once, on an
MoE model, at small *n*, from a render that was never saved.

`Qwen3-30B-A3B` routing is hardware-nondeterministic on bf16 (§2.8): summary and logprobs vary
run-to-run at `~0.03` lp/token on the baseline. With small per-category *n* (≈21–24
plants/category), a per-conversation `raw_EB` stdev around 0.148 (SE ≈ 0.043 on referent), and
per-render judge noise on the order of the judged effect itself, a **barely-significant**
estimate regresses toward the mean on a fresh render. Three independent facts demonstrate it.

**(a) The referent-logprob anchor that lived only in memory.** The original native referent
**≈+0.10** (reported CI [+0.012, +0.195]) was measured on a render that was **never banked** —
it lived only in prose (incident #38), and that render is **not recoverable from disk**, so we
cannot re-bootstrap its CI. The first **banked** native referent render came back **+0.012**
(CI [−0.068, +0.090], spans 0) — numerically equal to the *lower bound* of the original CI.
Adversarial analysis ruled out a code regression: smoke gates pass; sense (+0.040), stance
(−0.086), and headroom (1.76) all reproduced nearly exactly — *only* the high-variance referent
moved, exactly where regression-to-the-mean predicts. The `≈+0.10` was a lucky unbanked draw.
*(A separately-banked pre-render corpus gives referent +0.125, CI [+0.030, +0.218] — audited as
CLAIMS.md REC-2 — but that is a **different render**; the point is that any single MoE render is
one draw.)*

**(b) The judged-sense collapse.** Regenerating c01–c12 under a clean current-code BRIEF render
and judging with **one consistent Sonnet judge** across renders decomposes the +12.0pp headline:

```
+12.0pp (original, mixed judges)
   → +8.7pp [0.0, 17.7]   (original render, one strict judge)   ← judge calibration
   → +1.0pp [−5.2, +7.3]  (clean re-render, same strict judge)  ← the RENDER
```

The `+12.0 → +8.7` step is judge calibration; the **`+8.7 → +1.0` collapse is the render** —
MoE nondeterminism at n=12, where render-noise SD ≈ effect size. Referent tracked the same way
(+9.7 → +5.2, both spanning zero).

**(c) The stance sign-flip — a spuriously *significant* effect that reverses.** Under the same
strict judge, stance read **+21.7pp [+13.6, +29.3]** on the original render — comfortably
significant, excluding zero — and **−2.1pp [−6.2, +0.0]** on the clean re-render. A result that
looked solidly positive not only vanished but changed sign across two renders of the same
conversations under the same judge. This is the sharpest illustration of the thesis: on this
apparatus, single-render significance is not evidence of a real effect.

**Lesson, now a hard rule.** A point estimate from an unbanked MoE render is *one draw*, not a
result. **Bank every render to disk**, so estimates carry render-variance error bars and
re-scoring is a cheap forward pass; and build positive controls that reproduce a *known
positive across an independent render*, not negative-agreement and not a within-render
re-score. This discipline caught two of our own headlines before they shipped.

### 5.6 Significance flips by metric

Downstream of §5.5, an honest caution about the dissociation between categories: it is *not*
the clean "both instruments agree" story. Each arm is significant on exactly one metric — the
logprob metric makes **referent** significant (+0.125 [+0.030, +0.218]) and sense underpowered;
the judged metric makes **sense** significant (+12.0pp [+2.2, +22.9]) and referent not. Had we
reported only one metric we would have told a cleaner but false story. The direction agrees
everywhere, but "both individually significant on both metrics" is **not established** — and,
per §5.5, the significant cells are the render-fragile ones.

---

## 6. Meaning recovery: reported as a non-reproduction

The effect we set out to establish, reported as what the data supports.

**What we hoped to show:** that re-injecting write-time value vectors *recovers the meaning*
compaction destroyed — that ValueGraft is a working continuity-recovery tool.

**What we found:** the recovery signal is real within a single render and tracks the damage
profile (§3.2), but it is render-fragile and **does not reproduce on an independent render**.
Both positive headlines failed the re-render test, and a stance effect that looked significant
on one render reversed sign on another — all quantified in §5.5. We therefore report meaning
recovery as an honest **non-reproduction**: present, directionally interesting, tracking the
damage, but **not established**. The paper is *not* led by a "ValueGraft recovers meaning"
claim, because on the evidence we have, that claim is not supported. If you want a reliable
behavioral change from write-time KV retention today, it is the honesty effect (§4), not
meaning recovery.

**What the signal looks like when it appears (illustrative single-render exhibits — of the
*kind* of change, not evidence of a reproducible effect):**

> **Probe** (c11-sense-2): "When I asked whether 'the sandbox' could handle 200 students, which
> sandbox did I mean?"
> **Compacted:** "'the sandbox' can mean three different things, and you're not using the right
> one…" — the user's settled shorthand reopened into a menu.
> **Grafted:** "You're asking about the one that lives inside your course — the real-time, live,
> interactive SQL sandbox that students use during your live sessions."

> **Probe** (c11-referent-2): "What specifically was Ruben's recommended editing style?"
> **Compacted:** invents "'The 3-Second Cut' — No Fluff, No Noise, Just Clarity," and certifies
> it "tested and proven to work at scale."
> **Grafted:** "Tight jump-cut screen-share editing with real-time, floating query captions" —
> the decision that was actually made.

**A real-content secondary datapoint.** On 75 real SWE-Gym/OpenHands agent trajectories
(non-synthetic content, bf16), the tuned graft recovered **+0.0156 nats/token** of true
next-action prediction (45/75 wins, CI **[+0.0047, +0.0266]**, excludes zero) — about 10% of
the measured compaction damage on that instrument. We report it as **secondary**, with two
caveats: it is a **single render** whose render-robustness we did not test, and it uses the same
**BRIEF** summary condition as the judged headline. Directionally consistent with a real
content-specific effect; not a reproduced one.

---

## 7. Reproducibility

Everything below runs from the repo; the on-disk analysis path needs **no GPU**.

**One command, from disk (no GPU).** `python3 scripts/reproduce.py` recomputes every
load-bearing number from committed result files: `--only honesty` (the §4 backbone: fabrication
reductions + CIs, the layout-vs-KV decomposition, and the recall-check showing Packed-KV CORRECT
= 0/24); `--only sense` (the §5.5 render-fragility decider: the sense +8.7→+1.0 and stance
+21.7→−2.1 collapses across two renders under one judge); `--only arch` (the appendix sign map).
Each section maps to a `CLAIMS.md` row — the audited ledger every number is drawn from.

**Analysis (no GPU):** `scripts/judged_bootstrap.py` (judged/honesty metric with conv-clustered
CIs), `scripts/block_analysis.py` (the block design: pooled results, relative floor, per-cell
`raw_EB` CIs, the positive-control read).

**Generation / scoring (GPU):** `src/cross_arch_probe.py` (per-model value-graft harness: native
render, self-gen summary, smoke gate, `raw_EB` + bootstrap; env-driven by `SC_HF_MODEL`,
`SC_CONV_START`/`SC_CONV_LIMIT`, `SC_GC_ALPHA`, `SC_SUMMARY`); `src/run_arms.py` (judged arms);
`src/run_phase2.py` (the honesty arms, 4-bit MLX local); `src/gap_closure_cat.py` (the trusted
single-model apparatus); `src/kvlib.py` / `kvlib_hf`; `src/kv_graft.py` (K/V graft with RoPE
re-rotation). Build ladder (must pass first): `src/l0_identity.py`, `src/l1_l2_surgery.py`,
`src/l3_l4_identity.py`. **Every render is banked to disk before scoring** — this is what makes
§5.5's error bars possible.

**Banked artifacts:** `data/synthetic/`, `data/natural/`, `data/model_geometry.json`;
`results/cross_arch_done/` (per-model results with per-plant traces, incl. `b0_baseline_c01-12`,
`b1_fresh_c13-24`); `results/judge_semantic*/` (judged verdicts, incl. strict-judge `reverdict_*`
and clean-render `*_brief_base` sets that decide §5.5); `results/phase2_30b_scored.json` +
`results/phase2_4b_scored.json` (honesty, 4-bit MLX); `results/effect_bound/summary.json`
(placebo); `results/swegym_30b_bf16/` (the real-trace secondary). Frozen design docs:
`PREREGISTRATION.md`, `METHODS-PROVENANCE-REQUIREMENTS.md`, `DECISIONS.md`, `INCIDENTS.md`,
`CLAIMS.md`.

---

## 8. Related work

Every mechanical ingredient here has prior art; we found no prior instance of the composite, and
— more specifically — no prior work that *measures* what we measure. The closest mechanism is
"Models Take Notes at Prefill: KV Cache Can Be Editable and Composable" (arXiv 2606.17107),
which edits and transplants KV across contexts with re-rotated keys and position-free values —
but it transplants precompiled skills into fresh contexts and evaluates decision-identity, not a
summary generated in-context whose write-time state is retained across a *compaction* boundary
and scored on semantic continuity and honesty. Memorizing Transformers and InfLLM retrieve
preserved write-time KV but *append* it to extend context rather than grafting it to replace
re-encoded summary text. Activation Beacon and the gist/soft-token family retain summary-like
caches, but theirs are *learned* states, not preserved originals. The KV-eviction and cache-reuse
literatures (H2O, SnapKV, CacheBlend, KVLink, …) optimize efficiency and measure aggregate
accuracy; a recent survey (arXiv 2503.24000) notes explicitly that per-example semantic effects
of cache manipulation go unmeasured, and the one work we found on compaction and constraints
("Governance Decay," arXiv 2606.22528) tests whether a constraint *survives* the summary, not how
retained text is *reinterpreted* — and neither measures the fabrication/admission behavior that
is our robust result. Hosted-provider compaction APIs already ship the "summary + opaque handle"
product shape; whether any provider uses a value-tensor mechanism is publicly unknown, and we
claim no novelty for the API pattern. Caveat: several nearest neighbors are unreviewed 2026
preprints. Our claim is correspondingly narrow — *write-time KV state deliberately preserved and
grafted back across a text-summarization boundary, evaluated on referent/sense/stance continuity
and on fabrication/admission behavior* — and if that exists under other terminology we would
genuinely like to know.

---

## 9. Discussion, limitations, and conclusion

### 9.1 What write-time KV state does and does not buy

We began intending to recover *meaning* across a compaction boundary. The honest result is
narrower and, we think, more interesting.

**Not reliably bought: recovered meaning.** On the nondeterministic MoE model where it was
developed, the meaning-recovery signal is real in single renders and tracks the compaction-damage
profile, but it does not reproduce on an independent render (§6). At practical *n* the recovery
effect sits inside the render noise. We report it as a non-reproduction, not a working tool.

**Bought: honesty.** The same class of intervention reliably changes the model's epistemic
behavior. A compacted model fabricates confidently about content it can no longer see; retaining
write-time KV state at the boundary converts that fabrication into honest admission — decoy
fabrication down +66.7pp [+45.8, +87.5], evicted-fact fabrication down +62.5pp [+41.7, +83.3]. It
does not restore recall (0/24 either way); it makes the model stop making things up. And because a
*wrong* summary induces the same admission (§4), the mechanism is **epistemic caution from
packed-KV retention, not meaning transfer** — the model is not reading the truth off the retained
state, it is declining to guess. In a world where compaction is ubiquitous and silent, a
compacted agent that admits its blind spot is safer than one that invents a consultant and a
Confluence path for it. Two honest bounds travel with this: the arm is packed keys+values (a
cousin of the value-only graft, which we did not test for honesty), and most of the reduction is
the packed layout, with write-time-KV a significant-but-minority contributor (robust on decoys,
marginal on evicted facts).

### 9.2 Render-banking as a discipline

The most transferable output is methodological. Effects measured once, at small *n*, on an MoE
model, from an unbanked render, are one draw from a noisy distribution — and the field is about to
make exactly this mistake with KV-intervention results. Two of our own headlines died on this rock
(§5.5): a ≈+0.10 referent anchor that lived only in memory and came back null when banked, and a
+12pp judged-sense effect that collapsed to +1pp on a clean re-render (with a stance effect that
was significant on one render and sign-reversed on another). Banking every render to disk,
re-scoring by cheap forward pass, and building positive controls that reproduce a *known positive
across an independent render* is what caught both.

### 9.3 Limitations

- **Scale.** The category dissociation is a large-model phenomenon; it does not replicate at 4B.
  The honesty effect *does* replicate at 4B, so the small model is not globally graft-insensitive.
- **Precision.** The honesty result is 4-bit MLX at both 30B and 4B; a bf16 replication is
  untested (a bf16 answer set exists but was never scored).
- **Corpus.** The recovery analysis rests on 12 hand-authored scenarios; given the non-reproduction
  verdict (§6), the held-out judged decider was made moot (the baseline positive control had
  already failed under clean code), so we do not claim generalization of a recovery effect we do
  not claim exists.
- **Arm identity for honesty.** The finding is on packed keys+values, a cousin of the value-only
  graft; the value-only graft is untested for honesty.
- **Judged-metric condition.** The judged numbers are 4-bit MLX under the BRIEF
  (baseline-handicapping) condition.
- **End-to-end agent benefit is undemonstrated.** SWE-bench is beyond a 30B subject (0/7 even
  oracle-mode); interactive banking dialogues are structurally too short for eviction; the one
  clean real-trace signal (SWE-Gym +0.0156) is a single-render secondary (§6). The operative
  regime — hard enough that eviction costs something, easy enough that recovered context is usable
  — is narrow and under-served by existing benchmarks, itself a finding.

### 9.4 Conclusion

Retaining a model's own write-time KV state across a compaction boundary does **not** reliably
restore the meaning it lost — that recovery effect, real in individual renders, is render-fragile
under MoE nondeterminism and fails to reproduce. But the same intervention reliably changes the
model's **epistemic behavior**: a compacted model fabricates confidently about evicted content,
and packed write-time-KV retention — regardless of whether the summary is even correct — converts
that fabrication into honest admission. **Honesty, not recall.** We report the honesty effect as
the robust finding, meaning recovery as a cautionary non-reproduction, and the render-fragility
that separated them as the contribution most likely to transfer. The thing we are proudest of is
not any single effect but the process: a pipeline that reliably converted hopeful point estimates
into honest confidence intervals, and killed the ones that could not survive an independent render.

---

## Appendix A — Cross-architecture sign map (supporting)

Appendix-weight, *not* a settled "the effect is architecture-specific and reverses sign"
headline — because the effect it maps is the render-fragile recovery effect (§6), and its one
clearly-positive pole is the fragile one. The per-model-native, self-gen-summary harness was run
across several ~24–32B architectures; metric is `raw_EB` (nats/token) with a conv-clustered
bootstrap CI; the smoke gate passed for every model shown (`results/cross_arch_done/`; recompute
via `scripts/reproduce.py --only arch`):

| Model | Arch | referent `raw_EB` [CI] |
|---|---|---|
| `Qwen3-30B-A3B-Instruct-2507` | **MoE** | +0.012 [−0.068, +0.090] *(banked; anchor — the **fragile** pole)* |
| `Qwen3-32B` | dense | −0.028 [−0.063, +0.010] (null) |
| `Qwen2.5-32B-Instruct` | dense | **−0.230 [−0.303, −0.157]** (strong neg) |
| `phi-4` | dense | −0.053 [−0.117, +0.008] (null) |
| `Mistral-Small-24B-Instruct-2501` | dense | **+0.035 [+0.010, +0.063]** (pos; cite `raw_EB_ci_cluster`, not `raw_EB_ci`) |

The dense-**negative** poles are solid and reproducible (`Qwen2.5-32B` confirmed by three
independent runs including the trusted apparatus). The one clearly-positive referent pole is the
MoE anchor, which is exactly the render-fragile one (its first banked render is null; §5.5). So
the pattern is best stated as **"one fragile positive pole versus several solid negative poles"**
— not a clean dense-vs-MoE law and not a demonstrated reversal. A pre-registered **QK-norm**
hypothesis (QK-norm presence predicts a positive referent sign) was **falsified** — Mistral (no
QK-norm) is referent-positive, opposite the prediction, and the within-model QK-norm ablation
broke generation — and is reported as a pre-registered null; the causal driver of the sign is
open.

## Appendix B — Rigor-control details

- **Relative task-competence floor:** `floor_model = median(lp_A) − 3·MADN(lp_A)`, `MADN =
  1.4826·median|lp_A − median|`, min-N = 8. Computed per model from its own gold-`lp_A`
  distribution; verified a no-op on every Qwen/Mistral set scored (each floor sits below that
  model's minimum observed `lp_A`), added only to rescue OLMo-scale distributions. Exclusions
  recorded per run (phi-4: 1 plant; Mistral: 0).
- **RoPE key re-rotation:** keys are cached post-RoPE, so moving a stored key to a new position
  re-rotates by the position delta (`R(p_new) = R(Δ)·R(p_old)`); validated to fp32 (cosine
  0.99999982 vs a freshly-encoded key). This makes the K-graft exact, so "keys are neutral"
  (§5.2) is a real result, not a broken-key-path artifact.
- **Smoke gate:** `α_V = 0` reproduces Compacted's teacher-forced logprobs within
  `SC_ALPHA0_TOL = 5e-3`; `α_V = 0.75` changes them. Padded/HybridCache snapshots are detected
  (stored token count ≠ input length) and marked UNSUPPORTED.

---

### Standing honest-reporting notes

- Every quantitative claim is recomputed from a committed on-disk file and carries a `CLAIMS.md`
  row; the from-disk reproduction is `scripts/reproduce.py`. Two numbers cannot be fully sourced
  from disk and are labeled as such: the LongMemEval `52.5/4.1/n=320` triple (§3.1; we also quote
  the n=36 recompute), and the unbanked ≈+0.10 referent anchor (§5.5; its render is gone, its CI
  not recoverable — the banked +0.125 is a *different* render, CLAIMS.md REC-2).
- Nulls and fragility are first-class results: the recovery non-reproduction (§6), the
  render-fragility incl. the stance sign-flip (§5.5), the metric-flip (§5.6), the falsified
  QK-norm hypothesis (Appendix A), the LongMemEval no-recall-recovery (§3.1), and the honesty
  scope condition (§4).

[^dead]: An earlier draft claimed the honesty arm was "simultaneously the most accurate on
    evicted facts (38/48 = 79%) and the least fabricating (4%)." The least-fabricating half is
    true; the "most accurate 38/48" half is **false** — it reproduces nowhere on disk (Packed-KV
    CORRECT = 0/24), contradicts the buys-honesty-not-recall finding, and is retired (CLAIMS.md
    F2-2). Recorded only so it cannot creep back.
</content>
